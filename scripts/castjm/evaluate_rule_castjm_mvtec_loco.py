#!/usr/bin/env python3
"""Evaluate CASTJM, optionally preceded by the five binary rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from pro_innovation.adapters import (
    ClipAttributeDetector,
    DinoV2PatchFeatureExtractor,
    LocalClipRuntime,
    Sam3Segmenter,
)
from pro_innovation.attributes import RoutedAttributeDetector
from pro_innovation.evaluation import (
    MVTEC_LOCO,
    SUPPORTED_DATASETS,
    _collect_evaluation_samples,
    _split_specs_for_dataset,
    binary_auroc,
)
from pro_innovation.gpu_power import NvidiaGpuPowerMonitor
from pro_innovation.memory_modules.castjm import (
    CASTJMGlobalPatchBank,
    CASTJMGlobalPatchScorer,
    CASTJMMemoryBank,
    CASTJMScorer,
)
from pro_innovation.mask_processing import MaskPostprocessor
from pro_innovation.pipeline import (
    LogicAnomalyPipeline,
    RULE_MEMORY_SCHEDULE_BINARY_PARALLEL,
    RULE_MEMORY_SCHEDULE_BINARY_SELECTIVE,
    RULE_MEMORY_SCHEDULES,
)
from pro_innovation.sam3_calibration import load_sam3_threshold_policy
from pro_innovation.templates import collect_categories, parse_template


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CASTJM/FiveRules evaluation")
    parser.add_argument("--class-name", required=True)
    parser.add_argument(
        "--dataset-root",
        required=True,
        help=(
            "VisA dataset root containing split_csv/1cls.csv, or the class test "
            "tree for MVTec LOCO/MVTec AD"
        ),
    )
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=MVTEC_LOCO)
    parser.add_argument("--generic")
    parser.add_argument("--standard")
    parser.add_argument("--attribute-vocabulary")
    parser.add_argument("--threshold-policy")
    parser.add_argument("--global-bank", required=True)
    parser.add_argument(
        "--joint-bank",
        help=(
            "required for threshold-backed CASTJM; ignored by no-threshold "
            "CASTJM, which returns the Global Patch score directly"
        ),
    )
    parser.add_argument(
        "--memory-module",
        choices=("castjm",),
        default="castjm",
        help="compatibility option; CASTJM is the only supported memory module",
    )
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam3-repo", default="/home/lxq/models/SAM3")
    parser.add_argument("--sam3-checkpoint", default="/home/lxq/weights/sam3/sam3.pt")
    parser.add_argument(
        "--dino-checkpoint",
        default="/home/lxq/weights/DINOV2/dinov2_vitl14_reg4_pretrain.pth",
    )
    parser.add_argument("--clip-repo", default="/home/lxq/models/CLIP")
    parser.add_argument(
        "--clip-checkpoint", default="/home/lxq/weights/CLIP/ViT-L-14.pt"
    )
    parser.add_argument("--clip-threshold", type=float, default=0.35)
    parser.add_argument("--sam3-threshold", type=float, default=0.40)
    parser.add_argument("--sam3-cache-root", default="cache/sam3")
    parser.add_argument(
        "--sam3-cache-mode",
        choices=("off", "read", "read-write", "refresh"),
        default="read-write",
    )
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--extraction-batch-size", type=int, default=4)
    parser.add_argument(
        "--gpu-power-sampling",
        action="store_true",
        help=(
            "sample whole-board NVIDIA GPU power with nvidia-smi and write "
            "gpu_power.csv plus gpu_energy summary fields"
        ),
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--gpu-power-sampling-interval-ms",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--schedule",
        choices=RULE_MEMORY_SCHEDULES,
        default=RULE_MEMORY_SCHEDULE_BINARY_SELECTIVE,
        help=(
            "binary-selective returns 1 on the first failed rule and runs "
            "CASTJM only when all rules pass; binary-parallel runs both "
            "branches and takes the maximum of their scores"
        ),
    )
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def _load_completed(path: Path) -> dict[str, dict]:
    completed: dict[str, dict] = {}
    if not path.is_file():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            completed[str(record["image"])] = record
    return completed


def _subgroup_auroc(records: list[dict], positive_split: str) -> float | None:
    selected = [
        record
        for record in records
        if record.get("score") is not None
        and record.get("split") in ("good", positive_split)
    ]
    labels = [int(record["label"]) for record in selected]
    if set(labels) != {0, 1}:
        return None
    return binary_auroc(labels, [float(record["score"]) for record in selected])


def _rule_failed(record: dict) -> bool:
    return any(
        float(rule.get("violation_score", 0.0) or 0.0) >= 1.0
        for rule in record.get("rules", ())
    )


def _memory_executed(record: dict) -> bool:
    timings = record.get("timings_seconds", {})
    return bool(
        float(timings.get("memory_total", 0.0) or 0.0) > 0.0
        or record.get("castjm_evidence") is not None
        or record.get("global_patch_evidence") is not None
    )


def _memory_execution_summary(records: list[dict]) -> dict[str, float | int]:
    valid = [record for record in records if record.get("score") is not None]
    failed = [record for record in valid if _rule_failed(record)]
    executed = [record for record in valid if _memory_executed(record)]
    executed_after_failure = [
        record for record in failed if _memory_executed(record)
    ]

    def timing_total(selected: list[dict], key: str) -> float:
        return sum(
            float(record.get("timings_seconds", {}).get(key, 0.0) or 0.0)
            for record in selected
        )

    return {
        "total_images": len(valid),
        "rule_failed_images": len(failed),
        "executed_images": len(executed),
        "execution_rate": len(executed) / len(valid) if valid else 0.0,
        "skipped_images": len(valid) - len(executed),
        "executed_after_rule_failure": len(executed_after_failure),
        "skipped_after_rule_failure": len(failed) - len(executed_after_failure),
        "dinov2_seconds": timing_total(valid, "memory_feature_extraction"),
        "castjm_matching_seconds": timing_total(valid, "memory_composition"),
        "total_branch_seconds": timing_total(valid, "memory_total"),
        "dinov2_seconds_after_rule_failure": timing_total(
            executed_after_failure, "memory_feature_extraction"
        ),
        "castjm_matching_seconds_after_rule_failure": timing_total(
            executed_after_failure, "memory_composition"
        ),
        "total_branch_seconds_after_rule_failure": timing_total(
            executed_after_failure, "memory_total"
        ),
    }


def _synchronize_cuda(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _initialize_models(
    *,
    segmenter: Sam3Segmenter | None,
    extractor: DinoV2PatchFeatureExtractor,
    clip_runtime: LocalClipRuntime | None,
    device: str,
) -> tuple[dict[str, float], float]:
    initializers = []
    if isinstance(segmenter, Sam3Segmenter):
        initializers.append(("sam3", segmenter.initialize))
    initializers.append(("dinov2", extractor.initialize))
    if clip_runtime is not None:
        initializers.append(("clip", clip_runtime.load))

    components: dict[str, float] = {}
    total_started = time.perf_counter()
    _synchronize_cuda(device)
    for name, initialize in initializers:
        _synchronize_cuda(device)
        started = time.perf_counter()
        initialize()
        _synchronize_cuda(device)
        components[name] = time.perf_counter() - started
    total = time.perf_counter() - total_started
    return components, total


def main() -> int:
    args = _arguments()
    module_label = "CASTJM"
    dataset = Path(args.dataset_root).expanduser().resolve()
    generic_path = Path(args.generic).expanduser().resolve() if args.generic else None
    standard_path = (
        Path(args.standard).expanduser().resolve() if args.standard else None
    )
    templates_available = bool(
        generic_path is not None
        and generic_path.is_file()
        and standard_path is not None
        and standard_path.is_file()
    )
    if templates_available:
        assert generic_path is not None
        assert standard_path is not None
        generic = parse_template(
            generic_path.read_text(encoding="utf-8"),
            allow_missing_values=True,
        )
        standard = parse_template(
            standard_path.read_text(encoding="utf-8"),
            allow_missing_values=False,
        )
    else:
        generic = ()
        standard = ()
    vocabulary = (
        json.loads(
            Path(args.attribute_vocabulary)
            .expanduser()
            .resolve()
            .read_text(encoding="utf-8")
        )
        if templates_available and args.attribute_vocabulary
        else {}
    )
    threshold_policy_path = (
        Path(args.threshold_policy).expanduser().resolve()
        if args.threshold_policy
        else None
    )
    policy = (
        load_sam3_threshold_policy(
            threshold_policy_path, expected_class_name=args.class_name
        )
        if threshold_policy_path is not None and threshold_policy_path.is_file()
        else None
    )
    whole_image_mode = policy is None
    direct_global_patch_mode = whole_image_mode
    rules_enabled = templates_available and not whole_image_mode
    if direct_global_patch_mode:
        categories: tuple[str, ...] = ()
        missing: tuple[str, ...] = ()
        print(
            "[CASTJM] no threshold policy; returning the Global Patch anomaly "
            "score directly without segmentation or joint-memory processing",
            flush=True,
        )
    else:
        assert policy is not None
        categories = collect_categories(generic) or tuple(policy.selected_thresholds)
        if not categories:
            raise ValueError(
                "SAM3-backed CASTJM requires template or threshold-policy categories"
            )
        missing = tuple(
            category
            for category in categories
            if category not in policy.selected_thresholds
        )
        if missing:
            print(
                "[Rule+CASTJM] warning: threshold policy is missing "
                + ", ".join(missing)
                + f"; using global fallback {args.sam3_threshold:.2f}",
                flush=True,
            )
    if not templates_available:
        print(
            f"[{module_label}] generic/standard templates unavailable; five-rule "
            "inference is disabled",
            flush=True,
        )

    global_bank = CASTJMGlobalPatchBank.load(
        args.global_bank,
        expected_class_name=args.class_name,
        expected_checkpoint_path=args.dino_checkpoint,
    )
    extractor = DinoV2PatchFeatureExtractor(
        checkpoint_path=args.dino_checkpoint, device=args.device
    )
    global_patch_scorer = CASTJMGlobalPatchScorer(
        bank=global_bank,
        extractor=extractor,
        query_batch_size=args.query_batch_size,
        extraction_batch_size=args.extraction_batch_size,
    )
    memory_bank: Any | None = None
    if direct_global_patch_mode:
        memory_scorer: Any = global_patch_scorer
    elif not args.joint_bank:
        raise ValueError(
            "--joint-bank is required unless CASTJM runs without a threshold policy"
        )
    else:
        memory_bank = CASTJMMemoryBank.load(
            args.joint_bank,
            expected_class_name=args.class_name,
            expected_categories=categories,
            expected_checkpoint_path=args.dino_checkpoint,
            expected_global_patch_bank_path=args.global_bank,
        )
        memory_scorer = CASTJMScorer(
            bank=memory_bank,
            global_patch_scorer=global_patch_scorer,
        )
    if direct_global_patch_mode:
        segmenter = None
    else:
        assert policy is not None
        segmenter = Sam3Segmenter(
            repo_path=args.sam3_repo,
            checkpoint_path=args.sam3_checkpoint,
            device=args.device,
            confidence_threshold=args.sam3_threshold,
            category_thresholds={
                category: float(
                    policy.selected_thresholds.get(category, args.sam3_threshold)
                )
                for category in categories
            },
            prompt_template="{category}",
            cache_root=args.sam3_cache_root,
            cache_mode=args.sam3_cache_mode,
        )
    processor = (
        None
        if direct_global_patch_mode
        else MaskPostprocessor(
            dedup_iou_threshold=0.80,
            dedup_containment_threshold=0.85,
            dedup_enabled=True,
        )
    )
    clip_runtime = (
        LocalClipRuntime(
            repo_path=args.clip_repo,
            checkpoint_path=args.clip_checkpoint,
            device=args.device,
        )
        if rules_enabled and vocabulary
        else None
    )
    clip_detector = (
        ClipAttributeDetector(
            repo_path=args.clip_repo,
            checkpoint_path=args.clip_checkpoint,
            vocabulary=vocabulary,
            device=args.device,
            confidence_threshold=args.clip_threshold,
            runtime=clip_runtime,
        )
        if rules_enabled and vocabulary
        else None
    )
    detector = (
        RoutedAttributeDetector(semantic_detector=clip_detector, vocabulary=vocabulary)
        if rules_enabled
        else None
    )
    pipeline = LogicAnomalyPipeline(
        segmenter,
        detector,
        mask_processor=processor,
        patch_scorer=memory_scorer if direct_global_patch_mode else None,
        composition_scorer=None if direct_global_patch_mode else memory_scorer,
        shared_feature_extractor=None if direct_global_patch_mode else memory_scorer,
        rule_inference_enabled=rules_enabled,
        rule_memory_schedule=args.schedule,
        post_fusion_tau=1.0,
        mask_categories=categories,
    )

    split_specs = _split_specs_for_dataset(args.dataset, dataset)
    splits = tuple(split_specs)
    samples = _collect_evaluation_samples(
        dataset_name=args.dataset,
        dataset_root=dataset,
        class_name=args.class_name,
        selected_splits=splits,
        split_specs=split_specs,
    )
    run_directory = Path(args.result_root).expanduser().resolve() / args.class_name
    run_directory.mkdir(parents=True, exist_ok=True)
    power_monitor = (
        NvidiaGpuPowerMonitor(
            gpu_index=args.gpu_index,
            interval_ms=args.gpu_power_sampling_interval_ms,
        )
        if args.gpu_power_sampling
        else None
    )
    if power_monitor is not None and not power_monitor.start():
        print(
            f"[GPU power] warning: {power_monitor.error}; power fields unavailable",
            flush=True,
        )
    startup_power_started = time.perf_counter()
    model_startup_components, model_startup_seconds = _initialize_models(
        segmenter=segmenter,
        extractor=extractor,
        clip_runtime=clip_runtime,
        device=args.device,
    )
    startup_power_ended = time.perf_counter()
    startup_payload = {
        "total_seconds": model_startup_seconds,
        "components_seconds": model_startup_components,
    }
    (run_directory / "model_startup.json").write_text(
        json.dumps(startup_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    component_text = " ".join(
        f"{name}={seconds:.3f}s"
        for name, seconds in model_startup_components.items()
    )
    print(
        f"[model startup] {component_text} total={model_startup_seconds:.3f}s",
        flush=True,
    )
    progress_path = run_directory / "all.jsonl"
    summary_path = run_directory / "all_summary.json"
    completed = {} if args.restart else _load_completed(progress_path)
    mode = "w" if args.restart else "a"
    inference_power_started = time.perf_counter()
    try:
        with progress_path.open(mode, encoding="utf-8") as progress:
            for index, (image, label, split) in enumerate(samples, start=1):
                key = str(image)
                if key in completed:
                    print(f"[{index}/{len(samples)}] resume {split}/{image.name}")
                    continue
                started = time.perf_counter()
                try:
                    report = pipeline.run(image, generic, standard)
                    payload = report.to_dict()
                    evidence_key = (
                        "global_patch_evidence"
                        if direct_global_patch_mode
                        else "castjm_evidence"
                    )
                    evidence_payload = (
                        payload["patch_evidence"]
                        if direct_global_patch_mode
                        else payload["composition_evidence"]
                    )
                    record = {
                        "image": key,
                        "split": split,
                        "label": int(label),
                        "score": report.image_anomaly_score,
                        "execution_mode": report.execution_mode,
                        evidence_key: evidence_payload,
                        "rules": payload["results"],
                        "mask_counts": dict(report.mask_counts),
                        "timings_seconds": payload["timings_seconds"],
                        "elapsed_seconds": time.perf_counter() - started,
                        "error": None,
                    }
                except Exception as exc:
                    record = {
                        "image": key,
                        "split": split,
                        "label": int(label),
                        "score": None,
                        "elapsed_seconds": time.perf_counter() - started,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                progress.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress.flush()
                completed[key] = record
                print(
                    f"[{index}/{len(samples)}] {split}/{image.name} "
                    f"mode={record.get('execution_mode')} score={record['score']} "
                    f"seconds={record['elapsed_seconds']:.2f}",
                    flush=True,
                )
    finally:
        inference_power_ended = time.perf_counter()
        if power_monitor is not None:
            power_monitor.stop()

    power_phases = {
        "model_startup": (startup_power_started, startup_power_ended),
        "inference": (inference_power_started, inference_power_ended),
    }
    if power_monitor is None:
        gpu_energy: dict[str, object] = {
            "available": False,
            "gpu_index": args.gpu_index,
            "sampling_interval_ms": args.gpu_power_sampling_interval_ms,
            "scope": "whole_gpu_board_power",
            "includes_other_gpu_processes": True,
            "error": "GPU power sampling disabled",
            "model_startup": None,
            "inference": None,
            "total": None,
        }
    else:
        gpu_energy = power_monitor.payload(power_phases)
        power_monitor.write_csv(
            run_directory / "gpu_power.csv",
            phases=power_phases,
        )

    records = [completed[str(image)] for image, _, _ in samples]
    valid = [record for record in records if record.get("score") is not None]
    overall = (
        binary_auroc(
            [int(record["label"]) for record in valid],
            [float(record["score"]) for record in valid],
        )
        if {int(record["label"]) for record in valid} == {0, 1}
        else None
    )
    logical = (
        _subgroup_auroc(records, "logical_anomalies")
        if "logical_anomalies" in splits
        else None
    )
    structural = (
        _subgroup_auroc(records, "structural_anomalies")
        if "structural_anomalies" in splits
        else None
    )
    split_aurocs = {
        split: _subgroup_auroc(records, split)
        for split, (label, _) in split_specs.items()
        if label == 1
    }
    elapsed = sum(float(record["elapsed_seconds"]) for record in records)
    timing_totals = {
        key: sum(
            float(record.get("timings_seconds", {}).get(key, 0.0)) for record in valid
        )
        for key in (
            "rule_total",
            "memory_total",
            "branch_wall_total",
            "branch_overlap",
        )
    }
    memory_execution = _memory_execution_summary(records)
    binary_parallel = (
        rules_enabled and args.schedule == RULE_MEMORY_SCHEDULE_BINARY_PARALLEL
    )
    summary = {
        "module": (
            "Global Patch"
            if direct_global_patch_mode
            else module_label
            if not rules_enabled
            else (
                "FiveRules-Binary+CASTJM-Parallel"
                if binary_parallel
                else "FiveRules-Binary+CASTJM"
            )
        ),
        "class_name": args.class_name,
        "dataset": args.dataset,
        "splits": list(splits),
        "rule_inference_enabled": rules_enabled,
        "rule_score_mode": ("binary" if rules_enabled else None),
        "schedule": (
            (
                "global_patch_only"
                if direct_global_patch_mode
                else "castjm_only"
            )
            if not rules_enabled
            else (
                "binary_parallel_five_rules_and_castjm"
                if binary_parallel
                else "binary_selective_five_rules_then_castjm"
            )
        ),
        "fusion": (
            (
                "direct_global_patch_score"
                if direct_global_patch_mode
                else "one_castjm_score"
            )
            if not rules_enabled
            else (
                "max_binary_rule_and_castjm_scores"
                if binary_parallel
                else "binary_rule_fail_early_exit_else_one_castjm_score"
            )
        ),
        "model_startup_seconds": model_startup_seconds,
        "model_startup_components_seconds": model_startup_components,
        "gpu_energy": gpu_energy,
        "castjm_execution": (
            None if direct_global_patch_mode else memory_execution
        ),
        "global_patch_execution": (
            memory_execution if direct_global_patch_mode else None
        ),
        "global_patch_bank": str(Path(args.global_bank).expanduser().resolve()),
        "joint_bank": (
            None
            if direct_global_patch_mode
            else str(Path(args.joint_bank).expanduser().resolve())
        ),
        "memory_module": "global_patch" if direct_global_patch_mode else args.memory_module,
        "semantic_category_embedding": not direct_global_patch_mode,
        "texture_statistics": (
            []
            if direct_global_patch_mode
            else ["mean", "top_fraction_mean", "max"]
        ),
        "top_fraction": None if memory_bank is None else memory_bank.top_fraction,
        "top_fractions": (
            []
            if direct_global_patch_mode
            else [memory_bank.top_fraction]
        ),
        "include_max": (
            None
            if direct_global_patch_mode
            else True
        ),
        "segmentation_mode": (
            "not_used"
            if direct_global_patch_mode
            else "whole_image_single_object"
            if whole_image_mode
            else "sam3_category_masks"
        ),
        "generic_template": str(generic_path) if templates_available else None,
        "standard_template": str(standard_path) if templates_available else None,
        "threshold_policy": (
            str(threshold_policy_path) if policy is not None else None
        ),
        "missing_policy_categories_using_global_fallback": list(missing),
        "sample_count": len(records),
        "valid_count": len(valid),
        "failure_count": len(records) - len(valid),
        "elapsed_seconds": elapsed,
        "seconds_per_image": elapsed / len(records) if records else None,
        "rule_seconds": timing_totals["rule_total"] if rules_enabled else 0.0,
        "memory_seconds": timing_totals["memory_total"],
        "castjm_seconds": (
            timing_totals["memory_total"]
            if not direct_global_patch_mode
            else None
        ),
        "global_patch_seconds": (
            timing_totals["memory_total"] if direct_global_patch_mode else None
        ),
        "branch_wall_seconds": timing_totals["branch_wall_total"],
        "branch_overlap_seconds": timing_totals["branch_overlap"],
        "image_auroc": overall,
        "logical_image_auroc": logical,
        "structural_image_auroc": structural,
        "split_image_aurocs": split_aurocs,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
