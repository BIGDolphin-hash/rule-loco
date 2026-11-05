#!/usr/bin/env python3
"""Evaluate selective five-rule inference followed by isolated GPSTM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from pro_innovation.adapters import (
    ClipAttributeDetector,
    DinoV2PatchFeatureExtractor,
    LocalClipRuntime,
    Sam3Segmenter,
)
from pro_innovation.attributes import RoutedAttributeDetector
from pro_innovation.evaluation import (
    MVTEC_LOCO,
    _collect_evaluation_samples,
    _split_specs_for_dataset,
    binary_auroc,
)
from pro_innovation.memory_modules.gpstm import (
    GPSTMGlobalPatchBank,
    GPSTMGlobalPatchScorer,
    GlobalPatchSemanticTextureMemoryBank,
    GlobalPatchSemanticTextureScorer,
)
from pro_innovation.mask_processing import MaskPostprocessor
from pro_innovation.pipeline import LogicAnomalyPipeline
from pro_innovation.sam3_calibration import load_sam3_threshold_policy
from pro_innovation.templates import collect_categories, parse_template


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selective five rules followed by GPSTM"
    )
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--generic", required=True)
    parser.add_argument("--standard", required=True)
    parser.add_argument("--attribute-vocabulary")
    parser.add_argument("--threshold-policy", required=True)
    parser.add_argument("--global-bank", required=True)
    parser.add_argument("--gpstm-bank", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam3-repo", default="models/SAM3")
    parser.add_argument("--sam3-checkpoint", default="weights/sam3/sam3.pt")
    parser.add_argument(
        "--dino-checkpoint",
        default="weights/DINOV2/dinov2_vitl14_reg4_pretrain.pth",
    )
    parser.add_argument("--clip-repo", default="models/CLIP")
    parser.add_argument(
        "--clip-checkpoint", default="weights/CLIP/ViT-L-14.pt"
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


def main() -> int:
    args = _arguments()
    dataset = Path(args.dataset_root).expanduser().resolve()
    generic = parse_template(
        Path(args.generic).expanduser().resolve().read_text(encoding="utf-8"),
        allow_missing_values=True,
    )
    standard = parse_template(
        Path(args.standard).expanduser().resolve().read_text(encoding="utf-8"),
        allow_missing_values=False,
    )
    categories = collect_categories(generic)
    vocabulary = (
        json.loads(
            Path(args.attribute_vocabulary)
            .expanduser()
            .resolve()
            .read_text(encoding="utf-8")
        )
        if args.attribute_vocabulary
        else {}
    )
    policy = load_sam3_threshold_policy(
        args.threshold_policy, expected_class_name=args.class_name
    )
    missing = tuple(
        category
        for category in categories
        if category not in policy.selected_thresholds
    )
    if missing:
        print(
            "[FiveRules+GPSTM] warning: threshold policy is missing "
            + ", ".join(missing)
            + f"; using global fallback {args.sam3_threshold:.2f}",
            flush=True,
        )

    global_bank = GPSTMGlobalPatchBank.load(
        args.global_bank,
        expected_class_name=args.class_name,
        expected_checkpoint_path=args.dino_checkpoint,
    )
    gpstm_bank = GlobalPatchSemanticTextureMemoryBank.load(
        args.gpstm_bank,
        expected_class_name=args.class_name,
        expected_categories=categories,
        expected_checkpoint_path=args.dino_checkpoint,
        expected_global_patch_bank_path=args.global_bank,
    )
    extractor = DinoV2PatchFeatureExtractor(
        checkpoint_path=args.dino_checkpoint, device=args.device
    )
    global_patch_scorer = GPSTMGlobalPatchScorer(
        bank=global_bank,
        extractor=extractor,
        query_batch_size=args.query_batch_size,
        extraction_batch_size=args.extraction_batch_size,
    )
    gpstm_scorer = GlobalPatchSemanticTextureScorer(
        bank=gpstm_bank, global_patch_scorer=global_patch_scorer
    )
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
    processor = MaskPostprocessor(
        dedup_iou_threshold=0.80,
        dedup_containment_threshold=0.85,
        dedup_enabled=True,
    )
    clip_runtime = (
        LocalClipRuntime(
            repo_path=args.clip_repo,
            checkpoint_path=args.clip_checkpoint,
            device=args.device,
        )
        if vocabulary
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
        if vocabulary
        else None
    )
    detector = RoutedAttributeDetector(
        semantic_detector=clip_detector, vocabulary=vocabulary
    )
    pipeline = LogicAnomalyPipeline(
        segmenter,
        detector,
        mask_processor=processor,
        composition_scorer=gpstm_scorer,
        shared_feature_extractor=gpstm_scorer,
        rule_inference_enabled=True,
        rule_memory_schedule="binary-selective",
        post_fusion_tau=1.0,
    )

    splits = ("good", "logical_anomalies", "structural_anomalies")
    samples = _collect_evaluation_samples(
        dataset_name=MVTEC_LOCO,
        dataset_root=dataset,
        class_name=args.class_name,
        selected_splits=splits,
        split_specs=_split_specs_for_dataset(MVTEC_LOCO, dataset),
    )
    run_directory = Path(args.result_root).expanduser().resolve() / args.class_name
    run_directory.mkdir(parents=True, exist_ok=True)
    progress_path = run_directory / "all.jsonl"
    summary_path = run_directory / "all_summary.json"
    completed = {} if args.restart else _load_completed(progress_path)
    mode = "w" if args.restart else "a"
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
                record = {
                    "image": key,
                    "split": split,
                    "label": int(label),
                    "score": report.image_anomaly_score,
                    "execution_mode": report.execution_mode,
                    "gpstm_evidence": payload["composition_evidence"],
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
    logical = _subgroup_auroc(records, "logical_anomalies")
    structural = _subgroup_auroc(records, "structural_anomalies")
    elapsed = sum(float(record["elapsed_seconds"]) for record in records)
    summary = {
        "module": "FiveRules+GPSTM",
        "class_name": args.class_name,
        "dataset": "MVTec LOCO",
        "splits": list(splits),
        "rule_inference_enabled": True,
        "schedule": "selective_five_rules_then_gpstm",
        "fusion": "rule_fail_early_exit_else_one_gpstm_score",
        "category_descriptors_enabled": False,
        "global_patch_bank": str(Path(args.global_bank).expanduser().resolve()),
        "gpstm_bank": str(Path(args.gpstm_bank).expanduser().resolve()),
        "missing_policy_categories_using_global_fallback": list(missing),
        "sample_count": len(records),
        "valid_count": len(valid),
        "failure_count": len(records) - len(valid),
        "elapsed_seconds": elapsed,
        "seconds_per_image": elapsed / len(records) if records else None,
        "image_auroc": overall,
        "logical_image_auroc": logical,
        "structural_image_auroc": structural,
        "split_image_aurocs": {
            "logical_anomalies": logical,
            "structural_anomalies": structural,
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
