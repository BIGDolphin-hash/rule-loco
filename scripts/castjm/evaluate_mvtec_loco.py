#!/usr/bin/env python3
"""Evaluate isolated CAST-JM on one MVTec LOCO class."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from pro_innovation.adapters import (
    DinoV2PatchFeatureExtractor,
    Sam3Segmenter,
)
from pro_innovation.evaluation import (
    MVTEC_LOCO,
    SUPPORTED_DATASETS,
    _collect_evaluation_samples,
    _split_specs_for_dataset,
    binary_auroc,
)
from pro_innovation.memory_modules.castjm import (
    CASTJMGlobalPatchBank,
    CASTJMMemoryBank,
)
from pro_innovation.mask_processing import MaskPostprocessor
from pro_innovation.sam3_thresholds import load_sam3_threshold_policy
from pro_innovation.templates import collect_categories, parse_template


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate isolated CAST-JM")
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=MVTEC_LOCO)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--generic")
    parser.add_argument("--threshold-policy")
    parser.add_argument("--global-bank", required=True)
    parser.add_argument(
        "--joint-bank",
        help="required only when a threshold policy enables joint CAST-JM scoring",
    )
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam3-repo", default="/home/lxq/models/SAM3")
    parser.add_argument("--sam3-checkpoint", default="/home/lxq/weights/sam3/sam3.pt")
    parser.add_argument(
        "--dino-checkpoint",
        default="/home/lxq/weights/DINOV2/dinov2_vitl14_reg4_pretrain.pth",
    )
    parser.add_argument("--sam3-threshold", type=float, default=0.40)
    parser.add_argument("--sam3-prompt", default="{category}")
    parser.add_argument("--sam3-cache-root", default="cache/sam3")
    parser.add_argument(
        "--sam3-cache-mode",
        choices=("off", "read", "read-write", "refresh"),
        default="read-write",
    )
    parser.add_argument("--mask-dedup-iou", type=float, default=0.80)
    parser.add_argument("--mask-dedup-containment", type=float, default=0.85)
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
    """Compare one anomaly type against the shared good-image negatives."""

    subset = [
        record
        for record in records
        if record["score"] is not None and record["split"] in ("good", positive_split)
    ]
    labels = {int(record["label"]) for record in subset}
    if labels != {0, 1}:
        return None
    return binary_auroc(
        [int(record["label"]) for record in subset],
        [float(record["score"]) for record in subset],
    )


def main() -> int:
    args = _arguments()
    generic_path = Path(args.generic).expanduser().resolve() if args.generic else None
    generic = (
        parse_template(
            generic_path.read_text(encoding="utf-8"),
            allow_missing_values=True,
        )
        if generic_path is not None and generic_path.is_file()
        else ()
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
    if whole_image_mode:
        categories: tuple[str, ...] = ()
        missing: tuple[str, ...] = ()
        print(
            "[CAST-JM] no threshold policy; returning the Global Patch anomaly "
            "score directly without segmentation or joint-memory processing",
            flush=True,
        )
    else:
        assert policy is not None
        categories = collect_categories(generic) or tuple(policy.selected_thresholds)
        if not categories:
            raise ValueError(
                "SAM3-backed CAST-JM requires template or threshold-policy categories"
            )
        missing = tuple(
            category
            for category in categories
            if category not in policy.selected_thresholds
        )
        if missing:
            print(
                "[CAST-JM] warning: threshold policy is missing "
                + ", ".join(missing)
                + f"; using global fallback {args.sam3_threshold:.2f}",
                flush=True,
            )

    global_bank = CASTJMGlobalPatchBank.load(
        args.global_bank,
        expected_class_name=args.class_name,
        expected_checkpoint_path=args.dino_checkpoint,
    )
    joint_bank = None
    if not whole_image_mode:
        if not args.joint_bank:
            raise ValueError("--joint-bank is required when a threshold policy is used")
        joint_bank = CASTJMMemoryBank.load(
            args.joint_bank,
            expected_class_name=args.class_name,
            expected_categories=categories,
            expected_checkpoint_path=args.dino_checkpoint,
            expected_global_patch_bank_path=args.global_bank,
        )
        if global_bank.feature_grid != joint_bank.feature_grid:
            raise ValueError("Global Patch and joint-bank feature grids differ")

    thresholds = (
        {
            category: float(
                policy.selected_thresholds.get(category, args.sam3_threshold)
            )
            for category in categories
        }
        if policy is not None
        else {}
    )
    segmenter = (
        None
        if whole_image_mode
        else Sam3Segmenter(
            repo_path=args.sam3_repo,
            checkpoint_path=args.sam3_checkpoint,
            device=args.device,
            confidence_threshold=args.sam3_threshold,
            category_thresholds=thresholds,
            prompt_template=args.sam3_prompt,
            cache_root=args.sam3_cache_root,
            cache_mode=args.sam3_cache_mode,
        )
    )
    processor = (
        None
        if whole_image_mode
        else MaskPostprocessor(
            dedup_iou_threshold=args.mask_dedup_iou,
            dedup_containment_threshold=args.mask_dedup_containment,
            dedup_enabled=True,
        )
    )
    extractor = DinoV2PatchFeatureExtractor(
        checkpoint_path=args.dino_checkpoint, device=args.device
    )

    dataset = Path(args.dataset_root).expanduser().resolve()
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
                bundles = extractor.extract_batch(
                    [image], batch_size=args.extraction_batch_size
                )
                if len(bundles) != 1:
                    raise RuntimeError("DINOv2 returned an unexpected batch size")
                bundle = bundles[0]
                if whole_image_mode:
                    evidence, _ = global_bank.score_with_map(
                        bundle,
                        device=args.device,
                        query_batch_size=args.query_batch_size,
                    )
                    mask_counts = {}
                else:
                    assert processor is not None
                    assert segmenter is not None
                    assert joint_bank is not None
                    pool = processor.process(
                        categories, segmenter.segment(image, categories)
                    )
                    anomaly_map = global_bank.anomaly_map(
                        bundle,
                        device=args.device,
                        query_batch_size=args.query_batch_size,
                    )
                    evidence = joint_bank.score_features(bundle, pool, anomaly_map)
                    mask_counts = {
                        category: len(pool.get(category, ()))
                        for category in categories
                    }
                record = {
                    "image": key,
                    "split": split,
                    "label": int(label),
                    "score": evidence.calibrated_score,
                    "evidence": asdict(evidence),
                    "mask_counts": mask_counts,
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
                f"score={record['score']} seconds={record['elapsed_seconds']:.2f}",
                flush=True,
            )

    records = [completed[str(image)] for image, _, _ in samples]
    valid = [record for record in records if record["score"] is not None]
    elapsed = sum(float(record["elapsed_seconds"]) for record in records)
    overall_auroc = (
        binary_auroc(
            [int(record["label"]) for record in valid],
            [float(record["score"]) for record in valid],
        )
        if {int(record["label"]) for record in valid} == {0, 1}
        else None
    )
    logical_auroc = (
        _subgroup_auroc(records, "logical_anomalies")
        if "logical_anomalies" in splits
        else None
    )
    structural_auroc = (
        _subgroup_auroc(records, "structural_anomalies")
        if "structural_anomalies" in splits
        else None
    )
    split_aurocs = {
        split: _subgroup_auroc(records, split)
        for split, (label, _) in split_specs.items()
        if label == 1
    }
    summary = {
        "module": "Global Patch" if whole_image_mode else "CAST-JM",
        "class_name": args.class_name,
        "dataset": args.dataset,
        "splits": list(splits),
        "scoring": (
            "direct_global_patch_score"
            if whole_image_mode
            else "one_ordered_joint_1nn_score"
        ),
        "rule_inference_enabled": False,
        "segmentation_mode": (
            "not_used" if whole_image_mode else "sam3_category_masks"
        ),
        "generic_template": (
            str(generic_path)
            if generic_path is not None and generic_path.is_file()
            else None
        ),
        "threshold_policy": (
            str(threshold_policy_path) if policy is not None else None
        ),
        "global_patch_bank": str(Path(args.global_bank).expanduser().resolve()),
        "joint_bank": (
            None
            if whole_image_mode
            else str(Path(args.joint_bank).expanduser().resolve())
        ),
        "texture_weight": None if joint_bank is None else joint_bank.texture_weight,
        "top_fraction": None if joint_bank is None else joint_bank.top_fraction,
        "missing_policy_categories_using_global_fallback": list(missing),
        "sample_count": len(records),
        "valid_count": len(valid),
        "failure_count": len(records) - len(valid),
        "elapsed_seconds": elapsed,
        "seconds_per_image": elapsed / len(records) if records else None,
        "auroc": overall_auroc,
        "image_auroc": overall_auroc,
        "logical_image_auroc": logical_auroc,
        "structural_image_auroc": structural_auroc,
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
