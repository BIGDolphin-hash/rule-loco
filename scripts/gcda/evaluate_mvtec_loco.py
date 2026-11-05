#!/usr/bin/env python3
"""Evaluate isolated GCDA on one MVTec LOCO class without rule inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from pro_innovation.adapters import (
    DinoV2PatchFeatureExtractor,
    Sam3Segmenter,
)
from pro_innovation.evaluation import (
    MVTEC_LOCO,
    _collect_evaluation_samples,
    _split_specs_for_dataset,
    binary_auroc,
)
from pro_innovation.memory_modules.gcda import (
    GCDAGlobalPatchBank,
    GCDAGlobalPatchScorer,
    CategoryOnlyCompositionMemoryBank,
    CategoryOnlyCompositionScorer,
)
from pro_innovation.mask_processing import MaskPostprocessor
from pro_innovation.pipeline import LogicAnomalyPipeline
from pro_innovation.sam3_calibration import load_sam3_threshold_policy
from pro_innovation.templates import collect_categories, parse_template


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GCDA = Global Texture Memory + Category Composition Memory"
    )
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--generic", required=True)
    parser.add_argument("--standard", required=True)
    parser.add_argument("--attribute-vocabulary")
    parser.add_argument("--threshold-policy", required=True)
    parser.add_argument("--global-bank", required=True)
    parser.add_argument("--category-only-bank")
    parser.add_argument(
        "--category-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable the category-only Composition path",
    )
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
    parser.add_argument("--patch-query-batch-size", type=int, default=256)
    parser.add_argument("--extraction-batch-size", type=int, default=4)
    parser.add_argument("--fusion-tau", type=float, default=1.0)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def _load_completed(path: Path) -> dict[str, dict]:
    completed: dict[str, dict] = {}
    if not path.is_file():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        completed[str(record["image"])] = record
    return completed


def main() -> int:
    args = _arguments()
    dataset = Path(args.dataset_root).expanduser().resolve()
    generic_path = Path(args.generic).expanduser().resolve()
    generic = parse_template(
        generic_path.read_text(encoding="utf-8"), allow_missing_values=True
    )
    standard = parse_template(
        Path(args.standard).expanduser().resolve().read_text(encoding="utf-8"),
        allow_missing_values=False,
    )
    categories = collect_categories(generic)
    policy = load_sam3_threshold_policy(
        args.threshold_policy, expected_class_name=args.class_name
    )

    global_bank = GCDAGlobalPatchBank.load(
        args.global_bank,
        expected_class_name=args.class_name,
        expected_checkpoint_path=args.dino_checkpoint,
    )
    if args.category_only:
        if not args.category_only_bank:
            raise ValueError("--category-only requires --category-only-bank")
        category_bank = CategoryOnlyCompositionMemoryBank.load(
            args.category_only_bank,
            expected_class_name=args.class_name,
            expected_checkpoint_path=args.dino_checkpoint,
            expected_categories=categories,
        )
    else:
        category_bank = None
    extractor = DinoV2PatchFeatureExtractor(
        checkpoint_path=args.dino_checkpoint, device=args.device
    )
    global_scorer = GCDAGlobalPatchScorer(
        bank=global_bank,
        extractor=extractor,
        query_batch_size=args.patch_query_batch_size,
        extraction_batch_size=args.extraction_batch_size,
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
    pipeline = LogicAnomalyPipeline(
        segmenter,
        None,
        mask_processor=processor,
        patch_scorer=global_scorer,
        composition_scorer=(
            CategoryOnlyCompositionScorer(category_bank)
            if category_bank is not None
            else None
        ),
        rule_inference_enabled=False,
        post_fusion_tau=args.fusion_tau,
    )

    splits = ("good", "logical_anomalies", "structural_anomalies")
    split_specs = _split_specs_for_dataset(MVTEC_LOCO, dataset)
    samples = _collect_evaluation_samples(
        dataset_name=MVTEC_LOCO,
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
                report = pipeline.run(image, generic, standard)
                payload = report.to_dict()
                record = {
                    "image": key,
                    "split": split,
                    "label": int(label),
                    "score": report.image_anomaly_score,
                    "execution_mode": report.execution_mode,
                    "patch_evidence": payload["patch_evidence"],
                    "composition_evidence": payload["composition_evidence"],
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
                f"score={record['score']} seconds={record['elapsed_seconds']:.2f}",
                flush=True,
            )

    records = [completed[str(image)] for image, _, _ in samples]
    valid = [record for record in records if record["score"] is not None]
    summary = {
        "module": "GCDA",
        "class_name": args.class_name,
        "dataset": "MVTec LOCO",
        "splits": list(splits),
        "rule_inference_enabled": False,
        "schedule": "gcda_only",
        "fusion": (
            "r_global_plus_r_category_only_then_r_over_r_plus_tau"
            if args.category_only
            else "global_patch_only"
        ),
        "fusion_tau": args.fusion_tau,
        "global_bank": str(Path(args.global_bank).expanduser().resolve()),
        "category_only_enabled": bool(args.category_only),
        "category_only_bank": (
            str(Path(args.category_only_bank).expanduser().resolve())
            if args.category_only_bank and args.category_only
            else None
        ),
        "sample_count": len(records),
        "valid_count": len(valid),
        "failure_count": len(records) - len(valid),
        "auroc": (
            binary_auroc(
                [int(record["label"]) for record in valid],
                [float(record["score"]) for record in valid],
            )
            if valid
            else None
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
