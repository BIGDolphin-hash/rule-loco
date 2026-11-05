#!/usr/bin/env python3
"""Evaluate pure-texture Global Patch Pixel AUROC on MVTec AD or VisA."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from pro_innovation.adapters.patch_dinov2 import (
    DEFAULT_PATCH_DINOV2_CHECKPOINT,
    DinoV2PatchFeatureExtractor,
)
from pro_innovation.memory_modules.gpojm import (
    GPOJMGlobalPatchBank,
    GPOJMGlobalPatchScorer,
    GlobalPatchPixelLocalizer,
    GlobalPatchPixelScorer,
)
from pro_innovation.pixel_evaluation import (
    MVTEC_AD,
    VISA,
    SUPPORTED_PIXEL_DATASETS,
    binary_auroc,
    collect_pixel_samples,
    load_binary_mask,
    macro_average,
    pixel_auroc,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOTS = {
    MVTEC_AD: PROJECT_ROOT / "datasets" / "mvtec_anomaly_detection",
    VISA: PROJECT_ROOT / "datasets" / "visa",
}
DEFAULT_GLOBAL_ROOT = PROJECT_ROOT / "memory_banks" / "Global patch"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results" / "gppl_pixel"
RUN_SCHEMA = "pro-innovation.gppl-pixel-evaluation.v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the pure-texture GPPL head with existing Global Patch banks; "
            "SAM3, templates, threshold policies, and GPO-JM joint memories are not used."
        )
    )
    parser.add_argument("--dataset", choices=SUPPORTED_PIXEL_DATASETS, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--classes", nargs="+")
    parser.add_argument(
        "--shot",
        choices=("full", "1", "2", "4"),
        default="full",
        help="select the matching full-shot or few-shot Global Patch bank directory",
    )
    parser.add_argument(
        "--global-root", type=Path, default=DEFAULT_GLOBAL_ROOT
    )
    parser.add_argument(
        "--bank-root",
        type=Path,
        help="override the resolved directory containing one Global Patch bank per class",
    )
    parser.add_argument(
        "--dino-checkpoint", type=Path, default=DEFAULT_PATCH_DINOV2_CHECKPOINT
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--extraction-batch-size", type=int, default=4)
    parser.add_argument(
        "--evaluation-size",
        type=int,
        default=448,
        help="square resolution used for both predicted maps and ground-truth masks",
    )
    parser.add_argument("--smoothing-sigma", type=float, default=0.0)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard cached Patch maps for each selected class before evaluation",
    )
    return parser.parse_args()


def _available_classes(dataset: str, dataset_root: Path) -> tuple[str, ...]:
    if dataset == MVTEC_AD:
        return tuple(
            sorted(
                path.name
                for path in dataset_root.iterdir()
                if path.is_dir() and (path / "test").is_dir()
            )
        )
    manifest = dataset_root / "split_csv" / "1cls.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        return tuple(sorted({row["object"] for row in csv.DictReader(stream)}))


def _resolved_bank_root(args: argparse.Namespace) -> Path:
    if args.bank_root is not None:
        return args.bank_root.expanduser().resolve()
    global_root = args.global_root.expanduser().resolve()
    if args.shot == "full":
        return global_root / "full" / args.dataset / "global"
    return global_root / "few_shot" / f"{args.shot}shot" / args.dataset / "global"


def _resolve_bank(bank_root: Path, class_name: str, shot: str) -> Path:
    if shot == "full":
        matches = sorted(bank_root.glob(f"{class_name}.dino_full_*.npz"))
    else:
        matches = sorted(bank_root.glob(f"{class_name}.dino_{shot}shot_fullpatch.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {shot}-shot Global Patch bank for {class_name} "
            f"under {bank_root}, found {len(matches)}"
        )
    return matches[0].resolve()


def _run_configuration(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    class_name: str,
    bank_path: Path,
    samples,
) -> dict[str, object]:
    bank_stat = bank_path.stat()
    return {
        "schema": RUN_SCHEMA,
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "class_name": class_name,
        "shot": args.shot,
        "global_patch_bank": str(bank_path),
        "global_patch_bank_size": bank_stat.st_size,
        "global_patch_bank_mtime_ns": bank_stat.st_mtime_ns,
        "dino_checkpoint": str(args.dino_checkpoint.expanduser().resolve()),
        "evaluation_size": args.evaluation_size,
        "smoothing_sigma": args.smoothing_sigma,
        "samples": [
            {
                "image": str(sample.image_path),
                "mask": None if sample.mask_path is None else str(sample.mask_path),
                "split": sample.split,
            }
            for sample in samples
        ],
    }


def _prepare_class_output(
    class_output: Path,
    configuration: dict[str, object],
    *,
    restart: bool,
) -> Path:
    if restart and class_output.exists():
        shutil.rmtree(class_output)
    config_path = class_output / "run_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        # AUPRO was removed from this evaluator; keep its already-computed Patch
        # maps reusable when upgrading an earlier run directory.
        existing.pop("aupro_max_fpr", None)
        existing.pop("aupro_thresholds", None)
        if existing != configuration:
            raise ValueError(
                f"cached GPPL run differs for {class_output}; rerun with --restart"
            )
    elif class_output.exists() and any(class_output.iterdir()):
        raise ValueError(
            f"GPPL output has no compatible run_config.json: {class_output}"
        )
    class_output.mkdir(parents=True, exist_ok=True)
    (class_output / "maps").mkdir(exist_ok=True)
    config_path.write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return class_output / "maps"


def _load_cached_patch_map(path: Path, feature_grid: tuple[int, int]) -> np.ndarray | None:
    if not path.is_file():
        return None
    values = np.load(path, allow_pickle=False)
    if values.shape != feature_grid or values.dtype != np.float32:
        return None
    if not np.isfinite(values).all() or bool((values < 0.0).any()):
        return None
    return values


def _save_patch_map(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, values.astype(np.float32, copy=False), allow_pickle=False)
    temporary.replace(path)


def _evaluate_class(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    class_name: str,
    bank_path: Path,
    extractor: DinoV2PatchFeatureExtractor,
    output_root: Path,
) -> dict[str, object]:
    samples = collect_pixel_samples(
        dataset=args.dataset,
        dataset_root=dataset_root,
        class_name=class_name,
    )
    configuration = _run_configuration(
        args=args,
        dataset_root=dataset_root,
        class_name=class_name,
        bank_path=bank_path,
        samples=samples,
    )
    class_output = output_root / class_name
    map_root = _prepare_class_output(
        class_output, configuration, restart=args.restart
    )

    bank = GPOJMGlobalPatchBank.load(
        bank_path,
        expected_class_name=class_name,
        expected_checkpoint_path=args.dino_checkpoint,
    )
    global_patch_scorer = GPOJMGlobalPatchScorer(
        bank=bank,
        extractor=extractor,
        query_batch_size=args.query_batch_size,
        extraction_batch_size=args.extraction_batch_size,
    )
    localizer = GlobalPatchPixelLocalizer(smoothing_sigma=args.smoothing_sigma)
    pixel_scorer = GlobalPatchPixelScorer(
        global_patch_scorer=global_patch_scorer,
        localizer=localizer,
    )
    evaluation_size = (args.evaluation_size, args.evaluation_size)
    started = time.perf_counter()
    cache_hits = 0
    for index, sample in enumerate(samples, start=1):
        map_path = map_root / f"{index:04d}.npy"
        patch_map = _load_cached_patch_map(map_path, bank.feature_grid)
        if patch_map is None:
            localization = pixel_scorer.localize(
                sample.image_path,
                output_size=evaluation_size,
            )
            patch_map = localization.patch_anomaly_map.numpy().astype(
                np.float32, copy=False
            )
            _save_patch_map(map_path, patch_map)
            status = "inferred"
        else:
            cache_hits += 1
            status = "cached"
        print(
            f"[{class_name} {index}/{len(samples)}] {sample.split}/"
            f"{sample.image_path.name} {status}",
            flush=True,
        )

    score_maps = np.empty(
        (len(samples), args.evaluation_size, args.evaluation_size), dtype=np.float32
    )
    masks = np.empty(score_maps.shape, dtype=np.bool_)
    image_scores = np.empty(len(samples), dtype=np.float32)
    image_labels = np.empty(len(samples), dtype=np.bool_)
    for index, sample in enumerate(samples, start=1):
        patch_map = np.load(map_root / f"{index:04d}.npy", allow_pickle=False)
        image_scores[index - 1] = float(patch_map.max())
        image_labels[index - 1] = sample.anomalous
        localization = localizer.localize(
            torch.from_numpy(patch_map),
            feature_grid=bank.feature_grid,
            output_size=evaluation_size,
        )
        score_maps[index - 1] = localization.pixel_anomaly_map.numpy()
        masks[index - 1] = load_binary_mask(sample, output_size=evaluation_size)

    image_auroc = binary_auroc(image_labels, image_scores)
    pixel_score = pixel_auroc(masks, score_maps)
    summary: dict[str, object] = {
        "schema": RUN_SCHEMA,
        "dataset": args.dataset,
        "class_name": class_name,
        "shot": args.shot,
        "images": len(samples),
        "anomalous_images": sum(sample.anomalous for sample in samples),
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_score,
        "evaluation_size": args.evaluation_size,
        "smoothing_sigma": args.smoothing_sigma,
        "cache_hits": cache_hits,
        "elapsed_seconds": time.perf_counter() - started,
        "global_patch_bank": str(bank_path),
        "uses_sam3": False,
        "uses_templates": False,
        "uses_threshold_policy": False,
        "uses_gpojm_joint_memory": False,
    }
    (class_output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[{class_name}] Image AUROC={image_auroc * 100:.2f}% "
        f"Pixel AUROC={pixel_score * 100:.2f}%",
        flush=True,
    )
    return summary


def main() -> int:
    args = _arguments()
    if args.evaluation_size <= 0:
        raise ValueError("--evaluation-size must be positive")
    if args.query_batch_size <= 0 or args.extraction_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    dataset_root = (
        args.dataset_root or DEFAULT_DATASET_ROOTS[args.dataset]
    ).expanduser().resolve()
    checkpoint = args.dino_checkpoint.expanduser().resolve()
    args.dino_checkpoint = checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    available = _available_classes(args.dataset, dataset_root)
    classes = tuple(args.classes or available)
    unknown = sorted(set(classes) - set(available))
    if unknown:
        raise ValueError(f"unknown {args.dataset} classes: {', '.join(unknown)}")

    bank_root = _resolved_bank_root(args)
    banks = {
        class_name: _resolve_bank(bank_root, class_name, args.shot)
        for class_name in classes
    }
    shot_directory = "full" if args.shot == "full" else f"{args.shot}shot"
    output_root = (
        args.result_root.expanduser().resolve() / args.dataset / shot_directory
    )
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"dataset={args.dataset} shot={args.shot} classes={len(classes)}")
    print(f"bank_root={bank_root}")
    print(f"output_root={output_root}")
    print(
        "mode=GPPL pure texture (no SAM3/template/threshold policy/joint memory)",
        flush=True,
    )

    extractor = DinoV2PatchFeatureExtractor(
        checkpoint_path=checkpoint,
        device=args.device,
    )
    summaries = [
        _evaluate_class(
            args=args,
            dataset_root=dataset_root,
            class_name=class_name,
            bank_path=banks[class_name],
            extractor=extractor,
            output_root=output_root,
        )
        for class_name in classes
    ]
    metric_summaries = [
        {
            "image_auroc": float(item["image_auroc"]),
            "pixel_auroc": float(item["pixel_auroc"]),
        }
        for item in summaries
    ]
    macro = macro_average(metric_summaries)
    aggregate = {
        "schema": RUN_SCHEMA,
        "dataset": args.dataset,
        "shot": args.shot,
        "classes": summaries,
        "macro_image_auroc": macro["image_auroc"],
        "macro_pixel_auroc": macro["pixel_auroc"],
        "evaluation_size": args.evaluation_size,
        "smoothing_sigma": args.smoothing_sigma,
    }
    (output_root / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[macro] Image AUROC={macro['image_auroc'] * 100:.2f}% "
        f"Pixel AUROC={macro['pixel_auroc'] * 100:.2f}%",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted; cached Patch maps are retained", file=sys.stderr)
        raise SystemExit(130)
