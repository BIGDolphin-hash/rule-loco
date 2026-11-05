#!/usr/bin/env python3
"""Build MVTec AD or VisA few-shot Global Patch and GPO-JM banks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

from pro_innovation.adapters.patch_dinov2 import DinoV2PatchFeatureExtractor
from pro_innovation.memory_modules.global_patch import build_patch_memory_from_images
from pro_innovation.memory_modules.gpojm import (
    GPOJMGlobalPatchBank,
    GlobalPatchOrderedMemoryBank,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "mvtec_anomaly_detection"
DEFAULT_VISA_DATASET_ROOT = PROJECT_ROOT / "datasets" / "visa"
DEFAULT_GLOBAL_ROOT = PROJECT_ROOT / "memory_banks" / "Global patch" / "few_shot"
DEFAULT_GPOJM_ROOT = PROJECT_ROOT / "memory_banks" / "global_patch_ordered" / "few_shot"
DEFAULT_MANIFEST_ROOT = PROJECT_ROOT / "config" / "few_shot"
DEFAULT_DINO_CHECKPOINT = Path(
    "weights/DINOV2/dinov2_vitl14_reg4_pretrain.pth"
)


def _selection_seed(class_name: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{class_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def select_nested_images(
    images: tuple[Path, ...],
    *,
    class_name: str,
    seed: int,
    maximum_shot: int,
) -> tuple[Path, ...]:
    """Return one deterministic permutation shared by every requested shot."""

    if maximum_shot < 1:
        raise ValueError("few-shot selection requires at least one normal image")
    if len(images) < maximum_shot:
        raise ValueError(
            f"{class_name} has {len(images)} normal images, fewer than {maximum_shot}"
        )
    selected = list(images)
    random.Random(_selection_seed(class_name, seed)).shuffle(selected)
    return tuple(selected[:maximum_shot])


def _run(command: list[str], *, environment: dict[str, str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


def _normal_images_by_class(
    dataset: str, dataset_root: Path
) -> dict[str, tuple[Path, ...]]:
    if dataset == "mvtec_ad":
        return {
            path.name: tuple(
                image.resolve()
                for image in sorted((path / "train" / "good").glob("*.png"))
            )
            for path in sorted(dataset_root.iterdir())
            if path.is_dir()
        }
    manifest = dataset_root / "split_csv" / "1cls.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    grouped: dict[str, list[Path]] = {}
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row["split"].lower() == "train" and row["label"].lower() == "normal":
                grouped.setdefault(row["object"], []).append(
                    (dataset_root / row["image"]).resolve()
                )
    return {
        class_name: tuple(sorted(set(images)))
        for class_name, images in sorted(grouped.items())
    }


def _global_bank_matches(
    path: Path,
    *,
    class_name: str,
    images: tuple[Path, ...],
    checkpoint: Path,
) -> bool:
    if not path.is_file():
        return False
    bank = GPOJMGlobalPatchBank.load(
        path,
        expected_class_name=class_name,
        expected_checkpoint_path=checkpoint,
    )
    return bank.normal_images == tuple(str(image) for image in images) and np.isclose(
        bank.coreset_ratio, 1.0
    )


def _gpojm_bank_matches(
    path: Path,
    *,
    class_name: str,
    images: tuple[Path, ...],
    checkpoint: Path,
    global_bank: Path,
) -> bool:
    if not path.is_file():
        return False
    bank = GlobalPatchOrderedMemoryBank.load(
        path,
        expected_class_name=class_name,
        expected_categories=(class_name,),
        expected_checkpoint_path=checkpoint,
        expected_global_patch_bank_path=global_bank,
    )
    return bank.normal_images == tuple(str(image) for image in images)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build nested MVTec AD/VisA few-shot Global Patch and GPO-JM banks; "
            "all selected Patch features are retained (coreset ratio 1.0)."
        )
    )
    parser.add_argument("--dataset", choices=("mvtec_ad", "visa"), default="mvtec_ad")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--global-root", type=Path, default=DEFAULT_GLOBAL_ROOT)
    parser.add_argument("--gpojm-root", type=Path, default=DEFAULT_GPOJM_ROOT)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--dino-checkpoint", type=Path, default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--shots", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--extraction-batch-size", type=int, default=4)
    parser.add_argument("--texture-weight", type=float, default=0.30)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument(
        "--global-only",
        action="store_true",
        help="build only Global Patch banks and skip GPO-JM joint memories",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    default_dataset_root = (
        DEFAULT_DATASET_ROOT if args.dataset == "mvtec_ad" else DEFAULT_VISA_DATASET_ROOT
    )
    dataset_root = (args.dataset_root or default_dataset_root).expanduser().resolve()
    global_root = args.global_root.expanduser().resolve()
    gpojm_root = args.gpojm_root.expanduser().resolve()
    manifest_root = args.manifest_root.expanduser().resolve()
    checkpoint = args.dino_checkpoint.expanduser().resolve()
    shots = tuple(sorted(set(int(value) for value in args.shots)))
    if not shots or shots[0] < 1:
        raise ValueError("shots must contain positive integers")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    normal_images = _normal_images_by_class(args.dataset, dataset_root)
    available = sorted(normal_images)
    classes = tuple(args.classes or available)
    unknown = sorted(set(classes) - set(available))
    if unknown:
        raise ValueError(f"unknown {args.dataset} classes: " + ", ".join(unknown))

    selections: dict[str, tuple[Path, ...]] = {}
    maximum_shot = max(shots)
    for class_name in classes:
        images = normal_images[class_name]
        selections[class_name] = select_nested_images(
            images,
            class_name=class_name,
            seed=args.seed,
            maximum_shot=maximum_shot,
        )

    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        manifest_root
        / args.dataset
        / f"seed{args.seed}_shots-{'-'.join(map(str, shots))}.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "pro-innovation.few-shot-full-patch-selection.v2",
                "dataset": args.dataset,
                "dataset_root": str(dataset_root),
                "seed": args.seed,
                "shots": list(shots),
                "selection": "deterministic_nested_random",
                "patch_retention": "all_selected_patch_features",
                "patch_coreset_ratio": 1.0,
                "classes": {
                    class_name: [str(path) for path in images]
                    for class_name, images in selections.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"selection_manifest={manifest_path}", flush=True)

    environment = dict(os.environ)
    source_root = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else source_root + os.pathsep + environment["PYTHONPATH"]
    )

    shared_patch_extractor: DinoV2PatchFeatureExtractor | None = None

    for shot in shots:
        shot_global_root = global_root / f"{shot}shot" / args.dataset / "global"
        shot_gpojm_root = gpojm_root / f"{shot}shot" / args.dataset
        shot_global_root.mkdir(parents=True, exist_ok=True)
        if not args.global_only:
            shot_gpojm_root.mkdir(parents=True, exist_ok=True)
        for index, class_name in enumerate(classes, start=1):
            images = selections[class_name][:shot]
            normal_dir = images[0].parent
            global_bank = (
                shot_global_root / f"{class_name}.dino_{shot}shot_fullpatch.npz"
            )
            joint_bank = (
                shot_gpojm_root
                / f"{class_name}.gpojm_{shot}shot_fullpatch_tw030_top010.npz"
            )

            print(
                f"[{shot}-shot {index}/{len(classes)}] {class_name}: Global Patch",
                flush=True,
            )
            if not _global_bank_matches(
                global_bank,
                class_name=class_name,
                images=images,
                checkpoint=checkpoint,
            ):
                if global_bank.exists():
                    print(
                        f"replace mismatched Global Patch bank: {global_bank}",
                        flush=True,
                    )
                if shared_patch_extractor is None:
                    print("load shared DINOv2 patch extractor", flush=True)
                    shared_patch_extractor = DinoV2PatchFeatureExtractor(
                        checkpoint_path=checkpoint,
                        device=args.device,
                    )
                build_patch_memory_from_images(
                    class_name=class_name,
                    normal_images=images,
                    output_path=global_bank,
                    checkpoint_path=checkpoint,
                    device=args.device,
                    query_batch_size=args.query_batch_size,
                    extraction_batch_size=args.extraction_batch_size,
                    coreset_ratio=1.0,
                    extractor=shared_patch_extractor,
                )

            if args.global_only:
                continue

            print(
                f"[{shot}-shot {index}/{len(classes)}] {class_name}: GPO-JM",
                flush=True,
            )
            if shot == 1:
                print(
                    "skip one-shot joint bank: no-threshold GPO-JM directly uses "
                    "the one-shot Global Patch bank",
                    flush=True,
                )
                continue
            if not _gpojm_bank_matches(
                joint_bank,
                class_name=class_name,
                images=images,
                checkpoint=checkpoint,
                global_bank=global_bank,
            ):
                if joint_bank.exists():
                    print(
                        f"replace mismatched GPO-JM bank: {joint_bank}",
                        flush=True,
                    )
                _run(
                    [
                        sys.executable,
                        "scripts/global_patch_ordered/build_memory.py",
                        "--class-name",
                        class_name,
                        "--normal-dir",
                        str(normal_dir),
                        "--global-bank",
                        str(global_bank),
                        "--output",
                        str(joint_bank),
                        "--dino-checkpoint",
                        str(checkpoint),
                        "--device",
                        args.device,
                        "--batch-size",
                        str(args.extraction_batch_size),
                        "--query-batch-size",
                        str(args.query_batch_size),
                        "--texture-weight",
                        str(args.texture_weight),
                        "--top-fraction",
                        str(args.top_fraction),
                    ],
                    environment=environment,
                )

    built_modules = "Global Patch" if args.global_only else "Global Patch/GPO-JM"
    print(f"{args.dataset} few-shot {built_modules} construction complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
