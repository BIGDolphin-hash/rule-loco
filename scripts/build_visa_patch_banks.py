#!/usr/bin/env python3
"""Build resumable Reg4 Patch banks for every VisA class."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image

from pro_innovation.errors import MemoryBankError
from pro_innovation.memory_modules.global_patch import (
    DEFAULT_PATCH_DINOV2_CHECKPOINT,
    PatchMemoryBank,
    patch_memory_run_tag,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "visa"
DEFAULT_BANK_ROOT = (
    PROJECT_ROOT / "memory_banks" / "Global patch" / "full" / "visa" / "global"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "visa_patch_selection.json"


def _readable_images(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    readable: list[Path] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.load()
        except Exception as exc:
            print(f"skip unreadable training image {path}: {exc}", flush=True)
            continue
        readable.append(path)
    return tuple(readable)


def _load_train_normals(dataset_root: Path) -> dict[str, tuple[Path, ...]]:
    manifest = dataset_root / "split_csv" / "1cls.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    grouped: dict[str, list[Path]] = {}
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"object", "split", "label", "image"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"VisA split manifest is missing columns {sorted(missing)}: {manifest}"
            )
        for row in reader:
            if row["split"].lower() != "train" or row["label"].lower() != "normal":
                continue
            image = (dataset_root / row["image"]).resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            grouped.setdefault(row["object"], []).append(image)
    return {
        category: tuple(sorted(set(images)))
        for category, images in sorted(grouped.items())
    }


def _bank_matches_selection(
    bank_path: Path,
    category: str,
    memory_images: tuple[Path, ...],
    checkpoint: Path,
) -> bool:
    if not bank_path.is_file():
        return False
    try:
        bank = PatchMemoryBank.load(
            bank_path,
            expected_class_name=category,
            expected_checkpoint_path=checkpoint,
        )
    except MemoryBankError:
        return False
    expected = tuple(str(path) for path in memory_images)
    return tuple(bank.normal_images) == expected and tuple(bank.calibration_images) == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--bank-root", type=Path, default=DEFAULT_BANK_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(DEFAULT_PATCH_DINOV2_CHECKPOINT)
    )
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--extraction-batch-size", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    bank_root = args.bank_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    all_train_normals = _load_train_normals(dataset_root)
    available = tuple(all_train_normals)
    categories = tuple(args.classes) if args.classes else available
    unknown = sorted(set(categories) - set(available))
    if unknown:
        raise ValueError("unknown VisA classes: " + ", ".join(unknown))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    selections: dict[str, dict[str, object]] = {}
    prepared: dict[str, tuple[tuple[Path, ...], Path]] = {}
    for category in categories:
        memory_images = _readable_images(all_train_normals[category])
        if len(memory_images) < 2:
            raise ValueError(f"not enough normal images for VisA class {category}")
        bank_path = bank_root / f"{category}.{patch_memory_run_tag()}.npz"
        prepared[category] = (memory_images, bank_path)
        selections[category] = {
            "train_normal_count": len(memory_images),
            "selection_strategy": "all_official_train_normal_with_patch_coreset",
            "normal_images": [str(path) for path in memory_images],
            "calibration_strategy": "all_train_normal_leave_one_image_out",
            "patch_bank": str(bank_path),
        }

    manifest = {
        "schema": "pro-innovation.visa-full-coreset-selection.v1",
        "dataset_root": str(dataset_root),
        "split_manifest": str(dataset_root / "split_csv" / "1cls.csv"),
        "checkpoint": str(checkpoint),
        "classes": selections,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"selection_manifest={args.manifest.resolve()}", flush=True)
    if args.prepare_only:
        return 0

    bank_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    source_root = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else source_root + os.pathsep + environment["PYTHONPATH"]
    )
    for index, category in enumerate(categories, start=1):
        memory_images, bank_path = prepared[category]
        if _bank_matches_selection(
            bank_path,
            category,
            memory_images,
            checkpoint,
        ):
            print(f"[{index}/{len(categories)}] skip verified {category}", flush=True)
            continue
        if bank_path.exists():
            raise RuntimeError(
                f"existing bank does not match the saved selection: {bank_path}"
            )
        command = [
            sys.executable,
            "-m",
            "pro_innovation",
            "build-patch-memory",
            "--class-name",
            category,
            "--normal-images",
            *(str(path) for path in memory_images),
            "--output",
            str(bank_path),
            "--patch-dinov2-checkpoint",
            str(checkpoint),
            "--device",
            args.device,
            "--patch-extraction-batch-size",
            str(args.extraction_batch_size),
        ]
        print(f"[{index}/{len(categories)}] build {category}", flush=True)
        for attempt in range(1, args.attempts + 1):
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                check=False,
            )
            bank_valid = _bank_matches_selection(
                bank_path,
                category,
                memory_images,
                checkpoint,
            )
            if bank_valid:
                if completed.returncode != 0:
                    print(
                        f"[{index}/{len(categories)}] child exit "
                        f"{completed.returncode}; bank passed independent verification",
                        flush=True,
                    )
                break
            if attempt < args.attempts:
                print(
                    f"[{index}/{len(categories)}] retry {category} "
                    f"after failed attempt {attempt}",
                    flush=True,
                )
        else:
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(completed.returncode, command)
            raise RuntimeError(f"bank verification failed: {bank_path}")
        print(f"[{index}/{len(categories)}] verified {category}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
