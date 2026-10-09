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

from pro_innovation.errors import MemoryBankError
from pro_innovation.patch_memory import (
    DEFAULT_PATCH_DINOV2_CHECKPOINT,
    PatchMemoryBank,
    patch_memory_run_tag,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "visa"
DEFAULT_CALIBRATION_ROOT = PROJECT_ROOT / "datasets" / "visa_calibration"
DEFAULT_BANK_ROOT = PROJECT_ROOT / "memory_banks" / "4-shot" / "visa"
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "visa_patch_selection.json"
MEMORY_COUNT = 4
CALIBRATION_COUNT = 16


def _stratum_centers(length: int, count: int) -> tuple[int, ...]:
    if length < count:
        raise ValueError(f"cannot select {count} distinct items from {length}")
    return tuple(round((index + 0.5) * length / count - 0.5) for index in range(count))


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


def _select_images(
    images: tuple[Path, ...], category: str
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    memory_indices = _stratum_centers(len(images), MEMORY_COUNT)
    memory_images = tuple(images[index] for index in memory_indices)
    remaining = tuple(
        image for index, image in enumerate(images) if index not in memory_indices
    )
    calibration_images = tuple(
        remaining[index] for index in _stratum_centers(len(remaining), CALIBRATION_COUNT)
    )
    if set(memory_images) & set(calibration_images):
        raise RuntimeError(f"memory/calibration overlap for VisA class {category}")
    return memory_images, calibration_images


def _ensure_calibration_view(
    category: str,
    sources: tuple[Path, ...],
    calibration_root: Path,
) -> Path:
    destination = calibration_root / category / "normal"
    destination.mkdir(parents=True, exist_ok=True)
    expected_names = {source.name for source in sources}
    unexpected = [path for path in destination.iterdir() if path.name not in expected_names]
    if unexpected:
        raise RuntimeError(
            f"unexpected entries in calibration view {destination}: "
            + ", ".join(str(path) for path in unexpected)
        )
    for source in sources:
        link = destination / source.name
        if link.is_symlink():
            if link.resolve() != source:
                raise RuntimeError(f"calibration symlink points elsewhere: {link}")
            continue
        if link.exists():
            raise RuntimeError(f"calibration entry is not a symlink: {link}")
        link.symlink_to(source)
    return destination


def _bank_matches_selection(
    bank_path: Path,
    category: str,
    memory_images: tuple[Path, ...],
    calibration_images: tuple[Path, ...],
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
    return (
        tuple(bank.normal_images) == tuple(str(path) for path in memory_images)
        and tuple(bank.calibration_images)
        == tuple(str(path) for path in calibration_images)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--calibration-root", type=Path, default=DEFAULT_CALIBRATION_ROOT
    )
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
    calibration_root = args.calibration_root.expanduser().resolve()
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
    prepared: dict[str, tuple[tuple[Path, ...], tuple[Path, ...], Path]] = {}
    for category in categories:
        train_normals = all_train_normals[category]
        memory_images, calibration_images = _select_images(train_normals, category)
        calibration_dir = _ensure_calibration_view(
            category, calibration_images, calibration_root
        )
        bank_path = bank_root / f"{category}.{patch_memory_run_tag()}.npz"
        prepared[category] = (memory_images, calibration_images, bank_path)
        selections[category] = {
            "train_normal_count": len(train_normals),
            "selection_strategy": "official_train_normal_sorted_path_stratum_centers",
            "normal_images": [str(path) for path in memory_images],
            "calibration_images": [str(path) for path in calibration_images],
            "calibration_view": str(calibration_dir),
            "patch_bank": str(bank_path),
        }

    manifest = {
        "schema": "pro-innovation.visa-patch-selection.v1",
        "dataset_root": str(dataset_root),
        "split_manifest": str(dataset_root / "split_csv" / "1cls.csv"),
        "checkpoint": str(checkpoint),
        "memory_images_per_class": MEMORY_COUNT,
        "calibration_images_per_class": CALIBRATION_COUNT,
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
        memory_images, calibration_images, bank_path = prepared[category]
        if _bank_matches_selection(
            bank_path,
            category,
            memory_images,
            calibration_images,
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
            "--calibration-dir",
            str(calibration_root / category / "normal"),
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
                calibration_images,
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
