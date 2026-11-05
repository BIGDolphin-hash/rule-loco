#!/usr/bin/env python3
"""Build resumable Reg4 Patch banks for every MVTec AD class."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image

from pro_innovation.memory_modules.global_patch import (
    DEFAULT_PATCH_DINOV2_CHECKPOINT,
    PatchMemoryBank,
    patch_memory_run_tag,
)
from pro_innovation.errors import MemoryBankError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "mvtec_anomaly_detection"
DEFAULT_BANK_ROOT = (
    PROJECT_ROOT / "memory_banks" / "Global patch" / "full" / "mvtec_ad" / "global"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "mvtec_ad_patch_selection.json"


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
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    bank_root = args.bank_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    available = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    categories = args.classes or available
    unknown = sorted(set(categories) - set(available))
    if unknown:
        raise ValueError("unknown MVTec AD classes: " + ", ".join(unknown))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    selections: dict[str, dict[str, object]] = {}
    prepared: dict[str, tuple[tuple[Path, ...], Path, Path]] = {}
    for category in categories:
        train_good = dataset_root / category / "train" / "good"
        memory_images = _readable_images(
            tuple(path.resolve() for path in sorted(train_good.glob("*.png")))
        )
        if len(memory_images) < 2:
            raise ValueError(f"not enough normal images under {train_good}")
        bank_path = bank_root / f"{category}.{patch_memory_run_tag()}.npz"
        prepared[category] = (memory_images, train_good, bank_path)
        selections[category] = {
            "train_good_count": len(memory_images),
            "selection_strategy": "all_train_good_with_patch_coreset",
            "normal_images": [str(path) for path in memory_images],
            "calibration_strategy": "all_train_good_leave_one_image_out",
            "patch_bank": str(bank_path),
        }

    manifest = {
        "schema": "pro-innovation.mvtec-ad-full-coreset-selection.v1",
        "dataset_root": str(dataset_root),
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
        memory_images, train_good, bank_path = prepared[category]
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
            "--normal-dir",
            str(train_good),
            "--output",
            str(bank_path),
            "--patch-dinov2-checkpoint",
            str(checkpoint),
            "--device",
            args.device,
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
                        f"{completed.returncode}; bank passed independent "
                        "verification",
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
