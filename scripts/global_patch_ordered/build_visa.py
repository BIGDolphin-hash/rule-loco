#!/usr/bin/env python3
"""Build full-shot Global Patch and GPO-JM banks for VisA."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

from pro_innovation.errors import MemoryBankError
from pro_innovation.memory_modules.global_patch import patch_memory_run_tag
from pro_innovation.memory_modules.gpojm import GlobalPatchOrderedMemoryBank


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "visa"
DEFAULT_GLOBAL_ROOT = (
    PROJECT_ROOT / "memory_banks" / "Global patch" / "full" / "visa" / "global"
)
DEFAULT_GPOJM_ROOT = (
    PROJECT_ROOT / "memory_banks" / "global_patch_ordered" / "full" / "visa"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "visa_patch_selection.json"
DEFAULT_DINO_CHECKPOINT = Path(
    "weights/DINOV2/dinov2_vitl14_reg4_pretrain.pth"
)


def _load_train_normal_dirs(dataset_root: Path) -> dict[str, Path]:
    manifest = dataset_root / "split_csv" / "1cls.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    grouped: dict[str, set[Path]] = {}
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
            grouped.setdefault(row["object"], set()).add(image.parent)

    normal_dirs: dict[str, Path] = {}
    for class_name, directories in sorted(grouped.items()):
        if len(directories) != 1:
            raise ValueError(
                f"VisA class {class_name} has multiple train-normal directories: "
                + ", ".join(str(path) for path in sorted(directories))
            )
        normal_dirs[class_name] = next(iter(directories))
    return normal_dirs


def _gpojm_run_tag(texture_weight: float, top_fraction: float) -> str:
    return (
        f"gpojm_tw{round(texture_weight * 100):03d}_"
        f"top{round(top_fraction * 100):03d}"
    )


def _joint_bank_matches(
    path: Path,
    *,
    class_name: str,
    checkpoint: Path,
    global_bank: Path,
) -> bool:
    if not path.is_file():
        return False
    try:
        GlobalPatchOrderedMemoryBank.load(
            path,
            expected_class_name=class_name,
            expected_categories=(class_name,),
            expected_checkpoint_path=checkpoint,
            expected_global_patch_bank_path=global_bank,
        )
    except (MemoryBankError, FileNotFoundError, ValueError):
        return False
    return True


def _run(command: list[str], *, environment: dict[str, str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build full-shot VisA Global Patch and GPO-JM banks. Without a "
            "threshold policy, GPO-JM treats the whole image as one object."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--global-root", type=Path, default=DEFAULT_GLOBAL_ROOT)
    parser.add_argument("--gpojm-root", type=Path, default=DEFAULT_GPOJM_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--dino-checkpoint", type=Path, default=DEFAULT_DINO_CHECKPOINT
    )
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--extraction-batch-size", type=int, default=1)
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--texture-weight", type=float, default=0.30)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    dataset_root = args.dataset_root.expanduser().resolve()
    global_root = args.global_root.expanduser().resolve()
    gpojm_root = args.gpojm_root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    checkpoint = args.dino_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    normal_dirs = _load_train_normal_dirs(dataset_root)
    available = tuple(normal_dirs)
    classes = tuple(args.classes) if args.classes else available
    unknown = sorted(set(classes) - set(available))
    if unknown:
        raise ValueError("unknown VisA classes: " + ", ".join(unknown))

    environment = dict(os.environ)
    source_root = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else source_root + os.pathsep + environment["PYTHONPATH"]
    )

    global_root.mkdir(parents=True, exist_ok=True)
    gpojm_root.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "scripts/build_visa_patch_banks.py",
            "--dataset-root",
            str(dataset_root),
            "--bank-root",
            str(global_root),
            "--manifest",
            str(manifest),
            "--checkpoint",
            str(checkpoint),
            "--classes",
            *classes,
            "--device",
            args.device,
            "--extraction-batch-size",
            str(args.extraction_batch_size),
        ],
        environment=environment,
    )

    joint_tag = _gpojm_run_tag(args.texture_weight, args.top_fraction)
    for index, class_name in enumerate(classes, start=1):
        global_bank = global_root / f"{class_name}.{patch_memory_run_tag()}.npz"
        joint_bank = gpojm_root / f"{class_name}.{joint_tag}.npz"
        if not global_bank.is_file():
            raise FileNotFoundError(global_bank)
        if _joint_bank_matches(
            joint_bank,
            class_name=class_name,
            checkpoint=checkpoint,
            global_bank=global_bank,
        ):
            print(f"[{index}/{len(classes)}] skip verified GPO-JM {class_name}")
            continue
        if joint_bank.exists():
            raise RuntimeError(f"existing GPO-JM bank failed validation: {joint_bank}")

        print(f"[{index}/{len(classes)}] build GPO-JM {class_name}", flush=True)
        _run(
            [
                sys.executable,
                "scripts/global_patch_ordered/build_memory.py",
                "--class-name",
                class_name,
                "--normal-dir",
                str(normal_dirs[class_name]),
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
        if not _joint_bank_matches(
            joint_bank,
            class_name=class_name,
            checkpoint=checkpoint,
            global_bank=global_bank,
        ):
            raise RuntimeError(f"GPO-JM bank validation failed: {joint_bank}")

    print("VisA full-shot Global Patch and GPO-JM construction complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
