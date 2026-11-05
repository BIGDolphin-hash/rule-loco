#!/usr/bin/env python3
"""Check and optionally repair full-normal Global Patch texture banks."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
from pathlib import Path
import re
import shutil
import uuid
import zipfile

import numpy as np

from pro_innovation.adapters.patch_dinov2 import (
    DinoV2PatchFeatureExtractor,
    PATCH_MATCHING_LAYERS,
)
from pro_innovation.memory_modules.global_patch.core import (
    PATCH_MEMORY_IMAGE_SHARD_SCHEMA,
    PATCH_MEMORY_PER_IMAGE_STORAGE,
    PatchMemoryBank,
    _build_coreset,
    _sha256_file,
    _write_npz_verified,
)

GLOBAL_PATTERN = "*.dino_full_coreset05_b6b12b18b24.npz"


def _read_metadata(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise RuntimeError(f"manifest CRC failed: {bad_member}")
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["metadata"].item()))


def _remove_verified_manifest_backups(bank_path: Path) -> None:
    """Delete repair snapshots only after the active bank passes verification."""

    pattern = f"{bank_path.name}.manifest-before-repair-*"
    for backup in sorted(bank_path.parent.glob(pattern)):
        if not backup.is_file():
            continue
        backup.unlink()
        print(f"DELETE verified old manifest {backup.name}", flush=True)


def _repair_global_shard(
    bank_path: Path,
    metadata: dict,
    image_index: int,
    *,
    device: str,
) -> None:
    entry = dict(metadata["image_shards"][image_index])
    store = bank_path.parent / metadata["image_shard_store"]
    old_shard = store / str(entry["file"])
    source_image = Path(entry["image_path"]).expanduser().resolve()
    if not source_image.is_file():
        raise RuntimeError(f"source image is missing: {source_image}")

    preserved: dict[int, np.ndarray] = {}
    damaged_layers: set[int] = set(PATCH_MATCHING_LAYERS)
    if old_shard.is_file():
        readable: dict[int, np.ndarray] = {}
        for layer in PATCH_MATCHING_LAYERS:
            try:
                with np.load(old_shard, allow_pickle=False) as archive:
                    readable[layer] = archive[f"features_block_{layer}"].copy()
            except Exception:
                continue
        with zipfile.ZipFile(old_shard) as archive:
            bad_member = archive.testzip()
        match = re.fullmatch(r"features_block_(6|12|18|24)\.npy", bad_member or "")
        if match is not None:
            damaged_layers = {int(match.group(1))}
            preserved = {
                layer: values
                for layer, values in readable.items()
                if layer not in damaged_layers
            }

    extractor = DinoV2PatchFeatureExtractor(
        checkpoint_path=metadata["checkpoint_path"],
        device=device,
    )
    bundles = extractor.extract_batch((source_image,), batch_size=1)
    if len(bundles) != 1:
        raise RuntimeError("DINOv2 did not return exactly one feature bundle")
    if tuple(bundles[0].feature_grid) != tuple(metadata["feature_grid"]):
        raise RuntimeError("regenerated feature grid differs from the manifest")
    rebuilt, sources = _build_coreset(
        bundles,
        ratio=float(metadata["coreset_ratio"]),
        seed=int(metadata["coreset_seed"]),
        image_index_offset=image_index,
    )
    if set(sources.tolist()) != {image_index}:
        raise RuntimeError("regenerated global source indices are invalid")

    patch_count = int(sources.numel())
    shard_metadata = {
        "schema": PATCH_MEMORY_IMAGE_SHARD_SCHEMA,
        "image_index": image_index,
        "image_path": str(source_image),
        "layers": list(PATCH_MATCHING_LAYERS),
        "patch_count": patch_count,
    }
    arrays: dict[str, np.ndarray] = {
        "metadata": np.asarray(json.dumps(shard_metadata, ensure_ascii=False))
    }
    feature_shapes: dict[str, list[int]] = {}
    for layer in PATCH_MATCHING_LAYERS:
        values = preserved.get(layer, rebuilt[layer].numpy())
        expected_shape = entry.get("feature_shapes", {}).get(str(layer))
        if expected_shape is not None and list(values.shape) != expected_shape:
            raise RuntimeError(
                f"Block {layer} shape changed: {list(values.shape)} != {expected_shape}"
            )
        if not np.isfinite(values).all():
            raise RuntimeError(f"Block {layer} contains non-finite values")
        arrays[f"features_block_{layer}"] = values
        feature_shapes[str(layer)] = list(values.shape)

    repair_name = f"image_{image_index:06d}.repair-{uuid.uuid4().hex[:12]}.npz"
    repair_path = store / repair_name
    digest = _write_npz_verified(repair_path, arrays)
    entry.update(
        {
            "file": repair_name,
            "sha256": digest,
            "patch_count": patch_count,
            "feature_shapes": feature_shapes,
            "feature_dtype": "float32",
            "repair": {
                "old_file": old_shard.name,
                "regenerated_blocks": sorted(damaged_layers),
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            },
        }
    )
    metadata["image_shards"][image_index] = entry
    _write_npz_verified(
        bank_path,
        {"metadata": np.asarray(json.dumps(metadata, ensure_ascii=False))},
    )
    del extractor, bundles, rebuilt
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _global_bad_indices(bank_path: Path, metadata: dict) -> list[int]:
    if metadata.get("storage") != PATCH_MEMORY_PER_IMAGE_STORAGE:
        return []
    store = bank_path.parent / metadata["image_shard_store"]
    failed = []
    for image_index, entry in enumerate(metadata["image_shards"]):
        shard = store / str(entry["file"])
        if not shard.is_file() or _sha256_file(shard) != entry.get("sha256"):
            failed.append(image_index)
    return failed


def check_global(
    root: Path,
    *,
    repair: bool,
    device: str,
    pattern: str = GLOBAL_PATTERN,
) -> list[str]:
    failures: list[str] = []
    bank_paths = sorted(root.glob(pattern))
    if not bank_paths:
        message = f"FAIL global: no banks matched {root / pattern}"
        print(message, flush=True)
        return [message]
    for bank_path in bank_paths:
        try:
            metadata = _read_metadata(bank_path)
            bad_indices = _global_bad_indices(bank_path, metadata)
            if bad_indices and repair:
                backup = bank_path.with_name(
                    f"{bank_path.name}.manifest-before-repair-"
                    f"{dt.datetime.now():%Y%m%d-%H%M%S}"
                )
                shutil.copy2(bank_path, backup)
                for image_index in bad_indices:
                    print(
                        f"REPAIR global {bank_path.name} image_index={image_index}",
                        flush=True,
                    )
                    _repair_global_shard(
                        bank_path,
                        metadata,
                        image_index,
                        device=device,
                    )
            elif bad_indices:
                raise RuntimeError(f"damaged image shards: {bad_indices}")
            bank = PatchMemoryBank.load(bank_path)
            _remove_verified_manifest_backups(bank_path)
            print(
                f"OK global {bank_path.name}: images={len(bank.normal_images)} "
                f"patches={bank.memory_source_indices.numel()}",
                flush=True,
            )
            del bank
            gc.collect()
        except Exception as exc:
            message = f"FAIL global {bank_path.name}: {exc}"
            print(message, flush=True)
            failures.append(message)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="memory_banks/Global patch/full/mvtec_loco",
    )
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--scope",
        choices=("all", "global"),
        default="all",
    )
    parser.add_argument(
        "--global-pattern",
        default=GLOBAL_PATTERN,
        help="filename glob used inside the global directory",
    )
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    global_failures = (
        check_global(
            root / "global",
            repair=args.repair,
            device=args.device,
            pattern=args.global_pattern,
        )
        if args.scope in ("all", "global")
        else []
    )
    failures = global_failures
    if failures:
        print(f"FAILED: {len(failures)} bank(s)", flush=True)
        return 1
    if args.scope == "all":
        message = "ALL GLOBAL BANKS ARE HEALTHY"
    else:
        message = f"ALL {args.scope.upper()} BANKS ARE HEALTHY"
    print(message, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
