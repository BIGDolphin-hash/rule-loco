#!/usr/bin/env python3
"""Build one isolated Category-Aligned Semantic–Texture Joint Memory bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pro_innovation.adapters import (
    DinoV2PatchFeatureExtractor,
    Sam3Segmenter,
    WholeImageSegmenter,
)
from pro_innovation.memory_modules.castjm import (
    DEFAULT_TEXTURE_WEIGHT,
    DEFAULT_TOP_FRACTION,
    CASTJMGlobalPatchBank,
    CASTJMMemoryBank,
    category_only_embedding,
    global_patch_texture_signature,
)
from pro_innovation.mask_processing import MaskPostprocessor
from pro_innovation.sam3_thresholds import load_sam3_threshold_policy
from pro_innovation.templates import collect_categories, parse_template


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build isolated Category-Aligned Semantic–Texture Joint Memory"
    )
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--normal-dir", required=True)
    parser.add_argument("--generic")
    parser.add_argument("--threshold-policy")
    parser.add_argument("--global-bank", required=True)
    parser.add_argument("--output", required=True)
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--texture-weight", type=float, default=DEFAULT_TEXTURE_WEIGHT)
    parser.add_argument("--top-fraction", type=float, default=DEFAULT_TOP_FRACTION)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    normal_dir = Path(args.normal_dir).expanduser().resolve()
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
        categories = (args.class_name,)
        missing: tuple[str, ...] = ()
        print(
            "[CAST-JM] no threshold policy; treating the complete image as "
            f"one {args.class_name!r} object without SAM3",
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
    images = tuple(Path(value) for value in global_bank.normal_images)
    if len(images) < 2:
        raise ValueError("joint memory requires at least two normal images")
    if any(image.parent != normal_dir for image in images):
        raise ValueError(
            "Global Patch bank normal-image paths do not match --normal-dir"
        )
    missing_images = [str(image) for image in images if not image.is_file()]
    if missing_images:
        raise FileNotFoundError("missing normal images: " + ", ".join(missing_images))

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
        WholeImageSegmenter()
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
    processor = MaskPostprocessor(
        dedup_iou_threshold=args.mask_dedup_iou,
        dedup_containment_threshold=args.mask_dedup_containment,
        dedup_enabled=True,
    )
    extractor = DinoV2PatchFeatureExtractor(
        checkpoint_path=args.dino_checkpoint, device=args.device
    )

    category_rows: list[torch.Tensor] = []
    texture_rows: list[torch.Tensor] = []
    feature_grid: tuple[int, int] | None = None
    total = len(images)
    for start in range(0, total, args.batch_size):
        batch = images[start : start + args.batch_size]
        pools = [
            processor.process(categories, segmenter.segment(image, categories))
            for image in batch
        ]
        bundles = extractor.extract_batch(batch, batch_size=args.batch_size)
        if len(bundles) != len(batch):
            raise RuntimeError("DINOv2 returned an unexpected batch size")
        for offset, (bundle, pool) in enumerate(zip(bundles, pools)):
            image_index = start + offset
            grid = tuple(bundle.feature_grid)
            if feature_grid is None:
                feature_grid = grid
            elif feature_grid != grid:
                raise RuntimeError("DINOv2 feature grid changed during construction")
            vector, slices = category_only_embedding(bundle, pool, categories)
            category_rows.append(
                torch.stack(
                    [vector[slices[category]] for category in categories], dim=0
                )
            )
            anomaly_map = global_bank.anomaly_map(
                bundle,
                device=args.device,
                query_batch_size=args.query_batch_size,
                excluded_reference=image_index,
            )
            texture_rows.append(
                global_patch_texture_signature(
                    anomaly_map,
                    pool,
                    categories,
                    feature_grid=grid,
                    top_fraction=args.top_fraction,
                )[0]
            )
        completed = min(start + len(batch), total)
        print(
            f"[CAST-JM] {args.class_name}: {completed}/{total} "
            f"({completed / total * 100:.1f}%)",
            flush=True,
        )

    if feature_grid is None:
        raise RuntimeError("no joint-memory features were produced")
    bank = CASTJMMemoryBank.build_from_components(
        class_name=args.class_name,
        categories=categories,
        category_features=torch.stack(category_rows, dim=0),
        texture_features=torch.stack(texture_rows, dim=0),
        normal_images=images,
        checkpoint_path=args.dino_checkpoint,
        global_patch_bank_path=args.global_bank,
        feature_grid=feature_grid,
        texture_weight=args.texture_weight,
        top_fraction=args.top_fraction,
    )
    output = bank.save(args.output)
    print(
        json.dumps(
            {
                "module": "CAST-JM",
                "bank": str(output),
                "class_name": bank.class_name,
                "categories": list(bank.categories),
                "normal_count": len(bank.normal_images),
                "global_patch_bank": bank.global_patch_bank_path,
                "texture_weight": bank.texture_weight,
                "top_fraction": bank.top_fraction,
                "loo_max_distance": bank.loo_max_distance,
                "segmentation_mode": (
                    "whole_image_single_object"
                    if whole_image_mode
                    else "sam3_category_masks"
                ),
                "generic_template": (
                    str(generic_path)
                    if generic_path is not None and generic_path.is_file()
                    else None
                ),
                "threshold_policy": (
                    str(threshold_policy_path) if policy is not None else None
                ),
                "missing_policy_categories_using_global_fallback": list(missing),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
