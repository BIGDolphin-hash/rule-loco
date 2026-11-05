#!/usr/bin/env python3
"""Build one isolated category-only semantic bank for GCDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pro_innovation.adapters import DinoV2PatchFeatureExtractor, Sam3Segmenter
from pro_innovation.memory_modules.gcda import (
    CategoryOnlyCompositionMemoryBank,
    category_only_embedding,
)
from pro_innovation.image_validation import filter_decodable_images
from pro_innovation.mask_processing import MaskPostprocessor
from pro_innovation.sam3_calibration import load_sam3_threshold_policy
from pro_innovation.templates import collect_categories, parse_template


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Build an isolated [f1;...;fC] category-only memory for GCDA")
    )
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--normal-dir", required=True)
    parser.add_argument("--generic", required=True)
    parser.add_argument("--threshold-policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam3-repo", default="models/SAM3")
    parser.add_argument("--sam3-checkpoint", default="weights/sam3/sam3.pt")
    parser.add_argument(
        "--dino-checkpoint",
        default="weights/DINOV2/dinov2_vitl14_reg4_pretrain.pth",
    )
    parser.add_argument("--sam3-threshold", type=float, default=0.40)
    parser.add_argument("--sam3-prompt", default="{category}")
    parser.add_argument(
        "--sam3-cache-root", default="cache/sam3"
    )
    parser.add_argument(
        "--sam3-cache-mode",
        choices=("off", "read", "read-write", "refresh"),
        default="read-write",
    )
    parser.add_argument("--mask-dedup-iou", type=float, default=0.80)
    parser.add_argument("--mask-dedup-containment", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    normal_dir = Path(args.normal_dir).expanduser().resolve()
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = tuple(
        path
        for path in sorted(normal_dir.iterdir())
        if path.is_file() and path.suffix.lower() in image_suffixes
    )
    images = filter_decodable_images(images)
    if len(images) < 2:
        raise ValueError("category-only memory requires at least two normal images")

    generic = parse_template(
        Path(args.generic).read_text(encoding="utf-8"),
        allow_missing_values=True,
    )
    categories = collect_categories(generic)
    policy = load_sam3_threshold_policy(
        args.threshold_policy,
        expected_class_name=args.class_name,
    )
    missing_thresholds = sorted(set(categories) - set(policy.selected_thresholds))
    if missing_thresholds:
        raise ValueError(
            "threshold policy is missing current template categories: "
            + ", ".join(missing_thresholds)
        )

    segmenter = Sam3Segmenter(
        repo_path=args.sam3_repo,
        checkpoint_path=args.sam3_checkpoint,
        device=args.device,
        confidence_threshold=args.sam3_threshold,
        category_thresholds={
            category: float(policy.selected_thresholds[category])
            for category in categories
        },
        prompt_template=args.sam3_prompt,
        cache_root=args.sam3_cache_root,
        cache_mode=args.sam3_cache_mode,
    )
    processor = MaskPostprocessor(
        dedup_iou_threshold=args.mask_dedup_iou,
        dedup_containment_threshold=args.mask_dedup_containment,
        dedup_enabled=True,
    )
    extractor = DinoV2PatchFeatureExtractor(
        checkpoint_path=args.dino_checkpoint,
        device=args.device,
    )

    embeddings: list[torch.Tensor] = []
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
        for bundle, pool in zip(bundles, pools):
            grid = tuple(bundle.feature_grid)
            if feature_grid is None:
                feature_grid = grid
            elif feature_grid != grid:
                raise RuntimeError("DINOv2 feature grid changed during construction")
            embeddings.append(category_only_embedding(bundle, pool, categories)[0])
        completed = min(start + len(batch), total)
        print(
            f"[category-only] {args.class_name}: "
            f"{completed}/{total} ({completed / total * 100:.1f}%)",
            flush=True,
        )

    if feature_grid is None:
        raise RuntimeError("no category-only embeddings were produced")
    bank = CategoryOnlyCompositionMemoryBank.build_from_embeddings(
        class_name=args.class_name,
        categories=categories,
        embeddings=torch.stack(embeddings, dim=0),
        normal_images=images,
        checkpoint_path=args.dino_checkpoint,
        feature_grid=feature_grid,
    )
    output = bank.save(args.output)
    print(
        json.dumps(
            {
                "bank": str(output),
                "class_name": bank.class_name,
                "categories": list(bank.categories),
                "normal_count": len(bank.normal_images),
                "loo_max_distance": bank.loo_max_distance,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
