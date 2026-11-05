"""Deterministic same-category duplicate suppression for SAM3 masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .models import MaskInstance


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Return intersection-over-union for two equally shaped binary masks."""

    left_mask = np.asarray(left, dtype=bool)
    right_mask = np.asarray(right, dtype=bool)
    if left_mask.shape != right_mask.shape:
        raise ValueError(
            f"masks must share a shape, got {left_mask.shape} and {right_mask.shape}"
        )
    intersection = int(np.count_nonzero(left_mask & right_mask))
    union = int(np.count_nonzero(left_mask | right_mask))
    return float(intersection / union) if union else 0.0


def mask_containment(left: np.ndarray, right: np.ndarray) -> float:
    """Return intersection over the smaller mask area for nested duplicates."""

    left_mask = np.asarray(left, dtype=bool)
    right_mask = np.asarray(right, dtype=bool)
    if left_mask.shape != right_mask.shape:
        raise ValueError(
            f"masks must share a shape, got {left_mask.shape} and {right_mask.shape}"
        )
    intersection = int(np.count_nonzero(left_mask & right_mask))
    smaller_area = min(
        int(np.count_nonzero(left_mask)),
        int(np.count_nonzero(right_mask)),
    )
    return float(intersection / smaller_area) if smaller_area else 0.0


@dataclass(frozen=True)
class MaskPostprocessor:
    """Suppress duplicate masks independently within each category.

    Duplicate suppression is spatial, not semantic: two separate objects are
    retained even when they have the same category and measured length.
    """

    dedup_iou_threshold: float = 0.80
    dedup_containment_threshold: float = 0.85
    dedup_enabled: bool = True

    def __post_init__(self) -> None:
        threshold = float(self.dedup_iou_threshold)
        containment_threshold = float(self.dedup_containment_threshold)
        if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("mask deduplication IoU threshold must be in (0, 1]")
        if (
            not np.isfinite(containment_threshold)
            or not 0.0 < containment_threshold <= 1.0
        ):
            raise ValueError(
                "mask deduplication containment threshold must be in (0, 1]"
            )
        object.__setattr__(self, "dedup_iou_threshold", threshold)
        object.__setattr__(
            self, "dedup_containment_threshold", containment_threshold
        )

    def _process_category(
        self, instances: Sequence[MaskInstance]
    ) -> tuple[MaskInstance, ...]:
        prepared = list(enumerate(instances))

        if not self.dedup_enabled:
            return tuple(instance for _, instance in prepared)

        ranked = sorted(prepared, key=lambda item: (-item[1].score, item[0]))
        kept: list[tuple[int, MaskInstance]] = []
        for original_index, candidate in ranked:
            if any(
                mask_iou(candidate.mask, existing.mask)
                >= self.dedup_iou_threshold
                or mask_containment(candidate.mask, existing.mask)
                >= self.dedup_containment_threshold
                for _, existing in kept
            ):
                continue
            kept.append((original_index, candidate))
        kept.sort(key=lambda item: item[0])
        return tuple(instance for _, instance in kept)

    def process(
        self,
        categories: Sequence[str],
        instances: Mapping[str, Sequence[MaskInstance]],
    ) -> dict[str, tuple[MaskInstance, ...]]:
        """Process categories independently so cross-category masks never merge."""

        return {
            category: self._process_category(instances.get(category, ()))
            for category in categories
        }
