"""Deterministic cleanup and duplicate suppression for SAM3 masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .models import MaskInstance


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Return the largest 8-connected foreground component.

    SAM3 masks occasionally contain small, detached regions from another
    fastener or a reflection on the bag.  Keeping the main connected body
    prevents those remote pixels from stretching geometric measurements.
    """

    foreground = np.asarray(mask, dtype=bool)
    if foreground.ndim != 2:
        raise ValueError(f"mask must be two-dimensional, got shape {foreground.shape}")

    height, width = foreground.shape
    visited = np.zeros_like(foreground, dtype=bool)
    best_component: list[int] = []

    for y_value, x_value in np.argwhere(foreground):
        y = int(y_value)
        x = int(x_value)
        if visited[y, x]:
            continue

        visited[y, x] = True
        stack = [y * width + x]
        component: list[int] = []
        while stack:
            flat_index = stack.pop()
            current_y, current_x = divmod(flat_index, width)
            component.append(flat_index)
            for neighbor_y in range(max(0, current_y - 1), min(height, current_y + 2)):
                for neighbor_x in range(max(0, current_x - 1), min(width, current_x + 2)):
                    if (
                        foreground[neighbor_y, neighbor_x]
                        and not visited[neighbor_y, neighbor_x]
                    ):
                        visited[neighbor_y, neighbor_x] = True
                        stack.append(neighbor_y * width + neighbor_x)

        if len(component) > len(best_component):
            best_component = component

    cleaned = np.zeros_like(foreground, dtype=bool)
    if best_component:
        flat = cleaned.reshape(-1)
        flat[np.asarray(best_component, dtype=np.int64)] = True
    return cleaned


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


def _mask_box(mask: np.ndarray) -> tuple[float, float, float, float]:
    rows, columns = np.nonzero(mask)
    return (
        float(columns.min()),
        float(rows.min()),
        float(columns.max() + 1),
        float(rows.max() + 1),
    )


@dataclass(frozen=True)
class MaskPostprocessor:
    """Clean each candidate, then suppress duplicate masks within its category.

    Duplicate suppression is spatial, not semantic: two separate objects are
    retained even when they have the same category and measured length.
    """

    dedup_iou_threshold: float = 0.80
    dedup_containment_threshold: float = 0.85
    minimum_area_pixels: int = 1
    cleanup_enabled: bool = True
    dedup_enabled: bool = True

    def __post_init__(self) -> None:
        threshold = float(self.dedup_iou_threshold)
        containment_threshold = float(self.dedup_containment_threshold)
        minimum_area = int(self.minimum_area_pixels)
        if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("mask deduplication IoU threshold must be in (0, 1]")
        if (
            not np.isfinite(containment_threshold)
            or not 0.0 < containment_threshold <= 1.0
        ):
            raise ValueError(
                "mask deduplication containment threshold must be in (0, 1]"
            )
        if minimum_area < 1:
            raise ValueError("minimum mask area must be at least one pixel")
        object.__setattr__(self, "dedup_iou_threshold", threshold)
        object.__setattr__(
            self, "dedup_containment_threshold", containment_threshold
        )
        object.__setattr__(self, "minimum_area_pixels", minimum_area)

    def _clean(self, instance: MaskInstance) -> MaskInstance | None:
        cleaned = largest_connected_component(instance.mask)
        if int(np.count_nonzero(cleaned)) < self.minimum_area_pixels:
            return None
        return MaskInstance(
            mask=cleaned,
            score=instance.score,
            box=_mask_box(cleaned),
        )

    def _process_category(
        self, instances: Sequence[MaskInstance]
    ) -> tuple[MaskInstance, ...]:
        prepared: list[tuple[int, MaskInstance]] = []
        for index, instance in enumerate(instances):
            candidate = self._clean(instance) if self.cleanup_enabled else instance
            if candidate is not None:
                prepared.append((index, candidate))

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
