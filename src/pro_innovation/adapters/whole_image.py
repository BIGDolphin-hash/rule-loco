"""Mask adapter treating the complete image as one object per requested category."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ..models import MaskInstance


class WholeImageSegmenter:
    """Return one deterministic full-image object without loading SAM3."""

    cache_hits = 0
    cache_misses = 0
    cache_writes = 0

    @staticmethod
    def _size(image: Any) -> tuple[int, int]:
        if isinstance(image, Image.Image):
            return image.size
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                return opened.size
        array = np.asarray(image)
        if array.ndim < 2:
            raise ValueError(f"expected an image array, got shape {array.shape}")
        return int(array.shape[1]), int(array.shape[0])

    def segment(
        self, image: Any, categories: Sequence[str]
    ) -> dict[str, tuple[MaskInstance, ...]]:
        ordered = tuple(dict.fromkeys(str(category) for category in categories))
        if not ordered:
            raise ValueError("whole-image segmentation requires a category")
        width, height = self._size(image)
        if width <= 0 or height <= 0:
            raise ValueError(f"image dimensions must be positive, got {width}x{height}")
        mask = np.ones((height, width), dtype=np.bool_)
        return {
            category: (
                MaskInstance(
                    mask=mask,
                    score=1.0,
                    box=(0.0, 0.0, float(width), float(height)),
                ),
            )
            for category in ordered
        }
