"""CASTJM-owned ordered semantic descriptor construction.

CASTJM owns this implementation so its semantic descriptor remains independent
of other memory modules.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from ...adapters.patch_dinov2 import PATCH_MATCHING_LAYERS, PatchFeatureBundle
from ...errors import MemoryBankError
from ...models import MaskInstance


def _resize_category_mask(
    instance: MaskInstance,
    feature_grid: tuple[int, int],
) -> torch.Tensor:
    mask = np.asarray(instance.mask, dtype=bool)
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    resized = functional.interpolate(tensor, size=feature_grid, mode="nearest")[0, 0]
    selected = resized.reshape(-1) >= 0.5
    if selected.any() or not mask.any():
        return selected
    coverage = functional.interpolate(tensor, size=feature_grid, mode="area")[0, 0]
    return coverage.reshape(-1) > 0.0


def category_only_embedding(
    features: PatchFeatureBundle,
    mask_pool: Mapping[str, Sequence[MaskInstance]],
    categories: Sequence[str],
) -> tuple[torch.Tensor, dict[str, slice]]:
    """Build CASTJM's fixed-order semantic vector ``[f1; ...; fC]``."""

    ordered = tuple(str(category) for category in categories)
    if not ordered or len(set(ordered)) != len(ordered):
        raise MemoryBankError("CASTJM categories must be unique and non-empty")
    if tuple(sorted(features.layer_features)) != PATCH_MATCHING_LAYERS:
        raise MemoryBankError("CASTJM features are missing required DINO layers")
    positions = features.feature_grid[0] * features.feature_grid[1]
    dimensions = {int(value.shape[-1]) for value in features.layer_features.values()}
    if len(dimensions) != 1:
        raise MemoryBankError("CASTJM DINO layers use different dimensions")
    dimension = next(iter(dimensions))
    for layer in PATCH_MATCHING_LAYERS:
        values = features.layer_features[layer]
        if values.ndim != 2 or tuple(values.shape) != (positions, dimension):
            raise MemoryBankError(f"invalid CASTJM Block {layer} feature shape")

    chunks: list[torch.Tensor] = []
    slices: dict[str, slice] = {}
    cursor = 0
    for category in ordered:
        instance_vectors: list[torch.Tensor] = []
        instance_areas: list[float] = []
        for instance in mask_pool.get(category, ()):
            selected = _resize_category_mask(instance, features.feature_grid)
            if not selected.any():
                raise MemoryBankError(
                    f"CASTJM instance for {category!r} disappears on the DINO grid"
                )
            layer_vectors = []
            for layer in PATCH_MATCHING_LAYERS:
                values = features.layer_features[layer].detach().float().cpu()
                pooled = values[selected].mean(dim=0)
                layer_vectors.append(functional.normalize(pooled, dim=0, eps=1e-12))
            instance_vector = torch.stack(layer_vectors, dim=0).mean(dim=0)
            instance_vectors.append(
                functional.normalize(instance_vector, dim=0, eps=1e-12)
            )
            instance_areas.append(float(np.count_nonzero(instance.mask)))

        if instance_vectors:
            weights = torch.tensor(instance_areas, dtype=torch.float32)
            weights = weights / weights.sum()
            chunk = torch.sum(
                torch.stack(instance_vectors, dim=0) * weights[:, None], dim=0
            )
            chunk = functional.normalize(chunk, dim=0, eps=1e-12)
        else:
            chunk = torch.zeros(dimension, dtype=torch.float32)
        chunks.append(chunk)
        slices[category] = slice(cursor, cursor + int(chunk.numel()))
        cursor += int(chunk.numel())
    return torch.cat(chunks, dim=0), slices
