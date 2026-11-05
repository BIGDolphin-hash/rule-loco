"""GPSTM-owned texture descriptor construction.

The implementation is intentionally local to GPSTM.  GPO-JM has its own copy
because the two modules must be editable and testable without importing one
another.
"""

from __future__ import annotations

from math import ceil, isfinite
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from ...errors import MemoryBankError
from ...models import MaskInstance

BACKGROUND_BLOCK = "__background__"
DEFAULT_TOP_FRACTION = 0.10
STANDARDIZATION_EPSILON = 1e-10


def _mask_on_feature_grid(
    instance: MaskInstance,
    feature_grid: tuple[int, int],
) -> torch.Tensor:
    mask = np.asarray(instance.mask, dtype=np.float32)
    tensor = torch.from_numpy(mask)[None, None]
    selected = (
        functional.interpolate(tensor, size=feature_grid, mode="nearest")[0, 0].reshape(
            -1
        )
        >= 0.5
    )
    if selected.any() or not mask.any():
        return selected
    return (
        functional.interpolate(tensor, size=feature_grid, mode="area")[0, 0].reshape(-1)
        > 0.0
    )


def _three_texture_statistics(
    values: torch.Tensor,
    *,
    top_fraction: float,
) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("GPSTM statistics require a flat anomaly map")
    if values.numel() == 0:
        return torch.zeros(3, dtype=torch.float32)
    count = max(1, min(values.numel(), ceil(values.numel() * top_fraction)))
    top_mean = torch.topk(values, k=count, largest=True).values.mean()
    return torch.stack((values.mean(), top_mean, values.max())).float().cpu()


def global_patch_texture_signature(
    anomaly_map: torch.Tensor,
    mask_pool: Mapping[str, Sequence[MaskInstance]],
    categories: Sequence[str],
    *,
    feature_grid: tuple[int, int],
    top_fraction: float = DEFAULT_TOP_FRACTION,
) -> tuple[torch.Tensor, dict[str, tuple[float, float, float]]]:
    """Return ordered GPSTM statistics for categories plus background."""

    ordered = tuple(str(category) for category in categories)
    if not ordered or len(set(ordered)) != len(ordered):
        raise MemoryBankError("GPSTM categories must be unique and non-empty")
    if not isfinite(float(top_fraction)) or not 0.0 < float(top_fraction) <= 1.0:
        raise ValueError("GPSTM top fraction must be in (0, 1]")
    values = anomaly_map.detach().float().cpu().reshape(-1)
    positions = int(feature_grid[0]) * int(feature_grid[1])
    if values.numel() != positions or not torch.isfinite(values).all():
        raise MemoryBankError("GPSTM anomaly map does not match feature grid")

    union = torch.zeros(positions, dtype=torch.bool)
    blocks: list[torch.Tensor] = []
    labels = (*ordered, BACKGROUND_BLOCK)
    for category in ordered:
        selected = torch.zeros(positions, dtype=torch.bool)
        for instance in mask_pool.get(category, ()):
            selected |= _mask_on_feature_grid(instance, feature_grid)
        union |= selected
        blocks.append(
            _three_texture_statistics(
                values[selected], top_fraction=float(top_fraction)
            )
        )
    blocks.append(
        _three_texture_statistics(values[~union], top_fraction=float(top_fraction))
    )
    signature = torch.stack(blocks, dim=0)
    payload = {
        label: tuple(float(value) for value in signature[index].tolist())
        for index, label in enumerate(labels)
    }
    return signature, payload
