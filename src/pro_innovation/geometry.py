"""NumPy geometry used by LENGTH and AREA tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .models import MaskInstance


@dataclass(frozen=True)
class RankGroup:
    rank: int
    representative: float
    count: int
    values: tuple[float, ...]


def mask_area(instance: MaskInstance) -> float:
    """Return foreground area in pixels."""

    return float(np.count_nonzero(instance.mask))


def mask_length(instance: MaskInstance) -> float:
    """Estimate orientation-invariant major-axis extent in pixels.

    V1 projects foreground coordinates onto their first principal component.
    This is more stable for rotated parts than an axis-aligned bounding-box side.
    """

    coordinates = np.argwhere(instance.mask)
    if len(coordinates) == 1:
        return 1.0
    xy = coordinates[:, ::-1].astype(np.float64)
    centered = xy - xy.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    major_axis = eigenvectors[:, -1]
    projected = centered @ major_axis
    return float(projected.max() - projected.min() + 1.0)


def group_ranked_values(
    values: Iterable[float], *, relative_tolerance: float = 0.10
) -> tuple[RankGroup, ...]:
    """Group similar values, then rank groups from largest to smallest."""

    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative")
    sorted_values = sorted((float(value) for value in values), reverse=True)
    clusters: list[list[float]] = []
    for value in sorted_values:
        if not clusters:
            clusters.append([value])
            continue
        representative = float(np.mean(clusters[-1]))
        scale = max(abs(value), abs(representative), 1.0)
        if abs(value - representative) / scale <= relative_tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])

    return tuple(
        RankGroup(
            rank=index,
            representative=float(np.mean(cluster)),
            count=len(cluster),
            values=tuple(cluster),
        )
        for index, cluster in enumerate(clusters, start=1)
    )


def ranked_measurements(
    instances: Iterable[MaskInstance],
    *,
    measure: str,
    relative_tolerance: float = 0.10,
) -> tuple[RankGroup, ...]:
    if measure == "length":
        values = (mask_length(instance) for instance in instances)
    elif measure == "area":
        values = (mask_area(instance) for instance in instances)
    else:
        raise ValueError(f"unsupported measure: {measure}")
    return group_ranked_values(values, relative_tolerance=relative_tolerance)

