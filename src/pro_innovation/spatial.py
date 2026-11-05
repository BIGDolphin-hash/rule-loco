"""Geometric predicates for SPATIAL_COMBINATION."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .errors import ExecutionError
from .models import MaskInstance

SELF_SYMMETRY_THRESHOLD = 0.80


@dataclass(frozen=True)
class SpatialConfig:
    containment_threshold: float = 0.95
    middle_min_fraction: float = 0.40
    middle_max_fraction: float = 0.60
    left_max_fraction: float = 0.30
    right_min_fraction: float = 0.70
    top_max_fraction: float = 0.30
    lower_min_fraction: float = 0.70
    center_x_tolerance_fraction: float = 0.08
    center_y_tolerance_fraction: float = 0.08


@dataclass(frozen=True)
class _MaskGeometry:
    """Reusable exact geometry derived from one binary instance mask."""

    shape: tuple[int, ...]
    rows: np.ndarray
    columns: np.ndarray
    bounds: tuple[float, float, float, float]


def _mask_geometry(mask: np.ndarray) -> _MaskGeometry:
    prepared = np.asarray(mask, dtype=bool)
    rows, columns = np.nonzero(prepared)
    bounds = (
        float(columns.min()),
        float(rows.min()),
        float(columns.max() + 1),
        float(rows.max() + 1),
    )
    return _MaskGeometry(prepared.shape, rows, columns, bounds)


def _geometry_containment_fraction(
    inner: _MaskGeometry,
    container: _MaskGeometry,
) -> float:
    if inner.shape != container.shape:
        raise ExecutionError(
            "spatial masks must share a shape, "
            f"got {inner.shape} and {container.shape}"
        )
    left, top, right, bottom = container.bounds
    inside = (
        (inner.columns >= left)
        & (inner.columns < right)
        & (inner.rows >= top)
        & (inner.rows < bottom)
    )
    return float(np.count_nonzero(inside) / max(len(inner.rows), 1))


def _center(mask: np.ndarray) -> tuple[float, float]:
    rows, columns = np.nonzero(mask)
    return float(columns.mean()), float(rows.mean())


def _same_shape(left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape:
        raise ExecutionError(
            f"spatial masks must share a shape, got {left.shape} and {right.shape}"
        )


def _bounds(mask: np.ndarray) -> tuple[float, float, float, float]:
    rows, columns = np.nonzero(mask)
    return (
        float(columns.min()),
        float(rows.min()),
        float(columns.max() + 1),
        float(rows.max() + 1),
    )


def _relative_center(
    subject: np.ndarray, object_: np.ndarray
) -> tuple[float, float]:
    subject_x, subject_y = _center(subject)
    left, top, right, bottom = _bounds(object_)
    width = max(right - left, 1.0)
    height = max(bottom - top, 1.0)
    return (subject_x - left) / width, (subject_y - top) / height


def position_containment_fraction(
    inner: np.ndarray, container: np.ndarray
) -> float:
    """Return the fraction of inner-mask pixels inside the container bounds.

    Semantic masks for a container and its contents are normally disjoint.  A
    raw mask intersection therefore cannot represent spatial containment.  As
    with the center relation, the container mask supplies an object-relative
    coordinate frame; its bounds are treated as a filled spatial region.
    """

    return _geometry_containment_fraction(
        _mask_geometry(inner),
        _mask_geometry(container),
    )


def _relative_center_in_object_frame(
    subject: np.ndarray, object_: np.ndarray
) -> tuple[float, float]:
    """Locate ``subject`` in the scale- and rotation-normalized object frame."""

    rows, columns = np.nonzero(object_)
    points = np.column_stack((columns, rows)).astype(float)
    if len(points) < 3:
        return _relative_center(subject, object_)

    origin = points.mean(axis=0)
    centered = points - origin
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = float(eigenvalues[-1])
    if largest <= 1e-12:
        return _relative_center(subject, object_)

    # PCA has no stable orientation for near-isotropic masks. In that case the
    # image axes are already the least arbitrary object frame.
    if float(eigenvalues[-1] - eigenvalues[0]) / largest < 0.05:
        return _relative_center(subject, object_)

    axes = eigenvectors[:, ::-1]
    object_projection = centered @ axes
    subject_projection = (np.asarray(_center(subject)) - origin) @ axes
    minimum = object_projection.min(axis=0)
    maximum = object_projection.max(axis=0)
    extent = np.maximum(maximum - minimum, 1.0)
    relative = (subject_projection - minimum) / extent
    return float(relative[0]), float(relative[1])


def _self_symmetry_score(mask: np.ndarray, relation: str) -> float:
    """Return Dice overlap with the mask mirrored inside its own bounds."""

    rows, columns = np.nonzero(mask)
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    cropped = mask[top:bottom, left:right]
    if relation == "self_left_right_symmetric":
        mirrored = np.fliplr(cropped)
    elif relation == "self_top_bottom_symmetric":
        mirrored = np.flipud(cropped)
    else:
        raise ExecutionError(f"unsupported self-symmetry relation: {relation}")
    intersection = int(np.count_nonzero(cropped & mirrored))
    total = int(np.count_nonzero(cropped)) + int(np.count_nonzero(mirrored))
    return float(2.0 * intersection / max(total, 1))


def _canonical_relation(relation: str) -> str:
    aliases = {
        "left": "left_of",
        "right": "right_of",
        "top": "above",
        "bottom": "below",
        "upper_left": "upper_left_of",
        "top_left": "upper_left_of",
        "upper_right": "upper_right_of",
        "top_right": "upper_right_of",
        "lower_left": "lower_left_of",
        "bottom_left": "lower_left_of",
        "lower_right": "lower_right_of",
        "bottom_right": "lower_right_of",
        "left_right_symmetric": "self_left_right_symmetric",
        "top_bottom_symmetric": "self_top_bottom_symmetric",
        "contain": "contains",
        "middle": "at_middle",
        "lower": "at_lower",
        "center": "at_center",
        "centered": "at_center",
        "left_region_of": "at_left",
        "right_region_of": "at_right",
        "top_region_of": "at_top",
        "middle_region_of": "at_middle",
        "lower_region_of": "at_lower",
        "centered_in": "at_center",
    }
    lowered = relation.strip().lower().replace("-", "_").replace(" ", "_")
    if lowered.startswith("spatial."):
        lowered = lowered.split(".", 1)[1]
    return aliases.get(lowered, lowered)


def _evaluate_pair(
    subject: MaskInstance,
    object_: MaskInstance,
    relation: str,
    config: SpatialConfig,
) -> bool:
    left = subject.mask
    right = object_.mask
    _same_shape(left, right)
    relation = _canonical_relation(relation)
    if relation == "inside":
        return (
            position_containment_fraction(left, right)
            >= config.containment_threshold
        )
    if relation == "contains":
        return (
            position_containment_fraction(right, left)
            >= config.containment_threshold
        )

    subject_center = _center(left)
    object_center = _center(right)

    if relation == "left_of":
        return subject_center[0] < object_center[0]
    if relation == "right_of":
        return subject_center[0] > object_center[0]
    if relation == "above":
        return subject_center[1] < object_center[1]
    if relation == "below":
        return subject_center[1] > object_center[1]
    if relation == "upper_left_of":
        return (
            subject_center[0] < object_center[0]
            and subject_center[1] < object_center[1]
        )
    if relation == "upper_right_of":
        return (
            subject_center[0] > object_center[0]
            and subject_center[1] < object_center[1]
        )
    if relation == "lower_left_of":
        return (
            subject_center[0] < object_center[0]
            and subject_center[1] > object_center[1]
        )
    if relation == "lower_right_of":
        return (
            subject_center[0] > object_center[0]
            and subject_center[1] > object_center[1]
        )

    if relation == "at_middle":
        relative_x, relative_y = _relative_center_in_object_frame(left, right)
        return (
            config.middle_min_fraction
            <= relative_x
            <= config.middle_max_fraction
            and config.middle_min_fraction
            <= relative_y
            <= config.middle_max_fraction
        )

    relative_x, relative_y = _relative_center(left, right)
    if relation == "at_left":
        return relative_x <= config.left_max_fraction
    if relation == "at_right":
        return relative_x >= config.right_min_fraction
    if relation == "at_top":
        return relative_y <= config.top_max_fraction
    if relation == "at_lower":
        return relative_y >= config.lower_min_fraction
    if relation == "at_center":
        return (
            abs(relative_x - 0.5) <= config.center_x_tolerance_fraction
            and abs(relative_y - 0.5) <= config.center_y_tolerance_fraction
        )

    raise ExecutionError(f"unsupported spatial relation: {relation}")


def evaluate_spatial_relation(
    subjects: Sequence[MaskInstance],
    objects: Sequence[MaskInstance],
    relation: str,
    *,
    config: SpatialConfig | None = None,
) -> bool:
    """Return existential pair semantics for the V1 boolean relation."""

    return any(
        evaluate_spatial_relation_by_subject(
            subjects,
            objects,
            relation,
            config=config,
        )
    )


def evaluate_spatial_relation_by_subject(
    subjects: Sequence[MaskInstance],
    objects: Sequence[MaskInstance],
    relation: str,
    *,
    config: SpatialConfig | None = None,
) -> tuple[bool, ...]:
    """Return whether each subject has at least one matching object."""

    relation = _canonical_relation(relation)
    settings = config or SpatialConfig()
    if relation in {
        "self_left_right_symmetric",
        "self_top_bottom_symmetric",
    }:
        return tuple(
            _self_symmetry_score(subject.mask, relation)
            >= SELF_SYMMETRY_THRESHOLD
            for subject in subjects
        )
    return tuple(
        any(
            _evaluate_pair(subject, object_, relation, settings)
            for object_ in objects
        )
        for subject in subjects
    )
