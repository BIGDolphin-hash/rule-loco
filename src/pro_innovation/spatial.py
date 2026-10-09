"""Geometric predicates for SPATIAL_COMBINATION."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, hypot
from typing import Sequence

import numpy as np

from .errors import ExecutionError
from .models import MaskInstance


@dataclass(frozen=True)
class SpatialConfig:
    containment_threshold: float = 0.95
    near_diagonal_fraction: float = 0.15
    far_diagonal_fraction: float = 0.50
    direction_transition_fraction: float = 0.05
    distance_transition_fraction: float = 0.05
    containment_transition: float = 0.05
    middle_min_fraction: float = 0.40
    middle_max_fraction: float = 0.60
    lower_min_fraction: float = 0.70
    center_x_tolerance_fraction: float = 0.07
    center_y_tolerance_fraction: float = 0.07
    position_transition_fraction: float = 0.05


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

    _same_shape(inner, container)
    rows, columns = np.nonzero(inner)
    left, top, right, bottom = _bounds(container)
    inside = (
        (columns >= left)
        & (columns < right)
        & (rows >= top)
        & (rows < bottom)
    )
    return float(np.count_nonzero(inside) / max(len(rows), 1))


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


def _dilate_8(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for dy in range(3):
        for dx in range(3):
            result |= padded[dy : dy + height, dx : dx + width]
    return result


def _canonical_relation(relation: str) -> str:
    aliases = {
        "left": "left_of",
        "right": "right_of",
        "top": "above",
        "bottom": "below",
        "contact": "touching",
        "contain": "contains",
        "centered": "center",
        "middle_region_of": "middle",
        "lower_region_of": "lower",
        "centered_in": "center",
    }
    lowered = relation.lower()
    if lowered.startswith("spatial."):
        lowered = lowered.split(".", 1)[1]
    return aliases.get(lowered, lowered)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    exponential = exp(value)
    return exponential / (1.0 + exponential)


def _evaluate_pair(
    subject: MaskInstance,
    object_: MaskInstance,
    relation: str,
    config: SpatialConfig,
) -> bool:
    left = subject.mask
    right = object_.mask
    _same_shape(left, right)
    subject_center = _center(left)
    object_center = _center(right)
    relation = _canonical_relation(relation)

    if relation == "left_of":
        return subject_center[0] < object_center[0]
    if relation == "right_of":
        return subject_center[0] > object_center[0]
    if relation == "above":
        return subject_center[1] < object_center[1]
    if relation == "below":
        return subject_center[1] > object_center[1]

    if relation == "middle":
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
    if relation == "lower":
        return relative_y >= config.lower_min_fraction
    if relation == "center":
        return (
            abs(relative_x - 0.5) <= config.center_x_tolerance_fraction
            and abs(relative_y - 0.5) <= config.center_y_tolerance_fraction
        )

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

    intersection = int(np.count_nonzero(left & right))
    if relation == "overlap":
        return intersection > 0
    if relation == "disjoint":
        return intersection == 0
    if relation == "touching":
        return bool(np.any(_dilate_8(left) & right))
    distance = hypot(
        subject_center[0] - object_center[0],
        subject_center[1] - object_center[1],
    )
    diagonal = hypot(left.shape[0], left.shape[1])
    normalized_distance = distance / max(diagonal, 1.0)
    if relation == "near":
        return normalized_distance <= config.near_diagonal_fraction
    if relation == "far":
        return normalized_distance >= config.far_diagonal_fraction
    raise ExecutionError(f"unsupported spatial relation: {relation}")


def _pair_relation_support(
    subject: MaskInstance,
    object_: MaskInstance,
    relation: str,
    config: SpatialConfig,
) -> float:
    """Return continuous support for one geometric relation predicate."""

    left = subject.mask
    right = object_.mask
    _same_shape(left, right)
    subject_center = _center(left)
    object_center = _center(right)
    relation = _canonical_relation(relation)
    diagonal = max(hypot(left.shape[0], left.shape[1]), 1.0)

    if relation in {"left_of", "right_of", "above", "below"}:
        signed_difference = {
            "left_of": object_center[0] - subject_center[0],
            "right_of": subject_center[0] - object_center[0],
            "above": object_center[1] - subject_center[1],
            "below": subject_center[1] - object_center[1],
        }[relation]
        scale = max(config.direction_transition_fraction, 1e-6)
        return _sigmoid((signed_difference / diagonal) / scale)

    position_scale = max(config.position_transition_fraction, 1e-6)
    if relation == "middle":
        relative_x, relative_y = _relative_center_in_object_frame(left, right)
        after_left = _sigmoid(
            (relative_x - config.middle_min_fraction) / position_scale
        )
        before_right = _sigmoid(
            (config.middle_max_fraction - relative_x) / position_scale
        )
        after_top = _sigmoid(
            (relative_y - config.middle_min_fraction) / position_scale
        )
        before_bottom = _sigmoid(
            (config.middle_max_fraction - relative_y) / position_scale
        )
        return min(after_left, before_right, after_top, before_bottom)

    relative_x, relative_y = _relative_center(left, right)
    if relation == "lower":
        return _sigmoid(
            (relative_y - config.lower_min_fraction) / position_scale
        )
    if relation == "center":
        horizontal = _sigmoid(
            (
                config.center_x_tolerance_fraction
                - abs(relative_x - 0.5)
            )
            / position_scale
        )
        vertical = _sigmoid(
            (
                config.center_y_tolerance_fraction
                - abs(relative_y - 0.5)
            )
            / position_scale
        )
        return min(horizontal, vertical)

    if relation in {"inside", "contains"}:
        contained_fraction = (
            position_containment_fraction(left, right)
            if relation == "inside"
            else position_containment_fraction(right, left)
        )
        scale = max(config.containment_transition, 1e-6)
        return _sigmoid(
            (contained_fraction - config.containment_threshold) / scale
        )

    intersection = int(np.count_nonzero(left & right))
    left_area = int(np.count_nonzero(left))
    right_area = int(np.count_nonzero(right))
    overlap_fraction = intersection / max(min(left_area, right_area), 1)
    overlap_support = (
        0.0 if intersection == 0 else 0.5 + 0.5 * overlap_fraction
    )
    if relation == "overlap":
        return overlap_support
    if relation == "disjoint":
        return 1.0 - overlap_support
    if relation == "touching":
        return float(bool(np.any(_dilate_8(left) & right)))

    distance = hypot(
        subject_center[0] - object_center[0],
        subject_center[1] - object_center[1],
    )
    normalized_distance = distance / diagonal
    scale = max(config.distance_transition_fraction, 1e-6)
    if relation == "near":
        return _sigmoid((config.near_diagonal_fraction - normalized_distance) / scale)
    if relation == "far":
        return _sigmoid((normalized_distance - config.far_diagonal_fraction) / scale)
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
    return tuple(
        any(
            _evaluate_pair(subject, object_, relation, settings)
            for object_ in objects
        )
        for subject in subjects
    )


def score_spatial_relation(
    subjects: Sequence[MaskInstance],
    objects: Sequence[MaskInstance],
    relation: str,
    *,
    config: SpatialConfig | None = None,
) -> float:
    """Return max-pair support for the V1 existential relation semantics."""

    supports = score_spatial_relation_by_subject(
        subjects,
        objects,
        relation,
        config=config,
    )
    return max(supports, default=0.0)


def score_spatial_relation_by_subject(
    subjects: Sequence[MaskInstance],
    objects: Sequence[MaskInstance],
    relation: str,
    *,
    config: SpatialConfig | None = None,
) -> tuple[float, ...]:
    """Return strongest matching-object support for every subject instance."""

    relation = _canonical_relation(relation)
    settings = config or SpatialConfig()
    return tuple(
        max(
            (
                subject.score
                * object_.score
                * _pair_relation_support(subject, object_, relation, settings)
                for object_ in objects
            ),
            default=0.0,
        )
        for subject in subjects
    )
