"""Convert the six V1 task outputs into a common constraint score.

The values in this module are evidence scores in ``[0, 1]``.  They become
probabilities only after an external calibration step on labelled validation
data.  Keeping that distinction explicit prevents a model confidence or a
geometric margin from being over-interpreted as a calibrated probability.
"""

from __future__ import annotations

from itertools import product
from math import isfinite
from typing import Sequence

import numpy as np

from .geometry import ranked_measurements
from .models import AttributePrediction, MaskInstance


def bounded_score(value: float) -> float:
    """Return a finite score clipped to the shared ``[0, 1]`` range."""

    number = float(value)
    if not isfinite(number):
        raise ValueError(f"score must be finite, got {value!r}")
    return float(np.clip(number, 0.0, 1.0))


def score_pair(satisfaction_score: float) -> tuple[float, float]:
    """Return complementary satisfaction and violation scores."""

    satisfaction = bounded_score(satisfaction_score)
    return satisfaction, 1.0 - satisfaction


def _normalize_expected_counts(expected_counts: Sequence[int]) -> frozenset[int]:
    counts = tuple(expected_counts)
    if not counts:
        raise ValueError("expected_counts must contain at least one count")
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in counts
    ):
        raise ValueError("expected_counts must contain only non-negative integers")
    return frozenset(counts)


def count_satisfaction_score(
    instances: Sequence[MaskInstance], expected_counts: Sequence[int]
) -> float:
    """Estimate ``P(N in expected_counts)`` from independent mask scores.

    Each returned SAM3 mask is treated as a candidate object's existence
    evidence.  This models false positive candidates, but it cannot model an
    object that SAM3 failed to propose at all.
    """

    allowed_counts = _normalize_expected_counts(expected_counts)
    distribution = np.zeros(len(instances) + 1, dtype=np.float64)
    distribution[0] = 1.0
    for instance_index, instance in enumerate(instances, start=1):
        probability = bounded_score(instance.score)
        previous = distribution.copy()
        distribution[: instance_index + 1] = 0.0
        distribution[:instance_index] += previous[:instance_index] * (1.0 - probability)
        distribution[1 : instance_index + 1] += (
            previous[:instance_index] * probability
        )
    support = sum(
        distribution[count]
        for count in allowed_counts
        if count < len(distribution)
    )
    return bounded_score(support)


def relation_count_satisfaction_score(
    subject_supports: Sequence[float], expected_counts: Sequence[int]
) -> float:
    """Score an allowed count of subjects that satisfy one spatial relation.

    Every item is the strongest relation support between one subject instance
    and any object instance.  Exact-count support uses fuzzy cardinality: the
    expected present subjects must have strong support and the next subject
    outside the expected count must have weak support.  Taking the maximum over
    allowed counts gives the same OR semantics used by other count arrays.
    """

    allowed_counts = _normalize_expected_counts(expected_counts)
    ordered = sorted(
        (bounded_score(value) for value in subject_supports), reverse=True
    )
    subject_count = len(ordered)
    supports = []
    for count in allowed_counts:
        if count > subject_count:
            supports.append(0.0)
            continue
        present_support = 1.0 if count == 0 else ordered[count - 1]
        absent_support = 1.0 if count == subject_count else 1.0 - ordered[count]
        supports.append(min(present_support, absent_support))
    return bounded_score(max(supports, default=0.0))


def relation_count_at_least_satisfaction_score(
    subject_supports: Sequence[float], minimum_counts: Sequence[int]
) -> float:
    """Score a spatial constraint requiring at least one allowed count.

    ``OP=GE`` retains count-array OR semantics: a rule passes if its observed
    unique-subject relation count is greater than or equal to any listed
    threshold.  Unlike exact cardinality, extra matching subjects are not
    penalized.  For a threshold ``k``, the kth strongest subject-relation
    support is evidence that at least ``k`` subjects satisfy the relation.
    """

    thresholds = _normalize_expected_counts(minimum_counts)
    ordered = sorted(
        (bounded_score(value) for value in subject_supports), reverse=True
    )
    supports = [
        1.0
        if threshold == 0
        else ordered[threshold - 1]
        if threshold <= len(ordered)
        else 0.0
        for threshold in thresholds
    ]
    return bounded_score(max(supports, default=0.0))


def _rank_constraint_holds(
    instances: Sequence[MaskInstance],
    *,
    measure: str,
    rank: int,
    expected_counts: frozenset[int],
    relative_tolerance: float,
) -> bool:
    groups = ranked_measurements(
        instances,
        measure=measure,
        relative_tolerance=relative_tolerance,
    )
    actual = next((group.count for group in groups if group.rank == rank), 0)
    return actual in expected_counts


def ranked_count_satisfaction_score(
    instances: Sequence[MaskInstance],
    *,
    measure: str,
    rank: int,
    expected_counts: Sequence[int],
    relative_tolerance: float,
    max_exact_candidates: int = 16,
    monte_carlo_samples: int = 4096,
) -> tuple[float, str]:
    """Score a LENGTH/AREA allowed rank-count set over mask-existence evidence.

    For a small candidate set this exactly marginalizes every possible subset
    implied by SAM3 mask scores.  A deterministic Monte Carlo fallback keeps
    the route bounded for unusually large candidate sets.
    """

    allowed_counts = _normalize_expected_counts(expected_counts)
    fixed = [instance for instance in instances if instance.score >= 1.0]
    uncertain = [instance for instance in instances if 0.0 < instance.score < 1.0]
    if len(uncertain) <= max_exact_candidates:
        support = 0.0
        for choices in product((False, True), repeat=len(uncertain)):
            probability = 1.0
            selected = list(fixed)
            for include, instance in zip(choices, uncertain):
                probability *= instance.score if include else 1.0 - instance.score
                if include:
                    selected.append(instance)
            if probability and _rank_constraint_holds(
                selected,
                measure=measure,
                rank=rank,
                expected_counts=allowed_counts,
                relative_tolerance=relative_tolerance,
            ):
                support += probability
        return bounded_score(support), "exact_mask_subset_marginalization"

    probabilities = np.asarray([item.score for item in uncertain], dtype=np.float64)
    generator = np.random.default_rng(0)
    holds = 0
    for sample in generator.random((monte_carlo_samples, len(uncertain))):
        selected = fixed + [
            item for item, include in zip(uncertain, sample < probabilities) if include
        ]
        holds += int(
            _rank_constraint_holds(
                selected,
                measure=measure,
                rank=rank,
                expected_counts=allowed_counts,
                relative_tolerance=relative_tolerance,
            )
        )
    return holds / monte_carlo_samples, "deterministic_mask_subset_sampling"


def attribute_satisfaction_score(
    prediction: AttributePrediction, expected: object
) -> tuple[float, str]:
    """Return support for one expected attribute value.

    Full candidate probabilities are preferred.  The confidence fallback keeps
    compatibility with detectors that only expose their winning label.
    """

    expected_label = (
        "true" if expected is True else "false" if expected is False else str(expected)
    )
    if prediction.probabilities:
        return (
            bounded_score(prediction.probabilities.get(expected_label, 0.0)),
            "candidate_probability",
        )
    if prediction.confidence is None:
        return (
            float(prediction.value == expected_label),
            "deterministic_boolean_fallback",
        )
    confidence = bounded_score(prediction.confidence)
    if prediction.value == expected_label:
        return confidence, "winning_label_confidence_fallback"
    return 1.0 - confidence, "winning_label_confidence_fallback"


def conjunction_satisfaction_score(scores: Sequence[float]) -> float:
    """Use hard-conjunction semantics without multiplying by tuple length."""

    if not scores:
        raise ValueError("a constraint combination needs at least one score")
    return min(bounded_score(score) for score in scores)
