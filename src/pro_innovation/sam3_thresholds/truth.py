"""Extract per-object count truth from concrete rule templates."""

from __future__ import annotations

from typing import Sequence

from ..models import Rule, TaskType


def extract_exact_count_truth(rules: Sequence[Rule]) -> dict[str, int]:
    """Read calibratable object counts, preferring explicit COUNT rules."""

    truth: dict[str, int] = {}
    # A dedicated COUNT describes the total SAM3 object count. Measurement-rule
    # COUNT fields describe how many instances satisfy that range and must not
    # override the object's total count (for example the two screw length groups).
    for task in (TaskType.COUNT, TaskType.LENGTH, TaskType.AREA):
        for rule in rules:
            if not rule.subject or rule.task is not task:
                continue
            if task is not TaskType.COUNT and rule.subject in truth:
                continue
            values = rule.expected if task is TaskType.COUNT else rule.count
            if values is None:
                continue
            if not isinstance(values, tuple) or not values:
                raise ValueError(
                    f"{rule.rule_id}: SAM3 calibration needs at least one count "
                    f"value for {rule.subject!r}"
                )
            if any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in values
            ):
                raise ValueError(
                    f"{rule.rule_id}: invalid COUNT truth for {rule.subject!r}"
                )
            value = max(values)
            previous = truth.get(rule.subject)
            if previous is not None and previous != value:
                raise ValueError(
                    f"conflicting COUNT truth values for {rule.subject!r}: "
                    f"{previous} and {value}"
                )
            truth[rule.subject] = value

    if not truth:
        raise ValueError(
            "truth template contains no calibratable COUNT/LENGTH/AREA rules"
        )
    return truth
