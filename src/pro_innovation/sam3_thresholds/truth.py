"""Extract per-object count truth from concrete rule templates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..models import Rule, TaskType


def extract_exact_count_truth(rules: Sequence[Rule]) -> dict[str, int]:
    """Read single-valued image-specific counts, preferring explicit COUNT rules."""

    for rule in rules:
        if not rule.subject or rule.task not in {
            TaskType.COUNT, TaskType.LENGTH, TaskType.AREA
        }:
            continue
        values = rule.expected if rule.task is TaskType.COUNT else rule.count
        if values is None:
            continue
        if not isinstance(values, tuple) or not values:
            raise ValueError(
                f"{rule.rule_id}: SAM3 calibration needs exactly one count "
                f"value for {rule.subject!r}"
            )
        # Deferred measurement counts may remain unknown when a higher-priority
        # rule supplies the exact object count. An unknown actually used as
        # calibration truth is rejected below.
        if values == ("unknown",):
            continue
        if any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in values
        ):
            raise ValueError(
                f"{rule.rule_id}: invalid COUNT truth for {rule.subject!r}"
            )
        if len(values) != 1:
            raise ValueError(
                f"{rule.rule_id}: ambiguous COUNT truth for {rule.subject!r}: "
                f"{list(values)}; SAM3 calibration requires exactly one count "
                "per image; provide image-specific --truth-templates or "
                "--truth-counts/--truth-subject-counts"
            )

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
            value = values[0]
            if value == "unknown":
                raise ValueError(
                    f"{rule.rule_id}: resolve unknown COUNT for {rule.subject!r} "
                    "before SAM3 threshold calibration"
                )
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


def extract_count_truths_by_image(
    *,
    normal_images: Sequence[Any],
    truth_rules_by_image: Sequence[Sequence[Rule]],
    truth_template_paths: Sequence[str | Path] | None = None,
    expected_subjects: Sequence[str] | None = None,
) -> tuple[dict[str, int], ...]:
    """Validate image/template alignment and identify invalid truth by its image."""

    sample_count = len(normal_images)
    if len(truth_rules_by_image) != sample_count:
        raise ValueError(
            "SAM3 calibration needs one matching truth template per normal image"
        )
    if truth_template_paths is not None and len(truth_template_paths) != sample_count:
        raise ValueError(
            "SAM3 calibration needs one matching truth template path per normal image"
        )
    subjects = set(expected_subjects) if expected_subjects is not None else None
    truths = []
    for index, (image, rules) in enumerate(zip(normal_images, truth_rules_by_image)):
        image_name = (
            str(Path(image).expanduser().resolve())
            if isinstance(image, (str, Path))
            else f"sample_{index}"
        )
        context = f"SAM3 calibration truth for image {image_name!r}"
        if truth_template_paths is not None:
            context += f" (template {str(truth_template_paths[index])!r})"
        try:
            truth = extract_exact_count_truth(rules)
        except ValueError as exc:
            raise ValueError(f"{context}: {exc}") from exc
        if subjects is None:
            subjects = set(truth)
        if set(truth) != subjects:
            missing = sorted(subjects - set(truth))
            extra = sorted(set(truth) - subjects)
            raise ValueError(
                f"{context} has different COUNT subjects; "
                f"missing={missing}, extra={extra}"
            )
        truths.append(truth)
    return tuple(truths)
