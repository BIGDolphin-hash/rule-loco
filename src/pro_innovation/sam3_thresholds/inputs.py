"""Prepare normal images and aligned per-image truth without loading SAM3."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from ..image_validation import ProgressCallback, filter_decodable_images
from ..models import Rule, TaskType
from ..templates import parse_template
from .config import MINIMUM_CALIBRATION_SAMPLE_COUNT
from .truth import extract_count_truths_by_image


_COUNT_TASKS = {TaskType.COUNT, TaskType.LENGTH, TaskType.AREA}


@dataclass(frozen=True)
class CalibrationInputs:
    standard_path: Path
    normal_images: tuple[Path, ...]
    truth_rules_by_image: tuple[tuple[Rule, ...], ...]
    truth_template_paths: tuple[Path, ...] | None


def resolve_normal_images(
    *,
    normal_images: Sequence[str | Path] | None = None,
    normal_dir: str | Path | None = None,
) -> tuple[Path, ...]:
    """Resolve distinct images in explicit order or sorted directory order."""

    if bool(normal_images) == bool(normal_dir):
        raise ValueError("provide exactly one of --normal-images or --normal-dir")
    if normal_dir:
        root = Path(normal_dir).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"normal image directory not found: {root}")
        paths = tuple(
            path.resolve()
            for path in sorted(root.iterdir())
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    else:
        paths = tuple(Path(path).expanduser().resolve() for path in normal_images)
    if not paths:
        raise ValueError("normal memory requires at least one image")
    if len(set(paths)) != len(paths):
        raise ValueError("normal memory images must be distinct")
    return paths


def _count_values(rule: Rule) -> object:
    return rule.expected if rule.task is TaskType.COUNT else rule.count


def _replace_counts(
    rules: Sequence[Rule], counts: Mapping[str, int]
) -> tuple[Rule, ...]:
    explicit_count_subjects = {
        rule.subject for rule in rules if rule.task is TaskType.COUNT
    }
    return tuple(
        replace(rule, expected=(counts[rule.subject],))
        if rule.task is TaskType.COUNT and rule.subject in counts
        else replace(rule, count=(counts[rule.subject],))
        if rule.task in {TaskType.LENGTH, TaskType.AREA}
        and rule.subject in counts
        and rule.subject not in explicit_count_subjects
        else rule
        for rule in rules
    )


def prepare_calibration_inputs(
    *,
    standard_path: str | Path,
    normal_images: Sequence[str | Path] | None = None,
    normal_dir: str | Path | None = None,
    truth_templates: Sequence[str | Path] | None = None,
    truth_counts: Sequence[int] | None = None,
    truth_subject: str | None = None,
    truth_subject_counts: Mapping[str, Sequence[int]] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> CalibrationInputs:
    """Align truth with readable images, including after corrupt-image skips.

    Use a single-valued standard template, one concrete template per image,
    or inline per-image counts. Every calibratable quantity must resolve to
    one integer before inference. Inline counts follow the original image order.
    """

    sources = (truth_templates, truth_counts, truth_subject_counts)
    if sum(source is not None for source in sources) > 1:
        raise ValueError(
            "use only one of --truth-templates, --truth-counts, "
            "or --truth-subject-counts"
        )
    if truth_subject is not None and truth_counts is None:
        raise ValueError("--truth-subject requires --truth-counts")
    standard = Path(standard_path).expanduser().resolve()
    standard_rules = parse_template(
        standard.read_text(encoding="utf-8"), allow_missing_values=False,
        allow_deferred_lengths=True, allow_unknown_values=True,
    )
    input_images = resolve_normal_images(
        normal_images=normal_images, normal_dir=normal_dir
    )
    missing = [str(path) for path in input_images if not path.is_file()]
    if missing:
        raise ValueError("normal calibration images not found: " + ", ".join(missing))
    images = filter_decodable_images(input_images, progress_callback=progress_callback)
    if len(images) < MINIMUM_CALIBRATION_SAMPLE_COUNT:
        raise ValueError(
            "SAM3 calibration needs at least two decodable normal images after skips"
        )
    retained = set(images)
    indices = tuple(i for i, path in enumerate(input_images) if path in retained)

    subjects = tuple(dict.fromkeys(
        rule.subject
        for rule in standard_rules
        if rule.task in _COUNT_TASKS and rule.subject
    ))

    def validated_inputs(
        rules: tuple[tuple[Rule, ...], ...], paths: tuple[Path, ...] | None
    ) -> CalibrationInputs:
        extract_count_truths_by_image(
            normal_images=images,
            truth_rules_by_image=rules,
            truth_template_paths=paths,
            expected_subjects=subjects,
        )
        return CalibrationInputs(standard, images, rules, paths)

    if truth_templates is not None:
        paths = tuple(Path(path).expanduser().resolve() for path in truth_templates)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise ValueError("truth templates not found: " + ", ".join(missing))
        if len(paths) != len(input_images):
            raise ValueError(
                f"--truth-templates needs {len(input_images)} paths, found {len(paths)}"
            )
        paths = tuple(paths[index] for index in indices)
        rules = tuple(
            parse_template(
                path.read_text(encoding="utf-8"), allow_missing_values=False,
                allow_deferred_lengths=True, allow_unknown_values=True,
            )
            for path in paths
        )
        return validated_inputs(rules, paths)
    variable_subjects = {
        rule.subject
        for rule in standard_rules
        if rule.task in _COUNT_TASKS
        and rule.subject
        and isinstance(values := _count_values(rule), tuple)
        and len(values) > 1
    }
    overrides = dict(truth_subject_counts or {})
    if truth_counts is not None:
        if truth_subject is None:
            if len(variable_subjects) != 1:
                raise ValueError(
                    "--truth-counts needs --truth-subject when the standard template "
                    "has zero or multiple variable COUNT subjects"
                )
            truth_subject = next(iter(variable_subjects))
        overrides[truth_subject] = truth_counts
    if truth_counts is not None or truth_subject_counts is not None:
        missing_subjects = sorted(variable_subjects - overrides.keys())
        if missing_subjects:
            raise ValueError(
                "missing inline truth counts for variable COUNT subjects: "
                + ", ".join(missing_subjects)
            )
    for subject, values in overrides.items():
        if subject not in subjects:
            raise ValueError(
                f"inline truth subject {subject!r} is not a COUNT subject "
                "in the standard template"
            )
        if len(values) != len(input_images):
            raise ValueError(
                f"inline truth counts for {subject!r} need {len(input_images)} "
                f"values, found {len(values)}"
            )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values
        ):
            raise ValueError("inline truth count values must be non-negative integers")

    if overrides:
        rules = tuple(
            _replace_counts(
                standard_rules,
                {subject: values[i] for subject, values in overrides.items()},
            )
            for i in indices
        )
        return validated_inputs(rules, None)
    return validated_inputs(
        tuple(standard_rules for _ in images), (standard,) * len(images)
    )
