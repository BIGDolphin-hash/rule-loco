"""Normal-only, object-specific SAM3 confidence-threshold calibration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .interfaces import MaskProcessor, Segmenter
from .models import Rule, TaskType


CALIBRATION_SCHEMA = "pro-innovation.sam3-object-thresholds.v6"
CALIBRATION_SCHEMA_PATTERN = re.compile(
    r"pro-innovation\.sam3-object-thresholds\.v[0-9]+"
)
CALIBRATION_METHOD = "few-shot-confidence-object-specific-q95-q05-proposal-dynamic-floor010-v6"
CALIBRATION_SAMPLE_COUNT = 4
CALIBRATION_PROPOSAL_THRESHOLD = 0.10
CALIBRATION_THRESHOLD_STEP = 0.10


@dataclass(frozen=True)
class Sam3ThresholdPolicy:
    class_name: str
    proposal_threshold: float
    selected_thresholds: Mapping[str, float]
    configuration: Mapping[str, Any]
    source_path: Path


def extract_exact_count_truth(rules: Sequence[Rule]) -> dict[str, int]:
    """Read one exact count per COUNT/LENGTH/AREA subject from a template.

    An allowed set such as ``[4,6,10]`` is not a sample-specific truth value, so
    callers must provide a concrete template for that support image instead of
    guessing which member applies.
    """

    truth: dict[str, int] = {}
    for rule in rules:
        if not rule.subject or rule.task not in {
            TaskType.COUNT,
            TaskType.LENGTH,
            TaskType.AREA,
        }:
            continue
        values = rule.expected if rule.task is TaskType.COUNT else rule.count
        if values is None:
            continue
        if not isinstance(values, tuple) or len(values) != 1:
            raise ValueError(
                f"{rule.rule_id}: SAM3 calibration needs one exact count value "
                f"for {rule.subject!r}; provide a sample-specific truth template"
            )
        value = values[0]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"{rule.rule_id}: invalid exact COUNT truth for {rule.subject!r}"
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


def _quantile(values: Sequence[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def floor_sam3_threshold(value: float) -> float:
    """Floor a threshold to the 0.10 grid and return two-decimal precision."""

    threshold = float(value)
    if not np.isfinite(threshold):
        raise ValueError("SAM3 threshold must be finite")
    step = Decimal("0.10")
    # Absorb only floating-point noise at an exact grid boundary.
    decimal_value = Decimal(str(threshold)) + Decimal("0.000000000001")
    units = (decimal_value / step).to_integral_value(rounding=ROUND_FLOOR)
    floored = units * step
    # There is no fixed lower bound for the selected threshold.  The
    # proposal threshold is 0.10, so successful calibration data normally
    # keeps the selected value at or above that evidence boundary naturally.
    bounded = max(Decimal("0.00"), min(Decimal("1.00"), floored))
    return float(bounded.quantize(Decimal("0.00")))


def calibrate_sam3_thresholds(
    *,
    class_name: str,
    normal_images: Sequence[Any],
    truth_rules_by_image: Sequence[Sequence[Rule]],
    segmenter: Segmenter,
    mask_processor: MaskProcessor,
    proposal_threshold: float = CALIBRATION_PROPOSAL_THRESHOLD,
    configuration: Mapping[str, Any] | None = None,
    truth_template_paths: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Calibrate fixed per-object thresholds from exactly four normal images.

    For true count ``K`` in each image, ``s_K`` is the lowest required proposal
    and ``s_(K+1)`` is the highest extra proposal.  Missing extra proposals are
    conservatively bounded by the proposal-generation threshold.
    """

    threshold = float(proposal_threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "SAM3 candidate-generation threshold must be within [0.0, 1.0]"
        )
    if len(normal_images) != CALIBRATION_SAMPLE_COUNT:
        raise ValueError("SAM3 calibration requires exactly four normal images")
    if len(truth_rules_by_image) != CALIBRATION_SAMPLE_COUNT:
        raise ValueError("SAM3 calibration requires four matching truth templates")

    truths = [extract_exact_count_truth(rules) for rules in truth_rules_by_image]
    category_order = tuple(truths[0])
    expected_categories = set(category_order)
    for sample_index, sample_truth in enumerate(truths[1:], start=2):
        if set(sample_truth) != expected_categories:
            missing = sorted(expected_categories - set(sample_truth))
            extra = sorted(set(sample_truth) - expected_categories)
            raise ValueError(
                f"truth template {sample_index} has different COUNT subjects; "
                f"missing={missing}, extra={extra}"
            )

    processed_by_image = []
    for image in normal_images:
        proposals = segmenter.segment(image, category_order)
        processed_by_image.append(
            mask_processor.process(category_order, proposals)
        )

    calibrations: dict[str, dict[str, Any]] = {}
    selected_thresholds: dict[str, float] = {}
    for category in category_order:
        keep_scores: list[float] = []
        extra_upper_bounds: list[float] = []
        samples: list[dict[str, Any]] = []
        failure_reason: str | None = None

        for index, (image, sample_truth, proposals) in enumerate(
            zip(normal_images, truths, processed_by_image)
        ):
            true_count = sample_truth[category]
            ranked = sorted(
                (float(instance.score) for instance in proposals.get(category, ())),
                reverse=True,
            )
            record: dict[str, Any] = {
                "sample_index": index,
                "image": str(Path(image).expanduser().resolve())
                if isinstance(image, (str, Path))
                else f"sample_{index}",
                "true_count": true_count,
                "candidate_count_at_proposal": len(ranked),
                "candidate_scores": ranked,
                "keep_boundary_score": None,
                "extra_boundary_score": None,
                "extra_score_upper_bound": None,
            }
            if true_count == 0:
                failure_reason = "zero_truth_count_has_no_keep_boundary"
            elif len(ranked) < true_count:
                failure_reason = "candidate_count_below_truth_at_proposal"
            else:
                keep_score = ranked[true_count - 1]
                extra_score = ranked[true_count] if len(ranked) > true_count else None
                extra_upper_bound = (
                    extra_score if extra_score is not None else threshold
                )
                keep_scores.append(keep_score)
                extra_upper_bounds.append(extra_upper_bound)
                record["keep_boundary_score"] = keep_score
                record["extra_boundary_score"] = extra_score
                record["extra_score_upper_bound"] = extra_upper_bound
            samples.append(record)

        entry: dict[str, Any] = {
            "status": "failed" if failure_reason else "pending",
            "failure_reason": failure_reason,
            "samples": samples,
            "lower_bound_q95_extra": None,
            "upper_bound_q05_keep": None,
            "stable_range": None,
            "raw_selected_threshold": None,
            "selected_threshold": None,
        }
        if failure_reason is None:
            lower = _quantile(extra_upper_bounds, 0.95)
            upper = _quantile(keep_scores, 0.05)
            entry["lower_bound_q95_extra"] = lower
            entry["upper_bound_q05_keep"] = upper
            entry["stable_range"] = [lower, upper] if lower < upper else None
            if lower < upper:
                raw_selected = (lower + upper) / 2.0
                selected = floor_sam3_threshold(raw_selected)
                entry["status"] = "calibrated"
                entry["raw_selected_threshold"] = raw_selected
                entry["selected_threshold"] = selected
                selected_thresholds[category] = selected
            else:
                entry["status"] = "failed"
                entry["failure_reason"] = "no_confidence_gap"
        calibrations[category] = entry

    calibrated_count = len(selected_thresholds)
    if calibrated_count == len(category_order):
        status = "complete"
    elif calibrated_count:
        status = "partial"
    else:
        status = "failed"

    return {
        "schema": CALIBRATION_SCHEMA,
        "class_name": class_name,
        "status": status,
        "method": CALIBRATION_METHOD,
        "calibration_scope": "normal_only",
        "calibration_samples": CALIBRATION_SAMPLE_COUNT,
        "proposal_threshold": threshold,
        "threshold_grid_step": CALIBRATION_THRESHOLD_STEP,
        "threshold_rounding": "floor",
        "model_parameters_updated": False,
        "normal_images": [
            str(Path(image).expanduser().resolve())
            if isinstance(image, (str, Path))
            else f"sample_{index}"
            for index, image in enumerate(normal_images)
        ],
        "truth_templates": (
            [str(Path(path).expanduser().resolve()) for path in truth_template_paths]
            if truth_template_paths is not None
            else None
        ),
        "selected_thresholds": selected_thresholds,
        "calibrations": calibrations,
        "configuration": dict(configuration or {}),
    }


def save_sam3_threshold_policy(
    payload: Mapping[str, Any], path: str | Path
) -> Path:
    serializable = deepcopy(dict(payload))
    replacements: dict[str, str] = {}

    def fixed_two_decimals(container: dict[str, Any], key: str, index: int) -> int:
        value = container.get(key)
        if value is None:
            return index
        marker = f"__SAM3_FIXED_THRESHOLD_{index}__"
        replacements[marker] = f"{float(value):.2f}"
        container[key] = marker
        return index + 1

    marker_index = 0
    selected = serializable.get("selected_thresholds")
    if isinstance(selected, dict):
        for category in sorted(selected):
            marker_index = fixed_two_decimals(selected, category, marker_index)
    calibrations = serializable.get("calibrations")
    if isinstance(calibrations, dict):
        for category in sorted(calibrations):
            entry = calibrations.get(category)
            if isinstance(entry, dict):
                marker_index = fixed_two_decimals(
                    entry, "selected_threshold", marker_index
                )

    text = json.dumps(serializable, ensure_ascii=False, indent=2)
    for marker, fixed_value in replacements.items():
        text = text.replace(json.dumps(marker), fixed_value)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        text + "\n",
        encoding="utf-8",
    )
    return target


def load_sam3_threshold_policy(
    path: str | Path, *, expected_class_name: str | None = None
) -> Sam3ThresholdPolicy:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if not isinstance(schema, str) or CALIBRATION_SCHEMA_PATTERN.fullmatch(
        schema
    ) is None:
        raise ValueError(f"unsupported SAM3 threshold policy schema in {source}")
    class_name = payload.get("class_name")
    if not isinstance(class_name, str) or not class_name:
        raise ValueError(f"SAM3 threshold policy has no class name: {source}")
    if expected_class_name is not None and class_name != expected_class_name:
        raise ValueError(
            f"SAM3 threshold policy class mismatch: expected "
            f"{expected_class_name!r}, found {class_name!r}"
        )
    if payload.get("calibration_samples") != CALIBRATION_SAMPLE_COUNT:
        raise ValueError("SAM3 threshold policy was not calibrated from four samples")
    if payload.get("calibration_scope") != "normal_only":
        raise ValueError("SAM3 threshold policy is not normal-only calibration")
    if payload.get("model_parameters_updated") is not False:
        raise ValueError(
            "SAM3 threshold policy must record model_parameters_updated=false"
        )
    proposal_threshold = float(payload.get("proposal_threshold", -1.0))
    if not np.isfinite(proposal_threshold) or not 0.0 <= proposal_threshold <= 1.0:
        raise ValueError(
            "SAM3 threshold policy has an invalid candidate-generation threshold"
        )
    if abs(float(payload.get("threshold_grid_step", -1.0)) - 0.10) > 1e-12:
        raise ValueError("SAM3 threshold policy does not use the 0.10 grid")
    if payload.get("threshold_rounding") != "floor":
        raise ValueError("SAM3 threshold policy does not use downward rounding")
    raw_thresholds = payload.get("selected_thresholds")
    if not isinstance(raw_thresholds, dict):
        raise ValueError("SAM3 threshold policy has invalid selected_thresholds")
    selected: dict[str, float] = {}
    for category, value in raw_thresholds.items():
        if not isinstance(category, str) or not category:
            raise ValueError("SAM3 threshold policy has an invalid category")
        threshold = float(value)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"invalid calibrated SAM3 threshold for {category!r}: {value!r}"
            )
        if abs(threshold - floor_sam3_threshold(threshold)) > 1e-12:
            raise ValueError(
                f"calibrated SAM3 threshold for {category!r} is not on the "
                "0.10 grid: {value!r}"
            )
        selected[category] = threshold
    configuration = payload.get("configuration", {})
    if not isinstance(configuration, dict):
        raise ValueError("SAM3 threshold policy has invalid configuration")
    return Sam3ThresholdPolicy(
        class_name=class_name,
        proposal_threshold=proposal_threshold,
        selected_thresholds=selected,
        configuration=configuration,
        source_path=source,
    )
