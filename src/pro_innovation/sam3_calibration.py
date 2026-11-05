"""Normal-only, object-specific SAM3 confidence-threshold calibration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .interfaces import MaskProcessor, Segmenter
from .models import Rule, TaskType


CALIBRATION_SCHEMA_PATTERN = re.compile(
    r"pro-innovation\.sam3-object-thresholds\.v[0-9]+"
)
LEGACY_CALIBRATION_SCHEMA = "pro-innovation.sam3-object-thresholds.v7"
LEGACY_CALIBRATION_METHOD = (
    "full-normal-confidence-object-specific-q95-q05-maximum-count-"
    "proposal-dynamic-floor010-v7"
)
CONFIDENCE_GAP_MIDPOINT_SELECTION = "confidence_gap_midpoint"
CALIBRATION_SELECTION_METHODS = (
    CONFIDENCE_GAP_MIDPOINT_SELECTION,
)
MINIMUM_CALIBRATION_SAMPLE_COUNT = 2
CALIBRATION_PROPOSAL_THRESHOLD = 0.20


@dataclass(frozen=True)
class Sam3ThresholdPolicy:
    class_name: str
    proposal_threshold: float
    selected_thresholds: Mapping[str, float]
    configuration: Mapping[str, Any]
    source_path: Path


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
    # The proposal threshold bounds unobserved extras, so successful calibration data normally
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
    selection_method: str = CONFIDENCE_GAP_MIDPOINT_SELECTION,
    configuration: Mapping[str, Any] | None = None,
    truth_template_paths: Sequence[str | Path] | None = None,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Calibrate fixed per-object thresholds from all supplied normal images.

    SAM3 proposals are generated once at ``proposal_threshold``.  The default
    ``confidence_gap_midpoint`` computes the Q95/Q05 confidence interval,
    takes its midpoint, and floors it to the 0.10 grid.
    """

    threshold = float(proposal_threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "SAM3 candidate-generation threshold must be within [0.0, 1.0]"
        )
    if selection_method not in CALIBRATION_SELECTION_METHODS:
        raise ValueError(
            f"unsupported SAM3 threshold selection method {selection_method!r}; "
            f"expected one of {CALIBRATION_SELECTION_METHODS}"
        )
    sample_count = len(normal_images)
    if sample_count < MINIMUM_CALIBRATION_SAMPLE_COUNT:
        raise ValueError("SAM3 calibration requires at least two normal images")
    if len(truth_rules_by_image) != sample_count:
        raise ValueError(
            "SAM3 calibration needs one matching truth template per normal image"
        )

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

    processed_at_proposal = []
    for index, image in enumerate(normal_images, start=1):
        proposals = segmenter.segment(image, category_order)
        filtered = {
            category: tuple(
                instance
                for instance in proposals.get(category, ())
                if float(instance.score) >= threshold
            )
            for category in category_order
        }
        processed_at_proposal.append(
            mask_processor.process(category_order, filtered)
        )
        if progress_callback is not None:
            image_name = (
                Path(image).name
                if isinstance(image, (str, Path))
                else f"sample_{index - 1}"
            )
            progress_callback("segment+mask", index, sample_count, image_name)

    calibrations: dict[str, dict[str, Any]] = {}
    selected_thresholds: dict[str, float] = {}
    for category in category_order:
        samples: list[dict[str, Any]] = []
        failure_reason: str | None = None

        for index, (image, sample_truth, proposals) in enumerate(
            zip(normal_images, truths, processed_at_proposal)
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
                "count_at_selected_threshold": None,
            }
            if true_count == 0:
                failure_reason = "zero_truth_count_has_no_keep_boundary"
                record["keep_boundary_score"] = None
                record["extra_boundary_score"] = ranked[0] if ranked else None
                record["extra_score_upper_bound"] = (
                    ranked[0] if ranked else threshold
                )
            elif len(ranked) >= true_count:
                record["keep_boundary_score"] = ranked[true_count - 1]
                record["extra_boundary_score"] = (
                    ranked[true_count] if len(ranked) > true_count else None
                )
                record["extra_score_upper_bound"] = (
                    ranked[true_count]
                    if len(ranked) > true_count
                    else threshold
                )
            if len(ranked) < true_count:
                failure_reason = "candidate_count_below_truth_at_proposal"
            samples.append(record)

        entry: dict[str, Any] = {
            "status": "failed" if failure_reason else "pending",
            "failure_reason": failure_reason,
            "samples": samples,
            "selected_threshold": None,
        }
        entry.update(
            {
                "lower_bound_q95_extra": None,
                "upper_bound_q05_keep": None,
                "stable_range": None,
                "raw_selected_threshold": None,
            }
        )

        if failure_reason is None:
            lower_bound = float(
                np.quantile(
                    [sample["extra_score_upper_bound"] for sample in samples],
                    0.95,
                )
            )
            upper_bound = float(
                np.quantile(
                    [sample["keep_boundary_score"] for sample in samples],
                    0.05,
                )
            )
            entry["lower_bound_q95_extra"] = lower_bound
            entry["upper_bound_q05_keep"] = upper_bound
            entry["stable_range"] = [lower_bound, upper_bound]
            if lower_bound < upper_bound:
                raw_selected = (lower_bound + upper_bound) / 2.0
                selected = floor_sam3_threshold(raw_selected)
                entry["status"] = "calibrated"
                entry["raw_selected_threshold"] = raw_selected
                entry["selected_threshold"] = selected
                for sample in samples:
                    sample["count_at_selected_threshold"] = sum(
                        score >= selected for score in sample["candidate_scores"]
                    )
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
    if progress_callback is not None:
        progress_callback(
            "aggregate-thresholds",
            sample_count,
            sample_count,
            f"{calibrated_count}/{len(category_order)} categories calibrated",
        )

    return {
        "schema": LEGACY_CALIBRATION_SCHEMA,
        "class_name": class_name,
        "status": status,
        "method": LEGACY_CALIBRATION_METHOD,
        "calibration_scope": "all_train_good",
        "calibration_samples": sample_count,
        "multi_value_count_policy": "maximum",
        "proposal_threshold": threshold,
        "threshold_grid_step": 0.10,
        "threshold_selection": selection_method,
        "threshold_precision_decimals": 2,
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


def save_sam3_threshold_calibration_result(
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
    for key in ("proposal_threshold", "threshold_grid_step"):
        marker_index = fixed_two_decimals(serializable, key, marker_index)
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


def save_sam3_threshold_policy(
    payload: Mapping[str, Any], path: str | Path
) -> Path:
    """Save the compact runtime policy and its calibration provenance."""

    raw_selected = payload.get("selected_thresholds")
    if not isinstance(raw_selected, Mapping):
        raise ValueError("SAM3 threshold payload has invalid selected_thresholds")
    selected: dict[str, float] = {}
    for category, value in raw_selected.items():
        if not isinstance(category, str) or not category:
            raise ValueError("SAM3 threshold payload has an invalid category")
        threshold = float(value)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"invalid calibrated SAM3 threshold for {category!r}: {value!r}"
            )
        selected[category] = threshold

    calibration_images: list[str] = []
    raw_images = payload.get("normal_images")
    if not isinstance(raw_images, Sequence) or isinstance(
        raw_images, (str, bytes)
    ):
        raise ValueError("SAM3 threshold payload has invalid normal_images")
    for image in raw_images:
        if not isinstance(image, str) or not image:
            raise ValueError(
                "SAM3 threshold payload has an invalid normal image"
            )
        calibration_images.append(image)

    initial_threshold = float(payload.get("proposal_threshold", -1.0))
    if not np.isfinite(initial_threshold) or not 0.0 <= initial_threshold <= 1.0:
        raise ValueError("SAM3 threshold payload has an invalid proposal_threshold")

    lines = ["{", '  "selected_thresholds": {']
    ordered = list(selected.items())
    for index, (category, threshold) in enumerate(ordered):
        suffix = "," if index + 1 < len(ordered) else ""
        lines.append(
            f"    {json.dumps(category, ensure_ascii=False)}: {threshold:.2f}{suffix}"
        )
    lines.append("  },")
    lines.append('  "calibration_images": [')
    for index, image in enumerate(calibration_images):
        suffix = "," if index + 1 < len(calibration_images) else ""
        lines.append(f"    {json.dumps(image, ensure_ascii=False)}{suffix}")
    lines.append("  ],")
    lines.append(f'  "initial_threshold": {initial_threshold:.2f}')
    lines.append("}")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def load_sam3_threshold_policy(
    path: str | Path, *, expected_class_name: str | None = None
) -> Sam3ThresholdPolicy:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    compact_policy = set(payload) in (
        {"selected_thresholds"},
        {"selected_thresholds", "initial_threshold"},
        {"selected_thresholds", "calibration_images", "initial_threshold"},
    )
    if compact_policy:
        class_name = expected_class_name or source.stem
        proposal_threshold = float(payload.get("initial_threshold", 0.0))
        if not np.isfinite(proposal_threshold) or not 0.0 <= proposal_threshold <= 1.0:
            raise ValueError(
                "SAM3 threshold policy has an invalid initial threshold"
            )
        calibration_images = payload.get("calibration_images", [])
        if (
            not isinstance(calibration_images, list)
            or any(
                not isinstance(image, str) or not image
                for image in calibration_images
            )
        ):
            raise ValueError(
                "SAM3 threshold policy has invalid calibration_images"
            )
        schema_version = None
        raw_thresholds = payload.get("selected_thresholds")
        configuration: dict[str, Any] = {}
    else:
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
        calibration_samples = payload.get("calibration_samples")
        if (
            not isinstance(calibration_samples, int)
            or isinstance(calibration_samples, bool)
            or calibration_samples < MINIMUM_CALIBRATION_SAMPLE_COUNT
        ):
            raise ValueError(
                "SAM3 threshold policy needs at least two calibration samples"
            )
        if payload.get("calibration_scope") not in {"normal_only", "all_train_good"}:
            raise ValueError("SAM3 threshold policy is not normal-only calibration")
        if payload.get("model_parameters_updated") is not False:
            raise ValueError(
                "SAM3 threshold policy must record model_parameters_updated=false"
            )
        proposal_threshold = float(payload.get("proposal_threshold", -1.0))
        if (
            not np.isfinite(proposal_threshold)
            or not 0.0 <= proposal_threshold <= 1.0
        ):
            raise ValueError(
                "SAM3 threshold policy has an invalid candidate-generation threshold"
            )
        schema_version = int(schema.rsplit("v", maxsplit=1)[1])
        grid_step = float(payload.get("threshold_grid_step", -1.0))
        if schema_version >= 8:
            raise ValueError(
                "SAM3 threshold policy uses a removed threshold selection method"
            )
        if abs(grid_step - 0.10) > 1e-12:
            raise ValueError("SAM3 threshold policy does not use the 0.10 grid")
        if payload.get("threshold_rounding") != "floor":
            raise ValueError(
                "SAM3 threshold policy does not use downward rounding"
            )
        raw_thresholds = payload.get("selected_thresholds")
        raw_configuration = payload.get("configuration", {})
        if not isinstance(raw_configuration, dict):
            raise ValueError("SAM3 threshold policy has invalid configuration")
        configuration = raw_configuration
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
                f"0.10 grid: {value!r}"
            )
        selected[category] = threshold
    return Sam3ThresholdPolicy(
        class_name=class_name,
        proposal_threshold=proposal_threshold,
        selected_thresholds=selected,
        configuration=configuration,
        source_path=source,
    )
