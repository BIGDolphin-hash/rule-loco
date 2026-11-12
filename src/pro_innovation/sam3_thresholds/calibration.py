"""Orchestrate normal-only calibration through the Segmenter/MaskProcessor ports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..interfaces import MaskProcessor, Segmenter
from ..models import Rule
from .config import (
    CALIBRATION_PROPOSAL_THRESHOLD,
    CALIBRATION_SELECTION_METHODS,
    CONFIDENCE_GAP_MIDPOINT_SELECTION,
    LEGACY_CALIBRATION_METHOD,
    LEGACY_CALIBRATION_SCHEMA,
    MINIMUM_CALIBRATION_SAMPLE_COUNT,
)
from .selection import calibrate_category
from .truth import extract_exact_count_truth


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

    image_names = [
        str(Path(image).expanduser().resolve())
        if isinstance(image, (str, Path))
        else f"sample_{index}"
        for index, image in enumerate(normal_images)
    ]
    # Keep confidence scores rather than retaining every full-resolution mask.
    scores_by_category: dict[str, list[tuple[float, ...]]] = {
        category: [] for category in category_order
    }
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
        processed = mask_processor.process(category_order, filtered)
        for category in category_order:
            scores_by_category[category].append(
                tuple(float(instance.score) for instance in processed.get(category, ()))
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
        entry = calibrate_category(
            scores_by_image=scores_by_category[category],
            true_counts=[truth[category] for truth in truths],
            image_names=image_names,
            proposal_threshold=threshold,
        )
        if entry["status"] == "calibrated":
            selected_thresholds[category] = entry["selected_threshold"]
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
        "normal_images": image_names,
        "truth_templates": (
            [str(Path(path).expanduser().resolve()) for path in truth_template_paths]
            if truth_template_paths is not None
            else None
        ),
        "selected_thresholds": selected_thresholds,
        "calibrations": calibrations,
        "configuration": dict(configuration or {}),
    }


