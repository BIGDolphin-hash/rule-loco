"""Orchestrate normal-only calibration through the Segmenter/MaskProcessor ports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..interfaces import MaskProcessor, Segmenter, ThresholdSegmenter
from ..models import Rule
from .config import (
    CALIBRATION_METHOD,
    CALIBRATION_PROPOSAL_THRESHOLD,
    CALIBRATION_SELECTION_METHODS,
    CONFIDENCE_GAP_MIDPOINT_SELECTION,
    LEGACY_CALIBRATION_SCHEMA,
    MINIMUM_CALIBRATION_SAMPLE_COUNT,
    calibration_threshold_schedule,
)
from .selection import calibrate_category
from .truth import extract_count_truths_by_image


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

    Start at ``proposal_threshold`` (0.80 by default), retry failed categories
    in 0.10 steps down to 0.40, and keep the first calibrated result per category.
    Each attempt uses the Q95/Q05 midpoint, floored to the 0.10 grid.

    Threshold-aware adapters receive the cutoff before proposal generation.
    Other injected segmenters must return unfiltered proposals covering 0.40.
    """

    schedule = calibration_threshold_schedule(proposal_threshold)
    if selection_method not in CALIBRATION_SELECTION_METHODS:
        raise ValueError(
            f"unsupported SAM3 threshold selection method {selection_method!r}; "
            f"expected one of {CALIBRATION_SELECTION_METHODS}"
        )
    sample_count = len(normal_images)
    if sample_count < MINIMUM_CALIBRATION_SAMPLE_COUNT:
        raise ValueError("SAM3 calibration requires at least two normal images")
    truths = extract_count_truths_by_image(
        normal_images=normal_images,
        truth_rules_by_image=truth_rules_by_image,
        truth_template_paths=truth_template_paths,
    )
    category_order = tuple(truths[0])

    image_names = [
        str(Path(image).expanduser().resolve())
        if isinstance(image, (str, Path))
        else f"sample_{index}"
        for index, image in enumerate(normal_images)
    ]
    truth_paths = (
        [str(Path(path).expanduser().resolve()) for path in truth_template_paths]
        if truth_template_paths is not None
        else None
    )
    calibrations: dict[str, dict[str, Any]] = {}
    selected_thresholds: dict[str, float] = {}
    attempts: dict[str, list[dict[str, Any]]] = {
        category: [] for category in category_order
    }
    threshold_aware = isinstance(segmenter, ThresholdSegmenter)
    for attempt_index, threshold in enumerate(schedule):
        pending = tuple(
            category for category in category_order if category not in selected_thresholds
        )
        if not pending:
            break
        if attempt_index and progress_callback is not None:
            progress_callback(
                "retry-threshold", attempt_index + 1, len(schedule),
                f"proposal_threshold={threshold:.2f}; categories={', '.join(pending)}",
            )
        # Keep scores for this attempt, rather than retaining full-resolution masks.
        scores_by_category: dict[str, list[tuple[float, ...]]] = {
            category: [] for category in pending
        }
        for index, image in enumerate(normal_images, start=1):
            if threshold_aware:
                proposals = segmenter.segment_at_threshold(
                    image, pending, threshold=threshold
                )
            else:
                proposals = segmenter.segment(image, pending)
            filtered = {
                category: tuple(
                    instance
                    for instance in proposals.get(category, ())
                    if float(instance.score) >= threshold
                )
                for category in pending
            }
            processed = mask_processor.process(pending, filtered)
            for category in pending:
                scores_by_category[category].append(
                    tuple(float(instance.score) for instance in processed.get(category, ()))
                )
            if progress_callback is not None:
                progress_callback(
                    "segment+mask", index, sample_count,
                    f"{Path(image_names[index - 1]).name}; proposal_threshold={threshold:.2f}",
                )

        for category in pending:
            entry = calibrate_category(
                scores_by_image=scores_by_category[category],
                true_counts=[truth[category] for truth in truths],
                image_names=image_names,
                proposal_threshold=threshold,
            )
            attempts[category].append({"proposal_threshold": threshold, **entry})
            if entry["status"] == "calibrated":
                selected_thresholds[category] = entry["selected_threshold"]
            calibrations[category] = {
                **entry,
                "proposal_threshold": threshold,
                "attempts": attempts[category],
            }

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
        "method": CALIBRATION_METHOD,
        "calibration_scope": "all_train_good",
        "calibration_samples": sample_count,
        "multi_value_count_policy": "require_exact_per_image",
        "proposal_threshold": schedule[0],
        "proposal_threshold_schedule": list(schedule),
        "minimum_proposal_threshold": schedule[-1],
        "threshold_grid_step": 0.10,
        "threshold_selection": selection_method,
        "threshold_precision_decimals": 2,
        "threshold_rounding": "floor",
        "model_parameters_updated": False,
        "normal_images": image_names,
        "truth_templates": truth_paths,
        "image_truths": [
            {
                "sample_index": index,
                "image": image_names[index],
                "truth_template": truth_paths[index] if truth_paths is not None else None,
                "true_counts": dict(truth),
            }
            for index, truth in enumerate(truths)
        ],
        "selected_thresholds": selected_thresholds,
        "calibrations": calibrations,
        "configuration": dict(configuration or {}),
    }


