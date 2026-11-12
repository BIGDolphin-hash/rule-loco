"""Pure confidence-gap selection, independent of SAM3 and filesystem I/O."""

from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR
from typing import Any, Sequence

import numpy as np


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
    # Preserve the historical grid rounding, even when it crosses a confidence
    # boundary; count_at_selected_threshold in the report exposes that outcome.
    bounded = max(Decimal("0.00"), min(Decimal("1.00"), floored))
    return float(bounded.quantize(Decimal("0.00")))


def calibrate_category(
    *,
    scores_by_image: Sequence[Sequence[float]],
    true_counts: Sequence[int],
    image_names: Sequence[str],
    proposal_threshold: float,
) -> dict[str, Any]:
    """Compute per-sample boundaries and the Q95/Q05 midpoint for one category."""

    samples: list[dict[str, Any]] = []
    failure_reason: str | None = None

    for index, (image_name, true_count, scores) in enumerate(
        zip(image_names, true_counts, scores_by_image)
    ):
        ranked = sorted((float(score) for score in scores), reverse=True)
        record: dict[str, Any] = {
            "sample_index": index,
            "image": image_name,
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
                ranked[0] if ranked else proposal_threshold
            )
        elif len(ranked) >= true_count:
            record["keep_boundary_score"] = ranked[true_count - 1]
            record["extra_boundary_score"] = (
                ranked[true_count] if len(ranked) > true_count else None
            )
            record["extra_score_upper_bound"] = (
                ranked[true_count]
                if len(ranked) > true_count
                else proposal_threshold
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
        else:
            entry["status"] = "failed"
            entry["failure_reason"] = "no_confidence_gap"
    return entry
