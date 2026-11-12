"""Read legacy policies and write compact policies or full calibration reports."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import CALIBRATION_SCHEMA_PATTERN, MINIMUM_CALIBRATION_SAMPLE_COUNT
from .selection import floor_sam3_threshold


@dataclass(frozen=True)
class Sam3ThresholdPolicy:
    class_name: str
    proposal_threshold: float
    selected_thresholds: Mapping[str, float]
    configuration: Mapping[str, Any]
    source_path: Path


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
