"""Reusable generation and loading of normal-only, per-object SAM3 thresholds."""

from .calibration import calibrate_sam3_thresholds
from .config import (
    CALIBRATION_PROPOSAL_THRESHOLD,
    CALIBRATION_SCHEMA_PATTERN,
    CALIBRATION_SELECTION_METHODS,
    CONFIDENCE_GAP_MIDPOINT_SELECTION,
    LEGACY_CALIBRATION_METHOD,
    LEGACY_CALIBRATION_SCHEMA,
    MINIMUM_CALIBRATION_SAMPLE_COUNT,
    Sam3CalibrationConfig,
    build_sam3_threshold_configuration,
)
from .generator import generate_sam3_thresholds
from .inputs import CalibrationInputs, prepare_calibration_inputs, resolve_normal_images
from .policy import (
    Sam3ThresholdPolicy,
    load_sam3_threshold_policy,
    save_sam3_threshold_calibration_result,
    save_sam3_threshold_policy,
)
from .selection import floor_sam3_threshold
from .truth import extract_exact_count_truth

__all__ = [
    "CALIBRATION_PROPOSAL_THRESHOLD",
    "CALIBRATION_SCHEMA_PATTERN",
    "CALIBRATION_SELECTION_METHODS",
    "CONFIDENCE_GAP_MIDPOINT_SELECTION",
    "LEGACY_CALIBRATION_METHOD",
    "LEGACY_CALIBRATION_SCHEMA",
    "MINIMUM_CALIBRATION_SAMPLE_COUNT",
    "CalibrationInputs",
    "Sam3CalibrationConfig",
    "Sam3ThresholdPolicy",
    "build_sam3_threshold_configuration",
    "calibrate_sam3_thresholds",
    "extract_exact_count_truth",
    "floor_sam3_threshold",
    "generate_sam3_thresholds",
    "load_sam3_threshold_policy",
    "prepare_calibration_inputs",
    "resolve_normal_images",
    "save_sam3_threshold_calibration_result",
    "save_sam3_threshold_policy",
]
