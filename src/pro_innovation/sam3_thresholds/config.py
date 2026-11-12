"""Shared defaults and configuration for SAM3 threshold generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any


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

DEFAULT_SAM3_THRESHOLD_POLICY_BASE_ROOT = "/home/lxq/pro-innovation/config/sam3_thresholds"
DEFAULT_SAM3_THRESHOLD_POLICY_ROOT = (
    f"{DEFAULT_SAM3_THRESHOLD_POLICY_BASE_ROOT}/confidence_gap_midpoint_4-shot"
)
DEFAULT_SAM3_THRESHOLD_RESULT_ROOT = (
    "/home/lxq/pro-innovation/results/calibration/sam3_threshold_calibration"
)


def build_sam3_threshold_configuration(
    *,
    standard_path: str | Path,
    checkpoint_path: str | Path,
    prompt_template: str = "{category}",
    mask_dedup_iou: float = 0.80,
    mask_dedup_containment: float = 0.85,
    mask_dedup: bool = True,
) -> dict[str, Any]:
    """Describe the model, prompt and template used by calibration or inference."""

    standard = Path(standard_path).expanduser().resolve()
    return {
        "sam3_prompt": prompt_template,
        "sam3_checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "mask_dedup_iou_threshold": float(mask_dedup_iou),
        "mask_dedup_containment_threshold": float(mask_dedup_containment),
        "mask_dedup_enabled": bool(mask_dedup),
        "standard_template": str(standard),
        "standard_template_sha256": hashlib.sha256(standard.read_bytes()).hexdigest(),
    }


@dataclass(frozen=True)
class Sam3CalibrationConfig:
    """Model and postprocessing settings shared by both generation entry points."""

    repo_path: str | Path = "/home/lxq/models/SAM3"
    checkpoint_path: str | Path = "/home/lxq/weights/sam3/sam3.pt"
    device: str = "cuda"
    proposal_threshold: float = CALIBRATION_PROPOSAL_THRESHOLD
    selection_method: str = CONFIDENCE_GAP_MIDPOINT_SELECTION
    prompt_template: str = "{category}"
    mask_dedup_iou: float = 0.80
    mask_dedup_containment: float = 0.85
    mask_dedup: bool = True

    def configuration(self, standard_path: str | Path) -> dict[str, Any]:
        return build_sam3_threshold_configuration(
            standard_path=standard_path,
            checkpoint_path=self.checkpoint_path,
            prompt_template=self.prompt_template,
            mask_dedup_iou=self.mask_dedup_iou,
            mask_dedup_containment=self.mask_dedup_containment,
            mask_dedup=self.mask_dedup,
        )
