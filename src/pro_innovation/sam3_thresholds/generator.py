"""Generate and save threshold artifacts independently of the project CLI."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from ..adapters.sam3 import Sam3Segmenter
from ..image_validation import ProgressCallback
from ..interfaces import MaskProcessor, Segmenter
from ..mask_processing import MaskPostprocessor
from .calibration import calibrate_sam3_thresholds
from .config import (
    DEFAULT_SAM3_THRESHOLD_POLICY_BASE_ROOT,
    DEFAULT_SAM3_THRESHOLD_RESULT_ROOT,
    Sam3CalibrationConfig,
)
from .inputs import CalibrationInputs
from .policy import save_sam3_threshold_calibration_result, save_sam3_threshold_policy


def generate_sam3_thresholds(
    *,
    class_name: str,
    inputs: CalibrationInputs,
    config: Sam3CalibrationConfig | None = None,
    output: str | Path | None = None,
    policy_root: str | Path = DEFAULT_SAM3_THRESHOLD_POLICY_BASE_ROOT,
    result_root: str | Path = DEFAULT_SAM3_THRESHOLD_RESULT_ROOT,
    overwrite: bool = False,
    segmenter: Segmenter | None = None,
    mask_processor: MaskProcessor | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run calibration and save its compact runtime policy and detailed report.

    Failed categories remain in the report and are omitted from the policy,
    preserving the runtime global-threshold fallback. Model ports can be
    injected for offline processing and verification.
    """

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", class_name):
        raise ValueError(
            "class name may contain letters, numbers, underscores, and hyphens only"
        )
    target = (
        Path(output or Path(policy_root) / f"{class_name}.json")
        .expanduser()
        .resolve()
    )
    if target.exists() and not overwrite:
        raise ValueError(
            f"SAM3 threshold policy already exists: {target}; "
            "choose a different --output path or pass --overwrite explicitly"
        )
    result_path = (
        Path(result_root).expanduser().resolve()
        / target.parent.name
        / f"{class_name}.json"
    )
    if result_path == target:
        raise ValueError(
            "SAM3 threshold policy and calibration report must use different paths"
        )
    settings = config or Sam3CalibrationConfig()
    configuration = settings.configuration(inputs.standard_path)
    if segmenter is None:
        segmenter = Sam3Segmenter(
            repo_path=settings.repo_path,
            checkpoint_path=settings.checkpoint_path,
            device=settings.device,
            confidence_threshold=settings.proposal_threshold,
            prompt_template=settings.prompt_template,
        )
    if mask_processor is None:
        mask_processor = MaskPostprocessor(
            dedup_iou_threshold=settings.mask_dedup_iou,
            dedup_containment_threshold=settings.mask_dedup_containment,
            dedup_enabled=settings.mask_dedup,
        )
    result = calibrate_sam3_thresholds(
        class_name=class_name,
        normal_images=inputs.normal_images,
        truth_rules_by_image=inputs.truth_rules_by_image,
        segmenter=segmenter,
        mask_processor=mask_processor,
        proposal_threshold=settings.proposal_threshold,
        selection_method=settings.selection_method,
        configuration=configuration,
        truth_template_paths=inputs.truth_template_paths,
        progress_callback=progress_callback,
    )
    result["saved_path"] = str(target)
    result["calibration_result_path"] = str(result_path)
    save_sam3_threshold_calibration_result(result, result_path)
    save_sam3_threshold_policy(result, target)
    if progress_callback is not None:
        count = len(inputs.normal_images)
        progress_callback("saved", count, count, str(target))
    return result
