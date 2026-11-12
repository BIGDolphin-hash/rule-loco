"""CLI shared by the standalone module and calibrate-sam3-thresholds command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..errors import ProInnovationError
from .config import (
    CALIBRATION_PROPOSAL_THRESHOLD,
    CALIBRATION_SELECTION_METHODS,
    DEFAULT_SAM3_THRESHOLD_POLICY_BASE_ROOT,
    DEFAULT_SAM3_THRESHOLD_RESULT_ROOT,
    Sam3CalibrationConfig,
)
from .generator import generate_sam3_thresholds
from .inputs import prepare_calibration_inputs


def probability_argument(value: str) -> float:
    """Parse a CLI probability/threshold in the closed interval [0, 1]."""

    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number in [0, 1]") from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return parsed


def _progress_reporter(module: str):
    """Return a flushed stderr reporter without contaminating JSON stdout."""

    def report(stage: str, current: int, total: int, detail: str) -> None:
        percentage = 100.0 * current / total if total else 100.0
        print(
            f"[{module}][{stage}] {current}/{total} " f"({percentage:5.1f}%) {detail}",
            file=sys.stderr,
            flush=True,
        )

    return report


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--class-name", required=True)
    calibrate_normal_source = parser.add_mutually_exclusive_group(required=True)
    calibrate_normal_source.add_argument("--normal-images", nargs="+")
    calibrate_normal_source.add_argument(
        "--normal-dir", help="directory containing all train/good calibration images"
    )
    truth_source = parser.add_mutually_exclusive_group()
    truth_source.add_argument(
        "--truth-templates",
        nargs="+",
        help="optional sample-specific concrete templates, one per normal image",
    )
    truth_source.add_argument(
        "--truth-counts",
        nargs="+",
        type=int,
        help=(
            "per-image COUNT truths; expands a variable COUNT rule in memory "
            "without requiring sample-specific template files"
        ),
    )
    parser.add_argument(
        "--truth-subject", help="COUNT subject to replace when --truth-counts is used"
    )
    truth_source.add_argument(
        "--truth-subject-counts",
        nargs="+",
        action="append",
        metavar="SUBJECT_OR_COUNT",
        help="repeatable SUBJECT followed by one count per normal image",
    )
    parser.add_argument(
        "--template-root", default="/home/lxq/pro-innovation/templates/store"
    )
    parser.add_argument("--standard")
    parser.add_argument("--output")
    parser.add_argument(
        "--calibration-result-root",
        default=DEFAULT_SAM3_THRESHOLD_RESULT_ROOT,
        help="directory for full calibration reports; runtime policy stays minimal",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly allow replacing an existing threshold policy",
    )
    parser.add_argument(
        "--sam3-threshold-policy-root", default=DEFAULT_SAM3_THRESHOLD_POLICY_BASE_ROOT
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sam3-threshold",
        type=probability_argument,
        default=CALIBRATION_PROPOSAL_THRESHOLD,
        help="candidate-generation threshold for calibration in [0, 1] (default: 0.20)",
    )
    parser.add_argument(
        "--threshold-selection-method",
        choices=CALIBRATION_SELECTION_METHODS,
        default="confidence_gap_midpoint",
        help=(
            "confidence_gap_midpoint computes the Q95/Q05 interval midpoint "
            "and rounds it downward to the 0.10 grid"
        ),
    )
    parser.add_argument("--mask-dedup-iou", type=float, default=0.8)
    parser.add_argument("--mask-dedup-containment", type=float, default=0.85)
    parser.add_argument(
        "--mask-dedup", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--sam3-prompt", default="{category}")
    parser.add_argument("--sam3-repo", default="/home/lxq/models/SAM3")
    parser.add_argument(
        "--sam3-checkpoint", default="/home/lxq/weights/sam3/sam3.pt"
    )


def _inline_subject_counts(
    values: list[list[str]] | None,
) -> dict[str, tuple[int, ...]] | None:
    if values is None:
        return None
    counts: dict[str, tuple[int, ...]] = {}
    for raw_values in values:
        subject = raw_values[0]
        if subject in counts:
            raise ValueError(f"duplicate inline truth counts for {subject!r}")
        counts[subject] = tuple(int(value) for value in raw_values[1:])
    return counts


def run(args: argparse.Namespace) -> int:
    progress = _progress_reporter("sam3-threshold")
    inputs = prepare_calibration_inputs(
        standard_path=(
            args.standard
            or Path(args.template_root) / f"{args.class_name}.standard.rules"
        ),
        normal_images=args.normal_images,
        normal_dir=args.normal_dir,
        truth_templates=args.truth_templates,
        truth_counts=args.truth_counts,
        truth_subject=args.truth_subject,
        truth_subject_counts=_inline_subject_counts(args.truth_subject_counts),
        progress_callback=progress,
    )
    config = Sam3CalibrationConfig(
        repo_path=args.sam3_repo,
        checkpoint_path=args.sam3_checkpoint,
        device=args.device,
        proposal_threshold=args.sam3_threshold,
        selection_method=args.threshold_selection_method,
        prompt_template=args.sam3_prompt,
        mask_dedup_iou=args.mask_dedup_iou,
        mask_dedup_containment=args.mask_dedup_containment,
        mask_dedup=args.mask_dedup,
    )
    result = generate_sam3_thresholds(
        class_name=args.class_name,
        inputs=inputs,
        config=config,
        output=args.output,
        policy_root=args.sam3_threshold_policy_root,
        result_root=args.calibration_result_root,
        overwrite=args.overwrite,
        progress_callback=progress,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "complete" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate normal-only, per-object SAM3 thresholds"
    )
    add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ProInnovationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
