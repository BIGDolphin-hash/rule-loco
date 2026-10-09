"""Small command-line entry point for template handling and local inference."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

from .adapters import (
    ClipAttributeDetector,
    DinoV2PatchFeatureExtractor,
    LocalClipRuntime,
    Sam3Segmenter,
)
from .attributes import RoutedAttributeDetector
from .errors import ProInnovationError
from .evaluation import (
    MVTEC_AD,
    MVTEC_LOCO,
    VISA,
    SUPPORTED_DATASETS,
    evaluate_dataset,
    evaluate_screw_bag,
    validate_attribute_vocabulary,
)
from .mask_processing import MaskPostprocessor
from .patch_memory import (
    DEFAULT_PATCH_LAYER_FUSION,
    patch_memory_run_tag,
    DinoV2PatchImageScorer,
    PatchMemoryBank,
    build_patch_memory_from_images,
)
from .adapters.patch_dinov2 import (
    DEFAULT_PATCH_DINOV2_CHECKPOINT,
)
from .pipeline import LogicAnomalyPipeline
from .sam3_calibration import (
    CALIBRATION_PROPOSAL_THRESHOLD,
    calibrate_sam3_thresholds,
    load_sam3_threshold_policy,
    save_sam3_threshold_policy,
)
from .template_store import TemplateRepository
from .templates import (
    parse_template,
    collect_categories,
    serialize_template,
    to_generic_template,
    validate_generic_values_missing,
)


DEFAULT_PATCH_BANK_ROOT = "/home/lxq/pro-innovation/memory_banks/4-shot"
DEFAULT_SAM3_THRESHOLD_POLICY_ROOT = (
    "/home/lxq/pro-innovation/config/sam3_thresholds"
)
DEFAULT_DATASET_BASES = {
    MVTEC_LOCO: "/home/lxq/pro-innovation/datasets/mvtec_loco_anomaly_detection",
    MVTEC_AD: "/home/lxq/pro-innovation/datasets/mvtec_anomaly_detection",
    VISA: "/home/lxq/pro-innovation/datasets/visa",
}


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _write_or_print(text: str, output: str | None) -> None:
    if output:
        Path(output).write_text(text.rstrip() + "\n", encoding="utf-8")
    else:
        print(text)


def _probability_argument(value: str) -> float:
    """Parse a CLI probability/threshold in the closed interval [0, 1]."""

    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number in [0, 1]") from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return parsed


def _command_validate(args: argparse.Namespace) -> int:
    rules = parse_template(_read(args.template), allow_missing_values=args.generic)
    if args.generic:
        validate_generic_values_missing(rules)
    print(json.dumps({"valid": True, "rule_count": len(rules)}, ensure_ascii=False))
    return 0


def _command_make_generic(args: argparse.Namespace) -> int:
    standard = parse_template(_read(args.standard), allow_missing_values=False)
    text = serialize_template(to_generic_template(standard))
    _write_or_print(text, args.output)
    return 0


def _command_save_standard(args: argparse.Namespace) -> int:
    repository = TemplateRepository(args.template_root)
    standard_path, generic_path = repository.save_standard(
        args.class_name, _read(args.standard), overwrite=args.overwrite
    )
    print(
        json.dumps(
            {"standard": str(standard_path), "generic": str(generic_path)},
            ensure_ascii=False,
        )
    )
    return 0


def _load_vocabulary(path: str | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    value = json.loads(_read(path))
    if not isinstance(value, dict) or not all(
        isinstance(key, str)
        and "_" in key
        and isinstance(labels, list)
        and all(isinstance(label, str) for label in labels)
        for key, labels in value.items()
    ):
        raise ValueError(
            "attribute vocabulary must be a JSON object of string lists using "
            "<subject>_<attribute> field names"
        )
    return value


def _validate_class_name(class_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", class_name):
        raise ValueError(
            "class name may contain letters, numbers, underscores, and hyphens only"
        )
    return class_name


def _patch_bank_path(args: argparse.Namespace, class_name: str | None) -> Path:
    explicit = getattr(args, "patch_bank", None)
    if explicit:
        return Path(explicit).expanduser().resolve()
    if class_name is None:
        raise ValueError(
            "patch matching is enabled: provide --class-name or an explicit "
            "--patch-bank"
        )
    dataset_name = getattr(args, "dataset", MVTEC_LOCO)
    return (
        Path(args.patch_bank_root).expanduser().resolve()
        / dataset_name
        / f"{class_name}.{patch_memory_run_tag()}.npz"
    )


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sam3_threshold_configuration(
    args: argparse.Namespace, *, standard_path: str
) -> dict[str, Any]:
    standard = Path(standard_path).expanduser().resolve()
    return {
        "sam3_prompt": str(getattr(args, "sam3_prompt", "{category}")),
        "sam3_checkpoint": str(Path(args.sam3_checkpoint).expanduser().resolve()),
        "mask_dedup_iou_threshold": float(args.mask_dedup_iou),
        "mask_dedup_containment_threshold": float(args.mask_dedup_containment),
        "mask_minimum_area_pixels": int(args.mask_min_area),
        "mask_cleanup_enabled": bool(args.mask_cleanup),
        "mask_dedup_enabled": bool(args.mask_dedup),
        "standard_template": str(standard),
        "standard_template_sha256": _file_sha256(standard),
    }


def _resolve_sam3_threshold_runtime(
    *,
    class_name: str | None,
    enabled: bool,
    policy_root: str | Path,
    explicit_policy_path: str | None,
    expected_configuration: Mapping[str, Any],
) -> tuple[dict[str, float], str, str | None]:
    if not enabled:
        return {}, "global_only", None
    if explicit_policy_path:
        path = Path(explicit_policy_path).expanduser().resolve()
    elif class_name is not None:
        path = Path(policy_root).expanduser().resolve() / f"{class_name}.json"
    else:
        return {}, "global_only", None
    if not path.is_file():
        if explicit_policy_path:
            raise ValueError(f"SAM3 threshold policy not found: {path}")
        return {}, "global_only", None

    policy = load_sam3_threshold_policy(
        path, expected_class_name=class_name
    )
    mismatched = []
    for key, expected in expected_configuration.items():
        actual = policy.configuration.get(key)
        if isinstance(expected, float) and isinstance(actual, (int, float)):
            matches = abs(float(actual) - expected) <= 1e-12
        else:
            matches = actual == expected
        if not matches:
            mismatched.append(key)
    if mismatched:
        raise ValueError(
            "saved SAM3 threshold policy configuration does not match runtime: "
            + ", ".join(sorted(mismatched))
        )
    return (
        dict(policy.selected_thresholds),
        "few_shot_object_specific",
        str(policy.source_path),
    )


def _make_patch_scorer(
    args: argparse.Namespace,
    *,
    class_name: str | None,
) -> DinoV2PatchImageScorer | None:
    if not args.patch_matching:
        return None
    bank = PatchMemoryBank.load(
        _patch_bank_path(args, class_name),
        expected_class_name=class_name,
        expected_checkpoint_path=args.patch_dinov2_checkpoint,
    )
    return DinoV2PatchImageScorer(
        bank=bank,
        extractor=DinoV2PatchFeatureExtractor(
            checkpoint_path=args.patch_dinov2_checkpoint,
            device=args.device,
        ),
        query_batch_size=args.patch_query_batch_size,
        extraction_batch_size=args.patch_extraction_batch_size,
    )


def _add_patch_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--patch-matching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "enable or disable four-shot DINOv2 LogSAD patch matching "
            "(disabled by default)"
        ),
    )
    parser.add_argument(
        "--patch-bank-root",
        default=DEFAULT_PATCH_BANK_ROOT,
        help=(
            "four-shot bank root containing one dataset directory per layout; "
            f"banks are <dataset>/<class-name>.{patch_memory_run_tag()}.npz"
        ),
    )
    parser.add_argument(
        "--patch-bank",
        help="explicit patch memory path; overrides --patch-bank-root",
    )
    parser.add_argument(
        "--patch-dinov2-checkpoint",
        default=DEFAULT_PATCH_DINOV2_CHECKPOINT,
    )
    parser.add_argument("--patch-query-batch-size", type=int, default=256)
    parser.add_argument("--patch-extraction-batch-size", type=int, default=4)


def _add_rule_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rule-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "enable or disable template/SAM3 rule inference; disabling it "
            "requires --patch-matching and skips all rule dependencies"
        ),
    )


def _validate_runtime_branches(args: argparse.Namespace) -> None:
    if args.rule_inference:
        return
    if not args.patch_matching:
        raise ValueError("--no-rule-inference requires --patch-matching")


def _add_evaluation_scope_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--all-test-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "evaluate every dataset test split; evaluation scope is "
            "independent of the Patch switch"
        ),
    )


def _add_sam3_threshold_runtime_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--object-sam3-thresholds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "automatically use a saved per-object SAM3 threshold policy; "
            "disable for the global-threshold ablation"
        ),
    )
    parser.add_argument(
        "--sam3-threshold-policy-root",
        default=DEFAULT_SAM3_THRESHOLD_POLICY_ROOT,
    )
    parser.add_argument(
        "--sam3-threshold-policy",
        help="explicit saved SAM3 threshold policy path",
    )


def _command_infer(args: argparse.Namespace) -> int:
    _validate_runtime_branches(args)
    class_name = _validate_class_name(args.class_name) if args.class_name else None
    if args.rule_inference:
        if args.generic is None or args.standard is None:
            raise ValueError(
                "rule inference requires both --generic and --standard"
            )
        generic = parse_template(_read(args.generic), allow_missing_values=True)
        standard = parse_template(_read(args.standard), allow_missing_values=False)
        categories = collect_categories(generic)
        sam3_category_thresholds, sam3_threshold_policy, sam3_policy_path = (
            _resolve_sam3_threshold_runtime(
                class_name=class_name,
                enabled=args.object_sam3_thresholds,
                policy_root=args.sam3_threshold_policy_root,
                explicit_policy_path=args.sam3_threshold_policy,
                expected_configuration=_sam3_threshold_configuration(
                    args, standard_path=args.standard
                ),
            )
        )
        segmenter = Sam3Segmenter(
            repo_path=args.sam3_repo,
            checkpoint_path=args.sam3_checkpoint,
            device=args.device,
            confidence_threshold=args.sam3_threshold,
            category_thresholds=sam3_category_thresholds,
            prompt_template=args.sam3_prompt,
        )
        vocabulary = _load_vocabulary(args.attribute_vocabulary)
        validate_attribute_vocabulary(standard, vocabulary)
        clip_runtime = LocalClipRuntime(
            repo_path=args.clip_repo,
            checkpoint_path=args.clip_checkpoint,
            device=args.device,
        )
        clip_detector = ClipAttributeDetector(
            repo_path=args.clip_repo,
            checkpoint_path=args.clip_checkpoint,
            vocabulary=vocabulary,
            device=args.device,
            confidence_threshold=args.clip_threshold,
            runtime=clip_runtime,
        )
        detector = RoutedAttributeDetector(
            semantic_detector=clip_detector if vocabulary else None,
            vocabulary=vocabulary,
        )
        mask_processor = MaskPostprocessor(
            dedup_iou_threshold=args.mask_dedup_iou,
            dedup_containment_threshold=args.mask_dedup_containment,
            minimum_area_pixels=args.mask_min_area,
            cleanup_enabled=args.mask_cleanup,
            dedup_enabled=args.mask_dedup,
        )
    else:
        generic = ()
        standard = ()
        segmenter = None
        detector = None
        mask_processor = None
        sam3_category_thresholds = {}
        sam3_threshold_policy = "disabled"
        sam3_policy_path = None
    patch_scorer = _make_patch_scorer(
        args,
        class_name=class_name,
    )
    pipeline = LogicAnomalyPipeline(
        segmenter,
        detector,
        mask_processor=mask_processor,
        patch_scorer=patch_scorer,
        grouping_tolerance=args.grouping_tolerance,
        rule_inference_enabled=args.rule_inference,
    )
    report = pipeline.run(args.image, generic, standard)
    report_payload = report.to_dict()
    report_payload["patch_matching_enabled"] = bool(args.patch_matching)
    report_payload["rule_inference_enabled"] = bool(args.rule_inference)
    report_payload["patch_primary_layer"] = None
    report_payload["patch_layer_fusion"] = (
        DEFAULT_PATCH_LAYER_FUSION if args.patch_matching else None
    )
    report_payload["sam3_threshold_policy"] = sam3_threshold_policy
    report_payload["sam3_global_fallback_threshold"] = (
        args.sam3_threshold if args.rule_inference else None
    )
    report_payload["clip_attribute_confidence_threshold"] = (
        args.clip_threshold if args.rule_inference else None
    )
    report_payload["sam3_category_thresholds"] = sam3_category_thresholds
    report_payload["sam3_threshold_policy_path"] = sam3_policy_path
    payload = json.dumps(report_payload, ensure_ascii=False, indent=2)
    _write_or_print(payload, args.output)
    if report.is_anomaly is None:
        return 2
    return 1 if report.is_anomaly else 0


def _command_evaluate_screw_bag(args: argparse.Namespace) -> int:
    _validate_runtime_branches(args)
    if args.rule_inference:
        sam3_category_thresholds, sam3_threshold_policy, sam3_policy_path = (
            _resolve_sam3_threshold_runtime(
                class_name="screw_bag",
                enabled=args.object_sam3_thresholds,
                policy_root=args.sam3_threshold_policy_root,
                explicit_policy_path=args.sam3_threshold_policy,
                expected_configuration=_sam3_threshold_configuration(
                    args, standard_path=args.standard
                ),
            )
        )
    else:
        sam3_category_thresholds = {}
        sam3_threshold_policy = "disabled"
        sam3_policy_path = None
    summary = evaluate_screw_bag(
                dataset_root=args.dataset_root,
                generic_path=args.generic,
                standard_path=args.standard,
                result_root=args.result_root,
                sam3_threshold=args.sam3_threshold,
                sam3_category_thresholds=sam3_category_thresholds,
                sam3_threshold_policy=sam3_threshold_policy,
                sam3_threshold_policy_path=sam3_policy_path,
                sam3_repo=args.sam3_repo,
                sam3_checkpoint=args.sam3_checkpoint,
                clip_repo=args.clip_repo,
                clip_checkpoint=args.clip_checkpoint,
                clip_threshold=args.clip_threshold,
                device=args.device,
                sam3_prompt=args.sam3_prompt,
                mask_dedup_iou=args.mask_dedup_iou,
                mask_dedup_containment=args.mask_dedup_containment,
                mask_min_area=args.mask_min_area,
                mask_cleanup=args.mask_cleanup,
                mask_dedup=args.mask_dedup,
                rule_inference_enabled=args.rule_inference,
                patch_enabled=args.patch_matching,
                patch_bank_path=(
                    str(_patch_bank_path(args, "screw_bag"))
                    if args.patch_matching
                    else None
                ),
                patch_dinov2_checkpoint=args.patch_dinov2_checkpoint,
                patch_query_batch_size=args.patch_query_batch_size,
                patch_extraction_batch_size=args.patch_extraction_batch_size,
                include_structural_anomalies=args.all_test_splits,
                restart=args.restart,
                test_split=args.test_split,
                test_splits=args.test_splits,
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failure_count"] == 0 else 2


def _command_evaluate(args: argparse.Namespace) -> int:
    _validate_runtime_branches(args)
    class_name = _validate_class_name(args.class_name)
    dataset_base = args.dataset_base or DEFAULT_DATASET_BASES[args.dataset]
    dataset_root = args.dataset_root or str(
        Path(dataset_base)
        if args.dataset == VISA
        else Path(dataset_base) / class_name / "test"
    )
    generic_path = args.generic or str(
        Path(args.template_root) / f"{class_name}.generic.rules"
    )
    standard_path = args.standard or str(
        Path(args.template_root) / f"{class_name}.standard.rules"
    )
    if args.rule_inference:
        sam3_category_thresholds, sam3_threshold_policy, sam3_policy_path = (
            _resolve_sam3_threshold_runtime(
                class_name=class_name,
                enabled=args.object_sam3_thresholds,
                policy_root=args.sam3_threshold_policy_root,
                explicit_policy_path=args.sam3_threshold_policy,
                expected_configuration=_sam3_threshold_configuration(
                    args, standard_path=standard_path
                ),
            )
        )
        vocabulary = _load_vocabulary(args.attribute_vocabulary)
    else:
        sam3_category_thresholds = {}
        sam3_threshold_policy = "disabled"
        sam3_policy_path = None
        vocabulary = {}
    summary = evaluate_dataset(
                class_name=class_name,
                dataset_root=dataset_root,
                generic_path=generic_path,
                standard_path=standard_path,
                result_root=args.result_root,
                sam3_threshold=args.sam3_threshold,
                sam3_category_thresholds=sam3_category_thresholds,
                sam3_threshold_policy=sam3_threshold_policy,
                sam3_threshold_policy_path=sam3_policy_path,
                sam3_repo=args.sam3_repo,
                sam3_checkpoint=args.sam3_checkpoint,
                clip_repo=args.clip_repo,
                clip_checkpoint=args.clip_checkpoint,
                clip_threshold=args.clip_threshold,
                attribute_vocabulary=vocabulary,
                device=args.device,
                grouping_tolerance=args.grouping_tolerance,
                sam3_prompt=args.sam3_prompt,
                mask_dedup_iou=args.mask_dedup_iou,
                mask_dedup_containment=args.mask_dedup_containment,
                mask_min_area=args.mask_min_area,
                mask_cleanup=args.mask_cleanup,
                mask_dedup=args.mask_dedup,
                rule_inference_enabled=args.rule_inference,
                patch_enabled=args.patch_matching,
                patch_bank_path=(
                    str(_patch_bank_path(args, class_name))
                    if args.patch_matching
                    else None
                ),
                patch_dinov2_checkpoint=args.patch_dinov2_checkpoint,
                patch_query_batch_size=args.patch_query_batch_size,
                patch_extraction_batch_size=args.patch_extraction_batch_size,
                include_structural_anomalies=args.all_test_splits,
                restart=args.restart,
                test_split=args.test_split,
                test_splits=args.test_splits,
                dataset_name=args.dataset,
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failure_count"] == 0 else 2


def _command_build_patch_memory(args: argparse.Namespace) -> int:
    class_name = _validate_class_name(args.class_name)
    calibration_dir = Path(args.calibration_dir).expanduser().resolve()
    calibration_images = tuple(
        path
        for path in sorted(calibration_dir.iterdir())
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(calibration_images) < 2:
        raise ValueError(
            "LogSAD calibration directory must contain at least two JPG or PNG images"
        )
    output = args.output or str(
        Path(args.patch_bank_root).expanduser().resolve()
        / f"{class_name}.{patch_memory_run_tag()}.npz"
    )
    bank = build_patch_memory_from_images(
        class_name=class_name,
        normal_images=args.normal_images,
        calibration_images=calibration_images,
        output_path=output,
        checkpoint_path=args.patch_dinov2_checkpoint,
        device=args.device,
        query_batch_size=args.patch_query_batch_size,
        extraction_batch_size=args.patch_extraction_batch_size,
    )
    print(
        json.dumps(
            {
                "patch_bank": str(Path(output).expanduser().resolve()),
                "class_name": class_name,
                "normal_images": list(bank.normal_images),
                "calibration_images": list(bank.calibration_images),
                "calibration_count": len(bank.calibration_scores),
                "layers": list(bank.layers),
                "input_size": bank.input_size,
                "feature_grid": list(bank.feature_grid),
                "matching": "global_cosine_nearest_neighbor",
                "layer_aggregation": "mean_anomaly_map",
                "image_aggregation": "spatial_max",
                "coreset_enabled": False,
                "score_mean": bank.score_mean,
                "score_unbiased_std": bank.score_unbiased_std,
                "calibration_mode": bank.calibration_mode,
                "memory_scope": "full_image_only",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _command_calibrate_sam3(args: argparse.Namespace) -> int:
    class_name = _validate_class_name(args.class_name)
    standard_path = args.standard or str(
        Path(args.template_root) / f"{class_name}.standard.rules"
    )
    standard_rules = parse_template(
        Path(standard_path).read_text(encoding="utf-8"),
        allow_missing_values=False,
    )
    normal_images = tuple(
        Path(path).expanduser().resolve() for path in args.normal_images
    )
    if len(set(normal_images)) != 4:
        raise ValueError("SAM3 calibration requires four distinct normal images")
    missing_images = [str(path) for path in normal_images if not path.is_file()]
    if missing_images:
        raise ValueError(
            "normal calibration images not found: " + ", ".join(missing_images)
        )

    if args.truth_templates and (args.truth_counts or args.truth_subject_counts):
        raise ValueError(
            "use either --truth-templates or inline truth counts, not both"
        )
    if args.truth_subject_counts:
        count_subjects = {
            rule.subject
            for rule in standard_rules
            if rule.task.value in {"COUNT", "LENGTH", "AREA"} and rule.subject
        }
        subject_counts: dict[str, tuple[int, int, int, int]] = {}
        for raw_values in args.truth_subject_counts:
            subject = raw_values[0]
            if subject in subject_counts:
                raise ValueError(f"duplicate inline truth counts for {subject!r}")
            if subject not in count_subjects:
                raise ValueError(
                    f"--truth-subject-counts subject {subject!r} is not a COUNT "
                    "subject in the standard template"
                )
            values = tuple(int(value) for value in raw_values[1:])
            if any(value < 0 for value in values):
                raise ValueError(
                    "inline truth count values must be non-negative integers"
                )
            subject_counts[subject] = values  # type: ignore[assignment]
        variable_count_subjects = {
            rule.subject
            for rule in standard_rules
            if (
                rule.task.value in {"COUNT", "LENGTH", "AREA"}
                and rule.subject
                and isinstance(
                    rule.expected if rule.task.value == "COUNT" else rule.count,
                    tuple,
                )
                and len(
                    rule.expected if rule.task.value == "COUNT" else rule.count
                )
                > 1
            )
        }
        missing_subjects = sorted(variable_count_subjects - subject_counts.keys())
        if missing_subjects:
            raise ValueError(
                "missing inline truth counts for variable COUNT subjects: "
                + ", ".join(missing_subjects)
            )
        truth_rules = tuple(
            tuple(
                replace(
                    rule,
                    expected=(subject_counts[rule.subject][sample_index],)
                    if rule.task.value == "COUNT"
                    else rule.expected,
                    count=(subject_counts[rule.subject][sample_index],)
                    if rule.task.value in {"LENGTH", "AREA"}
                    else rule.count,
                )
                if (
                    rule.task.value in {"COUNT", "LENGTH", "AREA"}
                    and rule.subject in subject_counts
                )
                else rule
                for rule in standard_rules
            )
            for sample_index in range(4)
        )
        truth_paths = None
    elif args.truth_counts:
        variable_count_subjects = {
            rule.subject
            for rule in standard_rules
            if (
                rule.task.value in {"COUNT", "LENGTH", "AREA"}
                and rule.subject
                and isinstance(
                    rule.expected if rule.task.value == "COUNT" else rule.count,
                    tuple,
                )
                and len(
                    rule.expected if rule.task.value == "COUNT" else rule.count
                )
                > 1
            )
        }
        if args.truth_subject:
            if args.truth_subject not in {
                rule.subject
                for rule in standard_rules
                if rule.task.value in {"COUNT", "LENGTH", "AREA"}
            }:
                raise ValueError(
                    f"--truth-subject {args.truth_subject!r} is not a COUNT subject "
                    "in the standard template"
                )
            truth_subject = args.truth_subject
        elif len(variable_count_subjects) == 1:
            truth_subject = next(iter(variable_count_subjects))
        else:
            raise ValueError(
                "--truth-counts needs --truth-subject when the standard template "
                "has zero or multiple variable COUNT subjects"
            )
        truth_counts = tuple(int(value) for value in args.truth_counts)
        if any(value < 0 for value in truth_counts):
            raise ValueError("--truth-counts values must be non-negative integers")
        truth_rules = tuple(
            tuple(
                replace(rule, expected=(count,))
                if rule.task.value == "COUNT" and rule.subject == truth_subject
                else replace(rule, count=(count,))
                if rule.task.value in {"LENGTH", "AREA"}
                and rule.subject == truth_subject
                else rule
                for rule in standard_rules
            )
            for count in truth_counts
        )
        truth_paths = None
    elif args.truth_templates:
        truth_paths = tuple(
            Path(path).expanduser().resolve() for path in args.truth_templates
        )
        missing_templates = [str(path) for path in truth_paths if not path.is_file()]
        if missing_templates:
            raise ValueError(
                "truth templates not found: " + ", ".join(missing_templates)
            )
        truth_rules = tuple(
            parse_template(path.read_text(encoding="utf-8"), allow_missing_values=False)
            for path in truth_paths
        )
    else:
        truth_paths = (Path(standard_path).expanduser().resolve(),) * 4
        truth_rules = tuple(standard_rules for _ in range(4))

    segmenter = Sam3Segmenter(
        repo_path=args.sam3_repo,
        checkpoint_path=args.sam3_checkpoint,
        device=args.device,
        confidence_threshold=CALIBRATION_PROPOSAL_THRESHOLD,
        prompt_template=args.sam3_prompt,
    )
    result = calibrate_sam3_thresholds(
        class_name=class_name,
        normal_images=normal_images,
        truth_rules_by_image=truth_rules,
        segmenter=segmenter,
        mask_processor=MaskPostprocessor(
            dedup_iou_threshold=args.mask_dedup_iou,
            dedup_containment_threshold=args.mask_dedup_containment,
            minimum_area_pixels=args.mask_min_area,
            cleanup_enabled=args.mask_cleanup,
            dedup_enabled=args.mask_dedup,
        ),
        proposal_threshold=args.sam3_threshold,
        configuration=_sam3_threshold_configuration(
            args, standard_path=standard_path
        ),
        truth_template_paths=truth_paths,
    )
    output = args.output or str(
        Path(args.sam3_threshold_policy_root) / f"{class_name}.json"
    )
    saved_path = save_sam3_threshold_policy(result, output)
    result["saved_path"] = str(saved_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "complete" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pro-innovation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a rule template")
    validate.add_argument("template")
    validate.add_argument("--generic", action="store_true")
    validate.set_defaults(func=_command_validate)

    make_generic = subparsers.add_parser(
        "make-generic", help="strip expected values from a standard template"
    )
    make_generic.add_argument("standard")
    make_generic.add_argument("--output")
    make_generic.set_defaults(func=_command_make_generic)

    save = subparsers.add_parser(
        "save-standard", help="validate and save a standard/generic template pair"
    )
    save.add_argument("class_name")
    save.add_argument("standard")
    save.add_argument("--template-root", default="templates/store")
    save.add_argument("--overwrite", action="store_true")
    save.set_defaults(func=_command_save_standard)

    build_patch = subparsers.add_parser(
        "build-patch-memory",
        help=(
            "build LogSAD DINOv2 Block 6/12/18/24 patch memory from four "
            "normal images and calibrate it on validation/good"
        ),
    )
    build_patch.add_argument("--class-name", required=True)
    build_patch.add_argument("--normal-images", nargs=4, required=True)
    build_patch.add_argument(
        "--calibration-dir",
        required=True,
        help="directory containing the independent normal validation PNG images",
    )
    build_patch.add_argument("--output")
    build_patch.add_argument(
        "--patch-bank-root",
        default=str(Path(DEFAULT_PATCH_BANK_ROOT) / MVTEC_LOCO),
    )
    build_patch.add_argument("--device", default="cuda")
    build_patch.add_argument(
        "--patch-dinov2-checkpoint",
        default=DEFAULT_PATCH_DINOV2_CHECKPOINT,
    )
    build_patch.add_argument("--patch-query-batch-size", type=int, default=256)
    build_patch.add_argument("--patch-extraction-batch-size", type=int, default=4)
    build_patch.set_defaults(func=_command_build_patch_memory)

    calibrate_sam3 = subparsers.add_parser(
        "calibrate-sam3-thresholds",
        help=(
            "derive and save fixed per-object SAM3 thresholds from exactly "
            "four normal truth samples"
        ),
    )
    calibrate_sam3.add_argument("--class-name", required=True)
    calibrate_sam3.add_argument("--normal-images", nargs=4, required=True)
    calibrate_sam3.add_argument(
        "--truth-templates",
        nargs=4,
        help=(
            "four sample-specific concrete templates; required when the shared "
            "standard template contains multiple allowed COUNT values"
        ),
    )
    calibrate_sam3.add_argument(
        "--truth-counts",
        nargs=4,
        type=int,
        help=(
            "four per-image COUNT truths; expands a variable COUNT rule in memory "
            "without requiring sample-specific template files"
        ),
    )
    calibrate_sam3.add_argument(
        "--truth-subject",
        help="COUNT subject to replace when --truth-counts is used",
    )
    calibrate_sam3.add_argument(
        "--truth-subject-counts",
        nargs=5,
        action="append",
        metavar=("SUBJECT", "K1", "K2", "K3", "K4"),
        help=(
            "repeatable per-subject inline truths, for example "
            "--truth-subject-counts orange_clip 4 6 10 10"
        ),
    )
    calibrate_sam3.add_argument(
        "--template-root",
        default="/home/lxq/pro-innovation/templates/store",
    )
    calibrate_sam3.add_argument("--standard")
    calibrate_sam3.add_argument("--output")
    calibrate_sam3.add_argument(
        "--sam3-threshold-policy-root",
        default=DEFAULT_SAM3_THRESHOLD_POLICY_ROOT,
    )
    calibrate_sam3.add_argument("--device", default="cuda")
    calibrate_sam3.add_argument(
        "--sam3-threshold",
        type=_probability_argument,
        default=CALIBRATION_PROPOSAL_THRESHOLD,
        help=(
            "candidate-generation threshold for calibration in [0, 1] "
            "(default: 0.10)"
        ),
    )
    calibrate_sam3.add_argument("--mask-dedup-iou", type=float, default=0.80)
    calibrate_sam3.add_argument(
        "--mask-dedup-containment", type=float, default=0.85
    )
    calibrate_sam3.add_argument("--mask-min-area", type=int, default=16)
    calibrate_sam3.add_argument(
        "--mask-dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    calibrate_sam3.add_argument(
        "--mask-cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    calibrate_sam3.add_argument("--sam3-prompt", default="{category}")
    calibrate_sam3.add_argument("--sam3-repo", default="/home/lxq/models/SAM3")
    calibrate_sam3.add_argument(
        "--sam3-checkpoint", default="/home/lxq/weights/sam3/sam3.pt"
    )
    calibrate_sam3.set_defaults(func=_command_calibrate_sam3)

    infer = subparsers.add_parser(
        "infer", help="run local SAM3 rules with optional Patch matching"
    )
    infer.add_argument(
        "--class-name",
        help="class identity used to select the default Patch bank",
    )
    infer.add_argument("--image", required=True)
    infer.add_argument("--generic")
    infer.add_argument("--standard")
    infer.add_argument("--output")
    infer.add_argument("--attribute-vocabulary")
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--grouping-tolerance", type=float, default=0.10)
    infer.add_argument("--sam3-threshold", type=float, default=0.40)
    infer.add_argument("--mask-dedup-iou", type=float, default=0.80)
    infer.add_argument("--mask-dedup-containment", type=float, default=0.85)
    infer.add_argument("--mask-min-area", type=int, default=16)
    infer.add_argument(
        "--mask-dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable same-category duplicate-mask suppression",
    )
    infer.add_argument(
        "--mask-cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable largest-component cleanup and minimum-area filtering",
    )
    infer.add_argument("--sam3-prompt", default="{category}")
    infer.add_argument("--sam3-repo", default="/home/lxq/models/SAM3")
    infer.add_argument("--sam3-checkpoint", default="/home/lxq/weights/sam3/sam3.pt")
    _add_sam3_threshold_runtime_arguments(infer)
    infer.add_argument("--clip-repo", default="/home/lxq/models/CLIP")
    infer.add_argument(
        "--clip-checkpoint", default="/home/lxq/weights/CLIP/ViT-L-14.pt"
    )
    infer.add_argument(
        "--clip-threshold",
        type=float,
        default=0.35,
        help=(
            "minimum confidence for CLIP semantic-attribute predictions; "
            "this does not enable SAM3 mask category verification"
        ),
    )
    _add_patch_runtime_arguments(infer)
    _add_rule_runtime_arguments(infer)
    infer.set_defaults(func=_command_infer)

    evaluate_generic = subparsers.add_parser(
        "evaluate",
        help="evaluate a MVTec LOCO, MVTec AD, or VisA class",
    )
    evaluate_generic.add_argument("--class-name", required=True)
    evaluate_generic.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default=MVTEC_LOCO,
        help=(
            "dataset layout: mvtec_loco uses fixed logical/structural splits; "
            "mvtec_ad discovers every non-good defect folder; visa reads the "
            "official split_csv/1cls.csv test rows"
        ),
    )
    evaluate_generic.add_argument(
        "--dataset-base",
        help=(
            "dataset root containing one directory per class; defaults to the "
            "project datasets symlink selected by --dataset"
        ),
    )
    evaluate_generic.add_argument(
        "--dataset-root",
        help=(
            "override the dataset path; for VisA this is the root containing "
            "split_csv, otherwise <dataset-base>/<class-name>/test"
        ),
    )
    evaluate_generic.add_argument(
        "--template-root",
        default="/home/lxq/pro-innovation/templates/store",
    )
    evaluate_generic.add_argument(
        "--generic",
        help="override the default <template-root>/<class-name>.generic.rules",
    )
    evaluate_generic.add_argument(
        "--standard",
        help="override the default <template-root>/<class-name>.standard.rules",
    )
    evaluate_generic.add_argument(
        "--result-root", default="/home/lxq/pro-innovation/result"
    )
    evaluate_generic.add_argument("--attribute-vocabulary")
    evaluate_generic.add_argument("--device", default="cuda")
    evaluate_generic.add_argument("--grouping-tolerance", type=float, default=0.10)
    evaluate_generic.add_argument("--sam3-threshold", type=float, default=0.40)
    evaluate_generic.add_argument("--mask-dedup-iou", type=float, default=0.80)
    evaluate_generic.add_argument(
        "--mask-dedup-containment", type=float, default=0.85
    )
    evaluate_generic.add_argument("--mask-min-area", type=int, default=16)
    evaluate_generic.add_argument(
        "--mask-dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable same-category duplicate-mask suppression",
    )
    evaluate_generic.add_argument(
        "--mask-cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable largest-component cleanup and minimum-area filtering",
    )
    evaluate_generic.add_argument("--sam3-prompt", default="{category}")
    evaluate_generic.add_argument("--sam3-repo", default="/home/lxq/models/SAM3")
    evaluate_generic.add_argument(
        "--sam3-checkpoint", default="/home/lxq/weights/sam3/sam3.pt"
    )
    _add_sam3_threshold_runtime_arguments(evaluate_generic)
    evaluate_generic.add_argument("--clip-repo", default="/home/lxq/models/CLIP")
    evaluate_generic.add_argument(
        "--clip-checkpoint", default="/home/lxq/weights/CLIP/ViT-L-14.pt"
    )
    evaluate_generic.add_argument(
        "--clip-threshold",
        type=float,
        default=0.35,
        help=(
            "minimum confidence for CLIP semantic-attribute predictions; "
            "this does not enable SAM3 mask category verification"
        ),
    )
    _add_patch_runtime_arguments(evaluate_generic)
    _add_rule_runtime_arguments(evaluate_generic)
    _add_evaluation_scope_arguments(evaluate_generic)
    evaluate_generic.add_argument(
        "--restart",
        action="store_true",
        help="discard the parameter-matched progress file and run from the start",
    )
    evaluate_generic.add_argument(
        "--test-split",
        help="evaluate only one test folder; models are loaded once",
    )
    evaluate_generic.add_argument(
        "--test-splits",
        nargs="+",
        help="evaluate the selected test folders; models are loaded once",
    )
    evaluate_generic.set_defaults(func=_command_evaluate)

    evaluate = subparsers.add_parser(
        "evaluate-screw-bag",
        help="evaluate screw_bag with optional Patch matching and compute AUROC",
    )
    evaluate.add_argument(
        "--dataset-root",
        default="/home/lxq/Data/mvtec_loco_anomaly_detection/screw_bag/test",
    )
    evaluate.add_argument(
        "--generic",
        default="/home/lxq/pro-innovation/templates/store/screw_bag.generic.rules",
    )
    evaluate.add_argument(
        "--standard",
        default="/home/lxq/pro-innovation/templates/store/screw_bag.standard.rules",
    )
    evaluate.add_argument(
        "--result-root", default="/home/lxq/pro-innovation/result"
    )
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--sam3-threshold", type=float, default=0.40)
    evaluate.add_argument("--mask-dedup-iou", type=float, default=0.80)
    evaluate.add_argument(
        "--mask-dedup-containment", type=float, default=0.85
    )
    evaluate.add_argument("--mask-min-area", type=int, default=16)
    evaluate.add_argument(
        "--mask-dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable same-category duplicate-mask suppression",
    )
    evaluate.add_argument(
        "--mask-cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable largest-component cleanup and minimum-area filtering",
    )
    evaluate.add_argument("--sam3-repo", default="/home/lxq/models/SAM3")
    evaluate.add_argument(
        "--sam3-checkpoint", default="/home/lxq/weights/sam3/sam3.pt"
    )
    evaluate.add_argument("--sam3-prompt", default="{category}")
    _add_sam3_threshold_runtime_arguments(evaluate)
    evaluate.add_argument("--clip-repo", default="/home/lxq/models/CLIP")
    evaluate.add_argument(
        "--clip-checkpoint", default="/home/lxq/weights/CLIP/ViT-L-14.pt"
    )
    evaluate.add_argument(
        "--clip-threshold",
        type=float,
        default=0.35,
        help=(
            "minimum confidence for CLIP semantic-attribute predictions; "
            "this does not enable SAM3 mask category verification"
        ),
    )
    _add_patch_runtime_arguments(evaluate)
    _add_rule_runtime_arguments(evaluate)
    _add_evaluation_scope_arguments(evaluate)
    evaluate.add_argument(
        "--restart",
        action="store_true",
        help="discard the parameter-matched progress file and run from the start",
    )
    evaluate.add_argument(
        "--test-split",
        choices=("good", "logical_anomalies", "structural_anomalies"),
        help="evaluate only one test folder; models are loaded once",
    )
    evaluate.add_argument(
        "--test-splits",
        nargs="+",
        choices=("good", "logical_anomalies", "structural_anomalies"),
        help="evaluate the selected test folders; models are loaded once",
    )
    evaluate.set_defaults(func=_command_evaluate_screw_bag)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ProInnovationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
