"""Small command-line entry point for template handling and local inference."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


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
    build_sam3_dataset_cache,
    evaluate_dataset,
    evaluate_screw_bag,
    validate_attribute_vocabulary,
)
from .mask_processing import MaskPostprocessor
from .image_validation import filter_decodable_images
from .memory_modules.global_patch import (
    PATCH_FEATURE_GLOBAL,
    PATCH_MEMORY_PER_IMAGE_STORAGE,
    DEFAULT_CORESET_RATIO,
    DEFAULT_CORESET_SEED,
    DEFAULT_PATCH_LAYER_FUSION,
    patch_memory_run_tag,
    DinoV2PatchImageScorer,
    PatchMemoryBank,
    build_patch_memory_from_images,
)
from .adapters.patch_dinov2 import (
    DEFAULT_PATCH_DINOV2_CHECKPOINT,
)
from .pipeline import (
    LogicAnomalyPipeline,
    RULE_MEMORY_SCHEDULE_BINARY_SELECTIVE,
    RULE_MEMORY_SCHEDULES,
)
from .sam3_calibration import (
    CALIBRATION_SELECTION_METHODS,
    CALIBRATION_PROPOSAL_THRESHOLD,
    calibrate_sam3_thresholds,
    load_sam3_threshold_policy,
    save_sam3_threshold_calibration_result,
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

DEFAULT_PATCH_BANK_ROOT = "/home/lxq/pro-innovation/memory_banks/Global patch/full"

DEFAULT_SAM3_THRESHOLD_POLICY_BASE_ROOT = (
    "/home/lxq/pro-innovation/config/sam3_thresholds"
)
DEFAULT_SAM3_THRESHOLD_POLICY_ROOT = (
    f"{DEFAULT_SAM3_THRESHOLD_POLICY_BASE_ROOT}/" "confidence_gap_midpoint_4-shot"
)
DEFAULT_SAM3_THRESHOLD_RESULT_ROOT = (
    "/home/lxq/pro-innovation/results/sam3_threshold_calibration"
)
DEFAULT_SAM3_CACHE_ROOT = "/home/lxq/pro-innovation/cache/sam3"
DEFAULT_DATASET_BASES = {
    MVTEC_LOCO: "/home/lxq/pro-innovation/datasets/mvtec_loco_anomaly_detection",
    MVTEC_AD: "/home/lxq/pro-innovation/datasets/mvtec_anomaly_detection",
    VISA: "/home/lxq/pro-innovation/datasets/visa",
}


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _resolve_normal_images(args: argparse.Namespace) -> tuple[Path, ...]:
    explicit = getattr(args, "normal_images", None)
    directory = getattr(args, "normal_dir", None)
    if bool(explicit) == bool(directory):
        raise ValueError("provide exactly one of --normal-images or --normal-dir")
    if directory:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"normal image directory not found: {root}")
        paths = tuple(
            path.resolve()
            for path in sorted(root.iterdir())
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    else:
        paths = tuple(Path(path).expanduser().resolve() for path in explicit)
    if len(paths) < 1:
        raise ValueError("normal memory requires at least one image")
    if len(set(paths)) != len(paths):
        raise ValueError("normal memory images must be distinct")
    return paths


def _write_or_print(text: str, output: str | None) -> None:
    if output:
        Path(output).write_text(text.rstrip() + "\n", encoding="utf-8")
    else:
        print(text)


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
        / "global"
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
    expected_categories: Sequence[str] | None = None,
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

    policy = load_sam3_threshold_policy(path, expected_class_name=class_name)
    if expected_categories is not None:
        expected_category_set = set(expected_categories)
        saved_category_set = set(policy.selected_thresholds)
        extra = sorted(saved_category_set - expected_category_set)
        if extra:
            raise ValueError(
                "saved SAM3 threshold policy categories do not match runtime: "
                + "extra="
                + ",".join(extra)
            )
    mismatched = []
    if policy.configuration:
        for key, expected in expected_configuration.items():
            if key == "standard_template_sha256":
                continue
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
    extractor = DinoV2PatchFeatureExtractor(
        checkpoint_path=args.patch_dinov2_checkpoint,
        device=args.device,
    )
    bank = PatchMemoryBank.load(
        _patch_bank_path(args, class_name),
        expected_class_name=class_name,
        expected_checkpoint_path=args.patch_dinov2_checkpoint,
    )
    return DinoV2PatchImageScorer(
        bank=bank,
        extractor=extractor,
        query_batch_size=args.patch_query_batch_size,
        extraction_batch_size=args.patch_extraction_batch_size,
    )


def _add_patch_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--patch-matching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "enable or disable full-normal Coreset DINOv2 LogSAD matching "
            "(disabled by default)"
        ),
    )
    parser.add_argument(
        "--patch-feature-mode",
        choices=(PATCH_FEATURE_GLOBAL,),
        default=PATCH_FEATURE_GLOBAL,
        help=("use full-image Global Patch memory matching"),
    )
    parser.add_argument(
        "--patch-bank-root",
        default=DEFAULT_PATCH_BANK_ROOT,
        help=(
            "texture bank root containing one dataset directory per layout; "
            f"banks are <dataset>/global/<class-name>.{patch_memory_run_tag()}.npz"
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
            "requires --patch-matching"
        ),
    )
    parser.add_argument(
        "--rule-memory-schedule",
        dest="rule_memory_schedule",
        choices=RULE_MEMORY_SCHEDULES,
        default=RULE_MEMORY_SCHEDULE_BINARY_SELECTIVE,
        help=(
            "binary-selective stops at the first failed rule with score 1 "
            "and runs memory detection only when all rules pass; "
            "binary-parallel runs both branches concurrently and takes the "
            "maximum of the 0/1 rule score and memory score"
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


def _add_sam3_cache_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sam3-cache-root",
        default=DEFAULT_SAM3_CACHE_ROOT,
        help="directory containing reusable per-image, per-prompt SAM3 proposals",
    )
    parser.add_argument(
        "--sam3-cache-mode",
        choices=Sam3Segmenter.CACHE_MODES,
        default="read-write",
        help=(
            "off disables caching; read requires a hit; read-write reuses hits "
            "and fills misses; refresh recomputes and replaces entries"
        ),
    )


def _command_infer(args: argparse.Namespace) -> int:
    class_name = _validate_class_name(args.class_name) if args.class_name else None
    sam3_category_thresholds, sam3_threshold_policy, sam3_policy_path = (
        _resolve_sam3_threshold_runtime(
            class_name=class_name,
            enabled=args.object_sam3_thresholds,
            policy_root=args.sam3_threshold_policy_root,
            explicit_policy_path=args.sam3_threshold_policy,
            expected_configuration=(
                _sam3_threshold_configuration(args, standard_path=args.standard)
                if args.standard is not None
                else {}
            ),
            expected_categories=None,
        )
    )
    _validate_runtime_branches(args)
    mask_inference_enabled = args.rule_inference
    if mask_inference_enabled:
        if args.generic is None or (args.rule_inference and args.standard is None):
            raise ValueError(
                "mask-based inference requires --generic; rule inference also requires --standard"
            )
        generic = parse_template(_read(args.generic), allow_missing_values=True)
        standard = (
            parse_template(_read(args.standard), allow_missing_values=False)
            if args.rule_inference
            else ()
        )
        categories = collect_categories(generic)
        if sam3_policy_path is not None:
            sam3_category_thresholds, sam3_threshold_policy, sam3_policy_path = (
                _resolve_sam3_threshold_runtime(
                    class_name=class_name,
                    enabled=args.object_sam3_thresholds,
                    policy_root=args.sam3_threshold_policy_root,
                    explicit_policy_path=args.sam3_threshold_policy,
                    expected_configuration=(
                        _sam3_threshold_configuration(args, standard_path=args.standard)
                        if args.standard is not None
                        else {}
                    ),
                    expected_categories=categories,
                )
            )
        segmenter = Sam3Segmenter(
            repo_path=args.sam3_repo,
            checkpoint_path=args.sam3_checkpoint,
            device=args.device,
            confidence_threshold=args.sam3_threshold,
            category_thresholds=sam3_category_thresholds,
            prompt_template=args.sam3_prompt,
            cache_root=args.sam3_cache_root,
            cache_mode=args.sam3_cache_mode,
        )
        if args.rule_inference:
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
        else:
            detector = None
        mask_processor = MaskPostprocessor(
            dedup_iou_threshold=args.mask_dedup_iou,
            dedup_containment_threshold=args.mask_dedup_containment,
            dedup_enabled=args.mask_dedup,
        )
    else:
        generic = ()
        standard = ()
        segmenter = None
        detector = None
        mask_processor = None
        sam3_category_thresholds = {}
        sam3_threshold_policy = "global_only"
    patch_scorer = _make_patch_scorer(args, class_name=class_name)
    pipeline = LogicAnomalyPipeline(
        segmenter,
        detector,
        mask_processor=mask_processor,
        patch_scorer=patch_scorer,
        grouping_tolerance=args.grouping_tolerance,
        rule_inference_enabled=args.rule_inference,
        rule_memory_schedule=args.rule_memory_schedule,
    )
    report = pipeline.run(args.image, generic, standard)
    report_payload = report.to_dict()
    report_payload["patch_matching_enabled"] = bool(args.patch_matching)
    report_payload["patch_feature_mode"] = "global" if args.patch_matching else None
    report_payload["category_threshold_policy_active"] = bool(
        sam3_policy_path is not None and sam3_category_thresholds
    )
    report_payload["rule_inference_enabled"] = bool(args.rule_inference)
    report_payload["patch_layer_fusion"] = (
        DEFAULT_PATCH_LAYER_FUSION if args.patch_matching else None
    )
    report_payload["sam3_threshold_policy"] = sam3_threshold_policy
    report_payload["sam3_global_fallback_threshold"] = (
        args.sam3_threshold if mask_inference_enabled else None
    )
    report_payload["clip_attribute_confidence_threshold"] = (
        args.clip_threshold if args.rule_inference else None
    )
    report_payload["sam3_category_thresholds"] = sam3_category_thresholds
    report_payload["sam3_threshold_policy_path"] = sam3_policy_path
    report_payload["sam3_cache_root"] = (
        str(Path(args.sam3_cache_root).expanduser().resolve())
        if mask_inference_enabled
        else None
    )
    report_payload["sam3_cache_mode"] = (
        args.sam3_cache_mode if mask_inference_enabled else "off"
    )
    report_payload["sam3_cache_hits"] = (
        segmenter.cache_hits if segmenter is not None else 0
    )
    report_payload["sam3_cache_misses"] = (
        segmenter.cache_misses if segmenter is not None else 0
    )
    report_payload["sam3_cache_writes"] = (
        segmenter.cache_writes if segmenter is not None else 0
    )
    payload = json.dumps(report_payload, ensure_ascii=False, indent=2)
    _write_or_print(payload, args.output)
    if report.is_anomaly is None:
        return 2
    if report.is_anomaly is None:
        return 0
    return 1 if report.is_anomaly else 0


def _command_evaluate_screw_bag(args: argparse.Namespace) -> int:
    initial_configuration = (
        _sam3_threshold_configuration(args, standard_path=args.standard)
        if Path(args.standard).expanduser().is_file()
        else {}
    )
    sam3_category_thresholds, sam3_threshold_policy, sam3_policy_path = (
        _resolve_sam3_threshold_runtime(
            class_name="screw_bag",
            enabled=args.object_sam3_thresholds,
            policy_root=args.sam3_threshold_policy_root,
            explicit_policy_path=args.sam3_threshold_policy,
            expected_configuration=initial_configuration,
            expected_categories=None,
        )
    )
    _validate_runtime_branches(args)
    if sam3_policy_path is not None:
        sam3_category_thresholds, sam3_threshold_policy, sam3_policy_path = (
            _resolve_sam3_threshold_runtime(
                class_name="screw_bag",
                enabled=args.object_sam3_thresholds,
                policy_root=args.sam3_threshold_policy_root,
                explicit_policy_path=args.sam3_threshold_policy,
                expected_configuration=_sam3_threshold_configuration(
                    args, standard_path=args.standard
                ),
                expected_categories=collect_categories(
                    parse_template(_read(args.generic), allow_missing_values=True)
                ),
            )
        )
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
        mask_dedup=args.mask_dedup,
        rule_inference_enabled=args.rule_inference,
        rule_memory_schedule=args.rule_memory_schedule,
        patch_enabled=args.patch_matching,
        patch_feature_mode="global",
        patch_bank_path=(
            str(_patch_bank_path(args, "screw_bag")) if args.patch_matching else None
        ),
        patch_dinov2_checkpoint=args.patch_dinov2_checkpoint,
        patch_query_batch_size=args.patch_query_batch_size,
        patch_extraction_batch_size=args.patch_extraction_batch_size,
        include_structural_anomalies=args.all_test_splits,
        restart=args.restart,
        test_split=args.test_split,
        test_splits=args.test_splits,
        sam3_cache_root=args.sam3_cache_root,
        sam3_cache_mode=args.sam3_cache_mode,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failure_count"] == 0 else 2


def _command_evaluate(args: argparse.Namespace) -> int:
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
    initial_configuration = (
        _sam3_threshold_configuration(args, standard_path=standard_path)
        if Path(standard_path).expanduser().is_file()
        else {}
    )
    sam3_category_thresholds, sam3_threshold_policy, sam3_policy_path = (
        _resolve_sam3_threshold_runtime(
            class_name=class_name,
            enabled=args.object_sam3_thresholds,
            policy_root=args.sam3_threshold_policy_root,
            explicit_policy_path=args.sam3_threshold_policy,
            expected_configuration=initial_configuration,
            expected_categories=None,
        )
    )
    _validate_runtime_branches(args)
    mask_inference_enabled = args.rule_inference
    if mask_inference_enabled:
        runtime_categories = collect_categories(
            parse_template(_read(generic_path), allow_missing_values=True)
        )
        if sam3_policy_path is not None:
            sam3_category_thresholds, sam3_threshold_policy, sam3_policy_path = (
                _resolve_sam3_threshold_runtime(
                    class_name=class_name,
                    enabled=args.object_sam3_thresholds,
                    policy_root=args.sam3_threshold_policy_root,
                    explicit_policy_path=args.sam3_threshold_policy,
                    expected_configuration=_sam3_threshold_configuration(
                        args, standard_path=standard_path
                    ),
                    expected_categories=runtime_categories,
                )
            )
        vocabulary = (
            _load_vocabulary(args.attribute_vocabulary) if args.rule_inference else {}
        )
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
        mask_dedup=args.mask_dedup,
        rule_inference_enabled=args.rule_inference,
        rule_memory_schedule=args.rule_memory_schedule,
        patch_enabled=args.patch_matching,
        patch_feature_mode="global",
        patch_bank_path=(
            str(_patch_bank_path(args, class_name)) if args.patch_matching else None
        ),
        patch_dinov2_checkpoint=args.patch_dinov2_checkpoint,
        patch_query_batch_size=args.patch_query_batch_size,
        patch_extraction_batch_size=args.patch_extraction_batch_size,
        include_structural_anomalies=args.all_test_splits,
        restart=args.restart,
        test_split=args.test_split,
        test_splits=args.test_splits,
        dataset_name=args.dataset,
        sam3_cache_root=args.sam3_cache_root,
        sam3_cache_mode=args.sam3_cache_mode,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failure_count"] == 0 else 2


def _command_cache_sam3(args: argparse.Namespace) -> int:
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
    summary = build_sam3_dataset_cache(
        class_name=class_name,
        dataset_root=dataset_root,
        generic_path=generic_path,
        cache_root=args.sam3_cache_root,
        sam3_threshold=args.sam3_threshold,
        sam3_repo=args.sam3_repo,
        sam3_checkpoint=args.sam3_checkpoint,
        device=args.device,
        sam3_prompt=args.sam3_prompt,
        test_splits=args.test_splits,
        dataset_name=args.dataset,
        refresh=args.refresh,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failure_count"] == 0 else 2


def _command_build_patch_memory(args: argparse.Namespace) -> int:
    class_name = _validate_class_name(args.class_name)
    normal_images = _resolve_normal_images(args)
    output = args.output or str(
        Path(args.patch_bank_root).expanduser().resolve()
        / f"{class_name}.{patch_memory_run_tag(args.patch_coreset_ratio)}.npz"
    )
    bank = build_patch_memory_from_images(
        class_name=class_name,
        normal_images=normal_images,
        output_path=output,
        checkpoint_path=args.patch_dinov2_checkpoint,
        device=args.device,
        query_batch_size=args.patch_query_batch_size,
        extraction_batch_size=args.patch_extraction_batch_size,
        coreset_ratio=args.patch_coreset_ratio,
        coreset_seed=args.patch_coreset_seed,
        progress_callback=_progress_reporter("patch-memory"),
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
                "coreset_enabled": True,
                "coreset_ratio": bank.coreset_ratio,
                "coreset_seed": bank.coreset_seed,
                "coreset_patch_count": int(bank.memory_source_indices.numel()),
                "score_mean": bank.score_mean,
                "score_unbiased_std": bank.score_unbiased_std,
                "calibration_mode": bank.calibration_mode,
                "memory_scope": "all_train_good_images_patch_coreset",
                "storage": PATCH_MEMORY_PER_IMAGE_STORAGE,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _command_calibrate_sam3(args: argparse.Namespace) -> int:
    class_name = _validate_class_name(args.class_name)
    output = (
        Path(
            args.output or Path(args.sam3_threshold_policy_root) / f"{class_name}.json"
        )
        .expanduser()
        .resolve()
    )
    if output.exists() and not args.overwrite:
        raise ValueError(
            f"SAM3 threshold policy already exists: {output}; "
            "choose a different --output path or pass --overwrite explicitly"
        )
    standard_path = args.standard or str(
        Path(args.template_root) / f"{class_name}.standard.rules"
    )
    standard_rules = parse_template(
        Path(standard_path).read_text(encoding="utf-8"),
        allow_missing_values=False,
    )
    input_normal_images = _resolve_normal_images(args)
    missing_images = [str(path) for path in input_normal_images if not path.is_file()]
    if missing_images:
        raise ValueError(
            "normal calibration images not found: " + ", ".join(missing_images)
        )
    progress = _progress_reporter("sam3-threshold")
    normal_images = filter_decodable_images(
        input_normal_images,
        progress_callback=progress,
    )
    if len(normal_images) < 2:
        raise ValueError(
            "SAM3 calibration needs at least two decodable normal images " "after skips"
        )
    retained = set(normal_images)
    retained_indices = tuple(
        index for index, path in enumerate(input_normal_images) if path in retained
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
        subject_counts: dict[str, tuple[int, ...]] = {}
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
            if len(values) != len(input_normal_images):
                raise ValueError(
                    f"--truth-subject-counts for {subject!r} needs "
                    f"{len(input_normal_images)} values, found {len(values)}"
                )
            if any(value < 0 for value in values):
                raise ValueError(
                    "inline truth count values must be non-negative integers"
                )
            subject_counts[subject] = tuple(values[index] for index in retained_indices)
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
                and len(rule.expected if rule.task.value == "COUNT" else rule.count) > 1
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
                (
                    replace(
                        rule,
                        expected=(
                            (subject_counts[rule.subject][sample_index],)
                            if rule.task.value == "COUNT"
                            else rule.expected
                        ),
                        count=(
                            (subject_counts[rule.subject][sample_index],)
                            if rule.task.value in {"LENGTH", "AREA"}
                            else rule.count
                        ),
                    )
                    if (
                        rule.task.value in {"COUNT", "LENGTH", "AREA"}
                        and rule.subject in subject_counts
                    )
                    else rule
                )
                for rule in standard_rules
            )
            for sample_index in range(len(normal_images))
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
                and len(rule.expected if rule.task.value == "COUNT" else rule.count) > 1
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
        if len(truth_counts) != len(input_normal_images):
            raise ValueError(
                f"--truth-counts needs {len(input_normal_images)} values, "
                f"found {len(truth_counts)}"
            )
        if any(value < 0 for value in truth_counts):
            raise ValueError("--truth-counts values must be non-negative integers")
        truth_counts = tuple(truth_counts[index] for index in retained_indices)
        truth_rules = tuple(
            tuple(
                (
                    replace(rule, expected=(count,))
                    if rule.task.value == "COUNT" and rule.subject == truth_subject
                    else (
                        replace(rule, count=(count,))
                        if rule.task.value in {"LENGTH", "AREA"}
                        and rule.subject == truth_subject
                        else rule
                    )
                )
                for rule in standard_rules
            )
            for count in truth_counts
        )
        truth_paths = None
    elif args.truth_templates:
        input_truth_paths = tuple(
            Path(path).expanduser().resolve() for path in args.truth_templates
        )
        missing_templates = [
            str(path) for path in input_truth_paths if not path.is_file()
        ]
        if missing_templates:
            raise ValueError(
                "truth templates not found: " + ", ".join(missing_templates)
            )
        if len(input_truth_paths) != len(input_normal_images):
            raise ValueError(
                f"--truth-templates needs {len(input_normal_images)} paths, "
                f"found {len(input_truth_paths)}"
            )
        truth_paths = tuple(input_truth_paths[index] for index in retained_indices)
        truth_rules = tuple(
            parse_template(path.read_text(encoding="utf-8"), allow_missing_values=False)
            for path in truth_paths
        )
    else:
        truth_paths = (Path(standard_path).expanduser().resolve(),) * len(normal_images)
        truth_rules = tuple(standard_rules for _ in normal_images)

    segmenter = Sam3Segmenter(
        repo_path=args.sam3_repo,
        checkpoint_path=args.sam3_checkpoint,
        device=args.device,
        confidence_threshold=args.sam3_threshold,
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
            dedup_enabled=args.mask_dedup,
        ),
        proposal_threshold=args.sam3_threshold,
        selection_method=args.threshold_selection_method,
        configuration=_sam3_threshold_configuration(args, standard_path=standard_path),
        truth_template_paths=truth_paths,
        progress_callback=progress,
    )
    result_path = (
        Path(args.calibration_result_root).expanduser().resolve()
        / output.parent.name
        / f"{class_name}.json"
    )
    result["saved_path"] = str(output)
    result["calibration_result_path"] = str(result_path)
    saved_result_path = save_sam3_threshold_calibration_result(
        result,
        result_path,
    )
    saved_path = save_sam3_threshold_policy(result, output)
    progress("saved", len(normal_images), len(normal_images), str(saved_path))
    result["saved_path"] = str(saved_path)
    result["calibration_result_path"] = str(saved_result_path)
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
    cache_sam3 = subparsers.add_parser(
        "cache-sam3",
        help="precompute reusable SAM3 proposals without running rules or DINO",
    )
    cache_sam3.add_argument("--class-name", required=True)
    cache_sam3.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=MVTEC_LOCO)
    cache_sam3.add_argument("--dataset-base")
    cache_sam3.add_argument("--dataset-root")
    cache_sam3.add_argument(
        "--template-root", default="/home/lxq/pro-innovation/templates/store"
    )
    cache_sam3.add_argument("--generic")
    cache_sam3.add_argument("--device", default="cuda")
    cache_sam3.add_argument(
        "--sam3-threshold",
        type=_probability_argument,
        default=CALIBRATION_PROPOSAL_THRESHOLD,
        help="proposal-generation threshold; 0.20 caches candidates reusable by the current stricter object thresholds",
    )
    cache_sam3.add_argument("--sam3-prompt", default="{category}")
    cache_sam3.add_argument("--sam3-repo", default="/home/lxq/models/SAM3")
    cache_sam3.add_argument(
        "--sam3-checkpoint", default="/home/lxq/weights/sam3/sam3.pt"
    )
    cache_sam3.add_argument("--sam3-cache-root", default=DEFAULT_SAM3_CACHE_ROOT)
    cache_sam3.add_argument(
        "--test-splits",
        nargs="+",
        help="dataset split names to cache; defaults to every official test split",
    )
    cache_sam3.add_argument(
        "--refresh",
        action="store_true",
        help="recompute and atomically replace matching cache entries",
    )
    cache_sam3.set_defaults(func=_command_cache_sam3)
    build_patch = subparsers.add_parser(
        "build-patch-memory",
        help="build a full-normal LogSAD DINOv2 patch coreset and calibrate it with leave-one-image-out normal scores",
    )
    build_patch.add_argument("--class-name", required=True)
    patch_normal_source = build_patch.add_mutually_exclusive_group(required=True)
    patch_normal_source.add_argument("--normal-images", nargs="+")
    patch_normal_source.add_argument(
        "--normal-dir", help="directory containing every normal training image"
    )
    build_patch.add_argument("--output")
    build_patch.add_argument(
        "--patch-bank-root",
        default=str(Path(DEFAULT_PATCH_BANK_ROOT) / MVTEC_LOCO / "global"),
    )
    build_patch.add_argument("--device", default="cuda")
    build_patch.add_argument(
        "--patch-dinov2-checkpoint", default=DEFAULT_PATCH_DINOV2_CHECKPOINT
    )
    build_patch.add_argument("--patch-query-batch-size", type=int, default=256)
    build_patch.add_argument("--patch-extraction-batch-size", type=int, default=4)
    build_patch.add_argument(
        "--patch-coreset-ratio", type=float, default=DEFAULT_CORESET_RATIO
    )
    build_patch.add_argument(
        "--patch-coreset-seed", type=int, default=DEFAULT_CORESET_SEED
    )
    build_patch.set_defaults(func=_command_build_patch_memory)
    calibrate_sam3 = subparsers.add_parser(
        "calibrate-sam3-thresholds",
        help="derive fixed per-object SAM3 thresholds from all train/good images",
    )
    calibrate_sam3.add_argument("--class-name", required=True)
    calibrate_normal_source = calibrate_sam3.add_mutually_exclusive_group(required=True)
    calibrate_normal_source.add_argument("--normal-images", nargs="+")
    calibrate_normal_source.add_argument(
        "--normal-dir", help="directory containing all train/good calibration images"
    )
    calibrate_sam3.add_argument(
        "--truth-templates",
        nargs="+",
        help="optional sample-specific concrete templates, one per normal image",
    )
    calibrate_sam3.add_argument(
        "--truth-counts",
        nargs="+",
        type=int,
        help="per-image COUNT truths; expands a variable COUNT rule in memory without requiring sample-specific template files",
    )
    calibrate_sam3.add_argument(
        "--truth-subject", help="COUNT subject to replace when --truth-counts is used"
    )
    calibrate_sam3.add_argument(
        "--truth-subject-counts",
        nargs="+",
        action="append",
        metavar="SUBJECT_OR_COUNT",
        help="repeatable SUBJECT followed by one count per normal image",
    )
    calibrate_sam3.add_argument(
        "--template-root", default="/home/lxq/pro-innovation/templates/store"
    )
    calibrate_sam3.add_argument("--standard")
    calibrate_sam3.add_argument("--output")
    calibrate_sam3.add_argument(
        "--calibration-result-root",
        default=DEFAULT_SAM3_THRESHOLD_RESULT_ROOT,
        help="directory for full calibration reports; runtime policy stays minimal",
    )
    calibrate_sam3.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly allow replacing an existing threshold policy",
    )
    calibrate_sam3.add_argument(
        "--sam3-threshold-policy-root", default=DEFAULT_SAM3_THRESHOLD_POLICY_BASE_ROOT
    )
    calibrate_sam3.add_argument("--device", default="cuda")
    calibrate_sam3.add_argument(
        "--sam3-threshold",
        type=_probability_argument,
        default=CALIBRATION_PROPOSAL_THRESHOLD,
        help="candidate-generation threshold for calibration in [0, 1] (default: 0.20)",
    )
    calibrate_sam3.add_argument(
        "--threshold-selection-method",
        choices=CALIBRATION_SELECTION_METHODS,
        default="confidence_gap_midpoint",
        help="confidence_gap_midpoint computes the Q95/Q05 interval midpoint and rounds it downward to the 0.10 grid",
    )
    calibrate_sam3.add_argument("--mask-dedup-iou", type=float, default=0.8)
    calibrate_sam3.add_argument("--mask-dedup-containment", type=float, default=0.85)
    calibrate_sam3.add_argument(
        "--mask-dedup", action=argparse.BooleanOptionalAction, default=True
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
        "--class-name", help="class identity used to select the default Patch bank"
    )
    infer.add_argument("--image", required=True)
    infer.add_argument("--generic")
    infer.add_argument("--standard")
    infer.add_argument("--output")
    infer.add_argument("--attribute-vocabulary")
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--grouping-tolerance", type=float, default=0.1)
    infer.add_argument("--sam3-threshold", type=float, default=0.4)
    infer.add_argument("--mask-dedup-iou", type=float, default=0.8)
    infer.add_argument("--mask-dedup-containment", type=float, default=0.85)
    infer.add_argument(
        "--mask-dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable same-category duplicate-mask suppression",
    )
    infer.add_argument("--sam3-prompt", default="{category}")
    infer.add_argument("--sam3-repo", default="/home/lxq/models/SAM3")
    infer.add_argument("--sam3-checkpoint", default="/home/lxq/weights/sam3/sam3.pt")
    _add_sam3_threshold_runtime_arguments(infer)
    _add_sam3_cache_runtime_arguments(infer)
    infer.add_argument("--clip-repo", default="/home/lxq/models/CLIP")
    infer.add_argument(
        "--clip-checkpoint", default="/home/lxq/weights/CLIP/ViT-L-14.pt"
    )
    infer.add_argument(
        "--clip-threshold",
        type=float,
        default=0.35,
        help="minimum confidence for CLIP semantic-attribute predictions; this does not enable SAM3 mask category verification",
    )
    _add_patch_runtime_arguments(infer)
    _add_rule_runtime_arguments(infer)
    infer.set_defaults(func=_command_infer)
    evaluate_generic = subparsers.add_parser(
        "evaluate", help="evaluate a MVTec LOCO, MVTec AD, or VisA class"
    )
    evaluate_generic.add_argument("--class-name", required=True)
    evaluate_generic.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default=MVTEC_LOCO,
        help="dataset layout: mvtec_loco uses fixed logical/structural splits; mvtec_ad discovers every non-good defect folder; visa reads the official split_csv/1cls.csv test rows",
    )
    evaluate_generic.add_argument(
        "--dataset-base",
        help="dataset root containing one directory per class; defaults to the project datasets symlink selected by --dataset",
    )
    evaluate_generic.add_argument(
        "--dataset-root",
        help="override the dataset path; for VisA this is the root containing split_csv, otherwise <dataset-base>/<class-name>/test",
    )
    evaluate_generic.add_argument(
        "--template-root", default="/home/lxq/pro-innovation/templates/store"
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
    evaluate_generic.add_argument("--grouping-tolerance", type=float, default=0.1)
    evaluate_generic.add_argument("--sam3-threshold", type=float, default=0.4)
    evaluate_generic.add_argument("--mask-dedup-iou", type=float, default=0.8)
    evaluate_generic.add_argument("--mask-dedup-containment", type=float, default=0.85)
    evaluate_generic.add_argument(
        "--mask-dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable same-category duplicate-mask suppression",
    )
    evaluate_generic.add_argument("--sam3-prompt", default="{category}")
    evaluate_generic.add_argument("--sam3-repo", default="/home/lxq/models/SAM3")
    evaluate_generic.add_argument(
        "--sam3-checkpoint", default="/home/lxq/weights/sam3/sam3.pt"
    )
    _add_sam3_threshold_runtime_arguments(evaluate_generic)
    _add_sam3_cache_runtime_arguments(evaluate_generic)
    evaluate_generic.add_argument("--clip-repo", default="/home/lxq/models/CLIP")
    evaluate_generic.add_argument(
        "--clip-checkpoint", default="/home/lxq/weights/CLIP/ViT-L-14.pt"
    )
    evaluate_generic.add_argument(
        "--clip-threshold",
        type=float,
        default=0.35,
        help="minimum confidence for CLIP semantic-attribute predictions; this does not enable SAM3 mask category verification",
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
        "--test-split", help="evaluate only one test folder; models are loaded once"
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
        default="/home/lxq/pro-innovation/datasets/mvtec_loco_anomaly_detection/screw_bag/test",
    )
    evaluate.add_argument(
        "--generic",
        default="/home/lxq/pro-innovation/templates/store/screw_bag.generic.rules",
    )
    evaluate.add_argument(
        "--standard",
        default="/home/lxq/pro-innovation/templates/store/screw_bag.standard.rules",
    )
    evaluate.add_argument("--result-root", default="/home/lxq/pro-innovation/result")
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--sam3-threshold", type=float, default=0.4)
    evaluate.add_argument("--mask-dedup-iou", type=float, default=0.8)
    evaluate.add_argument("--mask-dedup-containment", type=float, default=0.85)
    evaluate.add_argument(
        "--mask-dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable same-category duplicate-mask suppression",
    )
    evaluate.add_argument("--sam3-repo", default="/home/lxq/models/SAM3")
    evaluate.add_argument("--sam3-checkpoint", default="/home/lxq/weights/sam3/sam3.pt")
    evaluate.add_argument("--sam3-prompt", default="{category}")
    _add_sam3_threshold_runtime_arguments(evaluate)
    _add_sam3_cache_runtime_arguments(evaluate)
    evaluate.add_argument("--clip-repo", default="/home/lxq/models/CLIP")
    evaluate.add_argument(
        "--clip-checkpoint", default="/home/lxq/weights/CLIP/ViT-L-14.pt"
    )
    evaluate.add_argument(
        "--clip-threshold",
        type=float,
        default=0.35,
        help="minimum confidence for CLIP semantic-attribute predictions; this does not enable SAM3 mask category verification",
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
