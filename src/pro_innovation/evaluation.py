"""MVTec LOCO and MVTec AD evaluation with parameter-derived output names."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

from .adapters import (
    ClipAttributeDetector,
    LocalClipRuntime,
    Sam3Segmenter,
)
from .adapters.patch_dinov2 import (
    DEFAULT_PATCH_DINOV2_CHECKPOINT,
    PATCH_MATCHING_LAYERS,
    DinoV2PatchFeatureExtractor,
)
from .attributes import (
    RoutedAttributeDetector,
    attribute_vocabulary_key,
    validate_color_candidates,
)
from .mask_processing import MaskPostprocessor
from .patch_memory import (
    DEFAULT_PATCH_LAYER_FUSION,
    patch_memory_run_tag,
    DinoV2PatchImageScorer,
    PatchMemoryBank,
)
from .models import Rule, TaskType
from .pipeline import (
    LogicAnomalyPipeline,
    SCORE_FUSION_POLICY,
)
from .templates import collect_categories, parse_template


MVTEC_LOCO = "mvtec_loco"
MVTEC_AD = "mvtec_ad"
VISA = "visa"
SUPPORTED_DATASETS = (MVTEC_LOCO, MVTEC_AD, VISA)
DATASET_DISPLAY_NAMES = {
    MVTEC_LOCO: "MVTec LOCO",
    MVTEC_AD: "MVTec AD",
    VISA: "VisA",
}


def _split_specs_for_dataset(
    dataset_name: str,
    dataset_root: Path,
) -> dict[str, tuple[int, bool]]:
    """Return image labels and structural flags for one dataset test tree."""

    if dataset_name == MVTEC_LOCO:
        return {
            "good": (0, False),
            "logical_anomalies": (1, False),
            "structural_anomalies": (1, True),
        }
    if dataset_name == MVTEC_AD:
        split_names = sorted(
            path.name for path in dataset_root.iterdir() if path.is_dir()
        )
        if "good" not in split_names:
            raise ValueError(f"MVTec AD test tree has no good split: {dataset_root}")
        return {
            split: (0, False) if split == "good" else (1, True)
            for split in ("good", *(name for name in split_names if name != "good"))
        }
    if dataset_name == VISA:
        manifest = dataset_root / "split_csv" / "1cls.csv"
        if not manifest.is_file():
            raise ValueError(f"VisA 1-class split manifest not found: {manifest}")
        return {
            "Normal": (0, False),
            "Anomaly": (1, False),
        }
    raise ValueError(
        f"unsupported dataset {dataset_name!r}; expected one of {SUPPORTED_DATASETS}"
    )


def _collect_evaluation_samples(
    *,
    dataset_name: str,
    dataset_root: Path,
    class_name: str,
    selected_splits: Sequence[str],
    split_specs: Mapping[str, tuple[int, bool]],
) -> list[tuple[Path, int, str]]:
    """Collect test samples while preserving each dataset's official layout."""

    if dataset_name != VISA:
        samples: list[tuple[Path, int, str]] = []
        for split in selected_splits:
            label, _ = split_specs[split]
            samples.extend(
                (path, label, split)
                for path in sorted((dataset_root / split).glob("*.png"))
            )
        return samples

    manifest = dataset_root / "split_csv" / "1cls.csv"
    samples = []
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"object", "split", "label", "image"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"VisA split manifest is missing columns {sorted(missing)}: {manifest}"
            )
        for row in reader:
            if row["object"] != class_name or row["split"].lower() != "test":
                continue
            label_name = row["label"].lower()
            if label_name == "normal":
                split = "Normal"
            elif label_name == "anomaly":
                split = "Anomaly"
            else:
                raise ValueError(
                    f"unsupported VisA label {row['label']!r} in {manifest}"
                )
            if split not in selected_splits:
                continue
            image = dataset_root / row["image"]
            if not image.is_file():
                raise ValueError(f"VisA image listed by manifest not found: {image}")
            samples.append((image, split_specs[split][0], split))
    return sorted(samples, key=lambda item: (item[2], str(item[0])))


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both positive and negative samples")

    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label for _, label in ordered[index:end]
        )
        index = end
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def infer_tolerance_percent(
    rules: Sequence[Rule], task: TaskType
) -> float | None:
    tolerances = []
    for rule in rules:
        if (
            rule.task is task
            and rule.minimum is not None
            and rule.maximum is not None
        ):
            center = (rule.minimum + rule.maximum) / 2.0
            if center > 0.0:
                tolerances.append(
                    (rule.maximum - rule.minimum) / (2.0 * center) * 100.0
                )
    if not tolerances:
        return None
    if max(tolerances) - min(tolerances) > 0.05:
        return None
    return sum(tolerances) / len(tolerances)


def infer_length_tolerance_percent(rules: Sequence[Rule]) -> float | None:
    """Retain the original public helper for screw_bag compatibility."""

    return infer_tolerance_percent(rules, TaskType.LENGTH)


def _length_tag(value: float | None) -> str:
    return _measurement_tag("length", value)


def _measurement_tag(name: str, value: float | None) -> str:
    if value is None:
        return f"{name}custom"
    rounded = round(value, 2)
    if abs(rounded - round(rounded)) < 0.01:
        return f"{name}{int(round(rounded))}"
    return f"{name}{str(rounded).replace('.', 'p')}"


def infer_measurement_tag(rules: Sequence[Rule]) -> str:
    """Describe every measurement family present in a standard template."""

    tags = []
    for task, name in ((TaskType.LENGTH, "length"), (TaskType.AREA, "area")):
        if any(rule.task is task for rule in rules):
            tags.append(_measurement_tag(name, infer_tolerance_percent(rules, task)))
    return "-".join(tags) if tags else "nomeasurement"


def build_run_stem(
    class_name: str,
    sam3_threshold: float,
    length_tolerance_percent: float | None,
    clip_threshold: float = 0.35,
    mask_dedup_iou: float = 0.80,
    mask_dedup_containment: float = 0.85,
    mask_min_area: int = 16,
    mask_cleanup: bool = True,
    mask_dedup: bool = True,
    measurement_tag: str | None = None,
    rule_inference_enabled: bool = True,
    patch_enabled: bool = False,
    include_structural_anomalies: bool = False,
    sam3_category_thresholds: Mapping[str, float] | None = None,
    test_splits: Sequence[str] | None = None,
    dataset_name: str = MVTEC_LOCO,
) -> str:
    safe_class = re.sub(r"[^A-Za-z0-9_-]+", "_", class_name).strip("_")
    patch_tag = patch_memory_run_tag() if patch_enabled else "rules"
    scope = "-".join(test_splits) if test_splits else (
        "good-logical-structural" if include_structural_anomalies else "good-logical"
    )
    tags = [safe_class, patch_tag]
    if dataset_name != MVTEC_LOCO:
        tags.insert(0, dataset_name)
    if not rule_inference_enabled:
        tags.append("patchonly")
    tags.append(scope)
    return "_".join(tags)


def validate_attribute_vocabulary(
    rules: Sequence[Rule], vocabulary: Mapping[str, Sequence[str]]
) -> None:
    """Fail before model loading when subject-scoped attributes lack candidates."""

    required: set[tuple[str, str]] = set()
    expected_by_key: dict[str, set[str]] = {}
    for rule in rules:
        if rule.task is TaskType.ATTRIBUTE_ERROR and rule.subject and rule.property:
            required.add((rule.subject, rule.property))
            if rule.expected is not None:
                key = attribute_vocabulary_key(rule.subject, rule.property)
                expected_by_key.setdefault(key, set()).add(
                    str(rule.expected)
                )
        elif rule.task is TaskType.ATTRIBUTE_COMBINATION:
            required.update((item.subject, item.property) for item in rule.attributes)
            if isinstance(rule.expected, tuple):
                for combination in rule.expected:
                    for item, value in zip(rule.attributes, combination.split(":")):
                        key = attribute_vocabulary_key(item.subject, item.property)
                        expected_by_key.setdefault(key, set()).add(value)
            else:
                for item in rule.attributes:
                    if item.expected is not None:
                        key = attribute_vocabulary_key(item.subject, item.property)
                        expected_by_key.setdefault(key, set()).add(
                            str(item.expected)
                        )

    missing = []
    absent_values: list[str] = []
    for subject, property_name in sorted(required):
        attribute = property_name.split(".", 1)[-1].lower()
        if attribute in {"shape", "count", "length", "area"} or attribute.startswith(
            "spatial_combination."
        ):
            continue
        key = attribute_vocabulary_key(subject, property_name)
        candidates = vocabulary.get(key)
        if candidates is None:
            missing.append(key)
            continue
        if attribute == "color":
            normalized_candidates = validate_color_candidates(candidates)
            configured = set(normalized_candidates)
        elif len(candidates) < 2:
            missing.append(key)
            continue
        else:
            configured = {str(candidate).strip() for candidate in candidates}

        absent_values.extend(
            f"{key}={value}"
            for value in sorted(expected_by_key.get(key, ()))
            if value not in configured
        )
    if missing:
        raise ValueError(
            "attribute vocabulary is missing required candidates for: "
            + ", ".join(missing)
        )
    if absent_values:
        raise ValueError(
            "standard template values missing from configured vocabulary: "
            + ", ".join(absent_values)
        )


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("error") is None:
            completed[record["image"]] = record
    return completed


def evaluate_dataset(
    *,
    class_name: str,
    dataset_root: str | Path,
    generic_path: str | Path | None,
    standard_path: str | Path | None,
    result_root: str | Path,
    sam3_threshold: float,
    sam3_repo: str | Path,
    sam3_checkpoint: str | Path,
    clip_repo: str | Path,
    clip_checkpoint: str | Path,
    clip_threshold: float = 0.35,
    attribute_vocabulary: Mapping[str, Sequence[str]] | None = None,
    device: str = "cuda",
    grouping_tolerance: float = 0.10,
    sam3_prompt: str = "{category}",
    mask_dedup_iou: float = 0.80,
    mask_dedup_containment: float = 0.85,
    mask_min_area: int = 16,
    mask_cleanup: bool = True,
    mask_dedup: bool = True,
    rule_inference_enabled: bool = True,
    patch_enabled: bool = False,
    patch_bank_path: str | Path | None = None,
    patch_dinov2_checkpoint: str | Path = DEFAULT_PATCH_DINOV2_CHECKPOINT,
    patch_query_batch_size: int = 256,
    patch_extraction_batch_size: int = 4,
    include_structural_anomalies: bool | None = None,
    sam3_category_thresholds: Mapping[str, float] | None = None,
    sam3_threshold_policy: str = "global_only",
    sam3_threshold_policy_path: str | Path | None = None,
    restart: bool = False,
    test_split: str | None = None,
    test_splits: Sequence[str] | None = None,
    dataset_name: str = MVTEC_LOCO,
) -> dict[str, Any]:
    dataset = Path(dataset_root).expanduser().resolve()
    output_root = Path(result_root).expanduser().resolve()
    if not dataset.is_dir():
        raise ValueError(f"dataset test directory not found: {dataset}")
    if not rule_inference_enabled:
        if not patch_enabled:
            raise ValueError("disabled rule inference requires patch matching")
        generic_file = None
        standard_file = None
        generic: tuple[Rule, ...] = ()
        standard: tuple[Rule, ...] = ()
        categories: tuple[str, ...] = ()
        vocabulary: dict[str, Sequence[str]] = {}
        length_tolerance = None
        area_tolerance = None
        measurement_tag = "disabled"
    else:
        if generic_path is None or standard_path is None:
            raise ValueError("rule inference requires generic and standard templates")
        generic_file = Path(generic_path).expanduser().resolve()
        standard_file = Path(standard_path).expanduser().resolve()
        generic = parse_template(
            generic_file.read_text(encoding="utf-8"), allow_missing_values=True
        )
        standard = parse_template(standard_file.read_text(encoding="utf-8"))
        categories = collect_categories(generic)
        vocabulary = dict(attribute_vocabulary or {})
        validate_attribute_vocabulary(standard, vocabulary)
        length_tolerance = infer_length_tolerance_percent(standard)
        area_tolerance = infer_tolerance_percent(standard, TaskType.AREA)
        measurement_tag = infer_measurement_tag(standard)
    split_specs = _split_specs_for_dataset(dataset_name, dataset)
    if test_splits is not None and test_split is not None:
        raise ValueError("use either test_split or test_splits, not both")
    selected_splits = (
        tuple(test_splits)
        if test_splits is not None
        else (test_split,)
        if test_split is not None
        else tuple(split_specs)
        if dataset_name in {MVTEC_AD, VISA}
        and include_structural_anomalies is not False
        else ("good",)
        if dataset_name == MVTEC_AD
        else ("Normal",)
        if dataset_name == VISA
        else ("good", "logical_anomalies", "structural_anomalies")
        if include_structural_anomalies
        else ("good", "logical_anomalies")
    )
    if any(split not in split_specs for split in selected_splits):
        raise ValueError(f"unsupported test split selection: {selected_splits}")
    include_structural = any(split_specs[split][1] for split in selected_splits)
    run_stem = build_run_stem(
        class_name=class_name,
        sam3_threshold=sam3_threshold,
        length_tolerance_percent=length_tolerance,
        clip_threshold=clip_threshold,
        mask_dedup_iou=mask_dedup_iou,
        mask_dedup_containment=mask_dedup_containment,
        mask_min_area=mask_min_area,
        mask_cleanup=mask_cleanup,
        mask_dedup=mask_dedup,
        measurement_tag=measurement_tag,
        rule_inference_enabled=rule_inference_enabled,
        patch_enabled=patch_enabled,
        include_structural_anomalies=include_structural,
        sam3_category_thresholds=sam3_category_thresholds,
        test_splits=selected_splits,
        dataset_name=dataset_name,
    )
    progress_path = output_root / f"{run_stem}.jsonl"
    summary_path = output_root / f"{run_stem}_summary.json"
    output_root.mkdir(parents=True, exist_ok=True)

    samples = _collect_evaluation_samples(
        dataset_name=dataset_name,
        dataset_root=dataset,
        class_name=class_name,
        selected_splits=selected_splits,
        split_specs=split_specs,
    )
    if not samples:
        raise ValueError(f"no evaluation images found under {dataset}")

    completed = {} if restart else _load_completed(progress_path)
    if rule_inference_enabled:
        segmenter = Sam3Segmenter(
            repo_path=sam3_repo,
            checkpoint_path=sam3_checkpoint,
            device=device,
            confidence_threshold=sam3_threshold,
            category_thresholds=sam3_category_thresholds,
            prompt_template=sam3_prompt,
        )
        clip_runtime = LocalClipRuntime(
            repo_path=clip_repo,
            checkpoint_path=clip_checkpoint,
            device=device,
        )
        clip_detector = ClipAttributeDetector(
            repo_path=clip_repo,
            checkpoint_path=clip_checkpoint,
            vocabulary=vocabulary,
            device=device,
            confidence_threshold=clip_threshold,
            runtime=clip_runtime,
        )
        detector = RoutedAttributeDetector(
            semantic_detector=clip_detector if vocabulary else None,
            vocabulary=vocabulary,
        )
        mask_processor = MaskPostprocessor(
            dedup_iou_threshold=mask_dedup_iou,
            dedup_containment_threshold=mask_dedup_containment,
            minimum_area_pixels=mask_min_area,
            cleanup_enabled=mask_cleanup,
            dedup_enabled=mask_dedup,
        )
    else:
        segmenter = None
        clip_runtime = None
        detector = None
        mask_processor = None
    patch_scorer = None
    resolved_patch_bank: str | None = None
    patch_threshold: float | None = None
    if patch_enabled:
        if patch_bank_path is None:
            raise ValueError(
                "patch matching is enabled but no patch memory bank was provided"
            )
        patch_bank = PatchMemoryBank.load(
            patch_bank_path,
            expected_class_name=class_name,
            expected_checkpoint_path=patch_dinov2_checkpoint,
        )
        resolved_patch_bank = str(Path(patch_bank_path).expanduser().resolve())
        patch_threshold = patch_bank.threshold
        patch_scorer = DinoV2PatchImageScorer(
            bank=patch_bank,
            extractor=DinoV2PatchFeatureExtractor(
                checkpoint_path=patch_dinov2_checkpoint,
                device=device,
            ),
            query_batch_size=patch_query_batch_size,
            extraction_batch_size=patch_extraction_batch_size,
        )
    pipeline = LogicAnomalyPipeline(
        segmenter,
        detector,
        mask_processor=mask_processor,
        patch_scorer=patch_scorer,
        grouping_tolerance=grouping_tolerance,
        rule_inference_enabled=rule_inference_enabled,
    )

    run_started = time.perf_counter()
    mode = "w" if restart else "a"
    with progress_path.open(mode, encoding="utf-8") as progress:
        for sample_index, (path, label, split) in enumerate(samples, start=1):
            image_key = str(path)
            if image_key in completed:
                print(
                    f"[{sample_index}/{len(samples)}] resume {split}/{path.name}",
                    flush=True,
                )
                continue
            started = time.perf_counter()
            try:
                report = pipeline.run(path, generic, standard)
                report_payload = report.to_dict()
                record: dict[str, Any] = {
                    "image": image_key,
                    "split": split,
                    "label": label,
                    "score": report.image_anomaly_score,
                    "status": report.status,
                    "logical_anomaly": report.logical_anomaly,
                    "rule_inference_enabled": report.rule_inference_enabled,
                    "execution_mode": report_payload["execution_mode"],
                    "patch_evidence": report_payload["patch_evidence"],
                    "mask_counts": dict(report.mask_counts),
                    "rules": [
                        {
                            "rule_id": item.rule_id,
                            "satisfied": item.satisfied,
                            "actual": item.actual,
                            "expected": item.expected,
                            "violation_score": item.violation_score,
                        }
                        for item in report.results
                    ],
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": None,
                }
            except Exception as exc:
                record = {
                    "image": image_key,
                    "split": split,
                    "label": label,
                    "score": None,
                    "status": "ERROR",
                    "patch_evidence": None,
                    "mask_counts": None,
                    "rules": None,
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            progress.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress.flush()
            completed[image_key] = record
            print(
                f"[{sample_index}/{len(samples)}] {split}/{path.name} "
                f"score={record['score']} status={record['status']} "
                f"seconds={record['elapsed_seconds']:.2f}",
                flush=True,
            )

    records = [completed[str(path)] for path, _, _ in samples]
    failures = [record for record in records if record["score"] is None]
    valid = [record for record in records if record["score"] is not None]

    negative_splits = tuple(
        split for split in selected_splits if split_specs[split][0] == 0
    )
    positive_splits = tuple(
        split for split in selected_splits if split_specs[split][0] == 1
    )

    def subgroup_auroc(positive_split: str) -> float | None:
        subset = [
            record
            for record in records
            if record["split"] in {*negative_splits, positive_split}
            and record["score"] is not None
        ]
        if not subset:
            return None
        if not any(record["split"] == positive_split for record in subset):
            return None
        if not any(record["split"] in negative_splits for record in subset):
            return None
        return binary_auroc(
            [int(record["label"]) for record in subset],
            [float(record["score"]) for record in subset],
        )

    summary: dict[str, Any] = {
        "class_name": class_name,
        "dataset_name": dataset_name,
        "dataset": f"{DATASET_DISPLAY_NAMES[dataset_name]} {class_name}/test",
        "dataset_root": str(dataset),
        "scope": "+".join(selected_splits),
        "negative_split": "+".join(negative_splits) or None,
        "positive_split": "+".join(positive_splits) or None,
        "positive_splits": list(positive_splits),
        "include_structural_anomalies": include_structural,
        "rule_inference_enabled": rule_inference_enabled,
        "execution_mode": (
            "rule_patch"
            if rule_inference_enabled and patch_enabled
            else "rule_only"
            if rule_inference_enabled
            else "patch_only"
        ),
        "patch_matching_enabled": patch_enabled,
        "patch_bank_path": resolved_patch_bank,
        "patch_dinov2_checkpoint": (
            str(Path(patch_dinov2_checkpoint).expanduser().resolve())
            if patch_enabled
            else None
        ),
        "patch_layers": list(PATCH_MATCHING_LAYERS) if patch_enabled else [],
        "patch_layer_fusion": (
            DEFAULT_PATCH_LAYER_FUSION if patch_enabled else None
        ),
        "patch_primary_layer": None,
        "patch_input_size": 448 if patch_enabled else None,
        "patch_feature_grid": [64, 64] if patch_enabled else None,
        "patch_coreset_enabled": False if patch_enabled else None,
        "patch_top_fraction": None,
        "patch_position_radius": None,
        "patch_threshold": patch_threshold,
        "patch_score_mean": patch_bank.score_mean if patch_enabled else None,
        "patch_score_unbiased_std": (
            patch_bank.score_unbiased_std if patch_enabled else None
        ),
        "patch_calibration_mode": (
            patch_bank.calibration_mode if patch_enabled else None
        ),
        "patch_calibration_count": (
            len(patch_bank.calibration_scores) if patch_enabled else 0
        ),
        "patch_matching": (
            "global_cosine_nearest_neighbor" if patch_enabled else None
        ),
        "patch_layer_aggregation": (
            "mean_anomaly_map" if patch_enabled else None
        ),
        "patch_image_aggregation": "spatial_max" if patch_enabled else None,
        "patch_memory_scope": "full_image_only" if patch_enabled else None,
        "patch_query_batch_size": patch_query_batch_size if patch_enabled else None,
        "patch_extraction_batch_size": (
            patch_extraction_batch_size if patch_enabled else None
        ),
        "score_fusion_policy": (
            "patch_only_no_fusion"
            if not rule_inference_enabled
            else SCORE_FUSION_POLICY
            if patch_enabled
            else "rule_only_raw_violation_max"
        ),
        "sam3_confidence_threshold": (
            sam3_threshold if rule_inference_enabled else None
        ),
        "clip_attribute_confidence_threshold": (
            clip_threshold if rule_inference_enabled else None
        ),
        "sam3_global_fallback_threshold": (
            sam3_threshold if rule_inference_enabled else None
        ),
        "sam3_category_thresholds": (
            dict(sam3_category_thresholds or {})
            if rule_inference_enabled
            else {}
        ),
        "sam3_threshold_policy": (
            sam3_threshold_policy if rule_inference_enabled else "disabled"
        ),
        "sam3_threshold_policy_path": (
            str(Path(sam3_threshold_policy_path).expanduser().resolve())
            if rule_inference_enabled and sam3_threshold_policy_path is not None
            else None
        ),
        "mask_dedup_iou_threshold": (
            mask_dedup_iou if rule_inference_enabled else None
        ),
        "mask_dedup_containment_threshold": (
            mask_dedup_containment if rule_inference_enabled else None
        ),
        "mask_minimum_area_pixels": (
            mask_min_area if rule_inference_enabled else None
        ),
        "mask_cleanup_enabled": (
            mask_cleanup if rule_inference_enabled else None
        ),
        "mask_dedup_enabled": mask_dedup if rule_inference_enabled else None,
        "grouping_tolerance": (
            grouping_tolerance if rule_inference_enabled else None
        ),
        "sam3_prompt": sam3_prompt if rule_inference_enabled else None,
        "length_tolerance_percent": length_tolerance,
        "area_tolerance_percent": area_tolerance,
        "measurement_tag": measurement_tag,
        "attribute_vocabulary": {
            key: list(values) for key, values in sorted(vocabulary.items())
        },
        "sample_count": len(samples),
        "negative_count": sum(record["label"] == 0 for record in records),
        "positive_count": sum(record["label"] == 1 for record in records),
        "valid_score_count": len(valid),
        "failure_count": len(failures),
        "image_auroc": (
            binary_auroc(
                [int(record["label"]) for record in valid],
                [float(record["score"]) for record in valid],
            )
            if valid
            and {int(record["label"]) for record in valid} == {0, 1}
            else None
        ),
        "auroc_excluded_failure_count": len(failures),
        "logical_image_auroc": subgroup_auroc("logical_anomalies"),
        "structural_image_auroc": (
            subgroup_auroc("structural_anomalies")
            if include_structural
            else None
        ),
        "split_image_aurocs": {
            split: subgroup_auroc(split)
            for split in positive_splits
        },
        "elapsed_seconds_this_run": time.perf_counter() - run_started,
        "progress_path": str(progress_path),
        "summary_path": str(summary_path),
        "standard_template": (
            str(standard_file) if standard_file is not None else None
        ),
        "generic_template": str(generic_file) if generic_file is not None else None,
        "failures": [
            {"image": record["image"], "error": record["error"]}
            for record in failures
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def evaluate_screw_bag(
    *,
    dataset_root: str | Path,
    generic_path: str | Path,
    standard_path: str | Path,
    result_root: str | Path,
    sam3_threshold: float,
    sam3_repo: str | Path,
    sam3_checkpoint: str | Path,
    clip_repo: str | Path,
    clip_checkpoint: str | Path,
    clip_threshold: float = 0.35,
    device: str = "cuda",
    mask_dedup_iou: float = 0.80,
    mask_dedup_containment: float = 0.85,
    mask_min_area: int = 16,
    mask_cleanup: bool = True,
    mask_dedup: bool = True,
    rule_inference_enabled: bool = True,
    patch_enabled: bool = False,
    patch_bank_path: str | Path | None = None,
    patch_dinov2_checkpoint: str | Path = DEFAULT_PATCH_DINOV2_CHECKPOINT,
    patch_query_batch_size: int = 256,
    patch_extraction_batch_size: int = 4,
    include_structural_anomalies: bool | None = None,
    sam3_category_thresholds: Mapping[str, float] | None = None,
    sam3_threshold_policy: str = "global_only",
    sam3_threshold_policy_path: str | Path | None = None,
    sam3_prompt: str = "{category}",
    restart: bool = False,
    test_split: str | None = None,
    test_splits: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper for the original evaluation entry point."""

    return evaluate_dataset(
        class_name="screw_bag",
        dataset_root=dataset_root,
        generic_path=generic_path,
        standard_path=standard_path,
        result_root=result_root,
        sam3_threshold=sam3_threshold,
        sam3_repo=sam3_repo,
        sam3_checkpoint=sam3_checkpoint,
        clip_repo=clip_repo,
        clip_checkpoint=clip_checkpoint,
        clip_threshold=clip_threshold,
        device=device,
        sam3_prompt=sam3_prompt,
        mask_dedup_iou=mask_dedup_iou,
        mask_dedup_containment=mask_dedup_containment,
        mask_min_area=mask_min_area,
        mask_cleanup=mask_cleanup,
        mask_dedup=mask_dedup,
        rule_inference_enabled=rule_inference_enabled,
        patch_enabled=patch_enabled,
        patch_bank_path=patch_bank_path,
        patch_dinov2_checkpoint=patch_dinov2_checkpoint,
        patch_query_batch_size=patch_query_batch_size,
        patch_extraction_batch_size=patch_extraction_batch_size,
        include_structural_anomalies=include_structural_anomalies,
        sam3_category_thresholds=sam3_category_thresholds,
        sam3_threshold_policy=sam3_threshold_policy,
        sam3_threshold_policy_path=sam3_threshold_policy_path,
        restart=restart,
        test_split=test_split,
        test_splits=test_splits,
    )
