#!/usr/bin/env python3
"""Refill LENGTH/AREA standard-template ranges from every train/good image."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any

from pro_innovation.adapters.sam3 import Sam3Segmenter
from pro_innovation.geometry import ranked_measurements
from pro_innovation.mask_processing import MaskPostprocessor
from pro_innovation.models import Rule, TaskType
from pro_innovation.templates import parse_template, serialize_template


DEFAULT_CLASSES = (
    "breakfast_box",
    "juice_bottle",
    "screw_bag",
    "splicing_connectors",
)


def _images(directory: Path) -> tuple[Path, ...]:
    paths = tuple(
        sorted(
            path.resolve()
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
    )
    if not paths:
        raise ValueError(f"no training images found: {directory}")
    return paths


def _measurement_rules(rules: tuple[Rule, ...]) -> tuple[Rule, ...]:
    return tuple(rule for rule in rules if rule.task in {TaskType.LENGTH, TaskType.AREA})


def _value_for_json(value: float, task: TaskType) -> float | int:
    if task is TaskType.AREA:
        return int(round(value))
    return round(float(value), 2)


def fill_class(args: argparse.Namespace, class_name: str, run_dir: Path) -> dict[str, Any]:
    template_path = args.template_root / f"{class_name}.standard.rules"
    threshold_path = args.threshold_root / f"{class_name}.json"
    train_dir = args.dataset_root / class_name / "train" / "good"
    rules = parse_template(template_path.read_text(encoding="utf-8"))
    targets = _measurement_rules(rules)
    if not targets:
        raise ValueError(f"{class_name} has no LENGTH/AREA rules")

    policy = json.loads(threshold_path.read_text(encoding="utf-8"))
    category_thresholds = {
        str(key): float(value)
        for key, value in policy["selected_thresholds"].items()
    }
    subjects = tuple(dict.fromkeys(rule.subject or "" for rule in targets))
    missing_thresholds = [subject for subject in subjects if subject not in category_thresholds]
    if missing_thresholds:
        raise ValueError(f"{class_name} lacks thresholds for: {missing_thresholds}")

    segmenter = Sam3Segmenter(
        repo_path=args.sam3_repo,
        checkpoint_path=args.sam3_checkpoint,
        device=args.device,
        confidence_threshold=args.sam3_threshold,
        category_thresholds={key: category_thresholds[key] for key in subjects},
        prompt_template="{category}",
        cache_root=args.cache_root,
        cache_mode=args.cache_mode,
    )
    processor = MaskPostprocessor(
        dedup_iou_threshold=args.mask_dedup_iou,
        dedup_containment_threshold=args.mask_dedup_containment,
    )
    images = _images(train_dir)
    values: dict[str, list[float]] = {rule.rule_id: [] for rule in targets}
    failures: list[dict[str, Any]] = []

    grouped_rules: dict[tuple[TaskType, str], list[Rule]] = {}
    for rule in targets:
        grouped_rules.setdefault((rule.task, rule.subject or ""), []).append(rule)

    for index, image_path in enumerate(images, start=1):
        try:
            pool = processor.process(subjects, segmenter.segment(image_path, subjects))
            for (task, subject), subject_rules in grouped_rules.items():
                measure = "length" if task is TaskType.LENGTH else "area"
                groups = ranked_measurements(
                    pool.get(subject, ()),
                    measure=measure,
                    relative_tolerance=args.grouping_tolerance,
                )
                if len(groups) < len(subject_rules):
                    raise ValueError(
                        f"{task.value} {subject!r}: needs {len(subject_rules)} ranked "
                        f"groups, found {len(groups)}"
                    )
                for rank, rule in enumerate(subject_rules, start=1):
                    group = groups[rank - 1]
                    expected_count = tuple(rule.count or ())
                    if expected_count and group.count not in expected_count:
                        raise ValueError(
                            f"{rule.rule_id}: rank {rank} count {group.count} not in "
                            f"{list(expected_count)}"
                        )
                    values[rule.rule_id].append(float(group.representative))
        except Exception as exc:
            failures.append({"image": str(image_path), "error": str(exc)})
        if index == 1 or index % 25 == 0 or index == len(images):
            print(
                f"[{class_name}] {index}/{len(images)} "
                f"failures={len(failures)} cache_hits={segmenter.cache_hits} "
                f"cache_misses={segmenter.cache_misses}",
                flush=True,
            )

    report: dict[str, Any] = {
        "class_name": class_name,
        "train_directory": str(train_dir.resolve()),
        "image_count": len(images),
        "valid_image_count": len(images) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
        "cache_hits": segmenter.cache_hits,
        "cache_misses": segmenter.cache_misses,
        "cache_writes": segmenter.cache_writes,
        "rules": {},
        "template_updated": False,
    }
    replacements: dict[str, tuple[float | int, float | int]] = {}
    for rule in targets:
        measured = values[rule.rule_id]
        if measured:
            lower = _value_for_json(min(measured), rule.task)
            upper = _value_for_json(max(measured), rule.task)
            replacements[rule.rule_id] = (lower, upper)
        report["rules"][rule.rule_id] = {
            "task": rule.task.value,
            "subject": rule.subject,
            "rank": grouped_rules[(rule.task, rule.subject or "")].index(rule) + 1,
            "measurement_count": len(measured),
            "minimum": replacements.get(rule.rule_id, (None, None))[0],
            "maximum": replacements.get(rule.rule_id, (None, None))[1],
        }

    class_report_path = run_dir / f"{class_name}.json"
    class_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if failures and not args.allow_incomplete:
        print(f"[{class_name}] not updating template because some images failed", flush=True)
        return report

    updated = tuple(
        replace(rule, minimum=float(replacements[rule.rule_id][0]), maximum=float(replacements[rule.rule_id][1]))
        if rule.rule_id in replacements
        else rule
        for rule in rules
    )
    backup_path = run_dir / template_path.name
    shutil.copy2(template_path, backup_path)
    temporary = template_path.with_suffix(template_path.suffix + ".tmp")
    temporary.write_text(serialize_template(updated), encoding="utf-8")
    parse_template(temporary.read_text(encoding="utf-8"))
    temporary.replace(template_path)
    report["template_updated"] = True
    report["update_policy"] = (
        "all_successful_measurements_after_scanning_every_train_good_image"
        if failures
        else "all_train_good_images"
    )
    report["backup_path"] = str(backup_path)
    class_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[{class_name}] updated {template_path}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/lxq/pro-innovation/datasets/mvtec_loco_anomaly_detection"),
    )
    parser.add_argument("--template-root", type=Path, default=Path("templates/store"))
    parser.add_argument(
        "--threshold-root", type=Path, default=Path("config/sam3_thresholds")
    )
    parser.add_argument("--output-root", type=Path, default=Path("result/template_fill"))
    parser.add_argument("--cache-root", type=Path, default=Path("cache/sam3"))
    parser.add_argument(
        "--cache-mode", choices=Sam3Segmenter.CACHE_MODES, default="read-write"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam3-threshold", type=float, default=0.40)
    parser.add_argument("--sam3-repo", default="/home/lxq/models/SAM3")
    parser.add_argument("--sam3-checkpoint", default="/home/lxq/weights/sam3/sam3.pt")
    parser.add_argument("--mask-dedup-iou", type=float, default=0.80)
    parser.add_argument("--mask-dedup-containment", type=float, default=0.85)
    parser.add_argument("--grouping-tolerance", type=float, default=0.10)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "update from every successful measurement after scanning all images; "
            "retain images without a measurable ranked group in the report"
        ),
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (args.output_root / stamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "classes": [],
        "range_policy": "exact_minimum_and_maximum_over_all_valid_train_good_rank_groups",
        "allow_incomplete": args.allow_incomplete,
        "grouping_tolerance": args.grouping_tolerance,
    }
    for class_name in args.classes:
        summary["classes"].append(fill_class(args, class_name, run_dir))
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"summary: {run_dir / 'summary.json'}", flush=True)
    return 0 if all(item["template_updated"] for item in summary["classes"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
