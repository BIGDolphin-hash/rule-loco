#!/usr/bin/env python3
"""Recompute GPO-JM overall/logical/structural metrics from saved JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pro_innovation.evaluation import binary_auroc


def _auroc(
    records: list[dict], positive_splits: set[str] | None = None
) -> float | None:
    selected = [
        record
        for record in records
        if record.get("score") is not None
        and (
            positive_splits is None
            or int(record["label"]) == 0
            or record.get("split") in positive_splits
        )
    ]
    labels = [int(record["label"]) for record in selected]
    if set(labels) != {0, 1}:
        return None
    return binary_auroc(labels, [float(record["score"]) for record in selected])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    root = Path(args.result_root).expanduser().resolve()
    summaries = []
    for progress_path in sorted(root.glob("*/all.jsonl")):
        records = [
            json.loads(line)
            for line in progress_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        class_name = progress_path.parent.name
        overall = _auroc(records)
        logical = _auroc(records, {"logical_anomalies"})
        structural = _auroc(records, {"structural_anomalies"})
        summary_path = progress_path.parent / "all_summary.json"
        summary: dict[str, object] = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file()
            else {"module": "GPO-JM", "class_name": class_name}
        )
        summary.update(
            {
                "auroc": overall,
                "image_auroc": overall,
                "logical_image_auroc": logical,
                "structural_image_auroc": structural,
            }
        )
        summary_splits = summary.get("splits")
        if isinstance(summary_splits, (list, tuple)) and any(
            split in {"logical_anomalies", "structural_anomalies"}
            for split in summary_splits
        ):
            summary["split_image_aurocs"] = {
                "logical_anomalies": logical,
                "structural_anomalies": structural,
            }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        row = {
            "class_name": class_name,
            "sample_count": summary.get("sample_count"),
            "valid_count": summary.get("valid_count"),
            "failure_count": summary.get("failure_count"),
            "elapsed_seconds": summary.get("elapsed_seconds"),
            "seconds_per_image": summary.get("seconds_per_image"),
            "rule_seconds": summary.get("rule_seconds"),
            "gpojm_seconds": summary.get("gpojm_seconds"),
            "branch_wall_seconds": summary.get("branch_wall_seconds"),
            "branch_overlap_seconds": summary.get("branch_overlap_seconds"),
            "image_auroc": overall,
            "logical_image_auroc": logical,
            "structural_image_auroc": structural,
        }
        summaries.append(row)
        print(json.dumps(row, ensure_ascii=False))

    complete = [row for row in summaries if row["image_auroc"] is not None]
    if complete:
        macro_auroc = {}
        for key in (
            "image_auroc",
            "logical_image_auroc",
            "structural_image_auroc",
        ):
            values = [row[key] for row in complete if row[key] is not None]
            macro_auroc[key] = (
                sum(float(value) for value in values) / len(values)
                if len(values) == len(complete)
                else None
            )
        timed = [
            row
            for row in complete
            if row["elapsed_seconds"] is not None and row["sample_count"] is not None
        ]
        total_samples = sum(int(row["sample_count"]) for row in timed)
        total_elapsed = sum(float(row["elapsed_seconds"]) for row in timed)
        timing = {
            "class_count": len(timed),
            "sample_count": total_samples,
            "elapsed_seconds": total_elapsed,
            "seconds_per_image": (
                total_elapsed / total_samples if total_samples else None
            ),
        }
        for key in (
            "rule_seconds",
            "gpojm_seconds",
            "branch_wall_seconds",
            "branch_overlap_seconds",
        ):
            values = [row[key] for row in timed if row[key] is not None]
            timing[key] = (
                sum(float(value) for value in values)
                if len(values) == len(timed)
                else None
            )
        print(
            json.dumps(
                {
                    "macro_complete_classes": macro_auroc,
                    "timing_complete_classes": timing,
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
