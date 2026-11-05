"""Dataset sample collection and AUROC utilities for GPO-JM evaluation."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Mapping, Sequence

MVTEC_LOCO = "mvtec_loco"
MVTEC_AD = "mvtec_ad"
VISA = "visa"
SUPPORTED_DATASETS = (MVTEC_LOCO, MVTEC_AD, VISA)


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
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )
