"""Dataset loading and dependency-free Pixel AUROC metrics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

MVTEC_AD = "mvtec_ad"
VISA = "visa"
SUPPORTED_PIXEL_DATASETS = (MVTEC_AD, VISA)


@dataclass(frozen=True)
class PixelEvaluationSample:
    image_path: Path
    mask_path: Path | None
    split: str

    @property
    def anomalous(self) -> bool:
        return self.mask_path is not None


def collect_pixel_samples(
    *,
    dataset: str,
    dataset_root: str | Path,
    class_name: str,
) -> tuple[PixelEvaluationSample, ...]:
    """Collect official test images and ground-truth masks for one class."""

    root = Path(dataset_root).expanduser().resolve()
    if dataset == MVTEC_AD:
        test_root = root / class_name / "test"
        ground_truth_root = root / class_name / "ground_truth"
        if not test_root.is_dir():
            raise FileNotFoundError(test_root)
        samples: list[PixelEvaluationSample] = []
        for split_path in sorted(path for path in test_root.iterdir() if path.is_dir()):
            for image_path in sorted(split_path.glob("*.png")):
                mask_path = None
                if split_path.name != "good":
                    mask_path = (
                        ground_truth_root
                        / split_path.name
                        / f"{image_path.stem}_mask.png"
                    )
                    if not mask_path.is_file():
                        raise FileNotFoundError(mask_path)
                samples.append(
                    PixelEvaluationSample(
                        image_path=image_path.resolve(),
                        mask_path=None if mask_path is None else mask_path.resolve(),
                        split=split_path.name,
                    )
                )
        if not samples:
            raise ValueError(f"no MVTec AD test images found for {class_name}")
        return tuple(samples)

    if dataset == VISA:
        manifest = root / "split_csv" / "1cls.csv"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        samples = []
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"object", "split", "label", "image", "mask"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"VisA split manifest is missing columns {sorted(missing)}"
                )
            for row in reader:
                if row["object"] != class_name or row["split"].lower() != "test":
                    continue
                image_path = (root / row["image"]).resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                label = row["label"].lower()
                if label == "normal":
                    mask_path = None
                elif label == "anomaly":
                    if not row["mask"]:
                        raise ValueError(f"VisA anomaly has no mask: {image_path}")
                    mask_path = (root / row["mask"]).resolve()
                    if not mask_path.is_file():
                        raise FileNotFoundError(mask_path)
                else:
                    raise ValueError(f"unsupported VisA label {row['label']!r}")
                samples.append(
                    PixelEvaluationSample(
                        image_path=image_path,
                        mask_path=mask_path,
                        split="Normal" if mask_path is None else "Anomaly",
                    )
                )
        if not samples:
            raise ValueError(f"no VisA test images found for {class_name}")
        return tuple(sorted(samples, key=lambda item: (item.split, str(item.image_path))))

    raise ValueError(
        f"unsupported pixel dataset {dataset!r}; expected {SUPPORTED_PIXEL_DATASETS}"
    )


def load_binary_mask(
    sample: PixelEvaluationSample,
    *,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Load a mask as HxW bool, using nearest-neighbor resizing."""

    height, width = int(output_size[0]), int(output_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("pixel evaluation output size must be positive")
    if sample.mask_path is None:
        return np.zeros((height, width), dtype=np.bool_)
    with Image.open(sample.mask_path) as opened:
        mask = opened.convert("L").resize((width, height), resample=Image.Resampling.NEAREST)
        values = np.asarray(mask, dtype=np.uint8)
    return values > 0


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute exact binary AUROC with average handling for score ties."""

    truth = np.asarray(labels, dtype=np.bool_).reshape(-1)
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if truth.shape != values.shape:
        raise ValueError("AUROC labels and scores must have the same shape")
    if not np.isfinite(values).all():
        raise ValueError("AUROC scores contain non-finite values")
    positives = int(truth.sum())
    negatives = int(truth.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires positive and negative samples")

    order = np.argsort(values, kind="quicksort")
    sorted_values = values[order]
    sorted_truth = truth[order]
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(sorted_values[1:] != sorted_values[:-1]).astype(np.int64)
            + 1,
        )
    )
    ends = np.concatenate((starts[1:], np.array([truth.size], dtype=np.int64)))
    counts = ends - starts
    positive_counts = np.add.reduceat(sorted_truth.astype(np.int64), starts)
    negative_counts = counts - positive_counts
    negative_before = np.cumsum(negative_counts, dtype=np.int64) - negative_counts
    concordant = np.sum(
        positive_counts * negative_before
        + 0.5 * positive_counts * negative_counts,
        dtype=np.float64,
    )
    return float(concordant / (positives * negatives))


def pixel_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute exact Pixel AUROC over flattened prediction and mask arrays."""

    return binary_auroc(labels, scores)


def macro_average(summaries: Sequence[dict[str, float]]) -> dict[str, float]:
    if not summaries:
        raise ValueError("macro average requires at least one class")
    return {
        "image_auroc": float(np.mean([item["image_auroc"] for item in summaries])),
        "pixel_auroc": float(np.mean([item["pixel_auroc"] for item in summaries])),
    }
