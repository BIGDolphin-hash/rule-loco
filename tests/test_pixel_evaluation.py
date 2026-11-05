from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pro_innovation.pixel_evaluation import (
    binary_auroc,
    collect_pixel_samples,
    load_binary_mask,
    pixel_auroc,
)


def _save_image(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


class PixelEvaluationTests(unittest.TestCase):
    def test_pixel_auroc_is_exact_and_handles_ties(self) -> None:
        labels = np.array([0, 1, 0, 1], dtype=np.uint8)
        perfect_scores = np.array([0.0, 1.0, 0.5, 1.0], dtype=np.float32)
        tied_scores = np.ones(4, dtype=np.float32)

        self.assertAlmostEqual(pixel_auroc(labels, perfect_scores), 1.0)
        self.assertAlmostEqual(pixel_auroc(labels, tied_scores), 0.5)

    def test_binary_auroc_supports_image_level_scores(self) -> None:
        labels = np.array([0, 0, 1, 1], dtype=np.uint8)
        scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

        self.assertAlmostEqual(binary_auroc(labels, scores), 1.0)

    def test_mvtec_ad_collection_resolves_ground_truth_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _save_image(
                root / "bottle" / "test" / "good" / "000.png",
                np.zeros((4, 4), dtype=np.uint8),
            )
            _save_image(
                root / "bottle" / "test" / "broken" / "001.png",
                np.zeros((4, 4), dtype=np.uint8),
            )
            _save_image(
                root / "bottle" / "ground_truth" / "broken" / "001_mask.png",
                np.pad(np.ones((2, 2), dtype=np.uint8) * 255, 1),
            )

            samples = collect_pixel_samples(
                dataset="mvtec_ad", dataset_root=root, class_name="bottle"
            )

            self.assertEqual(len(samples), 2)
            anomalous = next(sample for sample in samples if sample.anomalous)
            self.assertEqual(anomalous.split, "broken")
            self.assertEqual(
                load_binary_mask(anomalous, output_size=(8, 8)).shape,
                (8, 8),
            )

    def test_visa_collection_uses_official_test_rows_and_mask_column(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            normal = root / "candle" / "Data" / "Images" / "Normal" / "000.JPG"
            anomaly = root / "candle" / "Data" / "Images" / "Anomaly" / "001.JPG"
            mask = root / "candle" / "Data" / "Masks" / "Anomaly" / "001.png"
            _save_image(normal, np.zeros((4, 4), dtype=np.uint8))
            _save_image(anomaly, np.zeros((4, 4), dtype=np.uint8))
            _save_image(mask, np.ones((4, 4), dtype=np.uint8) * 255)
            split = root / "split_csv" / "1cls.csv"
            split.parent.mkdir(parents=True)
            split.write_text(
                "object,split,label,image,mask\n"
                "candle,train,normal,candle/Data/Images/Normal/skip.JPG,\n"
                "candle,test,normal,candle/Data/Images/Normal/000.JPG,\n"
                "candle,test,anomaly,candle/Data/Images/Anomaly/001.JPG,"
                "candle/Data/Masks/Anomaly/001.png\n",
                encoding="utf-8",
            )

            samples = collect_pixel_samples(
                dataset="visa", dataset_root=root, class_name="candle"
            )

            self.assertEqual(len(samples), 2)
            self.assertEqual(sum(sample.anomalous for sample in samples), 1)


if __name__ == "__main__":
    unittest.main()
