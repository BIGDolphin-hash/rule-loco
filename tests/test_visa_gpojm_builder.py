from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "global_patch_ordered" / "build_visa.py"
SPEC = importlib.util.spec_from_file_location("build_visa_gpojm", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SUMMARY_SCRIPT = (
    PROJECT_ROOT / "scripts" / "global_patch_ordered" / "summarize_results.py"
)
SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "summarize_gpojm_results", SUMMARY_SCRIPT
)
assert SUMMARY_SPEC is not None and SUMMARY_SPEC.loader is not None
SUMMARY_MODULE = importlib.util.module_from_spec(SUMMARY_SPEC)
SUMMARY_SPEC.loader.exec_module(SUMMARY_MODULE)


class VisaGpojmBuilderTest(unittest.TestCase):
    def test_load_train_normal_dirs_uses_official_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            normal = root / "candle" / "Data" / "Images" / "Normal"
            anomaly = root / "candle" / "Data" / "Images" / "Anomaly"
            normal.mkdir(parents=True)
            anomaly.mkdir(parents=True)
            train_image = normal / "0000.JPG"
            test_image = normal / "0001.JPG"
            anomaly_image = anomaly / "0002.JPG"
            for image in (train_image, test_image, anomaly_image):
                image.touch()
            split_dir = root / "split_csv"
            split_dir.mkdir()
            with (split_dir / "1cls.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=("object", "split", "label", "image", "mask")
                )
                writer.writeheader()
                writer.writerows(
                    (
                        {
                            "object": "candle",
                            "split": "train",
                            "label": "Normal",
                            "image": str(train_image.relative_to(root)),
                            "mask": "",
                        },
                        {
                            "object": "candle",
                            "split": "test",
                            "label": "Normal",
                            "image": str(test_image.relative_to(root)),
                            "mask": "",
                        },
                        {
                            "object": "candle",
                            "split": "test",
                            "label": "Anomaly",
                            "image": str(anomaly_image.relative_to(root)),
                            "mask": "",
                        },
                    )
                )

            self.assertEqual(
                MODULE._load_train_normal_dirs(root),
                {"candle": normal.resolve()},
            )

    def test_default_joint_tag_matches_existing_convention(self) -> None:
        self.assertEqual(MODULE._gpojm_run_tag(0.30, 0.10), "gpojm_tw030_top010")

    def test_summary_auroc_uses_visa_labels(self) -> None:
        records = [
            {"split": "Normal", "label": 0, "score": 0.1},
            {"split": "Normal", "label": 0, "score": 0.2},
            {"split": "Anomaly", "label": 1, "score": 0.8},
            {"split": "Anomaly", "label": 1, "score": 0.9},
        ]
        self.assertEqual(SUMMARY_MODULE._auroc(records), 1.0)


if __name__ == "__main__":
    unittest.main()
