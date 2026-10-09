from __future__ import annotations

import unittest

from pro_innovation.adapters.clip import ClipAttributeDetector
from pro_innovation.errors import ExecutionError


class ClipAttributeTests(unittest.TestCase):
    def test_attribute_confidence_threshold_defaults_to_point_three_five(self) -> None:
        detector = ClipAttributeDetector(
            repo_path="/unused",
            checkpoint_path="/unused/clip.pt",
            vocabulary={},
        )
        self.assertEqual(detector.confidence_threshold, 0.35)

    def test_attribute_confidence_threshold_must_be_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold must be in"):
            ClipAttributeDetector(
                repo_path="/unused",
                checkpoint_path="/unused/clip.pt",
                vocabulary={},
                confidence_threshold=1.01,
            )

    def test_attribute_candidates_are_strictly_subject_scoped(self) -> None:
        detector = ClipAttributeDetector(
            repo_path="/unused",
            checkpoint_path="/unused/clip.pt",
            vocabulary={
                "fruit_icon_type": ["orange", "banana", "cherry"],
                "liquid_type": ["juice", "water"],
            },
        )
        self.assertEqual(
            detector._candidates("fruit_icon", "attribute.type"),
            ("orange", "banana", "cherry"),
        )
        self.assertEqual(
            detector._candidates("liquid", "attribute.type"),
            ("juice", "water"),
        )
        with self.assertRaisesRegex(ExecutionError, "bottle_type"):
            detector._candidates("bottle", "attribute.type")


if __name__ == "__main__":
    unittest.main()
