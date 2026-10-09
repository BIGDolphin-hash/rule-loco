from __future__ import annotations

import unittest

import numpy as np

from pro_innovation.attributes import (
    RoutedAttributeDetector,
    SimpleAttributeDetector,
    attribute_vocabulary_key,
)
from pro_innovation.errors import ExecutionError
from pro_innovation.models import MaskInstance


class AttributeTests(unittest.TestCase):
    def test_vocabulary_key_names_the_attribute_owner(self) -> None:
        self.assertEqual(
            attribute_vocabulary_key("juice_bottle", "attribute.color"),
            "juice_bottle_color",
        )

    def test_simple_color_and_shape(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        image[mask] = (220, 35, 35)
        instance = MaskInstance(mask)
        detector = SimpleAttributeDetector(
            color_candidates=["red", "white", "yellow"]
        )
        color = detector.detect(image, "part", "attribute.color", [instance])
        shape = detector.detect(image, "part", "attribute.shape", [instance])
        self.assertEqual(color.value, "red")
        self.assertEqual(
            color.probabilities,
            {"red": 1.0, "white": 0.0, "yellow": 0.0},
        )
        self.assertEqual(shape.value, "square")
        self.assertEqual(shape.probabilities, {"square": 1.0})

    def test_color_is_restricted_to_configured_candidates(self) -> None:
        image = np.full((8, 8, 3), 128, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        detector = SimpleAttributeDetector(
            color_candidates=["red", "white", "yellow"]
        )
        prediction = detector.detect(
            image,
            "liquid",
            "attribute.color",
            [MaskInstance(mask)],
        )
        self.assertIn(prediction.value, {"red", "white", "yellow"})
        self.assertNotEqual(prediction.value, "gray")
        self.assertEqual(
            set(prediction.probabilities), {"red", "white", "yellow"}
        )

    def test_juice_colors_use_the_configured_specialized_palette(self) -> None:
        mask = np.ones((4, 4), dtype=bool)
        instance = MaskInstance(mask)
        detector = SimpleAttributeDetector(
            color_candidates=["red", "light_yellow", "dark_yellow"]
        )
        samples = {
            "red": (220, 35, 35),
            "light_yellow": (245, 230, 140),
            "dark_yellow": (185, 145, 20),
        }

        for expected, rgb in samples.items():
            image = np.full((4, 4, 3), rgb, dtype=np.uint8)
            prediction = detector.detect(
                image, "liquid", "attribute.color", [instance]
            )
            self.assertEqual(prediction.value, expected)

    def test_color_requires_valid_configured_candidates(self) -> None:
        detector = SimpleAttributeDetector()
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.ones((4, 4), dtype=bool)
        with self.assertRaisesRegex(ExecutionError, "configured vocabulary"):
            detector.detect(
                image,
                "liquid",
                "attribute.color",
                [MaskInstance(mask)],
            )
        with self.assertRaisesRegex(ValueError, "unsupported configured colors"):
            SimpleAttributeDetector(color_candidates=["red", "cyan"])

    def test_routed_color_vocabulary_is_subject_scoped(self) -> None:
        image = np.full((4, 4, 3), (220, 35, 35), dtype=np.uint8)
        instance = MaskInstance(np.ones((4, 4), dtype=bool))
        detector = RoutedAttributeDetector(
            semantic_detector=None,
            vocabulary={"liquid_color": ["red", "yellow"]},
        )
        prediction = detector.detect(
            image, "liquid", "attribute.color", [instance]
        )
        self.assertEqual(prediction.value, "red")
        with self.assertRaisesRegex(ExecutionError, "bottle_color"):
            detector.detect(image, "bottle", "attribute.color", [instance])


if __name__ == "__main__":
    unittest.main()
