from __future__ import annotations

import unittest

import numpy as np

from pro_innovation.mask_processing import (
    MaskPostprocessor,
    largest_connected_component,
    mask_containment,
    mask_iou,
)
from pro_innovation.models import MaskInstance


def instance(mask: np.ndarray, score: float) -> MaskInstance:
    return MaskInstance(mask=mask, score=score)


class MaskProcessingTests(unittest.TestCase):
    def test_largest_component_removes_detached_pixels(self) -> None:
        mask = np.zeros((12, 12), dtype=bool)
        mask[2:5, 2:7] = True
        mask[10, 10] = True
        cleaned = largest_connected_component(mask)
        self.assertEqual(int(cleaned.sum()), 15)
        self.assertFalse(cleaned[10, 10])

    def test_iou(self) -> None:
        left = np.zeros((8, 8), dtype=bool)
        right = np.zeros((8, 8), dtype=bool)
        left[1:4, 1:4] = True
        right[2:5, 2:5] = True
        self.assertAlmostEqual(mask_iou(left, right), 4.0 / 14.0)
        self.assertAlmostEqual(mask_containment(left, right), 4.0 / 9.0)

    def test_containment_suppresses_nested_partial_mask(self) -> None:
        complete = np.zeros((16, 16), dtype=bool)
        partial = np.zeros((16, 16), dtype=bool)
        complete[2:12, 2:12] = True
        partial[5:9, 5:9] = True

        self.assertAlmostEqual(mask_iou(complete, partial), 0.16)
        self.assertEqual(mask_containment(complete, partial), 1.0)

        processor = MaskPostprocessor(
            dedup_iou_threshold=0.80,
            dedup_containment_threshold=0.85,
            minimum_area_pixels=1,
        )
        result = processor.process(
            ("fruit_icon",),
            {
                "fruit_icon": (
                    instance(complete, 0.77),
                    instance(partial, 0.625),
                )
            },
        )

        self.assertEqual(len(result["fruit_icon"]), 1)
        self.assertEqual(result["fruit_icon"][0].score, 0.77)

    def test_deduplicates_same_object_but_keeps_separate_objects(self) -> None:
        first = np.zeros((16, 16), dtype=bool)
        duplicate = np.zeros((16, 16), dtype=bool)
        separate_same_size = np.zeros((16, 16), dtype=bool)
        first[2:6, 2:8] = True
        duplicate[2:6, 2:8] = True
        separate_same_size[10:14, 7:13] = True

        processor = MaskPostprocessor(
            dedup_iou_threshold=0.80,
            minimum_area_pixels=1,
        )
        result = processor.process(
            ("screw",),
            {
                "screw": (
                    instance(first, 0.90),
                    instance(duplicate, 0.70),
                    instance(separate_same_size, 0.80),
                )
            },
        )

        self.assertEqual(len(result["screw"]), 2)
        self.assertEqual([item.score for item in result["screw"]], [0.90, 0.80])

    def test_never_deduplicates_across_categories(self) -> None:
        shared = np.zeros((10, 10), dtype=bool)
        shared[2:8, 2:8] = True
        processor = MaskPostprocessor(minimum_area_pixels=1)
        result = processor.process(
            ("screw", "nut"),
            {
                "screw": (instance(shared, 0.90),),
                "nut": (instance(shared, 0.80),),
            },
        )
        self.assertEqual(len(result["screw"]), 1)
        self.assertEqual(len(result["nut"]), 1)

    def test_cleanup_can_be_disabled_for_ablation(self) -> None:
        dirty = np.zeros((12, 12), dtype=bool)
        dirty[2:5, 2:7] = True
        dirty[10, 10] = True
        tiny = np.zeros((12, 12), dtype=bool)
        tiny[0, 0] = True
        processor = MaskPostprocessor(
            minimum_area_pixels=16,
            cleanup_enabled=False,
            dedup_enabled=False,
        )

        result = processor.process(
            ("screw",),
            {"screw": (instance(dirty, 0.90), instance(tiny, 0.80))},
        )

        self.assertEqual(len(result["screw"]), 2)
        self.assertTrue(result["screw"][0].mask[10, 10])

    def test_deduplication_can_be_disabled_for_ablation(self) -> None:
        duplicate = np.zeros((12, 12), dtype=bool)
        duplicate[2:7, 2:7] = True
        processor = MaskPostprocessor(
            minimum_area_pixels=1,
            cleanup_enabled=True,
            dedup_enabled=False,
        )

        result = processor.process(
            ("screw",),
            {
                "screw": (
                    instance(duplicate, 0.90),
                    instance(duplicate.copy(), 0.80),
                )
            },
        )

        self.assertEqual(len(result["screw"]), 2)


if __name__ == "__main__":
    unittest.main()
