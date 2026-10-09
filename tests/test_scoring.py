from __future__ import annotations

import unittest

import numpy as np

from pro_innovation.models import AttributePrediction, MaskInstance
from pro_innovation.scoring import (
    attribute_satisfaction_score,
    conjunction_satisfaction_score,
    count_satisfaction_score,
    ranked_count_satisfaction_score,
    relation_count_at_least_satisfaction_score,
    relation_count_satisfaction_score,
)
from pro_innovation.spatial import (
    SpatialConfig,
    evaluate_spatial_relation,
    position_containment_fraction,
    score_spatial_relation,
)


def rectangle(
    y0: int, y1: int, x0: int, x1: int, *, score: float = 1.0
) -> MaskInstance:
    mask = np.zeros((12, 12), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return MaskInstance(mask, score=score)


class ScoringTests(unittest.TestCase):
    def test_count_uses_mask_existence_scores(self) -> None:
        instances = (
            rectangle(1, 3, 1, 3, score=1.0),
            rectangle(4, 6, 1, 3, score=1.0),
            rectangle(7, 9, 1, 3, score=0.2),
        )
        self.assertAlmostEqual(count_satisfaction_score(instances, [2]), 0.8)
        self.assertAlmostEqual(count_satisfaction_score(instances, [3]), 0.2)
        self.assertAlmostEqual(count_satisfaction_score(instances, [2, 3]), 1.0)

    def test_relation_count_uses_fuzzy_exact_cardinality(self) -> None:
        supports = [0.9, 0.8, 0.1]
        self.assertAlmostEqual(
            relation_count_satisfaction_score(supports, [2]), 0.8
        )
        self.assertAlmostEqual(
            relation_count_satisfaction_score(supports, [3]), 0.1
        )
        self.assertAlmostEqual(
            relation_count_satisfaction_score(supports, [0]), 0.1
        )
        self.assertAlmostEqual(
            relation_count_satisfaction_score([], [0]), 1.0
        )

    def test_relation_count_at_least_does_not_penalize_extra_matches(self) -> None:
        supports = [0.9, 0.8, 0.1]
        self.assertAlmostEqual(
            relation_count_at_least_satisfaction_score(supports, [1]), 0.9
        )
        self.assertAlmostEqual(
            relation_count_at_least_satisfaction_score(supports, [2]), 0.8
        )
        self.assertEqual(
            relation_count_at_least_satisfaction_score([], [1]), 0.0
        )

    def test_rank_count_marginalizes_uncertain_mask(self) -> None:
        instances = (
            rectangle(1, 2, 1, 7, score=1.0),
            rectangle(4, 5, 1, 7, score=0.75),
        )
        support, method = ranked_count_satisfaction_score(
            instances,
            measure="length",
            rank=1,
            expected_counts=[2],
            relative_tolerance=0.10,
        )
        self.assertAlmostEqual(support, 0.75)
        self.assertEqual(method, "exact_mask_subset_marginalization")

        either_support, _ = ranked_count_satisfaction_score(
            instances,
            measure="length",
            rank=1,
            expected_counts=[1, 2],
            relative_tolerance=0.10,
        )
        self.assertAlmostEqual(either_support, 1.0)

    def test_attribute_prefers_expected_candidate_probability(self) -> None:
        prediction = AttributePrediction(
            value="blue",
            confidence=0.7,
            probabilities={"red": 0.2, "blue": 0.7, "green": 0.1},
        )
        support, method = attribute_satisfaction_score(prediction, "red")
        self.assertAlmostEqual(support, 0.2)
        self.assertEqual(method, "candidate_probability")

    def test_attribute_combination_uses_weakest_member(self) -> None:
        self.assertAlmostEqual(conjunction_satisfaction_score([0.94, 0.62, 0.89]), 0.62)

    def test_spatial_score_follows_relation_direction(self) -> None:
        subject = rectangle(2, 4, 1, 3, score=0.9)
        object_ = rectangle(2, 4, 8, 10, score=0.8)
        left_support = score_spatial_relation(
            [subject], [object_], "spatial.left_of"
        )
        right_support = score_spatial_relation(
            [subject], [object_], "spatial.right_of"
        )
        self.assertGreater(left_support, 0.7)
        self.assertLess(right_support, 0.01)

    def test_spatial_position_relations_use_object_relative_coordinates(self) -> None:
        object_ = rectangle(1, 11, 1, 11)
        middle = rectangle(5, 7, 5, 7)
        lower = rectangle(8, 10, 5, 7)
        off_center = rectangle(2, 4, 2, 4)

        self.assertTrue(evaluate_spatial_relation([middle], [object_], "middle"))
        self.assertTrue(evaluate_spatial_relation([lower], [object_], "lower"))
        self.assertTrue(evaluate_spatial_relation([middle], [object_], "center"))
        self.assertFalse(
            evaluate_spatial_relation([off_center], [object_], "center")
        )

    def test_containment_uses_filled_bounds_not_raw_mask_overlap(self) -> None:
        outline_mask = np.zeros((12, 12), dtype=bool)
        outline_mask[1, 1:11] = True
        outline_mask[10, 1:11] = True
        outline_mask[1:11, 1] = True
        outline_mask[1:11, 10] = True
        inner_mask = np.zeros((12, 12), dtype=bool)
        inner_mask[4:7, 4:7] = True
        self.assertFalse(np.any(outline_mask & inner_mask))

        container = MaskInstance(outline_mask)
        inner = MaskInstance(inner_mask)
        self.assertEqual(
            position_containment_fraction(inner.mask, container.mask), 1.0
        )
        self.assertTrue(
            evaluate_spatial_relation([container], [inner], "contains")
        )
        self.assertTrue(
            evaluate_spatial_relation([inner], [container], "inside")
        )

    def test_containment_threshold_uses_inner_pixel_fraction(self) -> None:
        container_mask = np.zeros((12, 12), dtype=bool)
        container_mask[2, 2:10] = True
        container_mask[9, 2:10] = True
        container_mask[2:10, 2] = True
        container_mask[2:10, 9] = True
        inner_mask = np.zeros((12, 12), dtype=bool)
        inner_mask[4:7, 4:7] = True
        inner_mask[4, 10] = True
        container = MaskInstance(container_mask)
        inner = MaskInstance(inner_mask)

        self.assertAlmostEqual(
            position_containment_fraction(inner.mask, container.mask), 0.9
        )
        self.assertFalse(
            evaluate_spatial_relation([container], [inner], "contains")
        )
        self.assertTrue(
            evaluate_spatial_relation(
                [container],
                [inner],
                "contains",
                config=SpatialConfig(containment_threshold=0.9),
            )
        )

    def test_middle_relation_rejects_horizontal_and_vertical_misplacement(self) -> None:
        object_mask = np.zeros((100, 100), dtype=bool)
        object_mask[10:90, 10:90] = True
        centered_mask = np.zeros((100, 100), dtype=bool)
        centered_mask[49:51, 49:51] = True
        horizontal_shift_mask = np.zeros((100, 100), dtype=bool)
        horizontal_shift_mask[49:51, 70:72] = True
        vertical_shift_mask = np.zeros((100, 100), dtype=bool)
        vertical_shift_mask[62:64, 49:51] = True

        object_ = MaskInstance(object_mask)
        centered = MaskInstance(centered_mask)
        horizontal_shift = MaskInstance(horizontal_shift_mask)
        vertical_shift = MaskInstance(vertical_shift_mask)

        self.assertEqual(SpatialConfig().middle_min_fraction, 0.40)
        self.assertEqual(SpatialConfig().middle_max_fraction, 0.60)
        self.assertTrue(evaluate_spatial_relation([centered], [object_], "middle"))
        self.assertFalse(
            evaluate_spatial_relation([horizontal_shift], [object_], "middle")
        )
        self.assertFalse(
            evaluate_spatial_relation([vertical_shift], [object_], "middle")
        )

    def test_middle_relation_uses_the_objects_rotated_local_frame(self) -> None:
        object_mask = np.zeros((100, 100), dtype=bool)
        for index in range(10, 90):
            object_mask[index - 2 : index + 3, index] = True
        centered_mask = np.zeros((100, 100), dtype=bool)
        centered_mask[49:52, 49:52] = True
        off_axis_mask = np.zeros((100, 100), dtype=bool)
        off_axis_mask[47:50, 56:59] = True

        object_ = MaskInstance(object_mask)
        centered = MaskInstance(centered_mask)
        off_axis = MaskInstance(off_axis_mask)

        self.assertTrue(evaluate_spatial_relation([centered], [object_], "middle"))
        self.assertFalse(evaluate_spatial_relation([off_axis], [object_], "middle"))

    def test_center_relation_uses_reduced_eight_percent_tolerance(self) -> None:
        object_mask = np.zeros((100, 100), dtype=bool)
        object_mask[10:90, 10:90] = True
        shifted_mask = np.zeros((100, 100), dtype=bool)
        shifted_mask[49:51, 62:64] = True
        object_ = MaskInstance(object_mask)
        shifted = MaskInstance(shifted_mask)

        self.assertEqual(SpatialConfig().center_x_tolerance_fraction, 0.08)
        self.assertEqual(SpatialConfig().center_y_tolerance_fraction, 0.08)
        self.assertFalse(evaluate_spatial_relation([shifted], [object_], "center"))
        self.assertTrue(
            evaluate_spatial_relation(
                [shifted],
                [object_],
                "center",
                config=SpatialConfig(
                    center_x_tolerance_fraction=0.18,
                    center_y_tolerance_fraction=0.18,
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
