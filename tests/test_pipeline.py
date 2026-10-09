from __future__ import annotations

from dataclasses import replace
import json
import unittest

import numpy as np

from pro_innovation.errors import ExecutionError
from pro_innovation.models import AttributePrediction, MaskInstance
from pro_innovation.pipeline import LogicAnomalyPipeline
from pro_innovation.templates import parse_template, to_generic_template


def rectangle(y0: int, y1: int, x0: int, x1: int) -> MaskInstance:
    mask = np.zeros((12, 12), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return MaskInstance(mask)


class FakeSegmenter:
    def __init__(self) -> None:
        self.calls = []
        self.pool = {
            "part_a": (rectangle(2, 3, 1, 6), rectangle(6, 8, 2, 4)),
            "part_b": (rectangle(4, 7, 8, 11),),
        }

    def segment(self, image, categories):
        self.calls.append((image, tuple(categories)))
        return {category: self.pool.get(category, ()) for category in categories}


class FakeAttributeDetector:
    values = {
        ("part_a", "attribute.color"): "red",
        ("part_b", "attribute.material"): "metal",
    }

    def detect(self, image, subject, property_name, instances):
        del image, instances
        return AttributePrediction(self.values[(subject, property_name)], 0.9)


class RecordingAttributeDetector:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def detect(self, image, subject, property_name, instances):
        del image, instances
        self.calls.append((subject, property_name))
        value = self.values[(subject, property_name)]
        return AttributePrediction(
            value=value,
            confidence=1.0,
            probabilities={value: 1.0},
        )


class FailingAttributeDetector:
    def detect(self, image, subject, property_name, instances):
        del image, subject, property_name, instances
        raise ExecutionError("attribute evidence unavailable")


STANDARD = """
[C001] TASK=COUNT | SUBJECT=part_a | PROPERTY=count | OP=EQ | VALUE=[2]
[C002] TASK=LENGTH | SUBJECT=part_a | PROPERTY=length | OP=RANGE | VALUE=[4.5,5.5] | COUNT=[1]
[C003] TASK=AREA | SUBJECT=part_a | PROPERTY=area | OP=RANGE | VALUE=[4,5] | COUNT=[2]
[C004] TASK=SPATIAL_COMBINATION | SUBJECT=part_a | PROPERTY=spatial.position | OBJECT=part_b | OP=EQ | VALUE=left_of
[C005] TASK=ATTRIBUTE_ERROR | SUBJECT=part_a | PROPERTY=attribute.color | OP=EQ | VALUE=red
[C006] TASK=ATTRIBUTE_COMBINATION | SUBJECT_1=part_a | PROPERTY_1=attribute.color | SUBJECT_2=part_b | PROPERTY_2=attribute.material | OP=EQ | VALUE=["red:metal","blue:plastic"]
"""


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.standard = parse_template(STANDARD)
        self.generic = to_generic_template(self.standard)
        self.segmenter = FakeSegmenter()
        self.pipeline = LogicAnomalyPipeline(
            self.segmenter, FakeAttributeDetector(), grouping_tolerance=0.10
        )
        self.image = np.zeros((12, 12, 3), dtype=np.uint8)

    def test_all_six_tasks_share_one_segmentation_pass(self) -> None:
        report = self.pipeline.run(self.image, self.generic, self.standard)
        self.assertFalse(report.logical_anomaly)
        self.assertTrue(all(item.passed for item in report.results))
        self.assertEqual(len(self.segmenter.calls), 1)
        self.assertEqual(self.segmenter.calls[0][1], ("part_a", "part_b"))
        self.assertEqual(report.mask_counts, {"part_a": 2, "part_b": 1})
        self.assertAlmostEqual(report.final_anomaly_score, 0.1)
        self.assertIn("VALUE_1=red", report.filled_template)
        self.assertIn("VALUE=red:metal", report.filled_template)
        report_dict = report.to_dict()
        self.assertEqual(report_dict["status"], "NORMAL")
        self.assertIn("satisfaction_score", report_dict["results"][0])
        self.assertIn("violation_score", report_dict["results"][0])
        payload = json.dumps(report_dict)
        self.assertIn('"task": "COUNT"', payload)

    def test_one_failed_rule_sets_logical_anomaly(self) -> None:
        changed = list(self.standard)
        changed[0] = replace(changed[0], expected=(3,))
        report = self.pipeline.run(self.image, self.generic, changed)
        self.assertTrue(report.logical_anomaly)
        self.assertFalse(report.results[0].passed)
        self.assertEqual(report.results[0].violation_score, 1.0)
        self.assertEqual(report.final_anomaly_score, 1.0)
        self.assertEqual(report.results[0].actual, 2)
        self.assertEqual(report.results[0].expected, [3])

    def test_execution_failure_is_unknown_not_product_anomaly(self) -> None:
        pipeline = LogicAnomalyPipeline(
            self.segmenter, FailingAttributeDetector(), grouping_tolerance=0.10
        )
        report = pipeline.run(self.image, self.generic, self.standard)
        self.assertIsNone(report.logical_anomaly)
        failed = report.results[4]
        self.assertIsNone(failed.satisfied)
        self.assertFalse(failed.evidence_valid)
        self.assertIsNone(failed.violation_score)
        self.assertEqual(report.to_dict()["status"], "UNKNOWN")

    def test_length_and_area_compare_range_and_count(self) -> None:
        standard = parse_template(
            "[L001] TASK=LENGTH | SUBJECT=part_a | PROPERTY=length | "
            "OP=RANGE | VALUE=[4.5,5.5] | COUNT=[1,2]\n"
            "[A001] TASK=AREA | SUBJECT=part_a | PROPERTY=area | "
            "OP=RANGE | VALUE=[4,5] | COUNT=[1,2]"
        )
        report = self.pipeline.run(
            self.image,
            to_generic_template(standard),
            standard,
        )
        self.assertFalse(report.logical_anomaly)
        self.assertEqual(report.results[0].actual["matching_count"], 1)
        self.assertEqual(report.results[1].actual["matching_count"], 2)
        self.assertEqual(report.results[0].expected["count"], [1, 2])
        self.assertEqual(report.results[1].expected["count"], [1, 2])
        self.assertIn("VALUE=[4.5,5.5] | COUNT=[1]", report.filled_template)

    def test_count_accepts_any_value_in_array(self) -> None:
        standard = parse_template(
            "[C001] TASK=COUNT | SUBJECT=part_a | PROPERTY=count | "
            "OP=EQ | VALUE=[1,2]"
        )
        report = self.pipeline.run(
            self.image,
            to_generic_template(standard),
            standard,
        )
        self.assertFalse(report.logical_anomaly)
        self.assertTrue(report.results[0].passed)
        self.assertEqual(report.results[0].expected, [1, 2])
        self.assertIn("VALUE=[2]", report.filled_template)

    def test_spatial_count_counts_unique_subjects_not_matching_pairs(self) -> None:
        first_compartment = rectangle(0, 6, 0, 6)
        second_compartment = rectangle(6, 12, 6, 12)
        first_pin = rectangle(1, 2, 1, 2)
        second_pin = rectangle(3, 4, 3, 4)
        self.segmenter.pool = {
            "compartment": (first_compartment, second_compartment),
            "pin": (first_pin, second_pin),
        }
        standard = parse_template(
            "[S001] TASK=SPATIAL_COMBINATION | SUBJECT=compartment | "
            "PROPERTY=spatial.position | OBJECT=pin | OP=EQ | "
            "VALUE=contains | COUNT=[2]"
        )

        report = self.pipeline.run(
            self.image,
            to_generic_template(standard),
            standard,
        )

        result = report.results[0]
        self.assertTrue(report.logical_anomaly)
        self.assertFalse(result.passed)
        self.assertEqual(result.actual, {"relation": "contains", "count": 1})
        self.assertEqual(
            result.expected,
            {"relation": "contains", "count": [2]},
        )
        self.assertEqual(result.details["pair_semantics"], "unique_subject_count")
        self.assertEqual(
            result.details["subject_relation_matches"], [True, False]
        )
        self.assertIn("VALUE=contains | COUNT=[1]", report.filled_template)

    def test_spatial_count_supports_non_containment_relations(self) -> None:
        self.segmenter.pool = {
            "part_a": (
                rectangle(1, 3, 1, 3),
                rectangle(5, 7, 2, 4),
            ),
            "part_b": (rectangle(2, 6, 8, 10),),
        }
        standard = parse_template(
            "[S001] TASK=SPATIAL_COMBINATION | SUBJECT=part_a | "
            "PROPERTY=spatial.position | OBJECT=part_b | OP=EQ | "
            "VALUE=left_of | COUNT=[2]"
        )

        report = self.pipeline.run(
            self.image,
            to_generic_template(standard),
            standard,
        )

        result = report.results[0]
        self.assertFalse(report.logical_anomaly)
        self.assertTrue(result.passed)
        self.assertEqual(result.actual, {"relation": "left_of", "count": 2})
        self.assertEqual(
            result.details["subject_relation_matches"], [True, True]
        )

    def test_spatial_count_ge_accepts_extra_matching_subjects(self) -> None:
        self.segmenter.pool = {
            "almonds": (
                rectangle(1, 3, 1, 3),
                rectangle(5, 7, 2, 4),
            ),
            "box": (rectangle(0, 12, 0, 12),),
        }
        standard = parse_template(
            "[S001] TASK=SPATIAL_COMBINATION | SUBJECT=almonds | "
            "PROPERTY=spatial.position | OBJECT=box | OP=GE | "
            "VALUE=inside | COUNT=[1]"
        )

        report = self.pipeline.run(
            self.image,
            to_generic_template(standard),
            standard,
        )

        result = report.results[0]
        self.assertFalse(report.logical_anomaly)
        self.assertTrue(result.passed)
        self.assertEqual(result.actual, {"relation": "inside", "count": 2})
        self.assertEqual(
            result.expected,
            {"relation": "inside", "count": [1], "operator": "GE"},
        )
        self.assertEqual(
            result.details["score_method"],
            "fuzzy_at_least_unique_subject_relation_count",
        )

    def test_allowed_attribute_combination_uses_or_semantics(self) -> None:
        report = self.pipeline.run(self.image, self.generic, self.standard)
        combination = report.results[5]
        self.assertTrue(combination.passed)
        self.assertEqual(combination.actual["VALUE"], "red:metal")
        self.assertEqual(
            combination.expected["VALUE"],
            ["red:metal", "blue:plastic"],
        )

        changed = list(self.standard)
        changed[5] = replace(changed[5], expected=("blue:metal", "red:plastic"))
        anomaly = self.pipeline.run(self.image, self.generic, changed)
        self.assertTrue(anomaly.logical_anomaly)
        self.assertFalse(anomaly.results[5].passed)

    def test_attribute_count_reuses_shared_mask_count_in_allowed_combination(self) -> None:
        standard = parse_template(
            "[C001] TASK=COUNT | SUBJECT=orange_clip | PROPERTY=count | "
            "OP=EQ | VALUE=[4,6,10]\n"
            "[C004] TASK=ATTRIBUTE_COMBINATION | "
            "SUBJECT_1=orange_clip | PROPERTY_1=attribute.count | "
            "SUBJECT_2=wire | PROPERTY_2=attribute.color | OP=EQ | "
            'VALUE=["10:red","4:yellow","6:blue"]'
        )
        segmenter = FakeSegmenter()
        segmenter.pool = {
            "orange_clip": (
                rectangle(0, 2, 0, 2),
                rectangle(0, 2, 3, 5),
                rectangle(3, 5, 0, 2),
                rectangle(3, 5, 3, 5),
            ),
            "wire": (rectangle(8, 9, 1, 11),),
        }
        detector = RecordingAttributeDetector(
            {("wire", "attribute.color"): "yellow"}
        )
        pipeline = LogicAnomalyPipeline(segmenter, detector)

        report = pipeline.run(
            self.image,
            to_generic_template(standard),
            standard,
        )

        self.assertFalse(report.logical_anomaly)
        self.assertEqual(report.results[0].actual, 4)
        combination = report.results[1]
        self.assertTrue(combination.passed)
        self.assertEqual(
            combination.actual,
            {"VALUE_1": 4, "VALUE_2": "yellow", "VALUE": "4:yellow"},
        )
        count_details = combination.details["attributes"]["VALUE_1"]
        self.assertEqual(count_details["source"], "shared_mask_pool_count")
        self.assertEqual(count_details["candidate_probabilities"]["4"], 1.0)
        self.assertEqual(detector.calls, [("wire", "attribute.color")])
        parse_template(report.filled_template)

    def test_attribute_count_reuses_masks_in_scalar_attribute_rules(self) -> None:
        combination_standard = parse_template(
            "[C001] TASK=ATTRIBUTE_COMBINATION | "
            "SUBJECT_1=part_a | PROPERTY_1=attribute.count | VALUE_1=2 | "
            "SUBJECT_2=part_b | PROPERTY_2=attribute.material | VALUE_2=metal | "
            "OP=EQ | VALUE=2+metal"
        )
        detector = RecordingAttributeDetector(
            {("part_b", "attribute.material"): "metal"}
        )
        pipeline = LogicAnomalyPipeline(self.segmenter, detector)
        combination_report = pipeline.run(
            self.image,
            to_generic_template(combination_standard),
            combination_standard,
        )
        self.assertFalse(combination_report.logical_anomaly)
        self.assertEqual(
            combination_report.results[0].actual["VALUE_1"], 2
        )
        self.assertEqual(detector.calls, [("part_b", "attribute.material")])

        count_standard = parse_template(
            "[C002] TASK=ATTRIBUTE_ERROR | SUBJECT=part_a | "
            "PROPERTY=attribute.count | OP=EQ | VALUE=2"
        )
        count_pipeline = LogicAnomalyPipeline(
            self.segmenter,
            FailingAttributeDetector(),
        )
        count_report = count_pipeline.run(
            self.image,
            to_generic_template(count_standard),
            count_standard,
        )
        self.assertFalse(count_report.logical_anomaly)
        self.assertEqual(count_report.results[0].actual, 2)
        self.assertEqual(
            count_report.results[0].details["source"],
            "shared_mask_pool_count",
        )

    def test_attribute_length_and_area_reuse_shared_mask_geometry(self) -> None:
        standard = parse_template(
            "[C001] TASK=ATTRIBUTE_COMBINATION | "
            "SUBJECT_1=part_b | PROPERTY_1=attribute.area | "
            "SUBJECT_2=part_b | PROPERTY_2=attribute.length | "
            "SUBJECT_3=part_a | PROPERTY_3=attribute.color | OP=EQ | "
            'VALUE=["9:3:red"]'
        )
        detector = RecordingAttributeDetector(
            {("part_a", "attribute.color"): "red"}
        )
        pipeline = LogicAnomalyPipeline(self.segmenter, detector)

        report = pipeline.run(
            self.image,
            to_generic_template(standard),
            standard,
        )

        self.assertFalse(report.logical_anomaly)
        result = report.results[0]
        self.assertEqual(result.actual["VALUE"], "9:3:red")
        self.assertEqual(
            result.details["attributes"]["VALUE_1"]["source"],
            "shared_mask_pool_area",
        )
        self.assertEqual(
            result.details["attributes"]["VALUE_2"]["source"],
            "shared_mask_pool_length",
        )
        self.assertEqual(detector.calls, [("part_a", "attribute.color")])
        parse_template(report.filled_template)

    def test_scalar_attribute_length_and_area_bypass_attribute_detector(self) -> None:
        standard = parse_template(
            "[A001] TASK=ATTRIBUTE_ERROR | SUBJECT=part_b | "
            "PROPERTY=attribute.area | OP=EQ | VALUE=9\n"
            "[L001] TASK=ATTRIBUTE_ERROR | SUBJECT=part_b | "
            "PROPERTY=attribute.length | OP=EQ | VALUE=3"
        )
        pipeline = LogicAnomalyPipeline(
            self.segmenter,
            FailingAttributeDetector(),
        )

        report = pipeline.run(
            self.image,
            to_generic_template(standard),
            standard,
        )

        self.assertFalse(report.logical_anomaly)
        self.assertEqual([item.actual for item in report.results], [9, 3])
        self.assertEqual(
            [item.details["source"] for item in report.results],
            ["shared_mask_pool_area", "shared_mask_pool_length"],
        )

    def test_attribute_spatial_combination_reuses_matching_spatial_rule(self) -> None:
        standard = parse_template(
            "[S001] TASK=SPATIAL_COMBINATION | SUBJECT=part_a | "
            "PROPERTY=spatial.option | OBJECT=part_b | OP=EQ | VALUE=left_of\n"
            "[C001] TASK=ATTRIBUTE_COMBINATION | "
            "SUBJECT_1=part_a | "
            "PROPERTY_1=attribute.spatial_combination.part_b | "
            "SUBJECT_2=part_a | PROPERTY_2=attribute.color | OP=EQ | "
            'VALUE=["left_of:red"]'
        )
        detector = RecordingAttributeDetector(
            {("part_a", "attribute.color"): "red"}
        )
        pipeline = LogicAnomalyPipeline(self.segmenter, detector)

        report = pipeline.run(
            self.image,
            to_generic_template(standard),
            standard,
        )

        self.assertFalse(report.logical_anomaly)
        combination = report.results[1]
        self.assertEqual(combination.actual["VALUE"], "left_of:red")
        spatial_details = combination.details["attributes"]["VALUE_1"]
        self.assertEqual(spatial_details["source"], "spatial_combination_rule")
        self.assertEqual(spatial_details["reused_rule_id"], "S001")
        self.assertEqual(detector.calls, [("part_a", "attribute.color")])
        parse_template(report.filled_template)

    def test_attribute_spatial_combination_requires_unique_matching_rule(self) -> None:
        missing_standard = parse_template(
            "[C001] TASK=ATTRIBUTE_COMBINATION | "
            "SUBJECT_1=part_a | "
            "PROPERTY_1=attribute.spatial_combination.part_b | OP=EQ | "
            'VALUE=["left_of"]'
        )
        missing = self.pipeline.run(
            self.image,
            to_generic_template(missing_standard),
            missing_standard,
        )
        self.assertIsNone(missing.logical_anomaly)
        self.assertIn("needs one SPATIAL_COMBINATION rule", missing.results[0].error)

        ambiguous_standard = parse_template(
            "[S001] TASK=SPATIAL_COMBINATION | SUBJECT=part_a | "
            "PROPERTY=spatial.option | OBJECT=part_b | OP=EQ | VALUE=left_of\n"
            "[S002] TASK=SPATIAL_COMBINATION | SUBJECT=part_a | "
            "PROPERTY=spatial.option | OBJECT=part_b | OP=EQ | VALUE=disjoint\n"
            "[C001] TASK=ATTRIBUTE_COMBINATION | "
            "SUBJECT_1=part_a | "
            "PROPERTY_1=attribute.spatial_combination.part_b | OP=EQ | "
            'VALUE=["left_of"]'
        )
        ambiguous = self.pipeline.run(
            self.image,
            to_generic_template(ambiguous_standard),
            ambiguous_standard,
        )
        self.assertIsNone(ambiguous.results[2].satisfied)
        self.assertIn("ambiguous", ambiguous.results[2].error)

    def test_juice_bottle_relations_and_flavor_allowlist_round_trip(self) -> None:
        standard = parse_template(
            "[C006] TASK=SPATIAL_COMBINATION | SUBJECT=pattern_label | "
            "PROPERTY=spatial.position | OBJECT=juice_bottle | OP=EQ | "
            "VALUE=middle\n"
            "[C007] TASK=SPATIAL_COMBINATION | SUBJECT=text_label | "
            "PROPERTY=spatial.position | OBJECT=juice_bottle | OP=EQ | "
            "VALUE=lower\n"
            "[C008] TASK=SPATIAL_COMBINATION | SUBJECT=fruit_icon | "
            "PROPERTY=spatial.position | OBJECT=pattern_label | OP=EQ | "
            "VALUE=center\n"
            "[C010] TASK=ATTRIBUTE_COMBINATION | SUBJECT_1=fruit_icon | "
            "PROPERTY_1=attribute.type | SUBJECT_2=liquid | "
            "PROPERTY_2=attribute.type | OP=EQ | "
            'VALUE=["cherry:cherry","banana:banana","orange:orange"]'
        )
        segmenter = FakeSegmenter()
        segmenter.pool = {
            "juice_bottle": (rectangle(1, 11, 1, 11),),
            "pattern_label": (rectangle(4, 7, 3, 9),),
            "text_label": (rectangle(8, 10, 3, 9),),
            "fruit_icon": (rectangle(5, 7, 5, 8),),
            "liquid": (rectangle(2, 10, 2, 10),),
        }
        detector = FakeAttributeDetector()
        detector.values = {
            ("fruit_icon", "attribute.type"): "orange",
            ("liquid", "attribute.type"): "orange",
        }
        pipeline = LogicAnomalyPipeline(segmenter, detector)
        report = pipeline.run(
            self.image,
            to_generic_template(standard),
            standard,
        )

        self.assertFalse(report.logical_anomaly)
        self.assertEqual(
            [item.actual for item in report.results[:3]],
            ["middle", "lower", "center"],
        )
        self.assertEqual(report.results[3].actual["VALUE"], "orange:orange")
        parse_template(report.filled_template)

        detector.values[("liquid", "attribute.type")] = "cherry"
        mismatch = pipeline.run(
            self.image,
            to_generic_template(standard),
            standard,
        )
        self.assertTrue(mismatch.logical_anomaly)
        self.assertFalse(mismatch.results[3].passed)
        self.assertEqual(mismatch.results[3].actual["VALUE"], "orange:cherry")


if __name__ == "__main__":
    unittest.main()
