from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pro_innovation.adapters.sam3 import Sam3Segmenter
from pro_innovation.models import MaskInstance
from pro_innovation.sam3_calibration import (
    CALIBRATION_PROPOSAL_THRESHOLD,
    CALIBRATION_SCHEMA,
    calibrate_sam3_thresholds,
    extract_exact_count_truth,
    floor_sam3_threshold,
    load_sam3_threshold_policy,
    save_sam3_threshold_policy,
)
from pro_innovation.templates import parse_template


def _truth(count: int):
    return parse_template(
        "[C001] TASK=COUNT | SUBJECT=part | PROPERTY=count | "
        f"OP=EQ | VALUE=[{count}]"
    )


def _instance(score: float, index: int) -> MaskInstance:
    mask = np.zeros((3, 8), dtype=bool)
    mask[index % 3, index] = True
    return MaskInstance(mask=mask, score=score)


class _FakeSegmenter:
    def __init__(self, scores_by_image):
        self.scores_by_image = scores_by_image

    def segment(self, image, categories):
        return {
            category: tuple(
                _instance(score, index)
                for index, score in enumerate(self.scores_by_image[image])
            )
            for category in categories
        }


class _IdentityProcessor:
    def process(self, categories, instances):
        return {category: tuple(instances.get(category, ())) for category in categories}


class Sam3CalibrationTests(unittest.TestCase):
    def test_truth_count_is_read_per_sample_instead_of_hardcoded(self) -> None:
        images = ("a", "b", "c", "d")
        result = calibrate_sam3_thresholds(
            class_name="demo",
            normal_images=images,
            truth_rules_by_image=(_truth(1), _truth(2), _truth(3), _truth(1)),
            segmenter=_FakeSegmenter(
                {
                    "a": [0.90, 0.70],
                    "b": [0.92, 0.85, 0.68],
                    "c": [0.95, 0.89, 0.83, 0.66],
                    "d": [0.91, 0.69],
                }
            ),
            mask_processor=_IdentityProcessor(),
        )

        self.assertEqual(result["status"], "complete")
        samples = result["calibrations"]["part"]["samples"]
        self.assertEqual([sample["true_count"] for sample in samples], [1, 2, 3, 1])
        selected = result["selected_thresholds"]["part"]
        self.assertEqual(selected, 0.70)
        self.assertGreater(
            result["calibrations"]["part"]["raw_selected_threshold"],
            selected,
        )
        self.assertFalse(result["model_parameters_updated"])

    def test_selected_threshold_is_floored_to_0_10_grid(self) -> None:
        self.assertEqual(floor_sam3_threshold(0.719845), 0.70)
        self.assertEqual(floor_sam3_threshold(0.700000), 0.70)
        self.assertEqual(floor_sam3_threshold(0.6999999999999999), 0.70)
        self.assertEqual(floor_sam3_threshold(0.699999), 0.60)
        self.assertEqual(floor_sam3_threshold(0.612345), 0.60)
        self.assertEqual(floor_sam3_threshold(0.392345), 0.30)

    def test_missing_extra_candidate_uses_0_10_as_conservative_bound(self) -> None:
        result = calibrate_sam3_thresholds(
            class_name="demo",
            normal_images=("a", "b", "c", "d"),
            truth_rules_by_image=tuple(_truth(1) for _ in range(4)),
            segmenter=_FakeSegmenter(
                {key: [0.90] for key in ("a", "b", "c", "d")}
            ),
            mask_processor=_IdentityProcessor(),
        )
        entry = result["calibrations"]["part"]
        self.assertEqual(entry["lower_bound_q95_extra"], 0.10)
        self.assertTrue(
            all(
                sample["extra_score_upper_bound"] == 0.10
                for sample in entry["samples"]
            )
        )

    def test_fewer_candidates_than_truth_at_0_10_fails_category(self) -> None:
        result = calibrate_sam3_thresholds(
            class_name="demo",
            normal_images=("a", "b", "c", "d"),
            truth_rules_by_image=tuple(_truth(2) for _ in range(4)),
            segmenter=_FakeSegmenter(
                {
                    "a": [0.90],
                    "b": [0.92, 0.80],
                    "c": [0.93, 0.81],
                    "d": [0.94, 0.82],
                }
            ),
            mask_processor=_IdentityProcessor(),
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["selected_thresholds"], {})
        self.assertEqual(
            result["calibrations"]["part"]["failure_reason"],
            "candidate_count_below_truth_at_proposal",
        )

    def test_allowed_count_set_is_not_guessed_as_sample_truth(self) -> None:
        rules = parse_template(
            "[C001] TASK=COUNT | SUBJECT=clip | PROPERTY=count | "
            "OP=EQ | VALUE=[4,6,10]"
        )
        with self.assertRaisesRegex(ValueError, "sample-specific truth template"):
            extract_exact_count_truth(rules)

    def test_area_count_is_also_used_as_a_calibration_truth(self) -> None:
        rules = parse_template(
            "[C001] TASK=COUNT | SUBJECT=fruit | PROPERTY=count | "
            "OP=EQ | VALUE=[1]\n"
            "[A001] TASK=AREA | SUBJECT=fruit_region | PROPERTY=area | "
            "OP=RANGE | VALUE=[100,200] | COUNT=[1]"
        )
        self.assertEqual(
            extract_exact_count_truth(rules),
            {"fruit": 1, "fruit_region": 1},
        )

    def test_policy_round_trip_preserves_selected_thresholds(self) -> None:
        payload = {
            "schema": CALIBRATION_SCHEMA,
            "class_name": "demo",
            "calibration_samples": 4,
            "calibration_scope": "normal_only",
            "proposal_threshold": CALIBRATION_PROPOSAL_THRESHOLD,
            "threshold_grid_step": 0.10,
            "threshold_rounding": "floor",
            "model_parameters_updated": False,
            "selected_thresholds": {"part": 0.70},
            "calibrations": {
                "part": {"selected_threshold": 0.70},
            },
            "configuration": {"sam3_prompt": "{category}"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = save_sam3_threshold_policy(
                payload, Path(directory) / "demo.json"
            )
            loaded = load_sam3_threshold_policy(
                path, expected_class_name="demo"
            )
            saved_text = path.read_text(encoding="utf-8")
        self.assertIn('"part": 0.70', saved_text)
        self.assertIn('"selected_threshold": 0.70', saved_text)
        self.assertEqual(loaded.selected_thresholds, {"part": 0.70})
        self.assertEqual(loaded.proposal_threshold, 0.10)

    def test_policy_loader_accepts_older_schema_versions(self) -> None:
        payload = {
            "schema": "pro-innovation.sam3-object-thresholds.v3",
            "class_name": "demo",
            "calibration_samples": 4,
            "calibration_scope": "normal_only",
            "proposal_threshold": 0.60,
            "threshold_grid_step": 0.10,
            "threshold_rounding": "floor",
            "model_parameters_updated": False,
            "selected_thresholds": {"part": 0.70},
            "configuration": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_sam3_threshold_policy(
                path, expected_class_name="demo"
            )
        self.assertEqual(loaded.selected_thresholds, {"part": 0.70})
        self.assertEqual(loaded.proposal_threshold, 0.60)

    def test_policy_loader_still_rejects_unrelated_json_schema(self) -> None:
        payload = {
            "schema": "unrelated.v3",
            "class_name": "demo",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "foreign.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported.*schema"):
                load_sam3_threshold_policy(path)

    def test_segmenter_uses_lowest_threshold_for_proposal_generation(self) -> None:
        segmenter = Sam3Segmenter(
            repo_path=".",
            checkpoint_path="checkpoint.pt",
            confidence_threshold=0.70,
            category_thresholds={"part": 0.64, "other": 0.82},
        )
        self.assertEqual(segmenter.proposal_threshold, 0.64)
        self.assertEqual(segmenter.category_thresholds["other"], 0.82)

    def test_calibration_threshold_can_be_dynamic(self) -> None:
        result = calibrate_sam3_thresholds(
            class_name="demo",
            normal_images=("a", "b", "c", "d"),
            truth_rules_by_image=tuple(_truth(1) for _ in range(4)),
            segmenter=_FakeSegmenter(
                {key: [0.9] for key in ("a", "b", "c", "d")}
            ),
            mask_processor=_IdentityProcessor(),
            proposal_threshold=0.60,
        )
        self.assertEqual(result["proposal_threshold"], 0.60)

    def test_calibration_threshold_must_be_in_probability_range(self) -> None:
        with self.assertRaisesRegex(ValueError, r"within \[0\.0, 1\.0\]"):
            calibrate_sam3_thresholds(
                class_name="demo",
                normal_images=("a", "b", "c", "d"),
                truth_rules_by_image=tuple(_truth(1) for _ in range(4)),
                segmenter=_FakeSegmenter(
                    {key: [0.9] for key in ("a", "b", "c", "d")}
                ),
                mask_processor=_IdentityProcessor(),
                proposal_threshold=1.1,
            )


if __name__ == "__main__":
    unittest.main()
