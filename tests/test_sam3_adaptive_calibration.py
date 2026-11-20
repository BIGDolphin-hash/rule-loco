from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from pro_innovation.adapters.sam3 import Sam3Segmenter
from pro_innovation.cli import build_parser as build_project_parser
from pro_innovation.errors import ModelAdapterError
from pro_innovation.models import MaskInstance
from pro_innovation.sam3_thresholds import Sam3CalibrationConfig, calibrate_sam3_thresholds
from pro_innovation.sam3_thresholds.cli import build_parser
from pro_innovation.templates import parse_template


def _rules(counts):
    return parse_template("\n".join(
        f"[C{index:03}] TASK=COUNT | SUBJECT={category} | PROPERTY=count | "
        f"OP=EQ | VALUE=[{count}]"
        for index, (category, count) in enumerate(counts.items(), 1)
    ))


def _instances(scores):
    instances = []
    for index, score in enumerate(scores):
        mask = np.zeros((2, 8), dtype=bool)
        mask[0, index] = True
        instances.append(MaskInstance(mask, score=score))
    return tuple(instances)


class _Processor:
    def process(self, categories, proposals):
        return proposals


class _ThresholdSegmenter:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def segment(self, image, categories):
        raise AssertionError("calibration must pass the cutoff to proposal generation")

    def segment_at_threshold(self, image, categories, *, threshold):
        self.calls.append((threshold, image, tuple(categories)))
        return {
            category: _instances(
                score for score in self.scores(threshold, image, category)
                if score >= threshold
            )
            for category in categories
        }


class Sam3AdaptiveCalibrationTests(unittest.TestCase):
    def calibrate(self, segmenter, counts=None, **kwargs):
        rules = _rules(counts or {"part": 1})
        return calibrate_sam3_thresholds(
            class_name="demo", normal_images=("a", "b"),
            truth_rules_by_image=(rules, rules), segmenter=segmenter,
            mask_processor=_Processor(), **kwargs,
        )

    def test_first_valid_threshold_stops_retries(self):
        segmenter = _ThresholdSegmenter(lambda *_: [0.95, 0.60])
        result = self.calibrate(segmenter)
        self.assertEqual(result["selected_thresholds"], {"part": 0.80})
        self.assertEqual([call[0] for call in segmenter.calls], [0.80, 0.80])
        self.assertEqual(len(result["calibrations"]["part"]["attempts"]), 1)

    def test_missing_candidates_retry_until_calibrated_then_stop(self):
        segmenter = _ThresholdSegmenter(lambda *_: [0.95, 0.57, 0.30])
        result = self.calibrate(segmenter, {"part": 2})
        entry = result["calibrations"]["part"]
        self.assertEqual(result["status"], "complete")
        self.assertEqual(entry["proposal_threshold"], 0.50)
        self.assertEqual(entry["selected_threshold"], 0.50)
        self.assertEqual(
            [attempt["proposal_threshold"] for attempt in entry["attempts"]],
            [0.80, 0.70, 0.60, 0.50],
        )
        self.assertEqual(
            [attempt["status"] for attempt in entry["attempts"]],
            ["failed", "failed", "failed", "calibrated"],
        )
        self.assertTrue(all(call[0] >= 0.50 for call in segmenter.calls))

    def test_confidence_gap_failure_retries_even_when_candidate_count_is_sufficient(self):
        def scores(threshold, image, category):
            if threshold == 0.80:
                return [0.99, 0.98] if image == "a" else [0.84, 0.83]
            return [0.94, 0.74] if image == "a" else [0.93, 0.73]

        result = self.calibrate(_ThresholdSegmenter(scores))
        attempts = result["calibrations"]["part"]["attempts"]
        self.assertEqual(attempts[0]["failure_reason"], "no_confidence_gap")
        self.assertTrue(all(
            sample["candidate_count_at_proposal"] == 2
            for sample in attempts[0]["samples"]
        ))
        self.assertEqual(attempts[1]["proposal_threshold"], 0.70)
        self.assertEqual(attempts[1]["status"], "calibrated")
        self.assertEqual(result["status"], "complete")

    def test_floor_is_attempted_and_can_succeed(self):
        segmenter = _ThresholdSegmenter(lambda *_: [0.95, 0.46])
        result = self.calibrate(segmenter, {"part": 2})
        entry = result["calibrations"]["part"]
        self.assertEqual(entry["status"], "calibrated")
        self.assertEqual(entry["proposal_threshold"], 0.40)
        self.assertEqual(entry["selected_threshold"], 0.40)
        self.assertEqual(len(entry["attempts"]), 5)

    def test_failure_at_floor_stops_and_keeps_all_reasons(self):
        segmenter = _ThresholdSegmenter(lambda *_: [0.95])
        events = []
        result = self.calibrate(
            segmenter, {"part": 2},
            progress_callback=lambda *event: events.append(event),
        )
        entry = result["calibrations"]["part"]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["selected_thresholds"], {})
        self.assertEqual(result["minimum_proposal_threshold"], 0.40)
        self.assertEqual(result["proposal_threshold_schedule"], [0.8, 0.7, 0.6, 0.5, 0.4])
        self.assertEqual(
            [attempt["proposal_threshold"] for attempt in entry["attempts"]],
            [0.80, 0.70, 0.60, 0.50, 0.40],
        )
        self.assertTrue(all(
            attempt["failure_reason"] == "candidate_count_below_truth_at_proposal"
            for attempt in entry["attempts"]
        ))
        self.assertEqual(min(call[0] for call in segmenter.calls), 0.40)
        self.assertEqual(sum(event[0] == "retry-threshold" for event in events), 4)

    def test_successful_categories_are_frozen_while_failed_categories_retry(self):
        segmenter = _ThresholdSegmenter(
            lambda threshold, image, category:
            [0.95] if category == "part" else [0.90, 0.65]
        )
        result = self.calibrate(segmenter, {"part": 1, "other": 2})
        self.assertEqual(result["selected_thresholds"], {"part": 0.80, "other": 0.60})
        self.assertEqual(len(result["calibrations"]["part"]["attempts"]), 1)
        self.assertEqual(len(result["calibrations"]["other"]["attempts"]), 3)
        self.assertTrue(all(
            categories == ("other",) for _, _, categories in segmenter.calls[2:]
        ))

    def test_explicit_start_uses_remaining_schedule(self):
        result = self.calibrate(
            _ThresholdSegmenter(lambda *_: [0.95, 0.46]),
            {"part": 2}, proposal_threshold=0.60,
        )
        self.assertEqual(result["proposal_threshold"], 0.60)
        self.assertEqual(result["proposal_threshold_schedule"], [0.60, 0.50, 0.40])

    def test_model_errors_propagate_without_lowering_threshold(self):
        def scores(*_):
            raise ModelAdapterError("model unavailable")

        segmenter = _ThresholdSegmenter(scores)
        with self.assertRaisesRegex(ModelAdapterError, "model unavailable"):
            self.calibrate(segmenter)
        self.assertEqual([call[0] for call in segmenter.calls], [0.80])

    def test_cli_and_config_reject_starts_outside_schedule(self):
        for value in (0.90, 0.30, 0.75, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Sam3CalibrationConfig(proposal_threshold=value)
        for parser, prefix in (
            (build_parser(), []),
            (build_project_parser(), ["calibrate-sam3-thresholds"]),
        ):
            for value in ("0.9", "0.3", "0.75"):
                with (
                    self.subTest(value=value, prefix=prefix),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    parser.parse_args([
                        *prefix, "--class-name", "demo", "--normal-dir", "/unused",
                        "--sam3-threshold", value,
                    ])


class Sam3CalibrationAdapterTests(unittest.TestCase):
    def segmenter(self, **kwargs):
        return Sam3Segmenter(
            repo_path=".", checkpoint_path="unused.pt", device="cpu",
            confidence_threshold=0.60, category_thresholds={"part": 0.95}, **kwargs,
        )

    def test_attempt_changes_model_cutoff_and_restores_runtime_thresholds(self):
        segmenter = self.segmenter()
        original_categories = segmenter.category_thresholds
        processor = SimpleNamespace(confidence_threshold=0.60)
        processor.set_confidence_threshold = lambda value: setattr(
            processor, "confidence_threshold", value
        )
        segmenter._processor = processor
        seen = []

        def segment(image, categories):
            seen.append((segmenter.proposal_threshold, processor.confidence_threshold))
            proposals = {"part": _instances(
                score for score in (0.90, 0.75) if score > processor.confidence_threshold
            )}
            return segmenter._apply_thresholds(proposals, categories)

        with patch.object(segmenter, "segment", side_effect=segment):
            high = segmenter.segment_at_threshold(None, ("part",), threshold=0.80)
            low = segmenter.segment_at_threshold(None, ("part",), threshold=0.70)
        self.assertEqual(len(high["part"]), 1)
        self.assertEqual(len(low["part"]), 2)
        self.assertEqual(seen, [(0.80, 0.80), (0.70, 0.70)])
        self.assertEqual(segmenter.confidence_threshold, 0.60)
        self.assertEqual(segmenter.proposal_threshold, 0.60)
        self.assertIs(segmenter.category_thresholds, original_categories)
        self.assertEqual(processor.confidence_threshold, 0.60)

    def test_attempt_restores_thresholds_on_inference_error(self):
        segmenter = self.segmenter()
        with patch.object(segmenter, "segment", side_effect=ModelAdapterError("failed")):
            with self.assertRaises(ModelAdapterError):
                segmenter.segment_at_threshold(None, ("part",), threshold=0.40)
        self.assertEqual(segmenter.proposal_threshold, 0.60)
        self.assertEqual(segmenter.confidence_threshold, 0.60)
        self.assertEqual(segmenter.category_thresholds, {"part": 0.95})

    def test_lowering_cutoff_cannot_reuse_high_threshold_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "sam3.pt"
            checkpoint.write_bytes(b"checkpoint")
            segmenter = Sam3Segmenter(
                repo_path=root, checkpoint_path=checkpoint, device="cpu",
                confidence_threshold=0.80, cache_root=root / "cache", cache_mode="read",
            )
            image = Image.new("RGB", (8, 2))
            segmenter._write_cache(image, "part", _instances([0.95]))
            high = segmenter.segment_at_threshold(image, ("part",), threshold=0.80)
            self.assertEqual(len(high["part"]), 1)
            with self.assertRaisesRegex(ModelAdapterError, "cache miss"):
                segmenter.segment_at_threshold(image, ("part",), threshold=0.70)
            self.assertEqual(segmenter.proposal_threshold, 0.80)
            self.assertEqual(segmenter.confidence_threshold, 0.80)
            self.assertIsNone(segmenter._processor)


if __name__ == "__main__":
    unittest.main()
