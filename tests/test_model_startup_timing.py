from __future__ import annotations

import unittest
from unittest.mock import patch

from pro_innovation.adapters import Sam3Segmenter
from scripts.global_patch_ordered.evaluate_rule_gpojm_mvtec_loco import (
    _arguments,
    _initialize_models,
    _memory_execution_summary,
)


class _Initializer:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def initialize(self) -> None:
        self.calls.append(self.name)


class ModelStartupTimingTests(unittest.TestCase):
    def test_models_initialize_before_sample_timing_in_fixed_order(self) -> None:
        calls: list[str] = []
        segmenter = object.__new__(Sam3Segmenter)
        segmenter.initialize = lambda: calls.append("sam3")
        extractor = _Initializer("dinov2", calls)

        components, total = _initialize_models(
            segmenter=segmenter,
            extractor=extractor,
            clip_runtime=None,
            device="cpu",
        )

        self.assertEqual(calls, ["sam3", "dinov2"])
        self.assertEqual(tuple(components), ("sam3", "dinov2"))
        self.assertTrue(all(seconds >= 0.0 for seconds in components.values()))
        self.assertGreaterEqual(total, sum(components.values()))

    def test_no_segmenter_initializes_only_dinov2(self) -> None:
        calls: list[str] = []
        extractor = _Initializer("dinov2", calls)

        components, _ = _initialize_models(
            segmenter=None,
            extractor=extractor,
            clip_runtime=None,
            device="cpu",
        )

        self.assertEqual(calls, ["dinov2"])
        self.assertEqual(tuple(components), ("dinov2",))

    def test_joint_bank_is_optional_for_direct_global_patch_cli(self) -> None:
        argv = [
            "evaluate_rule_gpojm_mvtec_loco.py",
            "--class-name",
            "bottle",
            "--dataset-root",
            "/dataset",
            "--global-bank",
            "/banks/global.zip",
            "--result-root",
            "/results",
        ]
        with patch("sys.argv", argv):
            args = _arguments()

        self.assertEqual(args.memory_module, "gpojm")
        self.assertIsNone(args.threshold_policy)
        self.assertIsNone(args.joint_bank)

    def test_memory_execution_summary_counts_avoidable_parallel_work(self) -> None:
        records = [
            {
                "score": 1.0,
                "rules": [{"violation_score": 1.0}],
                "gpojm_evidence": {"anomaly": False},
                "timings_seconds": {
                    "memory_total": 0.5,
                    "memory_feature_extraction": 0.2,
                    "memory_composition": 0.3,
                },
            },
            {
                "score": 0.2,
                "rules": [{"violation_score": 0.0}],
                "gpojm_evidence": {"anomaly": False},
                "timings_seconds": {
                    "memory_total": 0.4,
                    "memory_feature_extraction": 0.1,
                    "memory_composition": 0.3,
                },
            },
        ]

        summary = _memory_execution_summary(records)

        self.assertEqual(summary["total_images"], 2)
        self.assertEqual(summary["rule_failed_images"], 1)
        self.assertEqual(summary["executed_images"], 2)
        self.assertEqual(summary["executed_after_rule_failure"], 1)
        self.assertAlmostEqual(
            float(summary["total_branch_seconds_after_rule_failure"]), 0.5
        )


if __name__ == "__main__":
    unittest.main()
