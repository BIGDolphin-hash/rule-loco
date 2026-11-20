from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from pro_innovation import sam3_calibration
from pro_innovation.cli import build_parser as build_project_parser
from pro_innovation.models import MaskInstance
from pro_innovation.sam3_thresholds import (
    Sam3CalibrationConfig,
    calibrate_sam3_thresholds,
    extract_exact_count_truth,
    generate_sam3_thresholds,
    load_sam3_threshold_policy,
    prepare_calibration_inputs,
)
from pro_innovation.sam3_thresholds.cli import build_parser, main


def _count_rule(subject: str, values: str, rule_id: str = "C001") -> str:
    return (
        f"[{rule_id}] TASK=COUNT | SUBJECT={subject} | PROPERTY=count | "
        f"OP=EQ | VALUE=[{values}]"
    )


class _Segmenter:
    def __init__(self, scores=None):
        self.scores = scores or {"part": [0.90, 0.65], "other": [0.30]}
        self.calls = []

    def segment(self, image, categories):
        self.calls.append((image, tuple(categories)))
        instances = {}
        for category in categories:
            masks = []
            for index, score in enumerate(self.scores[category]):
                mask = np.zeros((2, 4), dtype=bool)
                mask[0, index] = True
                masks.append(MaskInstance(mask=mask, score=score))
            instances[category] = tuple(masks)
        return instances


class Sam3ThresholdGenerationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = tuple(self.root / f"{name}.png" for name in ("a", "b", "c"))
        for path in self.images:
            Image.new("RGB", (4, 2)).save(path)
        self.standard = self.root / "demo.standard.rules"
        self.standard.write_text(_count_rule("part", "1,2,3"), encoding="utf-8")

    def prepare(self, **kwargs):
        return prepare_calibration_inputs(
            standard_path=self.standard, normal_images=self.images, **kwargs
        )

    def test_directory_uses_sorted_readable_images_and_standard_truth(self):
        self.standard.write_text(_count_rule("part", "3"), encoding="utf-8")
        self.images[1].write_bytes(b"corrupt PNG")
        events = []
        inputs = prepare_calibration_inputs(
            standard_path=self.standard,
            normal_dir=self.root,
            progress_callback=lambda *event: events.append(event),
        )
        self.assertEqual(inputs.normal_images, (self.images[0], self.images[2]))
        self.assertEqual(inputs.truth_template_paths, (self.standard,) * 2)
        self.assertEqual(
            [extract_exact_count_truth(rules) for rules in inputs.truth_rules_by_image],
            [{"part": 3}, {"part": 3}],
        )
        self.assertEqual(events[0][0], "skip-unreadable")

    def test_single_subject_counts_follow_original_order_after_image_skip(self):
        self.images[1].write_bytes(b"corrupt PNG")
        original = self.standard.read_bytes()
        inputs = self.prepare(truth_counts=[1, 99, 2])
        self.assertEqual(
            [extract_exact_count_truth(rules) for rules in inputs.truth_rules_by_image],
            [{"part": 1}, {"part": 2}],
        )
        self.assertIsNone(inputs.truth_template_paths)
        self.assertEqual(self.standard.read_bytes(), original)

    def test_multiple_subject_counts_remain_aligned_after_image_skip(self):
        self.standard.write_text(
            _count_rule("part", "1,2,3") + "\n" + _count_rule("other", "1,2", "C002"),
            encoding="utf-8",
        )
        self.images[1].write_bytes(b"corrupt PNG")
        inputs = self.prepare(truth_subject_counts={"part": [1, 99, 2], "other": [2, 99, 1]})
        self.assertEqual(
            [extract_exact_count_truth(rules) for rules in inputs.truth_rules_by_image],
            [{"part": 1, "other": 2}, {"part": 2, "other": 1}],
        )
        with self.assertRaisesRegex(ValueError, "missing inline truth counts.*other"):
            self.prepare(truth_subject_counts={"part": [1, 2, 3]})

    def test_truth_template_paths_remain_aligned_after_image_skip(self):
        paths = tuple(self.root / f"truth_{i}.rules" for i in range(3))
        for path, count in zip(paths, (1, 99, 2)):
            path.write_text(_count_rule("part", str(count)), encoding="utf-8")
        self.images[1].write_bytes(b"corrupt PNG")
        inputs = self.prepare(truth_templates=paths)
        self.assertEqual(inputs.truth_template_paths, (paths[0], paths[2]))
        self.assertEqual(
            [extract_exact_count_truth(rules) for rules in inputs.truth_rules_by_image],
            [{"part": 1}, {"part": 2}],
        )

    def test_counts_are_validated_against_original_image_count(self):
        self.images[1].write_bytes(b"corrupt PNG")
        with self.assertRaisesRegex(ValueError, "need 3 values, found 2"):
            self.prepare(truth_counts=[1, 2])
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            self.prepare(truth_counts=[1, -1, 2])

    def test_conflicting_truth_sources_and_orphan_subject_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "use only one"):
            self.prepare(truth_counts=[1, 2, 3], truth_subject_counts={"part": [1, 2, 3]})
        with self.assertRaisesRegex(ValueError, "requires --truth-counts"):
            self.prepare(truth_subject="part")

    def test_at_least_two_images_must_remain_after_skips(self):
        for path in self.images[:2]:
            path.write_bytes(b"corrupt PNG")
        with self.assertRaisesRegex(ValueError, "two decodable normal images"):
            self.prepare()

    def test_generation_saves_partial_policy_and_full_report(self):
        self.standard.write_text(
            _count_rule("part", "1") + "\n" + _count_rule("other", "1", "C002"),
            encoding="utf-8",
        )
        inputs = self.prepare()
        segmenter = _Segmenter()
        events = []
        target = self.root / "policies" / "demo.json"
        result = generate_sam3_thresholds(
            class_name="demo",
            inputs=inputs,
            config=Sam3CalibrationConfig(proposal_threshold=0.60),
            output=target,
            result_root=self.root / "reports",
            segmenter=segmenter,
            progress_callback=lambda *event: events.append(event),
        )
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["selected_thresholds"], {"part": 0.70})
        self.assertEqual(
            result["calibrations"]["other"]["failure_reason"],
            "candidate_count_below_truth_at_proposal",
        )
        self.assertEqual(len(segmenter.calls), 9)
        self.assertTrue(all(categories == ("other",) for _, categories in segmenter.calls[3:]))
        self.assertEqual(events[-1][0], "saved")
        policy = load_sam3_threshold_policy(target, expected_class_name="demo")
        self.assertEqual(policy.selected_thresholds, {"part": 0.70})
        self.assertEqual(policy.proposal_threshold, 0.60)
        self.assertEqual(
            set(json.loads(target.read_text())),
            {"selected_thresholds", "calibration_images", "initial_threshold"},
        )
        report = json.loads(Path(result["calibration_result_path"]).read_text())
        self.assertEqual(report, result)

    def test_overwrite_guard_prevents_inference_and_preserves_policy(self):
        inputs = self.prepare(truth_counts=[1, 1, 1])
        target = self.root / "demo.json"
        target.write_text("existing policy", encoding="utf-8")
        segmenter = Mock()
        with self.assertRaisesRegex(ValueError, "already exists"):
            generate_sam3_thresholds(
                class_name="demo", inputs=inputs, output=target, segmenter=segmenter
            )
        segmenter.segment.assert_not_called()
        self.assertEqual(target.read_text(), "existing policy")

    def test_report_cannot_overwrite_policy_at_same_path(self):
        inputs = self.prepare(truth_counts=[1, 1, 1])
        target = self.root / "policies" / "demo.json"
        segmenter = Mock()
        with self.assertRaisesRegex(ValueError, "different paths"):
            generate_sam3_thresholds(
                class_name="demo", inputs=inputs, output=target,
                result_root=self.root, segmenter=segmenter,
            )
        segmenter.segment.assert_not_called()
        self.assertFalse(target.exists())

    def test_unresolved_standard_count_set_is_rejected_before_model_creation(self):
        with self.assertRaisesRegex(ValueError, "a.png.*ambiguous COUNT truth"):
            self.prepare()
        with (
            patch("pro_innovation.sam3_thresholds.generator.Sam3Segmenter") as factory,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = main([
                "--class-name", "demo", "--normal-dir", str(self.root),
                "--standard", str(self.standard),
                "--output", str(self.root / "policies/demo.json"),
            ])
        self.assertEqual(code, 2)
        factory.assert_not_called()
        self.assertFalse((self.root / "policies/demo.json").exists())

    def test_truth_template_with_count_set_identifies_image_and_template(self):
        paths = tuple(self.root / f"truth_{i}.rules" for i in range(3))
        for path, values in zip(paths, ("1", "1,2", "2")):
            path.write_text(_count_rule("part", values), encoding="utf-8")
        with self.assertRaisesRegex(
            ValueError, "b.png.*truth_1.rules.*ambiguous COUNT truth"
        ):
            self.prepare(truth_templates=paths)

    def test_one_subject_override_cannot_leave_other_count_sets_unresolved(self):
        self.standard.write_text(
            _count_rule("part", "1,2,3") + "\n" + _count_rule("other", "1,2", "C002"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "missing inline truth counts.*other"):
            self.prepare(truth_counts=[1, 2, 1], truth_subject="part")

    def test_all_image_templates_must_cover_standard_count_subjects(self):
        self.standard.write_text(
            _count_rule("part", "1,2,3") + "\n" + _count_rule("other", "1", "C002"),
            encoding="utf-8",
        )
        path = self.root / "incomplete.rules"
        path.write_text(_count_rule("part", "1"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing=.*other"):
            self.prepare(truth_templates=[path] * 3)
        path.write_text(
            _count_rule("part", "1") + "\n" + _count_rule("other", "1", "C002")
            + "\n" + _count_rule("extra", "1", "C003"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "extra=.*extra"):
            self.prepare(truth_templates=[path] * 3)

    def test_total_count_override_preserves_measurement_subgroup_counts(self):
        self.standard.write_text(
            _count_rule("part", "2,3") + "\n"
            "[L001] TASK=LENGTH | SUBJECT=part | PROPERTY=length | "
            "OP=RANGE | VALUE=[10,20] | COUNT=[1]\n"
            "[L002] TASK=LENGTH | SUBJECT=part | PROPERTY=length | "
            "OP=RANGE | VALUE=[30,40] | COUNT=[2]",
            encoding="utf-8",
        )
        inputs = self.prepare(truth_counts=[3, 3, 3])
        for rules in inputs.truth_rules_by_image:
            self.assertEqual(extract_exact_count_truth(rules), {"part": 3})
            self.assertEqual([rule.count for rule in rules[1:]], [(1,), (2,)])

    def test_inline_counts_can_resolve_unknown_truth_per_image(self):
        self.standard.write_text(_count_rule("part", "unknown"), encoding="utf-8")
        inputs = self.prepare(truth_counts=[1, 2, 1], truth_subject="part")
        self.assertEqual(
            [extract_exact_count_truth(rules) for rules in inputs.truth_rules_by_image],
            [{"part": 1}, {"part": 2}, {"part": 1}],
        )

    def test_report_pairs_unsorted_images_with_exact_templates_after_skip(self):
        images = (self.images[2], self.images[1], self.images[0])
        paths = tuple(self.root / f"truth_{i}.rules" for i in range(3))
        for path, count in zip(paths, (2, 3, 1)):
            path.write_text(_count_rule("part", str(count)), encoding="utf-8")
        self.images[1].write_bytes(b"corrupt PNG")
        inputs = prepare_calibration_inputs(
            standard_path=self.standard, normal_images=images, truth_templates=paths
        )
        segmenter = Mock()
        counts = {self.images[2]: 2, self.images[0]: 1}
        segmenter.segment.side_effect = lambda image, categories: _Segmenter(
            {"part": [0.90] * counts[image] + [0.40]}
        ).segment(image, categories)
        result = generate_sam3_thresholds(
            class_name="demo", inputs=inputs,
            output=self.root / "policies/demo.json",
            result_root=self.root / "reports", segmenter=segmenter,
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["image_truths"], [
            {"sample_index": 0, "image": str(images[0]),
             "truth_template": str(paths[0]), "true_counts": {"part": 2}},
            {"sample_index": 1, "image": str(images[2]),
             "truth_template": str(paths[2]), "true_counts": {"part": 1}},
        ])
        self.assertEqual(
            [sample["count_at_selected_threshold"]
             for sample in result["calibrations"]["part"]["samples"]],
            [2, 1],
        )
        report = json.loads(Path(result["calibration_result_path"]).read_text())
        self.assertEqual(report["image_truths"], result["image_truths"])

    def test_legacy_module_reexports_current_api(self):
        self.assertIs(sam3_calibration.calibrate_sam3_thresholds, calibrate_sam3_thresholds)
        self.assertIs(sam3_calibration.load_sam3_threshold_policy, load_sam3_threshold_policy)

    def test_existing_saved_policies_remain_readable(self):
        root = Path(__file__).resolve().parents[1] / "config/sam3_thresholds"
        paths = tuple(root.rglob("*.json"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path):
                policy = load_sam3_threshold_policy(path, expected_class_name=path.stem)
                self.assertTrue(policy.selected_thresholds)

    def test_both_cli_entry_points_generate_equivalent_artifacts(self):
        common = [
            "--class-name", "demo", "--normal-dir", str(self.root),
            "--standard", str(self.standard), "--truth-counts", "1", "1", "1",
            "--sam3-threshold", "0.60", "--sam3-prompt", "a {category}",
            "--calibration-result-root", str(self.root / "reports"),
        ]
        policies = []
        for entry in ("standalone", "project"):
            target = self.root / entry / "demo.json"
            arguments = common + ["--output", str(target)]
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch(
                    "pro_innovation.sam3_thresholds.generator.Sam3Segmenter",
                    return_value=_Segmenter(),
                ) as factory,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                if entry == "standalone":
                    code = main(arguments)
                else:
                    args = build_project_parser().parse_args(
                        ["calibrate-sam3-thresholds", *arguments]
                    )
                    code = args.func(args)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "complete")
            self.assertIn("[sam3-threshold][saved]", stderr.getvalue())
            self.assertEqual(factory.call_args.kwargs["confidence_threshold"], 0.60)
            self.assertEqual(factory.call_args.kwargs["prompt_template"], "a {category}")
            policies.append(json.loads(target.read_text()))
        self.assertEqual(policies[0], policies[1])

    def test_cli_partial_calibration_returns_two_and_keeps_report(self):
        arguments = [
            "--class-name", "demo", "--normal-dir", str(self.root),
            "--standard", str(self.standard), "--truth-counts", "1", "1", "1",
            "--output", str(self.root / "policies/demo.json"),
            "--calibration-result-root", str(self.root / "reports"),
        ]
        stdout = io.StringIO()
        with (
            patch(
                "pro_innovation.sam3_thresholds.generator.Sam3Segmenter",
                return_value=_Segmenter({"part": []}),
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(arguments), 2)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["status"], "failed")
        self.assertTrue(Path(result["calibration_result_path"]).is_file())

    def test_saved_report_preserves_retry_history_and_initial_threshold(self):
        result = generate_sam3_thresholds(
            class_name="demo", inputs=self.prepare(truth_counts=[1, 1, 1]),
            output=self.root / "policies/demo.json", result_root=self.root / "reports",
            segmenter=_Segmenter({"part": [0.52]}),
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["selected_thresholds"], {"part": 0.50})
        report_path = Path(result["calibration_result_path"])
        report = json.loads(report_path.read_text())
        self.assertEqual(report, result)
        self.assertEqual(
            [attempt["proposal_threshold"]
             for attempt in report["calibrations"]["part"]["attempts"]],
            [0.80, 0.70, 0.60, 0.50],
        )
        policy = json.loads(Path(result["saved_path"]).read_text())
        self.assertEqual(policy["initial_threshold"], 0.80)
        self.assertEqual(policy["selected_thresholds"], {"part": 0.50})
        self.assertIn('"proposal_threshold": 0.50', report_path.read_text())

    def test_cli_rejects_conflicting_truth_sources(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args([
                "--class-name", "demo", "--normal-dir", str(self.root),
                "--truth-counts", "1", "1", "1",
                "--truth-subject-counts", "part", "1", "1", "1",
            ])


if __name__ == "__main__":
    unittest.main()
