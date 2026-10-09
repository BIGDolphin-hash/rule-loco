from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch

from pro_innovation.cli import (
    _make_patch_scorer,
    _patch_bank_path,
    _validate_runtime_branches,
    build_parser,
)
from pro_innovation.evaluation import (
    MVTEC_AD,
    VISA,
    _collect_evaluation_samples,
    _split_specs_for_dataset,
    binary_auroc,
    build_run_stem,
    evaluate_dataset,
    infer_length_tolerance_percent,
    infer_measurement_tag,
    validate_attribute_vocabulary,
)
from pro_innovation.models import PatchEvidence
from pro_innovation.templates import parse_template


class EvaluationTests(unittest.TestCase):
    def test_sam3_runtime_defaults_are_point_four(self) -> None:
        parser = build_parser()
        commands = (
            ["infer", "--image", "sample.png"],
            ["evaluate", "--class-name", "breakfast_box"],
            ["evaluate-screw-bag"],
        )
        for command in commands:
            with self.subTest(command=command[0]):
                self.assertEqual(parser.parse_args(command).sam3_threshold, 0.40)

    def test_patch_only_evaluation_skips_templates_sam3_and_clip(self) -> None:
        class FixedScorer:
            def score(self, image):
                path = Path(image)
                if path.name == "broken.png":
                    raise OSError("broken data stream when reading image file")
                calibrated_score = (
                    0.75 if path.parent.name == "structural_anomalies" else 0.25
                )
                return PatchEvidence(
                    layer_scores={"6": 0.2, "12": 0.2, "18": 0.2, "24": 0.2},
                    aggregate_raw_score=0.2,
                    threshold=0.3,
                    calibrated_score=calibrated_score,
                    anomaly=calibrated_score > 0.5,
                )

        bank = SimpleNamespace(
            threshold=0.3,
            score_mean=0.2,
            score_unbiased_std=0.1,
            calibration_mode="normal_validation_zscore_sigmoid",
            calibration_scores=(0.1, 0.2),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset" / "good").mkdir(parents=True)
            (root / "dataset" / "good" / "sample.png").write_bytes(b"unused")
            (root / "dataset" / "structural_anomalies").mkdir(parents=True)
            (root / "dataset" / "structural_anomalies" / "sample.png").write_bytes(
                b"unused"
            )
            (root / "dataset" / "structural_anomalies" / "broken.png").write_bytes(
                b"unused"
            )
            with (
                patch(
                    "pro_innovation.evaluation.PatchMemoryBank.load",
                    return_value=bank,
                ),
                patch(
                    "pro_innovation.evaluation.DinoV2PatchFeatureExtractor",
                    return_value=object(),
                ),
                patch(
                    "pro_innovation.evaluation.DinoV2PatchImageScorer",
                    return_value=FixedScorer(),
                ),
                patch(
                    "pro_innovation.evaluation.Sam3Segmenter",
                    side_effect=AssertionError("SAM3 must stay unloaded"),
                ),
                patch(
                    "pro_innovation.evaluation.LocalClipRuntime",
                    side_effect=AssertionError("CLIP must stay unloaded"),
                ),
            ):
                summary = evaluate_dataset(
                    class_name="demo",
                    dataset_root=root / "dataset",
                    generic_path=root / "missing.generic.rules",
                    standard_path=root / "missing.standard.rules",
                    result_root=root / "result",
                    sam3_threshold=0.7,
                    sam3_repo="/missing/sam3",
                    sam3_checkpoint="/missing/sam3.pt",
                    clip_repo="/missing/clip",
                    clip_checkpoint="/missing/clip.pt",
                    rule_inference_enabled=False,
                    patch_enabled=True,
                    patch_bank_path=root / "patch.npz",
                    test_splits=("good", "structural_anomalies"),
                    restart=True,
                )

        self.assertEqual(summary["execution_mode"], "patch_only")
        self.assertFalse(summary["rule_inference_enabled"])
        self.assertEqual(summary["score_fusion_policy"], "patch_only_no_fusion")
        self.assertEqual(summary["sam3_threshold_policy"], "disabled")
        self.assertIsNone(summary["mask_cleanup_enabled"])
        self.assertIsNone(summary["standard_template"])
        self.assertIsNone(summary["generic_template"])
        self.assertEqual(summary["valid_score_count"], 2)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(summary["auroc_excluded_failure_count"], 1)
        self.assertEqual(summary["image_auroc"], 1.0)
        self.assertEqual(summary["structural_image_auroc"], 1.0)
        self.assertIsNone(summary["logical_image_auroc"])

    def test_mvtec_ad_discovers_all_defect_directories(self) -> None:
        class FixedScorer:
            def score(self, image):
                calibrated_score = 0.25 if Path(image).parent.name == "good" else 0.75
                return PatchEvidence(
                    layer_scores={"6": 0.2, "12": 0.2, "18": 0.2, "24": 0.2},
                    aggregate_raw_score=0.2,
                    threshold=0.3,
                    calibrated_score=calibrated_score,
                    anomaly=calibrated_score > 0.5,
                )

        bank = SimpleNamespace(
            threshold=0.3,
            score_mean=0.2,
            score_unbiased_std=0.1,
            calibration_mode="normal_validation_zscore_sigmoid",
            calibration_scores=(0.1, 0.2),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "bottle" / "test"
            for split in ("good", "broken_large", "contamination"):
                (dataset / split).mkdir(parents=True)
                (dataset / split / "sample.png").write_bytes(b"unused")
            with (
                patch(
                    "pro_innovation.evaluation.PatchMemoryBank.load",
                    return_value=bank,
                ),
                patch(
                    "pro_innovation.evaluation.DinoV2PatchFeatureExtractor",
                    return_value=object(),
                ),
                patch(
                    "pro_innovation.evaluation.DinoV2PatchImageScorer",
                    return_value=FixedScorer(),
                ),
            ):
                summary = evaluate_dataset(
                    class_name="bottle",
                    dataset_name=MVTEC_AD,
                    dataset_root=dataset,
                    generic_path=None,
                    standard_path=None,
                    result_root=root / "result",
                    sam3_threshold=0.4,
                    sam3_repo="/missing/sam3",
                    sam3_checkpoint="/missing/sam3.pt",
                    clip_repo="/missing/clip",
                    clip_checkpoint="/missing/clip.pt",
                    rule_inference_enabled=False,
                    patch_enabled=True,
                    patch_bank_path=root / "patch.npz",
                    restart=True,
                )

        self.assertEqual(summary["dataset_name"], MVTEC_AD)
        self.assertEqual(summary["dataset"], "MVTec AD bottle/test")
        self.assertEqual(summary["scope"], "good+broken_large+contamination")
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["negative_count"], 1)
        self.assertEqual(summary["positive_count"], 2)
        self.assertEqual(summary["image_auroc"], 1.0)
        self.assertEqual(
            summary["split_image_aurocs"],
            {"broken_large": 1.0, "contamination": 1.0},
        )

    def test_run_name_is_derived_from_parameters_and_template(self) -> None:
        rules = parse_template(
            "[C004] TASK=LENGTH | SUBJECT=screw | PROPERTY=length | "
            "OP=RANGE | VALUE=[97,103] | COUNT=[1]"
        )
        tolerance = infer_length_tolerance_percent(rules)
        self.assertAlmostEqual(tolerance or 0.0, 3.0)
        self.assertEqual(
            build_run_stem("screw_bag", 0.70, tolerance),
            "screw_bag_rules_good-logical",
        )

    def test_auroc_handles_tied_scores(self) -> None:
        self.assertEqual(binary_auroc([0, 1], [0.5, 0.5]), 0.5)

    def test_run_name_records_disabled_mask_operations(self) -> None:
        self.assertEqual(
            build_run_stem(
                "screw_bag",
                0.70,
                5.0,
                mask_cleanup=False,
                mask_dedup=False,
            ),
            "screw_bag_rules_good-logical",
        )

    def test_patch_run_name_records_fixed_logsad_configuration(self) -> None:
        self.assertEqual(
            build_run_stem(
                "screw_bag",
                0.70,
                5.0,
                patch_enabled=True,
            ),
            "screw_bag_dino_b6b12b18b24_good-logical",
        )

    def test_patch_only_run_name_is_isolated(self) -> None:
        self.assertEqual(
            build_run_stem(
                "screw_bag",
                0.70,
                None,
                rule_inference_enabled=False,
                patch_enabled=True,
            ),
            "screw_bag_dino_b6b12b18b24_patchonly_good-logical",
        )

    def test_full_test_scope_gets_distinct_run_name(self) -> None:
        self.assertEqual(
            build_run_stem(
                "screw_bag",
                0.70,
                5.0,
                include_structural_anomalies=True,
            ),
            "screw_bag_rules_good-logical-structural",
        )

    def test_area_template_gets_class_specific_measurement_tag(self) -> None:
        rules = parse_template(
            "[C009] TASK=AREA | SUBJECT=liquid | PROPERTY=area | "
            "OP=RANGE | VALUE=[210843,233037] | COUNT=[1]"
        )
        measurement_tag = infer_measurement_tag(rules)
        self.assertEqual(measurement_tag, "area5")
        self.assertEqual(
            build_run_stem(
                "juice_bottle",
                0.70,
                None,
                measurement_tag=measurement_tag,
            ),
            "juice_bottle_rules_good-logical",
        )

    def test_semantic_attribute_vocabulary_is_checked_before_evaluation(self) -> None:
        rules = parse_template(
            "[C010] TASK=ATTRIBUTE_COMBINATION | "
            "SUBJECT_1=fruit_icon | PROPERTY_1=attribute.type | "
            "SUBJECT_2=liquid | PROPERTY_2=attribute.type | OP=EQ | "
            'VALUE=["cherry:cherry","banana:banana","orange:orange"]'
        )
        with self.assertRaisesRegex(ValueError, "fruit_icon_type"):
            validate_attribute_vocabulary(rules, {})
        validate_attribute_vocabulary(
            rules,
            {
                "fruit_icon_type": ["orange", "banana", "cherry"],
                "liquid_type": ["orange", "banana", "cherry"],
            },
        )

    def test_color_attribute_uses_only_configured_palette_entries(self) -> None:
        rules = parse_template(
            "[C010] TASK=ATTRIBUTE_COMBINATION | "
            "SUBJECT_1=fruit_icon | PROPERTY_1=attribute.type | "
            "SUBJECT_2=liquid | PROPERTY_2=attribute.color | OP=EQ | "
            'VALUE=["cherry:red","banana:light_yellow",'
            '"orange:dark_yellow"]'
        )
        with self.assertRaisesRegex(ValueError, "liquid_color"):
            validate_attribute_vocabulary(
                rules,
                {"fruit_icon_type": ["orange", "banana", "cherry"]},
            )
        validate_attribute_vocabulary(
            rules,
            {
                "fruit_icon_type": ["orange", "banana", "cherry"],
                "liquid_color": ["red", "light_yellow", "dark_yellow"],
            },
        )

        with self.assertRaisesRegex(
            ValueError, "standard template values missing"
        ):
            validate_attribute_vocabulary(
                rules,
                {
                    "fruit_icon_type": ["orange", "banana", "cherry"],
                    "liquid_color": ["red", "yellow", "light_yellow"],
                },
            )

    def test_attribute_count_reuses_masks_and_needs_no_vocabulary(self) -> None:
        rules = parse_template(
            "[C004] TASK=ATTRIBUTE_COMBINATION | "
            "SUBJECT_1=orange_clip | PROPERTY_1=attribute.count | "
            "SUBJECT_2=wire | PROPERTY_2=attribute.color | OP=EQ | "
            'VALUE=["10:red","4:yellow","6:blue"]'
        )

        validate_attribute_vocabulary(
            rules,
            {"wire_color": ["red", "yellow", "blue"]},
        )

    def test_attribute_length_and_area_need_no_vocabulary(self) -> None:
        rules = parse_template(
            "[C001] TASK=ATTRIBUTE_COMBINATION | "
            "SUBJECT_1=part | PROPERTY_1=attribute.area | "
            "SUBJECT_2=part | PROPERTY_2=attribute.length | OP=EQ | "
            'VALUE=["9:3"]'
        )

        validate_attribute_vocabulary(rules, {})

    def test_attribute_spatial_combination_needs_no_vocabulary(self) -> None:
        rules = parse_template(
            "[S001] TASK=SPATIAL_COMBINATION | SUBJECT=part_a | "
            "PROPERTY=spatial.option | OBJECT=part_b | OP=EQ | VALUE=left_of\n"
            "[C001] TASK=ATTRIBUTE_COMBINATION | "
            "SUBJECT_1=part_a | "
            "PROPERTY_1=attribute.spatial_combination.part_b | "
            "SUBJECT_2=part_a | PROPERTY_2=attribute.color | OP=EQ | "
            'VALUE=["left_of:red"]'
        )

        validate_attribute_vocabulary(
            rules,
            {"part_a_color": ["red", "blue"]},
        )

    def test_generic_evaluate_command_has_class_driven_defaults(self) -> None:
        args = build_parser().parse_args(
            [
                "evaluate",
                "--class-name",
                "juice_bottle",
                "--attribute-vocabulary",
                "config/attribute_vocabulary.example.json",
            ]
        )
        self.assertEqual(args.class_name, "juice_bottle")
        self.assertIsNone(args.dataset_root)
        self.assertIsNone(args.generic)
        self.assertIsNone(args.standard)
        self.assertEqual(args.mask_dedup_iou, 0.80)
        self.assertEqual(args.mask_dedup_containment, 0.85)
        self.assertEqual(args.mask_min_area, 16)
        self.assertEqual(args.sam3_threshold, 0.40)
        self.assertFalse(hasattr(args, "clip_verification"))
        self.assertEqual(args.clip_threshold, 0.35)
        self.assertFalse(hasattr(args, "memory"))
        self.assertFalse(hasattr(args, "memory_bank"))
        self.assertFalse(hasattr(args, "dinov2_checkpoint"))
        self.assertFalse(args.patch_matching)
        self.assertFalse(hasattr(args, "patch_layer_fusion"))
        self.assertFalse(hasattr(args, "patch_primary_layer"))
        self.assertTrue(args.rule_inference)

    def test_generic_evaluate_supports_mvtec_ad_layout(self) -> None:
        args = build_parser().parse_args(
            ["evaluate", "--dataset", "mvtec_ad", "--class-name", "bottle"]
        )
        self.assertEqual(args.dataset, MVTEC_AD)
        self.assertIsNone(args.dataset_base)
        self.assertIsNone(args.dataset_root)

    def test_generic_evaluate_supports_visa_layout(self) -> None:
        args = build_parser().parse_args(
            ["evaluate", "--dataset", "visa", "--class-name", "candle"]
        )
        self.assertEqual(args.dataset, VISA)
        self.assertIsNone(args.dataset_base)
        self.assertIsNone(args.dataset_root)

    def test_default_patch_bank_path_is_grouped_by_shot_and_dataset(self) -> None:
        parser = build_parser()
        cases = (
            ("mvtec_loco", "screw_bag"),
            ("mvtec_ad", "bottle"),
            ("visa", "candle"),
        )
        for dataset, class_name in cases:
            with self.subTest(dataset=dataset):
                args = parser.parse_args(
                    ["evaluate", "--dataset", dataset, "--class-name", class_name]
                )
                expected = (
                    Path("/home/lxq/pro-innovation/memory_banks/4-shot")
                    / dataset
                    / f"{class_name}.dino_b6b12b18b24.npz"
                )
                self.assertEqual(_patch_bank_path(args, class_name), expected)

    def test_visa_uses_only_official_test_rows_and_jpg_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "split_csv").mkdir()
            rows = (
                "object,split,label,image,mask\n"
                "candle,train,normal,candle/Data/Images/Normal/train.JPG,\n"
                "candle,test,normal,candle/Data/Images/Normal/test.JPG,\n"
                "candle,test,anomaly,candle/Data/Images/Anomaly/test.JPG,mask.png\n"
                "pcb1,test,normal,pcb1/Data/Images/Normal/test.JPG,\n"
            )
            (root / "split_csv" / "1cls.csv").write_text(rows, encoding="utf-8")
            for relative in (
                "candle/Data/Images/Normal/train.JPG",
                "candle/Data/Images/Normal/test.JPG",
                "candle/Data/Images/Anomaly/test.JPG",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"unused")

            specs = _split_specs_for_dataset(VISA, root)
            samples = _collect_evaluation_samples(
                dataset_name=VISA,
                dataset_root=root,
                class_name="candle",
                selected_splits=tuple(specs),
                split_specs=specs,
            )

        self.assertEqual(len(samples), 2)
        self.assertEqual([sample[1] for sample in samples], [1, 0])
        self.assertEqual([sample[2] for sample in samples], ["Anomaly", "Normal"])
        self.assertTrue(all(sample[0].suffix == ".JPG" for sample in samples))

    def test_rule_inference_is_default_on_and_switchable(self) -> None:
        parser = build_parser()
        default = parser.parse_args(["evaluate", "--class-name", "screw_bag"])
        patch_only = parser.parse_args(
            [
                "evaluate",
                "--class-name",
                "screw_bag",
                "--no-rule-inference",
                "--patch-matching",
            ]
        )

        self.assertTrue(default.rule_inference)
        self.assertFalse(patch_only.rule_inference)
        _validate_runtime_branches(patch_only)

    def test_disabled_rules_require_patch(self) -> None:
        parser = build_parser()
        no_branch = parser.parse_args(
            ["evaluate", "--class-name", "screw_bag", "--no-rule-inference"]
        )
        with self.assertRaisesRegex(ValueError, "requires --patch-matching"):
            _validate_runtime_branches(no_branch)

    def test_patch_matching_is_default_off_and_switchable(self) -> None:
        parser = build_parser()
        disabled = parser.parse_args(
            ["evaluate", "--class-name", "screw_bag"]
        )
        enabled = parser.parse_args(
            [
                "evaluate",
                "--class-name",
                "screw_bag",
                "--patch-matching",
            ]
        )
        self.assertFalse(disabled.patch_matching)
        self.assertTrue(enabled.patch_matching)
        self.assertEqual(enabled.patch_query_batch_size, 256)
        self.assertFalse(hasattr(disabled, "patch_primary_layer"))
        self.assertFalse(hasattr(enabled, "patch_layer_fusion"))

    def test_disabled_patch_module_does_not_touch_a_bank_or_model(self) -> None:
        self.assertIsNone(
            _make_patch_scorer(Namespace(patch_matching=False), class_name="demo")
        )

    def test_build_patch_memory_keeps_all_layers_and_uses_full_images(self) -> None:
        args = build_parser().parse_args(
            [
                "build-patch-memory",
                "--class-name",
                "screw_bag",
                "--normal-images",
                "a.png",
                "b.png",
                "c.png",
                "d.png",
                "--calibration-dir",
                "validation/good",
            ]
        )
        self.assertEqual(len(args.normal_images), 4)
        self.assertEqual(args.calibration_dir, "validation/good")
        self.assertEqual(args.patch_query_batch_size, 256)
        self.assertEqual(args.patch_extraction_batch_size, 4)
        self.assertFalse(hasattr(args, "patch_threshold_margin"))
        self.assertFalse(hasattr(args, "patch_position_radius"))
        self.assertFalse(hasattr(args, "patch_fusion_quantile"))
        self.assertFalse(hasattr(args, "dataset_base"))
        self.assertFalse(hasattr(args, "patch_crop_padding_fraction"))
        self.assertFalse(hasattr(args, "sam3_threshold"))
        self.assertFalse(hasattr(args, "object_sam3_thresholds"))

    def test_generic_evaluate_can_keep_all_splits(self) -> None:
        args = build_parser().parse_args(
            [
                "evaluate",
                "--class-name",
                "screw_bag",
                "--all-test-splits",
            ]
        )
        self.assertTrue(args.all_test_splits)

    def test_generic_evaluate_all_splits_flag_is_on_by_default(self) -> None:
        args = build_parser().parse_args(
            ["evaluate", "--class-name", "screw_bag"]
        )
        self.assertTrue(args.all_test_splits)

    def test_calibrate_sam3_command_fixes_proposal_threshold_at_0_10(self) -> None:
        args = build_parser().parse_args(
            [
                "calibrate-sam3-thresholds",
                "--class-name",
                "juice_bottle",
                "--normal-images",
                "a.png",
                "b.png",
                "c.png",
                "d.png",
            ]
        )
        self.assertEqual(len(args.normal_images), 4)
        self.assertEqual(args.sam3_threshold, 0.10)
        self.assertIsNone(args.truth_templates)

    def test_object_sam3_threshold_policy_can_be_disabled_for_ablation(self) -> None:
        args = build_parser().parse_args(
            [
                "evaluate",
                "--class-name",
                "juice_bottle",
                "--no-object-sam3-thresholds",
            ]
        )
        self.assertFalse(args.object_sam3_thresholds)

if __name__ == "__main__":
    unittest.main()
