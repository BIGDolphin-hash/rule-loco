from __future__ import annotations

import math
import threading
import unittest

import numpy as np
import torch

from pro_innovation.adapters.patch_dinov2 import (
    PATCH_MATCHING_LAYERS,
    PatchFeatureBundle,
)
from pro_innovation.memory_modules.gcda import (
    CategoryOnlyCompositionMemoryBank,
    CategoryOnlyCompositionScorer,
)
from pro_innovation.memory_modules.gpojm import (
    GlobalPatchOrderedMemoryBank,
    GlobalPatchOrderedScorer,
)
from pro_innovation.memory_modules.gpstm import (
    GlobalPatchSemanticTextureMemoryBank,
    GlobalPatchSemanticTextureScorer,
)
from pro_innovation.models import (
    AttributePrediction,
    CompositionEvidence,
    MaskInstance,
    PatchEvidence,
)
from pro_innovation.pipeline import (
    LogicAnomalyPipeline,
    RULE_MEMORY_SCHEDULE_BINARY_PARALLEL,
    RULE_MEMORY_SCHEDULE_BINARY_SELECTIVE,
)
from pro_innovation.templates import parse_template, to_generic_template


def _mask(left: bool) -> MaskInstance:
    values = np.zeros((4, 4), dtype=bool)
    values[:, :2] = left
    values[:, 2:] = not left
    return MaskInstance(values)


def _bundle(left_angle: float, right_angle: float) -> PatchFeatureBundle:
    vectors = torch.tensor(
        [
            [math.cos(left_angle), math.sin(left_angle)],
            [math.cos(left_angle), math.sin(left_angle)],
            [math.cos(right_angle), math.sin(right_angle)],
            [math.cos(right_angle), math.sin(right_angle)],
        ],
        dtype=torch.float32,
    )
    return PatchFeatureBundle(
        layer_features={layer: vectors.clone() for layer in PATCH_MATCHING_LAYERS},
        feature_grid=(2, 2),
    )


def _pool():
    return {"left": (_mask(True),), "right": (_mask(False),)}


class _Segmenter:
    def segment(self, image, categories):
        del image
        return {category: _pool()[category] for category in categories}


class _Attributes:
    def detect(self, image, subject, property_name, instances):
        raise AssertionError("attribute inference is not used")


class _SharedPatchScorer:
    def __init__(
        self,
        features: PatchFeatureBundle,
        *,
        fusion_score: float | None = None,
    ) -> None:
        self.features = features
        self.fusion_score = fusion_score
        self.extract_calls = 0
        self.score_calls = 0

    def extract_features(self, image):
        del image
        self.extract_calls += 1
        return self.features

    def score_features(self, features):
        assert features is self.features
        self.score_calls += 1
        return PatchEvidence(
            layer_scores={str(layer): 0.1 for layer in PATCH_MATCHING_LAYERS},
            aggregate_raw_score=0.1,
            threshold=0.2,
            calibrated_score=0.2,
            anomaly=False,
            fusion_score=self.fusion_score,
        )

    def score_features_with_map(self, features):
        evidence = self.score_features(features)
        return evidence, torch.tensor([0.1, 0.2, 0.3, 0.4])

    def score(self, image):
        return self.score_features(self.extract_features(image))


class _CompositionScorer:
    def __init__(self, anomaly: bool, adaptive_score: float = 1.8) -> None:
        self.anomaly = anomaly
        self.adaptive_score = adaptive_score
        self.calls = 0

    def score_features(self, features, mask_pool):
        del features, mask_pool
        self.calls += 1
        return CompositionEvidence(
            category_distances={"left": 0.8, "right": 0.1},
            aggregate_raw_score=0.9,
            threshold=0.5,
            adaptive_score=self.adaptive_score,
            calibrated_score=self.adaptive_score / (self.adaptive_score + 1.0),
            anomaly=self.anomaly,
            nearest_reference_index=0,
        )


class MemoryRoutingTests(unittest.TestCase):

    def _remaining_scheme(self, scheme):
        global_scorer = _SharedPatchScorer(_bundle(0.1, 0.2), fusion_score=0.25)
        common = dict(
            class_name="demo",
            categories=("left", "right"),
            normal_images=("/tmp/a.png", "/tmp/b.png", "/tmp/c.png"),
            checkpoint_path="/tmp/dino.pth",
        )
        texture = torch.tensor(
            [
                [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.1, 0.1, 0.1]],
                [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.2, 0.2, 0.2]],
                [[0.5, 0.6, 0.7], [0.4, 0.5, 0.6], [0.3, 0.4, 0.5]],
            ]
        )
        if scheme == "Global Patch":
            return global_scorer, dict(patch_scorer=global_scorer)
        if scheme == "GCDA":
            bank = CategoryOnlyCompositionMemoryBank.build(
                **common,
                features=(_bundle(0.0, 0.1), _bundle(0.1, 0.3), _bundle(0.4, 0.5)),
                mask_pools=(_pool(), _pool(), _pool()),
            )
            return global_scorer, dict(
                patch_scorer=global_scorer,
                composition_scorer=CategoryOnlyCompositionScorer(bank),
            )
        if scheme == "GPSTM":
            bank = GlobalPatchSemanticTextureMemoryBank.build_from_signatures(
                **common,
                texture_features=texture,
                global_patch_bank_path="/tmp/global.npz",
                feature_grid=(2, 2),
            )
            scorer = GlobalPatchSemanticTextureScorer(
                bank=bank,
                global_patch_scorer=global_scorer,
            )
        else:
            bank = GlobalPatchOrderedMemoryBank.build_from_components(
                **common,
                category_features=torch.tensor(
                    [
                        [[0.0, 0.0], [0.0, 1.0]],
                        [[1.0, 0.0], [1.0, 1.0]],
                        [[2.0, 1.0], [1.0, 2.0]],
                    ]
                ),
                texture_features=texture,
                global_patch_bank_path="/tmp/global.npz",
                feature_grid=(2, 2),
                texture_weight=0.3,
            )
            scorer = GlobalPatchOrderedScorer(
                bank=bank,
                global_patch_scorer=global_scorer,
            )
        return global_scorer, dict(
            composition_scorer=scorer,
            shared_feature_extractor=scorer,
        )

    def test_all_four_remaining_schemes_preserve_binary_routing(self):
        for scheme in ("Global Patch", "GCDA", "GPSTM", "GPO-JM"):
            for schedule in ("binary-selective", "binary-parallel"):
                for failed in (False, True):
                    with self.subTest(scheme=scheme, schedule=schedule, failed=failed):
                        global_scorer, components = self._remaining_scheme(scheme)
                        standard = parse_template(
                            "[C001] TASK=COUNT | SUBJECT=left | PROPERTY=count | "
                            f"OP=EQ | VALUE=[{2 if failed else 1}]\n"
                            "[C002] TASK=COUNT | SUBJECT=right | PROPERTY=count | "
                            "OP=EQ | VALUE=[1]"
                        )
                        pipeline = LogicAnomalyPipeline(
                            _Segmenter(),
                            _Attributes(),
                            rule_memory_schedule=schedule,
                            **components,
                        )
                        report = pipeline.run(
                            np.zeros((4, 4, 3), dtype=np.uint8),
                            to_generic_template(standard),
                            standard,
                        )
                        if failed:
                            self.assertEqual(report.image_anomaly_score, 1.0)
                            self.assertEqual(len(report.results), 1)
                        if failed and schedule == "binary-selective":
                            self.assertEqual(global_scorer.extract_calls, 0)
                            self.assertIsNone(report.patch_evidence)
                            self.assertIsNone(report.composition_evidence)
                        else:
                            self.assertEqual(global_scorer.extract_calls, 1)
                            if scheme == "Global Patch":
                                memory_score = 0.2
                            else:
                                ratio = report.composition_evidence.adaptive_score
                                if scheme == "GCDA":
                                    ratio += report.patch_evidence.fusion_score
                                memory_score = ratio / (ratio + 1.0)
                            self.assertAlmostEqual(
                                report.image_anomaly_score,
                                max(float(failed), memory_score),
                            )

    def test_active_pipeline_accepts_composition_runtime(self) -> None:
        pipeline = LogicAnomalyPipeline(
            _Segmenter(),
            _Attributes(),
            patch_scorer=_SharedPatchScorer(_bundle(0.0, 0.1)),
            composition_scorer=_CompositionScorer(anomaly=True),
        )

        self.assertIsNotNone(pipeline.composition_scorer)

    def test_rule_pass_shares_one_dino_forward_between_post_paths(self) -> None:
        standard = parse_template(
            "[C001] TASK=COUNT | SUBJECT=left | PROPERTY=count | OP=EQ | VALUE=[1]"
        )
        patch = _SharedPatchScorer(_bundle(0.0, 0.1))
        composition = _CompositionScorer(anomaly=True)
        pipeline = LogicAnomalyPipeline(
            _Segmenter(),
            _Attributes(),
            patch_scorer=patch,
            composition_scorer=composition,
        )

        report = pipeline.run(
            np.zeros((4, 4, 3), dtype=np.uint8),
            to_generic_template(standard),
            standard,
        )

        self.assertEqual(patch.extract_calls, 1)
        self.assertEqual(patch.score_calls, 1)
        self.assertEqual(composition.calls, 1)
        self.assertEqual(report.execution_mode, "rule_binary_dual_post")
        self.assertAlmostEqual(report.image_anomaly_score, 1.8 / 2.8)
        payload = report.to_dict()
        self.assertNotIn("status", payload)
        self.assertNotIn("anomaly", payload)
        self.assertNotIn("logical_anomaly", payload)
        self.assertNotIn("anomaly", payload["patch_evidence"])
        self.assertNotIn("anomaly", payload["composition_evidence"])
        self.assertNotIn("satisfied", payload["results"][0])
        timings = payload["timings_seconds"]
        self.assertGreater(timings["memory_total"], 0.0)
        self.assertGreater(timings["memory_feature_extraction"], 0.0)
        self.assertGreater(timings["memory_composition"], 0.0)
        self.assertGreater(timings["memory_global_patch"], 0.0)

    def test_selective_rule_failure_skips_both_post_paths(self) -> None:
        standard = parse_template(
            "[C001] TASK=COUNT | SUBJECT=left | PROPERTY=count | OP=EQ | VALUE=[2]"
        )
        patch = _SharedPatchScorer(_bundle(0.0, 0.1))
        composition = _CompositionScorer(anomaly=True)
        pipeline = LogicAnomalyPipeline(
            _Segmenter(),
            _Attributes(),
            patch_scorer=patch,
            composition_scorer=composition,
        )

        report = pipeline.run(
            np.zeros((4, 4, 3), dtype=np.uint8),
            to_generic_template(standard),
            standard,
        )

        self.assertEqual(report.status, "RULE_ANOMALY")
        self.assertEqual(report.execution_mode, "rule_binary_early_exit")
        self.assertEqual(patch.extract_calls, 0)
        self.assertEqual(patch.score_calls, 0)
        self.assertEqual(composition.calls, 0)
        self.assertIsNone(report.patch_evidence)
        self.assertIsNone(report.composition_evidence)
        self.assertAlmostEqual(
            report.image_anomaly_score,
            1.0,
        )
        payload = report.to_dict()
        self.assertIsNone(payload["patch_evidence"])
        self.assertIsNone(payload["composition_evidence"])
        self.assertEqual(payload["timings_seconds"]["memory_total"], 0.0)

    def test_binary_selective_stops_at_first_failed_rule_with_score_one(self) -> None:
        standard = parse_template(
            "[C001] TASK=COUNT | SUBJECT=left | PROPERTY=count | OP=EQ | VALUE=[2]\n"
            "[C002] TASK=COUNT | SUBJECT=right | PROPERTY=count | OP=EQ | VALUE=[1]"
        )
        patch = _SharedPatchScorer(_bundle(0.0, 0.1))
        composition = _CompositionScorer(anomaly=True)
        pipeline = LogicAnomalyPipeline(
            _Segmenter(),
            _Attributes(),
            patch_scorer=patch,
            composition_scorer=composition,
            rule_memory_schedule=RULE_MEMORY_SCHEDULE_BINARY_SELECTIVE,
        )

        report = pipeline.run(
            np.zeros((4, 4, 3), dtype=np.uint8),
            to_generic_template(standard),
            standard,
        )

        self.assertEqual(report.status, "RULE_ANOMALY")
        self.assertEqual(report.execution_mode, "rule_binary_early_exit")
        self.assertEqual(report.image_anomaly_score, 1.0)
        self.assertEqual([result.rule_id for result in report.results], ["C001"])
        self.assertEqual(patch.extract_calls, 0)
        self.assertEqual(composition.calls, 0)

    def test_binary_selective_all_rules_pass_with_rule_score_zero(self) -> None:
        standard = parse_template(
            "[C001] TASK=COUNT | SUBJECT=left | PROPERTY=count | OP=EQ | VALUE=[1]\n"
            "[C002] TASK=COUNT | SUBJECT=right | PROPERTY=count | OP=EQ | VALUE=[1]"
        )
        pipeline = LogicAnomalyPipeline(
            _Segmenter(),
            _Attributes(),
            rule_memory_schedule=RULE_MEMORY_SCHEDULE_BINARY_SELECTIVE,
        )

        report = pipeline.run(
            np.zeros((4, 4, 3), dtype=np.uint8),
            to_generic_template(standard),
            standard,
        )

        self.assertEqual(report.status, "NORMAL")
        self.assertEqual(report.execution_mode, "rule_binary_only")
        self.assertEqual(report.image_anomaly_score, 0.0)
        self.assertEqual(
            [result.rule_id for result in report.results], ["C001", "C002"]
        )

    def test_binary_parallel_fuses_one_for_any_failed_rule(self) -> None:
        standard = parse_template(
            "[C001] TASK=COUNT | SUBJECT=left | PROPERTY=count | OP=EQ | VALUE=[2]\n"
            "[C002] TASK=COUNT | SUBJECT=right | PROPERTY=count | OP=EQ | VALUE=[1]"
        )
        patch = _SharedPatchScorer(_bundle(0.0, 0.1))
        composition = _CompositionScorer(anomaly=False, adaptive_score=0.25)
        pipeline = LogicAnomalyPipeline(
            _Segmenter(),
            _Attributes(),
            patch_scorer=patch,
            composition_scorer=composition,
            rule_memory_schedule=RULE_MEMORY_SCHEDULE_BINARY_PARALLEL,
        )

        report = pipeline.run(
            np.zeros((4, 4, 3), dtype=np.uint8),
            to_generic_template(standard),
            standard,
        )

        self.assertEqual(report.execution_mode, "rule_binary_memory_parallel")
        self.assertEqual(report.image_anomaly_score, 1.0)
        self.assertEqual([result.rule_id for result in report.results], ["C001"])
        self.assertEqual(patch.extract_calls, 1)
        self.assertEqual(composition.calls, 1)

    def test_binary_parallel_uses_gpojm_score_when_all_rules_pass(self) -> None:
        standard = parse_template(
            "[C001] TASK=COUNT | SUBJECT=left | PROPERTY=count | OP=EQ | VALUE=[1]\n"
            "[C002] TASK=COUNT | SUBJECT=right | PROPERTY=count | OP=EQ | VALUE=[1]"
        )
        patch = _SharedPatchScorer(_bundle(0.0, 0.1))
        composition = _CompositionScorer(anomaly=False, adaptive_score=0.25)
        pipeline = LogicAnomalyPipeline(
            _Segmenter(),
            _Attributes(),
            patch_scorer=patch,
            composition_scorer=composition,
            rule_memory_schedule=RULE_MEMORY_SCHEDULE_BINARY_PARALLEL,
        )

        report = pipeline.run(
            np.zeros((4, 4, 3), dtype=np.uint8),
            to_generic_template(standard),
            standard,
        )

        self.assertEqual(report.execution_mode, "rule_binary_memory_parallel")
        self.assertAlmostEqual(report.image_anomaly_score, 0.2)
        self.assertEqual(
            [result.rule_id for result in report.results], ["C001", "C002"]
        )
        self.assertEqual(patch.extract_calls, 1)
        self.assertEqual(composition.calls, 1)

    def test_binary_parallel_schedule_starts_rule_and_memory_concurrently(self) -> None:
        barrier = threading.Barrier(2)

        class BarrierAttributes:
            def detect(self, image, subject, property_name, instances):
                del image, subject, property_name, instances
                barrier.wait(timeout=2.0)
                return AttributePrediction("red", 1.0)

        class BarrierPatch(_SharedPatchScorer):
            def extract_features(self, image):
                barrier.wait(timeout=2.0)
                return super().extract_features(image)

        standard = parse_template(
            "[C001] TASK=ATTRIBUTE_COMBINATION | SUBJECT_1=left | "
            "PROPERTY_1=attribute.type | VALUE_1=red | OP=EQ | VALUE=red"
        )
        patch = BarrierPatch(_bundle(0.0, 0.1))
        composition = _CompositionScorer(anomaly=False)
        pipeline = LogicAnomalyPipeline(
            _Segmenter(),
            BarrierAttributes(),
            patch_scorer=patch,
            composition_scorer=composition,
            rule_memory_schedule=RULE_MEMORY_SCHEDULE_BINARY_PARALLEL,
        )

        report = pipeline.run(
            np.zeros((4, 4, 3), dtype=np.uint8),
            to_generic_template(standard),
            standard,
        )

        self.assertEqual(report.execution_mode, "rule_binary_memory_parallel")
        self.assertEqual(patch.extract_calls, 1)
        self.assertAlmostEqual(report.image_anomaly_score, 1.8 / 2.8)

    def test_composition_only_uses_masks_and_dino_without_rules_or_patch(self) -> None:
        standard = parse_template(
            "[C001] TASK=COUNT | SUBJECT=left | PROPERTY=count | OP=EQ | VALUE=[1]\n"
            "[C002] TASK=COUNT | SUBJECT=right | PROPERTY=count | OP=EQ | VALUE=[1]"
        )
        extractor = _SharedPatchScorer(_bundle(0.0, 0.1))
        composition = _CompositionScorer(anomaly=True)
        pipeline = LogicAnomalyPipeline(
            _Segmenter(),
            None,
            composition_scorer=composition,
            shared_feature_extractor=extractor,
            rule_inference_enabled=False,
        )

        report = pipeline.run(
            np.zeros((4, 4, 3), dtype=np.uint8),
            to_generic_template(standard),
            (),
        )

        self.assertEqual(report.execution_mode, "composition_only")
        self.assertEqual(extractor.extract_calls, 1)
        self.assertEqual(extractor.score_calls, 0)
        self.assertEqual(composition.calls, 1)
        self.assertIsNone(report.patch_evidence)
        self.assertAlmostEqual(report.image_anomaly_score, 1.8 / 2.8)

    def test_no_rule_dual_post_scores_global_patch_and_composition(self) -> None:
        standard = parse_template(
            "[C001] TASK=COUNT | SUBJECT=left | PROPERTY=count | OP=EQ | VALUE=[1]"
        )
        patch = _SharedPatchScorer(_bundle(0.0, 0.1), fusion_score=0.4)
        composition = _CompositionScorer(anomaly=True)
        pipeline = LogicAnomalyPipeline(
            _Segmenter(),
            None,
            patch_scorer=patch,
            composition_scorer=composition,
            rule_inference_enabled=False,
        )

        report = pipeline.run(
            np.zeros((4, 4, 3), dtype=np.uint8),
            to_generic_template(standard),
            (),
        )

        self.assertEqual(report.execution_mode, "dual_post")
        self.assertEqual(patch.extract_calls, 1)
        self.assertEqual(patch.score_calls, 1)
        self.assertEqual(composition.calls, 1)
        self.assertIsNotNone(report.patch_evidence)
        self.assertAlmostEqual(report.image_anomaly_score, 2.2 / 3.2)

    def test_post_path_ratios_sum_before_shared_mapping(self) -> None:
        standard = parse_template(
            "[C001] TASK=COUNT | SUBJECT=left | PROPERTY=count | OP=EQ | VALUE=[1]"
        )
        patch = _SharedPatchScorer(_bundle(0.0, 0.1), fusion_score=2.0)
        composition = _CompositionScorer(anomaly=False)
        pipeline = LogicAnomalyPipeline(
            _Segmenter(),
            _Attributes(),
            patch_scorer=patch,
            composition_scorer=composition,
            post_fusion_tau=1.0,
        )

        report = pipeline.run(
            np.zeros((4, 4, 3), dtype=np.uint8),
            to_generic_template(standard),
            standard,
        )

        # r_texture=2, r_composition=1.8, then 3.8/(3.8+1).
        self.assertAlmostEqual(report.image_anomaly_score, 3.8 / 4.8)

    def test_two_subthreshold_branches_can_cross_final_half_boundary(self) -> None:
        standard = parse_template(
            "[C001] TASK=COUNT | SUBJECT=left | PROPERTY=count | OP=EQ | VALUE=[1]"
        )
        patch = _SharedPatchScorer(_bundle(0.0, 0.1), fusion_score=0.6)
        composition = _CompositionScorer(anomaly=False)
        composition.score_features = lambda features, mask_pool: CompositionEvidence(
            category_distances={"left": 0.6},
            aggregate_raw_score=0.6,
            threshold=1.0,
            adaptive_score=0.6,
            calibrated_score=0.375,
            anomaly=False,
            nearest_reference_index=0,
        )
        pipeline = LogicAnomalyPipeline(
            _Segmenter(),
            _Attributes(),
            patch_scorer=patch,
            composition_scorer=composition,
        )

        report = pipeline.run(
            np.zeros((4, 4, 3), dtype=np.uint8),
            to_generic_template(standard),
            standard,
        )

        self.assertAlmostEqual(report.image_anomaly_score, 1.2 / 2.2)
        self.assertTrue(report.image_anomaly)
        self.assertEqual(report.anomaly_source, "mixed")


if __name__ == "__main__":
    unittest.main()
