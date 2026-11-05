from __future__ import annotations

import math
import json
import tempfile
from pathlib import Path
import unittest
from argparse import Namespace

import numpy as np
import torch

from pro_innovation.adapters.patch_dinov2 import (
    DEFAULT_PATCH_DINOV2_CHECKPOINT,
    PATCH_MATCHING_FEATURE_GRID,
    PATCH_MATCHING_INPUT_SIZE,
    PATCH_MATCHING_LAYERS,
    PatchFeatureBundle,
)
from pro_innovation.models import PatchEvidence
from pro_innovation.memory_modules.global_patch import (
    DEFAULT_CORESET_RATIO,
    DEFAULT_PATCH_LAYER_FUSION,
    DinoV2PatchImageScorer,
    PatchMemoryBank,
    PATCH_MEMORY_PER_IMAGE_STORAGE,
    patch_memory_run_tag,
)
from pro_innovation.memory_modules.gpojm import GPOJMGlobalPatchBank
from pro_innovation.pipeline import LogicAnomalyPipeline
from pro_innovation.cli import _resolve_normal_images
from pro_innovation.templates import parse_template, to_generic_template


def _vectors(angles: list[float]) -> torch.Tensor:
    return torch.tensor(
        [[math.cos(angle), math.sin(angle)] for angle in angles],
        dtype=torch.float32,
    )


def bundle(
    angle: float = 0.0,
    *,
    layer_angles: dict[int, float] | None = None,
    layer_position_angles: dict[int, list[float]] | None = None,
) -> PatchFeatureBundle:
    values = {}
    for layer in PATCH_MATCHING_LAYERS:
        angles = (layer_position_angles or {}).get(layer)
        if angles is None:
            layer_angle = (layer_angles or {}).get(layer, angle)
            angles = [layer_angle] * 4
        values[layer] = _vectors(angles)
    return PatchFeatureBundle(layer_features=values, feature_grid=(2, 2))


class EmptySegmenter:
    def segment(self, image, categories):
        del image
        return {category: () for category in categories}


class UnusedAttributeDetector:
    def detect(self, image, subject, property_name, instances):
        raise AssertionError("attribute detector should not be used")


class FixedPatchScorer:
    def __init__(self, calibrated_score: float, anomaly: bool) -> None:
        self.calibrated_score = calibrated_score
        self.anomaly = anomaly

    def score(self, image):
        del image
        return PatchEvidence(
            layer_scores={str(layer): 0.8 for layer in PATCH_MATCHING_LAYERS},
            aggregate_raw_score=0.8,
            threshold=0.2,
            calibrated_score=self.calibrated_score,
            anomaly=self.anomaly,
            bank_path="patch.npz",
            winning_layer=24,
        )


class FixedBatchExtractor:
    def __init__(self) -> None:
        self.checkpoint_path = Path("/tmp/dinov2.pth")
        self.input_size = 448
        self.layers = PATCH_MATCHING_LAYERS
        self.feature_grid = (2, 2)
        self.device = "cpu"
        self.input_count = 0

    def extract_batch(self, images, *, batch_size):
        del batch_size
        self.input_count = len(images)
        return tuple(bundle() for _ in images)


class PatchMemoryTests(unittest.TestCase):

    def test_cli_accepts_one_explicit_normal_image(self) -> None:
        args = Namespace(normal_images=["/tmp/normal-one.png"], normal_dir=None)
        self.assertEqual(
            _resolve_normal_images(args),
            (Path("/tmp/normal-one.png").resolve(),),
        )

    def test_patch_memory_run_tag_records_coreset_percentage(self) -> None:
        self.assertEqual(
            patch_memory_run_tag(0.01),
            "dino_full_coreset01_b6b12b18b24",
        )
        self.assertEqual(
            patch_memory_run_tag(0.05),
            "dino_full_coreset05_b6b12b18b24",
        )
        self.assertEqual(
            patch_memory_run_tag(0.025),
            "dino_full_coreset2p5_b6b12b18b24",
        )

    def test_fixed_logsad_dino_configuration(self) -> None:
        self.assertEqual(PATCH_MATCHING_LAYERS, (6, 12, 18, 24))
        self.assertEqual(PATCH_MATCHING_INPUT_SIZE, 448)
        self.assertEqual(PATCH_MATCHING_FEATURE_GRID, (64, 64))
        self.assertTrue(
            DEFAULT_PATCH_DINOV2_CHECKPOINT.endswith("dinov2_vitl14_reg4_pretrain.pth")
        )

    def build_bank(self) -> PatchMemoryBank:
        normal_angles = [0.0, 0.05, 0.2, 0.5]
        return PatchMemoryBank.build(
            class_name="demo",
            features=[bundle(angle) for angle in normal_angles],
            normal_images=[f"/tmp/normal-{index}.png" for index in range(4)],
            checkpoint_path="/tmp/dinov2.pth",
            device="cpu",
            query_batch_size=2,
        )

    def test_one_shot_full_patch_memory_uses_patch_self_exclusion(self) -> None:
        reference = bundle(
            layer_position_angles={
                layer: [0.0, 0.2, 0.4, 0.6] for layer in PATCH_MATCHING_LAYERS
            }
        )
        bank = PatchMemoryBank.build(
            class_name="demo",
            features=[reference],
            normal_images=["/tmp/normal-one.png"],
            checkpoint_path="/tmp/dinov2.pth",
            device="cpu",
            query_batch_size=2,
            coreset_ratio=1.0,
        )

        self.assertEqual(bank.normal_images, ("/tmp/normal-one.png",))
        self.assertEqual(len(bank.calibration_scores), 1)
        self.assertGreater(bank.score_unbiased_std, 0.0)
        evidence = bank.score(reference, device="cpu", query_batch_size=2)
        self.assertTrue(math.isfinite(evidence.calibrated_score))

        near = bank.score(bundle(0.7), device="cpu", query_batch_size=2)
        far = bank.score(bundle(1.2), device="cpu", query_batch_size=2)
        self.assertLess(near.calibrated_score, far.calibrated_score)
        self.assertLess(far.calibrated_score, 1.0)
        self.assertIsNone(far.full_image_score["standardized_score"])

        with tempfile.TemporaryDirectory() as directory:
            path = bank.save(Path(directory) / "one-shot.npz")
            gpojm_view = GPOJMGlobalPatchBank.load(
                path,
                expected_class_name="demo",
                expected_checkpoint_path="/tmp/dinov2.pth",
            )
            gpojm_evidence = gpojm_view.score(
                bundle(1.2), device="cpu", query_batch_size=2
            )
        self.assertEqual(gpojm_view.normal_images, ("/tmp/normal-one.png",))
        self.assertAlmostEqual(
            gpojm_evidence.calibrated_score,
            far.calibrated_score,
            places=7,
        )

    def test_logsad_uses_mean_layer_map_then_spatial_max(self) -> None:
        bank = self.build_bank()
        evidence = bank.score(
            bundle(layer_position_angles={6: [math.pi, 0.0, 0.0, 0.0]}),
            device="cpu",
            query_batch_size=2,
        )

        self.assertEqual(evidence.layer_fusion, DEFAULT_PATCH_LAYER_FUSION)
        self.assertEqual(evidence.winning_layer, 6)
        self.assertEqual(set(evidence.layer_scores), {"6", "12", "18", "24"})
        expected = 1.0 - math.cos(math.pi - 0.5)
        self.assertAlmostEqual(evidence.layer_scores["6"], expected, places=6)
        self.assertAlmostEqual(evidence.aggregate_raw_score, expected / 4.0, places=6)
        self.assertGreater(evidence.calibrated_score, 0.5)
        self.assertTrue(evidence.anomaly)

    def test_score_with_map_preserves_existing_patch_evidence(self) -> None:
        bank = self.build_bank()
        query = bundle(layer_position_angles={6: [math.pi, 0.0, 0.0, 0.0]})

        evidence = bank.score(query, device="cpu", query_batch_size=2)
        mapped_evidence, anomaly_map = bank.score_with_map(
            query, device="cpu", query_batch_size=2
        )

        self.assertEqual(anomaly_map.shape, (4,))
        self.assertAlmostEqual(
            float(anomaly_map.max().item()),
            mapped_evidence.aggregate_raw_score,
            places=6,
        )
        self.assertEqual(evidence, mapped_evidence)

    def test_global_nearest_neighbor_has_no_position_restriction(self) -> None:
        position_angles = [0.0, math.pi / 2, math.pi, -math.pi / 2]
        memory_bundle = bundle(
            layer_position_angles={
                layer: position_angles for layer in PATCH_MATCHING_LAYERS
            }
        )
        bank = PatchMemoryBank(
            class_name="demo",
            feature_memory={
                layer: torch.cat([memory_bundle.layer_features[layer]] * 4)
                for layer in PATCH_MATCHING_LAYERS
            },
            memory_source_indices=torch.arange(4).repeat_interleave(4),
            feature_grid=(2, 2),
            checkpoint_path="/tmp/dinov2.pth",
            normal_images=[f"/tmp/normal-{index}.png" for index in range(4)],
            score_mean=0.1,
            score_unbiased_std=0.05,
        )
        shifted = [math.pi, -math.pi / 2, 0.0, math.pi / 2]
        query = bundle(
            layer_position_angles={layer: shifted for layer in PATCH_MATCHING_LAYERS}
        )

        layer_scores, raw_score = bank.raw_score(
            query, device="cpu", query_batch_size=2
        )

        self.assertTrue(all(abs(value) < 1e-6 for value in layer_scores.values()))
        self.assertAlmostEqual(raw_score, 0.0, places=6)

    def test_normal_reference_max_is_the_half_probability_threshold(self) -> None:
        normal = bundle()
        normal_images = [f"/tmp/normal-{index}.png" for index in range(4)]
        bank = PatchMemoryBank(
            class_name="demo",
            feature_memory={
                layer: torch.cat([normal.layer_features[layer]] * 4)
                for layer in PATCH_MATCHING_LAYERS
            },
            memory_source_indices=torch.arange(4).repeat_interleave(4),
            feature_grid=(2, 2),
            checkpoint_path="/tmp/dinov2.pth",
            normal_images=normal_images,
            score_mean=0.35,
            score_unbiased_std=0.1,
            calibration_images=normal_images,
            calibration_scores=(0.2, 0.3, 0.4, 0.5),
        )
        evidence = bank.score(
            bundle(layer_position_angles={6: [math.pi, 0.0, 0.0, 0.0]}),
            device="cpu",
            query_batch_size=2,
        )

        self.assertAlmostEqual(evidence.aggregate_raw_score, 0.5, places=6)
        self.assertAlmostEqual(evidence.threshold, 0.5, places=6)
        self.assertAlmostEqual(evidence.calibrated_score, 0.5, places=6)
        self.assertFalse(evidence.anomaly)

    def test_multi_shot_mapping_is_strictly_monotonic_without_sigmoid_saturation(
        self,
    ) -> None:
        bank = self.build_bank()
        layer_scores = {str(layer): 0.0 for layer in PATCH_MATCHING_LAYERS}

        lower = bank._evidence_from_raw(layer_scores, 1.9)
        higher = bank._evidence_from_raw(layer_scores, 2.0)

        self.assertLess(lower.calibrated_score, higher.calibrated_score)
        self.assertLess(higher.calibrated_score, 1.0)
        self.assertIsNotNone(higher.fusion_score)
        fusion_score = float(higher.fusion_score)
        self.assertAlmostEqual(
            higher.calibrated_score,
            fusion_score / (fusion_score + 1.0),
        )
        self.assertEqual(
            higher.full_image_score["score_mapping"],
            "ratio_over_one_plus_ratio",
        )

    def test_build_calibrates_with_unbiased_normal_validation_std(self) -> None:
        bank = self.build_bank()
        angles = [0.0, 0.05, 0.2, 0.5]
        expected_scores = [
            min(1.0 - math.cos(angle - other) for other in angles if other != angle)
            for angle in angles
        ]

        self.assertAlmostEqual(
            bank.score_mean, float(np.mean(expected_scores)), places=6
        )
        self.assertAlmostEqual(
            bank.score_unbiased_std,
            float(np.std(expected_scores, ddof=1)),
            places=6,
        )
        self.assertEqual(len(bank.calibration_scores), 4)
        self.assertEqual(bank.calibration_images, bank.normal_images)

    def test_patch_exposes_psad_max_scaled_fusion_score(self) -> None:
        bank = self.build_bank()
        evidence = bank.score(bundle(math.pi), device="cpu", query_batch_size=2)

        self.assertAlmostEqual(bank.psad_scale, max(bank.calibration_scores))
        self.assertAlmostEqual(
            evidence.fusion_score,
            evidence.aggregate_raw_score / max(bank.calibration_scores),
        )
        self.assertEqual(evidence.fusion_threshold, 1.0)

    def test_full_normal_coreset_keeps_every_image_represented(self) -> None:
        bank = self.build_bank()

        self.assertEqual(bank.coreset_ratio, DEFAULT_CORESET_RATIO)
        self.assertEqual(
            set(bank.memory_source_indices.tolist()),
            {0, 1, 2, 3},
        )
        self.assertTrue(
            all(values.ndim == 2 for values in bank.feature_memory.values())
        )

    def test_save_load_preserves_logsad_configuration(self) -> None:
        bank = self.build_bank()
        with tempfile.TemporaryDirectory() as directory:
            path = bank.save(Path(directory) / "demo.npz")
            loaded = PatchMemoryBank.load(
                path,
                expected_class_name="demo",
                expected_checkpoint_path="/tmp/dinov2.pth",
            )

        self.assertEqual(loaded.layers, PATCH_MATCHING_LAYERS)
        self.assertEqual(loaded.feature_grid, (2, 2))
        self.assertEqual(loaded.input_size, 448)
        self.assertAlmostEqual(loaded.score_mean, bank.score_mean)
        self.assertAlmostEqual(loaded.score_unbiased_std, bank.score_unbiased_std)
        self.assertEqual(loaded.calibration_images, bank.calibration_images)
        self.assertEqual(loaded.coreset_ratio, bank.coreset_ratio)
        torch.testing.assert_close(
            loaded.memory_source_indices, bank.memory_source_indices
        )

    def test_per_image_sharded_save_keeps_four_layers_together(self) -> None:
        bank = self.build_bank()
        with tempfile.TemporaryDirectory() as directory:
            path = bank.save(
                Path(directory) / "demo.npz",
                shard_by_image=True,
            )
            with np.load(path, allow_pickle=False) as manifest:
                metadata = json.loads(str(manifest["metadata"].item()))
            self.assertEqual(metadata["storage"], PATCH_MEMORY_PER_IMAGE_STORAGE)
            self.assertEqual(len(metadata["image_shards"]), 4)
            shard_store = path.parent / metadata["image_shard_store"]
            for image_index, entry in enumerate(metadata["image_shards"]):
                with np.load(shard_store / entry["file"], allow_pickle=False) as shard:
                    self.assertEqual(
                        set(shard.files),
                        {
                            "metadata",
                            "features_block_6",
                            "features_block_12",
                            "features_block_18",
                            "features_block_24",
                        },
                    )
                    shard_metadata = json.loads(str(shard["metadata"].item()))
                    self.assertEqual(shard_metadata["image_index"], image_index)
            loaded = PatchMemoryBank.load(path)

        torch.testing.assert_close(
            loaded.memory_source_indices, bank.memory_source_indices
        )
        for layer in PATCH_MATCHING_LAYERS:
            torch.testing.assert_close(
                loaded.feature_memory[layer], bank.feature_memory[layer]
            )

    def test_per_image_sharded_load_rejects_damaged_shard(self) -> None:
        bank = self.build_bank()
        with tempfile.TemporaryDirectory() as directory:
            path = bank.save(
                Path(directory) / "demo.npz",
                shard_by_image=True,
            )
            with np.load(path, allow_pickle=False) as manifest:
                metadata = json.loads(str(manifest["metadata"].item()))
            first = metadata["image_shards"][0]
            shard = path.parent / metadata["image_shard_store"] / first["file"]
            shard.write_bytes(b"damaged")
            with self.assertRaisesRegex(Exception, "checksum failed"):
                PatchMemoryBank.load(path)

    def test_old_patch_schema_is_explicitly_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.npz"
            np.savez_compressed(
                path,
                metadata=np.asarray(json.dumps({"schema": "old-patch-schema"})),
            )
            with self.assertRaisesRegex(Exception, "rebuild the LogSAD bank"):
                PatchMemoryBank.load(path)

    def test_runtime_scores_full_image_only(self) -> None:
        bank = self.build_bank()
        extractor = FixedBatchExtractor()
        scorer = DinoV2PatchImageScorer(bank=bank, extractor=extractor)
        evidence = scorer.score(np.zeros((6, 6, 3), dtype=np.uint8))

        self.assertEqual(extractor.input_count, 1)
        self.assertIsNotNone(evidence.full_image_score)

    def test_pipeline_records_patch_evidence_and_structural_status(self) -> None:
        standard = parse_template(
            "[C001] TASK=COUNT | SUBJECT=part | PROPERTY=count | OP=EQ | VALUE=[0]"
        )
        pipeline = LogicAnomalyPipeline(
            EmptySegmenter(),
            UnusedAttributeDetector(),
            patch_scorer=FixedPatchScorer(0.8, True),
        )
        report = pipeline.run(
            np.zeros((4, 4, 3), dtype=np.uint8),
            to_generic_template(standard),
            standard,
        )

        self.assertEqual(report.status, "STRUCTURAL_ANOMALY")
        self.assertEqual(report.anomaly_source, "structural")
        payload = report.to_dict()
        self.assertEqual(
            set(payload["patch_evidence"]["layer_scores"]),
            {"6", "12", "18", "24"},
        )
        self.assertIn("aggregate_raw_score", payload["patch_evidence"])
        self.assertIn("calibrated_score", payload["patch_evidence"])

    def test_patch_only_pipeline_skips_rule_dependencies(self) -> None:
        pipeline = LogicAnomalyPipeline(
            None,
            None,
            patch_scorer=FixedPatchScorer(0.8, True),
            rule_inference_enabled=False,
        )

        report = pipeline.run(
            np.zeros((4, 4, 3), dtype=np.uint8),
            (),
            (),
        )

        self.assertEqual(report.image_anomaly_score, 0.8)
        self.assertTrue(report.image_anomaly)
        self.assertIsNone(report.logical_anomaly)
        self.assertEqual(report.results, ())
        self.assertEqual(report.mask_counts, {})
        self.assertIsNone(report.filled_template)
        self.assertFalse(report.rule_inference_enabled)
        self.assertEqual(report.to_dict()["execution_mode"], "patch_only")

    def test_disabling_rules_without_patch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires patch matching"):
            LogicAnomalyPipeline(
                None,
                None,
                rule_inference_enabled=False,
            )

    def test_higher_rule_score_beats_patch_score(self) -> None:
        standard = parse_template(
            "[C001] TASK=COUNT | SUBJECT=part | PROPERTY=count | OP=EQ | VALUE=[1]"
        )
        pipeline = LogicAnomalyPipeline(
            EmptySegmenter(),
            UnusedAttributeDetector(),
            patch_scorer=FixedPatchScorer(0.8, True),
        )
        report = pipeline.run(
            np.zeros((4, 4, 3), dtype=np.uint8),
            to_generic_template(standard),
            standard,
        )

        self.assertEqual(report.status, "RULE_ANOMALY")


if __name__ == "__main__":
    unittest.main()
