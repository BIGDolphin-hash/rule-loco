from __future__ import annotations

import unittest
from typing import cast

import torch

from pro_innovation.adapters.patch_dinov2 import PatchFeatureBundle
from pro_innovation.errors import MemoryBankError
from pro_innovation.memory_modules.gpojm import (
    GlobalPatchOrderedScorer,
    GlobalPatchOrderedMemoryBank,
    GlobalPatchPixelLocalizer,
    GlobalPatchPixelScorer,
)


def _features() -> PatchFeatureBundle:
    values = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.25, 0.75]],
        dtype=torch.float32,
    )
    return PatchFeatureBundle(
        layer_features={layer: values for layer in (6, 12, 18, 24)},
        feature_grid=(2, 2),
    )


class _GlobalPatchScorer:
    def __init__(self) -> None:
        self.map_calls = 0
        self.extract_calls = 0

    def extract_features(self, image):
        del image
        self.extract_calls += 1
        return _features()

    def score_features_with_map(self, features):
        self.map_calls += 1
        self.last_features = features
        return object(), torch.tensor([0.0, 1.0, 2.0, 3.0])


class _JointBank:
    def __init__(self) -> None:
        self.score_calls = 0

    def score_features(self, features, mask_pool, anomaly_map):
        self.score_calls += 1
        self.last_call = (features, mask_pool, anomaly_map.clone())
        return "joint-evidence"


class GlobalPatchPixelLocalizerTests(unittest.TestCase):
    def test_localize_restores_patch_map_to_image_resolution(self) -> None:
        localizer = GlobalPatchPixelLocalizer()

        result = localizer.localize(
            torch.tensor([0.0, 1.0, 2.0, 3.0]),
            feature_grid=(2, 2),
            output_size=(6, 8),
        )

        torch.testing.assert_close(
            result.patch_anomaly_map,
            torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        )
        self.assertEqual(tuple(result.pixel_anomaly_map.shape), (6, 8))
        self.assertEqual(result.output_size, (6, 8))
        self.assertEqual(result.smoothing_sigma, 0.0)
        self.assertGreaterEqual(float(result.pixel_anomaly_map.min()), 0.0)
        self.assertLessEqual(float(result.pixel_anomaly_map.max()), 3.0)

    def test_optional_smoothing_reduces_a_local_peak(self) -> None:
        raw = GlobalPatchPixelLocalizer().localize(
            torch.tensor([0.0, 0.0, 0.0, 4.0]),
            feature_grid=(2, 2),
            output_size=(8, 8),
        )
        smoothed = GlobalPatchPixelLocalizer(smoothing_sigma=1.0).localize(
            torch.tensor([0.0, 0.0, 0.0, 4.0]),
            feature_grid=(2, 2),
            output_size=(8, 8),
        )

        self.assertEqual(tuple(smoothed.pixel_anomaly_map.shape), (8, 8))
        self.assertLess(
            float(smoothed.pixel_anomaly_map.max()),
            float(raw.pixel_anomaly_map.max()),
        )

    def test_invalid_anomaly_map_is_rejected(self) -> None:
        localizer = GlobalPatchPixelLocalizer()
        with self.assertRaisesRegex(MemoryBankError, "does not match"):
            localizer.localize(
                torch.tensor([0.0, 1.0, 2.0]),
                feature_grid=(2, 2),
                output_size=(4, 4),
            )
        with self.assertRaisesRegex(MemoryBankError, "non-finite"):
            localizer.localize(
                torch.tensor([0.0, 1.0, 2.0, float("nan")]),
                feature_grid=(2, 2),
                output_size=(4, 4),
            )

    def test_pure_texture_path_skips_joint_memory(self) -> None:
        global_patch = _GlobalPatchScorer()
        joint_bank = _JointBank()
        scorer = GlobalPatchOrderedScorer(
            bank=cast(GlobalPatchOrderedMemoryBank, joint_bank),
            global_patch_scorer=global_patch,
        )

        result = scorer.localize_features(_features(), output_size=(5, 7))

        self.assertEqual(global_patch.map_calls, 1)
        self.assertEqual(joint_bank.score_calls, 0)
        self.assertEqual(tuple(result.pixel_anomaly_map.shape), (5, 7))

    def test_standalone_pixel_scorer_needs_no_joint_bank(self) -> None:
        global_patch = _GlobalPatchScorer()
        scorer = GlobalPatchPixelScorer(global_patch_scorer=global_patch)

        result = scorer.localize("image", output_size=(5, 7))

        self.assertEqual(global_patch.extract_calls, 1)
        self.assertEqual(global_patch.map_calls, 1)
        self.assertEqual(tuple(result.pixel_anomaly_map.shape), (5, 7))

    def test_joint_and_pixel_outputs_share_one_global_patch_map(self) -> None:
        global_patch = _GlobalPatchScorer()
        joint_bank = _JointBank()
        scorer = GlobalPatchOrderedScorer(
            bank=cast(GlobalPatchOrderedMemoryBank, joint_bank),
            global_patch_scorer=global_patch,
        )

        evidence, result = scorer.score_features_with_pixel_map(
            _features(),
            {"demo": ()},
            output_size=(5, 7),
        )

        self.assertEqual(evidence, "joint-evidence")
        self.assertEqual(global_patch.map_calls, 1)
        self.assertEqual(joint_bank.score_calls, 1)
        torch.testing.assert_close(
            joint_bank.last_call[2],
            result.patch_anomaly_map.reshape(-1),
        )


if __name__ == "__main__":
    unittest.main()
