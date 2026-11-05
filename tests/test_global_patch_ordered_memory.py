from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from pro_innovation.adapters.patch_dinov2 import PatchFeatureBundle
from pro_innovation.memory_modules.gpojm import (
    BACKGROUND_BLOCK,
    GlobalPatchOrderedMemoryBank,
    global_patch_texture_signature,
)
from pro_innovation.models import MaskInstance


def _mask(left: bool) -> MaskInstance:
    value = np.zeros((2, 2), dtype=bool)
    value[:, :1] = left
    value[:, 1:] = not left
    return MaskInstance(value)


def _features(left: float, right: float) -> PatchFeatureBundle:
    values = torch.tensor(
        [[left, 0.0], [right, 0.0], [left, 0.0], [right, 0.0]],
        dtype=torch.float32,
    )
    return PatchFeatureBundle(
        layer_features={layer: values for layer in (6, 12, 18, 24)},
        feature_grid=(2, 2),
    )


class GlobalPatchOrderedMemoryTests(unittest.TestCase):
    def test_texture_signature_keeps_category_order_and_background(self) -> None:
        anomaly_map = torch.tensor([1.0, 2.0, 3.0, 4.0])
        signature, payload = global_patch_texture_signature(
            anomaly_map,
            {"left": (_mask(True),)},
            ("left", "missing"),
            feature_grid=(2, 2),
            top_fraction=0.5,
        )

        torch.testing.assert_close(signature[0], torch.tensor([2.0, 3.0, 3.0]))
        torch.testing.assert_close(signature[1], torch.zeros(3))
        torch.testing.assert_close(signature[2], torch.tensor([3.0, 4.0, 4.0]))
        self.assertEqual(tuple(payload), ("left", "missing", BACKGROUND_BLOCK))

    def test_full_image_object_keeps_background_block_as_zero(self) -> None:
        anomaly_map = torch.tensor([1.0, 2.0, 3.0, 4.0])
        full = MaskInstance(np.ones((2, 2), dtype=bool))

        signature, payload = global_patch_texture_signature(
            anomaly_map,
            {"demo": (full,)},
            ("demo",),
            feature_grid=(2, 2),
            top_fraction=0.5,
        )

        torch.testing.assert_close(signature[0], torch.tensor([2.5, 3.5, 4.0]))
        torch.testing.assert_close(signature[1], torch.zeros(3))
        self.assertEqual(tuple(payload), ("demo", BACKGROUND_BLOCK))

    def test_joint_memory_emits_one_score_and_round_trips(self) -> None:
        bank = GlobalPatchOrderedMemoryBank.build_from_components(
            class_name="demo",
            categories=("a", "b"),
            category_features=torch.tensor(
                [
                    [[0.0, 0.0], [0.0, 1.0]],
                    [[1.0, 0.0], [1.0, 1.0]],
                    [[2.0, 1.0], [1.0, 2.0]],
                ]
            ),
            texture_features=torch.tensor(
                [
                    [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.1, 0.1, 0.1]],
                    [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.2, 0.2, 0.2]],
                    [[0.5, 0.6, 0.7], [0.4, 0.5, 0.6], [0.3, 0.4, 0.5]],
                ]
            ),
            normal_images=("/tmp/a.png", "/tmp/b.png", "/tmp/c.png"),
            checkpoint_path="/tmp/dino.pth",
            global_patch_bank_path="/tmp/global.npz",
            feature_grid=(2, 2),
            texture_weight=0.3,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = bank.save(Path(directory) / "joint.npz")
            loaded = GlobalPatchOrderedMemoryBank.load(
                path,
                expected_class_name="demo",
                expected_categories=("a", "b"),
                expected_global_patch_bank_path="/tmp/global.npz",
            )
        self.assertEqual(loaded.texture_weight, 0.3)
        self.assertGreater(loaded.loo_max_distance, 0.0)
        torch.testing.assert_close(loaded.texture_memory, bank.texture_memory)

        evidence = loaded.score_features(
            _features(1.0, 2.0),
            {"a": (_mask(True),), "b": (_mask(False),)},
            torch.tensor([0.1, 0.2, 0.3, 0.4]),
        )
        self.assertGreaterEqual(evidence.calibrated_score, 0.0)
        self.assertLess(evidence.calibrated_score, 1.0)
        self.assertEqual(tuple(evidence.category_distances), ("a", "b"))
        self.assertEqual(
            tuple(evidence.texture_distances), ("a", "b", BACKGROUND_BLOCK)
        )


if __name__ == "__main__":
    unittest.main()
