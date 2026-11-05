from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from pro_innovation.adapters.patch_dinov2 import PatchFeatureBundle
from pro_innovation.memory_modules.gpstm import (
    BACKGROUND_BLOCK,
    GPSTM_MODE,
    GlobalPatchSemanticTextureMemoryBank,
)
from pro_innovation.models import MaskInstance


def _mask(left: bool) -> MaskInstance:
    value = np.zeros((2, 2), dtype=bool)
    value[:, :1] = left
    value[:, 1:] = not left
    return MaskInstance(value)


def _features() -> PatchFeatureBundle:
    values = torch.ones((4, 2), dtype=torch.float32)
    return PatchFeatureBundle(
        layer_features={layer: values for layer in (6, 12, 18, 24)},
        feature_grid=(2, 2),
    )


class GlobalPatchSemanticTextureMemoryTests(unittest.TestCase):
    def test_texture_only_bank_round_trips_and_emits_one_score(self) -> None:
        bank = GlobalPatchSemanticTextureMemoryBank.build_from_signatures(
            class_name="demo",
            categories=("a", "b"),
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
        )
        with tempfile.TemporaryDirectory() as directory:
            path = bank.save(Path(directory) / "gpstm.npz")
            loaded = GlobalPatchSemanticTextureMemoryBank.load(
                path,
                expected_class_name="demo",
                expected_categories=("a", "b"),
                expected_global_patch_bank_path="/tmp/global.npz",
            )

        self.assertFalse(hasattr(loaded, "category_memory"))
        evidence = loaded.score_features(
            _features(),
            {"a": (_mask(True),), "b": (_mask(False),)},
            torch.tensor([0.1, 0.2, 0.3, 0.4]),
        )
        self.assertGreaterEqual(evidence.calibrated_score, 0.0)
        self.assertLess(evidence.calibrated_score, 1.0)
        self.assertEqual(evidence.mode, GPSTM_MODE)
        self.assertEqual(
            tuple(evidence.texture_distances), ("a", "b", BACKGROUND_BLOCK)
        )
        self.assertFalse(hasattr(evidence, "category_distances"))


if __name__ == "__main__":
    unittest.main()
