from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch

from pro_innovation.errors import MemoryBankError
from pro_innovation.memory_modules.gcda import (
    CATEGORY_ONLY_FEATURE_MODE,
    CategoryOnlyCompositionMemoryBank,
    GlobalCategoryOnlyDualScorer,
)
from pro_innovation.models import CompositionEvidence, PatchEvidence


def _bank() -> CategoryOnlyCompositionMemoryBank:
    return CategoryOnlyCompositionMemoryBank.build_from_embeddings(
        class_name="demo",
        categories=("a", "b"),
        embeddings=torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 2.0, 1.0], [3.0, 1.0, 1.0, 3.0]]
        ),
        normal_images=("/tmp/a.png", "/tmp/b.png", "/tmp/c.png"),
        checkpoint_path="/tmp/dino.pth",
        feature_grid=(2, 2),
    )


class _GlobalPatch:
    def extract_features(self, image):
        return image

    def score_features(self, features):
        del features
        return PatchEvidence(
            layer_scores={"6": 0.1},
            aggregate_raw_score=0.1,
            threshold=0.2,
            calibrated_score=0.4,
            anomaly=False,
            fusion_score=0.75,
        )


class _CategoryOnly:
    def score_features(self, features, mask_pool):
        del features, mask_pool
        return CompositionEvidence(
            category_distances={"a": 0.2, "b": 0.3},
            aggregate_raw_score=0.5,
            threshold=0.5,
            adaptive_score=1.25,
            calibrated_score=1.25 / 2.25,
            anomaly=True,
            nearest_reference_index=0,
        )


class GlobalCategoryOnlyDualTests(unittest.TestCase):
    def test_bank_uses_distinct_schema_and_round_trips(self) -> None:
        bank = _bank()
        with tempfile.TemporaryDirectory() as directory:
            path = bank.save(Path(directory) / "demo.category_only.npz")
            loaded = CategoryOnlyCompositionMemoryBank.load(
                path,
                expected_class_name="demo",
                expected_categories=("a", "b"),
            )
        self.assertEqual(loaded.feature_mode, CATEGORY_ONLY_FEATURE_MODE)
        torch.testing.assert_close(loaded.embeddings, bank.embeddings)
        self.assertEqual(loaded.loo_max_distance, bank.loo_max_distance)

    def test_unrelated_bank_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.npz"
            torch.save({"not": "an npz"}, path)
            with self.assertRaises(MemoryBankError):
                CategoryOnlyCompositionMemoryBank.load(path)

    def test_dual_fusion_adds_raw_ratios_before_calibration(self) -> None:
        scorer = GlobalCategoryOnlyDualScorer(
            global_patch_scorer=_GlobalPatch(),
            category_only_scorer=_CategoryOnly(),  # type: ignore[arg-type]
        )
        evidence = scorer.score_features(object(), {})  # type: ignore[arg-type]
        self.assertEqual(evidence.aggregate_ratio, 2.0)
        self.assertAlmostEqual(evidence.calibrated_score, 2.0 / 3.0)
        self.assertTrue(evidence.anomaly)


if __name__ == "__main__":
    unittest.main()
