from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from pro_innovation.adapters import WholeImageSegmenter
from pro_innovation.models import CompositionEvidence
from pro_innovation.pipeline import LogicAnomalyPipeline


class _Extractor:
    def extract_features(self, image):
        return image


class _CompositionScorer:
    def __init__(self) -> None:
        self.mask_pool = None

    def score_features(self, features, mask_pool):
        del features
        self.mask_pool = mask_pool
        return CompositionEvidence(
            category_distances={"demo": 0.1},
            aggregate_raw_score=0.1,
            threshold=0.2,
            adaptive_score=0.5,
            calibrated_score=1.0 / 3.0,
            anomaly=False,
            nearest_reference_index=0,
        )


class WholeImageRuntimeTests(unittest.TestCase):
    def test_segmenter_returns_one_full_image_object(self) -> None:
        image = np.zeros((3, 5, 3), dtype=np.uint8)

        pool = WholeImageSegmenter().segment(image, ("demo",))

        self.assertEqual(tuple(pool), ("demo",))
        self.assertEqual(len(pool["demo"]), 1)
        instance = pool["demo"][0]
        self.assertEqual(instance.mask.shape, (3, 5))
        self.assertTrue(instance.mask.all())
        self.assertEqual(instance.score, 1.0)
        self.assertEqual(instance.box, (0.0, 0.0, 5.0, 3.0))

    def test_segmenter_reads_file_dimensions_without_sam3(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (7, 4)).save(path)

            instance = WholeImageSegmenter().segment(path, ("demo",))["demo"][0]

        self.assertEqual(instance.mask.shape, (4, 7))
        self.assertTrue(instance.mask.all())

    def test_no_template_pipeline_uses_explicit_mask_category_and_no_rules(
        self,
    ) -> None:
        scorer = _CompositionScorer()
        pipeline = LogicAnomalyPipeline(
            WholeImageSegmenter(),
            None,
            composition_scorer=scorer,
            shared_feature_extractor=_Extractor(),
            rule_inference_enabled=False,
            mask_categories=("demo",),
        )

        report = pipeline.run(np.zeros((3, 5, 3), dtype=np.uint8), (), ())

        self.assertFalse(report.rule_inference_enabled)
        self.assertEqual(report.results, ())
        self.assertEqual(report.execution_mode, "composition_only")
        self.assertEqual(report.categories, ("demo",))
        self.assertEqual(report.mask_counts, {"demo": 1})
        self.assertIsNotNone(scorer.mask_pool)
        assert scorer.mask_pool is not None
        self.assertTrue(scorer.mask_pool["demo"][0].mask.all())
