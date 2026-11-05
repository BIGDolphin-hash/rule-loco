from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from pro_innovation.adapters.sam3 import Sam3Segmenter
from pro_innovation.errors import ModelAdapterError
from pro_innovation.models import MaskInstance


class Sam3CacheTests(unittest.TestCase):
    def _segmenter(
        self,
        root: Path,
        checkpoint: Path,
        *,
        threshold: float,
        mode: str,
    ) -> Sam3Segmenter:
        return Sam3Segmenter(
            repo_path=root,
            checkpoint_path=checkpoint,
            device="cpu",
            confidence_threshold=threshold,
            cache_root=root / "cache",
            cache_mode=mode,
        )

    def test_lower_threshold_cache_is_reused_and_filtered_at_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "sam3.pt"
            checkpoint.write_bytes(b"checkpoint")
            image = Image.new("RGB", (8, 6), "white")
            low = self._segmenter(
                root, checkpoint, threshold=0.30, mode="read-write"
            )
            mask_a = np.zeros((6, 8), dtype=bool)
            mask_b = np.zeros((6, 8), dtype=bool)
            mask_a[1, 1] = True
            mask_b[2, 2] = True
            low._write_cache(
                image,
                "part",
                (
                    MaskInstance(mask_a, score=0.85, box=(0, 0, 2, 2)),
                    MaskInstance(mask_b, score=0.45, box=(1, 1, 3, 3)),
                ),
            )

            high = self._segmenter(root, checkpoint, threshold=0.70, mode="read")
            result = high.segment(image, ("part",))

            self.assertEqual(len(result["part"]), 1)
            self.assertAlmostEqual(result["part"][0].score, 0.85, places=6)
            self.assertEqual(high.cache_hits, 1)
            self.assertIsNone(high._processor)

    def test_higher_threshold_cache_cannot_serve_lower_threshold_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "sam3.pt"
            checkpoint.write_bytes(b"checkpoint")
            image = Image.new("RGB", (4, 4), "black")
            high = self._segmenter(root, checkpoint, threshold=0.70, mode="read-write")
            mask = np.ones((4, 4), dtype=bool)
            high._write_cache(
                image,
                "part",
                (MaskInstance(mask, score=0.90, box=(0, 0, 4, 4)),),
            )

            low = self._segmenter(root, checkpoint, threshold=0.30, mode="read")
            with self.assertRaisesRegex(ModelAdapterError, "cache miss"):
                low.segment(image, ("part",))

    def test_float32_threshold_rounding_does_not_cause_equal_threshold_miss(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "sam3.pt"
            checkpoint.write_bytes(b"checkpoint")
            image = Image.new("RGB", (4, 4), "white")
            writer = self._segmenter(
                root, checkpoint, threshold=0.60, mode="read-write"
            )
            mask = np.ones((4, 4), dtype=bool)
            writer._write_cache(
                image,
                "part",
                (MaskInstance(mask, score=0.80, box=(0, 0, 4, 4)),),
            )
            cache_path = writer._cache_path(image, "part")
            with np.load(cache_path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["proposal_threshold"] = np.asarray(0.60, dtype=np.float32)
            np.savez_compressed(cache_path, **arrays)

            reader = self._segmenter(root, checkpoint, threshold=0.60, mode="read")
            result = reader.segment(image, ("part",))

            self.assertEqual(len(result["part"]), 1)
            self.assertEqual(reader.cache_hits, 1)
            self.assertEqual(reader.cache_misses, 0)

    def test_new_cache_stores_threshold_as_float64(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "sam3.pt"
            checkpoint.write_bytes(b"checkpoint")
            image = Image.new("RGB", (4, 4), "white")
            writer = self._segmenter(
                root, checkpoint, threshold=0.60, mode="read-write"
            )
            writer._write_cache(image, "part", ())

            with np.load(
                writer._cache_path(image, "part"), allow_pickle=False
            ) as archive:
                stored = archive["proposal_threshold"]

            self.assertEqual(stored.dtype, np.dtype(np.float64))
            self.assertEqual(float(stored.item()), 0.60)

    def test_cache_identity_changes_with_image_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "sam3.pt"
            checkpoint.write_bytes(b"first")
            segmenter = self._segmenter(
                root, checkpoint, threshold=0.30, mode="read-write"
            )
            first = segmenter._cache_path(Image.new("RGB", (3, 3), "red"), "part")
            second = segmenter._cache_path(
                Image.new("RGB", (3, 3), "blue"), "part"
            )
            checkpoint.write_bytes(b"a different checkpoint")
            third = segmenter._cache_path(Image.new("RGB", (3, 3), "red"), "part")

            self.assertNotEqual(first, second)
            self.assertNotEqual(first, third)


if __name__ == "__main__":
    unittest.main()
