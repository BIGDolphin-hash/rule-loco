from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from pro_innovation.image_validation import filter_decodable_images


class ImageValidationTests(unittest.TestCase):
    def test_unreadable_image_is_reported_and_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good.png"
            bad = root / "bad.png"
            Image.new("RGB", (4, 4), color="white").save(good)
            bad.write_bytes(b"not a valid image stream")
            events = []

            retained = filter_decodable_images(
                (good, bad),
                progress_callback=lambda *event: events.append(event),
            )

        self.assertEqual(retained, (good.resolve(),))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "skip-unreadable")
        self.assertEqual(events[0][1:3], (2, 2))
        self.assertIn("bad.png", events[0][3])

    def test_png_syntax_error_is_reported_and_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-idat.png"
            path.write_bytes(b"placeholder")
            events = []

            with patch(
                "pro_innovation.image_validation.Image.open",
                side_effect=SyntaxError(
                    "broken PNG file (bad header checksum in b'IDAT')"
                ),
            ):
                retained = filter_decodable_images(
                    (path,),
                    progress_callback=lambda *event: events.append(event),
                )

        self.assertEqual(retained, ())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "skip-unreadable")
        self.assertIn("bad-idat.png", events[0][3])
        self.assertIn("bad header checksum", events[0][3])


if __name__ == "__main__":
    unittest.main()
