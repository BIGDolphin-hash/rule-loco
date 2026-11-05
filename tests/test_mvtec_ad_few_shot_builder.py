from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "global_patch_ordered"
    / "build_mvtec_ad_few_shot.py"
)
SPEC = importlib.util.spec_from_file_location("build_mvtec_ad_few_shot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MvtecAdFewShotBuilderTests(unittest.TestCase):
    def test_selection_is_deterministic_and_nested(self) -> None:
        images = tuple(Path(f"/{index:03d}.png") for index in range(20))
        first = MODULE.select_nested_images(
            images,
            class_name="bottle",
            seed=7,
            maximum_shot=8,
        )
        second = MODULE.select_nested_images(
            images,
            class_name="bottle",
            seed=7,
            maximum_shot=8,
        )

        self.assertEqual(first, second)
        self.assertEqual(first[:2], second[:4][:2])
        self.assertEqual(len(set(first)), 8)

    def test_selection_changes_with_seed(self) -> None:
        images = tuple(Path(f"/{index:03d}.png") for index in range(20))
        first = MODULE.select_nested_images(
            images,
            class_name="bottle",
            seed=0,
            maximum_shot=4,
        )
        second = MODULE.select_nested_images(
            images,
            class_name="bottle",
            seed=1,
            maximum_shot=4,
        )

        self.assertNotEqual(first, second)

    def test_one_shot_is_supported(self) -> None:
        selected = MODULE.select_nested_images(
            (Path("/one.png"),),
            class_name="bottle",
            seed=0,
            maximum_shot=1,
        )
        self.assertEqual(selected, (Path("/one.png"),))

    def test_main_reuses_one_patch_extractor_for_all_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "dinov2.pth"
            checkpoint.write_bytes(b"checkpoint")
            first_image = root / "bottle.png"
            second_image = root / "cable.png"
            first_image.write_bytes(b"image")
            second_image.write_bytes(b"image")
            args = SimpleNamespace(
                dataset="mvtec_ad",
                dataset_root=root / "dataset",
                global_root=root / "global",
                gpojm_root=root / "gpojm",
                manifest_root=root / "manifests",
                dino_checkpoint=checkpoint,
                shots=(1,),
                seed=0,
                classes=None,
                device="cuda",
                query_batch_size=256,
                extraction_batch_size=4,
                texture_weight=0.30,
                top_fraction=0.10,
                global_only=True,
            )
            shared_extractor = object()
            extractor_constructor = Mock(return_value=shared_extractor)
            build_memory = Mock()

            with (
                patch.object(MODULE, "_arguments", return_value=args),
                patch.object(
                    MODULE,
                    "_normal_images_by_class",
                    return_value={
                        "bottle": (first_image,),
                        "cable": (second_image,),
                    },
                ),
                patch.object(MODULE, "_global_bank_matches", return_value=False),
                patch.object(
                    MODULE,
                    "DinoV2PatchFeatureExtractor",
                    extractor_constructor,
                ),
                patch.object(
                    MODULE,
                    "build_patch_memory_from_images",
                    build_memory,
                ),
            ):
                self.assertEqual(MODULE.main(), 0)

            extractor_constructor.assert_called_once_with(
                checkpoint_path=checkpoint.resolve(),
                device="cuda",
            )
            self.assertEqual(build_memory.call_count, 2)
            self.assertTrue(
                all(
                    call.kwargs["extractor"] is shared_extractor
                    for call in build_memory.call_args_list
                )
            )


if __name__ == "__main__":
    unittest.main()
