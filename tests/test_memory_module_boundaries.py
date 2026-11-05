from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

from pro_innovation.adapters.patch_dinov2 import (
    PATCH_MATCHING_LAYERS,
    PatchFeatureBundle,
)
from pro_innovation.memory_modules.gcda import GCDAGlobalPatchBank
from pro_innovation.memory_modules.global_patch import PatchMemoryBank
from pro_innovation.memory_modules.gpojm import GPOJMGlobalPatchBank
from pro_innovation.memory_modules.gpstm import GPSTMGlobalPatchBank

MODULE_ROOT = (
    Path(__file__).resolve().parents[1] / "src" / "pro_innovation" / "memory_modules"
)
MODULE_NAMES = ("global_patch", "gpstm", "gcda", "gpojm")
LEGACY_ENTRYPOINTS = {
    "pro_innovation.patch_memory": "global_patch",
    "pro_innovation.global_patch_semantic_texture_memory": "gpstm",
    "pro_innovation.global_category_only_dual": "gcda",
    "pro_innovation.global_patch_ordered_memory": "gpojm",
}


def _package_for(path: Path) -> str:
    relative = path.relative_to(MODULE_ROOT)
    return ".".join(("pro_innovation", "memory_modules", *relative.parts[:-1]))


def _resolve_import(node: ast.ImportFrom, package: str) -> str:
    if node.level == 0:
        return node.module or ""
    target = "." * node.level + (node.module or "")
    return importlib.util.resolve_name(target, package)


class MemoryModuleBoundaryTests(unittest.TestCase):
    def test_four_module_packages_exist(self) -> None:
        self.assertEqual(
            tuple(name for name in MODULE_NAMES if (MODULE_ROOT / name).is_dir()),
            MODULE_NAMES,
        )
        for name in MODULE_NAMES:
            self.assertTrue((MODULE_ROOT / name / "core.py").is_file())
        for name in ("gpstm", "gcda", "gpojm"):
            self.assertTrue((MODULE_ROOT / name / "global_patch.py").is_file())

    def test_module_packages_do_not_import_each_other(self) -> None:
        violations: list[str] = []
        for owner in MODULE_NAMES:
            for path in sorted((MODULE_ROOT / owner).glob("*.py")):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                package = _package_for(path)
                imported_names: list[str] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported_names.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        imported_names.append(_resolve_import(node, package))
                for imported in imported_names:
                    parts = imported.split(".")
                    if (
                        parts[:2] == ["pro_innovation", "memory_modules"]
                        and len(parts) >= 3
                    ):
                        imported_owner = parts[2]
                        if imported_owner in MODULE_NAMES and imported_owner != owner:
                            violations.append(
                                f"{path.name}: {owner} -> {imported_owner}"
                            )
                    legacy_owner = LEGACY_ENTRYPOINTS.get(imported)
                    if legacy_owner is not None and legacy_owner != owner:
                        violations.append(f"{path.name}: {owner} -> {legacy_owner}")
        self.assertEqual(violations, [])

    def test_private_global_patch_readers_preserve_scores_and_maps(self) -> None:
        rows = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [2.0**-0.5, 2.0**-0.5]],
            dtype=torch.float32,
        )
        normal_images = ("/tmp/normal-0.png", "/tmp/normal-1.png", "/tmp/normal-2.png")
        bank = PatchMemoryBank(
            class_name="demo",
            feature_memory={layer: rows.clone() for layer in PATCH_MATCHING_LAYERS},
            memory_source_indices=torch.tensor([0, 1, 2]),
            feature_grid=(1, 1),
            checkpoint_path="/tmp/dino.pth",
            normal_images=normal_images,
            score_mean=0.30,
            score_unbiased_std=0.10,
            calibration_images=normal_images,
            calibration_scores=(0.20, 0.30, 0.40),
            coreset_ratio=1.0,
            features_are_normalized=True,
        )
        query = PatchFeatureBundle(
            layer_features={
                layer: torch.tensor([[0.6, 0.8]], dtype=torch.float32)
                for layer in PATCH_MATCHING_LAYERS
            },
            feature_grid=(1, 1),
        )

        with tempfile.TemporaryDirectory() as directory:
            for sharded in (False, True):
                path = bank.save(
                    Path(directory) / f"global-{int(sharded)}.npz",
                    shard_by_image=sharded,
                )
                expected_evidence, expected_map = bank.score_with_map(
                    query, device="cpu"
                )
                for private_type in (
                    GPSTMGlobalPatchBank,
                    GCDAGlobalPatchBank,
                    GPOJMGlobalPatchBank,
                ):
                    loaded = private_type.load(
                        path,
                        expected_class_name="demo",
                        expected_checkpoint_path="/tmp/dino.pth",
                    )
                    evidence, anomaly_map = loaded.score_with_map(query, device="cpu")
                    torch.testing.assert_close(anomaly_map, expected_map)
                    self.assertEqual(
                        evidence.layer_scores, expected_evidence.layer_scores
                    )
                    self.assertAlmostEqual(
                        evidence.aggregate_raw_score,
                        expected_evidence.aggregate_raw_score,
                    )
                    self.assertAlmostEqual(
                        evidence.calibrated_score,
                        expected_evidence.calibrated_score,
                    )
                    self.assertAlmostEqual(
                        evidence.fusion_score,
                        expected_evidence.fusion_score,
                    )
                    self.assertEqual(
                        evidence.full_image_score,
                        expected_evidence.full_image_score,
                    )


if __name__ == "__main__":
    unittest.main()
