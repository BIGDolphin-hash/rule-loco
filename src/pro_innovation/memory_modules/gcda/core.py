"""GCDA memory construction, persistence, fusion, and runtime scoring.

It owns a distinct memory schema and can be instantiated only by callers that
explicitly select GCDA.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from ...adapters.patch_dinov2 import (
    PATCH_MATCHING_INPUT_SIZE,
    PATCH_MATCHING_LAYERS,
    PatchFeatureBundle,
)
from ...errors import MemoryBankError
from ...models import CompositionEvidence, MaskInstance, PatchEvidence

CATEGORY_ONLY_MEMORY_SCHEMA = (
    "pro-innovation-legacy-category-only-composition-memory-v1"
)
LEGACY_CATEGORY_ONLY_MEMORY_SCHEMA = (
    "pro-innovation-psad-composition-reg4-full-memory-v2"
)
CATEGORY_ONLY_FEATURE_MODE = "category_only"
GLOBAL_CATEGORY_ONLY_DUAL_MODE = "global_category_only_dual"
DEFAULT_DUAL_FUSION_TAU = 1.0
CATEGORY_ONLY_STANDARDIZATION_EPSILON = 1e-10


def _resize_category_mask(
    instance: MaskInstance,
    feature_grid: tuple[int, int],
) -> torch.Tensor:
    mask = np.asarray(instance.mask, dtype=bool)
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    resized = functional.interpolate(tensor, size=feature_grid, mode="nearest")[0, 0]
    selected = resized.reshape(-1) >= 0.5
    if selected.any() or not mask.any():
        return selected
    coverage = functional.interpolate(tensor, size=feature_grid, mode="area")[0, 0]
    return coverage.reshape(-1) > 0.0


def category_only_embedding(
    features: PatchFeatureBundle,
    mask_pool: Mapping[str, Sequence[MaskInstance]],
    categories: Sequence[str],
) -> tuple[torch.Tensor, dict[str, slice]]:
    """Build the legacy fixed-order ``[f1; ...; fC]`` descriptor."""

    ordered = tuple(str(category) for category in categories)
    if not ordered or len(set(ordered)) != len(ordered):
        raise MemoryBankError("category-only categories must be unique and non-empty")
    if tuple(sorted(features.layer_features)) != PATCH_MATCHING_LAYERS:
        raise MemoryBankError("category-only features are missing required DINO layers")
    positions = features.feature_grid[0] * features.feature_grid[1]
    dimensions = {int(value.shape[-1]) for value in features.layer_features.values()}
    if len(dimensions) != 1:
        raise MemoryBankError("category-only DINO layers use different dimensions")
    dimension = next(iter(dimensions))
    for layer in PATCH_MATCHING_LAYERS:
        values = features.layer_features[layer]
        if values.ndim != 2 or tuple(values.shape) != (positions, dimension):
            raise MemoryBankError(f"invalid category-only Block {layer} feature shape")

    chunks: list[torch.Tensor] = []
    slices: dict[str, slice] = {}
    cursor = 0
    for category in ordered:
        instance_vectors: list[torch.Tensor] = []
        instance_areas: list[float] = []
        for instance in mask_pool.get(category, ()):
            selected = _resize_category_mask(instance, features.feature_grid)
            if not selected.any():
                raise MemoryBankError(
                    f"category-only instance for {category!r} disappears "
                    "on the DINO feature grid"
                )
            layer_vectors = []
            for layer in PATCH_MATCHING_LAYERS:
                values = features.layer_features[layer].detach().float().cpu()
                pooled = values[selected].mean(dim=0)
                layer_vectors.append(functional.normalize(pooled, dim=0, eps=1e-12))
            instance_vector = torch.stack(layer_vectors, dim=0).mean(dim=0)
            instance_vectors.append(
                functional.normalize(instance_vector, dim=0, eps=1e-12)
            )
            instance_areas.append(float(np.count_nonzero(instance.mask)))

        if instance_vectors:
            weights = torch.tensor(instance_areas, dtype=torch.float32)
            weights = weights / weights.sum()
            chunk = torch.sum(
                torch.stack(instance_vectors, dim=0) * weights[:, None], dim=0
            )
            chunk = functional.normalize(chunk, dim=0, eps=1e-12)
        else:
            chunk = torch.zeros(dimension, dtype=torch.float32)
        chunks.append(chunk)
        slices[category] = slice(cursor, cursor + int(chunk.numel()))
        cursor += int(chunk.numel())
    return torch.cat(chunks, dim=0), slices


@dataclass(frozen=True)
class GlobalCategoryOnlyDualEvidence:
    """Evidence from the two independent legacy branches and their fusion."""

    global_patch: PatchEvidence
    category_only: CompositionEvidence
    aggregate_ratio: float
    calibrated_score: float
    anomaly: bool
    fusion_tau: float
    mode: str = GLOBAL_CATEGORY_ONLY_DUAL_MODE


class CategoryOnlyCompositionMemoryBank:
    """Fixed-order ``[f1; ...; fC]`` memory without category-patch modulation."""

    def __init__(
        self,
        *,
        class_name: str,
        categories: Sequence[str],
        embeddings: torch.Tensor,
        normal_images: Sequence[str | Path],
        checkpoint_path: str | Path,
        feature_grid: tuple[int, int],
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        loo_max_distance: float,
        source_path: str | None = None,
    ) -> None:
        self.class_name = str(class_name)
        self.categories = tuple(str(value) for value in categories)
        self.embeddings = embeddings.detach().float().cpu()
        self.normal_images = tuple(
            str(Path(path).expanduser().resolve()) for path in normal_images
        )
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.feature_grid = (int(feature_grid[0]), int(feature_grid[1]))
        self.feature_mean = feature_mean.detach().float().cpu()
        self.feature_std = feature_std.detach().float().cpu()
        self.loo_max_distance = float(loo_max_distance)
        self.source_path = source_path
        self.input_size = PATCH_MATCHING_INPUT_SIZE
        self.layers = PATCH_MATCHING_LAYERS
        self.feature_mode = CATEGORY_ONLY_FEATURE_MODE
        self._validate()

    def _validate(self) -> None:
        if not self.class_name:
            raise MemoryBankError("category-only memory class name is empty")
        if not self.categories or len(set(self.categories)) != len(self.categories):
            raise MemoryBankError(
                "category-only categories must be unique and non-empty"
            )
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] < 2:
            raise MemoryBankError(
                "category-only memory requires at least two embeddings"
            )
        if self.embeddings.shape[0] != len(self.normal_images):
            raise MemoryBankError("category-only embeddings and normal images differ")
        if len(set(self.normal_images)) != len(self.normal_images):
            raise MemoryBankError("category-only normal images must be distinct")
        if not torch.isfinite(self.embeddings).all():
            raise MemoryBankError("category-only memory contains non-finite values")
        dimension = int(self.embeddings.shape[1])
        if dimension % len(self.categories):
            raise MemoryBankError(
                "category-only embedding cannot be divided into category blocks"
            )
        if self.feature_mean.shape != (dimension,) or self.feature_std.shape != (
            dimension,
        ):
            raise MemoryBankError(
                "category-only statistics do not match embedding dimension"
            )
        if (
            not torch.isfinite(self.feature_mean).all()
            or not torch.isfinite(self.feature_std).all()
        ):
            raise MemoryBankError("category-only statistics are non-finite")
        if (self.feature_std < 0).any():
            raise MemoryBankError("category-only standard deviation is negative")
        if not isfinite(self.loo_max_distance) or self.loo_max_distance <= 0.0:
            raise MemoryBankError("category-only leave-one-out scale must be positive")
        if self.feature_grid[0] <= 0 or self.feature_grid[1] <= 0:
            raise MemoryBankError("category-only feature grid must be positive")

    @classmethod
    def build(
        cls,
        *,
        class_name: str,
        categories: Sequence[str],
        features: Sequence[PatchFeatureBundle],
        mask_pools: Sequence[Mapping[str, Sequence[MaskInstance]]],
        normal_images: Sequence[str | Path],
        checkpoint_path: str | Path,
    ) -> "CategoryOnlyCompositionMemoryBank":
        if len(features) != len(mask_pools) or len(features) != len(normal_images):
            raise ValueError("category-only features, masks, and images must align")
        grids = {tuple(bundle.feature_grid) for bundle in features}
        if len(grids) != 1:
            raise ValueError("normal category-only features use different grids")
        vectors = torch.stack(
            [
                category_only_embedding(bundle, pool, categories)[0]
                for bundle, pool in zip(features, mask_pools)
            ],
            dim=0,
        )
        return cls.build_from_embeddings(
            class_name=class_name,
            categories=categories,
            embeddings=vectors,
            normal_images=normal_images,
            checkpoint_path=checkpoint_path,
            feature_grid=next(iter(grids)),
        )

    @classmethod
    def build_from_embeddings(
        cls,
        *,
        class_name: str,
        categories: Sequence[str],
        embeddings: torch.Tensor,
        normal_images: Sequence[str | Path],
        checkpoint_path: str | Path,
        feature_grid: tuple[int, int],
    ) -> "CategoryOnlyCompositionMemoryBank":
        if embeddings.ndim != 2 or embeddings.shape[0] != len(normal_images):
            raise ValueError("category-only embeddings and images must align")
        if embeddings.shape[0] < 2:
            raise ValueError("category-only memory construction requires two images")
        raw = embeddings.detach().float().cpu()
        mean = raw.mean(dim=0)
        std = raw.std(dim=0, unbiased=True)
        standardized = (raw - mean) / (std + CATEGORY_ONLY_STANDARDIZATION_EPSILON)
        pairwise = torch.cdist(standardized, standardized, p=2)
        pairwise.fill_diagonal_(float("inf"))
        loo_max = float(pairwise.min(dim=1).values.max().item())
        if not isfinite(loo_max) or loo_max <= 1e-12:
            raise MemoryBankError(
                "normal category-only embeddings have zero leave-one-out distance"
            )
        return cls(
            class_name=class_name,
            categories=categories,
            embeddings=standardized,
            normal_images=normal_images,
            checkpoint_path=checkpoint_path,
            feature_grid=feature_grid,
            feature_mean=mean,
            feature_std=std,
            loo_max_distance=loo_max,
        )

    def score_features(
        self,
        features: PatchFeatureBundle,
        mask_pool: Mapping[str, Sequence[MaskInstance]],
    ) -> CompositionEvidence:
        if tuple(features.feature_grid) != self.feature_grid:
            raise MemoryBankError("category-only query grid does not match memory")
        raw_query, slices = category_only_embedding(
            features, mask_pool, self.categories
        )
        if raw_query.shape != self.embeddings.shape[1:]:
            raise MemoryBankError("category-only query dimension does not match memory")
        query = (raw_query - self.feature_mean) / (
            self.feature_std + CATEGORY_ONLY_STANDARDIZATION_EPSILON
        )
        distances = torch.linalg.vector_norm(self.embeddings - query[None, :], dim=1)
        nearest_index = int(distances.argmin().item())
        raw_score = float(distances[nearest_index].item())
        ratio = raw_score / self.loo_max_distance
        reference = self.embeddings[nearest_index]
        category_distances = {
            category: float(
                torch.linalg.vector_norm(query[region] - reference[region]).item()
            )
            for category, region in slices.items()
        }
        return CompositionEvidence(
            category_distances=category_distances,
            aggregate_raw_score=raw_score,
            threshold=self.loo_max_distance,
            adaptive_score=ratio,
            calibrated_score=ratio / (1.0 + ratio),
            anomaly=ratio > 1.0,
            nearest_reference_index=nearest_index,
            nearest_reference_image=self.normal_images[nearest_index],
            bank_path=self.source_path,
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": CATEGORY_ONLY_MEMORY_SCHEMA,
            "class_name": self.class_name,
            "categories": list(self.categories),
            "feature_mode": self.feature_mode,
            "backbone": "dinov2_vitl14_reg4",
            "layers": list(self.layers),
            "feature_grid": list(self.feature_grid),
            "input_size": self.input_size,
            "checkpoint_path": self.checkpoint_path,
            "normal_images": list(self.normal_images),
            "loo_max_distance": self.loo_max_distance,
            "matching": "standardized_full_vector_1nn_l2",
            "fusion_role": "independent_category_only_ratio",
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez(
                    handle,
                    metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
                    embeddings=self.embeddings.numpy(),
                    feature_mean=self.feature_mean.numpy(),
                    feature_std=self.feature_std.numpy(),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        self.source_path = str(destination)
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_class_name: str | None = None,
        expected_checkpoint_path: str | Path | None = None,
        expected_categories: Sequence[str] | None = None,
    ) -> "CategoryOnlyCompositionMemoryBank":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise MemoryBankError(f"category-only memory bank not found: {source}")
        try:
            with np.load(source, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                if metadata.get("schema") not in {
                    CATEGORY_ONLY_MEMORY_SCHEMA,
                    LEGACY_CATEGORY_ONLY_MEMORY_SCHEMA,
                }:
                    raise MemoryBankError(
                        "bank is not an isolated category-only composition memory"
                    )
                bank = cls(
                    class_name=metadata["class_name"],
                    categories=metadata["categories"],
                    embeddings=torch.from_numpy(
                        archive["embeddings"].astype(np.float32, copy=False)
                    ),
                    normal_images=metadata["normal_images"],
                    checkpoint_path=metadata["checkpoint_path"],
                    feature_grid=tuple(metadata["feature_grid"]),
                    feature_mean=torch.from_numpy(
                        archive["feature_mean"].astype(np.float32, copy=False)
                    ),
                    feature_std=torch.from_numpy(
                        archive["feature_std"].astype(np.float32, copy=False)
                    ),
                    loo_max_distance=metadata["loo_max_distance"],
                    source_path=str(source),
                )
        except MemoryBankError:
            raise
        except Exception as exc:
            raise MemoryBankError(
                f"cannot load category-only memory bank {source}: {exc}"
            ) from exc
        if expected_class_name is not None and bank.class_name != expected_class_name:
            raise MemoryBankError("category-only memory class does not match runtime")
        if expected_categories is not None and bank.categories != tuple(
            expected_categories
        ):
            raise MemoryBankError(
                "category-only memory categories do not match template"
            )
        if expected_checkpoint_path is not None:
            expected = str(Path(expected_checkpoint_path).expanduser().resolve())
            if bank.checkpoint_path != expected:
                raise MemoryBankError(
                    "category-only memory DINOv2 checkpoint does not match runtime"
                )
        return bank


class CategoryOnlyCompositionScorer:
    """Runtime adapter for category-only composition scoring."""

    def __init__(self, bank: CategoryOnlyCompositionMemoryBank) -> None:
        self.bank = bank

    def score_features(
        self,
        features: PatchFeatureBundle,
        mask_pool: Mapping[str, Sequence[MaskInstance]],
    ) -> CompositionEvidence:
        return self.bank.score_features(features, mask_pool)


class GlobalCategoryOnlyDualScorer:
    """Run global Patch and category-only Composition on one DINO forward pass."""

    def __init__(
        self,
        *,
        global_patch_scorer: Any,
        category_only_scorer: CategoryOnlyCompositionScorer,
        fusion_tau: float = DEFAULT_DUAL_FUSION_TAU,
    ) -> None:
        if not isfinite(float(fusion_tau)) or float(fusion_tau) <= 0.0:
            raise ValueError("dual fusion tau must be finite and positive")
        for name in ("extract_features", "score_features"):
            if not hasattr(global_patch_scorer, name):
                raise TypeError(f"global Patch scorer lacks {name}()")
        self.global_patch_scorer = global_patch_scorer
        self.category_only_scorer = category_only_scorer
        self.fusion_tau = float(fusion_tau)

    def score(
        self,
        image: Any,
        mask_pool: Mapping[str, Sequence[MaskInstance]],
    ) -> GlobalCategoryOnlyDualEvidence:
        features = self.global_patch_scorer.extract_features(image)
        return self.score_features(features, mask_pool)

    def score_features(
        self,
        features: PatchFeatureBundle,
        mask_pool: Mapping[str, Sequence[MaskInstance]],
    ) -> GlobalCategoryOnlyDualEvidence:
        patch = self.global_patch_scorer.score_features(features)
        composition = self.category_only_scorer.score_features(features, mask_pool)
        if patch.fusion_score is None:
            raise MemoryBankError(
                "global Patch memory has no leave-one-out fusion ratio"
            )
        aggregate = float(patch.fusion_score) + float(composition.adaptive_score)
        score = aggregate / (aggregate + self.fusion_tau)
        return GlobalCategoryOnlyDualEvidence(
            global_patch=patch,
            category_only=composition,
            aggregate_ratio=aggregate,
            calibrated_score=score,
            anomaly=score > 0.5,
            fusion_tau=self.fusion_tau,
        )
