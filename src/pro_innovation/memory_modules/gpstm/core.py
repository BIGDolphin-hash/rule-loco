"""GPSTM memory construction, persistence, and runtime scoring.

This module keeps category/background
texture statistics and removes every category DINO descriptor and category
distance term.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch

from ...adapters.patch_dinov2 import (
    PATCH_MATCHING_INPUT_SIZE,
    PATCH_MATCHING_LAYERS,
    PatchFeatureBundle,
)
from ...errors import MemoryBankError
from .texture import (
    BACKGROUND_BLOCK,
    DEFAULT_TOP_FRACTION,
    STANDARDIZATION_EPSILON,
    global_patch_texture_signature,
)
from ...models import MaskInstance
from .global_patch import GPSTMGlobalPatchBank

GPSTM_MEMORY_SCHEMA = "pro-innovation-global-patch-semantic-texture-memory-v1"
GPSTM_MODE = "global_patch_semantic_texture_only"


@dataclass(frozen=True)
class GlobalPatchSemanticTextureEvidence:
    """The sole texture-only score emitted by GPSTM."""

    aggregate_raw_score: float
    threshold: float
    adaptive_score: float
    calibrated_score: float
    anomaly: bool
    nearest_reference_index: int
    nearest_reference_image: str
    texture_distances: Mapping[str, float]
    texture_statistics: Mapping[str, tuple[float, float, float]]
    bank_path: str | None = None
    mode: str = GPSTM_MODE


def _texture_distances(
    query: torch.Tensor,
    memory: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one block-balanced distance and per-region diagnostics."""

    region_squared = ((memory - query[None, :, :]) ** 2).mean(dim=2)
    aggregate = torch.sqrt(region_squared.mean(dim=1))
    return aggregate, torch.sqrt(region_squared)


class GlobalPatchSemanticTextureMemoryBank:
    """One ordered memory containing category/background texture statistics."""

    def __init__(
        self,
        *,
        class_name: str,
        categories: Sequence[str],
        texture_memory: torch.Tensor,
        normal_images: Sequence[str | Path],
        checkpoint_path: str | Path,
        global_patch_bank_path: str | Path,
        feature_grid: tuple[int, int],
        texture_mean: torch.Tensor,
        texture_std: torch.Tensor,
        loo_max_distance: float,
        top_fraction: float = DEFAULT_TOP_FRACTION,
        source_path: str | None = None,
    ) -> None:
        self.class_name = str(class_name)
        self.categories = tuple(str(value) for value in categories)
        self.texture_memory = texture_memory.detach().float().cpu()
        self.normal_images = tuple(
            str(Path(path).expanduser().resolve()) for path in normal_images
        )
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.global_patch_bank_path = str(
            Path(global_patch_bank_path).expanduser().resolve()
        )
        self.feature_grid = (int(feature_grid[0]), int(feature_grid[1]))
        self.texture_mean = texture_mean.detach().float().cpu()
        self.texture_std = texture_std.detach().float().cpu()
        self.loo_max_distance = float(loo_max_distance)
        self.top_fraction = float(top_fraction)
        self.source_path = source_path
        self.layers = PATCH_MATCHING_LAYERS
        self.input_size = PATCH_MATCHING_INPUT_SIZE
        self._validate()

    def _validate(self) -> None:
        count = len(self.normal_images)
        category_count = len(self.categories)
        expected_shape = (count, category_count + 1, 3)
        if not self.class_name or not self.categories:
            raise MemoryBankError("GPSTM identity is empty")
        if len(set(self.categories)) != category_count:
            raise MemoryBankError("GPSTM categories must be unique")
        if self.texture_memory.shape != expected_shape:
            raise MemoryBankError("GPSTM texture memory has an invalid shape")
        if count < 2 or len(set(self.normal_images)) != count:
            raise MemoryBankError("GPSTM needs at least two distinct normal images")
        if self.texture_mean.shape != (category_count + 1, 3):
            raise MemoryBankError("GPSTM texture mean has an invalid shape")
        if self.texture_std.shape != self.texture_mean.shape:
            raise MemoryBankError("GPSTM texture std has an invalid shape")
        if not all(
            torch.isfinite(value).all()
            for value in (self.texture_memory, self.texture_mean, self.texture_std)
        ):
            raise MemoryBankError("GPSTM memory is non-finite")
        if (self.texture_std < 0).any():
            raise MemoryBankError("GPSTM texture std is negative")
        if not isfinite(self.loo_max_distance) or self.loo_max_distance <= 0.0:
            raise MemoryBankError("GPSTM LOO scale must be positive")
        if not isfinite(self.top_fraction) or not 0.0 < self.top_fraction <= 1.0:
            raise MemoryBankError("GPSTM top fraction must be in (0, 1]")

    @classmethod
    def build(
        cls,
        *,
        class_name: str,
        categories: Sequence[str],
        features: Sequence[PatchFeatureBundle],
        mask_pools: Sequence[Mapping[str, Sequence[MaskInstance]]],
        normal_images: Sequence[str | Path],
        global_patch_bank: GPSTMGlobalPatchBank,
        device: str,
        query_batch_size: int = 256,
        top_fraction: float = DEFAULT_TOP_FRACTION,
    ) -> "GlobalPatchSemanticTextureMemoryBank":
        if len(features) != len(mask_pools) or len(features) != len(normal_images):
            raise ValueError("GPSTM features, masks, and images must align")
        resolved_images = tuple(
            str(Path(path).expanduser().resolve()) for path in normal_images
        )
        if resolved_images != global_patch_bank.normal_images:
            raise MemoryBankError(
                "Global Patch references must align exactly with GPSTM images"
            )
        if global_patch_bank.source_path is None:
            raise MemoryBankError(
                "Global Patch bank must have a source path before GPSTM build"
            )
        rows = []
        for index, (bundle, pool) in enumerate(zip(features, mask_pools)):
            anomaly_map = global_patch_bank.anomaly_map(
                bundle,
                device=device,
                query_batch_size=query_batch_size,
                excluded_reference=index,
            )
            rows.append(
                global_patch_texture_signature(
                    anomaly_map,
                    pool,
                    categories,
                    feature_grid=tuple(bundle.feature_grid),
                    top_fraction=top_fraction,
                )[0]
            )
        return cls.build_from_signatures(
            class_name=class_name,
            categories=categories,
            texture_features=torch.stack(rows, dim=0),
            normal_images=normal_images,
            checkpoint_path=global_patch_bank.checkpoint_path,
            global_patch_bank_path=global_patch_bank.source_path,
            feature_grid=global_patch_bank.feature_grid,
            top_fraction=top_fraction,
        )

    @classmethod
    def build_from_signatures(
        cls,
        *,
        class_name: str,
        categories: Sequence[str],
        texture_features: torch.Tensor,
        normal_images: Sequence[str | Path],
        checkpoint_path: str | Path,
        global_patch_bank_path: str | Path,
        feature_grid: tuple[int, int],
        top_fraction: float = DEFAULT_TOP_FRACTION,
    ) -> "GlobalPatchSemanticTextureMemoryBank":
        raw = texture_features.detach().float().cpu()
        expected = (len(normal_images), len(tuple(categories)) + 1, 3)
        if raw.shape != expected or raw.shape[0] < 2:
            raise ValueError("GPSTM texture features have an invalid shape")
        mean = raw.mean(dim=0)
        std = raw.std(dim=0, unbiased=True)
        memory = (raw - mean) / (std + STANDARDIZATION_EPSILON)
        loo = []
        for index in range(memory.shape[0]):
            distances, _ = _texture_distances(memory[index], memory)
            distances[index] = float("inf")
            loo.append(distances.min())
        loo_max = float(torch.stack(loo).max().item())
        if not isfinite(loo_max) or loo_max <= 1e-12:
            raise MemoryBankError("GPSTM has zero LOO distance")
        return cls(
            class_name=class_name,
            categories=categories,
            texture_memory=memory,
            normal_images=normal_images,
            checkpoint_path=checkpoint_path,
            global_patch_bank_path=global_patch_bank_path,
            feature_grid=feature_grid,
            texture_mean=mean,
            texture_std=std,
            loo_max_distance=loo_max,
            top_fraction=top_fraction,
        )

    def score_features(
        self,
        features: PatchFeatureBundle,
        mask_pool: Mapping[str, Sequence[MaskInstance]],
        anomaly_map: torch.Tensor,
    ) -> GlobalPatchSemanticTextureEvidence:
        if tuple(features.feature_grid) != self.feature_grid:
            raise MemoryBankError("query feature grid does not match GPSTM memory")
        raw, payload = global_patch_texture_signature(
            anomaly_map,
            mask_pool,
            self.categories,
            feature_grid=self.feature_grid,
            top_fraction=self.top_fraction,
        )
        query = (raw - self.texture_mean) / (self.texture_std + STANDARDIZATION_EPSILON)
        distances, region_distances = _texture_distances(query, self.texture_memory)
        nearest = int(distances.argmin().item())
        raw_score = float(distances[nearest].item())
        ratio = raw_score / self.loo_max_distance
        labels = (*self.categories, BACKGROUND_BLOCK)
        return GlobalPatchSemanticTextureEvidence(
            aggregate_raw_score=raw_score,
            threshold=self.loo_max_distance,
            adaptive_score=ratio,
            calibrated_score=ratio / (ratio + 1.0),
            anomaly=ratio > 1.0,
            nearest_reference_index=nearest,
            nearest_reference_image=self.normal_images[nearest],
            texture_distances={
                label: float(region_distances[nearest, index].item())
                for index, label in enumerate(labels)
            },
            texture_statistics=payload,
            bank_path=self.source_path,
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": GPSTM_MEMORY_SCHEMA,
            "mode": GPSTM_MODE,
            "class_name": self.class_name,
            "categories": list(self.categories),
            "normal_images": list(self.normal_images),
            "checkpoint_path": self.checkpoint_path,
            "global_patch_bank_path": self.global_patch_bank_path,
            "feature_grid": list(self.feature_grid),
            "layers": list(self.layers),
            "input_size": self.input_size,
            "texture_statistics": ["mean", "top_fraction_mean", "max"],
            "top_fraction": self.top_fraction,
            "loo_max_distance": self.loo_max_distance,
            "matching": "ordered_semantic_texture_only_1nn",
            "category_descriptors_enabled": False,
            "score_count": 1,
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez(
                    handle,
                    metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
                    texture_memory=self.texture_memory.numpy(),
                    texture_mean=self.texture_mean.numpy(),
                    texture_std=self.texture_std.numpy(),
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
        expected_categories: Sequence[str] | None = None,
        expected_checkpoint_path: str | Path | None = None,
        expected_global_patch_bank_path: str | Path | None = None,
    ) -> "GlobalPatchSemanticTextureMemoryBank":
        source = Path(path).expanduser().resolve()
        try:
            with np.load(source, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                if metadata.get("schema") != GPSTM_MEMORY_SCHEMA:
                    raise MemoryBankError("unsupported GPSTM schema")
                bank = cls(
                    class_name=metadata["class_name"],
                    categories=metadata["categories"],
                    texture_memory=torch.from_numpy(archive["texture_memory"]),
                    normal_images=metadata["normal_images"],
                    checkpoint_path=metadata["checkpoint_path"],
                    global_patch_bank_path=metadata["global_patch_bank_path"],
                    feature_grid=tuple(metadata["feature_grid"]),
                    texture_mean=torch.from_numpy(archive["texture_mean"]),
                    texture_std=torch.from_numpy(archive["texture_std"]),
                    loo_max_distance=metadata["loo_max_distance"],
                    top_fraction=metadata["top_fraction"],
                    source_path=str(source),
                )
        except MemoryBankError:
            raise
        except Exception as exc:
            raise MemoryBankError(f"cannot load GPSTM memory {source}: {exc}") from exc
        if expected_class_name is not None and bank.class_name != expected_class_name:
            raise MemoryBankError("GPSTM class does not match")
        if expected_categories is not None and bank.categories != tuple(
            expected_categories
        ):
            raise MemoryBankError("GPSTM categories do not match")
        for actual, expected, label in (
            (bank.checkpoint_path, expected_checkpoint_path, "checkpoint"),
            (
                bank.global_patch_bank_path,
                expected_global_patch_bank_path,
                "Global Patch bank",
            ),
        ):
            if expected is not None and actual != str(
                Path(expected).expanduser().resolve()
            ):
                raise MemoryBankError(f"GPSTM {label} does not match")
        return bank


class GlobalPatchSemanticTextureScorer:
    """Share one DINO pass and consume only the Global Patch anomaly map."""

    def __init__(
        self,
        *,
        bank: GlobalPatchSemanticTextureMemoryBank,
        global_patch_scorer,
    ) -> None:
        if not hasattr(global_patch_scorer, "score_features_with_map"):
            raise TypeError("Global Patch scorer must expose score_features_with_map")
        self.bank = bank
        self.global_patch_scorer = global_patch_scorer

    def extract_features(self, image):
        return self.global_patch_scorer.extract_features(image)

    def score_features(
        self,
        features: PatchFeatureBundle,
        mask_pool: Mapping[str, Sequence[MaskInstance]],
    ) -> GlobalPatchSemanticTextureEvidence:
        _, anomaly_map = self.global_patch_scorer.score_features_with_map(features)
        return self.bank.score_features(features, mask_pool, anomaly_map)

    def score(
        self,
        image,
        mask_pool: Mapping[str, Sequence[MaskInstance]],
    ) -> GlobalPatchSemanticTextureEvidence:
        return self.score_features(self.extract_features(image), mask_pool)
