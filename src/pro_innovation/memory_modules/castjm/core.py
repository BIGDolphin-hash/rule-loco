"""CASTJM memory construction, persistence, and runtime scoring.

Global Patch supplies intermediate texture statistics only; it does not emit a
second image-level score in this module.  CASTJM owns every descriptor and
runtime dependency used by the joint memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import ceil, isfinite
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from ...adapters.patch_dinov2 import (
    PATCH_MATCHING_INPUT_SIZE,
    PATCH_MATCHING_LAYERS,
    PatchFeatureBundle,
)
from ...errors import MemoryBankError
from ...models import MaskInstance
from .global_patch import CASTJMGlobalPatchBank
from .pixel_texture import (
    GlobalPatchPixelLocalization,
    GlobalPatchPixelLocalizer,
)
from .semantic import category_only_embedding

CASTJM_MEMORY_SCHEMA = (
    "pro-innovation-castjm-joint-memory-v1"
)
CASTJM_MODE = "castjm_joint"
BACKGROUND_BLOCK = "__background__"
DEFAULT_TEXTURE_WEIGHT = 0.30
DEFAULT_TOP_FRACTION = 0.10
STANDARDIZATION_EPSILON = 1e-10


@dataclass(frozen=True)
class CASTJMEvidence:
    """The one and only score emitted by the joint ordered memory."""

    aggregate_raw_score: float
    threshold: float
    adaptive_score: float
    calibrated_score: float
    anomaly: bool
    nearest_reference_index: int
    nearest_reference_image: str
    category_distances: Mapping[str, float]
    texture_distances: Mapping[str, float]
    texture_statistics: Mapping[str, tuple[float, float, float]]
    bank_path: str | None = None
    mode: str = CASTJM_MODE


def _mask_on_feature_grid(
    instance: MaskInstance,
    feature_grid: tuple[int, int],
) -> torch.Tensor:
    mask = np.asarray(instance.mask, dtype=np.float32)
    tensor = torch.from_numpy(mask)[None, None]
    selected = (
        functional.interpolate(tensor, size=feature_grid, mode="nearest")[0, 0].reshape(
            -1
        )
        >= 0.5
    )
    if selected.any() or not mask.any():
        return selected
    return (
        functional.interpolate(tensor, size=feature_grid, mode="area")[0, 0].reshape(-1)
        > 0.0
    )


def _three_texture_statistics(
    values: torch.Tensor,
    *,
    top_fraction: float,
) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("Global Patch statistics require a flat anomaly map")
    if values.numel() == 0:
        return torch.zeros(3, dtype=torch.float32)
    count = max(1, min(values.numel(), ceil(values.numel() * top_fraction)))
    top_mean = torch.topk(values, k=count, largest=True).values.mean()
    return torch.stack((values.mean(), top_mean, values.max())).float().cpu()


def global_patch_texture_signature(
    anomaly_map: torch.Tensor,
    mask_pool: Mapping[str, Sequence[MaskInstance]],
    categories: Sequence[str],
    *,
    feature_grid: tuple[int, int],
    top_fraction: float = DEFAULT_TOP_FRACTION,
) -> tuple[torch.Tensor, dict[str, tuple[float, float, float]]]:
    """Pool mean, top-fraction mean, and max for each category and background."""

    ordered = tuple(str(category) for category in categories)
    if not ordered or len(set(ordered)) != len(ordered):
        raise MemoryBankError(
            "CASTJM categories must be unique and non-empty"
        )
    if not isfinite(float(top_fraction)) or not 0.0 < float(top_fraction) <= 1.0:
        raise ValueError("Global Patch top fraction must be in (0, 1]")
    values = anomaly_map.detach().float().cpu().reshape(-1)
    positions = int(feature_grid[0]) * int(feature_grid[1])
    if values.numel() != positions or not torch.isfinite(values).all():
        raise MemoryBankError("Global Patch anomaly map does not match feature grid")

    union = torch.zeros(positions, dtype=torch.bool)
    blocks: list[torch.Tensor] = []
    labels = (*ordered, BACKGROUND_BLOCK)
    for category in ordered:
        selected = torch.zeros(positions, dtype=torch.bool)
        for instance in mask_pool.get(category, ()):
            selected |= _mask_on_feature_grid(instance, feature_grid)
        union |= selected
        blocks.append(
            _three_texture_statistics(
                values[selected], top_fraction=float(top_fraction)
            )
        )
    blocks.append(
        _three_texture_statistics(values[~union], top_fraction=float(top_fraction))
    )
    signature = torch.stack(blocks, dim=0)
    payload = {
        label: tuple(float(value) for value in signature[index].tolist())
        for index, label in enumerate(labels)
    }
    return signature, payload


def _joint_distances(
    category_query: torch.Tensor,
    texture_query: torch.Tensor,
    category_memory: torch.Tensor,
    texture_memory: torch.Tensor,
    *,
    texture_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one weighted distance plus its diagnostic block distances."""

    category_squared = ((category_memory - category_query[None, :, :]) ** 2).mean(dim=2)
    texture_squared = ((texture_memory - texture_query[None, :, :]) ** 2).mean(dim=2)
    category_total = category_squared.mean(dim=1)
    texture_total = texture_squared.mean(dim=1)
    joint = torch.sqrt(
        (1.0 - texture_weight) * category_total + texture_weight * texture_total
    )
    return joint, torch.sqrt(category_squared), torch.sqrt(texture_squared)


class CASTJMMemoryBank:
    """One ordered joint memory containing category and Global Patch blocks."""

    def __init__(
        self,
        *,
        class_name: str,
        categories: Sequence[str],
        category_memory: torch.Tensor,
        texture_memory: torch.Tensor,
        normal_images: Sequence[str | Path],
        checkpoint_path: str | Path,
        global_patch_bank_path: str | Path,
        feature_grid: tuple[int, int],
        category_mean: torch.Tensor,
        category_std: torch.Tensor,
        texture_mean: torch.Tensor,
        texture_std: torch.Tensor,
        loo_max_distance: float,
        texture_weight: float = DEFAULT_TEXTURE_WEIGHT,
        top_fraction: float = DEFAULT_TOP_FRACTION,
        source_path: str | None = None,
    ) -> None:
        self.class_name = str(class_name)
        self.categories = tuple(str(value) for value in categories)
        self.category_memory = category_memory.detach().float().cpu()
        self.texture_memory = texture_memory.detach().float().cpu()
        self.normal_images = tuple(
            str(Path(path).expanduser().resolve()) for path in normal_images
        )
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.global_patch_bank_path = str(
            Path(global_patch_bank_path).expanduser().resolve()
        )
        self.feature_grid = (int(feature_grid[0]), int(feature_grid[1]))
        self.category_mean = category_mean.detach().float().cpu()
        self.category_std = category_std.detach().float().cpu()
        self.texture_mean = texture_mean.detach().float().cpu()
        self.texture_std = texture_std.detach().float().cpu()
        self.loo_max_distance = float(loo_max_distance)
        self.texture_weight = float(texture_weight)
        self.top_fraction = float(top_fraction)
        self.source_path = source_path
        self.layers = PATCH_MATCHING_LAYERS
        self.input_size = PATCH_MATCHING_INPUT_SIZE
        self._validate()

    def _validate(self) -> None:
        count = len(self.normal_images)
        category_count = len(self.categories)
        if not self.class_name or not self.categories:
            raise MemoryBankError("CASTJM identity is empty")
        if len(set(self.categories)) != category_count:
            raise MemoryBankError("CASTJM categories must be unique")
        if self.category_memory.ndim != 3 or self.category_memory.shape[:2] != (
            count,
            category_count,
        ):
            raise MemoryBankError("ordered category memory has an invalid shape")
        if self.texture_memory.shape != (count, category_count + 1, 3):
            raise MemoryBankError("ordered texture memory has an invalid shape")
        if count < 2 or len(set(self.normal_images)) != count:
            raise MemoryBankError("CASTJM memory needs distinct images")
        if not all(
            torch.isfinite(value).all()
            for value in (
                self.category_memory,
                self.texture_memory,
                self.category_mean,
                self.category_std,
                self.texture_mean,
                self.texture_std,
            )
        ):
            raise MemoryBankError("CASTJM memory is non-finite")
        category_dimension = int(self.category_memory.shape[2])
        if self.category_mean.shape != (category_count, category_dimension):
            raise MemoryBankError("ordered category mean has an invalid shape")
        if self.category_std.shape != self.category_mean.shape:
            raise MemoryBankError("ordered category std has an invalid shape")
        if self.texture_mean.shape != (category_count + 1, 3):
            raise MemoryBankError("ordered texture mean has an invalid shape")
        if self.texture_std.shape != self.texture_mean.shape:
            raise MemoryBankError("ordered texture std has an invalid shape")
        if (self.category_std < 0).any() or (self.texture_std < 0).any():
            raise MemoryBankError("CASTJM std is negative")
        if not isfinite(self.loo_max_distance) or self.loo_max_distance <= 0.0:
            raise MemoryBankError("CASTJM LOO scale must be positive")
        if not 0.0 <= self.texture_weight <= 1.0:
            raise MemoryBankError("texture weight must be in [0, 1]")
        if not 0.0 < self.top_fraction <= 1.0:
            raise MemoryBankError("top fraction must be in (0, 1]")

    @classmethod
    def build(
        cls,
        *,
        class_name: str,
        categories: Sequence[str],
        features: Sequence[PatchFeatureBundle],
        mask_pools: Sequence[Mapping[str, Sequence[MaskInstance]]],
        normal_images: Sequence[str | Path],
        global_patch_bank: CASTJMGlobalPatchBank,
        device: str,
        query_batch_size: int = 256,
        texture_weight: float = DEFAULT_TEXTURE_WEIGHT,
        top_fraction: float = DEFAULT_TOP_FRACTION,
    ) -> "CASTJMMemoryBank":
        if len(features) != len(mask_pools) or len(features) != len(normal_images):
            raise ValueError("ordered features, masks, and images must align")
        resolved_images = tuple(
            str(Path(path).expanduser().resolve()) for path in normal_images
        )
        if resolved_images != global_patch_bank.normal_images:
            raise MemoryBankError(
                "Global Patch references must align exactly with ordered images"
            )
        if global_patch_bank.source_path is None:
            raise MemoryBankError(
                "Global Patch bank must be loaded from or saved to a path "
                "before joint build"
            )
        category_rows = []
        texture_rows = []
        for index, (bundle, pool) in enumerate(zip(features, mask_pools)):
            vector, slices = category_only_embedding(bundle, pool, categories)
            category_rows.append(
                torch.stack(
                    [vector[slices[category]] for category in categories], dim=0
                )
            )
            anomaly_map = global_patch_bank.anomaly_map(
                bundle,
                device=device,
                query_batch_size=query_batch_size,
                excluded_reference=index,
            )
            texture_rows.append(
                global_patch_texture_signature(
                    anomaly_map,
                    pool,
                    categories,
                    feature_grid=tuple(bundle.feature_grid),
                    top_fraction=top_fraction,
                )[0]
            )
        return cls.build_from_components(
            class_name=class_name,
            categories=categories,
            category_features=torch.stack(category_rows, dim=0),
            texture_features=torch.stack(texture_rows, dim=0),
            normal_images=normal_images,
            checkpoint_path=global_patch_bank.checkpoint_path,
            global_patch_bank_path=global_patch_bank.source_path,
            feature_grid=global_patch_bank.feature_grid,
            texture_weight=texture_weight,
            top_fraction=top_fraction,
        )

    @classmethod
    def build_from_components(
        cls,
        *,
        class_name: str,
        categories: Sequence[str],
        category_features: torch.Tensor,
        texture_features: torch.Tensor,
        normal_images: Sequence[str | Path],
        checkpoint_path: str | Path,
        global_patch_bank_path: str | Path,
        feature_grid: tuple[int, int],
        texture_weight: float = DEFAULT_TEXTURE_WEIGHT,
        top_fraction: float = DEFAULT_TOP_FRACTION,
    ) -> "CASTJMMemoryBank":
        category_raw = category_features.detach().float().cpu()
        texture_raw = texture_features.detach().float().cpu()
        if category_raw.ndim != 3 or texture_raw.ndim != 3:
            raise ValueError("ordered joint components must be three-dimensional")
        if category_raw.shape[0] != len(normal_images) or category_raw.shape[0] < 2:
            raise ValueError("ordered joint components and images must align")
        category_mean = category_raw.mean(dim=0)
        category_std = category_raw.std(dim=0, unbiased=True)
        texture_mean = texture_raw.mean(dim=0)
        texture_std = texture_raw.std(dim=0, unbiased=True)
        category_memory = (category_raw - category_mean) / (
            category_std + STANDARDIZATION_EPSILON
        )
        texture_memory = (texture_raw - texture_mean) / (
            texture_std + STANDARDIZATION_EPSILON
        )
        loo = []
        for index in range(category_memory.shape[0]):
            distances, _, _ = _joint_distances(
                category_memory[index],
                texture_memory[index],
                category_memory,
                texture_memory,
                texture_weight=float(texture_weight),
            )
            distances[index] = float("inf")
            loo.append(distances.min())
        loo_max = float(torch.stack(loo).max().item())
        if not isfinite(loo_max) or loo_max <= 1e-12:
            raise MemoryBankError("ordered joint memory has zero LOO distance")
        return cls(
            class_name=class_name,
            categories=categories,
            category_memory=category_memory,
            texture_memory=texture_memory,
            normal_images=normal_images,
            checkpoint_path=checkpoint_path,
            global_patch_bank_path=global_patch_bank_path,
            feature_grid=feature_grid,
            category_mean=category_mean,
            category_std=category_std,
            texture_mean=texture_mean,
            texture_std=texture_std,
            loo_max_distance=loo_max,
            texture_weight=texture_weight,
            top_fraction=top_fraction,
        )

    def score_features(
        self,
        features: PatchFeatureBundle,
        mask_pool: Mapping[str, Sequence[MaskInstance]],
        anomaly_map: torch.Tensor,
    ) -> CASTJMEvidence:
        if tuple(features.feature_grid) != self.feature_grid:
            raise MemoryBankError(
                "query feature grid does not match CASTJM memory"
            )
        vector, slices = category_only_embedding(features, mask_pool, self.categories)
        category_raw = torch.stack(
            [vector[slices[category]] for category in self.categories], dim=0
        )
        texture_raw, texture_payload = global_patch_texture_signature(
            anomaly_map,
            mask_pool,
            self.categories,
            feature_grid=self.feature_grid,
            top_fraction=self.top_fraction,
        )
        category_query = (category_raw - self.category_mean) / (
            self.category_std + STANDARDIZATION_EPSILON
        )
        texture_query = (texture_raw - self.texture_mean) / (
            self.texture_std + STANDARDIZATION_EPSILON
        )
        distances, category_distances, texture_distances = _joint_distances(
            category_query,
            texture_query,
            self.category_memory,
            self.texture_memory,
            texture_weight=self.texture_weight,
        )
        nearest = int(distances.argmin().item())
        raw_score = float(distances[nearest].item())
        ratio = raw_score / self.loo_max_distance
        texture_labels = (*self.categories, BACKGROUND_BLOCK)
        return CASTJMEvidence(
            aggregate_raw_score=raw_score,
            threshold=self.loo_max_distance,
            adaptive_score=ratio,
            calibrated_score=ratio / (ratio + 1.0),
            anomaly=ratio > 1.0,
            nearest_reference_index=nearest,
            nearest_reference_image=self.normal_images[nearest],
            category_distances={
                category: float(category_distances[nearest, index].item())
                for index, category in enumerate(self.categories)
            },
            texture_distances={
                label: float(texture_distances[nearest, index].item())
                for index, label in enumerate(texture_labels)
            },
            texture_statistics=texture_payload,
            bank_path=self.source_path,
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": CASTJM_MEMORY_SCHEMA,
            "mode": CASTJM_MODE,
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
            "texture_weight": self.texture_weight,
            "loo_max_distance": self.loo_max_distance,
            "matching": "castjm_category_texture_1nn",
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
                    category_memory=self.category_memory.numpy(),
                    texture_memory=self.texture_memory.numpy(),
                    category_mean=self.category_mean.numpy(),
                    category_std=self.category_std.numpy(),
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
    ) -> "CASTJMMemoryBank":
        source = Path(path).expanduser().resolve()
        try:
            with np.load(source, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                if metadata.get("schema") != CASTJM_MEMORY_SCHEMA:
                    raise MemoryBankError("unsupported CASTJM schema")
                bank = cls(
                    class_name=metadata["class_name"],
                    categories=metadata["categories"],
                    category_memory=torch.from_numpy(archive["category_memory"]),
                    texture_memory=torch.from_numpy(archive["texture_memory"]),
                    normal_images=metadata["normal_images"],
                    checkpoint_path=metadata["checkpoint_path"],
                    global_patch_bank_path=metadata["global_patch_bank_path"],
                    feature_grid=tuple(metadata["feature_grid"]),
                    category_mean=torch.from_numpy(archive["category_mean"]),
                    category_std=torch.from_numpy(archive["category_std"]),
                    texture_mean=torch.from_numpy(archive["texture_mean"]),
                    texture_std=torch.from_numpy(archive["texture_std"]),
                    loo_max_distance=metadata["loo_max_distance"],
                    texture_weight=metadata["texture_weight"],
                    top_fraction=metadata["top_fraction"],
                    source_path=str(source),
                )
        except MemoryBankError:
            raise
        except Exception as exc:
            raise MemoryBankError(
                f"cannot load CASTJM memory {source}: {exc}"
            ) from exc
        if expected_class_name is not None and bank.class_name != expected_class_name:
            raise MemoryBankError("CASTJM class does not match")
        if expected_categories is not None and bank.categories != tuple(
            expected_categories
        ):
            raise MemoryBankError("CASTJM categories do not match")
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
                raise MemoryBankError(f"CASTJM {label} does not match")
        return bank


class CASTJMScorer:
    """Share one DINO pass and consume only the Global Patch anomaly map."""

    def __init__(
        self,
        *,
        bank: CASTJMMemoryBank,
        global_patch_scorer,
        pixel_localizer: GlobalPatchPixelLocalizer | None = None,
    ) -> None:
        if not hasattr(global_patch_scorer, "score_features_with_map"):
            raise TypeError("Global Patch scorer must expose score_features_with_map")
        self.bank = bank
        self.global_patch_scorer = global_patch_scorer
        self.pixel_localizer = pixel_localizer or GlobalPatchPixelLocalizer()

    def extract_features(self, image):
        return self.global_patch_scorer.extract_features(image)

    def score_features(
        self,
        features: PatchFeatureBundle,
        mask_pool: Mapping[str, Sequence[MaskInstance]],
    ) -> CASTJMEvidence:
        _, anomaly_map = self.global_patch_scorer.score_features_with_map(features)
        return self.bank.score_features(features, mask_pool, anomaly_map)

    def localize_features(
        self,
        features: PatchFeatureBundle,
        *,
        output_size: tuple[int, int],
    ) -> GlobalPatchPixelLocalization:
        """Run the pure-texture Pixel path without semantic or joint scoring."""

        _, anomaly_map = self.global_patch_scorer.score_features_with_map(features)
        return self.pixel_localizer.localize(
            anomaly_map,
            feature_grid=(
                int(features.feature_grid[0]),
                int(features.feature_grid[1]),
            ),
            output_size=output_size,
        )

    def score_features_with_pixel_map(
        self,
        features: PatchFeatureBundle,
        mask_pool: Mapping[str, Sequence[MaskInstance]],
        *,
        output_size: tuple[int, int],
    ) -> tuple[CASTJMEvidence, GlobalPatchPixelLocalization]:
        """Emit image-level CASTJM evidence and Pixel localization from one map."""

        _, anomaly_map = self.global_patch_scorer.score_features_with_map(features)
        evidence = self.bank.score_features(features, mask_pool, anomaly_map)
        localization = self.pixel_localizer.localize(
            anomaly_map,
            feature_grid=(
                int(features.feature_grid[0]),
                int(features.feature_grid[1]),
            ),
            output_size=output_size,
        )
        return evidence, localization

    def score(
        self,
        image,
        mask_pool: Mapping[str, Sequence[MaskInstance]],
    ) -> CASTJMEvidence:
        return self.score_features(self.extract_features(image), mask_pool)

    def localize(
        self,
        image,
        *,
        output_size: tuple[int, int],
    ) -> GlobalPatchPixelLocalization:
        """Extract features once and run only the pure-texture Pixel path."""

        return self.localize_features(
            self.extract_features(image),
            output_size=output_size,
        )

    def score_with_pixel_map(
        self,
        image,
        mask_pool: Mapping[str, Sequence[MaskInstance]],
        *,
        output_size: tuple[int, int],
    ) -> tuple[CASTJMEvidence, GlobalPatchPixelLocalization]:
        """Extract features once for joint image scoring and Pixel localization."""

        return self.score_features_with_pixel_map(
            self.extract_features(image),
            mask_pool,
            output_size=output_size,
        )
