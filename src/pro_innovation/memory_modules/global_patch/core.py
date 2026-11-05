"""Global Patch memory construction, persistence, and runtime scoring."""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence
import zipfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional

from ...adapters.patch_dinov2 import (
    DEFAULT_PATCH_DINOV2_CHECKPOINT,
    PATCH_MATCHING_INPUT_SIZE,
    PATCH_MATCHING_LAYERS,
    DinoV2PatchFeatureExtractor,
    PatchFeatureBundle,
)
from ...errors import MemoryBankError
from ...image_validation import filter_decodable_images
from ...models import PatchEvidence

PATCH_MEMORY_SCHEMA = "pro-innovation-logsad-patch-reg4-full-coreset-memory-v1"
PATCH_MEMORY_IMAGE_SHARD_SCHEMA = (
    "pro-innovation-logsad-patch-reg4-full-coreset-image-shard-v1"
)
PATCH_MEMORY_PER_IMAGE_STORAGE = "per_image_npz_sha256_v1"
PATCH_FEATURE_GLOBAL = "global"
PATCH_FEATURE_MODES = (PATCH_FEATURE_GLOBAL,)
MINIMUM_NORMAL_IMAGE_COUNT = 1
DEFAULT_CORESET_RATIO = 0.05
DEFAULT_CORESET_SEED = 0
DEFAULT_PATCH_LAYER_FUSION = "logsad_mean_map"
LOGSAD_CALIBRATION_MODE = "normal_reference_max_ratio_bounded"


def _single_image_self_exclusion_score(
    features: PatchFeatureBundle,
    *,
    device: str,
    query_batch_size: int,
) -> float:
    """Estimate a one-shot normal score without using an identical patch match."""

    positions = features.feature_grid[0] * features.feature_grid[1]
    if positions < 2:
        raise MemoryBankError("one-shot patch calibration needs at least two patches")
    layer_maps: list[torch.Tensor] = []
    with torch.inference_mode():
        for layer in PATCH_MATCHING_LAYERS:
            values = functional.normalize(
                features.layer_features[layer].detach().float(), dim=-1, eps=1e-12
            ).to(device)
            chunks: list[torch.Tensor] = []
            for start in range(0, positions, query_batch_size):
                stop = min(start + query_batch_size, positions)
                similarity = values[start:stop] @ values.T
                row_indices = torch.arange(stop - start, device=device)
                similarity[row_indices, torch.arange(start, stop, device=device)] = -1.0
                chunks.append(
                    (1.0 - similarity.max(dim=1).values).clamp(0.0, 2.0).cpu()
                )
            layer_maps.append(torch.cat(chunks))
    return float(torch.stack(layer_maps, dim=0).mean(dim=0).max().item())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_npz_verified(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    attempts: int = 3,
) -> str:
    """Atomically write an uncompressed NPZ and verify every member."""

    last_error: Exception | None = None
    for _ in range(attempts):
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            with zipfile.ZipFile(temporary_name) as archive:
                bad_member = archive.testzip()
            if bad_member is not None:
                raise MemoryBankError(f"NPZ CRC validation failed: {bad_member}")
            with np.load(temporary_name, allow_pickle=False) as archive:
                if set(archive.files) != set(arrays):
                    raise MemoryBankError(f"NPZ member validation failed: {path.name}")
                for key, expected in arrays.items():
                    loaded = archive[key]
                    if loaded.shape != expected.shape or loaded.dtype != expected.dtype:
                        raise MemoryBankError(
                            f"NPZ array validation failed: {path.name}:{key}"
                        )
            os.replace(temporary_name, path)
            first_digest = _sha256_file(path)
            second_digest = _sha256_file(path)
            if first_digest != second_digest:
                raise MemoryBankError(f"NPZ checksum is unstable: {path.name}")
            return first_digest
        except Exception as exc:
            last_error = exc
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    raise MemoryBankError(f"cannot write a verified NPZ {path.name}: {last_error}")


def patch_memory_run_tag(
    coreset_ratio: float = DEFAULT_CORESET_RATIO,
) -> str:
    """Return a bank tag that records the exact Patch coreset percentage."""

    ratio = float(coreset_ratio)
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("patch coreset ratio must be in (0, 1]")
    percentage = ratio * 100.0
    rounded = round(percentage)
    if abs(percentage - rounded) <= 1e-9:
        ratio_tag = f"{rounded:02d}"
    else:
        ratio_tag = f"{percentage:.10f}".rstrip("0").rstrip(".").replace(".", "p")
    layer_tag = "".join(f"b{layer}" for layer in PATCH_MATCHING_LAYERS)
    return f"dino_full_coreset{ratio_tag}_{layer_tag}"


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    return functional.normalize(tensor.detach().float(), dim=-1, eps=1e-12)


def _image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            converted = opened.convert("RGB")
            converted.load()
            return converted.copy()
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise MemoryBankError(f"expected an RGB image, got shape {array.shape}")
    return Image.fromarray(array[..., :3].astype(np.uint8), mode="RGB")


def _coreset_indices(
    positions: int,
    *,
    ratio: float,
    seed: int,
    image_index: int,
) -> torch.Tensor:
    """Select a deterministic, image-stratified random patch coreset."""

    if positions <= 0:
        raise ValueError("patch position count must be positive")
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("patch coreset ratio must be in (0, 1]")
    if int(seed) < 0:
        raise ValueError("patch coreset seed must be non-negative")
    count = max(1, int(math.ceil(positions * float(ratio))))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(image_index))
    return torch.randperm(positions, generator=generator)[:count]


def _build_coreset(
    features: Sequence[PatchFeatureBundle],
    *,
    ratio: float,
    seed: int,
    image_index_offset: int = 0,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Compress full-image patches while retaining every source image."""

    layer_chunks: dict[int, list[torch.Tensor]] = {
        layer: [] for layer in PATCH_MATCHING_LAYERS
    }
    sources: list[torch.Tensor] = []
    for local_index, bundle in enumerate(features):
        image_index = int(image_index_offset) + local_index
        if tuple(sorted(bundle.layer_features)) != PATCH_MATCHING_LAYERS:
            raise ValueError("full-image patch features are missing layers")
        positions = bundle.feature_grid[0] * bundle.feature_grid[1]
        indices = _coreset_indices(
            positions,
            ratio=ratio,
            seed=seed,
            image_index=image_index,
        )
        for layer in PATCH_MATCHING_LAYERS:
            values = bundle.layer_features[layer]
            if values.ndim != 2 or values.shape[0] != positions:
                raise ValueError(f"invalid Block {layer} full-image feature shape")
            layer_chunks[layer].append(_normalize(values[indices]).cpu())
        sources.append(torch.full((indices.numel(),), image_index, dtype=torch.long))
    if not sources:
        raise ValueError("cannot build a patch coreset without normal images")
    return (
        {layer: torch.cat(chunks, dim=0) for layer, chunks in layer_chunks.items()},
        torch.cat(sources, dim=0),
    )


class PatchMemoryBank:
    """Full-normal Patch coreset with leave-one-image-out calibration."""

    def __init__(
        self,
        *,
        class_name: str,
        feature_memory: Mapping[int, torch.Tensor],
        memory_source_indices: torch.Tensor,
        feature_grid: tuple[int, int],
        checkpoint_path: str | Path,
        normal_images: Sequence[str | Path],
        score_mean: float,
        score_unbiased_std: float,
        calibration_images: Sequence[str | Path] = (),
        calibration_scores: Sequence[float] = (),
        input_size: int = PATCH_MATCHING_INPUT_SIZE,
        coreset_ratio: float = DEFAULT_CORESET_RATIO,
        coreset_seed: int = DEFAULT_CORESET_SEED,
        source_path: str | None = None,
        features_are_normalized: bool = False,
    ) -> None:
        prepare = (
            (lambda values: values.detach().float().cpu())
            if features_are_normalized
            else (lambda values: _normalize(values).cpu())
        )
        self.class_name = str(class_name)
        self.feature_memory = {
            int(layer): prepare(values) for layer, values in feature_memory.items()
        }
        self.memory_source_indices = memory_source_indices.detach().long().cpu()
        self.feature_grid = (int(feature_grid[0]), int(feature_grid[1]))
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.normal_images = tuple(
            str(Path(path).expanduser().resolve()) for path in normal_images
        )
        self.score_mean = float(score_mean)
        self.score_unbiased_std = float(score_unbiased_std)
        self.calibration_images = tuple(
            str(Path(path).expanduser().resolve()) for path in calibration_images
        )
        self.calibration_scores = tuple(float(value) for value in calibration_scores)
        self.input_size = int(input_size)
        self.coreset_ratio = float(coreset_ratio)
        self.coreset_seed = int(coreset_seed)
        self.source_path = source_path
        self.calibration_mode = LOGSAD_CALIBRATION_MODE
        self._validate(require_calibration=bool(self.calibration_images))

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted(self.feature_memory))

    @property
    def threshold(self) -> float:
        """Raw Global Patch score mapped to the bounded score 0.5."""

        scale = self.psad_scale
        return scale if scale is not None else self.score_mean

    @property
    def psad_scale(self) -> float | None:
        """Maximum normal leave-one-image-out score used by PSAD scaling."""

        if not self.calibration_scores:
            return None
        value = max(self.calibration_scores)
        return value if np.isfinite(value) and value > 1e-12 else None

    def _validate(self, *, require_calibration: bool) -> None:
        if not self.class_name:
            raise MemoryBankError("patch memory class name is empty")
        if self.layers != PATCH_MATCHING_LAYERS:
            raise MemoryBankError(
                f"patch memory layers must be {PATCH_MATCHING_LAYERS}"
            )
        if len(self.normal_images) < MINIMUM_NORMAL_IMAGE_COUNT:
            raise MemoryBankError("patch memory requires at least one normal image")
        if len(set(self.normal_images)) != len(self.normal_images):
            raise MemoryBankError("patch memory normal images must be distinct")
        if not 0.0 < self.coreset_ratio <= 1.0:
            raise MemoryBankError("patch coreset ratio must be in (0, 1]")
        if self.coreset_seed < 0:
            raise MemoryBankError("patch coreset seed must be non-negative")
        if self.input_size != PATCH_MATCHING_INPUT_SIZE:
            raise MemoryBankError(
                f"patch memory input size must be {PATCH_MATCHING_INPUT_SIZE}"
            )
        if self.feature_grid[0] <= 0 or self.feature_grid[1] <= 0:
            raise MemoryBankError("patch memory feature grid must be positive")
        if not np.isfinite(self.score_mean) or not 0.0 <= self.score_mean <= 2.0:
            raise MemoryBankError("LogSAD patch score mean must be in [0, 2]")
        if not np.isfinite(self.score_unbiased_std) or self.score_unbiased_std <= 0.0:
            raise MemoryBankError(
                "LogSAD patch score standard deviation must be positive"
            )
        if len(self.calibration_images) != len(self.calibration_scores):
            raise MemoryBankError(
                "patch calibration images and scores must have equal length"
            )
        if len(set(self.calibration_images)) != len(self.calibration_images):
            raise MemoryBankError("patch calibration images must be distinct")
        if any(
            not np.isfinite(value) or not 0.0 <= value <= 2.0
            for value in self.calibration_scores
        ):
            raise MemoryBankError("patch calibration scores must be in [0, 2]")
        minimum_calibration_count = 1 if len(self.normal_images) == 1 else 2
        if require_calibration and len(self.calibration_scores) < minimum_calibration_count:
            raise MemoryBankError(
                "patch calibration does not contain enough normal scores"
            )
        if require_calibration and self.calibration_images != self.normal_images:
            raise MemoryBankError(
                "full-normal Patch calibration must score every normal memory image"
            )

        dimensions = set()
        for layer, values in self.feature_memory.items():
            if values.ndim != 2 or values.shape[0] <= 0:
                raise MemoryBankError(f"invalid Block {layer} coreset shape")
            if not torch.isfinite(values).all():
                raise MemoryBankError(f"non-finite Block {layer} coreset")
            if values.shape[0] != self.memory_source_indices.numel():
                raise MemoryBankError("patch coreset layers and source indices differ")
            dimensions.add(int(values.shape[1]))
        if len(dimensions) != 1:
            raise MemoryBankError("patch memory layers use different dimensions")
        if self.memory_source_indices.ndim != 1:
            raise MemoryBankError(
                "patch coreset source indices must be one-dimensional"
            )
        if self.memory_source_indices.numel() == 0:
            raise MemoryBankError("patch coreset is empty")
        if int(self.memory_source_indices.min()) < 0 or int(
            self.memory_source_indices.max()
        ) >= len(self.normal_images):
            raise MemoryBankError("patch coreset source index is out of range")
        represented = set(int(value) for value in self.memory_source_indices.tolist())
        if represented != set(range(len(self.normal_images))):
            raise MemoryBankError("every normal image must contribute to the coreset")

    @classmethod
    def build(
        cls,
        *,
        class_name: str,
        features: Sequence[PatchFeatureBundle],
        normal_images: Sequence[str | Path],
        checkpoint_path: str | Path,
        device: str,
        query_batch_size: int = 256,
        coreset_ratio: float = DEFAULT_CORESET_RATIO,
        coreset_seed: int = DEFAULT_CORESET_SEED,
    ) -> "PatchMemoryBank":
        if len(features) != len(normal_images):
            raise ValueError("patch features and normal images must align")
        if len(features) < MINIMUM_NORMAL_IMAGE_COUNT:
            raise ValueError("patch memory requires at least one normal image")
        if query_batch_size <= 0:
            raise ValueError("patch query batch size must be positive")
        grids = {tuple(item.feature_grid) for item in features}
        if len(grids) != 1:
            raise ValueError("normal patch features use different feature grids")
        memory, source_indices = _build_coreset(
            features,
            ratio=coreset_ratio,
            seed=coreset_seed,
        )
        provisional = cls(
            class_name=class_name,
            feature_memory=memory,
            memory_source_indices=source_indices,
            feature_grid=next(iter(grids)),
            checkpoint_path=checkpoint_path,
            normal_images=normal_images,
            score_mean=0.0,
            score_unbiased_std=1.0,
            coreset_ratio=coreset_ratio,
            coreset_seed=coreset_seed,
        )
        if len(features) == 1:
            if not np.isclose(float(coreset_ratio), 1.0):
                raise ValueError("one-shot patch memory requires coreset ratio 1.0")
            raw_scores = (
                _single_image_self_exclusion_score(
                    features[0], device=device, query_batch_size=query_batch_size
                ),
            )
        else:
            raw_scores = tuple(
                provisional.raw_score(
                    item,
                    device=device,
                    query_batch_size=query_batch_size,
                    excluded_reference=index,
                )[1]
                for index, item in enumerate(features)
            )
        score_mean = float(np.mean(np.asarray(raw_scores, dtype=np.float64)))
        score_unbiased_std = (
            max(abs(score_mean) * 0.1, 1e-6)
            if len(raw_scores) == 1
            else float(np.std(np.asarray(raw_scores, dtype=np.float64), ddof=1))
        )
        if not np.isfinite(score_unbiased_std) or score_unbiased_std <= 1e-12:
            raise MemoryBankError(
                "leave-one-out normal scores have zero variance; LogSAD calibration "
                "cannot standardize them"
            )
        bank = cls(
            class_name=class_name,
            feature_memory=memory,
            memory_source_indices=source_indices,
            feature_grid=next(iter(grids)),
            checkpoint_path=checkpoint_path,
            normal_images=normal_images,
            score_mean=score_mean,
            score_unbiased_std=score_unbiased_std,
            calibration_images=normal_images,
            calibration_scores=raw_scores,
            coreset_ratio=coreset_ratio,
            coreset_seed=coreset_seed,
        )
        bank._validate(require_calibration=True)
        return bank

    def _validate_query(self, features: PatchFeatureBundle) -> None:
        if tuple(features.feature_grid) != self.feature_grid:
            raise MemoryBankError("patch query grid does not match memory grid")
        if tuple(sorted(features.layer_features)) != self.layers:
            raise MemoryBankError("patch query layers do not match memory layers")
        positions = self.feature_grid[0] * self.feature_grid[1]
        dimension = next(iter(self.feature_memory.values())).shape[-1]
        for layer in self.layers:
            values = features.layer_features[layer]
            if values.shape != (positions, dimension):
                raise MemoryBankError(
                    f"patch query Block {layer} shape does not match memory"
                )
            if not torch.isfinite(values).all():
                raise MemoryBankError(f"patch query Block {layer} is non-finite")

    def _raw_maps_against(
        self,
        features: PatchFeatureBundle,
        memories: Mapping[int, torch.Tensor],
        *,
        device: str,
        query_batch_size: int,
        excluded_reference: int | None = None,
    ) -> tuple[dict[str, float], torch.Tensor]:
        """Return per-layer maxima and the fused LogSAD anomaly map."""

        self._validate_query(features)
        if query_batch_size <= 0:
            raise ValueError("patch query batch size must be positive")
        if excluded_reference is not None and not (
            0 <= int(excluded_reference) < len(self.normal_images)
        ):
            raise ValueError("excluded patch reference index is out of range")

        layer_scores: dict[str, float] = {}
        anomaly_maps: list[torch.Tensor] = []
        with torch.inference_mode():
            for layer in self.layers:
                references = memories[layer]
                if excluded_reference is not None:
                    keep = self.memory_source_indices != int(excluded_reference)
                    references = references[keep]
                if references.shape[0] == 0:
                    raise MemoryBankError(
                        "leave-one-out patch coreset has no remaining references"
                    )
                references = references.to(device)
                query = _normalize(features.layer_features[layer]).to(device)
                distances = []
                for start in range(0, query.shape[0], query_batch_size):
                    query_chunk = query[start : start + query_batch_size]
                    similarity = query_chunk @ references.T
                    distances.append(
                        (1.0 - similarity.max(dim=1).values).clamp(0.0, 2.0)
                    )
                anomaly_map = torch.cat(distances)
                anomaly_maps.append(anomaly_map)
                layer_scores[str(layer)] = float(anomaly_map.max().cpu().item())

        aggregate_map = torch.stack(anomaly_maps, dim=0).mean(dim=0)
        return layer_scores, aggregate_map.cpu()

    def _raw_scores_against(
        self,
        features: PatchFeatureBundle,
        memories: Mapping[int, torch.Tensor],
        *,
        device: str,
        query_batch_size: int,
        excluded_reference: int | None = None,
    ) -> tuple[dict[str, float], float]:
        """Implement LogSAD: global NN, layer-map mean, then spatial max."""

        layer_scores, aggregate_map = self._raw_maps_against(
            features,
            memories,
            device=device,
            query_batch_size=query_batch_size,
            excluded_reference=excluded_reference,
        )
        return layer_scores, float(aggregate_map.max().item())

    def anomaly_map(
        self,
        features: PatchFeatureBundle,
        *,
        device: str,
        query_batch_size: int = 256,
        excluded_reference: int | None = None,
    ) -> torch.Tensor:
        """Return the four-layer mean Patch anomaly map for semantic pooling."""

        return self._raw_maps_against(
            features,
            self.feature_memory,
            device=device,
            query_batch_size=query_batch_size,
            excluded_reference=excluded_reference,
        )[1]

    def raw_score(
        self,
        features: PatchFeatureBundle,
        *,
        device: str,
        query_batch_size: int = 256,
        excluded_reference: int | None = None,
    ) -> tuple[dict[str, float], float]:
        """Return per-layer maxima and LogSAD's fused raw structural score."""

        return self._raw_scores_against(
            features,
            self.feature_memory,
            device=device,
            query_batch_size=query_batch_size,
            excluded_reference=excluded_reference,
        )

    def _evidence_from_raw(
        self,
        layer_scores: Mapping[str, float],
        raw_score: float,
    ) -> PatchEvidence:
        """Map raw Global Patch evidence without saturating its ranking."""

        layer_scores = dict(layer_scores)
        psad_scale = self.psad_scale
        if psad_scale is None:
            raise MemoryBankError("Global Patch normal-reference scale is unavailable")
        psad_adaptive_score = raw_score / psad_scale
        calibrated_score = psad_adaptive_score / (psad_adaptive_score + 1.0)
        standardized: float | None = (
            None
            if len(self.normal_images) == 1
            else (raw_score - self.score_mean) / self.score_unbiased_std
        )
        winning_layer = max(
            PATCH_MATCHING_LAYERS,
            key=lambda layer: layer_scores[str(layer)],
        )
        full_payload = {
            "layer_scores": dict(layer_scores),
            "layer_fusion": DEFAULT_PATCH_LAYER_FUSION,
            "raw_score": raw_score,
            "score_mean": self.score_mean,
            "score_unbiased_std": self.score_unbiased_std,
            "standardized_score": standardized,
            "calibrated_score": calibrated_score,
            "psad_train_loo_max": psad_scale,
            "psad_adaptive_score": psad_adaptive_score,
            "score_mapping": "ratio_over_one_plus_ratio",
            "matching": "global_cosine_nearest_neighbor",
            "layer_aggregation": "mean_anomaly_map",
            "image_aggregation": "spatial_max",
        }
        return PatchEvidence(
            layer_scores=layer_scores,
            aggregate_raw_score=raw_score,
            threshold=psad_scale,
            calibrated_score=calibrated_score,
            anomaly=psad_adaptive_score > 1.0,
            fusion_score=psad_adaptive_score,
            fusion_threshold=1.0,
            bank_path=self.source_path,
            layer_fusion=DEFAULT_PATCH_LAYER_FUSION,
            winning_layer=winning_layer,
            full_image_score=full_payload,
        )

    def score_with_map(
        self,
        features: PatchFeatureBundle,
        *,
        device: str,
        query_batch_size: int = 256,
    ) -> tuple[PatchEvidence, torch.Tensor]:
        """Score once and retain the aggregate map for composition conditioning."""

        layer_scores, aggregate_map = self._raw_maps_against(
            features,
            self.feature_memory,
            device=device,
            query_batch_size=query_batch_size,
        )
        raw_score = float(aggregate_map.max().item())
        return (
            self._evidence_from_raw(layer_scores, raw_score),
            aggregate_map,
        )

    def score(
        self,
        features: PatchFeatureBundle,
        *,
        device: str,
        query_batch_size: int = 256,
    ) -> PatchEvidence:
        return self.score_with_map(
            features,
            device=device,
            query_batch_size=query_batch_size,
        )[0]

    def save(
        self,
        path: str | Path,
        *,
        shard_by_image: bool = False,
    ) -> Path:
        self._validate(require_calibration=True)
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": PATCH_MEMORY_SCHEMA,
            "class_name": self.class_name,
            "backbone": "dinov2_vitl14_reg4",
            "layers": list(self.layers),
            "feature_grid": list(self.feature_grid),
            "input_size": self.input_size,
            "checkpoint_path": self.checkpoint_path,
            "normal_images": list(self.normal_images),
            "normal_reference_count": len(self.normal_images),
            "calibration_images": list(self.calibration_images),
            "calibration_scores": list(self.calibration_scores),
            "calibration_count": len(self.calibration_scores),
            "score_mean": self.score_mean,
            "score_unbiased_std": self.score_unbiased_std,
            "calibration_mode": LOGSAD_CALIBRATION_MODE,
            "coreset_enabled": True,
            "coreset_sampling": "image_stratified_deterministic_random",
            "coreset_ratio": self.coreset_ratio,
            "coreset_seed": self.coreset_seed,
            "coreset_patch_count": int(self.memory_source_indices.numel()),
            "matching": "global_per_layer_cosine_nearest_neighbor",
            "layer_aggregation": "mean_anomaly_map",
            "image_aggregation": "spatial_max",
            "final_mapping": "ratio_over_one_plus_ratio",
            "memory_scope": "all_train_good_images_patch_coreset",
        }
        arrays: dict[str, Any] = {
            "metadata": np.asarray(json.dumps(metadata, ensure_ascii=False)),
            "memory_source_indices": self.memory_source_indices.numpy(),
        }
        for layer, values in self.feature_memory.items():
            arrays[f"features_block_{layer}"] = values.numpy()

        if shard_by_image:
            temporary_store = Path(
                tempfile.mkdtemp(
                    prefix=f"{destination.name}.images.", dir=destination.parent
                )
            )
            published = False
            try:
                image_shards: list[dict[str, Any]] = []
                for image_index, image_path in enumerate(self.normal_images):
                    selected = self.memory_source_indices == image_index
                    patch_count = int(selected.sum().item())
                    if patch_count <= 0:
                        raise MemoryBankError(
                            f"normal image {image_index} has no Patch coreset rows"
                        )
                    shard_metadata = {
                        "schema": PATCH_MEMORY_IMAGE_SHARD_SCHEMA,
                        "image_index": image_index,
                        "image_path": image_path,
                        "layers": list(self.layers),
                        "patch_count": patch_count,
                    }
                    shard_arrays: dict[str, np.ndarray] = {
                        "metadata": np.asarray(
                            json.dumps(shard_metadata, ensure_ascii=False)
                        )
                    }
                    feature_shapes: dict[str, list[int]] = {}
                    for layer in self.layers:
                        values = self.feature_memory[layer][selected].numpy()
                        shard_arrays[f"features_block_{layer}"] = values
                        feature_shapes[str(layer)] = list(values.shape)
                    shard_name = f"image_{image_index:06d}.npz"
                    digest = _write_npz_verified(
                        temporary_store / shard_name,
                        shard_arrays,
                    )
                    image_shards.append(
                        {
                            "image_index": image_index,
                            "image_path": image_path,
                            "file": shard_name,
                            "sha256": digest,
                            "patch_count": patch_count,
                            "feature_shapes": feature_shapes,
                            "feature_dtype": "float32",
                        }
                    )
                _fsync_directory(temporary_store)
                for entry in image_shards:
                    shard_path = temporary_store / str(entry["file"])
                    if _sha256_file(shard_path) != entry["sha256"]:
                        raise MemoryBankError(
                            "global Patch pre-publish checksum failed: "
                            f"image {entry['image_index']}"
                        )
                metadata["storage"] = PATCH_MEMORY_PER_IMAGE_STORAGE
                metadata["image_shard_store"] = temporary_store.name
                metadata["image_shards"] = image_shards
                _write_npz_verified(
                    destination,
                    {"metadata": np.asarray(json.dumps(metadata, ensure_ascii=False))},
                )
                published = True
                _fsync_directory(destination.parent)
            except Exception:
                if not published and temporary_store.exists():
                    shutil.rmtree(temporary_store)
                raise
            self.source_path = str(destination)
            return destination

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                # DINO patch tensors are effectively incompressible.  Keeping the
                # NPZ entries uncompressed avoids long zlib writes and makes the
                # large bank archives more robust while preserving the schema.
                np.savez(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
            directory_descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
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
    ) -> "PatchMemoryBank":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise MemoryBankError(f"patch memory bank not found: {source}")
        try:
            with zipfile.ZipFile(source) as manifest_zip:
                bad_member = manifest_zip.testzip()
            if bad_member is not None:
                raise MemoryBankError(
                    f"patch memory manifest CRC validation failed: {bad_member}"
                )
            with np.load(source, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                if metadata.get("schema") != PATCH_MEMORY_SCHEMA:
                    raise MemoryBankError(
                        "unsupported patch memory schema; rebuild the LogSAD bank"
                    )
                layers = tuple(int(value) for value in metadata["layers"])
                storage = metadata.get("storage")
                if storage == PATCH_MEMORY_PER_IMAGE_STORAGE:
                    store_name = metadata.get("image_shard_store")
                    if (
                        not isinstance(store_name, str)
                        or not store_name
                        or Path(store_name).name != store_name
                    ):
                        raise MemoryBankError(
                            "patch memory manifest has an invalid image shard store"
                        )
                    shard_entries = metadata.get("image_shards")
                    if not isinstance(shard_entries, list) or len(shard_entries) != len(
                        metadata["normal_images"]
                    ):
                        raise MemoryBankError(
                            "patch memory image shard list does not match normal images"
                        )
                    shard_store = source.parent / store_name
                    layer_chunks: dict[int, list[torch.Tensor]] = {
                        layer: [] for layer in layers
                    }
                    source_chunks: list[torch.Tensor] = []
                    for image_index, entry in enumerate(shard_entries):
                        if not isinstance(entry, dict):
                            raise MemoryBankError(
                                f"invalid patch memory image shard {image_index}"
                            )
                        shard_name = entry.get("file")
                        if (
                            not isinstance(shard_name, str)
                            or Path(shard_name).name != shard_name
                        ):
                            raise MemoryBankError(
                                f"invalid patch memory shard filename {image_index}"
                            )
                        if (
                            entry.get("image_index") != image_index
                            or entry.get("image_path")
                            != metadata["normal_images"][image_index]
                        ):
                            raise MemoryBankError(
                                f"patch memory shard identity mismatch: {image_index}"
                            )
                        shard_path = shard_store / shard_name
                        if not shard_path.is_file():
                            raise MemoryBankError(
                                f"patch memory image shard not found: {shard_path}"
                            )
                        if _sha256_file(shard_path) != entry.get("sha256"):
                            raise MemoryBankError(
                                "patch memory image shard checksum failed: "
                                f"{image_index}"
                            )
                        with zipfile.ZipFile(shard_path) as shard_zip:
                            shard_bad_member = shard_zip.testzip()
                        if shard_bad_member is not None:
                            raise MemoryBankError(
                                "patch memory image shard CRC failed: "
                                f"{image_index}:{shard_bad_member}"
                            )
                        with np.load(shard_path, allow_pickle=False) as shard:
                            shard_metadata = json.loads(str(shard["metadata"].item()))
                            if (
                                shard_metadata.get("schema")
                                != PATCH_MEMORY_IMAGE_SHARD_SCHEMA
                                or shard_metadata.get("image_index") != image_index
                                or shard_metadata.get("image_path")
                                != metadata["normal_images"][image_index]
                                or tuple(shard_metadata.get("layers", ())) != layers
                            ):
                                raise MemoryBankError(
                                    "patch memory image shard metadata mismatch: "
                                    f"{image_index}"
                                )
                            patch_count = int(entry.get("patch_count", -1))
                            if shard_metadata.get("patch_count") != patch_count:
                                raise MemoryBankError(
                                    "patch memory image shard count mismatch: "
                                    f"{image_index}"
                                )
                            for layer in layers:
                                key = f"features_block_{layer}"
                                values = shard[key].astype(np.float32, copy=False)
                                if (
                                    list(values.shape)
                                    != entry.get("feature_shapes", {}).get(str(layer))
                                    or values.shape[0] != patch_count
                                ):
                                    raise MemoryBankError(
                                        "patch memory image shard shape mismatch: "
                                        f"{image_index}:Block {layer}"
                                    )
                                layer_chunks[layer].append(
                                    torch.from_numpy(values.copy())
                                )
                        source_chunks.append(
                            torch.full((patch_count,), image_index, dtype=torch.long)
                        )
                    feature_memory = {
                        layer: torch.cat(chunks, dim=0)
                        for layer, chunks in layer_chunks.items()
                    }
                    memory_source_indices = torch.cat(source_chunks, dim=0)
                elif storage is None:
                    feature_memory = {
                        layer: torch.from_numpy(
                            archive[f"features_block_{layer}"].astype(
                                np.float32, copy=False
                            )
                        )
                        for layer in layers
                    }
                    memory_source_indices = torch.from_numpy(
                        archive["memory_source_indices"].astype(np.int64, copy=False)
                    )
                else:
                    raise MemoryBankError(
                        f"unsupported patch memory storage layout: {storage!r}"
                    )
                bank = cls(
                    class_name=metadata["class_name"],
                    feature_memory=feature_memory,
                    memory_source_indices=memory_source_indices,
                    feature_grid=tuple(metadata["feature_grid"]),
                    checkpoint_path=metadata["checkpoint_path"],
                    normal_images=metadata["normal_images"],
                    score_mean=metadata["score_mean"],
                    score_unbiased_std=metadata["score_unbiased_std"],
                    calibration_images=metadata["calibration_images"],
                    calibration_scores=metadata["calibration_scores"],
                    input_size=metadata["input_size"],
                    coreset_ratio=metadata["coreset_ratio"],
                    coreset_seed=metadata["coreset_seed"],
                    source_path=str(source),
                    features_are_normalized=True,
                )
                bank._validate(require_calibration=True)
        except MemoryBankError:
            raise
        except Exception as exc:
            raise MemoryBankError(
                f"cannot load patch memory bank {source}: {exc}"
            ) from exc
        if expected_class_name is not None and bank.class_name != expected_class_name:
            raise MemoryBankError(
                f"patch memory class {bank.class_name!r} does not match "
                f"{expected_class_name!r}"
            )
        if expected_checkpoint_path is not None:
            expected = str(Path(expected_checkpoint_path).expanduser().resolve())
            if bank.checkpoint_path != expected:
                raise MemoryBankError(
                    "patch memory DINOv2 checkpoint does not match runtime"
                )
        return bank


class DinoV2PatchImageScorer:
    """Score a full image with LogSAD's DINOv2 patch detector."""

    def __init__(
        self,
        *,
        bank: PatchMemoryBank,
        extractor: DinoV2PatchFeatureExtractor,
        query_batch_size: int = 256,
        extraction_batch_size: int = 4,
    ) -> None:
        if bank.checkpoint_path != str(extractor.checkpoint_path):
            raise MemoryBankError("patch memory and DINOv2 checkpoint differ")
        if bank.input_size != extractor.input_size:
            raise MemoryBankError("patch memory and DINOv2 input sizes differ")
        if bank.layers != extractor.layers:
            raise MemoryBankError("patch memory and DINOv2 layers differ")
        if bank.feature_grid != extractor.feature_grid:
            raise MemoryBankError("patch memory and DINOv2 feature grids differ")
        if query_batch_size <= 0 or extraction_batch_size <= 0:
            raise ValueError("patch batch sizes must be positive")
        self.bank = bank
        self.extractor = extractor
        self.query_batch_size = int(query_batch_size)
        self.extraction_batch_size = int(extraction_batch_size)

    def extract_features(self, image: Any) -> PatchFeatureBundle:
        """Extract once so Patch and composition paths can share DINO features."""

        bundles = self.extractor.extract_batch(
            [_image(image)], batch_size=self.extraction_batch_size
        )
        if len(bundles) != 1:
            raise MemoryBankError(
                "DINOv2 returned an unexpected number of full-image features"
            )
        return bundles[0]

    def score_features(self, features: PatchFeatureBundle) -> PatchEvidence:
        """Score a pre-extracted bundle without a second DINO forward pass."""

        return self.bank.score(
            features,
            device=self.extractor.device,
            query_batch_size=self.query_batch_size,
        )

    def score_features_with_map(
        self,
        features: PatchFeatureBundle,
    ) -> tuple[PatchEvidence, torch.Tensor]:
        """Score once and retain the aggregate map for semantic composition."""

        return self.bank.score_with_map(
            features,
            device=self.extractor.device,
            query_batch_size=self.query_batch_size,
        )

    def score(self, image: Any) -> PatchEvidence:
        return self.score_features(self.extract_features(image))


class DinoV2ImageFeatureExtractor:
    """Extract one DINOv2 bundle for composition-only inference."""

    def __init__(
        self,
        *,
        extractor: DinoV2PatchFeatureExtractor,
        extraction_batch_size: int = 4,
    ) -> None:
        if extraction_batch_size <= 0:
            raise ValueError("patch extraction batch size must be positive")
        self.extractor = extractor
        self.extraction_batch_size = int(extraction_batch_size)

    def extract_features(self, image: Any) -> PatchFeatureBundle:
        bundles = self.extractor.extract_batch(
            [_image(image)], batch_size=self.extraction_batch_size
        )
        if len(bundles) != 1:
            raise MemoryBankError(
                "DINOv2 returned an unexpected number of full-image features"
            )
        return bundles[0]


def build_patch_memory_from_images(
    *,
    class_name: str,
    normal_images: Sequence[str | Path],
    output_path: str | Path,
    checkpoint_path: str | Path = DEFAULT_PATCH_DINOV2_CHECKPOINT,
    device: str = "cuda",
    query_batch_size: int = 256,
    extraction_batch_size: int = 4,
    coreset_ratio: float = DEFAULT_CORESET_RATIO,
    coreset_seed: int = DEFAULT_CORESET_SEED,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
    extractor: DinoV2PatchFeatureExtractor | None = None,
) -> PatchMemoryBank:
    paths = tuple(Path(path).expanduser().resolve() for path in normal_images)
    if len(paths) < MINIMUM_NORMAL_IMAGE_COUNT:
        raise ValueError("patch memory requires at least one normal image")
    if len(set(paths)) != len(paths):
        raise ValueError("normal patch images must be distinct")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("normal patch images not found: " + ", ".join(missing))
    paths = filter_decodable_images(
        paths,
        progress_callback=progress_callback,
    )
    if len(paths) < MINIMUM_NORMAL_IMAGE_COUNT:
        raise ValueError(
            "patch memory needs at least one decodable normal image after skips"
        )
    if len(paths) == 1 and not np.isclose(float(coreset_ratio), 1.0):
        raise ValueError("one-shot patch memory requires coreset ratio 1.0")
    if not 0.0 < float(coreset_ratio) <= 1.0:
        raise ValueError("patch coreset ratio must be in (0, 1]")
    if int(coreset_seed) < 0:
        raise ValueError("patch coreset seed must be non-negative")

    requested_checkpoint = Path(checkpoint_path).expanduser().resolve()
    if extractor is None:
        extractor = DinoV2PatchFeatureExtractor(
            checkpoint_path=requested_checkpoint,
            device=device,
        )
    else:
        if extractor.checkpoint_path != requested_checkpoint:
            raise ValueError(
                "shared DINOv2 extractor checkpoint does not match the requested "
                "patch-memory checkpoint"
            )
        if extractor.device != str(device):
            raise ValueError(
                "shared DINOv2 extractor device does not match the requested "
                "patch-memory device"
            )
    layer_chunks: dict[int, list[torch.Tensor]] = {
        layer: [] for layer in PATCH_MATCHING_LAYERS
    }
    source_chunks: list[torch.Tensor] = []
    feature_grids: set[tuple[int, int]] = set()
    for start in range(0, len(paths), extraction_batch_size):
        batch_paths = paths[start : start + extraction_batch_size]
        bundles = extractor.extract_batch(
            batch_paths,
            batch_size=extraction_batch_size,
        )
        if len(bundles) != len(batch_paths):
            raise MemoryBankError(
                "DINOv2 returned an unexpected number of full-image memory features"
            )
        feature_grids.update(tuple(item.feature_grid) for item in bundles)
        batch_memory, batch_sources = _build_coreset(
            bundles,
            ratio=coreset_ratio,
            seed=coreset_seed,
            image_index_offset=start,
        )
        for layer in PATCH_MATCHING_LAYERS:
            layer_chunks[layer].append(batch_memory[layer])
        source_chunks.append(batch_sources)
        if progress_callback is not None:
            progress_callback(
                "extract+coreset",
                start + len(batch_paths),
                len(paths),
                batch_paths[-1].name,
            )
    if len(feature_grids) != 1:
        raise MemoryBankError("normal patch features use different feature grids")
    feature_memory = {
        layer: torch.cat(chunks, dim=0) for layer, chunks in layer_chunks.items()
    }
    memory_source_indices = torch.cat(source_chunks, dim=0)
    provisional = PatchMemoryBank(
        class_name=class_name,
        feature_memory=feature_memory,
        memory_source_indices=memory_source_indices,
        feature_grid=next(iter(feature_grids)),
        normal_images=paths,
        checkpoint_path=extractor.checkpoint_path,
        score_mean=0.0,
        score_unbiased_std=1.0,
        coreset_ratio=coreset_ratio,
        coreset_seed=coreset_seed,
        features_are_normalized=True,
    )

    # Multi-shot uses leave-one-image-out calibration. One-shot instead excludes
    # each patch's identical self-match and calibrates against the other patches
    # from the sole normal reference image.
    calibration_scores: list[float] = []
    for start in range(0, len(paths), extraction_batch_size):
        batch_paths = paths[start : start + extraction_batch_size]
        batch_bundles = extractor.extract_batch(
            batch_paths,
            batch_size=extraction_batch_size,
        )
        if len(batch_bundles) != len(batch_paths):
            raise MemoryBankError(
                "DINOv2 returned an unexpected number of calibration features"
            )
        if len(paths) == 1:
            calibration_scores.append(
                _single_image_self_exclusion_score(
                    batch_bundles[0],
                    device=device,
                    query_batch_size=query_batch_size,
                )
            )
        else:
            calibration_scores.extend(
                provisional.raw_score(
                    item,
                    device=device,
                    query_batch_size=query_batch_size,
                    excluded_reference=start + local_index,
                )[1]
                for local_index, item in enumerate(batch_bundles)
            )
        if progress_callback is not None:
            progress_callback(
                "leave-one-out-calibration",
                start + len(batch_paths),
                len(paths),
                batch_paths[-1].name,
            )

    score_values = np.asarray(calibration_scores, dtype=np.float64)
    score_mean = float(np.mean(score_values))
    score_unbiased_std = (
        max(abs(score_mean) * 0.1, 1e-6)
        if len(calibration_scores) == 1
        else float(np.std(score_values, ddof=1))
    )
    if not np.isfinite(score_unbiased_std) or score_unbiased_std <= 1e-12:
        raise MemoryBankError(
            "leave-one-out normal scores have zero variance; LogSAD calibration "
            "cannot standardize them"
        )
    bank = PatchMemoryBank(
        class_name=class_name,
        feature_memory=feature_memory,
        memory_source_indices=memory_source_indices,
        feature_grid=next(iter(feature_grids)),
        checkpoint_path=extractor.checkpoint_path,
        normal_images=paths,
        score_mean=score_mean,
        score_unbiased_std=score_unbiased_std,
        calibration_images=paths,
        calibration_scores=calibration_scores,
        coreset_ratio=coreset_ratio,
        coreset_seed=coreset_seed,
        features_are_normalized=True,
    )
    bank.save(output_path, shard_by_image=True)
    if progress_callback is not None:
        progress_callback("saved", len(paths), len(paths), str(output_path))
    return bank
