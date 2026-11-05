"""GCDA-private reader and scorer for Global Patch bank files.

This code deliberately does not import the standalone Global Patch module.
GCDA owns this implementation so changes in another memory module cannot
silently change GCDA behavior.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
import zipfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional

from ...adapters.patch_dinov2 import (
    PATCH_MATCHING_INPUT_SIZE,
    PATCH_MATCHING_LAYERS,
    DinoV2PatchFeatureExtractor,
    PatchFeatureBundle,
)
from ...errors import MemoryBankError
from ...models import PatchEvidence

_BANK_SCHEMA = "pro-innovation-logsad-patch-reg4-full-coreset-memory-v1"
_SHARD_SCHEMA = "pro-innovation-logsad-patch-reg4-full-coreset-image-shard-v1"
_SHARDED_STORAGE = "per_image_npz_sha256_v1"
_LAYER_FUSION = "logsad_mean_map"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(values: torch.Tensor) -> torch.Tensor:
    return functional.normalize(values.detach().float(), dim=-1, eps=1e-12)


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
        raise MemoryBankError(f"GCDA expected an RGB image, got {array.shape}")
    return Image.fromarray(array[..., :3].astype(np.uint8), mode="RGB")


class GCDAGlobalPatchBank:
    """GCDA-owned immutable view of a Global Patch bank."""

    def __init__(
        self,
        *,
        class_name: str,
        feature_memory: Mapping[int, torch.Tensor],
        memory_source_indices: torch.Tensor,
        feature_grid: tuple[int, int],
        checkpoint_path: str | Path,
        normal_images: tuple[str, ...],
        score_mean: float,
        score_unbiased_std: float,
        calibration_scores: tuple[float, ...],
        input_size: int,
        coreset_ratio: float,
        source_path: str,
    ) -> None:
        self.class_name = str(class_name)
        self.feature_memory = {
            int(layer): values.detach().float().cpu()
            for layer, values in feature_memory.items()
        }
        self.memory_source_indices = memory_source_indices.detach().long().cpu()
        self.feature_grid = (int(feature_grid[0]), int(feature_grid[1]))
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.normal_images = tuple(
            str(Path(path).expanduser().resolve()) for path in normal_images
        )
        self.score_mean = float(score_mean)
        self.score_unbiased_std = float(score_unbiased_std)
        self.calibration_scores = tuple(float(value) for value in calibration_scores)
        self.input_size = int(input_size)
        self.coreset_ratio = float(coreset_ratio)
        self.source_path = str(source_path)
        self._validate()

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted(self.feature_memory))

    @property
    def psad_scale(self) -> float | None:
        if not self.calibration_scores:
            return None
        scale = max(self.calibration_scores)
        return scale if np.isfinite(scale) and scale > 1e-12 else None

    def _validate(self) -> None:
        if self.layers != PATCH_MATCHING_LAYERS:
            raise MemoryBankError("GCDA Global Patch layers do not match DINOv2")
        if len(self.normal_images) < 2 or len(set(self.normal_images)) != len(
            self.normal_images
        ):
            raise MemoryBankError("GCDA Global Patch references are invalid")
        if self.input_size != PATCH_MATCHING_INPUT_SIZE:
            raise MemoryBankError("GCDA Global Patch input size is unsupported")
        if self.score_unbiased_std <= 0.0 or not np.isfinite(self.score_unbiased_std):
            raise MemoryBankError("GCDA Global Patch calibration is invalid")
        if self.memory_source_indices.ndim != 1:
            raise MemoryBankError("GCDA Global Patch source index is invalid")
        rows = int(self.memory_source_indices.numel())
        if rows <= 0:
            raise MemoryBankError("GCDA Global Patch memory is empty")
        dimensions = set()
        for layer, values in self.feature_memory.items():
            if values.ndim != 2 or values.shape[0] != rows:
                raise MemoryBankError(f"GCDA Global Patch Block {layer} is invalid")
            if not torch.isfinite(values).all():
                raise MemoryBankError(f"GCDA Global Patch Block {layer} is non-finite")
            dimensions.add(int(values.shape[1]))
        if len(dimensions) != 1:
            raise MemoryBankError("GCDA Global Patch dimensions differ by layer")
        represented = set(int(value) for value in self.memory_source_indices.tolist())
        if represented != set(range(len(self.normal_images))):
            raise MemoryBankError("GCDA Global Patch references are incomplete")

    @staticmethod
    def _load_storage(
        source: Path,
        archive: Any,
        metadata: Mapping[str, Any],
        layers: tuple[int, ...],
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        storage = metadata.get("storage")
        if storage is None:
            memory = {
                layer: torch.from_numpy(
                    archive[f"features_block_{layer}"].astype(np.float32, copy=False)
                )
                for layer in layers
            }
            indices = torch.from_numpy(
                archive["memory_source_indices"].astype(np.int64, copy=False)
            )
            return memory, indices
        if storage != _SHARDED_STORAGE:
            raise MemoryBankError(f"unsupported GCDA Global Patch storage: {storage}")
        store_name = metadata.get("image_shard_store")
        entries = metadata.get("image_shards")
        if (
            not isinstance(store_name, str)
            or Path(store_name).name != store_name
            or not isinstance(entries, list)
            or len(entries) != len(metadata["normal_images"])
        ):
            raise MemoryBankError("GCDA Global Patch shard manifest is invalid")
        chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
        source_chunks: list[torch.Tensor] = []
        for image_index, entry in enumerate(entries):
            if not isinstance(entry, dict) or entry.get("image_index") != image_index:
                raise MemoryBankError("GCDA Global Patch shard identity is invalid")
            shard_name = entry.get("file")
            if not isinstance(shard_name, str) or Path(shard_name).name != shard_name:
                raise MemoryBankError("GCDA Global Patch shard name is invalid")
            shard_path = source.parent / store_name / shard_name
            if not shard_path.is_file() or _sha256_file(shard_path) != entry.get(
                "sha256"
            ):
                raise MemoryBankError(f"GCDA Global Patch shard {image_index} failed")
            with zipfile.ZipFile(shard_path) as shard_zip:
                if shard_zip.testzip() is not None:
                    raise MemoryBankError("GCDA Global Patch shard CRC failed")
            with np.load(shard_path, allow_pickle=False) as shard:
                shard_metadata = json.loads(str(shard["metadata"].item()))
                count = int(entry.get("patch_count", -1))
                if (
                    shard_metadata.get("schema") != _SHARD_SCHEMA
                    or shard_metadata.get("image_index") != image_index
                    or tuple(shard_metadata.get("layers", ())) != layers
                    or shard_metadata.get("patch_count") != count
                ):
                    raise MemoryBankError("GCDA Global Patch shard metadata differs")
                for layer in layers:
                    values = shard[f"features_block_{layer}"].astype(
                        np.float32, copy=False
                    )
                    if values.shape[0] != count:
                        raise MemoryBankError(
                            "GCDA Global Patch shard row count differs"
                        )
                    chunks[layer].append(torch.from_numpy(values.copy()))
            source_chunks.append(torch.full((count,), image_index, dtype=torch.long))
        return (
            {layer: torch.cat(values, dim=0) for layer, values in chunks.items()},
            torch.cat(source_chunks, dim=0),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_class_name: str | None = None,
        expected_checkpoint_path: str | Path | None = None,
    ) -> "GCDAGlobalPatchBank":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise MemoryBankError(f"GCDA Global Patch bank not found: {source}")
        try:
            with zipfile.ZipFile(source) as manifest_zip:
                if manifest_zip.testzip() is not None:
                    raise MemoryBankError("GCDA Global Patch manifest CRC failed")
            with np.load(source, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                if metadata.get("schema") != _BANK_SCHEMA:
                    raise MemoryBankError(
                        "GCDA received an unsupported Global Patch bank"
                    )
                layers = tuple(int(value) for value in metadata["layers"])
                memory, indices = cls._load_storage(source, archive, metadata, layers)
                bank = cls(
                    class_name=metadata["class_name"],
                    feature_memory=memory,
                    memory_source_indices=indices,
                    feature_grid=tuple(metadata["feature_grid"]),
                    checkpoint_path=metadata["checkpoint_path"],
                    normal_images=tuple(metadata["normal_images"]),
                    score_mean=metadata["score_mean"],
                    score_unbiased_std=metadata["score_unbiased_std"],
                    calibration_scores=tuple(metadata.get("calibration_scores", ())),
                    input_size=metadata["input_size"],
                    coreset_ratio=metadata["coreset_ratio"],
                    source_path=str(source),
                )
        except MemoryBankError:
            raise
        except Exception as exc:
            raise MemoryBankError(
                f"cannot load GCDA Global Patch bank {source}: {exc}"
            ) from exc
        if expected_class_name is not None and bank.class_name != expected_class_name:
            raise MemoryBankError("GCDA Global Patch class does not match runtime")
        if expected_checkpoint_path is not None:
            expected = str(Path(expected_checkpoint_path).expanduser().resolve())
            if bank.checkpoint_path != expected:
                raise MemoryBankError("GCDA Global Patch checkpoint does not match")
        return bank

    def _raw_maps(
        self,
        features: PatchFeatureBundle,
        *,
        device: str,
        query_batch_size: int,
        excluded_reference: int | None = None,
    ) -> tuple[dict[str, float], torch.Tensor]:
        if tuple(features.feature_grid) != self.feature_grid:
            raise MemoryBankError("GCDA query grid does not match Global Patch")
        if tuple(sorted(features.layer_features)) != self.layers:
            raise MemoryBankError("GCDA query layers do not match Global Patch")
        if query_batch_size <= 0:
            raise ValueError("GCDA query batch size must be positive")
        if excluded_reference is not None and not (
            0 <= int(excluded_reference) < len(self.normal_images)
        ):
            raise ValueError("GCDA excluded reference index is out of range")
        positions = self.feature_grid[0] * self.feature_grid[1]
        dimension = next(iter(self.feature_memory.values())).shape[1]
        for layer in self.layers:
            values = features.layer_features[layer]
            if (
                values.shape != (positions, dimension)
                or not torch.isfinite(values).all()
            ):
                raise MemoryBankError(f"GCDA query Block {layer} is invalid")
        maps: list[torch.Tensor] = []
        scores: dict[str, float] = {}
        with torch.inference_mode():
            for layer in self.layers:
                references = self.feature_memory[layer]
                if excluded_reference is not None:
                    references = references[
                        self.memory_source_indices != int(excluded_reference)
                    ]
                if references.shape[0] == 0:
                    raise MemoryBankError("GCDA leave-one-out memory is empty")
                references = references.to(device)
                query = _normalize(features.layer_features[layer]).to(device)
                chunks = []
                for start in range(0, query.shape[0], query_batch_size):
                    similarity = query[start : start + query_batch_size] @ references.T
                    chunks.append((1.0 - similarity.max(dim=1).values).clamp(0.0, 2.0))
                anomaly_map = torch.cat(chunks)
                maps.append(anomaly_map)
                scores[str(layer)] = float(anomaly_map.max().cpu().item())
        return scores, torch.stack(maps, dim=0).mean(dim=0).cpu()

    def anomaly_map(
        self,
        features: PatchFeatureBundle,
        *,
        device: str,
        query_batch_size: int = 256,
        excluded_reference: int | None = None,
    ) -> torch.Tensor:
        return self._raw_maps(
            features,
            device=device,
            query_batch_size=query_batch_size,
            excluded_reference=excluded_reference,
        )[1]

    def score_with_map(
        self,
        features: PatchFeatureBundle,
        *,
        device: str,
        query_batch_size: int = 256,
    ) -> tuple[PatchEvidence, torch.Tensor]:
        layer_scores, anomaly_map = self._raw_maps(
            features, device=device, query_batch_size=query_batch_size
        )
        raw_score = float(anomaly_map.max().item())
        standardized = (raw_score - self.score_mean) / self.score_unbiased_std
        scale = self.psad_scale
        if scale is None:
            raise MemoryBankError("GCDA Global Patch normal-reference scale is unavailable")
        ratio = raw_score / scale
        calibrated = ratio / (ratio + 1.0)
        winning_layer = max(self.layers, key=lambda layer: layer_scores[str(layer)])
        full_payload = {
            "layer_scores": dict(layer_scores),
            "layer_fusion": _LAYER_FUSION,
            "raw_score": raw_score,
            "score_mean": self.score_mean,
            "score_unbiased_std": self.score_unbiased_std,
            "standardized_score": standardized,
            "calibrated_score": calibrated,
            "psad_train_loo_max": scale,
            "psad_adaptive_score": ratio,
            "score_mapping": "ratio_over_one_plus_ratio",
            "matching": "global_cosine_nearest_neighbor",
            "layer_aggregation": "mean_anomaly_map",
            "image_aggregation": "spatial_max",
        }
        evidence = PatchEvidence(
            layer_scores=layer_scores,
            aggregate_raw_score=raw_score,
            threshold=scale,
            calibrated_score=calibrated,
            anomaly=ratio > 1.0,
            fusion_score=ratio,
            fusion_threshold=1.0,
            bank_path=self.source_path,
            layer_fusion=_LAYER_FUSION,
            winning_layer=winning_layer,
            full_image_score=full_payload,
        )
        return evidence, anomaly_map

    def score(
        self,
        features: PatchFeatureBundle,
        *,
        device: str,
        query_batch_size: int = 256,
    ) -> PatchEvidence:
        return self.score_with_map(
            features, device=device, query_batch_size=query_batch_size
        )[0]


class GCDAGlobalPatchScorer:
    """GCDA-owned image adapter for its private Global Patch bank."""

    def __init__(
        self,
        *,
        bank: GCDAGlobalPatchBank,
        extractor: DinoV2PatchFeatureExtractor,
        query_batch_size: int = 256,
        extraction_batch_size: int = 4,
    ) -> None:
        if bank.checkpoint_path != str(extractor.checkpoint_path):
            raise MemoryBankError("GCDA Global Patch checkpoint differs from DINOv2")
        if (
            bank.feature_grid != extractor.feature_grid
            or bank.layers != extractor.layers
        ):
            raise MemoryBankError("GCDA Global Patch shape differs from DINOv2")
        if query_batch_size <= 0 or extraction_batch_size <= 0:
            raise ValueError("GCDA batch sizes must be positive")
        self.bank = bank
        self.extractor = extractor
        self.query_batch_size = int(query_batch_size)
        self.extraction_batch_size = int(extraction_batch_size)

    def extract_features(self, image: Any) -> PatchFeatureBundle:
        bundles = self.extractor.extract_batch(
            [_image(image)], batch_size=self.extraction_batch_size
        )
        if len(bundles) != 1:
            raise MemoryBankError("GCDA DINOv2 returned an unexpected batch size")
        return bundles[0]

    def score_features(self, features: PatchFeatureBundle) -> PatchEvidence:
        return self.bank.score(
            features,
            device=self.extractor.device,
            query_batch_size=self.query_batch_size,
        )

    def score_features_with_map(
        self, features: PatchFeatureBundle
    ) -> tuple[PatchEvidence, torch.Tensor]:
        return self.bank.score_with_map(
            features,
            device=self.extractor.device,
            query_batch_size=self.query_batch_size,
        )

    def score(self, image: Any) -> PatchEvidence:
        return self.score_features(self.extract_features(image))
