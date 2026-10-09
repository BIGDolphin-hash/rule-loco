"""LogSAD few-shot DINOv2 patch memory and image-level scoring."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional

from .adapters.patch_dinov2 import (
    DEFAULT_PATCH_DINOV2_CHECKPOINT,
    PATCH_MATCHING_INPUT_SIZE,
    PATCH_MATCHING_LAYERS,
    DinoV2PatchFeatureExtractor,
    PatchFeatureBundle,
)
from .errors import MemoryBankError
from .models import PatchEvidence


PATCH_MEMORY_SCHEMA = "pro-innovation-logsad-patch-reg4-memory-v8"
NORMAL_REFERENCE_COUNT = 4
DEFAULT_PATCH_LAYER_FUSION = "logsad_mean_map"
LOGSAD_CALIBRATION_MODE = "normal_validation_mean_unbiased_std_sigmoid"


def patch_memory_run_tag() -> str:
    """Return the bank tag for the fixed LogSAD DINOv2 configuration."""

    layer_tag = "".join(f"b{layer}" for layer in PATCH_MATCHING_LAYERS)
    return f"dino_{layer_tag}"


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    return functional.normalize(tensor.detach().float(), dim=-1, eps=1e-12)


def _sigmoid(value: float) -> float:
    """Numerically stable scalar sigmoid used by LogSAD calibration."""

    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise MemoryBankError(f"expected an RGB image, got shape {array.shape}")
    return Image.fromarray(array[..., :3].astype(np.uint8), mode="RGB")


class PatchMemoryBank:
    """Four-shot full-image Patch memory with LogSAD score calibration."""

    def __init__(
        self,
        *,
        class_name: str,
        feature_memory: Mapping[int, torch.Tensor],
        feature_grid: tuple[int, int],
        checkpoint_path: str | Path,
        normal_images: Sequence[str | Path],
        score_mean: float,
        score_unbiased_std: float,
        calibration_images: Sequence[str | Path] = (),
        calibration_scores: Sequence[float] = (),
        input_size: int = PATCH_MATCHING_INPUT_SIZE,
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
            int(layer): prepare(values)
            for layer, values in feature_memory.items()
        }
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
        self.source_path = source_path
        self.calibration_mode = LOGSAD_CALIBRATION_MODE
        self._validate(require_calibration=bool(self.calibration_images))

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted(self.feature_memory))

    @property
    def threshold(self) -> float:
        """Raw structural score mapped to calibrated probability 0.5."""

        return self.score_mean

    def _validate(self, *, require_calibration: bool) -> None:
        if not self.class_name:
            raise MemoryBankError("patch memory class name is empty")
        if self.layers != PATCH_MATCHING_LAYERS:
            raise MemoryBankError(
                f"patch memory layers must be {PATCH_MATCHING_LAYERS}"
            )
        if len(self.normal_images) != NORMAL_REFERENCE_COUNT:
            raise MemoryBankError("patch memory must contain exactly four normal images")
        if len(set(self.normal_images)) != NORMAL_REFERENCE_COUNT:
            raise MemoryBankError("patch memory normal images must be distinct")
        if self.input_size != PATCH_MATCHING_INPUT_SIZE:
            raise MemoryBankError(
                f"patch memory input size must be {PATCH_MATCHING_INPUT_SIZE}"
            )
        if self.feature_grid[0] <= 0 or self.feature_grid[1] <= 0:
            raise MemoryBankError("patch memory feature grid must be positive")
        if not np.isfinite(self.score_mean) or not 0.0 <= self.score_mean <= 2.0:
            raise MemoryBankError("LogSAD patch score mean must be in [0, 2]")
        if (
            not np.isfinite(self.score_unbiased_std)
            or self.score_unbiased_std <= 0.0
        ):
            raise MemoryBankError("LogSAD patch score standard deviation must be positive")
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
        if require_calibration and len(self.calibration_scores) < 2:
            raise MemoryBankError(
                "LogSAD calibration requires at least two normal validation images"
            )

        positions = self.feature_grid[0] * self.feature_grid[1]
        dimensions = set()
        for layer, values in self.feature_memory.items():
            if values.ndim != 3 or values.shape[0] != NORMAL_REFERENCE_COUNT:
                raise MemoryBankError(
                    f"full-image Block {layer} must keep four references"
                )
            if values.shape[1] != positions or not torch.isfinite(values).all():
                raise MemoryBankError(f"invalid full-image Block {layer} memory")
            dimensions.add(int(values.shape[2]))
        if len(dimensions) != 1:
            raise MemoryBankError("patch memory layers use different dimensions")

    @classmethod
    def build(
        cls,
        *,
        class_name: str,
        features: Sequence[PatchFeatureBundle],
        normal_images: Sequence[str | Path],
        calibration_features: Sequence[PatchFeatureBundle],
        calibration_images: Sequence[str | Path],
        checkpoint_path: str | Path,
        device: str,
        query_batch_size: int = 256,
    ) -> "PatchMemoryBank":
        if len(features) != NORMAL_REFERENCE_COUNT:
            raise ValueError("patch memory construction requires four full images")
        if len(normal_images) != NORMAL_REFERENCE_COUNT:
            raise ValueError("patch memory construction requires four normal images")
        if len(calibration_features) != len(calibration_images):
            raise ValueError(
                "patch calibration requires one feature bundle per validation image"
            )
        if len(calibration_features) < 2:
            raise ValueError(
                "LogSAD patch calibration requires at least two validation images"
            )
        if query_batch_size <= 0:
            raise ValueError("patch query batch size must be positive")
        grids = {
            tuple(item.feature_grid)
            for item in tuple(features) + tuple(calibration_features)
        }
        if len(grids) != 1:
            raise ValueError("normal patch features use different feature grids")
        for item in tuple(features) + tuple(calibration_features):
            if tuple(sorted(item.layer_features)) != PATCH_MATCHING_LAYERS:
                raise ValueError("full-image patch features are missing layers")

        memory = {
            layer: torch.stack(
                [_normalize(item.layer_features[layer]) for item in features], dim=0
            )
            for layer in PATCH_MATCHING_LAYERS
        }
        provisional = cls(
            class_name=class_name,
            feature_memory=memory,
            feature_grid=next(iter(grids)),
            checkpoint_path=checkpoint_path,
            normal_images=normal_images,
            score_mean=0.0,
            score_unbiased_std=1.0,
        )
        raw_scores = tuple(
            provisional.raw_score(
                item,
                device=device,
                query_batch_size=query_batch_size,
            )[1]
            for item in calibration_features
        )
        score_mean = float(np.mean(np.asarray(raw_scores, dtype=np.float64)))
        score_unbiased_std = float(
            np.std(np.asarray(raw_scores, dtype=np.float64), ddof=1)
        )
        if not np.isfinite(score_unbiased_std) or score_unbiased_std <= 1e-12:
            raise MemoryBankError(
                "normal validation scores have zero variance; LogSAD calibration "
                "cannot standardize them"
            )
        bank = cls(
            class_name=class_name,
            feature_memory=memory,
            feature_grid=next(iter(grids)),
            checkpoint_path=checkpoint_path,
            normal_images=normal_images,
            score_mean=score_mean,
            score_unbiased_std=score_unbiased_std,
            calibration_images=calibration_images,
            calibration_scores=raw_scores,
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

        self._validate_query(features)
        if query_batch_size <= 0:
            raise ValueError("patch query batch size must be positive")
        if excluded_reference is not None and not (
            0 <= int(excluded_reference) < NORMAL_REFERENCE_COUNT
        ):
            raise ValueError("excluded patch reference index is out of range")

        layer_scores: dict[str, float] = {}
        anomaly_maps: list[torch.Tensor] = []
        with torch.inference_mode():
            for layer in self.layers:
                references = memories[layer]
                if excluded_reference is not None:
                    keep = [
                        index
                        for index in range(NORMAL_REFERENCE_COUNT)
                        if index != int(excluded_reference)
                    ]
                    references = references[keep]
                references = references.reshape(-1, references.shape[-1]).to(device)
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
        aggregate_score = float(aggregate_map.max().cpu().item())
        return layer_scores, aggregate_score

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

    def score(
        self,
        features: PatchFeatureBundle,
        *,
        device: str,
        query_batch_size: int = 256,
    ) -> PatchEvidence:
        layer_scores, raw_score = self.raw_score(
            features,
            device=device,
            query_batch_size=query_batch_size,
        )
        standardized = (raw_score - self.score_mean) / self.score_unbiased_std
        calibrated_score = _sigmoid(standardized)
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
            "matching": "global_cosine_nearest_neighbor",
            "layer_aggregation": "mean_anomaly_map",
            "image_aggregation": "spatial_max",
        }
        return PatchEvidence(
            layer_scores=layer_scores,
            aggregate_raw_score=raw_score,
            threshold=self.score_mean,
            calibrated_score=calibrated_score,
            anomaly=calibrated_score > 0.5,
            fusion_score=calibrated_score,
            fusion_threshold=0.5,
            top_fraction=None,
            bank_path=self.source_path,
            layer_thresholds={},
            layer_calibrated_scores={},
            layer_fusion=DEFAULT_PATCH_LAYER_FUSION,
            primary_layer=None,
            winning_layer=winning_layer,
            full_image_score=full_payload,
        )

    def save(self, path: str | Path) -> Path:
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
            "normal_reference_count": NORMAL_REFERENCE_COUNT,
            "calibration_images": list(self.calibration_images),
            "calibration_scores": list(self.calibration_scores),
            "calibration_count": len(self.calibration_scores),
            "score_mean": self.score_mean,
            "score_unbiased_std": self.score_unbiased_std,
            "calibration_mode": LOGSAD_CALIBRATION_MODE,
            "coreset_enabled": False,
            "matching": "global_per_layer_cosine_nearest_neighbor",
            "layer_aggregation": "mean_anomaly_map",
            "image_aggregation": "spatial_max",
            "final_mapping": "sigmoid_standardized_score",
            "memory_scope": "full_image_only",
        }
        arrays: dict[str, Any] = {
            "metadata": np.asarray(json.dumps(metadata, ensure_ascii=False))
        }
        for layer, values in self.feature_memory.items():
            arrays[f"features_block_{layer}"] = values.numpy()

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
            with np.load(source, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                if metadata.get("schema") != PATCH_MEMORY_SCHEMA:
                    raise MemoryBankError(
                        "unsupported patch memory schema; rebuild the LogSAD bank"
                    )
                layers = tuple(int(value) for value in metadata["layers"])
                bank = cls(
                    class_name=metadata["class_name"],
                    feature_memory={
                        layer: torch.from_numpy(
                            archive[f"features_block_{layer}"].astype(
                                np.float32, copy=False
                            )
                        )
                        for layer in layers
                    },
                    feature_grid=tuple(metadata["feature_grid"]),
                    checkpoint_path=metadata["checkpoint_path"],
                    normal_images=metadata["normal_images"],
                    score_mean=metadata["score_mean"],
                    score_unbiased_std=metadata["score_unbiased_std"],
                    calibration_images=metadata["calibration_images"],
                    calibration_scores=metadata["calibration_scores"],
                    input_size=metadata["input_size"],
                    source_path=str(source),
                    features_are_normalized=True,
                )
                bank._validate(require_calibration=True)
        except MemoryBankError:
            raise
        except Exception as exc:
            raise MemoryBankError(f"cannot load patch memory bank {source}: {exc}") from exc
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

    def score(self, image: Any) -> PatchEvidence:
        bundles = self.extractor.extract_batch(
            [_image(image)], batch_size=self.extraction_batch_size
        )
        if len(bundles) != 1:
            raise MemoryBankError(
                "DINOv2 returned an unexpected number of full-image features"
            )
        return self.bank.score(
            bundles[0],
            device=self.extractor.device,
            query_batch_size=self.query_batch_size,
        )


def build_patch_memory_from_images(
    *,
    class_name: str,
    normal_images: Sequence[str | Path],
    calibration_images: Sequence[str | Path],
    output_path: str | Path,
    checkpoint_path: str | Path = DEFAULT_PATCH_DINOV2_CHECKPOINT,
    device: str = "cuda",
    query_batch_size: int = 256,
    extraction_batch_size: int = 4,
) -> PatchMemoryBank:
    if len(normal_images) != NORMAL_REFERENCE_COUNT:
        raise ValueError("patch memory construction requires exactly four normal images")
    paths = tuple(Path(path).expanduser().resolve() for path in normal_images)
    if len(set(paths)) != NORMAL_REFERENCE_COUNT:
        raise ValueError("the four normal patch images must be distinct")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("normal patch images not found: " + ", ".join(missing))

    calibration_paths = tuple(
        Path(path).expanduser().resolve() for path in calibration_images
    )
    if len(calibration_paths) < 2:
        raise ValueError(
            "LogSAD patch calibration requires at least two normal validation images"
        )
    if len(set(calibration_paths)) != len(calibration_paths):
        raise ValueError("patch calibration images must be distinct")
    if set(paths) & set(calibration_paths):
        raise ValueError(
            "LogSAD memory images and normal validation images must be independent"
        )
    missing_calibration = [
        str(path) for path in calibration_paths if not path.is_file()
    ]
    if missing_calibration:
        raise ValueError(
            "patch calibration images not found: " + ", ".join(missing_calibration)
        )

    extractor = DinoV2PatchFeatureExtractor(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    bundles = extractor.extract_batch(paths, batch_size=extraction_batch_size)
    if len(bundles) != NORMAL_REFERENCE_COUNT:
        raise MemoryBankError(
            "DINOv2 returned an unexpected number of full-image memory features"
        )

    feature_grids = {tuple(item.feature_grid) for item in bundles}
    if len(feature_grids) != 1:
        raise MemoryBankError("normal patch features use different feature grids")
    feature_memory = {
        layer: torch.stack(
            [_normalize(item.layer_features[layer]) for item in bundles], dim=0
        )
        for layer in PATCH_MATCHING_LAYERS
    }
    provisional = PatchMemoryBank(
        class_name=class_name,
        feature_memory=feature_memory,
        feature_grid=next(iter(feature_grids)),
        normal_images=paths,
        checkpoint_path=extractor.checkpoint_path,
        score_mean=0.0,
        score_unbiased_std=1.0,
    )

    # Stream validation features so a full LOCO validation split never resides
    # in host memory at once. This changes memory use only, not LogSAD scores.
    calibration_scores: list[float] = []
    for start in range(0, len(calibration_paths), extraction_batch_size):
        batch_paths = calibration_paths[start : start + extraction_batch_size]
        batch_bundles = extractor.extract_batch(
            batch_paths,
            batch_size=extraction_batch_size,
        )
        if len(batch_bundles) != len(batch_paths):
            raise MemoryBankError(
                "DINOv2 returned an unexpected number of calibration features"
            )
        calibration_scores.extend(
            provisional.raw_score(
                item,
                device=device,
                query_batch_size=query_batch_size,
            )[1]
            for item in batch_bundles
        )

    score_values = np.asarray(calibration_scores, dtype=np.float64)
    score_mean = float(np.mean(score_values))
    score_unbiased_std = float(np.std(score_values, ddof=1))
    if not np.isfinite(score_unbiased_std) or score_unbiased_std <= 1e-12:
        raise MemoryBankError(
            "normal validation scores have zero variance; LogSAD calibration "
            "cannot standardize them"
        )
    bank = PatchMemoryBank(
        class_name=class_name,
        feature_memory=feature_memory,
        feature_grid=next(iter(feature_grids)),
        checkpoint_path=extractor.checkpoint_path,
        normal_images=paths,
        score_mean=score_mean,
        score_unbiased_std=score_unbiased_std,
        calibration_images=calibration_paths,
        calibration_scores=calibration_scores,
        features_are_normalized=True,
    )
    bank.save(output_path)
    return bank
