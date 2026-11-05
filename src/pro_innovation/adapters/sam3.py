"""Adapter for the user's local Meta SAM3 checkout and checkpoint."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
from PIL import Image

from ..errors import ModelAdapterError
from ..models import MaskInstance


class Sam3Segmenter:
    CACHE_SCHEMA = "pro-innovation.sam3-proposals.v1"
    CACHE_MODES = ("off", "read", "read-write", "refresh")
    CACHE_THRESHOLD_ATOL = 1e-6

    def __init__(
        self,
        *,
        repo_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda",
        confidence_threshold: float = 0.40,
        category_thresholds: Mapping[str, float] | None = None,
        prompt_template: str = "{category}",
        cache_root: str | Path | None = None,
        cache_mode: str = "off",
    ) -> None:
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = device
        self.confidence_threshold = self._validate_threshold(
            confidence_threshold, "confidence threshold"
        )
        self.category_thresholds = {
            str(category): self._validate_threshold(
                threshold, f"threshold for {category!r}"
            )
            for category, threshold in (category_thresholds or {}).items()
        }
        self.proposal_threshold = min(
            (self.confidence_threshold, *self.category_thresholds.values())
        )
        self.prompt_template = prompt_template
        if cache_mode not in self.CACHE_MODES:
            raise ValueError(
                f"unsupported SAM3 cache mode {cache_mode!r}; "
                f"expected one of {self.CACHE_MODES}"
            )
        if cache_mode != "off" and cache_root is None:
            raise ValueError("SAM3 cache mode requires cache_root")
        self.cache_root = (
            Path(cache_root).expanduser().resolve() if cache_root is not None else None
        )
        self.cache_mode = cache_mode
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_writes = 0
        self._torch: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        if self._processor is not None:
            return
        if not self.repo_path.is_dir():
            raise ModelAdapterError(f"SAM3 repository not found: {self.repo_path}")
        if not self.checkpoint_path.is_file():
            raise ModelAdapterError(f"SAM3 checkpoint not found: {self.checkpoint_path}")
        sys.path.insert(0, str(self.repo_path))
        try:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except Exception as exc:  # pragma: no cover - environment-specific
            raise ModelAdapterError(f"cannot import local SAM3: {exc}") from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ModelAdapterError("CUDA was requested for SAM3 but is not available")
        try:
            model = build_sam3_image_model(
                checkpoint_path=str(self.checkpoint_path),
                load_from_HF=False,
                device=self.device,
                eval_mode=True,
            )
            self._processor = Sam3Processor(
                model,
                device=self.device,
                # Generate proposals at the least restrictive active threshold.
                # Per-object thresholds are applied below without a second model run.
                confidence_threshold=self.proposal_threshold,
            )
            self._torch = torch
        except Exception as exc:  # pragma: no cover - heavyweight runtime
            raise ModelAdapterError(f"cannot initialize local SAM3: {exc}") from exc

    def initialize(self) -> None:
        """Load the model before per-image timing begins."""

        self._load()

    @staticmethod
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
            raise ModelAdapterError(f"expected an RGB image, got shape {array.shape}")
        return Image.fromarray(array[:, :, :3].astype(np.uint8), mode="RGB")

    def segment(
        self, image: Any, categories: Sequence[str]
    ) -> dict[str, tuple[MaskInstance, ...]]:
        pil_image = self._image(image)
        requested = tuple(dict.fromkeys(str(category) for category in categories))
        proposals: dict[str, tuple[MaskInstance, ...]] = {}
        missing: list[str] = []
        if self.cache_mode != "off" and self.cache_mode != "refresh":
            for category in requested:
                cached = self._read_cache(pil_image, category)
                if cached is None:
                    self.cache_misses += 1
                    missing.append(category)
                else:
                    self.cache_hits += 1
                    proposals[category] = cached
        else:
            missing.extend(requested)
            if self.cache_mode == "refresh":
                self.cache_misses += len(requested)
        if not missing:
            return self._apply_thresholds(proposals, requested)
        if self.cache_mode == "read":
            raise ModelAdapterError(
                "SAM3 cache miss in read mode for categories: "
                + ", ".join(missing)
            )

        self._load()
        try:
            precision_context = (
                self._torch.autocast(
                    device_type="cuda", dtype=self._torch.bfloat16
                )
                if self.device.startswith("cuda")
                else nullcontext()
            )
            with self._torch.inference_mode(), precision_context:
                state = self._processor.set_image(pil_image)
                for category in missing:
                    prompt = self.prompt_template.format(category=category)
                    output = self._processor.set_text_prompt(prompt=prompt, state=state)
                    instances = self._instances(output)
                    proposals[category] = instances
                    if self.cache_mode in {"read-write", "refresh"}:
                        self._write_cache(pil_image, category, instances)
                        self.cache_writes += 1
            return self._apply_thresholds(proposals, requested)
        except ModelAdapterError:
            raise
        except Exception as exc:  # pragma: no cover - heavyweight runtime
            raise ModelAdapterError(f"SAM3 inference failed: {exc}") from exc

    def _apply_thresholds(
        self,
        proposals: Mapping[str, Sequence[MaskInstance]],
        categories: Sequence[str],
    ) -> dict[str, tuple[MaskInstance, ...]]:
        return {
            category: tuple(
                instance
                for instance in proposals.get(category, ())
                if instance.score
                >= self.category_thresholds.get(category, self.confidence_threshold)
            )
            for category in categories
        }

    def _checkpoint_identity(self) -> dict[str, Any]:
        try:
            stat = self.checkpoint_path.stat()
        except OSError as exc:
            raise ModelAdapterError(
                f"SAM3 checkpoint not found: {self.checkpoint_path}"
            ) from exc
        return {
            "path": str(self.checkpoint_path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    @staticmethod
    def _image_digest(image: Image.Image) -> str:
        rgb = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"{rgb.width}x{rgb.height}:RGB\0".encode("ascii"))
        digest.update(np.asarray(rgb, dtype=np.uint8).tobytes(order="C"))
        return digest.hexdigest()

    def _cache_path(self, image: Image.Image, category: str) -> Path:
        assert self.cache_root is not None
        identity = {
            "schema": self.CACHE_SCHEMA,
            "checkpoint": self._checkpoint_identity(),
            "prompt_template": self.prompt_template,
        }
        config_digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        category_digest = hashlib.sha256(category.encode("utf-8")).hexdigest()[:16]
        return (
            self.cache_root
            / self.CACHE_SCHEMA.rsplit(".", 1)[-1]
            / config_digest
            / category_digest
            / f"{self._image_digest(image)}.npz"
        )

    def _read_cache(
        self, image: Image.Image, category: str
    ) -> tuple[MaskInstance, ...] | None:
        path = self._cache_path(image, category)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                schema = str(archive["schema"].item())
                stored_category = str(archive["category"].item())
                proposal_threshold = float(archive["proposal_threshold"].item())
                masks = np.asarray(archive["masks"], dtype=np.bool_)
                scores = np.asarray(archive["scores"], dtype=np.float32).reshape(-1)
                boxes = np.asarray(archive["boxes"], dtype=np.float32)
            if schema != self.CACHE_SCHEMA or stored_category != category:
                return None
            # A cache generated at a lower proposal threshold contains every
            # candidate needed by a later, stricter runtime configuration.
            if (
                proposal_threshold
                > self.proposal_threshold + self.CACHE_THRESHOLD_ATOL
            ):
                return None
            if masks.ndim != 3 or scores.shape != (masks.shape[0],):
                return None
            if masks.shape[1:] != (image.height, image.width):
                return None
            if boxes.shape != (masks.shape[0], 4):
                return None
            return tuple(
                MaskInstance(
                    mask=masks[index],
                    score=float(scores[index]),
                    box=tuple(float(value) for value in boxes[index]),
                )
                for index in range(masks.shape[0])
            )
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            return None

    def _write_cache(
        self,
        image: Image.Image,
        category: str,
        instances: Sequence[MaskInstance],
    ) -> None:
        path = self._cache_path(image, category)
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = image.height, image.width
        masks = (
            np.stack([instance.mask for instance in instances]).astype(np.bool_)
            if instances
            else np.empty((0, height, width), dtype=np.bool_)
        )
        scores = np.asarray([instance.score for instance in instances], dtype=np.float32)
        boxes = np.asarray(
            [
                instance.box if instance.box is not None else (np.nan,) * 4
                for instance in instances
            ],
            dtype=np.float32,
        ).reshape(-1, 4)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.stem}.", suffix=".npz", delete=False
            ) as temporary:
                temporary_name = temporary.name
            np.savez_compressed(
                temporary_name,
                schema=np.asarray(self.CACHE_SCHEMA),
                category=np.asarray(category),
                proposal_threshold=np.asarray(
                    self.proposal_threshold,
                    dtype=np.float64,
                ),
                masks=masks,
                scores=scores,
                boxes=boxes,
            )
            os.replace(temporary_name, path)
        finally:
            if temporary_name is not None and os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @staticmethod
    def _validate_threshold(value: float, label: str) -> float:
        threshold = float(value)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"SAM3 {label} must be in [0, 1]")
        return threshold

    @staticmethod
    def _instances(output: dict[str, Any]) -> tuple[MaskInstance, ...]:
        # NumPy normally shares storage with a CPU torch tensor.  Some SAM3
        # processor implementations reuse their output buffers on the next
        # text prompt, so retaining such views can make an earlier category's
        # masks change underneath the postprocessor.  Own every runtime array
        # at this boundary instead.
        masks = np.array(
            output["masks"].detach().bool().cpu().contiguous().numpy(),
            dtype=np.bool_,
            copy=True,
            order="C",
        )
        scores = np.array(
            output["scores"].detach().float().cpu().contiguous().numpy(),
            dtype=np.float32,
            copy=True,
            order="C",
        ).reshape(-1)
        boxes = np.array(
            output["boxes"].detach().float().cpu().contiguous().numpy(),
            dtype=np.float32,
            copy=True,
            order="C",
        )
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None, ...]
        if masks.ndim != 3:
            raise ModelAdapterError(f"unexpected SAM3 mask shape: {masks.shape}")
        instances = []
        for index, mask in enumerate(masks):
            if not np.asarray(mask, dtype=bool).any():
                continue
            box = tuple(float(value) for value in boxes[index].reshape(-1)[:4])
            instances.append(
                MaskInstance(
                    mask=np.array(mask, dtype=np.bool_, copy=True, order="C"),
                    score=float(scores[index]),
                    box=box,
                )
            )
        return tuple(instances)
