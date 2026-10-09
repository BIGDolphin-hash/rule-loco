"""Semantic attribute inference with a shared local CLIP runtime."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from ..attributes import attribute_vocabulary_key
from ..errors import ExecutionError, ModelAdapterError
from ..models import AttributePrediction, MaskInstance


class LocalClipRuntime:
    """Lazily load one local CLIP model shared by all CLIP-based branches."""

    def __init__(
        self,
        *,
        repo_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda",
    ) -> None:
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = device
        self.torch: Any = None
        self.clip: Any = None
        self.model: Any = None
        self.preprocess: Any = None

    def load(self) -> tuple[Any, Any, Any, Any]:
        if self.model is not None:
            return self.torch, self.clip, self.model, self.preprocess
        if not self.repo_path.is_dir():
            raise ModelAdapterError(f"CLIP repository not found: {self.repo_path}")
        if not self.checkpoint_path.is_file():
            raise ModelAdapterError(f"CLIP checkpoint not found: {self.checkpoint_path}")
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))
        try:
            import clip
            import torch
        except Exception as exc:  # pragma: no cover - environment-specific
            raise ModelAdapterError(f"cannot import local CLIP: {exc}") from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ModelAdapterError("CUDA was requested for CLIP but is not available")
        try:
            model, preprocess = clip.load(
                str(self.checkpoint_path), device=self.device, jit=False
            )
            model.eval()
        except Exception as exc:  # pragma: no cover - heavyweight runtime
            raise ModelAdapterError(f"cannot initialize local CLIP: {exc}") from exc
        self.torch = torch
        self.clip = clip
        self.model = model
        self.preprocess = preprocess
        return torch, clip, model, preprocess


class ClipAttributeDetector:
    """Classify semantic attributes against an explicit candidate vocabulary.

    Candidate labels are configuration, never copied from the standard template,
    so expected values cannot leak into inference.
    """

    def __init__(
        self,
        *,
        repo_path: str | Path,
        checkpoint_path: str | Path,
        vocabulary: Mapping[str, Sequence[str]],
        device: str = "cuda",
        prompt_template: str = "a photo of a {subject} whose {attribute} is {label}",
        confidence_threshold: float = 0.35,
        runtime: LocalClipRuntime | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.vocabulary = {
            key: tuple(str(value) for value in values) for key, values in vocabulary.items()
        }
        self.device = device
        self.confidence_threshold = float(confidence_threshold)
        if not isfinite(self.confidence_threshold) or not (
            0.0 <= self.confidence_threshold <= 1.0
        ):
            raise ValueError("CLIP attribute confidence threshold must be in [0, 1]")
        self.runtime = runtime or LocalClipRuntime(
            repo_path=self.repo_path,
            checkpoint_path=self.checkpoint_path,
            device=device,
        )
        if (
            self.runtime.repo_path != self.repo_path
            or self.runtime.checkpoint_path != self.checkpoint_path
            or self.runtime.device != self.device
        ):
            raise ValueError("shared CLIP runtime configuration does not match detector")
        self.prompt_template = prompt_template
        self._torch: Any = None
        self._clip: Any = None
        self._model: Any = None
        self._preprocess: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        (
            self._torch,
            self._clip,
            self._model,
            self._preprocess,
        ) = self.runtime.load()

    def _candidates(self, subject: str, property_name: str) -> tuple[str, ...]:
        key = attribute_vocabulary_key(subject, property_name)
        candidates = self.vocabulary.get(key)
        if not candidates:
            raise ExecutionError(
                f"no CLIP candidate vocabulary configured for {key}"
            )
        if len(candidates) < 2:
            raise ExecutionError(
                f"CLIP vocabulary for {key} needs at least two candidates"
            )
        return tuple(candidates)

    @staticmethod
    def _image(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                return opened.convert("RGB")
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in {3, 4}:
            raise ExecutionError(f"expected an RGB image, got shape {array.shape}")
        return Image.fromarray(array[:, :, :3].astype(np.uint8), mode="RGB")

    @staticmethod
    def _masked_crop(image: Image.Image, instance: MaskInstance) -> Image.Image:
        rows, columns = np.nonzero(instance.mask)
        left, right = int(columns.min()), int(columns.max()) + 1
        top, bottom = int(rows.min()), int(rows.max()) + 1
        crop = image.crop((left, top, right, bottom))
        mask = Image.fromarray(
            (instance.mask[top:bottom, left:right].astype(np.uint8) * 255), mode="L"
        )
        background = Image.new("RGB", crop.size, color=(127, 127, 127))
        background.paste(crop, mask=mask)
        return background

    def detect(
        self,
        image: Any,
        subject: str,
        property_name: str,
        instances: Sequence[MaskInstance],
    ) -> AttributePrediction:
        if not instances:
            raise ExecutionError("CLIP attribute inference found no segmented instances")
        candidates = self._candidates(subject, property_name)
        self._load()
        pil_image = self._image(image)
        crops = [self._masked_crop(pil_image, item) for item in instances]
        attribute = property_name.split(".", 1)[-1]
        prompts = [
            self.prompt_template.format(
                subject=subject, attribute=attribute, label=label
            )
            for label in candidates
        ]
        try:
            image_tensor = self._torch.stack(
                [self._preprocess(crop) for crop in crops]
            ).to(self.device)
            text_tensor = self._clip.tokenize(prompts).to(self.device)
            with self._torch.inference_mode():
                image_features = self._model.encode_image(image_tensor)
                text_features = self._model.encode_text(text_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                probabilities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                average = probabilities.mean(dim=0)
                best_index = int(average.argmax().item())
                per_instance_indexes = probabilities.argmax(dim=-1).tolist()
                candidate_probabilities = {
                    label: float(average[index].item())
                    for index, label in enumerate(candidates)
                }
                confidence = float(average[best_index].item())
            if confidence < self.confidence_threshold:
                raise ExecutionError(
                    "CLIP attribute confidence "
                    f"{confidence:.6f} is below threshold "
                    f"{self.confidence_threshold:.6f}"
                )
            return AttributePrediction(
                value=candidates[best_index],
                confidence=confidence,
                per_instance=tuple(candidates[index] for index in per_instance_indexes),
                probabilities=candidate_probabilities,
            )
        except ExecutionError:
            raise
        except Exception as exc:  # pragma: no cover - heavyweight runtime
            raise ModelAdapterError(f"CLIP inference failed: {exc}") from exc
