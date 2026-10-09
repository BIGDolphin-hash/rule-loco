"""Adapter for the user's local Meta SAM3 checkout and checkpoint."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from ..errors import ModelAdapterError
from ..models import MaskInstance


class Sam3Segmenter:
    def __init__(
        self,
        *,
        repo_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda",
        confidence_threshold: float = 0.40,
        category_thresholds: Mapping[str, float] | None = None,
        prompt_template: str = "{category}",
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

    @staticmethod
    def _image(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                return opened.convert("RGB")
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in {3, 4}:
            raise ModelAdapterError(f"expected an RGB image, got shape {array.shape}")
        return Image.fromarray(array[:, :, :3].astype(np.uint8), mode="RGB")

    def segment(
        self, image: Any, categories: Sequence[str]
    ) -> dict[str, tuple[MaskInstance, ...]]:
        self._load()
        pil_image = self._image(image)
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
                result: dict[str, tuple[MaskInstance, ...]] = {}
                for category in categories:
                    prompt = self.prompt_template.format(category=category)
                    output = self._processor.set_text_prompt(prompt=prompt, state=state)
                    threshold = self.category_thresholds.get(
                        category, self.confidence_threshold
                    )
                    result[category] = tuple(
                        instance
                        for instance in self._instances(output)
                        if instance.score >= threshold
                    )
            return result
        except ModelAdapterError:
            raise
        except Exception as exc:  # pragma: no cover - heavyweight runtime
            raise ModelAdapterError(f"SAM3 inference failed: {exc}") from exc

    @staticmethod
    def _validate_threshold(value: float, label: str) -> float:
        threshold = float(value)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"SAM3 {label} must be in [0, 1]")
        return threshold

    @staticmethod
    def _instances(output: dict[str, Any]) -> tuple[MaskInstance, ...]:
        masks = output["masks"].detach().bool().cpu().numpy()
        scores = output["scores"].detach().float().cpu().numpy().reshape(-1)
        boxes = output["boxes"].detach().float().cpu().numpy()
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
                    mask=np.asarray(mask, dtype=bool),
                    score=float(scores[index]),
                    box=box,
                )
            )
        return tuple(instances)
