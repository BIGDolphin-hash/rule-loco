"""DINOv2-Reg4 features for the LogSAD-style patch-matching branch."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from ..errors import ModelAdapterError


DEFAULT_PATCH_DINOV2_CHECKPOINT = (
    "/home/lxq/weights/DINOV2/dinov2_vitl14_reg4_pretrain.pth"
)
PATCH_MATCHING_LAYERS = (6, 12, 18, 24)
PATCH_MATCHING_INPUT_SIZE = 448
PATCH_MATCHING_TOKEN_GRID = (32, 32)
PATCH_MATCHING_FEATURE_GRID = (64, 64)


@dataclass(frozen=True)
class PatchFeatureBundle:
    """L2-normalized 64x64 patch features indexed by one-based block number."""

    layer_features: Mapping[int, Any]
    feature_grid: tuple[int, int] = PATCH_MATCHING_FEATURE_GRID


class DinoV2PatchFeatureExtractor:
    """Extract fixed Block 6/12/18/24 DINOv2 ViT-L/14-Reg4 features."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path = DEFAULT_PATCH_DINOV2_CHECKPOINT,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = str(device)
        self.input_size = PATCH_MATCHING_INPUT_SIZE
        self.layers = PATCH_MATCHING_LAYERS
        self.token_grid = PATCH_MATCHING_TOKEN_GRID
        self.feature_grid = PATCH_MATCHING_FEATURE_GRID
        self._torch: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint_path.is_file():
            raise ModelAdapterError(
                f"DINOv2 checkpoint not found: {self.checkpoint_path}"
            )
        try:
            import torch
            from dinov2.hub.backbones import dinov2_vitl14_reg
        except Exception as exc:  # pragma: no cover - environment-specific
            raise ModelAdapterError(f"cannot import installed DINOv2: {exc}") from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ModelAdapterError("CUDA was requested for DINOv2 but is not available")
        try:
            model = dinov2_vitl14_reg(
                pretrained=True,
                weights=str(self.checkpoint_path),
            )
            model.eval().to(self.device)
        except Exception as exc:  # pragma: no cover - heavyweight runtime
            raise ModelAdapterError(f"cannot initialize local DINOv2: {exc}") from exc
        if len(list(model.blocks)) != 24:
            raise ModelAdapterError(
                "DINOv2 ViT-L/14-Reg4 must expose 24 blocks, got "
                f"{len(list(model.blocks))}"
            )
        self._torch = torch
        self._model = model

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
        return Image.fromarray(array[..., :3].astype(np.uint8), mode="RGB")

    def _preprocess(self, image: Any) -> Any:
        resized = self._image(image).resize(
            (self.input_size, self.input_size),
            Image.Resampling.BILINEAR,
        )
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = self._torch.from_numpy(array).permute(2, 0, 1)
        # LogSAD applies this CLIP normalization to the shared 448px input
        # before both its CLIP and DINOv2 feature paths.
        mean = tensor.new_tensor((0.48145466, 0.4578275, 0.40821073))[:, None, None]
        std = tensor.new_tensor((0.26862954, 0.26130258, 0.27577711))[:, None, None]
        return ((tensor - mean) / std).unsqueeze(0).to(self.device)

    def _extract_tensor(self, tensor: Any) -> tuple[PatchFeatureBundle, ...]:
        self._load()
        zero_based = [layer - 1 for layer in self.layers]

        def forward(*, autocast_enabled: bool) -> Any:
            precision_context = (
                self._torch.autocast(
                    device_type="cuda", dtype=self._torch.bfloat16
                )
                if autocast_enabled
                else nullcontext()
            )
            with self._torch.inference_mode(), precision_context:
                return self._model.get_intermediate_layers(
                    tensor,
                    n=zero_based,
                    reshape=False,
                    return_class_token=True,
                    norm=True,
                )

        try:
            use_autocast = self.device.startswith("cuda")
            outputs = forward(autocast_enabled=use_autocast)
            if use_autocast and any(
                not self._torch.isfinite(output[0]).all() for output in outputs
            ):
                outputs = forward(autocast_enabled=False)
        except Exception as exc:  # pragma: no cover - heavyweight runtime
            raise ModelAdapterError(f"DINOv2 patch extraction failed: {exc}") from exc

        try:
            bundles: list[dict[int, Any]] = [
                {} for _ in range(int(tensor.shape[0]))
            ]
            token_count = self.token_grid[0] * self.token_grid[1]
            for layer, output in zip(self.layers, outputs):
                patches, _class_token = output
                patches = patches.detach().float()
                if patches.ndim != 3 or patches.shape[1] != token_count:
                    raise ModelAdapterError(
                        f"DINOv2 Block {layer} returned shape {tuple(patches.shape)}; "
                        f"expected {token_count} after removing CLS/register tokens"
                    )
                feature_map = patches.reshape(
                    patches.shape[0],
                    self.token_grid[0],
                    self.token_grid[1],
                    patches.shape[-1],
                ).permute(0, 3, 1, 2)
                upsampled = self._torch.nn.functional.interpolate(
                    feature_map,
                    size=self.feature_grid,
                    mode="bilinear",
                    align_corners=True,
                ).permute(0, 2, 3, 1).reshape(
                    patches.shape[0],
                    self.feature_grid[0] * self.feature_grid[1],
                    -1,
                )
                normalized = self._torch.nn.functional.normalize(
                    upsampled,
                    dim=-1,
                    eps=1e-12,
                )
                normalized = normalized.detach().float().cpu()
                for index in range(len(bundles)):
                    bundles[index][layer] = normalized[index]
            if any(set(bundle) != set(self.layers) for bundle in bundles):
                raise ModelAdapterError(
                    "DINOv2 did not return every requested patch layer"
                )
            return tuple(
                PatchFeatureBundle(
                    layer_features=bundle,
                    feature_grid=self.feature_grid,
                )
                for bundle in bundles
            )
        except ModelAdapterError:
            raise
        except Exception as exc:  # pragma: no cover - heavyweight runtime
            raise ModelAdapterError(f"cannot prepare DINOv2 patch features: {exc}") from exc

    def extract(self, image: Any) -> PatchFeatureBundle:
        self._load()
        return self._extract_tensor(self._preprocess(image))[0]

    def extract_batch(
        self,
        images: Sequence[Any],
        *,
        batch_size: int = 4,
    ) -> tuple[PatchFeatureBundle, ...]:
        """Extract several full images with one shared backbone."""

        if batch_size <= 0:
            raise ValueError("DINOv2 extraction batch size must be positive")
        if not images:
            return ()
        self._load()
        result: list[PatchFeatureBundle] = []
        for start in range(0, len(images), int(batch_size)):
            tensors = [
                self._preprocess(image)
                for image in images[start : start + int(batch_size)]
            ]
            result.extend(self._extract_tensor(self._torch.cat(tensors, dim=0)))
        return tuple(result)
