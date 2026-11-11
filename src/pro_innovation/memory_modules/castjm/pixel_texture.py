"""Pure-texture pixel localization from a Global Patch anomaly map."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite

import torch
import torch.nn.functional as functional

from ...errors import MemoryBankError


@dataclass(frozen=True)
class GlobalPatchPixelLocalization:
    """Patch-grid and image-resolution views of one texture anomaly map."""

    patch_anomaly_map: torch.Tensor
    pixel_anomaly_map: torch.Tensor
    feature_grid: tuple[int, int]
    output_size: tuple[int, int]
    smoothing_sigma: float


class GlobalPatchPixelLocalizer:
    """Restore a Global Patch anomaly map to image resolution without semantics."""

    def __init__(self, *, smoothing_sigma: float = 0.0) -> None:
        sigma = float(smoothing_sigma)
        if not isfinite(sigma) or sigma < 0.0:
            raise ValueError("pixel texture smoothing sigma must be finite and non-negative")
        self.smoothing_sigma = sigma

    @staticmethod
    def _validate_size(name: str, size: tuple[int, int]) -> tuple[int, int]:
        if len(size) != 2:
            raise ValueError(f"{name} must contain height and width")
        height, width = int(size[0]), int(size[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"{name} dimensions must be positive")
        return height, width

    @staticmethod
    def _gaussian_blur(values: torch.Tensor, sigma: float) -> torch.Tensor:
        radius = max(1, int(ceil(3.0 * sigma)))
        coordinates = torch.arange(
            -radius,
            radius + 1,
            dtype=values.dtype,
            device=values.device,
        )
        kernel = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum()
        horizontal = kernel.reshape(1, 1, 1, -1)
        vertical = kernel.reshape(1, 1, -1, 1)
        blurred = functional.conv2d(
            functional.pad(values, (radius, radius, 0, 0), mode="replicate"),
            horizontal,
        )
        return functional.conv2d(
            functional.pad(blurred, (0, 0, radius, radius), mode="replicate"),
            vertical,
        )

    def localize(
        self,
        anomaly_map: torch.Tensor,
        *,
        feature_grid: tuple[int, int],
        output_size: tuple[int, int],
    ) -> GlobalPatchPixelLocalization:
        """Reshape, upsample, and optionally smooth one raw texture anomaly map."""

        grid_height, grid_width = self._validate_size("feature_grid", feature_grid)
        output_height, output_width = self._validate_size("output_size", output_size)
        values = anomaly_map.detach().float().cpu()
        expected = grid_height * grid_width
        if values.numel() != expected:
            raise MemoryBankError(
                "Global Patch anomaly map does not match the pixel-localization grid"
            )
        if not torch.isfinite(values).all():
            raise MemoryBankError("Global Patch pixel anomaly map contains non-finite values")
        if bool((values < 0.0).any()):
            raise MemoryBankError("Global Patch pixel anomaly map contains negative values")

        patch_map = values.reshape(grid_height, grid_width).contiguous()
        pixel_map = functional.interpolate(
            patch_map[None, None],
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=False,
        )
        if self.smoothing_sigma > 0.0:
            pixel_map = self._gaussian_blur(pixel_map, self.smoothing_sigma)
        return GlobalPatchPixelLocalization(
            patch_anomaly_map=patch_map,
            pixel_anomaly_map=pixel_map[0, 0].contiguous(),
            feature_grid=(grid_height, grid_width),
            output_size=(output_height, output_width),
            smoothing_sigma=self.smoothing_sigma,
        )


class GlobalPatchPixelScorer:
    """Pure-texture Pixel scorer backed only by a Global Patch scorer."""

    def __init__(
        self,
        *,
        global_patch_scorer,
        localizer: GlobalPatchPixelLocalizer | None = None,
    ) -> None:
        if not hasattr(global_patch_scorer, "score_features_with_map"):
            raise TypeError("Global Patch scorer must expose score_features_with_map")
        if not hasattr(global_patch_scorer, "extract_features"):
            raise TypeError("Global Patch scorer must expose extract_features")
        self.global_patch_scorer = global_patch_scorer
        self.localizer = localizer or GlobalPatchPixelLocalizer()

    def extract_features(self, image):
        return self.global_patch_scorer.extract_features(image)

    def localize_features(
        self,
        features,
        *,
        output_size: tuple[int, int],
    ) -> GlobalPatchPixelLocalization:
        """Localize one image without a template, mask pool, or joint memory."""

        _, anomaly_map = self.global_patch_scorer.score_features_with_map(features)
        return self.localizer.localize(
            anomaly_map,
            feature_grid=(
                int(features.feature_grid[0]),
                int(features.feature_grid[1]),
            ),
            output_size=output_size,
        )

    def localize(
        self,
        image,
        *,
        output_size: tuple[int, int],
    ) -> GlobalPatchPixelLocalization:
        return self.localize_features(
            self.extract_features(image),
            output_size=output_size,
        )
