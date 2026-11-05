"""Lazy local adapters for heavyweight vision models."""

from .clip import ClipAttributeDetector, LocalClipRuntime
from .patch_dinov2 import DinoV2PatchFeatureExtractor, PatchFeatureBundle
from .sam3 import Sam3Segmenter
from .whole_image import WholeImageSegmenter

__all__ = [
    "ClipAttributeDetector",
    "DinoV2PatchFeatureExtractor",
    "LocalClipRuntime",
    "PatchFeatureBundle",
    "Sam3Segmenter",
    "WholeImageSegmenter",
]
