"""CASTJM: Category-Aligned Semantic–Texture Joint Memory.

Public API for the independent semantic and texture joint-memory module.
"""

from .core import (
    BACKGROUND_BLOCK,
    DEFAULT_TEXTURE_WEIGHT,
    DEFAULT_TOP_FRACTION,
    CASTJM_MEMORY_SCHEMA,
    CASTJM_MODE,
    STANDARDIZATION_EPSILON,
    CASTJMEvidence,
    CASTJMMemoryBank,
    CASTJMScorer,
    global_patch_texture_signature,
)
from .global_patch import CASTJMGlobalPatchBank, CASTJMGlobalPatchScorer
from .pixel_texture import (
    GlobalPatchPixelLocalization,
    GlobalPatchPixelLocalizer,
    GlobalPatchPixelScorer,
)
from .semantic import category_only_embedding

__all__ = (
    "BACKGROUND_BLOCK",
    "DEFAULT_TEXTURE_WEIGHT",
    "DEFAULT_TOP_FRACTION",
    "CASTJM_MEMORY_SCHEMA",
    "CASTJM_MODE",
    "CASTJMGlobalPatchBank",
    "CASTJMGlobalPatchScorer",
    "STANDARDIZATION_EPSILON",
    "CASTJMEvidence",
    "CASTJMMemoryBank",
    "CASTJMScorer",
    "GlobalPatchPixelLocalization",
    "GlobalPatchPixelLocalizer",
    "GlobalPatchPixelScorer",
    "global_patch_texture_signature",
    "category_only_embedding",
)
