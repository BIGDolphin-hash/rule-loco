"""Public API for the independent GCDA module."""

from .core import (
    CATEGORY_ONLY_FEATURE_MODE,
    CATEGORY_ONLY_MEMORY_SCHEMA,
    DEFAULT_DUAL_FUSION_TAU,
    GLOBAL_CATEGORY_ONLY_DUAL_MODE,
    CategoryOnlyCompositionMemoryBank,
    CategoryOnlyCompositionScorer,
    GlobalCategoryOnlyDualEvidence,
    GlobalCategoryOnlyDualScorer,
    category_only_embedding,
)
from .global_patch import GCDAGlobalPatchBank, GCDAGlobalPatchScorer

__all__ = (
    "CATEGORY_ONLY_FEATURE_MODE",
    "CATEGORY_ONLY_MEMORY_SCHEMA",
    "DEFAULT_DUAL_FUSION_TAU",
    "GLOBAL_CATEGORY_ONLY_DUAL_MODE",
    "GCDAGlobalPatchBank",
    "GCDAGlobalPatchScorer",
    "CategoryOnlyCompositionMemoryBank",
    "CategoryOnlyCompositionScorer",
    "GlobalCategoryOnlyDualEvidence",
    "GlobalCategoryOnlyDualScorer",
    "category_only_embedding",
)
