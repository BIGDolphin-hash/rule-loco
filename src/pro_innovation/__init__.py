"""Template-driven rule and patch anomaly detection.

Global Patch, GPSTM, GCDA and GPO-JM provide post-rule memory detection.
"""

from .models import (
    CompositionEvidence,
    InferenceReport,
    PatchEvidence,
    Rule,
    RuleResult,
    TaskType,
)
from .mask_processing import MaskPostprocessor
from .memory_modules.global_patch import DinoV2PatchImageScorer, PatchMemoryBank
from .pipeline import LogicAnomalyPipeline
from .templates import parse_template, serialize_template, to_generic_template

__all__ = [
    "InferenceReport",
    "CompositionEvidence",
    "LogicAnomalyPipeline",
    "MaskPostprocessor",
    "PatchMemoryBank",
    "PatchEvidence",
    "DinoV2PatchImageScorer",
    "Rule",
    "RuleResult",
    "TaskType",
    "parse_template",
    "serialize_template",
    "to_generic_template",
]
