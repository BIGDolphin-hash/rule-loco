"""Template-driven logic and patch anomaly detection."""

from .models import InferenceReport, PatchEvidence, Rule, RuleResult, TaskType
from .mask_processing import MaskPostprocessor
from .patch_memory import DinoV2PatchImageScorer, PatchMemoryBank
from .pipeline import LogicAnomalyPipeline
from .templates import parse_template, serialize_template, to_generic_template

__all__ = [
    "InferenceReport",
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
