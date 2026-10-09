"""Shared data models for templates, masks, and inference reports."""

from __future__ import annotations

from builtins import property as builtin_property
from dataclasses import asdict, dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Mapping, TypeAlias

import numpy as np

Scalar: TypeAlias = str | int | float | bool | None
CountValues: TypeAlias = tuple[int, ...]
RuleValue: TypeAlias = Scalar | tuple[str, ...] | CountValues


class TaskType(str, Enum):
    COUNT = "COUNT"
    LENGTH = "LENGTH"
    AREA = "AREA"
    SPATIAL_COMBINATION = "SPATIAL_COMBINATION"
    ATTRIBUTE_COMBINATION = "ATTRIBUTE_COMBINATION"
    ATTRIBUTE_ERROR = "ATTRIBUTE_ERROR"


@dataclass(frozen=True)
class AttributeTarget:
    index: int
    subject: str
    property: str
    expected: Scalar = None


@dataclass(frozen=True)
class Rule:
    rule_id: str
    task: TaskType
    subject: str | None = None
    property: str | None = None
    object: str | None = None
    op: str = "EQ"
    expected: RuleValue = None
    minimum: float | None = None
    maximum: float | None = None
    count: CountValues | None = None
    attributes: tuple[AttributeTarget, ...] = ()

    @builtin_property
    def categories(self) -> tuple[str, ...]:
        if self.task is TaskType.ATTRIBUTE_COMBINATION:
            return tuple(item.subject for item in self.attributes)
        categories = []
        if self.subject:
            categories.append(self.subject)
        if self.object:
            categories.append(self.object)
        return tuple(categories)


@dataclass(frozen=True)
class MaskInstance:
    mask: np.ndarray
    score: float = 1.0
    box: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        array = np.asarray(self.mask, dtype=bool)
        if array.ndim != 2:
            raise ValueError(f"mask must be two-dimensional, got shape {array.shape}")
        if not array.any():
            raise ValueError("mask must contain at least one foreground pixel")
        score = float(self.score)
        if not isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"mask score must be in [0, 1], got {self.score!r}")
        object.__setattr__(self, "mask", array)
        object.__setattr__(self, "score", score)


@dataclass(frozen=True)
class AttributePrediction:
    value: str
    confidence: float | None = None
    per_instance: tuple[str, ...] = ()
    probabilities: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    task: TaskType
    satisfied: bool | None
    actual: Any
    expected: Any
    satisfaction_score: float | None
    violation_score: float | None
    evidence_valid: bool = True
    details: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        scores = (self.satisfaction_score, self.violation_score)
        if self.evidence_valid:
            if self.satisfied is None or any(score is None for score in scores):
                raise ValueError("valid rule evidence requires a decision and two scores")
        elif self.satisfied is not None or any(score is not None for score in scores):
            raise ValueError("invalid rule evidence must use UNKNOWN and null scores")
        for score in scores:
            if score is not None and (
                not isfinite(float(score)) or not 0.0 <= float(score) <= 1.0
            ):
                raise ValueError(f"rule score must be in [0, 1], got {score!r}")
        if all(score is not None for score in scores) and not np.isclose(
            float(self.satisfaction_score) + float(self.violation_score), 1.0
        ):
            raise ValueError("satisfaction_score and violation_score must sum to 1")

    @builtin_property
    def passed(self) -> bool:
        """Compatibility view for callers that previously consumed ``passed``."""

        return self.satisfied is True


@dataclass(frozen=True)
class PatchEvidence:
    """Image-level evidence retained from the LogSAD-style patch branch."""

    layer_scores: Mapping[str, float]
    aggregate_raw_score: float
    threshold: float
    calibrated_score: float
    anomaly: bool
    fusion_score: float | None = None
    fusion_threshold: float | None = None
    top_fraction: float | None = None
    bank_path: str | None = None
    layer_thresholds: Mapping[str, float] = field(default_factory=dict)
    layer_calibrated_scores: Mapping[str, float] = field(default_factory=dict)
    layer_fusion: str = "logsad_mean_map"
    primary_layer: int | None = None
    winning_layer: int | None = None
    full_image_score: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw_values = tuple(self.layer_scores.values()) + (
            self.aggregate_raw_score,
            self.threshold,
        )
        if any(
            not isfinite(float(value)) or not 0.0 <= float(value) <= 2.0
            for value in raw_values
        ):
            raise ValueError("patch raw scores and threshold must be in [0, 2]")
        if not isfinite(float(self.calibrated_score)) or not 0.0 <= float(
            self.calibrated_score
        ) <= 1.0:
            raise ValueError("calibrated patch score must be in [0, 1]")
        for label, value in (
            ("patch fusion score", self.fusion_score),
            ("patch fusion threshold", self.fusion_threshold),
        ):
            if value is not None and (
                not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{label} must be in [0, 1]")
        if self.top_fraction is not None and (
            not isfinite(float(self.top_fraction))
            or not 0.0 < float(self.top_fraction) <= 1.0
        ):
            raise ValueError("patch top fraction must be in (0, 1]")
        if self.layer_fusion != "logsad_mean_map":
            raise ValueError("unsupported patch layer fusion")
        if self.primary_layer is not None and self.primary_layer not in {
            6,
            12,
            18,
            24,
        }:
            raise ValueError("unsupported primary patch layer")
        if self.winning_layer is not None and self.winning_layer not in {
            6,
            12,
            18,
            24,
        }:
            raise ValueError("unsupported winning patch layer")
        if self.layer_thresholds and set(self.layer_thresholds) != set(
            self.layer_scores
        ):
            raise ValueError("patch layer thresholds must match layer scores")
        if self.layer_calibrated_scores and set(
            self.layer_calibrated_scores
        ) != set(self.layer_scores):
            raise ValueError("calibrated patch layers must match layer scores")
        if any(
            not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            for value in self.layer_calibrated_scores.values()
        ):
            raise ValueError("calibrated patch layer scores must be in [0, 1]")


@dataclass(frozen=True)
class InferenceReport:
    image_anomaly: bool | None
    image_anomaly_score: float | None
    logical_anomaly: bool | None
    anomaly_source: str | None
    categories: tuple[str, ...]
    mask_counts: Mapping[str, int]
    results: tuple[RuleResult, ...]
    filled_template: str | None
    patch_evidence: PatchEvidence | None = None
    rule_inference_enabled: bool = True
    execution_mode: str = "rule_only"

    def __post_init__(self) -> None:
        if self.image_anomaly_score is not None and (
            not isfinite(float(self.image_anomaly_score))
            or not 0.0 <= float(self.image_anomaly_score) <= 1.0
        ):
            raise ValueError("image anomaly score must be in [0, 1]")
        if self.anomaly_source not in {
            None,
            "rule",
            "structural",
            "mixed",
        }:
            raise ValueError("unsupported anomaly source")
        if self.execution_mode not in {
            "rule_only",
            "rule_patch",
            "patch_only",
        }:
            raise ValueError("unsupported execution mode")

    @builtin_property
    def is_anomaly(self) -> bool | None:
        return self.image_anomaly

    @builtin_property
    def final_anomaly_score(self) -> float | None:
        """Compatibility alias for callers predating image-level fusion."""

        return self.image_anomaly_score

    @builtin_property
    def status(self) -> str:
        if self.image_anomaly is None:
            return "UNKNOWN"
        if not self.image_anomaly:
            return "NORMAL"
        return {
            "rule": "RULE_ANOMALY",
            "structural": "STRUCTURAL_ANOMALY",
            "mixed": "MIXED_ANOMALY",
        }.get(self.anomaly_source, "ANOMALY")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("image_anomaly")
        result.pop("anomaly_source")
        for item in result["results"]:
            item["task"] = item["task"].value
        result["categories"] = list(self.categories)
        result["status"] = self.status
        result["anomaly"] = self.is_anomaly
        return result
