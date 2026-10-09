"""Lightweight attributes and routing to an optional semantic detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .errors import ExecutionError
from .interfaces import AttributeDetector
from .models import AttributePrediction, MaskInstance

_COLOR_PALETTE = {
    "black": (20, 20, 20),
    "white": (235, 235, 235),
    "gray": (128, 128, 128),
    "red": (220, 35, 35),
    "green": (35, 170, 65),
    "blue": (45, 90, 210),
    "yellow": (225, 205, 40),
    "light_yellow": (245, 230, 140),
    "dark_yellow": (185, 145, 20),
    "orange": (230, 125, 35),
    "purple": (140, 70, 170),
    "brown": (125, 80, 45),
}


def attribute_vocabulary_key(subject: str, property_name: str) -> str:
    """Return the required subject-scoped vocabulary field name."""

    owner = str(subject).strip()
    attribute = property_name.split(".", 1)[-1].strip().lower()
    if not owner or not attribute:
        raise ValueError("attribute vocabulary keys require a subject and attribute")
    return f"{owner}_{attribute}"


def validate_color_candidates(
    candidates: Sequence[str] | None,
) -> tuple[str, ...]:
    """Return normalized configured colors and reject implicit fallbacks."""

    if candidates is None or isinstance(candidates, (str, bytes)):
        raise ValueError("attribute.color requires a configured color candidate list")
    normalized = tuple(dict.fromkeys(str(label).strip().lower() for label in candidates))
    if not normalized or any(not label for label in normalized):
        raise ValueError("attribute.color requires at least one configured color")
    unknown = [label for label in normalized if label not in _COLOR_PALETTE]
    if unknown:
        raise ValueError(
            "unsupported configured colors: " + ", ".join(sorted(unknown))
        )
    return normalized


def _as_rgb_array(image: Any) -> np.ndarray:
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return np.asarray(opened.convert("RGB"))
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ExecutionError(f"expected an RGB image, got shape {array.shape}")
    return array[:, :, :3]


def _vote_probabilities(
    values: Sequence[str], instances: Sequence[MaskInstance]
) -> dict[str, float]:
    weights = [item.score for item in instances]
    if sum(weights) <= 0.0:
        weights = [1.0] * len(instances)
    totals: dict[str, float] = {}
    for value, weight in zip(values, weights):
        totals[value] = totals.get(value, 0.0) + weight
    denominator = sum(totals.values())
    return {value: weight / denominator for value, weight in totals.items()}


class SimpleAttributeDetector:
    """Deterministic V1 detector for color and coarse shape."""

    supported_properties = frozenset({"color", "shape"})

    def __init__(self, *, color_candidates: Sequence[str] | None = None) -> None:
        self.color_candidates = (
            validate_color_candidates(color_candidates)
            if color_candidates is not None
            else ()
        )

    def detect(
        self,
        image: Any,
        subject: str,
        property_name: str,
        instances: Sequence[MaskInstance],
    ) -> AttributePrediction:
        del subject
        attribute = property_name.split(".", 1)[-1].lower()
        if attribute not in self.supported_properties:
            raise ExecutionError(f"simple detector does not support attribute.{attribute}")
        if not instances:
            raise ExecutionError("attribute inference found no segmented instances")
        if attribute == "color":
            if not self.color_candidates:
                raise ExecutionError(
                    "attribute.color requires candidates from the configured vocabulary"
                )
            rgb = _as_rgb_array(image)
            predictions = tuple(
                self._color(rgb, item.mask, self.color_candidates)
                for item in instances
            )
        else:
            predictions = tuple(self._shape(item.mask) for item in instances)
        observed_probabilities = _vote_probabilities(predictions, instances)
        value = max(
            observed_probabilities,
            key=lambda label: (
                observed_probabilities[label],
                -predictions.index(label),
            ),
        )
        probabilities = (
            {
                label: observed_probabilities.get(label, 0.0)
                for label in self.color_candidates
            }
            if attribute == "color"
            else observed_probabilities
        )
        return AttributePrediction(
            value=value,
            confidence=probabilities[value],
            per_instance=predictions,
            probabilities=probabilities,
        )

    @staticmethod
    def _color(
        rgb: np.ndarray, mask: np.ndarray, candidates: Sequence[str]
    ) -> str:
        if rgb.shape[:2] != mask.shape:
            raise ExecutionError(
                f"image and mask shapes differ: {rgb.shape[:2]} versus {mask.shape}"
            )
        mean = rgb[mask].astype(np.float64).mean(axis=0)
        return min(
            candidates,
            key=lambda label: float(
                np.linalg.norm(mean - np.asarray(_COLOR_PALETTE[label], dtype=np.float64))
            ),
        )

    @staticmethod
    def _shape(mask: np.ndarray) -> str:
        rows, columns = np.nonzero(mask)
        height = int(rows.max() - rows.min() + 1)
        width = int(columns.max() - columns.min() + 1)
        ratio = max(height, width) / max(min(height, width), 1)
        fill = np.count_nonzero(mask) / (height * width)
        if ratio >= 1.8:
            return "elongated"
        if fill >= 0.90:
            return "square"
        return "round"


class RoutedAttributeDetector:
    """Use simple image rules first and CLIP-like semantics for other fields."""

    def __init__(
        self,
        semantic_detector: AttributeDetector | None,
        simple_detector: SimpleAttributeDetector | None = None,
        *,
        vocabulary: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.vocabulary = {
            str(key): tuple(str(value) for value in values)
            for key, values in (vocabulary or {}).items()
        }
        self.simple = simple_detector
        self.semantic = semantic_detector

    def detect(
        self,
        image: Any,
        subject: str,
        property_name: str,
        instances: Sequence[MaskInstance],
    ) -> AttributePrediction:
        attribute = property_name.split(".", 1)[-1].lower()
        if attribute in SimpleAttributeDetector.supported_properties:
            detector = self.simple
            if detector is None:
                color_candidates = None
                if attribute == "color":
                    key = attribute_vocabulary_key(subject, property_name)
                    color_candidates = self.vocabulary.get(key)
                    if color_candidates is None:
                        raise ExecutionError(
                            f"attribute.color requires vocabulary field {key}"
                        )
                detector = SimpleAttributeDetector(
                    color_candidates=color_candidates
                )
            return detector.detect(image, subject, property_name, instances)
        if self.semantic is None:
            raise ExecutionError(
                f"attribute.{attribute} needs a semantic detector and vocabulary"
            )
        return self.semantic.detect(image, subject, property_name, instances)
