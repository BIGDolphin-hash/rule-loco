"""Narrow interfaces separating the core pipeline from local model runtimes."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .models import AttributePrediction, MaskInstance, PatchEvidence


class Segmenter(Protocol):
    def segment(
        self, image: Any, categories: Sequence[str]
    ) -> Mapping[str, Sequence[MaskInstance]]:
        """Segment each unique category once and return its object instances."""


class MaskProcessor(Protocol):
    def process(
        self,
        categories: Sequence[str],
        instances: Mapping[str, Sequence[MaskInstance]],
    ) -> Mapping[str, Sequence[MaskInstance]]:
        """Clean masks and suppress duplicate proposals before semantic checks."""


class AttributeDetector(Protocol):
    def detect(
        self,
        image: Any,
        subject: str,
        property_name: str,
        instances: Sequence[MaskInstance],
    ) -> AttributePrediction:
        """Infer one property without consulting the standard template value."""


class PatchAnomalyScorer(Protocol):
    def score(self, image: Any) -> PatchEvidence:
        """Return full-image patch evidence."""


class StandardTemplateGenerator(Protocol):
    def generate(self, image: Any, prompt: str) -> str:
        """Generate standard-template text from a normal reference image."""
