"""Provider-neutral request types for visual language models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from ..errors import ProInnovationError


class VLMError(ProInnovationError, RuntimeError):
    """A model configuration, request, or response is invalid."""


@dataclass(frozen=True)
class VisionImage:
    """An encoded image with native dimensions, independent of an API protocol."""

    data: str = field(repr=False)
    width: int
    height: int
    media_type: str = "image/png"

    @property
    def byte_size(self) -> int:
        return len(self.data) * 3 // 4 - (len(self.data) - len(self.data.rstrip("=")))


@dataclass(frozen=True)
class VisionMessage:
    role: Literal["user", "assistant"]
    content: tuple[str | VisionImage, ...]


@dataclass(frozen=True)
class VisionRequest:
    system_prompt: str
    messages: tuple[VisionMessage, ...]
    json_schema: Mapping[str, Any]
    schema_name: str = "structured_output"


@runtime_checkable
class VisionLanguageClient(Protocol):
    """Implement this one method to connect an additional model protocol."""

    def complete(self, request: VisionRequest) -> str:
        """Return only the final text; raise VLMError on failure or truncation."""
        ...
