"""Compatibility entry point; new integrations should use pro_innovation.vlm."""

from __future__ import annotations

import os

from ..vlm import ChatCompletionsClient, VLMConfig
from .errors import TemplateGenerationError

MODEL = "qwen3.8-max"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class QwenClient(ChatCompletionsClient):
    """Keep the original constructor; complete() now uses VisionRequest."""

    def __init__(
        self, api_key: str, base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 180.0,
    ) -> None:
        super().__init__(VLMConfig(
            provider="qwen", model=MODEL, api_key=api_key,
            base_url=base_url, timeout_seconds=timeout_seconds,
        ))

    @classmethod
    def from_env(cls) -> QwenClient:
        return cls(
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            base_url=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
        )
