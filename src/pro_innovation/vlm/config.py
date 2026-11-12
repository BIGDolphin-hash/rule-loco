"""Deployment configuration, separate from images, UTS and fixed task prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import isfinite
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit

from .contracts import VLMError

_DEFAULT_URLS = {
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
}
_KEY_ENVS = {
    "qwen": "DASHSCOPE_API_KEY", "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY",
}
_RESERVED_OPTIONS = {
    "model", "messages", "stream", "response_format", "system", "contents",
    "systemInstruction", "generationConfig", "output_config", "max_tokens",
    "max_completion_tokens", "maxOutputTokens", "responseMimeType", "responseJsonSchema",
}
_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "vlm" / "vlm.local.json"
)
_CONFIG_FIELDS = {
    "provider", "model", "api_key", "base_url", "structured_output", "timeout_seconds",
    "max_output_tokens", "request_options", "max_request_bytes",
}


def _local_settings() -> dict[str, Any]:
    configured = os.environ.get("VLM_CONFIG_FILE")
    if configured == "":
        return {}
    path = Path(configured).expanduser() if configured is not None else _DEFAULT_CONFIG_PATH
    if configured is None and not path.exists():
        return {}
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise VLMError("could not read VLM_CONFIG_FILE as a JSON configuration") from None
    if not isinstance(settings, dict) or set(settings) - _CONFIG_FIELDS:
        raise VLMError("local VLM configuration must be an object containing supported deployment fields")
    return settings


@dataclass(frozen=True)
class VLMConfig:
    provider: str = "qwen"
    model: str = ""
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    structured_output: str = "json_schema"
    timeout_seconds: float = 180.0
    max_output_tokens: int | None = None
    request_options: Mapping[str, Any] = field(default_factory=dict, repr=False)
    max_request_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or self.provider not in {*_DEFAULT_URLS, "openai-compatible"}:
            raise VLMError("VLM_PROVIDER must be qwen, openai, openai-compatible, anthropic or gemini")
        if not self.model and self.provider == "qwen":
            object.__setattr__(self, "model", "qwen3.8-max")
        if not isinstance(self.model, str) or not self.model or any(c.isspace() for c in self.model):
            raise VLMError("VLM_MODEL must be a nonempty model ID without whitespace")
        if not isinstance(self.api_key, str) or not self.api_key or any(c.isspace() for c in self.api_key):
            name = _KEY_ENVS.get(self.provider, "VLM_API_KEY")
            raise VLMError(f"set VLM_API_KEY or {name} to a nonempty key without whitespace")
        base_url = self.base_url or _DEFAULT_URLS.get(self.provider, "")
        if not isinstance(base_url, str):
            raise VLMError("VLM_BASE_URL must be an HTTPS API base URL")
        try:
            parsed = urlsplit(base_url)
            port = parsed.port
        except ValueError:
            raise VLMError("VLM_BASE_URL must be a valid HTTPS API base URL") from None
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise VLMError("VLM_BASE_URL must be an HTTPS API base URL")
        if port is not None and port < 1:
            raise VLMError("VLM_BASE_URL contains an invalid port")
        object.__setattr__(self, "base_url", base_url.rstrip("/"))
        if self.structured_output not in {"json_schema", "json_object", "prompt"}:
            raise VLMError("VLM_STRUCTURED_OUTPUT must be json_schema, json_object or prompt")
        if self.provider == "anthropic" and self.structured_output == "json_object":
            raise VLMError("Anthropic supports json_schema or prompt output mode")
        if (isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float))
                or not isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 600):
            raise VLMError("VLM_TIMEOUT_SECONDS must be finite and between 0 and 600")
        for name, value in (("max_output_tokens", self.max_output_tokens), ("max_request_bytes", self.max_request_bytes)):
            if value is not None and (type(value) is not int or value < 1):
                raise VLMError(f"{name} must be a positive integer")
        try:
            options = json.loads(json.dumps(dict(self.request_options), allow_nan=False))
        except (TypeError, ValueError):
            raise VLMError("VLM_REQUEST_OPTIONS must be a JSON object") from None
        if set(options) & _RESERVED_OPTIONS:
            raise VLMError("model options cannot override messages, model, output schema or token budget")
        # Copy caller-owned containers before retaining the options.
        object.__setattr__(self, "request_options", MappingProxyType(options))

    @classmethod
    def from_env(cls) -> VLMConfig:
        local = _local_settings()
        provider = os.environ.get("VLM_PROVIDER", local.get("provider", "qwen"))
        if not isinstance(provider, str):
            raise VLMError("VLM_PROVIDER must be a supported provider name")
        # Switching providers must not reuse another provider's saved credentials
        # or endpoint. Explicit environment settings always take precedence.
        if provider != local.get("provider", "qwen"):
            local = {}
        key_name = _KEY_ENVS.get(provider, "VLM_API_KEY")
        model = os.environ.get("VLM_MODEL", local.get("model", "qwen3.8-max" if provider == "qwen" else ""))
        base_url = os.environ.get("VLM_BASE_URL", "")
        if not base_url and provider == "qwen":
            base_url = os.environ.get("DASHSCOPE_BASE_URL", "")
        if not base_url:
            base_url = local.get("base_url", "")
        try:
            raw_options = os.environ.get("VLM_REQUEST_OPTIONS")
            options = json.loads(raw_options) if raw_options is not None else local.get("request_options", {})
            if not isinstance(options, dict):
                raise ValueError
            raw_timeout = os.environ.get("VLM_TIMEOUT_SECONDS")
            timeout = float(raw_timeout) if raw_timeout is not None else local.get("timeout_seconds", 180)
            output_tokens = os.environ.get("VLM_MAX_OUTPUT_TOKENS")
            request_bytes = os.environ.get("VLM_MAX_REQUEST_BYTES")
            return cls(
                provider=provider, model=model,
                api_key=os.environ.get("VLM_API_KEY", os.environ.get(key_name, local.get("api_key", ""))),
                base_url=base_url,
                structured_output=os.environ.get("VLM_STRUCTURED_OUTPUT", local.get("structured_output", "json_schema")),
                timeout_seconds=timeout,
                max_output_tokens=int(output_tokens) if output_tokens is not None else local.get("max_output_tokens"),
                max_request_bytes=int(request_bytes) if request_bytes is not None else local.get("max_request_bytes"),
                request_options=options,
            )
        except (ValueError, TypeError):
            raise VLMError("invalid VLM environment settings; check numeric fields and VLM_REQUEST_OPTIONS JSON") from None
