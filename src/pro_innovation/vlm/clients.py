"""Model protocol adapters. No CQT/NRT-specific logic belongs here."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from urllib.parse import quote, urlsplit

from .config import VLMConfig
from .contracts import VLMError, VisionImage, VisionLanguageClient, VisionRequest
from .transport import JsonTransport

_MIB = 1024 * 1024


def _images(request: VisionRequest) -> list[VisionImage]:
    return [part for message in request.messages for part in message.content if isinstance(part, VisionImage)]


class _HttpClient:
    providers: frozenset[str]

    def __init__(self, config: VLMConfig, transport: JsonTransport | None = None) -> None:
        if config.provider not in self.providers:
            raise VLMError("selected provider does not match the client protocol")
        self.config = config
        default_limit = 32 * _MIB
        if config.provider == "qwen":
            host = urlsplit(config.base_url).hostname or ""
            default_limit = 16 * _MIB if host.startswith("dashscope") else 64 * _MIB
        elif config.provider == "gemini":
            default_limit = 20 * _MIB  # Conservative inline-data budget.
        self.transport = transport if transport is not None else JsonTransport(
            timeout_seconds=config.timeout_seconds,
            max_request_bytes=config.max_request_bytes or default_limit,
        )

    def _check_images(self, request: VisionRequest) -> None:
        images = _images(request)
        if self.config.provider == "qwen":
            if len(images) > 2048 or any(image.width * image.height > 16_777_216 for image in images):
                raise VLMError("images exceed the selected Qwen deployment's count or 16M pixel limit")
            if any(len(image.data) + len(f"data:{image.media_type};base64,") > 20 * _MIB for image in images):
                raise VLMError("an image exceeds Qwen's 20 MiB encoded limit")
        if self.config.provider == "openai" and any(image.byte_size > 8 * _MIB for image in images):
            raise VLMError("an image exceeds the OpenAI Chat Completions 8 MiB image budget")
        if self.config.provider == "anthropic" and any(image.byte_size > 5 * _MIB for image in images):
            raise VLMError("an image exceeds the Anthropic adapter's 5 MiB image budget")


class ChatCompletionsClient(_HttpClient):
    """OpenAI-compatible protocol; provider-specific options stay in this adapter."""

    providers = frozenset({"qwen", "openai", "openai-compatible"})

    def complete(self, request: VisionRequest) -> str:
        self._check_images(request)
        messages = [{"role": "system", "content": request.system_prompt}]
        for message in request.messages:
            content = []
            for part in message.content:
                if isinstance(part, str):
                    content.append({"type": "text", "text": part})
                else:
                    content.append({"type": "image_url", "image_url": {
                        "url": f"data:{part.media_type};base64,{part.data}",
                    }})
            messages.append({"role": message.role, "content": content})
        payload: dict[str, Any] = {"model": self.config.model, "messages": messages, "stream": False}
        if self.config.structured_output == "json_schema":
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": request.schema_name, "strict": True, "schema": deepcopy(dict(request.json_schema)),
            }}
        elif self.config.structured_output == "json_object":
            payload["response_format"] = {"type": "json_object"}
        if self.config.provider == "qwen":
            payload.update(enable_thinking=False, temperature=0, vl_high_resolution_images=True)
        if self.config.max_output_tokens is not None:
            key = "max_completion_tokens" if self.config.provider == "openai" else "max_tokens"
            payload[key] = self.config.max_output_tokens
        payload.update(deepcopy(dict(self.config.request_options)))
        result = self.transport.post(
            self.config.base_url + "/chat/completions",
            {"Authorization": "Bearer " + self.config.api_key}, payload,
        )
        try:
            choices = result["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise VLMError("model must return exactly one completion")
            choice = choices[0]
            message = choice["message"]
            if choice.get("finish_reason") != "stop" or message.get("refusal"):
                raise VLMError("model response was refused, truncated or incomplete")
            text = message["content"]
            if isinstance(text, list):
                if any(part.get("type") != "text" for part in text):
                    raise VLMError("model returned unsupported content blocks")
                text = "".join(part["text"] for part in text)
            if not isinstance(text, str) or not text.strip():
                raise VLMError("model returned no final text")
            return text
        except (KeyError, TypeError, IndexError, AttributeError):
            raise VLMError("invalid Chat Completions response") from None


class AnthropicClient(_HttpClient):
    """Anthropic Messages API with native image and schema serialization."""

    providers = frozenset({"anthropic"})

    def complete(self, request: VisionRequest) -> str:
        self._check_images(request)
        messages = []
        for message in request.messages:
            content = []
            for part in message.content:
                if isinstance(part, str):
                    content.append({"type": "text", "text": part})
                else:
                    content.append({"type": "image", "source": {
                        "type": "base64", "media_type": part.media_type, "data": part.data,
                    }})
            messages.append({"role": message.role, "content": content})
        payload = {
            "model": self.config.model, "system": request.system_prompt, "messages": messages,
            "max_tokens": self.config.max_output_tokens or 16384, "stream": False,
            **deepcopy(dict(self.config.request_options)),
        }
        if self.config.structured_output == "json_schema":
            payload["output_config"] = {"format": {
                "type": "json_schema", "schema": deepcopy(dict(request.json_schema)),
            }}
        result = self.transport.post(
            self.config.base_url + "/messages",
            {"x-api-key": self.config.api_key, "anthropic-version": "2026-10-21"}, payload,
        )
        try:
            if result.get("stop_reason") != "end_turn":
                raise VLMError("Anthropic response was refused, truncated or incomplete")
            blocks = result["content"]
            if not isinstance(blocks, list):
                raise VLMError("invalid Anthropic content")
            if any(block.get("type") not in {"text", "thinking", "redacted_thinking"} for block in blocks):
                raise VLMError("Anthropic returned unsupported content blocks")
            text = "".join(block["text"] for block in blocks if block["type"] == "text")
            if not text.strip():
                raise VLMError("Anthropic returned no final text")
            return text
        except (KeyError, TypeError, AttributeError):
            raise VLMError("invalid Anthropic response") from None


class GeminiClient(_HttpClient):
    """Gemini generateContent API; API keys stay in headers, never URLs."""

    providers = frozenset({"gemini"})

    def complete(self, request: VisionRequest) -> str:
        self._check_images(request)
        contents = []
        for message in request.messages:
            parts = []
            for part in message.content:
                if isinstance(part, str):
                    parts.append({"text": part})
                else:
                    parts.append({"inlineData": {"mimeType": part.media_type, "data": part.data}})
            contents.append({"role": "model" if message.role == "assistant" else "user", "parts": parts})
        generation = deepcopy(dict(self.config.request_options))
        if self.config.structured_output != "prompt":
            generation["responseMimeType"] = "application/json"
        if self.config.structured_output == "json_schema":
            generation["responseJsonSchema"] = deepcopy(dict(request.json_schema))
        if self.config.max_output_tokens is not None:
            generation["maxOutputTokens"] = self.config.max_output_tokens
        payload = {
            "systemInstruction": {"parts": [{"text": request.system_prompt}]},
            "contents": contents, "generationConfig": generation,
        }
        model = quote(self.config.model.removeprefix("models/"), safe="")
        result = self.transport.post(
            self.config.base_url + f"/models/{model}:generateContent",
            {"x-goog-api-key": self.config.api_key}, payload,
        )
        try:
            candidates = result["candidates"]
            if not isinstance(candidates, list) or len(candidates) != 1:
                raise VLMError("Gemini must return exactly one candidate")
            candidate = candidates[0]
            if candidate.get("finishReason") != "STOP":
                raise VLMError("Gemini response was refused, truncated or incomplete")
            parts = candidate["content"]["parts"]
            if not isinstance(parts, list) or any("text" not in part for part in parts):
                raise VLMError("Gemini returned unsupported content parts")
            text = "".join(part["text"] for part in parts if not part.get("thought", False))
            if not text.strip():
                raise VLMError("Gemini returned no final text")
            return text
        except (KeyError, TypeError, IndexError, AttributeError):
            raise VLMError("invalid Gemini response") from None


def create_vlm_client(config: VLMConfig | None = None) -> VisionLanguageClient:
    settings = config if config is not None else VLMConfig.from_env()
    if settings.provider == "anthropic":
        return AnthropicClient(settings)
    if settings.provider == "gemini":
        return GeminiClient(settings)
    return ChatCompletionsClient(settings)
