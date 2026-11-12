"""Shared bounded JSON transport for all model protocols."""

from __future__ import annotations

from dataclasses import dataclass
from http.client import HTTPException
import json
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .contracts import VLMError


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True)
class JsonTransport:
    timeout_seconds: float = 180.0
    max_request_bytes: int = 32 * 1024 * 1024
    max_response_bytes: int = 8 * 1024 * 1024

    def post(self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ValueError, TypeError):
            raise VLMError("model request is not valid JSON") from None
        if len(body) > self.max_request_bytes:
            raise VLMError("model request exceeds configured body limit; use fewer images or a larger-capacity endpoint")
        request = Request(url, data=body, headers={
            "Content-Type": "application/json", "Accept": "application/json", **headers,
        }, method="POST")
        opener = build_opener(_NoRedirects())
        for attempt in range(3):
            try:
                with opener.open(request, timeout=self.timeout_seconds) as response:
                    raw = response.read(self.max_response_bytes + 1)
                break
            except HTTPError as exc:
                status = exc.code
                exc.close()
                if status in {429, 500, 502, 503, 504, 529} and attempt < 2:
                    time.sleep(0.5 * 2**attempt)
                    continue
                raise VLMError(
                    f"VLM API returned HTTP {status}; check credentials, model access, base URL and quota"
                ) from None
            except (URLError, OSError, HTTPException):
                # A timeout can occur after billing. Avoid resubmitting it silently.
                raise VLMError("VLM API connection failed or timed out; check network and base URL") from None
        if len(raw) > self.max_response_bytes:
            raise VLMError("model response exceeds configured size limit")
        try:
            result = json.loads(raw)
        except (TypeError, ValueError):
            raise VLMError("model API returned invalid JSON") from None
        if not isinstance(result, dict) or "error" in result:
            raise VLMError("model API returned an invalid response or provider error")
        return result
