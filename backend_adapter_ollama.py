"""Ollama (OpenAI-compatible) backend adapter for HuBrIS.

Handles all HTTP communication with the configured backend endpoint.
Compatible with any OpenAI-compatible server: Ollama, LM Studio, etc.
"""

import json
import re
import struct
from typing import Any

import httpx

from log import get_logger

_log = get_logger("hubris.backend.ollama")


class OllamaAdapter:
    """Backend adapter for Ollama and any OpenAI-compatible endpoint."""

    ADAPTER_NAME = "ollama"

    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        api_key: str = "",
        timeout: float = 600.0,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def complete(self, system_prompt: str, user_prompt: str, model: str) -> str:
        """
        POST to /v1/chat/completions using SSE streaming and return the
        assembled content string.

        Streaming is used deliberately: non-streaming requests give Ollama
        zero bytes to send until the full response is buffered, so a slow
        cold-start will trip any hang watchdog sitting in front of the endpoint.
        With streaming, each token resets the watchdog clock.

        Raises httpx.HTTPError on HTTP or transport failure.
        Raises ValueError when the model streams zero content.
        """
        url = f"{self._endpoint}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "think": False,
        }
        # read timeout: gap between consecutive SSE chunks. Once the model is
        # generating, each token resets this clock. Set to self._timeout so
        # long model cold-starts (VRAM load) are covered before the first token.
        timeout = httpx.Timeout(
            connect=10.0,
            read=self._timeout,
            write=30.0,
            pool=5.0,
        )
        pieces: list[str] = []
        with httpx.stream(
            "POST", url, json=payload, headers=self._headers(), timeout=timeout
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                try:
                    chunk = json.loads(data_str)
                    if "choices" not in chunk:
                        continue
                    delta = chunk["choices"][0].get("delta", {})
                    piece = delta.get("content") or ""
                    if piece:
                        pieces.append(piece)
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    pass  # Malformed or non-data line - skip silently

        if not pieces:
            _log.warning("LLM streamed 0 content pieces (model=%s)", model)
            raise ValueError("LLM returned no content")

        content = "".join(pieces).strip()
        # Qwen3/Qwen3.5 emit <think>...</think> blocks regardless of the
        # "think" payload flag. Strip ALL occurrences (leading, trailing, or
        # mid-content) so callers always receive only the actual response.
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content

    def embed(self, text: str, model: str) -> bytes | None:
        """
        POST to /v1/embeddings and return the embedding as serialized float32
        bytes compatible with sqlite-vec, or None on any error.

        Errors are logged at DEBUG level so a missing or unloaded model is
        silent and does not interrupt the watcher pipeline.
        """
        url = f"{self._endpoint}/v1/embeddings"
        payload = {"model": model, "input": text}
        try:
            response = httpx.post(
                url, json=payload, headers=self._headers(), timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            floats: list[float] = data["data"][0]["embedding"]
            return struct.pack(f"{len(floats)}f", *floats)
        except Exception as exc:
            _log.debug("embed() failed (%s): %s", type(exc).__name__, exc)
            return None

    def list_models(self) -> list[str]:
        """
        Return a sorted list of model names available at the configured endpoint.

        Calls GET /v1/models (OpenAI-compatible). Returns [] on any error
        (endpoint unreachable, timeout, unexpected response shape) so callers
        can safely fall back to free-text entry.

        Uses a short 5-second connect+read timeout - this is a UI preflight
        call, not an inference call.
        """
        url = f"{self._endpoint}/v1/models"
        try:
            response = httpx.get(
                url,
                headers=self._headers(),
                timeout=httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0),
            )
            response.raise_for_status()
            data = response.json()
            names = [m["id"] for m in data.get("data", []) if "id" in m]
            return sorted(names)
        except Exception as exc:
            _log.debug("list_models() failed (%s): %s", type(exc).__name__, exc)
            return []
