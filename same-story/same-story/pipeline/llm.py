"""Thin wrappers around the Gemini REST API and any OpenAI-compatible API (e.g. OpenRouter)."""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import requests

log = logging.getLogger(__name__)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(Exception):
    pass


class LLMFatal(LLMError):
    """Problems retrying can't fix: bad key, wrong model name, no access."""


class LLM:
    def __init__(self, provider: str, model: str, api_key: str, base_url: Optional[str] = None,
                 sleep: Callable[[float], None] = time.sleep, session=None):
        if provider not in ("gemini", "openai_compat"):
            raise LLMError(f"Unknown provider: {provider}")
        if not api_key:
            raise LLMError("No API key supplied")
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self._sleep = sleep
        self._http = session or requests

    @property
    def can_embed(self) -> bool:
        return self.provider == "gemini"

    # ---- low-level request with retry -------------------------------------------------
    def _post(self, url: str, headers: dict, payload: dict, attempts: int = 4) -> dict:
        last = ""
        for attempt in range(attempts):
            try:
                resp = self._http.post(url, headers=headers, json=payload, timeout=60)
            except requests.RequestException as exc:
                last = str(exc)
                resp = None
            if resp is not None:
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise LLMError(f"Invalid JSON from API: {exc}")
                last = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code in (401, 403, 404):
                    raise LLMFatal(last)
                if resp.status_code not in RETRY_STATUS:
                    raise LLMError(last)
                wait = _retry_after(resp)
            else:
                wait = None
            if attempt < attempts - 1:
                self._sleep(wait if wait is not None else min(60, 5 * 2 ** attempt))
        raise LLMError(f"Gave up after {attempts} attempts. Last error: {last}")

    # ---- text generation ---------------------------------------------------------------
    def generate_json(self, system: str, user: str) -> str:
        if self.provider == "gemini":
            return self._gemini_generate(system, user)
        return self._openai_generate(system, user)

    def _gemini_generate(self, system: str, user: str) -> str:
        url = f"{GEMINI_BASE}/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
        }
        data = self._post(url, headers, payload)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"No usable text in response (possibly blocked): {str(data)[:200]}")
        if not text.strip():
            raise LLMError("Empty response")
        return text

    def _openai_generate(self, system: str, user: str) -> str:
        if not self.base_url:
            raise LLMError("base_url is required for openai_compat provider")
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
        }
        try:
            data = self._post(url, headers, payload)
        except LLMError as exc:
            if "HTTP 400" not in str(exc):
                raise
            # some free models don't support JSON mode: retry with a fresh payload without it
            plain = {k: v for k, v in payload.items() if k != "response_format"}
            data = self._post(url, headers, plain)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"No usable text in response: {str(data)[:200]}")

    # ---- embeddings (Gemini only) ------------------------------------------------------
    def embed(self, texts: list, model: str = "gemini-embedding-001") -> list:
        if not self.can_embed:
            raise LLMError("Embeddings are only wired up for the gemini provider")
        url = f"{GEMINI_BASE}/models/{model}:batchEmbedContents"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        vectors = []
        for start in range(0, len(texts), 100):
            chunk = texts[start:start + 100]
            payload = {"requests": [
                {"model": f"models/{model}", "content": {"parts": [{"text": t}]}, "taskType": "SEMANTIC_SIMILARITY"}
                for t in chunk
            ]}
            data = self._post(url, headers, payload)
            try:
                vectors.extend(e["values"] for e in data["embeddings"])
            except (KeyError, TypeError):
                raise LLMError("Unexpected embedding response")
            if start + 100 < len(texts):
                self._sleep(2)
        if len(vectors) != len(texts):
            raise LLMError("Embedding count mismatch")
        return vectors


def _retry_after(resp) -> Optional[float]:
    value = resp.headers.get("Retry-After") if getattr(resp, "headers", None) else None
    try:
        return min(float(value), 120) if value else None
    except ValueError:
        return None
