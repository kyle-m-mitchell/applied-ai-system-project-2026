"""Provider text generation for Cadence's voice (optional, offline-safe).

Mirrors the embeddings adapter: a provider-agnostic interface, a deterministic
offline fake for tests and key-free runs, and a real Gemini REST client that is
used lazily and never logs the key or prompt. Generated text is
non-deterministic, so it is never cached; the deterministic voice is always the
reproducible fallback.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.embeddings import _make_ssl_context


# The handbook researched "gemini-3.5-flash-lite", which the live model list does
# not expose; the current flash-lite line is 3.1 / -latest. The alias is robust to
# version rolls; any failure falls back to the deterministic voice.
TEXT_MODEL = "gemini-flash-lite-latest"

# One user/model exchange used as a few-shot example.
FewShot = Sequence[tuple[str, str]]


class TextGenerator(ABC):
    """Provider-agnostic short-text generator."""

    model_id: str

    @abstractmethod
    def generate(self, system: str, few_shot: FewShot, user: str) -> str:
        """Return a short completion for ``user`` given a system instruction."""


class FakeTextGenerator(TextGenerator):
    """Deterministic, offline generator for tests and key-free development.

    It does not really 'write'; it returns a fixed, grounded framing phrase so the
    voice pipeline can be exercised without a network call or hallucination risk.
    """

    model_id = "fake-generator-v1"

    def generate(self, system: str, few_shot: FewShot, user: str) -> str:
        return "Here's a little set that should fit the mood."


class GeminiTextGenerator(TextGenerator):
    """Real generator calling the Gemini REST ``generateContent`` endpoint.

    Reuses the embeddings adapter's approach: key in a header (never a URL, never
    logged), certifi TLS, and one bounded retry on rate limits. Model id and
    payload shape drift; verify against the live API before relying on output. Any
    failure raises, and the voice falls back to the deterministic renderer.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = TEXT_MODEL,
        *,
        timeout: float = 30.0,
        max_retries: int = 1,  # one bounded retry, matching the design's promise
        max_output_tokens: int = 200,
    ) -> None:
        self.model_id = model_id
        self._timeout = timeout
        self._max_retries = max_retries
        self._max_output_tokens = max_output_tokens
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set; provide a key via the environment "
                "(a git-ignored .env), never in code."
            )
        self._ssl_context = _make_ssl_context()

    def generate(self, system: str, few_shot: FewShot, user: str) -> str:
        contents: list[dict] = []
        for example_user, example_model in few_shot:
            contents.append({"role": "user", "parts": [{"text": example_user}]})
            contents.append({"role": "model", "parts": [{"text": example_model}]})
        contents.append({"role": "user", "parts": [{"text": user}]})
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"maxOutputTokens": self._max_output_tokens},
        }
        data = self._post("generateContent", payload)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as exc:  # blocked or empty response
            raise RuntimeError("Gemini generation returned no text") from exc

    def _post(self, method: str, payload: dict) -> dict:
        url = f"{self.BASE_URL}/models/{self.model_id}:{method}"
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(self._max_retries + 1):
            request = urllib.request.Request(url, data=body, method="POST")
            request.add_header("Content-Type", "application/json")
            request.add_header("x-goog-api-key", self._api_key)  # key in header, not URL
            try:
                with urllib.request.urlopen(
                    request, timeout=self._timeout, context=self._ssl_context
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:  # never leak the key or prompt text
                if exc.code in (429, 500, 503) and attempt < self._max_retries:
                    time.sleep(min(2**attempt, 20))
                    continue
                raise RuntimeError(f"Gemini generation failed: HTTP {exc.code}") from exc
        raise RuntimeError("Gemini generation failed after retries")
