"""Thin async wrapper around the Google Gen AI SDK (Gemini).

Adds: bounded concurrency, retry with exponential backoff on 429/5xx, structured JSON
output validated into Pydantic models, batched embeddings, and usage accounting.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from .config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
_EMBED_BATCH = 100


class GeminiError(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise GeminiError("GEMINI_API_KEY is not set")
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model
        self.embedding_model = settings.gemini_embedding_model
        self.max_retries = settings.gemini_max_retries
        self._sem = asyncio.Semaphore(max(1, settings.gemini_concurrency))
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    async def _with_retry(self, fn, what: str):
        attempt = 0
        while True:
            try:
                async with self._sem:
                    return await fn()
            except errors.APIError as e:
                code = getattr(e, "code", None)
                if code not in _RETRYABLE_CODES or attempt >= self.max_retries:
                    raise GeminiError(f"Gemini {what} failed ({code}): {e}") from e
            except (ValidationError, ValueError) as e:  # malformed structured output
                if attempt >= min(2, self.max_retries):
                    raise GeminiError(f"Gemini {what} returned invalid output: {e}") from e
            delay = min(60.0, (2**attempt) + random.uniform(0, 1))
            attempt += 1
            log.warning("Gemini %s failed, retry %d in %.1fs", what, attempt, delay)
            await asyncio.sleep(delay)

    async def generate_json(self, prompt: str, schema: type[T], system: str,
                            temperature: float = 0.2, max_output_tokens: int = 8192) -> T:
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=schema,
        )

        async def call() -> T:
            resp = await self._client.aio.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
            self.calls += 1
            usage = resp.usage_metadata
            if usage:
                self.input_tokens += usage.prompt_token_count or 0
                self.output_tokens += (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
            parsed = resp.parsed
            if isinstance(parsed, schema):
                return parsed
            text = resp.text
            if not text:
                reason = resp.candidates[0].finish_reason if resp.candidates else "no candidates"
                raise ValueError(f"empty response (finish_reason={reason})")
            return schema.model_validate_json(text)

        return await self._with_retry(call, "generate")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH):
            batch = texts[i : i + _EMBED_BATCH]

            async def call(batch=batch) -> list[list[float]]:
                resp = await self._client.aio.models.embed_content(
                    model=self.embedding_model,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
                )
                self.calls += 1
                return [list(e.values or []) for e in (resp.embeddings or [])]

            vectors = await self._with_retry(call, "embed")
            if len(vectors) != len(batch):
                raise GeminiError(f"embedding count mismatch: sent {len(batch)}, got {len(vectors)}")
            out.extend(vectors)
        return out
