from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when an LLM cannot generate a response."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tokens_used: int | None
    latency_ms: float
    model: str
    provider: str
    used_fallback: bool = False


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_message: str) -> LLMResponse: ...


def _validate_prompts(system_prompt: str, user_message: str) -> None:
    if not system_prompt.strip() or not user_message.strip():
        raise LLMError("System prompt and user message cannot be empty")


class ClaudeAPIClient:
    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        if client is None:
            if not self.settings.anthropic_api_key:
                raise LLMError("ANTHROPIC_API_KEY is required for Claude")
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise LLMError("Install the anthropic package to use Claude") from exc
            client = Anthropic(
                api_key=self.settings.anthropic_api_key,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=0,
            )
        self.client = client

    def generate(self, system_prompt: str, user_message: str) -> LLMResponse:
        _validate_prompts(system_prompt, user_message)
        started = time.perf_counter()
        for attempt in range(self.settings.llm_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.settings.anthropic_model,
                    max_tokens=self.settings.llm_max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                text = "".join(
                    block.text for block in response.content if getattr(block, "type", "") == "text"
                ).strip()
                if not text:
                    raise LLMError("Claude returned an empty response")
                usage = getattr(response, "usage", None)
                token_values = (
                    getattr(usage, "input_tokens", None),
                    getattr(usage, "output_tokens", None),
                )
                tokens_used = sum(value for value in token_values if isinstance(value, int)) or None
                result = LLMResponse(
                    text=text,
                    tokens_used=tokens_used,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    model=self.settings.anthropic_model,
                    provider="claude",
                )
                logger.info("LLM response provider=claude model=%s tokens=%s", result.model, tokens_used)
                return result
            except LLMError:
                raise
            except Exception as exc:
                if attempt == self.settings.llm_retries:
                    logger.warning("Claude generation failed: %s", exc)
                    raise LLMError(f"Claude generation failed: {exc}") from exc
                logger.warning("Claude attempt %s failed; retrying", attempt + 1)
        raise LLMError("Claude generation failed")


class GeminiAPIClient:
    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is required for Gemini")
        self.client = client or httpx.Client(timeout=self.settings.llm_timeout_seconds)

    def generate(self, system_prompt: str, user_message: str) -> LLMResponse:
        _validate_prompts(system_prompt, user_message)
        started = time.perf_counter()
        for attempt in range(self.settings.llm_retries + 1):
            try:
                response = self.client.post(
                    f"{self.settings.gemini_url.rstrip('/')}/models/"
                    f"{self.settings.gemini_model}:generateContent",
                    headers={"x-goog-api-key": self.settings.gemini_api_key},
                    json={
                        "system_instruction": {"parts": [{"text": system_prompt}]},
                        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
                        "generationConfig": {
                            "temperature": self.settings.llm_temperature,
                            "maxOutputTokens": self.settings.llm_max_tokens,
                        },
                    },
                )
                response.raise_for_status()
                payload = response.json()
                candidates = payload.get("candidates", [])
                parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
                text = "".join(
                    str(part.get("text", "")) for part in parts if isinstance(part, dict)
                ).strip()
                if not text:
                    raise LLMError("Gemini returned an empty response")
                usage = payload.get("usageMetadata", {})
                token_values = (
                    usage.get("promptTokenCount"),
                    usage.get("candidatesTokenCount"),
                )
                tokens_used = sum(value for value in token_values if isinstance(value, int)) or None
                result = LLMResponse(
                    text=text,
                    tokens_used=tokens_used,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    model=self.settings.gemini_model,
                    provider="gemini",
                )
                logger.info("LLM response provider=gemini model=%s tokens=%s", result.model, tokens_used)
                return result
            except LLMError:
                raise
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text[:500]
                logger.warning("Gemini generation failed status=%s: %s", exc.response.status_code, detail)
                raise LLMError(
                    f"Gemini generation failed with HTTP {exc.response.status_code}: {detail}"
                ) from exc
            except Exception as exc:
                if attempt == self.settings.llm_retries:
                    logger.warning("Gemini generation failed: %s", exc)
                    raise LLMError(f"Gemini generation failed: {exc}") from exc
                logger.warning("Gemini attempt %s failed; retrying", attempt + 1)
        raise LLMError("Gemini generation failed")


class OllamaClient:
    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or httpx.Client(timeout=self.settings.llm_timeout_seconds)

    def generate(self, system_prompt: str, user_message: str) -> LLMResponse:
        _validate_prompts(system_prompt, user_message)
        started = time.perf_counter()
        for attempt in range(self.settings.llm_retries + 1):
            try:
                response = self.client.post(
                    f"{self.settings.ollama_url.rstrip('/')}/api/chat",
                    json={
                        "model": self.settings.ollama_model,
                        "stream": False,
                        "options": {"temperature": self.settings.llm_temperature},
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                    },
                )
                response.raise_for_status()
                payload = response.json()
                text = str(payload.get("message", {}).get("content", "")).strip()
                if not text:
                    raise LLMError("Ollama returned an empty response")
                prompt_tokens = payload.get("prompt_eval_count")
                output_tokens = payload.get("eval_count")
                tokens_used = (
                    prompt_tokens + output_tokens
                    if isinstance(prompt_tokens, int) and isinstance(output_tokens, int)
                    else None
                )
                result = LLMResponse(
                    text=text,
                    tokens_used=tokens_used,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    model=self.settings.ollama_model,
                    provider="ollama",
                )
                logger.info("LLM response provider=ollama model=%s tokens=%s", result.model, tokens_used)
                return result
            except LLMError:
                raise
            except Exception as exc:
                if attempt == self.settings.llm_retries:
                    logger.warning("Ollama generation failed: %s", exc)
                    raise LLMError(f"Ollama generation failed: {exc}") from exc
                logger.warning("Ollama attempt %s failed; retrying", attempt + 1)
        raise LLMError("Ollama generation failed")


class FallbackLLMClient:
    def __init__(self, primary: LLMClient, fallback: LLMClient) -> None:
        self.primary = primary
        self.fallback = fallback

    def generate(self, system_prompt: str, user_message: str) -> LLMResponse:
        try:
            return self.primary.generate(system_prompt, user_message)
        except LLMError as primary_error:
            logger.warning("Using fallback LLM after primary failure: %s", primary_error)
            try:
                response = self.fallback.generate(system_prompt, user_message)
            except LLMError as fallback_error:
                raise LLMError(
                    f"Primary and fallback LLMs failed: {primary_error}; {fallback_error}"
                ) from fallback_error
            return LLMResponse(
                text=response.text,
                tokens_used=response.tokens_used,
                latency_ms=response.latency_ms,
                model=response.model,
                provider=response.provider,
                used_fallback=True,
            )


class LLMFactory:
    @staticmethod
    def get_client(settings: Settings | None = None) -> LLMClient:
        settings = settings or get_settings()
        provider = settings.llm_provider.lower().replace("_", "-")
        if provider == "ollama":
            return OllamaClient(settings)
        if provider == "gemini":
            return GeminiAPIClient(settings)
        if provider == "claude":
            return ClaudeAPIClient(settings)
        if provider not in {"auto", "fallback"}:
            raise LLMError(f"Unknown LLM provider: {settings.llm_provider}")
        ollama = OllamaClient(settings)
        if settings.gemini_api_key:
            return FallbackLLMClient(GeminiAPIClient(settings), ollama)
        if settings.anthropic_api_key:
            return FallbackLLMClient(ClaudeAPIClient(settings), ollama)
        return ollama