from types import SimpleNamespace

import pytest

from app.config import Settings
from app.services.llm import (
    ClaudeAPIClient,
    FallbackLLMClient,
    GeminiAPIClient,
    LLMError,
    LLMFactory,
    LLMResponse,
    OllamaClient,
)


class FakeClaudeMessages:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClaude:
    def __init__(self, messages: FakeClaudeMessages) -> None:
        self.messages = messages


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


class FakeHTTPClient:
    def __init__(self, response: FakeHTTPResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    def post(self, url: str, json: dict[str, object]) -> FakeHTTPResponse:
        self.calls.append((url, json))
        return self.response


class FakeGeminiHTTPClient:
    def __init__(self, response: FakeHTTPResponse) -> None:
        self.response = response
        self.url = ""
        self.headers: dict[str, str] = {}
        self.json: dict[str, object] = {}

    def post(
        self,
        url: str,
        headers: dict[str, str],
        json: dict[str, object],
    ) -> FakeHTTPResponse:
        self.url = url
        self.headers = headers
        self.json = json
        return self.response


def claude_response(text: str = "An answer") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=4, output_tokens=6),
    )


def test_claude_client_translates_request_and_response() -> None:
    messages = FakeClaudeMessages([claude_response()])
    settings = Settings(anthropic_api_key="test-key", anthropic_model="claude-test")
    result = ClaudeAPIClient(settings, client=FakeClaude(messages)).generate("System", "Question")

    assert result == LLMResponse("An answer", 10, result.latency_ms, "claude-test", "claude")
    assert messages.calls[0]["system"] == "System"
    assert messages.calls[0]["messages"] == [{"role": "user", "content": "Question"}]
    assert "temperature" not in messages.calls[0]


def test_claude_client_retries_transient_failures() -> None:
    messages = FakeClaudeMessages([RuntimeError("temporary"), claude_response("Recovered")])
    settings = Settings(anthropic_api_key="test-key", llm_retries=1)

    result = ClaudeAPIClient(settings, client=FakeClaude(messages)).generate("System", "Question")

    assert result.text == "Recovered"
    assert len(messages.calls) == 2


def test_ollama_client_parses_response_and_usage() -> None:
    client = FakeHTTPClient(
        FakeHTTPResponse(
            {
                "message": {"content": "Local answer"},
                "prompt_eval_count": 3,
                "eval_count": 7,
            }
        )
    )
    settings = Settings(ollama_url="http://ollama.test", ollama_model="local-test")

    result = OllamaClient(settings, client=client).generate("System", "Question")

    assert result.text == "Local answer"
    assert result.tokens_used == 10
    assert client.calls[0][0] == "http://ollama.test/api/chat"


def test_gemini_client_parses_response_and_sends_generation_config() -> None:
    client = FakeGeminiHTTPClient(
        FakeHTTPResponse(
            {
                "candidates": [{"content": {"parts": [{"text": "Gemini answer"}]}}],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 8},
            }
        )
    )
    settings = Settings(
        gemini_api_key="test-key",
        gemini_url="https://gemini.test/v1beta",
        gemini_model="gemini-test",
    )

    result = GeminiAPIClient(settings, client=client).generate("System", "Question")

    assert result.text == "Gemini answer"
    assert result.tokens_used == 13
    assert client.url.endswith("/models/gemini-test:generateContent")
    assert client.headers == {"x-goog-api-key": "test-key"}
    assert client.json["generationConfig"] == {
        "temperature": 0.7,
        "maxOutputTokens": 1024,
    }


def test_fallback_client_marks_fallback_response() -> None:
    class FailingClient:
        def generate(self, system_prompt: str, user_message: str) -> LLMResponse:
            raise LLMError("unavailable")

    class WorkingClient:
        def generate(self, system_prompt: str, user_message: str) -> LLMResponse:
            return LLMResponse("Local", 2, 1.0, "local", "ollama")

    result = FallbackLLMClient(FailingClient(), WorkingClient()).generate("System", "Question")

    assert result.text == "Local"
    assert result.used_fallback is True


def test_factory_requires_key_only_for_explicit_claude() -> None:
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        LLMFactory.get_client(Settings(llm_provider="claude"))

    assert isinstance(LLMFactory.get_client(Settings(llm_provider="ollama")), OllamaClient)
    assert isinstance(
        LLMFactory.get_client(Settings(llm_provider="gemini", gemini_api_key="test-key")),
        GeminiAPIClient,
    )


def test_clients_reject_empty_prompts() -> None:
    with pytest.raises(LLMError, match="cannot be empty"):
        OllamaClient(Settings(), client=FakeHTTPClient(FakeHTTPResponse({}))).generate(" ", "Question")