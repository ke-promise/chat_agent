from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError, APITimeoutError, AuthenticationError

from chat_agent.agent.provider import LLMProvider
from chat_agent.config import LLMConfig


class TimeoutCompletions:
    async def create(self, **kwargs):
        raise APITimeoutError(httpx.Request("POST", "https://example.test/v1/chat/completions"))


class AuthCompletions:
    async def create(self, **kwargs):
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        response = httpx.Response(401, request=request)
        raise AuthenticationError("bad key", response=response, body=None)


class ImageUrlRejectedThenSuccessCompletions:
    def __init__(self) -> None:
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            response = httpx.Response(
                400,
                request=request,
                json={"error": {"message": "messages[0]: unknown variant `image_url`, expected `text`"}},
            )
            raise APIStatusError("bad content", response=response, body=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))])


def _provider(completions) -> LLMProvider:
    provider = LLMProvider(
        LLMConfig(
            model="test-model",
            api_key="test-key",
            base_url="https://example.test/v1",
            timeout_seconds=1,
            max_tokens=100,
        )
    )
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return provider


def test_llm_config_supports_vision_and_profile_defaults() -> None:
    config = LLMConfig("model", "key", "https://example.test/v1", 1, 100, enable_vision=True, profile="main")

    assert config.enable_vision is True
    assert config.profile == "main"


@pytest.mark.asyncio
async def test_llm_provider_timeout_returns_friendly_message() -> None:
    result = await _provider(TimeoutCompletions()).chat([{"role": "user", "content": "你好"}])

    assert result.ok is False
    assert "超时" in result.content


@pytest.mark.asyncio
async def test_llm_provider_401_returns_friendly_message() -> None:
    result = await _provider(AuthCompletions()).chat([{"role": "user", "content": "你好"}])

    assert result.ok is False
    assert "API key" in result.content


@pytest.mark.asyncio
async def test_llm_provider_retries_image_url_rejection_as_text() -> None:
    completions = ImageUrlRejectedThenSuccessCompletions()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "https://example.test/a.jpg"}},
            ],
        }
    ]

    result = await _provider(completions).chat(messages)

    assert result.ok is True
    assert result.content == "ok"
    assert len(completions.calls) == 2
    assert isinstance(completions.calls[1]["messages"][0]["content"], str)
    assert completions.calls[1]["messages"][0]["content"].startswith("describe")
