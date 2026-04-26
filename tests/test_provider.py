from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError, AuthenticationError

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
