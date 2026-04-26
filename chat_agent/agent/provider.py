"""OpenAI-compatible LLM Provider 封装。

本模块只负责和模型服务通信：创建 AsyncOpenAI 客户端、发送 chat completion 请求、
解析 tool_calls，并把认证、限流、超时等外部 API 错误转换成用户可读的友好文案。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)

from chat_agent.config import LLMConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMResult:
    """一次 LLM chat completion 调用的标准化结果。

    字段:
        content: 模型返回的自然语言正文。若模型只返回 tool_calls，这里可能为空字符串。
        tool_calls: OpenAI-compatible tool calling 的调用列表，每项包含 id、name、arguments
            和 raw_arguments，供 Reasoner 执行工具循环。
        ok: 是否调用成功。False 表示 provider 已经把异常转换成用户可读提示，调用方不需要
            再解析底层 SDK 异常。
    """

    content: str
    tool_calls: list[dict[str, Any]]
    ok: bool = True


class LLMProvider:
    """OpenAI-compatible Chat Completions 客户端封装。

    职责:
        - 根据 LLMConfig 创建 AsyncOpenAI client。
        - 支持 base_url/api_key/model/timeout/max_tokens。
        - 统一把 SDK、网络、认证、限流等错误转换成友好 LLMResult。
        - 保留 OpenAI tools/function calling 参数透传能力。

    说明:
        项目会创建 main 与 fast 两个 provider。二者代码完全相同，只通过 config.profile
        区分日志和模型职责。
    """

    def __init__(self, config: LLMConfig) -> None:
        """初始化 LLM provider。

        参数:
            config: 单个模型配置，包含 profile、model、api_key、base_url、timeout_seconds、
                max_tokens 和 enable_vision 等字段。
        """
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=1,
        )

    @property
    def profile(self) -> str:
        """返回当前 provider 的配置档位名称，例如 main 或 fast。"""
        return self.config.profile

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResult:
        """调用 OpenAI-compatible chat completions。

        参数:
            messages: 已经组装好的 OpenAI messages。content 可以是普通字符串，也可以是
                multimodal blocks 列表。
            tools: 可选工具 schema。传入时会设置 tool_choice="auto"，允许模型自主选择工具。

        返回:
            LLMResult。发生 401、429、timeout、网络错误等情况时不会抛给上层，而是返回
            ok=False 和一段中文友好提示。
        """
        logger.info("[llm.%s] Calling model=%s messages=%s tools=%s", self.profile, self.config.model, len(messages), bool(tools))
        try:
            kwargs: dict[str, Any] = {"model": self.config.model, "messages": messages}
            if self.config.max_tokens:
                kwargs["max_tokens"] = self.config.max_tokens
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            response = await self.client.chat.completions.create(**kwargs)
        except AuthenticationError:
            logger.warning("[llm.%s] authentication failed", self.profile)
            return LLMResult("模型这边没认出钥匙，请检查 API key 是否正确。", [], ok=False)
        except RateLimitError:
            logger.warning("[llm.%s] rate limited", self.profile)
            return LLMResult("模型服务有点忙，可能是限流或额度不足，我们稍后再试。", [], ok=False)
        except APITimeoutError:
            logger.warning("[llm.%s] request timed out", self.profile)
            return LLMResult("模型想太久超时啦，稍后我再陪你试一次。", [], ok=False)
        except APIConnectionError:
            logger.warning("[llm.%s] connection error", self.profile)
            return LLMResult("我现在够不到模型服务，网络可能在闹小脾气，稍后再试。", [], ok=False)
        except APIStatusError as exc:
            logger.warning("[llm.%s] status error HTTP %s", self.profile, exc.status_code)
            if exc.status_code == 401:
                content = "模型认证失败，请检查 API key。"
            elif exc.status_code == 429:
                content = "模型服务限流或额度不足，请稍后再试。"
            else:
                content = f"模型服务返回错误（HTTP {exc.status_code}），请稍后再试。"
            return LLMResult(content, [], ok=False)
        except OpenAIError:
            logger.warning("[llm.%s] API error", self.profile, exc_info=True)
            return LLMResult("调用模型时绊了一下，我已经记下日志，稍后再来。", [], ok=False)
        except Exception:
            logger.exception("[llm.%s] unexpected error", self.profile)
            return LLMResult("处理回复时遇到一点未知状况，先别急，我们稍后再试。", [], ok=False)

        message = response.choices[0].message
        calls: list[dict[str, Any]] = []
        for call in message.tool_calls or []:
            raw_args = call.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
            calls.append(
                {
                    "id": call.id,
                    "name": call.function.name,
                    "arguments": args,
                    "raw_arguments": raw_args,
                }
            )
        return LLMResult(content=message.content or "", tool_calls=calls, ok=True)


def make_multimodal_user_content(text: str, image_urls: list[str]) -> str | list[dict[str, Any]]:
    """构造 OpenAI-compatible 的多模态 user content。

    参数:
        text: 用户文本或图片 caption。可以为空。
        image_urls: 图片 URL 或 data URL 列表。

    返回:
        没有图片时返回纯字符串，保持普通文本请求最简；有图片时返回 content blocks，
        每张图片使用 {"type": "image_url"} 格式，兼容支持视觉能力的 OpenAI-like 模型。
    """
    if not image_urls:
        return text
    blocks: list[dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})
    for url in image_urls:
        blocks.append({"type": "image_url", "image_url": {"url": url}})
    return blocks
