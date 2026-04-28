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


def _status_error_detail(exc: APIStatusError) -> str:
    """从上游 HTTP 错误中提取适合写入日志的简短详情。"""
    try:
        data = exc.response.json()
    except Exception:
        data = getattr(exc.response, "text", "") or str(exc)
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            detail = error.get("message") or error.get("code") or error
        else:
            detail = data.get("message") or data
    else:
        detail = data
    text = str(detail).replace("\n", " ").strip()
    return text[:500]


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
    reasoning_content: str | None = None


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
        kwargs: dict[str, Any] = {"model": self.config.model, "messages": messages}
        if self.config.max_tokens:
            kwargs["max_tokens"] = self.config.max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
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
            return await _handle_status_error(self.client, self.profile, kwargs, messages, exc)
        except OpenAIError:
            logger.warning("[llm.%s] API error", self.profile, exc_info=True)
            return LLMResult("调用模型时绊了一下，我已经记下日志，稍后再来。", [], ok=False)
        except Exception:
            logger.exception("[llm.%s] unexpected error", self.profile)
            return LLMResult("处理回复时遇到一点未知状况，先别急，我们稍后再试。", [], ok=False)

        message = response.choices[0].message
        reasoning_content = _get_message_extra(message, "reasoning_content")
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
        return LLMResult(
            content=message.content or "",
            tool_calls=calls,
            ok=True,
            reasoning_content=reasoning_content if isinstance(reasoning_content, str) else None,
        )


async def _handle_status_error(
    client: Any,
    profile: str,
    kwargs: dict[str, Any],
    messages: list[dict[str, Any]],
    exc: APIStatusError,
) -> LLMResult:
    """处理模型服务 HTTP 错误，并在接口不支持图片块时尝试降级为纯文本重试。

    参数:
        client: OpenAI-compatible 客户端实例。
        profile: 当前模型档位名称，用于日志区分 main/fast。
        kwargs: 原始请求参数，重试时会复制并替换 messages。
        messages: 原始消息列表，用于判断是否包含 image_url。
        exc: SDK 抛出的 HTTP 状态异常。

    返回:
        标准化的 LLMResult。若重试成功则返回重试响应；否则返回友好错误提示。
    """
    detail = _status_error_detail(exc)
    logger.warning("[llm.%s] status error HTTP %s: %s", profile, exc.status_code, detail)
    if not _should_retry_without_image_url(exc.status_code, detail, messages):
        return _status_error_result(exc.status_code)

    logger.warning("[llm.%s] image_url content rejected; retrying as text-only", profile)
    retry_kwargs = dict(kwargs)
    retry_kwargs["messages"] = _strip_image_url_blocks(messages)
    try:
        response = await client.chat.completions.create(**retry_kwargs)
    except APIStatusError as retry_exc:
        retry_detail = _status_error_detail(retry_exc)
        logger.warning("[llm.%s] retry status error HTTP %s: %s", profile, retry_exc.status_code, retry_detail)
        return _status_error_result(retry_exc.status_code)
    except OpenAIError:
        logger.warning("[llm.%s] retry API error", profile, exc_info=True)
        return LLMResult("调用模型时绊了一下，我已经记下日志，稍后再来。", [], ok=False)
    except Exception:
        logger.exception("[llm.%s] retry unexpected error", profile)
        return LLMResult("处理回复时遇到一点未知状况，先别急，我们稍后再试。", [], ok=False)
    return _response_to_result(response)


def _status_error_result(status_code: int) -> LLMResult:
    """把常见 HTTP 状态码转换成用户可读的模型错误回复。"""
    if status_code == 401:
        return LLMResult("模型这边没认出钥匙，请检查 API key 是否正确。", [], ok=False)
    if status_code == 429:
        return LLMResult("模型服务有点忙，可能是限流或额度不足，我们稍后再试。", [], ok=False)
    return LLMResult(f"模型服务刚刚绊了一下（HTTP {status_code}），我们稍后再试。", [], ok=False)


def _should_retry_without_image_url(status_code: int, detail: str, messages: list[dict[str, Any]]) -> bool:
    """判断当前 400 错误是否适合移除图片块后自动重试。"""
    if status_code != 400 or not _messages_have_image_url(messages):
        return False
    lowered = detail.lower()
    return "image_url" in lowered and ("unknown variant" in lowered or "expected `text`" in lowered or "expected text" in lowered)


def _messages_have_image_url(messages: list[dict[str, Any]]) -> bool:
    """检查消息列表中是否包含 OpenAI 多模态 image_url 块。"""
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    return True
    return False


def _strip_image_url_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """移除多模态图片块，并保留文本内容供降级重试使用。"""
    stripped: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            stripped.append(dict(message))
            continue
        text_parts = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and str(block.get("text", "")).strip()
        ]
        next_message = dict(message)
        suffix = "（图片附件已因当前模型接口不支持 image_url 而降级为纯文本处理。）"
        next_message["content"] = ("\n".join(text_parts).strip() + "\n" + suffix).strip() if text_parts else suffix
        stripped.append(next_message)
    return stripped


def _response_to_result(response: Any) -> LLMResult:
    """把 OpenAI SDK 响应对象转换成项目内部统一的 LLMResult。"""
    message = response.choices[0].message
    reasoning_content = _get_message_extra(message, "reasoning_content")
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
    return LLMResult(
        content=message.content or "",
        tool_calls=calls,
        ok=True,
        reasoning_content=reasoning_content if isinstance(reasoning_content, str) else None,
    )


def _get_message_extra(message: Any, key: str) -> Any:
    """读取 OpenAI SDK 保留下来的厂商扩展字段。"""
    value = getattr(message, key, None)
    if value is not None:
        return value
    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(key)
    if hasattr(message, "model_dump"):
        try:
            return message.model_dump().get(key)
        except Exception:
            return None
    return None


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
