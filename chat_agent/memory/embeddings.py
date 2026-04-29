"""Embedding Provider 封装。

本模块只负责把文本送到 OpenAI-compatible embeddings 接口并返回向量。
向量写入、检索、排序由 vector_store.py 和 retriever.py 负责，避免 Provider 混入存储逻辑。
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """OpenAI-compatible embeddings 客户端。

    参数:
        model: embedding 模型名，例如 text-embedding-v4。
        api_key: OpenAI-compatible API key。
        base_url: OpenAI-compatible 服务地址。
        timeout_seconds: 请求超时时间。

    说明:
        这个 provider 只负责把文本变成向量；向量保存和检索由 vector_store.py 负责。
    """

    def __init__(self, model: str, api_key: str, base_url: str, timeout_seconds: float = 30, dimension: int | None = None) -> None:
        """初始化 OpenAI-compatible embedding 客户端。

        参数:
            model: embedding 模型名，例如 text-embedding-v4。
            api_key: API key，来自配置或环境变量，不能写死在代码中。
            base_url: OpenAI-compatible 服务地址。
            timeout_seconds: 单次 embedding 请求超时时间。
            dimension: 可选向量维度；百炼 text-embedding-v3/v4 支持 dimensions 参数。
        """
        self.model = model
        self.dimension = dimension
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=1,
        )

    async def embed(self, text: str) -> list[float] | None:
        """生成单条文本 embedding。

        参数:
            text: 需要向量化的文本。

        返回:
            成功时返回 float 向量；失败时返回 None，并写入日志，不阻断主对话。
        """
        text = text.strip()
        if not text:
            return None
        try:
            kwargs: dict[str, Any] = {"model": self.model, "input": text, "encoding_format": "float"}
            if self.dimension:
                kwargs["dimensions"] = self.dimension
            response = await self.client.embeddings.create(**kwargs)
            raw_embedding: Any = response.data[0].embedding
            return [float(value) for value in raw_embedding]
        except AuthenticationError:
            logger.warning("[embedding] authentication failed")
        except RateLimitError:
            logger.warning("[embedding] rate limited")
        except APITimeoutError:
            logger.warning("[embedding] request timed out")
        except APIConnectionError:
            logger.warning("[embedding] connection error")
        except APIStatusError as exc:
            logger.warning("[embedding] status error HTTP %s", exc.status_code)
        except OpenAIError:
            logger.warning("[embedding] API error", exc_info=True)
        except Exception:
            logger.exception("[embedding] unexpected error")
        return None
