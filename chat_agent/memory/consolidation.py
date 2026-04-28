"""会话长期记忆整理服务。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from chat_agent.memory.embeddings import EmbeddingProvider
from chat_agent.memory.files import MemoryFiles
from chat_agent.memory.store import SQLiteStore
from chat_agent.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ConsolidationResult:
    """一次会话整理任务的执行结果。

    字段:
        ran: 本次整理是否实际执行。
        source_ref: 本次整理对应的会话窗口引用，便于审计去重。
        message_count: 参与整理的会话消息数。
        memory_count: 本轮提炼出的长期记忆条数。
        candidate_count: 本轮生成的候选记忆条数。
        reason: 未执行或特殊分支的原因标记。
    """
    ran: bool
    source_ref: str
    message_count: int
    memory_count: int
    candidate_count: int = 0
    reason: str = ""


class ConsolidationService:
    """把旧会话窗口整理成 summary、candidate 和记忆审计导出。"""

    def __init__(
        self,
        store: SQLiteStore,
        memory_files: MemoryFiles | None = None,
        provider: Any | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        keep_recent: int = 20,
        max_window: int = 80,
    ) -> None:
        """初始化长期记忆整理服务。

        参数:
            store: 用于读取会话窗口并写回整理结果的存储层。
            memory_files: 可选的审计导出器，用于把整理结果导出到工作区。
            provider: 可选的 LLM provider，用于从旧消息窗口中抽取摘要和记忆。
            embedding_provider: 可选的 embedding provider，用于给新记忆生成向量。
            vector_store: 可选的向量存储实现，用于持久化 embedding。
            keep_recent: 每次整理时保留多少条最新消息不参与压缩。
            max_window: 单次整理允许处理的最大旧消息窗口大小。
        """
        self.store = store
        self.memory_files = memory_files
        self.provider = provider
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.keep_recent = keep_recent
        self.max_window = max_window

    async def run_once(self, chat_id: str) -> ConsolidationResult:
        """对单个会话执行一次长期记忆整理。

        参数:
            chat_id: 目标会话 id。

        返回:
            返回 `ConsolidationResult`，描述是否运行、处理了多少消息以及产出了多少记忆。
        """
        last = await self.store.get_last_consolidated(chat_id)
        window = await self.store.get_consolidation_window(
            chat_id,
            after_id=last,
            keep_recent=self.keep_recent,
            limit=self.max_window,
        )
        if not window:
            return ConsolidationResult(False, "", 0, 0, 0, "no_window")

        first_id = int(window[0]["id"])
        last_id = int(window[-1]["id"])
        source_ref = f"session:{chat_id}:{first_id}-{last_id}"
        if await self.store.has_consolidation_event(source_ref):
            return ConsolidationResult(False, source_ref, len(window), 0, 0, "duplicate")

        payload = await self._extract(window)
        recent_context = str(payload.get("recent_context") or self._fallback_recent_context(window)).strip()
        count = await self.store.count_session_messages(chat_id)
        await self.store.upsert_summary(chat_id, recent_context, count)

        memory_count = 0
        candidate_count = 0
        for item in payload.get("memories", []):
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            memory_type = str(item.get("type") or "fact")
            tags = item.get("tags") if isinstance(item.get("tags"), list) else [memory_type, "consolidated"]
            importance = float(item.get("importance") or 0.55)
            confidence = float(item.get("confidence") or (0.75 if importance >= 0.8 else 0.55))
            if importance >= 0.75 or confidence >= 0.75:
                memory_id = await self.store.add_memory(
                    chat_id,
                    content,
                    tags=tags,
                    memory_type=memory_type,
                    importance=importance,
                    source_chat_id=chat_id,
                    source_ref=source_ref,
                    extra={"source": "consolidation"},
                    emotional_weight=float(item.get("emotional_weight") or 0.0),
                    source_kind="inferred",
                    confidence=max(confidence, 0.7),
                )
                await self._embed(chat_id, memory_id, content)
                memory_count += 1
                for old_id in item.get("supersedes") or []:
                    with _ignore_bad_supersede():
                        await self.store.supersede_memory(chat_id, int(old_id), memory_id, "consolidation")
            else:
                await self.store.add_memory_candidate(
                    chat_id,
                    content,
                    tags=tags,
                    memory_type=memory_type,
                    importance=importance,
                    source_kind="candidate",
                    confidence=confidence,
                    source_ref=source_ref,
                )
                candidate_count += 1

        await self.store.set_last_consolidated(chat_id, last_id)
        await self.store.add_consolidation_event(
            chat_id,
            source_ref,
            details={"messages": len(window), "memories": memory_count, "candidates": candidate_count},
        )
        await self._export_audit(chat_id)
        logger.info(
            "Consolidated memory chat_id=%s source_ref=%s messages=%s memories=%s candidates=%s",
            chat_id,
            source_ref,
            len(window),
            memory_count,
            candidate_count,
        )
        return ConsolidationResult(True, source_ref, len(window), memory_count, candidate_count)

    async def _extract(self, window: list[dict[str, Any]]) -> dict[str, Any]:
        """提取相关逻辑。

        参数:
            window: 参与提取相关逻辑的 `window` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        text = "\n".join(f"#{row['id']} {row['role']}: {row['content']}" for row in window)
        if not self.provider:
            return self._fallback_extract(window)
        try:
            result = await self.provider.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是个人智能体的记忆整理器。请只输出 JSON，格式为："
                            "{\"memories\":[{\"type\":\"preference|fact|event|procedure\","
                            "\"content\":\"...\",\"tags\":[\"...\"],\"importance\":0.5,"
                            "\"confidence\":0.5,\"emotional_weight\":0,\"supersedes\":[]}],"
                            "\"recent_context\":\"...\"}。"
                            "只抽取稳定事实、偏好、流程和有后续价值的事件，不要把闲聊逐句变长期记忆。"
                        ),
                    },
                    {"role": "user", "content": text},
                ]
            )
            if not getattr(result, "ok", True):
                return self._fallback_extract(window)
            parsed = _loads_json_object(result.content)
            return parsed if parsed else self._fallback_extract(window)
        except Exception:
            logger.exception("Memory consolidation extraction failed; using fallback")
            return self._fallback_extract(window)

    def _fallback_extract(self, window: list[dict[str, Any]]) -> dict[str, Any]:
        """处理`extract`。

        参数:
            window: 参与处理`extract`的 `window` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        memories: list[dict[str, Any]] = []
        for row in window:
            content = str(row["content"])
            if row["role"] == "user" and any(token in content for token in ["喜欢", "偏好", "习惯", "我叫", "我住在"]):
                memories.append(
                    {
                        "type": "preference" if "喜欢" in content or "偏好" in content else "fact",
                        "content": content[:300],
                        "tags": ["auto", "consolidated"],
                        "importance": 0.55,
                        "confidence": 0.55,
                    }
                )
        return {
            "memories": memories[:8],
            "recent_context": self._fallback_recent_context(window),
        }

    def _fallback_recent_context(self, window: list[dict[str, Any]]) -> str:
        """处理`recent`、`context`。

        参数:
            window: 参与处理`recent`、`context`的 `window` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        snippets = [f"- {row['role']}: {str(row['content'])[:160]}" for row in window[-10:]]
        return "\n".join(snippets)

    async def _embed(self, chat_id: str, memory_id: int, content: str) -> None:
        """生成向量相关逻辑。

        参数:
            chat_id: 参与生成向量相关逻辑的 `chat_id` 参数。
            memory_id: 参与生成向量相关逻辑的 `memory_id` 参数。
            content: 参与生成向量相关逻辑的 `content` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if not self.embedding_provider or not self.vector_store:
            return
        try:
            embedding = await self.embedding_provider.embed(content)
            if embedding:
                await self.vector_store.upsert_memory(chat_id, memory_id, embedding)
        except NotImplementedError as exc:
            logger.warning("Memory consolidation embedding unavailable: %s", exc)
        except Exception:
            logger.exception("Memory consolidation embedding failed memory_id=%s", memory_id)

    async def _export_audit(self, chat_id: str) -> None:
        """处理`audit`。

        参数:
            chat_id: 参与处理`audit`的 `chat_id` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if not self.memory_files:
            return
        try:
            summary = await self.store.get_summary(chat_id) or ""
            user_profile = await self.store.get_user_profile(chat_id)
            memories = await self.store.list_active_memories(chat_id, limit=200)
            candidates = await self.store.get_memory_candidates(chat_id, limit=200)
            replacements = await self.store.list_memory_replacements(chat_id, limit=100)
            self.memory_files.export_chat_snapshot(chat_id, summary, memories, candidates, replacements, user_profile=user_profile)
        except Exception:
            logger.exception("Failed to export memory audit snapshot chat_id=%s", chat_id)


class _ignore_bad_supersede:
    """忽略 supersede 写入阶段的非关键错误。"""
    def __enter__(self) -> None:
        """处理相关逻辑。

        返回:
            返回与本函数处理结果对应的数据。"""
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        """处理相关逻辑。

        参数:
            exc_type: 参与处理相关逻辑的 `exc_type` 参数。
            exc: 参与处理相关逻辑的 `exc` 参数。
            tb: 参与处理相关逻辑的 `tb` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        return True


def _loads_json_object(text: str) -> dict[str, Any] | None:
    """处理`json`、`object`。

    参数:
        text: 参与处理`json`、`object`的 `text` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    raw = text.strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.removeprefix("json").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
