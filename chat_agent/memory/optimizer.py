"""记忆审计导出循环。

该模块保留原 memory optimizer 的入口名称，但职责已降级为离线审计导出：
它把 SQLite 中的 summary / memories / candidates / replacements 导出到
`workspace/memory/<chat_id>/`，不参与在线推理。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from chat_agent.memory.files import MemoryFiles
from chat_agent.memory.store import SQLiteStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OptimizerResult:
    """一次离线导出任务的结果。

    Attributes:
        ran: 本次是否真的执行了导出。
        reason: 没有执行或提前返回时的原因。
        exported_chats: 已成功导出的会话数量。
    """
    ran: bool
    reason: str = ""
    exported_chats: int = 0


class MemoryOptimizer:
    """离线导出当前记忆审计快照。"""

    def __init__(self, store: SQLiteStore, memory_files: MemoryFiles) -> None:
        """初始化 `MemoryOptimizer` 实例。

        参数:
            store: 初始化 `MemoryOptimizer` 时需要的 `store` 参数。
            memory_files: 初始化 `MemoryOptimizer` 时需要的 `memory_files` 参数。
        """
        self.store = store
        self.memory_files = memory_files

    async def run_once(self, chat_ids: list[str]) -> OptimizerResult:
        """执行`once`。

        参数:
            chat_ids: 参与执行`once`的 `chat_ids` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if not chat_ids:
            return OptimizerResult(False, "no_chat", 0)
        exported = 0
        for chat_id in chat_ids:
            summary = await self.store.get_summary(chat_id) or ""
            user_profile = await self.store.get_user_profile(chat_id)
            memories = await self.store.list_active_memories(chat_id, limit=200)
            candidates = await self.store.get_memory_candidates(chat_id, limit=200)
            replacements = await self.store.list_memory_replacements(chat_id, limit=100)
            self.memory_files.export_chat_snapshot(chat_id, summary, memories, candidates, replacements, user_profile=user_profile)
            exported += 1
        return OptimizerResult(True, exported_chats=exported)


class MemoryOptimizerLoop:
    """后台定时运行导出器；默认不用于主运行时。"""

    def __init__(self, optimizer: MemoryOptimizer, interval_seconds: int = 300, enabled: bool = False) -> None:
        """初始化 `MemoryOptimizerLoop` 实例。

        参数:
            optimizer: 初始化 `MemoryOptimizerLoop` 时需要的 `optimizer` 参数。
            interval_seconds: 初始化 `MemoryOptimizerLoop` 时需要的 `interval_seconds` 参数。
            enabled: 初始化 `MemoryOptimizerLoop` 时需要的 `enabled` 参数。
        """
        self.optimizer = optimizer
        self.interval_seconds = max(30, int(interval_seconds))
        self.enabled = enabled
        self._stopped = asyncio.Event()

    async def run(self, chat_ids: list[str]) -> None:
        """执行相关逻辑。

        参数:
            chat_ids: 参与执行相关逻辑的 `chat_ids` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if not self.enabled:
            return
        logger.info("Memory optimizer(export-only) loop started interval=%s", self.interval_seconds)
        while not self._stopped.is_set():
            try:
                result = await self.optimizer.run_once(chat_ids)
                logger.info("Memory optimizer(export-only) tick ran=%s reason=%s exported=%s", result.ran, result.reason, result.exported_chats)
            except Exception:
                logger.exception("Memory optimizer(export-only) loop tick failed")
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        """停止相关逻辑。

        返回:
            返回与本函数处理结果对应的数据。"""
        self._stopped.set()
