"""记忆审计导出文件管理。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chat_agent.memory.interests import build_interest_watchlist, render_interest_watchlist_md


class MemoryFiles:
    """管理 `workspace/memory/<chat_id>/` 下的审计导出文件。

    这些文件只用于人工检查、导出和调试，不参与在线记忆推理。
    """

    def __init__(self, root: str | Path) -> None:
        """初始化 `MemoryFiles` 实例。

        参数:
            root: 初始化 `MemoryFiles` 时需要的 `root` 参数。
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def export_chat_snapshot(
        self,
        chat_id: str,
        summary: str,
        memories: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        replacements: list[dict[str, Any]],
        user_profile: dict[str, Any] | None = None,
    ) -> Path:
        """导出指定 chat 的当前记忆审计快照。"""
        chat_root = self.chat_root(chat_id)
        self._write(
            chat_root / "SUMMARY.md",
            "# Summary\n\n" + (summary.strip() or "(empty)") + "\n",
        )
        self._write(chat_root / "MEMORIES.md", self._format_memories(memories))
        interest_hints = build_interest_watchlist(user_profile or {}, memories)
        self._write(chat_root / "INTERESTS.md", render_interest_watchlist_md(interest_hints))
        self._write(chat_root / "CANDIDATES.md", self._format_candidates(candidates))
        self._write(chat_root / "REPLACEMENTS.md", self._format_replacements(replacements))
        return chat_root

    def chat_root(self, chat_id: str) -> Path:
        """处理`root`。

        参数:
            chat_id: 参与处理`root`的 `chat_id` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        safe_chat_id = _safe_component(chat_id)
        path = self.root / safe_chat_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write(self, path: Path, text: str) -> None:
        """写入相关逻辑。

        参数:
            path: 参与写入相关逻辑的 `path` 参数。
            text: 参与写入相关逻辑的 `text` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        path.write_text(text, encoding="utf-8")

    def _format_memories(self, memories: list[dict[str, Any]]) -> str:
        """格式化记忆集合。

        参数:
            memories: 参与格式化记忆集合的 `memories` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        lines = ["# Memories", ""]
        if not memories:
            lines.append("(empty)")
            return "\n".join(lines) + "\n"
        for item in memories:
            lines.append(
                f"- #{item['id']} [{item.get('type','fact')}] {item.get('content','')} "
                f"(source={item.get('source_kind','inferred')}, confidence={float(item.get('confidence',0.0)):.2f}, "
                f"importance={float(item.get('importance',0.0)):.2f}, reinforcement={int(item.get('reinforcement',1))})"
            )
        return "\n".join(lines) + "\n"

    def _format_candidates(self, candidates: list[dict[str, Any]]) -> str:
        """格式化候选集合。

        参数:
            candidates: 参与格式化候选集合的 `candidates` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        lines = ["# Candidates", ""]
        if not candidates:
            lines.append("(empty)")
            return "\n".join(lines) + "\n"
        for item in candidates:
            lines.append(
                f"- #{item['id']} [{item.get('type','fact')}] {item.get('content','')} "
                f"(status={item.get('status','pending')}, confidence={float(item.get('confidence',0.0)):.2f}, "
                f"evidence={int(item.get('evidence_count',1))})"
            )
        return "\n".join(lines) + "\n"

    def _format_replacements(self, replacements: list[dict[str, Any]]) -> str:
        """格式化`replacements`。

        参数:
            replacements: 参与格式化`replacements`的 `replacements` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        lines = ["# Replacements", ""]
        if not replacements:
            lines.append("(empty)")
            return "\n".join(lines) + "\n"
        for item in replacements:
            lines.append(
                f"- {item.get('created_at','')}: #{item.get('old_memory_id')} -> #{item.get('new_memory_id')} "
                f"({item.get('reason','')})"
            )
        return "\n".join(lines) + "\n"


def _safe_component(value: str) -> str:
    """安全处理`component`。

    参数:
        value: 参与安全处理`component`的 `value` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value) or "default"
