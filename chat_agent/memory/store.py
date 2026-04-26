"""SQLite 持久化层。

本模块封装会话历史、长期记忆、提醒、trace、MCP 日志、proactive 记录等本地数据表。
对外只暴露 async 方法；内部大量名为 run() 的局部函数是传给 asyncio.to_thread 的
同步 SQLite 事务闭包，用来避免阻塞事件循环。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def utc_now() -> datetime:
    """返回带 UTC 时区的当前时间。

    说明:
        数据库内所有时间都统一存 UTC，展示时再转换到本地时区，避免跨时区或夏令时问题。
    """
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    """返回当前 UTC 时间的 ISO 字符串，供 SQLite 存储使用。"""
    return utc_now().isoformat()


def _to_iso(dt: datetime) -> str:
    """把 datetime 标准化为 UTC ISO 字符串。

    参数:
        dt: 任意 datetime。若没有 tzinfo，会按 UTC 处理。
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _from_iso(value: str | None) -> datetime | None:
    """从 SQLite 中保存的 ISO 字符串恢复 UTC datetime。

    参数:
        value: ISO 时间字符串或 None。

    返回:
        带 UTC 时区的 datetime；输入为 None 时返回 None。
    """
    if value is None:
        return None
    return datetime.fromisoformat(value).astimezone(timezone.utc)


@dataclass(slots=True)
class MemoryItem:
    """长期记忆的数据结构。

    字段:
        id: 数据库自增 id。
        type: 记忆类型，例如 preference、fact、event、procedure。
        content: 记忆正文。
        tags: 标签列表，用于 LIKE 检索和展示。
        source_chat_id: 这条记忆来源的 chat id。
        created_at: 创建时间，UTC ISO 字符串。
        updated_at: 更新时间，UTC ISO 字符串。
        importance: 重要性分数，越高检索排序越靠前。
        last_used_at: 最近一次被检索使用的时间，可能为空。
    """

    id: int
    type: str
    content: str
    tags: list[str]
    source_chat_id: str
    created_at: str
    updated_at: str
    importance: float
    last_used_at: str | None


class SQLiteStore:
    """SQLite 持久化层。

    该类封装项目所有本地状态表，包括：
    - chats: Telegram chat 与最后活跃时间。
    - session_messages: 原始对话历史。
    - memories: 长期记忆。
    - reminders: 定时提醒。
    - proactive_tick_log/proactive_deliveries/seen_items: 主动系统状态。
    - message_trace/mcp_tool_log/drift_runs: 观测与排查记录。

    线程模型:
        sqlite3 是同步库，所有公开 async 方法都通过 asyncio.to_thread 执行阻塞 SQL，
        避免卡住 Telegram 和 proactive 的事件循环。
    """

    def __init__(self, database_path: str | Path) -> None:
        """初始化 SQLiteStore 并自动建表/迁移。

        参数:
            database_path: SQLite 数据库文件路径。父目录不存在时会自动创建。
        """
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """创建一个新的 SQLite 连接。

        返回:
            row_factory 已设置为 sqlite3.Row 的连接，便于按列名读取结果。
        """
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        """初始化数据库 schema，并执行轻量向前兼容迁移。

        说明:
            使用 CREATE TABLE IF NOT EXISTS 和 _ensure_column 组合，允许用户从旧版本直接升级，
            不需要手动删除 workspace/agent.sqlite。
        """
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id TEXT PRIMARY KEY,
                    username TEXT,
                    last_seen_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_session_chat_id ON session_messages(chat_id, id);

                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    type TEXT NOT NULL DEFAULT 'fact',
                    content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    source_chat_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT '',
                    importance REAL NOT NULL DEFAULT 0.5,
                    last_used_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_memories_chat_id ON memories(chat_id, id);

                CREATE TABLE IF NOT EXISTS memory_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    type TEXT NOT NULL DEFAULT 'fact',
                    content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    importance REAL NOT NULL DEFAULT 0.5,
                    source_kind TEXT NOT NULL DEFAULT 'candidate',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    source_ref TEXT NOT NULL DEFAULT '',
                    content_hash TEXT NOT NULL DEFAULT '',
                    evidence_count INTEGER NOT NULL DEFAULT 1,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    expires_at TEXT,
                    status TEXT NOT NULL DEFAULT 'pending'
                );
                CREATE INDEX IF NOT EXISTS idx_memory_candidates_chat_id ON memory_candidates(chat_id, id);
                CREATE INDEX IF NOT EXISTS idx_memory_candidates_status ON memory_candidates(chat_id, status);

                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id INTEGER PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_embeddings_chat_id ON memory_embeddings(chat_id);

                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    chat_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    message_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_profiles (
                    chat_id TEXT PRIMARY KEY,
                    profile TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    delivered_at TEXT,
                    cancelled_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(delivered_at, cancelled_at, due_at);

                CREATE TABLE IF NOT EXISTS message_trace (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    assistant_reply TEXT NOT NULL,
                    model_main TEXT NOT NULL DEFAULT '',
                    model_fast TEXT NOT NULL DEFAULT '',
                    tools_used TEXT NOT NULL DEFAULT '[]',
                    mcp_tools_used TEXT NOT NULL DEFAULT '[]',
                    memory_hits TEXT NOT NULL DEFAULT '[]',
                    hyde_used INTEGER NOT NULL DEFAULT 0,
                    attachments_count INTEGER NOT NULL DEFAULT 0,
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS proactive_tick_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tick_at TEXT NOT NULL,
                    action TEXT NOT NULL,
                    skip_reason TEXT,
                    reminders_due INTEGER NOT NULL DEFAULT 0,
                    content_count INTEGER NOT NULL DEFAULT 0,
                    sent_count INTEGER NOT NULL DEFAULT 0,
                    sent_message TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS proactive_deliveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    source TEXT NOT NULL,
                    delivered_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS proactive_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    body TEXT NOT NULL DEFAULT '',
                    url TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.0,
                    novelty REAL NOT NULL DEFAULT 0.0,
                    user_fit REAL NOT NULL DEFAULT 0.0,
                    priority REAL NOT NULL DEFAULT 0.0,
                    shareable INTEGER NOT NULL DEFAULT 0,
                    dedupe_key TEXT NOT NULL DEFAULT '',
                    artifact_path TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    score REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    drop_reason TEXT,
                    sent_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_proactive_candidates_chat_id ON proactive_candidates(chat_id, id);
                CREATE INDEX IF NOT EXISTS idx_proactive_candidates_status ON proactive_candidates(status, source_type);

                CREATE TABLE IF NOT EXISTS seen_items (
                    item_key TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    title TEXT,
                    url TEXT,
                    seen_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS mcp_tool_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    server TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    args_preview TEXT,
                    result_preview TEXT,
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS drift_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    result TEXT NOT NULL,
                    output_path TEXT,
                    notified INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS drift_task_state (
                    task_id TEXT PRIMARY KEY,
                    last_run_at TEXT,
                    last_status TEXT NOT NULL DEFAULT '',
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    last_artifact_path TEXT NOT NULL DEFAULT '',
                    last_artifact_at TEXT
                );

                CREATE TABLE IF NOT EXISTS consolidation_state (
                    chat_id TEXT PRIMARY KEY,
                    last_consolidated INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS consolidation_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    source_ref TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_replacements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    old_memory_id INTEGER NOT NULL,
                    new_memory_id INTEGER NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                """
            )
            for column, definition in {
                "type": "TEXT NOT NULL DEFAULT 'fact'",
                "tags": "TEXT NOT NULL DEFAULT '[]'",
                "source_chat_id": "TEXT NOT NULL DEFAULT ''",
                "updated_at": "TEXT NOT NULL DEFAULT ''",
                "importance": "REAL NOT NULL DEFAULT 0.5",
                "last_used_at": "TEXT",
                "content_hash": "TEXT NOT NULL DEFAULT ''",
                "status": "TEXT NOT NULL DEFAULT 'active'",
                "source_ref": "TEXT NOT NULL DEFAULT ''",
                "extra_json": "TEXT NOT NULL DEFAULT '{}'",
                "reinforcement": "INTEGER NOT NULL DEFAULT 1",
                "emotional_weight": "REAL NOT NULL DEFAULT 0.0",
                "source_kind": "TEXT NOT NULL DEFAULT 'inferred'",
                "confidence": "REAL NOT NULL DEFAULT 0.7",
            }.items():
                self._ensure_column(conn, "memories", column, definition)
            for column, definition in {
                "type": "TEXT NOT NULL DEFAULT 'fact'",
                "tags": "TEXT NOT NULL DEFAULT '[]'",
                "importance": "REAL NOT NULL DEFAULT 0.5",
                "source_kind": "TEXT NOT NULL DEFAULT 'candidate'",
                "confidence": "REAL NOT NULL DEFAULT 0.5",
                "source_ref": "TEXT NOT NULL DEFAULT ''",
                "content_hash": "TEXT NOT NULL DEFAULT ''",
                "evidence_count": "INTEGER NOT NULL DEFAULT 1",
                "first_seen_at": "TEXT NOT NULL DEFAULT ''",
                "last_seen_at": "TEXT NOT NULL DEFAULT ''",
                "expires_at": "TEXT",
                "status": "TEXT NOT NULL DEFAULT 'pending'",
            }.items():
                self._ensure_column(conn, "memory_candidates", column, definition)
            self._ensure_column(conn, "reminders", "cancelled_at", "TEXT")
            for column, definition in {
                "model_main": "TEXT NOT NULL DEFAULT ''",
                "model_fast": "TEXT NOT NULL DEFAULT ''",
                "mcp_tools_used": "TEXT NOT NULL DEFAULT '[]'",
                "hyde_used": "INTEGER NOT NULL DEFAULT 0",
                "attachments_count": "INTEGER NOT NULL DEFAULT 0",
            }.items():
                self._ensure_column(conn, "message_trace", column, definition)
            for column, definition in {
                "content_count": "INTEGER NOT NULL DEFAULT 0",
                "sent_count": "INTEGER NOT NULL DEFAULT 0",
            }.items():
                self._ensure_column(conn, "proactive_tick_log", column, definition)

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        """确保某张表存在指定列，不存在时执行 ALTER TABLE。

        参数:
            conn: 已打开的 SQLite 连接。
            table: 表名。
            column: 列名。
            definition: SQLite 列定义，例如 "TEXT NOT NULL DEFAULT ''"。
        """
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    async def record_chat(self, chat_id: str, username: str | None) -> None:
        """记录或更新一个 Telegram chat 的最后活跃时间。

        参数:
            chat_id: Telegram chat id 字符串。
            username: Telegram 用户名，可能为空。
        """
        def run() -> None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO chats(chat_id, username, last_seen_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(chat_id) DO UPDATE SET
                        username = excluded.username,
                        last_seen_at = excluded.last_seen_at
                    """,
                    (chat_id, username, _utc_now_iso()),
                )

        await asyncio.to_thread(run)

    async def get_last_seen_at(self, chat_id: str) -> datetime | None:
        """读取指定 chat 的最后活跃时间。

        参数:
            chat_id: Telegram chat id 字符串。

        返回:
            UTC datetime；没有记录时返回 None。
        """
        def run() -> datetime | None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute("SELECT last_seen_at FROM chats WHERE chat_id = ?", (chat_id,)).fetchone()
            return _from_iso(row["last_seen_at"]) if row else None

        return await asyncio.to_thread(run)

    async def add_session_message(self, chat_id: str, role: str, content: str) -> None:
        """追加一条会话历史。

        参数:
            chat_id: Telegram chat id。
            role: 消息角色，通常是 user、assistant、system 或 tool。
            content: 消息正文。附件会在上层转换为可读摘要后写入。
        """
        def run() -> None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO session_messages(chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (chat_id, role, content, _utc_now_iso()),
                )

        await asyncio.to_thread(run)

    async def get_recent_session_messages(self, chat_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """读取指定 chat 最近的会话历史。

        参数:
            chat_id: Telegram chat id。
            limit: 最多读取多少条，按时间从旧到新返回。
        """
        def run() -> list[dict[str, Any]]:
            """在线程池中读取最近会话历史，并按时间升序整理返回。"""
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT role, content, created_at
                    FROM session_messages
                    WHERE chat_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (chat_id, limit),
                ).fetchall()
            return [dict(row) for row in reversed(rows)]

        return await asyncio.to_thread(run)

    async def get_consolidation_window(
        self,
        chat_id: str,
        after_id: int,
        keep_recent: int = 20,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """读取可整理的旧会话窗口，同时保留最新 keep_recent 条作为热上下文。

        参数:
            chat_id: 当前 Telegram 会话 id。
            after_id: 已整理检查点，只读取 id 大于它的消息。
            keep_recent: 最新多少条消息暂不整理，避免刚发生的上下文被过早归档。
            limit: 单次最多整理多少条，避免一次 prompt 过长。

        返回:
            按 id 从小到大排列的 session_messages 行，包含 id/role/content/created_at。
        """

        def run() -> list[dict[str, Any]]:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT id, role, content, created_at
                    FROM session_messages
                    WHERE chat_id = ? AND id > ?
                      AND id NOT IN (
                          SELECT id FROM session_messages
                          WHERE chat_id = ?
                          ORDER BY id DESC
                          LIMIT ?
                      )
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (chat_id, after_id, chat_id, keep_recent, limit),
                ).fetchall()
            return [dict(row) for row in rows]

        return await asyncio.to_thread(run)

    async def count_session_messages(self, chat_id: str) -> int:
        """统计指定 chat 的会话历史条数。"""
        def run() -> int:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) AS count FROM session_messages WHERE chat_id = ?", (chat_id,)).fetchone()
            return int(row["count"])

        return await asyncio.to_thread(run)

    async def prune_session_messages(self, chat_id: str, keep: int) -> int:
        """清理指定 chat 的旧会话历史，只保留最新 keep 条。

        参数:
            chat_id: Telegram chat id。
            keep: 保留条数；小于等于 0 时不删除。

        返回:
            实际删除的行数。
        """
        if keep <= 0:
            return 0

        def run() -> int:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM session_messages
                    WHERE chat_id = ?
                      AND id NOT IN (
                          SELECT id FROM session_messages
                          WHERE chat_id = ?
                          ORDER BY id DESC
                          LIMIT ?
                      )
                    """,
                    (chat_id, chat_id, keep),
                )
                return int(cursor.rowcount)

        return await asyncio.to_thread(run)

    async def get_summary(self, chat_id: str) -> str | None:
        """读取指定 chat 的近期上下文摘要。"""
        def run() -> str | None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute("SELECT summary FROM conversation_summaries WHERE chat_id = ?", (chat_id,)).fetchone()
            return str(row["summary"]) if row else None

        return await asyncio.to_thread(run)

    async def upsert_summary(self, chat_id: str, summary: str, message_count: int) -> None:
        """新增或更新指定 chat 的近期上下文摘要。

        参数:
            chat_id: Telegram chat id。
            summary: 摘要正文。
            message_count: 生成摘要时对应的 session message 总数。
        """
        def run() -> None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO conversation_summaries(chat_id, summary, message_count, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(chat_id) DO UPDATE SET
                        summary = excluded.summary,
                        message_count = excluded.message_count,
                        updated_at = excluded.updated_at
                    """,
                    (chat_id, summary, message_count, _utc_now_iso()),
                )

        await asyncio.to_thread(run)

    async def get_user_profile(self, chat_id: str) -> dict[str, Any]:
        """读取指定 chat 的用户画像。

        参数:
            chat_id: Telegram chat id。

        返回:
            用户画像字典。没有画像或 JSON 损坏时返回空字典。
        """
        def run() -> dict[str, Any]:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute("SELECT profile FROM user_profiles WHERE chat_id = ?", (chat_id,)).fetchone()
            if not row:
                return {}
            try:
                data = json.loads(row["profile"] or "{}")
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}

        return await asyncio.to_thread(run)

    async def update_user_profile(self, chat_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        """合并更新指定 chat 的用户画像。

        参数:
            chat_id: Telegram chat id。
            updates: 要合并进画像的字段。None 值会被忽略，list 字段会去重合并。

        返回:
            更新后的完整画像。
        """
        current = await self.get_user_profile(chat_id)
        for key, value in updates.items():
            if value is None:
                continue
            if isinstance(value, list):
                existing = current.get(key, [])
                if not isinstance(existing, list):
                    existing = [existing]
                merged = [str(item) for item in existing if str(item).strip()]
                for item in value:
                    text = str(item).strip()
                    if text and text not in merged:
                        merged.append(text)
                current[key] = merged[:50]
            elif isinstance(value, dict):
                existing = current.get(key, {})
                if not isinstance(existing, dict):
                    existing = {}
                existing.update(value)
                current[key] = existing
            else:
                current[key] = value

        def run() -> None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO user_profiles(chat_id, profile, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(chat_id) DO UPDATE SET
                        profile = excluded.profile,
                        updated_at = excluded.updated_at
                    """,
                    (chat_id, json.dumps(current, ensure_ascii=False), _utc_now_iso()),
                )

        await asyncio.to_thread(run)
        return current

    async def add_memory(
        self,
        chat_id: str,
        content: str,
        tags: list[str] | str | None = None,
        memory_type: str = "fact",
        importance: float = 0.5,
        source_chat_id: str | None = None,
        source_ref: str | None = None,
        extra: dict[str, Any] | None = None,
        emotional_weight: float = 0.0,
        status: str = "active",
        source_kind: str = "inferred",
        confidence: float = 0.7,
    ) -> int:
        """写入一条长期记忆。

        参数:
            chat_id: 记忆归属的 chat id。
            content: 记忆正文。
            tags: 标签列表或逗号/空格分隔字符串。
            memory_type: 记忆类型，例如 fact、preference、event、procedure。
            importance: 重要性分数，影响检索排序。
            source_chat_id: 记忆来源 chat id；为空时使用 chat_id。

        返回:
            新插入记忆的自增 id。
        """
        normalized_content = " ".join(content.split())
        tag_text = json.dumps(_normalize_tags(tags), ensure_ascii=False)
        extra_text = json.dumps(extra or {}, ensure_ascii=False)
        content_hash = _content_hash(normalized_content)
        now = _utc_now_iso()

        def run() -> int:
            """在线程池中执行 SQLite 写事务，避免阻塞 asyncio 事件循环。"""
            with self._connect() as conn:
                duplicate = conn.execute(
                    """
                    SELECT id, reinforcement, importance, emotional_weight, source_kind, confidence
                    FROM memories
                    WHERE chat_id = ? AND content_hash = ? AND status = 'active'
                    ORDER BY id ASC
                    LIMIT 1
                    """,
                    (chat_id, content_hash),
                ).fetchone()
                if duplicate:
                    memory_id = int(duplicate["id"])
                    conn.execute(
                        """
                        UPDATE memories
                        SET reinforcement = reinforcement + 1,
                            importance = MAX(importance, ?),
                            emotional_weight = MAX(emotional_weight, ?),
                            confidence = MAX(confidence, ?),
                            source_kind = ?,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            importance,
                            emotional_weight,
                            confidence,
                            _better_source_kind(str(duplicate["source_kind"] or "inferred"), source_kind),
                            now,
                            memory_id,
                        ),
                    )
                    return memory_id
                cursor = conn.execute(
                    """
                    INSERT INTO memories(chat_id, type, content, tags, source_chat_id, created_at, updated_at,
                                         importance, content_hash, status, source_ref, extra_json,
                                         reinforcement, emotional_weight, source_kind, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chat_id,
                        memory_type,
                        normalized_content,
                        tag_text,
                        source_chat_id or chat_id,
                        now,
                        now,
                        importance,
                        content_hash,
                        status,
                        source_ref or "",
                        extra_text,
                        1,
                        emotional_weight,
                        source_kind,
                        confidence,
                    ),
                )
                return int(cursor.lastrowid)

        return await asyncio.to_thread(run)

    async def add_memory_candidate(
        self,
        chat_id: str,
        content: str,
        tags: list[str] | str | None = None,
        memory_type: str = "fact",
        importance: float = 0.5,
        source_kind: str = "candidate",
        confidence: float = 0.5,
        source_ref: str | None = None,
        expires_at: datetime | None = None,
    ) -> int:
        """写入或强化一条候选记忆。"""
        normalized_content = " ".join(content.split())
        if not normalized_content:
            raise ValueError("candidate memory content cannot be empty")
        tag_text = json.dumps(_normalize_tags(tags), ensure_ascii=False)
        content_hash = _content_hash(normalized_content)
        now = _utc_now_iso()
        expires_text = _to_iso(expires_at) if expires_at else _to_iso(utc_now() + timedelta(days=30))

        def run() -> int:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                duplicate = conn.execute(
                    """
                    SELECT id, evidence_count, confidence, importance, source_kind
                    FROM memory_candidates
                    WHERE chat_id = ? AND content_hash = ? AND status = 'pending'
                    ORDER BY id ASC
                    LIMIT 1
                    """,
                    (chat_id, content_hash),
                ).fetchone()
                if duplicate:
                    candidate_id = int(duplicate["id"])
                    conn.execute(
                        """
                        UPDATE memory_candidates
                        SET evidence_count = evidence_count + 1,
                            confidence = MAX(confidence, ?),
                            importance = MAX(importance, ?),
                            source_kind = ?,
                            last_seen_at = ?,
                            expires_at = ?
                        WHERE id = ?
                        """,
                        (
                            confidence,
                            importance,
                            _better_source_kind(str(duplicate["source_kind"] or "candidate"), source_kind),
                            now,
                            expires_text,
                            candidate_id,
                        ),
                    )
                    return candidate_id
                cursor = conn.execute(
                    """
                    INSERT INTO memory_candidates(chat_id, type, content, tags, importance, source_kind, confidence,
                                                  source_ref, content_hash, evidence_count, first_seen_at,
                                                  last_seen_at, expires_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chat_id,
                        memory_type,
                        normalized_content,
                        tag_text,
                        importance,
                        source_kind,
                        confidence,
                        source_ref or "",
                        content_hash,
                        1,
                        now,
                        now,
                        expires_text,
                        "pending",
                    ),
                )
                return int(cursor.lastrowid)

        return await asyncio.to_thread(run)

    async def upsert_memory_embedding(self, chat_id: str, memory_id: int, embedding: list[float]) -> None:
        """写入或更新某条长期记忆的 embedding。

        参数:
            chat_id: 记忆归属 chat id。
            memory_id: memories 表 id。
            embedding: float 向量，会以 JSON 字符串保存。
        """
        vector_text = json.dumps([float(value) for value in embedding])

        def run() -> None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO memory_embeddings(memory_id, chat_id, embedding, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(memory_id) DO UPDATE SET
                        chat_id = excluded.chat_id,
                        embedding = excluded.embedding,
                        updated_at = excluded.updated_at
                    """,
                    (memory_id, chat_id, vector_text, _utc_now_iso()),
                )

        await asyncio.to_thread(run)

    async def list_memory_embeddings(self, chat_id: str) -> list[dict[str, Any]]:
        """列出指定 chat 的所有 memory embedding。

        返回:
            每项包含 memory_id 和 embedding。损坏的 JSON 向量会被跳过。
        """
        def run() -> list[dict[str, Any]]:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT memory_id, embedding FROM memory_embeddings WHERE chat_id = ?",
                    (chat_id,),
                ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                try:
                    vector = json.loads(row["embedding"])
                except json.JSONDecodeError:
                    continue
                if isinstance(vector, list):
                    result.append({"memory_id": int(row["memory_id"]), "embedding": [float(value) for value in vector]})
            return result

        return await asyncio.to_thread(run)

    async def get_memories_by_ids(self, chat_id: str, memory_ids: list[int]) -> list[dict[str, Any]]:
        """按 id 批量读取长期记忆。

        参数:
            chat_id: 当前 chat id。
            memory_ids: memory id 列表。
        """
        if not memory_ids:
            return []

        def run() -> list[dict[str, Any]]:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            placeholders = ",".join("?" for _ in memory_ids)
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT *
                    FROM memories
                    WHERE chat_id = ? AND status = 'active' AND id IN ({placeholders})
                    """,
                    [chat_id, *memory_ids],
                ).fetchall()
                conn.execute(
                    f"UPDATE memories SET last_used_at = ? WHERE chat_id = ? AND id IN ({placeholders})",
                    [_utc_now_iso(), chat_id, *memory_ids],
                )
            return [_memory_row(row) for row in rows]

        return await asyncio.to_thread(run)

    async def get_memory_candidates(
        self,
        chat_id: str,
        status: str = "pending",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """读取指定 chat 的候选记忆。"""

        def run() -> list[dict[str, Any]]:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM memory_candidates
                    WHERE chat_id = ? AND status = ?
                    ORDER BY last_seen_at DESC, id DESC
                    LIMIT ?
                    """,
                    (chat_id, status, limit),
                ).fetchall()
            return [_candidate_row(row) for row in rows]

        return await asyncio.to_thread(run)

    async def promote_ready_candidates(self, chat_id: str, min_evidence: int = 2) -> list[dict[str, Any]]:
        """把满足证据阈值的候选记忆转正为长期记忆。"""

        def run() -> list[dict[str, Any]]:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            now = utc_now()
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM memory_candidates
                    WHERE chat_id = ? AND status = 'pending' AND evidence_count >= ?
                    ORDER BY last_seen_at DESC, id DESC
                    """,
                    (chat_id, min_evidence),
                ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                item = _candidate_row(row)
                expires_at = item.get("expires_at")
                if isinstance(expires_at, datetime) and expires_at < now:
                    continue
                result.append(item)
            return result

        return await asyncio.to_thread(run)

    async def archive_memory_candidate(self, chat_id: str, candidate_id: int, status: str = "promoted") -> bool:
        """归档一条候选记忆。"""

        def run() -> bool:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE memory_candidates
                    SET status = ?, last_seen_at = ?
                    WHERE chat_id = ? AND id = ? AND status = 'pending'
                    """,
                    (status, _utc_now_iso(), chat_id, candidate_id),
                )
                return cursor.rowcount > 0

        return await asyncio.to_thread(run)

    async def search_memories(self, chat_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """按关键词搜索长期记忆。

        参数:
            chat_id: 当前 chat id，只搜索该 chat 的记忆。
            query: 搜索词。为空时返回最近/重要的记忆。
            limit: 最多返回多少条。

        返回:
            记忆字典列表，同时会更新命中记忆的 last_used_at。
        """
        terms = [term for term in re_split_query(query) if term]

        def run() -> list[dict[str, Any]]:
            """在线程池中执行 SQLite 查询，并在返回前整理结果。"""
            with self._connect() as conn:
                if terms:
                    where = " OR ".join("(content LIKE ? OR tags LIKE ? OR type LIKE ?)" for _ in terms)
                    params: list[Any] = []
                    for term in terms:
                        like = f"%{term}%"
                        params.extend([like, like, like])
                    rows = conn.execute(
                        f"""
                        SELECT *
                        FROM memories
                        WHERE chat_id = ? AND status = 'active' AND ({where})
                        ORDER BY importance DESC, COALESCE(last_used_at, created_at) DESC, id DESC
                        LIMIT ?
                        """,
                        [chat_id, *params, limit],
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT *
                        FROM memories
                        WHERE chat_id = ? AND status = 'active'
                        ORDER BY importance DESC, id DESC
                        LIMIT ?
                        """,
                        (chat_id, limit),
                    ).fetchall()
                ids = [int(row["id"]) for row in rows]
                if ids:
                    placeholders = ",".join("?" for _ in ids)
                    conn.execute(f"UPDATE memories SET last_used_at = ? WHERE id IN ({placeholders})", [_utc_now_iso(), *ids])
            return [_memory_row(row) for row in rows]

        return await asyncio.to_thread(run)

    async def list_recent_memories(self, chat_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """列出最近/重要的长期记忆，等价于空 query 搜索。"""
        return await self.search_memories(chat_id, "", limit=limit)

    async def delete_memory(self, chat_id: str, memory_id: int) -> bool:
        """删除指定长期记忆。

        参数:
            chat_id: 当前 chat id，避免删除其他 chat 的记忆。
            memory_id: 记忆 id。

        返回:
            True 表示删除成功；False 表示未找到。
        """
        def run() -> bool:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                cursor = conn.execute(
                    "UPDATE memories SET status = 'deleted', updated_at = ? WHERE chat_id = ? AND id = ? AND status = 'active'",
                    (_utc_now_iso(), chat_id, memory_id),
                )
                return cursor.rowcount > 0

        return await asyncio.to_thread(run)

    async def count_memories(self, chat_id: str) -> int:
        """统计指定 chat 的长期记忆数量。"""
        def run() -> int:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS count FROM memories WHERE chat_id = ? AND status = 'active'",
                    (chat_id,),
                ).fetchone()
            return int(row["count"])

        return await asyncio.to_thread(run)

    async def list_active_memories(self, chat_id: str, limit: int = 200) -> list[dict[str, Any]]:
        """列出指定 chat 的活跃长期记忆。"""

        def run() -> list[dict[str, Any]]:
            """在线程池中读取状态为 active 的长期记忆列表。"""
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM memories
                    WHERE chat_id = ? AND status = 'active'
                    ORDER BY importance DESC, updated_at DESC, id DESC
                    LIMIT ?
                    """,
                    (chat_id, limit),
                ).fetchall()
            return [_memory_row(row) for row in rows]

        return await asyncio.to_thread(run)

    async def list_memory_replacements(self, chat_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """列出指定 chat 的记忆替换历史。"""

        def run() -> list[dict[str, Any]]:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT old_memory_id, new_memory_id, reason, created_at
                    FROM memory_replacements
                    WHERE chat_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (chat_id, limit),
                ).fetchall()
            return [dict(row) for row in rows]

        return await asyncio.to_thread(run)

    async def get_last_consolidated(self, chat_id: str) -> int:
        """读取指定 chat 的长期整理检查点；没有整理过时返回 0。"""

        def run() -> int:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT last_consolidated FROM consolidation_state WHERE chat_id = ?",
                    (chat_id,),
                ).fetchone()
            return int(row["last_consolidated"]) if row else 0

        return await asyncio.to_thread(run)

    async def set_last_consolidated(self, chat_id: str, message_id: int) -> None:
        """推进指定 chat 的整理检查点，只会向前推进。"""

        def run() -> None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO consolidation_state(chat_id, last_consolidated, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(chat_id) DO UPDATE SET
                        last_consolidated = MAX(last_consolidated, excluded.last_consolidated),
                        updated_at = excluded.updated_at
                    """,
                    (chat_id, message_id, _utc_now_iso()),
                )

        await asyncio.to_thread(run)

    async def has_consolidation_event(self, source_ref: str) -> bool:
        """检查某个整理 source_ref 是否已经记录过。"""

        def run() -> bool:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute("SELECT 1 FROM consolidation_events WHERE source_ref = ?", (source_ref,)).fetchone()
            return row is not None

        return await asyncio.to_thread(run)

    async def add_consolidation_event(
        self,
        chat_id: str,
        source_ref: str,
        status: str = "done",
        details: dict[str, Any] | None = None,
    ) -> bool:
        """写入幂等整理事件，source_ref 全局唯一。"""
        details_text = json.dumps(details or {}, ensure_ascii=False)

        def run() -> bool:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO consolidation_events(chat_id, source_ref, status, details_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (chat_id, source_ref, status, details_text, _utc_now_iso()),
                )
                return cursor.rowcount > 0

        return await asyncio.to_thread(run)

    async def supersede_memory(self, chat_id: str, old_memory_id: int, new_memory_id: int, reason: str = "") -> bool:
        """把旧记忆标记为 superseded，并记录新旧记忆的替换关系。"""

        def run() -> bool:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE memories
                    SET status = 'superseded', updated_at = ?
                    WHERE chat_id = ? AND id = ? AND status = 'active'
                    """,
                    (_utc_now_iso(), chat_id, old_memory_id),
                )
                if cursor.rowcount <= 0:
                    return False
                conn.execute(
                    """
                    INSERT INTO memory_replacements(chat_id, old_memory_id, new_memory_id, reason, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (chat_id, old_memory_id, new_memory_id, reason, _utc_now_iso()),
                )
                return True

        return await asyncio.to_thread(run)

    async def add_reminder(self, chat_id: str, user_id: str, content: str, due_at: datetime) -> int:
        """创建一条待发送提醒。

        参数:
            chat_id: 提醒发送目标 chat id。
            user_id: 创建提醒的用户 id。
            content: 提醒内容。
            due_at: 到期时间；会标准化为 UTC ISO 字符串保存。

        返回:
            新提醒的自增 id。
        """
        def run() -> int:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO reminders(chat_id, user_id, content, due_at, delivered_at, cancelled_at)
                    VALUES (?, ?, ?, ?, NULL, NULL)
                    """,
                    (chat_id, user_id, content, _to_iso(due_at)),
                )
                return int(cursor.lastrowid)

        return await asyncio.to_thread(run)

    async def list_pending_reminders(self, chat_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """列出指定 chat 未发送、未取消的提醒。"""
        def run() -> list[dict[str, Any]]:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT id, chat_id, user_id, content, due_at
                    FROM reminders
                    WHERE chat_id = ? AND delivered_at IS NULL AND cancelled_at IS NULL
                    ORDER BY due_at ASC
                    LIMIT ?
                    """,
                    (chat_id, limit),
                ).fetchall()
            return [_reminder_row(row) for row in rows]

        return await asyncio.to_thread(run)

    async def cancel_reminder(self, chat_id: str, reminder_id: int) -> bool:
        """取消一条未发送提醒。

        参数:
            chat_id: 当前 chat id。
            reminder_id: 提醒 id。

        返回:
            True 表示取消成功；False 表示提醒不存在、已发送或已取消。
        """
        def run() -> bool:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE reminders SET cancelled_at = ?
                    WHERE chat_id = ? AND id = ? AND delivered_at IS NULL AND cancelled_at IS NULL
                    """,
                    (_utc_now_iso(), chat_id, reminder_id),
                )
                return cursor.rowcount > 0

        return await asyncio.to_thread(run)

    async def count_pending_reminders(self, chat_id: str) -> int:
        """统计指定 chat 当前待发送提醒数量。"""
        def run() -> int:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS count FROM reminders
                    WHERE chat_id = ? AND delivered_at IS NULL AND cancelled_at IS NULL
                    """,
                    (chat_id,),
                ).fetchone()
            return int(row["count"])

        return await asyncio.to_thread(run)

    async def get_due_reminders(self, now: datetime | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """查询已经到期且尚未发送的提醒。

        参数:
            now: 判断到期的基准时间；为空时使用 utc_now()。
            limit: 单次最多返回多少条。
        """
        now = now or utc_now()

        def run() -> list[dict[str, Any]]:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT id, chat_id, user_id, content, due_at
                    FROM reminders
                    WHERE delivered_at IS NULL AND cancelled_at IS NULL AND due_at <= ?
                    ORDER BY due_at ASC
                    LIMIT ?
                    """,
                    (_to_iso(now), limit),
                ).fetchall()
            return [_reminder_row(row) for row in rows]

        return await asyncio.to_thread(run)

    async def mark_reminder_delivered(self, reminder_id: int) -> None:
        """把提醒标记为已发送，避免 proactive loop 重复推送。"""
        def run() -> None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                conn.execute("UPDATE reminders SET delivered_at = ? WHERE id = ?", (_utc_now_iso(), reminder_id))

        await asyncio.to_thread(run)

    async def add_message_trace(
        self,
        chat_id: str,
        user_message: str,
        assistant_reply: str,
        tools_used: list[str],
        memory_hits: list[dict[str, Any]],
        latency_ms: int,
        error: str | None = None,
        model_main: str = "",
        model_fast: str = "",
        mcp_tools_used: list[str] | None = None,
        hyde_used: bool = False,
        attachments_count: int = 0,
    ) -> None:
        """写入一条被动消息处理 trace。

        参数:
            chat_id: 当前 chat id。
            user_message: 用户输入文本。
            assistant_reply: 助手最终回复。
            tools_used: 本轮使用的内置工具名称。
            memory_hits: 注入上下文的结构化命中摘要。
            latency_ms: 本轮总耗时毫秒。
            error: 可选错误信息。
            model_main: 主模型名称。
            model_fast: 快模型名称。
            mcp_tools_used: 本轮使用的 MCP 工具名称。
            hyde_used: 是否启用了 HyDE 记忆检索。
            attachments_count: 本轮附件数量。
        """
        mcp_tools_used = mcp_tools_used or []

        def run() -> None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO message_trace(chat_id, user_message, assistant_reply, model_main, model_fast,
                                              tools_used, mcp_tools_used, memory_hits, hyde_used,
                                              attachments_count, latency_ms, error, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chat_id,
                        user_message,
                        assistant_reply,
                        model_main,
                        model_fast,
                        json.dumps(tools_used, ensure_ascii=False),
                        json.dumps(mcp_tools_used, ensure_ascii=False),
                        json.dumps(memory_hits, ensure_ascii=False),
                        1 if hyde_used else 0,
                        attachments_count,
                        latency_ms,
                        error,
                        _utc_now_iso(),
                    ),
                )

        await asyncio.to_thread(run)

    async def add_proactive_tick_log(
        self,
        action: str,
        skip_reason: str | None,
        reminders_due: int,
        sent_message: str | None,
        error: str | None = None,
        content_count: int = 0,
        sent_count: int = 0,
    ) -> None:
        """写入一条主动循环 tick 记录。

        参数:
            action: sent、skip、error、drift 等动作类型。
            skip_reason: 跳过原因，例如 no_source、cooldown、busy。
            reminders_due: 本轮到期提醒数量。
            sent_message: 实际发送内容或简短说明。
            error: 可选错误信息。
            content_count: feed 事件数量。
            sent_count: 本轮主动发送数量。
        """
        def run() -> None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO proactive_tick_log(tick_at, action, skip_reason, reminders_due, content_count, sent_count, sent_message, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (_utc_now_iso(), action, skip_reason, reminders_due, content_count, sent_count, sent_message, error),
                )

        await asyncio.to_thread(run)

    async def get_last_proactive_tick(self) -> dict[str, Any] | None:
        """读取最近一次 proactive tick 记录。"""
        def run() -> dict[str, Any] | None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT tick_at, action, skip_reason, reminders_due, content_count, sent_count, sent_message, error
                    FROM proactive_tick_log
                    ORDER BY id DESC
                    LIMIT 1
                    """
                ).fetchone()
            return dict(row) if row else None

        return await asyncio.to_thread(run)

    async def add_proactive_candidate(
        self,
        chat_id: str,
        candidate_id: str,
        source_type: str,
        title: str,
        body: str,
        url: str,
        confidence: float,
        novelty: float,
        user_fit: float,
        priority: float,
        shareable: bool,
        dedupe_key: str,
        artifact_path: str | None,
        created_at: datetime,
        expires_at: datetime | None,
        score: float,
        status: str,
        drop_reason: str | None = None,
        sent_at: datetime | None = None,
    ) -> int:
        """记录一条 proactive 候选审计记录。"""

        def run() -> int:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO proactive_candidates(
                        candidate_id, chat_id, source_type, title, body, url, confidence, novelty,
                        user_fit, priority, shareable, dedupe_key, artifact_path, created_at, expires_at,
                        score, status, drop_reason, sent_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate_id,
                        chat_id,
                        source_type,
                        title,
                        body,
                        url,
                        float(confidence),
                        float(novelty),
                        float(user_fit),
                        float(priority),
                        1 if shareable else 0,
                        dedupe_key,
                        artifact_path,
                        _to_iso(created_at),
                        _to_iso(expires_at) if expires_at else None,
                        float(score),
                        status,
                        drop_reason,
                        _to_iso(sent_at) if sent_at else None,
                    ),
                )
                return int(cursor.lastrowid)

        return await asyncio.to_thread(run)

    async def mark_seen_item(self, item_key: str, source: str, title: str | None = None, url: str | None = None) -> None:
        """把 feed 事件标记为已见，避免重复推送。"""
        def run() -> None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO seen_items(item_key, source, title, url, seen_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (item_key, source, title, url, _utc_now_iso()),
                )

        await asyncio.to_thread(run)

    async def has_seen_item(self, item_key: str) -> bool:
        """检查 feed 事件是否已经处理过。"""
        def run() -> bool:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute("SELECT 1 FROM seen_items WHERE item_key = ?", (item_key,)).fetchone()
            return row is not None

        return await asyncio.to_thread(run)

    async def count_seen_items(self) -> int:
        """统计 seen_items 表中已处理 feed 事件总数。"""
        def run() -> int:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) AS count FROM seen_items").fetchone()
            return int(row["count"])

        return await asyncio.to_thread(run)

    async def add_mcp_tool_log(
        self,
        server: str,
        tool: str,
        args_preview: str,
        result_preview: str,
        latency_ms: int,
        error: str | None = None,
    ) -> None:
        """写入一条 MCP 工具调用日志。

        参数:
            server: MCP server 名称。
            tool: MCP 工具名。
            args_preview: 参数预览，会裁剪到 500 字符。
            result_preview: 结果预览，会裁剪到 500 字符。
            latency_ms: 调用耗时毫秒。
            error: 可选错误信息。
        """
        def run() -> None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO mcp_tool_log(server, tool, args_preview, result_preview, latency_ms, error, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (server, tool, args_preview[:500], result_preview[:500], latency_ms, error, _utc_now_iso()),
                )

        await asyncio.to_thread(run)

    async def add_drift_run(
        self,
        task_id: str,
        title: str,
        result: str,
        output_path: str | None = None,
        error: str | None = None,
    ) -> int:
        """记录一次 drift 运行结果。

        参数:
            task_id: drift 任务 id。
            title: 任务标题。
            result: drift 输出正文。
            output_path: 写入的 Markdown 文件路径。
            error: 可选错误信息。

        返回:
            drift_runs 自增 id。
        """
        def run() -> int:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO drift_runs(task_id, title, result, output_path, notified, error, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (task_id, title, result, output_path, 0, error, _utc_now_iso()),
                )
                return int(cursor.lastrowid)

        return await asyncio.to_thread(run)

    async def last_drift_run_at(self) -> datetime | None:
        """读取最近一次 drift 运行时间。"""
        def run() -> datetime | None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute("SELECT created_at FROM drift_runs ORDER BY id DESC LIMIT 1").fetchone()
            return _from_iso(row["created_at"]) if row else None

        return await asyncio.to_thread(run)

    async def count_drift_runs_since(self, since: datetime) -> int:
        """统计某个时间点之后的 drift 运行次数。"""
        def run() -> int:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS count FROM drift_runs WHERE created_at >= ?",
                    (_to_iso(since),),
                ).fetchone()
            return int(row["count"])

        return await asyncio.to_thread(run)

    async def get_last_drift_run(self) -> dict[str, Any] | None:
        """读取最近一次 drift 运行记录。"""
        def run() -> dict[str, Any] | None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT task_id, title, result, output_path, notified, error, created_at
                    FROM drift_runs
                    ORDER BY id DESC
                    LIMIT 1
                    """
                ).fetchone()
            return dict(row) if row else None

        return await asyncio.to_thread(run)

    async def get_drift_task_states(self) -> dict[str, dict[str, Any]]:
        """读取所有 drift 任务状态。"""

        def run() -> dict[str, dict[str, Any]]:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT task_id, last_run_at, last_status, failure_count, last_artifact_path, last_artifact_at
                    FROM drift_task_state
                    """
                ).fetchall()
            states: dict[str, dict[str, Any]] = {}
            for row in rows:
                item = dict(row)
                item["last_run_at"] = _from_iso(item.get("last_run_at"))
                item["last_artifact_at"] = _from_iso(item.get("last_artifact_at"))
                states[str(item["task_id"])] = item
            return states

        return await asyncio.to_thread(run)

    async def update_drift_task_state(
        self,
        task_id: str,
        last_status: str,
        last_run_at: datetime | None = None,
        artifact_path: str | None = None,
        artifact_at: datetime | None = None,
        reset_failures: bool = False,
        increment_failures: bool = False,
    ) -> None:
        """更新单个 drift 任务的状态。"""

        def run() -> None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT failure_count, last_artifact_path, last_artifact_at
                    FROM drift_task_state
                    WHERE task_id = ?
                    """,
                    (task_id,),
                ).fetchone()
                failure_count = int(row["failure_count"]) if row else 0
                if reset_failures:
                    failure_count = 0
                elif increment_failures:
                    failure_count += 1
                conn.execute(
                    """
                    INSERT INTO drift_task_state(task_id, last_run_at, last_status, failure_count, last_artifact_path, last_artifact_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(task_id) DO UPDATE SET
                        last_run_at = excluded.last_run_at,
                        last_status = excluded.last_status,
                        failure_count = excluded.failure_count,
                        last_artifact_path = excluded.last_artifact_path,
                        last_artifact_at = excluded.last_artifact_at
                    """,
                    (
                        task_id,
                        _to_iso(last_run_at) if last_run_at else None,
                        last_status,
                        failure_count,
                        artifact_path or (str(row["last_artifact_path"]) if row else ""),
                        _to_iso(artifact_at) if artifact_at else (_to_iso(_from_iso(row["last_artifact_at"])) if row and row["last_artifact_at"] else None),
                    ),
                )

        await asyncio.to_thread(run)

    async def add_proactive_delivery(self, chat_id: str, message: str, source: str) -> None:
        """记录一条已经主动发送给用户的消息。

        参数:
            chat_id: 目标 chat id。
            message: 已发送正文。
            source: 主动来源，例如 reminder、feed、drift、fallback。
        """
        def run() -> None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO proactive_deliveries(chat_id, message, source, delivered_at) VALUES (?, ?, ?, ?)",
                    (chat_id, message, source, _utc_now_iso()),
                )

        await asyncio.to_thread(run)

    async def count_non_reminder_proactive_deliveries_since(self, chat_id: str, since: datetime) -> int:
        """统计指定 chat 的非 reminder 主动发送次数。"""

        def run() -> int:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS count FROM proactive_deliveries
                    WHERE chat_id = ? AND delivered_at >= ? AND source != 'reminder'
                    """,
                    (chat_id, _to_iso(since)),
                ).fetchone()
            return int(row["count"])

        return await asyncio.to_thread(run)

    async def count_proactive_deliveries_for_source_since(self, chat_id: str, source: str, since: datetime) -> int:
        """统计指定 chat 某个 source 的主动发送次数。"""

        def run() -> int:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS count FROM proactive_deliveries
                    WHERE chat_id = ? AND source = ? AND delivered_at >= ?
                    """,
                    (chat_id, source, _to_iso(since)),
                ).fetchone()
            return int(row["count"])

        return await asyncio.to_thread(run)

    async def count_proactive_deliveries_since(self, chat_id: str, since: datetime) -> int:
        """统计指定 chat 从某个时间点后的主动发送次数。"""
        def run() -> int:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS count FROM proactive_deliveries
                    WHERE chat_id = ? AND delivered_at >= ?
                    """,
                    (chat_id, _to_iso(since)),
                ).fetchone()
            return int(row["count"])

        return await asyncio.to_thread(run)

    async def last_proactive_delivery_at(self, chat_id: str) -> datetime | None:
        """读取指定 chat 最近一次主动发送时间。"""
        def run() -> datetime | None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT delivered_at FROM proactive_deliveries
                    WHERE chat_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (chat_id,),
                ).fetchone()
            return _from_iso(row["delivered_at"]) if row else None

        return await asyncio.to_thread(run)

    async def last_non_reminder_proactive_delivery_at(self, chat_id: str) -> datetime | None:
        """读取指定 chat 最近一次非 reminder 主动发送时间。"""

        def run() -> datetime | None:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT delivered_at FROM proactive_deliveries
                    WHERE chat_id = ? AND source != 'reminder'
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (chat_id,),
                ).fetchone()
            return _from_iso(row["delivered_at"]) if row else None

        return await asyncio.to_thread(run)


def _normalize_tags(tags: list[str] | str | None) -> list[str]:
    """把 tags 参数标准化成字符串列表。"""
    if tags is None:
        return []
    if isinstance(tags, str):
        return [tag for tag in re_split_query(tags) if tag]
    return [str(tag) for tag in tags if str(tag).strip()]


def re_split_query(query: str) -> list[str]:
    """用空格、英文逗号和中文逗号切分简单检索词。"""
    return [term for term in query.replace("，", " ").replace(",", " ").split() if term]


def _memory_row(row: sqlite3.Row) -> dict[str, Any]:
    """把 memories 表的一行转换成普通 dict，并解析 tags JSON。"""
    item = dict(row)
    try:
        item["tags"] = json.loads(item.get("tags") or "[]")
    except json.JSONDecodeError:
        item["tags"] = re_split_query(item.get("tags") or "")
    try:
        item["extra"] = json.loads(item.get("extra_json") or "{}")
    except json.JSONDecodeError:
        item["extra"] = {}
    return item


def _candidate_row(row: sqlite3.Row) -> dict[str, Any]:
    """把 memory_candidates 表的一行转换成普通 dict。"""
    item = dict(row)
    try:
        item["tags"] = json.loads(item.get("tags") or "[]")
    except json.JSONDecodeError:
        item["tags"] = re_split_query(item.get("tags") or "")
    item["first_seen_at"] = _from_iso(item.get("first_seen_at"))
    item["last_seen_at"] = _from_iso(item.get("last_seen_at"))
    item["expires_at"] = _from_iso(item.get("expires_at"))
    return item


def _content_hash(content: str) -> str:
    """生成稳定内容哈希，用于重复记忆强化和幂等写入。"""
    return hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()


def _better_source_kind(current: str, incoming: str) -> str:
    """返回两个 source_kind 中更可信的一个。"""
    ranks = {
        "explicit": 4,
        "promoted": 3,
        "inferred": 2,
        "candidate": 1,
    }
    return incoming if ranks.get(incoming, 0) >= ranks.get(current, 0) else current


def _reminder_row(row: sqlite3.Row) -> dict[str, Any]:
    """把 reminders 表的一行转换成普通 dict，并恢复 due_at datetime。"""
    item = dict(row)
    item["due_at"] = _from_iso(item["due_at"])
    return item
