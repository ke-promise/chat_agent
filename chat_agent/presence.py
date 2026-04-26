"""用户在线状态与 busy 状态管理。

proactive loop 在主动发送前会读取 PresenceTracker，避免用户刚发完消息或该 chat 正在处理时
继续插入主动推送。这个模块只保存运行期状态，不做持久化。
"""

from __future__ import annotations

from datetime import timedelta

from chat_agent.memory.store import SQLiteStore, utc_now


class PresenceTracker:
    """用户在线/忙碌状态跟踪器。

    用途:
        - AgentLoop 处理某个 chat 时调用 mark_busy/mark_idle。
        - ProactiveLoop 发送主动消息前检查 is_busy，避免打断正在处理中的会话。
        - is_active 可根据 chats.last_seen_at 判断用户近期是否活跃，供后续主动策略扩展。
    """

    def __init__(self, store: SQLiteStore, active_window_minutes: int = 10) -> None:
        """初始化 PresenceTracker。

        参数:
            store: SQLiteStore，用于读取 chat 最后活跃时间。
            active_window_minutes: 多久以内有消息视为 active。
        """
        self.store = store
        self.active_window = timedelta(minutes=active_window_minutes)
        self._busy: set[str] = set()

    def mark_busy(self, chat_id: str) -> None:
        """标记指定 chat 正在被动处理消息。"""
        self._busy.add(chat_id)

    def mark_idle(self, chat_id: str) -> None:
        """清除指定 chat 的 busy 标记。"""
        self._busy.discard(chat_id)

    def is_busy(self, chat_id: str) -> bool:
        """判断指定 chat 当前是否正在处理消息。"""
        return chat_id in self._busy

    async def is_active(self, chat_id: str) -> bool:
        """判断指定 chat 是否在 active_window 内活跃过。"""
        last_seen = await self.store.get_last_seen_at(chat_id)
        if last_seen is None:
            return False
        return utc_now() - last_seen <= self.active_window
