"""运行观测与被动消息 trace 写入辅助。

项目的观测数据包含几类表：被动回复写入 message_trace，主动循环写入
proactive_tick_log/proactive_deliveries，MCP 调用写入 mcp_tool_log。
本模块里的 TraceRecorder 目前只封装被动回复 trace，避免 AgentLoop 直接拼 message_trace 字段。
"""

from __future__ import annotations

from chat_agent.memory.store import SQLiteStore


class TraceRecorder:
    """被动回复 trace 记录器。

    该类是 AgentLoop 写 message_trace 的薄封装，目的是让业务层不直接关心表结构。
    主动消息由 ProactiveLoop 直接写 proactive_tick_log/proactive_deliveries。
    未来如果要把 trace 同步到文件、OpenTelemetry 或其他可观测系统，可以从这里扩展。
    """

    def __init__(self, store: SQLiteStore) -> None:
        """初始化 TraceRecorder。

        参数:
            store: SQLiteStore，实际写入 add_message_trace。
        """
        self.store = store

    async def record_message(
        self,
        chat_id: str,
        user_message: str,
        assistant_reply: str,
        tools_used: list[str],
        memory_hits: list[dict],
        latency_ms: int,
        error: str | None = None,
        model_main: str = "",
        model_fast: str = "",
        mcp_tools_used: list[str] | None = None,
        hyde_used: bool = False,
        attachments_count: int = 0,
    ) -> None:
        """记录一轮被动消息处理。

        参数:
            chat_id: 当前 chat id。
            user_message: 用户输入。
            assistant_reply: 助手最终回复。
            tools_used: 内置工具使用列表。
            memory_hits: 注入 prompt 的结构化记忆命中摘要。
            latency_ms: 处理耗时毫秒。
            error: 可选错误信息。
            model_main: 主模型名称。
            model_fast: 快模型名称。
            mcp_tools_used: MCP 工具使用列表。
            hyde_used: 是否启用了 HyDE 检索。
            attachments_count: 附件数量。
        """
        await self.store.add_message_trace(
            chat_id=chat_id,
            user_message=user_message,
            assistant_reply=assistant_reply,
            tools_used=tools_used,
            memory_hits=memory_hits,
            latency_ms=latency_ms,
            error=error,
            model_main=model_main,
            model_fast=model_fast,
            mcp_tools_used=mcp_tools_used,
            hyde_used=hyde_used,
            attachments_count=attachments_count,
        )
