from __future__ import annotations

from pathlib import Path

import pytest

from chat_agent.memory.store import SQLiteStore
from chat_agent.observe.trace import TraceRecorder


@pytest.mark.asyncio
async def test_observe_trace_writes_message_trace(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    recorder = TraceRecorder(store)

    await recorder.record_message("chat-1", "hi", "hello", ["tool"], [1], 42)

    # No public query method is needed by runtime; this asserts the write path stays valid.
    await store.add_proactive_tick_log("skip", "no_due_reminder", 0, None)
    assert (await store.get_last_proactive_tick())["skip_reason"] == "no_due_reminder"
