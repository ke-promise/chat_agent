from __future__ import annotations

from pathlib import Path

import pytest

from chat_agent.config import ConfigError, load_config


def test_load_config_expands_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QWEN_API_KEY", "key")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
        [llm.main]
        model = "qwen-vl-plus"
        api_key = "${QWEN_API_KEY}"
        base_url = "https://example.test/v1"
        timeout_seconds = 12
        max_tokens = 300
        enable_vision = true

        [llm.fast]
        model = "qwen-flash"
        api_key = "${QWEN_API_KEY}"
        base_url = "https://example.test/v1"
        enable_vision = false

        [telegram]
        token = "${TELEGRAM_BOT_TOKEN}"
        allow_from = ["@alice"]
        download_images = true
        image_max_mb = 9

        [embedding]
        enabled = true
        provider = "sqlite_json"
        model = "text-embedding-v4"
        api_key = "${QWEN_API_KEY}"
        base_url = "https://example.test/v1"
        dimension = 1024
        top_k = 7
        min_score = 0.3
        external_url = "http://localhost:8000"
        external_api_key = "vector-key"

        [mcp]
        enabled = true
        config_path = "workspace/mcp_servers.json"
        allowed_servers = ["duckduckgo"]
        allowed_tools = ["duckduckgo:web_search"]

        [tools]
        extra_model_tools = ["write_file", "send_message"]

        [skills]
        enabled = true
        builtin_dir = "skills"
        workspace_dir = "workspace/skills"
        inject_catalog = true
        max_catalog_chars = 1234

        [proactive.loop]
        enabled = true
        tick_interval_seconds = 45
        target_chat_id = "chat-1"

        [proactive.budget]
        daily_max = 4
        min_interval_minutes = 120
        quiet_hours_start = "23:00"
        quiet_hours_end = "07:00"

        [proactive.feed]
        enabled = true
        sources_path = "workspace/proactive_sources.json"
        daily_cap = 3

        [proactive.drift]
        enabled = true
        tasks_path = "workspace/drift_tasks.json"
        output_dir = "workspace/drift_runs"
        run_cooldown_minutes = 30
        daily_run_cap = 2
        promotion_enabled = true
        daily_cap = 1

        [proactive.drift.skills]
        enabled = true
        workspace_dir = "workspace/drift/skills"
        include_builtin = false

        [proactive.fallback]
        enabled = true
        probability = 0.2
        daily_cap = 2

        [logging]
        file = "logs/test.log"
        """,
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.llm.main.api_key == "key"
    assert config.llm.main.timeout_seconds == 12
    assert config.llm.main.max_tokens == 300
    assert config.llm.main.enable_vision is True
    assert config.llm.fast.model == "qwen-flash"
    assert config.llm.fast.enable_vision is False
    assert config.telegram.token == "token"
    assert config.telegram.allow_from == ["alice"]
    assert config.telegram.image_max_mb == 9
    assert config.memory.database_path == tmp_path / "workspace" / "agent.sqlite3"
    assert config.embedding.enabled is True
    assert config.embedding.provider == "sqlite_json"
    assert config.embedding.top_k == 7
    assert config.embedding.min_score == 0.3
    assert config.embedding.external_api_key == "vector-key"
    assert config.logging.file == tmp_path / "logs" / "test.log"
    assert config.observe.database_path == tmp_path / "observe" / "observe.db"
    assert config.tools.file_workspace == tmp_path / "workspace" / "files"
    assert config.tools.extra_model_tools == ["write_file", "send_message"]
    assert config.mcp.enabled is True
    assert config.mcp.allowed_servers == ["duckduckgo"]
    assert config.mcp.allowed_tools == ["duckduckgo:web_search"]
    assert config.proactive.feed.enabled is True
    assert config.proactive.drift.enabled is True
    assert config.proactive.loop.tick_interval_seconds == 45
    assert config.proactive.loop.target_chat_id == "chat-1"
    assert config.proactive.budget.daily_max == 4
    assert config.proactive.budget.min_interval_minutes == 120
    assert config.proactive.feed.daily_cap == 3
    assert config.proactive.drift.daily_run_cap == 2
    assert config.proactive.drift.daily_cap == 1
    assert config.proactive.fallback.daily_cap == 2
    assert config.proactive.drift.skills_enabled is True
    assert config.proactive.drift.skills_include_builtin is False
    assert config.skills.enabled is True
    assert config.skills.max_catalog_chars == 1234
    assert config.skills.workspace_dir == tmp_path / "workspace" / "skills"


def test_load_config_missing_env_is_clear(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
        [llm.main]
        model = "qwen-plus"
        api_key = "${QWEN_API_KEY}"
        base_url = "https://example.test/v1"

        [telegram]
        token = "token"
        """,
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="QWEN_API_KEY"):
        load_config(config_file)


def test_load_config_rejects_legacy_proactive_schema(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QWEN_API_KEY", "key")
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
        [llm.main]
        model = "qwen-plus"
        api_key = "${QWEN_API_KEY}"
        base_url = "https://example.test/v1"

        [telegram]
        token = "token"

        [proactive]
        tick_interval_seconds = 60

        [proactive.feed]
        poll_interval_seconds = 150

        [proactive.drift]
        notify = true
        """,
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Deprecated proactive config keys"):
        load_config(config_file)
