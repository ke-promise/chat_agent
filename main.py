"""项目命令行入口。

负责初始化工作目录、加载配置并启动 Telegram 智能体运行时。
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
from pathlib import Path

from chat_agent.config import AppConfig, ConfigError, load_config
from chat_agent.logging_setup import setup_logging


ROOT = Path(__file__).resolve().parent
EXAMPLE_CONFIG = ROOT / "config.example.toml"
EXAMPLE_CONFIG_CONTENT = """[llm.main]
model = "qwen-vl-plus"
api_key = "${QWEN_API_KEY}"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
timeout_seconds = 60
max_tokens = 4096
enable_vision = true

[llm.fast]
model = "qwen-flash"
api_key = "${QWEN_API_KEY}"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
timeout_seconds = 20
max_tokens = 1024
enable_vision = false

[channel]
# telegram 或 qq
type = "telegram"

[telegram]
token = "${TELEGRAM_BOT_TOKEN}"
allow_from = ["your_username"]
unauthorized_reply = true
download_images = true
image_max_mb = 10

[qq]
app_id = "${QQ_BOT_APP_ID}"
app_secret = "${QQ_BOT_APP_SECRET}"
sandbox = false
host = "0.0.0.0"
port = 8080
path = "/qqbot"
verify_signature = true
# QQ openid / 群成员 openid 白名单；留空表示允许所有来源。
allow_from = []
unauthorized_reply = true
download_images = false
image_max_mb = 10
max_text_chars = 1800

[memory]
enabled = true
database_path = "workspace/agent.sqlite3"
history_window = 20
top_k = 5
summary_enabled = true
summary_after_messages = 40
max_messages_per_chat = 500
query_rewrite_enabled = true
hyde_enabled = true

[embedding]
enabled = false
provider = "sqlite_json"
model = "text-embedding-v4"
api_key = "${QWEN_API_KEY}"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
timeout_seconds = 30
dimension = 1024
top_k = 5
min_score = 0.2
external_url = "http://localhost:8000"
external_api_key = ""
collection = "chat_agent_memories"

[tools]
tool_calling_enabled = true
max_iterations = 8
tool_search_enabled = true
web_fetch_timeout_seconds = 10
file_workspace = "workspace/files"
# 本项目内置工具：
# 默认可见：memorize、recall_memory、create_reminder、list_reminders、tool_search、list_memes
# 可通过 tool_search 发现：cancel_reminder、web_fetch、list_files、read_file
# 当 skills.enabled=true 时：list_skills、read_skill 默认可见；create_skill、update_skill 仍默认隐藏
# 默认隐藏；确实需要时可以在这里暴露给模型：
# write_file, send_message, send_emoji, send_meme, create_skill, update_skill
# MCP 工具也可以在加载后按注册名写到这里，例如：
# duckduckgo_web_search, rss_get_content, web_content_fetch_page,
# feed_bridge_poll_feeds, feed_bridge_get_proactive_events, feed_bridge_ack_events
extra_model_tools = []

[mcp]
enabled = true
config_path = "workspace/mcp_servers.json"
# 当前项目/工作区中可用的 MCP server：
# duckduckgo, rss, web_content, feed_bridge
allowed_servers = ["duckduckgo", "rss", "web_content", "feed_bridge"]
# allowed_tools 使用原始 MCP 名称：<server>:<tool>
# 当前项目确认会用到的工具：
# duckduckgo:web-search
# rss:get_content
# web_content:fetch_page
# feed_bridge:poll_feeds, feed_bridge:get_proactive_events, feed_bridge:ack_events
allowed_tools = [
  "duckduckgo:web-search",
  "rss:get_content",
  "web_content:fetch_page",
  "feed_bridge:poll_feeds",
  "feed_bridge:get_proactive_events",
  "feed_bridge:ack_events",
]

[skills]
enabled = true
builtin_dir = "skills"
workspace_dir = "workspace/skills"
inject_catalog = true
max_catalog_chars = 4000

[proactive.loop]
enabled = true
tick_interval_seconds = 60
target_chat_id = ""

[proactive.budget]
daily_max = 6
min_interval_minutes = 90
quiet_hours_start = ""
quiet_hours_end = ""

[proactive.fallback]
enabled = false
probability = 0.03
daily_cap = 2

[proactive.feed]
enabled = true
sources_path = "workspace/proactive_sources.json"
daily_cap = 3

[proactive.drift]
enabled = false
tasks_path = "workspace/drift_tasks.json"
output_dir = "workspace/drift_runs"
run_cooldown_minutes = 180
daily_run_cap = 3
promotion_enabled = true
daily_cap = 2

[proactive.drift.skills]
enabled = true
workspace_dir = "workspace/drift/skills"
include_builtin = true

[proactive.presence]
active_window_minutes = 10
skip_when_busy = true

[scheduler]
enabled = true
max_due_per_tick = 50

[logging]
level = "INFO"
file = "logs/app.log"

[observe]
database_path = "observe/observe.db"
"""
MCP_SERVERS_EXAMPLE = """{
  "servers": {
    "duckduckgo": {
      "enabled": true,
      "command": ["node", "package\\\\bin\\\\cli.js"],
      "env": {}
    },
    "rss": {
      "enabled": false,
      "command": ["npx.cmd", "-y", "mcp_rss"],
      "env": {
        "OPML_FILE_PATH": "workspace\\\\rss_feeds.opml",
        "DB_HOST": "localhost",
        "DB_PORT": "3306",
        "DB_USERNAME": "root",
        "DB_PASSWORD": "123456",
        "DB_DATABASE": "mcp_rss",
        "RSS_UPDATE_INTERVAL": "1"
      }
    },
    "web_content": {
      "enabled": true,
      "command": [".venv\\\\Scripts\\\\python.exe", "workspace\\\\mcp_servers\\\\web_content_mcp_server.py"],
      "env": {
        "PYTHONUTF8": "1"
      }
    },
    "feed_bridge": {
      "enabled": true,
      "command": [".venv\\\\Scripts\\\\python.exe", "workspace\\\\mcp_servers\\\\feed_bridge_mcp_server.py"],
      "env": {
        "PYTHONUTF8": "1",
        "WEB_FEEDS": "[{\\"name\\":\\"python-insider\\",\\"url\\":\\"https://blog.python.org/rss.xml\\"}]"
      }
    }
  }
}
"""
PROACTIVE_SOURCES_EXAMPLE = """{
  "sources": [
    {
      "server": "rss",
      "channel": "content",
      "poll_tool": null,
      "get_tool": "get_content",
      "get_args": {
        "status": "normal",
        "limit": 10
      },
      "ack_tool": null,
      "enabled": false
    },
    {
      "server": "feed_bridge",
      "channel": "content",
      "poll_tool": "poll_feeds",
      "get_tool": "get_proactive_events",
      "ack_tool": "ack_events",
      "enabled": true
    }
  ]
}
"""
RSS_FEEDS_OPML_EXAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<opml version="2.0">
  <head>
    <title>chat_agent feeds</title>
  </head>
  <body>
    <outline text="Python Blog" title="Python Blog" type="rss" xmlUrl="https://www.python.org/blogs/rss/" />
  </body>
</opml>
"""
DRIFT_TASKS_EXAMPLE = """{
  "tasks": [
    {
      "id": "memory_review",
      "title": "整理近期记忆线索",
      "prompt": "阅读近期摘要、长期记忆和待提醒，生成一份后台整理笔记，指出用户最近关注的话题、可能的待跟进事项，以及是否有值得后续主动提起的线索。只有当内容真正适合直接告诉用户时，才在结果头部元数据里把 shareable 设为 true。",
      "enabled": true
    },
    {
      "id": "followup_draft",
      "title": "准备轻量跟进草稿",
      "prompt": "根据近期上下文准备一份内部 follow-up 草稿，帮助后续在合适时机继续陪伴用户。默认不要直接对用户发送，除非你能明确给出高置信、高相关的可分享内容。",
      "enabled": true
    }
  ]
}
"""


def init_project() -> None:
    """初始化项目运行所需的目录和示例文件。

    功能:
        - 创建 `workspace/`、`logs/`、`observe/` 等本地目录。
        - 在配置文件缺失时写入 `config.example.toml`、MCP 示例配置和 feed 示例配置。
        - 保留用户已有文件，不覆盖现有配置。
    """
    workspace = ROOT / "workspace"
    logs = ROOT / "logs"
    observe = ROOT / "observe"
    memory_workspace = workspace / "memory"
    workspace_skills = workspace / "skills"
    drift_skills = workspace / "drift" / "skills"
    workspace.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    observe.mkdir(exist_ok=True)
    memory_workspace.mkdir(parents=True, exist_ok=True)
    workspace_skills.mkdir(parents=True, exist_ok=True)
    drift_skills.mkdir(parents=True, exist_ok=True)
    if not EXAMPLE_CONFIG.exists():
        EXAMPLE_CONFIG.write_text(EXAMPLE_CONFIG_CONTENT, encoding="utf-8")
    mcp_config = workspace / "mcp_servers.json"
    proactive_sources = workspace / "proactive_sources.json"
    drift_tasks = workspace / "drift_tasks.json"
    rss_feeds = workspace / "rss_feeds.opml"
    if not mcp_config.exists():
        mcp_config.write_text(MCP_SERVERS_EXAMPLE, encoding="utf-8")
    if not proactive_sources.exists():
        proactive_sources.write_text(PROACTIVE_SOURCES_EXAMPLE, encoding="utf-8")
    if not drift_tasks.exists():
        drift_tasks.write_text(DRIFT_TASKS_EXAMPLE, encoding="utf-8")
    if not rss_feeds.exists():
        rss_feeds.write_text(RSS_FEEDS_OPML_EXAMPLE, encoding="utf-8")
    workspace_skills_readme = workspace_skills / "README.md"
    drift_skills_readme = drift_skills / "README.md"
    if not workspace_skills_readme.exists():
        workspace_skills_readme.write_text(
            "# Workspace Skills\n\n在这里创建 `<skill-name>/SKILL.md`，同名 skill 会覆盖项目内置 skill。\n",
            encoding="utf-8",
        )
    if not drift_skills_readme.exists():
        drift_skills_readme.write_text(
            "# Drift Skills\n\n在这里创建 drift 专用 `<skill-name>/SKILL.md`，后台空闲任务会优先使用这里的说明书。\n",
            encoding="utf-8",
        )
    print(f"Initialized workspace: {workspace}")
    print(f"Initialized logs: {logs}")
    print(f"Initialized observe: {observe}")
    print(f"Initialized memory files: {memory_workspace}")
    print(f"Config example ready: {EXAMPLE_CONFIG}")
    print(f"MCP config ready: {mcp_config}")
    print(f"Proactive sources ready: {proactive_sources}")
    print(f"Drift tasks ready: {drift_tasks}")
    print(f"RSS OPML ready: {rss_feeds}")
    print(f"Workspace skills ready: {workspace_skills}")
    print(f"Drift skills ready: {drift_skills}")


async def run_bot(config: AppConfig) -> None:
    """根据应用配置装配并启动整套机器人服务。

    参数:
        config: 已完成校验与环境变量展开的全局配置对象。

    说明:
        该函数会依次创建存储层、模型 Provider、工具注册表、MCP 注册表、被动对话循环、
        Telegram 通道以及主动触达循环，并负责运行期的清理收尾。
    """
    try:
        from chat_agent.agent.provider import LLMProvider
        from chat_agent.context import ContextBuilder
        from chat_agent.loop import AgentLoop
        from chat_agent.mcp.registry import MCPRegistry
        from chat_agent.memory.consolidation import ConsolidationService
        from chat_agent.memory.embeddings import EmbeddingProvider
        from chat_agent.memory.files import MemoryFiles
        from chat_agent.memory.indexer import MemoryIndexer
        from chat_agent.memory.reranker import HttpReranker
        from chat_agent.memory.retriever import MemoryRetriever
        from chat_agent.memory.store import SQLiteStore
        from chat_agent.memory.vector_store import create_vector_store
        from chat_agent.observe.trace import TraceRecorder
        from chat_agent.presence import PresenceTracker
        from chat_agent.proactive.drift import DriftManager
        from chat_agent.proactive.feed import ProactiveFeedManager
        from chat_agent.proactive.loop import ProactiveLoop
        from chat_agent.reasoner import Reasoner
        from chat_agent.skills import SkillsLoader
        from chat_agent.tools.builtin import build_default_registry, register_message_push_tool
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise SystemExit(
            f"Missing Python dependency: {missing}\n"
            'Please install dependencies with: pip install -e ".[dev]"\n'
            "If you are using a virtual environment, activate it before running python main.py."
        ) from exc

    store = SQLiteStore(config.memory.database_path)
    observe_store = SQLiteStore(config.observe.database_path)
    memory_files = MemoryFiles(config.memory.database_path.parent / "memory")
    skills_loader = (
        SkillsLoader(
            workspace=config.skills.workspace_dir,
            builtin_skills_dir=config.skills.builtin_dir,
            max_catalog_chars=config.skills.max_catalog_chars,
        )
        if config.skills.enabled
        else None
    )
    main_provider = LLMProvider(config.llm.main)
    fast_provider = LLMProvider(config.llm.fast)
    embedding_provider = (
        EmbeddingProvider(
            model=config.embedding.model,
            api_key=config.embedding.api_key,
            base_url=config.embedding.base_url,
            timeout_seconds=config.embedding.timeout_seconds,
            dimension=config.embedding.dimension,
        )
        if config.embedding.enabled and config.embedding.model and config.embedding.api_key and config.embedding.base_url
        else None
    )
    vector_store = create_vector_store(config.embedding, store) if config.embedding.enabled else None
    memory_indexer = MemoryIndexer(embedding_provider, vector_store)
    reranker = HttpReranker(config.reranker) if config.reranker.enabled else None
    if config.llm.fast.model == config.llm.main.model and config.llm.fast.base_url == config.llm.main.base_url:
        logging.getLogger(__name__).warning("[llm.fast] uses the same model/base_url as main; this is allowed but less efficient")
    retriever = MemoryRetriever(
        store,
        enabled=config.memory.enabled,
        fast_provider=fast_provider,
        query_rewrite_enabled=config.memory.query_rewrite_enabled,
        hyde_enabled=config.memory.hyde_enabled,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        vector_top_k=config.memory.vector_top_k,
        vector_min_score=config.embedding.min_score,
        bm25_top_k=config.memory.bm25_top_k,
        rrf_top_k=config.memory.rrf_top_k,
        rrf_k=config.memory.rrf_k,
        reranker=reranker,
    )
    tools = build_default_registry(
        store,
        fetch_timeout=config.tools.web_fetch_timeout_seconds,
        tool_search_enabled=config.tools.tool_search_enabled,
        file_workspace=config.tools.file_workspace,
        skills_loader=skills_loader,
        extra_model_tools=config.tools.extra_model_tools,
        memory_indexer=memory_indexer,
        memory_retriever=retriever,
    )
    mcp_registry = (
        MCPRegistry(
            config.mcp.config_path,
            tools,
            observe_store,
            allowed_servers=config.mcp.allowed_servers,
            allowed_tools=config.mcp.allowed_tools,
        )
        if config.mcp.enabled
        else None
    )
    if mcp_registry:
        await mcp_registry.load()
    context_builder = ContextBuilder(
        store=store,
        retriever=retriever,
        tools=tools,
        history_window=config.memory.history_window,
        memory_top_k=config.memory.top_k,
        summary_enabled=config.memory.summary_enabled,
        vision_enabled=config.llm.main.enable_vision,
        skills_loader=skills_loader,
        inject_skills_catalog=config.skills.inject_catalog,
    )
    from chat_agent.memes import MemeService

    meme_service = MemeService(config.tools.file_workspace)
    reasoner = Reasoner(
        provider=main_provider,
        tools=tools,
        max_iterations=config.tools.max_iterations,
        tool_loop_enabled=config.tools.tool_calling_enabled,
    )
    trace = TraceRecorder(observe_store)
    presence = PresenceTracker(store, active_window_minutes=config.proactive.presence.active_window_minutes)
    consolidation_service = (
        ConsolidationService(
            store=store,
            memory_files=memory_files,
            provider=main_provider,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            memory_indexer=memory_indexer,
            keep_recent=config.memory.history_window,
            max_window=max(config.memory.history_window * 4, 40),
        )
        if config.memory.enabled
        else None
    )
    agent = AgentLoop(
        store=store,
        context_builder=context_builder,
        reasoner=reasoner,
        trace_recorder=trace,
        presence=presence,
        memory_enabled=config.memory.enabled,
        max_messages_per_chat=config.memory.max_messages_per_chat,
        scheduler_enabled=config.scheduler.enabled,
        summary_enabled=config.memory.summary_enabled,
        summary_after_messages=config.memory.summary_after_messages,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        memory_indexer=memory_indexer,
        memory_retriever=retriever,
        consolidation_service=consolidation_service,
        meme_service=meme_service,
        model_main=config.llm.main.model,
        model_fast=config.llm.fast.model,
    )

    if config.channel == "qq":
        from chat_agent.channels.qq import QQBotChannel

        channel = QQBotChannel(
            config.qq,
            agent.handle_message,
            store=store,
            mcp_registry=mcp_registry,
            skills_loader=skills_loader,
        )
    else:
        from chat_agent.channels.telegram import TelegramChannel

        channel = TelegramChannel(
            config.telegram,
            agent.handle_message,
            store=store,
            mcp_registry=mcp_registry,
            skills_loader=skills_loader,
        )
    register_message_push_tool(
        tools,
        channel,
        default_chat_id=config.proactive.loop.target_chat_id,
        file_workspace=config.tools.file_workspace,
    )
    feed_manager = ProactiveFeedManager(config.proactive.feed.sources_path, mcp_registry) if config.proactive.feed.enabled else None
    drift_skills_loader = (
        SkillsLoader(
            workspace=config.proactive.drift.skills_workspace_dir,
            builtin_skills_dir=config.skills.builtin_dir if config.proactive.drift.skills_include_builtin else None,
            max_catalog_chars=config.skills.max_catalog_chars,
        )
        if config.skills.enabled and config.proactive.drift.skills_enabled
        else None
    )
    drift_manager = (
        DriftManager(
            store=store,
            provider=fast_provider,
            tasks_path=config.proactive.drift.tasks_path,
            output_dir=config.proactive.drift.output_dir,
            run_cooldown_minutes=config.proactive.drift.run_cooldown_minutes,
            daily_run_cap=config.proactive.drift.daily_run_cap,
            promotion_enabled=config.proactive.drift.promotion_enabled,
            target_chat_id=config.proactive.loop.target_chat_id,
            skills_loader=drift_skills_loader,
            tools=tools,
            max_iterations=config.tools.max_iterations,
        )
        if config.proactive.drift.enabled
        else None
    )
    proactive = ProactiveLoop(
        store=store,
        channel=channel,
        enabled=config.proactive.loop.enabled and config.scheduler.enabled,
        tick_interval_seconds=config.proactive.loop.tick_interval_seconds,
        max_due_per_tick=config.scheduler.max_due_per_tick,
        target_chat_id=config.proactive.loop.target_chat_id,
        budget=config.proactive.budget,
        fallback_config=config.proactive.fallback,
        feed_config=config.proactive.feed,
        drift_config=config.proactive.drift,
        presence=presence,
        skip_when_busy=config.proactive.presence.skip_when_busy,
        fallback_provider=fast_provider,
        feed_manager=feed_manager,
        drift_manager=drift_manager,
        observe_store=observe_store,
        meme_service=meme_service,
        embedding_provider=embedding_provider,
    )

    proactive_task: asyncio.Task | None = None
    try:
        await channel.start()
        proactive_task = asyncio.create_task(proactive.run(), name="proactive-loop")
        await channel.idle()
    except asyncio.CancelledError:
        logging.getLogger(__name__).info("Shutdown requested")
        raise
    finally:
        if proactive_task:
            proactive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await proactive_task
        await proactive.stop()
        if mcp_registry:
            await mcp_registry.shutdown()
        await channel.stop()


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    返回:
        返回 `argparse.Namespace`，包含待执行命令以及配置文件路径。
    """
    parser = argparse.ArgumentParser(description="Personal agent MVP")
    parser.add_argument("command", nargs="?", choices=["init"], help="Initialize workspace and example config")
    parser.add_argument("--config", default="config.toml", help="Path to TOML config file")
    return parser.parse_args()


def main() -> None:
    """程序主入口。

    行为:
        - 当命令为 `init` 时，初始化项目目录。
        - 否则读取配置并启动 personal agent。
    """
    args = parse_args()
    if args.command == "init":
        setup_logging()
        init_project()
        return

    try:
        config = load_config(Path(args.config))
    except ConfigError as exc:
        raise SystemExit(f"Config error: {exc}") from exc

    setup_logging(config.logging)
    logging.getLogger(__name__).info("Starting personal agent channel=%s", config.channel)
    try:
        asyncio.run(run_bot(config))
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Stopped by user")


if __name__ == "__main__":
    main()
