"""TOML 配置加载与校验。

本模块负责读取 config.toml、递归展开 `${ENV_NAME}` 环境变量、应用默认值，并把结果
整理成强类型 dataclass。运行时其他模块只接收 AppConfig 或子配置对象，不直接读取 TOML。
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    """配置错误。

    典型场景：
    - config.toml 不存在。
    - 必填项为空，例如 telegram.token 或 llm.main.api_key。
    - `${ENV_NAME}` 引用了未设置的环境变量。
    """


@dataclass(frozen=True)
class LLMConfig:
    """单个 LLM profile 的配置。

    Attributes:
        model: 模型名，例如 qwen-plus、qwen-vl-plus。
        api_key: OpenAI-compatible API key。由环境变量展开后填入，禁止写死到代码。
        base_url: OpenAI-compatible 服务地址。
        timeout_seconds: 请求超时时间，单位秒。
        max_tokens: 最大输出 token，None 表示不显式传入。
        enable_vision: 是否允许这个 profile 接收 image_url 多模态内容。
        profile: 日志中的 profile 名，通常是 main 或 fast。
    """

    model: str
    api_key: str
    base_url: str
    timeout_seconds: float
    max_tokens: int | None
    enable_vision: bool = False
    profile: str = "main"


@dataclass(frozen=True)
class LLMProfilesConfig:
    """主/次 LLM 配置集合。

    main 用于普通对话、多模态、工具循环最终推理；fast 用于 query rewrite、
    HyDE、主动系统轻量判断等。
    """

    main: LLMConfig
    fast: LLMConfig


@dataclass(frozen=True)
class TelegramConfig:
    """Telegram channel 配置。

    Attributes:
        token: Telegram Bot Token，必须来自环境变量或本地 config。
        allow_from: username 白名单。为空时允许所有用户。
        unauthorized_reply: 未授权用户发消息时是否回复无权限提示。
        download_images: 是否把 Telegram 图片下载到 workspace/attachments。
        image_max_mb: 单张图片最大大小，单位 MB。
    """

    token: str
    allow_from: list[str]
    unauthorized_reply: bool
    download_images: bool = True
    image_max_mb: int = 10


@dataclass(frozen=True)
class MemoryConfig:
    """会话历史、长期记忆和检索增强配置。"""

    enabled: bool
    database_path: Path
    history_window: int
    top_k: int
    summary_enabled: bool
    summary_after_messages: int
    max_messages_per_chat: int
    query_rewrite_enabled: bool
    hyde_enabled: bool


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding 与向量检索配置。

    第一阶段 provider=sqlite_json，会把向量 JSON 存在 SQLite 中并用 Python 余弦相似度检索；
    第二阶段可以切换到 qdrant/chroma/pgvector 等外部向量数据库。
    """

    enabled: bool
    provider: str
    model: str
    api_key: str
    base_url: str
    timeout_seconds: float
    dimension: int | None
    top_k: int
    min_score: float
    external_url: str
    external_api_key: str
    collection: str


@dataclass(frozen=True)
class ToolsConfig:
    """工具循环和内置工具配置。"""

    web_fetch_timeout_seconds: int
    tool_calling_enabled: bool
    max_iterations: int
    tool_search_enabled: bool
    file_workspace: Path
    extra_model_tools: list[str]


@dataclass(frozen=True)
class ReasonerConfig:
    """Reasoner 兼容旧配置段。

    当前实际优先读取 [tools]，保留该结构是为了兼容早期 config.toml。
    """

    max_iterations: int
    tool_loop_enabled: bool


@dataclass(frozen=True)
class MCPConfig:
    """MCP registry 配置。"""

    enabled: bool
    config_path: Path
    allowed_servers: list[str]
    allowed_tools: list[str]


@dataclass(frozen=True)
class ProactiveLoopConfig:
    """主动循环基础配置。"""

    enabled: bool
    tick_interval_seconds: int
    target_chat_id: str


@dataclass(frozen=True)
class ProactiveBudgetConfig:
    """主动触达预算配置。"""

    daily_max: int
    min_interval_minutes: int
    quiet_hours_start: str
    quiet_hours_end: str


@dataclass(frozen=True)
class PresenceConfig:
    """用户活跃/忙碌状态配置，用于避免主动消息打扰正在对话的用户。"""

    active_window_minutes: int
    skip_when_busy: bool


@dataclass(frozen=True)
class FeedConfig:
    """Proactive feed source 配置。"""

    enabled: bool
    sources_path: Path
    daily_cap: int


@dataclass(frozen=True)
class DriftConfig:
    """Drift 空闲任务配置。

    Drift 是后台 preparation layer：运行结果默认只归档，是否提升为主动候选由 promotion
    策略和主循环统一决定。
    """

    enabled: bool
    tasks_path: Path
    output_dir: Path
    run_cooldown_minutes: int
    daily_run_cap: int
    promotion_enabled: bool
    daily_cap: int
    skills_enabled: bool
    skills_workspace_dir: Path
    skills_include_builtin: bool


@dataclass(frozen=True)
class FallbackConfig:
    """陪伴型 fallback 候选配置。"""

    enabled: bool
    probability: float
    daily_cap: int


@dataclass(frozen=True)
class ProactiveConfig:
    """主动系统总配置。"""

    loop: ProactiveLoopConfig
    budget: ProactiveBudgetConfig
    presence: PresenceConfig
    feed: FeedConfig
    drift: DriftConfig
    fallback: FallbackConfig


@dataclass(frozen=True)
class SchedulerConfig:
    """提醒调度器配置。"""

    enabled: bool
    max_due_per_tick: int


@dataclass(frozen=True)
class LoggingConfig:
    """日志配置。file 为 None 时只输出到控制台。"""

    level: str
    file: Path | None


@dataclass(frozen=True)
class ObserveConfig:
    """可观测性数据库配置。

    Attributes:
        database_path: trace、MCP 工具日志、proactive tick 等观测记录的 SQLite 路径。
            业务状态仍保存在 [memory].database_path；observe 数据库可以单独备份或清理。
    """

    database_path: Path


@dataclass(frozen=True)
class SkillsConfig:
    """SKILL.md 技能说明书系统配置。"""

    enabled: bool
    builtin_dir: Path
    workspace_dir: Path
    inject_catalog: bool
    max_catalog_chars: int


@dataclass(frozen=True)
class AppConfig:
    """应用完整配置对象。

    load_config() 会把 TOML、环境变量和默认值折叠成该对象。运行时其他模块只接收
    AppConfig 或其中的子配置，避免到处读取 TOML。
    """

    llm: LLMProfilesConfig
    telegram: TelegramConfig
    memory: MemoryConfig
    embedding: EmbeddingConfig
    reasoner: ReasonerConfig
    proactive: ProactiveConfig
    scheduler: SchedulerConfig
    logging: LoggingConfig
    observe: ObserveConfig
    tools: ToolsConfig
    mcp: MCPConfig
    skills: SkillsConfig


def _expand_env(value: Any, path: str = "") -> Any:
    """递归展开配置中的 `${ENV_NAME}`。

    Args:
        value: 任意 TOML 解析值，可以是 str/list/dict/其他基础类型。
        path: 当前值在配置树中的路径，仅用于错误提示。

    Returns:
        展开环境变量后的同结构值。

    Raises:
        ConfigError: 引用的环境变量不存在，或展开后仍残留 `${...}`。
    """
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            """把单个 `${ENV_NAME}` 占位符替换成环境变量值。"""
            name = match.group(1)
            if name not in os.environ:
                raise ConfigError(f"Environment variable {name} referenced by {path or 'config'} is not set")
            return os.environ[name]

        expanded = ENV_PATTERN.sub(replace, value)
        if "${" in expanded:
            raise ConfigError(f"Unexpanded environment placeholder in {path or 'config'}")
        return expanded
    if isinstance(value, list):
        return [_expand_env(item, f"{path}[]") for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item, f"{path}.{key}" if path else str(key)) for key, item in value.items()}
    return value


def _require(value: Any, name: str) -> str:
    """读取必填字符串配置，空值时给出清晰错误。"""
    text = str(value or "").strip()
    if not text:
        raise ConfigError(f"Missing required config value: {name}")
    return text


def _relative_to_config(config_path: Path, value: str | Path) -> Path:
    """把相对路径解析到 config.toml 所在目录。"""
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _load_llm_profile(raw: dict[str, Any], name: str) -> LLMConfig:
    """从 TOML 子字典加载一个 LLM profile。"""
    return LLMConfig(
        model=_require(raw.get("model"), f"llm.{name}.model"),
        api_key=_require(raw.get("api_key"), f"llm.{name}.api_key"),
        base_url=_require(raw.get("base_url"), f"llm.{name}.base_url"),
        timeout_seconds=float(raw.get("timeout_seconds", 60)),
        max_tokens=int(raw["max_tokens"]) if raw.get("max_tokens") is not None else None,
        enable_vision=bool(raw.get("enable_vision", False)),
        profile=name,
    )


def load_config(path: str | Path) -> AppConfig:
    """加载 config.toml 并返回 AppConfig。

    Args:
        path: TOML 配置文件路径。

    Returns:
        已完成环境变量展开、默认值填充和相对路径解析的 AppConfig。

    Raises:
        ConfigError: 配置文件不存在、必填项缺失或环境变量未设置。
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("rb") as f:
        raw = _expand_env(tomllib.load(f))

    llm_raw = raw.get("llm", {})
    if "main" in llm_raw:
        main_llm = _load_llm_profile(llm_raw.get("main", {}), "main")
        fast_llm = _load_llm_profile(llm_raw.get("fast", llm_raw.get("main", {})), "fast")
    else:
        main_llm = _load_llm_profile(llm_raw, "main")
        fast_llm = LLMConfig(
            model=main_llm.model,
            api_key=main_llm.api_key,
            base_url=main_llm.base_url,
            timeout_seconds=main_llm.timeout_seconds,
            max_tokens=main_llm.max_tokens,
            enable_vision=False,
            profile="fast",
        )

    telegram = raw.get("telegram", {})
    memory = raw.get("memory", {})
    embedding = raw.get("embedding", {})
    tools = raw.get("tools", {})
    reasoner = raw.get("reasoner", {})
    proactive = raw.get("proactive", {})
    if not isinstance(proactive, dict):
        raise ConfigError("proactive must be a table")
    deprecated_proactive_keys = {"enabled", "tick_interval_seconds", "daily_max", "target", "context_fallback"}
    used_deprecated = sorted(key for key in deprecated_proactive_keys if key in proactive)
    if used_deprecated:
        raise ConfigError(
            "Deprecated proactive config keys are not supported: "
            + ", ".join(f"proactive.{key}" for key in used_deprecated)
        )
    loop = proactive.get("loop", {})
    budget = proactive.get("budget", {})
    fallback = proactive.get("fallback", {})
    presence = proactive.get("presence", {})
    feed = proactive.get("feed", {})
    drift = proactive.get("drift", raw.get("drift", {}))
    drift_skills = drift.get("skills", {}) if isinstance(drift, dict) else {}
    deprecated_feed_keys = {"poll_interval_seconds"}
    used_feed_deprecated = sorted(key for key in deprecated_feed_keys if key in feed)
    if used_feed_deprecated:
        raise ConfigError(
            "Deprecated proactive feed config keys are not supported: "
            + ", ".join(f"proactive.feed.{key}" for key in used_feed_deprecated)
        )
    deprecated_drift_keys = {"cooldown_minutes", "daily_max", "notify"}
    used_drift_deprecated = sorted(key for key in deprecated_drift_keys if isinstance(drift, dict) and key in drift)
    if used_drift_deprecated:
        raise ConfigError(
            "Deprecated proactive drift config keys are not supported: "
            + ", ".join(f"proactive.drift.{key}" for key in used_drift_deprecated)
        )
    scheduler = raw.get("scheduler", {})
    logging_config = raw.get("logging", {})
    observe = raw.get("observe", {})
    mcp = raw.get("mcp", {})
    skills = raw.get("skills", {})

    log_file = logging_config.get("file", logging_config.get("file_path", "logs/app.log"))
    tool_max_iterations = int(tools.get("max_iterations", reasoner.get("max_iterations", 5)))

    return AppConfig(
        llm=LLMProfilesConfig(main=main_llm, fast=fast_llm),
        telegram=TelegramConfig(
            token=_require(telegram.get("token"), "telegram.token"),
            allow_from=[str(item).lstrip("@") for item in telegram.get("allow_from", [])],
            unauthorized_reply=bool(telegram.get("unauthorized_reply", True)),
            download_images=bool(telegram.get("download_images", True)),
            image_max_mb=int(telegram.get("image_max_mb", 10)),
        ),
        memory=MemoryConfig(
            enabled=bool(memory.get("enabled", True)),
            database_path=_relative_to_config(config_path, memory.get("database_path", "workspace/agent.sqlite3")),
            history_window=int(memory.get("history_window", memory.get("history_limit", 20))),
            top_k=int(memory.get("top_k", memory.get("memory_limit", 5))),
            summary_enabled=bool(memory.get("summary_enabled", True)),
            summary_after_messages=int(memory.get("summary_after_messages", 40)),
            max_messages_per_chat=int(memory.get("max_messages_per_chat", 500)),
            query_rewrite_enabled=bool(memory.get("query_rewrite_enabled", False)),
            hyde_enabled=bool(memory.get("hyde_enabled", False)),
        ),
        embedding=EmbeddingConfig(
            enabled=bool(embedding.get("enabled", False)),
            provider=str(embedding.get("provider", "sqlite_json")),
            model=str(embedding.get("model", "")),
            api_key=str(embedding.get("api_key", "")),
            base_url=str(embedding.get("base_url", "")),
            timeout_seconds=float(embedding.get("timeout_seconds", 30)),
            dimension=int(embedding["dimension"]) if embedding.get("dimension") is not None else None,
            top_k=int(embedding.get("top_k", memory.get("top_k", 5))),
            min_score=float(embedding.get("min_score", 0.2)),
            external_url=str(embedding.get("external_url", "")),
            external_api_key=str(embedding.get("external_api_key", "")),
            collection=str(embedding.get("collection", "chat_agent_memories")),
        ),
        tools=ToolsConfig(
            web_fetch_timeout_seconds=int(tools.get("web_fetch_timeout_seconds", 10)),
            tool_calling_enabled=bool(tools.get("tool_calling_enabled", reasoner.get("tool_loop_enabled", True))),
            max_iterations=tool_max_iterations,
            tool_search_enabled=bool(tools.get("tool_search_enabled", True)),
            file_workspace=_relative_to_config(config_path, tools.get("file_workspace", "workspace/files")),
            extra_model_tools=[str(item).strip() for item in tools.get("extra_model_tools", []) if str(item).strip()],
        ),
        reasoner=ReasonerConfig(
            max_iterations=tool_max_iterations,
            tool_loop_enabled=bool(tools.get("tool_calling_enabled", reasoner.get("tool_loop_enabled", True))),
        ),
        proactive=ProactiveConfig(
            loop=ProactiveLoopConfig(
                enabled=bool(loop.get("enabled", True)),
                tick_interval_seconds=int(loop.get("tick_interval_seconds", 60)),
                target_chat_id=str(loop.get("target_chat_id", "")),
            ),
            budget=ProactiveBudgetConfig(
                daily_max=int(budget.get("daily_max", 6)),
                min_interval_minutes=int(budget.get("min_interval_minutes", 90)),
                quiet_hours_start=str(budget.get("quiet_hours_start", "")),
                quiet_hours_end=str(budget.get("quiet_hours_end", "")),
            ),
            presence=PresenceConfig(
                active_window_minutes=int(presence.get("active_window_minutes", 10)),
                skip_when_busy=bool(presence.get("skip_when_busy", True)),
            ),
            feed=FeedConfig(
                enabled=bool(feed.get("enabled", False)),
                sources_path=_relative_to_config(config_path, feed.get("sources_path", "workspace/proactive_sources.json")),
                daily_cap=int(feed.get("daily_cap", 3)),
            ),
            drift=DriftConfig(
                enabled=bool(drift.get("enabled", False)),
                tasks_path=_relative_to_config(config_path, drift.get("tasks_path", "workspace/drift_tasks.json")),
                output_dir=_relative_to_config(config_path, drift.get("output_dir", "workspace/drift_runs")),
                run_cooldown_minutes=int(drift.get("run_cooldown_minutes", 180)),
                daily_run_cap=int(drift.get("daily_run_cap", 3)),
                promotion_enabled=bool(drift.get("promotion_enabled", True)),
                daily_cap=int(drift.get("daily_cap", 2)),
                skills_enabled=bool(drift_skills.get("enabled", False)),
                skills_workspace_dir=_relative_to_config(config_path, drift_skills.get("workspace_dir", "workspace/drift/skills")),
                skills_include_builtin=bool(drift_skills.get("include_builtin", True)),
            ),
            fallback=FallbackConfig(
                enabled=bool(fallback.get("enabled", False)),
                probability=float(fallback.get("probability", 0.03)),
                daily_cap=int(fallback.get("daily_cap", 2)),
            ),
        ),
        scheduler=SchedulerConfig(
            enabled=bool(scheduler.get("enabled", True)),
            max_due_per_tick=int(scheduler.get("max_due_per_tick", 50)),
        ),
        logging=LoggingConfig(
            level=str(logging_config.get("level", "INFO")).upper(),
            file=_relative_to_config(config_path, log_file) if log_file else None,
        ),
        observe=ObserveConfig(
            database_path=_relative_to_config(config_path, observe.get("database_path", "observe/observe.db")),
        ),
        mcp=MCPConfig(
            enabled=bool(mcp.get("enabled", False)),
            config_path=_relative_to_config(config_path, mcp.get("config_path", "workspace/mcp_servers.json")),
            allowed_servers=[str(item).strip() for item in mcp.get("allowed_servers", []) if str(item).strip()],
            allowed_tools=[str(item).strip() for item in mcp.get("allowed_tools", []) if str(item).strip()],
        ),
        skills=SkillsConfig(
            enabled=bool(skills.get("enabled", True)),
            builtin_dir=_relative_to_config(config_path, skills.get("builtin_dir", "skills")),
            workspace_dir=_relative_to_config(config_path, skills.get("workspace_dir", "workspace/skills")),
            inject_catalog=bool(skills.get("inject_catalog", True)),
            max_catalog_chars=int(skills.get("max_catalog_chars", 4000)),
        ),
    )
