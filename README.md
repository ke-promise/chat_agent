# Telegram 个人智能体

一个专注 Telegram 单通道的个人智能体项目，围绕 OpenAI-compatible LLM、长期记忆、提醒、MCP 工具、主动消息和技能说明书构建。它不做多平台抽象，重点是把“陪伴型个人助手”这条链路在本地跑通，并且尽量保持结构清晰、可扩展。

## 项目特点

- Telegram-only，入口简单，部署和排障成本低。
- 支持文本对话和图片输入，主模型可直接处理多模态消息。
- 同时维护会话历史、长期记忆、用户画像、摘要和文本档案。
- 内置工具循环，支持 memory、reminder、网页抓取、文件读写、表情包等能力。
- 支持 MCP server 接入，可把 DuckDuckGo 搜索、网页正文抓取、RSS/feed 能力接进模型工具链。
- 支持 `SKILL.md` 说明书机制，让模型按 SOP 工作，而不是只靠 prompt 硬写规则。
- 支持主动触达：到期提醒、feed 候选、drift 后台整理、轻量 fallback 问候。
- 内置表情包目录和自动挂图逻辑，也支持把收到的图片直接收录进本地表情包库。

## 当前能力

- 被动对话
  处理 Telegram 文本、caption 和图片消息，支持白名单用户名控制。
- 长期记忆
  支持显式“记住：...”、自然语言偏好抽取、候选记忆晋升、记忆纠正和回忆查询。
- 提醒系统
  支持自然语言提醒，例如“1 分钟后提醒我喝水”。
- 工具循环
  支持 OpenAI-compatible `tool_calls`，也兼容文本协议 `<tool_call ...>...</tool_call>`。
- Skills
  支持内置 skills、`workspace/skills/` 覆盖、依赖检查、catalog 注入和按需全文注入。
- MCP
  当前仓库已内置/预留 `duckduckgo`、`web_content`、`feed_bridge`、`rss` 这几类 server 配置。
- 主动消息
  后台循环会优先发送到期提醒，再评估 feed、drift 和 fallback 候选，并遵守预算、静默时段、去重和忙碌状态。
- 表情包系统
  支持 `list_memes`、`send_meme`，也支持发送图片后用文字把它收录到本地表情包分类。

## 目录结构

```text
.
├─ chat_agent/                 核心 Python 代码
│  ├─ channels/                Telegram 通道
│  ├─ memory/                  记忆、摘要、向量检索、文本档案
│  ├─ mcp/                     MCP client/registry
│  ├─ proactive/               主动消息、feed、drift
│  ├─ tools/                   内置工具注册
│  ├─ context.py               上下文拼装
│  ├─ loop.py                  被动消息主循环
│  ├─ reasoner.py              模型推理与工具协调
│  ├─ skills.py                SKILL.md 加载与注入
│  └─ memes.py                 表情包目录与自动挂图
├─ skills/                     项目内置技能说明书
├─ workspace/                  运行时工作目录
│  ├─ mcp_servers/             本地 MCP 示例 server
│  ├─ skills/                  用户自定义 skills
│  ├─ drift/skills/            drift 专用 skills
│  └─ memory/                  文本档案输出
├─ package/                    DuckDuckGo MCP Node 包
├─ tests/                      单元测试
├─ main.py                     CLI 入口
├─ config.example.toml         配置模板
└─ README.md
```

## 快速开始

### 1. 安装依赖

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
cd package
npm install
cd ..
```

### 2. 初始化项目目录

```powershell
python main.py init
Copy-Item config.example.toml config.toml
```

`python main.py init` 会在当前仓库下准备这些运行目录和示例文件：

- `workspace/`
- `logs/`
- `observe/`
- `workspace/mcp_servers.json`
- `workspace/proactive_sources.json`
- `workspace/drift_tasks.json`
- `workspace/rss_feeds.opml`
- `workspace/skills/`
- `workspace/drift/skills/`
- `workspace/memory/`

### 3. 配置环境变量

默认模板使用 DashScope 的 OpenAI-compatible 接口，你也可以换成任意兼容的模型服务。

```powershell
$env:QWEN_API_KEY="your-api-key"
$env:TELEGRAM_BOT_TOKEN="your-telegram-bot-token"
```

### 4. 修改 `config.toml`

最少要确认这些配置：

- `[llm.main]`：主对话模型，负责正常回复、图片理解、最终工具推理。
- `[llm.fast]`：轻量模型，负责 query rewrite、HyDE、主动消息改写和 tie-break。
- `[telegram]`：填好 bot token、允许访问的用户名白名单 `allow_from`。
- `[memory]`：主业务 SQLite 默认是 `workspace/agent.sqlite3`。
- `[observe]`：观测库默认是 `observe/observe.db`。

### 5. 启动

```powershell
python main.py
```

如果你使用了其他配置文件：

```powershell
python main.py --config path\to\config.toml
```

## 配置概览

### LLM

`[llm.main]` 和 `[llm.fast]` 都是 OpenAI-compatible 配置。

- `main` 适合主回复和视觉输入。
- `fast` 适合低成本判断、检索增强和主动系统润色。
- 当两者指向同一个模型也可以运行，只是性价比更低。

### Telegram

`[telegram]` 里最重要的是：

- `token`
- `allow_from`
- `download_images`
- `image_max_mb`

当 `download_images = true` 时，收到的图片会下载到 `workspace/attachments/`，并作为附件注入上下文。

### 记忆与向量检索

`[memory]` 控制：

- 会话历史窗口
- 长期记忆召回数量
- 摘要阈值
- query rewrite
- HyDE

`[embedding]` 当前支持两种模式：

- `provider = "sqlite_json"`：不依赖外部服务，向量存在 SQLite。
- `provider = "chroma"`：把向量检索接到外部 Chroma，记忆正文仍然以 SQLite 为准。

如果你开启 `chroma`，需要先单独启动 Chroma 服务，例如：

```powershell
chroma run --host localhost --port 8000
```

### 工具

`[tools]` 控制工具循环和文件工作区。默认文件工作区是 `workspace/files`。

当前内置工具大致分为三层：

- 默认可见：`memorize`、`recall_memory`、`create_reminder`、`list_reminders`、`tool_search`、`list_memes`
- 可发现：`cancel_reminder`、`web_fetch`、`list_files`、`read_file`、`send_meme`
- 默认隐藏：`write_file`、`send_message`、`send_emoji`、`create_skill`、`update_skill`

隐藏工具可以通过 `[tools].extra_model_tools` 主动暴露给模型。

### MCP

`[mcp]` 默认从 `workspace/mcp_servers.json` 加载 server 定义。当前模板里包含：

- `duckduckgo`
- `rss`
- `web_content`
- `feed_bridge`

其中：

- `duckduckgo` 依赖 `package/` 里的 Node 包，首次使用前需要执行过 `npm install`
- `web_content` 是本仓库自带的网页正文预览 MCP server
- `feed_bridge` 是本仓库自带的 RSS/Atom 到 proactive 事件桥接 server
- `rss` 示例依赖第三方 `mcp_rss` 和 MySQL，默认关闭

### Skills

`[skills]` 控制 `SKILL.md` 说明书系统。技能来源有两类：

- 项目内置：`skills/<name>/SKILL.md`
- 工作区覆盖：`workspace/skills/<name>/SKILL.md`

同名时，`workspace` 版本优先。

### 主动系统

主动系统相关配置位于：

- `[proactive.loop]`
- `[proactive.budget]`
- `[proactive.feed]`
- `[proactive.drift]`
- `[proactive.drift.skills]`
- `[proactive.fallback]`
- `[proactive.presence]`
- `[scheduler]`

如果你希望 bot 主动发消息，至少需要：

- 开启 `[proactive.loop].enabled = true`
- 设置 `[proactive.loop].target_chat_id`
- 保持 `[scheduler].enabled = true`

## 运行机制

### 被动对话链路

```text
Telegram Update
  -> channels/telegram.py
  -> InboundMessage
  -> AgentLoop
  -> ContextBuilder
  -> Reasoner
  -> ToolRegistry / MCP
  -> SQLiteStore + TraceRecorder
  -> OutboundMessage
  -> Telegram send
```

### 主动消息链路

```text
ProactiveLoop tick
  -> due reminders
  -> feed candidates
  -> drift result
  -> fallback candidate
  -> budget / quiet-hours / dedupe / busy filters
  -> choose winner
  -> Telegram send
```

## Skills 说明书系统

这个项目里的 `skills` 不是“可执行函数”，而是给模型读的说明书。

- `tools` 是动作，例如存记忆、读文件、抓网页。
- `skills` 是 SOP，例如“总结时保留什么”“天气问题该怎么查”。

完整 `SKILL.md` 不会每轮全量注入。系统会先注入一个 catalog 摘要，只有在这些场景下才会加载全文：

- skill 被标记为 `metadata.chat_agent.always = true`
- 用户显式提到 `@skill-name`
- 用户显式提到 `skill:skill-name`
- 用户文本里直接命中 skill 名称

## 记忆系统

当前项目的记忆不是单层结构，而是几条链路一起工作：

- 会话历史：每轮 `user` 和 `assistant` 消息都会进入 `session_messages`
- 长期记忆：显式记忆、推断记忆、候选记忆、纠正后的记忆都会落库
- 用户画像：会提取姓名、喜好、禁忌、回复风格等轻量信息
- 会话摘要：达到阈值后自动刷新
- 文本档案：在 `workspace/memory/<chat_id>/` 下维护多份 markdown 档案

默认还能处理这些自然语言场景：

- `记住：我喜欢简洁回答`
- `你记得我喜欢什么？`
- `不是这个，我喜欢的是...`
- `对，就是这个`

## 表情包系统

表情包目录默认位于 `workspace/files/memes/`，支持两种用法：

- 显式发送：模型可通过 `list_memes` 和 `send_meme` 选择并发送
- 自动挂图：系统会根据消息内容、情绪信号、冷却时间和来源策略决定是否自动附图

你也可以把收到的图片直接收录为表情包。例如发一张图，并附带文字：

```text
存成表情包：开心
```

或：

```text
加入表情包 可爱
```

收录后的素材会进入本地目录，并维护 `manifest.json` 元数据。

## MCP 与主动 Feed

如果你启用 `feed_bridge`，项目可以把外部 RSS/Atom 内容转成主动候选。默认示例配置文件是：

- `workspace/mcp_servers.json`
- `workspace/proactive_sources.json`
- `workspace/rss_feeds.opml`

其中 `feed_bridge` 会暴露这些工具：

- `poll_feeds`
- `get_proactive_events`
- `ack_events`

`web_content` 会暴露：

- `fetch_page`

## 常用 Telegram 命令

- `/start`
- `/help`
- `/status`
- `/memory`
- `/forget <id>`
- `/mcp`
- `/mcp_reload`
- `/skills`
- `/proactive_status`

## 示例

文本对话：

```text
你好
记住：我喜欢简洁回答
你记得我喜欢什么？
1分钟后提醒我喝水
```

图片理解：

```text
帮我看看这张图里有什么
```

表情包收录：

```text
存成表情包：抱抱
```

## 测试

```powershell
pytest
python -m compileall chat_agent main.py tests
```

当前测试不会去调用真实的 LLM、Telegram Bot API 或外部 MCP server。

## 运行产物

默认运行后你会主要看到这些本地数据：

- `workspace/agent.sqlite3`：业务主库，保存消息、记忆、提醒、主动候选等
- `observe/observe.db`：观测库，保存 trace、MCP 调用日志、proactive tick 等
- `logs/app.log`：应用日志
- `workspace/attachments/`：下载的 Telegram 图片
- `workspace/files/`：工具可访问的文件工作区
- `workspace/memory/`：长期运行时整理出的 markdown 档案

## 常见问题

- 启动时报 `Config error`
  通常是 `config.toml` 缺字段，或 `${ENV_NAME}` 对应的环境变量未设置。
- 图片无法理解
  检查 `[llm.main].enable_vision = true`，并确认主模型本身支持视觉输入。
- DuckDuckGo MCP 用不了
  先进入 `package/` 执行一次 `npm install`。
- 主动消息不发送
  重点检查 `proactive.loop.target_chat_id`、预算配置、静默时段和 `presence.skip_when_busy`。
- Chroma 检索无效
  确认已启动 Chroma 服务；如果没有外部服务，切回 `provider = "sqlite_json"`。

## 适合继续扩展的方向

- 增加更多内置 skills 和 drift 专用 skills
- 接更多 MCP server
- 为主动消息增加更强的排序策略
- 补充更细的记忆治理和清理工具
- 为表情包系统增加管理命令和审核流程
