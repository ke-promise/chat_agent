---
name: create-skill
description: 创建或改写当前 chat_agent 项目的 SKILL.md。用户要求新增技能、适配旧技能、修改技能说明书时使用。
metadata: {"chat_agent":{"always":false,"drift":false,"triggers":["创建 skill","新增技能","修改技能说明书","SKILL.md"],"requires":{"bins":[],"env":[],"tools":[]}}}
---

# 创建 Skill

本项目的 skill 是给 Agent 阅读的操作说明书，不是 Python 插件，也不是 MCP server。

## 目录

```text
workspace/skills/<skill-name>/SKILL.md   # 用户自定义技能，优先级最高
skills/<skill-name>/SKILL.md             # 项目内置技能，随代码分发
```

同名时 `workspace/skills/` 会覆盖 `skills/`。优先把用户新增或个性化的技能写到 `workspace/skills/`。

## SKILL.md 格式

```markdown
---
name: skill-name
description: 一句话说明功能和触发场景，尽量包含用户可能说出的关键词。
metadata: {"chat_agent":{"always":false,"drift":false,"triggers":[],"requires":{"bins":[],"env":[],"tools":[]}}}
---

# Skill 标题

写给 Agent 的操作步骤、约束、示例工具调用和注意事项。
```

字段说明：
- `name`：小写字母、数字和连字符，必须和目录名一致。
- `description`：用于 catalog 和触发判断，要具体。
- `metadata.chat_agent.always`：为 true 时每轮完整注入 prompt，慎用。
- `metadata.chat_agent.drift`：为 true 时可被空闲 drift 任务使用。
- `metadata.chat_agent.triggers`：用户文本命中这些短语时会完整注入 skill。
- `requires.bins`：依赖的本地命令，例如 `curl`、`npx.cmd`。
- `requires.env`：依赖的环境变量，例如 `QWEN_API_KEY`。
- `requires.tools`：依赖已注册工具，例如 `web_fetch`、`read_file`；不改变工具默认可见性。

## 创建流程

1. 确认 skill 名称，使用小写连字符，例如 `daily-news-briefing`。
2. 优先写入 `workspace/skills/<name>/SKILL.md`。
3. 正文只写当前项目真的具备的能力：Telegram、ToolRegistry、MCP、memory、proactive、drift。
4. 如果需要工具，先写明可用工具名，例如 `tool_search`、`read_skill`、`web_fetch`、`read_file`、`write_file`。
5. 不要引用不存在的项目路径或不存在的函数名；当前项目确实支持 Telegram 与 QQ 官方 Bot。

## 写作原则

- 说明书要短，避免每轮 prompt 过长。
- 不要把 API key、token、私密路径写进 skill。
- 如果某个动作依赖 MCP，写成“先用 `tool_search` 搜索并解锁工具”，不要硬编码不确定的工具名。
- 对 workspace 文件操作，使用项目工具的相对路径，不要要求访问项目外路径。
