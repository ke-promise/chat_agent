# Builtin Skills

这里是当前 `chat_agent` 项目的内置 SKILL.md 说明书。

Skill 的作用是给 Agent 提供“做事前看的 SOP”，不是 Python 插件，也不是 MCP server。

```text
tools  = Agent 可以调用的动作，例如 web_fetch、memorize、read_file、MCP 工具
skills = Agent 做事前阅读的说明书，例如天气查询流程、feed 管理流程
```

## 目录优先级

```text
workspace/skills/<name>/SKILL.md       # 用户自定义技能，优先级最高
skills/<name>/SKILL.md                 # 项目内置技能
workspace/drift/skills/<name>/SKILL.md # 空闲 drift 专用技能
```

同名 workspace skill 会覆盖内置 skill。

## 本项目边界

- 支持 Telegram 与 QQ 官方 Bot 渠道。
- 不包含 NapCat、Discord 或复杂多通道抽象。
- 可通过 MCP 接入外部工具，但 skill 本身不启动 MCP server。
- 文件工具默认只能操作 `[tools].file_workspace`，通常是 `workspace/files`。

## Metadata

推荐格式：

```yaml
metadata: {"chat_agent":{"always":false,"drift":false,"triggers":[],"requires":{"bins":[],"env":[],"tools":[]}}}
```

- `always=true`：每轮注入完整 skill，慎用。
- `drift=true`：可被空闲 DriftManager 作为后台任务执行。
- `triggers`：显式触发词，命中用户文本时注入完整 skill。
- `requires.bins`：依赖本地命令。
- `requires.env`：依赖环境变量。
- `requires.tools`：依赖已注册工具；只检查工具是否存在，不改变工具可见性。
