# Builtin Skills

这里是当前 Telegram-only `chat_agent` 项目的内置 SKILL.md 说明书。

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

- 只支持 Telegram。
- 不包含 QQ、NapCat、Discord 或复杂多通道抽象。
- 可通过 MCP 接入外部工具，但 skill 本身不启动 MCP server。
- 文件工具默认只能操作 `[tools].file_workspace`，通常是 `workspace/files`。

## Metadata

推荐格式：

```yaml
metadata: {"chat_agent":{"always":false,"drift":false,"requires":{"bins":[],"env":[]}}}
```

- `always=true`：每轮注入完整 skill，慎用。
- `drift=true`：可被空闲 DriftManager 作为后台任务执行。
- `requires.bins`：依赖本地命令。
- `requires.env`：依赖环境变量。
