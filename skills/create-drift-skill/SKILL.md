---
name: create-drift-skill
description: 在 workspace/drift/skills 下创建或更新 drift skill，用于把新的长期空闲任务沉淀成可复用说明书。
metadata: {"chat_agent":{"always":false,"drift":false,"triggers":["创建 drift skill","新增 drift 任务","长期任务","空闲任务","后台任务"],"requires":{"bins":[],"env":[],"tools":["read_skill"]}}}
---

# 创建 Drift Skill

## 目标

把适合反复执行的小任务沉淀到 `workspace/drift/skills/<skill-name>/SKILL.md`。这些 skill 会被 `DriftManager` 当作空闲任务说明书读取，前提是 `[proactive.drift.skills].enabled = true`。

## 何时使用

- 用户想把一个长期、小频率、可后台执行的任务交给 drift。
- 现有 drift skill 太旧，需要补充流程、输出格式或边界。
- 某类主动整理任务反复出现，适合沉淀成说明书。

## 当前项目限制

- 普通 `create_skill` / `update_skill` 工具只写 `workspace/skills/`，不会写 `workspace/drift/skills/`。
- 默认 `write_file` 只能访问 `[tools].file_workspace`，通常是 `workspace/files`，也不能直接写 `workspace/drift/skills/`。
- 因此运行中的 Agent 通常只能帮用户生成 drift skill 内容和目标路径；真正写入需要开发侧文件权限、手工放置，或专门扩展工具。

## 工作流

1. 确认任务是否适合 drift：
   - 可以低频执行
   - 不要求立刻回复用户
   - 输出可以被 proactive 预算、去重和 presence 规则过滤
2. 确认目标 skill 名，使用小写字母、数字和连字符，例如 `weekly-interest-review`。
3. 如果能读取现有 skill，先检查 `workspace/drift/skills/<skill-name>/SKILL.md` 是否已有内容。
4. 新建或更新 `SKILL.md` 时，front matter 至少包含：

```text
---
name: <skill-name>
description: <一句话描述 drift 任务和触发价值>
metadata: {"chat_agent":{"always":false,"drift":true,"triggers":[],"requires":{"bins":[],"env":[],"tools":[]}}}
---
```

5. 正文只写完成当前任务真正需要的最小流程，包含：
   - 任务目标
   - 可用上下文或工具
   - 输出格式
   - 不要做什么

## Drift 输出要求

- 输出应能被 `DriftManager` 解析为候选内容或归档 artifact。
- 面向用户的第一句话要自然，不要写成“后台任务报告”。
- 不要主动发送敏感信息、内部路径、token、cookie 或未经确认的私人数据。
- 如果任务需要实时信息，必须使用可用搜索/MCP 工具；工具失败时不要编造。

## 约束

- skill 文件必须放在 `workspace/drift/skills/`，不要放到仓库内置 `skills/`，除非是在开发内置能力。
- 不要为了一次性动作创建 drift skill。
- 不要把 drift skill 当成可执行插件；它仍然只是给模型读的 SOP。
- 如果只是当前任务状态变化，优先更新外部状态文件或任务说明，不要不断新建 skill。
