---
name: weather
description: 查询天气、温度、降雨、风力、空气情况或未来预报。用户问天气、明天是否下雨、出门穿什么时使用。
homepage: https://wttr.in/:help
metadata: {"chat_agent":{"always":false,"drift":false,"triggers":["天气","下雨","温度","风力","空气质量","预报","出门穿什么"],"requires":{"bins":[],"env":[],"tools":["web_fetch"]}}}
---

# 天气查询

当前项目优先通过 `web_fetch` 直接访问天气数据源获取天气信息，不把 DuckDuckGo 搜索作为第一选择。不要直接声称没有联网能力；如果工具失败，再给出友好说明。

## 触发场景

- “明天天气怎么样？”
- “北京今天会下雨吗？”
- “出门穿什么？”
- “未来几天气温如何？”

## 工作流程

1. 如果用户没说城市，先询问城市或根据记忆中的常住地推断，但要说明是推断。
2. 先用 `tool_search(query="web_fetch")` 解锁网页抓取工具；这是本地工具发现，不是外网搜索。
3. 优先用 `web_fetch` 访问 `wttr.in` 直连天气页面，例如 `https://wttr.in/Wuhan?format=3` 或 `https://wttr.in/Wuhan?T`。
4. 如果 wttr.in 不可用，再考虑 Open-Meteo；只有直接天气源不可用时，才用搜索/MCP 搜索工具找替代来源。
5. 回答时包含地点、时间范围、温度、降雨/风力、出行建议。
6. 如果工具不可用或超时，不要编造天气；请用户稍后重试或提供网页链接。

## 可用网页

无需 API key 的候选：

```text
https://wttr.in/Beijing?format=3
https://wttr.in/Beijing?T
https://wttr.in/Wuhan?format=3
https://wttr.in/Wuhan?T
https://api.open-meteo.com/
```

如果使用 `web_fetch`，URL 里的空格要替换成 `+`。中文城市优先转成常见英文拼写，例如 `武汉` 使用 `Wuhan`，`北京` 使用 `Beijing`，`上海` 使用 `Shanghai`。

## 输出示例

```text
北京明天可能偏冷，建议带外套。当前查询到的信息显示：...
来源：...
```

## 注意

- 天气是强实时信息，必须优先使用工具。
- 天气查询优先直连天气源，不要先用 DuckDuckGo 搜索天气。
- 不确定时明确说“不确定”，不要用模型旧知识回答。
- 用户只问“明天天气”时，城市不明确就先追问。
- `requires.tools` 只保证 `web_fetch` 已注册；如果它当前不可见，先用 `tool_search(query="web_fetch")` 解锁。
