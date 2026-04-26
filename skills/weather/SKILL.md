---
name: weather
description: 查询天气、温度、降雨、风力、空气情况或未来预报。用户问天气、明天是否下雨、出门穿什么时使用。
homepage: https://wttr.in/:help
metadata: {"chat_agent":{"always":false,"drift":false,"requires":{"bins":[],"env":[]}}}
---

# 天气查询

当前项目可以通过搜索/MCP 工具或 `web_fetch` 获取天气信息。不要直接声称没有联网能力；如果工具失败，再给出友好说明。

## 触发场景

- “明天天气怎么样？”
- “北京今天会下雨吗？”
- “出门穿什么？”
- “未来几天气温如何？”

## 工作流程

1. 如果用户没说城市，先询问城市或根据记忆中的常住地推断，但要说明是推断。
2. 使用 `tool_search(query="weather search web")` 查找可用搜索或网页工具。
3. 优先查询实时来源，例如天气网站、搜索结果、wttr.in、Open-Meteo。
4. 回答时包含地点、时间范围、温度、降雨/风力、出行建议。
5. 如果工具不可用或超时，不要编造天气；请用户稍后重试或提供网页链接。

## 可用网页

无需 API key 的候选：

```text
https://wttr.in/Beijing?format=3
https://wttr.in/Beijing?T
https://api.open-meteo.com/
```

如果使用 `web_fetch`，URL 里的空格要替换成 `+`。

## 输出示例

```text
北京明天可能偏冷，建议带外套。当前查询到的信息显示：...
来源：...
```

## 注意

- 天气是强实时信息，必须优先使用工具。
- 不确定时明确说“不确定”，不要用模型旧知识回答。
- 用户只问“明天天气”时，城市不明确就先追问。
