---
name: meme-manage
description: 维护个人表情包索引。用户要添加表情图片、整理表情分类、更新 manifest、管理 memes 目录时使用。
metadata: {"chat_agent":{"always":false,"drift":false,"requires":{"bins":[],"env":[]}}}
---

# 表情包索引管理

当前项目已经支持通过 `list_memes` / `send_meme` 工具发送本地表情包；这个 skill 负责维护 workspace 中的表情包资料和 manifest，让 Agent 能更稳定地找到并发对图。

## 推荐目录

```text
workspace/files/memes/manifest.json
workspace/files/memes/<category>/001.png
workspace/files/memes/<category>/002.jpg
```

文件工具默认只能访问 `[tools].file_workspace`，通常是 `workspace/files`，所以在工具调用里路径应写成：

```text
read_file("memes/manifest.json")
write_file("memes/manifest.json", ...)
```

## 工作流程

1. 读取当前 manifest：
   - `tool_search(query="file read write")`
   - `read_file("memes/manifest.json")`
2. 确认分类是否存在。
3. 更新分类说明、别名和 enabled 状态。
4. 如果用户通过 Telegram 发送图片，先确认上下文里是否有附件本地路径；当前文本文件工具不适合直接写二进制图片，必要时请用户把图片文件放到目标目录。
5. 操作完成后说明更新了哪些分类或索引。

## manifest 格式

```json
{
  "version": 1,
  "categories": {
    "silent": {
      "desc": "无语、沉默、懒得解释时使用",
      "aliases": ["无语", "沉默"],
      "enabled": true,
      "files": ["001.png", "002.jpg"]
    }
  }
}
```

## 注意

- 不要引用 `agent/memes/` 或 `$HOME/.akashic`。
- 当前 bot 已经可以通过工具发送本地表情包，但前提是 `workspace/files/memes/` 和 `manifest.json` 维护正确。
- 不要把外部绝对路径写入 manifest，优先使用相对路径。
