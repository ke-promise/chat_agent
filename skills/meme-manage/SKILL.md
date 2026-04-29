---
name: meme-manage
description: 维护个人表情包索引。用户要添加表情图片、整理表情分类、更新 manifest、管理 memes 目录时使用。
metadata: {"chat_agent":{"always":false,"drift":false,"triggers":["表情包","斗图","meme","梗图","贴纸","收录图片","管理 memes"],"requires":{"bins":[],"env":[],"tools":["tool_search"]}}}
---

# 表情包索引管理

当前项目支持通过 `list_memes` / `send_meme` 工具发送本地表情包，也支持用户发送图片后用“存成表情包：分类”这类文本把图片收录进本地库。这个 skill 负责维护 workspace 中的表情包资料和 manifest，让 Agent 能更稳定地找到并发对图。

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

1. 查看当前表情包分类：
   - 优先调用 `list_memes`
   - 需要编辑 manifest 时，再用 `tool_search(query="file read write")` 查找文件工具
2. 如果用户只是要发图：
   - 先用 `list_memes` 判断可用分类
   - 需要发送时用 `send_meme`；它是 discoverable/hidden 场景下可能需要先通过工具发现或配置暴露
3. 如果用户要收录图片：
   - 用户本轮带图片，或 5 分钟内刚发过图片时，可以让系统自动收录
   - 推荐话术是“存成表情包：分类”或“加入表情包 分类”
   - QQ 收图依赖 `[qq].download_images = true`，否则图片可能没有本地路径
4. 如果用户要整理分类：
   - 读取 `memes/manifest.json`
   - 更新分类说明、别名、enabled 状态和文件列表
   - 不要直接写二进制图片；图片素材应来自已下载附件或用户放入 `workspace/files/memes/`
5. 操作完成后说明更新了哪些分类、文件或索引。

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
- 不要把外部绝对路径写入 manifest，优先使用分类内相对文件名。
- 用户说“不是表情包”“普通照片”“截图”等否定语时，不要自动收录。
