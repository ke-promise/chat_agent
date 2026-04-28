"""Telegram 回复文本格式化。

模型输出通常包含 Markdown、编号列表或引用说明；本模块负责做轻量清理，
让 Telegram 中展示更舒适，同时避免改变回答本身的事实内容。
"""

from __future__ import annotations

import re


def format_reply(text: str) -> str:
    """统一整理发送给 Telegram 的回复文本。

    参数:
        text: LLM 或直接规则生成的原始回复。

    返回:
        适合直接发送的文本。会去掉包裹引号、压缩多余空白、弱化 Markdown 装饰，并整理列表间距。
    """
    text = text.strip()
    if not text:
        return text
    text = _strip_wrapping_quotes(text)
    text = _normalize_spacing(text)
    text = _normalize_markdown(text)
    text = _normalize_list_spacing(text)
    return text.strip()


def _strip_wrapping_quotes(text: str) -> str:
    """去掉模型偶尔包在整段回复外层的单/双引号。"""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"', "“", "”"}:
        return text[1:-1].strip()
    return text


def _normalize_spacing(text: str) -> str:
    """规范换行和空格，特别处理中文之间异常插入的空格。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    # 修正模型分词偶尔在常见中文词内部插入空格的问题。
    replacements = {
        "提 质": "提质",
        "致 力": "致力",
        "杀伤 性": "杀伤性",
        "高级 副总裁": "高级副总裁",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    return text


def _normalize_markdown(text: str) -> str:
    """移除 Telegram 普通文本中容易显得杂乱的 Markdown 强调和标题符号。"""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    return text


def _normalize_list_spacing(text: str) -> str:
    """整理列表回复的空行，避免 Telegram 中挤成一团或空行过多。"""
    lines = [line.rstrip() for line in text.split("\n")]
    compact: list[str] = []
    previous_blank = False
    for line in lines:
        blank = not line.strip()
        if blank and previous_blank:
            continue
        compact.append(line)
        previous_blank = blank
    text = "\n".join(compact)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n(\d+\.\s)", r"\n\n\1", text)
    text = re.sub(r"\n\n\n+", "\n\n", text)
    text = re.sub(r"^\n+", "", text)
    return text
