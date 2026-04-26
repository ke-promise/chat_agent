"""用户兴趣线索提取与匹配。"""

from __future__ import annotations

import re
from typing import Any


_CJK_OR_WORD_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fffA-Za-z0-9+_.:/-]{1,}")
_COMMON_PREFIXES = (
    "用户喜欢",
    "用户偏好",
    "用户希望",
    "用户想看",
    "我喜欢",
    "喜欢玩",
    "喜欢",
    "偏好",
    "希望",
    "user likes playing",
    "user likes",
    "likes playing",
)
_STOP_TOKENS = {
    "用户",
    "喜欢",
    "偏好",
    "希望",
    "定期",
    "获取",
    "今日",
    "当日",
    "最近",
    "简洁",
    "回答",
    "摘要",
    "新闻",
    "热点",
}


def build_interest_watchlist(user_profile: dict[str, Any], memories: list[dict[str, Any]], limit: int = 12) -> list[str]:
    """从用户画像和 preference 记忆中提取稳定兴趣线索。"""
    hints: list[str] = []
    preferences = user_profile.get("preferences", []) if isinstance(user_profile, dict) else []
    if isinstance(preferences, list):
        for item in preferences:
            text = str(item).strip()
            if text:
                hints.append(text)
    for key in ("interests", "likes", "topics"):
        value = user_profile.get(key) if isinstance(user_profile, dict) else None
        if isinstance(value, list):
            for item in value:
                text = str(item).strip()
                if text:
                    hints.append(text)
    for item in memories:
        tags = item.get("tags") or []
        normalized_tags = {str(tag).strip().lower() for tag in tags} if isinstance(tags, list) else set()
        if str(item.get("type")) != "preference" and "preference" not in normalized_tags:
            continue
        text = str(item.get("content", "")).strip()
        if text:
            hints.append(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        normalized = normalize_interest_text(hint)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(hint)
    return deduped[:limit]


def render_interest_watchlist_md(hints: list[str]) -> str:
    """把兴趣线索渲染成 Markdown。"""
    lines = ["# Interests", ""]
    if not hints:
        lines.append("(empty)")
        return "\n".join(lines) + "\n"
    for hint in hints:
        terms = extract_interest_terms(hint)
        suffix = f"  关键词：{' / '.join(terms[:6])}" if terms else ""
        lines.append(f"- {hint}{suffix}")
    return "\n".join(lines) + "\n"


def parse_interest_watchlist_md(text: str) -> list[str]:
    """从 INTERESTS.md 中读取兴趣线索。"""
    hints: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        hint = line[2:].split("关键词：", 1)[0].strip()
        if hint:
            hints.append(hint)
    deduped: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        normalized = normalize_interest_text(hint)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(hint)
    return deduped


def extract_interest_terms(hint: str) -> list[str]:
    """从一条兴趣线索中提取更适合匹配与搜索的关键词。"""
    text = normalize_interest_text(hint)
    if not text:
        return []
    lowered = text.lower()
    for prefix in _COMMON_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip(" ：:，,。.!！")
            lowered = text.lower()
            break
    parts = re.split(r"[，,。；;、\s/|]+", text)
    terms: list[str] = []
    for part in parts:
        token = part.strip("()[]{}<>\"'“”‘’：:，,。.!！ ")
        if not token:
            continue
        if token.lower() not in _STOP_TOKENS:
            terms.append(token)
        for match in _CJK_OR_WORD_PATTERN.findall(token):
            candidate = match.strip()
            if len(candidate) < 2:
                continue
            if candidate.lower() in _STOP_TOKENS:
                continue
            if candidate not in terms:
                terms.append(candidate)
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(term)
    return deduped[:8]


def interest_match_score(text: str, hints: list[str]) -> tuple[float, list[str]]:
    """计算一段文本与兴趣 watchlist 的粗粒度匹配分。"""
    haystack = normalize_interest_text(text)
    if not haystack or not hints:
        return 0.0, []
    matched: list[str] = []
    score = 0.0
    for hint in hints:
        terms = extract_interest_terms(hint)
        if not terms:
            continue
        local_hits = [term for term in terms if normalize_interest_text(term) and normalize_interest_text(term) in haystack]
        if not local_hits:
            continue
        matched.extend(local_hits)
        if len(local_hits) >= 2:
            score = max(score, 0.95)
        else:
            longest = max(len(term) for term in local_hits)
            score = max(score, 0.9 if longest >= 4 else 0.78)
    deduped: list[str] = []
    seen: set[str] = set()
    for term in matched:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(term)
    return min(score, 1.0), deduped[:6]


def normalize_interest_text(text: str) -> str:
    """标准化兴趣文本，便于做简单包含匹配。"""
    return re.sub(r"\s+", "", str(text or "")).lower()
