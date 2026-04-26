"""主动系统共享数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class ProactiveCandidate:
    """主动触达系统的统一候选结构。

    Attributes:
        candidate_id: 候选唯一 id，用于去重、审计和确认发送。
        source_type: 候选来源类型，例如 fallback、feed 或 drift。
        title: 候选标题，便于日志和后台审计。
        body: 候选主体内容，最终可能转换成用户可见消息。
        url: 关联链接；没有链接时可为空字符串。
        confidence: 候选本身的可信度分数。
        novelty: 相对用户近期上下文的新鲜度分数。
        user_fit: 与当前用户兴趣、画像和会话状态的匹配度。
        priority: 综合优先级，用于候选排序。
        shareable: 是否允许直接面向用户发送。
        created_at: 候选生成时间。
        expires_at: 过期时间；为空表示短期内不过期。
        dedupe_key: 用于跨轮去重的稳定键。
        artifact_path: 关联产物路径，例如 drift 生成的文件。
        summary: 对候选的简短摘要。
        source_label: 面向日志或审计的人类可读来源标签。
        image_url: 可选配图地址，供主动推送时附图使用。
    """

    candidate_id: str
    source_type: str
    title: str
    body: str
    url: str
    confidence: float
    novelty: float
    user_fit: float
    priority: float
    shareable: bool
    created_at: datetime
    expires_at: datetime | None
    dedupe_key: str
    artifact_path: str | None = None
    summary: str = ""
    source_label: str = ""
    image_url: str = ""
