"""本地表情包目录、素材治理与自动挂图决策。"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chat_agent.messages import Attachment, OutboundAttachment, OutboundMessage


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
STICKER_SUFFIXES = {".webp", ".tgs"}
CATEGORY_RE = re.compile(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+")
MEME_REQUEST_RE = re.compile(r"(?:(?:来|发|给我|整)(?P<query>[\u4e00-\u9fffA-Za-z0-9_\-]{0,20})?(?:张|个)?|)(?:表情包|斗图|meme)")
EMOTION_QUERY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("开心", ("开心", "高兴", "好耶", "太棒", "夸夸", "嘿嘿")),
    ("可爱", ("可爱", "俏皮", "卖萌", "贴贴", "宝", "软乎乎")),
    ("抱抱", ("抱抱", "安慰", "难过", "委屈", "辛苦", "心疼", "哭哭")),
    ("无语", ("无语", "沉默", "尴尬", "汗颜", "额")),
    ("生气", ("生气", "气死", "火大", "炸毛")),
    ("害羞", ("害羞", "脸红", "不好意思")),
    ("晚安", ("晚安", "困", "睡觉", "打哈欠")),
]
PRIORITY_INBOUND_EMOTION_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("抱抱", ("委屈", "难受", "想哭", "好想哭", "崩溃", "伤心", "低落", "心累", "好累", "受伤", "emo")),
    ("生气", ("气死", "气炸", "火大", "破防", "炸毛")),
    ("开心", ("太开心", "超开心", "开心死了", "激动", "爽到", "好耶")),
]
PRIORITY_INBOUND_EMOTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("抱抱", re.compile(r"(委屈|难受|想哭|好想哭|崩溃|伤心|低落|心累|好累|受伤|emo)", re.IGNORECASE)),
    ("生气", re.compile(r"(气死|气炸|火大|破防|炸毛)", re.IGNORECASE)),
    ("开心", re.compile(r"(太开心|超开心|开心死了|激动|爽到|好耶)", re.IGNORECASE)),
]
DEFAULT_SOURCE_POLICY: dict[str, dict[str, Any]] = {
    "passive": {
        "enabled": True,
        "max_chars": 240,
        "max_lines": 6,
        "min_confidence": 0.66,
        "cooldown_seconds": 90,
        "repeat_window_seconds": 360,
        "allow_random": False,
        "skip_urls": True,
    },
    "reminder": {
        "enabled": True,
        "max_chars": 42,
        "max_lines": 2,
        "min_confidence": 0.84,
        "cooldown_seconds": 240,
        "repeat_window_seconds": 600,
        "allow_random": False,
        "skip_urls": True,
    },
    "feed": {
        "enabled": True,
        "max_chars": 80,
        "max_lines": 3,
        "min_confidence": 0.86,
        "cooldown_seconds": 300,
        "repeat_window_seconds": 900,
        "allow_random": False,
        "skip_urls": True,
    },
    "drift": {
        "enabled": True,
        "max_chars": 90,
        "max_lines": 4,
        "min_confidence": 0.74,
        "cooldown_seconds": 180,
        "repeat_window_seconds": 480,
        "allow_random": False,
        "skip_urls": True,
    },
    "fallback": {
        "enabled": True,
        "max_chars": 56,
        "max_lines": 3,
        "min_confidence": 0.70,
        "cooldown_seconds": 180,
        "repeat_window_seconds": 480,
        "allow_random": False,
        "skip_urls": False,
    },
    "_default": {
        "enabled": True,
        "max_chars": 72,
        "max_lines": 3,
        "min_confidence": 0.76,
        "cooldown_seconds": 180,
        "repeat_window_seconds": 480,
        "allow_random": False,
        "skip_urls": True,
    },
}


@dataclass(frozen=True, slots=True)
class MemeMatch:
    """一次表情包检索命中结果。

    字段:
        path: 命中的本地文件路径。
        category: 命中的表情包分类。
        description: 命中素材的描述文本，通常来自 manifest。
    """

    path: Path
    category: str
    description: str = ""

    @property
    def kind(self) -> str:
        """根据文件后缀推断是 photo 还是 sticker。"""
        return "sticker" if self.path.suffix.lower() in STICKER_SUFFIXES else "photo"


@dataclass(frozen=True, slots=True)
class MemeRequest:
    """显式表情包请求解析结果。

    字段:
        explicit: 当前消息是否明确表达了要表情包。
        query: 用户请求中的分类、情绪或关键词。
    """

    explicit: bool
    query: str = ""


@dataclass(frozen=True, slots=True)
class MemeDecision:
    """自动挂图决策结果。

    字段:
        should_attach: 本轮是否应该自动附带表情包。
        reason: 做出该决策的原因标签。
        query: 最终用于检索表情包的查询词。
        confidence: 该挂图决策的置信度。
        explicit: 是否来自用户显式请求。
        driver: 触发挂图的驱动来源，例如 explicit、emotion 或 random。
    """

    should_attach: bool
    reason: str
    query: str = ""
    confidence: float = 0.0
    explicit: bool = False
    driver: str = "none"


@dataclass(frozen=True, slots=True)
class MemeIngestResult:
    """表情包收录结果。

    字段:
        status: 收录状态，例如 created、duplicate 或 invalid。
        match: 收录后对应的命中对象；失败时可能为空。
        reason: 附加原因说明，便于调试或向用户解释。
        content_hash: 新素材的内容哈希，用于去重。
    """

    status: str
    match: MemeMatch | None = None
    reason: str = ""
    content_hash: str = ""


class MemeCatalog:
    """本地表情包目录索引。"""

    def __init__(self, file_workspace: str | Path) -> None:
        """初始化 `MemeCatalog` 实例。

        参数:
            file_workspace: 初始化 `MemeCatalog` 时需要的 `file_workspace` 参数。
        """
        self.file_workspace = Path(file_workspace)
        self.root = self.file_workspace / "memes"
        self.manifest_path = self.root / "manifest.json"

    def list_categories(self) -> list[dict[str, Any]]:
        """列出可发送的非空表情包分类摘要。"""
        manifest = self._load_manifest()
        if manifest:
            results: list[dict[str, Any]] = []
            for name, info in manifest.get("categories", {}).items():
                if not isinstance(info, dict) or not info.get("enabled", True):
                    continue
                files = [item for item in info.get("files", []) if self._resolve_category_file(name, str(item))]
                count = len(files)
                if count <= 0:
                    continue
                results.append(
                    {
                        "name": str(name),
                        "desc": str(info.get("desc", "")),
                        "aliases": _coerce_string_list(info.get("aliases", [])),
                        "count": count,
                        "mood_tags": _coerce_string_list(info.get("mood_tags", [])),
                        "usage_scenarios": _coerce_string_list(info.get("usage_scenarios", [])),
                        "source_allowlist": _coerce_string_list(info.get("source_allowlist", [])),
                        "priority": _coerce_float(info.get("priority", 0.0)),
                        "auto_attach_enabled": bool(info.get("auto_attach_enabled", True)),
                    }
                )
            return results

        if not self.root.exists():
            return []

        results = []
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue
            count = len([path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES])
            if count:
                results.append(
                    {
                        "name": folder.name,
                        "desc": "",
                        "aliases": [],
                        "count": count,
                        "mood_tags": [],
                        "usage_scenarios": [],
                        "source_allowlist": [],
                        "priority": 0.0,
                        "auto_attach_enabled": True,
                    }
                )
        return results

    def pick_any(
        self,
        source: str = "",
        auto_only: bool = False,
        exclude_categories: set[str] | None = None,
        exclude_paths: set[str] | None = None,
    ) -> MemeMatch | None:
        """按当前筛选条件随机挑选一张表情包。"""
        return self.pick(
            source=source,
            auto_only=auto_only,
            exclude_categories=exclude_categories,
            exclude_paths=exclude_paths,
        )

    def pick(
        self,
        query: str = "",
        category: str = "",
        source: str = "",
        auto_only: bool = False,
        exclude_categories: set[str] | None = None,
        exclude_paths: set[str] | None = None,
    ) -> MemeMatch | None:
        """按分类或查询词挑选一个本地表情包。"""
        manifest = self._load_manifest()
        if manifest:
            return self._pick_from_manifest(
                manifest,
                query=query,
                category=category,
                source=source,
                auto_only=auto_only,
                exclude_categories=exclude_categories,
                exclude_paths=exclude_paths,
            )
        return self._pick_from_scan(
            query=query,
            category=category,
            exclude_categories=exclude_categories,
            exclude_paths=exclude_paths,
        )

    def _pick_from_manifest(
        self,
        manifest: dict[str, Any],
        query: str = "",
        category: str = "",
        source: str = "",
        auto_only: bool = False,
        exclude_categories: set[str] | None = None,
        exclude_paths: set[str] | None = None,
    ) -> MemeMatch | None:
        """挑选`from`、manifest 索引。

        参数:
            manifest: 参与挑选`from`、manifest 索引的 `manifest` 参数。
            query: 参与挑选`from`、manifest 索引的 `query` 参数。
            category: 参与挑选`from`、manifest 索引的 `category` 参数。
            source: 参与挑选`from`、manifest 索引的 `source` 参数。
            auto_only: 参与挑选`from`、manifest 索引的 `auto_only` 参数。
            exclude_categories: 参与挑选`from`、manifest 索引的 `exclude_categories` 参数。
            exclude_paths: 参与挑选`from`、manifest 索引的 `exclude_paths` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        categories = manifest.get("categories", {})
        if not isinstance(categories, dict):
            return None

        candidates: list[tuple[int, MemeMatch]] = []
        wanted_category = category.strip().lower()
        wanted_query = query.strip().lower()
        normalized_source = source.strip().lower()
        excluded_categories = {item.lower() for item in (exclude_categories or set())}
        excluded_paths = {item for item in (exclude_paths or set())}

        for name, info in categories.items():
            if not isinstance(info, dict) or not info.get("enabled", True):
                continue
            name_text = str(name)
            if name_text.lower() in excluded_categories:
                continue
            if auto_only and not info.get("auto_attach_enabled", True):
                continue
            source_allowlist = [item.lower() for item in _coerce_string_list(info.get("source_allowlist", []))]
            if auto_only and normalized_source and source_allowlist and normalized_source not in source_allowlist:
                continue

            aliases = _coerce_string_list(info.get("aliases", []))
            desc = str(info.get("desc", ""))
            mood_tags = _coerce_string_list(info.get("mood_tags", []))
            usage_scenarios = _coerce_string_list(info.get("usage_scenarios", []))
            priority = _coerce_float(info.get("priority", 0.0))
            score = int(priority * 10)
            haystack = " ".join([name_text, desc, *aliases, *mood_tags, *usage_scenarios]).lower()

            if wanted_category:
                if wanted_category == name_text.lower():
                    score += 100
                elif wanted_category in [alias.lower() for alias in aliases]:
                    score += 80
                else:
                    continue

            if wanted_query:
                if wanted_query in haystack:
                    score += 40
                elif any(part and part in haystack for part in wanted_query.split()):
                    score += 20

            files = [self._resolve_category_file(name_text, str(item)) for item in info.get("files", [])]
            resolved = [path for path in files if path and str(path) not in excluded_paths]
            if not resolved:
                continue

            chosen = random.choice(resolved)
            if not wanted_category and not wanted_query:
                score += 1
            candidates.append((score, MemeMatch(path=chosen, category=name_text, description=desc)))

        if not candidates:
            return None
        return self._pick_top_candidate(candidates)

    def _pick_from_scan(
        self,
        query: str = "",
        category: str = "",
        exclude_categories: set[str] | None = None,
        exclude_paths: set[str] | None = None,
    ) -> MemeMatch | None:
        """挑选`from`、`scan`。

        参数:
            query: 参与挑选`from`、`scan`的 `query` 参数。
            category: 参与挑选`from`、`scan`的 `category` 参数。
            exclude_categories: 参与挑选`from`、`scan`的 `exclude_categories` 参数。
            exclude_paths: 参与挑选`from`、`scan`的 `exclude_paths` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if not self.root.exists():
            return None

        wanted_category = category.strip().lower()
        wanted_query = query.strip().lower()
        excluded_categories = {item.lower() for item in (exclude_categories or set())}
        excluded_paths = {item for item in (exclude_paths or set())}
        candidates: list[tuple[int, MemeMatch]] = []

        for path in self.root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if str(path) in excluded_paths:
                continue
            rel = path.relative_to(self.root).as_posix().lower()
            category_name = path.parent.name
            if category_name.lower() in excluded_categories:
                continue
            score = 1
            if wanted_category:
                if wanted_category == category_name.lower():
                    score += 100
                elif wanted_category not in rel:
                    continue
            if wanted_query:
                if wanted_query in rel or wanted_query in path.stem.lower():
                    score += 40
                elif any(part and part in rel for part in wanted_query.split()):
                    score += 20
            candidates.append((score, MemeMatch(path=path, category=category_name)))

        if not candidates:
            return None
        return self._pick_top_candidate(candidates)

    def _pick_top_candidate(self, candidates: list[tuple[int, MemeMatch]]) -> MemeMatch:
        """挑选`top`、候选项。

        参数:
            candidates: 参与挑选`top`、候选项的 `candidates` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        top_score = max(score for score, _ in candidates)
        top_matches = [match for score, match in candidates if score == top_score]
        return random.choice(top_matches)

    def _resolve_category_file(self, category: str, value: str) -> Path | None:
        """解析分类、文件。

        参数:
            category: 参与解析分类、文件的 `category` 参数。
            value: 参与解析分类、文件的 `value` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        raw = Path(value)
        if raw.is_absolute():
            return None
        path = (self.root / category / raw).resolve()
        if self.root.resolve() not in path.parents:
            return None
        if not path.exists() or not path.is_file():
            return None
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            return None
        return path

    def _load_manifest(self) -> dict[str, Any] | None:
        """加载manifest 索引。

        返回:
            返回与本函数处理结果对应的数据。"""
        if not self.manifest_path.exists():
            return None
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None


class MemeService:
    """表情包收录与自动挂图服务。"""

    def __init__(self, file_workspace: str | Path, max_category_files: int = 50) -> None:
        """初始化 `MemeService` 实例。

        参数:
            file_workspace: 初始化 `MemeService` 时需要的 `file_workspace` 参数。
            max_category_files: 初始化 `MemeService` 时需要的 `max_category_files` 参数。
        """
        self.catalog = MemeCatalog(file_workspace)
        self.max_category_files = max_category_files
        self._recent_attaches: dict[str, deque[dict[str, Any]]] = {}

    def ingest_attachment(self, attachment: Attachment, category: str, description: str = "") -> MemeIngestResult:
        """把一张已下载到本地的入站图片收录进表情包目录。"""
        if attachment.kind != "image" or not attachment.local_path:
            return MemeIngestResult(status="rejected", reason="unsupported_attachment")
        source = Path(attachment.local_path)
        if not source.exists() or not source.is_file():
            return MemeIngestResult(status="rejected", reason="missing_source")

        clean_category = self._sanitize_category(category)
        target_dir = self.catalog.root / clean_category
        target_dir.mkdir(parents=True, exist_ok=True)

        source_hash = _file_sha256(source)
        duplicate = self._find_duplicate_by_hash(target_dir, source_hash)
        if duplicate:
            self._upsert_manifest(clean_category, duplicate.name, description=description)
            return MemeIngestResult(
                status="duplicate",
                match=MemeMatch(path=duplicate, category=clean_category, description=description),
                reason="duplicate_hash",
                content_hash=source_hash,
            )

        existing_files = [path for path in target_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
        if len(existing_files) >= self.max_category_files:
            return MemeIngestResult(status="rejected", reason="category_full", content_hash=source_hash)

        suffix = source.suffix.lower() or self._guess_suffix(attachment.mime_type)
        if suffix not in IMAGE_SUFFIXES:
            suffix = ".jpg"
        filename = self._next_filename(target_dir, suffix)
        target = target_dir / filename
        shutil.copy2(source, target)
        self._upsert_manifest(clean_category, filename, description=description)
        return MemeIngestResult(
            status="added",
            match=MemeMatch(path=target, category=clean_category, description=description),
            content_hash=source_hash,
        )

    def decorate_outbound(
        self,
        message: OutboundMessage,
        inbound_text: str = "",
        source: str = "passive",
    ) -> OutboundMessage:
        """根据规则门禁和语义信号，为出站消息统一决策是否挂图。"""
        if message.attachments:
            return message

        decision = self.should_attach_meme(message, inbound_text=inbound_text, source=source)
        if not decision.should_attach:
            return message

        match = self.select_meme(message, decision, source=source)
        if not match:
            return message

        self._record_attach(message.chat_id, source, decision, match)
        content = self._sanitize_text_for_attached_meme(message.content)
        return OutboundMessage(
            channel=message.channel,
            chat_id=message.chat_id,
            content=content,
            attachments=[OutboundAttachment(kind=match.kind, local_path=str(match.path))],
            reply_to_message_id=message.reply_to_message_id,
            metadata={
                **message.metadata,
                "meme_decision_source": source,
                "meme_decision_driver": decision.driver,
                "meme_decision_reason": decision.reason,
                "meme_decision_confidence": round(decision.confidence, 3),
                "meme_decision_explicit": decision.explicit,
                "auto_meme_query": decision.query,
                "auto_meme_category": match.category,
            },
        )

    def should_attach_meme(
        self,
        message: OutboundMessage,
        inbound_text: str = "",
        source: str = "passive",
    ) -> MemeDecision:
        """先做规则门禁，再做语义判断，决定是否应该挂图。"""
        request = self.extract_requested_meme(inbound_text)
        if request.explicit:
            return MemeDecision(
                should_attach=True,
                reason="explicit_request",
                query=request.query,
                confidence=1.0,
                explicit=True,
                driver="explicit_request",
            )

        text = message.content.strip()
        if not text:
            return MemeDecision(False, "empty_content")

        policy = self._get_source_policy(source)
        if not policy.get("enabled", True):
            return MemeDecision(False, "source_disabled")
        if self._is_cooling_down(message.chat_id, source, policy):
            return MemeDecision(False, "cooldown_active")

        priority_signal = self._guess_priority_inbound_emotion(inbound_text)
        if self._is_information_heavy(text, policy) and not self._can_override_information_gate(
            priority_signal,
            source=source,
            text=text,
            policy=policy,
        ):
            return MemeDecision(False, "information_heavy")

        signal = priority_signal if priority_signal.query else self._guess_emotion_signal(f"{inbound_text}\n{text}")
        if not signal.query:
            return MemeDecision(False, "no_emotion_signal")
        min_confidence = float(policy.get("min_confidence", 0.75))
        if priority_signal.query and source == "passive":
            min_confidence = min(min_confidence, 0.55)
        if signal.confidence < min_confidence:
            return MemeDecision(False, "emotion_too_weak", query=signal.query, confidence=signal.confidence)

        return MemeDecision(
            should_attach=True,
            reason=signal.reason,
            query=signal.query,
            confidence=signal.confidence,
            explicit=False,
            driver=signal.driver,
        )

    def select_meme(self, message: OutboundMessage, decision: MemeDecision, source: str = "passive") -> MemeMatch | None:
        """根据决策结果，从目录里选择最合适的一张图。"""
        policy = self._get_source_policy(source)
        exclude_categories, exclude_paths = self._recent_exclusions(
            message.chat_id,
            source,
            window_seconds=int(policy.get("repeat_window_seconds", 480)),
        )
        auto_only = not decision.explicit
        if decision.query:
            match = self.catalog.pick(
                query=decision.query,
                source=source,
                auto_only=auto_only,
                exclude_categories=exclude_categories,
                exclude_paths=exclude_paths,
            )
            if match:
                return match

        if decision.explicit and not decision.query:
            return self.catalog.pick_any(source=source, auto_only=False)
        if policy.get("allow_random", False):
            return self.catalog.pick_any(
                source=source,
                auto_only=True,
                exclude_categories=exclude_categories,
                exclude_paths=exclude_paths,
            )
        return None

    def extract_requested_meme(self, text: str) -> MemeRequest:
        """从“来个开心表情包”这类文本里提取显式请求。"""
        raw = text.strip()
        if not raw:
            return MemeRequest(explicit=False)
        match = MEME_REQUEST_RE.search(raw)
        if match:
            return MemeRequest(explicit=True, query=self._normalize_request_query(match.group("query") or ""))

        lowered = raw.lower()
        if not any(token in lowered for token in ("表情包", "斗图", "meme")):
            return MemeRequest(explicit=False)
        if not any(token in lowered for token in ("来", "发", "给我", "整")):
            return MemeRequest(explicit=False)

        cleaned = raw
        for token in ("表情包", "斗图", "meme", "给我", "来", "发", "整", "一个", "一张", "一套", "个", "张"):
            cleaned = cleaned.replace(token, " ")
        query = self._normalize_request_query(" ".join(part for part in re.split(r"\s+", cleaned) if part).strip())
        return MemeRequest(explicit=True, query=query)

    def extract_requested_query(self, text: str) -> str:
        """兼容旧接口，返回显式请求中的 query。"""
        return self.extract_requested_meme(text).query

    def _guess_emotion_signal(self, text: str) -> MemeDecision:
        """猜测`emotion`、`signal`。

        参数:
            text: 参与猜测`emotion`、`signal`的 `text` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        lowered = text.lower()
        best_query = ""
        best_tokens: list[str] = []
        for query, tokens in EMOTION_QUERY_RULES:
            matched = [token for token in tokens if token.lower() in lowered]
            if len(matched) > len(best_tokens):
                best_query = query
                best_tokens = matched

        if not best_query:
            return MemeDecision(False, "no_emotion_signal")

        confidence = min(0.95, 0.56 + 0.13 * len(best_tokens))
        if best_query.lower() in lowered:
            confidence = max(confidence, 0.78)
        return MemeDecision(
            should_attach=True,
            reason=f"emotion_signal:{','.join(best_tokens[:3])}",
            query=best_query,
            confidence=confidence,
            driver="emotion_rule",
        )

    def _guess_priority_inbound_emotion(self, text: str) -> MemeDecision:
        """猜测`priority`、`inbound`、`emotion`。

        参数:
            text: 参与猜测`priority`、`inbound`、`emotion`的 `text` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        lowered = text.lower().strip()
        if not lowered:
            return MemeDecision(False, "no_priority_inbound_emotion")

        best_query = ""
        best_tokens: list[str] = []
        for query, pattern in PRIORITY_INBOUND_EMOTION_PATTERNS:
            matched = [item for item in pattern.findall(lowered) if str(item).strip()]
            if len(matched) > len(best_tokens):
                best_query = query
                best_tokens = matched

        if not best_query:
            return MemeDecision(False, "no_priority_inbound_emotion")

        confidence = min(0.98, 0.82 + 0.06 * len(best_tokens))
        return MemeDecision(
            should_attach=True,
            reason=f"priority_inbound_emotion:{','.join(best_tokens[:3])}",
            query=best_query,
            confidence=confidence,
            driver="priority_emotion",
        )

    def _sanitize_category(self, value: str) -> str:
        """处理分类。

        参数:
            value: 参与处理分类的 `value` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        cleaned = CATEGORY_RE.sub("-", value.strip())
        cleaned = cleaned.strip("-_ ")
        return cleaned or "misc"

    def _next_filename(self, folder: Path, suffix: str) -> str:
        """处理`filename`。

        参数:
            folder: 参与处理`filename`的 `folder` 参数。
            suffix: 参与处理`filename`的 `suffix` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        max_index = 0
        for path in folder.iterdir():
            if not path.is_file():
                continue
            stem = path.stem
            if stem.isdigit():
                max_index = max(max_index, int(stem))
        return f"{max_index + 1:03d}{suffix}"

    def _find_duplicate_by_hash(self, folder: Path, content_hash: str) -> Path | None:
        """查找`duplicate`、`by`、哈希值。

        参数:
            folder: 参与查找`duplicate`、`by`、哈希值的 `folder` 参数。
            content_hash: 参与查找`duplicate`、`by`、哈希值的 `content_hash` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        for path in folder.iterdir():
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if _file_sha256(path) == content_hash:
                return path
        return None

    def _upsert_manifest(self, category: str, filename: str, description: str = "") -> None:
        """处理manifest 索引。

        参数:
            category: 参与处理manifest 索引的 `category` 参数。
            filename: 参与处理manifest 索引的 `filename` 参数。
            description: 参与处理manifest 索引的 `description` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        self.catalog.root.mkdir(parents=True, exist_ok=True)
        manifest = self.catalog._load_manifest() or {"version": 1, "categories": {}}
        categories = manifest.setdefault("categories", {})
        entry = categories.setdefault(
            category,
            {
                "desc": description or f"{category} 类表情包",
                "aliases": [category],
                "enabled": True,
                "files": [],
                "mood_tags": [],
                "usage_scenarios": [],
                "source_allowlist": [],
                "priority": 0,
                "auto_attach_enabled": True,
            },
        )
        if description and not entry.get("desc"):
            entry["desc"] = description
        files = entry.setdefault("files", [])
        if filename not in files:
            files.append(filename)
        aliases = entry.setdefault("aliases", [])
        if category not in aliases:
            aliases.append(category)
        entry.setdefault("mood_tags", [])
        entry.setdefault("usage_scenarios", [])
        entry.setdefault("source_allowlist", [])
        entry.setdefault("priority", 0)
        entry.setdefault("auto_attach_enabled", True)
        self.catalog.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _guess_suffix(self, mime_type: str | None) -> str:
        """猜测`suffix`。

        参数:
            mime_type: 参与猜测`suffix`的 `mime_type` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if mime_type == "image/png":
            return ".png"
        if mime_type == "image/webp":
            return ".webp"
        if mime_type == "image/gif":
            return ".gif"
        return ".jpg"

    def _get_source_policy(self, source: str) -> dict[str, Any]:
        """处理数据源、`policy`。

        参数:
            source: 参与处理数据源、`policy`的 `source` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        return DEFAULT_SOURCE_POLICY.get(source, DEFAULT_SOURCE_POLICY["_default"])

    def _is_information_heavy(self, text: str, policy: dict[str, Any]) -> bool:
        """处理`information`、`heavy`。

        参数:
            text: 参与处理`information`、`heavy`的 `text` 参数。
            policy: 参与处理`information`、`heavy`的 `policy` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        stripped = text.strip()
        if len(stripped) > int(policy.get("max_chars", 80)):
            return True
        if stripped.count("\n") + 1 > int(policy.get("max_lines", 3)):
            return True
        if policy.get("skip_urls", True) and any(token in stripped.lower() for token in ("http://", "https://", "www.")):
            return True
        if len(re.findall(r"[：:;/|]", stripped)) >= 4 and len(stripped) >= 40:
            return True
        sentence_count = len([part for part in re.split(r"[。！？!?]", stripped) if part.strip()])
        return sentence_count >= 4 and len(stripped) >= int(policy.get("max_chars", 80)) * 0.6

    def _can_override_information_gate(
        self,
        signal: MemeDecision,
        *,
        source: str,
        text: str,
        policy: dict[str, Any],
    ) -> bool:
        """处理`override`、`information`、`gate`。

        参数:
            signal: 参与处理`override`、`information`、`gate`的 `signal` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if source != "passive" or not signal.query:
            return False
        stripped = text.strip()
        max_chars = int(policy.get("max_chars", 80))
        max_lines = int(policy.get("max_lines", 3))
        if len(stripped) > max_chars * 2:
            return False
        if stripped.count("\n") + 1 > max_lines + 3:
            return False
        return True

    def _is_cooling_down(self, chat_id: str, source: str, policy: dict[str, Any]) -> bool:
        """处理`cooling`、`down`。

        参数:
            chat_id: 参与处理`cooling`、`down`的 `chat_id` 参数。
            source: 参与处理`cooling`、`down`的 `source` 参数。
            policy: 参与处理`cooling`、`down`的 `policy` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        cooldown = int(policy.get("cooldown_seconds", 0))
        if cooldown <= 0:
            return False
        cutoff = time.time() - cooldown
        history = self._recent_attaches.get(chat_id)
        if not history:
            return False
        for item in reversed(history):
            if item["timestamp"] < cutoff:
                break
            if item["source"] == source:
                return True
        return False

    def _recent_exclusions(self, chat_id: str, source: str, window_seconds: int) -> tuple[set[str], set[str]]:
        """处理`exclusions`。

        参数:
            chat_id: 参与处理`exclusions`的 `chat_id` 参数。
            source: 参与处理`exclusions`的 `source` 参数。
            window_seconds: 参与处理`exclusions`的 `window_seconds` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if window_seconds <= 0:
            return set(), set()
        cutoff = time.time() - window_seconds
        history = self._recent_attaches.get(chat_id)
        if not history:
            return set(), set()
        categories: set[str] = set()
        paths: set[str] = set()
        for item in reversed(history):
            if item["timestamp"] < cutoff:
                break
            if item["source"] != source:
                continue
            categories.add(str(item["category"]))
            paths.add(str(item["path"]))
        return categories, paths

    def _record_attach(self, chat_id: str, source: str, decision: MemeDecision, match: MemeMatch) -> None:
        """处理`attach`。

        参数:
            chat_id: 参与处理`attach`的 `chat_id` 参数。
            source: 参与处理`attach`的 `source` 参数。
            decision: 参与处理`attach`的 `decision` 参数。
            match: 参与处理`attach`的 `match` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        history = self._recent_attaches.setdefault(chat_id, deque(maxlen=8))
        history.append(
            {
                "timestamp": time.time(),
                "source": source,
                "driver": decision.driver,
                "query": decision.query,
                "category": match.category,
                "path": str(match.path),
            }
        )

    def _sanitize_text_for_attached_meme(self, text: str) -> str:
        """挂图已确定时，去掉多余征询和“发不了图”类自相矛盾文案。"""
        if "表情包" not in text and "发图" not in text and "图片" not in text:
            return text

        paragraphs = [part for part in text.split("\n\n") if part.strip()]
        filtered = [part for part in paragraphs if not _is_meme_meta_paragraph(part)]
        if filtered:
            return "\n\n".join(filtered).strip()

        sentences = [part.strip() for part in re.split(r"(?<=[。！？!?])", text) if part.strip()]
        filtered_sentences = [part for part in sentences if not _is_meme_meta_paragraph(part)]
        if filtered_sentences:
            return "".join(filtered_sentences).strip()
        return text

    def _normalize_request_query(self, raw_query: str) -> str:
        """归一化请求、查询词。

        参数:
            raw_query: 参与归一化请求、查询词的 `raw_query` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        query = raw_query.strip()
        if not query:
            return ""
        filler_tokens = {"个", "张", "一个", "一张", "一套", "个图", "张图"}
        return "" if query in filler_tokens else query


def _coerce_string_list(value: Any) -> list[str]:
    """转换`string`、`list`。

    参数:
        value: 参与转换`string`、`list`的 `value` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _coerce_float(value: Any) -> float:
    """转换浮点值。

    参数:
        value: 参与转换浮点值的 `value` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _file_sha256(path: Path) -> str:
    """处理`sha256`。

    参数:
        path: 参与处理`sha256`的 `path` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_meme_offer_paragraph(text: str) -> bool:
    """处理`meme`、`offer`、`paragraph`。

    参数:
        text: 参与处理`meme`、`offer`、`paragraph`的 `text` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    stripped = text.strip()
    if "表情包" not in stripped:
        return False
    return any(token in stripped for token in ("要不要", "要不", "给你发个", "给你发一", "我给你发个", "我给你发一"))


def _is_attachment_contradiction_paragraph(text: str) -> bool:
    """处理`attachment`、`contradiction`、`paragraph`。

    参数:
        text: 参与处理`attachment`、`contradiction`、`paragraph`的 `text` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    stripped = text.strip()
    contradiction_tokens = (
        "发不了图",
        "没办法直接给你发图片",
        "没办法直接给你发图片文件",
        "暂时还没办法直接给你发图片",
        "不能直接给你发图片",
        "还没办法直接给你发图",
        "虽然发不了图",
    )
    if any(token in stripped for token in contradiction_tokens):
        return True
    return ("图片" in stripped or "发图" in stripped) and any(
        token in stripped for token in ("没办法", "不能", "发不了", "暂时还没法")
    )


def _is_meme_meta_paragraph(text: str) -> bool:
    """处理`meme`、`meta`、`paragraph`。

    参数:
        text: 参与处理`meme`、`meta`、`paragraph`的 `text` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    return _is_meme_offer_paragraph(text) or _is_attachment_contradiction_paragraph(text)
