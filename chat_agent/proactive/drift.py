"""空闲 Drift preparation 任务执行器。"""

from __future__ import annotations

import json
import logging
import re
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from chat_agent.agent.provider import LLMProvider
from chat_agent.context import ContextBundle
from chat_agent.memory.interests import build_interest_watchlist
from chat_agent.memory.store import SQLiteStore, utc_now
from chat_agent.messages import InboundMessage
from chat_agent.proactive.models import ProactiveCandidate
from chat_agent.reasoner import Reasoner
from chat_agent.skills import SkillsLoader
from chat_agent.tools.registry import Tool, ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

_SCORE_WORDS = {
    "very_low": 0.1,
    "low": 0.25,
    "medium_low": 0.4,
    "medium": 0.55,
    "medium_high": 0.7,
    "high": 0.82,
    "very_high": 0.93,
}

_TRUE_WORDS = {"true", "1", "yes", "y", "on", "share", "shareable"}
_FALSE_WORDS = {"false", "0", "no", "n", "off", "none", "not_shareable"}


@dataclass(slots=True)
class DriftTask:
    """一次 drift 可执行任务的描述。"""

    id: str
    title: str
    prompt: str
    enabled: bool = True


@dataclass(slots=True)
class DriftResult:
    """DriftManager.run_once 的执行结果。"""

    ran: bool
    reason: str | None = None
    task_id: str | None = None
    output_path: str | None = None
    candidate: ProactiveCandidate | None = None


class DriftManager:
    """空闲任务执行器。"""

    def __init__(
        self,
        store: SQLiteStore,
        provider: LLMProvider,
        tasks_path: Path,
        output_dir: Path,
        run_cooldown_minutes: int = 180,
        daily_run_cap: int = 3,
        promotion_enabled: bool = True,
        target_chat_id: str = "",
        skills_loader: SkillsLoader | None = None,
        tools: ToolRegistry | None = None,
        max_iterations: int = 5,
    ) -> None:
        """初始化 `DriftManager` 实例。

        参数:
            store: 初始化 `DriftManager` 时需要的 `store` 参数。
            provider: 初始化 `DriftManager` 时需要的 `provider` 参数。
            tasks_path: 初始化 `DriftManager` 时需要的 `tasks_path` 参数。
            output_dir: 初始化 `DriftManager` 时需要的 `output_dir` 参数。
            run_cooldown_minutes: 初始化 `DriftManager` 时需要的 `run_cooldown_minutes` 参数。
            daily_run_cap: 初始化 `DriftManager` 时需要的 `daily_run_cap` 参数。
            promotion_enabled: 初始化 `DriftManager` 时需要的 `promotion_enabled` 参数。
            target_chat_id: 初始化 `DriftManager` 时需要的 `target_chat_id` 参数。
            skills_loader: 初始化 `DriftManager` 时需要的 `skills_loader` 参数。
            tools: 初始化 `DriftManager` 时需要的 `tools` 参数。
            max_iterations: 初始化 `DriftManager` 时需要的 `max_iterations` 参数。
        """
        self.store = store
        self.provider = provider
        self.tasks_path = Path(tasks_path)
        self.output_dir = Path(output_dir)
        self.run_cooldown = timedelta(minutes=run_cooldown_minutes)
        self.daily_run_cap = daily_run_cap
        self.promotion_enabled = promotion_enabled
        self.target_chat_id = target_chat_id
        self.skills_loader = skills_loader
        self.tools = tools
        self.max_iterations = max_iterations
        self._last_tool_evidence: dict[str, Any] = {}

    def load_tasks(self) -> list[DriftTask]:
        """加载当前可用 drift 任务。"""
        skill_tasks = self.load_skill_tasks()
        json_tasks: list[DriftTask] = []
        if not self.tasks_path.exists():
            logger.warning("Drift tasks config not found: %s", self.tasks_path)
            return skill_tasks
        data = json.loads(self.tasks_path.read_text(encoding="utf-8"))
        for item in data.get("tasks", []):
            json_tasks.append(
                DriftTask(
                    id=str(item.get("id", "")),
                    title=str(item.get("title", "")),
                    prompt=str(item.get("prompt", "")),
                    enabled=bool(item.get("enabled", True)),
                )
            )
        return [task for task in [*skill_tasks, *json_tasks] if task.enabled and task.id and task.prompt]

    def load_skill_tasks(self) -> list[DriftTask]:
        """从 SkillsLoader 中提取适合 drift 的任务。"""
        if not self.skills_loader:
            return []
        tasks: list[DriftTask] = []
        for item in self.skills_loader.list_skills(filter_unavailable=True):
            metadata = item.get("metadata", {})
            chat_agent_meta = metadata.get("chat_agent", {}) if isinstance(metadata, dict) else {}
            drift_enabled = bool(chat_agent_meta.get("drift", item.get("source") == "workspace"))
            if not drift_enabled:
                continue
            body = self.skills_loader.load_skill(str(item["name"]))
            if not body:
                continue
            tasks.append(
                DriftTask(
                    id=f"skill-{item['name']}",
                    title=str(item.get("description") or item["name"]),
                    prompt=body,
                    enabled=True,
                )
            )
        return tasks

    async def can_run(self) -> tuple[bool, str | None]:
        """判断当前是否允许运行 drift。"""
        if self.daily_run_cap <= 0:
            return False, "daily_run_cap"
        today = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
        if await self.store.count_drift_runs_since(today) >= self.daily_run_cap:
            return False, "daily_run_cap"
        if not self.load_tasks():
            return False, "no_task"
        return True, None

    async def run_once(self) -> DriftResult:
        """尝试运行一次 drift 任务。"""
        ok, reason = await self.can_run()
        if not ok:
            return DriftResult(ran=False, reason=reason)

        tasks = self.load_tasks()
        states = await self.store.get_drift_task_states()
        task = self._select_task(tasks, states)
        if not task:
            return DriftResult(ran=False, reason="cooldown")

        started_at = utc_now()
        try:
            now_text = started_at.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            current_year = started_at.year
            context = await self._build_context()
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是个人智能体的后台 drift 执行器。"
                        f"当前日期时间是 {now_text}，当前年份是 {current_year}。"
                        f"除非用户明确要求回顾历史，否则搜索查询不得使用早于 {current_year} 年的年份。"
                        "请完成一个后台准备任务，输出必须包含两个部分。"
                        "第一部分是一个 JSON 对象，字段固定为 shareable,title,body,priority,confidence,novelty,user_fit。"
                        "shareable 默认应为 false，只有在内容高价值、高置信且适合直接主动发给用户时才设为 true。"
                        "如果 shareable=true，body 必须直接写成可以发给用户的自然中文短消息，像顺手分享一个刚看到的有趣发现，"
                        "不要写成汇报、简报、条目摘要或后台备注。"
                        "第二部分是可保存的中文 Markdown 归档正文。"
                        "最终格式必须严格如下：<candidate>{JSON}</candidate>\\n<artifact>Markdown 正文</artifact>。"
                        "如果任务需要实时信息，先用 tool_search 查找并解锁 MCP 工具，再调用相关工具。"
                        "涉及新闻、版本更新、活动、发布日期、价格、天气等时效内容时，必须优先找最近 30 天内的来源，"
                        f"搜索词要包含 {current_year} 或“最新/今天/本周/本月”等时间限定，并在归档中记录来源日期。"
                        "如果只找到旧资料、发布日期不明或结果彼此矛盾，candidate.shareable 必须为 false，只能写后台归档。"
                        "不要把已经过去的发布日期、旧版本更新或过期活动包装成“刚看到的新发现”。"
                        "搜索方向不能只看用户兴趣本身，也要结合近期上下文、待提醒、用户可能受益的实用信息和明确的新变化；"
                        "但只有与用户有实际关系且足够新鲜可靠时才允许主动发。"
                        "不要编造你没有从上下文或工具里获得的事实。"
                    ),
                },
                {"role": "user", "content": f"任务：{task.title}\n要求：{task.prompt}\n\n上下文：\n{context}"},
            ]
            output = await self._run_task(messages)
            if not output:
                await self.store.add_drift_run(task.id, task.title, "drift task failed", error="llm_error")
                await self.store.update_drift_task_state(task.id, "llm_error", last_run_at=started_at, increment_failures=True)
                return DriftResult(ran=False, reason="llm_error")

            candidate_meta, artifact = self._parse_output(task, output)
            candidate_meta = self._apply_evidence_policy(candidate_meta, artifact)
            output_path = self._write_output(task, artifact)
            await self.store.add_drift_run(task.id, task.title, artifact, output_path=str(output_path), error=None)
            await self.store.update_drift_task_state(
                task.id,
                "completed",
                last_run_at=started_at,
                artifact_path=str(output_path),
                artifact_at=utc_now(),
                reset_failures=True,
            )
            candidate = self._build_candidate(task, candidate_meta, output_path) if self.promotion_enabled else None
            logger.info("Drift completed task_id=%s title=%r output=%s", task.id, task.title, output_path)
            return DriftResult(ran=True, reason=None, task_id=task.id, output_path=str(output_path), candidate=candidate)
        except Exception as exc:
            logger.exception("Drift task failed task_id=%s", task.id)
            await self.store.add_drift_run(task.id, task.title, "", error=str(exc))
            await self.store.update_drift_task_state(task.id, "error", last_run_at=started_at, increment_failures=True)
            return DriftResult(ran=False, reason="error", task_id=task.id)

    def _select_task(self, tasks: list[DriftTask], states: dict[str, dict[str, Any]]) -> DriftTask | None:
        """按轮换与失败惩罚选择下一条 drift 任务。"""
        now = utc_now()
        ranked: list[tuple[float, DriftTask]] = []
        for task in tasks:
            state = states.get(task.id, {})
            last_run_at = state.get("last_run_at")
            if last_run_at and now - last_run_at < self.run_cooldown:
                continue
            failure_count = int(state.get("failure_count") or 0)
            last_artifact_at = state.get("last_artifact_at")
            age_score = 10_000.0 if not last_run_at else min((now - last_run_at).total_seconds() / 60.0, 10_000.0)
            artifact_score = 1_000.0 if not last_artifact_at else min((now - last_artifact_at).total_seconds() / 60.0, 1_000.0)
            context_hint = 120.0 if any(token in task.prompt for token in ("记忆", "摘要", "跟进", "用户", "兴趣", "搜索")) else 40.0
            score = age_score + artifact_score + context_hint - (failure_count * 300.0)
            ranked.append((score, task))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1].id), reverse=True)
        return ranked[0][1]

    async def _run_task(self, messages: list[dict[str, Any]]) -> str:
        """执行 drift LLM 调用。"""
        self._last_tool_evidence = {
            "search_succeeded": False,
            "fetch_succeeded": False,
            "search_failed": False,
            "search_empty": False,
            "search_blocked": False,
        }
        if not self.tools:
            result = await self.provider.chat(messages)
            return result.content if result.ok else ""
        tools = _DriftToolRegistry(self.tools, self._last_tool_evidence)
        inbound = InboundMessage(
            channel="proactive",
            chat_id=self.target_chat_id or "drift",
            sender="drift",
            content=str(messages[-1].get("content", "")),
        )
        reasoner = Reasoner(
            provider=self.provider,
            tools=tools,
            max_iterations=self.max_iterations,
            tool_loop_enabled=True,
        )
        result = await reasoner.run(ContextBundle(messages=messages, memory_hits=[], trace={}), inbound)
        return result.reply if not result.error else ""

    async def _build_context(self) -> str:
        """为 drift 任务构造轻量上下文。"""
        chat_id = self.target_chat_id
        now_text = utc_now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        summary = await self.store.get_summary(chat_id) if chat_id else None
        user_profile = await self.store.get_user_profile(chat_id) if chat_id else {}
        memories = await self.store.list_recent_memories(chat_id, limit=8) if chat_id else []
        reminders = await self.store.list_pending_reminders(chat_id, limit=8) if chat_id else []
        lines: list[str] = [f"当前时间：{now_text}"]
        if summary:
            lines.append(f"近期摘要：{summary}")
        interest_hints = build_interest_watchlist(user_profile, memories)
        if interest_hints:
            lines.append("用户兴趣线索：")
            lines.extend(f"- {hint}" for hint in interest_hints)
            lines.append("搜索建议：优先围绕上面的兴趣线索、明确偏好和最近话题去搜索，不要泛泛抓全网热点。")
        if memories:
            lines.append("近期记忆：")
            lines.extend(f"- #{item['id']} [{item['type']}] {item['content']}" for item in memories)
        if reminders:
            lines.append("待提醒：")
            lines.extend(f"- #{item['id']} {item['due_at'].astimezone().strftime('%Y-%m-%d %H:%M:%S')} {item['content']}" for item in reminders)
        return "\n".join(lines) if lines else "暂无足够上下文。"

    def _parse_output(self, task: DriftTask, output: str) -> tuple[dict[str, Any], str]:
        """从 drift 输出中提取候选元数据和归档正文。"""
        candidate_meta: dict[str, Any] = {
            "shareable": False,
            "title": task.title,
            "body": "",
            "priority": 0.45,
            "confidence": 0.55,
            "novelty": 0.35,
            "user_fit": 0.7,
        }
        candidate_start = output.find("<candidate>")
        candidate_end = output.find("</candidate>")
        artifact_start = output.find("<artifact>")
        artifact_end = output.find("</artifact>")
        if candidate_start != -1 and candidate_end != -1:
            raw = output[candidate_start + len("<candidate>"):candidate_end].strip()
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    candidate_meta.update(parsed)
            except json.JSONDecodeError:
                logger.warning("Drift candidate JSON parse failed task_id=%s", task.id)
        artifact = output
        if artifact_start != -1 and artifact_end != -1:
            artifact = output[artifact_start + len("<artifact>"):artifact_end].strip()
        else:
            artifact = output.strip()
        if not artifact:
            artifact = f"# {task.title}\n\n暂无整理结果。"
        if not candidate_meta.get("body"):
            first_line = next((line.strip() for line in artifact.splitlines() if line.strip()), task.title)
            candidate_meta["body"] = first_line[:120]
        candidate_meta["shareable"] = _coerce_bool(candidate_meta.get("shareable", False), default=False)
        if candidate_meta["shareable"] and _looks_stale_for_proactive(f"{candidate_meta.get('title', '')}\n{candidate_meta.get('body', '')}"):
            candidate_meta["shareable"] = False
            candidate_meta["priority"] = min(_coerce_score(candidate_meta.get("priority", 0.45), default=0.45), 0.25)
            candidate_meta["confidence"] = min(_coerce_score(candidate_meta.get("confidence", 0.55), default=0.55), 0.35)
        candidate_meta["priority"] = _coerce_score(candidate_meta.get("priority", 0.45), default=0.45)
        candidate_meta["confidence"] = _coerce_score(candidate_meta.get("confidence", 0.55), default=0.55)
        candidate_meta["novelty"] = _coerce_score(candidate_meta.get("novelty", 0.35), default=0.35)
        candidate_meta["user_fit"] = _coerce_score(candidate_meta.get("user_fit", 0.7), default=0.7)
        return candidate_meta, artifact

    def _apply_evidence_policy(self, candidate_meta: dict[str, Any], artifact: str) -> dict[str, Any]:
        """缺少实时证据时，禁止把时效类 drift 结果直接主动发给用户。"""
        if not _coerce_bool(candidate_meta.get("shareable", False), default=False):
            return candidate_meta
        text = f"{candidate_meta.get('title', '')}\n{candidate_meta.get('body', '')}\n{artifact}"
        if not _looks_time_sensitive_for_proactive(text):
            return candidate_meta
        evidence = self._last_tool_evidence or {}
        has_evidence = bool(evidence.get("search_succeeded") or evidence.get("fetch_succeeded"))
        if has_evidence:
            return candidate_meta
        candidate_meta = dict(candidate_meta)
        candidate_meta["shareable"] = False
        candidate_meta["priority"] = min(_coerce_score(candidate_meta.get("priority", 0.45), default=0.45), 0.25)
        candidate_meta["confidence"] = min(_coerce_score(candidate_meta.get("confidence", 0.55), default=0.55), 0.35)
        logger.info(
            "Drift candidate withheld because time-sensitive evidence is missing or degraded title=%r evidence=%s",
            candidate_meta.get("title", ""),
            evidence,
        )
        return candidate_meta

    def _write_output(self, task: DriftTask, content: str) -> Path:
        """把 drift 结果写入 Markdown 文件。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = utc_now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"{stamp}_{task.id}.md"
        path.write_text(f"# {task.title}\n\n{content}\n", encoding="utf-8")
        return path

    def _build_candidate(self, task: DriftTask, meta: dict[str, Any], output_path: Path) -> ProactiveCandidate | None:
        """把可分享的 drift 元信息提升成主动候选。"""
        shareable = _coerce_bool(meta.get("shareable", False), default=False)
        if not shareable:
            return None
        now = utc_now()
        title = str(meta.get("title") or task.title)
        body = str(meta.get("body") or title).strip()
        dedupe_key = _drift_dedupe_key(task.id, title, body)
        return ProactiveCandidate(
            candidate_id=f"drift:{task.id}:{now.strftime('%Y%m%d%H%M%S')}",
            source_type="drift",
            title=title,
            body=body,
            url="",
            confidence=_coerce_score(meta.get("confidence", 0.55), default=0.55),
            novelty=_coerce_score(meta.get("novelty", 0.35), default=0.35),
            user_fit=_coerce_score(meta.get("user_fit", 0.7), default=0.7),
            priority=_coerce_score(meta.get("priority", 0.45), default=0.45),
            shareable=True,
            created_at=now,
            expires_at=now + timedelta(hours=24),
            dedupe_key=dedupe_key,
            artifact_path=str(output_path),
        )


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """转换布尔值。

    参数:
        value: 参与转换布尔值的 `value` 参数。
        default: 参与转换布尔值的 `default` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in _TRUE_WORDS:
            return True
        if normalized in _FALSE_WORDS:
            return False
    return default


def _coerce_score(value: Any, default: float) -> float:
    """转换`score`。

    参数:
        value: 参与转换`score`的 `value` 参数。
        default: 参与转换`score`的 `default` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return _clamp_score(float(value), default)
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in _SCORE_WORDS:
            return _SCORE_WORDS[normalized]
        try:
            return _clamp_score(float(normalized), default)
        except ValueError:
            return default
    return default


def _clamp_score(value: float, default: float) -> float:
    """限制`score`。

    参数:
        value: 参与限制`score`的 `value` 参数。
        default: 参与限制`score`的 `default` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    if value != value:
        return default
    return max(0.0, min(1.0, value))


class _DriftToolRegistry:
    """给 drift 工具调用加一层轻量时效保护。"""

    def __init__(self, inner: ToolRegistry, evidence: dict[str, Any] | None = None) -> None:
        self.inner = inner
        self.evidence = evidence if evidence is not None else {}

    def get_schema(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        return self.inner.get_schema(names)

    def resolve_visible_names(self, session_tool_names=None) -> list[str]:
        return self.inner.resolve_visible_names(session_tool_names)

    def default_visible_names(self) -> list[str]:
        return self.inner.default_visible_names()

    def get_tool(self, name: str) -> Tool | None:
        return self.inner.get_tool(name)

    def get(self, name: str) -> Tool | None:
        return self.inner.get(name)

    def search(self, query: str, limit: int = 8, exposures: set[str] | None = None) -> list[Tool]:
        return self.inner.search(query, limit=limit, exposures=exposures)

    def tool_names(self) -> list[str]:
        return self.inner.tool_names()

    def list_descriptions(self, only_visible: bool = False, names: list[str] | None = None) -> str:
        return self.inner.list_descriptions(only_visible=only_visible, names=names)

    def visible_count(self) -> int:
        return self.inner.visible_count()

    def visible_names(self) -> list[str]:
        return self.inner.visible_names()

    async def execute(self, name: str, args: dict[str, Any], message: InboundMessage) -> str:
        stale_reason = _stale_search_query_reason(name, args)
        if stale_reason:
            self.evidence["search_blocked"] = True
            return stale_reason
        result = await self.inner.execute(name, args, message)
        self._record_tool_result(name, result)
        return result

    async def call(self, name: str, arguments: dict[str, Any], message: InboundMessage) -> str:
        return await self.execute(name, arguments, message)

    def _record_tool_result(self, name: str, result: str) -> None:
        lowered = name.lower()
        if lowered == "tool_search":
            return
        is_search = "search" in lowered
        is_fetch = "fetch" in lowered or "web_content" in lowered or "rss_get_content" in lowered
        if is_search:
            try:
                payload = json.loads(result)
            except json.JSONDecodeError:
                if "失败" in result or "failed" in result.lower():
                    self.evidence["search_failed"] = True
                elif result.strip():
                    self.evidence["search_succeeded"] = True
                return
            if payload.get("degraded") or payload.get("error"):
                self.evidence["search_failed"] = True
                return
            rows = payload.get("results") if isinstance(payload, dict) else None
            if isinstance(rows, list) and rows:
                self.evidence["search_succeeded"] = True
            elif isinstance(rows, list):
                self.evidence["search_empty"] = True
            return
        if is_fetch:
            if result.strip() and "失败" not in result and "failed" not in result.lower():
                self.evidence["fetch_succeeded"] = True


def _stale_search_query_reason(tool_name: str, args: dict[str, Any], now: datetime | None = None) -> str:
    """识别 drift 中明显旧年份的搜索查询。"""
    lowered_name = tool_name.lower()
    if "search" not in lowered_name:
        return ""
    query = str(args.get("query") or args.get("q") or "").strip()
    if not query:
        return ""
    current_year = (now or utc_now()).year
    years = [int(match) for match in re.findall(r"(?<!\d)(20\d{2})(?!\d)", query)]
    stale_years = sorted({year for year in years if year < current_year})
    if not stale_years:
        return ""
    return (
        f"drift 搜索被拦截：查询 {query!r} 包含旧年份 {', '.join(str(year) for year in stale_years)}。"
        f"当前年份是 {current_year}；请改用 {current_year}、最新、今天、本周、本月或近30天等时间限定重新搜索。"
    )


def _drift_dedupe_key(task_id: str, title: str, body: str) -> str:
    """生成对措辞变化不敏感的 drift 去重 key。"""
    title_text = str(title or "").strip()
    source_text = title_text if len(title_text) >= 4 else body
    topic_text = _normalize_drift_topic(source_text)
    digest = hashlib.sha1(topic_text.encode("utf-8")).hexdigest()[:16]
    return f"drift:{digest}"


def _normalize_drift_topic(text: str) -> str:
    """把主动消息规整成偏主题的文本，减少重复改写绕过去重。"""
    normalized = str(text or "").lower()
    normalized = re.sub(r"https?://\S+", " ", normalized)
    versions = [f"version:{item}" for item in re.findall(r"(?<!\d)(\d+(?:\.\d+)+)(?!\d)", normalized)]
    normalized = re.sub(r"[0-9]+[多余几]?抽|[0-9]+连|[0-9]+星|[0-9]+号|[0-9]+日|[0-9]+月|[0-9]+年", " ", normalized)
    normalized = re.sub(r"[0-9]+(?:\.[0-9]+)?", " ", normalized)
    normalized = re.sub(r"[，。！？、；：,.!?;:（）()【】\[\]「」『』“”\"'~～—_\\/\s]+", " ", normalized)
    stop_words = (
        "刚看到",
        "刚刷到",
        "刚注意到",
        "顺手",
        "等等",
        "嘿",
        "诶",
        "话说",
        "你",
        "我",
        "啦",
        "了",
        "呀",
        "吗",
        "打算",
        "关注",
        "哪个",
        "感觉",
        "就要",
        "而且",
        "还是",
        "好像",
        "挺大的",
        "啥的",
        "明天",
        "今天",
        "昨天",
        "最近",
        "准备",
        "轻量",
        "草稿",
        "提醒",
        "版本",
        "更新",
        "上线",
        "开服",
        "新角色",
        "新地图",
        "活动",
        "兑换码",
        "福利",
        "庆典",
        "庆",
        "跟进",
    )
    for word in stop_words:
        normalized = normalized.replace(word, " ")
    tokens: list[str] = list(versions)
    for raw_token in normalized.split():
        token = raw_token.strip()
        anniversaries = re.findall(r"[一二三四五六七八九十0-9]+周年", token)
        tokens.extend(anniversaries)
        for anniversary in anniversaries:
            token = token.replace(anniversary, " ")
        tokens.extend(part for part in token.split() if len(part) >= 2)
    tokens = [token for token in tokens if len(token) >= 2]
    seen: set[str] = set()
    deduped: list[str] = []
    for token in sorted(tokens):
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return " ".join(deduped[:16]) or "empty"


def _looks_time_sensitive_for_proactive(text: str) -> bool:
    """识别需要实时来源支撑的主动分享内容。"""
    if not text.strip():
        return False
    markers = (
        "最新",
        "刚看到",
        "刚刷到",
        "明天",
        "今天",
        "本周",
        "本月",
        "近30天",
        "新闻",
        "版本",
        "更新",
        "上线",
        "开服",
        "活动",
        "发布",
        "发布日期",
        "直播",
        "兑换码",
        "福利",
        "天气",
        "价格",
        "涨跌",
        "二周年",
        "周年",
    )
    return any(marker in text for marker in markers)


def _looks_stale_for_proactive(text: str, now: datetime | None = None) -> bool:
    """粗略识别不适合主动推送的过期时效内容。"""
    current = now or utc_now()
    if not text.strip():
        return False
    stale_years = {str(year) for year in range(2000, current.year)}
    stale_markers = (
        "已经开了",
        "已经上线",
        "已上线",
        "已开启",
        "已经结束",
        "已结束",
        "去年",
        "过期",
    )
    if any(year in text for year in stale_years) and any(marker in text for marker in stale_markers):
        return True
    old_event_markers = (
        "版本安排",
        "版本更新",
        "新版本",
        "活动",
        "发布日期",
        "更新",
    )
    if any(marker in text for marker in old_event_markers):
        dated_months = [int(month) for month in re.findall(r"(?<!\d)(1[0-2]|[1-9])月(?:\d{1,2}[号日])?", text)]
        if dated_months and all(month < current.month for month in dated_months):
            return True
    return False
