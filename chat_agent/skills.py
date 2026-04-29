"""Akashic 风格 SKILL.md 技能说明书系统。

skills 与 tools 不同：tools 是 Agent 能执行的动作，skills 是 Agent 做事前阅读的 SOP。
本模块负责扫描内置 skills、workspace skills、drift skills，解析 front matter，
检查依赖可用性，并为 ContextBuilder 提供 catalog 和 active skill 正文。
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


@dataclass(slots=True)
class SkillRecord:
    """扫描到的一份 SKILL.md 元数据。

    字段:
        name: skill 名称，必须是小写字母/数字/连字符。
        description: skill 简介，用于 catalog 中帮助模型判断何时使用。
        path: SKILL.md 的实际路径。
        source: 来源，通常是 builtin 或 workspace；workspace 同名会覆盖 builtin。
        metadata: front matter 中的 metadata 字典。
        available: 依赖检查是否通过。
        missing_bins: 缺失的本机命令列表。
        missing_env: 缺失的环境变量列表。
        missing_tools: 缺失的工具列表。
    """

    name: str
    description: str
    path: Path
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    available: bool = True
    missing_bins: list[str] = field(default_factory=list)
    missing_env: list[str] = field(default_factory=list)
    missing_tools: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """把 SkillRecord 转换成可序列化字典，供命令、工具和测试使用。"""
        return {
            "name": self.name,
            "description": self.description,
            "path": str(self.path),
            "source": self.source,
            "metadata": self.metadata,
            "available": self.available,
            "missing_bins": self.missing_bins,
            "missing_env": self.missing_env,
            "missing_tools": self.missing_tools,
        }


class SkillsLoader:
    """Akashic 风格 SKILL.md 技能说明书加载器。

    skills 与 tools 的区别:
        tools 是 Agent 可以执行的动作；skills 是 Agent 做事前阅读的说明书/SOP。SkillsLoader
        负责扫描内置目录和 workspace 目录，生成 catalog，并按需读取完整 SKILL.md 注入 prompt。
    """

    def __init__(
        self,
        workspace: Path,
        builtin_skills_dir: Path | None = None,
        max_catalog_chars: int = 4000,
    ) -> None:
        """初始化 SkillsLoader。

        参数:
            workspace: workspace skills 根目录，用户创建/更新的 skill 会写到这里。
            builtin_skills_dir: 项目内置 skills 目录；为空时只扫描 workspace。
            max_catalog_chars: 注入 prompt 的 skills catalog 最大字符数，超出会裁剪。
        """
        self.workspace = Path(workspace)
        self.builtin_skills_dir = Path(builtin_skills_dir) if builtin_skills_dir else None
        self.max_catalog_chars = max_catalog_chars

    def list_skills(self, filter_unavailable: bool = True, available_tools: set[str] | None = None) -> list[dict[str, Any]]:
        """列出所有已发现的 skill。

        参数:
            filter_unavailable: True 时过滤掉缺失 bin/env 依赖的 skill。
            available_tools: 可选的已注册工具名集合；传入时也检查 requires.tools。

        返回:
            skill 字典列表，按名称排序。
        """
        records = self._scan(available_tools=available_tools)
        if filter_unavailable:
            records = {name: record for name, record in records.items() if record.available}
        return [record.as_dict() for record in sorted(records.values(), key=lambda item: item.name)]

    def load_skill(self, name: str, available_tools: set[str] | None = None) -> str | None:
        """读取指定 skill 的完整 SKILL.md 正文。

        参数:
            name: skill 名称。
            available_tools: 可选的已注册工具名集合；传入时也检查 requires.tools。

        返回:
            文件完整内容；名称非法、未找到或依赖不可用时返回 None。
        """
        if not is_valid_skill_name(name):
            return None
        record = self._scan(available_tools=available_tools).get(name)
        if not record or not record.available:
            return None
        return record.path.read_text(encoding="utf-8")

    def load_skills_for_context(self, names: list[str], available_tools: set[str] | None = None) -> str:
        """读取多个 skill 并包装成 prompt 中的 XML 块。

        参数:
            names: 需要注入完整说明的 skill 名称列表。
            available_tools: 可选的已注册工具名集合；传入时也检查 requires.tools。

        返回:
            多个 <skill name="...">...</skill> 块拼接后的字符串。
        """
        blocks: list[str] = []
        seen: set[str] = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            body = self.load_skill(name, available_tools=available_tools)
            if body:
                blocks.append(f'<skill name="{name}">\n{body}\n</skill>')
        return "\n\n".join(blocks)

    def get_skill_metadata(self, name: str, available_tools: set[str] | None = None) -> dict[str, Any] | None:
        """返回指定 skill 的 metadata。"""
        record = self._scan(available_tools=available_tools).get(name)
        return record.metadata if record else None

    def get_always_skills(self, available_tools: set[str] | None = None) -> list[str]:
        """返回 metadata.chat_agent.always=true 且可用的 skill 名称。"""
        result: list[str] = []
        for record in self._scan(available_tools=available_tools).values():
            if record.available and bool(_deep_get(record.metadata, ["chat_agent", "always"], False)):
                result.append(record.name)
        return sorted(result)

    def build_skills_summary(self, available_tools: set[str] | None = None) -> str:
        """构建给模型看的 skills catalog 摘要。

        返回:
            XML 风格文本，包含名称、描述、路径、可用状态和缺失依赖。只注入摘要，不注入所有
            完整 SKILL.md，以避免 prompt 膨胀。
        """
        records = sorted(self._scan(available_tools=available_tools).values(), key=lambda item: (item.source != "workspace", item.name))
        lines = ["<skills>"]
        for record in records:
            missing = ",".join(record.missing_bins + record.missing_env + record.missing_tools)
            triggers = _skill_triggers(record)
            lines.extend(
                [
                    f'  <skill available="{str(record.available).lower()}" source="{record.source}">',
                    f"    <name>{_xml_escape(record.name)}</name>",
                    f"    <description>{_xml_escape(record.description)}</description>",
                    f"    <location>{_xml_escape(_display_path(record.path))}</location>",
                ]
            )
            if triggers:
                lines.append(f"    <triggers>{_xml_escape(','.join(triggers[:8]))}</triggers>")
            if missing:
                lines.append(f"    <missing>{_xml_escape(missing)}</missing>")
            lines.append("  </skill>")
        lines.append("</skills>")
        summary = "\n".join(lines)
        if len(summary) > self.max_catalog_chars:
            return summary[: self.max_catalog_chars - 20] + "\n<!-- truncated -->"
        return summary

    def extract_triggered_skills(self, text: str, available_tools: set[str] | None = None) -> list[dict[str, str]]:
        """从用户文本中提取触发的 skill，并返回触发原因。

        触发方式:
            @skill-name、skill:skill-name、文本中独立出现 skill name，或命中 metadata.chat_agent.triggers。
        """
        text_lower = text.lower()
        reasons: dict[str, str] = {}
        records = self._scan(available_tools=available_tools)
        for match in re.finditer(r"(?:@|skill:)([a-z0-9][a-z0-9-]{0,63})", text_lower):
            name = match.group(1)
            record = records.get(name)
            if record and record.available:
                reasons.setdefault(name, f"mention: {name}")
        for name, record in records.items():
            if not record.available:
                continue
            if re.search(rf"(?<![a-z0-9-]){re.escape(name)}(?![a-z0-9-])", text_lower):
                reasons.setdefault(name, f"name: {name}")
            for trigger in _skill_triggers(record):
                trigger_lower = trigger.lower()
                if trigger_lower and trigger_lower in text_lower:
                    reasons.setdefault(name, f"trigger: {trigger}")
                    break
        return [{"name": name, "reason": reasons[name]} for name in sorted(reasons)]

    def extract_triggered_skill_names(self, text: str, available_tools: set[str] | None = None) -> list[str]:
        """从用户文本中提取显式触发的 skill 名称。

        触发方式:
            @skill-name、skill:skill-name、文本中独立出现 skill name，或命中 metadata.chat_agent.triggers。
        """
        return [item["name"] for item in self.extract_triggered_skills(text, available_tools=available_tools)]

    def workspace_skill_path(self, name: str) -> Path:
        """计算 workspace 中某个 skill 的 SKILL.md 路径并做安全校验。

        参数:
            name: skill 名称，只允许小写字母、数字和连字符。

        返回:
            workspace/<name>/SKILL.md 的绝对路径。

        异常:
            名称非法或路径逃逸 workspace 时抛 ValueError。
        """
        if not is_valid_skill_name(name):
            raise ValueError("skill name must contain only lowercase letters, numbers, and hyphens")
        path = (self.workspace / name / "SKILL.md").resolve()
        root = self.workspace.resolve()
        if root not in path.parents:
            raise ValueError("skill path escapes workspace")
        return path

    def write_workspace_skill(self, name: str, description: str, body: str, always: bool = False) -> Path:
        """创建或覆盖 workspace skill。

        参数:
            name: skill 名称。
            description: skill 简介。
            body: SKILL.md 正文内容。
            always: 是否每轮都注入完整 skill；默认 False，避免 prompt 过长。
        """
        path = self.workspace_skill_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"chat_agent": {"always": bool(always), "drift": False, "triggers": [], "requires": {"bins": [], "env": [], "tools": []}}}
        content = (
            "---\n"
            f"name: {name}\n"
            f"description: {description.strip()}\n"
            f"metadata: {json.dumps(metadata, ensure_ascii=False)}\n"
            "---\n\n"
            f"{body.strip()}\n"
        )
        path.write_text(content, encoding="utf-8")
        return path

    def update_workspace_skill(self, name: str, body: str) -> Path:
        """更新 workspace 中已存在 skill 的正文。

        参数:
            name: skill 名称。
            body: 新正文。front matter 会保留原文件中的 name/description/metadata。
        """
        path = self.workspace_skill_path(name)
        if not path.exists():
            raise FileNotFoundError(f"workspace skill not found: {name}")
        front, _ = _parse_skill_file(path)
        content = _format_front_matter(front) + "\n\n" + body.strip() + "\n"
        path.write_text(content, encoding="utf-8")
        return path

    def _scan(self, available_tools: set[str] | None = None) -> dict[str, SkillRecord]:
        """扫描内置与 workspace skills，并应用 workspace 覆盖规则。"""
        records: dict[str, SkillRecord] = {}
        if self.builtin_skills_dir:
            records.update(self._scan_dir(self.builtin_skills_dir, "builtin", available_tools=available_tools))
        # 后扫描 workspace，让同名用户 skill 覆盖内置 skill。
        records.update(self._scan_dir(self.workspace, "workspace", available_tools=available_tools))
        return records

    def _scan_dir(self, root: Path, source: str, available_tools: set[str] | None = None) -> dict[str, SkillRecord]:
        """扫描单个 skills 根目录。

        参数:
            root: skills 根目录。
            source: 来源标记，通常是 builtin 或 workspace。
            available_tools: 可选的已注册工具名集合；传入时也检查 requires.tools。
        """
        if not root.exists():
            return {}
        records: dict[str, SkillRecord] = {}
        for path in root.glob("*/SKILL.md"):
            try:
                front, _ = _parse_skill_file(path)
                name = str(front.get("name") or path.parent.name).strip()
                if not is_valid_skill_name(name):
                    continue
                metadata = front.get("metadata") if isinstance(front.get("metadata"), dict) else {}
                record = SkillRecord(
                    name=name,
                    description=str(front.get("description") or "").strip(),
                    path=path.resolve(),
                    source=source,
                    metadata=metadata,
                )
                self._apply_requirements(record, available_tools=available_tools)
                records[name] = record
            except Exception:
                continue
        return records

    def _apply_requirements(self, record: SkillRecord, available_tools: set[str] | None = None) -> None:
        """根据 metadata.chat_agent.requires 检查 skill 依赖。"""
        requires = _deep_get(record.metadata, ["chat_agent", "requires"], {}) or {}
        bins = requires.get("bins", []) if isinstance(requires, dict) else []
        envs = requires.get("env", []) if isinstance(requires, dict) else []
        tools = requires.get("tools", []) if isinstance(requires, dict) else []
        record.missing_bins = [str(item) for item in bins if shutil.which(str(item)) is None]
        import os

        record.missing_env = [str(item) for item in envs if not os.environ.get(str(item))]
        record.missing_tools = []
        if available_tools is not None:
            record.missing_tools = [str(item) for item in tools if str(item) not in available_tools]
        record.available = not record.missing_bins and not record.missing_env and not record.missing_tools


def is_valid_skill_name(name: str) -> bool:
    """校验 skill 名称是否安全。"""
    return bool(SKILL_NAME_PATTERN.fullmatch(name.strip()))


def _parse_skill_file(path: Path) -> tuple[dict[str, Any], str]:
    """解析 SKILL.md 的简易 front matter。

    参数:
        path: SKILL.md 文件路径。

    返回:
        (front, body)。metadata 字段按 JSON 解析，其他字段按简单 key:value 解析。
    """
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {"name": path.parent.name, "description": "", "metadata": {}}, text
    end = text.find("\n---", 4)
    if end < 0:
        return {"name": path.parent.name, "description": "", "metadata": {}}, text
    raw_front = text[4:end].strip()
    body = text[end + len("\n---") :].lstrip()
    front: dict[str, Any] = {}
    for line in raw_front.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "metadata":
            try:
                front[key] = json.loads(value)
            except json.JSONDecodeError:
                front[key] = {}
        else:
            front[key] = value.strip('"').strip("'")
    front.setdefault("name", path.parent.name)
    front.setdefault("description", "")
    front.setdefault("metadata", {})
    return front, body


def _format_front_matter(front: dict[str, Any]) -> str:
    """把解析出的 front matter 字典格式化回 SKILL.md 头部。"""
    lines = ["---"]
    for key in ("name", "description"):
        if key in front:
            lines.append(f"{key}: {front[key]}")
    metadata = front.get("metadata", {})
    lines.append(f"metadata: {json.dumps(metadata, ensure_ascii=False)}")
    lines.append("---")
    return "\n".join(lines)


def _deep_get(data: dict[str, Any], path: list[str], default: Any = None) -> Any:
    """安全读取嵌套字典路径。"""
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _skill_triggers(record: SkillRecord) -> list[str]:
    """读取 metadata.chat_agent.triggers 并规整成字符串列表。"""
    triggers = _deep_get(record.metadata, ["chat_agent", "triggers"], []) or []
    if not isinstance(triggers, list):
        return []
    return [str(item).strip() for item in triggers if str(item).strip()]


def _xml_escape(value: str) -> str:
    """转义 XML 文本中的基础特殊字符。"""
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _display_path(path: Path) -> str:
    """优先把路径显示为相对当前工作目录的形式。"""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)
