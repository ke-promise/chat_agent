"""日志初始化与敏感信息脱敏。

本模块统一配置控制台/文件日志，并对 Telegram token、API key 等敏感片段做脱敏。
项目启动时由 main.py 调用 setup_logging，其他模块只需要通过 logging.getLogger 使用日志。
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chat_agent.config import LoggingConfig


class SecretFilter(logging.Filter):
    """日志敏感信息过滤器。

    过滤范围:
        - Telegram Bot API URL 中的 /bot<id>:<token>。
        - 常见的 api_key=...、token=... 查询参数或日志片段。

    说明:
        过滤器会处理 record.msg 和 record.args，避免格式化前后的敏感字符串泄漏到控制台或文件。
    """

    TELEGRAM_BOT_TOKEN_RE = re.compile(r"/bot(\d+):([A-Za-z0-9_-]+)")
    GENERIC_SECRET_RE = re.compile(r"(?i)(api[_-]?key|token)=([^&\s]+)")

    def filter(self, record: logging.LogRecord) -> bool:
        """对一条 LogRecord 做脱敏处理。

        参数:
            record: logging 框架传入的日志记录。

        返回:
            始终 True，表示该日志仍应继续输出，只是内容已脱敏。
        """
        if isinstance(record.msg, str):
            record.msg = self._redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {key: self._redact_arg(value) for key, value in record.args.items()}
            else:
                record.args = tuple(self._redact_arg(value) for value in record.args)
        return True

    def _redact_arg(self, value):
        """脱敏单个日志参数；非字符串参数原样返回。"""
        if isinstance(value, str):
            return self._redact(value)
        return value

    def _redact(self, value: str) -> str:
        """对字符串执行具体脱敏规则。"""
        value = self.TELEGRAM_BOT_TOKEN_RE.sub(r"/bot\1:<redacted>", value)
        value = self.GENERIC_SECRET_RE.sub(r"\1=<redacted>", value)
        return value


def setup_logging(config: "LoggingConfig | None" = None) -> None:
    """配置项目日志。

    参数:
        config: LoggingConfig；为空时使用 INFO 控制台日志。

    行为:
        - 清空 root logger 旧 handler，避免重复输出。
        - 始终输出到 stdout。
        - 配置了 file 时额外写入 logs/app.log。
        - 为所有 handler 添加 SecretFilter。
        - 降低 httpx/httpcore/telegram/openai 的日志级别，避免轮询噪音和 token URL 泄漏。
    """
    level_name = config.level if config else "INFO"
    level = getattr(logging, level_name.upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if config and config.file:
        log_path = Path(config.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        handler.addFilter(SecretFilter())
        root.addHandler(handler)

    # python-telegram-bot polls getUpdates through httpx. At INFO level httpx logs
    # every polling request, which is noisy and can include Telegram bot tokens in URLs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
