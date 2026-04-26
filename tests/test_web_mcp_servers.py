from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    module_dir = str(path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_web_content_fetch_page_extracts_title_and_body(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("web_content_mcp_server_test", "workspace/mcp_servers/web_content_mcp_server.py")
    monkeypatch.setattr(
        module,
        "http_get",
        lambda url: "<html><head><title>Example Title</title></head><body>Hello <b>world</b></body></html>",
    )

    result = module.fetch_page("https://example.com/page")

    assert result["url"] == "https://example.com/page"
    assert result["title"] == "Example Title"
    assert result["content"] == "Example Title Hello world"


def test_web_content_fetch_page_rejects_local_file_url() -> None:
    module = _load_module("web_content_mcp_server_security_test", "workspace/mcp_servers/web_content_mcp_server.py")

    with pytest.raises(ValueError, match="http:// and https://"):
        module.fetch_page("file:///tmp/secret.txt")


def test_feed_bridge_poll_get_and_ack(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("feed_bridge_mcp_server_test", "workspace/mcp_servers/feed_bridge_mcp_server.py")
    module.EVENTS.clear()
    module.ACKED.clear()
    monkeypatch.setenv("WEB_FEEDS", '[{"name":"Example Feed","url":"https://example.com/rss.xml"}]')
    monkeypatch.setattr(
        module,
        "http_get",
        lambda url, limit=800_000: (
            "<?xml version='1.0' encoding='UTF-8'?>"
            "<rss><channel><item><guid>abc</guid><title>Fresh Item</title>"
            "<link>https://example.com/a</link><description>Body</description>"
            "<pubDate>Tue, 22 Apr 2026 12:00:00 GMT</pubDate></item></channel></rss>"
        ),
    )

    polled = module.poll_feeds()
    fetched = module.get_proactive_events()

    assert polled["ok"] is True
    assert polled["events"] == 1
    assert fetched["events"][0]["title"] == "Fresh Item"
    assert fetched["events"][0]["url"] == "https://example.com/a"

    acked = module.ack_events([fetched["events"][0]["event_id"]])

    assert acked["acked"] == 1
    assert module.get_proactive_events()["events"] == []


def test_feed_bridge_extracts_image_from_enclosure(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("feed_bridge_mcp_server_image_test", "workspace/mcp_servers/feed_bridge_mcp_server.py")
    module.EVENTS.clear()
    module.ACKED.clear()
    monkeypatch.setenv("WEB_FEEDS", '[{"name":"Example Feed","url":"https://example.com/rss.xml"}]')
    monkeypatch.setattr(
        module,
        "http_get",
        lambda url, limit=800_000: (
            "<?xml version='1.0' encoding='UTF-8'?>"
            "<rss><channel><item><guid>abc</guid><title>Fresh Item</title>"
            "<link>https://example.com/a</link><description>Body</description>"
            '<enclosure url="https://example.com/a.jpg" type="image/jpeg" />'
            "<pubDate>Tue, 22 Apr 2026 12:00:00 GMT</pubDate></item></channel></rss>"
        ),
    )

    fetched = module.poll_feeds()
    event = module.get_proactive_events()["events"][0]

    assert fetched["ok"] is True
    assert event["image_url"] == "https://example.com/a.jpg"
