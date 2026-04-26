from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chat_agent.agent.provider import make_multimodal_user_content
from chat_agent.channels.telegram import TelegramChannel
from chat_agent.config import TelegramConfig


def test_provider_multimodal_content_format() -> None:
    content = make_multimodal_user_content("看看这张图", ["https://example.test/image.jpg"])

    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "看看这张图"}
    assert content[1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_telegram_photo_to_attachment(tmp_path: Path, monkeypatch) -> None:
    async def handler(message):
        raise AssertionError("not used")

    channel = TelegramChannel(
        TelegramConfig(token="123:abc", allow_from=[], unauthorized_reply=True, download_images=False, image_max_mb=10),
        handler,
    )

    class FakeBot:
        async def get_file(self, file_id):
            return SimpleNamespace(file_path="https://api.telegram.org/file/bot123/photo.jpg")

    photo = SimpleNamespace(file_id="file-1", file_unique_id="unique-1", file_size=100)
    update = SimpleNamespace(message=SimpleNamespace(photo=[photo]))
    context = SimpleNamespace(bot=FakeBot())

    attachments = await channel._build_attachments(update, context)

    assert len(attachments) == 1
    assert attachments[0].kind == "image"
    assert attachments[0].url == "https://api.telegram.org/file/bot123/photo.jpg"
