from __future__ import annotations

from pathlib import Path

import pytest

from chat_agent.channels.telegram import is_allowed_username
from chat_agent.config import TelegramConfig
from chat_agent.messages import OutboundAttachment, OutboundMessage


def test_allow_from_empty_allows_anyone() -> None:
    config = TelegramConfig(token="token", allow_from=[], unauthorized_reply=True)

    assert is_allowed_username(config, None)
    assert is_allowed_username(config, "alice")


def test_allow_from_requires_matching_username() -> None:
    config = TelegramConfig(token="token", allow_from=["alice"], unauthorized_reply=True)

    assert is_allowed_username(config, "alice")
    assert is_allowed_username(config, "@alice")
    assert not is_allowed_username(config, "bob")
    assert not is_allowed_username(config, None)


@pytest.mark.asyncio
async def test_send_photo_attachment_uses_send_photo(tmp_path: Path) -> None:
    from chat_agent.channels.telegram import TelegramChannel

    class FakeBot:
        def __init__(self) -> None:
            self.photos = []
            self.messages = []

        async def send_photo(self, **kwargs):
            self.photos.append(kwargs)

        async def send_message(self, **kwargs):
            self.messages.append(kwargs)

    channel = TelegramChannel(
        TelegramConfig(token="token", allow_from=[], unauthorized_reply=True),
        handler=lambda message: None,  # type: ignore[arg-type]
    )
    channel.application = type("App", (), {"bot": FakeBot()})()

    meme_path = tmp_path / "meme.png"
    meme_path.write_bytes(b"fake-image")

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="chat-1",
            content="收好这张",
            attachments=[OutboundAttachment(kind="photo", local_path=str(meme_path))],
        )
    )

    assert len(channel.application.bot.photos) == 1
    assert channel.application.bot.photos[0]["caption"] == "收好这张"
    assert channel.application.bot.messages == []
