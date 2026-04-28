from __future__ import annotations

import pytest

from chat_agent.channels.qq import QQBotAPI, QQBotChannel, clean_qq_content, sign_qq_payload, split_qq_chat_id, verify_qq_signature
from chat_agent.config import QQBotConfig
from chat_agent.messages import OutboundAttachment, OutboundMessage


async def _handler(message):
    return OutboundMessage(channel=message.channel, chat_id=message.chat_id, content="ok")


def test_split_qq_chat_id() -> None:
    assert split_qq_chat_id("c2c:user-openid") == ("c2c", "user-openid")
    assert split_qq_chat_id("group:group-openid") == ("group", "group-openid")
    assert split_qq_chat_id("channel-id") == ("channel", "channel-id")


def test_clean_qq_content_removes_mentions() -> None:
    assert clean_qq_content("<@!123456> 你好") == "你好"


def test_clean_qq_content_removes_face_markup() -> None:
    assert clean_qq_content('<faceType=6,faceId="0",ext="eyJ0ZXh0IjoiIn0=">') == ""
    assert clean_qq_content('<@!123456> <faceType=6,faceId="0"> hello') == "hello"


def test_qq_signature_round_trip() -> None:
    body = b'{"op":0}'
    timestamp = "1710000000"
    signature = sign_qq_payload("secret", timestamp, body)

    assert verify_qq_signature(
        "secret",
        {"X-Signature-Ed25519": signature, "X-Signature-Timestamp": timestamp},
        body,
    )
    assert not verify_qq_signature(
        "secret",
        {"X-Signature-Ed25519": signature, "X-Signature-Timestamp": timestamp},
        b'{"op":1}',
    )


@pytest.mark.asyncio
async def test_qq_group_event_to_inbound_message() -> None:
    channel = QQBotChannel(
        QQBotConfig(app_id="app-id", app_secret="secret", verify_signature=False),
        _handler,
    )
    inbound = await channel._build_inbound(
        {
            "op": 0,
            "t": "GROUP_AT_MESSAGE_CREATE",
            "id": "event-id",
            "d": {
                "id": "msg-id",
                "group_openid": "group-openid",
                "op_member_openid": "member-openid",
                "content": "<@!123456> 你好",
            },
        }
    )

    assert inbound is not None
    assert inbound.channel == "qq"
    assert inbound.chat_id == "group:group-openid"
    assert inbound.sender == "member-openid"
    assert inbound.content == "你好"
    assert inbound.message_id == "msg-id"


@pytest.mark.asyncio
async def test_qq_channel_uploads_local_meme_attachment(tmp_path) -> None:
    meme = tmp_path / "meme.png"
    meme.write_bytes(b"fake image")

    class FakeAPI:
        def __init__(self) -> None:
            self.sent_files = []
            self.sent_texts = []

        async def send_image_file(self, chat_id, file_path, caption="", reply_to=None, event_id=None):
            self.sent_files.append(
                {
                    "chat_id": chat_id,
                    "file_path": file_path,
                    "caption": caption,
                    "reply_to": reply_to,
                    "event_id": event_id,
                }
            )

        async def send_text(self, chat_id, content, reply_to=None, event_id=None):
            self.sent_texts.append(
                {
                    "chat_id": chat_id,
                    "content": content,
                    "reply_to": reply_to,
                    "event_id": event_id,
                }
            )

    channel = QQBotChannel(
        QQBotConfig(app_id="app-id", app_secret="secret", verify_signature=False),
        _handler,
    )
    fake_api = FakeAPI()
    channel.api = fake_api

    await channel.send(
        OutboundMessage(
            channel="qq",
            chat_id="group:group-openid",
            content="配图",
            attachments=[OutboundAttachment(kind="photo", local_path=str(meme))],
            reply_to_message_id="msg-id",
            metadata={"qq_event_id": "event-id"},
        )
    )

    assert fake_api.sent_files == [
        {
            "chat_id": "group:group-openid",
            "file_path": str(meme),
            "caption": "",
            "reply_to": None,
            "event_id": None,
        }
    ]
    assert fake_api.sent_texts == [
        {
            "chat_id": "group:group-openid",
            "content": "配图",
            "reply_to": "msg-id",
            "event_id": "event-id",
        }
    ]


@pytest.mark.asyncio
async def test_qq_api_sends_local_image_as_base64_file_data(tmp_path) -> None:
    image = tmp_path / "meme.png"
    image.write_bytes(b"fake image")

    class FakeAPI(QQBotAPI):
        def __init__(self) -> None:
            super().__init__(QQBotConfig(app_id="app-id", app_secret="secret", verify_signature=False))
            self.requests = []

        async def request(self, method, path, body):
            self.requests.append((method, path, body))
            return {}

    api = FakeAPI()

    await api.send_image_file("c2c:user-openid", image, caption="配图", reply_to="msg-id", event_id="event-id")

    assert len(api.requests) == 1
    method, path, body = api.requests[0]
    assert method == "POST"
    assert path == "/v2/users/user-openid/files"
    assert body["file_type"] == 1
    assert body["file_data"] == "ZmFrZSBpbWFnZQ=="
    assert body["srv_send_msg"] is True
    assert body["content"] == "配图"
    assert body["msg_id"] == "msg-id"
    assert body["event_id"] == "event-id"
