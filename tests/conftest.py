"""Synthetic Telegram Desktop export. Shaped after tdesktop's export_output_json.cpp; contains no real chat data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

BASE = 1_760_000_000  # unix seconds


def _msg(mid: int, offset_min: int, sender: str, uid: int, text, **extra) -> dict:
    ts = BASE + offset_min * 60
    parts = text if isinstance(text, list) else [text]
    entities = [{"type": "plain", "text": p} if isinstance(p, str) else p for p in parts]
    return {
        "id": mid,
        "type": "message",
        "date": "2025-10-09T10:00:00",
        "date_unixtime": str(ts),
        "from": sender,
        "from_id": f"user{uid}",
        "text": text,
        "text_entities": entities,
        **extra,
    }


def synthetic_export() -> dict:
    return {
        "name": "Hiking Club",
        "type": "private_supergroup",
        "id": 1234567890,
        "messages": [
            {
                "id": 1,
                "type": "service",
                "date": "2025-10-09T09:00:00",
                "date_unixtime": str(BASE - 3600),
                "actor": "Alice Moreau",
                "actor_id": "user1",
                "action": "invite_members",
                "members": ["Bram Okafor"],
                "text": "",
                "text_entities": [],
            },
            _msg(2, 0, "Alice Moreau", 1, "Saturday hike: Mount Tam loop, meet 8am at the trailhead?"),
            _msg(3, 2, "Bram Okafor", 2, "I'm in. Alice can you share the route?", reply_to_message_id=2),
            _msg(
                4,
                5,
                "Alice Moreau",
                1,
                [
                    "Here: ",
                    {"type": "text_link", "text": "route map", "href": "https://example.org/route/42"},
                    " — call me on +1 415 555 0134 if lost",
                ],
            ),
            _msg(5, 6, "Bram Okafor", 2, "", photo="chats/chat_01/photos/photo_1@09-10-2025_10-06-00.jpg"),
            _msg(
                6,
                7,
                "Chen Wei",
                3,
                "",
                file="(File not included. Change data exporting settings to download.)",
                media_type="voice_message",
                mime_type="audio/ogg",
                duration_seconds=12,
            ),
            _msg(7, 8, "Chen Wei", 3, "", media_type="sticker", sticker_emoji="👍", file="stickers/s.webp"),
            _msg(
                8,
                300,
                "Bram Okafor",
                2,
                "",
                poll={
                    "question": "Bring a stove?",
                    "closed": False,
                    "total_voters": 3,
                    "answers": [
                        {"text": "Yes", "voters": 2, "chosen": False},
                        {"text": "No", "voters": 1, "chosen": True},
                    ],
                },
            ),
            _msg(
                9,
                302,
                "Chen Wei",
                3,
                "Decision: Bram brings the stove, I bring water filters. ping @chen_hikes",
                reactions=[
                    {
                        "type": "emoji",
                        "count": 3,
                        "emoji": "👍",
                        "recent": [{"from": "Alice Moreau", "from_id": "user1"}],
                    },
                    {"type": "custom_emoji", "count": 1, "document_id": "5368324170671202286"},
                    {"type": "paid", "count": 2},
                ],
            ),
        ],
    }


@pytest.fixture
def export_dir(tmp_path: Path) -> Path:
    d = tmp_path / "ChatExport_test"
    d.mkdir()
    (d / "result.json").write_text(json.dumps(synthetic_export(), ensure_ascii=False), encoding="utf-8")
    return d


@pytest.fixture
def fake_llm(monkeypatch):
    """Route every model call through a Pydantic AI FunctionModel.

    Usage: `seen = fake_llm(respond)`, where `respond(user_prompt)` returns the response text. `seen` collects
    the repr of every request, so tests can assert on exactly what a provider would receive.
    """
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.models.function import FunctionModel

    from tg_summary import llm

    def install(respond):
        seen: list[str] = []

        def fn(messages, info):
            seen.append(repr(messages))
            parts = [p for m in messages if isinstance(m, ModelRequest) for p in m.parts]
            user = next(p.content for p in parts if isinstance(p, UserPromptPart))
            return ModelResponse(parts=[TextPart(respond(user))])

        monkeypatch.setattr(llm, "build_model", lambda role, client: FunctionModel(fn))
        return seen

    return install
