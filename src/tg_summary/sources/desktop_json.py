"""Telegram Desktop single-chat JSON export ("Export chat history" -> JSON).

Format reference: https://core.telegram.org/import-export and tdesktop's
Telegram/SourceFiles/export/output/export_output_json.cpp.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tg_summary.models import Chat, Media, Message

_MEDIA_TYPES = {
    "sticker": "sticker",
    "video_message": "video_message",
    "voice_message": "voice",
    "animation": "animation",
    "video_file": "video",
    "audio_file": "audio",
}


class DesktopJsonSource:
    name = "desktop_json"

    def __init__(self, path: Path):
        path = path.expanduser()
        self._file = path / "result.json" if path.is_dir() else path
        with self._file.open(encoding="utf-8") as fh:
            self._data: dict[str, Any] = json.load(fh)
        if "messages" not in self._data:
            raise ValueError(
                f"{self._file} is not a single-chat export. In Telegram Desktop, open the group, "
                "choose ⋯ → Export chat history, and pick the JSON format."
            )

    def chat(self) -> Chat:
        return Chat(id=int(self._data["id"]), name=self._data.get("name") or "", type=self._data["type"])

    def media_root(self) -> Path:
        return self._file.parent

    def messages(self) -> Iterator[Message]:
        for raw in self._data["messages"]:
            if raw.get("type") in ("message", "service"):
                yield parse_message(raw)


def parse_message(raw: dict[str, Any]) -> Message:
    service = raw["type"] == "service"
    return Message(
        id=int(raw["id"]),
        date=_unix(raw["date_unixtime"]),
        sender_id=raw.get("actor_id" if service else "from_id"),
        sender=raw.get("actor" if service else "from"),
        text=render_text(raw.get("text_entities"), raw.get("text")),
        reply_to=raw.get("reply_to_message_id") if "reply_to_peer_id" not in raw else None,
        forwarded_from=raw.get("forwarded_from"),
        edited=_unix(raw["edited_unixtime"]) if "edited_unixtime" in raw else None,
        media=None if service else parse_media(raw),
        service_action=raw.get("action") if service else None,
        reactions=parse_reactions(raw.get("reactions")),
    )


_REACTION_LABELS = {"custom_emoji": "custom emoji", "paid": "⭐"}


def parse_reactions(raw: list[dict] | None) -> tuple[tuple[str, int], ...]:
    """(label, count) per reaction type: the emoji itself, "custom emoji" or "⭐" for paid reactions."""
    out = []
    for r in raw or []:
        label = r.get("emoji") if r.get("type") == "emoji" else _REACTION_LABELS.get(r.get("type", ""), "other")
        count = int(r.get("count", 0))
        if label and count > 0:
            out.append((label, count))
    return tuple(out)


def render_text(entities: list | None, text: str | list | None) -> str:
    """Plain text with link targets kept, since links carry meaning for summaries.

    `text_entities` holds only objects; the legacy `text` field mixes bare strings and objects.
    """
    parts = entities if entities is not None else text
    if not isinstance(parts, list):
        return parts or ""
    out = []
    for ent in parts:
        if isinstance(ent, str):
            out.append(ent)
            continue
        part = ent.get("text", "")
        href = ent.get("href")
        if ent.get("type") == "text_link" and href and href != part:
            part = f"{part} ({href})"
        out.append(part)
    return "".join(out)


def parse_media(raw: dict[str, Any]) -> Media | None:
    if "photo" in raw:
        return Media(kind="photo", path=_path(raw["photo"]))
    if "file" in raw or "media_type" in raw:
        kind = _MEDIA_TYPES.get(raw.get("media_type", ""), "document")
        detail = None
        if kind == "sticker":
            detail = raw.get("sticker_emoji")
        elif kind == "audio":
            detail = " - ".join(p for p in (raw.get("performer"), raw.get("title")) if p) or None
        return Media(
            kind=kind,
            path=_path(raw.get("file")),
            mime=raw.get("mime_type"),
            file_name=raw.get("file_name"),
            duration=raw.get("duration_seconds"),
            detail=detail,
        )
    if "poll" in raw:
        poll = raw["poll"]
        answers = ", ".join(
            f"{render_text(None, a.get('text'))} ({a.get('voters', 0)})" for a in poll.get("answers", [])
        )
        return Media(kind="poll", detail=f"{render_text(None, poll.get('question'))} — {answers}")
    if "todo_list" in raw:
        todo = raw["todo_list"]
        items = "; ".join(render_text(None, i.get("text")) for i in todo.get("answers", []))
        return Media(kind="todo", detail=f"{render_text(None, todo.get('title'))}: {items}")
    if "location_information" in raw or "place_name" in raw:
        detail = ", ".join(p for p in (raw.get("place_name"), raw.get("address")) if p) or None
        return Media(kind="location", detail=detail)
    if "contact_information" in raw:
        info = raw["contact_information"]
        name = " ".join(p for p in (info.get("first_name"), info.get("last_name")) if p)
        return Media(kind="contact", detail=name or None)
    if "rich_message" in raw:
        return Media(kind="rich", detail=" ".join(_strings_under(raw["rich_message"], "text")) or None)
    if any(k in raw for k in ("invoice_information", "game_title", "giveaway_information", "paid_stars_amount")):
        return Media(kind="other")
    return None


def _path(value: str | None) -> str | None:
    # Skipped files are written as "(File not included. ...)" instead of a path.
    if not value or value.startswith("("):
        return None
    # Exports write plain relative paths ("photos/photo_1@...jpg"). Anything that could point outside the export
    # (absolute, "..", backslashes, drive letters) is treated as a file that wasn't included.
    parts = value.split("/")
    if value.startswith("/") or "\\" in value or ":" in parts[0] or ".." in parts:
        return None
    return value


def _unix(value: str | int) -> datetime:
    return datetime.fromtimestamp(int(value), tz=UTC)


def _strings_under(node: Any, key: str) -> Iterator[str]:
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key and isinstance(v, str):
                yield v
            else:
                yield from _strings_under(v, key)
    elif isinstance(node, list):
        for v in node:
            yield from _strings_under(v, key)
