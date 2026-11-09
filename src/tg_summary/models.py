"""Source-independent records. Every source adapter produces these; nothing downstream knows where they came from."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

PRIVATE_LINKABLE = frozenset({"private_supergroup", "private_channel"})


@dataclass(frozen=True, slots=True)
class Chat:
    id: int  # bare peer id, as Telegram Desktop exports it
    name: str
    type: str  # export chat type, e.g. "private_supergroup"

    def message_link(self, msg_id: int) -> str | None:
        if self.type in PRIVATE_LINKABLE:
            return f"https://t.me/c/{self.id}/{msg_id}"
        return None


@dataclass(frozen=True, slots=True)
class Media:
    # photo, video, video_message, voice, audio, animation, sticker, document,
    # poll, todo, location, contact, rich, other
    kind: str
    path: str | None = None  # relative to the source's media root; None when not downloaded
    mime: str | None = None
    file_name: str | None = None
    duration: int | None = None
    detail: str | None = None  # sticker emoji, rendered poll, place name, ...

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True, slots=True)
class Message:
    id: int
    date: datetime  # timezone-aware UTC
    sender_id: str | None
    sender: str | None
    text: str
    reply_to: int | None = None
    forwarded_from: str | None = None
    edited: datetime | None = None
    media: Media | None = None
    service_action: str | None = None  # set only for service events (joins, pins, ...)
    # (label, count) per reaction: the emoji itself, "custom emoji" or "⭐" for paid star reactions.
    reactions: tuple[tuple[str, int], ...] = ()

    @property
    def reaction_total(self) -> int:
        return sum(n for _, n in self.reactions)
