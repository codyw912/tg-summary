"""Ingest seam. Telegram Desktop JSON exports and (later) the Telegram API both implement `Source`."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from tg_summary.models import Chat, Message


class Source(Protocol):
    name: str  # stored with the data, e.g. "desktop_json"

    def chat(self) -> Chat: ...

    def messages(self) -> Iterator[Message]: ...

    def media_root(self) -> Path | None:
        """Directory that `Media.path` is relative to, if media files are available locally."""
        ...
