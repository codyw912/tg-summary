"""One SQLite file per chat: messages, LLM response cache, redaction vault, hosted-call audit log, and the text
derived from media and links."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tg_summary.models import Chat, Media, Message

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    date INTEGER NOT NULL,
    sender_id TEXT,
    sender TEXT,
    text TEXT NOT NULL,
    reply_to INTEGER,
    forwarded_from TEXT,
    edited INTEGER,
    media TEXT,
    service_action TEXT,
    reactions TEXT
);
CREATE INDEX IF NOT EXISTS messages_date ON messages (date);
CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, response TEXT NOT NULL, created INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS vault (
    kind TEXT NOT NULL,
    n INTEGER NOT NULL,
    norm TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (kind, n),
    UNIQUE (kind, norm)
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    endpoint TEXT NOT NULL,
    model TEXT NOT NULL,
    request TEXT NOT NULL,
    response TEXT NOT NULL,
    unmatched TEXT NOT NULL
);
-- Text derived from one media file or link by one role. `item` is "sha256:<hex>" for a media file and
-- "url:<normalized url>" for a link, so a repost or a link shared again is the same item.
CREATE TABLE IF NOT EXISTS derivations (
    item TEXT NOT NULL,
    role TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    text TEXT,
    model TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    error TEXT,
    evidence TEXT,
    created INTEGER NOT NULL,
    updated INTEGER NOT NULL,
    PRIMARY KEY (item, role)
);
CREATE TABLE IF NOT EXISTS message_items (
    msg_id INTEGER NOT NULL,
    item TEXT NOT NULL,
    PRIMARY KEY (msg_id, item)
);
-- Digests written by `tg-summary-eval compare`, with the exact judge prompts (claim plus cited messages as the run
-- saw them), so `tg-summary-eval judge` can grade them later, with one model loaded, on the same evidence.
CREATE TABLE IF NOT EXISTS compare_digests (
    run_id TEXT PRIMARY KEY,
    created INTEGER NOT NULL,
    variant TEXT NOT NULL,
    digest TEXT NOT NULL,
    judge_prompts TEXT
);
"""

# Derivations with these statuses carry usable text and are reused by later runs.
FINAL_STATUSES = ("done", "opened", "unverified")
# Link failures that will not change on a retry with the same endpoint, model and settings, so they are reused
# too. `content_too_large` counts: a different `max_content_tokens` in `extra_body` changes the fingerprint.
# Everything else that failed (`url_not_accessible`, `too_many_requests`, `unavailable`, `max_uses_exceeded`,
# `url_not_in_prior_context`, HTTP errors, truncation) and `over_limit` is retried next run.
PERMANENT_ERRORS = ("url_not_allowed", "url_too_long", "invalid_tool_input", "content_too_large")


def _final_sql(row: str) -> str:
    statuses = ", ".join(f"'{s}'" for s in FINAL_STATUSES)
    errors = ", ".join(f"'{e}'" for e in PERMANENT_ERRORS)
    return f"({row}.status IN ({statuses}) OR ({row}.status = 'failed' AND {row}.error IN ({errors})))"


_COLUMNS = "id, date, sender_id, sender, text, reply_to, forwarded_from, edited, media, service_action, reactions"


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Map calls run on worker threads; one connection guarded by a lock keeps writes serialized.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript(SCHEMA)
            columns = {row[1] for row in self._db.execute("PRAGMA table_info(messages)")}
            if "reactions" not in columns:  # databases from before reactions were stored; re-ingest fills them
                self._db.execute("ALTER TABLE messages ADD COLUMN reactions TEXT")
            columns = {row[1] for row in self._db.execute("PRAGMA table_info(compare_digests)")}
            if "judge_prompts" not in columns:  # digests saved before their evidence was; those stay unjudged
                self._db.execute("ALTER TABLE compare_digests ADD COLUMN judge_prompts TEXT")

    def close(self) -> None:
        self._db.close()

    # --- chat identity -------------------------------------------------

    def bind_chat(self, chat: Chat, source: str, media_root: Path | None) -> None:
        existing = self.chat_or_none()
        if existing is not None and existing.id != chat.id:
            raise ValueError(f"database holds chat {existing.id} ({existing.name}), not {chat.id} ({chat.name})")
        meta = {
            "chat_id": str(chat.id),
            "chat_name": chat.name,
            "chat_type": chat.type,
            "source": source,
            "media_root": str(media_root) if media_root else "",
        }
        with self._lock, self._db:
            self._db.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", meta.items())

    def media_root(self) -> Path | None:
        """Directory that `Media.path` is relative to, as stored at ingest."""
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key = 'media_root'").fetchone()
        return Path(row[0]) if row and row[0] else None

    def chat_or_none(self) -> Chat | None:
        with self._lock:
            meta = dict(self._db.execute("SELECT key, value FROM meta"))
        if "chat_id" not in meta:
            return None
        return Chat(id=int(meta["chat_id"]), name=meta["chat_name"], type=meta["chat_type"])

    def chat(self) -> Chat:
        chat = self.chat_or_none()
        if chat is None:
            raise ValueError("database is empty; run `tg-summary ingest` first")
        return chat

    # --- messages ------------------------------------------------------

    def upsert_messages(self, messages: Iterable[Message]) -> int:
        rows = [_to_row(m) for m in messages]
        with self._lock, self._db:
            self._db.executemany(
                f"INSERT OR REPLACE INTO messages ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
        return len(rows)

    def select(
        self, since: datetime | None = None, until: datetime | None = None, last: int | None = None
    ) -> list[Message]:
        """Regular (non-service) messages in [since, until), keeping only the newest `last`. Oldest first."""
        where, args = _window(since, until)
        sql = f"SELECT {_COLUMNS} FROM messages WHERE service_action IS NULL{where} ORDER BY date DESC, id DESC"
        if last is not None:
            sql += " LIMIT ?"
            args.append(last)
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [_from_row(r) for r in reversed(rows)]

    def service_counts(self, since: datetime, until: datetime) -> dict[str, int]:
        with self._lock:
            rows = self._db.execute(
                "SELECT service_action, COUNT(*) FROM messages "
                "WHERE service_action IS NOT NULL AND date >= ? AND date <= ? GROUP BY service_action",
                (_ts(since), _ts(until)),
            ).fetchall()
        return dict(rows)

    def get_messages(self, ids: Iterable[int]) -> dict[int, Message]:
        ids = list(set(ids))
        if not ids:
            return {}
        out: dict[int, Message] = {}
        with self._lock:
            for i in range(0, len(ids), 500):
                part = ids[i : i + 500]
                q = f"SELECT {_COLUMNS} FROM messages WHERE id IN ({','.join('?' * len(part))})"
                out.update((r[0], _from_row(r)) for r in self._db.execute(q, part))
        return out

    def sender_names(self) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT DISTINCT sender FROM messages WHERE sender IS NOT NULL "
                "UNION SELECT DISTINCT forwarded_from FROM messages WHERE forwarded_from IS NOT NULL"
            ).fetchall()
        return [r[0] for r in rows]

    # --- LLM cache -----------------------------------------------------

    def cache_get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT response FROM llm_cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def cache_put(self, key: str, response: str) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?)", (key, response, int(time.time())))

    # --- redaction vault -----------------------------------------------

    def vault_values(self) -> list[tuple[str, str, str]]:
        """(kind, norm, value) of every value redacted so far; `norm` is the identity it was numbered by."""
        with self._lock:
            return self._db.execute("SELECT kind, norm, value FROM vault").fetchall()

    def vault_assign(self, kind: str, norm: str, value: str) -> int:
        """Stable number for (kind, normalized value); assigns the next free number on first sight."""
        with self._lock, self._db:
            row = self._db.execute("SELECT n FROM vault WHERE kind = ? AND norm = ?", (kind, norm)).fetchone()
            if row:
                return row[0]
            (n,) = self._db.execute("SELECT COALESCE(MAX(n), 0) + 1 FROM vault WHERE kind = ?", (kind,)).fetchone()
            self._db.execute("INSERT INTO vault VALUES (?, ?, ?, ?)", (kind, n, norm, value))
            return n

    # --- compare digests -----------------------------------------------

    def save_digest(self, run_id: str, variant: str, digest_json: str, judge_prompts: list[str]) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO compare_digests (run_id, created, variant, digest, judge_prompts) VALUES (?, ?, ?, ?, ?)",
                (run_id, int(time.time()), variant, digest_json, json.dumps(judge_prompts, ensure_ascii=False)),
            )

    def load_judge_prompts(self, run_id: str) -> list[str] | None:
        """The judge prompts saved with a run's digest; None if the run wasn't saved or predates saved prompts."""
        with self._lock:
            row = self._db.execute("SELECT judge_prompts FROM compare_digests WHERE run_id = ?", (run_id,)).fetchone()
        return json.loads(row[0]) if row and row[0] is not None else None

    # --- hosted-call audit ---------------------------------------------

    def audit(self, endpoint: str, model: str, request: str, response: str, unmatched: list[str]) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO audit (ts, endpoint, model, request, response, unmatched) VALUES (?, ?, ?, ?, ?, ?)",
                (int(time.time()), endpoint, model, request, response, json.dumps(unmatched)),
            )

    # --- media and link derivations ------------------------------------

    def derivation(self, item: str, role: str) -> Derivation | None:
        with self._lock:
            row = self._db.execute(
                "SELECT item, role, kind, status, text, model, fingerprint, error, evidence "
                "FROM derivations WHERE item = ? AND role = ?",
                (item, role),
            ).fetchone()
        return Derivation(*row[:8], evidence=json.loads(row[8]) if row[8] else None) if row else None

    def put_derivation(self, d: Derivation) -> None:
        """Record the latest outcome for (item, role). A final result is never replaced by a non-final one."""
        now = int(time.time())
        final_new, final_old = _final_sql("excluded"), _final_sql("derivations")
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO derivations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (item, role) DO UPDATE SET kind = excluded.kind, status = excluded.status, "
                "text = excluded.text, model = excluded.model, fingerprint = excluded.fingerprint, "
                "error = excluded.error, evidence = excluded.evidence, updated = excluded.updated "
                f"WHERE {final_new} OR NOT {final_old}",
                (
                    d.item,
                    d.role,
                    d.kind,
                    d.status,
                    d.text,
                    d.model,
                    d.fingerprint,
                    d.error,
                    json.dumps(d.evidence, ensure_ascii=False) if d.evidence is not None else None,
                    now,
                    now,
                ),
            )

    def link_items(self, pairs: Iterable[tuple[int, str]]) -> None:
        """Remember which items each message carries (message id, item key)."""
        with self._lock, self._db:
            self._db.executemany("INSERT OR IGNORE INTO message_items VALUES (?, ?)", pairs)

    def message_items(self, msg_id: int) -> list[str]:
        with self._lock:
            rows = self._db.execute("SELECT item FROM message_items WHERE msg_id = ? ORDER BY item", (msg_id,))
            return [r[0] for r in rows]


@dataclass(frozen=True, slots=True)
class Derivation:
    item: str
    role: str
    kind: str  # media kind, or "link"
    status: str  # done | opened | unverified | failed | over_limit
    text: str | None  # description, transcript or link summary; None unless `usable`
    model: str
    fingerprint: str  # endpoint, model and prompt; a different one means the item is derived again
    error: str | None = None  # short code, e.g. "url_not_allowed", "http_429", "truncated"
    evidence: dict | None = None  # links: the fetches the provider reported, and citation count

    @property
    def usable(self) -> bool:
        """Carries text that replaces the placeholder."""
        return self.status in FINAL_STATUSES

    @property
    def final(self) -> bool:
        """Reused by later runs (while the fingerprint matches) instead of being derived again."""
        return self.usable or (self.status == "failed" and self.error in PERMANENT_ERRORS)


def _ts(d: datetime) -> int:
    return int(d.timestamp())


def _window(since: datetime | None, until: datetime | None) -> tuple[str, list]:
    where, args = "", []
    if since is not None:
        where += " AND date >= ?"
        args.append(_ts(since))
    if until is not None:
        where += " AND date < ?"
        args.append(_ts(until))
    return where, args


def _to_row(m: Message) -> tuple:
    return (
        m.id,
        _ts(m.date),
        m.sender_id,
        m.sender,
        m.text,
        m.reply_to,
        m.forwarded_from,
        _ts(m.edited) if m.edited else None,
        json.dumps(m.media.to_dict(), ensure_ascii=False) if m.media else None,
        m.service_action,
        json.dumps(m.reactions, ensure_ascii=False) if m.reactions else None,
    )


def _from_row(r: tuple) -> Message:
    return Message(
        id=r[0],
        date=datetime.fromtimestamp(r[1], tz=UTC),
        sender_id=r[2],
        sender=r[3],
        text=r[4],
        reply_to=r[5],
        forwarded_from=r[6],
        edited=datetime.fromtimestamp(r[7], tz=UTC) if r[7] is not None else None,
        media=Media(**json.loads(r[8])) if r[8] else None,
        service_action=r[9],
        reactions=tuple((label, n) for label, n in json.loads(r[10])) if r[10] else (),
    )
