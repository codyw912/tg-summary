"""Turn messages into model-ready text and pack them into chunks that fit a token budget."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta

from tg_summary.models import Media, Message

SNIPPET_CHARS = 80
CHARS_PER_TOKEN = 3.5  # deliberately pessimistic; servers don't expose tokenizers uniformly


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


@dataclass(frozen=True, slots=True)
class Derived:
    """Text the media and link stage produced for one message."""

    media: str | None = None  # description and/or transcript of the message's media file
    links: tuple[str, ...] = ()  # rendered link notes, e.g. "[link: example.org/x — a route map]"


def media_placeholder(media: Media, derived: str | None = None) -> str:
    """`[voice 0:12]`, or with derived text `[voice 0:12: transcript "…"]`."""
    base = _placeholder(media)
    if not derived:
        return base
    if not (base.startswith("[") and base.endswith("]")):
        base = f"[{media.kind} {base}]"  # a sticker renders as its bare emoji
    return f"{base[:-1]}: {derived}]"


def _placeholder(media: Media) -> str:
    dur = f" {media.duration // 60}:{media.duration % 60:02d}" if media.duration else ""
    match media.kind:
        case "sticker":
            return media.detail or "[sticker]"
        case "photo":
            return "[photo]"
        case "video":
            return f"[video{dur}]"
        case "video_message":
            return f"[video note{dur}]"
        case "voice":
            return f"[voice{dur}]"
        case "animation":
            return "[GIF]"
        case "audio":
            return f"[audio{dur}: {media.detail}]" if media.detail else f"[audio{dur}]"
        case "document":
            return f"[file: {media.file_name}]" if media.file_name else "[file]"
        case "poll":
            return f"[poll: {media.detail}]"
        case "todo":
            return f"[checklist: {media.detail}]"
        case "location":
            return f"[location: {media.detail}]" if media.detail else "[location]"
        case "contact":
            return f"[contact: {media.detail}]" if media.detail else "[contact]"
        case "rich":
            return media.detail or "[rich message]"
        case _:
            return f"[{media.kind}]"


def snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= SNIPPET_CHARS else flat[: SNIPPET_CHARS - 1] + "…"


def render_message(
    msg: Message, in_chunk: set[int], context: dict[int, Message], derived: Mapping[int, Derived] | None = None
) -> str:
    """`#id YYYY-MM-DD HH:MM Sender ↪#reply [fwd: X]: body`. Replies outside the chunk get a quoted snippet.

    `derived` holds the media and link stage's text per message id; it replaces bare media placeholders."""
    return _render_line(msg, in_chunk, context, derived)[0]


def _render_line(
    msg: Message, in_chunk: set[int], context: dict[int, Message], derived: Mapping[int, Derived] | None
) -> tuple[str, list[tuple[int, int]]]:
    """The line, and where in it this function wrote format rather than chat text: the "#id YYYY-MM-DD HH:MM"
    stamp and the "↪#id" reply reference. Those offsets, never a pattern match, tell redaction what to skip."""
    derived = derived or {}
    stamp = f"#{msg.id} {msg.date.astimezone():%Y-%m-%d %H:%M}"
    fmt = [(0, len(stamp))]
    head = f"{stamp} {msg.sender or 'Unknown'}"
    if msg.reply_to is not None:
        ref = f"↪#{msg.reply_to}"
        fmt.append((len(head) + 1, len(head) + 1 + len(ref)))
        head += f" {ref}"
        target = context.get(msg.reply_to)
        if msg.reply_to not in in_chunk and target is not None:
            head += f' ({target.sender or "Unknown"}: "{snippet(_body(target, derived.get(target.id)))}")'
    if msg.forwarded_from:
        head += f" [fwd: {msg.forwarded_from}]"
    body = _body(msg, derived.get(msg.id)).replace("\n", "\n    ")
    return f"{head}: {body}", fmt


def _body(msg: Message, derived: Derived | None = None) -> str:
    parts = [msg.text] if msg.text else []
    if msg.media is not None:
        parts.append(media_placeholder(msg.media, derived.media if derived else None))
    if derived:
        parts.extend(derived.links)
    return " ".join(parts)


@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    messages: list[Message]
    text: str
    # (start, end) of the text the renderer added as format (message stamps, reply references, gap markers),
    # which never holds PII; redaction skips exactly these offsets and nothing that merely looks like them.
    format: tuple[tuple[int, int], ...] = ()

    @property
    def ids(self) -> set[int]:
        return {m.id for m in self.messages}


def build_chunks(
    messages: list[Message],
    context: dict[int, Message],
    budget_tokens: int,
    gap: timedelta,
    derived: Mapping[int, Derived] | None = None,
) -> list[Chunk]:
    """Split at the best quiet gap until every piece fits, then pack neighbouring pieces into chunks.

    Splitting at quiet gaps that no reply crosses keeps conversations together; packing keeps the number of
    model calls low. Quiet gaps longer than `gap` are marked inside a chunk so the model sees the boundary.
    """
    if not messages:
        return []
    # Worst case per line: assume every reply needs its snippet. +2 covers the newline and gap markers.
    cost = _costs(messages, context, derived)
    pos = {m.id: i for i, m in enumerate(messages)}
    pieces = _split(0, len(messages), messages, cost, budget_tokens, pos)

    groups: list[tuple[int, int]] = []
    for start, end in pieces:
        if groups and sum(cost[groups[-1][0] : end]) <= budget_tokens:
            groups[-1] = (groups[-1][0], end)
        else:
            groups.append((start, end))
    return [
        _render(index, messages[start:end], context, gap, derived, budget_tokens)
        for index, (start, end) in enumerate(groups)
    ]


def split_chunk(
    chunk: Chunk, context: dict[int, Message], gap: timedelta, derived: Mapping[int, Derived] | None = None
) -> list[Chunk]:
    """The chunk in two parts, cut in its middle half at the quietest point that breaks the fewest reply chains.

    For a chunk whose notes didn't fit the output limit. Replies that end up across the cut get a quoted
    snippet, as at any chunk boundary."""
    msgs = chunk.messages
    cost = _costs(msgs, context, derived)
    cut = _best_cut(0, len(msgs), msgs, cost, {m.id: i for i, m in enumerate(msgs)}, balance=0.25)
    budget = sum(cost)  # lines were already fitted to the run's budget; this keeps them whole
    return [
        _render(chunk.index, msgs[:cut], context, gap, derived, budget),
        _render(chunk.index, msgs[cut:], context, gap, derived, budget),
    ]


def _costs(messages: list[Message], context: dict[int, Message], derived: Mapping[int, Derived] | None) -> list[int]:
    return [estimate_tokens(render_message(m, set(), context, derived)) + 2 for m in messages]


def _render(
    index: int,
    msgs: list[Message],
    context: dict[int, Message],
    gap: timedelta,
    derived: Mapping[int, Derived] | None,
    budget: int,
) -> Chunk:
    ids = {m.id for m in msgs}
    lines: list[str] = []
    fmt: list[tuple[int, int]] = []
    pos = 0

    def add(line: str, ranges: list[tuple[int, int]]) -> None:
        nonlocal pos
        fmt.extend((pos + a, pos + min(b, len(line))) for a, b in ranges if a < len(line))  # _fit may cut the end
        lines.append(line)
        pos += len(line) + 1

    for prev, msg in zip([None, *msgs], msgs, strict=False):
        if prev is not None and msg.date - prev.date > gap:
            marker = f"--- {_human(msg.date - prev.date)} later ---"
            add(marker, [(0, len(marker))])
        line, ranges = _render_line(msg, ids, context, derived)
        add(_fit(line, budget), ranges)
    return Chunk(index=index, messages=msgs, text="\n".join(lines), format=tuple(fmt))


def _split(
    start: int, end: int, messages: list[Message], cost: list[int], budget: int, pos: dict[int, int]
) -> list[tuple[int, int]]:
    if end - start <= 1 or sum(cost[start:end]) <= budget:
        return [(start, end)]
    cut = _best_cut(start, end, messages, cost, pos)
    return _split(start, cut, messages, cost, budget, pos) + _split(cut, end, messages, cost, budget, pos)


def _best_cut(
    start: int, end: int, messages: list[Message], cost: list[int], pos: dict[int, int], balance: float = 0.0
) -> int:
    """Index of the message to cut before: the longest silence, discounted by the replies that would cross it
    (score = silence / (1 + crossing replies)). With `balance`, each side keeps at least that share of the cost."""
    crossing = [0] * (end - start + 1)  # difference array over cut positions start..end
    for j in range(start, end):
        p = pos.get(messages[j].reply_to) if messages[j].reply_to is not None else None
        if p is not None and start <= p < j:  # a reply to an earlier message crosses every cut in (p, j]
            crossing[p + 1 - start] += 1
            crossing[j + 1 - start] -= 1
    for k in range(1, len(crossing)):
        crossing[k] += crossing[k - 1]

    total = sum(cost[start:end])
    before = 0
    candidates = []
    for i in range(start + 1, end):
        before += cost[i - 1]
        if before >= total * balance and total - before >= total * balance:
            candidates.append(i)

    def score(i: int) -> float:
        return (messages[i].date - messages[i - 1].date).total_seconds() / (1 + crossing[i - start])

    return max(candidates or range(start + 1, end), key=score)


def _fit(line: str, budget: int) -> str:
    """A single message larger than the whole budget is truncated rather than dropped."""
    max_chars = int(budget * CHARS_PER_TOKEN)
    return line if len(line) <= max_chars else line[: max_chars - 15] + " …[truncated]"


def _human(delta: timedelta) -> str:
    hours = delta.total_seconds() / 3600
    return f"{hours:.0f}h" if hours < 48 else f"{hours / 24:.0f}d"
