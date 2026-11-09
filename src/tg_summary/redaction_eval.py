"""Compare redaction detectors on a real chat window, for choosing which to enable.

Runs each statistical detector (Presidio, PII-Tracer) on its own over the chunk text a hosted summarizer would
get, next to the always-on member-name and URL/email/handle rules. Detectors are called directly, so nothing
is redacted, sent or added to the vault.

The terminal gets counts only. The report file holds chat text (spans in their lines, for hand-labelling) and
belongs somewhere private and git-ignored; `reports/` is the default.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from tg_summary.config import Config
from tg_summary.models import Message
from tg_summary.redact import Detector, PiiTracerDetector, PresidioDetector, RosterDetector, Span, rule_detector
from tg_summary.segment import Derived, build_chunks
from tg_summary.store import Store
from tg_summary.summarize import budget_for

# Tool and product names this kind of chat mentions constantly and that must stay readable. A hit on one is
# very likely a false positive. Override with --keep.
DEFAULT_KEEP = (
    "Claude, Anthropic, ChatGPT, OpenAI, GPT, Codex, Cursor, Copilot, Gemini, Google, Qwen, Llama, Mistral, "
    "DeepSeek, Ollama, LM Studio, oMLX, MLX, Splash, vLLM, Hugging Face, GitHub, Python, Rust, macOS, Linux, "
    "Windows, Telegram, Discord, Slack"
)


@dataclass
class _Tally:
    spans: int = 0
    kinds: Counter[str] = field(default_factory=Counter)
    member_hits: int = 0
    keep_hits: list[tuple[str, str]] = field(default_factory=list)  # (span text, its line)
    only_this: list[tuple[str, str, str]] = field(default_factory=list)  # (kind, note, its line)


def run_redaction_eval(
    args: argparse.Namespace, config: Config, store: Store, messages: list[Message], derived: dict[int, Derived] | None
) -> int:
    settings = config.redaction
    allow = {a.casefold() for a in settings.allow}
    keep = [k.strip() for k in (args.keep or DEFAULT_KEEP).split(",") if k.strip()]
    keep_re = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(k) for k in keep) + r")(?!\w)", re.IGNORECASE)

    roster = RosterDetector(store.sender_names(), settings.allow)
    statistical: dict[str, Detector] = {}
    for name in args.detectors.split(","):
        name = name.strip()
        if name == "presidio":
            statistical[name] = PresidioDetector(settings)
        elif name == "pii_tracer":
            statistical[name] = PiiTracerDetector(settings)
        elif name:
            raise ValueError(f"unknown detector {name!r}: use presidio and/or pii_tracer")

    context = store.get_messages(m.reply_to for m in messages if m.reply_to is not None)
    context.update({m.id: m for m in messages})
    gap = timedelta(minutes=config.summarize_settings.gap_minutes)
    chunks = build_chunks(
        messages, context, budget_for(config.summarize, config.summarize_settings.chunk_tokens), gap, derived
    )
    _log(f"{len(messages)} messages in {len(chunks)} chunk(s); detectors: {', '.join(statistical)}")

    tallies = {name: _Tally() for name in statistical}
    members = 0
    for i, chunk in enumerate(chunks, 1):
        text = chunk.text

        def kept(spans: Iterable[Span]) -> list[Span]:
            return [s for s in spans if s.value.casefold() not in allow and s.end > s.start]

        names = kept(roster(text))
        rules = kept(rule_detector(text))
        members += len(names)
        found = {name: kept(detect(text)) for name, detect in statistical.items()}
        for name, spans in found.items():
            t = tallies[name]
            others = [s for other, ss in found.items() if other != name for s in ss] + names + rules
            t.spans += len(spans)
            t.kinds.update(s.kind for s in spans)
            t.member_hits += sum(covers(spans, n) for n in names)
            for s in spans:
                excerpt = text[s.start : s.end]
                if keep_re.search(excerpt):
                    t.keep_hits.append((excerpt, _line(text, s)))
                if not covers(others, s):
                    partly = any(o.start < s.end and s.start < o.end for o in others)
                    t.only_this.append((s.kind, "partly covered by others" if partly else "", _line(text, s)))
        _log(f"chunk {i}/{len(chunks)} checked")

    _write_report(args.out, args, len(messages), len(chunks), members, tallies)
    print(_summary_table(members, tallies))
    _log(f"wrote {args.out} (contains chat text: keep it private)")
    return 0


def covers(spans: Iterable[Span], span: Span) -> bool:
    """Whether `spans` together cover every character of `span`. Overlap alone isn't coverage: finding "Alice"
    in "Alice Moreau" leaves the surname exposed."""
    reach = span.start
    for s in sorted((s for s in spans if s.start < span.end and span.start < s.end), key=lambda s: s.start):
        if s.start > reach:
            return False  # a gap before this span
        reach = max(reach, s.end)
        if reach >= span.end:
            return True
    return reach >= span.end


def _line(text: str, span: Span) -> str:
    """The line holding the span, with the span marked as >>span<<."""
    start = text.rfind("\n", 0, span.start) + 1
    end = text.find("\n", span.end)
    end = len(text) if end == -1 else end
    return f"{text[start : span.start]}>>{text[span.start : span.end]}<<{text[span.end : end]}"


def _summary_table(members: int, tallies: dict[str, _Tally]) -> str:
    rows = [
        "| Detector | Spans | Member names fully caught | Keep-list hits | Not fully covered by others |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, t in tallies.items():
        recall = f"{t.member_hits}/{members} ({100 * t.member_hits / members:.0f}%)" if members else "-"
        rows.append(f"| {name} | {t.spans} | {recall} | {len(t.keep_hits)} | {len(t.only_this)} |")
    return "\n".join(rows)


def _write_report(
    out: Path, args: argparse.Namespace, n_messages: int, n_chunks: int, members: int, tallies: dict[str, _Tally]
) -> None:
    rng = random.Random(0)
    parts = [
        "# Redaction detector comparison",
        "",
        "**Private: this file contains chat text. Keep it local and out of git.**",
        "",
        (
            f"{n_messages} messages in {n_chunks} chunk(s), the text a hosted summarizer would receive. "
            "Member-name and URL/email/handle rules always run; this compares the statistical detectors on top."
        ),
        "",
        _summary_table(members, tallies),
        "",
        (
            "- **Member names fully caught**: mentions of group members (known from the member list) whose every "
            "character the detector covered on its own; finding only a first name doesn't count. A rough measure "
            "of how well it finds names in this chat."
        ),
        "- **Keep-list hits**: spans containing a tool or product name that should stay readable. Likely false positives.",
        (
            "- **Not fully covered by others**: spans with at least one character neither the other detector nor "
            "the rules cover. A sample of these follows for hand-labelling."
        ),
        "",
        "Spans by kind: "
        + "; ".join(
            f"{name}: " + ", ".join(f"{k} {n}" for k, n in t.kinds.most_common()) for name, t in tallies.items()
        ),
        "",
    ]
    for name, t in tallies.items():
        parts += [f"## {name}: keep-list hits ({len(t.keep_hits)})", ""]
        parts += [f"- `{_inert(line)}`" for _, line in t.keep_hits[: args.sample]] or ["(none)"]
        parts += [""]
        sample = rng.sample(t.only_this, min(args.sample, len(t.only_this)))
        parts += [
            f"## {name}: not fully covered by any other detector or rule ({len(t.only_this)}; {len(sample)} sampled)",
            "",
            "Mark each: is the marked text something that should be hidden from a hosted model? Spans marked",
            '"partly covered" overlap another detector\'s find but reach further, e.g. a surname after a first name.',
            "",
        ]
        parts += [
            f"- [ ] PII  [ ] not PII  ({kind}{', ' + note if note else ''}) `{_inert(line)}`"
            for kind, note, line in sample
        ] or ["(none)"]
        parts += [""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")


def _inert(line: str) -> str:
    return line.replace("`", "'")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
