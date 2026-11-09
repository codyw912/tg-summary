"""Compare redaction detectors on a real chat window, for choosing which to enable.

Runs each statistical detector (Presidio, PII-Tracer) on its own over the chunk text a hosted summarizer would
get, next to the always-on member-name and URL/email/handle rules. Detectors are called directly, so nothing
is redacted, sent or added to the vault.

The terminal gets counts only. The report is a self-contained HTML page for hand-labelling: each span in its
line, buttons (or number keys) to label it, live tallies, and a box with the counts alone to copy. It holds chat
text and belongs somewhere private and git-ignored; `reports/` is the default. Labels are kept in the
browser's local storage for that page.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from tg_summary.redact import (
    Detector,
    PiiTracerDetector,
    PresidioDetector,
    RosterDetector,
    Span,
    outside_format,
    rule_detector,
)
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
    keep_hits: list[tuple[str, str, str]] = field(default_factory=list)  # (before, span, after) on its line
    only_this: list[tuple[str, str, str, str, str]] = field(default_factory=list)  # (kind, note, before, span, after)


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
        names = _kept(text, roster(text), allow, chunk.format)
        rules = _kept(text, rule_detector(text), allow, chunk.format)
        members += len(names)
        found = {name: _kept(text, detect(text), allow, chunk.format) for name, detect in statistical.items()}
        for name, spans in found.items():
            t = tallies[name]
            others = [s for other, ss in found.items() if other != name for s in ss] + names + rules
            t.spans += len(spans)
            t.kinds.update(s.kind for s in spans)
            t.member_hits += sum(covers(spans, n) for n in names)
            for s in spans:
                excerpt = text[s.start : s.end]
                if keep_re.search(excerpt):
                    t.keep_hits.append(_line(text, s))
                if not covers(others, s):
                    partly = any(o.start < s.end and s.start < o.end for o in others)
                    t.only_this.append((s.kind, "partly covered by others" if partly else "", *_line(text, s)))
        _log(f"chunk {i}/{len(chunks)} checked")

    _write_report(args.out, args, len(messages), len(chunks), members, tallies)
    print(_summary_table(members, tallies))
    _log(f"wrote {args.out} (contains chat text: keep it private)")
    return 0


def _kept(text: str, spans: Iterable[Span], allow: set[str], protected: Iterable[tuple[int, int]]) -> list[Span]:
    """What redaction would act on: the renderer's format offsets cut out, allowed words dropped."""
    return [s for s in outside_format(text, list(spans), protected) if s.value.casefold() not in allow]


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


def _line(text: str, span: Span) -> tuple[str, str, str]:
    """The line holding the span, split as (text before it, the span, text after it)."""
    start = text.rfind("\n", 0, span.start) + 1
    end = text.find("\n", span.end)
    end = len(text) if end == -1 else end
    return text[start : span.start], text[span.start : span.end], text[span.end : end]


def _summary_table(members: int, tallies: dict[str, _Tally]) -> str:
    rows = [
        "| Detector | Spans | Member names fully caught | Keep-list hits | Not fully covered by others |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, t in tallies.items():
        recall = f"{t.member_hits}/{members} ({100 * t.member_hits / members:.0f}%)" if members else "-"
        rows.append(f"| {name} | {t.spans} | {recall} | {len(t.keep_hits)} | {len(t.only_this)} |")
    return "\n".join(rows)


# Public figures get their own label: redacting them is deliberate (see DESIGN.md, "Public figures"), so those
# are harmless over-redaction, unlike products, general places and ordinary words.
_PUBLIC = "public figure (fine to redact)"
_ONLY_OPTIONS = ["PII, complete", "PII, incomplete (part left outside the highlight)", "not PII", _PUBLIC]
_KEEP_OPTIONS = ["not PII (should stay readable)", "real PII", _PUBLIC]


def _write_report(
    out: Path, args: argparse.Namespace, n_messages: int, n_chunks: int, members: int, tallies: dict[str, _Tally]
) -> None:
    rng = random.Random(0)
    sections, keep_sections = [], []  # the review lists first; keep-list hits after
    for name, t in tallies.items():
        keep = t.keep_hits[: args.sample]
        keep_sections.append(
            {
                "id": f"{name}-keep",
                "title": f"{name}: keep-list hits",
                "total": len(t.keep_hits),
                "help": "Spans containing a tool or product name. Is the highlighted text something to hide?",
                "options": _KEEP_OPTIONS,
                "items": [{"before": b, "span": s, "after": a} for b, s, a in keep],
            }
        )
        sample = rng.sample(t.only_this, min(args.sample, len(t.only_this)))
        sections.append(
            {
                "id": f"{name}-only",
                "title": f"{name}: not fully covered by any other detector or rule",
                "total": len(t.only_this),
                "help": (
                    'Should the highlighted text be hidden from a hosted model? "Partly covered" spans reach beyond '
                    "another detector's find, e.g. a surname after a first name: judge that extra part. Names of "
                    "public figures get their own label (4): redacting them is fine."
                ),
                "options": _ONLY_OPTIONS,
                "items": [
                    {"kind": kind, "note": note, "before": b, "span": s, "after": a} for kind, note, b, s, a in sample
                ],
            }
        )
    sections += keep_sections
    data = {
        "intro": (
            f"{n_messages} messages in {n_chunks} chunk(s), the text a hosted summarizer would receive. "
            "Member-name and URL/email/handle rules always run; this compares the statistical detectors on top."
        ),
        "summary": _summary_table(members, tallies),
        "kinds": {name: dict(t.kinds.most_common()) for name, t in tallies.items()},
        "sections": sections,
    }
    payload = json.dumps(data, ensure_ascii=False)
    key = hashlib.sha256(payload.encode()).hexdigest()[:16]  # labels survive reloads, not a different report
    # Every "<" escaped, so no chat text (e.g. "</script>") can end the script block; the page builds the cards
    # with text nodes only, so nothing in a message is ever parsed as HTML.
    safe = payload.replace("<", "\\u003c")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_PAGE.replace("__KEY__", key).replace("__DATA__", safe), encoding="utf-8")


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>Redaction detector comparison (private)</title>
<style>
 body { font: 15px/1.45 system-ui, sans-serif; max-width: 70rem; margin: 2rem auto; padding: 0 1rem; color: #222; }
 .warn { background: #fff3cd; padding: .5rem .8rem; border-radius: 6px; }
 pre.summary { background: #f6f6f6; padding: .6rem; overflow-x: auto; }
 #tally { position: sticky; top: 0; background: #fff; border-bottom: 1px solid #ddd; padding: .5rem 0; z-index: 1; }
 #tally table { border-collapse: collapse; font-size: 14px; }
 #tally td, #tally th { padding: .15rem .6rem; text-align: right; } #tally td:first-child { text-align: left; }
 .item { border: 1px solid #ddd; border-radius: 6px; padding: .5rem .7rem; margin: .5rem 0; outline: none;
         scroll-margin-top: 12rem; }
 .item:focus { border-color: #3b82f6; box-shadow: 0 0 0 2px #bfdbfe; }
 .item.done { opacity: .55; } .item.done:focus { opacity: 1; }
 .line { font-family: ui-monospace, monospace; font-size: 13px; white-space: pre-wrap; word-break: break-word; }
 .hit { background: #fde68a; padding: 0 2px; border-radius: 3px; font-weight: 600; }
 .meta { color: #666; font-size: 12px; } button { margin: .3rem .3rem 0 0; cursor: pointer; }
 button.on { background: #2563eb; color: #fff; border-color: #2563eb; }
 textarea { width: 100%; height: 9rem; font-family: ui-monospace, monospace; font-size: 12px; }
</style></head><body>
<h1>Redaction detector comparison</h1>
<p class="warn"><b>Private:</b> this page contains chat text. Keep it local and out of git.</p>
<p class="warn" id="nosave" hidden><b>Progress is not being saved:</b> this browser blocks storage for this page,
so labels last only until you close or reload it. Copy the counts at the bottom before you leave.</p>
<p id="intro"></p><pre class="summary" id="summary"></pre><p class="meta" id="kinds"></p>
<p class="meta">Keys: <b>1</b>-<b>4</b> label the selected item and move on, <b>u</b> clears it,
<b>j</b>/<b>k</b> or arrows move. Labels are saved in this browser for this report when it allows that.</p>
<div id="tally"></div>
<div id="sections"></div>
<h2>Counts to share</h2>
<p class="meta">Numbers only, no chat text.</p>
<textarea id="counts" readonly></textarea><br><button id="copy">Copy counts</button>
<script>
const DATA = __DATA__;
const KEY = "tg-summary-redaction-eval:__KEY__";
// Storage can be refused (file:// pages, privacy settings): then labelling still works, in memory only.
let saving = true, labels = {};
const noSave = () => { saving = false; document.getElementById("nosave").hidden = false; };
try { const kept = localStorage.getItem(KEY); try { labels = JSON.parse(kept || "{}") || {}; } catch (e) { labels = {}; } }
catch (e) { noSave(); }
function save() { if (!saving) return; try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) { noSave(); } }
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text; return e; };
document.getElementById("intro").textContent = DATA.intro;
document.getElementById("summary").textContent = DATA.summary;
document.getElementById("kinds").textContent = "Spans by kind: " + Object.entries(DATA.kinds)
  .map(([d, k]) => d + ": " + Object.entries(k).map(([n, c]) => n + " " + c).join(", ")).join("; ");
const cards = [];
for (const sec of DATA.sections) {
  const box = el("section");
  box.append(el("h2", null, `${sec.title} (${sec.items.length} shown of ${sec.total})`), el("p", "meta", sec.help));
  if (!sec.items.length) box.append(el("p", "meta", "(none)"));
  sec.items.forEach((it, i) => {
    const id = sec.id + ":" + i, card = el("div", "item"); card.tabIndex = 0; card.dataset.id = id;
    const meta = [it.kind, it.note].filter(Boolean).join(", ");
    if (meta) card.append(el("div", "meta", meta));
    const line = el("div", "line"); line.append(el("span", null, it.before), el("span", "hit", it.span),
      el("span", null, it.after)); card.append(line);
    sec.options.forEach((opt, n) => { const b = el("button", null, `${n + 1} · ${opt}`); b.dataset.n = n;
      b.onclick = () => { setLabel(card, n); }; card.append(b); });
    card.sec = sec; box.append(card); cards.push(card); paint(card);
  });
  document.getElementById("sections").append(box);
}
function paint(card) { const v = labels[card.dataset.id];
  card.classList.toggle("done", v !== undefined);
  card.querySelectorAll("button").forEach(b => b.classList.toggle("on", String(v) === b.dataset.n)); }
function setLabel(card, n) {
  if (n === null) delete labels[card.dataset.id]; else labels[card.dataset.id] = n;
  save(); paint(card); tally();
  if (n !== null) { const next = cards.slice(cards.indexOf(card) + 1).find(c => labels[c.dataset.id] === undefined);
    (next || card).focus(); (next || card).scrollIntoView({ block: "center" }); } }
function tally() {
  const t = el("table"), head = el("tr"); head.append(el("th", null, "Section"));
  const rows = [], lines = [DATA.intro, DATA.summary, ""];
  for (const sec of DATA.sections) {
    const counts = sec.options.map(() => 0); let open = 0;
    sec.items.forEach((_, i) => { const v = labels[sec.id + ":" + i]; if (v === undefined) open++; else counts[v]++; });
    const tr = el("tr"); tr.append(el("td", null, sec.title));
    sec.options.forEach((o, n) => tr.append(el("td", null, `${o.split(" (")[0]}: ${counts[n]}`)));
    tr.append(el("td", null, `unlabelled: ${open}`)); rows.push(tr);
    lines.push(`${sec.title} (${sec.items.length} of ${sec.total}): ` +
      sec.options.map((o, n) => `${o} ${counts[n]}`).join(", ") + `, unlabelled ${open}`);
  }
  t.append(...rows); const box = document.getElementById("tally"); box.replaceChildren(t);
  document.getElementById("counts").value = lines.join("\\n");
}
document.getElementById("copy").onclick = () => { const c = document.getElementById("counts");
  c.select(); (navigator.clipboard ? navigator.clipboard.writeText(c.value) : Promise.reject())
    .catch(() => document.execCommand("copy")); };
document.addEventListener("keydown", e => {
  const card = document.activeElement.closest ? document.activeElement.closest(".item") : null;
  if (e.target.tagName === "TEXTAREA") return;
  const move = d => { const i = card ? cards.indexOf(card) + d : 0, c = cards[Math.max(0, Math.min(cards.length - 1, i))];
    if (c) { c.focus(); c.scrollIntoView({ block: "center" }); } };
  if (e.key === "j" || e.key === "ArrowDown") { move(1); e.preventDefault(); }
  else if (e.key === "k" || e.key === "ArrowUp") { move(-1); e.preventDefault(); }
  else if (card && e.key === "u") setLabel(card, null);
  else if (card && /^[1-9]$/.test(e.key) && +e.key <= card.sec.options.length) setLabel(card, +e.key - 1);
});
tally();
(cards.find(c => labels[c.dataset.id] === undefined) || cards[0] || document.body).focus();
</script></body></html>
"""


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
