"""tg-summary: ingest a Telegram chat, enrich media and links, preview redaction, and write a cited digest.

Evaluation tools (comparing settings and models, judging digests) live in `tg-summary-eval`."""

from __future__ import annotations

import argparse
import contextlib
import functools
import os
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import NoReturn

from tg_summary import enrich
from tg_summary.config import Config, load_config
from tg_summary.llm import CallStats, LLMError, StructuredCaller
from tg_summary.models import Chat, Message
from tg_summary.redact import Redactor, RequestMap, build_redactor
from tg_summary.report import Coverage, render
from tg_summary.segment import Chunk, build_chunks, split_chunk
from tg_summary.sources.desktop_json import DesktopJsonSource
from tg_summary.store import Store
from tg_summary.summarize import budget_for, summarize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tg-summary", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="load a Telegram Desktop single-chat JSON export into a chat database")
    ing.add_argument("export", type=Path, help="export folder (containing result.json) or the result.json file")
    ing.add_argument("--db", type=Path, help="database path (default: data/chat-<id>.sqlite)")

    for name, help_text in (
        ("summarize", "write a Markdown digest"),
        ("preview-redaction", "write exactly what an external endpoint would receive, without sending it"),
        ("enrich", "describe media, transcribe audio and read links for the window, without summarizing"),
    ):
        p = sub.add_parser(name, help=help_text)
        add_window_args(p)
        if name != "enrich":
            p.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            return _ingest(args)
        with keep_awake():
            return _run(args)
    except (ValueError, LLMError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        interrupted("every finished model call is saved, so rerunning the same command resumes")


@contextlib.contextmanager
def keep_awake() -> Iterator[None]:
    """Hold off idle sleep while a long run works (macOS `caffeinate -i`, tied to this process so it can't
    outlive it). Sleep from closing the lid or a low battery still happens."""
    proc = None
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        proc = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    try:
        yield
    finally:
        if proc is not None:
            proc.terminate()


def interrupted(resume_hint: str) -> NoReturn:
    """Exit now on Ctrl-C, without waiting for a model request in flight: its answer would be lost anyway, and
    every finished call is already committed. (Waiting would also turn a second Ctrl-C into a traceback.)"""
    print(f"\ninterrupted; {resume_hint}", file=sys.stderr, flush=True)
    sys.stdout.flush()
    os._exit(130)


def add_window_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("config.toml"))
    p.add_argument("--since", type=_date, help="start date, inclusive (YYYY-MM-DD, local time)")
    p.add_argument("--until", type=_date, help="end date, exclusive (YYYY-MM-DD, local time)")
    p.add_argument("--last", type=int, help="only the newest N messages (after date filters)")


@dataclass
class Window:
    """One selected window of chat, with its config, store, and media/link results."""

    config: Config
    store: Store
    chat: Chat
    messages: list[Message]
    redactor_factory: Callable[[], Redactor]
    enrichment: enrich.Enrichment | None

    @property
    def derived(self):
        return self.enrichment.derived if self.enrichment else None


def load_window(args: argparse.Namespace, offline: bool) -> Window:
    """Open the database, select the window, and run the media and link stage (`offline`: stored results only,
    nothing sent). Prints the stage's counts to stderr."""
    config = load_config(args.config)
    if not args.db.exists():
        raise FileNotFoundError(f"{args.db} does not exist; run `tg-summary ingest` first")
    store = Store(args.db)
    chat = store.chat()
    messages = store.select(args.since, args.until, args.last)
    if not messages:
        raise ValueError("no messages in the selected window")

    @functools.cache
    def redactor_factory():
        # One redactor for every role, so the report's redaction counts cover media and link calls too.
        return build_redactor(store.vault_assign, store.vault_values, store.sender_names(), config.redaction)

    enrichment = None
    if enrich.enabled(config):
        enrichment = enrich.enrich(messages, store, config, redactor_factory, offline=offline)
        counts = ", ".join(f"{k} {s} {n}" for (k, s), n in sorted(enrichment.counts().items())) or "no items"
        print(f"media and links: {counts}", file=sys.stderr)
        if reasons := enrichment.skip_reasons():
            print("  skipped because: " + ", ".join(f"{r} {n}" for r, n in sorted(reasons.items())), file=sys.stderr)
        if enrichment.pending:
            todo = ", ".join(f"{kind} → {role} {n}" for (kind, role), n in sorted(enrichment.pending.items()))
            print(f"  not processed yet (a real run would): {todo}", file=sys.stderr)
    return Window(config, store, chat, messages, redactor_factory, enrichment)


def _ingest(args: argparse.Namespace) -> int:
    source = DesktopJsonSource(args.export)
    chat = source.chat()
    store = Store(args.db or Path("data") / f"chat-{chat.id}.sqlite")
    store.bind_chat(chat, source.name, source.media_root())
    n = store.upsert_messages(source.messages())
    print(f"{chat.name} ({chat.type}): {n} records stored")
    return 0


def _run(args: argparse.Namespace) -> int:
    # Media and links become text before chunking. The preview only reuses stored results and sends nothing.
    w = load_window(args, offline=args.command == "preview-redaction")
    if args.command == "enrich":
        return 0
    config, store, chat, messages, redactor_factory = w.config, w.store, w.chat, w.messages, w.redactor_factory
    enrichment, derived = w.enrichment, w.derived
    role = config.summarize

    digest_role = config.roles.get("digest", role)
    chunk_tokens = config.summarize_settings.chunk_tokens
    budget = budget_for(role, chunk_tokens)
    context = store.get_messages(m.reply_to for m in messages if m.reply_to is not None)
    context.update({m.id: m for m in messages})
    gap = timedelta(minutes=config.summarize_settings.gap_minutes)
    chunks = build_chunks(messages, context, budget, gap, derived)

    caller = StructuredCaller(role, store, redactor_factory)

    if args.command == "preview-redaction":
        outbound = enrich.preview_requests(messages, store, config, redactor_factory) if enrich.enabled(config) else []
        _preview(args.out, chunks, caller, outbound)
        return 0

    digest_caller = StructuredCaller(digest_role, store, redactor_factory) if digest_role is not role else caller
    print(
        f"{len(messages)} messages → {len(chunks)} chunk(s) for {role.model} on {role.endpoint.name}", file=sys.stderr
    )
    split = functools.partial(split_chunk, context=context, gap=gap, derived=derived)
    outcome = summarize(chunks, caller, budget, digest_caller, budget_for(digest_role, chunk_tokens), split)
    callers = [caller] if digest_caller is caller else [caller, digest_caller]
    model, endpoint = role.model, role.endpoint.name
    if digest_role.model != model:
        model += f" + {digest_role.model}"
    if digest_role.endpoint.name != endpoint:
        endpoint += f" + {digest_role.endpoint.name}"
    external = any(c.external for c in callers) or bool(enrichment and enrichment.external_calls)
    coverage = Coverage(
        messages=messages,
        service_counts=store.service_counts(messages[0].date, messages[-1].date),
        model=model,
        endpoint=endpoint,
        external=external,
        stats=CallStats.combined(*(c.stats for c in callers)),
        redactions=caller.redactor.counts if external else Counter(),  # one shared redactor for every role
        enrichment=enrichment,
        media_appendix=config.report.media_appendix,
        bangers=config.report.bangers,
        bangers_min_reactions=config.report.bangers_min_reactions,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(chat, outcome, coverage), encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


def _preview(out: Path, chunks: list[Chunk], caller: StructuredCaller, outbound: list[enrich.Outbound]) -> None:
    """Chunk text as an external summarize/digest endpoint would get it, then every hosted media/link request
    exactly as a run would build it (each item's message text is redacted on its own, as the run does)."""
    redactor = caller.redactor  # preview always redacts, even when the configured endpoint is private
    parts = []
    for c in chunks:
        request = RequestMap()
        parts.append(
            f"===== summary chunk {c.index + 1}/{len(chunks)} (sent only if summarize/digest is external) =====\n"
            f"{redactor.redact(c.text, request)}"
        )
    for i, o in enumerate(outbound, 1):
        body = o.user or "(no text; only the attachment)"
        parts.append(
            f"===== hosted request {i}/{len(outbound)}: {o.role} {o.kind} {o.key[:24]} =====\n"
            f"attachments: {o.attachments}\n--- user message ---\n{body}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    counts = ", ".join(f"{k} {v}" for k, v in redactor.counts.most_common()) or "nothing"
    print(
        f"wrote {out}: {len(chunks)} chunk(s), {len(outbound)} hosted media/link request(s); redacted {counts}",
        file=sys.stderr,
    )


def _date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").astimezone()


if __name__ == "__main__":
    sys.exit(main())
