"""Chunk notes that don't fit, failed chunks, and failing servers: runs finish or resume instead of starting over."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic_ai.exceptions import ModelConnectionError, ModelHTTPError
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from tg_summary import llm
from tg_summary.config import Endpoint, Role
from tg_summary.llm import LLMError, StructuredCaller
from tg_summary.models import Message
from tg_summary.segment import _costs, build_chunks, split_chunk
from tg_summary.sources.desktop_json import DesktopJsonSource
from tg_summary.store import Store
from tg_summary.summarize import summarize

EMPTY = {"decisions": [], "open_questions": [], "action_items": [], "links": []}
ROLE = Role(endpoint=Endpoint(name="local", api="openai", trust="private", base_url="http://x"), model="m")
GAP = timedelta(hours=2)


def _store(export_dir: Path, tmp_path: Path) -> Store:
    source = DesktopJsonSource(export_dir)
    store = Store(tmp_path / "chat.sqlite")
    store.bind_chat(source.chat(), source.name, source.media_root())
    store.upsert_messages(source.messages())
    return store


def _model(monkeypatch, respond) -> list[str]:
    """Every request goes to `respond(user_prompt)`, which returns a ModelResponse or raises."""
    seen: list[str] = []

    def fn(messages, info):
        user = next(
            p.content for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, UserPromptPart)
        )
        seen.append(user)
        return respond(user)

    monkeypatch.setattr(llm, "build_model", lambda role, client: FunctionModel(fn))
    return seen


def _ids(user: str) -> list[int]:
    return [int(i) for i in re.findall(r"^#(\d+) ", user, re.MULTILINE)]


def _notes_for(user: str) -> ModelResponse:
    ids = _ids(user)
    notes = {"topics": [{"title": "t", "summary": "s", "participants": [], "msg_ids": ids[:1]}], **EMPTY}
    return ModelResponse(parts=[TextPart(json.dumps(notes))])


def _digest() -> ModelResponse:
    digest = {"overview": "o", "themes": [{"title": "t", "summary": "s", "key_points": [], "msg_ids": [2]}], **EMPTY}
    return ModelResponse(parts=[TextPart(json.dumps(digest))])


def _msg(mid: int, minute: int, reply_to: int | None = None) -> Message:
    date = datetime(2025, 10, 9, 10, tzinfo=UTC) + timedelta(minutes=minute)
    return Message(
        id=mid, date=date, sender_id="u1", sender="Ann", text=f"message number {mid} " * 3, reply_to=reply_to
    )


def test_chunks_are_cut_at_a_quiet_gap_that_no_reply_crosses():
    # The 60-minute gap before #4 is crossed by #5 replying to #2; the 37-minute gap before #6 is crossed by nothing.
    msgs = [_msg(1, 0), _msg(2, 1), _msg(3, 2), _msg(4, 62), _msg(5, 63, reply_to=2), _msg(6, 100)]
    context = {m.id: m for m in msgs}
    one_cut = sum(_costs(msgs, context, None)) - 1  # the whole window doesn't fit; anything smaller does
    chunks = build_chunks(msgs, context, budget_tokens=one_cut, gap=GAP)
    assert [sorted(c.ids) for c in chunks] == [[1, 2, 3, 4, 5], [6]]

    # Halving a chunk only cuts in its middle half, so the crossed gap is the best place left there.
    whole = build_chunks(msgs, context, 10_000, GAP)[0]
    assert [sorted(h.ids) for h in split_chunk(whole, context, GAP)] == [[1, 2, 3], [4, 5, 6]]


def test_a_chunk_whose_notes_run_out_is_summarized_in_halves_and_a_rerun_skips_the_truncated_call(
    export_dir, tmp_path, monkeypatch
):
    store = _store(export_dir, tmp_path)
    msgs = store.select()
    chunks = build_chunks(msgs, {m.id: m for m in msgs}, 10_000, GAP)
    assert len(chunks) == 1 and len(chunks[0].messages) > 4

    def respond(user: str) -> ModelResponse:
        if "Notes 1" in user:
            return _digest()
        if len(_ids(user)) > 4:  # the whole chunk doesn't fit the output limit; halves do
            return ModelResponse(parts=[TextPart('{"topics": [')], finish_reason="length")
        return _notes_for(user)

    seen = _model(monkeypatch, respond)

    def split(chunk):
        return split_chunk(chunk, {m.id: m for m in msgs}, GAP)

    outcome = summarize(chunks, StructuredCaller(ROLE, store, lambda: None), 10_000, split=split)

    assert outcome.splits >= 1
    halves = [set(_ids(u)) for u in seen if "Notes 1" not in u and 0 < len(_ids(u)) <= 4]
    assert set().union(*halves) == chunks[0].ids  # every message reached a successful notes call

    seen.clear()
    rerun = StructuredCaller(ROLE, store, lambda: None)
    summarize(chunks, rerun, 10_000, split=split)
    assert seen == [] and rerun.stats.calls == 0  # truncation remembered, halves and digest cached


def test_a_part_citing_a_message_only_its_sibling_part_contains_loses_that_citation(export_dir, tmp_path, monkeypatch):
    store = _store(export_dir, tmp_path)
    msgs = store.select()
    chunks = build_chunks(msgs, {m.id: m for m in msgs}, 10_000, GAP)
    whole = chunks[0].ids
    parts = 0

    def respond(user: str) -> ModelResponse:
        nonlocal parts
        if "Notes 1" in user:
            return _digest()
        ids = _ids(user)
        if len(ids) == len(whole):
            return ModelResponse(parts=[TextPart('{"topics": [')], finish_reason="length")
        parts += 1
        sibling = min(whole - set(ids))  # in the chunk, but not in this part; cited again after the repair
        topic = {"title": "t", "summary": "s", "participants": [], "msg_ids": [ids[0], sibling]}
        return ModelResponse(parts=[TextPart(json.dumps({"topics": [topic], **EMPTY}))])

    _model(monkeypatch, respond)
    split = lambda c: split_chunk(c, {m.id: m for m in msgs}, GAP)
    outcome = summarize(chunks, StructuredCaller(ROLE, store, lambda: None), 10_000, split=split)

    assert outcome.splits == 1 and parts == 4  # two parts, each answered twice (first try and repair)
    assert outcome.dropped_citations == 2  # each part's citation of its sibling's message


def test_a_failed_chunk_does_not_stop_the_others_and_a_rerun_redoes_only_that_chunk(export_dir, tmp_path, monkeypatch):
    store = _store(export_dir, tmp_path)
    msgs = store.select()
    chunks = build_chunks(msgs, {m.id: m for m in msgs}, budget_tokens=160, gap=GAP)
    assert len(chunks) >= 2
    broken = {"on": True}

    def respond(user: str) -> ModelResponse:
        if "Notes 1" in user:
            return _digest()
        if broken["on"] and 9 in _ids(user):
            raise ModelHTTPError(status_code=400, model_name="m", body="bad request")
        return _notes_for(user)

    seen = _model(monkeypatch, respond)
    with pytest.raises(LLMError) as failed:
        summarize(chunks, StructuredCaller(ROLE, store, lambda: None), 160)
    assert failed.value.code == "http_400" and f"1 of {len(chunks)} chunk(s) failed" in str(failed.value)
    assert sum("Notes 1" not in u for u in seen) == len(chunks)  # every chunk was attempted

    broken["on"] = False
    seen.clear()
    summarize(chunks, StructuredCaller(ROLE, store, lambda: None), 160)
    map_calls = [u for u in seen if "Notes 1" not in u]
    assert len(map_calls) == 1 and 9 in _ids(map_calls[0])


def test_transient_failures_are_retried_then_reported_with_a_short_code(export_dir, tmp_path, monkeypatch):
    store = _store(export_dir, tmp_path)
    msgs = store.select()
    chunks = build_chunks(msgs, {m.id: m for m in msgs}, 10_000, GAP)
    monkeypatch.setattr(llm, "RETRY_BACKOFF", (0, 0))
    failures = {"left": 1}

    def respond(user: str) -> ModelResponse:
        if failures["left"]:
            failures["left"] -= 1
            raise ModelConnectionError("m", "connection refused", phase="connect")
        return _digest() if "Notes 1" in user else _notes_for(user)

    _model(monkeypatch, respond)
    caller = StructuredCaller(ROLE, store, lambda: None)
    summarize(chunks, caller, 10_000)
    assert caller.stats.transient_retries == 1

    failures["left"] = 99
    fresh = StructuredCaller(ROLE, _store(export_dir, tmp_path / "other"), lambda: None)
    with pytest.raises(LLMError) as gave_up:
        summarize(chunks, fresh, 10_000)
    assert gave_up.value.code == "connection" and fresh.stats.transient_retries == 2


def test_lists_over_their_cap_are_trimmed_and_counted(export_dir, tmp_path, monkeypatch):
    store = _store(export_dir, tmp_path)
    msgs = store.select()
    chunks = build_chunks(msgs, {m.id: m for m in msgs}, 10_000, GAP)
    themes = [{"title": f"t{i}", "summary": "s", "key_points": [], "msg_ids": [2]} for i in range(12)]

    def respond(user: str) -> ModelResponse:
        if "Notes 1" in user:
            return ModelResponse(parts=[TextPart(json.dumps({"overview": "o", "themes": themes, **EMPTY}))])
        return _notes_for(user)

    _model(monkeypatch, respond)
    outcome = summarize(chunks, StructuredCaller(ROLE, store, lambda: None), 10_000)
    assert [t.title for t in outcome.digest.themes] == [f"t{i}" for i in range(10)]
    assert outcome.trimmed_entries == 2
