from __future__ import annotations

import json
import re
from datetime import timedelta
from pathlib import Path

import pytest

from tg_summary.config import Endpoint, Role
from tg_summary.llm import StructuredCaller
from tg_summary.redact import Redactor, RosterDetector, rule_detector
from tg_summary.segment import build_chunks
from tg_summary.sources.desktop_json import DesktopJsonSource
from tg_summary.store import Store
from tg_summary.summarize import summarize


def _ingest(export_dir: Path, tmp_path: Path) -> Store:
    source = DesktopJsonSource(export_dir)
    store = Store(tmp_path / "chat.sqlite")
    store.bind_chat(source.chat(), source.name, source.media_root())
    store.upsert_messages(source.messages())
    return store


def test_ingest_renders_text_links_media_and_skips_service_events(export_dir, tmp_path):
    store = _ingest(export_dir, tmp_path)
    msgs = {m.id: m for m in store.select()}

    assert 1 not in msgs  # service event
    assert msgs[4].text == "Here: route map (https://example.org/route/42) — call me on +1 415 555 0134 if lost"
    assert msgs[5].media.kind == "photo" and msgs[5].media.path.endswith(".jpg")
    assert msgs[6].media.kind == "voice" and msgs[6].media.path is None  # skipped file
    assert msgs[7].media.detail == "👍"
    assert msgs[8].media.detail == "Bring a stove? — Yes (2), No (1)"
    assert store.chat().message_link(9) == "https://t.me/c/1234567890/9"
    assert msgs[9].reactions == (("👍", 3), ("custom emoji", 1), ("⭐", 2)) and msgs[9].reaction_total == 6
    assert msgs[2].reactions == ()


def test_a_database_from_before_reactions_opens_and_reingest_fills_them(export_dir, tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, date INTEGER NOT NULL, sender_id TEXT, sender TEXT, "
        "text TEXT NOT NULL, reply_to INTEGER, forwarded_from TEXT, edited INTEGER, media TEXT, service_action TEXT)"
    )
    old.execute("INSERT INTO messages (id, date, text) VALUES (9, 1760018120, 'old row')")
    old.commit()
    old.close()

    store = Store(path)
    assert store.get_messages([9])[9].reactions == ()  # readable before re-ingest
    source = DesktopJsonSource(export_dir)
    store.bind_chat(source.chat(), source.name, source.media_root())
    store.upsert_messages(source.messages())
    assert store.get_messages([9])[9].reaction_total == 6


def test_ingest_refuses_a_different_chat(export_dir, tmp_path):
    store = _ingest(export_dir, tmp_path)
    data = json.loads((export_dir / "result.json").read_text())
    data["id"] = 42
    (export_dir / "result.json").write_text(json.dumps(data))
    source = DesktopJsonSource(export_dir)
    with pytest.raises(ValueError, match="database holds chat"):
        store.bind_chat(source.chat(), source.name, None)


def test_last_n_keeps_newest_regular_messages_in_order(export_dir, tmp_path):
    store = _ingest(export_dir, tmp_path)
    assert [m.id for m in store.select(last=3)] == [7, 8, 9]


def test_chunks_split_at_the_longest_silence_and_quote_replies_from_other_chunks(export_dir, tmp_path):
    store = _ingest(export_dir, tmp_path)
    msgs = store.select()
    context = {m.id: m for m in msgs}
    whole = build_chunks(msgs, context, budget_tokens=10_000, gap=timedelta(hours=2))
    assert len(whole) == 1 and "--- 5h later ---" in whole[0].text

    split = build_chunks(msgs, context, budget_tokens=160, gap=timedelta(hours=2))
    assert [sorted(c.ids) for c in split][-1] == [8, 9]  # cut lands on the 5h gap

    tiny = build_chunks(msgs[1:], {m.id: m for m in msgs}, budget_tokens=40, gap=timedelta(hours=2))
    first = tiny[0].text.splitlines()[0]
    assert first.startswith("#3 ") and '↪#2 (Alice Moreau: "Saturday hike' in first


EMPTY = {"decisions": [], "open_questions": [], "action_items": [], "links": []}


def test_invalid_citations_trigger_one_repair_then_get_dropped(export_dir, tmp_path, fake_llm):
    store = _ingest(export_dir, tmp_path)
    msgs = store.select()
    chunks = build_chunks(msgs, {m.id: m for m in msgs}, 10_000, timedelta(hours=2))
    bad_notes = {"topics": [{"title": "Hike", "summary": "s", "participants": [], "msg_ids": [2, 999]}], **EMPTY}
    digest = {
        "overview": "o",
        "themes": [{"title": "Hike", "summary": "s", "key_points": [], "msg_ids": [2, 777]}],
        **EMPTY,
        "decisions": [{"text": "invented", "msg_ids": [555]}],
    }
    seen = fake_llm(lambda user: json.dumps(digest if "Notes 1" in user else bad_notes))
    role = Role(endpoint=Endpoint(name="local", api="openai", trust="private", base_url="http://x"), model="m")
    caller = StructuredCaller(role, store, lambda: None)

    outcome = summarize(chunks, caller, budget_tokens=10_000)

    assert len(seen) == 4  # map + repair, digest + repair
    assert "999" in seen[1]  # the repair request names the bad id
    assert caller.stats.repairs == 2
    assert outcome.digest.themes[0].msg_ids == [2]
    assert outcome.digest.decisions == []
    assert outcome.dropped_items == 1


def test_external_endpoint_sees_only_placeholders_and_report_gets_names_back(export_dir, tmp_path, fake_llm):
    store = _ingest(export_dir, tmp_path)
    msgs = store.select()
    chunks = build_chunks(msgs, {m.id: m for m in msgs}, 10_000, timedelta(hours=2))

    def respond(user: str) -> str:
        match = re.search(r"\[PERSON_\d+\]", user)
        assert match is not None
        person = match.group(0)
        if "Notes 1" in user:
            theme = {"title": "Hike", "summary": f"{person} leads", "key_points": [], "msg_ids": [2]}
            return json.dumps({"overview": f"{person} planned a hike", "themes": [theme], **EMPTY})
        notes = {"topics": [{"title": "Hike", "summary": f"{person} plans", "participants": [person], "msg_ids": [2]}]}
        return json.dumps({**notes, **EMPTY})

    seen = fake_llm(respond)
    role = Role(endpoint=Endpoint(name="hosted", api="anthropic", trust="external"), model="m")
    names = store.sender_names()
    caller = StructuredCaller(role, store, lambda: Redactor(store.vault_assign, [RosterDetector(names), rule_detector]))

    outcome = summarize(chunks, caller, budget_tokens=10_000)

    sent = "\n".join(seen)
    for raw in ("Alice", "Bram", "Okafor", "Chen", "example.org", "@chen_hikes"):
        assert raw not in sent
    assert "Alice Moreau" in outcome.digest.overview
    assert caller.stats.external_calls == 2
