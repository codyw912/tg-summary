from __future__ import annotations

import dataclasses
import json
from datetime import timedelta
from pathlib import Path

import httpx2
import pytest
from conftest import _msg

from tg_summary import links
from tg_summary.config import Config, Endpoint, LinkSettings, MediaSettings, RedactionSettings, Role, SummarizeSettings
from tg_summary.enrich import enrich
from tg_summary.redact import Redactor, RosterDetector, rule_detector
from tg_summary.segment import build_chunks
from tg_summary.sources.desktop_json import DesktopJsonSource
from tg_summary.store import Store

PHOTO = b"\xff\xd8\xff\xe0 synthetic jpeg bytes"
LOCAL = Endpoint(name="local", api="openai", trust="private", base_url="http://local.invalid/v1")
HOSTED = Endpoint(name="claude", api="anthropic", trust="external", api_key_env="TEST_ANTHROPIC_KEY")


def _export(tmp_path: Path, messages: list[dict]) -> Store:
    d = tmp_path / "export"
    (d / "photos").mkdir(parents=True)
    (d / "photos" / "a.jpg").write_bytes(PHOTO)
    (d / "photos" / "repost.jpg").write_bytes(PHOTO)  # same bytes under another name
    data = {"name": "Hiking Club", "type": "private_supergroup", "id": 1234567890, "messages": messages}
    (d / "result.json").write_text(json.dumps(data), encoding="utf-8")
    source = DesktopJsonSource(d)
    store = Store(tmp_path / "chat.sqlite")
    store.bind_chat(source.chat(), source.name, source.media_root())
    store.upsert_messages(source.messages())
    return store


def _config(**roles: Role) -> Config:
    return Config(
        summarize=Role(endpoint=LOCAL, model="m"),
        summarize_settings=SummarizeSettings(),
        redaction=RedactionSettings(),
        roles=roles,
        media=MediaSettings(),
        links=LinkSettings(),
    )


MEDIA_MESSAGES = [
    _msg(2, 0, "Bram Okafor", 2, "Bram Okafor at the summit", photo="photos/a.jpg"),
    _msg(3, 1, "Alice Moreau", 1, "", photo="photos/repost.jpg"),
    _msg(4, 2, "Chen Wei", 3, "", media_type="voice_message", file="(File not included.)", duration_seconds=12),
    _msg(5, 3, "Chen Wei", 3, "", media_type="sticker", sticker_emoji="👍", file="stickers/s.webp"),
    _msg(6, 4, "Chen Wei", 3, "see https://example.org/route/42 and https://t.me/c/123/45"),
]


def test_reposted_photo_costs_one_call_is_reused_next_run_and_replaces_the_placeholder(tmp_path, fake_llm):
    store = _export(tmp_path, MEDIA_MESSAGES)
    seen = fake_llm(lambda user: "A trail map with a red route.")
    config = _config(describe_media=Role(endpoint=LOCAL, model="vision"))
    msgs = store.select()

    first = enrich(msgs, store, config, lambda: None)
    again = enrich(msgs, store, config, lambda: None)

    assert len(seen) == 1 and first.calls["describe_media"] == 1
    assert again.calls["describe_media"] == 0 and again.cache_hits == 1
    counts = first.counts()
    assert counts[("photo", "done")] == 1  # one item, two messages
    assert counts[("voice", "missing")] == 1
    assert counts[("sticker", "skipped")] == 1
    assert counts[("link", "skipped")] == 2  # no read_link role; t.me/c/ is never sent anyway
    text = build_chunks(msgs, {m.id: m for m in msgs}, 10_000, timedelta(hours=2), again.derived)[0].text
    assert text.count("[photo: A trail map with a red route.]") == 2
    assert "[voice 0:12]" in text  # missing file keeps its placeholder


def test_export_paths_that_could_leave_the_export_are_dropped_at_ingest():
    from tg_summary.sources.desktop_json import _path

    for crafted in ("/etc/hosts", "../secret.txt", "photos/../../secret.txt", "photos\\..\\x.jpg", "C:/x.jpg"):
        assert _path(crafted) is None
    assert _path("photos/photo_1@09-10-2025_10-06-00.jpg") == "photos/photo_1@09-10-2025_10-06-00.jpg"


def test_files_outside_the_media_root_are_never_read_or_sent(tmp_path, fake_llm):
    import dataclasses

    from tg_summary.models import Media

    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"\xff\xd8\xff\xe0 TOP-SECRET-BYTES")  # looks like a JPEG so only containment can stop it
    store = _export(tmp_path, MEDIA_MESSAGES[:1])
    (tmp_path / "export" / "photos" / "leak.jpg").symlink_to(secret)
    base = store.select()[0]
    # Bypass ingest: an older database or another source could hold any path string.
    crafted = [str(secret), "../secret.txt", "photos/../../secret.txt", "photos/leak.jpg"]
    store.upsert_messages(
        dataclasses.replace(base, id=100 + i, media=Media(kind="photo", path=p)) for i, p in enumerate(crafted)
    )
    seen = fake_llm(lambda user: "A photo.")
    config = _config(describe_media=Role(endpoint=LOCAL, model="vision"))

    result = enrich(store.select(), store, config, lambda: None)

    assert len(seen) == 1  # only the real photo inside the export
    assert all("TOP-SECRET" not in s for s in seen)
    counts = result.counts()
    assert counts[("photo", "done")] == 1
    assert counts[("photo", "missing")] == len(crafted)


def test_max_calls_leaves_the_placeholder_and_is_retried_later(tmp_path, fake_llm):
    store = _export(tmp_path, MEDIA_MESSAGES)
    fake_llm(lambda user: "A photo.")
    msgs = store.select()

    limited = enrich(msgs, store, _config(describe_media=Role(endpoint=LOCAL, model="v", max_calls=0)), lambda: None)
    assert limited.counts()[("photo", "over_limit")] == 1 and limited.derived == {}

    later = enrich(msgs, store, _config(describe_media=Role(endpoint=LOCAL, model="v")), lambda: None)
    assert later.counts()[("photo", "done")] == 1


def test_external_describe_gets_redacted_text_but_the_media_bytes_as_is(tmp_path, fake_llm, monkeypatch):
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "k")
    store = _export(tmp_path, MEDIA_MESSAGES)
    replies = []

    def respond(user):
        text = user[0]
        assert "Bram" not in text and "[PERSON_1]" in text
        assert any(getattr(part, "data", None) == PHOTO for part in user[1:])
        replies.append(text)
        return "[PERSON_1] holding a trail map."

    fake_llm(respond)
    names = store.sender_names()
    enriched = enrich(
        store.select(),
        store,
        _config(describe_media=Role(endpoint=HOSTED, model="vision")),
        lambda: Redactor(store.vault_assign, [RosterDetector(names), rule_detector]),
    )
    assert len(replies) == 1
    assert enriched.derived[2].media == "Bram Okafor holding a trail map."  # restored for this request only


def test_preview_shows_each_hosted_request_exactly_as_the_run_sends_it(tmp_path, fake_llm, monkeypatch):
    from tg_summary.enrich import preview_requests

    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "k")
    store = _export(tmp_path, MEDIA_MESSAGES)
    names = store.sender_names()
    config = _config(describe_media=Role(endpoint=HOSTED, model="vision", max_calls=1))

    def redactor():
        return Redactor(store.vault_assign, [RosterDetector(names), rule_detector])

    previewed = preview_requests(store.select(), store, config, redactor)
    sent: list[str] = []
    fake_llm(lambda user: sent.append(user[0]) or "A photo.")
    enrich(store.select(), store, config, redactor)

    assert [p.user for p in previewed] == sent  # same text, same redaction, same max_calls admission
    assert "Bram" not in sent[0]
    assert previewed[0].attachments.startswith("1 file (image/jpeg)")
    assert preview_requests(store.select(), store, config, redactor) == []  # stored results are not resent


# --- read_link against Anthropic-shaped responses ------------------------------------------------------------


def _anthropic(content: list[dict], stop: str = "end_turn") -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _call(i: int, url: str) -> dict:
    return {"type": "server_tool_use", "id": f"srvtoolu_{i}", "name": "web_fetch", "input": {"url": url}}


def _ok(i: int, url: str) -> dict:
    doc = {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "page"}, "title": "T"}
    return {
        "type": "web_fetch_tool_result",
        "tool_use_id": f"srvtoolu_{i}",
        "content": {"type": "web_fetch_result", "url": url, "content": doc, "retrieved_at": "2026-01-01T00:00:00Z"},
    }


def _err(i: int, code: str) -> dict:
    return {
        "type": "web_fetch_tool_result",
        "tool_use_id": f"srvtoolu_{i}",
        "content": {"type": "web_fetch_tool_result_error", "error_code": code},
    }


def _text(t: str) -> dict:
    return {"type": "text", "text": t}


GH, X = "https://github.com/example/repo", "https://x.com/someone/status/1"
SCRIPTS = {
    GH: [_anthropic([_text("Fetching."), _call(1, GH), _ok(1, GH), _text("A GitHub repository for a hiking app.")])],
    # The model claims success; the raw result says the fetch was refused.
    X: [_anthropic([_call(2, X), _err(2, "url_not_allowed"), _text("A post about hiking.")])],
}


@pytest.fixture
def anthropic_http(monkeypatch):
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "k")

    def install(scripts: dict[str, list[dict]]):
        sent: list[dict] = []
        queues = {url: list(responses) for url, responses in scripts.items()}

        def handler(request: httpx2.Request) -> httpx2.Response:
            body = json.loads(request.content)
            sent.append(body)
            first_user = json.dumps(body["messages"][0])
            url = next(u for u in queues if u in first_user)
            return httpx2.Response(200, json=queues[url].pop(0))

        monkeypatch.setattr(httpx2, "AsyncHTTPTransport", lambda *a, **k: httpx2.MockTransport(handler))
        return sent

    return install


def test_site_rules_pick_the_longest_matching_domain_and_youtube_brief_fetches_oembed():
    rules = {"youtube.com": "brief", "x.com": "skip", "docs.x.com": "read", "*": "read"}
    assert links.mode_for("https://m.youtube.com/watch?v=abc", rules) == "brief"
    assert links.mode_for("https://x.com/a/status/1", rules) == "skip"
    assert links.mode_for("https://docs.x.com/page", rules) == "read"  # longer domain wins
    assert links.mode_for("https://notx.com/", rules) == "read"  # not a subdomain of x.com
    assert links.mode_for("https://example.org", {"*": "brief"}) == "brief"
    # Ports and query-only URLs still resolve to the host the rule names.
    assert links.mode_for("https://x.com:443/a/status/1", rules) == "skip"
    assert links.mode_for("https://x.com?foo=bar", rules) == "skip"
    assert links.mode_for("https://DOCS.X.COM:8443/page", rules) == "read"
    yt = "https://youtube.com/watch?v=abc"
    assert links.fetch_url(yt, "brief").startswith("https://www.youtube.com/oembed?format=json&url=https%3A%2F%2F")
    assert links.fetch_url(yt, "read") == yt


def test_skip_rules_send_nothing_and_brief_links_get_one_capped_fetch(tmp_path, anthropic_http):
    yt = "https://youtube.com/shorts/abc"
    oembed = links.fetch_url(links.normalize(yt), "brief")
    sent = anthropic_http(
        {"oembed": [_anthropic([_call(1, oembed), _ok(1, oembed), _text("A Short by Some Channel about a game.")])]}
    )
    store = _export(tmp_path, [_msg(2, 0, "Bram Okafor", 2, f"look {yt} and {X}")])
    config = dataclasses.replace(
        _config(read_link=Role(endpoint=HOSTED, model="claude-test")),
        links=LinkSettings(rules={"youtube.com": "brief", "x.com": "skip"}),
    )

    result = enrich(store.select(), store, config, lambda: Redactor(store.vault_assign, [rule_detector]))

    assert len(sent) == 1  # x.com never sent
    tool = next(t for t in sent[0]["tools"] if t.get("name") == "web_fetch")
    assert tool["max_uses"] == 1 and tool["max_content_tokens"] == 2000
    assert oembed in json.dumps(sent[0]["messages"][0])
    assert result.counts()[("link", "opened")] == 1
    assert result.skip_reasons()["link rule"] == 1


def test_link_status_comes_from_raw_tool_results_not_the_models_claim(tmp_path, anthropic_http):
    sent = anthropic_http(SCRIPTS)
    store = _export(tmp_path, [_msg(2, 0, "Bram Okafor", 2, f"Bram Okafor shared {GH} and {X}")])
    config = _config(read_link=Role(endpoint=HOSTED, model="claude-test"))
    redactor = Redactor(store.vault_assign, [RosterDetector(store.sender_names()), rule_detector])

    first = enrich(store.select(), store, config, lambda: redactor)

    by_item = {d.item: d for d in first.outcomes}
    assert by_item[f"url:{GH}"].status == "opened"
    assert by_item[f"url:{X}"].status == "failed" and by_item[f"url:{X}"].error == "url_not_allowed"
    assert first.derived[2].links == ("[link: github.com/example/repo — A GitHub repository for a hiking app.]",)
    for body in sent:
        user = json.dumps(body["messages"][0])
        assert "Bram" not in user  # message text redacted; the URL itself goes as-is
        assert any(t.get("name") == "web_fetch" for t in body["tools"])
    assert first.external_calls == 2


def test_permanent_link_failures_are_reused_and_transient_ones_are_retried(tmp_path, anthropic_http):
    flaky = [_anthropic([_call(3, GH), _err(3, "url_not_accessible"), _text("Could not open it.")])]
    anthropic_http({GH: flaky, X: SCRIPTS[X]})
    store = _export(tmp_path, [_msg(2, 0, "Alice Moreau", 1, f"{GH} {X}")])
    config = _config(read_link=Role(endpoint=HOSTED, model="claude-test"))
    redactor = Redactor(store.vault_assign, [rule_detector])

    first = enrich(store.select(), store, config, lambda: redactor)
    assert {d.item: d.error for d in first.outcomes} == {
        f"url:{GH}": "url_not_accessible",
        f"url:{X}": "url_not_allowed",
    }

    sent = anthropic_http({GH: SCRIPTS[GH]})  # only the transient failure may be requested again
    again = enrich(store.select(), store, config, lambda: redactor)

    assert len(sent) == 1 and GH in json.dumps(sent[0]["messages"][0])
    assert again.calls["read_link"] == 1 and again.cache_hits == 1
    assert {d.item: d.status for d in again.outcomes} == {f"url:{GH}": "opened", f"url:{X}": "failed"}
    assert again.derived[2].links == ("[link: github.com/example/repo — A GitHub repository for a hiking app.]",)


def test_paused_turn_is_continued_and_its_fetch_counts(tmp_path, anthropic_http):
    paused = _anthropic([_call(1, GH), _ok(1, GH)], stop="pause_turn")
    done = _anthropic([_text("A repository README.")])
    sent = anthropic_http({GH: [paused, done]})
    store = _export(tmp_path, [_msg(2, 0, "Alice Moreau", 1, GH)])

    redactor = Redactor(store.vault_assign, [rule_detector])
    out = enrich(store.select(), store, _config(read_link=Role(endpoint=HOSTED, model="claude-test")), lambda: redactor)

    assert len(sent) == 2
    assert out.outcomes[0].status == "opened" and out.outcomes[0].text == "A repository README."


def test_no_fetch_evidence_is_unverified_and_error_blocks_without_text_still_fail():
    assert links.status(GH, [], "A repo.") == ("unverified", None)
    assert links.status(GH, [], "") == ("failed", "no_output")
    other = links.Fetch("https://example.org", ok=True)
    assert links.status(GH, [other], "A repo.") == ("failed", "url_not_fetched")
    assert links.status(GH + "/", [links.Fetch("http://www.GITHUB.com/example/repo", ok=True)], "") == ("opened", None)


def test_private_message_and_invite_links_are_never_sent():
    skip = LinkSettings().skip
    for url in (
        "https://t.me/c/123/45",
        "t.me/+AbCd",
        "https://t.me/joinchat/xyz",
        "https://www.telegram.me/c/1/2",
        "https://t.me:443/c/123/45",  # a port must not hide a private link
        "https://T.ME/c/9/9?single",
    ):
        assert links.skipped(links.normalize(url), skip)
    assert not links.skipped(links.normalize("https://t.me/somechannel/12"), skip)
