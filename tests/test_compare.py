from __future__ import annotations

import dataclasses
import json
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.openai import OpenAIModelProfile

from tg_summary.compare import Tally, judge_digest, resolve_variant
from tg_summary.config import Config, Endpoint, RedactionSettings, Role, SummarizeSettings
from tg_summary.eval_cli import main
from tg_summary.llm import StructuredCaller, _settings
from tg_summary.segment import build_chunks
from tg_summary.sources.desktop_json import DesktopJsonSource
from tg_summary.store import Store
from tg_summary.summarize import Cited, Digest, Notes, Theme, summarize

LOCAL = Endpoint(name="local", api="openai", trust="private", base_url="http://x")
EMPTY = {"decisions": [], "open_questions": [], "action_items": [], "links": []}
NOTES = {"topics": [{"title": "Hike", "summary": "s", "participants": [], "msg_ids": [2]}], **EMPTY}
DIGEST = {"overview": "o", "themes": [{"title": "Hike", "summary": "s", "key_points": [], "msg_ids": [2]}], **EMPTY}


def _store(export_dir: Path, tmp_path: Path) -> Store:
    source = DesktopJsonSource(export_dir)
    store = Store(tmp_path / "chat.sqlite")
    store.bind_chat(source.chat(), source.name, source.media_root())
    store.upsert_messages(source.messages())
    return store


def _config(**roles: Role) -> Config:
    return Config(
        summarize=Role(endpoint=LOCAL, model="m", temperature=0.2),
        summarize_settings=SummarizeSettings(),
        redaction=RedactionSettings(),
        roles=dict(roles),
    )


# --- variants ------------------------------------------------------------------


def test_variant_digest_inherits_summarize_overrides_when_no_digest_role_is_configured():
    v = resolve_variant(
        {"name": "v", "summarize": {"thinking": False, "top_k": 20}, "digest": {"thinking": "low"}}, _config(), "v"
    )
    assert (v.summarize.thinking, v.summarize.top_k) == (False, 20)
    assert (v.digest.thinking, v.digest.top_k) == ("low", 20)


def test_variant_overrides_start_from_the_configured_digest_role():
    digest = Role(endpoint=LOCAL, model="big", context_tokens=65536)
    v = resolve_variant({"name": "v", "summarize": {"top_k": 20}}, _config(digest=digest), "v")
    assert v.digest.model == "big" and v.digest.top_k is None
    assert v.summarize.model == "m" and v.summarize.top_k == 20


def test_variant_default_sends_nothing_and_unknown_fields_are_rejected():
    v = resolve_variant({"name": "v", "summarize": {"temperature": "default", "thinking": "default"}}, _config(), "v")
    assert v.summarize.temperature is None and v.summarize.thinking is None
    with pytest.raises(ValueError, match="endpoint"):
        resolve_variant({"name": "v", "summarize": {"endpoint": "other"}}, _config(), "v")
    with pytest.raises(ValueError, match="thinking"):
        resolve_variant({"name": "v", "digest": {"thinking": "extreme"}}, _config(), "v")


def test_variant_can_switch_model_and_server_by_endpoint_name():
    import dataclasses

    lms = Endpoint(name="lmstudio", api="openai", trust="private", base_url="http://y")
    config = dataclasses.replace(_config(), endpoints={"local": LOCAL, "lmstudio": lms})
    v = resolve_variant({"name": "moe", "summarize": {"endpoint": "lmstudio", "model": "qwen-moe"}}, config, "v")
    assert (v.summarize.endpoint, v.summarize.model) == (lms, "qwen-moe")
    assert (v.digest.endpoint, v.digest.model) == (lms, "qwen-moe")  # no digest role: follows summarize
    with pytest.raises(ValueError, match="must name one of: local, lmstudio"):
        resolve_variant({"name": "v", "summarize": {"endpoint": "typo"}}, config, "v")


# --- request settings ----------------------------------------------------------


def test_local_thinking_switch_is_merged_under_the_roles_extra_body():
    off = _settings(Role(endpoint=LOCAL, model="m", thinking=False, top_k=20), OpenAIModelProfile())
    assert off["extra_body"] == {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
    }

    user = {"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False}, "reasoning_effort": "low"}
    mixed = _settings(Role(endpoint=LOCAL, model="m", thinking=False, extra_body=user), OpenAIModelProfile())
    assert mixed["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False},
        "reasoning_effort": "low",
    }

    unset = _settings(Role(endpoint=LOCAL, model="m", temperature=None), OpenAIModelProfile())
    assert "extra_body" not in unset and "temperature" not in unset


def test_thinking_reserve_is_added_to_max_tokens_only_when_thinking_is_on():
    on = Role(endpoint=LOCAL, model="m", max_output_tokens=4000, thinking="low", thinking_tokens=6000)
    assert _settings(on, OpenAIModelProfile())["max_tokens"] == 10_000
    assert (
        _settings(Role(endpoint=LOCAL, model="m", max_output_tokens=4000, thinking=False), OpenAIModelProfile())[
            "max_tokens"
        ]
        == 4000
    )


def test_anthropic_gets_a_fixed_thinking_budget_only_where_the_model_accepts_one():
    claude = Endpoint(name="claude", api="anthropic", trust="external")
    role = Role(endpoint=claude, model="m", thinking=True, thinking_tokens=3000)
    budget = _settings(role, anthropic_model_profile("claude-haiku-4-5"))
    assert budget["thinking"] is True
    assert budget["anthropic_thinking"] == {"type": "enabled", "budget_tokens": 3000}
    adaptive = _settings(role, anthropic_model_profile("claude-haiku-5-5"))
    assert adaptive["thinking"] is True and "anthropic_thinking" not in adaptive


# --- cache bypass --------------------------------------------------------------


def test_cache_bypass_repeats_the_call_and_leaves_the_cache_for_normal_runs(export_dir, tmp_path, fake_llm):
    store = _store(export_dir, tmp_path)
    seen = fake_llm(lambda user: json.dumps(NOTES))
    role = Role(endpoint=LOCAL, model="m")

    def call(caller: StructuredCaller) -> None:
        caller.call("sys", caller.prepare("#2 hello"), Notes, lambda v: (v, []))

    fresh = StructuredCaller(role, store, lambda: None, use_cache=False)
    call(fresh)
    call(fresh)
    assert len(seen) == 2

    cached = StructuredCaller(role, store, lambda: None)
    call(cached)
    call(cached)
    assert len(seen) == 3 and cached.stats.cache_hits == 1  # the bypassed calls wrote nothing


# --- digest role ---------------------------------------------------------------


def test_merge_and_digest_calls_go_to_the_digest_caller(export_dir, tmp_path, fake_llm):
    store = _store(export_dir, tmp_path)
    msgs = store.select()
    chunks = build_chunks(msgs, {m.id: m for m in msgs}, 160, timedelta(hours=2))
    assert len(chunks) == 2
    fake_llm(lambda user: json.dumps(DIGEST if "Notes 1" in user else NOTES))
    caller = StructuredCaller(Role(endpoint=LOCAL, model="small"), store, lambda: None)
    digest_caller = StructuredCaller(Role(endpoint=LOCAL, model="big"), store, lambda: None)

    summarize(chunks, caller, 160, digest_caller, digest_budget=10_000)

    assert caller.stats.calls == 2  # one per chunk
    assert digest_caller.stats.calls == 1  # notes fit the digest budget, so no merge round


# --- judge ---------------------------------------------------------------------


def test_tally_rates_exclude_failed_judge_calls():
    t = Tally()
    for label in ("supported", "supported", "partial", "unsupported", None):
        t.add(label)
    assert t.rates() == {"supported_rate": 0.5, "partial_rate": 0.25, "unsupported_rate": 0.25}
    assert Tally().rates()["supported_rate"] is None


def test_judge_sees_each_claim_with_only_its_cited_messages(export_dir, tmp_path, fake_llm):
    store = _store(export_dir, tmp_path)
    context = {m.id: m for m in store.select()}
    digest = Digest(
        overview="o",
        themes=[
            Theme(
                title="Hike", summary="Saturday hike", key_points=[Cited(text="Route shared", msg_ids=[4])], msg_ids=[2]
            )
        ],
        decisions=[Cited(text="Bram brings the stove", msg_ids=[9])],
        open_questions=[],
        action_items=[Cited(text="Chen brings filters", msg_ids=[9])],
        links=[],
    )
    prompts: list[str] = []

    def respond(user: str) -> str:
        prompts.append(user)
        if "Chen brings filters" in user:
            return "not json"  # judge failure, twice (repair round included)
        return json.dumps({"verdict": "supported" if "stove" in user or "Hike" in user else "unsupported"})

    fake_llm(respond)
    judge = StructuredCaller(Role(endpoint=LOCAL, model="judge"), store, lambda: None)

    tally = judge_digest(digest, judge, context)

    assert (tally.supported, tally.partial, tally.unsupported, tally.failed) == (2, 0, 1, 1)
    route = next(p for p in prompts if "Route shared" in p)
    assert "#4 " in route and "#2 " not in route and "#9 " not in route


# --- command -------------------------------------------------------------------


def _write_config(tmp_path: Path, judge_trust: str) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f"""
[endpoints.local]
api = "openai"
base_url = "http://x"
trust = "private"
[endpoints.judgehost]
api = "openai"
base_url = "http://y"
trust = "{judge_trust}"
[roles.summarize]
endpoint = "local"
model = "m"
[roles.judge]
endpoint = "judgehost"
model = "j"
"""
    )
    return cfg


def _write_variants(tmp_path: Path) -> Path:
    path = tmp_path / "variants.toml"
    path.write_text(
        """
[[variant]]
name = "off"
summarize = { thinking = false }
[[variant]]
name = "digest-low"
summarize = { thinking = false }
digest = { thinking = "low" }
"""
    )
    return path


def test_truncated_runs_name_the_stage_that_ran_out_of_output(export_dir, tmp_path, monkeypatch):
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.models.function import FunctionModel

    from tg_summary import llm

    db = tmp_path / "chat.sqlite"
    _store(export_dir, tmp_path).close()
    cut = {"stage": "map"}

    def fn(messages, info):
        user = next(p.content for m in messages if isinstance(m, ModelRequest) for p in m.parts
                    if isinstance(p, UserPromptPart))  # fmt: skip
        stage = "digest" if "Notes 1" in user else "map"
        if stage == cut["stage"]:
            return ModelResponse(parts=[TextPart('{"topics": [')], finish_reason="length")
        return ModelResponse(parts=[TextPart(json.dumps(NOTES))])

    monkeypatch.setattr(llm, "build_model", lambda role, client: FunctionModel(fn))
    config, variants = str(_write_config(tmp_path, "private")), str(_write_variants(tmp_path))
    for stage in ("map", "digest"):
        cut["stage"] = stage
        out = tmp_path / f"{stage}.json"
        assert main(["compare", "--db", str(db), "--config", config, "--variants", variants,
                     "--repeats", "1", "--out", str(out), "--no-judge"]) == 0  # fmt: skip
        runs = json.loads(out.read_text())["runs"]
        assert {(r["status"], r["truncated_stage"]) for r in runs} == {("truncated", stage)}


def test_compare_writes_metrics_only_and_samples_every_repeat(export_dir, tmp_path, fake_llm, capsys):
    db = tmp_path / "chat.sqlite"
    _store(export_dir, tmp_path).close()
    seen = fake_llm(
        lambda user: json.dumps(
            {"verdict": "supported"} if user.startswith("Claim:") else DIGEST if "Notes 1" in user else NOTES
        )
    )
    out = tmp_path / "results.json"

    code = main(
        ["compare", "--db", str(db), "--config", str(_write_config(tmp_path, "private")),
         "--variants", str(_write_variants(tmp_path)), "--repeats", "2", "--out", str(out)]
    )  # fmt: skip

    assert code == 0
    result = json.loads(out.read_text())
    assert [(r["variant"], r["status"]) for r in result["runs"]] == [
        ("off", "ok"), ("digest-low", "ok"), ("off", "ok"), ("digest-low", "ok")
    ]  # fmt: skip
    assert result["variants"]["digest-low"]["digest"]["thinking"] == "low"
    assert all(r["judge"]["supported_rate"] == 1.0 for r in result["runs"])
    summarize_calls = len([s for s in seen if "Claim:" not in s])
    assert summarize_calls == 4 * 2  # map + digest per run: nothing came from the cache
    printed = capsys.readouterr()
    for chat_text in ("Saturday", "Alice", "Mount Tam", "stove", "Hike"):
        assert chat_text not in printed.out and chat_text not in printed.err and chat_text not in out.read_text()


def test_resume_keeps_finished_runs_and_reruns_only_missing_or_server_failed_ones(export_dir, tmp_path, fake_llm):
    db = tmp_path / "chat.sqlite"
    _store(export_dir, tmp_path).close()
    seen = fake_llm(lambda user: json.dumps(DIGEST if "Notes 1" in user else NOTES))
    out = tmp_path / "results.json"
    args = ["compare", "--db", str(db), "--config", str(_write_config(tmp_path, "private")),
            "--variants", str(_write_variants(tmp_path)), "--no-judge", "--out", str(out)]  # fmt: skip

    assert main([*args, "--repeats", "1"]) == 0
    per_run = len(seen) // 2
    first = json.loads(out.read_text())
    first["runs"][0].update(status="error", error="connection")  # the server was down during that run
    out.write_text(json.dumps(first))
    seen.clear()

    assert main([*args, "--repeats", "2", "--resume"]) == 0
    result = json.loads(out.read_text())
    assert [(r["variant"], r["repeat"], r["status"]) for r in result["runs"]] == [
        ("off", 0, "ok"), ("digest-low", 0, "ok"), ("off", 1, "ok"), ("digest-low", 1, "ok")
    ]  # fmt: skip
    assert len(seen) == per_run * 3  # the failed run plus the two new repeats; the finished run was kept
    assert result["runs"][1]["run_id"] == first["runs"][1]["run_id"]

    assert main([*args, "--repeats", "2", "--resume", "--last", "3"]) == 1  # a different window can't resume


def test_resume_refuses_results_from_before_a_message_was_edited(export_dir, tmp_path, fake_llm, capsys):
    db = tmp_path / "chat.sqlite"
    _store(export_dir, tmp_path).close()
    seen = fake_llm(lambda user: json.dumps(DIGEST if "Notes 1" in user else NOTES))
    out = tmp_path / "results.json"
    base = ["compare", "--db", str(db), "--config", str(_write_config(tmp_path, "private")),
            "--variants", str(_write_variants(tmp_path)), "--no-judge", "--out", str(out)]  # fmt: skip
    assert main([*base, "--repeats", "1"]) == 0
    before = json.loads(out.read_text())

    store = Store(db)  # a re-ingested edit: same id, same message count, same chunking, different text
    edited = next(m for m in store.select() if m.id == 3)
    store.upsert_messages([dataclasses.replace(edited, text=edited.text + " (edited: actually 9am)")])
    store.close()
    seen.clear()

    assert main([*base, "--repeats", "2", "--resume"]) == 1
    assert seen == [] and json.loads(out.read_text()) == before  # nothing ran, nothing was overwritten
    assert "different chat text" in capsys.readouterr().err


def test_compare_refuses_a_judge_on_an_external_endpoint(export_dir, tmp_path, fake_llm, capsys):
    _store(export_dir, tmp_path).close()
    seen = fake_llm(lambda user: json.dumps(NOTES))
    code = main(
        ["compare", "--db", str(tmp_path / "chat.sqlite"), "--config", str(_write_config(tmp_path, "external")),
         "--variants", str(_write_variants(tmp_path)), "--repeats", "1", "--out", str(tmp_path / "r.json")]
    )  # fmt: skip
    assert code == 1 and seen == []
    assert "roles.judge must use a private endpoint" in capsys.readouterr().err


def test_runs_saved_without_a_judge_are_graded_later_across_results_files(export_dir, tmp_path, fake_llm, capsys):
    db = tmp_path / "chat.sqlite"
    _store(export_dir, tmp_path).close()
    seen = fake_llm(
        lambda user: json.dumps(
            {"verdict": "partial"} if user.startswith("Claim:") else DIGEST if "Notes 1" in user else NOTES
        )
    )
    config = str(_write_config(tmp_path, "private"))
    a_vars, b_vars = tmp_path / "a.toml", tmp_path / "b.toml"
    a_vars.write_text('[[variant]]\nname = "model-a"\nsummarize = { thinking = false }\n')
    b_vars.write_text('[[variant]]\nname = "model-b"\nsummarize = { thinking = false }\n')
    a, b, both = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "both.json"
    for variants, out in ((a_vars, a), (b_vars, b)):
        assert main(["compare", "--db", str(db), "--config", config, "--variants", str(variants),
                     "--repeats", "2", "--out", str(out), "--no-judge"]) == 0  # fmt: skip
    assert not any(s.startswith("Claim:") or "Claim:" in s for s in seen)  # nothing judged yet
    assert all("judge" not in r for r in json.loads(a.read_text())["runs"])

    assert main(["judge", "--db", str(db), "--config", config,
                 "--results", str(a), "--results", str(b), "--out", str(both)]) == 0  # fmt: skip

    combined = json.loads(both.read_text())
    assert sorted(combined["variants"]) == ["model-a", "model-b"]
    assert len(combined["runs"]) == 4 and all(r["judge"]["partial_rate"] == 1.0 for r in combined["runs"])
    printed = capsys.readouterr()
    assert "| model-a | 2/2 |" in printed.out and "| model-b | 2/2 |" in printed.out
    for chat_text in ("Saturday", "Alice", "Mount Tam", "stove", "Hike"):
        assert chat_text not in both.read_text() and chat_text not in printed.out

    # Results from a different window, or reusing a variant name, are refused.
    other = json.loads(b.read_text())
    other["window"]["fingerprint"] = "different"
    (tmp_path / "other.json").write_text(json.dumps(other))
    assert main(["judge", "--db", str(db), "--config", config, "--results", str(a),
                 "--results", str(tmp_path / "other.json"), "--out", str(both)]) == 1  # fmt: skip
    assert main(["judge", "--db", str(db), "--config", config, "--results", str(a),
                 "--results", str(a), "--out", str(both)]) == 1  # fmt: skip
    err = capsys.readouterr().err
    assert "different chat windows" in err and "unique across results files" in err


def test_deferred_judging_grades_the_evidence_the_run_saw_not_todays_messages(export_dir, tmp_path, fake_llm):
    import dataclasses

    db = tmp_path / "chat.sqlite"
    _store(export_dir, tmp_path).close()
    judged: list[str] = []

    def respond(user: str) -> str:
        if user.startswith("Claim:"):
            judged.append(user)
            return json.dumps({"verdict": "supported" if "Saturday hike" in user else "unsupported"})
        return json.dumps(DIGEST if "Notes 1" in user else NOTES)

    fake_llm(respond)
    config = str(_write_config(tmp_path, "private"))
    variants = tmp_path / "v.toml"
    variants.write_text('[[variant]]\nname = "v"\nsummarize = { thinking = false }\n')
    results, graded = tmp_path / "r.json", tmp_path / "g.json"
    assert main(["compare", "--db", str(db), "--config", config, "--variants", str(variants),
                 "--repeats", "1", "--out", str(results), "--no-judge"]) == 0  # fmt: skip

    # Between generating and judging, the cited message is edited (as a re-ingest of an edited message would).
    store = Store(db)
    edited = dataclasses.replace(store.get_messages([2])[2], text="something else entirely")
    store.upsert_messages([edited])
    store.close()

    assert main(["judge", "--db", str(db), "--config", config,
                 "--results", str(results), "--out", str(graded)]) == 0  # fmt: skip

    assert judged and all("Saturday hike" in u and "something else entirely" not in u for u in judged)
    assert json.loads(graded.read_text())["runs"][0]["judge"]["supported_rate"] == 1.0
