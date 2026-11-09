"""The redaction detector comparison: counts on the terminal, chat text only in the private report."""

from __future__ import annotations

import json
import re

from tg_summary import pii_tracer
from tg_summary.eval_cli import main
from tg_summary.pii_tracer import Found
from tg_summary.redact import P_NER, PresidioDetector, Span
from tg_summary.redaction_eval import covers
from tg_summary.sources.desktop_json import DesktopJsonSource
from tg_summary.store import Store


def test_comparison_prints_counts_only_and_lists_disagreements_for_labelling(export_dir, tmp_path, monkeypatch, capsys):
    source = DesktopJsonSource(export_dir)
    store = Store(tmp_path / "chat.sqlite")
    store.bind_chat(source.chat(), source.name, source.media_root())
    store.upsert_messages(source.messages())
    store.close()
    config = tmp_path / "config.toml"
    config.write_text('[endpoints.local]\napi = "openai"\nbase_url = "http://x"\ntrust = "private"\n'
                      '[roles.summarize]\nendpoint = "local"\nmodel = "m"\n')  # fmt: skip

    def spans(text: str, pattern: str, kind: str) -> list[Span]:
        return [Span(m.start(), m.end(), kind, m.group(0), P_NER) for m in re.finditer(pattern, text)]

    # Presidio: one member name, and a trail name only it finds. PII-Tracer: every member mention, and a
    # product name it shouldn't have flagged.
    monkeypatch.setattr(PresidioDetector, "__init__", lambda self, settings: None)
    monkeypatch.setattr(
        PresidioDetector,
        "__call__",
        lambda self, t: spans(t, "Alice Moreau", "PERSON") + spans(t, "Mount Tam", "LOCATION"),
    )
    monkeypatch.setattr(
        pii_tracer.PiiTracer,
        "find",
        lambda self, t: [
            Found(s.start, s.end, "private_person")
            for s in spans(t, r"Alice Moreau|Bram Okafor|Chen Wei|\bAlice\b|\bBram\b|\bChen\b|stove", "PERSON")
        ],
    )
    out = tmp_path / "private" / "report.md"

    code = main(["redaction", "--db", str(tmp_path / "chat.sqlite"), "--config", str(config), "--out", str(out),
                 "--keep", "stove"])  # fmt: skip

    assert code == 0
    printed = capsys.readouterr()
    for chat_text in ("Mount Tam", "Alice", "Bram", "stove", "trailhead"):
        assert chat_text not in printed.out and chat_text not in printed.err
    report = _sections(out)

    def row(name: str) -> list[str]:
        line = next(line for line in printed.out.splitlines() if line.startswith(f"| {name} |"))
        return [cell.strip() for cell in line.split("|")[1:-1]]  # detector, spans, members, keep hits, only this

    tracer, presidio = row("pii_tracer"), row("presidio")
    assert tracer[2].endswith("(100%)") and int(tracer[3]) >= 1  # every member mention; the product name flagged
    assert not presidio[2].endswith("(100%)") and presidio[3] == "0"
    assert any(i["span"] == "Mount Tam" for i in report["presidio-only"]["items"])
    assert any(i["span"] == "stove" for i in report["pii_tracer-keep"]["items"])


def _span(start: int, end: int) -> Span:
    return Span(start, end, "PERSON", "", P_NER)


def test_overlap_is_not_coverage_but_adjacent_spans_can_add_up_to_it():
    full_name = _span(0, 12)  # "Alice Moreau"
    assert not covers([_span(0, 5)], full_name)  # "Alice" alone leaves the surname exposed
    assert not covers([_span(0, 5), _span(7, 12)], full_name)  # a one-character gap is still a gap
    assert covers([_span(0, 6), _span(6, 12)], full_name)  # touching spans cover it between them
    assert covers([_span(0, 8), _span(4, 12)], full_name)  # so do overlapping ones
    assert not covers([], full_name)


def test_a_span_reaching_beyond_the_other_detectors_find_is_listed_for_review(
    export_dir, tmp_path, monkeypatch, capsys
):
    source = DesktopJsonSource(export_dir)
    store = Store(tmp_path / "chat.sqlite")
    store.bind_chat(source.chat(), source.name, source.media_root())
    store.upsert_messages(source.messages())
    store.close()
    config = tmp_path / "config.toml"
    config.write_text('[endpoints.local]\napi = "openai"\nbase_url = "http://x"\ntrust = "private"\n'
                      '[roles.summarize]\nendpoint = "local"\nmodel = "m"\n')  # fmt: skip

    def spans(text: str, pattern: str) -> list[Span]:
        return [Span(m.start(), m.end(), "PERSON", m.group(0), P_NER) for m in re.finditer(pattern, text)]

    # Presidio finds "Alice" but not "Alice Moreau", and "Mount" but not "Mount Tam"; PII-Tracer finds both
    # in full. The member name is covered by the always-on member rule; the place isn't, so it's listed.
    monkeypatch.setattr(PresidioDetector, "__init__", lambda self, settings: None)
    monkeypatch.setattr(PresidioDetector, "__call__", lambda self, t: spans(t, r"\bAlice\b|\bMount\b"))
    monkeypatch.setattr(
        pii_tracer.PiiTracer,
        "find",
        lambda self, t: [Found(s.start, s.end, "private_address") for s in spans(t, "Alice Moreau|Mount Tam")],
    )
    out = tmp_path / "report.html"
    assert main(["redaction", "--db", str(tmp_path / "chat.sqlite"), "--config", str(config), "--out", str(out)]) == 0

    rows = {line.split("|")[1].strip(): line for line in capsys.readouterr().out.splitlines() if line.startswith("| ")}
    review = _sections(out)["pii_tracer-only"]["items"]
    assert {"span": "Mount Tam", "note": "partly covered by others"}.items() <= next(
        i for i in review if i["span"] == "Mount Tam"
    ).items()
    assert not any(i["span"] == "Alice Moreau" for i in review)  # fully covered by the member-name rule
    # Presidio's "Alice" inside "Alice Moreau" isn't a caught member mention.
    caught, total = map(int, re.search(r"(\d+)/(\d+)", rows["presidio"]).groups())
    assert caught < total


def _sections(path) -> dict[str, dict]:
    """The report's embedded data, by section id."""
    page = path.read_text(encoding="utf-8")
    payload = page.split("const DATA = ", 1)[1].split(";\nconst KEY", 1)[0]
    return {s["id"]: s for s in json.loads(payload)["sections"]}


def test_the_report_page_cannot_be_broken_or_scripted_by_chat_text(tmp_path):
    import argparse

    from tg_summary.redaction_eval import _Tally, _write_report

    hostile = '</script><img src=x onerror="alert(1)"><!--'
    tally = _Tally(spans=1)
    tally.only_this.append(("PERSON", "", f"see {hostile} ", "Alice", " </SCRIPT> end"))
    out = tmp_path / "report.html"
    _write_report(out, argparse.Namespace(sample=50), 1, 1, 1, {"presidio": tally})

    page = out.read_text(encoding="utf-8")
    assert page.lower().count("</script>") == 1  # only the page's own closing tag
    assert "<img" not in page and "<!--" not in page.split("<script>", 1)[1]
    item = _sections(out)["presidio-only"]["items"][0]
    assert (item["before"], item["span"], item["after"]) == (f"see {hostile} ", "Alice", " </SCRIPT> end")
