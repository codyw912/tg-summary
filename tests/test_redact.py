from __future__ import annotations

from pathlib import Path

from tg_summary.redact import (
    P_NER,
    P_RULE,
    Redactor,
    RequestMap,
    RosterDetector,
    Span,
    VaultDetector,
    rehydrate,
    rule_detector,
)
from tg_summary.store import Store


def _redactor(tmp_path: Path, names=("Alice Moreau", "Bram Okafor")) -> Redactor:
    store = Store(tmp_path / "chat.sqlite")
    return Redactor(store.vault_assign, [RosterDetector(names), rule_detector])


def _fixed(*spans: tuple[int, int, str, int]):
    """A detector that always reports the given (start, end, kind, priority) spans."""
    return lambda text: [Span(s, e, kind, text[s:e], p) for s, e, kind, p in spans]


def test_the_renderers_format_is_never_redacted_but_look_alikes_in_chat_text_are(tmp_path):
    import re
    from datetime import UTC, datetime, timedelta

    from tg_summary.models import Message
    from tg_summary.segment import build_chunks

    first = datetime(2026, 10, 28, 20, 48, tzinfo=UTC)
    body = "call me on ↪#4155550199 or quote #66651 2026-10-28 20:48 back to me"  # format look-alikes typed by a person
    msgs = [
        Message(id=66651, date=first, sender_id="u1", sender="Bob Smith", text=body),
        Message(id=66652, date=first + timedelta(hours=3), sender_id="u2", sender="Ann", text="ok", reply_to=66651),
    ]
    chunk = build_chunks(msgs, {m.id: m for m in msgs}, 10_000, timedelta(hours=1))[0]
    text = chunk.text
    numbers = lambda t: [Span(m.start(), m.end(), "ACCOUNT", m.group(0), P_NER) for m in re.finditer(r"\d+", t)]
    header = lambda t: [Span(0, t.index("Bob Smith") + 9, "PERSON", t[: t.index("Bob Smith") + 9], P_NER)]
    store = Store(tmp_path / "chat.sqlite")
    r = Redactor(store.vault_assign, [VaultDetector(store.vault_values), numbers, header])
    request = RequestMap()
    sent = r.redact(text, request, chunk.format)

    assert len(chunk.format) == 4  # two message stamps, one reply reference, one gap marker
    for a, b in chunk.format:
        assert text[a:b] in sent  # generated ids, timestamps, "↪#66651", "--- 3h later ---" intact
    assert "Bob Smith" not in sent and "4155550199" not in sent  # the sender and the number typed in the body
    assert sent.count("66651") == 2  # the stamp and the reply reference; the typed "#66651" in the body is hidden
    assert rehydrate(sent, request, []) == text

    # A caption sent on its own has no renderer offsets, so nothing in it is exempt: the stored phone number is
    # redacted even behind a reply-reference look-alike.
    caption = Redactor(store.vault_assign, [VaultDetector(store.vault_values)])
    assert "4155550199" not in caption.redact("my phone is ↪#4155550199", RequestMap())


def test_crossing_spans_merge_so_nothing_either_detector_flagged_is_sent(tmp_path):
    text = "call Bram Okafor Smith tomorrow"  # one detector flags "Bram Okafor", another "Okafor Smith"
    detectors = [_fixed((5, 16, "PERSON", P_NER)), _fixed((10, 22, "LOCATION", P_NER))]
    r = Redactor(Store(tmp_path / "chat.sqlite").vault_assign, detectors)
    request = RequestMap()
    sent = r.redact(text, request)
    assert sent == "call [LOCATION_1] tomorrow"  # the longer span names the kind; both are covered
    assert rehydrate(sent, request, []) == text


def test_merged_regions_with_the_same_phone_number_stay_distinct_across_requests(tmp_path):
    # A phone number (high priority) crossing a low-priority span on each side: both merged regions keep the
    # PHONE kind, but "Alice 1234" and "Bob 1234" must not share a placeholder, or one restores as the other.
    # The second request finds both through the vault, which must keep their identities apart too.
    store = Store(tmp_path / "chat.sqlite")
    text = "Alice 1234; Bob 1234"
    detectors = [
        VaultDetector(store.vault_values),
        _fixed((6, 10, "PHONE", P_RULE), (16, 20, "PHONE", P_RULE)),
        _fixed((0, 8, "PERSON", P_NER), (12, 18, "PERSON", P_NER)),
    ]
    r = Redactor(store.vault_assign, detectors)
    for _ in range(2):
        request = RequestMap()
        sent = r.redact(text, request)
        assert sent == "[PHONE_1]; [PHONE_2]"
        assert rehydrate(sent, request, []) == text


def test_a_name_inside_a_longer_lower_priority_span_covers_the_whole_span(tmp_path):
    # The roster matches "Alice" (high priority); a statistical detector flags "Alice Lindqvist-Moreau" (low).
    # Picking the roster match alone would send "Lindqvist-Moreau".
    store = Store(tmp_path / "chat.sqlite")
    text = "ask Alice Lindqvist-Moreau about it"
    r = Redactor(store.vault_assign, [RosterDetector(["Alice Moreau"]), _fixed((4, 26, "PERSON", P_NER))])
    request = RequestMap()
    sent = r.redact(text, request)
    assert sent == "ask [PERSON_1] about it"
    assert rehydrate(sent, request, []) == text

    # A low-priority span wholly inside a roster match keeps the roster's canonical value (the full name).
    request = RequestMap()
    r = Redactor(store.vault_assign, [RosterDetector(["Alice Moreau"]), _fixed((3, 8, "PERSON", P_NER))])
    assert r.redact("hi alice!", request) == "hi [PERSON_2]!"
    assert rehydrate("[PERSON_2]", request, []) == "Alice Moreau"


def test_tokens_are_stable_across_requests_and_first_names_share_the_full_name_token(tmp_path):
    r = _redactor(tmp_path)
    a = r.redact("Alice Moreau said hi", RequestMap())
    b = r.redact("then alice left", RequestMap())
    assert a == "[PERSON_1] said hi"
    assert b == "then [PERSON_1] left"


def test_rehydrate_restores_only_tokens_emitted_in_this_request(tmp_path):
    r = _redactor(tmp_path)
    r.redact("Bram Okafor is here", RequestMap())  # Bram gets a vault token in an earlier request
    request = RequestMap()
    sent = r.redact("Alice Moreau is here", request)
    bram_token = "[PERSON_1]"
    alice_token = sent.split()[0]
    assert alice_token != bram_token

    unmatched: list[str] = []
    out = rehydrate(f"{alice_token} met {bram_token} and [PERSON_99]", request, unmatched)

    assert out == f"Alice Moreau met {bram_token} and [PERSON_99]"
    assert unmatched == [bram_token, "[PERSON_99]"]


def test_token_shaped_source_text_round_trips_verbatim(tmp_path):
    r = _redactor(tmp_path)
    request = RequestMap()
    sent = r.redact("literal [PERSON_1] in chat, and Alice Moreau", request)
    assert sent == "literal [LITERAL_1] in chat, and [PERSON_1]"

    unmatched: list[str] = []
    assert rehydrate(sent, request, unmatched) == "literal [PERSON_1] in chat, and Alice Moreau"
    assert unmatched == []


def test_rehydrate_is_single_pass_and_matches_whole_tokens():
    request = RequestMap()
    request.tokens["[PERSON_1]"] = "[PERSON_10]"  # a restored value must not be expanded again
    request.tokens["[PERSON_10]"] = "Zed"
    unmatched: list[str] = []
    assert rehydrate("[PERSON_1] / [PERSON_10]", request, unmatched) == "[PERSON_10] / Zed"


def test_bare_placeholders_are_restored_only_when_sent_and_ordinary_identifiers_are_left_alone(tmp_path):
    r = _redactor(tmp_path)
    r.redact("Bram Okafor first", RequestMap())  # [PERSON_1] belongs to an earlier request
    request = RequestMap()
    assert r.redact("Alice Moreau", request) == "[PERSON_2]"

    unmatched: list[str] = []
    out = rehydrate("PERSON_2 met PERSON_1 over HTTP_2 during COVID_19; PERSON_2's plan", request, unmatched)

    assert out == "Alice Moreau met PERSON_1 over HTTP_2 during COVID_19; Alice Moreau's plan"
    assert unmatched == ["PERSON_1"]


def test_bare_look_alikes_in_chat_text_are_escaped_but_ordinary_identifiers_are_not(tmp_path):
    r = _redactor(tmp_path)
    request = RequestMap()
    sent = r.redact("the bug PERSON_1 is about HTTP_2, ask Alice Moreau", request)
    assert sent == "the bug [LITERAL_1] is about HTTP_2, ask [PERSON_1]"

    unmatched: list[str] = []
    assert rehydrate("PERSON_1 says LITERAL_1", request, unmatched) == "Alice Moreau says PERSON_1"
    assert unmatched == []


def test_look_alikes_of_configured_kinds_are_escaped_when_the_request_emits_that_kind(tmp_path):
    from tg_summary.redact import P_NER, Span

    def ssn(text: str):  # stands in for a configured Presidio entity such as US_SSN
        i = text.find("123-45-6789")
        return [Span(i, i + 11, "US_SSN", "123-45-6789", P_NER)] if i >= 0 else []

    store = Store(tmp_path / "chat.sqlite")
    r = Redactor(store.vault_assign, [ssn])
    request = RequestMap()
    sent = r.redact("ticket US_SSN_1 over HTTP_2, ssn 123-45-6789", request)
    assert sent == "ticket [LITERAL_1] over HTTP_2, ssn [US_SSN_1]"

    unmatched: list[str] = []
    out = rehydrate("US_SSN_1 / LITERAL_1 / HTTP_2 / US_SSN_7", request, unmatched)
    assert out == "123-45-6789 / US_SSN_1 / HTTP_2 / US_SSN_7"
    assert unmatched == ["US_SSN_7"]


def test_rules_catch_urls_emails_handles_without_trailing_punctuation(tmp_path):
    r = _redactor(tmp_path, names=())
    request = RequestMap()
    sent = r.redact("see https://example.org/a?b=1, mail x.y@example.com or ping @some_user.", request)
    assert sent == "see [URL_1], mail [EMAIL_1] or ping [HANDLE_1]."


def test_only_nt_contractions_spare_a_name_and_every_other_name_suffix_is_still_redacted(tmp_path):
    r = _redactor(tmp_path, names=("Don Draper", "Alice Moreau", "Haven Cole"))
    sent = r.redact(
        "I don't know, don\u2019t ask, we haven't. Don's idea; Don'll come, Alice'll join, "
        "Don Draper'd agreed, Alice\u2019ve been, ask don",
        RequestMap(),
    )
    assert sent == (
        "I don't know, don\u2019t ask, we haven't. [PERSON_1]'s idea; [PERSON_1]'ll come, [PERSON_2]'ll join, "
        "[PERSON_1]'d agreed, [PERSON_2]\u2019ve been, ask [PERSON_1]"
    )
    # Values remembered in the vault follow the same rule.
    from tg_summary.redact import VaultDetector

    store = Store(tmp_path / "vault.sqlite")
    store.vault_assign("PERSON", "won", "Won")
    v = Redactor(store.vault_assign, [VaultDetector(store.vault_values)])
    assert v.redact("I won't go; Won's call, Won'd say, Won'll see", RequestMap()) == (
        "I won't go; [PERSON_1]'s call, [PERSON_1]'d say, [PERSON_1]'ll see"
    )


def test_handle_style_names_give_no_first_name_alias_but_multi_word_names_do(tmp_path):
    r = _redactor(tmp_path, names=("deploy-notify-bot", "Alice Moreau"))
    sent = r.redact("new deploy out, posted by deploy-notify-bot; alice liked it", RequestMap())
    assert sent == "new deploy out, posted by [PERSON_1]; [PERSON_2] liked it"


def test_allowlisted_names_are_not_redacted(tmp_path):
    store = Store(tmp_path / "chat.sqlite")
    r = Redactor(store.vault_assign, [RosterDetector(["Will Smith"], allow=["will"])], allow=["will"])
    assert r.redact("I will ask Will Smith", RequestMap()) == "I will ask [PERSON_1]"


def test_a_value_detected_once_is_redacted_even_where_its_detector_misses_it(tmp_path):
    from tg_summary.redact import P_NER, Span, VaultDetector

    store = Store(tmp_path / "chat.sqlite")
    fired = []

    def ner_once(text: str):  # context-dependent NER: recognizes "Mount Tam" only the first time
        if fired:
            return []
        fired.append(True)
        i = text.index("Mount Tam")
        return [Span(i, i + 9, "LOCATION", "Mount Tam", P_NER)]

    r = Redactor(store.vault_assign, [VaultDetector(store.vault_values), ner_once])
    assert r.redact("hike at Mount Tam loop", RequestMap()) == "hike at [LOCATION_1] loop"
    assert r.redact('{"summary": "mount tam it is"}', RequestMap()) == '{"summary": "[LOCATION_1] it is"}'
