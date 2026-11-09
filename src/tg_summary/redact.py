"""Reversible redaction for anything sent to an `external` endpoint.

Two lookups, deliberately separate:
- The vault (SQLite, per chat, persistent) gives each detected value a stable number, so `[PERSON_3]` is the
  same person across requests and runs. It is never used to restore text.
- The request map (one per hosted request) holds only tokens emitted into that request's payload, plus escaped
  source text that already looked like a token. Rehydration consults only this map, in one regex pass, so an
  invented or replayed token can never restore an identity that was not part of the request.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from tg_summary.config import RedactionSettings

_KIND = r"[A-Z]+(?:_[A-Z]+)*"
# Models sometimes drop the brackets ("PERSON_1"). This also matches the bare form, as a whole word only.
_LOOSE_RE = re.compile(rf"\[{_KIND}_\d+\]|(?<![\w\[]){_KIND}_\d+(?![\w\]])")
LITERAL = "LITERAL"

# Detector priorities: escaped literals beat exact rules, exact rules beat statistical NER.
P_LITERAL, P_RULE, P_NER = 3, 2, 1

_URL_RE = re.compile(r"(?:https?://|www\.|t\.me/)[^\s<>()\"']+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_HANDLE_RE = re.compile(r"(?<![\w@./])@[A-Za-z][A-Za-z0-9_]{3,31}(?!\w)")
_TRAILING_PUNCT = ".,;:!?)]}'\""
# End of a name match: not inside a word, and not the first half of an "n't" contraction ("don't", "won't",
# "can't", "haven't"), where a word ending in "n" can equal a first name. Everything else still matches,
# including "Alice'll", "Don Draper'd" and possessives like "Don's". \u2019 is the typographic apostrophe.
_NAME_END = r"(?!\w)(?!(?<=[nN])['\u2019]t\b)"


def find_urls(text: str) -> list[tuple[int, int, str]]:
    """(start, end, url) for http(s), www. and t.me links, without trailing punctuation."""
    found = []
    for m in _URL_RE.finditer(text):
        url = m.group(0).rstrip(_TRAILING_PUNCT)
        found.append((m.start(), m.start() + len(url), url))
    return found


_PRESIDIO_KINDS = {
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "PHONE_NUMBER": "PHONE",
    "EMAIL_ADDRESS": "EMAIL",
    "IP_ADDRESS": "IP",
    "CREDIT_CARD": "CARD",
    "IBAN_CODE": "IBAN",
    "URL": "URL",
}

# Kinds this redactor always emits. Configured Presidio entities can add more (US_SSN, ...); those count as
# soon as a request emits them. A bare look-alike of a counted kind is treated as a placeholder; other
# identifier-shaped text such as HTTP_2 or COVID_19 is ordinary text.
KNOWN_KINDS = frozenset({*_PRESIDIO_KINDS.values(), "HANDLE", LITERAL})


def _bracketed(found: str) -> str:
    return found if found.startswith("[") else f"[{found}]"


def _placeholder_like(found: str, kinds: frozenset[str]) -> bool:
    """Bracketed token-shaped text always counts; bare text only when it names a counted kind."""
    return found.startswith("[") or found.rsplit("_", 1)[0] in kinds


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    kind: str
    value: str  # canonical value restored on rehydration (e.g. full name for a first-name match)
    priority: int
    # Identity in the vault, when it isn't the kind's usual normalization: a merged region that reaches beyond
    # its strongest span is identified by its whole text (a PHONE's digits-only form would make "Alice 1234"
    # equal "Bob 1234"), and vault matches carry the identity stored with them.
    norm: str | None = None


Detector = Callable[[str], Iterable[Span]]
Assign = Callable[[str, str, str], int]  # (kind, normalized value, value) -> stable number


class RequestMap:
    """Tokens emitted into one hosted request. Only these are restored from its response."""

    def __init__(self) -> None:
        self.tokens: dict[str, str] = {}
        self._literals: dict[str, str] = {}

    def literal(self, source: str) -> str:
        if source not in self._literals:
            token = f"[{LITERAL}_{len(self._literals) + 1}]"
            self._literals[source] = token
            self.tokens[token] = source
        return self._literals[source]

    def kinds(self) -> frozenset[str]:
        """Every kind this request emits, on top of the ones the redactor always emits."""
        return KNOWN_KINDS | {token[1:].rsplit("_", 1)[0] for token in self.tokens}


class Redactor:
    def __init__(self, assign: Assign, detectors: list[Detector], allow: Iterable[str] = ()):
        self._assign = assign
        self._detectors = detectors
        self._allow = {a.casefold() for a in allow}
        self.counts: Counter[str] = Counter()

    def redact(self, text: str, request: RequestMap, protected: Iterable[tuple[int, int]] = ()) -> str:
        """`protected`: (start, end) offsets the renderer wrote as format (a `Chunk.format`); detections there are
        dropped. Only those exact offsets: text that merely looks like format (in a message or a caption) is
        redacted like any other text."""
        found = outside_format(text, [s for detect in self._detectors for s in detect(text)], protected)
        detected = [s for s in found if s.value.casefold() not in self._allow]
        # Escape look-alikes of every kind this request can emit, including configured ones like US_SSN, so a
        # literal "US_SSN_1" in chat text can never be restored to the value behind a real [US_SSN_1].
        kinds = request.kinds() | {s.kind for s in detected}
        spans = [
            Span(m.start(), m.end(), LITERAL, m.group(0), P_LITERAL)
            for m in _LOOSE_RE.finditer(text)
            if _placeholder_like(m.group(0), kinds)
        ]
        spans.extend(detected)

        out, pos = [], 0
        for span in _covering(spans, text):
            out.append(text[pos : span.start])
            out.append(self._token(span, request))
            pos = span.end
        out.append(text[pos:])
        return "".join(out)

    def _token(self, span: Span, request: RequestMap) -> str:
        if span.kind == LITERAL:
            return request.literal(span.value)
        norm = span.norm if span.norm is not None else _normalize(span.kind, span.value)
        n = self._assign(span.kind, norm, span.value)
        token = f"[{span.kind}_{n}]"
        request.tokens[token] = span.value
        self.counts[span.kind] += 1
        return token


def rehydrate(text: str, request: RequestMap, unmatched: list[str]) -> str:
    """One pass over bracketed and bare tokens, restoring only those emitted in this request.

    Restored values are never rescanned, and whole-token matching keeps `[PERSON_1]` from matching inside
    `[PERSON_10]`. A bare `PERSON_1` is restored only if `[PERSON_1]` was part of this request.
    """
    kinds = request.kinds()

    def swap(m: re.Match[str]) -> str:
        found = m.group(0)
        token = _bracketed(found)
        if token in request.tokens:
            return request.tokens[token]
        if _placeholder_like(found, kinds):
            unmatched.append(found)
        return found

    return _LOOSE_RE.sub(swap, text)


def rehydrate_tree(node: Any, request: RequestMap, unmatched: list[str]) -> Any:
    """Rehydrate every string in parsed JSON. Done after parsing so restored quotes can't break the JSON."""
    if isinstance(node, str):
        return rehydrate(node, request, unmatched)
    if isinstance(node, list):
        return [rehydrate_tree(v, request, unmatched) for v in node]
    if isinstance(node, dict):
        return {k: rehydrate_tree(v, request, unmatched) for k, v in node.items()}
    return node


# --- detectors -------------------------------------------------------------


class RosterDetector:
    """Group members' display names. Catches nicknames and first names that NER misses.

    Matching is case-insensitive on purpose: chat text is often lowercase, and over-redaction is the safe
    failure. Add false positives (a member called "Will") to `redaction.allow`.
    """

    def __init__(self, names: Iterable[str], allow: Iterable[str] = ()):
        allowed = {a.casefold() for a in allow}
        canon = sorted({n.strip() for n in names if n and n.strip() and n.strip().casefold() not in allowed})
        aliases: dict[str, str] = {}
        firsts = Counter(f for n in canon if (f := _first_name(n)))
        for name in canon:
            aliases.setdefault(name.casefold(), name)
            words = _words(name)
            if not words:
                continue
            aliases.setdefault(" ".join(words).casefold(), name)
            first = _first_name(name)
            if first and len(first) >= 3 and firsts[first] == 1 and first not in allowed:
                aliases.setdefault(first, name)
        self._aliases = aliases
        ordered = sorted(aliases, key=len, reverse=True)
        self._re = (
            re.compile(r"(?<!\w)(?:" + "|".join(re.escape(a) for a in ordered) + r")" + _NAME_END, re.IGNORECASE)
            if ordered
            else None
        )

    def __call__(self, text: str) -> Iterable[Span]:
        if self._re is None:
            return []
        return [
            Span(m.start(), m.end(), "PERSON", self._aliases[m.group(0).casefold()], P_RULE)
            for m in self._re.finditer(text)
        ]


def rule_detector(text: str) -> Iterable[Span]:
    spans = []
    for start, end, url in find_urls(text):
        spans.append(Span(start, end, "URL", url, P_RULE))
    for m in _EMAIL_RE.finditer(text):
        spans.append(Span(m.start(), m.end(), "EMAIL", m.group(0), P_RULE))
    for m in _HANDLE_RE.finditer(text):
        spans.append(Span(m.start(), m.end(), "HANDLE", m.group(0), P_RULE))
    return spans


class VaultDetector:
    """Every value redacted before is redacted again, wherever it appears.

    NER depends on context: a place recognized in a chat line can go unnoticed inside JSON notes. Matching
    known values exactly closes that gap for later requests in the same run and in future runs.
    """

    def __init__(self, known: Callable[[], list[tuple[str, str, str]]]):
        self._known = known
        self._size = -1
        self._re: re.Pattern[str] | None = None
        self._kinds: dict[str, tuple[str, str, str]] = {}

    def __call__(self, text: str) -> Iterable[Span]:
        rows = self._known()
        if len(rows) != self._size:  # the vault only grows
            self._size = len(rows)
            self._kinds = {value.casefold(): (kind, norm, value) for kind, norm, value in rows if value.strip()}
            ordered = sorted(self._kinds, key=len, reverse=True)
            self._re = (
                re.compile(r"(?<!\w)(?:" + "|".join(re.escape(v) for v in ordered) + r")" + _NAME_END, re.IGNORECASE)
                if ordered
                else None
            )
        if self._re is None:
            return []
        spans = []
        for m in self._re.finditer(text):
            kind, norm, value = self._kinds[m.group(0).casefold()]
            spans.append(Span(m.start(), m.end(), kind, value, P_RULE, norm))
        return spans


class PresidioDetector:
    def __init__(self, settings: RedactionSettings):
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": settings.language, "model_name": settings.spacy_model}],
            }
        )
        self._engine = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=[settings.language])
        self._settings = settings

    def __call__(self, text: str) -> Iterable[Span]:
        results = self._engine.analyze(
            text=text,
            language=self._settings.language,
            entities=self._settings.entities,
            score_threshold=self._settings.score_threshold,
        )
        return [Span(r.start, r.end, _kind(r.entity_type), text[r.start : r.end], P_NER) for r in results]


class PiiTracerDetector:
    """PII-Tracer, a local model trained on conversations (optional install: `uv sync --extra pii`)."""

    def __init__(self, settings: RedactionSettings):
        from tg_summary.pii_tracer import PiiTracer

        self._model = PiiTracer(settings.pii_tracer_runtime, settings.pii_tracer_device)
        self._labels = settings.pii_tracer_labels

    def __call__(self, text: str) -> Iterable[Span]:
        from tg_summary.pii_tracer import spans_for

        found = self._model.find(text)
        return [Span(s, e, kind, text[s:e], P_NER) for s, e, kind in spans_for(text, found, self._labels)]


def build_redactor(
    assign: Assign,
    known: Callable[[], list[tuple[str, str, str]]],
    member_names: Iterable[str],
    settings: RedactionSettings,
) -> Redactor:
    detectors: list[Detector] = [
        RosterDetector(member_names, settings.allow),
        rule_detector,
        VaultDetector(known),
    ]
    if "presidio" in settings.detectors:
        detectors.append(PresidioDetector(settings))
    if "pii_tracer" in settings.detectors:
        detectors.append(PiiTracerDetector(settings))
    return Redactor(assign, detectors, settings.allow)


# --- helpers ---------------------------------------------------------------


def outside_format(text: str, spans: list[Span], protected: Iterable[tuple[int, int]]) -> list[Span]:
    """Spans with the renderer's format offsets cut out (message stamps, reply references, gap markers): that
    text never holds PII, and redacting it would break the message ids summaries cite. A span that was all
    format disappears; one that straddles format and chat text keeps its chat-text part(s)."""
    ranges = list(protected)
    if not ranges:
        return spans
    out = []
    for s in spans:
        pieces = [(s.start, s.end)]
        for a, b in ranges:
            if a < s.end and s.start < b:
                pieces = [p for x, y in pieces for p in ((x, min(y, a)), (max(x, b), y)) if p[0] < p[1]]
        if pieces == [(s.start, s.end)]:
            out.append(s)
            continue
        for x, y in pieces:
            while x < y and text[x].isspace():
                x += 1
            while y > x and text[y - 1].isspace():
                y -= 1
            if x < y:
                out.append(Span(x, y, s.kind, text[x:y], s.priority))
    return out


def _covering(spans: list[Span], text: str) -> list[Span]:
    """One span per group of overlapping spans, covering all of them, so no character any detector flagged is
    sent. Dropping one of two crossing spans would expose the part only it covered.

    The merged span takes its kind from its strongest member (highest priority, then longest). It keeps that
    member's canonical value when it covers exactly that member; otherwise its value is the covered text, so
    restoring returns the original wording."""
    out: list[Span] = []
    group: list[Span] = []
    end = 0
    for span in sorted((s for s in spans if s.end > s.start), key=lambda s: s.start):
        if group and span.start >= end:  # touching is not overlapping
            out.append(_merge(group, end, text))
            group = []
        end = max(end, span.end) if group else span.end
        group.append(span)
    if group:
        out.append(_merge(group, end, text))
    return out


def _merge(group: list[Span], end: int, text: str) -> Span:
    start = group[0].start
    best = max(group, key=lambda s: (s.priority, s.end - s.start, -s.start))
    if (best.start, best.end) == (start, end):
        return best
    value = text[start:end]
    return Span(start, end, best.kind, value, best.priority, " ".join(value.split()).casefold())


def _normalize(kind: str, value: str) -> str:
    if kind == "PHONE":
        return re.sub(r"\D", "", value)
    if kind == "URL":
        return value
    return " ".join(value.split()).casefold()


def _words(name: str) -> list[str]:
    return re.findall(r"\w+", name)


def _first_name(name: str) -> str | None:
    """The first space-separated word of a multi-word display name, casefolded; None for one-word names.

    Splitting on spaces only keeps handles such as "deploy-notify-bot" from yielding "deploy" as a first name,
    which would redact that ordinary word everywhere."""
    parts = name.split()
    if len(parts) < 2:
        return None
    first = re.sub(r"^\W+|\W+$", "", parts[0]).casefold()
    return first if re.search(r"[^\W\d_]", first) else None


def _kind(entity_type: str) -> str:
    return _PRESIDIO_KINDS.get(entity_type) or re.sub(r"[^A-Z]+", "_", entity_type.upper()).strip("_") or "PII"
