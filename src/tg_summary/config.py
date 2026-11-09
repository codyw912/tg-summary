"""TOML config: endpoints (where models live and how far they are trusted) and roles (which model does what)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

Api = Literal["openai", "anthropic"]
Trust = Literal["private", "external"]
Structured = Literal["json_schema", "prompt"]
Effort = Literal["minimal", "low", "medium", "high", "xhigh", "max"]
EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh", "max")

# PII-Tracer's labels (kept here so the config validates without the optional install).
PII_TRACER_LABELS = (
    "private_person",
    "private_email",
    "private_phone",
    "private_address",
    "private_url",
    "private_date",
    "account_number",
    "secret",
    "other_pii",
)
PII_TRACER_DEFAULT_LABELS = tuple(label for label in PII_TRACER_LABELS if label != "private_date")

DEFAULT_ENTITIES = [
    "PERSON",
    "LOCATION",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "IP_ADDRESS",
    "CREDIT_CARD",
    "IBAN_CODE",
    "URL",
]


@dataclass(frozen=True)
class Endpoint:
    name: str
    api: Api
    trust: Trust  # "private" may see raw chat text; "external" gets chat text redacted (media and URLs go as-is)
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        _check(self.api, ("openai", "anthropic"), f"endpoints.{self.name}.api")
        _check(self.trust, ("private", "external"), f"endpoints.{self.name}.trust")
        if self.api == "openai" and not self.base_url:
            raise ValueError(f"endpoints.{self.name}.base_url is required for api = 'openai'")


@dataclass(frozen=True)
class Role:
    endpoint: Endpoint
    model: str
    context_tokens: int = 32768
    max_output_tokens: int = 4096  # output limit when thinking is off
    structured: Structured = "json_schema"
    temperature: float | None = 0.2  # None sends none (Claude 5.x rejects any non-default value)
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    concurrency: int = 1
    # None leaves the server/provider default in place; true/false toggle it; a level also sets the effort.
    thinking: bool | Effort | None = None
    # Extra headroom added to max_tokens when `thinking` is on. It is one shared limit, so reasoning can still use
    # all of it; only Anthropic models that take a fixed budget get this as an enforced budget_tokens cap.
    thinking_tokens: int = 8192
    extra_body: dict[str, Any] = field(default_factory=dict)  # merged into every request body as-is
    extra_headers: dict[str, str] = field(default_factory=dict)
    max_calls: int | None = None  # per run; items beyond it keep their placeholder

    def __post_init__(self) -> None:
        _check(self.structured, ("json_schema", "prompt"), "roles.*.structured")
        if not isinstance(self.thinking, bool) and self.thinking is not None:
            _check(self.thinking, EFFORTS, "roles.*.thinking (or true/false)")
        if self.thinking_tokens < 1024:
            raise ValueError("roles.*.thinking_tokens must be at least 1024 (Anthropic's minimum thinking budget)")
        if self.presence_penalty is not None and self.endpoint.api == "anthropic":
            raise ValueError("roles.*.presence_penalty is not supported by the Anthropic API")

    @property
    def thinks(self) -> bool:
        return self.thinking is not None and self.thinking is not False

    @property
    def output_limit(self) -> int:
        """max_tokens per request: max_output_tokens, plus thinking_tokens of headroom when thinking is on.

        Thinking counts against max_tokens on every server tested. The headroom makes truncation less likely but
        does not reserve room for the answer. `thinking = None` adds nothing, because the setting is unknown."""
        return self.max_output_tokens + (self.thinking_tokens if self.thinks else 0)


@dataclass(frozen=True)
class SummarizeSettings:
    chunk_tokens: int = 8000  # upper bound per chunk; smaller chunks keep more detail
    gap_minutes: int = 120  # silences longer than this are marked as conversation breaks


@dataclass(frozen=True)
class ReportSettings:
    # List every processed photo, video, voice note and link with its description, transcript or summary.
    media_appendix: bool = True
    # "Bangers": the most-reacted messages of the period. 0 turns the section off.
    bangers: int = 5
    bangers_min_reactions: int = 3  # a message needs at least this many reactions in total to be listed


@dataclass(frozen=True)
class RedactionSettings:
    spacy_model: str = "en_core_web_md"
    language: str = "en"
    score_threshold: float = 0.35  # Presidio scores phone numbers without nearby context words at 0.4
    entities: list[str] = field(default_factory=lambda: list(DEFAULT_ENTITIES))
    allow: list[str] = field(default_factory=lambda: ["Deleted Account"])  # never redacted
    # Statistical detectors, on top of member names, rules and known values: "presidio", "pii_tracer" (optional
    # install: uv sync --extra pii).
    detectors: list[str] = field(default_factory=lambda: ["presidio"])
    # PII-Tracer labels to redact. Plain dates are left out by default: they matter to summaries.
    pii_tracer_labels: list[str] = field(default_factory=lambda: list(PII_TRACER_DEFAULT_LABELS))
    pii_tracer_device: str = "auto"  # "auto" picks the Apple GPU (mps), then CUDA, then the CPU

    def __post_init__(self) -> None:
        for d in self.detectors:
            _check(d, ("presidio", "pii_tracer"), "redaction.detectors")
        for label in self.pii_tracer_labels:
            _check(label, PII_TRACER_LABELS, "redaction.pii_tracer_labels")
        _check(self.pii_tracer_device, ("auto", "mps", "cuda", "cpu"), "redaction.pii_tracer_device")


OPTIONAL_ROLES = ("transcribe", "describe_media", "read_link", "digest", "judge")

# What happens to each media kind in the media and link stage. Kinds not listed here (polls, locations, ...)
# already render as text.
MediaAction = Literal["describe", "transcribe", "skip"]
_ACTIONS: dict[str, tuple[str, ...]] = {
    "photo": ("describe", "skip"),
    "video": ("describe", "transcribe", "skip"),
    "animation": ("describe", "skip"),
    "video_message": ("describe", "transcribe", "skip"),
    "voice": ("transcribe", "skip"),
    "audio": ("transcribe", "skip"),
    "document": ("describe", "skip"),  # images and PDFs only; other files are skipped
    "sticker": ("describe", "skip"),  # static .webp only; the emoji is already in the text
}
MEDIA_KINDS = tuple(_ACTIONS)


@dataclass(frozen=True)
class MediaSettings:
    photo: MediaAction = "describe"
    video: MediaAction = "describe"  # key frames + transcript unless the describe model accepts video
    animation: MediaAction = "describe"
    video_message: MediaAction = "transcribe"  # round video notes are mostly someone talking
    voice: MediaAction = "transcribe"
    audio: MediaAction = "skip"  # usually music
    document: MediaAction = "skip"
    sticker: MediaAction = "skip"
    frames: int = 4  # key frames per video when the describe model takes images only

    def __post_init__(self) -> None:
        for kind, allowed in _ACTIONS.items():
            _check(getattr(self, kind), allowed, f"media.{kind}")
        if self.frames < 1:
            raise ValueError("media.frames must be at least 1")

    def action(self, kind: str) -> str:
        return getattr(self, kind) if kind in _ACTIONS else "skip"


@dataclass(frozen=True)
class LinkSettings:
    enabled: bool = True
    # URL prefixes (without scheme and "www.") that are never sent anywhere: private message links and invite
    # links, which nobody outside the group can open.
    skip: list[str] = field(
        default_factory=lambda: [
            "t.me/c/",
            "t.me/+",
            "t.me/joinchat/",
            "telegram.me/c/",
            "telegram.me/+",
            "telegram.me/joinchat/",
        ]
    )
    # Per-site handling, matched by domain (subdomains included; the longest match wins; "*" is the default):
    # read = open and summarize in 1-3 sentences; brief = one fetch, one sentence (title, author or channel,
    # topic); skip = never sent. YouTube in brief mode fetches the video's oEmbed metadata.
    rules: dict[str, str] = field(default_factory=lambda: {"youtube.com": "brief", "youtu.be": "brief"})
    page_tokens: int = 20000  # most page text a read fetch puts in front of the model
    brief_page_tokens: int = 2000  # most page text a brief fetch puts in front of the model

    def __post_init__(self) -> None:
        for domain, mode in self.rules.items():
            _check(mode, ("read", "brief", "skip"), f"links.rules.{domain!r}")


@dataclass(frozen=True)
class Config:
    summarize: Role
    summarize_settings: SummarizeSettings
    redaction: RedactionSettings
    roles: dict[str, Role] = field(default_factory=dict)  # configured optional roles, keyed by OPTIONAL_ROLES name
    media: MediaSettings = field(default_factory=MediaSettings)
    links: LinkSettings = field(default_factory=LinkSettings)
    report: ReportSettings = field(default_factory=ReportSettings)
    endpoints: dict[str, Endpoint] = field(default_factory=dict)  # every configured endpoint, by name


def load_config(path: Path) -> Config:
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    _only(raw, {"endpoints", "roles", "summarize", "redaction", "media", "links", "report"}, "top level")
    endpoints = {
        name: Endpoint(name=name, **_kwargs(Endpoint, body, f"endpoints.{name}", skip=frozenset({"name"})))
        for name, body in raw.get("endpoints", {}).items()
    }
    roles = raw.get("roles", {})
    _only(roles, {"summarize", *OPTIONAL_ROLES}, "roles")
    if "summarize" not in roles:
        raise ValueError("config needs a [roles.summarize] section")
    return Config(
        summarize=_role(roles["summarize"], endpoints, "roles.summarize"),
        summarize_settings=SummarizeSettings(**_kwargs(SummarizeSettings, raw.get("summarize", {}), "summarize")),
        redaction=RedactionSettings(**_kwargs(RedactionSettings, raw.get("redaction", {}), "redaction")),
        roles={name: _role(roles[name], endpoints, f"roles.{name}") for name in OPTIONAL_ROLES if name in roles},
        media=MediaSettings(**_kwargs(MediaSettings, raw.get("media", {}), "media")),
        links=LinkSettings(**_kwargs(LinkSettings, raw.get("links", {}), "links")),
        report=ReportSettings(**_kwargs(ReportSettings, raw.get("report", {}), "report")),
        endpoints=endpoints,
    )


def _role(body: dict[str, Any], endpoints: dict[str, Endpoint], where: str) -> Role:
    body = dict(body)
    name = body.pop("endpoint", None)
    if name not in endpoints:
        raise ValueError(f"{where}.endpoint must name one of: {', '.join(endpoints) or '(no endpoints defined)'}")
    return Role(endpoint=endpoints[name], **_role_fields(body, where))


def override_role(role: Role, body: dict[str, Any], where: str, endpoints: dict[str, Endpoint]) -> Role:
    """`role` with the fields in `body` replaced. `endpoint` may name another configured endpoint."""
    body = dict(body)
    name = body.pop("endpoint", None)
    if name is not None and name not in endpoints:
        raise ValueError(f"{where}.endpoint must name one of: {', '.join(endpoints) or '(no endpoints defined)'}")
    endpoint = endpoints[name] if name is not None else role.endpoint
    return replace(role, endpoint=endpoint, **_role_fields(body, where))


# TOML has no null, so `"default"` stands for "send nothing; keep the server's own default".
_NULLABLE = frozenset({"temperature", "top_p", "top_k", "presence_penalty", "thinking"})


def _role_fields(body: dict[str, Any], where: str) -> dict[str, Any]:
    body = _kwargs(Role, body, where, skip=frozenset({"endpoint"}))
    return {k: None if k in _NULLABLE and v == "default" else v for k, v in body.items()}


def _kwargs(cls: Any, body: dict[str, Any], where: str, skip: frozenset[str] = frozenset()) -> dict[str, Any]:
    _only(body, {f.name for f in fields(cls)} - skip, where)
    return body


def _only(body: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(body) - allowed
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {', '.join(sorted(unknown))}")


def _check(value: str, allowed: tuple[str, ...], where: str) -> None:
    if value not in allowed:
        raise ValueError(f"{where} must be one of {allowed}, got {value!r}")
