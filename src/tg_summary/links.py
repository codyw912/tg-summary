"""Links for `read_link`: which URLs a message carries, which are never sent, and whether a provider opened one.

Whether a link was opened is decided from the provider's own tool results, not from what the model says.
Pydantic AI labels a failed fetch `outcome="success"`; the failure is only visible in the raw result content,
whose type ends in `_tool_result_error` and carries an `error_code` (Anthropic: `web_fetch_tool_result_error`,
e.g. `url_not_allowed` for x.com).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from pydantic_ai import NativeToolCallPart, NativeToolReturnPart
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.native_tools import WebFetchTool

from tg_summary.redact import find_urls


def normalize(url: str) -> str:
    """Stable form for caching: https:// added when missing, host lowercased without "www.", no fragment,
    no utm_* parameters, no trailing slash."""
    if not re.match(r"[a-z][a-z0-9+.-]*://", url, re.IGNORECASE):
        url = "https://" + url
    parts = urlsplit(url)
    host = (parts.hostname or "").removeprefix("www.")
    netloc = host + (f":{parts.port}" if parts.port else "")
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not k.startswith("utm_")])
    return urlunsplit((parts.scheme.lower(), netloc, parts.path.rstrip("/"), query, ""))


def bare(url: str) -> str:
    """Normalized URL without its scheme, for skip patterns and display."""
    return normalize(url).split("://", 1)[1]


def message_urls(text: str) -> list[str]:
    """Distinct normalized URLs in a message, in order of appearance."""
    return list(dict.fromkeys(normalize(url) for _, _, url in find_urls(text)))


def skipped(url: str, patterns: Iterable[str]) -> bool:
    """Whether `url` starts with a skip pattern (host and path, no scheme or "www."). The port and query are
    ignored, so `https://t.me:443/c/1/2` and `https://t.me/c/1/2` are the same private link."""
    parts = urlsplit(normalize(url))
    target = (host(url) + parts.path).casefold()
    return any(target.startswith(p.casefold().removeprefix("www.")) for p in patterns)


MODES = ("read", "brief", "skip")
_YOUTUBE = ("youtube.com", "youtu.be")


def host(url: str) -> str:
    """The URL's hostname: lowercase, without "www.", port, credentials, path or query."""
    return (urlsplit(normalize(url)).hostname or "").removeprefix("www.")


def _on(url: str, domain: str) -> bool:
    h, d = host(url), domain.casefold().removeprefix("www.")
    return h == d or h.endswith("." + d)


def mode_for(url: str, rules: dict[str, str]) -> str:
    """The rule for a URL's host: the longest matching domain (subdomains included), else `*`, else read."""
    matches = [(len(d), mode) for d, mode in rules.items() if d != "*" and _on(url, d)]
    return max(matches)[1] if matches else rules.get("*", "read")


def fetch_url(url: str, mode: str) -> str:
    """What the provider is asked to fetch. A YouTube page's text is nearly empty, so a brief look at a video
    fetches YouTube's public oEmbed metadata instead (title and channel, as JSON). The provider fetches it;
    this code never does."""
    if mode == "brief" and any(_on(url, d) for d in _YOUTUBE):
        return "https://www.youtube.com/oembed?format=json&url=" + quote(url, safe="")
    return url


@dataclass(frozen=True, slots=True)
class Fetch:
    url: str | None  # what the tool was asked to fetch, or reported fetching
    ok: bool
    error_code: str | None = None


def fetches(messages: list[ModelMessage], exchanges: list[tuple[str, str]]) -> list[Fetch]:
    """Every web fetch the provider reported running, paired with the URL it was asked for.

    Read from Pydantic AI's parts and from the raw Anthropic response bodies: after a `pause_turn`,
    Pydantic AI 2.66 keeps only the final segment's parts, so a fetch made before the pause is only in the
    raw body of the paused response.
    """
    asked: dict[str, str | None] = {}
    results: dict[str, dict] = {}
    for p in (p for m in messages if isinstance(m, ModelResponse) for p in m.parts):
        if isinstance(p, NativeToolCallPart) and p.tool_name == WebFetchTool.kind:
            asked[p.tool_call_id] = p.args_as_dict().get("url")
        elif isinstance(p, NativeToolReturnPart) and p.tool_name == WebFetchTool.kind:
            results[p.tool_call_id] = p.content if isinstance(p.content, dict) else {}
    for _, got in exchanges:
        try:
            blocks = json.loads(got).get("content")
        except (ValueError, AttributeError):
            continue
        for b in blocks if isinstance(blocks, list) else []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "server_tool_use" and b.get("name") == "web_fetch":
                asked.setdefault(b.get("id", ""), (b.get("input") or {}).get("url"))
            elif b.get("type") == "web_fetch_tool_result" and isinstance(b.get("content"), dict):
                results.setdefault(b.get("tool_use_id", ""), b["content"])
    found = []
    for call_id, content in results.items():
        kind = str(content.get("type", ""))
        url = asked.get(call_id) or content.get("url")
        if kind.endswith("_tool_result_error") or "error_code" in content:
            found.append(Fetch(url, ok=False, error_code=str(content.get("error_code") or kind or "error")))
        else:
            found.append(Fetch(url, ok=True))
    return found


def status(target: str, found: list[Fetch], text: str) -> tuple[str, str | None]:
    """(status, error code) for one link, from the fetches the provider reported.

    - opened: at least one successful fetch of this URL
    - failed: fetches were recorded but none of this URL succeeded, whatever the text says
    - unverified: no fetch evidence at all (providers that don't report tool calls), but a description came back
    """
    if not found:
        return ("unverified", None) if text.strip() else ("failed", "no_output")
    mine = [f for f in found if f.url and bare(f.url) == bare(target)]
    if any(f.ok for f in mine):
        return "opened", None
    errors = [f.error_code for f in mine if f.error_code] or [f.error_code for f in found if f.error_code]
    return "failed", errors[0] if errors else "url_not_fetched"


def raw_citations(exchanges: list[tuple[str, str]]) -> int:
    """Citations in the raw response bodies.

    Pydantic AI 2.66 merges Anthropic's text blocks into one part and drops their citations, so they are
    counted here: `citations` on Anthropic text blocks, `annotations` on OpenAI-style messages.
    """
    count = 0

    def walk(node: object) -> None:
        nonlocal count
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("citations", "annotations") and isinstance(value, list):
                    count += len(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for _, got in exchanges:
        try:
            body = json.loads(got)
        except ValueError:
            continue  # streamed or non-JSON bodies carry no countable citations
        walk(body)
    return count
