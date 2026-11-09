"""Model calls through Pydantic AI, over the two wire formats that cover local and hosted servers.

- `openai`: /chat/completions — LM Studio, oMLX, llama-server, vLLM, OpenAI, OpenRouter, ...
- `anthropic`: /messages — Anthropic (oMLX also speaks it)

Pydantic AI owns the wire formats, JSON output modes and the repair round. This module keeps what the project
depends on: the response cache, redaction before external calls, request-scoped rehydration, and an audit log
of the exact HTTP bodies sent to external endpoints. Conversation roles never pass tools.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields
from typing import Any, TypeVar

import httpx2
from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    ModelRetry,
    NativeOutput,
    NativeToolCallPart,
    NativeToolReturnPart,
    PromptedOutput,
    RunContext,
    TextPart,
    ToolCallPart,
    UnexpectedModelBehavior,
    UserContent,
    capture_run_messages,
)
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelRequestTimeout,
    ModelServerError,
    ModelTimeoutError,
    ModelUnavailableError,
)
from pydantic_ai.messages import ModelMessage, ModelResponse, ThinkingPart
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.native_tools import AbstractNativeTool
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from tg_summary.config import Role
from tg_summary.redact import Redactor, RequestMap, rehydrate, rehydrate_tree
from tg_summary.store import Store

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    def __init__(self, message: str, code: str = "error"):
        super().__init__(message)
        self.code = code  # short, for reports: "http_429", "truncated", "local_tool_requested", ...


@dataclass
class TextResult:
    text: str  # the final answer, placeholders restored
    messages: list[ModelMessage]  # every request and response of the run, provider tool calls included
    exchanges: list[tuple[str, str]]  # exact HTTP request and response bodies


class Truncated(LLMError):
    """The response hit max_tokens before the answer was complete."""

    def __init__(self, message: str = "") -> None:
        super().__init__(message or _TRUNCATED, code="truncated")


class Transient(LLMError):
    """A failure worth retrying: the server was unreachable, timed out, overloaded, rate limiting or erroring."""

    def __init__(self, message: str, code: str, retry_after: float | None = None) -> None:
        super().__init__(message, code)
        self.retry_after = retry_after


# Waits before each retry of a transient failure: about 8 minutes in all, long enough for a local server to
# restart (Splash recalibrates in about 1.5 minutes) or a hosted API to recover from overload.
RETRY_BACKOFF: tuple[float, ...] = (10, 30, 60, 120, 240)
_TRANSIENT_ERRORS = (
    ModelConnectionError,
    ModelServerError,
    ModelUnavailableError,
    ModelRateLimitError,
    ModelRequestTimeout,
)
# Stored in the response cache when a call ran out of output tokens, so a rerun goes straight to the fallback
# (splitting the chunk) instead of paying for the same truncated answer again. Never valid model output.
_TRUNCATION_MARK = '{"__truncated__": true}'


@dataclass
class CallStats:
    calls: int = 0
    cache_hits: int = 0
    external_calls: int = 0
    repairs: int = 0
    transient_retries: int = 0  # retries after a connection failure, timeout, overload or server error
    unmatched_tokens: list[str] = field(default_factory=list)
    # Usage of the calls actually sent (cache hits add nothing), including failed and truncated ones.
    input_tokens: int = 0
    output_tokens: int = 0  # includes reasoning
    reasoning_tokens: int = 0  # where the server reports it (oMLX, LM Studio); 0 otherwise
    thinking_chars: int = 0  # length of the returned reasoning text, for servers that don't report tokens

    def add_usage(self, responses: list[ModelResponse]) -> None:
        for r in responses:
            self.input_tokens += r.usage.input_tokens
            self.output_tokens += r.usage.output_tokens
            self.reasoning_tokens += r.usage.output_reasoning_tokens or r.usage.details.get("reasoning_tokens", 0)
            self.thinking_chars += sum(len(p.content) for p in r.parts if isinstance(p, ThinkingPart))

    @staticmethod
    def combined(*stats: CallStats) -> CallStats:
        out = CallStats()
        for s in stats:
            for f in fields(CallStats):
                setattr(out, f.name, getattr(out, f.name) + getattr(s, f.name))
        return out


class _Recorder(httpx2.AsyncBaseTransport):
    """Keeps the exact request and response bodies of every HTTP exchange, for the audit log."""

    def __init__(self) -> None:
        self._inner = httpx2.AsyncHTTPTransport()
        self.exchanges: list[tuple[str, str]] = []

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        sent = request.content.decode("utf-8", errors="replace")
        response = await self._inner.handle_async_request(request)
        body = await response.aread()  # decoded; drop encoding headers so it isn't decoded twice
        self.exchanges.append((sent, body.decode("utf-8", errors="replace")))
        headers = [
            (k, v) for k, v in response.headers.items() if k.lower() not in ("content-encoding", "content-length")
        ]
        return httpx2.Response(response.status_code, headers=headers, content=body, request=request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def build_model(role: Role, http_client: httpx2.AsyncClient) -> Model:
    ep = role.endpoint
    key = os.environ.get(ep.api_key_env) if ep.api_key_env else None
    if ep.api_key_env and not key:
        raise LLMError(f"environment variable {ep.api_key_env} is not set (endpoint {ep.name})", code="missing_api_key")
    if ep.api == "openai":
        # Local servers ignore the key, but the OpenAI client refuses to start without one.
        provider = OpenAIProvider(base_url=ep.base_url, api_key=key or "unused", http_client=http_client)
        # "prompt" mode is for servers whose response_format support can't be trusted, so send none at all.
        # Pydantic AI would otherwise send {"type": "json_object"}, which let oMLX return bare arrays.
        profile = {"supports_json_object_output": False} if role.structured == "prompt" else None
        return OpenAIChatModel(role.model, provider=provider, profile=profile)
    return AnthropicModel(
        role.model, provider=AnthropicProvider(api_key=key, base_url=ep.base_url, http_client=http_client)
    )


class StructuredCaller:
    """Schema-validated calls with caching, redaction for external endpoints, and one repair round."""

    def __init__(self, role: Role, store: Store, redactor_factory: Callable[[], Redactor], *, use_cache: bool = True):
        self.role = role
        self.store = store
        self.external = role.endpoint.trust == "external"
        self.use_cache = use_cache  # off for comparisons, where repeats must be independent samples
        self._redactor_factory = redactor_factory
        self._redactor: Redactor | None = None
        self.stats = CallStats()

    @property
    def redactor(self) -> Redactor:
        if self._redactor is None:
            self._redactor = self._redactor_factory()
        return self._redactor

    def prepare(self, user: str) -> tuple[str, RequestMap | None]:
        """Redaction runs on the calling thread before any request is dispatched (the vault is shared)."""
        if not self.external:
            return user, None
        request = RequestMap()
        return self.redactor.redact(user, request), request

    def call(
        self,
        system: str,
        prepared: tuple[str, RequestMap | None],
        model: type[T],
        check: Callable[[T], tuple[T, list[str]]],
        *,
        remember_truncation: bool = False,
    ) -> T:
        """`check` returns the value and a list of problems; problems trigger one repair round.

        Problems that survive the repair are left for the caller to clean, so a usable answer is never lost.
        `remember_truncation` caches a truncated outcome, for callers with a fallback, so a rerun goes straight
        to it. Transient failures are retried with backoff before giving up.
        """
        user, request = prepared
        # Our own format rules, not Pydantic AI's templates: on the same model the framework's defaults led to
        # bare-array answers in prompt mode and to placeholders losing their brackets.
        system += _format_rules(model.model_json_schema())
        if request is not None:
            system += _PLACEHOLDER_RULES
        key = self._cache_key(system, user, model)
        raw = self.store.cache_get(key) if self.use_cache else None
        exchanges: list[tuple[str, str]] = []
        if raw == _TRUNCATION_MARK:
            self.stats.cache_hits += 1
            raise Truncated()
        if raw is not None:
            self.stats.cache_hits += 1
        else:
            try:
                raw, exchanges = self._retrying(lambda: self._run(system, user, model, check))
            except Truncated:
                if self.use_cache and remember_truncation:
                    self.store.cache_put(key, _TRUNCATION_MARK)
                raise
            if self.use_cache:
                self.store.cache_put(key, raw)

        data: Any = json.loads(raw)
        if request is not None:
            unmatched: list[str] = []
            data = rehydrate_tree(data, request, unmatched)
            self.stats.unmatched_tokens.extend(unmatched)
            if exchanges:
                self._audit(exchanges, unmatched)
        return model.model_validate(data)

    def _audit(self, exchanges: list[tuple[str, str]], unmatched: list[str]) -> None:
        self.store.audit(
            self.role.endpoint.name,
            self.role.model,
            json.dumps([sent for sent, _ in exchanges], ensure_ascii=False),
            json.dumps([got for _, got in exchanges], ensure_ascii=False),
            unmatched,
        )

    def _retrying[R](self, attempt: Callable[[], R]) -> R:
        """Run `attempt`, retrying transient failures after each wait in RETRY_BACKOFF (or the server's
        retry-after, if longer), so a server restart or a busy hosted API doesn't end a long run."""
        for n, wait in enumerate((*RETRY_BACKOFF, None), 1):
            try:
                return attempt()
            except Transient as e:
                if wait is None:
                    raise
                wait = max(wait, e.retry_after or 0)
                self.stats.transient_retries += 1
                # Only the short code: an error body may quote the request.
                print(
                    f"{self.role.endpoint.name}: {e.code}; retry {n} of {len(RETRY_BACKOFF)} in {wait:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait)
        raise AssertionError("unreachable")

    def _run(
        self, system: str, user: str, model: type[T], check: Callable[[T], tuple[T, list[str]]]
    ) -> tuple[str, list[tuple[str, str]]]:
        """One agent run. Returns the validated output as JSON before rehydration (what the cache stores),
        plus the exact HTTP exchanges."""
        recorder = _Recorder()
        client = httpx2.AsyncClient(transport=recorder, timeout=self.role.endpoint.timeout_seconds, trust_env=False)
        if self.role.structured == "json_schema":
            output = NativeOutput(model, template=False)
        else:
            output = PromptedOutput(model, template=False)
        built = build_model(self.role, client)
        agent = Agent(
            built,
            output_type=output,
            instructions=system,
            model_settings=_settings(self.role, built.profile),
            retries={"output": 1},
        )

        @agent.output_validator
        def validate(ctx: RunContext[None], value: T) -> T:
            _, problems = check(value)
            if problems and ctx.retry == 0:
                raise ModelRetry("Your JSON had these problems:\n- " + "\n- ".join(problems))
            return value

        with capture_run_messages() as messages:
            try:
                result = _run_closing(agent, client, user)
            except (ModelAPIError, UnexpectedModelBehavior) as e:
                if self.external and recorder.exchanges:
                    self._audit(recorder.exchanges, [])  # data left the machine even though the run failed
                if isinstance(e, ModelAPIError):
                    raise _api_error(self.role.endpoint.name, e) from e
                responses = [m for m in messages if isinstance(m, ModelResponse)]
                if responses and responses[-1].finish_reason == "length":
                    raise Truncated() from e
                raise LLMError(f"model output unusable: {e}", code="unusable_output") from e
            finally:
                self.stats.calls += 1
                self.stats.add_usage([m for m in messages if isinstance(m, ModelResponse)])

        if self.external:
            self.stats.external_calls += 1
        if result.response.finish_reason == "length":
            if self.external:
                self._audit(recorder.exchanges, [])
            raise Truncated()
        # Every model response after the first answered a repair request.
        self.stats.repairs += sum(isinstance(m, ModelResponse) for m in result.all_messages()) - 1
        return result.output.model_dump_json(), recorder.exchanges

    def call_text(
        self,
        system: str,
        prepared: tuple[str, RequestMap | None],
        attachments: Sequence[UserContent] = (),
        native_tools: Sequence[AbstractNativeTool] = (),
    ) -> TextResult:
        """One free-text answer about a single media item or link.

        Not cached here: the media stage stores results itself, and only once it has judged them complete.
        `native_tools` run on the provider's side; no tool ever runs on this machine. Raises LLMError with a
        short `code` when the request fails, is truncated, or asks this machine to run a tool. Transient
        failures are retried with backoff first.
        """
        user, request = prepared
        if request is not None:
            system += _PLACEHOLDER_RULES
        prompt = [user, *attachments] if attachments else user
        result, exchanges = self._retrying(lambda: self._text_run(system, prompt, native_tools))
        text = _final_text(result.response) or result.output
        unmatched: list[str] = []
        if request is not None:
            text = rehydrate(text, request, unmatched)
            self.stats.unmatched_tokens.extend(unmatched)
        if self.external:
            self._audit(exchanges, unmatched)
        return TextResult(" ".join(text.split()), result.all_messages(), exchanges)

    def _text_run(
        self, system: str, prompt: Any, native_tools: Sequence[AbstractNativeTool]
    ) -> tuple[Any, list[tuple[str, str]]]:
        """One agent run for `call_text`: the result and the exact HTTP exchanges."""
        recorder = _Recorder()
        client = httpx2.AsyncClient(transport=recorder, timeout=self.role.endpoint.timeout_seconds, trust_env=False)
        built = build_model(self.role, client)
        agent = Agent(
            built,
            instructions=system,
            model_settings=_settings(self.role, built.profile),
            capabilities=[NativeTool(t) for t in native_tools],
            retries={"output": 0},
        )
        with capture_run_messages() as messages:
            try:
                result = _run_closing(agent, client, prompt)
            except (ModelAPIError, UnexpectedModelBehavior) as e:
                if self.external and recorder.exchanges:
                    self._audit(recorder.exchanges, [])
                if isinstance(e, ModelAPIError):
                    raise _api_error(self.role.endpoint.name, e) from e
                responses = [m for m in messages if isinstance(m, ModelResponse)]
                if any(isinstance(p, ToolCallPart) for r in responses for p in r.parts):
                    raise LLMError("the response asked this machine to run a tool", code="local_tool_requested") from e
                if responses and responses[-1].finish_reason == "length":
                    raise Truncated() from e
                raise LLMError(f"model output unusable: {e}", code="unusable_output") from e
            finally:
                self.stats.calls += 1
                self.stats.add_usage([m for m in messages if isinstance(m, ModelResponse)])

        if self.external:
            self.stats.external_calls += 1
        if result.response.finish_reason == "length":
            if self.external:
                self._audit(recorder.exchanges, [])
            raise Truncated()
        return result, recorder.exchanges

    def _cache_key(self, system: str, user: str, model: type[BaseModel]) -> str:
        ep, r = self.role.endpoint, self.role
        return hashlib.sha256(
            json.dumps(
                [
                    ep.api,
                    ep.base_url,
                    r.model,
                    _settings_key(r),
                    r.structured,
                    model.model_json_schema(),
                    system,
                    user,
                ],
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()


def _api_error(endpoint: str, e: ModelAPIError) -> LLMError:
    """Our error for a failed request: `Transient` when retrying may help, with a short code for reports."""
    status = getattr(e, "status_code", None)
    if status is not None:
        message, code = f"{endpoint} returned {status}: {str(e.body)[:500]}", f"http_{status}"
    elif isinstance(e, (ModelTimeoutError, ModelRequestTimeout)):
        message, code = f"{endpoint} timed out: {e}", "timeout"
    elif isinstance(e, ModelConnectionError):
        message, code = f"could not reach {endpoint}: {e}", "connection"
    else:
        message, code = f"{endpoint} failed: {e}", type(e).__name__
    if isinstance(e, _TRANSIENT_ERRORS):
        return Transient(message, code, e.retry_after)
    return LLMError(message, code)


def _final_text(response: ModelResponse) -> str:
    """Text after the provider's last tool call or result, so "I'll fetch the page" preambles are dropped."""
    last_tool = max(
        (i for i, p in enumerate(response.parts) if isinstance(p, (NativeToolCallPart, NativeToolReturnPart))),
        default=-1,
    )
    return "".join(p.content for p in response.parts[last_tool + 1 :] if isinstance(p, TextPart))


_TRUNCATED = (
    "output hit max_tokens before the answer was complete (reasoning models can spend the whole budget "
    "thinking); raise max_output_tokens, or thinking_tokens when thinking is on, or turn thinking off"
)


_PLACEHOLDER_RULES = (
    "\n\nSome values in the input are replaced by placeholders such as [PERSON_3] or [URL_1]. "
    "Copy placeholders exactly as written wherever you refer to them. Never invent new placeholders."
)


def _run_closing(agent: Agent, client: httpx2.AsyncClient, prompt: Any) -> Any:
    """Run one agent call in its own event loop and close the HTTP client before returning.

    The client is created per call (for the audit recorder), so its connections must be closed on the loop
    that opened them; otherwise every call leaks a socket. A fresh loop per call is safe from worker threads.
    """

    async def go() -> Any:
        async with client:
            return await agent.run(prompt)

    return asyncio.run(go())


def _settings(role: Role, profile: ModelProfile) -> ModelSettings:
    settings = ModelSettings(max_tokens=role.output_limit)
    if role.temperature is not None:
        settings["temperature"] = role.temperature
    if role.top_p is not None:
        settings["top_p"] = role.top_p
    body: dict[str, Any] = {}
    if role.endpoint.api == "anthropic":
        if role.top_k is not None:
            settings["top_k"] = role.top_k
        if role.thinking is not None:
            settings["thinking"] = role.thinking
            # Models with adaptive thinking reject a fixed budget; Pydantic AI picks adaptive for them.
            if role.thinks and not profile.get("anthropic_supports_adaptive_thinking", False):
                settings["anthropic_thinking"] = {"type": "enabled", "budget_tokens": role.thinking_tokens}  # type: ignore[typeddict-unknown-key]
    else:
        if role.presence_penalty is not None:
            settings["presence_penalty"] = role.presence_penalty
        if role.top_k is not None:
            body["top_k"] = role.top_k  # not an OpenAI parameter; llama-server, LM Studio, oMLX and vLLM read it
        body.update(_thinking_body(role.thinking))
    body = _merge_body(body, role.extra_body)
    if body:
        settings["extra_body"] = body
    if role.extra_headers:
        settings["extra_headers"] = role.extra_headers
    return settings


def _thinking_body(thinking: bool | str | None) -> dict[str, Any]:
    """Thinking switch for OpenAI-compatible local servers.

    `chat_template_kwargs.enable_thinking` reaches the chat template (Qwen and others) on llama-server and oMLX.
    LM Studio ignores it but honours `reasoning_effort: "none"`, so `false` sends both. An effort level is passed
    through as `reasoning_effort` for templates that read it.
    """
    if thinking is None:
        return {}
    body: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": thinking is not False}}
    if thinking is False:
        body["reasoning_effort"] = "none"
    elif isinstance(thinking, str):
        body["reasoning_effort"] = thinking
    return body


def _merge_body(ours: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    """The role's `extra_body` wins; `chat_template_kwargs` is merged key by key so both can add to it."""
    merged = {**ours, **user}
    if isinstance(ours.get("chat_template_kwargs"), dict) and isinstance(user.get("chat_template_kwargs"), dict):
        merged["chat_template_kwargs"] = {**ours["chat_template_kwargs"], **user["chat_template_kwargs"]}
    return merged


def _settings_key(role: Role) -> list[Any]:
    """Every role setting that changes the request, for the cache key."""
    return [
        role.temperature,
        role.top_p,
        role.top_k,
        role.presence_penalty,
        role.output_limit,
        role.thinking,
        role.thinking_tokens if role.thinks else None,
        role.extra_body,
    ]


def _format_rules(schema: dict) -> str:
    return (
        "\n\nRespond with a single JSON object that matches this JSON Schema. No prose, no code fences.\n"
        + json.dumps(schema, separators=(",", ":"))
    )
