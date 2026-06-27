"""LLM client — Protocol + FakeLLMClient + LlamaCppClient.

The architecture document (:mod:`docs/03_architecture_v1.md` §6.4)
defines the ``LLMClient`` Protocol as the single seam between TinyRAG
and the language model. This module provides three things:

1. :class:`LLMClient` — the Protocol itself.
2. :class:`FakeLLMClient` — a deterministic, in-memory implementation
   for unit tests. No network, no model weights, no GPU.
3. :class:`LlamaCppClient` — the real implementation, talking to a
   ``llama-server`` process over HTTP using the OpenAI-compatible
   ``/v1/chat/completions`` endpoint with Server-Sent Events (SSE).

Design rules
------------
- **No business logic.** ``LLMClient.generate`` only generates text.
  Prompt construction (system prompt + retrieved chunks + user
  question) is the job of :mod:`tinyrag.generation.prompt_builder`,
  which will be added in Phase 4. Keeping these split means a
  different prompt strategy can be A/B-tested without touching the
  HTTP layer.
- **Streaming is mandatory.** Every real LLM call should stream
  tokens so the FastAPI endpoint can use
  :func:`sse_starlette.sse.EventSourceResponse` (Phase 4). The
  Protocol's return type is ``Iterator[str]`` — a non-streaming
  variant would block the entire response chain.
- **Typed errors.** Callers can distinguish "the model server is
  down" (:class:`LLMUnavailableError`) from "the model refused" or
  "the prompt was malformed" (:class:`LLMRefusedError`). This is how
  the API layer will decide between 503 and 400.

Location: ``src/tinyrag/generation/llm_client.py``
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default OpenAI-compatible endpoint on llama-server.
DEFAULT_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

#: HTTP request timeout for a single chat-completion call.
#: 120 s is generous: a 512-token Phi-3 reply on a 10-core CPU
#: takes ~30-60 s. If you bump max_tokens, bump this too.
DEFAULT_TIMEOUT_S = 120.0

#: Default sampling temperature. 0.0 = deterministic greedy, which
#: is what we want for an eval set (Phase 5).
DEFAULT_TEMPERATURE = 0.0

#: Default cap on output tokens.
DEFAULT_MAX_TOKENS = 512


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatMessage:
    """A single message in a chat conversation.

    Matches the OpenAI Chat API shape: ``{"role": ..., "content": ...}``.
    We use a dataclass (not a dict) so the type checker can catch
    typos like ``ChatMessage(role="sysem", ...)``.
    """

    role: str  # "system" | "user" | "assistant"
    content: str

    def to_openai(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class GenerationStats:
    """Counters returned alongside the generated text.

    Populated by ``LlamaCppClient.generate`` from llama-server's
    ``usage`` block (and a wall-clock timer). Useful for the smoke
    test and for the Phase 5 evaluation harness.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        """Output tokens per second over the duration of the call."""
        return (
            self.completion_tokens / self.duration_seconds
            if self.duration_seconds > 0
            else 0.0
        )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Base class for everything in this module."""


class LLMUnavailableError(LLMError):
    """The server is unreachable, returned 5xx, or timed out.

    Callers should map this to HTTP 503 ("Service Unavailable") and
    optionally retry with backoff.
    """


class LLMRefusedError(LLMError):
    """The server rejected the request (400-class).

    Usually a malformed prompt or a context-length overflow. Callers
    should map this to HTTP 400 ("Bad Request").
    """


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """The single seam between TinyRAG and the language model.

    Architectural fit: see ``docs/03_architecture_v1.md`` §6.4. Any
    object with ``generate``, ``model_name``, and ``is_healthy``
    methods that match these signatures satisfies the Protocol —
    duck-typed, no inheritance required.
    """

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> tuple[str, GenerationStats]:
        """Generate a completion for ``messages``.

        Returns
        -------
        (text, stats):
            The full generated text (concatenation of all streamed
            tokens) and timing/token statistics.
        """
        ...

    def stream_generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Iterator[str]:
        """Yield one LLM token at a time.

        Equivalent to :meth:`generate` but yields each token as it
        arrives, so the HTTP layer can emit per-token SSE events
        (Step 4.19). The Protocol declares a **sync** generator —
        ``sse-starlette`` accepts sync iterators and wraps them in
        ``anyio.to_thread.run_sync`` so the FastAPI event loop is
        never blocked (see ``sse_starlette/sse.py:190-193``).

        Contract
        ---------
        - Tokens are yielded in the order the model emits them.
        - The caller is responsible for assembling the full text by
          joining the yielded tokens — implementations do NOT
          accumulate internally (so the caller can ``yield``
          immediately to the SSE wire).
        - Implementations MAY raise :class:`LLMUnavailableError` or
          :class:`LLMRefusedError` mid-stream if the server fails;
          the HTTP layer wraps this into a single
          ``{"event":"error"}`` SSE frame (no partial tokens lost).
        - Implementations SHOULD update ``self.stats`` so callers
          that follow with :meth:`generate` (or that read ``stats``
          after the loop) see the totals.
        """
        ...

    def model_name(self) -> str:
        """Return the model id this client is configured to call."""
        ...

    def is_healthy(self, *, timeout_s: float = 5.0) -> bool:
        """Return ``True`` iff the LLM backend is reachable.

        Used by the ``/health`` API endpoint. Must never raise —
        return ``False`` on any error so the caller can use the
        boolean directly.
        """
        ...


# ---------------------------------------------------------------------------
# Fake implementation (for tests + offline dev)
# ---------------------------------------------------------------------------


@dataclass
class FakeLLMClient:
    """Deterministic in-memory LLM stand-in.

    Returns ``default_response`` (or a per-message override) as if it
    had been streamed. Tests use this to exercise the entire rest of
    the pipeline — retrieval, prompt building, API routes — without
    loading a real model.

    Parameters
    ----------
    default_response:
        Text returned when no override matches. Default is a short
        canned "I'm a fake LLM" reply.
    response_overrides:
        Map from a substring that must appear in the user message to
        the canned response. The first matching substring wins.
        Lets a test assert "if the prompt mentions 'microwave',
        return this specific answer" without monkey-patching.
    token_delay_seconds:
        Sleep this many seconds between "tokens" so the streaming
        behaviour of the API layer is exercised. Default 0 (instant).
    raise_after_tokens:
        If set, raise :class:`LLMRefusedError` after emitting this
        many tokens. Useful for testing the error path.
    """

    default_response: str = "I'm a fake LLM. This is a canned response."
    response_overrides: dict[str, str] = field(default_factory=dict)
    token_delay_seconds: float = 0.0
    raise_after_tokens: int | None = None
    stats: GenerationStats = field(default_factory=GenerationStats)
    #: Override the value returned by :meth:`is_healthy`. Defaults to
    #: ``True`` (healthy). Tests of the API ``/health`` endpoint flip
    #: this to ``False`` to exercise the degraded path.
    healthy: bool = True
    #: The model name reported by :meth:`model_name`.
    model: str = "fake-llm"

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> tuple[str, GenerationStats]:
        start = time.monotonic()
        # Pick a response based on user-message content overrides.
        response = self.default_response
        for needle, canned in self.response_overrides.items():
            if any(needle in m.content for m in messages):
                response = canned
                break

        # Simulate streaming by yielding one whitespace-separated word
        # at a time. This makes the FakeClient exercise the same code
        # path as the real client (loop + accumulate).
        full = []
        for i, word in enumerate(response.split(" ")):
            if self.token_delay_seconds > 0:
                time.sleep(self.token_delay_seconds)
            if self.raise_after_tokens is not None and i >= self.raise_after_tokens:
                raise LLMRefusedError(
                    f"FakeLLMClient configured to raise after {self.raise_after_tokens} tokens"
                )
            full.append(word)

        text = " ".join(full)
        self.stats = GenerationStats(
            prompt_tokens=sum(len(m.content.split()) for m in messages),
            completion_tokens=len(text.split()),
            total_tokens=sum(len(m.content.split()) for m in messages) + len(text.split()),
            duration_seconds=time.monotonic() - start,
        )
        return text, self.stats

    def stream_generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Iterator[str]:
        """Yield one whitespace-separated word at a time.

        Mirrors the :meth:`generate` word-loop but yields each word
        instead of appending. The order, the
        :attr:`token_delay_seconds` pause, and the
        :attr:`raise_after_tokens` failure mode are all preserved so
        the SSE code path is exercised with the same knob set as
        the non-streaming one. ``self.stats`` is updated at the end
        (or on early raise) so subsequent :meth:`generate` callers
        see the totals.

        Note: ``max_tokens`` and ``temperature`` are accepted but
        ignored — the fake client has no model to apply them to.
        They exist only to match the :class:`LLMClient` Protocol
        signature so a single dependency-injection site can call
        either method.
        """
        start = time.monotonic()
        # Pick a response based on user-message content overrides.
        response = self.default_response
        for needle, canned in self.response_overrides.items():
            if any(needle in m.content for m in messages):
                response = canned
                break

        full: list[str] = []
        try:
            for i, word in enumerate(response.split(" ")):
                if self.token_delay_seconds > 0:
                    time.sleep(self.token_delay_seconds)
                if (
                    self.raise_after_tokens is not None
                    and i >= self.raise_after_tokens
                ):
                    raise LLMRefusedError(
                        f"FakeLLMClient configured to raise after "
                        f"{self.raise_after_tokens} tokens"
                    )
                full.append(word)
                yield word
        finally:
            # Update stats whether the loop finished naturally or
            # exited via the raise-after-tokens guard — callers
            # that probe ``self.stats`` after the loop ends should
            # see the totals actually produced.
            text = " ".join(full)
            self.stats = GenerationStats(
                prompt_tokens=sum(len(m.content.split()) for m in messages),
                completion_tokens=len(text.split()),
                total_tokens=sum(len(m.content.split()) for m in messages)
                + len(text.split()),
                duration_seconds=time.monotonic() - start,
            )

    def model_name(self) -> str:
        return self.model

    def is_healthy(self, *, timeout_s: float = 5.0) -> bool:
        # The fake never makes a real network call; its health is a
        # configuration knob so the API /health endpoint can be
        # exercised without a live model.
        return self.healthy


# ---------------------------------------------------------------------------
# Real implementation (talks to llama-server)
# ---------------------------------------------------------------------------


@dataclass
class LlamaCppClient:
    """Synchronous streaming client for ``llama-server``.

    Talks to the OpenAI-compatible endpoint exposed by ``llama-server``
    when started with ``--host 127.0.0.1 --port 8080`` (the default in
    :file:`scripts/run_llamacpp.sh` / :file:`Makefile` target ``run-llm``).

    The client buffers the full response in memory and returns it as a
    single string. The streaming is what makes the underlying HTTP call
    low-latency on the first byte (TTFB), even though we accumulate
    before returning — the FastAPI endpoint in Phase 4 will re-stream
    from the buffer to the browser via SSE.

    Parameters
    ----------
    base_url:
        ``http://127.0.0.1:8080`` for a local llama-server. Include
        scheme + host + port; no trailing slash.
    model:
        The model id to send in the request body. Must match what
        llama-server reports at ``GET /v1/models``. For our setup
        this is the on-disk filename without ``.gguf`` (e.g.
        ``"phi-3-mini"``).
    timeout_s:
        HTTP request timeout. The default (120 s) covers up to ~1k
        output tokens on a 10-core CPU.
    client:
        Optional ``httpx.Client`` for tests. Defaults to a fresh
        client constructed from ``timeout_s``.
    """

    base_url: str = "http://127.0.0.1:8080"
    model: str = "phi-3-mini"
    timeout_s: float = DEFAULT_TIMEOUT_S
    chat_completions_path: str = DEFAULT_CHAT_COMPLETIONS_PATH
    client: httpx.Client | None = None

    def __post_init__(self) -> None:
        # Strip a trailing slash so base_url + path concat is clean.
        self.base_url = self.base_url.rstrip("/")
        # Lazily construct the httpx.Client so import-time is cheap.
        # We don't open a connection until the first request.
        if self.client is None:
            self._owns_client = True
        else:
            self._owns_client = False

    # ----- internal helpers --------------------------------------------

    def _http(self) -> httpx.Client:
        if self.client is None:
            # Per-operation timeouts, NOT a single ``timeout=self.timeout_s``
            # passed to the client. Why? The streaming generator
            # (:meth:`stream_generate`) reads chunks over a long-lived
            # connection — a single 120 s budget is consumed by the
            # *whole* response. If llama.cpp's prompt eval is slow on a
            # cold KV cache (~8 s for Phi-3 Mini 3.8B Q4_K_M) and the
            # first eval token takes another 30-60 s to surface, the
            # read of that first chunk alone eats a huge slice of the
            # 120 s budget. Subsequent chunks then race against the
            # *remaining* time and httpx raises ``TimeoutException``
            # mid-stream with the cryptic
            # ``peer closed connection without sending complete
            # message body`` message in the browser.
            #
            # Setting ``read=self.timeout_s`` gives EACH chunk read its
            # own fresh 120 s window. ``connect`` / ``write`` / ``pool``
            # are short (10 s) because they correspond to TCP setup +
            # request upload, which should be near-instant for an
            # in-process llama-server.
            self.client = httpx.Client(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self.timeout_s,
                    write=10.0,
                    pool=10.0,
                )
            )
        return self.client

    def close(self) -> None:
        """Close the underlying ``httpx.Client`` if we created it."""
        if self._owns_client and self.client is not None:
            self.client.close()
            self.client = None

    # ----- public API --------------------------------------------------

    # ----- inspection helpers -----------------------------------------

    def model_name(self) -> str:
        """Return the model id this client is configured to call.

        The API layer uses this to surface ``which model answered?`` in
        the response payload and in the observability logs.
        """
        return self.model

    def is_healthy(self, *, timeout_s: float = 5.0) -> bool:
        """Return ``True`` iff llama-server is up and reachable.

        Talks to ``GET /v1/models`` (OpenAI-standard endpoint that
        llama-server exposes). Any error — connection refused, timeout,
        non-2xx — counts as unhealthy. The default 5-second timeout is
        much shorter than :attr:`timeout_s` because this is called from
        the ``/health`` endpoint and must not block the API thread.

        Raises
        ------
        Nothing. ``is_healthy`` is intentionally never-throwing so the
        API layer can call it directly without a try/except.
        """
        try:
            resp = self._http().get(
                f"{self.base_url}/v1/models", timeout=timeout_s
            )
            return 200 <= resp.status_code < 300
        except httpx.HTTPError:
            return False

    # ----- public API --------------------------------------------------

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> tuple[str, GenerationStats]:
        """POST to /v1/chat/completions with ``stream=true`` and accumulate.

        Raises
        ------
        LLMUnavailableError:
            Connection error, timeout, or 5xx response.
        LLMRefusedError:
            4xx response (model refused the prompt).
        """
        url = f"{self.base_url}{self.chat_completions_path}"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        start = time.monotonic()
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        text_parts: list[str] = []

        try:
            with self._http().stream("POST", url, json=payload) as response:
                if response.status_code >= 500:
                    raise LLMUnavailableError(
                        f"llama-server returned {response.status_code}: "
                        f"{response.read().decode('utf-8', errors='replace')[:200]}"
                    )
                if response.status_code >= 400:
                    raise LLMRefusedError(
                        f"llama-server rejected the request ({response.status_code}): "
                        f"{response.read().decode('utf-8', errors='replace')[:200]}"
                    )

                # llama-server sends SSE lines of the form
                #   data: {"choices":[{"delta":{"content":"tok"}, ...}], ...}
                # terminated by a final ``data: [DONE]``.
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            logger.warning("Skipping malformed SSE line: %r", line[:200])
                            continue
                        # Extract the delta content, if any.
                        for choice in chunk.get("choices", []):
                            delta = choice.get("delta") or {}
                            content = delta.get("content")
                            if content:
                                text_parts.append(content)
                        # Some chunks carry the usage block (usually the last one).
                        if "usage" in chunk:
                            usage = chunk["usage"]
                            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                            completion_tokens = usage.get(
                                "completion_tokens", completion_tokens
                            )
                            total_tokens = usage.get("total_tokens", total_tokens)
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError(
                f"llama-server timed out after {self.timeout_s}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"HTTP error talking to llama-server: {exc}") from exc

        text = "".join(text_parts)
        # Defensive fallback if the server didn't include a usage block
        # (older builds or certain quantization combos). Estimate
        # token count by whitespace splitting — not exact, but it's
        # the best we can do without re-tokenising.
        if completion_tokens == 0 and text:
            completion_tokens = len(text.split())
        if prompt_tokens == 0:
            prompt_tokens = sum(len(m.content.split()) for m in messages)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        return text, GenerationStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_seconds=time.monotonic() - start,
        )

    def stream_generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Iterator[str]:
        """POST to /v1/chat/completions with ``stream=true`` and yield tokens.

        Equivalent to :meth:`generate` but yields each
        ``choices[].delta.content`` as soon as it arrives, instead of
        accumulating into a single string. The HTTP layer (Step 4.19)
        wraps this generator in an ``EventSourceResponse`` so the
        dashboard can render tokens one-by-one.

        The usage block (``"usage": {...}`` in the last chunk) is
        still captured into ``self.stats`` so callers that read the
        field after the loop ends see real ``prompt_tokens`` /
        ``completion_tokens`` values.

        Raises
        ------
        LLMUnavailableError:
            Connection error, timeout, or 5xx response (raised before
            any token is yielded, OR mid-stream if the server
            disconnects after the first byte).
        LLMRefusedError:
            4xx response — the model rejected the prompt. Raised
            before any token is yielded.
        """
        url = f"{self.base_url}{self.chat_completions_path}"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        start = time.monotonic()
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        text_parts: list[str] = []

        # NB: the ``stats`` attribute is mutated AFTER the loop so
        # the totals reflect what was actually streamed (rather
        # than zero on early failure). Callers should NOT read
        # ``self.stats`` mid-stream — wait for the generator to
        # finish.
        try:
            with self._http().stream("POST", url, json=payload) as response:
                if response.status_code >= 500:
                    raise LLMUnavailableError(
                        f"llama-server returned {response.status_code}: "
                        f"{response.read().decode('utf-8', errors='replace')[:200]}"
                    )
                if response.status_code >= 400:
                    raise LLMRefusedError(
                        f"llama-server rejected the request ({response.status_code}): "
                        f"{response.read().decode('utf-8', errors='replace')[:200]}"
                    )

                # llama-server sends SSE lines of the form
                #   data: {"choices":[{"delta":{"content":"tok"}, ...}], ...}
                # terminated by a final ``data: [DONE]``.
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            logger.warning(
                                "Skipping malformed SSE line: %r", line[:200]
                            )
                            continue
                        # Extract the delta content, if any — yield
                        # IMMEDIATELY (no buffering) so the SSE wire
                        # sees the token as soon as the server sent it.
                        for choice in chunk.get("choices", []):
                            delta = choice.get("delta") or {}
                            content = delta.get("content")
                            if content:
                                text_parts.append(content)
                                yield content
                        # Some chunks carry the usage block (usually the last one).
                        if "usage" in chunk:
                            usage = chunk["usage"]
                            prompt_tokens = usage.get(
                                "prompt_tokens", prompt_tokens
                            )
                            completion_tokens = usage.get(
                                "completion_tokens", completion_tokens
                            )
                            total_tokens = usage.get(
                                "total_tokens", total_tokens
                            )
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError(
                f"llama-server timed out after {self.timeout_s}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"HTTP error talking to llama-server: {exc}"
            ) from exc

        text = "".join(text_parts)
        # Defensive fallback if the server didn't include a usage block
        # (older builds or certain quantization combos). Estimate
        # token count by whitespace splitting — not exact, but it's
        # the best we can do without re-tokenising.
        if completion_tokens == 0 and text:
            completion_tokens = len(text.split())
        if prompt_tokens == 0:
            prompt_tokens = sum(len(m.content.split()) for m in messages)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        # Mirror :meth:`generate` so callers reading ``self.stats``
        # after the stream ends see the same numbers.
        self.stats = GenerationStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_seconds=time.monotonic() - start,
        )


__all__ = [
    "LLMClient",
    "ChatMessage",
    "GenerationStats",
    "LLMError",
    "LLMUnavailableError",
    "LLMRefusedError",
    "FakeLLMClient",
    "LlamaCppClient",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_CHAT_COMPLETIONS_PATH",
]
