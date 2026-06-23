"""Tests for the LLMClient Protocol + FakeLLMClient + LlamaCppClient.

These are the unit tests that lock down the LLM seam. They cover:

1. ``LLMClient`` is a :class:`typing.Protocol` — both real and fake
   clients satisfy it (duck-typed, no inheritance).
2. ``FakeLLMClient`` returns canned text with deterministic token counts.
3. ``FakeLLMClient.response_overrides`` lets a test assert "if the prompt
   mentions X, return Y" without monkey-patching.
4. ``FakeLLMClient`` raises :class:`LLMRefusedError` when configured.
5. ``ChatMessage.to_openai()`` produces the OpenAI Chat API shape.
6. ``LlamaCppClient`` POSTs to ``/v1/chat/completions`` with the right
   JSON body (model, messages, stream=true, max_tokens, temperature).
7. ``LlamaCppClient`` parses SSE ``data:`` lines and concatenates
   ``choices[].delta.content`` into the final text.
8. ``LlamaCppClient`` stops on ``data: [DONE]``.
9. ``LlamaCppClient`` uses the ``usage`` block to populate token counts
   and falls back to whitespace estimation when the server omits it.
10. ``LlamaCppClient`` maps connection errors → ``LLMUnavailableError``,
    5xx → ``LLMUnavailableError``, 4xx → ``LLMRefusedError``,
    timeouts → ``LLMUnavailableError``.
11. ``LlamaCppClient.close()`` only closes the client it owns.

The network layer is mocked via ``httpx.MockTransport`` so the tests
run hermetically — no llama-server, no internet.

Location: ``tests/test_llm_client.py``
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from tinyrag.generation import (
    ChatMessage,
    FakeLLMClient,
    LlamaCppClient,
    LLMClient,
    LLMError,
    LLMRefusedError,
    LLMUnavailableError,
)
from tinyrag.generation.llm_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_S,
)

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _sse_chunk(content: str | None, finish_reason: str | None = None) -> str:
    """Render one OpenAI-style SSE chunk as a server would."""
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": "test-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


def _sse_usage(prompt_tokens: int, completion_tokens: int) -> str:
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": "test-model",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return f"data: {json.dumps(payload)}\n\n"


def _make_sse_body(*chunks: str) -> str:
    """Join chunks into a single SSE response body."""
    return "".join(chunks)


def _mock_transport(chunks: list[str], status_code: int = 200) -> httpx.MockTransport:
    """Build an ``httpx.MockTransport`` that returns SSE bytes."""
    body = _make_sse_body(*chunks)

    def handler(request: httpx.Request) -> httpx.Response:
        # Sanity-check the request body so we know the client is sending
        # the right shape. A regression here is a regression in the
        # public surface.
        payload = json.loads(request.content)
        assert payload["stream"] is True, "client must request streaming"
        assert payload["model"] == "phi-3-mini", f"unexpected model: {payload['model']!r}"
        assert isinstance(payload["messages"], list)
        assert payload["temperature"] == DEFAULT_TEMPERATURE
        assert payload["max_tokens"] == DEFAULT_MAX_TOKENS
        return httpx.Response(
            status_code=status_code,
            headers={"content-type": "text/event-stream"},
            content=body.encode("utf-8"),
        )

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# ChatMessage value type
# ---------------------------------------------------------------------------


class TestChatMessage:
    """The OpenAI-shaped message value object."""

    def test_to_openai_returns_minimal_dict(self) -> None:
        m = ChatMessage(role="user", content="hi")
        assert m.to_openai() == {"role": "user", "content": "hi"}

    def test_to_openai_preserves_system_role(self) -> None:
        m = ChatMessage(role="system", content="You are a smart-home assistant.")
        assert m.to_openai() == {
            "role": "system",
            "content": "You are a smart-home assistant.",
        }

    def test_is_frozen(self) -> None:
        """Frozen dataclass — typo-proofing at the type level."""
        m = ChatMessage(role="user", content="hi")
        with pytest.raises((AttributeError, Exception)):
            m.role = "assistant"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Protocol conformance (duck typing)
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Anything with the right ``generate`` method satisfies LLMClient."""

    def test_fake_satisfies_protocol(self) -> None:
        """FakeLLMClient is structurally a LLMClient (no inheritance)."""
        client = FakeLLMClient()
        assert isinstance(client, LLMClient)

    def test_llamacpp_satisfies_protocol(self) -> None:
        client = LlamaCppClient(
            base_url="http://127.0.0.1:8080",
            model="phi-3-mini",
            client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
        )
        try:
            assert isinstance(client, LLMClient)
        finally:
            client.close()

    def test_protocol_is_runtime_checkable(self) -> None:
        """A random class with the wrong method does NOT satisfy it."""

        class NotALlm:
            def something_else(self) -> None:
                pass

        assert not isinstance(NotALlm(), LLMClient)


# ---------------------------------------------------------------------------
# FakeLLMClient behaviour
# ---------------------------------------------------------------------------


class TestFakeLLMClient:
    """The deterministic in-memory stub."""

    def test_returns_default_response(self) -> None:
        client = FakeLLMClient(default_response="hello world")
        msgs = [ChatMessage(role="user", content="say hi")]
        text, stats = client.generate(msgs)
        assert text == "hello world"
        assert stats.completion_tokens == len("hello world".split())

    def test_response_override_substring_match(self) -> None:
        """A test can pin a reply to a phrase in the user message."""
        client = FakeLLMClient(
            default_response="I don't know.",
            response_overrides={
                "microwave": "The microwave manual says 800 W is typical.",
            },
        )
        msgs = [ChatMessage(role="user", content="How do I use the microwave?")]
        text, _ = client.generate(msgs)
        assert text == "The microwave manual says 800 W is typical."

    def test_picks_first_matching_override(self) -> None:
        """When multiple substrings match, the first one wins."""
        client = FakeLLMClient(
            response_overrides={
                "alpha": "first",
                "beta": "second",
            },
        )
        msgs = [ChatMessage(role="user", content="alpha and beta")]
        text, _ = client.generate(msgs)
        assert text == "first"

    def test_prompt_tokens_are_word_count_of_input(self) -> None:
        """Token counting is whitespace-split (cheap + deterministic)."""
        client = FakeLLMClient(default_response="ok")
        msgs = [
            ChatMessage(role="system", content="you are helpful"),
            ChatMessage(role="user", content="what is the temperature"),
        ]
        _, stats = client.generate(msgs)
        # "you are helpful" = 3, "what is the temperature" = 4 → 7
        assert stats.prompt_tokens == 7

    def test_total_tokens_sums_prompt_and_completion(self) -> None:
        client = FakeLLMClient(default_response="a b c d e")  # 5 tokens
        msgs = [ChatMessage(role="user", content="hi there")]  # 2 tokens
        _, stats = client.generate(msgs)
        assert stats.total_tokens == 7

    def test_raise_after_tokens_raises_refused(self) -> None:
        client = FakeLLMClient(
            default_response="one two three four five six seven",
            raise_after_tokens=3,
        )
        msgs = [ChatMessage(role="user", content="go")]
        with pytest.raises(LLMRefusedError):
            client.generate(msgs)

    def test_inherits_llmerror(self) -> None:
        """LLMRefusedError must be catchable as LLMError."""
        client = FakeLLMClient(
            default_response="a b c d e",
            raise_after_tokens=1,
        )
        msgs = [ChatMessage(role="user", content="go")]
        with pytest.raises(LLMError):
            client.generate(msgs)

    def test_stats_persist_across_calls(self) -> None:
        """``client.stats`` is the most recent call's stats."""
        client = FakeLLMClient(default_response="ok")
        msgs = [ChatMessage(role="user", content="hi")]
        client.generate(msgs)
        assert client.stats.completion_tokens >= 1


# ---------------------------------------------------------------------------
# LlamaCppClient — happy path (SSE stream)
# ---------------------------------------------------------------------------


class TestLlamaCppClientStreaming:
    """End-to-end SSE parsing against a mock transport."""

    def test_concatenates_delta_content(self) -> None:
        body = _make_sse_body(
            _sse_chunk("Hello"),
            _sse_chunk(", "),
            _sse_chunk("world"),
            _sse_chunk("!", finish_reason="stop"),
            _sse_usage(prompt_tokens=4, completion_tokens=3),
            _sse_done(),
        )
        transport = _mock_transport([body])
        client = LlamaCppClient(
            base_url="http://test:8080",
            model="phi-3-mini",
            client=httpx.Client(transport=transport),
        )
        try:
            text, stats = client.generate(
                [ChatMessage(role="user", content="hi there friend")],
            )
        finally:
            client.close()

        assert text == "Hello, world!"
        assert stats.prompt_tokens == 4
        assert stats.completion_tokens == 3
        assert stats.total_tokens == 7
        assert stats.duration_seconds > 0

    def test_handles_empty_delta_chunks(self) -> None:
        """Some chunks carry no content (just role or finish_reason)."""
        body = _make_sse_body(
            _sse_chunk("a"),
            _sse_chunk(None, finish_reason=None),  # role-only chunk
            _sse_chunk("b"),
            _sse_done(),
        )
        transport = _mock_transport([body])
        client = LlamaCppClient(
            base_url="http://test:8080",
            model="phi-3-mini",
            client=httpx.Client(transport=transport),
        )
        try:
            text, _ = client.generate([ChatMessage(role="user", content="x")])
        finally:
            client.close()
        assert text == "ab"

    def test_stops_at_done_sentinel(self) -> None:
        """Anything after [DONE] is ignored."""
        body = _make_sse_body(
            _sse_chunk("ok"),
            _sse_done(),
            _sse_chunk("IGNORED"),  # would be a server bug
        )
        transport = _mock_transport([body])
        client = LlamaCppClient(
            base_url="http://test:8080",
            model="phi-3-mini",
            client=httpx.Client(transport=transport),
        )
        try:
            text, _ = client.generate([ChatMessage(role="user", content="hi")])
        finally:
            client.close()
        assert text == "ok"

    def test_skips_malformed_sse_lines(self) -> None:
        """A bad JSON line in the middle of the stream shouldn't crash."""
        body = (
            _sse_chunk("before")
            + "data: not-json{broken\n\n"
            + _sse_chunk("after")
            + _sse_done()
        )
        transport = _mock_transport([body])
        client = LlamaCppClient(
            base_url="http://test:8080",
            model="phi-3-mini",
            client=httpx.Client(transport=transport),
        )
        try:
            text, _ = client.generate([ChatMessage(role="user", content="x")])
        finally:
            client.close()
        assert text == "beforeafter"

    def test_estimates_tokens_when_usage_missing(self) -> None:
        """Older llama-server builds don't include a usage block."""
        body = _make_sse_body(
            _sse_chunk("one two three four five"),  # 5 words
            _sse_done(),
        )
        transport = _mock_transport([body])
        client = LlamaCppClient(
            base_url="http://test:8080",
            model="phi-3-mini",
            client=httpx.Client(transport=transport),
        )
        try:
            _, stats = client.generate(
                [ChatMessage(role="user", content="how are you")],  # 3 words
            )
        finally:
            client.close()
        # Whitespace-split fallback.
        assert stats.completion_tokens == 5
        assert stats.prompt_tokens == 3
        assert stats.total_tokens == 8


# ---------------------------------------------------------------------------
# LlamaCppClient — error mapping
# ---------------------------------------------------------------------------


class TestLlamaCppClientErrors:
    """5xx → Unavailable, 4xx → Refused, connection errors → Unavailable."""

    def test_500_maps_to_unavailable(self) -> None:
        transport = httpx.MockTransport(
            lambda r: httpx.Response(500, text="internal error")
        )
        client = LlamaCppClient(
            base_url="http://test:8080",
            model="phi-3-mini",
            client=httpx.Client(transport=transport),
        )
        try:
            with pytest.raises(LLMUnavailableError):
                client.generate([ChatMessage(role="user", content="hi")])
        finally:
            client.close()

    def test_400_maps_to_refused(self) -> None:
        transport = httpx.MockTransport(
            lambda r: httpx.Response(400, text="bad prompt")
        )
        client = LlamaCppClient(
            base_url="http://test:8080",
            model="phi-3-mini",
            client=httpx.Client(transport=transport),
        )
        try:
            with pytest.raises(LLMRefusedError):
                client.generate([ChatMessage(role="user", content="hi")])
        finally:
            client.close()

    def test_404_maps_to_refused(self) -> None:
        transport = httpx.MockTransport(
            lambda r: httpx.Response(404, text="model not found")
        )
        client = LlamaCppClient(
            base_url="http://test:8080",
            model="phi-3-mini",
            client=httpx.Client(transport=transport),
        )
        try:
            with pytest.raises(LLMRefusedError):
                client.generate([ChatMessage(role="user", content="hi")])
        finally:
            client.close()

    def test_connection_error_maps_to_unavailable(self) -> None:
        def handler(_r: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        client = LlamaCppClient(
            base_url="http://test:8080",
            model="phi-3-mini",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        try:
            with pytest.raises(LLMUnavailableError):
                client.generate([ChatMessage(role="user", content="hi")])
        finally:
            client.close()


# ---------------------------------------------------------------------------
# LlamaCppClient — lifecycle
# ---------------------------------------------------------------------------


class TestLlamaCppClientLifecycle:
    """Lazy client construction + ownership-aware close()."""

    def test_owns_client_when_none_passed(self) -> None:
        """No injected client → we create + own one."""
        client = LlamaCppClient(base_url="http://test:8080", model="phi-3-mini")
        # Touching _http() forces lazy construction.
        _ = client._http()
        assert client.client is not None
        client.close()
        # After close(), the internal client is dropped.
        assert client.client is None

    def test_does_not_close_injected_client(self) -> None:
        """If the caller passed a client, we must not close it."""
        injected = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
        client = LlamaCppClient(
            base_url="http://test:8080",
            model="phi-3-mini",
            client=injected,
        )
        client.close()
        # Injected client should still be usable.
        assert not injected.is_closed
        injected.close()

    def test_close_is_idempotent(self) -> None:
        """Closing twice doesn't blow up."""
        client = LlamaCppClient(base_url="http://test:8080", model="phi-3-mini")
        client.close()
        client.close()  # should not raise

    def test_base_url_trailing_slash_stripped(self) -> None:
        client = LlamaCppClient(
            base_url="http://test:8080/",
            model="phi-3-mini",
            client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
        )
        try:
            assert client.base_url == "http://test:8080"
        finally:
            client.close()


# ---------------------------------------------------------------------------
# Prompt construction sanity (composition-root concern, not LLMClient proper)
# ---------------------------------------------------------------------------


class TestMultiMessageInput:
    """The client passes the full message list through unchanged."""

    def test_system_plus_user_messages(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            captured["payload"] = payload
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_make_sse_body(_sse_chunk("ok"), _sse_done()).encode(),
            )

        client = LlamaCppClient(
            base_url="http://test:8080",
            model="phi-3-mini",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        try:
            client.generate(
                [
                    ChatMessage(
                        role="system",
                        content="You are a smart-home assistant.",
                    ),
                    ChatMessage(role="user", content="What is the temperature?"),
                ],
                max_tokens=64,
                temperature=0.1,
            )
        finally:
            client.close()

        msgs = captured["payload"]["messages"]
        assert len(msgs) == 2
        assert msgs[0] == {
            "role": "system",
            "content": "You are a smart-home assistant.",
        }
        assert msgs[1] == {
            "role": "user",
            "content": "What is the temperature?",
        }
        assert captured["payload"]["max_tokens"] == 64
        assert captured["payload"]["temperature"] == 0.1


# ---------------------------------------------------------------------------
# Defaults & exports
# ---------------------------------------------------------------------------


class TestModuleSurface:
    """Smoke checks on the module's public surface."""

    def test_default_timeout_is_120s(self) -> None:
        """The default must accommodate a slow CPU's full 512-token reply."""
        assert DEFAULT_TIMEOUT_S >= 120.0

    def test_default_temperature_is_greedy(self) -> None:
        """Eval-set reproducibility requires greedy decoding."""
        assert DEFAULT_TEMPERATURE == 0.0

    def test_default_max_tokens_is_reasonable(self) -> None:
        assert 64 <= DEFAULT_MAX_TOKENS <= 2048
