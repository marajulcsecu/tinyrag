"""Tests for tinyrag.core.prompt_builder (Step 4.11 — grounded prompt construction).

Test layout
-----------
- TestPublicSurface             — every documented symbol is importable
  (PromptBuilder, Prompt, PromptBuilderError, the 3 module constants,
  the default factory, the from_chunking_settings helper).
- TestPromptDataclass           — the Prompt value type is frozen, has
  the documented fields, exposes the used_trimming convenience
  property.
- TestBuilderConstruction       — happy-path construction, default
  values, from_chunking_settings wiring, validation of bad args
  (empty encoding, negative limits, reserved >= max, empty system
  prompt, bad encoding name).
- TestBuildNoChunks             — 0 chunks produces a usable refusal
  prompt: 2 messages (system + user), system prompt carries the
  refusal instruction, used=0, dropped=0, prompt_tokens > 0.
- TestBuildOneChunk             — a single chunk fits, is numbered
  [1], carries the source+page header, end-to-end token count is
  sane.
- TestBuildMaxChunks            — N chunks fit within budget, all
  numbered, citation ids are contiguous [1..N].
- TestBuildVeryLongChunks       — the trimming path drops tail
  chunks to fit the budget; used_trimming=True; chunks_dropped>0;
  the dropped chunks are the LATER ones (preserves similarity
  ranking).
- TestBuildEmptyTextSkipped     — chunks with empty/whitespace text
  are silently skipped, don't inflate the count.
- TestBuildCitationFormat       — every chunk in the context block
  is rendered as ``[N] (source, p.X) <text>`` and the question
  line uses the documented USER_MESSAGE_TEMPLATE format.
- TestBuildEmptyQueryRaises     — empty / whitespace query raises
  PromptBuilderError.
- TestCountTokens               — count_tokens uses tiktoken under
  the hood (matches a hand-encoded known string).
- TestPromptFitsBudget          — every built prompt has
  prompt_tokens <= max_prompt_tokens (the budget invariant).
- TestMessagesShape             — the returned Prompt.messages is
  always exactly 2 entries (system + user), in that order, with
  the expected roles.
- TestMessagesPassableToLlm     — the assembled messages can be
  passed straight into FakeLLMClient.generate() without rewriting
  (this is the integration glue test that pins the LLMClient
  seam contract from Step 4.10).
- TestCustomSystemPrompt        — a caller-supplied system prompt
  is reflected verbatim in the Prompt.

Hermetic?
---------
100% hermetic. No network, no llama-server, no model weights. Uses
the FakeLLMClient from Step 4.10 for the integration test (which
itself never touches the network).

Location: ``tests/test_prompt_builder.py``
"""

from __future__ import annotations

import pytest
import tiktoken

from tinyrag.config import ChunkingSettings
from tinyrag.core import Chunk
from tinyrag.core.prompt_builder import (
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_RESERVED_FOR_ANSWER_TOKENS,
    DEFAULT_SYSTEM_PROMPT,
    USER_MESSAGE_TEMPLATE,
    Prompt,
    PromptBuilder,
    PromptBuilderError,
    default_prompt_builder,
)
from tinyrag.generation import ChatMessage, FakeLLMClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    text: str,
    *,
    source: str = "test-doc.pdf",
    page: int | None = 1,
    chunk_index: int = 0,
    char_offset: int = 0,
    token_count: int | None = None,
) -> Chunk:
    """Construct a Chunk with sensible defaults for tests."""
    if token_count is None:
        # Approximate token count from whitespace split. Tests don't
        # need exact numbers — just non-zero values.
        token_count = max(1, len(text.split()))
    return Chunk(
        text=text,
        source=source,
        page=page,
        chunk_index=chunk_index,
        char_offset=char_offset,
        token_count=token_count,
    )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    """The module exposes the documented symbols."""

    def test_prompt_builder_class(self) -> None:
        assert callable(PromptBuilder)

    def test_prompt_dataclass(self) -> None:
        p = Prompt()
        assert isinstance(p, Prompt)
        assert p.messages == []
        assert p.system_prompt == ""
        assert p.user_message == ""

    def test_prompt_builder_error_is_value_error(self) -> None:
        # Inherits ValueError so callers can catch both with `except ValueError`.
        assert issubclass(PromptBuilderError, ValueError)

    def test_default_factory_returns_builder(self) -> None:
        b = default_prompt_builder()
        assert isinstance(b, PromptBuilder)

    def test_module_constants_present(self) -> None:
        assert DEFAULT_MAX_PROMPT_TOKENS == 4096
        assert DEFAULT_RESERVED_FOR_ANSWER_TOKENS == 512
        assert DEFAULT_SYSTEM_PROMPT  # non-empty
        assert USER_MESSAGE_TEMPLATE  # non-empty

    def test_default_factory_uses_default_constants(self) -> None:
        b = default_prompt_builder()
        assert b.encoding_name == "cl100k_base"
        assert b.max_prompt_tokens == DEFAULT_MAX_PROMPT_TOKENS
        assert b.system_prompt == DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Prompt dataclass
# ---------------------------------------------------------------------------


class TestPromptDataclass:
    """The frozen value type returned by PromptBuilder.build()."""

    def test_is_frozen(self) -> None:
        p = Prompt()
        with pytest.raises((AttributeError, Exception)):
            p.prompt_tokens = 999  # type: ignore[misc]

    def test_used_trimming_default_false(self) -> None:
        assert Prompt().used_trimming is False

    def test_used_trimming_true_when_dropped(self) -> None:
        p = Prompt(chunks_dropped=1)
        assert p.used_trimming is True

    def test_used_trimming_false_when_zero_dropped(self) -> None:
        p = Prompt(chunks_dropped=0)
        assert p.used_trimming is False

    def test_required_fields_have_defaults(self) -> None:
        # All fields should be constructable positionally or by name.
        p = Prompt(
            messages=[],
            system_prompt="sys",
            user_message="user",
            prompt_tokens=42,
            chunks_used=0,
            chunks_dropped=0,
            encoding_name="cl100k_base",
        )
        assert p.prompt_tokens == 42


# ---------------------------------------------------------------------------
# PromptBuilder construction
# ---------------------------------------------------------------------------


class TestBuilderConstruction:
    """PromptBuilder validates its inputs at construction time."""

    def test_defaults_via_constructor(self) -> None:
        b = PromptBuilder()
        assert b.encoding_name == "cl100k_base"
        assert b.max_prompt_tokens == DEFAULT_MAX_PROMPT_TOKENS

    def test_custom_overrides(self) -> None:
        b = PromptBuilder(
            encoding_name="p50k_base",
            max_prompt_tokens=2048,
            reserved_for_answer_tokens=256,
            system_prompt="Custom system prompt.",
        )
        assert b.encoding_name == "p50k_base"
        assert b.max_prompt_tokens == 2048
        assert b.system_prompt == "Custom system prompt."

    def test_from_chunking_settings_uses_encoding(self) -> None:
        chunking = ChunkingSettings(encoding="p50k_base")
        b = PromptBuilder.from_chunking_settings(chunking)
        assert b.encoding_name == "p50k_base"

    def test_from_chunking_settings_respects_budget(self) -> None:
        chunking = ChunkingSettings()
        b = PromptBuilder.from_chunking_settings(
            chunking, max_prompt_tokens=2048, reserved_for_answer_tokens=128
        )
        assert b.max_prompt_tokens == 2048

    def test_empty_encoding_name_rejected(self) -> None:
        with pytest.raises(PromptBuilderError, match="encoding_name"):
            PromptBuilder(encoding_name="")

    def test_zero_max_prompt_tokens_rejected(self) -> None:
        with pytest.raises(PromptBuilderError, match="max_prompt_tokens"):
            PromptBuilder(max_prompt_tokens=0)

    def test_negative_max_prompt_tokens_rejected(self) -> None:
        with pytest.raises(PromptBuilderError, match="max_prompt_tokens"):
            PromptBuilder(max_prompt_tokens=-1)

    def test_negative_reserved_rejected(self) -> None:
        with pytest.raises(PromptBuilderError, match="reserved_for_answer_tokens"):
            PromptBuilder(reserved_for_answer_tokens=-1)

    def test_reserved_geq_max_rejected(self) -> None:
        with pytest.raises(PromptBuilderError, match="must be <"):
            PromptBuilder(max_prompt_tokens=512, reserved_for_answer_tokens=512)

    def test_reserved_greater_than_max_rejected(self) -> None:
        with pytest.raises(PromptBuilderError, match="must be <"):
            PromptBuilder(max_prompt_tokens=100, reserved_for_answer_tokens=200)

    def test_empty_system_prompt_rejected(self) -> None:
        with pytest.raises(PromptBuilderError, match="system_prompt"):
            PromptBuilder(system_prompt="")

    def test_unknown_encoding_rejected(self) -> None:
        with pytest.raises(PromptBuilderError, match="unknown tiktoken encoding"):
            PromptBuilder(encoding_name="bogus-encoding-xyz")


# ---------------------------------------------------------------------------
# count_tokens helper
# ---------------------------------------------------------------------------


class TestCountTokens:
    """count_tokens delegates to tiktoken with the configured encoding."""

    def test_matches_tiktoken_on_small_string(self) -> None:
        b = PromptBuilder()
        text = "Hello world"
        expected = len(tiktoken.get_encoding(b.encoding_name).encode(text))
        assert b.count_tokens(text) == expected

    def test_handles_empty_string(self) -> None:
        b = PromptBuilder()
        assert b.count_tokens("") == 0

    def test_different_encodings_produce_different_counts(self) -> None:
        # cl100k_base (GPT-4) and p50k_base (Codex) tokenise the
        # same string differently — this proves the encoding choice
        # actually flows through.
        text = "function add(a, b) { return a + b; }"
        a = PromptBuilder(encoding_name="cl100k_base").count_tokens(text)
        b = PromptBuilder(encoding_name="p50k_base").count_tokens(text)
        assert a != b


# ---------------------------------------------------------------------------
# Build path: 0 chunks (refusal path)
# ---------------------------------------------------------------------------


class TestBuildNoChunks:
    """Zero chunks → 2 messages, system prompt carries the refusal text."""

    def test_returns_prompt(self) -> None:
        b = PromptBuilder()
        p = b.build("What is the temperature?", [])
        assert isinstance(p, Prompt)

    def test_two_messages(self) -> None:
        p = default_prompt_builder().build("What is the temperature?", [])
        assert len(p.messages) == 2

    def test_message_roles_system_then_user(self) -> None:
        p = default_prompt_builder().build("Q?", [])
        assert p.messages[0].role == "system"
        assert p.messages[1].role == "user"

    def test_system_prompt_carries_refusal_instruction(self) -> None:
        p = default_prompt_builder().build("Q?", [])
        assert "I don't have enough information" in p.system_prompt

    def test_chunks_used_zero_dropped_zero(self) -> None:
        p = default_prompt_builder().build("Q?", [])
        assert p.chunks_used == 0
        assert p.chunks_dropped == 0
        assert p.used_trimming is False

    def test_user_message_is_just_the_question(self) -> None:
        p = default_prompt_builder().build("What is 2+2?", [])
        # No context block — just the template.
        assert p.user_message == USER_MESSAGE_TEMPLATE.format(question="What is 2+2?")

    def test_no_citation_brackets_in_user_message(self) -> None:
        # Without context, the user message has no [N] markers.
        p = default_prompt_builder().build("Q?", [])
        assert "[1]" not in p.user_message

    def test_prompt_tokens_accounts_for_both_messages(self) -> None:
        b = default_prompt_builder()
        p = b.build("Q?", [])
        expected = b.count_tokens(p.system_prompt) + b.count_tokens(p.user_message)
        assert p.prompt_tokens == expected


# ---------------------------------------------------------------------------
# Build path: 1 chunk
# ---------------------------------------------------------------------------


class TestBuildOneChunk:
    """A single chunk fits, is numbered [1], carries the source+page header."""

    def test_chunks_used_one(self) -> None:
        p = default_prompt_builder().build(
            "Q?", [_chunk("Some text.", page=5, chunk_index=3)]
        )
        assert p.chunks_used == 1
        assert p.chunks_dropped == 0

    def test_chunk_numbered_one(self) -> None:
        p = default_prompt_builder().build(
            "Q?", [_chunk("Some text.", page=5, chunk_index=3)]
        )
        assert "[1]" in p.user_message
        # And NOT [2] or higher.
        assert "[2]" not in p.user_message

    def test_chunk_carries_source_and_page(self) -> None:
        p = default_prompt_builder().build(
            "Q?", [_chunk("Some text.", source="manual.pdf", page=12)]
        )
        assert "manual.pdf" in p.user_message
        assert "p.12" in p.user_message

    def test_text_appears_in_context_block(self) -> None:
        p = default_prompt_builder().build(
            "Q?", [_chunk("The unique marker text xyz123.")]
        )
        assert "The unique marker text xyz123." in p.user_message

    def test_question_line_appears(self) -> None:
        p = default_prompt_builder().build(
            "What is the password?", [_chunk("The password is 1234.")]
        )
        assert "Question: What is the password?" in p.user_message


# ---------------------------------------------------------------------------
# Build path: max chunks (budget respected)
# ---------------------------------------------------------------------------


class TestBuildMaxChunks:
    """N small chunks fit, numbered contiguously [1..N]."""

    def test_three_chunks_numbered_1_to_3(self) -> None:
        chunks = [
            _chunk(f"Chunk number {i}.", source="doc.pdf", page=i)
            for i in range(1, 4)
        ]
        p = default_prompt_builder().build("Q?", chunks)
        assert p.chunks_used == 3
        assert p.chunks_dropped == 0
        for i in range(1, 4):
            assert f"[{i}]" in p.user_message

    def test_ten_chunks_all_fit(self) -> None:
        # 10 short chunks at the default 4096-token budget — plenty
        # of headroom.
        chunks = [_chunk(f"Chunk {i}.", chunk_index=i) for i in range(10)]
        p = default_prompt_builder().build("Q?", chunks)
        assert p.chunks_used == 10
        assert p.chunks_dropped == 0

    def test_fits_within_budget(self) -> None:
        chunks = [_chunk(f"Word {i} " * 50, chunk_index=i) for i in range(20)]
        b = PromptBuilder(max_prompt_tokens=2048, reserved_for_answer_tokens=256)
        p = b.build("Q?", chunks)
        # Every prompt must fit the configured budget.
        assert p.prompt_tokens <= b.max_prompt_tokens


# ---------------------------------------------------------------------------
# Build path: very long chunks → trimming drops tail
# ---------------------------------------------------------------------------


class TestBuildVeryLongChunks:
    """Trimming drops the LATER chunks (preserves similarity ranking)."""

    def test_drops_tail_to_fit_budget(self) -> None:
        # Six chunks of ~600 tokens each. Budget = 1500 - 256 = 1244.
        # Only the first chunk fits.
        chunks = [
            _chunk(
                ("word " * 600) + f" END_{i}",
                source=f"doc-{i}.pdf",
                chunk_index=i,
            )
            for i in range(6)
        ]
        b = PromptBuilder(max_prompt_tokens=1500, reserved_for_answer_tokens=256)
        p = b.build("Q?", chunks)
        assert p.chunks_used < 6
        assert p.chunks_dropped > 0
        assert p.used_trimming is True

    def test_drops_later_chunks_preserves_ranking(self) -> None:
        # The DROP should be the LAST N chunks, not the first N.
        # We verify by including a unique marker in chunk 0 and
        # chunk 5 and confirming chunk 0's marker survives.
        chunks = [
            _chunk(
                ("FIRST_CHUNK_MARKER " * 200) + " padding",  # ~600 tokens
                chunk_index=0,
            ),
            _chunk("middle", chunk_index=1),
            _chunk("middle", chunk_index=2),
            _chunk("middle", chunk_index=3),
            _chunk("middle", chunk_index=4),
            _chunk(
                ("LAST_CHUNK_MARKER " * 200) + " padding",  # ~600 tokens
                chunk_index=5,
            ),
        ]
        b = PromptBuilder(max_prompt_tokens=1500, reserved_for_answer_tokens=256)
        p = b.build("Q?", chunks)
        # The first chunk should still be present; the last was dropped.
        if p.chunks_dropped > 0:
            assert "FIRST_CHUNK_MARKER" in p.user_message
            assert "LAST_CHUNK_MARKER" not in p.user_message

    def test_used_trimming_when_at_least_one_dropped(self) -> None:
        chunks = [_chunk(("word " * 600) + f" END_{i}") for i in range(6)]
        b = PromptBuilder(max_prompt_tokens=1500, reserved_for_answer_tokens=256)
        p = b.build("Q?", chunks)
        assert p.used_trimming is True

    def test_prompt_fits_budget_after_trim(self) -> None:
        chunks = [_chunk(("word " * 600) + f" END_{i}") for i in range(6)]
        b = PromptBuilder(max_prompt_tokens=1500, reserved_for_answer_tokens=256)
        p = b.build("Q?", chunks)
        assert p.prompt_tokens <= b.max_prompt_tokens


# ---------------------------------------------------------------------------
# Build path: empty-text chunks are skipped
# ---------------------------------------------------------------------------


class TestBuildEmptyTextSkipped:
    """Chunks with empty/whitespace text don't inflate the chunk count."""

    def test_empty_string_chunk_skipped(self) -> None:
        chunks = [
            _chunk(""),  # empty
            _chunk("Real text."),
        ]
        p = default_prompt_builder().build("Q?", chunks)
        # Only the real chunk counts.
        assert p.chunks_used == 1
        assert "Real text." in p.user_message
        assert "[1]" in p.user_message

    def test_whitespace_only_chunk_skipped(self) -> None:
        chunks = [
            _chunk("   \n\t  "),
            _chunk("Real text."),
        ]
        p = default_prompt_builder().build("Q?", chunks)
        assert p.chunks_used == 1

    def test_all_empty_chunks_yields_zero_chunks_used(self) -> None:
        chunks = [_chunk(""), _chunk("   "), _chunk("\n\n")]
        p = default_prompt_builder().build("Q?", chunks)
        assert p.chunks_used == 0
        # System prompt still tells the model to refuse.
        assert "I don't have enough information" in p.system_prompt


# ---------------------------------------------------------------------------
# Build path: citation format
# ---------------------------------------------------------------------------


class TestBuildCitationFormat:
    """Every chunk is rendered as ``[N] (source, p.X) <text>``."""

    def test_chunk_with_page(self) -> None:
        p = default_prompt_builder().build(
            "Q?", [_chunk("hello", source="manual.pdf", page=7)]
        )
        # The exact format string used by _format_chunk.
        assert "[1] (manual.pdf, p.7) hello" in p.user_message

    def test_chunk_without_page(self) -> None:
        p = default_prompt_builder().build(
            "Q?", [_chunk("hello", source="notes.txt", page=None)]
        )
        # No "p." when page is None.
        assert "[1] (notes.txt) hello" in p.user_message
        assert "p." not in p.user_message.split("Question:")[0]

    def test_question_uses_documented_template(self) -> None:
        p = default_prompt_builder().build(
            "my question", [_chunk("ctx")]
        )
        assert USER_MESSAGE_TEMPLATE.format(question="my question") in p.user_message

    def test_citation_ids_are_contiguous(self) -> None:
        chunks = [_chunk(f"c{i}", chunk_index=i) for i in range(5)]
        p = default_prompt_builder().build("Q?", chunks)
        for i in range(1, 6):
            assert f"[{i}]" in p.user_message
        # And no higher number leaked in.
        assert "[6]" not in p.user_message


# ---------------------------------------------------------------------------
# Build path: empty query
# ---------------------------------------------------------------------------


class TestBuildEmptyQueryRaises:
    """Empty / whitespace queries raise PromptBuilderError."""

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(PromptBuilderError, match="query"):
            default_prompt_builder().build("", [_chunk("ctx")])

    def test_whitespace_only_rejected(self) -> None:
        with pytest.raises(PromptBuilderError, match="query"):
            default_prompt_builder().build("   \n\t  ", [_chunk("ctx")])

    def test_query_rejected_even_with_empty_chunks(self) -> None:
        with pytest.raises(PromptBuilderError, match="query"):
            default_prompt_builder().build("", [])


# ---------------------------------------------------------------------------
# Budget invariant — every prompt fits the configured budget
# ---------------------------------------------------------------------------


class TestPromptFitsBudget:
    """Hard invariant: prompt_tokens <= max_prompt_tokens for every build."""

    def test_default_budget_respected_with_many_chunks(self) -> None:
        chunks = [_chunk(("x " * 100) + f" end{i}") for i in range(50)]
        b = default_prompt_builder()
        p = b.build("Q?", chunks)
        assert p.prompt_tokens <= b.max_prompt_tokens

    def test_custom_budget_respected(self) -> None:
        chunks = [_chunk(("x " * 100) + f" end{i}") for i in range(50)]
        b = PromptBuilder(max_prompt_tokens=512, reserved_for_answer_tokens=64)
        p = b.build("Q?", chunks)
        assert p.prompt_tokens <= b.max_prompt_tokens

    def test_empty_chunks_prompt_within_budget(self) -> None:
        b = default_prompt_builder()
        p = b.build("Q?", [])
        assert p.prompt_tokens <= b.max_prompt_tokens


# ---------------------------------------------------------------------------
# Messages shape — always exactly 2, in the right order
# ---------------------------------------------------------------------------


class TestMessagesShape:
    """The returned Prompt.messages is always exactly [system, user]."""

    def test_two_messages_no_chunks(self) -> None:
        p = default_prompt_builder().build("Q?", [])
        assert [m.role for m in p.messages] == ["system", "user"]

    def test_two_messages_with_chunks(self) -> None:
        chunks = [_chunk("ctx1"), _chunk("ctx2"), _chunk("ctx3")]
        p = default_prompt_builder().build("Q?", chunks)
        assert [m.role for m in p.messages] == ["system", "user"]

    def test_messages_are_chatmessage_instances(self) -> None:
        p = default_prompt_builder().build("Q?", [_chunk("ctx")])
        for m in p.messages:
            assert isinstance(m, ChatMessage)

    def test_messages_match_documented_fields(self) -> None:
        p = default_prompt_builder().build("Q?", [_chunk("ctx")])
        assert p.messages[0].content == p.system_prompt
        assert p.messages[1].content == p.user_message


# ---------------------------------------------------------------------------
# Integration with FakeLLMClient — proves the messages pass through cleanly
# ---------------------------------------------------------------------------


class TestMessagesPassableToLlm:
    """The PromptBuilder output is exactly what LLMClient.generate() wants."""

    def test_fake_llm_receives_correct_message_list(self) -> None:
        # FakeLLMClient.response_overrides matches on substring of any
        # message content. We pin a unique substring in the system
        # prompt and check it gets through.
        unique = "I don't have enough information in the provided documents."
        fake = FakeLLMClient(
            default_response="I don't know.",
            response_overrides={unique: "Refusal response from LLM."},
        )
        p = default_prompt_builder().build("Q?", [_chunk("ctx")])
        text, stats = fake.generate(p.messages)
        assert text == "Refusal response from LLM."
        assert stats.completion_tokens > 0

    def test_user_query_visible_in_messages(self) -> None:
        # The query substring must reach the LLM so it knows what was asked.
        fake = FakeLLMClient(
            default_response="ok",
            response_overrides={"unique_marker_query": "marker hit"},
        )
        p = default_prompt_builder().build(
            "unique_marker_query — please answer",
            [_chunk("ctx")],
        )
        text, _ = fake.generate(p.messages)
        assert text == "marker hit"

    def test_chunk_text_visible_in_messages(self) -> None:
        fake = FakeLLMClient(
            default_response="ok",
            response_overrides={"unique_chunk_marker": "chunk hit"},
        )
        p = default_prompt_builder().build(
            "Q?", [_chunk("This has unique_chunk_marker embedded.")]
        )
        text, _ = fake.generate(p.messages)
        assert text == "chunk hit"


# ---------------------------------------------------------------------------
# Custom system prompt override
# ---------------------------------------------------------------------------


class TestCustomSystemPrompt:
    """A caller-supplied system prompt is reflected verbatim."""

    def test_custom_system_prompt_used(self) -> None:
        custom = "You are a pirate. Answer in pirate-speak."
        b = PromptBuilder(system_prompt=custom)
        p = b.build("Q?", [_chunk("ctx")])
        assert p.system_prompt == custom
        assert p.messages[0].content == custom

    def test_custom_system_prompt_can_disable_refusal(self) -> None:
        # A custom prompt without "I don't have enough information"
        # means the model has no refusal instruction.
        b = PromptBuilder(system_prompt="Answer everything freely.")
        p = b.build("Q?", [])
        assert "I don't have enough information" not in p.system_prompt


# ---------------------------------------------------------------------------
# Diagnostics — encoding_name + chunks_used match what was passed in
# ---------------------------------------------------------------------------


class TestDiagnostics:
    """The Prompt reports what was used so callers can log it."""

    def test_encoding_name_recorded(self) -> None:
        b = PromptBuilder(encoding_name="p50k_base")
        p = b.build("Q?", [_chunk("ctx")])
        assert p.encoding_name == "p50k_base"

    def test_default_encoding_recorded(self) -> None:
        p = default_prompt_builder().build("Q?", [_chunk("ctx")])
        assert p.encoding_name == "cl100k_base"

    def test_chunks_used_matches_actual_in_context(self) -> None:
        chunks = [_chunk(f"c{i}", chunk_index=i) for i in range(3)]
        p = default_prompt_builder().build("Q?", chunks)
        assert p.chunks_used == 3
        # The actual number of [N] markers in the user message
        # should match chunks_used.
        count = sum(
            1 for n in range(1, 10) if f"[{n}]" in p.user_message
        )
        assert count == p.chunks_used
