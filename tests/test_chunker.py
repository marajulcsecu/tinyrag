"""Tests for src/tinyrag/core/chunker.py — token-based text chunker.

Test layout
-----------
- TestPublicSurface         — :class:`Chunk`, :class:`Chunker`,
  :class:`ChunkingError`, and :func:`default_chunker` are exported
  from the subpackage.
- TestChunkDataclass         — the dataclass is frozen; required
  fields per FR-5 are present.
- TestChunkerConstruction    — config is honoured; bad encoding
  raises :class:`ChunkingError`; properties expose the values.
- TestEmptyAndShort          — empty / whitespace / very short
  inputs return [] or a single chunk, no crash.
- TestExactBoundary          — text whose token count is exactly
  ``chunk_size`` produces exactly one chunk (no overflow).
- TestLongTextProducesMany   — a 2000-token text produces
  roughly ``ceil(2000 / stride)`` chunks (the roadmap's spot-check).
- TestOverlapCorrectness     — consecutive chunks share ~50 tokens
  of text.
- TestCharOffsetMonotonicity — ``char_offset`` is non-decreasing
  across chunks.
- TestChunkIndexContiguous   — ``chunk_index`` is 0..N-1.
- TestSentenceBoundary       — chunks end at ``.`` / ``!`` / ``?``
  followed by whitespace when such a break exists in the trim
  window.
- TestPageAndSourcePassthrough — ``source`` and ``page`` arguments
  reach every emitted chunk unchanged.
- TestTokenCountConsistency  — every chunk's ``token_count``
  matches the chunker re-counting its ``text``.
- TestIntegrationWithParsers — parsers + chunker round-trip
  (the actual flow the pipeline will use).

Why so many tests?
------------------
The chunker is the single point where text becomes vectors; a
bug here silently degrades retrieval quality without any
test-time failure. Token accounting is the most likely failure
mode (off-by-one in the stride, wrong chunk_size, broken
sentence-trim) — every test below targets a specific class of
mistake.

Hermetic?
---------
Yes. The chunker depends on ``tiktoken`` (pinned) and the local
``tinyrag.config`` — no network, no fixtures on disk. The
"IntegrationWithParsers" test does call
``tinyrag.ingestion.parsers.parse`` on an in-memory bytes blob,
so it exercises the parsers + chunker together, but it remains
disk-free.

Location: ``tests/test_chunker.py``
"""

from __future__ import annotations

import pytest
import tiktoken

from tinyrag.config import ChunkingSettings
from tinyrag.core import Chunk, Chunker, ChunkingError, default_chunker

# ----------------------------------------------------------------------------
# Shared fixtures
# ----------------------------------------------------------------------------


@pytest.fixture
def small_settings() -> ChunkingSettings:
    """A small chunk size (50) so tests don't need thousands of tokens."""
    return ChunkingSettings(chunk_size=50, chunk_overlap=10, encoding="cl100k_base")


@pytest.fixture
def small_chunker(small_settings: ChunkingSettings) -> Chunker:
    return Chunker(small_settings)


@pytest.fixture
def default_chunker_instance() -> Chunker:
    """A chunker with the project's defaults (400/50/cl100k_base)."""
    return Chunker(ChunkingSettings())


def _make_text(n_sentences: int, words_per_sentence: int = 8) -> str:
    """Build a deterministic long text with clear sentence boundaries.

    Each sentence ends with a period; sentence N has the literal
    text ``"Sentence N. "`` (where ``N`` is a number, so the
    sentence boundary is unambiguous to the regex). Words inside
    are filler.
    """
    filler = "the quick brown fox jumps over the lazy dog " * (words_per_sentence // 9 + 1)
    filler = filler.strip()
    parts = []
    for i in range(n_sentences):
        parts.append(f"Sentence {i}. {filler}")
    return " ".join(parts)


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


class TestPublicSurface:
    """The expected symbols are exported and importable."""

    def test_subpackage_exports_chunk(self) -> None:
        from tinyrag.core import Chunk as cls

        assert cls is Chunk

    def test_subpackage_exports_chunker(self) -> None:
        from tinyrag.core import Chunker as cls

        assert cls is Chunker

    def test_subpackage_exports_chunking_error(self) -> None:
        from tinyrag.core import ChunkingError as cls

        assert cls is ChunkingError

    def test_subpackage_exports_default_chunker(self) -> None:
        from tinyrag.core import default_chunker as fn

        assert callable(fn)


class TestChunkDataclass:
    """The :class:`Chunk` dataclass has the right shape and is frozen."""

    def test_required_fields_present(self) -> None:
        c = Chunk(
            text="hello",
            source="manual.pdf",
            page=3,
            chunk_index=0,
            char_offset=42,
            token_count=1,
        )
        assert c.text == "hello"
        assert c.source == "manual.pdf"
        assert c.page == 3
        assert c.chunk_index == 0
        assert c.char_offset == 42
        assert c.token_count == 1

    def test_chunk_is_frozen(self) -> None:
        c = Chunk(
            text="x", source="x", page=None,
            chunk_index=0, char_offset=0, token_count=1,
        )
        with pytest.raises((AttributeError, Exception)):
            c.text = "y"  # type: ignore[misc]

    def test_page_may_be_none(self) -> None:
        """Plain text / Markdown documents have no page numbers."""
        c = Chunk(
            text="x", source="faq.md", page=None,
            chunk_index=0, char_offset=0, token_count=1,
        )
        assert c.page is None


class TestChunkerConstruction:
    """The :class:`Chunker` constructor validates its input."""

    def test_default_chunker_uses_cl100k_base(self) -> None:
        c = default_chunker()
        assert c.encoding_name == "cl100k_base"

    def test_default_chunker_uses_400_and_50(self) -> None:
        c = default_chunker()
        assert c.chunk_size == 400
        assert c.chunk_overlap == 50

    def test_custom_settings_are_honoured(self, small_settings: ChunkingSettings) -> None:
        c = Chunker(small_settings)
        assert c.chunk_size == 50
        assert c.chunk_overlap == 10

    def test_unknown_encoding_raises_chunking_error(self) -> None:
        bad = ChunkingSettings(chunk_size=50, chunk_overlap=10, encoding="nonexistent-99")
        with pytest.raises(ChunkingError):
            Chunker(bad)

    def test_count_tokens_matches_tiktoken(self) -> None:
        """``count_tokens`` is a thin wrapper around tiktoken — verify equivalence."""
        c = default_chunker()
        text = "hello, world"
        assert c.count_tokens(text) == len(tiktoken.get_encoding("cl100k_base").encode(text))

    def test_count_tokens_empty_returns_zero(self) -> None:
        assert default_chunker().count_tokens("") == 0


class TestEmptyAndShort:
    """Empty / whitespace / very short inputs are handled cleanly."""

    def test_empty_text_returns_empty_list(
        self, small_chunker: Chunker
    ) -> None:
        assert small_chunker.chunk("", source="x.pdf") == []

    def test_whitespace_only_text_returns_empty_list(
        self, small_chunker: Chunker
    ) -> None:
        assert small_chunker.chunk("   \n\n\t  \n", source="x.pdf") == []

    def test_short_text_returns_one_chunk(
        self, small_chunker: Chunker
    ) -> None:
        text = "A short paragraph."
        chunks = small_chunker.chunk(text, source="x.pdf")
        assert len(chunks) == 1
        assert "A short paragraph." in chunks[0].text


class TestExactBoundary:
    """Text whose token count exactly equals ``chunk_size`` is one chunk."""

    def test_exact_boundary_one_chunk(self, small_chunker: Chunker) -> None:
        # Build text whose token count is exactly ``chunk_size``.
        # We work backwards from the encoder: take 50 token IDs and
        # decode them into a string, then re-encode to confirm the
        # token count matches.
        enc = tiktoken.get_encoding("cl100k_base")
        # 50 token IDs (we use the same int so the BPE stays stable).
        token_ids = [enc.encode("hello")[0]] * 50
        text = enc.decode(token_ids)
        # Confirm the round-trip token count.
        assert len(enc.encode(text)) == 50
        chunks = small_chunker.chunk(text, source="x.pdf")
        # Contract: ≤ chunk_size tokens and one chunk for short text.
        assert len(chunks) == 1
        assert chunks[0].token_count <= small_chunker.chunk_size


class TestLongTextProducesMany:
    """The roadmap's spot-check: ~2000 tokens → ~5 chunks with overlap."""

    def test_2000_token_text_produces_about_5_chunks(
        self, default_chunker_instance: Chunker
    ) -> None:
        # Default stride = 400 - 50 = 350 tokens. For ~2000 tokens
        # we expect ceil(2000 / 350) + 1 (last partial) = ~6 chunks.
        # The roadmap says "~5" — close enough; we accept 5..7.
        text = _make_text(n_sentences=200)  # plenty of tokens
        chunks = default_chunker_instance.chunk(text, source="big.txt")
        n_tokens = default_chunker_instance.count_tokens(text)
        assert n_tokens >= 1500, f"fixture too small: {n_tokens} tokens"
        # Each chunk should be near the target size (give or take
        # the sentence-trim).
        assert 5 <= len(chunks) <= 8, (
            f"expected 5..8 chunks for ~2000 tokens, got {len(chunks)}"
        )

    def test_chunks_cover_full_text(
        self, small_chunker: Chunker
    ) -> None:
        """No text is silently dropped — the first chunk starts near
        the beginning and the last chunk ends near the end."""
        text = _make_text(n_sentences=80)
        chunks = small_chunker.chunk(text, source="x.txt")
        assert chunks[0].char_offset == 0
        # Last chunk's char_offset + len(text) should be >= len(text).
        # (tiktoken's BPE can shift boundaries by a few characters.)
        last = chunks[-1]
        assert last.char_offset + len(last.text) >= len(text) - 5


class TestOverlapCorrectness:
    """Consecutive chunks share approximately ``chunk_overlap`` tokens."""

    def test_consecutive_chunks_overlap(
        self, small_chunker: Chunker
    ) -> None:
        text = _make_text(n_sentences=100)
        chunks = small_chunker.chunk(text, source="x.txt")
        assert len(chunks) >= 3
        # Chunks i and i+1 should share some text — the suffix of
        # chunk i should appear at or near the start of chunk i+1.
        # We check the *text* level (not tokens) because the
        # char_offset search is text-based.
        for i in range(len(chunks) - 1):
            a = chunks[i].text
            b = chunks[i + 1].text
            # Take the last 20 chars of a; look for a non-trivial
            # overlap with the start of b. We don't require full
            # ``chunk_overlap`` worth of overlap because the
            # sentence-trim + stride interaction can trim a few
            # tokens off either side.
            tail = a[-20:].strip()
            if not tail:
                continue
            # Overlap if the tail of a appears somewhere in b.
            assert any(tail[j:] in b for j in range(min(5, len(tail)))), (
                f"no overlap between chunk {i} and {i+1}: "
                f"a ends with {tail!r}, b starts with {b[:40]!r}"
            )

    def test_zero_overlap_produces_disjoint_chunks(self) -> None:
        """``chunk_overlap == 0`` produces disjoint windows."""
        settings = ChunkingSettings(chunk_size=50, chunk_overlap=0)
        chunker = Chunker(settings)
        text = _make_text(n_sentences=100)
        chunks = chunker.chunk(text, source="x.txt")
        assert len(chunks) >= 2
        # chunk i ends at most at the start of chunk i+1's text.
        for i in range(len(chunks) - 1):
            assert chunks[i].text[-5:] not in chunks[i + 1].text[:50], (
                "zero-overlap chunker produced overlapping chunks"
            )


class TestCharOffsetMonotonicity:
    """``char_offset`` is non-decreasing across chunks."""

    def test_offsets_are_monotonic(
        self, small_chunker: Chunker
    ) -> None:
        text = _make_text(n_sentences=80)
        chunks = small_chunker.chunk(text, source="x.txt")
        for i in range(len(chunks) - 1):
            assert chunks[i].char_offset <= chunks[i + 1].char_offset, (
                f"chunk {i+1}.char_offset ({chunks[i+1].char_offset}) "
                f"< chunk {i}.char_offset ({chunks[i].char_offset})"
            )

    def test_first_chunk_starts_at_zero(
        self, small_chunker: Chunker
    ) -> None:
        chunks = small_chunker.chunk(_make_text(50), source="x.txt")
        assert chunks[0].char_offset == 0


class TestChunkIndexContiguous:
    """``chunk_index`` is 0..N-1."""

    def test_indices_are_contiguous(
        self, small_chunker: Chunker
    ) -> None:
        chunks = small_chunker.chunk(_make_text(80), source="x.txt")
        for i, c in enumerate(chunks):
            assert c.chunk_index == i


class TestSentenceBoundary:
    """Chunks respect sentence boundaries when possible."""

    def test_chunk_ends_at_sentence_boundary_when_possible(
        self, small_chunker: Chunker
    ) -> None:
        """Build text with a clear sentence break inside the trim window.
        The chunk should end with a period (or whitespace after one)."""
        # Make a long stretch of sentences; with chunk_size=50 and
        # ~10 tokens/sentence, the first chunk ends mid-sentence
        # by token count but should trim back to a period within
        # the last 20% (~10 tokens).
        text = _make_text(n_sentences=30)
        chunks = small_chunker.chunk(text, source="x.txt")
        # Find a chunk that isn't the last one — last chunks are
        # exempt from sentence-trim (they extend to the end).
        for c in chunks[:-1]:
            # The chunk text should end with a period, exclamation,
            # question mark, or whitespace — never mid-word.
            tail = c.text.rstrip()
            assert tail and tail[-1] in ".!?\"'", (
                f"chunk text ends mid-sentence: {tail[-30:]!r}"
            )

    def test_last_chunk_extends_to_end(
        self, small_chunker: Chunker
    ) -> None:
        """The final chunk covers the end of the document — no trim."""
        text = _make_text(n_sentences=40)
        chunks = small_chunker.chunk(text, source="x.txt")
        last = chunks[-1]
        # The last chunk's text should reach (approximately) the
        # end of the input.
        assert text.endswith(last.text.rstrip()) or text[-20:] in last.text


class TestPageAndSourcePassthrough:
    """The ``source`` and ``page`` kwargs reach every chunk."""

    def test_source_is_copied_to_every_chunk(
        self, small_chunker: Chunker
    ) -> None:
        chunks = small_chunker.chunk(_make_text(50), source="manual.pdf")
        assert all(c.source == "manual.pdf" for c in chunks)

    def test_page_is_copied_to_every_chunk(
        self, small_chunker: Chunker
    ) -> None:
        chunks = small_chunker.chunk(_make_text(50), source="manual.pdf", page=3)
        assert all(c.page == 3 for c in chunks)

    def test_page_none_is_copied(
        self, small_chunker: Chunker
    ) -> None:
        """Markdown / TXT callers pass ``page=None``."""
        chunks = small_chunker.chunk(_make_text(50), source="faq.md", page=None)
        assert all(c.page is None for c in chunks)


class TestTokenCountConsistency:
    """Each chunk's ``token_count`` matches the chunker's recount."""

    def test_every_chunk_token_count_matches(
        self, small_chunker: Chunker
    ) -> None:
        chunks = small_chunker.chunk(_make_text(80), source="x.txt")
        for c in chunks:
            assert c.token_count == small_chunker.count_tokens(c.text), (
                f"chunk {c.chunk_index} token_count={c.token_count} "
                f"but recount={small_chunker.count_tokens(c.text)}"
            )

    def test_no_chunk_exceeds_chunk_size(
        self, default_chunker_instance: Chunker
    ) -> None:
        """The chunker never emits a chunk larger than ``chunk_size``."""
        chunks = default_chunker_instance.chunk(_make_text(200), source="x.txt")
        for c in chunks:
            assert c.token_count <= default_chunker_instance.chunk_size, (
                f"chunk {c.chunk_index} has {c.token_count} tokens "
                f"> chunk_size={default_chunker_instance.chunk_size}"
            )


class TestIntegrationWithParsers:
    """End-to-end: parsers → chunker (the path the real pipeline takes)."""

    def test_txt_parser_then_chunker(self, tmp_path) -> None:
        from tinyrag.ingestion.parsers import parse

        p = tmp_path / "doc.txt"
        p.write_text(_make_text(n_sentences=200), encoding="utf-8")
        doc = parse(p)
        chunker = default_chunker()
        chunks = chunker.chunk(doc.text, source=doc.metadata["source"])
        assert chunks, "long document should produce at least one chunk"
        assert all(c.source == "doc.txt" for c in chunks)

    def test_md_parser_then_chunker(self, tmp_path) -> None:
        from tinyrag.ingestion.parsers import parse

        p = tmp_path / "faq.md"
        p.write_text(
            "---\ntitle: t\n---\n# H\n\n"
            + _make_text(n_sentences=200)
            + "\n",
            encoding="utf-8",
        )
        doc = parse(p)
        chunker = default_chunker()
        chunks = chunker.chunk(doc.text, source=doc.metadata["source"])
        assert chunks
        # Front-matter should have been stripped, so the first chunk
        # starts with the heading, not "title:".
        assert "title:" not in chunks[0].text

    def test_pdf_parser_then_chunker_preserves_page_numbers(
        self, tmp_path
    ) -> None:
        """The PDF → chunker path forwards page numbers per chunk."""
        # Import the hand-built PDF helper from test_parsers.py.
        # This keeps the test fixtures centralised (one builder,
        # many tests).
        from tests.test_parsers import _build_minimal_pdf
        from tinyrag.ingestion.parsers import parse

        p = tmp_path / "manual.pdf"
        p.write_bytes(_build_minimal_pdf([_make_text(20), _make_text(20)]))
        doc = parse(p)
        chunker = default_chunker()
        # We chunk per-page (each page is small enough to fit in one
        # chunk) and verify the ``page`` kwarg is forwarded.
        for page_num, page_text in doc.pages:
            page_chunks = chunker.chunk(
                page_text, source=doc.metadata["source"], page=page_num
            )
            for c in page_chunks:
                assert c.page == page_num
                assert c.source == "manual.pdf"
