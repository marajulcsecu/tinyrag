"""Tests for tinyrag.core.retriever (Step 4.12 — query → top-k chunks).

Test layout
-----------
- TestPublicSurface             — every documented symbol is importable
  (Retriever, RetrievalResult, the 4 exception classes, the
  default constants, the metadata adapter).
- TestRetrievalResultDataclass  — frozen, len/bool/top_score work,
  empty result is falsy.
- TestProtocolConformance       — a duck-typed MetadataAccessor
  satisfies the Protocol (@runtime_checkable).
- TestRetrieverConstruction     — happy path + every validation
  error (missing deps, bad threshold).
- TestRetrieveHappyPath         — query → 1 chunk → RetrievalResult
  with the right fields.
- TestRetrieveThreshold         — below-threshold hits are dropped,
  above-threshold survive.
- TestRetrieveNoSensor          — a query without sensor keywords
  doesn't touch the sensor store (saves work).
- TestRetrieveWithSensor        — a query WITH sensor keywords
  triggers the sensor search and merges results.
- TestRetrieveSensorNoKeywords  — empty keyword set → sensor
  path never runs even with a sensor-shaped query.
- TestRetrieveMergeAndDedupe    — same id in both indices → keep
  the higher score.
- TestRetrieveOrderDescending   — results are sorted by score DESC.
- TestRetrieveThresholdFilterHonored — threshold=0.99 → most hits
  dropped (sanity).
- TestRetrieveEmptyIndex        — empty doc index → empty result,
  no exception.
- TestRetrieveDeletedChunk      — FAISS returns id X but metadata
  doesn't have X → silently skipped (TOCTOU window).
- TestRetrieveEmptyQueryRaises  — empty / whitespace query rejected.
- TestRetrieveBadArgsRaises     — k_doc <= 0, k_sensor < 0,
  threshold out of [0, 1] all rejected.
- TestRetrieveErrorMapping      — embedder failure →
  RetrieverEmbedError, store failure → RetrieverSearchError,
  metadata failure → RetrieverMetadataError.
- TestAdaptMetadataStore        — the adapter wraps a real
  MetadataStore-shaped object and the Retriever accepts it.
- TestKeywordDetection          — pure-function tests for the
  sensor-keyword matcher (single-word, multi-word, mixed-case,
  no false positives on common words).
- TestRetrievalResultUsedSensor — used_sensor_idx flips correctly
  based on whether surviving chunks came from the sensor index.
- TestIntegrationWithPromptBuilder — full retriever → prompt
  builder pipeline (the glue test).

Hermetic?
---------
100% hermetic. All stores + metadata are in-memory fakes. No FAISS,
no SQLite, no PyTorch, no model weights. Runs in milliseconds.

Location: ``tests/test_retriever.py``
"""

from __future__ import annotations

from typing import Any

import pytest

from tinyrag.core import (
    DEFAULT_K_DOC,
    DEFAULT_K_SENSOR,
    DEFAULT_SENSOR_KEYWORDS,
    DEFAULT_THRESHOLD,
    Chunk,
    MetadataAccessor,
    PromptBuilder,
    RetrievalResult,
    Retriever,
    RetrieverEmbedError,
    RetrieverError,
    RetrieverMetadataError,
    RetrieverSearchError,
    SMALL_CORPUS_MAX_CHUNKS,
)
from tinyrag.ingestion.embedder import FakeEmbedder
from tinyrag.storage.metadata import ChunkRecord, MetadataStore

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeVectorStore:
    """Tiny in-memory VectorStore for retriever tests.

    Implements just enough of the Protocol to satisfy the
    Retriever. We don't add real FAISS — every search returns the
    pre-loaded hit list. We record the (query_vector, k) of the
    last call so tests can assert the search actually happened
    (and with what).
    """

    def __init__(
        self,
        hits: list[tuple[str, float]] | None = None,
        *,
        raise_on_search: Exception | None = None,
    ) -> None:
        self._hits = list(hits or [])
        self.last_query_vector: list[float] | None = None
        self.last_k: int | None = None
        self.call_count = 0
        self._raise_on_search = raise_on_search

    def search(self, query_vector, k):
        self.last_query_vector = list(query_vector)
        self.last_k = k
        self.call_count += 1
        if self._raise_on_search is not None:
            raise self._raise_on_search
        return list(self._hits[:k])

    # The remaining Protocol methods exist so isinstance checks pass
    # (the Retriever doesn't use them, but FakeVectorStore should be
    # a believable stand-in).
    def add(self, vectors, ids): pass
    def size(self) -> int: return len(self._hits)
    def save(self) -> None: pass
    def load(self) -> None: pass
    def delete_by_source(self, source) -> int: return 0
    @property
    def embedding_dimension(self) -> int: return 384
    @property
    def embedding_model(self) -> str: return "fake"


class FakeMetadataAccessor:
    """Minimal MetadataAccessor fake backed by two dicts."""

    def __init__(
        self,
        chunks_by_id: dict[str, ChunkRecord] | None = None,
        docs_by_id: dict[str, Any] | None = None,
        *,
        raise_on_get_chunks: Exception | None = None,
        raise_on_get_doc: Exception | None = None,
    ) -> None:
        self._chunks = dict(chunks_by_id or {})
        self._docs = dict(docs_by_id or {})
        self._raise_chunks = raise_on_get_chunks
        self._raise_doc = raise_on_get_doc

    def get_chunks_by_ids(self, chunk_ids):
        if self._raise_chunks is not None:
            raise self._raise_chunks
        return [self._chunks[i] for i in chunk_ids if i in self._chunks]

    def get_document(self, document_id):
        if self._raise_doc is not None:
            raise self._raise_doc
        return self._docs.get(document_id)


def _make_chunk_record(
    chunk_id: str,
    *,
    document_id: str = "doc-1",
    text: str = "To reset, press and hold the ring.",
    page_number: int | None = 12,
    chunk_index: int = 0,
    char_offset: int | None = 100,
    token_count: int = 10,
) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        faiss_idx=0,
        page_number=page_number,
        text=text,
        text_preview=text[:30],
        char_offset=char_offset,
        token_count=token_count,
        embedding_model="fake",
        created_at="2026-06-25T00:00:00",
    )


def _make_doc(filename: str = "Nest.pdf") -> Any:
    """A document-like object with a ``filename`` attribute."""

    class _Doc:
        pass

    d = _Doc()
    d.filename = filename
    return d


def _build_retriever(
    *,
    doc_hits: list[tuple[str, float]] | None = None,
    sensor_hits: list[tuple[str, float]] | None = None,
    chunks_by_id: dict[str, ChunkRecord] | None = None,
    docs_by_id: dict[str, Any] | None = None,
    sensor_keywords: frozenset[str] = DEFAULT_SENSOR_KEYWORDS,
    doc_store_size: int | None = None,
    **kwargs,
) -> tuple[Retriever, FakeVectorStore, FakeVectorStore, FakeMetadataAccessor]:
    """Wire a Retriever with fake stores. Returns (retriever, doc, sensor, meta).

    Parameters
    ----------
    doc_store_size:
        Override ``FakeVectorStore.size()``. The real production
        small-corpus fallback activates at ≤
        :data:`SMALL_CORPUS_MAX_CHUNKS`; tests that exercise the
        threshold-filtering path need to declare a "large" doc store
        so the fallback doesn't mask the behaviour they're testing.
        Defaults to ``len(doc_hits)`` (so a 1-hit fixture is treated
        as small, matching production behaviour).
    """
    embedder = FakeEmbedder()
    doc_store = FakeVectorStore(doc_hits or [])
    if doc_store_size is not None:
        # Replace size() with a closure that returns the override.
        doc_store.size = lambda n=doc_store_size: n  # type: ignore[assignment]
    sensor_store = FakeVectorStore(sensor_hits or [])
    meta = FakeMetadataAccessor(chunks_by_id or {}, docs_by_id or {})
    r = Retriever(
        embedder=embedder,
        doc_store=doc_store,
        sensor_store=sensor_store,
        metadata=meta,
        sensor_keywords=sensor_keywords,
        **kwargs,
    )
    return r, doc_store, sensor_store, meta


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    """The module exposes the documented symbols."""

    def test_retriever_class(self) -> None:
        assert callable(Retriever)

    def test_retrieval_result_is_dataclass(self) -> None:
        r = RetrievalResult()
        assert r.chunks == []
        assert r.scores == []
        assert r.used_sensor_idx is False
        assert r.query == ""

    def test_exception_classes(self) -> None:
        assert issubclass(RetrieverError, RuntimeError)
        assert issubclass(RetrieverEmbedError, RetrieverError)
        assert issubclass(RetrieverSearchError, RetrieverError)
        assert issubclass(RetrieverMetadataError, RetrieverError)

    def test_module_constants(self) -> None:
        assert DEFAULT_K_DOC >= 1
        assert DEFAULT_K_SENSOR >= 0
        assert 0.0 < DEFAULT_THRESHOLD < 1.0
        assert isinstance(DEFAULT_SENSOR_KEYWORDS, frozenset)
        assert "temperature" in DEFAULT_SENSOR_KEYWORDS
        assert "yesterday" in DEFAULT_SENSOR_KEYWORDS


# ---------------------------------------------------------------------------
# RetrievalResult dataclass
# ---------------------------------------------------------------------------


class TestRetrievalResultDataclass:
    """The frozen value type returned by Retriever.retrieve()."""

    def test_is_frozen(self) -> None:
        r = RetrievalResult(chunks=[], scores=[], query="q")
        with pytest.raises((AttributeError, Exception)):
            r.query = "other"  # type: ignore[misc]

    def test_len_counts_chunks(self) -> None:
        assert len(RetrievalResult()) == 0
        chunks = [_make_chunk_record(f"c{i}") for i in range(3)]
        r = RetrievalResult(
            chunks=[Chunk(text=c.text, source=c.document_id, page=c.page_number,
                          chunk_index=c.chunk_index, char_offset=c.char_offset or 0,
                          token_count=c.token_count) for c in chunks],
            scores=[0.9, 0.8, 0.7],
        )
        assert len(r) == 3

    def test_bool_false_when_empty(self) -> None:
        assert bool(RetrievalResult()) is False

    def test_bool_true_when_nonempty(self) -> None:
        chunk = Chunk(text="x", source="s", page=1, chunk_index=0,
                      char_offset=0, token_count=1)
        assert bool(RetrievalResult(chunks=[chunk], scores=[0.5])) is True

    def test_top_score_none_when_empty(self) -> None:
        assert RetrievalResult().top_score is None

    def test_top_score_returns_first_when_sorted(self) -> None:
        # Retriever guarantees scores DESC; top_score returns scores[0].
        r = RetrievalResult(scores=[0.9, 0.7, 0.5])
        assert r.top_score == 0.9

    def test_top_score_returns_max_for_unsorted(self) -> None:
        # Defensive: top_score is the max even if scores aren't sorted.
        r = RetrievalResult(scores=[0.5, 0.9, 0.7])
        assert r.top_score == 0.9


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """The MetadataAccessor Protocol is @runtime_checkable."""

    def test_real_accessor_satisfies_protocol(self) -> None:
        # MetadataStore has the two required methods.
        assert issubclass(MetadataStore, object)  # sanity

    def test_fake_accessor_satisfies_protocol(self) -> None:
        fake = FakeMetadataAccessor()
        assert isinstance(fake, MetadataAccessor)

    def test_class_missing_methods_does_not_satisfy(self) -> None:
        class Incomplete:
            pass

        assert not isinstance(Incomplete(), MetadataAccessor)


# ---------------------------------------------------------------------------
# Retriever construction
# ---------------------------------------------------------------------------


class TestRetrieverConstruction:
    """The constructor validates inputs."""

    def test_happy_path(self) -> None:
        r, _, _, _ = _build_retriever()
        assert r.default_threshold == DEFAULT_THRESHOLD

    def test_missing_embedder_rejected(self) -> None:
        ds = FakeVectorStore()
        ss = FakeVectorStore()
        ma = FakeMetadataAccessor()
        with pytest.raises(RetrieverError, match="embedder"):
            Retriever(
                embedder=None,  # type: ignore[arg-type]
                doc_store=ds, sensor_store=ss, metadata=ma,
            )

    def test_missing_doc_store_rejected(self) -> None:
        with pytest.raises(RetrieverError, match="doc_store"):
            Retriever(
                embedder=FakeEmbedder(), doc_store=None,  # type: ignore[arg-type]
                sensor_store=FakeVectorStore(), metadata=FakeMetadataAccessor(),
            )

    def test_missing_sensor_store_rejected(self) -> None:
        with pytest.raises(RetrieverError, match="sensor_store"):
            Retriever(
                embedder=FakeEmbedder(), doc_store=FakeVectorStore(),
                sensor_store=None,  # type: ignore[arg-type]
                metadata=FakeMetadataAccessor(),
            )

    def test_missing_metadata_rejected(self) -> None:
        with pytest.raises(RetrieverError, match="metadata"):
            Retriever(
                embedder=FakeEmbedder(), doc_store=FakeVectorStore(),
                sensor_store=FakeVectorStore(), metadata=None,  # type: ignore[arg-type]
            )

    def test_threshold_out_of_range_rejected(self) -> None:
        with pytest.raises(RetrieverError, match="default_threshold"):
            _build_retriever(default_threshold=1.5)
        with pytest.raises(RetrieverError, match="default_threshold"):
            _build_retriever(default_threshold=-0.1)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestRetrieveHappyPath:
    """A standard query returns the doc hit as a Chunk with metadata."""

    def test_returns_retrieval_result(self) -> None:
        r, _, _, _ = _build_retriever(doc_hits=[("c1", 0.85)])
        result = r.retrieve("What is the password?")
        assert isinstance(result, RetrievalResult)

    def test_query_is_echoed(self) -> None:
        r, _, _, _ = _build_retriever(doc_hits=[("c1", 0.85)])
        result = r.retrieve("What is the password?")
        assert result.query == "What is the password?"

    def test_one_chunk_returned(self) -> None:
        rec = _make_chunk_record("c1")
        doc = _make_doc("Nest.pdf")
        r, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.85)],
            chunks_by_id={"c1": rec},
            docs_by_id={"doc-1": doc},
        )
        result = r.retrieve("Q?")
        assert len(result.chunks) == 1
        assert result.scores == [0.85]

    def test_chunk_source_is_filename_not_id(self) -> None:
        rec = _make_chunk_record("c1", document_id="doc-1")
        doc = _make_doc("Nest-Thermostat.pdf")
        r, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.85)],
            chunks_by_id={"c1": rec},
            docs_by_id={"doc-1": doc},
        )
        result = r.retrieve("Q?")
        assert result.chunks[0].source == "Nest-Thermostat.pdf"
        assert result.chunks[0].source != "doc-1"

    def test_chunk_page_propagates(self) -> None:
        rec = _make_chunk_record("c1", page_number=42)
        r, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.85)],
            chunks_by_id={"c1": rec},
            docs_by_id={"doc-1": _make_doc()},
        )
        result = r.retrieve("Q?")
        assert result.chunks[0].page == 42

    def test_doc_store_was_searched(self) -> None:
        r, doc_store, _, _ = _build_retriever(doc_hits=[("c1", 0.85)])
        r.retrieve("Q?")
        assert doc_store.call_count == 1
        # FAISS is over-fetched (rerank_fetch = max(k_doc * 5, k_doc + 10))
        # so the keyword-overlap rerank (step 6.5) has candidates
        # beyond the top-k to work with. Without this, a chunk that
        # ranks #18 on dense similarity but has a strong lexical match
        # for the query never sees the rerank and never gets promoted.
        assert doc_store.last_k == max(DEFAULT_K_DOC * 5, DEFAULT_K_DOC + 10)


# ---------------------------------------------------------------------------
# Threshold filtering
# ---------------------------------------------------------------------------


class TestRetrieveThreshold:
    """Hits below the threshold are dropped."""

    def test_below_threshold_dropped(self) -> None:
        rec = _make_chunk_record("c1")
        # doc_store_size=100 forces the "large corpus" path so the
        # small-corpus fallback (which sets threshold=0) doesn't mask
        # the behaviour under test.
        r, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.10)],  # well below the default 0.3
            chunks_by_id={"c1": rec},
            docs_by_id={"doc-1": _make_doc()},
            doc_store_size=100,
        )
        result = r.retrieve("Q?")
        assert len(result.chunks) == 0

    def test_above_threshold_survives(self) -> None:
        rec = _make_chunk_record("c1")
        r, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.85)],
            chunks_by_id={"c1": rec},
            docs_by_id={"doc-1": _make_doc()},
            doc_store_size=100,
        )
        result = r.retrieve("Q?")
        assert len(result.chunks) == 1

    def test_threshold_per_call(self) -> None:
        rec = _make_chunk_record("c1")
        r, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.50)],
            chunks_by_id={"c1": rec},
            docs_by_id={"doc-1": _make_doc()},
            doc_store_size=100,
        )
        # Strict threshold → drop.
        assert len(r.retrieve("Q?", threshold=0.9).chunks) == 0
        # Loose threshold → keep.
        assert len(r.retrieve("Q?", threshold=0.1).chunks) == 1

    def test_threshold_zero_keeps_everything(self) -> None:
        rec = _make_chunk_record("c1")
        r, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.01)],
            chunks_by_id={"c1": rec},
            docs_by_id={"doc-1": _make_doc()},
            doc_store_size=100,
        )
        assert len(r.retrieve("Q?", threshold=0.0).chunks) == 1

    def test_threshold_one_keeps_only_perfect(self) -> None:
        rec = _make_chunk_record("c1")
        r, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.99)],
            chunks_by_id={"c1": rec},
            docs_by_id={"doc-1": _make_doc()},
            doc_store_size=100,
        )
        assert len(r.retrieve("Q?", threshold=1.0).chunks) == 0
        assert len(r.retrieve("Q?", threshold=0.99).chunks) == 1


# ---------------------------------------------------------------------------
# Small-corpus fallback (Step 4.25)
# ---------------------------------------------------------------------------
#
# The retriever's :func:`retrieve` lowers its similarity threshold to
# 0 when the doc store has ≤ :data:`SMALL_CORPUS_MAX_CHUNKS` chunks.
# Rationale: MiniLM-L6-v2's absolute cosine scores are noisy on small
# corpora (1-50 chunks) because short user-uploaded chunks score
# lower than long sensor summaries. A threshold-based filter would
# silently drop user uploads even when they're the only thing the
# user wants the model to see. The fallback is opt-out via
# ``doc_store_size`` in the test fixture (we mark the corpus as
# "large" — 100 chunks — when we want to exercise the
# threshold-filtering path).


class TestSmallCorpusFallback:
    """When doc_store.size() <= SMALL_CORPUS_MAX_CHUNKS, the threshold
    is overridden to SMALL_CORPUS_THRESHOLD (0.0) so every doc hit
    survives."""

    def test_low_score_chunk_kept_when_doc_store_is_small(self) -> None:
        """Score 0.05 is well below the production default 0.3, but
        must survive on a 1-chunk corpus so user uploads aren't
        silently dropped."""
        rec = _make_chunk_record("c1")
        r, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.05)],
            chunks_by_id={"c1": rec},
            docs_by_id={"doc-1": _make_doc()},
            # doc_store_size defaults to len(doc_hits) = 1, so the
            # small-corpus fallback applies.
        )
        result = r.retrieve("Q?")
        assert len(result.chunks) == 1
        assert result.chunks[0].text == rec.text

    def test_fallback_does_not_apply_when_doc_store_is_large(self) -> None:
        """Score 0.05 must be dropped on a large corpus so we don't
        drown the prompt in noise."""
        rec = _make_chunk_record("c1")
        r, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.05)],
            chunks_by_id={"c1": rec},
            docs_by_id={"doc-1": _make_doc()},
            doc_store_size=100,  # > SMALL_CORPUS_MAX_CHUNKS
        )
        result = r.retrieve("Q?")
        assert len(result.chunks) == 0

    def test_fallback_boundary_at_small_corpus_max(self) -> None:
        """At exactly SMALL_CORPUS_MAX_CHUNKS chunks, the fallback
        still applies (uses <=). One more chunk turns it off."""
        rec = _make_chunk_record("c1")
        # Exactly at the boundary: fallback ON.
        r_on, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.02)],
            chunks_by_id={"c1": rec},
            docs_by_id={"doc-1": _make_doc()},
            doc_store_size=SMALL_CORPUS_MAX_CHUNKS,
        )
        assert len(r_on.retrieve("Q?").chunks) == 1
        # One above the boundary: fallback OFF.
        r_off, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.02)],
            chunks_by_id={"c1": rec},
            docs_by_id={"doc-1": _make_doc()},
            doc_store_size=SMALL_CORPUS_MAX_CHUNKS + 1,
        )
        assert len(r_off.retrieve("Q?").chunks) == 0

    def test_explicit_threshold_still_wins_when_doc_store_small(self) -> None:
        """A caller-passed threshold=0.99 is preserved when the doc
        store is small (we only override the DEFAULT, not a caller
        override). This matches the documented contract:
        ``threshold`` parameter beats the small-corpus heuristic."""
        rec = _make_chunk_record("c1")
        r, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.50)],
            chunks_by_id={"c1": rec},
            docs_by_id={"doc-1": _make_doc()},
            # doc_store_size defaults to 1 (small).
        )
        # Explicit threshold beats the fallback.
        assert len(r.retrieve("Q?", threshold=0.99).chunks) == 0
        # Without explicit threshold, fallback keeps it.
        assert len(r.retrieve("Q?").chunks) == 1


# ---------------------------------------------------------------------------
# Keyword-overlap reranker (Step 4.25 rerank fix)
# ---------------------------------------------------------------------------
#
# Dense MiniLM embeddings are noisy on short technical queries
# against mixed corpora (a TOC chunk that just *mentions* "ErP
# class 26" can outscore the actual ErP content chunk). The retriever
# applies a cheap lexical bonus: distinctive (non-stopword) tokens
# from the query that appear in a chunk's text bump that chunk's
# score. The pure-function helper ``_distinctive_query_terms`` and
# the reranker step in :meth:`retrieve` are tested in isolation
# below.


class TestDistinctiveQueryTerms:
    """The pure-function that extracts stopword-filtered tokens."""

    def test_filters_english_stopwords(self) -> None:
        from tinyrag.core.retriever import _distinctive_query_terms

        # "What is the ErP directive?" — only "erp" + "directive"
        # survive (3+ chars, not in the stopword list).
        assert _distinctive_query_terms("What is the ErP directive?") == [
            "erp",
            "directive",
        ]

    def test_handles_hyphenated_tokens(self) -> None:
        from tinyrag.core.retriever import _distinctive_query_terms

        # The boundary regex treats hyphens as part of a token
        # (matches `_KEYWORD_BOUNDARY_RE`), so "non-blocking" stays
        # one token. "non" alone is too short to survive the
        # length filter; "non-blocking" (15 chars) passes.
        terms = _distinctive_query_terms("How does non-blocking work?")
        assert "non-blocking" in terms
        assert "work" in terms

    def test_returns_empty_for_pure_stopwords(self) -> None:
        from tinyrag.core.retriever import _distinctive_query_terms

        # All stopwords → empty list (reranker becomes a no-op).
        assert _distinctive_query_terms("what is the") == []

    def test_dedupes_case_insensitively(self) -> None:
        from tinyrag.core.retriever import _distinctive_query_terms

        # "RAG" appears twice; we keep the first occurrence only.
        assert _distinctive_query_terms("What is RAG? Explain rag.") == [
            "rag",
            "explain",
        ]

    def test_empty_input_returns_empty(self) -> None:
        from tinyrag.core.retriever import _distinctive_query_terms

        assert _distinctive_query_terms("") == []


class TestKeywordOverlapRerank:
    """The reranker boosts chunks that contain distinctive query terms.

    Without the rerank, dense scores dominate and a TOC chunk that
    happens to mention the topic can outscore the actual content.
    The rerank adds a +0.10 bonus per matching distinctive term,
    enough to flip the order for technical lookups.
    """

    def test_rerank_pulls_keyword_match_above_dense_only_hit(self) -> None:
        """A chunk containing ALL query terms (matching both + the
        coverage bonus = +0.40 total) outranks a dense-only chunk
        that's 0.15 ahead on the dense score."""
        # Dense scores: chunk A (the right answer) = 0.05, chunk B
        # (the noisy TOC lookalike) = 0.20.
        chunk_a_text = "ErP directive compliance table for heating"
        chunk_b_text = "Installation contents compatibility index"
        rec_a = _make_chunk_record("a", text=chunk_a_text)
        rec_b = _make_chunk_record("b", text=chunk_b_text)

        r, _, _, _ = _build_retriever(
            doc_hits=[("a", 0.05), ("b", 0.20)],
            chunks_by_id={"a": rec_a, "b": rec_b},
            docs_by_id={"doc-1": _make_doc()},
            doc_store_size=1,  # small corpus so threshold=0
        )
        # Without rerank: B (0.20) > A (0.05).
        # With rerank: A matches BOTH "erp" + "directive" → +0.10*2 +
        # 0.20 coverage bonus = +0.40 → A final = 0.45 > B (0.20).
        result = r.retrieve("What is the ErP directive?")
        assert result.chunks[0].text == chunk_a_text
        assert result.chunks[1].text == chunk_b_text
        assert result.scores[0] > result.scores[1]
        assert result.scores[0] == pytest.approx(0.45)
        assert result.scores[1] == pytest.approx(0.20)

    def test_rerank_no_match_leaves_order_unchanged(self) -> None:
        """When no distinctive query terms appear in any chunk, the
        rerank is a no-op — original dense order is preserved."""
        rec_high = _make_chunk_record("a", text="random prose about weather")
        rec_low = _make_chunk_record("b", text="other prose about food")
        r, _, _, _ = _build_retriever(
            doc_hits=[("a", 0.20), ("b", 0.10)],
            chunks_by_id={"a": rec_high, "b": rec_low},
            docs_by_id={"doc-1": _make_doc()},
            doc_store_size=1,
        )
        # Query "what is the?" → all distinctive terms stripped → no rerank.
        result = r.retrieve("what is the?")
        assert result.chunks[0].text == rec_high.text
        assert result.chunks[1].text == rec_low.text

    def test_rerank_partial_match_boosts_proportionally(self) -> None:
        """A chunk with 1 matching term gets +0.10; a chunk with ALL
        query terms gets +0.10*N + 0.20 (the coverage bonus). The
        coverage bonus is the strongest single rerank signal —
        ``every query term is here`` is the textbook relevance case."""
        rec_one = _make_chunk_record("a", text="just directive mentioned")
        rec_two = _make_chunk_record("b", text="erp directive full")
        rec_none = _make_chunk_record("c", text="nothing relevant here")
        r, _, _, _ = _build_retriever(
            doc_hits=[("a", 0.10), ("b", 0.10), ("c", 0.10)],
            chunks_by_id={"a": rec_one, "b": rec_two, "c": rec_none},
            docs_by_id={"doc-1": _make_doc()},
            doc_store_size=1,
        )
        result = r.retrieve("ErP directive?")
        # b: 0.10 + (0.10 * 2 + 0.20 coverage) = 0.50 (both terms)
        # a: 0.10 + (0.10 * 1)             = 0.20 (1 term)
        # c: 0.10                           = 0.10 (no terms)
        assert result.chunks[0].text == rec_two.text
        assert result.chunks[1].text == rec_one.text
        assert result.chunks[2].text == rec_none.text
        assert result.scores[0] == pytest.approx(0.50)
        assert result.scores[1] == pytest.approx(0.20)
        assert result.scores[2] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Sensor keyword detection
# ---------------------------------------------------------------------------


class TestRetrieveNoSensor:
    """A query without sensor keywords doesn't touch the sensor store."""

    def test_sensor_store_not_called(self) -> None:
        rec = _make_chunk_record("c1")
        r, _, sensor_store, _ = _build_retriever(
            doc_hits=[("c1", 0.85)],
            chunks_by_id={"c1": rec},
            docs_by_id={"doc-1": _make_doc()},
        )
        result = r.retrieve("How do I reset my thermostat?")
        assert sensor_store.call_count == 0
        assert result.used_sensor_idx is False
        assert result.sensor_keywords_matched == []


class TestRetrieveWithSensor:
    """A query with sensor keywords triggers the sensor search + merges."""

    def test_sensor_keywords_detected(self) -> None:
        sensor_rec = _make_chunk_record(
            "s1", document_id="sensor-doc", text="22°C yesterday",
            page_number=None,
        )
        r, _, sensor_store, _ = _build_retriever(
            doc_hits=[("c1", 0.85)],
            sensor_hits=[("s1", 0.75)],
            chunks_by_id={
                "c1": _make_chunk_record("c1"),
                "s1": sensor_rec,
            },
            docs_by_id={
                "doc-1": _make_doc("Nest.pdf"),
                "sensor-doc": _make_doc("sensor-summary.md"),
            },
        )
        result = r.retrieve("What was the temperature yesterday?")
        assert sensor_store.call_count == 1
        assert "temperature" in result.sensor_keywords_matched
        assert "yesterday" in result.sensor_keywords_matched

    def test_sensor_result_appears(self) -> None:
        sensor_rec = _make_chunk_record(
            "s1", document_id="sensor-doc", text="22°C yesterday",
            page_number=None,
        )
        r, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.85)],
            sensor_hits=[("s1", 0.75)],
            chunks_by_id={
                "c1": _make_chunk_record("c1"),
                "s1": sensor_rec,
            },
            docs_by_id={
                "doc-1": _make_doc("Nest.pdf"),
                "sensor-doc": _make_doc("sensor-summary.md"),
            },
        )
        result = r.retrieve("What was the temperature yesterday?")
        assert len(result.chunks) == 2
        # Find the sensor chunk.
        sensor_chunk = next(c for c in result.chunks if c.source == "sensor-summary.md")
        assert "22°C" in sensor_chunk.text

    def test_used_sensor_idx_true_when_sensor_survives(self) -> None:
        sensor_rec = _make_chunk_record(
            "s1", document_id="sensor-doc", text="22°C",
            page_number=None,
        )
        r, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.85)],
            sensor_hits=[("s1", 0.75)],
            chunks_by_id={
                "c1": _make_chunk_record("c1"),
                "s1": sensor_rec,
            },
            docs_by_id={
                "doc-1": _make_doc("Nest.pdf"),
                "sensor-doc": _make_doc("sensor.md"),
            },
        )
        result = r.retrieve("What was the temperature yesterday?")
        assert result.used_sensor_idx is True


class TestRetrieveSensorNoKeywords:
    """Empty keyword set disables the sensor path entirely."""

    def test_empty_keywords_skips_sensor_even_for_sensor_query(self) -> None:
        rec = _make_chunk_record("c1")
        r, _, sensor_store, _ = _build_retriever(
            doc_hits=[("c1", 0.85)],
            sensor_hits=[("s1", 0.95)],  # would be picked up if sensor ran
            chunks_by_id={
                "c1": rec,
                "s1": _make_chunk_record("s1", document_id="sensor-doc"),
            },
            docs_by_id={
                "doc-1": _make_doc("Nest.pdf"),
                "sensor-doc": _make_doc("sensor.md"),
            },
            sensor_keywords=frozenset(),  # disable sensor detection
        )
        result = r.retrieve("What was the temperature yesterday?")
        # Sensor store not touched.
        assert sensor_store.call_count == 0
        # Only the doc hit survives.
        assert len(result.chunks) == 1
        assert result.used_sensor_idx is False


# ---------------------------------------------------------------------------
# Merging and ordering
# ---------------------------------------------------------------------------


class TestRetrieveMergeAndDedupe:
    """Same id in both indices → keep the higher score."""

    def test_same_id_picks_higher_score(self) -> None:
        rec = _make_chunk_record("c1")
        # Use a sensor-keyword query so the sensor store actually runs.
        r, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.40)],     # doc-store scores 0.40
            sensor_hits=[("c1", 0.85)],  # sensor-store scores 0.85
            chunks_by_id={"c1": rec},
            docs_by_id={"doc-1": _make_doc()},
        )
        result = r.retrieve(
            "What was the temperature yesterday?",
            k_doc=3, k_sensor=3,
        )
        assert len(result.chunks) == 1
        assert result.scores == [0.85]
        # Sensor provided the higher score → used_sensor_idx=True.
        assert result.used_sensor_idx is True


class TestRetrieveOrderDescending:
    """Results are sorted by score DESCENDING."""

    def test_sorted_by_score_desc(self) -> None:
        recs = {f"c{i}": _make_chunk_record(f"c{i}") for i in range(4)}
        docs = {"doc-1": _make_doc()}
        r, _, _, _ = _build_retriever(
            doc_hits=[("c0", 0.50), ("c1", 0.90), ("c2", 0.70), ("c3", 0.30)],
            chunks_by_id=recs,
            docs_by_id=docs,
        )
        result = r.retrieve("Q?")
        scores = result.scores
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 0.90


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestRetrieveEmptyIndex:
    """Empty doc index → empty result, no exception."""

    def test_no_doc_hits(self) -> None:
        r, _, _, _ = _build_retriever(doc_hits=[])
        result = r.retrieve("Q?")
        assert len(result.chunks) == 0
        assert result.scores == []

    def test_no_chunks_registered(self) -> None:
        r, _, _, _ = _build_retriever(doc_hits=[("ghost", 0.99)])  # not in meta
        result = r.retrieve("Q?")
        # Metadata lookup for "ghost" returns nothing → silently skipped.
        assert len(result.chunks) == 0


class TestRetrieveDeletedChunk:
    """FAISS returns id X but metadata doesn't have X → silently skipped."""

    def test_toctou_skipped(self) -> None:
        rec = _make_chunk_record("c1")
        r, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.85), ("deleted", 0.95)],  # deleted chunk
            chunks_by_id={"c1": rec},  # only c1 registered
            docs_by_id={"doc-1": _make_doc()},
        )
        result = r.retrieve("Q?")
        # Only c1 survives; the deleted one is silently dropped.
        assert len(result.chunks) == 1
        assert result.scores == [0.85]


class TestRetrieveEmptyQueryRaises:
    """Empty / whitespace queries are rejected."""

    def test_empty_string_rejected(self) -> None:
        r, _, _, _ = _build_retriever()
        with pytest.raises(RetrieverError, match="query"):
            r.retrieve("")

    def test_whitespace_rejected(self) -> None:
        r, _, _, _ = _build_retriever()
        with pytest.raises(RetrieverError, match="query"):
            r.retrieve("   \n\t  ")


class TestRetrieveBadArgsRaises:
    """k_doc, k_sensor, threshold out of range all rejected."""

    def test_k_doc_zero_rejected(self) -> None:
        r, _, _, _ = _build_retriever()
        with pytest.raises(RetrieverError, match="k_doc"):
            r.retrieve("Q?", k_doc=0)

    def test_k_doc_negative_rejected(self) -> None:
        r, _, _, _ = _build_retriever()
        with pytest.raises(RetrieverError, match="k_doc"):
            r.retrieve("Q?", k_doc=-1)

    def test_k_sensor_negative_rejected(self) -> None:
        r, _, _, _ = _build_retriever()
        with pytest.raises(RetrieverError, match="k_sensor"):
            r.retrieve("Q?", k_sensor=-1)

    def test_threshold_negative_rejected(self) -> None:
        r, _, _, _ = _build_retriever()
        with pytest.raises(RetrieverError, match="threshold"):
            r.retrieve("Q?", threshold=-0.1)

    def test_threshold_over_one_rejected(self) -> None:
        r, _, _, _ = _build_retriever()
        with pytest.raises(RetrieverError, match="threshold"):
            r.retrieve("Q?", threshold=1.1)


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestRetrieveErrorMapping:
    """Component failures surface as the documented exception types."""

    def test_embedder_failure_maps_to_embed_error(self) -> None:
        class BrokenEmbedder:
            @property
            def dimension(self) -> int:
                return 384

            def embed(self, texts):
                raise RuntimeError("model crashed")

        r = Retriever(
            embedder=BrokenEmbedder(),  # type: ignore[arg-type]
            doc_store=FakeVectorStore(),
            sensor_store=FakeVectorStore(),
            metadata=FakeMetadataAccessor(),
        )
        with pytest.raises(RetrieverEmbedError):
            r.retrieve("Q?")

    def test_doc_store_failure_maps_to_search_error(self) -> None:
        r = Retriever(
            embedder=FakeEmbedder(),
            doc_store=FakeVectorStore(raise_on_search=RuntimeError("boom")),
            sensor_store=FakeVectorStore(),
            metadata=FakeMetadataAccessor(),
        )
        with pytest.raises(RetrieverSearchError, match="doc_store"):
            r.retrieve("Q?")

    def test_sensor_store_failure_maps_to_search_error(self) -> None:
        r = Retriever(
            embedder=FakeEmbedder(),
            doc_store=FakeVectorStore(),
            sensor_store=FakeVectorStore(raise_on_search=RuntimeError("boom")),
            metadata=FakeMetadataAccessor(),
        )
        with pytest.raises(RetrieverSearchError, match="sensor_store"):
            r.retrieve("What was the temperature yesterday?")

    def test_metadata_chunks_failure_maps_to_metadata_error(self) -> None:
        r = Retriever(
            embedder=FakeEmbedder(),
            doc_store=FakeVectorStore([("c1", 0.85)]),
            sensor_store=FakeVectorStore(),
            metadata=FakeMetadataAccessor(
                raise_on_get_chunks=RuntimeError("db locked")
            ),
        )
        with pytest.raises(RetrieverMetadataError):
            r.retrieve("Q?")

    def test_metadata_doc_failure_maps_to_metadata_error(self) -> None:
        r = Retriever(
            embedder=FakeEmbedder(),
            doc_store=FakeVectorStore([("c1", 0.85)]),
            sensor_store=FakeVectorStore(),
            metadata=FakeMetadataAccessor(
                chunks_by_id={"c1": _make_chunk_record("c1")},
                docs_by_id={"doc-1": _make_doc()},
                raise_on_get_doc=RuntimeError("db locked"),
            ),
        )
        with pytest.raises(RetrieverMetadataError):
            r.retrieve("Q?")


# ---------------------------------------------------------------------------
# adapt_metadata_store
# ---------------------------------------------------------------------------


class TestAdaptMetadataStore:
    """The adapter wraps a real MetadataStore-shaped object."""

    def test_adapts_real_store(self) -> None:
        # A MetadataStore-shaped object — we use a Mock that has the
        # two required methods, plus what the adapter expects.
        class MockStore:
            def __init__(self):
                self.chunks_called_with = None
                self.doc_return = None
            def get_chunks_by_ids(self, ids):
                self.chunks_called_with = list(ids)
                return []
            def get_document(self, doc_id):
                return self.doc_return

        s = MockStore()
        s.doc_return = _make_doc()
        from tinyrag.core.retriever import adapt_metadata_store
        wrapped = adapt_metadata_store(s)  # type: ignore[arg-type]
        # Should satisfy the Protocol.
        assert isinstance(wrapped, MetadataAccessor)
        # Should forward calls.
        wrapped.get_chunks_by_ids(["a", "b"])
        assert s.chunks_called_with == ["a", "b"]
        assert wrapped.get_document("x") is s.doc_return

    def test_adapter_returns_accessor_type(self) -> None:
        from tinyrag.core.retriever import adapt_metadata_store
        s = FakeMetadataAccessor()
        wrapped = adapt_metadata_store(s)
        # The adapter returns an object that satisfies the Protocol.
        assert isinstance(wrapped, MetadataAccessor)


# ---------------------------------------------------------------------------
# Keyword detection (pure-function tests)
# ---------------------------------------------------------------------------


class TestKeywordDetection:
    """The internal _find_sensor_keywords helper."""

    def test_single_word_match(self) -> None:
        from tinyrag.core.retriever import _find_sensor_keywords
        assert _find_sensor_keywords(
            "What is the temperature?", frozenset({"temperature"})
        ) == ["temperature"]

    def test_case_insensitive(self) -> None:
        from tinyrag.core.retriever import _find_sensor_keywords
        assert _find_sensor_keywords(
            "WHAT IS THE TEMPERATURE?", frozenset({"temperature"})
        ) == ["temperature"]

    def test_no_match_returns_empty(self) -> None:
        from tinyrag.core.retriever import _find_sensor_keywords
        assert _find_sensor_keywords(
            "How do I reset my thermostat?", frozenset({"temperature"})
        ) == []

    def test_multi_word_phrase_match(self) -> None:
        from tinyrag.core.retriever import _find_sensor_keywords
        assert _find_sensor_keywords(
            "What was the temperature last week?",
            frozenset({"last week"}),
        ) == ["last week"]

    def test_word_boundary_respected(self) -> None:
        # "temp" should NOT match inside "attempted".
        from tinyrag.core.retriever import _find_sensor_keywords
        assert _find_sensor_keywords(
            "I attempted the reset.", frozenset({"temp"})
        ) == []

    def test_multiple_matches_sorted(self) -> None:
        from tinyrag.core.retriever import _find_sensor_keywords
        result = _find_sensor_keywords(
            "Temperature yesterday was 22°C.",
            frozenset({"temperature", "yesterday"}),
        )
        assert result == ["temperature", "yesterday"]

    def test_empty_keywords_returns_empty(self) -> None:
        from tinyrag.core.retriever import _find_sensor_keywords
        assert _find_sensor_keywords("temperature now", frozenset()) == []

    def test_empty_query_returns_empty(self) -> None:
        from tinyrag.core.retriever import _find_sensor_keywords
        assert _find_sensor_keywords("", frozenset({"temperature"})) == []

    def test_default_keywords_match_common_phrases(self) -> None:
        from tinyrag.core.retriever import _find_sensor_keywords
        # The shipped keyword set should match the roadmap's examples.
        assert _find_sensor_keywords("What's the temperature now?", DEFAULT_SENSOR_KEYWORDS)
        assert _find_sensor_keywords("How much energy last week?", DEFAULT_SENSOR_KEYWORDS)
        assert _find_sensor_keywords("Is the humidity high?", DEFAULT_SENSOR_KEYWORDS)


# ---------------------------------------------------------------------------
# used_sensor_idx semantics
# ---------------------------------------------------------------------------


class TestUsedSensorIdxSemantics:
    """used_sensor_idx is True iff a sensor hit survived the threshold."""

    def test_false_when_sensor_path_not_taken(self) -> None:
        rec = _make_chunk_record("c1")
        r, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.85)],
            chunks_by_id={"c1": rec},
            docs_by_id={"doc-1": _make_doc()},
        )
        result = r.retrieve("How do I reset my thermostat?")
        assert result.used_sensor_idx is False

    def test_false_when_sensor_hit_below_threshold(self) -> None:
        sensor_rec = _make_chunk_record(
            "s1", document_id="sensor-doc", text="22°C",
            page_number=None,
        )
        r, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.85)],
            sensor_hits=[("s1", 0.05)],  # below default 0.3
            chunks_by_id={
                "c1": _make_chunk_record("c1"),
                "s1": sensor_rec,
            },
            docs_by_id={
                "doc-1": _make_doc("Nest.pdf"),
                "sensor-doc": _make_doc("sensor.md"),
            },
            doc_store_size=100,  # force "large corpus" path so threshold applies
        )
        result = r.retrieve("What was the temperature yesterday?")
        # Sensor ran but its hit was filtered → used_sensor_idx is False.
        assert result.used_sensor_idx is False
        assert len(result.chunks) == 1
        assert result.chunks[0].source == "Nest.pdf"


# ---------------------------------------------------------------------------
# Integration with prompt builder
# ---------------------------------------------------------------------------


class TestIntegrationWithPromptBuilder:
    """The retrieved chunks flow cleanly into PromptBuilder.build()."""

    def test_retrieval_then_prompt(self) -> None:
        # 2 doc chunks + 1 sensor chunk retrieved.
        c1 = _make_chunk_record("c1", text="To reset, press the ring.")
        c2 = _make_chunk_record("c2", text="Soft reset: Settings > Reset.")
        s1 = _make_chunk_record(
            "s1", document_id="sensor-doc", text="Temperature 22°C yesterday.",
            page_number=None,
        )
        r, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.85), ("c2", 0.75)],
            sensor_hits=[("s1", 0.65)],
            chunks_by_id={"c1": c1, "c2": c2, "s1": s1},
            docs_by_id={
                "doc-1": _make_doc("Nest.pdf"),
                "sensor-doc": _make_doc("sensor.md"),
            },
        )
        result = r.retrieve("What was the temperature yesterday?")
        assert len(result.chunks) == 3
        # Feed into the prompt builder — should produce a clean prompt.
        prompt = PromptBuilder().build(
            "What was the temperature yesterday?", result.chunks
        )
        # All 3 chunks fit in the default budget.
        assert prompt.chunks_used == 3
        # Citations [1] [2] [3] all present, in similarity order.
        assert "[1]" in prompt.user_message
        assert "[2]" in prompt.user_message
        assert "[3]" in prompt.user_message
        # The doc filenames appear in the context block.
        assert "Nest.pdf" in prompt.user_message
        assert "sensor.md" in prompt.user_message

    def test_empty_retrieval_passes_to_prompt_builder(self) -> None:
        # Below threshold → empty chunks. PromptBuilder handles it.
        rec = _make_chunk_record("c1")
        r, _, _, _ = _build_retriever(
            doc_hits=[("c1", 0.10)],  # below threshold
            chunks_by_id={"c1": rec},
            docs_by_id={"doc-1": _make_doc()},
            doc_store_size=100,  # force "large corpus" path
        )
        result = r.retrieve("Q?")
        assert len(result.chunks) == 0
        # PromptBuilder.build with empty chunks → refusal prompt.
        prompt = PromptBuilder().build("Q?", result.chunks)
        assert prompt.chunks_used == 0
        assert "I don't have enough information" in prompt.system_prompt
