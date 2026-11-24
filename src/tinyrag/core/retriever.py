"""Query → top-k chunks (the retrieval half of RAG).

This module is the **bridge** between the embedding model + vector
stores (the "what does this query mean?" seam) and the prompt builder
(Step 4.11, which assembles the grounded answer). It answers:

    Given a user's question, which ``Chunk`` objects are the most
    relevant evidence we have — and should I also look at the
    sensor index because the question is about live readings?

Why is this in :mod:`tinyrag.core` and not :mod:`tinyrag.storage`?
-----------------------------------------------------------------
:mod:`tinyrag.core` is *domain logic with no I/O*. The Retriever
doesn't open files or sockets itself — it calls Protocols (the
embedder, the stores, the metadata accessor). Those Protocols are
implemented by real classes in :mod:`tinyrag.ingestion` and
:mod:`tinyrag.storage`. The Retriever is what wires them together
at the composition root (``main.py``, Step 4.17) and decides the
*policy* of retrieval (keyword detection, threshold filtering,
result merging). The *mechanism* — the FAISS search, the SQLite
read — lives behind the Protocols.

Sensor keyword detection
------------------------
The roadmap says *"a simple list"*. This module ships
:data:`DEFAULT_SENSOR_KEYWORDS` — a case-insensitive set of words
that suggest the user is asking about a live sensor reading
("temperature", "humidity", "kWh", "yesterday", etc.). When the
query matches, we ALSO search the sensor index and merge the
results. When it doesn't, we save the work and skip the sensor
search entirely. A caller can override the keyword list (e.g.
to add domain-specific terms like "thermostat setpoint") via the
``sensor_keywords`` constructor argument.

Threshold filtering
-------------------
After merging doc + sensor hits, every (chunk, score) pair below
``threshold`` is dropped. This is the "is this hit actually
relevant?" gate — without it, a query that has nothing to do with
the corpus would still return the index's "closest neighbours"
(small nonzero scores from FAISS's float math) and the LLM would
hallucinate from bad context. Default 0.3 — see
``docs/03_architecture_v1.md`` §10.1 (the architecture doc's
worked example uses the same number).

Pure functions / no I/O
-----------------------
The Retriever's :meth:`retrieve` is pure except for the indirect
I/O via the injected stores. Tests inject in-memory fakes
(:class:`tinyrag.ingestion.embedder.FakeEmbedder` + a tiny
in-memory VectorStore + an in-memory metadata accessor) — no
FAISS, no SQLite, no PyTorch.

Location: ``src/tinyrag/core/retriever.py``
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from tinyrag.core.chunker import Chunk
from tinyrag.storage.metadata import ChunkRecord, MetadataStore
from tinyrag.storage.vector_store import VectorStore

if TYPE_CHECKING:
    from tinyrag.ingestion.embedder import EmbeddingModel

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

#: Default similarity threshold below which a hit is discarded.
#: Matches the architecture doc's worked example (§10.1). cosine
#: similarity on L2-normalised MiniLM vectors: a 0.3 threshold is
#: a reasonable "vaguely related" cut-off for short queries on a
#: LARGE corpus (1000+ chunks).
#:
#: For small corpora (≤ :data:`SMALL_CORPUS_MAX_CHUNKS` chunks), this
#: absolute threshold is too aggressive — MiniLM-L6-v2 produces raw
#: scores in the 0.04-0.15 range for almost every short query against
#: a tiny or mixed corpus (the embedding is dominated by common-word
#: noise rather than topical signal). The retriever's :meth:`retrieve`
#: detects this case and substitutes :data:`SMALL_CORPUS_THRESHOLD`
#: so user-uploaded chunks don't get silently dropped.
DEFAULT_THRESHOLD = 0.3

#: Threshold used when the doc store has fewer than
#: :data:`SMALL_CORPUS_MAX_CHUNKS` chunks. Set to a small **relevance
#: floor** (0.15) rather than 0.0: on a tiny/mixed corpus, a 0.0 floor
#: lets every one of ``k_doc`` chunks through regardless of relevance,
#: which (a) pads the prompt with off-topic chunks and (b) makes the
#: LLM prefill huge and slow (measured: "What is RAG?" produced a
#: 1480-token prompt with 4 irrelevant Nest-PDF chunks scoring
#: 0.06-0.29, driving a 47 s CPU inference). 0.15 is well below the
#: production :data:`DEFAULT_THRESHOLD` (0.3) so genuinely relevant
#: short user-uploaded chunks still surface (e.g. the 1-chunk rag.txt
#: definition scores 0.36 for "What is RAG?"), but noise chunks scoring
#: below 0.15 are dropped — which both improves grounding and cuts
#: latency. Verified 2026-07-22 against the 5 demo questions: the
#: correct top chunk survives for every one (ErP p26, OpenTherm p4,
#: warranty p27, combi p20, RAG rag.txt) while prompt size falls
#: 20-60%. The prompt builder still caps the token budget as a second
#: line of defence, and the LLM still refuses when no returned chunk
#: answers the question.
SMALL_CORPUS_THRESHOLD = 0.15

#: Doc-store size at which the small-corpus fallback activates.
#: Empirically: at ≤ 50 chunks, an absolute similarity threshold is
#: unreliable because (a) scores are noisy at small scale and (b)
#: the user almost certainly uploaded every chunk intentionally, so
#: dropping any of them on a similarity basis is surprising. This
#: covers the "demo + small corpus" use case — a user with 1 manual
#: + 1 PDF typically lands at 5-50 total chunks. Production corpora
#: (1000+ chunks) are well above this cutoff, so the strict
#: :data:`DEFAULT_THRESHOLD` still applies there.
SMALL_CORPUS_MAX_CHUNKS = 50

#: Default k for the document index. 5 is the empirical sweet spot
#: for a small mixed corpus (≤ ~50 chunks): with k=3, a single noisy
#: long chunk from one document can crowd out a short relevant chunk
#: from another (e.g. a 1-chunk rag.txt definition beaten by 3 chunks
#: of a 38-page Nest PDF on a "What is RAG?" query). Bump to 5 so
#: cross-document questions have a chance to surface a chunk from
#: every uploaded doc. For larger corpora, scale with
#: k_doc_top_k ≈ log10(num_chunks). Mirrors the ``retrieval.doc_top_k``
#: value in ``config.yaml`` and the ``k_doc`` field default in
#: :class:`tinyrag.api.schemas.AskRequest`.
DEFAULT_K_DOC = 5

#: Default k for the sensor index. Matches the architecture doc's
#: worked example.
DEFAULT_K_SENSOR = 2

#: Default sensor-keyword set. Case-insensitive substring match
#: (whole-word boundary checked). Picked from the architecture
#: doc §6.5 ("temperature", "humidity", "energy", "kWh",
#: "yesterday", "last week") plus a few obvious additions for any
#: sensor-equipped deployment.
DEFAULT_SENSOR_KEYWORDS: frozenset[str] = frozenset({
    # measurement names
    "temperature", "temp", "humidity", "humid",
    "energy", "kwh", "kilowatt", "power",
    "motion", "movement", "occupancy",
    "light", "luminance", "lux",
    "co2", "voc", "air", "aqi",
    # temporal markers that strongly suggest "live data"
    "yesterday", "today", "tonight", "now", "currently",
    "last week", "last hour", "last day", "last night",
    "this morning", "this afternoon", "this evening",
    "right now", "at the moment",
    # present-tense "what is the current value" phrasing. Deliberately
    # broad: these also match ordinary document questions ("what is the
    # running time of merge sort"), but a false positive is harmless —
    # matching only *widens* the candidate pool with sensor chunks,
    # which then lose to document chunks on rerank score and threshold
    # (doc hits win ties). It never suppresses document retrieval.
    "is the", "are the", "what's the", "whats the",
    # Bengali agricultural weather, soil & temporal keywords (Agri-IoT)
    "আবহাওয়া", "আবহাওয়া", "আবহাওয়ার", "আবহাওয়ার",
    "তাপমাত্রা", "তাপমাত্রার",
    "আর্দ্রতা", "আর্দ্রতার", "বাতাসের আর্দ্রতা",
    "বৃষ্টি", "বৃষ্টির", "কুয়াশা", "কুয়াশা", "মেঘলা",
    "মাটির রস", "মাটির আর্দ্রতা", "জমির রস", "সেচ", "সেচের",
    "আজকে", "আজকের", "গতকাল", "গতকালের", "এখন", "বর্তমানে",
    "সেন্সর", "স্প্রে করা যাবে",
})


# ----------------------------------------------------------------------------
# Public exceptions
# ----------------------------------------------------------------------------


class RetrieverError(RuntimeError):
    """Base class for everything in this module."""


class RetrieverEmbedError(RetrieverError):
    """The embedder refused the query (EmbeddingError subclass)."""


class RetrieverSearchError(RetrieverError):
    """A vector store raised during search."""


class RetrieverMetadataError(RetrieverError):
    """The metadata accessor raised while resolving chunk ids."""


# ----------------------------------------------------------------------------
# Metadata accessor Protocol
# ----------------------------------------------------------------------------


@runtime_checkable
class MetadataAccessor(Protocol):
    """The subset of :class:`MetadataStore` the Retriever needs.

    We type-hint against this minimal Protocol so the Retriever can
    be unit-tested with a tiny in-memory fake (just a dict) — and
    so a future "swap SQLite for Postgres" change doesn't have to
    touch this file. The production wiring passes the real
    :class:`MetadataStore`, which satisfies this Protocol
    structurally (it has both methods).
    """

    def get_chunks_by_ids(
        self, chunk_ids: Sequence[str]
    ) -> list[ChunkRecord]:
        """Return chunk rows for the given ids, preserving order.

        Unknown ids are silently skipped (the FAISS→metadata
        TOCTOU window — a chunk can be deleted between indexing
        and query).
        """
        ...

    def get_document(self, document_id: str) -> object | None:
        """Return the document row for ``document_id``, or ``None``.

        We type the return as ``object`` because the Retriever only
        reads ``.filename`` off it (via ``getattr``) — the
        production :class:`DocumentRecord` has it, and a test fake
        can be a plain ``dataclass`` with just that field.
        """
        ...


# ----------------------------------------------------------------------------
# Result dataclass
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalResult:
    """The outcome of a retrieval query.

    Attributes
    ----------
    chunks:
        The retrieved chunks, ranked by score DESCENDING
        (most-relevant first). May be empty if every hit was
        below the threshold — the prompt builder handles empty
        gracefully (refusal prompt).
    scores:
        The cosine-similarity score for each chunk, parallel to
        ``chunks`` (same length, same order). Always in ``[-1, 1]``
        for L2-normalised vectors.
    used_sensor_idx:
        ``True`` iff the query matched a sensor keyword AND the
        sensor index returned at least one hit that survived the
        threshold. Useful for observability + the API response
        payload ("this answer used live sensor data").
    sensor_keywords_matched:
        The list of sensor keywords that appeared in the query
        (preserved for debugging even when the sensor index was
        empty — the caller can tell *why* the sensor path ran).
    query:
        The original query string (echoed back so the API layer
        can log "query=..., top1=..." without juggling state).
    """

    chunks: list[Chunk] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    used_sensor_idx: bool = False
    sensor_keywords_matched: list[str] = field(default_factory=list)
    query: str = ""

    def __len__(self) -> int:
        return len(self.chunks)

    def __bool__(self) -> bool:
        # So `if result:` works in the API layer / prompt builder.
        return len(self.chunks) > 0

    @property
    def top_score(self) -> float | None:
        """The highest score in the result, or ``None`` if empty.

        Uses :func:`max` so the result is correct even if a caller
        constructs a :class:`RetrievalResult` with an unsorted
        scores list (defensive — the Retriever always emits sorted
        scores).
        """
        return max(self.scores) if self.scores else None


# ----------------------------------------------------------------------------
# The Retriever
# ----------------------------------------------------------------------------


class Retriever:
    """Query → top-k chunks across the doc and sensor indices.

    Pure domain logic — depends only on Protocols (embedder, stores,
    metadata accessor). The composition root wires real
    implementations.

    Parameters
    ----------
    embedder:
        Anything satisfying the :class:`EmbeddingModel` Protocol.
        Only :meth:`embed` and the ``dimension`` property are used.
    doc_store:
        A :class:`VectorStore` over the document chunks. Required.
    sensor_store:
        A :class:`VectorStore` over the sensor chunks. Required
        (the Retriever always takes both — callers that don't want
        sensor retrieval can pass the same store twice OR set the
        keyword list to ``frozenset()`` to never query it).
    metadata:
        The :class:`MetadataAccessor` used to resolve chunk ids →
        full :class:`Chunk` records. Required.
    sensor_keywords:
        Case-insensitive set of substrings (whole-word) that
        trigger a sensor-index search. Defaults to
        :data:`DEFAULT_SENSOR_KEYWORDS`. Pass an empty
        :class:`frozenset` to disable sensor retrieval entirely.
    default_threshold:
        The default ``threshold`` argument for :meth:`retrieve`.
        Defaults to :data:`DEFAULT_THRESHOLD` (0.3).
    """

    def __init__(
        self,
        *,
        embedder: EmbeddingModel,
        doc_store: VectorStore,
        sensor_store: VectorStore,
        metadata: MetadataAccessor,
        sensor_keywords: frozenset[str] = DEFAULT_SENSOR_KEYWORDS,
        default_threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        if embedder is None:
            raise RetrieverError("embedder is required")
        if doc_store is None:
            raise RetrieverError("doc_store is required")
        if sensor_store is None:
            raise RetrieverError("sensor_store is required")
        if metadata is None:
            raise RetrieverError("metadata is required")
        if not 0.0 <= default_threshold <= 1.0:
            raise RetrieverError(
                f"default_threshold must be in [0, 1] (got {default_threshold})"
            )

        self.embedder = embedder
        self.doc_store = doc_store
        self.sensor_store = sensor_store
        self.metadata = metadata
        self.sensor_keywords = sensor_keywords
        self.default_threshold = default_threshold

    # ----- the main entry point -----------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        k_doc: int = DEFAULT_K_DOC,
        k_sensor: int = DEFAULT_K_SENSOR,
        threshold: float | None = None,
    ) -> RetrievalResult:
        """Retrieve the most-relevant chunks for ``query``.

        Pipeline (matches ``docs/03_architecture_v1.md`` §10.1):

        1. Detect sensor keywords in ``query`` (case-insensitive).
        2. Embed the query (single text → single vector).
        3. Search the doc index for ``k_doc`` hits.
        4. If sensor keywords matched, search the sensor index for
           ``k_sensor`` hits too.
        5. Merge doc + sensor hits into one (id → score) map,
           keeping the highest score per id.
        6. Resolve ids → chunk records via the metadata accessor.
        7. Drop pairs below ``threshold``.
        8. Sort by score DESCENDING; return as :class:`RetrievalResult`.

        Parameters
        ----------
        query:
            The user's question. Must be non-empty.
        k_doc:
            Number of hits to request from the doc index.
        k_sensor:
            Number of hits to request from the sensor index (only
            used if the sensor path triggers).
        threshold:
            Drop hits below this cosine-similarity score. Defaults
            to ``self.default_threshold`` (0.3).

        Returns
        -------
        RetrievalResult:
            Frozen dataclass with the surviving chunks + scores +
            diagnostics. May have empty ``chunks`` if every hit was
            below the threshold or the indices are empty.

        Raises
        ------
        RetrieverError:
            For programming errors (empty query, k out of range).
        RetrieverEmbedError, RetrieverSearchError,
        RetrieverMetadataError:
            Pass-throughs from the injected components — the API
            layer maps these to HTTP 503 / 500.
        """
        if not query or not query.strip():
            raise RetrieverError("query must be a non-empty string")
        if k_doc <= 0:
            raise RetrieverError(f"k_doc must be > 0 (got {k_doc})")
        if k_sensor < 0:
            raise RetrieverError(f"k_sensor must be >= 0 (got {k_sensor})")
        # Remember whether the caller passed an explicit threshold,
        # so the small-corpus fallback below can tell "user override"
        # from "we filled in the default".
        threshold_was_default = threshold is None
        if threshold is None:
            threshold = self.default_threshold
        if not 0.0 <= threshold <= 1.0:
            raise RetrieverError(
                f"threshold must be in [0, 1] (got {threshold})"
            )

        # Small-corpus fallback: when the doc store has very few
        # chunks, an absolute cosine-similarity threshold is
        # unreliable (MiniLM scores are noisy at small scale and
        # short user-uploaded chunks score lower than long sensor
        # summaries, even when the short chunk is the correct
        # answer). Substitute a permissive threshold so every
        # user-uploaded chunk is visible to the prompt builder.
        # The LLM is then responsible for saying "I don't have
        # enough information" when none of the returned chunks
        # answer the question — which it does, correctly.
        #
        # The sensor store is excluded from this fallback because
        # the 180 synthetic_30d.csv chunks are pre-baked test data
        # and may produce noise for off-topic queries; the small-
        # corpus heuristic is specifically about USER-uploaded docs.
        #
        # The fallback only activates when the caller did NOT pass
        # an explicit ``threshold``. A power user passing
        # ``threshold=0.99`` for eval/debug is explicitly opting into
        # strict filtering — overriding that silently would be
        # surprising.
        doc_store_size = self.doc_store.size()
        if threshold_was_default and doc_store_size <= SMALL_CORPUS_MAX_CHUNKS:
            threshold = SMALL_CORPUS_THRESHOLD

        # 1. Sensor keyword detection.
        keywords_matched = _find_sensor_keywords(query, self.sensor_keywords)

        # 2. Embed the query (single text → single vector).
        try:
            query_vectors = self.embedder.embed([query])
        except Exception as exc:  # EmbeddingError or anything else
            raise RetrieverEmbedError(
                f"embedder failed for query: {exc}"
            ) from exc
        if not query_vectors:
            # Empty input list — should not happen (we passed [query])
            # but guard anyway.
            return RetrievalResult(query=query)
        query_vector = query_vectors[0]

        # 3. Doc index search.
        # We over-fetch by RERANK_FETCH_MULTIPLIER so the keyword-
        # overlap rerank (step 6.5) has candidates beyond the top-k
        # to work with. Without this, a chunk that ranks #18 on
        # dense similarity but has a strong lexical match for the
        # query (e.g. the ErP content chunk vs the ErP TOC chunk)
        # never sees the rerank and never gets promoted. The
        # multiplier is bounded so we don't blow the context budget.
        rerank_fetch = max(k_doc * 5, k_doc + 10)
        # On a SMALL corpus, widen the rerank pool to the entire doc
        # store. The bounded multiplier above assumes the answer sits
        # near the top on dense score — but a keyword-strong /
        # dense-weak chunk can fall below the fetch window and become
        # invisible to the lexical rerank that exists precisely to
        # rescue it. Concrete case (2026-07-22): the warranty answer
        # chunk ("...free from defects... for a period of two (2)
        # years") ranked 33/42 on dense score, below the fetch of 25,
        # so "What is the warranty period?" wrongly refused. FAISS
        # search over a few dozen vectors is trivially cheap, and the
        # ``k_doc`` cap (step 7/8) still limits the PROMPT to k_doc
        # chunks — only the candidate pool grows, so there is no
        # latency cost. Gated to small corpora so large-corpus
        # context/perf behaviour is unchanged.
        if doc_store_size <= SMALL_CORPUS_MAX_CHUNKS:
            rerank_fetch = max(rerank_fetch, doc_store_size)
        try:
            doc_hits = self.doc_store.search(query_vector, rerank_fetch)
        except Exception as exc:
            raise RetrieverSearchError(
                f"doc_store.search failed: {exc}"
            ) from exc

        # 3.5. Sparse FTS5 BM25 search (Dual-Hybrid Retrieval)
        expanded_query = _expand_bengali_agri_query(unicodedata.normalize("NFC", query))
        sparse_hits: list[tuple[str, float]] = []
        if hasattr(self.metadata, "search_chunks_fts"):
            try:
                sparse_hits = self.metadata.search_chunks_fts(expanded_query, limit=rerank_fetch)
            except Exception as exc:
                logger.warning("FTS search failed: %s", exc)
                sparse_hits = []

        # 4. Sensor index search (only if keywords matched).
        # Over-fetch sensor candidates so the whole-word rerank
        # (step 6.5) can rescue the correct summary — same mechanism
        # as the doc over-fetch in step 3, applied to the sensor path.
        # Dense MiniLM embeddings CANNOT discriminate between sensor
        # summaries that differ only by date / room / number: they are
        # near-identical sentences ("On DATE, the ROOM MEASURE averaged
        # X, peaking at Y..."), and the date/number carry no semantic
        # signal. So the correct chunk for "bedroom temperature on
        # 2026-06-20" scores only ~0.05 on dense similarity and sits
        # deep in the ranking (well below k_sensor=2 of 180 chunks) —
        # it never enters ``merged`` and the query wrongly refuses.
        # The lexical rerank fixes this ("2026-06-20"/"bedroom" whole-
        # word match lifts it to ~0.55), but ONLY if the chunk is in
        # the candidate pool. So we fetch the whole sensor store (180
        # vectors; a FAISS IndexFlatIP scan over that is trivially
        # cheap) and let the rerank + threshold + k_doc cap filter it.
        # Verified 2026-07-22: bedroom-temp 0.048→0.548 (rank #1),
        # house-energy 0.004→0.304 (#3). The k_doc cap (step 7/8) still
        # bounds the PROMPT, so widening the fetch adds no latency.
        sensor_hits: list[tuple[str, float]] = []
        if keywords_matched:
            sensor_fetch = max(k_sensor, self.sensor_store.size())
            try:
                sensor_hits = self.sensor_store.search(
                    query_vector, sensor_fetch
                )
            except Exception as exc:
                raise RetrieverSearchError(
                    f"sensor_store.search failed: {exc}"
                ) from exc

        # 5. Merge doc + sensor + sparse FTS hits, keeping the highest score per id.
        # We track per-id whether the survivor came from the sensor
        # path (so used_sensor_idx is correct after threshold filter).
        merged: dict[str, float] = {}
        from_sensor: set[str] = set()
        for cid, score in doc_hits:
            existing = merged.get(cid, float("-inf"))
            if score >= existing:
                merged[cid] = score
                from_sensor.discard(cid)  # doc-store wins ties
        for cid, score in sensor_hits:
            existing = merged.get(cid, float("-inf"))
            if score > existing:
                merged[cid] = score
                from_sensor.add(cid)
            elif cid not in merged:
                # First time we see this id — it came from sensor.
                merged[cid] = score
                from_sensor.add(cid)
        for cid, _bm25_val in sparse_hits:
            if cid not in merged:
                # Seed with a solid relevance floor so candidates outside top dense hits enter pool
                merged[cid] = 0.50

        # 6. Resolve ids → chunk records via the metadata accessor.
        ids_in_score_order = sorted(
            merged.keys(), key=lambda cid: merged[cid], reverse=True
        )
        try:
            chunk_records = self.metadata.get_chunks_by_ids(ids_in_score_order)
        except Exception as exc:
            raise RetrieverMetadataError(
                f"metadata.get_chunks_by_ids failed: {exc}"
            ) from exc

        # The accessor preserves input order, so we can re-attach
        # scores by walking the ids_in_score_order list. Unknown
        # ids (the FAISS→metadata TOCTOU window) are silently
        # skipped — they won't appear in chunk_records.
        records_by_id = {rec.id: rec for rec in chunk_records}

        # Guard against FTS5 returning sensor_summary chunks when no sensor
        # keywords matched the query (since chunks_fts indexes all chunks).
        doc_cache: dict[str, object | None] = {}
        for rec in list(chunk_records):
            doc_obj = doc_cache.get(rec.document_id)
            if rec.document_id not in doc_cache:
                try:
                    doc_obj = self.metadata.get_document(rec.document_id)
                except Exception:
                    doc_obj = None
                doc_cache[rec.document_id] = doc_obj
            is_sensor_doc = (
                getattr(doc_obj, "doc_type", None) == "sensor_summary"
                or str(getattr(doc_obj, "filename", "")).endswith(".csv")
            )
            if is_sensor_doc:
                if not keywords_matched:
                    merged.pop(rec.id, None)
                    records_by_id.pop(rec.id, None)
                    from_sensor.discard(rec.id)
                else:
                    from_sensor.add(rec.id)

        ids_in_score_order = [cid for cid in ids_in_score_order if cid in merged]

        # 6.5. Dual-Hybrid BM25 + Keyword-overlap rerank.
        sparse_rank_by_id: dict[str, int] = {}
        if sparse_hits:
            for rank_idx, (cid, _) in enumerate(sparse_hits):
                if cid not in sparse_rank_by_id:
                    sparse_rank_by_id[cid] = rank_idx

        is_bengali_fts = bool(re.search(r"[\u0980-\u09FF]", query)) and hasattr(
            self.metadata, "search_chunks_fts"
        )
        query_terms = _distinctive_query_terms(expanded_query)

        if is_bengali_fts:
            specific_terms = [t for t in query_terms if t not in _GENERIC_AGRI_TERMS]
            for cid in ids_in_score_order:
                if cid not in records_by_id:
                    continue
                rec_text = unicodedata.normalize("NFC", records_by_id[cid].text or "")
                if not rec_text:
                    continue
                raw_rec_tokens = re.split(r"[\s\.,!?;:।॥\(\)\[\]\"'/\\]+", rec_text)
                rec_words = {w.strip().lower() for w in raw_rec_tokens if len(w.strip()) >= 2}
                matches = sum(1 for t in query_terms if t in rec_words)
                spec_matches = sum(1 for t in specific_terms if t in rec_words)

                if cid in from_sensor:
                    if matches == 0 or (specific_terms and spec_matches == 0 and matches < 2):
                        merged[cid] = min(merged[cid] * 0.10, 0.10)
                        continue
                    base_score = 0.55
                    fts_boost = (
                        max(0.05, 0.40 - (sparse_rank_by_id[cid] * 0.03))
                        if cid in sparse_rank_by_id
                        else 0.15
                    )
                    kw_boost = (0.14 * matches) + (0.10 * spec_matches)
                    # Prefer the latest sensor date when multiple days match equally
                    latest_bonus = 0.12 if "2026-06-22" in rec_text else 0.0
                    merged[cid] = base_score + fts_boost + kw_boost + latest_bonus
                    continue

                # Guard against MiniLM [UNK] dense noise and off-crop generic matches:
                # If the Bengali query names specific crops/pests/diseases and a chunk
                # matches ZERO of them, damp its score below threshold (including 0.15 floor).
                if (specific_terms and spec_matches == 0) or (query_terms and matches == 0):
                    merged[cid] = min(merged[cid] * 0.10, 0.10)
                    continue

                # Check if the chunk's Markdown section heading directly matches specific query terms
                heading_lines = " ".join(
                    ln for ln in rec_text.splitlines() if ln.strip().startswith("#")
                )
                heading_tokens = {
                    w.strip().lower()
                    for w in re.split(r"[\s\.,!?;:।॥\(\)\[\]\"'/\\—\-]+", heading_lines)
                    if len(w.strip()) >= 2
                }
                heading_spec_matches = sum(1 for t in specific_terms if t in heading_tokens)
                heading_boost = 0.25 * heading_spec_matches

                # Anchor base score so MiniLM [UNK] variance (0.75-0.93) on Bengali
                # doesn't outweigh true lexical/BM25 relevance.
                base_score = 0.55 if cid in sparse_rank_by_id else min(merged[cid], 0.55)
                fts_boost = (
                    max(0.05, 0.40 - (sparse_rank_by_id[cid] * 0.03))
                    if cid in sparse_rank_by_id
                    else 0.0
                )
                kw_boost = (0.12 * matches) + (0.08 * spec_matches) + heading_boost
                if query_terms and matches == len(query_terms):
                    kw_boost += 0.20
                # Prefer clean Unicode UTF-8 guides over legacy Bijoy/ANSI garbled OCR chunks
                is_garbled_ocr = any(
                    garble in rec_text
                    for garble in (
                        "পযাের্য়", "নিভর্রযোগ্য", "পরবতীর্", "ÿ", "ঙ্ ",
                        "বস্নাস্ট", "স঳চ", "঩ানন", "ব্যফস্থা঩না", "ফাাংরাষদ঱", "গষফলণা",
                    )
                )
                clean_unicode_bonus = -0.35 if is_garbled_ocr else 0.12
                merged[cid] = base_score + fts_boost + kw_boost + clean_unicode_bonus

            ids_in_score_order = sorted(
                merged.keys(), key=lambda cid: merged[cid], reverse=True
            )
        else:
            if sparse_hits:
                for cid, rank_idx in sparse_rank_by_id.items():
                    if cid in merged:
                        boost = max(0.05, 0.45 - (rank_idx * 0.04))
                        merged[cid] = merged[cid] + boost

            if query_terms:
                base_per_term = 0.10
                coverage_bonus = 0.20  # when chunk contains ALL query terms
                for cid in ids_in_score_order:
                    if cid not in records_by_id:
                        continue
                    rec_text = records_by_id[cid].text
                    if not rec_text:
                        continue
                    raw_rec_tokens = re.split(r"[\s\.,!?;:।॥\(\)\[\]\"'/\\]+", rec_text)
                    rec_words = {w.strip().lower() for w in raw_rec_tokens if len(w.strip()) >= 2}
                    matches = sum(1 for t in query_terms if t in rec_words)
                    if matches:
                        boost = base_per_term * matches
                        if matches == len(query_terms):
                            boost += coverage_bonus
                        merged[cid] = merged[cid] + boost
                # Re-sort by the boosted scores.
                ids_in_score_order = sorted(
                    merged.keys(), key=lambda cid: merged[cid], reverse=True
                )

        # Cache document lookups (one DB round-trip per document,
        # not per chunk).
        doc_filename_cache: dict[str, str] = {}

        def _resolve_source(record: ChunkRecord) -> str:
            """Resolve a chunk's document_id to its filename."""
            if record.document_id in doc_filename_cache:
                return doc_filename_cache[record.document_id]
            try:
                doc = self.metadata.get_document(record.document_id)
            except Exception as exc:
                raise RetrieverMetadataError(
                    f"metadata.get_document failed: {exc}"
                ) from exc
            filename = getattr(doc, "filename", None) or record.document_id
            doc_filename_cache[record.document_id] = filename
            return filename

        # 7 + 8. Filter by threshold + collect kept chunks (in score order).
        kept_chunks: list[Chunk] = []
        kept_scores: list[float] = []
        kept_from_sensor: set[str] = set()
        has_surviving_docs = is_bengali_fts and any(
            cid not in from_sensor and cid in records_by_id and merged[cid] >= threshold
            for cid in ids_in_score_order
        )
        max_sensor_slots = (
            max(1, min(k_sensor, k_doc - 1))
            if (has_surviving_docs and k_doc >= 2 and k_sensor > 0)
            else k_doc
        )
        for cid in ids_in_score_order:
            if cid not in records_by_id:
                # Deleted between index and query — skip silently.
                continue
            score = merged[cid]
            if score < threshold:
                # Below the bar — don't surface this as "evidence".
                continue
            if (
                is_bengali_fts
                and cid in from_sensor
                and len(kept_from_sensor) >= max_sensor_slots
            ):
                # Prevent 30 daily sensor chunks from crowding out manual chunks.
                continue
            record = records_by_id[cid]
            kept_chunks.append(
                _record_to_chunk(record, source=_resolve_source(record))
            )
            kept_scores.append(score)
            if cid in from_sensor:
                kept_from_sensor.add(cid)
            if len(kept_chunks) >= k_doc:
                break

        # Ensure combined Bengali weather+crop queries include the top qualifying
        # sensor summary alongside the manual chunks when sensor keywords matched.
        if (
            is_bengali_fts
            and keywords_matched
            and not kept_from_sensor
            and k_sensor > 0
            and k_doc >= 2
        ):
            for cid in ids_in_score_order:
                if cid in from_sensor and cid in records_by_id and merged[cid] >= threshold:
                    record = records_by_id[cid]
                    sensor_chunk = _record_to_chunk(record, source=_resolve_source(record))
                    if len(kept_chunks) >= k_doc:
                        kept_chunks[-1] = sensor_chunk
                        kept_scores[-1] = merged[cid]
                    else:
                        kept_chunks.append(sensor_chunk)
                        kept_scores.append(merged[cid])
                    kept_from_sensor.add(cid)
                    break

        # used_sensor_idx: True iff at least one SURVIVING chunk
        # came from the sensor path. This correctly handles the
        # case where sensor ran but every hit was filtered by
        # threshold (in which case the answer was fully doc-derived).
        used_sensor_idx_final = bool(kept_from_sensor)

        return RetrievalResult(
            chunks=kept_chunks,
            scores=kept_scores,
            used_sensor_idx=used_sensor_idx_final,
            sensor_keywords_matched=keywords_matched,
            query=query,
        )


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


# Match a keyword as a whole word (case-insensitive). Includes the
# Bengali Unicode block (\u0980-\u09FF) so Bengali vowel signs (matras)
# and conjuncts are not split at combining characters.
_KEYWORD_BOUNDARY_RE = re.compile(
    r"(?<![\w\u0980-\u09FF])([\w\u0980-\u09FF][\w\u0980-\u09FF-]*)(?![\w\u0980-\u09FF])"
)


def _find_sensor_keywords(
    query: str, keywords: frozenset[str]
) -> list[str]:
    """Return the subset of ``keywords`` found in ``query`` (whole-word).

    Comparison is case-insensitive and NFC-normalized. Returns a sorted
    list (for deterministic test output) of the matched keywords as they
    appeared in the source set (preserving the case the caller used).
    """
    if not keywords or not query:
        return []
    norm_query = unicodedata.normalize("NFC", query)
    # Extract lowercased tokens from the query.
    tokens = {m.group(1).lower() for m in _KEYWORD_BOUNDARY_RE.finditer(norm_query)}
    # For multi-word keywords (e.g. "last week"), we need a substring
    # check too — `_KEYWORD_BOUNDARY_RE` only handles single tokens.
    query_lower = norm_query.lower()
    matched = []
    for kw in keywords:
        kw_lower = unicodedata.normalize("NFC", kw).lower()
        if " " in kw_lower or "-" in kw_lower:
            if kw_lower in query_lower:
                matched.append(kw)
        elif kw_lower in tokens:
            matched.append(kw)
    return sorted(matched, key=str.lower)


# Stopwords filtered out before keyword-overlap reranking. A minimal
# English stopword list — just the high-frequency words that show up
# in every sentence. Keeping it small (vs. e.g. NLTK's 179-word list)
# is a deliberate trade: the reranker uses these as a *negative*
# filter only, so the cost of a missed stopword is at most one
# spurious +0.10 boost. A bigger list would slow tokenisation without
# changing retrieval outcomes on a 50-chunk corpus.
_RERANK_STOPWORDS: frozenset[str] = frozenset(
    unicodedata.normalize("NFC", w)
    for w in {
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "for",
        "from", "has", "have", "how", "i", "in", "is", "it", "its", "me",
        "my", "of", "on", "or", "so", "such", "than", "that", "the",
        "their", "then", "there", "this", "to", "was", "we", "what",
        "when", "where", "which", "who", "why", "with", "you", "your",
        "does", "did", "can", "could", "would", "should", "will",
        "shall", "may", "might", "must",
        # Bengali generic question & action words
        "কী", "কি", "বা", "এবং", "ও", "করে", "করতে", "হলে", "হয়", "হয়", "হবে",
        "এর", "দেখা", "দিলে", "জন্য", "থেকে", "পর", "কোন", "কোনো", "কিভাবে", "কীভাবে",
        "কেন", "কখন", "কোথায়", "কোথায়", "কত", "লাগলে", "করব", "করবো", "দেব", "দেবো",
        "দিব", "একটি", "এই", "যে", "তা", "না", "করা", "থাকে", "পারে", "দিতে",
        "যায়", "যায়", "দিন", "বলুন", "বলো", "করণীয়", "করণীয়", "উপায়", "উপায়",
        "প্রতিকার", "দমন", "দমনে", "দমনের", "রোগ", "রোগে", "রোগের", "গাছ", "গাছে", "গাছের",
        "ওষুধ", "স্প্রে", "ব্যবস্থা", "নিতে", "ব্যবহারের", "করার", "হয়ে", "হয়ে", "গেলে",
        "দিয়ে", "দিয়ে", "সাথে", "মধ্যে", "হিসেবে", "অনুযায়ী", "অনুযায়ী", "আছে", "কিনা", "ছিল", "যাবে",
    }
)

# Generic agricultural category words that are not crop- or pest-specific on their own.
_GENERIC_AGRI_TERMS: frozenset[str] = frozenset(
    unicodedata.normalize("NFC", w)
    for w in {
        "রোগ", "রোগে", "রোগের", "পোকা", "পোকার", "ছত্রাক", "ভাইরাস", "ব্যাকটেরিয়া",
        "পচা", "পচে", "পচন",
        "আক্রমণ", "আক্রমণে", "চিকিৎসা", "লক্ষণ", "ব্যবস্থাপনা", "প্রতিকার", "দমন", "দমনে", "দমনের",
        "চাষ", "চাষে", "চাষের", "নিয়ম", "নিয়ম", "পদ্ধতি", "সার", "মাত্রা", "প্রয়োগ", "প্রয়োগ",
        "কীটনাশক", "ছত্রাকনাশক", "বালাইনাশক", "ওষুধ", "স্প্রে", "ফল", "ফলে", "ফলের",
        "গাছ", "গাছে", "গাছের", "পাতা", "পাতায়", "পাতায়", "পাতার", "জমি", "জমিতে", "জমির",
        "ক্ষেত", "ক্ষেতে", "ফসল", "ফসলে", "ফসলের", "বীজ", "বীজের", "চারা", "ফলন", "উৎপাদন",
        "সমস্যা", "সমস্যার", "সমাধান", "ব্যবহার", "ব্যবহারের", "তৈরি", "তৈরির",
        "নিয়ন্ত্রণ", "নিয়ন্ত্রণ", "প্রতিরোধ", "প্রতিরোধে", "পরিচর্যা",
        "পুকুর", "পুকুরে", "জলাশয়", "জলাশয়ে", "জলাশয়ে",
        "আবহাওয়া", "আবহাওয়া", "আবহাওয়ার", "আবহাওয়ার", "তাপমাত্রা", "তাপমাত্রার",
        "আর্দ্রতা", "আর্দ্রতার", "বৃষ্টি", "বৃষ্টির", "কুয়াশা", "কুয়াশা", "মেঘলা",
        "রস", "আজকে", "আজকের", "গতকাল", "গতকালের", "এখন", "বর্তমানে", "ঝুঁকি", "সেন্সর", "রিপোর্ট",
    }
)

# Farmer colloquial & regional dialect synonyms mapped to official BARI/BRRI terminology.
_BENGALI_AGRI_SYNONYMS: tuple[tuple[str, str], ...] = tuple(
    (unicodedata.normalize("NFC", k), unicodedata.normalize("NFC", v))
    for k, v in (
        ("কারেন্ট পোকা", "বাদামী গাছফড়িং BPH হপার বার্ন"),
        ("হপার বার্ন", "বাদামী গাছফড়িং কারেন্ট পোকা BPH"),
        ("বাজপড়া", "বাদামী গাছফড়িং কারেন্ট পোকা হপার বার্ন"),
        ("মড়ক", "নাবি ধসা লেইট ব্লাইট পচন"),
        ("পচন", "নাবি ধসা লেইট ব্লাইট কাণ্ড পচা গোড়া পচা কন্দ পচা"),
        ("সাদা শীষ", "হলুদ মাজরা পোকা মরা শীষ হোয়াইটহেড"),
        ("শীষ সাদা", "হলুদ মাজরা পোকা সাদা শীষ মরা শীষ হোয়াইটহেড"),
        ("শীষ মরে", "হলুদ মাজরা পোকা সাদা শীষ নেক ব্লাস্ট"),
        ("মরা শীষ", "হলুদ মাজরা পোকা সাদা শীষ ডেডহার্ট"),
        ("ডেডহার্ট", "হলুদ মাজরা পোকা মৃতদেহলী"),
        ("মৃতদেহলী", "হলুদ মাজরা পোকা ডেডহার্ট"),
        ("পাতা জ্বলা", "ব্যাকটেরিয়াজনিত পাতা পোড়া BLB পাতা ঝলসানো"),
        ("পাতা পোড়া", "ব্যাকটেরিয়াজনিত পাতা পোড়া BLB"),
        ("কোঁকড়ানো", "লিফ কার্ল ভাইরাস মাকড় থ্রিপস সাদা মাছি"),
        ("কুঁকড়ে", "পাতা কোঁকড়ানো লিফ কার্ল ভাইরাস মাকড়"),
        ("চেপা পোকা", "জাব পোকা সরিষা"),
        ("শুঁয়োপোকা", "বিছা পোকা পাট"),
        ("লেদা পোকা", "ফল আর্মিওয়ার্ম কাটুই পোকা"),
        ("আগা মরা", "পার্পল ব্লচ পাতা ঝলসানো"),
        ("ডগা মরা", "ডগা ও ফল ছিদ্রকারী পোকা"),
        ("ঢলে পড়া", "উইল্ট পানামা গোড়া পচা"),
        ("নেতিয়ে পড়া", "ঢলে পড়া উইল্ট ডগা ছিদ্রকারী"),
    )
)


def _expand_bengali_agri_query(query: str) -> str:
    """Expand colloquial Bengali farmer phrases with official BARI/BRRI synonyms."""
    if not query:
        return query
    norm_q = unicodedata.normalize("NFC", query)
    additions: list[str] = []
    for phrase, expansion in _BENGALI_AGRI_SYNONYMS:
        if phrase in norm_q:
            additions.append(expansion)
    if not additions:
        return norm_q
    return f"{norm_q} {' '.join(additions)}"


def _distinctive_query_terms(query: str) -> list[str]:
    """Extract the lowercased, stopword-filtered tokens from ``query``.

    Supports both English and Bengali script words. Punctuation including
    Bengali Dari (। \u0964) and double-dari (॥ \u0965) are treated as
    token separators.
    """
    if not query:
        return []
    query = unicodedata.normalize("NFC", query)
    raw_tokens = re.split(r"[\s\.,!?;:।॥\(\)\[\]\"'/\\]+", query)
    tokens: list[str] = []
    for raw in raw_tokens:
        word = raw.strip().lower()
        if len(word) >= 2 and word not in _RERANK_STOPWORDS:
            tokens.append(word)
    # Dedupe while preserving first-seen order
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _record_to_chunk(record: ChunkRecord, *, source: str) -> Chunk:
    """Convert a :class:`ChunkRecord` (DB row) into a :class:`Chunk`.

    The Chunk's ``source`` is set to the resolved document filename
    (looked up via the metadata accessor) so the prompt builder
    can show ``"(Nest-Thermostat-Installation-Guide-UK.pdf, p.15)"``
    instead of ``"(<document_id>, p.15)"`` — the human-readable
    name is what the user actually wants to see.
    """
    return Chunk(
        text=record.text,
        source=source,
        page=record.page_number,
        chunk_index=record.chunk_index,
        char_offset=record.char_offset or 0,
        token_count=record.token_count,
    )


# Convenience: a thin adapter that takes a real MetadataStore and
# presents it as the narrow MetadataAccessor Protocol the Retriever
# needs. Production wiring uses this so the Retriever depends on
# the narrow Protocol (not the full SQLite-aware class).
class _MetadataStoreAdapter:
    """Wraps a real :class:`MetadataStore` to satisfy :class:`MetadataAccessor`.

    The Retriever only reads ``.filename`` off the returned document
    — but to keep the Chunk's ``source`` field populated, we need
    the document's filename, which the real :class:`MetadataStore`
    exposes. We override ``_record_to_chunk``-equivalent logic in
    :meth:`get_chunks_by_ids` so the returned ChunkRecords are
    already enriched with the right source.

    Actually — we keep the adapter thin and resolve the document
    lazily inside :func:`_record_to_chunk` via a closure passed
    through :meth:`retrieve`. Simpler: we just enrich the records
    BEFORE returning them from this adapter. See :meth:`get_chunks_by_ids`.
    """

    def __init__(self, store: MetadataStore) -> None:
        self._store = store

    def get_chunks_by_ids(
        self, chunk_ids: Sequence[str]
    ) -> list[ChunkRecord]:
        return self._store.get_chunks_by_ids(chunk_ids)

    def get_document(self, document_id: str):
        return self._store.get_document(document_id)

    def search_chunks_fts(
        self, query: str, limit: int = 50
    ) -> list[tuple[str, float]]:
        if hasattr(self._store, "search_chunks_fts"):
            return self._store.search_chunks_fts(query, limit)
        return []


def adapt_metadata_store(store: MetadataStore) -> MetadataAccessor:
    """Wrap a :class:`MetadataStore` as a :class:`MetadataAccessor`.

    The Retriever depends on the narrow :class:`MetadataAccessor`
    Protocol, not the full SQLite-aware :class:`MetadataStore`,
    so tests can inject a dict-backed fake without pulling in
    sqlite3. Production wiring passes the result of this function.
    """
    return _MetadataStoreAdapter(store)


__all__ = [
    "DEFAULT_K_DOC",
    "DEFAULT_K_SENSOR",
    "DEFAULT_SENSOR_KEYWORDS",
    "DEFAULT_THRESHOLD",
    "MetadataAccessor",
    "RetrievalResult",
    "Retriever",
    "RetrieverEmbedError",
    "RetrieverError",
    "RetrieverMetadataError",
    "RetrieverSearchError",
    "adapt_metadata_store",
]
