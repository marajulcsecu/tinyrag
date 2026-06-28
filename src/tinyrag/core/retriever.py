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
#: scores in the 0.04–0.15 range for almost every short query against
#: a tiny or mixed corpus (the embedding is dominated by common-word
#: noise rather than topical signal). The retriever's :meth:`retrieve`
#: detects this case and substitutes :data:`SMALL_CORPUS_THRESHOLD`
#: so user-uploaded chunks don't get silently dropped.
DEFAULT_THRESHOLD = 0.3

#: Threshold used when the doc store has fewer than
#: :data:`SMALL_CORPUS_MAX_CHUNKS` chunks. 0.0 means "include every
#: chunk that has any positive similarity" — effectively "show the
#: user everything they uploaded". The prompt builder caps the
#: token budget, so even if this returns 5 chunks, only the ones
#: that fit get rendered. The model itself is then responsible for
#: saying "I don't have enough information" when none of the
#: returned chunks answer the question.
SMALL_CORPUS_THRESHOLD = 0.0

#: Doc-store size at which the small-corpus fallback activates.
#: Empirically: at ≤ 50 chunks, an absolute similarity threshold is
#: unreliable because (a) scores are noisy at small scale and (b)
#: the user almost certainly uploaded every chunk intentionally, so
#: dropping any of them on a similarity basis is surprising. This
#: covers the "demo + small corpus" use case — a user with 1 manual
#: + 1 PDF typically lands at 5–50 total chunks. Production corpora
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
#: "yesterday", "last week") plus a few obvious additions for a
#: smart-home deployment.
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
    # smart-home-specific phrasing
    "is the", "are the", "what's the", "whats the",
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
        try:
            doc_hits = self.doc_store.search(query_vector, rerank_fetch)
        except Exception as exc:
            raise RetrieverSearchError(
                f"doc_store.search failed: {exc}"
            ) from exc

        # 4. Sensor index search (only if keywords matched).
        sensor_hits: list[tuple[str, float]] = []
        if keywords_matched:
            try:
                sensor_hits = self.sensor_store.search(query_vector, k_sensor)
            except Exception as exc:
                raise RetrieverSearchError(
                    f"sensor_store.search failed: {exc}"
                ) from exc

        # 5. Merge doc + sensor hits, keeping the highest score per id.
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

        # 6.5. Keyword-overlap rerank (a poor-man's BM25 booster).
        # Dense MiniLM embeddings are noisy on short technical
        # queries against mixed corpora (e.g. "What is the ErP
        # directive?" — the TOC chunk that just *mentions* "ErP
        # class 26" outscores the actual ErP content chunk). The
        # fix is a cheap lexical bonus: extract distinctive
        # non-stopword tokens from the query, count how many appear
        # in each chunk's text, and add a per-hit bonus to the
        # dense score.
        #
        # The boost is computed as ``BASE_PER_TERM * matches +
        # FULL_COVERAGE_BONUS when matches == total_terms`` —
        # i.e. a chunk that contains ALL the query's distinctive
        # terms gets a flat bonus on top of the per-term additions,
        # because "every query term is here" is a much stronger
        # relevance signal than "one query term is here". Empirically
        # this lifts the ErP content chunk from FAISS rank #18 to
        # top-1 because the TOC chunk only matches "erp" (1 term,
        # +0.10) while the ErP content matches both "erp" + "directive"
        # (2 terms + coverage bonus, +0.30 total).
        #
        # This is NOT a full BM25 implementation — we don't do IDF
        # weighting or document-frequency lookups. It's a deliberate
        # trade: a small, well-defined constant that fixes the most
        # common dense-only retrieval failure mode without taking on
        # a new dependency.
        query_terms = _distinctive_query_terms(query)
        if query_terms:
            base_per_term = 0.10
            coverage_bonus = 0.20  # when chunk contains ALL query terms
            for cid in ids_in_score_order:
                if cid not in records_by_id:
                    continue
                rec_text = records_by_id[cid].text
                if not rec_text:
                    continue
                rec_lower = rec_text.lower()
                matches = sum(1 for t in query_terms if t in rec_lower)
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
        for cid in ids_in_score_order:
            if cid not in records_by_id:
                # Deleted between index and query — skip silently.
                continue
            score = merged[cid]
            if score < threshold:
                # Below the bar — don't surface this as "evidence".
                continue
            record = records_by_id[cid]
            kept_chunks.append(
                _record_to_chunk(record, source=_resolve_source(record))
            )
            kept_scores.append(score)
            if cid in from_sensor:
                kept_from_sensor.add(cid)
            # Hard cap at ``k_doc`` so the prompt builder never sees
            # more chunks than the caller asked for. The over-fetch in
            # step 3 (``rerank_fetch = max(k_doc * 5, k_doc + 10)``)
            # gives the rerank more candidates to choose from, but
            # the API contract is "k_doc hits, no more". Without this
            # cap, a small-corpus query with permissive threshold can
            # push 20-25 chunks into the prompt and overflow the
            # LLM's context window (see the 4662-token bug we hit on
            # 2026-06-28). The PromptBuilder still does its own
            # token-budget truncation as a second line of defence.
            if len(kept_chunks) >= k_doc:
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


# Match a keyword as a whole word (case-insensitive). The word
# boundary is non-alphanumeric on either side, so "kWh" matches
# in "the kWh was 2.3" but not in "kWhisper".
_KEYWORD_BOUNDARY_RE = re.compile(r"(?<![\w])(\w[\w-]*)(?![\w])")


def _find_sensor_keywords(
    query: str, keywords: frozenset[str]
) -> list[str]:
    """Return the subset of ``keywords`` found in ``query`` (whole-word).

    Comparison is case-insensitive. Returns a sorted list (for
    deterministic test output) of the matched keywords as they
    appeared in the source set (preserving the case the caller
    used).
    """
    if not keywords or not query:
        return []
    # Extract lowercased tokens from the query.
    tokens = {m.group(1).lower() for m in _KEYWORD_BOUNDARY_RE.finditer(query)}
    # For multi-word keywords (e.g. "last week"), we need a substring
    # check too — `_KEYWORD_BOUNDARY_RE` only handles single tokens.
    query_lower = query.lower()
    matched = []
    for kw in keywords:
        kw_lower = kw.lower()
        if " " in kw_lower or "-" in kw_lower:
            # Phrase match (substring is fine for sensor keywords —
            # they're never going to accidentally substring-match
            # common words like "the").
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
_RERANK_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for",
    "from", "has", "have", "how", "i", "in", "is", "it", "its", "me",
    "my", "of", "on", "or", "so", "such", "than", "that", "the",
    "their", "then", "there", "this", "to", "was", "we", "what",
    "when", "where", "which", "who", "why", "with", "you", "your",
    "does", "did", "do", "can", "could", "would", "should", "will",
    "shall", "may", "might", "must",
})


def _distinctive_query_terms(query: str) -> list[str]:
    """Extract the lowercased, stopword-filtered tokens from ``query``.

    Used by the keyword-overlap rerank step (see
    :meth:`Retriever.retrieve`). Returns an empty list when the
    query has no distinctive terms (e.g. just stopwords).

    Tokens are whole-word alphabetic runs (so "OpenTherm" stays
    one token, "non-blocking" splits on the hyphen). Length ≥ 3
    so we skip 1-2 letter words like "I", "a" defensively even if
    they slipped past the stopword list.
    """
    if not query:
        return []
    tokens: list[str] = []
    for m in _KEYWORD_BOUNDARY_RE.finditer(query):
        word = m.group(1).lower()
        if len(word) >= 3 and word not in _RERANK_STOPWORDS:
            tokens.append(word)
    # Dedupe while preserving first-seen order (deterministic for
    # tests). Returned list is used for substring matching against
    # chunk text, not scoring, so order doesn't matter for the
    # algorithm — only for test reproducibility.
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
