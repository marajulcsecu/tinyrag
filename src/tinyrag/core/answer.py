"""Final answer dataclass for the RAG query pipeline.

This module is the **terminus** of TinyRAG's RAG pipeline. It bundles
the three things a caller of :func:`scripts.ask.run_ask` wants to
have in one place:

1. **The LLM's reply** — the text the model generated.
2. **The citations** — the numbered source chunks the prompt builder
   used, paired with the original :class:`Chunk` records so the
   caller can render a "Sources:" footer, link to a document, or
   store the provenance in a database.
3. **The diagnostics** — per-stage timings, token counts, model name,
   retrieval diagnostics. The API layer (Step 4.14) surfaces these
   in the response payload; the CLI surfaces them in the pretty
   banner.

Why a frozen dataclass (and not a dict)?
----------------------------------------
Same reason :class:`tinyrag.core.prompt_builder.Prompt` and
:class:`tinyrag.core.retriever.RetrievalResult` are frozen
dataclasses:

- **Type-checked field access.** A typo (``answere``) is a
  ``NameError`` at the call site, not a ``KeyError`` deep in a
  template.
- **Stable shape for tests.** The ``to_dict()`` method is
  JSON-serialisable; the same shape is asserted in
  ``tests/test_answer.py`` and reused by the CLI.
- **Frozen** = safe to pass around between the orchestrator, the
  printer, and the JSON serializer without worrying about a
  downstream mutation.

Why a separate :class:`Citation` dataclass?
-------------------------------------------
A citation is a *projection* of a :class:`Chunk` — the same chunk
the prompt builder rendered as ``[3] (Nest.pdf, p.7) Some text...``,
with the score and chunk_id added. Storing the citation
independently of the chunk lets the API layer render the footer
as a list of links (``Citation 1: doc.pdf p.7 score=0.82``) without
re-walking the prompt, and lets the eval set (Phase 5) compare
expected vs. retrieved sources.

Pure functions / no I/O
-----------------------
This module is in :mod:`tinyrag.core` (the "no I/O" layer). It
takes already-built objects (:class:`RetrievalResult`,
:class:`Prompt`, the LLM's text + :class:`GenerationStats`) and
assembles an :class:`Answer`. The actual LLM call + retrieval are
in :mod:`scripts.ask` and :mod:`tinyrag.core.retriever`.

Location: ``src/tinyrag/core/answer.py``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tinyrag.core.chunker import Chunk

if TYPE_CHECKING:
    from tinyrag.core.retriever import RetrievalResult


# ----------------------------------------------------------------------------
# Public value types
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    """One numbered citation in the answer's "Sources:" footer.

    Attributes
    ----------
    number:
        1-based citation number — matches the ``[N]`` marker the
        prompt builder renders in the context block. So
        ``Citation.ref == "[3]"`` and the model is expected to use
        that exact marker in its answer.
    chunk_id:
        The UUID of the underlying chunk row in the metadata DB.
        Lets the API layer link the citation to the original
        document (``SELECT * FROM chunks WHERE id = ?``) for a
        "view source" affordance.
    source:
        The human-readable source label (filename for doc chunks,
        ``"sensor-summary"`` for sensor chunks). Same string the
        prompt builder renders in ``[N] (source, p.X)``.
    page:
        The page number for PDF chunks, ``None`` for TXT/MD/sensor
        chunks. Echoed in the citation so the user can flip to
        the right page.
    score:
        The cosine-similarity score from the retriever, in
        ``[0.0, 1.0]`` (always ≥ the threshold — citations
        correspond to surviving chunks only). Useful for ranking
        ("the model's top source was score 0.82").
    preview:
        The first ~120 characters of the chunk text, whitespace-
        normalised. Used to render the citation footer without
        re-walking the chunk list. The full text is recoverable
        from the metadata DB via ``chunk_id``.
    """

    number: int
    chunk_id: str
    source: str
    page: int | None
    score: float
    preview: str

    @property
    def ref(self) -> str:
        """The ``[N]`` marker the model sees in the context block.

        Example: ``Citation(3, ...).ref == "[3]"``.
        """
        return f"[{self.number}]"

    @property
    def location(self) -> str:
        """``source, p.X`` or just ``source`` (no page).

        Mirrors the format the prompt builder uses so the
        citation footer reads naturally: ``[3] Nest.pdf, p.7``.
        """
        if self.page is None:
            return self.source
        return f"{self.source}, p.{self.page}"


@dataclass(frozen=True)
class Answer:
    """The final output of one RAG query.

    Bundles the model's reply, the numbered citations, and the
    per-stage diagnostics. JSON-serialisable via :meth:`to_dict`
    so the same shape is used by the CLI pretty banner, the CLI
    ``--json`` mode, and the API response payload.

    Attributes
    ----------
    query:
        The original user question, echoed back.
    text:
        The model's reply — the full generated text (concatenation
        of every streamed token). May be the refusal sentence
        ``"I don't have enough information in the provided
        documents."`` if the retriever returned no chunks.
    used_sensor_idx:
        ``True`` iff the answer's retrieved context includes at
        least one sensor-summary chunk (i.e. the query matched a
        sensor keyword AND the sensor index returned at least one
        hit that survived the threshold). Surfaced in the response
        so the user can tell "this answer used live sensor data".
    top_score:
        The highest cosine-similarity score across the retrieved
        chunks, or ``None`` if the retrieval returned nothing.
        Useful for confidence gating — a low ``top_score`` is a
        signal that the model may have hallucinated.
    model_name:
        The model id that produced the answer (e.g.
        ``"phi-3-mini"`` or ``"fake-llm"``). Surfaced in the
        response payload so the API can answer "which model
        answered?".
    citations:
        The numbered :class:`Citation` list, in prompt order
        (so ``citations[0]`` is the ``[1]`` the model saw first).
        Empty when retrieval returned nothing.
    chunks_used:
        Number of chunks that actually fit in the prompt (after
        the budget trim). Mirrors :attr:`Prompt.chunks_used`.
    chunks_dropped:
        Number of chunks that had to be dropped to fit the token
        budget. Mirrors :attr:`Prompt.chunks_dropped`.
    prompt_tokens:
        Tokens consumed by the prompt (system + context + user
        message). Mirrors :attr:`Prompt.prompt_tokens`.
    completion_tokens:
        Tokens generated by the LLM. Mirrors
        :attr:`GenerationStats.completion_tokens`.
    total_tokens:
        ``prompt_tokens + completion_tokens``. Mirrors
        :attr:`GenerationStats.total_tokens`.
    duration_retrieve_ms:
        Wall-clock ms for the retriever stage. Useful for
        diagnosing "is it the vector search or the LLM?".
    duration_prompt_ms:
        Wall-clock ms for the prompt builder stage. Usually
        small (<5 ms) — the chunker-compatible tiktoken encoding
        is fast.
    duration_llm_ms:
        Wall-clock ms for the LLM ``generate()`` call. The
        dominant cost on real models (~2-30 s).
    duration_total_ms:
        Sum of the three stage durations, rounded to 2 dp by
        :meth:`to_dict`. The number the roadmap's "<3 s on
        laptop" gate compares against.
    """

    query: str
    text: str
    used_sensor_idx: bool = False
    top_score: float | None = None
    model_name: str = ""
    citations: list[Citation] = field(default_factory=list)
    chunks_used: int = 0
    chunks_dropped: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_retrieve_ms: float = 0.0
    duration_prompt_ms: float = 0.0
    duration_llm_ms: float = 0.0
    duration_total_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (rounds floats to 2 dp).

        The shape is the CLI's ``--json`` mode and the API's
        response payload. The ``citations`` list is a list of
        plain dicts (not :class:`Citation` objects) so the result
        is JSON-serialisable without any custom encoder.
        """
        return {
            "query": self.query,
            "text": self.text,
            "used_sensor_idx": self.used_sensor_idx,
            "top_score": (
                round(self.top_score, 4) if self.top_score is not None else None
            ),
            "model_name": self.model_name,
            "citations": [
                {
                    "number": c.number,
                    "ref": c.ref,
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "page": c.page,
                    "score": round(c.score, 4),
                    "location": c.location,
                    "preview": c.preview,
                }
                for c in self.citations
            ],
            "chunks_used": self.chunks_used,
            "chunks_dropped": self.chunks_dropped,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "duration_retrieve_ms": round(self.duration_retrieve_ms, 2),
            "duration_prompt_ms": round(self.duration_prompt_ms, 2),
            "duration_llm_ms": round(self.duration_llm_ms, 2),
            "duration_total_ms": round(self.duration_total_ms, 2),
        }

    @property
    def is_refusal(self) -> bool:
        """``True`` iff the answer is the documented refusal sentence.

        The prompt builder's :data:`DEFAULT_SYSTEM_PROMPT` tells
        the model to reply exactly with ``"I don't have enough
        information in the provided documents."`` when the
        context is empty. The API layer uses this flag to set
        the ``confidence: "low"`` field in the response; the
        eval set (Phase 5) uses it to mark an answer as
        ungrounded.
        """
        # Compare case-folded, stripped — the small models
        # occasionally add whitespace or change case.
        return self.text.strip().lower().startswith(
            "i don't have enough information"
        )


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


#: Preview length (in characters) for the citation's ``preview`` field.
#: Picked to be short enough for a "Sources:" footer line but long
#: enough to give the user a hint of the chunk's content. Trimmed
#: at a word boundary to avoid mid-word cuts.
_CITATION_PREVIEW_CHARS = 120


def _make_preview(text: str, *, max_chars: int = _CITATION_PREVIEW_CHARS) -> str:
    """Return a single-line, whitespace-collapsed preview of ``text``.

    Used for the :class:`Citation.preview` field. The full text is
    recoverable from the metadata DB via ``chunk_id``; the preview
    is just a one-line hint for the citation footer.

    Truncation rules:

    1. Collapse runs of whitespace (newlines, tabs, multiple spaces)
       to a single space — chunk text often contains both, and
       they make the preview look ragged in a CLI footer.
    2. Strip leading/trailing whitespace.
    3. If the result is longer than ``max_chars``, cut at the
       last space ≤ ``max_chars`` and append ``"…"`` (a single
       Unicode horizontal ellipsis) so the user knows it was
       truncated. Never truncates mid-word.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    # Cut at the last space ≤ max_chars so we don't break a word.
    cut = collapsed.rfind(" ", 0, max_chars)
    if cut <= 0:
        # No space found in the first max_chars — cut hard.
        return collapsed[:max_chars] + "…"
    return collapsed[:cut] + "…"


def build_citations(
    retrieval: RetrievalResult,
    *,
    chunk_ids: list[str],
) -> list[Citation]:
    """Pair the retriever's chunks with their stable chunk_id + score.

    The prompt builder numbers the surviving chunks 1..N. To render
    a "Sources:" footer, the caller needs to know which underlying
    chunk (UUID, filename, page, score) each ``[N]`` refers to.

    Parameters
    ----------
    retrieval:
        The :class:`RetrievalResult` returned by
        :meth:`Retriever.retrieve`. ``retrieval.chunks`` is in
        score-DESC order, parallel to ``retrieval.scores``.
    chunk_ids:
        The UUIDs of the chunks the prompt builder ACTUALLY used,
        in prompt order (after the budget trim). This is what
        :class:`tinyrag.storage.metadata.MetadataStore.get_chunks_by_ids`
        would return when called with the prompt-builder's selected
        indices. The CLI doesn't have easy access to the
        chunk-by-chunk UUID mapping (the :class:`Chunk` dataclass
        doesn't carry its DB id — only its text/source/page), so
        this helper is called by the API layer where the mapping
        IS available. The :mod:`scripts.ask` CLI instead uses
        :func:`build_citations_from_chunks` which derives the
        same info from just the surviving ``Chunk`` list.

    Returns
    -------
    list[Citation]
        Numbered 1..N, parallel to ``retrieval.chunks`` (which is
        the prompt order — prompt builder never reorders). Empty
        when ``retrieval.chunks`` is empty.
    """
    citations: list[Citation] = []
    for n, (chunk, score) in enumerate(
        zip(retrieval.chunks, retrieval.scores, strict=True), start=1
    ):
        cid = chunk_ids[n - 1] if n - 1 < len(chunk_ids) else ""
        citations.append(
            Citation(
                number=n,
                chunk_id=cid,
                source=chunk.source,
                page=chunk.page,
                score=score,
                preview=_make_preview(chunk.text),
            )
        )
    return citations


def build_citations_from_chunks(
    chunks: list[Chunk],
    scores: list[float],
) -> list[Citation]:
    """Build :class:`Citation` objects from a surviving chunk list.

    Convenience for the CLI / script layer which doesn't have the
    full :class:`RetrievalResult` — it has the chunk list (from
    :attr:`Prompt.chunks_used` count) and the scores (from
    :attr:`RetrievalResult.scores`). The ``chunk_id`` is left as
    the empty string (the CLI doesn't query the DB to resolve
    it; the API layer can if needed).

    Parameters
    ----------
    chunks:
        The :class:`Chunk` objects the prompt builder used, in
        prompt order. Usually ``retrieval.chunks`` (or a prefix
        of it, if the prompt builder dropped some).
    scores:
        Parallel list of cosine-similarity scores (same length,
        same order).

    Returns
    -------
    list[Citation]
        Numbered 1..N, parallel to ``chunks``. Empty when
        ``chunks`` is empty.
    """
    citations: list[Citation] = []
    for n, (chunk, score) in enumerate(
        zip(chunks, scores, strict=True), start=1
    ):
        citations.append(
            Citation(
                number=n,
                chunk_id="",  # CLI doesn't resolve chunk_id
                source=chunk.source,
                page=chunk.page,
                score=score,
                preview=_make_preview(chunk.text),
            )
        )
    return citations


__all__ = [
    "Answer",
    "Citation",
    "build_citations",
    "build_citations_from_chunks",
]
