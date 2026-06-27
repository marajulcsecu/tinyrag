"""Pydantic request + response models for the HTTP API.

The :mod:`tinyrag.api` layer is the only place FastAPI's request /
response types live. They are intentionally **thin** — they validate
shape (types + non-empty strings + sensible bounds) and serialise to
JSON. Business logic stays in :mod:`tinyrag.core` + :mod:`scripts.ask`.

Why Pydantic v2 BaseModels (and not dataclasses)?
-------------------------------------------------
- FastAPI accepts ``pydantic.BaseModel`` natively for ``body:`` /
  ``response_model=`` arguments — using anything else means hand-
  rolling the JSON serialiser.
- Pydantic gives us declarative validation (``min_length=1``,
  ``ge=1``, ``le=1.0``) that surfaces as a clean 422 response on
  bad input — we don't need to write ``if len(query) == 0`` at
  the top of every route.
- The ``model_config = ConfigDict(extra="forbid")`` setting means a
  client sending ``{"query": "...", "hack": true}`` gets a 422
  with the extra field's name, not a silent passthrough.

Why no domain types?
--------------------
The :class:`tinyrag.core.answer.Answer` dataclass is the *domain*
type; the :class:`AskResponse` here is the *wire* type. They
happen to share the same shape today (Step 4.17 just returns the
``Answer.to_dict()`` JSON), but Step 4.19 will start adding
streaming-only fields (``first_token_ms``, ``token_events``) that
the domain dataclass doesn't carry. Keeping the wire type separate
means we don't pollute the domain layer with HTTP-only concepts.

Pure Pydantic / no I/O
----------------------
This module is dependency-free apart from ``pydantic``. It must not
import FastAPI (so it can be unit-tested without spinning up a
test client) and must not import anything from :mod:`tinyrag.core`
or :mod:`tinyrag.storage` (those dependencies belong in the route
handlers, not the schema declarations).

Location: ``src/tinyrag/api/schemas.py``
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ----------------------------------------------------------------------------
# Common
# ----------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Uniform JSON body for every non-2xx response.

    The HTTP layer's exception handlers (:mod:`tinyrag.api.errors`)
    produce this shape regardless of which exception fired. Keeping
    the shape uniform means the dashboard / mobile client can write
    one error-display widget instead of one per status code.

    Attributes
    ----------
    error:
        A short machine-readable error code (e.g. ``"value_error"``,
        ``"metadata_error"``, ``"llm_unavailable"``). Snake_case so
        it plays nicely with TypeScript discriminated unions.
    detail:
        A human-readable explanation safe to surface to the user.
        Never contains the original Python traceback (see
        :func:`tinyrag.api.errors.install_exception_handlers`).
    """

    error: str
    detail: str | None = None

    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------------------
# POST /api/query
# ----------------------------------------------------------------------------


class AskRequest(BaseModel):
    """Body schema for ``POST /api/query``.

    Mirrors the CLI flags from ``scripts/ask.py`` one-for-one so the
    dashboard can expose the same knobs (k-doc, threshold, etc.) as
    the CLI without a second code path.

    Attributes
    ----------
    query:
        The user's question. Must be non-empty after stripping
        whitespace — the same invariant the prompt builder enforces
        (an empty context would let the model hallucinate freely).
    k_doc:
        How many document-index hits to retrieve. Defaults to 3
        (matches :data:`tinyrag.core.retriever.DEFAULT_K_DOC`).
        Bounded to ``[1, 50]`` so a misconfigured client can't
        ask for the whole index.
    k_sensor:
        How many sensor-index hits to retrieve. Defaults to 2.
        Bounded identically to ``k_doc``.
    threshold:
        Minimum cosine-similarity for a chunk to be considered a
        hit. Defaults to 0.3 (matches
        :data:`tinyrag.core.retriever.DEFAULT_THRESHOLD`). Bounded
        to ``[0.0, 1.0]`` — the retriever's invariant.
    max_tokens:
        Cap on generated tokens per response. Defaults to 512.
        Bounded to ``[1, 4096]`` — 4096 is the largest context
        the configured models support.
    log_query:
        Whether to append a row to the ``query_log`` table. Set to
        ``False`` for ad-hoc smoke tests / unit tests that don't
        want to touch the DB. Defaults to ``True``.
    """

    query: str = Field(min_length=1, description="The user's question.")
    k_doc: int = Field(default=3, ge=1, le=50)
    k_sensor: int = Field(default=2, ge=1, le=50)
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    log_query: bool = Field(default=True)

    model_config = ConfigDict(extra="forbid")


#: Response body for ``POST /api/query`` (non-streaming).
#:
#: We type this as ``dict[str, Any]`` because the shape is the
#: :meth:`tinyrag.core.answer.Answer.to_dict` output — a curated dict
#: with citation sub-objects + 2dp-rounded floats + token counts +
#: per-stage timings.
#:
#: Step 4.19 added the SSE streaming path (see ``?stream=true`` on the
#: route). The streaming variant emits
#: ``text/event-stream`` frames rather than this JSON blob:
#:
#: - ``{"event": "token", "delta": "<token>"}`` per LLM token
#: - ``{"event": "done", "answer": <this same dict shape>}`` final frame
#: - ``{"event": "error", "error": "llm_failed", "detail": "..."}``
#:   if the LLM fails mid-stream
#:
#: The JSON contract is unchanged — the streaming path's final
#: ``done`` event carries the same dict this type describes. See
#: ``tinyrag.api.routes_query._stream_answer`` for the wire format.
AskResponse = dict[str, Any]


# ----------------------------------------------------------------------------
# GET /api/status
# ----------------------------------------------------------------------------


class StatusResponse(BaseModel):
    """Body schema for ``GET /api/status``.

    Mirrors FR-39 (SRS §3.5) and NFR-37 (§4). The dashboard polls this
    endpoint to populate the "System Status" panel; the values are
    also surfaced by the CLI's pretty banner (Step 4.16).

    All fields are required — the route always computes every one of
    them so the dashboard doesn't have to handle ``null``. The two
    exceptions are ``ram_mb`` (may be unknown on some platforms —
    e.g. macOS without ``psutil``) and ``llama_cpp_status`` (``"up"``
    or ``"down"``).

    Attributes
    ----------
    ok:
        ``True`` if every subsystem probed is healthy (LLM reachable,
        both FAISS stores loaded, metadata schema initialised). The
        dashboard shows a green pill when ``True`` and a red one when
        ``False``. Set to ``False`` if *any* subsystem check fails.
    model_name:
        The id of the LLM that would answer a query (e.g.
        ``"phi-3-mini"`` or ``"fake-llm"``). ``"unknown"`` if the
        model couldn't be introspected.
    embedding_model:
        The id of the embedder (e.g. ``"sentence-transformers/all-
        MiniLM-L6-v2"``). ``"unknown"`` until the embedder is
        actually loaded (the real embedder is lazy — see Step 4.6).
    embedding_dim:
        The dimensionality of the vector indices. Currently always
        384 (MiniLM). Surfaced so the dashboard can show "384-dim
        embeddings" without parsing the FAISS meta file itself.
    doc_chunk_count:
        Number of vectors in the document FAISS index (the "chunk
        count" the dashboard shows).
    sensor_chunk_count:
        Number of vectors in the sensor FAISS index.
    doc_index_path:
        On-disk path of the doc FAISS index (the dashboard's "Open
        in Finder" link).
    sensor_index_path:
        On-disk path of the sensor FAISS index.
    metadata_db_path:
        On-disk path of the SQLite metadata DB.
    ram_mb:
        Current process resident-set-size in MB, rounded to 1 dp.
        ``None`` if the platform doesn't expose RSS cheaply (we
        fall back to ``None`` rather than fake a number).
    llama_cpp_status:
        ``"up"`` if the configured llama.cpp ``/health`` endpoint
        responded ``200``, ``"down"`` otherwise. Lets the dashboard
        show a clear "LLM offline" banner.
    llama_cpp_url:
        The configured server URL (so the dashboard can show
        "llama.cpp @ 127.0.0.1:8080" in the status panel).
    sensor_source:
        The configured sensor source mode (e.g. ``"simulated"``,
        ``"real_serial"``, ``"mqtt"``). Per FR-41.
    deployment_target:
        The configured deployment target (``"laptop"`` or
        ``"raspberry_pi"``). Useful for the dashboard to render
        a "Running on Pi" badge.
    """

    # ``protected_namespaces=()`` silences Pydantic's warning
    # about the ``model_name`` field shadowing BaseModel's
    # ``model_*`` namespace (it's not actually shadowing —
    # ``model_name`` is a custom field, not a model_config
    # setter — but the validator can't tell).
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    ok: bool
    model_name: str
    embedding_model: str
    embedding_dim: int
    doc_chunk_count: int
    sensor_chunk_count: int
    doc_index_path: str
    sensor_index_path: str
    metadata_db_path: str
    # ``ram_mb`` defaults to ``None`` because some platforms (Windows
    # without Cygwin, locked-down containers) don't expose RSS cheaply.
    ram_mb: float | None = None
    llama_cpp_status: str  # "up" | "down"
    llama_cpp_url: str
    sensor_source: str
    deployment_target: str


# ----------------------------------------------------------------------------
# Skeleton endpoints (4.18 / 4.19 will replace)
# ----------------------------------------------------------------------------


class NotImplementedResponse(BaseModel):
    """Body for the skeleton endpoints that Step 4.17 leaves as 501.

    Steps 4.18 (document management) and 4.19 (SSE streaming) replace
    the skeletons with real handlers. Until then, hitting
    ``POST /api/documents`` etc. returns this body with HTTP 501 so
    clients (and the dashboard router) get a clear "not built yet"
    signal rather than a 404.

    Attributes
    ----------
    error:
        Always ``"not_implemented"`` — a stable string the dashboard
        can branch on.
    detail:
        Human-readable explanation including the step that will fill
        this endpoint in.
    """

    error: str = "not_implemented"
    detail: str

    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------------------
# GET /api/documents
# ----------------------------------------------------------------------------


class DocumentListItemResponse(BaseModel):
    """One row in the :class:`DocumentListResponse.documents` array.

    Mirrors the :class:`tinyrag.storage.metadata.DocumentRecord` row
    shape exactly — same field names, same nullability — so the
    dashboard's "documents table" widget can bind one-to-one. The
    fields are kept flat (no nested objects) so the JSON shape stays
    trivial to render and to ``grep`` in the structured log.

    ``metadata_json`` is the *raw* JSON text from the metadata store
    (i.e. the contents of the ``documents.metadata_json`` column);
    the dashboard parses it client-side if it wants the structured
    view. We deliberately do NOT pre-parse it server-side — that
    would couple the wire schema to whatever the per-doc-type
    metadata shape is today.

    Attributes
    ----------
    id:
        UUID v4 generated at ingest time (the same id stored in
        ``MetadataStore`` and used as the FAISS int↔UUID key).
    filename:
        The basename of the uploaded file (e.g. ``"manual.pdf"``).
        Sanitised server-side to strip any path traversal prefix
        (``Path(filename).name``).
    doc_type:
        One of ``"manual" | "note" | "spec"`` per the
        ``Settings.paths.documents_dir`` ingest convention.
    source_path:
        The on-disk path the file was read from (the tempfile
        path for HTTP uploads; the original ``args.path`` for the
        CLI). Useful provenance even though the temp file is
        unlinked after the response.
    size_bytes:
        File size in bytes as recorded at ingest time.
    num_chunks:
        Number of chunks the chunker produced. Matches the
        ``chunks.document_id = id`` row count.
    content_hash:
        SHA-256 hex digest of the file's bytes — the dedup signal
        at re-ingest (``MetadataStore.get_document_by_hash``).
    ingested_at, last_modified:
        ISO-8601 UTC timestamps. ``ingested_at`` is set once at
        insert; ``last_modified`` is bumped by every
        :meth:`MetadataStore.update_document_chunk_count` call.
    metadata_json:
        Raw JSON text from the ``documents.metadata_json`` column
        (``None`` when the ingest didn't record any per-document
        metadata — e.g. PDF page count for a TXT file).
    """

    id: str
    filename: str
    doc_type: str
    source_path: str
    size_bytes: int
    num_chunks: int
    content_hash: str
    ingested_at: str
    last_modified: str
    metadata_json: str | None = None

    model_config = ConfigDict(extra="forbid")


class DocumentListResponse(BaseModel):
    """Body for ``GET /api/documents``.

    Returns one page of documents plus enough pagination metadata
    for the dashboard's "next page" button. The contract mirrors the
    simplest possible cursor-pagination shape:

    - ``documents`` — the page contents (already in the API's
      newest-first ordering).
    - ``count`` — the **total** document count, not the page size.
      The dashboard uses this for the "N documents" header.
    - ``limit`` / ``offset`` — the values the server actually used
      (echoes the query params so a buggy client can compare).
    - ``next_offset`` — the offset to fetch the next page, or
      ``None`` when there are no more pages. The dashboard renders
      a disabled "Next" button when this is ``None``.

    Attributes
    ----------
    documents:
        The page contents (newest-ingested first).
    count:
        Total number of documents in the corpus. Equal to
        ``len(documents)`` on a single-page response.
    limit:
        Maximum documents per page. The route handler clamps the
        query param to ``[1, 500]`` and echoes the value here.
    offset:
        Page offset — the number of documents skipped before this
        page starts. ``0`` for the first page.
    next_offset:
        ``offset + limit`` when more pages remain; ``None``
        otherwise. Computed server-side so the client doesn't
        have to know the total to decide "is there a next page?".
    """

    documents: list[DocumentListItemResponse]
    count: int = Field(ge=0, description="Total documents in the corpus.")
    limit: int = Field(ge=1, description="Page size used.")
    offset: int = Field(ge=0, description="Page offset used.")
    next_offset: int | None = Field(
        default=None,
        description=(
            "Offset to fetch the next page; ``None`` when no more "
            "pages remain."
        ),
    )

    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------------------
# DELETE /api/documents/{document_id}
# ----------------------------------------------------------------------------


class DocumentDeleteResponse(BaseModel):
    """Body for ``DELETE /api/documents/{id}`` (200 happy path).

    Reports exactly what was removed so the dashboard can render a
    precise "Deleted document X and its 44 chunks / 44 vectors"
    toast without a follow-up GET. Counts come from the cascade
    paths the route handler walks:

    - ``chunks_removed`` — the chunk rows that were cascade-deleted
      by the SQLite ``ON DELETE CASCADE`` FK on
      ``chunks.document_id``.
    - ``vectors_removed`` — the FAISS slots that were removed via
      ``FAISSStore.remove_ids``. May differ from ``chunks_removed``
      in a TOCTOU race (chunks deleted between the metadata read
      and the FAISS remove), so we report them separately.

    A 404 response (the document didn't exist) uses the uniform
    :class:`ErrorResponse` shape instead.

    Attributes
    ----------
    document_id:
        The UUID the route handler received — echoed verbatim so
        the client can match the response to its own log.
    chunks_removed:
        Number of chunk rows the SQLite cascade deleted.
    vectors_removed:
        Number of FAISS vectors actually removed. Equal to
        ``chunks_removed`` in the happy path; can be less in a
        TOCTOU race (FAISS soft-deletes unknown UUIDs silently).
    """

    document_id: str
    chunks_removed: int = Field(ge=0)
    vectors_removed: int = Field(ge=0)

    model_config = ConfigDict(extra="forbid")


#: Response body for ``POST /api/documents`` — the
#: :meth:`tinyrag.scripts.ingest.IngestionReport.to_dict` shape
#: (already JSON-safe). Typed as ``dict[str, Any]`` for the same
#: reason :data:`AskResponse` is: the shape comes from a CLI
#: dataclass, not from a Pydantic model, and Step 4.19 will add
#: upload-only fields (e.g. ``etag``) without changing the JSON
#: contract. The dashboard reads ``body["ok"]``, ``body["doc_id"]``,
#: ``body["num_chunks"]``, ``body["error"]`` — the same keys the
#: ``scripts/ingest.py --json`` output already produces.
DocumentUploadResponse = dict[str, Any]


__all__ = [
    "AskRequest",
    "AskResponse",
    "DocumentDeleteResponse",
    "DocumentListItemResponse",
    "DocumentListResponse",
    "DocumentUploadResponse",
    "ErrorResponse",
    "NotImplementedResponse",
    "StatusResponse",
]
