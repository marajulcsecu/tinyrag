"""SQLite metadata store — chunk + document + query-log persistence.

This module is the **source of truth** for everything TinyRAG knows
about the documents it has ingested, the chunks those documents were
split into, and the queries it has answered. It is the read/write
seam of the persistence layer; the FAISS vector store (Step 4.8) is
the *other* half of persistence (the embeddings themselves) and
these two stores are kept in lock-step by the ingestion pipeline
(Step 4.9).

Architecture contract
---------------------
The architecture doc (§6, §10.2) pins this module to a single class
``MetadataStore`` plus an exception hierarchy. There is intentionally
no Protocol here — there is exactly one metadata store in the system
(SQLite), so an interface would be ceremony without value. (Compare
with the ``VectorStore`` Protocol in Step 4.8, which exists because
a future contributor may swap FAISS for ChromaDB.)

The schema is **defined in** ``docs/04_database_design_v1.md`` §5.2
verbatim — four tables (``documents``, ``chunks``, ``query_log``,
``schema_version``) with their columns, indexes, and constraints.
This module does NOT re-interpret that schema; it just executes it.
If the schema ever changes, update the doc first, then bump the
``SCHEMA_VERSION`` constant here, then add a migration in
``_MIGRATIONS``.

Why a per-request connection?
-----------------------------
SQLite is fastest with a fresh connection per request — the docs
are explicit: *"Connection per request in FastAPI (short-lived,
cheap with SQLite)"* (§5.4). Opening is ~1 ms; closing releases
the GIL. Long-lived shared connections across FastAPI threads are
a recipe for "database is locked" errors. The class therefore
opens a connection in ``__init__`` and re-opens it on every public
method (the construction-time connection is used by ``__enter__``
context-manager support, but a new one is opened per call).

WAL mode + foreign keys
-----------------------
The DDL section sets ``PRAGMA journal_mode=WAL`` (concurrent reads
during ingestion) and ``PRAGMA foreign_keys=ON`` (enforces the
``ON DELETE CASCADE`` from ``chunks`` to ``documents``). Both are
applied on every fresh connection — SQLite has no
"connection-default" concept, so we re-apply them.

Why parameterized queries everywhere?
-------------------------------------
The doc §5.4 mandates it. Every public method builds its SQL with
``?`` placeholders and passes values as a tuple. String interpolation
is forbidden (and the test suite asserts it). This is the standard
defense against SQL injection; it also lets SQLite cache the
prepared statement across calls.

Batched transactions
--------------------
``insert_chunks`` runs all rows in a single ``BEGIN``/``COMMIT``
block. If any row violates a constraint (e.g. duplicate
``(document_id, chunk_index)``), the whole batch rolls back. This
is the unit of atomicity the ingestion pipeline relies on: a
failure mid-batch never leaves a half-written document.

Public surface
--------------
- :class:`MetadataStore` — the class.
- :class:`MetadataError` and subclasses — typed exception hierarchy.
- :data:`SCHEMA_VERSION` — the current schema version (int).
- :data:`SUPPORTED_DOC_TYPES` — the closed set of ``doc_type`` values
  (mirrors the ``CHECK`` constraint in the DDL).
- :data:`TEXT_PREVIEW_CHARS` — preview length for the UI citation card.

Location: ``src/tinyrag/storage/metadata.py``
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

#: Current schema version. Bump this + add a row to ``_MIGRATIONS`` whenever
#: the DDL in :func:`MetadataStore._schema_sql` changes.
SCHEMA_VERSION: int = 1

#: Closed set of ``documents.doc_type`` values (mirrors the ``CHECK`` constraint).
#: Centralised so the API and the DB can never disagree.
SUPPORTED_DOC_TYPES: frozenset[str] = frozenset({"manual", "faq", "sensor_summary"})

#: How many characters of each chunk's text to keep in ``chunks.text_preview``
#: for the UI citation card (per the DDL).
TEXT_PREVIEW_CHARS: int = 200

#: Maximum number of ?-placeholders to include in a single ``IN (...)`` query.
#: SQLite has a default limit of 999; we leave headroom for the rest of the
#: query and pick a safe round number.
MAX_IN_CLAUSE_BATCH: int = 500


# ----------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------


class MetadataError(Exception):
    """Base class for every metadata-store failure.

    The API layer (Step 4.13) catches this once and decides whether
    to retry / 5xx / surface a clean message. Always subclass rather
    than raising a bare ``Exception`` so the catch site is exact.
    """

    def __init__(self, message: str, *, db_path: str | None = None) -> None:
        super().__init__(message)
        # Preserve the offending path so log lines + 500 responses can
        # show which DB file was involved.
        self.db_path: str | None = db_path


class MetadataSchemaError(MetadataError):
    """The DB file exists but doesn't match our schema.

    Raised when the file is a SQLite DB but lacks the ``schema_version``
    table entirely (i.e. it's a third-party DB the user pointed us at
    by mistake). Distinct from "DB doesn't exist" — that's a happy
    path that triggers ``init_schema``.
    """


class MetadataIntegrityError(MetadataError):
    """A constraint was violated (FK, UNIQUE, CHECK, NOT NULL).

    Raised by the writer methods when the row they tried to insert
    or update breaks a constraint. The original ``sqlite3.IntegrityError``
    is chained via ``__cause__`` for full diagnostics.
    """


class MetadataNotFoundError(MetadataError):
    """A ``get_*`` call asked for a record that doesn't exist.

    Distinct from the (silent) "empty list" return value of
    :meth:`MetadataStore.get_chunks_by_ids` when the IDs are valid
    UUIDs but unknown to this DB — :meth:`get_chunks_by_ids` returns
    ``[]`` in that case (the common "no hits" path). This exception
    is for explicit single-row lookups where the absence is a bug.
    """


# ----------------------------------------------------------------------------
# Dataclasses — typed read API
# ----------------------------------------------------------------------------
#
# SQLite's default cursor returns tuples, which is error-prone. The
# read methods return these frozen dataclasses instead so callers
# don't have to remember column order. The dataclasses are *exactly*
# the rows in the DB — no derived fields, no computed defaults.


@dataclass(frozen=True)
class DocumentRecord:
    """One row of the ``documents`` table."""

    id: str
    filename: str
    doc_type: str
    source_path: str
    size_bytes: int
    num_chunks: int
    content_hash: str
    ingested_at: str
    last_modified: str
    metadata_json: str | None  # raw JSON text; parse with json.loads if needed


@dataclass(frozen=True)
class ChunkRecord:
    """One row of the ``chunks`` table."""

    id: str
    document_id: str
    chunk_index: int
    faiss_idx: int
    page_number: int | None
    text: str
    text_preview: str
    char_offset: int | None
    token_count: int
    embedding_model: str
    created_at: str


@dataclass(frozen=True)
class QueryLogRecord:
    """One row of the ``query_log`` table."""

    id: int
    timestamp: str
    query: str
    top1_score: float | None
    num_chunks: int | None
    retrieval_ms: int | None
    generation_ms: int | None
    total_ms: int | None
    model: str | None
    used_sensor_idx: int
    feedback: str | None


# ----------------------------------------------------------------------------
# MetadataStore
# ----------------------------------------------------------------------------


class MetadataStore:
    """SQLite-backed metadata store for TinyRAG.

    Owns the on-disk SQLite file at ``db_path``. Every public method
    opens its own connection (per the §5.4 best practice) and applies
    the WAL + foreign-keys pragmas on open. Use as a context manager
    (``with MetadataStore(path) as store: ...``) for a slightly
    faster hot path (the construction connection is reused by
    :meth:`_connect` for the first call); standalone usage is
    equivalent.

    Parameters
    ----------
    db_path:
        Path to the SQLite file. Created on first write if it
        doesn't exist (the parent directory is auto-created too).
        Pass ``":memory:"`` for an in-memory DB — handy in tests.
    """

    def __init__(self, db_path: str | Path) -> None:
        # Normalise to a string for sqlite3 (it doesn't like Path
        # objects uniformly across Python versions).
        self._db_path: str = str(db_path)

    # ---- public surface ----------------------------------------------------

    @property
    def db_path(self) -> str:
        """The path to the SQLite file this store reads/writes."""
        return self._db_path

    def __enter__(self) -> MetadataStore:
        # Open one connection up-front for the context-manager path.
        # The public methods always open a fresh one (per §5.4) so
        # this cached connection is unused for normal I/O — but it
        # gives the ``with`` block something to close at exit time.
        self._ctx_conn: sqlite3.Connection = self._open_connection()
        return self

    def __exit__(self, *exc: object) -> None:
        # ``hasattr`` guard for the standalone-usage case (no
        # ``__enter__`` called → no ``_ctx_conn`` to close).
        conn = getattr(self, "_ctx_conn", None)
        if conn is not None:
            conn.close()
            del self._ctx_conn

    def init_schema(self) -> None:
        """Create the schema (documents / chunks / query_log / schema_version).

        Idempotent — every statement uses ``IF NOT EXISTS`` so calling
        it on an already-initialised DB is a no-op (the only effect
        is a handful of cheap "already exists" returns from SQLite).
        Safe to call on every app startup.

        Also writes the ``schema_version`` row if it doesn't exist;
        that row is the basis for future migrations.
        """
        with self._connect() as conn:
            for stmt in self._schema_sql():
                conn.execute(stmt)
            # Record the current schema version. Use INSERT OR IGNORE
            # so re-running init_schema doesn't bump the version
            # every time (a migration bump is a deliberate action).
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version, description) "
                "VALUES (?, ?)",
                (SCHEMA_VERSION, "Initial schema"),
            )
            conn.commit()

    def get_schema_version(self) -> int | None:
        """Return the current schema version, or ``None`` if uninitialised.

        Two distinct "uninitialised" cases:

        - The ``schema_version`` **table exists** but has no rows
          (the very first write hasn't happened) — return ``None``.
        - The ``schema_version`` **table doesn't exist** at all (the
          DB exists but isn't ours) — raise
          :class:`MetadataSchemaError` so the caller knows the file
          is foreign, not just empty.
        """
        with self._connect() as conn:
            # First check whether the table itself exists. This
            # distinguishes "no table → foreign DB" from "table
            # exists but is empty → our DB, not yet initialised".
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='schema_version'"
            ).fetchone()
            if not exists:
                raise MetadataSchemaError(
                    f"DB at {self._db_path!r} is not a TinyRAG metadata DB "
                    f"(no schema_version table). Run init_schema() or "
                    f"point at a different file.",
                    db_path=self._db_path,
                )
            row = conn.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()
            return int(row[0]) if row and row[0] is not None else None

    def insert_document(
        self,
        *,
        filename: str,
        doc_type: str,
        source_path: str,
        size_bytes: int,
        content_hash: str,
        metadata: dict[str, Any] | None = None,
        document_id: str | None = None,
    ) -> str:
        """Insert a new ``documents`` row; return its id.

        The id is a UUID v4 generated here unless the caller passes
        one explicitly (the latter is useful for tests + for
        idempotent re-ingest). All other fields are required; the
        ``num_chunks`` column is left at 0 and updated by
        :meth:`update_document_chunk_count` after chunking.

        Parameters
        ----------
        filename:
            The basename of the source file (e.g. ``"manual.pdf"``).
        doc_type:
            One of :data:`SUPPORTED_DOC_TYPES`. A bad value raises
            :class:`ValueError` *before* the SQL is built — fail
            fast on a programmer error.
        source_path:
            The relative path under ``data/documents/``.
        size_bytes:
            The file size in bytes (of the source file on disk).
        content_hash:
            The SHA-256 of the extracted text. Used for dedup
            detection at re-ingest time.
        metadata:
            Optional free-form dict (page count, author, etc.).
            Serialised to JSON; ``None`` stores SQL NULL.
        document_id:
            Optional explicit UUID. If ``None`` a fresh uuid4 is
            generated.

        Returns
        -------
        str
            The ``id`` of the newly inserted row.

        Raises
        ------
        ValueError
            ``doc_type`` is not in :data:`SUPPORTED_DOC_TYPES`.
        MetadataIntegrityError
            A UNIQUE / NOT NULL / CHECK constraint was violated
            (most likely: duplicate ``content_hash`` or bad ``doc_type``).
        """
        if doc_type not in SUPPORTED_DOC_TYPES:
            raise ValueError(
                f"doc_type must be one of {sorted(SUPPORTED_DOC_TYPES)}, "
                f"got {doc_type!r}"
            )
        new_id = document_id or str(uuid.uuid4())
        metadata_json = json.dumps(metadata) if metadata is not None else None
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO documents ("
                    "  id, filename, doc_type, source_path, size_bytes,"
                    "  num_chunks, content_hash, metadata_json"
                    ") VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
                    (
                        new_id,
                        filename,
                        doc_type,
                        source_path,
                        size_bytes,
                        content_hash,
                        metadata_json,
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                raise MetadataIntegrityError(
                    f"could not insert document {new_id!r} "
                    f"(filename={filename!r}, content_hash={content_hash!r}): {exc}",
                    db_path=self._db_path,
                ) from exc
        return new_id

    def update_document_chunk_count(self, document_id: str, num_chunks: int) -> None:
        """Set ``documents.num_chunks`` and bump ``last_modified``.

        Called by the ingestion pipeline after the chunker has
        produced N chunks and the chunks have been written. A no-op
        (silently) if the document id doesn't exist — the caller
        can use :meth:`get_document` first if it needs to detect
        that.
        """
        if num_chunks < 0:
            raise ValueError(f"num_chunks must be >= 0, got {num_chunks}")
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET num_chunks = ?, "
                "last_modified = ? WHERE id = ?",
                (num_chunks, _now_iso(), document_id),
            )
            conn.commit()

    def insert_chunks(self, chunks: Sequence[dict[str, Any]]) -> list[str]:
        """Insert a batch of chunks in a single transaction.

        All rows go in or none do — if any row violates a constraint
        (e.g. duplicate ``(document_id, chunk_index)``), the entire
        batch rolls back and :class:`MetadataIntegrityError` is
        raised with the original ``sqlite3.IntegrityError`` chained.

        Parameters
        ----------
        chunks:
            A sequence of dicts, one per chunk row. Required keys:
            ``id`` (str UUID), ``document_id`` (str), ``chunk_index``
            (int), ``faiss_idx`` (int), ``text`` (str), ``token_count``
            (int), ``embedding_model`` (str). Optional keys:
            ``page_number`` (int or None), ``char_offset`` (int or None).
            ``text_preview`` is auto-computed (first
            :data:`TEXT_PREVIEW_CHARS` chars of ``text``) — callers
            can override it by passing ``text_preview`` explicitly.

        Returns
        -------
        list[str]
            The ids of the inserted chunks, in the same order as the input.

        Raises
        ------
        ValueError
            ``chunks`` is empty, or any dict is missing a required
            key, or a value has the wrong type.
        MetadataIntegrityError
            A constraint was violated (most commonly: duplicate
            ``(document_id, chunk_index)`` — the chunker emitted the
            same index twice).
        """
        if not chunks:
            raise ValueError("insert_chunks requires at least one chunk")
        required = (
            "id",
            "document_id",
            "chunk_index",
            "faiss_idx",
            "text",
            "token_count",
            "embedding_model",
        )
        rows: list[tuple[Any, ...]] = []
        ids: list[str] = []
        for i, c in enumerate(chunks):
            missing = [k for k in required if k not in c]
            if missing:
                raise ValueError(
                    f"chunk at index {i} is missing required keys: {missing}"
                )
            text = str(c["text"])
            # Auto-compute text_preview if not provided. The DDL says
            # it's NOT NULL, so we always have a value.
            text_preview = c.get("text_preview", text[:TEXT_PREVIEW_CHARS])
            rows.append(
                (
                    str(c["id"]),
                    str(c["document_id"]),
                    int(c["chunk_index"]),
                    int(c["faiss_idx"]),
                    c.get("page_number"),  # may be None
                    text,
                    text_preview,
                    c.get("char_offset"),  # may be None
                    int(c["token_count"]),
                    str(c["embedding_model"]),
                )
            )
            ids.append(str(c["id"]))

        with self._connect() as conn:
            try:
                conn.executemany(
                    "INSERT INTO chunks ("
                    "  id, document_id, chunk_index, faiss_idx, page_number,"
                    "  text, text_preview, char_offset, token_count,"
                    "  embedding_model"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                # Roll back is implicit on context-manager exit, but
                # explicit doesn't hurt and clarifies intent.
                conn.rollback()
                raise MetadataIntegrityError(
                    f"batch insert of {len(chunks)} chunks failed (likely a "
                    f"duplicate (document_id, chunk_index) or a missing "
                    f"document_id FK): {exc}",
                    db_path=self._db_path,
                ) from exc
        return ids

    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Return one document by id, or ``None`` if not found.

        The read methods use :meth:`_connect` to get a fresh
        connection so a long-running writer (the ingestion pipeline)
        can run in parallel with reads (the query path).
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, filename, doc_type, source_path, size_bytes, "
                "num_chunks, content_hash, ingested_at, last_modified, "
                "metadata_json FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_document(row)

    def get_document_by_hash(self, content_hash: str) -> DocumentRecord | None:
        """Return the first document with the given ``content_hash``.

        Used at re-ingest time to detect duplicates — the ingestion
        pipeline computes the SHA-256 of the extracted text and asks
        the store "have we seen this content before?" before doing
        the (expensive) chunking + embedding.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, filename, doc_type, source_path, size_bytes, "
                "num_chunks, content_hash, ingested_at, last_modified, "
                "metadata_json FROM documents WHERE content_hash = ? "
                "ORDER BY ingested_at ASC LIMIT 1",
                (content_hash,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_document(row)

    def list_documents(self) -> list[DocumentRecord]:
        """Return every document, newest-ingested first.

        Used by the UI manage-page (Step 4.22) and by the "what's
        in the corpus?" admin command in ``scripts/inspect_db.py``
        (Step 4.16).

        Ties on ``ingested_at`` (multiple rows inserted in the same
        wall-clock second — common in tests + bulk imports) are
        broken by the SQLite-internal ``rowid``, which is a
        monotonic 64-bit integer assigned at insert time. That
        gives a stable, deterministic order even when the
        ISO-8601 timestamps are identical.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, filename, doc_type, source_path, size_bytes, "
                "num_chunks, content_hash, ingested_at, last_modified, "
                "metadata_json FROM documents "
                "ORDER BY ingested_at DESC, rowid DESC"
            ).fetchall()
        return [_row_to_document(r) for r in rows]

    def get_chunks_by_ids(self, chunk_ids: Sequence[str]) -> list[ChunkRecord]:
        """Return chunks for the given ids, in input order.

        Unknown ids are silently skipped — the caller passed in a
        FAISS hit list, and FAISS may legitimately return a hit on
        a chunk that was deleted between indexing and query
        (TOCTOU window). Empty input returns ``[]``.

        The ``IN`` clause is split into batches of
        :data:`MAX_IN_CLAUSE_BATCH` to stay well under SQLite's
        999-placeholder hard limit even for very large hit lists.
        """
        if not chunk_ids:
            return []
        # Dedupe + preserve order so the caller's order survives.
        seen: set[str] = set()
        unique: list[str] = []
        for cid in chunk_ids:
            if cid not in seen:
                seen.add(cid)
                unique.append(cid)

        results: dict[str, ChunkRecord] = {}
        with self._connect() as conn:
            for batch in _batched(unique, MAX_IN_CLAUSE_BATCH):
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT id, document_id, chunk_index, faiss_idx, "
                    f"page_number, text, text_preview, char_offset, "
                    f"token_count, embedding_model, created_at "
                    f"FROM chunks WHERE id IN ({placeholders})",
                    batch,
                ).fetchall()
                for r in rows:
                    rec = _row_to_chunk(r)
                    results[rec.id] = rec
        # Re-order by the caller's input order so retrieval and
        # re-ranking don't have to re-sort. Unknown ids are skipped.
        return [results[cid] for cid in unique if cid in results]

    def get_chunks_by_document(self, document_id: str) -> list[ChunkRecord]:
        """Return every chunk for a document, ordered by ``chunk_index``.

        Used by the citation-card UI (Step 4.22) and by the
        re-ingest delete flow (Step 4.16).
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, document_id, chunk_index, faiss_idx, "
                "page_number, text, text_preview, char_offset, "
                "token_count, embedding_model, created_at "
                "FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC",
                (document_id,),
            ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def delete_document(self, document_id: str) -> int:
        """Delete a document and cascade-delete its chunks.

        The ``ON DELETE CASCADE`` on ``chunks.document_id`` means
        one DELETE removes the document and every chunk that
        references it. Returns the number of ``documents`` rows
        actually deleted (0 or 1 — the cascade count on ``chunks``
        is not included; use :meth:`get_chunks_by_document` first
        if you need the pre-delete count).

        Foreign-key enforcement is per-connection, so the
        :meth:`_connect` helper applies ``PRAGMA foreign_keys=ON``
        on every fresh connection — without it, the cascade would
        silently not fire.
        """
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            conn.commit()
            return cur.rowcount

    def count_documents(self) -> int:
        """Return the total number of documents. Cheap (uses ``COUNT(*)``)."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
            return int(row[0])

    def count_chunks(self) -> int:
        """Return the total number of chunks across all documents."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            return int(row[0])

    def list_documents_by_filename(
        self, filename: str, *, doc_type: str | None = None
    ) -> list[DocumentRecord]:
        """Return every ``documents`` row whose ``filename`` matches.

        Used by the sensor ingestion CLI (Step 4.15) to detect
        re-ingest of the same CSV: if a ``doc_type='sensor_summary'``
        row already exists with the same ``filename``, the script
        deletes the old chunks + FAISS vectors before re-ingesting,
        keeping the system idempotent without violating the
        ``UNIQUE (document_id, chunk_index)`` constraint on
        ``chunks``.

        Parameters
        ----------
        filename:
            Exact match on ``documents.filename`` (the basename
            the CLI passes in — usually the CSV filename).
        doc_type:
            Optional filter; when set, only rows with that
            ``doc_type`` are returned. This is the recommended
            path for callers that want "the sensor summary for
            this CSV" (set ``doc_type='sensor_summary'`` so a
            future ``doc_type='manual'`` row with the same
            filename won't collide).

        Returns
        -------
        list[DocumentRecord]
            All matching rows, newest-ingested first (same order
            as :meth:`list_documents`). Empty list if nothing
            matches — the caller treats that as "no prior ingest".
        """
        with self._connect() as conn:
            if doc_type is None:
                rows = conn.execute(
                    "SELECT id, filename, doc_type, source_path, size_bytes, "
                    "num_chunks, content_hash, ingested_at, last_modified, "
                    "metadata_json FROM documents WHERE filename = ? "
                    "ORDER BY ingested_at DESC, rowid DESC",
                    (filename,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, filename, doc_type, source_path, size_bytes, "
                    "num_chunks, content_hash, ingested_at, last_modified, "
                    "metadata_json FROM documents "
                    "WHERE filename = ? AND doc_type = ? "
                    "ORDER BY ingested_at DESC, rowid DESC",
                    (filename, doc_type),
                ).fetchall()
        return [_row_to_document(r) for r in rows]

    def log_query(
        self,
        *,
        query: str,
        top1_score: float | None = None,
        num_chunks: int | None = None,
        retrieval_ms: int | None = None,
        generation_ms: int | None = None,
        total_ms: int | None = None,
        model: str | None = None,
        used_sensor_idx: int = 0,
        feedback: str | None = None,
    ) -> int:
        """Append a row to ``query_log``; return the new auto-id.

        All latency fields are in milliseconds (integer). The
        ``used_sensor_idx`` flag distinguishes "doc index only"
        from "both indices" — the DDL defaults it to 0; the
        retriever (Step 4.10) sets it to 1 when sensor data was
        queried.

        The ``query`` field is required; everything else is optional
        (nullable) so the call site can log partial results (e.g.
        a query that failed before retrieval finished).
        """
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO query_log ("
                "  query, top1_score, num_chunks, retrieval_ms, "
                "  generation_ms, total_ms, model, used_sensor_idx, feedback"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    query,
                    top1_score,
                    num_chunks,
                    retrieval_ms,
                    generation_ms,
                    total_ms,
                    model,
                    used_sensor_idx,
                    feedback,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_recent_queries(self, limit: int = 20) -> list[QueryLogRecord]:
        """Return the ``limit`` most recent ``query_log`` rows.

        Used by the admin/debug UI and by the evaluation harness
        (Phase 5). Ordering is ``timestamp DESC`` (newest first).
        """
        if limit <= 0:
            raise ValueError(f"limit must be > 0, got {limit}")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, timestamp, query, top1_score, num_chunks, "
                "retrieval_ms, generation_ms, total_ms, model, "
                "used_sensor_idx, feedback "
                "FROM query_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_query_log(r) for r in rows]

    # ---- internal helpers --------------------------------------------------

    def _open_connection(self) -> sqlite3.Connection:
        """Open a single connection with our standard pragmas applied.

        Used by :meth:`__enter__` to back the context-manager path.
        The public read/write methods use :meth:`_connect` instead
        so they get short-lived connections per §5.4.
        """
        self._ensure_parent_dir()
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, apply pragmas, yield, close.

        Every public method calls this. Each connection is short-lived
        (~1 ms to open) and thread-safe (SQLite's isolation between
        connections makes cross-thread use safe; the
        ``check_same_thread=False`` flag is *not* set because we
        don't share a connection across threads).

        Caveat — ``:memory:`` databases: SQLite's ``:memory:`` DB is
        **per-connection** (every ``sqlite3.connect(":memory:")`` opens
        a fresh empty DB). So ``init_schema()`` followed by a method
        call that re-``_connect()`` would see no tables. We work
        around this by sharing the in-memory DB across all connections
        in one store via a thread-local handle. For real (file-system)
        DBs, the short-lived-connection model works as documented.
        """
        # If the DB path is a real filesystem path, ensure its parent
        # directory exists. SQLite won't create it for us; failing
        # here is a much better error than "unable to open database
        # file" deep in a call.
        self._ensure_parent_dir()

        if self._db_path == ":memory:":
            # Reuse the thread-local in-memory connection across calls
            # so schema persists. Each thread gets its own DB (which
            # is the standard SQLite ":memory:" model — two threads
            # can't share an in-memory DB without cache=shared, and
            # we don't need that here).
            import threading

            if not hasattr(self, "_mem_local"):
                self._mem_local = threading.local()
            local = self._mem_local
            if getattr(local, "conn", None) is None:
                local.conn = sqlite3.connect(":memory:")
                local.conn.execute("PRAGMA foreign_keys = ON")
                local.conn.row_factory = sqlite3.Row
            yield local.conn
            return

        conn = sqlite3.connect(self._db_path)
        try:
            # WAL: concurrent readers + one writer, no blocking.
            # foreign_keys: enforce ON DELETE CASCADE.
            # Both are per-connection; SQLite has no default mechanism.
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA foreign_keys = ON")
            # ``Row`` factory so column access by name is possible
            # for callers that want it (our own readers use the
            # positional ``_row_to_*`` helpers for type safety).
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            conn.close()

    def _ensure_parent_dir(self) -> None:
        """Create the parent directory of ``self._db_path`` if needed.

        No-op for ``:memory:`` (no filesystem). SQLite won't create
        parent directories for us, so failing here gives a much
        cleaner error than "unable to open database file" deep in a
        call.
        """
        if self._db_path == ":memory:":
            return
        parent = Path(self._db_path).parent
        if str(parent) and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _schema_sql() -> list[str]:
        """Return the list of DDL statements for :meth:`init_schema`.

        Kept as a staticmethod so the SQL is grep-able and so the
        schema can be inspected from a Python REPL without a DB
        file. Order matters: ``documents`` before ``chunks`` (FK),
        ``schema_version`` last (it's the one we INSERT into after
        creating the rest).
        """
        return [
            # documents — registry of ingested source files
            """
            CREATE TABLE IF NOT EXISTS documents (
                id              TEXT PRIMARY KEY,
                filename        TEXT NOT NULL,
                doc_type        TEXT NOT NULL,
                source_path     TEXT NOT NULL,
                size_bytes      INTEGER NOT NULL,
                num_chunks      INTEGER DEFAULT 0,
                content_hash    TEXT NOT NULL,
                ingested_at     TEXT NOT NULL DEFAULT (datetime('now')),
                last_modified   TEXT NOT NULL DEFAULT (datetime('now')),
                metadata_json   TEXT,
                CONSTRAINT chk_doc_type CHECK (doc_type IN
                    ('manual', 'faq', 'sensor_summary'))
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_documents_type     ON documents(doc_type)",
            "CREATE INDEX IF NOT EXISTS idx_documents_hash     ON documents(content_hash)",
            "CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename)",
            # chunks — one row per text chunk
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id              TEXT PRIMARY KEY,
                document_id     TEXT NOT NULL,
                chunk_index     INTEGER NOT NULL,
                faiss_idx       INTEGER NOT NULL,
                page_number     INTEGER,
                text            TEXT NOT NULL,
                text_preview    TEXT NOT NULL,
                char_offset     INTEGER,
                token_count     INTEGER NOT NULL,
                embedding_model TEXT NOT NULL,
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (document_id) REFERENCES documents(id)
                    ON DELETE CASCADE,
                CONSTRAINT uniq_doc_chunk UNIQUE (document_id, chunk_index)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_faiss_idx   ON chunks(faiss_idx)",
            # query_log — local log of every query for debugging + eval
            """
            CREATE TABLE IF NOT EXISTS query_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL DEFAULT (datetime('now')),
                query           TEXT NOT NULL,
                top1_score      REAL,
                num_chunks      INTEGER,
                retrieval_ms    INTEGER,
                generation_ms   INTEGER,
                total_ms        INTEGER,
                model           TEXT,
                used_sensor_idx INTEGER NOT NULL DEFAULT 0,
                feedback        TEXT
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_query_log_timestamp ON query_log(timestamp)",
            # schema_version — tracks DDL version for future migrations
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version         INTEGER PRIMARY KEY,
                applied_at      TEXT NOT NULL DEFAULT (datetime('now')),
                description     TEXT
            )
            """,
        ]


# ----------------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Used by the explicit ``last_modified`` update in
    :meth:`MetadataStore.update_document_chunk_count` — SQLite's
    ``datetime('now')`` would also work, but doing it in Python
    keeps the format consistent with what we read back via the
    ``Row`` factory (always UTC, always ISO-8601).
    """
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _batched(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    """Yield consecutive slices of ``items``, each of length <= ``size``.

    Used by :meth:`MetadataStore.get_chunks_by_ids` to chunk an
    arbitrarily-large ``IN`` clause into batches that stay under
    SQLite's 999-placeholder hard limit. Standard library's
    ``itertools.batched`` only landed in Python 3.12; we duplicate
    it here to keep the minimum-version story explicit.
    """
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
    """Map a ``documents`` row to a :class:`DocumentRecord`."""
    # ``Row`` is dict-like AND indexable; we use index access here so
    # the column order matches the SELECT in :meth:`get_document`.
    return DocumentRecord(
        id=row[0],
        filename=row[1],
        doc_type=row[2],
        source_path=row[3],
        size_bytes=row[4],
        num_chunks=row[5],
        content_hash=row[6],
        ingested_at=row[7],
        last_modified=row[8],
        metadata_json=row[9],
    )


def _row_to_chunk(row: sqlite3.Row) -> ChunkRecord:
    """Map a ``chunks`` row to a :class:`ChunkRecord`."""
    return ChunkRecord(
        id=row[0],
        document_id=row[1],
        chunk_index=row[2],
        faiss_idx=row[3],
        page_number=row[4],
        text=row[5],
        text_preview=row[6],
        char_offset=row[7],
        token_count=row[8],
        embedding_model=row[9],
        created_at=row[10],
    )


def _row_to_query_log(row: sqlite3.Row) -> QueryLogRecord:
    """Map a ``query_log`` row to a :class:`QueryLogRecord`."""
    return QueryLogRecord(
        id=row[0],
        timestamp=row[1],
        query=row[2],
        top1_score=row[3],
        num_chunks=row[4],
        retrieval_ms=row[5],
        generation_ms=row[6],
        total_ms=row[7],
        model=row[8],
        used_sensor_idx=row[9],
        feedback=row[10],
    )
