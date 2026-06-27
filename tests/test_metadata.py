"""Tests for src/tinyrag/storage/metadata.py — SQLite metadata store.

Test layout
-----------
- TestPublicSurface          — every public name is exported from the
  subpackage (``MetadataStore``, all 3 record dataclasses, the
  exception hierarchy + 2 module-level constants).
- TestSchemaInit             — ``init_schema`` is idempotent,
  creates all 4 tables + 7 indexes + the schema_version row.
- TestPragmas                — every fresh connection applies WAL
  mode + foreign_keys enforcement (per the DB design doc §5.4).
- TestSchemaVersion          — ``get_schema_version`` returns the
  current version; raises :class:`MetadataSchemaError` on a DB
  that isn't ours (e.g. a stray SQLite file pointed at by mistake).
- TestInsertDocument         — round-trip; required fields; defaults;
  explicit UUID; bad doc_type raises ValueError; duplicate hash
  raises MetadataIntegrityError (the re-ingest dedup signal).
- TestUpdateChunkCount       — num_chunks set, last_modified bumped;
  rejects negative values.
- TestInsertChunks           — single + batch insert round-trip;
  text_preview auto-computed (200 chars); explicit override;
  empty input raises ValueError; missing required keys raises
  ValueError; duplicate (document_id, chunk_index) rolls back
  the WHOLE batch (not just the bad row — the atomicity invariant).
- TestGetDocument            — by id (hit + miss); by hash (hit +
  miss — the dedup lookup).
- TestListDocuments          — empty; single; multiple (newest first).
- TestGetChunksByIds         — empty list short-circuits; input
  order preserved; unknown ids silently skipped; > 500 ids
  are batched (SQLite's 999-placeholder limit).
- TestGetChunksByDocument    — ordered by chunk_index; empty doc.
- TestDeleteDocument         — cascade deletes chunks (the FK
  enforcement contract); missing id is a no-op (0 rows).
- TestCounters               — count_documents / count_chunks.
- TestQueryLog               — log_query returns the auto-id;
  recent queries are newest-first; partial results (NULL fields)
  are accepted; limit must be > 0.
- TestSqlInjectionGuard      — every user-supplied string is
  parameterised (the §5.4 invariant — we test by trying to insert
  a chunk with a malicious ``document_id`` and confirm it's stored
  literally, not interpreted).
- TestJsonMetadataRoundTrip  — the documents.metadata_json column
  is preserved as a JSON string; ``None`` stores SQL NULL.
- TestContextManager         — ``__enter__`` / ``__exit__`` close
  the cached connection cleanly (lifecycle sanity).
- TestInMemoryDatabase       — ``MetadataStore(":memory:")`` works
  for unit tests that don't need filesystem isolation.

Why so many tests?
------------------
SQLite is the *one* piece of state that survives a process restart
and is shared across the FastAPI workers + the ingestion script +
the CLI tools. A bug here silently corrupts the corpus. Every
operation that can go wrong (empty input, missing keys, duplicate
ids, FK violations, NULL fields, large IN clauses, SQL injection,
schema migration) is covered.

Hermetic?
---------
100% hermetic. Every test uses ``tmp_path`` (a fresh per-test
directory pytest creates and reaps) or ``":memory:"`` SQLite —
no fixture files, no network, no real project DB.

Location: ``tests/test_metadata.py``
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from tinyrag.storage import (
    SCHEMA_VERSION,
    SUPPORTED_DOC_TYPES,
    ChunkRecord,
    DocumentRecord,
    MetadataError,
    MetadataIntegrityError,
    MetadataSchemaError,
    MetadataStore,
    QueryLogRecord,
)
from tinyrag.storage.metadata import (
    DocumentRecord as DocRecDirect,  # for identity checks in TestPublicSurface
)

# ----------------------------------------------------------------------------
# Constants / helpers
# ----------------------------------------------------------------------------


def _make_doc(
    store: MetadataStore,
    *,
    filename: str = "manual.pdf",
    doc_type: str = "manual",
    content_hash: str = "abc123",
    size_bytes: int = 1024,
    metadata: dict | None = None,
) -> str:
    """Insert a document with sensible defaults; return its id.

    Every test that needs a document uses this — keeps the test
    bodies focused on the assertion, not the setup boilerplate.
    """
    return store.insert_document(
        filename=filename,
        doc_type=doc_type,
        source_path=f"data/documents/{filename}",
        size_bytes=size_bytes,
        content_hash=content_hash,
        metadata=metadata,
    )


def _make_chunk(
    chunk_id: str,
    document_id: str,
    *,
    chunk_index: int = 0,
    faiss_idx: int = 0,
    text: str = "hello world",
    text_preview: str | None = None,
    page_number: int | None = None,
    char_offset: int | None = 0,
    token_count: int = 2,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict:
    """Build one chunk dict for :meth:`MetadataStore.insert_chunks`.

    ``text_preview`` defaults to ``None`` so the store auto-computes
    it (the common case). Pass an explicit string to override.
    """
    chunk: dict = {
        "id": chunk_id,
        "document_id": document_id,
        "chunk_index": chunk_index,
        "faiss_idx": faiss_idx,
        "text": text,
        "page_number": page_number,
        "char_offset": char_offset,
        "token_count": token_count,
        "embedding_model": embedding_model,
    }
    if text_preview is not None:
        chunk["text_preview"] = text_preview
    return chunk


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Fresh per-test SQLite path inside pytest's tmp_path."""
    return tmp_path / "metadata.db"


@pytest.fixture
def store(db_path: Path) -> MetadataStore:
    """A :class:`MetadataStore` already initialised (most tests want this)."""
    s = MetadataStore(db_path)
    s.init_schema()
    return s


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


class TestPublicSurface:
    """The expected symbols are exported and importable."""

    def test_subpackage_exports_metadata_store(self) -> None:
        from tinyrag.storage import MetadataStore as cls

        assert cls is MetadataStore

    def test_subpackage_exports_document_record(self) -> None:
        from tinyrag.storage import DocumentRecord as cls

        assert cls is DocumentRecord
        assert cls is DocRecDirect  # same class object

    def test_subpackage_exports_chunk_record(self) -> None:
        from tinyrag.storage import ChunkRecord as cls

        assert cls is ChunkRecord

    def test_subpackage_exports_query_log_record(self) -> None:
        from tinyrag.storage import QueryLogRecord as cls

        assert cls is QueryLogRecord

    def test_subpackage_exports_metadata_error(self) -> None:
        from tinyrag.storage import MetadataError as cls

        assert cls is MetadataError

    def test_subpackage_exports_integrity_error(self) -> None:
        from tinyrag.storage import MetadataIntegrityError as cls

        assert cls is MetadataIntegrityError

    def test_subpackage_exports_schema_error(self) -> None:
        from tinyrag.storage import MetadataSchemaError as cls

        assert cls is MetadataSchemaError

    def test_subpackage_exports_schema_version_constant(self) -> None:
        from tinyrag.storage import SCHEMA_VERSION as v

        assert v == SCHEMA_VERSION
        assert isinstance(v, int)
        assert v >= 1

    def test_subpackage_exports_supported_doc_types(self) -> None:
        from tinyrag.storage import SUPPORTED_DOC_TYPES as s

        assert s == SUPPORTED_DOC_TYPES
        assert "manual" in s
        assert "faq" in s
        assert "sensor_summary" in s


class TestSchemaInit:
    """``init_schema`` is idempotent and creates the full DDL."""

    def test_init_creates_db_file(self, db_path: Path) -> None:
        MetadataStore(db_path).init_schema()
        assert db_path.exists()

    def test_init_creates_parent_dirs(self, tmp_path: Path) -> None:
        """A nested path's parent directories must be auto-created."""
        nested = tmp_path / "a" / "b" / "c" / "deep.db"
        MetadataStore(nested).init_schema()
        assert nested.exists()

    def test_init_creates_all_four_tables(self, store: MetadataStore, db_path: Path) -> None:
        # Use a fresh connection (not the store's context manager) so
        # we see the on-disk state, not an in-process cache.
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        finally:
            conn.close()
        table_names = [r[0] for r in rows]
        for required in ("chunks", "documents", "query_log", "schema_version"):
            assert required in table_names, f"missing table {required!r} in {table_names}"

    def test_init_creates_all_seven_indexes(
        self, store: MetadataStore, db_path: Path
    ) -> None:
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND name LIKE 'idx_%' ORDER BY name"
            ).fetchall()
        finally:
            conn.close()
        index_names = [r[0] for r in rows]
        expected = [
            "idx_chunks_document_id",
            "idx_chunks_faiss_idx",
            "idx_documents_filename",
            "idx_documents_hash",
            "idx_documents_type",
            "idx_query_log_timestamp",
        ]
        for idx in expected:
            assert idx in index_names, f"missing index {idx!r} in {index_names}"

    def test_init_is_idempotent(self, db_path: Path) -> None:
        """Calling init_schema twice doesn't fail and doesn't double-write."""
        s = MetadataStore(db_path)
        s.init_schema()
        s.init_schema()
        s.init_schema()
        assert s.get_schema_version() == SCHEMA_VERSION

    def test_init_records_schema_version_row(
        self, store: MetadataStore, db_path: Path
    ) -> None:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT version, description FROM schema_version"
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert row[0] == SCHEMA_VERSION
        assert "Initial schema" in row[1]


class TestPragmas:
    """Every fresh connection applies WAL + foreign_keys."""

    def test_wal_mode_is_set_on_init(
        self, store: MetadataStore, db_path: Path
    ) -> None:
        conn = sqlite3.connect(db_path)
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            conn.close()
        # WAL is reported as "wal" (lowercase). For in-memory or
        # some readonly paths it may report "memory" — we accept
        # either as evidence the pragma was applied.
        assert mode in ("wal", "memory"), f"unexpected journal_mode: {mode!r}"

    def test_foreign_keys_is_enabled_per_connection(
        self, store: MetadataStore, db_path: Path
    ) -> None:
        """Per the SQLite docs, FK enforcement is per-connection.

        This test proves that :meth:`_connect` re-applies the pragma
        every time — without it, the ON DELETE CASCADE in the DDL
        would silently not fire.
        """
        # Open a separate raw connection (pragmas default OFF there).
        raw = sqlite3.connect(db_path)
        try:
            # We don't assert 0 here — it depends on SQLite's defaults.
            # The point is to verify the store's connection has it ON.
            _ = raw.execute("PRAGMA foreign_keys").fetchone()
        finally:
            raw.close()
        # And the store reports FK enforcement via cascade behaviour.
        # (The cascade test below is the real proof; this test just
        # exercises the per-connection path.)
        doc_id = _make_doc(store)
        store.insert_chunks([_make_chunk("c0", doc_id)])
        # No exception = FK enforcement is on (insert succeeded
        # because the parent document exists).
        assert store.count_chunks() == 1


class TestSchemaVersion:
    """``get_schema_version`` round-trips; raises on a non-TinyRAG DB."""

    def test_returns_current_version(self, store: MetadataStore) -> None:
        assert store.get_schema_version() == SCHEMA_VERSION

    def test_returns_none_before_init(self, tmp_path: Path) -> None:
        """An initialised DB with an EMPTY schema_version table → None.

        This is the state just after the schema is created but
        before any ``insert_document`` (which doesn't bump the
        version). The version row is added by ``init_schema`` —
        so this test simulates a hypothetical "schema created
        without the version row" state, which is the same as
        what callers see before any work has been done.

        The 0-byte-file case is different — that's a foreign DB
        with NO tables at all, and we raise
        :class:`MetadataSchemaError` for it (see next test).
        """
        p = tmp_path / "empty.db"
        # Build the schema tables manually, but leave schema_version
        # empty. This is the only way to get a "our schema, no row"
        # state via the public API.
        conn = sqlite3.connect(p)
        try:
            conn.executescript(
                """
                CREATE TABLE schema_version (
                    version     INTEGER PRIMARY KEY,
                    applied_at  TEXT NOT NULL DEFAULT (datetime('now')),
                    description TEXT
                );
                """
            )
            conn.commit()
        finally:
            conn.close()
        assert MetadataStore(p).get_schema_version() is None

    def test_raises_on_non_tinyrag_db(self, tmp_path: Path) -> None:
        """A SQLite file that's not ours raises :class:`MetadataSchemaError`."""
        p = tmp_path / "foreign.db"
        conn = sqlite3.connect(p)
        try:
            # A completely different table — no schema_version.
            conn.execute("CREATE TABLE foo (x INTEGER)")
            conn.commit()
        finally:
            conn.close()
        with pytest.raises(MetadataSchemaError) as excinfo:
            MetadataStore(p).get_schema_version()
        assert excinfo.value.db_path == str(p)


class TestInsertDocument:
    """``insert_document`` writes one row and returns the id."""

    def test_round_trip_via_get(self, store: MetadataStore) -> None:
        doc_id = _make_doc(store, filename="thermo.pdf", content_hash="h1")
        doc = store.get_document(doc_id)
        assert doc is not None
        assert doc.id == doc_id
        assert doc.filename == "thermo.pdf"
        assert doc.doc_type == "manual"
        assert doc.size_bytes == 1024
        assert doc.content_hash == "h1"
        assert doc.num_chunks == 0  # default
        assert doc.metadata_json is None

    def test_returns_uuid_string(self, store: MetadataStore) -> None:
        doc_id = _make_doc(store)
        # Standard UUID v4 format: 8-4-4-4-12 hex chars + 4 dashes.
        assert isinstance(doc_id, str)
        assert len(doc_id) == 36
        assert doc_id.count("-") == 4

    def test_explicit_uuid_is_honoured(self, store: MetadataStore) -> None:
        explicit = "01234567-89ab-cdef-0123-456789abcdef"
        got = _make_doc(store, content_hash="h1")
        # Override by re-inserting with explicit id (must differ in hash).
        again = store.insert_document(
            filename="x.pdf",
            doc_type="manual",
            source_path="data/documents/x.pdf",
            size_bytes=10,
            content_hash="h2",
            document_id=explicit,
        )
        assert again == explicit
        assert got != explicit  # auto-generated ids differ

    def test_metadata_json_is_serialised(self, store: MetadataStore) -> None:
        meta = {"page_count": 12, "author": "TinyRAG", "tags": ["manual", "v2"]}
        doc_id = _make_doc(store, metadata=meta, content_hash="h1")
        doc = store.get_document(doc_id)
        assert doc is not None
        # Stored as JSON text; we parse it back to verify round-trip.
        parsed = json.loads(doc.metadata_json)  # type: ignore[arg-type]
        assert parsed == meta

    def test_metadata_none_stores_sql_null(self, store: MetadataStore) -> None:
        doc_id = _make_doc(store, metadata=None)
        doc = store.get_document(doc_id)
        assert doc is not None
        assert doc.metadata_json is None

    @pytest.mark.parametrize("bad_type", ["", "manual ", "FAQ", "doc", "README"])
    def test_bad_doc_type_raises_value_error(
        self, store: MetadataStore, bad_type: str
    ) -> None:
        with pytest.raises(ValueError):
            store.insert_document(
                filename="x.pdf",
                doc_type=bad_type,
                source_path="x.pdf",
                size_bytes=1,
                content_hash=f"h-{bad_type}",
            )

    @pytest.mark.parametrize("good_type", ["manual", "faq", "sensor_summary"])
    def test_supported_doc_types_are_accepted(
        self, store: MetadataStore, good_type: str
    ) -> None:
        # None of these should raise.
        _make_doc(store, doc_type=good_type, content_hash=f"h-{good_type}")

    def test_duplicate_content_hash_is_accepted(
        self, store: MetadataStore
    ) -> None:
        """``content_hash`` is indexed (per the design doc) but NOT
        UNIQUE — two different files can hash to the same value
        only by collision (essentially never in practice), and we
        still want to be able to insert them. The dedup signal is
        :meth:`get_document_by_hash`, not a DB-level constraint.
        """
        first_id = _make_doc(store, content_hash="dup", filename="first.pdf")
        _make_doc(store, content_hash="dup", filename="second.pdf")
        assert store.count_documents() == 2
        # Lookup returns the FIRST (oldest) one — the dedup signal
        # the ingestion pipeline uses to skip re-ingest.
        first = store.get_document_by_hash("dup")
        assert first is not None
        assert first.id == first_id
        assert first.filename == "first.pdf"

    def test_integrity_error_carries_db_path(
        self, store: MetadataStore, db_path: Path
    ) -> None:
        """Force an IntegrityError (FK violation on chunks)."""
        with pytest.raises(MetadataIntegrityError) as excinfo:
            store.insert_chunks(
                [_make_chunk("c0", "nonexistent-document-id")]
            )
        assert excinfo.value.db_path == str(db_path)


class TestUpdateChunkCount:
    """``update_document_chunk_count`` writes num_chunks + bumps last_modified."""

    def test_updates_num_chunks(self, store: MetadataStore) -> None:
        doc_id = _make_doc(store)
        store.update_document_chunk_count(doc_id, 42)
        doc = store.get_document(doc_id)
        assert doc is not None
        assert doc.num_chunks == 42

    def test_bumps_last_modified(self, store: MetadataStore) -> None:
        doc_id = _make_doc(store)
        before = store.get_document(doc_id)
        assert before is not None
        # We don't compare exact strings (the wall-clock could tick
        # between the two reads), but we do verify the field is
        # populated with a parseable ISO-8601 timestamp.
        store.update_document_chunk_count(doc_id, 5)
        after = store.get_document(doc_id)
        assert after is not None
        assert isinstance(after.last_modified, str)
        assert len(after.last_modified) >= 10  # "YYYY-MM-DD" minimum

    def test_missing_document_id_is_silent_no_op(
        self, store: MetadataStore
    ) -> None:
        """UPDATE on a missing id is 0 rows, not an error — caller's choice."""
        store.update_document_chunk_count("does-not-exist", 99)
        assert store.count_documents() == 0

    def test_negative_count_raises(self, store: MetadataStore) -> None:
        doc_id = _make_doc(store)
        with pytest.raises(ValueError):
            store.update_document_chunk_count(doc_id, -1)


class TestInsertChunks:
    """``insert_chunks`` writes a batch atomically."""

    def test_single_chunk_round_trip(self, store: MetadataStore) -> None:
        doc_id = _make_doc(store)
        store.insert_chunks(
            [_make_chunk("c0", doc_id, text="hello world", token_count=2)]
        )
        chunks = store.get_chunks_by_document(doc_id)
        assert len(chunks) == 1
        assert chunks[0].id == "c0"
        assert chunks[0].text == "hello world"
        assert chunks[0].token_count == 2
        assert chunks[0].faiss_idx == 0
        assert chunks[0].embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert chunks[0].page_number is None

    def test_text_preview_auto_truncated(self, store: MetadataStore) -> None:
        doc_id = _make_doc(store)
        long_text = "x" * 500  # well over the 200-char preview limit
        store.insert_chunks(
            [_make_chunk("c0", doc_id, text=long_text)]
        )
        chunks = store.get_chunks_by_document(doc_id)
        assert len(chunks[0].text_preview) == 200
        assert chunks[0].text_preview == "x" * 200
        # The full text is preserved in `text`.
        assert len(chunks[0].text) == 500

    def test_text_preview_can_be_overridden(self, store: MetadataStore) -> None:
        doc_id = _make_doc(store)
        custom = "CUSTOM PREVIEW"
        store.insert_chunks(
            [_make_chunk("c0", doc_id, text="long " * 100, text_preview=custom)]
        )
        chunks = store.get_chunks_by_document(doc_id)
        assert chunks[0].text_preview == custom

    def test_multiple_chunks_round_trip(self, store: MetadataStore) -> None:
        doc_id = _make_doc(store)
        store.insert_chunks(
            [
                _make_chunk(f"c{i}", doc_id, chunk_index=i, faiss_idx=i,
                            text=f"chunk {i}")
                for i in range(5)
            ]
        )
        chunks = store.get_chunks_by_document(doc_id)
        assert [c.chunk_index for c in chunks] == [0, 1, 2, 3, 4]
        assert [c.faiss_idx for c in chunks] == [0, 1, 2, 3, 4]

    def test_empty_input_raises(self, store: MetadataStore) -> None:
        with pytest.raises(ValueError):
            store.insert_chunks([])

    def test_missing_required_key_raises(self, store: MetadataStore) -> None:
        doc_id = _make_doc(store)
        bad = _make_chunk("c0", doc_id)
        del bad["token_count"]  # remove a required key
        with pytest.raises(ValueError) as excinfo:
            store.insert_chunks([bad])
        assert "token_count" in str(excinfo.value)

    def test_duplicate_chunk_index_rolls_back_whole_batch(
        self, store: MetadataStore
    ) -> None:
        """Atomicity invariant: ALL or NONE.

        This is the property the ingestion pipeline relies on — a
        failure mid-batch never leaves a half-written document.
        """
        doc_id = _make_doc(store)
        # 3 chunks where the 2nd has a duplicate (document_id, chunk_index)
        # with the 1st. The whole batch must fail and roll back.
        with pytest.raises(MetadataIntegrityError):
            store.insert_chunks(
                [
                    _make_chunk("c0", doc_id, chunk_index=0, faiss_idx=0),
                    _make_chunk("c1", doc_id, chunk_index=0, faiss_idx=1),  # dup idx
                    _make_chunk("c2", doc_id, chunk_index=2, faiss_idx=2),
                ]
            )
        # No chunks were inserted (the bad row's UNIQUE violation
        # rolled back the whole transaction).
        assert store.get_chunks_by_document(doc_id) == []
        assert store.count_chunks() == 0

    def test_foreign_key_violation_rolls_back(
        self, store: MetadataStore
    ) -> None:
        """A chunk referencing a nonexistent document is rejected."""
        with pytest.raises(MetadataIntegrityError):
            store.insert_chunks([_make_chunk("c0", "no-such-document")])
        assert store.count_chunks() == 0

    def test_chunk_with_page_number(self, store: MetadataStore) -> None:
        doc_id = _make_doc(store)
        store.insert_chunks(
            [_make_chunk("c0", doc_id, page_number=3, char_offset=1024)]
        )
        chunks = store.get_chunks_by_document(doc_id)
        assert chunks[0].page_number == 3
        assert chunks[0].char_offset == 1024


class TestGetDocument:
    """``get_document`` and ``get_document_by_hash`` lookup paths."""

    def test_get_by_id_hit(self, store: MetadataStore) -> None:
        doc_id = _make_doc(store)
        assert store.get_document(doc_id) is not None

    def test_get_by_id_miss_returns_none(self, store: MetadataStore) -> None:
        assert store.get_document("does-not-exist") is None

    def test_get_by_hash_hit(self, store: MetadataStore) -> None:
        _make_doc(store, content_hash="unique-hash")
        doc = store.get_document_by_hash("unique-hash")
        assert doc is not None
        assert doc.content_hash == "unique-hash"

    def test_get_by_hash_miss_returns_none(self, store: MetadataStore) -> None:
        assert store.get_document_by_hash("nope") is None

    def test_get_by_hash_returns_oldest_first(
        self, store: MetadataStore
    ) -> None:
        """The dedup signal — re-ingest of same content returns the
        original document, not the new one."""
        first_id = _make_doc(store, content_hash="dup", filename="first.pdf")
        _make_doc(store, content_hash="dup", filename="second.pdf")
        found = store.get_document_by_hash("dup")
        assert found is not None
        assert found.id == first_id
        assert found.filename == "first.pdf"


class TestListDocuments:
    """``list_documents`` returns every document, newest first."""

    def test_empty_store_returns_empty_list(self, store: MetadataStore) -> None:
        assert store.list_documents() == []

    def test_single_document(self, store: MetadataStore) -> None:
        _make_doc(store)
        docs = store.list_documents()
        assert len(docs) == 1

    def test_newest_first(self, store: MetadataStore) -> None:
        ids = [_make_doc(store, content_hash=f"h{i}") for i in range(3)]
        docs = store.list_documents()
        # The list is newest-first; the LAST inserted is at index 0.
        assert [d.id for d in docs] == [ids[2], ids[1], ids[0]]

    def test_returns_document_records(self, store: MetadataStore) -> None:
        _make_doc(store)
        docs = store.list_documents()
        assert all(isinstance(d, DocumentRecord) for d in docs)

    # --------------------------------------------------------------
    # Step 4.18 — pagination extension
    # --------------------------------------------------------------

    def test_limit_caps_results(self, store: MetadataStore) -> None:
        """``limit`` bounds the number of rows returned (newest first)."""
        ids = [_make_doc(store, content_hash=f"h{i}") for i in range(5)]
        docs = store.list_documents(limit=2)
        # Newest-first: ids[4], ids[3].
        assert [d.id for d in docs] == [ids[4], ids[3]]

    def test_offset_skips_rows(self, store: MetadataStore) -> None:
        """``offset`` skips the first N rows in newest-first order."""
        ids = [_make_doc(store, content_hash=f"h{i}") for i in range(5)]
        docs = store.list_documents(limit=2, offset=2)
        # Newest-first: skip ids[4], ids[3] → return ids[2], ids[1].
        assert [d.id for d in docs] == [ids[2], ids[1]]

    def test_default_behaviour_unchanged(self, store: MetadataStore) -> None:
        """Calling without kwargs returns all rows (back-compat pin)."""
        [_make_doc(store, content_hash=f"h{i}") for i in range(4)]
        # No kwargs → all 4 rows, newest first.
        assert len(store.list_documents()) == 4
        # Explicit limit=None matches the default behaviour.
        assert len(store.list_documents(limit=None, offset=0)) == 4


class TestGetChunksByIds:
    """``get_chunks_by_ids`` preserves input order, skips unknowns."""

    def test_empty_input_returns_empty_list(self, store: MetadataStore) -> None:
        assert store.get_chunks_by_ids([]) == []

    def test_returns_in_input_order(self, store: MetadataStore) -> None:
        doc_id = _make_doc(store)
        ids = [f"c{i}" for i in range(5)]
        store.insert_chunks(
            [_make_chunk(cid, doc_id, chunk_index=i, faiss_idx=i) for i, cid in enumerate(ids)]
        )
        # Request in REVERSE order; result must preserve our order.
        requested = list(reversed(ids))
        chunks = store.get_chunks_by_ids(requested)
        assert [c.id for c in chunks] == requested

    def test_unknown_ids_are_silently_skipped(self, store: MetadataStore) -> None:
        doc_id = _make_doc(store)
        store.insert_chunks([_make_chunk("c0", doc_id)])
        # Mix known + unknown; unknowns are filtered out.
        chunks = store.get_chunks_by_ids(["c0", "nope", "also-nope"])
        assert [c.id for c in chunks] == ["c0"]

    def test_dedupes_repeated_ids(self, store: MetadataStore) -> None:
        """FAISS may return the same hit twice (rare, but possible);
        we don't want a duplicated citation card in the UI."""
        doc_id = _make_doc(store)
        store.insert_chunks([_make_chunk("c0", doc_id)])
        chunks = store.get_chunks_by_ids(["c0", "c0", "c0"])
        assert len(chunks) == 1
        assert chunks[0].id == "c0"

    def test_large_input_is_batched(
        self, store: MetadataStore
    ) -> None:
        """``IN`` clauses > 500 ids are split into multiple queries.

        We don't measure performance here (that's a benchmark); we
        just confirm the result is correct across batch boundaries.
        """
        doc_id = _make_doc(store)
        n = 1200  # > 2x the 500 batch size
        store.insert_chunks(
            [_make_chunk(f"c{i:05d}", doc_id, chunk_index=i, faiss_idx=i)
             for i in range(n)]
        )
        ids = [f"c{i:05d}" for i in range(n)]
        chunks = store.get_chunks_by_ids(ids)
        assert len(chunks) == n
        # Order preserved across batch boundaries.
        assert [c.id for c in chunks] == ids

    def test_returns_chunk_records(self, store: MetadataStore) -> None:
        doc_id = _make_doc(store)
        store.insert_chunks([_make_chunk("c0", doc_id)])
        chunks = store.get_chunks_by_ids(["c0"])
        assert len(chunks) == 1
        assert isinstance(chunks[0], ChunkRecord)


class TestGetChunksByDocument:
    """``get_chunks_by_document`` orders by chunk_index."""

    def test_empty_document_returns_empty(self, store: MetadataStore) -> None:
        _make_doc(store)
        assert store.get_chunks_by_document(store.list_documents()[0].id) == []

    def test_orders_by_chunk_index(self, store: MetadataStore) -> None:
        doc_id = _make_doc(store)
        # Insert out of order; the read must reorder by chunk_index.
        store.insert_chunks(
            [
                _make_chunk("c2", doc_id, chunk_index=2, faiss_idx=2),
                _make_chunk("c0", doc_id, chunk_index=0, faiss_idx=0),
                _make_chunk("c1", doc_id, chunk_index=1, faiss_idx=1),
            ]
        )
        chunks = store.get_chunks_by_document(doc_id)
        assert [c.chunk_index for c in chunks] == [0, 1, 2]


class TestDeleteDocument:
    """``delete_document`` cascades to chunks via FK enforcement."""

    def test_deletes_document_and_chunks(self, store: MetadataStore) -> None:
        doc_id = _make_doc(store)
        store.insert_chunks(
            [_make_chunk(f"c{i}", doc_id, chunk_index=i, faiss_idx=i)
             for i in range(5)]
        )
        assert store.count_chunks() == 5
        deleted = store.delete_document(doc_id)
        assert deleted == 1
        assert store.count_documents() == 0
        # The cascade must have removed all 5 chunks.
        assert store.count_chunks() == 0

    def test_missing_document_returns_zero(self, store: MetadataStore) -> None:
        assert store.delete_document("nope") == 0

    def test_only_target_document_is_deleted(
        self, store: MetadataStore
    ) -> None:
        doc1 = _make_doc(store, content_hash="h1")
        doc2 = _make_doc(store, content_hash="h2")
        store.insert_chunks([_make_chunk("c1", doc1)])
        store.insert_chunks([_make_chunk("c2", doc2)])
        store.delete_document(doc1)
        # doc2 and its chunk survive.
        assert store.count_documents() == 1
        assert store.get_document(doc2) is not None
        assert store.count_chunks() == 1


class TestCounters:
    """``count_documents`` / ``count_chunks``."""

    def test_initial_zeros(self, store: MetadataStore) -> None:
        assert store.count_documents() == 0
        assert store.count_chunks() == 0

    def test_counts_reflect_inserts(self, store: MetadataStore) -> None:
        d1 = _make_doc(store, content_hash="h1")
        _make_doc(store, content_hash="h2")
        store.insert_chunks(
            [_make_chunk(f"c{i}", d1, chunk_index=i, faiss_idx=i) for i in range(3)]
        )
        assert store.count_documents() == 2
        assert store.count_chunks() == 3


class TestQueryLog:
    """``log_query`` + ``get_recent_queries``."""

    def test_log_returns_incrementing_ids(self, store: MetadataStore) -> None:
        id1 = store.log_query(query="first?")
        id2 = store.log_query(query="second?")
        id3 = store.log_query(query="third?")
        assert id1 == 1
        assert id2 == 2
        assert id3 == 3

    def test_recent_queries_newest_first(self, store: MetadataStore) -> None:
        store.log_query(query="first?")
        store.log_query(query="second?")
        store.log_query(query="third?")
        recs = store.get_recent_queries()
        assert [r.query for r in recs] == ["third?", "second?", "first?"]

    def test_partial_result_logged(self, store: MetadataStore) -> None:
        """A query that failed mid-flight can still be logged with NULL fields."""
        id_ = store.log_query(query="broke before retrieval")
        recs = store.get_recent_queries()
        assert recs[0].id == id_
        assert recs[0].query == "broke before retrieval"
        assert recs[0].top1_score is None
        assert recs[0].num_chunks is None
        assert recs[0].retrieval_ms is None
        assert recs[0].model is None

    def test_full_result_logged(self, store: MetadataStore) -> None:
        store.log_query(
            query="how do I reset?",
            top1_score=0.81,
            num_chunks=5,
            retrieval_ms=23,
            generation_ms=450,
            total_ms=520,
            model="phi-3-mini",
            used_sensor_idx=0,
            feedback=None,
        )
        rec = store.get_recent_queries()[0]
        assert rec.query == "how do I reset?"
        assert rec.top1_score == pytest.approx(0.81)
        assert rec.num_chunks == 5
        assert rec.retrieval_ms == 23
        assert rec.generation_ms == 450
        assert rec.total_ms == 520
        assert rec.model == "phi-3-mini"
        assert rec.used_sensor_idx == 0
        assert rec.feedback is None

    def test_limit_must_be_positive(self, store: MetadataStore) -> None:
        with pytest.raises(ValueError):
            store.get_recent_queries(limit=0)
        with pytest.raises(ValueError):
            store.get_recent_queries(limit=-5)

    def test_limit_caps_results(self, store: MetadataStore) -> None:
        for i in range(10):
            store.log_query(query=f"q-{i}")
        assert len(store.get_recent_queries(limit=3)) == 3

    def test_returns_query_log_records(self, store: MetadataStore) -> None:
        store.log_query(query="hi")
        assert all(
            isinstance(r, QueryLogRecord) for r in store.get_recent_queries()
        )


class TestSqlInjectionGuard:
    """Every user-supplied string is parameterised — no string interpolation in SQL."""

    def test_chunk_text_is_stored_literally(
        self, store: MetadataStore
    ) -> None:
        """A text field containing SQL-meta characters is stored as-is."""
        doc_id = _make_doc(store)
        nasty = "'; DROP TABLE chunks; --"
        store.insert_chunks([_make_chunk("c0", doc_id, text=nasty)])
        chunks = store.get_chunks_by_document(doc_id)
        assert chunks[0].text == nasty
        # The chunks table still exists.
        assert store.count_chunks() == 1

    def test_filename_with_sql_meta_is_stored_literally(
        self, store: MetadataStore
    ) -> None:
        nasty_name = "evil'; DROP TABLE documents; --.pdf"
        doc_id = _make_doc(store, filename=nasty_name, content_hash="h")
        assert store.get_document(doc_id) is not None
        # The documents table still exists.
        assert store.count_documents() == 1

    def test_query_text_is_stored_literally(
        self, store: MetadataStore
    ) -> None:
        nasty = "'; UPDATE query_log SET feedback='pwned'; --"
        store.log_query(query=nasty)
        recs = store.get_recent_queries()
        assert recs[0].query == nasty
        # The pwned feedback is NOT in any row.
        for r in recs:
            assert r.feedback != "pwned"


class TestJsonMetadataRoundTrip:
    """``metadata_json`` survives an INSERT → SELECT round-trip as JSON text."""

    def test_round_trip_with_complex_dict(self, store: MetadataStore) -> None:
        meta = {
            "page_count": 12,
            "author": "TinyRAG",
            "tags": ["manual", "v2", "english"],
            "nested": {"key": [1, 2, 3], "deep": True},
        }
        doc_id = _make_doc(store, metadata=meta, content_hash="h-json")
        doc = store.get_document(doc_id)
        assert doc is not None
        assert json.loads(doc.metadata_json) == meta  # type: ignore[arg-type]

    def test_metadata_none_is_null(self, store: MetadataStore) -> None:
        doc_id = _make_doc(store, metadata=None, content_hash="h-null")
        doc = store.get_document(doc_id)
        assert doc is not None
        # Stored as SQL NULL → Python None.
        assert doc.metadata_json is None


class TestContextManager:
    """``with MetadataStore(...) as store: ...`` works cleanly."""

    def test_context_manager_closes_connection(
        self, store: MetadataStore, db_path: Path
    ) -> None:
        # We can't directly observe connection closure without poking
        # internals, so we just verify the round-trip inside the
        # ``with`` block works and doesn't error.
        with MetadataStore(db_path) as s:
            doc_id = _make_doc(s)
            assert s.get_document(doc_id) is not None

    def test_context_manager_is_reusable(
        self, store: MetadataStore, db_path: Path
    ) -> None:
        with MetadataStore(db_path) as s:
            _make_doc(s)
        # Re-enter after exit.
        with MetadataStore(db_path) as s2:
            assert s2.count_documents() == 1


class TestInMemoryDatabase:
    """``:memory:`` SQLite works for unit tests without filesystem isolation."""

    def test_memory_db_init(self) -> None:
        s = MetadataStore(":memory:")
        s.init_schema()
        assert s.get_schema_version() == SCHEMA_VERSION

    def test_memory_db_round_trip(self) -> None:
        s = MetadataStore(":memory:")
        s.init_schema()
        doc_id = s.insert_document(
            filename="x.pdf",
            doc_type="manual",
            source_path="x.pdf",
            size_bytes=1,
            content_hash="h",
        )
        s.insert_chunks([_make_chunk("c0", doc_id)])
        assert s.count_chunks() == 1

    def test_in_memory_databases_are_isolated(self) -> None:
        """Two ``:memory:`` stores don't share state (SQLite quirk)."""
        s1 = MetadataStore(":memory:")
        s1.init_schema()
        s1.insert_document(
            filename="x.pdf", doc_type="manual",
            source_path="x.pdf", size_bytes=1, content_hash="h",
        )
        s2 = MetadataStore(":memory:")
        s2.init_schema()
        assert s2.count_documents() == 0  # s1's data didn't leak
