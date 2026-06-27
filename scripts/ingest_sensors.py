#!/usr/bin/env python3
"""Sensor ingestion pipeline — Step 4.15.

The **mirror** of :mod:`scripts.ingest` (Step 4.9) but for the
sensor side of the corpus. Where :mod:`scripts.ingest` turns a
PDF/TXT/MD into ~400-token text chunks that live in the **document**
FAISS index, this script turns the 30-day synthetic sensor CSV
(or a real :class:`SensorSource` in Phase 6) into one **human-readable
summary chunk per (day, sensor_id)** that lives in the **sensor**
FAISS index.

Why a separate script (and not a sub-command of ingest.py)?
----------------------------------------------------------
The two pipelines share the "embed → FAISS → metadata DB" tail
but diverge wildly on the upstream side:

- **Docs** are parsed, then chunked by token count + sentence
  boundary. Many chunks per document; ``doc_type='manual'``.
- **Sensors** are grouped by ``(date, sensor_id)`` and rendered
  into one natural-language summary per group; ``doc_type='sensor_summary'``;
  the FAISS index lives at a *different path* so the retriever
  (Step 4.12) can query the two indices independently.

Splitting the CLIs keeps each one under 300 lines and lets the
two FAISS indices evolve independently (the sensor index uses
the same 384-dim space as the doc index because the embedder is
the same — same model, same dim).

What it does, in order
----------------------
1. **Source.** Pick a :class:`~tinyrag.sensors.base.SensorSource`
   (default :class:`~tinyrag.sensors.simulated.SimulatedCSVSource`
   for the laptop path). Apply an optional ``--since`` filter.
2. **Summarize.** :class:`~tinyrag.core.sensor_summarizer.SensorSummarizer`
   turns the DataFrame into ``list[Chunk]`` — one chunk per
   (date, sensor_id) with a human-readable text body. The
   summarizer is the **chunking step** for the sensor pipeline
   (Step 4.14).
3. **Embed.** :class:`~tinyrag.ingestion.embedder.SentenceTransformerEmbedder`
   (or :class:`~tinyrag.ingestion.embedder.FakeEmbedder` with
   ``--embedder fake``) turns each chunk's ``text`` into a
   384-dim L2-normalised vector.
4. **Persist.**
   - ``documents`` row with ``doc_type='sensor_summary'``,
     ``filename`` = the CSV basename, ``size_bytes`` = the
     CSV's on-disk size, ``content_hash`` = SHA-256 of the
     CSV bytes (used as the dedup signal on re-ingest),
     ``metadata_json`` = the sensor-specific fields
     (``source_label``, ``since`` cutoff, ``num_rows_read``,
     ``num_chunks``, etc.).
   - ``chunks`` rows (one per summary chunk) with the same UUIDs
     that go into the FAISS index.
   - Sensor FAISS index at ``config.retrieval.sensor_index_path``
     (defaults to ``data/vector_store/sensor.faiss``) with the
     same dim + embedding-model metadata as the doc index.

Idempotent re-ingest
--------------------
If the script is run twice against the same CSV (with the same
``--since`` cutoff), the second run replaces the first: it
finds the previous ``documents`` row by ``(filename, doc_type)``,
cascade-deletes its ``chunks`` rows, removes the corresponding
vectors from the FAISS sensor index, and re-ingests. This makes
the script safe to run on a cron / as part of ``make ingest-sensors``
without accumulating stale data.

CLI flags
---------

    --csv PATH                 Path to the sensor CSV (positional).
                               Default: config.sensors.csv_path.
    --config PATH              Path to config.yaml (default ./config.yaml).
    --db-path PATH             Override metadata DB path.
    --index-path PATH          Override sensor FAISS index path.
    --source {simulated|mqtt|real_serial}
                               Sensor source. Default: from config.yaml.
    --since ISO8601            Floor on the sensor timestamp; rows
                               older than this are dropped before
                               summarisation. Default: None (use
                               the source's default_since if set).
    --embedder {real|fake}     Embedder kind. Default: real.
    --force                    Bypass the idempotency check (just
                               append; useful for stress-testing).
    --json                     Print JSON result instead of pretty text.
    --quiet                    Print only the JSON summary on success.

Exit codes
----------

    0   Ingestion succeeded.
    1   Pipeline error (source / summarize / embed / store).
    2   Bad CLI args.
    3   Sensor CSV not found / not readable.

Companion docs
--------------
- ``src/tinyrag/sensors/simulated.py`` — :class:`SimulatedCSVSource`
- ``src/tinyrag/core/sensor_summarizer.py`` — :class:`SensorSummarizer`
- ``src/tinyrag/ingestion/embedder.py`` — :class:`SentenceTransformerEmbedder`
- ``src/tinyrag/storage/metadata.py`` — :class:`MetadataStore`
- ``src/tinyrag/storage/vector_store.py`` — :class:`FAISSStore`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Make ``src/`` importable when this script is run directly without
# ``pip install -e .``. After Phase 4 the project will be installed
# and this block becomes a no-op.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tinyrag.config import Settings, load_settings  # noqa: E402
from tinyrag.core import (  # noqa: E402
    Chunk,
    SensorSummarizer,
    SensorSummarizerEmptyError,
    SensorSummarizerError,
    SensorSummarizerSchemaError,
)
from tinyrag.ingestion import (  # noqa: E402
    EmbeddingError,
    FakeEmbedder,
    SentenceTransformerEmbedder,
)
from tinyrag.sensors.base import (  # noqa: E402
    SensorSource,
    SensorSourceError,
)
from tinyrag.sensors.simulated import SimulatedCSVSource  # noqa: E402
from tinyrag.storage import FAISSStore, MetadataStore  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The doc_type written to ``documents.doc_type`` for every sensor
#: summary. Must be in :data:`tinyrag.storage.metadata.SUPPORTED_DOC_TYPES`.
DOC_TYPE_SENSOR_SUMMARY: str = "sensor_summary"

#: The string used as the ``documents.filename`` for the sensor
#: ingest. Allows the idempotency check
#: (:meth:`MetadataStore.list_documents_by_filename`) to find prior
#: ingests of the same CSV. The basename of the CSV is preferred
#: when available; this constant is the fallback (e.g. when the
#: source isn't a file at all, like an MQTT broker).
DEFAULT_FILENAME: str = "sensor_summary"

#: Sub-key for the sensor-specific metadata blob stored in
#: ``documents.metadata_json``. Centralised so the API layer
#: can introspect sensor rows uniformly.
META_SINCE_KEY: str = "since"
META_SOURCE_LABEL_KEY: str = "source_label"
META_NUM_ROWS_KEY: str = "num_rows_read"
META_NUM_DAYS_KEY: str = "num_days"
META_SENSOR_TYPES_KEY: str = "sensor_types"
META_SENSOR_IDS_KEY: str = "sensor_ids"
META_INGESTED_VIA_KEY: str = "ingested_via"


# ---------------------------------------------------------------------------
# Result type — JSON-serialisable, mirrors IngestionReport shape
# ---------------------------------------------------------------------------


@dataclass
class SensorIngestionReport:
    """Outcome of one :func:`run_ingest_sensors` call.

    Mirrors the shape of :class:`scripts.ingest.IngestionReport`
    where it overlaps (duration fields, ok, error) so the
    downstream JSON consumer (the future ``make ingest-all`` target)
    can normalise them. Adds sensor-specific fields
    (``num_rows_read``, ``num_days``, ``sensor_types``,
    ``since``) on top of the doc-pipeline's fields.

    All ``duration_ms`` fields are wall-clock durations of each
    stage. ``extra`` is a free-form dict for non-fatal warnings
    (e.g. "real_serial source is a stub — raising NotImplementedError"
    on the laptop path).
    """

    ok: bool
    csv: str
    doc_id: str | None
    num_rows_read: int
    num_chunks: int
    num_days: int
    embedding_dimension: int
    embedding_model: str
    sensor_types: list[str]
    sensor_ids: list[str]
    since: str | None
    db_path: str
    index_path: str
    index_size: int
    duration_read_ms: float
    duration_summarize_ms: float
    duration_embed_ms: float
    duration_metadata_ms: float
    duration_vector_ms: float
    duration_save_ms: float
    duration_total_ms: float
    replaced_prior: bool
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (rounds floats to 2 dp)."""
        return {
            "ok": self.ok,
            "csv": self.csv,
            "doc_id": self.doc_id,
            "num_rows_read": self.num_rows_read,
            "num_chunks": self.num_chunks,
            "num_days": self.num_days,
            "embedding_dimension": self.embedding_dimension,
            "embedding_model": self.embedding_model,
            "sensor_types": sorted(self.sensor_types),
            "sensor_ids": sorted(self.sensor_ids),
            "since": self.since,
            "db_path": self.db_path,
            "index_path": self.index_path,
            "index_size": self.index_size,
            "duration_read_ms": round(self.duration_read_ms, 2),
            "duration_summarize_ms": round(self.duration_summarize_ms, 2),
            "duration_embed_ms": round(self.duration_embed_ms, 2),
            "duration_metadata_ms": round(self.duration_metadata_ms, 2),
            "duration_vector_ms": round(self.duration_vector_ms, 2),
            "duration_save_ms": round(self.duration_save_ms, 2),
            "duration_total_ms": round(self.duration_total_ms, 2),
            "replaced_prior": self.replaced_prior,
            "error": self.error,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Source factory + helpers
# ---------------------------------------------------------------------------


def _make_source(
    *,
    source_kind: str,
    csv_path: Path,
    settings: Settings,
) -> SensorSource:
    """Build the configured :class:`SensorSource`.

    For ``simulated`` we return a :class:`SimulatedCSVSource`
    pointing at ``csv_path``. ``real_serial`` and ``mqtt`` are
    *imported but not constructed* in Phase 4 (the concrete
    classes raise :class:`NotImplementedError` on ``read()`` in
    Phase 6 the stubs are filled in). We surface a clear error
    if the user picks one of them so the script doesn't dump a
    confusing stack trace.

    The ``settings`` argument is accepted for API symmetry with
    :func:`_make_embedder` and for future per-source config (e.g.
    Pi-specific GPIO pin numbers in Phase 6). Phase 4 doesn't
    use it.
    """
    del settings  # currently unused; see docstring above
    if source_kind == "simulated":
        return SimulatedCSVSource(
            path=csv_path,
            default_since=None,
        )
    if source_kind == "real_serial":
        # Lazy import — pulling in serial_dht on the laptop would
        # require the (Pi-only) ``RPi.GPIO`` module.
        try:
            from tinyrag.sensors.serial_dht import RealSerialSource  # noqa: F401
        except ImportError as exc:
            raise SensorSourceError(
                f"real_serial source unavailable: {exc}. "
                "RealSerialSource requires Pi-only deps "
                "(libgpiod + adafruit-circuitpython-dht).",
            ) from exc
        raise SensorSourceError(
            "real_serial source is a Phase 6 stub — it raises "
            "NotImplementedError on read(). Run Phase 6 first or "
            "switch to --source simulated.",
        )
    if source_kind == "mqtt":
        raise SensorSourceError(
            "mqtt source is a Phase 6 stub — it raises "
            "NotImplementedError on read(). Run Phase 6 first or "
            "switch to --source simulated.",
        )
    raise ValueError(f"unknown sensor source kind: {source_kind!r}")


#: Default FAISS embedding dimension. Matches the real
#: ``all-MiniLM-L6-v2`` model's output; hard-coded because the
#: config doesn't carry a ``dimension`` field (the embedder
#: itself asserts dimension at load time when configured).
_DEFAULT_EMBEDDING_DIMENSION: int = 384


def _make_embedder(settings: Settings, *, kind: str):
    """Build an :class:`EmbeddingModel` per the ``--embedder`` flag."""
    if kind == "fake":
        # Fake embedder's dimension is hard-coded; the real model's
        # actual dimension is asserted at load time, so it doesn't
        # need to be threaded through here.
        return FakeEmbedder(dimension=_DEFAULT_EMBEDDING_DIMENSION)
    if kind == "real":
        # Construct via the Settings (the real embedder's
        # constructor takes EmbeddingSettings directly).
        return SentenceTransformerEmbedder(
            settings.embedding,
        )
    raise ValueError(f"unknown embedder kind: {kind!r}")


def _sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes (used as content_hash)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(64 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _parse_since(value: str | None) -> datetime | None:
    """Parse a ``--since`` argument (ISO 8601) into a tz-aware datetime.

    Accepts both ``2026-06-15`` (assumed UTC midnight) and the
    fully-qualified ``2026-06-15T12:00:00Z``. Returns ``None`` when
    ``value`` is ``None``.
    """
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        # ``fromisoformat`` accepts the Z-suffix in Python 3.11+; for
        # the common case (laptop runs Python 3.12) we can pass it
        # directly. Fall back to stripping the Z for older builds.
        if raw.endswith("Z"):
            return datetime.fromisoformat(raw[:-1]).replace(tzinfo=UTC)
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError as exc:
        raise ValueError(
            f"could not parse --since value {value!r} as ISO 8601: {exc}",
        ) from exc


def _iso(dt: datetime | None) -> str | None:
    """Render a datetime as an ISO-8601 string (or ``None`` if input is None)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    # ``.isoformat()`` produces "2026-06-15T00:00:00+00:00"; we want
    # the Z-suffixed form for the report JSON (matches the metadata
    # store's ISO format elsewhere).
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Idempotency helpers — clean prior chunks + vectors before re-ingest
# ---------------------------------------------------------------------------


def _clear_prior_ingest(
    *,
    store: MetadataStore,
    faiss: FAISSStore,
    filename: str,
    doc_type: str,
) -> bool:
    """Remove any prior ingest for ``(filename, doc_type)``.

    Returns ``True`` if at least one document row was removed
    (``replaced_prior=True`` in the report), ``False`` otherwise.

    The cleanup walks three layers:

    1. **FAISS sensor index.** Pull the chunk UUIDs out of
       :meth:`MetadataStore.get_chunks_by_document`, then call
       :meth:`FAISSStore.remove_ids` on them. The int→UUID mapping
       in the sidecar is updated atomically with the FAISS removal.
    2. **Metadata DB chunks.** :meth:`MetadataStore.delete_document`
       cascade-deletes via ``ON DELETE CASCADE`` (the chunks
       FK on ``documents.id`` declares cascade).
    3. **Metadata DB documents row.** Same call; the rowcount is
       ``len(prior_docs)``.

    The two stores are not in a single transaction — if the FAISS
    removal succeeds and the SQL DELETE crashes, we leak an
    "orphan vector" (the FAISS slot is gone but the chunks row
    never was). That's a known limitation of the two-store
    architecture; the doc pipeline has the same issue. The
    alternative is a write-ahead log, which is overkill for a
    180-vector sensor index. ``re-ingest is idempotent at the
    file-level`` is the property we promise, not ``crash-safe``.
    """
    prior = store.list_documents_by_filename(filename, doc_type=doc_type)
    if not prior:
        return False

    for doc in prior:
        chunk_records = store.get_chunks_by_document(doc.id)
        chunk_ids = [c.id for c in chunk_records]
        # FAISS first — once the sidecar mapping is gone, the SQL
        # delete is harmless even if it fails.
        if chunk_ids:
            faiss.remove_ids(chunk_ids)
        # Then the SQL cascade — deletes both the chunks (via FK
        # cascade) and the documents row.
        store.delete_document(doc.id)
    return True


# ---------------------------------------------------------------------------
# The ingest run
# ---------------------------------------------------------------------------


def run_ingest_sensors(
    *,
    csv_path: Path,
    settings: Settings,
    source_kind: str,
    since: datetime | None,
    embedder_kind: str,
    db_path_override: str | None,
    index_path_override: str | None,
    force: bool = False,
) -> SensorIngestionReport:
    """Run the sensor ingest pipeline end-to-end. Returns a report.

    Every stage's exception is caught and re-packaged as a failed
    report — the script should never crash with a traceback. The
    stage-level ``try/except`` blocks also accumulate partial
    timings so a failed report still shows *where* the failure
    happened (helpful when something takes 5 s and then dies).
    """
    db_path = (
        Path(db_path_override) if db_path_override else Path(settings.paths.metadata_db)
    )
    index_path = (
        Path(index_path_override)
        if index_path_override
        else Path(settings.retrieval.sensor_index_path)
    )

    timings: dict[str, float] = {}
    extra: dict[str, Any] = {}
    t_total_start = time.monotonic()

    # ---- Stage 0: build the source + verify the file exists --------------
    if source_kind == "simulated" and not csv_path.exists():
        return _failed_report(
            csv=str(csv_path),
            db_path=db_path,
            index_path=index_path,
            error=f"file not found: {csv_path}",
        )

    source: SensorSource
    try:
        source = _make_source(
            source_kind=source_kind,
            csv_path=csv_path,
            settings=settings,
        )
    except Exception as exc:
        return _failed_report(
            csv=str(csv_path),
            db_path=db_path,
            index_path=index_path,
            error=f"source init failed: {type(exc).__name__}: {exc}",
        )

    # The "filename" used for idempotency tracking is the basename
    # of the CSV for the simulated source; the literal string
    # "sensor_summary" otherwise (MQTT/real_serial don't have a
    # single file to key on).
    filename = csv_path.name if source_kind == "simulated" else DEFAULT_FILENAME

    # ---- Stage 1: read ---------------------------------------------------
    t = time.monotonic()
    try:
        df = source.read(since=since)
    except SensorSourceError as exc:
        return _failed_report(
            csv=str(csv_path),
            db_path=db_path,
            index_path=index_path,
            error=f"read failed: {exc}",
            **timings,
        )
    except Exception as exc:
        return _failed_report(
            csv=str(csv_path),
            db_path=db_path,
            index_path=index_path,
            error=f"read failed: {type(exc).__name__}: {exc}",
            **timings,
        )
    timings["read_ms"] = (time.monotonic() - t) * 1000.0

    # ---- Stage 2: summarize ----------------------------------------------
    t = time.monotonic()
    try:
        summarizer = SensorSummarizer(source_label=DOC_TYPE_SENSOR_SUMMARY)
        chunks = summarizer.summarize(df)
    except SensorSummarizerSchemaError as exc:
        return _failed_report(
            csv=str(csv_path),
            db_path=db_path,
            index_path=index_path,
            error=f"summarize failed (schema): {exc}",
            num_rows_read=len(df),
            **timings,
        )
    except SensorSummarizerEmptyError as exc:
        return _failed_report(
            csv=str(csv_path),
            db_path=db_path,
            index_path=index_path,
            error=f"summarize failed (empty): {exc}",
            num_rows_read=len(df),
            **timings,
        )
    except SensorSummarizerError as exc:
        return _failed_report(
            csv=str(csv_path),
            db_path=db_path,
            index_path=index_path,
            error=f"summarize failed: {exc}",
            num_rows_read=len(df),
            **timings,
        )
    except Exception as exc:
        return _failed_report(
            csv=str(csv_path),
            db_path=db_path,
            index_path=index_path,
            error=f"summarize failed: {type(exc).__name__}: {exc}",
            num_rows_read=len(df),
            **timings,
        )
    timings["summarize_ms"] = (time.monotonic() - t) * 1000.0

    if not chunks:
        # Defensive — ``SensorSummarizer.summarize`` raises
        # ``SensorSummarizerEmptyError`` on this, but a future
        # summarizer that returns ``[]`` should still surface
        # the problem rather than silently produce an empty index.
        return _failed_report(
            csv=str(csv_path),
            db_path=db_path,
            index_path=index_path,
            error="summarize produced 0 chunks (no (date, sensor_id) groups)",
            num_rows_read=len(df),
            **timings,
        )

    # ---- Stage 3: embed --------------------------------------------------
    embedder = _make_embedder(settings, kind=embedder_kind)
    embedding_dimension = embedder.dimension
    embedding_model_name = (
        settings.embedding.model_name
        if embedder_kind == "real"
        else f"fake:{settings.embedding.model_name}"
    )
    t = time.monotonic()
    try:
        texts = [c.text for c in chunks]
        vectors = embedder.embed(texts)
    except EmbeddingError as exc:
        return _failed_report(
            csv=str(csv_path),
            db_path=db_path,
            index_path=index_path,
            error=f"embed failed: {exc}",
            num_rows_read=len(df),
            num_chunks=len(chunks),
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model_name,
            **timings,
        )
    except Exception as exc:
        return _failed_report(
            csv=str(csv_path),
            db_path=db_path,
            index_path=index_path,
            error=f"embed failed: {type(exc).__name__}: {exc}",
            num_rows_read=len(df),
            num_chunks=len(chunks),
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model_name,
            **timings,
        )
    timings["embed_ms"] = (time.monotonic() - t) * 1000.0

    if len(vectors) != len(chunks):
        return _failed_report(
            csv=str(csv_path),
            db_path=db_path,
            index_path=index_path,
            error=(
                f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            ),
            num_rows_read=len(df),
            num_chunks=len(chunks),
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model_name,
            **timings,
        )

    # ---- Stage 4: metadata DB (write) ------------------------------------
    t = time.monotonic()
    try:
        store = MetadataStore(db_path)
        store.init_schema()

        # Idempotency: clear any prior ingest of the same CSV
        # *before* we start adding new rows, so we don't trip
        # the ``UNIQUE (document_id, chunk_index)`` constraint
        # mid-batch. ``force=True`` skips the clear — useful for
        # stress testing, but produces a broken index in practice.
        replaced_prior = False
        if not force:
            # Open the FAISS store early so we can clear its slots
            # as part of the same idempotency pass.
            faiss_store = FAISSStore(
                index_path,
                embedding_dimension=embedding_dimension,
                embedding_model=embedding_model_name,
            )
            faiss_store.load()  # no-op if file doesn't exist
            replaced_prior = _clear_prior_ingest(
                store=store,
                faiss=faiss_store,
                filename=filename,
                doc_type=DOC_TYPE_SENSOR_SUMMARY,
            )
        else:
            faiss_store = FAISSStore(
                index_path,
                embedding_dimension=embedding_dimension,
                embedding_model=embedding_model_name,
            )
            faiss_store.load()

        # content_hash: SHA-256 of the CSV bytes. For the
        # non-simulated sources there's no file, so we hash an
        # empty payload — distinct runs of the same source get
        # the same hash (which is fine; the idempotency check is
        # filename-based for those).
        if source_kind == "simulated" and csv_path.exists():
            content_hash = _sha256_file(csv_path)
            size_bytes = csv_path.stat().st_size
        else:
            content_hash = hashlib.sha256(b"").hexdigest()
            size_bytes = 0

        # Sensor-specific fields for the metadata blob. Used by the
        # admin UI (Step 4.22) to render the "Sensor sources" panel
        # and by the eval harness (Phase 5) to verify data drift.
        sensor_types = (
            sorted(df["sensor_type"].dropna().unique().tolist()) if len(df) > 0 else []
        )
        sensor_ids = (
            sorted(df["sensor_id"].dropna().unique().tolist()) if len(df) > 0 else []
        )
        num_days = (
            len({d.isoformat() for d in df["timestamp"].dt.date.unique()})
            if len(df) > 0
            else 0
        )

        doc_metadata = {
            META_SOURCE_LABEL_KEY: DOC_TYPE_SENSOR_SUMMARY,
            META_SINCE_KEY: _iso(since),
            META_NUM_ROWS_KEY: int(len(df)),
            META_NUM_DAYS_KEY: int(num_days),
            META_SENSOR_TYPES_KEY: sensor_types,
            META_SENSOR_IDS_KEY: sensor_ids,
            META_INGESTED_VIA_KEY: "scripts/ingest_sensors.py",
        }

        doc_id = store.insert_document(
            filename=filename,
            doc_type=DOC_TYPE_SENSOR_SUMMARY,
            source_path=str(csv_path) if source_kind == "simulated" else source_kind,
            size_bytes=size_bytes,
            content_hash=content_hash,
            metadata=doc_metadata,
        )

        # Chunk UUIDs — same UUIDs go to FAISS for the int↔UUID
        # lock-step the architecture requires.
        chunk_records = _chunk_records(
            chunks,
            document_id=doc_id,
            embedding_model=embedding_model_name,
        )
        chunk_ids = store.insert_chunks(chunk_records)
    except Exception as exc:
        return _failed_report(
            csv=str(csv_path),
            db_path=db_path,
            index_path=index_path,
            error=f"metadata failed: {type(exc).__name__}: {exc}",
            num_rows_read=len(df),
            num_chunks=len(chunks),
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model_name,
            **timings,
        )
    timings["metadata_ms"] = (time.monotonic() - t) * 1000.0

    # ---- Stage 5: FAISS (write) ------------------------------------------
    t = time.monotonic()
    try:
        faiss_store.add(vectors, chunk_ids)
    except Exception as exc:
        return _failed_report(
            csv=str(csv_path),
            db_path=db_path,
            index_path=index_path,
            error=f"vector store failed: {type(exc).__name__}: {exc}",
            num_rows_read=len(df),
            num_chunks=len(chunks),
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model_name,
            doc_id=doc_id,
            **timings,
        )
    timings["vector_ms"] = (time.monotonic() - t) * 1000.0

    # ---- Stage 6: persist FAISS ------------------------------------------
    t = time.monotonic()
    try:
        faiss_store.save()
    except Exception as exc:
        return _failed_report(
            csv=str(csv_path),
            db_path=db_path,
            index_path=index_path,
            error=f"save failed: {type(exc).__name__}: {exc}",
            num_rows_read=len(df),
            num_chunks=len(chunks),
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model_name,
            doc_id=doc_id,
            **timings,
        )
    timings["save_ms"] = (time.monotonic() - t) * 1000.0

    # ---- Stage 7: update the documents row's chunk count -----------------
    t = time.monotonic()
    try:
        store.update_document_chunk_count(doc_id, len(chunks))
    except Exception as exc:
        # Non-fatal — the chunks are in the DB; only the
        # ``documents.num_chunks`` counter is wrong. Surface as
        # a warning but still mark ok=True (the chunks ARE
        # searchable).
        extra["warning"] = f"update_document_chunk_count failed: {exc}"
    timings["update_ms"] = (time.monotonic() - t) * 1000.0

    timings["total_ms"] = (time.monotonic() - t_total_start) * 1000.0

    return SensorIngestionReport(
        ok=True,
        csv=str(csv_path),
        doc_id=doc_id,
        num_rows_read=int(len(df)),
        num_chunks=int(len(chunks)),
        num_days=int(num_days),
        embedding_dimension=embedding_dimension,
        embedding_model=embedding_model_name,
        sensor_types=sensor_types,
        sensor_ids=sensor_ids,
        since=_iso(since),
        db_path=str(db_path),
        index_path=str(index_path),
        index_size=int(faiss_store.size()),
        duration_read_ms=timings.get("read_ms", 0.0),
        duration_summarize_ms=timings.get("summarize_ms", 0.0),
        duration_embed_ms=timings.get("embed_ms", 0.0),
        duration_metadata_ms=timings.get("metadata_ms", 0.0),
        duration_vector_ms=timings.get("vector_ms", 0.0),
        duration_save_ms=timings.get("save_ms", 0.0),
        duration_total_ms=timings.get("total_ms", 0.0),
        replaced_prior=bool(replaced_prior),
        error=None,
        extra=extra,
    )


def _failed_report(
    *,
    csv: str,
    db_path: Path,
    index_path: Path,
    error: str,
    num_rows_read: int = 0,
    num_chunks: int = 0,
    embedding_dimension: int = 0,
    embedding_model: str = "",
    doc_id: str | None = None,
    **timings: float,
) -> SensorIngestionReport:
    """Construct a failed :class:`SensorIngestionReport`.

    ``timings`` is the (possibly partial) dict of stage durations
    accumulated so far. We render each as 0.0 if not present, so
    the JSON output always has the same shape regardless of where
    the pipeline died.
    """
    return SensorIngestionReport(
        ok=False,
        csv=csv,
        doc_id=doc_id,
        num_rows_read=num_rows_read,
        num_chunks=num_chunks,
        num_days=0,
        embedding_dimension=embedding_dimension,
        embedding_model=embedding_model,
        sensor_types=[],
        sensor_ids=[],
        since=None,
        db_path=str(db_path),
        index_path=str(index_path),
        index_size=0,
        duration_read_ms=timings.get("read_ms", 0.0),
        duration_summarize_ms=timings.get("summarize_ms", 0.0),
        duration_embed_ms=timings.get("embed_ms", 0.0),
        duration_metadata_ms=timings.get("metadata_ms", 0.0),
        duration_vector_ms=timings.get("vector_ms", 0.0),
        duration_save_ms=timings.get("save_ms", 0.0),
        duration_total_ms=timings.get("total_ms", 0.0),
        replaced_prior=False,
        error=error,
    )


# ---------------------------------------------------------------------------
# Stage-level helpers (each is independently testable)
# ---------------------------------------------------------------------------


def _chunk_records(
    chunks: list[Chunk],
    document_id: str,
    embedding_model: str,
) -> list[dict[str, Any]]:
    """Map :class:`Chunk` dataclasses → dicts for :meth:`MetadataStore.insert_chunks`.

    Generates UUID v4 per chunk (the metadata store accepts both
    explicit and auto-UUID; we generate here so the same UUID can
    be passed to ``FAISSStore.add(vectors, ids)`` for the int↔UUID
    lock-step the architecture requires).

    Mirrors :func:`scripts.ingest._chunk_records` but without the
    cross-page renumbering — the sensor summarizer already emits
    chunks in global ordinal order (0..N-1) so the FAISS int ID
    the i-th record gets is exactly the i-th ``chunk_index``.
    """
    records: list[dict[str, Any]] = []
    for c in chunks:
        records.append(
            {
                "id": str(uuid.uuid4()),
                "document_id": document_id,
                "chunk_index": c.chunk_index,
                # ``faiss_idx`` is the INT index in the FAISS index —
                # assigned by FAISSStore.add() in ingestion order
                # (see Step 4.8). We patch it back into the chunk
                # row AFTER FAISS has assigned the int IDs so the
                # DB and the index agree. Placeholder -1 for now.
                "faiss_idx": -1,
                "text": c.text,
                "page_number": c.page,  # always None for sensor summaries
                "char_offset": c.char_offset,  # always 0 for sensor summaries
                "token_count": c.token_count,
                "embedding_model": embedding_model,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _green(s: str) -> str:
    return _c("32", s)


def _red(s: str) -> str:
    return _c("31", s)


def _bold(s: str) -> str:
    return _c("1", s)


def print_human(report: SensorIngestionReport, *, quiet: bool) -> None:
    """Print a friendly summary to stdout (or quiet JSON on success)."""
    if quiet:
        if report.ok:
            print(json.dumps(report.to_dict()))
        else:
            print(f"ERROR: {report.error}", file=sys.stderr)
        return

    print(_bold("==> TinyRAG — Sensor Ingestion Report"))
    print(f"    csv:                 {report.csv}")
    print(f"    doc_id:              {report.doc_id}")
    print("    doc_type:            sensor_summary")
    print(f"    num_rows_read:       {report.num_rows_read}")
    print(f"    num_chunks:          {report.num_chunks}")
    print(f"    num_days:            {report.num_days}")
    print(f"    sensor_types:        {', '.join(report.sensor_types) or '(none)'}")
    print(f"    sensor_ids:          {', '.join(report.sensor_ids) or '(none)'}")
    print(f"    since:               {report.since or '(no filter)'}")
    print(f"    embedding_model:     {report.embedding_model}")
    print(f"    embedding_dimension: {report.embedding_dimension}")
    print(f"    db_path:             {report.db_path}")
    print(f"    index_path:          {report.index_path}")
    print(f"    index_size:          {report.index_size}")
    print(f"    replaced_prior:      {report.replaced_prior}")
    print()
    print("    timings:")
    print(f"      read:      {report.duration_read_ms:>8.2f} ms")
    print(f"      summarize: {report.duration_summarize_ms:>8.2f} ms")
    print(f"      embed:     {report.duration_embed_ms:>8.2f} ms")
    print(f"      metadata:  {report.duration_metadata_ms:>8.2f} ms")
    print(f"      vector:    {report.duration_vector_ms:>8.2f} ms")
    print(f"      save:      {report.duration_save_ms:>8.2f} ms")
    print(f"      TOTAL:     {report.duration_total_ms:>8.2f} ms")
    print()
    if report.ok:
        print(_green("[ OK ]") + " Sensor data ingested — summaries are now searchable.")
        if report.extra.get("warning"):
            print(_c("33", f"      warning: {report.extra['warning']}"))
    else:
        print(_red("[FAIL]") + f" {report.error}")
        sys.exit(1)


def print_json(report: SensorIngestionReport) -> None:
    """Print the result as a single JSON object."""
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Settings helper — load with optional override of config path
# ---------------------------------------------------------------------------


def _load_settings(config_path: str | None) -> Settings:
    """Load the typed :class:`Settings`, optionally from a custom config."""
    if config_path is None:
        return load_settings()
    # ``load_settings`` accepts a positional ``path`` argument —
    # the legacy ``config_path=`` kwarg was renamed in Phase 4 and
    # is no longer recognised (the call would raise TypeError).
    from tinyrag.config import load_settings as _ls

    return _ls(config_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ingest_sensors.py",
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "csv",
        nargs="?",
        default=None,
        help=(
            "Path to the sensor CSV (positional; optional). "
            "Defaults to the value in config.yaml."
        ),
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (default: ./config.yaml).",
    )
    p.add_argument(
        "--db-path",
        default=None,
        help="Override the metadata DB path (default: from config.yaml).",
    )
    p.add_argument(
        "--index-path",
        default=None,
        help="Override the sensor FAISS index path (default: from config.yaml).",
    )
    p.add_argument(
        "--source",
        choices=("simulated", "real_serial", "mqtt"),
        default=None,
        help=(
            "Sensor source. Default: value of config.sensors.source "
            "(usually 'simulated' on the laptop)."
        ),
    )
    p.add_argument(
        "--since",
        default=None,
        help=(
            "ISO-8601 floor on the sensor timestamp. Rows older "
            "than this are dropped before summarisation. "
            "Example: '2026-06-15' or '2026-06-15T00:00:00Z'."
        ),
    )
    p.add_argument(
        "--embedder",
        choices=("real", "fake"),
        default="real",
        help="Which EmbeddingModel to use. Default: real.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Skip the idempotency check (append, don't replace). "
            "Rarely useful — provided for stress testing only."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print JSON result instead of pretty text.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress pretty banner; print only the JSON summary.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    args = _build_parser().parse_args(argv)
    settings = _load_settings(args.config)

    # Resolve the CSV path (CLI > config default).
    csv_path = (
        Path(args.csv)
        if args.csv is not None
        else Path(settings.sensors.csv_path)
    )

    # Resolve the source kind (CLI > config default). The config
    # carries an enum (``SensorSource.SIMULATED`` etc.), not a
    # raw string — ``.value`` gives us the lowercase string the
    # ``_make_source`` factory expects.
    if args.source is not None:
        source_kind = args.source
    else:
        raw = settings.sensors.source
        source_kind = raw.value if hasattr(raw, "value") else str(raw)

    try:
        since = _parse_since(args.since)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = run_ingest_sensors(
        csv_path=csv_path,
        settings=settings,
        source_kind=source_kind,
        since=since,
        embedder_kind=args.embedder,
        db_path_override=args.db_path,
        index_path_override=args.index_path,
        force=args.force,
    )

    if args.json:
        print_json(report)
        return 0 if report.ok else 1
    print_human(report, quiet=args.quiet)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
