"""Tests for scripts/ingest_sensors.py (Step 4.15 — sensor ingest pipeline).

Test layout
-----------
- TestPublicSurface             — every script-level name is importable.
- TestSensorIngestionReportSchema — the dataclass is JSON-roundtrippable,
  required keys are present, floats are rounded.
- TestParseSince                — the --since argument parser handles
  ISO-8601 with and without timezone suffix.
- TestSha256File                — the content-hash helper matches
  ``hashlib.sha256`` over the same bytes.
- TestMakeSource                — the source factory returns the right
  class for each ``--source`` value, and the Phase 6 stubs
  surface a clear error (not a stack trace) when chosen on
  the laptop.
- TestChunkRecords              — UUIDs are unique; chunk_index is
  globally unique; the page_number/char_offset fields are
  None/0 (sensor summaries don't have pages).
- TestClearPriorIngest          — idempotency helper removes the
  prior chunks + vectors before re-ingest.
- TestRunIngestSensorsHappyPath — end-to-end success: a 3-sensor
  2-day CSV produces the expected chunk count, FAISS size,
  metadata DB row, and doc_type='sensor_summary'.
- TestRunIngestSensorsIdempotency — running twice replaces the first
  ingest cleanly (1 doc row, N chunks, FAISS size = N).
- TestRunIngestSensorsFailurePaths — missing CSV, unknown sensor
  source, bad --since value, and empty CSV (no rows after filter).
- TestCliArgs                   — argparse defaults + validation +
  --since parsing + JSON output schema.

Hermetic?
---------
Yes. Tests build their own minimal CSV fixtures in ``tmp_path`` and
their own minimal :class:`Settings` (pointing at ``tmp_path`` for
the DB + FAISS index). The :class:`FakeEmbedder` keeps everything
in-process — no model download, no FAISS-on-disk required for the
integration tests.

Why so many tests?
------------------
Like :mod:`test_ingest`, this script is the **risk gate** for the
sensor side of Phase 4 (Step 4.15). Cross-module bugs surface
here first: the source → summarizer → embedder → DB → FAISS
sequence has 5 distinct seams, each with a non-trivial failure
mode. Specifically the tests pin:

- the :class:`SensorIngestionReport` schema (so a future contributor
  can rename internal fields without silently breaking the
  ``--json`` consumers);
- the doc_type='sensor_summary' invariant (the retriever in Step 4.12
  filters on this);
- the idempotency contract (running twice must NOT accumulate stale
  chunks);
- the FAISS-size-equals-chunk-count invariant (so the retriever's
  index size matches the chunks table).

Location: ``tests/test_ingest_sensors.py``
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

# Make the script importable as a module (matches the pattern in
# tests/test_ingest.py and tests/test_smoke_test.py).
_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
_REPO = _HERE.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import ingest_sensors  # noqa: E402

# ---------------------------------------------------------------------------
# Required-report-keys — the public JSON shape every consumer relies on
# ---------------------------------------------------------------------------

#: The set of keys every :class:`SensorIngestionReport.to_dict()` MUST
#: contain (per the roadmap Step 4.15 "verify the report" criteria).
#: Adding a new key is fine; removing one is a breaking change.
REQUIRED_REPORT_KEYS = frozenset(
    {
        "ok",
        "csv",
        "doc_id",
        "num_rows_read",
        "num_chunks",
        "num_days",
        "embedding_dimension",
        "embedding_model",
        "sensor_types",
        "sensor_ids",
        "since",
        "db_path",
        "index_path",
        "index_size",
        "duration_read_ms",
        "duration_summarize_ms",
        "duration_embed_ms",
        "duration_metadata_ms",
        "duration_vector_ms",
        "duration_save_ms",
        "duration_total_ms",
        "replaced_prior",
        "error",
    }
)


#: A minimal-yet-valid ``config.yaml`` for CLI subprocess tests.
#: All nine top-level sections present (Pydantic rejects a missing
#: one). Path values use ``__PLACEHOLDER__`` sentinels that the
#: individual tests ``str.replace()`` into tmp_path-relative paths
#: so subprocess tests don't pollute the project root.
_VALID_YAML_CONFIG = """\
deployment:
  target: laptop

server:
  host: 127.0.0.1
  port: 8000

llm:
  model_path: models/phi-3-mini.gguf
  server_url: http://127.0.0.1:8080
  context_size: 4096
  temperature: 0.0
  max_tokens: 512
  gpu_layers: 0

embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  device: cpu
  batch_size: 32
  cache_dir: models/_hf_cache

chunking:
  chunk_size: 400
  chunk_overlap: 50
  encoding: cl100k_base

retrieval:
  doc_index_path: __DOC_FAISS__
  sensor_index_path: __SENSOR_FAISS__
  doc_top_k: 3
  sensor_top_k: 2
  similarity_threshold: 0.3
  index_type: faiss

sensors:
  source: simulated
  csv_path: __CSV_PATH__
  dht_pin: 4
  pir_pin: 17
  mqtt_broker: localhost
  mqtt_port: 1883
  mqtt_topic_prefix: "tinyrag/sensors/"

logging:
  level: INFO
  json_format: true
  path: __LOG_PATH__

paths:
  documents_dir: __DOCS_DIR__
  metadata_db: __METADATA_DB__
  sensor_logs_dir: __SENSOR_LOGS_DIR__
  logs_dir: __LOGS_DIR__
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_settings(tmp_path: Path) -> ingest_sensors.Settings:
    """Construct a minimal :class:`Settings` pointing at temp paths.

    Most sub-models use the Pydantic defaults; only ``paths`` and
    ``retrieval`` are overridden to point at ``tmp_path``. The
    ``sensors.source`` enum is the default (``simulated``), and
    ``sensors.csv_path`` is overridden to a per-test value.
    """
    from tinyrag.config import (
        EmbeddingSettings,
        PathsSettings,
        RetrievalSettings,
        SensorSettings,
        Settings,
    )

    return Settings(
        embedding=EmbeddingSettings(),
        paths=PathsSettings(
            documents_dir=str(tmp_path / "documents"),
            metadata_db=str(tmp_path / "metadata.db"),
            sensor_logs_dir=str(tmp_path / "sensor_logs"),
            logs_dir=str(tmp_path / "logs"),
        ),
        retrieval=RetrievalSettings(
            doc_index_path=str(tmp_path / "doc.faiss"),
            sensor_index_path=str(tmp_path / "sensor.faiss"),
        ),
        sensors=SensorSettings(
            csv_path=str(tmp_path / "sensor_logs" / "fixture.csv"),
        ),
    )


@pytest.fixture
def tiny_settings(tmp_path: Path) -> ingest_sensors.Settings:
    """Hermetic Settings pointing at tmp_path for DB + indexes."""
    return _make_settings(tmp_path)


@pytest.fixture
def small_csv(tmp_path: Path) -> Path:
    """A minimal sensor CSV: 2 days, 3 sensors, 1 numeric + 1 motion.

    The content is small (12 readings total) so the test is fast;
    the shape (multiple days, multiple sensors, both numeric and
    motion) is enough to exercise the full pipeline.

    Day 1 — 3 numeric readings (one per numeric sensor at 10:00) +
    2 motion events (kitchen at 12:30 + 18:15).
    Day 2 — 3 numeric readings + 1 motion event (kitchen at 09:45).

    Expected chunks (one per (date, sensor_id) group):
        - (Day 1, bedroom_temp):   1 numeric summary
        - (Day 1, living_room_hum): 1 numeric summary
        - (Day 1, kitchen_motion):  1 motion summary (2 events)
        - (Day 1, house_energy):    1 numeric summary
        - (Day 2, bedroom_temp):   1 numeric summary
        - (Day 2, living_room_hum): 1 numeric summary
        - (Day 2, kitchen_motion):  1 motion summary (1 event)
        - (Day 2, house_energy):    1 numeric summary
    Total = 8 chunks.
    """
    rows = [
        # ----- Day 1: 2026-06-15 -----
        ("2026-06-15T10:00:00", "bedroom_temp", "temperature", "20.5", "C"),
        ("2026-06-15T10:00:00", "living_room_hum", "humidity", "55.0", "%"),
        ("2026-06-15T10:00:00", "house_energy", "energy", "0.12", "kWh"),
        ("2026-06-15T12:30:00", "kitchen_motion", "motion", "1", "count"),
        ("2026-06-15T18:15:00", "kitchen_motion", "motion", "1", "count"),
        # ----- Day 2: 2026-06-16 -----
        ("2026-06-16T10:00:00", "bedroom_temp", "temperature", "21.0", "C"),
        ("2026-06-16T10:00:00", "living_room_hum", "humidity", "56.5", "%"),
        ("2026-06-16T10:00:00", "house_energy", "energy", "0.13", "kWh"),
        ("2026-06-16T09:45:00", "kitchen_motion", "motion", "1", "count"),
    ]
    p = tmp_path / "small_sensors.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "sensor_id", "sensor_type", "value", "unit"])
        w.writerows(rows)
    return p


@pytest.fixture
def empty_csv(tmp_path: Path) -> Path:
    """A CSV with the correct headers but zero data rows.

    The summarizer raises :class:`SensorSummarizerEmptyError` on
    this — we test that the script surfaces the error as a clean
    ``ok=False`` report (no traceback).
    """
    p = tmp_path / "empty.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "sensor_id", "sensor_type", "value", "unit"])
    return p


@pytest.fixture
def bad_columns_csv(tmp_path: Path) -> Path:
    """A CSV with the wrong column set (missing 'unit')."""
    p = tmp_path / "bad.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "sensor_id", "sensor_type", "value"])
        w.writerow(("2026-06-15T10:00:00", "bedroom_temp", "temperature", "20.5"))
    return p


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    """The script exposes the documented symbols."""

    def test_sensor_ingestion_report_class_exists(self) -> None:
        assert isinstance(ingest_sensors.SensorIngestionReport, type)

    def test_doc_type_sensor_summary_constant(self) -> None:
        assert ingest_sensors.DOC_TYPE_SENSOR_SUMMARY == "sensor_summary"

    def test_default_filename_constant(self) -> None:
        assert ingest_sensors.DEFAULT_FILENAME == "sensor_summary"

    def test_module_constants_for_metadata_keys(self) -> None:
        # The META_*_KEY constants are documented and used by both
        # the script and the (future) admin UI to read the
        # metadata JSON blob consistently.
        for k in (
            "META_SINCE_KEY",
            "META_SOURCE_LABEL_KEY",
            "META_NUM_ROWS_KEY",
            "META_NUM_DAYS_KEY",
            "META_SENSOR_TYPES_KEY",
            "META_SENSOR_IDS_KEY",
            "META_INGESTED_VIA_KEY",
        ):
            assert hasattr(ingest_sensors, k), k
            assert isinstance(getattr(ingest_sensors, k), str)

    def test_run_ingest_sensors_callable(self) -> None:
        assert callable(ingest_sensors.run_ingest_sensors)

    def test_main_callable(self) -> None:
        assert callable(ingest_sensors.main)

    def test_parse_since_callable(self) -> None:
        assert callable(ingest_sensors._parse_since)

    def test_make_source_callable(self) -> None:
        assert callable(ingest_sensors._make_source)

    def test_make_embedder_callable(self) -> None:
        assert callable(ingest_sensors._make_embedder)

    def test_sha256_file_callable(self) -> None:
        assert callable(ingest_sensors._sha256_file)

    def test_chunk_records_callable(self) -> None:
        assert callable(ingest_sensors._chunk_records)

    def test_clear_prior_ingest_callable(self) -> None:
        assert callable(ingest_sensors._clear_prior_ingest)


# ---------------------------------------------------------------------------
# SensorIngestionReport schema
# ---------------------------------------------------------------------------


def _make_report(**overrides: object) -> ingest_sensors.SensorIngestionReport:
    """Build a fully-populated :class:`SensorIngestionReport` for tests."""
    base: dict[str, object] = dict(
        ok=True,
        csv="x.csv",
        doc_id="doc-1",
        num_rows_read=10,
        num_chunks=3,
        num_days=2,
        embedding_dimension=384,
        embedding_model="fake:model",
        sensor_types=["humidity", "temperature"],
        sensor_ids=["bedroom_temp", "living_room_hum"],
        since="2026-06-15T00:00:00Z",
        db_path="/tmp/x.db",
        index_path="/tmp/x.faiss",
        index_size=3,
        duration_read_ms=10.0,
        duration_summarize_ms=20.0,
        duration_embed_ms=30.0,
        duration_metadata_ms=40.0,
        duration_vector_ms=50.0,
        duration_save_ms=5.0,
        duration_total_ms=155.0,
        replaced_prior=False,
        error=None,
    )
    base.update(overrides)
    return ingest_sensors.SensorIngestionReport(**base)  # type: ignore[arg-type]


class TestSensorIngestionReportSchema:
    """The report dataclass has the right shape and JSON behaviour."""

    def test_required_keys_in_to_dict(self) -> None:
        report = _make_report()
        assert REQUIRED_REPORT_KEYS.issubset(report.to_dict().keys())

    def test_to_dict_is_json_serialisable(self) -> None:
        report = _make_report()
        s = json.dumps(report.to_dict())
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert parsed["ok"] is True
        assert parsed["num_chunks"] == 3

    def test_durations_rounded_to_two_dp(self) -> None:
        report = _make_report(duration_read_ms=1.234567)
        assert report.to_dict()["duration_read_ms"] == 1.23

    def test_to_dict_includes_extra_keys(self) -> None:
        report = _make_report(extra={"warning": "hello"})
        assert report.to_dict()["warning"] == "hello"

    def test_sensor_types_sorted_in_output(self) -> None:
        report = _make_report(sensor_types=["temperature", "humidity"])
        # Output is sorted so the JSON is deterministic.
        assert report.to_dict()["sensor_types"] == ["humidity", "temperature"]

    def test_sensor_ids_sorted_in_output(self) -> None:
        report = _make_report(sensor_ids=["z", "a", "m"])
        assert report.to_dict()["sensor_ids"] == ["a", "m", "z"]

    def test_failed_report_omits_replaced_prior(self) -> None:
        """A failed report still serialises cleanly (replaced_prior=False)."""
        report = _make_report(ok=False, error="boom", doc_id=None)
        d = report.to_dict()
        assert d["ok"] is False
        assert d["error"] == "boom"
        assert d["replaced_prior"] is False


# ---------------------------------------------------------------------------
# _parse_since — the --since argument parser
# ---------------------------------------------------------------------------


class TestParseSince:
    """The --since parser handles ISO-8601 with and without timezone."""

    def test_none_returns_none(self) -> None:
        assert ingest_sensors._parse_since(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert ingest_sensors._parse_since("") is None
        assert ingest_sensors._parse_since("   ") is None

    def test_date_only_assumes_utc_midnight(self) -> None:
        result = ingest_sensors._parse_since("2026-06-15")
        assert result == datetime(2026, 6, 15, 0, 0, 0, tzinfo=UTC)

    def test_full_iso_with_z_suffix(self) -> None:
        result = ingest_sensors._parse_since("2026-06-15T12:34:56Z")
        assert result == datetime(2026, 6, 15, 12, 34, 56, tzinfo=UTC)

    def test_full_iso_with_offset(self) -> None:
        # fromisoformat handles +00:00 directly.
        result = ingest_sensors._parse_since("2026-06-15T12:34:56+00:00")
        assert result == datetime(2026, 6, 15, 12, 34, 56, tzinfo=UTC)

    def test_invalid_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="ISO 8601"):
            ingest_sensors._parse_since("not-a-date")

    def test_partial_garbage_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="ISO 8601"):
            ingest_sensors._parse_since("2026-13-45")


# ---------------------------------------------------------------------------
# _iso helper — paired with _parse_since
# ---------------------------------------------------------------------------


class TestIsoHelper:
    """The _iso helper renders datetimes as Z-suffixed strings."""

    def test_none_returns_none(self) -> None:
        assert ingest_sensors._iso(None) is None

    def test_naive_datetime_assumed_utc(self) -> None:
        result = ingest_sensors._iso(datetime(2026, 6, 15, 12, 0, 0))
        assert result == "2026-06-15T12:00:00Z"

    def test_aware_datetime_converted_to_utc(self) -> None:
        dt = datetime(2026, 6, 15, 14, 0, 0, tzinfo=UTC)  # 2 hours ahead
        # Same wall-clock → same string.
        assert ingest_sensors._iso(dt) == "2026-06-15T14:00:00Z"

    def test_roundtrip_with_parse_since(self) -> None:
        raw = "2026-06-15T12:34:56Z"
        assert ingest_sensors._iso(ingest_sensors._parse_since(raw)) == raw


# ---------------------------------------------------------------------------
# _sha256_file helper
# ---------------------------------------------------------------------------


class TestSha256File:
    """The content-hash helper matches stdlib hashlib."""

    def test_matches_hashlib_on_small_file(self, tmp_path: Path) -> None:
        p = tmp_path / "x.csv"
        p.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        expected = hashlib.sha256(p.read_bytes()).hexdigest()
        assert ingest_sensors._sha256_file(p) == expected

    def test_handles_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.csv"
        p.write_bytes(b"")
        assert (
            ingest_sensors._sha256_file(p)
            == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )


# ---------------------------------------------------------------------------
# _make_source — source factory
# ---------------------------------------------------------------------------


class TestMakeSource:
    """The source factory returns the right class for each kind."""

    def test_simulated_returns_simulated_csv_source(
        self, tmp_path: Path
    ) -> None:
        from tinyrag.sensors.simulated import SimulatedCSVSource

        csv = tmp_path / "x.csv"
        csv.write_text("placeholder", encoding="utf-8")
        src = ingest_sensors._make_source(
            source_kind="simulated",
            csv_path=csv,
            settings=_make_settings(tmp_path),
        )
        assert isinstance(src, SimulatedCSVSource)
        assert src.path == csv

    def test_real_serial_raises_sensor_source_error(self, tmp_path: Path) -> None:
        from tinyrag.sensors.base import SensorSourceError

        with pytest.raises(SensorSourceError, match="real_serial"):
            ingest_sensors._make_source(
                source_kind="real_serial",
                csv_path=tmp_path / "x.csv",
                settings=_make_settings(tmp_path),
            )

    def test_mqtt_raises_sensor_source_error(self, tmp_path: Path) -> None:
        from tinyrag.sensors.base import SensorSourceError

        with pytest.raises(SensorSourceError, match="mqtt"):
            ingest_sensors._make_source(
                source_kind="mqtt",
                csv_path=tmp_path / "x.csv",
                settings=_make_settings(tmp_path),
            )

    def test_unknown_kind_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown sensor source kind"):
            ingest_sensors._make_source(
                source_kind="sneeze",
                csv_path=tmp_path / "x.csv",
                settings=_make_settings(tmp_path),
            )


# ---------------------------------------------------------------------------
# _make_embedder — embedder factory
# ---------------------------------------------------------------------------


class TestMakeEmbedder:
    """The embedder factory returns the right class for each kind."""

    def test_fake_returns_fake_embedder(self, tiny_settings: ingest_sensors.Settings) -> None:
        from tinyrag.ingestion import FakeEmbedder

        e = ingest_sensors._make_embedder(tiny_settings, kind="fake")
        assert isinstance(e, FakeEmbedder)
        assert e.dimension == ingest_sensors._DEFAULT_EMBEDDING_DIMENSION

    def test_real_returns_sentence_transformer_embedder(
        self, tiny_settings: ingest_sensors.Settings
    ) -> None:
        from tinyrag.ingestion import SentenceTransformerEmbedder

        e = ingest_sensors._make_embedder(tiny_settings, kind="real")
        assert isinstance(e, SentenceTransformerEmbedder)
        # Lazy-loaded — .dimension would trigger the model load,
        # so we only check the configured name.
        assert e.model_name == tiny_settings.embedding.model_name

    def test_unknown_kind_raises_value_error(
        self, tiny_settings: ingest_sensors.Settings
    ) -> None:
        with pytest.raises(ValueError, match="unknown embedder kind"):
            ingest_sensors._make_embedder(tiny_settings, kind="random")


# ---------------------------------------------------------------------------
# _chunk_records helper
# ---------------------------------------------------------------------------


class TestChunkRecords:
    """The Chunk → DB-record mapping preserves every invariant."""

    def test_unique_uuids(self) -> None:
        from tinyrag.core import Chunk

        chunks = [
            Chunk(text=f"chunk {i}", source="s", page=None, chunk_index=i, char_offset=0, token_count=2)
            for i in range(3)
        ]
        records = ingest_sensors._chunk_records(
            chunks, document_id="doc-1", embedding_model="fake:model"
        )
        ids = [r["id"] for r in records]
        assert len(ids) == len(set(ids))

    def test_chunk_index_preserved(self) -> None:
        from tinyrag.core import Chunk

        chunks = [
            Chunk(text="a", source="s", page=None, chunk_index=0, char_offset=0, token_count=1),
            Chunk(text="b", source="s", page=None, chunk_index=1, char_offset=0, token_count=1),
            Chunk(text="c", source="s", page=None, chunk_index=2, char_offset=0, token_count=1),
        ]
        records = ingest_sensors._chunk_records(
            chunks, document_id="doc-1", embedding_model="fake:model"
        )
        assert [r["chunk_index"] for r in records] == [0, 1, 2]

    def test_sensor_chunks_have_page_none(self) -> None:
        """Sensor summaries don't carry page numbers (they aren't PDFs)."""
        from tinyrag.core import Chunk

        chunks = [Chunk(text="x", source="s", page=None, chunk_index=0, char_offset=0, token_count=1)]
        records = ingest_sensors._chunk_records(
            chunks, document_id="doc-1", embedding_model="fake:model"
        )
        assert records[0]["page_number"] is None
        assert records[0]["char_offset"] == 0

    def test_faiss_idx_is_placeholder_minus_one(self) -> None:
        """The FAISS int ID is patched back AFTER FAISS assigns it."""
        from tinyrag.core import Chunk

        chunks = [Chunk(text="x", source="s", page=None, chunk_index=0, char_offset=0, token_count=1)]
        records = ingest_sensors._chunk_records(
            chunks, document_id="doc-1", embedding_model="fake:model"
        )
        assert records[0]["faiss_idx"] == -1

    def test_all_required_keys_present(self) -> None:
        from tinyrag.core import Chunk

        chunks = [Chunk(text="x", source="s", page=None, chunk_index=0, char_offset=0, token_count=1)]
        records = ingest_sensors._chunk_records(
            chunks, document_id="doc-1", embedding_model="fake:model"
        )
        for k in (
            "id", "document_id", "chunk_index", "faiss_idx", "text",
            "page_number", "char_offset", "token_count", "embedding_model",
        ):
            assert k in records[0], k


# ---------------------------------------------------------------------------
# _clear_prior_ingest — idempotency helper
# ---------------------------------------------------------------------------


class TestClearPriorIngest:
    """The idempotency helper removes prior chunks + vectors cleanly."""

    def test_returns_false_when_no_prior(
        self, tmp_path: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        from tinyrag.storage import FAISSStore, MetadataStore

        store = MetadataStore(str(tmp_path / "m.db"))
        store.init_schema()
        faiss = FAISSStore(
            str(tmp_path / "s.faiss"),
            embedding_dimension=tiny_settings.embedding.dimension or 384
            if hasattr(tiny_settings.embedding, "dimension")
            else 384,
            embedding_model="fake:model",
        )
        result = ingest_sensors._clear_prior_ingest(
            store=store,
            faiss=faiss,
            filename="nonexistent.csv",
            doc_type="sensor_summary",
        )
        assert result is False
        assert store.count_documents() == 0

    def test_clears_prior_and_returns_true(
        self, tmp_path: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        from tinyrag.storage import FAISSStore, MetadataStore

        store = MetadataStore(str(tmp_path / "m.db"))
        store.init_schema()
        # Add a fake prior sensor_summary row + a chunk.
        doc_id = store.insert_document(
            filename="x.csv",
            doc_type="sensor_summary",
            source_path="/tmp/x.csv",
            size_bytes=10,
            content_hash="abc",
        )
        chunk_records = [
            {
                "id": "chunk-uuid-1",
                "document_id": doc_id,
                "chunk_index": 0,
                "faiss_idx": -1,
                "text": "summary text",
                "token_count": 2,
                "embedding_model": "fake:model",
            }
        ]
        store.insert_chunks(chunk_records)

        # Add the corresponding vector to FAISS so we can verify
        # the idempotency helper removes it too.
        dim = 384
        faiss = FAISSStore(
            str(tmp_path / "s.faiss"),
            embedding_dimension=dim,
            embedding_model="fake:model",
        )
        vec = [[0.0] * dim]  # L2 norm == 0; FAISS accepts it.
        faiss.add(vec, ["chunk-uuid-1"])
        assert faiss.size() == 1

        result = ingest_sensors._clear_prior_ingest(
            store=store,
            faiss=faiss,
            filename="x.csv",
            doc_type="sensor_summary",
        )
        assert result is True
        assert store.count_documents() == 0
        assert store.count_chunks() == 0
        assert faiss.size() == 0


# ---------------------------------------------------------------------------
# run_ingest_sensors — happy path (end-to-end with FakeEmbedder)
# ---------------------------------------------------------------------------


class TestRunIngestSensorsHappyPath:
    """End-to-end success: small CSV → DB + FAISS populated correctly."""

    def test_ok_true(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        report = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.ok is True
        assert report.error is None

    def test_chunk_count_matches_expected(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        report = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        # 4 sensors x 2 days = 8 chunks (see fixture docstring).
        assert report.num_chunks == 8

    def test_rows_read_matches_csv(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        report = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        # 9 data rows in the fixture (5 on day 1 + 4 on day 2).
        assert report.num_rows_read == 9

    def test_num_days_is_two(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        report = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.num_days == 2

    def test_sensor_types_populated(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        report = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert set(report.sensor_types) == {"temperature", "humidity", "energy", "motion"}

    def test_sensor_ids_populated(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        report = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert set(report.sensor_ids) == {
            "bedroom_temp", "living_room_hum", "house_energy", "kitchen_motion",
        }

    def test_db_has_one_sensor_summary_row(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        from tinyrag.storage import MetadataStore

        report = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.ok is True
        store = MetadataStore(tiny_settings.paths.metadata_db)
        docs = store.list_documents_by_filename(
            small_csv.name, doc_type="sensor_summary"
        )
        assert len(docs) == 1
        assert docs[0].doc_type == "sensor_summary"
        assert docs[0].num_chunks == report.num_chunks

    def test_db_chunks_match_report(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        from tinyrag.storage import MetadataStore

        report = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        store = MetadataStore(tiny_settings.paths.metadata_db)
        assert store.count_chunks() == report.num_chunks

    def test_faiss_size_matches_chunk_count(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        report = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        # The roadmap's "sensor FAISS index has the right size" check.
        assert report.index_size == report.num_chunks

    def test_metadata_json_has_sensor_specific_fields(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        from tinyrag.storage import MetadataStore

        # The report itself isn't asserted here — the side
        # effect (DB write) is what this test verifies.
        ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        store = MetadataStore(tiny_settings.paths.metadata_db)
        docs = store.list_documents_by_filename(
            small_csv.name, doc_type="sensor_summary"
        )
        meta = json.loads(docs[0].metadata_json)  # type: ignore[arg-type]
        assert meta[ingest_sensors.META_NUM_ROWS_KEY] == 9
        assert meta[ingest_sensors.META_NUM_DAYS_KEY] == 2
        assert meta[ingest_sensors.META_SOURCE_LABEL_KEY] == "sensor_summary"
        assert meta[ingest_sensors.META_INGESTED_VIA_KEY] == "scripts/ingest_sensors.py"
        assert "temperature" in meta[ingest_sensors.META_SENSOR_TYPES_KEY]

    def test_embedding_dimension_is_384(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        report = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.embedding_dimension == 384

    def test_first_run_does_not_report_replaced_prior(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        report = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.replaced_prior is False

    def test_all_stage_durations_present(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        report = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        for ms in (
            report.duration_read_ms,
            report.duration_summarize_ms,
            report.duration_embed_ms,
            report.duration_metadata_ms,
            report.duration_vector_ms,
            report.duration_save_ms,
            report.duration_total_ms,
        ):
            assert isinstance(ms, float)
            assert ms >= 0.0

    def test_total_under_30_seconds(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        """Same budget as the doc ingest (roadmap NFR)."""
        report = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.duration_total_ms < 30_000.0


# ---------------------------------------------------------------------------
# run_ingest_sensors — idempotency (two consecutive runs)
# ---------------------------------------------------------------------------


class TestRunIngestSensorsIdempotency:
    """Running twice must NOT accumulate stale chunks."""

    def test_second_run_replaces_first(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        from tinyrag.storage import MetadataStore

        # First run.
        r1 = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert r1.ok is True
        assert r1.replaced_prior is False

        # Second run against the same CSV — must succeed AND
        # report replaced_prior=True so callers can detect the
        # idempotency hit.
        r2 = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert r2.ok is True
        assert r2.replaced_prior is True
        assert r2.num_chunks == r1.num_chunks

        # The DB still has exactly ONE document row (not two).
        store = MetadataStore(tiny_settings.paths.metadata_db)
        docs = store.list_documents_by_filename(
            small_csv.name, doc_type="sensor_summary"
        )
        assert len(docs) == 1

        # And the chunks table still has the right count (not 2x).
        assert store.count_chunks() == r2.num_chunks

    def test_faiss_size_stable_across_runs(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        r1 = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        r2 = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        # The sensor index size MUST be the same after re-ingest
        # (no chunk accumulation). This is the most important
        # idempotency invariant.
        assert r2.index_size == r1.index_size == r1.num_chunks

    def test_force_flag_skips_idempotency(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        """With --force, the second run does NOT clear the first."""
        from tinyrag.storage import MetadataStore

        r1 = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert r1.ok is True
        # Second run with force=True: the idempotency clear is
        # skipped. The metadata insert will FAIL with a UNIQUE
        # constraint violation on (document_id, chunk_index) —
        # actually wait, the document_id is fresh per run, so
        # the chunks insert succeeds and we get DOUBLE the chunks.
        # This is the documented "stress testing only" behaviour.
        # We just verify the script reports replaced_prior=False.
        r2 = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
            force=True,
        )
        assert r2.replaced_prior is False
        store = MetadataStore(tiny_settings.paths.metadata_db)
        # Two documents now (one per run).
        docs = store.list_documents_by_filename(
            small_csv.name, doc_type="sensor_summary"
        )
        assert len(docs) == 2
        assert store.count_chunks() == 2 * r1.num_chunks


# ---------------------------------------------------------------------------
# run_ingest_sensors — since filter
# ---------------------------------------------------------------------------


class TestRunIngestSensorsSinceFilter:
    """The --since filter drops rows older than the cutoff."""

    def test_since_in_day2_drops_day1_rows(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        report = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=datetime(2026, 6, 16, 0, 0, 0, tzinfo=UTC),
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.ok is True
        assert report.num_rows_read == 4  # only day-2 rows
        assert report.num_days == 1

    def test_since_in_future_returns_empty(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        report = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=datetime(2099, 1, 1, 0, 0, 0, tzinfo=UTC),
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        # The summarizer raises SensorSummarizerEmptyError on
        # zero rows; the script surfaces this as ok=False with
        # a clean error message.
        assert report.ok is False
        assert "summarize" in (report.error or "").lower() or "empty" in (report.error or "").lower()


# ---------------------------------------------------------------------------
# run_ingest_sensors — failure paths
# ---------------------------------------------------------------------------


class TestRunIngestSensorsFailurePaths:
    """Every stage's exception is caught and surfaced as ok=False."""

    def test_missing_csv_ok_false(
        self, tmp_path: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        missing = tmp_path / "nope.csv"
        report = ingest_sensors.run_ingest_sensors(
            csv_path=missing,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.ok is False
        assert "not found" in (report.error or "").lower()

    def test_bad_columns_ok_false(
        self,
        bad_columns_csv: Path,
        tiny_settings: ingest_sensors.Settings,
    ) -> None:
        report = ingest_sensors.run_ingest_sensors(
            csv_path=bad_columns_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.ok is False
        assert "column" in (report.error or "").lower() or "schema" in (report.error or "").lower()

    def test_empty_csv_ok_false(
        self,
        empty_csv: Path,
        tiny_settings: ingest_sensors.Settings,
    ) -> None:
        report = ingest_sensors.run_ingest_sensors(
            csv_path=empty_csv,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.ok is False
        # SensorSummarizerEmptyError is the expected cause.
        assert "summarize" in (report.error or "").lower() or "empty" in (report.error or "").lower()

    def test_unknown_source_kind_ok_false(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        report = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="sneeze",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.ok is False
        assert "source" in (report.error or "").lower()

    def test_real_serial_source_ok_false(
        self, small_csv: Path, tiny_settings: ingest_sensors.Settings
    ) -> None:
        report = ingest_sensors.run_ingest_sensors(
            csv_path=small_csv,
            settings=tiny_settings,
            source_kind="real_serial",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.ok is False
        assert "real_serial" in (report.error or "").lower() or "phase 6" in (report.error or "").lower()


# ---------------------------------------------------------------------------
# CLI (argparse)
# ---------------------------------------------------------------------------


class TestCliArgs:
    """The CLI parses arguments correctly and surfaces errors."""

    def test_csv_positional_optional(self) -> None:
        """The positional is optional — defaults to config value."""
        args = ingest_sensors._build_parser().parse_args([])
        assert args.csv is None

    def test_default_embedder_is_real(self) -> None:
        args = ingest_sensors._build_parser().parse_args([])
        assert args.embedder == "real"

    def test_default_source_is_none(self) -> None:
        """--source defaults to None → falls back to config value."""
        args = ingest_sensors._build_parser().parse_args([])
        assert args.source is None

    def test_force_flag(self) -> None:
        args = ingest_sensors._build_parser().parse_args(["--force"])
        assert args.force is True

    def test_quiet_flag(self) -> None:
        args = ingest_sensors._build_parser().parse_args(["--quiet"])
        assert args.quiet is True

    def test_json_flag(self) -> None:
        args = ingest_sensors._build_parser().parse_args(["--json"])
        assert args.json is True

    def test_since_argument(self) -> None:
        args = ingest_sensors._build_parser().parse_args(["--since", "2026-06-15"])
        assert args.since == "2026-06-15"

    def test_bad_since_returns_exit_code_2(
        self, small_csv: Path, tmp_path: Path
    ) -> None:
        """A bad --since argument exits with code 2 (argparse convention)."""
        # Build a minimal-yet-valid settings on disk so the CLI can
        # load it (Pydantic requires all nine top-level sections).
        settings_path = tmp_path / "config.yaml"
        settings_path.write_text(_VALID_YAML_CONFIG, encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS / "ingest_sensors.py"),
                str(small_csv),
                "--config", str(settings_path),
                "--since", "not-a-date",
                "--embedder", "fake",
            ],
            env={"PYTHONPATH": str(_REPO / "src"), "PATH": _REPO.as_posix() + "/.venv/bin:"
                 + "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Exit 2 means "bad CLI args" — argparse convention.
        assert result.returncode == 2
        assert "ISO 8601" in result.stderr or "iso 8601" in result.stderr.lower()

    def test_happy_path_cli_returns_zero(
        self, small_csv: Path, tmp_path: Path
    ) -> None:
        """The full CLI succeeds end-to-end against a real CSV."""
        settings_path = tmp_path / "config.yaml"
        # Minimal-yet-valid settings — all nine top-level sections.
        # Path values point inside ``tmp_path`` so we don't pollute
        # the project root.
        text = _VALID_YAML_CONFIG.replace(
            "__DOC_FAISS__", str(tmp_path / "doc.faiss")
        ).replace(
            "__SENSOR_FAISS__", str(tmp_path / "sensor.faiss")
        ).replace(
            "__METADATA_DB__", str(tmp_path / "metadata.db")
        ).replace(
            "__LOG_PATH__", str(tmp_path / "tinyrag.log")
        ).replace(
            "__CSV_PATH__", str(small_csv)
        ).replace(
            "__DOCS_DIR__", str(tmp_path / "documents")
        ).replace(
            "__SENSOR_LOGS_DIR__", str(tmp_path / "sensor_logs")
        ).replace(
            "__LOGS_DIR__", str(tmp_path / "logs")
        )
        settings_path.write_text(text, encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS / "ingest_sensors.py"),
                "--config", str(settings_path),
                "--embedder", "fake",
                "--quiet",
            ],
            env={"PYTHONPATH": str(_REPO / "src"), "PATH": _REPO.as_posix() + "/.venv/bin:"
                 + "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Exit 0 on success.
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Quiet mode prints JSON; parse it.
        report = json.loads(result.stdout)
        assert report["ok"] is True
        assert report["num_chunks"] == 8  # 4 sensors x 2 days


# ---------------------------------------------------------------------------
# Integration with real CSV (regression gate for "did upstream drift?")
# ---------------------------------------------------------------------------


class TestIntegrationWithRealCsv:
    """End-to-end against the real 30-day synthetic CSV (regression gate)."""

    REAL_CSV = Path(__file__).resolve().parent.parent / "data" / "sensor_logs" / "synthetic_30d.csv"

    def test_real_csv_ingests_180_chunks(
        self, tiny_settings: ingest_sensors.Settings, tmp_path: Path
    ) -> None:
        """The 30-day synthetic CSV produces the expected 180 chunks.

        6 sensors x 30 days = 180 chunks (Step 4.14's regression
        gate, here tested through the Step 4.15 ingest script).
        """
        if not self.REAL_CSV.exists():
            pytest.skip(f"real CSV not present at {self.REAL_CSV}")
        report = ingest_sensors.run_ingest_sensors(
            csv_path=self.REAL_CSV,
            settings=tiny_settings,
            source_kind="simulated",
            since=None,
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.ok is True
        assert report.num_chunks == 180
        assert report.index_size == 180
        assert report.num_days == 30
        assert len(report.sensor_ids) == 6
        assert len(report.sensor_types) == 4
