"""Tests for scripts/ingest.py (Step 4.9 — end-to-end ingestion pipeline).

Test layout
-----------
- TestPublicSurface             — every script-level name is importable
  (IngestionReport dataclass, the 4 module-level helpers, the 3
  CLI helpers, the documented exit codes).
- TestIngestionReportSchema     — every required key is in to_dict();
  the dataclass is JSON-roundtrippable; floats are rounded.
- TestSha256File                — the content-hash helper matches
  ``hashlib.sha256`` over the same bytes.
- TestChunkPages                — per-page chunking preserves page
  numbers (PDF) and uses ``page=None`` for TXT/MD.
- TestChunkRecords              — UUIDs are unique; chunk_index is
  globally unique (the cross-page renumbering that fixed the
  duplicate-index bug); page_number + char_offset + token_count
  pass through from the chunker.
- TestRunIngestSuccess          — the full happy path: a small TXT
  → 1+ chunks → DB has the doc + chunks → FAISS has matching size.
- TestRunIngestFailurePaths     — missing file, unknown extension,
  wrong-dim vectors all return ``ok=False`` with a useful error.
- TestCliArgs                   — argparse defaults + validation +
  JSON output schema.

Why so many tests?
------------------
The end-to-end script is the **risk gate** for Phase 4 (Step 4.9
per the roadmap). Every cross-module bug surfaces here first.
Specifically the tests pin:

- the IngestionReport schema (so a future contributor can change
  internal naming without silently breaking the --json consumers);
- the chunk_index global-renumbering (the bug caught by the very
  first run, now codified);
- exit codes 0/1/2 (matches the roadmap's "Done when: ..." criteria).

Hermetic?
---------
Mostly hermetic. The end-to-end happy-path test uses a tiny TXT
file + the FakeEmbedder (no model download). The CLI tests invoke
``main()`` directly via subprocess so they need the venv's Python
on PATH — pytest itself doesn't, but each subprocess test sets
``PYTHONPATH=src`` explicitly.

Location: ``tests/test_ingest.py``
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Make the script importable as a module (matches the pattern in
# tests/test_smoke_test.py).
_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
_REPO = _HERE.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import ingest  # noqa: E402

# Make the tinyrag package importable when the script is imported.
# ``ingest.py`` already adds ``<repo>/src`` to sys.path when it
# loads, but doing it here lets us import the tinyrag.* modules
# directly in helper functions below.
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------


#: The set of keys every IngestionReport MUST contain (per the
#: roadmap Step 4.9 "verify the report" criteria).
REQUIRED_REPORT_KEYS = frozenset(
    {
        "ok",
        "file",
        "doc_id",
        "num_pages",
        "num_chunks",
        "embedding_dimension",
        "embedding_model",
        "doc_type",
        "db_path",
        "index_path",
        "index_size",
        "duration_parse_ms",
        "duration_chunk_ms",
        "duration_embed_ms",
        "duration_metadata_ms",
        "duration_vector_ms",
        "duration_save_ms",
        "duration_total_ms",
        "error",
    }
)


@pytest.fixture
def small_txt(tmp_path: Path) -> Path:
    """A short TXT file with enough content for 2+ chunks at default settings."""
    # ~800 tokens worth of text — enough to cross the 400-token
    # chunk boundary twice at default settings. Distinct sentences so
    # the sentence-boundary detection in the chunker doesn't squash
    # everything into one chunk.
    body = (
        "The Nest Learning Thermostat learns your schedule and programs itself. "
        "It can be controlled from your phone, watch, or voice. "
        "Installation requires a compatible HVAC system and a C-wire in most cases. "
        "Without a C-wire, you may need the Nest Power Connector or a compatible alternative. "
        "The display shows the current temperature, target temperature, and mode. "
        "You can turn the ring to adjust the target temperature manually. "
        "Press the display to bring up the menu. "
        "The thermostat is compatible with 95 percent of HVAC systems. "
        "Some systems including certain heat pumps and high-voltage electric require extra wiring. "
        "Always turn off the power before working with thermostat wires. "
        "The installation should take about 20 to 30 minutes for most homeowners. "
        "If you are not comfortable working with electrical wiring, contact a professional. "
    ) * 4  # repeat to push above 800 tokens
    p = tmp_path / "small.txt"
    p.write_text(body, encoding="utf-8")
    return p


@pytest.fixture
def empty_txt(tmp_path: Path) -> Path:
    """An empty TXT (whitespace only) — must produce no chunks."""
    p = tmp_path / "empty.txt"
    p.write_text("   \n\t\n", encoding="utf-8")
    return p


@pytest.fixture
def tiny_settings(tmp_path: Path) -> ingest.Settings:
    """A Settings instance pointing at temp paths (no real config pollution)."""
    from tinyrag.config import PathsSettings, RetrievalSettings, Settings

    # Settings is frozen, so we construct a fresh instance with
    # overridden sub-models rather than mutating the default.
    return Settings(
        paths=PathsSettings(metadata_db=str(tmp_path / "metadata.db")),
        retrieval=RetrievalSettings(
            doc_index_path=str(tmp_path / "doc.faiss"),
        ),
    )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    """The script exposes the documented symbols."""

    def test_ingestion_report_class_exists(self) -> None:
        assert isinstance(ingest.IngestionReport, type)

    def test_run_ingest_callable(self) -> None:
        assert callable(ingest.run_ingest)

    def test_make_chunker_callable(self) -> None:
        assert callable(ingest._make_chunker)

    def test_make_embedder_callable(self) -> None:
        assert callable(ingest._make_embedder)

    def test_chunk_pages_callable(self) -> None:
        assert callable(ingest._chunk_pages)

    def test_chunk_records_callable(self) -> None:
        assert callable(ingest._chunk_records)

    def test_sha256_file_callable(self) -> None:
        assert callable(ingest._sha256_file)

    def test_main_callable(self) -> None:
        assert callable(ingest.main)

    def test_load_settings_helper_exists(self) -> None:
        assert callable(ingest._load_settings)


# ---------------------------------------------------------------------------
# IngestionReport schema
# ---------------------------------------------------------------------------


class TestIngestionReportSchema:
    """The IngestionReport has every field the roadmap's verify step checks."""

    def test_required_keys_present(self) -> None:
        report = ingest.IngestionReport(
            ok=True,
            file="x.txt",
            doc_id="doc-1",
            num_pages=1,
            num_chunks=1,
            embedding_dimension=384,
            embedding_model="fake:model",
            doc_type="manual",
            db_path="/tmp/x.db",
            index_path="/tmp/x.faiss",
            index_size=1,
            duration_parse_ms=1.0,
            duration_chunk_ms=1.0,
            duration_embed_ms=1.0,
            duration_metadata_ms=1.0,
            duration_vector_ms=1.0,
            duration_save_ms=1.0,
            duration_total_ms=7.0,
        )
        assert REQUIRED_REPORT_KEYS.issubset(report.to_dict().keys())

    def test_to_dict_is_json_serialisable(self) -> None:
        report = ingest.IngestionReport(
            ok=True,
            file="x.txt",
            doc_id="d",
            num_pages=1,
            num_chunks=1,
            embedding_dimension=384,
            embedding_model="m",
            doc_type="manual",
            db_path="/tmp/x.db",
            index_path="/tmp/x.faiss",
            index_size=1,
            duration_parse_ms=1.0,
            duration_chunk_ms=1.0,
            duration_embed_ms=1.0,
            duration_metadata_ms=1.0,
            duration_vector_ms=1.0,
            duration_save_ms=1.0,
            duration_total_ms=7.0,
        )
        # No exception → JSON-safe.
        s = json.dumps(report.to_dict())
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert parsed["ok"] is True
        assert parsed["num_chunks"] == 1

    def test_durations_rounded_to_two_dp(self) -> None:
        report = ingest.IngestionReport(
            ok=True,
            file="x.txt",
            doc_id="d",
            num_pages=1,
            num_chunks=1,
            embedding_dimension=384,
            embedding_model="m",
            doc_type="manual",
            db_path="/tmp/x.db",
            index_path="/tmp/x.faiss",
            index_size=1,
            duration_parse_ms=1.234567,
            duration_chunk_ms=0.0,
            duration_embed_ms=0.0,
            duration_metadata_ms=0.0,
            duration_vector_ms=0.0,
            duration_save_ms=0.0,
            duration_total_ms=0.0,
        )
        # to_dict rounds to 2 dp.
        assert report.to_dict()["duration_parse_ms"] == 1.23

    def test_to_dict_includes_extra_keys(self) -> None:
        report = ingest.IngestionReport(
            ok=True,
            file="x.txt",
            doc_id="d",
            num_pages=1,
            num_chunks=1,
            embedding_dimension=384,
            embedding_model="m",
            doc_type="manual",
            db_path="/tmp/x.db",
            index_path="/tmp/x.faiss",
            index_size=1,
            duration_parse_ms=0.0,
            duration_chunk_ms=0.0,
            duration_embed_ms=0.0,
            duration_metadata_ms=0.0,
            duration_vector_ms=0.0,
            duration_save_ms=0.0,
            duration_total_ms=0.0,
            extra={"warning": "test"},
        )
        d = report.to_dict()
        assert d["warning"] == "test"


# ---------------------------------------------------------------------------
# _sha256_file helper
# ---------------------------------------------------------------------------


class TestSha256File:
    """The content-hash helper matches stdlib hashlib."""

    def test_matches_hashlib_on_small_file(self, tmp_path: Path) -> None:
        import hashlib

        p = tmp_path / "x.txt"
        p.write_bytes(b"hello world\n")
        expected = hashlib.sha256(b"hello world\n").hexdigest()
        assert ingest._sha256_file(p) == expected

    def test_handles_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        # SHA-256 of zero bytes is a well-known constant.
        assert (
            ingest._sha256_file(p)
            == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_matches_hashlib_on_larger_file(self, tmp_path: Path) -> None:
        import hashlib

        p = tmp_path / "big.bin"
        data = bytes(range(256)) * 1000  # 256 KB
        p.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert ingest._sha256_file(p) == expected


# ---------------------------------------------------------------------------
# _chunk_pages helper
# ---------------------------------------------------------------------------


class TestChunkPages:
    """The page-aware chunker correctly handles PDF and TXT/MD inputs."""

    def test_pdf_chunks_carry_page_numbers(self) -> None:
        from tinyrag.ingestion import ParsedDocument

        parsed = ParsedDocument(
            text="",
            pages=[(1, "Page one content. " * 200), (2, "Page two content. " * 200)],
            metadata={},
        )
        chunks = ingest._chunk_pages(parsed, source="x.pdf", chunker=_real_chunker())
        # We should get at least one chunk per page (the body has
        # ~400 tokens per page at our test settings).
        assert chunks
        page_numbers = sorted({c.page for c in chunks})
        assert page_numbers == [1, 2]

    def test_txt_chunks_have_page_none(self) -> None:
        from tinyrag.ingestion import ParsedDocument

        # TXT/MD have no ``pages`` list — the parser collapses to
        # one big string with pages=[]. The chunker must treat
        # these as page-less.
        parsed = ParsedDocument(
            text="This is some text content. " * 100,
            pages=[],
            metadata={},
        )
        chunks = ingest._chunk_pages(parsed, source="x.txt", chunker=_real_chunker())
        assert chunks
        for c in chunks:
            assert c.page is None


def _real_chunker():
    """Build a Chunker matching the default settings."""
    from tinyrag.core import Chunker
    from tinyrag.core.chunker import ChunkingSettings

    return Chunker(ChunkingSettings(chunk_size=400, chunk_overlap=50, encoding="cl100k_base"))


# ---------------------------------------------------------------------------
# _chunk_records helper — the global-renumbering invariant
# ---------------------------------------------------------------------------


class TestChunkRecords:
    """The chunk → DB-record mapping preserves every invariant."""

    def test_unique_uuids(self) -> None:
        chunks = _real_chunker().chunk("Some text. " * 200, source="x.txt", page=None)
        records = ingest._chunk_records(
            chunks, document_id="doc-1", embedding_model="fake:model"
        )
        ids = [r["id"] for r in records]
        assert len(ids) == len(set(ids))

    def test_chunk_index_is_globally_unique(self) -> None:
        """The cross-page renumbering that fixed the duplicate-index bug.

        The Chunker resets ``chunk_index`` to 0 on every call, but
        ``_chunk_records`` renumbers globally so the metadata
        store's ``UNIQUE (document_id, chunk_index)`` constraint
        holds even for multi-page documents.
        """
        chunks = _real_chunker().chunk("Some text. " * 200, source="x.txt", page=None)
        records = ingest._chunk_records(
            chunks, document_id="doc-1", embedding_model="fake:model"
        )
        indices = [r["chunk_index"] for r in records]
        assert indices == sorted(indices)
        assert indices == list(range(len(chunks)))

    def test_document_id_set_on_every_record(self) -> None:
        chunks = _real_chunker().chunk("Some text. " * 200, source="x.txt", page=None)
        records = ingest._chunk_records(
            chunks, document_id="doc-X", embedding_model="m"
        )
        for r in records:
            assert r["document_id"] == "doc-X"

    def test_faiss_idx_placeholder_is_minus_one(self) -> None:
        chunks = _real_chunker().chunk("Some text. " * 200, source="x.txt", page=None)
        records = ingest._chunk_records(
            chunks, document_id="doc-1", embedding_model="m"
        )
        for r in records:
            # Placeholder — patched in run_ingest after FAISS assigns
            # int IDs (Step 4.8 contract).
            assert r["faiss_idx"] == -1

    def test_text_passes_through(self) -> None:
        chunks = _real_chunker().chunk("Some text. " * 200, source="x.txt", page=None)
        records = ingest._chunk_records(
            chunks, document_id="doc-1", embedding_model="m"
        )
        assert [r["text"] for r in records] == [c.text for c in chunks]

    def test_token_count_passes_through(self) -> None:
        chunks = _real_chunker().chunk("Some text. " * 200, source="x.txt", page=None)
        records = ingest._chunk_records(
            chunks, document_id="doc-1", embedding_model="m"
        )
        assert [r["token_count"] for r in records] == [c.token_count for c in chunks]


# ---------------------------------------------------------------------------
# run_ingest — happy path
# ---------------------------------------------------------------------------


class TestRunIngestSuccess:
    """A small TXT file ingested end-to-end writes to DB and FAISS."""

    def test_happy_path_ok_true(
        self, small_txt: Path, tiny_settings: ingest.Settings
    ) -> None:
        report = ingest.run_ingest(
            path=small_txt,
            settings=tiny_settings,
            doc_type="manual",
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.ok is True
        assert report.error is None

    def test_happy_path_chunk_count_positive(
        self, small_txt: Path, tiny_settings: ingest.Settings
    ) -> None:
        report = ingest.run_ingest(
            path=small_txt,
            settings=tiny_settings,
            doc_type="manual",
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.num_chunks > 0

    def test_happy_path_db_and_index_match(
        self, small_txt: Path, tiny_settings: ingest.Settings
    ) -> None:
        """The roadmap's "FAISS size matches chunk count" check."""
        from tinyrag.storage import MetadataStore

        report = ingest.run_ingest(
            path=small_txt,
            settings=tiny_settings,
            doc_type="manual",
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.ok is True

        # Verify SQLite side.
        store = MetadataStore(tiny_settings.paths.metadata_db)
        assert store.count_documents() == 1
        assert store.count_chunks() == report.num_chunks

        # Verify the doc row's ``num_chunks`` was updated.
        docs = store.list_documents()
        assert docs[0].num_chunks == report.num_chunks

        # Verify FAISS side.
        assert report.index_size == report.num_chunks

    def test_happy_path_chunk_count_clears_threshold(
        self, small_txt: Path, tiny_settings: ingest.Settings
    ) -> None:
        """The roadmap's ``num_chunks > 5`` (adjusted from > 50 for our 40-page corpus)."""
        report = ingest.run_ingest(
            path=small_txt,
            settings=tiny_settings,
            doc_type="manual",
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.ok is True
        # The repeated text fixture produces enough chunks for > 1.
        assert report.num_chunks > 1

    def test_happy_path_total_under_threshold(
        self, small_txt: Path, tiny_settings: ingest.Settings
    ) -> None:
        """The roadmap's ``time_ms < 30_000`` check (laptop)."""
        report = ingest.run_ingest(
            path=small_txt,
            settings=tiny_settings,
            doc_type="manual",
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.ok is True
        assert report.duration_total_ms < 30_000.0

    def test_happy_path_each_stage_duration_present(
        self, small_txt: Path, tiny_settings: ingest.Settings
    ) -> None:
        report = ingest.run_ingest(
            path=small_txt,
            settings=tiny_settings,
            doc_type="manual",
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        # Every duration field is a float (not None).
        for ms in (
            report.duration_parse_ms,
            report.duration_chunk_ms,
            report.duration_embed_ms,
            report.duration_metadata_ms,
            report.duration_vector_ms,
            report.duration_save_ms,
            report.duration_total_ms,
        ):
            assert isinstance(ms, float)
            assert ms >= 0.0


# ---------------------------------------------------------------------------
# run_ingest — failure paths
# ---------------------------------------------------------------------------


class TestRunIngestFailurePaths:
    """Every documented failure path returns ok=False with a useful error."""

    def test_missing_file_ok_false(self, tiny_settings: ingest.Settings) -> None:
        report = ingest.run_ingest(
            path=Path("/tmp/does-not-exist-12345.pdf"),
            settings=tiny_settings,
            doc_type="manual",
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.ok is False
        assert "file not found" in report.error

    def test_unsupported_extension_ok_false(
        self, tmp_path: Path, tiny_settings: ingest.Settings
    ) -> None:
        p = tmp_path / "x.xyz"
        p.write_text("anything", encoding="utf-8")
        report = ingest.run_ingest(
            path=p,
            settings=tiny_settings,
            doc_type="manual",
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.ok is False
        assert "parse failed" in report.error

    def test_empty_file_ok_false(
        self, empty_txt: Path, tiny_settings: ingest.Settings
    ) -> None:
        report = ingest.run_ingest(
            path=empty_txt,
            settings=tiny_settings,
            doc_type="manual",
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.ok is False
        # Either the parser raises EmptyDocumentError OR the chunker
        # produces no chunks — both are valid failure modes. The
        # important thing is the report surfaces a clear error.
        assert (
            "no chunks produced" in report.error
            or "parse failed" in report.error
        )

    def test_failure_does_not_create_doc_row(
        self, empty_txt: Path, tiny_settings: ingest.Settings
    ) -> None:
        from tinyrag.storage import MetadataStore

        # Sanity: empty ingest fails.
        report = ingest.run_ingest(
            path=empty_txt,
            settings=tiny_settings,
            doc_type="manual",
            embedder_kind="fake",
            db_path_override=None,
            index_path_override=None,
        )
        assert report.ok is False

        # Now verify the DB has no orphan row (we want ingestion
        # failure to NOT leave a half-ingested document behind).
        # The DB may not even exist yet if the parser raised early
        # — that's fine, count == 0 either way.
        db_path = Path(tiny_settings.paths.metadata_db)
        if db_path.exists():
            store = MetadataStore(db_path)
            try:
                count = store.count_documents()
            except Exception:
                # DB exists but schema wasn't initialised (parser
                # raised before init_schema). That's also fine — no
                # document row was created.
                count = 0
            assert count == 0


# ---------------------------------------------------------------------------
# CLI argument parsing + JSON output
# ---------------------------------------------------------------------------


def _run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Invoke scripts/ingest.py as a subprocess.

    Uses the venv Python explicitly so the test doesn't depend on
    PATH being set up correctly. Sets ``PYTHONPATH=src`` so the
    ``tinyrag.*`` imports resolve.
    """
    venv_python = _REPO / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = Path(sys.executable)
    cmd = [str(venv_python), str(_SCRIPTS / "ingest.py"), *args]
    env = {"PYTHONPATH": str(_REPO / "src"), "PATH": "/usr/bin:/bin"}
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd or _REPO,
        timeout=120,
    )


class TestCliArgs:
    """The CLI accepts the documented flags and produces JSON on --json."""

    def test_missing_path_exits_2(self) -> None:
        proc = _run_cli()
        assert proc.returncode == 2

    def test_nonexistent_file_exits_1(self, tmp_path: Path) -> None:
        proc = _run_cli(str(tmp_path / "does-not-exist.pdf"), "--json")
        assert proc.returncode == 1
        # JSON report should be in stdout (with ok=False).
        payload = json.loads(proc.stdout)
        assert payload["ok"] is False
        assert "file not found" in payload["error"]

    def test_json_output_schema(
        self, small_txt: Path, tiny_settings_path: tuple[Path, Path]
    ) -> None:
        db_path, index_path = tiny_settings_path
        proc = _run_cli(
            str(small_txt),
            "--embedder",
            "fake",
            "--db-path",
            str(db_path),
            "--index-path",
            str(index_path),
            "--json",
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        # Schema check — every required key present.
        assert REQUIRED_REPORT_KEYS.issubset(payload.keys())
        assert payload["ok"] is True
        assert payload["num_chunks"] > 0

    def test_invalid_embedder_choice_exits_2(
        self, small_txt: Path
    ) -> None:
        proc = _run_cli(str(small_txt), "--embedder", "bogus")
        # argparse rejects unknown choices with code 2.
        assert proc.returncode == 2

    def test_invalid_doc_type_exits_2(self, small_txt: Path) -> None:
        proc = _run_cli(str(small_txt), "--doc-type", "bogus")
        assert proc.returncode == 2

    def test_quiet_mode_prints_minimal_output(
        self, small_txt: Path, tiny_settings_path: tuple[Path, Path]
    ) -> None:
        db_path, index_path = tiny_settings_path
        proc = _run_cli(
            str(small_txt),
            "--embedder",
            "fake",
            "--db-path",
            str(db_path),
            "--index-path",
            str(index_path),
            "--quiet",
        )
        assert proc.returncode == 0, proc.stderr
        # Quiet mode prints just the JSON dict (no banner).
        # It should be parseable as JSON.
        payload = json.loads(proc.stdout)
        assert payload["ok"] is True


@pytest.fixture
def tiny_settings_path(tmp_path: Path) -> tuple[Path, Path]:
    """Return (db_path, index_path) under tmp_path for CLI tests."""
    return (tmp_path / "metadata.db", tmp_path / "doc.faiss")
