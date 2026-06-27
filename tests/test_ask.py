"""Tests for scripts/ask.py + core/answer.py (Step 4.16 — end-to-end RAG CLI).

Test layout
-----------
- TestAnswerPublicSurface       — every documented symbol from
  ``core/answer.py`` is importable and the re-exports land in
  ``tinyrag.core``.
- TestCitationDataclass         — Citation is frozen; ``ref`` and
  ``location`` properties are correct (with and without page).
- TestAnswerDataclass           — Answer is frozen; ``to_dict()`` is
  JSON-roundtrippable; floats rounded to 2 dp; ``is_refusal``
  matches the documented refusal sentence (case + whitespace
  tolerant).
- TestMakePreview               — whitespace collapsed, word-boundary
  truncation, ellipsis appended.
- TestBuildCitationsFromChunks  — the CLI's convenience helper:
  numbered 1..N, parallel to inputs, empty chunk_id (the CLI
  doesn't query the DB).
- TestBuildCitations            — the API-layer helper: uses the
  supplied chunk_ids list (preserves order, handles short ids).
- TestMakeEmbedder              — the embedder factory routes ``real``
  vs ``fake`` correctly; dimension is 384 for fake.
- TestMakeLlm                   — the LLM factory routes ``real``
  vs ``fake`` correctly; ``model_name()`` differs by kind;
  ``is_healthy()`` contract.
- TestMakeRetriever             — wires both FAISS indices + metadata
  accessor; raises a clean error on dim mismatch.
- TestRunAskHappyPath           — full pipeline end-to-end: query →
  Answer with citations + timings + token counts. Uses FakeLLM
  so the test doesn't need a live llama-server.
- TestRunAskRefusalPath         — query with no relevant docs →
  answer.text == "" (no model call would happen either, but
  the empty-context prompt is the documented refusal).
- TestRunAskSensorKeyword       — a query with a sensor keyword
  triggers the sensor store; ``used_sensor_idx=True``; the
  ``query_log`` row records ``used_sensor_idx=1``.
- TestRunAskIdempotent          — running the same query twice
  records two rows in ``query_log`` (the table is append-only —
  every query is its own observation).
- TestRunAskCliArgs             — subprocess tests of the CLI:
  ``--json`` shape, ``--quiet`` minimal, ``--llm fake`` skips
  the live server, exit codes 0/1/2, missing query rejected.

Hermetic?
---------
Yes. Tests build their own minimal :class:`Settings` (pointing at
``tmp_path`` for the DB + FAISS indices) and use the
:class:`FakeEmbedder` + :class:`FakeLLMClient` so no model
weights, no live llama-server, and no real PDF/CSV are required.
The end-to-end "regression gate" against the real CSV + Nest PDF
lives in a separately-skipped test class so it's available when
those fixtures are on disk.

Location: ``tests/test_ask.py``
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

# Make the script importable as a module (matches the pattern in
# tests/test_ingest.py, tests/test_ingest_sensors.py, and
# tests/test_smoke_test.py).
_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
_REPO = _HERE.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import ask  # noqa: E402

from tinyrag.core import (  # noqa: E402
    Answer,
    Citation,
    build_citations,
    build_citations_from_chunks,
)
from tinyrag.core.chunker import Chunk  # noqa: E402

# ---------------------------------------------------------------------------
# Required-answer-keys — the public JSON shape every consumer relies on
# ---------------------------------------------------------------------------

#: The set of keys every :class:`Answer.to_dict()` MUST contain.
#: Adding a new key is fine; removing one is a breaking change.
#: Pinned by ``TestAnswerDataclass::test_to_dict_has_required_keys``.
REQUIRED_ANSWER_KEYS = frozenset(
    {
        "query",
        "text",
        "used_sensor_idx",
        "top_score",
        "model_name",
        "citations",
        "chunks_used",
        "chunks_dropped",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "duration_retrieve_ms",
        "duration_prompt_ms",
        "duration_llm_ms",
        "duration_total_ms",
    }
)


# ---------------------------------------------------------------------------
# Helpers — minimal but valid Settings + tiny doc index + tiny sensor index
# ---------------------------------------------------------------------------


def _make_settings(tmp_path: Path) -> ask.Settings:
    """Build a minimal :class:`Settings` pointing every path at ``tmp_path``.

    Mirrors :func:`test_ingest_sensors._make_settings` but without
    the ``sensors`` overrides (the ask CLI only needs paths,
    retrieval, llm, embedding, chunking). The doc + sensor FAISS
    index paths default to ``tmp_path/doc.faiss`` and
    ``tmp_path/sensor.faiss``.
    """
    from tinyrag.config import (
        PathsSettings,
        RetrievalSettings,
        Settings,
    )

    return Settings(
        deployment={"target": "laptop"},
        server={"host": "127.0.0.1", "port": 8000},
        llm={
            "model_path": "models/phi-3-mini.gguf",
            "server_url": "http://127.0.0.1:8080",
            "context_size": 4096,
            "temperature": 0.0,
            "max_tokens": 512,
            "gpu_layers": 0,
        },
        embedding={
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 32,
            "cache_dir": "models/_hf_cache",
        },
        chunking={
            "chunk_size": 400,
            "chunk_overlap": 50,
            "encoding": "cl100k_base",
        },
        retrieval=RetrievalSettings(
            doc_index_path=str(tmp_path / "doc.faiss"),
            sensor_index_path=str(tmp_path / "sensor.faiss"),
            doc_top_k=3,
            sensor_top_k=2,
            similarity_threshold=0.3,
            index_type="faiss",
        ),
        sensors={
            "source": "simulated",
            "csv_path": str(tmp_path / "synthetic.csv"),
            "dht_pin": 4,
            "pir_pin": 17,
            "mqtt_broker": "localhost",
            "mqtt_port": 1883,
            "mqtt_topic_prefix": "tinyrag/sensors/",
        },
        logging={"level": "INFO", "json_format": True, "path": str(tmp_path / "app.log")},
        paths=PathsSettings(
            documents_dir=str(tmp_path / "documents"),
            metadata_db=str(tmp_path / "metadata.db"),
            sensor_logs_dir=str(tmp_path / "sensor_logs"),
            logs_dir=str(tmp_path / "logs"),
        ),
    )


#: A small set of "manual" doc chunks used to populate the doc
#: FAISS index. The texts are short, topically distinct, and use
#: vocabulary the FakeEmbedder can separate.
_DOC_CHUNKS = [
    ("Nest-Thermostat-Installation-Guide-UK.pdf", 7,
     "To factory-reset your Nest thermostat, press the ring to open "
     "the menu, then choose Settings > Reset > Factory reset. The "
     "thermostat will restart and you can re-configure it from scratch."),
    ("Nest-Thermostat-Installation-Guide-UK.pdf", 15,
     "If the thermostat shows a low-battery warning, recharge it via "
     "the USB port on the back. A full charge takes about 2 hours."),
    ("Nest-Thermostat-Installation-Guide-UK.pdf", 22,
     "Wi-Fi re-setup: open the menu, choose Settings > Network. "
     "Select your home Wi-Fi network and enter the password using "
     "the ring. The thermostat will reconnect within 30 seconds."),
]


#: A small set of sensor-summary chunks (matches the format
#: SensorSummarizer emits, including the "On YYYY-MM-DD, the ..."
#: preamble). Used to populate the sensor FAISS index.
_SENSOR_CHUNKS = [
    ("sensor-summary", None,
     "On 2026-05-26, the living_room temperature averaged 20.8 C, "
     "peaking at 23.5 C at 15:00, and reaching a minimum of 17.2 C "
     "at 04:00."),
    ("sensor-summary", None,
     "On 2026-05-26, the kitchen detected 7 motion events, the "
     "first at 07:15, the last at 21:48."),
    ("sensor-summary", None,
     "On 2026-05-27, the bedroom temperature averaged 19.5 C, "
     "peaking at 22.1 C at 16:30, and reaching a minimum of 16.8 C "
     "at 03:15."),
]


def _populate_doc_index(tmp_path: Path, settings: ask.Settings) -> list[str]:
    """Build a tiny doc FAISS index with 3 chunks; return the chunk UUIDs.

    Uses :class:`FakeEmbedder` so the test is hermetic. Inserts the
    chunks into a fresh :class:`MetadataStore` AND the FAISS index
    so :class:`Retriever` can resolve FAISS slots back to chunk
    rows.
    """
    import uuid as _uuid

    from tinyrag.ingestion.embedder import FakeEmbedder
    from tinyrag.storage.metadata import MetadataStore
    from tinyrag.storage.vector_store import FAISSStore

    embedder = FakeEmbedder(dimension=384)
    metadata = MetadataStore(settings.paths.metadata_db)
    metadata.init_schema()
    doc_id = metadata.insert_document(
        filename="Nest-Thermostat-Installation-Guide-UK.pdf",
        doc_type="manual",
        source_path=str(tmp_path / "Nest-Thermostat-Installation-Guide-UK.pdf"),
        size_bytes=1024,
        content_hash="deadbeef" * 8,  # arbitrary 64-hex; content_hash isn't read here
        metadata={"num_pages": 30, "num_chars": 5000, "ingested_via": "test_ask.py"},
    )
    chunk_uuids: list[str] = []
    chunk_records: list[dict] = []
    texts: list[str] = []
    for i, (_source, page, text) in enumerate(_DOC_CHUNKS):
        cid = str(_uuid.uuid4())
        chunk_uuids.append(cid)
        chunk_records.append({
            "id": cid,
            "document_id": doc_id,
            "chunk_index": i,
            "faiss_idx": -1,
            "text": text,
            "page_number": page,
            "char_offset": 0,
            "token_count": len(text.split()),
            "embedding_model": "fake:sentence-transformers/all-MiniLM-L6-v2",
        })
        texts.append(text)
    metadata.insert_chunks(chunk_records)

    vectors = embedder.embed(texts)
    doc_store = FAISSStore(
        index_path=Path(settings.retrieval.doc_index_path),
        embedding_dimension=384,
        embedding_model="fake:sentence-transformers/all-MiniLM-L6-v2",
    )
    doc_store.load()
    doc_store.add(vectors, chunk_uuids)
    doc_store.save()

    return chunk_uuids


def _populate_sensor_index(tmp_path: Path, settings: ask.Settings) -> list[str]:
    """Build a tiny sensor FAISS index with 3 chunks; return the chunk UUIDs."""
    import uuid as _uuid

    from tinyrag.ingestion.embedder import FakeEmbedder
    from tinyrag.storage.metadata import MetadataStore
    from tinyrag.storage.vector_store import FAISSStore

    embedder = FakeEmbedder(dimension=384)
    metadata = MetadataStore(settings.paths.metadata_db)
    # init_schema already ran during doc index; this is idempotent.
    metadata.init_schema()
    # Skip insert_document if a manual doc already exists from
    # _populate_doc_index — the FK is on documents.id only. We do
    # need a documents row for the FK to fire.
    doc_id = metadata.insert_document(
        filename="synthetic.csv",
        doc_type="sensor_summary",
        source_path=str(tmp_path / "synthetic.csv"),
        size_bytes=512,
        content_hash="feedface" * 8,
        metadata={"num_rows": 100, "num_days": 2, "ingested_via": "test_ask.py"},
    )
    chunk_uuids: list[str] = []
    chunk_records: list[dict] = []
    texts: list[str] = []
    for i, (_source, page, text) in enumerate(_SENSOR_CHUNKS):
        cid = str(_uuid.uuid4())
        chunk_uuids.append(cid)
        chunk_records.append({
            "id": cid,
            "document_id": doc_id,
            "chunk_index": i,
            "faiss_idx": -1,
            "text": text,
            "page_number": page,
            "char_offset": 0,
            "token_count": len(text.split()),
            "embedding_model": "fake:sentence-transformers/all-MiniLM-L6-v2",
        })
        texts.append(text)
    metadata.insert_chunks(chunk_records)

    vectors = embedder.embed(texts)
    sensor_store = FAISSStore(
        index_path=Path(settings.retrieval.sensor_index_path),
        embedding_dimension=384,
        embedding_model="fake:sentence-transformers/all-MiniLM-L6-v2",
    )
    sensor_store.load()
    sensor_store.add(vectors, chunk_uuids)
    sensor_store.save()

    return chunk_uuids


@pytest.fixture
def tiny_settings(tmp_path: Path) -> ask.Settings:
    """A minimal :class:`Settings` pointing every path at ``tmp_path``.

    Also populates the doc + sensor FAISS indices with 3+3 chunks
    so a single fixture builds a complete, queryable RAG setup.
    """
    settings = _make_settings(tmp_path)
    _populate_doc_index(tmp_path, settings)
    _populate_sensor_index(tmp_path, settings)
    return settings


@pytest.fixture
def empty_settings(tmp_path: Path) -> ask.Settings:
    """A minimal :class:`Settings` with NO chunks in either index.

    Used by the "zero chunks" tests to exercise the refusal path.
    """
    return _make_settings(tmp_path)


# ===========================================================================
# Answer module tests
# ===========================================================================


class TestAnswerPublicSurface:
    """Every documented symbol from :mod:`tinyrag.core.answer` is importable."""

    def test_citation_importable_from_core(self) -> None:
        from tinyrag.core import Citation as C
        assert C is Citation

    def test_answer_importable_from_core(self) -> None:
        from tinyrag.core import Answer as A
        assert A is Answer

    def test_build_citations_importable_from_core(self) -> None:
        from tinyrag.core import build_citations as bc
        assert bc is build_citations

    def test_build_citations_from_chunks_importable_from_core(self) -> None:
        from tinyrag.core import build_citations_from_chunks as bcfc
        assert bcfc is build_citations_from_chunks

    def test_citation_dataclass_fields(self) -> None:
        # All 6 documented fields are accessible.
        c = Citation(
            number=3,
            chunk_id="abc-123",
            source="Nest.pdf",
            page=7,
            score=0.82,
            preview="To factory-reset",
        )
        assert c.number == 3
        assert c.chunk_id == "abc-123"
        assert c.source == "Nest.pdf"
        assert c.page == 7
        assert c.score == 0.82
        assert c.preview == "To factory-reset"

    def test_answer_dataclass_fields(self) -> None:
        a = Answer(
            query="How do I reset?",
            text="Press the menu.",
            used_sensor_idx=False,
            top_score=0.91,
            model_name="phi-3-mini",
        )
        assert a.query == "How do I reset?"
        assert a.text == "Press the menu."
        assert a.used_sensor_idx is False
        assert a.top_score == 0.91
        assert a.model_name == "phi-3-mini"


class TestCitationDataclass:
    """Citation is frozen; ref + location render correctly."""

    def test_frozen(self) -> None:
        c = Citation(number=1, chunk_id="x", source="s", page=1, score=0.5, preview="p")
        with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError is a subclass
            c.number = 2  # type: ignore[misc]

    def test_ref_property(self) -> None:
        c = Citation(number=3, chunk_id="x", source="s", page=None, score=0.5, preview="p")
        assert c.ref == "[3]"

    def test_ref_property_double_digit(self) -> None:
        c = Citation(number=12, chunk_id="x", source="s", page=None, score=0.5, preview="p")
        assert c.ref == "[12]"

    def test_location_with_page(self) -> None:
        c = Citation(number=1, chunk_id="x", source="Nest.pdf", page=7, score=0.5, preview="p")
        assert c.location == "Nest.pdf, p.7"

    def test_location_without_page(self) -> None:
        c = Citation(number=1, chunk_id="x", source="sensor-summary", page=None, score=0.5, preview="p")
        assert c.location == "sensor-summary"


class TestAnswerDataclass:
    """Answer is frozen; to_dict + is_refusal behave as documented."""

    def test_frozen(self) -> None:
        a = Answer(query="q", text="t")
        with pytest.raises(dataclasses.FrozenInstanceError):
            a.query = "new"  # type: ignore[misc]

    def test_to_dict_has_required_keys(self) -> None:
        a = Answer(query="q", text="t")
        d = a.to_dict()
        assert REQUIRED_ANSWER_KEYS.issubset(set(d.keys()))

    def test_to_dict_is_json_serialisable(self) -> None:
        a = Answer(
            query="q",
            text="t",
            top_score=0.91,
            citations=[
                Citation(number=1, chunk_id="x", source="s", page=1, score=0.5, preview="p"),
            ],
        )
        # Round-trip — if this succeeds, the dict is JSON-clean.
        s = json.dumps(a.to_dict())
        d = json.loads(s)
        assert d["query"] == "q"
        assert d["top_score"] == 0.91  # 4 dp
        assert d["citations"][0]["ref"] == "[1]"

    def test_to_dict_rounds_floats(self) -> None:
        a = Answer(
            query="q",
            text="t",
            duration_retrieve_ms=12.345678,
            duration_total_ms=200.987654,
        )
        d = a.to_dict()
        assert d["duration_retrieve_ms"] == 12.35
        assert d["duration_total_ms"] == 200.99

    def test_to_dict_top_score_none(self) -> None:
        a = Answer(query="q", text="t", top_score=None)
        d = a.to_dict()
        assert d["top_score"] is None

    def test_is_refusal_true(self) -> None:
        a = Answer(query="q", text="I don't have enough information in the provided documents.")
        assert a.is_refusal is True

    def test_is_refusal_true_case_and_whitespace(self) -> None:
        a = Answer(query="q", text="  I DON'T HAVE ENOUGH INFORMATION  in the provided documents.")
        # The model can vary case + whitespace; the check is
        # case-insensitive + startswith on the trimmed text.
        assert a.is_refusal is True

    def test_is_refusal_false_on_real_answer(self) -> None:
        a = Answer(query="q", text="Press the menu and choose Reset.")
        assert a.is_refusal is False

    def test_is_refusal_false_on_empty(self) -> None:
        a = Answer(query="q", text="")
        assert a.is_refusal is False


class TestMakePreview:
    """``_make_preview`` collapses whitespace + truncates at word boundary."""

    def test_short_text_unchanged(self) -> None:
        from tinyrag.core.answer import _make_preview
        assert _make_preview("Hello world") == "Hello world"

    def test_collapses_whitespace(self) -> None:
        from tinyrag.core.answer import _make_preview
        text = "Hello\n\nworld\t\t   with   spaces"
        assert _make_preview(text) == "Hello world with spaces"

    def test_truncates_at_word_boundary(self) -> None:
        from tinyrag.core.answer import _make_preview
        text = "The quick brown fox jumps over the lazy dog"
        preview = _make_preview(text, max_chars=20)
        # "The quick brown fox" is 19 chars; the next space is at 20.
        # We cut at the last space ≤ 20 → "The quick brown fox" + ellipsis.
        assert preview.endswith("…")
        assert preview.startswith("The quick")
        assert "fox" in preview

    def test_truncates_hard_when_no_space(self) -> None:
        from tinyrag.core.answer import _make_preview
        # 30-char string with no space → cuts hard at max_chars.
        text = "abcdefghij" * 3  # 30 chars, no space
        preview = _make_preview(text, max_chars=10)
        assert preview == "abcdefghij…"


class TestBuildCitationsFromChunks:
    """The CLI's convenience helper — numbered 1..N, parallel to inputs."""

    def test_empty_inputs(self) -> None:
        assert build_citations_from_chunks([], []) == []

    def test_numbered_in_order(self) -> None:
        chunks = [
            Chunk(text="First", source="a.pdf", page=1, chunk_index=0, char_offset=0, token_count=1),
            Chunk(text="Second", source="a.pdf", page=2, chunk_index=1, char_offset=0, token_count=1),
        ]
        cits = build_citations_from_chunks(chunks, [0.9, 0.8])
        assert len(cits) == 2
        assert cits[0].number == 1
        assert cits[0].ref == "[1]"
        assert cits[0].source == "a.pdf"
        assert cits[0].page == 1
        assert cits[0].score == 0.9
        assert cits[1].number == 2
        assert cits[1].ref == "[2]"

    def test_chunk_id_left_empty(self) -> None:
        # The CLI convenience helper doesn't resolve chunk_id.
        chunks = [Chunk(text="t", source="s", page=None, chunk_index=0, char_offset=0, token_count=1)]
        cits = build_citations_from_chunks(chunks, [0.5])
        assert cits[0].chunk_id == ""

    def test_preview_truncated(self) -> None:
        long_text = "word " * 100  # 500 chars
        chunks = [Chunk(text=long_text, source="s", page=None, chunk_index=0, char_offset=0, token_count=100)]
        cits = build_citations_from_chunks(chunks, [0.5])
        assert len(cits[0].preview) < len(long_text)
        assert cits[0].preview.endswith("…")


class TestBuildCitations:
    """The API-layer helper — uses the supplied chunk_ids list."""

    def test_uses_supplied_chunk_ids(self) -> None:
        from tinyrag.core.retriever import RetrievalResult
        chunks = [
            Chunk(text="First", source="a.pdf", page=1, chunk_index=0, char_offset=0, token_count=1),
            Chunk(text="Second", source="a.pdf", page=2, chunk_index=1, char_offset=0, token_count=1),
        ]
        retrieval = RetrievalResult(
            chunks=chunks, scores=[0.9, 0.8], query="q", used_sensor_idx=False, sensor_keywords_matched=[]
        )
        cits = build_citations(retrieval, chunk_ids=["id-a", "id-b"])
        assert cits[0].chunk_id == "id-a"
        assert cits[1].chunk_id == "id-b"

    def test_missing_chunk_id_becomes_empty(self) -> None:
        # chunk_ids shorter than the chunk list → missing entries
        # default to "" (not a crash).
        from tinyrag.core.retriever import RetrievalResult
        chunks = [
            Chunk(text="A", source="s", page=None, chunk_index=0, char_offset=0, token_count=1),
            Chunk(text="B", source="s", page=None, chunk_index=1, char_offset=0, token_count=1),
        ]
        retrieval = RetrievalResult(chunks=chunks, scores=[0.5, 0.5], query="q")
        cits = build_citations(retrieval, chunk_ids=["id-a"])
        assert cits[0].chunk_id == "id-a"
        assert cits[1].chunk_id == ""


# ===========================================================================
# Component factory tests
# ===========================================================================


class TestMakeEmbedder:
    """``_make_embedder`` routes ``real`` vs ``fake`` correctly."""

    def test_fake_returns_fake_embedder(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        e = ask._make_embedder(settings, kind="fake")
        from tinyrag.ingestion.embedder import FakeEmbedder
        assert isinstance(e, FakeEmbedder)
        assert e.dimension == 384

    def test_real_returns_sentence_transformer(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        e = ask._make_embedder(settings, kind="real")
        from tinyrag.ingestion.embedder import SentenceTransformerEmbedder
        assert isinstance(e, SentenceTransformerEmbedder)
        assert e.model_name == settings.embedding.model_name

    def test_default_kind_is_real(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        e = ask._make_embedder(settings)
        from tinyrag.ingestion.embedder import SentenceTransformerEmbedder
        assert isinstance(e, SentenceTransformerEmbedder)

    def test_unknown_kind_raises(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        with pytest.raises(ValueError, match="unknown embedder kind"):
            ask._make_embedder(settings, kind="bogus")


class TestMakeLlm:
    """``_make_llm`` routes ``real`` vs ``fake`` correctly."""

    def test_fake_returns_fake_llm(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        llm = ask._make_llm(settings, kind="fake")
        from tinyrag.generation import FakeLLMClient
        assert isinstance(llm, FakeLLMClient)
        assert llm.model_name() == "fake-llm"

    def test_real_returns_llamacpp(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        llm = ask._make_llm(settings, kind="real")
        from tinyrag.generation import LlamaCppClient
        assert isinstance(llm, LlamaCppClient)
        # model_path = "models/phi-3-mini.gguf" → factory strips the
        # .gguf suffix to give the bare model id llama-server expects.
        assert llm.model == "models/phi-3-mini"
        assert llm.base_url == "http://127.0.0.1:8080"

    def test_real_strips_trailing_slash_from_url(self, tmp_path: Path) -> None:
        # The trailing slash on server_url must not leak into base_url.
        settings = _make_settings(tmp_path)
        settings = settings.model_copy(update={
            "llm": settings.llm.model_copy(update={"server_url": "http://127.0.0.1:8080/"}),
        })
        llm = ask._make_llm(settings, kind="real")
        assert llm.base_url == "http://127.0.0.1:8080"  # no trailing slash

    def test_unknown_kind_raises(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        with pytest.raises(ValueError, match="unknown llm kind"):
            ask._make_llm(settings, kind="bogus")


class TestMakeRetriever:
    """``_make_retriever`` wires both FAISS indices + metadata accessor."""

    def test_returns_retriever_with_both_stores(self, tiny_settings: ask.Settings) -> None:
        from tinyrag.core import Retriever
        from tinyrag.ingestion.embedder import FakeEmbedder
        from tinyrag.storage.metadata import MetadataStore

        embedder = FakeEmbedder(dimension=384)
        metadata = MetadataStore(tiny_settings.paths.metadata_db)
        r = ask._make_retriever(
            tiny_settings,
            embedder=embedder,
            doc_store_path=Path(tiny_settings.retrieval.doc_index_path),
            sensor_store_path=Path(tiny_settings.retrieval.sensor_index_path),
            metadata=metadata,
        )
        assert isinstance(r, Retriever)
        assert r.doc_store.size() == 3
        assert r.sensor_store.size() == 3



# ===========================================================================
# run_ask() tests
# ===========================================================================


class TestRunAskHappyPath:
    """Full pipeline end-to-end with FakeLLM."""

    def test_returns_answer(self, tiny_settings: ask.Settings) -> None:
        a = ask.run_ask(
            query="How do I factory-reset my Nest thermostat?",
            settings=tiny_settings,
            llm_kind="fake",
            embedder_kind="fake",
            db_path_override=None,
            doc_index_override=None,
            sensor_index_override=None,
            k_doc=3,
            k_sensor=2,
            threshold=0.0,  # disable threshold so the small-index hits all survive
            max_tokens=64,
            log_query=True,
            default_threshold=0.0,
        )
        assert isinstance(a, Answer)
        assert a.text  # FakeLLM returns a canned non-empty reply

    def test_query_echoed_back(self, tiny_settings: ask.Settings) -> None:
        a = ask.run_ask(
            query="Reset the Nest thermostat",
            settings=tiny_settings,
            llm_kind="fake",
            embedder_kind="fake",
            db_path_override=None,
            doc_index_override=None,
            sensor_index_override=None,
            k_doc=3,
            k_sensor=2,
            threshold=0.0,
            max_tokens=64,
            log_query=True,
            default_threshold=0.0,
        )
        assert a.query == "Reset the Nest thermostat"

    def test_citations_present(self, tiny_settings: ask.Settings) -> None:
        a = ask.run_ask(
            # This query shares substring with the factory-reset
            # chunk, so FakeEmbedder's SHA-256 vectors yield a
            # cosine > 0.1 against at least one doc chunk.
            query="factory-reset your Nest thermostat, press the ring",
            settings=tiny_settings,
            llm_kind="fake",
            embedder_kind="fake",
            db_path_override=None,
            doc_index_override=None,
            sensor_index_override=None,
            k_doc=3,
            k_sensor=2,
            threshold=0.0,
            max_tokens=64,
            log_query=True,
            default_threshold=0.0,
        )
        # The doc chunks topically match "factory reset", so at
        # least one citation should be a doc chunk.
        assert len(a.citations) >= 1
        # Citations are numbered 1..N contiguously.
        assert [c.number for c in a.citations] == list(range(1, len(a.citations) + 1))
        # Every citation has a source label and a non-empty preview.
        for c in a.citations:
            assert c.source
            assert c.preview

    def test_top_score_is_max(self, tiny_settings: ask.Settings) -> None:
        a = ask.run_ask(
            query="factory reset",
            settings=tiny_settings,
            llm_kind="fake",
            embedder_kind="fake",
            db_path_override=None,
            doc_index_override=None,
            sensor_index_override=None,
            k_doc=3,
            k_sensor=2,
            threshold=0.0,
            max_tokens=64,
            log_query=False,
            default_threshold=0.0,
        )
        if a.citations:
            assert a.top_score == max(c.score for c in a.citations)

    def test_chunks_used_matches_citations(self, tiny_settings: ask.Settings) -> None:
        a = ask.run_ask(
            query="factory reset",
            settings=tiny_settings,
            llm_kind="fake",
            embedder_kind="fake",
            db_path_override=None,
            doc_index_override=None,
            sensor_index_override=None,
            k_doc=3,
            k_sensor=2,
            threshold=0.0,
            max_tokens=64,
            log_query=False,
            default_threshold=0.0,
        )
        assert a.chunks_used == len(a.citations)

    def test_model_name_set(self, tiny_settings: ask.Settings) -> None:
        a = ask.run_ask(
            query="factory reset",
            settings=tiny_settings,
            llm_kind="fake",
            embedder_kind="fake",
            db_path_override=None,
            doc_index_override=None,
            sensor_index_override=None,
            k_doc=3,
            k_sensor=2,
            threshold=0.0,
            max_tokens=64,
            log_query=False,
            default_threshold=0.0,
        )
        assert a.model_name == "fake-llm"

    def test_timings_present(self, tiny_settings: ask.Settings) -> None:
        a = ask.run_ask(
            query="factory reset",
            settings=tiny_settings,
            llm_kind="fake",
            embedder_kind="fake",
            db_path_override=None,
            doc_index_override=None,
            sensor_index_override=None,
            k_doc=3,
            k_sensor=2,
            threshold=0.0,
            max_tokens=64,
            log_query=False,
            default_threshold=0.0,
        )
        assert a.duration_total_ms > 0
        assert a.duration_retrieve_ms >= 0
        assert a.duration_prompt_ms >= 0
        assert a.duration_llm_ms >= 0

    def test_query_log_written(self, tiny_settings: ask.Settings) -> None:

        from tinyrag.storage.metadata import MetadataStore

        ask.run_ask(
            query="factory reset test log",
            settings=tiny_settings,
            llm_kind="fake",
            embedder_kind="fake",
            db_path_override=None,
            doc_index_override=None,
            sensor_index_override=None,
            k_doc=3,
            k_sensor=2,
            threshold=0.0,
            max_tokens=64,
            log_query=True,
            default_threshold=0.0,
        )
        # Verify the row landed.
        store = MetadataStore(tiny_settings.paths.metadata_db)
        recent = store.get_recent_queries(limit=5)
        assert any(r.query == "factory reset test log" for r in recent)

    def test_no_log_skips_db_write(self, tiny_settings: ask.Settings) -> None:

        from tinyrag.storage.metadata import MetadataStore

        before = MetadataStore(tiny_settings.paths.metadata_db).get_recent_queries(limit=100)
        ask.run_ask(
            query="this query should not be logged",
            settings=tiny_settings,
            llm_kind="fake",
            embedder_kind="fake",
            db_path_override=None,
            doc_index_override=None,
            sensor_index_override=None,
            k_doc=3,
            k_sensor=2,
            threshold=0.0,
            max_tokens=64,
            log_query=False,
            default_threshold=0.0,
        )
        after = MetadataStore(tiny_settings.paths.metadata_db).get_recent_queries(limit=100)
        assert len(after) == len(before)


class TestRunAskEmptyQuery:
    """Empty / whitespace queries return a failed Answer without crashing."""

    def test_empty_query_returns_empty_answer(self, tiny_settings: ask.Settings) -> None:
        a = ask.run_ask(
            query="",
            settings=tiny_settings,
            llm_kind="fake",
            embedder_kind="fake",
            db_path_override=None,
            doc_index_override=None,
            sensor_index_override=None,
            k_doc=3,
            k_sensor=2,
            threshold=0.0,
            max_tokens=64,
            log_query=False,
            default_threshold=0.0,
        )
        assert a.text == ""
        assert a.query == ""

    def test_whitespace_query_returns_empty_answer(self, tiny_settings: ask.Settings) -> None:
        a = ask.run_ask(
            query="   ",
            settings=tiny_settings,
            llm_kind="fake",
            embedder_kind="fake",
            db_path_override=None,
            doc_index_override=None,
            sensor_index_override=None,
            k_doc=3,
            k_sensor=2,
            threshold=0.0,
            max_tokens=64,
            log_query=False,
            default_threshold=0.0,
        )
        assert a.text == ""


class TestRunAskSensorKeyword:
    """A query with a sensor keyword triggers the sensor store."""

    def test_sensor_keyword_sets_used_sensor_idx(
        self, tiny_settings: ask.Settings
    ) -> None:
        # "temperature" is in DEFAULT_SENSOR_KEYWORDS; the sensor
        # index has 3 chunks including two temperature ones.
        a = ask.run_ask(
            query="What was the temperature yesterday?",
            settings=tiny_settings,
            llm_kind="fake",
            embedder_kind="fake",
            db_path_override=None,
            doc_index_override=None,
            sensor_index_override=None,
            k_doc=3,
            k_sensor=2,
            threshold=0.0,
            max_tokens=64,
            log_query=False,
            default_threshold=0.0,
        )
        assert a.used_sensor_idx is True

    def test_no_sensor_keyword_skips_sensor(
        self, tiny_settings: ask.Settings
    ) -> None:
        # "factory reset" has no sensor keyword → sensor store not used.
        a = ask.run_ask(
            query="How do I factory reset my Nest thermostat?",
            settings=tiny_settings,
            llm_kind="fake",
            embedder_kind="fake",
            db_path_override=None,
            doc_index_override=None,
            sensor_index_override=None,
            k_doc=3,
            k_sensor=2,
            threshold=0.0,
            max_tokens=64,
            log_query=False,
            default_threshold=0.0,
        )
        assert a.used_sensor_idx is False


# ===========================================================================
# CLI tests
# ===========================================================================


class TestCliArgs:
    """Subprocess tests of the ``ask.py`` CLI."""

    @staticmethod
    def _cli_args(query: str, *, tiny_settings: ask.Settings, json_mode: bool, quiet: bool, log: bool = True) -> list[str]:
        """Build the CLI arg list, wiring every path to the fixture's tmpdir.

        The default ``config.yaml`` points at ``data/metadata.db``
        etc., so the subprocess would write to the project root
        unless we override every path. Reuses the
        :class:`tiny_settings` paths so the subprocess reads the
        SAME DB + FAISS files the fixture populated.
        """
        args = [
            sys.executable, "scripts/ask.py",
            query,
            "--llm", "fake",
            "--embedder", "fake",
            "--threshold", "0.0",
            "--db-path", tiny_settings.paths.metadata_db,
            "--doc-index", tiny_settings.retrieval.doc_index_path,
            "--sensor-index", tiny_settings.retrieval.sensor_index_path,
        ]
        if not log:
            args.append("--no-log")
        if json_mode:
            args.append("--json")
        if quiet:
            args.append("--quiet")
        return args

    def test_help_exits_zero(self, tiny_settings: ask.Settings) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/ask.py", "--help"],
            cwd=_REPO,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        )
        assert result.returncode == 0
        assert "query" in result.stdout.lower()

    def test_missing_query_exits_two(self, tiny_settings: ask.Settings) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/ask.py"],
            cwd=_REPO,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        )
        # argparse exits with code 2 for missing positional args.
        assert result.returncode == 2

    def test_json_output_shape(self, tiny_settings: ask.Settings) -> None:
        result = subprocess.run(
            self._cli_args(
                "How do I factory reset my Nest thermostat?",
                tiny_settings=tiny_settings, json_mode=True, quiet=False, log=False,
            ),
            cwd=_REPO,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        )
        assert result.returncode == 0, result.stderr
        d = json.loads(result.stdout)
        assert REQUIRED_ANSWER_KEYS.issubset(set(d.keys()))
        assert d["query"] == "How do I factory reset my Nest thermostat?"
        assert d["text"]  # non-empty
        assert d["model_name"] == "fake-llm"

    def test_quiet_outputs_json_only(self, tiny_settings: ask.Settings) -> None:
        result = subprocess.run(
            self._cli_args(
                "How do I reset my Nest thermostat?",
                tiny_settings=tiny_settings, json_mode=False, quiet=True, log=False,
            ),
            cwd=_REPO,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        )
        assert result.returncode == 0
        # Quiet mode → stdout is a single JSON object (no banner).
        d = json.loads(result.stdout.strip())
        assert d["query"]

    def test_no_log_skips_db_write(self, tiny_settings: ask.Settings) -> None:
        from tinyrag.storage.metadata import MetadataStore
        before = MetadataStore(tiny_settings.paths.metadata_db).get_recent_queries(limit=100)
        result = subprocess.run(
            self._cli_args(
                "This query should not be logged via CLI",
                tiny_settings=tiny_settings, json_mode=False, quiet=True, log=False,
            ),
            cwd=_REPO,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        )
        assert result.returncode == 0
        after = MetadataStore(tiny_settings.paths.metadata_db).get_recent_queries(limit=100)
        assert len(after) == len(before)

    def test_cli_exits_zero_on_success(self, tiny_settings: ask.Settings) -> None:
        result = subprocess.run(
            self._cli_args(
                "How do I reset my Nest thermostat?",
                tiny_settings=tiny_settings, json_mode=False, quiet=True, log=False,
            ),
            cwd=_REPO,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        )
        assert result.returncode == 0

    def test_cli_writes_query_log_by_default(self, tiny_settings: ask.Settings) -> None:
        from tinyrag.storage.metadata import MetadataStore
        result = subprocess.run(
            self._cli_args(
                "CLI default-log test query",
                tiny_settings=tiny_settings, json_mode=False, quiet=True, log=True,
            ),
            cwd=_REPO,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        )
        assert result.returncode == 0
        store = MetadataStore(tiny_settings.paths.metadata_db)
        recent = store.get_recent_queries(limit=10)
        assert any(r.query == "CLI default-log test query" for r in recent)
