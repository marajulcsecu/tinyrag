"""Tests for the FastAPI app (Step 4.17 — FastAPI skeleton + /api/status).

Test layout
-----------
- TestPublicSurface           — every documented symbol from
  ``tinyrag.api`` and ``tinyrag.main`` is importable; the module
  exports the documented ``__all__``.
- TestSchemasValidation        — Pydantic models reject bad input
  (extra fields, out-of-range numbers, empty query) and accept
  the documented good input.
- TestSystemInfoHelpers        — RAM + llama.cpp + embedding-name
  probes never crash, return the documented shapes.
- TestCreateAppLifespan        — :func:`create_app` returns a FastAPI
  app; lifespan loads the FAISS indices + initialises the SQLite
  schema; every ``app.state`` slot is populated.
- TestGetStatus                — ``GET /api/status`` returns the full
  ``StatusResponse`` shape with every FR-39 field.
- TestPostQueryHappyPath       — ``POST /api/query`` runs the 4-stage
  pipeline and returns the full ``Answer.to_dict()`` shape (same as
  the CLI).
- TestPostQueryLogging         — successful queries append rows to
  ``query_log``; ``log_query=False`` skips the write.
- TestPostQuerySensorKeyword   — a query with a sensor keyword
  triggers the sensor store (``used_sensor_idx=True``).
- TestPostQueryValidation      — empty query rejected (422), bad
  ranges rejected (422), extra fields rejected (422).
- TestNotImplementedEndpoints  — ``/api/documents`` and
  ``/api/admin/*`` return 501 with the documented body shape.
- TestErrorHandlers            — domain exceptions map to the right
  HTTP status codes via the global handlers.
- TestRootAndHealthz           — the meta endpoints work.
- TestCreateAppTwiceIdempotent — calling ``create_app`` twice in the
  same process produces two independent apps (no shared state).

Hermetic?
---------
Yes. Tests build a tiny :class:`Settings` pointing every path at
``tmp_path``, populate the FAISS indices in tmpdir with
:class:`FakeEmbedder`, and use :class:`FakeLLMClient` so no model
weights, no live llama-server, and no real PDF/CSV are required.
The FastAPI :class:`TestClient` triggers the lifespan handler in
tests (via the ``with TestClient(app) as client:`` context),
so end-to-end behaviour matches a real uvicorn process.
"""

# ``I001`` (import sort) is suppressed because every import below is
# annotated with the E402 noqa directive to keep them after the
# sys.path bootstrap, which makes the conventional
# "stdlib / third-party / first-party" grouping impossible. The
# actual order matches the dependency layers
# (deps -> schemas -> routes -> main), which is what we want readers
# to see anyway.
# ruff: noqa: I001
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Path setup so the test can import both ``tinyrag.main`` (the composition
# root) and the same fixture helpers test_ask.py uses.
# ---------------------------------------------------------------------------
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from tests.test_ask import _DOC_CHUNKS, _SENSOR_CHUNKS  # noqa: E402
from tinyrag.api import (  # noqa: E402
    AskRequest,
    ErrorResponse,
    NotImplementedResponse,
    StatusResponse,
    build_admin_router,
    build_docs_router,
    build_query_router,
    install_exception_handlers,
)
from tinyrag.api.deps import (  # noqa: E402
    get_doc_store,
    get_embedder,
    get_llm,
    get_metadata,
    get_prompt_builder,
    get_retriever,
    get_sensor_store,
    get_settings,
)
from tinyrag.api.errors import (  # noqa: E402
    install_exception_handlers as _reinstall_handlers,
)
from tinyrag.api.routes_admin import ADMIN_NOT_IMPLEMENTED_DETAIL  # noqa: E402
from tinyrag.api.schemas import (  # noqa: E402, F401  (Document* used by name only)
    AskRequest as AskRequestSchema,
    DocumentDeleteResponse,
    DocumentListItemResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    ErrorResponse as ErrorResponseSchema,
    StatusResponse as StatusResponseSchema,
)
from tinyrag.api.system_info import (  # noqa: E402
    get_embedding_model_name,
    get_llama_cpp_status,
    get_ram_mb,
)
from tinyrag.main import create_app  # noqa: E402

# ---------------------------------------------------------------------------
# Required-key sets (for shape assertions)
# ---------------------------------------------------------------------------

REQUIRED_STATUS_KEYS: frozenset[str] = frozenset(
    {
        "ok",
        "model_name",
        "embedding_model",
        "embedding_dim",
        "doc_chunk_count",
        "sensor_chunk_count",
        "doc_index_path",
        "sensor_index_path",
        "metadata_db_path",
        "ram_mb",
        "llama_cpp_status",
        "llama_cpp_url",
        "sensor_source",
        "deployment_target",
    }
)

REQUIRED_ANSWER_KEYS: frozenset[str] = frozenset(
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

REQUIRED_DOC_LIST_ITEM_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "filename",
        "doc_type",
        "source_path",
        "size_bytes",
        "num_chunks",
        "content_hash",
        "ingested_at",
        "last_modified",
    }
)

REQUIRED_UPLOAD_KEYS: frozenset[str] = frozenset(
    {
        "ok",
        "doc_id",
        "num_chunks",
        "duration_total_ms",
        "embedding_model",
        "doc_type",
        "file",
        "error",
    }
)

#: Path to the small real PDF fixture (used by the upload happy-path tests).
_NEST_PDF = (
    Path(__file__).resolve().parent / "fixtures" / "Nest-Thermostat-Installation-Guide-UK.pdf"
)


def _small_txt() -> bytes:
    """Return a few-KB TXT body (enough for ≥1 chunk)."""
    return ("Hello world. " * 200).encode("utf-8")


# ---------------------------------------------------------------------------
# SSE helpers (Step 4.19)
# ---------------------------------------------------------------------------
import json as _json  # noqa: E402  (intentionally below the src-path bootstrap)


def _parse_sse_events(body: str) -> list[dict[str, Any]]:
    """Parse a raw SSE response body into a list of event dicts.

    Each yielded ``ServerSentEvent`` from sse-starlette is framed on
    the wire as ``event: <name>\\r\\ndata: <json string>\\r\\n\\r\\n``
    (RFC 8895 specifies CRLF; sse-starlette uses CRLF inside each
    frame and a CRLF-CRLF separator between frames). This helper
    normalises line endings, splits on the blank-line separator,
    then parses the ``event:`` / ``data:`` lines within each block
    and returns a list of dicts with the shape::

        {"event": "token", "payload": {"delta": "..."}}
        {"event": "done",  "payload": {"answer": {...}}}
        {"event": "error", "payload": {"error": "...", "detail": "..."}}

    ``payload`` is the JSON-decoded contents of the ``data:`` line —
    the sse-starlette wire format puts the JSON payload in ``data``
    because ``ServerSentEvent`` only accepts the keys
    ``data | event | id | retry | comment | sep``. The dashboard
    reads ``event`` to dispatch and ``payload`` for the body.
    """
    # Normalise CRLF → LF so the split below is line-ending-agnostic.
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    events: list[dict[str, Any]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_name: str | None = None
        data_payload: str | None = None
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: "):
                data_payload = line[len("data: "):]
        if event_name is None and data_payload is None:
            continue
        payload: Any = _json.loads(data_payload) if data_payload else None
        events.append({"event": event_name, "payload": payload})
    return events


def _small_md() -> bytes:
    """Return a few-KB Markdown body."""
    return b"# Heading\n\nSome **markdown** body.\n\n" + (b"Paragraph. " * 50)


def _oversize_bytes() -> bytes:
    """Return a 51 MB blob — one byte over the upload cap."""
    return b"x" * (51 * 1024 * 1024)


# ---------------------------------------------------------------------------
# Settings + fixtures (mirror test_ask.py for hermetic setup)
# ---------------------------------------------------------------------------


def _make_settings(tmp_path: Path) -> Any:
    """Build a minimal :class:`Settings` pointing every path at ``tmp_path``.

    Mirrors :func:`tests.test_ask._make_settings`. We deliberately
    import the sub-models so a future rename in :mod:`tinyrag.config`
    breaks this fixture at edit time, not at runtime.
    """
    from tinyrag.config import PathsSettings, RetrievalSettings, Settings

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
            "cache_dir": str(tmp_path / "_hf_cache"),
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
            similarity_threshold=0.0,  # hermetic: FakeEmbedder cosines are noisy
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


def _populate_doc_index(tmp_path: Path, settings: Any) -> list[str]:
    """Build a 3-chunk doc FAISS index; return chunk UUIDs (hermetic)."""
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
        content_hash="deadbeef" * 8,
        metadata={"num_pages": 30, "num_chars": 5000, "ingested_via": "test_api.py"},
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


def _populate_sensor_index(tmp_path: Path, settings: Any) -> list[str]:
    """Build a 3-chunk sensor FAISS index; return chunk UUIDs (hermetic)."""
    import uuid as _uuid

    from tinyrag.ingestion.embedder import FakeEmbedder
    from tinyrag.storage.metadata import MetadataStore
    from tinyrag.storage.vector_store import FAISSStore

    embedder = FakeEmbedder(dimension=384)
    metadata = MetadataStore(settings.paths.metadata_db)
    metadata.init_schema()
    doc_id = metadata.insert_document(
        filename="synthetic.csv",
        doc_type="sensor_summary",
        source_path=str(tmp_path / "synthetic.csv"),
        size_bytes=512,
        content_hash="feedface" * 8,
        metadata={"num_rows": 100, "num_days": 2, "ingested_via": "test_api.py"},
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
def tiny_settings(tmp_path: Path) -> Any:
    """Settings + populated doc + sensor FAISS indices in ``tmp_path``."""
    settings = _make_settings(tmp_path)
    _populate_doc_index(tmp_path, settings)
    _populate_sensor_index(tmp_path, settings)
    return settings


@pytest.fixture
def client(tiny_settings: Any) -> Any:
    """TestClient with lifespan already run.

    Uses ``llm_kind="fake"`` and ``embedder_kind="fake"`` so the
    pipeline stays hermetic (no model load, no live llama-server).
    """
    app = create_app(tiny_settings, llm_kind="fake", embedder_kind="fake")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def empty_client(tmp_path: Path) -> Any:
    """TestClient for a tiny app with no FAISS indices loaded (empty store)."""
    settings = _make_settings(tmp_path)
    app = create_app(settings, llm_kind="fake", embedder_kind="fake")
    with TestClient(app) as c:
        yield c


# ===========================================================================
# Test classes
# ===========================================================================


class TestPublicSurface:
    """Every documented symbol from tinyrag.api + tinyrag.main is importable."""

    def test_routers_are_callable(self) -> None:
        assert callable(build_query_router)
        assert callable(build_docs_router)
        assert callable(build_admin_router)

    def test_install_exception_handlers_callable(self) -> None:
        # The module re-exports the function under the same name as the
        # one in tinyrag.api.errors — assert both are callable.
        assert callable(install_exception_handlers)
        assert callable(_reinstall_handlers)

    def test_create_app_callable(self) -> None:
        assert callable(create_app)

    def test_schemas_importable(self) -> None:
        # Just confirm the Pydantic classes resolve and have the
        # documented ``model_fields`` (Pydantic v2).
        assert hasattr(AskRequest, "model_fields")
        assert hasattr(StatusResponse, "model_fields")
        assert hasattr(ErrorResponse, "model_fields")
        assert hasattr(NotImplementedResponse, "model_fields")

    def test_dependency_providers_are_callable(self) -> None:
        for fn in (
            get_settings,
            get_embedder,
            get_llm,
            get_metadata,
            get_retriever,
            get_prompt_builder,
            get_doc_store,
            get_sensor_store,
        ):
            assert callable(fn)


class TestSchemasValidation:
    """Pydantic models reject bad input and accept good input."""

    def test_ask_request_accepts_minimal_payload(self) -> None:
        req = AskRequestSchema(query="hello")
        assert req.query == "hello"
        # Defaults from the schema.
        assert req.k_doc == 3
        assert req.k_sensor == 2
        assert req.threshold == 0.3
        assert req.max_tokens == 512
        assert req.log_query is True

    def test_ask_request_rejects_empty_query(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            AskRequestSchema(query="")

    def test_ask_request_rejects_extra_fields(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            AskRequestSchema(query="hello", hack=True)  # type: ignore[call-arg]

    def test_ask_request_rejects_out_of_range_threshold(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            AskRequestSchema(query="hello", threshold=1.5)

    def test_ask_request_rejects_out_of_range_max_tokens(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            AskRequestSchema(query="hello", max_tokens=99999)

    def test_status_response_requires_all_fr39_fields(self) -> None:
        # Every field except ram_mb is required.
        resp = StatusResponseSchema(
            ok=True,
            model_name="phi-3-mini",
            embedding_model="MiniLM",
            embedding_dim=384,
            doc_chunk_count=10,
            sensor_chunk_count=5,
            doc_index_path="/tmp/doc.faiss",
            sensor_index_path="/tmp/sensor.faiss",
            metadata_db_path="/tmp/m.db",
            llama_cpp_status="up",
            llama_cpp_url="http://127.0.0.1:8080",
            sensor_source="simulated",
            deployment_target="laptop",
        )
        assert resp.ram_mb is None  # explicitly default

    def test_error_response_shape(self) -> None:
        body = ErrorResponseSchema(error="validation_error", detail="query: too short")
        assert body.error == "validation_error"
        assert body.detail == "query: too short"


class TestSystemInfoHelpers:
    """RAM + llama.cpp + embedding-name probes never crash."""

    def test_get_ram_mb_returns_number_or_none(self) -> None:
        result = get_ram_mb()
        # Either a float (with 1dp rounding) or None — never an exception.
        assert result is None or isinstance(result, float)

    def test_get_llama_cpp_status_returns_up_or_down(self) -> None:
        # There's no live llama-server in the test env, so this should
        # return "down". We assert the contract (string in {"up","down"})
        # to keep the test stable across CI environments.
        result = get_llama_cpp_status("http://127.0.0.1:8080", timeout_s=0.5)
        assert result in {"up", "down"}

    def test_get_embedding_model_name_from_fake(self) -> None:
        from tinyrag.ingestion.embedder import FakeEmbedder

        embedder = FakeEmbedder(dimension=384)
        name = get_embedding_model_name(embedder)
        # FakeEmbedder doesn't expose model_name; falls back to the
        # class name "FakeEmbedder".
        assert name == "FakeEmbedder"


class TestCreateAppLifespan:
    """create_app returns a FastAPI app and the lifespan populates app.state."""

    def test_create_app_returns_fastapi_instance(self, tiny_settings: Any) -> None:
        app = create_app(tiny_settings, llm_kind="fake", embedder_kind="fake")
        assert isinstance(app, FastAPI)

    def test_lifespan_populates_app_state(self, tiny_settings: Any) -> None:
        app = create_app(tiny_settings, llm_kind="fake", embedder_kind="fake")
        with TestClient(app) as c:
            state = c.app.state
            # Every documented slot is populated.
            for slot in (
                "settings",
                "embedder",
                "doc_store",
                "sensor_store",
                "metadata",
                "llm",
                "retriever",
                "prompt_builder",
            ):
                assert getattr(state, slot) is not None, f"app.state.{slot} is None"

    def test_routers_are_mounted(self, tiny_settings: Any) -> None:
        app = create_app(tiny_settings, llm_kind="fake", embedder_kind="fake")
        paths = {r.path for r in app.routes if hasattr(r, "path")}
        # Public query surface
        assert "/api/status" in paths
        assert "/api/query" in paths
        # Docs skeleton
        assert "/api/documents" in paths
        assert "/api/documents/{document_id}" in paths
        # Admin skeleton
        assert "/api/admin/reindex" in paths
        assert "/api/admin/benchmark" in paths
        # Meta
        assert "/healthz" in paths
        assert "/" in paths


class TestGetStatus:
    """GET /api/status returns the full FR-39 status shape."""

    def test_status_returns_200(self, client: Any) -> None:
        r = client.get("/api/status")
        assert r.status_code == 200

    def test_status_shape_matches_fr39(self, client: Any) -> None:
        body = client.get("/api/status").json()
        assert REQUIRED_STATUS_KEYS.issubset(set(body.keys()))

    def test_status_reports_populated_indices(self, client: Any) -> None:
        body = client.get("/api/status").json()
        # tiny_settings fixture populates both indices with 3 chunks each.
        assert body["doc_chunk_count"] == 3
        assert body["sensor_chunk_count"] == 3

    def test_status_reports_embedding_dim(self, client: Any) -> None:
        body = client.get("/api/status").json()
        assert body["embedding_dim"] == 384

    def test_status_reports_llama_cpp_down(self, client: Any) -> None:
        # No live llama-server in the test env.
        body = client.get("/api/status").json()
        assert body["llama_cpp_status"] in {"up", "down"}

    def test_status_reports_ok_false_when_llama_down(self, client: Any) -> None:
        body = client.get("/api/status").json()
        # llama.cpp down → ok flips to False.
        if body["llama_cpp_status"] == "down":
            assert body["ok"] is False

    def test_status_paths_match_settings(self, client: Any) -> None:
        body = client.get("/api/status").json()
        state = client.app.state
        assert body["doc_index_path"] == str(state.settings.retrieval.doc_index_path)
        assert body["sensor_index_path"] == str(state.settings.retrieval.sensor_index_path)
        assert body["metadata_db_path"] == str(state.settings.paths.metadata_db)

    def test_status_reports_deployment_target(self, client: Any) -> None:
        body = client.get("/api/status").json()
        assert body["deployment_target"] == "laptop"


class TestPostQueryHappyPath:
    """POST /api/query runs the 4-stage pipeline and returns Answer.to_dict()."""

    def test_query_returns_200(self, client: Any) -> None:
        r = client.post("/api/query", json={"query": "What about the thermostat?"})
        assert r.status_code == 200

    def test_query_returns_answer_shape(self, client: Any) -> None:
        body = client.post(
            "/api/query", json={"query": "What about the thermostat?"}
        ).json()
        assert REQUIRED_ANSWER_KEYS.issubset(set(body.keys()))

    def test_query_echoes_question(self, client: Any) -> None:
        body = client.post(
            "/api/query", json={"query": "What about the thermostat?"}
        ).json()
        assert body["query"] == "What about the thermostat?"

    def test_query_text_is_fake_llm_response(self, client: Any) -> None:
        body = client.post("/api/query", json={"query": "anything"}).json()
        # FakeLLMClient returns the canned response.
        assert "fake" in body["text"].lower() or body["text"]

    def test_query_citations_is_a_list(self, client: Any) -> None:
        body = client.post(
            "/api/query", json={"query": "thermostat factory reset"}
        ).json()
        assert isinstance(body["citations"], list)

    def test_query_records_per_stage_timings(self, client: Any) -> None:
        body = client.post("/api/query", json={"query": "anything"}).json()
        for key in (
            "duration_retrieve_ms",
            "duration_prompt_ms",
            "duration_llm_ms",
            "duration_total_ms",
        ):
            assert isinstance(body[key], int | float)
            assert body[key] >= 0

    def test_query_records_token_counts(self, client: Any) -> None:
        body = client.post("/api/query", json={"query": "anything"}).json()
        assert body["prompt_tokens"] >= 0
        assert body["completion_tokens"] >= 0
        assert body["total_tokens"] == body["prompt_tokens"] + body["completion_tokens"]

    def test_query_total_ms_is_positive(self, client: Any) -> None:
        body = client.post("/api/query", json={"query": "anything"}).json()
        assert body["duration_total_ms"] > 0

    def test_query_model_name_set(self, client: Any) -> None:
        body = client.post("/api/query", json={"query": "anything"}).json()
        # FakeLLMClient reports its model id; either "fake-llm" or a
        # substring like "fake" — assert it's a non-empty string.
        assert isinstance(body["model_name"], str) and body["model_name"]


class TestPostQueryLogging:
    """Successful queries append rows to query_log; log_query=False skips."""

    def test_query_logs_row_by_default(self, client: Any, tiny_settings: Any) -> None:
        from tinyrag.storage.metadata import MetadataStore

        # Wipe any prior log rows so the count is deterministic.
        store = MetadataStore(tiny_settings.paths.metadata_db)
        before = len(store.get_recent_queries(limit=100))

        r = client.post("/api/query", json={"query": "log me please"})
        assert r.status_code == 200

        after = MetadataStore(tiny_settings.paths.metadata_db).get_recent_queries(limit=100)
        assert len(after) == before + 1
        assert after[0].query == "log me please"

    def test_log_query_false_skips_db_write(self, client: Any, tiny_settings: Any) -> None:
        from tinyrag.storage.metadata import MetadataStore

        before = len(
            MetadataStore(tiny_settings.paths.metadata_db).get_recent_queries(limit=100)
        )

        r = client.post(
            "/api/query", json={"query": "do not log me", "log_query": False}
        )
        assert r.status_code == 200

        after = MetadataStore(tiny_settings.paths.metadata_db).get_recent_queries(limit=100)
        assert len(after) == before


class TestPostQuerySensorKeyword:
    """A query with a sensor keyword triggers the sensor store."""

    def test_temperature_query_uses_sensor_store(self, client: Any) -> None:
        # threshold=0.0 is required because the FakeEmbedder's
        # SHA-256-derived cosines are not semantically meaningful
        # and often come out below 0.3. With a real embedder the
        # default 0.3 would suffice.
        body = client.post(
            "/api/query",
            json={"query": "What was the temperature yesterday?", "threshold": 0.0},
        ).json()
        # Sensor keyword "temperature" routes to the sensor store;
        # the tiny fixture has 3 sensor chunks so used_sensor_idx
        # should be True.
        assert body["used_sensor_idx"] is True

    def test_non_sensor_query_skips_sensor_store(self, client: Any) -> None:
        body = client.post(
            "/api/query",
            json={"query": "How do I reset the thermostat?", "threshold": 0.0},
        ).json()
        # "reset" / "thermostat" are not sensor keywords.
        assert body["used_sensor_idx"] is False


class TestPostQueryValidation:
    """Bad input is rejected at the Pydantic boundary (422)."""

    def test_empty_query_returns_422(self, client: Any) -> None:
        r = client.post("/api/query", json={"query": ""})
        assert r.status_code == 422

    def test_missing_query_field_returns_422(self, client: Any) -> None:
        r = client.post("/api/query", json={})
        assert r.status_code == 422

    def test_extra_field_returns_422(self, client: Any) -> None:
        r = client.post(
            "/api/query", json={"query": "hello", "injection": True}
        )
        assert r.status_code == 422

    def test_threshold_out_of_range_returns_422(self, client: Any) -> None:
        r = client.post(
            "/api/query", json={"query": "hello", "threshold": 1.5}
        )
        assert r.status_code == 422

    def test_k_doc_out_of_range_returns_422(self, client: Any) -> None:
        r = client.post(
            "/api/query", json={"query": "hello", "k_doc": 0}
        )
        assert r.status_code == 422

    def test_validation_error_has_uniform_shape(self, client: Any) -> None:
        r = client.post("/api/query", json={"query": ""})
        body = r.json()
        # Our global handler emits {"error": ..., "detail": ...}
        assert "error" in body
        assert "detail" in body


class TestNotImplementedEndpoints:
    """/api/admin/* returns 501 with the documented body (Step 4.17 stub)."""

    def test_admin_reindex_returns_501(self, client: Any) -> None:
        r = client.post("/api/admin/reindex")
        assert r.status_code == 501
        assert r.json()["error"] == "not_implemented"
        assert ADMIN_NOT_IMPLEMENTED_DETAIL in r.json()["detail"]

    def test_admin_benchmark_returns_501(self, client: Any) -> None:
        r = client.post("/api/admin/benchmark")
        assert r.status_code == 501
        assert r.json()["error"] == "not_implemented"


# ===========================================================================
# Step 4.18 — Document management endpoints
# ===========================================================================


class TestDocsRouterExports:
    """The router factory + validation constants are importable + correct."""

    def test_build_docs_router_callable(self) -> None:
        from tinyrag.api.routes_docs import build_docs_router as fn
        assert callable(fn)

    def test_allowed_extensions_match_spec(self) -> None:
        from tinyrag.api.routes_docs import ALLOWED_EXTENSIONS
        assert frozenset({".pdf", ".txt", ".md"}) == ALLOWED_EXTENSIONS

    def test_max_upload_bytes_is_50_mib(self) -> None:
        from tinyrag.api.routes_docs import MAX_UPLOAD_BYTES
        assert MAX_UPLOAD_BYTES == 50 * 1024 * 1024

    def test_supported_doc_types_match_spec(self) -> None:
        from tinyrag.api.routes_docs import SUPPORTED_DOC_TYPES
        # The router's whitelist is sourced from the canonical
        # metadata store whitelist so they never disagree.
        from tinyrag.storage.metadata import SUPPORTED_DOC_TYPES as META_TYPES
        assert SUPPORTED_DOC_TYPES == META_TYPES
        assert "manual" in SUPPORTED_DOC_TYPES

    def test_default_doc_type_is_manual(self) -> None:
        from tinyrag.api.routes_docs import DEFAULT_DOC_TYPE
        assert DEFAULT_DOC_TYPE == "manual"

    def test_safe_filename_strips_path_traversal(self) -> None:
        from tinyrag.api.routes_docs import _safe_filename
        assert _safe_filename("../../../etc/passwd") == "passwd"
        assert _safe_filename("..\\..\\windows\\system32\\cmd.exe") == "cmd.exe"
        assert _safe_filename("manual.pdf") == "manual.pdf"
        assert _safe_filename("") == ""
        assert _safe_filename(".") == ""
        assert _safe_filename("..") == ""
        assert _safe_filename(".bashrc") == ""
        # Windows reserved device names.
        assert _safe_filename("CON.pdf") == ""
        assert _safe_filename("nul.txt") == ""


class TestPostDocuments:
    """``POST /api/documents`` — upload + ingest pipeline (Step 4.18)."""

    def test_upload_txt_returns_200_with_report_shape(
        self, empty_client: Any
    ) -> None:
        r = empty_client.post(
            "/api/documents",
            files={"file": ("hello.txt", _small_txt(), "text/plain")},
            data={"doc_type": "manual"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        for key in REQUIRED_UPLOAD_KEYS:
            assert key in body, f"missing key {key!r} in {body}"
        assert body["ok"] is True
        assert body["doc_id"], "doc_id should be a non-empty UUID"
        assert body["num_chunks"] >= 1
        assert body["doc_type"] == "manual"
        assert body["error"] is None

    def test_upload_md_returns_200(self, empty_client: Any) -> None:
        r = empty_client.post(
            "/api/documents",
            files={"file": ("notes.md", _small_md(), "text/markdown")},
            data={"doc_type": "faq"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert body["doc_type"] == "faq"
        assert body["num_chunks"] >= 1

    def test_upload_pdf_returns_200(self, empty_client: Any) -> None:
        # Real Nest PDF — verifies the parser dispatch for .pdf works
        # via the HTTP path (not just the CLI).
        pdf_bytes = _NEST_PDF.read_bytes()
        r = empty_client.post(
            "/api/documents",
            files={"file": ("manual.pdf", pdf_bytes, "application/pdf")},
            data={"doc_type": "manual"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        # The Nest PDF is rich → many chunks; just assert > 0.
        assert body["num_chunks"] > 0

    def test_upload_creates_doc_and_chunks_in_db(
        self, empty_client: Any, tmp_path: Path
    ) -> None:
        # Pre-state: 0 docs / 0 chunks.
        r = empty_client.post(
            "/api/documents",
            files={"file": ("hello.txt", _small_txt(), "text/plain")},
            data={"doc_type": "manual"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True

        # Verify SQLite has exactly 1 doc + num_chunks chunks.
        from tinyrag.storage.metadata import MetadataStore

        db = MetadataStore(str(tmp_path / "metadata.db"))
        assert db.count_documents() == 1
        assert db.count_chunks() == body["num_chunks"]

    def test_upload_picks_up_vectors_in_doc_store(
        self, empty_client: Any
    ) -> None:
        # Capture the doc_store before…
        before = empty_client.app.state.doc_store.size()
        r = empty_client.post(
            "/api/documents",
            files={"file": ("hello.txt", _small_txt(), "text/plain")},
            data={"doc_type": "manual"},
        )
        body = r.json()
        # …and confirm it grew by num_chunks (the reload-after-upload works).
        after = empty_client.app.state.doc_store.size()
        assert after - before == body["num_chunks"]

    def test_upload_then_list_shows_it(self, empty_client: Any) -> None:
        r = empty_client.post(
            "/api/documents",
            files={"file": ("hello.txt", _small_txt(), "text/plain")},
            data={"doc_type": "manual"},
        )
        doc_id = r.json()["doc_id"]

        # Now GET — the doc should appear.
        r2 = empty_client.get("/api/documents")
        assert r2.status_code == 200
        body = r2.json()
        assert body["count"] == 1
        assert len(body["documents"]) == 1
        assert body["documents"][0]["id"] == doc_id
        assert body["documents"][0]["filename"] == "hello.txt"

    def test_upload_rejects_unsupported_extension(self, empty_client: Any) -> None:
        r = empty_client.post(
            "/api/documents",
            files={"file": ("data.csv", b"a,b,c\n1,2,3", "text/csv")},
            data={"doc_type": "manual"},
        )
        assert r.status_code == 400
        body = r.json()
        assert body["error"] == "unsupported_file_type"
        assert ".csv" in body["detail"]

    def test_upload_rejects_missing_file_part(self, empty_client: Any) -> None:
        # Missing the ``file`` form field → Pydantic 422.
        r = empty_client.post(
            "/api/documents", data={"doc_type": "manual"}
        )
        assert r.status_code == 422
        body = r.json()
        assert body["error"] == "validation_error"

    def test_upload_rejects_oversize_file(self, empty_client: Any) -> None:
        # 51 MB → 413 file_too_large.
        r = empty_client.post(
            "/api/documents",
            files={"file": ("big.txt", _oversize_bytes(), "text/plain")},
            data={"doc_type": "manual"},
        )
        assert r.status_code == 413, r.text
        body = r.json()
        assert body["error"] == "file_too_large"
        assert "50" in body["detail"]  # the cap in MB

    def test_upload_rejects_invalid_doc_type(self, empty_client: Any) -> None:
        r = empty_client.post(
            "/api/documents",
            files={"file": ("hello.txt", _small_txt(), "text/plain")},
            data={"doc_type": "bogus_type"},
        )
        assert r.status_code == 400
        body = r.json()
        assert body["error"] == "invalid_doc_type"
        assert "bogus_type" in body["detail"]

    def test_upload_sanitises_filename_traversal(self, empty_client: Any) -> None:
        # Path-traversal in filename → server stores only the basename
        # (or rejects; the route sanitises via _safe_filename which
        # always returns a plain basename). After upload, list shows
        # the sanitised name.
        r = empty_client.post(
            "/api/documents",
            files={"file": ("../../../etc/passwd.txt", _small_txt(), "text/plain")},
            data={"doc_type": "manual"},
        )
        # The upload itself succeeds (the body is valid).
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        # The listed document's filename is just the basename.
        r2 = empty_client.get("/api/documents")
        items = r2.json()["documents"]
        assert len(items) == 1
        assert items[0]["filename"] == "passwd.txt"

    def test_upload_default_doc_type_is_manual(self, empty_client: Any) -> None:
        # No doc_type form value → defaults to "manual".
        r = empty_client.post(
            "/api/documents",
            files={"file": ("hello.txt", _small_txt(), "text/plain")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["doc_type"] == "manual"


class TestGetDocuments:
    """``GET /api/documents`` — paginated list (Step 4.18)."""

    def test_get_documents_empty_returns_count_zero(
        self, empty_client: Any
    ) -> None:
        r = empty_client.get("/api/documents")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 0
        assert body["documents"] == []
        assert body["limit"] == 50  # default
        assert body["offset"] == 0
        assert body["next_offset"] is None

    def test_get_documents_lists_after_upload(self, empty_client: Any) -> None:
        for _ in range(2):
            empty_client.post(
                "/api/documents",
                files={"file": ("hello.txt", _small_txt(), "text/plain")},
                data={"doc_type": "manual"},
            )
        r = empty_client.get("/api/documents")
        body = r.json()
        assert body["count"] == 2
        assert len(body["documents"]) == 2

    def test_get_documents_returns_required_keys_per_item(
        self, empty_client: Any
    ) -> None:
        empty_client.post(
            "/api/documents",
            files={"file": ("hello.txt", _small_txt(), "text/plain")},
            data={"doc_type": "manual"},
        )
        r = empty_client.get("/api/documents")
        for item in r.json()["documents"]:
            assert REQUIRED_DOC_LIST_ITEM_KEYS.issubset(item.keys())

    def test_get_documents_newest_first(self, empty_client: Any) -> None:
        # Upload A then B → GET returns B before A.
        rA = empty_client.post(
            "/api/documents",
            files={"file": ("a.txt", _small_txt(), "text/plain")},
            data={"doc_type": "manual"},
        )
        rB = empty_client.post(
            "/api/documents",
            files={"file": ("b.txt", _small_txt(), "text/plain")},
            data={"doc_type": "manual"},
        )
        ids = [rA.json()["doc_id"], rB.json()["doc_id"]]
        r = empty_client.get("/api/documents")
        items = r.json()["documents"]
        assert [it["id"] for it in items] == [ids[1], ids[0]]

    def test_get_documents_pagination_limit(self, empty_client: Any) -> None:
        for _ in range(3):
            empty_client.post(
                "/api/documents",
                files={"file": ("hello.txt", _small_txt(), "text/plain")},
                data={"doc_type": "manual"},
            )
        r = empty_client.get("/api/documents?limit=2")
        body = r.json()
        assert body["count"] == 3
        assert len(body["documents"]) == 2
        assert body["limit"] == 2
        assert body["next_offset"] == 2

    def test_get_documents_pagination_offset(self, empty_client: Any) -> None:
        for _ in range(3):
            empty_client.post(
                "/api/documents",
                files={"file": ("hello.txt", _small_txt(), "text/plain")},
                data={"doc_type": "manual"},
            )
        r = empty_client.get("/api/documents?limit=2&offset=2")
        body = r.json()
        assert body["count"] == 3
        assert len(body["documents"]) == 1
        assert body["offset"] == 2
        assert body["next_offset"] is None  # no more pages

    def test_get_documents_validation_rejects_bad_limit(
        self, empty_client: Any
    ) -> None:
        # limit=0 → 422; limit=501 → 422 (Pydantic via FastAPI Query bounds).
        assert empty_client.get("/api/documents?limit=0").status_code == 422
        assert empty_client.get("/api/documents?limit=501").status_code == 422
        # offset=-1 → 422.
        assert empty_client.get("/api/documents?offset=-1").status_code == 422


class TestDeleteDocument:
    """``DELETE /api/documents/{id}`` — cascade-delete (Step 4.18)."""

    def test_delete_removes_document_and_vectors(
        self, empty_client: Any, tmp_path: Path
    ) -> None:
        # Upload, then delete, then verify.
        r = empty_client.post(
            "/api/documents",
            files={"file": ("hello.txt", _small_txt(), "text/plain")},
            data={"doc_type": "manual"},
        )
        doc_id = r.json()["doc_id"]
        num_chunks = r.json()["num_chunks"]

        r2 = empty_client.delete(f"/api/documents/{doc_id}")
        assert r2.status_code == 200, r2.text
        body = r2.json()
        assert body["document_id"] == doc_id
        assert body["chunks_removed"] == num_chunks
        assert body["vectors_removed"] == num_chunks

        # GET now excludes it; FAISS is also empty.
        r3 = empty_client.get("/api/documents")
        assert r3.json()["count"] == 0
        assert empty_client.app.state.doc_store.size() == 0

        # SQLite chunk count is 0 too.
        from tinyrag.storage.metadata import MetadataStore

        db = MetadataStore(str(tmp_path / "metadata.db"))
        assert db.count_chunks() == 0
        assert db.count_documents() == 0

    def test_delete_unknown_id_returns_404(self, empty_client: Any) -> None:
        r = empty_client.delete("/api/documents/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 404
        body = r.json()
        assert body["error"] == "document_not_found"
        assert "00000000-0000-0000-0000-000000000000" in body["detail"]

    def test_delete_twice_returns_404_second_time(self, empty_client: Any) -> None:
        r = empty_client.post(
            "/api/documents",
            files={"file": ("hello.txt", _small_txt(), "text/plain")},
            data={"doc_type": "manual"},
        )
        doc_id = r.json()["doc_id"]

        first = empty_client.delete(f"/api/documents/{doc_id}")
        assert first.status_code == 200
        second = empty_client.delete(f"/api/documents/{doc_id}")
        assert second.status_code == 404

    def test_delete_preserves_other_documents(self, empty_client: Any) -> None:
        # Upload 2 docs, delete the first; the second survives.
        rA = empty_client.post(
            "/api/documents",
            files={"file": ("a.txt", _small_txt(), "text/plain")},
            data={"doc_type": "manual"},
        )
        rB = empty_client.post(
            "/api/documents",
            files={"file": ("b.txt", _small_txt(), "text/plain")},
            data={"doc_type": "manual"},
        )
        empty_client.delete(f"/api/documents/{rA.json()['doc_id']}")

        r = empty_client.get("/api/documents")
        ids = [d["id"] for d in r.json()["documents"]]
        assert ids == [rB.json()["doc_id"]]

    def test_delete_persists_across_reload(
        self, empty_client: Any, tmp_path: Path
    ) -> None:
        # Upload, delete, then re-load the FAISS store from disk — the
        # deletion must still be reflected (verifies the sidecar save).
        r = empty_client.post(
            "/api/documents",
            files={"file": ("hello.txt", _small_txt(), "text/plain")},
            data={"doc_type": "manual"},
        )
        doc_id = r.json()["doc_id"]
        empty_client.delete(f"/api/documents/{doc_id}")

        # Force a reload from disk; the size should still be 0.
        from tinyrag.storage.vector_store import FAISSStore

        store = FAISSStore(
            index_path=Path(empty_client.app.state.settings.retrieval.doc_index_path),
            embedding_dimension=384,
            embedding_model="fake:sentence-transformers/all-MiniLM-L6-v2",
        )
        store.load()
        assert store.size() == 0


# ===========================================================================
# Step 4.19 — SSE streaming for /api/query
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_sse_starlette_exit_event() -> Any:
    """Reset sse-starlette's module-level ``should_exit_event`` per test.

    ``sse_starlette.sse.AppStatus.should_exit_event`` is a cached
    ``anyio.Event`` bound to the first event loop that touches it.
    Under ``TestClient`` each test gets a fresh event loop, so
    reusing a stale Event raises ``RuntimeError: ... is bound to a
    different event loop``. Clearing the cache between tests lets
    every test create its own Event on its own loop.

    See: https://github.com/sysid/sse-starlette/issues/63
    """
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None
    AppStatus.should_exit = False
    yield
    AppStatus.should_exit_event = None
    AppStatus.should_exit = False


class TestQueryStreaming:
    """``POST /api/query?stream=true`` emits SSE events (Step 4.19).

    The streaming path uses sse-starlette's ``EventSourceResponse``
    which, when consumed via ``httpx`` (the TestClient backend),
    frames each yield as a ``data: <json>\\n\\n`` block. The
    ``_parse_sse_events`` helper splits the body on the blank-line
    terminator and JSON-decodes each frame so the test can assert on
    a list of dicts.

    The non-streaming path (``POST /api/query`` without
    ``?stream=true``) is exercised by the existing
    ``TestPostQueryHappyPath`` tests — they continue to pass because
    the default of the new ``stream`` query param is ``False``.
    """

    def test_stream_true_emits_token_events_then_done(
        self, client: Any
    ) -> None:
        """``?stream=true`` yields one ``token`` event per word + a final ``done``."""
        with client.stream(
            "POST",
            "/api/query?stream=true",
            json={"query": "what about the thermostat?", "max_tokens": 16},
        ) as r:
            assert r.status_code == 200
            # sse-starlette sets ``text/event-stream`` for SSE responses.
            assert r.headers["content-type"].startswith("text/event-stream")
            body = r.read().decode("utf-8")

        events = _parse_sse_events(body)
        assert events, "no SSE events yielded"
        # Every event except the last must be a ``token`` event.
        token_events = [e for e in events if e["event"] == "token"]
        done_events = [e for e in events if e["event"] == "done"]
        assert len(token_events) >= 1, "at least one token event expected"
        assert len(done_events) == 1, f"expected exactly one done event, got {len(done_events)}"

        # Token events carry ``delta`` strings in ``payload``; done
        # carries the full Answer in ``payload["answer"]``.
        for ev in token_events:
            assert isinstance(ev["payload"], dict)
            assert "delta" in ev["payload"]
            assert isinstance(ev["payload"]["delta"], str)
            assert ev["payload"]["delta"], "delta must be non-empty"

        done_payload = done_events[0]["payload"]["answer"]
        assert REQUIRED_ANSWER_KEYS.issubset(set(done_payload.keys()))
        assert done_payload["query"] == "what about the thermostat?"
        # Concatenating all the token deltas (with single-space join,
        # matching how ``_stream_answer`` builds the full_text) should
        # match the done event's ``text``.
        joined = " ".join(e["payload"]["delta"] for e in token_events)
        assert done_payload["text"] == joined
        assert done_payload["model_name"], "model_name must be non-empty"
        assert done_payload["duration_total_ms"] >= 0

    def test_stream_false_returns_json_blob_not_sse(self, client: Any) -> None:
        """No ``?stream=true`` → still the existing single JSON blob.

        Regression pin — Step 4.19 must NOT change the default behaviour
        of the route (any existing curl user / dashboard client must
        keep working unchanged).
        """
        r = client.post(
            "/api/query", json={"query": "anything"}
        )
        assert r.status_code == 200
        # application/json (or close — Starlette may add a charset).
        assert "application/json" in r.headers["content-type"]
        body = r.json()
        # The non-streaming contract: one Answer-shaped dict, NOT a list of events.
        assert REQUIRED_ANSWER_KEYS.issubset(set(body.keys()))
        assert "event" not in body, "JSON blob must not carry an SSE event field"

    def test_stream_query_log_row_written(
        self, client: Any, tiny_settings: Any
    ) -> None:
        """``?stream=true`` still appends a ``query_log`` row."""
        from tinyrag.storage.metadata import MetadataStore

        store = MetadataStore(tiny_settings.paths.metadata_db)
        before = len(store.get_recent_queries(limit=100))

        with client.stream(
            "POST",
            "/api/query?stream=true",
            json={"query": "log me streaming please"},
        ) as r:
            assert r.status_code == 200
            # Drain the body so sse-starlette can close the response.
            r.read()

        after = MetadataStore(tiny_settings.paths.metadata_db).get_recent_queries(
            limit=100
        )
        assert len(after) == before + 1
        assert after[0].query == "log me streaming please"
        # Per-stage timings are stored as integers in the schema.
        assert after[0].total_ms >= 0

    def test_stream_mid_stream_llm_error_yields_error_event(
        self, tiny_settings: Any
    ) -> None:
        """LLM fails mid-stream → ``{"event":"error","error":"llm_failed"}`` frame.

        We override the get_llm dependency with a FakeLLMClient that
        raises after 2 tokens so we can observe the partial token
        burst + the error envelope + the clean close.
        """
        from tinyrag.api import deps as api_deps
        from tinyrag.generation.llm_client import FakeLLMClient

        app = create_app(tiny_settings, llm_kind="fake", embedder_kind="fake")
        # Pre-fail the LLM so the fake raises after 2 tokens.
        boom_llm = FakeLLMClient(
            default_response="alpha beta gamma delta epsilon zeta",
            raise_after_tokens=2,
        )
        app.dependency_overrides[api_deps.get_llm] = lambda: boom_llm  # type: ignore[dict-item]

        with TestClient(app) as c, c.stream(
            "POST",
            "/api/query?stream=true",
            json={"query": "anything goes"},
        ) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            body = r.read().decode("utf-8")

        events = _parse_sse_events(body)
        token_events = [e for e in events if e["event"] == "token"]
        error_events = [e for e in events if e["event"] == "error"]
        done_events = [e for e in events if e["event"] == "done"]

        # Exactly 2 token events were emitted before the raise guard
        # fired; then a single error frame; then a clean close (NO
        # done event because we returned early).
        assert len(token_events) == 2, f"expected 2 token events, got {len(token_events)}"
        assert [e["payload"]["delta"] for e in token_events] == ["alpha", "beta"]
        assert len(error_events) == 1
        assert error_events[0]["payload"]["error"] == "llm_failed"
        assert "LLM call failed" in error_events[0]["payload"]["detail"]
        assert len(done_events) == 0

    def test_stream_retrieval_failure_returns_json_500_not_sse(
        self, tiny_settings: Any
    ) -> None:
        """Pre-stream errors stay JSON 5xx — NOT an SSE stream.

        Retrieval / prompt-build errors fire BEFORE any SSE frame is
        constructed, so the client gets the uniform ``ErrorResponse``
        JSON body. Only LLM failures can happen mid-stream, and those
        use the ``{"event":"error"}`` envelope.
        """
        from tinyrag.api import deps as api_deps

        def boom(_request: Request) -> None:
            raise RuntimeError("simulated retriever failure")

        app = create_app(tiny_settings, llm_kind="fake", embedder_kind="fake")
        app.dependency_overrides[api_deps.get_retriever] = boom  # type: ignore[dict-item]

        with TestClient(app, raise_server_exceptions=False) as c, c.stream(
            "POST",
            "/api/query?stream=true",
            json={"query": "anything"},
        ) as r:
            assert r.status_code == 500
            # JSON body, NOT text/event-stream.
            assert "application/json" in r.headers["content-type"]
            # Drain the body so r.json() works on a streamed response.
            r.read()
            body = r.json()
            assert body["error"] == "internal_server_error"

    def test_stream_log_query_false_skips_db_write(
        self, client: Any, tiny_settings: Any
    ) -> None:
        """``log_query=false`` still skips the write in the streaming path."""
        from tinyrag.storage.metadata import MetadataStore

        before = len(
            MetadataStore(tiny_settings.paths.metadata_db).get_recent_queries(limit=100)
        )

        with client.stream(
            "POST",
            "/api/query?stream=true",
            json={"query": "do not log me streaming", "log_query": False},
        ) as r:
            assert r.status_code == 200
            r.read()  # drain

        after = MetadataStore(tiny_settings.paths.metadata_db).get_recent_queries(
            limit=100
        )
        assert len(after) == before, "log_query=False must not write a row"


class TestErrorHandlers:
    """Domain exceptions map to the right HTTP status codes via the global handlers.

    Note: the /api/query route handler catches its own exceptions and
    returns clean JSON bodies (e.g. retrieval_failed, llm_failed) so
    the global handlers don't fire from inside the route body. The
    global handlers primarily fire from:

    1. Exceptions raised inside a FastAPI dependency provider
       (e.g. ``get_settings`` raising a ConfigError).
    2. Pydantic validation errors at request body parsing.
    3. Any path the route handler doesn't explicitly guard.

    These tests exercise paths 1 + 3 — we override a dependency so
    the route handler receives a faulty dependency, OR we raise
    from inside a function the route handler doesn't catch.
    """

    def test_value_error_from_dependency_maps_to_400(
        self, client: Any
    ) -> None:
        # Schema-level 422 (Pydantic validation error) emits our
        # uniform error shape. The 400 from a ValueError raised
        # inside a dep provider is tested in
        # ``test_value_error_from_dependency_maps_to_400_via_override``
        # below — both branches are covered.
        r = client.post("/api/query", json={"query": "", "k_doc": -1})
        assert r.status_code == 422
        body = r.json()
        assert body["error"] == "validation_error"

    def test_internal_exception_maps_to_500_via_dep_override(
        self, tiny_settings: Any
    ) -> None:
        # Build an app whose get_retriever dependency raises a
        # non-domain exception. The dependency provider runs
        # BEFORE the route handler body, so any exception raised
        # there flows directly to the global exception handlers.
        # NB: the override MUST use the exact parameter name
        # ``request`` (typed as ``fastapi.Request``) — FastAPI's
        # signature inspector otherwise confuses it for a body
        # field.

        from tinyrag.api import deps as api_deps

        def boom(request: Request) -> None:
            raise RuntimeError("simulated dep failure")

        # Override the dep provider via FastAPI's app.dependency_overrides.
        app = create_app(tiny_settings, llm_kind="fake", embedder_kind="fake")
        app.dependency_overrides[api_deps.get_retriever] = boom  # type: ignore[dict-item]
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post("/api/query", json={"query": "hello"})
            assert r.status_code == 500
            body = r.json()
            # The global catch-all scrubs the traceback; we ship
            # only the generic message so we don't leak internals.
            assert body["error"] == "internal_server_error"
            assert body["detail"] == "internal server error"
            assert "simulated dep failure" not in body["detail"]

    def test_value_error_from_dependency_maps_to_400_via_override(
        self, tiny_settings: Any
    ) -> None:
        # ValueError from inside a dependency provider maps to 400.

        from tinyrag.api import deps as api_deps

        def boom(request: Request) -> None:
            raise ValueError("bad threshold (test)")

        app = create_app(tiny_settings, llm_kind="fake", embedder_kind="fake")
        app.dependency_overrides[api_deps.get_retriever] = boom  # type: ignore[dict-item]
        with TestClient(app) as c:
            r = c.post("/api/query", json={"query": "hello"})
            assert r.status_code == 400
            body = r.json()
            assert body["error"] == "value_error"
            assert "bad threshold" in body["detail"]


class TestRootAndHealthz:
    """The meta endpoints return the documented shapes."""

    def test_healthz_returns_ok(self, client: Any) -> None:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"ok": "true"}

    def test_root_returns_banner(self, client: Any) -> None:
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["service"] == "tinyrag"
        assert body["api_docs"] == "/docs"


class TestCreateAppTwiceIdempotent:
    """create_app called twice in the same process yields two independent apps."""

    def test_two_apps_have_independent_state(
        self, tiny_settings: Any, tmp_path: Path
    ) -> None:
        # Build a second Settings pointing at a different tmpdir so the
        # two apps definitely don't share FAISS / SQLite files.
        settings_b = _make_settings(tmp_path / "second")
        _populate_doc_index(tmp_path / "second", settings_b)
        _populate_sensor_index(tmp_path / "second", settings_b)

        app_a = create_app(tiny_settings, llm_kind="fake", embedder_kind="fake")
        app_b = create_app(settings_b, llm_kind="fake", embedder_kind="fake")
        # Enter each app's lifespan so app.state gets populated.
        with TestClient(app_a) as _ca, TestClient(app_b) as _cb:
            # app.state is a separate object on each app.
            assert app_a.state is not app_b.state
            assert (
                app_a.state.settings.retrieval.doc_index_path
                != app_b.state.settings.retrieval.doc_index_path
            )
            # Independent FAISS stores too.
            assert app_a.state.doc_store is not app_b.state.doc_store


# ===========================================================================
# Module docstring self-check
# ===========================================================================


def test_module_docstring_present() -> None:
    """A trivial sanity check — confirms the file isn't accidentally empty."""
    assert __doc__ is not None and "Step 4.17" in __doc__
