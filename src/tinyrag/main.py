"""Composition root — ``create_app(settings) -> FastAPI``.

This module is the **single place** in the codebase that imports
concrete classes (:class:`SentenceTransformerEmbedder`,
:class:`FAISSStore`, :class:`MetadataStore`,
:class:`LlamaCppClient`). Every other module imports from
:mod:`tinyrag.core` + :mod:`tinyrag.generation` + the
``Protocol``-typed interfaces — they don't know whether they're
running under uvicorn or as a CLI script.

Why a factory, not a module-level ``app``?
------------------------------------------
A module-level ``app = FastAPI(...)`` would:

- Lock the embedder + LLM to the values in ``config.yaml`` at
  import time. Tests can't swap them for fakes without monkey-
  patching the module.
- Make ``uvicorn tinyrag.main:app`` work but ``uvicorn
  tinyrag.main:create_app()`` impossible (uvicorn expects an
  instance, not a factory).
- Couple the process lifetime to the app lifetime — a future
  test that wants a *second* app in the same process (e.g.
  the Step 4.20 portability self-test) would clobber the first
  one's state.

A factory (``create_app(settings)``) sidesteps all three:

- Tests call ``create_app(_tiny_settings(tmp_path))`` with a
  tmpdir-backed :class:`Settings` and FakeLLM/FakeEmbedder.
- The lifespan handler is bound at construction time so each
  factory call gets its own lifespan.
- Two factories in the same process get two independent
  ``app.state``s.

Lifespan
--------
The lifespan handler in :func:`create_app` is responsible for:

1. **Building** every singleton (embedder, both FAISS stores,
   metadata store, LLM, retriever, prompt builder) and stashing
   it on ``app.state``.
2. **Loading** the FAISS indices from disk (cheap if they're
   already in memory; cold start is the slow path).
3. **Initialising** the SQLite schema (idempotent — ``CREATE
   TABLE IF NOT EXISTS``).
4. On shutdown, **saving** the FAISS stores so any new vectors
   are flushed to disk.

If any step fails during startup the lifespan raises and
uvicorn refuses to bind the port — a 503 "service unavailable"
is much clearer than a series of 500s on the first request.

CLI vs library use
------------------
This module is importable as a library (``from tinyrag.main
import create_app``) but the canonical entry point is
``uvicorn tinyrag.main:app --host 127.0.0.1 --port 8000``. The
``app`` symbol is provided as a thin wrapper that reads
``config.yaml`` once at import time — see the bottom of this
file.

Web UI (Step 4.21)
------------------
``create_app`` also wires the **chat web UI**:

- ``GET /`` renders ``ui/templates/index.html`` (Jinja2) — the
  full chat page with topbar, message history, composer form,
  and footer.
- ``app.mount("/static", StaticFiles(directory=ui/static))``
  serves ``ui/static/chat.js`` and ``ui/static/style.css``
  directly to the browser.

Both the templates dir and the static dir are resolved relative
to ``config.yaml``'s ``project_root`` (via
``settings.project_root()``) so a packaged install + a clone
checked out at ``/srv/tinyrag`` both find ``ui/`` correctly.
Falls back to ``<repo>/ui`` if the project root can't be
determined.

Location: ``src/tinyrag/main.py``
"""

from __future__ import annotations

from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from tinyrag.api.errors import install_exception_handlers
from tinyrag.api.routes_admin import build_admin_router
from tinyrag.api.routes_docs import build_docs_router
from tinyrag.api.routes_query import build_query_router
from tinyrag.config import Settings, load_settings
from tinyrag.core.prompt_builder import PromptBuilder
from tinyrag.core.retriever import Retriever, adapt_metadata_store
from tinyrag.generation.llm_client import (
    LlamaCppClient,
    LLMClient,
)
from tinyrag.ingestion.embedder import (
    EmbeddingModel,
    FakeEmbedder,
    SentenceTransformerEmbedder,
)
from tinyrag.observability.logger import (
    configure_logging,
    get_logger,
)
from tinyrag.storage.metadata import MetadataStore
from tinyrag.storage.vector_store import DEFAULT_EMBEDDING_DIMENSION, FAISSStore

if TYPE_CHECKING:
    pass


_log = get_logger(__name__)


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------


def create_app(
    settings: Settings | None = None,
    *,
    llm_kind: str = "real",
    embedder_kind: str = "real",
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
) -> FastAPI:
    """Build a fully-wired :class:`FastAPI` app.

    Parameters
    ----------
    settings:
        The typed :class:`tinyrag.config.Settings`. If ``None``
        (the default), the factory calls :func:`load_settings()`
        which reads ``config.yaml`` from the current working
        directory. Pass an explicit value from tests.
    llm_kind:
        ``"real"`` → :class:`LlamaCppClient` against
        ``settings.llm.server_url``.
        ``"fake"`` → :class:`FakeLLMClient`. Tests use this so
        the suite stays hermetic (no live llama-server required).
    embedder_kind:
        ``"real"`` → :class:`SentenceTransformerEmbedder` with
        the configured model (lazy-loaded on first ``.embed()``).
        ``"fake"`` → :class:`FakeEmbedder` at dimension 384. Tests
        that built the FAISS index with FakeEmbedder pass
        ``"fake"`` so the query and chunks live in the same
        vector space.
    embedding_dimension:
        The dimensionality used to create the FAISS stores if
        they don't already exist on disk. Defaults to
        :data:`tinyrag.storage.vector_store.DEFAULT_EMBEDDING_DIMENSION`
        (384 — MiniLM). Tests may override to use a smaller
        dim with the FakeEmbedder for speed.

    Returns
    -------
    FastAPI
        A fully-wired app. Lifespan has NOT been run yet (FastAPI
        runs it on first request). To trigger it manually, use
        ``with app.router.lifespan_context(app): ...`` or just
        make a request.
    """
    if settings is None:
        settings = load_settings()

    # Configure the structured logger once per process. Idempotent —
    # if create_app is called twice (e.g. in the test suite),
    # the second call replaces the dictConfig but the result is
    # the same. Anchored to the project's root so log files land
    # next to the config rather than in whatever CWD uvicorn
    # happens to start in. Tests that build a Settings by hand
    # (without load_settings) get a CWD-anchored logger — still
    # fine because the test environment usually disables file
    # logging via settings.logging.file = "".
    try:
        project_root = settings.project_root()
    except RuntimeError:
        project_root = None
    configure_logging(settings.logging, project_root=project_root)

    # --- The lifespan: builds singletons, loads from disk, persists on exit.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _log.info(
            "app_starting",
            llm_kind=llm_kind,
            embedder_kind=embedder_kind,
            deployment=settings.deployment.target.value
            if hasattr(settings.deployment.target, "value")
            else str(settings.deployment.target),
        )

        # 1. Build embedder (lazy — model is only loaded on first .embed())
        embedder = _build_embedder(settings, kind=embedder_kind)

        # 2. Build both FAISS stores. We pass the embedder's dim
        #    so the freshly-created index matches the chunks the
        #    user already ingested (or will ingest).
        doc_store = FAISSStore(
            index_path=settings.retrieval.doc_index_path,
            embedding_dimension=embedding_dimension,
            embedding_model=settings.embedding.model_name,
        )
        sensor_store = FAISSStore(
            index_path=settings.retrieval.sensor_index_path,
            embedding_dimension=embedding_dimension,
            embedding_model=settings.embedding.model_name,
        )
        # load() is idempotent — no-op if the .faiss file is missing
        # (first run) or already loaded.
        with suppress(FileNotFoundError):  # First run; will be created on first .add()
            doc_store.load()
        with suppress(FileNotFoundError):
            sensor_store.load()

        # 3. Build + initialise the metadata store. init_schema is
        #    idempotent (CREATE TABLE IF NOT EXISTS).
        metadata = MetadataStore(settings.paths.metadata_db)
        metadata.init_schema()

        # 4. Build the LLM. Real mode does not open a connection
        #    until the first .generate() call (matches Step 4.10).
        llm = _build_llm(settings, kind=llm_kind)

        # 5. Build the retriever (composes embedder + both stores
        #    + the metadata accessor). The default threshold comes
        #    from config so the dashboard and CLI see the same
        #    behaviour; tests override via build_retriever().
        retriever = Retriever(
            embedder=embedder,
            doc_store=doc_store,
            sensor_store=sensor_store,
            metadata=adapt_metadata_store(metadata),
            default_threshold=settings.retrieval.similarity_threshold,
        )

        # 6. Build the prompt builder from the chunking settings
        #    so chunk-token-counts and prompt-token-counts are in
        #    the same units.
        prompt_builder = PromptBuilder.from_chunking_settings(
            settings.chunking,
        )

        # 7. Stash everything on app.state so the dependency
        #    providers (:mod:`tinyrag.api.deps`) can find them.
        app.state.settings = settings
        app.state.embedder = embedder
        app.state.doc_store = doc_store
        app.state.sensor_store = sensor_store
        app.state.metadata = metadata
        app.state.llm = llm
        app.state.retriever = retriever
        app.state.prompt_builder = prompt_builder

        _log.info("app_ready", model_name=llm.model_name)

        # ----- The actual lifespan body -----
        try:
            yield
        finally:
            _log.info("app_stopping")
            # Persist any in-memory state. We swallow exceptions
            # in shutdown so an intermittent disk error doesn't
            # mask a more important error that fired earlier in
            # the lifespan.
            try:
                doc_store.save()
            except Exception as exc:  # pragma: no cover
                _log.warning("doc_store_save_failed", error=str(exc))
            try:
                sensor_store.save()
            except Exception as exc:  # pragma: no cover
                _log.warning("sensor_store_save_failed", error=str(exc))

    # --- Build the app + wire the routers.
    app = FastAPI(
        title="TinyRAG",
        version="0.4.0",
        description=(
            "Edge IoT Retrieval-Augmented Generation. "
            "Phase 4 API surface."
        ),
        lifespan=lifespan,
    )

    # Global exception handlers (must run before routers are mounted
    # so a Pydantic validation error in a route body still triggers
    # our handler, not FastAPI's default).
    install_exception_handlers(app)

    # Routers — order doesn't matter for routing, but the tags
    # appear in /docs in this order.
    app.include_router(build_query_router())
    app.include_router(build_docs_router())
    app.include_router(build_admin_router())

    # Tiny health probe — returns {"ok": true} if the process is
    # up. Distinct from /api/status which probes every subsystem.
    # Useful for "is the container alive" checks (Kubernetes
    # liveness probe, the Step 4.20 portability self-test, etc.).
    @app.get("/healthz", tags=["meta"], summary="Process liveness probe.")
    async def healthz() -> dict[str, str]:
        return {"ok": "true"}

    # ---- Web UI (Step 4.21) ------------------------------------------
    #
    # The chat page is a Jinja2 template rendered at GET /. Static
    # assets (chat.js, style.css) are mounted at /static and fetched
    # directly by the browser — FastAPI's StaticFiles handles the
    # mime types + caching headers for us.
    #
    # Both paths resolve relative to ``settings.project_root()``.
    # If the project root can't be computed (e.g. a test built a
    # Settings without a config.yaml ancestor), fall back to
    # ``<this file>/../../../ui`` which works for the common
    # checkout layout (``src/tinyrag/main.py`` -> 3 dirs up is the
    # repo root).
    try:
        ui_root = Path(settings.project_root()) / "ui"
    except RuntimeError:
        ui_root = Path(__file__).resolve().parent.parent.parent / "ui"

    templates_dir = ui_root / "templates"
    static_dir = ui_root / "static"

    # ``StaticFiles`` raises on missing dir; if ui/ is absent (e.g.
    # partial install) we still want the API to be reachable — log
    # a warning + skip the mount. Tests that don't need the UI can
    # pass a Settings with a project root that has no ui/ subtree.
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    else:
        _log.warning("ui_static_dir_missing", path=str(static_dir))

    templates = Jinja2Templates(directory=str(templates_dir)) \
        if templates_dir.is_dir() else None

    @app.get("/", tags=["meta"], summary="Chat web UI (Jinja2).")
    async def root(request: Request):
        """Render the chat page.

        Pre-4.21 this returned a tiny JSON banner. Step 4.21 turns
        it into the chat UI entry point so the user just opens
        ``http://127.0.0.1:8000/`` and sees the composer. If the
        template dir is missing (broken install), fall back to the
        JSON banner so the API surface stays reachable.
        """
        if templates is None:
            return {
                "service": "tinyrag",
                "version": "0.4.0",
                "api_docs": "/docs",
                "ui_error": f"templates dir missing: {templates_dir}",
            }
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "service": "tinyrag",
                "version": "0.4.0",
                "api_docs": "/docs",
            },
        )

    return app


# ----------------------------------------------------------------------------
# Builders (private — the only place concrete LLM/embedder classes
# are imported outside the lifespan handler)
# ----------------------------------------------------------------------------


def _build_embedder(settings: Settings, *, kind: str) -> EmbeddingModel:
    """Build an :class:`EmbeddingModel` per ``kind``.

    ``"real"`` is the default — wraps sentence-transformers with
    the configured model + device + batch size + cache dir.
    ``"fake"`` is the test/dev escape hatch — a 384-dim
    :class:`FakeEmbedder` that does no model loading.
    """
    if kind == "fake":
        return FakeEmbedder(dimension=DEFAULT_EMBEDDING_DIMENSION)
    if kind == "real":
        return SentenceTransformerEmbedder(settings.embedding)
    raise ValueError(f"unknown embedder kind: {kind!r}")


def _build_llm(settings: Settings, *, kind: str) -> LLMClient:
    """Build an :class:`LLMClient` per ``kind``.

    ``"real"`` → :class:`LlamaCppClient`. Strips ``.gguf`` from
    the model id and the trailing slash from the server URL
    (matches the pattern in :mod:`scripts.ask`).
    ``"fake"`` → :class:`FakeLLMClient` (deterministic canned
    reply; perfect for offline development).
    """
    if kind == "fake":
        from tinyrag.generation.llm_client import FakeLLMClient

        return FakeLLMClient()
    if kind == "real":
        model_id = settings.llm.model_path
        if model_id.endswith(".gguf"):
            model_id = model_id[: -len(".gguf")]
        return LlamaCppClient(
            base_url=settings.llm.server_url.rstrip("/"),
            model=model_id,
        )
    raise ValueError(f"unknown llm kind: {kind!r}")


# ----------------------------------------------------------------------------
# ``uvicorn tinyrag.main:app`` entry point
# ----------------------------------------------------------------------------
#
# ``app`` is constructed at import time so uvicorn can pick it up
# via ``uvicorn tinyrag.main:app``. The factory still accepts an
# explicit ``settings`` for tests — the import-level ``app`` just
# reads ``config.yaml`` from the CWD.
#
# The structured logger is configured **inside** the factory so a
# test that calls ``create_app(_tiny_settings(...))`` doesn't
# double-configure the global log handler.

app = create_app()


__all__ = ["create_app"]
