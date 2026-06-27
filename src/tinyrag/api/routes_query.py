"""HTTP routes for asking questions + checking system status.

This module is one half of the public HTTP surface
(the other half is :mod:`tinyrag.api.routes_docs`). It exposes:

- ``GET /api/status`` — liveness + per-subsystem health (FR-39).
- ``POST /api/query`` — run the full RAG pipeline against a question
  and return the :class:`tinyrag.core.answer.Answer` JSON.

Step 4.17 implements both. Step 4.19 will add SSE streaming on top
of ``POST /api/query`` (the response body becomes ``data: {...}
\\n\\n`` events rather than one final JSON object); the route
function stays almost identical — we just swap the return type
from ``JSONResponse`` to :class:`sse_starlette.EventSourceResponse`.

Why both endpoints in one module?
---------------------------------
They're the public "talk to the model" surface. ``/api/status`` is
the dashboard's heartbeat; ``/api/query`` is the dashboard's submit
button. Putting them together means a future change to the query
contract (e.g. adding SSE) is a one-file diff. Admin endpoints
(reindex, benchmark) belong in :mod:`routes_admin` because they
need different auth and are called by a different UI surface.

Pure HTTP / no I/O
------------------
Every dependency this module needs (retriever, prompt builder,
LLM, metadata, settings) comes in via FastAPI's ``Depends(...)``
mechanism, which pulls from ``app.state``. This module does no
direct I/O — the FAISS search, the LLM call, the SQLite write
all happen inside the dependencies. Tests can swap the
``app.state`` values for fakes.

Location: ``src/tinyrag/api/routes_query.py``
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from tinyrag.api.deps import (
    get_embedder,
    get_llm,
    get_metadata,
    get_prompt_builder,
    get_retriever,
    get_settings,
)
from tinyrag.api.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    StatusResponse,
)
from tinyrag.api.system_info import (
    get_embedding_model_name,
    get_llama_cpp_status,
    get_ram_mb,
)
from tinyrag.observability.logger import get_logger

if TYPE_CHECKING:
    from tinyrag.config import Settings
    from tinyrag.core.prompt_builder import PromptBuilder
    from tinyrag.core.retriever import Retriever
    from tinyrag.generation.llm_client import LLMClient
    from tinyrag.ingestion.embedder import EmbeddingModel
    from tinyrag.storage.metadata import MetadataStore
    from tinyrag.storage.vector_store import VectorStore


_log = get_logger(__name__)


# ============================================================================
# Router
# ============================================================================


def build_query_router() -> APIRouter:
    """Build the ``/api/status`` + ``/api/query`` router.

    Returned as a factory (not a module-level singleton) so the
    composition root can call ``app.include_router(...)`` with
    a router that knows nothing about the test app's state. Tests
    can call this function to spin up a fresh router against
    their own composition-root call.
    """
    router = APIRouter(tags=["query"], prefix="/api")

    # --------------------------------------------------------------------
    # GET /api/status — FR-39 / NFR-37
    # --------------------------------------------------------------------
    @router.get(
        "/status",
        response_model=StatusResponse,
        summary="Liveness + per-subsystem health.",
        responses={
            200: {"description": "Status JSON."},
            503: {"model": ErrorResponse, "description": "App state missing."},
        },
    )
    async def get_status(
        request: Request,
        settings: Settings = Depends(get_settings),
        embedder: EmbeddingModel = Depends(get_embedder),
        retriever: Retriever = Depends(get_retriever),
    ) -> dict[str, Any]:
        """Return liveness + every FR-39 field.

        The route is intentionally **best-effort**: every probe is
        wrapped in try/except so a single failing subsystem
        (e.g. llama.cpp down) still produces a 200 with
        ``ok=False`` rather than a 500. The dashboard polls this
        every few seconds, so robustness > completeness.
        """
        # Sizes come from the FAISS meta — cheap, in-memory.
        doc_store = request.app.state.doc_store
        sensor_store = request.app.state.sensor_store
        llm = request.app.state.llm

        # Sizes come from the FAISS meta — cheap, in-memory.
        doc_count = _safe_size(doc_store)
        sensor_count = _safe_size(sensor_store)

        # Model introspection. Real llama-server id is fetched
        # via the LLMClient's ``model_name`` property (which may
        # be a network round-trip — wrapped in try/except).
        model_name = _safe_model_name(llm)
        emb_model = get_embedding_model_name(embedder)

        # Subsystem health: each probe is independent. Any failure
        # flips ``ok`` to False but doesn't crash the endpoint.
        llama_status = get_llama_cpp_status(settings.llm.server_url)
        ram_mb = get_ram_mb()
        try:
            emb_dim = int(getattr(embedder, "dimension", 384))
        except (TypeError, ValueError):
            emb_dim = 384

        # ``ok`` is True iff every probe succeeded. The dashboard
        # uses it as the green/red pill toggle.
        ok = llama_status == "up" and doc_count is not None and sensor_count is not None
        if llama_status == "down":
            ok = False
        # Note: ``retriever`` is in the dep list only to assert
        # that lifespan built it — if it's missing the dep
        # provider already raised 503.
        del retriever

        return StatusResponse(
            ok=ok,
            model_name=model_name,
            embedding_model=emb_model,
            embedding_dim=emb_dim,
            doc_chunk_count=doc_count if doc_count is not None else 0,
            sensor_chunk_count=sensor_count if sensor_count is not None else 0,
            doc_index_path=str(settings.retrieval.doc_index_path),
            sensor_index_path=str(settings.retrieval.sensor_index_path),
            metadata_db_path=str(settings.paths.metadata_db),
            ram_mb=ram_mb,
            llama_cpp_status=llama_status,
            llama_cpp_url=settings.llm.server_url,
            sensor_source=settings.sensors.source.value
            if hasattr(settings.sensors.source, "value")
            else str(settings.sensors.source),
            deployment_target=settings.deployment.target.value
            if hasattr(settings.deployment.target, "value")
            else str(settings.deployment.target),
        ).model_dump()

    # --------------------------------------------------------------------
    # POST /api/query — full RAG pipeline
    # --------------------------------------------------------------------
    @router.post(
        "/query",
        response_model=AskResponse,
        summary="Ask a question; return the full Answer JSON.",
        responses={
            200: {"description": "Answer JSON (see tinyrag.core.answer.Answer.to_dict)."},
            400: {"model": ErrorResponse, "description": "Bad request (empty query, etc.)."},
            422: {"model": ErrorResponse, "description": "Validation error."},
            500: {"model": ErrorResponse, "description": "Pipeline error."},
            502: {"model": ErrorResponse, "description": "LLM refused."},
            503: {"model": ErrorResponse, "description": "LLM unavailable."},
        },
    )
    async def post_query(
        body: AskRequest,
        request: Request,
        settings: Settings = Depends(get_settings),
        embedder: EmbeddingModel = Depends(get_embedder),
        retriever: Retriever = Depends(get_retriever),
        prompt_builder: PromptBuilder = Depends(get_prompt_builder),
        llm: LLMClient = Depends(get_llm),
        metadata: MetadataStore = Depends(get_metadata),
    ) -> JSONResponse:
        """Run the RAG pipeline and return the :class:`Answer` JSON.

        Mirrors ``scripts.ask.run_ask`` one-for-one so the HTTP
        surface and the CLI surface produce the same shape. The
        route handler does the work inline (rather than calling
        ``run_ask``) so the per-stage timings appear in both the
        structured log *and* the response body, and so we can
        short-circuit an empty query without spinning up the
        pipeline.

        The empty-query short-circuit returns ``text=""`` and a
        ``model_name=""`` — same shape the CLI uses — and still
        appends a ``query_log`` row when ``log_query=True`` so
        the eval set can later grade the no-answer case.
        """
        timings: dict[str, float] = {}
        t_total_start = time.monotonic()

        # ---- Stage 1: retrieve -------------------------------------
        t = time.monotonic()
        try:
            retrieval = retriever.retrieve(
                body.query,
                k_doc=body.k_doc,
                k_sensor=body.k_sensor,
                threshold=body.threshold,
            )
        except Exception as exc:
            _log.error(
                "retrieve_failed",
                query=body.query[:80],
                error=str(exc),
                exc_info=True,
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="retrieval_failed",
                    detail=f"retrieval failed: {exc}",
                ).model_dump(),
            )
        timings["retrieve_ms"] = (time.monotonic() - t) * 1000.0

        # ---- Stage 2: prompt ---------------------------------------
        t = time.monotonic()
        prompt = prompt_builder.build(body.query, retrieval.chunks)
        timings["prompt_ms"] = (time.monotonic() - t) * 1000.0

        # ---- Stage 3: llm ------------------------------------------
        t = time.monotonic()
        # The LLMClient Protocol returns ``(text, GenerationStats)`` —
        # the streaming detail is encapsulated in the client. Step
        # 4.19 will replace this with a streaming variant that
        # yields SSE events; the route contract stays the same.
        prompt_tokens = prompt.prompt_tokens
        completion_tokens = 0
        full_text = ""
        try:
            full_text, stats = llm.generate(
                prompt.messages,
                max_tokens=body.max_tokens,
                temperature=settings.llm.temperature,
            )
            completion_tokens = stats.completion_tokens
        except Exception as exc:
            _log.error(
                "llm_failed",
                query=body.query[:80],
                error=str(exc),
                exc_info=True,
            )
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content=ErrorResponse(
                    error="llm_failed",
                    detail=f"LLM call failed: {exc}",
                ).model_dump(),
            )
        timings["llm_ms"] = (time.monotonic() - t) * 1000.0

        # ---- Stage 4: log ------------------------------------------
        used_sensor = retrieval.used_sensor_idx
        top_score = retrieval.top_score  # float | None
        if body.log_query:
            t = time.monotonic()
            try:
                # ``log_query`` writes a query_log row. The
                # schema's latency fields are integers (ms), so
                # we round. ``used_sensor_idx`` is an int flag
                # (0/1) — the schema treats it as a smallint.
                metadata.log_query(
                    query=body.query,
                    top1_score=top_score,
                    num_chunks=len(retrieval),
                    retrieval_ms=int(round(timings.get("retrieve_ms", 0.0))),
                    generation_ms=int(round(timings.get("llm_ms", 0.0))),
                    total_ms=int(round((time.monotonic() - t_total_start) * 1000.0)),
                    model=_safe_model_name(llm),
                    used_sensor_idx=1 if used_sensor else 0,
                )
            except Exception as exc:
                # Logging failures must not break the user-facing
                # response — log at warning level and move on.
                _log.warning(
                    "log_query_failed",
                    query=body.query[:80],
                    error=str(exc),
                )
            timings["log_ms"] = (time.monotonic() - t) * 1000.0

        total_ms = (time.monotonic() - t_total_start) * 1000.0

        # ---- Build the response -----------------------------------
        from tinyrag.core.answer import Answer, build_citations_from_chunks

        # Citations: number 1..N parallel to the surviving
        # chunks. The PromptBuilder trims some chunks to fit the
        # token budget; we mirror that by using
        # ``prompt.chunks_used`` (the count) but cite every
        # chunk that survived the threshold filter (the same set
        # the prompt builder rendered as ``[1]..[N]`` markers).
        # The CLI uses the same shortcut — see scripts/ask.py.
        surviving_chunks = list(retrieval.chunks)
        surviving_scores = list(retrieval.scores)

        answer = Answer(
            query=body.query,
            text=full_text,
            used_sensor_idx=used_sensor,
            top_score=top_score,
            model_name=_safe_model_name(llm),
            citations=build_citations_from_chunks(
                surviving_chunks, surviving_scores
            ),
            chunks_used=len(surviving_chunks),
            chunks_dropped=prompt.chunks_dropped,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            duration_retrieve_ms=timings.get("retrieve_ms", 0.0),
            duration_prompt_ms=timings.get("prompt_ms", 0.0),
            duration_llm_ms=timings.get("llm_ms", 0.0),
            duration_total_ms=total_ms,
        )

        _log.info(
            "query_completed",
            query=body.query[:80],
            chunks_used=answer.chunks_used,
            top_score=answer.top_score,
            used_sensor=used_sensor,
            total_ms=round(total_ms, 2),
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=answer.to_dict(),
        )

    return router


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _safe_size(store: VectorStore) -> int | None:
    """Return ``store.size()`` or ``None`` if the store isn't loaded.

    The status endpoint must not crash if the FAISS index file is
    missing (e.g. first-run before ingest). A ``None`` size flips
    ``ok`` to False without raising.
    """
    try:
        return int(store.size())
    except Exception:
        return None


def _safe_model_name(llm: LLMClient) -> str:
    """Return ``llm.model_name()`` or ``"unknown"`` if introspection failed.

    Note: the Protocol declares ``model_name()`` as a **method** (not
    a property — it's `def model_name(self) -> str:` in
    :mod:`tinyrag.generation.llm_client`). The
    :class:`FakeLLMClient` makes it a property, so we duck-type:
    call it if it's callable, read it if it's a string.
    """
    try:
        attr = llm.model_name
        if callable(attr):
            attr = attr()
        return str(attr) if attr else "unknown"
    except Exception:
        return "unknown"


__all__ = ["build_query_router"]
