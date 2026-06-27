"""FastAPI dependency providers — pull singletons from ``app.state``.

This module is the **DI seam** for the HTTP layer. Every route
handler declares its dependencies via :func:`fastapi.Depends`, and
those providers read pre-built singletons out of
``request.app.state``. No route handler instantiates a concrete
class — that's the composition root's job
(:func:`tinyrag.main.create_app`).

Why ``app.state`` instead of module-level globals?
--------------------------------------------------
- **Testability.** The test suite builds a tiny app via
  ``create_app(_tiny_settings(tmp_path), *, llm_kind="fake", ...)``
  with tmpdir-backed FAISS + SQLite, so the test app's state
  never leaks into a different test.
- **Multiple apps per process.** A future "admin" process or the
  portability self-test (Step 4.20) can spin up a second app in
  the same Python process with different settings — module-level
  globals would clobber each other.
- **No global mutation.** Reading from ``app.state`` is read-only
  by design; the composition root is the only writer.

Why a separate module for the providers?
----------------------------------------
The route modules are about HTTP (status codes, response shapes,
OpenAPI tags). The providers are about the dependency graph
(which subsystems each route needs). Putting them in separate
modules keeps the diffs small when either changes — adding a new
subsystem is one new provider function, not a route module edit.

No I/O
------
This module is glue. It reads attributes off ``app.state`` and
returns them. No database calls, no file I/O, no network. A
provider that needs to do real work (e.g. ``get_ram_mb``) belongs
in :mod:`tinyrag.api.system_info` (added when it's needed), not
here.

Location: ``src/tinyrag/api/deps.py``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request, status

if TYPE_CHECKING:
    from tinyrag.config import Settings
    from tinyrag.core.prompt_builder import PromptBuilder
    from tinyrag.core.retriever import Retriever
    from tinyrag.generation.llm_client import LLMClient
    from tinyrag.ingestion.embedder import EmbeddingModel
    from tinyrag.storage.metadata import MetadataStore
    from tinyrag.storage.vector_store import VectorStore


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _require(request: Request, key: str) -> Any:
    """Return ``request.app.state[key]`` or raise 503 if missing.

    The lifespan handler in :func:`tinyrag.main.create_app` is
    responsible for populating every key this module reads. If a
    key is missing, either the lifespan crashed (and the app
    shouldn't have started accepting requests) or someone forgot
    to wire a new dependency. Either way, a 503 is the right
    response — "service unavailable, retry shortly".
    """
    try:
        value = getattr(request.app.state, key)
    except AttributeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"app.state.{key} not initialised (lifespan did not run?)",
        ) from exc
    if value is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"app.state.{key} is None (subsystem failed to load?)",
        )
    return value


# ----------------------------------------------------------------------------
# Providers
# ----------------------------------------------------------------------------


def get_settings(request: Request) -> Settings:
    """Return the :class:`tinyrag.config.Settings` the app was built with."""
    return _require(request, "settings")


def get_embedder(request: Request) -> EmbeddingModel:
    """Return the singleton :class:`EmbeddingModel` (real or fake)."""
    return _require(request, "embedder")


def get_doc_store(request: Request) -> VectorStore:
    """Return the singleton document :class:`VectorStore`."""
    return _require(request, "doc_store")


def get_sensor_store(request: Request) -> VectorStore:
    """Return the singleton sensor :class:`VectorStore`."""
    return _require(request, "sensor_store")


def get_metadata(request: Request) -> MetadataStore:
    """Return the singleton :class:`MetadataStore`."""
    return _require(request, "metadata")


def get_llm(request: Request) -> LLMClient:
    """Return the singleton :class:`LLMClient` (real or fake)."""
    return _require(request, "llm")


def get_retriever(request: Request) -> Retriever:
    """Return the singleton :class:`Retriever`.

    The retriever is constructed once at lifespan time and reused
    for every request. It owns no mutable state (the FAISS stores
    it points at are themselves thread-safe thanks to their
    internal locks), so reuse is safe.
    """
    return _require(request, "retriever")


def get_prompt_builder(request: Request) -> PromptBuilder:
    """Return the singleton :class:`PromptBuilder`."""
    return _require(request, "prompt_builder")


# ----------------------------------------------------------------------------
# Higher-level dependency: an already-built Answer (for tests)
# ----------------------------------------------------------------------------
#
# The /api/query route runs the pipeline inline rather than via a
# ``get_answer`` dependency because it needs the per-stage timings
# in its response (and the structured log). A future "stream this
# answer via SSE" route (Step 4.19) will likely introduce a real
# factory here, but for Step 4.17 the inline pattern is the
# simplest correct thing.


__all__ = [
    "get_doc_store",
    "get_embedder",
    "get_llm",
    "get_metadata",
    "get_prompt_builder",
    "get_retriever",
    "get_sensor_store",
    "get_settings",
]
