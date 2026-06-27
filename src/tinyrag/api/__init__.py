"""HTTP layer — FastAPI routes for query, documents, and admin.

The :mod:`tinyrag.api` subpackage is the only place that knows the
project is a web service. Everything in here depends on FastAPI; nothing
in :mod:`tinyrag.core` or :mod:`tinyrag.generation` knows FastAPI exists.

Modules
-------
- :mod:`tinyrag.api.routes_query` — ``POST /api/query`` (full RAG
  pipeline; SSE streaming lands in Step 4.19) and ``GET /api/status``
  (liveness + model + index + RAM + llama.cpp status per FR-39).
- :mod:`tinyrag.api.routes_docs` — ``POST /api/documents`` (upload),
  ``GET /api/documents`` (list), ``DELETE /api/documents/{id}``.
  Skeleton in Step 4.17 (returns 501); filled in by Step 4.18.
- :mod:`tinyrag.api.routes_admin` — ``POST /api/admin/reindex`` and
  ``POST /api/admin/benchmark``. Skeleton in Step 4.17.
- :mod:`tinyrag.api.schemas` — Pydantic request / response models.
- :mod:`tinyrag.api.deps` — FastAPI dependency providers pulling
  singletons out of ``app.state``.
- :mod:`tinyrag.api.errors` — global exception handlers mapping
  domain errors to HTTP status codes + the uniform
  :class:`ErrorResponse` shape.
- :mod:`tinyrag.api.system_info` — RAM + llama.cpp reachability
  probes (used by ``GET /api/status``).

Why a thin layer?
-----------------
- The HTTP layer should not contain business logic. It validates
  requests, calls into :mod:`tinyrag.core` + :mod:`tinyrag.generation`,
  and serialises responses. Anything more is a smell.
- Putting each route group in its own file keeps diffs localised
  (e.g. tweaking the SSE stream format only touches
  ``routes_query.py``).
- ``routes_admin.py`` is intentionally separate from the public
  query/document routes so the admin endpoints can be guarded
  differently in the future (auth, IP allowlist, etc.).

Location: ``src/tinyrag/api/``
"""

from __future__ import annotations

from tinyrag.api.errors import install_exception_handlers
from tinyrag.api.routes_admin import ADMIN_NOT_IMPLEMENTED_DETAIL, build_admin_router
from tinyrag.api.routes_docs import build_docs_router
from tinyrag.api.routes_query import build_query_router
from tinyrag.api.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    NotImplementedResponse,
    StatusResponse,
)

__all__ = [
    "ADMIN_NOT_IMPLEMENTED_DETAIL",
    "AskRequest",
    "AskResponse",
    "ErrorResponse",
    "NotImplementedResponse",
    "StatusResponse",
    "build_admin_router",
    "build_docs_router",
    "build_query_router",
    "install_exception_handlers",
]
