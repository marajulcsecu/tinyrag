"""HTTP layer — FastAPI routes for query, documents, and admin.

The :mod:`tinyrag.api` subpackage is the only place that knows the
project is a web service. Everything in here depends on FastAPI; nothing
in :mod:`tinyrag.core` or :mod:`tinyrag.generation` knows FastAPI exists.

Modules (to be added in later Phase 4 steps)
--------------------------------------------
- :mod:`tinyrag.api.routes_query` — ``POST /api/query`` (SSE streaming
  answer) and ``GET /api/status`` (liveness + model id).
- :mod:`tinyrag.api.routes_docs` — ``POST /api/documents`` (upload),
  ``GET /api/documents`` (list), ``DELETE /api/documents/{id}``.
- :mod:`tinyrag.api.routes_admin` — ``POST /api/admin/reindex`` and
  ``POST /api/admin/benchmark``.

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

# Subpackage is currently a placeholder. Modules will be re-exported
# here as they are implemented in later Phase 4 steps (4.17, 4.18, 4.19).
