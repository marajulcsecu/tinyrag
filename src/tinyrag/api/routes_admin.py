"""HTTP routes for admin operations — STUB for Step 4.17.

Admin endpoints (``POST /api/admin/reindex``, ``POST
/api/admin/benchmark``) belong on a separate router from
:mod:`tinyrag.api.routes_query` and :mod:`tinyrag.api.routes_docs`
because they need a different access-control story (the dashboard
will hide them from non-admin users) and they are called by
operational tooling, not the public dashboard. Step 4.17 lands
the skeleton only.

Why a separate file?
--------------------
- ``routes_admin.py`` is the natural place for future
  ``POST /api/admin/reindex`` (rebuild the FAISS index from
  scratch — needed after a model upgrade) and ``POST
  /api/admin/benchmark`` (run the eval set against the live
  index). Both are independent of the public surface and have
  independent testing needs.
- In a future deployment the admin router can be mounted on a
  separate port or behind a different auth middleware without
  touching the public routes.

Pure HTTP / no I/O
------------------
Every route in this module is a one-liner returning ``501``. No
FAISS or SQLite calls happen here — those arrive with the real
implementation in later steps.

Location: ``src/tinyrag/api/routes_admin.py``
"""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from tinyrag.api.schemas import NotImplementedResponse

#: The ``detail`` string for every admin 501 response. Mirrors
#: the docs stub's :data:`NOT_IMPLEMENTED_DETAIL` constant.
ADMIN_NOT_IMPLEMENTED_DETAIL = (
    "Admin endpoints land in a later Phase 4 step "
    "(see docs/06_roadmap_v2.md). Use scripts/ingest.py, "
    "scripts/ingest_sensors.py, and scripts/eval.py from the CLI "
    "until the admin router is wired up."
)


def build_admin_router() -> APIRouter:
    """Build the admin router (Step 4.17 stub).

    Returned as a factory for the same reason as the query + docs
    routers — keeps the composition root's ``include_router``
    calls symmetric.
    """
    router = APIRouter(tags=["admin"], prefix="/api/admin")

    @router.post(
        "/reindex",
        response_model=NotImplementedResponse,
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        summary="Rebuild the FAISS indices from scratch. (Stub.)",
    )
    async def reindex() -> JSONResponse:
        """Rebuild both FAISS indices from the metadata DB.

        Stub for Step 4.17. The real implementation will iterate
        every ``Chunk`` row, re-embed with the current model, and
        rewrite both ``.faiss`` files atomically.
        """
        return _not_implemented()

    @router.post(
        "/benchmark",
        response_model=NotImplementedResponse,
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        summary="Run the eval set against the live index. (Stub.)",
    )
    async def benchmark() -> JSONResponse:
        """Run the gold-set eval against the live pipeline.

        Stub for Step 4.17. The real implementation will load the
        gold set, run every query through ``run_ask``, score the
        answers, and return the same JSON shape
        ``scripts/eval.py`` writes.
        """
        return _not_implemented()

    return router


def _not_implemented() -> JSONResponse:
    """Return the standard 501 body for every admin stub route."""
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content=NotImplementedResponse(
            detail=ADMIN_NOT_IMPLEMENTED_DETAIL
        ).model_dump(),
    )


__all__ = ["ADMIN_NOT_IMPLEMENTED_DETAIL", "build_admin_router"]
