"""HTTP routes for document management — STUB for Step 4.17.

The full document management surface — ``POST /api/documents``
(upload + ingest), ``GET /api/documents`` (list), ``DELETE
/api/documents/{id}`` (delete + vector-store cleanup) — is
**Step 4.18**'s job. Step 4.17 lands the skeleton: each route
returns ``501 Not Implemented`` with a body that points the
client at the future step.

Why a stub now?
---------------
- The dashboard's router (Step 4.21) will declare links to
  ``/api/documents``; without the route even existing,
  ``GET /api/documents`` would 404 and the dashboard's
  "Documents" page would be unreachable. ``501`` is the right
  status: the endpoint **will** exist, just not yet.
- Tests can assert the stub contract (status code + body shape)
  today and the real behaviour tomorrow without changing the
  test surface.
- ``NOT_IMPLEMENTED_DETAIL`` is a stable string the dashboard
  can match on for a "coming soon" placeholder card.

Pure HTTP / no I/O
------------------
Every route in this module is a one-liner that returns a
:class:`NotImplementedResponse`. No FAISS or SQLite calls happen
here — those arrive with Step 4.18's real implementation.

Location: ``src/tinyrag/api/routes_docs.py``
"""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from tinyrag.api.schemas import NotImplementedResponse

#: The ``detail`` string included in every 501 response from this
#: module. Surfaced as a constant so tests can match on the exact
#: string and so a future "this is ready" search hits one line.
NOT_IMPLEMENTED_DETAIL = (
    "Document management endpoints land in Step 4.18 "
    "(see docs/06_roadmap_v2.md Step 4.18). "
    "Until then, use scripts/ingest.py from the CLI."
)


def build_docs_router() -> APIRouter:
    """Build the documents router (Step 4.17 stub).

    Returned as a factory (not a module-level singleton) for the
    same reason :func:`tinyrag.api.routes_query.build_query_router`
    is — the composition root calls ``app.include_router(...)``
    with whatever this function returns.
    """
    router = APIRouter(tags=["documents"], prefix="/api/documents")

    @router.post(
        "",
        response_model=NotImplementedResponse,
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        summary="Upload + ingest a document. (Stub — Step 4.18.)",
    )
    async def upload_document() -> JSONResponse:
        """Upload a PDF/TXT/MD file and run the ingest pipeline.

        Stub for Step 4.17. The real implementation in Step 4.18
        will accept ``multipart/form-data``, validate the
        extension + size (≤50 MB), write to a temp file, call
        ``IngestionPipeline.run(...)``, and return the
        ``IngestionReport.to_dict()`` shape.
        """
        return _not_implemented()

    @router.get(
        "",
        response_model=NotImplementedResponse,
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        summary="List ingested documents. (Stub — Step 4.18.)",
    )
    async def list_documents() -> JSONResponse:
        """Return the list of ingested documents.

        Stub for Step 4.17. The real implementation in Step 4.18
        will call ``MetadataStore.list_documents()`` and shape
        the response with ``filename``, ``doc_type``, ``chunk_count``,
        ``ingested_at``, etc.
        """
        return _not_implemented()

    @router.delete(
        "/{document_id}",
        response_model=NotImplementedResponse,
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        summary="Delete a document + its vectors. (Stub — Step 4.18.)",
    )
    async def delete_document(document_id: str) -> JSONResponse:
        """Delete a document from the metadata DB and remove its vectors.

        Stub for Step 4.17. The real implementation in Step 4.18
        will call ``MetadataStore.delete_document(document_id)``
        and ``FAISSStore.delete_by_source(...)`` to keep both
        stores consistent.
        """
        return _not_implemented()

    return router


def _not_implemented() -> JSONResponse:
    """Return the standard 501 body for every stub route."""
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content=NotImplementedResponse(detail=NOT_IMPLEMENTED_DETAIL).model_dump(),
    )


__all__ = ["NOT_IMPLEMENTED_DETAIL", "build_docs_router"]
