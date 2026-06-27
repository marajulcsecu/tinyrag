"""HTTP routes for document management (Step 4.18).

Three real endpoints replace the 501 stubs Step 4.17 left behind:

- ``POST /api/documents`` — multipart upload + ingest pipeline.
  Streams the upload to a **tempfile** (auto-cleaned), validates
  the extension + size + doc_type, then calls
  :func:`scripts.ingest.run_ingest` and returns the resulting
  :class:`scripts.ingest.IngestionReport.to_dict` shape.
- ``GET /api/documents?limit=N&offset=M`` — paginated list from
  the metadata store. Returns a uniform
  :class:`tinyrag.api.schemas.DocumentListResponse` with cursor
  pagination (``next_offset``).
- ``DELETE /api/documents/{id}`` — cascade-delete: removes the
  FAISS vectors for the document's chunks via
  :meth:`tinyrag.storage.vector_store.VectorStore.remove_ids`,
  then deletes the document row from SQLite (which CASCADE-
  deletes its chunks). Returns a precise
  :class:`tinyrag.api.schemas.DocumentDeleteResponse` with the
  count of chunks / vectors removed.

Why this module is split from ``routes_query.py``
-------------------------------------------------
The query router owns "answer a question" (read path). The
documents router owns "manage the corpus" (write path).
Keeping them separate means the dashboard's "Documents" page
can lazy-load just this router if it ever ships as a SPA bundle.

Why the ingest script is imported via ``sys.path``
--------------------------------------------------
``scripts/ingest.py`` is a top-level CLI script — it's NOT
under ``src/tinyrag/`` so the package distribution
(``pyproject.toml``'s ``exclude`` list) deliberately doesn't
ship it. We therefore add the repo-root's ``scripts/`` directory
to :data:`sys.path` exactly once at import time, mirroring the
pattern already used by ``tests/test_ingest.py`` and
``tests/test_ask.py``. A cleaner future would refactor
``run_ingest`` into ``src/tinyrag/ingestion/pipeline.py`` (a
Step 4.22 follow-up) — but Step 4.18's scope is the HTTP
surface, not the ingest-script location.

Pure HTTP / single source of truth
----------------------------------
Validation rules (extension whitelist, size cap, doc_type
whitelist) live here as module constants; the rest of the
codebase can import them from ``tinyrag.api.routes_docs`` if
they ever need to. The actual filesystem work happens in
``run_ingest`` — this module is just glue.

Location: ``src/tinyrag/api/routes_docs.py``
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse

from tinyrag.api.deps import get_doc_store, get_metadata, get_settings
from tinyrag.api.schemas import (
    DocumentDeleteResponse,
    DocumentListItemResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    ErrorResponse,
)
from tinyrag.storage.metadata import SUPPORTED_DOC_TYPES as _META_DOC_TYPES

if TYPE_CHECKING:
    from tinyrag.config import Settings
    from tinyrag.storage.metadata import MetadataStore
    from tinyrag.storage.vector_store import VectorStore


# ---------------------------------------------------------------------------
# scripts/ sys.path bootstrap (see module docstring for rationale)
# ---------------------------------------------------------------------------
#
# ``scripts/`` lives at the repo root and isn't part of the
# ``tinyrag`` package. We add it to ``sys.path`` once so we can
# ``from ingest import run_ingest`` exactly like the test suite
# does (see ``tests/test_ingest.py`` lines 60-65). Idempotent —
# guarded with ``not in sys.path`` so re-imports don't grow the
# path list.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Imported AFTER the sys.path bootstrap so the ``from ingest``
# works. Imported lazily (inside the route handler would also
# work but a top-level import makes ``mypy`` + IDEs happier).
# ``I001`` is suppressed because this import is intentionally
# separated from the ``tinyrag.*`` block above — it's a
# repo-root script, not part of the package.
from ingest import IngestionReport, run_ingest  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Validation constants — surfaced as module exports so tests +
# dashboard can import the canonical values without re-declaring.
# ---------------------------------------------------------------------------

#: Whitelist of accepted file extensions (lowercase, with leading dot).
#: Per FR-1 the dashboard upload form will reject anything else;
#: this router applies the same gate server-side so a curl power
#: user can't bypass it.
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md"})

#: Hard upload size cap in bytes (FR-10). 50 MiB. Exceeding this
#: during streaming returns 413 ``file_too_large`` before the rest
#: of the request body is read.
MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024

#: Whitelist of accepted ``doc_type`` form values. Sourced from
#: the canonical :data:`tinyrag.storage.metadata.SUPPORTED_DOC_TYPES`
#: so the API and the metadata store never disagree (the metadata
#: store raises :class:`ValueError` for unknown doc_types, which
#: would surface as a 400 ``metadata_error`` instead of our clean
#: 400 ``invalid_doc_type``).
SUPPORTED_DOC_TYPES: frozenset[str] = _META_DOC_TYPES

#: Default ``doc_type`` when the form omits the field. Matches
#: the CLI's default (``scripts/ingest.py`` argparse default).
DEFAULT_DOC_TYPE: str = "manual"

#: Structured-log logger handle (the api layer's
#: ``get_logger(__name__)`` would couple us to a yet-to-be-added
#: helper; raw ``logging.getLogger`` is enough for one warning).
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_filename(name: str) -> str:
    """Return a path-traversal-free basename.

    Strips any directory prefix manually (handles BOTH POSIX
    ``/`` and Windows ``\\`` separators, since
    :meth:`pathlib.Path.name` only knows the platform's own
    separator), then rejects:

    - empty strings,
    - dotfiles (``.``, ``..``, ``.bashrc``) — to avoid
      accidentally clobbering cwd-relative paths,
    - Windows-reserved device names (``CON``, ``PRN``, ``AUX``,
      ``NUL``, ``COM1..COM9``, ``LPT1..LPT9``) — case-
      insensitive.

    Returns the cleaned basename. Callers map a ``False``-y
    result to a 400.
    """
    raw = (name or "").replace("\\", "/").strip()
    base = Path(raw).name
    if not base or base in {".", ".."} or base.startswith("."):
        return ""
    upper = base.upper().split(".", 1)[0]
    if upper in {
        "CON", "PRN", "AUX", "NUL",
        *[f"COM{i}" for i in range(1, 10)],
        *[f"LPT{i}" for i in range(1, 10)],
    }:
        return ""
    return base


def _make_tempfile(suffix: str) -> tuple[Path, Any]:
    """Create a writable tempfile with the given suffix.

    Returns ``(path, file_handle)``. The handle must be closed
    by the caller before :func:`scripts.ingest.run_ingest` re-
    opens it for reading on POSIX (``NamedTemporaryFile`` keeps
    an exclusive lock on the file on Linux). The path is
    returned so the caller can ``unlink`` it in ``finally``.
    """
    # NOTE: ``SIM115`` (use a ``with`` block) is intentionally
    # disabled here — we need to return the open handle to the
    # caller (which closes it itself after writing) AND keep
    # the file alive after the function returns (``delete=False``).
    # A ``with`` block would close the handle on scope exit.
    fh = tempfile.NamedTemporaryFile(  # noqa: SIM115
        delete=False, suffix=suffix, dir=tempfile.gettempdir()
    )
    return Path(fh.name), fh


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def build_docs_router() -> APIRouter:
    """Build the documents router (Step 4.18 real surface).

    Returned as a factory (not a module-level singleton) so the
    composition root can call ``app.include_router(...)`` with
    whatever this function returns — same pattern as
    :func:`tinyrag.api.routes_query.build_query_router`.
    """
    router = APIRouter(tags=["documents"], prefix="/api/documents")

    # -----------------------------------------------------------------
    # POST /api/documents
    # -----------------------------------------------------------------
    @router.post(
        "",
        response_model=DocumentUploadResponse,
        status_code=status.HTTP_200_OK,
        responses={
            400: {"model": ErrorResponse, "description": "Bad upload."},
            413: {"model": ErrorResponse, "description": "File too large."},
        },
        summary="Upload + ingest a PDF/TXT/MD document.",
    )
    async def upload_document(
        request: Request,
        file: UploadFile = File(..., description="PDF, TXT, or MD file (≤50MB)."),
        doc_type: str = Form(DEFAULT_DOC_TYPE),
        settings: Settings = Depends(get_settings),
        doc_store: VectorStore = Depends(get_doc_store),
        metadata: MetadataStore = Depends(get_metadata),
    ) -> JSONResponse:
        """Upload a document and run the ingest pipeline.

        Flow
        ----
        1. Sanitise the filename + validate extension.
        2. Validate the ``doc_type`` form value.
        3. Stream the body to a tempfile, aborting at
           :data:`MAX_UPLOAD_BYTES` with 413.
        4. Call :func:`scripts.ingest.run_ingest` with the
           configured ``settings`` so the data lands in the same
           files the singleton :class:`MetadataStore` and
           :class:`FAISSStore` are pointing at.
        5. On success, refresh ``app.state.doc_store`` from disk
           so subsequent ``/api/query`` calls see the new vectors.
        6. Always unlink the tempfile.
        7. Return the :class:`IngestionReport.to_dict` shape —
           HTTP 200 if ``ok``, 400 otherwise so the dashboard can
           surface ``body["error"]`` in its toast.
        """
        # ---- 1. Sanitise filename + extension -------------------
        safe_name = _safe_filename(file.filename or "upload")
        suffix = Path(safe_name).suffix.lower() if safe_name else ""
        if suffix not in ALLOWED_EXTENSIONS:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error="unsupported_file_type",
                    detail=(
                        f"extension {suffix!r} not in "
                        f"{sorted(ALLOWED_EXTENSIONS)}"
                    ),
                ).model_dump(),
            )

        # ---- 2. Validate doc_type ------------------------------
        if doc_type not in SUPPORTED_DOC_TYPES:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error="invalid_doc_type",
                    detail=(
                        f"doc_type {doc_type!r} not in "
                        f"{sorted(SUPPORTED_DOC_TYPES)}"
                    ),
                ).model_dump(),
            )

        # ---- 3. Stream upload to tempfile + size cap ------------
        tmp_path, tmp_fh = _make_tempfile(suffix)
        bytes_written = 0
        try:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    tmp_fh.close()
                    Path(tmp_path).unlink(missing_ok=True)
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content=ErrorResponse(
                            error="file_too_large",
                            detail=(
                                f"file exceeds "
                                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB cap"
                            ),
                        ).model_dump(),
                    )
                tmp_fh.write(chunk)
            tmp_fh.flush()
            tmp_fh.close()  # release the OS lock before run_ingest reopens
        except Exception:
            tmp_fh.close()
            Path(tmp_path).unlink(missing_ok=True)
            raise

        # ---- 4. Run the pipeline --------------------------------
        report: IngestionReport
        try:
            report = run_ingest(
                path=tmp_path,
                settings=settings,
                doc_type=doc_type,
                embedder_kind="fake",  # Step 4.18 default; Step 4.22 may override
                db_path_override=None,
                index_path_override=None,
            )
        finally:
            # ---- 6. Always clean up the tempfile -----------------
            Path(tmp_path).unlink(missing_ok=True)

        # ---- 5. Pick up new vectors on the happy path -----------
        if report.ok:
            # Overwrite the ingest-script's ``filename = path.name``
            # (which would be the tempfile's randomised basename,
            # e.g. ``tmp6uw572ef.pdf``) with the user-supplied
            # basename. Also overwrite ``source_path`` so the
            # dashboard's "Open in Finder" link doesn't point at a
            # deleted tempfile. ``content_hash`` + ``size_bytes``
            # + ``num_chunks`` etc. are left untouched — those are
            # derived from the actual bytes and are authoritative.
            try:
                metadata.update_document_provenance(
                    report.doc_id,
                    filename=safe_name,
                    source_path=str(tmp_path),  # now-unlinked; provenance only
                )
            except Exception as exc:
                _log.warning(
                    "doc_provenance_update_failed_after_upload",
                    extra={"doc_id": report.doc_id, "error": str(exc)},
                )
            try:
                doc_store.load()
            except Exception as exc:
                # Reload failure shouldn't poison the response —
                # the data is on disk, the dashboard can refresh.
                _log.warning(
                    "doc_store_reload_failed_after_upload",
                    extra={"doc_id": report.doc_id, "error": str(exc)},
                )

        # ---- 7. Map ok=False to 400 -----------------------------
        status_code = (
            status.HTTP_200_OK if report.ok else status.HTTP_400_BAD_REQUEST
        )
        return JSONResponse(
            status_code=status_code, content=report.to_dict()
        )

    # -----------------------------------------------------------------
    # GET /api/documents
    # -----------------------------------------------------------------
    @router.get(
        "",
        response_model=DocumentListResponse,
        summary="List ingested documents (paginated).",
    )
    async def list_documents_route(
        request: Request,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        metadata: MetadataStore = Depends(get_metadata),
    ) -> DocumentListResponse:
        """Return one page of ingested documents, newest-ingested first.

        The ``count`` field is the **total** document count (not
        the page size) so the dashboard can render the "N
        documents" header without a follow-up request.
        ``next_offset`` is ``None`` once the page is the last
        one — the dashboard uses it to disable its "Next" button.
        """
        total = metadata.count_documents()
        rows = metadata.list_documents(limit=limit, offset=offset)
        items = [
            DocumentListItemResponse(
                id=r.id,
                filename=r.filename,
                doc_type=r.doc_type,
                source_path=r.source_path,
                size_bytes=r.size_bytes,
                num_chunks=r.num_chunks,
                content_hash=r.content_hash,
                ingested_at=r.ingested_at,
                last_modified=r.last_modified,
                metadata_json=r.metadata_json,
            )
            for r in rows
        ]
        next_offset: int | None = None
        if offset + limit < total:
            next_offset = offset + limit
        return DocumentListResponse(
            documents=items,
            count=total,
            limit=limit,
            offset=offset,
            next_offset=next_offset,
        )

    # -----------------------------------------------------------------
    # DELETE /api/documents/{document_id}
    # -----------------------------------------------------------------
    @router.delete(
        "/{document_id}",
        response_model=DocumentDeleteResponse,
        responses={
            404: {"model": ErrorResponse, "description": "Document not found."},
        },
        summary="Delete a document + its vectors.",
    )
    async def delete_document_route(
        document_id: str,
        request: Request,
        metadata: MetadataStore = Depends(get_metadata),
        doc_store: VectorStore = Depends(get_doc_store),
    ) -> JSONResponse:
        """Cascade-delete a document.

        Order matters:

        1. Look up the chunk IDs (needed to address FAISS).
        2. Remove the FAISS vectors.
        3. Delete the document row (the SQLite
           ``ON DELETE CASCADE`` FK removes the chunks).
        4. Persist the FAISS sidecar via
           :meth:`VectorStore.save`.
        5. If step 3 deleted zero rows, return 404 — the FAISS
           removal in step 2 was a no-op against a non-existent
           doc and is silently absorbed.

        Returns precise ``chunks_removed`` / ``vectors_removed``
        counts so the dashboard can render a "Deleted document X
        + 44 chunks / 44 vectors" toast without a follow-up GET.
        """
        chunks = metadata.get_chunks_by_document(document_id)
        vectors_removed = doc_store.remove_ids([c.id for c in chunks])
        rows_deleted = metadata.delete_document(document_id)
        if rows_deleted == 0:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=ErrorResponse(
                    error="document_not_found",
                    detail=f"no document with id {document_id!r}",
                ).model_dump(),
            )
        doc_store.save()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=DocumentDeleteResponse(
                document_id=document_id,
                chunks_removed=len(chunks),
                vectors_removed=vectors_removed,
            ).model_dump(),
        )

    return router


__all__ = [
    "ALLOWED_EXTENSIONS",
    "DEFAULT_DOC_TYPE",
    "MAX_UPLOAD_BYTES",
    "SUPPORTED_DOC_TYPES",
    "_safe_filename",  # exposed for tests
    "build_docs_router",
]
