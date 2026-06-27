"""Global FastAPI exception handlers — map domain errors to HTTP responses.

Every exception raised anywhere in the request lifecycle
(dependency providers, route handlers, Pydantic validation, the
llama-server client, the FAISS store) flows through one of the
handlers installed by :func:`install_exception_handlers`. The goal
is **uniform JSON shape** + **no Python tracebacks leaking to the
client** + **the right HTTP status code** for every failure mode.

Why custom handlers (not FastAPI's defaults)?
----------------------------------------------
FastAPI's default behaviour is:

- Return a ``422 Unprocessable Entity`` with the raw Pydantic
  error list when the request body fails validation. That's
  fine for the dashboard but unhelpful when the failure happens
  deep inside the LLM call (where there's no request body to
  blame).
- Return a bare ``500 Internal Server Error`` with a generic
  ``"Internal Server Error"`` body for any unhandled exception.
  That hides everything, including bugs we'd want to see in the
  structured log.

The handlers here:

- Convert every domain exception class to the right HTTP code
  (``LLMUnavailableError`` → ``503``, ``VectorStoreError`` →
  ``500``, ``ValueError`` → ``400``, etc.).
- Always emit the :class:`tinyrag.api.schemas.ErrorResponse`
  shape so the client has a single parser.
- Log the *full* traceback to the structured log but only ship
  the ``str(exc)`` to the client (no stack frames, no
  ``__class__.__name__`` noise).

The catch-all handler
---------------------
The last handler installed (``Exception``) catches anything we
didn't anticipate. It always logs at ``error`` level with the
traceback. Without this, a single uncaught ``KeyError`` would
silently become a bare 500 and we'd lose the breadcrumbs.

Location: ``src/tinyrag/api/errors.py``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from tinyrag.api.schemas import ErrorResponse
from tinyrag.observability.logger import get_logger

if TYPE_CHECKING:
    pass


_log = get_logger(__name__)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _error_payload(code: str, detail: str) -> dict[str, str]:
    """Build a JSON-safe dict matching :class:`ErrorResponse`.

    Returns a plain dict (not the Pydantic model) so the
    ``JSONResponse`` constructor doesn't double-encode. The shape
    is exactly what the dashboard's error widget parses.
    """
    return ErrorResponse(error=code, detail=detail).model_dump()


def _json(error: str, detail: str, status_code: int) -> JSONResponse:
    """Build a JSONResponse with the uniform error shape."""
    return JSONResponse(
        status_code=status_code,
        content=_error_payload(error, detail),
    )


# ----------------------------------------------------------------------------
# Exception → HTTP mapping
# ----------------------------------------------------------------------------


async def _on_value_error(_: Request, exc: ValueError) -> JSONResponse:
    """Map a ``ValueError`` to a 400 with a clear detail.

    ``ValueError`` is what most domain modules raise for
    "the caller passed garbage" (bad threshold range, bad
    encoding name, etc.). Pydantic handles request-body
    validation separately, so this handler only fires for
    runtime ValueErrors inside the pipeline.
    """
    return _json("value_error", str(exc), status.HTTP_400_BAD_REQUEST)


async def _on_pydantic_validation(
    _: Request, exc: ValidationError | RequestValidationError
) -> JSONResponse:
    """Map Pydantic validation errors to 422 with field-level detail.

    The dashboard uses the ``detail`` field to highlight the
    offending input on the form, so we concatenate the per-field
    errors with ``"; "`` rather than the default JSON array.
    """
    if isinstance(exc, RequestValidationError):
        # FastAPI wraps the Pydantic errors in its own object.
        raw_errors = exc.errors()
    else:
        raw_errors = exc.errors()
    bits: list[str] = []
    for err in raw_errors:
        loc = ".".join(str(p) for p in err.get("loc", []))
        msg = err.get("msg", "invalid value")
        bits.append(f"{loc}: {msg}" if loc else msg)
    return _json("validation_error", "; ".join(bits), status.HTTP_422_UNPROCESSABLE_ENTITY)


async def _on_llm_unavailable(_: Request, exc: Exception) -> JSONResponse:
    """Map ``LLMUnavailableError`` → 503 (the llama.cpp server is down)."""
    return _json(
        "llm_unavailable",
        str(exc),
        status.HTTP_503_SERVICE_UNAVAILABLE,
    )


async def _on_llm_refused(_: Request, exc: Exception) -> JSONResponse:
    """Map ``LLMRefusedError`` → 502 (llama-server responded but said no)."""
    return _json("llm_refused", str(exc), status.HTTP_502_BAD_GATEWAY)


async def _on_metadata_error(_: Request, exc: Exception) -> JSONResponse:
    """Map any ``MetadataError`` subclass → 500 (SQLite problem)."""
    return _json("metadata_error", str(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)


async def _on_vector_store_error(_: Request, exc: Exception) -> JSONResponse:
    """Map any ``VectorStoreError`` subclass → 500 (FAISS problem)."""
    return _json("vector_store_error", str(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)


async def _on_retriever_error(_: Request, exc: Exception) -> JSONResponse:
    """Map any ``RetrieverError`` subclass → 500 (pipeline problem)."""
    return _json("retriever_error", str(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)


async def _on_config_error(_: Request, exc: Exception) -> JSONResponse:
    """Map any ``ConfigError`` subclass → 500 (settings problem)."""
    return _json("config_error", str(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)


async def _on_catch_all(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for any unhandled exception.

    Logs at ``error`` level with the full traceback so the
    operator can find it in the structured log, but returns
    only ``"internal server error"`` to the client. We never
    ship the traceback string — it can leak file paths, SQL
    fragments, etc.
    """
    _log.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        exc_type=type(exc).__name__,
        exc_message=str(exc),
        exc_info=True,
    )
    return _json(
        "internal_server_error",
        "internal server error",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


# ----------------------------------------------------------------------------
# Installation
# ----------------------------------------------------------------------------


def install_exception_handlers(app: FastAPI) -> None:
    """Wire every handler above onto ``app``.

    Idempotent — calling twice replaces the handlers (FastAPI's
    ``add_exception_handler`` is last-wins for the same class).
    Called once from :func:`tinyrag.main.create_app`.

    The order of registration doesn't matter (handlers are
    matched by exception class), but the catch-all ``Exception``
    handler MUST be registered after every specific subclass or
    it would shadow them.
    """
    # Built-in / stdlib exceptions
    app.add_exception_handler(ValueError, _on_value_error)
    app.add_exception_handler(ValidationError, _on_pydantic_validation)
    app.add_exception_handler(RequestValidationError, _on_pydantic_validation)

    # Domain exceptions. Importing here (not at module top) keeps
    # the import graph shallow: this module can be loaded even if
    # the domain packages are mid-edit.
    try:
        from tinyrag.config import ConfigError

        app.add_exception_handler(ConfigError, _on_config_error)
    except ImportError:  # pragma: no cover
        pass

    try:
        from tinyrag.storage.metadata import MetadataError

        app.add_exception_handler(MetadataError, _on_metadata_error)
    except ImportError:  # pragma: no cover
        pass

    try:
        from tinyrag.storage.vector_store import VectorStoreError

        app.add_exception_handler(VectorStoreError, _on_vector_store_error)
    except ImportError:  # pragma: no cover
        pass

    try:
        from tinyrag.core.retriever import RetrieverError

        app.add_exception_handler(RetrieverError, _on_retriever_error)
    except ImportError:  # pragma: no cover
        pass

    try:
        from tinyrag.generation.llm_client import LLMRefusedError, LLMUnavailableError

        app.add_exception_handler(LLMUnavailableError, _on_llm_unavailable)
        app.add_exception_handler(LLMRefusedError, _on_llm_refused)
    except ImportError:  # pragma: no cover
        pass

    # Catch-all LAST so it doesn't shadow the specific handlers.
    app.add_exception_handler(Exception, _on_catch_all)


# jsonable_encoder re-exported here so route modules don't need
# to import from fastapi.encoders directly — keeps the import
# surface narrow.
__all__ = [
    "install_exception_handlers",
    "jsonable_encoder",
]
