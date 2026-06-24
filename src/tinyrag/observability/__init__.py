"""Observability — structured logging + (future) metrics.

The :mod:`tinyrag.observability` subpackage is the only place in
TinyRAG that should call ``logging.getLogger`` or a third-party
structured-logging library. The rest of the codebase receives a
configured logger via :func:`tinyrag.observability.get_logger`.

Modules
-------
- :mod:`tinyrag.observability.logger` — structured logger built on
  :mod:`structlog` (already pinned in ``requirements.txt``).
  Provides :func:`~tinyrag.observability.logger.configure_logging`
  (called once at startup in :mod:`tinyrag.main`) and
  :func:`~tinyrag.observability.logger.get_logger` (called from
  every other module).

Why a dedicated subpackage?
---------------------------
- The architecture document (§5.2) lists "professional logging" as a
  non-negotiable principle. A dedicated subpackage makes it easy to
  audit "is anyone using ``print()`` or ``logging.basicConfig``?".
- Future contributors may add Prometheus metrics, OpenTelemetry
  tracing, or health-check probes. Putting them next to the logger
  keeps observability concerns in one folder.

Public surface
--------------
- :func:`configure_logging` — one-shot setup (called from
  :mod:`tinyrag.main`).
- :func:`get_logger` — called from every other module.
- :class:`LoggingError` — typed exception for config failures.

Location: ``src/tinyrag/observability/``
"""

from __future__ import annotations

from tinyrag.observability.logger import (
    LoggingError,
    configure_logging,
    get_logger,
)

__all__ = [
    "LoggingError",
    "configure_logging",
    "get_logger",
]
