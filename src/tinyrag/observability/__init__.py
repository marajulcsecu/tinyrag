"""Observability — structured logging + (future) metrics.

The :mod:`tinyrag.observability` subpackage is the only place in
TinyRAG that should call ``logging.getLogger`` or a third-party
structured-logging library. The rest of the codebase receives a
configured logger (or just the standard ``logging`` module) via
dependency injection.

Modules (to be added in later Phase 4 steps)
--------------------------------------------
- :mod:`tinyrag.observability.logger` — structured JSON logger
  built on :mod:`structlog` (already pinned in
  ``requirements.txt``). Provides :func:`configure_logging` (called
  once at startup in :mod:`tinyrag.main`) and :func:`get_logger`
  (called from every other module).

Why a dedicated subpackage?
---------------------------
- The architecture document (§5.2) lists "professional logging" as a
  non-negotiable principle. A dedicated subpackage makes it easy to
  audit "is anyone using ``print()`` or ``logging.basicConfig``?".
- Future contributors may add Prometheus metrics, OpenTelemetry
  tracing, or health-check probes. Putting them next to the logger
  keeps observability concerns in one folder.

Location: ``src/tinyrag/observability/``
"""

from __future__ import annotations

# Subpackage is currently a placeholder. Modules will be re-exported
# here as they are implemented in later Phase 4 steps (4.3).
