"""Structured logging — the project's single seam for log output.

This module is the ONLY place in TinyRAG that configures logging. Every
other module calls :func:`get_logger` and uses the returned object
exactly like a ``structlog.stdlib.BoundLogger``::

    from tinyrag.observability.logger import get_logger
    log = get_logger(__name__)
    log.info("ingestion_complete", doc_id=42, chunks=143, ms=87)

Configuration
-------------
Call :func:`configure_logging` once at startup, passing the
:class:`LoggingSettings` from :mod:`tinyrag.config`::

    from tinyrag.config import load_settings
    from tinyrag.observability.logger import configure_logging

    settings = load_settings("config.yaml")
    configure_logging(settings.logging)

After configuration, ``get_logger(__name__)`` returns a structlog
logger that:

- Emits one **structured event per line** to stdout (human-readable
  by default, JSON when ``json_format: true`` in ``config.yaml``).
- Emits the same events as **JSON to a file** at
  ``logging.path`` (defaults to ``logs/tinyrag.log``; pass ``None``
  in the config to disable file logging).

Why two pipelines?
------------------
The architecture document §12.1 specifies:

> Logs go to: ``stdout`` (for human reading during dev).
>               ``logs/tinyrag.log`` (JSON, append-only, for postmortem).

Stdout is for the developer staring at a terminal — pretty colours
help. The log file is for grep / awk / Loki / whatever the
postmortem toolchain is — JSON is the universal interchange format.
Splitting the two means the operator can set ``json_format: true``
in the config to get JSON on stdout too (useful when piping the
output through ``jq`` or a log shipper), without affecting the
file format that downstream tools already know how to parse.

Why ``structlog.stdlib.ProcessorFormatter``?
--------------------------------------------
Structlog's "bypass stdlib" mode (``structlog.configure`` with a
``LoggerFactory``) is fast but doesn't share handlers with the
standard library — every handler needs its own pipeline. Bridging
to stdlib via ``structlog.stdlib.ProcessorFormatter`` lets us
configure both handlers (stdout + file) once via stdlib's
``dictConfig`` and have structlog emit through it. The single
shared processor chain (timestamp, level, module, etc.) runs
once per log call, and the format choice (JSON vs pretty) is
made per-handler by the formatter — which is exactly what we
want.

Why ``dictConfig`` instead of ``basicConfig``?
----------------------------------------------
``dictConfig`` is the only stdlib API that supports *two handlers
with different formatters*. ``basicConfig`` is a single-handler
shortcut. We need the dual-pipeline design, so ``dictConfig`` it is.

Public surface
--------------
- :func:`configure_logging` — call once at startup.
- :func:`get_logger` — call from every module.
- :class:`LoggingError` — typed exception for config failures.

Location: ``src/tinyrag/observability/logger.py``
"""

from __future__ import annotations

import logging
import logging.config
import sys
from pathlib import Path
from typing import Any

import structlog
from structlog.types import Processor

from tinyrag.config import LoggingSettings, LogLevel

# ----------------------------------------------------------------------------
# Public exceptions
# ----------------------------------------------------------------------------


class LoggingError(Exception):
    """Raised when :func:`configure_logging` cannot set up the loggers.

    Catching this in :mod:`tinyrag.main` lets the composition root
    translate log-config failures into a clean startup error message
    instead of a traceback.
    """


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


# Map from our ``LogLevel`` enum to the stdlib ``logging`` constants.
# Defined once at module load — stdlib levels are stable, and the
# LogLevel enum is a thin wrapper around the same string values.
_LEVEL_MAP: dict[LogLevel, int] = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARNING: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
}


# The processors that run on every log call, before the per-handler
# formatter decides between JSON and pretty. Order matters:
#
# 1. ``add_log_level``        — adds ``level`` key (e.g. "info")
# 2. ``merge_contextvars``    — pulls in contextvars (request_id, etc.)
# 3. ``TimeStamper(fmt=...)`` — adds ``timestamp`` key in ISO 8601
# 4. ``StackInfoRenderer``    — turns ``stack_info=True`` into a string
# 5. ``format_exc_info``      — turns exc_info into a formatted traceback
# 6. ``add_logger_name``      — adds ``logger`` key (the module name)
# 7. ``dict_tracebacks``      — exc_info as a structured dict (JSON-friendly)
#
# ``dict_tracebacks`` runs *after* ``format_exc_info`` would have run,
# but since we use the stdlib bridge the order is handled by the
# ``ProcessorFormatter`` — see the dictConfig below. We list the
# shared processors here so they're applied consistently to both
# handlers.
_SHARED_PROCESSORS: list[Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.stdlib.add_logger_name,
    structlog.processors.StackInfoRenderer(),
    # ``format_exc_info`` runs in the formatter, not the shared
    # chain, because exc_info handling differs by formatter
    # (pretty prints a multi-line traceback, JSON encodes it as a
    # structured dict).
]


def _build_dict_config(
    settings: LoggingSettings,
    log_file_path: Path | None,
) -> dict[str, Any]:
    """Build a stdlib ``dictConfig`` for the two-pipeline layout.

    Parameters
    ----------
    settings:
        The :class:`LoggingSettings` from :mod:`tinyrag.config`.
    log_file_path:
        The resolved path to the log file, or ``None`` to disable
        file logging entirely. The caller (``configure_logging``)
        resolves the relative path against the project root; this
        helper just receives the absolute path.

    Returns
    -------
    dict
        A ``logging.config.dictConfig``-shaped dict.
    """
    level = _LEVEL_MAP[settings.level]

    # ---- Handlers --------------------------------------------------------
    handlers: dict[str, dict[str, Any]] = {
        # Stdout: pretty by default, JSON when json_format=true.
        # ``ProcessorFormatter`` reads ``formatter`` from the
        # ``LogRecord`` to choose its format. We supply two
        # formatters below and let the handler pick.
        "stdout": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "json" if settings.json_format else "pretty",
            "level": level,
        },
    }
    if log_file_path is not None:
        handlers["file"] = {
            "class": "logging.handlers.WatchedFileHandler",
            "filename": str(log_file_path),
            "encoding": "utf-8",
            "formatter": "json",
            "level": level,
        }

    # ---- Formatters -------------------------------------------------------
    # Both formatters run the shared processors via the
    # ``processor_formatter`` key, then apply their final renderer.
    formatters: dict[str, dict[str, Any]] = {
        "pretty": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
            "foreign_pre_chain": _SHARED_PROCESSORS,
        },
        "json": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.processors.JSONRenderer(),
            "foreign_pre_chain": _SHARED_PROCESSORS,
        },
    }

    # ---- Loggers ----------------------------------------------------------
    # Root logger receives the configured handlers. We don't disable
    # propagation on the root so the application code's
    # ``logging.getLogger(__name__)`` calls also flow through the
    # same handlers (useful for third-party libraries that don't
    # use structlog).
    loggers: dict[str, dict[str, Any]] = {
        "": {
            "handlers": list(handlers.keys()),
            "level": level,
            "propagate": True,
        },
        # Quiet down a few well-known chatty libraries. These are
        # the only ones TinyRAG itself depends on; if a future
        # dependency becomes noisy, add it here.
        "httpx": {"level": logging.WARNING},
        "httpcore": {"level": logging.WARNING},
        "sentence_transformers": {"level": logging.WARNING},
    }

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": loggers,
    }


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def configure_logging(
    settings: LoggingSettings,
    *,
    project_root: Path | None = None,
) -> None:
    """Configure the project's structured-logging stack.

    Call this exactly once at process startup (typically from
    :mod:`tinyrag.main`). Calling it more than once is safe but
    wasteful — the dictConfig call replaces the previous config.

    Parameters
    ----------
    settings:
        The :class:`LoggingSettings` from :mod:`tinyrag.config`.
    project_root:
        Anchor for the relative ``settings.path``. If ``None``,
        the file path (if any) is treated as relative to the
        current working directory. Pass
        :meth:`Settings.project_root` to anchor against the
        config file's directory, which is what
        :mod:`tinyrag.main` will do in Step 4.17.

    Raises
    ------
    LoggingError
        The log file's parent directory cannot be created, or the
        stdlib ``dictConfig`` rejected the configuration. The
        original exception is chained via ``raise ... from``.
    """
    # Resolve the log file path up front. Relative paths are
    # anchored to ``project_root`` if given, else to the CWD. We
    # create the parent directory eagerly so a permission error
    # surfaces here (cleaner traceback) instead of at first write.
    log_file_path: Path | None = None
    if settings.path is not None:
        log_file_path = Path(settings.path)
        if not log_file_path.is_absolute() and project_root is not None:
            log_file_path = project_root / log_file_path
        try:
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LoggingError(
                f"cannot create log file directory {log_file_path.parent}: {exc}"
            ) from exc

    # Apply the stdlib config first — this is what wires up the
    # two handlers and their formatters.
    dict_config = _build_dict_config(settings, log_file_path)
    try:
        logging.config.dictConfig(dict_config)
    except (ValueError, TypeError) as exc:
        raise LoggingError(
            f"logging dictConfig failed (this is a bug; please report): {exc}"
        ) from exc

    # Now bridge structlog to stdlib. ``structlog.stdlib.LoggerFactory``
    # creates a stdlib ``logging.Logger`` under the hood, so all
    # ``log.info(...)`` calls flow through the dictConfig above.
    # ``wrap_for_formatter`` is the magic that attaches the
    # shared processor chain to the bound logger, so the formatter
    # sees a fully-processed event dict.
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *_SHARED_PROCESSORS,
            # ``format_exc_info`` runs here (not in the formatter)
            # so both pretty and JSON output get a structured
            # exc_info. ``format_exc_info`` returns the rendered
            # string; the per-handler formatter decides whether
            # to keep it as a string (pretty) or re-encode as JSON.
            structlog.processors.format_exc_info,
            # ``ProcessorFormatter.wrap_for_formatter`` hands off
            # to the formatter chosen by the handler.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog logger.

    Call this from every module that wants to log::

        from tinyrag.observability.logger import get_logger
        log = get_logger(__name__)
        log.info("event_name", key1="value1", key2=42)

    The returned logger is a :class:`structlog.stdlib.BoundLogger`
    — bound to ``name`` (the module name) and configured to
    emit through the stdlib handlers set up by
    :func:`configure_logging`.

    Parameters
    ----------
    name:
        The logger name. Pass ``__name__`` from the calling module
        so logs are filterable by module. ``None`` returns the
        root logger (rarely useful).
    """
    return structlog.get_logger(name)
