"""Lightweight system-introspection helpers for ``GET /api/status``.

These are the small, never-fail probes the status endpoint calls:
process RSS, llama.cpp reachability. They live in their own module
because they're **infrastructure-y** — they touch the OS / network,
not the RAG domain — and they need their own try/except wrappers so
a missing ``/proc`` or unreachable llama-server never crashes the
status endpoint.

Why no ``psutil`` dep?
----------------------
``psutil`` would give us cross-platform RSS in one line, but it's
another 1.5 MB dependency and the only thing we use it for. Linux's
``/proc/self/statm`` (and macOS's ``resource.getrusage``) is good
enough for a "approximately how much RAM am I using" dashboard
field, which is all FR-39 asks for. The :func:`get_ram_mb`
function falls back to ``None`` if the platform doesn't expose RSS
cheaply — that's the right answer for a status field.

Why no async?
-------------
Each helper is a single ``open()`` + ``read()`` or a single
``httpx.get()`` call. The cost is microseconds. Wrapping them in
``async`` would add complexity (the status endpoint would have
to ``await`` four probes instead of just calling them) without
measurable benefit. They run inside FastAPI's threadpool because
``run_in_threadpool`` is the default for ``def`` (non-``async``)
dependencies in FastAPI 0.115+.

Location: ``src/tinyrag/api/system_info.py``
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from tinyrag.observability.logger import get_logger

_log = get_logger(__name__)


# ----------------------------------------------------------------------------
# RAM probe
# ----------------------------------------------------------------------------


def _read_rss_kb_from_proc() -> int | None:
    """Read RSS in KiB from ``/proc/self/statm`` (Linux only).

    Returns ``None`` if ``/proc/self/statm`` doesn't exist (macOS,
    Windows) or is unreadable. We never raise — the status endpoint
    must work even on platforms without ``/proc``.
    """
    statm_path = Path("/proc/self/statm")
    try:
        text = statm_path.read_text(encoding="ascii").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    fields = text.split()
    if len(fields) < 2:
        return None
    try:
        # field 1 = resident set size in pages; multiply by page size.
        pages = int(fields[1])
        page_bytes = os.sysconf("SC_PAGESIZE")
        if page_bytes <= 0:
            return None
        return pages * page_bytes // 1024  # KiB
    except (ValueError, OSError):
        return None


def _read_rss_kb_from_resource() -> int | None:
    """Read max-RSS in KiB from ``resource.getrusage`` (POSIX).

    Returns ``None`` on Windows (no ``resource`` module). On macOS
    this is **max** RSS rather than current RSS — the kernel only
    updates the high-water mark at process exit — but it's still a
    useful upper-bound estimate for a status field.
    """
    import resource

    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except (OSError, ValueError):
        return None
    # On Linux, ru_maxrss is in KiB. On macOS, it's in bytes.
    # Detect by platform: the difference is ~1000x for any non-trivial
    # process, so we use a heuristic.
    rss = usage.ru_maxrss
    if rss <= 0:
        return None
    # macOS returns bytes, Linux returns KiB. Heuristic: anything
    # > 100 MB is almost certainly bytes (macOS), anything < 1 GB
    # of "KiB" is almost certainly already KiB (Linux).
    if rss > 100 * 1024 * 1024:
        return rss // 1024  # bytes → KiB
    return rss  # already KiB


def get_ram_mb() -> float | None:
    """Return current process RSS in MB, rounded to 1 dp, or ``None``.

    Tries ``/proc/self/statm`` first (Linux, gives current RSS)
    then ``resource.getrusage`` (POSIX, gives max-RSS). Returns
    ``None`` if neither path works (Windows without Cygwin).

    The value is rounded to 1 dp because the dashboard shows it in
    a status panel — ``"182.4 MB"`` is more useful than ``"182437
    KiB"``, and the underlying number is noisy enough that 2+ dp is
    meaningless.
    """
    rss_kb = _read_rss_kb_from_proc()
    if rss_kb is None:
        rss_kb = _read_rss_kb_from_resource()
    if rss_kb is None:
        return None
    return round(rss_kb / 1024, 1)


# ----------------------------------------------------------------------------
# llama.cpp probe
# ----------------------------------------------------------------------------


def get_llama_cpp_status(server_url: str, *, timeout_s: float = 1.5) -> str:
    """Return ``"up"`` if the llama.cpp ``/health`` endpoint responds ``200``.

    Any other outcome (timeout, connection refused, non-200 status)
    maps to ``"down"``. We never raise — a status probe that itself
    crashes is worse than a status probe that says "down".

    The probe is intentionally cheap: ``/health`` on llama-server
    returns ``"OK"`` with a 200 in microseconds, no auth required.
    The 1.5-second timeout keeps the dashboard snappy even if the
    server is hanging.
    """
    # Lazy import so the test suite (which uses FakeLLMClient and
    # never touches the network) doesn't pay the httpx import cost.
    import httpx

    url = server_url.rstrip("/") + "/health"
    try:
        resp = httpx.get(url, timeout=timeout_s)
    except (httpx.HTTPError, OSError) as exc:
        _log.debug("llama_cpp_health_failed", url=url, error=str(exc))
        return "down"
    return "up" if resp.status_code == 200 else "down"


# ----------------------------------------------------------------------------
# Embedding model name probe
# ----------------------------------------------------------------------------


def get_embedding_model_name(embedder: object) -> str:
    """Return a human-readable model id from an embedder, or ``"unknown"``.

    Tries, in order:

    1. ``embedder.model_name`` (some embedders expose it directly).
    2. ``embedder._model_name`` / ``embedder._model_id`` (private
       but commonly set on sentence-transformers wrappers).
    3. ``type(embedder).__name__`` as a last resort (e.g.
       ``"SentenceTransformerEmbedder"`` or ``"FakeEmbedder"``).

    Never raises. The status field is a string the dashboard
    renders; ``"unknown"`` is acceptable.
    """
    for attr in ("model_name", "_model_name", "_model_id", "model_id"):
        with contextlib.suppress(AttributeError):
            value = getattr(embedder, attr)
            if isinstance(value, str) and value:
                return value
    return type(embedder).__name__


__all__ = [
    "get_embedding_model_name",
    "get_llama_cpp_status",
    "get_ram_mb",
]
