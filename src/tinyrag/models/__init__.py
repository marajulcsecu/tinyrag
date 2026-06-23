"""Model registry and downloader for TinyRAG.

This subpackage owns the *catalog* of GGUF models that TinyRAG can run and
the *mechanism* to fetch them from Hugging Face. It deliberately knows
nothing about llama.cpp's HTTP API, embeddings, or retrieval — its single
responsibility is "given a model id, ensure the GGUF file is on disk and
hasn't been tampered with."

Public surface
--------------
- :data:`MODEL_REGISTRY` — the canonical catalog (used by both the
  downloader and by the Phase 4 ``LLMClient`` factory).
- :class:`ModelEntry` — the dataclass describing one catalog row.
- :class:`ModelDownloader` — the class that actually does the I/O.
- :class:`DownloadError`, :class:`ChecksumMismatchError`,
  :class:`UnknownModelError` — the typed exceptions callers may catch.

Why a dedicated subpackage?
---------------------------
- The catalog and the downloader change for different reasons (a new
  model release vs. a network/HTTP bug) — keeping them in one file
  would conflate the two.
- Future Phase 4 code (``src/tinyrag/generation/llm_client.py``) will
  need ``MODEL_REGISTRY`` to resolve ``config.yaml`` model names to
  on-disk paths. Putting the registry in a small, dependency-free
  module makes that import trivial.
- Unit-testing the SHA-256 logic is easy when the I/O is a small
  class with explicit ``dest_dir`` and ``progress_cb`` parameters.

Location: ``src/tinyrag/models/``
"""

from __future__ import annotations

from tinyrag.models.downloader import (
    ChecksumMismatchError,
    DownloadError,
    ModelDownloader,
    UnknownModelError,
)
from tinyrag.models.registry import MODEL_REGISTRY, ModelEntry

__all__ = [
    "MODEL_REGISTRY",
    "ModelEntry",
    "ModelDownloader",
    "DownloadError",
    "ChecksumMismatchError",
    "UnknownModelError",
]
