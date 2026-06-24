"""TinyRAG — a privacy-preserving, fully on-device RAG system for smart homes.

TinyRAG (the project) = this Python package (``tinyrag``). It is a small,
self-contained implementation of Retrieval-Augmented Generation designed to
run entirely on a Raspberry Pi 5 (or a developer laptop) without any
cloud calls. The package is intentionally split into subpackages that
mirror the C4 Level 3 module decomposition in
``docs/03_architecture_v1.md`` Section 5:

- :mod:`tinyrag.api` — FastAPI HTTP routes (query, documents, admin).
- :mod:`tinyrag.core` — domain logic with no I/O knowledge (chunker,
  retriever, prompt builder, answer, sensor summarizer).
- :mod:`tinyrag.ingestion` — document-to-vector-store pipeline (parsers,
  embedder, orchestrator).
- :mod:`tinyrag.generation` — the LLM seam (Protocol + concrete clients).
- :mod:`tinyrag.storage` — persistence (FAISS vector store, SQLite metadata).
- :mod:`tinyrag.sensors` — pluggable sensor data sources
  (simulated CSV, real serial, MQTT).
- :mod:`tinyrag.input_adapters` — pluggable user input (text, voice).
- :mod:`tinyrag.ui` — static web assets (HTML/CSS/JS) for the chat page
  and admin page.
- :mod:`tinyrag.observability` — structured logging + future metrics.

There is also :mod:`tinyrag.models`, a Phase 3 helper subpackage that owns
the GGUF model catalog and downloader. It is not part of the C4 diagram
(it predates Phase 4) but is re-used by :mod:`tinyrag.generation` to
resolve ``config.yaml`` model ids to on-disk paths.

Why one package, many subpackages?
----------------------------------
- Subpackages are the natural unit of ownership in Python: one
  ``__init__.py`` = one public surface = one place to document what
  the rest of the system is allowed to depend on.
- Splitting by responsibility (I/O layer, domain logic, persistence,
  etc.) makes the dependency graph one-directional: ``api/`` depends on
  ``core/`` + ``generation/``; nothing in ``core/`` knows ``api/``
  exists. This is what the architecture document calls "no upward
  dependencies" (§5.2).
- Tests can be colocated 1-to-1 with subpackages later
  (``tests/test_chunker.py`` mirrors ``src/tinyrag/core/chunker.py``),
  making the test suite self-explanatory.

Location: ``src/tinyrag/``
"""

from __future__ import annotations

# No symbols are re-exported at the package root. Callers should import
# from the appropriate subpackage (e.g. ``tinyrag.generation.LLMClient``).
# This keeps the public surface explicit and prevents accidental
# cross-coupling.
