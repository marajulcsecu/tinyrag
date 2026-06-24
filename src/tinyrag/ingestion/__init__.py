"""Document ingestion pipeline — PDF/TXT/MD → vector store.

The :mod:`tinyrag.ingestion` subpackage orchestrates the
*document-to-vector-store* flow: read raw files, chunk them, embed
them, and write the embeddings + metadata to :mod:`tinyrag.storage`.

It is the seam between the file system (uploads) and the persistence
layer (vector store + metadata DB). It depends on
:mod:`tinyrag.core` (for the chunker) and :mod:`tinyrag.storage` (for
the writes); it is itself called from the FastAPI document routes in
:mod:`tinyrag.api` and from the CLI script ``scripts/ingest.py``.

Modules (to be added in later Phase 4 steps)
--------------------------------------------
- :mod:`tinyrag.ingestion.parsers` — PDF / TXT / MD → raw text.
- :mod:`tinyrag.ingestion.embedder` — wrapper around
  ``sentence-transformers`` behind a Protocol.
- :mod:`tinyrag.ingestion.pipeline` — orchestrator: parse → chunk →
  embed → store.

Why a subpackage and not a single file?
---------------------------------------
- Parsers change for different reasons (add a new file format) than
  the pipeline (add retry, add progress reporting). Keeping them
  separate makes diffs localised.
- The embedder is the most likely place a future contributor will want
  to swap the model (e.g. ``all-MiniLM-L6-v2`` → ``bge-small-en-v1.5``).
  Hiding it behind a Protocol means the swap is a config change.
- The pipeline is the unit that the test suite (and later, the
  ``scripts/ingest.py`` CLI) actually calls; the other two modules
  are its building blocks.

Location: ``src/tinyrag/ingestion/``
"""

from __future__ import annotations

# Subpackage is currently a placeholder. Modules will be re-exported
# here as they are implemented in later Phase 4 steps (4.4, 4.6, 4.9).
