"""Persistence layer — FAISS vector store + SQLite metadata.

The :mod:`tinyrag.storage` subpackage owns the *on-disk state* of
TinyRAG: the FAISS index file holding the chunk embeddings, and the
SQLite database holding chunk metadata + the document registry.

It is the seam between the ingestion pipeline (which writes) and the
retriever (which reads). Both the ingestion and retrieval paths go
through this subpackage — neither side ever touches the file system
directly.

Modules (to be added in later Phase 4 steps)
--------------------------------------------
- :mod:`tinyrag.storage.vector_store` — FAISS wrapper (add, search,
  save, load, ntotal).
- :mod:`tinyrag.storage.metadata` — SQLite wrapper (document registry,
  chunk metadata, source-of-truth lookups by id).

Why a subpackage and not a single file?
---------------------------------------
- FAISS and SQLite are different storage engines with different
  failure modes. Keeping them in separate files means a SQLite lock
  bug doesn't risk touching the FAISS code path and vice versa.
- The vector store is append-only during ingestion but mutable during
  reindex; the metadata store is mutable throughout. Conflating
  these lifecycles in one class would be confusing.
- Both modules need to expose a Protocol so they can be swapped (e.g.
  ChromaDB instead of FAISS, Postgres instead of SQLite) without
  touching the call sites.

Location: ``src/tinyrag/storage/``
"""

from __future__ import annotations

# Subpackage is currently a placeholder. Modules will be re-exported
# here as they are implemented in later Phase 4 steps (4.7, 4.8).
