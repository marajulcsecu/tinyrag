"""Document ingestion pipeline — PDF/TXT/MD → vector store.

The :mod:`tinyrag.ingestion` subpackage orchestrates the
*document-to-vector-store* flow: read raw files, chunk them, embed
them, and write the embeddings + metadata to :mod:`tinyrag.storage`.

It is the seam between the file system (uploads) and the persistence
layer (vector store + metadata DB). It depends on
:mod:`tinyrag.core` (for the chunker) and :mod:`tinyrag.storage` (for
the writes); it is itself called from the FastAPI document routes in
:mod:`tinyrag.api` and from the CLI script ``scripts/ingest.py``.

Modules
-------
- :mod:`tinyrag.ingestion.parsers` — PDF / TXT / MD → raw text
  (Step 4.4). Provides :func:`~tinyrag.ingestion.parsers.parse`,
  :class:`~tinyrag.ingestion.parsers.DocumentParser` Protocol,
  and :class:`~tinyrag.ingestion.parsers.ParsedDocument` dataclass.
- :mod:`tinyrag.ingestion.embedder` — wrapper around
  ``sentence-transformers`` behind a Protocol (Step 4.6).
- :mod:`tinyrag.ingestion.pipeline` — orchestrator: parse → chunk →
  embed → store (Step 4.9).

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

from tinyrag.ingestion.embedder import (
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingModel,
    EmbeddingModelNotFoundError,
    FakeEmbedder,
    SentenceTransformerEmbedder,
)
from tinyrag.ingestion.parsers import (
    DocumentParser,
    EmptyDocumentError,
    MarkdownParser,
    ParsedDocument,
    ParserError,
    PdfParser,
    PdfReadError,
    TxtParser,
    UnsupportedFormatError,
    parse,
)

__all__ = [
    # Parsers (Step 4.4)
    "DocumentParser",
    "EmptyDocumentError",
    "MarkdownParser",
    "ParsedDocument",
    "ParserError",
    "PdfParser",
    "PdfReadError",
    "TxtParser",
    "UnsupportedFormatError",
    "parse",
    # Embedder (Step 4.6)
    "EmbeddingDimensionMismatchError",
    "EmbeddingError",
    "EmbeddingModel",
    "EmbeddingModelNotFoundError",
    "FakeEmbedder",
    "SentenceTransformerEmbedder",
]
