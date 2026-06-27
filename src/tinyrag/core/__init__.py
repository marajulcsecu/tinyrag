"""Domain logic — pure functions, no I/O, no third-party service calls.

The :mod:`tinyrag.core` subpackage holds the *brain* of TinyRAG. Every
module here is allowed to depend only on the Python standard library,
:mod:`tinyrag.generation` (for the LLM Protocol type), and other
modules inside :mod:`tinyrag.core`. They must NOT import from
:mod:`tinyrag.api`, :mod:`tinyrag.ingestion`, :mod:`tinyrag.storage`,
:mod:`tinyrag.sensors`, or :mod:`tinyrag.ui`.

This one-way dependency rule is what makes the domain logic unit-testable
without spinning up FAISS, llama-server, or a FastAPI app.

Modules
-------
- :mod:`tinyrag.core.chunker` — token-based text chunking with
  overlap (Step 4.5). Provides :class:`~tinyrag.core.chunker.Chunk`
  dataclass and :class:`~tinyrag.core.chunker.Chunker` class.
- :mod:`tinyrag.core.retriever` — query → top-k chunks (wraps the
  vector store + metadata store behind a Protocol) (Step 4.12).
  Provides :class:`~tinyrag.core.retriever.Retriever` and
  :class:`~tinyrag.core.retriever.RetrievalResult`.
- :mod:`tinyrag.core.prompt_builder` — context + query → grounded
  prompt string (Step 4.11). Provides
  :class:`~tinyrag.core.prompt_builder.PromptBuilder` and
  :class:`~tinyrag.core.prompt_builder.Prompt`.
- :mod:`tinyrag.core.answer` — the dataclass for a final answer +
  citation list (Step 4.16). Provides :class:`~tinyrag.core.answer.Answer`
  and :class:`~tinyrag.core.answer.Citation`.
- :mod:`tinyrag.core.sensor_summarizer` — sensor data →
  text-summary chunks for indexing (Step 4.14). Provides
  :class:`~tinyrag.core.sensor_summarizer.SensorSummarizer` and
  the :class:`~tinyrag.core.sensor_summarizer.SensorSummarizerError`
  exception hierarchy.

Why no I/O?
-----------
- Pure functions are trivially testable. The ``test_chunker.py`` suite
  in Step 4.5 runs in milliseconds with no fixtures.
- Pure functions are trivial to swap. A future "use a different
  retriever" change is a one-class swap in the composition root
  (``main.py``, Step 4.17), not a refactor across the codebase.
- Pure functions cannot accidentally talk to the network. The
  architecture's "no cloud calls at runtime" guarantee is enforced
  structurally, not just by code review.

Location: ``src/tinyrag/core/``
"""

from __future__ import annotations

from tinyrag.core.answer import (
    Answer,
    Citation,
    build_citations,
    build_citations_from_chunks,
)
from tinyrag.core.chunker import Chunk, Chunker, ChunkingError, default_chunker
from tinyrag.core.prompt_builder import (
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_RESERVED_FOR_ANSWER_TOKENS,
    DEFAULT_SYSTEM_PROMPT,
    USER_MESSAGE_TEMPLATE,
    Prompt,
    PromptBuilder,
    PromptBuilderError,
    default_prompt_builder,
)
from tinyrag.core.retriever import (
    DEFAULT_K_DOC,
    DEFAULT_K_SENSOR,
    DEFAULT_SENSOR_KEYWORDS,
    DEFAULT_THRESHOLD,
    MetadataAccessor,
    RetrievalResult,
    Retriever,
    RetrieverEmbedError,
    RetrieverError,
    RetrieverMetadataError,
    RetrieverSearchError,
    SMALL_CORPUS_MAX_CHUNKS,
    SMALL_CORPUS_THRESHOLD,
    adapt_metadata_store,
)
from tinyrag.core.sensor_summarizer import (
    SensorSummarizer,
    SensorSummarizerEmptyError,
    SensorSummarizerError,
    SensorSummarizerSchemaError,
)

__all__ = [
    "Answer",
    "Chunk",
    "Chunker",
    "ChunkingError",
    "Citation",
    "DEFAULT_K_DOC",
    "DEFAULT_K_SENSOR",
    "DEFAULT_MAX_PROMPT_TOKENS",
    "DEFAULT_RESERVED_FOR_ANSWER_TOKENS",
    "DEFAULT_SENSOR_KEYWORDS",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_THRESHOLD",
    "MetadataAccessor",
    "Prompt",
    "PromptBuilder",
    "PromptBuilderError",
    "RetrievalResult",
    "Retriever",
    "RetrieverEmbedError",
    "RetrieverError",
    "RetrieverMetadataError",
    "RetrieverSearchError",
    "SMALL_CORPUS_MAX_CHUNKS",
    "SMALL_CORPUS_THRESHOLD",
    "SensorSummarizer",
    "SensorSummarizerEmptyError",
    "SensorSummarizerError",
    "SensorSummarizerSchemaError",
    "USER_MESSAGE_TEMPLATE",
    "adapt_metadata_store",
    "build_citations",
    "build_citations_from_chunks",
    "default_chunker",
    "default_prompt_builder",
]
