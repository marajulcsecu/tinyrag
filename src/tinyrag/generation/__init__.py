"""Generation subsystem — wraps the LLM behind a swappable Protocol.

The :mod:`tinyrag.generation` subpackage is the only place in TinyRAG
that talks to the language model. It exposes a single
:class:`LLMClient` Protocol so the rest of the system can be tested
without an LLM, and so swapping llama.cpp for (say) Ollama or a remote
API in the future is a 2-line change in the composition root.

Public surface
--------------
- :class:`LLMClient` — the Protocol every backend must satisfy.
- :class:`FakeLLMClient` — a deterministic stub for unit tests.
- :class:`LlamaCppClient` — the real client (talks to llama-server's
  ``/v1/chat/completions`` SSE endpoint).
- :class:`LLMError`, :class:`LLMUnavailableError`,
  :class:`LLMRefusedError` — typed exceptions.

Why a Protocol (not an ABC)?
----------------------------
- Protocols are duck-typed at runtime; the type checker enforces
  structural conformance. This means a test double doesn't need to
  inherit from a base class.
- Aligns with the architecture document (``docs/03_architecture_v1.md``
  §6.4) which already calls for a Protocol interface.

Location: ``src/tinyrag/generation/``
"""

from __future__ import annotations

from tinyrag.generation.llm_client import (
    ChatMessage,
    FakeLLMClient,
    LlamaCppClient,
    LLMClient,
    LLMError,
    LLMRefusedError,
    LLMUnavailableError,
)

__all__ = [
    "LLMClient",
    "ChatMessage",
    "FakeLLMClient",
    "LlamaCppClient",
    "LLMError",
    "LLMUnavailableError",
    "LLMRefusedError",
]
