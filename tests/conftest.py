"""Shared pytest fixtures for the TinyRAG test suite.

This file is the canonical place for *cross-test* fixtures — fixtures
that more than one test file needs. Single-test fixtures should live
in the test file that uses them.

What belongs here
-----------------
- A ``tmp_data_dir`` fixture pointing at a fresh, gitignored
  ``tmp_path``-based directory for FAISS / SQLite / model files.
- A ``fake_llm_client`` fixture returning a ``FakeLLMClient`` instance
  with deterministic canned responses.
- A ``sample_documents`` fixture pointing at a small in-repo set of
  test PDFs/MDs (added in Step 4.4).
- A ``config_for_tests`` fixture returning a ``Settings`` object
  populated from a test-only ``config.test.yaml`` (added in Step 4.2).

What does NOT belong here
-------------------------
- Module-specific fixtures (e.g. ``sample_embedding`` belongs in
  ``test_embedder.py``).
- Helper functions that are not fixtures.
- Test data files themselves — those live under
  ``tests/fixtures/`` (gitignored if large) or ``tests/data/``.

Why a ``conftest.py``?
----------------------
- Pytest discovers ``conftest.py`` files automatically; no import is
  needed in test files. A fixture defined here is available to every
  test in this directory and all subdirectories.
- Pytest also lets multiple ``conftest.py`` files coexist (one per
  directory). We currently only need this top-level one; future
  subpackage-specific fixtures can live in
  ``tests/ingestion/conftest.py``, etc.

Location: ``tests/conftest.py``
"""

from __future__ import annotations

# Currently empty. Shared fixtures will be added in later Phase 4
# steps as the test suite grows (Step 4.2 introduces the first
# ``config_for_tests`` fixture, Step 4.5 introduces
# ``sample_documents`` and ``fake_llm_client``).
