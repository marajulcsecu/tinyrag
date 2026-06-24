"""Project skeleton integrity tests (Step 4.1).

This test file is the safety net for the project layout. If a future
contributor accidentally deletes an ``__init__.py``, renames a
subpackage, or breaks a top-level import, these tests fail *before*
the rest of the suite gets a chance to mask the problem with confusing
collection errors.

The expected layout is defined in
``docs/03_architecture_v1.md`` Section 5. If you change the layout,
update both this file and the architecture doc in the same commit.

What this test guards
---------------------
- Every subpackage listed in the architecture doc exists as a
  directory under ``src/tinyrag/`` and is importable.
- Every subpackage has a non-empty ``__init__.py`` (a one-line
  docstring is the minimum bar — see Step 4.1 review notes).
- The two static-asset subdirectories (``ui/static/`` and
  ``ui/templates/``) exist and are tracked by git (via
  ``.gitkeep`` placeholders).
- ``tests/conftest.py`` exists at the canonical location.
- ``tests/test_smoke.py`` exists and is the same one written in
  Step 3.2.

What this test does NOT guard
-----------------------------
- The *contents* of each module — those are the responsibility of the
  individual test files (``test_chunker.py``, ``test_retriever.py``,
  etc.) that arrive in later Phase 4 steps.
- The ``models/`` subpackage — it predates Phase 4 and is covered by
  ``test_download_models.py`` instead.

Location: ``tests/test_skeleton.py``
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# ----------------------------------------------------------------------------
# Constants — the canonical layout from docs/03_architecture_v1.md §5
# ----------------------------------------------------------------------------

# Project root: this file is tests/test_skeleton.py → parent is project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src" / "tinyrag"

# Every subpackage listed in §5, in the same order as the doc.
# ``ui`` is included even though it is a static-asset container — its
# presence as a subpackage is what makes ``tinyrag.ui`` importable.
SUBPACKAGES = [
    "tinyrag",            # top-level package
    "tinyrag.api",
    "tinyrag.core",
    "tinyrag.ingestion",
    "tinyrag.generation",
    "tinyrag.storage",
    "tinyrag.sensors",
    "tinyrag.input_adapters",
    "tinyrag.ui",
    "tinyrag.observability",
    # ``tinyrag.models`` is intentionally NOT listed here — it is a
    # Phase 3 helper subpackage that predates §5. It has its own
    # test file (``test_download_models.py``) and should not gate the
    # Phase 4 skeleton tests.
]

# UI subdirectories that need to be tracked by git even when empty.
UI_SUBDIRS = [
    SRC_ROOT / "ui" / "static",
    SRC_ROOT / "ui" / "templates",
]

# Files that must exist at known locations.
REQUIRED_FILES = [
    SRC_ROOT / "__init__.py",
    PROJECT_ROOT / "tests" / "conftest.py",
    PROJECT_ROOT / "tests" / "test_smoke.py",
    PROJECT_ROOT / "tests" / "test_skeleton.py",
]


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


class TestSubpackagesExist:
    """Every subpackage listed in §5 is a directory on disk."""

    @pytest.mark.parametrize("subpkg", SUBPACKAGES)
    def test_subpackage_directory_exists(self, subpkg: str) -> None:
        """Each subpackage must have a directory under src/tinyrag/."""
        rel_path = Path(*subpkg.split("."))
        full_path = PROJECT_ROOT / "src" / rel_path
        assert full_path.is_dir(), (
            f"Subpackage directory missing: {full_path}\n"
            f"  (architecture: docs/03_architecture_v1.md §5)"
        )

    @pytest.mark.parametrize("subpkg", SUBPACKAGES)
    def test_subpackage_has_init(self, subpkg: str) -> None:
        """Each subpackage directory must contain an __init__.py."""
        rel_path = Path(*subpkg.split("."))
        init_path = PROJECT_ROOT / "src" / rel_path / "__init__.py"
        assert init_path.is_file(), f"__init__.py missing: {init_path}"

    @pytest.mark.parametrize("subpkg", SUBPACKAGES)
    def test_subpackage_init_is_non_empty(self, subpkg: str) -> None:
        """Each __init__.py must contain at least a docstring.

        Empty ``__init__.py`` files are technically valid Python but
        defeat the purpose of having a subpackage: a future
        contributor reading the code should be able to open any
        subpackage's ``__init__.py`` and see *what lives here*.
        """
        rel_path = Path(*subpkg.split("."))
        init_path = PROJECT_ROOT / "src" / rel_path / "__init__.py"
        # Strip leading whitespace and quotes; a docstring or comment
        # line satisfies this check.
        text = init_path.read_text(encoding="utf-8").strip()
        assert len(text) > 0, f"__init__.py is empty: {init_path}"


class TestSubpackagesImport:
    """Every subpackage can be imported without error."""

    @pytest.mark.parametrize("subpkg", SUBPACKAGES)
    def test_import_subpackage(self, subpkg: str) -> None:
        """``import tinyrag.<subpkg>`` must succeed."""
        try:
            importlib.import_module(subpkg)
        except ImportError as exc:
            pytest.fail(f"Cannot import {subpkg}: {exc}")


class TestUiSubdirsTracked:
    """The UI static-asset directories are tracked by git."""

    @pytest.mark.parametrize("ui_dir", UI_SUBDIRS)
    def test_ui_subdir_exists(self, ui_dir: Path) -> None:
        """The UI subdirectory must exist (created in Step 4.1)."""
        assert ui_dir.is_dir(), f"UI subdirectory missing: {ui_dir}"

    @pytest.mark.parametrize("ui_dir", UI_SUBDIRS)
    def test_ui_subdir_has_gitkeep(self, ui_dir: Path) -> None:
        """Each empty UI subdir must have a .gitkeep so git tracks it.

        Until the actual CSS/JS/HTML files are added in Steps 4.21,
        4.22, 4.23, these directories would be empty and therefore
        invisible to git. The ``.gitkeep`` placeholder is the
        conventional fix.
        """
        gitkeep = ui_dir / ".gitkeep"
        assert gitkeep.is_file(), f".gitkeep placeholder missing: {gitkeep}"


class TestConventionsPresent:
    """The standard pytest infrastructure is in place."""

    def test_conftest_exists(self) -> None:
        """``tests/conftest.py`` is the anchor for shared fixtures."""
        conftest = PROJECT_ROOT / "tests" / "conftest.py"
        assert conftest.is_file(), f"conftest.py missing: {conftest}"

    def test_conftest_has_docstring(self) -> None:
        """``conftest.py`` should document the shared-fixtures policy."""
        conftest = PROJECT_ROOT / "tests" / "conftest.py"
        text = conftest.read_text(encoding="utf-8").strip()
        assert text, "conftest.py is empty — it should at minimum have a docstring"

    def test_smoke_test_still_present(self) -> None:
        """``test_smoke.py`` from Step 3.2 must still exist."""
        smoke = PROJECT_ROOT / "tests" / "test_smoke.py"
        assert smoke.is_file(), f"test_smoke.py missing: {smoke}"
        # It should be non-trivial — at least the Python-version check.
        text = smoke.read_text(encoding="utf-8")
        assert "test_python_version" in text, (
            "test_smoke.py is missing the test_python_version check "
            "added in Step 3.2"
        )


class TestSkeletonIsolated:
    """The skeleton does not import any heavy third-party libraries.

    This is a cheap guard: if a future contributor accidentally adds
    ``import faiss`` or ``import fastapi`` to a top-level
    ``__init__.py``, the smoke-test import check (``make smoke``) will
    fail on a machine that hasn't installed the runtime deps yet. We
    want the *skeleton* to be importable with only the Python standard
    library.
    """

    @pytest.mark.parametrize("subpkg", SUBPACKAGES)
    def test_init_does_not_import_runtime_deps(self, subpkg: str) -> None:
        """No ``__init__.py`` may import a runtime dependency.

        Runtime deps are listed in ``requirements.txt`` and the
        ``RUNTIME_DEPS`` set in ``test_smoke.py``. Adding any of them
        to a skeleton ``__init__.py`` would create a circular ordering
        problem: you couldn't even ``import tinyrag`` before
        ``make install`` had run.
        """
        rel_path = Path(*subpkg.split("."))
        init_path = PROJECT_ROOT / "src" / rel_path / "__init__.py"
        text = init_path.read_text(encoding="utf-8")
        # Cheap regex-free check: each banned token is a top-level
        # package name (e.g. "faiss" but not "faiss_cpp"). A
        # ``from __future__ import annotations`` import is fine.
        banned = (
            "import faiss",
            "import fastapi",
            "import sentence_transformers",
            "import torch",
            "import structlog",
            "import pydantic",
            "import yaml",
            "import pdfplumber",
        )
        for token in banned:
            assert token not in text, (
                f"{init_path} contains '{token}' — skeleton __init__.py "
                f"files must not import runtime dependencies."
            )
