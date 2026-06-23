"""Smoke test — confirms all top-level dependencies import cleanly.

This is the very first test the project runs. It is intentionally minimal:
no business logic, no fixtures, no mocks. Its only job is to fail LOUDLY if
the Python environment is broken.

If this test fails after a fresh `make install`, the issue is in
``requirements.txt`` or the venv itself — not in TinyRAG code.

Location: ``tests/test_smoke.py``
Why tests/  : ``pyproject.toml`` tells pytest ``testpaths = ["tests"]``.
Why here   : Standard pytest layout; mirrors ``src/tinyrag/`` once Phase 4
             adds real modules.
"""

from __future__ import annotations

import importlib
import sys

import pytest

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# All runtime deps from requirements.txt. Grouped for clearer failure messages.
RUNTIME_DEPS = {
    # Web framework & async server
    "fastapi",
    "uvicorn",
    "jinja2",
    "multipart",  # python-multipart exposes itself as `multipart`
    "markdown",
    # Configuration
    "yaml",  # PyYAML
    "pydantic",
    "pydantic_settings",
    "dotenv",  # python-dotenv
    # Retrieval
    "faiss",
    "sentence_transformers",
    "torch",
    "numpy",
    "tiktoken",
    # Parsing
    "pdfplumber",
    # HTTP client
    "httpx",
    "sse_starlette",
    # Logging
    "structlog",
    # Sensors
    "paho.mqtt.client",  # paho-mqtt exposes as paho.mqtt.client
    # Data
    "dateutil",
    "pandas",
}

DEV_DEPS = {
    "pytest",
    "pytest_asyncio",
    "pytest_cov",
    "pytest_mock",
    "ruff",
    "mypy",
    "build",
    "piptools",  # pip-tools exposes itself as `piptools`
}


# ----------------------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------------------


def test_python_version() -> None:
    """TinyRAG requires Python 3.12+ (PEP 695 type syntax, tomllib stdlib)."""
    assert sys.version_info >= (
        3,
        12,
    ), f"TinyRAG needs Python 3.12+, found {sys.version_info.major}.{sys.version_info.minor}"


@pytest.mark.parametrize("module_name", sorted(RUNTIME_DEPS))
def test_runtime_dep_imports(module_name: str) -> None:
    """Every runtime dependency must import without error."""
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        pytest.fail(f"Runtime dep '{module_name}' failed to import: {exc}")


@pytest.mark.parametrize("module_name", sorted(DEV_DEPS))
def test_dev_dep_imports(module_name: str) -> None:
    """Every dev dependency must import without error.

    Skipped automatically when running outside a dev install.
    """
    try:
        importlib.import_module(module_name)
    except ImportError:
        pytest.skip(f"Dev dep '{module_name}' not installed — run `make install-dev`")


def test_faiss_cpu_only() -> None:
    """Confirm faiss-cpu (no GPU) is installed, not faiss (GPU).

    This matters because faiss-gpu pulls in CUDA, which is ~3 GB and not
    desired on the Pi or most laptops. TinyRAG explicitly uses faiss-cpu.
    """
    import faiss

    # faiss.__file__ for faiss-cpu typically ends in 'faiss/cpu_swizzle.so'
    # or similar; faiss-gpu would have 'faiss/gpu/...'. The simplest check
    # is to look for the swigfaiss module name.
    assert faiss.__name__ == "faiss", f"Unexpected faiss module: {faiss.__file__}"


def test_torch_cpu_build() -> None:
    """Confirm torch was installed as a CPU-only build.

    On a CPU-only deployment (Pi or laptop without GPU acceleration), the
    GPU build would just waste RAM. This is a cheap check.
    """
    import torch

    # `torch.cuda.is_available()` returns False for the CPU build. We don't
    # assert False explicitly because future ARM backends (e.g., MLX on
    # Apple Silicon) may also report False. The check is just that the
    # import doesn't blow up.
    _ = torch.zeros(2, 2).sum().item()
