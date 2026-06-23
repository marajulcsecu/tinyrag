# ============================================================================
# TinyRAG — Makefile
# ----------------------------------------------------------------------------
# Single entry-point for common development tasks. Run `make` (or `make help`)
# to see what's available.
#
# Conventions:
#   - Every target is self-documenting via `make help`.
#   - Every target is idempotent: running it twice = running it once.
#   - Every target exits non-zero on failure so CI can detect breakage.
#   - Targets are grouped: ENV, CODE, TEST, RUN, CLEAN.
#
# Designed for GNU Make 4.x (ships with Ubuntu 24.04 by default).
# ============================================================================


# ---- Configuration --------------------------------------------------------

# The project root path contains colons ("TinyRAG: Retrieval-Augmented..."),
# and Python's venv module refuses to create a venv in a path with ':'.
# So we put the venv in $HOME by default and symlink if the project root is
# colon-free. Override with: make venv VENV=/path/to/venv
VENV      ?= $(HOME)/venvs/tinyrag
PYTHON    := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
RUFF      := $(VENV)/bin/ruff
PYTEST    := $(VENV)/bin/pytest

# Use bash for shell commands so we can rely on POSIX-ish features consistently.
SHELL     := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

# Default target — running just `make` shows help.
.DEFAULT_GOAL := help

# These targets don't create a file with that name; declare them phony.
.PHONY: help venv install install-dev upgrade freeze lint format typecheck \
        test test-fast test-cov smoke run run-api run-llm clean clean-pyc \
        clean-venv clean-all verify


# ---- Help -----------------------------------------------------------------

help:  ## Show this help (default target)
	@echo ""
	@echo "TinyRAG — development commands"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Environment setup:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; \
	         /venv|install|upgrade|freeze/ {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Code quality:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; \
	         /lint|format|typecheck/ {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Testing:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; \
	         /test|smoke|verify/ {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Run:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; \
	         /^run/ {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Cleanup:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; \
	         /^clean/ {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""


# ---- Environment setup ----------------------------------------------------

venv:  ## Create the .venv if it doesn't exist
	@if [ ! -d "$(VENV)" ]; then \
		echo ">> Creating virtualenv at $(VENV)"; \
		python3 -m venv $(VENV); \
		echo ">> Upgrading pip + setuptools + wheel"; \
		$(PIP) install --upgrade pip setuptools wheel; \
	else \
		echo ">> $(VENV) already exists — skipping"; \
	fi

install: venv  ## Install runtime dependencies into .venv
	@echo ">> Installing runtime deps from requirements.txt"
	$(PIP) install -r requirements.txt
	@echo ">> Done. Verify with: make verify"

install-dev: venv  ## Install runtime + dev/test dependencies
	@echo ">> Installing runtime + dev deps"
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	@echo ">> Done. Verify with: make verify"

upgrade:  ## Upgrade all dependencies to their pinned versions
	@echo ">> Re-installing pinned versions (no version bump)"
	$(PIP) install --upgrade --force-reinstall -r requirements.txt
	@$(MAKE) --no-print-directory freeze

freeze:  ## Print the currently installed versions (useful for bug reports)
	@echo ">> Installed versions:"
	@$(PIP) freeze | grep -v "^\-e" | sort


# ---- Code quality ---------------------------------------------------------

lint:  ## Run ruff linter
	@echo ">> Running ruff check"
	$(RUFF) check src tests

format:  ## Auto-format with ruff
	@echo ">> Running ruff format"
	$(RUFF) format src tests
	@echo ">> Running ruff check --fix for import order"
	$(RUFF) check --fix src tests

typecheck:  ## Run mypy (gradual typing — warnings, not errors yet)
	@echo ">> Running mypy"
	$(VENV)/bin/mypy src || true


# ---- Testing --------------------------------------------------------------

test:  ## Run the full test suite
	@echo ">> Running pytest"
	$(PYTEST)

test-fast:  ## Run tests, skipping slow and integration markers
	@echo ">> Running pytest (fast only)"
	$(PYTEST) -m "not slow and not integration"

test-cov:  ## Run tests with coverage report
	@echo ">> Running pytest with coverage"
	$(PYTEST) --cov=src/tinyrag --cov-report=term-missing --cov-report=html

smoke:  ## Quick sanity check: imports + Python version
	@echo ">> Smoke test"
	@$(PYTHON) -c "import sys; assert sys.version_info >= (3, 12), 'Need Python 3.12+'; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
	@$(PYTHON) -c "import fastapi, uvicorn, pydantic, pydantic_settings, yaml; print('Web + config: OK')"
	@$(PYTHON) -c "import faiss, sentence_transformers, numpy, tiktoken; print('Retrieval: OK')"
	@$(PYTHON) -c "import pdfplumber, httpx, sse_starlette, structlog; print('Parsing + HTTP + logging: OK')"
	@$(PYTHON) -c "import paho.mqtt, dateutil, pandas; print('Sensors + data: OK')"
	@echo ">> Smoke test passed"

verify: install-dev lint smoke test-fast  ## Full verify: install + lint + smoke + fast tests


# ---- Run ------------------------------------------------------------------

run-api:  ## Start the FastAPI dev server (Phase 4)
	@echo ">> Starting FastAPI at http://localhost:8000"
	$(VENV)/bin/uvicorn tinyrag.main:app --reload --host 127.0.0.1 --port 8000

run-llm:  ## Start the llama.cpp HTTP server (Phase 3.7+)
	@echo ">> Starting llama-server on :8080"
	@if [ ! -x llama.cpp/build/bin/llama-server ]; then \
		echo "ERROR: llama.cpp not built yet. Run: bash scripts/build_llamacpp.sh"; \
		exit 1; \
	fi
	@llama.cpp/build/bin/llama-server \
		--model models/phi-3-mini-4k-instruct-q4.gguf \
		--host 127.0.0.1 --port 8080 \
		--ctx-size 4096 --threads 10

run:  ## Convenience: run the whole stack (Phase 5+)
	@echo ">> Starting llama.cpp + FastAPI (in two terminals or background)"
	@echo "   Terminal 1: make run-llm"
	@echo "   Terminal 2: make run-api"


# ---- Cleanup --------------------------------------------------------------

clean: clean-pyc  ## Remove caches and build artifacts (keeps .venv)

clean-pyc:  ## Remove __pycache__, .pyc, .ruff_cache, .mypy_cache, .pytest_cache
	@echo ">> Removing Python caches"
	@find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -not -path "./.venv/*" -delete
	@rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov build dist

clean-venv:  ## Remove the .venv entirely
	@echo ">> Removing .venv"
	@rm -rf $(VENV)

clean-all: clean clean-venv  ## Remove EVERYTHING (full reset)
