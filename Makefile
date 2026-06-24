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
        test test-fast test-cov smoke smoke-e2e smoke-llm smoke-llm-all run \
        run-api run-llm clean clean-pyc clean-venv clean-all verify \
        deps-system deps-verify deps-extras \
        build build-llamacpp llama-dir \
        list-models download-llm download-llm-all download-llm-force \
        verify-llm verify-llm-all setup-laptop \
        sensors-generate sensors-summary


# ---- Help -----------------------------------------------------------------

help:  ## Show this help (default target)
	@echo ""
	@echo "TinyRAG — development commands"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Environment setup:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; \
	         /venv|install|upgrade|freeze/ {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Code quality:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; \
	         /lint|format|typecheck/ {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Testing:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; \
	         /test|smoke/ {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Sensors:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; \
	         /sensors-/ {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Models:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; \
	         /list-models|download|verify-llm|setup-laptop|smoke-llm/ {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Native build:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; \
	         /deps-system|deps-verify|deps-extras|build|llama-dir/ {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Run:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; \
	         /^run/ {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Cleanup:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
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

deps-system:  ## Install apt packages needed to build llama.cpp (idempotent)
	@echo ">> Installing system dependencies (apt)"
	@bash scripts/install_system_deps.sh

deps-verify:  ## Verify system deps are installed (does NOT install)
	@echo ">> Verifying system dependencies"
	@bash scripts/install_system_deps.sh --check

deps-extras:  ## Install optional apt extras (pkg-config, ninja-build)
	@echo ">> Installing optional system extras"
	@bash scripts/install_system_deps.sh --with-extras


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

# Phase 3 end-to-end checkpoint (docs/06_roadmap_v2.md Step 3.9).
# Sends a hard-coded query through the LLMClient (real or fake) and
# asserts a non-empty response. Default client is "real" so a fresh
# laptop clone can use this as the one-shot "is everything wired?"
# check. Pass E2E_CLIENT=fake to run hermetically without a
# llama-server (useful in CI).
smoke-e2e:  ## Phase 3 end-to-end smoke test — sends a query through LLMClient
	@echo ">> Phase 3 end-to-end smoke test"
	@if [ "$(E2E_CLIENT)" = "fake" ]; then \
		echo "   (using FakeLLMClient — no llama-server needed)"; \
		$(PYTHON) scripts/smoke_test.py --client fake; \
	else \
		$(PYTHON) scripts/smoke_test.py --client real --base-url $(SMOKE_BASE_URL); \
	fi

verify: install-dev lint smoke test-fast  ## Full verify: install + lint + smoke + fast tests


# ---- Run ------------------------------------------------------------------

# ---- Build native components (Phase 3) ----
# Step 3.4 — Build llama.cpp. The script is written but the build itself
# is Step 3.4; this target is here so `make build` works end-to-end once
# both steps are complete.
LLAMACPP_DIR := llama.cpp
LLAMACPP_BIN := $(LLAMACPP_DIR)/build/bin/llama-server

llama-dir:  ## Clone llama.cpp source into ./llama.cpp (if not already present)
	@if [ ! -d "$(LLAMACPP_DIR)" ]; then \
		echo ">> Cloning llama.cpp into $(LLAMACPP_DIR)"; \
		git clone https://github.com/ggerganov/llama.cpp.git $(LLAMACPP_DIR); \
	else \
		echo ">> $(LLAMACPP_DIR) already exists — skipping clone"; \
	fi

build-llamacpp: llama-dir deps-system  ## Build llama.cpp with OpenBLAS (laptop)
	@echo ">> Building llama.cpp"
	@bash scripts/build_llamacpp.sh

build: build-llamacpp  ## Build all native components (alias for build-llamacpp in Phase 3)


# ---- Step 3.5: Download GGUF models ---------------------------------------
# We standardise on the model id as the on-disk filename:
# `models/<id>.gguf` — so the path doesn't change when the upstream HF
# filename changes (a real footgun if the path is hardcoded everywhere).
MODELS_DIR ?= models
LLM_MODEL  ?= phi-3-mini
LLM_GGUF   := $(MODELS_DIR)/$(LLM_MODEL).gguf

list-models:  ## List every model in the catalog (no I/O)
	@$(PYTHON) scripts/download_models.py --list

download-llm:  ## Download the primary LLM ($(LLM_MODEL)) to $(MODELS_DIR)/
	@echo ">> Downloading $(LLM_MODEL) to $(MODELS_DIR)/"
	@$(PYTHON) scripts/download_models.py --model $(LLM_MODEL) --models-dir $(MODELS_DIR)

download-llm-all:  ## Download every model in the catalog (large!)
	@echo ">> Downloading all models to $(MODELS_DIR)/"
	@$(PYTHON) scripts/download_models.py --all --models-dir $(MODELS_DIR)

download-llm-force:  ## Force re-download of the primary LLM (ignores cache)
	@$(PYTHON) scripts/download_models.py --model $(LLM_MODEL) --models-dir $(MODELS_DIR) --force

verify-llm:  ## Re-hash on-disk models against the manifest (no network)
	@$(PYTHON) scripts/download_models.py --verify-only --model $(LLM_MODEL) --models-dir $(MODELS_DIR)

verify-llm-all:  ## Re-hash every model on disk (no network)
	@$(PYTHON) scripts/download_models.py --verify-only --all --models-dir $(MODELS_DIR)

# ---- Step 3.7: Smoke test the LLM client against a running llama-server --
# These targets assume llama-server is up (`make run-llm` in another
# terminal). They exercise the OpenAI-compatible /v1/chat/completions
# endpoint end-to-end through our LlamaCppClient and report
# tokens/sec. Run from inside the project root.
SMOKE_BASE_URL ?= http://127.0.0.1:8080

smoke-llm:  ## Smoke-test the primary LLM ($(LLM_MODEL)) — requires llama-server running
	@echo ">> Smoke-testing $(LLM_MODEL) at $(SMOKE_BASE_URL)"
	@$(PYTHON) scripts/smoke_test_llm.py --model $(LLM_MODEL) --base-url $(SMOKE_BASE_URL)

smoke-llm-all:  ## Smoke-test every model on disk — requires llama-server to be restarted per model
	@echo ">> Smoke-testing every model on disk at $(SMOKE_BASE_URL)"
	@echo "   NOTE: llama-server holds one model at a time — restart it between runs."
	@$(PYTHON) scripts/smoke_test_llm.py --all --base-url $(SMOKE_BASE_URL) --models-dir $(MODELS_DIR)

# Convenience: build + download primary in one shot.
setup-laptop: build download-llm  ## Build llama.cpp + download the primary LLM


# ---- Step 3.8: Synthetic sensor data -------------------------------------
# The default output path lives in data/sensor_logs/ which is gitignored
# (see .gitignore). Re-run any time — SEED=42 makes it reproducible.
SENSOR_DATA ?= data/sensor_logs/synthetic_30d.csv

sensors-generate:  ## Generate 30 days of synthetic sensor data to $(SENSOR_DATA)
	@echo ">> Generating 30-day synthetic sensor dataset"
	@$(PYTHON) scripts/generate_synthetic_sensors.py --out $(SENSOR_DATA)

sensors-summary:  ## Summarise the on-disk sensor dataset (no I/O if missing)
	@if [ ! -f $(SENSOR_DATA) ]; then \
		echo "ERROR: $(SENSOR_DATA) not found. Run: make sensors-generate"; \
		exit 1; \
	fi
	@$(PYTHON) scripts/generate_synthetic_sensors.py --out $(SENSOR_DATA) --summary


run-api:  ## Start the FastAPI dev server (Phase 4)
	@echo ">> Starting FastAPI at http://localhost:8000"
	$(VENV)/bin/uvicorn tinyrag.main:app --reload --host 127.0.0.1 --port 8000

run-llm:  ## Start the llama.cpp HTTP server (Phase 3.7+)
	@echo ">> Starting llama-server on :8080"
	@if [ ! -x llama.cpp/build/bin/llama-server ]; then \
		echo "ERROR: llama.cpp not built yet. Run: bash scripts/build_llamacpp.sh"; \
		echo "       (On this laptop the real binary lives at: $$HOME/.cache/llamacpp-build/build/bin/llama-server)"; \
		echo "       (Pre-Step 3.4a builds were in /tmp/llamacpp-build — those may have been wiped on reboot.)"; \
		exit 1; \
	fi
	@if [ ! -f $(LLM_GGUF) ]; then \
		echo "ERROR: $(LLM_GGUF) not found. Run: make download-llm"; \
		exit 1; \
	fi
	@llama.cpp/build/bin/llama-server \
		--model $(LLM_GGUF) \
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
