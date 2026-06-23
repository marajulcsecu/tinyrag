# TinyRAG — Development Roadmap v2 (CANONICAL)

**Project Title:** TinyRAG — A Lightweight, On-Device Retrieval-Augmented Generation Assistant for Smart Home IoT
**Document version:** 2.0 (canonical)
**Date:** 2026-06-23
**Status:** Draft — awaiting student review
**Time budget:** ~10 hours/week
**Total duration:** 10 weeks
**Approach:** **Laptop-first full development, then deploy to real Pi + sensors as the final step**

> **📋 Roadmap history:**
> - `docs/06_roadmap_v1.md` — Pi-primary plan (kept for historical reference, not active).
> - `docs/laptop_fallback/06_roadmap_laptop_v1.md` — Earlier laptop-primary plan (kept for historical reference, not active).
> - **This document (`06_roadmap_v2.md`) is the active, canonical plan.** It reflects the student's explicit decision: build everything on the laptop first, then deploy to Pi + sensors as a final step.

---

## 0. The Big Idea (Read This First)

You will build TinyRAG in **two distinct phases**:

### Phase A — Laptop Build (Weeks 1-8): "Make it work."

You build, test, and demo the **entire** TinyRAG system on your laptop (Dell Inspiron 15 3520, Ubuntu 24.04 LTS). The laptop version is:
- A complete, working RAG system.
- Tested with a 20-question gold set.
- Compared across 3 LLMs.
- Demoed to your advisor at the end of Week 8.

**The laptop version is your project.** If the Pi never arrives, you still have a complete, submittable capstone.

### Phase B — Deployment (Week 9): "Make it real."

You take the **same code** (thanks to clean architecture) and deploy it to a real Raspberry Pi 5 with real DHT22 + PIR sensors. This adds:
- Hardware wiring and GPIO testing.
- The capstone's "wow moment" — a small device answering questions offline.

**The Pi deployment is the bonus.** It demonstrates the architecture's portability.

### Why this order is right

| Reason | Detail |
|--------|--------|
| **Faster iteration** | Laptop is 3-4× faster than Pi for builds and tests |
| **Lower risk** | One machine, one OS, one set of variables — until the very end |
| **Cleaner advisor demos** | "Look, it works fully on my laptop" beats "I have a half-finished Pi port" |
| **Clean architecture pays off** | Same code, different machine = config change only |
| **Graceful degradation** | If Pi or sensors don't arrive, your project is still complete |

---

## 1. How to Read This Document

```
5 phases (3, 4, 5, 6, 7)
   ↓
each phase has 6-22 steps
   ↓
each step has: What AI does / What you do / Review intensity / Done when / Time
```

### 1.1 Review intensity tags

| Tag | Meaning | Your time investment |
|-----|---------|---------------------|
| 🟢 **Light** | Read the code, looks like it does what it says. | 5 min |
| 🟡 **Standard** | Read + run it + verify a test passes + check edge cases mentally. | 15-30 min |
| 🔴 **Deep** | Understand the *why*, run benchmarks, possibly rewrite or push back. Make a judgment call. | 1-2 hours |

### 1.2 Risk gates (the "go/no-go" checkpoints)

Some steps are marked with a 🛑 **RISK GATE**. These are decision points where, if the result is bad, you stop and make a fallback decision before proceeding. **Do not skip risk gates.**

### 1.3 Iteration budget

Each phase has an **"expected iteration budget"** — the number of AI↔you review-fix cycles. Plan for it.

### 1.4 Time estimates

- **AI time** = how long the AI will spend generating.
- **You time** = how long *you* will spend reviewing, running, deciding.

**Your total per week: ~10 hours.** The roadmap respects this.

---

## 2. Big Picture — Phases & Timeline

```
WEEK  1  2  3  4  5  6  7  8  9  10
      │  │  │  │  │  │  │  │  │   │
      ├──┴──┤  │  │  │  │  │  │   │   PHASE 3 — SETUP (laptop)
      │     ├──┴──┴──┤  │  │  │   │   PHASE 4 — BUILD (laptop)
      │     │        ├──┴──┴──┤   │   PHASE 5 — TEST (laptop, gold-set + benchmarks)
      │     │        │        │ ←─┤   ADVISOR DEMO at end of Week 8
      │     │        │        │  ──→│  PHASE 6 — DEPLOY to real Pi + sensors
      │     │        │        │  ──→│  └── 6a: code deploy; 6b: sensor integration
      │     │        │        │  ──→├──┤  PHASE 7 — REPORT & FINAL DEMO
      └─────┴────────┴────────┴─────┘
            CHECKPOINTS (full demo runs at end of each phase)
```

| Phase | Weeks | Steps | Goal | End deliverable |
|-------|-------|-------|------|-----------------|
| **3. Setup** | 1-2 | 9 | Working "Hello World" LLM call on laptop | `make smoke` passes |
| **4. Build** | 3-6 | 22 | Full RAG pipeline on laptop + UI | CLI demo + web UI working |
| **5. Test** | 7-8 | 11 | Tests + 3-model evaluation | Reports + advisor demo |
| **6. Deploy** | 9 | 8 | Real Pi + sensors working | Pi demo + bonus section in report |
| **7. Report & Demo** | 10 | 10 | Final report, slides, demo | Submission + live demo |

**Total: 60 steps, ~85 hours, ~9-10 weeks.**

---

## 3. PHASE 3 — SETUP (Weeks 1-2, on laptop)

**Goal:** A reproducible environment on your laptop, with llama.cpp built, models downloaded, and a "Hello World" LLM call working.

**Why this phase is important:** all later work depends on this. If setup is wrong, nothing else works. We do it carefully.

**Expected iteration budget:** 3-5 review-fix cycles.

**Laptop-specific notes for this phase:**
- All `apt-get` commands assume Ubuntu 24.04 (your OS).
- llama.cpp will be built with **OpenBLAS** for ~2× speedup (your i5-1235U has 10 cores, this matters).
- We do NOT install any GPIO libraries (libgpiod, RPi.GPIO) — those are for Phase 6.

---

### Step 3.1 — Initialize the Git repository

**Phase:** 3
**Goal:** A clean, versioned project on GitHub.

**What AI does:**
- Write `.gitignore` (excludes `models/`, `data/`, `logs/`, `reports/`, `.venv/`, `__pycache__/`, `*.pyc`, `.env`).
- Write `README.md` skeleton (project name, one-paragraph description, "in progress" status, **a "Running on laptop" section since the Pi is not here yet**).
- Write `LICENSE` (MIT).
- Initialize `git init`, make initial commit.
- Create a `main` branch.

**What you do:**
- 🟢 Create a new GitHub repo (public, MIT license). Name it `tinyrag`.
- 🟢 Add the remote: `git remote add origin git@github.com:<your-username>/tinyrag.git`.
- 🟢 `git push -u origin main`.

**Review intensity:** 🟢 Light
**Done when:** repo exists on GitHub with README, LICENSE, and `.gitignore` visible.
**Time:** AI 10 min + you 10 min.

---

### Step 3.2 — Set up Python venv and pinned requirements

**Phase:** 3
**Goal:** Reproducible Python environment.

**What AI does:**
- Generate `requirements.txt` (the pinned list from `docs/05_tech_stack_v1.md` Section 4.1).
- Generate `pyproject.toml` for packaging metadata.
- Write a small `Makefile` with targets: `venv`, `install`, `test`, `lint`, `run`, `smoke`.

**What you do:**
- 🟢 Ubuntu 24.04 ships Python 3.12 by default. Verify: `python3 --version`.
- 🟢 `python3 -m venv .venv` (in the project root).
- 🟢 `source .venv/bin/activate`.
- 🟢 `pip install -r requirements.txt`.
- 🟢 Verify: `python -c "import fastapi, sentence_transformers, faiss; print('OK')"`.

**Review intensity:** 🟢 Light
**Done when:** all imports succeed, no errors.
**Time:** AI 10 min + you 15 min (pip install takes ~5 min on laptop).

---

### Step 3.3 — Install system dependencies for llama.cpp + OpenBLAS

**Phase:** 3
**Goal:** All native libraries in place for a fast llama.cpp build.

**What AI does:**
- Write `scripts/install_system_deps.sh` that runs:
  ```bash
  sudo apt-get update
  sudo apt-get install -y \
      build-essential cmake git \
      libopenblas-dev liblapack-dev \
      sqlite3 tree
  ```
- Document each package's purpose.

**What you do:**
- 🟢 `bash scripts/install_system_deps.sh`.
- 🟢 Verify: `dpkg -l libopenblas-dev liblapack-dev` shows both installed.

**Review intensity:** 🟢 Light
**Done when:** all packages are installed and `pkg-config --libs openblas` returns a path.
**Time:** AI 5 min + you 5 min.

---

### Step 3.4 — Build llama.cpp from source with OpenBLAS

**Phase:** 3
**Goal:** A working `llama-server` binary on your laptop with OpenBLAS acceleration.

**What AI does:**
- Write `scripts/build_llamacpp.sh` with the **laptop-specific** cmake flags:
  ```bash
  cmake -B build \
      -DGGML_OPENBLAS=ON \
      -DGGML_BLAS_VENDOR=OpenBLAS \
      -DCMAKE_BUILD_TYPE=Release
  cmake --build build --config Release -j 10
  ```
- Document the pinned llama.cpp commit hash in `BUILDS.md` once built.
- Add a `make build` target.

**What you do:**
- 🟡 `chmod +x scripts/build_llamacpp.sh`.
- 🟡 Run `bash scripts/build_llamacpp.sh` (or `make build`).
- 🟡 Wait ~5-10 minutes (laptop is fast).
- 🟡 Verify: `ls llama.cpp/build/bin/llama-server` exists, is executable.
- 🟡 Verify OpenBLAS is actually used: `ldd llama.cpp/build/bin/llama-server | grep openblas` (should show libopenblas).

**Review intensity:** 🟡 Standard (OpenBLAS linkage matters)
**Done when:** `llama-server` binary exists, OpenBLAS is linked.
**Time:** AI 15 min + you 15 min.

---

### Step 3.5 — Download Phi-3 Mini GGUF model

**Phase:** 3
**Goal:** The primary LLM file on disk.

**What AI does:**
- Write `scripts/download_models.py` — takes model names as args, downloads from HuggingFace, verifies SHA-256, writes `_manifest.json`.
- Make it idempotent (skips already-downloaded models).
- Set the default URL to the Phi-3 Q4_K_M GGUF on Hugging Face.

**What you do:**
- 🟡 `python scripts/download_models.py phi-3-mini-3.8b-instruct-q4`.
- 🟡 Verify: `ls -lh models/phi-3-mini-3.8b-instruct-q4.gguf` shows ~2.3 GB.
- 🟡 Verify SHA-256 matches the one in the manifest.

**Review intensity:** 🟡 Standard (we're trusting a 2.3 GB download)
**Done when:** file exists, size matches, SHA-256 matches.
**Time:** AI 15 min + you 5-10 min (download time depends on network).

---

### Step 3.6 — 🛑 RISK GATE: First llama.cpp server run on laptop

**Phase:** 3
**Goal:** Confirm llama.cpp can serve the model on your i5-1235U.

**What AI does:**
- Provide the exact `llama-server` invocation command with **laptop-optimized flags**:
  ```bash
  ./llama.cpp/build/bin/llama-server \
      --model models/phi-3-mini-3.8b-instruct-q4.gguf \
      --host 127.0.0.1 --port 8080 \
      --ctx-size 4096 \
      --n-gpu-layers 0 \
      --threads 10 \
      --cont-batching
  ```
- Note: `--threads 10` matches your physical core count (2P + 8E).
- Provide a `curl` command to test the `/v1/chat/completions` endpoint.

**What you do:**
- 🔴 Start the server in one terminal (use background process or new terminal).
- 🔴 In another terminal, run a test curl:
  ```bash
  curl -s http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Say hello in 5 words."}], "max_tokens": 50, "temperature": 0}' | jq
  ```
- 🔴 Verify: you get a JSON response with a "hello"-like message.
- 🔴 **Measure:** first-token latency (should be ~0.8-1.5 s on your i5-1235U with OpenBLAS).
- 🔴 **Measure:** peak RAM with `ps aux | grep llama-server` (should be ~1.8 GB).

**Review intensity:** 🔴 Deep — this is the foundation; if it doesn't work, nothing else does.
**Done when:** a JSON response comes back with a sensible answer in < 2 s for the first token.
**Time:** you 15-20 min.

**🛑 Decision point:**
- ✅ Works, < 2s first token → move to Step 3.7.
- ❌ Server crashes → check `logs/llamacpp.log` (AI helps you debug).
- ❌ Response is gibberish → check model file integrity (re-download).
- ❌ Very slow (>3s for first token) → reduce `--ctx-size` to 2048; try `--threads 8`.
- ❌ RAM > 3 GB → reduce `--ctx-size` to 2048.

---

### Step 3.7 — Download comparison models (TinyLlama, Llama 3.2 3B)

**Phase:** 3
**Goal:** All 3 LLMs ready for later evaluation.

**What AI does:**
- Extend `scripts/download_models.py` with the other 2 model URLs (TinyLlama 1.1B Q4, Llama 3.2 3B Q4).

**What you do:**
- 🟡 `python scripts/download_models.py tinyllama-1.1b-chat-v1.0-q4 llama-3.2-3b-instruct-q4`.
- 🟡 Verify all 3 GGUF files are present and sizes match.
- 🟡 (Optional) Quick smoke test: load each one in turn, send 1 query, confirm response.

**Review intensity:** 🟡 Standard
**Done when:** all 3 model files exist with correct sizes.
**Time:** AI 5 min + you 10-15 min (depends on network).

**Storage check:** 3 models × ~2 GB avg = ~5 GB. You have 512 GB SSD — no problem.

---

### Step 3.8 — Generate synthetic sensor data

**Phase:** 3
**Goal:** 30 days of fake but realistic sensor data.

**What AI does:**
- Write `scripts/generate_synthetic_sensors.py` — produces `data/sensor_logs/synthetic_30d.csv` with realistic patterns (daily temperature cycles, weekend energy spikes, etc.).
- Use `pandas` + `numpy.random` with a fixed seed (`SEED=42`) for reproducibility.
- Generate data for: `living_room_temp`, `living_room_hum`, `bedroom_temp`, `bedroom_hum`, `kitchen_motion`, `house_energy`.

**What you do:**
- 🟡 Run it: `python scripts/generate_synthetic_sensors.py`.
- 🟡 Open the CSV in any text editor / spreadsheet; verify the data looks sensible (no NaN, realistic values, ~17,000 rows for 30 days × 4 sensors at 5-min resolution).

**Review intensity:** 🟡 Standard
**Done when:** CSV exists with ~17,000 rows and realistic patterns.
**Time:** AI 15 min + you 5 min.

---

### Step 3.9 — Phase 3 checkpoint: end-to-end smoke test

**Phase:** 3
**Goal:** Confirm everything is wired and reproducible.

**What AI does:**
- Write `scripts/smoke_test.py` — a single script that:
  1. Loads config.
  2. Starts llama.cpp (or assumes it's running).
  3. Sends a hard-coded query.
  4. Prints the response.
  5. Asserts the response is non-empty.
- Add `make smoke` target to the Makefile.

**What you do:**
- 🟡 Run `make smoke`.
- 🟡 Verify: you see a sensible answer to "What is 2+2?" in < 3 seconds.
- 🟡 Commit: `git add . && git commit -m "Phase 3 complete: working setup + smoke test"`.
- 🟡 Push to GitHub.

**Review intensity:** 🟡 Standard
**Done when:** `make smoke` exits 0 with a coherent answer.
**Time:** AI 15 min + you 10 min.

**🛑 Phase 3 exit gate:**
- ✅ All 9 steps done.
- ✅ `make smoke` passes on the laptop.
- ✅ Repo is on GitHub with at least 3 commits.
- → **Move to Phase 4 (Build).**

---

## 4. PHASE 4 — BUILD (Weeks 3-6, on laptop)

**Goal:** A working end-to-end RAG pipeline on your laptop, accessible via a web UI.

**Why this is the longest phase:** we're building 7+ modules from scratch. Each is small, but the cumulative work is significant.

**Expected iteration budget:** 15-25 review-fix cycles. **This is normal.** Don't panic if a step takes 2-3 review iterations.

**Build order is strict:** each step depends on the previous. Don't skip ahead.

---

### Step 4.1 — Initialize the project skeleton (folders only)

**Phase:** 4
**Goal:** Empty folders with `__init__.py` files, ready for code.

**What AI does:**
- Create the full `src/tinyrag/` directory tree from `docs/03_architecture_v1.md` Section 5.
- Each module folder gets an empty `__init__.py` with a one-line docstring.
- Create `tests/` directory with `conftest.py` and an empty `test_smoke.py`.

**What you do:**
- 🟢 Verify the structure with `tree src/ -L 3`.
- 🟢 Commit: `git add . && git commit -m "Initial project skeleton"`.

**Review intensity:** 🟢 Light
**Done when:** directory structure matches the architecture doc.
**Time:** AI 5 min + you 5 min.

---

### Step 4.2 — Set up `config.yaml` and `Settings` loader

**Phase:** 4
**Goal:** Type-safe config loading.

**What AI does:**
- Write `config.yaml` with `deployment.target: laptop` (so we get the laptop defaults).
- Write `src/tinyrag/config.py` with a Pydantic `Settings` model and a `load_settings()` function.
- Add a test: `tests/test_config.py` that loads a sample config and asserts all fields are populated.

**What you do:**
- 🟡 Read the config.yaml — sanity check the values.
- 🟡 Run `pytest tests/test_config.py -v` — should pass.
- 🟡 Try: `python -c "from tinyrag.config import load_settings; s = load_settings(); print(s.llm.model_path)"` — should print the model path.

**Review intensity:** 🟡 Standard (config errors break everything downstream)
**Done when:** `load_settings()` works, all FR-49 to FR-52 are satisfied.
**Time:** AI 20 min + you 15 min.

---

### Step 4.3 — Implement structured logging

**Phase:** 4
**Goal:** Replace any `print()` with proper JSON logging.

**What AI does:**
- Write `src/tinyrag/observability/logger.py` — a `get_logger(name)` factory using `structlog`.
- Configure for both stdout (pretty) and file (JSON).
- Add a test that captures log output and asserts JSON format.

**What you do:**
- 🟢 Run `pytest tests/test_logger.py -v`.
- 🟢 Quick demo: in a Python REPL, `from tinyrag.observability.logger import get_logger; log = get_logger("test"); log.info("hello", key="value")`.

**Review intensity:** 🟢 Light
**Done when:** logs are JSON in the file, pretty on stdout.
**Time:** AI 15 min + you 5 min.

---

### Step 4.4 — Implement the document parsers (PDF, TXT, MD)

**Phase:** 4
**Goal:** `parsers.py` can extract clean text from any of the 3 formats.

**What AI does:**
- Write `src/tinyrag/ingestion/parsers.py` with:
  - `parse_pdf(path) -> ParsedDocument` (uses pdfplumber, preserves page numbers)
  - `parse_txt(path) -> ParsedDocument`
  - `parse_md(path) -> ParsedDocument`
  - `parse(path) -> ParsedDocument` (dispatcher by extension)
- Define `ParsedDocument` dataclass: `text: str`, `pages: list[tuple[int, str]]` (for PDFs), `metadata: dict`.
- Write `tests/test_parsers.py` with 3 fixtures: a tiny PDF, a TXT, a MD.

**What you do:**
- 🟡 Run `pytest tests/test_parsers.py -v`.
- 🟡 Manually test: `python -c "from tinyrag.ingestion.parsers import parse; print(parse('tests/fixtures/sample.pdf').text[:200])"`.

**Review intensity:** 🟡 Standard (PDF parsing is finicky; edge cases matter)
**Done when:** all 3 parsers work, page numbers preserved for PDFs.
**Time:** AI 30 min + you 15 min.

---

### Step 4.5 — Implement the chunker

**Phase:** 4
**Goal:** Token-based chunking with overlap.

**What AI does:**
- Write `src/tinyrag/core/chunker.py` with:
  - `Chunker.chunk(text: str, source: str, page: int | None) -> list[Chunk]`
  - Uses `tiktoken` to count tokens (`cl100k_base`).
  - Splits at 400 tokens with 50-token overlap.
  - Respects sentence boundaries when possible (find the nearest `.` before the cutoff).
- Define `Chunk` dataclass: `text: str`, `source: str`, `page: int | None`, `chunk_index: int`, `char_offset: int`, `token_count: int`.
- Write `tests/test_chunker.py` with 5+ test cases (short text, long text, exact boundary, empty, etc.).

**What you do:**
- 🟡 Run `pytest tests/test_chunker.py -v`.
- 🟡 Spot-check: feed a 2000-token text, verify you get ~5 chunks with overlap.

**Review intensity:** 🟡 Standard
**Done when:** all tests pass, chunks look sensible on a real document.
**Time:** AI 25 min + you 15 min.

---

### Step 4.6 — Implement the embedder (Protocol + concrete)

**Phase:** 4
**Goal:** A working `SentenceTransformerEmbedder` that loads the embedding model once and embeds batches.

**What AI does:**
- Write `src/tinyrag/ingestion/embedder.py` with:
  - `EmbeddingModel` Protocol (from architecture doc).
  - `SentenceTransformerEmbedder` class implementing it.
  - Loads model lazily on first use.
  - `.embed(texts: list[str]) -> list[list[float]]`.
  - `.dimension` property.
- Write `tests/test_embedder.py`: test with 2-3 short texts, verify output is a list of 384-dim vectors.

**What you do:**
- 🟡 Run `pytest tests/test_embedder.py -v`.
- 🟡 Confirm dimension matches the one in `config.yaml` (384 for all-MiniLM-L6-v2).

**Review intensity:** 🟡 Standard
**Done when:** embedding a list of texts returns 384-dim vectors, deterministic across runs.
**Time:** AI 20 min + you 10 min.

---

### Step 4.7 — Implement the metadata store (SQLite wrapper)

**Phase:** 4
**Goal:** `MetadataStore` class with the schema from the DB design doc.

**What AI does:**
- Write `src/tinyrag/storage/metadata.py` with:
  - `MetadataStore(db_path: str)` class.
  - `init_schema()` (idempotent CREATE TABLE IF NOT EXISTS ...).
  - `insert_document(...)`, `insert_chunks(...)`, `get_chunks_by_ids(...)`, `list_documents()`, `delete_document(...)`, `log_query(...)`.
  - Uses parameterized queries (no SQL injection).
- Write `tests/test_metadata.py` with 6+ tests.

**What you do:**
- 🟡 Run `pytest tests/test_metadata.py -v`.
- 🟡 Open the SQLite DB in DB Browser for SQLite and visually confirm the schema.

**Review intensity:** 🟡 Standard
**Done when:** all tests pass, schema matches `docs/04_database_design_v1.md` Section 5.2.
**Time:** AI 30 min + you 15 min.

---

### Step 4.8 — Implement the FAISS vector store wrapper

**Phase:** 4
**Goal:** `FAISSStore` class implementing the `VectorStore` Protocol.

**What AI does:**
- Write `src/tinyrag/storage/vector_store.py` with:
  - `VectorStore` Protocol.
  - `FAISSStore` class: `add(vectors, ids)`, `search(query_vector, k) -> list[(int_idx, score)]`, `delete_by_source(...)`, `save()`, `load()`, `size()`.
  - Use `IndexFlatIP` (inner product on L2-normalized vectors = cosine sim).
  - Maintain an int↔UUID mapping (saved to a sidecar JSON).
- Write `tests/test_vector_store.py`: test add, search, delete, save/load round-trip.

**What you do:**
- 🟡 Run `pytest tests/test_vector_store.py -v`.
- 🟡 Verify: after save+load, the index returns the same search results.

**Review intensity:** 🟡 Standard
**Done when:** all Protocol methods work, search results are sensible.
**Time:** AI 35 min + you 15 min.

---

### Step 4.9 — 🛑 RISK GATE: end-to-end ingestion pipeline

**Phase:** 4
**Goal:** Confirm a real PDF can be ingested end-to-end (parse → chunk → embed → store).

**What AI does:**
- Write `scripts/ingest.py` — CLI: `python scripts/ingest.py <file>`.
  - Calls `parse()`, `chunk()`, `embedder.embed()`, `vector_store.add()`, `metadata.insert_*()`.
  - Prints an `IngestionReport` at the end.
- Provide a script to **download a real device manual** (e.g., Nest thermostat manual from the manufacturer's site) to `tests/fixtures/`.

**What you do:**
- 🔴 Run `python scripts/ingest.py tests/fixtures/nest_thermostat_manual.pdf`.
- 🔴 Verify the report: `num_chunks > 50`, `time_ms < 30_000` (on laptop, should be fast).
- 🔴 Open the SQLite DB — confirm the document and chunks are there.
- 🔴 Open the FAISS index — confirm the size matches.
- 🔴 Manually query: pick a chunk text from the SQLite, embed a similar query, verify it's retrieved at top-1.

**Review intensity:** 🔴 Deep — this is the foundation of the RAG pipeline.
**Done when:** a real PDF is parsed, chunked, embedded, stored, and retrievable.
**Time:** AI 30 min + you 30 min.

**🛑 Decision point:**
- ✅ Works → continue to Step 4.10.
- ❌ Embedding is slow (>0.5 sec per chunk) → check if model is on CPU; reduce batch size.
- ❌ PDF parsing extracts gibberish → try a different PDF first; pdfplumber may need config tweaks.
- ❌ FAISS returns wrong results → verify normalization is applied correctly.

---

### Step 4.10 — Implement the LLM client (llama.cpp wrapper)

**Phase:** 4
**Goal:** A streaming `LlamaCppClient` that talks to llama-server's HTTP API.

**What AI does:**
- Write `src/tinyrag/generation/llm_client.py` with:
  - `LLMClient` Protocol.
  - `LlamaCppClient` class.
  - `generate(prompt, max_tokens, temperature) -> Iterator[str]` — uses `httpx.Client.stream()` for SSE.
  - `model_name()`, `is_healthy()`.
- Write `tests/test_llm_client.py` with mocked HTTP responses (or use `vcr.py`).

**What you do:**
- 🟡 Start llama-server manually (in background).
- 🟡 Run `pytest tests/test_llm_client.py -v`.
- 🟡 Manually: `python -c "from tinyrag.generation.llm_client import LlamaCppClient; c = LlamaCppClient(...); print(''.join(c.generate('Say hi')))"`.

**Review intensity:** 🟡 Standard
**Done when:** a real streaming call to llama.cpp returns a sensible answer.
**Time:** AI 30 min + you 15 min.

---

### Step 4.11 — Implement the prompt builder

**Phase:** 4
**Goal:** A function that constructs a grounded prompt from system instructions + retrieved chunks + query.

**What AI does:**
- Write `src/tinyrag/core/prompt_builder.py` with:
  - `PromptBuilder.build(query: str, chunks: list[Chunk]) -> Prompt`.
  - System prompt (well-engineered for grounded answering + citation).
  - Context block: numbered chunks `[1] ... [2] ... [3] ...`.
  - User question.
  - Total length must fit in the LLM's context window (4096 tokens).
- Write `tests/test_prompt_builder.py` with 4+ cases (no chunks, 1 chunk, max chunks, very long chunks).

**What you do:**
- 🟢 Run `pytest tests/test_prompt_builder.py -v`.
- 🟢 Manually inspect a generated prompt — does it look right?

**Review intensity:** 🟡 Standard (prompt quality directly affects answer quality)
**Done when:** generated prompts look well-structured, fit in context.
**Time:** AI 25 min + you 10 min.

---

### Step 4.12 — Implement the retriever

**Phase:** 4
**Goal:** `Retriever` class that embeds a query, searches both indices, merges results, filters by threshold.

**What AI does:**
- Write `src/tinyrag/core/retriever.py` with:
  - `Retriever(embedder, doc_store, sensor_store, metadata)`.
  - `retrieve(query: str, k_doc: int, k_sensor: int, threshold: float) -> RetrievalResult`.
  - Detects sensor keywords (simple list: "temperature", "humidity", "energy", "kWh", "yesterday", "last week", etc.) to decide whether to also search the sensor index.
  - Filters chunks by similarity threshold.
  - Returns a `RetrievalResult` dataclass: `chunks: list[Chunk]`, `scores: list[float]`, `used_sensor_idx: bool`.
- Write `tests/test_retriever.py` with mocked stores (fast unit tests).

**What you do:**
- 🟡 Run `pytest tests/test_retriever.py -v`.
- 🟡 Integration test: ask "How do I reset my thermostat?" → should retrieve from the doc index, top score > 0.4.

**Review intensity:** 🟡 Standard
**Done when:** retrieval returns sensible top-k chunks, threshold filtering works.
**Time:** AI 35 min + you 20 min.

---

### Step 4.13 — Implement the sensor source (simulated only on laptop)

**Phase:** 4
**Goal:** The `SensorSource` Protocol + `SimulatedCSVSource` working.

**What AI does:**
- Write `src/tinyrag/sensors/base.py` with the Protocol.
- Write `src/tinyrag/sensors/simulated.py` with `SimulatedCSVSource.read(since=None) -> pd.DataFrame`.
- Write `tests/test_sensors.py` with 3+ cases.

**What you do:**
- 🟢 Run `pytest tests/test_sensors.py -v`.
- 🟢 Spot-check: `python -c "from tinyrag.sensors.simulated import SimulatedCSVSource; src = SimulatedCSVSource('data/sensor_logs/synthetic_30d.csv'); df = src.read(); print(df.head()); print(f'Total: {len(df)} rows')"`.

**Review intensity:** 🟢 Light
**Done when:** `SimulatedCSVSource.read()` returns a valid DataFrame.
**Time:** AI 20 min + you 10 min.

**Laptop-specific note:** we skip `RealSerialSource` (DHT22 + PIR over GPIO) for now — not applicable on a laptop. The file `serial_dht.py` is still written (as a stub) so the architecture is complete and ready for Phase 6.

---

### Step 4.14 — Implement the sensor summarizer

**Phase:** 4
**Goal:** Convert raw sensor DataFrame into text-summary chunks for the vector store.

**What AI does:**
- Write `src/tinyrag/core/sensor_summarizer.py` with:
  - `SensorSummarizer.summarize(df: pd.DataFrame) -> list[Chunk]`.
  - Default mode: per-day, per-sensor-type summaries (avg, min, max, peak time).
  - Special handling for `motion` (event-based, not stats).
- Write `tests/test_sensor_summarizer.py` with 3+ cases.

**What you do:**
- 🟡 Run `pytest tests/test_sensor_summarizer.py -v`.
- 🟡 Manually: feed the synthetic 30-day data, inspect the generated text summaries.

**Review intensity:** 🟡 Standard
**Done when:** summaries are human-readable and capture the right info.
**Time:** AI 30 min + you 15 min.

---

### Step 4.15 — Wire sensor summarization into the ingestion pipeline

**Phase:** 4
**Goal:** A separate `scripts/ingest_sensors.py` that ingests sensor data into the sensor vector store.

**What AI does:**
- Write `scripts/ingest_sensors.py` — reads CSV, summarizes, embeds, adds to sensor index, logs in metadata DB.
- Extend `metadata.py` to also handle `doc_type='sensor_summary'`.

**What you do:**
- 🟡 `python scripts/ingest_sensors.py`.
- 🟡 Verify: sensor chunks appear in the SQLite DB; sensor FAISS index has ~30 entries (one per day per sensor type).

**Review intensity:** 🟡 Standard
**Done when:** sensor data is queryable.
**Time:** AI 20 min + you 10 min.

---

### Step 4.16 — 🛑 RISK GATE: end-to-end RAG via CLI

**Phase:** 4
**Goal:** A working CLI: `python scripts/ask.py "How do I reset my thermostat?"` returns a cited answer.

**What AI does:**
- Write `scripts/ask.py` — orchestrates: embed query → retrieve → build prompt → stream LLM → print tokens → print citations.
- Write a small `Answer` dataclass in `core/answer.py`.

**What you do:**
- 🔴 `python scripts/ask.py "How do I reset my Nest thermostat to factory settings?"`.
- 🔴 Verify: you get a coherent, cited answer that mentions "reset" or "factory" from the manual.
- 🔴 Try 5 different queries (mix of doc + sensor questions).
- 🔴 Measure: end-to-end latency should be < 3 s on the laptop (vs < 5 s on Pi 5).

**Review intensity:** 🔴 Deep — this is the entire RAG pipeline working.
**Done when:** a manual question returns a correct, cited answer in < 3s.
**Time:** AI 30 min + you 30 min.

**🛑 Decision point:**
- ✅ Works → move to UI step.
- ❌ Answer is wrong → check retrieval (is the right chunk retrieved? if not, fix embedder or chunker).
- ❌ Answer is correct but no citation → fix prompt builder to enforce citations.
- ❌ Latency > 5s on laptop → check where time is spent (add timing logs); reduce ctx-size to 2048.

---

### Step 4.17 — Implement the FastAPI app skeleton + `/api/status`

**Phase:** 4
**Goal:** A running FastAPI server with a working status endpoint.

**What AI does:**
- Write `src/tinyrag/main.py` with the FastAPI app factory and lifespan management.
- Write `src/tinyrag/api/routes_query.py` with `GET /api/status`.
- Write `src/tinyrag/api/routes_docs.py` (skeleton, will fill in 4.18).
- Write `src/tinyrag/api/routes_admin.py` (skeleton).

**What you do:**
- 🟡 Start the server: `uvicorn tinyrag.main:app --host 127.0.0.1 --port 8000`.
- 🟡 Open `http://127.0.0.1:8000/api/status` in browser — should return JSON with model, chunk count, RAM.

**Review intensity:** 🟡 Standard
**Done when:** the server starts, /api/status returns valid JSON.
**Time:** AI 30 min + you 15 min.

---

### Step 4.18 — Implement document management endpoints

**Phase:** 4
**Goal:** POST /api/documents (upload), GET /api/documents (list), DELETE /api/documents/{id} (delete).

**What AI does:**
- Fill in `routes_docs.py` with the 3 endpoints.
- Wire them to `IngestionPipeline` and `MetadataStore`.
- Add validation: file size ≤ 50MB, extension whitelist, filename sanitization.

**What you do:**
- 🟡 Use `curl` or Postman to upload a PDF.
- 🟡 List documents — verify it appears.
- 🟡 Delete it — verify it disappears from list AND from vector store.
- 🟡 Restart the server — verify the data persists.

**Review intensity:** 🟡 Standard
**Done when:** all 3 endpoints work, persistence verified.
**Time:** AI 30 min + you 20 min.

---

### Step 4.19 — Implement the query endpoint with SSE streaming

**Phase:** 4
**Goal:** POST /api/query streams answer tokens back to the client.

**What AI does:**
- Implement `POST /api/query` using SSE (`sse-starlette`).
- The route calls the same RAG pipeline as `ask.py` but streams.
- Returns Server-Sent Events: `data: {"token": "Hello"}\n\n`, etc.

**What you do:**
- 🟡 Test with `curl -N` (no-buffer): `curl -N -X POST http://127.0.0.1:8000/api/query -H "Content-Type: application/json" -d '{"query":"What is 2+2?"}'`.
- 🟡 Verify: tokens appear one by one, then a `data: [DONE]` event at the end.

**Review intensity:** 🟡 Standard
**Done when:** SSE stream works, end-to-end latency matches CLI.
**Time:** AI 30 min + you 15 min.

---

### Step 4.20 — 🛑 Portability self-test (preparing for Phase 6)

**Phase:** 4
**Goal:** Verify the code can be set up from scratch on a different machine — the foundation for Phase 6's Pi deployment.

**What AI does:**
- Write `scripts/portability_check.sh` that:
  1. Clones the repo to `/tmp/tinyrag-portability-test/`.
  2. Runs `bash setup.sh` from scratch in that directory.
  3. Runs `make smoke` and asserts the response is non-empty.
  4. Cleans up `/tmp/tinyrag-portability-test/`.

**What you do:**
- 🔴 `bash scripts/portability_check.sh`.
- 🔴 Verify: setup.sh runs cleanly on the clone, smoke test passes.
- 🔴 If anything fails: fix it NOW. The Pi deployment in Phase 6 will have the same failure modes.

**Review intensity:** 🔴 Deep — this is your "works on a fresh machine" proof.
**Done when:** the portability check passes end-to-end.
**Time:** AI 15 min + you 30-45 min (setup.sh takes ~20-30 min to run on the laptop).

**Why this step exists:** "Works on my machine" is the #1 source of deployment bugs. This step catches them on the laptop (where you have time to fix) rather than on the Pi (where you're racing the deadline).

---

### Step 4.21 — Build the web UI: chat page

**Phase:** 4
**Goal:** A working chat interface at `http://127.0.0.1:8000/`.

**What AI does:**
- Write `ui/templates/index.html` (Jinja2) — chat box, send button, message history.
- Write `ui/static/chat.js` — uses `fetch()` with SSE reader, appends tokens to the message bubble.
- Write `ui/static/style.css` — clean, simple styling.
- Wire FastAPI to serve `ui/` static files and render `index.html` at `/`.

**What you do:**
- 🟡 Open `http://127.0.0.1:8000/` in browser.
- 🟡 Type a question, hit send, watch the answer stream in.
- 🟡 Try 3 different questions; verify sources appear as cards below the answer.

**Review intensity:** 🟡 Standard (UI bugs are common, but easy to fix)
**Done when:** chat UI works end-to-end, sources render correctly.
**Time:** AI 60 min + you 30 min.

---

### Step 4.22 — Build the web UI: admin / documents page

**Phase:** 4
**Goal:** A working document management UI at `http://127.0.0.1:8000/admin`.

**What AI does:**
- Write `ui/templates/admin.html` — upload form, list of documents, delete buttons.
- Write `ui/static/admin.js` — handles form submit, list refresh, delete confirm.
- Add `GET /admin` route in FastAPI that renders the template.

**What you do:**
- 🟡 Open the admin page, upload a PDF, see it in the list, delete it.
- 🟡 Try uploading a non-PDF → should fail with a clear error.

**Review intensity:** 🟢 Light
**Done when:** upload, list, delete all work in the UI.
**Time:** AI 45 min + you 20 min.

---

### Step 4.23 — Implement the system status panel

**Phase:** 4
**Goal:** A live status panel in the UI showing model, RAM, vector store size, sensor source.

**What AI does:**
- Write `ui/static/status.js` — polls `/api/status` every 5 seconds, updates DOM.
- Add a small status card to `index.html`.

**What you do:**
- 🟢 Verify the panel updates: upload a doc, watch "num chunks" increase; ask a question, watch RAM tick up briefly.

**Review intensity:** 🟢 Light
**Done when:** panel updates live and shows accurate values.
**Time:** AI 20 min + you 10 min.

---

### Step 4.24 — Write `run.sh` and `stop.sh` (the one-command start)

**Phase:** 4
**Goal:** `./run.sh` brings up the entire system from cold; `./stop.sh` tears it down.

**What AI does:**
- Write `run.sh` — starts llama-server in background, waits for health, starts uvicorn, traps signals to clean up.
- Write `stop.sh` — kills both processes.

**What you do:**
- 🟡 `./stop.sh` (or kill processes), then `./run.sh`.
- 🟡 Verify: UI loads, query works.
- 🟡 `Ctrl-C` the script — both processes die cleanly.

**Review intensity:** 🟡 Standard (process management is finicky)
**Done when:** `./run.sh` brings up everything, `./stop.sh` tears it down.
**Time:** AI 20 min + you 15 min.

---

### Step 4.25 — 🛑 PHASE 4 CHECKPOINT: full demo on laptop

**Phase:** 4
**Goal:** A complete end-to-end demo on your laptop.

**What AI does:**
- Write `docs/demo_script_laptop.md` — a 5-step demo script with sample questions.

**What you do:**
- 🔴 Cold start: `./run.sh`.
- 🔴 Demo 1: upload a PDF via UI, see it indexed.
- 🔴 Demo 2: ask a manual question, get cited answer.
- 🔴 Demo 3: ask a sensor question, get cited answer.
- 🔴 Demo 4: show the status panel updating live.
- 🔴 Demo 5: kill the process, restart, verify data persists.
- 🔴 Record the screen as a backup video (use OBS Studio, or `ffmpeg -f x11grab` on Linux).

**Review intensity:** 🔴 Deep — this is the laptop demo working end-to-end.
**Done when:** all 5 demo steps succeed without manual intervention.
**Time:** you 1-2 hours.

**🛑 Phase 4 exit gate:**
- ✅ All 25 steps done.
- ✅ Full demo works on laptop.
- ✅ Portability self-test passes.
- → **Move to Phase 5 (Test).**

---

## 5. PHASE 5 — TEST (Weeks 7-8, on laptop)

**Goal:** Comprehensive test coverage, a 20-question gold set, and a 3-model evaluation report.

**Expected iteration budget:** 8-12 review-fix cycles.

**Laptop-specific note:** all tests run on the laptop. The Pi is reserved for Phase 6 (deployment) — we don't redo evaluation on the Pi in this phase.

---

### Step 5.1 — Achieve ≥ 60% test coverage

**Phase:** 5
**Goal:** Cover all core modules with unit tests.

**What AI does:**
- Run `pytest --cov=tinyrag --cov-report=term-missing` to see current coverage.
- Generate additional tests for under-covered modules.
- Add edge-case tests: empty input, very long input, malformed input, concurrent access.

**What you do:**
- 🟢 `pytest --cov` — verify ≥ 60% line coverage.
- 🟢 Read the coverage report; identify any obviously-untested critical paths.

**Review intensity:** 🟡 Standard
**Done when:** coverage ≥ 60% on `core/`, `ingestion/`, `generation/`, `storage/`.
**Time:** AI 45 min + you 20 min.

---

### Step 5.2 — Set up CI on GitHub Actions

**Phase:** 5
**Goal:** Every push automatically runs tests + lint.

**What AI does:**
- Write `.github/workflows/ci.yml` — runs `pytest` and `ruff check` on every push/PR.
- Add a status badge to the README.

**What you do:**
- 🟢 Push to GitHub, verify the CI runs and passes.
- 🟢 (Optional) Add a "branch protection rule" requiring CI to pass before merge.

**Review intensity:** 🟢 Light
**Done when:** CI runs on GitHub and shows green ✅.
**Time:** AI 15 min + you 10 min.

---

### Step 5.3 — Write the 20-question gold set

**Phase:** 5
**Goal:** A test set with known correct answers, saved as JSON.

**What AI does:**
- Write `data/evaluation/gold_set.json` with 20 questions, structured as:
  ```json
  [
    {
      "id": "Q01",
      "query": "How do I reset my Nest thermostat to factory settings?",
      "expected_keywords": ["reset", "factory", "settings"],
      "expected_source_type": "manual",
      "expected_source_hint": "nest_thermostat_manual",
      "category": "manual_lookup"
    },
    ...
  ]
  ```
- The 20 questions follow the distribution from `docs/02_srs_v1.md` Appendix A: ~10 manual/FAQ, ~10 sensor, with mixed difficulty.
- Save to `data/evaluation/gold_set.json` (in git, for reproducibility).

**What you do:**
- 🟡 Read all 20 questions.
- 🟡 Adjust `expected_keywords` if the AI's draft doesn't match what the docs actually say.
- 🟡 Verify the questions are answerable from your ingested docs.

**Review intensity:** 🟡 Standard (gold set quality is critical for evaluation)
**Done when:** 20 questions, all answerable from ingested docs, keywords correct.
**Time:** AI 30 min + you 30 min.

---

### Step 5.4 — Write the evaluation script

**Phase:** 5
**Goal:** `scripts/eval.py --model <name>` runs all 20 questions and produces a CSV.

**What AI does:**
- Write `scripts/eval.py`:
  1. Load the gold set.
  2. For each model in the config: switch llama.cpp to that model, run all 20 questions, record (query, model, answer, latency, top1_score, keyword_match).
  3. Output `reports/eval_<model>_<date>.csv`.
- Define a simple scoring function: % of `expected_keywords` present in the answer.

**What you do:**
- 🟡 `python scripts/eval.py --model phi-3-mini-3.8b-instruct-q4` — verify it runs.
- 🟡 Inspect the CSV — do the answers look right? Are the keyword matches sensible?

**Review intensity:** 🟡 Standard
**Done when:** eval script runs and produces a sensible CSV.
**Time:** AI 45 min + you 20 min.

---

### Step 5.5 — Write the manual-judgment scoring rubric

**Phase:** 5
**Goal:** A rubric for human-judging the answers.

**What AI does:**
- Write `docs/evaluation/scoring_rubric.md`:
  - ✅ **Correct & cited** (3 points): answer is factually right, includes citation, no hallucination.
  - ⚠️ **Partially correct** (2 points): answer is roughly right but missing detail or has a minor hallucination.
  - ❌ **Wrong** (0 points): answer is factually wrong or refuses to answer.
- Provide example judgments for 3-5 sample answers.

**What you do:**
- 🟢 Read the rubric.
- 🟢 Adjust scoring if needed (e.g., add a "1 point" tier).

**Review intensity:** 🟢 Light
**Done when:** rubric is clear and you can apply it consistently.
**Time:** AI 15 min + you 10 min.

---

### Step 5.6 — Run evaluation on all 3 models (laptop)

**Phase:** 5
**Goal:** Generate the 3-model comparison results.

**What AI does:**
- Provide commands to run `eval.py` for each of: TinyLlama, Llama 3.2 3B, Phi-3 Mini.
- Provide a script to merge the 3 CSVs into a comparison table.

**What you do:**
- 🔴 Run the eval for each of the 3 models (~5-10 min per model on laptop).
- 🔴 Manually judge each answer using the rubric (~30 min per model = 1.5 hours total).
- 🔴 Generate the comparison table: model × accuracy, avg latency, peak RAM.

**Review intensity:** 🔴 Deep — this is the most important data for your report.
**Done when:** 3-model comparison table is complete and saved to `reports/`.
**Time:** AI 15 min + you 3-4 hours (most of this is human judgment).

---

### Step 5.7 — Write the benchmark script (latency + RAM)

**Phase:** 5
**Goal:** Automated latency and RAM measurement.

**What AI does:**
- Write `scripts/benchmark.py`:
  - For each model, run a fixed set of 5 queries.
  - Measure: first-token latency, end-to-end latency, peak RSS (RAM), model load time.
  - Use `psutil` for RAM, `time.perf_counter()` for timing.
  - Output `reports/benchmark_<date>.csv`.

**What you do:**
- 🟡 Run `python scripts/benchmark.py`.
- 🟡 Sanity-check the numbers: are they within the NFR budgets?

**Review intensity:** 🟡 Standard
**Done when:** benchmark runs cleanly, numbers are sensible.
**Time:** AI 30 min + you 15 min.

---

### Step 5.8 — Write the RAG vs no-RAG comparison

**Phase:** 5
**Goal:** Show that RAG actually helps (vs. just asking the LLM).

**What AI does:**
- Write `scripts/compare_rag_vs_norag.py`:
  - For each gold-set question, run two pipelines:
    - **With RAG:** retrieve chunks, build prompt, generate.
    - **Without RAG:** just send the question directly to the LLM.
  - Score both with the same rubric.
  - Output a side-by-side comparison.

**What you do:**
- 🟡 Run the comparison for the primary model (Phi-3).
- 🟡 Verify: RAG answers are clearly better (or at least, citations help).

**Review intensity:** 🟡 Standard
**Done when:** comparison report is generated.
**Time:** AI 30 min + you 30 min.

---

### Step 5.9 — Generate plots/visualizations for the report

**Phase:** 5
**Goal:** Charts ready to embed in the final report.

**What AI does:**
- Write `scripts/generate_plots.py`:
  - Bar chart: model × accuracy.
  - Bar chart: model × avg latency.
  - Bar chart: model × peak RAM.
  - RAG vs no-RAG accuracy comparison.
- Use `matplotlib` (install via `pip install matplotlib` if not already).

**What you do:**
- 🟢 Run the script.
- 🟢 Inspect the PNGs — do they look professional?

**Review intensity:** 🟢 Light
**Done when:** 4 plots are saved to `reports/figures/`.
**Time:** AI 20 min + you 5 min.

---

### Step 5.10 — Implement the "offline mode" proof (for the demo)

**Phase:** 5
**Goal:** A way to demonstrate the system runs without network.

**What AI does:**
- Write `scripts/prove_offline.sh`:
  1. Print current network state (`ip route`, `curl ifconfig.me`).
  2. Block all outbound traffic using `iptables`: `sudo iptables -A OUTPUT -j DROP`.
  3. Verify: try `curl ifconfig.me` → should fail.
  4. Run the demo: ask a question, get a cited answer.
  5. Restore: `sudo iptables -F`.

**What you do:**
- 🔴 `bash scripts/prove_offline.sh` (requires sudo).
- 🔴 Verify: the demo works while outbound traffic is blocked.
- 🔴 Verify: `iptables -F` restores normal network.

**Review intensity:** 🟡 Standard (iptables can lock you out if you mess up — be careful)
**Done when:** the script demonstrates offline operation cleanly.
**Time:** AI 10 min + you 20 min.

---

### Step 5.11 — 🛑 PHASE 5 CHECKPOINT: advisor demo + evaluation complete

**Phase:** 5
**Goal:** All evaluation artifacts in place; demo to advisor.

**What you do:**
- 🔴 `reports/` contains:
  - `eval_<model>_*.csv` × 3 models
  - `benchmark_*.csv`
  - `rag_vs_norag_*.csv`
  - `figures/*.png` × 4 plots
- 🔴 **Schedule a meeting with your advisor** to demo the laptop version.
- 🔴 Run the full demo live, in front of them, on the laptop.
- 🔴 Show the comparison table, the plots, the offline-mode proof.
- 🔴 Get their feedback (write it down — this will inform the final report).

**Review intensity:** 🔴 Deep — this is your first formal demonstration.
**Done when:** advisor sees the working system, gives feedback, and approves the direction.
**Time:** you 2-3 hours (including the meeting).

**🛑 Phase 5 exit gate:**
- ✅ 3-model evaluation done.
- ✅ RAG vs no-RAG done.
- ✅ Latency + RAM benchmarks done.
- ✅ Advisor demo done.
- ✅ Advisor feedback captured.
- → **Move to Phase 6 (Deploy to real Pi + sensors).**

---

## 6. PHASE 6 — DEPLOY TO REAL PI + SENSORS (Week 9)

**Goal:** Take the working laptop build, deploy it to a real Raspberry Pi 5, and wire up real DHT22 + PIR sensors. Add this as the capstone's "wow factor" section.

**Why this phase is short (1 week):** the clean architecture means deploying is mostly a config change. The hard work (the RAG pipeline) is already done in Phases 3-5.

**Time-box rule:** Phase 6 has a hard 1-week limit. If sensors aren't working by Day 4, document the simulated path as the primary and use a recorded Pi video for the report. **Do not let Pi issues delay Phase 7.**

**Expected iteration budget:** 5-8 review-fix cycles.

**This phase is two sub-phases:**

```
Phase 6a — Code Deployment (Days 1-3): Get the same code running on the Pi
Phase 6b — Sensor Integration (Days 3-5): Wire DHT22 + PIR, verify real data flows
```

---

### Step 6.1 — 🛑 RISK GATE: verify Pi is actually available

**Phase:** 6
**Goal:** Confirm the Raspberry Pi 5 and sensors are physically in your hands and working.

**What you do:**
- 🔴 **Check 1:** Do you have the Raspberry Pi 5 (8 GB recommended)?
- 🔴 **Check 2:** Do you have a microSD card (≥ 16 GB) flashed with Raspberry Pi OS 64-bit?
- 🔴 **Check 3:** Do you have a DHT22 sensor (or whatever your lab provides)?
- 🔴 **Check 4:** Do you have a PIR motion sensor?
- 🔴 **Check 5:** Do you have jumper wires, a breadboard, and a way to power the Pi?

**Review intensity:** 🔴 Deep (physical reality check)
**Done when:** all 5 checks pass.

**🛑 Critical decision point:**
- ✅ **All 5 checks pass** → continue Phase 6 normally.
- ⚠️ **Pi is available but sensors are not** → do Phase 6a (code deploy) only. Document sensors as "future work."
- ❌ **Pi is not available** → **skip Phase 6 entirely.** Proceed to Phase 7. The laptop build is your project. Document the Pi deployment as "deferred to future work" in the report.
- ❌ **Pi is available but you have < 5 days until the deadline** → do a minimal Pi smoke test (just run the system, don't integrate sensors) and proceed to Phase 7.

**Time:** you 30 min (just the check).

---

### Step 6.2 — Set up the Raspberry Pi 5 (base OS + SSH)

**Phase:** 6a (Code Deployment)
**Goal:** A working Pi 5 with Raspberry Pi OS 64-bit, reachable via SSH.

**What AI does:**
- Generate `docs/pi_setup_checklist.md` with the exact Raspberry Pi Imager settings.
- Provide the `apt-get` command for Pi-specific deps.

**What you do:**
- 🔴 Physically set up the Pi: insert microSD, connect power, connect HDMI monitor + keyboard for first boot (or use Pi Imager's "advanced options" to pre-configure SSH).
- 🔴 SSH in from your laptop: `ssh pi@<pi-hostname>.local`.
- 🔴 Run `sudo apt update && sudo apt full-upgrade -y`.
- 🔴 Set a static IP on your router (optional, but makes the demo URL predictable).

**Review intensity:** 🔴 Deep (physical setup)
**Done when:** you can SSH into the Pi, `uname -a` shows `aarch64`.
**Time:** you 1-2 hours.

---

### Step 6.3 — Clone the repo and run `setup.sh` on the Pi

**Phase:** 6a
**Goal:** The same `setup.sh` works on Pi (auto-detection was designed for this).

**What AI does:**
- Verify `setup.sh` correctly auto-detects aarch64 and uses Pi flags (`-mcpu=cortex-a76 -mfpu=neon-fp-armv8`, no OpenBLAS).
- Provide a troubleshooting section for common Pi build errors.

**What you do:**
- 🟡 `git clone git@github.com:<you>/tinyrag.git` on the Pi.
- 🟡 `cd tinyrag && bash setup.sh`.
- 🟡 Wait ~1.5-2 hours (llama.cpp compile is slow on Pi).
- 🟡 Verify: `ls llama.cpp/build/bin/llama-server` exists.

**Review intensity:** 🟡 Standard
**Done when:** `setup.sh` completes without error on Pi.
**Time:** AI 10 min + you 2-3 hours (mostly waiting).

---

### Step 6.4 — Sync data from laptop to Pi

**Phase:** 6a
**Goal:** The Pi has the same ingested documents and vector store as your laptop.

**What AI does:**
- Write `scripts/sync_to_pi.sh` that uses `rsync` to copy `data/` (excluding logs) to the Pi.

**What you do:**
- 🟡 `bash scripts/sync_to_pi.sh`.
- 🟡 Verify on the Pi: `./run.sh` and the same demo questions work.

**Review intensity:** 🟡 Standard
**Done when:** Pi serves the same content as the laptop.
**Time:** AI 10 min + you 15 min.

---

### Step 6.5 — 🛑 RISK GATE: first LLM call on Pi

**Phase:** 6a
**Goal:** Confirm the Pi can run the primary LLM within latency budget.

**What AI does:**
- Provide the exact `llama-server` command with Pi flags (`-mcpu=cortex-a76`, `--threads 4`).

**What you do:**
- 🔴 Start llama-server on Pi.
- 🔴 Curl a simple query.
- 🔴 Measure latency — is it < 5s for a 200-token answer?

**Review intensity:** 🔴 Deep — this is the make-or-break moment for the Pi.
**Done when:** Pi serves a 200-token answer in < 7 seconds.
**Time:** you 30 min.

**🛑 Decision point (CRITICAL):**
- ✅ < 5s → continue with Phi-3 as primary.
- ⚠️ 5-7s → continue but consider switching to Llama 3.2 3B as primary.
- ❌ > 7s → **switch primary to TinyLlama 1.1B** and document the trade-off in the report.
- ❌ OOM crash → reduce ctx-size to 2048, switch to TinyLlama.

---

### Step 6.6 — Wire the DHT22 temperature/humidity sensor

**Phase:** 6b (Sensor Integration)
**Goal:** Read real temperature and humidity from a DHT22 connected to the Pi's GPIO.

**What AI does:**
- Activate the stub `src/tinyrag/sensors/serial_dht.py` — implement `RealSerialSource` using `adafruit-circuitpython-dht` and `libgpiod`.
- Provide the wiring diagram (DHT22 data pin → GPIO 4, VCC → 3.3V, GND → GND).
- Provide a 10KΩ pull-up resistor note (required for DHT22).
- Write a test: `tests/test_sensors_real.py` that reads 5 readings and asserts they're in valid ranges.

**What you do:**
- 🔴 **Wire the DHT22** to the Pi's GPIO pins (refer to the diagram AI provides).
- 🔴 `pip install adafruit-circuitpython-dht` on the Pi.
- 🔴 `python -c "from tinyrag.sensors.serial_dht import RealSerialSource; s = RealSerialSource(); print(s.read().head())"`.
- 🔴 Verify: you see real temperature/humidity values, not zeros or errors.

**Review intensity:** 🔴 Deep — this is hardware + software together.
**Done when:** the Pi reads real DHT22 values into the DataFrame.
**Time:** AI 20 min + you 1-2 hours (wiring + debugging is unpredictable).

---

### Step 6.7 — Wire the PIR motion sensor (if available)

**Phase:** 6b
**Goal:** Detect real motion events via the PIR sensor.

**What AI does:**
- Extend `RealSerialSource` to also read the PIR pin.
- Provide the wiring diagram (PIR OUT → GPIO 17, VCC → 5V, GND → GND).
- Note: PIR may need a 5V supply; check the sensor's spec.

**What you do:**
- 🔴 **Wire the PIR** to the Pi.
- 🔴 Wave your hand in front of it; check that motion is detected.
- 🔴 Update `config.yaml`: `sensors.source: real_serial`.
- 🔴 Restart TinyRAG, ask a motion question, verify it answers from real data.

**Review intensity:** 🔴 Deep
**Done when:** motion events are detected and retrievable.
**Time:** AI 15 min + you 1 hour.

---

### Step 6.8 — 🛑 PHASE 6 CHECKPOINT: full Pi demo with real sensors

**Phase:** 6
**Goal:** The capstone's "wow demo" — Pi with real sensors, fully offline.

**What you do:**
- 🔴 `./run.sh` on the Pi.
- 🔴 Demo on Pi: ask a manual question, ask a sensor question (using real DHT22 + PIR data), show the status panel.
- 🔴 **Record the entire demo as a video** (use `ffmpeg` or screen recorder).
- 🔴 (Optional) Set up systemd for auto-start on boot.

**Review intensity:** 🔴 Deep — this is the money demo for the report.
**Done when:** the full demo works on Pi with real sensors.
**Time:** you 1-2 hours.

**🛑 Phase 6 exit gate:**
- ✅ Pi serves the full system (with or without sensors).
- ✅ Demo video recorded.
- → **Move to Phase 7 (Report & Demo).**

---

## 7. PHASE 7 — REPORT & FINAL DEMO (Week 10)

**Goal:** A professional capstone report, presentation slides, and a polished demo.

**Expected iteration budget:** 5-8 review-fix cycles (mostly for writing).

---

### Step 7.1 — Outline the final report

**Phase:** 7
**Goal:** A section-by-section outline.

**What AI does:**
- Generate `reports/final_report_outline.md` with the 9 standard sections (Abstract, Intro, Related Work, System Design, Implementation, Evaluation, Discussion, Conclusion, References).
- For each section, provide 1-2 paragraphs of what to include.

**What you do:**
- 🟢 Review the outline.
- 🟢 Adjust based on your advisor's requirements (if any).

**Review intensity:** 🟢 Light
**Done when:** outline is approved.
**Time:** AI 20 min + you 30 min.

---

### Step 7.2 — Write the Abstract + Introduction

**Phase:** 7
**Goal:** First 2 pages of the report.

**What AI does:**
- Generate a draft Abstract (200-300 words) and Introduction (1-2 pages) based on the scope, SRS, and architecture docs.

**What you do:**
- 🟡 Read the draft — is it accurate? Does it sound like your voice?
- 🟡 Edit for clarity, your writing style, advisor's preferences.

**Review intensity:** 🟡 Standard (this is the first thing the panel reads)
**Done when:** both sections read well and are factually correct.
**Time:** AI 30 min + you 1-2 hours.

---

### Step 7.3 — Write the Related Work section

**Phase:** 7
**Goal:** 1-page comparison to existing tools.

**What AI does:**
- Write a brief comparison: TinyRAG vs. Ollama, PrivateGPT, GPT4All, LocalAI.
- Highlight: what TinyRAG does that's different/better (edge-first, pluggable, Raspberry Pi benchmarked).

**What you do:**
- 🟡 Verify the claims about other tools are accurate.
- 🟡 Add 1-2 sentences of your own perspective.

**Review intensity:** 🟡 Standard
**Done when:** 1 page, accurate, well-written.
**Time:** AI 30 min + you 30 min.

---

### Step 7.4 — Write the System Design + Implementation sections

**Phase:** 7
**Goal:** The technical core of the report (~5-8 pages).

**What AI does:**
- Generate prose from the architecture doc.
- Include the system diagram, the module diagram, the data flow diagrams.
- Include 2-3 key code snippets (with explanation) — pick the most interesting ones (e.g., the prompt builder, the LLM client).
- **Bonus section:** if you deployed to the Pi in Phase 6, include a "Raspberry Pi Deployment" subsection with the wiring diagram and Pi benchmarks.

**What you do:**
- 🟡 Read the draft.
- 🟡 Add 1-2 sentences about *why* you made each design decision.
- 🟡 Make sure the figures are properly captioned and referenced.

**Review intensity:** 🔴 Deep
**Done when:** the section is accurate, well-illustrated, and reads as a coherent narrative.
**Time:** AI 60 min + you 2-3 hours.

---

### Step 7.5 — Write the Evaluation section

**Phase:** 7
**Goal:** The data-heavy section (~3-4 pages).

**What AI does:**
- Generate prose around the evaluation tables and plots.
- Insert the comparison table (3 models × accuracy, latency, RAM).
- Insert the plots from `reports/figures/`.
- Add a discussion of surprising results.
- **Bonus:** if Pi data exists, add a laptop vs Pi latency comparison.

**What you do:**
- 🔴 Read carefully — make sure the numbers match the CSVs exactly.
- 🔴 Add your interpretation: why did the smaller model win/lose? What does the latency-RAM trade-off look like?

**Review intensity:** 🔴 Deep
**Done when:** the section is data-accurate, well-narrated, with plots.
**Time:** AI 45 min + you 1-2 hours.

---

### Step 7.6 — Write the Discussion + Conclusion

**Phase:** 7
**Goal:** Last 2 pages of main text.

**What AI does:**
- Write the Discussion (limitations, what didn't work, what you'd do differently).
- Write the Conclusion + Future Work (multi-language, voice, real smart-home APIs, mobile — all already designed-in).
- Include a "Deployment Experience" subsection reflecting on the Pi deployment (if you did it).

**What you do:**
- 🟡 Add 1-2 honest sentences about what you learned.
- 🟡 Be specific about limitations (don't just say "future work").

**Review intensity:** 🟡 Standard
**Done when:** both sections are honest, specific, and forward-looking.
**Time:** AI 30 min + you 1 hour.

---

### Step 7.7 — Format and polish the report

**Phase:** 7
**Goal:** A clean PDF ready to submit.

**What AI does:**
- Provide a LaTeX or Markdown template that produces a professional-looking PDF.
- Generate the References section (BibTeX or Markdown).
- Add a table of contents, list of figures, list of tables.

**What you do:**
- 🔴 Compile to PDF.
- 🔴 Read the whole thing cover-to-cover.
- 🔴 Fix typos, broken cross-references, missing captions.

**Review intensity:** 🔴 Deep
**Done when:** you have a polished PDF.
**Time:** AI 30 min + you 2-3 hours.

---

### Step 7.8 — Polish the README with screenshots and badges

**Phase:** 7
**Goal:** A GitHub-ready README that impresses anyone who visits the repo.

**What AI does:**
- Add sections: features, architecture diagram, screenshots (placeholders — you provide actual screenshots), quickstart, evaluation table, deployment section (laptop + Pi), license, citation.
- Add a CI badge, a "made with" badge, a license badge.

**What you do:**
- 🟡 Take 3-4 screenshots of the running UI (chat, admin, status panel, citation cards).
- 🟡 If you deployed to Pi: add a photo of the wired-up Pi + sensors.
- 🟡 Insert them into the README.
- 🟡 Add a "Results" section with the comparison table from Step 5.6.

**Review intensity:** 🟢 Light (looks matter for the repo)
**Done when:** README is professional and informative.
**Time:** AI 15 min + you 30 min.

---

### Step 7.9 — Build the presentation slides

**Phase:** 7
**Goal:** 10-15 slides for the defense.

**What AI does:**
- Generate a slide outline (1 slide per major section).
- Provide content for each slide (key points, not full sentences).
- Suggest images/figures to embed.

**What you do:**
- 🟡 Build the slides in your tool of choice (PowerPoint, Google Slides, LaTeX Beamer).
- 🟡 Rehearse the 10-15 min presentation at least twice.
- 🟡 Prepare answers for likely questions.

**Review intensity:** 🟡 Standard
**Done when:** slides are ready and you've rehearsed.
**Time:** AI 30 min + you 2-3 hours.

---

### Step 7.10 — 🛑 FINAL CHECKPOINT: submission

**Phase:** 7
**Goal:** Everything submitted.

**What you do:**
- 🔴 Push final code to GitHub.
- 🔴 Submit the report PDF.
- 🔴 Submit any required forms (deliverable list, abstract, etc.).
- 🔴 Deliver the live demo (on the laptop, with the offline-mode proof; or on the Pi if it worked).

**Review intensity:** 🔴 Deep
**Done when:** 🎉 **Capstone complete.**

---

## 8. Risk Gates Summary

For easy reference, all 🛑 risk gates in one place:

| Step | Risk | Fallback decision |
|------|------|-------------------|
| **3.6** | First llama.cpp run fails on laptop | Debug; check OpenBLAS linkage; reduce ctx-size |
| **4.9** | End-to-end ingestion fails | Fix parser/embedder/store individually; try different PDF |
| **4.16** | First end-to-end RAG fails | Check retrieval → check prompt → check LLM |
| **4.20** | Portability self-test fails | Fix "works on my machine" issues NOW, on the laptop |
| **4.25** | Phase 4 checkpoint fails | Iterate on failing step; don't proceed |
| **5.11** | Phase 5 checkpoint fails | More time on testing; do not skip evaluation |
| **6.1** | **Pi or sensors unavailable** | **Skip Phase 6 entirely. Proceed to Phase 7.** The laptop is your project. |
| **6.5** | Pi too slow for Phi-3 | Switch primary to Llama 3.2 3B or TinyLlama |
| **6.6/6.7** | Sensor wiring fails | Document the issue; fall back to simulated sensor data on the Pi |
| **6.8** | Pi demo not ready | Use the recorded video; laptop demo is still the primary |

---

## 9. Time Budget Summary

| Phase | Steps | AI time | You time | Total |
|-------|-------|---------|----------|-------|
| 3 — Setup | 9 | ~3 hours | ~6 hours | ~9 hours |
| 4 — Build | 25 | ~10 hours | ~25 hours | ~35 hours |
| 5 — Test | 11 | ~5 hours | ~12 hours | ~17 hours |
| 6 — Deploy | 8 | ~1 hour | ~12 hours | ~13 hours |
| 7 — Report | 10 | ~5 hours | ~12 hours | ~17 hours |
| **Total (with Pi)** | **63** | **~24 hours** | **~67 hours** | **~91 hours** |
| **Total (Pi skipped)** | **55** | **~23 hours** | **~55 hours** | **~78 hours** |

At 10 hours/week:
- **With Pi:** ~91 / 10 = **~9.5 weeks** ✅
- **Without Pi:** ~78 / 10 = **~8 weeks** ✅ (lots of buffer)

---

## 10. What Comes After This

After you approve this roadmap:
- I write the `evaluation/gold_set.md` and `evaluation/scoring_rubric.md` files (referenced in Phase 5).
- I update the `AGENT.md` to reflect "Phase 2 complete — ready to execute Phase 3."
- Then we **start Step 3.1 together** — the first real coding step (the `.gitignore`, README skeleton, LICENSE, initial commit).

**The roadmap is the contract. The plan is set. Now we execute.**

---

## 11. Document Approval

| Role | Name | Approval | Date |
|------|------|----------|------|
| Student | Marajul Haque | ⏳ pending | |
| Advisor | Abu Nowshed Chy | (not required for v1) | |

---

*End of Canonical Roadmap v2.*
