# TinyRAG

> A lightweight, fully on-device Retrieval-Augmented Generation (RAG) assistant for smart-home IoT — running on a Raspberry Pi 5, with **zero cloud calls**.

![CI](https://github.com/marajulcsecu/tinyrag/actions/workflows/ci.yml/badge.svg)
![Status](https://img.shields.io/badge/status-in%20development-yellow)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%205%20%7C%20Linux-lightgrey)

---

## What is TinyRAG?

TinyRAG is a **privacy-preserving document assistant** that runs entirely on a small edge device. It reads any documents you give it — PDF, plain text, or Markdown — and answers natural-language questions about them using a **local** small language model (Llama 3.2 3B Instruct, Q4-quantized, served via llama.cpp). It can also summarise IoT sensor data (temperature, humidity, energy, motion) when a sensor source is configured, which is where the smart-home framing comes from; the retrieval pipeline itself is general-purpose.

**The entire pipeline runs offline.** Turn off your Wi-Fi — it still works.

### Example queries

- *"How do I reset my Nest thermostat to factory settings?"* → answers from the Nest manual.
- *"What was the average temperature in the living room this week?"* → answers from sensor logs.
- *"Why is my energy bill higher than usual this month?"* → cross-references sensor data + home FAQ.
- *"What is the meaning of life?"* → correctly refuses with the fallback message.

---

## Current Status (August 2026)

**Phase 4 (Build) is complete and verified end-to-end; Phase 5 (evaluation) is in progress and Phase 6 (Pi deployment) is now unblocked.** The system runs on a Dell Inspiron 15 3520 (Ubuntu 24.04 LTS, i5-1235U, 8 GB RAM) with a chat UI, REST API, document upload, and **1409 passing tests**. A pre-demo verification pass found and fixed six substantive bugs (see [`docs/VERIFICATION_ROADMAP.md`](docs/VERIFICATION_ROADMAP.md)), and the lab has since provided a **Raspberry Pi 5 (8 GB)**, so the deployment phase is greenlit.

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 3 | Setup (repo, venv, llama.cpp build) | Done |
| Phase 4 | Build (ingestion, retrieval, generation, sensors, API) | **Done** — verified end-to-end |
| Phase 5 | Test (gold set + benchmarks, 3-model comparison) | **In progress** — 22-question gold set + eval corpus done (Step 5.3); eval harness next |
| Phase 6 | Deploy (Raspberry Pi 5) | **Code deploy done** — verified answering on a Pi 5 (8 GB) 2026-08-04; sensor wiring deferred pending advisor guidance |
| Phase 7 | Report (capstone report + final demo) | Planned |

See [`docs/06_roadmap_v2.md`](docs/06_roadmap_v2.md) for the full 60-step plan.

---

## Architecture at a Glance

TinyRAG follows a **clean, modular, protocol-oriented** architecture. Every external dependency (LLM, embedding model, vector store, sensor source, UI input) is hidden behind a Python `Protocol` interface, so any component can be swapped without rewriting the rest of the system.

```
                  ┌──────────────┐
                  │   Web UI     │  (HTML + vanilla JS)
                  └──────┬───────┘
                         │ HTTP
                  ┌──────▼───────┐
                  │   FastAPI    │  (async API + auto-docs)
                  └──────┬───────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   ┌────▼─────┐    ┌─────▼─────┐    ┌─────▼─────┐
   │Retrieval │    │Generation │    │  Sensors   │
   │(FAISS)   │    │(llama.cpp)│    │(CSV/Serial)│
   └────┬─────┘    └─────┬─────┘    └───────────┘
        │                │
   ┌────▼─────┐    ┌─────▼─────┐
   │Embedding │    │  SQLite   │
   │ (MiniLM) │    │(metadata) │
   └──────────┘    └───────────┘
```

Full breakdown: [`docs/03_architecture_v1.md`](docs/03_architecture_v1.md) (C4 model + module boundaries + Protocol interfaces).

---

## Quick Demo (3 commands)

If the system is already installed and built:

```bash
bash run.sh                          # starts llama-server + uvicorn (≈30 s warmup)
# open http://127.0.0.1:8000/ in your browser → chat UI
# open http://127.0.0.1:8000/admin    → document management UI
# open http://127.0.0.1:8000/api/status → JSON health endpoint
```

**Verified demo questions** (in order):
1. *"What is RAG?"* — answers from `rag.txt` (the primer we ingested).
2. *"What is the ErP directive?"* — answers from Nest install guide p.26 (keyword rerank promotes the definition over the TOC).
3. *"What is OpenTherm?"* — answers from p.4 + p.24 (compatibility + wiring).
4. *"Is the Nest compatible with combi boilers?"* — answers from p.20-21.
5. *"What is the capital of France?"* — correctly refuses with *"I don't have enough information in the provided documents."*

Full demo script + 6 more questions with expected citations and talking points: [`docs/DEMO_QUESTIONS.md`](docs/DEMO_QUESTIONS.md).
Deep-dive architecture explainer for teacher demo: [`docs/EXPLANATION.md`](docs/EXPLANATION.md).

---

## Running on Laptop or Raspberry Pi 5

> **The Raspberry Pi 5 is now available** (8 GB model, 64 GB microSD, provided by the lab). The same `setup.sh` + `run.sh` bring the stack up on either machine: the code, model, and `config.yaml` are shared, and the thread count is detected from the CPU core count at launch (4 on the Pi, ~12 on the laptop). Set `deployment.target` to `raspberry_pi` in `config.yaml` when running on the Pi.

### Prerequisites

- **OS:** Ubuntu 24.04 LTS, or Raspberry Pi OS (64-bit) on the Pi
- **Python:** 3.12 (Ubuntu 24.04 ships this; the Pi needs a 3.12+ image)
- **RAM:** 8 GB minimum
- **Disk:** ~10 GB free (for llama.cpp build + the GGUF model + 30 days of sensor data)
- **Build tools:** `build-essential`, `cmake`, `git` (installed in Step 3.3)
- **BLAS:** `libopenblas-dev`, `liblapack-dev` (installed in Step 3.3)

### One-command install

```bash
git clone https://github.com/marajulcsecu/tinyrag.git
cd tinyrag
bash setup.sh        # installs Python deps + builds llama.cpp + downloads the model (~20 min)
bash run.sh          # starts FastAPI on http://127.0.0.1:8000 + llama-server on :8080
```

`setup.sh` is idempotent — re-run after pulling new commits to pick up new dependencies.
`run.sh` is also idempotent — calling it while the stack is already running is a no-op (it just reports the PIDs).

### Test the install

```bash
PYTHONPATH=src pytest tests/ -q   # 1409 passed, 3 skipped (~17 min)
curl http://127.0.0.1:8000/api/status | head -c 300
```

### Notes for the Raspberry Pi 5

**Verified working on a Pi 5 (8 GB) as of 2026-08-04** — retrieval,
generation, and citation, fully offline. Full walkthrough and field notes:
[`docs/PI_DEPLOYMENT.md`](docs/PI_DEPLOYMENT.md).

The Pi runs the same two commands as the laptop — there is no Pi-specific
branch. What differs in practice:

- **⚠️ Python 3.12 must be installed first.** Current 64-bit Raspberry Pi
  OS (Debian 13 trixie) ships **Python 3.13**, and the pinned `torch==2.4.1`
  / `numpy==1.26.4` have no 3.13 wheels — `pip` would try to compile torch
  from source on 4 ARM cores. `apt` has no 3.12 on any Debian channel, so
  install a prebuilt standalone CPython 3.12 and create the venv from it
  before running `setup.sh`. Exact commands in
  [`docs/PI_DEPLOYMENT.md`](docs/PI_DEPLOYMENT.md) §2.
- **Reaching the UI from another machine.** Both services bind
  `127.0.0.1`, so on a headless Pi the browser on your laptop cannot see
  them. The safe option is an SSH tunnel, which needs no config change:
  ```bash
  ssh -L 8000:127.0.0.1:8000 <user>@<pi-ip>   # then open http://127.0.0.1:8000/
  ```
  Alternatively `API_HOST=0.0.0.0 bash run.sh` binds the API to all
  interfaces — convenient on a trusted lab network, but the UI then has
  **no authentication**, so do not do this on an open network. Leave
  `LLM_HOST` at `127.0.0.1` either way.
- **First start is slow.** llama-server reads the full 1.9 GB model before
  answering its health check; from microSD that is tens of seconds.
  `run.sh` allows 180 s (`HEALTH_TIMEOUT`), which is ample — raise it only
  if you see exit code 14.
- **Threads are detected, not configured.** The llama.cpp thread count
  defaults to the CPU core count (4 on the Pi). Do not raise it above the
  core count: inference is CPU-bound, so extra threads only add context
  switching and make generation slower.
- **The corpus is not in the clone.** `data/` is gitignored, so a fresh Pi
  starts with an empty index — rebuild it per
  [`data/evaluation/corpus/README.md`](data/evaluation/corpus/README.md).

**Measured on the Pi:** a 1730-token prompt takes **78 s** end to end
(70 s prompt eval at 24.6 tok/s, 8 s generation at 3.6 tok/s), using
**487 MB of RAM**. That is roughly 3–5× slower than the laptop's 15–25 s,
and prompt processing is ~90% of it.

---

## Project Structure

```
tinyrag/
├── AGENT.md                  ← context handoff file (read first)
├── README.md                 ← this file
├── LICENSE                   ← MIT
├── .gitignore                ← see file for full list
│
├── docs/                     ← all planning docs (read in order)
│   ├── 00_high_level_plan.md     ← whole-journey visualization
│   ├── 01_project_scope_v2.md    ← refined scope (canonical)
│   ├── 02_srs_v1.md              ← requirements (58 FRs, 37 NFRs)
│   ├── 03_architecture_v1.md     ← C4 model + Protocols
│   ├── 04_database_design_v1.md  ← FAISS + SQLite + CSV
│   ├── 05_tech_stack_v1.md       ← pinned versions
│   ├── 06_roadmap_v2.md          ← 60-step plan (canonical)
│   ├── EXPLANATION.md            ← teacher-demo architecture deep-dive
│   ├── DEMO_QUESTIONS.md         ← teacher-demo verified Q&A run-sheet
│   └── evaluation/
│       ├── gold_set.md           ← 20 evaluation questions
│       └── scoring_rubric.md     ← human-judgment rubric
│
├── src/tinyrag/              ← source code (written in Phase 4)
├── tests/                    ← pytest unit tests
├── scripts/                  ← operational scripts (ingest, evaluate, benchmark)
├── models/                   ← downloaded GGUF models (gitignored)
├── data/                     ← runtime data (gitignored)
└── reports/                  ← generated benchmarks + reports (gitignored)
```

---

## Documentation Index

**If you are new to this project, read in this order:**

1. [`AGENT.md`](AGENT.md) — project context, decisions, status
2. [`docs/00_high_level_plan.md`](docs/00_high_level_plan.md) — journey map
3. [`docs/01_project_scope_v2.md`](docs/01_project_scope_v2.md) — what we're building
4. [`docs/03_architecture_v1.md`](docs/03_architecture_v1.md) — how it's built
5. [`docs/06_roadmap_v2.md`](docs/06_roadmap_v2.md) — when & how

**For the teacher demo / capstone presentation:**
- [`docs/EXPLANATION.md`](docs/EXPLANATION.md) — architecture deep-dive with diagrams, engineering-decision rationale, anticipated Q&A.
- [`docs/DEMO_QUESTIONS.md`](docs/DEMO_QUESTIONS.md) — verified demo questions with expected citations + FAQ cheat-sheet.
- [`docs/PI_DEPLOYMENT.md`](docs/PI_DEPLOYMENT.md) — Raspberry Pi 5 deployment walkthrough, measured performance, and field notes.

**For evaluators / advisors:** scope + SRS + EXPLANATION are enough.
**For new developers:** AGENT.md + architecture + roadmap.
**For contributors:** see [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Tech Stack (Pinned)

| Component | Choice | Why |
|-----------|--------|-----|
| LLM (primary) | **Llama 3.2 3B Instruct, Q4_K_M GGUF** | Coherent at RAG prompt sizes (2500–3600 tok); replaced Phi-3 Mini, which emitted garbage above ~2048 tokens |
| LLM (compare) | TinyLlama 1.1B, Phi-3 Mini 3.8B | Required for 3+ model eval |
| LLM server | llama.cpp HTTP server | Mature, well-documented, CPU-only |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (384-d) | Small, fast, good quality |
| Vector store | FAISS `IndexFlatIP` | Simple, CPU-friendly, cosine via normalized inner product |
| Metadata DB | SQLite 3 (WAL mode) | Embedded, zero-config |
| API | FastAPI 0.115 + Uvicorn | Async, auto-docs |
| PDF parsing | **PyMuPDF 1.24.10** (`get_text("blocks", sort=True)`) | Column-aware reading order; replaced pdfplumber, which interleaved two-column pages across the gutter |
| Token counting | tiktoken | Industry standard |
| Logging | structlog | Structured JSON logs |
| Tests | pytest | Standard |
| Lint | ruff | Fast, opinionated |

Full pinning: [`docs/05_tech_stack_v1.md`](docs/05_tech_stack_v1.md).

---

## License

MIT — see [`LICENSE`](LICENSE). You are free to use, modify, and distribute this project, with attribution.

## Author

**Marajul Haque** — Capstone student, advised by Abu Nowshed Chy.
Built as a capstone project demonstrating edge AI, IoT, and LLM/RAG integration.

---

## Acknowledgments

- **llama.cpp** team (Georgi Gerganov et al.) — for the inference engine that makes local LLMs practical.
- **Microsoft** (Phi-3 Mini), **Meta** (Llama 3.2), **TinyLlama** team — for the open models.
- **Sentence-Transformers** (UKPLab), **FAISS** (Meta) — for the retrieval stack.
