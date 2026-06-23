# TinyRAG

> A lightweight, fully on-device Retrieval-Augmented Generation (RAG) assistant for smart-home IoT — running on a Raspberry Pi 5, with **zero cloud calls**.

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%205%20%7C%20Linux-lightgrey)

---

## What is TinyRAG?

TinyRAG is a **privacy-preserving smart-home assistant** that runs entirely on a small edge device. It reads your device manuals (PDF), a custom home FAQ (Markdown), and live IoT sensor data (temperature, humidity, energy, motion), and answers natural-language questions about your home using a **local** small language model (Phi-3 Mini 3.8B, Q4-quantized, served via llama.cpp).

**The entire pipeline runs offline.** Turn off your Wi-Fi — it still works.

### Example queries

- *"How do I reset my Nest thermostat to factory settings?"* → answers from the Nest manual.
- *"What was the average temperature in the living room this week?"* → answers from sensor logs.
- *"Why is my energy bill higher than usual this month?"* → cross-references sensor data + home FAQ.
- *"What is the meaning of life?"* → correctly refuses with the fallback message.

---

## Current Status (June 2026)

**Active phase: Phase 3 — Project Setup (laptop-first).**

We are building the entire system on a Dell Inspiron 15 3520 laptop first (Ubuntu 24.04 LTS, i5-1235U, 8 GB RAM). A Raspberry Pi 5 deployment is the **final** step, after the laptop version is fully working.

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 3 | Setup (repo, venv, llama.cpp build) | **In progress** |
| Phase 4 | Build (ingestion, retrieval, generation, sensors, API) | Planned |
| Phase 5 | Test (20-Q gold set + benchmarks, 3-model comparison) | Planned |
| Phase 6 | Deploy (Raspberry Pi 5 + real DHT22/PIR sensors) | Planned (Week 9) |
| Phase 7 | Report (capstone report + final demo) | Planned (Week 10) |

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

## Running on Laptop (current target)

> ⚠️ **The Raspberry Pi 5 is not here yet.** All development is on the laptop. The architecture is portable: the same code, models, and `config.yaml` work on both — only the deployment target in config changes.

### Prerequisites

- **OS:** Ubuntu 24.04 LTS (or 22.04 LTS, or any modern Debian/Ubuntu)
- **Python:** 3.12 (Ubuntu 24.04 ships this)
- **RAM:** 8 GB minimum
- **Disk:** ~10 GB free (for llama.cpp build + 3 GGUF models + 30 days of sensor data)
- **Build tools:** `build-essential`, `cmake`, `git` (installed in Step 3.3)
- **BLAS:** `libopenblas-dev`, `liblapack-dev` (installed in Step 3.3)

### One-command install (planned for Step 4.x)

```bash
git clone https://github.com/marajul/tinyrag.git
cd tinyrag
bash setup.sh        # installs Python deps + builds llama.cpp + downloads models
bash run.sh          # starts FastAPI on http://localhost:8000
```

> ⚠️ `setup.sh` and `run.sh` are written in later steps. For now, follow [`docs/06_roadmap_v2.md`](docs/06_roadmap_v2.md) step by step.

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

**For evaluators / advisors:** scope + SRS are enough.
**For new developers:** AGENT.md + architecture + roadmap.
**For contributors:** see [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Tech Stack (Pinned)

| Component | Choice | Why |
|-----------|--------|-----|
| LLM (primary) | Phi-3 Mini 3.8B Instruct, Q4_K_M GGUF | Best quality in the ≤4B class |
| LLM (compare) | TinyLlama 1.1B, Llama 3.2 3B | Required for 3+ model eval |
| LLM server | llama.cpp HTTP server | Mature, well-documented, CPU-only |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (384-d) | Small, fast, good quality |
| Vector store | FAISS `IndexFlatIP` | Simple, CPU-friendly, cosine via normalized inner product |
| Metadata DB | SQLite 3 (WAL mode) | Embedded, zero-config |
| API | FastAPI 0.115 + Uvicorn | Async, auto-docs |
| PDF parsing | pdfplumber | Robust, MIT |
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
