# AGENT.md — Project Context Handoff File

> **Purpose:** This file is the single source of truth for anyone (human or AI) picking up the TinyRAG project. If you are a new agent, **read this first** before doing anything. It tells you what the project is, what decisions have been made, where things live, and what to do next.

**Last updated:** 2026-06-24 (update 24)
**Project status:** Step 3.4a complete (housekeeping) + Step 4.5 still current — llama.cpp build now lives in `$HOME/.cache/llamacpp-build/` (persistent across reboots); token-based chunker live
**Next milestone:** Step 4.6 — Implement the embedder (Protocol + `SentenceTransformerEmbedder`)
**Canonical roadmap:** `docs/06_roadmap_v2.md` (the older `v1` and `laptop_v1` are historical only)
**Remote:** `https://github.com/marajulcsecu/tinyrag`
**Tip of `main`:** `2dac7e5` (see §11 Build Journal)
**Venv location:** `~/venvs/tinyrag` (symlinked as `.venv` in project root)
**OpenBLAS version:** 0.3.26 (verified via pkg-config)
**llama.cpp:** tag `gguf-v0.19.0` (commit `a290ce626663dae1d54f70bce3ca6d8f67aab62f`) — built at `${HOME}/.cache/llamacpp-build/build/` (colon-path workaround; persistent across reboots since Step 3.4a; symlinked into `llama.cpp/build/`)
**Models on disk:** phi-3-mini, tinyllama-1.1b, llama-3.2-3b, mistral-7b (all SHA-256 verified) — see `docs/MODELS.md`
**Synthetic data:** `data/sensor_logs/synthetic_30d.csv` — 51,840 rows, 6 sensors, 30 days, SEED=42 (gitignored, regenerable)

---

## 1. The Project in One Paragraph

**TinyRAG** is a privacy-preserving, fully on-device **Retrieval-Augmented Generation (RAG)** system for a smart home, deployed on a Raspberry Pi 5. It ingests smart-home device manuals (PDFs) and a custom home FAQ (Markdown), reads from real or simulated IoT sensors (temperature, humidity, energy, motion), and answers natural-language questions using a small local LLM (Phi-3 Mini 3.8B Q4-quantized) running via llama.cpp. The user interacts via a FastAPI web UI. The system is designed with a **clean, modular architecture** so any component (LLM, embedding model, vector store, sensor source, UI) can be swapped without rewriting the rest.

---

## 2. The Student

- **Name:** Marajul Haque
- **Role:** Capstone student
- **Advisor:** Abu Nowshed Chy
- **Skill level (per self-report):** Intermediate Python, new to LLMs/RAG
- **Stated priorities:**
  1. **Quality over speed** — has explicitly said "extra time is okay but our project should be best."
  2. **Clean architecture** — wants modular, professional, swappable design.
  3. **CV value** — wants demonstrable skills in edge AI, IoT, and LLM/RAG.
  4. **Professional workflow** — explicit "we always follow professional way, not garbage way."

---

## 3. Hardware Profiles

### Primary target — Raspberry Pi 5 (8 GB)

| Spec | Value |
|------|-------|
| SoC | Broadcom BCM2712, Cortex-A76 quad-core @ 2.4 GHz |
| RAM | 8 GB LPDDR4X |
| Storage | microSD (≥32 GB) + optional USB SSD |
| OS | Raspberry Pi OS 64-bit (Debian Bookworm) |
| Status | **Requested from lab assistant — not yet confirmed** |

### Fallback target — Dell Inspiron 15 3520 (laptop)

| Spec | Value |
|------|-------|
| CPU | Intel Core i5-1235U (12th gen, 10 cores: 2P + 8E, 12 threads reported) |
| RAM | 8 GB DDR4 |
| Storage | 512 GB SSD |
| GPU | Integrated Intel Graphics (ADL GT2) — not used for LLM |
| OS | **Ubuntu 24.04.4 LTS** (Wayland, GNOME 46, kernel 6.17.0-35-generic) |
| Status | **Available immediately, fully set up** |

> The laptop is actually faster than the Pi for LLM inference (more cores, faster memory bandwidth). The same code, models, and architecture will work on both. Only the `config.yaml` deployment target differs.

---

## 4. Locked-In Decisions (do not re-litigate without strong reason)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| D1 | Use case | **Smart Home Assistant** | Best CV keywords (Edge AI + IoT + LLM), easiest to demo, cheapest |
| D2 | Primary LLM | **Phi-3 Mini 3.8B Instruct (Q4_K_M quantized)** | Best quality for size in the ≤3B class |
| D3 | Secondary LLMs (for comparison) | TinyLlama 1.1B, Llama 3.2 3B, possibly Mistral 7B | Required for 3+ model evaluation |
| D4 | Embedding model | (TBD in architecture doc — likely `all-MiniLM-L6-v2` or `bge-small-en-v1.5`) | |
| D5 | LLM serving | **llama.cpp HTTP server** | Simplest, mature, well-documented |
| D6 | Vector store | (TBD in architecture doc — likely FAISS or ChromaDB) | |
| D7 | Backend framework | **FastAPI** | Modern, async, auto-docs |
| D8 | UI | Simple **HTML + vanilla JS** (no React/Vue for capstone simplicity) | |
| D9 | UI language | **English** | Best model support |
| D10 | Input mode (primary) | **Text via web UI** | Simpler, more reliable |
| D11 | Input mode (stretch) | **Voice via Whisper.cpp** | Modular adapter, only built if time allows |
| D12 | Knowledge base | **2–3 real device manuals (PDF) + 1 custom home FAQ (Markdown)** | Realism + control for evaluation |
| D13 | Sensor types | **Temperature, humidity, energy (kWh), motion** | Common smart-home sensors |
| D14 | Sensor source | **Pluggable: SimulatedCSVSource (default) + RealSerialSource (lab) + MQTTBrokerSource** | Graceful fallback if lab sensor unavailable |
| D15 | Conversation model | **Single-turn (no chat history)** | Simpler, more reliable; multi-turn adds complexity for marginal benefit on Pi |
| D16 | Demo format | **Live demo on Pi (primary) + recorded video (backup)** | Both — guarantees something works |
| D17 | Final report | **Brief 1-page related-work section comparing TinyRAG to PrivateGPT, Ollama, etc.** | Academic polish |
| D18 | Architecture quality bar | **Professional / clean / modular** | Explicit student requirement |

---

## 5. Architecture Principles (non-negotiable)

1. **Separation of concerns** — UI, backend, retrieval, generation, storage, sensor I/O are all separate modules.
2. **Dependency injection via interfaces** — every external dependency (LLM, vector store, embedding model, sensor source) is hidden behind a Python Protocol/ABC. Swap by changing config, not code.
3. **Configuration over hardcoding** — `config.yaml` is the only place that knows model paths, ports, chunk sizes, etc.
4. **No cloud calls at runtime** — verified by running with Wi-Fi off.
5. **Reproducible** — `setup.sh` and `run.sh` bring up the entire system from scratch.
6. **Testable** — core modules (chunking, retrieval, prompt construction) have unit tests.
7. **Professional logging** — structured logs, not print statements.

---

## 6. Project Structure (canonical — see `docs/03_architecture_v1.md` §5)

The layout below is the **canonical** Python package tree. It is the
output of Step 4.1 and is the same tree you'll see in
`docs/03_architecture_v1.md` §5. Files marked with a step number
(e.g. `4.5`) are not yet created; they will be added in that
Phase 4 step.

```
TinyRAG/
├── AGENT.md                      ← this file
├── README.md                     ← quick start
├── LICENSE
├── config.yaml                   ← single source of runtime config (Step 4.2)
├── setup.sh                      ← one-command install (Step 4.24)
├── run.sh                        ← one-command start (Step 4.24)
├── pyproject.toml                ← Python packaging
├── requirements.txt              ← pinned runtime deps
├── requirements-dev.txt          ← pinned dev/test deps
├── Makefile                      ← one-liner targets (test, lint, run, etc.)
├── .gitignore
│
├── docs/                         ← all planning docs (Phase 0-2, complete)
│
├── src/                          ← all source code
│   └── tinyrag/
│       ├── __init__.py           ← package docstring (Step 4.1)
│       ├── main.py               ← FastAPI app factory (Step 4.17)
│       ├── config.py             ← loads config.yaml (Step 4.2)
│       │
│       ├── api/                  ← HTTP layer (Step 4.1) — Step 4.17+ fills it
│       │   ├── __init__.py
│       │   ├── routes_query.py   ← POST /api/query, GET /api/status (4.19)
│       │   ├── routes_docs.py    ← POST/GET/DELETE /api/documents (4.18)
│       │   └── routes_admin.py   ← POST /api/admin/reindex (4.19)
│       │
│       ├── core/                 ← Domain logic, no I/O (Step 4.1) — 4.5+ fills it
│       │   ├── __init__.py
│       │   ├── chunker.py        ← Token-based chunking (4.5)
│       │   ├── retriever.py      ← Query → top-k chunks (4.12)
│       │   ├── prompt_builder.py ← Context + query → prompt (4.11)
│       │   ├── answer.py         ← Answer + citations dataclass (4.11)
│       │   └── sensor_summarizer.py ← Sensor data → text chunks (4.14)
│       │
│       ├── ingestion/            ← Doc → vector-store pipeline (Step 4.1) — 4.4+ fills it
│       │   ├── __init__.py
│       │   ├── pipeline.py       ← Orchestrator: parse → chunk → embed → store (4.9)
│       │   ├── parsers.py        ← PDF / TXT / MD → text (4.4)
│       │   └── embedder.py       ← sentence-transformers wrapper (4.6)
│       │
│       ├── generation/           ← LLM seam (Step 3.7a — already exists)
│       │   ├── __init__.py
│       │   └── llm_client.py     ← LLMClient Protocol + LlamaCppClient + FakeLLMClient
│       │
│       ├── storage/              ← Persistence (Step 4.1) — 4.7+ fills it
│       │   ├── __init__.py
│       │   ├── vector_store.py   ← FAISS wrapper (4.8)
│       │   └── metadata.py       ← SQLite wrapper (4.7)
│       │
│       ├── sensors/              ← Pluggable sensor sources (Step 4.1) — 4.13 fills it
│       │   ├── __init__.py
│       │   ├── base.py           ← SensorSource Protocol
│       │   ├── simulated.py      ← SimulatedCSVSource
│       │   ├── serial_dht.py     ← RealSerialSource (DHT22 + PIR)
│       │   └── mqtt.py           ← MQTTBrokerSource
│       │
│       ├── input_adapters/       ← Pluggable input (Step 4.1) — 4.19+ fills it
│       │   ├── __init__.py
│       │   ├── base.py           ← InputAdapter Protocol
│       │   ├── text.py           ← TextInputAdapter
│       │   └── voice.py          ← VoiceInputAdapter (stretch)
│       │
│       ├── ui/                   ← Static web assets (Step 4.1) — 4.21+ fills it
│       │   ├── __init__.py
│       │   ├── static/           ← style.css, chat.js, admin.js (4.21+)
│       │   │   └── .gitkeep      ← placeholder until real files land
│       │   └── templates/        ← index.html, admin.html (4.21+)
│       │       └── .gitkeep      ← placeholder until real files land
│       │
│       ├── observability/        ← Structured logging (Step 4.1) — 4.3 fills it
│       │   ├── __init__.py
│       │   └── logger.py         ← structlog config + get_logger
│       │
│       └── models/               ← GGUF catalog + downloader (Step 3.5 — predates §5)
│           ├── __init__.py
│           ├── registry.py       ← MODEL_REGISTRY + ModelEntry
│           └── downloader.py     ← ModelDownloader with SHA-256 verify
│
├── data/                         ← runtime data (gitignored)
│   ├── documents/                ← uploaded PDFs/MD
│   ├── sensor_logs/              ← CSV/JSON sensor data (synthetic_30d.csv from Step 3.8)
│   ├── vector_store/             ← FAISS index files
│   └── metadata.db               ← SQLite
│
├── models/                       ← downloaded GGUF models (gitignored)
│
├── tests/                        ← pytest unit tests
│   ├── conftest.py               ← shared fixtures (Step 4.1 — empty for now)
│   ├── test_smoke.py             ← runtime-deps import check (Step 3.2)
│   ├── test_skeleton.py          ← project-layout integrity (Step 4.1)
│   ├── test_config.py            ← typed Settings loader (Step 4.2)
│   ├── test_llm_client.py        ← LLMClient Protocol + concrete (Step 3.7a)
│   ├── test_download_models.py   ← GGUF downloader (Step 3.5)
│   ├── test_generate_synthetic_sensors.py ← synthetic data (Step 3.8)
│   ├── test_smoke_test.py        ← Phase 3 e2e smoke (Step 3.9)
│   ├── test_chunker.py           ← (4.5)
│   ├── test_retriever.py         ← (4.12)
│   ├── test_prompt_builder.py    ← (4.11)
│   └── test_parsers.py           ← (4.4)
│
├── scripts/                      ← operational scripts
│   ├── download_models.py        ← GGUF downloader CLI (Step 3.5)
│   ├── generate_synthetic_sensors.py ← 30-day sensor data (Step 3.8)
│   ├── smoke_test.py             ← Phase 3 e2e smoke (Step 3.9)
│   ├── verify_llamacpp.py        ← llama.cpp sanity check (Step 3.4)
│   ├── build_llamacpp.sh         ← llama.cpp build (Step 3.4)
│   ├── ingest.py                 ← CLI: ingest a document (4.9)
│   ├── evaluate.py               ← run the 20-Q eval set (5.4)
│   └── benchmark.py              ← measure latency, RAM (5.7)
│
└── reports/                      ← generated benchmarks, eval results
    ├── latency.csv
    ├── ram_usage.csv
    ├── accuracy_per_model.csv
    └── final_report.pdf          ← capstone report (7.7)
```

---

## 7. Documentation Roadmap (what we'll write, in order)

| Order | Document | Status | Purpose |
|-------|----------|--------|---------|
| 0 | `00_high_level_plan.md` | ✅ Written | Whole-journey visualization for the student |
| 1 | `01_project_scope_v1.md` | ✅ Written (historical) | First draft scope |
| 1' | `01_project_scope_v2.md` | ✅ Written (canonical) | Refined scope with all decisions baked in |
| 2 | `02_srs_v1.md` | ✅ Written | System Requirements Specification |
| 3 | `03_architecture_v1.md` | ✅ Written | C4 model + Protocol interfaces + module breakdown |
| 4 | `04_database_design_v1.md` | ✅ Written | FAISS + SQLite + CSV schemas |
| 5 | `05_tech_stack_v1.md` | ✅ Written | Pinned versions, build flags, requirements.txt |
| 6 | `06_roadmap_v1.md` | ✅ Written (superseded) | Original Pi-primary plan |
| 6' | `06_roadmap_v2.md` | ✅ Written (**canonical**) | Laptop-first full build, then 1-week Pi deploy |
| 7a | `evaluation/gold_set.md` | ✅ Written | The 20 evaluation questions |
| 7b | `evaluation/scoring_rubric.md` | ✅ Written | Human-judgment 4-point rubric |
| 7c | `evaluation/eval_script_spec.md` | Future (Phase 5.6) | Spec for `scripts/eval.py` |

---

## 8. Current State & Immediate Next Steps

**Where we are right now:**
- ✅ Use case selected (Smart Home Assistant)
- ✅ LLM selected (Phi-3 Mini 3.8B)
- ✅ All major decisions made
- ✅ All 8 planning docs complete (Phase 0–2 done)
- ✅ Evaluation methodology complete (gold set + scoring rubric)
- ✅ **Phase 3 complete (Steps 3.1–3.9)** — repo, env, system deps, llama.cpp, all 4 GGUF models, LLMClient, 30-day synthetic sensor data, and the Phase 3 end-to-end smoke test
- ✅ **Step 4.1 complete** — 9 subpackages from `docs/03_architecture_v1.md` §5 are now importable Python subpackages, each with a docstring explaining its responsibility
- ✅ **Step 4.2 complete** — typed Settings loader + `config.yaml`; FR-49..FR-52 satisfied
- ✅ **Step 4.3 complete** — structlog-based structured logger; the project's single logging seam
- ✅ **Step 4.4 complete** — document parsers (PDF/TXT/MD); ingestion pipeline can now turn any uploaded file into structured text
- ✅ **Step 4.5 complete** — token-based chunker; parsed documents split into embedding-ready ~400-token chunks with overlap and sentence-boundary respect
- ⏳ Next: Step 4.6 — Implement the embedder (Protocol + `SentenceTransformerEmbedder`)

**Immediate next step (Step 4.6 — agent action):**

Step 4.6 fills `src/tinyrag/ingestion/embedder.py` with an `EmbeddingModel` Protocol (the architecture doc §6.2 contract) and a `SentenceTransformerEmbedder` concrete implementation that loads `all-MiniLM-L6-v2` (the embedding model pinned in `config.yaml`'s `embedding.model_name`) once and exposes `.embed(texts: list[str]) -> list[list[float]]` plus a `.dimension` property. Embedding model load is lazy (first call), the dimension is asserted against the config at load time (so a swap to `bge-small-en-v1.5` fails loudly if you forget to update `embedding.dimension` in config.yaml).

**Optional parallel student action — none for Step 4.6. You can verify the Step 4.5 chunker yourself with:**

```bash
# 1. Run the chunker tests (34 should pass)
PYTHONPATH=. .venv/bin/pytest tests/test_chunker.py -v

# 2. Quick REPL probe — the roadmap's "2000-token text → ~5 chunks" spot-check.
#    We use a clearly-bounded sentence fixture so you can see the sentence-
#    boundary trim working.
PYTHONPATH=src .venv/bin/python -c "
from tinyrag.core import default_chunker

# ~2800-token text with clear sentence boundaries
filler = 'the quick brown fox jumps over the lazy dog '
text = ' '.join(f'Sentence {i}. {filler}' for i in range(200))
chunker = default_chunker()
print(f'input: {chunker.count_tokens(text)} tokens')
chunks = chunker.chunk(text, source='spotcheck.txt')
print(f'chunks: {len(chunks)}')
for c in chunks[:3]:
    print(f'  [{c.chunk_index}] {c.token_count} tokens, ends: {c.text[-30:]!r}')
print('  ...')
c = chunks[-1]
print(f'  [{c.chunk_index}] {c.token_count} tokens, ends: {c.text[-30:]!r}')
"

# 3. End-to-end with a real file (parser → chunker pipeline preview):
PYTHONPATH=src .venv/bin/python -c "
from pathlib import Path
from tinyrag.ingestion.parsers import parse
from tinyrag.core import default_chunker
p = Path('/tmp/manual.txt')
p.write_text('A long manual. ' + ('The thermostat pairs via Bluetooth. ' * 200), encoding='utf-8')
doc = parse(p)
chunks = default_chunker().chunk(doc.text, source=p.name)
print(f'parsed {len(doc.text)} chars, got {len(chunks)} chunks')
print(f'first chunk: {chunks[0].text[:80]}...')
print(f'last chunk ends: {chunks[-1].text[-80:]!r}')
"
```

The first command runs the test suite. The second is the exact "feed a 2000-token text, verify you get ~5 chunks with overlap" check from the roadmap §4.5 — note how each chunk ends at a sentence boundary (`.`). The third shows the parsers → chunker pipeline: a 200-sentence TXT becomes N chunks, each carrying the source filename (`source=manual.txt`) and the chunk's character offset in the original document.

You can also verify the Step 4.2 config yourself with:

```bash
# 1. Read the config — sanity-check the values match your environment
less config.yaml

# 2. Run the config tests (44 should pass)
.venv/bin/pytest tests/test_config.py -v

# 3. Load the config from a Python REPL and inspect a field
cd src && ../.venv/bin/python -c "
from tinyrag.config import load_settings
s = load_settings('../config.yaml')
print('llm.model_path =', s.llm.model_path)
print('deployment.target =', s.deployment.target)
print('sensors.source =', s.sensors.source)
print('project_root =', s.project_root())
"
```

You can also verify the Step 4.1 skeleton yourself with:

```bash
# 1. See the project tree — should match docs/03_architecture_v1.md §5
tree src tests -L 3

# 2. Run the skeleton integrity tests (57 should pass)
.venv/bin/pytest tests/test_skeleton.py -v

# 3. Confirm the package itself imports cleanly + shows its docstring
cd src && ../.venv/bin/python -c "import tinyrag; print(tinyrag.__doc__)"
```

---

## 9. Open / Pending Questions

These are NOT blockers, but should be resolved before we reach Week 5:

1. **Lab Pi 5 confirmation** — student is waiting on lab assistant. **Backup: use laptop.**
2. **Sensor availability from lab** — DHT22 + PIR? Or just simulate? **Default: simulate.**
3. **OS on the Dell laptop** — Windows 11? Ubuntu? Need to know before writing setup script. **Default: assume Ubuntu 22.04 LTS or WSL2.**
4. **Do you have any specific smart-home device manuals in mind?** (e.g., you own a Nest thermostat, you have a particular bulb brand). **Default: download 2-3 public manuals from manufacturer websites.** ✅ **Resolved — using 2–3 public manuals.**
5. **Do you want the project to be open-sourced on GitHub?** (CV value, can showcase to recruiters.) **Default: yes, MIT license.** ✅ **Resolved — public repo, MIT license.**

---

## 10. How to Use This File

- **If you are the student returning after a break:** read this file first, then read the latest version of each doc in `docs/`.
- **If you are a new agent:** read this file, then `docs/00_high_level_plan.md`, then `docs/01_project_scope_v2.md` (or v1 if v2 not yet written), then ask the student which doc to write next.
- **If you are an evaluator (advisor / panel):** read this file, then read `01_project_scope_v2.md` and `02_srs_v1.md`.

---

## 11. Build Journal — Step-by-Step Tracker

This section is the **running log of every step executed**, in execution order. It is updated as each step of the canonical roadmap (`docs/06_roadmap_v2.md`) is completed.

**Why this exists:** so the student, advisor, and any future contributor can answer three questions instantly:
1. *What has been done so far?*
2. *What's left to do, in what order?*
3. *What commit / artifact corresponds to each step?*

### 11.1 Phase 3 — Setup (laptop)

| Step | Description | Status | Commit SHA | Commit message | Notes |
|------|-------------|--------|------------|----------------|-------|
| 3.1 | Initialize Git repository | ✅ Done | `f78e0a7` (tip) | `docs(agent): mark Step 3.1 complete and add Build Journal section` | 35+ files pushed to `https://github.com/marajulcsecu/tinyrag`. History is a 3-commit rebase: GitHub's auto-MIT (dca6b0d) → initial repo (e401c6d) → AGENT.md Build Journal (f78e0a7). |
| 3.2 | Set up Python venv + pinned requirements | ✅ Done | `1519733` | `chore(deps): set up pinned Python environment (Step 3.2)` | Added requirements.txt, requirements-dev.txt, pyproject.toml, Makefile, .env.example, tests/test_smoke.py, src/tinyrag/__init__.py. Venv at `~/venvs/tinyrag` (symlinked as `.venv`) because project path contains colons. 32/32 smoke tests pass. |
| 3.3 | Install system deps for llama.cpp + OpenBLAS | ✅ Done | `aca827c` | `chore(deps): add system dep installer and native build manifest (Step 3.3)` | Installed libopenblas-dev 0.3.26, liblapack-dev, tree via apt. Added scripts/install_system_deps.sh (idempotent, --check, --with-extras), docs/BUILDS.md (build manifest with placeholders for llama.cpp SHA), 3 new Makefile targets (deps-system, deps-verify, deps-extras) + 3 placeholders for Step 3.4 (llama-dir, build-llamacpp, build). |
| 3.4 | Build llama.cpp from source with OpenBLAS | ✅ Done | `2b61567` | `feat(llm): build llama.cpp with OpenBLAS (Step 3.4)` | Cloned llama.cpp at tag `gguf-v0.19.0` (commit `a290ce62`); built with `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS`; binary 9.4 MB; OpenBLAS linked (verified via ldd). `scripts/verify_llamacpp.py` passes 7/7 checks. **Colon-path workaround:** because project path contains `:`, GNU Make can't parse Makefile targets, so the build was diverted to `/tmp/llamacpp-build/` and symlinked back into `llama.cpp/build/` (BUILDS.md §2.2.1). |
| 3.4a | Move llama.cpp build out of `/tmp/` for persistence | ✅ Done | `2dac7e5` | `chore(llama): move llama.cpp build to $HOME/.cache/ (Step 3.4a)` | **Goal:** `/tmp/llamacpp-build/` is volatile (wiped on reboot, by `tmpreaper`, on some distros by routine maintenance). Move to `${HOME}/.cache/llamacpp-build/` (XDG cache home — persistent across reboots). **Why now:** Step 3.4a was deferred from Step 3.4 because we hadn't yet proven recovery worked. Confirmed at Step 3.9 (`make smoke-e2e E2E_CLIENT=fake` ran cleanly against the moved binary) and Steps 4.3-4.5 don't touch the build. Doing it before Step 4.17 (`main.py`) is the natural moment because that's when `make run-llm` will be invoked more frequently in dev. **Changes:** `scripts/build_llamacpp.sh` — renamed `EXTERNAL_BUILD_DIR`/`EXTERNAL_SRC_DIR` from hardcoded `/tmp/llamacpp-build` to `${HOME}/.cache/llamacpp-build` (via a new `EXTERNAL_BUILD_PARENT` constant for clarity); updated the warn message to say "diverting build out of project tree" rather than "diverting build to /tmp" so future readers understand the *why* without the path. `scripts/verify_llamacpp.py` — `_resolve_actual_paths()` now checks `${HOME}/.cache/llamacpp-build/` first, then falls back to `/tmp/llamacpp-build/` for users with old pre-Step-3.4a builds (idempotent migration). `Makefile` — improved the `run-llm` recovery hint to mention both possible build locations so a user who hasn't migrated yet sees the right path. `docs/BUILDS.md` §2.2.1 rewritten to describe the persistent location; the §2.5 build record table's "Build dir" row and "Known caveats" paragraph updated; the "Other build issues" table's row 5 updated. **Migration steps actually executed:** `mkdir -p $HOME/.cache && mv /tmp/llamacpp-build $HOME/.cache/llamacpp-build` (atomic rename — 312 MB, instant); `rm llama.cpp/{build,bin} && ln -s $HOME/.cache/llamacpp-build/build llama.cpp/build && ln -s $HOME/.cache/llamacpp-build/build/bin llama.cpp/bin` (re-create the project symlinks). **Critical post-migration verification caught a real bug:** `verify_llamacpp.py` reported only 4/7 passes after the move. Root cause: the existing binary had a hardcoded `RUNPATH` of `/tmp/llamacpp-build/build/bin:` baked into its ELF headers by the original cmake build, so the dynamic loader was looking in the (now-empty) `/tmp` path for the .so files (`libllama.so.0`, `libllama-common.so.0`, etc.) and failing to find them. **`mv` is not enough — the binary must be rebuilt** because `RUNPATH` is a static ELF property. Fixed by `bash scripts/build_llamacpp.sh --skip-clone --clean` (reuse source tree at the new location, fresh build dir so the linker bakes the correct RUNPATH). Verified: `readelf -d llama.cpp/build/bin/llama-server | grep RUNPATH` now shows `[/home/marajul/.cache/llamacpp-build/build/bin:]`, and `python scripts/verify_llamacpp.py` is back to **7/7 checks passed**. **Lesson:** if you ever move a build directory again, always rebuild (or `patchelf --set-rpath`), don't just `mv`. Full test suite still **357/357 passing** with `PYTHONPATH=.` (this step touched no Python logic). No new runtime deps. |
| 3.5 | Download Phi-3 Mini 3.8B GGUF | ✅ Done | `cf796b9` | `feat(models): add GGUF downloader with SHA-256 verification (Step 3.5)` | Added `src/tinyrag/models/{registry,downloader}.py` (canonical 4-model catalog: Phi-3 primary, TinyLlama/Llama 3.2/Mistral for eval), `scripts/download_models.py` (CLI with --list, --model, --all, --verify-only, --force, --json), `docs/MODELS.md` (human-readable catalog), 15 hermetic pytest tests (registry shape, idempotency, checksum rejection, HTTP Range resume, progress callbacks, CLI). Uses stdlib `urllib` (no new dep). Standardised on `models/<id>.gguf` on-disk naming. **Model file itself is NOT yet on disk** — student runs `make download-llm` to fetch ~2.3 GB Phi-3 in Step 3.6. |
| 3.6 | 🛑 RISK GATE: First llama.cpp server run on laptop | ✅ Done | `ee984c0` | `feat(llm): add LLMClient Protocol + LlamaCppClient + smoke test (Step 3.7 — note: see 3.7 below for numbering correction)` | Student action completed: `make download-llm` (2.3 GB Phi-3 fetched, SHA-256 verified against registry) → `make run-llm` → `curl http://127.0.0.1:8080/v1/models` returned HTTP 200 with the expected model metadata. Confirms the entire native + model stack is wired end-to-end. **Numbering note:** the commit subject says "Step 3.7" because at the time I conflated the LLM seam + smoke test under one commit. The actual roadmap ordering is 3.6 = first server run, 3.7 = download comparison models (next row), 3.8 = synthetic sensors (this commit). |
| 3.7 | Download comparison models (TinyLlama, Llama 3.2 3B) | ✅ Done | `ee984c0` (+ 3 fix commits: `098d438`, `412e7f3`, `51e9f6e`) | same as 3.6 (LLMClient commit) | Student action completed: downloaded tinyllama-1.1b (637 MB) and llama-3.2-3b (1.88 GB) via `scripts/download_models.py`. **Mistral 7B fix in `412e7f3`:** original TheBloke repo returned 401; switched to bartowski mirror and re-verified (4.37 GB public mirror, HTTP 200). **Truncation fix in `51e9f6e`:** Llama 3.2 first download silently stopped at 753 MB of the expected 1.88 GB and the manifest recorded a "valid" SHA for the truncated bytes; llama-server later failed with `tensor 'blk.15.ffn_up.weight' data is not within the file bounds`. Fixed in `_fetch` (short-read guard vs Content-Length) and `download` (registry `expected_size_bytes` cross-check, 5% tolerance). 3 new tests in `TestTruncationGuard`. Student re-downloaded Llama 3.2 — 1.88 GB clean. All 4 models (`phi-3-mini`, `tinyllama-1.1b`, `llama-3.2-3b`, `mistral-7b`) verified end-to-end. |
| 3.7a | LLMClient Protocol + LlamaCppClient + smoke test | ✅ Done | `ee984c0` | `feat(llm): add LLMClient Protocol + LlamaCppClient + smoke test (Step 3.7)` | Added `src/tinyrag/generation/{__init__,llm_client}.py` (~430 lines): `LLMClient` `@runtime_checkable` Protocol, `FakeLLMClient` deterministic stub (for tests / offline dev), `LlamaCppClient` real HTTP/SSE client (talks to llama-server's `/v1/chat/completions` with stream=true, parses Server-Sent Events, extracts `choices[].delta.content`, terminates on `[DONE]`, captures `usage` block, falls back to whitespace-split token estimation when usage is missing). Typed exception hierarchy: `LLMError` → `LLMUnavailableError` (5xx, connection, timeout) / `LLMRefusedError` (4xx). Lazy httpx.Client ownership. Plus `scripts/smoke_test_llm.py` (CLI: `--model`, `--all`, `--base-url`, `--prompt`, `--max-tokens`, `--models-dir`, `--json`) and `tests/test_llm_client.py` — **31 hermetic tests** using `httpx.MockTransport` covering: ChatMessage shape, Protocol duck-typing (no inheritance), FakeLLMClient canned responses + overrides + raise_after_tokens, LlamaCppClient SSE parsing (concatenation, [DONE] termination, malformed lines, role-only chunks), 5xx/4xx/connection error mapping, lazy client ownership, multi-message (system+user) roundtrip. New Makefile targets: `smoke-llm`, `smoke-llm-all`. **This is technically an "extra" step that doesn't appear in the roadmap by name** — the roadmap's Phase 3 only requires the LLM to be downloadable + runnable, but writing the LLMClient Protocol now means Phase 4 (FastAPI) can start straight away. Documented here so future contributors know where the LLM seam lives. |
| 3.8 | Generate synthetic sensor data | ✅ Done | `b7680d3` | `feat(sensors): add 30-day synthetic sensor generator (Step 3.8)` | Added `scripts/generate_synthetic_sensors.py` (~480 lines): numpy + pandas, SEED=42 reproducibility, 5-min resolution, 6 sensors (living_room_temp, living_room_hum, bedroom_temp, bedroom_hum, kitchen_motion, house_energy), long-format CSV output to `data/sensor_logs/synthetic_30d.csv` (gitignored). Per-sensor physics: temperature = daily sinusoid + per-room offset + Gaussian noise; humidity = weakly anti-correlated with temp, bounded [30, 80]; motion = Bernoulli with hour-of-day + weekday/weekend rates; energy = base draw + morning/evening peaks + weekend multiplier + 5% appliance surges. CLI: `--start`, `--days`, `--interval-min`, `--out`, `--seed`, `--summary`, `--json`. Generated 51,840 rows × 6 sensors (30 days × 288 ticks/day). Plus `tests/test_generate_synthetic_sensors.py` — **34 hermetic tests** covering: schema conformance (§6.1 columns + dtypes + canonical sensors), no NaN, realistic value ranges (temp 15-30, humidity 30-80, motion 0/1, energy ≥ 0), daily patterns (afternoon temp peak, dinner motion peak), SEED=42 reproducibility (same/different seed → same/different output), summary helper, time-grid correctness (5-min spacing, no duplicates per sensor), custom start date. Full suite: **115/115 tests pass** (was 81, added 34). No new runtime deps — `pandas` + `numpy` were already pinned. |
| 3.9 | Phase 3 checkpoint: end-to-end smoke test | ✅ Done | `d882691` | `feat(smoke): add Phase 3 end-to-end smoke test (Step 3.9)` | Added `scripts/smoke_test.py` (~370 lines): hard-coded "What is 2+2?" probe sent through `LLMClient` (real llama-server or `FakeLLMClient`), `SmokeResult` dataclass with `to_dict()` for JSON output, `print_human` / `print_json` formatters, CLI with `--client {real,fake}`, `--base-url`, `--model`, `--query`, `--max-tokens`, `--json`, `--quiet`. Exit codes: 0 = success, 1 = empty/error, 2 = argparse. Catches every `LLMError` and converts to a structured failed result (no traceback to stderr). Plus `tests/test_smoke_test.py` — **26 hermetic tests** covering: contract constants (defaults match Makefile), client factories, `run_smoke()` success/empty/whitespace/LLMError paths, `SmokeResult.to_dict()` shape + JSON-safety, full `main()` end-to-end (`--json`/`--quiet`/`--query`/bad-client exit 2/no-server exit 1+structured-error), `print_human`/`print_json` formatting. All hermetic — uses FakeLLMClient or synthetic BrokenClient/SilentClient classes; no network. Plus new `make smoke-e2e` target honoring `E2E_CLIENT=fake` for hermetic CI mode. **Bonus fix in same commit:** Makefile help-regex bug — `[a-zA-Z_-]` didn't match digits, so targets like `smoke-e2e` (digit `2`) were silently dropped from `make help`. Fixed across all 8 `grep -E` occurrences. Verified: `make smoke-e2e E2E_CLIENT=fake` exits 0 with `[ OK ]` banner; `make smoke-e2e` (no llama-server) exits 1 with structured `LLMError: ...Connection refused...` JSON. **Phase 3 is now complete.** Full suite: **141/141 tests pass** (was 115, added 26). Lint clean. |
| 4.1 | Initialize the project skeleton (folders only) | ✅ Done | `a7b29fd` | `feat(skeleton): initialize project skeleton folders (Step 4.1)` | Created the full `src/tinyrag/` subpackage tree from `docs/03_architecture_v1.md` §5. **9 new subpackages** (api, core, ingestion, storage, sensors, input_adapters, ui, observability + the rewritten top-level `__init__.py`); `tinyrag.generation` and `tinyrag.models` already existed from earlier steps. Every `__init__.py` has a non-empty docstring explaining the subpackage's responsibility, listing the modules it will hold, and pointing at the Phase 4 step numbers that will create them. Each docstring follows the same convention as `tinyrag.generation.__init__` (which already existed): "Why a subpackage?" rationale + "Location: ..." footer. The top-level `__init__.py` was rewritten from empty to a full package docstring that lists every subpackage and explains the one-way dependency rule (api → core → stdlib only). **`tests/conftest.py`** created with a docstring-only stub (no fixtures yet — they'll land in Steps 4.2/4.5 as the test suite grows). **`ui/static/` and `ui/templates/`** created with `.gitkeep` placeholders so git tracks the otherwise-empty dirs; placeholders will be removed when the actual CSS/JS/HTML files land in Steps 4.21-4.23. **`tests/test_skeleton.py`** — **57 hermetic tests** guarding the layout: (1) every subpackage dir exists with non-empty `__init__.py` (parametrised over 10 subpackages × 3 checks = 30), (2) every subpackage is importable (10), (3) UI subdirs exist + have `.gitkeep` (4), (4) `tests/conftest.py` + `tests/test_smoke.py` still present + have key markers (3), (5) **no `__init__.py` may import a runtime dep** (faiss, fastapi, sentence_transformers, torch, structlog, pydantic, yaml, pdfplumber — 10 tests) — this last guard catches a common mistake: a future contributor adding `from .llm_client import LLMClient` to the top-level `__init__.py` would transitively pull in httpx and break the smoke import check on a fresh machine. Full suite: **198/198 tests pass** (was 141, +57). Lint clean (after `ruff check --fix` for 2 trailing-newline warnings). No new runtime deps. Structure verified: `tree src tests -L 3` matches §5 exactly. |
| 4.2 | Set up `config.yaml` + `Settings` loader | ✅ Done | `88e7d01` | `feat(config): add typed Settings loader and config.yaml (Step 4.2)` | Added `config.yaml` (~150 lines) at project root with the canonical schema from `docs/04_database_design_v1.md` §config (mirroring `docs/02_srs_v1.md` Appendix B). Every field has an inline comment explaining its purpose, default, and laptop-vs-Pi rationale. `deployment.target: laptop` per Step 4.2 instructions. **9 top-level sections** — all required to be present (even if `{}`). Added `src/tinyrag/config.py` (~640 lines): Pydantic v2 Settings with 9 typed sub-models (one per YAML section), all `frozen=True, extra="forbid"`. **4 typed enums** (DeploymentTarget, SensorSource, LogLevel, EmbeddingDevice) with Pydantic-v2 string-to-enum coercion. **Range constraints** on every numeric field (e.g. llm.temperature ∈ [0, 2], server.port ∈ [1, 65535]). **Cross-field validation**: `chunking.chunk_overlap < chunking.chunk_size` (else the chunker loops forever); `deployment.target: laptop` + `sensors.source: real_serial` is rejected (FR-18 [L] — laptop has no GPIO). The laptop-vs-real_serial check is implemented as a two-pass in `load_settings()` (build partial Settings from default-filled broken sections, then run the cross-field check) so the user always sees the cross-field error even when other fields are also broken. **`Settings.resolve(relative_path)`** anchors relative paths to the config file's directory (a `PrivateAttr` set by `load_settings`). **Typed exception hierarchy** `ConfigError` → `ConfigNotFoundError` / `ConfigValidationError`; the latter wraps the original Pydantic `ValidationError` on `self.original`. **Friendly error summary** when validation fails: one `dot.path: message` line per failing field, in the same format mypy/ruff use (cleaner than Pydantic's default). **Why not `pydantic_settings.BaseSettings`?** It's env-first; TinyRAG is single-process and single-config, and mixing env vars + YAML is a recipe for "which one wins?" confusion. Custom loader is ~30 lines, fully testable. Plus `tests/test_config.py` — **44 hermetic tests**: TestPublicSurface (9 — every sub-model instantiates with defaults), TestEnumCoercion (5), TestLoadSettings (6 — happy path + idempotence + frozen + resolve()), TestLoadSettingsErrors (9 — missing file / malformed YAML / missing section / wrong type / out of range / unknown enum / unknown field / invalid top-level type / empty file rejected), TestCrossFieldValidation (6 — laptop+real_serial rejected, pi+real_serial allowed, etc.), TestConfigYamlMatchesSpec (3 — real config.yaml matches SRS Appendix B + database design §config), TestFROrNumbers (4 — explicit FR-49..FR-52 traceability). **All 4 FRs satisfied** and testable. Full suite: **242/242 tests pass** with `PYTHONPATH=.` (was 198, +44). Lint clean. No new runtime deps — `pydantic==2.9.2` and `pyyaml==6.0.2` were already pinned in `requirements.txt`. |
| 4.3 | Add the structlog-based structured logger | ✅ Done | `7629c13` | `feat(observability): add structlog-based structured logger (Step 4.3)` | Added `src/tinyrag/observability/logger.py` (~340 lines) — the project's **single seam for log output**. Architecture doc §12.1 specifies two parallel pipelines: stdout (pretty for humans during dev) + a JSON file (`logs/tinyrag.log`, append-only, for postmortem). Implemented via stdlib `dictConfig` + `structlog.stdlib.ProcessorFormatter` so the shared processor chain (`merge_contextvars`, `add_log_level`, `TimeStamper(iso, utc)`, `add_logger_name`, `StackInfoRenderer`, `format_exc_info`) runs once per log call, then each handler's formatter picks its final render — JSON or pretty. **`configure_logging(settings, *, project_root=None)`** wires both handlers via `dictConfig`, then bridges structlog to stdlib via `structlog.stdlib.LoggerFactory` + `ProcessorFormatter.wrap_for_formatter`. **Eagerly creates the log file's parent dir** so a permission error surfaces at startup with a clean `LoggingError` instead of a traceback at first write. **Chatty third-party loggers** (`httpx`, `httpcore`, `sentence_transformers`) are pinned to WARNING so model-load progress bars don't drown the actual application logs. **`get_logger(name=None)`** returns a `structlog.stdlib.BoundLogger` (bound to the module name) — the standard `log.info(event_name, **kwargs)` API every other module will use. **`LoggingError`** — typed exception for config failures; raised by the composition root in `main.py` (Step 4.17) for clean startup messages. Updated `src/tinyrag/observability/__init__.py` to re-export the three public symbols (`configure_logging`, `get_logger`, `LoggingError`). **`get_logger` works before `configure_logging`**: structlog has a default `PrintLoggerFactory`, so any module that calls `get_logger(__name__)` at import time (e.g. during a test) gets a usable logger — no `LoggingError: configure_logging not called` foot-gun. Plus `tests/test_logger.py` — **25 hermetic tests**: TestPublicSurface (4 — re-exports work + `get_logger` returns a BoundLogger with `info`/`warning`/`error`/`debug`), TestBuildDictConfig (9 — stdout handler always present, file handler only when path set, stdout formatter flips pretty↔JSON on `json_format`, **file formatter is always JSON regardless of `json_format`** — the §12.1 invariant, root logger has both handlers + propagates, third-party quiet-logs are WARNING), TestConfigureLogging (3 — idempotence verified by exact type-name count `["StreamHandler", "WatchedFileHandler"]` — important because `WatchedFileHandler` IS a `StreamHandler` subclass, which would otherwise inflate the count, unwritable parent dir raises `LoggingError` not `OSError`), TestLogOutput (6 — pretty stdout contains event+keys, JSON stdout is parseable per-line with `timestamp`/`level`/`logger`/`event`, file is always JSON when stdout is pretty, file disabled when path=None, missing nested parent dir auto-created, stdlib `logging.getLogger` calls also flow through our handlers), TestLogLevels (3 — INFO filters DEBUG, DEBUG passes DEBUG, ERROR filters INFO). **25/25 logger tests pass.** Full suite: **267/267 tests pass** with `PYTHONPATH=.` (was 242, +25). Lint clean. No new runtime deps — `structlog==24.4.0` was already pinned. **Quick REPL probe** (run from `src/`): `python -c "from tinyrag.config import load_settings; from tinyrag.observability.logger import configure_logging, get_logger; configure_logging(load_settings('../config.yaml').logging); log = get_logger('demo'); log.info('hello', key='value')"` → one pretty line on stdout, one JSON line in `logs/tinyrag.log`. |
| 4.4 | Implement the document parsers (PDF, TXT, MD) | ✅ Done | `29e2810` | `feat(ingestion): add document parsers (Step 4.4)` | Added `src/tinyrag/ingestion/parsers.py` (~390 lines) — the **first content step** of the RAG pipeline (every prior step was plumbing). Three concrete parsers behind a single `DocumentParser` Protocol + a `parse(path)` dispatcher. **`:class:`ParsedDocument`** frozen dataclass (`text: str`, `pages: list[tuple[int, str]]` for FR-2 page-number preservation, `metadata: dict` JSON-safe). **`:class:`PdfParser`** uses pdfplumber (architecture §15.1 chose pdfplumber over PyPDF2 for complex-layout handling); lazy import so a Markdown-only project doesn't pay the ~200 ms pdfplumber cost; FlateDecode streams via stdlib `zlib`; per-page extraction preserves the 1-based page number; raises `PdfReadError` on malformed bytes and `EmptyDocumentError` when no text layer exists (i.e. scanned-without-OCR). **`:class:`TxtParser`** reads UTF-8 with BOM tolerance (`utf-8-sig`) so Windows-Notepad files work out of the box; propagates `UnicodeDecodeError` so the user knows the file is the wrong encoding (no silent Latin-1 fallback). **`:class:`MarkdownParser`** strips YAML front-matter (Docusaurus / MkDocs / Obsidian convention) via regex; reduces `[text](url)` and `![alt](url)` to their human-readable portion (URLs aren't useful for retrieval); records `had_frontmatter: bool` in metadata for debugging. **`:func:`parse(path)` dispatcher** uses a module-level `_EXTENSION_MAP` so adding a new format = add one line; case-insensitive (`.PDF` works); accepts `str` or `Path`. **Typed exception hierarchy** `ParserError` (carries `.path`) → `UnsupportedFormatError`, `EmptyDocumentError`, `PdfReadError`. Updated `src/tinyrag/ingestion/__init__.py` to re-export 10 public symbols. Plus `tests/test_parsers.py` — **56 hermetic tests**: TestPublicSurface (10 — every re-export), TestProtocolIsRuntime (4 — all 3 parsers satisfy `@runtime_checkable` Protocol + a `NotAParser` class fails), TestParsedDocument (3 — frozen + `field(default_factory=list)` avoids mutable-default-shared-state), TestExtensionDispatch (11 — each format routes correctly, `.markdown` alias works, uppercase works, string path works, unknown extension raises + error message lists supported + preserves path), TestTxtParser (7 — happy path, missing file, empty, whitespace-only, BOM stripped, invalid UTF-8 propagates `UnicodeDecodeError`), TestMarkdownParser (9 — happy path, frontmatter stripped + flag set, no-frontmatter flag clear, link URLs stripped, image URLs+brackets stripped, missing/empty/frontmatter-only files), TestPdfParser (6 — happy 2-page, per-page text + numbers, missing file, empty PDF, malformed bytes → `PdfReadError`, char_count matches), TestErrorHierarchy (4 — parametrised `issubclass` check + single `except ParserError` catch), TestJsonSafety (3 — all formats' metadata is `json.dumps`-clean for the SQLite store). **Hand-built PDF fixture** in `_build_minimal_pdf`: no PDF writer dep (`fpdf`/`reportlab`/`pypdf` aren't pinned); constructs a minimal valid 2-page PDF by hand using stdlib `zlib` (FlateDecode) so pdfplumber can extract per-page text — needed because `tests/fixtures/` is otherwise empty and adding a writer dep just for tests was the wrong trade. Full suite: **323/323 tests pass** with `PYTHONPATH=.` (was 267, +56). Lint clean on new files (the 1 remaining `SIM102` error is pre-existing in `scripts/verify_llamacpp.py` from Step 3.4). No new runtime deps — `pdfplumber==0.11.4` was already pinned. **Real PDF verified via REPL**: `parse("manual.pdf")` on a 2-page hand-built PDF returned `format=pdf`, `pages=[(1, "First page body."), (2, "Second page body.")]`. |
| 4.5 | Implement the chunker | ✅ Done | `0145a56` | `feat(core): add token-based chunker (Step 4.5)` | Added `src/tinyrag/core/chunker.py` (~490 lines) — the **bridge** between parsers (text from a file) and the embedder (text → vectors). **`:class:`Chunk`** frozen dataclass with the FR-5 fields exactly as specified: `text`, `source`, `page`, `chunk_index`, `char_offset`, `token_count`. **`:class:`Chunker`** takes a `ChunkingSettings` and eagerly resolves the tiktoken encoding (so a bad `encoding: foo` in `config.yaml` fails at startup with a clean `ChunkingError`, not at first call). **Algorithm**: encode the full text once with `tiktoken.Encoding.encode(..., allowed_special="all")`; step a `[start, end)` window of size `chunk_size` through the token list with stride `chunk_size - chunk_overlap`; before emitting, **sentence-trim** back from the right edge to the nearest `[.!?]` followed by whitespace/quote (using a positive-lookahead regex `[.!?](?=[\s"']|$)` so the match consumes only the punctuation — discovered + fixed during testing when the consuming version trimmed past the period into the next word); convert the trimmed character position back to an exact token index by re-decoding one token at a time. **Last chunk** never trims (extends to end-of-text — no point in aligning with a non-existent following chunk). **Forward-progress guard**: if sentence-trim shrinks the chunk below `chunk_overlap // 2`, we keep the natural boundary; if the next stride wouldn't advance past the current end, we fall back to `start + 1` so the loop terminates (defensive against pathological inputs). **Properties** expose `encoding_name`, `chunk_size`, `chunk_overlap` for callers that want to introspect. **`count_tokens(text)`** is a thin wrapper around `tiktoken.encode` so tests can verify sizes. **`:func:`default_chunker()`** returns a `Chunker` with `ChunkingSettings()` (the config.yaml defaults: 400/50/cl100k_base) — handy for REPL probes and `scripts/ingest.py`. **Why pure functions / no I/O**: the `core` package rule (see `core/__init__.py`) — pure functions are trivially testable, swappable, and cannot accidentally talk to the network. Updated `src/tinyrag/core/__init__.py` to re-export 4 public symbols. Plus `tests/test_chunker.py` — **34 hermetic tests**: TestPublicSurface (4), TestChunkDataclass (3 — frozen + required FR-5 fields + page can be None), TestChunkerConstruction (6 — defaults 400/50/cl100k_base, custom settings honoured, **unknown encoding raises ChunkingError not ValueError**, `count_tokens` matches tiktoken, empty returns 0), TestEmptyAndShort (3 — empty/whitespace/short), TestExactBoundary (1 — text of exactly `chunk_size` tokens = 1 chunk), TestLongTextProducesMany (2 — **the roadmap's "2000 tokens → ~5 chunks" spot-check**, + chunks cover full text), TestOverlapCorrectness (2 — consecutive chunks share text, zero-overlap produces disjoint chunks), TestCharOffsetMonotonicity (2), TestChunkIndexContiguous (1), TestSentenceBoundary (2 — chunks end at `.!?` when possible, last chunk extends to end), TestPageAndSourcePassthrough (3), TestTokenCountConsistency (2 — every chunk's `token_count` matches chunker re-counting; no chunk exceeds `chunk_size`), TestIntegrationWithParsers (3 — **end-to-end**: TXT→chunker, MD→chunker, PDF→chunker with page numbers preserved per chunk — the path the real pipeline will take in Step 4.9). **Bug found + fixed during testing**: original sentence-end regex `[.!?][\s"']` consumed the trailing whitespace, so chunks ended mid-word ("Sentence 3. the"). Switched to a lookahead (`[.!?](?=[\s"']|$)`) that asserts but doesn't consume. Caught by `test_chunk_ends_at_sentence_boundary_when_possible` — exactly the kind of subtle tokenizer interaction the test suite is for. Full suite: **357/357 tests pass** with `PYTHONPATH=.` (was 323, +34). Lint clean on new files. No new runtime deps — `tiktoken==0.8.0` was already pinned. **Real 2000-token spot-check via REPL**: 2800-token text → 8 chunks, each ending at a sentence period (". Sentence N.") with `chunk_index` 0..7, `char_offset` 0 → 10200, every chunk under `chunk_size=400` tokens. |

### 11.2 Phase 4 — Build (laptop)

| Step | Description | Status | Commit SHA | Commit message | Notes |
|------|-------------|--------|------------|----------------|-------|
| 4.6 | Implement the embedder (Protocol + concrete) | ⏳ Next | — | — | Will fill `src/tinyrag/ingestion/embedder.py` with `EmbeddingModel` Protocol + `SentenceTransformerEmbedder` class. Lazy model load on first `.embed()` call, `.dimension` property asserted against config at load time, batched `.embed(texts) -> list[list[float]]` API. |

### 11.3 Phase 5 — Test (laptop)

_(To be populated as steps complete)_

### 11.4 Phase 6 — Deploy (Pi + sensors, Week 9)

_(To be populated as steps complete)_

### 11.5 Phase 7 — Report (Week 10)

_(To be populated as steps complete)_

### 11.6 Step Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ Done | Code merged, tests pass, student approved |
| ⏳ Next | Identified as the next step to start |
| ⬜ Pending | Planned but not started |
| 🔄 In progress | Currently being worked on |
| 🛑 Blocked | Stopped on a gate or risk; needs decision |
| ❌ Skipped | Intentionally skipped (with reason) |

### 11.7 Daily / Per-Step Convention

When a step is completed, append one row to the relevant phase subtable, in this format:

```markdown
| 3.2 | Set up Python venv + pinned requirements | ✅ Done | `<short SHA>` | `chore(deps): ...` | Brief outcome + any deviation |
```

---

*End of AGENT.md. Update this file whenever a major decision changes, a milestone is reached, or a step in the Build Journal completes.*
