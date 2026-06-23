# AGENT.md — Project Context Handoff File

> **Purpose:** This file is the single source of truth for anyone (human or AI) picking up the TinyRAG project. If you are a new agent, **read this first** before doing anything. It tells you what the project is, what decisions have been made, where things live, and what to do next.

**Last updated:** 2026-06-23 (update 16)
**Project status:** Step 3.8 complete — 30-day synthetic sensor dataset generated
**Next milestone:** Step 3.9 — Phase 3 end-to-end smoke test
**Canonical roadmap:** `docs/06_roadmap_v2.md` (the older `v1` and `laptop_v1` are historical only)
**Remote:** `https://github.com/marajulcsecu/tinyrag`
**Tip of `main`:** (see §11 Build Journal — pending this commit)
**Venv location:** `~/venvs/tinyrag` (symlinked as `.venv` in project root)
**OpenBLAS version:** 0.3.26 (verified via pkg-config)
**llama.cpp:** tag `gguf-v0.19.0` (commit `a290ce626663dae1d54f70bce3ca6d8f67aab62f`) — built at `/tmp/llamacpp-build/build/` (colon-path workaround; symlinked into `llama.cpp/build/`)
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

## 6. Project Structure (planned)

```
TinyRAG/
├── AGENT.md                      ← this file
├── README.md                     ← quick start
├── LICENSE
├── config.yaml                   ← single source of runtime config
├── setup.sh                      ← one-command install
├── run.sh                        ← one-command start
├── pyproject.toml                ← Python packaging
├── requirements.txt              ← pinned deps
├── .gitignore
│
├── docs/                         ← all planning docs
│   ├── 00_high_level_plan.md     ← journey map
│   ├── 01_project_scope_v2.md    ← refined scope
│   ├── 02_srs_v1.md              ← (next to write)
│   ├── 03_architecture_v1.md     ← (after SRS)
│   ├── 04_database_design_v1.md  ← (after architecture)
│   ├── 05_tech_stack_v1.md       ← (after database)
│   ├── 06_roadmap_v1.md          ← (after tech stack)
│   ├── evaluation/               ← evaluation methodology, gold-set
│   └── laptop_fallback/          ← laptop-specific notes
│
├── src/                          ← all source code
│   └── tinyrag/
│       ├── __init__.py
│       ├── main.py               ← FastAPI app entry
│       ├── config.py             ← loads config.yaml
│       │
│       ├── ingestion/            ← doc → chunks → embeddings → vector store
│       │   ├── parsers.py        ← PDF / TXT / MD
│       │   ├── chunker.py
│       │   ├── embedder.py
│       │   └── pipeline.py
│       │
│       ├── retrieval/            ← query → top-k chunks
│       │   ├── retriever.py
│       │   └── reranker.py       ← (optional, future)
│       │
│       ├── generation/           ← prompt + retrieved → answer
│       │   ├── prompt_builder.py
│       │   ├── llm_client.py     ← talks to llama.cpp
│       │   └── answer.py
│       │
│       ├── sensors/              ← abstract + concrete sources
│       │   ├── base.py           ← SensorSource protocol
│       │   ├── simulated.py
│       │   ├── serial_dht.py     ← DHT22 over GPIO
│       │   └── mqtt.py
│       │
│       ├── storage/              ← vector store + metadata DB
│       │   ├── vector_store.py
│       │   └── metadata.py       ← SQLite for chunk metadata
│       │
│       ├── api/                  ← FastAPI routes
│       │   ├── routes_query.py
│       │   ├── routes_docs.py
│       │   └── routes_admin.py
│       │
│       └── ui/                   ← static web assets
│           ├── static/
│           └── templates/
│
├── data/                         ← runtime data (gitignored)
│   ├── documents/                ← uploaded PDFs/MD
│   ├── sensor_logs/              ← CSV/JSON sensor data
│   ├── vector_store/             ← FAISS index files
│   └── metadata.db               ← SQLite
│
├── models/                       ← downloaded GGUF models (gitignored)
│
├── tests/                        ← pytest unit tests
│   ├── test_chunker.py
│   ├── test_retriever.py
│   ├── test_prompt_builder.py
│   └── test_parsers.py
│
├── scripts/                      ← operational scripts
│   ├── download_models.py
│   ├── ingest.py                 ← CLI: ingest a document
│   ├── evaluate.py               ← run the 20-Q eval set
│   └── benchmark.py              ← measure latency, RAM
│
└── reports/                      ← generated benchmarks, eval results
    ├── latency.csv
    ├── ram_usage.csv
    ├── accuracy_per_model.csv
    └── final_report.pdf          ← capstone report
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
- ✅ **Steps 3.1–3.8 complete** — repo, env, system deps, llama.cpp, all 4 GGUF models, LLMClient + smoke test, and 30-day synthetic sensor data
- ⏳ Next: Step 3.9 — Phase 3 end-to-end smoke test

**Immediate next step (Step 3.9 — agent action):**

Step 3.9 is a one-script checkpoint that exercises the entire Phase 3 stack end-to-end: starts llama-server, sends a hard-coded query, prints the response, and exits cleanly. No student action required.

**Optional parallel student action — none for Step 3.9. You can smoke-test the 4 LLMs any time you want to (already done for Phi-3 + TinyLlama; Llama 3.2 + Mistral are queued if you want to round it out):**

```bash
# In terminal 1:
make run-llm LLM_MODEL=llama-3.2-3b
# In terminal 2:
make smoke-llm LLM_MODEL=llama-3.2-3b

# Same for mistral (slower — ~2 tok/s on the laptop):
make run-llm LLM_MODEL=mistral-7b
make smoke-llm LLM_MODEL=mistral-7b
```
   # In another terminal:
   make smoke-llm
   ```
3. If `make smoke-llm` returns "All 1 model(s) passed", the LLM seam is verified end-to-end.

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
| 3.5 | Download Phi-3 Mini 3.8B GGUF | ✅ Done | `cf796b9` | `feat(models): add GGUF downloader with SHA-256 verification (Step 3.5)` | Added `src/tinyrag/models/{registry,downloader}.py` (canonical 4-model catalog: Phi-3 primary, TinyLlama/Llama 3.2/Mistral for eval), `scripts/download_models.py` (CLI with --list, --model, --all, --verify-only, --force, --json), `docs/MODELS.md` (human-readable catalog), 15 hermetic pytest tests (registry shape, idempotency, checksum rejection, HTTP Range resume, progress callbacks, CLI). Uses stdlib `urllib` (no new dep). Standardised on `models/<id>.gguf` on-disk naming. **Model file itself is NOT yet on disk** — student runs `make download-llm` to fetch ~2.3 GB Phi-3 in Step 3.6. |
| 3.6 | 🛑 RISK GATE: First llama.cpp server run on laptop | ✅ Done | `ee984c0` | `feat(llm): add LLMClient Protocol + LlamaCppClient + smoke test (Step 3.7 — note: see 3.7 below for numbering correction)` | Student action completed: `make download-llm` (2.3 GB Phi-3 fetched, SHA-256 verified against registry) → `make run-llm` → `curl http://127.0.0.1:8080/v1/models` returned HTTP 200 with the expected model metadata. Confirms the entire native + model stack is wired end-to-end. **Numbering note:** the commit subject says "Step 3.7" because at the time I conflated the LLM seam + smoke test under one commit. The actual roadmap ordering is 3.6 = first server run, 3.7 = download comparison models (next row), 3.8 = synthetic sensors (this commit). |
| 3.7 | Download comparison models (TinyLlama, Llama 3.2 3B) | ✅ Done | `ee984c0` (+ 3 fix commits: `098d438`, `412e7f3`, `51e9f6e`) | same as 3.6 (LLMClient commit) | Student action completed: downloaded tinyllama-1.1b (637 MB) and llama-3.2-3b (1.88 GB) via `scripts/download_models.py`. **Mistral 7B fix in `412e7f3`:** original TheBloke repo returned 401; switched to bartowski mirror and re-verified (4.37 GB public mirror, HTTP 200). **Truncation fix in `51e9f6e`:** Llama 3.2 first download silently stopped at 753 MB of the expected 1.88 GB and the manifest recorded a "valid" SHA for the truncated bytes; llama-server later failed with `tensor 'blk.15.ffn_up.weight' data is not within the file bounds`. Fixed in `_fetch` (short-read guard vs Content-Length) and `download` (registry `expected_size_bytes` cross-check, 5% tolerance). 3 new tests in `TestTruncationGuard`. Student re-downloaded Llama 3.2 — 1.88 GB clean. All 4 models (`phi-3-mini`, `tinyllama-1.1b`, `llama-3.2-3b`, `mistral-7b`) verified end-to-end. |
| 3.7a | LLMClient Protocol + LlamaCppClient + smoke test | ✅ Done | `ee984c0` | `feat(llm): add LLMClient Protocol + LlamaCppClient + smoke test (Step 3.7)` | Added `src/tinyrag/generation/{__init__,llm_client}.py` (~430 lines): `LLMClient` `@runtime_checkable` Protocol, `FakeLLMClient` deterministic stub (for tests / offline dev), `LlamaCppClient` real HTTP/SSE client (talks to llama-server's `/v1/chat/completions` with stream=true, parses Server-Sent Events, extracts `choices[].delta.content`, terminates on `[DONE]`, captures `usage` block, falls back to whitespace-split token estimation when usage is missing). Typed exception hierarchy: `LLMError` → `LLMUnavailableError` (5xx, connection, timeout) / `LLMRefusedError` (4xx). Lazy httpx.Client ownership. Plus `scripts/smoke_test_llm.py` (CLI: `--model`, `--all`, `--base-url`, `--prompt`, `--max-tokens`, `--models-dir`, `--json`) and `tests/test_llm_client.py` — **31 hermetic tests** using `httpx.MockTransport` covering: ChatMessage shape, Protocol duck-typing (no inheritance), FakeLLMClient canned responses + overrides + raise_after_tokens, LlamaCppClient SSE parsing (concatenation, [DONE] termination, malformed lines, role-only chunks), 5xx/4xx/connection error mapping, lazy client ownership, multi-message (system+user) roundtrip. New Makefile targets: `smoke-llm`, `smoke-llm-all`. **This is technically an "extra" step that doesn't appear in the roadmap by name** — the roadmap's Phase 3 only requires the LLM to be downloadable + runnable, but writing the LLMClient Protocol now means Phase 4 (FastAPI) can start straight away. Documented here so future contributors know where the LLM seam lives. |
| 3.8 | Generate synthetic sensor data | ✅ Done | _(short SHA pending push)_ | `feat(sensors): add 30-day synthetic sensor generator (Step 3.8)` | Added `scripts/generate_synthetic_sensors.py` (~480 lines): numpy + pandas, SEED=42 reproducibility, 5-min resolution, 6 sensors (living_room_temp, living_room_hum, bedroom_temp, bedroom_hum, kitchen_motion, house_energy), long-format CSV output to `data/sensor_logs/synthetic_30d.csv` (gitignored). Per-sensor physics: temperature = daily sinusoid + per-room offset + Gaussian noise; humidity = weakly anti-correlated with temp, bounded [30, 80]; motion = Bernoulli with hour-of-day + weekday/weekend rates; energy = base draw + morning/evening peaks + weekend multiplier + 5% appliance surges. CLI: `--start`, `--days`, `--interval-min`, `--out`, `--seed`, `--summary`, `--json`. Generated 51,840 rows × 6 sensors (30 days × 288 ticks/day). Plus `tests/test_generate_synthetic_sensors.py` — **34 hermetic tests** covering: schema conformance (§6.1 columns + dtypes + canonical sensors), no NaN, realistic value ranges (temp 15-30, humidity 30-80, motion 0/1, energy ≥ 0), daily patterns (afternoon temp peak, dinner motion peak), SEED=42 reproducibility (same/different seed → same/different output), summary helper, time-grid correctness (5-min spacing, no duplicates per sensor), custom start date. Full suite: **115/115 tests pass** (was 81, added 34). No new runtime deps — `pandas` + `numpy` were already pinned. |
| 3.9 | Phase 3 checkpoint: end-to-end smoke test | ⏳ Next | — | — | One-script checkpoint that exercises the entire Phase 3 stack end-to-end (config + llama-server + query + response). |
| 4.1 | Initialize the project skeleton (folders only) | ⬜ Pending | — | — | Per docs/06_roadmap_v2.md Phase 4. |
| 4.2 | Set up `config.yaml` + `Settings` loader | ⬜ Pending | — | — | Per docs/06_roadmap_v2.md Phase 4. |

### 11.2 Phase 4 — Build (laptop)

_(To be populated as steps complete)_

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
