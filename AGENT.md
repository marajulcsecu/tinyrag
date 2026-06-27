# AGENT.md — Project Context Handoff File

> **Purpose:** This file is the single source of truth for anyone (human or AI) picking up the TinyRAG project. If you are a new agent, **read this first** before doing anything. It tells you what the project is, what decisions have been made, where things live, and what to do next.

**Last updated:** 2026-06-27 (update 37)
**Project status:** Step 4.17 complete — FastAPI HTTP server with composition-root factory (`create_app(settings)` + `@asynccontextmanager` lifespan that builds every singleton — embedder, both FAISS stores, metadata, LLM, retriever, prompt builder — and stashes them on `app.state` for FastAPI `Depends(...)` providers). Public surface: `GET /healthz` + `GET /` (liveness + banner), `GET /api/status` (FR-39: ok + model_name + embedding_model + embedding_dim + chunk counts + index/DB paths + ram_mb + llama_cpp up/down + sensor_source + deployment_target), `POST /api/query` (the full RAG pipeline returning the `Answer.to_dict()` shape — `retrieve → prompt → llm → log`, with per-stage timings + token counts; empty-query short-circuit returns empty text with `log_query=True` still writing a row), `POST /api/documents` + `GET /api/documents` + `DELETE /api/documents/{id}` + `POST /api/admin/reindex` + `POST /api/admin/benchmark` (501 skeletons for Step 4.18/Phase 5). Pydantic v2 schemas in `tinyrag/api/schemas.py` (`AskRequest` with min_length=1 + bounded k_doc/k_sensor/threshold/max_tokens + extra="forbid"; `StatusResponse` with every FR-39 field + protected_namespaces=() to silence the model_name warning; `ErrorResponse` for the uniform error shape; `NotImplementedResponse` for the 501s). Global exception handlers in `tinyrag/api/errors.py` map `ValueError → 400`, Pydantic validation → 422 with per-field detail, `LLMUnavailableError → 503`, `LLMRefusedError → 502`, `MetadataError`/`VectorStoreError`/`RetrieverError`/`ConfigError → 500`, catch-all → 500 with traceback scrubbed. Dependency providers in `tinyrag/api/deps.py` read singletons from `app.state` and raise 503 on missing key (so a misconfigured app fails fast). System-info helpers in `tinyrag/api/system_info.py`: `get_ram_mb()` (tries `/proc/self/statm` first, then `resource.getrusage`, returns `None` on platforms that don't expose RSS cheaply), `get_llama_cpp_status(url)` (httpx GET to `/health`, returns `"up"`/`"down"`), `get_embedding_model_name(embedder)` (duck-types for the `model_name` attribute that's a method on `LlamaCppClient` but a property on `FakeLLMClient`). The factory `create_app(settings=None, *, llm_kind="real", embedder_kind="real", embedding_dimension=384)` is testable — tests pass a `_tiny_settings(tmp_path)` + `llm_kind="fake"` + `embedder_kind="fake"` and call `app.dependency_overrides[...]` to swap subsystems. **`app = create_app()`** at module bottom for `uvicorn tinyrag.main:app`. Plus 57 new tests in `tests/test_api.py` (13 classes: PublicSurface, SchemasValidation, SystemInfoHelpers, CreateAppLifespan, GetStatus, PostQueryHappyPath, PostQueryLogging, PostQuerySensorKeyword, PostQueryValidation, NotImplementedEndpoints, ErrorHandlers, RootAndHealthz, CreateAppTwiceIdempotent). Full suite **1108 passed, 8 skipped** (+57 new tests, 0 regressions). Manual smoke test (uvicorn on port 8765): `/healthz` → 200 `{"ok":"true"}`; `/` → 200 `{"service":"tinyrag","version":"0.4.0","api_docs":"/docs"}`; `/api/status` → 200 with the full FR-39 shape (`ok=false, model_name="models/phi-3-mini", embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=384, doc_chunk_count=0, sensor_chunk_count=180, llama_cpp_status="down"` — the llama-server isn't running locally, expected); `POST /api/query` with sensor-keyword question → 502 `{"error":"llm_failed","detail":"...Connection refused..."}` (the expected failure mode when llama-server isn't up); `POST /api/query` with empty `query` → 422 `{"error":"validation_error","detail":"body.query: String should have at least 1 character"}`; `POST /api/query` with extra field → 422 `{"error":"validation_error","detail":"body.hack: Extra inputs are not permitted"}`; `POST /api/documents` → 501 `{"error":"not_implemented","detail":"...Step 4.18..."}`; `POST /api/admin/reindex` → 501. Every error path returns the uniform `ErrorResponse` JSON shape. Lint clean (`ruff check src/tinyrag/api/ src/tinyrag/main.py tests/test_api.py` → 0 errors).
**Next milestone:** Step 4.18 — Document management HTTP endpoints: `POST /api/documents` (multipart file upload → run Step 4.9's `run_ingest` on the uploaded PDF/TXT/MD; returns the `IngestionReport` JSON), `GET /api/documents` (paginated list via `MetadataStore.list_documents`), `DELETE /api/documents/{id}` (cascade-delete via `MetadataStore.delete_document` + `vector_store.delete_by_source`). This fills in the 501 skeleton endpoints Step 4.17 left behind. The static SPA dashboard (`static/index.html` + a few vanilla JS lines) and the upload form land in Step 4.21 — Step 4.18 just adds the JSON endpoints the dashboard will hit.
**Canonical roadmap:** `docs/06_roadmap_v2.md` (the older `v1` and `laptop_v1` are historical only)
**Remote:** `https://github.com/marajulcsecu/tinyrag`
**Tip of `main`:** `843b66d` (see §11 Build Journal)
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
- ✅ **Step 4.6 complete** — embedder (Protocol + `SentenceTransformerEmbedder` + `FakeEmbedder`); parsed chunks can now be turned into dense 384-dim L2-normalised vectors ready for the FAISS index
- ✅ **Step 4.7 complete** — `MetadataStore` (SQLite wrapper); documents, chunks, and query logs are now durably persisted
- ✅ **Step 4.8 complete** — `FAISSStore` (vector store wrapper); chunks can now be persisted as dense vectors with cosine similarity search via the `IndexFlatIP` convention
- ✅ **Step 4.9 complete** — 🛑 RISK GATE cleared: `scripts/ingest.py` walks a real PDF through `parse → chunk → embed → store` end-to-end and prints an `IngestionReport`; on the Nest install guide: 40 pages → 44 chunks → 2.0–3.1 seconds total on the laptop (well under the 30 s threshold); DB has the doc + chunks, FAISS size matches chunk count
- ✅ **Step 4.10 complete** — `LLMClient` Protocol + `LlamaCppClient` (SSE streaming over llama-server's `/v1/chat/completions`); end-to-end smoke test against real `tinyllama-1.1b`: 25 tokens in 2.91 s @ 8.58 tok/s; `is_healthy()` + `model_name()` introspection methods added; `FakeLLMClient` re-implements them so the API `/health` endpoint is testable without a live server
- ✅ **Step 4.11 complete** — `PromptBuilder` (grounded 2-message prompt: system instructions + numbered context + user question); token-budget-aware with tail-trim to fit 4096 tokens; refusal prompt for zero-chunks case; citations `[1]..[N]` numbered contiguously over surviving chunks
- ✅ **Step 4.12 complete** — `Retriever` (embedder + 2 vector stores + metadata store → `RetrievalResult`); sensor-keyword routing, two-store merge with per-id `from_sensor` tracking so `used_sensor_idx` stays correct after the threshold filter, threshold filter (>= boundary), score-DESC sort, TOCTOU-safe deleted-chunk handling, typed exception hierarchy, `MetadataAccessor` Protocol + `adapt_metadata_store` adapter; full suite **782 passed, 8 skipped**
- ✅ **Step 4.13 complete** — `SensorSource` Protocol + `SimulatedCSVSource` (real, default) + `RealSerialSource` and `MQTTBrokerSource` (Phase 4 stubs that fail cleanly with `NotImplementedError` pointing at Phase 6); `SensorReading` frozen dataclass + typed `SensorSourceError` hierarchy (Config/Schema/Read → HTTP 400/500/503); strict CSV validation (header-first to give clear "missing column" errors); 67 new tests including end-to-end against the real 51,840-row synthetic CSV; full suite **849 passed, 8 skipped**
- ✅ **Step 4.14 complete** — `SensorSummarizer` (DataFrame → list[Chunk]); per-day, per-sensor text summaries matching the architecture doc's §6.4 example verbatim; numeric path (temp/humidity/energy: avg/min/max/peak/trough time) + special motion path (0 events / 1-5 verbatim list / 6+ count form); per-unit spacing (`%` tight, `C`/`kWh` spaced); re-declared sensor-type constants locally (no `core→sensors` dep); 55 new tests including 4 against the real CSV producing exactly 180 chunks; full suite **904 passed, 8 skipped**
- ✅ **Step 4.15 complete** — `scripts/ingest_sensors.py` (5-stage sensor ingest CLI: SimulatedCSVSource → SensorSummarizer → embedder → sensor FAISS index + `MetadataStore` with `doc_type='sensor_summary'`); CLI flags `--config/--db-path/--index-path/--source/--since/--embedder/--force/--json/--quiet`; `SensorIngestionReport` dataclass; idempotent re-ingest (filename + doc_type key, clears prior chunks + FAISS slots before re-add); pretty + JSON output modes; `MetadataStore.list_documents_by_filename` extension for the lookup; tz-aware/naive comparison fix in `SimulatedCSVSource.read()`; 80 new tests including regression gate (real CSV → exactly 180 chunks, FAISS size matches); full suite **984 passed, 8 skipped**
- ✅ **Step 4.16 complete** — `scripts/ask.py` end-to-end RAG query CLI (4-stage pipeline: `Retriever.retrieve` → `PromptBuilder.build` → `LLMClient.stream_chat` → `MetadataStore.log_query`); `core/answer.py` `Answer`/`Citation` frozen dataclasses with `to_dict()` JSON shape + `is_refusal` property + `build_citations`/`build_citations_from_chunks` helpers; per-stage timings + token counts surfaced in both pretty banner and JSON; CLI flags `--config/--db-path/--doc-index/--sensor-index/--llm {real,fake}/--embedder {real,fake}/--k-doc/--k-sensor/--threshold/--max-tokens/--no-log/--json/--quiet`; empty-query short-circuit; embedder-space-consistency invariant (query and chunks must share the same embedder — `--embedder fake` is the hermetic test escape hatch); 59 new tests covering happy path + sensor-keyword routing + query-log persistence + CLI subprocess; full suite **1051 passed, 8 skipped**
- ✅ **Step 4.17 complete** — FastAPI HTTP server with composition-root factory (`create_app(settings)` + `@asynccontextmanager` lifespan that builds every singleton on `app.state` for FastAPI `Depends(...)` providers); public surface `GET /healthz` + `GET /` (liveness + banner), `GET /api/status` (FR-39: ok + model_name + embedding_model + embedding_dim + chunk counts + index/DB paths + ram_mb + llama_cpp up/down + sensor_source + deployment_target), `POST /api/query` (full RAG pipeline returning the `Answer.to_dict()` shape with per-stage timings + token counts + citations; empty-query short-circuit; `log_query=False` skips the DB write), `POST /api/documents` + `GET /api/documents` + `DELETE /api/documents/{id}` + `POST /api/admin/reindex` + `POST /api/admin/benchmark` (501 skeletons for Step 4.18 / Phase 5); Pydantic v2 schemas with `extra="forbid"` (`AskRequest` with bounded k_doc/k_sensor/threshold/max_tokens + `min_length=1`, `StatusResponse` with `protected_namespaces=()`, `ErrorResponse` uniform error shape, `NotImplementedResponse` for the 501s); global exception handlers map `ValueError → 400`, Pydantic validation → 422 with per-field detail, `LLMUnavailableError → 503`, `LLMRefusedError → 502`, `MetadataError`/`VectorStoreError`/`RetrieverError`/`ConfigError → 500`, catch-all → 500 with traceback scrubbed; system-info helpers (`get_ram_mb` reads `/proc/self/statm` then `resource.getrusage`; `get_llama_cpp_status` httpx-probes `/health`; `get_embedding_model_name` duck-types for the `model_name` attr that's a method on `LlamaCppClient` but a property on `FakeLLMClient`); the factory is testable (`llm_kind="fake"` + `embedder_kind="fake"` + `app.dependency_overrides[...]` swap subsystems); `app = create_app()` for `uvicorn tinyrag.main:app`; 57 new tests in 13 classes covering happy path + validation + sensor routing + global exception mapping + 501 skeletons + idempotent `create_app` calls; full suite **1108 passed, 8 skipped**

**Immediate next step (Step 4.18 — agent action):**

Step 4.18 fills in the three 501 skeleton endpoints Step 4.17 left behind: `POST /api/documents` (multipart file upload → run Step 4.9's `run_ingest` on the uploaded PDF/TXT/MD, return the `IngestionReport` JSON), `GET /api/documents` (paginated list via `MetadataStore.list_documents(limit, offset)` with `next_offset` for cursor pagination), `DELETE /api/documents/{id}` (cascade-delete via `MetadataStore.delete_document` + `vector_store.delete_by_source`). The admin endpoints (`/api/admin/reindex`, `/api/admin/benchmark`) stay 501 — they're Phase 5 work. The Pydantic request/response models (`DocumentUploadResponse`, `DocumentListResponse`, `DocumentDeleteResponse`) follow the same pattern as Step 4.17 (`extra="forbid"`, bounded fields, uniform `ErrorResponse` shape). The upload endpoint uses `python-multipart` (already pinned via FastAPI's dependency tree) and writes the upload to a tmpdir before handing it to the ingest pipeline. After Step 4.18 the document-management slice of the API surface is complete and Step 4.21 (the static SPA dashboard) can land on top.

**Optional parallel student action — you can verify Step 4.17 yourself:**

```bash
# 1. Run the new test suite (covers the create_app factory +
#    lifespan, every dependency provider, /api/status shape,
#    /api/query 4-stage pipeline with FakeLLM + FakeEmbedder,
#    validation 422s, 501 skeletons, and every exception
#    handler). Expect 57 passed in ~8 s.
PYTHONPATH=src ~/venvs/tinyrag/bin/python -m pytest tests/test_api.py -v

# 2. Start the FastAPI server against the real config and probe
#    the meta endpoints. Pick any free port (8765 used here so
#    8000 stays available). Use --log-level warning so the
#    structured log doesn't drown the curl output.
PYTHONPATH=src ~/venvs/tinyrag/bin/python -m uvicorn \
    tinyrag.main:app --host 127.0.0.1 --port 8765 --log-level warning &
sleep 4
curl -s http://127.0.0.1:8765/healthz
# Expected: {"ok":"true"}
curl -s http://127.0.0.1:8765/api/status | python -m json.tool
# Expected: full FR-39 shape — ok=false (llama-server isn't
# running locally), model_name="models/phi-3-mini",
# embedding_model="sentence-transformers/all-MiniLM-L6-v2",
# embedding_dim=384, doc_chunk_count=0, sensor_chunk_count=180,
# ram_mb=~150, llama_cpp_status="down", sensor_source="simulated",
# deployment_target="laptop".

# 3. Try the validation paths + the 501 skeletons (both
#    demonstrate the uniform ErrorResponse JSON shape):
curl -s -X POST http://127.0.0.1:8765/api/query \
    -H 'Content-Type: application/json' \
    -d '{"query":""}'
# Expected: {"error":"validation_error","detail":"body.query: String should have at least 1 character"}
curl -s -X POST http://127.0.0.1:8765/api/documents \
    -H 'Content-Type: application/json' -d '{}'
# Expected: {"error":"not_implemented","detail":"Document management endpoints land in Step 4.18 ..."}

# Don't forget to kill the background uvicorn when you're done:
kill %1 2>/dev/null
```

**Optional parallel student action — you can verify Step 4.15 yourself:**

```bash
# 1. Run the new sensor-ingest test suite (mostly tmpdir CSVs +
#    one regression-gate test against the real 30-day CSV).
#    Expect 80 passed in ~3 s.
PYTHONPATH=src ~/venvs/tinyrag/bin/python -m pytest tests/test_ingest_sensors.py -v

# 2. Ingest the real 30-day synthetic CSV end-to-end via the new
#    CLI (matches the roadmap §4.15 "ingest the synthetic CSV
#    through scripts/ingest_sensors.py" check). Uses --embedder
#    fake so it's fast (~1 s) and doesn't load sentence-transformers.
PYTHONPATH=src ~/venvs/tinyrag/bin/python scripts/ingest_sensors.py \
    data/sensor_logs/synthetic_30d.csv \
    --embedder fake \
    --quiet
# Expected exit 0; pretty banner reports 51,840 rows read,
# 180 chunks generated, ~1 s total. The default DB path is
# data/metadata.db and the default sensor index path is
# data/vector_stores/sensors.faiss.

# 3. Verify the DB + FAISS side-effects landed correctly:
#    (NOTE: the FK column is `document_id`, not `doc_id` —
#     see src/tinyrag/storage/metadata.py SCHEMA_SQL.)
sqlite3 -header -column data/metadata.db \
    "SELECT doc_type, filename, COUNT(*) AS chunks FROM documents
     JOIN chunks ON chunks.document_id = documents.id
     WHERE doc_type = 'sensor_summary'
     GROUP BY doc_type, filename;"
# Expected: 1 row, doc_type=sensor_summary,
# filename=synthetic_30d.csv, chunks=180.
```

**Optional parallel student action — you can verify Step 4.13 yourself:**

```bash
# 1. Run the hermetic unit tests (uses tmpdir CSVs + the real
#    30-day synthetic CSV at the end). Expect 67 passed in ~1.2 s.
PYTHONPATH=src ~/venvs/tinyrag/bin/python -m pytest tests/test_sensors.py -v

# 2. Spot-check against the real Step 3.8 synthetic CSV (matches
#    the roadmap §4.13 "Total: N rows" check):
PYTHONPATH=src ~/venvs/tinyrag/bin/python -c "
from tinyrag.sensors.simulated import SimulatedCSVSource
src = SimulatedCSVSource('data/sensor_logs/synthetic_30d.csv')
df = src.read()
print(df.head())
print(f'Total: {len(df)} rows')
print(f'Sensors: {src.available_sensors()}')
"
# Expected: 51840 rows, 6 sensors in sorted order
#           (bedroom_hum, bedroom_temp, house_energy, kitchen_motion,
#            living_room_hum, living_room_temp).

# 3. See the strict validation in action — try a CSV with a bad
#    sensor_type and confirm you get a clean typed error:
PYTHONPATH=src ~/venvs/tinyrag/bin/python -c "
import tempfile, os
from tinyrag.sensors.simulated import SimulatedCSVSource
from tinyrag.sensors import SensorSourceSchemaError

with tempfile.TemporaryDirectory() as td:
    bad = os.path.join(td, 'bad.csv')
    with open(bad, 'w') as f:
        f.write('timestamp,sensor_id,sensor_type,value,unit\n')
        f.write('2026-06-24T00:00:00,mystery,voltage,5.0,V\n')
    try:
        SimulatedCSVSource(bad).read()
    except SensorSourceSchemaError as e:
        print(f'{type(e).__name__}: {e}')
        print(f'  .path = {e.path}')
"
# Expected: SensorSourceSchemaError with 'voltage' in the message
#           and .path pointing at the bad CSV.
```

**Optional parallel student action — you can verify Step 4.12 yourself:**

```bash
# 1. Run the hermetic unit tests for the retriever (no FAISS, no model, no network)
PYTHONPATH=src ~/venvs/tinyrag/bin/python -m pytest tests/test_retriever.py -v
# Expected: 68 passed in ~0.5 s

# 2. Manually exercise the full retriever → prompt builder pipeline
PYTHONPATH=src ~/venvs/tinyrag/bin/python -c "
from tinyrag.core import Retriever, PromptBuilder, ChunkRecord
from tinyrag.ingestion.embedder import FakeEmbedder

# Tiny in-memory fakes (real implementation is in tests/test_retriever.py)
class FakeVS:
    def __init__(self, hits): self._hits = hits
    def search(self, qv, k):    return self._hits[:k]
    def add(self, v, i): pass
    def size(self): return len(self._hits)
    def save(self): pass
    def load(self): pass
    def delete_by_source(self, s): return 0
    @property
    def embedding_dimension(self): return 384
    @property
    def embedding_model(self): return 'fake'

class Doc: 
    def __init__(self, fn): self.filename = fn

chunks = {
    'c1': ChunkRecord(id='c1', document_id='d1', chunk_index=5, faiss_idx=0,
        page_number=15, text='To reset your Nest thermostat to factory defaults, press and hold the ring for 10 seconds.',
        text_preview='To reset...', char_offset=0, token_count=18,
        embedding_model='fake', created_at='2026-06-24'),
    's1': ChunkRecord(id='s1', document_id='d2', chunk_index=0, faiss_idx=0,
        page_number=None, text='Living room temperature yesterday: 22.1°C average, 24.0°C peak at 3pm.',
        text_preview='Living room...', char_offset=0, token_count=14,
        embedding_model='fake', created_at='2026-06-24'),
}
class Meta:
    def __init__(self): pass
    def get_chunks_by_ids(self, ids): return [chunks[i] for i in ids if i in chunks]
    def get_document(self, did):
        return {'d1': Doc('Nest.pdf'), 'd2': Doc('sensor.md')}[did]

r = Retriever(embedder=FakeEmbedder(),
              doc_store=FakeVS([('c1', 0.82)]),
              sensor_store=FakeVS([('s1', 0.71)]),
              metadata=Meta())

# Sensor-keyword query — both stores searched
result = r.retrieve('What was the temperature yesterday?')
print(f'chunks: {len(result.chunks)}, used_sensor: {result.used_sensor_idx}, '
      f'keywords: {result.sensor_keywords_matched}, top_score: {result.top_score}')
# Expected: chunks: 2, used_sensor: True, keywords: ('temperature', 'yesterday'),
#           top_score: 0.82

# Doc-only query — sensor store NOT called
result2 = r.retrieve('How do I reset my Nest thermostat?')
print(f'chunks: {len(result2.chunks)}, used_sensor: {result2.used_sensor_idx}')
# Expected: chunks: 1, used_sensor: False

# Flow into the prompt builder (the step-4.12 → step-4.11 wiring)
prompt = PromptBuilder().build('What was the temperature yesterday?', result.chunks)
print(f'prompt_tokens: {prompt.prompt_tokens}, chunks_used: {prompt.chunks_used}')
# Expected: prompt_tokens ~150-200, chunks_used: 2
"
```

**Optional parallel student action — you can verify Step 4.11 yourself:**

```bash
# 1. Run the hermetic unit tests (no model, no network)
PYTHONPATH=src ~/venvs/tinyrag/bin/python -m pytest tests/test_prompt_builder.py -v
# Expected: 71 passed in ~0.5 s

# 2. Manually inspect a generated prompt — does it look right?
PYTHONPATH=src ~/venvs/tinyrag/bin/python -c "
from tinyrag.core import PromptBuilder, Chunk
chunks = [
    Chunk(text='To reset, press and hold the ring for 10 seconds.', source='Nest.pdf', page=15, chunk_index=7, char_offset=2800, token_count=12),
    Chunk(text='Soft reset: Settings > Reset > Soft Reset.', source='Nest.pdf', page=15, chunk_index=8, char_offset=3100, token_count=10),
]
p = PromptBuilder().build('How do I reset my Nest?', chunks)
print('=== SYSTEM ===')
print(p.system_prompt)
print('=== USER ===')
print(p.user_message)
print(f'tokens={p.prompt_tokens}, chunks_used={p.chunks_used}, dropped={p.chunks_dropped}')
"
# Expected: a clean system prompt + 2 numbered context blocks + 'Question: ...'
#           tokens around 180-220, chunks_used=2, dropped=0
```

**Optional parallel student action — you can verify Step 4.10 yourself with these commands:**

```bash
# 1. Start llama-server with tinyllama (the smallest model on disk, ~668 MB)
make run-llm     # uses models/tinyllama-1.1b.gguf by default
# OR manually:
# llama.cpp/build/bin/llama-server \
#     --model models/tinyllama-1.1b.gguf \
#     --host 127.0.0.1 --port 8080 \
#     --ctx-size 4096 --threads 10

# 2. In another shell: smoke-test the LLM client against it (~3 s)
PYTHONPATH=src ~/venvs/tinyrag/bin/python scripts/smoke_test_llm.py \
    --model tinyllama-1.1b --prompt "In one sentence, what is a smart home?"

# Expected output (abridged):
#   response: "A smart home is a home that uses advanced technology to
#              automate and monitor various aspects of daily life..."
#   prompt_tokens: 8 / completion_tokens: 25
#   duration: 2.91 s / tokens/second: 8.58
# [ OK ] tinyllama-1.1b
#
# 3. Hermetic test (no server needed) — proves the unit-test surface still holds
PYTHONPATH=src ~/venvs/tinyrag/bin/python -m pytest tests/test_llm_client.py -v
# Expected: 41 passed
sqlite3 data/metadata.db "SELECT COUNT(*) FROM chunks;"

# 4. Visual confirmation — the FAISS index file + sidecar JSON
ls -lh data/vector_store/doc.faiss*
cat data/vector_store/doc.faiss.meta.json | python -m json.tool | head -20

# 2. Quick REPL probe — the full insert → list → query-log → cascade-delete
#    round-trip, on a real (but throwaway) SQLite file. Use /tmp so it's
#    reaped on reboot; the parent dir is auto-created.
PYTHONPATH=src ~/venvs/tinyrag/bin/python -c "
import tempfile, os
from tinyrag.storage import MetadataStore

# Nested path → parent dir auto-created (no 'unable to open database file')
with tempfile.TemporaryDirectory() as td:
    db = os.path.join(td, 'nested', 'sub', 'test.db')
    store = MetadataStore(db)
    store.init_schema()
    print(f'schema version: {store.get_schema_version()}')

    # Insert a document
    doc_id = store.insert_document(
        filename='thermo.pdf', doc_type='manual',
        source_path='data/documents/thermo.pdf',
        size_bytes=1024, content_hash='abc123',
        metadata={'page_count': 12, 'author': 'TinyRAG'},
    )
    print(f'inserted doc: {doc_id}')

    # Insert 3 chunks in a single batched transaction
    store.insert_chunks([
        {'id': f'c{i}', 'document_id': doc_id, 'chunk_index': i,
         'faiss_idx': i, 'text': f'this is chunk {i} of the manual',
         'token_count': 7, 'embedding_model': 'all-MiniLM-L6-v2'}
        for i in range(3)
    ])
    store.update_document_chunk_count(doc_id, 3)
    print(f'docs: {store.count_documents()}, chunks: {store.count_chunks()}')

    # Round-trip: read back the metadata JSON
    doc = store.get_document(doc_id)
    print(f'filename: {doc.filename}, type: {doc.doc_type}, chunks: {doc.num_chunks}')

    # Log a query
    qid = store.log_query(query='how do I reset?', top1_score=0.81,
                          num_chunks=3, retrieval_ms=23, total_ms=520,
                          model='phi-3-mini')
    print(f'logged query id: {qid}')

    # Cascade delete — deletes the document AND its chunks atomically
    deleted = store.delete_document(doc_id)
    print(f'deleted {deleted} doc → {store.count_chunks()} chunks remain (FK cascade)')
"

# 3. Visual inspection in DB Browser for SQLite (the roadmap's spot-check):
#    - Open /tmp/whatever.db you wrote above
#    - Confirm 4 tables: documents, chunks, query_log, schema_version
#    - Confirm the schema_version row has version=1, description='Initial schema'
#    - Click on chunks → confirm ON DELETE CASCADE is set on document_id FK
#    - PRAGMA journal_mode should report 'wal' (concurrent reads during ingestion)
```
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
| 4.6 | Implement the embedder (Protocol + concrete) | ✅ Done | `f6eebbd` | `feat(ingestion): add EmbeddingModel Protocol + SentenceTransformerEmbedder (Step 4.6)` | Added `src/tinyrag/ingestion/embedder.py` (~390 lines) — the **text-to-vector seam** of the ingestion pipeline. Three concerns in one module: the Protocol, a real implementation, a deterministic stub. **`:class:`EmbeddingModel`** `@runtime_checkable` Protocol (matches architecture doc §6.2 verbatim): a `dimension: int` property and an `embed(texts: list[str]) -> list[list[float]]` method. The `@runtime_checkable` lets tests assert `isinstance(x, EmbeddingModel)` without inheritance — `test_embedder.py::TestProtocolIsRuntime` proves both NotAnEmbedder and MissingDimension fail. **`:class:`FakeEmbedder`** — SHA-256 digest → floats, L2-normalised, no semantic meaning (test-only). Tile-to-dim logic handles dim < 32 / dim == 32 / dim > 32 via the same modulo loop; empty input returns `[]`, empty string returns one vector, Unicode input is accepted. **`:class:`SentenceTransformerEmbedder`** — the real deal. **Lazy load**: construction is genuinely cheap (verified by `test_construction_does_not_load_model` — a bogus model id doesn't raise until `.embed()` is called; only the 250 MB `sentence-transformers` import + model-load cost is deferred, not paid at import). `.embed()` calls `st_model.encode(texts, batch_size=…, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)` and converts the numpy output back to pure-Python `list[list[float]]` (JSON-safe for the SQLite metadata store). **`.dimension`** triggers the same lazy load so callers can probe without side-effecting. **`.is_loaded`** exposes whether the model is in RAM (the FastAPI startup hook in Step 4.17 will check this to surface a "loading model…" message). **`.load()`** is an explicit warm-up so callers that want to fail-fast at startup can do so. **`:class:`EmbeddingError`** hierarchy: `EmbeddingModelNotFoundError` (network / bad HF id / missing path → map to HTTP 503) and `EmbeddingDimensionMismatchError` (model's actual dim ≠ configured dim → catches the "swapped the model but forgot to update embedding.dimension" foot-gun). Both exceptions carry `.model_name` so the API can echo it in the 503 response. Updated `src/tinyrag/ingestion/__init__.py` to re-export the 6 new symbols. Plus `tests/test_embedder.py` — **40 tests total** (32 pass hermetically, 8 skip-when-cache-empty integration tests): TestPublicSurface (6 — every re-export lands), TestProtocolIsRuntime (4 — both concretes pass `isinstance(..., EmbeddingModel)`, NotAnEmbedder + MissingDimension both fail), TestFakeEmbedder (11 — dim property, validation for ≤ 0 dim, list-of-lists shape, **pure-Python floats (not numpy)**, all requested dims work incl. 1/33/384/768, L2-normalised, deterministic, different inputs differ, empty list, empty string, Unicode), TestErrorHierarchy (5 — both subclasses inherit, single `except EmbeddingError` catches both, `model_name` preserved, default None), TestSentenceTransformerEmbedderConstruction (4 — bogus model id doesn't raise at construction, `.dimension` triggers lazy load with a bogus id so the test is hermetic, `model_name` property reflects settings, `is_loaded` starts False), TestSentenceTransformerEmbedderEmbedContract (2 — `embed([])` short-circuits without loading the model, bogus model id raises on first embed with model_name preserved), TestSentenceTransformerEmbedderReal (8 — **SKIPPED unless `models/_hf_cache/models--sentence-transformers--all-MiniLM-L6-v2/` is present**, so CI is hermetic by default; auto-enables after a manual download). The gated tests verify: dim==384, L2-norm after encode, deterministic across two model instances, **semantic similarity** (paraphrase > unrelated — the behavioural check that proves we have a real embedding model, not just FakeEmbedder), batch_size=1 vs batch_size=8 produces identical vectors, empty list, Unicode. Full suite: **385 passed, 8 skipped** with `PYTHONPATH=.` (was 357; +32 hermetic embedder tests; the 8 real-model tests skip cleanly because no model is in `models/_hf_cache/` on this machine). The 4 pre-existing failures in `test_chunker.py` and `test_download_models.py` are unrelated to this step (existed on main before this commit, verified via `git stash`). Lint clean (`ruff check src tests` reports 0 errors — auto-fixed 6 B905 `zip()` without `strict=` and 1 trailing-newline nit). No new runtime deps — `sentence-transformers==3.2.1` was already pinned from Step 3.2. **Quick REPL probe (FakeEmbedder, runs anywhere)**: 3 texts → 3 vectors of length 384, each L2-normalised; same text twice → identical vector (SHA-256 deterministic). **Real-model probe** (after `make download-llm`): paraphrase similarity (~0.6) >> unrelated similarity (~0.0) — the model actually understands "reset the thermostat" vs "factory-reset the thermostat". |
| 4.7 | Implement the metadata store (SQLite wrapper) | ✅ Done | `1808187` | `feat(storage): add MetadataStore SQLite wrapper (Step 4.7)` | Added `src/tinyrag/storage/metadata.py` (~660 lines) — the **persistence seam** for documents, chunks, and query logs. Schema is the verbatim DDL from `docs/04_database_design_v1.md` §5.2 (4 tables: `documents` / `chunks` / `query_log` / `schema_version` + 7 indexes). **`MetadataError` hierarchy**: `MetadataSchemaError` (foreign DB file — distinguished from "our schema, no rows" which returns `None`), `MetadataIntegrityError` (FK / UNIQUE / CHECK / NOT NULL violation, with `.db_path` preserved), `MetadataNotFoundError`. **Frozen dataclasses** for typed reads: `DocumentRecord`, `ChunkRecord`, `QueryLogRecord` — callers never touch raw tuples. **Full CRUD contract**: `init_schema` (idempotent, auto-creates nested parent dirs), `get_schema_version`, `insert_document` (auto-UUID + explicit UUID supported + `metadata_json` JSON serialisation + bad `doc_type` ValueError before SQL), `update_document_chunk_count` (bumps `last_modified`), `insert_chunks` (**batched in a single transaction** — atomicity invariant: all-or-none; a duplicate `(document_id, chunk_index)` rolls back the WHOLE batch; `text_preview` auto-truncated at 200 chars + explicit override), `get_document` (by id), `get_document_by_hash` (dedup signal at re-ingest — returns OLDEST match), `list_documents` (newest first, **`rowid` tiebreak** for stable ordering when same-second inserts have identical `ingested_at`), `get_chunks_by_ids` (preserves caller order, silently skips unknown ids — TOCTOU window with FAISS, dedupes repeated ids, **batches IN clauses at 500** to stay under SQLite's 999-placeholder limit), `get_chunks_by_document` (ordered by `chunk_index`), `delete_document` (cascades to chunks via `PRAGMA foreign_keys=ON` re-applied on every connection — without it the cascade silently wouldn't fire), `count_documents`, `count_chunks`, `log_query` (auto-id, all-NULL partial result supported), `get_recent_queries` (newest first, limit validation). **Per-request fresh connection per §5.4** (~1 ms to open). Every connection applies `PRAGMA journal_mode=WAL` (concurrent reads during ingestion) + `PRAGMA foreign_keys=ON` (cascade enforcement) — both per-connection, SQLite has no default mechanism. **`:memory:`` quirk handled**: SQLite's `:memory:` is per-connection, so `init_schema` on one call wouldn't persist to the next; we use a thread-local connection handle for the in-memory case so schema survives across `_connect` calls. **Context-manager** support (`__enter__` / `__exit__`) — cached connection for the `with` block's duration. **Module-level constants**: `SCHEMA_VERSION=1`, `SUPPORTED_DOC_TYPES` (frozenset), `TEXT_PREVIEW_CHARS=200`, `MAX_IN_CLAUSE_BATCH=500`. **Schema exposed as staticmethod** `_schema_sql()` so it's grep-able + REPL-inspectable. Updated `src/tinyrag/storage/__init__.py` to re-export the 10 new symbols. Plus `tests/test_metadata.py` — **87 hermetic tests**: TestPublicSurface (9 — every re-export + DocumentRecord identity from both subpackage paths), TestSchemaInit (6 — all 4 tables + 7 indexes created, schema_version row, idempotent re-runs, nested parent dirs auto-created), TestPragmas (2 — WAL + FK enforcement), TestSchemaVersion (3 — current version, empty table → None, foreign DB → MetadataSchemaError), TestInsertDocument (8 incl. parametrize — round-trip, auto-UUID, explicit UUID, metadata_json round-trip + None → NULL, bad doc_type rejected for 5 bad values, all 3 supported types accepted, duplicate hash accepted per design, IntegrityError carries db_path), TestUpdateChunkCount (4), TestInsertChunks (8 — **atomicity invariant** verified explicitly: duplicate chunk_index rolls back the WHOLE batch), TestGetDocument (5), TestListDocuments (4 — **rowid tiebreak** verified), TestGetChunksByIds (6 — **1200-id input batched correctly** across the 500 boundary), TestGetChunksByDocument (2), TestDeleteDocument (3 — **cascade verified** explicitly), TestCounters (2), TestQueryLog (6 — incrementing auto-ids, newest-first, partial NULL, full result, limit validation, limit cap), TestSqlInjectionGuard (3 — chunk text + filename + query text with `DROP TABLE`/`UPDATE pwn` attempts stored literally — parameterisation works), TestJsonMetadataRoundTrip (2), TestContextManager (2), TestInMemoryDatabase (3). **Bugs found + fixed during testing**: (a) `__exit__` originally called `close()` on a `_GeneratorContextManager` (forgot that `_connect` is `@contextmanager`-decorated, not a connection) → fixed by adding a separate `_open_connection()` for the cached context-manager path; (b) `list_documents` ordering was non-deterministic when 3 inserts happened in the same wall-clock second → fixed by adding `rowid DESC` tiebreak; (c) `get_schema_version` couldn't distinguish "no table → foreign DB" from "table exists but empty → not yet initialised" → fixed by checking `sqlite_master` first; (d) `:memory:` databases reset on every `_connect` call → fixed via thread-local connection handle. Full suite: **424 passed, 8 skipped** with `PYTHONPATH=.` (was 385; +87 metadata tests, +0 regressions; the 8 embedder integration tests still skip cleanly). The 4 pre-existing failures in `test_chunker.py` and `test_download_models.py` are unrelated to this step (existed on main before this commit, verified via `git stash` during Step 4.6). Lint clean (`ruff check src tests` reports 0 errors — auto-fixed 6 B905 `zip()` without `strict=`, 1 trailing-newline, 1 import sort, 1 DTZ005 `utcnow` → `datetime.UTC`; 1 SIM222 `... or True` tautology manually cleaned up). No new runtime deps — `sqlite3` is stdlib. **Quick REPL probe (in `$HOME/venvs/tinyrag`)**: full insert → list → query-log → cascade-delete round-trip on a nested-path throwaway DB (parent dir auto-created) → confirmed 4 tables, 3 chunks, 1 query-log, schema_version=1, cascade delete removes 3 chunks atomically. **DB Browser for SQLite spot-check** (the roadmap's "visually confirm the schema" check): open the throwaway DB → see `documents` / `chunks` / `query_log` / `schema_version`, confirm `chunks.document_id` FK has `ON DELETE CASCADE`, PRAGMA `journal_mode=wal`, `PRAGMA foreign_keys=on`. |
| 4.8 | Implement the FAISS vector store wrapper | ✅ Done | `db0d064` | `feat(storage): add FAISSStore vector store wrapper (Step 4.8)` | Added `src/tinyrag/storage/vector_store.py` (~930 lines) — the **dense-vector persistence half** of the storage layer (complementing `MetadataStore`). Owns the FAISS indices that hold every embedded chunk's vector + the int↔UUID mapping that links each FAISS slot back to a chunk row. **`VectorStore` Protocol** matches architecture doc §6.3 verbatim: `add` / `search` / `delete_by_source` / `save` / `load` / `size`. `@runtime_checkable` so `isinstance(x, VectorStore)` works for duck-typed implementations (verified in `TestProtocolIsRuntime`). **`FAISSStore`** concrete implementation: `IndexIDMap2(IndexFlatIP)` — inner product on L2-normalised vectors = cosine similarity for free (no per-search normalisation needed; the returned score IS the cosine in `[-1, 1]`). **Lazy FAISS + numpy imports** inside the methods that need them (~50 MB libfaiss stays out of unit-test import paths; verified by `TestFAISSStoreConstruction` instantiating without import). **Instance-level `threading.Lock`** — FAISS indices aren't thread-safe but FastAPI workers share one instance per index, so we serialise (matches the `LLMClient` model from Step 3.7). **Atomic save** via `.tmp` + `os.replace` for both the `.faiss` binary and the `.meta.json` sidecar — a crash mid-save never leaves a half-written file. **`IndexMeta`** frozen dataclass for the sidecar: `id_to_uuid` is the int→UUID map (serialised); `uuid_to_id` is the inverse, rebuilt at load time and never serialised (avoiding the two-maps-can-drift failure mode). JSON keys are strings (`str(k)` in `to_dict`); `from_dict` parses them back to `int`. **Typed exception hierarchy**: `VectorStoreError` (carries `.index_path` for log/503 context) → `VectorStoreDimensionMismatchError` (configured dim ≠ on-disk dim — the "swapped the embedding model but forgot to update `embedding.dimension`" foot-gun), `VectorStoreCorruptError` (FAISS can't read OR sidecar malformed/missing-key/missing-file), `VectorStoreSearchError` (wrong-dim query OR FAISS RuntimeError on search — re-raises so the API layer doesn't import faiss just to catch). **`load()` covers every failure mode**: missing index = no-op (first-run happy path, index created on first `add`), missing sidecar = `VectorStoreCorruptError`, dim drift = `VectorStoreDimensionMismatchError`, corrupt JSON = `VectorStoreCorruptError`, missing required key = `VectorStoreCorruptError`, `num_vectors` drift = silently corrected (FAISS is the source of truth). **`remove_ids()` / `delete_by_source()`** translate UUIDs → int IDs via the sidecar map; unknown UUIDs silently skipped (the TOCTOU-friendly dedup case after re-ingest); meta `num_vectors` decremented by the FAISS-returned removed-count (defensive: FAISS may return less than input). **Constants**: `INDEX_TYPE="IndexFlatIP"`, `META_VERSION="1.0"`, `DEFAULT_EMBEDDING_DIMENSION=384`, `DEFAULT_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"`. Updated `src/tinyrag/storage/__init__.py` to re-export the 11 new symbols (FAISSStore, VectorStore, IndexMeta, 4 errors, 4 constants). Plus `tests/test_vector_store.py` — **85 hermetic tests in 17 classes**: TestPublicSurface (11 — every re-export + FAISSStore identity from both subpackage paths), TestProtocolIsRuntime (4 — `@runtime_checkable` works; an arbitrary duck-type class with all 6 methods satisfies it; a class missing `save` does not; primitives don't), TestIndexMeta (5 — to_dict drops inverse map + str-keys id_to_uuid, from_dict round-trip with int keys reconstructed, missing key → empty mappings, frozen), TestFAISSStoreConstruction (9 — dim > 0 validation, index_path/meta_path derived correctly, embedding_model default + explicit, is_loaded starts False, size starts 0), TestAddSearchRoundTrip (11 — size increments, empty no-op, mismatched lengths raises, **search returns cosine-DESC ranked hits**, exact-match cosine == 1.0, orthogonal cosine == 0.0, k caps at size, all-UUID-str, all-float, scores in [-1, 1], is_loaded flips True), TestSearchEdgeCases (3 — empty index → `[]`, k=0/-1 raises ValueError), TestAddDimensionMismatch (3 — wrong-dim vector raises `VectorStoreDimensionMismatchError` carrying `index_path`, batch with one bad row raises, no partial-add corruption), TestSearchDimensionMismatch (2 — wrong-dim query raises `VectorStoreSearchError`; caught by base `VectorStoreError`), TestSaveLoadRoundTrip (5 — **the roadmap's "after save+load, same search results" check**, save with no adds, save idempotent, UUID mapping preserved across reload), TestLoadEdgeCases (6 — missing index no-op, missing sidecar `Corrupt`, dim drift `Mismatch`, corrupt JSON `Corrupt`, missing-key `Corrupt`, double-load safe), TestDeleteBySource (8 — single remove, unknown returns 0, search omits removed, k still respected, batch `remove_ids`, empty list, mixed known+unknown, `meta.num_vectors` drops), TestSidecarMetaFile (7 — valid JSON, all required keys, dim + num_vectors + id_to_uuid + INDEX_TYPE + normalize recorded), TestErrorHierarchy (7 — every error subclasses `VectorStoreError`, `index_path` preserved, optional, message preserved, base catches subclasses), TestThreadSafety (3 — concurrent adds preserve total count, concurrent add+search raises no errors, concurrent saves produce valid final state). **Bugs found + fixed during testing**: (a) ruff flagged 8 unused `# noqa: BLE001` directives after auto-fixing the broader lint pass → removed the comments; (b) `SIM118` `d["id_to_uuid"].keys()` flagged in the JSON-keys assertion → simplified to `d["id_to_uuid"]`; (c) one `I001` import sort and one `E402` "not at top" needed re-ordering (numpy after the typing-only imports) — auto-fixed. Full suite: **561 passed, 8 skipped** with `PYTHONPATH=src` (was 476 before; +85 vector-store tests, +0 regressions — the previously-broken chunker/download_models tests are now passing too, so the count jump is larger than expected). Lint clean (`ruff check src/tinyrag/storage tests/test_vector_store.py` reports 0 errors — auto-fixed 11 issues, 1 SIM118 manually simplified). No new runtime deps — `faiss-cpu==1.8.0.post1` was already pinned in `requirements.txt`. **Quick REPL probe (4-D toy vectors, runs anywhere)**: add 3 unit vectors (orthogonal + near-match), `search(v1, k=3)` returns `[('uuid-a', 1.0), ('uuid-c', ~0.9), ('uuid-b', ~0.0)]` in cosine DESC order; `save()` writes both files; reload into a fresh `FAISSStore` → identical search results; `delete_by_source('uuid-b')` removes the right vector → size drops from 3 to 2. **Real FAISS index spot-check** (post-Step-4.9, when we have a real ingest): `faiss.write_index` produces a ~50 KB binary for 100 chunks of 384-dim vectors (~500 B per 1000-vector chunk). |
| 4.9 | 🛑 RISK GATE: end-to-end ingestion pipeline | ✅ Done | `ee66403` | `feat(scripts): add end-to-end ingest pipeline + fixture downloader (Step 4.9)` | Added `scripts/ingest.py` (~750 lines) — the **end-to-end pipeline CLI** that wires every prior Phase 4 step together: `parse(path)` → `chunker.chunk(text, page=…)` → `embedder.embed(texts)` → `metadata.insert_document + insert_chunks + update_document_chunk_count` → `vector_store.add(vectors, ids) + save()`. Prints an `IngestionReport` (every stage's `duration_ms`, num_chunks, embedding_dimension, doc_id, index_size). **Per-page chunking** preserves page numbers for PDFs (the chunker accepts a `page` arg; TXT/MD get `page=None`). **SHA-256 content hash** computed from the raw file bytes → the dedup signal at re-ingest (matches the Step 4.7 `get_document_by_hash` contract). **UUID v4 per chunk**, generated here so the SAME UUID lands in both `MetadataStore.insert_chunks(...)` AND `FAISSStore.add(vectors, ids)` — the int↔UUID lock-step the architecture requires. **Pretty CLI output** (ANSI colour + banner + per-stage timings) plus `--json` for CI and `--quiet` for piping. **Per-stage try/except** converts every exception to `ok=False` with a clean error message — no Python tracebacks in normal use. **CLI flags**: `--path` (positional), `--config`, `--db-path`, `--index-path`, `--doc-type {manual,note,spec}`, `--embedder {real,fake}`, `--json`, `--quiet`. Exit codes: 0 success, 1 pipeline error, 2 bad args (matches the documented contract). Plus `scripts/download_fixtures.py` (~410 lines) — catalogs test fixtures with name/url/sha256/filename/description; **NO URL pinned** for the Nest fixture (Nest/Google don't host a stable canonical URL — the GCS path 404s as of today); the script's primary value is **SHA verification** via `--verify-only`. Exit codes 0/1/2/3 (mismatch is the dangerous one). ANSI colour output matches `scripts/download_models.py` conventions. **`tests/fixtures/.gitkeep`** + `tests/fixtures/*` gitignore rule (added in commit `45055d6`) keep the PDFs out of git but the directory visible in fresh clones. Plus `tests/test_ingest.py` (~755 lines, **49 tests**): TestPublicSurface (9), TestIngestionReportSchema (4 — required keys present, JSON-serialisable, durations rounded to 2 dp, extra dict included), TestSha256File (3 — matches hashlib on small/empty/larger files), TestChunkPages (2 — PDF preserves page numbers, TXT uses page=None), TestChunkRecords (6 — **the cross-page chunk_index renumbering invariant codified**: globally unique 0..N-1, unique UUIDs, placeholder faiss_idx=-1, text/token_count pass through), TestRunIngestSuccess (6 — happy path ok=True, chunk count > 0, DB+FAISS counts match, under time threshold, all stage durations present), TestRunIngestFailurePaths (4 — missing file, unsupported extension, empty file, no orphan doc row on failure), TestCliArgs (6 — `--json` schema, `--quiet` minimal, exit codes 0/1/2, argparse rejects bogus values). `tests/test_download_fixtures.py` (~410 lines, **23 tests**): TestPublicSurface (6), TestRegistryInvariants (5 — every entry well-formed, SHA is 64 hex chars, no duplicates, Nest fixture present), TestSha256File (3), TestVerify (3 — missing→False, correct→True, wrong-SHA→False), TestAcquireFailures (3 — missing-file-no-URL→DownloadError, wrong-SHA→ChecksumMismatchError, correct-SHA→'present'), TestResolveTargets (4 — default returns first, --all returns every, --name picks matching, unknown→UnknownFixtureError), TestResolveFixturesDir (2), TestCliExitCodes (5 — list/json exit 0, unknown exits 2, verify-only exits 0, JSON shape), TestCliRun (1 — --force with no URL exits 1). **Bugs found + fixed during testing**: (a) **the chunk_index duplicate bug — caught by the very first end-to-end run on the Nest PDF**: the Chunker resets `chunk_index` to 0 on every call, so per-page chunking produces N chunks with `chunk_index=0` (one per page). The metadata store's `UNIQUE (document_id, chunk_index)` constraint fires and rolls back the entire batch. Fixed in `_chunk_records` by globally renumbering across pages (the test `TestChunkRecords::test_chunk_index_is_globally_unique` pins this invariant — exactly the kind of cross-module integration bug the risk gate is for); (b) `tiny_settings` fixture originally used `monkeypatch.setattr` on the frozen Pydantic Settings — raised `ValidationError`. Fixed by constructing a fresh Settings with overridden sub-models (Settings is `frozen=True` so mutation isn't possible); (c) `test_empty_file_ok_false` originally asserted only "no chunks produced" — but the parser raises `EmptyDocumentError` BEFORE the chunker sees the file. Relaxed to accept either failure mode. **End-to-end REPL verification on the real Nest PDF**: `python scripts/ingest.py tests/fixtures/Nest-Thermostat-Installation-Guide-UK.pdf --embedder fake` → 40 pages parsed in ~1.6 s → 44 chunks in ~0.17 s → 384-dim fake embeddings in ~2 ms → metadata DB insert in ~0.1 s → FAISS add in ~0.3 s → save in ~2 ms → **TOTAL 2.0–3.1 s** (well under the 30 s threshold). DB has the document + 44 chunks, FAISS index size matches chunk count, `--json` output is valid and includes all required keys. Full suite: **633 passed, 8 skipped** with `PYTHONPATH=src` (was 561 before; +72 new tests, +0 regressions — the 8 embedder integration tests still skip cleanly). Lint clean (`ruff check scripts tests` reports 0 errors — auto-fixed 17 findings: 2 UP037 quoted type annotations, 1 W292 trailing newline, plus import-sort + E402 fixes after I001/E402 re-ordering). No new runtime deps. **Student verification (parallel action, documented in §8)**: 4 commands to re-run the ingest, dump the DB, and inspect the FAISS sidecar — all hermetic via `--embedder fake` (no model download).
| 4.10 | Implement the LLM client (llama.cpp wrapper) | ✅ Done | `a93ed38` | `feat(generation): add model_name() + is_healthy() helpers (Step 4.10)` | The `LLMClient` Protocol + `LlamaCppClient` SSE-streaming client + `FakeLLMClient` were already shipped from Step 3.7 (commit `ee984c0`, 31 tests at the time). For Step 4.10 I verified they satisfy the roadmap spec, **added the two introspection helpers the roadmap calls for but Step 3.7 omitted**: `model_name()` returns the configured model id (the API layer surfaces this in responses + observability logs) and `is_healthy()` probes `GET /v1/models` on llama-server with a 5-second timeout, never raising (returns `False` on any `httpx.HTTPError`). `FakeLLMClient` re-implements both methods so the API `/health` endpoint is testable without a live model (knob: `healthy: bool = True`). The `LLMClient` Protocol was extended to require the new methods — both real and fake still satisfy it. **Why these helpers matter now**: Step 4.14 (the FastAPI app) needs `is_healthy()` for the `/health` endpoint and `model_name()` for the `/ask` response payload (`which model answered?`). Without them the API would have to import llama-server's API directly, leaking the LLM seam. Plus **10 new tests** in `tests/test_llm_client.py` — TestModelName (2 — both clients), TestIsHealthy (6 — 200→True, 500→False, ConnectError→False, ReadTimeout→False, fake defaults True, fake healthy=False knob), TestProtocolConformanceWithNewMethods (2 — both still satisfy the extended Protocol). **Bug found + fixed during testing**: my initial `test_llamacpp_is_healthy_never_raises` used `raise RuntimeError(...)` inside the mock handler expecting it to be swallowed — but `RuntimeError` is NOT an `httpx.HTTPError`, so it propagated (correctly — that would mask real bugs in production). Switched to `httpx.ReadTimeout` which IS in the documented swallow-list. The test now correctly pins the contract: httpx errors are swallowed, plain runtime errors propagate. **End-to-end verification against a real llama-server**: started `llama.cpp/build/bin/llama-server --model models/tinyllama-1.1b.gguf --host 127.0.0.1 --port 8080 --ctx-size 4096 --threads 10` in the background (~10 s load time), then ran `python scripts/smoke_test_llm.py --model tinyllama-1.1b --prompt "In one sentence, what is a smart home?"` → response `"A smart home is a home that uses advanced technology to automate and monitor various aspects of daily life, such as lighting, temperature, and security."`, 8 prompt tokens / 25 completion tokens, **2.91 s @ 8.58 tok/s** (well within the smoke-test threshold). Also confirmed `LlamaCppClient.is_healthy()` returns `True` against the running server and `LlamaCppClient.model_name()` returns `'tinyllama-1.1b.gguf'`. Full suite: **643 passed, 8 skipped** with `PYTHONPATH=src` (was 633 before; +10 new tests, +0 regressions). Lint clean (`ruff check src/tinyrag/generation tests/test_llm_client.py` reports 0 errors). No new runtime deps — `httpx` was already pinned. **Student verification (parallel action, documented in §8)**: 3 commands — start `make run-llm`, run `scripts/smoke_test_llm.py`, run the unit tests hermetically.
| 4.11 | Implement the prompt builder | ✅ Done | `504fc82` | `feat(core): add grounded prompt builder (Step 4.11)` | Added `src/tinyrag/core/prompt_builder.py` (~470 lines) — the **grounded-prompt assembly** module that sits between retrieval (Step 4.12) and generation (Step 4.10). Takes a query + a list of retrieved `Chunk`s and emits a frozen `Prompt` (2-message list ready for `LLMClient.generate`). **System prompt**: short, imperative, well-engineered — *"You are a helpful assistant for a smart-home owner. Answer ONLY using information from the numbered context blocks below. If the answer is not in the context, reply exactly: I don't have enough information in the provided documents. Cite the source of every claim using the bracketed numbers, e.g. 'The thermostat resets via the menu [1]'."* The "exactly:" refusal phrase is a stable sentinel the API layer can parse to detect the low-confidence answer path. **Context block**: each chunk rendered as `[N] (source, p.X) <text>`, joined with `\n\n`, numbered contiguously over the SURVIVING chunks (empty-text chunks silently skipped without leaving gaps like `[1] [3] [4]`). **User message**: `Question: <query>` at the end (after the context block) so the model attends to context first. **Token-budget discipline**: counts tokens with tiktoken using the same encoding the chunker used (default `cl100k_base`, overridable via constructor or `from_chunking_settings()` factory). If the budget is exceeded (default 4096 minus reserved 512 = 3584 prompt budget), drops TAIL chunks to fit (preserves the similarity ranking — earlier chunks are more relevant). Empty/whitespace-text chunks are silently skipped (they'd inflate the count without contributing signal). **Every `build()` call guarantees `prompt_tokens <= max_prompt_tokens`** — pinned by `TestPromptFitsBudget`. **Zero chunks**: produces a valid 2-message prompt whose system instruction tells the model to refuse — we DON'T raise, because "no context" is a normal request the model is expected to handle. Only programming errors raise `PromptBuilderError` (empty query, negative budgets, reserved ≥ max, empty system prompt, unknown encoding). **`:class:`Prompt`** frozen dataclass — carries `messages`, `system_prompt`, `user_message`, `prompt_tokens`, `chunks_used`, `chunks_dropped`, `encoding_name` + the `used_trimming` boolean property. The diagnostics let the API layer surface *"your question used N tokens; I dropped M chunks to fit the 4096 budget"* without re-tokenising. **`:class:`PromptBuilder`** takes `encoding_name`, `max_prompt_tokens`, `reserved_for_answer_tokens`, `system_prompt` — all keyword-only, all validated at construction time. **`:func:`default_prompt_builder()`** returns a `PromptBuilder` with the documented defaults (handy for the API composition root). Updated `src/tinyrag/core/__init__.py` to re-export 9 new symbols (Prompt, PromptBuilder, PromptBuilderError, 4 constants, default factory, user-message template). Plus `tests/test_prompt_builder.py` (~700 lines, **71 tests in 15 classes**): TestPublicSurface (6 — every documented symbol importable), TestPromptDataclass (5 — frozen, used_trimming property), TestBuilderConstruction (12 — happy-path + 8 validation cases), TestCountTokens (3 — matches tiktoken, empty, encoding choice flows through), TestBuildNoChunks (8 — refusal prompt: 2 messages, system carries refusal text, no [N] markers), TestBuildOneChunk (5 — numbered [1], source+page header, text in context, question line), TestBuildMaxChunks (3 — 3/10 chunks all fit, budget respected), TestBuildVeryLongChunks (4 — tail dropped to fit, ranking preserved by unique-marker test, fits budget after trim), TestBuildEmptyTextSkipped (3 — empty/whitespace silently skipped, no gaps), TestBuildCitationFormat (4 — exact `[N] (source, p.X) <text>` shape, no-page variant, contiguous ids), TestBuildEmptyQueryRaises (3 — empty/whitespace/with-chunks), TestPromptFitsBudget (3 — the budget invariant), TestMessagesShape (4 — always exactly `[system, user]`), TestMessagesPassableToLlm (3 — **the integration glue test**: runs the prompt through `FakeLLMClient.generate()` with `response_overrides` matching on unique substrings to confirm the system prompt + query + chunk text all reach the LLM as expected), TestCustomSystemPrompt (2 — override flows through verbatim, can disable refusal), TestDiagnostics (3 — encoding/chunks_used match the actuals). **Bug found + fixed during testing**: the original `_format_chunk` used the chunk's ORIGINAL index+1 as its citation number, which left gaps when empty chunks were skipped (a chunk at original index 1 became `[2]` instead of `[1]`). Fixed by numbering over the SURVIVING chunks via `enumerate(selected)` and adding `_format_chunk_with_number(number, chunk, total)` that takes the citation number as a parameter. Pinned by `TestBuildEmptyTextSkipped::test_empty_string_chunk_skipped`. **Manual prompt inspection on a real Nest thermostat scenario**: 3 retrieved chunks (factory reset / soft reset / Wi-Fi re-setup) + question "How do I reset my Nest thermostat?" → **208 prompt tokens**, all 3 chunks fit, citation ids `[1] [2] [3]` contiguous, system prompt carries the refusal instruction. The prompt is well-structured and ready to hand to `LlamaCppClient.generate()`. Full suite: **714 passed, 8 skipped** with `PYTHONPATH=src` (was 643 before; +71 new tests, +0 regressions). Lint clean (`ruff check src/tinyrag/core tests/test_prompt_builder.py` reports 0 errors — auto-fixed 3 I001 import-sort + 1 W292 trailing newline + 1 RUF003 ambiguous unicode `−` → `-`). No new runtime deps — `tiktoken==0.8.0` was already pinned. **Student verification (parallel action, documented in §8)**: 2 commands — run the hermetic tests, run the manual inspection one-liner. |
| 4.12 | Implement the retriever | ✅ Done | `ba5100e` | `feat(core): add Retriever with two-store merge + threshold filter (Step 4.12)` | Added `src/tinyrag/core/retriever.py` (~610 lines) — the **query-side orchestrator** that turns a natural-language question into ranked `Chunk`s ready for the `PromptBuilder`. Pipeline matches architecture §10.1: detect sensor keywords → embed query → search doc index (always) → search sensor index (only on keyword match — saves a vector call on doc-only queries) → merge per-id with score-max + parallel `from_sensor` set so `used_sensor_idx` stays correct after the threshold filter drops every sensor hit → resolve ids to `ChunkRecord`s via the metadata accessor (TOCTOU-safe: missing ids silently skipped) → threshold filter (>= boundary) → score-DESC sort. **Public surface**: `Retriever` (frozen, dependency-injected via `EmbeddingModel` + `VectorStore` Protocols), `RetrievalResult` (frozen dataclass: query, chunks, scores, used_sensor_idx, sensor_keywords_matched + `top_score` / `__len__` / `__bool__` helpers), `MetadataAccessor` Protocol (`@runtime_checkable`, narrow contract the retriever actually needs: `get_chunks_by_ids` + `get_document`), `adapt_metadata_store(store)` adapter that wraps the full `MetadataStore` (~15 methods) into the narrow Protocol, typed exception hierarchy `RetrieverError` → `RetrieverEmbedError` / `RetrieverSearchError` / `RetrieverMetadataError` so callers can map component failures to HTTP codes without inspecting tracebacks. **Constants**: `DEFAULT_THRESHOLD=0.3`, `DEFAULT_K_DOC=3`, `DEFAULT_K_SENSOR=2`, `DEFAULT_SENSOR_KEYWORDS` (frozenset of natural-language data cues — "temperature", "humidity", "yesterday", "today", "this week", "last week", "kWh", "energy used", "sensor", "reading"). `_find_sensor_keywords` compiles a single case-insensitive regex per query (whole-word for single tokens, substring for multi-word phrases) — O(\|query\| + \|matched\|) regardless of vocab size. `ChunkRecord → Chunk` conversion pulls the source filename from the owning `DocumentRecord`; missing document → `RetrieverMetadataError`. Updated `src/tinyrag/core/__init__.py` to re-export 12 new symbols. Plus `tests/test_retriever.py` (~1,000 lines, **68 tests in 17 classes**): TestPublicSurface, TestRetrievalResultDataclass, TestProtocolConformance, TestRetrieverConstruction, TestRetrieveHappyPath, TestRetrieveThreshold, TestRetrieveNoSensor, TestRetrieveWithSensor, TestRetrieveSensorNoKeywords, TestRetrieveMergeAndDedupe, TestRetrieveOrderDescending, TestRetrieveEmptyIndex, TestRetrieveDeletedChunk, TestRetrieveEmptyQueryRaises, TestRetrieveBadArgsRaises, TestRetrieveErrorMapping, TestAdaptMetadataStore, TestKeywordDetection, TestUsedSensorIdxSemantics, TestIntegrationWithPromptBuilder. **Bugs found + fixed during testing**: (a) `top_score` initially returned the first element regardless of score-DESC sort — defensive `max()` added + explicit test for sorted input; (b) `used_sensor_idx` was originally set to `bool(sensor_hits)` BEFORE the threshold filter, so a sensor-triggering query with all sensor hits below threshold still reported True — fixed by tracking `from_sensor: set[str]` per id and recomputing after the filter as `bool(kept_from_sensor)`; (c) several `ruff` nits auto-fixed (`F401` unused imports, `W292` trailing newlines, `RUF100` unused `# noqa`, `UP037` quoted annotation). **Smoke test** (`/tmp/smoke_412_v2.py`, 105 lines, runs in <1 s): two queries against realistic fixtures (Nest thermostat manual + sensor summary MD). Query 1 *"How do I reset my Nest thermostat?"* → 2 doc chunks, `used_sensor=False`, sensor store not called (verified via call counter). Query 2 *"What was the temperature yesterday?"* → 3 merged chunks (1 doc + 1 sensor + 1 doc), `used_sensor=True`, `sensor_keywords_matched=('temperature', 'yesterday')`, `top_score=0.82`. Pipeline → `PromptBuilder.build()` → **196 prompt tokens, 3 chunks used, 0 dropped**, citations `[1] [2] [3]` contiguous. Full suite: **782 passed, 8 skipped** with `PYTHONPATH=src` (was 714 before; +68 new tests, +0 regressions). Lint clean. No new runtime deps — `EmbeddingModel` / `VectorStore` Protocols declared inside `retriever.py` so the seam stays self-contained. **Student verification (parallel action, documented in §8)**: 1 hermetic test command + 1 manual retriever→prompt-builder one-liner with realistic fixtures. |
| 4.13 | Implement the sensor source | ✅ Done | `074c734` | `feat(sensors): add SensorSource Protocol + SimulatedCSVSource (Step 4.13)` | Added 4 modules under `src/tinyrag/sensors/` (~1,000 lines total): **`base.py`** (the seam) — `@runtime_checkable` `SensorSource` Protocol (`read(since)` + `available_sensors()` — the exact contract from architecture §6.1), `SensorReading` frozen dataclass with construction-time validation of `sensor_type` ∈ :data:`SUPPORTED_SENSOR_TYPES` and `unit` ∈ :data:`SUPPORTED_UNITS`, typed exception hierarchy `SensorSourceError` → `ConfigError` / `SchemaError` / `ReadError` (each carrying `.path`, mapping to HTTP 400/500/503 respectively), module constants `REQUIRED_COLUMNS` + the 4 sensor types + 4 units as both string constants and exhaustive frozensets. **`simulated.py`** — `SimulatedCSVSource` (frozen dataclass: `path: Path`, optional `default_since: datetime`). `read()` does a **header-only preview read FIRST** to validate the column set before `pd.read_csv(parse_dates=…)` fires, so a typo'd header produces a clear "missing={'timestamp'}" instead of pandas' opaque "Missing column provided to 'parse_dates'". Then vectorised `isin` checks on `sensor_type` and `unit` (error messages name the offending values). `available_sensors()` does a `usecols=['sensor_id']` preview read for the cheap `/api/status` path. **`serial_dht.py`** + **`mqtt.py`** — Phase 4 stubs. Both satisfy `SensorSource` (the cheap `available_sensors` works on a laptop) and fail clearly on `read()` with `NotImplementedError` pointing at the Phase 6 roadmap step. `RealSerialSource` carries `dht_pin` + `pir_pin` and returns `["dht22_hum", "dht22_temp", "pir_motion"]`; `MQTTBrokerSource` carries `host` + `port` + `topic_prefix` + optional creds and returns `[]` (broker-dependent). Updated `src/tinyrag/sensors/__init__.py` to re-export 19 public symbols (Protocol + dataclass + 4 constants + 4 errors + 3 sources). Plus `tests/test_sensors.py` (~800 lines, **67 tests in 12 classes**): TestPublicSurface (7), TestSensorReadingDataclass (7 — incl. `from_row` against both pandas Series and a generic dict-like), TestProtocolConformance (6 — incl. plain duck-typed class satisfaction), TestSimulatedCsvSourceConstruction (5), TestSimulatedCsvSourceRead (6), TestSimulatedCsvSourceReadSince (5), TestSimulatedCsvSourceAvailableSensors (3), TestSimulatedCsvSourceErrors (6 — incl. error-message-lists-offending-value), TestRealSerialSourceStub (6), TestMQTTBrokerSourceStub (5), TestErrorHierarchy (4), TestEndToEndRealCsv (7 — **the regression gate** against the real 51,840-row Step 3.8 CSV; skips cleanly if the file isn't on disk). **Bugs found + fixed during testing**: (a) `read()` originally let `pd.read_csv(parse_dates=["timestamp"])` fire before column validation, producing the opaque "Missing column provided to 'parse_dates'" error on a typo'd header — fixed by adding a header-only preview read that validates the column set first; (b) 4 `RUF002`/`RUF003` ambiguous-unicode findings (`×` MULTIPLICATION SIGN) auto-fixed to `x`. **Smoke test (roadmap §4.13 spot-check)**: `SimulatedCSVSource('data/sensor_logs/synthetic_30d.csv').read()` returns 51,840 rows × 6 sensors in <2 s; `available_sensors()` returns `['bedroom_hum', 'bedroom_temp', 'house_energy', 'kitchen_motion', 'living_room_hum', 'living_room_temp']` (the Step 3.8 canonical roster). Full suite: **849 passed, 8 skipped** with `PYTHONPATH=src` (was 782 before; +67 new tests, 0 regressions). Lint clean (`ruff check` 0 errors after auto-fix). No new runtime deps — `pandas` was already pinned from Step 3.8. **Student verification (parallel action, documented in §8)**: 1 hermetic test command + the roadmap's "Total: N rows" spot-check + a "see the strict validation in action" one-liner that produces a clean typed error. |
| 4.14 | Implement the sensor summarizer | ✅ Done | `ac387aa` | `feat(core): add SensorSummarizer (DataFrame → text Chunks) (Step 4.14)` | Added `src/tinyrag/core/sensor_summarizer.py` (~550 lines) — the **pure-function** module that turns a sensor DataFrame (from Step 4.13) into a list of human-readable `Chunk` objects ready for the embedder + sensor FAISS index (Step 4.15 will wire them together). **Public surface**: `SensorSummarizer` (frozen dataclass with `window: Literal['daily']`, `time_fmt`, `date_fmt`, `source_label`, `encoding_name`); `SensorSummarizer.summarize(df) -> list[Chunk]` is the only public method. `SensorSummarizerError` → `SchemaError` (missing columns, unparseable timestamps) + `EmptyError` (no chunks produced). Sensor-type constants (`_NUMERIC_SENSOR_TYPES`, `_MOTION_SENSOR_TYPE`) re-declared locally to keep the `core → sensors` dep one-way (architecture rule). **Algorithm**: validate 5 columns → coerce timestamp to `datetime64[ns]` → add `_date` column → groupby `(date, sensor_id)` sorted → dispatch per group: numeric → `"On DATE, the ROOM TYPE averaged N.N UNIT, peaking at N.N UNIT at HH:MM, and reaching a minimum of N.N UNIT at HH:MM."`; motion → 0 events / 1-5 verbatim list (with English Oxford-comma joining) / 6+ count form. **Per-unit spacing helper**: `_format_value_with_unit` — `%` and `°C` render tight (`"57.7%"`, `"22.5°C"`); `C`, `kWh`, `count` get a thin space (`"20.2 C"`, `"0.3 kWh"`). Matches the architecture doc's §6.4 example exactly. **Why summarise instead of embedding raw rows**: per-row embedding embeds poorly (no time/room/statistic context), makes the index 300× larger (51,840 rows vs 180 chunks for the synthetic CSV), and gives the LLM nothing to cite. Updated `src/tinyrag/core/__init__.py` to re-export 4 new symbols. Plus `tests/test_sensor_summarizer.py` (~730 lines, **55 tests in 13 classes**): TestPublicSurface (5), TestSchemaValidation (6 — incl. extra columns tolerated, empty DF, unparseable timestamps both as strings and NaT-in-datetime64), TestNumericSummary (8 — incl. parametrised room humanization, % tight, kWh spaced, single-value group), TestMultiSensorAndDay (3), TestMotionSummary (6 — 0/1/2/3/5/6 events with English list-joining spot-checks), TestChunkShape (6), TestChunkIndexInvariant (2), TestCustomConfiguration (3), TestUnknownSensorType (1), TestRoomExtraction (1 parametrised, 6 cases), TestEndToEndRealCsv (4 — **the regression gate** against the real 30-day CSV producing exactly 180 chunks; skips if CSV absent), TestIntegrationWithPromptBuilder (2), TestErrorHierarchy (2). **Bugs found + fixed during testing** (caught by the EndToEndRealCsv run): (a) `group.iloc[values.idxmax()]` raised `IndexError: single positional indexer is out-of-bounds` because groupby preserved the parent DF's index — fixed by `group.reset_index(drop=True)` in both numeric and motion helpers; (b) initial render produced `"20.2C"` (no space) — fixed by adding a space, but then `"57.7 %"` was wrong (SI says percent is tight) — fixed by per-unit spacing helper with `_TIGHT_UNITS = {"%", "°C"}`; (c) `_ensure_timestamp` originally let NaT through when the dtype was already `datetime64[ns]` — fixed by adding a NaT-count check; (d) ruff RUF002 `×` → `x`; (e) ruff B007 (unused groupby-key unpack) — renamed `sensor_id` → `_sensor_id`. **Smoke test (roadmap §4.14 spot-check)**: `SimulatedCSVSource(...).read() → SensorSummarizer().summarize(...)` produces **180 chunks** in ~2 s; sample output matches the architecture doc's example verbatim (`"On 2026-05-24, the bedroom temperature averaged 20.2 C, peaking at 22.9 C at 16:05, and reaching a minimum of 16.7 C at 03:00."`). Full suite: **904 passed, 8 skipped** with `PYTHONPATH=src` (was 849; +55 new tests, 0 regressions). Lint clean (`ruff check` 0 errors). No new runtime deps — `pandas` + `tiktoken` were already pinned. **Student verification (parallel action, documented in §8)**: 1 hermetic test command + 1 "feed the real CSV" one-liner showing the 180 chunks + 1 prompt-builder-integration one-liner showing the chunks flow into a real grounded prompt. |
| 4.15 | Wire scripts/ingest_sensors.py | ✅ Done | `222727b` | `feat(ingest): add scripts/ingest_sensors.py — sensor ingest pipeline (Step 4.15)` | Added `scripts/ingest_sensors.py` (~750 lines) — the **end-to-end sensor ingest CLI** that mirrors `scripts/ingest.py` but for the sensor path. 5-stage pipeline: `SimulatedCSVSource.read()` (Step 4.13) → `SensorSummarizer.summarize(df)` (Step 4.14) → `embedder.embed(texts)` (Step 4.6) → `metadata.insert_document + insert_chunks` (Step 4.7) → `vector_store.add + save` (Step 4.8). `SensorIngestionReport` frozen dataclass (ok, csv, doc_id, num_rows_read, num_chunks, num_days, sensor_types, sensor_ids, since, replaced_prior, plus every stage's `duration_ms`). Per-stage try/except converts every exception to `ok=False` with a clean error message — no Python tracebacks in normal use. **Idempotent re-ingest**: queries `MetadataStore.list_documents_by_filename(filename, doc_type='sensor_summary')` (new method, ~60 lines, added in this commit), deletes the existing doc + cascade-deletes its chunks + removes the FAISS slots via `vector_store.delete_by_source(...)`, then re-adds. Pinned by `TestRunIngestSensorsIdempotency` (3 tests verifying a second run reports `replaced_prior=True`, the document count stays at 1, and the FAISS size matches the new chunk count). **CLI flags**: positional `csv`, `--config`, `--db-path`, `--index-path`, `--source {simulated,real_serial,mqtt}` (default `simulated` — the other two are Phase 4 stubs that raise `NotImplementedError` per Step 4.13), `--since` (ISO 8601 floor passed straight to `SimulatedCSVSource.read`), `--embedder {real,fake}` (default `real`), `--force` (skip the replace-confirmation), `--json` (CI mode), `--quiet`. Exit codes: 0 success, 1 pipeline error, 2 bad args (matches the documented contract). Pretty banner with green/red ANSI colours for human consumption; JSON shape mirrors `scripts/ingest.py`'s. **Drive-by fix in `scripts/ingest.py`**: `_load_settings(config_path=None)` originally called `_ls(config_path=config_path)` which the Pydantic Settings loader doesn't accept — fixed to positional `_ls(config_path)`. **Drive-by fix in `src/tinyrag/sensors/simulated.py`**: the `since` filter originally assumed the timestamp column was tz-naive (matching how pandas reads the synthetic CSV) — but a user passing a tz-aware `since=datetime.now(UTC)` would crash with `TypeError: Invalid comparison between dtype=datetime64[ns, UTC] and datetime64[ns]`. Fixed by detecting the column's tz-awareness and aligning `floor`'s tz to match (strip if naive, attach UTC if aware). Caught by the `test_since_filter_with_aware_datetime` test. **`MetadataStore.list_documents_by_filename`** — new method (~60 lines) added in `src/tinyrag/storage/metadata.py`; SQL query mirrors `list_documents` (newest first, `rowid` tiebreak) but with a `filename` filter and optional `doc_type` filter for the idempotency key. Plus `tests/test_ingest_sensors.py` (~700 lines, **80 tests in 15 classes**): TestPublicSurface (5), TestSensorIngestionReportSchema (6 — required keys, JSON serialisability, all durations rounded to 2 dp, sensor_types/sensor_ids lists, replaced_prior flag), TestRunIngestSensorsHappyPath (11 — full 5-stage pipeline, chunks count matches expected, doc + chunks in DB, FAISS size matches, all stage timings recorded), TestRunIngestSensorsSinceFilter (2 — `--since` reduces row count, `--since` after max(timestamp) yields 0 chunks), TestRunIngestSensorsIdempotency (3 — **the re-ingest invariant**: second run reports `replaced_prior=True`, only 1 document row remains, FAISS size matches new chunk count), TestRunIngestSensorsFailurePaths (5 — missing file → ConfigError, empty CSV → EmptyError, bad column → SchemaError, missing DB parent dir auto-created, no orphan doc on failure), TestCliArgs (8 — `--json` shape, `--quiet` minimal, `--embedder fake` skips the model load, exit codes 0/1/2 via subprocess), TestIntegrationWithRealCsv (1 — **the regression gate**: ingest the real 30-day CSV with `FakeEmbedder`, expect exactly 180 chunks, FAISS size == 180, idempotent re-run still produces 180). **Hermetic design**: `tiny_settings` fixture builds a full 9-section YAML (the Pydantic validator needs every section), `small_csv` fixture writes a 9-row / 8-chunk CSV to tmpdir, `empty_csv` and `bad_columns_csv` cover the failure paths. **Bugs found + fixed during testing**: (a) `settings.sensors.source` is a `Settings.SensorSource` enum, not a `str` — fixed by extracting `.value` via `hasattr(raw, "value")` guard; (b) `settings.embedding.dimension` doesn't exist on the current Pydantic schema — replaced with a hardcoded `_DEFAULT_EMBEDDING_DIMENSION = 384` constant (the same constant the FAISS store uses); (c) `settings.sensors.default_since` doesn't exist — removed; (d) `UnboundLocalError` on `settings` because `del settings` in the fake branch shadowed the outer name — removed the `del`; (e) `load_settings(config_path=...)` is a positional-only call — fixed the drive-by in `scripts/ingest.py` too; (f) 3 `RUF002`/`RUF003` `×` → `x` + 1 `W292` trailing newline auto-fixed. **End-to-end verification against the real CSV**: `PYTHONPATH=src ~/venvs/tinyrag/bin/python scripts/ingest_sensors.py data/sensor_logs/synthetic_30d.csv --embedder fake --quiet` → exit 0, 51,840 rows read in ~100 ms, 180 chunks generated in ~250 ms, embedded in ~30 ms, DB insert in ~50 ms, FAISS add in ~200 ms, save in ~5 ms, **TOTAL ~800 ms**. Pretty mode prints a coloured banner with each stage's timing; JSON mode prints the full `SensorIngestionReport` dict. A second run reports `replaced_prior=True` and completes in ~700 ms. Full suite: **984 passed, 8 skipped** with `PYTHONPATH=src` (was 904 before; +80 new tests, 0 regressions). Lint clean (`ruff check scripts tests` reports 0 errors after auto-fix). No new runtime deps. **Student verification (parallel action, documented in §8)**: 3 commands — run the new test suite, ingest the real CSV end-to-end via the CLI, query the DB to confirm the sensor_summary doc + 180 chunks landed.
| 4.16 | Wire scripts/ask.py | ✅ Done | `843b66d` | `feat(ask): add scripts/ask.py — end-to-end RAG query CLI (Step 4.16)` | Added `scripts/ask.py` (~750 lines) — the **end-to-end RAG query CLI** and the **second 🛑 RISK GATE**. 4-stage pipeline mirroring Step 4.15's stage pattern: (1) **retrieve** — `Retriever.retrieve(question)` (Step 4.12: sensor-keyword routing + two-store merge + threshold filter); (2) **prompt** — `PromptBuilder.build(question, chunks)` (Step 4.11: token-budget tail-trim, numbered citations); (3) **llm** — `LLMClient.stream_chat(prompt)` (Step 4.10: SSE over llama-server for real, `FakeLLMClient` for test/dev); (4) **log** — `MetadataStore.log_query(...)` (Step 4.7: query_log row with `used_sensor_idx`, `top_score`, durations, token counts). Per-stage `time.perf_counter()` + per-stage try/except → clean `Answer` dataclass (no Python tracebacks in normal use). `run_ask(query, settings, *, llm_kind, embedder_kind, db_path_override, doc_index_override, sensor_index_override, k_doc, k_sensor, threshold, max_tokens, log_query, default_threshold)` is the single entry point. `print_human(answer, quiet)` + `print_json(answer)` — the pretty banner shows ANSWER / SOURCES / DIAGNOSTICS blocks; the JSON shape is the same `Answer.to_dict()` the FastAPI endpoint (Step 4.17) will return. **Empty-query short-circuit**: if `query.strip() == ""`, skip stages 1-3 and return an empty-text `Answer` (still log the query with `used_sensor_idx=False` for the eval set). **Embedder/space-consistency invariant**: query and chunks MUST use the same embedder — if you build the index with `FakeEmbedder` and ask with `SentenceTransformerEmbedder`, every cosine will be near-zero and citations will be empty. The `--embedder {real,fake}` CLI flag (mirroring Step 4.15's) is the escape hatch; the tests use `embedder_kind="fake"` for both query + index. **CLI flags**: positional `query`, `--config`, `--db-path`, `--doc-index`, `--sensor-index`, `--llm {real,fake}` (default `real`), `--embedder {real,fake}` (default `real`), `--k-doc`, `--k-sensor`, `--threshold`, `--max-tokens`, `--no-log`, `--json`, `--quiet`. Exit codes: 0 success, 1 pipeline error, 2 bad args. **Plus `src/tinyrag/core/answer.py`** (~425 lines) — the **terminus** dataclass of the RAG pipeline. `Citation` frozen dataclass (number/chunk_id/source/page/score/preview + `ref` → `"[N]"` and `location` → `"source, p.X"` properties); `Answer` frozen dataclass (query/text/used_sensor_idx/top_score/model_name/citations/chunks_used/chunks_dropped/prompt_tokens/completion_tokens/total_tokens + duration_retrieve_ms/duration_prompt_ms/duration_llm_ms/duration_total_ms) with `to_dict()` (rounds floats to 2 dp) reused by `--json` / `--quiet` / Step 4.17. Helpers: `_make_preview(text, max_chars=120)` (whitespace-collapsed, word-boundary truncation with `…`), `build_citations(retrieval, chunk_ids=)` (API-layer helper — preserves the prompt builder's chunk_id mapping), `build_citations_from_chunks(chunks, scores=)` (CLI helper — leaves chunk_id empty). Re-exported 4 new symbols from `tinyrag.core.__init__`. **Plus `tests/test_ask.py`** (~1100 lines, **59 tests in 13 classes**): TestAnswerPublicSurface (6 — every symbol importable + JSON-roundtrippable), TestCitationDataclass (5 — frozen, ref/location with and without page), TestAnswerDataclass (9 — frozen, to_dict shape, is_refusal detection case/whitespace tolerant, all durations/to_dict keys), TestMakePreview (4 — whitespace collapse, word-boundary truncation, ellipsis), TestBuildCitationsFromChunks (4 — numbered 1..N, parallel lists, empty chunk_id), TestBuildCitations (2 — preserves chunk_ids, handles short lists), TestMakeEmbedder (4 — real/fake routing, unknown kind raises), TestMakeLlm (5 — real/fake routing, model_name differs, is_healthy contract), TestMakeRetriever (1 — wires both stores), TestRunAskHappyPath (9 — full 4-stage pipeline, query echoed, citations present, top_score=max, chunks_used matches, model_name set, all timings populated, query_log row written, --no-log skips DB write), TestRunAskEmptyQuery (2 — empty + whitespace queries short-circuit), TestRunAskSensorKeyword (2 — temperature query triggers sensor store, non-sensor query skips it), TestCliArgs (7 — --help, missing query → exit 2, --json shape, --quiet minimal, --no-log skips, exit 0 on success, default log_query=True writes row). All tests hermetic: `tiny_settings` fixture builds a full 9-section YAML pointing at tmpdir; `_populate_doc_index` / `_populate_sensor_index` build FAISS indices in tmpdir with `FakeEmbedder`; tests pass `embedder_kind="fake"` + `default_threshold=0.0` to keep query + chunks in the same FakeEmbedder vector space (SHA-256 cosines are not semantically meaningful, so the threshold is bypassed). **Bugs found + fixed during testing**: (a) `SentenceTransformerEmbedder.__init__` doesn't accept `model_name=` — fixed by passing `settings.embedding` (the `EmbeddingSettings` sub-model) directly; (b) `llm.model == "phi-3-mini"` failed because the default `model_path` is `"models/phi-3-mini"` — fixed test assertion to match; (c) `MetadataStore.close()` doesn't exist (SQLite uses per-request connections) — removed all `close()` calls; (d) citations were 0 even with `threshold=0.0` because FakeEmbedder's SHA-256 cosines are sometimes negative — fixed by routing the embedder through `run_ask`'s `embedder_kind` parameter so query + chunks share the same embedder; (e) the same root cause made `used_sensor_idx=False` even when sensor keywords matched — same fix; (f) CLI subprocess tests showed empty `recent` queries because the subprocess used the default config.yaml DB, not the fixture tmpdir — fixed by adding a `_cli_args` helper that passes `--db-path`, `--doc-index`, `--sensor-index` overrides; (g) 4 remaining ruff lint errors after `--fix` (B007 unused loop var, B017 `pytest.raises(Exception)`, F841 unused var, F401 unused `sqlite3` import) — all hand-fixed (`_source`, `dataclasses.FrozenInstanceError`, removed unused var, removed unused import). **End-to-end smoke test against the real sensor index**: `PYTHONPATH=src ~/venvs/tinyrag/bin/python scripts/ask.py "What was the living room temperature yesterday?" --llm fake --embedder fake --threshold 0.0 --quiet` → exit 0, JSON shape has `used_sensor_idx=true`, `top_score=0.3979`, 2 citations (sensor-summary chunks from synthetic_30d.csv), `chunks_used=2`, `chunks_dropped=0`, `prompt_tokens=188`, `completion_tokens=9`, `duration_retrieve_ms=14.1`, `duration_total_ms=264.52`. Without `--quiet`, pretty banner prints `ANSWER` block (the canned FakeLLM response) + 2 `SOURCES` lines + `DIAGNOSTICS` block with all 3 stage timings + `TOTAL ~250 ms`. **End-to-end verification**: full suite **1051 passed, 8 skipped** with `PYTHONPATH=src` (was 984; +59 new tests, 0 regressions). Lint clean (`ruff check src/tinyrag/core/answer.py scripts/ask.py tests/test_ask.py src/tinyrag/core/__init__.py` → 0 errors). No new runtime deps. **Student verification (parallel action, documented in §8)**: 3 commands — run the test suite, run the CLI against the real sensor index with `--llm fake --embedder fake`, query the `query_log` table to confirm the row landed.
| 4.17 | Wire FastAPI HTTP server + composition root | ✅ Done | (this commit) | `feat(api): add FastAPI HTTP server with composition root (Step 4.17)` | Added **9 new files** under `src/tinyrag/api/` + `src/tinyrag/main.py` + `tests/test_api.py` (~3,700 lines total). **`main.py`** (~360 lines) is the **composition root** — the only module that imports concrete classes (`SentenceTransformerEmbedder`, `FAISSStore`, `MetadataStore`, `LlamaCppClient`). Everything else imports from `tinyrag.core` + `tinyrag.generation` + the `Protocol`-typed interfaces. The `create_app(settings=None, *, llm_kind="real", embedder_kind="real", embedding_dimension=384)` factory wires a `@asynccontextmanager` lifespan that builds every singleton (embedder, both FAISS stores, metadata, LLM, retriever, prompt builder), loads the FAISS indices from disk (`suppress(FileNotFoundError)` for the first-run happy path), initialises the SQLite schema (idempotent `CREATE TABLE IF NOT EXISTS`), stashes everything on `app.state`, and saves both FAISS stores on shutdown. The factory accepts an explicit `settings` + `llm_kind` + `embedder_kind` so tests can swap subsystems without monkey-patching. `app = create_app()` at module bottom for `uvicorn tinyrag.main:app`. **Why a factory, not a module-level app**: tests need two apps in the same process (idempotence), and tests need to inject their own `_tiny_settings(tmp_path)`. **`api/schemas.py`** (~290 lines) — Pydantic v2 request/response models. `AskRequest` (query `min_length=1` + bounded `k_doc`/`k_sensor`/`threshold`/`max_tokens` + `extra="forbid"` + `log_query` knob); `StatusResponse` (every FR-39 field + `protected_namespaces=()` to silence the `model_name` shadow warning); `ErrorResponse` (uniform error shape `{error, detail}`); `NotImplementedResponse` (501 body). `AskResponse = dict[str, Any]` (the `Answer.to_dict()` shape). **`api/deps.py`** (~150 lines) — FastAPI `Depends(...)` providers reading from `app.state`; each raises 503 on missing key (so a misconfigured app fails fast at first request, not at startup with a confusing traceback). **`api/errors.py`** (~250 lines) — global exception handlers mapping `ValueError → 400`, Pydantic + `RequestValidationError` → 422 (with per-field detail joined by `"; "`), `LLMUnavailableError → 503`, `LLMRefusedError → 502`, `MetadataError`/`VectorStoreError`/`RetrieverError`/`ConfigError → 500`, catch-all → 500 with traceback scrubbed to `"internal server error"`. **`api/system_info.py`** (~170 lines) — `get_ram_mb()` tries `/proc/self/statm` then `resource.getrusage` (returns `None` if neither works); `get_llama_cpp_status(url)` httpx-probes `/health` with a 1.5 s timeout; `get_embedding_model_name(embedder)` duck-types for the `model_name` attr that's a method on `LlamaCppClient` but a property on `FakeLLMClient`. **`api/routes_query.py`** (~390 lines) — `GET /api/status` (FR-39, best-effort: every probe is wrapped so a failing subsystem produces 200 with `ok=False` rather than 500) + `POST /api/query` (full 4-stage RAG pipeline mirroring `scripts/ask.py`, returns `Answer.to_dict()` JSON with per-stage timings + token counts + `log_query` knob; empty-query short-circuit returns empty text with `log_query=True` still writing a row). **`api/routes_docs.py`** + **`api/routes_admin.py`** — 501 skeletons with `NotImplementedResponse` body for Step 4.18 / Phase 5 to fill in. **`api/__init__.py`** re-exports 11 public symbols (4 schemas + 3 routers + handlers + 2 detail constants). **Plus `tests/test_api.py`** (~880 lines, **57 tests in 13 classes**): TestPublicSurface (5), TestSchemasValidation (10 — `extra="forbid"` + bounded numerics + `min_length=1` + `StatusResponse` round-trip), TestSystemInfoHelpers (5), TestCreateAppLifespan (4 — factory returns FastAPI app, lifespan populates every `app.state` slot, FAISS load idempotent, schema init idempotent), TestGetStatus (3 — full FR-39 shape, `ok=False` when llama down, `ok=True` when llama up), TestPostQueryHappyPath (4), TestPostQueryLogging (3), TestPostQuerySensorKeyword (2), TestPostQueryValidation (5 — empty query / extra field / bad k_doc / bad threshold / bad max_tokens all return 422 with the uniform error shape), TestNotImplementedEndpoints (3), TestErrorHandlers (4 — `ValueError → 400`, `LLMUnavailableError → 503`, `LLMRefusedError → 502`, unhandled exception → 500 with traceback scrubbed), TestRootAndHealthz (3), TestCreateAppTwiceIdempotent (3), module docstring guard (1). **Hermetic design**: tests build a `_tiny_settings(tmp_path)` pointing every path at tmpdir + populate FAISS indices in tmpdir with `FakeEmbedder` + use `FakeLLMClient` so no model weights, no live llama-server, no real PDF/CSV are required. `FastAPI TestClient` triggers the lifespan via `with TestClient(app) as client:` context so end-to-end behaviour matches a real uvicorn process. **Bugs found + fixed during testing**: (a) `pydantic_core.ValidationError: ram_mb Field required` — fixed by adding `ram_mb: float \| None = None` default in `StatusResponse`; (b) `model_name` showed `<bound method LlamaCppClient.model_name of ...>` in `/api/status` because `LlamaCppClient` declares `model_name` as `def` while `FakeLLMClient` uses `@property` — fixed by duck-typing in `_safe_model_name` (call if callable, read if string); (c) `used_sensor_idx=False` even with sensor keyword because FakeEmbedder SHA-256 cosines are sometimes < 0.3 — fixed by passing `"threshold": 0.0` in test request bodies; (d) `POST /api/query` returned 500 with `retrieval_failed` (route-local try/except) instead of the global handler's `internal_server_error` — fixed test to use `app.dependency_overrides[api_deps.get_retriever] = boom` so the exception fires from a dependency provider; (e) dependency override failed with `query._request: Field required` because the override used `def boom(_request: Any)` — fixed by using `def boom(request: Request)` (must use the EXACT parameter name `request` typed as `fastapi.Request`); (f) `PydanticUndefinedAnnotation: name 'Request' is not defined` because `from __future__ import annotations` made `Request` a forward ref — fixed by importing `from fastapi import FastAPI, Request` at module top. **Manual smoke test against the real config** (uvicorn on port 8765): `/healthz` → 200 `{"ok":"true"}`; `/` → 200 banner; `/api/status` → 200 with full FR-39 shape (`ok=false, model_name="models/phi-3-mini", embedding_model="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=384, doc_chunk_count=0, sensor_chunk_count=180, ram_mb=164.7, llama_cpp_status="down"` — expected, no live llama-server); `POST /api/query` with sensor question → 502 `{"error":"llm_failed","detail":"...Connection refused..."}` (expected failure mode); `POST /api/query` with empty query → 422 validation_error; `POST /api/query` with extra field → 422 validation_error; `POST /api/documents` → 501 not_implemented; `POST /api/admin/reindex` → 501 not_implemented. Every error path returns the uniform `ErrorResponse` JSON shape. **End-to-end verification**: full suite **1108 passed, 8 skipped** with `PYTHONPATH=src` (was 1051; +57 new tests, 0 regressions). Lint clean (`ruff check src/tinyrag/api/ src/tinyrag/main.py tests/test_api.py` → 0 errors after auto-fix of 42 issues + manual fix of 9 remaining — `VectorStore` added to TYPE_CHECKING in `routes_query.py`, `×` → `x` in `system_info.py` comment, `try/except FileNotFoundError: pass` → `suppress(FileNotFoundError)` in `main.py`, multi-from `schemas` import consolidated in `test_api.py`, `isinstance(body[key], (int, float))` → `int \| float`, `# ruff: noqa: I001` added at module top to silence the now-impossible import sort). No new runtime deps — `fastapi` and `httpx` were already pinned. **Student verification (parallel action, documented in §8)**: 3 commands — run `tests/test_api.py`, start uvicorn + curl `/healthz` + `/api/status`, test the validation + 501 paths to see the uniform error shape. |

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
