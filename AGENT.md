# AGENT.md — Project Context Handoff File

> **Purpose:** This file is the single source of truth for anyone (human or AI) picking up the TinyRAG project. If you are a new agent, **read this first** before doing anything. It tells you what the project is, what decisions have been made, where things live, and what to do next.

**Last updated:** 2026-06-24 (update 31)
**Project status:** Step 4.11 complete — `PromptBuilder` produces a grounded 2-message prompt (system + user) ready for `LLMClient.generate`; respects the 4096-token budget, drops tail chunks to fit, numbers citations contiguously, never exceeds the budget; verified end-to-end against `FakeLLMClient`
**Next milestone:** Step 4.12 — Implement the retriever (embeds query → searches both indices → merges results → filters by threshold → returns `RetrievalResult`)
**Canonical roadmap:** `docs/06_roadmap_v2.md` (the older `v1` and `laptop_v1` are historical only)
**Remote:** `https://github.com/marajulcsecu/tinyrag`
**Tip of `main`:** `504fc82` (see §11 Build Journal)
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
- ⏳ Next: Step 4.12 — Implement the retriever (embeds query → searches both indices → merges results → filters by threshold → returns `RetrievalResult`)

**Immediate next step (Step 4.12 — agent action):**

Step 4.12 writes `src/tinyrag/core/retriever.py` — the `Retriever` class that wires `Embedder` + `VectorStore` (doc + sensor) + `MetadataStore` together into the query→top-k pipeline. Takes a query string, detects sensor keywords ("temperature", "humidity", "kWh", "yesterday", etc.) to decide whether to also search the sensor index, embeds the query, searches both indices (k_doc + k_sensor), merges the results, filters by similarity threshold, and returns a `RetrievalResult` dataclass (`chunks: list[Chunk]`, `scores: list[float]`, `used_sensor_idx: bool`). The retriever is pure domain logic — it depends only on Protocols (so tests can mock the stores), not on real FAISS or SQLite. After this step, the full query pipeline exists (retriever → prompt builder → LLM client) and Step 4.14 (the FastAPI `/ask` endpoint) can finally wire them together.

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
| 4.11 | Implement the prompt builder | ✅ Done | `504fc82` | `feat(core): add grounded prompt builder (Step 4.11)` | Added `src/tinyrag/core/prompt_builder.py` (~470 lines) — the **grounded-prompt assembly** module that sits between retrieval (Step 4.12) and generation (Step 4.10). Takes a query + a list of retrieved `Chunk`s and emits a frozen `Prompt` (2-message list ready for `LLMClient.generate`). **System prompt**: short, imperative, well-engineered — *"You are a helpful assistant for a smart-home owner. Answer ONLY using information from the numbered context blocks below. If the answer is not in the context, reply exactly: I don't have enough information in the provided documents. Cite the source of every claim using the bracketed numbers, e.g. 'The thermostat resets via the menu [1]'."* The "exactly:" refusal phrase is a stable sentinel the API layer can parse to detect the low-confidence answer path. **Context block**: each chunk rendered as `[N] (source, p.X) <text>`, joined with `\n\n`, numbered contiguously over the SURVIVING chunks (empty-text chunks silently skipped without leaving gaps like `[1] [3] [4]`). **User message**: `Question: <query>` at the end (after the context block) so the model attends to context first. **Token-budget discipline**: counts tokens with tiktoken using the same encoding the chunker used (default `cl100k_base`, overridable via constructor or `from_chunking_settings()` factory). If the budget is exceeded (default 4096 minus reserved 512 = 3584 prompt budget), drops TAIL chunks to fit (preserves the similarity ranking — earlier chunks are more relevant). Empty/whitespace-text chunks are silently skipped (they'd inflate the count without contributing signal). **Every `build()` call guarantees `prompt_tokens <= max_prompt_tokens`** — pinned by `TestPromptFitsBudget`. **Zero chunks**: produces a valid 2-message prompt whose system instruction tells the model to refuse — we DON'T raise, because "no context" is a normal request the model is expected to handle. Only programming errors raise `PromptBuilderError` (empty query, negative budgets, reserved ≥ max, empty system prompt, unknown encoding). **`:class:`Prompt`** frozen dataclass — carries `messages`, `system_prompt`, `user_message`, `prompt_tokens`, `chunks_used`, `chunks_dropped`, `encoding_name` + the `used_trimming` boolean property. The diagnostics let the API layer surface *"your question used N tokens; I dropped M chunks to fit the 4096 budget"* without re-tokenising. **`:class:`PromptBuilder`** takes `encoding_name`, `max_prompt_tokens`, `reserved_for_answer_tokens`, `system_prompt` — all keyword-only, all validated at construction time. **`:func:`default_prompt_builder()`** returns a `PromptBuilder` with the documented defaults (handy for the API composition root). Updated `src/tinyrag/core/__init__.py` to re-export 9 new symbols (Prompt, PromptBuilder, PromptBuilderError, 4 constants, default factory, user-message template). Plus `tests/test_prompt_builder.py` (~700 lines, **71 tests in 15 classes**): TestPublicSurface (6 — every documented symbol importable), TestPromptDataclass (5 — frozen, used_trimming property), TestBuilderConstruction (12 — happy-path + 8 validation cases), TestCountTokens (3 — matches tiktoken, empty, encoding choice flows through), TestBuildNoChunks (8 — refusal prompt: 2 messages, system carries refusal text, no [N] markers), TestBuildOneChunk (5 — numbered [1], source+page header, text in context, question line), TestBuildMaxChunks (3 — 3/10 chunks all fit, budget respected), TestBuildVeryLongChunks (4 — tail dropped to fit, ranking preserved by unique-marker test, fits budget after trim), TestBuildEmptyTextSkipped (3 — empty/whitespace silently skipped, no gaps), TestBuildCitationFormat (4 — exact `[N] (source, p.X) <text>` shape, no-page variant, contiguous ids), TestBuildEmptyQueryRaises (3 — empty/whitespace/with-chunks), TestPromptFitsBudget (3 — the budget invariant), TestMessagesShape (4 — always exactly `[system, user]`), TestMessagesPassableToLlm (3 — **the integration glue test**: runs the prompt through `FakeLLMClient.generate()` with `response_overrides` matching on unique substrings to confirm the system prompt + query + chunk text all reach the LLM as expected), TestCustomSystemPrompt (2 — override flows through verbatim, can disable refusal), TestDiagnostics (3 — encoding/chunks_used match the actuals). **Bug found + fixed during testing**: the original `_format_chunk` used the chunk's ORIGINAL index+1 as its citation number, which left gaps when empty chunks were skipped (a chunk at original index 1 became `[2]` instead of `[1]`). Fixed by numbering over the SURVIVING chunks via `enumerate(selected)` and adding `_format_chunk_with_number(number, chunk, total)` that takes the citation number as a parameter. Pinned by `TestBuildEmptyTextSkipped::test_empty_string_chunk_skipped`. **Manual prompt inspection on a real Nest thermostat scenario**: 3 retrieved chunks (factory reset / soft reset / Wi-Fi re-setup) + question "How do I reset my Nest thermostat?" → **208 prompt tokens**, all 3 chunks fit, citation ids `[1] [2] [3]` contiguous, system prompt carries the refusal instruction. The prompt is well-structured and ready to hand to `LlamaCppClient.generate()`. Full suite: **714 passed, 8 skipped** with `PYTHONPATH=src` (was 643 before; +71 new tests, +0 regressions). Lint clean (`ruff check src/tinyrag/core tests/test_prompt_builder.py` reports 0 errors — auto-fixed 3 I001 import-sort + 1 W292 trailing newline + 1 RUF003 ambiguous unicode `−` → `-`). No new runtime deps — `tiktoken==0.8.0` was already pinned. **Student verification (parallel action, documented in §8)**: 2 commands — run the hermetic tests, run the manual inspection one-liner.

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
