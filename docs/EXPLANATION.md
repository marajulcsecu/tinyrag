# TinyRAG — System Explanation for Teacher Demo

**Project**: TinyRAG — Retrieval-Augmented Generation for Edge IoT
**Author**: Marajul (Capstone Project)
**Demo corpus**: `rag.txt` (1 chunk) + `3rd-gen-Nest-Learning-Thermostat-Install-Guide-UK.pdf` (38 chunks) + 180 synthetic sensor summaries
**Live UI**: http://127.0.0.1:8000/

---

## 1. What is this project? (30-second elevator pitch)

TinyRAG is a **small, self-contained Retrieval-Augmented Generation (RAG) system** that runs on a laptop or a Raspberry Pi 5. It lets a user ask natural-language questions about a private document collection (PDFs, manuals, sensor logs) and get back answers that are:

- **Grounded** — every sentence cites a specific chunk of a specific document with a page number.
- **Reproducible** — temperature is set to 0.0, so the same question always yields the same answer.
- **Honest** — if no relevant context is found, the system *refuses to answer* instead of hallucinating.

The "tiny" in TinyRAG has two meanings:
1. **Small models** — 3.8 B-parameter Phi-3 Mini LLM + 80 MB MiniLM embedding model (vs. typical 70 B+ chat models).
2. **Edge deployment** — designed to run on a Raspberry Pi 5 (8 GB RAM) without any cloud dependency.

---

## 2. High-Level Architecture

```
                            ┌──────────────────────────────────────────┐
                            │            USER / BROWSER                │
                            │      (http://127.0.0.1:8000)             │
                            └────────────────┬─────────────────────────┘
                                             │
                                             ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        FastAPI HTTP SERVER (uvicorn)                       │
│  • POST /api/query     — main Q&A endpoint (JSON or SSE streaming)          │
│  • GET  /api/status    — system health (model name, index sizes, RAM)       │
│  • POST /api/documents — upload a new PDF/TXT/MD                           │
│  • GET  /api/documents — list ingested documents (paginated)                │
│  • DELETE /api/documents/{id} — remove a document and its chunks            │
│  • GET  /              — built-in chat UI (HTML/JS, no React)              │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         RETRIEVAL LAYER (core/)                            │
│                                                                            │
│  Retriever.retrieve(query, k_doc=5)                                        │
│    1. Embed query          ──► MiniLM-L6-v2 (384-dim)                      │
│    2. Detect sensor intent ──► "temperature", "humidity", "kWh"...?        │
│    3. Over-fetch from FAISS ──► k_doc * 5 = 25 candidates                  │
│    4. Keyword-overlap rerank ──► boost chunks matching query terms          │
│    5. Threshold filter     ──► drop chunks with score < 0.3                │
│    6. Slice to top-k       ──► exactly k_doc chunks returned               │
│                                                                            │
│   ┌─────────────────┐    ┌─────────────────┐                              │
│   │ Doc FAISS index │    │ Sensor FAISS    │                              │
│   │ (text chunks)   │    │ index (24-h     │                              │
│   │ 39 chunks       │    │  summaries)     │                              │
│   └────────┬────────┘    └────────┬────────┘                              │
│            │                      │                                        │
│            └──────────┬───────────┘                                        │
│                       ▼                                                    │
│              ┌──────────────────┐                                          │
│              │  MetadataStore   │  (SQLite: documents, chunks, query_log)  │
│              └──────────────────┘                                          │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │  (top-k Chunk objects with text + page)
                                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    PROMPT BUILDING (core/prompt_builder)                   │
│                                                                            │
│  PromptBuilder.build(query, chunks)                                         │
│    • System prompt: "Answer ONLY from the context. If no context, refuse."  │
│    • Number each chunk as [1], [2], [3]...                                  │
│    • Greedy-pack into the 4096-token budget (reserved 512 for the answer)   │
│    • Drop chunks that don't fit (record chunks_dropped)                     │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │  (system + user messages)
                                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                  LLM GENERATION (llama-server sidecar)                     │
│                                                                            │
│   POST http://127.0.0.1:8080/v1/chat/completions                            │
│                                                                            │
│   Model: Phi-3 Mini 3.8B (Q4_K_M quantised, ~2.4 GB on disk)               │
│   Backend: llama.cpp compiled with OpenBLAS, CPU-only                       │
│   Sampling: temperature=0.0, max_tokens=512                                 │
│                                                                            │
│   Response ──► streamed back via SSE if ?stream=true, else JSON             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Three-Stage RAG Pipeline (in plain English)

### Stage 1 — RETRIEVE
*"What do we know about this question?"*

- The user's question is fed into a **sentence embedding model** (`all-MiniLM-L6-v2`) — a small neural network that turns text into a 384-dimensional vector. Semantically similar texts end up close together in this 384-D space.
- We compute the **cosine similarity** between the query vector and every chunk vector in two FAISS indices (one for documents, one for sensor summaries) and pick the top candidates.
- A **keyword-overlap reranker** boosts chunks whose text contains the distinctive words from the query (e.g. "erp" + "directive") — this compensates for the fact that MiniLM sometimes ranks a chunk that *mentions* a term above a chunk that *defines* it.
- Chunks below a similarity threshold (0.3) are dropped — the model will refuse rather than guess if nothing is close enough.

### Stage 2 — AUGMENT
*"How do we put the question + evidence into the model's context?"*

- A **system prompt** instructs the model: *"Answer ONLY from the context below. Cite as [1], [2], [3] in-line. If the answer isn't in the context, say so."*
- The surviving chunks are wrapped as numbered blocks: `[1] Nest-Install-Guide.pdf, p.26: "Energy Related Product (ErP) Directive..."`.
- A **token budget** (4096 total − 512 reserved for the answer = 3584 for the prompt) is filled greedily with the highest-ranked chunks first. Any chunk that doesn't fit is dropped, and the drop count is surfaced in the response.

### Stage 3 — GENERATE
*"What does the LLM say, given only the evidence we found?"*

- The assembled prompt is sent to **Phi-3 Mini** running on llama-server.
- Sampling is **greedy** (temperature 0.0) — the same question yields the same answer every time, which is essential for a demo and for reproducibility.
- The response streams back token-by-token via Server-Sent Events (SSE) so the UI can show a "typing" effect; the final frame includes the full answer, the model name, per-stage timings, and a list of citations the user can click to jump back to the source.

---

## 4. Key Engineering Decisions

| Decision | What we picked | Why |
|----------|----------------|-----|
| **LLM** | Phi-3 Mini 3.8B Q4_K_M (~2.4 GB) | Fits in 8 GB Pi RAM alongside OS + llama-server; punches above its weight on grounded Q&A |
| **Embedder** | `all-MiniLM-L6-v2` (80 MB, 384-dim) | Fast on CPU, well-known, good-enough semantic quality for ≤ 1000-chunk corpora |
| **Vector DB** | FAISS `IndexFlatIP` (exact cosine) | Tiny corpora don't need HNSW/PQ approximations; exact search is fast enough and avoids the recall hit |
| **Metadata store** | SQLite | Single-file, no daemon, perfect for an edge device; one DB roundtrip per query |
| **Backend** | llama.cpp (CPU, OpenBLAS) | Vulkan GPU backend isn't usable on the Pi or our laptop's Intel iGPU; OpenBLAS gives the best CPU throughput |
| **Retrieval** | Dense (FAISS) + keyword rerank + threshold | Pure dense misses "ErP directive → page 26" because the TOC chunk outranks the content chunk; keyword rerank fixes that |
| **Deployment target** | Laptop first, Raspberry Pi 5 next | Config has a `target:` switch — laptop uses simulated sensors, Pi enables real GPIO |
| **API** | FastAPI + SSE streaming | FastAPI's lifespan manager wires everything once at startup; SSE keeps the LLM-first-token latency visible to the user |

---

## 5. The Corpus (what's loaded for the demo)

| Document | Type | Chunks | What's in it |
|---|---|---|---|
| `rag.txt` | manual | 1 | A 1-page primer explaining RAG itself |
| `3rd-gen-Nest-Learning-Thermostat-Install-Guide-UK.pdf` | manual | 38 | The full Nest UK install guide (compatibility, wiring, OpenTherm, ErP, boiler types, warranty…) |
| `synthetic_30d.csv` (summarised) | sensor_summary | 180 | 30 days × 6 windows/day of synthetic temperature + humidity readings, summarised by GPT-style heuristics |

Total: **219 chunks** indexed across the two FAISS indices.

---

## 6. Endpoints Cheat-Sheet

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Built-in chat UI (no external frontend needed) |
| GET | `/admin` | Document management UI (upload, list, delete) |
| POST | `/api/query` | Ask a question. Body: `{"query": "...", "k_doc": 5, "max_tokens": 512}` |
| GET | `/api/status` | System health JSON (model name, embedding dim, doc/sensor chunk counts, RAM, LLM up/down) |
| GET | `/api/documents` | List ingested documents (paginated: `?limit=20&offset=0`) |
| POST | `/api/documents` | Upload a new PDF/TXT/MD (multipart form) |
| DELETE | `/api/documents/{id}` | Remove a document and cascade-delete its chunks + FAISS vectors |

---

## 7. Repository Layout

```
TinyRAG/
├── README.md                    ← Quick Start (bash setup.sh && bash run.sh)
├── config.yaml                  ← Single source of truth for all runtime config
├── setup.sh                     ← Idempotent one-shot bootstrap (deps + build + download)
├── run.sh                       ← Bring up llama-server + uvicorn in one terminal
├── stop.sh                      ← Tear down both, idempotent
├── Makefile                     ← Fine-grained targets (deps-system, build-llamacpp, ...)
├── src/tinyrag/
│   ├── config.py                ← Typed Settings (Pydantic) loaded from config.yaml
│   ├── main.py                  ← FastAPI app factory + lifespan
│   ├── core/                    ← Pure domain logic (no I/O)
│   │   ├── retriever.py         ← query → top-k chunks
│   │   ├── prompt_builder.py    ← assemble grounded prompt
│   │   ├── chunker.py           ← token-aware text splitter
│   │   ├── sensor_summariser.py ← windowed aggregation of CSV rows
│   │   └── answer.py            ← result dataclass + to_dict()
│   ├── ingestion/               ← PDF/TXT parsing + embedding
│   ├── storage/                 ← FAISS vector store + SQLite metadata
│   ├── models/                  ← LLM client (llama-server HTTP wrapper) + registry
│   ├── sensors/                 ← simulated / real_serial / mqtt sources
│   └── api/                     ← FastAPI routes, schemas, error handlers
├── scripts/                     ← CLI entry points (ingest, ask, summarize_sensors)
├── tests/                       ← 1351 unit tests + integration tests
├── docs/                        ← Architecture, SRS, DB design, this EXPLANATION
└── data/
    ├── documents/               ← uploaded files
    ├── vector_store/            ← doc.faiss + sensor.faiss
    ├── metadata.db              ← SQLite
    └── sensor_logs/             ← raw + synthetic CSVs
```

---

## 8. Performance Characteristics (measured)

| Metric | Value | Notes |
|--------|-------|-------|
| Cold-start (first query) | ~30 s | llama-server loading model + warming up |
| Warm query latency | 5-15 s end-to-end | retrieval ~300 ms + prompt build ~50 ms + LLM ~5-15 s |
| Index size (39 doc chunks + 180 sensor) | 219 × 384 × 4 bytes = ~330 KB in RAM | FAISS loads the whole index at startup |
| RAM (uvicorn + llama-server) | ~3 GB total | Of which llama-server = ~2.1 GB for Phi-3 Mini |
| Test suite | 1351 passed, 2 skipped | Runs in ~4 minutes on the laptop |

---

## 9. Frequently-Asked Questions (anticipated)

### "What is RAG and why does it matter?"
Retrieval-Augmented Generation is a pattern where, instead of asking a language model to answer from its trained weights alone (where it can hallucinate), we first **retrieve** relevant snippets from a trusted source and then **generate** an answer *conditioned on those snippets*. The model can still say "I don't know" if the snippets don't contain the answer — that's the whole point: grounded, citeable, reproducible answers.

### "Why not just use ChatGPT?"
Three reasons: (1) **Privacy** — the documents never leave your machine, so you can ask questions about confidential manuals; (2) **Cost** — no per-token API bill; (3) **Recency** — ChatGPT's knowledge has a training cutoff, but our system always answers from the *latest* uploaded documents.

### "Why Phi-3 Mini specifically?"
It was the best 3-4 B model on grounding benchmarks at the time we picked it — small enough to run on 8 GB of Pi RAM, large enough to follow the "answer only from context" instruction reliably. We have a registry (`src/tinyrag/models/registry.py`) so swapping to TinyLlama, Llama-3.2-3B, or Mistral-7B is a one-line config change.

### "What if the corpus is huge?"
FAISS `IndexFlatIP` is exact search — it scales as O(N) per query. For ≤ 10 K chunks that's still sub-millisecond. Beyond that we'd switch to `IndexIVFFlat` or `IndexHNSWFlat` (approximate), but our v1 is explicitly tuned for "tiny" — the README's target deployment is the Pi, not a data centre.

### "How does the system know when to refuse?"
Two gates: (1) the retriever drops chunks below the cosine threshold (0.3 default), so if nothing is similar enough, the prompt builder emits an "empty context" prompt; (2) the system prompt *explicitly instructs* the model to refuse when the context doesn't contain the answer. Both gates are tested in `tests/test_api.py`.

### "Why CPU only?"
We deliberately avoided GPU acceleration because: (a) the Pi has no usable GPU; (b) our laptop's Intel iGPU isn't supported by llama.cpp's Vulkan backend in our build; (c) OpenBLAS on CPU is fast enough — a 200-token answer takes ~5-15 s, which is fine for an interactive demo.

### "How is this an 'Edge IoT' project?"
The Phase 6 plan moves the entire stack onto a Raspberry Pi 5 (8 GB) with real GPIO sensors (DHT22 temperature/humidity + PIR motion). The config already has a `deployment.target` switch (`laptop` vs `raspberry_pi`) and the sensors source supports `real_serial`. The system can answer questions like *"Was anyone in the living room yesterday afternoon?"* by combining manual context (the sensor locations) with the 24-hour sensor summaries.

### "Why keyword rerank on top of dense retrieval?"
Pure dense retrieval missed the "ErP directive → page 26" case in our corpus: the table-of-contents page mentioned "ErP class 26" and MiniLM ranked it above the actual ErP definition on p.26. The keyword rerank adds a discrete boost for each distinctive query term that appears in the chunk (with an extra bonus if all query terms match), which is the standard "BM25-lite" trick used by many production RAG stacks.

### "What's the test coverage?"
1351 tests across 30+ files. Pure-Python unit tests cover the retriever, chunker, prompt builder, sensor summariser, and metadata store with in-memory fakes. Integration tests run the live LLM and the live FAISS indices with a tiny corpus. Two portability tests are gated behind an env var because they take ~60 s end-to-end.

---

## 10. Demo Walkthrough Script (5 minutes)

1. **Open the chat UI** at http://127.0.0.1:8000/ — show the dark, minimal interface.
2. **Ask "What is RAG?"** — watch the answer stream in. Point out the 5 numbered sources underneath; click [1] to see the source preview.
3. **Ask "What is the ErP directive?"** — show that the system correctly returns the Nest PDF p.26 (the actual definition), not p.3 (the table-of-contents that mentions it).
4. **Ask "What is OpenTherm?"** — another Nest-specific question; show p.4 + p.24 citations.
5. **Open a second tab** to http://127.0.0.1:8000/admin — show the document list (2 documents: rag.txt + Nest PDF). Click to expand a document's metadata.
6. **Open http://127.0.0.1:8000/api/status** in a third tab — show the JSON health payload (model name, embedding dim, doc/sensor chunk counts, RAM, llama.cpp status).
7. **Ask an out-of-corpus question** ("What is the capital of France?") — show that the system *refuses* with a clear "no relevant context found" message instead of hallucinating.

---

## 11. What we'd add next (Phase 5-6 roadmap)

- **Phase 5**: Larger-corpus benchmark (1k-10k chunks), FAISS IVF index, latency SLOs.
- **Phase 6**: Real Raspberry Pi 5 deployment with DHT22 + PIR sensors over GPIO; question types like *"Was anyone in the living room yesterday afternoon?"*.
- **Evaluation harness**: BLEU/ROUGE against a held-out question set; per-stage latency breakdown.
- **Auth**: API key + per-user rate limiting for a multi-tenant dashboard.
