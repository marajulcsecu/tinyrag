# TinyRAG — Architecture & Module Design v1

**Project Title:** TinyRAG — A Lightweight, On-Device Retrieval-Augmented Generation Assistant for Smart Home IoT
**Document version:** 1.0
**Date:** 2026-06-23
**Status:** Draft — awaiting student review
**Source of truth:** `docs/01_project_scope_v2.md`, `docs/02_srs_v1.md`

---

## 1. Purpose of This Document

This document describes the **internal architecture** of TinyRAG: how the system is decomposed into modules, how those modules interact, and which technologies implement each one. It is the bridge between the **what** (SRS) and the **how** (code).

After reading this document, the student should be able to:
- Open any file in `src/tinyrag/` and know what it does and why.
- Swap the LLM, embedding model, or vector store by changing one config line.
- Add a new sensor source in under 30 minutes.
- Explain the architecture to the capstone panel.

---

## 2. Architectural Style

TinyRAG follows a **layered, dependency-injected, protocol-oriented** architecture. The key principles:

| Principle | Meaning in TinyRAG |
|-----------|---------------------|
| **Layered** | UI → API → Application Services → Domain Logic → Infrastructure. Each layer depends only on the layer below it through interfaces. |
| **Dependency Injection (DI)** | No module instantiates its dependencies directly. Dependencies are passed in at startup. |
| **Protocol-Oriented** | Every external dependency (LLM, vector store, embedding model, sensor source, input adapter) is hidden behind a Python `Protocol`. |
| **Configuration-Driven** | No module reads paths, hostnames, or magic numbers from the code. Everything comes from `config.yaml`. |
| **Single Responsibility** | Each module does one thing. Chunking, embedding, retrieval, generation are separate files. |
| **Testability First** | Every module can be instantiated and tested in isolation by passing in fake/mock dependencies. |

This is the **same architectural style** used in production systems at companies like Stripe, Shopify, and most modern Python backend codebases. It's a transferable skill beyond this capstone.

---

## 3. System Context (C4 Level 1)

The "big picture" — TinyRAG and the things around it.

```
┌────────────────────────────────────────────────────────────────────────┐
│                          User's Local Network                            │
│                                                                          │
│   ┌────────────┐                  ┌──────────────────────────────┐     │
│   │  Browser   │  HTTP (LAN)      │      Raspberry Pi 5           │     │
│   │  (any      │ ────────────────→│      (or Laptop fallback)     │     │
│   │  device)   │                  │                               │     │
│   └────────────┘                  │      ┌─────────────────────┐  │     │
│                                   │      │   TinyRAG System    │  │     │
│                                   │      └─────────────────────┘  │     │
│                                   │              │                  │     │
│                                   │              ↓ (optional)      │     │
│                                   │      ┌─────────────────────┐  │     │
│                                   │      │  Real IoT Sensors   │  │     │
│                                   │      │  (DHT22, PIR)       │  │     │
│                                   │      └─────────────────────┘  │     │
│                                   │              │                  │     │
│                                   │              ↓ (optional)      │     │
│                                   │      ┌─────────────────────┐  │     │
│                                   │      │  Local MQTT Broker  │  │     │
│                                   │      │  (Mosquitto)        │  │     │
│                                   │      └─────────────────────┘  │     │
│                                   └──────────────────────────────┘     │
│                                                                          │
│   (NO connection to the internet at runtime — all data stays local)     │
└────────────────────────────────────────────────────────────────────────┘
```

**External actors:**
- **User** (via browser) — sends queries, receives answers.
- **Real IoT sensors** (optional) — provide live temperature/humidity/motion data.
- **Local MQTT broker** (optional) — relays sensor data to TinyRAG.

**External systems TinyRAG does NOT talk to at runtime:**
- No cloud LLM APIs.
- No telemetry services.
- No analytics.
- No error reporting services.

---

## 4. Container Diagram (C4 Level 2)

The runtime containers (processes) inside the TinyRAG system.

```
┌──────────────── Raspberry Pi 5 / Laptop ────────────────────────┐
│                                                                   │
│  ┌─────────────────┐         ┌──────────────────┐                │
│  │  Web Browser    │  HTTP   │  FastAPI Backend │                │
│  │  (Static UI)    │ ←─────→ │  (Python process)│                │
│  └─────────────────┘         └────────┬─────────┘                │
│                                        │                           │
│                          ┌─────────────┼─────────────┐             │
│                          ↓             ↓             ↓             │
│                ┌──────────────┐ ┌─────────────┐ ┌──────────────┐  │
│                │  Retriever   │ │  LLM Client │ │ Ingestion    │  │
│                │  (in-proc)   │ │  (HTTP)     │ │ (CLI script) │  │
│                └──────┬───────┘ └──────┬──────┘ └──────┬───────┘  │
│                       │                │               │          │
│                       ↓                ↓               ↓          │
│              ┌──────────────┐  ┌──────────────┐ ┌──────────────┐  │
│              │  FAISS Index │  │ llama.cpp    │ │ PDF/Text     │  │
│              │  (on disk)   │  │ HTTP Server  │ │ Parser       │  │
│              └──────────────┘  └──────────────┘ └──────────────┘  │
│                                                                   │
│              ┌──────────────┐  ┌──────────────┐                   │
│              │ SQLite DB    │  │ Sensor       │                   │
│              │ (metadata)   │  │ Source(s)    │                   │
│              └──────────────┘  └──────────────┘                   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

**Containers:**

| Container | Process | Technology | Port |
|----------|---------|------------|------|
| **Web UI** | Served as static files | HTML + vanilla JS + CSS | (served by FastAPI) |
| **FastAPI Backend** | Python process | FastAPI 0.115+ on uvicorn | 8000 |
| **llama.cpp server** | C++ process | llama.cpp HTTP server | 8080 |
| **Ingestion pipeline** | Python script | run via CLI | N/A |
| **Vector store** | On-disk file | FAISS index | N/A |
| **Metadata DB** | On-disk file | SQLite | N/A |
| **Sensor source** | Python module | Pluggable (3 impls) | N/A (or MQTT 1883) |

**Why two processes (FastAPI + llama.cpp)?**
- **Process isolation:** if the LLM crashes, the API can auto-restart it (NFR-20).
- **Memory accounting:** each process's RAM is visible independently.
- **Standard practice:** llama.cpp is officially distributed as a separate server binary.

---

## 5. Module Decomposition (C4 Level 3)

TinyRAG is implemented as a single Python package: `src/tinyrag/`. Below is the module tree with the responsibility of each file.

```
src/tinyrag/
├── __init__.py
├── main.py                  # FastAPI app factory + lifespan management
├── config.py                # Loads + validates config.yaml
│
├── api/                     # HTTP layer (FastAPI routes)
│   ├── __init__.py
│   ├── routes_query.py      # POST /api/query, GET /api/status
│   ├── routes_docs.py       # POST /api/documents, GET /api/documents, DELETE /api/documents/{id}
│   └── routes_admin.py      # POST /api/admin/reindex, POST /api/admin/benchmark
│
├── core/                    # Domain logic (no I/O knowledge)
│   ├── __init__.py
│   ├── chunker.py           # Text → chunks (token-based, with overlap)
│   ├── retriever.py         # Query → top-k chunks
│   ├── prompt_builder.py    # Context + query → grounded prompt
│   ├── answer.py            # Data class for the final answer + citations
│   └── sensor_summarizer.py # Sensor data → text-summary chunks
│
├── ingestion/               # Document → vector store pipeline
│   ├── __init__.py
│   ├── pipeline.py          # Orchestrates: parse → chunk → embed → store
│   ├── parsers.py           # PDF / TXT / MD → raw text
│   └── embedder.py          # Wrapper around sentence-transformers
│
├── generation/              # LLM interaction
│   ├── __init__.py
│   └── llm_client.py        # Wraps llama.cpp HTTP API (OpenAI-compatible)
│
├── storage/                 # Persistence layer
│   ├── __init__.py
│   ├── vector_store.py      # FAISS wrapper
│   └── metadata.py          # SQLite chunk metadata + document registry
│
├── sensors/                 # Pluggable sensor data sources
│   ├── __init__.py
│   ├── base.py              # SensorSource Protocol
│   ├── simulated.py         # SimulatedCSVSource
│   ├── serial_dht.py        # RealSerialSource (DHT22 + PIR over GPIO)
│   └── mqtt.py              # MQTTBrokerSource
│
├── input_adapters/          # Pluggable user input (text, voice)
│   ├── __init__.py
│   ├── base.py              # InputAdapter Protocol
│   └── text.py              # TextInputAdapter (always works)
│   # voice.py              # VoiceInputAdapter (stretch, if built)
│
├── ui/                      # Web UI (static files)
│   ├── static/
│   │   ├── style.css
│   │   ├── chat.js
│   │   └── admin.js
│   └── templates/
│       ├── index.html       # Chat page
│       └── admin.html       # Document management page
│
└── observability/           # Logging, metrics
    ├── __init__.py
    └── logger.py            # Structured JSON logger
```

### 5.1 Module Responsibilities

| Module | Responsibility | Key Classes/Functions |
|--------|----------------|----------------------|
| `main.py` | FastAPI app factory; wires all modules together via DI; manages llama.cpp subprocess lifecycle. | `create_app(config) -> FastAPI` |
| `config.py` | Loads `config.yaml`, validates it, exposes typed settings object. | `Settings` (Pydantic model) |
| `api/routes_query.py` | HTTP routes for asking questions and viewing system status. | `POST /api/query`, `GET /api/status` |
| `api/routes_docs.py` | HTTP routes for managing documents (upload, list, delete). | `POST /api/documents`, `GET /api/documents`, `DELETE /api/documents/{id}` |
| `api/routes_admin.py` | HTTP routes for admin operations (reindex, benchmark). | `POST /api/admin/reindex` |
| `core/chunker.py` | Splits text into overlapping chunks based on token count. | `Chunker.chunk(text, source) -> list[Chunk]` |
| `core/retriever.py` | Embeds query, searches both indices, merges & filters results. | `Retriever.retrieve(query, k_doc, k_sensor) -> RetrievalResult` |
| `core/prompt_builder.py` | Constructs a grounded prompt with system instructions + context + query. | `PromptBuilder.build(query, chunks) -> Prompt` |
| `core/answer.py` | Data class for an answer (text, citations, latency, sources). | `Answer` dataclass |
| `core/sensor_summarizer.py` | Converts raw sensor DataFrame into text-summary chunks. | `SensorSummarizer.summarize(df) -> list[Chunk]` |
| `ingestion/pipeline.py` | Orchestrates: parse → chunk → embed → save to vector store + metadata DB. | `IngestionPipeline.run(file_path) -> IngestionReport` |
| `ingestion/parsers.py` | Detects file type and extracts text (+ page numbers for PDFs). | `parse(path) -> ParsedDocument` |
| `ingestion/embedder.py` | Loads the embedding model once; provides `.embed(texts) -> vectors`. | `Embedder` class wrapping sentence-transformers |
| `generation/llm_client.py` | Wraps llama.cpp HTTP API. Streams tokens. | `LLMClient.generate(prompt) -> Iterator[str]` |
| `storage/vector_store.py` | Wraps FAISS: add, search, save, load, delete-by-source. | `VectorStore` class |
| `storage/metadata.py` | SQLite: stores chunk metadata + document registry. | `MetadataStore` class |
| `sensors/base.py` | Defines the `SensorSource` Protocol. | `class SensorSource(Protocol)` |
| `sensors/simulated.py` | Reads from a CSV file. | `SimulatedCSVSource` |
| `sensors/serial_dht.py` | Reads DHT22 + PIR via libgpiod. | `RealSerialSource` |
| `sensors/mqtt.py` | Subscribes to an MQTT broker. | `MQTTBrokerSource` |
| `input_adapters/base.py` | Defines the `InputAdapter` Protocol. | `class InputAdapter(Protocol)` |
| `input_adapters/text.py` | Trivial text input (no transformation). | `TextInputAdapter` |
| `ui/templates/*.html` | Jinja2 templates for the chat and admin pages. | — |
| `ui/static/*.{css,js}` | Frontend assets. | — |
| `observability/logger.py` | JSON logger factory. | `get_logger(name) -> Logger` |

---

## 6. The Protocol Interfaces (the heart of the architecture)

These are the contracts that make TinyRAG swappable. **Memorize this section — it's the most important architectural idea in the project.**

### 6.1 `SensorSource` Protocol

```python
# src/tinyrag/sensors/base.py
from typing import Protocol
import pandas as pd

class SensorSource(Protocol):
    """Anything that can produce a DataFrame of sensor records."""

    def read(self, since: datetime | None = None) -> pd.DataFrame:
        """
        Return a DataFrame with columns:
            timestamp   (datetime)
            sensor_type (str: 'temperature' | 'humidity' | 'energy' | 'motion')
            sensor_id   (str: e.g., 'living_room_temp')
            value       (float)
            unit        (str: e.g., 'C', '%', 'kWh', 'count')
        """
        ...

    def available_sensors(self) -> list[str]:
        """List of sensor IDs this source can provide."""
        ...
```

**Three implementations:**

| Class | Backing | Use case |
|-------|---------|----------|
| `SimulatedCSVSource` | A CSV file | Default; always available |
| `RealSerialSource` | DHT22 + PIR over GPIO | When lab sensor is connected (Pi 5 only) |
| `MQTTBrokerSource` | A local Mosquitto broker | When sensors publish via MQTT |

Switching is one config line — **no code change**.

### 6.2 `EmbeddingModel` Protocol

```python
# src/tinyrag/ingestion/embedder.py (interface)
from typing import Protocol

class EmbeddingModel(Protocol):
    """Anything that can turn text into a dense vector."""

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns a list of vectors."""
        ...
```

**Default implementation:** `SentenceTransformerEmbedder` wrapping `sentence-transformers/all-MiniLM-L6-v2` (384-dim, 80 MB).

**Easy swaps:**
- `BgeEmbedder` wrapping `BAAI/bge-small-en-v1.5` (384-dim, 33 MB — smaller, similar quality).
- `MxbaiEmbedder` wrapping `mixedbread-ai/mxbai-embed-small-v1` (384-dim, 67 MB).

### 6.3 `VectorStore` Protocol

```python
# src/tinyrag/storage/vector_store.py (interface)
from typing import Protocol

class VectorStore(Protocol):
    """Anything that can store and search embedding vectors."""

    def add(self, vectors: list[list[float]], ids: list[str]) -> None: ...
    def search(self, query_vector: list[float], k: int) -> list[tuple[str, float]]: ...
    def delete_by_source(self, source_id: str) -> int: ...
    def save(self) -> None: ...
    def load(self) -> None: ...
    def size(self) -> int: ...
```

**Default implementation:** `FAISSStore` (uses `faiss-cpu`).

**Alternative:** `ChromaStore` (uses `chromadb`).

We use **two instances** of the same implementation: one for documents, one for sensor data. They are configured identically but hold disjoint data.

### 6.4 `LLMClient` Protocol

```python
# src/tinyrag/generation/llm_client.py (interface)
from typing import Protocol, Iterator

class LLMClient(Protocol):
    """Anything that can generate text from a prompt."""

    def generate(self, prompt: str, *, max_tokens: int = 512,
                 temperature: float = 0.0) -> Iterator[str]:
        """Yield answer tokens one at a time."""
        ...

    def model_name(self) -> str: ...
    def is_healthy(self) -> bool: ...
```

**Default implementation:** `LlamaCppClient` that talks to `llama-server` via HTTP (OpenAI-compatible `/v1/chat/completions` endpoint).

**Alternatives (future plug-ins):**
- `OllamaClient` — talks to local Ollama daemon.
- `LlamaCppPythonClient` — in-process via `llama-cpp-python` (no separate server).

### 6.5 `InputAdapter` Protocol

```python
# src/tinyrag/input_adapters/base.py
from typing import Protocol

class InputAdapter(Protocol):
    """Anything that can turn raw user input into a text query."""

    def transcribe(self, raw_input: bytes | str) -> str:
        """
        For TextInputAdapter, raw_input is the typed string (no-op).
        For VoiceInputAdapter, raw_input is audio bytes; returns transcribed text.
        """
        ...
```

**Implementations:**
- `TextInputAdapter` — always present, default.
- `VoiceInputAdapter` — optional/stretch, uses Whisper.cpp.

---

## 7. Dependency Injection Wiring

DI is what makes the architecture swappable. Here's how it works at startup.

### 7.1 The Composition Root

`main.py` is the **only** place that imports concrete classes. It builds the dependency graph and passes it down.

```python
# src/tinyrag/main.py (simplified, illustrative)
from contextlib import asynccontextmanager
from fastapi import FastAPI

from tinyrag.config import load_settings
from tinyrag.ingestion.embedder import SentenceTransformerEmbedder
from tinyrag.storage.vector_store import FAISSStore
from tinyrag.storage.metadata import MetadataStore
from tinyrag.generation.llm_client import LlamaCppClient
from tinyrag.sensors.simulated import SimulatedCSVSource
from tinyrag.sensors.serial_dht import RealSerialSource       # Pi only
from tinyrag.sensors.mqtt import MQTTBrokerSource
from tinyrag.input_adapters.text import TextInputAdapter
from tinyrag.api.routes_query import build_query_router
from tinyrag.api.routes_docs import build_docs_router

def create_app(settings) -> FastAPI:
    # --- Build infrastructure (concrete classes) ---
    embedder = SentenceTransformerEmbedder(settings.embedding.model_name)
    doc_store = FAISSStore(settings.retrieval.doc_index_path, dim=embedder.dimension)
    sensor_store = FAISSStore(settings.retrieval.sensor_index_path, dim=embedder.dimension)
    metadata = MetadataStore(settings.storage.metadata_db_path)
    llm = LlamaCppClient(settings.llm.server_url, settings.llm.model_path)

    # --- Build sensor source based on config ---
    sensor_source = _build_sensor_source(settings)

    # --- Build input adapter ---
    input_adapter = TextInputAdapter()

    # --- Lifespan: load resources ---
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        doc_store.load()
        sensor_store.load()
        metadata.init_schema()
        llm.ensure_server_running()   # spawns llama.cpp if not up
        yield
        doc_store.save()
        sensor_store.save()

    app = FastAPI(lifespan=lifespan)

    # --- Inject dependencies into routes ---
    app.include_router(build_query_router(
        retriever=..., prompt_builder=..., llm=llm,
        doc_store=doc_store, sensor_store=sensor_store, metadata=metadata,
    ))
    app.include_router(build_docs_router(
        pipeline=..., metadata=metadata, doc_store=doc_store,
    ))

    return app
```

### 7.2 Why this matters

- **Testing:** unit tests can pass in a `FakeLLMClient` and a `FakeVectorStore` — no real LLM, no real FAISS needed.
- **Swapping:** changing `FAISSStore` to `ChromaStore` is a 2-line change in `main.py` — no other file changes.
- **Clarity:** reading `main.py` tells you exactly what the system is made of.

---

## 8. Data Flow — End-to-End

### 8.1 Ingestion (one-time per document)

```
  User clicks "Upload" in UI
        ↓
  [FastAPI] POST /api/documents  (multipart/form-data, file)
        ↓
  [IngestionPipeline.run(file_path)]
        ↓
  [parsers.parse(file_path)]              →  ParsedDocument(text, page_map, type)
        ↓
  [chunker.chunk(text, metadata)]         →  list[Chunk(text, source, page, idx)]
        ↓
  [embedder.embed(chunk_texts)]           →  list[vector]
        ↓
  [vector_store.add(vectors, chunk_ids)]  →  FAISS index updated
        ↓
  [metadata.insert_chunks(chunks)]        →  SQLite rows inserted
        ↓
  [return IngestionReport(num_chunks, time)]
        ↓
  [FastAPI] 200 OK with report
        ↓
  UI shows "Indexed 47 chunks in 3.2s"
```

### 8.2 Query (text mode, end-to-end)

```
  User types "How do I reset my thermostat?" in chat box
        ↓
  [Browser] POST /api/query  (JSON: {query: "..."})
        ↓
  [FastAPI] route handler
        ↓
  [input_adapter.transcribe(query)]       →  "How do I reset my thermostat?"  (no-op for text)
        ↓
  [retriever.retrieve(query, k_doc=3, k_sensor=2)]
        ↓
       ├─ [embedder.embed([query])]       →  query_vector
       ├─ [doc_store.search(query_vector, 3)]    →  [(chunk_id_1, score_0.85), ...]
       ├─ [sensor_store.search(query_vector, 2)] →  [(chunk_id_5, score_0.61), ...]  (only if sensor keywords detected)
       └─ [metadata.get_chunks(ids)]      →  list[Chunk with text + source]
        ↓
  [retriever] filter by similarity threshold (≥ 0.3)
        ↓
  [prompt_builder.build(query, retrieved_chunks)]
        ↓
       System prompt:
         "You are TinyRAG, a smart home assistant.
          Answer the user's question using ONLY the provided context.
          If the context is insufficient, say so.
          Cite sources using [1], [2], etc."
       Context: "[1] thermostat manual p.12: 'To reset...'"
                "[2] custom FAQ: '...'"
       Question: "How do I reset my thermostat?"
        ↓
  [llm_client.generate(prompt)]           →  iterates over SSE tokens
        ↓
  [FastAPI] streams tokens back via Server-Sent Events
        ↓
  [Browser chat.js] appends tokens to the assistant message bubble
        ↓
  When stream ends, UI renders citation cards below the answer
```

### 8.3 Sensor Data Ingestion (one-time setup)

```
  [SensorSource.read(since=None)]          →  DataFrame (timestamp, type, value, ...)
        ↓
  [SensorSummarizer.summarize(df)]         →  list[Chunk]
        ↓
       e.g., "On 2026-06-15, living_room_temp averaged 24.3°C,
              peaking at 27.1°C at 16:00, lowest at 19.2°C at 05:30."
        ↓
  [embedder.embed(chunk_texts)]            →  vectors
        ↓
  [sensor_store.add(vectors, chunk_ids)]   →  FAISS sensor index updated
        ↓
  [metadata.insert_chunks(chunks, source='sensor_log')]
```

### 8.4 Sensor Source Selection at Boot

```
  config.yaml:  sensors.source: simulated | real_serial | mqtt
        ↓
  _build_sensor_source(settings)
        ↓
  switch settings.sensors.source:
    case "simulated":
        return SimulatedCSVSource(settings.sensors.csv_path)
    case "real_serial":
        if target == laptop:  raise ConfigError(...)
        return RealSerialSource(gpio_pin=4, pir_pin=17)
    case "mqtt":
        return MQTTBrokerSource(settings.sensors.mqtt_broker, settings.sensors.mqtt_topic)
```

---

## 9. Component Interaction Diagram

```
            ┌──────────────────────────────────────────────────────────────┐
            │                         main.py                              │
            │  (composition root — only place that knows concrete classes) │
            └──────────────┬───────────────────────────────────────────────┘
                           │ builds and injects
        ┌──────────────────┼────────────────────┬─────────────────────┐
        ↓                  ↓                    ↓                     ↓
  ┌───────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────────┐
  │ Embedder  │    │  Doc Vector  │    │ Sensor Vector  │    │   Metadata   │
  │ (sentence-│    │  Store       │    │ Store          │    │   (SQLite)   │
  │  transfr) │    │  (FAISS)     │    │  (FAISS)       │    │              │
  └─────┬─────┘    └──────┬───────┘    └───────┬────────┘    └──────┬───────┘
        │                 │                    │                    │
        │ injected into   │                    │                    │
        ↓                 ↓                    ↓                    ↓
  ┌────────────────────────────────────────────────────────────────────┐
  │                          core/                                     │
  │   chunker ──→ retriever ──→ prompt_builder ──→ answer              │
  │      │            │                │              │                │
  │      │            │                │              │                │
  │      ↓            ↓                ↓              ↓                │
  │   parsers    (uses embedder,    (uses llm)     (data class)         │
  │              uses both stores)                                     │
  └────────────────────────────────────────────────────────────────────┘
                           ↑
                           │ used by
            ┌──────────────┴───────────────┐
            │           api/               │
            │  routes_query ──→ retriever  │
            │  routes_docs  ──→ ingestion  │
            │  routes_admin ──→ admin ops  │
            └──────────────────────────────┘
                           ↑
                           │ HTTP from
            ┌──────────────┴───────────────┐
            │           ui/ (static)        │
            │  Browser loads chat + admin   │
            └──────────────────────────────┘
```

---

## 10. Data Model (preview — full version in `04_database_design_v1.md`)

### 10.1 Vector Store

Two FAISS indices on disk:

| Index | File | Content |
|-------|------|---------|
| Doc index | `data/vector_store/doc.faiss` | Embeddings of text chunks from documents |
| Sensor index | `data/vector_store/sensor.faiss` | Embeddings of text-summary chunks from sensor data |

Each index is a flat L2-normalized index (cosine similarity = inner product on L2-normalized vectors).

### 10.2 SQLite Metadata DB (`data/metadata.db`)

```sql
-- Documents registry
CREATE TABLE documents (
    id           TEXT PRIMARY KEY,         -- UUID
    filename     TEXT NOT NULL,
    doc_type     TEXT NOT NULL,            -- 'manual' | 'faq' | 'sensor'
    size_bytes   INTEGER,
    num_chunks   INTEGER,
    ingested_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    content_hash TEXT NOT NULL             -- SHA-256 of full text
);

-- Chunks (one row per chunk, joined to document)
CREATE TABLE chunks (
    id           TEXT PRIMARY KEY,         -- UUID, also the FAISS vector ID
    document_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,         -- 0-based index within document
    page_number  INTEGER,                  -- NULL for non-PDF
    text         TEXT NOT NULL,
    char_offset  INTEGER,
    token_count  INTEGER
);

CREATE INDEX idx_chunks_doc ON chunks(document_id);

-- Query log (for debugging & evaluation, not for analytics)
CREATE TABLE query_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    query          TEXT,
    top1_score     REAL,
    num_chunks     INTEGER,
    latency_ms     INTEGER,
    model          TEXT
);
```

### 10.3 Sensor Data (CSV/JSON on disk)

```csv
timestamp,sensor_id,sensor_type,value,unit
2026-06-22T14:00:00,living_room_temp,temperature,24.3,C
2026-06-22T14:00:00,living_room_hum,humidity,55.2,%
2026-06-22T14:00:00,house_energy,energy,0.12,kWh
2026-06-22T14:00:05,kitchen_motion,motion,1,count
```

---

## 11. Technology Stack (preview — pinned in `05_tech_stack_v1.md`)

| Layer | Technology | Why |
|-------|------------|-----|
| LLM inference | **llama.cpp** (HTTP server mode) | C++ performance, mature, Pi 5 + x86 builds |
| LLM (primary) | **Phi-3 Mini 3.8B Instruct Q4_K_M** | Best quality for size |
| LLM (alternates) | TinyLlama 1.1B, Llama 3.2 3B | For 3-model comparison |
| Embedding | **sentence-transformers / all-MiniLM-L6-v2** | Small (80 MB), fast, good quality |
| Vector store | **FAISS (faiss-cpu)** | Fast, in-process, well-supported |
| Metadata DB | **SQLite 3** | Zero-config, file-based, perfect for capstone scale |
| PDF parsing | **pdfplumber** | More accurate than PyPDF2 for complex layouts |
| Text parsing | Built-in Python (TXT, MD) | No extra dep |
| Backend | **FastAPI 0.115+** | Modern, async, auto-docs |
| ASGI server | **uvicorn** | Standard FastAPI runner |
| Frontend | **HTML + vanilla JS + Jinja2** | No build step, no framework hell |
| Sensor I/O (Pi) | **libgpiod + adafruit-circuitpython-dht** | Standard Pi GPIO access |
| MQTT (optional) | **paho-mqtt** | De facto Python MQTT client |
| Testing | **pytest** | Standard |
| Linting | **ruff** | Fast, modern |
| Config | **PyYAML + Pydantic v2** | Type-safe config loading |
| Logging | **structlog** | JSON structured logs |

---

## 12. Cross-Cutting Concerns

### 12.1 Logging

- All modules use `structlog` to emit JSON logs.
- Each log line includes: `timestamp`, `level`, `module`, `event`, plus event-specific fields.
- Logs go to:
  - `stdout` (for human reading during dev).
  - `logs/tinyrag.log` (JSON, append-only, for postmortem).
- The query endpoint logs: query, top-1 score, num chunks retrieved, retrieval latency, generation latency, total latency, model used (NFR-36).

### 12.2 Error Handling

- HTTP layer translates domain exceptions to HTTP responses (400, 404, 413, 500).
- Domain layer raises typed exceptions: `EmptyVectorStoreError`, `ChunkingError`, `LLMUnavailableError`, etc.
- No bare `except:` — every handler logs the exception with context.
- User-facing errors are human-readable; tracebacks never reach the UI (NFR-35).

### 12.3 Concurrency

- FastAPI is async; routes are `async def`.
- The LLM client wraps a synchronous HTTP streaming call in `asyncio.to_thread()` to avoid blocking the event loop.
- Sensor sources can be sync (file reads are fast enough).
- Ingestion is a **batch CLI operation** — runs in its own process, not via HTTP, to keep the API responsive.

### 12.4 Security

- All user-supplied filenames are sanitized (NFR-17).
- File size limited to 50 MB (NFR-22, FR-10).
- The FastAPI server binds to `127.0.0.1` by default (NFR-15).
- No authentication (NFR-16, out-of-scope).

### 12.5 Configuration

- `config.yaml` is loaded once at startup.
- All keys are validated by a Pydantic `Settings` model — typos cause startup failure with a clear error.
- The path to `config.yaml` can be overridden by `TINYRAG_CONFIG` env var (useful for testing).

---

## 13. Deployment Topology

### 13.1 On the Raspberry Pi 5

```
   ┌─────────────── Raspberry Pi 5 (Raspberry Pi OS 64-bit) ───────────┐
   │                                                                    │
   │  systemd service: tinyrag.service                                  │
   │  ├── ExecStart=/home/pi/tinyrag/run.sh                             │
   │  ├── Restart=on-failure                                            │
   │  ├── WorkingDirectory=/home/pi/tinyrag                             │
   │  └── Environment=PYTHONUNBUFFERED=1                                │
   │                                                                    │
   │  Files:                                                            │
   │  ├── /home/pi/tinyrag/         (code, venv)                        │
   │  ├── /home/pi/tinyrag/models/  (GGUF files, ~3 GB)                 │
   │  ├── /home/pi/tinyrag/data/    (vector store, metadata, sensor CSVs)│
   │  └── /home/pi/tinyrag/logs/    (log files)                         │
   │                                                                    │
   │  llama.cpp process: started by run.sh, runs in background         │
   │  FastAPI process: started by run.sh, runs in foreground            │
   │                                                                    │
   └────────────────────────────────────────────────────────────────────┘
```

### 13.2 On the Dell laptop (fallback)

Same structure, but:
- Path is `~/tinyrag/` (or wherever the student clones it).
- No systemd service — `run.sh` in a terminal, or background with `nohup`.
- llama.cpp built with OpenBLAS for ~2× speedup (NFR-12).

---

## 14. Build & Run Scripts (preview)

### 14.1 `setup.sh` (one-command install)

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "[1/5] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip python3-venv \
                        build-essential cmake git libgpiod-dev

echo "[2/5] Creating Python venv..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "[3/5] Building llama.cpp..."
if [ ! -d llama.cpp ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
fi
cd llama.cpp
# Platform-specific build flags (auto-detected)
if [[ "$(uname -m)" == "aarch64" ]]; then
    cmake -B build -DGGML_OPENBLAS=OFF -mcpu=cortex-a76
else
    cmake -B build -DGGML_OPENBLAS=ON
fi
cmake --build build --config Release -j
cd ..

echo "[4/5] Downloading models..."
python scripts/download_models.py

echo "[5/5] Initializing vector store + metadata DB..."
python scripts/init_storage.py

echo "Setup complete! Run: ./run.sh"
```

### 14.2 `run.sh` (one-command start)

```bash
#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

# Start llama.cpp server in background
./llama.cpp/build/bin/llama-server \
    --model models/phi-3-mini-3.8b-instruct-q4.gguf \
    --host 127.0.0.1 --port 8080 \
    --ctx-size 4096 --n-gpu-layers 0 \
    > logs/llamacpp.log 2>&1 &
LLAMA_PID=$!
trap "kill $LLAMA_PID" EXIT

# Wait for llama.cpp to be ready
for i in {1..30}; do
    if curl -s http://localhost:8080/health > /dev/null; then
        break
    fi
    sleep 1
done

# Start FastAPI
uvicorn tinyrag.main:app --host 127.0.0.1 --port 8000
```

---

## 15. Testing Strategy (preview)

| Test type | Tool | What it covers |
|-----------|------|----------------|
| **Unit tests** | `pytest` | chunker, retriever, prompt_builder, parsers, sensor_summarizer |
| **Integration tests** | `pytest` + `httpx.AsyncClient` | Full /api/query round-trip with a fake LLM |
| **LLM client tests** | `pytest` + `vcr.py` (HTTP recording) | LlamaCppClient against recorded HTTP responses |
| **Eval tests** | `scripts/eval.py` | 20-question gold set, scored against each model |
| **Smoke tests** | shell script | `setup.sh` + `run.sh` + a sample query |

Coverage target: ≥ 60% line coverage on `core/`, `ingestion/`, `generation/`, `storage/`.

---

## 16. Open Questions for the Student

These will be resolved when I write the next document (`04_database_design_v1.md`), but flag them now if you have opinions:

| # | Question | Default I'll go with |
|---|----------|----------------------|
| Q1 | Embedding model — `all-MiniLM-L6-v2` (80 MB, 384-dim) or `bge-small-en-v1.5` (33 MB, 384-dim)? | **all-MiniLM-L6-v2** — slightly higher quality, more battle-tested |
| Q2 | Vector store — FAISS or ChromaDB? | **FAISS** — simpler, in-process, no extra service |
| Q3 | PDF parser — `pdfplumber` or `PyPDF2`? | **pdfplumber** — better for complex layouts (tables, multi-column) |
| Q4 | Should the query log DB be queryable from the UI (e.g., "show last 50 queries")? | **No** — out-of-scope; logs are for the developer only |
| Q5 | For the multi-model comparison: do we re-index between models? | **No** — only the LLM swaps; the vector store stays the same |

---

## 17. Document Approval

| Role | Name | Approval | Date |
|------|------|----------|------|
| Student | Marajul Haque | ⏳ pending | |
| Advisor | Abu Nowshed Chy | (not required for v1) | |

---

*End of Architecture v1. Next: Database / Storage Design (`docs/04_database_design_v1.md`) — the student reviews and approves this Architecture first.*
