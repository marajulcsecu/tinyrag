# TinyRAG — System Requirements Specification (SRS) v1

**Project Title:** TinyRAG — A Lightweight, On-Device Retrieval-Augmented Generation Assistant for Smart Home IoT
**Document version:** 1.0
**Date:** 2026-06-23
**Status:** Draft — awaiting student review
**Supersedes:** (none — first SRS)
**Source of truth:** `docs/01_project_scope_v2.md`

---

## 1. Introduction

### 1.1 Purpose

This document specifies the **functional and non-functional requirements** for the TinyRAG system. It converts the project scope (`docs/01_project_scope_v2.md`) into a list of **testable, traceable requirements** that will be used to:
- Drive the design of the architecture.
- Define the acceptance test cases for the final defense.
- Provide a checklist to know when the project is "done."

### 1.2 Scope

TinyRAG is a **self-contained, on-device** web application that:
- Ingests smart-home device manuals (PDF) and a custom home FAQ (Markdown).
- Reads from real or simulated IoT sensors (temperature, humidity, energy, motion).
- Stores everything in a local vector store + metadata database.
- Answers natural-language user questions using a local small LLM (Phi-3 Mini 3.8B Q4) running on llama.cpp.
- Streams answers back to a web UI with source citations.
- Runs on a Raspberry Pi 5 (primary) or a Dell Inspiron 15 3520 laptop running Ubuntu 24.04 LTS (fallback).
- Makes **zero cloud calls** at runtime.

### 1.3 Definitions, Acronyms, Abbreviations

| Term | Definition |
|------|------------|
| **RAG** | Retrieval-Augmented Generation |
| **LLM** | Large Language Model |
| **SLM** | Small Language Model (used interchangeably with "small LLM" in this doc) |
| **Embedding** | Dense numerical vector representation of text |
| **Vector store** | Database optimized for similarity search over embeddings (FAISS or Chroma) |
| **Chunk** | A 200–500 token slice of a document, stored in the vector store |
| **Quantization** | Reducing numeric precision of model weights (e.g., 16-bit → 4-bit) to shrink size |
| **GGUF** | File format used by llama.cpp for quantized models |
| **Pi** | Raspberry Pi 5 (8 GB) — primary deployment target |
| **Laptop** | Dell Inspiron 15 3520 — fallback deployment target |
| **IoT** | Internet of Things — refers to sensors/actuators in this project |
| **FR / NFR** | Functional Requirement / Non-Functional Requirement |
| **SRS** | System Requirements Specification (this document) |
| **API** | Application Programming Interface |
| **UI** | User Interface |
| **AC** | Acceptance Criteria |
| **KL** | Knowledge base (the collection of ingested documents) |

### 1.4 References

- `docs/01_project_scope_v2.md` — Project scope
- `docs/00_high_level_plan.md` — Project journey
- `AGENT.md` — Project context handoff
- `docs/laptop_fallback/README.md` — Laptop-specific notes
- llama.cpp: https://github.com/ggerganov/llama.cpp
- FastAPI: https://fastapi.tiangolo.com/
- FAISS: https://github.com/facebookresearch/faiss
- sentence-transformers: https://www.sbert.net/
- Raspberry Pi OS: https://www.raspberrypi.com/software/
- IEEE 830 — IEEE Standard for SRS (style reference)

### 1.5 Document Conventions

- **Requirement IDs** follow the pattern `FR-N` (functional) and `NFR-N` (non-functional).
- **"shall"** indicates a mandatory requirement.
- **"should"** indicates a recommended but non-mandatory requirement.
- **"may"** indicates an optional capability.
- All requirements are testable — each has a corresponding acceptance criterion in Section 7.

---

## 2. Overall Description

### 2.1 Product Perspective

TinyRAG is a **standalone web application** running on a single edge device. It is composed of these **loosely-coupled modules**:

| Module | Purpose |
|--------|---------|
| **Web UI** | Browser-based chat interface served by the backend |
| **FastAPI backend** | REST API orchestrating the RAG pipeline |
| **Document ingestion pipeline** | PDF/TXT/MD → chunks → embeddings → vector store |
| **Retriever** | Query embedding → top-k similar chunks |
| **LLM client** | Talks to llama.cpp HTTP server |
| **Prompt builder** | Constructs grounded prompts with retrieved context + citations |
| **Sensor source** | Pluggable data source (simulated, real serial, MQTT) |
| **Vector store** | Persistent FAISS or Chroma index |
| **Metadata DB** | SQLite storing chunk-to-source mapping |
| **Configuration** | Single `config.yaml` controlling all runtime behavior |

### 2.2 User Classes and Characteristics

| User class | Description | Technical level |
|------------|-------------|-----------------|
| **Homeowner (primary)** | Uploads device manuals, asks natural-language questions | Non-technical |
| **Developer/Admin (secondary)** | Installs, configures, monitors, runs evaluation | Comfortable with CLI, Python basics |

### 2.3 Operating Environment

**Primary target — Raspberry Pi 5:**
- Raspberry Pi OS 64-bit (Debian Bookworm or later)
- Python 3.10+
- 8 GB RAM
- ≥ 16 GB free storage
- llama.cpp compiled with Cortex-A76 NEON optimizations

**Fallback target — Dell Inspiron 15 3520:**
- Ubuntu 24.04.4 LTS
- Python 3.10+ (system Python is fine; no conda needed)
- 8 GB RAM
- 512 GB SSD
- llama.cpp compiled with OpenBLAS acceleration

**Network:** required only for one-time model download. After setup, **zero outbound network calls**.

### 2.4 Design and Implementation Constraints

| # | Constraint | Reason |
|---|------------|--------|
| C1 | No cloud LLM APIs at runtime | Privacy is a primary value proposition |
| C2 | Open-source stack only (MIT-compatible licenses) | Project will be open-sourced |
| C3 | Single-user, single-device | Capstone scope; multi-user auth out-of-scope |
| C4 | English language only | Out-of-scope (multi-language is a future plug-in) |
| C5 | Total install size ≤ 10 GB | Disk constraint |
| C6 | LLM model size ≤ 2.5 GB on disk | RAM constraint on Pi 5 (8 GB) |
| C7 | Embedding model size ≤ 100 MB on disk | Disk + RAM constraint |
| C8 | Latency budget ≤ 5 s end-to-end on Pi 5 | UX requirement |
| C9 | All code follows clean architecture (Protocol interfaces, DI, no hardcoding) | Non-negotiable student requirement |
| C10 | One-command setup (`./setup.sh`) and one-command start (`./run.sh`) | Reproducibility |

### 2.5 Assumptions and Dependencies

- **A1:** The student has (or will get) a Raspberry Pi 5 / 8 GB from the lab. If not, the Dell laptop path is the documented fallback.
- **A2:** Real IoT sensors (DHT22, PIR) are *optional*. If unavailable, the simulated source is the default and the demo still works.
- **A3:** Internet is available during the initial setup (for downloading models and Python packages). After setup, the system runs offline.
- **A4:** The student has terminal access (SSH or local keyboard/monitor).
- **A5:** The student will use a Chromium-based or Firefox browser to access the UI.
- **A6:** The Phi-3 Mini 3.8B model file is available on Hugging Face (verified: yes).
- **A7:** The system is single-tenant — there is no concept of multiple users or roles.

### 2.6 Apportioning of Requirements

Requirements tagged **[L]** are laptop-specific extensions (relevant only on the laptop path, e.g., real sensors unavailable, OpenBLAS acceleration). All other requirements apply to both targets.

---

## 3. Functional Requirements

### 3.1 Document Ingestion

| ID | Requirement |
|----|-------------|
| **FR-1** | The system shall accept PDF, TXT, and Markdown files via (a) a CLI command and (b) a web UI upload form. |
| **FR-2** | The system shall extract plain text from PDF files, preserving reading order, and record the page number for each extracted text span. |
| **FR-3** | The system shall split each document into chunks of 200–500 tokens (configurable) with 50-token overlap (configurable). |
| **FR-4** | The system shall compute one embedding vector per chunk using a local sentence-transformer model. |
| **FR-5** | The system shall persist the following metadata for each chunk: source filename, document type (manual/FAQ/sensor), chunk index, page number (for PDFs), character offset in source, ingestion timestamp. |
| **FR-6** | The system shall store the resulting vectors in a persistent local vector store (FAISS or ChromaDB — final choice in architecture doc). |
| **FR-7** | The system shall support **re-ingestion** of a document (overwrite by source filename) and **deletion** of a document by source filename. |
| **FR-8** | The system shall deduplicate chunks whose SHA-256 content hash is identical to an already-stored chunk. |
| **FR-9** | The system shall log ingestion progress (file name, number of chunks, time taken) for each document. |
| **FR-10** | The system shall validate that uploaded files do not exceed 50 MB per file. |

### 3.2 Sensor Data Integration

| ID | Requirement |
|----|-------------|
| **FR-11** | The system shall read sensor data from a **pluggable `SensorSource` interface**. Three implementations shall be provided: (a) `SimulatedCSVSource` (default), (b) `RealSerialSource` (lab DHT22 + PIR), (c) `MQTTBrokerSource`. |
| **FR-12** | The system shall support the following sensor types: temperature (°C, float), humidity (%, float), energy consumption (kWh, float), motion (binary or event count, int). |
| **FR-13** | The system shall accept sensor data in CSV or JSON format with the schema: `timestamp, sensor_type, sensor_id, value, unit`. |
| **FR-14** | The system shall convert sensor data into text-summary chunks (e.g., "On 2026-06-15, the living-room temperature averaged 24.3°C, peaking at 27.1°C at 16:00") and embed them in a **separate** sensor vector index. |
| **FR-15** | The system shall support natural-language queries over sensor data, such as averages, peaks, comparisons across days, and time-range filtering. |
| **FR-16** | The system shall allow the active sensor source to be switched by changing one config key (`sensor_source: simulated | real_serial | mqtt`) — no code change required. |
| **FR-17** | The system shall generate 30 days of synthetic sensor data on first run if no real data is present, for demonstration. |
| **FR-18 [L]** | On the laptop path, only `SimulatedCSVSource` and `MQTTBrokerSource` shall be supported (no GPIO). |

### 3.3 Query Processing and Retrieval

| ID | Requirement |
|----|-------------|
| **FR-19** | The system shall accept a natural-language query from the user via the web UI text box. |
| **FR-20** | The system shall embed the query using the same embedding model used during ingestion. |
| **FR-21** | The system shall retrieve the **top-k (configurable, default k=3)** most similar chunks from the document index. |
| **FR-22** | The system shall retrieve the **top-k (configurable, default k=2)** most similar chunks from the sensor index, when the query contains sensor-related keywords OR when the user has selected "sensor data" mode. |
| **FR-23** | The system shall merge retrieved chunks from both indices (if applicable) and rank by similarity score. |
| **FR-24** | The system shall include retrieved chunks' source metadata (filename, page/chunk, excerpt) in the response payload. |
| **FR-25** | The system shall use cosine similarity for vector search. |
| **FR-26** | The system shall set a similarity threshold (configurable, default 0.3 cosine); if no chunk exceeds it, the system shall return a fixed fallback message: *"I don't have information about that in my knowledge base."* |

### 3.4 Answer Generation

| ID | Requirement |
|----|-------------|
| **FR-27** | The system shall construct a prompt consisting of: (a) a system prompt (instructing grounded answering and citation), (b) the retrieved chunks with `[1]`, `[2]` markers, (c) the user's question. |
| **FR-28** | The system shall generate an answer using a local LLM via the llama.cpp HTTP server. |
| **FR-29** | The system shall stream the generated answer to the UI **token-by-token** for perceived low latency. |
| **FR-30** | The system shall include numbered source citations `[1]`, `[2]`, ... in the generated answer, where each number maps to a retrieved chunk. |
| **FR-31** | The system shall set the LLM sampling temperature to **0** (deterministic) by default. |
| **FR-32** | The system shall set the LLM max output tokens to **512** by default. |
| **FR-33** | The system shall support swapping the LLM by replacing the GGUF model file referenced in `config.yaml` — no code change required. |
| **FR-34** | The system shall support comparing different LLM models via a CLI command (`scripts/benchmark.py --model <name>`). |

### 3.5 User Interface (Web)

| ID | Requirement |
|----|-------------|
| **FR-35** | The system shall provide a web UI accessible at `http://<host>:8000/` (default port, configurable). |
| **FR-36** | The UI shall display a chat-style interface: user query on the right, assistant answer (streamed) on the left. |
| **FR-37** | The UI shall display source-citation cards below each answer, showing filename, page/chunk number, and the cited excerpt. |
| **FR-38** | The UI shall provide a **"Manage Documents"** page where the user can upload new documents, list all ingested documents, and delete documents. |
| **FR-39** | The UI shall provide a **"System Status"** panel showing: (a) current LLM model name, (b) embedding model name, (c) number of indexed chunks (doc + sensor), (d) current RAM usage (MB), (e) llama.cpp server status (up/down). |
| **FR-40** | The UI shall allow the user to clear chat history and start a new conversation. |
| **FR-41** | The UI shall show the current sensor source mode (simulated / real / mqtt) in the status panel. |
| **FR-42** | The UI shall display a "Loading..." indicator while the LLM is generating. |
| **FR-43** | The UI shall be styled with simple, clean CSS (no external CSS frameworks required). |

### 3.6 Voice Input (Stretch Goal — Modular Adapter)

| ID | Requirement |
|----|-------------|
| **FR-44** | The system shall define an `InputAdapter` Protocol with a single method: `def transcribe(self, audio_bytes) -> str`. |
| **FR-45** | The system shall provide a `TextInputAdapter` implementation (always works, default). |
| **FR-46** | The system **may** provide a `VoiceInputAdapter` implementation using Whisper.cpp. *(Stretch — built only if time allows after core RAG is complete.)* |
| **FR-47** | If `VoiceInputAdapter` is built, the UI shall provide a microphone button that records audio and submits the transcribed text to the RAG pipeline identically to typed input. |
| **FR-48** | If voice is built, the system shall use the `whisper.cpp` `tiny.en` model (~75 MB) to keep the resource footprint small. |

### 3.7 Configuration

| ID | Requirement |
|----|-------------|
| **FR-49** | The system shall read all runtime configuration from a single `config.yaml` file at startup. |
| **FR-50** | The configuration shall include (at minimum): model path, embedding model name, chunk size, chunk overlap, top-k, similarity threshold, LLM temperature, server ports, sensor source mode, deployment target. |
| **FR-51** | The system shall validate `config.yaml` at startup and exit with a clear error if any required key is missing. |
| **FR-52** | The system shall support two deployment targets via config: `deployment.target: raspberry_pi` and `deployment.target: laptop`. The target controls llama.cpp build flags, sensor source defaults, and performance tuning. |

### 3.8 Operations and CLI

| ID | Requirement |
|----|-------------|
| **FR-53** | The system shall provide a CLI command `python scripts/ingest.py <file>` to ingest a single document. |
| **FR-54** | The system shall provide a CLI command `python scripts/ask.py "<question>"` to ask a question from the terminal (no UI). |
| **FR-55** | The system shall provide a CLI command `python scripts/benchmark.py` to run the full benchmark suite (latency, RAM, accuracy). |
| **FR-56** | The system shall provide a CLI command `python scripts/eval.py --model <name>` to run the 20-question gold-set evaluation against a specific model. |
| **FR-57** | The system shall write structured logs to `logs/tinyrag.log` in JSON format, including timestamp, level, module, message. |
| **FR-58** | The system shall provide a `./run.sh` script that starts the FastAPI server and the llama.cpp server, then prints the access URL. |

---

## 4. Non-Functional Requirements

### 4.1 Performance

| ID | Requirement | Target (Pi 5) | Target (Laptop) |
|----|-------------|---------------|-----------------|
| **NFR-1** | End-to-end query latency (query → first streamed token) | ≤ 2.0 s | ≤ 1.0 s |
| **NFR-2** | End-to-end query latency (query → full answer, ≤ 200 output tokens) | ≤ 5.0 s | ≤ 3.0 s |
| **NFR-3** | Embedding throughput (during ingestion) | ≥ 50 chunks/sec | ≥ 200 chunks/sec |
| **NFR-4** | Vector retrieval latency (top-5, 1000-chunk store) | ≤ 100 ms | ≤ 50 ms |
| **NFR-5** | UI page first-paint time | ≤ 1.0 s | ≤ 0.5 s |
| **NFR-6** | Cold-start time (system boot → ready to accept queries) | ≤ 30 s | ≤ 15 s |

### 4.2 Resource Footprint

| ID | Requirement | Target (Pi 5) | Target (Laptop) |
|----|-------------|---------------|-----------------|
| **NFR-7** | Idle RAM (after model load, no query) | ≤ 1.5 GB | ≤ 1.0 GB |
| **NFR-8** | Peak RAM during inference | ≤ 3.0 GB | ≤ 2.0 GB |
| **NFR-9** | LLM model size on disk | ≤ 2.5 GB (Q4-quantized) | same |
| **NFR-10** | Embedding model size on disk | ≤ 100 MB | same |
| **NFR-11** | Total install size (code + models + vector store + sensor data) | ≤ 10 GB | same |
| **NFR-12 [L]** | llama.cpp shall use OpenBLAS acceleration on the laptop | N/A | required |

### 4.3 Privacy and Security

| ID | Requirement |
|----|-------------|
| **NFR-13** | The system shall make **zero outbound network requests** during normal operation, after initial model download. *(Testable: run with `iptables` blocking all egress, or simply with Wi-Fi off, and verify the system works.)* |
| **NFR-14** | The system shall not log user query content to any external destination. Local logging is allowed. |
| **NFR-15** | The web UI shall bind to `127.0.0.1` by default; LAN access requires explicit configuration (`server.host: 0.0.0.0`). |
| **NFR-16** | The system shall not require any user authentication in the capstone version. *(Authentication is a future plug-in.)* |
| **NFR-17** | The system shall sanitize all user-supplied filenames to prevent path traversal attacks. |

### 4.4 Reliability

| ID | Requirement |
|----|-------------|
| **NFR-18** | The system shall return HTTP 400 (not crash) for malformed queries, empty queries, or queries exceeding 2000 characters. |
| **NFR-19** | The system shall handle an empty vector store gracefully — return the fallback message from FR-26, not an error. |
| **NFR-20** | The system shall auto-restart the llama.cpp subprocess once if it crashes during inference, and surface a clear error if it crashes twice. |
| **NFR-21** | The system shall not lose ingested data on a clean restart. *(Testable: ingest N docs, restart, verify N docs still present.)* |
| **NFR-22** | The system shall survive an unexpected large file upload by returning HTTP 413 (Payload Too Large), not crashing. |

### 4.5 Maintainability

| ID | Requirement |
|----|-------------|
| **NFR-23** | All public functions shall have docstrings. |
| **NFR-24** | All public functions shall have type hints. |
| **NFR-25** | Core modules (`chunker`, `retriever`, `prompt_builder`, `parsers`, `llm_client`) shall have unit tests with ≥ 60% line coverage. |
| **NFR-26** | The system shall pass `ruff` linting with no errors. |
| **NFR-27** | The system shall be installable with a single `./setup.sh` script. |
| **NFR-28** | The codebase shall be organized into clearly-separated modules (no monolithic `main.py` > 300 lines). |
| **NFR-29** | Every external dependency (LLM, vector store, embedding model, sensor source) shall be hidden behind a Python Protocol interface. |
| **NFR-30** | No business logic shall import from a concrete third-party library directly — only through the Protocol interface. |

### 4.6 Portability

| ID | Requirement |
|----|-------------|
| **NFR-31** | The system shall run on both `aarch64` (Pi 5) and `x86_64` (laptop) with no code changes — only config and platform-specific build flags differ. |
| **NFR-32** | The system shall not assume any specific Linux distribution beyond Debian-family (Ubuntu, Raspberry Pi OS, etc.). |

### 4.7 Usability

| ID | Requirement |
|----|-------------|
| **NFR-33** | A non-technical user shall be able to ask a question and receive an answer in ≤ 3 clicks from loading the UI. |
| **NFR-34** | A non-technical user shall be able to upload a document and ask a question about it in ≤ 1 minute of UI interaction. |
| **NFR-35** | All error messages in the UI shall be human-readable (no Python tracebacks shown to the end user). |

### 4.8 Observability

| ID | Requirement |
|----|-------------|
| **NFR-36** | The system shall log each query with: timestamp, query text, retrieval latency, generation latency, total latency, retrieved chunk count, top-1 similarity score, model used. |
| **NFR-37** | The system shall provide a `GET /api/status` endpoint returning JSON with the fields described in FR-39. |

---

## 5. External Interface Requirements

### 5.1 User Interfaces

- **Web browser** (Chrome ≥ 100, Firefox ≥ 100, Safari ≥ 15) accessing `http://<host>:8000/`.
- **CLI** (terminal) for ingestion, querying, benchmarking, evaluation.

### 5.2 Software Interfaces

| Interface | Technology | Endpoint |
|-----------|------------|----------|
| LLM inference | llama.cpp HTTP server | `http://localhost:8080/v1/chat/completions` (OpenAI-compatible) |
| Embedding model | sentence-transformers (Python library, in-process) | N/A |
| Vector store | FAISS or ChromaDB (Python library, in-process) | N/A |
| Metadata DB | SQLite 3 | `data/metadata.db` |
| Web framework | FastAPI 0.115+ | `http://localhost:8000/` |
| PDF parser | pdfplumber or PyPDF2 (Python library) | N/A |
| Sensor source | Pluggable — see FR-11 | N/A |

### 5.3 Communication Interfaces

- **Local HTTP only** (no external ports opened).
- **No MQTT, no WebSocket, no gRPC** in the core deliverable.
- **Optional:** local MQTT broker (Mosquitto) for `MQTTBrokerSource` sensor implementation.

---

## 6. System Models

### 6.1 Use Case Diagram (textual)

```
                       ┌────────────────────────────────────┐
                       │          TinyRAG System             │
                       ├────────────────────────────────────┤
   UC-1: Ingest doc    │ UC-2: Ask question    │ UC-3: Manage docs │
   UC-4: Configure     │ UC-5: View status     │ UC-6: Evaluate    │
   UC-7: Demo          │ UC-8: (stretch) Voice │                   │
                       └────────────────────────────────────┘
                              ▲                  ▲
                              │                  │
                ┌─────────────┘                  └─────────────┐
                │                                              │
        ┌───────┴────────┐                          ┌──────────┴────────┐
        │   Homeowner    │                          │  Developer/Admin │
        │ (non-technical)│                          │  (technical)     │
        └────────────────┘                          └───────────────────┘
```

| UC | Actor | Description |
|----|-------|-------------|
| UC-1 | Developer, Homeowner | Upload / ingest a document (PDF / TXT / MD) |
| UC-2 | Homeowner | Ask a natural-language question, receive streamed answer + sources |
| UC-3 | Homeowner, Developer | List, delete, re-ingest documents |
| UC-4 | Developer | Edit `config.yaml` to change models, ports, sensor source, etc. |
| UC-5 | Homeowner, Developer | View system status (model, RAM, indexed chunk count) |
| UC-6 | Developer | Run gold-set evaluation and benchmarks |
| UC-7 | Anyone | Watch the live demo |
| UC-8 | Homeowner | (stretch) Speak a question instead of typing |

### 6.2 Data Flow — Ingestion

```
   Document (PDF/TXT/MD)
         ↓
   [Parser]      →  raw text + page numbers (if PDF)
         ↓
   [Chunker]     →  list of (chunk_text, metadata) tuples
         ↓
   [Embedder]    →  list of (chunk_text, embedding_vector, metadata)
         ↓
   [Vector Store] ←  persisted to disk
   [Metadata DB] ←  chunk_id → source mapping persisted
```

### 6.3 Data Flow — Query

```
   User query (text)
         ↓
   [Embedder]              →  query_embedding
         ↓
   [Retriever]             →  top-k doc chunks + top-k sensor chunks
         ↓
   [Prompt Builder]        →  system_prompt + context + query
         ↓
   [LLM Client]            →  POST to llama.cpp server
         ↓
   [Streaming Response]    →  tokens → UI (token-by-token)
         ↓
   [Citation Mapper]       →  [1], [2] in answer → chunk metadata
         ↓
   [UI]                    →  rendered answer + source cards
```

### 6.4 Data Flow — Sensor

```
   Sensor source (simulated / real / MQTT)
         ↓
   [Sensor Adapter]        →  normalized (timestamp, type, value) records
         ↓
   [Sensor Summarizer]     →  daily-summary text chunks
         ↓
   [Embedder]              →  embeddings
         ↓
   [Sensor Vector Store]   →  persisted to disk (separate index)
```

---

## 7. Acceptance Criteria

The system is **accepted** when **all** the following are demonstrable at the final defense. Each AC maps back to one or more requirements.

| AC # | Acceptance Criterion | Maps to |
|------|----------------------|---------|
| **AC-1** | A PDF device manual can be uploaded via the UI and indexed within 60 seconds (for a 50-page manual). | FR-1, FR-2, FR-3, FR-6 |
| **AC-2** | A user can ask *"How do I reset my [device]?"* and receive a correct, cited answer in ≤ 5 seconds (Pi 5). | FR-19 to FR-32, NFR-1, NFR-2 |
| **AC-3** | A user can ask *"What was the average temperature in [room] this week?"* and receive a correct answer with cited sensor data. | FR-11 to FR-23 |
| **AC-4** | The system operates with Wi-Fi disabled (verified by physical disconnect or `iptables` block). | NFR-13 |
| **AC-5** | The 20-question gold set is run against at least 3 LLMs; results are tabulated. | FR-34, FR-56 |
| **AC-6** | Latency, RAM, and disk-usage measurements are reported for each model. | NFR-1, NFR-2, NFR-7, NFR-8 |
| **AC-7** | The system survives a clean restart without losing ingested data. | NFR-21 |
| **AC-8** | A malformed query returns HTTP 400, not a crash. | NFR-18 |
| **AC-9** | An empty vector store returns the fallback message, not an error. | FR-26, NFR-19 |
| **AC-10** | Swapping the LLM in `config.yaml` (different GGUF file) requires no code change. | FR-33 |
| **AC-11** | Unit tests for core modules pass with `pytest` and ≥ 60% coverage. | NFR-25 |
| **AC-12** | The code passes `ruff` linting with no errors. | NFR-26 |
| **AC-13** | `./setup.sh` followed by `./run.sh` brings up a working system on a clean Ubuntu 24.04 install. | NFR-27, FR-58 |
| **AC-14** | The demo runs live on a Raspberry Pi 5 with Wi-Fi disabled. | (demo criterion) |
| **AC-15** | A recorded demo video is available as backup. | (demo criterion) |

---

## 8. Future / Out-of-Scope Requirements (for the report's "Future Work" section)

These are **explicitly out of scope** for this capstone but **designed into the architecture** as future plug-ins:

| Future Feature | Architecture Hook |
|----------------|-------------------|
| Mobile native app | REST API (NFR-31) is already platform-agnostic |
| Multi-language support | i18n layer in UI + multilingual embedding model swap via config |
| Multi-user authentication | FastAPI middleware can be added to existing routes |
| Real voice I/O | `InputAdapter` Protocol (FR-44) already defined |
| Real smart-home API integration | `SensorSource` Protocol (FR-11) accepts new implementations |
| On-device fine-tuning | New module; LLM serving abstraction allows LoRA integration |
| Cloud sync (optional) | Would require relaxing NFR-13 — kept out by design |
| Distributed deployment | Architecture is single-device; would need message queue |

---

## 9. Appendices

### Appendix A: Sample Queries (used in evaluation)

1. *"How do I reset my smart thermostat to factory settings?"* → Nest manual
2. *"What does error code E3 mean on my Philips Hue bulb?"* → Philips Hue manual
3. *"How do I pair a new TP-Link Kasa smart plug?"* → Kasa manual
4. *"What's the Wi-Fi setup process for the Echo Dot?"* → Echo Dot manual
5. *"What was the average temperature in the living room this week?"* → Sensor log
6. *"Which day last week had the highest energy usage?"* → Sensor log
7. *"Was there motion in the kitchen between 2am and 3am last Tuesday?"* → Sensor log
8. *"Compare the humidity between the bedroom and living room yesterday."* → Sensor log
9. *"What smart devices do I have and how do I set them up?"* → Custom Home FAQ
10. *"Tell me about the home network setup."* → Custom Home FAQ
11. *"Why is my energy bill higher than usual?"* → RAG: combines sensor log + FAQ
12. *"What's the recommended humidity for comfortable sleep?"* → Custom Home FAQ
13. *"What is the warranty period for my thermostat?"* → Nest manual
14. *"How do I update the firmware on my Hue bridge?"* → Philips Hue manual
15. *"Did the temperature drop below 18°C at any point last week?"* → Sensor log
16. *"What's the average daily energy consumption this month?"* → Sensor log
17. *"Hello, who are you?"* → No retrieval; casual LLM response
18. *"What is the meaning of life?"* → Out-of-domain; should trigger fallback
19. *"When was my living room temperature highest last week?"* → Sensor log
20. *"Which device uses the most standby power?"* → Manual + sensor data combined

### Appendix B: Default Configuration Values (initial `config.yaml`)

```yaml
deployment:
  target: laptop   # raspberry_pi | laptop

server:
  host: 127.0.0.1
  port: 8000

llm:
  model_path: models/phi-3-mini-3.8b-instruct-q4.gguf
  server_url: http://localhost:8080
  context_size: 4096
  temperature: 0.0
  max_tokens: 512
  gpu_layers: 0   # 0 = CPU only

embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  device: cpu
  batch_size: 32

chunking:
  chunk_size: 400      # tokens
  chunk_overlap: 50    # tokens

retrieval:
  doc_top_k: 3
  sensor_top_k: 2
  similarity_threshold: 0.3
  index_type: faiss   # faiss | chroma

sensors:
  source: simulated   # simulated | real_serial | mqtt
  csv_path: data/sensor_logs/synthetic_30d.csv

logging:
  level: INFO
  json_format: true
  path: logs/tinyrag.log
```

### Appendix C: Glossary of UI Terms

| Term in UI | User-facing meaning |
|------------|---------------------|
| "Knowledge base" | The collection of uploaded documents |
| "Source" | A specific document the answer was retrieved from |
| "Citation" | A reference to a specific chunk in a specific document |
| "Chunk" | A small piece of a document (user doesn't need to know this) |
| "Model" | The LLM currently serving answers (shown in status panel) |

---

## 10. Document Approval

| Role | Name | Approval | Date |
|------|------|----------|------|
| Student | Marajul Haque | ⏳ pending | |
| Advisor | Abu Nowshed Chy | (not required for v1) | |

---

*End of SRS v1. Next: architecture design (`docs/03_architecture_v1.md`) — the student reviews and approves this SRS first.*
