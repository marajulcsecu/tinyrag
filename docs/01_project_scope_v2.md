# TinyRAG — Project Scope (v2, Refined — Decisions Locked)

**Project Title:** TinyRAG — A Lightweight, On-Device Retrieval-Augmented Generation Assistant for Smart Home IoT

**Student:** Marajul Haque
**Advisor:** Abu Nowshed Chy
**Duration:** 8–10 weeks (quality-first, not time-boxed)
**Status:** v2 — decisions locked, awaiting student approval to proceed to SRS
**Supersedes:** `01_project_scope_v1.md` (kept for history)

---

## 1. One-Line Vision

> A fully on-device, privacy-preserving, professionally-engineered smart home assistant that answers natural-language questions about home devices, sensor history, and appliance manuals — running on a Raspberry Pi 5 (or laptop fallback) with **zero cloud calls** and a **clean, modular architecture** that allows every component to be swapped independently.

---

## 2. Problem Statement

Today's smart home assistants (Alexa, Google Home, Siri) share three problems:

1. **Cloud dependency** — no internet, no assistant.
2. **Privacy concerns** — voice and behavioral data leave the home.
3. **Limited context awareness** — they cannot answer detailed questions about the *specific* devices in *your* home from *your* manuals.

TinyRAG addresses all three by running a small LLM + RAG pipeline **entirely on-device**, with the household's own documents and sensor logs as the knowledge base.

---

## 3. Locked-In Decisions

| # | Decision | Choice |
|---|----------|--------|
| D1 | **Use case** | Smart Home Assistant |
| D2 | **Primary LLM** | Phi-3 Mini 3.8B Instruct (Q4_K_M quantized) |
| D3 | **Secondary LLMs (for evaluation)** | TinyLlama 1.1B, Llama 3.2 3B, optionally Mistral 7B |
| D4 | **LLM serving** | llama.cpp HTTP server |
| D5 | **Backend framework** | FastAPI |
| D6 | **UI** | HTML + vanilla JS (served by FastAPI) |
| D7 | **UI language** | English |
| D8 | **Input mode (primary)** | Text via web UI |
| D9 | **Input mode (stretch)** | Voice via Whisper.cpp (modular adapter, only if time allows) |
| D10 | **Knowledge base** | 2–3 real device manuals (PDF) + 1 custom home FAQ (Markdown) |
| D11 | **Sensor types** | Temperature, humidity, energy (kWh), motion |
| D12 | **Sensor source** | Pluggable: Simulated (default) + Real lab sensor + MQTT |
| D13 | **Conversation model** | Single-turn (no chat history) |
| D14 | **Demo format** | Live demo on Pi (primary) + recorded video (backup) |
| D15 | **Related-work section** | Brief 1-page comparison vs. PrivateGPT, Ollama, etc. |
| D16 | **Architecture quality** | Professional / clean / modular — non-negotiable |
| D17 | **Primary target** | Raspberry Pi 5 / 8 GB (requested from lab) |
| D18 | **Fallback target** | Dell Inspiron 15 3520 (i5-1235U, 8 GB RAM, 512 GB SSD) — Ubuntu 24.04.4 LTS |
| D19 | **Time pressure** | None — quality over speed |
| D20 | **Source code license** | MIT License (open-source, public GitHub repo) |
| D21 | **Out-of-scope features** | Treated as **future plug-ins** — designed into the architecture, not hard-excluded |

---

## 4. What the System Will Do (In-Scope)

### 4.1 Knowledge base

The system ingests and indexes:

| Source | Format | Example |
|--------|--------|---------|
| Smart-home device manuals | PDF (real, from manufacturer websites) | Nest thermostat, Philips Hue, TP-Link Kasa |
| Quick-start guides | PDF or Markdown | Echo Dot setup |
| Custom home FAQ | Markdown (written by student) | 10–20 hand-written Q&A |
| Sensor logs | CSV / JSON | 30 days of synthetic data |

### 4.2 User interface (text-first)

- Web-based chat interface at `http://<host>:8000/`.
- Streamed token-by-token answer display.
- Source-citation cards below each answer (filename, page/chunk, excerpt).
- "Manage Documents" page: upload, list, delete.
- "System Status" panel: model loaded, vector store size, current RAM.

### 4.3 Voice (stretch goal — modular adapter)

- **Architecture from Day 1:** an `InputAdapter` Protocol with `TextInputAdapter` (always works) and `VoiceInputAdapter` (Whisper.cpp, only built if time allows).
- Same RAG/LLM pipeline for both. Voice = "fancy text input."
- Activated by a microphone button in the UI when built.

### 4.4 IoT / sensor integration (real + simulated, pluggable)

```
   ┌─────────────────────────────┐
   │   SensorSource (Protocol)   │
   ├─────────────────────────────┤
   │  • SimulatedCSVSource       │  ← default
   │  • RealSerialSource         │  ← lab DHT22 + PIR over GPIO
   │  • MQTTBrokerSource         │  ← lab MQTT broker
   └─────────────────────────────┘
            ↓
   (one config line picks which)
```

**Sensors:** temperature (°C), humidity (%), energy (kWh), motion (events).

**Default behavior:** synthetic 30-day CSV generated at install time.
**Lab mode:** if a DHT22 / PIR / smart plug is provided, switch `config.yaml` → `sensor_source: real` and the same code path serves real data.

### 4.5 Example queries the system must handle

| Query | Source retrieved |
|-------|------------------|
| *"How do I reset my Nest thermostat?"* | Nest manual PDF |
| *"What does error code E3 mean on my Hue bulb?"* | Philips Hue manual |
| *"What was the average living-room temperature this week?"* | Sensor log |
| *"Which day last week used the most energy?"* | Sensor log |
| *"Was there motion in the kitchen between 2-3am?"* | Sensor log |
| *"What smart bulbs do I have and how do I pair them?"* | Custom home FAQ |
| *"Hello, who are you?"* | (No retrieval; casual LLM response) |

### 4.6 Evaluation strategy

**Compare 3+ small open-source LLMs** on the same RAG pipeline:

| Model | Size (Q4) | Role in evaluation |
|-------|-----------|---------------------|
| TinyLlama 1.1B Chat | ~700 MB | Fastest baseline |
| Llama 3.2 3B Instruct | ~1.8 GB | Newest Meta, good quality |
| **Phi-3 Mini 3.8B (4k)** | ~2.3 GB | **Primary / shipped** |
| *(optional)* Mistral 7B | ~4 GB | Quality ceiling (likely too slow on Pi 5) |

**Measured on each model:**
- Answer accuracy (vs. 20-question gold set, manually judged)
- End-to-end latency (first token, full answer)
- Peak RAM usage
- Disk size

---

## 5. What the System Will NOT Do (Explicit Out-of-Scope)

- ❌ Any cloud LLM API calls (no OpenAI, Anthropic, etc.)
- ❌ Real voice I/O in the **core deliverable** (stretch goal only)
- ❌ Real smart-home hardware integration (Hue API, etc.) — simulated
- ❌ Multi-user authentication
- ❌ Mobile native app
- ❌ Multi-language support (English only)
- ❌ On-device fine-tuning or federated learning
- ❌ Real-time model retraining

---

## 6. High-Level Architecture (preview — full version in architecture doc)

```
┌──────────────────────────────────────────────────────────────┐
│               Raspberry Pi 5  (or Laptop)                     │
│                                                               │
│  ┌──────────┐  ┌─────────────┐  ┌──────────────────┐        │
│  │ Web UI   │←→│ FastAPI     │←→│ llama.cpp server │        │
│  │ HTML/JS  │  │ backend     │  │ Phi-3 Mini 3.8B  │        │
│  └──────────┘  └──────┬──────┘  └──────────────────┘        │
│                       │                                       │
│                       ↓                                       │
│              ┌────────────────┐    ┌──────────────────┐       │
│              │ Retriever      │←──→│ Vector store     │       │
│              │ top-k cosine   │    │ FAISS / Chroma   │       │
│              └────────┬───────┘    └──────────────────┘       │
│                       │                                       │
│              ┌────────┴────────┐                              │
│              ↓                 ↓                              │
│     ┌──────────────┐   ┌──────────────────┐                  │
│     │ Doc index    │   │ Sensor index     │                  │
│     │ manuals +FAQ │   │ temp/hum/energy/ │                  │
│     │              │   │ motion           │                  │
│     └──────────────┘   └──────────────────┘                  │
│                                                               │
│              ┌──────────────────────┐                         │
│              │  SensorSource        │                         │
│              │  (pluggable)         │                         │
│              └──────────────────────┘                         │
└──────────────────────────────────────────────────────────────┘
         ↑                                                        
         │ (optional)                                            
   ┌─────┴───────┐                                            
   │ Lab sensor  │                                            
   │ / MQTT      │                                            
   └─────────────┘                                            
```

---

## 7. Architecture Principles (non-negotiable)

1. **Separation of concerns** — UI, backend, retrieval, generation, storage, sensor I/O are all separate modules.
2. **Dependency injection via interfaces** — every external dependency is hidden behind a Python Protocol/ABC. Swap by changing config, not code.
3. **Configuration over hardcoding** — `config.yaml` is the single source of runtime config.
4. **No cloud calls at runtime** — verified by running with Wi-Fi off.
5. **Reproducible** — `setup.sh` and `run.sh` bring up the entire system from scratch.
6. **Testable** — core modules (chunking, retrieval, prompt construction) have unit tests.
7. **Professional logging** — structured logs, not print statements.
8. **Documented** — every module has a docstring; architecture decisions captured in `docs/`.

---

## 8. Success Criteria

The capstone is **accepted** when all the following are demonstrable:

1. ✅ User can upload a PDF device manual via the UI and the system indexes it.
2. ✅ User can ask a question and get a **cited** answer in under 5 seconds (text mode, on the primary target hardware).
3. ✅ User can ask a sensor-history question and get a correct answer from local logs.
4. ✅ The system runs **with Wi-Fi disabled** (verified at demo).
5. ✅ At least **3 models** are compared; results table is in the final report.
6. ✅ Latency, RAM, and accuracy numbers are measured, reported, and reproducible (via `scripts/benchmark.py`).
7. ✅ The code passes a basic code-quality check (linting, type hints, docstrings).
8. ✅ The final report is professionally formatted with all sections from the SRS.

---

## 9. Top Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LLM too slow on Pi 5 | Medium | Medium | Aggressive quantization (Q4_K_M); pick ≤3B models; benchmark in Week 2; fall back to TinyLlama if needed |
| Lab Pi 5 not provided | Medium | High | **Architecture is hardware-agnostic** — full laptop path is the documented fallback (see `docs/laptop_fallback/`) |
| Lab sensor unavailable | High | Low | Simulated source is the default; real source is opt-in via config |
| Hallucination despite RAG | Medium | High | Strong system prompt, temp=0, mandatory source citation, gold-set evaluation |
| Out of memory | Low | High | Lazy-load models; pick 1.5–2.5 GB LLM; monitor with `/proc/meminfo` |
| Voice mode eats time | Low | Low | Voice is a stretch goal, gated behind completion of core RAG + evaluation |
| llama.cpp build issues on Pi | Medium | Medium | Use prebuilt binaries if available; pin a known-working commit |

---

## 10. Deliverables

| # | Deliverable | Target week |
|---|-------------|-------------|
| 1 | Project Scope (this document) | ✅ Done |
| 2 | System Requirements Specification (SRS) | Week 2 |
| 3 | Architecture & Tech Stack document | Week 3 |
| 4 | Database / Storage Design | Week 3 |
| 5 | Development Roadmap | Week 4 |
| 6 | Working prototype on laptop | Week 6 |
| 7 | Deployed on Raspberry Pi 5 + benchmarks | Week 8 |
| 8 | Evaluation results across 3+ models | Week 9 |
| 9 | Final report + demo video + live demo | Week 10 |

---

## 11. Next Steps

1. **Student reviews and approves this scope** (or requests changes).
2. Once approved, I write **`docs/02_srs_v1.md`** (System Requirements Specification) — turning this scope into testable FRs and NFRs.
3. After SRS is approved: architecture, database design, tech stack, roadmap.

**This is the right way to do a capstone:** scope → requirements → architecture → implementation → evaluation. We follow it strictly.

---

*End of v2 scope. The journey continues — read `docs/00_high_level_plan.md` for the full visualization.*
