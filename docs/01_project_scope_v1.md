# TinyRAG — Project Scope (v1, Draft for Discussion)

**Project Title:** TinyRAG — A Lightweight, On-Device Retrieval-Augmented Generation Assistant for Smart Home IoT

**Student:** [Marajul Haque]
**Advisor:** [Abu Nowshed Chy]
**Duration:** 8–10 weeks
**Status:** v1 — awaiting your review and answers to open questions

---

## 1. One-Line Vision

> A fully on-device, privacy-preserving smart home assistant that can answer natural-language questions about home devices, sensor history, and appliance manuals — running entirely on a Raspberry Pi 5, with zero cloud calls.

---

## 2. Problem Statement

Today, smart home assistants (Alexa, Google Home, Siri) all share three problems:

1. **Cloud dependency** — no internet, no assistant.
2. **Privacy concerns** — voice and behavioral data leave the home.
3. **Limited context awareness** — they cannot answer detailed questions about the *specific* devices in *your* home, from *your* manuals.

TinyRAG addresses all three by running a small LLM + RAG pipeline **entirely on a Raspberry Pi 5**, with the household's own documents and sensor logs as the knowledge base.

---

## 3. Target Hardware

**Raspberry Pi 5 (8 GB RAM)** — recommended.
*Falls back to Pi 4 (4 GB) with reduced performance and smaller model.*

Storage requirement: ~10 GB free (models + vector store + sensor data).
Network: required **only for one-time model download**. After setup, fully offline.

---

## 4. What the System Will Do (In-Scope)

### 4.1 Knowledge base (what the system can answer questions about)

The system ingests and indexes the following types of documents:

| Source type | Format | Example |
|-------------|--------|---------|
| Smart-home device manuals | PDF (real) | Nest thermostat, Philips Hue, TP-Link Kasa manuals |
| Quick-start guides | PDF or Markdown | Echo dot setup guide |
| Custom home FAQ | Markdown (written by you) | 10–20 hand-written Q&A about your home setup |
| Sensor logs | CSV / JSON | Temperature, humidity, energy (kWh), motion events |

### 4.2 User interface (how people interact)

**Primary mode — Text (built in Weeks 4–7):**
- Web-based chat interface served by FastAPI on the Pi.
- Accessible at `http://<pi-ip>:8000/` from any device on the LAN.
- Streamed token-by-token answer display (like ChatGPT).
- Source-citation cards below each answer showing which documents were used.

**Stretch mode — Voice (designed from Day 1, implemented only if time allows in Weeks 8–9):**
- Input adapter for Whisper.cpp (on-device speech-to-text).
- Same LLM/RAG pipeline as text mode (no code duplication).
- Activated by a microphone button in the UI.

**Why this design:** Voice is a thin "input adapter" — it just produces text. The actual RAG/LLM core is identical. This means we can build the hard part first (text → RAG → LLM) and *optionally* add voice later without throwing away work.

### 4.3 IoT / sensor integration (real + simulated, gracefully degrading)

The system reads from a **pluggable sensor data source** with two implementations:

```
   ┌─────────────────────────────┐
   │   SensorSource (abstract)   │
   ├─────────────────────────────┤
   │  • SimulatedCSVSource       │  ← always available, default
   │  • RealSerialSource         │  ← lab-provided sensor
   │  • MQTTBrokerSource         │  ← lab MQTT broker
   └─────────────────────────────┘
            ↓
   (one config line picks which)
```

**Sensors planned (4 types, matching common smart-home platforms):**
- 🌡️ **Temperature** (°C)
- 💧 **Humidity** (%)
- ⚡ **Energy consumption** (kWh)
- 🚶 **Motion events** (binary or count)

**Default mode (no real hardware):** synthetic 30-day data is generated as CSV.
**Lab mode (if sensor available):** DHT22 (temp + humidity, ~$3) and/or PIR motion sensor (~$2) can be wired to the Pi's GPIO. Energy can be simulated or read from a smart plug's local API.

### 4.4 Example queries the system must handle

| Query | Source it retrieves from |
|-------|--------------------------|
| *"How do I reset my Nest thermostat to factory settings?"* | Nest manual PDF |
| *"What does error code E3 mean on my Hue bulb?"* | Philips Hue manual |
| *"What's the average living-room temperature this week?"* | Sensor log |
| *"Which day last week used the most energy?"* | Sensor log |
| *"Was there motion in the kitchen between 2-3am?"* | Sensor log |
| *"What smart bulbs do I have and how do I pair them?"* | Custom Home FAQ |
| *"I don't have a question, just say hi."* | (No retrieval; casual reply) |

### 4.5 Evaluation strategy

The project will **compare 3+ small open-source LLMs** on the same RAG pipeline:

| Model | Size (Q4) | Why include it |
|-------|-----------|----------------|
| TinyLlama 1.1B Chat | ~700 MB | Fastest baseline |
| Llama 3.2 3B Instruct | ~1.8 GB | Newest Meta model, good quality |
| Phi-3 Mini 3.8B (4k) | ~2.3 GB | Microsoft, strong for size |
| *(optional 4th)* Mistral 7B Q4 | ~4 GB | Quality ceiling, may be too slow |

Measured on each model: **answer accuracy** (vs. a 20-question gold set), **latency**, **RAM usage**.

---

## 5. What the System Will NOT Do (Out-of-Scope)

These are **explicitly excluded** to keep the project focused and feasible:

- ❌ Any cloud LLM API calls (no OpenAI, no Anthropic, no cloud inference)
- ❌ Real voice I/O in the **core deliverable** (stretch goal only)
- ❌ Real smart-home hardware integration (Philips Hue API, etc.) — simulated
- ❌ Multi-user authentication
- ❌ Mobile native app
- ❌ Multi-language support (English only)
- ❌ On-device fine-tuning or federated learning

---

## 6. High-Level Architecture (preview — full version in architecture doc)

```
┌──────────────────────────────────────────────────────────┐
│                    Raspberry Pi 5                          │
│                                                           │
│  ┌──────────┐  ┌─────────────┐  ┌──────────────────┐    │
│  │ Web UI   │←→│ FastAPI     │←→│ llama.cpp server │    │
│  │ (HTML/JS)│  │ backend     │  │ (Phi-3 / Llama)  │    │
│  └──────────┘  └──────┬──────┘  └──────────────────┘    │
│                       │                                    │
│                       ↓                                    │
│              ┌────────────────┐    ┌──────────────────┐  │
│              │ Retriever      │←──→│ Vector store     │  │
│              │ (top-k cosine) │    │ (FAISS / Chroma) │  │
│              └────────┬───────┘    └──────────────────┘  │
│                       │                                    │
│              ┌────────┴────────┐                           │
│              ↓                 ↓                           │
│     ┌──────────────┐   ┌──────────────────┐              │
│     │ Doc index    │   │ Sensor index     │              │
│     │ (manuals,FAQ)│   │ (temp/hum/energy/│              │
│     │              │   │  motion)         │              │
│     └──────────────┘   └──────────────────┘              │
└──────────────────────────────────────────────────────────┘
         ↑                                                   
         │  (sensor data, optional)                         
   ┌─────┴───────┐                                          
   │ Lab sensor  │                                          
   │ / MQTT      │                                          
   └─────────────┘                                          
```

---

## 7. Success Criteria

The capstone is **accepted** when all the following are demonstrable:

1. ✅ User can upload a PDF device manual via the UI and the system indexes it.
2. ✅ User can ask a question and get a **cited** answer in under 5 seconds (text mode).
3. ✅ User can ask a sensor-history question and get a correct answer from local logs.
4. ✅ The system runs **with Wi-Fi disabled** (verified at demo).
5. ✅ At least **3 models** are compared and results are in the report.
6. ✅ Latency, RAM, and accuracy numbers are measured and reported.

---

## 8. Top Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| LLM too slow on Pi | Quantize aggressively (Q4_K_M); pick ≤3B models; benchmark in Week 2 before committing |
| Out of memory | Lazy-load models; pick 1.5–2 GB LLM max; monitor with `/proc/meminfo` |
| Hallucination despite RAG | Strong system prompt, temp=0, mandatory source citation, evaluate on gold set |
| Lab sensor unavailable | Architecture supports simulated fallback — always ship a working demo |
| Voice mode eats time | Voice is a stretch goal, gated behind completion of core RAG + evaluation |

---

## 9. Open Questions for You (Please Answer Before I Write SRS)

I need your decisions on these to write the next document. **Each has a recommended answer — feel free to disagree.**

### Q1. Which Raspberry Pi do you have (or will you buy)?

- ☐ **Pi 5 / 8 GB** *(Recommended — best headroom for LLM)*
- ☐ Pi 5 / 4 GB
- ☐ Pi 4 / 8 GB
- ☐ Pi 4 / 4 GB
- ☐ Don't have one yet — planning to buy

### Q2. Which LLM do you want as the **primary** model (the one shipped in the demo)?

- ☐ **Phi-3 Mini 3.8B Q4** *(Recommended — best quality for size)*
- ☐ Llama 3.2 3B Q4 *(slightly faster)*
- ☐ TinyLlama 1.1B Q4 *(fastest, lowest quality)*

### Q3. What should the knowledge base contain?

- ☐ **2–3 real device manuals (PDF) + 1 custom home FAQ (Markdown)** *(Recommended)*
- ☐ 5–8 real device manuals (no custom FAQ)
- ☐ All custom Markdown (no PDFs)
- ☐ Other (please describe)

### Q4. Which language for the UI and the LLM's responses?

- ☐ **English** *(Recommended — best model support)*
- ☐ Other (please specify)

### Q5. Do you want the system to remember conversation history (multi-turn chat)?

- ☐ **Single-turn only** *(Recommended — simpler, more reliable on Pi)*
- ☐ Multi-turn with last 3–5 messages of context (adds ~1 day of work)
- ☐ Full multi-turn with rolling summary (more work, slower)

### Q6. Demo format — what do you want to show at the final defense?

- ☐ **Live demo on the Pi with Wi-Fi disabled** *(Recommended — most impressive)*
- ☐ Recorded screen video (lower risk)
- ☐ Both: live demo as primary, recorded video as backup

### Q7. How should the LLM be served? (Technical, but affects setup time)

- ☐ **llama.cpp HTTP server** *(Recommended — simplest, mature)*
- ☐ Ollama (easier CLI, but adds a dependency)
- ☐ llama-cpp-python (in-process, no separate server)

### Q8. For the final report, do you want to include a "related work" section comparing TinyRAG to existing tools (PrivateGPT, Ollama, etc.)?

- ☐ **Yes, brief (1 page)** *(Recommended — looks academic)*
- ☐ Yes, detailed (2-3 pages)
- ☐ No, skip it

---

## 10. What I'll Write Next (after you answer the questions above)

1. **SRS (System Requirements Specification)** — converts the scope into testable requirements (FRs, NFRs, acceptance criteria).
2. **Architecture & Tech Stack** — module breakdown, data model, technology choices with rationale.
3. **Database / Storage Design** — vector store layout, metadata schema, sensor data schema.
4. **Roadmap** — week-by-week plan with deliverables and checkpoints.

---

*End of v1 scope. Please review and answer the 8 questions in Section 9 — once I have your answers, I'll draft the SRS.*
