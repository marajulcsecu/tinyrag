# TinyRAG — High-Level Plan & Project Journey

> **Purpose of this document:** Give you a clear, beginner-friendly picture of the **entire journey** from "blank project" to "submitted capstone with a working demo." You don't need to understand every detail yet — this is the **map**. We'll fill in the details as we go, one document at a time.

**Read time:** ~10 minutes
**Audience:** the student (you), any advisor reviewing the project, and any future agent picking up the work.

---

## 0. The Big Picture (30-Second Summary)

You will build **TinyRAG** — a small, private, on-device AI assistant for a smart home. It will run on a Raspberry Pi 5 (or your laptop if the Pi is delayed). It will read PDFs of device manuals and logs of home sensors, and answer questions about them using a small local LLM. You'll compare 3 different LLMs to find the best one. At the end, you'll demo it live and write a professional report.

**That's it.** Everything else in this plan is just breaking that vision into achievable steps.

---

## 1. The Journey, Phase by Phase

Think of the project as **7 phases**. Each phase has a clear output, and you review/approve before moving on. **No skipping phases** — this is how professional software projects are run.

```
   ┌──────────────────────────────────────────────────────────────┐
   │                                                                │
   │   PHASE 0: PLANNING (we are here)                             │
   │   ↓                                                            │
   │   PHASE 1: REQUIREMENTS  (SRS)                                │
   │   ↓                                                            │
   │   PHASE 2: DESIGN  (Architecture + DB + Tech Stack)           │
   │   ↓                                                            │
   │   PHASE 3: SETUP  (Environment, dependencies, models)         │
   │   ↓                                                            │
   │   PHASE 4: BUILD  (Write the code, module by module)          │
   │   ↓                                                            │
   │   PHASE 5: TEST  (Unit tests, integration, evaluation)         │
   │   ↓                                                            │
   │   PHASE 6: DEPLOY  (Move to Pi, run benchmarks)               │
   │   ↓                                                            │
   │   PHASE 7: REPORT & DEMO  (Write up, record, present)         │
   │                                                                │
   └──────────────────────────────────────────────────────────────┘
```

Let me explain what happens in each phase, **why** it happens, and **what you'll produce**.

---

## 2. Phase 0 — Planning (Current Phase)

**What we're doing right now.**

**Why it matters:** A capstone without a plan is just a coding session. With a plan, you have a target, a definition of "done," and a way to measure progress.

**Documents produced in this phase:**
- ✅ `AGENT.md` — project context handoff file
- ✅ `docs/00_high_level_plan.md` — this file
- ✅ `docs/01_project_scope_v1.md` — first scope draft
- ✅ `docs/01_project_scope_v2.md` — refined, decisions-locked scope
- ✅ `docs/laptop_fallback/README.md` — laptop path notes

**Documents still to produce in this phase:** (none — Phase 0 is done after you approve v2 scope)

**Your action items:**
1. Read this plan.
2. Read `01_project_scope_v2.md`.
3. Tell me: "approved, proceed to SRS" OR "I want to change X, Y, Z."

---

## 3. Phase 1 — Requirements (SRS)

**What's a SRS?**
A **System Requirements Specification** turns the scope (what we *want*) into a list of **testable requirements** (what the system *must do*). Each requirement gets an ID like `FR-1` or `NFR-3` and we can later check "did the system meet FR-1?" with a yes/no answer.

**Why it matters:** When you write code, you need a checklist. The SRS is that checklist. Without it, you code randomly and don't know when you're "done."

**What you'll learn in this phase:** How to write professional software requirements. This is a skill that lands you jobs.

**Documents produced:**
- `docs/02_srs_v1.md` — System Requirements Specification

**Example requirement (you don't need to write it, just understand the style):**
> **FR-1:** The system shall accept PDF, TXT, and Markdown files via a CLI command or a UI upload.
> **NFR-5:** Idle RAM (after model load) shall not exceed 1.5 GB.

**Your action items:** Read the SRS, confirm the requirements match your vision.

---

## 4. Phase 2 — Design (Architecture + DB + Tech Stack)

This is the most important **thinking** phase. We don't write code here — we draw boxes and arrows and decide what technology goes where.

### 4.1 Architecture Design

**What:** A document that breaks the system into **modules** (separate, swappable pieces) and shows how data flows between them.

**Example (simplified, not the final version):**
```
   Web UI  →  FastAPI backend  →  Retriever  →  Vector Store
                  ↓
                  LLM Client  →  llama.cpp server
                  ↓
                  Streamed answer back to UI
```

**Why it matters:** Without architecture, you write spaghetti code. With it, you can swap the LLM, change the vector store, or add voice input **without rewriting everything**. This is the **clean architecture** principle you asked for.

**Document produced:** `docs/03_architecture_v1.md`

### 4.2 Database / Storage Design

**What:** Defines exactly what data is stored and where.

- **Vector store:** stores embeddings (numbers) for each text chunk
- **Metadata DB (SQLite):** stores `chunk_id → source_file, page_number, text` mapping
- **Sensor log:** CSV files for temperature, humidity, energy, motion

**Why it matters:** When you have 1000 chunks, you need to know which chunk came from which PDF page. The metadata DB makes that possible.

**Document produced:** `docs/04_database_design_v1.md`

### 4.3 Tech Stack

**What:** A pinned list of every library, tool, and version we'll use. Example: `llama.cpp v0.2.79`, `FastAPI 0.115`, `sentence-transformers 3.0`, `FAISS 1.8`.

**Why it matters:** Pinned versions = reproducible project. "It worked on my machine" stops being a problem.

**Document produced:** `docs/05_tech_stack_v1.md`

---

## 5. Phase 3 — Setup (Environment & Dependencies)

**What:** Install Python, install llama.cpp, download models, install Python libraries. Verify everything runs.

**Why it matters:** Setting up an edge-AI environment is the trickiest single step. llama.cpp needs to be compiled for your specific CPU (Pi 5 vs. laptop have different flags). Models are 1–2 GB each. If this phase is rushed, nothing else works.

**Time cost:** ~1 day on the laptop, ~1 day on the Pi.

**Outputs:**
- Working `setup.sh` that installs everything
- Downloaded model files in `models/`
- A "hello world" test: the LLM can answer "Hello" in < 1 second

**Your action items:** Run `setup.sh`, run a test query, confirm it works.

---

## 6. Phase 4 — Build (Code, Module by Module)

This is the **biggest** phase. We build the system piece by piece, in dependency order:

```
   ┌──────────────────────────────────────────────┐
   │  Step 1: Document parser (PDF → text)        │
   │  Step 2: Chunker (text → chunks)             │
   │  Step 3: Embedder (chunks → vectors)         │
   │  Step 4: Vector store (save + search)        │
   │  Step 5: LLM client (talk to llama.cpp)      │
   │  Step 6: Prompt builder (chunks + question)  │
   │  Step 7: RAG pipeline (end-to-end)           │
   │  Step 8: Sensor source (abstract + concrete) │
   │  Step 9: FastAPI routes (REST API)           │
   │  Step 10: Web UI (HTML + JS chat box)        │
   │  Step 11: System status panel                │
   │  Step 12: Document management UI             │
   └──────────────────────────────────────────────┘
```

**Why this order matters:** Each step builds on the previous one. You can test each step in isolation before moving to the next. If Step 4 (vector store) is broken, you find out at Step 4 — not at Step 12.

**Clean architecture payoff:** Each step is a separate Python file with a clear interface. You can test Step 7 (RAG) without Step 10 (UI) by using a CLI script.

**Outputs:**
- All code in `src/tinyrag/`
- A working CLI demo: `python scripts/ingest.py path/to/manual.pdf` then `python scripts/ask.py "How do I reset?"`

---

## 7. Phase 5 — Test (Quality & Evaluation)

**Three kinds of testing:**

### 7.1 Unit tests
- Test each module in isolation (chunking splits correctly, parser extracts PDF text, etc.)
- Run with `pytest`
- Goal: 60%+ coverage of core modules

### 7.2 Integration test
- Upload a manual → ask a question → get a correct answer
- Verifies the whole pipeline works end-to-end

### 7.3 Evaluation (the most important)
- Create a **20-question gold set** (10 from manuals, 10 from sensor data) with known correct answers
- Run each question through the system
- Manually judge each answer: ✅ correct & cited / ⚠️ partially correct / ❌ wrong
- Compare results across 3+ models
- Result: a table like this in your report:

| Model | Accuracy | Avg latency (Pi 5) | Peak RAM |
|-------|----------|-------------------|----------|
| TinyLlama 1.1B | 60% | 1.8 s | 1.4 GB |
| Llama 3.2 3B | 80% | 3.2 s | 2.3 GB |
| **Phi-3 Mini 3.8B** | **90%** | 4.5 s | 2.8 GB |

**Why it matters:** This table is the **#1 thing your capstone panel will ask about.** "How did you compare models? What were the trade-offs?" This table answers that.

**Outputs:**
- `tests/` — unit test suite
- `reports/accuracy_per_model.csv` — evaluation results
- `reports/latency.csv`, `reports/ram_usage.csv` — benchmarks

---

## 8. Phase 6 — Deploy (Move to Raspberry Pi)

**What:** Copy the working code from your laptop to the Pi. Change `config.yaml` from `deployment.target: laptop` to `deployment.target: raspberry_pi`. Run.

**Why this should be easy:** Because of clean architecture, this is mostly a config change. The only real work is re-compiling llama.cpp with Pi-specific flags.

**Outputs:**
- Working demo on the Pi
- Measured Pi performance numbers

---

## 9. Phase 7 — Report & Demo

### 9.1 Final Report (PDF, 15–25 pages)

Standard capstone structure:
1. Abstract
2. Introduction (problem + motivation)
3. Related Work (1-page comparison vs. PrivateGPT, Ollama, etc.)
4. System Design (architecture diagram + module descriptions)
5. Implementation (key code walkthroughs)
6. Evaluation (the comparison table from Phase 5)
7. Discussion (what worked, what didn't, limitations)
8. Conclusion + Future Work
9. References

### 9.2 Demo

- **Primary:** live demo on the Pi, with Wi-Fi disabled. Ask 5–10 questions live. Show source citations.
- **Backup:** recorded screen video of the same.
- **Slides:** 10–15 slides covering the journey + results.

**Why both live and recorded:** Live demos fail (Pi crashes, network glitches, audience questions derail). Always have a recording. This is what professional engineers do.

---

## 10. The Document Trail (in order)

For your reference, here is the **complete document trail** we'll build, in the order we'll build it:

```
docs/
├── 00_high_level_plan.md       ← ✅ You are here
├── 01_project_scope_v1.md      ← ✅ Done
├── 01_project_scope_v2.md      ← ✅ Done
├── 02_srs_v1.md                ← ⏳ Next (after you approve scope)
├── 03_architecture_v1.md       ← ⏳ After SRS
├── 04_database_design_v1.md    ← ⏳ After architecture
├── 05_tech_stack_v1.md         ← ⏳ After DB
├── 06_roadmap_v1.md            ← ⏳ After tech stack
├── evaluation/                 ← ⏳ After roadmap
│   ├── gold_set.md
│   └── scoring_rubric.md
└── laptop_fallback/
    └── README.md               ← ✅ Done
```

---

## 11. The Big Idea (in case you forget the goal)

You are building a **smart home assistant that:**
1. **Lives on a small computer** (Raspberry Pi 5, your laptop if needed).
2. **Reads PDF manuals and sensor data** from your home.
3. **Uses a small AI model** to answer questions about them.
4. **Never sends your data to the cloud.**
5. **Is built with professional software engineering practices** — modular, tested, documented, reproducible.
6. **Is compared against 2 other models** to show you made a thoughtful choice.
7. **Has a real demo** at the end.

That's it. Everything in the rest of the project is just turning that idea into reality, one clean step at a time.

---

## 12. What You Need To Do Right Now

1. **Read this plan** (you're doing it ✅).
2. **Read `01_project_scope_v2.md`** to confirm the scope matches your vision.
3. **Tell me one of:**
   - ✅ "Approved — proceed to SRS"
   - 🔄 "I want to change X" (tell me what)
   - ❓ "I have a question about Y" (ask it)

Once you approve, I write the SRS. Then we keep moving forward, one document at a time, with your review between each.

**This is how professional software is built. You're doing it right.**

---

*End of high-level plan. Welcome to the journey.*
