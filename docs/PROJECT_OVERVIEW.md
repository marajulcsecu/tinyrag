# TinyRAG — Project Overview (Read This First)

> **Who this document is for.** You are a team member (or reviewer) who did
> **not** write the code, but needs to understand *the whole project* well
> enough to explain any part of it out loud — how it works, what technology it
> uses, and **why** each choice was made. Read it top to bottom once; after
> that, the **[Advisor Q&A Cheat-Sheet](#14-advisor-qa-cheat-sheet)** at the end
> is your quick-revision page.
>
> No prior knowledge of RAG, embeddings, or LLMs is assumed. Every term is
> explained the first time it appears, and there is a **[Glossary](#13-glossary)**.

**Project:** TinyRAG — Retrieval-Augmented Generation for Edge Devices
**Author / lead developer:** Marajul Haque · **Advisor:** Abu Nowshed Chy
**Type:** Undergraduate capstone project

---

## Table of Contents

1. [The one-paragraph pitch](#1-the-one-paragraph-pitch)
2. [The problem we set out to solve](#2-the-problem-we-set-out-to-solve)
3. [What is RAG? (in plain English)](#3-what-is-rag-in-plain-english)
4. [Big-picture architecture](#4-big-picture-architecture)
5. [The journey of a single question](#5-the-journey-of-a-single-question)
6. [Stage 1 — Retrieve (finding the evidence)](#6-stage-1--retrieve-finding-the-evidence)
7. [Stage 2 — Augment (building the grounded prompt)](#7-stage-2--augment-building-the-grounded-prompt)
8. [Stage 3 — Generate (writing the answer)](#8-stage-3--generate-writing-the-answer)
9. [How documents get in: the ingestion pipeline](#9-how-documents-get-in-the-ingestion-pipeline)
10. [The technology stack, and why we chose each piece](#10-the-technology-stack-and-why-we-chose-each-piece)
11. [How the code is organised (design principles)](#11-how-the-code-is-organised-design-principles)
12. [Deployment: laptop and Raspberry Pi 5](#12-deployment-laptop-and-raspberry-pi-5)
13. [Glossary](#13-glossary)
14. [Advisor Q&A cheat-sheet](#14-advisor-qa-cheat-sheet)

---

## 1. The one-paragraph pitch

**TinyRAG is a private, offline question-answering assistant that runs entirely
on a small computer — a Raspberry Pi 5 — with no internet connection and no
cloud services.** You give it your own documents (PDF manuals, text files,
Markdown notes). You ask it questions in plain English. It finds the relevant
passages in *your* documents and uses a small local AI model to write an answer
**that cites exactly which document and page it came from**. If your documents
do not contain the answer, it says so instead of making something up. Because
everything runs on the device, your documents never leave the machine — you
could unplug the Wi-Fi and it still works.

The **"Tiny"** has two meanings:

1. **Tiny models** — a 3-billion-parameter language model (vs. the 100+ billion
   of ChatGPT-class models) and an 80 MB embedding model.
2. **Tiny hardware** — it targets a Raspberry Pi 5 (8 GB RAM), a credit-card-sized
   computer, not a data-centre GPU.

---

## 2. The problem we set out to solve

Large language models like ChatGPT are powerful, but for many real situations
they have three problems:

| Problem | Why it matters | How TinyRAG answers it |
|---|---|---|
| **Privacy** | Sending your home manuals, medical notes, or company documents to a cloud API means they leave your control. | Everything runs **on-device**. Documents never leave the machine. |
| **Cost & connectivity** | Cloud APIs bill per question and need a constant internet connection. Edge devices are often offline. | **Zero cloud calls**, zero API bills. Works with Wi-Fi off. |
| **Hallucination** | A plain LLM will confidently invent facts that sound right but are wrong. | TinyRAG only answers **from documents you provided**, cites its source, and **refuses** when it has no evidence. |

**The tool is general-purpose: it answers questions from any document you give
it**, whatever the subject — a textbook, a manual, a paper, a policy. Nothing in
the pipeline is tied to a domain.

The "**Edge**" in the framing refers to the target *hardware*, not a subject
area: a Raspberry Pi is a classic edge computer, chosen to prove the whole
pipeline fits on low-power, low-cost hardware. Because such boards are often
deployed alongside sensors, the project *additionally* includes a pluggable path
for summarising time-series readings (temperature, humidity, energy, motion).
That sensor capability is built but **deferred** — our advisor will guide that
part later — and it is optional, not a requirement of the system.

---

## 3. What is RAG? (in plain English)

**RAG** stands for **Retrieval-Augmented Generation**. It is a technique for
making a language model answer from a specific, trusted set of documents instead
of from its own memory.

### The open-book exam analogy

Imagine two students taking an exam about a car's user manual:

- **Student A (a plain LLM)** answers from memory. They have read millions of
  documents during training, so they *sound* confident — but they have never
  seen *this specific car's* manual, so they guess, and sometimes they are
  confidently wrong. This is **hallucination**.

- **Student B (RAG)** is allowed to **open the manual**, flip to the exact page
  that discusses the question, read that page, and then answer *based on what the
  page says* — quoting the page number. If the manual does not cover the
  question, Student B writes "the manual does not say."

TinyRAG is Student B. The three words in "Retrieval-Augmented Generation" are
literally the three steps:

```
   RETRIEVE                    AUGMENT                      GENERATE
   ─────────                   ───────                      ────────
"Find the pages in       "Put those pages in front     "Write an answer using
 my documents that        of the AI, with the           ONLY those pages, and
 relate to this           instruction: answer only      cite where each fact
 question."               from this, or say you         came from."
                          don't know."
```

Everything else in this project is machinery to do those three steps well, fast,
and on a tiny computer.

---

## 4. Big-picture architecture

Here is the whole system at a glance. Each box is a real module in the code.

```
                          ┌─────────────────────────────┐
                          │   USER'S BROWSER             │
                          │   chat UI at :8000           │
                          └──────────────┬──────────────┘
                                         │  HTTP / Server-Sent Events
                          ┌──────────────▼──────────────┐
                          │   FastAPI WEB SERVER          │   src/tinyrag/api/
                          │   (uvicorn, async)            │   + main.py
                          │   /api/query  /api/documents  │
                          │   /api/status /  /admin       │
                          └──────────────┬──────────────┘
                                         │
             ┌───────────────────────────┼───────────────────────────┐
             │                           │                           │
      ┌──────▼───────┐            ┌──────▼───────┐            ┌───────▼───────┐
      │  RETRIEVE     │            │  AUGMENT      │            │  GENERATE      │
      │  Retriever    │──chunks──► │ PromptBuilder │──prompt──► │ LlamaCppClient │
      │ core/         │            │ core/         │            │ generation/    │
      │ retriever.py  │            │prompt_builder │            │ llm_client.py  │
      └──────┬───────┘            └──────────────┘            └───────┬───────┘
             │                                                        │ HTTP
     ┌───────┴────────┐                                     ┌─────────▼─────────┐
     │                │                                     │  llama-server      │
┌────▼─────┐   ┌──────▼──────┐                              │  (llama.cpp)       │
│ Embedder │   │ FAISS stores│                              │  runs the 3B model │
│ MiniLM   │   │ doc + sensor│                              │  on :8080, CPU     │
│ 384-dim  │   │ (vectors)   │                              └───────────────────┘
└──────────┘   └──────┬──────┘
                      │
               ┌──────▼──────┐
               │  SQLite DB   │   storage/metadata.py
               │ documents,   │   (the text of each chunk,
               │ chunks,      │    page numbers, filenames,
               │ query_log    │    query history)
               └─────────────┘
```

**Two things to notice:**

1. **The LLM runs as a separate process** (`llama-server`, port 8080). Our Python
   app talks to it over local HTTP, exactly like it would talk to a cloud API —
   except it's on the same machine, so nothing leaves the device.
2. **There are two databases working together.** FAISS stores the *meaning* of
   each chunk as a vector (for "find similar text"); SQLite stores the *actual
   text, filename, and page number* of each chunk (for "show me the real
   passage"). Retrieval uses both: FAISS to rank, SQLite to fetch.

---

## 5. The journey of a single question

Let's trace one real question end-to-end. This is the single most useful thing
to understand — if you can narrate this, you understand the project.

> **User asks:** *"What processor does the Raspberry Pi 5 use?"*
> *(Assume the Raspberry Pi 5 product brief PDF has been ingested.)*

```
 1. BROWSER      User types the question, hits Enter.
      │          POST /api/query?stream=true   {"query": "What processor..."}
      ▼
 2. API          routes_query.py receives it, calls the Retriever.
      │
      ▼
 3. RETRIEVE     • Embed the question → a 384-number vector (its "meaning").
      │          • Search the document FAISS index for the closest chunks.
      │          • Keyword-rerank: chunks containing "processor"/"raspberry"
      │            get a small boost.
      │          • Keep chunks above the relevance threshold, cap at 5.
      │          → Returns 1 strong chunk: PDF page 2, score 0.807.
      ▼
 4. AUGMENT      prompt_builder.py assembles a chat prompt:
      │            SYSTEM: "Answer ONLY from the context. Cite [1],[2]. If
      │                     it's not there, say you don't have enough info."
      │            USER:   "[1] (raspberry-pi-5-brief.pdf, p.2) The Raspberry
      │                     Pi 5 is built around a Broadcom BCM2712 ...
      │                     Question: What processor does the Raspberry Pi 5 use?"
      │          It checks the prompt fits the token budget (drops extra
      │          chunks if needed).
      ▼
 5. GENERATE     llm_client.py sends the prompt to llama-server (the 3B model).
      │          Temperature 0.0 → deterministic. The model reads the context
      │          and writes an answer, streamed token-by-token.
      ▼
 6. API          Wraps the answer + citations + timings into an Answer object,
      │          streams tokens to the browser as they arrive (SSE), logs the
      │          query to SQLite.
      ▼
 7. BROWSER      Shows: "The Raspberry Pi 5 uses a Broadcom BCM2712 2.4GHz
                  quad-core 64-bit Arm Cortex-A76 CPU."   [1] p.2  (score 0.807)
```

If the user had asked *"What is the capital of France?"* — a question **none of
the ingested documents answer** — step 3 would return no chunk above the
threshold, and the model, following its system-prompt instruction, would reply:
*"I don't have enough information in the provided documents."* **That refusal is
a feature, not a bug** — it is what makes the answers trustworthy.

---

## 6. Stage 1 — Retrieve (finding the evidence)

**Goal:** given a question, return the handful of document passages ("chunks")
most likely to contain the answer.
**Code:** `src/tinyrag/core/retriever.py`

### 6.1 Embeddings: turning text into meaning

A computer cannot compare the *meaning* of two sentences directly. So we convert
every piece of text into a list of numbers called an **embedding** (a "vector").
The model that does this is **all-MiniLM-L6-v2**, a small 80 MB neural network.
It outputs **384 numbers** per text.

The key property: **texts with similar meaning get similar vectors.** "How do I
reset the thermostat?" and "steps to factory-reset the device" land close
together in this 384-dimensional space, even though they share almost no words.

We compare two vectors using **cosine similarity** — a score from -1 to 1 where
1 means "same direction / same meaning" and 0 means "unrelated." Because we
**normalise** every vector to length 1, cosine similarity is just their dot
product, which is exactly what FAISS computes.

> **Critical detail (a bug we hit and fixed):** there are *two* embedders in the
> code — the **real** MiniLM one, and a **fake** one used only for fast tests
> that turns text into a SHA-256 hash. The fake embedder's vectors have **no
> meaning** ("dog" and "cat" get unrelated hashes). If you ever ingest documents
> with the fake embedder by mistake, retrieval silently returns garbage while
> *looking* like it works. **Real ingestion must always use `--embedder real`.**

### 6.2 FAISS: the vector search engine

**FAISS** (Facebook AI Similarity Search) is a library that stores thousands of
vectors and, given a query vector, instantly finds the closest ones. We use the
simplest FAISS index, **`IndexFlatIP`** ("flat, inner-product"), wrapped in an
`IndexIDMap2` so each vector keeps a stable ID even after deletes.

- **"Flat"** means it compares against *every* stored vector — exact, no
  approximation. For our corpus size (tens to a few thousand chunks) that is
  fast (sub-millisecond) and avoids the accuracy loss of approximate indexes.
- **"IP" (inner product)** on our normalised vectors = cosine similarity, so we
  get "most similar meaning" for free.

There are **two** FAISS indexes: one for **documents**, one for **sensor
summaries**. They are searched separately and the results merged.

### 6.3 The keyword-overlap rerank (a "poor man's BM25")

Pure vector search has a known weakness: on short, technical questions it
sometimes ranks a chunk that *mentions* a term above the chunk that *defines*
it. A real example from our Nest thermostat corpus:

> Question: *"What is the ErP directive?"*
> Vector search put the **table-of-contents** chunk (which just lists "ErP … 26")
> **above** the actual page 26 that defines it.

The fix is a cheap lexical boost layered on top of the vector score:

1. Extract the distinctive (non-stopword) words from the question, e.g.
   `{"erp", "directive"}`.
2. For each candidate chunk, count how many of those words appear **as whole
   words** in the chunk's text.
3. Add **+0.10 per matching word**, plus a **+0.20 bonus if the chunk contains
   *all* of them**.

So the TOC chunk (matches only "erp": +0.10) loses to the real definition
(matches "erp" **and** "directive": +0.10 + +0.10 + +0.20 = +0.40). This lifted
the correct chunk from rank #18 to #1. It is deliberately **not** a full BM25
implementation — just a small, well-understood constant that fixes the most
common failure mode without adding a dependency.

> **Why "whole words" matters:** an earlier version matched substrings, so "rag"
> matched inside "sto**rag**e" and wrongly boosted an unrelated chunk. Matching
> whole words only fixed it.

To make sure the rerank has good candidates to work with, retrieval
**over-fetches** — it asks FAISS for more chunks than it will keep (5×, or the
*entire* store on a small corpus), reranks them, then keeps the top few. This is
cheap because scanning a few thousand vectors is trivial, and the final prompt
is still capped.

### 6.4 The relevance threshold (and the small-corpus twist)

After scoring, chunks below a **similarity threshold** are dropped — this is the
gate that lets the system **refuse** when nothing is relevant. The default is
**0.3**, tuned for large corpora (1000+ chunks).

But on a **small corpus** (≤ 50 chunks — our demo situation), MiniLM's raw
scores are noisy and run low (often 0.04–0.15) even for correct matches. A strict
0.3 would wrongly drop good chunks. So the retriever detects small corpora and
lowers the floor to **0.15** — permissive enough to surface the right chunk, but
still high enough to drop obvious noise. The LLM's own refusal instruction is the
final safety net.

Two constants govern this (both in `retriever.py`): `SMALL_CORPUS_MAX_CHUNKS = 50`
and `SMALL_CORPUS_THRESHOLD = 0.15`.

### 6.5 Sensor-query detection (the deferred optional sensor path)

Before searching, the retriever scans the question for **sensor keywords**
("temperature", "humidity", "kWh", "yesterday", "last week", …). If it finds
any, it *also* searches the sensor index and merges those hits. If not, it skips
the sensor search entirely to save work. This is how a question like *"What was
the temperature yesterday?"* would reach sensor data — the sensor **wiring**
itself is deferred pending advisor guidance, but the retrieval path exists.

### 6.6 Retrieve — summary

```
question ─► embed ─► search doc FAISS (over-fetch)
                  └─► (if sensor keywords) search sensor FAISS
                          │
                          ▼
                merge hits, keep best score per chunk
                          │
                          ▼
                keyword-overlap rerank (+0.10/word, +0.20 all)
                          │
                          ▼
                drop chunks below threshold (0.3, or 0.15 small corpus)
                          │
                          ▼
                fetch real text from SQLite, cap at 5 chunks ─► evidence
```

---

## 7. Stage 2 — Augment (building the grounded prompt)

**Goal:** package the question + the retrieved evidence into a single, carefully
worded instruction for the language model.
**Code:** `src/tinyrag/core/prompt_builder.py`

A "prompt" for a chat model is a list of messages. We build exactly two:

**1. The system message** (the rules the model must follow). Ours says, in
short:

> *"You are a helpful assistant that answers questions about the user's
> documents. Answer using ONLY the
> numbered context blocks below. Use complete sentences. If the context does not
> contain the answer, reply exactly: **I don't have enough information in the
> provided documents.** When you use a context block, cite it by its bracketed
> number — but ONLY cite numbers that actually appear."*

Every clause is there for a reason:
- *"ONLY … context blocks"* → this is what makes it **grounded** (RAG, not a chatbot).
- *the exact refusal sentence* → gives us a stable string the app can detect as
  "low-confidence / refused."
- *"ONLY cite numbers that actually appear"* → stops the small model from
  inventing citations like `[2] [3] … [25]` to pad a list (a real bug we saw).

**2. The user message** — the numbered evidence, then the question:

```
[1] (raspberry-pi-5-brief.pdf, p.2) The Raspberry Pi 5 is built around a
    Broadcom BCM2712 ...
[2] (raspberry-pi-5-brief.pdf, p.3) Power is supplied via USB-C ...

Question: What processor does the Raspberry Pi 5 use?
```

Numbering each chunk `[1]`, `[2]`, … and prefixing the `(filename, p.X)` is what
makes **citations** possible — the model refers back to `[1]` and the app maps
that to the real source.

### The token budget

A language model can only read so much text at once — its **context window**
(4096 tokens for our model; a "token" is roughly ¾ of a word). We reserve **512
tokens for the answer**, leaving ~3584 for the prompt. The builder counts tokens
(with `tiktoken`) and **greedily packs** the highest-ranked chunks until the
budget is full, dropping the rest. It records how many it used and dropped, so
the UI can be honest about it. This prevents a subtle bug where the model is
told to cite `[3]` but never actually received chunk 3 because it overflowed.

---

## 8. Stage 3 — Generate (writing the answer)

**Goal:** run the assembled prompt through the local language model and stream
the answer back.
**Code:** `src/tinyrag/generation/llm_client.py` + the `llama-server` process.

### The model: Llama 3.2 3B Instruct

We run **Meta's Llama 3.2 3B Instruct**, quantised to **Q4_K_M** (~1.8 GB on
disk). "Quantised" means the model's numbers are compressed from 16 bits to
~4 bits, which shrinks it ~4× and makes it run on a CPU with only a small
quality cost. "Instruct" means it is fine-tuned to follow instructions (like our
"answer only from context" rule).

> **Why not Phi-3 Mini?** We *started* with Microsoft's Phi-3 Mini, but that
> specific GGUF file emits **garbage tokens once the prompt exceeds ~2048
> tokens** — and RAG prompts routinely run 2500–3600 tokens. Llama 3.2 3B stays
> coherent at those sizes on the exact same server binary, so we switched
> (2026-07-23). Phi-3 is kept only as a comparison model for evaluation. This is
> a good example of an evidence-driven engineering decision.

### The server: llama.cpp

The model is run by **`llama-server`**, part of the **llama.cpp** project — a
mature, CPU-optimised C++ engine for running LLMs. We compile it with
**OpenBLAS** (a fast linear-algebra library) for speed. It exposes an
**OpenAI-compatible HTTP API** (`/v1/chat/completions`), so our Python client
talks to it the same way you would talk to the real OpenAI API — but it is
100% local. **No GPU** is used (`gpu_layers: 0`): the Pi has no usable GPU, and
our laptop's integrated GPU isn't supported by the engine's build, so CPU +
OpenBLAS is the correct choice for both.

### Two properties that matter

- **Determinism.** We set **temperature = 0.0**, which makes the model always
  pick its single most-likely next word. The same question yields the **same
  answer every time** — essential for a reproducible demo and for evaluation.
- **Streaming (SSE).** The client requests `stream=true` and forwards each token
  to the browser via **Server-Sent Events** as it is produced, so the user sees
  the answer "type out" instead of waiting in silence for the whole thing. The
  app also sends an early `status` frame ("found 1 source…") before the slow
  CPU part begins, so the UI never looks frozen.

### The final answer object

The pipeline ends in an `Answer` object (`core/answer.py`) that bundles
everything the UI and evaluators need:

- `text` — the answer (or the refusal sentence)
- `citations` — a list of `{number, source, page, score, preview}`
- `model_name`, `used_sensor_idx`, `top_score`
- token counts (`prompt_tokens`, `completion_tokens`, …)
- timings in milliseconds (`duration_retrieve_ms`, `duration_llm_ms`, `duration_total_ms`)
- `is_refusal` — true if the text is the "I don't have enough information" sentence

This is why the demo can show *"answered from page 2, score 0.807, took 78 s"* —
those numbers are all carried in the `Answer`.

---

## 9. How documents get in: the ingestion pipeline

Everything above assumes documents are already searchable. Here is how a raw PDF
becomes searchable chunks. This runs when you upload a file or run the ingest
script. **Code:** `src/tinyrag/ingestion/` + `src/tinyrag/core/chunker.py`.

```
  raw file (PDF / TXT / MD)
        │
        ▼
  1. PARSE        parsers.py extracts clean text.
        │           • PDF  → PyMuPDF (fitz), reading order preserved per page
        │           • TXT  → read as UTF-8
        │           • MD   → strip front-matter, keep link text
        ▼
  2. CHUNK        chunker.py splits text into ~400-token pieces with 50-token
        │         overlap, trimming to sentence boundaries so ideas stay whole.
        ▼
  3. EMBED        embedder.py turns each chunk into a 384-number vector (MiniLM).
        │
        ▼
  4. STORE        • vector → FAISS index (for similarity search)
                  • text + filename + page + token count → SQLite (for retrieval)
```

**Design decisions worth knowing:**

- **PDF parsing uses PyMuPDF, not pdfplumber.** We switched (2026-07-22) because
  pdfplumber read **two-column** pages straight across the gutter, interleaving
  the left and right columns and scrambling sentences — which poisoned
  retrieval. PyMuPDF's `get_text("blocks", sort=True)` respects column reading
  order. There is **no OCR** — scanned image-only PDFs are rejected with a clear
  error rather than returning gibberish.
- **Chunking is token-based, not character-based.** We count with `tiktoken`
  (the same tokenizer family GPT models use) so chunk sizes are predictable in
  the same units the model's context window is measured in. ~400 tokens is
  1–2 paragraphs — big enough to be self-contained, small enough to fit several
  in the context window. The **50-token overlap** (~12%) stops a sentence that
  straddles a boundary from being cut in half.
- **Two document types today:** `manual` (the default — any PDF/TXT/MD) and
  `faq`. A third, `sensor_summary`, is produced by the sensor path. These are
  enforced by a database `CHECK` constraint.

---

## 10. The technology stack, and why we chose each piece

This table is your reference for *"why did you use X?"* questions. Every choice
was made for the constraints of a tiny, offline, CPU-only device.

| Layer | Technology | Why this one |
|---|---|---|
| **Language model** | **Llama 3.2 3B Instruct**, Q4_K_M GGUF (~1.8 GB) | Small enough for 8 GB RAM, coherent at our 2500–3600-token RAG prompts (Phi-3 Mini was not), instruction-tuned, permissive license. |
| **LLM engine** | **llama.cpp** (`llama-server`) + OpenBLAS | The standard for running LLMs on CPU. Mature, fast, exposes an OpenAI-compatible API so our code is portable. No GPU needed. |
| **Embeddings** | **sentence-transformers `all-MiniLM-L6-v2`** (384-dim, 80 MB) | Tiny, fast on CPU, well-proven quality for English. Big enough model would be slower without meaningful gains at our corpus size. |
| **Vector store** | **FAISS `IndexFlatIP`** | Exact cosine search, no approximation error, sub-millisecond at our scale, zero external service. |
| **Metadata store** | **SQLite** (WAL mode) | A single-file database with no server process — perfect for an edge device. Holds chunk text, page numbers, and query history. |
| **PDF parsing** | **PyMuPDF (fitz) 1.24.10** | Column-aware reading order; replaced pdfplumber, which scrambled two-column pages. |
| **Tokenizer** | **tiktoken** (`cl100k_base`) | Industry-standard, fast, lets us count tokens without loading a heavyweight model. |
| **Web framework** | **FastAPI** + **Uvicorn** | Async, automatic API docs, native Server-Sent Events for token streaming. |
| **Streaming** | **Server-Sent Events (SSE)** via `sse-starlette` | Simplest way to stream tokens to a browser; no WebSocket complexity. |
| **UI** | Plain **HTML + vanilla JavaScript** (Jinja2 templates) | No React/build step — keeps the device footprint tiny and the code auditable. |
| **Config** | A single **`config.yaml`** + Pydantic validation | One place controls all runtime behaviour; invalid config fails loudly at startup. |
| **Logging** | **structlog** (JSON) | Structured, machine-parseable logs. |
| **Tests** | **pytest** (~1,300 test functions) | Standard; the pure-logic core is tested with in-memory fakes, no model needed. |
| **Lint** | **ruff** | Fast, catches errors and style issues in CI. |

**The one-sentence version:** *every layer is the smallest, simplest, offline-capable
option that meets the quality bar — because the target is a Raspberry Pi, not a
server.*

---

## 11. How the code is organised (design principles)

You don't need to read the code to explain *how it's structured*. Three ideas
carry the whole design.

### 11.1 Everything external hides behind a "Protocol"

A **Protocol** in Python is an interface — a promise that "any object with these
methods can be used here." TinyRAG hides every external dependency behind one:

- the LLM behind `LLMClient`,
- the embedder behind `EmbeddingModel`,
- the vector store behind `VectorStore`,
- each sensor source behind `SensorSource`.

**Why this matters:** you can swap any component without touching the rest. Want
to try a different embedding model, or FAISS → a different vector DB, or the real
LLM → a fake one for testing? Change one line at the "wiring" point. This is
also *why the code is so heavily testable* — tests inject fake implementations
(a fake LLM that returns canned text, a fake embedder, an in-memory store) and
exercise the real pipeline logic **without loading any model or hitting any
network**. That's how ~1,300 tests run in minutes.

### 11.2 The "core" has no input/output

The `src/tinyrag/core/` package (retriever, prompt builder, chunker, answer) is
**pure logic** — it never opens a file or a socket itself. It only calls
Protocols. All the messy I/O (reading FAISS, querying SQLite, HTTP to the model)
lives in the outer layers (`ingestion/`, `storage/`, `generation/`, `api/`).
Dependencies point **one way**: outer layers depend on core, never the reverse.
This keeps the important decision-making logic simple and independently testable.

### 11.3 One wiring point (the "composition root")

All the concrete pieces are assembled in exactly one place: the `create_app()`
factory and its startup "lifespan" in `src/tinyrag/main.py`. It builds the
embedder, both FAISS stores, the SQLite store, the LLM client, then composes
them into the `Retriever` and `PromptBuilder`, and stashes them for the request
handlers to use. Because it's a **factory function** (not a global), tests can
create a fully-wired app with fakes injected. This is the standard "dependency
injection at the composition root" pattern.

```
   main.py  create_app()  ─── builds & wires ──►  Embedder
                                                  FAISS (doc)  ─┐
                                                  FAISS (sensor)├─► Retriever
                                                  SQLite       ─┘
                                                  LLM client   ───► (used by /api/query)
                                                  PromptBuilder
```

---

## 12. Deployment: laptop and Raspberry Pi 5

The **same two commands** bring the whole stack up on either machine — there is
no Pi-specific code branch:

```bash
bash setup.sh    # installs deps, builds llama.cpp, downloads the model (~20-45 min)
bash run.sh      # starts llama-server + the FastAPI app
```

`setup.sh` is **idempotent** (safe to re-run — it skips anything already done).
The thread count is auto-detected from the CPU (4 on the Pi, ~12 on the laptop).

| | Laptop (dev machine) | Raspberry Pi 5 (target) |
|---|---|---|
| Hardware | Dell Inspiron 15 3520, i5-1235U, 8 GB | Pi 5, quad-core Cortex-A76, 8 GB |
| Same query latency | 15–25 s | **78 s** (1730-token prompt) |
| RAM used | ~3 GB | **487 MB** resident (of 8 GB) |
| Status | primary dev + test | **Verified answering end-to-end, 2026-08-04** |

**The one gotcha on the Pi:** current Raspberry Pi OS ships **Python 3.13**, but
our pinned `torch`/`numpy` versions have no 3.13 wheels, so you must install a
standalone **Python 3.12** first. Full walkthrough:
[`docs/PI_DEPLOYMENT.md`](PI_DEPLOYMENT.md).

**Why the Pi is ~3–5× slower:** it's a 4-core phone-class CPU with no GPU.
Notably, **~90% of the time is "prompt processing"** (the model reading the
context), not writing the answer — which is why keeping prompts short (the 0.15
relevance floor, the token budget) matters even more on the Pi. Memory is *not*
a problem; the device is CPU-bound, not RAM-bound.

---

## 13. Glossary

| Term | Plain-English meaning |
|---|---|
| **RAG** | Retrieval-Augmented Generation — find relevant text, then have the AI answer from it. |
| **LLM** | Large Language Model — the AI that writes the answer (here, Llama 3.2 3B). |
| **Embedding / vector** | A list of numbers (384 of them) representing a text's *meaning*. Similar meaning → similar numbers. |
| **Cosine similarity** | A score (−1 to 1) for how similar two vectors are. Higher = more related. |
| **FAISS** | The library that stores vectors and finds the closest ones to a query. |
| **Chunk** | A ~400-token slice of a document — the unit we search and cite. |
| **Token** | A piece of a word (~¾ of a word). Models measure text length in tokens. |
| **Context window** | The maximum tokens a model can read at once (4096 for ours). |
| **Quantisation (Q4_K_M)** | Compressing the model's numbers to ~4 bits so it's small and CPU-friendly. |
| **llama.cpp / llama-server** | The engine that actually runs the LLM on the CPU. |
| **SSE (Server-Sent Events)** | A way to stream the answer to the browser token-by-token. |
| **Temperature** | Randomness knob for the model. 0.0 = deterministic, same answer every time. |
| **Threshold** | Minimum similarity for a chunk to count as "relevant"; below it, we refuse. |
| **Rerank** | A second pass that re-orders search results (here, boosting keyword matches). |
| **Protocol** | A Python interface — lets us swap components (real vs. fake LLM, etc.). |
| **Hallucination** | When an AI confidently states something false. RAG + refusal prevents this. |
| **Ingestion** | The process of turning raw files into searchable chunks. |
| **Idempotent** | Safe to run again — re-running changes nothing already done. |

---

## 14. Advisor Q&A cheat-sheet

Short, confident answers to the questions an advisor is most likely to ask.
If you can answer these, you can defend the project.

**Q: In one sentence, what is TinyRAG?**
A private, offline document question-answering assistant that runs entirely on a
Raspberry Pi 5, answers only from the documents you give it, cites its sources,
and refuses when it doesn't know.

**Q: Explain how it works.**
Three stages. **Retrieve:** we turn the question into a 384-number "meaning"
vector, search a FAISS index for the most similar document chunks, and rerank
them. **Augment:** we build a prompt containing those chunks plus an instruction
to answer only from them and cite sources. **Generate:** a local 3B Llama model
reads that prompt and writes a cited answer — or refuses if the chunks don't
contain the answer.

**Q: Why is this better than just using ChatGPT?**
Privacy (documents never leave the device), no internet or API cost required,
and grounded answers with citations instead of hallucinations. It works with the
Wi-Fi off.

**Q: How does it avoid making things up?**
Two gates. The retriever drops chunks below a similarity threshold, so if nothing
relevant is found the model gets no context. And the system prompt explicitly
tells the model to reply "I don't have enough information in the provided
documents" when the context doesn't answer the question.

**Q: Why Llama 3.2 3B and not something bigger, or Phi-3?**
3B is the sweet spot that fits 8 GB of Pi RAM while following instructions
reliably. We first used Phi-3 Mini but its GGUF produced garbage above ~2048
prompt tokens, and our prompts run 2500–3600 tokens, so we switched to Llama 3.2
3B, which stays coherent. Bigger models (Mistral 7B) run but are too slow on the
Pi (~2 tokens/sec).

**Q: Why FAISS and SQLite together?**
FAISS finds chunks by *meaning* (vector similarity); SQLite stores the *actual
text, filename, and page*. Retrieval ranks with FAISS, then fetches the real
passages from SQLite to build the prompt and citations.

**Q: What's the embedding model, and what does it do?**
`all-MiniLM-L6-v2`. It converts any text into 384 numbers such that
similar-meaning texts get similar numbers — that's what makes semantic search
(not just keyword search) possible.

**Q: Why does retrieval need a keyword rerank if it has semantic search?**
Pure vector search sometimes ranks a chunk that *mentions* a term above the one
that *defines* it (our "ErP directive" case put the table-of-contents above the
real page). A small keyword boost — +0.10 per matching query word, +0.20 if all
match — fixes that cheaply.

**Q: Why does it run on CPU and not GPU?**
The Pi has no usable GPU, and our laptop's integrated GPU isn't supported by the
engine's build. llama.cpp with OpenBLAS on CPU is fast enough for an interactive
assistant, and it's what makes the project truly portable to edge hardware.

**Q: How do you know it actually works / how is it tested?**
~1,300 automated test functions across 27 files, plus a 22-question "gold set"
for evaluation. The pure-logic core is tested with fake components (no model
needed), and it's been verified answering end-to-end on real Pi 5 hardware.

**Q: What about the sensors part?**
The retrieval and summarisation code for sensor data exists (daily
temperature/humidity/energy/motion summaries become searchable chunks), but the
physical sensor wiring is deferred — the advisor will guide that. The core tool
is general-purpose: it answers from **any** documents.

**Q: What are the current limitations?**
It's CPU-slow on the Pi (~78 s per answer, ~90% of it just reading the prompt);
it handles small-to-medium corpora (the exact FAISS index would need upgrading
for very large ones); no OCR (scanned PDFs aren't supported); and the sensor
hardware path is not yet wired.

**Q: What would you improve next?**
Cut prompt-processing time (the main latency cost), add an evaluation harness
over the gold set to compare models, and — with the advisor — wire the real
sensors.

---

*For deeper technical detail see [`docs/03_architecture_v1.md`](03_architecture_v1.md)
(module boundaries + interfaces), [`docs/02_srs_v1.md`](02_srs_v1.md) (full
requirements), and [`docs/PI_DEPLOYMENT.md`](PI_DEPLOYMENT.md) (deployment field
notes). For a live demo run-sheet see [`docs/DEMO_QUESTIONS.md`](DEMO_QUESTIONS.md).*
