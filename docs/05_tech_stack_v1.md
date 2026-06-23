# TinyRAG — Tech Stack v1 (Pinned)

**Project Title:** TinyRAG — A Lightweight, On-Device Retrieval-Augmented Generation Assistant for Smart Home IoT
**Document version:** 1.0
**Date:** 2026-06-23
**Status:** Draft — awaiting student review
**Source of truth:** `docs/01_project_scope_v2.md`, `docs/02_srs_v1.md`, `docs/03_architecture_v1.md`, `docs/04_database_design_v1.md`

---

## 1. Purpose of This Document

This document pins down **every technology, library, and tool** TinyRAG uses, with **exact versions** and **rationale**. It is the source of truth for `requirements.txt` and the build flags in `setup.sh`.

After reading this, you should be able to:
- Reproduce the exact environment on any machine with `./setup.sh`.
- Justify every dependency choice in your final report.
- Know the difference between the **Pi 5 stack** and the **laptop stack**.

> **Pinning policy:** all Python deps are pinned to exact versions (`==`). C++ deps (llama.cpp) are pinned to a versioned `gguf-vX.Y.Z` tag. This ensures reproducibility per NFR-31 and NFR-21.

---

## 2. Tech Stack at a Glance

| Layer | Technology | Pi 5 | Laptop | Version |
|-------|------------|------|--------|---------|
| **OS** | Raspberry Pi OS 64-bit / Ubuntu 24.04 | ✅ | ❌ | latest LTS |
| **OS (laptop)** | Ubuntu 24.04.4 LTS | ❌ | ✅ | confirmed |
| **Python** | CPython | 3.11 | 3.12 | exact |
| **LLM engine** | llama.cpp (HTTP server) | ✅ | ✅ | tag `gguf-v0.19.0` pinned (see §3.2) |
| **Primary LLM** | Phi-3 Mini 3.8B Instruct Q4_K_M | ✅ | ✅ | GGUF |
| **Comparison LLM 1** | TinyLlama 1.1B Chat Q4_K_M | ✅ | ✅ | GGUF |
| **Comparison LLM 2** | Llama 3.2 3B Instruct Q4_K_M | ✅ | ✅ | GGUF |
| **Optional LLM 3** | Mistral 7B Instruct Q4_K_M | ⚠️ (slow) | ✅ | GGUF |
| **Embedding model** | all-MiniLM-L6-v2 | ✅ | ✅ | v2 (HF) |
| **Vector store** | FAISS (CPU) | ✅ | ✅ | faiss-cpu 1.8.0 |
| **Metadata DB** | SQLite 3 | ✅ | ✅ | system |
| **Backend framework** | FastAPI | ✅ | ✅ | 0.115.4 |
| **ASGI server** | Uvicorn | ✅ | ✅ | 0.32.0 |
| **Config** | PyYAML + Pydantic v2 | ✅ | ✅ | 6.0.2, 2.9.2 |
| **PDF parsing** | pdfplumber | ✅ | ✅ | 0.11.4 |
| **Text tokenization** | tiktoken | ✅ | ✅ | 0.8.0 |
| **Embedding lib** | sentence-transformers | ✅ | ✅ | 3.2.1 |
| **ML backend** | PyTorch (CPU) | ✅ | ✅ | 2.4.1 |
| **Logging** | structlog | ✅ | ✅ | 24.4.0 |
| **Testing** | pytest | ✅ | ✅ | 8.3.3 |
| **Linting** | ruff | ✅ | ✅ | 0.7.2 |
| **Sensor I/O (Pi)** | libgpiod + adafruit-circuitpython-dht | ✅ | ❌ | system / 4.0.4 |
| **Sensor I/O (laptop)** | paho-mqtt | optional | ✅ | 2.1.0 |
| **Frontend** | HTML + vanilla JS + Jinja2 | ✅ | ✅ | 3.1.4 |
| **Process supervision (Pi)** | systemd | ✅ | ❌ | system |

---

## 3. LLM Layer (llama.cpp + Phi-3)

### 3.1 Why llama.cpp?

| Reason | Detail |
|--------|--------|
| **CPU performance** | State-of-the-art CPU inference. Phi-3 4-bit hits ~8–12 tok/s on Pi 5. |
| **No GPU required** | Critical for Pi 5 (no usable GPU) and for laptops without NVIDIA cards. |
| **Quantization support** | Native Q4_K_M, Q5_K_M, Q8_0 — we use Q4 for size. |
| **Mature, stable** | Used by Ollama, LM Studio, text-generation-webui. |
| **OpenAI-compatible HTTP** | The `--server` mode exposes `/v1/chat/completions`, making integration trivial. |

### 3.2 Pinned Version (tag)

```
llama.cpp: tag gguf-v0.19.0  (released 2026-05-06; commit a290ce62)
```

(We pin to a versioned `gguf-vX.Y.Z` tag — llama.cpp's stable release surface, released roughly monthly — rather than `master` or daily `bNNNN` builds. The actual SHA is recorded in `docs/BUILDS.md` §2.1.)

### 3.3 Build Flags

**On Raspberry Pi 5 (aarch64):**
```bash
cmake -B build \
  -DGGML_OPENBLAS=OFF \
  -DGGML_NATIVE=OFF \
  -DCMAKE_C_FLAGS="-mcpu=cortex-a76 -mfpu=neon-fp-armv8" \
  -DCMAKE_CXX_FLAGS="-mcpu=cortex-a76 -mfpu=neon-fp-armv8" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 4
```

**On Dell laptop (x86_64, Ubuntu 24.04):**
```bash
cmake -B build \
  -DGGML_OPENBLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 10
```

The `setup.sh` script **auto-detects the architecture** and applies the right flags.

### 3.4 LLM Server Invocation

```bash
./llama.cpp/build/bin/llama-server \
    --model models/phi-3-mini-3.8b-instruct-q4.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    --ctx-size 4096 \
    --n-gpu-layers 0 \
    --threads 4 \
    --cont-batching
```

| Flag | Value | Why |
|------|-------|-----|
| `--host 127.0.0.1` | Local only | Privacy (NFR-15) |
| `--port 8080` | Configurable | Standard llama.cpp default |
| `--ctx-size 4096` | 4k context | Matches Phi-3-mini 4k variant |
| `--n-gpu-layers 0` | CPU only | No GPU on Pi 5 |
| `--threads 4` | All Pi 5 cores | Max throughput |
| `--cont-batching` | Continuous batching | Smoother streaming |

### 3.5 LLM Models (GGUF files)

| Role | Model | On-disk filename | Size | Source URL |
|------|-------|------------------|------|------------|
| **Primary** | Phi-3 Mini 3.8B Instruct (4k) Q4_K_M | `models/phi-3-mini.gguf` | ~2.3 GB | HF: `microsoft/Phi-3-mini-4k-instruct-gguf` |
| **Eval A** | TinyLlama 1.1B Chat v1.0 Q4_K_M | `models/tinyllama-1.1b.gguf` | ~700 MB | HF: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` |
| **Eval B** | Llama 3.2 3B Instruct Q4_K_M | `models/llama-3.2-3b.gguf` | ~1.8 GB | HF: `bartowski/Llama-3.2-3B-Instruct-GGUF` |
| **Eval C (optional)** | Mistral 7B Instruct v0.3 Q4_K_M | `models/mistral-7b.gguf` | ~4 GB | HF: `TheBloke/Mistral-7B-Instruct-v0.3-GGUF` |

> **On-disk filename convention.** We standardise on the model *id* as the on-disk filename (`<models_dir>/<id>.gguf`), not the upstream HF filename. This is a deliberate footgun-avoidance measure: if the upstream maintainer renames the file (e.g. `Phi-3-mini-4k-instruct-q4.gguf` → `Phi-3-mini-4k-instruct-q4_k_m.gguf`), our `make run-llm` keeps working. The mapping from id to upstream filename lives in `src/tinyrag/models/registry.py` and is documented in `docs/MODELS.md` §1.

> **Note on Llama 3.2:** the official Meta repo is gated; we use the community `bartowski` mirror which hosts the same weights. This is captured in the registry entry for `llama-3.2-3b` (`hf_repo = "bartowski/Llama-3.2-3B-Instruct-GGUF"`). For attribution in the final report, see the Llama 3.2 community license block on <https://llama.meta.com/>.

**Download mechanism.** Use `scripts/download_models.py` (or the `make download-llm` shortcut). The script:

1. Resolves the URL from the registry.
2. Streams the file in 1 MiB chunks.
3. Resumes interrupted downloads via HTTP `Range`.
4. Verifies the SHA-256 against `models/_manifest.json`.
5. Refuses to keep a file whose hash doesn't match.

**Why Q4_K_M?** It's the sweet spot — small enough to fit in 3 GB RAM, fast enough on CPU, quality retention ~95% of F16. This is what `ollama` ships by default for the same reason.

**Full catalog and SHA-256 pins:** see `docs/MODELS.md` (the human-readable mirror of `src/tinyrag/models/registry.py`).

---

## 4. Python Dependencies (`requirements.txt`)

The complete pinned `requirements.txt`:

```txt
# ===== Web framework =====
fastapi==0.115.4
uvicorn[standard]==0.32.0
jinja2==3.1.4
python-multipart==0.0.12            # for file uploads in FastAPI

# ===== Config =====
pyyaml==6.0.2
pydantic==2.9.2
pydantic-settings==2.6.1

# ===== Vector store & embeddings =====
faiss-cpu==1.8.0.post1
sentence-transformers==3.2.1
torch==2.4.1                         # CPU-only wheel
numpy==1.26.4
tiktoken==0.8.0

# ===== Parsing =====
pdfplumber==0.11.4

# ===== HTTP client (for llama.cpp) =====
httpx==0.27.2
sse-starlette==2.1.3                 # for Server-Sent Events streaming

# ===== Logging =====
structlog==24.4.0

# ===== Testing =====
pytest==8.3.3
pytest-asyncio==0.24.0
pytest-cov==5.0.0
httpx==0.27.2                        # already above, for TestClient

# ===== Linting =====
ruff==0.7.2

# ===== Sensor I/O (laptop: MQTT only) =====
paho-mqtt==2.1.0

# ===== Sensor I/O (Pi 5: GPIO + DHT) =====
# (installed conditionally in setup.sh, not in requirements.txt)
# RPi.GPIO / gpiozero / libgpiod are system packages on Raspberry Pi OS
# adafruit-circuitpython-dht==4.0.4
# adafruit-blinka==8.21.0

# ===== Misc =====
python-dateutil==2.9.0.post0
pandas==2.2.3
```

### 4.1 Why these specific versions?

| Package | Why this version |
|---------|------------------|
| `fastapi 0.115.4` | Latest stable as of 2024-11. Modern Python type-hint features. |
| `pydantic 2.9.2` | v2 (faster, stricter). v1 is being deprecated. |
| `faiss-cpu 1.8.0` | Stable. Pairs with `numpy 1.26`. |
| `sentence-transformers 3.2.1` | Latest as of 2024-11. |
| `torch 2.4.1` | CPU-only wheel, ~250 MB. Pinned because the GPU wheels can break things. |
| `pdfplumber 0.11.4` | More robust than PyPDF2 for complex PDFs. |
| `structlog 24.4.0` | JSON-native, modern. |
| `pytest 8.3.3` | Standard. |
| `ruff 0.7.2` | 10-100× faster than flake8+black+isort combined. |

### 4.2 Why not use Anaconda / conda?

- **Pip is the standard** for Python web projects. Conda is for data science with heavy native deps.
- Our deps are all pip-installable wheels — no need for conda's solver.
- Smaller virtualenv size.
- Easier to pin and reproduce.

### 4.3 About PyTorch size

`torch==2.4.1` (CPU-only) is ~250 MB. This is unavoidable because `sentence-transformers` depends on it. The embedding model itself is small (80 MB); the bulk is the PyTorch runtime.

**Memory budget after model load:**
- PyTorch runtime: ~200 MB
- Embedding model: ~80 MB
- llama.cpp server: ~1.5 GB (Phi-3 Q4)
- FastAPI + code: ~50 MB
- Vector store (in-memory): ~5 MB
- **Total: ~1.85 GB** — well within NFR-7 (≤ 1.5 GB idle, ≤ 3 GB peak).

---

## 5. Frontend Stack

### 5.1 No Framework Decision

| Considered | Verdict |
|------------|---------|
| **HTML + vanilla JS + Jinja2** | ✅ **Chosen.** Simple, no build step, no Node.js required. |
| React | ❌ — adds build complexity, 200+ MB of Node deps |
| Vue | ❌ — same as React |
| HTMX | ⚠️ — viable alternative, but vanilla JS is even simpler |
| Alpine.js | ⚠️ — viable alternative, ~15 KB |

**Rationale:** A capstone UI with 2 pages (chat + admin) does not justify a JS framework. Vanilla JS + Jinja2 keeps the project:
- Easy to read.
- Easy to grade (no minified bundles).
- Easy to deploy (no `npm install` step).
- Easy to extend (just add a `fetch()` call).

### 5.2 Frontend Files

| File | Purpose | Size estimate |
|------|---------|---------------|
| `ui/templates/index.html` | Chat page | ~150 lines |
| `ui/templates/admin.html` | Document management page | ~200 lines |
| `ui/static/style.css` | Styling | ~300 lines |
| `ui/static/chat.js` | Chat streaming logic | ~100 lines |
| `ui/static/admin.js` | Doc management logic | ~80 lines |

**No CDN dependencies** — everything is local (per NFR-13, no internet at runtime).

---

## 6. Build & Runtime Tools

### 6.1 System packages (installed by `setup.sh`)

**On both Pi 5 and laptop (Ubuntu 24.04 / Debian Bookworm):**
```bash
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    build-essential cmake git \
    libopenblas-dev liblapack-dev \
    sqlite3
```

**On Raspberry Pi 5 only (for real sensors):**
```bash
sudo apt-get install -y \
    libgpiod-dev gpiod \
    python3-lgpio python3-rpi.gpio
```

**On laptop only (for MQTT broker testing, optional):**
```bash
sudo apt-get install -y mosquitto mosquitto-clients
```

### 6.2 Why OpenBLAS?

- **Laptop:** OpenBLAS gives llama.cpp a ~2× speedup on x86_64 thanks to optimized SIMD kernels.
- **Pi 5:** OpenBLAS does not have great aarch64 NEON kernels, so we skip it. The plain NEON build is already optimal.

### 6.3 Build time expectations

| Component | Pi 5 (4 cores) | Laptop (10 cores) |
|-----------|----------------|-------------------|
| llama.cpp | 45–60 min | 5–10 min |
| Python deps (pip) | 10–15 min | 3–5 min |
| Model downloads (3 × ~2 GB) | 20–40 min (depends on net) | 5–10 min |
| **Total `setup.sh` time** | ~1.5–2 hours | ~20–30 min |

> **Tip:** do all the development on the laptop first. Only run `setup.sh` on the Pi in Week 7 (deployment week).

---

## 7. Why We Avoid Certain Things

| Avoided | Why |
|---------|-----|
| **Ollama** | Extra dependency, hides llama.cpp behind another layer, harder to control. We use llama.cpp directly. |
| **LangChain** | Massive dependency, opinionated, breaks on Pi, learning curve. We build the 200-line RAG pipeline ourselves — easier to understand, easier to demo, no surprise breakage. |
| **LlamaIndex** | Same reasons as LangChain. Overkill for our scope. |
| **Hugging Face Transformers + accelerate** | Heavy, slow on CPU, designed for GPUs. llama.cpp is faster on Pi. |
| **ChromaDB** | Adds a separate service / embedded complexity. FAISS is simpler and faster at our scale. |
| **PostgreSQL** | Overkill. SQLite is perfect. |
| **MongoDB** | Overkill. NoSQL not needed. |
| **Redis** | No caching needed for capstone scale. |
| **Docker** | Adds a virtualization layer; on Pi 5 it can hurt performance. We'll provide `setup.sh` instead. (Docker can be added as a deployment option later.) |
| **Cloud LLM APIs (OpenAI, Anthropic, etc.)** | Violates NFR-13 (no cloud at runtime). The whole point of TinyRAG is on-device. |

---

## 8. Platform-Specific Stack Differences

| Aspect | Raspberry Pi 5 | Dell laptop |
|--------|---------------|-------------|
| **OS** | Raspberry Pi OS 64-bit (Bookworm) | Ubuntu 24.04.4 LTS |
| **Python** | 3.11 (system) | 3.12 (system) |
| **llama.cpp flags** | `-mcpu=cortex-a76 -mfpu=neon-fp-armv8` | `-DGGML_OPENBLAS=ON` |
| **llama.cpp threads** | 4 | 10 |
| **Sensor I/O** | libgpiod + DHT + PIR | MQTT only (or simulated) |
| **Process supervision** | systemd (`tinyrag.service`) | none (just `run.sh`) |
| **Browser access** | `http://<pi-ip>:8000/` (LAN) | `http://localhost:8000/` (same machine) |
| **Power resilience** | read-only rootfs (optional) | n/a |

These differences live in `config.yaml` and `setup.sh` — the source code is identical.

---

## 9. Dependency Footprint Summary

| Component | Disk | RAM (idle) | RAM (peak) |
|-----------|------|------------|------------|
| llama.cpp binary | 5 MB | 0 | 0 |
| PyTorch (CPU) | 250 MB | 200 MB | 200 MB |
| sentence-transformers lib | 20 MB | 30 MB | 30 MB |
| Embedding model | 80 MB | 80 MB | 80 MB |
| Phi-3 Mini 3.8B Q4 | 2.3 GB | 1.5 GB | 1.8 GB |
| FAISS native lib | 30 MB | 50 MB | 50 MB |
| FastAPI + deps | 50 MB | 50 MB | 50 MB |
| App code | 5 MB | 30 MB | 30 MB |
| **Total (single LLM)** | **~2.8 GB** | **~1.9 GB** | **~2.3 GB** |
| + TinyLlama 1.1B | +700 MB | — | — |
| + Llama 3.2 3B | +1.8 GB | — | — |
| **Total (all 3 LLMs)** | **~5.3 GB** | (only one loaded at a time) | (only one loaded at a time) |

**Note:** only one LLM is loaded into memory at a time. Switching LLMs means restarting `llama-server` with a different model file. The model file stays on disk; the previous one is unloaded from RAM when the new one loads.

---

## 10. Reproducibility Checklist

To verify the project is reproducible, anyone should be able to:

1. Clone the repo.
2. Run `./setup.sh` (one command).
3. Wait for it to finish (≤ 2 h on Pi 5, ≤ 30 min on laptop).
4. Run `./run.sh`.
5. Open `http://localhost:8000/` (or `http://<pi-ip>:8000/` from another device).
6. Type a question, get a cited answer.

The `setup.sh` script will record the **exact llama.cpp commit hash** and **pip freeze output** to `BUILDS.md` for full reproducibility.

---

## 11. Upgrade Policy (post-capstone)

- All Python deps are pinned in `requirements.txt` with `==`.
- To upgrade, change a version, run tests, and commit.
- llama.cpp is pinned to a commit hash, not a tag, for maximum stability.
- Models are versioned in the filename (`phi-3-mini-3.8b-instruct-q4.gguf`).
- A future `scripts/check_updates.py` could alert when new llama.cpp commits or model versions are available, but this is out-of-scope for v1.

---

## 12. Open Questions for the Student

| # | Question | Default I'll go with |
|---|----------|----------------------|
| Q1 | Acceptable to use `bartowski` mirrors for Llama 3.2 (since Meta's repo is gated)? | **Yes** — same weights, easier access. Will document. |
| Q2 | Include Mistral 7B as a 4th comparison model? | **Optional** — only if time allows. We start with 3. |
| Q3 | PyTorch CPU wheel vs. building from source? | **CPU wheel** — faster install, sufficient. |
| Q4 | Use a pre-built llama.cpp binary (from llama.cpp releases) instead of compiling? | **Compile from source** — gives us the exact flags we need. Binary releases don't expose all flags. |
| Q5 | Add Docker as an alternative deployment option? | **Stretch only** — not in v1. |

---

## 13. Document Approval

| Role | Name | Approval | Date |
|------|------|----------|------|
| Student | Marajul Haque | ⏳ pending | |
| Advisor | Abu Nowshed Chy | (not required for v1) | |

---

*End of Tech Stack v1. Next: Development Roadmap (`docs/06_roadmap_v1.md`) — the student reviews and approves this Tech Stack first.*
