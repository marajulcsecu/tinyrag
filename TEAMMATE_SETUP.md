# AgriRAG (TinyRAG) — Teammate Quickstart Guide

Follow these steps to clone, set up, and test the **AgriRAG (`agrirag` branch)** voice-to-voice Bengali agricultural assistant on your own PC and Android phone.

---

## 1. Clone the `agrirag` Branch

```bash
git clone -b agrirag https://github.com/marajulcsecu/tinyrag.git
cd tinyrag
```

*(If you already cloned the repo earlier, just run `git fetch origin && git checkout agrirag && git pull`.)*

---

## 2. Create Python 3.12 Virtual Environment & Install Dependencies

Requires **Python 3.12**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1

# Optional (saves ~1.8 GB on CPU-only laptops):
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1

pip install -r requirements.txt -r requirements-dev.txt
```

---

## 3. Ingest the Curated BRRI / BARI / DAE Manuals & Synthetic Agro-IoT Sensors

The 4 curated Bengali agricultural manuals are already committed inside `data/agrirag/documents/`. Build your local SQLite + Dual FAISS indexes once:

```bash
PYTHONPATH=src python scripts/ingest.py --all
PYTHONPATH=src python scripts/generate_synthetic_sensors.py --days 30
PYTHONPATH=src python scripts/ingest_sensors.py
```

---

## 4. Run AgriRAG (Choose Fast Cloud Mode OR 100% Offline Local Mode)

### Option A — Fast Setup (Cloud API Mode, Zero `llama.cpp` Build Needed!)
Ideal for quick testing on any laptop (Windows / Linux / macOS) with low RAM:

```bash
PYTHONPATH=src uvicorn tinyrag.main:app --host 0.0.0.0 --port 8000
```

1. Open **http://localhost:8000/** in Chrome/Edge.
2. Click **`লোকাল ইঞ্জিন`** (top right) $\rightarrow$ select **ক্লাউড API মোড** $\rightarrow$ pick **Google Gemini** or **Groq (Fast)** $\rightarrow$ paste your free API key $\rightarrow$ Save.
3. Click the green **Mic button** to launch **AgriRAG ভয়েস** (uses Cloud Whisper / Gemini Audio + Bengali agricultural phonetic correction!).

### Option B — 100% Offline Local Mode (Linux / Raspberry Pi 5 with `llama.cpp`)

```bash
bash setup.sh        # builds llama.cpp + downloads Gemma 4 E2B GGUF
bash run.sh          # launches llama-server (8080) + FastAPI UI (8000) + adb reverse
```

---

## 5. Testing Voice Mode on an Android Phone

Android Chrome blocks microphone access on plain HTTP LAN IPs (`http://192.168.x.x:8000`), so connect your phone via USB with **USB Debugging** enabled and run:

```bash
adb reverse tcp:8000 tcp:8000
```

Then open **`http://localhost:8000/`** in your phone's Chrome browser and tap the emerald **Mic button**!

---

## 6. Run Automated Tests

```bash
pytest tests/test_voice_agent.py tests/test_web_ui.py tests/test_api.py -q
```
