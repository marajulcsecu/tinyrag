# AgriRAG Voice Agent Overlay & 4-Layer Bengali STT Design Spec

**Date:** 2026-11-23  
**Branch:** `agrirag`  
**Scope:** Mobile-first full-screen Voice Agent overlay + accurate hybrid Bengali speech-to-text (STT) pipeline with agricultural phonetic post-correction.

## 1. Goals & Problem Statement

Farmers on mobile browsers currently face two issues when speaking in Bengali (`bn-BD`):
1. Clicking the mic records inline inside a small textarea instead of offering a clear, distraction-free Voice Agent experience (waveform + large Speak/Stop controls).
2. Voice-to-text frequently produces garbled ("ulta-palta") Bengali queries because:
   - `recognition.continuous = true` on Android Chrome overlaps and scrambles utterances.
   - The existing ASCII `\b` duplicate-removal regex does not match Bengali Unicode words.
   - `scheduleAutoSendOnSilence()` submits unstable interim hypotheses without `isFinal` checks or user review.
   - Generic Web Speech API lacks Bangladeshi agricultural domain vocabulary (`ব্লাস্ট`, `কারেন্ট পোকা`, `মাজরা পোকা`, `আলুর মড়ক`, etc.).

## 2. Architecture

### 2.1 Backend — 4-Layer Bengali STT Pipeline
Files:
- `src/tinyrag/input_adapters/voice.py`
- `src/tinyrag/api/schemas.py`
- `src/tinyrag/api/routes_query.py`

1. **Agricultural Phonetic Post-Corrector (`correct_bengali_agri_transcript`)**:
   - NFC-normalizes text, collapses Bengali Unicode consecutive duplicate words/bigrams, and applies curated Bengali agricultural phonetic fixes (e.g., `ধানের ব্লাউজ/প্লাস্ট` -> `ধানের ব্লাস্ট`, `কারেন্ট পাকা` -> `কারেন্ট পোকা`, `আলুর মোড়ক/মরক` -> `আলুর মড়ক`, `মাঝরা/মাছরা পোকা` -> `মাজরা পোকা`, `আদ্রতা` -> `আর্দ্রতা`).
   - Returns `(corrected_text, corrections_applied)` so the UI can show a subtle "কৃষি শব্দ সংশোধিত" badge.
2. **Cloud Audio STT Router (`VoiceInputAdapter.transcribe`)**:
   - When `audio_bytes` + `engine_mode == "cloud"` + `api_key` are supplied to `POST /api/voice`:
     - **Groq (`api.groq.com`)**: `POST .../audio/transcriptions` (`whisper-large-v3-turbo`, `language="bn"`, `prompt=BENGALI_AGRI_STT_PROMPT`).
     - **OpenAI (`api.openai.com`)**: `POST .../audio/transcriptions` (`whisper-1`, `language="bn"`, `prompt=BENGALI_AGRI_STT_PROMPT`).
     - **Gemini (`generativelanguage.googleapis.com`)**: native `v1beta/models/{model}:generateContent` with inline base64 audio + strict Bengali agricultural verbatim transcription instruction.
   - Falls through gracefully to local `faster-whisper` (if installed) or `transcript_hint` fallback if the cloud call fails or times out.
3. **Offline `faster-whisper` Enhancements**:
   - Module-level model cache (`_WHISPER_MODEL_CACHE`) avoids reloading weights per request.
   - Passes `language="bn"`, `vad_filter=True`, and `initial_prompt=BENGALI_AGRI_STT_PROMPT`.
4. **Backward Compatibility**:
   - `VoiceQueryRequest` keeps all existing fields and adds optional `audio_mime_type: str | None = None`.
   - `VoiceTranscriptionResult.to_dict()` preserves all existing keys (`ok`, `transcript`, `engine`, `language`, `audio_bytes_len`, `duration_ms`, `fallback_reason`) and adds `raw_transcript` and `corrected` (boolean).

### 2.2 Frontend — Modular Voice Agent Overlay
Files:
- `ui/static/voice_agent.js` (new standalone IIFE module)
- `ui/templates/index.html` (overlay markup + `<script src="/static/voice_agent.js" defer>`)
- `ui/static/style.css` (full-screen mobile-first Voice Agent styles & responsive polish)
- `ui/static/chat.js` (clean bridge API on `window.__AgriRAGChat` + fixed Bengali Web Speech dedup/isFinal handling)

**Voice Agent State Machine:**
`closed` -> `idle` -> `listening` -> `transcribing` -> `review` -> `thinking` -> `speaking`

- **Audio Capture & Live Waveform**:
  - Uses `navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } })` + `AudioContext`/`AnalyserNode` to draw a live acoustic waveform on `<canvas id="va-waveform">`.
  - Records real audio (`MediaRecorder` with `audio/webm;codecs=opus` or `audio/mp4`) whenever Cloud Mode is active (or backend STT is available), while also showing live Web Speech interim preview when supported.
  - Fixes Web Speech API for Android Chrome: separates `isFinal` vs `interim`, uses `continuous = false` on Android with clean append, and applies Unicode-safe Bengali deduplication + client-side agricultural phonetic post-correction.
- **Review & Cancelable Auto-Send**:
  - Displays editable Bengali transcript inside the overlay before sending.
  - If `অটো-সেন্ড` is enabled, runs a visible 2.5s progress bar that pauses immediately if the user focuses/edits the transcript or taps `থামান`.
  - If `অটো-সেন্ড` is disabled, waits for the user to tap `পাঠান ✓` (or `আবার বলুন`).
- **Two-Button Farmer-Friendly Bottom Dock**:
  - Left large circular emerald button: `বলুন` / `বলা শেষ` / `পাঠান ✓`.
  - Right large circular red button: `STOP` / `থামান` (cancels recording, stops countdown, or interrupts TTS playback).
- **Live Answer Feedback inside Overlay**:
  - Submits through the existing `chat.js` SSE pipeline so chat history, citations, and streaming prefetch TTS stay 100% synchronized.
  - Mirrors streaming status (`উত্তর খুঁজছি...`) and spoken answer text inside the overlay while neural Bengali TTS plays.

## 3. Verification Plan
1. Backend unit & API tests in `tests/test_voice_agent.py` covering:
   - Bengali phonetic post-correction & Unicode duplicate collapsing.
   - Cloud STT routing for Groq, OpenAI, and Gemini (with mocked `httpx` responses + fallback on network failure).
   - `POST /api/voice` integration with `audio_mime_type` and `run_rag=False`.
   - Frontend HTML/CSS/JS structural & syntax checks (`node --check ui/static/voice_agent.js` and `chat.js`).
2. Full existing `pytest` suite (`tests/test_web_ui.py` and all unit/integration tests) + `ruff check`.
3. Live browser verification on desktop + mobile viewport (390×844) via running FastAPI server and Chrome DevTools / screenshot inspection.
