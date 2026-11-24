/* AgriRAG (TinyRAG) — Mobile-First Full-Screen Voice Agent & Accurate Bengali STT.
 *
 * Pure Vanilla ES2017+ IIFE, no build step. Loaded after chat.js via:
 *   <script src="/static/voice_agent.js" defer></script>
 *
 * Key engineering choices for real rural Bangladeshi phones (Android Chrome):
 *   1. Full-screen Voice Agent overlay triggered by #mic-btn with real-time
 *      waveform canvas, editable live transcript review card, and big circular
 *      "বলুন / পাঠানো" + red "STOP / থামান" controls.
 *   2. Zero Android mic-lock conflicts:
 *      - Cloud mode (Groq / Gemini / OpenAI API key configured): exclusively uses
 *        MediaRecorder + Web Audio AnalyserNode (noiseSuppression + echoCancellation
 *        + RMS silence VAD) and transcribes via POST /api/voice with Bengali
 *        agricultural prompt biasing.
 *      - Local / Offline mode: uses fixed Web Speech API (bn-BD) with Android-safe
 *        single-utterance mode, strict isFinal vs interim separation, Unicode-safe
 *        Bengali duplicate collapsing, and client-side agricultural phonetic repairs.
 *      - Firefox / browsers without Web Speech API: automatically uses MediaRecorder
 *        + POST /api/voice (local faster-whisper fallback).
 *   3. Review-before-send safety:
 *      - Never submits raw unstable interim speech.
 *      - When Auto-Send is ON, displays a 2.5s cancelable progress bar that pauses
 *        immediately if the user taps the transcript box to edit or hits STOP.
 *      - When Auto-Send is OFF, waits for an explicit tap on "পাঠান".
 */

(function () {
    "use strict";

    const IS_ANDROID = /Android/i.test(navigator.userAgent || "");
    const AUTO_SEND_DELAY_MS = 2500;
    const VAD_SPEECH_RMS = 0.022;
    const VAD_SILENCE_MS = 2200;
    const VAD_MIN_SPEECH_MS = 350;
    const MAX_RECORDING_MS = 30000;

    // Mirror of backend _AGRI_PHONETIC_FIXES so offline Web Speech gets instant repair
    const AGRI_PHONETIC_FIXES = [
        ["ধানের ব্লাউজ", "ধানের ব্লাস্ট"],
        ["ধানের প্লাস্টিক", "ধানের ব্লাস্ট"],
        ["ধানের প্লাস্ট", "ধানের ব্লাস্ট"],
        ["ধানের ফ্লাস্ট", "ধানের ব্লাস্ট"],
        ["ধানের ব্লাস্টিক", "ধানের ব্লাস্ট"],
        ["ব্লাউজ রোগ", "ব্লাস্ট রোগ"],
        ["প্লাস্ট রোগ", "ব্লাস্ট রোগ"],
        ["ফ্লাস্ট রোগ", "ব্লাস্ট রোগ"],
        ["ব্লাস্টিক রোগ", "ব্লাস্ট রোগ"],
        ["শিস ব্লাস্ট", "শীষ ব্লাস্ট"],
        ["শিশ ব্লাস্ট", "শীষ ব্লাস্ট"],
        ["শিস পচা", "শীষ পচা"],
        ["শিশ পচা", "শীষ পচা"],
        ["সাদা শিস", "সাদা শীষ"],
        ["সাদা শিশ", "সাদা শীষ"],
        ["কারেন্ট পাকা", "কারেন্ট পোকা"],
        ["কারেন্ট পুকা", "কারেন্ট পোকা"],
        ["ক্যারেন্ট পোকা", "কারেন্ট পোকা"],
        ["ক্যারাম পোকা", "কারেন্ট পোকা"],
        ["বাদামি গাছ ফড়িং", "বাদামী গাছফড়িং"],
        ["বাদামী গাছ ফড়িং", "বাদামী গাছফড়িং"],
        ["মাঝরা পোকা", "মাজরা পোকা"],
        ["মাছরা পোকা", "মাজরা পোকা"],
        ["মজরা পোকা", "মাজরা পোকা"],
        ["মাজরা পুকা", "মাজরা পোকা"],
        ["মরা দিক", "মরা ডিগ"],
        ["আলুর মোড়ক", "আলুর মড়ক"],
        ["আলোর মড়ক", "আলুর মড়ক"],
        ["আলুর মরক", "আলুর মড়ক"],
        ["আলু মড়ক", "আলুর মড়ক"],
        ["লেট লাইট", "লেট ব্লাইট"],
        ["কটুই পোকা", "কাটুই পোকা"],
        ["কাটই পোকা", "কাটুই পোকা"],
        ["কাটুই পুকা", "কাটুই পোকা"],
        ["পামরি পুকা", "পামরি পোকা"],
        ["পামড়ি পোকা", "পামরি পোকা"],
        ["পাতা কোকরানো", "পাতা কোঁকড়ানো"],
        ["পাতা কুকড়ানো", "পাতা কোঁকড়ানো"],
        ["পাতা কোঁকরানো", "পাতা কোঁকড়ানো"],
        ["পাতা কুকরানো", "পাতা কোঁকড়ানো"],
        ["মাছি পুকা", "মাছি পোকা"],
        ["জাব পুকা", "জাব পোকা"],
        ["সাদা মাছ", "সাদা মাছি"],
        ["বোরো দান", "বোরো ধান"],
        ["আমন দান", "আমন ধান"],
        ["আউশ দান", "আউশ ধান"],
        ["ধান গাছে পুকা", "ধান গাছে পোকা"],
        ["ছত্রাক নাশক", "ছত্রাকনাশক"],
        ["কিটনাশক", "কীটনাশক"],
        ["কীট নাশক", "কীটনাশক"],
        ["বালাই নাশক", "বালাইনাশক"],
        ["আদ্রতা", "আর্দ্রতা"],
        ["আদ্রর্তা", "আর্দ্রতা"],
        ["তাপ মাত্রা", "তাপমাত্রা"],
        ["ট্রাইসাইক্লাজল", "ট্রাইসাইক্লাজোল"],
        ["পাইমেট্রজিন", "পাইমেট্রোজিন"],
        ["ম্যাংকজেব", "ম্যাংকোজেব"],
        ["ম্যানকোজেব", "ম্যাংকোজেব"],
    ];

    /**
     * Unicode-safe Bengali consecutive word & 2-3 word phrase deduplication.
     * Avoids JS ASCII \b bug which fails on Bengali combining vowel signs.
     */
    function collapseBengaliDuplicates(rawText) {
        const tokens = String(rawText || "")
            .trim()
            .split(/\s+/)
            .filter(Boolean);
        if (tokens.length < 2) {
            return tokens.join(" ");
        }
        let deduped = [];
        for (const tok of tokens) {
            if (deduped.length === 0 || deduped[deduped.length - 1] !== tok) {
                deduped.push(tok);
            }
        }
        for (const n of [3, 2]) {
            const out = [];
            let i = 0;
            while (i < deduped.length) {
                let isDup = i + 2 * n <= deduped.length;
                if (isDup) {
                    for (let k = 0; k < n; k++) {
                        if (deduped[i + k] !== deduped[i + n + k]) {
                            isDup = false;
                            break;
                        }
                    }
                }
                if (isDup) {
                    for (let k = 0; k < n; k++) {
                        out.push(deduped[i + k]);
                    }
                    i += 2 * n;
                    while (i + n <= deduped.length) {
                        let repeatMore = true;
                        for (let k = 0; k < n; k++) {
                            if (deduped[i + k] !== out[out.length - n + k]) {
                                repeatMore = false;
                                break;
                            }
                        }
                        if (!repeatMore) break;
                        i += n;
                    }
                } else {
                    out.push(deduped[i]);
                    i += 1;
                }
            }
            deduped = out;
        }
        return deduped.join(" ");
    }

    /**
     * Normalize NFC, collapse Bengali stutter, and repair agricultural STT typos.
     */
    function correctBengaliAgriTranscript(rawText) {
        let normalized = String(rawText || "").trim();
        if (typeof normalized.normalize === "function") {
            normalized = normalized.normalize("NFC");
        }
        normalized = normalized.replace(/\s+/g, " ");
        if (!normalized) {
            return { text: "", corrected: false };
        }
        let cleaned = collapseBengaliDuplicates(normalized);
        let corrected = cleaned !== normalized;
        for (const [wrong, right] of AGRI_PHONETIC_FIXES) {
            const w = typeof wrong.normalize === "function" ? wrong.normalize("NFC") : wrong;
            const r = typeof right.normalize === "function" ? right.normalize("NFC") : right;
            if (cleaned.includes(w)) {
                cleaned = cleaned.split(w).join(r);
                corrected = true;
            }
        }
        cleaned = collapseBengaliDuplicates(cleaned.replace(/\s+/g, " ").trim());
        return { text: cleaned, corrected };
    }

    // --- DOM elements ---
    let overlayEl = null;
    let closeBtnEl = null;
    let engineBadgeEl = null;
    let statusTextEl = null;
    let statusSubEl = null;
    let waveStageEl = null;
    let waveformCanvasEl = null;
    let transcriptCardEl = null;
    let correctionBadgeEl = null;
    let transcriptInputEl = null;
    let countdownWrapEl = null;
    let countdownBarEl = null;
    let countdownLabelEl = null;
    let retryBtnEl = null;
    let replyCardEl = null;
    let replyTextEl = null;
    let viewChatBtnEl = null;
    let primaryBtnEl = null;
    let primaryIconEl = null;
    let primaryLabelEl = null;
    let stopBtnEl = null;
    let stopLabelEl = null;
    let micTriggerEl = null;

    // --- Runtime state ---
    // States: "closed" | "idle" | "listening" | "transcribing" | "review" | "thinking" | "speaking"
    let state = "closed";
    let forceBrowserFallback = false;
    let activeRecognition = null;
    let committedFinalText = "";
    let latestInterimText = "";
    let webSpeechCancelled = false;

    let mediaStream = null;
    let mediaRecorder = null;
    let recordedChunks = [];
    let recordedMimeType = "audio/webm";
    let audioContext = null;
    let analyserNode = null;
    let waveAnimFrame = null;
    let syntheticEnergy = 0.08;
    let recordingCancelled = false;
    let recordStartTs = 0;
    let speechDetectedTs = 0;
    let lastVoiceTs = 0;
    let maxRecordTimer = null;

    let countdownTimer = null;
    let countdownRaf = null;
    let countdownDeadline = 0;

    function getChatBridge() {
        return window.__AgriRAGChat || null;
    }

    function getEngineConfig() {
        const bridge = getChatBridge();
        if (bridge && typeof bridge.getEngineConfig === "function") {
            return bridge.getEngineConfig();
        }
        return { engine_mode: "local", api_key: "", api_base_url: "", api_model: "" };
    }

    function hasCloudSttConfigured() {
        if (forceBrowserFallback) return false;
        const cfg = getEngineConfig();
        return cfg.engine_mode === "cloud" && Boolean(cfg.api_key && cfg.api_key.trim());
    }

    function getSpeechRecognitionCtor() {
        return window.SpeechRecognition || window.webkitSpeechRecognition || null;
    }

    function shouldUseAudioRecorder() {
        const hasRecorder =
            typeof window.MediaRecorder !== "undefined" &&
            navigator.mediaDevices &&
            typeof navigator.mediaDevices.getUserMedia === "function";
        if (!hasRecorder) return false;
        if (hasCloudSttConfigured()) return true;
        // Fallback to MediaRecorder -> backend Whisper if browser lacks Web Speech API
        return !getSpeechRecognitionCtor();
    }

    function describeActiveSttEngine() {
        if (hasCloudSttConfigured()) {
            const cfg = getEngineConfig();
            const base = (cfg.api_base_url || "").toLowerCase();
            const key = (cfg.api_key || "").trim();
            if (base.includes("generativelanguage.googleapis.com") || (!base && key.startsWith("AIza"))) {
                return "হাইব্রিড ভয়েস (Gemini অডিও)";
            }
            if (base.includes("groq.com") || (!base && key.startsWith("gsk_"))) {
                return "হাইব্রিড ভয়েস (Groq Whisper)";
            }
            return "হাইব্রিড ভয়েস (Cloud Whisper)";
        }
        if (getSpeechRecognitionCtor()) {
            return "বাংলা ভয়েস + কৃষি কারেকশন";
        }
        return "লোকাল অডিও রেকর্ডার";
    }

    function updateEngineBadge() {
        if (engineBadgeEl) {
            engineBadgeEl.textContent = describeActiveSttEngine();
        }
    }

    // ----------------------------------------------------------------
    // Waveform Rendering (Real AnalyserNode or Smooth Reactive Wave)
    // ----------------------------------------------------------------

    function stopWaveformLoop() {
        if (waveAnimFrame) {
            cancelAnimationFrame(waveAnimFrame);
            waveAnimFrame = null;
        }
    }

    function startWaveformLoop() {
        stopWaveformLoop();
        if (!waveformCanvasEl) return;
        const ctx = waveformCanvasEl.getContext("2d");
        if (!ctx) return;

        const barCount = 29;
        const timeData = analyserNode ? new Uint8Array(analyserNode.fftSize) : null;

        function render(ts) {
            const w = waveformCanvasEl.width;
            const h = waveformCanvasEl.height;
            ctx.clearRect(0, 0, w, h);

            let energy = 0.08;
            if (state === "listening" && analyserNode && timeData) {
                analyserNode.getByteTimeDomainData(timeData);
                let sumSq = 0;
                for (let i = 0; i < timeData.length; i++) {
                    const v = (timeData[i] - 128) / 128;
                    sumSq += v * v;
                }
                const rms = Math.sqrt(sumSq / timeData.length);
                energy = Math.min(1, Math.max(0.08, rms * 5.2));
                handleVadSample(rms);
            } else if (state === "listening") {
                // Smooth lively breathing wave when Web Speech owns Android mic
                syntheticEnergy = Math.max(0.32, syntheticEnergy * 0.94);
                energy = Math.min(0.98, syntheticEnergy + 0.14 * Math.abs(Math.sin(ts / 220)));
            } else if (state === "transcribing" || state === "thinking") {
                energy = 0.24 + 0.14 * Math.sin(ts / 260);
            } else if (state === "speaking") {
                energy = 0.42 + 0.28 * Math.abs(Math.sin(ts / 170)) * Math.abs(Math.cos(ts / 310));
            } else {
                energy = 0.08;
            }

            const centerY = h / 2;
            const barWidth = 4;
            const gap = (w - barCount * barWidth) / (barCount + 1);
            const maxBarHeight = h * 0.82;

            let color = "#10b981";
            if (state === "transcribing" || state === "thinking") {
                color = "#38bdf8";
            } else if (state === "speaking") {
                color = "#34d399";
            } else if (state === "review" || state === "idle") {
                color = "#64748b";
            }
            ctx.fillStyle = color;

            for (let i = 0; i < barCount; i++) {
                const distFromCenter = Math.abs(i - (barCount - 1) / 2) / ((barCount - 1) / 2);
                const envelope = Math.max(0.12, 1 - distFromCenter * distFromCenter);
                const ripple =
                    0.55 + 0.45 * Math.sin(ts / 130 + i * 0.62) * Math.cos(ts / 240 - i * 0.35);
                const barH = Math.max(4, maxBarHeight * envelope * energy * ripple);
                const x = gap + i * (barWidth + gap);
                const y = centerY - barH / 2;
                ctx.beginPath();
                if (typeof ctx.roundRect === "function") {
                    ctx.roundRect(x, y, barWidth, barH, 3);
                } else {
                    ctx.rect(x, y, barWidth, barH);
                }
                ctx.fill();
            }

            if (state !== "closed") {
                waveAnimFrame = requestAnimationFrame(render);
            }
        }

        waveAnimFrame = requestAnimationFrame(render);
    }

    function handleVadSample(rms) {
        if (!mediaRecorder || mediaRecorder.state !== "recording") return;
        const now = Date.now();
        const elapsedSec = Math.floor((now - recordStartTs) / 1000);
        if (statusSubEl) {
            const mm = String(Math.floor(elapsedSec / 60)).padStart(1, "0");
            const ss = String(elapsedSec % 60).padStart(2, "0");
            statusSubEl.textContent = `রেকর্ড হচ্ছে (${mm}:${ss}) · কথা শেষে "বলা শেষ" চাপুন বা ২ সেকেন্ড থামুন`;
        }
        if (rms >= VAD_SPEECH_RMS) {
            if (!speechDetectedTs) {
                speechDetectedTs = now;
            }
            lastVoiceTs = now;
        } else if (
            speechDetectedTs &&
            lastVoiceTs &&
            lastVoiceTs - speechDetectedTs >= VAD_MIN_SPEECH_MS &&
            now - lastVoiceTs >= VAD_SILENCE_MS
        ) {
            finishAudioRecording();
        }
    }

    // ----------------------------------------------------------------
    // Overlay UI State Management
    // ----------------------------------------------------------------

    function setPrimaryIcon(kind) {
        if (!primaryIconEl) return;
        if (kind === "send") {
            primaryIconEl.innerHTML =
                '<path d="M22 2 11 13"></path><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>';
        } else if (kind === "check") {
            primaryIconEl.innerHTML = '<polyline points="20 6 9 17 4 12"></polyline>';
        } else {
            primaryIconEl.innerHTML =
                '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path>' +
                '<path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>' +
                '<line x1="12" y1="19" x2="12" y2="22"></line>';
        }
    }

    function setState(nextState, opts) {
        const options = opts || {};
        state = nextState;
        if (!overlayEl) return;

        overlayEl.dataset.vaState = nextState;
        if (waveStageEl) {
            waveStageEl.classList.toggle("is-listening", nextState === "listening");
            waveStageEl.classList.toggle("is-speaking", nextState === "speaking");
            waveStageEl.classList.toggle(
                "is-thinking",
                nextState === "thinking" || nextState === "transcribing",
            );
        }

        if (primaryBtnEl) {
            primaryBtnEl.disabled = nextState === "transcribing" || nextState === "thinking";
            primaryBtnEl.classList.toggle("va-btn-recording", nextState === "listening");
            primaryBtnEl.classList.toggle("va-btn-confirm", nextState === "review");
        }

        if (retryBtnEl) {
            retryBtnEl.classList.toggle("va-hidden", nextState !== "review" && nextState !== "idle");
        }

        if (replyCardEl) {
            const showReply = nextState === "thinking" || nextState === "speaking";
            replyCardEl.classList.toggle("va-hidden", !showReply);
        }

        if (transcriptCardEl) {
            const hideTranscript = nextState === "thinking" || nextState === "speaking";
            transcriptCardEl.classList.toggle("va-hidden", hideTranscript);
        }

        if (nextState !== "review") {
            clearAutoSendCountdown();
        }

        if (nextState === "idle") {
            if (statusTextEl) statusTextEl.textContent = options.title || "কথা বলতে 'বলুন' চাপুন";
            if (statusSubEl) {
                statusSubEl.textContent =
                    options.sub || "বাংলায় আপনার ফসলের রোগ, পোকা বা আবহাওয়া সম্পর্কে জিজ্ঞাসা করুন";
            }
            setPrimaryIcon("mic");
            if (primaryLabelEl) primaryLabelEl.textContent = "বলুন";
            if (stopLabelEl) stopLabelEl.textContent = "বন্ধ করুন";
        } else if (nextState === "listening") {
            if (statusTextEl) statusTextEl.textContent = "শুনছি... বাংলায় বলুন";
            if (statusSubEl) {
                statusSubEl.textContent = shouldUseAudioRecorder()
                    ? 'পরিষ্কারভাবে বলুন · শেষে "বলা শেষ" চাপুন বা ২ সেকেন্ড চুপ থাকুন'
                    : 'বাংলায় বলুন · কথা শেষে "বলা শেষ" চাপুন';
            }
            if (correctionBadgeEl) correctionBadgeEl.classList.add("va-hidden");
            setPrimaryIcon("check");
            if (primaryLabelEl) primaryLabelEl.textContent = "বলা শেষ";
            if (stopLabelEl) stopLabelEl.textContent = "থামান";
        } else if (nextState === "transcribing") {
            if (statusTextEl) statusTextEl.textContent = "কথা যাচাই করছি...";
            if (statusSubEl) statusSubEl.textContent = "বাংলা কৃষি শব্দগুলো মিলিয়ে দেখা হচ্ছে";
            setPrimaryIcon("mic");
            if (primaryLabelEl) primaryLabelEl.textContent = "অপেক্ষা…";
            if (stopLabelEl) stopLabelEl.textContent = "থামান";
        } else if (nextState === "review") {
            if (statusTextEl) statusTextEl.textContent = "কথাটি কি ঠিক আছে?";
            if (statusSubEl) {
                statusSubEl.textContent =
                    options.sub || "ভুল থাকলে লেখাটির ওপর ট্যাপ করে ঠিক করুন বা আবার বলুন";
            }
            setPrimaryIcon("send");
            if (primaryLabelEl) primaryLabelEl.textContent = "পাঠান";
            if (stopLabelEl) stopLabelEl.textContent = "থামান";
        } else if (nextState === "thinking") {
            if (statusTextEl) statusTextEl.textContent = "উত্তর খুঁজছি...";
            if (statusSubEl) statusSubEl.textContent = "BRRI / BARI নির্দেশিকা ও সেন্সর যাচাই চলছে";
            if (replyTextEl) replyTextEl.textContent = "অনুগ্রহ করে একটু অপেক্ষা করুন...";
            setPrimaryIcon("mic");
            if (primaryLabelEl) primaryLabelEl.textContent = "অপেক্ষা…";
            if (stopLabelEl) stopLabelEl.textContent = "থামান";
        } else if (nextState === "speaking") {
            if (statusTextEl) statusTextEl.textContent = "পরামর্শ বলছি...";
            if (statusSubEl) {
                statusSubEl.textContent = "থামাতে লাল STOP চাপুন অথবা নতুন প্রশ্ন করতে বলুন চাপুন";
            }
            setPrimaryIcon("mic");
            if (primaryLabelEl) primaryLabelEl.textContent = "বলুন";
            if (stopLabelEl) stopLabelEl.textContent = "থামান";
        }
    }

    // ----------------------------------------------------------------
    // Cancelable Auto-Send Countdown
    // ----------------------------------------------------------------

    function clearAutoSendCountdown() {
        if (countdownTimer) {
            clearTimeout(countdownTimer);
            countdownTimer = null;
        }
        if (countdownRaf) {
            cancelAnimationFrame(countdownRaf);
            countdownRaf = null;
        }
        if (countdownWrapEl) {
            countdownWrapEl.classList.add("va-hidden");
        }
        if (countdownBarEl) {
            countdownBarEl.style.width = "100%";
        }
    }

    function startAutoSendCountdown() {
        clearAutoSendCountdown();
        const bridge = getChatBridge();
        const autoOn = bridge && typeof bridge.isAutoSendEnabled === "function"
            ? bridge.isAutoSendEnabled()
            : false;
        if (!autoOn) return;

        if (!countdownWrapEl || !countdownBarEl) return;
        countdownWrapEl.classList.remove("va-hidden");
        const startTs = performance.now();
        countdownDeadline = startTs + AUTO_SEND_DELAY_MS;

        function tick(now) {
            if (state !== "review") {
                clearAutoSendCountdown();
                return;
            }
            const remaining = Math.max(0, countdownDeadline - now);
            const pct = (remaining / AUTO_SEND_DELAY_MS) * 100;
            countdownBarEl.style.width = `${pct.toFixed(1)}%`;
            if (countdownLabelEl) {
                const sec = (remaining / 1000).toFixed(1);
                countdownLabelEl.textContent = `${sec} সেকেন্ডে পাঠানো হবে · ঠিক করতে টেক্সটে ট্যাপ করুন`;
            }
            if (remaining > 0) {
                countdownRaf = requestAnimationFrame(tick);
            }
        }
        countdownRaf = requestAnimationFrame(tick);

        countdownTimer = setTimeout(() => {
            countdownTimer = null;
            if (state === "review") {
                submitReviewedTranscript();
            }
        }, AUTO_SEND_DELAY_MS);
    }

    // ----------------------------------------------------------------
    // Audio Recording (Path 1: Cloud STT / Offline Whisper via /api/voice)
    // ----------------------------------------------------------------

    function cleanupMediaResources() {
        if (maxRecordTimer) {
            clearTimeout(maxRecordTimer);
            maxRecordTimer = null;
        }
        if (mediaRecorder) {
            try {
                if (mediaRecorder.state !== "inactive") {
                    mediaRecorder.stop();
                }
            } catch (_) {
                /* ignore */
            }
            mediaRecorder = null;
        }
        if (mediaStream) {
            mediaStream.getTracks().forEach((t) => {
                try {
                    t.stop();
                } catch (_) {
                    /* ignore */
                }
            });
            mediaStream = null;
        }
        if (audioContext) {
            try {
                audioContext.close();
            } catch (_) {
                /* ignore */
            }
            audioContext = null;
            analyserNode = null;
        }
    }

    function pickSupportedAudioMime() {
        const candidates = [
            "audio/webm;codecs=opus",
            "audio/webm",
            "audio/mp4",
            "audio/ogg;codecs=opus",
        ];
        if (typeof window.MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") {
            return "";
        }
        for (const mime of candidates) {
            if (MediaRecorder.isTypeSupported(mime)) {
                return mime;
            }
        }
        return "";
    }

    async function startAudioRecording() {
        cleanupMediaResources();
        recordingCancelled = false;
        recordedChunks = [];
        speechDetectedTs = 0;
        lastVoiceTs = 0;

        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    channelCount: 1,
                },
            });
        } catch (err) {
            // Permission denied or mic busy -> try browser Web Speech if available
            if (getSpeechRecognitionCtor()) {
                forceBrowserFallback = true;
                updateEngineBadge();
                startWebSpeechRecognition();
                return;
            }
            setState("idle", {
                title: "মাইক্রোফোন অনুমতি পাওয়া যায়নি",
                sub: "ব্রাউজার সেটিংস থেকে মাইক্রোফোন Allow করে আবার বলুন চাপুন",
            });
            return;
        }

        // Set up real-time AnalyserNode for acoustic waveform + VAD
        try {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {
                audioContext = new AudioCtx();
                const source = audioContext.createMediaStreamSource(mediaStream);
                analyserNode = audioContext.createAnalyser();
                analyserNode.fftSize = 256;
                source.connect(analyserNode);
            }
        } catch (_) {
            analyserNode = null;
        }

        const chosenMime = pickSupportedAudioMime();
        recordedMimeType = chosenMime || "audio/webm";
        const recorderOpts = chosenMime ? { mimeType: chosenMime } : undefined;

        try {
            mediaRecorder = new MediaRecorder(mediaStream, recorderOpts);
        } catch (_) {
            mediaRecorder = new MediaRecorder(mediaStream);
        }

        mediaRecorder.ondataavailable = (evt) => {
            if (evt.data && evt.data.size > 0) {
                recordedChunks.push(evt.data);
            }
        };

        mediaRecorder.onstop = () => {
            const chunks = recordedChunks.slice();
            const mime = recordedMimeType;
            const wasCancelled = recordingCancelled;
            cleanupMediaResources();
            if (wasCancelled || state === "closed") {
                return;
            }
            const blob = new Blob(chunks, { type: mime });
            if (blob.size < 600) {
                setState("idle", {
                    title: "স্পষ্ট কথা শোনা যায়নি",
                    sub: "মাইকের কাছে স্পষ্ট করে বাংলায় আবার বলুন",
                });
                return;
            }
            transcribeRecordedBlob(blob, mime);
        };

        if (transcriptInputEl) {
            transcriptInputEl.value = "";
            transcriptInputEl.placeholder = "আপনার কথা রেকর্ড হচ্ছে...";
        }
        recordStartTs = Date.now();
        mediaRecorder.start(250);
        setState("listening");
        startWaveformLoop();

        maxRecordTimer = setTimeout(() => {
            finishAudioRecording();
        }, MAX_RECORDING_MS);
    }

    function finishAudioRecording() {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            setState("transcribing");
            try {
                mediaRecorder.stop();
            } catch (_) {
                cleanupMediaResources();
            }
        }
    }

    function blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                const res = String(reader.result || "");
                const idx = res.indexOf(",");
                resolve(idx >= 0 ? res.slice(idx + 1) : res);
            };
            reader.onerror = () => reject(new Error("Audio encoding failed"));
            reader.readAsDataURL(blob);
        });
    }

    async function transcribeRecordedBlob(blob, mimeType) {
        setState("transcribing");
        try {
            const b64 = await blobToBase64(blob);
            const cfg = getEngineConfig();
            const body = {
                audio_base64: b64,
                audio_mime_type: mimeType,
                language: "bn-BD",
                run_rag: false,
            };
            if (cfg.engine_mode === "cloud" && cfg.api_key) {
                body.engine_mode = "cloud";
                body.api_key = cfg.api_key;
                body.api_base_url = cfg.api_base_url;
                body.api_model = cfg.api_model;
            }

            const resp = await fetch("/api/voice", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            const data = await resp.json();
            if (state === "closed") return;

            const tr = data && data.transcription ? data.transcription : null;
            if (resp.ok && tr && tr.ok && tr.transcript && tr.transcript.trim().length >= 2) {
                enterReviewState(tr.transcript.trim(), Boolean(tr.corrected), tr.engine);
                return;
            }

            // Cloud/Whisper failed -> gracefully offer browser Web Speech fallback
            const reason =
                (tr && tr.fallback_reason) ||
                "কথাটি পরিষ্কার বোঝা যায়নি। আবার বলুন অথবা নিচে টাইপ করুন।";
            if (getSpeechRecognitionCtor() && hasCloudSttConfigured()) {
                forceBrowserFallback = true;
                updateEngineBadge();
                setState("idle", {
                    title: "ক্লাউড ভয়েস পাওয়া যায়নি",
                    sub: "ব্রাউজার বাংলা ভয়েস চালু করা হয়েছে — আবার 'বলুন' চাপুন",
                });
            } else {
                setState("idle", {
                    title: "আবার চেষ্টা করুন",
                    sub: reason,
                });
            }
        } catch (err) {
            if (state === "closed") return;
            if (getSpeechRecognitionCtor()) {
                forceBrowserFallback = true;
                updateEngineBadge();
            }
            setState("idle", {
                title: "ভয়েস যাচাইয়ে সমস্যা হয়েছে",
                sub: "নেটওয়ার্ক বা সার্ভার চেক করে আবার 'বলুন' চাপুন",
            });
        }
    }

    // ----------------------------------------------------------------
    // Fixed Web Speech API (Path 2: Zero-RAM Local / Offline Bengali STT)
    // ----------------------------------------------------------------

    function stopWebSpeech() {
        if (activeRecognition) {
            try {
                activeRecognition.abort();
            } catch (_) {
                /* ignore */
            }
            activeRecognition = null;
        }
    }

    function startWebSpeechRecognition() {
        stopWebSpeech();
        const Ctor = getSpeechRecognitionCtor();
        if (!Ctor) {
            setState("idle", {
                title: "ভয়েস সাপোর্ট নেই",
                sub: "এই ব্রাউজারে মাইক সাপোর্ট নেই — Chrome ব্যবহার করুন বা ক্লাউড মোড চালু করুন",
            });
            return;
        }

        webSpeechCancelled = false;
        committedFinalText = "";
        latestInterimText = "";
        syntheticEnergy = 0.25;
        let speechStarted = false;

        if (statusTextEl) {
            statusTextEl.textContent = "মাইক চালু হচ্ছে...";
        }
        if (statusSubEl) {
            statusSubEl.textContent = "ব্রাউজার মাইক্রোফোন অনুমতি চাইলে Allow চাপুন";
        }

        const rec = new Ctor();
        rec.lang = "bn-BD";
        // Critical Android fix: continuous=true causes overlapping phrase corruption on Android Chrome
        rec.continuous = !IS_ANDROID;
        rec.interimResults = true;
        rec.maxAlternatives = 3;
        activeRecognition = rec;

        rec.onstart = () => {
            if (state === "closed") return;
            speechStarted = true;
            if (transcriptInputEl) {
                transcriptInputEl.value = "";
                transcriptInputEl.placeholder = "শুনছি... বাংলায় বলুন...";
            }
            setState("listening");
            startWaveformLoop();
        };

        rec.onsoundstart = () => {
            syntheticEnergy = 0.65;
        };

        rec.onspeechstart = () => {
            syntheticEnergy = 0.88;
        };

        rec.onresult = (event) => {
            if (webSpeechCancelled || state !== "listening") return;
            syntheticEnergy = 0.92;

            let interim = "";
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const res = event.results[i];
                const best = pickBestBengaliAlternative(res);
                if (!best) continue;
                if (res.isFinal) {
                    // Only append genuinely new final segment (guards Android restarts)
                    if (!committedFinalText) {
                        committedFinalText = best;
                    } else if (best.startsWith(committedFinalText)) {
                        committedFinalText = best;
                    } else if (!committedFinalText.endsWith(best)) {
                        committedFinalText = `${committedFinalText} ${best}`.trim();
                    }
                } else {
                    interim = interim ? `${interim} ${best}` : best;
                }
            }
            latestInterimText = interim.trim();

            const liveCombined = [committedFinalText, latestInterimText]
                .filter(Boolean)
                .join(" ")
                .trim();
            if (transcriptInputEl) {
                const repaired = correctBengaliAgriTranscript(liveCombined);
                transcriptInputEl.value = repaired.text;
            }
        };

        rec.onerror = (event) => {
            if (webSpeechCancelled || state === "closed") return;
            const errCode = event && event.error ? event.error : "unknown";
            activeRecognition = null;
            if (errCode === "not-allowed" || errCode === "service-not-allowed") {
                setState("idle", {
                    title: "মাইক্রোফোন অনুমতি বন্ধ আছে",
                    sub: "ব্রাউজারের লক/টিউন আইকন থেকে Microphone Allow করে আবার 'বলুন' চাপুন, অথবা নিচের বক্সে লিখে পাঠান",
                });
                if (transcriptInputEl) transcriptInputEl.focus();
            } else if (errCode === "network") {
                setState("idle", {
                    title: "অফলাইন মোড — নিচের বক্সে লিখে পাঠান",
                    sub: "ইন্টারনেট ছাড়া ব্রাউজারের ভয়েস সার্ভিস বন্ধ থাকে, তবে নিচের বক্সে লিখে পাঠালে অফলাইন AI শতভাগ কাজ করবে",
                });
                if (transcriptInputEl) transcriptInputEl.focus();
            } else if (errCode === "no-speech") {
                setState("idle", {
                    title: "কোনো কথা শোনা যায়নি",
                    sub: "আবার 'বলুন' চেপে মাইকের কাছে বাংলায় প্রশ্ন করুন",
                });
            } else {
                setState("idle", {
                    title: "ভয়েস চিনতে সমস্যা হয়েছে",
                    sub: "আবার 'বলুন' চাপুন অথবা নিচের বক্সে বাংলায় টাইপ করে পাঠান",
                });
                if (transcriptInputEl) transcriptInputEl.focus();
            }
        };

        rec.onend = () => {
            activeRecognition = null;
            if (webSpeechCancelled || state === "closed") {
                return;
            }
            if (!speechStarted) {
                const isOffline = typeof navigator !== "undefined" && navigator.onLine === false;
                setState("idle", {
                    title: isOffline
                        ? "অফলাইন মোড — নিচের বক্সে লিখে পাঠান"
                        : "মাইক্রোফোন চালু হয়নি",
                    sub: isOffline
                        ? "ইন্টারনেট ছাড়া ব্রাউজারের ভয়েস সার্ভিস বন্ধ থাকে — নিচের বক্সে প্রশ্ন লিখে পাঠান"
                        : "ব্রাউজারের অ্যাড্রেস বারের আইকন থেকে Microphone Allow করুন, অথবা নিচের বক্সে বাংলায় লিখে পাঠান",
                });
                if (transcriptInputEl) transcriptInputEl.focus();
                return;
            }
            if (state !== "listening") {
                return;
            }
            const rawFinal = [committedFinalText, latestInterimText]
                .filter(Boolean)
                .join(" ")
                .trim();
            const repaired = correctBengaliAgriTranscript(rawFinal);
            if (repaired.text.length < 3) {
                setState("idle", {
                    title: "স্পষ্ট কথা শোনা যায়নি",
                    sub: "একটু পরিষ্কার করে আবার বলুন",
                });
                return;
            }
            enterReviewState(repaired.text, repaired.corrected, "web_speech");
        };

        try {
            rec.start();
        } catch (_) {
            setState("idle", {
                title: "মাইক চালু করা যায়নি",
                sub: "এক মুহূর্ত পর আবার 'বলুন' চাপুন",
            });
        }
    }

    /**
     * When Web Speech returns multiple alternatives, prefer one containing known
     * Bengali agricultural terms if confidence is close to top-1.
     */
    function pickBestBengaliAlternative(speechRecognitionResult) {
        if (!speechRecognitionResult || speechRecognitionResult.length === 0) {
            return "";
        }
        const top = speechRecognitionResult[0];
        const topText = (top && top.transcript ? top.transcript : "").trim();
        if (speechRecognitionResult.length === 1) {
            return topText;
        }
        const agriHints = [
            "ধান", "ব্লাস্ট", "পোকা", "কারেন্ট", "মাজরা", "মড়ক", "আলু",
            "বেগুন", "টমেটো", "ছত্রাক", "কীটনাশক", "আবহাওয়া", "আর্দ্রতা", "সার",
        ];
        let bestText = topText;
        let bestHits = agriHints.filter((w) => topText.includes(w)).length;
        for (let a = 1; a < speechRecognitionResult.length; a++) {
            const alt = speechRecognitionResult[a];
            const altText = (alt && alt.transcript ? alt.transcript : "").trim();
            if (!altText) continue;
            const altHits = agriHints.filter((w) => altText.includes(w)).length;
            if (altHits > bestHits) {
                bestText = altText;
                bestHits = altHits;
            }
        }
        return bestText;
    }

    function finishWebSpeech() {
        if (activeRecognition) {
            try {
                activeRecognition.stop();
            } catch (_) {
                /* ignore */
            }
        }
    }

    // ----------------------------------------------------------------
    // Review -> Submit -> Thinking -> Speaking Flow
    // ----------------------------------------------------------------

    function enterReviewState(cleanedText, wasCorrected, engineName) {
        if (transcriptInputEl) {
            transcriptInputEl.value = cleanedText;
        }
        if (correctionBadgeEl) {
            correctionBadgeEl.classList.toggle("va-hidden", !wasCorrected);
        }
        const engineTag =
            engineName && engineName.startsWith("cloud_")
                ? "ক্লাউড AI দিয়ে যাচাইকৃত · ঠিক থাকলে পাঠান"
                : "ঠিক থাকলে পাঠান, অথবা টেক্সটে ট্যাপ করে সংশোধন করুন";
        setState("review", { sub: engineTag });
        startAutoSendCountdown();
    }

    function submitReviewedTranscript() {
        clearAutoSendCountdown();
        const raw = transcriptInputEl ? transcriptInputEl.value.trim() : "";
        const repaired = correctBengaliAgriTranscript(raw);
        const query = repaired.text;
        if (query.length < 2) {
            setState("idle", {
                title: "প্রশ্নটি খালি আছে",
                sub: "অনুগ্রহ করে 'বলুন' চেপে আপনার প্রশ্নটি বলুন",
            });
            return;
        }

        const bridge = getChatBridge();
        if (!bridge || typeof bridge.submitQuery !== "function") {
            closeOverlay();
            return;
        }

        setState("thinking");
        startWaveformLoop();

        bridge.setVoiceEventCallbacks({
            onStatus: (statusLine) => {
                if (state === "thinking" && statusSubEl && statusLine) {
                    statusSubEl.textContent = statusLine;
                }
            },
            onToken: (_delta, fullSoFar) => {
                if ((state === "thinking" || state === "speaking") && replyTextEl) {
                    replyTextEl.textContent = cleanReplyPreview(fullSoFar);
                }
            },
            onDone: (finalText) => {
                if (state === "closed") return;
                if (replyTextEl) {
                    replyTextEl.textContent = cleanReplyPreview(finalText);
                }
                const ttsOn = bridge.isTtsEnabled ? bridge.isTtsEnabled() : true;
                if (!ttsOn) {
                    setState("idle", {
                        title: "উত্তর তৈরি হয়েছে",
                        sub: "চ্যাটে বিস্তারিত দেখুন অথবা নতুন প্রশ্ন করতে 'বলুন' চাপুন",
                    });
                }
            },
            onSpeechStart: () => {
                if (state !== "closed") {
                    setState("speaking");
                    startWaveformLoop();
                }
            },
            onSpeechEnd: () => {
                if (state === "speaking") {
                    setState("idle", {
                        title: "আরও কিছু জানতে চান?",
                        sub: "নতুন প্রশ্ন করতে 'বলুন' চাপুন অথবা চ্যাটে ফিরুন",
                    });
                }
            },
            onError: (errMsg) => {
                if (state !== "closed") {
                    setState("idle", {
                        title: "উত্তর আনতে সমস্যা হয়েছে",
                        sub: errMsg || "আবার চেষ্টা করুন",
                    });
                }
            },
        });

        bridge.submitQuery(query);
    }

    function cleanReplyPreview(text) {
        return String(text || "")
            .replace(/[*_#`~]+/g, "")
            .replace(/\s+/g, " ")
            .trim()
            .slice(0, 320);
    }

    // ----------------------------------------------------------------
    // User Actions (Open, Close, Primary Button, Red STOP Button)
    // ----------------------------------------------------------------

    function redirectZeroHostToLocalhost() {
        if (window.location && window.location.hostname === "0.0.0.0") {
            const portPart = window.location.port ? `:${window.location.port}` : "";
            const nextUrl = `${window.location.protocol}//localhost${portPart}${window.location.pathname}${window.location.search}${window.location.hash}`;
            window.location.replace(nextUrl);
            return true;
        }
        return false;
    }

    function beginListening() {
        if (redirectZeroHostToLocalhost()) {
            return;
        }
        const bridge = getChatBridge();
        if (bridge && typeof bridge.stopSpeech === "function") {
            bridge.stopSpeech();
        }
        clearAutoSendCountdown();
        updateEngineBadge();

        // Chrome blocks mic & Web Speech API on plain HTTP non-localhost IPs (http://192.168.x.x).
        if (window.isSecureContext === false) {
            // If user already typed a question in the box, submit it directly!
            if (transcriptInputEl && transcriptInputEl.value.trim().length >= 2) {
                submitReviewedTranscript();
                return;
            }
            const portPart = window.location.port ? `:${window.location.port}` : ":8000";
            setState("idle", {
                title: "মাইকের জন্য localhost সংযোগ প্রয়োজন",
                sub: `Chrome নিরাপত্তার কারণে ${window.location.host} ঠিকানায় মাইক ব্লক করে। পিসিতে বা USB ফোনে http://localhost${portPart} খুলুন, অথবা নিচের বক্সে লিখে পাঠান।`,
            });
            if (transcriptInputEl) {
                transcriptInputEl.focus();
            }
            return;
        }

        if (shouldUseAudioRecorder()) {
            startAudioRecording();
        } else {
            startWebSpeechRecognition();
        }
    }

    function openOverlay(autoStartListening) {
        if (redirectZeroHostToLocalhost()) {
            return;
        }
        if (!overlayEl) return;
        forceBrowserFallback = false;
        updateEngineBadge();
        overlayEl.hidden = false;
        overlayEl.classList.remove("va-hidden");
        document.body.classList.add("va-open");
        setState("idle");
        startWaveformLoop();
        if (autoStartListening) {
            beginListening();
        }
    }

    function closeOverlay() {
        clearAutoSendCountdown();
        recordingCancelled = true;
        webSpeechCancelled = true;
        cleanupMediaResources();
        stopWebSpeech();
        stopWaveformLoop();
        const bridge = getChatBridge();
        if (bridge && typeof bridge.setVoiceEventCallbacks === "function") {
            bridge.setVoiceEventCallbacks(null);
        }
        state = "closed";
        if (overlayEl) {
            overlayEl.classList.add("va-hidden");
            overlayEl.hidden = true;
        }
        document.body.classList.remove("va-open");
    }

    function handlePrimaryAction() {
        if (state === "idle") {
            if (transcriptInputEl && transcriptInputEl.value.trim().length >= 2) {
                submitReviewedTranscript();
                return;
            }
            beginListening();
        } else if (state === "speaking") {
            beginListening();
        } else if (state === "listening") {
            if (mediaRecorder && mediaRecorder.state === "recording") {
                finishAudioRecording();
            } else {
                finishWebSpeech();
            }
        } else if (state === "review") {
            submitReviewedTranscript();
        }
    }

    function handleStopAction() {
        clearAutoSendCountdown();
        const bridge = getChatBridge();
        if (state === "listening") {
            recordingCancelled = true;
            webSpeechCancelled = true;
            cleanupMediaResources();
            stopWebSpeech();
            setState("idle", {
                title: "রেকর্ডিং বাতিল করা হয়েছে",
                sub: "আবার কথা বলতে 'বলুন' চাপুন",
            });
        } else if (state === "review") {
            if (transcriptInputEl) transcriptInputEl.value = "";
            setState("idle", {
                title: "পাঠানো বাতিল করা হয়েছে",
                sub: "আবার স্পষ্ট করে প্রশ্ন বলতে 'বলুন' চাপুন",
            });
        } else if (state === "speaking" || state === "thinking") {
            if (bridge && typeof bridge.stopSpeech === "function") {
                bridge.stopSpeech();
            }
            setState("idle", {
                title: "ভয়েস থামানো হয়েছে",
                sub: "নতুন প্রশ্ন করতে 'বলুন' চাপুন",
            });
        } else {
            closeOverlay();
        }
    }

    // ----------------------------------------------------------------
    // Initialization
    // ----------------------------------------------------------------

    function init() {
        if (redirectZeroHostToLocalhost()) {
            return;
        }
        overlayEl = document.getElementById("voice-agent-overlay");
        closeBtnEl = document.getElementById("va-close-btn");
        engineBadgeEl = document.getElementById("va-engine-badge");
        statusTextEl = document.getElementById("va-status-text");
        statusSubEl = document.getElementById("va-status-sub");
        waveStageEl = document.getElementById("va-wave-stage");
        waveformCanvasEl = document.getElementById("va-waveform");
        transcriptCardEl = document.getElementById("va-transcript-card");
        correctionBadgeEl = document.getElementById("va-correction-badge");
        transcriptInputEl = document.getElementById("va-transcript-input");
        countdownWrapEl = document.getElementById("va-countdown-wrap");
        countdownBarEl = document.getElementById("va-countdown-bar");
        countdownLabelEl = document.getElementById("va-countdown-label");
        retryBtnEl = document.getElementById("va-retry-btn");
        replyCardEl = document.getElementById("va-reply-card");
        replyTextEl = document.getElementById("va-reply-text");
        viewChatBtnEl = document.getElementById("va-view-chat-btn");
        primaryBtnEl = document.getElementById("va-primary-btn");
        primaryIconEl = document.getElementById("va-primary-icon");
        primaryLabelEl = document.getElementById("va-primary-label");
        stopBtnEl = document.getElementById("va-stop-btn");
        stopLabelEl = document.getElementById("va-stop-label");
        micTriggerEl = document.getElementById("mic-btn");

        if (!overlayEl || !micTriggerEl) {
            return;
        }

        // Register globally so chat.js defers #mic-btn clicks to the Voice Agent overlay
        window.__AgriRAGVoiceAgent = {
            open: () => openOverlay(true),
            close: closeOverlay,
            correctBengaliAgriTranscript,
            collapseBengaliDuplicates,
        };

        micTriggerEl.addEventListener("click", (evt) => {
            evt.preventDefault();
            openOverlay(true);
        });

        if (closeBtnEl) closeBtnEl.addEventListener("click", closeOverlay);
        if (viewChatBtnEl) viewChatBtnEl.addEventListener("click", closeOverlay);
        if (primaryBtnEl) primaryBtnEl.addEventListener("click", handlePrimaryAction);
        if (stopBtnEl) stopBtnEl.addEventListener("click", handleStopAction);
        if (retryBtnEl) retryBtnEl.addEventListener("click", beginListening);

        // Pause auto-send immediately if the user taps or edits the transcript box,
        // and switch idle -> review ("পাঠান") if the user types manually!
        if (transcriptInputEl) {
            const onFocusOrEdit = () => {
                if (countdownTimer) {
                    clearAutoSendCountdown();
                    if (statusSubEl) {
                        statusSubEl.textContent = "সম্পাদনা শেষ হলে নিচের 'পাঠান' বাটনে চাপুন";
                    }
                }
                const val = transcriptInputEl.value.trim();
                if (state === "idle" && val.length >= 2) {
                    webSpeechCancelled = true;
                    stopWebSpeech();
                    setState("review", {
                        sub: "লেখা শেষ হলে নিচের 'পাঠান' বাটনে চাপুন",
                    });
                }
            };
            transcriptInputEl.addEventListener("focus", onFocusOrEdit);
            transcriptInputEl.addEventListener("input", onFocusOrEdit);
            transcriptInputEl.addEventListener("keydown", (e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    submitReviewedTranscript();
                }
            });
        }

        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && state !== "closed") {
                closeOverlay();
            }
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
