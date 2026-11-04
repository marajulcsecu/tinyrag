/* TinyRAG — chat client (Step 4.21).
 *
 * Vanilla ES2017+ JavaScript, no framework, no build step. Loaded via
 * <script src="/static/chat.js" defer> in ui/templates/index.html;
 * served at /static/chat.js by FastAPI's StaticFiles mount in main.py.
 *
 * Responsibilities:
 *   1. Poll /api/status every STATUS_POLL_MS to keep the topbar pill
 *      + active-model label fresh. Failures leave the pill in its
 *      current state — no console-spam loops.
 *   2. Intercept the composer form's submit, POST the user's query to
 *      /api/query?stream=true using fetch() + a ReadableStream reader,
 *      and append the streamed SSE tokens into an assistant bubble.
 *   3. On the terminal `event: done` frame, render the citation cards
 *      + diagnostics footer from the Answer.to_dict() payload.
 *   4. On any `event: error` frame (or a network failure), render a
 *      red error bubble and re-enable the composer.
 *
 * SSE wire format (matches Step 4.19's StreamingResponse):
 *   event: token
 *   data: {"delta": "..."}
 *
 *   event: done
 *   data: {"answer": {...Answer.to_dict()...}}
 *
 *   event: error
 *   data: {"error": "...", "code": "..."}
 *
 *   (a blank line terminates each frame; full frames end with \n\n)
 *
 * Why fetch() + reader, not EventSource?
 * ----------------------------------------
 * EventSource only supports GET. Our endpoint is POST with a JSON
 * body (AskRequest schema), so we have to use fetch() + reader to
 * consume the text/event-stream response. The reader approach also
 * gives us a single fetch() promise so we can catch network failures
 * uniformly (EventSource swallows some fetch errors silently).
 */

(function () {
    "use strict";

    // --- Config knobs (all overridable from window.__TINYRAG_CONFIG ---
    // useful for the test suite to inject a fake host).
    const CFG = Object.assign(
        {
            statusUrl: "/api/status",
            queryUrl: "/api/query?stream=true",
            statusPollMs: 5000,
        },
        window.__TINYRAG_CONFIG || {},
    );

    // --- DOM handles (resolved on DOMContentLoaded) ---
    let messagesEl = null;
    let composerEl = null;
    let queryEl = null;
    let sendBtnEl = null;
    let statusPillEl = null;
    let modelNameEl = null;
    let micBtnEl = null;
    let ttsToggleEl = null;
    let stopTtsBtnEl = null;
    let autoSendToggleEl = null;
    let engineBtnEl = null;
    let engineModalEl = null;
    let engineModalCloseEl = null;
    let engineSaveBtnEl = null;
    let modeLocalEl = null;
    let modeCloudEl = null;
    let cloudApiKeyEl = null;
    let cloudBaseUrlEl = null;
    let cloudModelEl = null;

    const ENGINE_STORAGE_KEY = "agrirag_engine_cfg";

    function getEngineConfig() {
        try {
            const raw = window.localStorage
                ? window.localStorage.getItem(ENGINE_STORAGE_KEY)
                : null;
            if (raw) {
                const parsed = JSON.parse(raw);
                return {
                    engine_mode: parsed.engine_mode === "cloud" ? "cloud" : "local",
                    api_key: String(parsed.api_key || "").trim(),
                    api_base_url:
                        String(parsed.api_base_url || "").trim() ||
                        "https://generativelanguage.googleapis.com/v1beta/openai",
                    api_model:
                        String(parsed.api_model || "").trim() ||
                        "gemini-2.5-flash",
                };
            }
        } catch (_) {
            /* ignore storage errors */
        }
        return {
            engine_mode: "local",
            api_key: "",
            api_base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
            api_model: "gemini-2.5-flash",
        };
    }

    function updateEngineButtonLabel() {
        if (!engineBtnEl) return;
        const cfg = getEngineConfig();
        const labelSpan = document.getElementById("engine-btn-label");
        if (cfg.engine_mode === "cloud" && cfg.api_key) {
            if (labelSpan) {
                labelSpan.textContent = `ক্লাউড (${cfg.api_model})`;
            } else {
                engineBtnEl.textContent = `ক্লাউড (${cfg.api_model})`;
            }
            engineBtnEl.classList.add("engine-cloud-active");
        } else {
            if (labelSpan) {
                labelSpan.textContent = "লোকাল ইঞ্জিন";
            } else {
                engineBtnEl.textContent = "লোকাল ইঞ্জিন";
            }
            engineBtnEl.classList.remove("engine-cloud-active");
        }
    }

    // ----------------------------------------------------------------
    // Status polling
    // ----------------------------------------------------------------

    /**
     * Fetch /api/status and update the topbar. Called on load and then
     * every STATUS_POLL_MS milliseconds. We don't clear the pill on
     * failure — a transient network blip shouldn't visually erase the
     * "ready" state. Errors are swallowed except for the very first
     * poll, where we at least surface the failure in the pill.
     */
    async function refreshStatus(isFirst) {
        try {
            const resp = await fetch(CFG.statusUrl, {
                headers: { Accept: "application/json" },
            });
            if (!resp.ok) {
                throw new Error(`status HTTP ${resp.status}`);
            }
            const data = await resp.json();

            // Pill text + class. The pill has 3 possible states
            // matching the .pill-ok / .pill-down / .pill-loading CSS:
            //   ready   -> "ready" (green)
            //   degraded-> "degraded" (red)
            //   loading -> "checking..." (grey) — only on first paint
            //
            // The /api/status response field is `ok` (boolean), not
            // `ready` — see StatusResponse in src/tinyrag/api/schemas.py.
            // Step 4.23 fixed a latent bug where we were reading
            // data.ready (undefined → pill stuck on "degraded").
            const ready = !!data.ok;
            setStatusPill(
                ready ? "ready" : "degraded",
                ready ? "ready" : "degraded",
            );
            if (data.model_name) {
                modelNameEl.textContent = data.model_name;
                modelNameEl.title = data.model_name;
            } else {
                modelNameEl.textContent = "";
                modelNameEl.title = "";
            }
        } catch (err) {
            if (isFirst) {
                setStatusPill("loading", "checking…");
            }
            // After the first attempt, keep the previous pill state
            // on transient failures. Log to console so devs can debug.
            // eslint-disable-next-line no-console
            console.warn("[tinyrag] status poll failed:", err);
        }
    }

    /**
     * Swap the pill's class + text. Pill class is one of
     *   pill-ok, pill-down, pill-loading
     * (see ui/static/style.css).
     */
    function setStatusPill(stateClass, text) {
        statusPillEl.classList.remove("pill-ok", "pill-down", "pill-loading");
        statusPillEl.classList.add(stateClass);
        statusPillEl.textContent = text;
    }

    // ----------------------------------------------------------------
    // Chat send + SSE stream consumption
    // ----------------------------------------------------------------

    /**
     * Composer submit handler. Prevents the default form post, builds
     * the AskRequest body, opens the SSE stream, and renders into a
     * new assistant bubble. Re-enables the composer on completion
     * (success or failure) so the user can ask the next question.
     */
    async function onSubmit(evt) {
        evt.preventDefault();
        const query = queryEl.value.trim();
        if (!query) {
            // The textarea has `required minlength="1"` so the browser
            // should block this — but be defensive in case JS is mid-
            // disable (e.g. browser autofill).
            queryEl.focus();
            return;
        }

        // Lock the composer so the user can't double-submit while a
        // stream is in flight. Enter inside the textarea still works
        // because we don't preventDefault on keydown — only on submit.
        setComposerBusy(true);

        // Auto-collapse starter suggestions drawer like Claude/Gemini
        // so the conversation gets maximum viewport height.
        const sugDrawer = document.getElementById("suggestions-drawer");
        const sugToggle = document.getElementById("suggestions-toggle-btn");
        if (sugDrawer) {
            sugDrawer.classList.add("is-collapsed");
        }
        if (sugToggle) {
            sugToggle.setAttribute("aria-expanded", "false");
        }
        const welcomeCard = document.getElementById("welcome-card");
        if (welcomeCard) {
            welcomeCard.classList.add("welcome-compact");
        }

        // 1. Echo the user's message into the chat history.
        appendUserMessage(query);

        // 2. Create the empty assistant bubble we will stream into.
        const assistantBubble = appendAssistantPlaceholder();

        // 3. Empty the textarea + restore focus so the user can type
        //    the next question without reaching for the mouse.
        queryEl.value = "";
        queryEl.focus();

        // 4. Open the SSE stream. Any failure (network, validation,
        //    HTTP 4xx/5xx) renders an error bubble and re-enables.
        try {
            await consumeStream(query, assistantBubble);
        } catch (err) {
            // eslint-disable-next-line no-console
            console.error("[tinyrag] stream error:", err);
            renderErrorBubble(
                assistantBubble,
                err && err.message ? err.message : String(err),
            );
        } finally {
            setComposerBusy(false);
        }
    }

    /**
     * POST the query, parse the text/event-stream response frame by
     * frame, and append tokens into the assistant bubble. Returns
     * when the server closes the stream OR an error frame arrives.
     */
    async function consumeStream(query, assistantBubble) {
        const engineCfg = getEngineConfig();
        const reqBody = { query: query };
        if (engineCfg.engine_mode === "cloud" && engineCfg.api_key) {
            reqBody.engine_mode = "cloud";
            reqBody.api_key = engineCfg.api_key;
            reqBody.api_base_url = engineCfg.api_base_url;
            reqBody.api_model = engineCfg.api_model;
        }
        const resp = await fetch(CFG.queryUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Accept: "text/event-stream",
            },
            body: JSON.stringify(reqBody),
        });

        if (!resp.ok) {
            // 4xx usually has a JSON error body from our exception
            // handlers; try to extract a useful message.
            let msg = `HTTP ${resp.status}`;
            try {
                const errBody = await resp.json();
                if (errBody && errBody.detail) {
                    msg = `${msg}: ${JSON.stringify(errBody.detail)}`;
                }
            } catch (_) {
                /* not JSON; fall back to the status code */
            }
            throw new Error(msg);
        }

        if (!resp.body) {
            throw new Error("response had no body (no ReadableStream)");
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        // SSE frames are separated by a blank line — the SSE spec
        // says "\n\n" but several servers (including sse-starlette,
        // which we use) emit "\r\n\r\n" because the spec allows CRLF
        // line endings anywhere. We accept BOTH so the parser works
        // regardless of which separator the producer picked. The
        // regex is intentionally non-capturing and uses `\r?\n`
        // twice so it matches "\n\n", "\r\n\r\n", "\r\n\n", and
        // "\n\r\n" — every legal SSE frame terminator.
        const FRAME_SEP = /\r?\n\r?\n/;

        while (true) {
            const { value, done } = await reader.read();
            if (done) {
                break;
            }
            buffer += decoder.decode(value, { stream: true });

            // Process every complete frame in the buffer. We loop
            // because a single read() can deliver multiple frames.
            let sepMatch;
            // exec() in a while-loop is intentional — splitting on
            // the first occurrence and re-searching lets us handle
            // any number of frames per read.
            while ((sepMatch = FRAME_SEP.exec(buffer)) !== null) {
                const rawFrame = buffer.slice(0, sepMatch.index);
                // Advance buffer past the separator. Note: we use
                // sepMatch[0].length (the actual matched bytes),
                // not a hard-coded 2, so we handle both "\n\n"
                // (length 2) and "\r\n\r\n" (length 4) correctly.
                buffer = buffer.slice(sepMatch.index + sepMatch[0].length);
                if (rawFrame.trim().length > 0) {
                    handleSseFrame(rawFrame, assistantBubble);
                }
            }
        }

        // Server closed the stream without a `done` frame. If we
        // accumulated some text, leave it; otherwise surface a soft
        // warning so the user knows something is off.
        if (assistantBubble.dataset.state === "streaming") {
            assistantBubble.dataset.state = "complete";
        }
    }

    /**
     * Parse a single SSE frame (no trailing \n\n — already split off)
     * and route to the right handler. We tolerate \r\n line endings
     * because some proxies normalise line endings.
     */
    function handleSseFrame(rawFrame, assistantBubble) {
        const lines = rawFrame.split(/\r?\n/);
        let eventName = "message"; // SSE default if no `event:` line
        const dataLines = [];
        for (const line of lines) {
            if (line.startsWith(":")) {
                // Comment line — ignore (SSE spec).
                continue;
            }
            if (line.startsWith("event:")) {
                eventName = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
                // The first space after `data:` is part of the
                // protocol (the spec allows "data: foo" OR
                // "data:foo"). We strip exactly one leading space.
                dataLines.push(line.slice(5).replace(/^ /, ""));
            }
        }
        const dataStr = dataLines.join("\n");
        if (dataStr.length === 0) {
            return; // No payload — ignore.
        }

        let payload;
        try {
            payload = JSON.parse(dataStr);
        } catch (err) {
            // eslint-disable-next-line no-console
            console.warn("[tinyrag] could not parse SSE data as JSON:", dataStr, err);
            return;
        }

        if (eventName === "status") {
            // Pipeline-stage update emitted before the LLM prefill.
            // Turn the structured payload into a human status line so
            // the thinking indicator reflects real progress (retrieval
            // → generation) rather than a blank wait.
            updateThinkingStatus(assistantBubble, formatStatus(payload));
        } else if (eventName === "token") {
            if (typeof payload.delta === "string") {
                // First token ends the "thinking" phase — drop the
                // indicator, then append. clearThinkingIndicator is
                // idempotent so calling it every token is harmless,
                // but we gate on the placeholder still being present.
                if (assistantBubble.querySelector(".thinking")) {
                    clearThinkingIndicator(assistantBubble);
                }
                appendTokenToBubble(assistantBubble, payload.delta);
            }
        } else if (eventName === "done") {
            // Clear the indicator in case the stream produced zero
            // tokens (e.g. an empty answer) so we never leave pulsing
            // dots behind. payload.answer is the full Answer.to_dict()
            // dict from src/tinyrag/core/answer.py.
            clearThinkingIndicator(assistantBubble);
            renderDonePayload(assistantBubble, payload.answer || payload);
        } else if (eventName === "error") {
            clearThinkingIndicator(assistantBubble);
            const msg = payload.error || payload.detail || "unknown error";
            renderErrorBubble(assistantBubble, msg);
        } else {
            // Unknown event — log + ignore. Forward-compat: future
            // server events won't break the chat.
            // eslint-disable-next-line no-console
            console.info("[tinyrag] unknown SSE event:", eventName, payload);
        }
    }

    // ----------------------------------------------------------------
    // Bubble rendering
    // ----------------------------------------------------------------

    /**
     * Append a right-aligned user bubble. Plain text (we don't trust
     * user input to be safe HTML).
     */
    function appendUserMessage(text) {
        const wrap = document.createElement("div");
        wrap.className = "message message-user";
        const bubble = document.createElement("div");
        bubble.className = "bubble";
        bubble.textContent = text; // textContent -> no HTML injection
        wrap.appendChild(bubble);
        messagesEl.appendChild(wrap);
        scrollToBottom();
    }

    /**
     * Append a left-aligned assistant bubble that the streamer will
     * fill in. Starts with an animated "thinking" indicator (a status
     * line + pulsing dots) so the ~15-25s LLM prefill doesn't look
     * frozen. The indicator is replaced by the first streamed token
     * (see clearThinkingIndicator) and any status frame updates its
     * text (see updateThinkingStatus). We remember dataset.state so
     * the SSE handler knows whether the stream is still open.
     */
    function appendAssistantPlaceholder() {
        const wrap = document.createElement("div");
        wrap.className = "message message-assistant";
        const bubble = document.createElement("div");
        bubble.className = "bubble assistant-text";
        bubble.dataset.state = "streaming";

        // Thinking indicator: a status line + three pulsing dots. Held
        // in one container so clearThinkingIndicator can remove it
        // wholesale when the first token lands.
        const thinking = document.createElement("div");
        thinking.className = "thinking";
        const statusLine = document.createElement("span");
        statusLine.className = "thinking-status";
        statusLine.textContent = "Searching your documents";
        const dots = document.createElement("span");
        dots.className = "thinking-dots";
        dots.innerHTML = "<span></span><span></span><span></span>";
        thinking.appendChild(statusLine);
        thinking.appendChild(dots);
        bubble.appendChild(thinking);

        wrap.appendChild(bubble);
        messagesEl.appendChild(wrap);
        scrollToBottom();
        return bubble;
    }

    /**
     * Update the thinking indicator's status line from a `status` SSE
     * frame. No-op if the indicator was already cleared (e.g. a token
     * raced ahead). Keeps the pulsing dots; only swaps the text.
     */
    function updateThinkingStatus(bubble, text) {
        const statusLine = bubble.querySelector(".thinking .thinking-status");
        if (statusLine) {
            statusLine.textContent = text;
            scrollToBottom();
        }
    }

    /**
     * Remove the thinking indicator. Called exactly once, when the
     * first token arrives — from then on the bubble holds streamed
     * text. Idempotent: safe to call when no indicator is present.
     */
    function clearThinkingIndicator(bubble) {
        const thinking = bubble.querySelector(".thinking");
        if (thinking) {
            thinking.remove();
        }
    }

    /**
     * Append one token of streamed text to an assistant bubble.
     * textContent assignment per token is fine for the volumes we
     * see (≤ a few hundred tokens per answer). If profiling shows
     * Turn a `status` SSE payload into a human-readable status line.
     * The backend sends {stage, n_chunks, n_sources, sources,
     * used_sensor}. We describe what the pipeline found and that it's
     * now generating — real progress, not a fake spinner. Defensive
     * about missing fields so a future payload shape can't throw.
     */
    function formatStatus(payload) {
        const n = Number(payload && payload.n_sources) || 0;
        if (n <= 0) {
            // Retriever found nothing above threshold — the LLM will
            // say it can't answer. Still show we're working.
            return "Generating answer";
        }
        const noun = n === 1 ? "source" : "sources";
        const kind = payload.used_sensor ? "document & sensor " : "";
        return `Found ${n} ${kind}${noun} · generating answer`;
    }

    /**
     * Append a streamed token to the bubble. textContent += keeps
     * HTML inert (no injection from model output); if this ever causes
     * jank we can switch to a text node + appendChild deltas.
     */
    function appendTokenToBubble(bubble, delta) {
        bubble.textContent += delta;
        scrollToBottom();
    }

    /**
     * Append inline text with **bold** segments converted to <strong> nodes
     * using .textContent exclusively (100% XSS-safe).
     */
    function appendFormattedInline(parent, lineText) {
        const parts = String(lineText || "").split(/(\*\*[^*]+\*\*)/g);
        for (const part of parts) {
            if (!part) continue;
            if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
                const strong = document.createElement("strong");
                strong.textContent = part.slice(2, -2);
                parent.appendChild(strong);
            } else {
                parent.appendChild(document.createTextNode(part));
            }
        }
    }

    /**
     * Render structured Markdown (headings, bullet lists, bold text, paragraphs)
     * into safe DOM elements inside the assistant bubble.
     */
    function renderFormattedMarkdown(bubble, text) {
        const lines = String(text || "").split(/\r?\n/);
        let currentList = null;

        for (const rawLine of lines) {
            const line = rawLine.trim();
            if (!line) {
                currentList = null;
                continue;
            }
            // Heading line: ### ...
            const headingMatch = line.match(/^#{1,4}\s+(.*)$/);
            if (headingMatch) {
                currentList = null;
                const h = document.createElement("h4");
                h.className = "md-heading";
                appendFormattedInline(h, headingMatch[1]);
                bubble.appendChild(h);
                continue;
            }
            // Bullet list item: * ... or - ... or • ...
            const bulletMatch = line.match(/^[\*\-•]\s+(.*)$/);
            if (bulletMatch) {
                if (!currentList) {
                    currentList = document.createElement("ul");
                    currentList.className = "md-list";
                    bubble.appendChild(currentList);
                }
                const li = document.createElement("li");
                li.className = "md-list-item";
                appendFormattedInline(li, bulletMatch[1]);
                currentList.appendChild(li);
                continue;
            }
            currentList = null;
            const p = document.createElement("p");
            p.className = "md-para";
            appendFormattedInline(p, line);
            bubble.appendChild(p);
        }

        if (bubble.children.length === 0) {
            const p = document.createElement("p");
            p.textContent = text;
            bubble.appendChild(p);
        }
    }

    function createSpeakerSvg() {
        const svgNS = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(svgNS, "svg");
        svg.setAttribute("viewBox", "0 0 24 24");
        svg.setAttribute("width", "14");
        svg.setAttribute("height", "14");
        svg.setAttribute("fill", "none");
        svg.setAttribute("stroke", "currentColor");
        svg.setAttribute("stroke-width", "2");
        svg.setAttribute("stroke-linecap", "round");
        svg.setAttribute("stroke-linejoin", "round");
        const poly = document.createElementNS(svgNS, "polygon");
        poly.setAttribute("points", "11 5 6 9 2 9 2 15 6 15 11 19 11 5");
        const path = document.createElementNS(svgNS, "path");
        path.setAttribute("d", "M15.54 8.46a5 5 0 0 1 0 7.07");
        svg.appendChild(poly);
        svg.appendChild(path);
        return svg;
    }

    function createStopSvg() {
        const svgNS = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(svgNS, "svg");
        svg.setAttribute("viewBox", "0 0 24 24");
        svg.setAttribute("width", "13");
        svg.setAttribute("height", "13");
        svg.setAttribute("fill", "currentColor");
        const rect = document.createElementNS(svgNS, "rect");
        rect.setAttribute("x", "6");
        rect.setAttribute("y", "6");
        rect.setAttribute("width", "12");
        rect.setAttribute("height", "12");
        rect.setAttribute("rx", "2");
        svg.appendChild(rect);
        return svg;
    }

    /**
     * On the terminal `done` event, render the citation cards and
     * diagnostics footer beneath the answer text. The answer dict
     * is whatever Answer.to_dict() returned server-side.
     */
    function renderDonePayload(bubble, answer) {
        bubble.dataset.state = "complete";

        // Preserve clean streamed text or take server answer text
        const streamed = bubble.textContent ? bubble.textContent.trim() : "";
        let text = typeof answer.text === "string" ? answer.text.trim() : "";
        if (!text && streamed) {
            text = streamed;
        }
        bubble.textContent = "";
        renderFormattedMarkdown(bubble, text);

        // Audio replay & stop buttons + collapsible citations bar
        if (text.trim()) {
            const audioBar = document.createElement("div");
            audioBar.className = "bubble-audio-bar";

            const replayBtn = document.createElement("button");
            replayBtn.type = "button";
            replayBtn.className = "btn-replay-audio";
            replayBtn.appendChild(createSpeakerSvg());
            const replayLbl = document.createElement("span");
            replayLbl.textContent = "বাংলায় শুনুন";
            replayBtn.appendChild(replayLbl);
            replayBtn.title = "কৃষি পরামর্শ মুখে শুনুন";
            replayBtn.addEventListener("click", () => speakText(text, true));
            audioBar.appendChild(replayBtn);

            const stopBtn = document.createElement("button");
            stopBtn.type = "button";
            stopBtn.className = "btn-stop-audio";
            stopBtn.appendChild(createStopSvg());
            const stopLbl = document.createElement("span");
            stopLbl.textContent = "থামান";
            stopBtn.appendChild(stopLbl);
            stopBtn.title = "চলমান ভয়েস পড়া থামান";
            stopBtn.addEventListener("click", () => stopSpeechOutput());
            audioBar.appendChild(stopBtn);

            bubble.appendChild(audioBar);

            // Automatically speak if TTS toggle is active
            speakText(text, false);
        }

        // Source cards inside a clean collapsible <details> accordion
        const citations = Array.isArray(answer.citations)
            ? answer.citations
            : [];
        if (citations.length > 0) {
            const sourcesWrap = document.createElement("details");
            sourcesWrap.className = "sources";
            const header = document.createElement("summary");
            header.className = "sources-header";
            header.textContent = `উৎস নথিপত্র (${citations.length})`;
            sourcesWrap.appendChild(header);
            const listWrap = document.createElement("div");
            listWrap.className = "sources-list";
            for (const cit of citations) {
                listWrap.appendChild(buildSourceCard(cit));
            }
            sourcesWrap.appendChild(listWrap);
            bubble.appendChild(sourcesWrap);
        }

        // Diagnostics footer — durations + token counts.
        bubble.appendChild(buildDiagnostics(answer));

        scrollToBottom();
    }

    /**
     * Clean raw response markdown text for natural Bengali speech synthesis.
     */
    function cleanTextForSpeech(raw) {
        if (!raw) return "";
        return raw
            .replace(/```[\s\S]*?```/g, "")
            .replace(/`([^`]+)`/g, "$1")
            .replace(/\[([^\]]+)\]\([^\x29]+\)/g, "$1")
            .replace(/[*_#~]/g, "")
            .replace(/---+|\u2500+/g, " ")
            .replace(/\[\s*[\d০-৯]+(?:\s*,\s*[\d০-৯]+)*\s*\]/g, "")
            .replace(/\[উৎস:[^\]]+\]/g, "")
            .replace(/উৎস:\s*[^।\n]+/g, "")
            .replace(/\s+/g, " ")
            .trim();
    }

    let activeAudioPlayer = null;
    let activeAudioUrl = null;

    /**
     * Stop any active neural MP3 audio or window.speechSynthesis playback immediately.
     */
    function stopSpeechOutput() {
        if (activeAudioPlayer) {
            try {
                activeAudioPlayer.pause();
                activeAudioPlayer.currentTime = 0;
            } catch (_) {
                /* ignore */
            }
            activeAudioPlayer = null;
        }
        if (activeAudioUrl) {
            try {
                URL.revokeObjectURL(activeAudioUrl);
            } catch (_) {
                /* ignore */
            }
            activeAudioUrl = null;
        }
        if ("speechSynthesis" in window) {
            window.speechSynthesis.cancel();
        }
    }

    /**
     * Fallback browser Web Speech synthesis when offline or /api/tts is unavailable.
     */
    function speakWithBrowserSynthesis(clean) {
        if (!("speechSynthesis" in window)) {
            return;
        }
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(clean);
        utterance.lang = "bn-BD";
        utterance.rate = 0.95;
        utterance.pitch = 1.0;

        const voices = window.speechSynthesis.getVoices();
        const bnVoice = voices.find(
            (v) =>
                v.lang.startsWith("bn") ||
                v.lang.toLowerCase().includes("bangla") ||
                v.name.toLowerCase().includes("bengali") ||
                v.name.toLowerCase().includes("bangla")
        );
        if (bnVoice) {
            utterance.voice = bnVoice;
        }
        window.speechSynthesis.speak(utterance);
    }

    /**
     * Speak Bengali text aloud using backend Neural Bangladeshi Bengali TTS (/api/tts)
     * so PC browsers (Linux/Windows/Mac) and Mobile browsers both play clear Bengali speech,
     * falling back to window.speechSynthesis only if offline.
     */
    async function speakText(rawText, forceManual) {
        if (!forceManual && ttsToggleEl && !ttsToggleEl.checked) {
            return;
        }
        stopSpeechOutput();

        const clean = cleanTextForSpeech(rawText);
        if (!clean) return;

        try {
            const resp = await fetch("/api/tts", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: clean,
                    voice: "bn-BD-NabanitaNeural",
                }),
            });
            if (!resp.ok) {
                throw new Error(`TTS HTTP ${resp.status}`);
            }
            const blob = await resp.blob();
            stopSpeechOutput();
            activeAudioUrl = URL.createObjectURL(blob);
            activeAudioPlayer = new Audio(activeAudioUrl);
            activeAudioPlayer.onended = () => {
                stopSpeechOutput();
            };
            await activeAudioPlayer.play();
        } catch (err) {
            // eslint-disable-next-line no-console
            console.warn("[AgriRAG] Backend neural TTS fallback to WebSpeech:", err);
            speakWithBrowserSynthesis(clean);
        }
    }

    /**
     * Build one .source-card div from a single citation dict. The
     * dict shape comes from Citation.to_dict() in core/answer.py.
     */
    function buildSourceCard(cit) {
        const card = document.createElement("div");
        card.className = "source-card";

        const ref = document.createElement("span");
        ref.className = "source-ref";
        ref.textContent = `[${cit.number}]`;
        card.appendChild(ref);

        if (cit.location) {
            const loc = document.createElement("span");
            loc.className = "source-loc";
            loc.textContent = String(cit.location);
            card.appendChild(loc);
        }

        if (typeof cit.score === "number") {
            const score = document.createElement("span");
            score.className = "source-score";
            score.textContent = `score=${cit.score.toFixed(3)}`;
            card.appendChild(score);
        }

        if (cit.preview) {
            const preview = document.createElement("div");
            preview.className = "source-preview";
            preview.textContent = String(cit.preview);
            card.appendChild(preview);
        }
        return card;
    }

    /**
     * Build the small grey diagnostics footer (durations + tokens).
     * We always render it so the user can see the server actually
     * responded, even if there are zero citations.
     */
    function buildDiagnostics(answer) {
        const diag = document.createElement("div");
        diag.className = "diagnostics";

        const durTotal = numOrNull(answer.duration_total_ms);
        const durRetr = numOrNull(answer.duration_retrieve_ms);
        const durLlm = numOrNull(answer.duration_llm_ms);
        const tokTotal = numOrNull(answer.total_tokens);
        const modelName = answer.model_name;

        if (durTotal !== null) {
            addDiagSpan(diag, `total ${fmtMs(durTotal)}`);
        }
        if (durRetr !== null) {
            addDiagSpan(diag, `retrieve ${fmtMs(durRetr)}`);
        }
        if (durLlm !== null) {
            addDiagSpan(diag, `llm ${fmtMs(durLlm)}`);
        }
        if (tokTotal !== null) {
            addDiagSpan(diag, `${tokTotal} tok`);
        }
        if (modelName) {
            addDiagSpan(diag, String(modelName));
        }
        return diag;
    }

    function addDiagSpan(parent, text) {
        const span = document.createElement("span");
        span.textContent = text;
        parent.appendChild(span);
    }

    function numOrNull(v) {
        return typeof v === "number" && Number.isFinite(v) ? v : null;
    }

    function fmtMs(n) {
        // 0–999 ms shows "123 ms"; >= 1000 ms shows "1.23 s".
        if (n < 1000) {
            return `${n.toFixed(0)} ms`;
        }
        return `${(n / 1000).toFixed(2)} s`;
    }

    /**
     * Convert an in-flight assistant bubble into an error bubble.
     * We replace the bubble's parent so the error gets the red
     * styling from .message-error .bubble.
     */
    function renderErrorBubble(bubble, message) {
        const wrap = bubble.parentElement;
        if (!wrap) return;
        wrap.classList.remove("message-assistant");
        wrap.classList.add("message-error");
        bubble.textContent = `error: ${message}`;
        bubble.dataset.state = "error";
        scrollToBottom();
    }

    // ----------------------------------------------------------------
    // Composer state
    // ----------------------------------------------------------------

    /**
     * Disable the textarea + send button while a stream is in flight.
     * Visual cue comes from CSS (opacity 0.5 + cursor: not-allowed).
     */
    function setComposerBusy(busy) {
        queryEl.disabled = busy;
        sendBtnEl.disabled = busy;
        if (!busy) {
            queryEl.focus();
        }
    }

    /**
     * Smoothly scroll the chat area to the bottom. Called after
     * every append so the user sees the latest token without manual
     * scrolling. Cheap because the chat area is bounded (~900px).
     */
    function scrollToBottom() {
        // Defer to the next frame so the DOM has the new height.
        requestAnimationFrame(() => {
            messagesEl.parentElement.scrollTop =
                messagesEl.parentElement.scrollHeight;
        });
    }

    // ----------------------------------------------------------------
    // Voice-to-Voice: Speech-to-Text (STT) via Web Speech API
    // ----------------------------------------------------------------

    let recognition = null;
    let isRecognizing = false;
    let finalTranscript = "";
    let silenceTimer = null;

    function clearSilenceTimer() {
        if (silenceTimer) {
            clearTimeout(silenceTimer);
            silenceTimer = null;
        }
    }

    function scheduleAutoSendOnSilence() {
        clearSilenceTimer();
        if (!autoSendToggleEl || !autoSendToggleEl.checked) {
            return;
        }
        silenceTimer = setTimeout(() => {
            silenceTimer = null;
            if (!queryEl || !composerEl || queryEl.disabled) {
                return;
            }
            const spoken = queryEl.value.trim();
            if (spoken.length >= 3) {
                if (isRecognizing && recognition) {
                    try {
                        recognition.stop();
                    } catch (_) {
                        /* ignore */
                    }
                }
                stopSpeechRecognition();
                composerEl.requestSubmit();
            }
        }, 2200);
    }

    function initSpeechRecognition() {
        const SpeechRecognition =
            window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            if (micBtnEl) {
                micBtnEl.title = "আপনার ব্রাউজারে ভয়েস রিকগনিশন সাপোর্ট নেই (Chrome/Edge ব্যবহার করুন)";
                micBtnEl.style.opacity = "0.5";
            }
            return;
        }

        try {
            recognition = new SpeechRecognition();
            recognition.lang = "bn-BD"; // Bengali (Bangladesh)
            recognition.continuous = true;
            recognition.interimResults = true;

            recognition.onstart = function () {
                isRecognizing = true;
                clearSilenceTimer();
                finalTranscript = queryEl.value ? queryEl.value.trim() : "";
                if (micBtnEl) {
                    micBtnEl.classList.add("recording");
                    micBtnEl.title = "কথা বলা শেষ হলে আবার মাইক বাটনে চাপুন";
                }
                if (queryEl) {
                    queryEl.placeholder = "শুনছি... বাংলায় আপনার ফসলের সমস্যার কথা বলুন (থেমে গেলে স্বয়ংক্রিয়ভাবে সেন্ড হবে)...";
                }
            };

            recognition.onresult = function (event) {
                // Rebuild utterance from event.results[0..N-1] and deduplicate
                // cumulative prefixes emitted by Android Chrome's SpeechRecognizer.
                let sessionTranscript = "";
                for (let i = 0; i < event.results.length; ++i) {
                    const seg = (event.results[i][0].transcript || "").trim();
                    if (!seg) continue;
                    if (!sessionTranscript) {
                        sessionTranscript = seg;
                    } else if (seg.startsWith(sessionTranscript)) {
                        // Android Chrome progressive cumulative hypothesis
                        sessionTranscript = seg;
                    } else if (sessionTranscript.endsWith(seg)) {
                        // Exact duplicate segment
                        continue;
                    } else {
                        sessionTranscript = `${sessionTranscript} ${seg}`;
                    }
                }
                // Collapse any accidental adjacent repeated 1-3 word phrases
                sessionTranscript = sessionTranscript
                    .replace(/\b(\S+)(?:\s+\1\b)+/g, "$1")
                    .replace(/\s+/g, " ")
                    .trim();
                if (queryEl) {
                    queryEl.value = finalTranscript
                        ? `${finalTranscript} ${sessionTranscript}`.trim()
                        : sessionTranscript;
                }
                scheduleAutoSendOnSilence();
            };

            recognition.onerror = function (event) {
                // eslint-disable-next-line no-console
                console.warn("[AgriRAG] Speech recognition error:", event.error);
                clearSilenceTimer();
                stopSpeechRecognition();
            };

            recognition.onend = function () {
                stopSpeechRecognition();
            };

            if (micBtnEl) {
                micBtnEl.addEventListener("click", toggleSpeechRecognition);
            }
        } catch (err) {
            // eslint-disable-next-line no-console
            console.warn("[AgriRAG] Speech recognition init failed:", err);
        }
    }

    function toggleSpeechRecognition() {
        if (!recognition) return;
        if (isRecognizing) {
            clearSilenceTimer();
            recognition.stop();
        } else {
            try {
                stopSpeechOutput();
                recognition.start();
            } catch (err) {
                // eslint-disable-next-line no-console
                console.warn("[AgriRAG] recognition start failed:", err);
            }
        }
    }

    function stopSpeechRecognition() {
        isRecognizing = false;
        if (micBtnEl) {
            micBtnEl.classList.remove("recording");
            micBtnEl.title = "ভয়েস ইনপুট (বাংলায় কথা বলুন)";
        }
        if (queryEl) {
            queryEl.placeholder = "ফসলের রোগ বা সমস্যা সম্পর্কে বলুন বা লিখুন… (যেমন: ধানের পাতা পোড়া রোগে কী স্প্রে করব?)";
        }
    }

    // ----------------------------------------------------------------
    // Hybrid Engine Modal & Category Tabs
    // ----------------------------------------------------------------

    function openEngineModal() {
        if (!engineModalEl) return;
        const cfg = getEngineConfig();
        if (modeLocalEl && modeCloudEl) {
            modeLocalEl.checked = cfg.engine_mode !== "cloud";
            modeCloudEl.checked = cfg.engine_mode === "cloud";
        }
        if (cloudApiKeyEl) cloudApiKeyEl.value = cfg.api_key;
        if (cloudBaseUrlEl) cloudBaseUrlEl.value = cfg.api_base_url;
        if (cloudModelEl) cloudModelEl.value = cfg.api_model;
        engineModalEl.classList.remove("modal-hidden");
    }

    function closeEngineModal() {
        if (!engineModalEl) return;
        engineModalEl.classList.add("modal-hidden");
    }

    function saveEngineSettings() {
        const mode =
            modeCloudEl && modeCloudEl.checked ? "cloud" : "local";
        const cfg = {
            engine_mode: mode,
            api_key: cloudApiKeyEl ? cloudApiKeyEl.value.trim() : "",
            api_base_url: cloudBaseUrlEl
                ? cloudBaseUrlEl.value.trim()
                : "https://generativelanguage.googleapis.com/v1beta/openai",
            api_model: cloudModelEl
                ? cloudModelEl.value.trim()
                : "gemini-2.5-flash",
        };
        try {
            if (window.localStorage) {
                window.localStorage.setItem(
                    ENGINE_STORAGE_KEY,
                    JSON.stringify(cfg),
                );
            }
        } catch (_) {
            /* ignore */
        }
        updateEngineButtonLabel();
        closeEngineModal();
    }

    // ----------------------------------------------------------------
    // Wire-up (runs once on DOMContentLoaded)
    // ----------------------------------------------------------------

    function init() {
        messagesEl = document.getElementById("messages");
        composerEl = document.getElementById("composer");
        queryEl = document.getElementById("query");
        sendBtnEl = document.getElementById("send");
        statusPillEl = document.getElementById("status-pill");
        modelNameEl = document.getElementById("model-name");
        micBtnEl = document.getElementById("mic-btn");
        ttsToggleEl = document.getElementById("tts-toggle");
        stopTtsBtnEl = document.getElementById("stop-tts-btn");
        autoSendToggleEl = document.getElementById("autosend-toggle");
        engineBtnEl = document.getElementById("engine-btn");
        engineModalEl = document.getElementById("engine-modal");
        engineModalCloseEl = document.getElementById("engine-modal-close");
        engineSaveBtnEl = document.getElementById("engine-save-btn");
        modeLocalEl = document.getElementById("mode-local");
        modeCloudEl = document.getElementById("mode-cloud");
        cloudApiKeyEl = document.getElementById("cloud-api-key");
        cloudBaseUrlEl = document.getElementById("cloud-base-url");
        cloudModelEl = document.getElementById("cloud-model");

        if (
            !messagesEl ||
            !composerEl ||
            !queryEl ||
            !sendBtnEl ||
            !statusPillEl ||
            !modelNameEl
        ) {
            // eslint-disable-next-line no-console
            console.error("[tinyrag] DOM init failed — required elements missing");
            return;
        }

        composerEl.addEventListener("submit", onSubmit);

        // Enter-to-send in textarea (Shift+Enter inserts a newline)
        queryEl.addEventListener("keydown", function (e) {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                composerEl.requestSubmit();
            }
        });

        if (stopTtsBtnEl) {
            stopTtsBtnEl.addEventListener("click", stopSpeechOutput);
        }

        // Collapse the 10-cell status grid by default on ALL screens (PC & mobile)
        // so technical metrics never clutter the farmer's main chat view.
        // Clicking the topbar status pill ("READY") toggles the diagnostics panel.
        const spPanel = document.getElementById("status-panel");
        const spToggle = document.getElementById("status-toggle");
        const ensureStatusCollapsedOnStart = () => {
            if (spPanel && spToggle) {
                spPanel.classList.add("is-collapsed");
                spToggle.setAttribute("aria-expanded", "false");
                spToggle.textContent = "Show";
                try {
                    window.localStorage.setItem("tinyrag.statusPanelCollapsed", "1");
                } catch (_err) {
                    // ignore storage errors
                }
            }
        };
        ensureStatusCollapsedOnStart();
        setTimeout(ensureStatusCollapsedOnStart, 0);
        if (statusPillEl && spToggle) {
            statusPillEl.style.cursor = "pointer";
            statusPillEl.addEventListener("click", () => {
                spToggle.click();
            });
        }

        // Wire up the collapsible starter suggestions drawer toggle
        const sugDrawer = document.getElementById("suggestions-drawer");
        const sugToggle = document.getElementById("suggestions-toggle-btn");
        if (sugDrawer && sugToggle) {
            sugToggle.addEventListener("click", () => {
                const collapsed = sugDrawer.classList.toggle("is-collapsed");
                sugToggle.setAttribute("aria-expanded", String(!collapsed));
            });
        }

        // Hybrid Engine Modal wire-up
        updateEngineButtonLabel();
        if (engineBtnEl) {
            engineBtnEl.addEventListener("click", openEngineModal);
        }
        if (engineModalCloseEl) {
            engineModalCloseEl.addEventListener("click", closeEngineModal);
        }
        if (engineSaveBtnEl) {
            engineSaveBtnEl.addEventListener("click", saveEngineSettings);
        }
        const presetBtns = document.querySelectorAll(".preset-btn");
        presetBtns.forEach((btn) => {
            btn.addEventListener("click", () => {
                if (modeCloudEl) modeCloudEl.checked = true;
                if (cloudBaseUrlEl && btn.dataset.url) {
                    cloudBaseUrlEl.value = btn.dataset.url;
                }
                if (cloudModelEl && btn.dataset.model) {
                    cloudModelEl.value = btn.dataset.model;
                }
            });
        });

        // Initialize Web Speech STT
        initSpeechRecognition();

        // Wire up category tabs for quick suggestion chips
        const catTabs = document.querySelectorAll(".chip-categories .cat-tab");
        const chips = document.querySelectorAll(".quick-chips .chip");
        catTabs.forEach((tab) => {
            tab.addEventListener("click", () => {
                catTabs.forEach((t) => t.classList.remove("active"));
                tab.classList.add("active");
                const selectedCat = tab.getAttribute("data-cat") || "all";
                chips.forEach((chip) => {
                    const chipCat = chip.getAttribute("data-cat") || "all";
                    if (selectedCat === "all" || chipCat === selectedCat) {
                        chip.style.display = "inline-flex";
                    } else {
                        chip.style.display = "none";
                    }
                });
            });
        });

        // Wire up quick suggestion chips (Exhibition demo mode)
        chips.forEach((chip) => {
            chip.addEventListener("click", () => {
                const q = chip.getAttribute("data-query");
                if (q && queryEl) {
                    queryEl.value = q;
                    queryEl.focus();
                    composerEl.requestSubmit();
                }
            });
        });

        // Kick off status polling. The first call also acts as our
        // "page is alive" check — if /api/status 5xx's, the pill
        // stays in its "checking..." initial state.
        refreshStatus(true);
        setInterval(() => refreshStatus(false), CFG.statusPollMs);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        // Script loaded after DOMContentLoaded (shouldn't happen
        // because we use `defer`, but be defensive).
        init();
    }
})();
