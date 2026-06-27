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
        const resp = await fetch(CFG.queryUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Accept: "text/event-stream",
            },
            body: JSON.stringify({ query: query }),
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

        // SSE frames are separated by a blank line, i.e. \n\n. We
        // accumulate bytes into `buffer` and split on that boundary
        // once we've seen enough. Each frame then has lines of the
        // form `event: <name>` and `data: <json>`.
        while (true) {
            const { value, done } = await reader.read();
            if (done) {
                break;
            }
            buffer += decoder.decode(value, { stream: true });

            // Process every complete frame in the buffer. We loop
            // because a single read() can deliver multiple frames.
            let sepIdx;
            // indexOf in a while-loop is intentional — splitting on
            // the first occurrence and re-searching lets us handle
            // any number of frames per read.
            // eslint-disable-next-line no-cond-assign
            while ((sepIdx = buffer.indexOf("\n\n")) !== -1) {
                const rawFrame = buffer.slice(0, sepIdx);
                buffer = buffer.slice(sepIdx + 2);
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

        if (eventName === "token") {
            if (typeof payload.delta === "string") {
                appendTokenToBubble(assistantBubble, payload.delta);
            }
        } else if (eventName === "done") {
            // payload.answer is the full Answer.to_dict() dict from
            // src/tinyrag/core/answer.py.
            renderDonePayload(assistantBubble, payload.answer || payload);
        } else if (eventName === "error") {
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
     * Append a left-aligned empty assistant bubble that the streamer
     * will fill in. We remember the dataset.state so the SSE handler
     * can know whether the stream is still open.
     */
    function appendAssistantPlaceholder() {
        const wrap = document.createElement("div");
        wrap.className = "message message-assistant";
        const bubble = document.createElement("div");
        bubble.className = "bubble assistant-text";
        bubble.dataset.state = "streaming";
        bubble.textContent = ""; // empty until first token arrives
        wrap.appendChild(bubble);
        messagesEl.appendChild(wrap);
        scrollToBottom();
        return bubble;
    }

    /**
     * Append one token of streamed text to an assistant bubble.
     * textContent assignment per token is fine for the volumes we
     * see (≤ a few hundred tokens per answer). If profiling shows
     * jank we can switch to a text node + appendChild deltas.
     */
    function appendTokenToBubble(bubble, delta) {
        bubble.textContent += delta;
        scrollToBottom();
    }

    /**
     * On the terminal `done` event, render the citation cards and
     * diagnostics footer beneath the answer text. The answer dict
     * is whatever Answer.to_dict() returned server-side.
     */
    function renderDonePayload(bubble, answer) {
        bubble.dataset.state = "complete";

        // Wrap text in a <p> so the CSS .bubble p margins apply.
        const text = typeof answer.text === "string" ? answer.text : "";
        bubble.textContent = "";
        const p = document.createElement("p");
        p.textContent = text;
        bubble.appendChild(p);

        // Source cards — only render if we actually retrieved
        // something. Empty arrays are common for a "I don't know"
        // answer; rendering an empty Sources block would just be
        // noise.
        const citations = Array.isArray(answer.citations)
            ? answer.citations
            : [];
        if (citations.length > 0) {
            const sourcesWrap = document.createElement("div");
            sourcesWrap.className = "sources";
            const header = document.createElement("div");
            header.className = "sources-header";
            header.textContent = `Sources (${citations.length})`;
            sourcesWrap.appendChild(header);
            for (const cit of citations) {
                sourcesWrap.appendChild(buildSourceCard(cit));
            }
            bubble.appendChild(sourcesWrap);
        }

        // Diagnostics footer — durations + token counts. We pull
        // ms numbers out and format them as "X.X ms" or "X tok".
        bubble.appendChild(buildDiagnostics(answer));

        scrollToBottom();
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
    // Wire-up (runs once on DOMContentLoaded)
    // ----------------------------------------------------------------

    function init() {
        messagesEl = document.getElementById("messages");
        composerEl = document.getElementById("composer");
        queryEl = document.getElementById("query");
        sendBtnEl = document.getElementById("send");
        statusPillEl = document.getElementById("status-pill");
        modelNameEl = document.getElementById("model-name");

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

        // Enter-to-send is already handled by the form's implicit
        // submit on Enter inside a textarea (Shift+Enter inserts a
        // newline). No extra keydown handler needed.

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
