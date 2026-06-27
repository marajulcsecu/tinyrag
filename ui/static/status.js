/* TinyRAG — system status panel client (Step 4.23).
 *
 * Vanilla ES2017+ JavaScript, no framework, no build step. Loaded via
 * <script src="/static/status.js" defer> in BOTH ui/templates/index.html
 * and ui/templates/admin.html; served at /static/status.js by FastAPI's
 * StaticFiles mount in main.py.
 *
 * Responsibilities:
 *   1. Poll /api/status every STATUS_POLL_MS (5 s default) and render
 *      the response into the 10 .status-value cells of the
 *      <section id="status-panel"> block (model, embeddings, vector
 *      dims, doc chunks, sensor chunks, LLM server, sensor source,
 *      RAM, deployment, last-update timestamp).
 *   2. Toggle the panel's collapsed/expanded state via a button click
 *      + persist that state in localStorage so a user who collapses
 *      it on / sees it stay collapsed when they navigate to /admin.
 *   3. Reflect the overall subsystem health (data.ok) on the panel's
 *      left border via the [data-ok="true|false"] attribute selector,
 *      matching the topbar pill colour scheme.
 *   4. Surface failures inside the panel (the "Last update" cell
 *      shows the error message and the border flips red) without
 *      spamming the console.
 *
 * Why a separate JS file from chat.js / admin.js?
 * -----------------------------------------------
 * chat.js (chat page) and admin.js (documents page) each already have
 * their own concerns (SSE streaming, upload, list/delete). Adding the
 * status panel to either one would couple unrelated subsystems. A
 * dedicated status.js keeps the polling logic + DOM-mutation logic for
 * the panel in one place and lets both pages share the same panel
 * implementation without forking either file.
 *
 * Why textContent everywhere?
 * ---------------------------
 * Model names, embedding model strings, sensor sources, and the LLM
 * health label are all user-influenced (the deployment sets them,
 * the running LLM determines llama_cpp_status, etc.). Every dynamic
 * string is assigned via .textContent so a malicious value like
 * "<img src=x onerror=alert(1)>" can't inject HTML. The codebase's
 * invariant (since Step 4.21): no innerHTML on dynamic strings.
 *
 * Why localStorage?
 * -----------------
 * A user who collapses the panel on / probably wants it collapsed
 * on /admin too. localStorage gives us a tiny cross-page flag
 * ("tinyrag.statusPanelCollapsed") without needing server-side
 * state. We tolerate localStorage being unavailable (Safari private
 * mode, file://) by wrapping the read/write in try/catch — the
 * panel still works, it just won't persist the choice.
 *
 * Last-write-wins semantics: if the user collapses on /, navigates
 * to /admin, expands, then collapses on /admin, the "collapsed"
 * flag wins (the last write). This is intentional — both pages
 * share the same panel state by definition.
 */

(function () {
    "use strict";

    // --- Config knobs (all overridable from window.__TINYRAG_CONFIG ---
    // useful for the test suite to inject a fake host).
    const CFG = Object.assign(
        {
            statusUrl: "/api/status",
            statusPollMs: 5000,
            storageKey: "tinyrag.statusPanelCollapsed",
        },
        window.__TINYRAG_CONFIG || {},
    );

    // --- DOM handles (resolved on DOMContentLoaded) ---
    let panelEl = null;
    let toggleEl = null;
    let bodyEl = null;

    // 10 value spans, one per StatusResponse field the panel surfaces.
    let modelEl = null;
    let embeddingEl = null;
    let dimEl = null;
    let docCountEl = null;
    let sensorCountEl = null;
    let llmEl = null;
    let sensorSourceEl = null;
    let ramEl = null;
    let deployEl = null;
    let updatedEl = null;

    // ----------------------------------------------------------------
    // Storage helpers — localStorage with try/catch + sensible defaults
    // ----------------------------------------------------------------

    function readCollapsedPref() {
        try {
            return window.localStorage.getItem(CFG.storageKey) === "1";
        } catch (_) {
            // localStorage unavailable (private mode, sandboxed iframe).
            return false;
        }
    }

    function writeCollapsedPref(collapsed) {
        try {
            window.localStorage.setItem(
                CFG.storageKey,
                collapsed ? "1" : "0",
            );
        } catch (_) {
            /* ignore — non-critical */
        }
    }

    // ----------------------------------------------------------------
    // Polling + rendering
    // ----------------------------------------------------------------

    /**
     * Fetch /api/status and update the panel cells. Called on load
     * and then every CFG.statusPollMs milliseconds. Failures don't
     * clear the last-good values — they flip the panel border red
     * and surface the error in the "Last update" cell so the user
     * sees something is wrong without us throwing.
     */
    async function refreshStatus() {
        try {
            const resp = await fetch(CFG.statusUrl, {
                headers: { Accept: "application/json" },
            });
            if (!resp.ok) {
                throw new Error(`status HTTP ${resp.status}`);
            }
            const data = await resp.json();
            renderStatus(data);
        } catch (err) {
            renderError(err);
            // eslint-disable-next-line no-console
            console.warn("[tinyrag] status panel poll failed:", err);
        }
    }

    /**
     * Update every cell from a fresh /api/status payload. Uses
     * textContent exclusively — see the file-header note.
     */
    function renderStatus(data) {
        if (!panelEl || !modelEl) {
            // init() hasn't run yet; nothing to render into.
            return;
        }

        // --- text cells ---
        modelEl.textContent = data.model_name || "unknown";
        embeddingEl.textContent = data.embedding_model || "unknown";
        sensorSourceEl.textContent = data.sensor_source || "unknown";
        deployEl.textContent = data.deployment_target || "unknown";

        // llama_cpp_status also drives the colour class so the user
        // can see at a glance whether the LLM is reachable.
        const llmStatus =
            typeof data.llama_cpp_status === "string" && data.llama_cpp_status
                ? data.llama_cpp_status
                : "unknown";
        llmEl.textContent = llmStatus;
        llmEl.className =
            "status-value status-llm-" + llmStatus.toLowerCase().replace(/[^a-z0-9]+/g, "-");

        // --- numeric cells (defensive: defaults to a sensible placeholder) ---
        dimEl.textContent = String(
            typeof data.embedding_dim === "number" ? data.embedding_dim : "—",
        );
        docCountEl.textContent = String(
            typeof data.doc_chunk_count === "number" ? data.doc_chunk_count : 0,
        );
        sensorCountEl.textContent = String(
            typeof data.sensor_chunk_count === "number"
                ? data.sensor_chunk_count
                : 0,
        );
        ramEl.textContent =
            typeof data.ram_mb === "number" && Number.isFinite(data.ram_mb)
                ? `${data.ram_mb.toFixed(1)} MB`
                : "—";

        updatedEl.textContent = new Date().toLocaleTimeString();

        // --- overall health flag drives the left border colour ---
        panelEl.dataset.ok = data.ok ? "true" : "false";
    }

    /**
     * Render a failure state without blanking the last-good values.
     * The left border flips red and "Last update" carries the message
     * so a debugging user sees something is wrong immediately.
     */
    function renderError(err) {
        if (!panelEl || !updatedEl) return;
        const msg = err && err.message ? err.message : String(err);
        updatedEl.textContent = `error: ${msg}`;
        panelEl.dataset.ok = "false";
    }

    // ----------------------------------------------------------------
    // Collapse toggle
    // ----------------------------------------------------------------

    /**
     * Flip the .is-collapsed class on the panel + flip the button's
     * aria-expanded attribute + swap the button text. Persist the
     * choice to localStorage so it survives page navigation.
     */
    function togglePanel() {
        if (!panelEl || !toggleEl) return;
        const collapsed = panelEl.classList.toggle("is-collapsed");
        toggleEl.setAttribute("aria-expanded", String(!collapsed));
        toggleEl.textContent = collapsed ? "Show" : "Hide";
        writeCollapsedPref(collapsed);
    }

    /**
     * Apply the persisted collapsed state on init. Called once at
     * boot — subsequent toggles update storage but don't read it
     * again until the next page load.
     */
    function restoreCollapsedState() {
        if (!panelEl || !toggleEl) return;
        if (readCollapsedPref()) {
            // Use the togglePanel() entry point so the button text +
            // aria-expanded stay in sync with the .is-collapsed class.
            togglePanel();
        }
    }

    // ----------------------------------------------------------------
    // Wire-up (runs once on DOMContentLoaded)
    // ----------------------------------------------------------------

    function init() {
        panelEl = document.getElementById("status-panel");
        toggleEl = document.getElementById("status-toggle");
        bodyEl = document.getElementById("status-body");

        modelEl = document.getElementById("status-model");
        embeddingEl = document.getElementById("status-embedding");
        dimEl = document.getElementById("status-dim");
        docCountEl = document.getElementById("status-doc-count");
        sensorCountEl = document.getElementById("status-sensor-count");
        llmEl = document.getElementById("status-llm");
        sensorSourceEl = document.getElementById("status-sensor-source");
        ramEl = document.getElementById("status-ram");
        deployEl = document.getElementById("status-deploy");
        updatedEl = document.getElementById("status-updated");

        // Required: panel + toggle + at least one value cell. If the
        // template is broken (e.g. partial install), bail loudly so a
        // dev can see it in the console instead of staring at empty
        // cells wondering why nothing's updating.
        if (
            !panelEl ||
            !toggleEl ||
            !bodyEl ||
            !modelEl ||
            !embeddingEl ||
            !dimEl ||
            !docCountEl ||
            !sensorCountEl ||
            !llmEl ||
            !sensorSourceEl ||
            !ramEl ||
            !deployEl ||
            !updatedEl
        ) {
            // eslint-disable-next-line no-console
            console.error(
                "[tinyrag] status panel DOM init failed — required elements missing",
            );
            return;
        }

        toggleEl.addEventListener("click", togglePanel);
        restoreCollapsedState();

        // Kick off initial load + start the polling loop. Errors are
        // handled inside refreshStatus so they don't bubble up to
        // window.onerror.
        refreshStatus();
        setInterval(refreshStatus, CFG.statusPollMs);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        // Script loaded after DOMContentLoaded (shouldn't happen
        // because we use `defer`, but be defensive).
        init();
    }
})();
