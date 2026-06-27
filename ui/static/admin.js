/* TinyRAG — admin / documents client (Step 4.22).
 *
 * Vanilla ES2017+ JavaScript, no framework, no build step. Loaded via
 * <script src="/static/admin.js" defer> in ui/templates/admin.html;
 * served at /static/admin.js by FastAPI's StaticFiles mount in main.py.
 *
 * Responsibilities:
 *   1. Poll /api/status every STATUS_POLL_MS to keep the topbar pill
 *      + active-model label fresh (mirrors chat.js so the two pages
 *      look consistent).
 *   2. Fetch GET /api/documents on load and render the response into
 *      a documents table (filename, doc_type, size, chunks,
 *      ingested_at, delete button).
 *   3. Intercept the upload form's submit, POST the selected file
 *      to /api/documents via FormData, show a success/error toast,
 *      and refresh the list on success.
 *   4. Wire per-row Delete buttons via event delegation. Each click
 *      triggers a window.confirm() guard before DELETE /api/documents/{id}.
 *   5. Surface non-2xx responses in a fixed-position toast that
 *      auto-hides after a few seconds (red for error, green for
 *      success).
 *
 * Why fetch() + FormData, not a third-party uploader?
 * --------------------------------------------------
 * The upload is small (≤50 MB, single-shot, no resumability needed).
 * FormData + fetch() is the native API, has no dependencies, and
 * matches the same wire format the rest of the codebase speaks.
 *
 * Why textContent everywhere?
 * ---------------------------
 * Document filenames + doc_type + error messages are user-controlled.
 * Every dynamic string is assigned via .textContent so a malicious
 * filename like "<img src=x onerror=alert(1)>" can't inject HTML.
 * The codebase's invariant (since Step 4.21): no innerHTML on
 * dynamic strings.
 */

(function () {
    "use strict";

    // --- Config knobs (all overridable from window.__TINYRAG_CONFIG ---
    // useful for the test suite to inject a fake host).
    const CFG = Object.assign(
        {
            statusUrl: "/api/status",
            listUrl: "/api/documents",
            uploadUrl: "/api/documents",
            statusPollMs: 5000,
            toastMs: 5000,
        },
        window.__TINYRAG_CONFIG || {},
    );

    // --- DOM handles (resolved on DOMContentLoaded) ---
    let statusPillEl = null;
    let modelNameEl = null;
    let uploadFormEl = null;
    let fileInputEl = null;
    let docTypeEl = null;
    let uploadButtonEl = null;
    let refreshButtonEl = null;
    let docsCountEl = null;
    let docsTableContainerEl = null;
    let emptyStateEl = null;
    let toastEl = null;

    // ----------------------------------------------------------------
    // Status polling (mirrors chat.js)
    // ----------------------------------------------------------------

    async function refreshStatus(isFirst) {
        try {
            const resp = await fetch(CFG.statusUrl, {
                headers: { Accept: "application/json" },
            });
            if (!resp.ok) {
                throw new Error(`status HTTP ${resp.status}`);
            }
            const data = await resp.json();
            const ready = !!data.ready;
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
            // eslint-disable-next-line no-console
            console.warn("[tinyrag] status poll failed:", err);
        }
    }

    function setStatusPill(stateClass, text) {
        statusPillEl.classList.remove("pill-ok", "pill-down", "pill-loading");
        statusPillEl.classList.add(stateClass);
        statusPillEl.textContent = text;
    }

    // ----------------------------------------------------------------
    // JSON fetch wrapper — throws on non-2xx with a useful message
    // ----------------------------------------------------------------

    /**
     * Wrapper around fetch() that:
     *  - sets Accept: application/json
     *  - throws an Error with a useful `message` on non-2xx
     *    (parses {error, detail} if present)
     *  - returns the parsed JSON body on 2xx
     */
    async function fetchJSON(url, init) {
        const opts = Object.assign(
            { headers: { Accept: "application/json" } },
            init || {},
        );
        // Caller-provided headers should win over the default Accept.
        opts.headers = Object.assign({}, init && init.headers, opts.headers);
        const resp = await fetch(url, opts);
        const text = await resp.text();
        let body = null;
        if (text) {
            try {
                body = JSON.parse(text);
            } catch (_) {
                body = null;
            }
        }
        if (!resp.ok) {
            const detail =
                body && (body.detail || body.error)
                    ? `${body.error || ""}${
                          body.detail ? `: ${body.detail}` : ""
                      }`
                    : text || `HTTP ${resp.status}`;
            throw new Error(`HTTP ${resp.status}: ${detail}`);
        }
        return body;
    }

    // ----------------------------------------------------------------
    // Toast
    // ----------------------------------------------------------------

    let toastTimer = null;

    /**
     * Show a toast in the bottom-right corner. kind is one of
     *   "success" | "error" | "info"
     * which maps to the .toast-{kind} CSS class. Auto-hides after
     * CFG.toastMs ms unless another toast supersedes it.
     */
    function showToast(message, kind) {
        if (!toastEl) return;
        if (toastTimer) {
            clearTimeout(toastTimer);
            toastTimer = null;
        }
        const cls = kind && ["success", "error", "info"].indexOf(kind) !== -1
            ? kind
            : "info";
        toastEl.classList.remove(
            "toast-hidden",
            "toast-success",
            "toast-error",
            "toast-info",
        );
        toastEl.classList.add(`toast-${cls}`);
        toastEl.textContent = message;
        toastTimer = setTimeout(() => {
            toastEl.classList.add("toast-hidden");
            toastTimer = null;
        }, CFG.toastMs);
    }

    // ----------------------------------------------------------------
    // Documents list + table rendering
    // ----------------------------------------------------------------

    async function refreshList() {
        try {
            const data = await fetchJSON(CFG.listUrl);
            const docs = Array.isArray(data.documents) ? data.documents : [];
            const total = typeof data.count === "number" ? data.count : docs.length;
            if (docsCountEl) {
                docsCountEl.textContent = `(${total})`;
            }
            if (docs.length === 0) {
                renderEmptyState();
            } else {
                renderTable(docs);
            }
        } catch (err) {
            // eslint-disable-next-line no-console
            console.warn("[tinyrag] list fetch failed:", err);
            showToast(`Failed to list documents: ${err.message}`, "error");
            if (docsCountEl) {
                docsCountEl.textContent = "(?)";
            }
            // Keep the table container in its current state.
        }
    }

    function renderEmptyState() {
        if (!docsTableContainerEl) return;
        docsTableContainerEl.textContent = "";
        const p = document.createElement("p");
        p.className = "empty-state";
        p.id = "empty-state";
        p.textContent = "No documents yet. Upload one above.";
        docsTableContainerEl.appendChild(p);
        emptyStateEl = p;
    }

    function renderTable(documents) {
        if (!docsTableContainerEl) return;
        docsTableContainerEl.textContent = "";
        emptyStateEl = null;

        const table = document.createElement("table");
        table.className = "docs-table";

        // Header
        const thead = document.createElement("thead");
        const trh = document.createElement("tr");
        for (const col of ["Filename", "Type", "Size", "Chunks", "Ingested", ""]) {
            const th = document.createElement("th");
            th.textContent = col;
            trh.appendChild(th);
        }
        thead.appendChild(trh);
        table.appendChild(thead);

        // Body
        const tbody = document.createElement("tbody");
        for (const doc of documents) {
            tbody.appendChild(buildRow(doc));
        }
        table.appendChild(tbody);
        docsTableContainerEl.appendChild(table);
    }

    function buildRow(doc) {
        const tr = document.createElement("tr");
        tr.className = "doc-row";

        const tdFilename = document.createElement("td");
        tdFilename.className = "col-filename";
        tdFilename.textContent = doc.filename || "(unnamed)";
        tr.appendChild(tdFilename);

        const tdType = document.createElement("td");
        tdType.className = "col-type";
        tdType.textContent = doc.doc_type || "?";
        tr.appendChild(tdType);

        const tdSize = document.createElement("td");
        tdSize.className = "col-size";
        tdSize.textContent = formatBytes(doc.size_bytes);
        tr.appendChild(tdSize);

        const tdChunks = document.createElement("td");
        tdChunks.className = "col-chunks";
        tdChunks.textContent = String(
            typeof doc.num_chunks === "number" ? doc.num_chunks : 0,
        );
        tr.appendChild(tdChunks);

        const tdTime = document.createElement("td");
        tdTime.className = "col-time";
        tdTime.textContent = formatDate(doc.ingested_at);
        tdTime.title = doc.ingested_at || "";
        tr.appendChild(tdTime);

        const tdAction = document.createElement("td");
        tdAction.className = "col-action";
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "delete-button";
        btn.textContent = "Delete";
        btn.dataset.documentId = doc.id || "";
        tdAction.appendChild(btn);
        tr.appendChild(tdAction);

        return tr;
    }

    function formatBytes(n) {
        const v = typeof n === "number" && Number.isFinite(n) ? n : 0;
        if (v < 1024) return `${v} B`;
        if (v < 1024 * 1024) return `${(v / 1024).toFixed(1)} KB`;
        if (v < 1024 * 1024 * 1024) {
            return `${(v / (1024 * 1024)).toFixed(1)} MB`;
        }
        return `${(v / (1024 * 1024 * 1024)).toFixed(2)} GB`;
    }

    function formatDate(iso) {
        if (!iso || typeof iso !== "string") return "";
        // Truncate "2026-06-28T14:32:00+00:00" -> "2026-06-28 14:32".
        // Cheap + locale-stable — no Intl.DateTimeFormat needed.
        const m = iso.match(
            /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})/,
        );
        if (!m) return iso;
        return `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}`;
    }

    // ----------------------------------------------------------------
    // Upload + delete handlers
    // ----------------------------------------------------------------

    async function onUploadSubmit(evt) {
        evt.preventDefault();
        if (!fileInputEl || !fileInputEl.files || fileInputEl.files.length === 0) {
            showToast("Choose a file first.", "error");
            return;
        }
        const file = fileInputEl.files[0];
        const docType = docTypeEl ? docTypeEl.value : "manual";

        const formData = new FormData();
        formData.append("file", file);
        formData.append("doc_type", docType);

        setUploadBusy(true);
        try {
            const report = await fetchJSON(CFG.uploadUrl, {
                method: "POST",
                body: formData,
            });
            if (report && report.ok) {
                const n =
                    typeof report.num_chunks === "number" ? report.num_chunks : 0;
                showToast(`Ingested ${file.name} (${n} chunks)`, "success");
                // Reset file input so the same file can't be re-uploaded
                // by accident on a follow-up click.
                fileInputEl.value = "";
            } else {
                const errMsg =
                    (report && (report.error || report.detail)) ||
                    "unknown ingest failure";
                showToast(`Ingest failed: ${errMsg}`, "error");
            }
        } catch (err) {
            // eslint-disable-next-line no-console
            console.warn("[tinyrag] upload failed:", err);
            showToast(err.message || "upload failed", "error");
        } finally {
            setUploadBusy(false);
            await refreshList();
        }
    }

    async function onDeleteClick(documentId) {
        if (!documentId) return;
        // eslint-disable-next-line no-alert
        const ok = window.confirm(
            `Delete document ${documentId}?\n\n` +
                "This removes its chunks from the metadata store " +
                "and its vectors from FAISS. The original file on disk " +
                "(if any) is left in place.",
        );
        if (!ok) return;
        try {
            const url = `${CFG.uploadUrl}/${encodeURIComponent(documentId)}`;
            const report = await fetchJSON(url, { method: "DELETE" });
            const chunks =
                report && typeof report.chunks_removed === "number"
                    ? report.chunks_removed
                    : 0;
            const vecs =
                report && typeof report.vectors_removed === "number"
                    ? report.vectors_removed
                    : 0;
            showToast(
                `Deleted document ${documentId} (${chunks} chunks / ${vecs} vectors)`,
                "success",
            );
        } catch (err) {
            // eslint-disable-next-line no-console
            console.warn("[tinyrag] delete failed:", err);
            showToast(err.message || "delete failed", "error");
        } finally {
            await refreshList();
        }
    }

    function setUploadBusy(busy) {
        if (fileInputEl) fileInputEl.disabled = busy;
        if (docTypeEl) docTypeEl.disabled = busy;
        if (uploadButtonEl) uploadButtonEl.disabled = busy;
    }

    // ----------------------------------------------------------------
    // Wire-up (runs once on DOMContentLoaded)
    // ----------------------------------------------------------------

    function init() {
        statusPillEl = document.getElementById("status-pill");
        modelNameEl = document.getElementById("model-name");
        uploadFormEl = document.getElementById("upload-form");
        fileInputEl = document.getElementById("file-input");
        docTypeEl = document.getElementById("doc-type");
        uploadButtonEl = document.getElementById("upload-button");
        refreshButtonEl = document.getElementById("refresh-button");
        docsCountEl = document.getElementById("docs-count");
        docsTableContainerEl = document.getElementById("docs-table-container");
        emptyStateEl = document.getElementById("empty-state");
        toastEl = document.getElementById("toast");

        if (
            !statusPillEl ||
            !uploadFormEl ||
            !fileInputEl ||
            !docsTableContainerEl ||
            !toastEl
        ) {
            // eslint-disable-next-line no-console
            console.error(
                "[tinyrag] admin DOM init failed — required elements missing",
            );
            return;
        }

        uploadFormEl.addEventListener("submit", onUploadSubmit);
        if (refreshButtonEl) {
            refreshButtonEl.addEventListener("click", refreshList);
        }
        // Event delegation: one click listener for every delete
        // button (current + future rows).
        docsTableContainerEl.addEventListener("click", (evt) => {
            const target = evt.target;
            if (!target || typeof target.closest !== "function") return;
            const btn = target.closest(".delete-button");
            if (!btn) return;
            const id = btn.dataset ? btn.dataset.documentId : "";
            onDeleteClick(id);
        });

        // Kick off initial loads.
        refreshStatus(true);
        setInterval(() => refreshStatus(false), CFG.statusPollMs);
        refreshList();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        // Script loaded after DOMContentLoaded (shouldn't happen
        // because we use `defer`, but be defensive).
        init();
    }
})();