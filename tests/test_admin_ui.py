"""Tests for the admin / documents web UI (Step 4.22 — second web UI page).

What this module covers
------------------------
- ``GET /admin`` renders the Jinja2 ``admin.html`` page (the dashboard
  for uploading + listing + deleting documents).
- The rendered HTML carries the structural elements admin.js needs
  (upload form, file input, doc-type select, docs table container,
  toast, status pill, model-name span, script tag).
- The Jinja context fills in the validation constants from
  :mod:`tinyrag.api.routes_docs` (``ALLOWED_EXTENSIONS``,
  ``MAX_UPLOAD_BYTES``, ``SUPPORTED_DOC_TYPES``, ``DEFAULT_DOC_TYPE``)
  so the upload form's <select> stays in sync with the server.
- ``/static/admin.js`` is served with the right MIME type, has a
  balanced structure, references ``/api/documents``, uses
  ``FormData`` + ``fetch``, and uses ``textContent`` (no
  ``innerHTML`` for XSS safety).
- style.css contains the admin classes (``.panel``, ``.docs-table``,
  ``.upload-form``, ``.delete-button``, ``.toast``, etc.).
- The full upload → list → delete flow works end-to-end against the
  Step 4.18 ``/api/documents`` endpoints — including the 413/400/404
  error paths.

Why a separate file (not folded into ``test_web_ui.py``)?
---------------------------------------------------------
Step 4.22 ships a second page with its own JS file, template, and
test surface. Keeping it in a named file makes the diff against
``main`` legible (one-step = one-test-file).

Hermetic?
---------
Yes. The :class:`TestClient` triggers the lifespan but uses
``llm_kind="fake"`` + ``embedder_kind="fake"`` — no live llama-
server, no model downloads. The admin.js is read from the project's
``ui/`` tree (committed in this step), so there's no runtime-only
path to mock.

Location: ``tests/test_admin_ui.py``
"""

# ``I001`` (import sort) suppressed — see the same noqa block in
# tests/test_api.py for the rationale.
# ruff: noqa: I001
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap (mirrors tests/test_web_ui.py)
# ---------------------------------------------------------------------------
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient  # noqa: E402

from tinyrag.main import create_app  # noqa: E402
from tinyrag.api.routes_docs import (  # noqa: E402
    ALLOWED_EXTENSIONS,
    DEFAULT_DOC_TYPE,
    MAX_UPLOAD_BYTES,
    SUPPORTED_DOC_TYPES,
)


# ---------------------------------------------------------------------------
# Paths to the admin UI assets.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = PROJECT_ROOT / "ui"
TEMPLATES_DIR = UI_DIR / "templates"
STATIC_DIR = UI_DIR / "static"
ADMIN_HTML_PATH = TEMPLATES_DIR / "admin.html"
ADMIN_JS_PATH = STATIC_DIR / "admin.js"
STYLE_CSS_PATH = STATIC_DIR / "style.css"


# ---------------------------------------------------------------------------
# Minimal Settings builder (no FAISS indices needed — the admin page
# doesn't query the index, only /api/documents does).
# ---------------------------------------------------------------------------


def _make_settings(tmp_path: Path) -> Any:
    """Build a minimal :class:`Settings` pointing every path at ``tmp_path``."""
    from tinyrag.config import PathsSettings, RetrievalSettings, Settings

    return Settings(
        deployment={"target": "laptop"},
        server={"host": "127.0.0.1", "port": 8000},
        llm={
            "model_path": "models/phi-3-mini.gguf",
            "server_url": "http://127.0.0.1:8080",
            "context_size": 4096,
            "temperature": 0.0,
            "max_tokens": 512,
            "gpu_layers": 0,
        },
        embedding={
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 32,
            "cache_dir": str(tmp_path / "_hf_cache"),
        },
        chunking={
            "chunk_size": 400,
            "chunk_overlap": 50,
            "encoding": "cl100k_base",
        },
        retrieval=RetrievalSettings(
            doc_index_path=str(tmp_path / "doc.faiss"),
            sensor_index_path=str(tmp_path / "sensor.faiss"),
            doc_top_k=3,
            sensor_top_k=2,
            similarity_threshold=0.0,
            index_type="faiss",
        ),
        sensors={
            "source": "simulated",
            "csv_path": str(tmp_path / "synthetic.csv"),
            "dht_pin": 4,
            "pir_pin": 17,
            "mqtt_broker": "localhost",
            "mqtt_port": 1883,
            "mqtt_topic_prefix": "tinyrag/sensors/",
        },
        logging={"level": "INFO", "json_format": True, "path": str(tmp_path / "app.log")},
        paths=PathsSettings(
            documents_dir=str(tmp_path / "documents"),
            metadata_db=str(tmp_path / "metadata.db"),
            sensor_logs_dir=str(tmp_path / "sensor_logs"),
            logs_dir=str(tmp_path / "logs"),
        ),
    )


@pytest.fixture
def client(tmp_path: Path) -> Any:
    """TestClient with the admin UI wired (Fake LLM + Fake embedder)."""
    settings = _make_settings(tmp_path)
    app = create_app(settings, llm_kind="fake", embedder_kind="fake")
    with TestClient(app) as c:
        yield c


# ===========================================================================
# Class 1 — UI files exist on disk
# ===========================================================================


class TestUIAdminAssetsOnDisk:
    """Sanity-check the admin UI assets actually exist on disk.

    A future refactor that moves the admin page will fail these
    tests at collection time — a clear signal that the test file
    needs to be pointed at the new location.
    """

    def test_admin_html_exists(self) -> None:
        assert ADMIN_HTML_PATH.is_file(), f"missing: {ADMIN_HTML_PATH}"

    def test_admin_js_exists(self) -> None:
        assert ADMIN_JS_PATH.is_file(), f"missing: {ADMIN_JS_PATH}"

    def test_style_css_exists(self) -> None:
        assert STYLE_CSS_PATH.is_file(), f"missing: {STYLE_CSS_PATH}"

    def test_ui_dir_layout(self) -> None:
        # The FastAPI mount expects ui/static/ + ui/templates/.
        assert TEMPLATES_DIR.is_dir(), f"missing dir: {TEMPLATES_DIR}"
        assert STATIC_DIR.is_dir(), f"missing dir: {STATIC_DIR}"


# ===========================================================================
# Class 2 — GET /admin (page render)
# ===========================================================================


class TestGetAdmin:
    """``GET /admin`` renders the documents page (HTML 200) with the
    structural elements admin.js expects to find."""

    def test_admin_returns_200(self, client: TestClient) -> None:
        resp = client.get("/admin")
        assert resp.status_code == 200

    def test_admin_returns_html(self, client: TestClient) -> None:
        resp = client.get("/admin")
        ct = resp.headers.get("content-type", "")
        assert ct.startswith("text/html"), f"unexpected content-type: {ct!r}"

    def test_admin_has_doctype(self, client: TestClient) -> None:
        resp = client.get("/admin")
        assert "<!DOCTYPE html>" in resp.text or "<!doctype html>" in resp.text.lower()

    def test_admin_has_upload_form(self, client: TestClient) -> None:
        body = client.get("/admin").text
        assert 'id="upload-form"' in body
        assert 'id="file-input"' in body
        assert 'id="doc-type"' in body
        assert 'id="upload-button"' in body

    def test_admin_has_docs_table_container(self, client: TestClient) -> None:
        body = client.get("/admin").text
        assert 'id="docs-table-container"' in body
        assert 'id="docs-count"' in body
        assert 'id="refresh-button"' in body
        assert 'id="empty-state"' in body

    def test_admin_has_status_pill_and_model(self, client: TestClient) -> None:
        body = client.get("/admin").text
        assert 'id="status-pill"' in body
        assert 'id="model-name"' in body

    def test_admin_has_toast(self, client: TestClient) -> None:
        body = client.get("/admin").text
        assert 'id="toast"' in body
        assert "toast-hidden" in body  # starts hidden

    def test_admin_has_brand_and_footer(self, client: TestClient) -> None:
        body = client.get("/admin").text
        assert "TinyRAG" in body
        assert "Admin" in body  # brand-suffix
        assert "footer" in body.lower()

    def test_admin_file_input_accepts_pdf_txt_md(self, client: TestClient) -> None:
        body = client.get("/admin").text
        # The browser-side hint. Server-side enforcement is in
        # ALLOWED_EXTENSIONS.
        assert 'accept=".pdf,.txt,.md"' in body

    def test_admin_doc_type_select_has_supported_types(self, client: TestClient) -> None:
        body = client.get("/admin").text
        for dt in sorted(SUPPORTED_DOC_TYPES):
            assert f'value="{dt}"' in body, f"missing <option value={dt!r}>"

    def test_admin_default_doc_type_is_selected(self, client: TestClient) -> None:
        body = client.get("/admin").text
        assert f'value="{DEFAULT_DOC_TYPE}"' in body
        assert "selected" in body

    def test_admin_loads_admin_js(self, client: TestClient) -> None:
        body = client.get("/admin").text
        assert 'src="/static/admin.js"' in body

    def test_admin_links_back_to_chat(self, client: TestClient) -> None:
        body = client.get("/admin").text
        assert 'href="/"' in body  # back-to-chat navlink

    def test_admin_links_to_api_docs(self, client: TestClient) -> None:
        body = client.get("/admin").text
        assert 'href="/docs"' in body


# ===========================================================================
# Class 3 — Chat page has an Admin navlink (regression pin for the index.html edit)
# ===========================================================================


class TestGetRootHasAdminLink:
    """``GET /`` (the chat page) now exposes an Admin navlink so users
    can navigate to /admin without typing the URL."""

    def test_root_links_to_admin(self, client: TestClient) -> None:
        body = client.get("/").text
        assert 'href="/admin"' in body
        assert "Admin" in body  # link text

    def test_root_uses_navlink_class(self, client: TestClient) -> None:
        body = client.get("/").text
        assert "navlink" in body


# ===========================================================================
# Class 4 — Jinja context fills in validation constants
# ===========================================================================


class TestGetAdminJinjaContext:
    """The admin page renders the validation constants from
    ``routes_docs.py`` so the upload form stays in sync with the
    server. Single source of truth: the route handler imports the
    constants and passes them to the template."""

    def test_help_text_lists_allowed_extensions(self, client: TestClient) -> None:
        body = client.get("/admin").text
        for ext in sorted(ALLOWED_EXTENSIONS):
            assert ext in body, f"help text missing extension: {ext}"

    def test_help_text_shows_max_size(self, client: TestClient) -> None:
        body = client.get("/admin").text
        expected_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        assert f"{expected_mb} MB" in body

    def test_doc_type_select_options_match_supported(self, client: TestClient) -> None:
        # Re-check the <option> tags with the actual sorted list
        # rendered server-side. The text appears both in
        # value="..." and between the tags (admin.html uses
        # {{ dt }} so the textContent == the value).
        body = client.get("/admin").text
        for dt in sorted(SUPPORTED_DOC_TYPES):
            # Both `value="dt"` and `>dt</option>` should appear.
            assert f'value="{dt}"' in body
            assert f">{dt}</option>" in body


# ===========================================================================
# Class 5 — /static/admin.js + /static/style.css
# ===========================================================================


class TestStaticAdminAsset:
    """The new admin.js is served with the right content type."""

    def test_admin_js_serves_200(self, client: TestClient) -> None:
        resp = client.get("/static/admin.js")
        assert resp.status_code == 200

    def test_admin_js_content_type(self, client: TestClient) -> None:
        ct = client.get("/static/admin.js").headers.get("content-type", "")
        assert "javascript" in ct, f"unexpected content-type: {ct!r}"

    def test_admin_js_content_matches_disk(self, client: TestClient) -> None:
        resp = client.get("/static/admin.js")
        disk = ADMIN_JS_PATH.read_bytes()
        assert resp.content == disk

    def test_style_css_serves_200(self, client: TestClient) -> None:
        resp = client.get("/static/style.css")
        assert resp.status_code == 200

    def test_style_css_content_matches_disk(self, client: TestClient) -> None:
        # Regression pin: the new admin CSS is part of the same
        # file. Disk and HTTP must agree byte-for-byte.
        resp = client.get("/static/style.css")
        disk = STYLE_CSS_PATH.read_bytes()
        assert resp.content == disk


# ===========================================================================
# Class 6 — admin.js structural integrity
# ===========================================================================


class TestAdminJS:
    """Static checks on the admin.js source.

    We don't execute the JS (no headless browser in the suite) —
    we just verify the script is well-formed and points at the
    documented endpoints so the page actually talks to the live API.
    """

    def _admin_js_text(self) -> str:
        return ADMIN_JS_PATH.read_text(encoding="utf-8")

    def test_admin_js_uses_iife(self) -> None:
        js = self._admin_js_text()
        assert "(function ()" in js, "admin.js missing IIFE wrapper"
        assert "})();" in js, "admin.js IIFE not invoked"

    def test_admin_js_references_api_documents(self) -> None:
        js = self._admin_js_text()
        assert "/api/documents" in js

    def test_admin_js_uses_fetch(self) -> None:
        js = self._admin_js_text()
        assert "fetch(" in js

    def test_admin_js_uses_formdata(self) -> None:
        js = self._admin_js_text()
        assert "FormData" in js

    def test_admin_js_uses_confirm_for_delete(self) -> None:
        js = self._admin_js_text()
        assert "confirm(" in js

    def test_admin_js_uses_textcontent(self) -> None:
        js = self._admin_js_text()
        # XSS safety: dynamic strings go through .textContent, not
        # .innerHTML. We expect multiple textContent assignments.
        assert ".textContent =" in js or ".textContent=" in js

    def test_admin_js_references_delete_button_class(self) -> None:
        js = self._admin_js_text()
        assert "delete-button" in js

    def test_admin_js_references_docs_table_class(self) -> None:
        js = self._admin_js_text()
        assert "docs-table" in js

    def test_admin_js_references_upload_form_id(self) -> None:
        js = self._admin_js_text()
        assert "upload-form" in js

    def test_admin_js_references_toast_class(self) -> None:
        js = self._admin_js_text()
        assert "toast" in js

    def test_admin_js_calls_api_status(self) -> None:
        js = self._admin_js_text()
        # The status poll must hit /api/status — same endpoint the
        # chat page uses so the topbar looks consistent.
        assert "/api/status" in js

    def test_admin_js_has_brace_balance(self) -> None:
        js = self._admin_js_text()
        assert js.count("{") == js.count("}"), (
            f"brace imbalance: {js.count('{')} open vs {js.count('}')} close"
        )
        assert js.count("(") == js.count(")"), (
            f"paren imbalance: {js.count('(')} open vs {js.count(')')} close"
        )


# ===========================================================================
# Class 7 — style.css structural integrity (admin classes)
# ===========================================================================


class TestAdminStyleCSS:
    """Static checks on the admin-specific CSS hooks."""

    def _css_text(self) -> str:
        return STYLE_CSS_PATH.read_text(encoding="utf-8")

    def test_css_has_panel_class(self) -> None:
        assert ".panel" in self._css_text()

    def test_css_has_docs_table_class(self) -> None:
        assert ".docs-table" in self._css_text()

    def test_css_has_upload_form_class(self) -> None:
        assert ".upload-form" in self._css_text()

    def test_css_has_delete_button_class(self) -> None:
        assert ".delete-button" in self._css_text()

    def test_css_has_toast_classes(self) -> None:
        css = self._css_text()
        assert ".toast" in css
        assert ".toast-hidden" in css
        assert ".toast-success" in css
        assert ".toast-error" in css
        assert ".toast-info" in css

    def test_css_has_navlink_class(self) -> None:
        assert ".navlink" in self._css_text()

    def test_css_has_btn_secondary_class(self) -> None:
        assert ".btn-secondary" in self._css_text()

    def test_css_has_empty_state_class(self) -> None:
        assert ".empty-state" in self._css_text()

    def test_css_has_brand_suffix_class(self) -> None:
        assert ".brand-suffix" in self._css_text()

    def test_css_brace_balance(self) -> None:
        css = self._css_text()
        assert css.count("{") == css.count("}"), (
            f"brace imbalance: {css.count('{')} open vs {css.count('}')} close"
        )


# ===========================================================================
# Class 8 — admin.html structural integrity
# ===========================================================================


class TestAdminHTML:
    """Static checks on the Jinja2 template that don't fit cleanly
    into the FastAPI client (the Jinja syntax markers, accessibility
    hints, etc.)."""

    def _html_text(self) -> str:
        return ADMIN_HTML_PATH.read_text(encoding="utf-8")

    def test_html_uses_utf8(self) -> None:
        assert 'charset="UTF-8"' in self._html_text()

    def test_html_has_viewport_meta(self) -> None:
        body = self._html_text()
        assert "viewport" in body
        assert "width=device-width" in body

    def test_html_has_lang_attribute(self) -> None:
        assert 'lang="en"' in self._html_text()

    def test_html_uses_jinja_loop_for_doc_types(self) -> None:
        # The template must use a Jinja loop to render the <option>
        # tags from the supported_doc_types context var — not
        # hard-code them (that would let the list drift from
        # SUPPORTED_DOC_TYPES).
        body = self._html_text()
        assert "{% for dt in supported_doc_types %}" in body
        assert "{% endfor %}" in body

    def test_html_uses_jinja_if_for_selected(self) -> None:
        body = self._html_text()
        assert "{% if dt == default_doc_type %}" in body

    def test_html_no_inline_event_handlers(self) -> None:
        body = self._html_text()
        for attr in ("onclick=", "onload=", "onerror=", "onmouseover=", "onsubmit="):
            assert attr not in body, f"found inline event handler: {attr!r}"

    def test_html_aria_live_on_toast(self) -> None:
        # The toast uses aria-live="polite" so screen readers
        # announce upload results without interrupting.
        body = self._html_text()
        assert "aria-live" in body

    def test_html_file_input_required(self) -> None:
        # Browser-level guard: don't submit with no file chosen.
        assert "required" in self._html_text()


# ===========================================================================
# Class 9 — End-to-end (upload → list → delete via the real API)
# ===========================================================================


class TestAdminEndToEnd:
    """The full upload → list → delete flow works through the
    TestClient. Uses the Step 4.18 ``/api/documents`` surface and
    the Fake LLM + Fake embedder so the test stays hermetic.

    The Fake embedder produces deterministic 384-dim vectors, so
    the upload pipeline runs end-to-end against the on-disk FAISS
    store — no model downloads, no llama-server.
    """

    def _txt_bytes(self, body: str = "Hello TinyRAG admin.\n") -> bytes:
        return body.encode("utf-8")

    def test_empty_admin_state(self, client: TestClient) -> None:
        # GET /api/documents on a fresh tmpdir returns an empty list.
        resp = client.get("/api/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["documents"] == []

    def test_upload_then_list(self, client: TestClient) -> None:
        # Upload a small TXT -> 200 + ok=True, list now has 1 doc.
        resp = client.post(
            "/api/documents",
            files={"file": ("hello.txt", io.BytesIO(self._txt_bytes()), "text/plain")},
            data={"doc_type": "manual"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ok"] is True
        assert body["doc_id"]
        assert body["doc_type"] == "manual"
        assert body["num_chunks"] >= 1
        # The IngestionReport carries the doc_id; the filename is
        # applied via metadata.update_document_provenance *after*
        # run_ingest returns. Verify via the list endpoint (the
        # authoritative view).
        listed = client.get("/api/documents").json()
        assert listed["count"] == 1
        assert listed["documents"][0]["filename"] == "hello.txt"
        assert listed["documents"][0]["doc_type"] == "manual"

    def test_upload_oversize_returns_413(self, client: TestClient) -> None:
        # 51 MB of zeros — well over the 50 MB cap.
        big = b"\0" * (MAX_UPLOAD_BYTES + 1024 * 1024)
        resp = client.post(
            "/api/documents",
            files={"file": ("big.txt", io.BytesIO(big), "text/plain")},
            data={"doc_type": "manual"},
        )
        assert resp.status_code == 413
        body = resp.json()
        assert body["error"] == "file_too_large"

    def test_upload_unsupported_extension_returns_400(self, client: TestClient) -> None:
        resp = client.post(
            "/api/documents",
            files={
                "file": (
                    "data.csv",
                    io.BytesIO(b"a,b,c\n1,2,3\n"),
                    "text/csv",
                )
            },
            data={"doc_type": "manual"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"] == "unsupported_file_type"

    def test_delete_after_upload_removes_doc(self, client: TestClient) -> None:
        # Upload → list shows 1 → delete → list shows 0.
        up = client.post(
            "/api/documents",
            files={"file": ("rm.txt", io.BytesIO(self._txt_bytes("rm")), "text/plain")},
            data={"doc_type": "manual"},
        )
        assert up.status_code == 200
        doc_id = up.json()["doc_id"]

        listed = client.get("/api/documents").json()
        assert listed["count"] == 1

        rm = client.delete(f"/api/documents/{doc_id}")
        assert rm.status_code == 200
        rm_body = rm.json()
        assert rm_body["document_id"] == doc_id
        assert rm_body["chunks_removed"] >= 1
        assert rm_body["vectors_removed"] >= 0

        listed_after = client.get("/api/documents").json()
        assert listed_after["count"] == 0

    def test_delete_unknown_id_returns_404(self, client: TestClient) -> None:
        resp = client.delete("/api/documents/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"] == "document_not_found"


# ===========================================================================
# Class 10 — Route wiring (defensive: confirm /admin is registered + serves HTML)
# ===========================================================================


class TestAdminRouteWiring:
    """``/admin`` must be registered on the FastAPI app + serve HTML."""

    def test_admin_route_registered(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        app = create_app(settings, llm_kind="fake", embedder_kind="fake")
        paths = {getattr(r, "path", None) for r in app.routes}
        assert "/admin" in paths, f"/admin not registered; got: {paths}"

    def test_admin_route_renders_template(self, client: TestClient) -> None:
        # Sanity-check that /admin serves the admin.html template
        # (not, say, the chat template by mistake).
        body = client.get("/admin").text
        assert "<form" in body
        assert 'id="upload-form"' in body
        assert 'id="docs-table-container"' in body


# ===========================================================================
# Class 11 — Full-page load smoke (regression pin that the admin page
# doesn't break /healthz, /api/status, or /static/* for the chat page)
# ===========================================================================


class TestAdminPageSmoke:
    """A browser navigating to /admin hits four endpoints on first
    paint — all must remain 2xx."""

    def test_four_endpoints_all_2xx(self, client: TestClient) -> None:
        r1 = client.get("/admin")
        r2 = client.get("/static/admin.js")
        r3 = client.get("/static/style.css")
        r4 = client.get("/api/status")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 200
        assert r4.status_code == 200

    def test_healthz_still_works(self, client: TestClient) -> None:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"ok": "true"}

    def test_chat_page_still_renders(self, client: TestClient) -> None:
        # Regression pin: GET / (chat) must still render even after
        # adding /admin.
        r = client.get("/")
        assert r.status_code == 200
        assert "TinyRAG" in r.text
        assert 'id="messages"' in r.text
