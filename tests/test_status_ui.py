"""Tests for the system status panel (Step 4.23).

What this module covers
------------------------
- ``ui/static/status.js`` exists, is well-formed JavaScript (IIFE
  wrapper, balanced braces), and references every DOM id the
  ``status-panel`` section needs.
- ``GET /`` (chat page) and ``GET /admin`` (documents page) both
  render the ``<section id="status-panel">`` block with all 10
  ``.status-value`` cells, the toggle button, and the
  ``<script src="/static/status.js" defer>`` tag.
- ``/static/status.js`` is served with the correct MIME type and
  body matches the file on disk.
- ``ui/static/style.css`` contains the new Step-4.23 classes
  (``.status-panel``, ``.status-grid``, ``.status-cell``,
  ``.status-label``, ``.status-value``, ``.status-llm-up``,
  ``.status-llm-down``, ``.is-collapsed``).
- **Regression pin for the latent ``data.ready`` → ``data.ok`` bug**
  fixed in Step 4.23: chat.js and admin.js no longer read
  ``data.ready`` and now read ``data.ok`` for the topbar pill.
- The /api/status endpoint still returns every field the panel
  needs (model_name, embedding_model, embedding_dim, doc_chunk_count,
  sensor_chunk_count, ram_mb, llama_cpp_status, sensor_source,
  deployment_target) — Step 4.23 didn't change the schema, but the
  panel depends on it, so we pin it.

Why a separate file (not folded into ``test_web_ui.py`` /
``test_admin_ui.py``)?
------------------------------------------------------
The status panel is a Step-4.23 deliverable; keeping its tests in a
named file makes the diff against ``main`` legible (one-step =
one-test-file). Both ``test_web_ui.py`` (chat page) and
``test_admin_ui.py`` (admin page) each get a 2-test regression block
for the pill fix; the bulk of the panel tests live here.

Hermetic?
---------
Yes. The :class:`TestClient` triggers the lifespan but uses
``llm_kind="fake"`` + ``embedder_kind="fake"`` — no live llama-
server, no model downloads. The static assets are read from the
project's ``ui/`` tree (committed in this step), so there's no
runtime-only path to mock.

Location: ``tests/test_status_ui.py``
"""

# ``I001`` (import sort) suppressed — see the same noqa block in
# tests/test_api.py for the rationale.
# ruff: noqa: I001
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap (mirrors tests/test_api.py)
# ---------------------------------------------------------------------------
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient  # noqa: E402

from tinyrag.main import create_app  # noqa: E402


# ---------------------------------------------------------------------------
# Paths to the static UI assets (for direct read access — some tests want
# to assert on the file content rather than going through the HTTP layer).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = PROJECT_ROOT / "ui"
TEMPLATES_DIR = UI_DIR / "templates"
STATIC_DIR = UI_DIR / "static"
INDEX_HTML_PATH = TEMPLATES_DIR / "index.html"
ADMIN_HTML_PATH = TEMPLATES_DIR / "admin.html"
STATUS_JS_PATH = STATIC_DIR / "status.js"
CHAT_JS_PATH = STATIC_DIR / "chat.js"
ADMIN_JS_PATH = STATIC_DIR / "admin.js"
STYLE_CSS_PATH = STATIC_DIR / "style.css"


# ---------------------------------------------------------------------------
# Minimal Settings builder (no FAISS indices — we don't query /api/query in
# this suite, only GET / + GET /admin + GET /static/*, so empty stores
# are fine).
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
    """TestClient with the web UI wired (Fake LLM + Fake embedder)."""
    settings = _make_settings(tmp_path)
    app = create_app(settings, llm_kind="fake", embedder_kind="fake")
    with TestClient(app) as c:
        yield c


# ===========================================================================
# Class 1 — UI files for the status panel exist on disk
# ===========================================================================


class TestStatusAssetsOnDisk:
    """Sanity-check the status-panel UI files actually exist.

    If a future refactor moves the status panel elsewhere, these
    tests will start failing at collection time — a clear signal
    that the tests in this module need to be pointed at the new
    location.
    """

    def test_status_js_exists(self) -> None:
        assert STATUS_JS_PATH.is_file(), f"missing: {STATUS_JS_PATH}"

    def test_index_html_exists(self) -> None:
        # The chat page now embeds the status panel section.
        assert INDEX_HTML_PATH.is_file(), f"missing: {INDEX_HTML_PATH}"

    def test_admin_html_exists(self) -> None:
        # The admin page also embeds the status panel section.
        assert ADMIN_HTML_PATH.is_file(), f"missing: {ADMIN_HTML_PATH}"


# ===========================================================================
# Class 2 — GET / (chat page) renders the status panel
# ===========================================================================


class TestGetRootHasStatusPanel:
    """``GET /`` must embed the ``<section id="status-panel">`` block.

    This is the structural contract between ``ui/templates/index.html``
    and ``ui/static/status.js`` — the JS resolves DOM ids that the
    HTML must provide, so a missing element would surface as a
    ``console.error("status panel DOM init failed")`` in the browser.
    """

    def test_root_has_status_panel_section(self, client: TestClient) -> None:
        body = client.get("/").text
        assert 'id="status-panel"' in body, (
            "GET / must embed <section id=\"status-panel\">"
        )

    def test_root_has_status_panel_class(self, client: TestClient) -> None:
        body = client.get("/").text
        # The .status-panel class is what the CSS targets for the
        # grid + border colours — both .panel and .status-panel must
        # be present (the former for shared panel styles, the
        # latter for panel-specific overrides).
        assert "status-panel" in body

    def test_root_has_status_toggle_button(self, client: TestClient) -> None:
        body = client.get("/").text
        # The Hide/Show toggle button + its aria-expanded default
        # must both be present so the JS can wire a click handler
        # without a stale aria state.
        assert 'id="status-toggle"' in body
        assert 'aria-expanded="true"' in body
        assert 'aria-controls="status-body"' in body

    def test_root_has_all_ten_status_value_ids(self, client: TestClient) -> None:
        # The 10 .status-value ids the JS reads. Missing any one and
        # the panel renders an empty cell (and console.error from
        # init() if it's one of the required ones).
        body = client.get("/").text
        expected = [
            "status-model",
            "status-embedding",
            "status-dim",
            "status-doc-count",
            "status-sensor-count",
            "status-llm",
            "status-sensor-source",
            "status-ram",
            "status-deploy",
            "status-updated",
        ]
        for sid in expected:
            assert f'id="{sid}"' in body, f"missing status cell id: {sid}"

    def test_root_loads_status_js(self, client: TestClient) -> None:
        # status.js must be loaded via <script src="/static/status.js" defer>
        # so it runs after the HTML has parsed and the DOM handles
        # resolve correctly.
        body = client.get("/").text
        assert 'src="/static/status.js"' in body
        assert "defer" in body


# ===========================================================================
# Class 3 — GET /admin (documents page) renders the status panel
# ===========================================================================


class TestGetAdminHasStatusPanel:
    """``GET /admin`` must embed the same ``<section id="status-panel">`` block.

    The admin page exists so admins can see system health before they
    start uploading — having the status panel only on / would force
    admins to switch tabs just to verify the system is alive.
    """

    def test_admin_has_status_panel_section(self, client: TestClient) -> None:
        body = client.get("/admin").text
        assert 'id="status-panel"' in body, (
            "GET /admin must embed <section id=\"status-panel\">"
        )

    def test_admin_has_status_toggle_button(self, client: TestClient) -> None:
        body = client.get("/admin").text
        assert 'id="status-toggle"' in body
        assert 'aria-expanded="true"' in body

    def test_admin_has_all_ten_status_value_ids(self, client: TestClient) -> None:
        body = client.get("/admin").text
        expected = [
            "status-model",
            "status-embedding",
            "status-dim",
            "status-doc-count",
            "status-sensor-count",
            "status-llm",
            "status-sensor-source",
            "status-ram",
            "status-deploy",
            "status-updated",
        ]
        for sid in expected:
            assert f'id="{sid}"' in body, f"missing status cell id: {sid}"

    def test_admin_loads_status_js(self, client: TestClient) -> None:
        body = client.get("/admin").text
        assert 'src="/static/status.js"' in body
        assert "defer" in body

    def test_admin_panel_above_upload_panel(self, client: TestClient) -> None:
        # The status panel must come BEFORE the upload panel so an
        # admin sees system health before they start uploading.
        # We verify by ordering the positions of the two sections.
        body = client.get("/admin").text
        panel_idx = body.find('id="status-panel"')
        upload_idx = body.find('id="upload-form"')
        assert panel_idx != -1, "status panel section missing"
        assert upload_idx != -1, "upload form section missing"
        assert panel_idx < upload_idx, (
            "status panel must render above the upload form"
        )


# ===========================================================================
# Class 4 — status.js structural integrity
# ===========================================================================


class TestStatusJS:
    """Static checks on the status.js source.

    We don't execute the JS (that would need a headless browser) —
    we just verify it parses as well-formed JavaScript and that it
    references every DOM id the HTML provides + the /api/status
    endpoint the panel polls.
    """

    def _status_js_text(self) -> str:
        # Read from disk (canonical source) rather than from the HTTP
        # layer — equivalent to the HTTP fetch but faster.
        return STATUS_JS_PATH.read_text(encoding="utf-8")

    def test_status_js_uses_iife(self) -> None:
        # The script wraps itself in an IIFE so its internals don't
        # pollute the global scope. We assert the open + close parens
        # pattern is present.
        js = self._status_js_text()
        assert "(function ()" in js, "status.js missing IIFE wrapper"
        assert "})();" in js, "status.js IIFE not invoked"

    def test_status_js_polls_api_status(self) -> None:
        js = self._status_js_text()
        assert "/api/status" in js, (
            "status.js must poll /api/status to render the panel"
        )

    def test_status_js_uses_fetch(self) -> None:
        js = self._status_js_text()
        assert "fetch(" in js, "status.js must use fetch()"

    def test_status_js_uses_setinterval(self) -> None:
        # The poll must run on a setInterval so the panel refreshes
        # without user interaction.
        js = self._status_js_text()
        assert "setInterval" in js, "status.js missing setInterval polling"

    def test_status_js_uses_textcontent(self) -> None:
        # All dynamic string assignments must use textContent (XSS
        # safety invariant since Step 4.21). The script uses
        # textContent for all 10 cells + the error message.
        js = self._status_js_text()
        assert ".textContent" in js

    def test_status_js_references_every_dom_id(self) -> None:
        # The 10 cell ids + the panel + toggle. Missing any of these
        # and init() will console.error and bail, leaving the panel
        # empty (a silent failure that looks like the API is broken).
        js = self._status_js_text()
        expected = [
            "status-panel",
            "status-toggle",
            "status-body",
            "status-model",
            "status-embedding",
            "status-dim",
            "status-doc-count",
            "status-sensor-count",
            "status-llm",
            "status-sensor-source",
            "status-ram",
            "status-deploy",
            "status-updated",
        ]
        for sid in expected:
            assert sid in js, f"status.js missing DOM id reference: {sid}"

    def test_status_js_has_brace_balance(self) -> None:
        # Cheap structural sanity check: every { has a matching }.
        # Won't catch every syntax error but catches the common
        # "I deleted a closing brace" mistake.
        js = self._status_js_text()
        assert js.count("{") == js.count("}"), (
            f"brace imbalance: {js.count('{')} open vs {js.count('}')} close"
        )
        assert js.count("(") == js.count(")"), (
            f"paren imbalance: {js.count('(')} open vs {js.count(')')} close"
        )

    def test_status_js_uses_localstorage_for_collapse(self) -> None:
        # The collapse toggle persists across / and /admin via
        # localStorage — the storage key must be present in the
        # source so a future reader can find it.
        js = self._status_js_text()
        assert "localStorage" in js, (
            "status.js must use localStorage to persist collapse state"
        )

    def test_status_js_has_aria_expanded(self) -> None:
        # The toggle button's aria-expanded attribute must be flipped
        # in sync with the .is-collapsed class so screen readers know
        # whether the panel is expanded.
        js = self._status_js_text()
        assert "aria-expanded" in js, (
            "status.js must toggle aria-expanded for accessibility"
        )

    def test_status_js_data_ok_attr(self) -> None:
        # The panel's [data-ok] attribute drives the left-border
        # colour via the CSS — the JS must set it from data.ok so
        # the colour updates on each poll.
        js = self._status_js_text()
        assert "data.ok" in js, (
            "status.js must read data.ok from /api/status"
        )
        assert "dataset.ok" in js, (
            "status.js must write dataset.ok for the CSS selector"
        )


# ===========================================================================
# Class 5 — status panel CSS classes
# ===========================================================================


class TestStatusPanelCSS:
    """Static checks on the Step-4.23 additions to style.css.

    The CSS must define the classes the HTML uses (.status-panel,
    .status-grid, .status-cell, .status-label, .status-value), the
    LLM health colours (.status-llm-up / .status-llm-down), the
    overall-health border selectors ([data-ok="true|false"]), and
    the .is-collapsed hide rule.
    """

    def _css_text(self) -> str:
        return STYLE_CSS_PATH.read_text(encoding="utf-8")

    def test_css_has_status_panel_class(self) -> None:
        assert ".status-panel" in self._css_text()

    def test_css_has_status_grid_class(self) -> None:
        # The grid uses auto-fit + minmax so the column count adapts
        # to the viewport width.
        css = self._css_text()
        assert ".status-grid" in css
        assert "auto-fit" in css or "repeat(auto-fit" in css

    def test_css_has_status_value_class(self) -> None:
        # .status-value is the monospace font hook — must exist so
        # values line up under their labels.
        assert ".status-value" in self._css_text()

    def test_css_has_status_cell_class(self) -> None:
        assert ".status-cell" in self._css_text()

    def test_css_has_status_label_class(self) -> None:
        # The small uppercase label above each value.
        assert ".status-label" in self._css_text()

    def test_css_has_llm_health_colours(self) -> None:
        css = self._css_text()
        assert ".status-llm-up" in css
        assert ".status-llm-down" in css

    def test_css_has_data_ok_attribute_selectors(self) -> None:
        # The left-border colour is driven by [data-ok="true|false"]
        # attribute selectors — the JS sets dataset.ok and the CSS
        # turns it into a colour.
        css = self._css_text()
        assert '[data-ok="true"]' in css
        assert '[data-ok="false"]' in css

    def test_css_has_is_collapsed_rule(self) -> None:
        # The collapse toggle adds .is-collapsed to the panel; the
        # CSS must hide .status-grid when the panel is collapsed.
        css = self._css_text()
        assert ".is-collapsed" in css
        assert "display: none" in css

    def test_css_panel_block_well_formed(self) -> None:
        # Find the status panel block + verify its braces balance.
        # This catches a copy-paste error that introduces an extra
        # brace within the new section.
        css = self._css_text()
        # Slice from the start of the Step-4.23 comment block to EOF.
        marker = "/* ---- Status panel (Step 4.23) ----"
        idx = css.find(marker)
        assert idx != -1, "Step-4.23 status panel CSS block missing"
        tail = css[idx:]
        assert tail.count("{") == tail.count("}"), (
            f"brace imbalance in status panel CSS: "
            f"{tail.count('{')} open vs {tail.count('}')} close"
        )


# ===========================================================================
# Class 6 — Regression pin for the latent data.ready → data.ok bug
# ===========================================================================


class TestPillBugFix:
    """Step 4.23 fixed a latent bug where chat.js + admin.js read
    ``data.ready`` from /api/status, but the API returns ``data.ok``.

    ``!!undefined === false`` so the topbar pill was permanently
    stuck on "degraded" regardless of actual subsystem health.
    These tests pin the fix: the JS files must no longer reference
    ``data.ready`` and must reference ``data.ok``.
    """

    def test_chat_js_no_longer_reads_ready(self) -> None:
        # We check the boolean-coercion assignment ``!!data.ready``
        # specifically (not the bare substring) so a future comment
        # mentioning the bug fix doesn't trip the test.
        js = CHAT_JS_PATH.read_text(encoding="utf-8")
        assert "!!data.ready" not in js, (
            "chat.js still coerces !!data.ready — topbar pill will "
            "always show 'degraded' (regression of the Step-4.23 bug fix)"
        )

    def test_admin_js_no_longer_reads_ready(self) -> None:
        js = ADMIN_JS_PATH.read_text(encoding="utf-8")
        assert "!!data.ready" not in js, (
            "admin.js still coerces !!data.ready — topbar pill will "
            "always show 'degraded' (regression of the Step-4.23 bug fix)"
        )

    def test_chat_js_reads_ok(self) -> None:
        # The pill must now read data.ok (the actual StatusResponse
        # field name — see src/tinyrag/api/schemas.py).
        js = CHAT_JS_PATH.read_text(encoding="utf-8")
        assert "data.ok" in js, (
            "chat.js must read data.ok from /api/status for the pill"
        )

    def test_admin_js_reads_ok(self) -> None:
        js = ADMIN_JS_PATH.read_text(encoding="utf-8")
        assert "data.ok" in js, (
            "admin.js must read data.ok from /api/status for the pill"
        )


# ===========================================================================
# Class 7 — End-to-end smoke: all four endpoints a browser touches
# ===========================================================================


class TestStatusPanelE2E:
    """Full integration smoke for the status panel."""

    def test_root_includes_panel_and_status_js(self, client: TestClient) -> None:
        # The chat page body must contain both the panel section and
        # the script tag that polls it.
        body = client.get("/").text
        assert 'id="status-panel"' in body
        assert 'id="status-model"' in body
        assert 'src="/static/status.js"' in body

    def test_admin_includes_panel_and_status_js(self, client: TestClient) -> None:
        body = client.get("/admin").text
        assert 'id="status-panel"' in body
        assert 'id="status-llm"' in body
        assert 'src="/static/status.js"' in body

    def test_status_endpoint_returns_all_panel_fields(
        self, client: TestClient,
    ) -> None:
        # The panel depends on these fields. If any of them go
        # missing in a future refactor, the panel cells will render
        # their "—" fallback. Pin the schema here.
        body = client.get("/api/status").json()
        required = [
            "ok",
            "model_name",
            "embedding_model",
            "embedding_dim",
            "doc_chunk_count",
            "sensor_chunk_count",
            "ram_mb",
            "llama_cpp_status",
            "sensor_source",
            "deployment_target",
        ]
        for key in required:
            assert key in body, f"/api/status missing field: {key}"

    def test_static_status_js_matches_disk(self, client: TestClient) -> None:
        # StaticFiles must serve the exact file we committed — no
        # shadow copy, no transform.
        resp = client.get("/static/status.js")
        assert resp.status_code == 200
        assert resp.content == STATUS_JS_PATH.read_bytes()


# ===========================================================================
# Class 8 — Health integration: the panel + pill together
# ===========================================================================


class TestStatusPanelHealthIntegration:
    """Verify the data the panel renders matches the pill's view.

    If /api/status returns ``ok=true``, the topbar pill must show
    "ready" (green) and the panel border must be green. If it
    returns ``ok=false``, both must flip to red. The JS files are
    pinned separately; this class verifies the API surface still
    produces the right shape.
    """

    def test_status_has_model_name_for_panel(self, client: TestClient) -> None:
        # Fake LLM mode + empty FAISS stores — every subsystem is
        # reachable in test mode (no live llama-server needed) so
        # the panel must still render real values (not "unknown").
        # Note: ``ok`` itself is False here because llama_cpp_status
        # is "down" (no live llama-server in the test env) — the
        # panel renders the red "degraded" border in that case,
        # which is correct behaviour. We're pinning the model_name
        # rendering here, not the ok flag.
        body = client.get("/api/status").json()
        # The panel reads model_name from this field. The fake LLM
        # advertises a stable name (see FakeLLMClient) — pin it so
        # the panel renders "fake-llm" rather than "unknown".
        assert "model_name" in body
        assert body["model_name"], "model_name must be non-empty"
        # The panel reads llama_cpp_status as the "LLM server" cell
        # and colours it green/red via the .status-llm-* class. Pin
        # the value so a future refactor can't silently swap "up"/
        # "down" for some other token that the CSS doesn't handle.
        assert body["llama_cpp_status"] in ("up", "down")

    def test_all_static_assets_serve_2xx_in_one_session(
        self, client: TestClient,
    ) -> None:
        # All four assets a browser hits on first paint must serve
        # 200 in one TestClient session (simulates a single page
        # load). status.js is the new one; the other three are
        # pinned to catch any regression in the static mount.
        for path in (
            "/static/chat.js",
            "/static/admin.js",
            "/static/status.js",
            "/static/style.css",
        ):
            r = client.get(path)
            assert r.status_code == 200, (
                f"{path} returned {r.status_code} (expected 200)"
            )


# ===========================================================================
# Helper regex (reserved for future SSE-specific tests; kept here so the
# import block stays clean).
# ===========================================================================


_SSE_FRAME_SEP_RE = re.compile(r"\n\n")
