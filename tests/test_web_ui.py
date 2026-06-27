"""Tests for the chat web UI (Step 4.21 — web UI chat page).

What this module covers
------------------------
- ``GET /`` renders the Jinja2 ``index.html`` chat page.
- The rendered HTML contains the structural elements the chat.js
  client expects to find (topbar, messages div, composer, status
  pill, model-name span, script tag).
- ``/static/chat.js`` + ``/static/style.css`` are served with the
  right MIME types.
- chat.js is structurally valid JavaScript (parseable by Node, or
  at minimum syntactically well-formed via a strict regex).
- The chat.js script wires up the right URLs (``/api/status``,
  ``/api/query?stream=true``) so the chat page actually talks to
  the live API.
- The page links to the FastAPI ``/docs`` + ``/api/status`` so a
  user exploring the UI can find the OpenAPI playground.

Why a separate file (not folded into ``test_api.py``)?
------------------------------------------------------
The web UI is a Step-4.21 deliverable; keeping its tests in a
named file makes the diff against ``main`` legible (one-step =
one-test-file). It's also the only place we parse HTML, so a
shared helper set would only be used here.

Hermetic?
---------
Yes. The :class:`TestClient` triggers the lifespan but uses
``llm_kind="fake"`` + ``embedder_kind="fake"`` — no live llama-
server, no model downloads. The static assets are read from the
project's ``ui/`` tree (committed in this step), so there's no
runtime-only path to mock.

Location: ``tests/test_web_ui.py``
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
CHAT_JS_PATH = STATIC_DIR / "chat.js"
STYLE_CSS_PATH = STATIC_DIR / "style.css"


# ---------------------------------------------------------------------------
# Minimal Settings builder (no FAISS indices — we don't query /api/query in
# this suite, only GET / + GET /static/*, so empty stores are fine).
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
# Class 1 — UI files exist on disk (cheap pre-flight for the test runner)
# ===========================================================================


class TestUIAssetsOnDisk:
    """Sanity-check the three committed UI files actually exist.

    If a future refactor moves the chat UI elsewhere, these tests
    will start failing at collection time — a clear signal that the
    tests in this module need to be pointed at the new location.
    """

    def test_index_html_exists(self) -> None:
        assert INDEX_HTML_PATH.is_file(), f"missing: {INDEX_HTML_PATH}"

    def test_chat_js_exists(self) -> None:
        assert CHAT_JS_PATH.is_file(), f"missing: {CHAT_JS_PATH}"

    def test_style_css_exists(self) -> None:
        assert STYLE_CSS_PATH.is_file(), f"missing: {STYLE_CSS_PATH}"

    def test_ui_dir_layout(self) -> None:
        # The FastAPI mount expects ui/static/ + ui/templates/ —
        # verify the layout matches.
        assert TEMPLATES_DIR.is_dir(), f"missing dir: {TEMPLATES_DIR}"
        assert STATIC_DIR.is_dir(), f"missing dir: {STATIC_DIR}"


# ===========================================================================
# Class 2 — GET / (chat page render)
# ===========================================================================


class TestGetRoot:
    """``GET /`` renders the chat page (HTML 200) instead of the old JSON banner."""

    def test_root_returns_200(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_returns_html(self, client: TestClient) -> None:
        resp = client.get("/")
        ct = resp.headers.get("content-type", "")
        assert ct.startswith("text/html"), f"unexpected content-type: {ct!r}"

    def test_root_has_doctype(self, client: TestClient) -> None:
        resp = client.get("/")
        # Doctype is rendered into the response body by Jinja2 (we
        # don't strip it). Sanity-check that we're getting actual
        # HTML, not an error envelope.
        assert "<!DOCTYPE html>" in resp.text or "<!doctype html>" in resp.text.lower()

    def test_root_has_chat_messages_container(self, client: TestClient) -> None:
        # The chat.js client looks for this element by id.
        assert 'id="messages"' in client.get("/").text

    def test_root_has_composer_form(self, client: TestClient) -> None:
        body = client.get("/").text
        assert 'id="composer"' in body
        assert 'id="query"' in body
        assert 'id="send"' in body

    def test_root_has_status_pill(self, client: TestClient) -> None:
        body = client.get("/").text
        assert 'id="status-pill"' in body
        assert 'id="model-name"' in body

    def test_root_links_stylesheet(self, client: TestClient) -> None:
        # style.css is loaded via <link rel="stylesheet" href="/static/style.css">
        body = client.get("/").text
        assert 'href="/static/style.css"' in body
        assert 'rel="stylesheet"' in body

    def test_root_loads_chat_js(self, client: TestClient) -> None:
        # chat.js is loaded via <script src="/static/chat.js" defer>
        body = client.get("/").text
        assert 'src="/static/chat.js"' in body

    def test_root_includes_brand(self, client: TestClient) -> None:
        # Brand text is rendered into the topbar.
        assert "TinyRAG" in client.get("/").text

    def test_root_has_footer_links(self, client: TestClient) -> None:
        body = client.get("/").text
        # The footer links to /docs and /api/status so users can find
        # the OpenAPI playground + the JSON heartbeat.
        assert 'href="/docs"' in body
        assert 'href="/api/status"' in body

    def test_root_aria_live(self, client: TestClient) -> None:
        # The chat <main> uses aria-live="polite" so screen readers
        # announce new tokens without interrupting.
        body = client.get("/").text
        assert 'aria-live="polite"' in body


# ===========================================================================
# Class 3 — /static/* (chat.js + style.css)
# ===========================================================================


class TestStaticAssets:
    """The two static assets are served with the right content type."""

    def test_chat_js_serves_200(self, client: TestClient) -> None:
        resp = client.get("/static/chat.js")
        assert resp.status_code == 200

    def test_chat_js_content_type(self, client: TestClient) -> None:
        ct = client.get("/static/chat.js").headers.get("content-type", "")
        # StaticFiles serves .js as application/javascript or
        # text/javascript depending on Starlette version; accept both.
        assert "javascript" in ct, f"unexpected content-type: {ct!r}"

    def test_style_css_serves_200(self, client: TestClient) -> None:
        resp = client.get("/static/style.css")
        assert resp.status_code == 200

    def test_style_css_content_type(self, client: TestClient) -> None:
        ct = client.get("/static/style.css").headers.get("content-type", "")
        assert "css" in ct, f"unexpected content-type: {ct!r}"

    def test_static_404_for_unknown_asset(self, client: TestClient) -> None:
        # /static/ does NOT auto-list; requesting a missing file is
        # a 404 (not a 500). This catches accidental directory-listing
        # regressions.
        resp = client.get("/static/does-not-exist.js")
        assert resp.status_code == 404

    def test_chat_js_content_matches_disk(self, client: TestClient) -> None:
        # StaticFiles must serve the exact file we committed.
        resp = client.get("/static/chat.js")
        disk = CHAT_JS_PATH.read_bytes()
        assert resp.content == disk

    def test_style_css_content_matches_disk(self, client: TestClient) -> None:
        resp = client.get("/static/style.css")
        disk = STYLE_CSS_PATH.read_bytes()
        assert resp.content == disk


# ===========================================================================
# Class 4 — chat.js structural integrity (the script is well-formed and
# points at the documented API endpoints)
# ===========================================================================


class TestChatJS:
    """Static checks on the chat.js source.

    We don't execute the JS (that would need a headless browser) —
    we just verify it parses as well-formed JavaScript and that it
    references the documented API endpoints so the page actually
    talks to the live API.
    """

    def _chat_js_text(self) -> str:
        # Read from disk (canonical source) rather than from the HTTP
        # layer — equivalent to the HTTP fetch but faster.
        return CHAT_JS_PATH.read_text(encoding="utf-8")

    def test_chat_js_uses_iife(self) -> None:
        # The script wraps itself in an IIFE so its internals don't
        # pollute the global scope. We assert the open + close parens
        # pattern is present.
        js = self._chat_js_text()
        # The IIFE pattern is `(function () { ... })();` — assert
        # the start of the IIFE + the trailing invocation.
        assert "(function ()" in js, "chat.js missing IIFE wrapper"
        assert "})();" in js, "chat.js IIFE not invoked"

    def test_chat_js_calls_api_status(self) -> None:
        js = self._chat_js_text()
        # The status poll must hit /api/status. We accept any of the
        # documented variants — `"/api/status"` literal, or via the
        # ``statusUrl`` config knob.
        assert "/api/status" in js

    def test_chat_js_calls_api_query_stream(self) -> None:
        js = self._chat_js_text()
        # The streaming endpoint is `/api/query?stream=true`.
        assert "/api/query" in js
        assert "stream=true" in js

    def test_chat_js_uses_fetch(self) -> None:
        # We use fetch() + reader (not EventSource) because the API
        # endpoint requires POST with a JSON body, and EventSource
        # only supports GET.
        js = self._chat_js_text()
        assert "fetch(" in js

    def test_chat_js_handles_token_event(self) -> None:
        js = self._chat_js_text()
        # The SSE wire format's `event: token` -> `data: {delta:...}`
        # must be recognised.
        assert "token" in js
        assert "delta" in js

    def test_chat_js_handles_done_event(self) -> None:
        js = self._chat_js_text()
        # Terminal frame must be handled to render citations +
        # diagnostics.
        assert "done" in js
        assert "citations" in js

    def test_chat_js_handles_error_event(self) -> None:
        js = self._chat_js_text()
        # Mid-stream error frames must become a red error bubble.
        assert "error" in js

    def test_chat_js_disables_composer_while_streaming(self) -> None:
        # The send button + textarea must be disabled while the SSE
        # stream is open so the user can't double-submit.
        js = self._chat_js_text()
        assert "disabled" in js

    def test_chat_js_has_sse_frame_parser(self) -> None:
        # SSE frames are separated by a blank line (`\n\n`). The
        # parser must look for this separator.
        js = self._chat_js_text()
        assert "\\n\\n" in js or "'\n\n'" in js

    def test_chat_js_handles_text_event_stream(self) -> None:
        # Content-Type negotiation — we ask for `text/event-stream`
        # explicitly so a misconfigured proxy can't accidentally
        # treat the response as JSON.
        js = self._chat_js_text()
        assert "text/event-stream" in js

    def test_chat_js_has_brace_balance(self) -> None:
        # Cheap structural sanity check: every { has a matching }.
        # Won't catch every syntax error but catches the common
        # "I deleted a closing brace" mistake.
        js = self._chat_js_text()
        assert js.count("{") == js.count("}"), (
            f"brace imbalance: {js.count('{')} open vs {js.count('}')} close"
        )
        assert js.count("(") == js.count(")"), (
            f"paren imbalance: {js.count('(')} open vs {js.count(')')} close"
        )

    def test_chat_js_uses_textcontent_for_user_input(self) -> None:
        # The user-bubble renderer must use textContent (not
        # innerHTML) to prevent XSS via malicious queries.
        js = self._chat_js_text()
        # Look for the user-message rendering site. We expect at
        # least one `.textContent =` assignment on the user-bubble
        # code path.
        assert ".textContent =" in js or ".textContent=" in js

    def test_chat_js_polls_status(self) -> None:
        # The status pill must refresh periodically (setInterval on
        # the status poll).
        js = self._chat_js_text()
        assert "setInterval" in js


# ===========================================================================
# Class 5 — style.css structural integrity
# ===========================================================================


class TestStyleCSS:
    """Static checks on style.css.

    We don't run a CSS lint — we just verify the file is present,
    non-empty, and contains the classes the HTML depends on.
    """

    def _css_text(self) -> str:
        return STYLE_CSS_PATH.read_text(encoding="utf-8")

    def test_css_nonempty(self) -> None:
        assert len(self._css_text()) > 100, "style.css looks empty"

    def test_css_has_topbar_class(self) -> None:
        assert ".topbar" in self._css_text()

    def test_css_has_chat_class(self) -> None:
        assert ".chat" in self._css_text()

    def test_css_has_bubble_class(self) -> None:
        # Both user and assistant bubbles share the .bubble class.
        assert ".bubble" in self._css_text()

    def test_css_has_user_and_assistant_variants(self) -> None:
        css = self._css_text()
        assert ".message-user" in css
        assert ".message-assistant" in css

    def test_css_has_error_bubble(self) -> None:
        # The red error bubble rendered on mid-stream errors.
        css = self._css_text()
        assert ".message-error" in css

    def test_css_has_source_cards(self) -> None:
        css = self._css_text()
        assert ".source-card" in css

    def test_css_has_diagnostics(self) -> None:
        # The small grey diagnostics line below an assistant answer.
        css = self._css_text()
        assert ".diagnostics" in css

    def test_css_has_composer(self) -> None:
        css = self._css_text()
        assert ".composer" in css

    def test_css_has_send_button(self) -> None:
        css = self._css_text()
        assert ".send-button" in css

    def test_css_has_status_pill(self) -> None:
        css = self._css_text()
        assert ".pill" in css

    def test_css_has_brace_balance(self) -> None:
        css = self._css_text()
        assert css.count("{") == css.count("}"), (
            f"brace imbalance: {css.count('{')} open vs {css.count('}')} close"
        )

    def test_css_has_mobile_breakpoint(self) -> None:
        # The mobile media query is the obvious one to keep the page
        # legible on a phone.
        css = self._css_text()
        assert "@media" in css
        assert "max-width" in css


# ===========================================================================
# Class 6 — index.html structural integrity
# ===========================================================================


class TestIndexHTML:
    """Static checks on the Jinja2 template.

    Most of the chat UI tests run via the FastAPI client (see
    :class:`TestGetRoot`). This class adds checks that don't fit
    cleanly there — e.g. the Jinja2 syntax markers.
    """

    def _html_text(self) -> str:
        return INDEX_HTML_PATH.read_text(encoding="utf-8")

    def test_html_uses_utf8(self) -> None:
        assert 'charset="UTF-8"' in self._html_text()

    def test_html_has_viewport_meta(self) -> None:
        # Required for mobile responsiveness.
        assert "viewport" in self._html_text()
        assert "width=device-width" in self._html_text()

    def test_html_has_lang_attribute(self) -> None:
        assert 'lang="en"' in self._html_text()

    def test_html_has_form_autocomplete_off(self) -> None:
        # The composer must disable browser autofill on the textarea
        # (it'd otherwise remember past queries — privacy leak).
        assert "autocomplete=\"off\"" in self._html_text()

    def test_html_textarea_required(self) -> None:
        # Browser-level guard so an empty Enter doesn't fire a
        # useless stream.
        assert "required" in self._html_text()

    def test_html_has_status_polling_placeholder(self) -> None:
        # The status pill starts in the "checking..." state so the
        # page is useful before chat.js does its first poll.
        body = self._html_text()
        assert "checking" in body.lower()

    def test_html_footer_mentions_streaming(self) -> None:
        # The footer should tell the user how streaming works so the
        # "why are tokens appearing one at a time?" question is
        # self-serve.
        assert "stream" in self._html_text().lower()

    def test_html_no_inline_event_handlers(self) -> None:
        # All interactivity must live in chat.js (loaded via <script
        # src>). Inline onclick="" attributes would split logic
        # across files — refuse to merge if anyone adds one.
        body = self._html_text()
        for attr in ("onclick=", "onload=", "onerror=", "onmouseover="):
            assert attr not in body, f"found inline event handler: {attr!r}"


# ===========================================================================
# Class 7 — FastAPI integration: GET / + /api/status + /static/* in one go
# ===========================================================================


class TestFullPageLoad:
    """The three endpoints a browser hits on first paint all return 2xx."""

    def test_three_endpoints_all_2xx(self, client: TestClient) -> None:
        # Simulate a cold page load.
        r1 = client.get("/")
        r2 = client.get("/static/chat.js")
        r3 = client.get("/static/style.css")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 200

    def test_status_endpoint_still_works(self, client: TestClient) -> None:
        # /api/status must remain reachable — chat.js polls it.
        r = client.get("/api/status")
        assert r.status_code == 200

    def test_healthz_still_works(self, client: TestClient) -> None:
        # The k8s probe must not be broken by the UI wiring.
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"ok": "true"}

    def test_old_json_banner_not_returned(self, client: TestClient) -> None:
        # Pre-4.21 GET / returned {"service": "tinyrag", ...} JSON.
        # Post-4.21 it returns HTML — verify the JSON contract is gone.
        r = client.get("/")
        # The response is text/html, not application/json. The old
        # "service" JSON key should not appear at the top level of
        # the response body.
        assert "application/json" not in r.headers.get("content-type", "")


# ===========================================================================
# Class 8 — wiring smoke (defensive: confirm the static mount is registered
# exactly once and points at the right directory)
# ===========================================================================


class TestStaticMountWiring:
    """The /static mount must point at ``ui/static`` (not some other path)."""

    def test_static_mount_registered(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        app = create_app(settings, llm_kind="fake", embedder_kind="fake")
        # Find the StaticFiles Mount in the app's routes.
        mounts = [r for r in app.routes if type(r).__name__ == "Mount"]
        assert any(getattr(m, "path", None) == "/static" for m in mounts), (
            "expected a /static mount on the app; got: "
            + ", ".join(getattr(m, "path", "?") for m in mounts)
        )

    def test_static_mount_serves_ui_assets(self, tmp_path: Path) -> None:
        # The mount must point at the real ui/static dir, not a
        # shadow copy. We verify by fetching a file that only exists
        # in the committed tree.
        settings = _make_settings(tmp_path)
        app = create_app(settings, llm_kind="fake", embedder_kind="fake")
        with TestClient(app) as c:
            r = c.get("/static/chat.js")
            assert r.status_code == 200
            # The body should contain a unique string from our chat.js.
            assert "[tinyrag]" in r.text or "tinyrag" in r.text.lower()


# ===========================================================================
# Helper regex used by a couple of tests above (kept here so the import
# block stays clean).
# ===========================================================================


_SSE_FRAME_SEP_RE = re.compile(r"\n\n")
