"""Unit and integration tests for Voice Agent v2 & 4-Layer Bengali STT pipeline.

Covers:
1. Bengali agricultural phonetic post-correction & Unicode-safe duplicate collapsing.
2. Cloud Audio STT routing for Groq Whisper, Google Gemini Audio, and OpenAI Whisper
   (with mocked httpx responses + graceful fallback to web_speech_hint on errors).
3. ``POST /api/voice`` integration with ``audio_mime_type`` and ``run_rag=False``.
4. Voice Agent overlay DOM markup, CSS classes, and JS syntax validity.
"""

# ruff: noqa: I001
from __future__ import annotations

import base64
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient  # noqa: E402

from tinyrag.input_adapters.voice import (  # noqa: E402
    BENGALI_AGRI_STT_PROMPT,
    VoiceInputAdapter,
    correct_bengali_agri_transcript,
)
from tinyrag.main import create_app  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOICE_AGENT_JS = PROJECT_ROOT / "ui" / "static" / "voice_agent.js"
CHAT_JS = PROJECT_ROOT / "ui" / "static" / "chat.js"
INDEX_HTML = PROJECT_ROOT / "ui" / "templates" / "index.html"
STYLE_CSS = PROJECT_ROOT / "ui" / "static" / "style.css"


def _make_settings(tmp_path: Path) -> Any:
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
        chunking={"chunk_size": 400, "chunk_overlap": 50, "encoding": "cl100k_base"},
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
    app = create_app(_make_settings(tmp_path), llm_kind="fake", embedder_kind="fake")
    with TestClient(app) as c:
        yield c


# ===========================================================================
# 1. Bengali Agricultural Phonetic Post-Corrector & Unicode Dedup
# ===========================================================================


class TestBengaliAgriPostCorrector:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ধানের ব্লাউজ রোগে কী স্প্রে করব?", "ধানের ব্লাস্ট রোগে কী স্প্রে করব?"),
            ("জমিতে কারেন্ট পাকা লেগেছে", "জমিতে কারেন্ট পোকা লেগেছে"),
            ("আলুর মোড়ক দমনে কোন ছত্রাক নাশক দেব?", "আলুর মড়ক দমনে কোন ছত্রাকনাশক দেব?"),
            ("ধানের মাঝরা পোকা ও সাদা শিস দেখা যাচ্ছে", "ধানের মাজরা পোকা ও সাদা শীষ দেখা যাচ্ছে"),
            ("টমেটোর পাতা কোকরানো ও সাদা মাছ", "টমেটোর পাতা কোঁকড়ানো ও সাদা মাছি"),
            ("আজকের বাতাসের আদ্রতা কত?", "আজকের বাতাসের আর্দ্রতা কত?"),
        ],
    )
    def test_phonetic_agricultural_repairs(self, raw: str, expected: str) -> None:
        cleaned, applied = correct_bengali_agri_transcript(raw)
        assert cleaned == unicodedata.normalize("NFC", expected)
        assert len(applied) >= 1

    def test_collapses_android_chrome_bengali_stutter(self) -> None:
        raw = "ধানের ধানের ব্লাস্ট রোগ আলুর মড়ক আলুর মড়ক কী করব"
        cleaned, _ = correct_bengali_agri_transcript(raw)
        assert cleaned == unicodedata.normalize("NFC", "ধানের ব্লাস্ট রোগ আলুর মড়ক কী করব")

    def test_clean_query_remains_unchanged(self) -> None:
        raw = "ধানের ব্লাস্ট রোগে কী ওষুধ দেব?"
        cleaned, applied = correct_bengali_agri_transcript(raw)
        assert cleaned == raw
        assert applied == []


# ===========================================================================
# 2. Cloud Audio STT Routing & Graceful Fallback
# ===========================================================================


class TestCloudAudioStt:
    def test_groq_whisper_stt_with_agri_prompt_and_correction(self) -> None:
        fake_webm = b"\x1a\x45\xdf\xa3" + b"\x00" * 100
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "ধানের ব্লাউজ রোগে কী করব?"}
        mock_resp.raise_for_status.return_value = None

        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_resp

        adapter = VoiceInputAdapter()
        with patch("tinyrag.input_adapters.voice.httpx.Client", return_value=mock_client):
            res = adapter.transcribe(
                fake_webm,
                audio_mime_type="audio/webm;codecs=opus",
                engine_mode="cloud",
                api_key="gsk_test_key",
                api_base_url="https://api.groq.com/openai/v1",
            )

        assert res.ok is True
        assert res.engine == "cloud_groq:whisper-large-v3-turbo"
        assert res.transcript == "ধানের ব্লাস্ট রোগে কী করব?"
        assert res.corrected is True
        call_kwargs = mock_client.__enter__.return_value.post.call_args.kwargs
        assert call_kwargs["data"]["language"] == "bn"
        assert call_kwargs["data"]["prompt"] == BENGALI_AGRI_STT_PROMPT

    def test_gemini_native_audio_stt(self) -> None:
        fake_webm = b"\x1a\x45\xdf\xa3" + b"\x00" * 100
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": "আলুর মোড়ক দমনের উপায় কী?"}]}}
            ]
        }
        mock_resp.raise_for_status.return_value = None

        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_resp

        adapter = VoiceInputAdapter()
        with patch("tinyrag.input_adapters.voice.httpx.Client", return_value=mock_client):
            res = adapter.transcribe(
                fake_webm,
                audio_mime_type="audio/webm",
                engine_mode="cloud",
                api_key="AIzaSyTestKey",
                api_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                api_model="gemini-2.5-flash",
            )

        assert res.ok is True
        assert res.engine == "cloud_gemini:gemini-2.5-flash"
        assert res.transcript == unicodedata.normalize("NFC", "আলুর মড়ক দমনের উপায় কী?")
        assert res.corrected is True

    def test_cloud_stt_failure_falls_back_to_transcript_hint(self) -> None:
        fake_webm = b"\x1a\x45\xdf\xa3" + b"\x00" * 100
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.side_effect = RuntimeError("network down")

        adapter = VoiceInputAdapter()
        with patch("tinyrag.input_adapters.voice.httpx.Client", return_value=mock_client):
            res = adapter.transcribe(
                fake_webm,
                transcript_hint="কারেন্ট পাকা দমনে কী স্প্রে করব?",
                engine_mode="cloud",
                api_key="gsk_test_key",
                api_base_url="https://api.groq.com/openai/v1",
            )

        assert res.ok is True
        assert res.engine == "web_speech_hint"
        assert res.transcript == "কারেন্ট পোকা দমনে কী স্প্রে করব?"
        assert res.corrected is True
        assert res.fallback_reason and "network down" in res.fallback_reason


# ===========================================================================
# 3. POST /api/voice API Integration + Voice Agent UI Assets
# ===========================================================================


class TestVoiceAgentIntegrationAndUi:
    def test_post_voice_applies_phonetic_correction_on_hint(self, client: TestClient) -> None:
        resp = client.post(
            "/api/voice",
            json={
                "transcript_hint": "ধানের ধানের ব্লাউজ রোগে কী করব?",
                "language": "bn-BD",
                "run_rag": False,
            },
        )
        assert resp.status_code == 200
        tr = resp.json()["transcription"]
        assert tr["ok"] is True
        assert tr["transcript"] == "ধানের ব্লাস্ট রোগে কী করব?"
        assert tr["corrected"] is True

    def test_post_voice_accepts_audio_mime_type(self, client: TestClient) -> None:
        fake_webm = b"\x1a\x45\xdf\xa3" + b"\x00" * 64
        resp = client.post(
            "/api/voice",
            json={
                "audio_base64": base64.b64encode(fake_webm).decode("ascii"),
                "audio_mime_type": "audio/webm;codecs=opus",
                "language": "bn-BD",
                "run_rag": False,
            },
        )
        assert resp.status_code == 200
        tr = resp.json()["transcription"]
        assert tr["audio_bytes_len"] == len(fake_webm)

    def test_voice_agent_overlay_present_in_html_and_assets_served(
        self, client: TestClient
    ) -> None:
        html = client.get("/").text
        assert 'id="voice-agent-overlay"' in html
        assert 'id="va-waveform"' in html
        assert 'id="va-transcript-input"' in html
        assert 'id="va-primary-btn"' in html
        assert 'id="va-stop-btn"' in html
        assert "/static/voice_agent.js" in html
        assert "/static/voice_agent.css" in html
        assert "বলুন" in html
        assert "STOP" in html

        js_resp = client.get("/static/voice_agent.js")
        assert js_resp.status_code == 200
        assert "correctBengaliAgriTranscript" in js_resp.text

        css_resp = client.get("/static/style.css")
        assert ".va-overlay" in css_resp.text
        assert ".va-circle-stop" in css_resp.text

    def test_zero_host_auto_redirects_to_localhost(self, client: TestClient) -> None:
        resp = client.get("/?v=22", headers={"host": "0.0.0.0:8000"}, follow_redirects=False)
        assert resp.status_code == 307
        assert resp.headers["location"] == "http://localhost:8000/?v=22"

    def test_js_files_pass_node_syntax_check(self) -> None:
        node = shutil.which("node")
        if not node:
            pytest.skip("node not installed")
        for js_file in (CHAT_JS, VOICE_AGENT_JS):
            proc = subprocess.run(
                [node, "--check", str(js_file)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert proc.returncode == 0, f"Syntax error in {js_file.name}: {proc.stderr}"
