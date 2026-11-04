"""Offline & hybrid voice input adapter for AgriRAG (Phase 5).

Provides :class:`VoiceInputAdapter` for handling farmer voice queries
submitted to ``POST /api/voice``.

Design
------
1. **Browser Web Speech API (Primary Zero-RAM Path):**
   Modern mobile/desktop browsers transcribe Bengali speech (``bn-BD``)
   directly into text or pass a ``transcript_hint`` alongside audio.
2. **Local Offline Whisper / SpeechRecognition (Optional Backend Path):**
   When raw audio bytes (WAV/WebM/Ogg/MP3) are uploaded without a
   client-side transcript, :class:`VoiceInputAdapter` checks whether an
   offline STT engine (``faster_whisper`` or ``whisper``) is installed
   locally and transcribes the audio in Bengali (``bn``).
3. **Graceful Structured Fallback:**
   If raw audio is uploaded on an edge device where no local STT model
   is installed, the adapter validates the audio container and returns
   a clear Bengali advisory instructing the client to use the built-in
   Web Speech mic button or type the question.

Location: ``src/tinyrag/input_adapters/voice.py``
"""

from __future__ import annotations

import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_AUDIO_BYTES: int = 10 * 1024 * 1024  # 10 MB safety cap


@dataclass(frozen=True)
class VoiceTranscriptionResult:
    """Structured result returned by :meth:`VoiceInputAdapter.transcribe`."""

    ok: bool
    transcript: str
    engine: str
    language: str = "bn-BD"
    audio_bytes_len: int = 0
    duration_ms: float = 0.0
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "transcript": self.transcript,
            "engine": self.engine,
            "language": self.language,
            "audio_bytes_len": self.audio_bytes_len,
            "duration_ms": round(self.duration_ms, 2),
            "fallback_reason": self.fallback_reason,
        }


class VoiceInputError(ValueError):
    """Raised when uploaded voice input is empty or exceeds size limits."""


class VoiceInputAdapter:
    """Adapter that normalizes and transcribes Bengali farmer voice queries."""

    def __init__(
        self,
        *,
        default_language: str = "bn-BD",
        max_audio_bytes: int = MAX_AUDIO_BYTES,
        whisper_model_size: str = "base",
    ) -> None:
        self.default_language = default_language
        self.max_audio_bytes = max_audio_bytes
        self.whisper_model_size = whisper_model_size

    @staticmethod
    def detect_audio_format(data: bytes) -> str:
        """Best-effort magic-byte inspection for common browser audio formats."""
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return "wav"
        if len(data) >= 4 and data[:4] == b"OggS":
            return "ogg"
        if len(data) >= 4 and data[:4] == b"\x1a\x45\xdf\xa3":
            return "webm"
        if len(data) >= 3 and (data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3")):
            return "mp3"
        return "raw"

    def transcribe(
        self,
        audio_bytes: bytes | None = None,
        *,
        transcript_hint: str | None = None,
        language: str | None = None,
        filename: str = "voice.wav",
    ) -> VoiceTranscriptionResult:
        """Transcribe Bengali voice input or normalize a pre-transcribed hint.

        Parameters
        ----------
        audio_bytes:
            Raw audio payload (WAV/WebM/Ogg/MP3) uploaded by the client.
        transcript_hint:
            Optional client-side STT transcript (e.g., from Web Speech API).
        language:
            BCP-47 language tag (defaults to ``"bn-BD"``).
        filename:
            Original filename hint for temporary file suffix resolution.
        """
        t0 = time.monotonic()
        lang = (language or self.default_language).strip() or "bn-BD"
        raw_len = len(audio_bytes) if audio_bytes else 0

        if raw_len > self.max_audio_bytes:
            raise VoiceInputError(
                f"Audio payload too large ({raw_len} bytes > {self.max_audio_bytes} bytes limit)."
            )

        # Path 1: Client already provided a non-empty transcript hint
        if transcript_hint and transcript_hint.strip():
            normalized = unicodedata.normalize("NFC", transcript_hint.strip())
            return VoiceTranscriptionResult(
                ok=True,
                transcript=normalized,
                engine="web_speech_hint",
                language=lang,
                audio_bytes_len=raw_len,
                duration_ms=(time.monotonic() - t0) * 1000.0,
            )

        if not audio_bytes or raw_len == 0:
            raise VoiceInputError(
                "No audio bytes or transcript_hint provided to VoiceInputAdapter."
            )

        # Path 2: Try optional local faster-whisper / openai-whisper if installed
        suffix = Path(filename).suffix or f".{self.detect_audio_format(audio_bytes)}"
        whisper_lang = "bn" if lang.lower().startswith("bn") else lang.split("-")[0]

        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                model = WhisperModel(
                    self.whisper_model_size, device="cpu", compute_type="int8"
                )
                segments, _ = model.transcribe(tmp.name, language=whisper_lang)
                text = " ".join(seg.text.strip() for seg in segments if seg.text).strip()
            normalized = unicodedata.normalize("NFC", text)
            return VoiceTranscriptionResult(
                ok=bool(normalized),
                transcript=normalized,
                engine=f"faster_whisper:{self.whisper_model_size}",
                language=lang,
                audio_bytes_len=raw_len,
                duration_ms=(time.monotonic() - t0) * 1000.0,
                fallback_reason=None if normalized else "empty_transcription",
            )
        except ImportError:
            pass
        except Exception as exc:
            return VoiceTranscriptionResult(
                ok=False,
                transcript="",
                engine="faster_whisper_error",
                language=lang,
                audio_bytes_len=raw_len,
                duration_ms=(time.monotonic() - t0) * 1000.0,
                fallback_reason=f"Local STT error: {exc}",
            )

        # Path 3: Graceful structured fallback when offline Whisper is not installed
        fmt = self.detect_audio_format(audio_bytes)
        return VoiceTranscriptionResult(
            ok=False,
            transcript="",
            engine=f"offline_stub:{fmt}",
            language=lang,
            audio_bytes_len=raw_len,
            duration_ms=(time.monotonic() - t0) * 1000.0,
            fallback_reason=(
                "সার্ভারে অফলাইন Whisper STT মডেল ইনস্টল করা নেই। "
                "অনুগ্রহ করে ব্রাউজারের মাইক বাটন (🎤 Web Speech API) ব্যবহার করুন অথবা বাংলায় প্রশ্নটি টাইপ করুন।"
            ),
        )


__all__ = [
    "MAX_AUDIO_BYTES",
    "VoiceInputAdapter",
    "VoiceInputError",
    "VoiceTranscriptionResult",
]
