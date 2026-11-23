"""Offline & hybrid Bengali voice input adapter for AgriRAG (Phase 5 + Voice Agent v2).

Provides :class:`VoiceInputAdapter` and :func:`correct_bengali_agri_transcript`
for handling farmer voice queries submitted to ``POST /api/voice``.

4-Layer Bengali STT Pipeline
----------------------------
1. **Cloud Audio STT (Groq Whisper / Google Gemini Audio / OpenAI Whisper):**
   When ``audio_bytes`` + ``engine_mode="cloud"`` + ``api_key`` are provided,
   sends real recorded audio with Bangladeshi agricultural prompt biasing
   (:data:`BENGALI_AGRI_STT_PROMPT`) so rural accents and crop terms
   (``ব্লাস্ট``, ``কারেন্ট পোকা``, ``মাজরা পোকা``, ``আলুর মড়ক``) transcribe
   accurately in ~0.5--1.5 seconds.
2. **Local Offline ``faster-whisper`` (Cached Edge Path):**
   When ``faster_whisper`` is installed locally, reuses a cached ``WhisperModel``
   instance with ``vad_filter=True`` and :data:`BENGALI_AGRI_STT_PROMPT`.
3. **Browser Web Speech API Hint (Zero-RAM Fallback):**
   Normalizes and repairs pre-transcribed Bengali hints from Web Speech API
   (also used automatically if Cloud STT fails or times out).
4. **Bengali Agricultural Phonetic Post-Correction:**
   Every successful transcript passes through :func:`correct_bengali_agri_transcript`,
   which applies Unicode NFC normalization, Bengali-safe consecutive duplicate
   collapsing, and deterministic phonetic fixes for common Bengali STT mishearings.

Location: ``src/tinyrag/input_adapters/voice.py``
"""

from __future__ import annotations

import base64
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

MAX_AUDIO_BYTES: int = 10 * 1024 * 1024  # 10 MB safety cap
CLOUD_STT_TIMEOUT_SEC: float = 25.0

#: Domain-specific Bengali agricultural prompt passed to Whisper / Gemini Audio
#: to bias speech recognition toward Bangladeshi crop, pest, disease, and
#: agrochemical vocabulary instead of generic homophones.
BENGALI_AGRI_STT_PROMPT: str = (
    "বাংলাদেশের কৃষি প্রশ্ন: ধান, বোরো ধান, আমন ধান, ধানের ব্লাস্ট, শীষ ব্লাস্ট, "
    "পাতা ব্লাস্ট, কারেন্ট পোকা, বাদামী গাছফড়িং, মাজরা পোকা, মরা ডিগ, সাদা শীষ, "
    "পামরি পোকা, খোলপোড়া, ব্যাকটেরিয়াল লিফ ব্লাইট, আলুর মড়ক, লেট ব্লাইট, কাটুই পোকা, "
    "বেগুনের ডগা ও ফল ছিদ্রকারী পোকা, টমেটোর পাতা কোঁকড়ানো, সাদা মাছি, জাব পোকা, "
    "আমের হপার, অ্যানথ্রাকনোজ, পেঁয়াজের পার্পল ব্লচ, থ্রিপস, সেক্স ফেরোমন ফাঁদ, "
    "ছত্রাকনাশক, কীটনাশক, বালাইনাশক, ট্রাইসাইক্লাজোল, পাইমেট্রোজিন, ম্যাংকোজেব, "
    "কার্টাপ, ইমিডাক্লোপ্রিড, ইউরিয়া, পটাশ, টিএসপি, তাপমাত্রা, আর্দ্রতা, মাটির রস, "
    "ব্রি (BRRI), বারি (BARI), কৃষি সম্প্রসারণ অধিদপ্তর (DAE)।"
)

#: Deterministic phonetic repairs for high-frequency Bengali STT mishearings
#: observed on Android Chrome Web Speech API and noisy outdoor recordings.
#: Ordered from longer multi-word phrases to shorter terms to avoid partial clobbering.
_AGRI_PHONETIC_FIXES: tuple[tuple[str, str], ...] = (
    # Rice Blast mishearings
    ("ধানের ব্লাউজ", "ধানের ব্লাস্ট"),
    ("ধানের প্লাস্টিক", "ধানের ব্লাস্ট"),
    ("ধানের প্লাস্ট", "ধানের ব্লাস্ট"),
    ("ধানের ফ্লাস্ট", "ধানের ব্লাস্ট"),
    ("ধানের ব্লাস্টিক", "ধানের ব্লাস্ট"),
    ("ব্লাউজ রোগ", "ব্লাস্ট রোগ"),
    ("প্লাস্ট রোগ", "ব্লাস্ট রোগ"),
    ("ফ্লাস্ট রোগ", "ব্লাস্ট রোগ"),
    ("ব্লাস্টিক রোগ", "ব্লাস্ট রোগ"),
    ("শিস ব্লাস্ট", "শীষ ব্লাস্ট"),
    ("শিশ ব্লাস্ট", "শীষ ব্লাস্ট"),
    ("শিস পচা", "শীষ পচা"),
    ("শিশ পচা", "শীষ পচা"),
    ("সাদা শিস", "সাদা শীষ"),
    ("সাদা শিশ", "সাদা শীষ"),
    # Rice Brown Planthopper (Current Poka)
    ("কারেন্ট পাকা", "কারেন্ট পোকা"),
    ("কারেন্ট পুকা", "কারেন্ট পোকা"),
    ("ক্যারেন্ট পোকা", "কারেন্ট পোকা"),
    ("ক্যারাম পোকা", "কারেন্ট পোকা"),
    ("বাদামি গাছ ফড়িং", "বাদামী গাছফড়িং"),
    ("বাদামী গাছ ফড়িং", "বাদামী গাছফড়িং"),
    # Stem Borer (Majra Poka)
    ("মাঝরা পোকা", "মাজরা পোকা"),
    ("মাছরা পোকা", "মাজরা পোকা"),
    ("মজরা পোকা", "মাজরা পোকা"),
    ("মাজরা পুকা", "মাজরা পোকা"),
    ("মরা দিক", "মরা ডিগ"),
    # Potato Late Blight (Alur Morok) & Cutworm
    ("আলুর মোড়ক", "আলুর মড়ক"),
    ("আলোর মড়ক", "আলুর মড়ক"),
    ("আলুর মরক", "আলুর মড়ক"),
    ("আলু মড়ক", "আলুর মড়ক"),
    ("লেট লাইট", "লেট ব্লাইট"),
    ("কটুই পোকা", "কাটুই পোকা"),
    ("কাটই পোকা", "কাটুই পোকা"),
    ("কাটুই পুকা", "কাটুই পোকা"),
    # Pamri / Leaf Curl / Eggplant / Fruit Fly
    ("পামরি পুকা", "পামরি পোকা"),
    ("পামড়ি পোকা", "পামরি পোকা"),
    ("পাতা কোকরানো", "পাতা কোঁকড়ানো"),
    ("পাতা কুকড়ানো", "পাতা কোঁকড়ানো"),
    ("পাতা কোঁকরানো", "পাতা কোঁকড়ানো"),
    ("পাতা কুকরানো", "পাতা কোঁকড়ানো"),
    ("মাছি পুকা", "মাছি পোকা"),
    ("জাব পুকা", "জাব পোকা"),
    ("সাদা মাছ", "সাদা মাছি"),
    # Crop / Season / Agrochemical spelling normalization
    ("বোরো দান", "বোরো ধান"),
    ("আমন দান", "আমন ধান"),
    ("আউশ দান", "আউশ ধান"),
    ("ধান গাছে পুকা", "ধান গাছে পোকা"),
    ("ছত্রাক নাশক", "ছত্রাকনাশক"),
    ("কিটনাশক", "কীটনাশক"),
    ("কীট নাশক", "কীটনাশক"),
    ("বালাই নাশক", "বালাইনাশক"),
    ("আদ্রতা", "আর্দ্রতা"),
    ("আদ্রর্তা", "আর্দ্রতা"),
    ("তাপ মাত্রা", "তাপমাত্রা"),
    ("ট্রাইসাইক্লাজল", "ট্রাইসাইক্লাজোল"),
    ("পাইমেট্রজিন", "পাইমেট্রোজিন"),
    ("ম্যাংকজেব", "ম্যাংকোজেব"),
    ("ম্যানকোজেব", "ম্যাংকোজেব"),
)

#: Module-level cache for local ``faster-whisper`` models so weights are not
#: reloaded from disk on every request.
_WHISPER_MODEL_CACHE: dict[str, Any] = {}


def _collapse_unicode_duplicates(text: str) -> str:
    """Collapse consecutive duplicate words and repeated 2-3 word phrases.

    JavaScript/Python ASCII ``\\b`` word boundaries fail on Bengali Unicode
    combining marks (matras). Splitting on whitespace handles Bengali cleanly.
    """
    tokens = text.split()
    if len(tokens) < 2:
        return " ".join(tokens)

    # Pass 1: single-token adjacent duplicates ("ধানের ধানের" -> "ধানের")
    deduped: list[str] = []
    for tok in tokens:
        if not deduped or deduped[-1] != tok:
            deduped.append(tok)

    # Pass 2: adjacent repeated 2-word and 3-word phrases (Android Chrome stutter)
    for n in (3, 2):
        i = 0
        out: list[str] = []
        while i < len(deduped):
            if i + 2 * n <= len(deduped) and deduped[i : i + n] == deduped[i + n : i + 2 * n]:
                out.extend(deduped[i : i + n])
                i += 2 * n
                # Skip any additional immediate repeats of the same n-gram
                while i + n <= len(deduped) and deduped[i : i + n] == out[-n:]:
                    i += n
            else:
                out.append(deduped[i])
                i += 1
        deduped = out

    return " ".join(deduped)


def correct_bengali_agri_transcript(raw_text: str) -> tuple[str, list[tuple[str, str]]]:
    """Normalize and repair common Bengali agricultural STT mishearings.

    Returns ``(cleaned_text, applied_replacements)`` where each entry in
    ``applied_replacements`` is ``(wrong_phrase, fixed_phrase)``.
    """
    normalized = unicodedata.normalize("NFC", (raw_text or "").strip())
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        return "", []

    cleaned = _collapse_unicode_duplicates(normalized)
    applied: list[tuple[str, str]] = []

    for wrong, right in _AGRI_PHONETIC_FIXES:
        wrong_nfc = unicodedata.normalize("NFC", wrong)
        right_nfc = unicodedata.normalize("NFC", right)
        if wrong_nfc in cleaned:
            cleaned = cleaned.replace(wrong_nfc, right_nfc)
            applied.append((wrong_nfc, right_nfc))

    # Collapse any duplicates introduced by replacements and normalize spacing
    cleaned = _collapse_unicode_duplicates(re.sub(r"\s+", " ", cleaned).strip())
    return cleaned, applied


def _mime_and_ext_for_audio(data: bytes, audio_mime_type: str | None, filename: str) -> tuple[str, str]:
    """Resolve safe MIME type and file extension for cloud/local STT uploads."""
    mime = (audio_mime_type or "").split(";")[0].strip().lower()
    if mime in ("audio/webm", "video/webm"):
        return "audio/webm", ".webm"
    if mime in ("audio/ogg", "application/ogg"):
        return "audio/ogg", ".ogg"
    if mime in ("audio/mp4", "audio/m4a", "audio/aac", "video/mp4"):
        return "audio/mp4", ".m4a"
    if mime in ("audio/mpeg", "audio/mp3"):
        return "audio/mpeg", ".mp3"
    if mime in ("audio/wav", "audio/x-wav", "audio/wave"):
        return "audio/wav", ".wav"

    detected = VoiceInputAdapter.detect_audio_format(data)
    ext_map = {
        "wav": ("audio/wav", ".wav"),
        "ogg": ("audio/ogg", ".ogg"),
        "webm": ("audio/webm", ".webm"),
        "mp3": ("audio/mpeg", ".mp3"),
        "mp4": ("audio/mp4", ".m4a"),
    }
    if detected in ext_map:
        return ext_map[detected]

    suffix = Path(filename).suffix.lower() or ".webm"
    return "audio/webm", suffix


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
    raw_transcript: str = ""
    corrected: bool = False
    corrections: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "transcript": self.transcript,
            "raw_transcript": self.raw_transcript or self.transcript,
            "corrected": self.corrected,
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
        cloud_timeout_sec: float = CLOUD_STT_TIMEOUT_SEC,
    ) -> None:
        self.default_language = default_language
        self.max_audio_bytes = max_audio_bytes
        self.whisper_model_size = whisper_model_size
        self.cloud_timeout_sec = cloud_timeout_sec

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
        if len(data) >= 12 and data[4:8] == b"ftyp":
            return "mp4"
        return "raw"

    def _transcribe_cloud_audio(
        self,
        audio_bytes: bytes,
        *,
        mime_type: str,
        ext: str,
        whisper_lang: str,
        api_key: str,
        api_base_url: str | None,
        api_model: str | None,
    ) -> tuple[str, str]:
        """Transcribe audio via Groq Whisper, Google Gemini Audio, or OpenAI Whisper.

        Returns ``(raw_transcript, engine_label)`` or raises an exception on failure.
        """
        key = api_key.strip()
        base = (api_base_url or "").strip().rstrip("/")
        base_lower = base.lower()

        is_gemini = "generativelanguage.googleapis.com" in base_lower or (
            not base_lower and key.startswith("AIza")
        )
        is_groq = "groq.com" in base_lower or (not base_lower and key.startswith("gsk_"))

        if is_gemini:
            return self._transcribe_gemini_audio(
                audio_bytes,
                mime_type=mime_type,
                api_key=key,
                api_model=api_model,
            )

        if is_groq:
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            stt_model = "whisper-large-v3-turbo"
            engine_label = f"cloud_groq:{stt_model}"
        else:
            root = base or "https://api.openai.com/v1"
            url = f"{root}/audio/transcriptions"
            stt_model = "whisper-1"
            engine_label = f"cloud_whisper:{stt_model}"

        files = {
            "file": (f"voice{ext}", audio_bytes, mime_type),
        }
        data = {
            "model": stt_model,
            "language": whisper_lang,
            "prompt": BENGALI_AGRI_STT_PROMPT,
            "temperature": "0",
            "response_format": "json",
        }
        headers = {"Authorization": f"Bearer {key}"}

        with httpx.Client(timeout=self.cloud_timeout_sec) as client:
            resp = client.post(url, headers=headers, data=data, files=files)
            resp.raise_for_status()
            payload = resp.json()

        text = str(payload.get("text") or "").strip()
        return text, engine_label

    def _transcribe_gemini_audio(
        self,
        audio_bytes: bytes,
        *,
        mime_type: str,
        api_key: str,
        api_model: str | None,
    ) -> tuple[str, str]:
        """Transcribe Bengali farmer audio using Gemini native multimodal audio input."""
        model_id = (api_model or "gemini-2.5-flash").strip()
        if not model_id.startswith("gemini"):
            model_id = "gemini-2.5-flash"

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_id}:generateContent?key={api_key}"
        )
        b64_audio = base64.b64encode(audio_bytes).decode("ascii")
        instruction = (
            "You are a Bangladeshi Bengali speech-to-text transcriber for farmers. "
            "Transcribe the spoken audio accurately into clean Bengali Unicode script (bn-BD). "
            "Pay special attention to Bangladeshi agricultural terms such as: "
            f"{BENGALI_AGRI_STT_PROMPT} "
            "Return ONLY the exact transcribed Bengali question or sentence. "
            "Do not answer the question, do not add quotes, labels, or explanations. "
            "If the audio is silent or unintelligible noise, return an empty string."
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": instruction},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_audio,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 256,
            },
        }
        with httpx.Client(timeout=self.cloud_timeout_sec) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            body = resp.json()

        candidates = body.get("candidates") or []
        text_parts: list[str] = []
        if candidates:
            parts = ((candidates[0].get("content") or {}).get("parts")) or []
            for part in parts:
                if isinstance(part, dict) and part.get("text"):
                    text_parts.append(str(part["text"]).strip())

        transcript = " ".join(text_parts).strip().strip('"').strip("'")
        return transcript, f"cloud_gemini:{model_id}"

    def _get_cached_whisper_model(self) -> Any:
        """Return a cached ``faster_whisper.WhisperModel`` instance."""
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]

        cached = _WHISPER_MODEL_CACHE.get(self.whisper_model_size)
        if cached is None:
            cached = WhisperModel(
                self.whisper_model_size, device="cpu", compute_type="int8"
            )
            _WHISPER_MODEL_CACHE[self.whisper_model_size] = cached
        return cached

    def transcribe(
        self,
        audio_bytes: bytes | None = None,
        *,
        transcript_hint: str | None = None,
        language: str | None = None,
        filename: str = "voice.wav",
        audio_mime_type: str | None = None,
        engine_mode: str | None = None,
        api_key: str | None = None,
        api_base_url: str | None = None,
        api_model: str | None = None,
    ) -> VoiceTranscriptionResult:
        """Transcribe Bengali voice input or normalize a pre-transcribed hint.

        Priority
        --------
        1. If ``audio_bytes`` is present AND ``engine_mode == "cloud"`` AND ``api_key``
           is provided: attempt Cloud Audio STT (Groq Whisper / Gemini Audio / OpenAI).
           If that fails, fall through gracefully to local Whisper or ``transcript_hint``.
        2. If ``transcript_hint`` is provided without active Cloud Audio STT:
           normalize and post-correct the Web Speech API transcript immediately.
        3. If ``audio_bytes`` is present: try local cached ``faster-whisper`` with
           Bengali agricultural ``initial_prompt`` + VAD filtering.
        4. If ``transcript_hint`` was provided as backup: return post-corrected hint.
        5. Otherwise return a structured Bengali fallback message.
        """
        t0 = time.monotonic()
        lang = (language or self.default_language).strip() or "bn-BD"
        raw_len = len(audio_bytes) if audio_bytes else 0
        whisper_lang = "bn" if lang.lower().startswith("bn") else lang.split("-")[0]

        if raw_len > self.max_audio_bytes:
            raise VoiceInputError(
                f"Audio payload too large ({raw_len} bytes > {self.max_audio_bytes} bytes limit)."
            )

        cloud_error: str | None = None
        want_cloud_audio = (
            bool(audio_bytes)
            and raw_len > 0
            and (engine_mode or "").strip().lower() == "cloud"
            and bool(api_key and api_key.strip())
        )

        # Path 1: Cloud Audio STT (Groq Whisper / Gemini Audio / OpenAI Whisper)
        if want_cloud_audio and audio_bytes is not None and api_key is not None:
            mime_type, ext = _mime_and_ext_for_audio(audio_bytes, audio_mime_type, filename)
            try:
                raw_cloud, engine_name = self._transcribe_cloud_audio(
                    audio_bytes,
                    mime_type=mime_type,
                    ext=ext,
                    whisper_lang=whisper_lang,
                    api_key=api_key,
                    api_base_url=api_base_url,
                    api_model=api_model,
                )
                raw_cloud_nfc = unicodedata.normalize("NFC", raw_cloud.strip())
                cleaned, applied = correct_bengali_agri_transcript(raw_cloud)
                if cleaned:
                    return VoiceTranscriptionResult(
                        ok=True,
                        transcript=cleaned,
                        raw_transcript=raw_cloud,
                        corrected=bool(applied) or (cleaned != raw_cloud_nfc),
                        corrections=tuple(applied),
                        engine=engine_name,
                        language=lang,
                        audio_bytes_len=raw_len,
                        duration_ms=(time.monotonic() - t0) * 1000.0,
                    )
                cloud_error = "cloud_stt_empty_transcription"
            except Exception as exc:
                cloud_error = f"Cloud STT error: {exc}"

        # Path 2: Direct Web Speech hint (when not trying Cloud Audio, or if Cloud STT failed)
        if transcript_hint and transcript_hint.strip() and not (want_cloud_audio and not cloud_error):
            raw_hint = unicodedata.normalize("NFC", transcript_hint.strip())
            cleaned, applied = correct_bengali_agri_transcript(raw_hint)
            return VoiceTranscriptionResult(
                ok=bool(cleaned),
                transcript=cleaned,
                raw_transcript=raw_hint,
                corrected=bool(applied) or (cleaned != raw_hint),
                corrections=tuple(applied),
                engine="web_speech_hint",
                language=lang,
                audio_bytes_len=raw_len,
                duration_ms=(time.monotonic() - t0) * 1000.0,
                fallback_reason=cloud_error,
            )

        if not audio_bytes or raw_len == 0:
            raise VoiceInputError(
                "No audio bytes or transcript_hint provided to VoiceInputAdapter."
            )

        # Path 3: Optional local cached faster-whisper with agricultural prompt biasing
        _, suffix = _mime_and_ext_for_audio(audio_bytes, audio_mime_type, filename)
        try:
            model = self._get_cached_whisper_model()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                segments, _ = model.transcribe(
                    tmp.name,
                    language=whisper_lang,
                    vad_filter=True,
                    initial_prompt=BENGALI_AGRI_STT_PROMPT,
                )
                raw_local = " ".join(
                    seg.text.strip() for seg in segments if seg.text
                ).strip()
            raw_local_nfc = unicodedata.normalize("NFC", raw_local.strip())
            cleaned, applied = correct_bengali_agri_transcript(raw_local)
            return VoiceTranscriptionResult(
                ok=bool(cleaned),
                transcript=cleaned,
                raw_transcript=raw_local,
                corrected=bool(applied) or (cleaned != raw_local_nfc),
                corrections=tuple(applied),
                engine=f"faster_whisper:{self.whisper_model_size}",
                language=lang,
                audio_bytes_len=raw_len,
                duration_ms=(time.monotonic() - t0) * 1000.0,
                fallback_reason=None if cleaned else "empty_transcription",
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

        # Path 4: Graceful structured fallback when neither Cloud STT nor Whisper succeeded
        fmt = self.detect_audio_format(audio_bytes)
        reason = (
            f"{cloud_error} — " if cloud_error else ""
        ) + (
            "সার্ভারে অফলাইন Whisper STT মডেল ইনস্টল করা নেই। "
            "অনুগ্রহ করে ব্রাউজারের মাইক বাটন (🎤 Web Speech API) ব্যবহার করুন অথবা বাংলায় প্রশ্নটি টাইপ করুন।"
        )
        return VoiceTranscriptionResult(
            ok=False,
            transcript="",
            engine=f"offline_stub:{fmt}",
            language=lang,
            audio_bytes_len=raw_len,
            duration_ms=(time.monotonic() - t0) * 1000.0,
            fallback_reason=reason,
        )


__all__ = [
    "BENGALI_AGRI_STT_PROMPT",
    "MAX_AUDIO_BYTES",
    "VoiceInputAdapter",
    "VoiceInputError",
    "VoiceTranscriptionResult",
    "correct_bengali_agri_transcript",
]
