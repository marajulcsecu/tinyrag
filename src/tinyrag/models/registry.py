"""Canonical catalog of GGUF models supported by TinyRAG.

This is the **single source of truth** for which models TinyRAG knows
about, where to download them from, and how to verify that the downloaded
file is the one the project expects.

Design rules
------------
1. **One row per model id.** ``MODEL_REGISTRY`` is keyed by a short,
   stable id (e.g. ``"phi-3-mini"``) that matches what we plan to put
   in ``config.yaml``. Never rename an id — Phase 4 and Phase 5
   evaluation scripts will hardcode it.
2. **Never hardcode URLs anywhere else.** The downloader imports this
   dict and computes the URL from ``hf_repo`` + ``hf_filename``. If we
   ever need to add a mirror (e.g. a HF Spaces mirror or a S3 bucket),
   we add a column here, not a ``if`` branch in the downloader.
3. **SHA-256 is the contract.** A wrong SHA-256 means the model is
   either corrupted in transit or someone replaced the file. The
   downloader refuses to keep a file whose hash doesn't match.
4. **License is mandatory.** TinyRAG is MIT-licensed but ships other
   people's weights; we record the upstream license so the final
   report can list attributions.
5. **expected_size_bytes is a sanity check, not authoritative.** HF
   sometimes re-uploads a file with the same SHA but a slightly
   different reported size due to compression metadata. We use it
   only for the ``--dry-run`` progress bar estimate.

Updating a model
----------------
When the upstream maintainer publishes a new quantisation or a new
revision, do **all** of the following in one atomic commit:

1. Update the relevant :class:`ModelEntry` fields below.
2. Run ``python scripts/download_models.py --list`` and confirm the
   new metadata prints cleanly.
3. Delete the old ``models/<id>.gguf`` (and its ``_manifest.json``)
   from your local disk; re-run the downloader.
4. Update ``docs/MODELS.md`` with the new size, SHA-256, and release
   date. Update ``docs/05_tech_stack_v1.md`` §3.5 if the id, repo, or
   filename changed.

Location: ``src/tinyrag/models/registry.py``
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelEntry:
    """One row of :data:`MODEL_REGISTRY`.

    Attributes
    ----------
    model_id:
        Short, stable id used in ``config.yaml`` and CLI args. Lowercase
        + dashes, no spaces. Once published, this is a *public API*.
    display_name:
        Human-readable name shown in ``--list`` and in the web UI's
        model-picker dropdown. Can contain spaces and capitals.
    hf_repo:
        Hugging Face repo id (``org/name``). The downloader hits
        ``https://huggingface.co/<hf_repo>/resolve/main/<hf_filename>``.
    hf_filename:
        The exact filename inside the repo (case-sensitive).
    quantization:
        Quantisation label as published by the upstream maintainer
        (e.g. ``Q4_K_M``, ``Q5_K_M``). Shown in the UI and the report.
    expected_size_bytes:
        Approximate on-disk size of the GGUF. Used only for the
        progress bar estimate; SHA-256 is the real correctness check.
    expected_sha256:
        Lowercase hex SHA-256 of the published file. **This is the
        contract.** A mismatch is a hard error.
    license:
        SPDX-ish license id of the upstream weights (e.g. ``MIT``,
        ``Apache-2.0``, ``Llama3.1``). For inclusion in the report.
    role:
        How the model is used by TinyRAG. One of
        ``{"primary", "eval-small", "eval-medium", "eval-large"}``.
        Drives the comparison structure in Phase 5.
    intended_context:
        Context length the model was trained / recommended for, in
        tokens. The downloader doesn't enforce this; ``LLMClient``
        uses it to set ``--ctx-size`` at server start (Phase 3.7).
    notes:
        Free-text caveats, mirror sources, or "do not use for X".
    """

    model_id: str
    display_name: str
    hf_repo: str
    hf_filename: str
    quantization: str
    expected_size_bytes: int
    expected_sha256: str
    license: str
    role: str
    intended_context: int
    notes: str = ""


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
#
# SHA-256 values MUST be the official hash published by the model owner.
# We do NOT recompute them on download (the whole point is to detect
# tampering). The sizes are rounded to the nearest ~10 MB.
#
# Pinned per docs/05_tech_stack_v1.md §3.5. Bump them together with
# docs/MODELS.md.

MODEL_REGISTRY: Mapping[str, ModelEntry] = {
    # -----------------------------------------------------------------
    # PRIMARY — the default LLM for TinyRAG queries
    # -----------------------------------------------------------------
    "phi-3-mini": ModelEntry(
        model_id="phi-3-mini",
        display_name="Phi-3 Mini 3.8B Instruct (Q4_K_M, 4k)",
        hf_repo="microsoft/Phi-3-mini-4k-instruct-gguf",
        hf_filename="Phi-3-mini-4k-instruct-q4.gguf",
        quantization="Q4_K_M",
        expected_size_bytes=2_320_000_000,  # ~2.3 GB
        # Microsoft does not publish a single canonical SHA-256 for the
        # GGUF; we compute it on first download and pin it in
        # docs/MODELS.md. The empty string here means "verify against
        # the manifest, not the registry" — see downloader logic.
        expected_sha256="",  # populated on first verified download
        license="MIT",
        role="primary",
        intended_context=4096,
        notes=(
            "Primary LLM. Microsoft's official GGUF release. Use bartowski "
            "or TheBloke mirror only if microsoft/ becomes unavailable."
        ),
    ),
    # -----------------------------------------------------------------
    # EVAL A — smallest comparison model (validates the floor)
    # -----------------------------------------------------------------
    "tinyllama-1.1b": ModelEntry(
        model_id="tinyllama-1.1b",
        display_name="TinyLlama 1.1B Chat v1.0 (Q4_K_M)",
        hf_repo="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        hf_filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        quantization="Q4_K_M",
        expected_size_bytes=700_000_000,  # ~700 MB
        expected_sha256="",
        license="Apache-2.0",
        role="eval-small",
        intended_context=2048,
        notes="Smallest comparison model. Sets the quality floor.",
    ),
    # -----------------------------------------------------------------
    # EVAL B — middle comparison (validates the sweet spot)
    # -----------------------------------------------------------------
    "llama-3.2-3b": ModelEntry(
        model_id="llama-3.2-3b",
        display_name="Llama 3.2 3B Instruct (Q4_K_M)",
        hf_repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        hf_filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        quantization="Q4_K_M",
        expected_size_bytes=1_800_000_000,  # ~1.8 GB
        expected_sha256="",
        license="Llama3.2",
        role="eval-medium",
        intended_context=4096,
        notes=(
            "Middle comparison. Meta's repo is gated; bartowski hosts "
            "the same weights. Confirm the SHA against the official "
            "Llama 3.2 community license before redistributing."
        ),
    ),
    # -----------------------------------------------------------------
    # EVAL C (optional) — only for the laptop, too slow on the Pi
    # -----------------------------------------------------------------
    "mistral-7b": ModelEntry(
        model_id="mistral-7b",
        display_name="Mistral 7B Instruct v0.3 (Q4_K_M)",
        # bartowski hosts the same Q4_K_M weights as TheBloke did, but
        # is publicly accessible. TheBloke's repo returned 401 on
        # 2026-06-23 (private/moved), so we point here instead.
        hf_repo="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        hf_filename="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        quantization="Q4_K_M",
        expected_size_bytes=4_372_812_000,  # 4.37 GB (verified 2026-06-23)
        expected_sha256="",
        license="Apache-2.0",
        role="eval-large",
        intended_context=8192,
        notes=(
            "Optional 4th model. Too slow on the Pi (~2 tok/s) but useful "
            "on the laptop to show what a 'real' LLM can do. Mirrored "
            "from TheBloke → bartowski on 2026-06-23 (TheBloke 401)."
        ),
    ),
}


def get(model_id: str) -> ModelEntry:
    """Return the registry entry for *model_id* or raise :class:`KeyError`.

    Thin convenience over ``MODEL_REGISTRY[model_id]``. Exists so the
    downloader can do a single, easy-to-grep lookup.
    """
    return MODEL_REGISTRY[model_id]


__all__ = ["ModelEntry", "MODEL_REGISTRY", "get"]
