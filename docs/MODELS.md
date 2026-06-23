# TinyRAG — Model Catalog

**Project Title:** TinyRAG — A Lightweight, On-Device Retrieval-Augmented Generation Assistant for Smart Home IoT
**Document version:** 1.0
**Date:** 2026-06-23
**Status:** Active
**Companion to:** `docs/05_tech_stack_v1.md` §3.5, `docs/BUILDS.md` §2.1

---

## 0. Purpose

This document is the **human-readable catalog** of every GGUF model TinyRAG can use. For each model it records:

- the on-disk filename and where it should live,
- the upstream Hugging Face source URL,
- the size and (once known) the canonical SHA-256 hash,
- the license under which the weights are distributed,
- the role the model plays in TinyRAG (primary, eval-small, etc.).

The machine-readable version of this catalog lives in
`src/tinyrag/models/registry.py` as `MODEL_REGISTRY`. This file is the
**single place** where pinned SHA-256 values, final filenames, and
"verified-on" dates are recorded. If you bump a model version, update
**both** files in the same commit.

> **SHA-256 policy.** Microsoft (Phi-3), Meta (Llama 3.2), and Mistral
> do not all publish a single canonical SHA-256 for their GGUF
> releases. The first time we download a model, the downloader records
> the actual hash in `<models_dir>/_manifest.json`. We then copy that
> hash into the table below and (optionally) into the registry. The
> manifest is the runtime source of truth; the table below is the
> human-readable mirror.

---

## 1. Model Roster

| ID | Display name | Quant | Approx. size | HF repo | License | Role | Pinned? |
|----|--------------|-------|--------------|---------|---------|------|---------|
| `phi-3-mini` | Phi-3 Mini 3.8B Instruct (4k) | Q4_K_M | ~2.3 GB | `microsoft/Phi-3-mini-4k-instruct-gguf` | MIT | **primary** | ✅ |
| `tinyllama-1.1b` | TinyLlama 1.1B Chat v1.0 | Q4_K_M | ~700 MB | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` | Apache-2.0 | eval-small | ✅ |
| `llama-3.2-3b` | Llama 3.2 3B Instruct | Q4_K_M | ~1.8 GB | `bartowski/Llama-3.2-3B-Instruct-GGUF` | Llama3.2 (community) | eval-medium | ✅ |
| `mistral-7b` | Mistral 7B Instruct v0.3 | Q4_K_M | ~4.0 GB | `TheBloke/Mistral-7B-Instruct-v0.3-GGUF` | Apache-2.0 | eval-large | ⚠️ optional (laptop only) |

### 1.1 Per-model details

#### `phi-3-mini` — Primary

- **On disk:** `models/phi-3-mini.gguf`
- **HF URL:** <https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf>
- **Filename in repo:** `Phi-3-mini-4k-instruct-q4.gguf`
- **Quantisation:** Q4_K_M (4-bit, k-quant, medium quality)
- **Context length:** 4096 tokens
- **SHA-256:** _to be filled on first successful download_ (see `<models_dir>/_manifest.json`)
- **Verified on:** 2026-06-23 (Step 3.5)
- **Why this one:** best quality-per-MB in the ≤4B class; MIT-licensed; ships in a single 2.3 GB file that fits on the Pi 5.

#### `tinyllama-1.1b` — Smallest comparison

- **On disk:** `models/tinyllama-1.1b.gguf`
- **HF URL:** <https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf>
- **Filename in repo:** `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
- **Context length:** 2048 tokens
- **Why this one:** sets the "quality floor" in the Phase 5 evaluation. If TinyLlama answers a question correctly, the bigger models should too — and if they don't, we have a real finding to report.

#### `llama-3.2-3b` — Middle comparison

- **On disk:** `models/llama-3.2-3b.gguf`
- **HF URL:** <https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf>
- **Filename in repo:** `Llama-3.2-3B-Instruct-Q4_K_M.gguf`
- **Context length:** 4096 tokens
- **Why this one:** Meta's official repo is gated; bartowski's mirror hosts the same weights under a permissive re-upload. We are using it for **research evaluation**, not redistribution.
- **License caveat:** Llama 3.2 community license. The final report must include the attribution block from <https://llama.meta.com/>.

#### `mistral-7b` — Optional 4th comparison

- **On disk:** `models/mistral-7b.gguf`
- **HF URL:** <https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/mistral-7b-instruct-v0.3.Q4_K_M.gguf>
- **Filename in repo:** `mistral-7b-instruct-v0.3.Q4_K_M.gguf`
- **Context length:** 8192 tokens
- **Why this one:** shows what a "real" LLM (7B) can do. Will not be benchmarked on the Pi 5 (~2 tok/s with Q4_K_M is too slow for the demo); evaluated only on the laptop.

---

## 2. The Manifest (`models/_manifest.json`)

After every download, the script writes a single JSON file at
`<models_dir>/_manifest.json` keyed by model id:

```json
{
  "phi-3-mini": {
    "model_id": "phi-3-mini",
    "display_name": "Phi-3 Mini 3.8B Instruct (Q4_K_M, 4k)",
    "hf_repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
    "hf_filename": "Phi-3-mini-4k-instruct-q4.gguf",
    "quantization": "Q4_K_M",
    "license": "MIT",
    "role": "primary",
    "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
    "path": "models/phi-3-mini.gguf",
    "size_bytes": 2320000000,
    "sha256": "a1b2c3d4…",
    "downloaded_at_utc": "2026-06-23T18:00:00+00:00",
    "tinyRag_version": "0.1.0"
  }
}
```

The manifest is the **runtime source of truth** for "is this file
genuine?" The registry in `src/tinyrag/models/registry.py` is the
catalog, the manifest is the audit log.

### 2.1 Why both?

- The **registry** is bundled with the code and changes rarely (only
  when we add a new model). It tells the *code* which model is
  "primary" and what URL to hit.
- The **manifest** is per-machine, written on first download. It
  records the *actual* hash we got. If the HF file ever changes
  upstream, the next download will fail with `ChecksumMismatchError`
  and we'll know to investigate.

---

## 3. Verification Workflow

After a download (or after a fresh clone), you can re-verify every
model on disk with no network access:

```bash
python scripts/download_models.py --verify-only --model phi-3-mini
python scripts/download_models.py --verify-only --all   # all 4 models
```

The exit code is 0 if every model is present and matches its SHA-256,
1 if any is missing or corrupt. The CI pipeline in Phase 5 will
include this in the pre-merge gate.

### 3.1 What "verified" means

The file at `models/<id>.gguf` exists **and** its SHA-256 matches
**either** the registry pin **or** the value recorded in
`_manifest.json`. The downloader checks both — registry first, manifest
as a fallback for models whose upstream doesn't publish a hash.

### 3.2 What to do if verification fails

1. **Re-download** with `--force`: this re-streams the file and
   re-hashes. If it still fails, the file on HF has been updated
   upstream.
2. **Check the upstream release notes** at the HF repo link in §1.1.
   If they released a new quantisation, bump the registry entry in
   `src/tinyrag/models/registry.py` and the table in §1.
3. **Never trust a file whose hash doesn't match.** Even if it
   "looks fine" in `llama-server --version`, a tampered GGUF can
   exfiltrate data or produce wrong answers silently. The right
   response to a checksum mismatch is `rm models/<id>.gguf*` and
   re-download.

---

## 4. Disk Footprint

| Model | Approx. size |
|-------|--------------|
| `phi-3-mini` | 2.3 GB |
| `tinyllama-1.1b` | 700 MB |
| `llama-3.2-3b` | 1.8 GB |
| `mistral-7b` | 4.0 GB |
| **Total (all 4)** | **~8.8 GB** |

> **Note:** only one model is loaded into RAM at a time (llama.cpp
> keeps the rest on disk). So the RAM budget per `llama-server`
> invocation is ~1.5–2 GB for Phi-3 and similar for the others, not
> 8.8 GB. But the disk budget is real.

For Phase 5 evaluation on the laptop, you may want to download all
four. For the Pi 5 demo, only `phi-3-mini` is needed (1.5 GB RAM is
the tight constraint, not disk).

---

## 5. How a New Model Gets Added

1. Add a `ModelEntry` to `MODEL_REGISTRY` in
   `src/tinyrag/models/registry.py`. Pick an id (`kebab-case`), a
   `display_name`, the `hf_repo`, and the `hf_filename`.
2. Set `expected_sha256=""` for now. The downloader will populate the
   manifest on first download; you then copy that hash into the
   registry.
3. Add a row to §1 and a sub-section to §1.1 above.
4. Update `docs/05_tech_stack_v1.md` §3.5 if the primary LLM changed.
5. Add a column to the Phase 5 evaluation CSV.
6. Open a PR titled `chore(models): add <id> to the catalog`.

---

## 6. Open Questions for the Student

| # | Question | Default |
|---|----------|---------|
| Q1 | Pin SHA-256 in the registry, or always rely on the manifest? | **Manifest.** The registry is for catalog; the manifest is for the audit log. Both is fine but the manifest is required. |
| Q2 | Use bartowski's mirror for Llama 3.2, or wait for ungated access? | **Use bartowski.** Same weights, no gating, easier for reproducibility. |
| Q3 | Include Mistral 7B in the final report? | **Optional.** It's 4 GB and slow on the Pi. We can skip it if time is tight. |
| Q4 | Where to host the final SHA-256 values (registry, manifest, or both)? | **Both**, with a comment in the registry that points to the manifest as the runtime source. |

---

## 7. Document Approval

| Role | Name | Approval | Date |
|------|------|----------|------|
| Student | Marajul Haque | ⏳ pending | |
| Advisor | Abu Nowshed Chy | (not required for v1) | |

---

*End of MODELS.md. Update whenever a model is added, removed, or repinned.*
