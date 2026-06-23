# Security Policy

## Our Security Promise

TinyRAG is a **privacy-first** system. The whole point is that your smart-home data — device manuals, custom FAQ, sensor readings, and your questions — **never leave the device**. This document explains the threat model we designed for, how to report a vulnerability, and what we will (and will not) do.

---

## 1. Supported Versions

| Version | Supported |
|---------|-----------|
| `main` branch | ✅ Yes |
| Latest tagged release | ✅ Yes |
| Older tags | ❌ No |

TinyRAG is currently in **active development** (pre-1.0). Until v1.0 is tagged, we may break things as we learn. Pin to a commit hash if you need stability.

---

## 2. Threat Model

### 2.1 What TinyRAG protects against

| Threat | Mitigation |
|--------|------------|
| Cloud exfiltration of user queries | **No cloud calls at runtime.** Verified by running with Wi-Fi off (test in Phase 5). |
| Cloud exfiltration of sensor data | Sensors are read locally (CSV file or GPIO); nothing is uploaded. |
| Cloud exfiltration of personal documents | Documents are embedded and indexed **on device**; original text is never sent anywhere. |
| Tampering with model weights | GGUF files are SHA-256-checked at download (see `scripts/download_models.py`). |
| Local filesystem leakage | `data/`, `models/`, `logs/` are all gitignored. `.env` is gitignored. |
| Dependency confusion / typosquatting | All deps pinned in `requirements.txt` with hashes (added in Step 3.2). |
| Insecure defaults | `config.yaml` shipped with safe defaults; no debug endpoints exposed by default. |

### 2.2 What TinyRAG does NOT (yet) protect against

| Threat | Status |
|--------|--------|
| Network-level eavesdropping on the local web UI | Out of scope for v1 (single-device, LAN-only). For public exposure, put behind a reverse proxy with TLS. |
| Adversarial prompts to the LLM | Out of scope for capstone. Mention in `SECURITY.md` as known limitation. |
| Compromised model weights from upstream | Out of scope; we trust Hugging Face repos. Mitigate later with reproducible builds. |
| Physical access to the device | Out of scope (assumes physical security of the Pi). |
| Multi-tenant isolation | N/A (single-user device). |

---

## 3. Reporting a Vulnerability

**Please do not open a public GitHub issue for security issues.**

Email **marajul.cu@gmail.com** with:

1. **Description** of the vulnerability
2. **Reproduction steps** (the exact commands / inputs)
3. **Impact assessment** (what an attacker could achieve)
4. **Environment** (commit hash, OS, model version)
5. **Whether you want public credit** (we'll respect your preference)

### 3.1 What to expect

| Time after report | What happens |
|-------------------|--------------|
| Within 48 hours | Acknowledgement email |
| Within 7 days | Initial triage (confirmed, won't fix, or needs more info) |
| Within 30 days | Patch released OR documented decision to not fix |
| After patch | Public disclosure (with your credit if you want it) |

---

## 4. Security Checklist for Contributors

Before opening a PR that touches any of the following, please verify:

- [ ] **No new outbound network calls** added at runtime (use the existing `NetworkGuard` from `src/tinyrag/utils/network_guard.py` if you must).
- [ ] **No new dependencies** without updating `pyproject.toml` AND `requirements.txt` AND checking the dep's CVE history.
- [ ] **No new files** that should be gitignored (models, data, logs, secrets).
- [ ] **No new shell=True subprocess calls** that interpolate user input.
- [ ] **No new SQL string concatenation** (use parameterized queries via SQLite placeholders).
- [ ] **No new `pickle.load()`** on untrusted data (use JSON instead).
- [ ] **No new file paths constructed from user input** without sanitization (path traversal risk).

---

## 5. Known Limitations (Disclosed Transparently)

- The LLM can still hallucinate, even with retrieval. The retrieval helps, but is not a guarantee. We surface this in the UI.
- Single-user design: no authentication on the web UI. If you expose the UI to a network, add auth.
- No encryption-at-rest for the FAISS index or SQLite DB. If the device is stolen, the indexed chunks can be read.
- The system prompt is **not** signed; a sophisticated attacker with shell access could swap the system prompt.

---

## 6. Security-Related Configuration

| Setting in `config.yaml` | Default | Recommended for production |
|--------------------------|---------|---------------------------|
| `api.host` | `127.0.0.1` | `0.0.0.0` only with a reverse proxy |
| `api.enable_docs` | `true` | `false` (don't expose OpenAPI publicly) |
| `api.cors_origins` | `[]` | Explicit allowlist |
| `llamacpp.allow_remote` | `false` | Keep `false` (the LLM is local) |
| `logging.level` | `INFO` | `WARNING` in production |
| `logging.include_pii` | `false` | Keep `false` |

---

## 7. Hall of Fame

_(No external reports yet — be the first!)_

---

## 8. License Note

This security policy is part of the TinyRAG project, licensed under MIT. The security promise above is a **good-faith commitment**, not a legal warranty.
