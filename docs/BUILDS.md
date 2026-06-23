# TinyRAG — Native Build Manifest

**Project Title:** TinyRAG — A Lightweight, On-Device Retrieval-Augmented Generation Assistant for Smart Home IoT
**Document version:** 1.0
**Date:** 2026-06-23
**Status:** Active
**Companion to:** `docs/05_tech_stack_v1.md` (Python deps) and `docs/06_roadmap_v2.md` Step 3.4

---

## 0. Purpose

This document is the **build manifest** for everything TinyRAG compiles natively (not pure Python). It serves four jobs:

1. **Pin the exact versions** of every native tool (llama.cpp commit, OpenBLAS version, etc.) so a fresh `git clone` produces a byte-identical binary.
2. **Document the build flags** used (so future maintainers know which knobs were flipped and why).
3. **Provide verification commands** (so we can confirm OpenBLAS is *actually* linked, not just installed).
4. **Be the rollback recipe** if a future upgrade breaks something.

If you bump a version here, update the **system deps script** (`scripts/install_system_deps.sh`) **and** the **Makefile** at the same time. Three-way drift is the #1 source of "works on my machine" bugs in native code.

---

## 1. System Packages (apt-installed)

These are installed by `scripts/install_system_deps.sh`. Full per-package rationale lives in comments at the top of that script.

| Package | Why we need it | Pinned version (Ubuntu 24.04) | Verified by |
|---------|----------------|-------------------------------|-------------|
| `build-essential` | gcc, g++, make, libc-dev — required to compile llama.cpp from source | `12.10ubuntu1` | `dpkg -l build-essential` |
| `cmake` | llama.cpp's build system (cmake ≥ 3.14 required) | `3.28.3-1build7` | `cmake --version` |
| `git` | Fetching llama.cpp source, submodules, model repos | `1:2.43.0-1ubuntu7.3` | `git --version` |
| `libopenblas-dev` | Optimized BLAS — gives llama.cpp a ~2× speedup on x86_64 | `0.3.26+ds-1` | `pkg-config --modversion openblas` |
| `liblapack-dev` | Linear algebra backend (OpenBLAS uses it internally) | `3.11.0-4` | `dpkg -l liblapack-dev` |
| `sqlite3` | CLI for inspecting the metadata DB during debugging | `3.45.1-1ubuntu2.5` | `sqlite3 --version` |
| `tree` | Pretty-print the project directory structure in docs | latest in repo | `tree --version` |
| `pkg-config` | Optional (--with-extras). Finds OpenBLAS for cmake. | latest in repo | `pkg-config --version` |
| `ninja-build` | Optional (--with-extras). Faster cmake backend. | latest in repo | `ninja --version` |

**Install:** `bash scripts/install_system_deps.sh` (or `make deps-system`).
**Verify:** `bash scripts/install_system_deps.sh --check`.

---

## 2. llama.cpp (LLM inference engine)

### 2.1 Pinned Version

| Field | Value | Pinned on | Verified by |
|-------|-------|-----------|-------------|
| **Repo** | `https://github.com/ggerganov/llama.cpp.git` | — | `git -C llama.cpp remote -v` |
| **Tag** | `gguf-v0.19.0` | 2026-06-23 (Step 3.4) | `git -C llama.cpp describe --tags` |
| **Commit SHA** | `a290ce626663dae1d54f70bce3ca6d8f67aab62f` | 2026-06-23 | `git -C llama.cpp rev-parse HEAD` |
| **Tag date** | 2026-05-06 | — | `git -C llama.cpp log -1 --format=%ci` |
| **Submodules** | `ggml` (recursive) | — | `git -C llama.cpp submodule status` |

> **Why pin by tag, not master?** `master` moves daily and breaks reproducibility. llama.cpp tags in the `bNNNN` range are auto-generated daily builds; the `gguf-vX.Y.Z` tags are versioned releases (roughly monthly) and are the stable surface for downstream users.

### 2.2 Build Flags (laptop / Ubuntu 24.04 / x86_64)

| Flag | Value | Why |
|------|-------|-----|
| `CMAKE_BUILD_TYPE` | `Release` | Strips debug symbols, enables `-O3` |
| `GGML_BLAS` | `ON` | **Use OpenBLAS for matrix math.** (Note: the legacy `GGML_OPENBLAS=ON` flag is deprecated in `gguf-v0.19.0` and is silently ignored — only `GGML_BLAS=ON` works.) |
| `GGML_BLAS_VENDOR` | `OpenBLAS` | Disambiguate from Accelerate / MKL |
| `GGML_NATIVE` | _unset (defaults to OFF)_ | We want a portable binary, not `-march=native` |
| `GGML_CPU_ALL_VARIANTS` | _unset_ | Not needed unless we want every quantized format |
| `LLAMA_BUILD_TESTS` | `OFF` | Skip building test binaries (smaller binary) |
| `LLAMA_BUILD_EXAMPLES` | `ON` | We use `llama-server`, `llama-bench`, `llama-cli` |
| `LLAMA_CURL` | `OFF` | TinyRAG never downloads models at runtime |
| `CMAKE_C_COMPILER` | system gcc | — |
| `CMAKE_CXX_COMPILER` | system g++ | — |

**Parallel build:** `cmake --build build --config Release -j 10` (i5-1235U has 10 cores, 12 threads).

### 2.2.1 Build Location (colon-in-path workaround)

**Default:** source and build live in `${PROJECT_ROOT}/llama.cpp/` and `${PROJECT_ROOT}/llama.cpp/build/`.

**Workaround (current laptop):** the project path `~/Desktop/Capstone Project/TinyRAG: Retrieval-Augmented Generation forEdge IoT./` contains colons, which GNU Make cannot parse inside auto-generated Makefile target names. To work around this, the build script `scripts/build_llamacpp.sh` auto-detects a colon in `PROJECT_ROOT` and diverts the source tree to `/tmp/llamacpp-build/` and the build directory to `/tmp/llamacpp-build/build/`. After building, the script symlinks `${PROJECT_ROOT}/llama.cpp/build` → `/tmp/llamacpp-build/build` so the rest of the toolchain (Makefile, scripts, run.sh) finds the binary at the expected path.

**Long-term fix:** rename the project directory to remove the colon (e.g. `TinyRAG-EdgeIoT/`). This is tracked as a P1 cleanup task.

### 2.3 Pi 5 Build Flags (for Phase 6 — placeholder)

Will differ in three ways:
- `GGML_OPENBLAS=OFF` → use the Pi's Cortex-A76 NEON path instead
- `-DGGML_NATIVE=OFF` → portable ARM build
- `-j 4` → only 4 cores

These will be added to a separate script (`scripts/build_llamacpp_pi.sh`) in Step 6.x.

### 2.4 Verification

After Step 3.4 completes, run all three:

```bash
# 1. Binary exists and is executable
ls -la llama.cpp/build/bin/llama-server
# Expected: -rwxr-xr-x ... ./llama.cpp/build/bin/llama-server

# 2. OpenBLAS is actually linked (not just present on the system)
ldd llama.cpp/build/bin/llama-server | grep -E "openblas|blas"
# Expected: libopenblas.so.0 => /usr/lib/x86_64-linux-gnu/libopenblas.so.0

# 3. Binary runs and reports its version
./llama.cpp/build/bin/llama-server --version
# Expected: version: a290ce6 (or the pinned tag's short SHA)

# 4. (preferred) Run the Python verification script
python scripts/verify_llamacpp.py
# Expected: "7/7 checks passed"
```

If `ldd` shows NO openblas line, the build fell back to the slow generic BLAS. Rebuild with `GGML_BLAS=ON GGML_BLAS_VENDOR=OpenBLAS`.

### 2.5 Verified Build Record (laptop, 2026-06-23)

The Step 3.4 build was completed and verified on 2026-06-23. Recorded here so future maintainers can compare against a known-good build.

| Field | Value |
|-------|-------|
| **Date (UTC)** | 2026-06-23 |
| **Host** | Dell Inspiron 15 3520 (i5-1235U, 10 cores, Ubuntu 24.04.4 LTS) |
| **OS kernel** | Linux 6.8.x |
| **Compiler** | gcc 13.3.0 |
| **CMake** | 3.28.3 |
| **OpenBLAS** | 0.3.26 (system package `libopenblas-dev`) |
| **OpenMPI / BLAS deps** | `liblapack-dev` 3.11.0 |
| **llama.cpp tag** | `gguf-v0.19.0` |
| **llama.cpp commit** | `a290ce626663dae1d54f70bce3ca6d8f67aab62f` |
| **Build dir** | `/tmp/llamacpp-build/build` (colon-in-path workaround) |
| **Project symlink** | `llama.cpp/build -> /tmp/llamacpp-build/build` |
| **Binary size** | 9,436,656 bytes (~9.4 MB) |
| **Binary version** | `version: 9046 (a290ce626)` |
| **OpenBLAS linked?** | YES — `libopenblas.so.0 => /lib/x86_64-linux-gnu/libopenblas.so.0` |
| **Verification** | `python scripts/verify_llamacpp.py` → 7/7 checks passed |
| **Build wall time** | ~7 min (first time) — incremental rebuilds <30 s |

**Known caveats:**
- The build directory is in `/tmp` (not in the project) due to the colon-in-path workaround (§2.2.1). It will be wiped on reboot. Re-run `bash scripts/build_llamacpp.sh` after a reboot to restore.
- After renaming the project directory to remove the colon, the build will land in `${PROJECT_ROOT}/llama.cpp/build/` as originally designed.

---

## 3. OpenBLAS-Specific Notes

### 3.1 Why OpenBLAS on the laptop?

| Backend | Speed on x86_64 | Multi-threading | Notes |
|---------|-----------------|-----------------|-------|
| **OpenBLAS** ✅ | ~2× generic BLAS | Yes (auto) | What we use |
| Reference BLAS | 1× baseline | No | Falls back to this if OpenBLAS not linked |
| Intel MKL | Slightly faster than OpenBLAS | Yes | Closed-source, ~1 GB extra deps |
| Apple Accelerate | Fast on M-series | Yes | macOS-only, irrelevant on Linux |
| CUDA / cuBLAS | Fastest | GPU-only | Out of scope (no GPU on Pi or most laptops) |

### 3.2 How many threads?

OpenBLAS auto-detects cores. On the i5-1235U (10 cores, 12 threads), you can also pin via `OPENBLAS_NUM_THREADS=8` to avoid over-subscription. The `llama-server` invocation in Step 3.7 will set `--threads 10`.

### 3.3 Models (cross-reference)

This document covers **native compilation** (llama.cpp + OpenBLAS). For the GGUF model files themselves, see:

- **`docs/MODELS.md`** — the human-readable catalog (id, repo, size, license, SHA-256, role).
- **`src/tinyrag/models/registry.py`** — the machine-readable catalog (`MODEL_REGISTRY`).
- **`models/_manifest.json`** — the per-machine audit log (written on first download; runtime source of truth for "is this file genuine?").

The build script (`scripts/build_llamacpp.sh`) only produces the inference engine. The model files are a separate Step (3.5) downloaded via `scripts/download_models.py` (or `make download-llm`).

---

## 4. Reproducibility Checklist

After a fresh clone, can someone reproduce the exact binary?

- [ ] Ubuntu 24.04.4 LTS (Noble Numbat)
- [ ] `bash scripts/install_system_deps.sh` (installs apt deps, idempotent)
- [ ] `git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp && git checkout <pinned-SHA>`
- [ ] `bash scripts/build_llamacpp.sh` (cmake flags per §2.2)
- [ ] Verification commands in §2.4 all pass

If any of these fail, the diff is almost certainly in the system packages (apt repo changed) or the llama.cpp commit (we forgot to bump §2.1).

---

## 5. Upgrade Policy

When bumping a native component:

1. **Read the upstream release notes / changelog** carefully.
2. **Open a feature branch** in this repo (`feat/bump-llamacpp-2026-07`).
3. **Update this file** (BUILDS.md) FIRST with the new pinned SHA.
4. **Update the build script** (`scripts/build_llamacpp.sh`) if flags changed.
5. **Build, verify, smoke-test** on the laptop.
6. **Commit BUILDS.md + script together** in one atomic commit.
7. **Merge only after a full Phase 5 evaluation run shows no regression.**

Never bump a native version in a hurry — these are the load-bearing pieces of the system.

---

## 6. Known Issues / TODO

| # | Issue | Workaround | Will fix in |
|---|-------|-----------|-------------|
| 1 | llama.cpp has no formal "stable" release tag — we pin by commit. | Document the commit SHA in §2.1 every time we rebuild. | Ongoing |
| 2 | The CUDA build of `torch` was installed (Step 3.2). Harmless but ~2 GB wasted. | Install via `--index-url https://download.pytorch.org/whl/cpu` on next clean install. | Step 3.2 hardening |
| 3 | OpenBLAS thread auto-detection sometimes over-subscribes on hyperthreaded CPUs. | Pass `OPENBLAS_NUM_THREADS=8` (or actual core count) when starting llama-server. | Step 3.7 |
| 4 | Pi 5 build flags are placeholders only. | Real Pi build script will be created in Step 6.4. | Phase 6 |
| 5 | **Project path contains `:` — GNU Make cannot parse the auto-generated Makefiles.** | Build is diverted to `/tmp/llamacpp-build/` and symlinked into the project. See §2.2.1. | P1 — rename the project directory to drop the colon. |
| 6 | The legacy `GGML_OPENBLAS=ON` cmake flag is silently ignored in `gguf-v0.19.0`+. Builds with this flag appear to succeed but don't actually link OpenBLAS. | Use `GGML_BLAS=ON` + `GGML_BLAS_VENDOR=OpenBLAS` (see §2.2). The build script handles this correctly. | Resolved in Step 3.4. |

---

## 7. Document Approval

| Role | Name | Approval | Date |
|------|------|----------|------|
| Student | Marajul Haque | ⏳ pending | |
| Advisor | Abu Nowshed Chy | (not required for v1) | |

---

*End of BUILDS.md. Update whenever a native version changes.*
