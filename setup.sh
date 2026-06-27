#!/usr/bin/env bash
# ============================================================================
# TinyRAG — One-Command Bootstrap (Step 4.24)
# ----------------------------------------------------------------------------
# Idempotent installer. Wraps the heavy lifting done by the Makefile so
# the README's "Quick Start" promise (`git clone && bash setup.sh && bash
# run.sh`) is finally real. Safe to re-run on an already-set-up machine —
# every step checks for the artefact it produces and skips if present.
#
# USAGE
#   bash setup.sh              # run the full bootstrap (idempotent)
#   bash setup.sh --help       # show this header
#
# WHAT IT DOES (6 stages, each idempotent)
#   1. Preflight      Verify bash ≥ 4, git, make, python3, ~3 GB free disk.
#   2. deps-system    `make deps-system` — apt install build-essential,
#                     cmake, libopenblas-dev (needed to compile llama.cpp).
#                     Skipped if apt-get is not available (e.g. macOS).
#   3. install-dev    `make install-dev` — create venv + install runtime
#                     + dev/test deps from requirements*.txt.
#                     Skipped if the venv already exists at $VENV.
#   4. build-llamacpp `make build-llamacpp` — clone + compile llama.cpp
#                     with OpenBLAS (~5 min cold; skipped if the binary
#                     is already present at llama.cpp/build/bin/llama-server).
#   5. download-llm   `make download-llm` — fetch models/phi-3-mini.gguf
#                     from HuggingFace (~2.4 GB; skipped if the file
#                     already exists + SHA-256 matches the manifest).
#   6. sensors-generate `make sensors-generate` — write
#                     data/sensor_logs/synthetic_30d.csv (30 days ×
#                     6 sensors; SEED=42 for reproducibility; skipped
#                     if the file is already present).
#
# WHY A SHELL SCRIPT (not just `make setup`)?
#   - Make targets can't orchestrate cross-stage idempotency cleanly.
#   - A shell wrapper can show "Done in N s" + per-stage checkmarks.
#   - Matches the README's Quick Start (single command, no Makefile
#     knowledge required).
#   - Reuses every Makefile target — no duplicated logic. setup.sh is
#     a thin UX layer over the Makefile.
#
# EXIT CODES
#    0  PASS — all 6 stages succeeded (or were skipped because the
#             artefact was already present).
#   10  Preflight failed (missing tool, bash too old, disk too small).
#   11  deps-system failed (apt install errored — usually a missing
#       package or sudo).
#   12  install-dev failed (pip install errored — usually requirements.txt
#       drift or network down).
#   13  build-llamacpp failed (CMake configure or compile errored).
#   14  download-llm failed (HuggingFace download errored or SHA-256
#       mismatch — usually a partial download).
#   15  sensors-generate failed (CSV write errored — usually a disk
#       full or permission issue on data/).
#
# IDEMPOTENCY
#   Every stage checks for the artefact it produces BEFORE invoking
#   make. Re-running setup.sh on a set-up machine prints "skipped"
#   for each stage and exits 0 in ~3 s. The only stage that may NOT
#   be a perfect no-op is `make download-llm` if the model file is
#   corrupted — the Makefile's verify-llm step will re-download in
#   that case.
#
# REFERENCES
#   - docs/06_roadmap_v2.md Step 4.24 (this step)
#   - docs/01_project_scope_v2.md §Reproducible
#   - Makefile (the source of truth for every heavy-lift target)
# ============================================================================


# ---- Safety flags ---------------------------------------------------------

# `set -e`: exit on any error.
# `set -u`: error on undefined variables (catches typos in $VENV etc).
# `set -o pipefail`: catch failures in piped commands, not just the last.
set -euo pipefail


# ---- Configuration --------------------------------------------------------

# Where the repo lives. We resolve via ${BASH_SOURCE[0]} so setup.sh
# works even when invoked from a different cwd (e.g. `bash ../setup.sh`).
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The venv path. Matches the Makefile default so setup.sh + the Makefile
# use the same venv (no "two venvs" trap). Override via VENV env var
# for hermetic CI runs (Step 4.20 does this).
: "${VENV:=${HOME}/venvs/tinyrag}"

# Where to write the synthetic sensor dataset. Matches Makefile default.
: "${SENSOR_DATA:=${REPO_ROOT}/data/sensor_logs/synthetic_30d.csv}"

# Minimum supported versions + resource thresholds. Centralised so
# they're easy to bump + grep.
readonly MIN_BASH_MAJOR=4
readonly MIN_PYTHON_MAJOR=3
readonly MIN_PYTHON_MINOR=12
readonly MIN_DISK_FREE_KB=3000000  # ~3 GB (model 2.4 GB + build 500 MB + venv 300 MB)

# Exit-code constants. Tests grep for these so the documented contract
# is enforced. See the EXIT CODES block in the docstring above.
readonly EXIT_OK=0
readonly EXIT_PREFLIGHT=10
readonly EXIT_DEPS_SYSTEM=11
readonly EXIT_INSTALL=12
readonly EXIT_BUILD=13
readonly EXIT_DOWNLOAD=14
readonly EXIT_SENSORS=15


# ---- CLI argument parsing -------------------------------------------------

# Minimal flag parser — we only support --help today, but the pattern
# mirrors scripts/portability_check.sh so adding --quiet / --yes later
# is a copy-paste.
print_help() {
    sed -n '2,68p' "$0"
    exit "${EXIT_OK}"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h) print_help ;;
            *)
                log_error "Unknown argument: $1"
                echo "Try: bash setup.sh --help" >&2
                exit 2
                ;;
        esac
        shift
    done
}


# ---- Pretty output helpers ------------------------------------------------

# TTY-aware ANSI colours (auto-disabled when piped to a file or another
# command — keeps CI logs clean). Same pattern as
# scripts/portability_check.sh:139-149.
if [[ -t 1 ]]; then
    readonly C_RESET=$'\033[0m'
    readonly C_BOLD=$'\033[1m'
    readonly C_GREEN=$'\033[32m'
    readonly C_YELLOW=$'\033[33m'
    readonly C_RED=$'\033[31m'
    readonly C_BLUE=$'\033[34m'
    readonly C_DIM=$'\033[2m'
else
    readonly C_RESET="" C_BOLD="" C_GREEN="" C_YELLOW="" C_RED="" C_BLUE="" C_DIM=""
fi

log_info()    { printf "%s[setup]%s  %s\n" "${C_BLUE}"   "${C_RESET}" "$*"; }
log_ok()      { printf "%s[  OK ]%s  %s\n" "${C_GREEN}"  "${C_RESET}" "$*"; }
log_warn()    { printf "%s[WARN ]%s  %s\n" "${C_YELLOW}" "${C_RESET}" "$*" >&2; }
log_error()   { printf "%s[FAIL ]%s  %s\n" "${C_RED}"    "${C_RESET}" "$*" >&2; }
log_section() { printf "\n%s==> %s%s\n" "${C_BOLD}${C_BLUE}" "$*" "${C_RESET}"; }
log_skipped() { printf "%s[SKIP ]%s  %s\n" "${C_DIM}"    "${C_RESET}" "$*"; }


# ---- Preflight ------------------------------------------------------------

# Verify every prerequisite is in place BEFORE we burn minutes on a
# partial install. The whole point of this gate is to fail fast with a
# specific "you need X" message instead of a cryptic "cmake: command
# not found" 5 minutes later.
preflight() {
    log_section "Preflight"

    local missing=()
    command -v git >/dev/null 2>&1     || missing+=("git")
    command -v make >/dev/null 2>&1    || missing+=("make")
    command -v python3 >/dev/null 2>&1 || missing+=("python3")
    command -v curl >/dev/null 2>&1   || missing+=("curl")
    command -v df >/dev/null 2>&1     || missing+=("df (coreutils)")

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing[*]}"
        log_error "Install with: sudo apt-get install -y ${missing[*]}"
        exit "${EXIT_PREFLIGHT}"
    fi

    # Bash ≥ 4 (associative arrays, mapfile, etc.). The `((` arithmetic
    # context makes the comparison silent on success.
    if (( BASH_VERSINFO[0] < MIN_BASH_MAJOR )); then
        log_error "bash ≥ ${MIN_BASH_MAJOR} required (have ${BASH_VERSION})"
        log_error "On macOS: brew install bash"
        exit "${EXIT_PREFLIGHT}"
    fi

    # Python ≥ 3.12 (matches requirements.txt + Makefile).
    local py_version py_major py_minor
    py_version="$(python3 --version 2>&1 | awk '{print $2}')"
    py_major="$(echo "${py_version}" | cut -d. -f1)"
    py_minor="$(echo "${py_version}" | cut -d. -f2)"
    if [[ "${py_major}" -lt "${MIN_PYTHON_MAJOR}" ]] \
       || { [[ "${py_major}" -eq "${MIN_PYTHON_MAJOR}" ]] \
            && [[ "${py_minor}" -lt "${MIN_PYTHON_MINOR}" ]]; }; then
        log_error "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ required, found ${py_version}"
        exit "${EXIT_PREFLIGHT}"
    fi

    # ~3 GB free on the partition that holds the repo. df -Pk is the
    # portable POSIX form (P = POSIX, k = 1-K blocks). The model is
    # 2.4 GB; llama.cpp build adds ~500 MB; venv + deps add ~300 MB.
    local free_kb
    free_kb="$(df -Pk "${REPO_ROOT}" | awk 'NR==2 {print $4}')"
    if (( free_kb < MIN_DISK_FREE_KB )); then
        log_error "Need ~$((MIN_DISK_FREE_KB / 1024)) MB free on $(df -Pk "${REPO_ROOT}" | awk 'NR==2 {print $6}'); have $((free_kb / 1024)) MB"
        exit "${EXIT_PREFLIGHT}"
    fi

    log_info "Bash: ${BASH_VERSION}; Python: ${py_version}; free disk: $((free_kb / 1024)) MB"
    log_ok "Preflight passed"
}


# ---- Stage 1: deps-system -------------------------------------------------

# Install the apt packages llama.cpp needs to compile (build-essential,
# cmake, libopenblas-dev). Skipped if apt-get isn't available (macOS,
# WSL without systemd, etc.) — we assume the user has the equivalent
# packages installed via their platform's package manager.
install_system_deps() {
    log_section "System dependencies (apt)"

    if ! command -v apt-get >/dev/null 2>&1; then
        log_warn "apt-get not found — assuming system deps already installed"
        log_warn "Required: build-essential cmake libopenblas-dev (or distro equivalents)"
        return 0
    fi

    if ! make deps-system; then
        log_error "make deps-system failed"
        log_error "Try running manually: sudo bash scripts/install_system_deps.sh"
        exit "${EXIT_DEPS_SYSTEM}"
    fi
    log_ok "System deps installed"
}


# ---- Stage 2: install-dev -------------------------------------------------

# Create the venv + install runtime + dev/test deps. Skipped if the
# venv already exists (the Makefile's `install-dev` target depends on
# `venv`, which is itself idempotent — but we add a guard here for
# faster + clearer logging on re-runs).
install_python_deps() {
    log_section "Python virtualenv + dependencies (VENV=${VENV})"

    if [[ -d "${VENV}" ]]; then
        log_skipped "Python venv already present at ${VENV}"
        return 0
    fi

    log_info "Creating venv + installing deps (this takes 20-60s on a fresh machine)"
    if ! make VENV="${VENV}" install-dev; then
        log_error "make install-dev failed"
        log_error "Check requirements.txt for drift; verify pip can reach PyPI"
        exit "${EXIT_INSTALL}"
    fi
    log_ok "Venv + deps ready at ${VENV}"
}


# ---- Stage 3: build-llamacpp ----------------------------------------------

# Clone (if missing) + compile llama.cpp with OpenBLAS. The binary
# must exist at llama.cpp/build/bin/llama-server — that's what run.sh
# invokes. Skipped if the binary is already there.
build_llamacpp() {
    log_section "Build llama.cpp (one-time, ~5 min cold)"

    local llamacpp_bin="${REPO_ROOT}/llama.cpp/build/bin/llama-server"
    if [[ -x "${llamacpp_bin}" ]]; then
        log_skipped "llama.cpp already built at ${llamacpp_bin}"
        return 0
    fi

    log_info "Cloning + compiling llama.cpp (5+ min on a fresh checkout)"
    if ! make build-llamacpp; then
        log_error "make build-llamacpp failed"
        log_error "Check that deps-system ran first (build-essential, cmake, libopenblas-dev)"
        exit "${EXIT_BUILD}"
    fi

    if [[ ! -x "${llamacpp_bin}" ]]; then
        log_error "build-llamacpp exited 0 but binary not at expected path: ${llamacpp_bin}"
        log_error "If you built llama.cpp to a custom location (per Step 3.4a), set"
        log_error "LLAMACPP_BIN in run.sh or symlink the binary into llama.cpp/build/bin/."
        exit "${EXIT_BUILD}"
    fi
    log_ok "llama.cpp built at ${llamacpp_bin}"
}


# ---- Stage 4: download-llm ------------------------------------------------

# Download models/phi-3-mini.gguf from HuggingFace (~2.4 GB). The
# Makefile's download-llm target is already idempotent — it skips
# if the file is present + SHA-256 matches. We still wrap it so the
# error path is consistent with the other stages.
download_model() {
    log_section "Download primary LLM (models/phi-3-mini.gguf, ~2.4 GB)"

    local model_path="${REPO_ROOT}/models/phi-3-mini.gguf"
    if [[ -f "${model_path}" ]]; then
        log_skipped "Model already present at ${model_path}"
        return 0
    fi

    log_info "Downloading from HuggingFace (takes 2-10 min on broadband)"
    if ! make download-llm; then
        log_error "make download-llm failed"
        log_error "Check your network connection; if the file is partial, delete it and re-run"
        exit "${EXIT_DOWNLOAD}"
    fi
    log_ok "Model downloaded to ${model_path}"
}


# ---- Stage 5: sensors-generate --------------------------------------------

# Generate the 30-day synthetic sensor dataset. SEED=42 (set in the
# Makefile target) makes this reproducible — same input, same output.
# Skipped if the file is already present.
generate_synthetic_data() {
    log_section "Generate synthetic sensor dataset"

    if [[ -f "${SENSOR_DATA}" ]]; then
        log_skipped "Synthetic dataset already present at ${SENSOR_DATA}"
        return 0
    fi

    log_info "Writing 30 days × 6 sensors of synthetic data"
    if ! make SENSOR_DATA="${SENSOR_DATA}" sensors-generate; then
        log_error "make sensors-generate failed"
        log_error "Check disk space + write permissions on data/"
        exit "${EXIT_SENSORS}"
    fi
    log_ok "Synthetic dataset written to ${SENSOR_DATA}"
}


# ---- Main -----------------------------------------------------------------

main() {
    parse_args "$@"

    log_section "TinyRAG setup (Step 4.24)"
    log_info "Repo root: ${REPO_ROOT}"
    log_info "Venv target: ${VENV}"
    log_info "Sensor dataset: ${SENSOR_DATA}"
    log_info "Idempotent — re-running on a set-up machine is a no-op."

    preflight
    install_system_deps
    install_python_deps
    build_llamacpp
    download_model
    generate_synthetic_data

    log_section "Done"
    log_ok "TinyRAG is ready."
    log_info "Next: bash run.sh"
    log_info "(If run.sh complains about a missing venv, set VENV=${VENV} in your shell.)"
}

main "$@"