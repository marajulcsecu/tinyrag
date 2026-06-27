#!/usr/bin/env bash
# ============================================================================
# TinyRAG — One-Command Stack Launcher (Step 4.24)
# ----------------------------------------------------------------------------
# Brings up the entire TinyRAG stack in one foreground process: backgrounds
# llama-server, waits for it to pass its /health check, backgrounds uvicorn,
# waits for it to serve /api/status, then blocks on uvicorn so the user
# sees uvicorn's logs in their terminal. Traps EXIT INT TERM so Ctrl+C
# tears down both children cleanly.
#
# USAGE
#   bash run.sh                  # start the full stack (foreground)
#   bash run.sh --help           # show this header
#
# ENVIRONMENT VARIABLES (override for tests / non-standard setups)
#   LLAMACPP_BIN      Path to llama-server. Default: $REPO_ROOT/llama.cpp/
#                     build/bin/llama-server. Override if you built
#                     llama.cpp to a custom location (per Step 3.4a).
#   LLM_GGUF          Path to the GGUF model. Default: $REPO_ROOT/models/
#                     phi-3-mini.gguf. Override for tinyllama / mistral
#                     demos.
#   LLM_HOST          llama-server bind address. Default: 127.0.0.1.
#   LLM_PORT          llama-server port. Default: 8080.
#   API_HOST          uvicorn bind address. Default: 127.0.0.1.
#   API_PORT          uvicorn port. Default: 8000.
#   VENV              Path to the Python venv. Default: $HOME/venvs/tinyrag.
#                     Must match what setup.sh used (or set the same env
#                     var when invoking setup.sh).
#   HEALTH_TIMEOUT    Seconds to wait for each service to become healthy.
#                     Default: 60.
#
# WHAT IT DOES (3 stages)
#   1. Preflight      Verify the llama-server binary + GGUF model exist;
#                     verify ports 8000/8080 are free.
#   2. start_llama    Background llama-server → write PID to logs/llama-
#                     server.pid → poll /health with a 60 s budget.
#   3. start_uvicorn  Background uvicorn → write PID to logs/uvicorn.pid
#                     → poll /api/status with a 60 s budget → `wait`
#                     on uvicorn so this script blocks until Ctrl+C.
#
# WHY FOREGROUND UVICORN (not just `&`)?
#   Running uvicorn as the foreground process of this script means:
#     - The user sees uvicorn's logs in their terminal (the way `make
#       run-api` works today — we don't change that behaviour, we just
#       add llama-server as a sibling).
#     - When the user hits Ctrl+C, the INT signal goes to the script,
#       which fires the EXIT trap, which kills BOTH children.
#   Running both in background would orphan llama-server when the user
#   Ctrl+C's run.sh — that's the bug we're fixing.
#
# WHY AN EXIT TRAP (not just an INT trap)?
#   EXIT catches every exit path:
#     - Normal uvicorn shutdown (e.g. after a fatal error)
#     - Ctrl+C → INT
#     - `kill <run.sh-pid>` → TERM
#     - Any unhandled error (set -e + ERR-style failure)
#   This is the critical race-condition fix: without it, Ctrl+C leaves
#   llama-server running, eating ~3 GB of RAM, with the user thinking
#   "I killed it".
#
# EXIT CODES
#    0  PASS — both services ran (and were torn down cleanly via the
#             trap).
#   10  Preflight failed (binary / model / port).
#   11  llama-server binary not found at $LLAMACPP_BIN.
#   12  GGUF model not found at $LLM_GGUF.
#   13  Port (8000 or 8080) is already in use by a foreign process.
#   14  Health check timed out (llama-server or uvicorn didn't become
#       healthy within HEALTH_TIMEOUT seconds).
#   15  llama-server crashed (process died before /health responded).
#
# HOW TO TURN OFF THE STACK
#   - Ctrl+C in the terminal running run.sh (graceful SIGTERM)
#   - `bash stop.sh` from another terminal (the Step 4.24 sibling)
#   - `kill <pid>` where <pid> is from logs/run.sh.pid (or just the
#     run.sh process visible in `ps`)
#
# REFERENCES
#   - docs/06_roadmap_v2.md Step 4.24 (this step)
#   - Makefile `run-llm` + `run-api` (the two targets this script merges)
#   - scripts/portability_check.sh (the strict-bash + trap pattern)
# ============================================================================


# ---- Safety flags ---------------------------------------------------------

# `set -e`: exit on any error.
# `set -u`: error on undefined variables (catches typos in $LLAMACPP_BIN etc).
# `set -o pipefail`: catch failures in piped commands, not just the last.
set -euo pipefail


# ---- Configuration --------------------------------------------------------

# Where the repo lives. Resolve via ${BASH_SOURCE[0]} so run.sh works
# even when invoked from a different cwd.
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# All paths/ports are env-var overridable so tests can stub them out
# without touching the real install. Defaults match setup.sh + the
# Makefile so the out-of-the-box experience is "just works".
readonly LLAMACPP_BIN="${LLAMACPP_BIN:-${REPO_ROOT}/llama.cpp/build/bin/llama-server}"
readonly LLM_GGUF="${LLM_GGUF:-${REPO_ROOT}/models/phi-3-mini.gguf}"
readonly LLM_HOST="${LLM_HOST:-127.0.0.1}"
readonly LLM_PORT="${LLM_PORT:-8080}"
readonly API_HOST="${API_HOST:-127.0.0.1}"
readonly API_PORT="${API_PORT:-8000}"
: "${VENV:=${HOME}/venvs/tinyrag}"
readonly VENV="${VENV}"
readonly HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT:-60}"

# Where to write runtime artefacts. logs/ is gitignored + already created
# during setup (run-llm writes its own logs there too). The PID files
# are the on-disk contract that stop.sh + on_exit use to find children.
readonly LOG_DIR="${REPO_ROOT}/logs"
readonly LLAMA_PIDFILE="${LOG_DIR}/llama-server.pid"
readonly UVICORN_PIDFILE="${LOG_DIR}/uvicorn.pid"
readonly LLAMA_LOG="${LOG_DIR}/llama-server.log"
readonly UVICORN_LOG="${LOG_DIR}/uvicorn.log"

# llama-server context size + thread count. Matches the Makefile's
# `run-llm` target exactly so the behaviour is identical whether you
# run `make run-llm` or `bash run.sh`.
readonly LLAMA_CTX_SIZE=4096
readonly LLAMA_THREADS=10

# Exit-code constants. Tests grep for these — see the EXIT CODES
# block in the docstring above.
readonly EXIT_OK=0
readonly EXIT_PREFLIGHT=10
readonly EXIT_LLAMA_MISSING=11
readonly EXIT_MODEL_MISSING=12
readonly EXIT_PORT_BUSY=13
readonly EXIT_HEALTH_TIMEOUT=14
readonly EXIT_LLAMA_CRASHED=15


# ---- PID bookkeeping ------------------------------------------------------

# These two globals are written by start_llama / start_uvicorn and read
# by on_exit. Declared here so on_exit can reference them before they're
# assigned (set -u would otherwise flag the `[[ -n "${LLAMA_PID}" ]]`
# check as an unbound-variable error).
LLAMA_PID=""
UVICORN_PID=""


# ---- CLI argument parsing -------------------------------------------------

print_help() {
    sed -n '2,80p' "$0"
    exit "${EXIT_OK}"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h) print_help ;;
            *)
                log_error "Unknown argument: $1"
                echo "Try: bash run.sh --help" >&2
                exit 2
                ;;
        esac
        shift
    done
}


# ---- Pretty output helpers ------------------------------------------------

# TTY-aware ANSI colours (auto-disabled when piped). Same shape as
# setup.sh + scripts/portability_check.sh so the three scripts look
# consistent in the terminal.
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

log_info()    { printf "%s[run]%s    %s\n" "${C_BLUE}"   "${C_RESET}" "$*"; }
log_ok()      { printf "%s[  OK ]%s  %s\n" "${C_GREEN}"  "${C_RESET}" "$*"; }
log_warn()    { printf "%s[WARN ]%s  %s\n" "${C_YELLOW}" "${C_RESET}" "$*" >&2; }
log_error()   { printf "%s[FAIL ]%s  %s\n" "${C_RED}"    "${C_RESET}" "$*" >&2; }
log_section() { printf "\n%s==> %s%s\n" "${C_BOLD}${C_BLUE}" "$*" "${C_RESET}"; }


# ---- Preflight ------------------------------------------------------------

# Fail fast on misconfiguration. The goal: if run.sh can't possibly
# succeed, exit BEFORE spawning any children so we don't leak half-
# started processes (which stop.sh would then have to clean up).
preflight() {
    log_section "Preflight"

    if [[ ! -x "${LLAMACPP_BIN}" ]]; then
        log_error "llama-server not found or not executable at:"
        log_error "  ${LLAMACPP_BIN}"
        log_error "Fix: bash setup.sh   (or override LLAMACPP_BIN)"
        exit "${EXIT_LLAMA_MISSING}"
    fi

    if [[ ! -f "${LLM_GGUF}" ]]; then
        log_error "Model file not found at:"
        log_error "  ${LLM_GGUF}"
        log_error "Fix: bash setup.sh   (or override LLM_GGUF)"
        exit "${EXIT_MODEL_MISSING}"
    fi

    if [[ ! -x "${VENV}/bin/python" ]]; then
        log_error "Python venv not found at:"
        log_error "  ${VENV}/bin/python"
        log_error "Fix: bash setup.sh   (or override VENV)"
        exit "${EXIT_PREFLIGHT}"
    fi

    # Reject if the ports are already bound by a foreign process.
    # lsof is on macOS by default and on every Linux dev box.
    if command -v lsof >/dev/null 2>&1; then
        local llama_pids api_pids
        llama_pids="$(lsof -ti:"${LLM_PORT}" 2>/dev/null || true)"
        api_pids="$(lsof -ti:"${API_PORT}" 2>/dev/null || true)"
        if [[ -n "${llama_pids}" ]]; then
            log_error "Port ${LLM_PORT} is already in use by pid(s): ${llama_pids}"
            log_error "Fix: bash stop.sh   (or override LLM_PORT)"
            exit "${EXIT_PORT_BUSY}"
        fi
        if [[ -n "${api_pids}" ]]; then
            log_error "Port ${API_PORT} is already in use by pid(s): ${api_pids}"
            log_error "Fix: bash stop.sh   (or override API_PORT)"
            exit "${EXIT_PORT_BUSY}"
        fi
    else
        log_warn "lsof not available — skipping port-in-use check"
    fi

    # Ensure the log directory exists so the redirection in start_llama
    # / start_uvicorn doesn't fail on a fresh clone.
    mkdir -p "${LOG_DIR}"

    log_ok "llama-server: ${LLAMACPP_BIN}"
    log_ok "Model:       ${LLM_GGUF}"
    log_ok "Venv:        ${VENV}"
    log_ok "Ports:       LLM=${LLM_PORT}, API=${API_PORT}"
}


# ---- start_llama ----------------------------------------------------------

# Background llama-server, write its PID, wait for /health.
#
# Health-check idiom (curl --retry with --retry-connrefused):
#   --retry 60 --retry-delay 1 --retry-connrefused --max-time 2
#   means: poll once per second for 60 s; treat ECONNREFUSED as retryable
#   (the kernel returns this for ~1 ms after bind); cap each individual
#   request at 2 s so a hung server doesn't burn the full budget.
#   This is the canonical "wait for a server" snippet from Step 4.19's
#   SSE smoke tests + Step 3.7's smoke_test_llm.py.
start_llama() {
    log_section "Starting llama-server"
    log_info "Command: ${LLAMACPP_BIN} --model ${LLM_GGUF} --host ${LLM_HOST} --port ${LLM_PORT} --ctx-size ${LLAMA_CTX_SIZE} --threads ${LLAMA_THREADS}"
    log_info "Log: ${LLAMA_LOG}"

    # Background the process and capture its PID. `&` puts it in the
    # background of this script's process group; on EXIT INT TERM we
    # send SIGTERM explicitly so the child isn't orphaned.
    "${LLAMACPP_BIN}" \
        --model "${LLM_GGUF}" \
        --host "${LLM_HOST}" --port "${LLM_PORT}" \
        --ctx-size "${LLAMA_CTX_SIZE}" --threads "${LLAMA_THREADS}" \
        >"${LLAMA_LOG}" 2>&1 &
    LLAMA_PID=$!
    echo "${LLAMA_PID}" > "${LLAMA_PIDFILE}"
    log_ok "llama-server started (pid=${LLAMA_PID})"

    # Verify the process didn't immediately die. If it did, the log
    # file will have a useful error message and we can fail fast
    # without waiting through the health-check timeout.
    sleep 1
    if ! kill -0 "${LLAMA_PID}" 2>/dev/null; then
        log_error "llama-server exited within 1 second of start. Last log lines:"
        tail -20 "${LLAMA_LOG}" >&2 || true
        exit "${EXIT_LLAMA_CRASHED}"
    fi

    # Wait for /health. Returns 200 with {"status":"ok"} when the
    # model is fully loaded (can take 5-30 s for Phi-3 Mini).
    log_info "Waiting for llama-server /health (timeout ${HEALTH_TIMEOUT_SECONDS}s)…"
    if ! curl --silent --show-error \
              --retry "${HEALTH_TIMEOUT_SECONDS}" --retry-delay 1 \
              --retry-connrefused --max-time 2 \
              "http://${LLM_HOST}:${LLM_PORT}/health" >/dev/null; then
        log_error "llama-server failed to become healthy in ${HEALTH_TIMEOUT_SECONDS}s"
        log_error "Last 20 lines of ${LLAMA_LOG}:"
        tail -20 "${LLAMA_LOG}" >&2 || true
        exit "${EXIT_HEALTH_TIMEOUT}"
    fi
    log_ok "llama-server is healthy at http://${LLM_HOST}:${LLM_PORT}/"
}


# ---- start_uvicorn --------------------------------------------------------

# Background uvicorn, write its PID, wait for /api/status, then `wait`
# on the PID so this script blocks until uvicorn exits.
start_uvicorn() {
    log_section "Starting uvicorn"
    log_info "Command: ${VENV}/bin/python -m uvicorn tinyrag.main:app --host ${API_HOST} --port ${API_PORT}"
    log_info "Log: ${UVICORN_LOG}"

    # Background uvicorn. PYTHONPATH=src is required so the import
    # `tinyrag.main` resolves without needing an editable install
    # (pip install -e . would also work but is heavier).
    # `cd` first so uvicorn's working directory is the repo root
    # (otherwise relative paths in config.yaml would break).
    cd "${REPO_ROOT}"
    PYTHONPATH=src "${VENV}/bin/python" -m uvicorn \
        tinyrag.main:app \
        --host "${API_HOST}" --port "${API_PORT}" \
        >"${UVICORN_LOG}" 2>&1 &
    UVICORN_PID=$!
    echo "${UVICORN_PID}" > "${UVICORN_PIDFILE}"
    log_ok "uvicorn started (pid=${UVICORN_PID})"

    sleep 1
    if ! kill -0 "${UVICORN_PID}" 2>/dev/null; then
        log_error "uvicorn exited within 1 second of start. Last log lines:"
        tail -20 "${UVICORN_LOG}" >&2 || true
        exit "${EXIT_HEALTH_TIMEOUT}"
    fi

    # Wait for /api/status — the route exists at import time, so it
    # becomes reachable as soon as uvicorn binds the port + the
    # FastAPI lifespan startup completes (~1-3 s).
    log_info "Waiting for uvicorn /api/status (timeout ${HEALTH_TIMEOUT_SECONDS}s)…"
    if ! curl --silent --show-error \
              --retry "${HEALTH_TIMEOUT_SECONDS}" --retry-delay 1 \
              --retry-connrefused --max-time 2 \
              "http://${API_HOST}:${API_PORT}/api/status" >/dev/null; then
        log_error "uvicorn failed to come up in ${HEALTH_TIMEOUT_SECONDS}s"
        log_error "Last 20 lines of ${UVICORN_LOG}:"
        tail -20 "${UVICORN_LOG}" >&2 || true
        exit "${EXIT_HEALTH_TIMEOUT}"
    fi
    log_ok "uvicorn is healthy at http://${API_HOST}:${API_PORT}/"
    log_info "Chat UI:  http://${API_HOST}:${API_PORT}/"
    log_info "Admin UI: http://${API_HOST}:${API_PORT}/admin"
    log_info "Press Ctrl+C to stop both services."
}


# ---- on_exit (the cleanup trap) -------------------------------------------

# Fires on EXIT INT TERM (every exit path). Sends SIGTERM to both
# children, waits up to 5 s for graceful shutdown, escalates to SIGKILL
# if they're still alive. The `kill -0` test guards against stale PID
# files pointing at recycled PIDs.
#
# This is the critical race-condition fix: without it, Ctrl+C leaves
# llama-server running (~3 GB of RAM) while the user thinks they killed
# the stack.
on_exit() {
    local exit_code=$?
    # Only print the banner on a non-zero exit (the EXIT trap fires
    # on normal shutdown too, and we don't want to spam "Shutting
    # down" on a clean run).
    if [[ "${exit_code}" -ne 0 ]]; then
        log_info "Shutting down (exit=${exit_code})…"
    fi

    if [[ -n "${UVICORN_PID}" ]] && kill -0 "${UVICORN_PID}" 2>/dev/null; then
        kill -TERM "${UVICORN_PID}" 2>/dev/null || true
    fi

    if [[ -n "${LLAMA_PID}" ]] && kill -0 "${LLAMA_PID}" 2>/dev/null; then
        kill -TERM "${LLAMA_PID}" 2>/dev/null || true
        # Wait up to 5 s for graceful shutdown, then escalate.
        local i
        for i in 1 2 3 4 5; do
            kill -0 "${LLAMA_PID}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${LLAMA_PID}" 2>/dev/null; then
            log_warn "llama-server did not exit after 5s; sending SIGKILL"
            kill -KILL "${LLAMA_PID}" 2>/dev/null || true
        fi
    fi

    # Final wait so the children fully reap before we return. Without
    # this, the subshell can race the trap and leave zombies.
    if [[ -n "${UVICORN_PID}" ]]; then
        wait "${UVICORN_PID}" 2>/dev/null || true
    fi
    if [[ -n "${LLAMA_PID}" ]]; then
        wait "${LLAMA_PID}" 2>/dev/null || true
    fi

    # Clean up PID files. Idempotent — safe to re-run.
    rm -f "${LLAMA_PIDFILE}" "${UVICORN_PIDFILE}"

    exit "${exit_code}"
}


# ---- Main -----------------------------------------------------------------

main() {
    parse_args "$@"

    log_section "TinyRAG stack launcher (Step 4.24)"
    log_info "Repo root: ${REPO_ROOT}"
    log_info "Logs: ${LOG_DIR}"

    # Install the cleanup trap BEFORE spawning anything. From this
    # point on, any exit path (normal, INT, TERM, ERR-style failure)
    # will tear down both children.
    trap on_exit EXIT INT TERM

    preflight
    start_llama
    start_uvicorn

    # Block on uvicorn. When uvicorn exits (cleanly, or via the trap's
    # SIGTERM), `wait` returns and we fall through to the EXIT trap
    # which runs cleanup again (idempotent — second run is a no-op).
    wait "${UVICORN_PID}"
}

main "$@"