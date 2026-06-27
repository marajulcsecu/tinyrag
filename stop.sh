#!/usr/bin/env bash
# ============================================================================
# TinyRAG — One-Command Stack Shutdown (Step 4.24)
# ----------------------------------------------------------------------------
# Tears down whatever `bash run.sh` (or any other launch path) left
# behind: llama-server, uvicorn, or both. Idempotent — returns exit 0
# even if nothing was running.
#
# USAGE
#   bash stop.sh                  # stop everything (safe to re-run)
#   bash stop.sh --help           # show this header
#
# WHY PID FILES + lsof (both, not either)?
#   - PID files (logs/llama-server.pid, logs/uvicorn.pid) are fast and
#     reliable when run.sh started the process normally. They cover the
#     common case ("I ran run.sh, now I want to stop").
#   - lsof -ti:<port> catches the failure modes where PID files are
#     useless:
#       (a) the user `kill -9`'d run.sh directly — the PID file is
#           gone, but llama-server + uvicorn are still bound to their
#           ports.
#       (b) run.sh crashed BEFORE writing the PID file (extremely rare
#           in practice, but the dual source makes it free to handle).
#       (c) the user ran llama-server + uvicorn manually (e.g. via
#           `make run-llm` + `make run-api`) without run.sh — there
#           are no PID files, but the ports are bound.
#   - We union both sources and dedupe, so the same process is never
#     killed twice.
#
# WHY TERM → KILL ESCALATION?
#   - SIGTERM lets llama-server flush its model + close sockets
#     gracefully (~1-2 s for llama.cpp; faster for uvicorn).
#   - If the process ignores SIGTERM (a hung C++ loop, a wedged
#     Python thread), we escalate to SIGKILL after 5 s.
#   - 5 s is short enough that a stuck Ctrl+C feels responsive; long
#     enough that llama.cpp's normal shutdown completes.
#
# EXIT CODES
#    0  Always — stop.sh is idempotent. "Nothing to stop" is success.
#   2   Argument parse error (e.g. --bogus).
#
# WHY ALWAYS EXIT 0?
#   stop.sh is meant to be safe in CI cleanup hooks, Makefile recipes,
#   and shell pipelines. A non-zero exit from "nothing was running"
#   would break all of those. If you need to know whether anything
#   was actually stopped, check the log output.
#
# REFERENCES
#   - docs/06_roadmap_v2.md Step 4.24 (this step)
#   - run.sh (the sibling that writes the PID files this script reads)
# ============================================================================


# ---- Safety flags ---------------------------------------------------------

# `set -e`: exit on any error.
# `set -u`: error on undefined variables.
# `set -o pipefail`: catch failures in piped commands.
set -euo pipefail


# ---- Configuration --------------------------------------------------------

# Where the repo lives. Resolve via ${BASH_SOURCE[0]} so stop.sh works
# even when invoked from a different cwd.
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# All paths/ports match run.sh exactly. Override via env vars for
# non-standard setups.
readonly LOG_DIR="${REPO_ROOT}/logs"
readonly LLAMA_PIDFILE="${LOG_DIR}/llama-server.pid"
readonly UVICORN_PIDFILE="${LOG_DIR}/uvicorn.pid"
readonly LLM_PORT="${LLM_PORT:-8080}"
readonly API_PORT="${API_PORT:-8000}"

# How long to wait between SIGTERM and SIGKILL. 5 s is short enough
# to feel responsive; long enough for llama.cpp's normal shutdown.
readonly SIGTERM_GRACE_SECONDS=5

# Exit-code constants. Even though stop.sh currently only has EXIT_OK,
# defining the constants in the same shape as setup.sh / run.sh makes
# the three scripts structurally consistent (and grep-able for tests).
readonly EXIT_OK=0
readonly EXIT_OK_NOOP=0  # alias — explicit "nothing to stop" is still success


# ---- CLI argument parsing -------------------------------------------------

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
                echo "Try: bash stop.sh --help" >&2
                exit 2
                ;;
        esac
        shift
    done
}


# ---- Pretty output helpers ------------------------------------------------

# TTY-aware ANSI colours (auto-disabled when piped). Matches setup.sh
# + run.sh so the three scripts look consistent.
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

log_info()    { printf "%s[stop]%s   %s\n" "${C_BLUE}"   "${C_RESET}" "$*"; }
log_ok()      { printf "%s[  OK ]%s  %s\n" "${C_GREEN}"  "${C_RESET}" "$*"; }
log_warn()    { printf "%s[WARN ]%s  %s\n" "${C_YELLOW}" "${C_RESET}" "$*" >&2; }
log_error()   { printf "%s[FAIL ]%s  %s\n" "${C_RED}"    "${C_RESET}" "$*" >&2; }


# ---- PID discovery --------------------------------------------------------

# Build the union of PIDs to kill, deduped, filtered for empties.
# Sources (in order):
#   1. PID file for llama-server
#   2. PID file for uvicorn
#   3. lsof -ti:<LLM_PORT>  (catches llama-server bound to :8080)
#   4. lsof -ti:<API_PORT>  (catches uvicorn bound to :8000)
#
# Output: one PID per line on stdout. Caller uses `mapfile` to read
# into an array.
collect_pids() {
    local pid
    local -a pids=()

    if [[ -f "${LLAMA_PIDFILE}" ]]; then
        pid="$(cat "${LLAMA_PIDFILE}" 2>/dev/null || true)"
        [[ -n "${pid}" ]] && pids+=("${pid}")
    fi
    if [[ -f "${UVICORN_PIDFILE}" ]]; then
        pid="$(cat "${UVICORN_PIDFILE}" 2>/dev/null || true)"
        [[ -n "${pid}" ]] && pids+=("${pid}")
    fi

    if command -v lsof >/dev/null 2>&1; then
        local port_pids
        port_pids="$(lsof -ti:"${LLM_PORT}" 2>/dev/null || true)"
        if [[ -n "${port_pids}" ]]; then
            # lsof outputs space-separated when multiple PIDs match.
            # Read into the array via read -ra.
            local -a extra=()
            read -ra extra -d '' <<<"${port_pids}" || true
            pids+=("${extra[@]}")
        fi
        port_pids="$(lsof -ti:"${API_PORT}" 2>/dev/null || true)"
        if [[ -n "${port_pids}" ]]; then
            local -a extra=()
            read -ra extra -d '' <<<"${port_pids}" || true
            pids+=("${extra[@]}")
        fi
    fi

    # Dedupe + drop empties. printf | awk is the canonical bash-3-portable
    # dedupe idiom (no `mapfile -t` since this code path needs to also
    # work on bash 3.x macOS, although our scripts target bash ≥4).
    if [[ ${#pids[@]} -eq 0 ]]; then
        return 0
    fi
    printf '%s\n' "${pids[@]}" | awk 'NF && !seen[$0]++'
}


# Label a PID by which service it likely belongs to. Purely cosmetic
# — the kill logic is the same regardless. Used to make the log
# output readable.
label_for_pid() {
    local pid=$1
    if [[ -f "${LLAMA_PIDFILE}" ]] && [[ "${pid}" == "$(cat "${LLAMA_PIDFILE}" 2>/dev/null)" ]]; then
        printf "llama-server"
        return 0
    fi
    if [[ -f "${UVICORN_PIDFILE}" ]] && [[ "${pid}" == "$(cat "${UVICORN_PIDFILE}" 2>/dev/null)" ]]; then
        printf "uvicorn"
        return 0
    fi
    # Fall back to "port XXXX" so the user can see which service.
    if command -v lsof >/dev/null 2>&1; then
        local port
        port="$(lsof -nP -p "${pid}" 2>/dev/null | awk '/TCP/ {print $9; exit}')"
        if [[ -n "${port}" ]]; then
            printf "process on %s" "${port}"
            return 0
        fi
    fi
    printf "process"
}


# Send SIGTERM, wait up to SIGTERM_GRACE_SECONDS for graceful shutdown,
# escalate to SIGKILL if still alive. Idempotent — returns 0 whether
# the process was alive or not.
stop_one() {
    local pid=$1
    local label=$2

    # kill -0 tests whether the PID exists AND we can signal it (no
    # permissions error). If it returns non-zero, the process is
    # already gone — treat as success.
    if ! kill -0 "${pid}" 2>/dev/null; then
        log_info "pid ${pid} (${label}) already gone"
        return 0
    fi

    log_info "Sending SIGTERM to ${label} (pid ${pid})…"
    kill -TERM "${pid}" 2>/dev/null || true

    local i
    for i in $(seq 1 "${SIGTERM_GRACE_SECONDS}"); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            log_ok "${label} stopped"
            return 0
        fi
        sleep 1
    done

    # Still alive after grace period — escalate.
    log_warn "${label} did not exit after ${SIGTERM_GRACE_SECONDS}s; sending SIGKILL"
    kill -KILL "${pid}" 2>/dev/null || true
    # Give the kernel a moment to reap.
    sleep 0.2
    if kill -0 "${pid}" 2>/dev/null; then
        log_warn "${label} (pid ${pid}) still alive after SIGKILL — kernel will reap on exit"
    fi
}


# ---- Main -----------------------------------------------------------------

main() {
    parse_args "$@"

    log_info "Stopping TinyRAG stack…"

    # Collect + dedupe PIDs.
    local -a pids=()
    local pid
    while IFS= read -r pid; do
        [[ -n "${pid}" ]] && pids+=("${pid}")
    done < <(collect_pids)

    if [[ ${#pids[@]} -eq 0 ]]; then
        log_info "Nothing to stop (no PID files or port-bound processes found)."
        # Idempotent: clean stale PID files defensively even if no
        # processes are alive.
        rm -f "${LLAMA_PIDFILE}" "${UVICORN_PIDFILE}"
        exit "${EXIT_OK_NOOP}"
    fi

    log_info "Found ${#pids[@]} candidate process(es):"
    for pid in "${pids[@]}"; do
        local label
        label="$(label_for_pid "${pid}")"
        log_info "  pid ${pid} (${label})"
    done

    # Kill each one (kill is idempotent — if a PID was already
    # removed from the table between collect_pids and stop_one, the
    # `kill -0` check at the top of stop_one is a no-op).
    for pid in "${pids[@]}"; do
        local label
        label="$(label_for_pid "${pid}")"
        stop_one "${pid}" "${label}"
    done

    # Clean PID files regardless of whether stop_one killed the
    # process (the process might have been a foreign one with no
    # matching PID file — that's fine, rm -f is idempotent).
    rm -f "${LLAMA_PIDFILE}" "${UVICORN_PIDFILE}"

    log_ok "Stack stopped."
    exit "${EXIT_OK}"
}

main "$@"