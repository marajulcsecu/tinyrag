#!/usr/bin/env bash
# ============================================================================
# TinyRAG — Portability Self-Test (Step 4.20)
# ----------------------------------------------------------------------------
# Proves the repo can be cloned to a fresh directory, installed from scratch,
# and runs end-to-end — the laptop-side "works on a fresh machine" gate we
# don't want to discover we fail on the Pi the night before the demo.
#
# USAGE
#   bash scripts/portability_check.sh              # run the full pipeline
#   bash scripts/portability_check.sh --keep       # skip the final cleanup
#                                                  # (useful for post-mortem)
#   bash scripts/portability_check.sh --yes        # auto-overwrite the clone
#                                                  # dir if it already exists
#   bash scripts/portability_check.sh --quiet      # suppress per-stage chatter
#   bash scripts/portability_check.sh --help       # show this header
#
# WHAT IT DOES (7 stages; each prints its name + duration)
#   1. Preflight    Verify Python 3.12+, git, make, /tmp writable.
#   2. Clone        `git clone --depth 1` into /tmp/tinyrag-portability-test/.
#   3. install-dev  `make VENV=$CLONE_DIR/.venv install-dev` (hermetic venv
#                   inside the clone so cleanup is a single rm -rf).
#   4. smoke        `make smoke` — the import-only sanity check (Python +
#                   every dep group imports cleanly).
#   5. smoke-e2e    `make smoke-e2e E2E_CLIENT=fake` — sends a real query
#                   through the LLMClient; uses FakeLLMClient so no llama-
#                   server is needed. This is the actual "does it answer?"
#                   gate.
#   6. assert       The smoke response is non-empty (>=5 chars, not just
#                   whitespace). The canned FakeLLM response is ~80 chars,
#                   so 5 is a generous floor that still catches a hard
#                   silent failure.
#   7. cleanup      `rm -rf /tmp/tinyrag-portability-test/` unless --keep
#                   was passed.
#
# WHY E2E_CLIENT=fake (not `real`)
#   The real client needs a live llama-server + a downloaded model. That
#   turns this into a 2-hour, sudo-required, network-heavy rehearsal — useful
#   exactly once before Phase 6 (Pi deploy), not every time you want to check
#   the laptop setup works. Fake mode is hermetic: no network, no sudo, no
#   2 GB downloads, runs in ~3-5 min on a laptop.
#
# EXIT CODES
#    0  PASS — all 7 stages succeeded.
#   10  Preflight failed (missing tool / wrong Python / /tmp not writable).
#   11  Clone failed (network down, repo URL wrong, etc.).
#   12  install-dev failed (pip install error — usually requirements.txt drift).
#   13  smoke failed (import error — usually a missing dependency).
#   14  smoke-e2e failed (the FakeLLMClient itself errored — bug in
#       scripts/smoke_test.py or LLMClient).
#   15  Empty-response assertion failed (smoke ran but printed nothing).
#   16  Cleanup failed (rm -rf errored — usually a permission issue).
#
# IDEMPOTENCY
#   Re-running the script with --yes will overwrite the existing clone dir.
#   Without --yes, it prompts first. The clone is always shallow (`--depth 1`)
#   so re-clones are fast (~3 s on a typical broadband connection).
#
# REFERENCES
#   - docs/06_roadmap_v2.md Step 4.20 (this step)
#   - docs/01_project_scope_v2.md §Reproducible ("setup.sh and run.sh bring
#     up the entire system from scratch")
# ============================================================================


# ---- Safety flags ---------------------------------------------------------

# `set -e`: exit on any error.
# `set -u`: error on undefined variables (catches typos in $CLONE_DIR etc).
# `set -o pipefail`: catch failures in piped commands, not just the last one.
set -euo pipefail


# ---- Configuration --------------------------------------------------------

# Where the clone lives. Overridable for branch testing / CI isolation:
#   TINYRAG_PORTABILITY_CLONE_DIR=/tmp/foo bash scripts/portability_check.sh
: "${TINYRAG_PORTABILITY_CLONE_DIR:=/tmp/tinyrag-portability-test}"

# Repository to clone. Overridable for fork testing:
#   TINYRAG_PORTABILITY_REPO_URL=https://github.com/me/myfork.git bash ...
: "${TINYRAG_PORTABILITY_REPO_URL:=https://github.com/marajulcsecu/tinyrag.git}"

# Branch to clone. Defaults to `main`.
: "${TINYRAG_PORTABILITY_BRANCH:=main}"

# If set, skip the clone step entirely and use the repo at this local path.
# Used by the integration test in tests/test_portability_check.py so the
# pytest suite doesn't need to hit the network.
: "${TINYRAG_PORTABILITY_LOCAL_REPO:=}"

# Minimum supported Python version. Matches Makefile + requirements.txt.
readonly MIN_PYTHON_MAJOR=3
readonly MIN_PYTHON_MINOR=12

# Exit-code constants. Centralised so tests can grep for them and so
# future refactors don't silently break the documented contract.
readonly EXIT_OK=0
readonly EXIT_PREFLIGHT=10
readonly EXIT_CLONE=11
readonly EXIT_INSTALL=12
readonly EXIT_SMOKE=13
readonly EXIT_SMOKE_E2E=14
readonly EXIT_EMPTY_RESPONSE=15
readonly EXIT_CLEANUP=16


# ---- CLI argument parsing -------------------------------------------------

FLAG_KEEP="false"     # --keep: skip cleanup
FLAG_YES="false"      # --yes: auto-overwrite existing clone dir
FLAG_QUIET="false"    # --quiet: suppress per-stage info logs

print_help() {
    sed -n '2,68p' "$0"
    exit "${EXIT_OK}"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --keep)   FLAG_KEEP="true";   shift ;;
            --yes)    FLAG_YES="true";    shift ;;
            --quiet)  FLAG_QUIET="true";  shift ;;
            --help|-h) print_help ;;
            *)
                log_error "Unknown argument: $1"
                echo "Try: bash scripts/portability_check.sh --help" >&2
                exit 2
                ;;
        esac
    done
}


# ---- Pretty output helpers ------------------------------------------------

# Only enable colors when output is a terminal (so logs stay clean in CI).
if [[ -t 1 ]]; then
    C_RESET=$'\033[0m'
    C_BOLD=$'\033[1m'
    C_BLUE=$'\033[34m'
    C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'
    C_RED=$'\033[31m'
    C_DIM=$'\033[2m'
else
    C_RESET="" C_BOLD="" C_BLUE="" C_GREEN="" C_YELLOW="" C_RED="" C_DIM=""
fi

log_info()    { if [[ "${FLAG_QUIET}" != "true" ]]; then
                    printf "%s[stage]%s  %s\n" "${C_BLUE}"    "${C_RESET}" "$*"
                fi; }
log_ok()      { printf "%s[ OK ]%s  %s\n"   "${C_GREEN}"   "${C_RESET}" "$*"; }
log_warn()    { printf "%s[WARN]%s  %s\n"   "${C_YELLOW}"  "${C_RESET}" "$*" >&2; }
log_error()   { printf "%s[FAIL]%s  %s\n"   "${C_RED}"     "${C_RESET}" "$*" >&2; }
log_section() { printf "\n%s==> %s%s\n"     "${C_BOLD}${C_BLUE}" "$*" "${C_RESET}"; }


# ---- Stage timing helper --------------------------------------------------

# Each stage prints "OK (3.2s)" on success and "FAIL" on error. Captures
# the elapsed seconds via SECONDS (a bash builtin that auto-increments).
declare -a STAGE_NAMES=()    # human-readable name per stage (for summary)
declare -a STAGE_DURS=()     # elapsed seconds per stage (for summary)

time_stage() {
    # Print the stage banner + run the function passed as $1.
    # On success: append to STAGE_NAMES / STAGE_DURS for the summary.
    local stage_name="$1"; shift
    local stage_num="$1"; shift
    local stage_total="$1"; shift
    local stage_fn="$1"; shift

    log_section "[stage ${stage_num}/${stage_total}] ${stage_name}..."
    local start="${SECONDS}"
    "${stage_fn}" "$@"
    local elapsed=$(( SECONDS - start ))
    log_ok "[stage ${stage_num}/${stage_total}] ${stage_name} (${elapsed}s)"

    STAGE_NAMES+=("${stage_name}")
    STAGE_DURS+=("${elapsed}")
}


# ---- Stage 1: Preflight ---------------------------------------------------

preflight() {
    # Verify every tool the pipeline needs is available BEFORE we clone.
    # Catching "git missing" before we burn 30 s on a partial install is
    # the whole point.

    local missing=()

    command -v git >/dev/null 2>&1    || missing+=("git")
    command -v make >/dev/null 2>&1   || missing+=("make")
    command -v python3 >/dev/null 2>&1 || missing+=("python3")
    # rsync is only required when running in local-repo mode (test mode);
    # the real clone uses `git clone` and never invokes rsync. This
    # keeps the prod dependency list minimal.
    if [[ -n "${TINYRAG_PORTABILITY_LOCAL_REPO}" ]]; then
        command -v rsync >/dev/null 2>&1 || missing+=("rsync")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing[*]}"
        log_error "Install with: sudo apt-get install -y ${missing[*]}"
        exit "${EXIT_PREFLIGHT}"
    fi

    # /tmp must be writable (we put the clone + venv there).
    if [[ ! -w /tmp ]]; then
        log_error "/tmp is not writable — cannot create ${TINYRAG_PORTABILITY_CLONE_DIR}"
        exit "${EXIT_PREFLIGHT}"
    fi

    # Python version check. `python3 --version` prints "Python 3.12.x".
    local py_version
    py_version="$(python3 --version 2>&1 | awk '{print $2}')"
    local py_major py_minor
    py_major="$(echo "${py_version}" | cut -d. -f1)"
    py_minor="$(echo "${py_version}" | cut -d. -f2)"
    if [[ "${py_major}" -lt "${MIN_PYTHON_MAJOR}" ]] \
       || { [[ "${py_major}" -eq "${MIN_PYTHON_MAJOR}" ]] \
            && [[ "${py_minor}" -lt "${MIN_PYTHON_MINOR}" ]]; }; then
        log_error "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ required, found ${py_version}"
        exit "${EXIT_PREFLIGHT}"
    fi

    log_info "Python: ${py_version}; tools: git, make, python3 — all OK"
}


# ---- Stage 2: Clone --------------------------------------------------------

clone_repo() {
    # Skip the clone entirely if TINYRAG_PORTABILITY_LOCAL_REPO is set
    # (the integration test uses this to point at the working copy).
    if [[ -n "${TINYRAG_PORTABILITY_LOCAL_REPO}" ]]; then
        local local_repo="${TINYRAG_PORTABILITY_LOCAL_REPO}"
        if [[ ! -d "${local_repo}" ]]; then
            log_error "TINYRAG_PORTABILITY_LOCAL_REPO points at a missing dir: ${local_repo}"
            exit "${EXIT_CLONE}"
        fi
        log_info "Using local repo (TINYRAG_PORTABILITY_LOCAL_REPO): ${local_repo}"

        # Mirror the local repo into CLONE_DIR via rsync so the rest of
        # the pipeline can treat it like a fresh clone. Excludes the
        # untracked data/models dirs (gitignored, would otherwise copy
        # hundreds of MB of fixture data + multi-GB GGUF models).
        rm -rf "${TINYRAG_PORTABILITY_CLONE_DIR}"
        mkdir -p "${TINYRAG_PORTABILITY_CLONE_DIR}"
        # rsync --exclude is portable across Linux + macOS (cp --exclude
        # is a GNU extension that's missing from some coreutils builds).
        rsync -a --exclude='.git' --exclude='data' --exclude='models' \
              --exclude='logs' --exclude='.venv' --exclude='__pycache__' \
              --exclude='.pytest_cache' --exclude='.ruff_cache' --exclude='.mypy_cache' \
              "${local_repo}/" "${TINYRAG_PORTABILITY_CLONE_DIR}/"
        return
    fi

    # If CLONE_DIR already exists, prompt before overwriting (unless --yes).
    if [[ -e "${TINYRAG_PORTABILITY_CLONE_DIR}" ]]; then
        if [[ "${FLAG_YES}" != "true" ]]; then
            printf "%s[stage]%s  %s exists. Overwrite? [y/N] " \
                "${C_BLUE}" "${C_RESET}" "${TINYRAG_PORTABILITY_CLONE_DIR}" >&2
            read -r reply
            if [[ "${reply}" != "y" && "${reply}" != "Y" ]]; then
                log_warn "Aborted by user (clone dir not overwritten)"
                exit "${EXIT_OK}"  # clean exit — user chose not to proceed
            fi
        fi
        log_info "Removing existing clone dir"
        rm -rf "${TINYRAG_PORTABILITY_CLONE_DIR}"
    fi

    log_info "Cloning ${TINYRAG_PORTABILITY_REPO_URL} (branch ${TINYRAG_PORTABILITY_BRANCH})"
    if ! git clone --depth 1 --branch "${TINYRAG_PORTABILITY_BRANCH}" \
         "${TINYRAG_PORTABILITY_REPO_URL}" \
         "${TINYRAG_PORTABILITY_CLONE_DIR}" 2>&1 | sed "s/^/    ${C_DIM}/"; then
        log_error "git clone failed (network? wrong URL? wrong branch?)"
        exit "${EXIT_CLONE}"
    fi

    log_info "Clone: $(git -C "${TINYRAG_PORTABILITY_CLONE_DIR}" log -1 --oneline)"
}


# ---- Stage 3: install-dev --------------------------------------------------

run_install_dev() {
    # `make install-dev` creates $(VENV) if missing, then pip-installs
    # both requirements.txt and requirements-dev.txt into it.
    #
    # VENV override puts the venv INSIDE the clone dir so the cleanup
    # stage's `rm -rf` removes the venv too. The Makefile default puts
    # the venv in $HOME which would survive cleanup and clutter the
    # user's real workspace.
    local venv="${TINYRAG_PORTABILITY_CLONE_DIR}/.venv"

    log_info "Using hermetic venv at ${venv}"
    log_info "This step typically takes 20-60s on a fresh checkout"

    if ! make -C "${TINYRAG_PORTABILITY_CLONE_DIR}" VENV="${venv}" install-dev; then
        log_error "make install-dev failed (check requirements.txt for drift)"
        exit "${EXIT_INSTALL}"
    fi

    log_info "Venv created: ${venv}"
}


# ---- Stage 4: smoke (import-only) -----------------------------------------

run_smoke_imports() {
    # `make smoke` runs 5 `python -c 'import ...'` lines to verify Python
    # + every dep group imports cleanly. Cheap (~2s) and catches missing
    # packages without needing to spin up the LLM.
    log_info "Running import-only sanity check"

    if ! make -C "${TINYRAG_PORTABILITY_CLONE_DIR}" \
              VENV="${TINYRAG_PORTABILITY_CLONE_DIR}/.venv" \
              smoke; then
        log_error "make smoke failed (one of the import-only checks errored)"
        exit "${EXIT_SMOKE}"
    fi
}


# ---- Stage 5: smoke-e2e ---------------------------------------------------

run_smoke_e2e() {
    # `make smoke-e2e E2E_CLIENT=fake` invokes scripts/smoke_test.py
    # with --client fake. The script instantiates FakeLLMClient and sends
    # the canonical "What is 2+2?" probe through it.
    #
    # This is the actual "does it answer a question end-to-end?" gate.
    # If scripts/smoke_test.py OR the LLMClient module is broken, this
    # fires EXIT_SMOKE_E2E rather than silently passing.
    log_info "Running E2E smoke test with FakeLLMClient (no llama-server)"

    if ! make -C "${TINYRAG_PORTABILITY_CLONE_DIR}" \
              VENV="${TINYRAG_PORTABILITY_CLONE_DIR}/.venv" \
              smoke-e2e E2E_CLIENT=fake; then
        log_error "make smoke-e2e failed (FakeLLMClient errored)"
        exit "${EXIT_SMOKE_E2E}"
    fi
}


# ---- Stage 6: assert response non-empty ----------------------------------

assert_response_nonempty() {
    # Re-run scripts/smoke_test.py with --quiet so we get JUST the
    # response text on stdout (no banner, no token counts). Then assert
    # the text is non-empty AND has at least 5 chars AND isn't just
    # whitespace.
    #
    # Why >=5 chars: the canned FakeLLM response is ~80 chars
    # ("4. (FakeLLMClient canned response — your laptop's llama-server
    # is not being contacted, which is fine for CI.)"), so 5 is a
    # generous floor that still catches a hard silent failure (script
    # returned without printing anything, or printed a newline).
    local venv="${TINYRAG_PORTABILITY_CLONE_DIR}/.venv"
    local response
    response="$(PYTHONPATH="${TINYRAG_PORTABILITY_CLONE_DIR}/src" \
               "${venv}/bin/python" \
               "${TINYRAG_PORTABILITY_CLONE_DIR}/scripts/smoke_test.py" \
               --client fake --quiet)"

    # Strip trailing newline (--quiet always appends one; we don't
    # want that to count as "content").
    response="${response%$'\n'}"

    if [[ -z "${response}" ]]; then
        log_error "smoke-e2e returned EMPTY response (no text at all)"
        exit "${EXIT_EMPTY_RESPONSE}"
    fi

    local response_len="${#response}"
    if [[ "${response_len}" -lt 5 ]]; then
        log_error "smoke-e2e returned SUSPICIOUSLY SHORT response (${response_len} chars): ${response!r}"
        exit "${EXIT_EMPTY_RESPONSE}"
    fi

    log_info "FakeLLM response (${response_len} chars): $(echo "${response}" | head -c 60)..."
}


# ---- Stage 7: cleanup -----------------------------------------------------

cleanup() {
    # Always runs unless --keep was passed. Always runs on error too
    # (see on_error trap), so /tmp doesn't fill up after a bad run.
    if [[ "${FLAG_KEEP}" == "true" ]]; then
        log_warn "Skipping cleanup (--keep): clone kept at ${TINYRAG_PORTABILITY_CLONE_DIR}"
        return 0
    fi

    if [[ ! -e "${TINYRAG_PORTABILITY_CLONE_DIR}" ]]; then
        log_info "Clone dir already gone (nothing to clean)"
        return 0
    fi

    log_info "Removing ${TINYRAG_PORTABILITY_CLONE_DIR}"
    if ! rm -rf "${TINYRAG_PORTABILITY_CLONE_DIR}"; then
        log_error "rm -rf failed (check permissions on /tmp)"
        exit "${EXIT_CLEANUP}"
    fi

    log_info "Cleanup complete"
}


# ---- Error trap -----------------------------------------------------------

# `trap on_error ERR` runs this on any command failure. Captures the exit
# code + line number for the FAIL summary, then runs cleanup so /tmp
# doesn't fill up with half-built clones.
on_error() {
    local exit_code=$1
    local line_no=$2
    log_error "FAILED at line ${line_no} with exit code ${exit_code}"
    log_error "Diagnose with:  bash scripts/portability_check.sh --keep   # keep the clone for post-mortem"
    cleanup || true
    exit "${exit_code}"
}


# ---- Summary --------------------------------------------------------------

print_summary() {
    local total=0
    for d in "${STAGE_DURS[@]}"; do
        total=$(( total + d ))
    done

    log_section "PASS — ${#STAGE_NAMES[@]}/${#STAGE_NAMES[@]} stages in ${total}s"
    for i in "${!STAGE_NAMES[@]}"; do
        log_info "  stage $((i+1)): ${STAGE_NAMES[$i]} (${STAGE_DURS[$i]}s)"
    done
}


# ---- Main -----------------------------------------------------------------

main() {
    parse_args "$@"

    log_section "TinyRAG portability self-test (Step 4.20)"
    log_info "Clone target: ${TINYRAG_PORTABILITY_CLONE_DIR}"
    if [[ -n "${TINYRAG_PORTABILITY_LOCAL_REPO}" ]]; then
        log_info "Local-repo mode: ${TINYRAG_PORTABILITY_LOCAL_REPO}"
    fi
    log_info "Quiet: ${FLAG_QUIET}; Keep: ${FLAG_KEEP}; Yes: ${FLAG_YES}"

    # ERR trap fires for any non-zero exit. Clean up + report.
    trap 'on_error $? $LINENO' ERR

    time_stage "preflight"    1 7 preflight
    time_stage "clone"        2 7 clone_repo
    time_stage "install-dev"  3 7 run_install_dev
    time_stage "smoke"        4 7 run_smoke_imports
    time_stage "smoke-e2e"    5 7 run_smoke_e2e
    time_stage "assert"       6 7 assert_response_nonempty
    time_stage "cleanup"      7 7 cleanup

    print_summary
}

main "$@"
