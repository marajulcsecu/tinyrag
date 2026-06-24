#!/usr/bin/env bash
# ============================================================================
# TinyRAG — Build llama.cpp with OpenBLAS (laptop target)
# ----------------------------------------------------------------------------
# Clones llama.cpp (if needed), checks out a pinned commit, and builds it
# with OpenBLAS acceleration for the LAPTOP (x86_64, Ubuntu 24.04).
#
# USAGE
#   bash scripts/build_llamacpp.sh                  # full build (clone+cmake+compile)
#   bash scripts/build_llamacpp.sh --skip-clone     # reuse existing ./llama.cpp
#   bash scripts/build_llamacpp.sh --clean          # nuke build/ first
#   bash scripts/build_llamacpp.sh --check         # only verify (no build)
#   bash scripts/build_llamacpp.sh --help           # show this header
#
# PINNED COMMIT
#   See docs/05_tech_stack_v1.md §3.2 and docs/BUILDS.md §2.1.
#   We pin to a commit, not a tag — llama.cpp's tags are infrequent and
#   `master` is unstable.
#
# BUILD TIME
#   - Cold (no cache): ~5-10 min on i5-1235U with -j 10
#   - Warm (incremental): ~10-30 s for a single file change
#   - After --clean: same as cold
#
# REFERENCES
#   - docs/06_roadmap_v2.md Step 3.4 (this step)
#   - docs/BUILDS.md (build manifest, verification commands)
#   - llama.cpp docs: https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md
# ============================================================================


# ---- Safety flags ---------------------------------------------------------
set -euo pipefail


# ---- Configuration --------------------------------------------------------

# Pinned llama.cpp version. See docs/BUILDS.md §2.1 for the resolved SHA.
# We pin to a STABLE FORMAT TAG (gguf-vX.Y.Z) rather than a commit hash.
# Format tags are released roughly monthly with explicit version bumps and
# are guaranteed to be reproducible. We update them when we want to pick
# up llama.cpp improvements; between updates, every clone produces a
# byte-identical binary.
#
# History:
#   - docs/05_tech_stack_v1.md §3.2 originally suggested pinning by commit
#     hash, but in practice llama.cpp's tag namespace is the stable surface.
LLAMACPP_PINNED_TAG="gguf-v0.19.0"
LLAMACPP_REPO="https://github.com/ggerganov/llama.cpp.git"
LLAMACPP_DIR="llama.cpp"
LLAMACPP_BUILD_DIR="${LLAMACPP_DIR}/build"

# Absolute path of the project root (parent of this script's directory).
# Used to safely cd back before referencing relative paths.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Number of parallel build jobs. Default: physical core count (not threads).
# On i5-1235U this is 10 (12 logical). -j 10 leaves 2 threads for the OS.
JOBS="${JOBS:-10}"

# Build type: Release = optimized, no debug symbols, smallest binary.
BUILD_TYPE="Release"


# ---- Pretty output --------------------------------------------------------
if [[ -t 1 ]]; then
    C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'
    C_BLUE=$'\033[34m'; C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'
else
    C_RESET=""; C_BOLD=""; C_BLUE=""; C_GREEN=""; C_YELLOW=""; C_RED=""
fi
log_info()    { printf "%s[INFO]%s  %s\n" "${C_BLUE}"    "${C_RESET}" "$*"; }
log_ok()      { printf "%s[ OK ]%s  %s\n" "${C_GREEN}"   "${C_RESET}" "$*"; }
log_warn()    { printf "%s[WARN]%s  %s\n" "${C_YELLOW}"  "${C_RESET}" "$*" >&2; }
log_error()   { printf "%s[ERR ]%s  %s\n" "${C_RED}"     "${C_RESET}" "$*" >&2; }
log_section() { printf "\n%s==> %s%s\n" "${C_BOLD}${C_BLUE}" "$*" "${C_RESET}"; }


# ---- Argument parsing -----------------------------------------------------
MODE="build"
SKIP_CLONE="false"
CLEAN_BUILD="false"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-clone) SKIP_CLONE="true"; shift ;;
        --clean)      CLEAN_BUILD="true"; shift ;;
        --check)      MODE="check"; shift ;;
        --jobs)       JOBS="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,35p' "$0"
            exit 0 ;;
        *) log_error "Unknown arg: $1"; exit 2 ;;
    esac
done


# ---- Pre-flight -----------------------------------------------------------
log_section "Pre-flight"
command -v cmake >/dev/null   || { log_error "cmake not found. Run: bash scripts/install_system_deps.sh"; exit 1; }
command -v git   >/dev/null   || { log_error "git not found."; exit 1; }
pkg-config --exists openblas  || { log_error "OpenBLAS not found. Run: bash scripts/install_system_deps.sh"; exit 1; }
log_ok "cmake: $(cmake --version | head -1)"
log_ok "git:   $(git --version)"
log_ok "OpenBLAS: $(pkg-config --modversion openblas)"
log_ok "Build jobs: ${JOBS}"
log_info "Pinned tag: ${LLAMACPP_PINNED_TAG}"


# ---- Colon-in-path workaround --------------------------------------------
# GNU Make pattern rules cannot contain ':' in target names. If the project
# lives at a path with a colon (e.g. "TinyRAG: Retrieval-Augmented .../"),
# the auto-generated CMake Makefiles produced inside the build dir will fail
# to parse. We detect this and divert the build OUT of the project tree
# (where Make would choke on the colon), then symlink the binary back into
# the project so the rest of the toolchain finds it at the expected path.
#
# **Why ${HOME}/.cache/llamacpp-build and not /tmp?**
# ``/tmp`` is volatile — it's wiped on reboot, by ``tmpreaper``, and
# on many systems by routine maintenance. Putting the build in the
# user's ``XDG cache dir`` (``$HOME/.cache``) makes it survive
# reboots, so the user doesn't have to re-run a 7-minute compile
# after a power cycle. The first migration from ``/tmp`` to
# ``$HOME/.cache`` is recorded in the build journal as "Step 3.4a"
# (see AGENT.md §11.1).
EXTERNAL_BUILD_DIR=""
EXTERNAL_SRC_DIR=""
if [[ "$PROJECT_ROOT" == *:* ]]; then
    log_warn "Project path contains ':' — diverting build out of project tree"
    log_warn "(GNU Make cannot parse Makefiles with colons in target names)"
    # ``${HOME}/.cache`` is the XDG cache home on Linux. We don't use
    # ``XDG_CACHE_HOME`` because the script is sometimes sourced
    # from non-login shells where the env var may be unset; falling
    # back to ``$HOME/.cache`` is the standard default.
    EXTERNAL_BUILD_PARENT="${HOME}/.cache/llamacpp-build"
    EXTERNAL_BUILD_DIR="${EXTERNAL_BUILD_PARENT}/build"
    EXTERNAL_SRC_DIR="${EXTERNAL_BUILD_PARENT}"
fi


# ---- Resolve source and build directories --------------------------------
# When the project path contains a colon (GNU Make limitation), the source
# tree AND build dir live under ``$HOME/.cache/llamacpp-build/``, which
# is structured as:
#   ${HOME}/.cache/llamacpp-build/         <- llama.cpp source (cloned here)
#   ${HOME}/.cache/llamacpp-build/build/    <- cmake build dir
# Otherwise both live in the project at llama.cpp/ and llama.cpp/build/.
if [[ -n "$EXTERNAL_SRC_DIR" ]]; then
    SRC_DIR="$EXTERNAL_SRC_DIR"
    SRC_PARENT="$(dirname "$EXTERNAL_SRC_DIR")"
    BUILD_DIR="$EXTERNAL_BUILD_DIR"
else
    SRC_DIR="${PROJECT_ROOT}/${LLAMACPP_DIR}"
    SRC_PARENT="$PROJECT_ROOT"
    BUILD_DIR="${PROJECT_ROOT}/${LLAMACPP_BUILD_DIR}"
fi


# ---- Clone llama.cpp ------------------------------------------------------
if [[ "$SKIP_CLONE" == "true" ]]; then
    if [[ ! -d "$SRC_DIR" ]]; then
        log_error "--skip-clone set but ${SRC_DIR} does not exist."
        exit 1
    fi
    log_info "Skipping clone (using existing ${SRC_DIR})"
else
    log_section "Cloning llama.cpp"
    if [[ -d "$SRC_DIR" ]]; then
        log_info "${SRC_DIR} already exists — verifying remote"
        cd "$SRC_DIR"
        if ! git remote get-url origin >/dev/null 2>&1; then
            log_error "Existing ${SRC_DIR} has no origin remote."
            exit 1
        fi
        log_info "Fetching latest refs (incl. tags)..."
        git fetch --depth=1 --tags origin master 2>&1 | tail -3
        cd ..
    else
        mkdir -p "$SRC_PARENT"
        log_info "git clone --depth=1 --tags ${LLAMACPP_REPO} ${SRC_DIR}"
        git clone --depth=1 --tags "$LLAMACPP_REPO" "$SRC_DIR"
    fi
fi


# ---- Checkout pinned version ---------------------------------------------
log_section "Checking out pinned version (${LLAMACPP_PINNED_TAG})"
cd "$SRC_DIR"

# Tags live in the ref namespace; we already fetched them with --tags above.
if ! git rev-parse "$LLAMACPP_PINNED_TAG" >/dev/null 2>&1; then
    log_warn "Tag ${LLAMACPP_PINNED_TAG} not in shallow history — fetching it."
    git fetch --depth=1 origin "refs/tags/${LLAMACPP_PINNED_TAG}:refs/tags/${LLAMACPP_PINNED_TAG}" 2>&1 | tail -3
fi

git checkout "$LLAMACPP_PINNED_TAG" 2>&1
ACTUAL_SHA=$(git rev-parse HEAD)
log_ok "HEAD is now ${ACTUAL_SHA}"
log_ok "On tag:   ${LLAMACPP_PINNED_TAG}"


# ---- Clean (optional) -----------------------------------------------------
if [[ "$CLEAN_BUILD" == "true" ]]; then
    log_section "Cleaning previous build"
    rm -rf "$BUILD_DIR"
    log_ok "Removed ${BUILD_DIR}"
fi


# ---- Init submodules ------------------------------------------------------
log_section "Initializing submodules"
if [[ -f .gitmodules ]]; then
    git submodule update --init --recursive --depth=1 2>&1 | tail -5
    log_ok "Submodules ready"
else
    log_info "No .gitmodules — skipping"
fi


# ---- Symlink binary back into project (colon-path workaround) ----------
# When we built in /tmp, expose the binary inside the project so the rest
# of the toolchain (Makefile, scripts, run.sh) can find it at the expected
# location. We symlink the bin/ directory so all companion .so files are
# also available (llama-server needs them next to itself to dlopen).
if [[ -n "$EXTERNAL_BUILD_DIR" ]]; then
    log_section "Symlinking binaries into project"
    mkdir -p "${PROJECT_ROOT}/${LLAMACPP_DIR}"
    # Remove any pre-existing bin/ inside the project llama.cpp/ (shouldn't
    # exist on a fresh tree, but be safe if rerun).
    rm -rf "${PROJECT_ROOT}/${LLAMACPP_DIR}/bin"
    ln -s "$BUILD_DIR/bin" "${PROJECT_ROOT}/${LLAMACPP_DIR}/bin"
    log_ok "Symlinked: ${PROJECT_ROOT}/${LLAMACPP_DIR}/bin -> ${BUILD_DIR}/bin"
    # Also create a build/ symlink for any tooling that looks for it.
    rm -rf "${PROJECT_ROOT}/${LLAMACPP_DIR}/build"
    ln -s "$BUILD_DIR" "${PROJECT_ROOT}/${LLAMACPP_DIR}/build"
    log_ok "Symlinked: ${PROJECT_ROOT}/${LLAMACPP_DIR}/build -> ${BUILD_DIR}"
fi


# ---- Check-only mode ------------------------------------------------------
if [[ "$MODE" == "check" ]]; then
    log_section "Check mode — verifying existing build"
    LLAMA_SERVER_PATH="${BUILD_DIR}/bin/llama-server"
    if [[ ! -x "$LLAMA_SERVER_PATH" ]]; then
        log_error "llama-server binary not found or not executable at $LLAMA_SERVER_PATH."
        exit 1
    fi
    log_ok "llama-server exists at $LLAMA_SERVER_PATH"
    if ldd "$LLAMA_SERVER_PATH" | grep -q openblas; then
        log_ok "OpenBLAS is linked"
    else
        log_warn "OpenBLAS does NOT appear to be linked — rebuild with --clean"
    fi
    exit 0
fi


# ---- Configure (cmake) ----------------------------------------------------
log_section "Configuring with cmake"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DGGML_BLAS=ON
    -DGGML_BLAS_VENDOR=OpenBLAS
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=ON
    -DLLAMA_CURL=OFF
)
log_info "cmake ${CMAKE_ARGS[*]} .."
cmake "${CMAKE_ARGS[@]}" "$SRC_DIR" 2>&1 | tail -15


# ---- Build ----------------------------------------------------------------
log_section "Building llama.cpp (this takes 5-10 min on i5-1235U)"
cd "$BUILD_DIR"
cmake --build . --config "$BUILD_TYPE" -j "$JOBS" 2>&1 | tail -30


# ---- Post-build verification ---------------------------------------------
log_section "Post-build verification"

LLAMA_SERVER="${BUILD_DIR}/bin/llama-server"

if [[ ! -x "$LLAMA_SERVER" ]]; then
    log_error "llama-server binary not found at $LLAMA_SERVER"
    log_error "Build may have failed — see output above."
    exit 1
fi
log_ok "llama-server exists and is executable"
ls -la "$LLAMA_SERVER" | awk '{print "      "$5" bytes, "$1" "$9}'

# Verify OpenBLAS linkage.
if ldd "$LLAMA_SERVER" 2>/dev/null | grep -q openblas; then
    OPENBLAS_LINK=$(ldd "$LLAMA_SERVER" | grep openblas)
    log_ok "OpenBLAS linked: ${OPENBLAS_LINK}"
else
    log_error "OpenBLAS NOT linked! Build fell back to generic BLAS."
    log_error "Try: bash scripts/build_llamacpp.sh --clean"
    exit 1
fi


# ---- Summary --------------------------------------------------------------
log_section "Build summary"
log_ok "Source dir:  ${SRC_DIR}"
log_ok "Build dir:   ${BUILD_DIR}"
log_ok "Pinned tag:  ${LLAMACPP_PINNED_TAG}"
log_ok "Actual SHA:  ${ACTUAL_SHA}"
log_ok "Binary:      ${LLAMA_SERVER}"
log_ok "OpenBLAS:    $(pkg-config --modversion openblas)"
log_ok "Build type:  ${BUILD_TYPE}"
log_ok "Jobs:        ${JOBS}"
if [[ -n "$EXTERNAL_BUILD_DIR" ]]; then
    log_ok "Symlink:     ${PROJECT_ROOT}/${LLAMACPP_DIR}/bin -> ${BUILD_DIR}/bin"
    log_info "(colon-path workaround: real binaries live in ${BUILD_DIR})"
fi
log_info "Next step: bash scripts/download_models.py --model phi3-mini"
log_info "Or:        make run-llm  (after a model is downloaded)"

echo ""
echo "==================================================================="
echo " RECORD THIS IN docs/BUILDS.md §2.1:"
echo "   Pinned tag:        ${LLAMACPP_PINNED_TAG}"
echo "   Actual SHA:        ${ACTUAL_SHA}"
echo "   Build date:        $(date -u +%Y-%m-%d)"
echo "   OpenBLAS version:  $(pkg-config --modversion openblas)"
if [[ -n "$EXTERNAL_BUILD_DIR" ]]; then
echo "   Build location:    ${BUILD_DIR}  (colon-path workaround)"
fi
echo "==================================================================="
