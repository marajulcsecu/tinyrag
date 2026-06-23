#!/usr/bin/env bash
# ============================================================================
# TinyRAG — Install System Dependencies
# ----------------------------------------------------------------------------
# Installs the apt packages that llama.cpp (and other native tools) need to
# build and run on Ubuntu 24.04 LTS / Debian 12 (Bookworm)+.
#
# USAGE
#   bash scripts/install_system_deps.sh             # install (idempotent)
#   bash scripts/install_system_deps.sh --check     # only verify, no install
#   bash scripts/install_system_deps.sh --help      # show this header
#
# IDEMPOTENCY
#   The script detects which packages are already installed and skips them.
#   Safe to run multiple times.
#
# SUDO
#   `apt-get install` needs root. The script auto-detects sudo. If you're
#   already root (e.g., in a container) it skips the sudo prefix.
#
# ROLLBACK
#   We use `apt-get install --no-install-recommends` to keep the dep tree
#   small. To remove everything this script installed, see the rollback
#   recipe at the bottom of this file.
#
# REFERENCES
#   - docs/05_tech_stack_v1.md §6.1 (system package list)
#   - docs/06_roadmap_v2.md Step 3.3 (this step)
#   - llama.cpp build docs: https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md
# ============================================================================


# ---- Safety flags ---------------------------------------------------------

# `set -e`: exit on any error (don't continue past a failed apt-get).
# `set -u`: error on undefined variables.
# `set -o pipefail`: catch failures in piped commands, not just the last one.
set -euo pipefail


# ---- Configuration --------------------------------------------------------

# Package list, grouped by purpose. Comments explain WHY each one is needed.
#
# Build toolchain (required for compiling llama.cpp from source)
PKG_BUILD="build-essential cmake git"

# Math libraries (give llama.cpp a 2x speedup on x86_64; required on Pi for SIMD)
PKG_MATH="libopenblas-dev liblapack-dev"

# Utilities (used by setup, debugging, runtime data inspection)
PKG_UTIL="sqlite3 tree"

# Optional packages — installed only if --with-extras is passed.
# Kept separate so the default install is minimal.
PKG_EXTRAS="pkg-config ninja-build"

# All "required" packages in one variable for easy apt-get invocation.
REQUIRED_PACKAGES="${PKG_BUILD} ${PKG_MATH} ${PKG_UTIL}"


# ---- Pretty output helpers ------------------------------------------------

# Only enable colors when output is a terminal (so logs stay clean).
if [[ -t 1 ]]; then
    C_RESET=$'\033[0m'
    C_BOLD=$'\033[1m'
    C_BLUE=$'\033[34m'
    C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'
    C_RED=$'\033[31m'
else
    C_RESET="" C_BOLD="" C_BLUE="" C_GREEN="" C_YELLOW="" C_RED=""
fi

log_info()    { printf "%s[INFO]%s  %s\n" "${C_BLUE}"    "${C_RESET}" "$*"; }
log_ok()      { printf "%s[ OK ]%s  %s\n" "${C_GREEN}"   "${C_RESET}" "$*"; }
log_warn()    { printf "%s[WARN]%s  %s\n" "${C_YELLOW}"  "${C_RESET}" "$*" >&2; }
log_error()   { printf "%s[ERR ]%s  %s\n" "${C_RED}"     "${C_RESET}" "$*" >&2; }
log_section() { printf "\n%s==> %s%s\n" "${C_BOLD}${C_BLUE}" "$*" "${C_RESET}"; }


# ---- Sudo handling --------------------------------------------------------

# If we're already root (uid 0), skip sudo. Otherwise prepend sudo to apt-get.
if [[ "$EUID" -eq 0 ]]; then
    SUDO=""
    log_info "Running as root (no sudo needed)."
else
    if ! command -v sudo >/dev/null 2>&1; then
        log_error "sudo is not installed and we're not root. Re-run as root or install sudo."
        exit 1
    fi
    SUDO="sudo"
    log_info "Will use sudo for apt-get."
fi


# ---- Argument parsing -----------------------------------------------------

MODE="install"            # default action
WITH_EXTRAS="false"       # --with-extras flag

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)
            MODE="check"
            shift
            ;;
        --with-extras)
            WITH_EXTRAS="true"
            shift
            ;;
        --help|-h)
            sed -n '2,30p' "$0"      # print the header comment
            exit 0
            ;;
        *)
            log_error "Unknown argument: $1"
            echo "Try: bash scripts/install_system_deps.sh --help"
            exit 2
            ;;
    esac
done


# ---- Pre-flight checks ----------------------------------------------------

log_section "Pre-flight checks"

# 1. Must be on a Debian-family distro (apt-get is required).
if ! command -v apt-get >/dev/null 2>&1; then
    log_error "This script requires apt-get (Debian/Ubuntu). Detected non-apt system."
    exit 1
fi
log_ok "apt-get found."

# 2. Detect OS (informational, not blocking).
if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    log_info "OS: ${PRETTY_NAME:-unknown}"
fi

# 3. Network sanity check (apt-get update needs it).
if ! curl -fsS --max-time 5 https://archive.ubuntu.com >/dev/null 2>&1 && \
   ! curl -fsS --max-time 5 http://archive.ubuntu.com >/dev/null 2>&1; then
    log_warn "Cannot reach Ubuntu archive. apt-get update may fail."
    log_warn "If you're offline, see docs/BUILDS.md for an offline recipe."
fi


# ---- Package classification ----------------------------------------------

# Check if a dpkg package name is installed.
is_installed() {
    dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q "install ok installed"
}

# Split the required list into MISSING and ALREADY_INSTALLED.
MISSING=()
ALREADY=()
for pkg in $REQUIRED_PACKAGES; do
    if is_installed "$pkg"; then
        ALREADY+=("$pkg")
    else
        MISSING+=("$pkg")
    fi
done

if [[ "$WITH_EXTRAS" == "true" ]]; then
    for pkg in $PKG_EXTRAS; do
        if is_installed "$pkg"; then
            ALREADY+=("$pkg")
        else
            MISSING+=("$pkg")
        fi
    done
fi


# ---- Check-only mode ------------------------------------------------------

if [[ "$MODE" == "check" ]]; then
    log_section "Check mode — verifying installed packages"
    FAIL=0
    for pkg in $REQUIRED_PACKAGES; do
        if is_installed "$pkg"; then
            log_ok "$pkg"
        else
            log_error "$pkg — MISSING"
            FAIL=1
        fi
    done
    # Verify OpenBLAS specifically (most common llama.cpp build issue).
    if pkg-config --exists openblas 2>/dev/null; then
        log_ok "pkg-config knows about OpenBLAS: $(pkg-config --modversion openblas)"
    else
        log_error "pkg-config cannot find openblas.pc (libopenblas-dev may be misconfigured)"
        FAIL=1
    fi
    if [[ $FAIL -eq 0 ]]; then
        log_ok "All required system dependencies are installed."
        exit 0
    else
        log_error "Some dependencies are missing. Run without --check to install them."
        exit 1
    fi
fi


# ---- Install mode ---------------------------------------------------------

log_section "Package inventory"

log_info "Already installed (${#ALREADY[@]}): ${ALREADY[*]:-none}"
log_info "Need to install  (${#MISSING[@]}): ${MISSING[*]:-none}"


if [[ ${#MISSING[@]} -eq 0 ]]; then
    log_ok "All required packages already present — nothing to do."
    log_ok "Verified via: pkg-config --libs openblas"
    pkg-config --libs openblas || true
    exit 0
fi


# ---- apt-get update + install --------------------------------------------

log_section "Running apt-get update"
$SUDO apt-get update


log_section "Installing missing packages"
log_info "Packages: ${MISSING[*]}"

# --no-install-recommends: keep the dep tree small; we don't need suggested extras.
# -y: assume yes (this script is non-interactive).
$SUDO apt-get install -y --no-install-recommends "${MISSING[@]}"


# ---- Post-install verification --------------------------------------------

log_section "Post-install verification"

# Verify each newly-installed package is actually present.
VERIFY_FAIL=0
for pkg in "${MISSING[@]}"; do
    if is_installed "$pkg"; then
        log_ok "$pkg installed"
    else
        log_error "$pkg STILL MISSING after install — something went wrong"
        VERIFY_FAIL=1
    fi
done

# Verify OpenBLAS specifically. This is the #1 thing llama.cpp needs.
if pkg-config --exists openblas; then
    OPENBLAS_VER=$(pkg-config --modversion openblas)
    OPENBLAS_LIBS=$(pkg-config --libs openblas)
    log_ok "OpenBLAS ${OPENBLAS_VER} found: ${OPENBLAS_LIBS}"
else
    log_error "pkg-config cannot locate openblas.pc"
    log_error "Check: ls /usr/lib/x86_64-linux-gnu/pkgconfig/openblas.pc"
    VERIFY_FAIL=1
fi


# ---- Final summary --------------------------------------------------------

echo ""
if [[ $VERIFY_FAIL -eq 0 ]]; then
    log_section "All system dependencies installed successfully."
    log_ok "Next step: bash scripts/build_llamacpp.sh  (or: make build)"
    exit 0
else
    log_section "Install completed but verification FAILED."
    log_error "See messages above. Do NOT proceed to llama.cpp build until fixed."
    exit 1
fi


# ============================================================================
# ROLLBACK (manual, not automated — too dangerous to run automatically)
# ----------------------------------------------------------------------------
# If you ever need to undo what this script installed:
#
#   sudo apt-get remove --purge \
#       build-essential cmake git \
#       libopenblas-dev liblapack-dev \
#       sqlite3 tree pkg-config ninja-build
#   sudo apt-get autoremove
#
# DO NOT actually run this unless you know what you're doing — removing
# build-essential / git / cmake will break your ability to develop anything.
# ============================================================================
