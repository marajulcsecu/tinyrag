#!/usr/bin/env python3
"""Verify the llama.cpp build is correct, portable, and matches our pin.

This script is the single source of truth for "is the LLM runtime ready?".
It runs the same checks every time and exits non-zero if anything is wrong.

Usage
-----
    python scripts/verify_llamacpp.py                # full check
    python scripts/verify_llamacpp.py --json         # machine-readable output
    python scripts/verify_llamacpp.py --quiet        # only print failures

Checks performed
----------------
1. llama.cpp source tree exists at the expected path.
2. llama.cpp HEAD matches the pinned commit (or the override passed in).
3. The build directory exists and was produced by cmake.
4. llama-server binary exists and is executable.
5. llama-server is linked against OpenBLAS (not just generic BLAS).
6. llama-server --version reports a sane version string.
7. llama-server can be invoked (--help exits 0).

Exit codes
----------
0 = all checks passed
1 = one or more checks failed
2 = script setup error (missing tools, wrong directory)

Why Python and not bash?
------------------------
- Subprocess + pathlib makes the checks more portable than bash ldd parsing.
- JSON output lets us use this in CI without parsing shell strings.
- The check functions are small, testable, and self-documenting.

Author: TinyRAG (auto-generated as part of Step 3.4)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths & pinned version (kept in sync with docs/BUILDS.md §2.1)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLAMACPP_DIR = PROJECT_ROOT / "llama.cpp"
LLAMACPP_BUILD_DIR = LLAMACPP_DIR / "build"
LLAMACPP_BIN = LLAMACPP_BUILD_DIR / "bin" / "llama-server"

# Colon-in-path workaround: if the project path contains a colon, the real
# build is in /tmp/llamacpp-build. Try /tmp first when the in-project path
# is missing AND the project path looks like a colon-path. Otherwise default
# to the in-project path.
def _resolve_actual_paths() -> tuple[Path, Path, Path]:
    """Return (src_dir, build_dir, bin) — handling the colon-path case.

    When the project lives at e.g. "TinyRAG: .../foo", GNU Make can't
    parse the auto-generated Makefiles, so the build is diverted to
    /tmp/llamacpp-build/. The source tree is also cloned there
    (under /tmp/llamacpp-build/ itself, not /tmp/llamacpp-build/llama.cpp).
    We check that location first if the project path has a colon, then
    fall back to the in-project path.
    """
    if ":" in str(PROJECT_ROOT):
        # /tmp/llamacpp-build/ IS the source tree (cloned at /tmp/llamacpp-build)
        # with build/ as a subdirectory. The project's llama.cpp/ has build/
        # and bin/ symlinked into it.
        if (Path("/tmp/llamacpp-build") / ".git").exists():
            src = Path("/tmp/llamacpp-build")
            build = Path("/tmp/llamacpp-build/build")
            bin_ = build / "bin" / "llama-server"
            return src, build, bin_
    return LLAMACPP_DIR, LLAMACPP_BUILD_DIR, LLAMACPP_BIN

# Pinned version. Update both this and docs/BUILDS.md §2.1 together.
# We pin to a format tag (gguf-vX.Y.Z) rather than a commit hash. Tags are
# the stable surface of llama.cpp's release history and are immutable.
PINNED_TAG = "gguf-v0.19.0"

# Allow override from CLI for local debugging.
COMMIT_OVERRIDE: str | None = None


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """The result of a single verification check."""

    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_source_tree() -> CheckResult:
    """Verify llama.cpp/ exists and is a git working tree."""
    src_dir, _, _ = _resolve_actual_paths()
    if not src_dir.exists():
        return CheckResult(
            name="source_tree",
            passed=False,
            message=f"llama.cpp source tree not found at {src_dir}",
        )
    if not (src_dir / ".git").exists():
        return CheckResult(
            name="source_tree",
            passed=False,
            message=f"{src_dir} exists but is not a git repository",
        )
    return CheckResult(
        name="source_tree",
        passed=True,
        message=f"llama.cpp source tree present at {src_dir}",
    )


def check_pinned_commit() -> CheckResult:
    """Verify llama.cpp HEAD is checked out at the pinned tag."""
    src_dir, _, _ = _resolve_actual_paths()
    if not src_dir.exists():
        return CheckResult(
            name="pinned_commit",
            passed=False,
            message="Cannot check — source tree missing",
        )

    expected_tag = COMMIT_OVERRIDE or PINNED_TAG

    # Resolve the tag to a SHA so we can compare against HEAD.
    # For annotated tags we need the peeled SHA; for lightweight tags the
    # tag SHA is the same as the commit SHA. `git rev-parse TAG^{commit}`
    # always returns the commit SHA in both cases.
    try:
        tag_sha_result = subprocess.run(
            ["git", "rev-parse", f"{expected_tag}" + "^{commit}"],
            cwd=src_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        tag_sha = tag_sha_result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return CheckResult(
            name="pinned_commit",
            passed=False,
            message=f"Tag {expected_tag} not found locally — fetch with --tags",
            details={"error": str(exc)},
        )

    try:
        head_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=src_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        actual_sha = head_result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return CheckResult(
            name="pinned_commit",
            passed=False,
            message=f"git rev-parse HEAD failed: {exc}",
        )

    passed = actual_sha == tag_sha
    return CheckResult(
        name="pinned_commit",
        passed=passed,
        message=(
            f"HEAD ({actual_sha[:12]}) is at pinned tag {expected_tag}"
            if passed
            else f"HEAD {actual_sha[:12]} != {expected_tag} ({tag_sha[:12]})"
        ),
        details={"actual_sha": actual_sha, "tag": expected_tag, "tag_sha": tag_sha},
    )


def check_build_dir() -> CheckResult:
    """Verify the cmake build directory exists and has CMakeCache.txt."""
    _, build_dir, _ = _resolve_actual_paths()
    if not build_dir.exists():
        return CheckResult(
            name="build_dir",
            passed=False,
            message=f"Build directory {build_dir} not found — run build_llamacpp.sh",
        )
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return CheckResult(
            name="build_dir",
            passed=False,
            message=f"{build_dir} exists but CMakeCache.txt missing — re-run cmake",
        )
    return CheckResult(
        name="build_dir",
        passed=True,
        message=f"Build directory present at {build_dir}",
    )


def check_binary() -> CheckResult:
    """Verify llama-server binary exists and is executable."""
    _, _, bin_path = _resolve_actual_paths()
    if not bin_path.exists():
        return CheckResult(
            name="binary",
            passed=False,
            message=f"llama-server not found at {bin_path} — run build_llamacpp.sh",
        )
    if not os.access(bin_path, os.X_OK):
        return CheckResult(
            name="binary",
            passed=False,
            message=f"{bin_path} exists but is not executable",
        )
    size = bin_path.stat().st_size
    return CheckResult(
        name="binary",
        passed=True,
        message=f"llama-server is executable ({size:,} bytes)",
        details={"size_bytes": size, "path": str(bin_path)},
    )


def check_openblas_linkage() -> CheckResult:
    """Verify llama-server is dynamically linked against OpenBLAS.

    This is the most important check. Without OpenBLAS, inference runs
    ~2x slower on x86_64 because llama.cpp falls back to reference BLAS.
    """
    _, _, bin_path = _resolve_actual_paths()
    if not bin_path.exists():
        return CheckResult(
            name="openblas_link",
            passed=False,
            message="Cannot check — binary missing",
        )

    ldd = shutil.which("ldd")
    if not ldd:
        return CheckResult(
            name="openblas_link",
            passed=False,
            message="ldd not found — cannot check linkage (non-Linux system?)",
        )

    try:
        result = subprocess.run(
            [ldd, str(bin_path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return CheckResult(
            name="openblas_link",
            passed=False,
            message=f"ldd failed: {exc}",
        )

    openblas_lines = [ln for ln in result.stdout.splitlines() if "openblas" in ln.lower()]
    if not openblas_lines:
        return CheckResult(
            name="openblas_link",
            passed=False,
            message="OpenBLAS is NOT linked — rebuild with GGML_BLAS=ON GGML_BLAS_VENDOR=OpenBLAS",
            details={"ldd_output": result.stdout},
        )
    return CheckResult(
        name="openblas_link",
        passed=True,
        message="OpenBLAS is dynamically linked",
        details={"openblas_line": openblas_lines[0]},
    )


def check_version_string() -> CheckResult:
    """Verify llama-server --version produces a sane string."""
    _, _, bin_path = _resolve_actual_paths()
    if not bin_path.exists():
        return CheckResult(
            name="version",
            passed=False,
            message="Cannot check — binary missing",
        )

    try:
        result = subprocess.run(
            [str(bin_path), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return CheckResult(
            name="version",
            passed=False,
            message=f"llama-server --version failed: {exc}",
        )

    # llama-server prints "version: <sha> (<date>)" on stdout.
    version_output = (result.stdout + result.stderr).strip()
    if "version" not in version_output.lower():
        return CheckResult(
            name="version",
            passed=False,
            message=f"Unexpected --version output: {version_output[:200]!r}",
            details={"raw_output": version_output},
        )
    return CheckResult(
        name="version",
        passed=True,
        message=f"Version string: {version_output.splitlines()[0][:120]}",
        details={"raw_output": version_output},
    )


def check_help_exits_zero() -> CheckResult:
    """Verify llama-server --help exits 0 (catches broken/missing libs)."""
    _, _, bin_path = _resolve_actual_paths()
    if not bin_path.exists():
        return CheckResult(
            name="help_exit_code",
            passed=False,
            message="Cannot check — binary missing",
        )

    try:
        result = subprocess.run(
            [str(bin_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return CheckResult(
            name="help_exit_code",
            passed=False,
            message=f"llama-server --help raised exception: {exc}",
        )

    if result.returncode != 0:
        return CheckResult(
            name="help_exit_code",
            passed=False,
            message=f"llama-server --help exited with code {result.returncode}",
            details={
                "stderr_tail": result.stderr[-500:],
                "stdout_tail": result.stdout[-500:],
            },
        )
    return CheckResult(
        name="help_exit_code",
        passed=True,
        message="llama-server --help exits 0",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


ALL_CHECKS = [
    check_source_tree,
    check_pinned_commit,
    check_build_dir,
    check_binary,
    check_openblas_linkage,
    check_version_string,
    check_help_exits_zero,
]


def run_all() -> list[CheckResult]:
    """Run every check and return the list of results."""
    return [check() for check in ALL_CHECKS]


def print_human(results: list[CheckResult], *, quiet: bool = False) -> None:
    """Pretty-print results to stdout."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\nllama.cpp verification: {passed}/{total} checks passed\n")
    for r in results:
        symbol = "[ OK ]" if r.passed else "[FAIL]"
        # Use ANSI colors only on TTY.
        color = "\033[32m" if r.passed else "\033[31m"
        reset = "\033[0m" if sys.stdout.isatty() else ""
        print(f"  {color}{symbol}{reset} {r.name:<20} {r.message}")
    print()
    if passed != total:
        print("Some checks FAILED. See messages above.")
    else:
        print("All checks passed — llama.cpp is ready to serve.")


def main() -> int:
    """Parse CLI args, run checks, return exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON to stdout (for CI).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print failed checks (suppress passes).",
    )
    parser.add_argument(
        "--commit",
        default=None,
        help="Override the expected pinned tag (for local debugging only).",
    )
    args = parser.parse_args()

    global COMMIT_OVERRIDE
    if args.commit:
        COMMIT_OVERRIDE = args.commit

    results = run_all()
    failed = [r for r in results if not r.passed]

    if args.json:
        payload = {
            "passed": len(failed) == 0,
            "total": len(results),
            "failed": len(failed),
            "checks": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details,
                }
                for r in results
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        if args.quiet:
            for r in failed:
                print(f"[FAIL] {r.name}: {r.message}")
            if not failed:
                print("All checks passed.")
        else:
            print_human(results)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
