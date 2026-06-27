"""Structural tests for ``scripts/portability_check.sh`` (Step 4.20).

Why these tests exist
---------------------

``scripts/portability_check.sh`` is a 3-5 minute integration test (clone +
install + smoke + smoke-e2e + cleanup). Running it from pytest on every CI
build is impractical. Instead we test the script's **structure** — every
property that can be verified in under a second — and leave the actual
pipeline as an opt-in ``@pytest.mark.integration`` test gated by the
``RUN_PORTABILITY_INTEGRATION`` env var.

What we cover (no I/O, no network, < 2 s for the whole class):

- Script exists + has the right shebang + is executable.
- ``--help`` prints the docstring and exits 0.
- Unknown flags exit with code 2 (argparse-like).
- ``set -euo pipefail`` is set (catches a class of future regressions).
- All 7 documented exit-code constants (10-16) appear in the script source.
- The ``--keep`` flag actually skips cleanup (idempotent on a missing dir).
- The preflight stage rejects a missing ``python3``.
- The preflight stage rejects Python < 3.12.

What we deliberately do NOT cover
----------------------------------

- The clone stage's actual git clone (requires network).
- ``make install-dev`` (takes 30+ s and re-installs ~30 packages).
- The ``make smoke-e2e`` e2e response itself — this is the
  ``@pytest.mark.integration`` test below; it runs the FULL pipeline
  against the local repo (no network) and asserts exit 0 + "PASS".

Hermetic design
---------------

The unit tests build a fake ``PATH`` from ``tmp_path`` (with optional
``python3`` shims that print the version we want to test) so we can
exercise the preflight stage without touching the host. The integration
test sets ``TINYRAG_PORTABILITY_LOCAL_REPO`` to skip the network clone
stage and copies the local repo into a tmpdir (mirroring how a real
clone behaves).

Location: ``tests/test_portability_check.py``
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths + the script under test
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "portability_check.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_script() -> str:
    """Return the full text of the portability script."""
    return SCRIPT.read_text(encoding="utf-8")


def _run_script(
    *args: str,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess:
    """Run the portability script with the given CLI args + env.

    The integration-test code path uses ``timeout=180`` (the full
    install-dev + smoke pipeline takes ~30-60s on a warm venv).

    Returns the :class:`subprocess.CompletedProcess`. Does NOT raise
    on non-zero exit — callers assert on ``result.returncode`` /
    ``result.stdout`` / ``result.stderr``.
    """
    full_env = os.environ.copy()
    if env is not None:
        full_env.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=full_env,
        cwd=cwd or PROJECT_ROOT,
        timeout=timeout,
        check=False,
    )


def _make_fake_python_shim(tmp_path: Path, version_string: str) -> Path:
    """Create a tmpdir containing a ``python3`` shim that prints ``version_string``.

    Used to simulate "Python 3.10 on the host" so we can assert the
    preflight stage rejects it. The shim is a 2-line bash script —
    bash is in /usr/bin on every Linux + macOS dev box.

    The preflight stage invokes ``python3 --version`` and parses the
    output. To reject an "old" Python, the shim prints a version below
    3.12; the preflight's `[[ "$py_minor" -lt 12 ]]` check then fires.
    """
    shim = tmp_path / "python3"
    shim.write_text(
        textwrap.dedent(f"""\
            #!/usr/bin/env bash
            # Fake python3 shim — for tests/test_portability_check.py only.
            if [[ "$1" == "--version" ]]; then
                echo "Python {version_string}"
                exit 0
            fi
            # Any other invocation: pretend to be a working interpreter
            # (the preflight stage only calls --version).
            exit 0
        """),
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    return tmp_path


def _make_absent_python_shim(tmp_path: Path) -> Path:
    """Return a tmpdir that does NOT contain ``python3``.

    Preflight's ``command -v python3`` will fail. The shim DOES contain
    ``git`` and ``make`` (symlinked from /usr/bin) so the OTHER
    preflight checks pass — proving the python3 check is the actual
    discriminator.
    """
    shim = tmp_path / "shims"
    shim.mkdir()
    # Symlink the OTHER tools (NOT python3) so only python3 is missing.
    for tool in ("git", "make"):
        src = Path(f"/usr/bin/{tool}")
        if src.exists():
            (shim / tool).symlink_to(src)
    return shim


# ---------------------------------------------------------------------------
# Structural tests — fast, no I/O beyond reading the script
# ---------------------------------------------------------------------------


class TestPortabilityScriptStructure:
    """Verify the script's structure + invariants — no pipeline runs."""

    def test_script_exists_and_is_executable(self) -> None:
        """The script file must exist and be executable (``bash <path>``
        and the shebang line both rely on this)."""
        assert SCRIPT.exists(), f"missing: {SCRIPT}"
        assert SCRIPT.is_file()
        assert os.access(SCRIPT, os.X_OK), (
            f"{SCRIPT} is not executable — did you forget chmod +x?"
        )

    def test_script_has_correct_shebang_and_docstring(self) -> None:
        """First line is ``#!/usr/bin/env bash`` and the second is the
        docstring header comment. Both are conventions every other
        ``scripts/*.sh`` follows."""
        text = _read_script()
        first_line = text.splitlines()[0]
        assert first_line == "#!/usr/bin/env bash", (
            f"unexpected shebang: {first_line!r}"
        )
        # Docstring marker — match the install_system_deps.sh convention
        # of "====" under the title line.
        assert "# TinyRAG — Portability Self-Test" in text
        assert "# ============================================================================" in text

    def test_script_uses_set_euo_pipefail(self) -> None:
        """Strict mode. Catches a class of regressions where a future
        refactor accidentally relaxes it (e.g. ``set +e`` for one
        command, then forgets to set -e again)."""
        text = _read_script()
        # Look for the canonical strict-mode line. Allow leading
        # whitespace + comment lines between the shebang and the
        # ``set`` line, but the ``set`` line must appear uncommented
        # somewhere early in the file (before any function defs).
        strict_mode_re = re.compile(r"^set -euo pipefail\s*$", re.MULTILINE)
        match = strict_mode_re.search(text)
        assert match is not None, "missing strict mode: 'set -euo pipefail'"

        # Must appear before the first function definition (function
        # definitions start with `name() {`).
        first_fn = text.find("() {")
        assert first_fn == -1 or match.start() < first_fn, (
            "set -euo pipefail must come BEFORE any function definitions"
        )

    def test_script_exit_code_constants_are_documented(self) -> None:
        """All 7 exit codes (10-16) must appear as named constants in
        the script so the documented contract is grep-able + the
        integration test below can rely on them."""
        text = _read_script()
        for code in (10, 11, 12, 13, 14, 15, 16):
            # Each code should appear as both a docstring line and a
            # `readonly EXIT_*=NN` assignment.
            assert f" {code} " in text or f" {code}  " in text, (
                f"exit code {code} is not documented in the script header"
            )
            assert f"={code}" in text, (
                f"exit code {code} has no `readonly EXIT_*={code}` assignment"
            )

        # And the OK exit code:
        assert "EXIT_OK=0" in text

    def test_help_flag_prints_usage_and_exits_0(self) -> None:
        """``--help`` prints the script's docstring header and exits 0."""
        result = _run_script("--help")
        assert result.returncode == 0, (
            f"--help should exit 0; got {result.returncode}\nstderr:\n{result.stderr}"
        )
        # The docstring's "WHAT IT DOES" section header should be in stdout.
        assert "WHAT IT DOES" in result.stdout
        assert "EXIT CODES" in result.stdout
        # ``--help`` should print SOMETHING substantive (not just a newline).
        assert len(result.stdout) > 500, (
            "--help output is suspiciously short — did the docstring get truncated?"
        )

    def test_unknown_flag_exits_with_code_2(self) -> None:
        """``--bogus`` is rejected like argparse's bad-args exit code 2.
        This is the convention every other ``scripts/*.py`` follows
        (``scripts/ingest.py``, ``scripts/ask.py``)."""
        result = _run_script("--bogus-flag-12345")
        assert result.returncode == 2, (
            f"--bogus should exit 2 (argparse convention); got {result.returncode}\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )
        # The error message should name the offending flag so the user
        # can see what went wrong.
        assert "--bogus-flag-12345" in result.stderr

    def test_preflight_fails_without_python3(self, tmp_path: Path) -> None:
        """If ``python3`` is not on ``$PATH``, preflight exits 10.

        Builds a sealed tmpdir containing symlinks for ``git`` + ``make``
        + ``bash`` (NOT python3). Uses the absolute ``/usr/bin/bash``
        as the script interpreter (since bash itself is not on the
        sealed PATH — the shebang ``#!/usr/bin/env bash`` can't resolve
        bash otherwise).
        """
        shim_dir = _make_absent_python_shim(tmp_path)
        bash_abs = "/usr/bin/bash"
        if not Path(bash_abs).exists():
            pytest.skip(f"{bash_abs} not found on this host")

        # TINYRAG_PORTABILITY_LOCAL_REPO points at a missing dir so
        # stage 2 (clone) would fail too — BUT preflight runs FIRST,
        # so we expect exit 10 (preflight), not 11 (clone).
        result = subprocess.run(
            [bash_abs, str(SCRIPT)],
            capture_output=True,
            text=True,
            env={
                "PATH": str(shim_dir),  # only git + make + bash symlinks; no python3
                "HOME": os.environ.get("HOME", "/tmp"),
                "TINYRAG_PORTABILITY_LOCAL_REPO": "/nonexistent/local-repo-path",
            },
            cwd=PROJECT_ROOT,
            timeout=10,
            check=False,
        )
        assert result.returncode == 10, (
            f"missing python3 should exit 10 (preflight); got {result.returncode}\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )
        assert "python3" in result.stderr.lower(), (
            f"error message should name python3 as the missing tool:\n"
            f"stderr: {result.stderr!r}"
        )

    def test_preflight_fails_with_old_python_version(self, tmp_path: Path) -> None:
        """A python3 that prints ``Python 3.10.0`` must fail preflight
        (project requires 3.12+).

        Path setup: tmpdir contains a fake python3 (prints 3.10.0) +
        symlinks to git/make/bash from /usr/bin. PATH is the tmpdir +
        a fallback to /bin for `env` (used by the shebang).
        """
        shim_dir = tmp_path / "shims"
        shim_dir.mkdir()
        # Fake python3 that reports 3.10.0 (below the 3.12 minimum).
        _make_fake_python_shim(shim_dir, "3.10.0")
        for tool in ("git", "make", "bash"):
            src = Path(f"/usr/bin/{tool}")
            if src.exists():
                (shim_dir / tool).symlink_to(src)
            elif Path(f"/bin/{tool}").exists():
                (shim_dir / tool).symlink_to(Path(f"/bin/{tool}"))

        bash_abs = "/usr/bin/bash"
        if not Path(bash_abs).exists():
            pytest.skip(f"{bash_abs} not found on this host")

        result = subprocess.run(
            [bash_abs, str(SCRIPT)],
            capture_output=True,
            text=True,
            env={
                # PATH: shim dir FIRST so our fake python3 wins, then
                # /bin so `env` (needed by the shebang) resolves.
                "PATH": f"{shim_dir}:/bin",
                "HOME": os.environ.get("HOME", "/tmp"),
                "TINYRAG_PORTABILITY_LOCAL_REPO": "/nonexistent/local-repo-path",
            },
            cwd=PROJECT_ROOT,
            timeout=10,
            check=False,
        )
        assert result.returncode == 10, (
            f"Python 3.10 should fail preflight (need 3.12+); "
            f"got exit {result.returncode}\nstderr: {result.stderr!r}"
        )
        # The error message should mention both the required and found
        # versions so the user can diagnose quickly.
        assert "3.12" in result.stderr, (
            f"error should mention the required 3.12:\n{result.stderr!r}"
        )
        assert "3.10" in result.stderr, (
            f"error should mention the found 3.10:\n{result.stderr!r}"
        )

    def test_preflight_passes_on_this_machine(self) -> None:
        """Sanity gate: on a working dev box (this CI machine), preflight
        passes. This is the contrapositive of the two negative tests
        above — if this fails, the negative tests aren't actually
        exercising the preflight logic.

        Point TINYRAG_PORTABILITY_LOCAL_REPO at a NON-EXISTENT path so
        the pipeline exits FAST at the clone stage (exit 11). The point
        is to confirm preflight (exit 10) did NOT fire — proving the
        negative tests above are actually exercising preflight.
        """
        result = _run_script(
            env={
                "TINYRAG_PORTABILITY_LOCAL_REPO": "/nonexistent/path/should/not/exist",
            },
            timeout=10,
        )
        # We expect exit 11 (clone failure — local repo path missing).
        # We do NOT expect exit 10 (preflight failure) — that would
        # mean the two negative tests above aren't actually testing
        # the preflight logic.
        assert result.returncode != 10, (
            f"preflight failed on this host — the negative tests "
            f"would be meaningless:\nstdout: {result.stdout!r}\n"
            f"stderr: {result.stderr!r}"
        )
        # And we expect the clone stage's error message to name the
        # missing local repo path so the user can see why it failed.
        assert result.returncode == 11, (
            f"expected exit 11 (clone — local repo missing); got {result.returncode}\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )
        assert "/nonexistent/path/should/not/exist" in result.stderr

    def test_cleanup_is_idempotent_on_missing_dir(self) -> None:
        """``cleanup`` must be a no-op (exit 0) when the clone dir
        doesn't exist. This is what makes the ERR trap safe to call
        cleanup on every failure path.

        Triggers cleanup-on-missing by pointing the local-repo path at
        a nonexistent dir. The clone stage fails fast (exit 11), the
        ERR trap fires, and cleanup is invoked — but cleanup must not
        itself error (which would pollute /tmp on every future failure).
        """
        # Pre-clean any leftover from previous runs so the assertion
        # below has a clean baseline.
        leftover = Path("/tmp/tinyrag-test-nonexistent-xyz")
        if leftover.exists():
            shutil.rmtree(leftover)

        result = _run_script(
            env={
                "TINYRAG_PORTABILITY_CLONE_DIR": str(leftover),
                "TINYRAG_PORTABILITY_LOCAL_REPO": "/nonexistent/cleanup-test-path",
            },
            timeout=10,
        )
        # The clone stage fails (exit 11), the ERR trap runs cleanup.
        # Cleanup must succeed (no EXIT_CLEANUP=16 from rm -rf on a
        # missing dir). The final exit code will be 11 (the original
        # clone failure), not 16.
        assert result.returncode != 16, (
            f"cleanup() errored on a missing clone dir — this means "
            f"the ERR trap will leave /tmp polluted on every failure:\n"
            f"exit={result.returncode}\nstderr: {result.stderr!r}"
        )
        # Verify the original failure (exit 11 = clone failure) is what
        # surfaced, not a downstream error from a botched cleanup.
        assert result.returncode == 11, (
            f"expected exit 11 (clone failure → cleanup was a no-op); "
            f"got {result.returncode}\nstderr: {result.stderr!r}"
        )

        # Clean up after ourselves — the script's own cleanup should
        # have removed the clone dir it created, but be defensive.
        if leftover.exists():
            shutil.rmtree(leftover)


# ---------------------------------------------------------------------------
# Integration test — opt-in via RUN_PORTABILITY_INTEGRATION=1
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPortabilityIntegration:
    """End-to-end pipeline test. Opt-in because it actually runs
    ``make install-dev`` (~30-60 s) + the e2e smoke (~3 s).

    Run with::

        RUN_PORTABILITY_INTEGRATION=1 PYTHONPATH=src \\
            ~/venvs/tinyrag/bin/python -m pytest \\
            tests/test_portability_check.py::TestPortabilityIntegration -v

    Uses ``TINYRAG_PORTABILITY_LOCAL_REPO`` so we don't need network.
    Runs the pipeline against the current working copy of the repo,
    which is the same code the script tests.
    """

    def test_full_pipeline_passes_against_local_repo(self) -> None:
        """Run the full pipeline against the local repo (no network).
        Asserts exit 0 and that the PASS summary is in stdout.

        Skips automatically when ``RUN_PORTABILITY_INTEGRATION`` is not
        set in the environment — this protects the default `pytest`
        run from paying the ~45 s cost on every CI invocation.
        """
        if os.environ.get("RUN_PORTABILITY_INTEGRATION") != "1":
            pytest.skip(
                "set RUN_PORTABILITY_INTEGRATION=1 to run the full "
                "portability pipeline (takes ~30-60s)"
            )

        # Use a tmpdir CLONE_DIR so we don't pollute /tmp if anything
        # goes wrong mid-run. The script will rm -rf it on success.
        # No apostrophe in the path — bash quoting + `make -C` interact
        # badly with `'` in the directory name (verified empirically).
        clone_dir = "/tmp/tinyrag-portability-py-test-runner"
        result = _run_script(
            env={
                "TINYRAG_PORTABILITY_LOCAL_REPO": str(PROJECT_ROOT),
                "TINYRAG_PORTABILITY_CLONE_DIR": clone_dir,
            },
            timeout=600,  # 10 min — install-dev can be slow on a cold cache
        )
        assert result.returncode == 0, (
            f"full pipeline failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\n\n"
            f"stderr:\n{result.stderr}"
        )
        assert "PASS" in result.stdout, (
            f"expected 'PASS' in stdout:\n{result.stdout}"
        )
        assert "7/7" in result.stdout, (
            f"expected '7/7 stages' in stdout:\n{result.stdout}"
        )

        # And the cleanup must have removed the clone dir.
        assert not Path(clone_dir).exists(), (
            f"clone dir {clone_dir} was not cleaned up after success"
        )

    def test_keep_flag_preserves_clone_dir(self) -> None:
        """``--keep`` should skip the final cleanup so the user can
        inspect the clone for post-mortem debugging."""
        if os.environ.get("RUN_PORTABILITY_INTEGRATION") != "1":
            pytest.skip(
                "set RUN_PORTABILITY_INTEGRATION=1 to run the full "
                "portability pipeline (takes ~30-60s)"
            )

        # No apostrophe — see test_full_pipeline_passes_against_local_repo
        # for the why.
        clone_dir = "/tmp/tinyrag-portability-py-test-keep-runner"
        try:
            result = _run_script(
                "--keep",
                env={
                    "TINYRAG_PORTABILITY_LOCAL_REPO": str(PROJECT_ROOT),
                    "TINYRAG_PORTABILITY_CLONE_DIR": clone_dir,
                },
                timeout=600,
            )
            assert result.returncode == 0
            # --keep must have left the clone dir behind.
            assert Path(clone_dir).exists(), (
                f"--keep should preserve {clone_dir} but it was removed"
            )
        finally:
            # Clean up ourselves — the script didn't because of --keep.
            if Path(clone_dir).exists():
                shutil.rmtree(clone_dir)
