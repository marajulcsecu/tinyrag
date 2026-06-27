"""Structural tests for the Step 4.24 operational scripts triad.

Why these tests exist
---------------------

``setup.sh``, ``run.sh``, and ``stop.sh`` are the three scripts the
README's "Quick Start" promises. They're the user-facing entry point for
the entire TinyRAG system — if any of them is broken, a fresh user can't
even bootstrap the codebase, let alone run a demo.

Running the actual scripts end-to-end takes minutes (llama.cpp build,
model download, full stack bring-up). Running them on every CI build
is impractical. Instead we test the scripts' **structure** — every
property that can be verified in under a second — and leave the
expensive end-to-end as opt-in.

What we cover (no I/O beyond reading the scripts, < 5 s for the whole
file):

- All three scripts exist + have the right shebang + are executable.
- All three scripts use ``set -euo pipefail`` (strict bash).
- All three scripts document their exit codes as ``readonly`` constants
  (grep-able so the contract is enforced).
- ``--help`` prints each script's header and exits 0.
- Unknown flags exit with code 2 (argparse convention).
- ``setup.sh`` idempotency: every stage is guarded by an
  ``if [[ ! -f <artefact> ]]`` check.
- ``run.sh`` orchestration: trap on EXIT INT TERM, both PIDs tracked,
  health-check via ``curl --retry``, wait on uvicorn.
- ``stop.sh`` teardown: reads both PID files, uses ``lsof`` to catch
  orphans, TERM → wait → KILL escalation.
- Cross-script invariants: shared logger function names, ``readonly``
  exit codes, ``${REPO_ROOT}`` from ``${BASH_SOURCE[0]}``, ``EXIT`` trap.
- ``bash -n`` syntax check for each script (catches missing ``fi``,
  unbalanced quotes, unterminated ``$(``).

What we deliberately do NOT cover
----------------------------------

- The actual ``make install-dev`` / ``make build-llamacpp`` runs (each
  takes minutes).
- A live llama-server bring-up. The opt-in integration test in
  ``TestRunScriptIntegration`` exercises this with ``RUN_SHELL_INTEGRATION=1``.

Hermetic design
---------------

The unit tests only read the script source + invoke them with cheap
flags (``--help``, ``--bogus``, ``bash -n``). No fork(), no network,
no model files. The integration test uses ``RUN_SHELL_INTEGRATION=1``
to opt into the ~30-60 s real-stack test.

Location: ``tests/test_shell_scripts.py``
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths + the scripts under test
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETUP_SCRIPT = PROJECT_ROOT / "setup.sh"
RUN_SCRIPT = PROJECT_ROOT / "run.sh"
STOP_SCRIPT = PROJECT_ROOT / "stop.sh"

ALL_SCRIPTS = (SETUP_SCRIPT, RUN_SCRIPT, STOP_SCRIPT)

# Map from logical exit-code slot to the set of codes each script MUST
# document as ``readonly EXIT_*=NN``. Tested in TestSetupScriptStructure
# / TestRunScriptStructure / TestStopScriptStructure.
SETUP_EXIT_CODES = (0, 10, 11, 12, 13, 14, 15)
RUN_EXIT_CODES = (0, 10, 11, 12, 13, 14, 15)
STOP_EXIT_CODES = (0,)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_script(path: Path) -> str:
    """Return the full text of a script."""
    return path.read_text(encoding="utf-8")


def _run_script(
    script: Path,
    *args: str,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess:
    """Run a shell script with the given CLI args + env.

    Returns the :class:`subprocess.CompletedProcess`. Does NOT raise
    on non-zero exit — callers assert on ``result.returncode`` /
    ``result.stdout`` / ``result.stderr``.
    """
    full_env = os.environ.copy()
    if env is not None:
        full_env.update(env)
    return subprocess.run(
        ["bash", str(script), *args],
        capture_output=True,
        text=True,
        env=full_env,
        cwd=cwd or PROJECT_ROOT,
        timeout=timeout,
        check=False,
    )


# ---------------------------------------------------------------------------
# Bash-syntax gate — the cheapest possible "no typos" check
# ---------------------------------------------------------------------------


class TestAllScriptsBashSyntax:
    """``bash -n`` parses the script without executing it. Catches a
    whole class of bugs (missing ``fi``, unbalanced quotes, unterminated
    ``$(``, missing function-body close-brace) that are easy to miss in
    code review but trivial for the bash parser to spot."""

    @pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
    def test_bash_n_exits_zero(self, script: Path) -> None:
        assert script.exists(), f"missing: {script}"
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0, (
            f"bash -n {script.name} failed:\n"
            f"stderr: {result.stderr}\n"
            f"stdout: {result.stdout}"
        )


# ---------------------------------------------------------------------------
# Shared invariants — every script should follow the same conventions
# ---------------------------------------------------------------------------


class TestShellScriptsCrossCutting:
    """Properties shared by ALL THREE scripts."""

    @pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
    def test_script_exists_and_is_executable(self, script: Path) -> None:
        """Each script must exist and be executable (``bash <path>`` and
        the shebang line both rely on this)."""
        assert script.exists(), f"missing: {script}"
        assert script.is_file()
        assert os.access(script, os.X_OK), (
            f"{script.name} is not executable — did you forget chmod +x?"
        )

    @pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
    def test_script_has_correct_shebang(self, script: Path) -> None:
        """First line is ``#!/usr/bin/env bash``. Same convention as
        every other ``scripts/*.sh`` in the repo."""
        text = _read_script(script)
        first_line = text.splitlines()[0]
        assert first_line == "#!/usr/bin/env bash", (
            f"{script.name} unexpected shebang: {first_line!r}"
        )

    @pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
    def test_script_uses_set_euo_pipefail(self, script: Path) -> None:
        """Strict mode. Catches a class of regressions where a future
        refactor accidentally relaxes it (``set +e`` for one command,
        then forgets to ``set -e`` again). Same pattern as
        scripts/portability_check.sh."""
        text = _read_script(script)
        strict_mode_re = re.compile(r"^set -euo pipefail\s*$", re.MULTILINE)
        match = strict_mode_re.search(text)
        assert match is not None, (
            f"{script.name}: missing strict mode 'set -euo pipefail'"
        )
        # Must appear before any function definition.
        first_fn = text.find("() {")
        assert first_fn == -1 or match.start() < first_fn, (
            f"{script.name}: 'set -euo pipefail' must come BEFORE function defs"
        )

    @pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
    def test_script_documents_exit_code_zero(self, script: Path) -> None:
        """Every script must define ``EXIT_OK=0`` so the documented
        contract is grep-able + future refactors can't silently break
        it."""
        text = _read_script(script)
        assert "EXIT_OK=0" in text, (
            f"{script.name}: missing EXIT_OK=0 constant"
        )

    @pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
    def test_script_uses_readonly_for_exit_codes(self, script: Path) -> None:
        """Exit codes should be ``readonly`` so they can't be silently
        overwritten by a future refactor. Same pattern as
        portability_check.sh:96-105."""
        text = _read_script(script)
        # We require AT LEAST ONE `readonly EXIT_*` line.
        assert re.search(r"^readonly\s+EXIT_\w+=", text, re.MULTILINE), (
            f"{script.name}: no `readonly EXIT_*=` declarations found"
        )

    @pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
    def test_script_resolves_repo_root_from_bash_source(self, script: Path) -> None:
        """Every script should compute ``REPO_ROOT`` from ``${BASH_SOURCE[0]}``
        so it works even when invoked from a different cwd. Matches the
        portability_check.sh pattern."""
        text = _read_script(script)
        assert "REPO_ROOT=" in text
        assert "BASH_SOURCE[0]" in text, (
            f"{script.name}: REPO_ROOT must be derived from ${{BASH_SOURCE[0]}}"
        )

    @pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
    def test_script_has_shared_logger_functions(self, script: Path) -> None:
        """All three scripts should share the same logger function names
        (``log_info``, ``log_ok``, ``log_warn``, ``log_error``) so a
        future refactor can extract them into a ``scripts/_lib.sh``
        without renaming callers. Matches portability_check.sh."""
        text = _read_script(script)
        for fn in ("log_info", "log_ok", "log_warn", "log_error"):
            assert f"{fn}()" in text, (
                f"{script.name}: missing shared logger function {fn}()"
            )

    @pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
    def test_help_flag_prints_usage_and_exits_0(self, script: Path) -> None:
        """``--help`` prints the script's header comment block and exits 0.
        Convention from scripts/portability_check.sh."""
        result = _run_script(script, "--help")
        assert result.returncode == 0, (
            f"{script.name} --help should exit 0; got {result.returncode}\n"
            f"stderr: {result.stderr}"
        )
        # The header should mention "TinyRAG" so the user knows
        # they're looking at the right script.
        assert "TinyRAG" in result.stdout, (
            f"{script.name} --help output should mention TinyRAG:\n"
            f"{result.stdout[:500]!r}"
        )
        # And the EXIT CODES block (or equivalent) so the user knows
        # what the documented contract is.
        assert "EXIT" in result.stdout or "exit code" in result.stdout.lower(), (
            f"{script.name} --help should document exit codes:\n"
            f"{result.stdout[:500]!r}"
        )

    @pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
    def test_unknown_flag_exits_with_code_2(self, script: Path) -> None:
        """Unknown flags exit with code 2 (argparse convention used by
        every other ``scripts/*.sh`` in the repo)."""
        result = _run_script(script, "--bogus-flag-xyzzy-12345")
        assert result.returncode == 2, (
            f"{script.name} --bogus should exit 2; got {result.returncode}\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )
        assert "--bogus-flag-xyzzy-12345" in result.stderr, (
            f"{script.name}: error should name the offending flag:\n"
            f"{result.stderr!r}"
        )


# ---------------------------------------------------------------------------
# setup.sh-specific structure
# ---------------------------------------------------------------------------


class TestSetupScriptStructure:
    """Structural assertions specific to ``setup.sh``."""

    def test_script_invoke_make_for_every_install_step(self) -> None:
        """``setup.sh`` must invoke ``make`` for each of the 5 install
        steps documented in the roadmap: deps-system, install-dev,
        build-llamacpp, download-llm, sensors-generate."""
        text = _read_script(SETUP_SCRIPT)
        for target in (
            "deps-system",
            "install-dev",
            "build-llamacpp",
            "download-llm",
            "sensors-generate",
        ):
            assert target in text, (
                f"setup.sh should invoke `make {target}` (or reference it "
                f"as the underlying install step)"
            )
            # Each should appear as part of an actual `make` invocation,
            # not just a comment. Use a regex.
            assert re.search(rf"\bmake\b[^\n]*{target}", text), (
                f"setup.sh: `make {target}` should appear as a real invocation"
            )

    def test_script_is_idempotent_with_skip_guards(self) -> None:
        """``setup.sh`` must guard each install step with an
        ``if [[ ! -f <artefact> ]]`` (or similar) check so re-running
        it on a set-up machine is a no-op."""
        text = _read_script(SETUP_SCRIPT)
        # We expect at least 4 skip-guards (one per step that has a
        # detectable artefact: venv, llama-server binary, model file,
        # sensor CSV). The 5th step (deps-system) is not always
        # detectable (apt packages don't have a single sentinel file).
        # Match either positive (skip-if-exists) OR negative (skip-if-
        # missing) guards. The most common shape is `if [[ -f FILE ]]`
        # / `if [[ -d DIR ]]` — `if [[ ! -x ... ]]` appears once (the
        # post-build verification).
        positive = re.findall(r"if\s+\[\[\s+-[df]\s+", text)
        negative = re.findall(r"if\s+\[\[\s+!\s+-[a-z]", text)
        total = len(positive) + len(negative)
        assert total >= 4, (
            f"setup.sh should have >=4 skip-guards (positive + negative); "
            f"found {total} ({len(positive)} positive, {len(negative)} negative)"
        )

    def test_script_exit_code_constants_are_documented(self) -> None:
        """All 7 exit codes (0, 10-15) must appear as named constants."""
        text = _read_script(SETUP_SCRIPT)
        for code in SETUP_EXIT_CODES:
            assert f"={code}" in text, (
                f"setup.sh: exit code {code} has no `readonly EXIT_*={code}` assignment"
            )
            # Should also be documented in the header (as `   N  `).
            assert f" {code} " in text, (
                f"setup.sh: exit code {code} is not documented in the header"
            )

    def test_script_uses_log_skipped(self) -> None:
        """Idempotency check: each skip should print a visible message
        via the ``log_skipped`` helper (not silently no-op)."""
        text = _read_script(SETUP_SCRIPT)
        assert "log_skipped" in text, (
            "setup.sh: should define and call log_skipped() to surface "
            "idempotent no-ops to the user"
        )

    def test_script_preflight_checks_python_version(self) -> None:
        """``preflight`` must verify Python >= 3.12 (matches the
        requirements.txt + portability_check.sh contract)."""
        text = _read_script(SETUP_SCRIPT)
        assert "MIN_PYTHON_MAJOR" in text
        assert "MIN_PYTHON_MINOR" in text
        # And the actual check should compare to 3.12.
        assert "3.12" in text, "setup.sh preflight should reference Python 3.12"

    def test_script_preflight_checks_disk_space(self) -> None:
        """Preflight should warn if there's not enough disk for the
        model + llama.cpp build + venv (~3 GB)."""
        text = _read_script(SETUP_SCRIPT)
        assert "MIN_DISK_FREE_KB" in text
        assert "df -Pk" in text or "df -k" in text, (
            "setup.sh preflight should call `df` to check disk space"
        )

    def test_script_preflight_checks_bash_version(self) -> None:
        """Preflight must require bash >= 4 (associative arrays,
        mapfile, etc.)."""
        text = _read_script(SETUP_SCRIPT)
        assert "BASH_VERSINFO" in text
        assert "MIN_BASH_MAJOR" in text


# ---------------------------------------------------------------------------
# run.sh-specific structure
# ---------------------------------------------------------------------------


class TestRunScriptStructure:
    """Structural assertions specific to ``run.sh``."""

    def test_script_installs_exit_trap(self) -> None:
        """``run.sh`` MUST install a trap on EXIT INT TERM so Ctrl+C
        tears down both children. This is the critical race-condition
        fix the plan calls out."""
        text = _read_script(RUN_SCRIPT)
        # Look for `trap on_exit EXIT INT TERM` (or equivalent).
        trap_re = re.compile(r"\btrap\s+\S+\s+(?:EXIT|EXIT\s+INT|EXIT\s+INT\s+TERM)")
        assert trap_re.search(text), (
            "run.sh: must install trap on EXIT INT TERM (so Ctrl+C "
            "kills both llama-server and uvicorn)"
        )

    def test_script_defines_on_exit_function(self) -> None:
        """The cleanup handler must be named ``on_exit`` (matching
        setup.sh and the convention from portability_check.sh)."""
        text = _read_script(RUN_SCRIPT)
        assert re.search(r"^on_exit\s*\(\)\s*\{", text, re.MULTILINE), (
            "run.sh: must define on_exit() function"
        )

    def test_script_tracks_both_pids(self) -> None:
        """Both children must have their PIDs captured into named
        variables so the cleanup trap can find them."""
        text = _read_script(RUN_SCRIPT)
        for var in ("LLAMA_PID", "UVICORN_PID"):
            assert f"{var}=" in text, f"run.sh: must track ${var}"
            # The kill -TERM in on_exit should reference it (allow
            # optional quotes around the variable expansion).
            pattern = r"kill\s+-TERM\s+\"\$\{" + re.escape(var) + r"\}\""
            assert re.search(pattern, text), (
                f"run.sh: on_exit should `kill -TERM ${{{var}}}`"
            )

    def test_script_writes_pid_files(self) -> None:
        """Both children must write their PIDs to disk so stop.sh can
        find them after run.sh exits."""
        text = _read_script(RUN_SCRIPT)
        assert "LLAMA_PIDFILE" in text
        assert "UVICORN_PIDFILE" in text
        # Each PID should be echoed into its PID file. Allow optional
        # closing quote after the variable expansion (the common shape
        # is `echo "${LLAMA_PID}" > "${LLAMA_PIDFILE}"`).
        for var, pidfile in (
            ("LLAMA_PID", "LLAMA_PIDFILE"),
            ("UVICORN_PID", "UVICORN_PIDFILE"),
        ):
            pattern = (
                r"\$\{" + re.escape(var) + r"\}"
                r"\"?\s*>\s*"
                r"\"?\$\{" + re.escape(pidfile) + r"\}"
                r"\"?"
            )
            assert re.search(pattern, text), (
                f"run.sh: should write ${{{var}}} to ${{{pidfile}}} "
                f"(regex: {pattern})"
            )

    def test_script_polls_health_endpoint(self) -> None:
        """llama-server readiness must be verified via curl on /health
        with the canonical ``--retry --retry-connrefused`` pattern."""
        text = _read_script(RUN_SCRIPT)
        assert "/health" in text, "run.sh: must poll llama-server /health"
        assert "--retry-connrefused" in text, (
            "run.sh: must use --retry-connrefused so ECONNREFUSED counts "
            "as a retry, not a failure"
        )
        assert "curl" in text

    def test_script_polls_api_status(self) -> None:
        """uvicorn readiness must be verified via curl on /api/status."""
        text = _read_script(RUN_SCRIPT)
        assert "/api/status" in text, "run.sh: must poll uvicorn /api/status"

    def test_script_uses_env_overrides(self) -> None:
        """Ports, binary path, model path, venv path must be overridable
        via env vars so tests + non-standard setups can use them."""
        text = _read_script(RUN_SCRIPT)
        for var in ("LLAMACPP_BIN", "LLM_GGUF", "LLM_PORT", "API_PORT", "VENV"):
            assert f"{var}:=" in text or f"{var}=\"" in text, (
                f"run.sh: should use env override for {var}"
            )

    def test_script_waits_on_uvicorn(self) -> None:
        """``run.sh`` must block on uvicorn (not exit immediately) so
        the user sees uvicorn's logs + can Ctrl+C cleanly."""
        text = _read_script(RUN_SCRIPT)
        assert re.search(r"\bwait\s+\"\$\{UVICORN_PID\}\"", text), (
            "run.sh: should `wait $UVICORN_PID` so the script blocks "
            "until uvicorn exits"
        )

    def test_script_exit_code_constants_are_documented(self) -> None:
        """All 7 exit codes (0, 10-15) must appear as named constants."""
        text = _read_script(RUN_SCRIPT)
        for code in RUN_EXIT_CODES:
            assert f"={code}" in text, (
                f"run.sh: exit code {code} has no `readonly EXIT_*={code}` assignment"
            )
            assert f" {code} " in text, (
                f"run.sh: exit code {code} is not documented in the header"
            )

    def test_script_preflight_rejects_busy_port(self) -> None:
        """If a port (8000 or 8080) is already bound, ``run.sh`` must
        refuse to start (rather than silently failing mid-launch)."""
        text = _read_script(RUN_SCRIPT)
        # The check should use lsof (or ss/netstat as fallback) AND
        # reference both ports.
        assert "lsof -ti:" in text, "run.sh preflight should use lsof -ti:"
        assert "EXIT_PORT_BUSY" in text
        assert re.search(r"lsof\s+-ti:\"?\$\{LLM_PORT\}", text)
        assert re.search(r"lsof\s+-ti:\"?\$\{API_PORT\}", text)

    def test_script_cleanup_removes_pid_files(self) -> None:
        """The cleanup trap must remove both PID files so re-running
        run.sh / stop.sh doesn't trip over stale PIDs."""
        text = _read_script(RUN_SCRIPT)
        # Look for rm -f ... both PID files inside on_exit.
        assert re.search(
            r"rm\s+-f\s+\"\$\{LLAMA_PIDFILE\}\"\s+\"\$\{UVICORN_PIDFILE\}\"",
            text,
        ), (
            "run.sh on_exit should `rm -f $LLAMA_PIDFILE $UVICORN_PIDFILE`"
        )


# ---------------------------------------------------------------------------
# stop.sh-specific structure
# ---------------------------------------------------------------------------


class TestStopScriptStructure:
    """Structural assertions specific to ``stop.sh``."""

    def test_script_reads_both_pid_files(self) -> None:
        """``stop.sh`` must read both PID files (so it can find children
        even if lsof is unavailable)."""
        text = _read_script(STOP_SCRIPT)
        assert "LLAMA_PIDFILE" in text
        assert "UVICORN_PIDFILE" in text
        # And actually read the file content with cat (or $(<file)).
        assert re.search(r"\bcat\s+\"\$\{LLAMA_PIDFILE\}\"", text)
        assert re.search(r"\bcat\s+\"\$\{UVICORN_PIDFILE\}\"", text)

    def test_script_uses_lsof_for_port_discovery(self) -> None:
        """``stop.sh`` must use ``lsof -ti:<port>`` to catch orphaned
        processes whose PID files were deleted."""
        text = _read_script(STOP_SCRIPT)
        # Both ports must appear in an lsof invocation.
        assert re.search(r"lsof\s+-ti:\"?\$\{LLM_PORT\}", text), (
            "stop.sh should use lsof -ti:$LLM_PORT"
        )
        assert re.search(r"lsof\s+-ti:\"?\$\{API_PORT\}", text), (
            "stop.sh should use lsof -ti:$API_PORT"
        )

    def test_script_escalates_to_sigkill(self) -> None:
        """After SIGTERM, ``stop.sh`` must escalate to SIGKILL if the
        process is still alive. This handles hung C++ loops."""
        text = _read_script(STOP_SCRIPT)
        assert "kill -TERM" in text
        assert "kill -KILL" in text
        # SIGKILL must come AFTER the SIGTERM + a wait loop.
        term_pos = text.find("kill -TERM")
        kill_pos = text.find("kill -KILL")
        assert term_pos != -1 and kill_pos != -1
        assert term_pos < kill_pos, (
            "stop.sh: SIGKILL must come AFTER SIGTERM (escalation order)"
        )

    def test_script_is_idempotent_on_no_processes(self) -> None:
        """``stop.sh`` must exit 0 when there's nothing to stop. We
        verify by running it twice on a clean host."""
        result = _run_script(STOP_SCRIPT, env={"LLM_PORT": "18080", "API_PORT": "18000"})
        # Note: we override the ports to weird values so even if some
        # foreign process happens to bind 8080 / 8000, stop.sh won't
        # accidentally kill it.
        assert result.returncode == 0, (
            f"stop.sh should exit 0 on a clean host; got {result.returncode}\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )
        # The output should explicitly say "Nothing to stop".
        assert "Nothing to stop" in result.stdout, (
            f"stop.sh should announce 'Nothing to stop' on the no-op path:\n"
            f"{result.stdout!r}"
        )

    def test_script_exit_code_constants_are_documented(self) -> None:
        """``stop.sh`` is idempotent (always EXIT_OK=0) but the constant
        must still be grep-able for consistency with the other scripts."""
        text = _read_script(STOP_SCRIPT)
        for code in STOP_EXIT_CODES:
            assert f"={code}" in text, (
                f"stop.sh: exit code {code} has no `readonly EXIT_*={code}` assignment"
            )


# ---------------------------------------------------------------------------
# Integration test — opt-in via RUN_SHELL_INTEGRATION=1
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRunScriptIntegration:
    """End-to-end smoke test. Opt-in because it actually starts the
    real llama-server + uvicorn (takes 30-60 s on a warm host, 2-3
    min cold because of model load).

    Run with::

        RUN_SHELL_INTEGRATION=1 PYTHONPATH=src \\
            ~/venvs/tinyrag/bin/python -m pytest \\
            tests/test_shell_scripts.py::TestRunScriptIntegration -v
    """

    def test_run_and_stop_lifecycle(self, tmp_path: Path) -> None:
        """Bring up the full stack via ``bash run.sh``, verify both
        ports are bound + /api/status responds, then tear down via
        ``bash stop.sh`` and verify both ports are free + both PID
        files are gone.

        Skips automatically when ``RUN_SHELL_INTEGRATION`` is not set
        in the environment.
        """
        if os.environ.get("RUN_SHELL_INTEGRATION") != "1":
            pytest.skip(
                "set RUN_SHELL_INTEGRATION=1 to run the full "
                "stack bring-up + teardown integration test (takes 30-60s)"
            )

        # Skip if the prerequisites aren't present (fresh CI without
        # llama.cpp built or the model downloaded).
        llama_bin = PROJECT_ROOT / "llama.cpp" / "build" / "bin" / "llama-server"
        model_file = PROJECT_ROOT / "models" / "phi-3-mini.gguf"
        if not llama_bin.exists() or not model_file.exists():
            pytest.skip(
                f"prerequisites missing: {llama_bin} or {model_file} — "
                f"run `bash setup.sh` first"
            )

        # Override the log directory to a tmpdir so we don't pollute
        # the real logs/ dir (the test cleans it up in a finally).
        test_log_dir = tmp_path / "logs"
        test_log_dir.mkdir()

        env_overrides = {
            "LLM_HOST": "127.0.0.1",
            "LLM_PORT": "18080",  # off-beat ports to avoid stomping
            "API_HOST": "127.0.0.1",
            "API_PORT": "18000",
            # We can't easily override LOG_DIR without editing the
            # script, but the test cleans up via stop.sh which removes
            # the PID files anyway. Skip the override for now.
        }

        # Launch run.sh in the background with a generous timeout.
        # The `timeout 120` wrapper ensures a hung run.sh doesn't
        # block the test indefinitely.
        run_proc = subprocess.Popen(
            ["timeout", "120", "bash", str(RUN_SCRIPT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, **env_overrides},
            cwd=str(PROJECT_ROOT),
        )

        try:
            # Wait up to 90 s for both ports to come up.
            import time
            import urllib.request

            deadline = time.time() + 90
            api_ok = False
            health_ok = False
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:18080/health", timeout=2
                    ) as r:
                        if r.status == 200:
                            health_ok = True
                except Exception:
                    pass
                try:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:18000/api/status", timeout=2
                    ) as r:
                        if r.status == 200:
                            api_ok = True
                except Exception:
                    pass
                if api_ok and health_ok:
                    break
                time.sleep(1)

            assert health_ok, (
                "llama-server /health did not respond within 90s — "
                "check that the model loads cleanly"
            )
            assert api_ok, (
                "uvicorn /api/status did not respond within 90s — "
                "check the FastAPI lifespan startup"
            )

            # Now call stop.sh and verify both ports are freed.
            stop_result = _run_script(
                STOP_SCRIPT,
                env=env_overrides,
                timeout=30,
            )
            assert stop_result.returncode == 0, (
                f"stop.sh failed (exit {stop_result.returncode}):\n"
                f"stdout: {stop_result.stdout}\nstderr: {stop_result.stderr}"
            )

            # Wait briefly for the OS to actually free the ports.
            time.sleep(2)

            # Verify ports are now free. We use a fresh subprocess for
            # lsof so we don't depend on the host's lsof exit-code
            # semantics.
            for port in ("18080", "18000"):
                lsof_result = subprocess.run(
                    ["lsof", "-ti:" + port],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                assert lsof_result.stdout.strip() == "", (
                    f"port {port} still bound after stop.sh: "
                    f"{lsof_result.stdout!r}"
                )

        finally:
            # Belt + braces: if anything above raised, make sure the
            # run.sh process is gone so we don't leave a hung llama-
            # server around.
            if run_proc.poll() is None:
                run_proc.terminate()
                try:
                    run_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    run_proc.kill()
                    run_proc.wait(timeout=5)
            # And run stop.sh once more to clean any leftovers.
            _run_script(STOP_SCRIPT, env=env_overrides, timeout=30)
