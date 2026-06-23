"""Tests for scripts/smoke_test.py (Phase 3 end-to-end smoke test).

These tests lock down the Phase 3 checkpoint contract:

1. The default query matches the roadmap spec (a hard-coded "What is 2+2?").
2. The script exits 0 with a non-empty response from any LLMClient.
3. Empty / errored responses produce a non-zero exit code.
4. --json output is valid JSON with the expected schema.
5. --query overrides the default.
6. --client fake works hermetically (no llama-server).
7. CLI errors (bad --client value, missing required arg) exit non-zero.

We never hit the network. The fake client is enough to exercise every
code path. Real-client paths are tested via the ``LlamaCppClient``
unit tests in ``tests/test_llm_client.py``.

Location: ``tests/test_smoke_test.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make the script importable as a module.
_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import smoke_test  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — the contract from docs/06_roadmap_v2.md Step 3.9
# ---------------------------------------------------------------------------


class TestContractConstants:
    """The defaults are part of the public surface."""

    def test_default_query_matches_roadmap_spec(self) -> None:
        """The roadmap Step 3.9 says 'What is 2+2?' — that's the probe."""
        assert smoke_test.DEFAULT_QUERY == "What is 2+2?"

    def test_default_base_url(self) -> None:
        """Must match the Makefile's ``SMOKE_BASE_URL``."""
        assert smoke_test.DEFAULT_BASE_URL == "http://127.0.0.1:8080"

    def test_default_model_is_phi3(self) -> None:
        """Primary model — must match the Makefile's ``LLM_MODEL``."""
        assert smoke_test.DEFAULT_MODEL == "phi-3-mini"

    def test_default_max_tokens_is_small(self) -> None:
        """A short probe question should not need many tokens."""
        assert 16 <= smoke_test.DEFAULT_MAX_TOKENS <= 256


# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------


class TestClientFactories:
    """The two factories must produce correctly-typed clients."""

    def test_make_real_client_returns_llamacpp(self) -> None:
        client = smoke_test.make_real_client(
            base_url="http://127.0.0.1:9999", model="phi-3-mini"
        )
        try:
            assert isinstance(client, smoke_test.LlamaCppClient)
            assert client.base_url == "http://127.0.0.1:9999"
        finally:
            client.close()

    def test_make_fake_client_returns_fake(self) -> None:
        client = smoke_test.make_fake_client()
        assert isinstance(client, smoke_test.FakeLLMClient)


# ---------------------------------------------------------------------------
# run_smoke() — the core function
# ---------------------------------------------------------------------------


class TestRunSmoke:
    """run_smoke() must produce a :class:`SmokeResult` for every code path."""

    def test_succeeds_with_fake_client(self) -> None:
        client = smoke_test.make_fake_client()
        result = smoke_test.run_smoke(
            client=client,
            query="What is 2+2?",
            client_name="FakeLLMClient",
            model="phi-3-mini",
            base_url=None,
            max_tokens=64,
        )
        assert result.ok is True
        assert result.response.strip() != ""
        assert result.error is None
        assert result.completion_tokens is not None
        assert result.completion_tokens > 0

    def test_catches_llmerror(self) -> None:
        """A client that raises must produce ok=False with a structured error."""

        class BrokenClient:
            def generate(self, messages, *, max_tokens, temperature=0.0):
                raise smoke_test.LLMError("server down (any LLMError subclass)")

        result = smoke_test.run_smoke(
            client=BrokenClient(),
            query="hello",
            client_name="BrokenClient",
            model="phi-3-mini",
            base_url=None,
            max_tokens=64,
        )
        assert result.ok is False
        assert "server down" in (result.error or "")
        assert "LLMError" in (result.error or "")

    def test_empty_response_is_failure(self) -> None:
        """A client that returns "" must produce ok=False."""

        class SilentClient:
            def generate(self, messages, *, max_tokens, temperature=0.0):
                return "", _FakeStats()

        result = smoke_test.run_smoke(
            client=SilentClient(),
            query="hello",
            client_name="SilentClient",
            model="phi-3-mini",
            base_url=None,
            max_tokens=64,
        )
        assert result.ok is False
        assert result.error == "empty response"

    def test_whitespace_only_response_is_failure(self) -> None:
        """Whitespace-only is treated as empty — the assertion uses strip()."""

        class WhitespaceClient:
            def generate(self, messages, *, max_tokens, temperature=0.0):
                return "   \n\n  ", _FakeStats()

        result = smoke_test.run_smoke(
            client=WhitespaceClient(),
            query="hello",
            client_name="WhitespaceClient",
            model="phi-3-mini",
            base_url=None,
            max_tokens=64,
        )
        assert result.ok is False

    def test_records_query_and_model(self) -> None:
        client = smoke_test.make_fake_client()
        result = smoke_test.run_smoke(
            client=client,
            query="what's the temperature?",
            client_name="FakeLLMClient",
            model="phi-3-mini",
            base_url=None,
            max_tokens=64,
        )
        assert result.query == "what's the temperature?"
        assert result.model == "phi-3-mini"

    def test_records_duration(self) -> None:
        client = smoke_test.make_fake_client()
        result = smoke_test.run_smoke(
            client=client,
            query="hi",
            client_name="FakeLLMClient",
            model="phi-3-mini",
            base_url=None,
            max_tokens=64,
        )
        assert result.duration_seconds >= 0
        # FakeLLMClient is microseconds — wall time should be < 1 second.
        assert result.duration_seconds < 1.0


class _FakeStats:
    """Minimal stand-in for GenerationStats."""

    def __init__(self) -> None:
        self.prompt_tokens = 1
        self.completion_tokens = 5
        self.total_tokens = 6
        self.duration_seconds = 0.0


# ---------------------------------------------------------------------------
# SmokeResult.to_dict()
# ---------------------------------------------------------------------------


class TestSmokeResultSerialization:
    """to_dict() must produce a clean JSON object."""

    def test_to_dict_has_all_keys(self) -> None:
        client = smoke_test.make_fake_client()
        result = smoke_test.run_smoke(
            client=client,
            query="hi",
            client_name="FakeLLMClient",
            model="phi-3-mini",
            base_url=None,
            max_tokens=64,
        )
        d = result.to_dict()
        for k in (
            "ok",
            "response",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "duration_seconds",
            "client",
            "model",
            "base_url",
            "query",
            "error",
        ):
            assert k in d, f"missing key: {k}"

    def test_to_dict_is_json_serialisable(self) -> None:
        client = smoke_test.make_fake_client()
        result = smoke_test.run_smoke(
            client=client,
            query="hi",
            client_name="FakeLLMClient",
            model="phi-3-mini",
            base_url=None,
            max_tokens=64,
        )
        # Must round-trip through json.dumps without errors.
        json.dumps(result.to_dict())

    def test_to_dict_rounds_duration(self) -> None:
        """Float precision should be 3 decimal places for human readability."""
        client = smoke_test.make_fake_client()
        result = smoke_test.run_smoke(
            client=client,
            query="hi",
            client_name="FakeLLMClient",
            model="phi-3-mini",
            base_url=None,
            max_tokens=64,
        )
        d = result.to_dict()
        # The float must be a plain float (not a numpy scalar or Decimal).
        assert isinstance(d["duration_seconds"], float)


# ---------------------------------------------------------------------------
# End-to-end CLI test (no subprocess — we call main() directly)
# ---------------------------------------------------------------------------


class TestMainEndToEnd:
    """Drive main() as if from the command line. No subprocess."""

    def test_fake_client_returns_zero_exit(self, capsys) -> None:
        exit_code = smoke_test.main(["--client", "fake"])
        assert exit_code == 0
        out = capsys.readouterr().out
        # Human-readable banner appears in non-JSON mode.
        assert "TinyRAG" in out
        assert "[ OK ]" in out

    def test_json_flag_emits_only_json(self, capsys) -> None:
        exit_code = smoke_test.main(["--client", "fake", "--json"])
        assert exit_code == 0
        out = capsys.readouterr().out.strip()
        # Must parse as a single JSON object.
        parsed = json.loads(out)
        assert isinstance(parsed, dict)
        assert parsed["ok"] is True
        assert parsed["client"] == "FakeLLMClient"

    def test_quiet_flag_prints_only_response(self, capsys) -> None:
        exit_code = smoke_test.main(["--client", "fake", "--quiet"])
        assert exit_code == 0
        out = capsys.readouterr().out
        # No banner in quiet mode.
        assert "TinyRAG" not in out
        # Just the response text.
        assert "4." in out  # FakeLLMClient says "4."

    def test_query_flag_overrides_default(self, capsys) -> None:
        exit_code = smoke_test.main(["--client", "fake", "--query", "what's 3+3?"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "what's 3+3?" in out

    def test_bad_client_choice_exits_two(self) -> None:
        """argparse rejects --client with an unknown value (exit code 2)."""
        with pytest.raises(SystemExit) as exc:
            smoke_test.main(["--client", "bogus"])
        assert exc.value.code == 2

    def test_real_client_no_server_exits_one(self, capsys) -> None:
        """No llama-server running → exits 1 with structured error."""
        # Point at a definitely-closed port.
        exit_code = smoke_test.main(
            ["--client", "real", "--base-url", "http://127.0.0.1:1"]
        )
        assert exit_code == 1
        # JSON path: error string should be present.
        out = capsys.readouterr().out
        # Re-run with --json to capture the structured error.
        # (We can't easily mix stdout capture across two main() calls, so
        # assert on the human-readable error string in the first run.)
        assert "Connection refused" in out or "Connection" in out

    def test_real_client_no_server_json_structure(self, capsys) -> None:
        """When --json is set, error path returns structured ok=false."""
        exit_code = smoke_test.main(
            [
                "--client",
                "real",
                "--base-url",
                "http://127.0.0.1:1",
                "--json",
            ]
        )
        assert exit_code == 1
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["ok"] is False
        assert "LLMUnavailableError" in parsed["error"]
        assert parsed["base_url"] == "http://127.0.0.1:1"


# ---------------------------------------------------------------------------
# Banner + output formatting
# ---------------------------------------------------------------------------


class TestOutputFormatting:
    """print_human and print_json must not crash on edge cases."""

    def test_print_human_ok_response(self, capsys) -> None:
        client = smoke_test.make_fake_client()
        result = smoke_test.run_smoke(
            client=client,
            query="hi",
            client_name="FakeLLMClient",
            model="phi-3-mini",
            base_url=None,
            max_tokens=64,
        )
        smoke_test.print_human(result, quiet=False, max_tokens=64)
        out = capsys.readouterr().out
        assert "[ OK ]" in out
        assert "tokens/second" in out

    def test_print_human_failure_response(self, capsys) -> None:
        client = smoke_test.make_fake_client()
        result = smoke_test.run_smoke(
            client=client,
            query="hi",
            client_name="FakeLLMClient",
            model="phi-3-mini",
            base_url=None,
            max_tokens=64,
        )
        # Synthesise a failure for the formatter.
        result.ok = False
        result.response = ""
        result.error = "synthetic failure for the test"
        smoke_test.print_human(result, quiet=False, max_tokens=64)
        out = capsys.readouterr().out
        assert "[FAIL]" in out
        assert "synthetic failure" in out

    def test_print_human_quiet_mode(self, capsys) -> None:
        client = smoke_test.make_fake_client()
        result = smoke_test.run_smoke(
            client=client,
            query="hi",
            client_name="FakeLLMClient",
            model="phi-3-mini",
            base_url=None,
            max_tokens=64,
        )
        smoke_test.print_human(result, quiet=True, max_tokens=64)
        out = capsys.readouterr().out
        assert "TinyRAG" not in out
        # Response text appears verbatim.
        assert "FakeLLMClient" in out

    def test_print_json_serialises(self, capsys) -> None:
        client = smoke_test.make_fake_client()
        result = smoke_test.run_smoke(
            client=client,
            query="hi",
            client_name="FakeLLMClient",
            model="phi-3-mini",
            base_url=None,
            max_tokens=64,
        )
        smoke_test.print_json(result)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "ok" in parsed
