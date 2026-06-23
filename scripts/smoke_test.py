#!/usr/bin/env python3
"""Phase 3 end-to-end smoke test — the canonical "is TinyRAG wired up?" check.

This is the **single-script Phase 3 checkpoint** from
``docs/06_roadmap_v2.md`` Step 3.9. It does four things:

1. Loads the LLMClient (real llama-server or fake, per --client).
2. Sends a hard-coded query (the simplest possible correctness probe).
3. Asserts the response is non-empty.
4. Prints the response + timing + token stats, and exits 0 on success.

Why this script matters
-----------------------

Phase 3 is "make the native + model stack run on the laptop". After
Step 3.9, ``make smoke-e2e`` is the single command a new contributor
runs to confirm their laptop is set up correctly. If it prints a
sensible answer, the entire stack works.

The hard-coded question is intentionally trivial ("What is 2+2?") so
that:
- the assertion is unambiguous (any non-empty answer passes);
- the response is fast (<3 s on a laptop CPU);
- a non-English / nonsense response is immediately visible to a human
  reviewer even though it wouldn't fail the assertion.

CLI flags
---------

    --client {real,fake}   Which LLMClient to use. Default: real.
                            "fake" uses FakeLLMClient (no llama-server
                            needed; great for CI).
    --base-url URL         llama-server URL (real client only).
                            Default: http://127.0.0.1:8080
    --model MODEL_ID       Model id to claim (real client only).
                            Default: phi-3-mini
    --query TEXT           Override the hard-coded question.
    --max-tokens N         Cap on response length. Default: 64.
    --json                 Print JSON result instead of pretty text.
    --quiet                Suppress the pretty banner; just print the
                            response text. Useful for piping.

Exit codes
----------

    0   Response received, non-empty.
    1   Empty response or LLM client errored.
    2   Bad CLI args (argparse handles this with code 2).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

# Make ``src/`` importable when this script is run directly.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tinyrag.generation import (  # noqa: E402
    ChatMessage,
    FakeLLMClient,
    LlamaCppClient,
    LLMClient,
    LLMError,
)

# ---------------------------------------------------------------------------
# Constants — the canonical Phase 3 checkpoint question.
# ---------------------------------------------------------------------------

#: The default probe question. Trivial on purpose:
#: - Any non-empty answer passes the assertion.
#: - Sub-3-second response on a laptop CPU.
#: - A wrong/hallucinated answer is instantly visible to a human reviewer.
DEFAULT_QUERY: str = "What is 2+2?"

#: Default response cap. 64 tokens is plenty for a single-sentence answer
#: (the model would need to write a paragraph to hit this).
DEFAULT_MAX_TOKENS: int = 64

#: Default llama-server URL. Matches the Makefile's SMOKE_BASE_URL.
DEFAULT_BASE_URL: str = "http://127.0.0.1:8080"

#: Default model id to claim. Matches the Makefile's LLM_MODEL default.
DEFAULT_MODEL: str = "phi-3-mini"

#: Banner shown in human-readable mode.
BANNER = "TinyRAG — Phase 3 end-to-end smoke test"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class SmokeResult:
    """Outcome of one smoke run. JSON-serialisable."""

    def __init__(
        self,
        *,
        ok: bool,
        response: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        duration_seconds: float,
        client: str,
        model: str,
        base_url: str | None,
        query: str,
        error: str | None = None,
    ) -> None:
        self.ok = ok
        self.response = response
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.duration_seconds = duration_seconds
        self.client = client
        self.model = model
        self.base_url = base_url
        self.query = query
        self.error = error

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict (rounds floats, drops None)."""
        return {
            "ok": self.ok,
            "response": self.response,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "duration_seconds": round(self.duration_seconds, 3),
            "client": self.client,
            "model": self.model,
            "base_url": self.base_url,
            "query": self.query,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------


def make_real_client(
    *, base_url: str, model: str
) -> LlamaCppClient:
    """Construct a :class:`LlamaCppClient` pointing at the local server."""
    return LlamaCppClient(
        base_url=base_url,
        model=model,
        timeout_s=120.0,  # generous: a slow CPU may take a while
    )


def make_fake_client() -> FakeLLMClient:
    """Construct a :class:`FakeLLMClient` for hermetic testing / CI."""
    return FakeLLMClient(
        default_response=(
            "4. (FakeLLMClient canned response — your laptop's llama-server "
            "is not being contacted, which is fine for CI.)"
        )
    )


# ---------------------------------------------------------------------------
# The smoke run
# ---------------------------------------------------------------------------


def run_smoke(
    *,
    client: LLMClient,
    query: str,
    client_name: str,
    model: str,
    base_url: str | None,
    max_tokens: int,
) -> SmokeResult:
    """Send the query through the client and return a :class:`SmokeResult`.

    Catches every :class:`LLMError` and converts it to a failed result
    (rather than letting the script crash with a traceback).
    """
    messages = [ChatMessage(role="user", content=query)]
    start = time.monotonic()
    try:
        text, stats = client.generate(messages, max_tokens=max_tokens)
    except LLMError as exc:
        duration = time.monotonic() - start
        return SmokeResult(
            ok=False,
            response="",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            duration_seconds=duration,
            client=client_name,
            model=model,
            base_url=base_url,
            query=query,
            error=f"{type(exc).__name__}: {exc}",
        )

    duration = time.monotonic() - start
    ok = bool(text.strip())
    return SmokeResult(
        ok=ok,
        response=text,
        prompt_tokens=stats.prompt_tokens,
        completion_tokens=stats.completion_tokens,
        total_tokens=stats.total_tokens,
        duration_seconds=duration,
        client=client_name,
        model=model,
        base_url=base_url,
        query=query,
        error=None if ok else "empty response",
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_human(result: SmokeResult, *, quiet: bool, max_tokens: int) -> None:
    """Print a friendly summary to stdout."""
    if quiet:
        # Pipe-friendly: just the response text + ok/exit code.
        sys.stdout.write(result.response)
        if not result.response.endswith("\n"):
            sys.stdout.write("\n")
        return

    print(f"==> {BANNER}")
    print(f"    client:   {result.client}")
    print(f"    model:    {result.model}")
    if result.base_url is not None:
        print(f"    base url: {result.base_url}")
    print(f"    query:    {result.query!r}")
    print(f"    max tok:  {max_tokens}")
    print("    sending request...")
    print()
    if result.ok:
        print("    response:")
        for line in result.response.splitlines() or [result.response]:
            print(f"      {line}")
        print()
        print(f"    prompt_tokens:     {result.prompt_tokens}")
        print(f"    completion_tokens: {result.completion_tokens}")
        print(f"    duration:          {result.duration_seconds:.2f} s")
        if result.completion_tokens and result.duration_seconds > 0:
            tps = result.completion_tokens / result.duration_seconds
            print(f"    tokens/second:     {tps:.2f}")
        print("[ OK ] Phase 3 smoke test passed.")
    else:
        print(f"[FAIL] {result.error or 'unknown failure'}")
        if result.response:
            print(f"      response was: {result.response!r}")


def print_json(result: SmokeResult) -> None:
    """Print the result as a single JSON object."""
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="smoke_test.py",
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--client",
        choices=("real", "fake"),
        default="real",
        help="Which LLMClient to use. Default: real (talks to llama-server).",
    )
    p.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"llama-server URL (real client). Default: {DEFAULT_BASE_URL}",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            f"Model id to claim (real client). Default: {DEFAULT_MODEL}. "
            f"Note: llama-server doesn't validate this — it serves whatever "
            f"it has loaded."
        ),
    )
    p.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Override the default question.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Cap on response length. Default: {DEFAULT_MAX_TOKENS}.",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Print JSON result instead of human-readable output.",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Print only the response text (for piping).",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.client == "real":
        client = make_real_client(base_url=args.base_url, model=args.model)
        client_name = "LlamaCppClient"
        base_url_for_result: str | None = args.base_url
    else:
        client = make_fake_client()
        client_name = "FakeLLMClient"
        base_url_for_result = None  # fake never touches the network

    try:
        result = run_smoke(
            client=client,
            query=args.query,
            client_name=client_name,
            model=args.model,
            base_url=base_url_for_result,
            max_tokens=args.max_tokens,
        )
    finally:
        # Close the real client's httpx session so the script exits cleanly.
        if hasattr(client, "close"):
            client.close()

    if args.json:
        print_json(result)
    else:
        print_human(result, quiet=args.quiet, max_tokens=args.max_tokens)

    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
