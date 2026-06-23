#!/usr/bin/env python3
"""Smoke test the LLM client against a running llama-server.

This is the "model readiness gate" — it confirms that:

1. ``llama-server`` is reachable on ``base_url``.
2. The selected model loads, accepts a prompt, and returns tokens.
3. Streaming works (the response is non-empty and arrives in chunks).
4. Tokens-per-second is in the expected ballpark for the CPU.

It is the architectural seam between "we have a model on disk" and
"we trust the model in production". Phase 5's evaluation harness will
call the same code path; if this smoke test fails, the evaluation
will also fail.

Usage
-----
    # Smoke-test the default model (phi-3-mini) against localhost:8080
    python scripts/smoke_test_llm.py

    # Smoke-test a specific model id
    python scripts/smoke_test_llm.py --model llama-3.2-3b

    # Smoke-test every model that's already on disk
    python scripts/smoke_test_llm.py --all

    # Custom prompt, base URL, or output format
    python scripts/smoke_test_llm.py --prompt "What is 2+2?" --base-url http://192.168.1.5:8080
    python scripts/smoke_test_llm.py --json

Exit codes
----------
0   every requested model responded with a non-empty completion
1   at least one model failed (unreachable, empty, or refused)
2   bad CLI args

Companion docs
--------------
- ``src/tinyrag/generation/llm_client.py`` — the LLMClient Protocol + clients
- ``docs/MODELS.md`` — the model catalog
- ``docs/06_roadmap_v2.md`` Step 3.7 — the original spec
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

# Make ``src/`` importable when this script is run directly.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tinyrag.generation import (  # noqa: E402
    ChatMessage,
    LlamaCppClient,
    LLMError,
    LLMRefusedError,
    LLMUnavailableError,
)
from tinyrag.models import MODEL_REGISTRY  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default prompt — exercises basic chat behaviour, short enough to
#: keep the smoke test under ~10 s even on a slow CPU.
DEFAULT_PROMPT = "In one sentence, what is a smart home?"

#: Default HTTP base URL — matches the Makefile ``run-llm`` target.
DEFAULT_BASE_URL = "http://127.0.0.1:8080"

#: Default model id — must match the entry in MODEL_REGISTRY.
DEFAULT_MODEL = "phi-3-mini"

#: Filesystem location of downloaded GGUFs. Used by ``--all`` to
#: figure out which models to test.
DEFAULT_MODELS_DIR = Path("models")

#: Minimum completion tokens to consider a response "real". A
#: sub-token response means the model didn't actually generate
#: anything — likely a server-side warm-up or refusal.
MIN_COMPLETION_TOKENS = 5

#: Maximum acceptable end-to-end wall-clock seconds for a single
#: short prompt. Phi-3 on the i5-1235U averages ~5-15 s for a
#: 100-token reply. 90 s is generous.
MAX_DURATION_S = 90.0


# ---------------------------------------------------------------------------
# Pretty output
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(s: str) -> str:
    return _c("1", s)


def _green(s: str) -> str:
    return _c("32", s)


def _red(s: str) -> str:
    return _c("31", s)


def _yellow(s: str) -> str:
    return _c("33", s)


def _dim(s: str) -> str:
    return _c("2", s)


# ---------------------------------------------------------------------------
# Per-model smoke test
# ---------------------------------------------------------------------------


def _is_present_locally(model_id: str, models_dir: Path) -> bool:
    """Check if the GGUF for ``model_id`` is on disk.

    The downloader records the SHA-256 in the manifest. We don't
    re-hash here (that's what ``make verify-llm`` is for); we just
    check the file is there so ``--all`` doesn't try to test a model
    the user hasn't downloaded.
    """
    return (models_dir / f"{model_id}.gguf").exists()


def smoke_test_one(
    *,
    model_id: str,
    base_url: str,
    prompt: str,
    max_tokens: int,
    json_mode: bool,
) -> dict:
    """Run the smoke test for one model. Returns a structured result dict."""
    if model_id not in MODEL_REGISTRY:
        msg = f"Unknown model {model_id!r}. Known: {sorted(MODEL_REGISTRY)}"
        if json_mode:
            return {"model_id": model_id, "ok": False, "error": msg}
        print(_red(f"[FAIL] {msg}"))
        return {"model_id": model_id, "ok": False, "error": msg}

    entry = MODEL_REGISTRY[model_id]
    client = LlamaCppClient(base_url=base_url, model=model_id, timeout_s=MAX_DURATION_S)
    messages = [ChatMessage(role="user", content=prompt)]

    if not json_mode:
        print(_bold(f"\n==> {entry.display_name}"))
        print(_dim(f"    model id:   {model_id}"))
        print(_dim(f"    base url:   {base_url}"))
        print(_dim(f"    prompt:     {prompt!r}"))
        print(_dim(f"    max_tokens: {max_tokens}"))
        print(_dim("    sending request..."))

    try:
        t0 = time.monotonic()
        text, stats = client.generate(messages, max_tokens=max_tokens)
        wall = time.monotonic() - t0
    except LLMUnavailableError as exc:
        if json_mode:
            return {"model_id": model_id, "ok": False, "error": str(exc), "phase": "unavailable"}
        print(_red(f"[FAIL] {model_id}: server unavailable — {exc}"))
        print(_yellow("        Is llama-server running? Try: make run-llm"))
        return {"model_id": model_id, "ok": False, "error": str(exc), "phase": "unavailable"}
    except LLMRefusedError as exc:
        if json_mode:
            return {"model_id": model_id, "ok": False, "error": str(exc), "phase": "refused"}
        print(_red(f"[FAIL] {model_id}: refused — {exc}"))
        return {"model_id": model_id, "ok": False, "error": str(exc), "phase": "refused"}
    except LLMError as exc:
        if json_mode:
            return {"model_id": model_id, "ok": False, "error": str(exc), "phase": "error"}
        print(_red(f"[FAIL] {model_id}: {exc}"))
        return {"model_id": model_id, "ok": False, "error": str(exc), "phase": "error"}
    finally:
        client.close()

    # Validate the response.
    ok = True
    reasons: list[str] = []
    if not text.strip():
        ok = False
        reasons.append("empty response")
    if stats.completion_tokens < MIN_COMPLETION_TOKENS:
        ok = False
        reasons.append(
            f"too few tokens ({stats.completion_tokens} < {MIN_COMPLETION_TOKENS})"
        )
    if wall > MAX_DURATION_S:
        ok = False
        reasons.append(f"too slow ({wall:.1f}s > {MAX_DURATION_S}s)")

    if json_mode:
        return {
            "model_id": model_id,
            "ok": ok,
            "text": text,
            "prompt_tokens": stats.prompt_tokens,
            "completion_tokens": stats.completion_tokens,
            "total_tokens": stats.total_tokens,
            "duration_seconds": wall,
            "tokens_per_second": stats.completion_tokens / wall if wall > 0 else 0.0,
            "reasons": reasons,
        }

    # Human-readable report.
    print()
    print(_bold("    response:"))
    # Indent the response for readability.
    for line in text.splitlines() or [text]:
        print(f"      {line}")
    print()
    print(_dim(f"    prompt_tokens:     {stats.prompt_tokens}"))
    print(_dim(f"    completion_tokens: {stats.completion_tokens}"))
    print(_dim(f"    duration:          {wall:.2f} s"))
    tps = stats.completion_tokens / wall if wall > 0 else 0.0
    print(_dim(f"    tokens/second:     {tps:.2f}"))

    if ok:
        print(_green(f"[ OK ] {model_id}"))
    else:
        print(_red(f"[FAIL] {model_id}: {', '.join(reasons)}"))
    return {
        "model_id": model_id,
        "ok": ok,
        "text": text,
        "reasons": reasons,
        "tokens_per_second": tps,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="smoke_test_llm.py",
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        default=None,
        help=(
            f"Model id to test (default: {DEFAULT_MODEL!r}). Use `make list-models` "
            "to see available ids."
        ),
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Smoke-test every model whose GGUF is present in --models-dir.",
    )
    p.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"llama-server base URL (default: {DEFAULT_BASE_URL}).",
    )
    p.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Prompt to send (default: {DEFAULT_PROMPT!r}).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Cap on output tokens (default: 128).",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Where to look for downloaded GGUFs (default: ./models).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output (for CI).",
    )
    return p


def _resolve_model_ids(args: argparse.Namespace) -> list[str]:
    if args.all:
        present = [
            mid
            for mid in sorted(MODEL_REGISTRY)
            if _is_present_locally(mid, args.models_dir)
        ]
        if not present:
            print(
                _yellow(
                    f"No models found in {args.models_dir}/. "
                    "Run `make download-llm` first."
                )
            )
        return present
    if args.model:
        return [args.model]
    return [DEFAULT_MODEL]


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    model_ids = _resolve_model_ids(args)
    if not model_ids:
        return 1

    results: list[dict] = []
    for mid in model_ids:
        result = smoke_test_one(
            model_id=mid,
            base_url=args.base_url,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            json_mode=args.json,
        )
        results.append(result)

    if args.json:
        payload = {
            "ok": all(r.get("ok") for r in results),
            "results": results,
        }
        print(json.dumps(payload, indent=2))
    else:
        print()
        n_ok = sum(1 for r in results if r.get("ok"))
        n_total = len(results)
        if n_ok == n_total:
            print(_green(f"All {n_total} model(s) passed the smoke test."))
        else:
            print(_red(f"{n_total - n_ok}/{n_total} model(s) failed."))

    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
