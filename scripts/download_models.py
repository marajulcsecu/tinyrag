#!/usr/bin/env python3
"""Download and verify GGUF models used by TinyRAG.

This is the **user-facing** entry point. The actual I/O lives in
``tinyrag.models.downloader.ModelDownloader``; this script handles
argument parsing, ANSI-coloured progress bars, and exit codes.

Usage
-----
    # List every model TinyRAG knows about (no I/O)
    python scripts/download_models.py --list

    # Download the primary LLM (Phi-3 Mini)
    python scripts/download_models.py --model phi-3-mini

    # Download all evaluation models (this takes a while)
    python scripts/download_models.py --all

    # Force re-download even if the file exists (e.g. SHA changed)
    python scripts/download_models.py --model phi-3-mini --force

    # Custom output directory
    python scripts/download_models.py --model phi-3-mini --models-dir /srv/models

    # Machine-readable output (for CI)
    python scripts/download_models.py --model phi-3-mini --json

    # Just verify what's already on disk (no network)
    python scripts/download_models.py --verify-only --model phi-3-mini

Exit codes
----------
0   success (every requested model verified)
1   one or more downloads failed
2   bad CLI args / registry error
3   checksum mismatch (this is the dangerous one — investigate)

Companion docs
--------------
- ``src/tinyrag/models/registry.py`` — the canonical catalog
- ``src/tinyrag/models/downloader.py`` — the I/O logic
- ``docs/MODELS.md`` — the human-readable model catalog
- ``docs/06_roadmap_v2.md`` Step 3.5 — the original spec
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

# Make ``src/`` importable when this script is run directly without
# ``pip install -e .``. After Phase 4 the project will be installed and
# this block becomes a no-op, but leaving it in keeps the script
# usable for the standalone ``scripts/`` invocation.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tinyrag.models import (  # noqa: E402  (sys.path tweak above)
    MODEL_REGISTRY,
    ChecksumMismatchError,
    DownloadError,
    ModelDownloader,
    UnknownModelError,
)
from tinyrag.models.downloader import DownloadProgress  # noqa: E402

# ---------------------------------------------------------------------------
# Pretty output
# ---------------------------------------------------------------------------

# ANSI escape codes, only emitted when stdout is a TTY. Keep them local
# so unit tests don't have to mock them.
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(s: str) -> str:
    return _c("1", s)


def _green(s: str) -> str:
    return _c("32", s)


def _red(s: str) -> str:
    return _c("31", s)


def _yellow(s: str) -> str:
    return _c("33", s)


def _cyan(s: str) -> str:
    return _c("36", s)


def _dim(s: str) -> str:
    return _c("2", s)


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------


class _ProgressBar:
    """Tiny single-line progress bar.

    Designed for the simple case (one model at a time). Multi-model
    concurrent download is out of scope for this CLI; Phase 5's eval
    runner handles parallelism.
    """

    BAR_WIDTH = 40

    def __init__(self, model_id: str, total: int | None) -> None:
        self.model_id = model_id
        self.total = total
        self._last_pct: int = -1
        self._last_done: int = 0

    def update(self, progress: DownloadProgress) -> None:
        if progress.phase == "verify":
            sys.stdout.write("\n" + _cyan(f"  [{self.model_id}] verifying SHA-256...\n"))
            sys.stdout.flush()
            return
        if progress.phase == "done":
            sys.stdout.write(
                _green(f"  [{self.model_id}] done ({progress.bytes_done/1_048_576:.1f} MB)\n")
            )
            sys.stdout.flush()
            return

        done = progress.bytes_done
        total = progress.bytes_total or self.total
        if total and total > 0:
            pct = int(done * 100 / total)
            pct = max(0, min(100, pct))
            filled = int(self.BAR_WIDTH * pct / 100)
            bar = "[" + "#" * filled + "-" * (self.BAR_WIDTH - filled) + "]"
            mb = done / 1_048_576
            line = f"\r  [{self.model_id}] {bar} {pct:3d}%  {mb:7.1f} MB"
            if pct != self._last_pct:
                sys.stdout.write(line)
                sys.stdout.flush()
                self._last_pct = pct
        else:
            mb = done / 1_048_576
            sys.stdout.write(f"\r  [{self.model_id}] {mb:7.1f} MB downloaded...")
            sys.stdout.flush()
        self._last_done = done


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Print every known model in a compact table. No I/O."""
    rows = sorted(MODEL_REGISTRY.values(), key=lambda e: (e.role, e.model_id))
    headers = ("ID", "Display name", "Quant", "Size", "License", "Role")
    widths = [22, 42, 10, 10, 12, 14]
    print(_bold(f"Known models ({len(rows)} total):"))
    print()
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=False))
    print(_dim(header_line))
    print(_dim("-" * len(header_line)))
    for e in rows:
        size_mb = f"{e.expected_size_bytes // 1_048_576} MB"
        print(
            "  ".join(
                [
                    e.model_id.ljust(widths[0]),
                    e.display_name.ljust(widths[1])[: widths[1]],
                    e.quantization.ljust(widths[2]),
                    size_mb.ljust(widths[3]),
                    e.license.ljust(widths[4]),
                    e.role.ljust(widths[5]),
                ]
            )
        )
    print()
    print(_dim(f"Total disk: {sum(e.expected_size_bytes for e in rows) // 1_048_576} MB"))
    return 0


def cmd_download(
    model_ids: Sequence[str],
    models_dir: Path,
    *,
    force: bool,
    json_mode: bool,
) -> int:
    """Download + verify each requested model. Returns shell exit code."""
    dl = ModelDownloader()
    overall_ok = True
    results: list[dict] = []

    for mid in model_ids:
        entry = MODEL_REGISTRY.get(mid)
        if entry is None:
            msg = f"Unknown model {mid!r}. Known: {sorted(MODEL_REGISTRY)}"
            if json_mode:
                results.append({"model_id": mid, "ok": False, "error": msg})
            else:
                print(_red(f"[FAIL] {mid}: {msg}"))
            overall_ok = False
            continue

        if json_mode:
            # In JSON mode, don't spam progress bars; emit one
            # structured line per phase.
            def cb(progress: DownloadProgress) -> None:

                results.append(
                    {
                        "model_id": progress.model_id,
                        "phase": progress.phase,
                        "bytes_done": progress.bytes_done,
                        "bytes_total": progress.bytes_total,
                    }
                )

        else:
            print()
            print(_bold(f"==> {entry.display_name}"))
            print(_dim(f"    repo:     {entry.hf_repo}"))
            print(_dim(f"    file:     {entry.hf_filename}"))
            print(_dim(f"    license:  {entry.license}"))
            print(_dim(f"    expected: ~{entry.expected_size_bytes // 1_048_576} MB"))
            bar = _ProgressBar(mid, entry.expected_size_bytes)
            cb = bar.update  # type: ignore[assignment]

        try:
            result = dl.download(
                mid, models_dir, force=force, progress_cb=cb
            )
        except ChecksumMismatchError as exc:
            if json_mode:
                results.append(
                    {
                        "model_id": mid,
                        "ok": False,
                        "error": "checksum_mismatch",
                        "expected_sha256": exc.expected,
                        "actual_sha256": exc.actual,
                    }
                )
            else:
                print(_red(f"[FAIL] {mid}: SHA-256 mismatch!"))
                print(_red(f"        expected: {exc.expected}"))
                print(_red(f"        actual:   {exc.actual}"))
                print(_red("        The file has been deleted. Investigate and re-run."))
            overall_ok = False
            return 3  # Hard stop — checksum mismatch is the dangerous case.
        except (DownloadError, OSError) as exc:
            if json_mode:
                results.append({"model_id": mid, "ok": False, "error": str(exc)})
            else:
                print(_red(f"[FAIL] {mid}: {exc}"))
            overall_ok = False
            continue

        if json_mode:
            results.append({"model_id": mid, "ok": True, **result.to_dict()})
        else:
            mb = result.size_bytes / 1_048_576
            line = (
                _green(f"[ OK ] {mid}: {result.path} ({mb:.1f} MB, "
                       f"{result.duration_seconds:.1f}s)")
                if not result.from_cache
                else _cyan(f"[CACHED] {mid}: {result.path} ({mb:.1f} MB)")
            )
            print(line)

    if json_mode:
        print(json.dumps({"ok": overall_ok, "results": results}, indent=2))
    else:
        print()
        if overall_ok:
            print(_green("All requested models are ready."))
        else:
            print(_red("One or more downloads failed."))
    return 0 if overall_ok else 1


def cmd_verify(model_ids: Sequence[str], models_dir: Path) -> int:
    """Re-hash each model on disk and report its status."""
    dl = ModelDownloader()
    all_ok = True
    for mid in model_ids:
        try:
            present = dl.verify(mid, models_dir)
        except UnknownModelError as exc:
            print(_red(f"[FAIL] {mid}: {exc}"))
            all_ok = False
            continue
        if present:
            path = models_dir / f"{mid}.gguf"
            size_mb = path.stat().st_size // 1_048_576
            print(_green(f"[ OK ] {mid}: {size_mb} MB at {path}"))
        else:
            print(_yellow(f"[MISS] {mid}: not present or SHA mismatch"))
            all_ok = False
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download_models.py",
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="ID",
        help=(
            "Model id to download (repeatable). Example: --model phi-3-mini. "
            "Use --list to see all available ids."
        ),
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Download every model in the registry (large!).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print the registry as a table and exit (no I/O).",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Re-hash on-disk files instead of downloading.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists and matches.",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to store GGUF files (default: ./models).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output (for CI).",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        return cmd_list(args, parser)

    # Decide which models to act on.
    if args.all:
        requested = sorted(MODEL_REGISTRY.keys())
    elif args.models:
        requested = list(args.models)
    else:
        # No --model and no --all: act on the primary by default, so a
        # bare `python scripts/download_models.py` does the useful thing.
        requested = ["phi-3-mini"]

    if args.verify_only:
        return cmd_verify(requested, args.models_dir)
    return cmd_download(
        requested,
        args.models_dir,
        force=args.force,
        json_mode=args.json,
    )


if __name__ == "__main__":
    sys.exit(main())
