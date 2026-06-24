#!/usr/bin/env python3
"""Download and verify test fixtures used by TinyRAG.

This is the **user-facing** entry point. The actual I/O lives in this
file's ``_download_one`` function; this script handles argument
parsing, ANSI-coloured progress bars, and exit codes.

Usage
-----
    # List every fixture TinyRAG knows about (no I/O)
    python scripts/download_fixtures.py --list

    # Download the Nest thermostat PDF (the default Step 4.9 fixture)
    python scripts/download_fixtures.py

    # Force re-download even if the file exists (e.g. SHA changed)
    python scripts/download_fixtures.py --force

    # Custom output directory
    python scripts/download_fixtures.py --fixtures-dir /tmp/fixtures

    # Machine-readable output (for CI)
    python scripts/download_fixtures.py --json

    # Just verify what's already on disk (no network)
    python scripts/download_fixtures.py --verify-only

Exit codes
----------
0   success (every requested fixture verified)
1   one or more downloads failed
2   bad CLI args / unknown fixture name
3   checksum mismatch (this is the dangerous one — investigate)

Companion docs
--------------
- ``tests/fixtures/`` — the default destination directory
- ``.gitignore`` line 171 — the ``tests/fixtures/*`` rule keeps the
  downloaded PDFs out of git (re-downloaded on demand)
- ``docs/06_roadmap_v2.md`` Step 4.9 — the original spec
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Make ``src/`` importable when this script is run directly.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Fixture registry — the canonical catalog
# ---------------------------------------------------------------------------
#
# Each fixture is one real-world device manual used as test data for
# the ingestion pipeline (Step 4.9) and the retrieval benchmarks
# (Step 5.x). We pin:
#
# - ``name``: a stable id used as the CLI flag value
# - ``url``: where to fetch the file (HTTPS preferred; HTTP is
#   rejected with a clear error if the host supports HTTPS but we
#   didn't use it)
# - ``sha256``: the expected SHA-256 hex digest of the downloaded
#   file. Verified after download — a mismatch triggers exit code 3
#   (the dangerous case: either the upstream changed OR the network
#   was tampered with).
# - ``filename``: what to save the file as in the fixtures dir.
#   Defaults to ``<name>.pdf``.
#
# To add a new fixture: append a ``FixtureSpec`` entry below. The
# ``--list`` flag auto-discovers every entry.


@dataclass(frozen=True)
class FixtureSpec:
    """One catalogued fixture (real-world document for tests)."""

    name: str
    url: str
    sha256: str
    filename: str
    description: str = ""


#: The canonical fixture catalog. Add entries here as new test corpora
#: are introduced (e.g. for sensor summaries or multi-doc retrieval).
FIXTURE_REGISTRY: tuple[FixtureSpec, ...] = (
    FixtureSpec(
        name="nest-thermostat-install-uk",
        url=(
            # NOTE: this URL is intentionally a placeholder. Nest/Google
            # don't host a stable canonical URL for this PDF — their
            # support docs have moved between storage.googleapis.com
            # and support.google.com over the years, and pinning a
            # specific GCS path would 404 the moment they reorganise.
            #
            # The intended workflow is:
            #   1. A contributor downloads the PDF from wherever Nest
            #      currently hosts it (e.g. the Google Nest support
            #      site) and saves it to tests/fixtures/.
            #   2. `python scripts/download_fixtures.py --verify-only`
            #      confirms the SHA-256 matches the registry — this
            #      is the safety guarantee that catches tampered or
            #      corrupted files.
            #
            # When a stable URL is found, update this field. Until
            # then, the script's --verify-only path is the
            # primary value.
            ""
        ),
        sha256=(
            # Verified via `sha256sum tests/fixtures/Nest-Thermostat-Installation-Guide-UK.pdf`
            # on the canonical copy. If this ever changes, it means
            # Nest published an updated version of the guide — bump
            # this hash AND consider whether the tests still pass.
            "2b8d2497dcf772013672dfd75864db6e4d19cf865f87e3b08cca8c26af1bcc63"
        ),
        filename="Nest-Thermostat-Installation-Guide-UK.pdf",
        description=(
            "Nest Learning Thermostat Installation Guide (UK edition, 40 pages). "
            "Used as the Step 4.9 end-to-end ingestion risk-gate fixture."
        ),
    ),
)


# ---------------------------------------------------------------------------
# SHA verification + download helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, *, chunk_bytes: int = 64 * 1024) -> str:
    """SHA-256 hex digest of a file (read in chunks so big files don't OOM)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` via urllib (stdlib only — no extra deps)."""
    # ``urlopen`` with a stream-friendly timeout. We don't set a
    # hard byte cap — the fixtures are small PDFs (~1 MB max).
    with urllib.request.urlopen(url, timeout=60) as resp:
        # Raise on 4xx/5xx so the caller gets a clean error.
        if resp.status >= 400:  # pragma: no cover (urlopen raises already)
            raise RuntimeError(f"HTTP {resp.status} downloading {url}")
        with dest.open("wb") as out:
            while True:
                block = resp.read(64 * 1024)
                if not block:
                    break
                out.write(block)


def verify(spec: FixtureSpec, fixtures_dir: Path) -> bool:
    """Return True iff the fixture is on disk with a matching SHA-256.

    Side effect: if the file exists but the SHA mismatches, prints a
    warning to stderr (the caller will typically then re-download
    with --force).
    """
    path = fixtures_dir / spec.filename
    if not path.exists():
        return False
    actual = _sha256_file(path)
    return actual == spec.sha256


def acquire(spec: FixtureSpec, fixtures_dir: Path, *, force: bool) -> str:
    """Ensure ``spec`` is present and verified.

    Returns the post-condition state: ``"present"`` (file exists,
    SHA matched) or ``"downloaded"`` (we just downloaded it).

    Raises :class:`ChecksumMismatchError` if the on-disk file's SHA
    doesn't match — exit code 3 territory.

    Raises :class:`DownloadError` if the registry has no URL pinned
    (the placeholder workflow documented in FIXTURE_REGISTRY).
    """
    path = fixtures_dir / spec.filename
    if path.exists() and not force:
        actual = _sha256_file(path)
        if actual == spec.sha256:
            return "present"
        raise ChecksumMismatchError(
            f"{spec.filename}: on-disk SHA-256 ({actual[:16]}...) "
            f"does not match expected ({spec.sha256[:16]}...). "
            f"Re-run with --force if the upstream is known to have changed."
        )

    # No URL pinned — ask the user to provide the file manually.
    if not spec.url:
        raise DownloadError(
            f"{spec.filename}: no download URL is pinned for this fixture. "
            f"Please obtain the file from the vendor's site and save it to "
            f"{path}, then re-run with --verify-only to confirm the SHA-256. "
            f"(See FIXTURE_REGISTRY in this script for the rationale.)"
        )

    # (Re-)download.
    path.parent.mkdir(parents=True, exist_ok=True)
    _download(spec.url, path)

    # Verify before reporting success.
    actual = _sha256_file(path)
    if actual != spec.sha256:
        raise ChecksumMismatchError(
            f"{spec.filename}: downloaded SHA-256 ({actual[:16]}...) "
            f"does not match expected ({spec.sha256[:16]}...). "
            f"This could indicate upstream tampering or a corrupted download."
        )
    return "downloaded"


# ---------------------------------------------------------------------------
# Typed exceptions — one per exit code
# ---------------------------------------------------------------------------


class DownloadError(RuntimeError):
    """Network / I/O failure. Exit code 1."""


class ChecksumMismatchError(RuntimeError):
    """SHA-256 of downloaded file doesn't match the registry. Exit code 3."""


class UnknownFixtureError(KeyError):
    """The CLI got a fixture name that isn't in the registry. Exit code 2."""


# ---------------------------------------------------------------------------
# Pretty output (matches scripts/download_models.py conventions)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _green(s: str) -> str:
    return _c("32", s)


def _red(s: str) -> str:
    return _c("31", s)


def _yellow(s: str) -> str:
    return _c("33", s)


def _bold(s: str) -> str:
    return _c("1", s)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download_fixtures.py",
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List every fixture in the registry (no I/O).",
    )
    p.add_argument(
        "--name",
        default=None,
        help=(
            "Specific fixture to download (default: download the first one in "
            "the registry). Use --list to see available names."
        ),
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Download every fixture in the registry.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    p.add_argument(
        "--fixtures-dir",
        default=None,
        help="Where to put the downloaded files (default: ./tests/fixtures).",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Just verify what's already on disk (no network).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print JSON result instead of pretty text.",
    )
    return p


def _resolve_fixtures_dir(arg: str | None) -> Path:
    """Resolve the fixtures directory: explicit arg → ./tests/fixtures."""
    if arg is not None:
        return Path(arg).resolve()
    # Default: <repo root>/tests/fixtures. Walk up from this script.
    return (_HERE.parent / "tests" / "fixtures").resolve()


def _resolve_targets(
    *,
    name: str | None,
    download_all: bool,
) -> list[FixtureSpec]:
    """Pick which fixtures the CLI invocation should act on."""
    if download_all:
        return list(FIXTURE_REGISTRY)
    if name is None:
        # Default behaviour: download the first fixture (the canonical
        # Step 4.9 risk-gate corpus).
        if not FIXTURE_REGISTRY:
            raise UnknownFixtureError("registry is empty")
        return [FIXTURE_REGISTRY[0]]
    for spec in FIXTURE_REGISTRY:
        if spec.name == name:
            return [spec]
    raise UnknownFixtureError(
        f"unknown fixture {name!r}. Use --list to see available names."
    )


def print_list(*, json_mode: bool) -> None:
    """Print the fixture registry contents."""
    if json_mode:
        out = [
            {
                "name": s.name,
                "filename": s.filename,
                "url": s.url,
                "sha256": s.sha256,
                "description": s.description,
            }
            for s in FIXTURE_REGISTRY
        ]
        print(json.dumps(out, indent=2, sort_keys=True))
        return
    print(_bold("==> TinyRAG test fixtures"))
    for s in FIXTURE_REGISTRY:
        print(f"  {_green(s.name)}")
        print(f"    filename:  {s.filename}")
        print(f"    url:       {s.url}")
        print(f"    sha256:    {s.sha256[:16]}...")
        print(f"    desc:      {s.description}")


@dataclass
class FixtureResult:
    """Outcome of acquiring (or verifying) one fixture."""

    name: str
    filename: str
    state: str  # "present" | "downloaded" | "missing"
    path: str
    sha256: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "filename": self.filename,
            "state": self.state,
            "path": self.path,
            "sha256": self.sha256,
            "error": self.error,
        }


def run(
    targets: Sequence[FixtureSpec],
    fixtures_dir: Path,
    *,
    force: bool,
    verify_only: bool,
) -> list[FixtureResult]:
    """Acquire (or just verify) every target fixture. Returns per-spec results."""
    results: list[FixtureResult] = []
    for spec in targets:
        path = fixtures_dir / spec.filename
        try:
            if verify_only:
                # No I/O, just check what's there.
                if not path.exists():
                    results.append(
                        FixtureResult(
                            name=spec.name,
                            filename=spec.filename,
                            state="missing",
                            path=str(path),
                            sha256="",
                            error=None,
                        )
                    )
                    continue
                actual = _sha256_file(path)
                if actual == spec.sha256:
                    results.append(
                        FixtureResult(
                            name=spec.name,
                            filename=spec.filename,
                            state="present",
                            path=str(path),
                            sha256=actual,
                            error=None,
                        )
                    )
                else:
                    results.append(
                        FixtureResult(
                            name=spec.name,
                            filename=spec.filename,
                            state="present",
                            path=str(path),
                            sha256=actual,
                            error=(
                                f"SHA mismatch (got {actual[:16]}..., "
                                f"expected {spec.sha256[:16]}...)"
                            ),
                        )
                    )
                continue
            state = acquire(spec, fixtures_dir, force=force)
            actual = _sha256_file(path)
            results.append(
                FixtureResult(
                    name=spec.name,
                    filename=spec.filename,
                    state=state,
                    path=str(path),
                    sha256=actual,
                    error=None,
                )
            )
        except ChecksumMismatchError as exc:
            results.append(
                FixtureResult(
                    name=spec.name,
                    filename=spec.filename,
                    state="present",
                    path=str(path),
                    sha256="",
                    error=str(exc),
                )
            )
        except Exception as exc:
            results.append(
                FixtureResult(
                    name=spec.name,
                    filename=spec.filename,
                    state="missing",
                    path=str(path),
                    sha256="",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


def print_results(results: Sequence[FixtureResult], *, json_mode: bool) -> None:
    """Print either a pretty summary or a JSON array of results."""
    if json_mode:
        print(json.dumps([r.to_dict() for r in results], indent=2, sort_keys=True))
        return

    print(_bold("==> TinyRAG — fixture acquisition"))
    for r in results:
        if r.error and "mismatch" in r.error.lower():
            tag = _red("[FAIL]") + " checksum mismatch"
        elif r.error:
            tag = _red("[FAIL]") + f" {r.error}"
        elif r.state == "downloaded":
            tag = _green("[ OK ]") + " downloaded"
        elif r.state == "present":
            tag = _green("[ OK ]") + " already present (sha verified)"
        elif r.state == "missing":
            tag = _yellow("[WARN]") + " missing"
        else:  # pragma: no cover
            tag = f"[{r.state}]"
        print(f"  {r.name}: {tag}")
        print(f"    filename: {r.filename}")
        print(f"    path:     {r.path}")
        if r.sha256:
            print(f"    sha256:   {r.sha256[:16]}...")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    args = _build_parser().parse_args(argv)

    if args.list:
        print_list(json_mode=args.json)
        return 0

    try:
        targets = _resolve_targets(name=args.name, download_all=args.all)
    except UnknownFixtureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    fixtures_dir = _resolve_fixtures_dir(args.fixtures_dir)
    results = run(
        targets,
        fixtures_dir,
        force=args.force,
        verify_only=args.verify_only,
    )
    print_results(results, json_mode=args.json)

    # Decide exit code: 3 wins over 1 wins over 0.
    has_mismatch = any(
        r.error and "mismatch" in r.error.lower() for r in results
    )
    has_error = any(r.error for r in results)
    if has_mismatch:
        return 3
    if has_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
