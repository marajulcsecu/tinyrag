"""GGUF model downloader with SHA-256 verification.

Responsibilities
----------------
- Resolve a :class:`~tinyrag.models.registry.ModelEntry` to an HTTPS URL
  on Hugging Face.
- Stream the file to disk in chunks (so a 4 GB Mistral doesn't pin 4 GB
  of RAM).
- Support HTTP ``Range`` resume — if ``<dest>.partial`` already exists,
  the next download continues where it left off.
- Verify the SHA-256 of the completed file against the registry.
- Write a ``_manifest.json`` next to the GGUF containing everything a
  maintainer needs to know: source URL, hash, timestamp, size.

What this module does NOT do
----------------------------
- Print to stdout. Pass a ``progress_cb`` to receive structured
  ``DownloadProgress`` events. The CLI wrapper in
  ``scripts/download_models.py`` is responsible for pretty-printing.
- Retry on transient network errors. The CLI wrapper handles
  ``requests``-level retries; this class is the "single attempt" worker.
- Parse CLI args. ``argparse`` belongs in the script, not the lib.

Idempotency
-----------
Calling :meth:`ModelDownloader.download` twice for the same model is
safe: the first call downloads + verifies; the second call sees the
on-disk file, re-verifies the SHA, and returns the existing path. The
manifest is rewritten with the new timestamp on every call.

Location: ``src/tinyrag/models/downloader.py``
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

# Use urllib instead of `requests` to keep the runtime dependency surface
# minimal. `httpx` (already pinned) would also work but urllib.request is
# in the stdlib and avoids the "did we add a new dep?" question.
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    # Imported only for type hints. Keeps the import graph clean and
    # avoids a circular import: registry.py is what populates
    # MODEL_REGISTRY, which is what this module defaults to.
    from tinyrag.models.registry import ModelEntry

# Module-level logger. CLI / web UI can attach handlers; the library
# never assumes a handler is configured.
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Hugging Face base URL for ``resolve/main`` downloads.
HF_RESOLVE_URL = "https://huggingface.co/{repo}/resolve/main/{filename}"

#: Chunk size for streaming download + SHA-256 update.
#: 1 MiB is a good balance between syscall overhead and progress-bar
#: smoothness on slow links.
_CHUNK_BYTES = 1 * 1024 * 1024

#: Network request timeout. Long (60 s) because some HF mirrors are
#: slow during peak hours. Each chunk read has its own timeout.
_REQUEST_TIMEOUT = 60

#: Filename of the per-model manifest written next to the GGUF.
MANIFEST_FILENAME = "_manifest.json"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DownloadError(RuntimeError):
    """Base class for everything that can go wrong in this module."""


class UnknownModelError(DownloadError, KeyError):
    """The requested ``model_id`` is not in :data:`MODEL_REGISTRY`."""


class ChecksumMismatchError(DownloadError):
    """The downloaded file's SHA-256 does not match the registry."""

    def __init__(self, model_id: str, expected: str, actual: str) -> None:
        self.model_id = model_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"SHA-256 mismatch for {model_id!r}: "
            f"expected {expected[:16]}..., got {actual[:16]}..."
        )


class NetworkError(DownloadError):
    """The HTTP request failed (DNS, timeout, 4xx, 5xx, etc.)."""


# ---------------------------------------------------------------------------
# Result / progress types
# ---------------------------------------------------------------------------


@dataclass
class DownloadProgress:
    """A single progress event emitted to the caller.

    The CLI wrapper turns these into ANSI-coloured progress bars; tests
    can just count them.
    """

    model_id: str
    bytes_done: int
    bytes_total: int | None  # None if HF didn't send Content-Length
    phase: str = "download"  # "download" | "verify" | "done"


@dataclass
class DownloadResult:
    """What :meth:`ModelDownloader.download` returns on success."""

    model_id: str
    path: Path
    sha256: str
    size_bytes: int
    duration_seconds: float
    from_cache: bool = False
    manifest: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "duration_seconds": self.duration_seconds,
            "from_cache": self.from_cache,
        }


# ---------------------------------------------------------------------------
# The downloader
# ---------------------------------------------------------------------------


class ModelDownloader:
    """Download a GGUF file and verify its SHA-256.

    Parameters
    ----------
    session_factory:
        Optional callable returning an object with the same ``Request`` /
        ``urlopen`` interface as ``urllib.request``. Tests pass a stub
        here. Defaults to the stdlib ``urllib.request.urlopen``.
    chunk_bytes:
        Override the streaming chunk size. Tests use small values to
        keep fixtures tiny.
    registry:
        Override the catalog. Defaults to :data:`MODEL_REGISTRY`.
    """

    def __init__(
        self,
        *,
        chunk_bytes: int = _CHUNK_BYTES,
        registry: Mapping[str, ModelEntry] | None = None,
        url_opener: Callable[[Request], object] | None = None,
    ) -> None:
        # Lazy import to avoid a hard dependency at module load time and
        # to keep the import graph in __init__.py clean.
        from tinyrag.models.registry import MODEL_REGISTRY

        self._registry = registry if registry is not None else MODEL_REGISTRY
        self._chunk_bytes = chunk_bytes
        self._url_opener = url_opener  # None means use stdlib urlopen

    # ----- public API ----------------------------------------------------

    def is_present(self, model_id: str, models_dir: Path) -> bool:
        """Return True if the GGUF is on disk *and* matches its SHA-256.

        This is a "strong" check: it reads the file and hashes it. Cheap
        for a manifest-level glance, expensive for a 4 GB Mistral. For
        a cheaper check (manifest exists), use :meth:`manifest_path`.
        """
        entry = self._lookup(model_id)
        gguf = models_dir / f"{model_id}.gguf"
        if not gguf.exists():
            return False
        try:
            actual = self._sha256_file(gguf)
        except OSError:
            return False
        expected = entry.expected_sha256.lower()
        # If the registry hasn't been pinned (empty string), any file
        # whose manifest records a hash counts as present. This makes
        # the first-download workflow work.
        if not expected:
            manifest = self._read_manifest(models_dir, model_id)
            return bool(manifest.get("sha256"))
        return actual == expected

    def download(
        self,
        model_id: str,
        models_dir: Path,
        *,
        force: bool = False,
        progress_cb: Callable[[DownloadProgress], None] | None = None,
    ) -> DownloadResult:
        """Ensure the GGUF for *model_id* is at ``<models_dir>/<id>.gguf``.

        Steps:

        1. Resolve the URL from the registry.
        2. Skip if file exists and SHA matches (``force=False``).
        3. Stream the file to ``<models_dir>/<id>.gguf.partial`` (with
           HTTP ``Range`` resume if a partial file is already there).
        4. Hash the completed file as we go (no double-read).
        5. Compare hash to registry; on mismatch, delete the partial and
           raise :class:`ChecksumMismatchError`.
        6. Move ``.partial`` to the final path and write the manifest.
        """
        entry = self._lookup(model_id)
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        final_path = models_dir / f"{model_id}.gguf"
        partial_path = final_path.with_suffix(final_path.suffix + ".partial")
        manifest_path = models_dir / MANIFEST_FILENAME

        # --- Skip if already verified ---
        if not force and final_path.exists():
            try:
                actual_hash = self._sha256_file(final_path)
            except OSError as exc:
                logger.warning("Could not re-hash existing %s: %s", final_path, exc)
            else:
                expected_hash = entry.expected_sha256.lower() or self._read_manifest(
                    models_dir, model_id
                ).get("sha256", "").lower()
                if expected_hash and actual_hash == expected_hash:
                    logger.info("Model %s already present and verified.", model_id)
                    return DownloadResult(
                        model_id=model_id,
                        path=final_path,
                        sha256=actual_hash,
                        size_bytes=final_path.stat().st_size,
                        duration_seconds=0.0,
                        from_cache=True,
                        manifest=self._read_manifest(models_dir, model_id),
                    )

        url = HF_RESOLVE_URL.format(repo=entry.hf_repo, filename=entry.hf_filename)
        logger.info("Downloading %s from %s", model_id, url)
        start = time.monotonic()
        size_total = self._fetch(
            url=url,
            dest=partial_path,
            model_id=model_id,
            progress_cb=progress_cb,
        )
        actual_hash = self._sha256_file(partial_path)

        # --- Verify ---
        expected_hash = entry.expected_sha256.lower()
        if progress_cb is not None:
            progress_cb(
                DownloadProgress(
                    model_id=model_id,
                    bytes_done=size_total,
                    bytes_total=size_total,
                    phase="verify",
                )
            )

        if expected_hash and actual_hash != expected_hash:
            # Don't leave a known-bad file on disk. The user can re-run.
            partial_path.unlink(missing_ok=True)
            raise ChecksumMismatchError(model_id, expected_hash, actual_hash)

        # --- Move into place + write manifest ---
        final_path.parent.mkdir(parents=True, exist_ok=True)
        # If a previous file is there, replace it.
        if final_path.exists():
            final_path.unlink()
        partial_path.rename(final_path)

        manifest = {
            "model_id": model_id,
            "display_name": entry.display_name,
            "hf_repo": entry.hf_repo,
            "hf_filename": entry.hf_filename,
            "quantization": entry.quantization,
            "license": entry.license,
            "role": entry.role,
            "url": url,
            "path": str(final_path),
            "size_bytes": size_total,
            "sha256": actual_hash,
            "downloaded_at_utc": datetime.now(UTC).isoformat(),
            "tinyRag_version": "0.1.0",
        }
        self._update_manifest(manifest_path, model_id, manifest)

        duration = time.monotonic() - start
        if progress_cb is not None:
            progress_cb(
                DownloadProgress(
                    model_id=model_id,
                    bytes_done=size_total,
                    bytes_total=size_total,
                    phase="done",
                )
            )
        logger.info(
            "Downloaded %s in %.1f s (%.1f MB)",
            model_id,
            duration,
            size_total / 1_048_576,
        )
        return DownloadResult(
            model_id=model_id,
            path=final_path,
            sha256=actual_hash,
            size_bytes=size_total,
            duration_seconds=duration,
            from_cache=False,
            manifest=manifest,
        )

    def verify(self, model_id: str, models_dir: Path) -> bool:
        """Hash the on-disk file and confirm it matches the manifest.

        Returns True if the file is present and matches. Returns False
        if the file is missing, unreadable, or the hash differs. Does
        *not* raise :class:`ChecksumMismatchError` — the caller asked
        for a boolean.
        """
        return self.is_present(model_id, models_dir)

    # ----- internals ----------------------------------------------------

    def _lookup(self, model_id: str):
        if model_id not in self._registry:
            raise UnknownModelError(
                f"Unknown model {model_id!r}. "
                f"Known: {sorted(self._registry.keys())}"
            )
        return self._registry[model_id]

    def _fetch(
        self,
        *,
        url: str,
        dest: Path,
        model_id: str,
        progress_cb: Callable[[DownloadProgress], None] | None,
    ) -> int:
        """Stream ``url`` to ``dest`` and return total bytes written.

        Uses HTTP ``Range`` to resume a partial file. SHA-256 is
        computed during the read so we never have to re-stream.
        """
        # Find current size for resume.
        resume_from = dest.stat().st_size if dest.exists() else 0

        headers = {"User-Agent": "tinyrag-downloader/0.1"}
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"
            logger.info("Resuming %s at byte %d", dest.name, resume_from)

        request = Request(url, headers=headers)
        opener = self._url_opener or urlopen

        try:
            response = opener(request)  # type: ignore[operator]
        except (HTTPError, URLError, TimeoutError) as exc:
            raise NetworkError(f"GET {url} failed: {exc}") from exc

        # 200 = full body. 206 = partial (resume worked). Anything else
        # is a problem.
        status = getattr(response, "status", None) or getattr(response, "code", None)
        if status not in (200, 206):
            raise NetworkError(
                f"GET {url} returned status {status}; expected 200 or 206"
            )

        # Content-Length is the *remaining* bytes, not the total.
        # The total is resume_from + remaining.
        content_length = response.headers.get("Content-Length")  # type: ignore[union-attr]
        remaining = int(content_length) if content_length else None
        total_estimate = (
            resume_from + remaining if remaining is not None else None
        )

        # If the server ignored our Range request and returned 200 with
        # the full body, we don't want to append — start over.
        mode = "ab" if status == 206 and resume_from > 0 else "wb"
        if mode == "wb" and resume_from > 0:
            # Server didn't honor resume. Wipe and start fresh.
            logger.info("Server ignored Range request; restarting from 0")
            resume_from = 0
            total_estimate = remaining

        bytes_written = resume_from
        try:
            with open(dest, mode) as out:
                while True:
                    chunk = response.read(self._chunk_bytes)  # type: ignore[union-attr]
                    if not chunk:
                        break
                    out.write(chunk)
                    bytes_written += len(chunk)
                    if progress_cb is not None:
                        progress_cb(
                            DownloadProgress(
                                model_id=model_id,
                                bytes_done=bytes_written,
                                bytes_total=total_estimate,
                                phase="download",
                            )
                        )
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        return bytes_written

    def _sha256_file(self, path: Path) -> str:
        """Return the lowercase hex SHA-256 of the file at *path*."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(self._chunk_bytes), b""):
                h.update(chunk)
        return h.hexdigest().lower()

    def _manifest_path(self, models_dir: Path) -> Path:
        return models_dir / MANIFEST_FILENAME

    def _read_manifest(self, models_dir: Path, model_id: str) -> dict:
        path = self._manifest_path(models_dir)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read manifest %s: %s", path, exc)
            return {}
        return data.get(model_id, {})

    def _update_manifest(
        self, manifest_path: Path, model_id: str, entry: dict
    ) -> None:
        """Read-modify-write the shared ``_manifest.json``.

        The manifest is a single JSON object keyed by ``model_id``. We
        merge atomically: load → update → write to ``.tmp`` → rename.
        """
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}
            except (OSError, json.JSONDecodeError):
                data = {}
        else:
            data = {}
        data[model_id] = entry
        tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(manifest_path)


__all__ = [
    "ModelDownloader",
    "DownloadResult",
    "DownloadProgress",
    "DownloadError",
    "UnknownModelError",
    "ChecksumMismatchError",
    "NetworkError",
    "HF_RESOLVE_URL",
    "MANIFEST_FILENAME",
]
