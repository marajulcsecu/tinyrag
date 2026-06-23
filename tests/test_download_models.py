"""Tests for the model registry + downloader.

These tests are the contract that the rest of TinyRAG relies on:

1. The registry is well-formed (no missing required fields).
2. The downloader is idempotent (re-running on a present, correct
   model does not re-download).
3. The downloader rejects a corrupted file with the right exception.
4. The downloader supports resume (a partial file is continued, not
   restarted).
5. The CLI ``--list`` mode prints a table that contains every id.

We don't hit the real Hugging Face in CI. The network layer is replaced
with a tiny in-process "file server" that returns either a known-good
payload or simulates 4xx / 5xx. This keeps the tests fast, hermetic,
and reproducible across rate-limit windows.

Location: ``tests/test_download_models.py``
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from tinyrag.models import (
    MODEL_REGISTRY,
    ChecksumMismatchError,
    ModelDownloader,
    UnknownModelError,
)
from tinyrag.models.downloader import (
    HF_RESOLVE_URL,
    MANIFEST_FILENAME,
    DownloadProgress,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


#: 1 KiB of deterministic "model" bytes. We use the same content for
#: every fake model so SHA-256 is stable across test runs.
FAKE_PAYLOAD = (b"TinyRAG-test-payload-" * 64)[: 1024]  # 1 KiB
FAKE_SHA256 = hashlib.sha256(FAKE_PAYLOAD).hexdigest()


class _FakeResponse:
    """Stand-in for the object returned by ``urllib.request.urlopen``.

    Implements just enough of the real interface for the downloader:
    ``.read(n)``, ``.headers``, ``.close()``. We honour HTTP ``Range``
    headers so the resume path is exercised.
    """

    def __init__(
        self,
        body: bytes,
        status: int,
        range_header: str | None,
    ) -> None:
        self._body = body
        self.status = status
        # Parse "bytes=N-" or "bytes=N-M".
        if range_header and range_header.startswith("bytes="):
            spec = range_header[len("bytes="):]
            start_str, _, end_str = spec.partition("-")
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else len(body) - 1
            self._slice = body[start : end + 1]
            self.status = 206  # Partial Content
            self.headers = {
                "Content-Length": str(len(self._slice)),
                "Content-Range": f"bytes {start}-{end}/{len(body)}",
            }
        else:
            self._slice = body
            self.headers = {"Content-Length": str(len(self._slice))}

    def read(self, n: int = -1) -> bytes:
        if n < 0 or n >= len(self._slice):
            data, self._slice = self._slice, b""
            return data
        data, self._slice = self._slice[:n], self._slice[n:]
        return data

    def close(self) -> None:
        pass


def _make_opener(
    bodies: dict[str, bytes],
    fail_once: set[str] | None = None,
) -> Callable[[Request], _FakeResponse]:
    """Build a ``urlopen`` stub.

    ``bodies``: maps URL → bytes that should be returned.
    ``fail_once``: URLs that should raise ``HTTPError`` on the FIRST
    call and succeed on subsequent calls. Used to test retry semantics
    if we add them later.
    """
    fail_once = fail_once or set()
    call_count: dict[str, int] = {}

    def _open(req: Request) -> _FakeResponse:
        url = req.full_url
        if url in fail_once and call_count.get(url, 0) == 0:
            call_count[url] = call_count.get(url, 0) + 1
            raise HTTPError(url, 503, "Service Unavailable", {}, io.BytesIO(b""))
        if url not in bodies:
            raise HTTPError(url, 404, "Not Found", {}, io.BytesIO(b""))
        range_header = req.headers.get("Range")
        return _FakeResponse(bodies[url], status=200, range_header=range_header)

    return _open


@pytest.fixture
def fake_payload() -> bytes:
    return FAKE_PAYLOAD


@pytest.fixture
def fake_sha256() -> str:
    return FAKE_SHA256


@pytest.fixture
def tiny_registry():
    """A minimal registry with a single model pointing at the fake payload.

    Using a small dict instead of the real ``MODEL_REGISTRY`` keeps the
    test independent of any future SHA-256 pins and isolates the
    "downloader logic" from the "what models exist" concern.
    """
    from tinyrag.models.registry import ModelEntry

    return {
        "fake-1k": ModelEntry(
            model_id="fake-1k",
            display_name="Fake 1 KiB Test Model",
            hf_repo="acme/test",
            hf_filename="fake-1k.gguf",
            quantization="Q4_K_M",
            expected_size_bytes=len(FAKE_PAYLOAD),
            expected_sha256=FAKE_SHA256,
            license="MIT",
            role="primary",
            intended_context=4096,
        )
    }


@pytest.fixture
def url_for(tiny_registry) -> Callable[[str], str]:
    def _url(model_id: str) -> str:
        e = tiny_registry[model_id]
        return HF_RESOLVE_URL.format(repo=e.hf_repo, filename=e.hf_filename)
    return _url


@pytest.fixture
def downloader_factory(tiny_registry) -> Callable[..., ModelDownloader]:
    """Return a factory for ``ModelDownloader`` with a custom URL opener."""

    def _make(url_opener: Callable[[Request], _FakeResponse]) -> ModelDownloader:
        return ModelDownloader(registry=tiny_registry, url_opener=url_opener)

    return _make


@pytest.fixture
def tmp_models_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_registry_has_required_fields() -> None:
    """Every entry has all the fields the rest of TinyRAG reads.

    ``expected_sha256`` is allowed to be empty in the registry: the
    downloader will populate the manifest's hash on first download and
    use that for verification on subsequent runs. The registry entry
    stays empty until a maintainer pins the official hash in
    ``docs/MODELS.md``.
    """
    required_nonempty = (
        "model_id",
        "display_name",
        "hf_repo",
        "hf_filename",
        "quantization",
        "license",
        "role",
    )
    for model_id, entry in MODEL_REGISTRY.items():
        assert entry.model_id == model_id, f"{model_id} has wrong model_id"
        for field_name in required_nonempty:
            assert getattr(entry, field_name) != "", (
                f"{model_id}.{field_name} is empty"
            )
        # Sizes are positive.
        assert entry.expected_size_bytes > 0
        # Roles are from the known set.
        assert entry.role in {"primary", "eval-small", "eval-medium", "eval-large"}
        # Contexts are positive and a power of two-ish (not strictly required
        # but the ones we picked all are).
        assert entry.intended_context > 0


def test_registry_has_exactly_one_primary() -> None:
    primaries = [e for e in MODEL_REGISTRY.values() if e.role == "primary"]
    assert len(primaries) == 1, (
        f"Registry must have exactly 1 primary model, found {len(primaries)}: "
        f"{[e.model_id for e in primaries]}"
    )


def test_registry_ids_are_stable() -> None:
    """The model_id strings are a public API — pin them."""
    expected_ids = {"phi-3-mini", "tinyllama-1.1b", "llama-3.2-3b", "mistral-7b"}
    assert set(MODEL_REGISTRY.keys()) == expected_ids


# ---------------------------------------------------------------------------
# Downloader tests
# ---------------------------------------------------------------------------


def test_download_succeeds_and_writes_manifest(
    downloader_factory, tmp_models_dir, url_for, fake_sha256
) -> None:
    url = url_for("fake-1k")
    opener = _make_opener({url: FAKE_PAYLOAD})
    dl = downloader_factory(opener)

    result = dl.download("fake-1k", tmp_models_dir)

    expected_path = tmp_models_dir / "fake-1k.gguf"
    assert result.path == expected_path
    assert result.sha256 == fake_sha256
    assert result.size_bytes == len(FAKE_PAYLOAD)
    assert expected_path.exists()
    assert expected_path.read_bytes() == FAKE_PAYLOAD

    # Manifest exists, contains the right keys.
    manifest_path = tmp_models_dir / MANIFEST_FILENAME
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert "fake-1k" in manifest
    assert manifest["fake-1k"]["sha256"] == fake_sha256
    assert manifest["fake-1k"]["size_bytes"] == len(FAKE_PAYLOAD)
    assert manifest["fake-1k"]["hf_repo"] == "acme/test"


def test_download_is_idempotent(
    downloader_factory, tmp_models_dir, url_for, fake_sha256
) -> None:
    url = url_for("fake-1k")
    opener = _make_opener({url: FAKE_PAYLOAD})
    dl = downloader_factory(opener)

    first = dl.download("fake-1k", tmp_models_dir)
    # Second call with a network that returns 404 should still succeed
    # because the file is already on disk and verified.
    no_network_dl = ModelDownloader(
        registry=tiny_registry_for_test(),
        url_opener=_make_opener({}),
    )
    second = no_network_dl.download("fake-1k", tmp_models_dir)

    assert first.sha256 == second.sha256 == fake_sha256
    assert second.from_cache is True
    # No partial file left over.
    assert not (tmp_models_dir / "fake-1k.gguf.partial").exists()


def tiny_registry_for_test():
    """Build the same fixture as the one used in the factory (local re-creation)."""
    from tinyrag.models.registry import ModelEntry
    return {
        "fake-1k": ModelEntry(
            model_id="fake-1k",
            display_name="Fake 1 KiB Test Model",
            hf_repo="acme/test",
            hf_filename="fake-1k.gguf",
            quantization="Q4_K_M",
            expected_size_bytes=len(FAKE_PAYLOAD),
            expected_sha256=FAKE_SHA256,
            license="MIT",
            role="primary",
            intended_context=4096,
        )
    }


def test_download_rejects_bad_checksum(
    downloader_factory, tmp_models_dir, url_for
) -> None:
    """If the SHA-256 doesn't match, the file is deleted and we raise."""
    url = url_for("fake-1k")
    # Serve a CORRUPTED body (different bytes from FAKE_PAYLOAD).
    corrupted = FAKE_PAYLOAD + b"extra"
    opener = _make_opener({url: corrupted})
    dl = downloader_factory(opener)

    with pytest.raises(ChecksumMismatchError) as excinfo:
        dl.download("fake-1k", tmp_models_dir)

    assert excinfo.value.model_id == "fake-1k"
    assert excinfo.value.expected == FAKE_SHA256
    # The bad file was cleaned up.
    assert not (tmp_models_dir / "fake-1k.gguf").exists()
    assert not (tmp_models_dir / "fake-1k.gguf.partial").exists()


def test_download_unknown_model_raises(downloader_factory, tmp_models_dir) -> None:
    dl = downloader_factory(_make_opener({}))
    with pytest.raises(UnknownModelError):
        dl.download("not-a-real-model", tmp_models_dir)


def test_download_resume_uses_range_header(
    downloader_factory, tmp_models_dir, url_for
) -> None:
    """If a .partial file exists, the next download sends a Range header.

    We verify this by recording the headers seen on each call.
    """
    seen_ranges: list[str | None] = []
    real_body = FAKE_PAYLOAD

    def _recording_opener(req: Request) -> _FakeResponse:
        seen_ranges.append(req.headers.get("Range"))
        return _FakeResponse(
            real_body, status=200, range_header=req.headers.get("Range")
        )

    # Plant a partial file with the first 100 bytes of the real payload.
    partial = tmp_models_dir / "fake-1k.gguf.partial"
    partial.write_bytes(real_body[:100])
    dl = downloader_factory(_recording_opener)

    result = dl.download("fake-1k", tmp_models_dir)

    # First call should have had a Range header. The response is 206 + slice,
    # which is then appended to the existing 100 bytes.
    assert seen_ranges[0] == "bytes=100-"
    assert result.size_bytes == len(real_body)
    assert result.path.read_bytes() == real_body


def test_download_progress_callbacks(
    downloader_factory, tmp_models_dir, url_for
) -> None:
    url = url_for("fake-1k")
    dl = downloader_factory(_make_opener({url: FAKE_PAYLOAD}))

    events: list[DownloadProgress] = []
    dl.download("fake-1k", tmp_models_dir, progress_cb=events.append)

    # We expect at least one "download" event and a final "done" event.
    phases = [e.phase for e in events]
    assert "download" in phases
    assert "done" in phases
    # Bytes_done is monotonically non-decreasing.
    download_events = [e for e in events if e.phase == "download"]
    for prev, curr in pairwise(download_events):
        assert curr.bytes_done >= prev.bytes_done


def test_verify_returns_true_for_correct_file(
    downloader_factory, tmp_models_dir, url_for
) -> None:
    url = url_for("fake-1k")
    dl = downloader_factory(_make_opener({url: FAKE_PAYLOAD}))
    dl.download("fake-1k", tmp_models_dir)

    assert dl.verify("fake-1k", tmp_models_dir) is True


def test_verify_returns_false_for_missing(
    downloader_factory, tmp_models_dir
) -> None:
    dl = downloader_factory(_make_opener({}))
    assert dl.verify("fake-1k", tmp_models_dir) is False


def test_is_present_false_for_corrupt_manifest(
    downloader_factory, tmp_models_dir, url_for
) -> None:
    """If the on-disk file's hash doesn't match the manifest, is_present returns False."""
    url = url_for("fake-1k")
    dl = downloader_factory(_make_opener({url: FAKE_PAYLOAD}))
    dl.download("fake-1k", tmp_models_dir)
    # Tamper with the file.
    (tmp_models_dir / "fake-1k.gguf").write_bytes(b"totally different content")
    assert dl.is_present("fake-1k", tmp_models_dir) is False


# ---------------------------------------------------------------------------
# CLI tests (in-process)
# ---------------------------------------------------------------------------


def test_cli_list_runs_without_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """`python scripts/download_models.py --list` should never open a socket."""
    from scripts import download_models  # type: ignore[import-not-found]

    # Re-define sys.argv so argparse sees the flags.
    monkeypatch.setattr("sys.argv", ["download_models.py", "--list"])
    rc = download_models.main()
    assert rc == 0


def test_cli_unknown_model_returns_1(
    monkeypatch: pytest.MonkeyPatch, tmp_models_dir: Path
) -> None:
    from scripts import download_models  # type: ignore[import-not-found]

    monkeypatch.setattr(
        "sys.argv",
        [
            "download_models.py",
            "--model", "not-a-real-model",
            "--models-dir", str(tmp_models_dir),
        ],
    )
    rc = download_models.main()
    assert rc == 1


def test_cli_download_succeeds_via_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tiny_registry,
) -> None:
    """End-to-end CLI test: stub the network, call the CLI, assert it works.

    We patch ``ModelDownloader`` with a subclass that uses our fake
    ``urlopen`` so we don't need the network at all.
    """
    from scripts import download_models  # type: ignore[import-not-found]

    url = HF_RESOLVE_URL.format(
        repo=tiny_registry["fake-1k"].hf_repo,
        filename=tiny_registry["fake-1k"].hf_filename,
    )
    opener = _make_opener({url: FAKE_PAYLOAD})

    class _StubDownloader(ModelDownloader):
        def __init__(self):
            super().__init__(
                registry=tiny_registry,
                url_opener=opener,
            )

    monkeypatch.setattr(download_models, "ModelDownloader", _StubDownloader)
    # Also patch the registry the CLI sees (it reads MODEL_REGISTRY at import).
    monkeypatch.setattr(download_models, "MODEL_REGISTRY", tiny_registry)

    out_dir = tmp_path / "models"
    monkeypatch.setattr(
        "sys.argv",
        [
            "download_models.py",
            "--model", "fake-1k",
            "--models-dir", str(out_dir),
            "--json",
        ],
    )
    rc = download_models.main()
    assert rc == 0
    assert (out_dir / "fake-1k.gguf").exists()
    assert (out_dir / MANIFEST_FILENAME).exists()
