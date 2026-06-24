"""Embedding models — text → dense vectors.

This module is the **text-to-vector** seam of the ingestion pipeline.
It wraps a sentence-transformers model (or any other embedding
backend) behind a common :class:`EmbeddingModel` Protocol so the
pipeline can treat models polymorphically — and so swapping
``all-MiniLM-L6-v2`` for ``bge-small-en-v1.5`` is a one-line
config change, not a refactor.

Architecture contract
---------------------
The architecture doc §6.2 pins the Protocol to::

    class EmbeddingModel(Protocol):
        @property
        def dimension(self) -> int: ...
        def embed(self, texts: list[str]) -> list[list[float]]: ...

This module provides:

- :class:`EmbeddingModel` — the Protocol itself.
- :class:`SentenceTransformerEmbedder` — concrete implementation
  wrapping ``sentence-transformers``.
- :class:`FakeEmbedder` — deterministic in-process stand-in for
  tests (no model download, no PyTorch).
- :class:`EmbeddingError` (and subclasses) — typed exception
  hierarchy so callers can catch one base type.

Why a Protocol and not an ABC?
------------------------------
Follows the architecture doc's "Protocol over ABC" rule (§6
intro). ``@runtime_checkable`` lets ``isinstance(x, EmbeddingModel)``
work for duck-typed test fakes — see ``test_embedder.py``.

Why lazy model load?
--------------------
``sentence-transformers`` + torch + the model weights add up to
~250 MB of memory and ~1-2 s of import time when actually loaded.
Constructing an ``SentenceTransformerEmbedder`` (e.g. in a test
fixture, in the FastAPI startup hook, or in ``scripts/ingest.py``)
should not pay that cost until the first real ``.embed()`` call.
Tests that only need the Protocol type or the ``.dimension``
property never trigger a download.

Why a ``FakeEmbedder`` in production code?
------------------------------------------
A deterministic stub is essential for hermetic tests — see
``test_embedder.py``. Putting it in this module (instead of in
``conftest.py``) means other contributors can use it as a
no-network fallback when they need to run a script on a machine
without the model downloaded. ``FakeEmbedder`` is also handy for
the upcoming Step 4.9 ``IngestionPipeline`` tests: the pipeline
needs *an* ``EmbeddingModel``; using a fake keeps the test fast
and offline.

Why assert dimension at load time?
----------------------------------
The whole point of embedding-based retrieval is that all vectors
live in the same dimensional space. If the user changes
``embedding.model_name`` in ``config.yaml`` from
``all-MiniLM-L6-v2`` (384-dim) to ``bge-large-en-v1.5`` (1024-dim)
but forgets to update ``embedding.dimension``, the new model's
vectors would silently be incompatible with the FAISS index built
from the old ones — and retrieval would just return noise.

We catch this by asserting that the actual model output
dimension matches ``EmbeddingSettings.dimension`` (if set) at load
time. A mismatch raises :class:`EmbeddingDimensionMismatchError`
with both numbers in the message — easy to diagnose.

Public surface
--------------
- :class:`EmbeddingModel` — Protocol.
- :class:`SentenceTransformerEmbedder` — concrete (real model).
- :class:`FakeEmbedder` — concrete (deterministic, for tests).
- :class:`EmbeddingError` and subclasses.

Location: ``src/tinyrag/ingestion/embedder.py``
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from tinyrag.config import EmbeddingSettings

# ----------------------------------------------------------------------------
# Public exceptions
# ----------------------------------------------------------------------------


class EmbeddingError(Exception):
    """Base class for every embedder failure.

    The ingestion pipeline catches this once and decides whether
    to retry / skip / 5xx-the-upload. Always subclass this
    rather than raising a bare ``Exception`` so downstream code
    can ``except EmbeddingError`` once.
    """

    def __init__(self, message: str, *, model_name: str | None = None) -> None:
        super().__init__(message)
        # Preserve the offending model name on the exception so log
        # lines and API responses can show what was attempted.
        self.model_name: str | None = model_name


class EmbeddingModelNotFoundError(EmbeddingError):
    """The configured model name could not be loaded.

    Raised when ``sentence_transformers.SentenceTransformer(...)``
    fails (HF model ID typo, no internet, missing local path,
    etc.). The pipeline should map this to HTTP 503 ("Service
    Unavailable") — it's a transient infrastructure problem, not
    a bad input.
    """


class EmbeddingDimensionMismatchError(EmbeddingError):
    """The model's actual output dimension doesn't match ``EmbeddingSettings.dimension``.

    Raised at first model load when ``EmbeddingSettings.dimension``
    is set (FR — see config) but doesn't match the model's actual
    output. The fix is to update ``embedding.dimension`` in
    ``config.yaml`` (or remove it to disable the check).
    """


# ----------------------------------------------------------------------------
# The Protocol
# ----------------------------------------------------------------------------


@runtime_checkable
class EmbeddingModel(Protocol):
    """Anything that can turn text into a dense vector.

    Concrete implementations: :class:`SentenceTransformerEmbedder`
    (real) and :class:`FakeEmbedder` (deterministic test stub).
    The ``@runtime_checkable`` decorator lets tests verify
    duck-typing (``isinstance(thing, EmbeddingModel)``) without
    requiring inheritance — see architecture doc §6.
    """

    @property
    def dimension(self) -> int:
        """The dimensionality of vectors produced by :meth:`embed`.

        Implementations may load the underlying model lazily, so
        the first access can be expensive — see
        :class:`SentenceTransformerEmbedder`. ``FakeEmbedder`` is
        a pure attribute, no I/O.
        """
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input.

        Parameters
        ----------
        texts:
            The texts to embed. Empty list returns an empty list.
            ``[""]`` returns a single vector (sentence-transformers
            handles empty strings — they typically embed to ~0).

        Returns
        -------
        list[list[float]]
            ``len(texts)`` vectors, each of length
            ``self.dimension``. Vectors are float lists (not
            numpy arrays) so they're JSON-serialisable for the
            SQLite metadata store.

        Raises
        ------
        EmbeddingError
            Subclass indicating the specific failure.
        """
        ...


# ----------------------------------------------------------------------------
# FakeEmbedder — deterministic stub for tests and offline dev
# ----------------------------------------------------------------------------


class FakeEmbedder:
    """Deterministic, zero-dependency embedder for tests.

    The vector for a text is ``hashlib.sha256(text.encode()).digest()``
    reinterpreted as floats, truncated (or padded) to ``dimension``.
    Two runs of the same text always produce the same vector;
    two different texts almost certainly produce different vectors
    (collision probability 1 in 2^256 per dimension pair).

    Why deterministic and not random?
    ---------------------------------
    Tests need reproducibility — a randomly-seeded embedder would
    fail intermittently when the seed wasn't fixed. SHA-256 is
    fast, well-distributed, and always available in the stdlib.

    This is NOT a useful embedder for retrieval (it has no
    semantic meaning — "dog" and "cat" get random hashes, not
    nearby vectors). It's purely a stand-in so the rest of the
    pipeline can be exercised without a 250 MB model download.
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return ``len(texts)`` deterministic vectors."""
        result: list[list[float]] = []
        for text in texts:
            # SHA-256 → 32 bytes → 32 floats (one byte each). We
            # then *repeat* those 32 floats to fill the requested
            # dimension (or truncate if ``dimension < 32``).
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            # Map each byte (0..255) to a float in [-1, 1] so the
            # vector has zero mean and unit-ish variance.
            base = [b / 127.5 - 1.0 for b in digest]
            # Tile to fill dimension.
            if self._dimension <= len(base):
                vec = base[: self._dimension]
            else:
                vec = list(base)
                # Repeat base vectors cyclically until we hit dim.
                idx = 0
                while len(vec) < self._dimension:
                    vec.append(base[idx % len(base)])
                    idx += 1
            # Normalise to unit length so cosine similarity is
            # well-defined. Real sentence-transformers also returns
            # L2-normalised vectors by default; matching that
            # behaviour keeps tests that use the FAISS index honest.
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            result.append(vec)
        return result


# ----------------------------------------------------------------------------
# SentenceTransformerEmbedder — real model wrapper
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _LoadedModel:
    """The result of a successful model load.

    Tiny internal dataclass — holds the loaded ``SentenceTransformer``
    instance and its measured output dimension. Frozen so we
    don't accidentally mutate the model state.

    Not in the public surface; this is an implementation detail
    of :class:`SentenceTransformerEmbedder`.
    """

    st_model: Any  # sentence_transformers.SentenceTransformer — typing.Any avoids the heavy import
    actual_dimension: int


class SentenceTransformerEmbedder:
    """Embedding model backed by a sentence-transformers checkpoint.

    Loads the model **lazily** on first ``.embed()`` (or first
    ``.dimension`` access). This keeps construction cheap
    (important for tests, FastAPI startup, and ``scripts/ingest.py``
    when the model is already in the HF cache).

    Parameters
    ----------
    settings:
        The :class:`EmbeddingSettings` from ``tinyrag.config``.
        ``settings.model_name`` is the HuggingFace model ID
        (e.g. ``"sentence-transformers/all-MiniLM-L6-v2"``) or
        a local path. ``settings.cache_dir`` controls where the
        downloaded weights live on disk.

    Raises
    ------
    EmbeddingError
        Model load failed (network, bad ID, missing path).
        ``EmbeddingDimensionMismatchError`` if ``settings.dimension``
        is set and doesn't match the loaded model's actual output.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        # Defensive copies. EmbeddingSettings is frozen, so this is
        # just local references. We extract the values we need
        # up front so the hot path doesn't keep dereferencing.
        self._model_name: str = settings.model_name
        self._device: str = settings.device.value
        self._batch_size: int = settings.batch_size
        self._cache_dir: Path = Path(settings.cache_dir)
        self._expected_dimension: int | None = (
            settings.dimension if hasattr(settings, "dimension") else None
        )

        # Lazy-loaded on first use. ``None`` means "not loaded yet".
        self._loaded: _LoadedModel | None = None

    # ---- public surface ----------------------------------------------------

    @property
    def model_name(self) -> str:
        """The configured model name (HF ID or local path)."""
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        """``True`` once the underlying model has been loaded into memory.

        Tests and the FastAPI startup hook use this to decide whether
        to show a "loading model…" message.
        """
        return self._loaded is not None

    @property
    def dimension(self) -> int:
        """The model's actual output dimension (loads the model on first access)."""
        return self._ensure_loaded().actual_dimension

    def load(self) -> None:
        """Eagerly load the underlying model.

        Optional — :meth:`embed` and :meth:`dimension` load
        transparently. ``load()`` exists so callers that want to
        "warm up" at startup (and surface load errors early) can
        do so explicitly. The ingestion pipeline in Step 4.9 will
        call this once before processing a batch of documents.
        """
        self._ensure_loaded()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Parameters
        ----------
        texts:
            List of strings to embed. Empty list returns ``[]``
            without triggering a model load (no work to do).

        Returns
        -------
        list[list[float]]
            One vector per text, each of length ``self.dimension``.
            Vectors are JSON-safe Python floats (not numpy scalars).

        Raises
        ------
        EmbeddingModelNotFoundError
            Model load failed (bad HF ID, no internet, etc.).
        EmbeddingDimensionMismatchError
            Model output dimension ≠ configured dimension.
        """
        if not texts:
            return []
        loaded = self._ensure_loaded()
        # ``encode`` is the single hot path. ``convert_to_numpy=True``
        # is the default; ``normalize_embeddings=True`` produces
        # unit vectors (matches what FAISS cosine-similarity search
        # expects). ``batch_size`` from settings controls memory.
        # ``show_progress_bar=False`` keeps our structured logs clean.
        vectors = loaded.st_model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # ``vectors`` is a numpy array (N, dimension). Convert to
        # list-of-lists of Python floats so the result is JSON-safe
        # (the metadata store stores these as JSON text in SQLite).
        return [[float(x) for x in row] for row in vectors]

    # ---- internal helpers --------------------------------------------------

    def _ensure_loaded(self) -> _LoadedModel:
        """Load the model if not already loaded; return the loaded model.

        Raises
        ------
        EmbeddingModelNotFoundError
            sentence-transformers failed to load the model.
        EmbeddingDimensionMismatchError
            Model's actual dim ≠ configured ``embedding.dimension``.
        """
        if self._loaded is not None:
            return self._loaded

        # Lazy import: sentence-transformers + torch + transformers
        # add up to ~1 s of import time and ~250 MB of RAM. We don't
        # want that cost when constructing the embedder for type
        # checks or tests that never call ``embed()``.
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise EmbeddingModelNotFoundError(
                "sentence-transformers is not installed; "
                "run `pip install -r requirements.txt`",
                model_name=self._model_name,
            ) from exc

        # Make sure the cache dir exists. sentence-transformers
        # writes here on first download.
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            st_model = SentenceTransformer(
                self._model_name,
                device=self._device,
                cache_folder=str(self._cache_dir),
            )
        except Exception as exc:  # sentence-transformers raises many types
            raise EmbeddingModelNotFoundError(
                f"could not load embedding model {self._model_name!r}: {exc}",
                model_name=self._model_name,
            ) from exc

        # ``get_sentence_embedding_dimension()`` returns the model's
        # actual output size — authoritative regardless of what the
        # config says.
        actual_dim = int(st_model.get_sentence_embedding_dimension())

        # If the config specified a dimension, verify it matches.
        # A mismatch is almost always a forgotten config update
        # (e.g. switching models without updating ``embedding.dimension``).
        if (
            self._expected_dimension is not None
            and self._expected_dimension != actual_dim
        ):
            raise EmbeddingDimensionMismatchError(
                f"embedding model {self._model_name!r} outputs {actual_dim}-dim "
                f"vectors, but config.embedding.dimension={self._expected_dimension}. "
                f"Update config.yaml to match the model (or remove the dimension "
                f"key to disable this check).",
                model_name=self._model_name,
            )

        self._loaded = _LoadedModel(st_model=st_model, actual_dimension=actual_dim)
        return self._loaded
