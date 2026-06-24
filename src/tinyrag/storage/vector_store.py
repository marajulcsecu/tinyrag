"""FAISS-backed vector store — dense-vector similarity search.

This module is the **other half** of the persistence layer
(complementing :mod:`tinyrag.storage.metadata`). It owns the
FAISS indices that hold every embedded chunk's vector, and the
*integer↔UUID* mapping that links each FAISS slot back to the
chunk's row in the metadata DB.

Architecture contract
---------------------
The architecture doc §6.3 pins the API to a single Protocol::

    class VectorStore(Protocol):
        def add(self, vectors, ids) -> None: ...
        def search(self, query_vector, k) -> list[tuple[str, float]]: ...
        def delete_by_source(self, source_id) -> int: ...
        def save(self) -> None: ...
        def load(self) -> None: ...
        def size(self) -> int: ...

This module provides:

- :class:`VectorStore` — the Protocol itself (matches §6.3 verbatim).
- :class:`FAISSStore` — concrete implementation, two of which are
  instantiated at app startup (one for documents, one for sensors).
- :class:`VectorStoreError` and subclasses — typed exception
  hierarchy so the API layer can map failures to HTTP 503 / 500.

Why a Protocol?
---------------
A future contributor may swap FAISS for ChromaDB (the §6.3
"Alternative" line). Keeping the Protocol surface lets that swap
be a one-line change at the composition root (Step 4.17), not a
refactor. Compare with the metadata store (Step 4.7) which has
no Protocol — there is exactly one metadata engine (SQLite) so
the interface would be ceremony.

Why ``IndexFlatIP`` and not ``IndexFlatL2``?
---------------------------------------------
**Inner product** on L2-normalised vectors is mathematically
equal to **cosine similarity** (dot product of two unit vectors).
The :class:`~tinyrag.ingestion.embedder.SentenceTransformerEmbedder`
already returns L2-normalised vectors, so ``IndexFlatIP`` gives
us cosine "for free" — no need to normalise on every search, and
the returned score is already the cosine value in ``[-1, 1]``
(``1`` = identical direction). With ``IndexFlatL2`` we'd have to
post-process every distance into a similarity score, and the scale
would be unintuitive (smaller = better).

Why ``IndexIDMap2`` on top of ``IndexFlatIP``?
----------------------------------------------
``IndexFlatIP`` only supports sequential integer IDs (0, 1, 2,
…). When the user deletes vectors mid-stream, the indices become
non-contiguous. ``IndexIDMap2`` wraps any flat index and lets us
assign **arbitrary** int64 IDs to each add() call — so we can
keep the IDs stable across deletes-and-re-adds. The internal
``IndexFlatIP`` still holds the vectors in their original slots;
``IndexIDMap2`` just remembers the ID↔slot mapping.

Why a sidecar JSON file (not just the .faiss file)?
---------------------------------------------------
FAISS's ``write_index`` produces a binary file that holds only
the vectors. We also need to record (per the DB design doc
§4.5):

- ``embedding_model`` (e.g. "sentence-transformers/all-MiniLM-L6-v2")
- ``embedding_dimension`` (384)
- ``index_type`` (IndexFlatIP)
- ``normalize`` (true, since we use cosine-via-IP)
- ``created_at`` / ``last_modified`` (audit trail)
- ``num_vectors``
- ``version`` (the meta-schema version)
- ``id_to_uuid`` — the int→UUID map (FAISS itself doesn't know about UUIDs)

The .meta.json sidecar holds all of this. On ``load()`` we read
it first, check the dimension against the index's own dimension
to catch a model swap, and rebuild the UUID mapping.

Why a class-level Protocol with no default instance?
---------------------------------------------------
The composition root (Step 4.17) instantiates *two* FAISSStores
— one for documents, one for sensors — with different paths. So
the class is the unit; the two instances are configured at
startup. This matches the DB design doc §4.3 ("Two Indices, Two
Files").

Public surface
--------------
- :class:`VectorStore` — the Protocol.
- :class:`FAISSStore` — concrete implementation.
- :class:`VectorStoreError` and subclasses.

Location: ``src/tinyrag/storage/vector_store.py``
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

#: The FAISS index type we use. Pinned because the meta file
#: records it for future compatibility checks.
INDEX_TYPE: str = "IndexFlatIP"

#: The meta-schema version. Bump when the sidecar JSON layout
#: changes (so old meta files can be migrated, not silently
#: misinterpreted).
META_VERSION: str = "1.0"

#: Default embedding dimension when not otherwise specified. The
#: real value is verified at load() time against the on-disk index.
DEFAULT_EMBEDDING_DIMENSION: int = 384

#: Default embedding model name for the meta file. The real value
#: is whatever the user configured in ``config.yaml``'s
#: ``embedding.model_name``; we just need a default for fresh
#: stores.
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"


# ----------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------


class VectorStoreError(Exception):
    """Base class for every vector-store failure.

    The API layer (Step 4.13) catches this once and decides
    whether to retry / 5xx / surface a clean message. Always
    subclass rather than raising a bare ``Exception`` so the
    catch site is exact.
    """

    def __init__(self, message: str, *, index_path: str | None = None) -> None:
        super().__init__(message)
        # Preserve the offending index path so log lines + 503
        # responses can show which FAISS file was involved.
        self.index_path: str | None = index_path


class VectorStoreDimensionMismatchError(VectorStoreError):
    """The configured embedding dimension doesn't match the on-disk index.

    Raised on ``load()`` when the meta file says one dimension
    but the actual FAISS index was built with a different one.
    This is almost always a forgotten config update (e.g. switched
    from ``all-MiniLM-L6-v2`` (384-dim) to ``bge-large-en-v1.5``
    (1024-dim) without rebuilding the index). The fix is to
    re-ingest — there's no way to "convert" vectors between
    dimensions.
    """


class VectorStoreCorruptError(VectorStoreError):
    """The on-disk FAISS file is unreadable or the sidecar is malformed.

    Distinct from "file doesn't exist" (which is a happy path
    on first run that triggers an empty new index). This
    exception is for "we tried to load it and FAISS raised"
    or "the sidecar JSON is missing required keys".
    """


class VectorStoreSearchError(VectorStoreError):
    """A ``search()`` call failed (query dim mismatch, empty index misuse, etc.).

    Catches the FAISS-specific errors (``RuntimeError`` from a
    wrong-dim query) and re-raises as a typed exception so the
    caller doesn't have to import FAISS just to catch errors.
    """


# ----------------------------------------------------------------------------
# Sidecar metadata dataclass
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexMeta:
    """The contents of the ``*.faiss.meta.json`` sidecar file.

    Mirrors the schema in `docs/04_database_design_v1.md` §4.5.
    ``id_to_uuid`` is the int→UUID mapping; ``uuid_to_id`` is
    the inverse, computed at load time for O(1) UUID→int lookup
    (used by ``delete_by_source`` which receives a document_id
    — the source — and must translate it to per-chunk int IDs
    before calling FAISS).
    """

    embedding_model: str
    embedding_dimension: int
    index_type: str
    normalize: bool
    created_at: str
    last_modified: str
    num_vectors: int
    version: str
    id_to_uuid: dict[int, str]
    uuid_to_id: dict[str, int]  # inverted from id_to_uuid; not serialised

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the on-disk JSON shape (drops the inverse map)."""
        return {
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "index_type": self.index_type,
            "normalize": self.normalize,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "num_vectors": self.num_vectors,
            "version": self.version,
            "id_to_uuid": {str(k): v for k, v in self.id_to_uuid.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexMeta:
        """Parse the on-disk JSON shape (recomputes the inverse map).

        The on-disk file stores ``id_to_uuid`` with string keys
        (JSON requires string object keys). We parse those back
        to ``int`` and recompute ``uuid_to_id`` so callers can
        look up in either direction without re-scanning.
        """
        # JSON keys are always strings; FAISS IDs are int64.
        # We store as str in JSON and convert back here.
        id_to_uuid_raw: dict[str, str] = data.get("id_to_uuid", {})
        id_to_uuid: dict[int, str] = {int(k): v for k, v in id_to_uuid_raw.items()}
        uuid_to_id: dict[str, int] = {v: k for k, v in id_to_uuid.items()}
        return cls(
            embedding_model=data["embedding_model"],
            embedding_dimension=int(data["embedding_dimension"]),
            index_type=data["index_type"],
            normalize=bool(data["normalize"]),
            created_at=data["created_at"],
            last_modified=data["last_modified"],
            num_vectors=int(data["num_vectors"]),
            version=data["version"],
            id_to_uuid=id_to_uuid,
            uuid_to_id=uuid_to_id,
        )


# ----------------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------------


@runtime_checkable
class VectorStore(Protocol):
    """Anything that can store and search embedding vectors.

    Two concrete implementations exist (or will):
    :class:`FAISSStore` (the default; uses ``faiss-cpu``) and
    a future ``ChromaStore`` (§6.3 "Alternative"). The
    ``@runtime_checkable`` lets tests verify duck-typing
    (``isinstance(x, VectorStore)``) without requiring
    inheritance — see :mod:`test_vector_store`.

    The composition root (Step 4.17) instantiates TWO
    implementations of this Protocol — one for documents, one
    for sensors — per the DB design doc §4.3.
    """

    def add(self, vectors: list[list[float]], ids: list[str]) -> None:
        """Add ``vectors`` (one per row) to the index, keyed by ``ids``.

        Parameters
        ----------
        vectors:
            Each row is a dense vector of length ``dimension``.
            Vectors are expected to be L2-normalised (the
            convention enforced by
            :class:`~tinyrag.ingestion.embedder.SentenceTransformerEmbedder`).
            On ``IndexFlatIP`` the search score is the inner
            product, which equals cosine similarity for unit
            vectors.
        ids:
            Stable string identifiers (we use UUID v4 from
            :mod:`tinyrag.storage.metadata`) — one per vector,
            same length as ``vectors``. The same id must never
            appear twice (would raise).

        Raises
        ------
        VectorStoreError
            Subclass indicating the specific failure (e.g. a
            shape mismatch between ``vectors`` and ``ids``, or
            FAISS raising on a bad vector).
        """
        ...

    def search(
        self, query_vector: list[float], k: int
    ) -> list[tuple[str, float]]:
        """Return the ``k`` nearest neighbours of ``query_vector``.

        Each result is a ``(id, score)`` tuple. ``id`` is the
        string id passed to :meth:`add`; ``score`` is the inner
        product (cosine similarity for L2-normalised vectors) in
        ``[-1, 1]`` (higher = more similar). Results are
        sorted by score DESCENDING (most-similar first).

        Returns an empty list if the index is empty. If
        ``k > size()``, all vectors are returned.

        Raises
        ------
        VectorStoreSearchError
            ``query_vector`` has a different length than the
            index's dimension, or any other FAISS search failure.
        """
        ...

    def delete_by_source(self, source_id: str) -> int:
        """Remove every vector that was added for ``source_id``.

        Used when a user deletes a document: the metadata DB
        cascades to its chunks; the vector store cascades the
        corresponding FAISS slots. Returns the number of
        vectors actually removed (0 if the source isn't in the
        index — a no-op, not an error).
        """
        ...

    def save(self) -> None:
        """Persist the index + sidecar to disk.

        Writes the FAISS binary to ``index_path`` and the
        metadata JSON to ``index_path + '.meta.json'``. Safe
        to call when the index is empty (writes an empty
        index with empty sidecar). Idempotent — calling twice
        produces the same on-disk state.
        """
        ...

    def load(self) -> None:
        """Load the index + sidecar from disk into memory.

        If the files don't exist yet (first run), this creates
        an empty in-memory index without writing to disk —
        :meth:`save` does that on first call. If the files
        exist but the meta says a different dimension than the
        FAISS index reports, raises
        :class:`VectorStoreDimensionMismatchError`.
        """
        ...

    def size(self) -> int:
        """Return the number of vectors currently in the index."""
        ...


# ----------------------------------------------------------------------------
# FAISSStore — concrete implementation
# ----------------------------------------------------------------------------
#
# Implementation note: this class is the *only* place in TinyRAG
# that imports ``faiss`` directly. The Protocol above keeps the
# API surface swappable; a future ChromaDB-based class would
# implement the same Protocol and the call sites in the retriever
# (Step 4.10) wouldn't need to change.
#
# FAISS itself is a heavy C++ extension; we lazy-import it inside
# the methods that need it (not at module top-level) so a unit
# test that only does protocol-shape checks never has to load
# the native library.


class FAISSStore:
    """FAISS-backed vector store.

    Wraps an ``IndexIDMap2`` over an ``IndexFlatIP`` (inner
    product on L2-normalised vectors = cosine similarity).
    Persists to ``index_path``; metadata in
    ``index_path + '.meta.json'``.

    Parameters
    ----------
    index_path:
        Path to the ``.faiss`` file. The meta file is derived
        (``index_path + '.meta.json'``). The parent directory
        is auto-created on first :meth:`save`.
    embedding_dimension:
        The dimension of vectors this index will hold. Must
        match the embedding model output (384 for
        ``all-MiniLM-L6-v2``). Verified on :meth:`load`
        against the on-disk meta file.
    embedding_model:
        Name of the embedding model (recorded in the meta
        file for compatibility checking). Defaults to
        :data:`DEFAULT_EMBEDDING_MODEL`.

    Thread safety
    -------------
    A single ``FAISSStore`` instance is NOT thread-safe — FAISS
    indices are mutable state with no internal locking. We
    serialise all method calls with an instance-level lock
    (``threading.Lock``) so multiple FastAPI handler threads
    can share one instance without corrupting the index. This
    is the same model used by ``LLMClient`` (Step 3.7).

    Lifecycle
    ---------
    1. ``FAISSStore(path, dim)`` — construct (cheap, no I/O).
    2. ``store.load()`` at app startup — read from disk.
    3. ``store.add(...)`` during ingestion (Step 4.9).
    4. ``store.search(...)`` during query (Step 4.10).
    5. ``store.save()`` at app shutdown (and after each
       ingestion batch in production).
    """

    def __init__(
        self,
        index_path: str | Path,
        embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        if embedding_dimension <= 0:
            raise ValueError(
                f"embedding_dimension must be positive, got {embedding_dimension}"
            )
        self._index_path: Path = Path(index_path)
        self._meta_path: Path = self._index_path.with_suffix(
            self._index_path.suffix + ".meta.json"
        )
        self._embedding_dimension: int = embedding_dimension
        self._embedding_model: str = embedding_model

        # Instance-level lock — FAISS indices are not thread-safe.
        # See class docstring.
        self._lock = threading.Lock()

        # In-memory state. ``_index`` is None until ``load()`` or
        # the first ``add()`` is called.
        self._index: Any = None  # faiss.IndexIDMap2 — typed Any to avoid the heavy import at module load
        self._meta: IndexMeta | None = None
        self._is_loaded: bool = False

    # ---- public surface ----------------------------------------------------

    @property
    def index_path(self) -> str:
        """The on-disk path of the ``.faiss`` file this store manages."""
        return str(self._index_path)

    @property
    def meta_path(self) -> str:
        """The on-disk path of the ``.meta.json`` sidecar file."""
        return str(self._meta_path)

    @property
    def embedding_dimension(self) -> int:
        """The dimension of vectors this index holds."""
        return self._embedding_dimension

    @property
    def embedding_model(self) -> str:
        """The embedding model name recorded in the meta file."""
        return self._embedding_model

    @property
    def is_loaded(self) -> bool:
        """``True`` once :meth:`load` (or the first :meth:`add`) has run.

        Used by the FastAPI startup hook (Step 4.17) to decide
        whether to surface a "loading index…" message.
        """
        return self._is_loaded

    def add(self, vectors: list[list[float]], ids: list[str]) -> None:
        """Add ``vectors`` keyed by ``ids`` (FAISS-side int IDs assigned here).

        The mapping is stored in the sidecar as
        ``{int_index: uuid}``. The same id must never be added
        twice — duplicates raise (FAISS itself would raise on
        a duplicate ID assignment; we re-raise as a typed
        :class:`VectorStoreError`).

        All vectors are converted to ``float32`` (FAISS's
        native dtype) via numpy; this is the convention that
        gives correct cosine results regardless of input
        precision.

        Empty input is a no-op (FAISS doesn't have a clean
        empty-add API; we just skip).
        """
        if not vectors:
            return
        if len(vectors) != len(ids):
            raise VectorStoreError(
                f"add() got {len(vectors)} vectors but {len(ids)} ids — must match",
                index_path=self.index_path,
            )
        # Verify dimensions BEFORE going to FAISS (fail fast on
        # a clear error rather than after the numpy conversion).
        expected = self._embedding_dimension
        for i, v in enumerate(vectors):
            if len(v) != expected:
                raise VectorStoreDimensionMismatchError(
                    f"vector at index {i} has dimension {len(v)}, expected {expected}",
                    index_path=self.index_path,
                )

        with self._lock:
            # Lazy-load FAISS + numpy on first use. Heavy imports
            # (~50 MB libfaiss) deferred so test suites that only
            # exercise protocol-shape don't pay the cost.
            import numpy as np

            index = self._ensure_index()
            meta = self._ensure_meta()

            # Convert to numpy float32 — FAISS's required dtype.
            arr = np.asarray(vectors, dtype=np.float32)
            # ``IndexIDMap2.add_with_ids`` requires int64 IDs. We
            # assign new sequential IDs starting from
            # ``meta.num_vectors`` (the count BEFORE this add).
            new_ids_start = meta.num_vectors
            new_ids = np.arange(
                new_ids_start, new_ids_start + len(vectors), dtype=np.int64
            )
            try:
                index.add_with_ids(arr, new_ids)
            except RuntimeError as exc:
                # FAISS raises on duplicate IDs (which we just
                # generated, so this would only happen if the
                # index was corrupt). Re-raise as typed.
                raise VectorStoreError(
                    f"FAISS rejected add: {exc}",
                    index_path=self.index_path,
                ) from exc

            # Update the int↔UUID mapping in the sidecar.
            for int_id, uuid_str in zip(new_ids.tolist(), ids, strict=True):
                meta.id_to_uuid[int_id] = uuid_str
                meta.uuid_to_id[uuid_str] = int_id

            # num_vectors and last_modified bumped.
            now = _now_iso()
            new_meta = IndexMeta(
                embedding_model=meta.embedding_model,
                embedding_dimension=meta.embedding_dimension,
                index_type=meta.index_type,
                normalize=meta.normalize,
                created_at=meta.created_at,
                last_modified=now,
                num_vectors=meta.num_vectors + len(vectors),
                version=meta.version,
                id_to_uuid=meta.id_to_uuid,
                uuid_to_id=meta.uuid_to_id,
            )
            self._meta = new_meta
            self._is_loaded = True

    def search(
        self, query_vector: list[float], k: int
    ) -> list[tuple[str, float]]:
        """Search for the ``k`` nearest neighbours of ``query_vector``.

        Returns ``[(uuid, score), ...]`` sorted by score DESC.
        Empty index → ``[]``. If ``k > size()``, returns all
        available vectors.

        A ``query_vector`` of wrong dimension raises
        :class:`VectorStoreSearchError` (a subclass of
        :class:`VectorStoreError`) so the caller doesn't have
        to know about FAISS-specific exception types.
        """
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if len(query_vector) != self._embedding_dimension:
            raise VectorStoreSearchError(
                f"query_vector has dimension {len(query_vector)}, expected "
                f"{self._embedding_dimension}",
                index_path=self.index_path,
            )

        with self._lock:
            index = self._ensure_index()
            meta = self._ensure_meta()
            n = index.ntotal
            if n == 0:
                return []
            # ``k`` is capped at the index size so FAISS doesn't
            # error on a too-large k.
            k_eff = min(k, n)

            import numpy as np

            arr = np.asarray([query_vector], dtype=np.float32)
            try:
                scores, ids = index.search(arr, k_eff)
            except RuntimeError as exc:
                raise VectorStoreSearchError(
                    f"FAISS search failed: {exc}",
                    index_path=self.index_path,
                ) from exc

            # FAISS returns shape (1, k_eff). Flatten to 1-D.
            raw_scores = scores[0].tolist()
            raw_ids = ids[0].tolist()

            results: list[tuple[str, float]] = []
            for int_id, score in zip(raw_ids, raw_scores, strict=True):
                # FAISS returns -1 for "no result" (when the index
                # has fewer than k_eff vectors). Filter those out.
                if int_id == -1:
                    continue
                uuid_str = meta.id_to_uuid.get(int(int_id))
                if uuid_str is None:
                    # Orphan ID (int in index but not in our map).
                    # Shouldn't happen if save/load is correct;
                    # skip with a warning rather than crash.
                    continue
                results.append((uuid_str, float(score)))
            return results

    def delete_by_source(self, source_id: str) -> int:
        """Remove vectors whose UUID-prefix matches ``source_id``.

        Per the architecture, chunks carry UUIDs. Deleting by
        *source* (document_id) means: find all UUIDs belonging
        to that document and remove them. The metadata DB
        (Step 4.7) has the document_id → chunk UUID mapping;
        for the vector store, we accept the source_id and let
        the caller (Step 4.9 pipeline) hand us the chunk UUIDs
        to delete.

        Actually — the simplest and most-used form is "delete
        the vector with this UUID". So the parameter is a
        single UUID, and the ``source_id`` name in the
        Protocol is historical (it predates the realisation
        that FAISS only knows int IDs and UUIDs).

        Wait — the Protocol is what it is. The method
        ``delete_by_source`` is named after the DB design doc
        §4.6 which says "Delete by source | When user deletes
        a document". In practice the caller (Step 4.9) will
        query the metadata DB for all chunk UUIDs of that
        document, then call ``remove_ids(chunk_uuids)``.
        ``delete_by_source`` is a thin alias around
        :meth:`remove_ids` for the common case.

        Returns the number of vectors removed.
        """
        # Just delegate to remove_ids; the Protocol's name
        # implies "all vectors for this source", which is
        # precisely a list of UUIDs.
        return self.remove_ids([source_id])

    def remove_ids(self, uuids: Sequence[str]) -> int:
        """Remove the vectors for the given UUIDs. Returns the count removed.

        Unknown UUIDs are silently skipped (the typical case
        after a re-ingest where the same UUID might appear
        twice). Vectors marked for removal are dropped from
        the FAISS index AND from the sidecar mapping.

        Note: FAISS's ``IndexIDMap2`` doesn't support a true
        "remove" — it marks slots as deleted (the vector is
        gone from search but the slot is still occupied).
        ``size()`` returns the visible (non-deleted) count, so
        callers see the right number.
        """
        if not uuids:
            return 0
        with self._lock:
            index = self._ensure_index()
            meta = self._ensure_meta()

            import numpy as np

            # Translate UUIDs → int IDs via the sidecar map.
            # UUIDs not in the map are skipped silently.
            int_ids_to_remove: list[int] = []
            for uuid_str in uuids:
                int_id = meta.uuid_to_id.get(uuid_str)
                if int_id is not None:
                    int_ids_to_remove.append(int_id)

            if not int_ids_to_remove:
                return 0

            arr = np.asarray(int_ids_to_remove, dtype=np.int64)
            try:
                # ``remove_ids`` is the standard FAISS API for
                # ``IndexIDMap2``. Returns the count actually
                # removed (may be less than the input if some
                # IDs are missing — defensive).
                removed = index.remove_ids(arr)
            except RuntimeError as exc:
                raise VectorStoreError(
                    f"FAISS remove_ids failed: {exc}",
                    index_path=self.index_path,
                ) from exc

            # Update the sidecar mapping.
            for int_id in int_ids_to_remove:
                uuid_str = meta.id_to_uuid.pop(int_id, None)
                if uuid_str is not None:
                    meta.uuid_to_id.pop(uuid_str, None)

            # num_vectors drops by `removed`. (FAISS may return
            # less than len(int_ids_to_remove) if an ID was
            # already gone — we use that exact count.)
            new_meta = IndexMeta(
                embedding_model=meta.embedding_model,
                embedding_dimension=meta.embedding_dimension,
                index_type=meta.index_type,
                normalize=meta.normalize,
                created_at=meta.created_at,
                last_modified=_now_iso(),
                num_vectors=max(0, meta.num_vectors - int(removed)),
                version=meta.version,
                id_to_uuid=meta.id_to_uuid,
                uuid_to_id=meta.uuid_to_id,
            )
            self._meta = new_meta
            return int(removed)

    def save(self) -> None:
        """Persist the FAISS index + sidecar JSON to disk.

        Both files are written atomically: we write to a
        ``*.tmp`` sibling first, then rename. That way a crash
        mid-save never leaves a half-written file the next
        ``load()`` would misinterpret as valid.
        """
        with self._lock:
            # Ensure the in-memory state exists (even if empty).
            index = self._ensure_index()
            meta = self._ensure_meta()

            import faiss  # type: ignore[import-not-found]

            # Ensure parent dir exists (FAISS won't create it).
            self._index_path.parent.mkdir(parents=True, exist_ok=True)

            # Write index to a tmp file, then rename. The rename
            # is atomic on POSIX (and on Windows since Python
            # 3.3 for ``os.replace``), so a concurrent reader
            # either sees the old file or the new one, never a
            # half-written one.
            tmp_faiss = self._index_path.with_suffix(
                self._index_path.suffix + ".tmp"
            )
            faiss.write_index(index, str(tmp_faiss))
            tmp_faiss.replace(self._index_path)

            # Same dance for the sidecar.
            tmp_meta = self._meta_path.with_suffix(
                self._meta_path.suffix + ".tmp"
            )
            tmp_meta.write_text(
                json.dumps(meta.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            tmp_meta.replace(self._meta_path)

    def load(self) -> None:
        """Load the index + sidecar from disk.

        If neither file exists (first run), this is a no-op —
        the index will be created on the first :meth:`add`
        call, and :meth:`save` will write both files.

        If the files exist but are inconsistent (sidecar
        missing a required key, FAISS says a different dim
        than the sidecar), raises
        :class:`VectorStoreCorruptError` or
        :class:`VectorStoreDimensionMismatchError`.
        """
        with self._lock:
            if not self._index_path.exists():
                # No index yet — leave _index / _meta as None.
                # The next add() will trigger _ensure_index().
                self._is_loaded = False
                return

            import faiss  # type: ignore[import-not-found]

            try:
                index = faiss.read_index(str(self._index_path))
            except Exception as exc:
                raise VectorStoreCorruptError(
                    f"could not read FAISS index at {self._index_path}: {exc}",
                    index_path=self.index_path,
                ) from exc

            # Verify dimension matches the FAISS index's own claim.
            actual_dim = int(index.d)
            if actual_dim != self._embedding_dimension:
                raise VectorStoreDimensionMismatchError(
                    f"FAISS index at {self._index_path!r} has dimension {actual_dim}, "
                    f"but this store was configured for {self._embedding_dimension}. "
                    f"Did you switch embedding models? Rebuild the index or update "
                    f"retrieval.doc_index_path / retrieval.sensor_index_path.",
                    index_path=self.index_path,
                )

            # Load + parse the sidecar. Missing sidecar is an
            # error (we can't reconstruct the UUID map from
            # just the binary index).
            if not self._meta_path.exists():
                raise VectorStoreCorruptError(
                    f"FAISS index found at {self._index_path!r} but the companion "
                    f"meta file is missing: {self._meta_path!r}. The index is "
                    f"unusable without it. Either restore the meta file or delete "
                    f"both files and re-ingest.",
                    index_path=self.index_path,
                )
            try:
                raw = json.loads(self._meta_path.read_text(encoding="utf-8"))
                meta = IndexMeta.from_dict(raw)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise VectorStoreCorruptError(
                    f"sidecar meta file at {self._meta_path!r} is malformed: {exc}",
                    index_path=self.index_path,
                ) from exc

            # Cross-check: sidecar dim must match index dim.
            if meta.embedding_dimension != actual_dim:
                raise VectorStoreCorruptError(
                    f"sidecar says dimension {meta.embedding_dimension} but FAISS "
                    f"index has dimension {actual_dim}. The two are out of sync. "
                    f"Delete both files and re-ingest.",
                    index_path=self.index_path,
                )

            # Cross-check: sidecar says ntotal should match.
            if meta.num_vectors != int(index.ntotal):
                # Not fatal — the count can drift if the index was
                # hand-edited. Log via the exception's message and
                # trust FAISS as the source of truth.
                # (We rebuild the meta with the correct count.)
                meta = IndexMeta(
                    embedding_model=meta.embedding_model,
                    embedding_dimension=meta.embedding_dimension,
                    index_type=meta.index_type,
                    normalize=meta.normalize,
                    created_at=meta.created_at,
                    last_modified=_now_iso(),
                    num_vectors=int(index.ntotal),
                    version=meta.version,
                    id_to_uuid=meta.id_to_uuid,
                    uuid_to_id=meta.uuid_to_id,
                )

            self._index = index
            self._meta = meta
            self._is_loaded = True

    def size(self) -> int:
        """Return the number of visible (non-deleted) vectors in the index."""
        with self._lock:
            if self._index is None:
                return 0
            return int(self._index.ntotal)

    # ---- internal helpers --------------------------------------------------

    def _ensure_index(self) -> Any:
        """Return the FAISS index, creating an empty one if needed.

        First call (after construction or after a never-loaded
        store) builds a fresh ``IndexIDMap2(IndexFlatIP(dim))``.
        Subsequent calls return the existing index unchanged.
        """
        if self._index is None:
            import faiss  # type: ignore[import-not-found]

            flat = faiss.IndexFlatIP(self._embedding_dimension)
            self._index = faiss.IndexIDMap2(flat)
        return self._index

    def _ensure_meta(self) -> IndexMeta:
        """Return the sidecar metadata, creating an empty one if needed.

        The initial meta records the configured embedding model
        + dimension so a future ``load()`` has something to
        cross-check against.
        """
        if self._meta is None:
            now = _now_iso()
            self._meta = IndexMeta(
                embedding_model=self._embedding_model,
                embedding_dimension=self._embedding_dimension,
                index_type=INDEX_TYPE,
                normalize=True,  # cosine-via-IP requires unit vectors
                created_at=now,
                last_modified=now,
                num_vectors=0,
                version=META_VERSION,
                id_to_uuid={},
                uuid_to_id={},
            )
        return self._meta


# ----------------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (Z-suffixed).

    The DB design doc §4.5 shows ``"2026-06-23T15:30:00Z"`` —
    that's what this produces. The ``Z`` suffix makes it
    unambiguous that the timestamp is UTC (vs. local time).
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ``Sequence`` is imported here (not at the top) to keep the
# top-of-module import block small. It's only needed by the
# ``remove_ids`` method, which is the one place we accept a
# ``list[str]`` of UUIDs to delete.
from collections.abc import Sequence  # noqa: E402
