"""Tests for src/tinyrag/storage/vector_store.py — FAISS-backed vector store.

Test layout
-----------
- TestPublicSurface           — every public name is exported from the
  subpackage (``FAISSStore``, ``VectorStore``, ``IndexMeta``, the
  exception hierarchy, 4 module-level constants).
- TestProtocolIsRuntime       — ``@runtime_checkable`` works:
  ``FAISSStore`` satisfies the protocol via duck-typing; a fake class
  with all six methods also does; a class missing ``save`` doesn't.
- TestIndexMeta               — the sidecar dataclass round-trips
  through ``to_dict`` / ``from_dict`` (str keys → int keys).
- TestFAISSStoreConstruction  — ``__init__`` validates dimension;
  derived paths (``meta_path``) are correct.
- TestAddSearchRoundTrip      — add vectors → search returns them in
  cosine-DESC order; L2-normalised input gives correct scores.
- TestSearchEdgeCases         — empty index returns ``[]``;
  ``k > size()`` is capped; ``k <= 0`` raises ``ValueError``.
- TestAddDimensionMismatch    — wrong-dim vector raises
  ``VectorStoreDimensionMismatchError``.
- TestSearchDimensionMismatch — wrong-dim query raises
  ``VectorStoreSearchError``.
- TestSaveLoadRoundTrip       — the roadmap's "after save+load, the
  index returns the same search results" check. Sidecar JSON is
  written; on reload, the index is queryable.
- TestDeleteBySource          — removes the right vectors; returns
  the count; unknown UUID is a no-op (returns 0).
- TestLoadEdgeCases           — missing index file is a no-op (empty
  in-memory); missing sidecar raises ``VectorStoreCorruptError``;
  dimension mismatch (configured vs on-disk) raises
  ``VectorStoreDimensionMismatchError``.
- TestSidecarMetaFile         — after ``save``, the meta file exists,
  is valid JSON, and has every required key.
- TestErrorHierarchy          — every error subclasses
  ``VectorStoreError``; carries ``index_path`` when constructed.
- TestThreadSafety            — concurrent adds don't corrupt the
  sidecar mapping; concurrent searches don't crash.

Why so many tests?
------------------
FAISS is the *one* piece of state that holds the entire knowledge
base. A bug here silently corrupts search results (returns wrong
neighbours, or none at all). Every operation that can go wrong
(empty input, dim mismatch, missing files, JSON corruption, dim
drift, concurrent threads) is covered.

Hermetic?
---------
100% hermetic. Every test uses ``tmp_path`` (a fresh per-test
directory pytest creates and reaps). No fixture files, no network,
no real project index.

Location: ``tests/test_vector_store.py``
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from tinyrag.storage import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    INDEX_TYPE,
    META_VERSION,
    FAISSStore,
    IndexMeta,
    VectorStore,
    VectorStoreCorruptError,
    VectorStoreDimensionMismatchError,
    VectorStoreError,
    VectorStoreSearchError,
)
from tinyrag.storage.vector_store import (
    FAISSStore as FAISSStoreDirect,  # for identity checks
)

# ----------------------------------------------------------------------------
# Constants / helpers
# ----------------------------------------------------------------------------

# A small dimension for fast tests. We don't use the real 384
# because L2-normalised geometry works the same at any dim — a
# 4-D unit vector is just easier to type out by hand.
TEST_DIM: int = 4


def _unit(v: list[float] | np.ndarray) -> list[float]:
    """Return the input as a Python list, L2-normalised.

    The :class:`FAISSStore` expects L2-normalised vectors (the
    ``IndexFlatIP`` = inner product = cosine convention). This
    helper makes every test vector explicitly unit-length so
    cosine == inner-product == the score FAISS returns.
    """
    arr = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError("cannot normalise a zero vector")
    return (arr / norm).tolist()


@pytest.fixture
def index_path(tmp_path: Path) -> Path:
    """Fresh per-test FAISS path inside pytest's tmp_path."""
    return tmp_path / "test.faiss"


@pytest.fixture
def store(index_path: Path) -> FAISSStore:
    """A :class:`FAISSStore` with the configured test dimension.

    Note: this fixture does NOT call ``load()`` — most tests want
    a fresh in-memory index. Tests that exercise load/save build
    their own path.
    """
    return FAISSStore(index_path, embedding_dimension=TEST_DIM)


# Small fixed corpus used by most add/search tests. Three vectors
# in 4-D: v1 and v2 are orthogonal (cosine 0), v3 is close to v1.
CORPUS_VECTORS: list[list[float]] = [
    _unit([1.0, 0.0, 0.0, 0.0]),  # uuid-a
    _unit([0.0, 1.0, 0.0, 0.0]),  # uuid-b (orthogonal to v1)
    _unit([0.9, 0.1, 0.0, 0.0]),  # uuid-c (close to v1)
]
CORPUS_IDS: list[str] = ["uuid-a", "uuid-b", "uuid-c"]


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


class TestPublicSurface:
    """The expected symbols are exported and importable."""

    def test_subpackage_exports_faiss_store(self) -> None:
        from tinyrag.storage import FAISSStore as cls

        assert cls is FAISSStore
        assert cls is FAISSStoreDirect  # same class object

    def test_subpackage_exports_vector_store_protocol(self) -> None:
        from tinyrag.storage import VectorStore as proto

        assert proto is VectorStore
        # ``Protocol`` doesn't have a useful isinstance check at
        # the class object level (it does at the instance level —
        # see TestProtocolIsRuntime below).

    def test_subpackage_exports_index_meta(self) -> None:
        from tinyrag.storage import IndexMeta as cls

        assert cls is IndexMeta

    def test_subpackage_exports_vector_store_error(self) -> None:
        from tinyrag.storage import VectorStoreError as cls

        assert cls is VectorStoreError

    def test_subpackage_exports_dimension_mismatch_error(self) -> None:
        from tinyrag.storage import VectorStoreDimensionMismatchError as cls

        assert cls is VectorStoreDimensionMismatchError
        assert issubclass(cls, VectorStoreError)

    def test_subpackage_exports_corrupt_error(self) -> None:
        from tinyrag.storage import VectorStoreCorruptError as cls

        assert cls is VectorStoreCorruptError
        assert issubclass(cls, VectorStoreError)

    def test_subpackage_exports_search_error(self) -> None:
        from tinyrag.storage import VectorStoreSearchError as cls

        assert cls is VectorStoreSearchError
        assert issubclass(cls, VectorStoreError)

    def test_subpackage_exports_index_type_constant(self) -> None:
        from tinyrag.storage import INDEX_TYPE as v

        assert v == INDEX_TYPE
        assert isinstance(v, str)
        assert v == "IndexFlatIP"

    def test_subpackage_exports_meta_version_constant(self) -> None:
        from tinyrag.storage import META_VERSION as v

        assert v == META_VERSION
        assert isinstance(v, str)

    def test_subpackage_exports_default_embedding_dimension(self) -> None:
        from tinyrag.storage import DEFAULT_EMBEDDING_DIMENSION as v

        assert v == DEFAULT_EMBEDDING_DIMENSION
        assert isinstance(v, int)
        assert v == 384

    def test_subpackage_exports_default_embedding_model(self) -> None:
        from tinyrag.storage import DEFAULT_EMBEDDING_MODEL as v

        assert v == DEFAULT_EMBEDDING_MODEL
        assert isinstance(v, str)
        assert "MiniLM" in v


class TestProtocolIsRuntime:
    """The VectorStore protocol is ``@runtime_checkable``."""

    def test_faiss_store_satisfies_protocol(self, store: FAISSStore) -> None:
        # An instance of FAISSStore — which doesn't explicitly
        # inherit from the Protocol — should still be detected
        # as a VectorStore by ``isinstance`` because of the
        # ``@runtime_checkable`` decorator.
        assert isinstance(store, VectorStore)

    def test_arbitrary_duck_type_satisfies_protocol(self) -> None:
        class _DuckStore:
            """A class that has the six methods but inherits from nothing."""

            def add(self, vectors, ids):
                pass

            def search(self, query_vector, k):
                return []

            def delete_by_source(self, source_id):
                return 0

            def save(self):
                pass

            def load(self):
                pass

            def size(self):
                return 0

        assert isinstance(_DuckStore(), VectorStore)

    def test_missing_method_does_not_satisfy_protocol(self) -> None:
        class _PartialStore:
            def add(self, vectors, ids):
                pass

            def search(self, query_vector, k):
                return []

            # ``delete_by_source``, ``save``, ``load``, ``size`` missing.

        assert not isinstance(_PartialStore(), VectorStore)

    def test_non_class_object_does_not_satisfy_protocol(self) -> None:
        assert not isinstance("not a store", VectorStore)
        assert not isinstance(42, VectorStore)
        assert not isinstance(None, VectorStore)


class TestIndexMeta:
    """The sidecar ``IndexMeta`` dataclass round-trips through dict."""

    def _make_meta(self) -> IndexMeta:
        return IndexMeta(
            embedding_model="test-model",
            embedding_dimension=4,
            index_type="IndexFlatIP",
            normalize=True,
            created_at="2026-01-01T00:00:00Z",
            last_modified="2026-01-01T00:00:00Z",
            num_vectors=2,
            version="1.0",
            id_to_uuid={0: "uuid-a", 1: "uuid-b"},
            uuid_to_id={"uuid-a": 0, "uuid-b": 1},
        )

    def test_to_dict_drops_inverse_map(self) -> None:
        meta = self._make_meta()
        d = meta.to_dict()
        # The on-disk JSON must NOT carry uuid_to_id — it can be
        # reconstructed from id_to_uuid. Including it would
        # bloat the file and risk inconsistency.
        assert "uuid_to_id" not in d

    def test_to_dict_keeps_id_to_uuid_with_str_keys(self) -> None:
        meta = self._make_meta()
        d = meta.to_dict()
        # JSON requires string keys — verify.
        assert all(isinstance(k, str) for k in d["id_to_uuid"])
        assert d["id_to_uuid"] == {"0": "uuid-a", "1": "uuid-b"}

    def test_from_dict_round_trip(self) -> None:
        meta = self._make_meta()
        d = meta.to_dict()
        parsed = IndexMeta.from_dict(d)
        # All scalar fields preserved.
        assert parsed.embedding_model == meta.embedding_model
        assert parsed.embedding_dimension == meta.embedding_dimension
        assert parsed.index_type == meta.index_type
        assert parsed.normalize == meta.normalize
        assert parsed.num_vectors == meta.num_vectors
        assert parsed.version == meta.version
        # id_to_uuid restored with int keys (FAISS's native type).
        assert parsed.id_to_uuid == {0: "uuid-a", 1: "uuid-b"}
        # uuid_to_id reconstructed automatically.
        assert parsed.uuid_to_id == {"uuid-a": 0, "uuid-b": 1}

    def test_from_dict_handles_missing_id_to_uuid(self) -> None:
        """A meta dict with no id_to_uuid → empty mappings."""
        d = {
            "embedding_model": "test-model",
            "embedding_dimension": 4,
            "index_type": "IndexFlatIP",
            "normalize": True,
            "created_at": "2026-01-01T00:00:00Z",
            "last_modified": "2026-01-01T00:00:00Z",
            "num_vectors": 0,
            "version": "1.0",
        }
        meta = IndexMeta.from_dict(d)
        assert meta.id_to_uuid == {}
        assert meta.uuid_to_id == {}

    def test_frozen(self) -> None:
        """IndexMeta is frozen — assigning to a field raises."""
        meta = self._make_meta()
        with pytest.raises((AttributeError, Exception)):
            meta.num_vectors = 99  # type: ignore[misc]


class TestFAISSStoreConstruction:
    """``__init__`` validates inputs and sets up paths."""

    def test_dimension_must_be_positive(self, index_path: Path) -> None:
        with pytest.raises(ValueError, match="embedding_dimension must be positive"):
            FAISSStore(index_path, embedding_dimension=0)

    def test_negative_dimension_rejected(self, index_path: Path) -> None:
        with pytest.raises(ValueError, match="embedding_dimension must be positive"):
            FAISSStore(index_path, embedding_dimension=-5)

    def test_index_path_property_returns_str(self, store: FAISSStore) -> None:
        assert isinstance(store.index_path, str)
        assert store.index_path.endswith(".faiss")

    def test_meta_path_is_derived_correctly(self, store: FAISSStore) -> None:
        # The sidecar is "<path>.faiss.meta.json" — a doubly-suffixed
        # filename chosen so a glob for "*.faiss" still matches the
        # binary (the meta file ends in ".json", not ".faiss").
        assert store.meta_path == store.index_path + ".meta.json"

    def test_embedding_dimension_property(self, store: FAISSStore) -> None:
        assert store.embedding_dimension == TEST_DIM

    def test_embedding_model_defaults(self, index_path: Path) -> None:
        s = FAISSStore(index_path, embedding_dimension=TEST_DIM)
        assert s.embedding_model == DEFAULT_EMBEDDING_MODEL

    def test_embedding_model_explicit(self, index_path: Path) -> None:
        s = FAISSStore(
            index_path, embedding_dimension=TEST_DIM, embedding_model="custom/model"
        )
        assert s.embedding_model == "custom/model"

    def test_is_loaded_false_before_load_or_add(self, store: FAISSStore) -> None:
        assert store.is_loaded is False

    def test_size_zero_before_add(self, store: FAISSStore) -> None:
        assert store.size() == 0


class TestAddSearchRoundTrip:
    """Add vectors, then search — verify cosine DESC + correct scores."""

    def test_add_increments_size(self, store: FAISSStore) -> None:
        assert store.size() == 0
        store.add([CORPUS_VECTORS[0]], [CORPUS_IDS[0]])
        assert store.size() == 1

    def test_add_lots_increments_size_correctly(self, store: FAISSStore) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        assert store.size() == 3

    def test_add_empty_is_noop(self, store: FAISSStore) -> None:
        store.add([], [])
        assert store.size() == 0

    def test_add_mismatched_lengths_raises(self, store: FAISSStore) -> None:
        with pytest.raises(VectorStoreError, match="must match"):
            store.add([CORPUS_VECTORS[0], CORPUS_VECTORS[1]], [CORPUS_IDS[0]])

    def test_search_returns_closest_first(self, store: FAISSStore) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        # Querying with v1 (uuid-a's vector) should rank uuid-a first
        # with cosine 1.0.
        results = store.search(CORPUS_VECTORS[0], k=3)
        assert len(results) == 3
        # Results are sorted by score DESC.
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
        # The top hit is the exact match.
        assert results[0][0] == "uuid-a"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_search_exact_match_cosine_is_one(self, store: FAISSStore) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        results = store.search(CORPUS_VECTORS[0], k=1)
        assert results == [("uuid-a", pytest.approx(1.0, abs=1e-5))]

    def test_search_orthogonal_cosine_is_zero(self, store: FAISSStore) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        # v1 · v2 (orthogonal unit vectors) == 0.
        results = store.search(CORPUS_VECTORS[0], k=3)
        # uuid-b (orthogonal) should have score ~0.
        uuid_b_score = next(s for uid, s in results if uid == "uuid-b")
        assert uuid_b_score == pytest.approx(0.0, abs=1e-5)

    def test_search_k_caps_at_size(self, store: FAISSStore) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        # k larger than size() — should return all available.
        results = store.search(CORPUS_VECTORS[0], k=999)
        assert len(results) == 3

    def test_search_returns_uuid_strings(self, store: FAISSStore) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        results = store.search(CORPUS_VECTORS[0], k=3)
        for uid, _ in results:
            assert isinstance(uid, str)

    def test_search_returns_float_scores(self, store: FAISSStore) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        results = store.search(CORPUS_VECTORS[0], k=3)
        for _, score in results:
            assert isinstance(score, float)

    def test_search_cosine_scores_in_range(self, store: FAISSStore) -> None:
        """For L2-normalised vectors, inner product ∈ [-1, 1]."""
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        results = store.search(CORPUS_VECTORS[0], k=3)
        for _, score in results:
            assert -1.0 - 1e-5 <= score <= 1.0 + 1e-5

    def test_add_marks_store_loaded(self, store: FAISSStore) -> None:
        assert store.is_loaded is False
        store.add([CORPUS_VECTORS[0]], [CORPUS_IDS[0]])
        assert store.is_loaded is True


class TestSearchEdgeCases:
    """Edge cases of ``search``."""

    def test_empty_index_returns_empty_list(self, store: FAISSStore) -> None:
        results = store.search(CORPUS_VECTORS[0], k=5)
        assert results == []

    def test_k_zero_raises(self, store: FAISSStore) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        with pytest.raises(ValueError, match="k must be > 0"):
            store.search(CORPUS_VECTORS[0], k=0)

    def test_k_negative_raises(self, store: FAISSStore) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        with pytest.raises(ValueError, match="k must be > 0"):
            store.search(CORPUS_VECTORS[0], k=-1)


class TestAddDimensionMismatch:
    """Wrong-dimension vectors raise ``VectorStoreDimensionMismatchError``."""

    def test_add_wrong_dim_raises(self, store: FAISSStore) -> None:
        wrong = _unit([1.0, 0.0])  # 2-D, store expects TEST_DIM=4
        with pytest.raises(VectorStoreDimensionMismatchError) as exc:
            store.add([wrong], ["uuid-wrong"])
        assert exc.value.index_path == store.index_path

    def test_one_bad_vector_in_batch_raises(self, store: FAISSStore) -> None:
        good = CORPUS_VECTORS[0]
        bad = _unit([1.0, 0.0])
        with pytest.raises(VectorStoreDimensionMismatchError):
            store.add([good, bad], ["uuid-good", "uuid-bad"])

    def test_wrong_dim_does_not_corrupt_index(self, store: FAISSStore) -> None:
        bad = _unit([1.0, 0.0])
        with pytest.raises(VectorStoreDimensionMismatchError):
            store.add([bad], ["uuid-bad"])
        # Size should still be 0 — failed add doesn't leave a half-add.
        assert store.size() == 0


class TestSearchDimensionMismatch:
    """Wrong-dimension query raises ``VectorStoreSearchError``."""

    def test_wrong_dim_query_raises(self, store: FAISSStore) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        wrong = _unit([1.0, 0.0])
        with pytest.raises(VectorStoreSearchError) as exc:
            store.search(wrong, k=3)
        assert exc.value.index_path == store.index_path

    def test_search_error_is_vector_store_error(self, store: FAISSStore) -> None:
        """The API layer catches ``VectorStoreError`` — search must too."""
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        wrong = _unit([1.0, 0.0])
        with pytest.raises(VectorStoreError):
            store.search(wrong, k=3)


class TestSaveLoadRoundTrip:
    """The roadmap's "after save+load, same search results" check."""

    def test_save_writes_both_files(self, store: FAISSStore, index_path: Path) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        store.save()
        assert index_path.exists()
        assert Path(store.meta_path).exists()

    def test_save_then_load_recreates_state(
        self, store: FAISSStore, index_path: Path
    ) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        store.save()

        # Build a fresh store, load, search — must match original.
        reloaded = FAISSStore(index_path, embedding_dimension=TEST_DIM)
        reloaded.load()
        assert reloaded.size() == 3

        original_results = store.search(CORPUS_VECTORS[0], k=3)
        reloaded_results = reloaded.search(CORPUS_VECTORS[0], k=3)
        # Compare UUIDs and scores (scores are float32, may differ by 1e-6).
        assert len(reloaded_results) == len(original_results)
        for (uid_a, score_a), (uid_b, score_b) in zip(
            original_results, reloaded_results, strict=True
        ):
            assert uid_a == uid_b
            assert score_a == pytest.approx(score_b, abs=1e-5)

    def test_save_with_no_adds_writes_empty_index(
        self, store: FAISSStore, index_path: Path
    ) -> None:
        # Saving an empty index must work (used by app startup
        # before any ingestion has happened).
        store.save()
        assert index_path.exists()
        assert Path(store.meta_path).exists()

    def test_save_is_idempotent(self, store: FAISSStore, index_path: Path) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        store.save()
        size_1 = index_path.stat().st_size
        store.save()
        size_2 = index_path.stat().st_size
        assert size_1 == size_2

    def test_load_preserves_uuid_mapping(self, store: FAISSStore, index_path: Path) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        store.save()

        reloaded = FAISSStore(index_path, embedding_dimension=TEST_DIM)
        reloaded.load()
        # The UUID → int mapping should be intact.
        results = reloaded.search(CORPUS_VECTORS[0], k=3)
        uuids = {uid for uid, _ in results}
        assert uuids == set(CORPUS_IDS)


class TestLoadEdgeCases:
    """Load error paths."""

    def test_load_missing_index_is_noop(self, store: FAISSStore) -> None:
        # No .faiss file at index_path — load is a no-op (first-run
        # case from the docstring).
        store.load()
        assert store.is_loaded is False
        assert store.size() == 0

    def test_load_index_without_sidecar_raises_corrupt(
        self, store: FAISSStore, index_path: Path
    ) -> None:
        # Add some vectors and save (this creates both files), then
        # delete only the sidecar. load() should raise Corrupt.
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        store.save()
        Path(store.meta_path).unlink()

        fresh = FAISSStore(index_path, embedding_dimension=TEST_DIM)
        with pytest.raises(VectorStoreCorruptError, match="meta file is missing"):
            fresh.load()

    def test_load_with_wrong_configured_dimension_raises(
        self, store: FAISSStore, index_path: Path
    ) -> None:
        # Save with dim=4, then try to load with dim=8 — mismatch.
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        store.save()

        wrong = FAISSStore(index_path, embedding_dimension=8)
        with pytest.raises(VectorStoreDimensionMismatchError, match="dimension"):
            wrong.load()

    def test_load_corrupt_sidecar_raises_corrupt(
        self, store: FAISSStore, index_path: Path
    ) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        store.save()
        # Overwrite the sidecar with invalid JSON.
        Path(store.meta_path).write_text("{not valid json", encoding="utf-8")

        fresh = FAISSStore(index_path, embedding_dimension=TEST_DIM)
        with pytest.raises(VectorStoreCorruptError, match="malformed"):
            fresh.load()

    def test_load_sidecar_missing_required_key_raises_corrupt(
        self, store: FAISSStore, index_path: Path
    ) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        store.save()
        # Overwrite with a valid JSON missing one required key.
        bad = {
            "embedding_model": "x",
            "embedding_dimension": 4,
            # "index_type" missing — IndexMeta.from_dict raises KeyError.
        }
        Path(store.meta_path).write_text(json.dumps(bad), encoding="utf-8")

        fresh = FAISSStore(index_path, embedding_dimension=TEST_DIM)
        with pytest.raises(VectorStoreCorruptError):
            fresh.load()

    def test_load_after_load_is_safe(self, store: FAISSStore, index_path: Path) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        store.save()
        fresh = FAISSStore(index_path, embedding_dimension=TEST_DIM)
        fresh.load()
        fresh.load()  # second load should be safe
        assert fresh.size() == 3


class TestDeleteBySource:
    """Remove vectors, verify size + sidecar mapping."""

    def test_delete_removes_one(self, store: FAISSStore) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        removed = store.delete_by_source("uuid-b")
        assert removed == 1
        assert store.size() == 2

    def test_delete_unknown_returns_zero(self, store: FAISSStore) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        removed = store.delete_by_source("uuid-doesnotexist")
        assert removed == 0
        assert store.size() == 3

    def test_delete_then_search_omits_removed(
        self, store: FAISSStore
    ) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        store.delete_by_source("uuid-b")
        results = store.search(CORPUS_VECTORS[0], k=3)
        uuids = {uid for uid, _ in results}
        assert "uuid-b" not in uuids
        assert uuids == {"uuid-a", "uuid-c"}

    def test_delete_then_search_k_still_respected(
        self, store: FAISSStore
    ) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        store.delete_by_source("uuid-b")
        # k larger than the new size — should return all available.
        results = store.search(CORPUS_VECTORS[0], k=10)
        assert len(results) == 2

    def test_remove_ids_batch(self, store: FAISSStore) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        # remove_ids is the underlying API — verify direct call works.
        removed = store.remove_ids(["uuid-a", "uuid-c"])
        assert removed == 2
        assert store.size() == 1
        results = store.search(CORPUS_VECTORS[1], k=5)
        # Only uuid-b should remain.
        assert [uid for uid, _ in results] == ["uuid-b"]

    def test_remove_ids_empty_list_returns_zero(self, store: FAISSStore) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        removed = store.remove_ids([])
        assert removed == 0
        assert store.size() == 3

    def test_remove_ids_silently_skips_unknown(
        self, store: FAISSStore
    ) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        # Mix of known + unknown — only the known one gets removed.
        removed = store.remove_ids(["uuid-a", "uuid-ghost"])
        assert removed == 1
        assert store.size() == 2

    def test_delete_updates_meta_num_vectors(
        self, store: FAISSStore
    ) -> None:
        # The sidecar's num_vectors should drop to match the
        # post-delete size (this is what the meta file will
        # record on the next save).
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        assert store._meta is not None
        assert store._meta.num_vectors == 3
        store.delete_by_source("uuid-b")
        assert store._meta is not None
        assert store._meta.num_vectors == 2


class TestSidecarMetaFile:
    """The ``.faiss.meta.json`` sidecar is correct after save."""

    def test_sidecar_is_valid_json(
        self, store: FAISSStore, index_path: Path
    ) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        store.save()
        raw = Path(store.meta_path).read_text(encoding="utf-8")
        data = json.loads(raw)
        assert isinstance(data, dict)

    def test_sidecar_has_required_keys(
        self, store: FAISSStore, index_path: Path
    ) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        store.save()
        data = json.loads(Path(store.meta_path).read_text(encoding="utf-8"))
        required = {
            "embedding_model",
            "embedding_dimension",
            "index_type",
            "normalize",
            "created_at",
            "last_modified",
            "num_vectors",
            "version",
            "id_to_uuid",
        }
        assert required.issubset(data.keys())

    def test_sidecar_records_dimension(
        self, store: FAISSStore, index_path: Path
    ) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        store.save()
        data = json.loads(Path(store.meta_path).read_text(encoding="utf-8"))
        assert data["embedding_dimension"] == TEST_DIM

    def test_sidecar_records_num_vectors(
        self, store: FAISSStore, index_path: Path
    ) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        store.save()
        data = json.loads(Path(store.meta_path).read_text(encoding="utf-8"))
        assert data["num_vectors"] == 3

    def test_sidecar_id_to_uuid_maps_correctly(
        self, store: FAISSStore, index_path: Path
    ) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        store.save()
        data = json.loads(Path(store.meta_path).read_text(encoding="utf-8"))
        # JSON keys are strings; FAISS IDs are ints. The mapping
        # should record every uuid we added.
        values = set(data["id_to_uuid"].values())
        assert values == set(CORPUS_IDS)

    def test_sidecar_index_type_matches_constant(
        self, store: FAISSStore, index_path: Path
    ) -> None:
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        store.save()
        data = json.loads(Path(store.meta_path).read_text(encoding="utf-8"))
        assert data["index_type"] == INDEX_TYPE

    def test_sidecar_normalize_true(
        self, store: FAISSStore, index_path: Path
    ) -> None:
        # The cosine-via-IP convention requires L2-normalised
        # vectors. The sidecar must record this so a future
        # load() can verify.
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        store.save()
        data = json.loads(Path(store.meta_path).read_text(encoding="utf-8"))
        assert data["normalize"] is True


class TestErrorHierarchy:
    """Every error subclasses ``VectorStoreError``; carries ``index_path``."""

    def test_dimension_mismatch_is_vector_store_error(self) -> None:
        err = VectorStoreDimensionMismatchError("dim", index_path="/tmp/x.faiss")
        assert isinstance(err, VectorStoreError)

    def test_corrupt_is_vector_store_error(self) -> None:
        err = VectorStoreCorruptError("bad", index_path="/tmp/x.faiss")
        assert isinstance(err, VectorStoreError)

    def test_search_is_vector_store_error(self) -> None:
        err = VectorStoreSearchError("search bad", index_path="/tmp/x.faiss")
        assert isinstance(err, VectorStoreError)

    def test_index_path_preserved(self) -> None:
        err = VectorStoreDimensionMismatchError("x", index_path="/data/x.faiss")
        assert err.index_path == "/data/x.faiss"

    def test_index_path_optional(self) -> None:
        # Errors can be constructed without an index_path (the
        # kwarg has a default of None).
        err = VectorStoreError("msg")
        assert err.index_path is None

    def test_error_message_preserved(self) -> None:
        err = VectorStoreError("the disk is on fire", index_path="/tmp/x.faiss")
        assert "the disk is on fire" in str(err)

    def test_dimension_mismatch_caught_by_base(self) -> None:
        # The API layer should be able to catch the base class
        # and dispatch on .index_path. Verify the base catches
        # every subclass.
        for cls in (
            VectorStoreDimensionMismatchError,
            VectorStoreCorruptError,
            VectorStoreSearchError,
        ):
            try:
                raise cls("test", index_path="/tmp/x.faiss")
            except VectorStoreError as caught:
                assert caught.index_path == "/tmp/x.faiss"
            else:
                pytest.fail(f"{cls.__name__} not caught by VectorStoreError")


class TestThreadSafety:
    """Concurrent adds/searches don't corrupt state."""

    def test_concurrent_adds_preserve_count(
        self, index_path: Path
    ) -> None:
        # Multiple threads adding to the same store — the
        # instance lock should serialise them so the final
        # size() == sum of adds.
        store = FAISSStore(index_path, embedding_dimension=TEST_DIM)
        n_threads = 5
        adds_per_thread = 10

        def worker(thread_id: int) -> None:
            for i in range(adds_per_thread):
                uid = f"t{thread_id}-v{i}"
                store.add([CORPUS_VECTORS[i % 3]], [uid])

        threads = [
            threading.Thread(target=worker, args=(t,)) for t in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert store.size() == n_threads * adds_per_thread

    def test_concurrent_add_and_search(self, store: FAISSStore) -> None:
        # Seed with a few vectors, then hammer the store with
        # concurrent adds + searches. We don't assert on results
        # (the search may or may not see a given addition
        # depending on order); we only assert no exception is
        # raised and the final size is correct.
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        n_threads = 4
        adds_per_thread = 5
        errors: list[BaseException] = []

        def writer() -> None:
            try:
                for i in range(adds_per_thread):
                    store.add([CORPUS_VECTORS[i % 3]], [f"new-uuid-{i}"])
            except BaseException as e:
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(20):
                    store.search(CORPUS_VECTORS[0], k=3)
            except BaseException as e:
                errors.append(e)

        threads: list[threading.Thread] = []
        for _ in range(n_threads // 2):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # Final size = 3 seeded + (n_threads/2 * adds_per_thread) added.
        expected = 3 + (n_threads // 2) * adds_per_thread
        assert store.size() == expected

    def test_concurrent_saves(self, store: FAISSStore, index_path: Path) -> None:
        # Saving the index while it's being written to: the lock
        # should serialise these. Multiple save() calls in
        # parallel should produce a valid final on-disk state.
        store.add(CORPUS_VECTORS, CORPUS_IDS)
        errors: list[BaseException] = []

        def saver() -> None:
            try:
                for _ in range(5):
                    store.save()
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=saver) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # The final on-disk state should still load cleanly.
        fresh = FAISSStore(index_path, embedding_dimension=TEST_DIM)
        fresh.load()
        assert fresh.size() == 3
