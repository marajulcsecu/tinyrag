"""Tests for src/tinyrag/ingestion/embedder.py — embedding model wrappers.

Test layout
-----------
- TestPublicSurface        — every public name is exported from the
  subpackage (``EmbeddingModel``, ``SentenceTransformerEmbedder``,
  ``FakeEmbedder``, ``EmbeddingError`` + subclasses).
- TestProtocolIsRuntime    — :class:`EmbeddingModel` is
  ``@runtime_checkable``; both concrete implementations satisfy it
  via duck-typing; a stub that lacks ``embed`` does not.
- TestFakeEmbedder         — deterministic SHA-256-based vectors;
  unit-norm; correct length; different texts → different vectors.
- TestErrorHierarchy       — :class:`EmbeddingError` is the base
  for :class:`EmbeddingModelNotFoundError` and
  :class:`EmbeddingDimensionMismatchError`.
- TestSentenceTransformerEmbedder (light) — construction is cheap
  (no model download until first ``embed()``); model_name is
  preserved; the Protocol methods are present.
- TestSentenceTransformerEmbedderReal (integration) — only runs
  if the configured model is already in the HF cache on disk
  (gated by :func:`_model_already_cached`). Verifies real
  384-dim output for ``all-MiniLM-L6-v2``, batch embedding,
  and that vectors are L2-normalised. Skipped on a fresh CI
  clone (no network) so the suite stays hermetic.

Why so much hermetic / so little integration?
----------------------------------------------
The real ``SentenceTransformerEmbedder`` downloads ~80 MB on first
load. We want ``pytest`` to be fast, offline, and green by default
(``make test-fast`` must work without internet). Most of the
behaviour — Protocol shape, error hierarchy, deterministic vectors,
the lazy-load contract — is fully testable with :class:`FakeEmbedder`
and a not-yet-loaded :class:`SentenceTransformerEmbedder`.

The single integration test class is a sanity check that catches
breaks in the ``sentence-transformers`` API surface
(``.encode(..., normalize_embeddings=True, ...)`` arguments) when
the model is locally available. It's skipped otherwise.

Why is FakeEmbedder in production code (not conftest.py)?
---------------------------------------------------------
See the module docstring of ``embedder.py``. We test the same
``FakeEmbedder`` that ``IngestionPipeline`` (Step 4.9) will use as
its default in tests, so by exercising it here we also confirm
the production code's stub still works.

Location: ``tests/test_embedder.py``
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tinyrag.config import EmbeddingSettings
from tinyrag.ingestion import (
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingModel,
    EmbeddingModelNotFoundError,
    FakeEmbedder,
    SentenceTransformerEmbedder,
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# The all-MiniLM-L6-v2 model id — must match the default in
# config.yaml. If you swap the default model, update the constant and
# the dimension (384) below.
_DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_DIM = 384

# Where the model lives once downloaded. We don't trigger a download
# in tests; we only run the real-model tests if the cache is present.
_DEFAULT_CACHE_DIR = Path("models") / "_hf_cache"


def _model_already_cached(model_id: str = _DEFAULT_MODEL_ID) -> bool:
    """Return True iff ``model_id`` is already in the HF cache.

    The HF cache layout (after one successful download) is
    ``models/_hf_cache/models--<org>--<name>/snapshots/<sha>/``.
    We check the directory's existence rather than any one file
    because the snapshot directory holds a small set of files
    (config, tokenizer, weights) and we don't want to enumerate
    them all here.
    """
    if not _DEFAULT_CACHE_DIR.exists():
        return False
    # Convert ``org/name`` to ``models--org--name`` (HF's on-disk format).
    folder_name = "models--" + model_id.replace("/", "--")
    return (_DEFAULT_CACHE_DIR / folder_name).exists()


def _make_settings(
    tmp_path: Path,
    *,
    model_name: str = _DEFAULT_MODEL_ID,
    cache_dir: Path | None = None,
    batch_size: int = 32,
) -> EmbeddingSettings:
    """Build an ``EmbeddingSettings`` instance pointed at ``tmp_path``.

    Using ``tmp_path`` keeps each test's cache_dir isolated and
    self-cleaning (pytest reaps it on teardown). We construct the
    object directly rather than loading from config.yaml so tests
    don't depend on the on-disk YAML.
    """
    return EmbeddingSettings(
        model_name=model_name,
        cache_dir=str(cache_dir if cache_dir is not None else tmp_path),
        batch_size=batch_size,
    )


# ----------------------------------------------------------------------------
# Test classes
# ----------------------------------------------------------------------------


class TestPublicSurface:
    """The expected symbols are exported and importable."""

    def test_subpackage_exports_embedding_model_protocol(self) -> None:
        from tinyrag.ingestion import EmbeddingModel as cls

        assert cls is EmbeddingModel

    def test_subpackage_exports_sentence_transformer_embedder(self) -> None:
        from tinyrag.ingestion import SentenceTransformerEmbedder as cls

        assert cls is SentenceTransformerEmbedder

    def test_subpackage_exports_fake_embedder(self) -> None:
        from tinyrag.ingestion import FakeEmbedder as cls

        assert cls is FakeEmbedder

    def test_subpackage_exports_embedding_error(self) -> None:
        from tinyrag.ingestion import EmbeddingError as cls

        assert cls is EmbeddingError

    def test_subpackage_exports_model_not_found_error(self) -> None:
        from tinyrag.ingestion import EmbeddingModelNotFoundError as cls

        assert cls is EmbeddingModelNotFoundError

    def test_subpackage_exports_dimension_mismatch_error(self) -> None:
        from tinyrag.ingestion import EmbeddingDimensionMismatchError as cls

        assert cls is EmbeddingDimensionMismatchError


class TestProtocolIsRuntime:
    """``EmbeddingModel`` is duck-typeable via ``@runtime_checkable``."""

    def test_fake_embedder_is_an_embedding_model(self) -> None:
        assert isinstance(FakeEmbedder(), EmbeddingModel)

    def test_sentence_transformer_embedder_is_an_embedding_model(
        self, tmp_path: Path
    ) -> None:
        settings = _make_settings(tmp_path)
        assert isinstance(SentenceTransformerEmbedder(settings), EmbeddingModel)

    def test_non_embedder_is_not_an_embedding_model(self) -> None:
        """An object lacking ``embed`` should fail isinstance()."""

        class NotAnEmbedder:
            @property
            def dimension(self) -> int:  # pragma: no cover — never called
                return 0

        assert not isinstance(NotAnEmbedder(), EmbeddingModel)

    def test_object_missing_dimension_is_not_an_embedding_model(self) -> None:
        class MissingDimension:
            def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
                return [[0.0]]

        assert not isinstance(MissingDimension(), EmbeddingModel)


class TestFakeEmbedder:
    """``FakeEmbedder`` produces deterministic, well-shaped vectors."""

    def test_dimension_property_matches_constructor(self) -> None:
        for d in (32, 64, 128, 384, 768):
            assert FakeEmbedder(dimension=d).dimension == d

    def test_zero_or_negative_dimension_raises(self) -> None:
        for d in (0, -1, -384):
            with pytest.raises(ValueError):
                FakeEmbedder(dimension=d)

    def test_embed_returns_one_vector_per_input(self) -> None:
        emb = FakeEmbedder(dimension=384)
        vecs = emb.embed(["hello", "world", "foo bar"])
        assert len(vecs) == 3

    def test_embed_returns_list_of_python_floats(self) -> None:
        emb = FakeEmbedder(dimension=32)
        vecs = emb.embed(["hello"])
        assert len(vecs) == 1
        vec = vecs[0]
        assert isinstance(vec, list)
        assert all(isinstance(x, float) for x in vec)
        # Must NOT be a numpy array — the metadata store (SQLite
        # + JSON) needs JSON-safe Python types.
        assert type(vec) is list

    def test_each_vector_has_requested_dimension(self) -> None:
        for d in (1, 16, 33, 384, 768):  # edge: 1, >32, exactly 32, large
            vecs = FakeEmbedder(dimension=d).embed(["a", "b"])
            assert len(vecs[0]) == d
            assert len(vecs[1]) == d

    def test_vectors_are_unit_length(self) -> None:
        """Real sentence-transformers returns L2-normalised vectors;
        ``FakeEmbedder`` must match so FAISS cosine similarity is
        consistent across the two."""
        emb = FakeEmbedder(dimension=384)
        for vec in emb.embed(["a", "b", "c", "longer text with spaces"]):
            norm = math.sqrt(sum(x * x for x in vec))
            assert abs(norm - 1.0) < 1e-9, f"vector not unit-length: norm={norm}"

    def test_deterministic_for_same_input(self) -> None:
        emb = FakeEmbedder(dimension=384)
        a = emb.embed(["the quick brown fox"])
        b = emb.embed(["the quick brown fox"])
        assert a == b  # exact float equality — SHA-256 is deterministic

    def test_different_inputs_produce_different_vectors(self) -> None:
        emb = FakeEmbedder(dimension=384)
        a = emb.embed(["foo"])
        b = emb.embed(["bar"])
        # 384-dim SHA-256-derived vectors: collision is astronomically
        # unlikely (1 in ~2^256). Asserting inequality is safe.
        assert a != b

    def test_empty_list_returns_empty_list(self) -> None:
        assert FakeEmbedder(dimension=384).embed([]) == []

    def test_empty_string_is_accepted(self) -> None:
        """sentence-transformers can embed ``""``; the fake must too."""
        vecs = FakeEmbedder(dimension=32).embed([""])
        assert len(vecs) == 1
        assert len(vecs[0]) == 32
        # Unit-norm even for the empty string.
        norm = math.sqrt(sum(x * x for x in vecs[0]))
        assert abs(norm - 1.0) < 1e-9

    def test_unicode_input_is_accepted(self) -> None:
        """UTF-8 text (multilingual docs) must not raise."""
        vecs = FakeEmbedder(dimension=32).embed(["café 🏠", "日本語テスト", "বাংলা"])
        assert len(vecs) == 3
        for v in vecs:
            assert len(v) == 32


class TestErrorHierarchy:
    """Both embedder exceptions inherit from :class:`EmbeddingError`."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            EmbeddingModelNotFoundError,
            EmbeddingDimensionMismatchError,
        ],
    )
    def test_subclass_of_embedding_error(
        self, exc_cls: type[EmbeddingError]
    ) -> None:
        assert issubclass(exc_cls, EmbeddingError)

    def test_can_catch_all_with_embedding_error(self) -> None:
        """A single ``except EmbeddingError`` catches both subclasses."""
        for exc_cls in (
            EmbeddingModelNotFoundError,
            EmbeddingDimensionMismatchError,
        ):
            with pytest.raises(EmbeddingError):
                raise exc_cls("test", model_name="dummy/model")

    def test_model_name_is_preserved_on_exception(self) -> None:
        exc = EmbeddingModelNotFoundError("boom", model_name="x/y")
        assert exc.model_name == "x/y"
        # And it survives str(exc).
        assert "boom" in str(exc)

    def test_default_model_name_is_none(self) -> None:
        exc = EmbeddingError("just a base error")
        assert exc.model_name is None


class TestSentenceTransformerEmbedderConstruction:
    """Construction is cheap — no model is downloaded at ``__init__``."""

    def test_construction_does_not_load_model(
        self, tmp_path: Path
    ) -> None:
        """A bad model name should NOT raise at construction —
        the load is deferred to the first ``embed()`` call."""
        settings = _make_settings(
            tmp_path, model_name="this/is/not/a/real/model/anywhere"
        )
        # Should not raise; we haven't asked for embedding yet.
        emb = SentenceTransformerEmbedder(settings)
        assert emb.is_loaded is False

    def test_model_name_property_returns_settings(
        self, tmp_path: Path
    ) -> None:
        settings = _make_settings(tmp_path, model_name="some/custom-model")
        emb = SentenceTransformerEmbedder(settings)
        assert emb.model_name == "some/custom-model"

    def test_is_loaded_starts_false(
        self, tmp_path: Path
    ) -> None:
        emb = SentenceTransformerEmbedder(_make_settings(tmp_path))
        assert emb.is_loaded is False

    def test_dimension_property_forces_load(
        self, tmp_path: Path
    ) -> None:
        """Reading ``.dimension`` triggers the model load — this is
        the documented contract.

        We use a deliberately bogus model id so the lazy load raises
        :class:`EmbeddingModelNotFoundError` regardless of whether the
        default model happens to be on disk. That makes the test
        hermetic (no network) while still proving that *reading*
        ``.dimension`` (not constructing) is what triggers the load.
        """
        emb = SentenceTransformerEmbedder(
            _make_settings(tmp_path, model_name="not/a/real/model-xyz")
        )
        assert emb.is_loaded is False
        with pytest.raises(EmbeddingModelNotFoundError):
            _ = emb.dimension


class TestSentenceTransformerEmbedderEmbedContract:
    """The ``.embed()`` method honours its documented contract."""

    def test_empty_list_returns_empty_without_loading(
        self, tmp_path: Path
    ) -> None:
        """``embed([])`` must short-circuit — no work, no model load."""
        emb = SentenceTransformerEmbedder(_make_settings(tmp_path))
        assert emb.embed([]) == []
        assert emb.is_loaded is False  # load was never triggered

    def test_bad_model_name_raises_on_first_embed(
        self, tmp_path: Path
    ) -> None:
        """Construction is cheap, but the first ``.embed()`` call
        is when we try to load the model — and a bogus name fails there."""
        emb = SentenceTransformerEmbedder(
            _make_settings(tmp_path, model_name="definitely/not/a/real/model-xyz")
        )
        with pytest.raises(EmbeddingModelNotFoundError) as excinfo:
            emb.embed(["hello"])
        # The exception should carry the bad name for diagnostics.
        assert excinfo.value.model_name == "definitely/not/a/real/model-xyz"


# ----------------------------------------------------------------------------
# Real-model integration tests
# ----------------------------------------------------------------------------
#
# These run only when the configured model is already on disk (no
# network calls during tests). On a fresh clone, they are skipped —
# the hermetic tests above still cover the contract.


@pytest.mark.skipif(
    not _model_already_cached(),
    reason=f"Model {_DEFAULT_MODEL_ID!r} not in {_DEFAULT_CACHE_DIR}; "
    "download once via `make download-llm` (or run scripts/download_models.py) "
    "to enable the real-model tests.",
)
class TestSentenceTransformerEmbedderReal:
    """Integration tests against a real sentence-transformers model.

    Skipped unless the model is already cached. They verify the
    exact contract advertised in the roadmap: 384-dim vectors,
    deterministic across runs, batched input works, L2-normalised.
    """

    def test_dimension_is_384(self, tmp_path: Path) -> None:
        emb = SentenceTransformerEmbedder(_make_settings(tmp_path))
        assert emb.dimension == _DEFAULT_DIM
        assert emb.is_loaded is True

    def test_embed_returns_one_vector_per_input(
        self, tmp_path: Path
    ) -> None:
        emb = SentenceTransformerEmbedder(_make_settings(tmp_path))
        vecs = emb.embed(["hello world", "goodbye world"])
        assert len(vecs) == 2
        for v in vecs:
            assert len(v) == _DEFAULT_DIM

    def test_vectors_are_l2_normalised(self, tmp_path: Path) -> None:
        emb = SentenceTransformerEmbedder(_make_settings(tmp_path))
        for v in emb.embed(["some text", "another phrase", "yet another"]):
            norm = math.sqrt(sum(x * x for x in v))
            assert abs(norm - 1.0) < 1e-3, f"expected unit norm, got {norm}"

    def test_vectors_are_deterministic(self, tmp_path: Path) -> None:
        emb1 = SentenceTransformerEmbedder(_make_settings(tmp_path))
        emb2 = SentenceTransformerEmbedder(_make_settings(tmp_path))
        v1 = emb1.embed(["the quick brown fox jumps over the lazy dog"])
        v2 = emb2.embed(["the quick brown fox jumps over the lazy dog"])
        # Different model instances, but the same model id and the
        # same text — vectors should be numerically equal to fp32
        # precision (sentence-transformers is deterministic in CPU
        # inference mode).
        for a, b in zip(v1[0], v2[0], strict=False):
            assert abs(a - b) < 1e-5

    def test_semantic_similarity(self, tmp_path: Path) -> None:
        """Paraphrases should be closer than unrelated sentences.

        This is the *behavioural* test: not just that we get 384-dim
        vectors, but that they carry semantic meaning (a property
        FakeEmbedder does NOT have, by design).
        """
        emb = SentenceTransformerEmbedder(_make_settings(tmp_path))
        vecs = emb.embed(
            [
                "how do I reset the thermostat?",
                "what's the procedure to factory-reset the thermostat?",
                "what is the capital of France?",
            ]
        )
        # Cosine similarity on L2-normalised vectors = dot product.
        def cos(a: list[float], b: list[float]) -> float:
            return sum(x * y for x, y in zip(a, b, strict=False))

        sim_paraphrase = cos(vecs[0], vecs[1])
        sim_unrelated = cos(vecs[0], vecs[2])
        assert sim_paraphrase > sim_unrelated, (
            f"paraphrase similarity ({sim_paraphrase}) should exceed "
            f"unrelated similarity ({sim_unrelated})"
        )

    def test_batch_size_is_honoured(self, tmp_path: Path) -> None:
        """Smaller batch_size should still produce correct vectors.

        We don't measure throughput here (that's a benchmark), just
        that the ``batch_size`` setting is accepted and doesn't break
        correctness.
        """
        emb_small = SentenceTransformerEmbedder(
            _make_settings(tmp_path, batch_size=1)
        )
        emb_large = SentenceTransformerEmbedder(
            _make_settings(tmp_path, batch_size=8)
        )
        texts = ["alpha", "beta", "gamma", "delta"]
        v_small = emb_small.embed(texts)
        v_large = emb_large.embed(texts)
        assert len(v_small) == len(v_large) == 4
        # Same model, same texts → vectors should match across batch sizes.
        for a, b in zip(v_small, v_large, strict=False):
            for x, y in zip(a, b, strict=False):
                assert abs(x - y) < 1e-5

    def test_empty_list_short_circuits(self, tmp_path: Path) -> None:
        emb = SentenceTransformerEmbedder(_make_settings(tmp_path))
        assert emb.embed([]) == []

    def test_unicode_text_is_embedded(self, tmp_path: Path) -> None:
        """Embedding model is English-trained, but must accept UTF-8
        input without raising (degraded quality is acceptable)."""
        emb = SentenceTransformerEmbedder(_make_settings(tmp_path))
        vecs = emb.embed(["café", "日本語", "বাংলা"])
        assert len(vecs) == 3
        for v in vecs:
            assert len(v) == _DEFAULT_DIM
