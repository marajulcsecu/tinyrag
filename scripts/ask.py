#!/usr/bin/env python3
"""End-to-end RAG query CLI — the Step 4.16 risk gate.

This is the **single-script Phase 4 checkpoint** from
``docs/06_roadmap_v2.md`` Step 4.16. It does four things, in order:

1. ``Retriever.retrieve(query)`` — embed the query, search the doc
   + (keyword-routed) sensor FAISS indices, merge per-id with
   score-max, threshold-filter, score-DESC sort (Step 4.12).
2. ``PromptBuilder.build(query, chunks)`` — assemble the grounded
   2-message prompt with token-budget-aware tail-trim (Step 4.11).
3. ``LLMClient.generate(messages)`` — stream tokens from
   ``llama-server`` (or the FakeLLMClient for hermetic tests) and
   accumulate the reply (Step 4.10).
4. ``MetadataStore.log_query(...)`` — append a row to the
   ``query_log`` table so the eval set (Phase 5) can grade
   answer quality and latency (Step 4.7).

At the end it prints an :class:`Answer` summary (the model's reply
plus a numbered "Sources:" footer) and a per-stage timing banner.

CLI flags
---------

    query                   Question to ask (positional; required).
    --config PATH           Path to config.yaml (default: ./config.yaml).
    --db-path PATH          Override metadata DB path.
    --doc-index PATH        Override doc FAISS index path.
    --sensor-index PATH     Override sensor FAISS index path.
    --llm {real|fake}       Which LLMClient to use. Default: real
                            (LlamaCppClient against llama-server).
                            "fake" uses FakeLLMClient — hermetic, no
                            model server required; great for CI.
    --k-doc INT             Number of doc-index hits to request (default: 3).
    --k-sensor INT          Number of sensor-index hits to request (default: 2).
    --threshold FLOAT       Minimum cosine similarity for a chunk (default: 0.3).
    --max-tokens INT        Cap on generated tokens (default: 512).
    --no-log                Skip writing the query_log row (for offline tests).
    --json                  Print JSON result instead of pretty text.
    --quiet                 Suppress pretty banner; print only the JSON
                            summary on success, error on failure.

Exit codes
----------

    0   Query answered; reply printed.
    1   Pipeline error (retriever / LLM / DB failure).
    2   Bad CLI args (argparse handles this with code 2).

Companion docs
--------------
- ``src/tinyrag/core/retriever.py`` — Retriever.retrieve
- ``src/tinyrag/core/prompt_builder.py`` — PromptBuilder.build
- ``src/tinyrag/core/answer.py`` — Answer + Citation dataclasses
- ``src/tinyrag/generation/llm_client.py`` — LLMClient.generate
- ``src/tinyrag/storage/metadata.py`` — MetadataStore.log_query
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Make ``src/`` importable when this script is run directly without
# ``pip install -e .``. After Phase 4 the project will be installed
# and this block becomes a no-op.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tinyrag.config import Settings, load_settings  # noqa: E402
from tinyrag.core import (  # noqa: E402
    Answer,
    PromptBuilder,
    PromptBuilderError,
    Retriever,
    adapt_metadata_store,
    build_citations_from_chunks,
)
from tinyrag.generation import (  # noqa: E402
    FakeLLMClient,
    LlamaCppClient,
    LLMClient,
    LLMUnavailableError,
)
from tinyrag.ingestion import (  # noqa: E402
    EmbeddingModel,
    FakeEmbedder,
    SentenceTransformerEmbedder,
)
from tinyrag.storage import FAISSStore, MetadataStore  # noqa: E402

# ---------------------------------------------------------------------------
# Component factories (each is independently testable)
# ---------------------------------------------------------------------------


def _make_embedder(settings: Settings, *, kind: str | None = None) -> EmbeddingModel:
    """Build an :class:`EmbeddingModel` from the typed Settings.

    The real model is lazy-loaded on first ``.embed()`` call (see
    Step 4.6) so this constructor is cheap. We default to the
    real embedder (matches the project's production path); tests
    pass ``kind="fake"`` to keep the suite hermetic.

    Parameters
    ----------
    settings:
        The typed :class:`Settings` (always passed — we never read
        config files from this module).
    kind:
        ``"real"`` (default) → :class:`SentenceTransformerEmbedder`
        with the configured model name + device + batch size.
        ``"fake"`` → :class:`FakeEmbedder` at dimension 384 (the
        MiniLM dimension, matches the FAISS index on disk).
    """
    if kind == "fake":
        return FakeEmbedder(dimension=384)
    if kind is None or kind == "real":
        # SentenceTransformerEmbedder takes the EmbeddingSettings
        # sub-model directly — it pulls model_name, device, batch_size,
        # cache_dir from there. See src/tinyrag/ingestion/embedder.py.
        return SentenceTransformerEmbedder(settings.embedding)
    raise ValueError(f"unknown embedder kind: {kind!r}")  # pragma: no cover


def _make_llm(settings: Settings, *, kind: str) -> LLMClient:
    """Build an :class:`LLMClient` (real or fake) per the --llm flag.

    The real :class:`LlamaCppClient` doesn't open a connection
    until :meth:`generate` is called (matches Step 4.10's lazy
    httpx.Client pattern). ``is_healthy()`` is the cheap probe
    used by the smoke test.

    Parameters
    ----------
    settings:
        The typed :class:`Settings`. ``server_url`` and
        ``model_path`` are mapped to the client's ``base_url`` +
        ``model`` args; ``max_tokens`` and ``temperature`` are
        the call-time defaults.
    kind:
        ``"real"`` → :class:`LlamaCppClient` against the
        configured server URL + model path.
        ``"fake"`` → :class:`FakeLLMClient` with a deterministic
        canned reply (returns ``"I'm a fake LLM. This is a canned
        response."`` unless overridden).
    """
    if kind == "fake":
        return FakeLLMClient()
    if kind == "real":
        # Strip the ".gguf" extension — OpenAI-compatible APIs use
        # the bare model id. The config stores the full filename
        # so the launcher (scripts/run_llamacpp.sh) can find the
        # weights, but the API request only needs the id.
        model_id = settings.llm.model_path
        if model_id.endswith(".gguf"):
            model_id = model_id[: -len(".gguf")]
        # Strip a trailing slash from server_url so base_url + path
        # concat is clean (mirrors LlamaCppClient.__post_init__).
        return LlamaCppClient(
            base_url=settings.llm.server_url.rstrip("/"),
            model=model_id,
        )
    raise ValueError(f"unknown llm kind: {kind!r}")  # pragma: no cover


def _make_retriever(
    settings: Settings,
    *,
    embedder: EmbeddingModel,
    doc_store_path: Path,
    sensor_store_path: Path,
    metadata: MetadataStore,
    default_threshold: float | None = None,
) -> Retriever:
    """Wire the doc + sensor FAISS indices, metadata accessor, and embedder.

    Both FAISS indices are loaded with the configured dimension
    (the embedder's ``.dimension`` — should be 384 for MiniLM).
    A dimension mismatch raises :class:`VectorStoreDimensionMismatchError`
    at ``load()`` time (per the Step 4.8 contract) so the caller
    sees a clean error instead of a corrupt index.

    ``default_threshold`` is the retriever's cosine-similarity
    cut-off (defaults to :class:`Retriever`'s own default, 0.3).
    The CLI passes through whatever the caller wants — tests use
    0.0 to bypass the cut-off against the FakeEmbedder (whose
    SHA-256-derived vectors don't model semantic similarity).
    """
    dimension = embedder.dimension
    doc_store = FAISSStore(
        index_path=doc_store_path,
        embedding_dimension=dimension,
        embedding_model=settings.embedding.model_name,
    )
    doc_store.load()
    sensor_store = FAISSStore(
        index_path=sensor_store_path,
        embedding_dimension=dimension,
        embedding_model=settings.embedding.model_name,
    )
    sensor_store.load()
    kwargs: dict[str, Any] = {
        "embedder": embedder,
        "doc_store": doc_store,
        "sensor_store": sensor_store,
        "metadata": adapt_metadata_store(metadata),
    }
    if default_threshold is not None:
        kwargs["default_threshold"] = default_threshold
    return Retriever(**kwargs)


# ---------------------------------------------------------------------------
# The query run
# ---------------------------------------------------------------------------


def run_ask(
    *,
    query: str,
    settings: Settings,
    llm_kind: str,
    db_path_override: str | None,
    doc_index_override: str | None,
    sensor_index_override: str | None,
    k_doc: int,
    k_sensor: int,
    threshold: float,
    max_tokens: int,
    log_query: bool,
    default_threshold: float | None = None,
    embedder_kind: str | None = None,
) -> Answer:
    """Run the full RAG pipeline for one query.

    Four stages, each timed:

    1. **Retrieve** — embed the query, search doc + (keyword-
       routed) sensor indices, threshold-filter, score-DESC sort.
    2. **Prompt** — assemble the grounded 2-message prompt with
       token-budget tail-trim.
    3. **LLM** — call ``LLMClient.generate(messages)`` and
       accumulate the streamed reply.
    4. **Log** — append a row to the ``query_log`` table
       (unless ``log_query=False``).

    Every stage's exception is caught and re-packaged into a
    failed :class:`Answer` — the script should never crash with
    a traceback (the caller wants a clean exit code).

    Parameters
    ----------
    query:
        The user's question. Must be non-empty (the CLI enforces
        this with argparse, but we re-check here for safety).
    settings:
        The typed :class:`Settings` (single source of truth for
        every config value).
    llm_kind:
        ``"real"`` → :class:`LlamaCppClient`; ``"fake"`` →
        :class:`FakeLLMClient`.
    db_path_override, doc_index_override, sensor_index_override:
        Optional CLI overrides for the default paths
        (``settings.paths.metadata_db``,
        ``settings.retrieval.doc_index_path``,
        ``settings.retrieval.sensor_index_path``).
    k_doc, k_sensor, threshold:
        Passed straight to :meth:`Retriever.retrieve`.
    max_tokens:
        Passed straight to :meth:`LLMClient.generate`.
    log_query:
        If ``False``, skip the ``query_log`` write. Tests use this
        so they can run the pipeline hermetically without writing
        to the DB.
    default_threshold:
        Optional override for the retriever's default cosine-
        similarity cut-off (used when ``threshold`` is not passed).
        The CLI never sets this; tests do so the FakeEmbedder's
        non-semantic SHA-256 vectors can still pass the filter.
    embedder_kind:
        ``"real"`` (default) → :class:`SentenceTransformerEmbedder`
        using the configured model. ``"fake"`` →
        :class:`FakeEmbedder` at dimension 384. The embedder used
        here MUST match the embedder used to build the FAISS
        indices (production is always "real"; tests that built
        the index with FakeEmbedder must pass ``"fake"``).
    """
    # Empty-query guard. The CLI's argparse also enforces this, but
    # a programmatic caller (or a future REPL) might skip the CLI.
    if not query or not query.strip():
        return Answer(
            query=query,
            text="",
            model_name="",
        )

    db_path = Path(db_path_override) if db_path_override else Path(settings.paths.metadata_db)
    doc_index_path = (
        Path(doc_index_override)
        if doc_index_override
        else Path(settings.retrieval.doc_index_path)
    )
    sensor_index_path = (
        Path(sensor_index_override)
        if sensor_index_override
        else Path(settings.retrieval.sensor_index_path)
    )

    timings: dict[str, float] = {}
    t_total_start = time.monotonic()

    # ---- Build components (no I/O yet — the embedder is lazy) ----
    # ``embedder_kind`` defaults to None → real (matches the
    # production path). Tests that built the index with
    # FakeEmbedder pass ``embedder_kind="fake"`` so the query and
    # chunks live in the same vector space.
    embedder = _make_embedder(settings, kind=embedder_kind or "real")
    llm = _make_llm(settings, kind=llm_kind)
    metadata = MetadataStore(db_path)
    metadata.init_schema()
    retriever = _make_retriever(
        settings,
        embedder=embedder,
        doc_store_path=doc_index_path,
        sensor_store_path=sensor_index_path,
        metadata=metadata,
        default_threshold=default_threshold,
    )
    prompt_builder = PromptBuilder.from_chunking_settings(settings.chunking)

    # ---- Stage 1: retrieve --------------------------------------------
    t = time.monotonic()
    retrieval_error: str | None = None
    try:
        retrieval = retriever.retrieve(
            query,
            k_doc=k_doc,
            k_sensor=k_sensor,
            threshold=threshold,
        )
    except Exception as exc:
        retrieval_error = f"retrieve failed: {type(exc).__name__}: {exc}"
        retrieval = None  # type: ignore[assignment]
    timings["retrieve_ms"] = (time.monotonic() - t) * 1000.0

    if retrieval_error is not None:
        return _failed_answer(query=query, error=retrieval_error, **timings)

    # ---- Stage 2: prompt ----------------------------------------------
    t = time.monotonic()
    prompt_error: str | None = None
    try:
        prompt = prompt_builder.build(query, retrieval.chunks)  # type: ignore[union-attr]
    except PromptBuilderError as exc:
        prompt_error = f"prompt build failed: {exc}"
        prompt = None  # type: ignore[assignment]
    timings["prompt_ms"] = (time.monotonic() - t) * 1000.0

    if prompt_error is not None or prompt is None:
        return _failed_answer(query=query, error=prompt_error or "prompt was None", **timings)

    # ---- Stage 3: LLM generate ---------------------------------------
    t = time.monotonic()
    llm_error: str | None = None
    try:
        text, gen_stats = llm.generate(prompt.messages, max_tokens=max_tokens)
    except LLMUnavailableError as exc:
        llm_error = f"LLM unavailable: {exc}"
        text, gen_stats = "", _zero_stats()
    except Exception as exc:
        llm_error = f"LLM failed: {type(exc).__name__}: {exc}"
        text, gen_stats = "", _zero_stats()
    timings["llm_ms"] = (time.monotonic() - t) * 1000.0

    if llm_error is not None:
        return _failed_answer(query=query, error=llm_error, **timings)

    timings["total_ms"] = (time.monotonic() - t_total_start) * 1000.0

    # ---- Build the citations list -------------------------------------
    # The prompt builder's selected-chunks are exactly the first
    # ``prompt.chunks_used`` items of ``retrieval.chunks`` (the
    # builder never reorders). Pair them with their scores for
    # the citation footer.
    citations = build_citations_from_chunks(
        chunks=retrieval.chunks[: prompt.chunks_used],  # type: ignore[union-attr]
        scores=retrieval.scores[: prompt.chunks_used],  # type: ignore[union-attr]
    )

    answer = Answer(
        query=query,
        text=text,
        used_sensor_idx=retrieval.used_sensor_idx,  # type: ignore[union-attr]
        top_score=retrieval.top_score,  # type: ignore[union-attr]
        model_name=llm.model_name(),
        citations=citations,
        chunks_used=prompt.chunks_used,
        chunks_dropped=prompt.chunks_dropped,
        prompt_tokens=prompt.prompt_tokens,
        completion_tokens=gen_stats.completion_tokens,
        total_tokens=gen_stats.total_tokens or (prompt.prompt_tokens + gen_stats.completion_tokens),
        duration_retrieve_ms=timings.get("retrieve_ms", 0.0),
        duration_prompt_ms=timings.get("prompt_ms", 0.0),
        duration_llm_ms=timings.get("llm_ms", 0.0),
        duration_total_ms=timings.get("total_ms", 0.0),
    )

    # ---- Stage 4: log the query (best-effort) -------------------------
    if log_query:
        t = time.monotonic()
        try:
            metadata.log_query(
                query=query,
                top1_score=retrieval.top_score,  # type: ignore[union-attr]
                num_chunks=prompt.chunks_used,
                retrieval_ms=int(timings.get("retrieve_ms", 0.0)),
                generation_ms=int(timings.get("llm_ms", 0.0)),
                total_ms=int(timings.get("total_ms", 0.0)),
                model=llm.model_name(),
                used_sensor_idx=1 if retrieval.used_sensor_idx else 0,  # type: ignore[union-attr]
            )
        except Exception as exc:
            # Non-fatal — the answer is already correct, the log
            # write is observability. Surface as a warning in the
            # CLI banner but don't fail the run.
            answer_dict = answer.to_dict()
            answer_dict.setdefault("warnings", []).append(f"log_query failed: {exc}")
            # Rebuild the Answer with the warning preserved. We
            # can't add fields to a frozen dataclass, so we encode
            # the warning via the JSON extras instead — see
            # print_human() for the rendering.
            answer = _with_warning(answer, f"log_query failed: {exc}")
        timings["log_ms"] = (time.monotonic() - t) * 1000.0

    return answer


def _zero_stats():
    """Return a zero-valued :class:`GenerationStats`-like tuple.

    Used when the LLM call raises — the function still returns
    an :class:`Answer` so the caller can surface the error.
    """
    from tinyrag.generation.llm_client import GenerationStats
    return GenerationStats()


def _failed_answer(*, query: str, error: str, **timings: float) -> Answer:
    """Build an :class:`Answer` with ``text=""`` and a recorded error.

    The CLI prints the error in red; the JSON mode surfaces it
    under ``answer.text == ""`` plus a marker. (We don't have an
    ``error`` field on :class:`Answer` itself — the failure
    semantics are encoded by the empty text + the CLI's exit
    code 1.)
    """
    return Answer(
        query=query,
        text="",
        duration_retrieve_ms=timings.get("retrieve_ms", 0.0),
        duration_prompt_ms=timings.get("prompt_ms", 0.0),
        duration_llm_ms=timings.get("llm_ms", 0.0),
        duration_total_ms=timings.get("total_ms", 0.0),
    )


def _with_warning(answer: Answer, warning: str) -> Answer:
    """Return a copy of ``answer`` with a warning attached.

    Frozen dataclasses can't carry mutable state, so we stash
    the warning as an attribute the dataclass doesn't know about
    via ``object.__setattr__`` (safe because we're not violating
    the frozen contract — the warning is metadata about the
    dataclass, not a field). The CLI checks for ``_warning``
    before printing.
    """
    object.__setattr__(answer, "_warning", warning)
    return answer


# ---------------------------------------------------------------------------
# Output formatting
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


def _bold(s: str) -> str:
    return _c("1", s)


def _dim(s: str) -> str:
    return _c("2", s)


def print_human(answer: Answer, *, quiet: bool) -> None:
    """Print a friendly summary to stdout.

    Layout:

    - Header banner with the query + model.
    - The answer text (bold).
    - "Sources:" footer with one line per citation.
    - A "diagnostics" block with per-stage timings + token counts.

    The ``--quiet`` flag suppresses everything except the JSON
    summary on success, or the error line on failure (mirrors
    ``scripts/ingest.py``).
    """
    if quiet:
        if answer.text:
            print(json.dumps(answer.to_dict()))
        else:
            print("ERROR: query failed (no answer text)", file=sys.stderr)
        return

    print(_bold("==> TinyRAG — Answer"))
    print(f"    query:    {answer.query}")
    print(f"    model:    {answer.model_name or '(unknown)'}")
    print()

    if not answer.text:
        print(_red("[FAIL]") + " Query failed (empty answer text).")
        return

    print(_bold("ANSWER:"))
    # Indent every line of the answer for readability.
    for line in answer.text.splitlines() or [""]:
        print(f"  {line}")
    print()

    if answer.citations:
        print(_bold("SOURCES:"))
        for c in answer.citations:
            score_str = f"score={c.score:.3f}"
            print(f"  {c.ref} {c.location}  ({score_str})")
            print(f"      {_dim(c.preview)}")
        print()

    # Diagnostics block.
    print(_bold("DIAGNOSTICS:"))
    print(
        f"  used_sensor_idx: {answer.used_sensor_idx}  "
        f"top_score: {answer.top_score if answer.top_score is not None else 'n/a'}"
    )
    print(
        f"  chunks_used: {answer.chunks_used}  "
        f"chunks_dropped: {answer.chunks_dropped}"
    )
    print(
        f"  prompt_tokens: {answer.prompt_tokens}  "
        f"completion_tokens: {answer.completion_tokens}  "
        f"total_tokens: {answer.total_tokens}"
    )
    print(f"  retrieve: {answer.duration_retrieve_ms:>7.2f} ms")
    print(f"  prompt:   {answer.duration_prompt_ms:>7.2f} ms")
    print(f"  llm:      {answer.duration_llm_ms:>7.2f} ms")
    print(f"  TOTAL:    {answer.duration_total_ms:>7.2f} ms")

    warning = getattr(answer, "_warning", None)
    if warning:
        print()
        print(_c("33", f"  warning: {warning}"))


def print_json(answer: Answer) -> None:
    """Print the result as a single JSON object.

    Includes the optional ``warning`` field if the dataclass was
    annotated with one via :func:`_with_warning`.
    """
    payload: dict[str, Any] = answer.to_dict()
    warning = getattr(answer, "_warning", None)
    if warning:
        payload["warning"] = warning
    print(json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Settings helper — load with optional override of config path
# ---------------------------------------------------------------------------


def _load_settings(config_path: str | None) -> Settings:
    """Load the typed :class:`Settings`, optionally from a custom config.

    Mirrors :func:`scripts.ingest._load_settings` — the override
    is for tests + power users.
    """
    if config_path is None:
        return load_settings()
    from tinyrag.config import load_settings as _ls
    return _ls(config_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ask.py",
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "query",
        nargs="?",
        help="The question to ask (positional; required).",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (default: ./config.yaml).",
    )
    p.add_argument(
        "--db-path",
        default=None,
        help="Override the metadata DB path (default: from config.yaml).",
    )
    p.add_argument(
        "--doc-index",
        default=None,
        help="Override the doc FAISS index path (default: from config.yaml).",
    )
    p.add_argument(
        "--sensor-index",
        default=None,
        help="Override the sensor FAISS index path (default: from config.yaml).",
    )
    p.add_argument(
        "--llm",
        choices=("real", "fake"),
        default="real",
        help="Which LLMClient to use. Default: real (LlamaCppClient).",
    )
    p.add_argument(
        "--embedder",
        choices=("real", "fake"),
        default="real",
        help="Which EmbeddingModel to use. Default: real (sentence-transformers).",
    )
    p.add_argument(
        "--k-doc",
        type=int,
        default=3,
        help="How many doc-index hits to retrieve (default: 3).",
    )
    p.add_argument(
        "--k-sensor",
        type=int,
        default=2,
        help="How many sensor-index hits to retrieve (default: 2).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Minimum cosine similarity for a chunk (default: 0.3).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Cap on generated tokens per response (default: 512).",
    )
    p.add_argument(
        "--no-log",
        action="store_true",
        help="Skip writing the query_log row (for offline tests).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print JSON result instead of pretty text.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress pretty banner; print only the JSON summary.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    args = _build_parser().parse_args(argv)
    if not args.query:
        print("error: missing query (positional arg)", file=sys.stderr)
        return 2

    settings = _load_settings(args.config)
    answer = run_ask(
        query=args.query,
        settings=settings,
        llm_kind=args.llm,
        db_path_override=args.db_path,
        doc_index_override=args.doc_index,
        sensor_index_override=args.sensor_index,
        k_doc=args.k_doc,
        k_sensor=args.k_sensor,
        threshold=args.threshold,
        max_tokens=args.max_tokens,
        log_query=not args.no_log,
        embedder_kind=args.embedder,
    )

    if args.json:
        print_json(answer)
        return 0 if answer.text else 1
    print_human(answer, quiet=args.quiet)
    return 0 if answer.text else 1


if __name__ == "__main__":
    raise SystemExit(main())
