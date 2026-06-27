#!/usr/bin/env python3
"""End-to-end ingestion pipeline — the Step 4.9 risk gate.

This is the **single-script Phase 4 checkpoint** from
``docs/06_roadmap_v2.md`` Step 4.9. It does five things, in order:

1. ``parse(path)`` — turn the file into a :class:`ParsedDocument`
   (PDF/TXT/MD supported, dispatched by extension).
2. ``chunker.chunk(text, source, page=...)`` — split per-page text
   into embedding-ready ~400-token chunks with overlap.
3. ``embedder.embed([c.text for c in chunks])`` — turn chunks into
   dense 384-dim L2-normalised vectors.
4. ``metadata.insert_document(...)`` + ``metadata.insert_chunks(...)``
   — persist the document row + every chunk row (atomic batch).
5. ``vector_store.add(vectors, ids)`` + ``vector_store.save()`` —
   persist the FAISS index + sidecar JSON.

At the end it prints an :class:`IngestionReport` summarising every
stage's timing and the final chunk count.

CLI flags
---------

    --path PATH              File to ingest (positional; required).
    --config PATH            Path to config.yaml (default: ./config.yaml).
    --db-path PATH           Override metadata DB path.
    --index-path PATH        Override FAISS index path.
    --doc-type {manual|note|spec}
                             Document type for the metadata store.
    --embedder {real|fake}   Which EmbeddingModel to use. Default: real
                             (sentence-transformers all-MiniLM-L6-v2).
                             "fake" uses FakeEmbedder — hermetic, no
                             model download required; great for CI.
    --json                   Print JSON result instead of pretty text.
    --quiet                  Suppress pretty banner; print only the
                             JSON summary on success, error on failure.

Exit codes
----------

    0   Ingestion succeeded; chunks are now searchable.
    1   Pipeline error (parser / embedder / store failure).
    2   Bad CLI args (argparse handles this with code 2).
    3   Input file doesn't exist or is unreadable.

Companion docs
--------------
- ``src/tinyrag/ingestion/parsers.py`` — parse()
- ``src/tinyrag/core/chunker.py`` — Chunker.chunk()
- ``src/tinyrag/ingestion/embedder.py`` — SentenceTransformerEmbedder
- ``src/tinyrag/storage/metadata.py`` — MetadataStore
- ``src/tinyrag/storage/vector_store.py`` — FAISSStore
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
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
from tinyrag.core import Chunk, Chunker  # noqa: E402
from tinyrag.ingestion import (  # noqa: E402
    EmbeddingError,
    FakeEmbedder,
    ParsedDocument,
    SentenceTransformerEmbedder,
    parse,
)
from tinyrag.storage import FAISSStore, MetadataStore  # noqa: E402

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class IngestionReport:
    """Outcome of one ingest run. JSON-serialisable.

    All ``duration_ms`` fields are wall-clock durations of each
    pipeline stage — useful for diagnosing slow stages without
    re-running the script with profiling.
    """

    ok: bool
    file: str
    doc_id: str | None
    num_pages: int
    num_chunks: int
    embedding_dimension: int
    embedding_model: str
    doc_type: str
    db_path: str
    index_path: str
    index_size: int
    duration_parse_ms: float
    duration_chunk_ms: float
    duration_embed_ms: float
    duration_metadata_ms: float
    duration_vector_ms: float
    duration_save_ms: float
    duration_total_ms: float
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (rounds floats to 2 dp)."""
        return {
            "ok": self.ok,
            "file": self.file,
            "doc_id": self.doc_id,
            "num_pages": self.num_pages,
            "num_chunks": self.num_chunks,
            "embedding_dimension": self.embedding_dimension,
            "embedding_model": self.embedding_model,
            "doc_type": self.doc_type,
            "db_path": self.db_path,
            "index_path": self.index_path,
            "index_size": self.index_size,
            "duration_parse_ms": round(self.duration_parse_ms, 2),
            "duration_chunk_ms": round(self.duration_chunk_ms, 2),
            "duration_embed_ms": round(self.duration_embed_ms, 2),
            "duration_metadata_ms": round(self.duration_metadata_ms, 2),
            "duration_vector_ms": round(self.duration_vector_ms, 2),
            "duration_save_ms": round(self.duration_save_ms, 2),
            "duration_total_ms": round(self.duration_total_ms, 2),
            "error": self.error,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Stage-level helpers (each is independently testable)
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes (used as content_hash).

    Reading in 64 KB chunks so a 1 GB file doesn't blow the heap.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_chunker(settings: Settings) -> Chunker:
    """Build a :class:`Chunker` from the typed Settings (single source)."""
    # Lazy import to avoid pulling tiktoken at module load.
    from tinyrag.core.chunker import ChunkingSettings

    return Chunker(
        ChunkingSettings(
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
            encoding=settings.chunking.encoding,
        )
    )


def _make_embedder(settings: Settings, *, kind: str):
    """Build an :class:`EmbeddingModel` (real or fake) per the --embedder flag.

    The real model is lazy-loaded on first ``.embed()`` call (see
    Step 4.6) so this constructor is cheap.
    """
    if kind == "fake":
        # FakeEmbedder is deterministic + model-free. The default
        # dimension 384 matches all-MiniLM-L6-v2 (the default real
        # model in config.yaml) so the FAISS index can be built
        # identically regardless of which path you take.
        return FakeEmbedder(dimension=384)
    if kind == "real":
        return SentenceTransformerEmbedder(
            model_name=settings.embedding.model_name,
            device=settings.embedding.device.value,
            batch_size=settings.embedding.batch_size,
        )
    raise ValueError(f"unknown embedder kind: {kind!r}")  # pragma: no cover


def _chunk_pages(parsed: ParsedDocument, source: str, chunker: Chunker) -> list[Chunk]:
    """Chunk a parsed document page-by-page, preserving the page number.

    Per the §4.5 invariant: ``chunk.page`` must be ``None`` for TXT/MD
    (no page concept) and ``1..N`` for PDF (pdfplumber is 1-indexed).
    """
    chunks: list[Chunk] = []
    if not parsed.pages:
        # Plain-text formats (TXT, MD) collapse to a single page-less
        # chunk. The chunker treats ``page=None`` correctly.
        chunks.extend(chunker.chunk(parsed.text, source=source, page=None))
        return chunks
    for page_num, page_text in parsed.pages:
        chunks.extend(chunker.chunk(page_text, source=source, page=page_num))
    return chunks


def _chunk_records(
    chunks: list[Chunk],
    document_id: str,
    embedding_model: str,
) -> list[dict[str, Any]]:
    """Map :class:`Chunk` dataclasses → dicts for :meth:`MetadataStore.insert_chunks`.

    Generates UUID v4 per chunk (the metadata store accepts both
    explicit and auto-UUID; we generate here so the same UUID can be
    passed to ``FAISSStore.add(vectors, ids)`` for the int↔UUID
    lock-step the architecture requires).

    NOTE on ``chunk_index``: the Chunker produces page-scoped indices
    (each call to ``chunker.chunk(page_text, ...)`` resets the counter
    to 0). The metadata store's schema requires
    ``UNIQUE (document_id, chunk_index)`` — a document-scoped
    constraint. So we renumber globally across pages here, which is
    what the FAISS ``add_with_ids`` order also assumes (the i-th
    record's UUID maps to the i-th FAISS slot).

    The original ``char_offset`` is preserved — it's a per-page offset
    within the page text, not a document offset, so the renumbering
    doesn't affect it.
    """
    import uuid as _uuid

    records: list[dict[str, Any]] = []
    for global_index, c in enumerate(chunks):
        records.append(
            {
                "id": str(_uuid.uuid4()),
                "document_id": document_id,
                "chunk_index": global_index,
                # ``faiss_idx`` is the INT index in the FAISS index —
                # assigned by FAISSStore.add() in ingestion order
                # (see Step 4.8). We patch it back into the chunk
                # row AFTER FAISS has assigned the int IDs so the
                # DB and the index agree. Placeholder -1 for now.
                "faiss_idx": -1,
                "text": c.text,
                "page_number": c.page,
                "char_offset": c.char_offset,
                "token_count": c.token_count,
                "embedding_model": embedding_model,
            }
        )
    return records


# ---------------------------------------------------------------------------
# The ingest run
# ---------------------------------------------------------------------------


def run_ingest(
    *,
    path: Path,
    settings: Settings,
    doc_type: str,
    embedder_kind: str,
    db_path_override: str | None,
    index_path_override: str | None,
) -> IngestionReport:
    """Run the full pipeline and return an :class:`IngestionReport`.

    Every stage's exception is caught and re-packaged as a failed
    report — the script should never crash with a traceback (the
    caller wants a clean exit code, not a Python stack trace).
    """
    if not path.exists():
        return _failed_report(
            file=str(path),
            settings=settings,
            db_path_override=db_path_override,
            index_path_override=index_path_override,
            doc_type=doc_type,
            error=f"file not found: {path}",
        )

    db_path = Path(db_path_override) if db_path_override else Path(settings.paths.metadata_db)
    index_path = (
        Path(index_path_override)
        if index_path_override
        else Path(settings.retrieval.doc_index_path)
    )

    timings: dict[str, float] = {}
    extra: dict[str, Any] = {}
    t_total_start = time.monotonic()

    # ---- Stage 1: parse ---------------------------------------------------
    t = time.monotonic()
    try:
        parsed = parse(path)
    except Exception as exc:
        return _failed_report(
            file=str(path),
            settings=settings,
            db_path_override=db_path_override,
            index_path_override=index_path_override,
            doc_type=doc_type,
            error=f"parse failed: {type(exc).__name__}: {exc}",
        )
    timings["parse_ms"] = (time.monotonic() - t) * 1000.0

    # ---- Stage 2: chunk ---------------------------------------------------
    t = time.monotonic()
    try:
        chunker = _make_chunker(settings)
        chunks = _chunk_pages(parsed, source=path.name, chunker=chunker)
    except Exception as exc:
        return _failed_report(
            file=str(path),
            settings=settings,
            db_path_override=db_path_override,
            index_path_override=index_path_override,
            doc_type=doc_type,
            error=f"chunk failed: {type(exc).__name__}: {exc}",
        )
    timings["chunk_ms"] = (time.monotonic() - t) * 1000.0

    if not chunks:
        return _failed_report(
            file=str(path),
            settings=settings,
            db_path_override=db_path_override,
            index_path_override=index_path_override,
            doc_type=doc_type,
            error="no chunks produced (file may be empty or non-text)",
        )

    # ---- Stage 3: embed ---------------------------------------------------
    embedder = _make_embedder(settings, kind=embedder_kind)
    embedding_dimension = embedder.dimension
    embedding_model_name = (
        settings.embedding.model_name
        if embedder_kind == "real"
        else f"fake:{settings.embedding.model_name}"
    )
    t = time.monotonic()
    try:
        texts = [c.text for c in chunks]
        vectors = embedder.embed(texts)
    except EmbeddingError as exc:
        return _failed_report(
            file=str(path),
            settings=settings,
            db_path_override=db_path_override,
            index_path_override=index_path_override,
            doc_type=doc_type,
            error=f"embed failed: {exc}",
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model_name,
            num_pages=len(parsed.pages) if parsed.pages else 1,
            **timings,
        )
    except Exception as exc:
        return _failed_report(
            file=str(path),
            settings=settings,
            db_path_override=db_path_override,
            index_path_override=index_path_override,
            doc_type=doc_type,
            error=f"embed failed: {type(exc).__name__}: {exc}",
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model_name,
            num_pages=len(parsed.pages) if parsed.pages else 1,
            **timings,
        )
    timings["embed_ms"] = (time.monotonic() - t) * 1000.0

    if len(vectors) != len(chunks):
        return _failed_report(
            file=str(path),
            settings=settings,
            db_path_override=db_path_override,
            index_path_override=index_path_override,
            doc_type=doc_type,
            error=(
                f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            ),
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model_name,
            num_pages=len(parsed.pages) if parsed.pages else 1,
            **timings,
        )

    # ---- Stage 4: metadata DB --------------------------------------------
    t = time.monotonic()
    try:
        store = MetadataStore(db_path)
        store.init_schema()

        # SHA-256 of the raw bytes is our ``content_hash`` — the
        # dedup signal for re-ingestion (returns the OLDEST match
        # per the Step 4.7 contract).
        content_hash = _sha256_file(path)
        size_bytes = path.stat().st_size

        doc_id = store.insert_document(
            filename=path.name,
            doc_type=doc_type,
            source_path=str(path),
            size_bytes=size_bytes,
            content_hash=content_hash,
            metadata={
                "num_pages": len(parsed.pages) if parsed.pages else 1,
                "num_chars": len(parsed.text),
                "ingested_via": "scripts/ingest.py",
            },
        )

        # Build chunk records with UUIDs — the SAME UUIDs are
        # passed to FAISSStore.add() so the int↔UUID mapping is
        # consistent across both stores.
        chunk_records = _chunk_records(chunks, document_id=doc_id, embedding_model=embedding_model_name)
        chunk_ids = store.insert_chunks(chunk_records)
    except Exception as exc:
        return _failed_report(
            file=str(path),
            settings=settings,
            db_path_override=db_path_override,
            index_path_override=index_path_override,
            doc_type=doc_type,
            error=f"metadata failed: {type(exc).__name__}: {exc}",
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model_name,
            num_pages=len(parsed.pages) if parsed.pages else 1,
            **timings,
        )
    timings["metadata_ms"] = (time.monotonic() - t) * 1000.0

    # ---- Stage 5: FAISS index --------------------------------------------
    t = time.monotonic()
    try:
        faiss_store = FAISSStore(
            index_path,
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model_name,
        )
        faiss_store.load()  # no-op if file doesn't exist; loads if it does
        # FAISSStore.add() assigns sequential int IDs starting from
        # current size — the int↔UUID mapping is recorded in the
        # sidecar JSON. ``chunk_ids`` (UUIDs from metadata) is
        # the same order as ``chunks``, so FAISS int ID i maps
        # to chunk_ids[i].
        faiss_store.add(vectors, chunk_ids)
    except Exception as exc:
        return _failed_report(
            file=str(path),
            settings=settings,
            db_path_override=db_path_override,
            index_path_override=index_path_override,
            doc_type=doc_type,
            error=f"vector store failed: {type(exc).__name__}: {exc}",
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model_name,
            num_pages=len(parsed.pages) if parsed.pages else 1,
            doc_id=doc_id,
            num_chunks=len(chunks),
            **timings,
        )
    timings["vector_ms"] = (time.monotonic() - t) * 1000.0

    # ---- Stage 6: persist FAISS ------------------------------------------
    t = time.monotonic()
    try:
        faiss_store.save()
    except Exception as exc:
        return _failed_report(
            file=str(path),
            settings=settings,
            db_path_override=db_path_override,
            index_path_override=index_path_override,
            doc_type=doc_type,
            error=f"save failed: {type(exc).__name__}: {exc}",
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model_name,
            num_pages=len(parsed.pages) if parsed.pages else 1,
            doc_id=doc_id,
            num_chunks=len(chunks),
            **timings,
        )
    timings["save_ms"] = (time.monotonic() - t) * 1000.0

    # ---- Stage 7: update document row with chunk count -------------------
    t = time.monotonic()
    try:
        store.update_document_chunk_count(doc_id, len(chunks))
    except Exception as exc:
        # Non-fatal — the chunks are in the DB; only the
        # ``documents.num_chunks`` counter is wrong. Surface as a
        # warning but still mark ok=True (the chunks ARE searchable).
        extra["warning"] = f"update_document_chunk_count failed: {exc}"
    timings["update_ms"] = (time.monotonic() - t) * 1000.0

    timings["total_ms"] = (time.monotonic() - t_total_start) * 1000.0

    return IngestionReport(
        ok=True,
        file=str(path),
        doc_id=doc_id,
        num_pages=len(parsed.pages) if parsed.pages else 1,
        num_chunks=len(chunks),
        embedding_dimension=embedding_dimension,
        embedding_model=embedding_model_name,
        doc_type=doc_type,
        db_path=str(db_path),
        index_path=str(index_path),
        index_size=faiss_store.size(),
        duration_parse_ms=timings.get("parse_ms", 0.0),
        duration_chunk_ms=timings.get("chunk_ms", 0.0),
        duration_embed_ms=timings.get("embed_ms", 0.0),
        duration_metadata_ms=timings.get("metadata_ms", 0.0),
        duration_vector_ms=timings.get("vector_ms", 0.0),
        duration_save_ms=timings.get("save_ms", 0.0),
        duration_total_ms=timings.get("total_ms", 0.0),
        error=None,
        extra=extra,
    )


def _failed_report(
    *,
    file: str,
    settings: Settings,
    db_path_override: str | None,
    index_path_override: str | None,
    doc_type: str,
    error: str,
    embedding_dimension: int = 0,
    embedding_model: str = "",
    num_pages: int = 0,
    doc_id: str | None = None,
    num_chunks: int = 0,
    **timings: float,
) -> IngestionReport:
    """Construct an IngestionReport with ok=False and the given error."""
    db_path = Path(db_path_override) if db_path_override else Path(settings.paths.metadata_db)
    index_path = (
        Path(index_path_override)
        if index_path_override
        else Path(settings.retrieval.doc_index_path)
    )
    return IngestionReport(
        ok=False,
        file=file,
        doc_id=doc_id,
        num_pages=num_pages,
        num_chunks=num_chunks,
        embedding_dimension=embedding_dimension,
        embedding_model=embedding_model,
        doc_type=doc_type,
        db_path=str(db_path),
        index_path=str(index_path),
        index_size=0,
        duration_parse_ms=timings.get("parse_ms", 0.0),
        duration_chunk_ms=timings.get("chunk_ms", 0.0),
        duration_embed_ms=timings.get("embed_ms", 0.0),
        duration_metadata_ms=timings.get("metadata_ms", 0.0),
        duration_vector_ms=timings.get("vector_ms", 0.0),
        duration_save_ms=timings.get("save_ms", 0.0),
        duration_total_ms=timings.get("total_ms", 0.0),
        error=error,
    )


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


def print_human(report: IngestionReport, *, quiet: bool) -> None:
    """Print a friendly summary to stdout."""
    if quiet:
        if report.ok:
            print(json.dumps(report.to_dict()))
        else:
            print(f"ERROR: {report.error}", file=sys.stderr)
        return

    print(_bold("==> TinyRAG — Ingestion Report"))
    print(f"    file:                {report.file}")
    print(f"    doc_type:            {report.doc_type}")
    print(f"    doc_id:              {report.doc_id}")
    print(f"    num_pages:           {report.num_pages}")
    print(f"    num_chunks:          {report.num_chunks}")
    print(f"    embedding_model:     {report.embedding_model}")
    print(f"    embedding_dimension: {report.embedding_dimension}")
    print(f"    db_path:             {report.db_path}")
    print(f"    index_path:          {report.index_path}")
    print(f"    index_size:          {report.index_size}")
    print()
    print("    timings:")
    print(f"      parse:     {report.duration_parse_ms:>8.2f} ms")
    print(f"      chunk:     {report.duration_chunk_ms:>8.2f} ms")
    print(f"      embed:     {report.duration_embed_ms:>8.2f} ms")
    print(f"      metadata:  {report.duration_metadata_ms:>8.2f} ms")
    print(f"      vector:    {report.duration_vector_ms:>8.2f} ms")
    print(f"      save:      {report.duration_save_ms:>8.2f} ms")
    print(f"      TOTAL:     {report.duration_total_ms:>8.2f} ms")
    print()
    if report.ok:
        print(_green("[ OK ]") + " Ingestion succeeded — chunks are now searchable.")
        if report.extra.get("warning"):
            print(_c("33", f"      warning: {report.extra['warning']}"))
    else:
        print(_red("[FAIL]") + f" {report.error}")
        sys.exit(1)


def print_json(report: IngestionReport) -> None:
    """Print the result as a single JSON object."""
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Settings helper — load with optional override of config path
# ---------------------------------------------------------------------------


def _load_settings(config_path: str | None) -> Settings:
    """Load the typed :class:`Settings`, optionally from a custom config."""
    if config_path is None:
        return load_settings()
    # If the user passed an explicit path, construct a fresh loader.
    # (load_settings() reads from the project root's config.yaml; the
    # override is for tests + power users.) ``load_settings`` takes
    # a positional ``path`` argument — the legacy ``config_path=``
    # kwarg is no longer recognised.
    from tinyrag.config import load_settings as _ls

    return _ls(config_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ingest.py",
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "path",
        nargs="?",
        help="File to ingest (PDF/TXT/MD).",
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
        "--index-path",
        default=None,
        help="Override the FAISS index path (default: from config.yaml).",
    )
    p.add_argument(
        "--doc-type",
        default="manual",
        choices=("manual", "note", "spec"),
        help="Document type for the metadata store. Default: manual.",
    )
    p.add_argument(
        "--embedder",
        choices=("real", "fake"),
        default="real",
        help="Which EmbeddingModel to use. Default: real (sentence-transformers).",
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
    if not args.path:
        print("error: missing file path (positional arg)", file=sys.stderr)
        return 2

    settings = _load_settings(args.config)
    report = run_ingest(
        path=Path(args.path),
        settings=settings,
        doc_type=args.doc_type,
        embedder_kind=args.embedder,
        db_path_override=args.db_path,
        index_path_override=args.index_path,
    )

    if args.json:
        print_json(report)
        return 0 if report.ok else 1
    print_human(report, quiet=args.quiet)
    # print_human exits with code 1 on failure; success returns 0.
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
