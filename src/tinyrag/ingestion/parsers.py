"""Document parsers — PDF / TXT / MD → structured text.

This module is the **first step** of the ingestion pipeline: it takes
a raw file on disk and produces a :class:`ParsedDocument` that the
chunker (:mod:`tinyrag.core.chunker`, Step 4.5) consumes.

Three concrete parsers are provided, all behind a common
:class:`DocumentParser` Protocol so the ingestion pipeline can
treat them polymorphically::

    from tinyrag.ingestion.parsers import parse
    doc = parse("data/documents/manual.pdf")
    doc.text            # full plain-text extraction (newlines between pages)
    doc.pages           # [(1, "page one text..."), (2, "..."), ...]
    doc.metadata        # {"source": "...", "format": "pdf", "page_count": 12, ...}

Format dispatch
---------------
The public entry point :func:`parse` detects the file format from the
file extension and picks the right parser. New formats are added in
one place — the ``_EXTENSION_MAP`` dict at the bottom of this file —
and automatically inherit the dispatcher behaviour.

Why a Protocol and not an ABC?
------------------------------
The architecture doc §6 ("the heart of the architecture") mandates
Protocols everywhere so future contributors can swap implementations
without inheriting from a concrete class. ``DocumentParser`` follows
the same pattern as ``SensorSource`` (§6.1) and ``EmbeddingModel``
(§6.2): a ``@runtime_checkable`` Protocol with a single
:meth:`~DocumentParser.parse` method.

Why PyMuPDF and not pdfplumber / PyPDF2?
----------------------------------------
The architecture doc §15.1 originally chose pdfplumber over PyPDF2
for layout handling. In verification (2026-07-22) we found
pdfplumber's default ``extract_text()`` still reads **two-column**
pages straight across the gutter, interleaving the columns into
scrambled sentences that embed poorly and vanish from retrieval.
PyMuPDF's ``get_text("blocks", sort=True)`` groups text into layout
blocks in natural reading order, keeping each column intact without
any per-page heuristic. For TinyRAG's ≤50 MB documents (see FR-10 in
``docs/02_srs_v1.md``) it is also faster than pdfplumber, and it
ships aarch64 wheels for the Raspberry Pi target.

Markdown handling
-----------------
``MarkdownParser`` does **not** render the markdown to HTML — the
downstream chunker cares about plain text only. It does strip a
leading YAML front-matter block (``---\\n...\\n---\\n``, common in
Docusaurus / MkDocs / Obsidian exports) so those key:value lines
don't pollute the vector store. Any ``[link text](url)`` and
``![alt](url)`` syntax is reduced to its human-readable portion —
the URL is metadata we don't want to embed.

Error handling
--------------
All parser failures raise a :class:`ParserError` subclass so the
ingestion pipeline can catch one exception type and decide whether
to retry / skip / 4xx-the-upload. We never return partial results
silently: an "empty" PDF or a malformed Markdown file raises
:class:`EmptyDocumentError` rather than handing the chunker a
zero-length string.

Public surface
--------------
- :func:`parse` — the dispatcher (call this from the pipeline).
- :class:`DocumentParser` — the Protocol.
- :class:`PdfParser` / :class:`TxtParser` / :class:`MarkdownParser`
  — concrete implementations (call directly in tests if you need
  format-pinned behaviour).
- :class:`ParsedDocument` — the result dataclass.
- :class:`ParserError` and its subclasses — typed exception hierarchy.

Location: ``src/tinyrag/ingestion/parsers.py``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# PyMuPDF (fitz) is the heaviest dependency in the ingestion path.
# We import it lazily inside PdfParser.parse() so a Markdown-only or
# TXT-only project can still import this module without pulling in
# PyMuPDF's shared library. (TXT and Markdown are pure-stdlib.)

# ----------------------------------------------------------------------------
# Public exceptions
# ----------------------------------------------------------------------------


class ParserError(Exception):
    """Base class for every parser failure.

    The ingestion pipeline catches this and translates it into an
    HTTP 4xx (or a clean CLI error) rather than a 500. Always
    subclass this rather than raising a bare ``Exception`` so
    downstream code can ``except ParserError`` once.
    """

    def __init__(self, message: str, *, path: Path | None = None) -> None:
        super().__init__(message)
        # Preserve the offending path on the exception so log lines
        # and API responses can show what went wrong without the
        # caller having to thread it through.
        self.path: Path | None = path


class UnsupportedFormatError(ParserError):
    """The file extension is not in ``_EXTENSION_MAP``.

    Raised by :func:`parse` when it sees an unknown extension. The
    pipeline should map this to HTTP 415 ("Unsupported Media Type")
    — there's no point retrying with different code.
    """


class EmptyDocumentError(ParserError):
    """The parser successfully opened the file but extracted zero text.

    This is distinct from "file not found" — the file exists and is
    readable, it's just empty (or, in the case of PDFs, contains
    only images / scanned pages with no OCR layer). Worth its own
    exception type so the CLI can print a friendly message instead
    of crashing later in the chunker with a less obvious error.
    """


class PdfReadError(ParserError):
    """PyMuPDF failed to read the PDF (corrupt, encrypted, etc.).

    Caught separately from :class:`UnsupportedFormatError` because
    the file *is* a PDF (so the dispatcher was right), but the
    underlying library rejected it. The pipeline should map this
    to HTTP 422 ("Unprocessable Entity").
    """


# ----------------------------------------------------------------------------
# Result dataclass
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedDocument:
    """The output of a successful parse.

    Attributes
    ----------
    text:
        Full plain-text extraction. For PDFs, pages are joined with
        a single newline and a form-feed (``\\f``) — this matches
        what ``pdftotext -layout`` produces and lets the chunker
        detect page boundaries if it ever needs to. For TXT/MD,
        ``text`` is just the file's contents (with markdown
        decorations stripped — see :class:`MarkdownParser`).
    pages:
        ``[(page_number_1based, page_text_1), ...]``. For PDFs,
        this preserves the per-page split that :class:`PdfParser`
        extracts (FR-2: "preserve reading order, record the page
        number for each extracted text span"). For TXT and MD,
        this is a single-element list ``[(1, text)]`` — there's
        no inherent page structure in plain text.
    metadata:
        Arbitrary dict with extra info: ``source`` (filename),
        ``format`` (one of ``"pdf"``, ``"txt"``, ``"md"``),
        ``char_count``, ``page_count`` (PDF only), and any
        format-specific extras. The pipeline forwards this to the
        chunker so each :class:`Chunk` carries the source metadata
        required by FR-5.
    """

    text: str
    pages: list[tuple[int, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------------


@runtime_checkable
class DocumentParser(Protocol):
    """Anything that can turn a file path into a :class:`ParsedDocument`.

    Concrete implementations: :class:`PdfParser`, :class:`TxtParser`,
    :class:`MarkdownParser`. The ``@runtime_checkable`` decorator lets
    tests and callers verify duck-typing (``isinstance(p, DocumentParser)``)
    without requiring inheritance — see the architecture doc §6
    "Protocol over ABC" rationale.
    """

    def parse(self, path: Path) -> ParsedDocument:
        """Parse the file at ``path`` and return a :class:`ParsedDocument`.

        Raises
        ------
        ParserError
            Subclass indicating the specific failure (file not
            found, empty, corrupt, unsupported). Never returns
            partial results.
        """
        ...


# ----------------------------------------------------------------------------
# PDF parser
# ----------------------------------------------------------------------------


class PdfParser:
    """Extracts text from a PDF using ``PyMuPDF`` (``fitz``).

    Per FR-2 we preserve reading order and the page number for each
    text span. We do **not** attempt OCR — a scanned PDF with no
    text layer will return an empty document and raise
    :class:`EmptyDocumentError` (the user can OCR it themselves and
    re-upload).

    Why PyMuPDF and not pdfplumber's ``extract_text()``?
    ----------------------------------------------------
    pdfplumber's default ``extract_text()`` walks the page's
    character stream in raw storage order and joins by (y, x). On a
    **two-column** layout (like the Nest install guide) that reads
    straight across the gutter, interleaving the left and right
    columns into scrambled sentences — e.g. "compatible with almost
    all central *You don't need Wi-Fi to use the* heating systems".
    The scrambled text then embeds poorly and the correct chunk
    drops out of retrieval entirely (verified 2026-07-22: the
    "combi boiler compatibility" chunk ranked 29/39 and the query
    falsely returned "I don't have enough information").

    PyMuPDF's ``get_text("blocks", sort=True)`` groups text into
    layout blocks and sorts them in natural reading order, keeping
    each column's sentences intact. This is column-aware without any
    per-page gutter heuristic, so it also handles PDFs we've never
    seen. See the VERIFICATION_ROADMAP finding for VA-series.

    PyMuPDF is imported lazily so a project that only ingests
    Markdown/TXT doesn't pay its import cost.
    """

    #: If the whole document yields less than this many
    #: non-whitespace characters, we treat it as having "no
    #: extractable text" (e.g. a scanned PDF with no OCR layer)
    #: rather than passing an effectively-empty doc downstream.
    _MIN_TOTAL_CHARS = 10

    def parse(self, path: Path) -> ParsedDocument:
        """Parse the PDF at ``path``.

        Raises
        ------
        ParserError
            File missing → :class:`ParserError`.
            PyMuPDF rejected it → :class:`PdfReadError`.
            No extractable text → :class:`EmptyDocumentError`.
        """
        if not path.is_file():
            raise ParserError(f"PDF file not found: {path}", path=path)

        try:
            import fitz  # PyMuPDF — lazy import, see class docstring
        except ImportError as exc:
            raise ParserError(
                "PyMuPDF is not installed; run `pip install PyMuPDF`",
                path=path,
            ) from exc

        try:
            with fitz.open(str(path)) as doc:
                pages: list[tuple[int, str]] = []
                for one_based_idx, page in enumerate(doc, start=1):
                    # ``blocks`` returns layout blocks as tuples
                    # ``(x0, y0, x1, y1, text, block_no, block_type)``.
                    # ``sort=True`` orders them in natural reading
                    # order (top-to-bottom, respecting columns), which
                    # keeps each column's sentences intact instead of
                    # reading straight across a two-column gutter.
                    blocks = page.get_text("blocks", sort=True)
                    page_text = "\n".join(
                        b[4].strip() for b in blocks if b[4].strip()
                    )
                    pages.append((one_based_idx, page_text))
        except Exception as exc:  # PyMuPDF raises fitz.FileDataError etc.
            raise PdfReadError(
                f"PyMuPDF failed to read {path}: {exc}", path=path
            ) from exc

        # Join pages with form-feed so a chunker that wants to know
        # where pages begin can scan for "\f".
        text = "\f".join(page_text for _, page_text in pages)

        if len(text.strip()) < self._MIN_TOTAL_CHARS:
            raise EmptyDocumentError(
                f"PDF {path.name} contains no extractable text "
                f"(likely a scanned document without OCR). "
                f"Run OCR on it first or convert to TXT/MD.",
                path=path,
            )

        return ParsedDocument(
            text=text,
            pages=pages,
            metadata={
                "source": path.name,
                "format": "pdf",
                "char_count": len(text),
                "page_count": len(pages),
            },
        )


# ----------------------------------------------------------------------------
# TXT parser
# ----------------------------------------------------------------------------


class TxtParser:
    """Reads a plain UTF-8 text file.

    Encoding is detected automatically — we honour a BOM if present
    (``utf-8-sig``) and fall back to ``utf-8``. Latin-1 / Windows-1252
    files are *not* auto-detected; the pipeline should fail loudly
    so the user knows to re-save the file as UTF-8. (Auto-detecting
    encoding would silently misinterpret some bytes and we'd never
    know.)
    """

    def parse(self, path: Path) -> ParsedDocument:
        """Read the file at ``path`` as UTF-8 text.

        Raises
        ------
        ParserError
            File missing.
        UnicodeDecodeError
            File isn't valid UTF-8 (propagates unchanged so the
            caller can surface "wrong encoding" to the user).
        EmptyDocumentError
            File is empty / whitespace-only.
        """
        if not path.is_file():
            raise ParserError(f"TXT file not found: {path}", path=path)

        text = path.read_text(encoding="utf-8-sig")
        if not text.strip():
            raise EmptyDocumentError(
                f"TXT file {path.name} is empty", path=path
            )

        return ParsedDocument(
            text=text,
            pages=[(1, text)],
            metadata={
                "source": path.name,
                "format": "txt",
                "char_count": len(text),
                "page_count": 1,
            },
        )


# ----------------------------------------------------------------------------
# Markdown parser
# ----------------------------------------------------------------------------


# A YAML front-matter block at the top of a Markdown file:
#
#     ---
#     title: My Manual
#     author: ACME
#     ---
#     # Heading
#     body...
#
# We strip everything from the opening "---" up to and including
# the closing "---" line, plus any blank lines immediately after.
_FRONTMATTER_RE = re.compile(
    r"\A\s*---\s*\n.*?\n---\s*\n+",
    re.DOTALL,
)

# Reduce Markdown link / image syntax to its human-readable portion:
#   [text](url)        -> text
#   ![alt](url)        -> alt
# We run image syntax first so the "!" doesn't trip up the link rule.
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


class MarkdownParser:
    """Reads a Markdown file and returns its plain-text body.

    We strip two things that would otherwise pollute the vector store:

    1. **YAML front-matter** (the ``---...---`` block at the top).
       Common in Docusaurus, MkDocs, Obsidian, and Hugo exports.
       Without stripping, the chunker sees ``title: My Manual`` and
       treats it as body text.

    2. **URL noise from links / images**. ``[link text](https://...)``
       becomes just ``link text``; ``![alt](https://...)`` becomes
       ``alt``. URLs aren't useful for retrieval — the question
       "what does the manual say about pairing?" doesn't benefit
       from seeing a ``https://example.com/support`` next to every
       reference.

    We **don't** strip other Markdown syntax (headings, bold, code
    fences, tables). The chunker and embedder are both robust to
    them, and stripping would lose information (``**Warning:**`` is
    semantically different from ``Warning:``). HTML embedded in the
    Markdown is left as-is for the same reason.
    """

    def parse(self, path: Path) -> ParsedDocument:
        """Read the Markdown file at ``path``.

        Raises
        ------
        ParserError
            File missing.
        UnicodeDecodeError
            File isn't valid UTF-8.
        EmptyDocumentError
            File is empty / whitespace-only after stripping.
        """
        if not path.is_file():
            raise ParserError(f"Markdown file not found: {path}", path=path)

        raw = path.read_text(encoding="utf-8-sig")

        # Strip front-matter, then images, then links. Order matters:
        # images first because ``![alt](url)`` would otherwise be
        # matched by the link regex as ``[alt](url)`` with the
        # leading "!" stranded.
        text = _FRONTMATTER_RE.sub("", raw)
        text = _IMAGE_RE.sub(r"\1", text)
        text = _LINK_RE.sub(r"\1", text)

        if not text.strip():
            raise EmptyDocumentError(
                f"Markdown file {path.name} is empty (or only contained "
                f"a front-matter block)",
                path=path,
            )

        return ParsedDocument(
            text=text,
            pages=[(1, text)],
            metadata={
                "source": path.name,
                "format": "md",
                "char_count": len(text),
                "page_count": 1,
                # Useful for debugging: was the front-matter actually
                # stripped, or was this file already plain Markdown?
                "had_frontmatter": bool(_FRONTMATTER_RE.search(raw)),
            },
        )


# ----------------------------------------------------------------------------
# Format dispatch
# ----------------------------------------------------------------------------


# Maps a lowercase extension (with leading dot) to the parser class
# that handles it. Adding a new format = add one line here.
_EXTENSION_MAP: dict[str, type[DocumentParser]] = {
    ".pdf": PdfParser,
    ".txt": TxtParser,
    ".md": MarkdownParser,
    # Common alias. Most tools treat ".markdown" identically to ".md".
    ".markdown": MarkdownParser,
}


# A single shared parser instance per format — parsers are stateless,
# so we don't need to construct one per call. ``functools.cache`` would
# also work; a module-level dict is simpler and the parsers are
# tiny, so the small duplication across calls is fine.
_PARSER_CACHE: dict[str, DocumentParser] = {
    ext: cls() for ext, cls in _EXTENSION_MAP.items()
}


def parse(path: Path | str) -> ParsedDocument:
    """Parse ``path`` using the parser for its file extension.

    This is the public entry point — the ingestion pipeline calls
    this, not the concrete parsers. The pipeline should not need to
    know about formats.

    Parameters
    ----------
    path:
        Path to the file. Can be a string (we coerce to ``Path``)
        or a :class:`pathlib.Path`. The extension lookup is
        case-insensitive (``manual.PDF`` and ``manual.pdf`` both
        work) — useful because some cameras / scanners write
        uppercase extensions.

    Returns
    -------
    ParsedDocument
        See :class:`ParsedDocument` for the shape.

    Raises
    ------
    UnsupportedFormatError
        ``path``'s extension isn't in :data:`_EXTENSION_MAP`.
    ParserError
        Subclass raised by the underlying parser.
    """
    p = Path(path)
    # ``.suffix`` preserves the original case; lowercase for lookup.
    ext = p.suffix.lower()
    parser = _PARSER_CACHE.get(ext)
    if parser is None:
        supported = ", ".join(sorted(_EXTENSION_MAP.keys()))
        raise UnsupportedFormatError(
            f"unsupported file extension {ext!r} for {p.name} "
            f"(supported: {supported})",
            path=p,
        )
    return parser.parse(p)
