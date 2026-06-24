"""Token-based text chunking with sentence-boundary respect.

This module is the **bridge** between the parsers (Step 4.4, which
turn a file into a :class:`tinyrag.ingestion.parsers.ParsedDocument`)
and the embedder (Step 4.6, which turns text into vectors). It
splits a long document into smaller, overlapping chunks sized for
the embedding model's input window.

Why token-based and not character-based?
----------------------------------------
Embedding models (sentence-transformers in Step 4.6) operate on
**tokens**, not characters. A naive 1500-character chunk could be
300 short tokens or 600 long tokens depending on the text — the
embedding model would silently truncate the long one, losing
information at the tail. Counting tokens with ``tiktoken`` keeps
the chunk size predictable across inputs.

The 400-token default is sized so 3-4 chunks fit in Phi-3 Mini's
4k context window alongside the system prompt and the user's
question. The 50-token overlap (~12%) prevents sentences from
being split across chunk boundaries — if a sentence spans the
end of chunk 1 and the start of chunk 2, retrieval still has a
complete copy in chunk 1.

Algorithm
---------
1. **Encode** the full text to a list of token IDs.
2. **Window** the token list: emit a chunk of up to ``chunk_size``
   tokens, then advance the window by ``chunk_size - chunk_overlap``
   tokens.
3. **Sentence trim**: before emitting each chunk, look back from
   the right edge for a sentence terminator (``.``, ``!``, ``?``
   followed by whitespace or end-of-text) within the last 20% of
   the window. If found, trim the chunk to end there. This
   prevents a chunk from ending mid-sentence, which hurts both
   retrieval (the chunk text looks incomplete to the embedder) and
   human readability (when a user clicks a citation).
4. **Decode** the trimmed token range back to a string and record
   its ``char_offset`` in the original text plus the
   ``chunk_index`` ordinal.

If ``chunk_overlap == 0`` (rare, but the config allows it), the
window slides with no backstep — every chunk is disjoint.

Why pure functions / no I/O?
----------------------------
The :mod:`tinyrag.core` package is the *domain logic* of TinyRAG.
It is the only layer with no I/O dependencies — see
:mod:`tinyrag.core`'s docstring for the one-way dependency rule.
A future "use a different chunker" change is a one-class swap in
the composition root (``main.py``, Step 4.17), not a refactor
across the codebase.

Why ``tiktoken`` and not a HuggingFace tokenizer?
-------------------------------------------------
``tiktoken`` is fast, pure-Python (no PyTorch at import time), and
uses the same BPE encoding as OpenAI's models, so chunk sizes
match what those models would see. sentence-transformers' own
tokenizer is faster for batched embedding but has a non-trivial
import cost — using tiktoken for *counting* and letting the
embedder use its own tokenizer for *encoding* keeps the chunker
independent of any specific embedding model.

Public surface
--------------
- :class:`Chunk` — frozen dataclass; the result of one chunk.
- :class:`Chunker` — the chunker itself.
- :class:`ChunkingError` — typed exception for bad config (e.g.
  a tiktoken encoding that doesn't exist).

Location: ``src/tinyrag/core/chunker.py``
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from tinyrag.config import ChunkingSettings

# ----------------------------------------------------------------------------
# Public exceptions
# ----------------------------------------------------------------------------


class ChunkingError(Exception):
    """Raised when the chunker cannot run (bad encoding, bad config).

    The pipeline catches this and translates it to a clean error;
    we never let a ``tiktoken`` ``KeyError`` leak out.
    """


# ----------------------------------------------------------------------------
# Result dataclass
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One chunk of a document.

    Attributes
    ----------
    text:
        The chunk's text. Decoded from the token range; may differ
        from the source string in whitespace at the boundaries
        (tiktoken's BPE is byte-level and may produce leading/
        trailing spaces that aren't in the source).
    source:
        The source filename or document identifier — usually the
        ``metadata['source']`` from the :class:`ParsedDocument` that
        produced this chunk. Required by FR-5.
    page:
        1-based page number (for PDFs) or ``None`` for plain
        text / Markdown. Required by FR-5.
    chunk_index:
        0-based ordinal of this chunk within its document. Two
        chunks from the same document are uniquely identified by
        ``(source, chunk_index)``. Required by FR-5.
    char_offset:
        The character position of the first character of
        ``text`` in the *original* document string. Required by
        FR-5 ("character offset in source"). For PDFs this refers
        to the position in the concatenated per-page text (with
        the form-feed separators ``PdfParser`` produces).
    token_count:
        The number of tokens in ``text`` *as the chunker counted
        them* (i.e. ``len(encoded_chunk_text)``). Cached on the
        chunk so the embedder can sanity-check sizes without
        re-encoding.
    """

    text: str
    source: str
    page: int | None
    chunk_index: int
    char_offset: int
    token_count: int


# ----------------------------------------------------------------------------
# The chunker
# ----------------------------------------------------------------------------


# How far back from a chunk's right edge we'll look for a sentence
# terminator. 20% of ``chunk_size`` keeps the trim local — looking
# back further would shave too much off the chunk. If no
# terminator is found in this window, the chunk is emitted at its
# natural ``chunk_size`` boundary (better an awkward edge than a
# tiny chunk).
_SENTENCE_TRIM_FRACTION = 0.20

# Characters that end a sentence. We don't split on every period
# (e.g. "Dr. Smith", "3.14") — we only trim when the period is
# followed by whitespace or end-of-text, which is a reasonable
# approximation. The same pattern is used by langchain's
# ``SentenceTransformersTokenTextSplitter`` and others.
#
# Note: the regex is a "raw" string (``r"..."``); the inner ``\s``
# is a regex escape, not a Python string escape.
#
# The trailing ``(?=[\s"']|$)`` is a **positive lookahead**: it
# *asserts* that the next character is whitespace, a quote, or
# end-of-text, but doesn't *consume* it. That way the match end
# is right after the punctuation — exactly where we want to trim.
_SENTENCE_END_RE = re.compile(r"[.!?](?=[\s\"']|$)")


class Chunker:
    """Token-based text chunker with sentence-boundary respect.

    Construct with a :class:`ChunkingSettings` (from
    :mod:`tinyrag.config`); the constructor looks up the
    ``tiktoken`` encoding once and caches the encoder on the
    instance. Multiple calls to :meth:`chunk` reuse the same
    encoder — there's no per-call setup cost.

    Parameters
    ----------
    settings:
        The :class:`ChunkingSettings` to use. Must satisfy
        ``chunk_overlap < chunk_size`` (enforced by the Pydantic
        validator on :class:`ChunkingSettings` — see
        :mod:`tinyrag.config`).
    encoding:
        Optional pre-loaded ``tiktoken.Encoding`` to use. Defaults
        to ``tiktoken.get_encoding(settings.encoding)``. Pass a
        pre-loaded encoding to share one tiktoken registry across
        the chunker and the embedder, or to use a custom encoding
        in tests.

    Raises
    ------
    ChunkingError
        The configured encoding name doesn't exist in tiktoken
        (e.g. typo in config.yaml).
    """

    def __init__(
        self,
        settings: ChunkingSettings,
        *,
        encoding: tiktoken.Encoding | None = None,
    ) -> None:
        # Defensive copy: ChunkingSettings is frozen, so this is
        # just a local reference. We extract the values we need
        # up front so the hot path doesn't keep dereferencing.
        self._chunk_size: int = settings.chunk_size
        self._chunk_overlap: int = settings.chunk_overlap
        self._encoding_name: str = settings.encoding

        # Resolve the tiktoken encoding. We do it eagerly so a
        # bad encoding name fails at Chunker construction (one
        # place to fix) rather than at the first ``chunk()`` call.
        if encoding is not None:
            self._encoding: tiktoken.Encoding = encoding
        else:
            try:
                self._encoding = tiktoken.get_encoding(self._encoding_name)
            except (KeyError, ValueError) as exc:
                # tiktoken raises ValueError on unknown encoding
                # names ("Unknown encoding foo.") and KeyError for
                # some plugin / module paths. Catch both.
                raise ChunkingError(
                    f"unknown tiktoken encoding {self._encoding_name!r}; "
                    f"see https://github.com/openai/tiktoken for the list "
                    f"of supported encodings"
                ) from exc

        # Pre-compute the trim-window size in tokens. We re-read
        # ``self._chunk_size`` each call so a subclass overriding
        # ``_chunk_size`` would still get correct behaviour — but
        # in practice this is a constant for the chunker's life.
        self._trim_window: int = max(
            1, int(self._chunk_size * _SENTENCE_TRIM_FRACTION)
        )

    # ---- public surface ----------------------------------------------------

    @property
    def encoding_name(self) -> str:
        """The tiktoken encoding name in use (e.g. ``"cl100k_base"``)."""
        return self._encoding_name

    @property
    def chunk_size(self) -> int:
        """The configured target tokens per chunk."""
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """The configured overlap between consecutive chunks (tokens)."""
        return self._chunk_overlap

    def count_tokens(self, text: str) -> int:
        """Count tokens in ``text`` using the chunker's encoding.

        Exposed so tests and other modules can verify sizes without
        re-loading the encoding. Cheap — ``tiktoken`` is fast.
        """
        # ``encode`` on an empty string returns ``[]``; we don't
        # special-case it.
        return len(self._encoding.encode(text))

    def chunk(
        self,
        text: str,
        source: str,
        page: int | None = None,
    ) -> list[Chunk]:
        """Split ``text`` into overlapping :class:`Chunk` objects.

        Parameters
        ----------
        text:
            The text to chunk. Empty / whitespace-only input
            returns an empty list (not a single empty chunk —
            an empty chunk is meaningless to the embedder).
        source:
            The source filename or document identifier. Forwarded
            to every emitted chunk.
        page:
            Optional 1-based page number (for PDFs). ``None`` for
            plain text / Markdown. Forwarded to every chunk.

        Returns
        -------
        list[Chunk]
            Zero or more chunks, in document order. ``chunk_index``
            is 0-based and contiguous. ``char_offset`` is the
            position of the chunk's first character in ``text``.
        """
        if not text or not text.strip():
            return []

        # ---- Step 1: encode --------------------------------------------
        # ``encode`` is the single hot path. ``allowed_special="all"``
        # permits any special tokens tiktoken might encounter
        # (rare for plain text / PDF / Markdown; important if a
        # future contributor passes a transcript with embedded
        # ``<|endoftext|>`` etc.).
        token_ids: list[int] = self._encoding.encode(text, allowed_special="all")
        n_tokens = len(token_ids)
        if n_tokens == 0:
            return []

        # ---- Step 2: window --------------------------------------------
        # We step through ``token_ids`` in fixed strides of
        # ``stride = chunk_size - chunk_overlap``. Each step
        # produces one chunk.
        stride: int = self._chunk_size - self._chunk_overlap
        if stride <= 0:
            # Defensive: ChunkingSettings's ``_overlap_less_than_size``
            # validator should prevent this, but if a future
            # contributor removes that check we want a clean error
            # rather than an infinite loop.
            raise ChunkingError(
                f"chunk_size ({self._chunk_size}) must be strictly greater "
                f"than chunk_overlap ({self._chunk_overlap})"
            )

        chunks: list[Chunk] = []
        chunk_index = 0
        start = 0
        # Track the most recent end-of-chunk to guarantee forward
        # progress even when sentence-trim shrinks a chunk past
        # ``start + stride`` (e.g. on very long sentences).
        last_end = 0
        while start < n_tokens:
            # Proposed window: [start, end) in token space.
            end = min(start + self._chunk_size, n_tokens)
            is_last = end >= n_tokens

            if is_last:
                # The final chunk always extends to the end of the
                # text — no point in trimming a tail that has no
                # following chunk to align with.
                trimmed_end = end
            else:
                # Look back from ``end`` for a sentence terminator
                # within the trim window. If found, set
                # ``trimmed_end`` to just after it (in tokens).
                trimmed_end = self._find_sentence_break(
                    token_ids, start, end, self._trim_window
                )

            # Guard against trim shrinking the window below the
            # overlap (which would defeat the purpose of overlap).
            # We accept some loss of overlap here in pathological
            # cases (a single sentence longer than ``chunk_size``);
            # the alternative is a tiny chunk with no content.
            if trimmed_end - start < max(1, self._chunk_overlap // 2):
                trimmed_end = end

            # ---- Step 3: decode & record --------------------------------
            chunk_token_ids = token_ids[start:trimmed_end]
            chunk_text = self._encoding.decode(chunk_token_ids)
            if not chunk_text.strip():
                # Defensive: shouldn't happen because the input
                # wasn't whitespace-only, but if a trim produced
                # nothing useful, skip rather than emit a noise
                # chunk.
                break

            # ``char_offset`` is the position of the first
            # character of ``chunk_text`` in the *original*
            # ``text``. We use ``full_decoded.find(chunk_text,
            # last_known_offset)`` to handle the rare case where
            # the same substring appears earlier in the document
            # (e.g. repeated headers).
            char_offset = self._char_offset_in(text, chunk_text, last_end)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=source,
                    page=page,
                    chunk_index=chunk_index,
                    char_offset=char_offset,
                    token_count=len(chunk_token_ids),
                )
            )
            chunk_index += 1
            last_end = trimmed_end

            # If we just emitted the final chunk, stop.
            if is_last:
                break

            # Advance ``start`` by ``stride``, but never backwards
            # (a sentence-trim that shrank the chunk past
            # ``start + stride`` must still make progress).
            new_start = max(last_end - self._chunk_overlap, start + 1)
            # And never past the end of the document.
            new_start = min(new_start, n_tokens - 1)
            if new_start <= start:
                # No forward progress — bail to avoid an infinite
                # loop. This should be unreachable in practice
                # (the ``trimmed_end - start < overlap // 2``
                # guard above forces ``last_end > start``).
                break
            start = new_start

        return chunks

    # ---- internal helpers --------------------------------------------------

    def _find_sentence_break(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        trim_window: int,
    ) -> int:
        """Find the best sentence break within the last ``trim_window`` tokens.

        Returns the token index *just after* the sentence-ending
        punctuation. If no break is found, returns ``end`` (the
        natural chunk boundary).

        The search is done in *text space* (decode the slice and
        scan with a regex) because matching ``[.!?][<ws>|<quote>]``
        in token IDs is brittle — a sentence-ending period is often
        fused with the next word in BPE.

        Why scan from the *right* edge backward?
        We want the *latest* sentence break inside the trim
        window, so the chunk carries as much content as possible
        while still ending at a sentence boundary.
        """
        # The leftmost token we still consider as a potential break.
        left = max(start, end - trim_window)
        if left >= end:
            return end

        # Decode just the slice we care about.
        slice_text = self._encoding.decode(token_ids[left:end])
        # Scan from the right for a sentence terminator followed
        # by whitespace/quote. ``re.finditer`` returns matches in
        # left-to-right order; we walk them in reverse.
        matches = list(_SENTENCE_END_RE.finditer(slice_text))
        if not matches:
            return end
        # Pick the last match. The match's end is the position in
        # ``slice_text`` just after the terminator character; we
        # don't need exact token alignment (we re-decode later),
        # so we approximate by character count.
        last_match = matches[-1]
        char_pos_in_slice = last_match.end()

        # Convert the character position back to a token index.
        # We re-decode progressively until our decoded text
        # reaches ``char_pos_in_slice`` characters — this gives a
        # tight, exact token boundary.
        cumulative = 0
        for tok_idx in range(left, end):
            tok_text = self._encoding.decode([token_ids[tok_idx]])
            cumulative += len(tok_text)
            if cumulative >= char_pos_in_slice:
                return tok_idx + 1
        return end

    @staticmethod
    def _char_offset_in(haystack: str, needle: str, search_from: int) -> int:
        """Return the offset of ``needle`` in ``haystack`` at or after ``search_from``.

        Used to compute the ``char_offset`` field of a chunk in
        the *original* input text. We start the search at
        ``search_from`` (the previous chunk's end) so that
        duplicate substrings earlier in the document don't
        confuse us.

        Falls back to ``search_from`` if the substring isn't found
        (which can happen when tiktoken's BPE round-trip alters
        whitespace at boundaries). The chunk is still useful for
        retrieval — the offset is informational.
        """
        idx = haystack.find(needle, max(0, search_from))
        if idx == -1:
            return search_from
        return idx


# ----------------------------------------------------------------------------
# Convenience: a default chunker
# ----------------------------------------------------------------------------


def default_chunker() -> Chunker:
    """Return a :class:`Chunker` with the project's default settings.

    The defaults (400 tokens, 50 overlap, ``cl100k_base``) match
    ``config.yaml`` and the SRS FR-3 requirements. Useful for
    REPL probes and one-off scripts that don't want to load
    settings themselves.
    """
    return Chunker(ChunkingSettings())
