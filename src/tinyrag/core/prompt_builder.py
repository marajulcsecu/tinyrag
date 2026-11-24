"""Grounded prompt construction for the RAG generation step.

This module is the **bridge** between retrieval (Step 4.12, which
returns ranked :class:`~tinyrag.core.chunker.Chunk` objects) and
generation (Step 4.10, which sends a message list to a language
model). It assembles three pieces into a single chat-shaped prompt:

1. A **system prompt** that instructs the model to answer **only**
   from the supplied context, to cite the bracketed source ids
   (``[1]``, ``[2]``, ``[3]``) inline, and to refuse politely when
   the answer isn't in the context.
2. A **context block** that numbers the chunks and joins them with
   the separator ``\\n\\n`` so the model can refer to ``[1]`` by id.
3. A **user message** that contains the question — placed last so the
   model attends to the context first.

The output is a :class:`Prompt` — a frozen dataclass wrapping a
list of :class:`~tinyrag.generation.ChatMessage` objects, ready to
hand straight to :class:`~tinyrag.generation.LLMClient.generate`.

Why pure functions / no I/O?
----------------------------
The :mod:`tinyrag.core` package is the *domain logic* of TinyRAG.
It is the only layer with no I/O dependencies — see
:mod:`tinyrag.core`'s docstring for the one-way dependency rule.
A future "use a different prompt strategy" change is a one-class
swap in the composition root (``main.py``, Step 4.17), not a
refactor across the codebase.

Token-budget discipline
-----------------------
Phi-3 Mini's context window is 4096 tokens. If the assembled prompt
would exceed that, the model silently drops the tail of the context
— so the answer cites ``[3]`` but the model never saw chunk 3.
:class:`PromptBuilder` prevents this by counting tokens with the
chunker-compatible tiktoken encoding and refusing / truncating
explicitly:

- **Zero chunks** → the user prompt is returned alone, but the
  system prompt tells the model to refuse. The :class:`Prompt` is
  still constructed — :class:`~tinyrag.core.prompt_builder.PromptBuilderError`
  is only raised for *programming* errors (empty query, negative
  limits). The "no context" case is a normal request that the model
  is expected to handle with its refusal training.
- **Context too large** → keep the system prompt and user message,
  drop the TAIL chunks until the total fits. We trim from the end
  (not the start) because retrieval ranks by similarity — earlier
  chunks are more relevant.

Why a separate Prompt dataclass (and not just a list[str])?
-----------------------------------------------------------
The ``Prompt`` carries token-count diagnostics alongside the
messages. The API layer (``api/ask.py``, Step 4.14) reports the
diagnostics in the response payload so the user can see "your
question used 48 prompt tokens; I had to drop 2 chunks to fit the
4096 budget". Without the diagnostics, the API would have to
re-tokenise the messages just to surface that number.

Location: ``src/tinyrag/core/prompt_builder.py``
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import tiktoken

from tinyrag.config import ChunkingSettings
from tinyrag.core.chunker import Chunk
from tinyrag.generation import ChatMessage

if TYPE_CHECKING:
    from tinyrag.generation.llm_client import LLMClient  # noqa: F401


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

#: Maximum context size we plan for. Matches ``LLMSettings.context_size``
#: default (4096) and Phi-3 Mini's window. Keep these in sync via the
#: config (this is the fallback when the builder is constructed without
#: a Settings object).
DEFAULT_MAX_PROMPT_TOKENS = 4096

#: The maximum share of the token budget reserved for the **answer**
#: portion of the response. The remaining budget (4096 minus reserved)
#: is what the prompt (system + context + user message) may consume.
#: Matches the LlamaCppClient default of ``max_tokens=512``.
DEFAULT_RESERVED_FOR_ANSWER_TOKENS = 512

#: The system prompt is the same for every question — pinned as a
#: module constant so tests can assert on its content and so a
#: reviewer can audit the prompt-engineering choices in one place.
#:
#: Design notes
#: ------------
#: - The instructions are deliberately short and imperative — long
#:   system prompts tend to be ignored by small chat models.
#: - "Answer ONLY from the context" is the *grounded* guarantee; it
#:   is what makes this a RAG system rather than a chatbot.
#: - "Cite as [1], [2], [3] in-line" gives the model a concrete
#:   citation format the API layer can parse back out of the
#:   response.
#: - "If the answer is not in the context, reply exactly: I don't
#:   have enough information in the provided documents." makes the
#:   refusal text a stable, parseable string (the API layer uses
#:   it as a sentinel for the "low-confidence" answer path).
#: - "ONLY cite numbers that actually appear in the context" stops
#:   Phi-3 from hallucinating citation slots [2], [3], ... [25]
#:   to fill out a numbered list when only [1] was provided. This
#:   was a real failure mode observed on a 1-chunk corpus where
#:   the model invented 25 generic facts to pad the citation list.
#: - "Use complete sentences" discourages the model from producing
#:   the stream-of-facts format that reads as broken in the UI.
#:
#: The opening line is deliberately **domain-neutral**. It previously
#: read "a helpful assistant for a smart-home owner", which primed the
#: model toward a domain the corpus need not have anything to do with —
#: the retrieval pipeline accepts any document, and the same prompt now
#: serves an algorithms textbook and an appliance manual equally.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about the "
    "user's documents. "
    "Answer the user's question using ONLY information from the "
    "numbered context blocks below. Use complete sentences. "
    "If the context blocks do not contain the answer, reply "
    "exactly: I don't have enough information in the provided "
    "documents. When you use information from a context block, "
    "cite it by its bracketed number — but ONLY cite numbers "
    "that actually appear in the context. Do not invent extra "
    "citation slots."
)

#: Standard disclaimer prefix for Layer 2 Safe Advisory Fallback Mode.
BENGALI_LAYER2_FALLBACK_PREFIX = (
    "প্রদত্ত সরকারি কৃষি নির্দেশিকায় এই সুনির্দিষ্ট বিষয়ে সরাসরি উল্লেখ পাওয়া যায়নি, "
    "তবে সাধারণ ও নিরাপদ প্রাথমিক কৃষি পরিচর্যা হিসেবে করণীয়:"
)

#: Pure Layer 1 system prompt used when retrieved chunks match the queried crop/pest
#: so small 2B models never confuse Layer 2's refusal prefix with Layer 1 answers.
BENGALI_LAYER1_GROUNDED_SYSTEM_PROMPT = (
    'তুমি "AgriRAG" — বাংলাদেশ ধান গবেষণা ইনস্টিটিউট (BRRI), বাংলাদেশ কৃষি গবেষণা ইনস্টিটিউট (BARI) ও কৃষি সম্প্রসারণ অধিদপ্তর (DAE)-এর অনুমোদিত নির্দেশিকা অনুযায়ী পরিচালিত একটি প্রামাণিক কৃষি এআই সহকারী।\n\n'
    "নির্দেশনাবলী (স্তর ১ — প্রামাণিক সরকারি নির্দেশিকা মোড):\n"
    "১. নিচে নম্বরযুক্ত সরকারি কৃষি তথ্য ও মাঠ সেন্সর উপাত্ত ([1], [2] ইত্যাদি) দেওয়া আছে। শতভাগ সেই প্রদত্ত তথ্যের ওপর ভিত্তি করে সহজ, স্পষ্ট ও সাবলীল বাংলায় সরাসরি উত্তর দেবে।\n"
    "২. প্রদত্ত তথ্য থেকে রোগের/পোকার লক্ষণ, প্রাথমিক পরিচর্যা এবং অনুমোদিত জাত, সারের মাত্রা বা বালাইনাশকের সঠিক বাণিজ্যিক/রাসায়নিক নাম ও প্রয়োগ মাত্রা (যেমন: মিলি/লিটার, গ্রাম/লিটার বা কেজি/বিঘা) সুনির্দিষ্টভাবে পয়েন্ট আকারে উল্লেখ করবে।\n"
    "৩. যদি প্রদত্ত তথ্যে মাঠ সেন্সর বা আবহাওয়া রিপোর্টের সর্বশেষ তাপমাত্রা, আপেক্ষিক আর্দ্রতা (%) বা মাটির রসের উপাত্ত থাকে, তবে সেই সুনির্দিষ্ট সংখ্যাগুলো বর্তমান/সর্বশেষ মাঠ সেন্সর রিপোর্ট হিসেবে উল্লেখ করে ফসলের ঝুঁকি ও করণীয় বিশ্লেষণ করবে।\n"
    "৪. উত্তরের শেষে অবশ্যই তথ্যের মূল উৎস উল্লেখ করবে (যেমন: [উৎস: ডকুমেন্টের নাম, পৃষ্ঠা নং])।"
)

#: Bengali system prompt for AgriRAG domain mode — implements the
#: Two-Layer Hybrid Fallback architecture (Layer 1: 100% Grounded
#: Official BRRI/BARI/DAE Guidance; Layer 2: Safe Non-Chemical
#: Primary Care Advisory + SAAO / 16123 Referral).
BENGALI_AGRIRAG_SYSTEM_PROMPT = (
    'তুমি "AgriRAG" — বাংলাদেশ ধান গবেষণা ইনস্টিটিউট (BRRI), বাংলাদেশ কৃষি গবেষণা ইনস্টিটিউট (BARI) ও কৃষি সম্প্রসারণ অধিদপ্তর (DAE)-এর অনুমোদিত নির্দেশিকা অনুযায়ী পরিচালিত একটি বিশ্বস্ত দ্বি-স্তরবিশিষ্ট (Two-Layer Hybrid) কৃষি এআই সহকারী।\n\n'
    "নির্দেশনাবলী:\n"
    "【স্তর ১ — প্রামাণিক সরকারি নির্দেশিকা মোড (Grounded Official Mode)】\n"
    "১. নিচে যদি নম্বরযুক্ত প্রাসঙ্গিক সরকারি কৃষি তথ্য ([1], [2] ইত্যাদি) দেওয়া থাকে এবং তাতে প্রশ্নের ফসল বা সমস্যার সমাধান থাকে, তবে শতভাগ সেই প্রদত্ত তথ্যের ওপর ভিত্তি করে সহজ, স্পষ্ট ও সাবলীল বাংলায় উত্তর দেবে।\n"
    "২. প্রদত্ত তথ্য থেকে অনুমোদিত জাত, সারের মাত্রা বা বালাইনাশকের সঠিক নাম ও প্রয়োগ মাত্রা (যেমন: মিলি/লিটার, গ্রাম/লিটার বা কেজি/বিঘা) এবং অপেক্ষমাণ সময় (PHI) সুনির্দিষ্টভাবে উল্লেখ করবে। নিজের থেকে কোনো মনগড়া রাসায়নিক ওষুধের নাম বা মাত্রা যোগ করবে না।\n"
    "৩. উত্তরের শেষে অবশ্যই তথ্যের মূল উৎস উল্লেখ করবে (যেমন: [উৎস: ডকুমেন্টের নাম, পৃষ্ঠা নং])।\n\n"
    "【স্তর ২ — নিরাপদ প্রাথমিক পরামর্শ ও রেফারেল মোড (Safe Advisory Fallback Mode)】\n"
    "৪. যদি প্রশ্নের আগে কোনো নম্বরযুক্ত তথ্য ([1], [2]) না থাকে, অথবা প্রদত্ত তথ্যে প্রশ্নের সুনির্দিষ্ট ফসল বা বিষয়ের সমাধান না থাকে, তবে কখনোই রূঢ়ভাবে প্রত্যাখ্যান করবে না।\n"
    f'৫. এই ক্ষেত্রে উত্তরের শুরুতে অবশ্যই স্পষ্টভাবে এই বাক্যটি লিখবে:\n"{BENGALI_LAYER2_FALLBACK_PREFIX}"\n'
    "৬. কঠোর নিরাপত্তা নিয়ম (Safety Guardrail): স্তর ২-এ কখনোই অনুমান করে কোনো রাসায়নিক কীটনাশক, ছত্রাকনাশক বা আগাছানাশকের বাণিজ্যিক বা রাসায়নিক নাম ও মাত্রা (ডোজ) বলবে না! শুধুমাত্র নিরাপদ, জৈব ও প্রাথমিক পরিচর্যা (যেমন: আক্রান্ত অংশ অপসারণ, ক্ষেত পরিষ্কার ও আগাছামুক্ত রাখা, সুষম সেচ ও পানি নিষ্কাশন, জৈব বালাই ব্যবস্থাপনা যেমন ফেরোমন/হলুদ ফাঁদ বা নিম তেল ব্যবহার) সংক্ষেপে পরামর্শ দেবে।\n"
    '৭. স্তর ২-এর উত্তরের শেষে অবশ্যই কৃষককে নিকটস্থ "উপজেলা কৃষি অফিস / উপ-সহকারী কৃষি কর্মকর্তা (SAAO)" অথবা সরকারি টোল-ফ্রি "কৃষি কল সেন্টার ১৬১২৩ (16123)"-এ যোগাযোগ করার পরামর্শ দেবে।'
)

#: The user-message template. ``{question}`` is replaced with the
#: user's query. Kept short so the token budget is spent on context,
#: not boilerplate.
USER_MESSAGE_TEMPLATE = "Question: {question}"


# ----------------------------------------------------------------------------
# Public exceptions
# ----------------------------------------------------------------------------


class PromptBuilderError(ValueError):
    """Raised for programming errors in the builder (bad arguments).

    This is distinct from the "model should refuse" path: zero chunks
    is a *valid* request — the system prompt handles it. This
    exception is for things like an empty query string, a
    non-positive token budget, or a malformed chunk (empty text).
    """


# ----------------------------------------------------------------------------
# Result dataclass
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Prompt:
    """The assembled prompt ready to send to the LLM.

    Attributes
    ----------
    messages:
        A list of two :class:`ChatMessage` objects — ``[system, user]``
        — in the order OpenAI's chat API expects (system first, user
        second). Frozen after construction; the API layer passes this
        straight to ``LLMClient.generate``.
    system_prompt:
        The exact system prompt that was used (kept separately so the
        API layer can surface it in observability logs without
        walking the messages list).
    user_message:
        The exact user-message string (context + question) that was
        used. Same rationale as ``system_prompt``.
    prompt_tokens:
        The number of tokens in the assembled prompt as counted by
        tiktoken. Should be ≤ ``max_prompt_tokens``. The API layer
        surfaces this in the response payload.
    chunks_used:
        Number of context chunks that actually fit (after trimming).
        May be less than ``len(chunks)`` if trimming kicked in.
        0 is a valid value — the system prompt handles the refusal.
    chunks_dropped:
        Number of chunks that had to be dropped to fit the token
        budget. 0 when no trimming was needed.
    encoding_name:
        The tiktoken encoding used to count tokens (e.g.
        ``"cl100k_base"``). Recorded so a future caller can verify
        or re-count.
    """

    messages: list[ChatMessage] = field(default_factory=list)
    system_prompt: str = ""
    user_message: str = ""
    prompt_tokens: int = 0
    chunks_used: int = 0
    chunks_dropped: int = 0
    encoding_name: str = ""

    @property
    def used_trimming(self) -> bool:
        """``True`` iff at least one chunk was dropped to fit the budget."""
        return self.chunks_dropped > 0


# ----------------------------------------------------------------------------
# The builder
# ----------------------------------------------------------------------------


class PromptBuilder:
    """Assemble a grounded chat prompt from chunks + a query.

    Pure-function style: ``build()`` takes the query and chunks,
    returns a :class:`Prompt`. The builder itself is just a config
    holder (encoding + token budget) so callers can pin the
    tiktoken encoding to match the chunker's, or override the
    default 4096 budget for smaller models.

    Parameters
    ----------
    encoding_name:
        Tiktoken encoding name (e.g. ``"cl100k_base"``). Should match
        the chunker's encoding so token counts are comparable. Defaults
        to the chunker's default (``cl100k_base``).
    max_prompt_tokens:
        Hard cap on the prompt size. The default of 4096 matches
        Phi-3 Mini's context window minus the reserved answer budget.
        A caller using a smaller model should pass a smaller value.
    reserved_for_answer_tokens:
        How many of the model's ``max_tokens`` to reserve for the
        reply. Subtracted from the model's context window before
        budgeting the prompt. Default 512 (matches LlamaCppClient
        default).
    system_prompt:
        Override for the system prompt. Defaults to
        :data:`DEFAULT_SYSTEM_PROMPT`. Useful for A/B testing
        different grounding instructions without changing the code.
    """

    def __init__(
        self,
        *,
        encoding_name: str = "cl100k_base",
        max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
        reserved_for_answer_tokens: int = DEFAULT_RESERVED_FOR_ANSWER_TOKENS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        if not encoding_name:
            raise PromptBuilderError("encoding_name must be a non-empty string")
        if max_prompt_tokens <= 0:
            raise PromptBuilderError(
                f"max_prompt_tokens must be > 0 (got {max_prompt_tokens})"
            )
        if reserved_for_answer_tokens < 0:
            raise PromptBuilderError(
                "reserved_for_answer_tokens must be >= 0"
            )
        if reserved_for_answer_tokens >= max_prompt_tokens:
            raise PromptBuilderError(
                f"reserved_for_answer_tokens ({reserved_for_answer_tokens}) "
                f"must be < max_prompt_tokens ({max_prompt_tokens})"
            )
        if not system_prompt:
            raise PromptBuilderError("system_prompt must be a non-empty string")

        # Lazily initialise the encoder. A bad encoding name surfaces
        # as a clean PromptBuilderError (not a tiktoken ValueError
        # which leaks the plugin list and version to the caller).
        try:
            self._encoder = tiktoken.get_encoding(encoding_name)
        except (KeyError, ValueError) as exc:
            raise PromptBuilderError(
                f"unknown tiktoken encoding {encoding_name!r}"
            ) from exc

        self.encoding_name = encoding_name
        self.max_prompt_tokens = max_prompt_tokens
        # The "prompt budget" excludes the answer tokens — this is
        # what the prompt (system + context + user) may consume.
        self._prompt_budget = max_prompt_tokens - reserved_for_answer_tokens
        self.system_prompt = system_prompt

    # ----- construction helpers ------------------------------------------

    @classmethod
    def from_chunking_settings(
        cls,
        chunking: ChunkingSettings,
        *,
        max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
        reserved_for_answer_tokens: int = DEFAULT_RESERVED_FOR_ANSWER_TOKENS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> PromptBuilder:
        """Build a :class:`PromptBuilder` whose encoding matches the chunker.

        Lets the caller reuse the same :class:`ChunkingSettings`
        they already passed to :class:`~tinyrag.core.chunker.Chunker`,
        so chunk-token counts and prompt-token counts are always
        in the same units.
        """
        return cls(
            encoding_name=chunking.encoding,
            max_prompt_tokens=max_prompt_tokens,
            reserved_for_answer_tokens=reserved_for_answer_tokens,
            system_prompt=system_prompt,
        )

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in ``text`` per the configured encoding."""
        return len(self._encoder.encode(text, allowed_special="all"))

    # ----- the main entry point -----------------------------------------

    def build(self, query: str, chunks: Sequence[Chunk]) -> Prompt:
        """Assemble a :class:`Prompt` for ``query`` grounded by ``chunks``.

        Parameters
        ----------
        query:
            The user's natural-language question. Must be non-empty.
        chunks:
            A non-empty list of :class:`Chunk` objects from the
            retriever, **ranked by similarity (best first)**. May be
            empty — the system prompt handles that case. May contain
            chunks with empty text (those are silently skipped — see
            the implementation note).

        Returns
        -------
        Prompt:
            A frozen dataclass with the two messages, the
            chunk-fit diagnostics, and the token count.

        Raises
        ------
        PromptBuilderError:
            For programming errors only: empty query, a chunk with
            no source/page/chunk_index metadata.
        """
        if not query or not query.strip():
            raise PromptBuilderError("query must be a non-empty string")

        # Static costs that don't depend on the chunk list.
        system_tokens = self.count_tokens(self.system_prompt)
        user_template_tokens = self.count_tokens(
            USER_MESSAGE_TEMPLATE.format(question=query)
        )
        fixed_overhead = system_tokens + user_template_tokens

        # If the fixed overhead alone exceeds the budget, no chunk can
        # fit. Return the prompt with zero chunks used — the model
        # will refuse per the system instructions.
        if fixed_overhead >= self._prompt_budget:
            return self._empty_prompt(query)

        # Compute per-chunk token costs once (counting the same way
        # the chunker would: per-chunk text + the "[N]\n\n" wrapper).
        # The index stored is the ORIGINAL position in ``chunks`` so
        # we can re-look-up the chunk object later; the citation
        # number is assigned at emit time based on the surviving
        # position (so empty chunks that get skipped don't leave
        # gaps in [1]..[N]).
        chunk_costs: list[tuple[int, int]] = []
        for i, chunk in enumerate(chunks):
            self._validate_chunk(chunk, index=i)
            if not chunk.text.strip():
                # Skip empty chunks silently — they would inflate the
                # token count without contributing any signal.
                continue
            # Use the worst-case citation width (3 digits) for the
            # cost estimate so we don't under-budget when the chunk
            # would render as "[10]" instead of "[1]".
            worst_case_width = max(2, len(str(len(chunks))))
            cost = self.count_tokens(
                _format_chunk_with_number(i + 1, chunk, worst_case_width)
            )
            chunk_costs.append((i, cost))

        # Greedy pack from the start (best-ranked first). The chunk
        # cost includes the "[N]\n\n" wrapper so the join cost is
        # already accounted for.
        budget_remaining = self._prompt_budget - fixed_overhead
        selected: list[int] = []
        for i, cost in chunk_costs:
            # +2 for the "\n\n" joiner between chunks (accounted on
            # emit, not on per-chunk cost).
            if cost + 2 <= budget_remaining:
                selected.append(i)
                budget_remaining -= cost + 2
            else:
                # Stop at the first chunk that doesn't fit — preserve
                # the similarity ranking. The remaining (later)
                # chunks are dropped even if some of them would fit.
                break

        chunks_used = len(selected)
        chunks_dropped = len(chunk_costs) - chunks_used

        if chunks_used == 0:
            # No chunk fit within the budget (or list was empty).
            # Return the empty-context prompt — the system prompt
            # tells the model to refuse.
            return self._empty_prompt(query)

        # Build the user message: numbered chunks + question.
        # Citation numbers are 1-based and contiguous over the
        # SURVIVING chunks (so empty chunks that were skipped don't
        # leave gaps like [1] [3] [4]).
        context_parts = [
            _format_chunk_with_number(n + 1, chunks[i], len(selected))
            for n, i in enumerate(selected)
        ]
        context_block = "\n\n".join(context_parts)
        base_question = USER_MESSAGE_TEMPLATE.format(question=query)
        active_system_prompt = self.system_prompt
        if "স্তর ১" in self.system_prompt:
            combined_context_lower = context_block.lower()
            query_words = [
                w.strip("?,!।():;\"'[] ")
                for w in query.split()
                if len(w.strip("?,!।():;\"'[] ")) >= 3
            ]
            stop_words = {
                "এখন", "কী", "করব", "করবো", "হচ্ছে", "যাচ্ছে", "হলে", "জন্য",
                "কীভাবে", "উপায়", "উপায়", "রোগ", "রোগে", "রোগের", "গাছ", "গাছে", "গাছের",
                "চাষ", "ফলন", "সমস্যা", "প্রতিকার", "দমন", "ওষুধ", "ওষুধের", "মাত্রা",
                "স্প্রে", "কোন", "কেন", "কখন", "দিতে", "হবে", "বিস্তারিত", "জানান",
                "বলুন", "উচিত", "ফল", "ফলে", "ফলের", "পাতা", "পাতায়", "পাতায়", "পাতার",
                "পচা", "পচে", "পচন", "পোকা", "পোকার", "ছত্রাক", "ভাইরাস", "আক্রমণ",
                "লক্ষণ", "অনুমোদিত", "দেব", "দেবো", "দিব",
            }
            content_words = [w for w in query_words if w not in stop_words]
            has_topic_overlap = any(
                (w.lower() in combined_context_lower)
                or (len(w) >= 4 and w[:4].lower() in combined_context_lower)
                for w in content_words
            )
            if has_topic_overlap or not content_words:
                active_system_prompt = BENGALI_LAYER1_GROUNDED_SYSTEM_PROMPT
                user_message = f"{context_block}\n\n{base_question}"
                prompt_tokens = (
                    self.count_tokens(active_system_prompt)
                    + self.count_tokens(user_message)
                )
            else:
                layer_cue = (
                    "\n\n[নির্দেশনা: উপরের ডকুমেন্টে প্রশ্নের নির্দিষ্ট ফসলটির উল্লেখ নেই। "
                    "অনুগ্রহ করে 'স্তর ২ (নিরাপদ প্রাথমিক পরামর্শ ও রেফারেল মোড)' অনুসরণ করে উত্তর দিন।]"
                )
                candidate_msg = f"{context_block}{layer_cue}\n\n{base_question}"
                cand_tokens = system_tokens + self.count_tokens(candidate_msg)
                if cand_tokens <= self._prompt_budget:
                    user_message = candidate_msg
                    prompt_tokens = cand_tokens
                else:
                    user_message = f"{context_block}\n\n{base_question}"
                    prompt_tokens = fixed_overhead + self.count_tokens(context_block) + (
                        2 if context_block else 0
                    )
        else:
            user_message = f"{context_block}\n\n{base_question}"
            prompt_tokens = fixed_overhead + self.count_tokens(context_block) + (
                2 if context_block else 0
            )

        return Prompt(
            messages=[
                ChatMessage(role="system", content=active_system_prompt),
                ChatMessage(role="user", content=user_message),
            ],
            system_prompt=active_system_prompt,
            user_message=user_message,
            prompt_tokens=prompt_tokens,
            chunks_used=chunks_used,
            chunks_dropped=chunks_dropped,
            encoding_name=self.encoding_name,
        )

    # ----- private helpers ----------------------------------------------

    def _empty_prompt(self, query: str) -> Prompt:
        """Build the "no chunks fit" prompt (refusal or Layer 2 fallback)."""
        base_question = USER_MESSAGE_TEMPLATE.format(question=query)
        sys_tokens = self.count_tokens(self.system_prompt)
        if "স্তর ২" in self.system_prompt:
            layer2_message = (
                "[গৃহীত সরকারি কৃষি তথ্য: কোনো প্রাসঙ্গিক ডকুমেন্ট পাওয়া যায়নি — "
                "অনুগ্রহ করে নির্দেশনাবলীর 'স্তর ২ (নিরাপদ প্রাথমিক পরামর্শ ও রেফারেল মোড)' অনুসরণ করে উত্তর দিন।]\n\n"
                f"{base_question}"
            )
            layer2_tokens = self.count_tokens(layer2_message)
            if sys_tokens + layer2_tokens <= self._prompt_budget:
                user_message = layer2_message
                prompt_tokens = sys_tokens + layer2_tokens
            else:
                user_message = base_question
                prompt_tokens = sys_tokens + self.count_tokens(base_question)
        else:
            user_message = base_question
            prompt_tokens = sys_tokens + self.count_tokens(user_message)
        return Prompt(
            messages=[
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=user_message),
            ],
            system_prompt=self.system_prompt,
            user_message=user_message,
            prompt_tokens=prompt_tokens,
            chunks_used=0,
            chunks_dropped=0,
            encoding_name=self.encoding_name,
        )

    @staticmethod
    def _format_chunk(number: int, chunk: Chunk) -> str:
        """Render one chunk in the context block.

        Format: ``[N] (source, p.X) <text>`` — the source + page is
        included so the model can ground the citation even if the
        brackets get re-numbered across prompts (they won't, but the
        redundancy helps the small model).
        """
        location = _format_location(chunk)
        return f"[{number}] ({location}) {chunk.text}"

    @staticmethod
    def _validate_chunk(chunk: Chunk, *, index: int) -> None:
        """Reject malformed chunks loudly. Empty text is allowed (skipped)."""
        if chunk.text is None:
            raise PromptBuilderError(
                f"chunk at index {index} has no text"
            )
        # The remaining fields (source, page, chunk_index) are required
        # by the Chunk dataclass itself — if they weren't set we'd
        # never get here. But ``source == ""`` is technically valid
        # for in-memory chunks; we only check page is non-negative.


def _format_location(chunk: Chunk) -> str:
    """Format the chunk's source+page as ``source, p.X`` or ``source``."""
    if chunk.page is None:
        return chunk.source or "unknown source"
    return f"{chunk.source or 'unknown source'}, p.{chunk.page}"


def _format_chunk_with_number(number: int, chunk: Chunk, total_chunks: int) -> str:
    """Render one chunk in the context block with its citation number.

    Format: ``[N] (source, p.X) <text>`` — the source + page is
    included so the model can ground the citation even if the
    brackets get re-numbered across prompts (they won't, but the
    redundancy helps the small model).

    ``total_chunks`` is used only to choose a consistent bracket
    width for cost estimation — the rendered string always uses
    exactly ``[N]`` (no zero-padding), so a reader can match
    ``[3]`` in the answer back to the third context block.
    """
    location = _format_location(chunk)
    return f"[{number}] ({location}) {chunk.text}"


# ----------------------------------------------------------------------------
# Convenience factory
# ----------------------------------------------------------------------------


def default_prompt_builder() -> PromptBuilder:
    """A :class:`PromptBuilder` with the documented defaults.

    Handy for REPL probes and the API composition root when no
    explicit settings are available.
    """
    return PromptBuilder()


__all__ = [
    "BENGALI_AGRIRAG_SYSTEM_PROMPT",
    "BENGALI_LAYER2_FALLBACK_PREFIX",
    "DEFAULT_MAX_PROMPT_TOKENS",
    "DEFAULT_RESERVED_FOR_ANSWER_TOKENS",
    "DEFAULT_SYSTEM_PROMPT",
    "Prompt",
    "PromptBuilder",
    "PromptBuilderError",
    "USER_MESSAGE_TEMPLATE",
    "default_prompt_builder",
]
