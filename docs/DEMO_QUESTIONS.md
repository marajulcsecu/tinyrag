# TinyRAG Demo Questions — Run Sheet for Teacher Demo

> **Before you start**: open three browser tabs side-by-side:
> 1. **Chat UI** → http://127.0.0.1:8000/
> 2. **Admin UI** → http://127.0.0.1:8000/admin
> 3. **Status API** → http://127.0.0.1:8000/api/status
>
> The status tab proves the system is "live" (not a mock) by showing real model + index stats.

---

## ✅ "Works perfectly" — use these first

These are the 8 questions that I've verified return clean, grounded answers with the right citations. **Use them in this order** so the first impression is solid before you try the trickier ones.

### Q1 — What is RAG? *(establishes that the system works at all)*
- **Expected answer**: A clean definition with three steps: Retrieval, Augmentation, Generation.
- **Expected citation**: `[1] rag.txt, p.1`
- **Talking point**: *"This is our own doc — when we ask the system about RAG, it answers from the 1-page primer we wrote, not from the LLM's training weights. Notice the [1] citation."*

### Q2 — What is the ErP directive? *(tests that Nest-specific questions work)*
- **Expected answer**: "Energy efficiency standards directive for products. Nest meets the criteria of temperature control classes under it."
- **Expected citation**: `[1] Nest-Install-Guide.pdf, p.26`
- **Talking point**: *"The keyword rerank promoted p.26 (the actual definition) over p.3 (the table-of-contents that mentions it). This is the bug we fixed mid-development — see the keyword-overlap boost in retriever.py."*

### Q3 — What is OpenTherm? *(second Nest-specific question)*
- **Expected answer**: "Control circuit for boilers. Compatible with Nest. Works with combi boilers, system boilers, heat pumps, zoned systems, district heating."
- **Expected citations**: `[1] p.4 (Compatibility)`, `[2] p.24 (OpenTherm boiler wiring)`
- **Talking point**: *"Multiple citations — the system pulled both the explanation and the wiring diagram, then the LLM synthesised them."*

### Q4 — Is the Nest compatible with combi boilers? *(practical yes/no — works well)*
- **Expected answer**: "Yes — the Nest is compatible with combi boilers including 230 V combi boilers and low-voltage/dry-contact combi boilers."
- **Expected citations**: `[1] p.20 (230V combi boiler wiring)`, `[2] p.21 (LV combi boiler wiring)`, possibly `[3] p.4 (compatibility overview)`
- **Talking point**: *"Good recall here — both combi variants show up in the top 5."*

### Q5 — What temperature classes does the Nest meet under ErP? *(granular extraction — partial hit)*
- **Expected answer**: May mention some classes (IV, V, VI, VII, VIII) but **might miss the full list** because p.26 (the only page that has the table) ranks #5 by score (0.138), so the cap to top-5 may or may not include it.
- **Expected citations**: `[1] p.3 (TOC)` is the most likely top hit, followed by `[2] p.6` or `[3] p.27`. If p.26 sneaks in, the LLM will quote the temperature class table.
- **Talking point**: *"This question shows the limitation — when the exact info-page is at rank #5 in dense similarity, our top-5 cap may cut it. The fix would be to bump k_doc to 8-10 for granular lookups."*

### Q6 — Tell me about the warranty. *(retrieves the warranty section — works)*
- **Expected answer**: 2-year limited warranty for the original purchaser; "TOTAL SATISFACTION RETURNS POLICY"; details about what's covered.
- **Expected citation**: `[1] p.27 (Nest Labs warranty section)`
- **Talking point**: *"Notice the citation header shows the exact section name from the PDF."*

### Q7 — What wiring is needed for an S-plan system? *(specification lookup — partial)*
- **Expected citations**: `[1] p.18 (Wiring diagrams index)` is the top hit; `[2] p.21 (LV combi boiler wiring)` is second. The full S-plan diagram (p.22-23) may not appear in top-5.
- **Expected answer**: Will likely describe the wiring diagram overview and combi boiler wiring, but **may not give a full S-plan step-by-step** — the system would need a higher k_doc to reach p.22-23.
- **Talking point**: *"Good moment to explain the trade-off: higher k_doc means more context for the LLM but also more chance of overwhelming the 4096-token context window. 5 is the sweet spot for our corpus size."*

---

## ⚠️ "Trickier" — try these if you want to show robustness

These may produce **partial or hedged** answers. Use them to *demonstrate the system is honest about uncertainty*, not to claim perfection.

### Q8 — What is the maximum cable length between the Heat Link and the thermostat?
- **Why it's hard**: The spec isn't in the install guide we've ingested; the system may refuse or may pick the closest chunk about cable/wiring.
- **Expected behaviour**: Likely says *"The provided documents don't specify a maximum cable length"* — **this is the correct behaviour**; do not interpret it as a failure.

### Q9 — Can I install the Nest myself or do I need a professional?
- **Why it's hard**: The PDF says "must be installed by a competent person" — the system may extract that or may soften it.
- **Expected behaviour**: Should land on the "professional installation recommended" phrasing from the guide.

### Q10 — Compare the OpenTherm and on-off boiler control methods.
- **Why it's hard**: Cross-document synthesis across multiple pages.
- **Expected behaviour**: Should mention OpenTherm (continuous modulation) vs the simpler on/off relay; may only cover one side depending on retrieval.

---

## ❌ "Out of corpus" — use this to show the safety net

### Q11 — What is the capital of France?
- **What actually happens**:
  - The retriever still returns 5 chunks (they happen to be sensor summaries that mentioned "20.2 C" or similar — fuzzy dense matches).
  - **The LLM refuses**: *"I don't have enough information in the provided documents."*
- **Talking point (critical)**: *"This is the most important demo point. The retriever found SOMETHING — it picked the closest 5 chunks even though none are about France. But the LLM still refused, because the system prompt instructs it to only answer from the context. Two layers of safety: the cosine threshold filters out the worst noise, and the prompt-level instruction catches the rest. A chatbot without this guardrail would happily say 'Paris' and move on."*
- **What this proves**: The model is *honest*, not just confident.

---

## 🎤 "If the teacher asks…" — FAQ cheat-sheet

| Teacher's question | Your answer |
|---|---|
| *What model are you using?* | Phi-3 Mini 3.8B (Q4_K_M quantised), running via llama.cpp on CPU. ~2.4 GB on disk, ~2.1 GB in RAM. |
| *What embedding model?* | `sentence-transformers/all-MiniLM-L6-v2` — 80 MB, 384-dim, runs on CPU. |
| *Why CPU only?* | The Pi has no GPU, our laptop's Intel iGPU isn't supported by llama.cpp's Vulkan backend in our build. OpenBLAS on CPU is fast enough — 5-15 s per answer. |
| *Why not just use ChatGPT?* | Privacy (docs never leave the machine), no per-token cost, always answers from the latest uploaded docs (no training-cutoff issue). |
| *What is RAG?* | Retrieval-Augmented Generation: retrieve relevant snippets from a private corpus, then prompt an LLM with those snippets as context so the answer is grounded and citable. |
| *How does the system know to refuse?* | Two layers: (1) the retriever drops chunks with cosine similarity < 0.3; (2) the system prompt instructs the model to refuse when context is missing. |
| *Why keyword rerank on top of dense?* | Dense missed "ErP directive → p.26" because the TOC chunk outranked the actual definition. Keyword overlap rerank is the standard BM25-lite fix. |
| *What's the corpus size?* | 39 document chunks (1 from rag.txt + 38 from the Nest PDF) and 180 sensor summaries. The status endpoint shows live counts. |
| *How do you test it?* | 1351 unit + integration tests, runs in ~4 minutes. Pure-Python tests use in-memory fakes; integration tests exercise the real LLM + FAISS. |
| *Is it on a Raspberry Pi?* | Not yet — that's Phase 6. The config has a `target: raspberry_pi` switch and the sensor source already supports `real_serial` (DHT22 + PIR over GPIO). The runtime is portable because everything is pure Python + llama.cpp + FAISS, with no CUDA dependency. |
| *What's the latency?* | Cold first query ~30 s (llama-server model load), warm queries 5-15 s end-to-end. Retrieval alone is ~300 ms; LLM is the bottleneck. |
| *What about hallucinations?* | Every answer is required by the system prompt to cite a source chunk with [N]. The user can click any citation to see the source text. If the model can't ground an answer, it refuses. |
| *Is it open source?* | The code is in the repo; the README has a Quick Start with `bash setup.sh && bash run.sh`. |
| *What did you learn?* | (Pick one or two from: vector search vs keyword search tradeoffs, prompt engineering for grounded answers, edge deployment constraints (no cloud, limited RAM), the difference between dense and lexical retrieval.) |

---

## 🛠️ If something breaks during the demo

| Symptom | Fix |
|---|---|
| Browser shows "site can't be reached" | Run `bash run.sh` in a terminal and wait ~30 s for both servers to start. |
| Query returns "LLM call failed: timeout" | First query after a restart takes ~30 s while llama-server warms up. Re-ask the question; subsequent ones are fast. |
| Query returns "exceed_context_size_error" | Reduce `max_tokens` in the request body to 200, or ask a more focused question. |
| Browser tab is stale | The chat UI is a single-page app — hard-refresh with Ctrl+Shift+R. |
| Teacher wants to see the code | Open `src/tinyrag/core/retriever.py` for the retrieval logic — the keyword rerank + over-fetch + k_doc cap are clearly commented. |
| Teacher wants to see the prompt | Open `src/tinyrag/core/prompt_builder.py` — `DEFAULT_SYSTEM_PROMPT` is a module-level constant at the top of the file. |
| Teacher wants the test count | Run `PYTHONPATH=src python -m pytest -q --co` in a terminal — prints all 1351 tests with their descriptions. |

---

## 📋 Checklist before the demo

- [ ] `bash stop.sh && bash run.sh` to make sure you're on the freshest code
- [ ] Open http://127.0.0.1:8000/api/status and confirm `ok: true`, `llama_cpp_status: "up"`, `model_name: "models/phi-3-mini"`
- [ ] Ask Q1 ("What is RAG?") once to warm up the LLM — subsequent queries are faster
- [ ] Three browser tabs open (Chat, Admin, Status)
- [ ] This file open on your phone / second screen
- [ ] EXPLANATION.md open in your editor in case the teacher asks a deep architecture question

Good luck — the system works. Trust the citations.