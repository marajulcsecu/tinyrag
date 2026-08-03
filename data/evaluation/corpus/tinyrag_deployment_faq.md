# TinyRAG Deployment FAQ

Operator notes for running TinyRAG on edge hardware. These are our own
deployment conventions, gathered while bringing the system up on a laptop
and porting it to a Raspberry Pi 5.

---

## Hardware and sizing

### What hardware is TinyRAG deployed on?

The reference deployment target is a Raspberry Pi 5 with 8 GB of RAM and a
64 GB microSD card. Development happens first on a Dell Inspiron 15 3520
laptop (Intel i5-1235U, 8 GB RAM, Ubuntu 24.04 LTS), then the same code and
configuration are moved to the Pi. Only the deployment target in the
configuration file changes between the two.

### How much RAM does TinyRAG need?

Plan for 8 GB. The quantized language model occupies roughly 2 GB of
resident memory once loaded, the embedding model adds about 90 MB, and the
FAISS index plus the Python process account for a few hundred megabytes
more. The remaining headroom is left for the operating system and page
cache. A 4 GB board can load the model but leaves very little margin, so we
do not recommend it.

### How much disk space is required?

Budget at least 10 GB of free space. The breakdown is approximately 2 GB
for the primary GGUF model, 500 MB for the compiled llama.cpp build tree,
300 MB for the Python virtual environment (dominated by PyTorch), and the
remainder for indexes, logs, and document storage. The setup script
enforces a 3 GB minimum before it will begin.

### Why is the thread count set automatically?

The number of inference threads defaults to the CPU core count, detected at
launch. The Raspberry Pi 5 has four cores, while the development laptop
reports twelve. Setting more threads than there are physical cores causes
threads to compete for the same cores, and because inference is entirely
CPU-bound, the extra context switching makes generation slower rather than
faster. Override the detected value with the `LLAMA_THREADS` environment
variable only when benchmarking.

---

## Models

### Which language model does TinyRAG use by default?

The default is Llama 3.2 3B Instruct, quantized to Q4_K_M, served through
llama.cpp. The on-disk file is about 1.9 GB.

### Why was the default model changed from Phi-3 Mini?

Phi-3 Mini produced incoherent output once the prompt exceeded roughly 2048
tokens. Retrieval-augmented prompts routinely reach 2500 to 3600 tokens, so
the failure was triggered on ordinary questions. The cause was traced to the
sliding-window attention path in that particular GGUF file, not to the
llama.cpp build: the identical binary and prompt produced coherent output
with Llama 3.2 3B. The default was therefore switched, and Phi-3 Mini is
retained only as a comparison model for evaluation.

### What embedding model is used?

Sentence-Transformers `all-MiniLM-L6-v2`, which produces 384-dimensional
vectors. It is small enough to run comfortably on the Pi and fast enough
that embedding is never the bottleneck.

### Why must the same embedding model be used for indexing and querying?

Vectors from different embedding models are not comparable. If documents
are indexed with one model and queries are embedded with another, every
similarity score collapses toward noise and retrieval silently stops
working — the system still returns answers, but they are not grounded in
the most relevant passages. We hit exactly this failure once, when the
document-upload path was left on a placeholder embedder while the query
path used the real model, and semantic retrieval did not work at all until
the corpus was re-embedded.

---

## Retrieval behaviour

### How many passages are retrieved per question?

Five document passages by default. The retriever fetches a wider pool of
candidates, reranks them using keyword overlap, discards anything below the
relevance threshold, and then keeps the top five for the prompt.

### Why does the relevance threshold change with corpus size?

On a very small corpus — fifty chunks or fewer — absolute similarity scores
are unreliable, because there are too few documents for a meaningful score
distribution to form. Below that size the system substitutes a lower
threshold of 0.15 and widens the candidate pool to the entire store. Above
fifty chunks the configured threshold of 0.3 applies and the candidate pool
is bounded. The wider pool costs nothing in latency, because the number of
passages placed in the prompt is capped either way.

### Why does keyword reranking match whole words only?

An early version matched keywords as substrings, so a query about "RAG"
also matched the letters inside "storage" and "coverage". A passage about
warranty coverage scored high enough to outrank the correct answer and pad
the prompt. Matching on tokenized whole words fixed it.

### What happens when the answer is not in the corpus?

The system refuses, replying that it does not have enough information in
the provided documents. It does not guess. Refusal is a correct outcome and
is scored as a pass when the question is genuinely out of scope.

---

## Operations

### How is the stack started and stopped?

`bash setup.sh` performs a one-time, idempotent install: system packages,
the Python environment, the llama.cpp build, and the model download.
`bash run.sh` then launches the model server and the web API together,
waits for both health checks, and tears both down on Ctrl+C. `bash stop.sh`
cleans up from another terminal. All three scripts are safe to re-run.

### What ports does TinyRAG use?

The web interface and REST API listen on port 8000. The llama.cpp model
server listens on port 8080 and is not intended to be exposed outside the
host.

### Does TinyRAG require an internet connection?

Only during installation, to download dependencies and model weights. Once
the models are on disk and documents are indexed, the system runs entirely
offline. Disconnecting the network does not affect question answering,
because no request leaves the device.

### What document formats can be ingested?

PDF, plain text, and Markdown. PDF text is extracted with column-aware
block ordering, which matters for multi-column layouts: reading a
two-column page straight across the gutter interleaves the columns and
produces text that cannot be retrieved meaningfully.

### How long does a typical answer take?

On the laptop, expect 15 to 25 seconds for a complete answer, most of which
is the model processing the prompt before the first token appears. Answers
stream token by token as they are produced, so text begins appearing before
generation finishes. Longer prompts and longer answers take proportionally
more time.
