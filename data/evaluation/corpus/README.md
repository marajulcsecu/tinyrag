# Evaluation Corpus (Step 5.3)

The gold set (`../gold_set.json`) is scored against a fixed 4-document
corpus. This directory documents how to reproduce that corpus exactly.

The source documents themselves live in `data/documents/`, which is
gitignored — two of the three are freely re-downloadable, and the third
is checked in here because it is authored source material.

## The corpus

| Document | Chunks | Origin | Gold set questions |
|---|---|---|---|
| `raspberry-pi-5-product-brief.pdf` | 9 | Downloaded (Raspberry Pi Ltd) | Q01–Q06 |
| `rfc7252-coap.txt` | 195 | Downloaded (IETF / RFC Editor) | Q07–Q12 |
| `3rd-gen-Nest-Learning-Thermostat-Install-Guide-UK.pdf` | 41 | Pre-existing fixture | Q13–Q15, Q22 |
| `rag.txt` | 1 | Pre-existing demo file | Q16 |
| `tinyrag_deployment_faq.md` | 5 | **Authored — kept in this directory** | Q17–Q20 |

Total: **251 document chunks.** Sensor summaries (180 chunks) are indexed
separately and are deliberately excluded from this gold set — sensor
evaluation is deferred pending advisor guidance.

### Why the corpus crosses 251 chunks

RFC 7252 alone contributes 195 chunks, which pushes the corpus past
`SMALL_CORPUS_MAX_CHUNKS = 50` in `src/tinyrag/core/retriever.py`. That is
intentional. Below that gate the retriever substitutes a relaxed 0.15
threshold and widens the candidate pool to the whole store; above it, the
configured `similarity_threshold: 0.3` and the bounded pool apply. The
earlier 42-chunk demo corpus never exercised the normal path. All four
previously-verified regression questions (warranty, ErP, combi-boiler,
RAG) were re-confirmed to still retrieve correctly under the stricter
regime — see the notes fields in `gold_set.json`.

## Reproducing the corpus from a clean clone

```bash
# 1. Fetch the two externally-hosted documents (SHA-256 verified)
python scripts/download_fixtures.py --name raspberry-pi-5-product-brief \
    --fixtures-dir data/documents
python scripts/download_fixtures.py --name rfc7252-coap \
    --fixtures-dir data/documents

# 2. Copy the authored FAQ into place
cp data/evaluation/corpus/tinyrag_deployment_faq.md data/documents/

# 3. Ingest all three with the REAL embedder (see warning below)
PYTHONPATH=src python scripts/ingest.py \
    data/documents/raspberry-pi-5-product-brief.pdf --doc-type manual --embedder real
PYTHONPATH=src python scripts/ingest.py \
    data/documents/rfc7252-coap.txt --doc-type manual --embedder real
PYTHONPATH=src python scripts/ingest.py \
    data/documents/tinyrag_deployment_faq.md --doc-type faq --embedder real
```

The Nest guide and `rag.txt` are assumed already ingested from the Phase 4
demo corpus. Verify the end state with:

```bash
sqlite3 data/metadata.db \
    "SELECT filename, doc_type, num_chunks FROM documents;"
```

### `--embedder real` is mandatory

Passing `--embedder fake` produces SHA-256 hash vectors with no semantic
meaning. Because the query path always uses real MiniLM, a corpus indexed
with the fake embedder yields vectors that are not comparable to query
vectors: similarity scores collapse toward noise and semantic retrieval
silently stops working, while the system still returns plausible-looking
answers. This exact train/serve mismatch went undetected for a long time
and is why every ingest command above passes `--embedder real` explicitly.

## Licensing

Both downloaded documents are freely distributable: the Raspberry Pi 5
product brief is published by Raspberry Pi Ltd for public distribution,
and RFC 7252 is distributable under the IETF Trust legal provisions. They
are used here solely as retrieval inputs for academic evaluation and are
not redistributed through this repository — only their URLs and SHA-256
checksums are recorded, in `scripts/download_fixtures.py`.
