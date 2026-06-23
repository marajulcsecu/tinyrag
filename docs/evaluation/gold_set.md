# TinyRAG — Evaluation Gold Set (20 Questions)

**Project Title:** TinyRAG — A Lightweight, On-Device Retrieval-Augmented Generation Assistant for Smart Home IoT
**Document version:** 1.0
**Date:** 2026-06-23
**Status:** Draft — referenced by Phase 5.3 of the canonical roadmap (`docs/06_roadmap_v2.md`)

---

## 0. Purpose

This document is the **evaluation test set** for TinyRAG. It contains **20 questions** that the system must answer. Each question has:

- An **expected answer hint** (what the answer should look like)
- **Expected keywords** (for automated scoring)
- **Expected source type** (manual / FAQ / sensor / combined)
- **Expected source hint** (which document or sensor the answer should come from)
- **Difficulty** (easy / medium / hard)
- **Category** (lookup / sensor_query / reasoning / fallback)

The questions are designed to:
1. Cover the **4 main use cases** of TinyRAG (manual Q&A, FAQ Q&A, sensor queries, combined).
2. Include **edge cases** (out-of-domain, ambiguous, multi-step reasoning).
3. Be **answerable** from a realistic corpus (3 device manuals + 1 home FAQ + 30 days of sensor data).
4. Be **non-trivial** — the answers are not just single keyword lookups; they require real retrieval and grounding.

## 1. Source Corpus (the ground truth)

The gold set assumes the following corpus is ingested:

| Source | Type | Approx. size | Filename (suggested) |
|--------|------|--------------|----------------------|
| Nest Thermostat Manual | PDF | ~50 pages | `nest_thermostat_manual.pdf` |
| Philips Hue Smart Lighting Manual | PDF | ~40 pages | `philips_hue_manual.pdf` |
| TP-Link Kasa Smart Plug Quick-Start Guide | PDF or TXT | ~15 pages | `tplink_kasa_plug_spec.txt` |
| Custom Home FAQ (hand-written) | MD | ~5 KB | `home_faq.md` |
| Synthetic 30-day sensor log | CSV | ~17,000 rows | `synthetic_30d.csv` |

**All 20 questions are answerable from this corpus.** If you ingest different documents, you'll need to adjust `expected_source_hint` accordingly.

## 2. Question Distribution

| Category | Count | Why this many |
|----------|-------|---------------|
| **Manual lookup** (specific fact from a device manual) | 6 | Most common real-world use case |
| **FAQ / custom home knowledge** | 4 | Tests custom knowledge base handling |
| **Sensor query** (historical sensor data) | 6 | Core IoT capability |
| **Combined / reasoning** (uses multiple sources) | 3 | Tests cross-source synthesis |
| **Out-of-domain / fallback** | 1 | Tests graceful "I don't know" behavior |
| **Total** | **20** | |

## 3. The 20 Questions

Each question is structured as a JSON-ready object. The actual `data/evaluation/gold_set.json` will use exactly this structure (Phase 5.6 generates it).

---

### Q01 — Easy manual lookup

```yaml
id: Q01
query: "How do I reset my Nest thermostat to factory settings?"
expected_keywords: ["reset", "factory", "settings"]
expected_source_type: manual
expected_source_hint: nest_thermostat_manual
category: manual_lookup
difficulty: easy
notes: "Most common support question. Answer should mention specific steps from the manual."
```

---

### Q02 — Easy manual lookup

```yaml
id: Q02
query: "What does error code E3 mean on my Philips Hue bulb?"
expected_keywords: ["E3", "error", "reset"]
expected_source_type: manual
expected_source_hint: philips_hue_manual
category: manual_lookup
difficulty: easy
notes: "Tests error code retrieval. Answer should explain the error and the fix."
```

---

### Q03 — Medium manual lookup

```yaml
id: Q03
query: "How do I pair a new TP-Link Kasa smart plug with my Wi-Fi network?"
expected_keywords: ["pair", "Wi-Fi", "Kasa", "app"]
expected_source_type: manual
expected_source_hint: tplink_kasa_plug_spec
category: manual_lookup
difficulty: medium
notes: "Multi-step process from the spec. Answer should outline the pairing flow."
```

---

### Q04 — Medium manual lookup

```yaml
id: Q04
query: "What is the warranty period for the Nest thermostat?"
expected_keywords: ["warranty", "year", "limited"]
expected_source_type: manual
expected_source_hint: nest_thermostat_manual
category: manual_lookup
difficulty: medium
notes: "Tests specific fact retrieval (numbers, dates)."
```

---

### Q05 — Hard manual lookup (multi-page)

```yaml
id: Q05
query: "How do I update the firmware on my Philips Hue bridge?"
expected_keywords: ["firmware", "update", "Hue", "app"]
expected_source_type: manual
expected_source_hint: philips_hue_manual
category: manual_lookup
difficulty: hard
notes: "Multi-step process; answer needs context from multiple sections."
```

---

### Q06 — Hard manual lookup (specific value)

```yaml
id: Q06
query: "What is the maximum wattage supported by the TP-Link Kasa smart plug?"
expected_keywords: ["watt", "W", "maximum"]
expected_source_type: manual
expected_source_hint: tplink_kasa_plug_spec
category: manual_lookup
difficulty: hard
notes: "Specific technical spec. Tests precise retrieval."
```

---

### Q07 — Easy FAQ lookup

```yaml
id: Q07
query: "What smart devices do I have in my home?"
expected_keywords: ["thermostat", "Hue", "Kasa", "devices"]
expected_source_type: faq
expected_source_hint: home_faq
category: faq_lookup
difficulty: easy
notes: "Tests custom home FAQ. Answer should list all 3 device types."
```

---

### Q08 — Easy FAQ lookup

```yaml
id: Q08
query: "Tell me about the home network setup."
expected_keywords: ["network", "Wi-Fi", "router"]
expected_source_type: faq
expected_source_hint: home_faq
category: faq_lookup
difficulty: easy
notes: "Tests custom FAQ retrieval for home-specific context."
```

---

### Q09 — Medium FAQ lookup

```yaml
id: Q09
query: "What is the recommended humidity range for comfortable sleep?"
expected_keywords: ["humidity", "%", "sleep"]
expected_source_type: faq
expected_source_hint: home_faq
category: faq_lookup
difficulty: medium
notes: "Domain knowledge + home FAQ. Answer should give a specific range."
```

---

### Q10 — Medium FAQ lookup

```yaml
id: Q10
query: "Where can I find the Wi-Fi password for my home network?"
expected_keywords: ["password", "Wi-Fi", "router"]
expected_source_type: faq
expected_source_hint: home_faq
category: faq_lookup
difficulty: medium
notes: "Tests that the FAQ can hold practical info, not just device docs."
```

---

### Q11 — Easy sensor query

```yaml
id: Q11
query: "What was the average temperature in the living room this week?"
expected_keywords: ["temperature", "living room", "average"]
expected_source_type: sensor
expected_source_hint: synthetic_30d
category: sensor_query
difficulty: easy
notes: "Direct sensor query. Answer should reference actual values from the CSV."
```

---

### Q12 — Easy sensor query

```yaml
id: Q12
query: "Which day last week had the highest energy usage?"
expected_keywords: ["energy", "day", "highest"]
expected_source_type: sensor
expected_source_hint: synthetic_30d
category: sensor_query
difficulty: easy
notes: "Max-finding query. Answer should name a specific day."
```

---

### Q13 — Medium sensor query

```yaml
id: Q13
query: "Was there any motion in the kitchen between 2 AM and 3 AM last Tuesday?"
expected_keywords: ["motion", "kitchen", "2 AM", "3 AM"]
expected_source_type: sensor
expected_source_hint: synthetic_30d
category: sensor_query
difficulty: medium
notes: "Time-range filtering + boolean answer."
```

---

### Q14 — Medium sensor query

```yaml
id: Q14
query: "Compare the humidity between the bedroom and living room yesterday."
expected_keywords: ["humidity", "bedroom", "living room"]
expected_source_type: sensor
expected_source_hint: synthetic_30d
category: sensor_query
difficulty: medium
notes: "Cross-sensor comparison. Answer should mention both rooms' values."
```

---

### Q15 — Hard sensor query

```yaml
id: Q15
query: "Did the temperature drop below 18 degrees Celsius at any point last week?"
expected_keywords: ["temperature", "18", "below"]
expected_source_type: sensor
expected_source_hint: synthetic_30d
category: sensor_query
difficulty: hard
notes: "Boolean query with threshold. Answer should be yes/no with timing."
```

---

### Q16 — Hard sensor query (aggregation)

```yaml
id: Q16
query: "What was the average daily energy consumption this month?"
expected_keywords: ["energy", "average", "daily"]
expected_source_type: sensor
expected_source_hint: synthetic_30d
category: sensor_query
difficulty: hard
notes: "Multi-step aggregation. Answer should give a specific kWh number."
```

---

### Q17 — Hard combined query (manual + sensor)

```yaml
id: Q17
query: "Why might my energy bill be higher than usual this month?"
expected_keywords: ["energy", "usage", "high"]
expected_source_type: combined
expected_source_hint: synthetic_30d + home_faq
category: combined_reasoning
difficulty: hard
notes: "Synthesizes sensor data (high usage) + FAQ (billing context). Answer should reference both."
```

---

### Q18 — Hard combined query (manual + sensor)

```yaml
id: Q18
query: "Which device in my home likely uses the most standby power?"
expected_keywords: ["standby", "power", "device"]
expected_source_type: combined
expected_source_hint: tplink_kasa_plug_spec + philips_hue_manual
category: combined_reasoning
difficulty: hard
notes: "Reasoning across multiple manuals. Answer should reason about typical standby draw."
```

---

### Q19 — Hard combined query (sensor + FAQ)

```yaml
id: Q19
query: "When was the living room temperature highest last week, and was that unusual?"
expected_keywords: ["temperature", "living room", "highest"]
expected_source_type: combined
expected_source_hint: synthetic_30d + home_faq
category: combined_reasoning
difficulty: hard
notes: "Multi-part query: specific time + comparison to typical range."
```

---

### Q20 — Fallback test (out-of-domain)

```yaml
id: Q20
query: "What is the meaning of life?"
expected_keywords: []   # intentionally empty — fallback expected
expected_source_type: none
expected_source_hint: null
category: fallback
difficulty: easy
notes: "Out-of-domain question. System MUST return the fallback message ('I don't have information about that in my knowledge base'), NOT make up an answer."
expected_behavior: fallback
```

---

## 4. Distribution Summary

By difficulty:

| Difficulty | Count | IDs |
|------------|-------|-----|
| Easy | 6 | Q01, Q02, Q07, Q08, Q11, Q12, Q20 (7 actually) |
| Medium | 7 | Q03, Q04, Q09, Q10, Q13, Q14 |
| Hard | 7 | Q05, Q06, Q15, Q16, Q17, Q18, Q19 |

By source:

| Source type | Count | IDs |
|-------------|-------|-----|
| Manual only | 6 | Q01-Q06 |
| FAQ only | 4 | Q07-Q10 |
| Sensor only | 6 | Q11-Q16 |
| Combined (multi-source) | 3 | Q17, Q18, Q19 |
| None (fallback) | 1 | Q20 |

## 5. JSON Format

The actual gold set file (`data/evaluation/gold_set.json`) will look like this:

```json
[
  {
    "id": "Q01",
    "query": "How do I reset my Nest thermostat to factory settings?",
    "expected_keywords": ["reset", "factory", "settings"],
    "expected_source_type": "manual",
    "expected_source_hint": "nest_thermostat_manual",
    "category": "manual_lookup",
    "difficulty": "easy",
    "notes": "Most common support question."
  },
  {
    "id": "Q02",
    "query": "What does error code E3 mean on my Philips Hue bulb?",
    "expected_keywords": ["E3", "error", "reset"],
    "expected_source_type": "manual",
    "expected_source_hint": "philips_hue_manual",
    "category": "manual_lookup",
    "difficulty": "easy",
    "notes": "Tests error code retrieval."
  },
  ...
  {
    "id": "Q20",
    "query": "What is the meaning of life?",
    "expected_keywords": [],
    "expected_source_type": "none",
    "expected_source_hint": null,
    "category": "fallback",
    "difficulty": "easy",
    "notes": "Out-of-domain. Expect fallback.",
    "expected_behavior": "fallback"
  }
]
```

## 6. How the Gold Set Is Used

The gold set is consumed by **`scripts/eval.py`** (Phase 5.6 of the canonical roadmap):

1. For each model in the comparison set (TinyLlama 1.1B, Llama 3.2 3B, Phi-3 Mini 3.8B):
   - Switch llama.cpp to that model.
   - For each of the 20 questions:
     - Run the question through TinyRAG.
     - Record the answer, latency, top-1 similarity score.
     - Score the answer using two methods:
       - **Automatic:** count overlap with `expected_keywords`.
       - **Human:** apply the rubric in `docs/evaluation/scoring_rubric.md`.
   - Output a CSV per model: `reports/eval_<model>_<date>.csv`.

2. **Aggregate** the per-model CSVs into a comparison table:
   - Model × average accuracy (auto + human).
   - Model × average latency (first token + end-to-end).
   - Model × peak RAM.
   - Model × score per question.

## 7. Adapting to Your Actual Corpus

The gold set assumes the 5 sources listed in Section 1. If your corpus is different:

| If your source is... | Update `expected_source_hint` to... |
|----------------------|--------------------------------------|
| A different thermostat brand | e.g., `ecobee_thermostat_manual` |
| A different bulb brand | e.g., `lifx_bulb_manual` |
| A different plug brand | e.g., `wyze_plug_spec` |
| A custom home FAQ in a different format | keep `home_faq` |
| Real sensor data (not synthetic) | keep `synthetic_30d` or rename to actual source |

The **questions stay the same** — only `expected_source_hint` needs to change.

## 8. What's NOT in the Gold Set (and why)

These are deliberately excluded to keep the set focused:

- ❌ **Voice queries** — text-only is the primary mode (voice is stretch).
- ❌ **Multi-turn follow-ups** — single-turn only by design.
- ❌ **Non-English queries** — English only by design.
- ❌ **Adversarial / injection attempts** — out of scope for capstone.
- ❌ **Real-time questions** ("what's the temperature RIGHT NOW") — the summarizer works on historical data; live queries are a future extension.

## 9. Open Questions for the Student

| # | Question | Default I'll go with |
|---|----------|----------------------|
| Q1 | Should Q20 (meaning of life) be replaced with a smart-home-specific out-of-domain question? | **Keep "meaning of life"** — it's the classic fallback test, easily recognizable. |
| Q2 | Should we add 5 more "combined reasoning" questions to make combined sources more represented? | **No — 3 is enough for v1.** Can add more in v2. |
| Q3 | Should `expected_keywords` be a strict requirement (all must match) or fuzzy (any match)? | **Any match** — gives partial credit for partial answers. |
| Q4 | Should the gold set be versioned (v1, v2, ...)? | **Yes — start with v1, this is it.** Track changes in git. |

## 10. Document Approval

| Role | Name | Approval | Date |
|------|------|----------|------|
| Student | Marajul Haque | ⏳ pending | |
| Advisor | Abu Nowshed Chy | (not required for v1) | |

---

*End of Gold Set v1. Used by Phase 5.3 and Phase 5.6 of the canonical roadmap.*