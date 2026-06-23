# TinyRAG — Evaluation Scoring Rubric

**Project Title:** TinyRAG — A Lightweight, On-Device Retrieval-Augmented Generation Assistant for Smart Home IoT
**Document version:** 1.0
**Date:** 2026-06-23
**Status:** Draft — referenced by Phase 5.5 of the canonical roadmap (`docs/06_roadmap_v2.md`)
**Companion document:** `docs/evaluation/gold_set.md` (the 20 evaluation questions)

---

## 0. Purpose

The **automatic scoring** in `scripts/eval.py` only counts keyword overlap. It cannot tell whether the answer is actually correct, hallucinated, or well-grounded. This rubric gives a **human judge** (the student, and optionally the advisor) a consistent, repeatable way to score each model answer on a 0–2 scale (or 0–3 if the optional "Excellent" tier is used).

The rubric is used in Phase 5.5 (`Run human evaluation`) of the canonical roadmap, and feeds into the final report (`reports/final_report.pdf`) in Phase 7.

---

## 1. Scoring Scale (default 3-point)

Each answer is scored **0, 1, or 2**:

| Score | Label | Meaning |
|-------|-------|---------|
| **2** | Correct & grounded | Answer is factually correct **and** grounded in retrieved context (or in sensor data). |
| **1** | Partially correct | Answer has some correct info but is incomplete, has minor errors, or is poorly grounded. |
| **0** | Wrong / hallucinated / fallback broken | Answer is wrong, fabricated, or — for fallback questions — fails to refuse. |

The optional **3rd tier** (Excellent) splits Score 2 into:

| Score | Label | Meaning |
|-------|-------|---------|
| **3** | Excellent | Score 2 + answer is well-formatted, concise, and cites the specific source. |
| **2** | Correct | Score 2 but answer is verbose, messy, or doesn't cite. |
| **1** | Partially correct | (unchanged) |
| **0** | Wrong | (unchanged) |

**Recommended for the capstone: use the 4-point scale (0–3).** It gives more granularity in the final report without much extra judging effort.

---

## 2. The Five Judging Criteria

For each answer, judge **five dimensions** independently, then combine them into the final score.

### Criterion A — Factual correctness

| Question to ask | Score 3 | Score 2 | Score 1 | Score 0 |
|-----------------|---------|---------|---------|---------|
| Are the **facts** in the answer correct? | All facts correct AND answer is the most important fact | All facts correct | Some facts correct, some wrong or missing | Mostly or entirely wrong |

### Criterion B — Grounding (no hallucination)

| Question to ask | Score 3 | Score 2 | Score 1 | Score 0 |
|-----------------|---------|---------|---------|---------|
| Is the answer **supported by retrieved context or sensor data**? | Every claim is grounded; nothing invented | Every claim is grounded | Most claims grounded but 1 minor invention | Major inventions; "made up" answer |

### Criterion C — Completeness

| Question to ask | Score 3 | Score 2 | Score 1 | Score 0 |
|-----------------|---------|---------|---------|---------|
| Does the answer **fully address the question**? | All parts addressed, including follow-up implications | All explicit parts addressed | Some parts missing | Most of the question left unaddressed |

### Criterion D — Format & clarity

| Question to ask | Score 3 | Score 2 | Score 1 | Score 0 |
|-----------------|---------|---------|---------|---------|
| Is the answer **readable and well-structured**? | Concise, well-punctuated, easy to scan | Readable but verbose or slightly awkward | Hard to parse, run-ons, or weird format | Incoherent |

### Criterion E — Citation / source mention

| Question to ask | Score 3 | Score 2 | Score 1 | Score 0 |
|-----------------|---------|---------|---------|---------|
| Does the answer **mention the source** (e.g., "according to the Nest manual", or "based on sensor data from June 14")? | Explicit citation with document name + section/date | Mentions the source type ("from the manual") | Vague hint ("from the docs") | No source mention at all |

---

## 3. Combining Criteria into Final Score

**Default formula (4-point scale):**

```
final = round((A + B + C + D + E) / 5)
```

But the rubric is designed so that **a single "0" on A or B is a hard floor**:

> If **A (correctness)** is 0 **OR** **B (grounding)** is 0, the **final score is 0**, regardless of the other criteria. Hallucination is the worst failure mode.

Worked example — Q01 ("How do I reset my Nest thermostat to factory settings?"):

| Criterion | Judge's score |
|-----------|---------------|
| A — Factual correctness | 3 (all steps correct) |
| B — Grounding | 3 (steps came from the manual) |
| C — Completeness | 2 (missed the warning about saving schedules first) |
| D — Format & clarity | 3 (numbered steps, clean) |
| E — Citation | 3 ("according to the Nest manual, page 12…") |

Average = (3+3+2+3+3)/5 = **2.8 → 3 (Excellent)**.

---

## 4. Edge Case Handling

These are the trickiest cases. Decide once, apply consistently.

### 4.1 Partial hallucination

**Example:** Q04 "What is the warranty period for the Nest thermostat?"
- Retrieved context: "Nest thermostats come with a standard 2-year limited warranty."
- Model answer: "The Nest thermostat has a 5-year warranty."

The "5-year" is invented. But the answer also doesn't say anything factually wrong about anything else.

**Rule:** If the answer contains **any fabricated specific fact** (number, date, name, step) that is not in the retrieved context, **Score B = 0 → final = 0**, even if the rest is correct.

### 4.2 Right answer, wrong reason

**Example:** Q17 "Why might my energy bill be higher than usual this month?"
- Retrieved: high usage on June 10 due to AC running.
- Model answer: "Your bill is high because you have a lot of devices plugged in." (correct fact, but **not from the retrieved context**)

**Rule:** If the answer is **plausible** but **not supported** by what was retrieved, **Score B = 1 at best**, final ≤ 1.

### 4.3 Citation accuracy

**Example:** Model says "According to the Philips Hue manual…", but the actual source was the Nest manual.

**Rule:** A **wrong citation** is worse than **no citation**:
- No citation → Criterion E gets 0 or 1, but other criteria unaffected.
- Wrong citation → Criterion E gets 0, **and** Criterion B (grounding) gets downgraded to 1.

### 4.4 Fallback questions (Q20)

**Example:** Q20 "What is the meaning of life?"
- **Expected behavior:** system returns the fallback message ("I don't have information about that in my knowledge base.").
- **Correct answer:** the fallback message, verbatim or close paraphrase. **Score = 3.**
- **Wrong answer:** the model tries to answer the question with general knowledge ("42", philosophical answer, etc.). **Score = 0.** This is the most important failure mode to catch.
- **Partial answer:** the model says "I don't have specific information about philosophical questions, but I can help with your smart home." → **Score = 2.** (Right intent, slightly off-format.)

### 4.5 Sensor questions with no data

**Example:** Q13 "Was there motion in the kitchen between 2 AM and 3 AM last Tuesday?"
- If the synthetic data does have a row for that time → judge factual correctness against the actual data.
- If the data is missing → expected answer is "No data available for that time range" → that's **Score 3** if the model gives that answer.
- If the model **makes up a "yes" or "no"** without saying it doesn't have data → **Score 0**.

### 4.6 Combined questions (Q17, Q18, Q19)

These **require** evidence from **multiple sources**. If the answer only uses one source and ignores the other:

| Sources used | Scoring guidance |
|--------------|------------------|
| Both sources used well | Score 3 |
| Only the more obvious source used; minor miss | Score 2 |
| Only one source, key part of the question unanswered | Score 1 |
| Sources ignored, answer is generic | Score 0 |

### 4.7 Numeric precision

For Q06 ("What is the maximum wattage…"), Q16 ("average daily energy consumption…"), and similar:

| Tolerance | Score 3 | Score 2 | Score 1 | Score 0 |
|-----------|---------|---------|---------|---------|
| Exact (within 1%) | ✓ | | | |
| Within 5% | | ✓ | | |
| Within 20% | | | ✓ | |
| Wrong by >20% or no number given | | | | ✓ |

**Worked example for Q06:** True answer = 1800 W. Model says 1750 W → within 5% → Score 2. Model says "about 1500 W" → wrong by ~17% → Score 1.

---

## 5. Worked Examples (one per score level)

These examples are taken from real-style answers a small LLM might produce.

### Example 1 — Score 3 (Excellent)

**Question:** Q01 — "How do I reset my Nest thermostat to factory settings?"

**Model answer:**
> "To reset your Nest thermostat to factory settings, according to the Nest manual (page 14):
> 1. Press the ring to open the Quick View menu.
> 2. Choose **Settings**.
> 3. Scroll to **Reset** and select **Factory Reset**.
> 4. Confirm by selecting **Reset** on the next screen.
>
> Note: this will erase all your schedules and learned behavior."

**Judgment:**
| Criterion | Score | Why |
|-----------|-------|-----|
| A — Correctness | 3 | All steps match the manual |
| B — Grounding | 3 | Every claim grounded in the retrieved page |
| C — Completeness | 3 | Includes the warning about losing schedules |
| D — Format | 3 | Numbered list, clean |
| E — Citation | 3 | "according to the Nest manual (page 14)" |

**Final: 3 (Excellent).**

---

### Example 2 — Score 2 (Correct)

**Question:** Q04 — "What is the warranty period for the Nest thermostat?"

**Model answer:**
> "The Nest thermostat comes with a 2-year limited warranty. You can find more details in the warranty section of the documentation."

**Judgment:**
| Criterion | Score | Why |
|-----------|-------|-----|
| A — Correctness | 3 | Correct fact |
| B — Grounding | 3 | Fact is from the manual |
| C — Completeness | 2 | Doesn't specify "limited" warranty type or where to make a claim |
| D — Format | 2 | Readable but the "more details in the documentation" is filler |
| E — Citation | 1 | Vague hint, no specific source |

**Average = (3+3+2+2+1)/5 = 2.2 → 2 (Correct).**

---

### Example 3 — Score 1 (Partially correct)

**Question:** Q11 — "What was the average temperature in the living room this week?"

**Model answer:**
> "The average temperature in the living room this week was around 22 degrees. The living room tends to be comfortable."

**Judgment:**
| Criterion | Score | Why |
|-----------|-------|-----|
| A — Correctness | 2 | Number is plausible but rounded; actual avg from data is 21.4°C → off by ~3% |
| B — Grounding | 1 | Number seems to come from the data but no date range or sample size mentioned |
| C — Completeness | 1 | Doesn't specify which week, doesn't give min/max |
| D — Format | 1 | "Tends to be comfortable" is fluff |
| E — Citation | 0 | No source mentioned |

**Average = (2+1+1+1+0)/5 = 1.0 → 1 (Partially correct).**

---

### Example 4 — Score 0 (Wrong / hallucinated)

**Question:** Q06 — "What is the maximum wattage supported by the TP-Link Kasa smart plug?"

**Model answer:**
> "The TP-Link Kasa smart plug supports up to 3000 watts, making it suitable for heavy appliances like space heaters."

**Judgment:**
| Criterion | Score | Why |
|-----------|-------|-----|
| A — Correctness | 0 | True answer is 1800 W; 3000 W is wrong (and dangerous) |
| B — Grounding | 0 | Number invented |

**Final: 0 (Wrong) — hard floor from Criterion A and B.**

---

### Example 5 — Score 3 (Excellent — Fallback)

**Question:** Q20 — "What is the meaning of life?"

**Model answer:**
> "I don't have information about that in my knowledge base. I can help with questions about your smart home devices, manuals, and sensor data."

**Judgment:**
| Criterion | Score | Why |
|-----------|-------|-----|
| A — Correctness | 3 | Correctly refused |
| B — Grounding | 3 | Nothing invented |
| C — Completeness | 3 | Explains what it *can* help with |
| D — Format | 3 | Clean |
| E — Citation | 3 | Implicit: "I don't have information" is the citation |

**Final: 3 (Excellent).**

---

### Example 6 — Score 0 (Fallback broken)

**Question:** Q20 — "What is the meaning of life?"

**Model answer:**
> "The meaning of life is a philosophical question with many answers. Some say it's 42, others say it's to find happiness. You should pursue what brings you joy."

**Judgment:**
| Criterion | Score | Why |
|-----------|-------|-----|
| A — Correctness | 0 | Tried to answer an out-of-domain question |
| B — Grounding | 0 | Used general pretraining knowledge, not the knowledge base |

**Final: 0 (Wrong).**

---

## 6. How to Apply the Rubric in Practice

### 6.1 For the student (single judge)

1. Open `reports/eval_<model>_<date>.csv` (output of `scripts/eval.py`).
2. For each row (question + answer + auto-score):
   - Re-read the question and the **expected answer hint** in `docs/evaluation/gold_set.md`.
   - Open the **retrieved context** for that question (logged in `data/eval_logs/<q_id>.json`).
   - Score the answer using this rubric.
   - Record the score in a new column `human_score` in the CSV.
3. Total time: ~3–5 seconds per question. For 20 questions × 3 models = 60 judgments ≈ 5–10 minutes.

### 6.2 For advisor + student (two judges, recommended)

- Each judge scores independently.
- Where scores disagree by 1 point (e.g., 2 vs 3) → discuss and agree on final score.
- Where scores disagree by ≥ 2 points (e.g., 0 vs 2) → re-read the rubric together; the disagreement usually reveals a rubric gap.
- Report **Cohen's kappa** as an inter-rater reliability metric in the final report (target: κ ≥ 0.7).

### 6.3 Scoring sheet template

Create `reports/human_scoring_<model>_<date>.csv` with these columns:

```
question_id, query, model_answer, auto_score, A_correctness,
B_grounding, C_completeness, D_format, E_citation,
final_score, judge_notes
```

---

## 7. Reporting Results

After scoring, compute and report in the final report:

| Metric | How to compute | What it means |
|--------|----------------|---------------|
| **Average auto-score** | Mean of `auto_score` column | Cheap proxy for keyword overlap |
| **Average human-score** | Mean of `final_score` column | True quality |
| **Auto–human correlation** | Pearson r between auto and human | Does the auto score actually track quality? |
| **Score per category** | Mean human-score grouped by `category` | Where does each model win/lose? |
| **Score per difficulty** | Mean human-score grouped by `difficulty` | Does the model scale with difficulty? |
| **Fallback pass rate** | For Q20, fraction of models scoring ≥ 2 | Did the model refuse? |
| **Hallucination rate** | Fraction of answers scoring 0 on B | How often does the model invent? |

A clean way to present this:

```
              Auto  Human  A   B   C   D   E   Final
TinyLlama     0.45  0.95   1.1 1.2 1.0 1.3 0.5  1.02
Llama 3.2 3B  0.62  1.40   1.7 1.6 1.5 1.7 0.9  1.48
Phi-3 Mini    0.78  1.85   2.2 2.1 1.9 2.1 1.6  1.98
```

---

## 8. Common Pitfalls to Avoid (when judging)

1. **Don't reward length.** A long, fluffy answer with one fact buried in it is not "Score 3". Penalize fluff in Criterion D.
2. **Don't punish a model for not citing** if it has no good way to do so. But always reward it when it does (Criterion E).
3. **Don't be lenient on hallucinations.** A single fabricated number = Score 0, full stop. This is the whole point of RAG.
4. **Be strict on the fallback.** Q20 is the canary. If a model fails Q20, the rest of its answers are suspect (it might be hallucinating everywhere).
5. **Judge each model independently.** Don't let a "good" answer from Phi-3 influence how you score the same question from TinyLlama.
6. **Use the retrieved context as ground truth, not your own knowledge.** If the model says "1800 W" and the spec actually says "1500 W", that's the spec's fault, not the model's. But if the spec says 1800 and the model says 3000, that's the model's fault.

---

## 9. Open Questions for the Student

| # | Question | Default I'll go with |
|---|----------|----------------------|
| Q1 | Use 3-point (0/1/2) or 4-point (0/1/2/3) scale? | **4-point.** More granularity, helps the final report. |
| Q2 | One judge (student) or two (student + advisor)? | **Two if time allows.** Cohen's kappa is a nice metric to report. |
| Q3 | Strict (any fabrication = 0) or lenient (deduct 1 point)? | **Strict.** This is a RAG system; grounding is the whole point. |
| Q4 | Should numeric tolerance be 1%, 5%, or 20%? | **5% for "Correct", 20% for "Partial"** as in Section 4.7. |
| Q5 | How should the rubric handle non-English answers? | **N/A for v1.** All questions are English by design. |

---

## 10. Document Approval

| Role | Name | Approval | Date |
|------|------|----------|------|
| Student | Marajul Haque | ⏳ pending | |
| Advisor | Abu Nowshed Chy | (not required for v1) | |

---

*End of Scoring Rubric v1. Used by Phase 5.5 of the canonical roadmap.*
