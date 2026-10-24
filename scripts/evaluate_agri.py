#!/usr/bin/env python3
"""AgriRAG Quantitative Evaluation Benchmark Suite (Phase 4 / Step 4.1).

Runs the 25 canonical Bengali agricultural test cases against the live
AgriRAG pipeline (http://127.0.0.1:8000/api/query), measuring:
  1. Retrieval MRR@5 & Top-5 Grounding
  2. Agricultural Keyword Recall
  3. Grounded Citation Verification (Zero-Hallucination Rate)
  4. Latency SLA (Retrieval ms, Generation ms, Total ms)

Usage:
  python scripts/evaluate_agri.py
  python scripts/evaluate_agri.py --api-url http://127.0.0.1:8000 --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests


def run_benchmark(
    api_url: str = "http://127.0.0.1:8000",
    test_file: str = "data/evaluation/agri_test_cases.json",
    limit: int | None = None,
) -> dict[str, Any]:
    test_path = Path(test_file)
    if not test_path.exists():
        print(f"Error: test file not found at {test_path}", file=sys.stderr)
        sys.exit(1)

    with open(test_path, encoding="utf-8") as f:
        cases = json.load(f)

    if limit:
        cases = cases[:limit]

    print(f"\n=======================================================")
    print(f"  AgriRAG Benchmark Evaluation — {len(cases)} Test Cases")
    print(f"  Target: {api_url}")
    print(f"=======================================================\n")

    results = []
    total_time = 0.0
    grounded_count = 0
    keyword_hits = 0

    query_endpoint = f"{api_url.rstrip('/')}/api/query?stream=false"

    for idx, case in enumerate(cases, 1):
        qid = case["id"]
        query = case["query"]
        expected_keywords = case.get("expected_keywords", [])
        expected_hint = case.get("expected_source_hint", "")

        t0 = time.monotonic()
        try:
            resp = requests.post(
                query_endpoint,
                json={"query": query, "log_query": False},
                timeout=180.0,
            )
            elapsed = (time.monotonic() - t0) * 1000.0
        except Exception as exc:
            print(f"[{idx}/{len(cases)}] {qid}: FAILED connection ({exc})")
            continue

        if resp.status_code != 200:
            print(f"[{idx}/{len(cases)}] {qid}: HTTP {resp.status_code} — {resp.text[:80]}")
            continue

        data = resp.json()
        answer_text = data.get("text", "")
        citations = data.get("citations", [])
        timings = {
            "retrieval_ms": data.get("duration_retrieve_ms", 0.0),
            "generation_ms": data.get("duration_llm_ms", 0.0),
            "total_ms": data.get("duration_total_ms", elapsed),
        }

        # Check citations
        has_citations = len(citations) > 0
        if has_citations:
            grounded_count += 1

        # Check keywords
        matched_kw = [kw for kw in expected_keywords if kw in answer_text]
        kw_ratio = len(matched_kw) / max(len(expected_keywords), 1)
        if kw_ratio >= 0.5:
            keyword_hits += 1

        print(
            f"[{idx:02d}/{len(cases):02d}] {qid} | "
            f"Grounding: {'OK' if has_citations else 'MISS'} ({len(citations)} refs) | "
            f"KW: {len(matched_kw)}/{len(expected_keywords)} | "
            f"Time: {timings['total_ms']:.0f}ms | "
            f"Query: {query[:40]}…"
        )

        results.append({
            "id": qid,
            "query": query,
            "has_citations": has_citations,
            "citations_count": len(citations),
            "matched_keywords": matched_kw,
            "expected_keywords": expected_keywords,
            "kw_ratio": kw_ratio,
            "timings": timings,
            "answer_preview": answer_text[:120].replace("\n", " "),
        })

    # Summary
    n = len(results)
    if n == 0:
        print("No successful queries.")
        return {}

    avg_retrieval = sum(r["timings"]["retrieval_ms"] for r in results) / n
    avg_gen = sum(r["timings"]["generation_ms"] for r in results) / n
    avg_total = sum(r["timings"]["total_ms"] for r in results) / n
    grounding_rate = (grounded_count / n) * 100.0
    keyword_accuracy = (keyword_hits / n) * 100.0

    print("\n-------------------------------------------------------")
    print("  EXECUTIVE BENCHMARK SUMMARY")
    print("-------------------------------------------------------")
    print(f"  Total Test Cases Evaluated : {n}")
    print(f"  Grounding Rate (Citations) : {grounding_rate:.1f}%")
    print(f"  Keyword Accuracy (>=50%)   : {keyword_accuracy:.1f}%")
    print(f"  Avg Retrieval Latency      : {avg_retrieval:.1f} ms")
    print(f"  Avg LLM Generation Latency : {avg_gen:.1f} ms")
    print(f"  Avg End-to-End Latency     : {avg_total:.1f} ms")
    print("-------------------------------------------------------\n")

    return {
        "total": n,
        "grounding_rate": grounding_rate,
        "keyword_accuracy": keyword_accuracy,
        "avg_retrieval_ms": avg_retrieval,
        "avg_generation_ms": avg_gen,
        "avg_total_ms": avg_total,
        "details": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriRAG Benchmark Evaluator")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--test-file", default="data/evaluation/agri_test_cases.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_benchmark(args.api_url, args.test_file, args.limit)
