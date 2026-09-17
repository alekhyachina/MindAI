"""
eval_live.py — DeepEval against an ALREADY-INGESTED repo.

eval.py's build_test_cases_from_live_pipeline() re-ingests the repository
first, which costs minutes and a full round of embedding calls. Once a repo
is already indexed there is nothing to re-ingest: this runs the same three
metrics against the live graph using the existing Qdrant collection, so the
numbers describe the deployed retrieval path rather than mock fixtures.

Usage:
    python eval_live.py                     # default collection below
    python eval_live.py <collection_name>
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from deepeval import evaluate  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from eval import build_metrics  # noqa: E402
import graph as G  # noqa: E402

# Event-Driven-Inventory-Analytics-Warehouse-BI-System — 705 chunks, the
# largest indexed repo in this database, so retrieval has real choices to
# make rather than returning nearly everything.
DEFAULT_COLLECTION = "repo_60ef819d-8e97-4f29-addc-6bfd269c11b9"

# Questions span the shapes the pipeline handles differently: a broad summary
# question (no single chunk is "the" match), specific factual lookups, a
# named-file question (exercises the filename boost), and a trace-shaped one
# (can route through graph_expand). expected_output feeds ContextualRecall,
# which measures whether retrieval surfaced what the answer needed.
QUESTIONS = [
    {
        "input": "What does this project do?",
        "expected_output": (
            "It is an event-driven inventory analytics warehouse for medical and "
            "pharmaceutical operations, tracking inventory events and serving analytics."
        ),
    },
    {
        "input": "What database is used?",
        "expected_output": "PostgreSQL, accessed via psycopg2 connections.",
    },
    {
        "input": "How are inventory events processed?",
        "expected_output": (
            "Events flow from the web application into a pipeline and are processed "
            "in batches by Spark jobs before landing in the warehouse."
        ),
    },
    {
        "input": "What is in requirements.txt?",
        "expected_output": "A list of Python dependencies for the project.",
    },
    {
        "input": "What tables exist in the data warehouse schema?",
        "expected_output": "Fact and dimension tables defined by the warehouse schema.",
    },
]


def main() -> None:
    collection = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_COLLECTION

    url = os.environ.get("QDRANT_URL", "").strip()
    if url:
        client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY") or None, timeout=60)
        mode = "server"
    else:
        client = QdrantClient(path=os.environ.get("MINDAI_QDRANT_PATH", "./qdrant_storage"))
        mode = "local"

    points = client.get_collection(collection).points_count
    print(f"collection : {collection}")
    print(f"mode       : {mode} | {points} points")
    print(f"questions  : {len(QUESTIONS)}")
    print()

    # Build deps and the compiled graph once — reloading the embedding and
    # reranker models per question would dominate the runtime.
    deps = G.PipelineDependencies(client)
    compiled = G.build_graph(deps)

    test_cases: list[LLMTestCase] = []
    refusals = 0
    for q in QUESTIONS:
        state = G.answer_question(
            q["input"], client, {}, collection_name=collection,
            deps=deps, compiled_graph=compiled,
        )
        answer = state.get("answer", "")
        if state.get("refused"):
            refusals += 1
        chunks = [*state.get("reranked_chunks", []), *state.get("expanded_chunks", [])]
        context = [
            f"{c['file_path']}:{c['start_line']}-{c['end_line']}\n{c['content']}"
            for c in chunks
        ]
        print(f"  {'REFUSED' if state.get('refused') else 'ANSWERED'} | {q['input']}")
        test_cases.append(
            LLMTestCase(
                input=q["input"],
                actual_output=answer,
                expected_output=q["expected_output"],
                retrieval_context=context or ["<no context retrieved>"],
            )
        )

    print()
    print(f"refusals: {refusals}/{len(QUESTIONS)}")
    print()

    # DeepEval's evaluate() fires every metric for every test case
    # concurrently — with 3 metrics over 5 cases, each making several judge
    # calls, that is ~45 requests in one burst and OpenAI answers with a 429.
    # Measuring one metric at a time, with a pause between test cases, keeps
    # the run inside the rate limit. Slower, but it produces scores instead
    # of a RetryError.
    import time as _time

    metrics = build_metrics()
    totals: dict[str, list[float]] = {}

    for i, case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] {case.input}")
        for metric in metrics:
            name = type(metric).__name__.replace("Metric", "")
            for attempt in range(4):
                try:
                    metric.measure(case)
                    totals.setdefault(name, []).append(metric.score)
                    flag = "PASS" if metric.score >= metric.threshold else "FAIL"
                    print(f"    {name:<17} {metric.score:.2f}  {flag}")
                    break
                except Exception as exc:  # rate limit or transient judge error
                    if attempt == 3:
                        print(f"    {name:<17}  error: {type(exc).__name__}")
                    else:
                        _time.sleep(20 * (attempt + 1))
            _time.sleep(3)
        print()

    print("=" * 58)
    print(f"{'METRIC':<20}{'MEAN':>8}{'THRESHOLD':>12}{'VERDICT':>12}")
    print("=" * 58)
    for metric in metrics:
        name = type(metric).__name__.replace("Metric", "")
        vals = totals.get(name, [])
        if not vals:
            print(f"{name:<20}{'n/a':>8}")
            continue
        mean = sum(vals) / len(vals)
        print(f"{name:<20}{mean:>8.3f}{metric.threshold:>12.2f}"
              f"{('PASS' if mean >= metric.threshold else 'FAIL'):>12}")
    print("=" * 58)


if __name__ == "__main__":
    main()
