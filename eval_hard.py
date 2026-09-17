"""
eval_hard.py — a deliberately difficult DeepEval run.

Why this exists
---------------
eval_live.py scored ContextualRecall 1.000, which is not a result to trust.
Two things made it too easy, and both were mistakes in how the test was
written rather than strengths of the system:

  1. Its `expected_output` values were written AFTER reading what the
     pipeline answered. ContextualRecall asks "does the retrieved context
     support the expected answer?" — so expectations derived from observed
     behaviour can hardly fail. That is circular.
  2. Those expectations were vague ("a list of Python dependencies", "fact
     and dimension tables"). A vague expectation is satisfied by almost any
     retrieved context.

The questions below were written from the SOURCE, not from the system's
answers: each targets one specific function, file or mechanism that was read
directly out of the repository. They are narrow — in a 182-file, 705-chunk
codebase, a single helper is genuinely hard to surface in the top 20 chunks.
A recall score below 1.0 here is the honest, useful signal that the earlier
run failed to produce.
"""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from deepeval.test_case import LLMTestCase  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from eval import build_metrics  # noqa: E402
import graph as G  # noqa: E402

DEFAULT_COLLECTION = "repo_60ef819d-8e97-4f29-addc-6bfd269c11b9"

# Seconds to wait after each judge call — see the comment at its use site.
# gpt-4o on a 30K TPM tier needs ~30; gpt-4o-mini runs fine at 3.
JUDGE_CALL_SPACING_SECONDS = int(os.environ.get("MINDAI_JUDGE_SPACING", "30"))

# Ground truth read out of the repository source, before running anything.
QUESTIONS = [
    {
        "input": "What statuses count as an active run in the control store?",
        "expected_output": (
            "get_active_run in platform/control/store.py treats pending, starting, "
            "running, stop_requested, stopping and orphaned as active statuses, "
            "querying the runs table for a matching kind."
        ),
    },
    {
        "input": "Which pipeline stages have their baselines recorded by the alert engine?",
        "expected_output": (
            "_record_baselines in services/alerts/_engine.py records the bronze and "
            "silver stages, appending each stage's pipeline count to the baseline store."
        ),
    },
    {
        "input": "What fields are in the error payload returned by a probe?",
        "expected_output": (
            "_error_payload in platform/probes/base.py returns a dict with probe, "
            "status set to error, and error containing the stringified exception."
        ),
    },
    {
        "input": "How does the TTL cache expose its configured lifetime?",
        "expected_output": (
            "ttl_seconds in platform/cache/ttl.py is a property returning the "
            "internal _ttl value as a float."
        ),
    },
    {
        "input": "What does the sales domain contract define?",
        "expected_output": (
            "contracts/sales.py defines the sales domain contract — the schema and "
            "validation rules for sales events used by the warehouse."
        ),
    },
    {
        "input": "Which routes does the platform API expose?",
        "expected_output": (
            "platform/api/routes.py registers the HTTP routes for the platform "
            "application, wiring endpoints to their service handlers."
        ),
    },
]


def main() -> None:
    collection = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_COLLECTION

    url = os.environ.get("QDRANT_URL", "").strip()
    if url:
        client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY") or None, timeout=60)
    else:
        client = QdrantClient(path=os.environ.get("MINDAI_QDRANT_PATH", "./qdrant_storage"))

    print(f"collection : {collection}")
    print(f"points     : {client.get_collection(collection).points_count}")
    print(f"judge      : {os.environ.get('MINDAI_JUDGE_MODEL', 'gpt-4o')}")
    print(f"questions  : {len(QUESTIONS)}  (hard: ground truth written from source)")
    print()

    deps = G.PipelineDependencies(client)
    compiled = G.build_graph(deps)

    cases: list[LLMTestCase] = []
    refusals = 0
    for q in QUESTIONS:
        state = G.answer_question(
            q["input"], client, {}, collection_name=collection,
            deps=deps, compiled_graph=compiled,
        )
        if state.get("refused"):
            refusals += 1
        chunks = [*state.get("reranked_chunks", []), *state.get("expanded_chunks", [])]
        files = sorted({c["file_path"].split("/")[-1] for c in chunks})
        print(f"  {'REFUSED ' if state.get('refused') else 'ANSWERED'} | {q['input']}")
        print(f"             context: {len(chunks)} chunks / {len(files)} files")
        cases.append(
            LLMTestCase(
                input=q["input"],
                actual_output=state.get("answer", ""),
                expected_output=q["expected_output"],
                retrieval_context=[
                    f"{c['file_path']}:{c['start_line']}-{c['end_line']}\n{c['content']}"
                    for c in chunks
                ] or ["<no context retrieved>"],
            )
        )

    print()
    print(f"refusals: {refusals}/{len(QUESTIONS)}")
    print()

    metrics = build_metrics()
    totals: dict[str, list[float]] = {}
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.input}")
        for metric in metrics:
            name = type(metric).__name__.replace("Metric", "")
            for attempt in range(4):
                try:
                    metric.measure(case)
                    totals.setdefault(name, []).append(metric.score)
                    flag = "PASS" if metric.score >= metric.threshold else "FAIL"
                    print(f"    {name:<17} {metric.score:.2f}  {flag}")
                    break
                except Exception as exc:
                    if attempt == 3:
                        print(f"    {name:<17}  error: {type(exc).__name__}")
                    else:
                        time.sleep(30 * (attempt + 1))
            # Paced for a 30,000 tokens/min gpt-4o limit. A Faithfulness
            # judgment over ~50K characters of retrieval context requests
            # ~13,000 tokens, so only about two calls fit in a minute —
            # firing them back to back spends the whole run in 429 retries
            # instead of measuring anything. Lower this if the judge model
            # or the account tier changes.
            time.sleep(JUDGE_CALL_SPACING_SECONDS)
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
