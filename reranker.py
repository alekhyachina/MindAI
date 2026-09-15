"""
reranker.py — Reranking provider, used by graph.py's rerank_node.

Architectural note: mirrors dense_embeddings.py's pattern for the same
reason — reranking (cross-encoder scoring of query<->chunk pairs) is CPU-
bound locally via sentence-transformers' CrossEncoder, which becomes the
dominant per-question latency once dense embedding was moved off-CPU to
OpenAI's API. Cohere's hosted Rerank API replaces that local CPU work
with a network call, the same tradeoff already made for dense embeddings.

Kept as a separate, swappable module (not inlined into graph.py) so the
local CrossEncoder path can be restored by changing one import if Cohere
is ever unavailable/undesired — same reasoning as dense_embeddings.py
being its own module rather than folded into ingestion.py/graph.py.
"""

from __future__ import annotations

import os

import cohere

RERANK_MODEL_NAME = "rerank-v3.5"


class CohereReranker:
    """
    Same call shape as sentence_transformers.CrossEncoder.predict(pairs) —
    graph.py's rerank_node calls .predict(pairs) and expects back a
    sequence of floats in the SAME ORDER as the input pairs. This class
    preserves that contract so rerank_node's surrounding logic (sorting,
    top-N selection, top_confidence) needed no changes beyond swapping
    which reranker object it holds.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._client = cohere.ClientV2(api_key=api_key or os.environ.get("MINDAI_COHERE_API_KEY"))

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        `pairs` is a list of (query, document) tuples, matching
        CrossEncoder's convention — but Cohere's rerank API takes ONE
        query plus a list of documents, since a rerank call is inherently
        "one query against many candidates," not many independent pairs.
        All pairs here share the same query (graph.py always builds them
        that way — one query reranked against N candidate chunks), so
        this asserts that invariant explicitly rather than silently
        reranking against the wrong query if it's ever violated.
        """
        if not pairs:
            return []

        query = pairs[0][0]
        if any(p[0] != query for p in pairs):
            raise ValueError(
                "CohereReranker.predict() received pairs with different queries; "
                "Cohere's rerank API reranks one query against many documents, not "
                "arbitrary (query, doc) pairs — this indicates a caller bug."
            )
        documents = [doc for _query, doc in pairs]

        response = self._client.rerank(
            model=RERANK_MODEL_NAME,
            query=query,
            documents=documents,
            top_n=len(documents),  # score every candidate; graph.py does its own top-N slicing
        )

        # Cohere returns results sorted by relevance with each result
        # carrying its ORIGINAL index into `documents` — reassemble into
        # the input order so the CrossEncoder-compatible contract (one
        # score per input pair, same order) holds.
        scores = [0.0] * len(documents)
        for result in response.results:
            scores[result.index] = result.relevance_score
        return scores
