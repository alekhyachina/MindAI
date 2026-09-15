"""
dense_embeddings.py — Dense (semantic) vector provider, shared by
ingestion.py (upsert-time) and graph.py (query-time).

Architectural note: the hybrid retrieval design (dense + sparse, fused via
RRF — see graph.py) is unchanged. Only WHERE the dense vector comes from
changes: local CPU inference (FastEmbed) is slow enough to make ingestion
take minutes on ordinary hardware, so dense embedding moved to OpenAI's
hosted API (network round-trip, seconds instead of minutes, no local
compute). Sparse/BM25 embedding stays on FastEmbed — it's lexical term
statistics, not a neural model, so it was already fast locally and moving
it would only add an API dependency for no speed benefit.

This means every existing Qdrant collection embedded with FastEmbed's
384-dim bge-small-en-v1.5 is INCOMPATIBLE with the new 1536-dim OpenAI
vectors — same reason any embedding-model change always forces
re-ingestion. There is no in-place migration; re-ingest affected repos.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

DENSE_MODEL_NAME = "text-embedding-3-small"
DENSE_VECTOR_SIZE = 1536

# OpenAI's embeddings endpoint accepts up to 2048 inputs per request, but
# batching more conservatively keeps individual request/response payloads
# reasonable and lets multiple batches run concurrently (see below) instead
# of one giant request holding up the whole ingestion on its own latency.
_EMBED_REQUEST_BATCH_SIZE = 100

# Each embed() request is pure network I/O (waiting on OpenAI's response),
# so several can be in flight at once without fighting over CPU — this is
# the main lever for ingestion speed once local compute is out of the
# picture. 8 is comfortably under OpenAI's per-org rate limits for this
# endpoint at typical usage tiers.
_MAX_CONCURRENT_REQUESTS = 8


class DenseEmbedder:
    """
    Thin wrapper so callers (ingestion.py, graph.py) don't need to know
    which provider is behind dense embedding — same shape as FastEmbed's
    TextEmbedding.embed(): pass texts in, get one vector (list[float]) per
    text back, in the same order.
    """

    def __init__(self, api_key: str | None = None) -> None:
        # Deliberately its own client, not graph.py's LLM client — that one
        # may point at Gemini's OpenAI-compatibility endpoint (see graph.py's
        # LLM_BASE_URL), which does not serve OpenAI's embedding models.
        # Dense embeddings always go straight to OpenAI's own API.
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=DENSE_MODEL_NAME, input=batch)
        # API guarantees response.data is returned in the same order as the
        # input batch, but sort by index defensively rather than trust
        # ordering alone.
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Splits into request-sized batches and fires them concurrently
        (bounded by _MAX_CONCURRENT_REQUESTS) rather than one request at a
        time — each request is I/O-bound network wait, so overlapping them
        cuts wall-clock time roughly in proportion to the concurrency level
        instead of paying every batch's latency serially.
        """
        if not texts:
            return []
        batches = [texts[i : i + _EMBED_REQUEST_BATCH_SIZE] for i in range(0, len(texts), _EMBED_REQUEST_BATCH_SIZE)]
        if len(batches) == 1:
            return self._embed_batch(batches[0])

        with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_REQUESTS) as pool:
            # map() preserves input order in its output order, so results
            # line up with `batches` without needing to track futures by hand.
            batch_results = list(pool.map(self._embed_batch, batches))

        vectors: list[list[float]] = []
        for batch_vectors in batch_results:
            vectors.extend(batch_vectors)
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
