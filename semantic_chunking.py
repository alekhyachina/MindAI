"""
semantic_chunking.py — Embedding-based ("semantic") chunk-boundary finder,
an alternative to ingestion.py's tree-sitter AST chunking.

Architectural note: mirrors dense_embeddings.py / reranker.py — a separate,
swappable provider module rather than logic inlined into ingestion.py, so
the chunking strategy is one env var away from being switched back (see
MINDAI_CHUNK_STRATEGY in ingestion.py).

What it does: instead of cutting at syntactic boundaries, it embeds small
consecutive slices of a file with the SAME OpenAI model used for retrieval
(text-embedding-3-small, via DenseEmbedder), measures cosine distance
between neighbouring slices, and cuts where that distance spikes — i.e.
where the file changes topic. This is the standard "semantic chunking"
technique popularised for prose.

This module deliberately returns LINE SPANS, not CodeChunk objects. Two
reasons: it keeps ingestion.py the single place that knows how a CodeChunk
is built (chunk_id derivation, call extraction, symbol naming), and it
avoids a circular import, since CodeChunk lives in ingestion.py which
imports this module. Spans are 1-indexed, inclusive, contiguous, and cover
every line of the file — so the start_line/end_line that graph.py cites
remain exact regardless of where the boundaries land.

Known tradeoffs vs AST chunking, recorded because they are inherent to the
technique rather than bugs to be fixed:
  - Boundaries can fall mid-function, so a retrieved chunk may contain a
    partial definition where AST chunking would have returned the whole one.
  - It costs an extra embedding pass over the entire repository at ingest
    time (one request batch per ~100 slices) on top of embedding the final
    chunks, roughly doubling ingestion's OpenAI call volume.
  - Semantic distance is computed on code text, whose token distribution is
    far more uniform than prose, so distance spikes are weaker signals than
    they are for natural language.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from dense_embeddings import DenseEmbedder

logger = logging.getLogger("mindai.semantic_chunking")

# Lines per embedded slice. One line at a time makes distances extremely
# noisy (a single `}` or blank line reads as a topic change) and multiplies
# API volume; a few lines is enough context for the embedding to be about
# something while still allowing fine-grained boundaries.
UNIT_LINES = 4

# Slices adjacent to the one being embedded are prepended/appended as
# context before embedding. Without this each slice is judged in isolation
# and boundaries land almost arbitrarily; with it, a slice is represented by
# its neighbourhood, which is what makes consecutive distances meaningful.
BUFFER_UNITS = 1

# A cut is made where the distance between neighbouring slices exceeds this
# percentile of all distances in the file. Percentile rather than an absolute
# threshold because absolute cosine distances vary widely by language and
# file — a fixed cutoff that works for Python produces one chunk per file in
# Go and one per slice in JSON.
BREAKPOINT_PERCENTILE = 90

# Guardrails applied after breakpoint detection. Without a floor, a spike
# cluster yields 2-line chunks with no retrievable context; without a
# ceiling, a uniform file (distances all equal, no spikes) becomes one chunk
# covering thousands of lines.
MIN_CHUNK_LINES = 6
MAX_CHUNK_LINES = 120

# Files below this are never worth an embedding round-trip to split — emit
# one span and skip the API call entirely.
MIN_LINES_TO_SPLIT = MIN_CHUNK_LINES * 2

# Best-effort symbol naming. Semantic chunks have no AST node to take a name
# from, so the first declaration-looking line in the chunk is used instead.
# Covers the declaration syntax of the languages in EXTENSION_TO_LANGUAGE;
# a miss just falls back to a positional name, so over-matching is the only
# real risk and these patterns are all anchored to a leading keyword.
_DECLARATION_PATTERN = re.compile(
    r"^\s*(?:export\s+|public\s+|private\s+|protected\s+|static\s+|final\s+|async\s+)*"
    r"(?:def|class|func|function|fn|struct|interface|impl|trait|type|enum|module|"
    r"const|let|var|sub|package)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)"
)


def _build_units(lines: list[str]) -> list[tuple[int, int]]:
    """Consecutive (start_idx, end_idx) 0-indexed half-open slices of `lines`."""
    return [(i, min(i + UNIT_LINES, len(lines))) for i in range(0, len(lines), UNIT_LINES)]


def _unit_text(lines: list[str], units: list[tuple[int, int]], index: int) -> str:
    """The text embedded for unit `index`: itself plus BUFFER_UNITS neighbours."""
    lo = max(0, index - BUFFER_UNITS)
    hi = min(len(units), index + BUFFER_UNITS + 1)
    start = units[lo][0]
    end = units[hi - 1][1]
    return "\n".join(lines[start:end])


def _cosine_distances(vectors: np.ndarray) -> np.ndarray:
    """Distance between each consecutive pair. Length is len(vectors) - 1."""
    # OpenAI returns L2-normalised embeddings, but normalise defensively so a
    # provider change cannot silently turn these into unnormalised dot
    # products (which would still "work" while ranking boundaries wrongly).
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit_vectors = vectors / norms
    similarities = np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1)
    return 1.0 - similarities


def _enforce_size_limits(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Merge spans below MIN_CHUNK_LINES into their predecessor and split spans
    above MAX_CHUNK_LINES, preserving contiguity and full line coverage.
    """
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and (end - start + 1) < MIN_CHUNK_LINES:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    capped: list[tuple[int, int]] = []
    for start, end in merged:
        length = end - start + 1
        if length <= MAX_CHUNK_LINES:
            capped.append((start, end))
            continue
        # Split into near-equal parts rather than MAX-sized parts plus a
        # possibly tiny remainder that would violate MIN_CHUNK_LINES.
        parts = -(-length // MAX_CHUNK_LINES)
        step = -(-length // parts)
        for offset in range(0, length, step):
            capped.append((start + offset, min(start + offset + step - 1, end)))
    return capped


def compute_semantic_spans(source_text: str, embedder: DenseEmbedder) -> list[tuple[int, int]]:
    """
    Return 1-indexed, inclusive (start_line, end_line) spans covering every
    line of `source_text`, cut at points of maximum semantic change.

    Falls back to a single whole-file span for input too small to be worth
    splitting, and to size-capped spans if the embedding call fails — a
    chunking strategy must never lose a file, so every failure path still
    returns full coverage.
    """
    lines = source_text.splitlines()
    if not lines:
        return []

    if len(lines) < MIN_LINES_TO_SPLIT:
        return _enforce_size_limits([(1, len(lines))])

    units = _build_units(lines)
    if len(units) < 2:
        return _enforce_size_limits([(1, len(lines))])

    texts = [_unit_text(lines, units, i) for i in range(len(units))]
    try:
        vectors = np.array(embedder.embed(texts), dtype=np.float32)
    except Exception as e:
        # A failed embedding call must not drop the file from the index.
        logger.warning("semantic chunking embed failed (%s); falling back to fixed spans", e)
        return _enforce_size_limits([(1, len(lines))])

    distances = _cosine_distances(vectors)
    if distances.size == 0:
        return _enforce_size_limits([(1, len(lines))])

    threshold = float(np.percentile(distances, BREAKPOINT_PERCENTILE))
    # Strictly greater: when a file is uniform every distance equals the
    # percentile, and `>=` would cut at every single unit boundary.
    breakpoints = [i for i, d in enumerate(distances) if float(d) > threshold]

    spans: list[tuple[int, int]] = []
    span_start_unit = 0
    for bp in breakpoints:
        # `bp` is the distance between unit bp and bp+1, so the cut falls
        # after unit bp — that unit's last line ends the span.
        spans.append((units[span_start_unit][0] + 1, units[bp][1]))
        span_start_unit = bp + 1
    spans.append((units[span_start_unit][0] + 1, len(lines)))

    return _enforce_size_limits(spans)


def symbol_name_for_span(content: str) -> str | None:
    """
    Best-effort name for a semantic chunk: the first declaration-looking
    identifier in it. Returns None when nothing matches, leaving the caller
    to fall back to a positional name.
    """
    for line in content.splitlines():
        match = _DECLARATION_PATTERN.match(line)
        if match:
            return match.group(1)
    return None
