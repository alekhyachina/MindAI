"""
graph.py — LangGraph execution pipeline for MindAI.

State machine:
    query_rewrite -> hybrid_retrieve -> rerank -> [conditional: graph_expand?] -> generate

Architectural notes:
  - State is a TypedDict threaded through every node (LangGraph's durable
    state pattern). Each node reads only what it needs and returns only the
    keys it updates; LangGraph merges partial updates into the running state.
  - Retrieval fuses dense + sparse hits via Reciprocal Rank Fusion (RRF)
    rather than a naive score blend, because dense cosine similarity and
    BM25 scores live on incomparable scales — RRF sidesteps that by fusing
    on rank position instead of raw score.
  - Graph expansion is a genuine conditional edge (not a always-run step):
    it only fires when the query looks like a trace/call-chain question
    AND the reranked top score suggests the vector-only context may be
    insufficient. This keeps the common case (single-hop Q&A) cheap.
  - Generation is the single hard gate against hallucination: the system
    prompt mandates `path:line` citations for every factual claim, and a
    post-generation regex check strips/rejects claims that don't carry a
    citation matching a chunk actually present in context. If reranked
    confidence is too low, the model is instructed (and structurally
    nudged via the prompt) to refuse rather than guess.
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, TypedDict

import tiktoken
import torch
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import CrossEncoder

from dense_embeddings import DenseEmbedder
from ingestion import COLLECTION_NAME, DENSE_VECTOR_NAME, SPARSE_MODEL_NAME, SPARSE_VECTOR_NAME
from reranker import CohereReranker

try:
    from langgraph.graph import END, StateGraph
except ImportError as e:
    raise ImportError("langgraph is required: pip install -r requirements.txt") from e

logger = logging.getLogger("mindai.graph")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

load_dotenv()  # populates os.environ from a local .env file, if present

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

GENERATION_MODEL = os.environ.get("MINDAI_GENERATION_MODEL", "gpt-4o-mini")
REWRITE_MODEL = os.environ.get("MINDAI_REWRITE_MODEL", "gpt-4o-mini")
# Set to swap providers without touching code: leave unset for OpenAI's
# default endpoint, or point at Gemini's OpenAI-compatibility endpoint
# (https://generativelanguage.googleapis.com/v1beta/openai/) with a
# Gemini API key and Gemini model names (e.g. gemini-2.0-flash) in
# MINDAI_GENERATION_MODEL / MINDAI_REWRITE_MODEL. The openai SDK's request/
# response shape is unchanged either way — only the base URL and key differ.
LLM_BASE_URL = os.environ.get("MINDAI_LLM_BASE_URL") or None
LLM_API_KEY = os.environ.get("MINDAI_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Reranking is the one remaining CPU-bound step in the pipeline (dense
# embedding was already moved to OpenAI's API — see dense_embeddings.py's
# docstring for that same tradeoff). Set MINDAI_COHERE_API_KEY to switch
# from the local sentence-transformers CrossEncoder to Cohere's hosted
# Rerank API — same reasoning, same swap pattern, see reranker.py.
USE_COHERE_RERANKER = bool(os.environ.get("MINDAI_COHERE_API_KEY"))

# Gemini 3.x models "think" before answering by default, and that
# reasoning silently consumes the same max_tokens budget as the visible
# completion — verified directly: at max_tokens=100 the query-rewrite call
# returned pure garbage (finish_reason="length") because ~400 tokens of
# invisible reasoning ate the whole budget before any answer text
# appeared. reasoning_effort="minimal" is Gemini's own lever to shrink
# that reasoning pass (lower latency/cost, not just a truncation
# workaround); it isn't a valid OpenAI Chat Completions parameter, so it's
# only sent when the configured model is actually a Gemini model.
_IS_GEMINI_MODEL = GENERATION_MODEL.startswith("gemini-") or REWRITE_MODEL.startswith("gemini-")

RRF_K = 60  # standard RRF damping constant
# Both raised from 50/10 after a real miss: "which algorithms are used
# here" on a repo whose answer lived in src/models/train.py (imports
# StackingRegressor, LGBMRegressor — real, well-chunked Python code) never
# made it into the top 10 reranked chunks, because the code doesn't
# literally contain the word "algorithm" and both dense/sparse search
# ranked dozens of other chunks above it out of 50 candidates. Wider nets
# at both stages give the actually-relevant chunk more chances to surface
# without changing how retrieval or reranking work — more candidates in,
# more chunks kept, same RRF fusion and cross-encoder scoring.
FUSED_CANDIDATE_COUNT = 100  # candidates pulled from each of dense/sparse before fusion
RERANK_TOP_N = 20  # chunks kept after cross-encoder reranking
CONTEXT_TOKEN_BUDGET = 8000  # hard cap on tokens spent on retrieved context; raised to fit RERANK_TOP_N=20

# --- History of this gate (read before changing MIN_* constants again) ---
# v1: refuse whenever top cross-encoder score < 0.15. Broke on broad
#   questions ("tell me about this repo") — no single chunk is "the best
#   match" for a summary question, so top_confidence lands near zero even
#   when retrieval is working correctly.
# v2: detect "broad question" via keyword regex, relax the bar only for
#   matches. Broke immediately — real phrasing ("tell about the content
#   of this project") doesn't match any fixed keyword list. Phrasing was
#   never the right signal.
# v3: replaced regex with "dense/sparse agreement" (chunks two independent
#   search algorithms both surfaced) as a phrasing-independent proxy.
#   Broke too, once measured across enough real questions: on one actual
#   repo, six consecutive real questions — several of which the model
#   answered CORRECTLY — produced top_confidence of 0.02, 0.003, 0.016,
#   0.005, and 0.0003, and dense_sparse_agreement bouncing between 1 and
#   23 based on minor query-rewrite wording differences with no relation
#   to whether the question was actually answerable. Neither the
#   cross-encoder score nor the agreement count reliably separates "good
#   retrieval" from "bad retrieval" for this reranker model on code
#   content — ms-marco-MiniLM-L-6-v2 was trained on web search
#   query/passage pairs, not code, and appears to score most code chunks
#   near zero regardless of true relevance.
#
# v4 (current): stop gating generation on any reranker/retrieval SCORE.
# Gate only on retrieval having found candidates at all (dense_hits or
# sparse_hits non-empty — reliably true whenever the repo has indexed
# content even remotely related to the query). The real anti-hallucination
# backstop is _validate_citations() downstream: every claim in the answer
# must cite a path that was actually in the retrieved context, checked
# after generation, not guessed at beforehand via an upstream score. A
# score-based pre-filter was trying to predict whether the LLM COULD
# ground its answer; checking citations after generation confirms whether
# it DID — a strictly stronger and more direct guarantee, and immune to
# any particular reranker model's score calibration on any particular
# content domain.

# Heuristic trigger words for "trace"-shaped questions (call chains, usage).
TRACE_QUERY_PATTERNS = re.compile(
    r"\b(calls?|callers?|callees?|invoke[sd]?|who\s+uses|used\s+by|call\s+chain|"
    r"trace|dependency|dependents?|depends?\s+on|invoked\s+by)\b",
    re.IGNORECASE,
)

CITATION_PATTERN = re.compile(r"([\w./\\-]+\.\w+):(\d+)(?:-(\d+))?")
# Safety-net pattern for near-miss formats the model occasionally produces
# despite the system prompt's explicit correct/incorrect examples — e.g.
# `README[1-100]` instead of `README.md:1-100`. Deliberately does NOT
# require a file extension (`README` alone is common) since this pattern
# exists to rescue answers that are actually well-grounded but slightly
# malformed, not to be the primary citation contract — see
# _validate_citations for how the two patterns are combined.
_LENIENT_CITATION_PATTERN = re.compile(r"\b([\w./\\-]+)\[(\d+)(?:-(\d+))?\]")

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text, disallowed_special=()))


# --------------------------------------------------------------------------- #
# State definition
# --------------------------------------------------------------------------- #

class RetrievedChunk(TypedDict):
    chunk_id: str
    file_path: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    content: str
    score: float


class GraphState(TypedDict, total=False):
    # Input
    raw_query: str
    collection_name: str
    call_graph: dict[str, dict[str, list[str]]]

    # query_rewrite output
    search_query: str

    # hybrid_retrieve output
    fused_candidates: list[RetrievedChunk]
    dense_sparse_agreement: int  # chunks both dense and sparse search surfaced independently

    # rerank output
    reranked_chunks: list[RetrievedChunk]
    # Top reranker score. NOT a calibrated probability — 0.7 does not mean
    # "70% confident". It is a relative retrieval-quality signal from one
    # specific reranker model, only meaningful compared against other scores
    # from that same model, which is why it is used solely as a threshold for
    # graph expansion and never surfaced to the user as a confidence figure.
    top_confidence: float

    # graph_expand output (optional — only set if the conditional edge fires)
    expanded_chunks: list[RetrievedChunk]

    # generate output
    answer: str
    citations_valid: bool
    refused: bool
    retry_count: int  # 1 if generation was re-attempted after a validation failure


# --------------------------------------------------------------------------- #
# Pipeline dependencies (constructed once, threaded through node closures)
# --------------------------------------------------------------------------- #

class PipelineDependencies:
    """
    Bundles the stateful clients/models each node needs. Constructed ONCE
    per process (not per repo, not per request) and passed into
    `build_graph`, avoiding the cost of re-instantiating embedding/reranker
    models — each hundreds of MBs — on every question asked. The Qdrant
    collection to query and the call graph to expand against are per-repo,
    so they travel through GraphState instead (see `answer_question`),
    letting one PipelineDependencies + compiled graph serve any number of
    ingested repos across a long-running server process.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        openai_api_key: str | None = None,
    ) -> None:
        self.qdrant_client = qdrant_client
        # Generation/query-rewrite client: may point at a non-OpenAI
        # provider via LLM_BASE_URL (e.g. Gemini's OpenAI-compatibility
        # endpoint — see the module-level LLM_BASE_URL/LLM_API_KEY comment).
        self.openai_client = OpenAI(api_key=openai_api_key or LLM_API_KEY, base_url=LLM_BASE_URL)
        # Dense embeddings always go straight to OpenAI's own API — a
        # separate client, deliberately not the one above, since that one
        # may be pointed at a provider that doesn't serve OpenAI's
        # embedding models. Must match ingestion.py's embedder exactly
        # (same model, same dimensionality) or query vectors won't be
        # comparable to the vectors already stored in Qdrant.
        self.dense_embedder = DenseEmbedder()
        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        # Reranker: Cohere's hosted API when MINDAI_COHERE_API_KEY is set
        # (see the USE_COHERE_RERANKER comment above), otherwise the local
        # CrossEncoder. Both expose the same .predict(pairs) -> list[float]
        # contract, so rerank_node needs no branching of its own.
        if USE_COHERE_RERANKER:
            self.reranker = CohereReranker()
        else:
            # ms-marco-MiniLM cross-encoders output raw unbounded logits by
            # default; applying a sigmoid activation bounds scores to [0,1]
            # so they're comparable against MIN_RERANK_CONFIDENCE / the
            # graph-expansion threshold, both of which assume a normalized
            # probability scale. NOTE: the constructor kwarg is
            # `default_activation_function` (verified against the installed
            # sentence-transformers==3.3.1 signature) — an earlier version
            # of this code used the incorrect kwarg name `activation_fn`,
            # which raised a TypeError at runtime and was only caught by
            # actually starting the server, not by static review.
            self.reranker = CrossEncoder(RERANKER_MODEL, default_activation_function=torch.nn.Sigmoid())


# --------------------------------------------------------------------------- #
# Node: query_rewrite
# --------------------------------------------------------------------------- #

QUERY_REWRITE_SYSTEM_PROMPT = """You rewrite user questions about a codebase into a single, optimized \
search query for a hybrid (semantic + keyword) code search engine. The search engine matches on \
LITERAL TEXT — it cannot infer that "algorithm" means "StackingRegressor" or that "database" means \
"QdrantClient". Your job is to bridge that gap: translate the user's natural-language concept into \
the concrete identifiers, library names, and API calls that concept is actually implemented with in \
code, not just synonyms of the English words used.

Expand abbreviations. Include likely symbol/function/class names, library imports, and config keys — \
whatever a developer would grep for to find this in the actual source. Keep it concise (one line, no \
explanation). Return ONLY the rewritten query, nothing else.

EXAMPLES:
"which algorithms are used here" -> "model training algorithm sklearn RandomForestRegressor \
LGBMRegressor StackingRegressor XGBoost classifier estimator fit predict"
"what database does this use" -> "database client connection QdrantClient psycopg SQLAlchemy \
create_engine ORM"
"how does auth work" -> "authentication login JWT token bcrypt hash session middleware decorator"
"""


def query_rewrite_node(state: GraphState, deps: PipelineDependencies) -> dict:
    raw_query = state["raw_query"]
    extra_kwargs = {"reasoning_effort": "minimal"} if _IS_GEMINI_MODEL else {}
    try:
        response = deps.openai_client.chat.completions.create(
            model=REWRITE_MODEL,
            messages=[
                {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": raw_query},
            ],
            temperature=0.0,
            # 1000 not 100: on Gemini, invisible reasoning tokens draw from
            # this same budget before any visible text — see
            # _IS_GEMINI_MODEL's comment. reasoning_effort="minimal" shrinks
            # that reasoning pass; this higher cap is a safety margin since
            # "minimal" is documented as a soft target, not a hard 0.
            max_tokens=1000,
            **extra_kwargs,
        )
        rewritten = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("Query rewrite failed (%s); falling back to raw query.", e)
        rewritten = ""

    search_query = rewritten or raw_query
    logger.info("Query rewrite: %r -> %r", raw_query, search_query)
    return {"search_query": search_query}


# --------------------------------------------------------------------------- #
# Node: hybrid_retrieve (dense + sparse, fused via RRF)
# --------------------------------------------------------------------------- #

def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = RRF_K,
) -> dict[str, float]:
    """
    Standard RRF: score(d) = sum over lists of 1 / (k + rank_in_list(d)).
    Operates on point IDs; caller resolves IDs back to payloads.
    """
    scores: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def hybrid_retrieve_node(state: GraphState, deps: PipelineDependencies) -> dict:
    query = state["search_query"]
    collection_name = state["collection_name"]

    # Dense embedding is an OpenAI API call and sparse is local CPU work, so
    # they are independent — running them concurrently costs the slower of the
    # two instead of their sum. The two Qdrant searches are likewise
    # independent of each other. This matters most in SERVER mode, where each
    # search is a network round-trip: profiling against Qdrant Cloud showed
    # this node at 3.44s versus 0.80s on local disk, almost entirely latency
    # rather than compute. Local mode is unaffected by the change beyond a
    # negligible thread hop.
    with ThreadPoolExecutor(max_workers=2) as pool:
        dense_future = pool.submit(deps.dense_embedder.embed_one, query)
        sparse_future = pool.submit(lambda: next(deps.sparse_model.embed([query])))
        dense_vec = dense_future.result()
        sparse_vec = sparse_future.result()

    sparse_query = qmodels.SparseVector(
        indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist()
    )

    def _search(vector, using):
        return deps.qdrant_client.query_points(
            collection_name=collection_name,
            query=vector,
            using=using,
            limit=FUSED_CANDIDATE_COUNT,
            with_payload=True,
        ).points

    # qdrant-client is thread-safe in SERVER mode (stateless HTTP). In LOCAL
    # mode backend/pipeline_store.py serialises all access behind a process
    # lock, so the two calls are effectively sequential there and simply
    # behave as before.
    with ThreadPoolExecutor(max_workers=2) as pool:
        dense_job = pool.submit(_search, dense_vec, DENSE_VECTOR_NAME)
        sparse_job = pool.submit(_search, sparse_query, SPARSE_VECTOR_NAME)
        dense_hits = dense_job.result()
        sparse_hits = sparse_job.result()

    payload_by_id = {}
    for hit in (*dense_hits, *sparse_hits):
        payload_by_id[str(hit.id)] = hit.payload

    dense_id_order = [str(h.id) for h in dense_hits]
    sparse_id_order = [str(h.id) for h in sparse_hits]

    # See MIN_DENSE_SPARSE_AGREEMENT's comment: two independent search
    # algorithms both surfacing the same chunk is phrasing-independent
    # evidence of real relevance, unlike the cross-encoder's single-best-
    # match score. Computed here (not in rerank_node) because it only
    # needs the two raw hit lists, not anything reranking produces.
    dense_sparse_agreement = len(set(dense_id_order) & set(sparse_id_order))

    fused_scores = _reciprocal_rank_fusion([dense_id_order, sparse_id_order])
    ranked_ids = sorted(fused_scores.keys(), key=lambda d: fused_scores[d], reverse=True)

    fused_candidates: list[RetrievedChunk] = []
    for doc_id in ranked_ids[:FUSED_CANDIDATE_COUNT]:
        payload = payload_by_id[doc_id]
        fused_candidates.append(
            RetrievedChunk(
                chunk_id=doc_id,
                file_path=payload["file_path"],
                symbol_name=payload["symbol_name"],
                symbol_type=payload["symbol_type"],
                start_line=payload["start_line"],
                end_line=payload["end_line"],
                content=payload["content"],
                score=fused_scores[doc_id],
            )
        )

    logger.info(
        "Hybrid retrieve: %d dense hits, %d sparse hits -> %d fused candidates (%d in both)",
        len(dense_hits), len(sparse_hits), len(fused_candidates), dense_sparse_agreement,
    )
    return {"fused_candidates": fused_candidates, "dense_sparse_agreement": dense_sparse_agreement}


# --------------------------------------------------------------------------- #
# Node: rerank (cross-encoder)
# --------------------------------------------------------------------------- #

def rerank_node(state: GraphState, deps: PipelineDependencies) -> dict:
    candidates = state["fused_candidates"]
    query = state["search_query"]

    if not candidates:
        return {"reranked_chunks": [], "top_confidence": 0.0}

    pairs = [(query, f"{c['file_path']} {c['symbol_name']}\n{c['content']}") for c in candidates]
    scores = deps.reranker.predict(pairs)

    # When the question names a file or symbol outright ("what does app.py
    # do", "explain train_xgboost"), that is the single strongest relevance
    # signal available — and the cross-encoder largely ignores it, scoring on
    # prose/content similarity. Observed: for "what does app.py do", the
    # app.py chunk ranked 15th at 0.024 while unrelated notebook output
    # scored 0.32, so the one chunk that could answer the question was
    # crowded out of the context budget and the model refused. Boosting exact
    # filename/symbol mentions from the ORIGINAL question (not the rewritten
    # query, which invents plausible-sounding terms like "Flask FastAPI" that
    # may appear nowhere in the repo) reorders those chunks to the top
    # without disturbing the relative order of everything else.
    raw_question = state.get("raw_query") or query
    mentioned = set(re.findall(r"[\w./-]+\.\w+|\b\w{4,}\b", raw_question.lower()))
    boosted: list[float] = []
    for chunk, score in zip(candidates, scores):
        bonus = 0.0
        path = chunk["file_path"].lower()
        basename = path.rsplit("/", 1)[-1]
        symbol = (chunk.get("symbol_name") or "").lower()
        if basename in mentioned or path in mentioned:
            bonus += 1.0
        if symbol and symbol in mentioned:
            bonus += 1.0
        boosted.append(float(score) + bonus)
    scores = boosted

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    # Cap how many chunks any single file may contribute before other files
    # get a slot. Notebooks are chunked into heavily overlapping line windows
    # (e.g. 1281-1380, 1361-1460, 1441-1540 ...), which score near-identically
    # and, unconstrained, filled 14 of 20 context slots from ONE notebook —
    # crowding out the .py modules that answer the question more directly and
    # burning the token budget on near-duplicate text. Filling per-file quotas
    # in score order keeps the best chunks while guaranteeing other files are
    # represented; the cap is lifted only if too few files exist to fill
    # RERANK_TOP_N, so small repos still use the full budget.
    per_file_cap = max(2, RERANK_TOP_N // 4)
    kept: list[tuple[dict, float]] = []
    overflow: list[tuple[dict, float]] = []
    file_counts: dict[str, int] = {}
    for chunk, score in scored:
        path = chunk["file_path"]
        if file_counts.get(path, 0) < per_file_cap:
            kept.append((chunk, score))
            file_counts[path] = file_counts.get(path, 0) + 1
        else:
            overflow.append((chunk, score))
        if len(kept) >= RERANK_TOP_N:
            break
    if len(kept) < RERANK_TOP_N:
        kept.extend(overflow[: RERANK_TOP_N - len(kept)])

    top_n = kept
    reranked_chunks = [{**chunk, "score": float(score)} for chunk, score in top_n]
    top_confidence = float(top_n[0][1]) if top_n else 0.0

    logger.info(
        "Rerank: kept top %d / %d from %d distinct files, top_confidence=%.4f",
        len(reranked_chunks), len(candidates), len(file_counts), top_confidence,
    )
    return {"reranked_chunks": reranked_chunks, "top_confidence": top_confidence}


# --------------------------------------------------------------------------- #
# Conditional edge: does this question need graph expansion?
# --------------------------------------------------------------------------- #

def needs_graph_expansion(state: GraphState) -> Literal["graph_expand", "generate"]:
    """
    Routes to graph_expand only when BOTH:
      1. The query is shaped like a trace/call-chain question ("what calls X").
      2. Reranked confidence suggests vector search alone under-delivered
         (a genuinely well-matched single chunk doesn't need graph help).
    Otherwise routes straight to generate — keeps the common case cheap.
    """
    query = state.get("search_query", "") + " " + state.get("raw_query", "")
    is_trace_question = bool(TRACE_QUERY_PATTERNS.search(query))
    confidence = state.get("top_confidence", 0.0)

    if is_trace_question and confidence < 0.5:
        logger.info("Routing to graph_expand (trace question, confidence=%.4f)", confidence)
        return "graph_expand"
    return "generate"


# --------------------------------------------------------------------------- #
# Node: graph_expand (callers/callees from the ingestion-time call graph)
# --------------------------------------------------------------------------- #

def graph_expand_node(state: GraphState, deps: PipelineDependencies) -> dict:
    """
    For each reranked chunk's symbol, pull directly connected chunks
    (callers + callees) from the in-memory call graph built during
    ingestion, and fetch their full content from Qdrant so the generator
    has the actual call-chain context, not just names.
    """
    reranked = state["reranked_chunks"]
    call_graph = state["call_graph"]
    collection_name = state["collection_name"]
    reranked_chunk_ids = {c["chunk_id"] for c in reranked}
    related_chunk_ids: set[str] = set()

    for chunk in reranked:
        # Call graph keys are qualified ids (file_path::symbol_name) — see
        # ingestion.build_call_graph — since bare symbol names collide
        # across files/classes (e.g. multiple `__init__` methods).
        qualified_id = f"{chunk['file_path']}::{chunk['symbol_name']}"
        edges = call_graph.get(qualified_id)
        if not edges:
            continue
        for related_qualified_id in (*edges.get("callers", []), *edges.get("callees", [])):
            related_edges = call_graph.get(related_qualified_id)
            if related_edges and related_edges["chunk_id"] not in reranked_chunk_ids:
                related_chunk_ids.add(related_edges["chunk_id"])

    if not related_chunk_ids:
        logger.info("Graph expand: no connected chunks found.")
        return {"expanded_chunks": []}

    fetched = deps.qdrant_client.retrieve(
        collection_name=collection_name,
        ids=list(related_chunk_ids),
        with_payload=True,
    )

    expanded_chunks: list[RetrievedChunk] = [
        RetrievedChunk(
            chunk_id=str(point.id),
            file_path=point.payload["file_path"],
            symbol_name=point.payload["symbol_name"],
            symbol_type=point.payload["symbol_type"],
            start_line=point.payload["start_line"],
            end_line=point.payload["end_line"],
            content=point.payload["content"],
            score=0.0,  # not vector-ranked; included for structural relevance
        )
        for point in fetched
    ]
    logger.info("Graph expand: added %d connected chunks", len(expanded_chunks))
    return {"expanded_chunks": expanded_chunks}


# --------------------------------------------------------------------------- #
# Node: generate (context assembly + grounded, cited answer)
# --------------------------------------------------------------------------- #

GENERATION_SYSTEM_PROMPT = """You are MindAI, a code-understanding assistant that answers questions about a \
specific GitHub repository using ONLY the retrieved code context provided below.

CRITICAL RULES (non-negotiable):
1. Every factual claim about the code MUST be immediately followed by a citation in the EXACT \
format `path:start_line-end_line` (e.g. `src/auth.py:42-50`) or `path:line` for a single line. \
The citation format is STRICT — copy the file path exactly as it appears in the context block's \
header (including its extension, e.g. `README.md` not `README`), and use a COLON between the \
path and the line number(s), never brackets or parentheses for the numbers.
   CORRECT:   The app uses FastAPI (`README.md:12-15`).
   INCORRECT: The app uses FastAPI (README[12-15]).
   INCORRECT: The app uses FastAPI (see README, lines 12-15).
   Use only paths and line numbers that appear in the provided context blocks — never invent one, \
and never shorten or abbreviate a file name in a citation.
2. Refuse ONLY when the context contains nothing relevant to the question. Respond with exactly: \
"I don't have enough information in the retrieved context to answer this confidently." optionally \
followed by one sentence naming what's missing. Do not guess, speculate, or answer from general \
programming knowledge about code you cannot see.
   PARTIAL context is NOT grounds for refusal. If the context answers the question even partly, \
ANSWER with what is actually there, cite it, and state briefly what you could not determine. A \
context block for a file IS evidence about that file: if asked what `app.py` does and its code is \
shown, describe what that code does — do not refuse because you cannot see the whole file. \
Similarly, if asked to list files, list the ones present in the context and say the list may be \
incomplete. Refusing when relevant code IS present is a failure, exactly as harmful as guessing.
3. Do not fabricate file paths, function names, or line numbers under any circumstances.
4. Be precise and concise. Prefer direct quotes/paraphrases of the actual code over generalization."""


def _assemble_context(chunks: list[RetrievedChunk], token_budget: int) -> str:
    """Greedily pack chunks (highest-ranked first) into the token budget."""
    blocks: list[str] = []
    used_tokens = 0
    for chunk in chunks:
        header = f"### {chunk['file_path']}:{chunk['start_line']}-{chunk['end_line']} ({chunk['symbol_name']})"
        block = f"{header}\n```\n{chunk['content']}\n```"
        block_tokens = _count_tokens(block)
        if used_tokens + block_tokens > token_budget:
            if not blocks:
                # Even the single highest-ranked chunk exceeds budget — truncate it
                # rather than emit empty context.
                truncated = _TOKENIZER.decode(_TOKENIZER.encode(block, disallowed_special=())[:token_budget])
                blocks.append(truncated)
            break
        blocks.append(block)
        used_tokens += block_tokens
    return "\n\n".join(blocks)


def _validate_citations(answer: str, available_chunks: list[RetrievedChunk]) -> bool:
    """
    Confirms every citation in the answer names a path AND a line range that
    were actually in the context given to the model.

    What this guarantees, precisely — the distinction matters, because it is
    easy to overstate:
      - Citation existence: the cited file was really retrieved.      CHECKED
      - Citation plausibility: the cited lines overlap a real chunk.  CHECKED
      - Citation format: `path:start-end`, with a lenient fallback.   CHECKED
      - Claim support: that those lines actually SAY what the answer
        claims they say.                                          NOT CHECKED

    So this catches fabricated sources — invented files and invented line
    numbers, the most common hallucination modes here — but it cannot catch a
    claim that misreads real code it correctly cited. Faithfulness of that
    kind is measured separately by the DeepEval harness, not enforced here.

    Checks the strict `path:line` format first; falls back to the lenient
    `path[line]` pattern only if the strict pattern found nothing. This is
    a real, observed failure mode (not hypothetical): a well-grounded
    answer synthesizing a broad "tell me about this repo" question cited
    `README[1-100]` instead of `README.md:1-100`, and got discarded outright
    despite being accurate. The strict format is still what the system
    prompt demands and what generate_node's citation-rendering UI expects —
    this fallback exists only to avoid punishing an answer that IS grounded
    in real retrieved paths for a formatting slip, not to loosen what
    counts as "grounded". Path matching accepts either an exact match or a
    basename match (e.g. "README" matching "README.md") for the same
    reason — the model dropping an extension isn't evidence of a
    fabricated source.
    """
    if not available_chunks:
        return False
    available_paths = {c["file_path"] for c in available_chunks}
    available_basenames = {p.rsplit("/", 1)[-1].rsplit(".", 1)[0] for p in available_paths}

    # Paths containing spaces ("notebooks/Invoice Flagging.ipynb") cannot be
    # captured by CITATION_PATTERN, whose character class excludes spaces —
    # it matches only the trailing "Flagging.ipynb", which is not in
    # available_paths, so a CORRECT answer citing a real retrieved notebook
    # was rejected and replaced with REFUSAL_TEXT. That silently broke every
    # question whose answer lived in a space-named file. Widening the regex
    # instead is not viable: with spaces allowed, the class is greedy across
    # ordinary prose ("... see Invoice Flagging.ipynb:12" would swallow
    # preceding words), so boundaries become ambiguous. Checking the
    # retrieved paths directly is unambiguous — a citation is grounded if
    # the answer mentions a real path followed by ":<line>", which is
    # exactly the contract the system prompt asks for.
    def _cited_space_paths() -> list[str]:
        found = []
        for path in available_paths:
            if " " not in path:
                continue
            if re.search(re.escape(path) + r":\d+", answer):
                found.append(path)
        return found

    def _path_is_grounded(path: str) -> bool:
        if path in available_paths:
            return True
        basename = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return basename in available_basenames

    space_path_citations = _cited_space_paths()

    # Line ranges that were actually supplied, per path. A citation naming a
    # real retrieved file but inventing line numbers (e.g. "config.py:900-950"
    # when only lines 1-40 were in context) passes a path-only check while
    # pointing at code the model never saw — the citation looks verifiable and
    # is not. Ranges are collected per basename as well, because path matching
    # already tolerates a dropped directory or extension.
    ranges_by_key: dict[str, list[tuple[int, int]]] = {}
    for chunk in available_chunks:
        span = (int(chunk["start_line"]), int(chunk["end_line"]))
        path = chunk["file_path"]
        for key in (path, path.rsplit("/", 1)[-1], path.rsplit("/", 1)[-1].rsplit(".", 1)[0]):
            ranges_by_key.setdefault(key, []).append(span)

    def _lines_are_grounded(path: str, start: str, end: str | None) -> bool:
        keys = (path, path.rsplit("/", 1)[-1], path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
        spans = next((ranges_by_key[k] for k in keys if k in ranges_by_key), None)
        if not spans:
            return True  # path grounding is handled separately; don't double-fail
        cited_start = int(start)
        cited_end = int(end) if end else cited_start
        # Overlap, not containment: a model summarising one chunk often cites a
        # slightly wider or narrower span than the chunk's exact boundaries,
        # which is a paraphrase of a real source rather than an invention.
        return any(cited_start <= s_end and cited_end >= s_start for s_start, s_end in spans)

    citations = CITATION_PATTERN.findall(answer)
    if citations:
        # A citation to a space-containing path yields a truncated capture
        # (the tail after the last space). Treat such a capture as grounded
        # when it is the tail of a real retrieved path that the answer cited
        # in full — otherwise the fragment fails the membership check and
        # discards an answer that is genuinely grounded.
        space_tails = {p.rsplit(" ", 1)[-1] for p in space_path_citations}
        for path, start, end in citations:
            if not (_path_is_grounded(path) or path in space_tails):
                return False
            if not _lines_are_grounded(path, start, end):
                logger.warning(
                    "Citation %s:%s-%s names a retrieved file but a line range that "
                    "was never in context.", path, start, end or start,
                )
                return False
        return True

    lenient_citations = _LENIENT_CITATION_PATTERN.findall(answer)
    if lenient_citations:
        return all(_path_is_grounded(path) for path, _start, _end in lenient_citations)

    # A citation to a space-containing path may not match either regex at
    # all; it is still a valid, grounded citation.
    if space_path_citations:
        return True

    # No citation in either format is only acceptable if the model refused.
    return False


REFUSAL_TEXT = "I don't have enough information in the retrieved context to answer this confidently."


def generate_node(state: GraphState, deps: PipelineDependencies) -> dict:
    reranked = state.get("reranked_chunks", [])
    expanded = state.get("expanded_chunks", [])
    new_expanded = [c for c in expanded if c["chunk_id"] not in {r["chunk_id"] for r in reranked}]

    # Score the graph-expanded chunks before they reach the context. They are
    # pulled in by call-graph adjacency, not by relevance, so previously they
    # were appended unscored and competed for the token budget purely by
    # arriving later — a caller three hops from the question could displace a
    # reranked chunk. Scoring them against the same query puts them in the
    # same ranking as everything else; they still get in, but on merit and in
    # a sensible order. Only runs when expansion actually fired (the usual
    # path has nothing to score, so it costs nothing).
    if new_expanded:
        try:
            pairs = [
                (state["search_query"], f"{c['file_path']} {c['symbol_name']}\n{c['content']}")
                for c in new_expanded
            ]
            scores = deps.reranker.predict(pairs)
            new_expanded = [
                {**c, "score": float(s)}
                for c, s in sorted(zip(new_expanded, scores), key=lambda p: p[1], reverse=True)
            ]
            logger.info("Reranked %d graph-expanded chunks before context assembly.", len(new_expanded))
        except Exception as exc:
            # Reranking expanded chunks is an ordering improvement, not a
            # correctness requirement — if the reranker call fails, keep the
            # adjacency order rather than dropping structurally relevant code.
            logger.warning("Could not rerank expanded chunks (%s); keeping call-graph order.", exc)

    all_chunks = reranked + new_expanded

    # Pre-generation gate: only checks whether retrieval found ANY
    # candidates at all — see the gate-history comment above
    # RERANK_TOP_N/CONTEXT_TOKEN_BUDGET for why this used to also check a
    # reranker-score threshold and why that was removed. The real
    # anti-hallucination check is _validate_citations() below, AFTER
    # generation — it confirms what the model actually cited is real,
    # rather than pre-guessing whether it would be able to.
    if not all_chunks:
        logger.info("Refusing: no candidates retrieved at all.")
        return {"answer": REFUSAL_TEXT, "citations_valid": True, "refused": True}

    context = _assemble_context(all_chunks, CONTEXT_TOKEN_BUDGET)
    logger.info(
        "Context assembled from: %s",
        [f"{c['file_path']}:{c['start_line']}-{c['end_line']}" for c in all_chunks],
    )

    user_prompt = (
        f"Question: {state['raw_query']}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Answer the question using only the context above, with `path:line` citations on every claim."
    )

    generation_extra_kwargs = {"reasoning_effort": "minimal"} if _IS_GEMINI_MODEL else {}
    response = deps.openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        # See query_rewrite_node's comment on _IS_GEMINI_MODEL: Gemini's
        # invisible reasoning draws from this same budget before any
        # visible answer text, so this is raised well above what the
        # visible answer alone would need, as a safety margin.
        max_tokens=2000,
        **generation_extra_kwargs,
    )
    answer = (response.choices[0].message.content or "").strip()

    is_refusal = REFUSAL_TEXT.lower() in answer.lower()
    citations_valid = True if is_refusal else _validate_citations(answer, all_chunks)

    # One retry before refusing. A validation failure usually means the model
    # got the citation FORMAT wrong or cited a path it only half-copied — not
    # that the context lacked the answer. Refusing on the first failure threw
    # away recoverable answers (that is exactly how the space-in-filename bug
    # manifested: correct answers replaced by refusals). Re-asking once, with
    # the specific mistake named and the valid paths listed, is cheap and
    # recovers those. Capped at one attempt so a persistently ungroundable
    # question cannot loop: second failure falls through to the refusal below.
    if not citations_valid:
        logger.info("Citation validation failed; retrying once. First answer was: %r", answer)
        allowed = sorted({c["file_path"] for c in all_chunks})
        retry_prompt = (
            f"{user_prompt}\n\n"
            "Your previous answer cited a file path that was NOT in the context above, "
            "so it was rejected. Cite ONLY these paths, copied exactly, each followed by "
            "a colon and the line numbers:\n"
            + "\n".join(f"  {p}" for p in allowed)
            + "\n\nAnswer again, or reply with the exact refusal sentence if the context "
              "genuinely does not contain the answer."
        )
        retry_response = deps.openai_client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": retry_prompt},
            ],
            temperature=0.0,
            max_tokens=2000,
            **generation_extra_kwargs,
        )
        retry_answer = (retry_response.choices[0].message.content or "").strip()
        retry_is_refusal = REFUSAL_TEXT.lower() in retry_answer.lower()
        if retry_is_refusal or _validate_citations(retry_answer, all_chunks):
            logger.info("Retry produced a %s answer.", "grounded" if not retry_is_refusal else "refusing")
            return {
                "answer": retry_answer,
                "citations_valid": True,
                "refused": retry_is_refusal,
                "retry_count": 1,
            }
        logger.warning("Retry also failed validation; refusing. Retry answer was: %r", retry_answer)

    if not citations_valid:
        # Fail closed: an answer with an unverifiable citation is worse than
        # a refusal, since it looks authoritative but may be fabricated.
        logger.warning("Citation validation failed — replacing answer with refusal.")
        return {"answer": REFUSAL_TEXT, "citations_valid": True, "refused": True, "retry_count": 1}

    return {"answer": answer, "citations_valid": citations_valid, "refused": is_refusal, "retry_count": 0}


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #

def build_graph(deps: PipelineDependencies):
    """Assemble and compile the LangGraph state machine."""
    workflow = StateGraph(GraphState)

    workflow.add_node("query_rewrite", lambda s: query_rewrite_node(s, deps))
    workflow.add_node("hybrid_retrieve", lambda s: hybrid_retrieve_node(s, deps))
    workflow.add_node("rerank", lambda s: rerank_node(s, deps))
    workflow.add_node("graph_expand", lambda s: graph_expand_node(s, deps))
    workflow.add_node("generate", lambda s: generate_node(s, deps))

    workflow.set_entry_point("query_rewrite")
    workflow.add_edge("query_rewrite", "hybrid_retrieve")
    workflow.add_edge("hybrid_retrieve", "rerank")
    workflow.add_conditional_edges(
        "rerank",
        needs_graph_expansion,
        {"graph_expand": "graph_expand", "generate": "generate"},
    )
    workflow.add_edge("graph_expand", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


def answer_question(
    question: str,
    qdrant_client: QdrantClient,
    call_graph: dict[str, dict[str, list[str]]],
    collection_name: str = COLLECTION_NAME,
    deps: PipelineDependencies | None = None,
    compiled_graph=None,
) -> GraphState:
    """
    Convenience entrypoint: run the graph once for one question, return the
    final state. `deps`/`compiled_graph` are optional — pass them in (built
    once) when calling repeatedly against a long-running process (e.g. the
    FastAPI backend, see backend/main.py) to avoid reloading the embedding
    and reranker models on every question. Omit them for one-off CLI usage.
    """
    if deps is None:
        deps = PipelineDependencies(qdrant_client)
    if compiled_graph is None:
        compiled_graph = build_graph(deps)
    final_state = compiled_graph.invoke(
        {"raw_query": question, "collection_name": collection_name, "call_graph": call_graph}
    )
    return final_state


if __name__ == "__main__":
    import sys

    from ingestion import ingest_repository

    if len(sys.argv) != 3:
        print("Usage: python graph.py <github_repo_url> <question>")
        sys.exit(1)

    repo_url, question = sys.argv[1], sys.argv[2]
    print(f"Ingesting {repo_url} ...")
    _result, client, call_graph = ingest_repository(repo_url)

    print(f"\nAsking: {question}\n")
    final_state = answer_question(question, client, call_graph)
    print("=" * 80)
    print(final_state["answer"])
    print("=" * 80)
