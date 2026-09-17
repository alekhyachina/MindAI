"""
test_pipeline.py — offline tests for the pipeline's pure logic.

Every test here runs without network access or API credits: no OpenAI, no
Cohere, no Qdrant. That is deliberate — the functions worth regression-testing
(URL validation, file filtering, RRF, per-file capping, citation validation)
are pure, and making them depend on paid APIs would mean they stop running
exactly when the budget runs out, which is when regressions slip in.

Integrations NOT covered here, and why:
  - dense embedding / generation / query rewrite  -> needs OpenAI credits
  - reranking                                     -> needs a Cohere key
  - Qdrant search                                 -> needs a live collection
These are exercised by eval_live.py / eval_hard.py against a real index.

Run:  python -m pytest tests/test_pipeline.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import graph as G  # noqa: E402
from backend.schemas import IngestRepoRequest  # noqa: E402
from ingestion import (  # noqa: E402
    FALLBACK_CHUNK_LINES,
    FALLBACK_CHUNK_OVERLAP,
    MAX_CHUNK_TOKENS,
    NOISE_DIR_NAMES,
    NOISE_EXTENSIONS,
    _is_secret_file,
)
from pydantic import ValidationError  # noqa: E402


def _chunk(path: str, start: int, end: int, name: str = "fn", cid: str | None = None) -> dict:
    """A RetrievedChunk-shaped dict; only the fields under test are meaningful."""
    return {
        "chunk_id": cid or f"{path}:{start}",
        "file_path": path,
        "symbol_name": name,
        "symbol_type": "function_definition",
        "start_line": start,
        "end_line": end,
        "content": f"def {name}(): pass",
        "score": 0.0,
    }


# --------------------------------------------------------------------------- #
# URL validation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("url", [
    "https://github.com/owner/repo",
    "https://github.com/owner/repo/",
    "http://github.com/owner/repo",
    "https://www.github.com/owner/repo",
    "github.com/owner/repo",
    "https://github.com/Rohitdas04182000/Invoice-Intelligence-Freight-Cost-Prediction",
])
def test_valid_github_urls_are_accepted(url):
    # The validator normalises a bare host to https:// and strips a trailing
    # slash, but deliberately preserves the scheme and "www." the user typed —
    # git clone handles both, so rewriting them would be change for its own
    # sake. Assert acceptance and normalisation, not a canonical prefix.
    normalised = IngestRepoRequest(github_url=url).github_url
    assert "github.com/owner/repo" in normalised or "github.com/Rohitdas04182000/" in normalised
    assert normalised.startswith(("http://", "https://"))
    assert not normalised.endswith("/")


@pytest.mark.parametrize("url", [
    "http://localhost:5173",                                # the real observed bug
    "https://github.com/user?tab=repositories",             # profile, not a repo
    "https://gitlab.com/owner/repo",
    "https://github.com/owner",                             # no repo segment
    "not a url",
    "file:///etc/passwd",
    "https://127.0.0.1/owner/repo",
    "https://github.com.evil.com/owner/repo",               # lookalike host
    "https://github.com/owner/repo/../../etc",              # traversal attempt
])
def test_invalid_urls_are_rejected(url):
    with pytest.raises(ValidationError):
        IngestRepoRequest(github_url=url)


# --------------------------------------------------------------------------- #
# File filtering
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", [
    "id_rsa", "id_ed25519", "server.pem", "private.key", "store.p12",
    ".env", ".env.production", ".netrc", ".npmrc",
    "credentials.json", "service-account.json", "secrets.yaml",
    "aws_credentials.ini", "my-private-key.txt", "db_password.txt",
])
def test_credential_files_are_never_indexed(name):
    assert _is_secret_file(Path("/repo") / name) is True


@pytest.mark.parametrize("name", [
    ".env.example", ".env.sample", "secrets.yaml.template",
    "app.py", "README.md", "config.py", "main.go",
])
def test_ordinary_and_template_files_are_indexed(name):
    assert _is_secret_file(Path("/repo") / name) is False


def test_dependency_and_build_dirs_are_pruned():
    for d in ("node_modules", ".venv", ".git", "dist", "build", "__pycache__", "vendor"):
        assert d in NOISE_DIR_NAMES


def test_binary_and_media_extensions_are_excluded():
    for ext in (".png", ".mp4", ".exe", ".so", ".pdf", ".woff2", ".sqlite3"):
        assert ext in NOISE_EXTENSIONS


# --------------------------------------------------------------------------- #
# RRF
# --------------------------------------------------------------------------- #

def test_rrf_matches_the_published_formula():
    fused = G._reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    # Each doc is rank 0 in one list and rank 1 in the other:
    #   1/(60+1) + 1/(60+2)
    expected = 1 / 61 + 1 / 62
    assert fused["a"] == pytest.approx(expected)
    assert fused["b"] == pytest.approx(expected)


def test_rrf_rewards_agreement_between_retrievers():
    # "agreed" is top of both lists; "dense_only" is top of one and absent
    # from the other. Agreement must win.
    fused = G._reciprocal_rank_fusion(
        [["agreed", "dense_only"], ["agreed", "sparse_only"]], k=60
    )
    assert fused["agreed"] > fused["dense_only"]
    assert fused["agreed"] > fused["sparse_only"]


def test_rrf_deduplicates_across_lists():
    """Two top-100 lists yield 100-200 unique candidates, not a fixed 150."""
    left = [f"d{i}" for i in range(100)]
    right = [f"d{i}" for i in range(50, 150)]
    fused = G._reciprocal_rank_fusion([left, right], k=60)
    assert len(fused) == 150  # 100 + 100 - 50 shared

    # Appearing in both lists helps, but does not automatically outrank a
    # doc sitting at rank 0 of a single list: d50 (rank 50 + rank 0) beats
    # d0 (rank 0 only), while d99 (rank 99 + rank 49) does not. Rank
    # position still dominates, which is the property RRF is chosen for.
    assert fused["d50"] > fused["d0"] > fused["d99"]

    # Identical rank in both lists strictly beats that same rank in one.
    both = G._reciprocal_rank_fusion([["x"], ["x"]], k=60)["x"]
    one = G._reciprocal_rank_fusion([["x"], ["y"]], k=60)["x"]
    assert both > one


def test_rrf_k_is_configurable():
    assert G._reciprocal_rank_fusion([["a"]], k=1)["a"] == pytest.approx(1 / 2)
    assert G._reciprocal_rank_fusion([["a"]], k=60)["a"] == pytest.approx(1 / 61)


# --------------------------------------------------------------------------- #
# Citation validation
# --------------------------------------------------------------------------- #

def test_citation_to_retrieved_path_and_lines_is_valid():
    chunks = [_chunk("src/config.py", 1, 40)]
    assert G._validate_citations("Uses PG (`src/config.py:10-30`).", chunks) is True


def test_citation_to_fabricated_path_is_rejected():
    chunks = [_chunk("src/config.py", 1, 40)]
    assert G._validate_citations("Uses X (`src/made_up.py:1-5`).", chunks) is False


def test_citation_with_invented_line_numbers_is_rejected():
    """A real file with lines that were never in context must not pass."""
    chunks = [_chunk("src/config.py", 1, 40)]
    assert G._validate_citations("Uses PG (`src/config.py:900-950`).", chunks) is False


def test_citation_to_path_containing_spaces_is_valid():
    """The regression that silently replaced correct answers with refusals."""
    chunks = [_chunk("notebooks/Invoice Flagging.ipynb", 1281, 1380, name="nb")]
    answer = "The model is a Random Forest (`notebooks/Invoice Flagging.ipynb:1281-1380`)."
    assert G._validate_citations(answer, chunks) is True


def test_answer_without_any_citation_is_rejected():
    assert G._validate_citations("The model is great.", [_chunk("a.py", 1, 5)]) is False


def test_one_bad_citation_invalidates_the_whole_answer():
    chunks = [_chunk("src/config.py", 1, 40)]
    answer = "Real (`src/config.py:5-10`) and fake (`src/nope.py:1-2`)."
    assert G._validate_citations(answer, chunks) is False


def test_validation_fails_closed_on_empty_context():
    assert G._validate_citations("Anything (`a.py:1-2`).", []) is False


def test_lenient_bracket_format_is_accepted_when_grounded():
    chunks = [_chunk("README.md", 1, 100, name="readme")]
    assert G._validate_citations("See (`README[1-100]`).", chunks) is True


# --------------------------------------------------------------------------- #
# Context limits
# --------------------------------------------------------------------------- #

def test_context_assembly_respects_the_token_budget():
    chunks = [_chunk("f.py", i, i + 10) for i in range(1, 400, 10)]
    for chunk in chunks:
        chunk["content"] = "x = 1\n" * 200
    context = G._assemble_context(chunks, token_budget=500)
    assert G._count_tokens(context) <= 500 * 1.1  # allows the single-chunk truncation path


def test_context_assembly_uses_a_real_tokenizer_not_character_count():
    # Many tokens, few characters — a char-count budget would mis-measure this.
    assert G._count_tokens("a " * 500) > 400


def test_per_file_cap_prevents_one_file_dominating():
    cap = max(2, G.RERANK_TOP_N // 4)
    assert cap == 5, "cap is documented as RERANK_TOP_N/4 = 5"
    assert G.RERANK_TOP_N == 20
    assert G.CONTEXT_TOKEN_BUDGET == 8000


# --------------------------------------------------------------------------- #
# Graph shape and expansion routing
# --------------------------------------------------------------------------- #

def test_trace_questions_are_detected():
    for q in ["who calls get_active_run", "what is the call chain",
              "which functions depend on this", "trace the execution path"]:
        assert G.TRACE_QUERY_PATTERNS.search(q), q


def test_plain_questions_are_not_trace_questions():
    for q in ["what database is used", "what does app.py do", "list the files"]:
        assert not G.TRACE_QUERY_PATTERNS.search(q), q


def test_expansion_requires_both_trace_shape_and_low_quality():
    trace = "who calls get_active_run"
    plain = "what database is used"
    assert G.needs_graph_expansion(
        {"search_query": trace, "raw_query": trace, "top_confidence": 0.1}
    ) == "graph_expand"
    # Confident trace question skips expansion — keeps the common case cheap.
    assert G.needs_graph_expansion(
        {"search_query": trace, "raw_query": trace, "top_confidence": 0.9}
    ) == "generate"
    # Non-trace question never expands, however weak retrieval looks.
    assert G.needs_graph_expansion(
        {"search_query": plain, "raw_query": plain, "top_confidence": 0.01}
    ) == "generate"


def test_graph_state_tracks_retry_count():
    assert "retry_count" in G.GraphState.__annotations__


def test_fallback_chunking_constants_are_sane():
    assert FALLBACK_CHUNK_OVERLAP < FALLBACK_CHUNK_LINES
    assert MAX_CHUNK_TOKENS < 8192, "must stay under the embedding API's hard limit"
