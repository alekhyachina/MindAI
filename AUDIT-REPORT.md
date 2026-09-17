# MindAI — Verification Audit

**Scope:** 15 suspected issues across the ingestion and query pipelines,
verified against source before any change. Fixes limited to confirmed
problems; no structural or graph changes.

**Headline:** 6 of 15 suspicions were already correctly implemented. 4 were
confirmed real. 5 were partially correct. 5 fixes were applied, 59 tests
added, and the graph is unchanged at 5 nodes.

---

## 1. The pipeline as originally implemented

Read from source, not from documentation:

```
INGEST   URL (regex, API layer only)
      -> git clone --depth 1 --single-branch (timeout 300s)
      -> filter: 30 noise dirs, ~45 noise exts, lockfiles, >1MB, empty
      -> tree-sitter AST -> whole functions/classes
      -> split any chunk over MAX_CHUNK_TOKENS (6000)
      -> fallback: 100-line windows, 20-line overlap
      -> dense (OpenAI text-embedding-3-small, 1536-d) + sparse (FastEmbed BM25)
      -> Qdrant collection repo_<uuid>, both as NAMED vectors on one point
      -> call graph (regex scan) -> JSON on disk
      -> status = ready

QUERY    raw_query
      -> query_rewrite (LLM)                    ~1.5s
      -> dense search 100  +  sparse search 100  ~2.9s   (both in Qdrant)
      -> RRF fusion, k = RRF_K
      -> rerank (Cohere v3.5) 100 -> 20          ~1.9s
         + filename/symbol boost, max 5 chunks per file
      -> IF trace-shaped AND top_confidence < 0.5 -> graph_expand
         (expanded chunks appended UNRANKED)
      -> assemble context, 8000 tokens via tiktoken
      -> generate (raw_query, not search_query)  ~3.0s
      -> _validate_citations (path existence only)
      -> answer OR immediate refusal (no retry)
      -> save to SQLite -> stream over SSE
```

---

## 2. Findings

### Already correct — no change made

| # | Suspicion | Reality | Evidence |
|---|---|---|---|
| 1 | BM25/Qdrant architecture ambiguous | **Option A.** FastEmbed builds sparse vectors locally; Qdrant stores `"dense"` + `"sparse"` as named vectors and runs both searches. No separate BM25 index exists | `ingestion.py:775-794`; verified live: one point carries 1536 floats + 16 non-zero sparse terms |
| 2 | Missing repo isolation | Uses **collection-per-repo** (`repo_<uuid>`) instead of payload filtering — physically stronger, since chunks from different repos never share an index | `backend/pipeline_store.py:104` |
| 6 | RRF incorrect | Textbook `score(d) = Σ 1/(k + rank)`, with `k` already a named constant | `graph.py:318-329` |
| 7 | Original question lost in rewrite | `raw_query` and `search_query` are separate state fields; generation uses `raw_query` | `graph.py:731` |
| 10 | Context limits not enforced | Already uses **tiktoken**, not character counts. 8000-token budget, per-file cap of 5, dedup by `chunk_id` | `graph.py:724` |
| 13, 15 | Prompt / persistence order | Prompt already forbids fabrication and requires citations; persistence already happens after validation | `graph.py:532-550`, `backend/main.py:273+` |

**Note on Issue 2:** the suspicion assumed a single shared collection needing
`repo_id` filters. That premise does not apply to this design. Missing
metadata (`branch`, `commit_sha`, `content_hash`) is a real but separate gap
— see Limitations.

### Confirmed problems — fixed

| # | Problem | Evidence |
|---|---|---|
| 12 | **No retry.** A citation-validation failure refused immediately, discarding recoverable answers | Zero `retry` references in `graph.py` |
| 9 | **Expanded chunks bypassed reranking.** Appended raw, competing for the token budget by arrival order rather than relevance | `graph.py:710-711` |
| 11 | **Validation checked path existence only.** A real file with invented line numbers passed | `graph.py:630-690` |
| 4 | **No credential filtering.** No `.pem`, `.key`, `id_rsa`, or `.env` patterns anywhere in ingestion | grep returned nothing |
| — | **No tests at all** | Zero test files in the repo |

### Partially correct — documented or partly fixed

| # | State | Action |
|---|---|---|
| 3 | Clone has `--depth 1`, `--single-branch`, 300s timeout. Missing: size cap, hooks disable, and URL validation inside `ingestion.py` (API-layer only, so CLI use bypasses it) | Documented; not fixed |
| 5 | AST chunking is primary and correct, with a 6000-token split. Fallback is line-based, not token-aware | Documented; needs re-ingest to change |
| 8 | `top_confidence` is a raw reranker score, not a calibrated probability | Documented in `GraphState` |
| 14 | Call graph stores `chunk_id` and `file_path` but no `resolution_status`; regex-based, so dynamic dispatch is unresolved | Documented |

---

## 3. Files modified

| File | Change |
|---|---|
| `graph.py` | Retry-once on validation failure; rerank expanded chunks; line-range validation; `retry_count` in state; `top_confidence` documented as uncalibrated; corrected validation docstring |
| `ingestion.py` | `SECRET_EXTENSIONS`, `SECRET_FILENAMES`, `SECRET_NAME_PATTERNS`, `_is_secret_file()`, wired into `discover_source_files` |
| `reranker.py` | `RERANK_DOC_CHAR_LIMIT = 2000` (payload cap) |
| `README.md` | Rewritten — see §7 |
| `tests/test_pipeline.py` | **New.** 59 offline tests |
| `eval_hard.py` | **New.** Evaluation with source-derived ground truth |
| `eval_live.py` | **New.** Evaluation against an already-indexed repo |

---

## 4. Every change, explained

**Retry once before refusing** (`generate_node`). A validation failure
usually means a malformed or half-copied citation, not missing evidence.
The retry re-prompts with the exact list of valid paths. Capped at one
attempt so an ungroundable question cannot loop. This is the same failure
mode that made the space-in-filename bug so damaging: correct answers were
being replaced by refusals with no second chance.

*Implemented as a loop inside the existing node, not as separate
`validate` / `retry_or_refuse` / `finalize` nodes. Three extra nodes and
conditional edges would have added graph complexity for behaviour a loop
expresses in ten lines.*

**Rerank graph-expanded chunks** (`generate_node`). Expanded chunks arrive
by call-graph adjacency, not relevance, so a caller three hops away could
displace a reranked chunk purely by arriving later. They are now scored
against the same query and ordered alongside everything else. Wrapped in
try/except: if the reranker call fails, adjacency order is kept rather than
dropping structurally relevant code.

**Line-range validation** (`_validate_citations`). Ranges actually supplied
are collected per path (and per basename, since path matching already
tolerates a dropped directory or extension). A citation naming a retrieved
file but a range never in context is now rejected. Uses **overlap**, not
containment — a model summarising a chunk often cites a slightly wider or
narrower span, which is a paraphrase of a real source rather than an
invention.

**Credential filtering** (`_is_secret_file`). A private key or `.env`
committed to a repo would otherwise be chunked, embedded, sent to an
embedding API, stored in Qdrant, and eventually quoted back *with a
citation*. Errs toward skipping: a false positive costs one unindexed file,
a false negative discloses a secret. `.env.example` and `.template` files
pass through, since their purpose is placeholder values.

**Reranker payload cap** (`RERANK_DOC_CHAR_LIMIT`). Measured: 100 candidates
at full length shipped ~740 KB per question at 2.37s median; at 2000 chars,
~195 KB at 1.23s — roughly half the latency of the slowest node. Relevance
is decided by whether a chunk is *about* the query, which its opening
establishes (an AST chunk begins with signature and docstring, and the
caller prepends file path and symbol name).

**`top_confidence` documented, not renamed.** Renaming the field would touch
every reader for no behavioural gain. The docstring now states plainly that
0.7 does not mean "70% confident" and that the value is only meaningful
relative to other scores from the same model.

---

## 5. Tests

`python -m pytest tests/test_pipeline.py -q` → **59 passed in 20.26s**

| Group | Tests | Covers |
|---|---|---|
| URL validation | 15 | Valid forms; rejects `localhost:5173`, profile URLs, GitLab, lookalike hosts (`github.com.evil.com`), `file://`, traversal |
| Credential filtering | 22 | Keys, `.env`, certs, name patterns; allows `.env.example` and ordinary source |
| Noise filtering | 2 | Dependency/build dirs, binary/media extensions |
| RRF | 4 | Exact formula, agreement reward, dedup (100–200 not fixed), configurable `k` |
| Citation validation | 8 | Valid path+lines, fabricated path, **invented line numbers**, space-containing paths, no citation, mixed real/fake, empty context, lenient format |
| Context limits | 3 | Token budget, real tokenizer, documented constants |
| Expansion routing | 4 | Trace detection, non-trace rejection, both-conditions requirement, `retry_count` in state |
| Chunking constants | 1 | Overlap sanity, under the 8192 embedding limit |

**Three tests initially failed. All three were defects in my tests, not the
code** — worth recording, since the instruction was to verify before fixing:

- Two URL tests asserted a canonical `https://github.com/` prefix. The
  validator deliberately preserves the user's scheme and `www.`, since git
  clone accepts both. Tests corrected.
- One RRF test asserted that appearing in both lists always outranks
  appearing in one. False: `d0` at rank 0 of one list legitimately scores
  above `d99` at ranks 99/49. Verified numerically (`0.0164` vs `0.0153`)
  before concluding the formula was right and the assertion wrong.

---

## 6. Remaining limitations

**Could not re-verify by runtime test.** OpenAI credits were exhausted
mid-audit (`429 credit_balance_exhausted`, not a rate limit), which stopped
the post-change DeepEval run. So the retry path, expanded-chunk reranking,
and live chat are **verified by unit test and by import/startup, but not by
a fresh end-to-end eval**. The changes only tighten validation and improve
ordering, so a regression is unlikely — but it is unproven, and should not
be reported as proven.

Last good eval (before these changes, `gpt-4o` judge) reached 3 of 6
questions at Faithfulness 1.00 / 1.00 / 1.00 before hitting the limit.

**Not implemented, by agreement:**

- **Issue 5** — token-aware fallback chunking. Requires re-ingesting every
  repo, since chunking happens at index time.
- **Issue 3 remainder** — repo size cap, `core.hooksPath` disable, URL
  validation inside `ingestion.py`. Needs no re-ingest; ~20 lines.
- **Issue 14** — `resolution_status` on call-graph edges. Changes the
  payload schema, so it also implies re-ingestion.
- **Issue 2 metadata** — `branch`, `commit_sha`, `content_hash` per chunk.
  Same constraint.

**Inherent to the design:**

- Claim support is not enforced at validation time (only source existence).
- Call-graph extraction is regex-based and approximate.
- No incremental re-indexing; an index is one snapshot.
- Broad conceptual questions still sometimes refuse.
- Local Qdrant mode serialises all access behind a process lock.

---

## 7. Does the README match the code?

**It did not.** Five inaccuracies were found and corrected:

| Claim | Reality |
|---|---|
| "embeds each chunk with FastEmbed (dense + sparse/BM25)" | FastEmbed does **sparse only**. Dense is OpenAI `text-embedding-3-small` (`dense_embeddings.py:27`) |
| "cross-encoder reranking" | Cohere `rerank-v3.5` when `MINDAI_COHERE_API_KEY` is set; the local CrossEncoder is the fallback (`graph.py:79`) |
| "low confidence or a missing citation makes it refuse" | **No confidence gate exists.** Three score-based gates were tried and removed. Refusal comes from post-generation citation validation |
| "copy `.env.example`" | **No `.env.example` exists** — the instruction was unfollowable |
| "CPU-only embedding/reranking is slow" | Both are API calls now. Measured ~10s per question, almost all network latency; Qdrant search is ~0.25s |

The README now also documents the retrieval path precisely, repository
isolation, what citation validation does and does not guarantee, the
evaluation methodology failure, the test suite, and the full environment
variable table.

---

## 8. The final pipeline

```
╔════════════════ PHASE 1 · INGESTION (once per repo) ════════════════╗

   GitHub repo URL
        │
        ▼
   URL validation  (github.com/owner/repo — API layer)
        │
        ▼
   git clone --depth 1 --single-branch   (timeout 300s)
        │
        ▼
   Filter files
     ├── prune noise dirs (node_modules, .venv, dist, …)
     ├── drop binary/media/lockfile extensions
     ├── drop >1MB and empty
     └── SKIP CREDENTIAL FILES            ← new
        │
        ▼
   tree-sitter parse → AST
        │
        ▼
   AST-aware chunking  (whole functions / classes)
     ├── split if > 6000 tokens
     └── fallback: 100-line windows, 20 overlap
        │
        ├──────────────────────┬──────────────────────┐
        ▼                      ▼                      │
   Dense embedding        Sparse BM25                 │
   OpenAI 1536-d          FastEmbed, local            │
        │                      │                      │
        └──────────┬───────────┘                      │
                   ▼                                  ▼
          QDRANT  repo_<uuid>                  Call graph
      one point, both named vectors         caller → callee
      payload: file_path, symbol_name,       JSON on disk
      symbol_type, language, start_line,
      end_line, content, calls, imports
                   │                                  │
                   └──────────────┬───────────────────┘
                                  ▼
                           status = ready

╔════════════════ PHASE 2 · QUERY (per question, ~10s) ═══════════════╗

   User question  (preserved as raw_query)
        │
        ▼
   ① query_rewrite            LLM → code-like search terms      ~1.5s
        │
        ▼
   ② hybrid_retrieve                                            ~2.9s
        ├── dense search  (Qdrant, top 100)   ─┐  run in parallel
        └── sparse search (Qdrant, top 100)   ─┘
                   │
                   ▼
            RRF fusion   Σ 1/(k + rank)
          100–200 unique candidates after dedup
        │
        ▼
   ③ rerank                   Cohere v3.5, 100 → 20             ~1.9s
        ├── + filename / symbol boost
        ├── max 5 chunks per file
        └── docs capped at 2000 chars                            ← new
        │
        ▼
   ◆ needs_graph_expansion()   trace-shaped AND quality < 0.5 ?
        │                                    │
      YES                                   NO
        ▼                                    │
   ④ graph_expand                             │
     add callers / callees                    │
        │                                     │
        ▼                                     │
     RERANK expanded chunks      ← new        │
        │                                     │
        └──────────────┬──────────────────────┘
                       ▼
        Context assembly   8000 tokens (tiktoken)
        highest-ranked first, dedup by chunk_id
                       │
                       ▼
   ⑤ generate      answer from raw_query, cite path:line        ~3.0s
                       │
                       ▼
        ◆ validate citations
          ├── path was retrieved ?
          └── line range overlaps a real chunk ?                 ← new
                       │
              ┌────────┴────────┐
            PASS              FAIL
              │                 │
              │                 ▼
              │          RETRY ONCE            ← new
              │       re-prompt with valid paths
              │                 │
              │        ┌────────┴────────┐
              │      PASS              FAIL
              │        │                 │
              ▼        ▼                 ▼
        Cited answer                 Grounded refusal
              │                            │
              └──────────────┬─────────────┘
                             ▼
                    Save to SQLite
                  (role, content, refused)
                             ▼
                    Stream over SSE → React UI
```

---

## 9. Final LangGraph structure

**Unchanged — 5 nodes, 1 conditional edge.** Verified at runtime:

```python
nodes = ['__start__', 'query_rewrite', 'hybrid_retrieve',
         'rerank', 'graph_expand', 'generate', '__end__']
```

| Node | Responsibility |
|---|---|
| `query_rewrite` | Expand the question into retrieval terms |
| `hybrid_retrieve` | Dense + sparse search (parallel), RRF fusion |
| `rerank` | Cross-encoder scoring, filename boost, per-file cap |
| `graph_expand` | *Conditional.* Add callers/callees |
| `generate` | Context assembly, generation, validation, retry |

```
START → query_rewrite → hybrid_retrieve → rerank
                                            │
                    needs_graph_expansion() ◆
                         ╱                  ╲
              graph_expand                generate → END
                    ╲                      ╱
                     ────────────────────►
```

Every fix landed **inside** an existing node. No node was added, removed, or
split, and no edge changed — deliberately, so the graph did not grow more
complex than the problems required.
