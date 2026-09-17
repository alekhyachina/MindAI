# MindAI

A Graph-RAG chatbot that answers questions about any GitHub repository with
verifiable, line-level citations — and refuses rather than guess when the
retrieved code can't support an answer.

## How it works

```
ingestion.py    Shallow-clones a repo (--depth 1), parses it with
                tree-sitter into AST-aware chunks (whole functions and
                classes, not fixed token windows), embeds each chunk twice
                — dense via OpenAI text-embedding-3-small, sparse/BM25
                via FastEmbed locally — and upserts both as NAMED VECTORS
                on a single Qdrant point. Also extracts a caller/callee
                call graph, and skips credential files outright.

graph.py        A LangGraph pipeline of five nodes: query rewrite ->
                hybrid retrieval (dense + sparse, fused via Reciprocal
                Rank Fusion) -> reranking -> conditional graph expansion
                (callers/callees, only for trace questions with weak
                retrieval) -> generation. Every answer must cite
                `path:line`; citations are verified against the retrieved
                context AFTER generation, and an answer citing anything
                that wasn't retrieved is discarded.

eval.py         DeepEval metric definitions (Faithfulness 0.75, Answer
                Relevancy 0.70, Contextual Recall 0.70) plus a mock gold
                dataset that runs standalone.

eval_live.py    The same metrics against an already-indexed repo.
eval_hard.py    Ditto, with ground truth written from the source rather
                than from the system's own answers — see "Evaluation".
```

`ingestion.py`, `graph.py`, and `eval.py` are the backbone — they run
standalone from the CLI with no web server required. `backend/` and
`frontend/` are a thin web UI wrapped around that backbone: FastAPI for
auth/persistence/HTTP, React for the chat interface. Neither changes how
retrieval, reranking, or generation actually work.

### The retrieval path, precisely

Hybrid search here is **Qdrant-native**, which is worth stating because it is
easy to describe wrongly: FastEmbed builds BM25-style sparse vectors
*locally at ingestion time*, those vectors are stored in Qdrant alongside the
dense ones, and **Qdrant performs both the dense and the sparse search**.
There is no separate local BM25 index and no second search engine. RRF then
fuses the two ranked lists by rank position — not by score, because cosine
similarity (~0.87) and BM25 (~14.2) are not on comparable scales.

Two top-100 searches yield **100–200 unique candidates** after
deduplication, not a fixed number.

### Repository isolation

Each repo gets **its own Qdrant collection** (`repo_<uuid>`), so chunks from
different repositories are never in the same index and cross-repo leakage is
structurally impossible. Ownership is additionally enforced per user on every
API route: another user's conversation returns **404, not 403**, so its
existence isn't disclosed.

## Project layout

```
mindai/
├── ingestion.py, graph.py, eval.py   Core RAG pipeline (see above)
├── eval_live.py, eval_hard.py        Evaluation against a live index
├── dense_embeddings.py               OpenAI dense embedder (concurrent batches)
├── reranker.py                       Cohere reranking provider
├── semantic_chunking.py              Alternative embedding-derived chunking
├── requirements.txt                  Core pipeline dependencies
├── tests/test_pipeline.py            59 offline tests (no API keys needed)
├── backend/                          FastAPI wrapper (auth, DB, HTTP API)
│   ├── main.py                       Routes: auth, repos, conversations, chat (SSE)
│   ├── auth.py, db.py, models.py, schemas.py, pipeline_store.py
│   └── requirements.txt              Backend deps (includes ../requirements.txt)
└── frontend/                         React + Vite + Tailwind chat UI
    └── src/{pages,components,lib}/
```

## Running it

### 1. Core pipeline only (CLI, no web server)

```
pip install -r requirements.txt
python ingestion.py https://github.com/<owner>/<repo>
python graph.py https://github.com/<owner>/<repo> "How does X work?"
python eval.py
```

Note that `python graph.py` **re-ingests the repo before answering** — it
clones, chunks and embeds from scratch on every invocation, which costs
minutes and a full round of embedding calls per question. It exists for
one-off inspection of the pipeline, not for repeated querying. Use the web
app (or `eval_live.py`) to ask many questions against one index.

### 2. Full web app (backend + frontend)

**Set up `.env`** in the project root. Copy `.env.example` to `.env` and fill
it in. At minimum you need:

```
OPENAI_API_KEY=sk-...
MINDAI_JWT_SECRET=<python -c "import secrets; print(secrets.token_urlsafe(32))">
```

Optional, all with working defaults:

| Variable | Default | Effect |
|---|---|---|
| `QDRANT_URL` / `QDRANT_API_KEY` | unset | Set → Qdrant server/cloud. Unset → local on-disk |
| `MINDAI_QDRANT_PATH` | `./qdrant_storage` | Local storage directory |
| `DATABASE_URL` | `sqlite:///./mindai.db` | Any SQLAlchemy URL (Postgres for production) |
| `MINDAI_GENERATION_MODEL` | `gpt-4o-mini` | Answer generation |
| `MINDAI_REWRITE_MODEL` | `gpt-4o-mini` | Query rewriting |
| `MINDAI_JUDGE_MODEL` | `gpt-4o` | DeepEval judge |
| `MINDAI_COHERE_API_KEY` | unset | Set → Cohere hosted reranking. Unset → local CrossEncoder |
| `MINDAI_CHUNK_STRATEGY` | `ast` | `ast` or `semantic`. Changing it requires re-ingesting |

**Install and start the backend:**
```
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --port 8001
```
First startup takes 1–2 minutes: the embedding and reranker models load
before the server accepts requests. Port 8000 may already be in use — check
with `netstat -ano | findstr :8000` on Windows, or pick any free port and
update `frontend/vite.config.ts`'s proxy target to match.

**Install and start the frontend** (separate terminal):
```
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** — sign up, ingest a repo from the sidebar,
ask it questions.

### 3. Tests

```
python -m pytest tests/test_pipeline.py -q
```

59 tests, all offline — no OpenAI, Cohere, or Qdrant needed. They cover URL
validation, credential-file filtering, RRF correctness and deduplication,
citation validation, context limits, and graph-expansion routing.

## Evaluation

```
python eval.py          # mock gold dataset, runs standalone
python eval_live.py     # live pipeline against an already-indexed repo
python eval_hard.py     # same, with ground truth written from source
```

Three LLM-judged metrics: **Faithfulness** (0.75), **Answer Relevancy**
(0.70), **Contextual Recall** (0.70).

`eval_hard.py` exists because of a real methodology failure worth
documenting. An earlier run scored Contextual Recall **1.000** — but its
expected answers had been written *after* reading what the pipeline produced.
Since that metric asks "does the retrieved context support the expected
answer?", expectations derived from observed output cannot meaningfully fail.
Rewriting the ground truth from the source first dropped recall to **0.917**,
which is the number worth trusting.

A second caveat found the same way: with a `gpt-4o-mini` judge, one question
scored Faithfulness **0.00** for a claim that was verifiably present in the
retrieved chunk — the judge lost the detail in ~50K characters of context.
The same case scored **1.00** under `gpt-4o`. **Validate the evaluator, not
just the system.**

Judge calls are token-heavy: a Faithfulness judgment over a full context can
request ~13,000 tokens, so on a 30,000 TPM tier only about two fit per
minute. `eval_hard.py` paces itself via `MINDAI_JUDGE_SPACING` (default 30s).

## What citation validation does and does not guarantee

The system's central claim is "no hallucination", so it's worth being exact
about what is enforced:

| Check | Enforced? |
|---|---|
| The cited file was actually retrieved | **Yes** |
| The cited line range overlaps a retrieved chunk | **Yes** |
| Citation format `path:start-end` (with a lenient fallback) | **Yes** |
| That those lines actually *say* what the answer claims | **No** |

So fabricated sources — invented files, invented line numbers — are caught
and the answer is discarded. A claim that *misreads real code it correctly
cited* is not caught here; that is what the Faithfulness metric measures.
Validation happens **after** generation rather than as a pre-generation
confidence threshold: three score-based gates were tried and removed, because
the reranker's scores sat near zero even for answers that were correct (it
was trained on web search, not code). Confirming what the model *did* cite
beats predicting what it *could* cite.

On a validation failure the pipeline **retries once**, re-prompting with the
list of valid paths, before falling back to a refusal.

## Known constraints (read before deploying beyond local/solo use)

- **Qdrant mode is chosen by `QDRANT_URL`.** Set it (plus `QDRANT_API_KEY`)
  to point at Qdrant Cloud or a Qdrant container, and the pipeline talks to
  it over HTTP — the server handles concurrency, so ingestions and chat
  questions run in parallel. Leave it unset and the client falls back to
  local/on-disk mode at `MINDAI_QDRANT_PATH`, which is **not** thread-safe
  for concurrent access (confirmed against Qdrant's own source and an open
  upstream issue); the backend then guards every call with a process-wide
  lock, so only one ingestion or one chat question runs at a time,
  server-wide. Fine for a single user, a real ceiling beyond that — see
  `backend/pipeline_store.py`'s docstring. Note the two modes have separate
  storage: collections ingested in one are not visible in the other.
- **SQLite by default.** Fine for development; set `DATABASE_URL` to a
  Postgres DSN for anything beyond local use.
- **Dense embedding and reranking are API calls, not local CPU.** Dense
  embedding goes to OpenAI (concurrent batches of 100) and reranking to
  Cohere when `MINDAI_COHERE_API_KEY` is set. Per-node profiling put a
  question at ~10s total, almost entirely network latency to three
  providers — Qdrant search itself is ~0.25s. Without the Cohere key,
  reranking falls back to a local sentence-transformers CrossEncoder, which
  is CPU-bound and slower.
- **Chat is not token-streamed.** The SSE endpoint streams status updates
  and the finished answer, not token-by-token generation — the LangGraph
  pipeline returns its final state as one unit. True token streaming would
  mean changing `generate_node`'s OpenAI call to `stream=True`.
- **Fallback chunking is line-based.** tree-sitter handles supported
  languages; anything it can't parse (notebooks, markdown, config) falls
  back to 100-line windows with 20 lines of overlap rather than
  token-aware chunks. An AST chunk over `MAX_CHUNK_TOKENS` (6000) is split
  to stay under the embedding API's 8192-token hard limit.
- **Call-graph extraction is approximate.** Caller/callee edges come from a
  regex scan of chunk bodies, so dynamic dispatch, reflection, and
  dependency injection are not resolved. Treat expansion as a helpful hint,
  not a complete call graph.
- **No incremental re-indexing.** Ingesting a repo captures one snapshot;
  new commits require re-ingesting, and nothing detects staleness.
- **Broad conceptual questions are the weak spot.** Questions like "how does
  X work?" where no single chunk is a strong match still sometimes refuse.
  Query decomposition would be the fix.
