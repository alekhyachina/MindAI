# MindAI

A Graph-RAG chatbot that answers questions about any GitHub repository with
verifiable, line-level citations — and refuses to answer rather than
hallucinate when it isn't confident.

## How it works

```
ingestion.py    Shallow-clones a repo, parses it with tree-sitter into
                AST-aware chunks (functions/classes, not fixed token
                windows), embeds each chunk with FastEmbed (dense +
                sparse/BM25), and upserts into Qdrant.

graph.py        A LangGraph pipeline: query rewrite -> hybrid retrieval
                (dense + sparse, fused via Reciprocal Rank Fusion) ->
                cross-encoder reranking -> conditional graph expansion
                (callers/callees, for trace questions) -> generation.
                Every answer must cite `path:line`; low confidence or a
                missing citation makes it refuse instead of guessing.

eval.py         A DeepEval harness (Faithfulness, Answer Relevancy,
                Contextual Recall) against a mock gold dataset.
```

`ingestion.py`, `graph.py`, and `eval.py` are the backbone — they run
standalone from the CLI with no web server required. `backend/` and
`frontend/` are a thin web UI wrapped around that backbone: FastAPI for
auth/persistence/HTTP, React for the chat interface. Neither changes how
retrieval, reranking, or generation actually work.

## Project layout

```
mindai/
├── ingestion.py, graph.py, eval.py   Core RAG pipeline (see above)
├── requirements.txt                  Core pipeline dependencies
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

### 2. Full web app (backend + frontend)

**Set up `.env`** (project root) — copy `.env.example` and fill in:
- `OPENAI_API_KEY` — required
- `MINDAI_JWT_SECRET` — generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`

**Install and start the backend:**
```
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --port 8001
```
(Port 8000 may already be in use by something else on your machine — check
first with `netstat -ano | findstr :8000` on Windows, or pick any free
port and update `frontend/vite.config.ts`'s proxy target to match.)

**Install and start the frontend** (separate terminal):
```
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** — sign up, ingest a repo from the sidebar,
ask it questions.

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
- **CPU-only embedding/reranking is slow.** A few hundred chunks take
  minutes to embed on CPU. A GPU (or a cloud embedding endpoint) will be
  much faster if ingesting larger repos regularly.
- **Chat is not token-streamed.** The SSE endpoint streams status updates
  and the finished answer, not token-by-token generation — the LangGraph
  pipeline returns its final state as one unit. True token streaming would
  mean changing `generate_node`'s OpenAI call to `stream=True`.
