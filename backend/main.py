"""
main.py — FastAPI application.

This layer is a thin wrapper around the RAG backbone (ingestion.py,
graph.py) — it owns HTTP concerns (auth, request/response shapes, DB
persistence of users/conversations) and nothing about retrieval,
chunking, fusion, reranking, or generation. Every call into the pipeline
goes through ingestion.ingest_repository / graph.answer_question exactly
as those modules define them; this file does not reimplement or shortcut
any of that logic.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Must run before any backend.* import: backend.auth and backend.db read
# MINDAI_JWT_SECRET / DATABASE_URL from os.environ at module import time.
# graph.py also calls load_dotenv(), but that happens later in this same
# import chain (see the `from graph import ...` below) — too late to affect
# the backend modules imported above it. Calling it here, first, ensures
# .env is loaded before anything reads from the environment.
load_dotenv()

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.auth import create_access_token, get_current_user, hash_password, verify_password
from backend.db import get_db, init_db
from backend.models import Conversation, Message, Repo, User, generate_id
from backend.pipeline_store import (
    collection_name_for_repo,
    get_shared_qdrant_client,
    load_call_graph,
    locked_qdrant_client,
    save_call_graph,
)
from backend.schemas import (
    ChatRequest,
    ConversationResponse,
    CreateConversationRequest,
    IngestRepoRequest,
    LoginRequest,
    MessageResponse,
    RepoResponse,
    SignUpRequest,
    TokenResponse,
    UserResponse,
)
from graph import PipelineDependencies, build_graph
from ingestion import ingest_repository

logger = logging.getLogger("mindai.backend")

# --------------------------------------------------------------------------- #
# App-lifetime pipeline singletons
#
# One PipelineDependencies (embedding + reranker models loaded once) and one
# compiled LangGraph are shared across every request/repo/user for the life
# of the process — see graph.py's PipelineDependencies docstring for why
# this must not be rebuilt per-request.
# --------------------------------------------------------------------------- #

_pipeline_deps: PipelineDependencies | None = None
_compiled_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline_deps, _compiled_graph
    init_db()
    client = get_shared_qdrant_client()
    _pipeline_deps = PipelineDependencies(client)
    _compiled_graph = build_graph(_pipeline_deps)
    logger.info("MindAI backend ready: pipeline models loaded, DB initialized.")
    yield


app = FastAPI(title="MindAI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Auth routes
# --------------------------------------------------------------------------- #

@app.post("/auth/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def signup(payload: SignUpRequest, db: Session = Depends(get_db)) -> TokenResponse:
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists.")

    user = User(email=payload.email, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    return TokenResponse(access_token=create_access_token(user.id))


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    invalid_credentials = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password."
    )
    user = db.query(User).filter(User.email == payload.email).first()
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise invalid_credentials

    return TokenResponse(access_token=create_access_token(user.id))


@app.get("/auth/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)) -> User:
    return current_user


# --------------------------------------------------------------------------- #
# Repo ingestion routes
# --------------------------------------------------------------------------- #

def _run_ingestion(repo_id: str, github_url: str, collection_name: str) -> None:
    """
    Background job body: runs the (multi-minute) ingestion backbone call
    unmodified, then updates the Repo row's status. Runs after the HTTP
    response has already been sent (FastAPI BackgroundTasks), so it opens
    its own DB session rather than reusing the request-scoped one, which
    closes when the request finishes.
    """
    from backend.db import SessionLocal

    db = SessionLocal()
    try:
        repo_row = db.get(Repo, repo_id)
        if repo_row is None:
            logger.error("Repo row %s vanished before background ingestion started", repo_id)
            return

        repo_row.status = "ingesting"
        db.commit()

        try:
            # Locked for the whole call (not just the Qdrant writes at the
            # tail end) because local-mode Qdrant is not thread-safe even
            # for reads — see pipeline_store.py's module docstring. This
            # means repo ingestion is fully serialized across all repos;
            # the clone/chunking phase pays that cost too even though it
            # never touches Qdrant, which is the accepted tradeoff of
            # local mode over running a real Qdrant server.
            with locked_qdrant_client() as client:
                result, _client, call_graph = ingest_repository(
                    github_url,
                    collection_name=collection_name,
                    qdrant_client=client,
                )
        except Exception as e:
            logger.exception("Ingestion failed for repo %s (%s)", repo_id, github_url)
            repo_row.status = "failed"
            repo_row.error_message = str(e)
            db.commit()
            return

        save_call_graph(collection_name, call_graph)
        repo_row.status = "ready"
        repo_row.total_chunks = result.total_chunks
        db.commit()
    finally:
        db.close()


@app.post("/repos", response_model=RepoResponse, status_code=status.HTTP_202_ACCEPTED)
def ingest_repo(
    payload: IngestRepoRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Repo:
    """
    Creates the Repo row immediately (status "pending") and returns it —
    the actual clone/chunk/embed/upsert work (several minutes on CPU) runs
    as a FastAPI background task after the response is sent, rather than
    holding the HTTP connection open for the whole duration. The frontend
    polls GET /repos (or is expected to) until status flips to "ready" or
    "failed".

    The row's id is generated explicitly via generate_id() rather than
    relying on the column's `default=` — that default only fires at
    flush/insert time, not at object construction, so reading `.id` off a
    not-yet-flushed Repo() would return None. That bug previously made
    every ingestion write into a shared "repo_None" Qdrant collection
    instead of one unique per repo, and crashed on retry with a UNIQUE
    constraint violation on qdrant_collection_name.
    """
    repo_id = generate_id()
    collection_name = collection_name_for_repo(repo_id)
    repo_row = Repo(
        id=repo_id,
        owner_id=current_user.id,
        github_url=payload.github_url,
        display_name=payload.github_url.rstrip("/").rsplit("/", 1)[-1],
        qdrant_collection_name=collection_name,
        total_chunks=0,
        status="pending",
    )
    db.add(repo_row)
    db.commit()
    db.refresh(repo_row)

    background_tasks.add_task(_run_ingestion, repo_id, payload.github_url, collection_name)
    return repo_row


@app.get("/repos", response_model=list[RepoResponse])
def list_repos(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[Repo]:
    return db.query(Repo).filter(Repo.owner_id == current_user.id).order_by(Repo.created_at.desc()).all()


# --------------------------------------------------------------------------- #
# Conversation routes
# --------------------------------------------------------------------------- #

@app.post("/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: CreateConversationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Conversation:
    repo = db.get(Repo, payload.repo_id)
    if repo is None or repo.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repo not found.")

    conversation = Conversation(owner_id=current_user.id, repo_id=repo.id, title=repo.display_name)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


@app.get("/conversations", response_model=list[ConversationResponse])
def list_conversations(
    repo_id: str | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[Conversation]:
    """
    The caller's conversations, newest first, optionally narrowed to one repo.

    Without this the client had no way to find a conversation it had already
    started: only POST /conversations existed, so every page load created a
    fresh empty one and the user's history — still sitting in the messages
    table — was never shown again. Filtering by owner_id (not just by repo)
    is what keeps one user's history out of another's list, matching the
    ownership check every other conversation route performs.
    """
    query = db.query(Conversation).filter(Conversation.owner_id == current_user.id)
    if repo_id is not None:
        query = query.filter(Conversation.repo_id == repo_id)
    return query.order_by(Conversation.created_at.desc()).all()


@app.get("/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
def get_messages(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[Message]:
    conversation = _get_owned_conversation(conversation_id, current_user, db)
    return conversation.messages


def _get_owned_conversation(conversation_id: str, current_user: User, db: Session) -> Conversation:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None or conversation.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return conversation


# --------------------------------------------------------------------------- #
# Chat route (SSE streaming)
# --------------------------------------------------------------------------- #

@app.post("/chat")
async def chat(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Runs one question through the unmodified LangGraph pipeline
    (graph.answer_question) and streams the result back over SSE.

    Note: the pipeline itself (query_rewrite -> hybrid_retrieve -> rerank ->
    graph_expand? -> generate) is not internally streamed — LangGraph
    returns the final state once `generate` completes. What's streamed here
    is transport framing (status events, then the finished answer), not
    token-by-token generation; true token streaming would require changing
    generate_node's OpenAI call to stream=True, which is a backbone change
    outside this HTTP layer's scope.
    """
    conversation = _get_owned_conversation(payload.conversation_id, current_user, db)
    repo = db.get(Repo, conversation.repo_id)
    if repo is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repo not found.")
    if repo.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Repo is not ready yet (status: {repo.status}). Wait for ingestion to finish before asking questions.",
        )

    call_graph = load_call_graph(repo.qdrant_collection_name)

    def _run_pipeline():
        from graph import answer_question

        # Locked for the whole pipeline run (retrieval + rerank + optional
        # graph expansion + generation), not just the individual Qdrant
        # calls inside it — local-mode Qdrant is not thread-safe even for
        # reads (see pipeline_store.py), so two questions running
        # concurrently must not interleave their Qdrant access at all.
        # This does mean the OpenAI calls inside the same run also sit
        # behind the lock, serializing chat across all users; the
        # alternative (fine-grained locking only around Qdrant calls
        # inside graph.py) would mean modifying the backbone's node
        # functions for an HTTP-layer concern, which is out of scope here.
        with locked_qdrant_client():
            return answer_question(
                question=payload.question,
                qdrant_client=_pipeline_deps.qdrant_client,
                call_graph=call_graph,
                collection_name=repo.qdrant_collection_name,
                deps=_pipeline_deps,
                compiled_graph=_compiled_graph,
            )

    async def event_stream():
        import json

        yield f"data: {json.dumps({'type': 'status', 'stage': 'retrieving'})}\n\n"

        try:
            # _run_pipeline is synchronous/blocking (OpenAI calls, Qdrant
            # queries, CrossEncoder inference) by design — that's correct
            # for CLI use, but calling it directly inside this async
            # generator would stall FastAPI's single event loop for the
            # whole pipeline duration, freezing every other concurrent
            # user's requests. run_in_threadpool offloads it to a worker
            # thread so the event loop stays free.
            final_state = await run_in_threadpool(_run_pipeline)
        except Exception:
            logger.exception("Pipeline failure while answering question for conversation %s", conversation.id)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Something went wrong answering that question. Please try again.'})}\n\n"
            return

        answer_text = final_state.get("answer", "")
        refused = bool(final_state.get("refused", False))

        # Both messages are saved together, only after the pipeline
        # succeeds, so a failed run (see the except block above) never
        # leaves an orphaned question with no reply in the conversation.
        user_message = Message(conversation_id=conversation.id, role="user", content=payload.question)
        assistant_message = Message(
            conversation_id=conversation.id, role="assistant", content=answer_text, refused=refused
        )
        db.add_all([user_message, assistant_message])
        db.commit()

        yield f"data: {json.dumps({'type': 'answer', 'content': answer_text, 'refused': refused})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
