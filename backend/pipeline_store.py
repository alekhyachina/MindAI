"""
pipeline_store.py — Shared, process-wide access to the Qdrant client and
per-repo call graphs.

Architectural note: ingestion.py and graph.py were designed as standalone
CLI-callable modules, each taking a collection_name/qdrant_path and
returning what they need directly. The web server wraps them without
modifying that contract: one persistent on-disk Qdrant client is shared
across all requests (multiple named collections, one per ingested repo),
and call graphs — an in-memory return value from ingest_repository — are
persisted to a small JSON file per collection so they survive server
restarts without requiring a second database table.

Thread-safety note: qdrant-client's LOCAL mode (`QdrantClient(path=...)`)
is NOT thread-safe. Its in-memory LocalCollection mutates plain Python
lists and numpy arrays across multiple non-atomic steps with no internal
locking, for both reads and writes (confirmed by reading qdrant_client's
local/local_collection.py source, and by a live upstream bug —
qdrant/qdrant-client#1193 — reproducing exactly this race under
concurrent upsert/query calls). Since the FastAPI backend serves this one
client instance from multiple threadpool worker threads, every call
against it MUST go through `locked_qdrant_client()` below rather than
touching the raw client directly.

That lock serializes Qdrant access server-wide, which is a real
throughput ceiling — only one ingestion or one chat question at a time.
It is a stop-gap for local mode, not a scaling strategy, and setting
QDRANT_URL lifts it: in SERVER mode the client is a stateless HTTP
wrapper and concurrency is the server's job, so `locked_qdrant_client()`
skips the lock entirely and concurrent requests actually run
concurrently. The contract for callers is unchanged either way — always
go through `locked_qdrant_client()`, and it does the right thing for
whichever mode is configured.
"""

from __future__ import annotations

import json
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from qdrant_client import QdrantClient

from ingestion import get_qdrant_client

QDRANT_STORAGE_PATH = os.environ.get("MINDAI_QDRANT_PATH", "./qdrant_storage")
CALL_GRAPH_DIR = Path(os.environ.get("MINDAI_CALL_GRAPH_DIR", "./call_graphs"))

# Read once at import: whether QDRANT_URL is set decides local-vs-server mode
# for the life of the process, and get_qdrant_client() reads the same variable.
SERVER_MODE = bool(os.environ.get("QDRANT_URL", "").strip())

_setup_lock = threading.Lock()
_qdrant_call_lock = threading.RLock()
_client: QdrantClient | None = None


def get_shared_qdrant_client() -> QdrantClient:
    """
    One QdrantClient per process. In local mode it is backed by an on-disk
    path so ingested repos survive server restarts, and Qdrant's local mode
    does not support concurrent client instances against the same path, so
    this must stay a singleton. In server mode (QDRANT_URL set) the path is
    ignored and persistence is the remote cluster's; the singleton is kept
    there simply to reuse one connection pool.

    Returns the raw client for call sites that only need to pass it
    through (e.g. into PipelineDependencies). Actual method calls against
    it must go through `locked_qdrant_client()`, not be made on this
    return value directly — see module docstring.
    """
    global _client
    with _setup_lock:
        if _client is None:
            _client = get_qdrant_client(QDRANT_STORAGE_PATH)
        return _client


@contextmanager
def locked_qdrant_client() -> Iterator[QdrantClient]:
    """
    Yields the shared QdrantClient, holding the process-wide Qdrant call
    lock only when it is actually needed. Use for every read or write:

        with locked_qdrant_client() as client:
            client.upsert(...)

    In LOCAL mode the lock is mandatory (see module docstring). In SERVER
    mode it is skipped — holding it there would serialize every request
    against a server that handles concurrency perfectly well on its own,
    reintroducing the exact bottleneck that moving to server mode removes.
    """
    if SERVER_MODE:
        yield get_shared_qdrant_client()
        return
    with _qdrant_call_lock:
        yield get_shared_qdrant_client()


def collection_name_for_repo(repo_id: str) -> str:
    # Qdrant collection names are unconstrained, but keep them filesystem-
    # and URL-safe since repo_id is also used to derive the call-graph
    # cache filename below.
    return f"repo_{repo_id}"


def _call_graph_path(collection_name: str) -> Path:
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", collection_name)
    return CALL_GRAPH_DIR / f"{safe_name}.json"


def save_call_graph(collection_name: str, call_graph: dict) -> None:
    CALL_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    path = _call_graph_path(collection_name)
    path.write_text(json.dumps(call_graph), encoding="utf-8")


def load_call_graph(collection_name: str) -> dict:
    path = _call_graph_path(collection_name)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
