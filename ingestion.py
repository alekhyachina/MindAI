"""
ingestion.py — Repository ingestion pipeline for MindAI.

Pipeline: shallow clone -> noise filtering -> AST-aware chunking (tree-sitter)
-> hybrid (dense + sparse) vectorization via FastEmbed -> upsert into Qdrant.

Architectural notes:
  - We use tree-sitter (not fixed-token windows) so that chunk boundaries align
    with logical code units (functions/classes/methods). This preserves
    semantic coherence in each chunk and gives us precise, verifiable
    start_line/end_line metadata for citation.
  - Hybrid vectors (dense semantic + sparse BM25) are stored as *named vectors*
    on the same Qdrant point, so a single upsert produces a point retrievable
    by either search modality (see graph.py's RRF fusion).
  - A lightweight call graph (caller -> callee edges, regex-derived from the
    chunk source) is persisted alongside the chunks so graph.py can do
    "graph expansion" for trace/call-chain questions without a second parse.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import tiktoken
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tree_sitter_language_pack import get_parser

from dense_embeddings import DENSE_MODEL_NAME, DENSE_VECTOR_SIZE, DenseEmbedder
from semantic_chunking import compute_semantic_spans, symbol_name_for_span

# Matches graph.py: this module is a documented standalone CLI entrypoint
# (`python ingestion.py <repo-url>`), so it cannot rely on the FastAPI app
# having loaded .env first. Without this, a CLI run picks up neither
# OPENAI_API_KEY nor QDRANT_URL, and would quietly write to a different
# Qdrant than the server reads from. Harmless under the server, which calls
# load_dotenv() before importing this module.
load_dotenv()

logger = logging.getLogger("mindai.ingestion")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# DENSE_MODEL_NAME / DENSE_VECTOR_SIZE now come from dense_embeddings.py
# (OpenAI text-embedding-3-small, hosted — see that module's docstring for
# why dense embedding moved off local CPU inference while sparse/BM25 stayed).
SPARSE_MODEL_NAME = "Qdrant/bm25"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
COLLECTION_NAME = "mindai_repo_chunks"

# Chunking strategy: "ast" (default) cuts at tree-sitter syntactic
# boundaries; "semantic" cuts where the OpenAI embedding model reports the
# file changing topic (see semantic_chunking.py, including its tradeoffs).
# Both produce exact start_line/end_line, so citations are unaffected by the
# choice — but they produce DIFFERENT chunks, so switching strategies means
# re-ingesting any repo whose collection was built under the other one.
CHUNK_STRATEGY = os.environ.get("MINDAI_CHUNK_STRATEGY", "ast").strip().lower()

# qdrant-client defaults to a 5s timeout, which is fine for local mode (no
# network) but too tight for server mode: a batched upsert of a few thousand
# chunks to a cloud cluster regularly takes longer than that, and the timeout
# surfaces as a failed ingestion rather than a slow one.
QDRANT_SERVER_TIMEOUT_SECONDS = 60

# Files larger than this are almost always generated/vendored/binary-adjacent;
# skip rather than pay tree-sitter parse cost for no benefit.
MAX_FILE_SIZE_BYTES = 1_000_000

# Fixed-window fallback size (lines) for files tree-sitter can't/shouldn't parse.
FALLBACK_CHUNK_LINES = 100
FALLBACK_CHUNK_OVERLAP = 20

# OpenAI's embeddings API hard-rejects any single input over 8192 tokens
# (real failure hit in production: a single AST-matched chunk — e.g. one
# very large function/class with no internal split points — exceeded this
# and failed the whole upsert batch with a 400). Unlike FALLBACK_CHUNK_LINES,
# AST chunking has no inherent size cap: tree-sitter chunks by syntactic
# unit, and a syntactic unit can be arbitrarily large. Set well under 8192
# (not right at it) so token-counting differences between our tokenizer and
# OpenAI's don't put us right on the edge of the same failure.
MAX_CHUNK_TOKENS = 6000

# Directories that are pure noise for a code-understanding RAG system.
NOISE_DIR_NAMES = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", ".env",
    "node_modules", "bower_components",
    "dist", "build", "out", "target", "bin", "obj",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    ".idea", ".vscode", ".vs",
    "coverage", ".nyc_output", "htmlcov",
    "vendor", "third_party", "third-party",
    ".next", ".nuxt", ".svelte-kit",
    "site-packages", "egg-info",
}

# Extensions that are binary/media/lockfile noise — never worth embedding.
NOISE_EXTENSIONS = {
    # binaries / archives
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".lib", ".pyd",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".jar", ".war",
    # media
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".flac", ".ogg",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    # documents / data blobs
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".db", ".sqlite", ".sqlite3", ".parquet",
    # lockfiles (keep out of chunk store, they are not "code")
    ".lock",
}

NOISE_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "Cargo.lock", "Gemfile.lock", "composer.lock", "go.sum",
    ".DS_Store", "Thumbs.db",
}

# Extension -> tree-sitter language identifier (as understood by
# tree_sitter_language_pack.get_parser).
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala",
}

# Tree-sitter node types that constitute a "logical chunk" boundary, per
# language. These are the syntactic units we chunk on instead of fixed
# token windows — functions, methods, classes.
CHUNK_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition"},
    "javascript": {
        "function_declaration", "class_declaration", "method_definition",
        "arrow_function", "generator_function_declaration",
    },
    "typescript": {
        "function_declaration", "class_declaration", "method_definition",
        "interface_declaration", "arrow_function",
    },
    "tsx": {
        "function_declaration", "class_declaration", "method_definition",
        "interface_declaration", "arrow_function",
    },
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "java": {"method_declaration", "class_declaration", "interface_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "trait_item"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "c_sharp": {"method_declaration", "class_declaration", "interface_declaration"},
    "ruby": {"method", "class", "module"},
    "php": {"function_definition", "method_declaration", "class_declaration"},
    "swift": {"function_declaration", "class_declaration"},
    "kotlin": {"function_declaration", "class_declaration"},
    "scala": {"function_definition", "class_definition", "object_definition"},
}

# Tree-sitter node types for a file's import/include statements, per
# language. Imports are a FILE-level property, not a chunk-level one — they
# are extracted once per file and attached to every chunk from it, so a
# retrieved function carries the context of what its file depends on
# (which framework, which internal module) even though the import lines
# themselves live outside the function's own line range.
IMPORT_NODE_TYPES: dict[str, set[str]] = {
    "python": {"import_statement", "import_from_statement", "future_import_statement"},
    "javascript": {"import_statement"},
    "typescript": {"import_statement", "import_alias"},
    "tsx": {"import_statement", "import_alias"},
    "go": {"import_declaration"},
    "java": {"import_declaration"},
    "rust": {"use_declaration", "extern_crate_declaration"},
    "c": {"preproc_include"},
    "cpp": {"preproc_include", "using_declaration"},
    "c_sharp": {"using_directive"},
    "ruby": set(),  # `require` is a method call, not a node type — regex only
    "php": {"namespace_use_declaration"},
    "swift": {"import_declaration"},
    "kotlin": {"import_header"},
    "scala": {"import_declaration"},
}

# Fallback import matcher, used for languages with no IMPORT_NODE_TYPES
# entry (ruby), for the fixed-window fallback path (which has no parse
# tree), and whenever tree-sitter parsing failed. Deliberately line-based
# and conservative: an import line in every language this project indexes
# starts with one of these keywords.
_IMPORT_LINE_PATTERN = re.compile(
    r"^\s*(?:from\s+\S+\s+import\b|import\b|#include\b|using\b|use\b|require\b|"
    r"require_relative\b|extern\s+crate\b|package\b)"
)

# Cap on how many import lines are attached to a chunk. A generated or
# barrel file can carry hundreds; past a point they stop being useful
# context and just inflate every payload from that file.
MAX_IMPORTS_PER_CHUNK = 40


@dataclass
class CodeChunk:
    """A single logical unit of source code plus its provenance metadata."""

    chunk_id: str
    file_path: str
    symbol_name: str
    symbol_type: str
    language: str
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    content: str
    # Best-effort caller/callee names found via regex scan of the chunk body,
    # used by graph.py for lightweight call-graph expansion.
    calls: list[str] = field(default_factory=list)
    # Import/include lines of the FILE this chunk came from (not of the chunk
    # itself — see IMPORT_NODE_TYPES). Gives a retrieved function the
    # dependency context that its own line range cannot contain.
    imports: list[str] = field(default_factory=list)


@dataclass
class IngestionResult:
    repo_url: str
    total_files_scanned: int
    total_files_skipped: int
    total_chunks: int
    collection_name: str


# --------------------------------------------------------------------------- #
# Step 1: Shallow clone
# --------------------------------------------------------------------------- #

def shallow_clone(github_url: str) -> Path:
    """
    Shallow-clone `github_url` (--depth 1) into a fresh temp directory and
    return its path. Caller is responsible for cleanup (see
    `cleanup_clone`), typically via a try/finally in the ingestion entrypoint.
    """
    dest = Path(tempfile.mkdtemp(prefix="mindai_clone_"))
    logger.info("Shallow cloning %s -> %s", github_url, dest)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch", github_url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.CalledProcessError as e:
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError(f"git clone failed for {github_url}: {e.stderr.strip()}") from e
    except subprocess.TimeoutExpired as e:
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError(f"git clone timed out for {github_url}") from e
    return dest


def cleanup_clone(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Step 2: Noise filtering / file discovery
# --------------------------------------------------------------------------- #

def discover_source_files(repo_root: Path) -> Iterator[Path]:
    """
    Walk `repo_root`, aggressively pruning noise directories, and yield paths
    to files worth parsing. Pruning noise *directories* (rather than
    filtering after the fact) avoids descending into node_modules/.venv
    entirely, which matters for repos where those directories dwarf the
    actual source.
    """
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in NOISE_DIR_NAMES for part in path.parts):
            continue
        if path.name in NOISE_FILENAMES:
            continue
        if path.suffix.lower() in NOISE_EXTENSIONS:
            continue
        if path.name.endswith((".min.js", ".min.css")):
            continue
        try:
            if path.stat().st_size > MAX_FILE_SIZE_BYTES:
                continue
            if path.stat().st_size == 0:
                continue
        except OSError:
            continue
        yield path


# --------------------------------------------------------------------------- #
# Step 3: AST chunking
# --------------------------------------------------------------------------- #

_CALL_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# Common keywords that match the call-pattern regex but are not function
# calls — filtered out of the derived call graph to reduce noise.
_CALL_STOPWORDS = {
    "if", "for", "while", "switch", "catch", "return", "def", "function",
    "class", "print", "super", "new", "async", "await", "yield", "with",
    "elif", "except", "case", "sizeof", "typeof", "instanceof",
}


def _extract_calls(source: str) -> list[str]:
    """Best-effort regex extraction of identifiers used in call position."""
    found = {
        m.group(1)
        for m in _CALL_PATTERN.finditer(source)
        if m.group(1) not in _CALL_STOPWORDS
    }
    return sorted(found)


def _extract_imports_regex(source_text: str) -> list[str]:
    """
    Line-based import extraction, for when no parse tree is available.
    Order is preserved (import blocks read top-to-bottom) and duplicates
    dropped, rather than sorting — an import list is more legible as
    written than alphabetised.
    """
    seen: list[str] = []
    for line in source_text.splitlines():
        if len(seen) >= MAX_IMPORTS_PER_CHUNK:
            break
        if _IMPORT_LINE_PATTERN.match(line):
            stripped = line.strip()
            if stripped not in seen:
                seen.append(stripped)
    return seen


def _extract_imports_ast(root_node, source_bytes: bytes, language: str, source_text: str) -> list[str]:
    """
    Collect a file's import statements from its parse tree, falling back to
    the regex scan for languages whose imports aren't a distinct node type
    (ruby's `require` is a method call) or that yield nothing.

    Only the tree's top level is scanned, not every nested node: imports are
    file-level in every language here, and a full walk would also pick up
    function-local imports, which say more about one branch than about the
    file's dependencies.
    """
    wanted = IMPORT_NODE_TYPES.get(language)
    if not wanted:
        return _extract_imports_regex(source_text)

    found: list[str] = []
    for node in root_node.children:
        if node.type not in wanted:
            continue
        text = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace").strip()
        if text and text not in found:
            found.append(text)
        if len(found) >= MAX_IMPORTS_PER_CHUNK:
            break

    # Go and Rust group imports inside a parenthesised/braced block, so the
    # node text is one multi-line string; some grammars also nest the real
    # import nodes one level down. A regex pass is the cheaper way to cover
    # both than special-casing each grammar's shape.
    return found or _extract_imports_regex(source_text)


def _node_symbol_name(node, source_bytes: bytes) -> str | None:
    """Extract the identifier name for a tree-sitter definition node."""
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="replace")
    # Fallback: first identifier-like child (covers grammars that don't
    # expose a "name" field, e.g. some Go/Rust node shapes).
    for child in node.children:
        if "identifier" in child.type:
            return source_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
    return None


def _walk_chunk_nodes(node, wanted_types: set[str]):
    """Depth-first walk yielding nodes whose type is a chunk boundary."""
    if node.type in wanted_types:
        yield node
        # Do not descend further into a matched node's children for the
        # *same* wanted types — this avoids re-chunking a method that is
        # already inside a class chunk. We still descend to catch nested
        # definitions of *different* semantic value (e.g. nested classes),
        # so we recurse but children will simply yield again if they match.
    for child in node.children:
        yield from _walk_chunk_nodes(child, wanted_types)


def chunk_file_with_ast(path: Path, repo_root: Path) -> list[CodeChunk]:
    """
    Parse `path` with tree-sitter and emit one CodeChunk per matched
    syntactic unit (function/class/method/etc). Falls back to fixed-line
    windowing if the language is unsupported or parsing fails.
    """
    rel_path = str(path.relative_to(repo_root)).replace("\\", "/")
    language = EXTENSION_TO_LANGUAGE.get(path.suffix.lower())

    try:
        source_text = path.read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, OSError):
        return []  # binary or unreadable; already should have been filtered

    if language is None or language not in CHUNK_NODE_TYPES:
        return _chunk_file_fixed_window(rel_path, source_text, language or "text")

    try:
        parser = get_parser(language)
        source_bytes = source_text.encode("utf-8")
        tree = parser.parse(source_bytes)
    except Exception as e:  # tree-sitter grammar issues, encoding edge cases
        logger.warning("tree-sitter parse failed for %s (%s); falling back to fixed windows", rel_path, e)
        return _chunk_file_fixed_window(rel_path, source_text, language)

    wanted_types = CHUNK_NODE_TYPES[language]
    # Extracted once per file, then attached to every chunk below.
    file_imports = _extract_imports_ast(tree.root_node, source_bytes, language, source_text)
    chunks: list[CodeChunk] = []
    seen_spans: set[tuple[int, int]] = set()

    for node in _walk_chunk_nodes(tree.root_node, wanted_types):
        span = (node.start_byte, node.end_byte)
        if span in seen_spans:
            continue
        seen_spans.add(span)

        symbol_name = _node_symbol_name(node, source_bytes) or "<anonymous>"
        content = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        chunks.append(
            CodeChunk(
                chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{rel_path}:{start_line}:{end_line}:{symbol_name}")),
                file_path=rel_path,
                symbol_name=symbol_name,
                symbol_type=node.type,
                language=language,
                start_line=start_line,
                end_line=end_line,
                content=content,
                calls=_extract_calls(content),
                imports=file_imports,
            )
        )

    if not chunks:
        # File parsed fine but had no matched top-level constructs (e.g. a
        # config-like .py file with only module-level statements) — still
        # index it as a single whole-file chunk so it's retrievable.
        return _chunk_file_fixed_window(rel_path, source_text, language)

    return chunks


def _chunk_file_fixed_window(rel_path: str, source_text: str, language: str) -> list[CodeChunk]:
    """Fallback: fixed-line windows with overlap, for unparseable content."""
    lines = source_text.splitlines()
    if not lines:
        return []

    # No parse tree on this path by definition, so imports come from the
    # regex scan — same metadata contract as the AST path.
    file_imports = _extract_imports_regex(source_text)
    chunks: list[CodeChunk] = []
    step = FALLBACK_CHUNK_LINES - FALLBACK_CHUNK_OVERLAP
    for i in range(0, len(lines), step):
        window = lines[i : i + FALLBACK_CHUNK_LINES]
        if not window:
            continue
        start_line = i + 1
        end_line = i + len(window)
        content = "\n".join(window)
        chunks.append(
            CodeChunk(
                chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{rel_path}:{start_line}:{end_line}")),
                file_path=rel_path,
                symbol_name=f"{Path(rel_path).stem}[{start_line}-{end_line}]",
                symbol_type="text_window",
                language=language,
                start_line=start_line,
                end_line=end_line,
                content=content,
                calls=_extract_calls(content),
                imports=file_imports,
            )
        )
        if end_line >= len(lines):
            break
    return chunks


_SEMANTIC_EMBEDDER: DenseEmbedder | None = None


def _semantic_embedder() -> DenseEmbedder:
    """
    One DenseEmbedder for the whole ingestion run. Built lazily so that the
    default "ast" strategy never constructs an OpenAI client it won't use,
    and reused across files so every file doesn't pay client setup.
    """
    global _SEMANTIC_EMBEDDER
    if _SEMANTIC_EMBEDDER is None:
        _SEMANTIC_EMBEDDER = DenseEmbedder()
    return _SEMANTIC_EMBEDDER


def chunk_file_semantic(path: Path, repo_root: Path) -> list[CodeChunk]:
    """
    Chunk `path` at embedding-derived semantic boundaries rather than
    syntactic ones. Same CodeChunk contract as chunk_file_with_ast: exact
    1-indexed line spans, regex-extracted `calls` so graph.py's call-graph
    expansion keeps working, and a best-effort symbol_name.

    symbol_type is "semantic_chunk" rather than a tree-sitter node type,
    which is what lets a collection built under this strategy be told apart
    from an AST-built one when debugging retrieval.
    """
    rel_path = str(path.relative_to(repo_root)).replace("\\", "/")
    language = EXTENSION_TO_LANGUAGE.get(path.suffix.lower()) or "text"

    try:
        source_text = path.read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, OSError):
        return []  # binary or unreadable; already should have been filtered

    lines = source_text.splitlines()
    file_imports = _extract_imports_regex(source_text)
    chunks: list[CodeChunk] = []
    for start_line, end_line in compute_semantic_spans(source_text, _semantic_embedder()):
        content = "\n".join(lines[start_line - 1 : end_line])
        if not content.strip():
            continue  # a span of only blank lines carries nothing to retrieve
        symbol_name = symbol_name_for_span(content) or f"{Path(rel_path).stem}[{start_line}-{end_line}]"
        chunks.append(
            CodeChunk(
                chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{rel_path}:{start_line}:{end_line}:{symbol_name}")),
                file_path=rel_path,
                symbol_name=symbol_name,
                symbol_type="semantic_chunk",
                language=language,
                start_line=start_line,
                end_line=end_line,
                content=content,
                calls=_extract_calls(content),
                imports=file_imports,
            )
        )
    return chunks


def chunk_file(path: Path, repo_root: Path) -> list[CodeChunk]:
    """
    Chunk one file using the configured strategy (MINDAI_CHUNK_STRATEGY).
    Single entrypoint so ingest_repository doesn't branch, and so both
    strategies are guaranteed to get the same oversized-chunk handling
    applied downstream.
    """
    if CHUNK_STRATEGY == "semantic":
        return chunk_file_semantic(path, repo_root)
    return chunk_file_with_ast(path, repo_root)


_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _split_oversized_chunk(chunk: CodeChunk) -> list[CodeChunk]:
    """
    If `chunk.content` exceeds MAX_CHUNK_TOKENS, split it into consecutive
    sub-chunks by line, each within budget, preserving citation accuracy
    (each sub-chunk gets its own correct start_line/end_line rather than
    the whole oversized span). No-op (returns [chunk]) for chunks already
    within budget — the common case, so this is cheap to call for every
    chunk regardless of source (AST or fallback window).
    """
    token_count = len(_TOKENIZER.encode(chunk.content, disallowed_special=()))
    if token_count <= MAX_CHUNK_TOKENS:
        return [chunk]

    lines = chunk.content.splitlines()
    if len(lines) <= 1:
        # A single unsplittable line over budget (e.g. a minified file that
        # slipped past noise filtering, or one enormous string literal) —
        # hard-truncate at the token level as a last resort so it can still
        # be embedded, rather than dropping it or failing ingestion.
        truncated_tokens = _TOKENIZER.encode(chunk.content, disallowed_special=())[:MAX_CHUNK_TOKENS]
        truncated_content = _TOKENIZER.decode(truncated_tokens)
        return [
            CodeChunk(
                chunk_id=chunk.chunk_id,
                file_path=chunk.file_path,
                symbol_name=chunk.symbol_name,
                symbol_type=chunk.symbol_type,
                language=chunk.language,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                content=truncated_content,
                calls=chunk.calls,
                imports=chunk.imports,
            )
        ]

    # Binary-search-free approach: split by line count proportional to the
    # token overage, then recurse — handles pathological cases (a few huge
    # lines among many small ones) without assuming uniform token density.
    midpoint = len(lines) // 2
    first_half_lines = lines[:midpoint]
    second_half_lines = lines[midpoint:]
    first_half_line_count = len(first_half_lines)

    first_chunk = CodeChunk(
        chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.chunk_id}:split:0")),
        file_path=chunk.file_path,
        symbol_name=f"{chunk.symbol_name}[part1]",
        symbol_type=chunk.symbol_type,
        language=chunk.language,
        start_line=chunk.start_line,
        end_line=chunk.start_line + first_half_line_count - 1,
        content="\n".join(first_half_lines),
        calls=chunk.calls,
        imports=chunk.imports,
    )
    second_chunk = CodeChunk(
        chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.chunk_id}:split:1")),
        file_path=chunk.file_path,
        symbol_name=f"{chunk.symbol_name}[part2]",
        symbol_type=chunk.symbol_type,
        language=chunk.language,
        start_line=chunk.start_line + first_half_line_count,
        end_line=chunk.end_line,
        content="\n".join(second_half_lines),
        calls=chunk.calls,
        imports=chunk.imports,
    )
    # Recurse — a half that's still oversized (very unevenly distributed
    # content) keeps splitting until every piece is within budget.
    return _split_oversized_chunk(first_chunk) + _split_oversized_chunk(second_chunk)


# --------------------------------------------------------------------------- #
# Step 4: Hybrid vectorization + Qdrant upsert
# --------------------------------------------------------------------------- #

def get_qdrant_client(path: str | None = None) -> QdrantClient:
    """
    Return a Qdrant client, in one of three modes:

      1. SERVER mode, if QDRANT_URL is set — talks to a remote Qdrant
         (Qdrant Cloud or a local container) over HTTP. This takes
         precedence over `path`, so setting the env var is all that's
         needed to move an existing deployment off local mode. Unlike
         local mode, the server handles concurrency itself, so callers do
         not need to serialize access (see backend/pipeline_store.py).
      2. MEMORY mode, if `path=None` and no QDRANT_URL — fully in-memory,
         non-persistent (fastest; good for demos/ephemeral sessions).
      3. LOCAL mode — an on-disk path that persists across restarts, but
         is NOT thread-safe (see backend/pipeline_store.py's docstring).

    QDRANT_API_KEY is required by Qdrant Cloud and ignored by an
    unauthenticated local container, so it is passed through as None when
    unset rather than being treated as an error.
    """
    url = os.environ.get("QDRANT_URL", "").strip()
    if url:
        # Qdrant Cloud hands out URLs with a trailing slash; qdrant-client
        # concatenates paths onto this value, so leaving it in produces
        # double-slashed request paths.
        return QdrantClient(
            url=url.rstrip("/"),
            api_key=os.environ.get("QDRANT_API_KEY") or None,
            timeout=QDRANT_SERVER_TIMEOUT_SECONDS,
        )
    if path is None:
        return QdrantClient(":memory:")
    return QdrantClient(path=path)


def ensure_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: qmodels.VectorParams(
                size=DENSE_VECTOR_SIZE,  # text-embedding-3-small dimensionality
                distance=qmodels.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(
                modifier=qmodels.Modifier.IDF,
            ),
        },
    )


def embed_and_upsert(
    client: QdrantClient,
    chunks: list[CodeChunk],
    collection_name: str = COLLECTION_NAME,
    upsert_batch_size: int = 256,
) -> None:
    """
    Compute dense (OpenAI, hosted, internally concurrent — see
    DenseEmbedder) + sparse (FastEmbed BM25, local CPU) embeddings for
    every chunk, then upsert into Qdrant in batches.

    Dense and sparse embedding run on separate threads for the WHOLE chunk
    set at once, not per upsert-batch: dense is network-bound (waiting on
    OpenAI), sparse is CPU-bound (local BM25 term statistics) — genuinely
    independent work, so overlapping them means the local CPU pass costs
    nothing extra on the wall-clock critical path instead of being paid
    serially after every network round trip.

    upsert_batch_size only controls how many points go into each Qdrant
    upsert() call; it no longer limits embedding batch size — that's
    DenseEmbedder's own internal concurrency, see dense_embeddings.py.
    """
    if not chunks:
        logger.warning("No chunks to embed — skipping upsert.")
        return

    dense_embedder = DenseEmbedder()
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    texts = [_embedding_text(c) for c in chunks]

    with ThreadPoolExecutor(max_workers=2) as pool:
        dense_future = pool.submit(dense_embedder.embed, texts)
        sparse_future = pool.submit(lambda: list(sparse_model.embed(texts)))
        dense_vecs = dense_future.result()
        sparse_vecs = sparse_future.result()

    for start in range(0, len(chunks), upsert_batch_size):
        batch_chunks = chunks[start : start + upsert_batch_size]
        batch_dense = dense_vecs[start : start + upsert_batch_size]
        batch_sparse = sparse_vecs[start : start + upsert_batch_size]

        points = []
        for chunk, dense_vec, sparse_vec in zip(batch_chunks, batch_dense, batch_sparse):
            points.append(
                qmodels.PointStruct(
                    id=chunk.chunk_id,
                    vector={
                        DENSE_VECTOR_NAME: dense_vec,
                        SPARSE_VECTOR_NAME: qmodels.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        ),
                    },
                    payload={
                        "file_path": chunk.file_path,
                        "symbol_name": chunk.symbol_name,
                        "symbol_type": chunk.symbol_type,
                        "language": chunk.language,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "content": chunk.content,
                        "calls": chunk.calls,
                        "imports": chunk.imports,
                    },
                )
            )
        client.upsert(collection_name=collection_name, points=points)
        logger.info("Upserted batch %d-%d / %d", start, start + len(batch_chunks), len(chunks))


def _embedding_text(chunk: CodeChunk) -> str:
    """
    Text actually fed to the embedding models. Prepending the file path and
    symbol name improves retrieval for queries like "the auth middleware" or
    "UserService.login" that reference symbol/file identity rather than
    only code semantics.
    """
    header = f"# {chunk.file_path} :: {chunk.symbol_type} {chunk.symbol_name}\n"
    return header + chunk.content


# --------------------------------------------------------------------------- #
# Call graph (for graph.py's expansion node)
# --------------------------------------------------------------------------- #

def _qualified_symbol_id(chunk: CodeChunk) -> str:
    """file_path::symbol_name — unique even when symbol names collide
    across files/classes (e.g. multiple `__init__` or `run` methods)."""
    return f"{chunk.file_path}::{chunk.symbol_name}"


def build_call_graph(chunks: list[CodeChunk]) -> dict[str, dict[str, list[str]]]:
    """
    Build a simple bidirectional call graph keyed by a *qualified* symbol id
    (file_path::symbol_name), not bare symbol_name:
      { "path/to/file.py::foo": {"callers": [...], "callees": [...], "chunk_id": ...} }

    Bare names collide constantly in real repos (every class's `__init__`,
    every module's `main`); keying on bare names would silently overwrite
    all-but-one same-named chunk and misattribute call edges. Call *sites*
    are still bare names (regex can't resolve imports), so a bare name may
    fan out to several qualified definitions — we conservatively link a
    call site to every definition sharing that name.

    This is intentionally lightweight (regex-derived, not a real resolver) —
    good enough for "what calls X" / "what does X call" expansion when
    retrieval alone doesn't surface a directly-connected chunk.
    """
    named_chunks = [c for c in chunks if c.symbol_name != "<anonymous>"]

    # bare name -> list of qualified ids sharing that name (for call-site resolution)
    name_to_qualified_ids: dict[str, list[str]] = {}
    for c in named_chunks:
        name_to_qualified_ids.setdefault(c.symbol_name, []).append(_qualified_symbol_id(c))

    graph: dict[str, dict[str, list[str]]] = {
        _qualified_symbol_id(c): {"callers": [], "callees": [], "chunk_id": c.chunk_id, "file_path": c.file_path}
        for c in named_chunks
    }

    for chunk in named_chunks:
        caller_id = _qualified_symbol_id(chunk)
        for called_name in chunk.calls:
            for callee_id in name_to_qualified_ids.get(called_name, []):
                if callee_id == caller_id:
                    continue
                graph[caller_id]["callees"].append(callee_id)
                graph[callee_id]["callers"].append(caller_id)

    return graph


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def ingest_repository(
    github_url: str,
    qdrant_path: str | None = None,
    collection_name: str = COLLECTION_NAME,
    qdrant_client: QdrantClient | None = None,
) -> tuple[IngestionResult, QdrantClient, dict[str, dict[str, list[str]]]]:
    """
    Full pipeline: clone -> filter -> chunk -> embed -> upsert.
    Returns the ingestion summary, the live Qdrant client (caller keeps it
    open for querying), and the derived call graph.

    Pass `qdrant_client` to reuse an existing client (e.g. a process-wide
    singleton against an on-disk collection — see backend/pipeline_store.py)
    instead of opening a new one. Qdrant's local mode does not support
    multiple concurrent client instances against the same on-disk path, so
    any long-running caller managing its own client MUST pass it in rather
    than let this function open a second one. `qdrant_path` is ignored when
    `qdrant_client` is provided.
    """
    clone_path = shallow_clone(github_url)
    all_chunks: list[CodeChunk] = []
    scanned = 0
    skipped = 0

    try:
        for file_path in discover_source_files(clone_path):
            scanned += 1
            file_chunks = chunk_file(file_path, clone_path)
            if not file_chunks:
                skipped += 1
                continue
            # No chunking strategy has an inherent token cap — an AST node
            # (a single function/class) can be arbitrarily large, and a
            # semantic span is capped in LINES, which says nothing about
            # tokens for minified or very long-lined files. Either can
            # exceed OpenAI's 8192-token embedding input limit and fail the
            # entire upsert batch — see MAX_CHUNK_TOKENS's comment. Applied
            # here rather than inside each chunker so it's a single
            # guarantee covering the AST, semantic, and fixed-window paths
            # alike, rather than three call sites to keep in sync.
            for chunk in file_chunks:
                all_chunks.extend(_split_oversized_chunk(chunk))

        logger.info(
            "Discovered %d chunks across %d files (%d skipped) using '%s' chunking",
            len(all_chunks), scanned, skipped, CHUNK_STRATEGY,
        )

        client = qdrant_client if qdrant_client is not None else get_qdrant_client(qdrant_path)
        ensure_collection(client, collection_name)
        embed_and_upsert(client, all_chunks, collection_name)
        call_graph = build_call_graph(all_chunks)

        result = IngestionResult(
            repo_url=github_url,
            total_files_scanned=scanned,
            total_files_skipped=skipped,
            total_chunks=len(all_chunks),
            collection_name=collection_name,
        )
        return result, client, call_graph
    finally:
        cleanup_clone(clone_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python ingestion.py <github_repo_url>")
        sys.exit(1)

    result, _client, _graph = ingest_repository(sys.argv[1])
    print(f"Ingested {result.total_chunks} chunks from {result.total_files_scanned} files "
          f"({result.total_files_skipped} skipped) into collection '{result.collection_name}'.")
