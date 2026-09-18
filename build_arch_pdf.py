"""
build_arch_pdf.py — generates the MindAI architecture document as a PDF.

Every fact here was read out of the codebase (graph.py, ingestion.py,
backend/*.py, frontend/src/*) rather than recalled, so the constants and
route lists below are the real ones.
"""

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

OUT = r"C:\Users\Rohit Das\Desktop\New folder\saymaindai\saymaindai\MindAI-Architecture.pdf"

INK = colors.HexColor("#1c1b18")
INK2 = colors.HexColor("#55524b")
INK3 = colors.HexColor("#8a867e")
ACCENT = colors.HexColor("#2f5d50")
ACCENT_SOFT = colors.HexColor("#e3ece8")
RULE = colors.HexColor("#d8d4cb")
SURF = colors.HexColor("#f6f5f2")
WARN = colors.HexColor("#9a5b2b")
WARN_SOFT = colors.HexColor("#f6ebdf")
BAD = colors.HexColor("#8d3b3b")

styles = getSampleStyleSheet()

H1 = ParagraphStyle("H1", parent=styles["Title"], fontName="Times-Bold",
                    fontSize=27, leading=31, textColor=INK, alignment=TA_LEFT,
                    spaceAfter=5)
SUBTITLE = ParagraphStyle("SUBTITLE", fontName="Helvetica", fontSize=11.5,
                          leading=16, textColor=INK2, spaceAfter=3)
EYEBROW = ParagraphStyle("EYEBROW", fontName="Courier-Bold", fontSize=8,
                         leading=11, textColor=INK3, spaceAfter=7)
H2 = ParagraphStyle("H2", fontName="Times-Bold", fontSize=16, leading=20,
                    textColor=INK, spaceBefore=17, spaceAfter=5)
H3 = ParagraphStyle("H3", fontName="Helvetica-Bold", fontSize=10.5, leading=14,
                    textColor=INK, spaceBefore=11, spaceAfter=3)
BODY = ParagraphStyle("BODY", fontName="Helvetica", fontSize=9.6, leading=14.2,
                      textColor=INK, spaceAfter=7)
NOTE = ParagraphStyle("NOTE", fontName="Helvetica-Oblique", fontSize=9,
                      leading=13, textColor=INK2, spaceAfter=7)
CELL = ParagraphStyle("CELL", fontName="Helvetica", fontSize=8.6, leading=12,
                      textColor=INK)
CELLM = ParagraphStyle("CELLM", fontName="Courier", fontSize=8.3, leading=12,
                       textColor=INK)
CELLH = ParagraphStyle("CELLH", fontName="Helvetica-Bold", fontSize=8,
                       leading=11, textColor=INK3)
CODE = ParagraphStyle("CODE", fontName="Courier", fontSize=8.5, leading=12.5,
                      textColor=INK, backColor=SURF, borderPadding=7,
                      leftIndent=3, spaceAfter=8)
CALLOUT = ParagraphStyle("CALLOUT", fontName="Helvetica", fontSize=9.2,
                         leading=13.4, textColor=INK, backColor=WARN_SOFT,
                         borderPadding=9, spaceAfter=9, leftIndent=2,
                         rightIndent=2)


def para(t, s=BODY):
    return Paragraph(t, s)


def table(rows, widths, header=True, zebra=True):
    data = []
    for i, row in enumerate(rows):
        st = CELLH if (header and i == 0) else None
        cells = []
        for c in row:
            if st:
                cells.append(Paragraph(str(c), CELLH))
            elif c.startswith("`") and c.endswith("`"):
                cells.append(Paragraph(c.strip("`"), CELLM))
            else:
                cells.append(Paragraph(str(c), CELL))
        data.append(cells)

    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
    ]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), SURF),
                 ("LINEBELOW", (0, 0), (-1, 0), 0.7, RULE)]
    if zebra:
        start = 1 if header else 0
        for i in range(start, len(data)):
            if (i - start) % 2 == 1:
                cmds.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fbfaf8")))
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle(cmds))
    return t


def flowbox(label, sub, fill=colors.white, border=RULE, w=150 * mm):
    """A single box in an ASCII-free flow diagram, drawn as a 1-cell table."""
    inner = [[Paragraph(f"<font face='Courier-Bold' size='9'>{label}</font>"
                        f"<br/><font face='Helvetica' size='8.2' color='#55524b'>{sub}</font>",
                        CELL)]]
    t = Table(inner, colWidths=[w])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("BOX", (0, 0), (-1, -1), 0.9, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def arrow(label=""):
    txt = f"&#9662;&nbsp;&nbsp;<font face='Courier' size='7.5' color='#8a867e'>{label}</font>" if label else "&#9662;"
    return Paragraph(txt, ParagraphStyle("ARR", fontName="Helvetica", fontSize=11,
                                         leading=13, textColor=INK3, leftIndent=16,
                                         spaceBefore=1, spaceAfter=1))


# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #

def on_page(canv, doc):
    canv.saveState()
    canv.setFont("Helvetica", 7.5)
    canv.setFillColor(INK3)
    canv.drawString(20 * mm, 12 * mm, "MindAI — System Architecture")
    canv.drawRightString(A4[0] - 20 * mm, 12 * mm, f"{canv.getPageNumber()}")
    canv.setStrokeColor(RULE)
    canv.setLineWidth(0.4)
    canv.line(20 * mm, 15.5 * mm, A4[0] - 20 * mm, 15.5 * mm)
    canv.restoreState()


def build():
    doc = BaseDocTemplate(OUT, pagesize=A4,
                          leftMargin=20 * mm, rightMargin=20 * mm,
                          topMargin=18 * mm, bottomMargin=20 * mm,
                          title="MindAI — System Architecture",
                          author="Rohit Das", subject="Graph-RAG architecture")
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=on_page)])

    S = []
    W = doc.width

    # ---------------- Cover ----------------
    S.append(Spacer(1, 30 * mm))
    S.append(para("GRAPH-RAG CODE ASSISTANT", EYEBROW))
    S.append(para("MindAI", H1))
    S.append(para("System Architecture", ParagraphStyle(
        "S2", parent=SUBTITLE, fontName="Times-Italic", fontSize=15,
        textColor=ACCENT, spaceAfter=14)))
    S.append(para(
        "A retrieval-augmented question-answering system for GitHub repositories. "
        "It answers in natural language with line-level citations back to the source, "
        "and refuses rather than guess when the retrieved context cannot support an answer.",
        SUBTITLE))
    S.append(Spacer(1, 9 * mm))
    S.append(table([
        ["Layer", "Technology"],
        ["Orchestration", "`LangGraph 1.2.11 — StateGraph, 5 nodes`"],
        ["Vector store", "`Qdrant — named dense + sparse vectors`"],
        ["Dense embeddings", "`OpenAI text-embedding-3-small (1536-d)`"],
        ["Sparse embeddings", "`FastEmbed Qdrant/bm25`"],
        ["Reranking", "`Cohere rerank (cross-encoder fallback)`"],
        ["Generation", "`OpenAI gpt-4o-mini`"],
        ["Parsing", "`tree-sitter — AST-aware chunking`"],
        ["API", "`FastAPI + SQLAlchemy + SQLite`"],
        ["Client", "`React + Vite + Tailwind`"],
    ], [42 * mm, W - 42 * mm]))
    S.append(Spacer(1, 7 * mm))
    S.append(para(
        "No LangChain. LangGraph supplies the state machine; OpenAI, Cohere and Qdrant "
        "are called through their own SDKs.", NOTE))

    S.append(PageBreak())

    # ---------------- 1. Overview ----------------
    S.append(para("1 · System overview", H2))
    S.append(para(
        "MindAI has two halves that share nothing but two on-disk artefacts. "
        "<b>Ingestion</b> runs once per repository and produces a Qdrant collection plus a call graph. "
        "<b>Query</b> runs per question and reads them. The web layer around both is a thin wrapper: "
        "it handles auth, persistence and HTTP, and changes nothing about how retrieval or generation work.", BODY))

    S.append(para("The two halves", H3))
    S.append(flowbox("INGEST  (once per repo)",
                     "clone &#8594; chunk &#8594; embed &#8594; upsert &#8594; call graph", SURF, RULE, W))
    S.append(arrow("writes"))
    S.append(flowbox("Qdrant collection  +  call graph JSON",
                     "the entire interface between the two halves", ACCENT_SOFT, ACCENT, W))
    S.append(arrow("read by"))
    S.append(flowbox("QUERY  (per question)",
                     "rewrite &#8594; retrieve &#8594; rerank &#8594; [expand] &#8594; generate &#8594; validate", SURF, RULE, W))

    S.append(para("Repository layout", H3))
    S.append(table([
        ["Path", "Role"],
        ["`ingestion.py`", "Clone, parse, chunk, embed, upsert. Builds the call graph."],
        ["`graph.py`", "The LangGraph pipeline — all five nodes and the citation gate."],
        ["`dense_embeddings.py`", "Dense embedder wrapper (OpenAI, concurrent batching)."],
        ["`reranker.py`", "Cohere / cross-encoder reranking."],
        ["`semantic_chunking.py`", "Alternative embedding-derived chunk boundaries."],
        ["`eval.py`", "DeepEval harness — faithfulness, relevancy, recall."],
        ["`backend/`", "FastAPI: auth, DB, HTTP, SSE streaming."],
        ["`frontend/`", "React chat client."],
    ], [38 * mm, W - 38 * mm]))

    # ---------------- 2. Ingestion ----------------
    S.append(PageBreak())
    S.append(para("2 · Ingestion pipeline", H2))
    S.append(para(
        "Triggered by <font face='Courier' size='8.5'>POST /repos</font>, which returns "
        "<font face='Courier' size='8.5'>202 Accepted</font> immediately and runs the work as a "
        "FastAPI background task — a full ingest takes minutes and would otherwise hold the "
        "HTTP connection open.", BODY))

    steps = [
        ("1 · clone", "Shallow git clone into a temp directory. Only the working tree is needed, not history."),
        ("2 · chunk", "tree-sitter parses each source file into an AST and cuts at <b>function and class boundaries</b>, so a chunk is a whole semantic unit with exact start/end lines. Files tree-sitter cannot parse (notebooks, markdown, config) fall back to 100-line windows with 20 lines of overlap."),
        ("3 · embed", "Each chunk is embedded twice — dense (<font face='Courier' size='8.3'>text-embedding-3-small</font>, 1536-d) for meaning and sparse BM25 for exact tokens. The two run concurrently in a thread pool."),
        ("4 · upsert", "Both vectors are stored as <b>named vectors on one Qdrant point</b>, with the chunk's path, line range, symbol name and content as payload. Each repo gets its own collection, <font face='Courier' size='8.3'>repo_&lt;uuid&gt;</font>."),
        ("5 · call graph", "Caller/callee edges are extracted from the AST and saved as JSON beside the collection — this is what makes the system Graph-RAG rather than plain RAG."),
    ]
    for name, desc in steps:
        S.append(KeepTogether([
            para(f"<font face='Courier-Bold' size='9.5' color='#2f5d50'>{name}</font>", H3),
            para(desc, BODY),
        ]))

    S.append(para("Why AST chunking matters", H3))
    S.append(para(
        "Fixed-size token windows cut functions in half, which produces chunks that cite line ranges "
        "spanning unrelated code and embeddings that average two unrelated meanings. Cutting at symbol "
        "boundaries keeps each chunk semantically whole and gives every citation a line range that "
        "genuinely delimits the thing being described.", BODY))

    S.append(para("What ends up in Qdrant", H3))
    S.append(table([
        ["Payload field", "Purpose"],
        ["`file_path`", "Repo-relative path — becomes the citation and the grounding check."],
        ["`start_line` / `end_line`", "Exact line range of the symbol."],
        ["`symbol_name`", "Function or class name; also reranked against."],
        ["`symbol_type`", "`function_definition`, `class_definition`, or `text_window`."],
        ["`content`", "The source text itself, packed into the model's context."],
        ["`calls`", "Callees found in this chunk — the call-graph edge list."],
        ["`language`", "Source language, from the file extension."],
    ], [44 * mm, W - 44 * mm]))

    # ---------------- 3. Query pipeline ----------------
    S.append(PageBreak())
    S.append(para("3 · Query pipeline — the LangGraph", H2))
    S.append(para(
        "Five nodes registered in <font face='Courier' size='8.5'>build_graph()</font>. State is a "
        "<font face='Courier' size='8.5'>TypedDict</font> threaded through every node; each node returns "
        "only the keys it changes and LangGraph merges them into the running state.", BODY))

    S.append(flowbox("1 · query_rewrite", "LLM expands the question into retrieval terms", colors.white, ACCENT, W))
    S.append(arrow("search_query"))
    S.append(flowbox("2 · hybrid_retrieve", "dense (100) + sparse BM25 (100), fused by Reciprocal Rank Fusion", colors.white, ACCENT, W))
    S.append(arrow("100 fused candidates"))
    S.append(flowbox("3 · rerank", "cross-encoder rescores all 100, keeps top 20", colors.white, ACCENT, W))
    S.append(arrow("conditional edge &#8212; needs_graph_expansion()"))
    S.append(flowbox("4 · graph_expand   [only if trace-shaped AND confidence &lt; 0.5]",
                     "adds callers / callees from the call graph", WARN_SOFT, WARN, W))
    S.append(arrow())
    S.append(flowbox("5 · generate", "packs 8000 tokens of context, produces a cited answer", colors.white, ACCENT, W))
    S.append(arrow("then, inside generate:"))
    S.append(flowbox("_validate_citations()", "every cited path must appear in the retrieved context &#8212; or the answer is discarded", ACCENT_SOFT, ACCENT, W))

    S.append(para("Node by node", H3))
    S.append(table([
        ["Node", "What it does"],
        ["`query_rewrite`", "A user question rarely contains the identifiers that appear in code. An LLM rewrites it into a retrieval query dense with plausible symbol and library names, which is what dense search actually matches on."],
        ["`hybrid_retrieve`", "Dense search finds semantic matches, sparse BM25 finds exact identifiers. Results are fused by <b>Reciprocal Rank Fusion</b> — fusing on rank position, not raw score, because cosine similarity and BM25 scores are not on comparable scales."],
        ["`rerank`", "A cross-encoder reads query and chunk together, which is far more accurate than comparing independent embeddings. Two guards shape the output: chunks whose filename or symbol the question names are boosted, and no single file may take more than a quarter of the slots."],
        ["`graph_expand`", "Conditional. Pulls callers and callees for the retrieved symbols — structural neighbours that vector similarity misses, needed for \"who calls this\" questions."],
        ["`generate`", "Packs chunks highest-ranked first into the token budget and answers under a strict contract: every claim carries a <font face='Courier' size='8'>path:line</font> citation."],
    ], [30 * mm, W - 30 * mm]))

    S.append(para("The conditional edge", H3))
    S.append(para(
        "The graph's only branch. It routes to <font face='Courier' size='8.5'>graph_expand</font> only when "
        "<b>both</b> conditions hold — the question is trace-shaped (matching patterns like "
        "<i>calls, callers, who uses, depends on, trace</i>) <b>and</b> reranker confidence is below 0.5. "
        "A confident answer skips expansion even for a trace question, which keeps the common single-hop "
        "case at four nodes instead of five.", BODY))

    # ---------------- 4. Anti-hallucination ----------------
    S.append(PageBreak())
    S.append(para("4 · The anti-hallucination design", H2))
    S.append(para(
        "The system's central promise is that it does not invent. That guarantee rests on one design "
        "decision, and it is worth stating precisely because the obvious approach was tried first and "
        "abandoned.", BODY))

    S.append(para("What was tried and discarded", H3))
    S.append(table([
        ["Ver.", "Gate", "Why it failed"],
        ["v1", "Refuse if top reranker score &lt; 0.15", "Broad questions (\"what is this repo\") have no single best-matching chunk, so confidence is near zero even when retrieval worked."],
        ["v2", "Relax the bar for keyword-matched broad questions", "Real phrasing never matches a fixed keyword list. Phrasing was never the right signal."],
        ["v3", "Require dense/sparse agreement instead", "Measured across real questions, confidence landed at 0.02, 0.003, 0.0003 on questions the model answered <i>correctly</i>. The reranker was trained on web search, not code, and scores most code near zero regardless of true relevance."],
        ["v4", "<b>No score gate at all</b>", "<b>Current.</b> Generation proceeds whenever retrieval returned anything; grounding is confirmed afterwards."],
    ], [15 * mm, 47 * mm, W - 62 * mm]))

    S.append(para("The gate that replaced them", H3))
    S.append(para(
        "After generation, <font face='Courier' size='8.5'>_validate_citations()</font> extracts every "
        "<font face='Courier' size='8.5'>path:line</font> citation from the answer and checks each path "
        "against the paths actually placed in the model's context. Any citation to a path that was not "
        "retrieved means the whole answer is discarded and replaced with the refusal text.", BODY))
    S.append(para(
        "<b>Why this is stronger.</b> A score threshold tries to predict whether the model <i>could</i> "
        "ground its answer. Checking citations afterwards confirms whether it <i>did</i> — a direct "
        "check rather than a proxy, and immune to how any particular reranker scores any particular "
        "kind of content.", CALLOUT))

    S.append(para("A failure mode this introduced", H3))
    S.append(para(
        "Because the check is strict, a flaw in it silently destroys correct answers. The citation regex "
        "excluded spaces from file paths, so a correct, properly-cited answer referencing "
        "<font face='Courier' size='8.3'>notebooks/Invoice Flagging.ipynb</font> captured only the fragment "
        "after the space, failed the membership test, and was replaced with \"I don't have enough "
        "information\". Every question whose answer lived in a space-named file failed this way. "
        "Validation now matches against the retrieved paths directly.", BODY))

    # ---------------- 5. Backend ----------------
    S.append(PageBreak())
    S.append(para("5 · Web layer", H2))
    S.append(para("API surface", H3))
    S.append(table([
        ["Method &amp; path", "Purpose"],
        ["`POST /auth/signup`", "Create an account; returns a JWT. → 201"],
        ["`POST /auth/login`", "Exchange credentials for a JWT."],
        ["`GET /auth/me`", "Current user from the bearer token."],
        ["`POST /repos`", "Start ingestion as a background task. → 202"],
        ["`GET /repos`", "The caller's repos, newest first."],
        ["`POST /conversations`", "Open a conversation against a ready repo. → 201"],
        ["`GET /conversations/{id}/messages`", "Message history."],
        ["`POST /chat`", "Ask a question; answer streamed over SSE."],
    ], [56 * mm, W - 56 * mm]))

    S.append(para("Data model", H3))
    S.append(para(
        "SQLAlchemy over SQLite (<font face='Courier' size='8.5'>DATABASE_URL</font> swaps in Postgres). "
        "Four tables, cascade-deleted from the owner down:", BODY))
    S.append(table([
        ["Table", "Key columns", "Notes"],
        ["`users`", "`id, email, hashed_password`", "bcrypt hashes; email unique and indexed."],
        ["`repos`", "`github_url, qdrant_collection_name, status, total_chunks`", "`status` is pending → ingesting → ready | failed. Collection name is unique per repo."],
        ["`conversations`", "`owner_id, repo_id, title`", "Scoped to one repo."],
        ["`messages`", "`conversation_id, role, content, refused`", "`refused` records whether the gate rejected the answer."],
    ], [26 * mm, 52 * mm, W - 78 * mm]))

    S.append(para("Authentication", H3))
    S.append(para(
        "Self-hosted: bcrypt for password storage, HS256 JWTs valid for 7 days, signed with "
        "<font face='Courier' size='8.5'>MINDAI_JWT_SECRET</font>. Every repo and conversation route "
        "checks ownership against the token subject, so one user cannot read another's repos.", BODY))

    S.append(para("Concurrency", H3))
    S.append(para(
        "Qdrant in local mode holds an <b>exclusive file lock</b> on its storage directory — two clients "
        "in one process is an immediate error. <font face='Courier' size='8.5'>pipeline_store.py</font> "
        "therefore keeps a single process-wide client behind a lock. Setting "
        "<font face='Courier' size='8.5'>QDRANT_URL</font> switches to server mode, where the client is "
        "stateless HTTP and the constraint disappears.", BODY))

    S.append(para("Streaming", H3))
    S.append(para(
        "<font face='Courier' size='8.5'>POST /chat</font> streams over SSE, but what is streamed is "
        "<i>stage framing</i> — retrieving, reranking, generating — not tokens. LangGraph returns the "
        "final state only once <font face='Courier' size='8.5'>generate</font> completes. True token "
        "streaming would require <font face='Courier' size='8.5'>stream=True</font> inside the "
        "generation node.", BODY))

    # ---------------- 6. Frontend + config ----------------
    S.append(PageBreak())
    S.append(para("6 · Client", H2))
    S.append(para(
        "React 18 with Vite and Tailwind. Vite proxies <font face='Courier' size='8.5'>/api</font> to the "
        "backend in development, so the browser sees one origin.", BODY))
    S.append(table([
        ["File", "Role"],
        ["`pages/Chat.tsx`", "The main screen: repo selection, message list, ingestion progress, SSE handling."],
        ["`pages/Login.tsx` / `SignUp.tsx`", "Credential forms; store the JWT."],
        ["`components/RepoSidebar.tsx`", "Repo list and the ingest form."],
        ["`components/CitationText.tsx`", "Parses `path:line` out of answers and renders them as source references."],
        ["`components/RequireAuth.tsx`", "Route guard — redirects when no valid token."],
        ["`lib/api.ts`", "Typed fetch wrapper; holds the bearer token in localStorage."],
    ], [48 * mm, W - 48 * mm]))

    S.append(para("7 · Configuration", H2))
    S.append(table([
        ["Variable", "Effect"],
        ["`OPENAI_API_KEY`", "Required — embeddings, rewriting, generation."],
        ["`MINDAI_JWT_SECRET`", "Required — JWT signing key."],
        ["`QDRANT_URL` / `QDRANT_API_KEY`", "If set, use Qdrant server/cloud. If unset, local on-disk mode."],
        ["`MINDAI_QDRANT_PATH`", "Local storage directory (default `./qdrant_storage`)."],
        ["`DATABASE_URL`", "SQLAlchemy URL (default SQLite)."],
        ["`MINDAI_GENERATION_MODEL`", "Default `gpt-4o-mini`."],
        ["`MINDAI_CHUNK_STRATEGY`", "`ast` or `semantic`. Switching requires re-ingesting."],
        ["`MINDAI_COHERE_API_KEY`", "Enables Cohere reranking instead of the local cross-encoder."],
    ], [50 * mm, W - 50 * mm]))

    S.append(para("Tuning constants", H3))
    S.append(para("These are the numbers to reach for when answers come back wrong.", BODY))
    S.append(table([
        ["Constant", "Value", "Governs"],
        ["`FUSED_CANDIDATE_COUNT`", "`100`", "Candidates pulled from <i>each</i> of dense and sparse before fusion. A wider net gives a buried chunk more chances to surface."],
        ["`RERANK_TOP_N`", "`20`", "Chunks kept after reranking; everything else never reaches the model."],
        ["`CONTEXT_TOKEN_BUDGET`", "`8000`", "Hard cap on context tokens, packed highest-ranked first."],
        ["per-file cap", "`5`", "`RERANK_TOP_N / 4` — stops one heavily-chunked file filling the whole context."],
        ["expansion threshold", "`0.5`", "Confidence below which a trace question also triggers graph expansion."],
        ["`FALLBACK_CHUNK_LINES`", "`100`", "Window size for files tree-sitter cannot parse (overlap 20)."],
    ], [42 * mm, 16 * mm, W - 58 * mm]))

    S.append(para("8 · Evaluation", H2))
    S.append(para(
        "<font face='Courier' size='8.5'>eval.py</font> runs a DeepEval harness with three LLM-judged "
        "metrics: <b>Faithfulness</b> (threshold 0.75) for claims unsupported by context, "
        "<b>Answer Relevancy</b> (0.7), and <b>Contextual Recall</b> (0.7) for whether retrieval surfaced "
        "what the answer needed. The bundled gold dataset is small and mock, so it runs standalone; "
        "pointing it at a live ingested repo evaluates the real pipeline end to end.", BODY))

    S.append(Spacer(1, 6 * mm))
    S.append(para(
        "Compiled from the source: <font face='Courier' size='8'>graph.py</font>, "
        "<font face='Courier' size='8'>ingestion.py</font>, <font face='Courier' size='8'>backend/</font> "
        "and <font face='Courier' size='8'>frontend/src/</font>.", NOTE))

    doc.build(S)
    print("written:", OUT)


if __name__ == "__main__":
    build()
