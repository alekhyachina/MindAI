import { useEffect, useRef, useState } from "react";
import { api, streamChat, type ChatMessage, type Conversation, type Repo } from "../lib/api";
import RepoSidebar from "../components/RepoSidebar";
import ChatMessageBubble from "../components/ChatMessageBubble";

export default function Chat() {
  const [repos, setRepos] = useState<Repo[]>([]);
  const [activeRepo, setActiveRepo] = useState<Repo | null>(null);
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [ingesting, setIngesting] = useState(false);
  const [asking, setAsking] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.listRepos().then((list) => {
      setRepos(list);
      if (list.length > 0) void selectRepo(list[0]);
      // Resume polling for any repo that was still ingesting when this
      // page loaded (e.g. the user refreshed mid-ingestion).
      for (const repo of list) {
        if (repo.status === "pending" || repo.status === "ingesting") {
          pollRepoStatus(repo.id);
        }
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function selectRepo(repo: Repo) {
    setActiveRepo(repo);
    if (repo.status !== "ready") {
      // Not ready yet — just show it as selected; the ingestion-progress
      // view in the main pane handles the rest. No conversation to create
      // against a repo that isn't queryable yet.
      setConversation(null);
      setMessages([]);
      return;
    }
    const conv = await api.createConversation(repo.id);
    setConversation(conv);
    setMessages(await api.getMessages(conv.id));
  }

  async function handleIngest(githubUrl: string) {
    setIngesting(true);
    try {
      // Ingestion runs as a background job on the server (can take several
      // minutes on CPU) — this call returns immediately with a "pending"
      // repo row rather than blocking the request for the full duration.
      const repo = await api.ingestRepo(githubUrl);
      setRepos((prev) => [repo, ...prev]);
      setActiveRepo(repo);
      pollRepoStatus(repo.id);
    } finally {
      setIngesting(false);
    }
  }

  function pollRepoStatus(repoId: string) {
    const intervalId = window.setInterval(async () => {
      const list = await api.listRepos();
      setRepos(list);
      const updated = list.find((r) => r.id === repoId);
      if (!updated) return;

      setActiveRepo((current) => (current?.id === repoId ? updated : current));

      if (updated.status === "ready" || updated.status === "failed") {
        window.clearInterval(intervalId);
        if (updated.status === "ready") {
          void selectRepo(updated);
        }
      }
    }, 3000);
  }

  async function handleAsk() {
    if (!question.trim() || !conversation || asking) return;
    const q = question.trim();
    setQuestion("");
    setMessages((prev) => [
      ...prev,
      { id: `local-${Date.now()}`, role: "user", content: q, refused: false, created_at: new Date().toISOString() },
    ]);
    setAsking(true);
    setStage("retrieving");
    setStreamError(null);

    try {
      await streamChat(conversation.id, q, (event) => {
        if (event.type === "status") setStage(event.stage ?? null);
        if (event.type === "error") {
          setStreamError(event.message ?? "Something went wrong. Please try again.");
        }
        if (event.type === "answer") {
          setStage(null);
          setMessages((prev) => [
            ...prev,
            {
              id: `local-${Date.now()}-a`,
              role: "assistant",
              content: event.content ?? "",
              refused: Boolean(event.refused),
              created_at: new Date().toISOString(),
            },
          ]);
        }
      });
    } catch {
      setStreamError("Lost connection while answering. Please try again.");
    } finally {
      setAsking(false);
      setStage(null);
    }
  }

  return (
    <div className="flex h-svh bg-ground">
      <RepoSidebar
        repos={repos}
        activeRepoId={activeRepo?.id ?? null}
        onSelectRepo={selectRepo}
        onIngest={handleIngest}
        ingesting={ingesting}
      />

      <main className="flex flex-1 flex-col">
        {!activeRepo ? (
          <div className="flex flex-1 items-center justify-center px-6 text-center">
            <div>
              <h2 className="font-display text-xl text-text">No repository selected</h2>
              <p className="mt-2 text-sm text-text-muted">Ingest a GitHub repo from the sidebar to start asking questions.</p>
            </div>
          </div>
        ) : activeRepo.status === "pending" || activeRepo.status === "ingesting" ? (
          <div className="flex flex-1 items-center justify-center px-6 text-center">
            <div>
              <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-2 border-border border-t-accent" />
              <h2 className="font-display text-xl text-text">Indexing {activeRepo.display_name}</h2>
              <p className="mt-2 max-w-sm text-sm text-text-muted">
                Cloning, parsing, and embedding the repository — this page updates automatically
                when it's ready.
              </p>
            </div>
          </div>
        ) : activeRepo.status === "failed" ? (
          <div className="flex flex-1 items-center justify-center px-6 text-center">
            <div className="max-w-md">
              <h2 className="font-display text-xl text-text">Ingestion failed</h2>
              <p className="mt-2 text-sm text-danger">
                {activeRepo.error_message ?? "Something went wrong while indexing this repository."}
              </p>
            </div>
          </div>
        ) : (
          <>
            <header className="border-b border-border px-6 py-4">
              <h1 className="font-display text-lg text-text">{activeRepo.display_name}</h1>
              <p className="mt-0.5 font-mono text-xs text-text-muted">
                {activeRepo.total_chunks.toLocaleString()} chunks indexed
              </p>
            </header>

            <div className="flex-1 overflow-y-auto px-6 py-6">
              <div className="mx-auto flex max-w-2xl flex-col gap-4">
                {messages.length === 0 && (
                  <p className="text-center text-sm text-text-muted">
                    Ask anything about {activeRepo.display_name} — every answer cites its source.
                  </p>
                )}
                {messages.map((m) => (
                  <ChatMessageBubble key={m.id} message={m} />
                ))}
                {stage && (
                  <div className="flex justify-start">
                    <div className="rounded-lg border border-border bg-surface px-4 py-3 font-mono text-xs text-text-muted">
                      {stage}…
                    </div>
                  </div>
                )}
                {streamError && (
                  <div className="flex justify-start">
                    <div className="rounded-lg border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger">
                      {streamError}
                    </div>
                  </div>
                )}
                <div ref={bottomRef} />
              </div>
            </div>

            <div className="border-t border-border px-6 py-4">
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  void handleAsk();
                }}
                className="mx-auto flex max-w-2xl gap-2"
              >
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="How does authentication work in this repo?"
                  disabled={asking}
                  className="flex-1 rounded-md border border-border bg-surface px-4 py-3 text-sm text-text outline-none focus:border-accent disabled:opacity-60"
                />
                <button
                  type="submit"
                  disabled={asking || !question.trim()}
                  className="rounded-md bg-accent px-5 py-3 text-sm font-medium text-ground hover:bg-accent-dim transition-colors disabled:opacity-50"
                >
                  Ask
                </button>
              </form>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
