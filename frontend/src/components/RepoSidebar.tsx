import { useState, type FormEvent } from "react";
import type { Repo } from "../lib/api";
import { useAuth } from "../lib/auth";

interface RepoSidebarProps {
  repos: Repo[];
  activeRepoId: string | null;
  onSelectRepo: (repo: Repo) => void;
  onIngest: (githubUrl: string) => Promise<void>;
  ingesting: boolean;
}

export default function RepoSidebar({ repos, activeRepoId, onSelectRepo, onIngest, ingesting }: RepoSidebarProps) {
  const { user, logOut } = useAuth();
  const [githubUrl, setGithubUrl] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!githubUrl.trim()) return;
    setError(null);
    try {
      await onIngest(githubUrl.trim());
      setGithubUrl("");
    } catch {
      setError("Couldn't ingest that repo. Check the URL and try again.");
    }
  }

  return (
    <aside className="flex h-full w-72 shrink-0 flex-col border-r border-border bg-surface">
      <div className="flex items-center justify-between px-4 py-4">
        <span className="font-display text-base text-text">
          mind<span className="text-accent">ai</span>
        </span>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-2 px-4 pb-4">
        <label className="text-xs uppercase tracking-wide text-text-muted">Ingest a repo</label>
        <input
          type="text"
          value={githubUrl}
          onChange={(e) => setGithubUrl(e.target.value)}
          placeholder="github.com/owner/repo"
          className="rounded-md border border-border bg-surface-raised px-3 py-2 text-sm text-text outline-none focus:border-accent"
        />
        {error && <p className="text-xs text-danger">{error}</p>}
        <button
          type="submit"
          disabled={ingesting || !githubUrl.trim()}
          className="rounded-md bg-accent px-3 py-2 text-sm font-medium text-ground hover:bg-accent-dim transition-colors disabled:opacity-50"
        >
          {ingesting ? "Ingesting…" : "Ingest"}
        </button>
      </form>

      <nav className="flex-1 overflow-y-auto px-2">
        <span className="mb-1 block px-2 text-xs uppercase tracking-wide text-text-muted">Repositories</span>
        {repos.length === 0 && (
          <p className="px-2 py-3 text-sm text-text-muted">No repos yet — ingest one above.</p>
        )}
        <ul className="flex flex-col gap-0.5">
          {repos.map((repo) => (
            <li key={repo.id}>
              <button
                onClick={() => onSelectRepo(repo)}
                className={`w-full truncate rounded-md px-2 py-2 text-left text-sm transition-colors ${
                  activeRepoId === repo.id
                    ? "bg-accent-wash text-accent"
                    : "text-text-muted hover:bg-surface-raised hover:text-text"
                }`}
                title={repo.github_url}
              >
                {repo.display_name}
              </button>
            </li>
          ))}
        </ul>
      </nav>

      <div className="flex items-center justify-between border-t border-border px-4 py-3">
        <span className="truncate text-xs text-text-muted">{user?.email}</span>
        <button onClick={logOut} className="shrink-0 text-xs text-text-muted hover:text-text">
          Log out
        </button>
      </div>
    </aside>
  );
}
