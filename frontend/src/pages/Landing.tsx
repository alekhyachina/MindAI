import { Link } from "react-router-dom";

export default function Landing() {
  return (
    <div className="relative min-h-svh overflow-hidden bg-ground">
      <div className="hero-glow" />

      <div className="relative z-10 flex min-h-svh flex-col">
        <header className="flex items-center justify-between px-6 py-6 sm:px-10">
          <span className="font-display text-lg tracking-tight text-text">
            mind<span className="text-gradient font-semibold">ai</span>
          </span>
          <nav className="flex items-center gap-6 text-sm text-text-muted">
            <Link to="/login" className="hover:text-text transition-colors">
              Log in
            </Link>
            <Link
              to="/signup"
              className="rounded-full bg-accent px-4 py-2 font-medium text-white shadow-[0_0_24px_-4px_var(--color-accent)] transition-all hover:bg-accent-dim hover:shadow-[0_0_32px_-4px_var(--color-accent)]"
            >
              Get started
            </Link>
          </nav>
        </header>

        <main className="flex flex-1 flex-col items-center justify-center px-6 py-16 text-center">
          <span className="mb-7 inline-flex items-center gap-2 rounded-full border border-border-bright bg-surface px-3.5 py-1.5 text-xs uppercase tracking-[0.14em] text-text-muted backdrop-blur-sm">
            <span className="h-1.5 w-1.5 rounded-full bg-accent-2 shadow-[0_0_8px_1px_var(--color-accent-2)]" />
            Graph-RAG for codebases
          </span>

          <h1 className="font-display text-balance text-4xl font-medium leading-[1.08] text-text sm:text-6xl">
            Ask any repository
            <br />
            <span className="text-gradient">a question it can prove.</span>
          </h1>

          <p className="mt-6 max-w-xl text-balance text-base leading-relaxed text-text-muted sm:text-lg">
            MindAI reads a GitHub repo down to the function, traces callers and
            callees through its structure, and answers with citations you can
            trace straight to the line. No context, no answer — it says so.
          </p>

          <div className="mt-10 flex flex-col items-center gap-3 sm:flex-row">
            <Link
              to="/signup"
              className="rounded-full bg-accent px-7 py-3.5 font-medium text-white shadow-[0_0_32px_-6px_var(--color-accent)] transition-all hover:bg-accent-dim hover:shadow-[0_0_40px_-6px_var(--color-accent)]"
            >
              Start reading a repo
            </Link>
            <Link
              to="/login"
              className="rounded-full border border-border-bright px-7 py-3.5 font-medium text-text transition-colors hover:border-accent hover:text-accent"
            >
              I have an account
            </Link>
          </div>

          <dl className="mt-20 grid w-full max-w-3xl grid-cols-1 gap-px overflow-hidden rounded-2xl border border-border bg-border text-left sm:grid-cols-3">
            <div className="bg-surface-solid px-6 py-7">
              <dt className="font-mono text-xs uppercase tracking-wide text-accent-2">
                path:line
              </dt>
              <dd className="mt-2.5 text-sm leading-relaxed text-text-muted">
                Every claim cites the exact file and line it came from.
              </dd>
            </div>
            <div className="bg-surface-solid px-6 py-7">
              <dt className="font-mono text-xs uppercase tracking-wide text-accent-2">
                AST chunks
              </dt>
              <dd className="mt-2.5 text-sm leading-relaxed text-text-muted">
                Parsed by function and class, not arbitrary token windows.
              </dd>
            </div>
            <div className="bg-surface-solid px-6 py-7">
              <dt className="font-mono text-xs uppercase tracking-wide text-accent-2">
                refuses to guess
              </dt>
              <dd className="mt-2.5 text-sm leading-relaxed text-text-muted">
                Low-confidence retrieval means it says so, not a hallucination.
              </dd>
            </div>
          </dl>
        </main>
      </div>
    </div>
  );
}
