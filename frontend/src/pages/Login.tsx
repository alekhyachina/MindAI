import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../lib/auth";
import { ApiError } from "../lib/api";

export default function Login() {
  const { logIn } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await logIn(email, password);
      navigate("/chat");
    } catch (err) {
      setError(err instanceof ApiError ? "Incorrect email or password." : "Something went wrong. Try again.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex min-h-svh items-center justify-center bg-ground px-6">
      <div className="w-full max-w-sm">
        <Link to="/" className="font-display text-lg text-text">
          mind<span className="text-accent">ai</span>
        </Link>

        <h1 className="mt-8 font-display text-2xl font-medium text-text">Welcome back</h1>
        <p className="mt-2 text-sm text-text-muted">Log in to pick up where you left off.</p>

        <form onSubmit={handleSubmit} className="mt-8 flex flex-col gap-4">
          <label className="flex flex-col gap-1.5 text-left">
            <span className="text-xs uppercase tracking-wide text-text-muted">Email</span>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="rounded-md border border-border bg-surface px-3 py-2.5 text-text outline-none focus:border-accent"
              placeholder="you@example.com"
            />
          </label>

          <label className="flex flex-col gap-1.5 text-left">
            <span className="text-xs uppercase tracking-wide text-text-muted">Password</span>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="rounded-md border border-border bg-surface px-3 py-2.5 text-text outline-none focus:border-accent"
              placeholder="••••••••"
            />
          </label>

          {error && (
            <p className="rounded-md border border-danger/40 bg-danger/10 px-3 py-2 text-sm text-danger">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={submitting}
            className="mt-2 rounded-md bg-accent px-4 py-2.5 font-medium text-ground hover:bg-accent-dim transition-colors disabled:opacity-60"
          >
            {submitting ? "Logging in…" : "Log in"}
          </button>
        </form>

        <p className="mt-6 text-center text-sm text-text-muted">
          New to MindAI?{" "}
          <Link to="/signup" className="text-accent hover:text-accent-dim">
            Create an account
          </Link>
        </p>
      </div>
    </div>
  );
}
