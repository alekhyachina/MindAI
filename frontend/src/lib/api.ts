// API client for the MindAI FastAPI backend. All requests go through the
// /api prefix, proxied to the backend by Vite's dev server (see
// vite.config.ts) so no CORS/base-URL juggling is needed in dev.

const TOKEN_STORAGE_KEY = "mindai_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_STORAGE_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_STORAGE_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_STORAGE_KEY);
}

class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers = new Headers(options.headers);
  headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const response = await fetch(`/api${path}`, { ...options, headers });

  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new ApiError(response.status, body.detail ?? "Request failed");
  }

  if (response.status === 204) return undefined as T;
  return response.json();
}

export interface User {
  id: string;
  email: string;
  created_at: string;
}

export type RepoStatus = "pending" | "ingesting" | "ready" | "failed";

export interface Repo {
  id: string;
  github_url: string;
  display_name: string;
  total_chunks: number;
  status: RepoStatus;
  error_message: string | null;
  created_at: string;
}

export interface Conversation {
  id: string;
  repo_id: string;
  title: string;
  created_at: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  refused: boolean;
  created_at: string;
}

export const api = {
  signup: (email: string, password: string) =>
    request<{ access_token: string }>("/auth/signup", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),

  login: (email: string, password: string) =>
    request<{ access_token: string }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),

  me: () => request<User>("/auth/me"),

  listRepos: () => request<Repo[]>("/repos"),

  ingestRepo: (githubUrl: string) =>
    request<Repo>("/repos", {
      method: "POST",
      body: JSON.stringify({ github_url: githubUrl }),
    }),

  listConversations: (repoId: string) =>
    request<Conversation[]>(`/conversations?repo_id=${encodeURIComponent(repoId)}`),

  createConversation: (repoId: string) =>
    request<Conversation>("/conversations", {
      method: "POST",
      body: JSON.stringify({ repo_id: repoId }),
    }),

  getMessages: (conversationId: string) =>
    request<ChatMessage[]>(`/conversations/${conversationId}/messages`),
};

export { ApiError };

// Chat uses SSE streaming, which needs raw fetch + ReadableStream handling
// rather than the JSON request() helper above.
export async function streamChat(
  conversationId: string,
  question: string,
  onEvent: (event: { type: string; content?: string; refused?: boolean; stage?: string; message?: string }) => void,
): Promise<void> {
  const token = getToken();
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ conversation_id: conversationId, question }),
  });

  if (!response.ok || !response.body) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new ApiError(response.status, body.detail ?? "Chat request failed");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const json = trimmed.slice("data:".length).trim();
      try {
        onEvent(JSON.parse(json));
      } catch {
        // ignore malformed frame
      }
    }
  }
}
