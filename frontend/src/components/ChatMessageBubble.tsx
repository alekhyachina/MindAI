import type { ChatMessage } from "../lib/api";
import CitationText from "./CitationText";

export default function ChatMessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-lg rounded-tr-sm bg-surface-raised px-4 py-3 text-[15px] text-text">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div
        className={`max-w-[75%] rounded-lg rounded-tl-sm border px-4 py-3 ${
          message.refused ? "border-border bg-surface text-text-muted" : "border-border bg-surface"
        }`}
      >
        {message.refused && (
          <span className="mb-1.5 block font-mono text-[11px] uppercase tracking-wide text-text-muted">
            insufficient context
          </span>
        )}
        <CitationText text={message.content} />
      </div>
    </div>
  );
}
