// Renders assistant answer text, turning every `path:line` or
// `path:line-line` citation into a distinct monospace chip. Mirrors the
// exact citation shape graph.py's CITATION_PATTERN enforces, so what's
// visually flagged as "a citation" here is exactly what the backbone
// validated before returning the answer.
const CITATION_PATTERN = /([\w./\\-]+\.\w+):(\d+)(?:-(\d+))?/g;

export default function CitationText({ text }: { text: string }) {
  const parts: (string | { file: string; start: string; end?: string })[] = [];
  let lastIndex = 0;

  for (const match of text.matchAll(CITATION_PATTERN)) {
    const [full, file, start, end] = match;
    const index = match.index ?? 0;
    if (index > lastIndex) parts.push(text.slice(lastIndex, index));
    parts.push({ file, start, end });
    lastIndex = index + full.length;
  }
  if (lastIndex < text.length) parts.push(text.slice(lastIndex));

  return (
    <p className="text-[15px] leading-relaxed text-text">
      {parts.map((part, i) =>
        typeof part === "string" ? (
          <span key={i}>{part}</span>
        ) : (
          <span
            key={i}
            className="mx-0.5 inline-flex items-center rounded border border-accent/40 bg-accent-wash px-1.5 py-0.5 font-mono text-[13px] text-accent"
          >
            {part.file}:{part.start}
            {part.end ? `-${part.end}` : ""}
          </span>
        ),
      )}
    </p>
  );
}
