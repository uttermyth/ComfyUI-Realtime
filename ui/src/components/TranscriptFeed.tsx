import type { ConversationEntry } from "../session/reducer";

export function TranscriptFeed({ conversation }: { conversation: ConversationEntry[] }) {
  return (
    <ul className="flex flex-col gap-1 list-none p-0 m-0" aria-label="transcript">
      {conversation.map((entry, i) => (
        <li key={i} className="flex flex-col gap-0.5">
          <span className="text-xs font-semibold text-muted-foreground uppercase">
            {entry.role === "user" ? "You" : "Assistant"}
          </span>
          <span className="text-sm text-text-primary">
            {entry.text}
            {entry.inProgress && <span className="text-muted-foreground">...</span>}
          </span>
        </li>
      ))}
    </ul>
  );
}
