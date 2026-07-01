import { useState } from "react";

export function TextInputBox({ onSend }: { onSend: (text: string) => void }) {
  const [text, setText] = useState("");
  const submit = () => {
    if (text.trim().length === 0) return;
    onSend(text);
    setText("");
  };
  return (
    <div className="flex gap-2">
      <input
        className="flex-1 rounded-sm border border-border-default bg-input-surface text-text-primary px-3 py-2 text-sm focus:outline-none"
        aria-label="message"
        placeholder="Type a message and press Enter"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") submit();
        }}
      />
      <button className="rounded-sm bg-primary-background text-text-primary px-3 py-2 text-sm cursor-pointer hover:bg-primary-background-hover" onClick={submit}>Send</button>
    </div>
  );
}
