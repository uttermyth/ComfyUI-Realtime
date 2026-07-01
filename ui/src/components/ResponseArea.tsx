export function ResponseArea({ text, inProgress }: { text: string; inProgress: boolean }) {
  return (
    <div className="flex flex-col gap-1" aria-label="response">
      <p className="text-sm text-text-primary m-0">{text}</p>
      {inProgress && <span className="text-sm text-muted-foreground">...</span>}
    </div>
  );
}
