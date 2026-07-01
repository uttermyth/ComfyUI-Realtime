export function MicControls({
  isSpeaking,
  isCapturing,
  onStart,
  onStop,
}: {
  isSpeaking: boolean;
  isCapturing: boolean;
  onStart: () => void;
  onStop: () => void;
}) {
  return (
    <div className="flex items-center gap-2">
      {isCapturing ? (
        <button className="rounded-sm bg-primary-background text-text-primary px-3 py-2 text-sm cursor-pointer hover:bg-primary-background-hover" onClick={onStop}>Stop talking</button>
      ) : (
        <button className="rounded-sm bg-primary-background text-text-primary px-3 py-2 text-sm cursor-pointer hover:bg-primary-background-hover" onClick={onStart}>Start talking</button>
      )}
      {isSpeaking && <span className="text-sm text-muted-foreground">Listening...</span>}
    </div>
  );
}
