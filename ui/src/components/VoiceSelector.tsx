export function VoiceSelector({
  voices,
  currentVoice,
  onSelect,
}: {
  voices: string[];
  currentVoice: string | null;
  onSelect: (voice: string) => void;
}) {
  if (voices.length < 2) return null;
  return (
    <div className="flex items-center gap-2">
      <label className="flex items-center gap-2 text-sm text-text-primary">
        Voice:{" "}
        <select className="rounded-sm border border-border-default bg-secondary-background text-text-primary px-2 py-1 text-sm" aria-label="voice" value={currentVoice ?? voices[0]} onChange={(e) => onSelect(e.target.value)}>
          {voices.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
