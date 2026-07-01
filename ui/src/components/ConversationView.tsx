import { useState } from "react";
import { useRealtimeSession } from "../session/useRealtimeSession";
import { MicControls } from "./MicControls";
import { TextInputBox } from "./TextInputBox";
import { TranscriptFeed } from "./TranscriptFeed";
import { VoiceSelector } from "./VoiceSelector";

export function ConversationView({ session, voices }: { session: ReturnType<typeof useRealtimeSession>; voices: string[] }) {
  const { state, startMic, stopMic, sendText } = session;
  const [isCapturing, setIsCapturing] = useState(false);

  if (state.status === "connecting") return <p className="text-sm text-muted-foreground">Starting session...</p>;
  if (state.status === "error") return <p className="text-sm text-muted-foreground">Connection failed: {state.errorMessage}</p>;
  if (state.status !== "connected") return null;

  const usesAudioInput = state.modalitiesInput.includes("audio");

  return (
    <div className="flex flex-col gap-4">
      <VoiceSelector voices={voices} currentVoice={state.currentVoice} onSelect={session.setVoice} />
      {usesAudioInput ? (
        <>
          <MicControls
            isSpeaking={state.isSpeaking}
            isCapturing={isCapturing}
            onStart={() => {
              setIsCapturing(true);
              void startMic();
            }}
            onStop={() => {
              setIsCapturing(false);
              stopMic();
            }}
          />
          <TranscriptFeed conversation={state.conversation} />
        </>
      ) : (
        <TextInputBox onSend={sendText} />
      )}
    </div>
  );
}
