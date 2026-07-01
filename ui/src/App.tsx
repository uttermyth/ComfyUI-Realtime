import { useState } from "react";
import { ConversationView } from "./components/ConversationView";
import { PipelineSelector } from "./components/PipelineSelector";
import { useRealtimeSession } from "./session/useRealtimeSession";
import type { PipelineSummary } from "./protocol/types";

export default function App() {
  const session = useRealtimeSession();
  const [connected, setConnected] = useState(false);
  const [availableVoices, setAvailableVoices] = useState<string[]>([]);

  const handleConnect = (pipeline: PipelineSummary) => {
    session.connect(pipeline.name);
    setAvailableVoices(pipeline.voices);
    setConnected(true);
  };

  const handleDisconnect = () => {
    session.disconnect();
    setConnected(false);
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex min-h-16 items-center border-b border-border-default px-3">
        <span className="truncate font-bold">ComfyUI Realtime</span>
      </div>
      <div className="flex flex-1 min-h-0 flex-col gap-4 overflow-y-auto overflow-x-hidden p-4">
        {connected ? (
          <>
            <button className="rounded-sm bg-secondary-background text-text-primary border border-border-default px-3 py-2 text-sm cursor-pointer hover:bg-secondary-background-hover" onClick={handleDisconnect}>Disconnect</button>
            <ConversationView session={session} voices={availableVoices} />
          </>
        ) : (
          <PipelineSelector onConnect={handleConnect} />
        )}
      </div>
    </div>
  );
}
