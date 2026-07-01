import { parseRealtimeEvent } from "./events";
import type { RealtimeEvent } from "./types";

type EventListener = (event: RealtimeEvent) => void;

/** Wraps the WebSocket connection to this project's own /v1/realtime
 * route -- the exact same route, same query param, same message shapes
 * any third-party client uses (no privileged access). The WebSocket
 * constructor is injected (defaulting to the real global) purely so
 * tests can substitute a fake one; production code never passes this
 * argument. */
export class RealtimeConnection {
  private ws: WebSocket | null = null;
  private listeners: EventListener[] = [];

  constructor(private webSocketCtor: typeof WebSocket = WebSocket) {}

  connect(pipelineName: string): void {
    const ws = new this.webSocketCtor(`/v1/realtime?model=${encodeURIComponent(pipelineName)}`);
    ws.onmessage = (event: { data: string }) => {
      const parsed = parseRealtimeEvent(event.data);
      for (const listener of this.listeners) {
        listener(parsed);
      }
    };
    this.ws = ws;
  }

  disconnect(): void {
    this.ws?.close();
    this.ws = null;
  }

  addEventListener(listener: EventListener): void {
    this.listeners.push(listener);
  }

  sendAudioChunk(base64Audio: string): void {
    this.send({ type: "input_audio_buffer.append", audio: base64Audio });
  }

  sendTextMessage(text: string): void {
    this.send({
      type: "conversation.item.create",
      item: { role: "user", content: [{ type: "input_text", text }] },
    });
    this.send({ type: "response.create" });
  }

  requestResponse(): void {
    this.send({ type: "response.create" });
  }

  sendSessionUpdate(voice: string): void {
    this.send({ type: "session.update", session: { voice } });
  }

  private send(message: Record<string, unknown>): void {
    this.ws?.send(JSON.stringify(message));
  }
}
