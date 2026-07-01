/** Mirrors comfyui_realtime/engine/dialects.py's GADialectSerializer wire
 * shapes -- this project's own GA dialect, the same one
 * scripts/manual_voice_test.html already speaks successfully. */

export interface SessionCreatedEvent {
  type: "session.created";
  session: {
    id: string;
    model: string;
    modalities: { input: string[]; output: string[] };
    turn_detection: { type: string } | null;
    voice: string | null;
  };
}

export interface SessionUpdatedEvent {
  type: "session.updated";
  session: {
    id: string;
    model: string;
    modalities: { input: string[]; output: string[] };
    turn_detection: { type: string } | null;
    voice: string | null;
  };
}

export interface ErrorEvent {
  type: "error";
  error: { code: string; message: string };
}

export interface SpeechStartedEvent {
  type: "input_audio_buffer.speech_started";
}

export interface SpeechStoppedEvent {
  type: "input_audio_buffer.speech_stopped";
}

export interface TranscriptionCompletedEvent {
  type: "conversation.item.input_audio_transcription.completed";
  transcript: string;
}

export interface TranscriptionFailedEvent {
  type: "conversation.item.input_audio_transcription.failed";
  error: { message: string };
}

export interface ItemAddedEvent {
  type: "conversation.item.added";
  item: { content: Array<{ text: string }> };
}

export interface ItemDoneEvent {
  type: "conversation.item.done";
}

export interface ResponseCreatedEvent {
  type: "response.created";
  response: { id: string; status: string };
}

export interface ResponseOutputTextDeltaEvent {
  type: "response.output_text.delta";
  delta: string;
}

export interface ResponseOutputTextDoneEvent {
  type: "response.output_text.done";
}

export interface ResponseOutputAudioDeltaEvent {
  type: "response.output_audio.delta";
  delta: string;
}

export interface ResponseDoneEvent {
  type: "response.done";
  response: { status: string };
}

export interface UnknownEvent {
  type: string;
}

/** Every event type this client actually understands. Kept as its own
 * union (not folded directly into RealtimeEvent) because UnknownEvent's
 * `type: string` structurally overlaps every literal type below -- if it
 * were mixed into the same union TypeScript could never narrow
 * `event.type === "session.created"` down to just SessionCreatedEvent (it
 * can't prove UnknownEvent is excluded, since any string literal is
 * assignable to `string`). Consumers should narrow via isKnownRealtimeEvent
 * (events.ts) first, then switch/narrow on KnownRealtimeEvent normally. */
export type KnownRealtimeEvent =
  | SessionCreatedEvent
  | SessionUpdatedEvent
  | ErrorEvent
  | SpeechStartedEvent
  | SpeechStoppedEvent
  | TranscriptionCompletedEvent
  | TranscriptionFailedEvent
  | ItemAddedEvent
  | ItemDoneEvent
  | ResponseCreatedEvent
  | ResponseOutputTextDeltaEvent
  | ResponseOutputTextDoneEvent
  | ResponseOutputAudioDeltaEvent
  | ResponseDoneEvent;

export type RealtimeEvent = KnownRealtimeEvent | UnknownEvent;

export interface PipelineSummary {
  name: string;
  modalities: { input: string[]; output: string[] };
  providers: Record<string, string>;
  voices: string[];
  registered_at: string;
}
