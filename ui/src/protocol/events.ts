import type { KnownRealtimeEvent, RealtimeEvent } from "./types";

/** Every wire message already has the shape RealtimeEvent expects (this
 * project's GA dialect serializer always sends well-formed events) -- the
 * one thing actually worth guarding here is malformed JSON, which would
 * otherwise surface as an opaque SyntaxError far from where it happened. */
export function parseRealtimeEvent(raw: string): RealtimeEvent {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (cause) {
    throw new Error(`failed to parse realtime event: ${raw}`, { cause });
  }
  return parsed as RealtimeEvent;
}

const KNOWN_EVENT_TYPE_VALUES: ReadonlySet<string> = new Set([
  "session.created",
  "session.updated",
  "error",
  "input_audio_buffer.speech_started",
  "input_audio_buffer.speech_stopped",
  "conversation.item.input_audio_transcription.completed",
  "conversation.item.input_audio_transcription.failed",
  "conversation.item.added",
  "conversation.item.done",
  "response.created",
  "response.output_text.delta",
  "response.output_text.done",
  "response.output_audio.delta",
  "response.done",
]);

/** Narrows RealtimeEvent down to KnownRealtimeEvent, excluding
 * UnknownEvent -- consumers should call this once, then switch/narrow on
 * the result normally (see types.ts for why UnknownEvent can't just sit
 * directly in the same union as the literal-typed events). */
export function isKnownRealtimeEvent(event: RealtimeEvent): event is KnownRealtimeEvent {
  return KNOWN_EVENT_TYPE_VALUES.has(event.type);
}
