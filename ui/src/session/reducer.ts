import { isKnownRealtimeEvent } from "../protocol/events";
import type { RealtimeEvent } from "../protocol/types";

export type ConversationEntry = {
  role: "user" | "assistant";
  text: string;
  inProgress: boolean;
};

export interface RealtimeSessionState {
  status: "idle" | "connecting" | "connected" | "error";
  errorMessage: string | null;
  modalitiesInput: string[];
  modalitiesOutput: string[];
  hasTurnDetection: boolean;
  isSpeaking: boolean;
  conversation: ConversationEntry[];
  hasEverResponded: boolean;
  currentVoice: string | null;
}

export type LocalAction =
  | { type: "local.connecting" }
  | { type: "local.disconnected" }
  | { type: "local.user_text_sent"; text: string };
export type ReducerAction = RealtimeEvent | LocalAction;

export const initialRealtimeSessionState: RealtimeSessionState = {
  status: "idle",
  errorMessage: null,
  modalitiesInput: [],
  modalitiesOutput: [],
  hasTurnDetection: false,
  isSpeaking: false,
  conversation: [],
  hasEverResponded: false,
  currentVoice: null,
};

/** Translates the protocol's event stream into UI state. The one
 * non-obvious rule here (see the design doc): hasEverResponded only flips
 * on an actual response.created event, never inferred from session.created
 * -- a continuous-transcription pipeline and a speech-to-text pipeline
 * report identical modalities, so this can't be predicted in advance.
 *
 * Local actions are checked first, before isKnownRealtimeEvent -- they
 * have their own clean literal types (LocalAction has no catch-all member
 * to exclude), so narrowing on them directly is fine. Only the remaining
 * RealtimeEvent branch needs the explicit isKnownRealtimeEvent exclusion
 * (see protocol/types.ts and protocol/events.ts for why UnknownEvent's
 * `type: string` can't be excluded by a plain switch/===, found and fixed
 * during Task 2's review, before this reducer existed to need it). */
export function realtimeSessionReducer(state: RealtimeSessionState, action: ReducerAction): RealtimeSessionState {
  if (action.type === "local.connecting") {
    return { ...initialRealtimeSessionState, status: "connecting" };
  }
  if (action.type === "local.disconnected") {
    return { ...initialRealtimeSessionState, status: "idle" };
  }
  if (action.type === "local.user_text_sent" && "text" in action) {

    return { ...state, conversation: [...state.conversation, { role: "user", text: action.text, inProgress: false }] };
  }
  if (!isKnownRealtimeEvent(action)) {
    return state; // a wire event this client doesn't model -- ignore, don't crash
  }
  switch (action.type) {
    case "session.created":
      return {
        ...state,
        status: "connected",
        modalitiesInput: action.session.modalities.input,
        modalitiesOutput: action.session.modalities.output,
        hasTurnDetection: action.session.turn_detection !== null,
        currentVoice: action.session.voice,
      };
    case "session.updated":
      return { ...state, currentVoice: action.session.voice };
    case "error":
      return { ...state, status: "error", errorMessage: action.error.message };
    case "input_audio_buffer.speech_started":
      return { ...state, isSpeaking: true };
    case "input_audio_buffer.speech_stopped":
      return { ...state, isSpeaking: false };
    case "conversation.item.input_audio_transcription.completed":
      return { ...state, conversation: [...state.conversation, { role: "user", text: action.transcript, inProgress: false }] };
    case "conversation.item.input_audio_transcription.failed":
      return { ...state, conversation: [...state.conversation, { role: "user", text: "(transcription failed: " + action.error.message + ")", inProgress: false }] };
    case "response.created":
      return {
        ...state,
        hasEverResponded: true,
        conversation: [...state.conversation, { role: "assistant", text: "", inProgress: true }],
      };
    case "response.output_text.delta": {
      const lastIdx = state.conversation.map((e, i) => ({ e, i })).reverse().find(({ e }) => e.role === "assistant" && e.inProgress)?.i;
      if (lastIdx === undefined) return state;
      const updated = state.conversation.map((entry, i) =>
        i === lastIdx ? { ...entry, text: entry.text + action.delta } : entry
      );
      return { ...state, conversation: updated };
    }
    case "response.done": {
      const updated = state.conversation.map((entry) =>
        entry.role === "assistant" && entry.inProgress ? { ...entry, inProgress: false } : entry
      );
      return { ...state, conversation: updated };
    }
    default:
      return state;
  }
}
