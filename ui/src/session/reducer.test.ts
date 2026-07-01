import { initialRealtimeSessionState, realtimeSessionReducer } from "./reducer";

test("session.created moves to connected and records modalities/turn_detection", () => {
  const state = realtimeSessionReducer(initialRealtimeSessionState, {
    type: "session.created",
    session: { id: "x", model: "m", modalities: { input: ["text", "audio"], output: ["text"] }, turn_detection: { type: "server_vad" }, voice: null },
  });
  expect(state.status).toBe("connected");
  expect(state.modalitiesInput).toEqual(["text", "audio"]);
  expect(state.modalitiesOutput).toEqual(["text"]);
  expect(state.hasTurnDetection).toBe(true);
});

test("error moves to the error state with the message", () => {
  const state = realtimeSessionReducer(initialRealtimeSessionState, { type: "error", error: { code: "x", message: "boom" } });
  expect(state.status).toBe("error");
  expect(state.errorMessage).toBe("boom");
});

test("speech_started/speech_stopped toggle isSpeaking", () => {
  let state = realtimeSessionReducer(initialRealtimeSessionState, { type: "input_audio_buffer.speech_started" });
  expect(state.isSpeaking).toBe(true);
  state = realtimeSessionReducer(state, { type: "input_audio_buffer.speech_stopped" });
  expect(state.isSpeaking).toBe(false);
});

test("completed transcriptions accumulate in order", () => {
  let state = initialRealtimeSessionState;
  state = realtimeSessionReducer(state, { type: "conversation.item.input_audio_transcription.completed", transcript: "first" });
  state = realtimeSessionReducer(state, { type: "conversation.item.input_audio_transcription.completed", transcript: "second" });
  expect(state.conversation).toHaveLength(2);
  expect(state.conversation[0]).toEqual({ role: "user", text: "first", inProgress: false });
  expect(state.conversation[1]).toEqual({ role: "user", text: "second", inProgress: false });
});

test("a failed transcription is recorded, not silently dropped", () => {
  const state = realtimeSessionReducer(initialRealtimeSessionState, {
    type: "conversation.item.input_audio_transcription.failed",
    error: { message: "audio too short" },
  });
  expect(state.conversation[0]).toEqual({ role: "user", text: "(transcription failed: audio too short)", inProgress: false });
});

test("hasEverResponded stays false until a response.created event actually arrives -- this is the load-bearing claim from the design doc, since session.created cannot predict it", () => {
  let state = realtimeSessionReducer(initialRealtimeSessionState, {
    type: "session.created",
    session: { id: "x", model: "m", modalities: { input: ["text", "audio"], output: ["text"] }, turn_detection: { type: "server_vad" }, voice: null },
  });
  expect(state.hasEverResponded).toBe(false);
  state = realtimeSessionReducer(state, {
    type: "conversation.item.input_audio_transcription.completed",
    transcript: "hello",
  });
  expect(state.hasEverResponded).toBe(false);
  state = realtimeSessionReducer(state, { type: "response.created", response: { id: "r1", status: "in_progress" } });
  expect(state.hasEverResponded).toBe(true);
});

test("response text accumulates from deltas and resets when a new response starts", () => {
  let state = realtimeSessionReducer(initialRealtimeSessionState, { type: "response.created", response: { id: "r1", status: "in_progress" } });
  state = realtimeSessionReducer(state, { type: "response.output_text.delta", delta: "Hel" });
  state = realtimeSessionReducer(state, { type: "response.output_text.delta", delta: "lo" });
  const lastEntry = state.conversation[state.conversation.length - 1];
  expect(lastEntry.text).toBe("Hello");
  expect(lastEntry.inProgress).toBe(true);
  state = realtimeSessionReducer(state, { type: "response.done", response: { status: "completed" } });
  expect(state.conversation[state.conversation.length - 1].inProgress).toBe(false);
  // A new response.created appends a NEW entry (not overwrites)
  state = realtimeSessionReducer(state, { type: "response.created", response: { id: "r2", status: "in_progress" } });
  expect(state.conversation).toHaveLength(2);
  expect(state.conversation[state.conversation.length - 1].text).toBe("");
});

test("local.connecting and local.disconnected reset to the right baseline status", () => {
  let state = realtimeSessionReducer(initialRealtimeSessionState, { type: "local.connecting" });
  expect(state.status).toBe("connecting");
  state = realtimeSessionReducer(state, { type: "local.disconnected" });
  expect(state.status).toBe("idle");
});

test("session.created sets currentVoice from the wire payload", () => {
  const state = realtimeSessionReducer(initialRealtimeSessionState, {
    type: "session.created",
    session: { id: "x", model: "m", modalities: { input: [], output: [] }, turn_detection: null, voice: "voice-a" },
  });
  expect(state.currentVoice).toBe("voice-a");
});

test("session.updated updates currentVoice", () => {
  const connected = realtimeSessionReducer(initialRealtimeSessionState, {
    type: "session.created",
    session: { id: "x", model: "m", modalities: { input: [], output: [] }, turn_detection: null, voice: "voice-a" },
  });
  const updated = realtimeSessionReducer(connected, {
    type: "session.updated",
    session: { id: "x", model: "m", modalities: { input: [], output: [] }, turn_detection: null, voice: "voice-b" },
  });
  expect(updated.currentVoice).toBe("voice-b");
});

test("local.user_text_sent appends a user entry to conversation", () => {
  const state = realtimeSessionReducer(initialRealtimeSessionState, { type: "local.user_text_sent", text: "hello there" });
  expect(state.conversation).toHaveLength(1);
  expect(state.conversation[0]).toEqual({ role: "user", text: "hello there", inProgress: false });
});

test("two complete responses both appear in conversation (regression: second response must not overwrite first)", () => {
  let state = initialRealtimeSessionState;
  // First response
  state = realtimeSessionReducer(state, { type: "response.created", response: { id: "r1", status: "in_progress" } });
  state = realtimeSessionReducer(state, { type: "response.output_text.delta", delta: "First response" });
  state = realtimeSessionReducer(state, { type: "response.done", response: { status: "completed" } });
  // Second response
  state = realtimeSessionReducer(state, { type: "response.created", response: { id: "r2", status: "in_progress" } });
  state = realtimeSessionReducer(state, { type: "response.output_text.delta", delta: "Second response" });
  state = realtimeSessionReducer(state, { type: "response.done", response: { status: "completed" } });

  const assistantEntries = state.conversation.filter((e) => e.role === "assistant");
  expect(assistantEntries).toHaveLength(2);
  expect(assistantEntries[0].text).toBe("First response");
  expect(assistantEntries[1].text).toBe("Second response");
});
