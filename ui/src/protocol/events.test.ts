import { parseRealtimeEvent } from "./events";
import type { SessionCreatedEvent, SessionUpdatedEvent } from "./types";

test("parses session.created", () => {
  const raw = JSON.stringify({
    type: "session.created",
    session: {
      id: "sess_x",
      model: "echo",
      modalities: { input: ["text"], output: ["text"] },
      turn_detection: null,
    },
  });
  const event = parseRealtimeEvent(raw);
  expect(event.type).toBe("session.created");
  // `UnknownEvent.type` is a plain `string`, so it structurally overlaps
  // every literal-typed member of the RealtimeEvent union -- TypeScript
  // can't narrow away `UnknownEvent` from an `event.type === "..."` check
  // alone. A cast (guarded by the assertion above) is the standard escape.
  if (event.type === "session.created") {
    const sessionEvent = event as SessionCreatedEvent;
    expect(sessionEvent.session.id).toBe("sess_x");
    expect(sessionEvent.session.turn_detection).toBeNull();
  }
});

test("parses response.output_text.delta", () => {
  const event = parseRealtimeEvent(JSON.stringify({ type: "response.output_text.delta", delta: "hi" }));
  expect(event).toEqual({ type: "response.output_text.delta", delta: "hi" });
});

test("falls back to an UnknownEvent for an unrecognized type, instead of throwing", () => {
  const event = parseRealtimeEvent(JSON.stringify({ type: "something.we.have.never.seen", foo: "bar" }));
  expect(event.type).toBe("something.we.have.never.seen");
});

test("throws a clear error on malformed JSON rather than a generic SyntaxError", () => {
  expect(() => parseRealtimeEvent("not json")).toThrow(/failed to parse realtime event/i);
});

test("parses session.created with a voice field", () => {
  const event = parseRealtimeEvent(
    JSON.stringify({
      type: "session.created",
      session: { id: "x", model: "m", modalities: { input: [], output: [] }, turn_detection: null, voice: "voice-a" },
    })
  );
  expect(event.type).toBe("session.created");
  if (event.type === "session.created") {
    const sessionEvent = event as SessionCreatedEvent;
    expect(sessionEvent.session.voice).toBe("voice-a");
  }
});

test("parses session.updated", () => {
  const event = parseRealtimeEvent(
    JSON.stringify({
      type: "session.updated",
      session: { id: "x", model: "m", modalities: { input: [], output: [] }, turn_detection: null, voice: "voice-b" },
    })
  );
  expect(event.type).toBe("session.updated");
  if (event.type === "session.updated") {
    const sessionEvent = event as SessionUpdatedEvent;
    expect(sessionEvent.session.voice).toBe("voice-b");
  }
});
