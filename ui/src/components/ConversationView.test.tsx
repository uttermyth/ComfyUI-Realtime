import { render, screen } from "@testing-library/react";
import { ConversationView } from "./ConversationView";
import { initialRealtimeSessionState } from "../session/reducer";
import type { useRealtimeSession } from "../session/useRealtimeSession";

function fakeSession(stateOverrides: Partial<typeof initialRealtimeSessionState>): ReturnType<typeof useRealtimeSession> {
  return {
    state: { ...initialRealtimeSessionState, ...stateOverrides },
    connect: jest.fn(),
    disconnect: jest.fn(),
    startMic: jest.fn(),
    stopMic: jest.fn(),
    sendText: jest.fn(),
    setVoice: jest.fn(),
  };
}

test("shows mic controls and the transcript feed when modalities_input includes audio", () => {
  render(<ConversationView session={fakeSession({ status: "connected", modalitiesInput: ["text", "audio"] })} voices={[]} />);
  expect(screen.getByRole("button", { name: /start talking/i })).toBeInTheDocument();
  expect(screen.getByLabelText("transcript")).toBeInTheDocument();
  expect(screen.queryByLabelText("message")).not.toBeInTheDocument();
});

test("shows the text input instead of mic controls when modalities_input is text only", () => {
  render(<ConversationView session={fakeSession({ status: "connected", modalitiesInput: ["text"] })} voices={[]} />);
  expect(screen.getByLabelText("message")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /start talking/i })).not.toBeInTheDocument();
});

test("no assistant entries appear before any response.created has fired", () => {
  render(
    <ConversationView
      session={fakeSession({ status: "connected", modalitiesInput: ["text", "audio"], hasEverResponded: false, conversation: [] })}
      voices={[]}
    />
  );
  expect(screen.queryByText("Assistant")).not.toBeInTheDocument();
});

test("an assistant entry's text appears in the transcript feed once a response has arrived", () => {
  render(
    <ConversationView
      session={fakeSession({
        status: "connected",
        modalitiesInput: ["text", "audio"],
        hasEverResponded: true,
        conversation: [{ role: "assistant", text: "hi", inProgress: false }],
      })}
      voices={[]}
    />
  );
  expect(screen.getByLabelText("transcript")).toBeInTheDocument();
  expect(screen.getByText("hi")).toBeInTheDocument();
});

test("shows a connecting message before the session is established", () => {
  render(<ConversationView session={fakeSession({ status: "connecting" })} voices={[]} />);
  expect(screen.getByText(/starting session/i)).toBeInTheDocument();
});

test("shows the error message on failure", () => {
  render(<ConversationView session={fakeSession({ status: "error", errorMessage: "pipeline_not_found" })} voices={[]} />);
  expect(screen.getByText(/pipeline_not_found/)).toBeInTheDocument();
  expect(screen.getByText(/connection failed/i)).toBeInTheDocument();
});

test("shows the voice selector when the pipeline has more than one voice", () => {
  render(
    <ConversationView
      session={fakeSession({ status: "connected", modalitiesInput: ["text", "audio"], currentVoice: "voice-a" })}
      voices={["voice-a", "voice-b"]}
    />
  );
  expect(screen.getByLabelText("voice")).toBeInTheDocument();
});

test("does not show the voice selector when the pipeline has only one voice", () => {
  render(
    <ConversationView
      session={fakeSession({ status: "connected", modalitiesInput: ["text", "audio"] })}
      voices={["voice-a"]}
    />
  );
  expect(screen.queryByLabelText("voice")).not.toBeInTheDocument();
});
