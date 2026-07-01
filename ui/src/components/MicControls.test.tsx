import { render, screen } from "@testing-library/react";
import { MicControls } from "./MicControls";

test("shows Start talking button when not capturing", () => {
  render(
    <MicControls isSpeaking={false} isCapturing={false} onStart={jest.fn()} onStop={jest.fn()} />
  );
  expect(screen.getByRole("button", { name: /start talking/i })).toBeInTheDocument();
});

test("shows Stop talking button when capturing", () => {
  render(
    <MicControls isSpeaking={false} isCapturing={true} onStart={jest.fn()} onStop={jest.fn()} />
  );
  expect(screen.getByRole("button", { name: /stop talking/i })).toBeInTheDocument();
});

test("shows Listening indicator when server detects speech", () => {
  render(
    <MicControls isSpeaking={true} isCapturing={true} onStart={jest.fn()} onStop={jest.fn()} />
  );
  expect(screen.getByText("Listening...")).toBeInTheDocument();
});

test("does not show Listening indicator when no speech is detected", () => {
  render(
    <MicControls isSpeaking={false} isCapturing={true} onStart={jest.fn()} onStop={jest.fn()} />
  );
  expect(screen.queryByText("Listening...")).not.toBeInTheDocument();
});
