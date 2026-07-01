import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { VoiceSelector } from "./VoiceSelector";

test("renders nothing when there are fewer than two voices", () => {
  render(<VoiceSelector voices={["voice-a"]} currentVoice="voice-a" onSelect={jest.fn()} />);
  expect(screen.queryByLabelText("voice")).not.toBeInTheDocument();
});

test("renders nothing when there are zero voices", () => {
  render(<VoiceSelector voices={[]} currentVoice={null} onSelect={jest.fn()} />);
  expect(screen.queryByLabelText("voice")).not.toBeInTheDocument();
});

test("lists every voice and calls onSelect when a different one is chosen", async () => {
  const onSelect = jest.fn();
  render(<VoiceSelector voices={["voice-a", "voice-b"]} currentVoice="voice-a" onSelect={onSelect} />);
  const select = screen.getByLabelText("voice");
  expect(screen.getByRole("option", { name: "voice-a" })).toBeInTheDocument();
  expect(screen.getByRole("option", { name: "voice-b" })).toBeInTheDocument();
  await userEvent.selectOptions(select, "voice-b");
  expect(onSelect).toHaveBeenCalledWith("voice-b");
});
