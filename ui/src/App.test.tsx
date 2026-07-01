import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import * as pipelineListHook from "./hooks/usePipelineList";

jest.mock("./hooks/usePipelineList");

test("shows the pipeline selector before connecting, and a Disconnect button after", async () => {
  jest.spyOn(pipelineListHook, "usePipelineList").mockReturnValue({
    pipelines: [{ name: "echo", modalities: { input: ["text"], output: ["text"] }, providers: { llm: "EchoLLM" }, voices: [], registered_at: "2026-01-01T00:00:00Z" }],
    loading: false,
    error: null,
    refresh: jest.fn(),
  });

  render(<App />);
  expect(screen.getByRole("button", { name: /connect/i })).toBeInTheDocument();

  await userEvent.click(screen.getByRole("button", { name: /connect/i }));
  expect(screen.getByRole("button", { name: /disconnect/i })).toBeInTheDocument();
  expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
});
