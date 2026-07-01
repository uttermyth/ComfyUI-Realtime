import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PipelineSelector } from "./PipelineSelector";
import * as pipelineListHook from "../hooks/usePipelineList";

jest.mock("../hooks/usePipelineList");

const samplePipelines = [
  { name: "echo", modalities: { input: ["text"], output: ["text"] }, providers: { llm: "EchoLLM" }, voices: [], registered_at: "2026-01-01T00:00:00Z" },
  { name: "phase2-full", modalities: { input: ["text", "audio"], output: ["text", "audio"] }, providers: { llm: "LlamaCppLLMProvider" }, voices: ["voice-a", "voice-b"], registered_at: "2026-01-02T00:00:00Z" },
];

test("lists every registered pipeline as a selectable option", () => {
  jest.spyOn(pipelineListHook, "usePipelineList").mockReturnValue({
    pipelines: samplePipelines,
    loading: false,
    error: null,
    refresh: jest.fn(),
  });
  render(<PipelineSelector onConnect={jest.fn()} />);
  expect(screen.getByRole("option", { name: /echo/ })).toBeInTheDocument();
  expect(screen.getByRole("option", { name: /phase2-full/ })).toBeInTheDocument();
});

test("clicking Connect calls onConnect with the full selected pipeline object", async () => {
  jest.spyOn(pipelineListHook, "usePipelineList").mockReturnValue({
    pipelines: samplePipelines,
    loading: false,
    error: null,
    refresh: jest.fn(),
  });
  const onConnect = jest.fn();
  render(<PipelineSelector onConnect={onConnect} />);

  await userEvent.selectOptions(screen.getByRole("combobox"), "phase2-full");
  await userEvent.click(screen.getByRole("button", { name: /connect/i }));

  expect(onConnect).toHaveBeenCalledWith(samplePipelines[1]);
});

test("shows an error message instead of the dropdown when the fetch failed", () => {
  jest.spyOn(pipelineListHook, "usePipelineList").mockReturnValue({
    pipelines: [],
    loading: false,
    error: "network down",
    refresh: jest.fn(),
  });
  render(<PipelineSelector onConnect={jest.fn()} />);
  expect(screen.getByText(/network down/)).toBeInTheDocument();
  expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
});

test("shows a 'Pipeline' label associated with the dropdown", () => {
  jest.spyOn(pipelineListHook, "usePipelineList").mockReturnValue({
    pipelines: samplePipelines,
    loading: false,
    error: null,
    refresh: jest.fn(),
  });
  render(<PipelineSelector onConnect={jest.fn()} />);
  // getByLabelText finds the select element via its associated label
  expect(screen.getByLabelText("Pipeline")).toBeInTheDocument();
});

test("shows helper text explaining what Connect does", () => {
  jest.spyOn(pipelineListHook, "usePipelineList").mockReturnValue({
    pipelines: samplePipelines,
    loading: false,
    error: null,
    refresh: jest.fn(),
  });
  render(<PipelineSelector onConnect={jest.fn()} />);
  expect(
    screen.getByText("Connect to start a real-time voice session with your workflow.")
  ).toBeInTheDocument();
});
