import { renderHook, waitFor } from "@testing-library/react";
import { usePipelineList } from "./usePipelineList";
import * as api from "../protocol/api";

jest.mock("../protocol/api");

test("fetches pipelines on mount and exposes them once loaded", async () => {
  const pipelines = [
    { name: "echo", modalities: { input: ["text"], output: ["text"] }, providers: { llm: "EchoLLM" }, voices: [], registered_at: "2026-01-01T00:00:00Z" },
  ];
  jest.spyOn(api, "fetchPipelines").mockResolvedValue(pipelines);

  const { result } = renderHook(() => usePipelineList());
  expect(result.current.loading).toBe(true);

  await waitFor(() => expect(result.current.loading).toBe(false));
  expect(result.current.pipelines).toEqual(pipelines);
  expect(result.current.error).toBeNull();
});

test("surfaces a fetch failure as an error, not a thrown exception", async () => {
  jest.spyOn(api, "fetchPipelines").mockRejectedValue(new Error("network down"));

  const { result } = renderHook(() => usePipelineList());
  await waitFor(() => expect(result.current.loading).toBe(false));
  expect(result.current.error).toBe("network down");
  expect(result.current.pipelines).toEqual([]);
});
