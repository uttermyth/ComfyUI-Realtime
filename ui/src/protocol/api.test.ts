import { fetchPipelines } from "./api";

describe("fetchPipelines", () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  test("returns the pipelines array from a successful response", async () => {
    const pipelines = [
      { name: "echo", modalities: { input: ["text"], output: ["text"] }, providers: { llm: "EchoLLM" }, registered_at: "2026-01-01T00:00:00Z" },
    ];
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ pipelines }),
    }) as unknown as typeof fetch;

    const result = await fetchPipelines();
    expect(result).toEqual(pipelines);
    expect(global.fetch).toHaveBeenCalledWith("/realtime/pipelines");
  });

  test("throws a clear error on a non-OK response", async () => {
    global.fetch = jest.fn().mockResolvedValue({ ok: false, status: 500 }) as unknown as typeof fetch;
    await expect(fetchPipelines()).rejects.toThrow(/500/);
  });
});
