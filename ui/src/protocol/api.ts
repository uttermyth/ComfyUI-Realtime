import type { PipelineSummary } from "./types";

/** Relative path, not an absolute URL: this extension is served from
 * ComfyUI's own origin (it's loaded via WEB_DIRECTORY), so a same-origin
 * relative fetch is correct and avoids the CORS/origin-check friction a
 * cross-origin client (like the standalone manual_voice_test.html tool)
 * has to work around. */
export async function fetchPipelines(): Promise<PipelineSummary[]> {
  const response = await fetch("/realtime/pipelines");
  if (!response.ok) {
    throw new Error(`GET /realtime/pipelines failed with status ${response.status}`);
  }
  const data = (await response.json()) as { pipelines: PipelineSummary[] };
  return data.pipelines;
}
