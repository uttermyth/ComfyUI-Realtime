import { useState } from "react";
import { usePipelineList } from "../hooks/usePipelineList";
import type { PipelineSummary } from "../protocol/types";

export function PipelineSelector({ onConnect }: { onConnect: (pipeline: PipelineSummary) => void }) {
  const { pipelines, loading, error, refresh } = usePipelineList();
  const [selected, setSelected] = useState<string>("");

  if (error) {
    return (
      <div className="flex flex-col gap-3">
        <p className="text-sm text-muted-foreground">Could not load pipelines: {error}</p>
        <button className="rounded-sm bg-secondary-background text-text-primary border border-border-default px-3 py-2 text-sm cursor-pointer hover:bg-secondary-background-hover" onClick={refresh}>Retry</button>
      </div>
    );
  }

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading pipelines...</p>;
  }

  if (pipelines.length === 0) {
    return (
      <div className="flex flex-col gap-3">
        <p className="text-sm text-muted-foreground">No pipelines registered yet. Queue a workflow with a RealtimePipelineNode, then refresh.</p>
        <button className="rounded-sm bg-secondary-background text-text-primary border border-border-default px-3 py-2 text-sm cursor-pointer hover:bg-secondary-background-hover" onClick={refresh}>Refresh</button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1">
        <label
          htmlFor="pipeline"
          className="text-xs font-semibold text-muted-foreground uppercase"
        >
          Pipeline
        </label>
        <select
          id="pipeline"
          className="min-w-0 w-full rounded-sm border border-border-default bg-secondary-background text-text-primary px-2 py-1 text-sm"
          value={selected || pipelines[0].name}
          onChange={(e) => setSelected(e.target.value)}
        >
          {pipelines.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name} ({Object.values(p.providers).join(", ")})
            </option>
          ))}
        </select>
      </div>
      <p className="text-sm text-muted-foreground">
        Connect to start a real-time session with your workflow.
      </p>
      <div className="flex gap-2">
        <button
          className="flex-1 rounded-sm bg-primary-background text-text-primary px-3 py-2 text-sm cursor-pointer hover:bg-primary-background-hover"
          onClick={() =>
            onConnect(
              pipelines.find((p) => p.name === (selected || pipelines[0].name)) ??
                pipelines[0]
            )
          }
        >
          Connect
        </button>
        <button
          className="rounded-sm bg-secondary-background text-text-primary border border-border-default px-3 py-2 text-sm cursor-pointer hover:bg-secondary-background-hover"
          onClick={refresh}
        >
          Refresh
        </button>
      </div>
    </div>
  );
}
