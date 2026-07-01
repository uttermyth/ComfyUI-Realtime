import { useCallback, useEffect, useState } from "react";
import { fetchPipelines } from "../protocol/api";
import type { PipelineSummary } from "../protocol/types";

export function usePipelineList() {
  const [pipelines, setPipelines] = useState<PipelineSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    setLoading(true);
    setError(null);
    fetchPipelines()
      .then((result) => setPipelines(result))
      .catch((err: Error) => {
        setPipelines([]);
        setError(err.message);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { pipelines, loading, error, refresh };
}
