# tests/test_cancellation_spike.py
"""Phase 0 cancellation spike (spec section 6.6, section 10, section 13).

Validates the load-bearing assumption behind the engine's barge-in design:
that abandoning iteration over llama-cpp-python's streaming generator, from
a worker thread, actually stops decoding promptly. If this fails, section
6.6's "perceived stop is engine-owned, provider releases at next boundary"
design doesn't hold for llama-cpp-python and an alternative LLM provider
(e.g. mlx-lm) must be prioritized (spec section 13).

Requires the model at models/qwen2.5-0.5b-instruct-q8_0.gguf (repo root).
"""
import pathlib
import threading
import time

import pytest

pytest.importorskip("llama_cpp")
from llama_cpp import Llama  # noqa: E402

MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "qwen2.5-0.5b-instruct-q8_0.gguf"


@pytest.mark.integration
def test_abandoning_stream_iteration_stops_promptly():
    assert MODEL_PATH.exists(), f"test model not found at {MODEL_PATH}"
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=512, verbose=False)

    stop_event = threading.Event()
    tokens_received = []
    wind_down_seconds = {}

    def worker():
        stream = llm(
            "Write a long, detailed story about a journey across the mountains.",
            max_tokens=256,
            stream=True,
        )
        start = time.perf_counter()
        for chunk in stream:
            tokens_received.append(chunk["choices"][0]["text"])
            if stop_event.is_set():
                break  # abandon iteration -- this is the behavior under test
        wind_down_seconds["value"] = time.perf_counter() - start

    thread = threading.Thread(target=worker)
    thread.start()

    # Let a few tokens generate, then signal abandonment.
    time.sleep(0.3)
    abandon_at = time.perf_counter()
    stop_event.set()

    thread.join(timeout=5.0)
    stop_latency = time.perf_counter() - abandon_at

    assert not thread.is_alive(), "worker thread did not stop within 5s of abandoning iteration"
    assert len(tokens_received) > 0, "no tokens were generated before abandonment"

    # Record findings -- the assertion bound is generous (spec section 6.6
    # targets "roughly one token time", 10-60ms typical, but the hard
    # contract is "release at the next boundary", not a specific millisecond
    # figure -- this assertion catches gross failures, not perf regressions).
    assert stop_latency < 2.0, (
        f"llama-cpp-python took {stop_latency:.3f}s to stop after abandoned "
        f"iteration -- exceeds the generous 2s sanity bound, investigate"
    )

    print(f"\nCancellation spike result: stop_latency={stop_latency:.4f}s, "
          f"tokens_before_abandon={len(tokens_received)}, "
          f"worker_wall_time={wind_down_seconds['value']:.4f}s")
