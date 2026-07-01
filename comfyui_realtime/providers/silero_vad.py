"""SileroVADProvider wraps silero-vad (PyPI) -- the model ships bundled
with the package, no separate download. analyze() is a single blocking call
(not a stream), so it's dispatched via asyncio.to_thread() rather than the
streaming executor_bridge (VAD is on the hot path; see Global Constraints).

VADIterator requires audio as a torch.Tensor of float32 samples in [-1, 1],
exactly 512 samples at a time (16kHz) -- callers are responsible for
re-chunking into that exact frame size before calling analyze(); this
provider does not re-chunk or resample itself.

Note: this calls the underlying model twice per chunk (once via
VADIterator's own internal call for boundary detection, once directly for
the raw probability VADResult always reports) -- VADIterator doesn't
expose the probability from a non-boundary call. Given Silero's ~1ms/frame
budget, doubling that is comfortably within the <10ms target; not worth a
more invasive wrapper unless profiling says otherwise.
"""
from __future__ import annotations

import asyncio
import threading

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

from .base import VADResult


class SileroVADProvider:
    sample_rate = 16000
    chunk_duration_ms = 32  # 512 samples at 16kHz

    def __init__(self, threshold: float = 0.5) -> None:
        self._model = load_silero_vad()
        self._iterator = VADIterator(self._model, sampling_rate=self.sample_rate, threshold=threshold)
        self._lock = threading.Lock()

    async def analyze(self, audio_chunk: bytes) -> VADResult:
        return await asyncio.to_thread(self._analyze_sync, audio_chunk)

    def _analyze_sync(self, audio_chunk: bytes) -> VADResult:
        samples = np.frombuffer(audio_chunk, dtype="<i2").astype(np.float32) / 32768.0
        tensor = torch.from_numpy(samples)
        with self._lock, torch.inference_mode():
            # Explicit inference_mode (not relying on ambient context) --
            # when this provider's model is loaded inside a host process
            # that itself runs node execution under torch.inference_mode()
            # (ComfyUI's execution.py does, for every node including this
            # one's own loader), the model's weights become inference
            # tensors. A later call to this model from a context that
            # isn't also inference_mode raises "Inference tensors cannot
            # be saved for backward" on the direct (non-VADIterator)
            # invocation below. VADIterator's own __call__ already wraps
            # itself in inference_mode internally, which is why only the
            # direct probability call needed this.
            speech_dict = self._iterator(tensor, return_seconds=False)
            probability = self._model(tensor, self.sample_rate).item()
        if speech_dict is None:
            return VADResult(speech_probability=probability)
        if "start" in speech_dict:
            return VADResult(speech_probability=probability, speech_started=True)
        return VADResult(speech_probability=probability, speech_ended=True)

    def unload(self) -> None:
        del self._model
        del self._iterator
