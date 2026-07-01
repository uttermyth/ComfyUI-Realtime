# comfyui_realtime/engine/sentence_chunker.py
"""Sentence-boundary chunker for LLM-to-TTS streaming.

Naive ". ! ?" boundary detection -- works for English but fails on
abbreviations, numbers, and multilingual text. A configurable chunker
strategy may be needed for non-English use cases. This lets TTS start on
the first complete sentence rather than waiting for the full LLM response,
which is what gives the realtime pipeline its low latency.
"""
from __future__ import annotations

_BOUNDARY_CHARS = (".", "!", "?")


class SentenceChunker:
    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        sentences: list[str] = []
        start = 0
        for i, ch in enumerate(self._buffer):
            if ch in _BOUNDARY_CHARS:
                sentences.append(self._buffer[start : i + 1].strip())
                start = i + 1
        self._buffer = self._buffer[start:]
        return sentences

    def flush(self) -> str | None:
        remainder = self._buffer.strip()
        self._buffer = ""
        return remainder if remainder else None
