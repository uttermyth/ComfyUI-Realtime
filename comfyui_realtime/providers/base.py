"""Provider interfaces.

The pipeline engine consumes only these interfaces, never a specific
library directly -- this insulates it from upstream library churn.
Implementations wrap synchronous libraries via engine/executor_bridge.py;
the interfaces themselves stay async so the engine never blocks on them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationOptions:
    temperature: float = 0.8
    max_tokens: int | None = None


@dataclass
class GenerationDelta:
    text: str
    finished: bool = False


@dataclass
class VoiceInfo:
    id: str
    name: str


class ILLMProvider(Protocol):
    """Language model. generate() is an async generator: each iteration
    yields one GenerationDelta as tokens become available."""

    async def generate(
        self, messages: list[ChatMessage], options: GenerationOptions
    ) -> AsyncIterator[GenerationDelta]:
        ...


class ITTSProvider(Protocol):
    """Text-to-Speech synthesis. Providers declare their native sample
    rate; the engine owns resampling to the wire format."""

    output_sample_rate: int
    output_format: str

    async def synthesize(
        self, text_stream: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[bytes]:
        ...

    def list_voices(self) -> list[VoiceInfo]:
        ...


@dataclass
class VADResult:
    """Result of one VAD analysis call. speech_started/speech_ended flag a
    detected boundary in THIS chunk; speech_probability is the raw model
    output regardless of whether a boundary fired."""

    speech_probability: float
    speech_started: bool = False
    speech_ended: bool = False


@dataclass
class TranscriptionResult:
    text: str
    language: str | None = None


class IVADProvider(Protocol):
    """Voice Activity Detection. analyze() is a single blocking call
    returning one result -- dispatched via asyncio.to_thread(), not the
    streaming executor_bridge (see Global Constraints)."""

    sample_rate: int
    chunk_duration_ms: int

    async def analyze(self, audio_chunk: bytes) -> VADResult:
        ...


class ISTTProvider(Protocol):
    """Speech-to-Text. transcribe() is one-shot on a complete buffered
    utterance -- streaming transcription is not implemented."""

    sample_rate: int

    async def transcribe(self, audio_buffer: bytes, language: str | None = None) -> TranscriptionResult:
        ...
