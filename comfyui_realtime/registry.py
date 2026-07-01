"""Pipeline registry.

Provider nodes register a PipelineConfig when their ComfyUI workflow runs,
on ComfyUI's prompt-executor thread. WebSocket sessions read the registry
from the aiohttp event-loop thread. A plain threading.Lock is correct here
-- the registry itself does no async work, so a lock (not an asyncio.Lock)
is the right primitive.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .providers.base import ILLMProvider, ISTTProvider, ITTSProvider, IVADProvider
from .session_registry import session_registry


@dataclass
class PipelineConfig:
    name: str
    llm: ILLMProvider | None = None
    vad: IVADProvider | None = None
    stt: ISTTProvider | None = None
    tts: ITTSProvider | None = None
    voice: str | None = None
    instructions: str | None = None
    temperature: float = 0.8
    registered_at: float = field(default_factory=time.time)
    modalities_input: list[str] = field(init=False, default_factory=lambda: ["text"])
    modalities_output: list[str] = field(init=False, default_factory=lambda: ["text"])

    def __post_init__(self) -> None:
        self.modalities_input = ["text", "audio"] if self.stt is not None else ["text"]
        self.modalities_output = ["text", "audio"] if self.tts is not None else ["text"]


class PipelineRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pipelines: dict[str, PipelineConfig] = {}

    def register(self, config: PipelineConfig) -> None:
        with self._lock:
            old_config = self._pipelines.get(config.name)
            self._pipelines[config.name] = config
            if old_config is not None:
                self._unload_orphaned_providers_locked(old_config)

    def get(self, name: str) -> PipelineConfig | None:
        with self._lock:
            return self._pipelines.get(name)

    def unregister(self, name: str) -> bool:
        with self._lock:
            old_config = self._pipelines.pop(name, None)
            if old_config is None:
                return False
            self._unload_orphaned_providers_locked(old_config)
            return True

    def list(self) -> list[PipelineConfig]:
        with self._lock:
            return list(self._pipelines.values())

    def _unload_orphaned_providers_locked(self, removed_config: "PipelineConfig") -> None:
        """Unload any provider from removed_config no longer referenced by
        any pipeline still in the registry -- but only if zero sessions are
        currently active. Per-provider tracking (defer unload until no session
        is actively using that specific provider) requires ownership tracking
        across both registries and is deliberately deferred as out of scope.
        Must be called while holding self._lock.
        """
        if session_registry.count() > 0:
            return

        still_referenced = set()
        for config in self._pipelines.values():
            for provider in (config.llm, config.vad, config.stt, config.tts):
                if provider is not None:
                    still_referenced.add(id(provider))

        for provider in (removed_config.llm, removed_config.vad, removed_config.stt, removed_config.tts):
            if provider is not None and id(provider) not in still_referenced and hasattr(provider, "unload"):
                provider.unload()


# Process-wide singleton -- the registry is shared between the websocket
# handler module and the node-registration module, both loaded once at
# custom-node import time.
pipeline_registry = PipelineRegistry()
