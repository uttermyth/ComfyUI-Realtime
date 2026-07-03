"""Provider node classes are imported dynamically -- pyproject.toml only
guarantees `[default]`'s providers are installed; faster-whisper and piper
are opt-in extras. Rather than hand-writing a try/except per provider (and
remembering to add one for every future provider), each entry in
_PROVIDER_MODULES is imported defensively: if its underlying library isn't
installed, that one node is skipped (with a warning naming the extra to
install) instead of crashing this whole package -- and every other
provider -- at custom-node load time. See root __init__.py's
get_node_list(), which builds its provider list from __all__ below rather
than a hardcoded import list, so a skipped provider here is skipped there too.
"""
from __future__ import annotations

import importlib
import logging

logger = logging.getLogger("comfyui_realtime")

_PROVIDER_MODULES = [
    ("faster_whisper_stt", "FasterWhisperSTTProviderNode"),
    ("llama_cpp_llm", "LlamaCppLLMProviderNode"),
    ("piper_tts", "PiperTTSProviderNode"),
    ("pocket_tts", "PocketTTSProviderNode"),
    ("silero_vad", "SileroVADProviderNode"),
    ("transformers_llm", "TransformersLLMProviderNode"),
    ("vllm_llm", "VLLMProviderNode"),
    ("whisper_cpp_stt", "WhisperCppSTTProviderNode"),
]

__all__: list[str] = []

for _module_name, _class_name in _PROVIDER_MODULES:
    try:
        _module = importlib.import_module(f".{_module_name}", __name__)
    except ImportError as exc:
        logger.warning(
            "comfyui-realtime: %s unavailable (%s) -- install its optional "
            "dependency (see README's provider dependency table) to enable it",
            _class_name,
            exc,
        )
        continue
    globals()[_class_name] = getattr(_module, _class_name)
    __all__.append(_class_name)
