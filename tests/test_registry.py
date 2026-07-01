import threading
import time as _time_module

from comfyui_realtime.registry import PipelineConfig, PipelineRegistry


class _StubLLM:
    async def generate(self, text: str) -> str:
        return f"echo: {text}"


def test_register_and_get():
    registry = PipelineRegistry()
    registry.register(PipelineConfig(name="echo", llm=_StubLLM()))
    found = registry.get("echo")
    assert found is not None
    assert found.name == "echo"


def test_get_missing_returns_none():
    registry = PipelineRegistry()
    assert registry.get("nonexistent") is None


def test_register_replaces_existing():
    registry = PipelineRegistry()
    registry.register(PipelineConfig(name="echo", llm=_StubLLM()))
    second_llm = _StubLLM()
    registry.register(PipelineConfig(name="echo", llm=second_llm))
    assert registry.get("echo").llm is second_llm


def test_unregister():
    registry = PipelineRegistry()
    registry.register(PipelineConfig(name="echo", llm=_StubLLM()))
    assert registry.unregister("echo") is True
    assert registry.get("echo") is None
    assert registry.unregister("echo") is False


def test_list_returns_all_registered():
    registry = PipelineRegistry()
    registry.register(PipelineConfig(name="a", llm=_StubLLM()))
    registry.register(PipelineConfig(name="b", llm=_StubLLM()))
    names = {config.name for config in registry.list()}
    assert names == {"a", "b"}


def test_pipeline_config_defaults_for_new_fields():
    config = PipelineConfig(name="x", llm=_StubLLM())
    assert config.tts is None
    assert config.voice is None
    assert config.instructions is None
    assert config.temperature == 0.8
    assert config.modalities_output == ["text"]


def test_pipeline_config_with_tts_reports_audio_output():
    class _StubTTS:
        pass

    config = PipelineConfig(name="x", llm=_StubLLM(), tts=_StubTTS())
    assert config.modalities_output == ["text", "audio"]


def test_pipeline_config_defaults_for_vad_stt():
    config = PipelineConfig(name="x", llm=_StubLLM())
    assert config.vad is None
    assert config.stt is None
    assert config.modalities_input == ["text"]


def test_pipeline_config_with_stt_reports_audio_input():
    class _StubSTT:
        pass

    config = PipelineConfig(name="x", llm=_StubLLM(), stt=_StubSTT())
    assert config.modalities_input == ["text", "audio"]


def test_pipeline_config_llm_defaults_to_none():
    config = PipelineConfig(name="x")
    assert config.llm is None


def test_pipeline_config_works_with_only_vad_and_stt():
    class _StubVAD:
        pass

    class _StubSTT:
        pass

    config = PipelineConfig(name="transcription-only", vad=_StubVAD(), stt=_StubSTT())
    assert config.llm is None
    assert config.modalities_input == ["text", "audio"]


def test_concurrent_register_and_get_does_not_raise():
    # Smoke test for the cross-thread registration contract (spec section
    # 6.8): provider nodes register from ComfyUI's executor thread while
    # sessions read from the event-loop thread.
    registry = PipelineRegistry()
    errors = []

    def writer():
        try:
            for i in range(200):
                registry.register(PipelineConfig(name=f"p{i % 5}", llm=_StubLLM()))
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            for i in range(200):
                registry.get(f"p{i % 5}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer) for _ in range(3)] + [
        threading.Thread(target=reader) for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_pipeline_config_registered_at_is_set_automatically():
    before = _time_module.time()
    config = PipelineConfig(name="x", llm=_StubLLM())
    after = _time_module.time()
    assert before <= config.registered_at <= after


class _UnloadTrackingLLM:
    def __init__(self) -> None:
        self.unloaded = False

    async def generate(self, messages, options):
        yield  # pragma: no cover -- never actually called in these tests


def test_unregister_does_not_unload_provider_while_a_session_is_active():
    from comfyui_realtime.engine.session_state import SessionState
    from comfyui_realtime.session_registry import session_registry

    provider = _UnloadTrackingLLM()
    provider.unload = lambda: setattr(provider, "unloaded", True)
    registry = PipelineRegistry()
    registry.register(PipelineConfig(name="x", llm=provider))

    session_registry.register("fake-active-session", "irrelevant", SessionState())
    try:
        registry.unregister("x")
        assert provider.unloaded is False, "must not unload while any session is active, regardless of which pipeline it uses"
    finally:
        session_registry.unregister("fake-active-session")


def test_unregister_unloads_provider_once_no_sessions_remain():
    provider = _UnloadTrackingLLM()
    provider.unload = lambda: setattr(provider, "unloaded", True)
    registry = PipelineRegistry()
    registry.register(PipelineConfig(name="x", llm=provider))
    registry.unregister("x")
    assert provider.unloaded is True


def test_unregister_does_not_unload_provider_still_used_by_another_pipeline():
    shared = _UnloadTrackingLLM()
    shared.unload = lambda: setattr(shared, "unloaded", True)
    registry = PipelineRegistry()
    registry.register(PipelineConfig(name="a", llm=shared))
    registry.register(PipelineConfig(name="b", llm=shared))
    registry.unregister("a")
    assert shared.unloaded is False, "still referenced by pipeline 'b' -- must not unload"
    registry.unregister("b")
    assert shared.unloaded is True


def test_register_replacing_a_pipeline_unloads_the_old_providers_orphan():
    old_provider = _UnloadTrackingLLM()
    old_provider.unload = lambda: setattr(old_provider, "unloaded", True)
    new_provider = _UnloadTrackingLLM()
    registry = PipelineRegistry()
    registry.register(PipelineConfig(name="x", llm=old_provider))
    registry.register(PipelineConfig(name="x", llm=new_provider))  # replaces
    assert old_provider.unloaded is True


def test_unregister_does_not_crash_on_a_provider_with_no_unload_method():
    registry = PipelineRegistry()
    registry.register(PipelineConfig(name="x", llm=_UnloadTrackingLLM()))  # no .unload() override -- plain instance
    registry.unregister("x")  # must not raise, even though _UnloadTrackingLLM has no real unload()
