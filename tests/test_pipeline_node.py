import logging

import pytest

from comfyui_realtime.nodes.pipeline_node import EchoLLMProviderNode, RealtimePipelineNode
from comfyui_realtime.providers.base import ChatMessage, GenerationDelta, GenerationOptions
from comfyui_realtime.registry import pipeline_registry


@pytest.fixture(autouse=True)
def clean_registry():
    yield
    for config in list(pipeline_registry.list()):
        pipeline_registry.unregister(config.name)


def test_fixed_string_llm_has_a_noop_unload():
    (llm,) = EchoLLMProviderNode.execute(fixed_response="hi").result
    llm.unload()  # must not raise


async def test_echo_llm_provider_node_returns_fixed_string_regardless_of_input():
    (llm,) = EchoLLMProviderNode.execute(fixed_response="You said: ").result
    options = GenerationOptions()

    for content in ("anything", "something else"):
        messages = [ChatMessage(role="user", content=content)]
        deltas = [delta async for delta in llm.generate(messages, options)]
        assert deltas == [
            GenerationDelta(text="You said: ", finished=False),
            GenerationDelta(text="", finished=True),
        ]


def test_realtime_pipeline_node_registers_pipeline():
    (llm,) = EchoLLMProviderNode.execute(fixed_response="hi").result

    result = RealtimePipelineNode.execute(pipeline_name="echo", llm=llm)

    assert "registered" in result.ui["status"][0]
    assert pipeline_registry.get("echo") is not None
    assert pipeline_registry.get("echo").llm is llm


def test_realtime_pipeline_node_passes_through_optional_fields():
    (llm,) = EchoLLMProviderNode.execute(fixed_response="hi").result

    RealtimePipelineNode.execute(
        pipeline_name="echo-with-options",
        llm=llm,
        voice="lessac-medium",
        instructions="Be terse.",
        temperature=0.5,
    )

    config = pipeline_registry.get("echo-with-options")
    assert config.voice == "lessac-medium"
    assert config.instructions == "Be terse."
    assert config.temperature == 0.5


def test_realtime_pipeline_node_passes_through_vad_stt():
    (llm,) = EchoLLMProviderNode.execute(fixed_response="hi").result

    class _StubVAD:
        pass

    class _StubSTT:
        pass

    vad, stt = _StubVAD(), _StubSTT()
    RealtimePipelineNode.execute(pipeline_name="echo-with-audio-in", llm=llm, vad=vad, stt=stt)

    config = pipeline_registry.get("echo-with-audio-in")
    assert config.vad is vad
    assert config.stt is stt
    assert config.modalities_input == ["text", "audio"]


def test_realtime_pipeline_node_register_without_llm():
    RealtimePipelineNode.execute(pipeline_name="transcription-only-node-test", vad=object(), stt=object())
    config = pipeline_registry.get("transcription-only-node-test")
    assert config.llm is None
    pipeline_registry.unregister("transcription-only-node-test")


def test_register_rejects_completely_empty_pipeline():
    with pytest.raises(ValueError, match="no providers configured"):
        RealtimePipelineNode.execute(pipeline_name="empty-pipeline-test")


def test_register_warns_on_vad_without_stt(caplog):
    try:
        with caplog.at_level(logging.WARNING):
            RealtimePipelineNode.execute(pipeline_name="vad-without-stt-test", vad=object())
        assert "vad" in caplog.text.lower()
        assert pipeline_registry.get("vad-without-stt-test") is not None
    finally:
        pipeline_registry.unregister("vad-without-stt-test")


def test_register_does_not_warn_when_vad_and_stt_both_present(caplog):
    try:
        with caplog.at_level(logging.WARNING):
            RealtimePipelineNode.execute(pipeline_name="vad-and-stt-test", vad=object(), stt=object())
        assert caplog.text == ""
    finally:
        pipeline_registry.unregister("vad-and-stt-test")


def test_two_pipelines_can_share_one_llm_provider_instance():
    (shared_llm,) = EchoLLMProviderNode.execute(fixed_response="ok").result
    RealtimePipelineNode.execute(
        pipeline_name="character-a-shared-test", llm=shared_llm, instructions="You are a pirate."
    )
    RealtimePipelineNode.execute(
        pipeline_name="character-b-shared-test", llm=shared_llm, instructions="You are a robot."
    )
    try:
        config_a = pipeline_registry.get("character-a-shared-test")
        config_b = pipeline_registry.get("character-b-shared-test")
        # The literal same provider object, registered under two pipeline
        # names -- this is the "shared instance pattern" open question 5
        # worried would need new code. It doesn't; ComfyUI's own
        # node-output caching plus per-pipeline instructions (not baked
        # into the provider at load time) already gives this for free.
        assert config_a.llm is config_b.llm
        assert config_a.instructions == "You are a pirate."
        assert config_b.instructions == "You are a robot."
    finally:
        pipeline_registry.unregister("character-a-shared-test")
        pipeline_registry.unregister("character-b-shared-test")


def test_realtime_pipeline_node_fingerprint_inputs_always_differs():
    # spec section 5.4 caching gotcha: fingerprint_inputs (renamed from
    # IS_CHANGED in ComfyUI's v3 node API) must return a never-equal value
    # so the node re-runs registration even when ComfyUI's cache would
    # otherwise skip it because inputs are unchanged.
    first = RealtimePipelineNode.fingerprint_inputs(pipeline_name="echo", llm=None)
    second = RealtimePipelineNode.fingerprint_inputs(pipeline_name="echo", llm=None)
    assert first != first  # NaN is never equal to itself
    assert second != second
