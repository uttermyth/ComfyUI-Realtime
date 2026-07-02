"""No real checkpoint download needed -- see tests/_tiny_transformers_fixture.py.
Not marked pytest.mark.integration (unlike test_llama_cpp_llm_provider.py /
test_pocket_tts_provider.py) since this provider's fixture model is built
from scratch locally and runs fast even on CPU."""
import logging
import pathlib
import threading

import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")

import torch

from comfyui_realtime.providers.base import ChatMessage, GenerationOptions
from comfyui_realtime.providers.transformers_llm import TransformersLLMProvider, _StopEventCriteria
from tests._tiny_transformers_fixture import build_tiny_transformers_model_dir


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory) -> pathlib.Path:
    target_dir = tmp_path_factory.mktemp("tiny_transformers_model")
    build_tiny_transformers_model_dir(target_dir)
    return target_dir


@pytest.fixture(scope="module")
def tiny_model_dir_without_chat_template(tmp_path_factory) -> pathlib.Path:
    target_dir = tmp_path_factory.mktemp("tiny_transformers_model_no_template")
    build_tiny_transformers_model_dir(target_dir, include_chat_template=False)
    return target_dir


def test_construction_succeeds_on_cpu(tiny_model_dir):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        assert hasattr(provider, "generate")
    finally:
        provider.unload()


def test_construction_raises_when_chat_template_missing(tiny_model_dir_without_chat_template):
    with pytest.raises(ValueError, match="chat_template"):
        TransformersLLMProvider(model_path=str(tiny_model_dir_without_chat_template), device="cpu")


def test_construction_raises_for_unavailable_cuda_device(tiny_model_dir, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda"):
        TransformersLLMProvider(model_path=str(tiny_model_dir), device="cuda")


def test_construction_raises_for_unavailable_mps_device(tiny_model_dir, monkeypatch):
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="mps"):
        TransformersLLMProvider(model_path=str(tiny_model_dir), device="mps")


def test_auto_dtype_resolves_to_float32_on_cpu(tiny_model_dir):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu", torch_dtype="auto")
    try:
        assert next(provider._model.parameters()).dtype == torch.float32
    finally:
        provider.unload()


def test_explicit_dtype_is_respected(tiny_model_dir):
    provider = TransformersLLMProvider(
        model_path=str(tiny_model_dir), device="cpu", torch_dtype="bfloat16"
    )
    try:
        assert next(provider._model.parameters()).dtype == torch.bfloat16
    finally:
        provider.unload()


def test_trust_remote_code_defaults_to_false(tiny_model_dir):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        assert provider._trust_remote_code is False
    finally:
        provider.unload()


def test_unload_releases_the_model(tiny_model_dir):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    provider.unload()
    assert provider._model is None


def test_stop_event_criteria_returns_false_until_event_is_set():
    event = threading.Event()
    criteria = _StopEventCriteria(event)
    assert criteria(None, None) is False
    event.set()
    assert criteria(None, None) is True


async def test_generate_yields_text_deltas_then_a_finished_delta(tiny_model_dir):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        messages = [ChatMessage(role="user", content="hello world")]
        deltas = []
        async for delta in provider.generate(messages, GenerationOptions(max_tokens=8)):
            deltas.append(delta)
        assert len(deltas) >= 1
        assert deltas[-1].finished is True
        assert deltas[-1].text == ""
        assert all(d.finished is False for d in deltas[:-1])
    finally:
        provider.unload()


async def test_system_prompt_is_included_in_the_formatted_prompt(tiny_model_dir, monkeypatch):
    provider = TransformersLLMProvider(
        model_path=str(tiny_model_dir), device="cpu", system_prompt="banana"
    )
    try:
        captured = {}
        original_apply = provider._tokenizer.apply_chat_template

        def spy_apply_chat_template(messages, **kwargs):
            captured["messages"] = messages
            return original_apply(messages, **kwargs)

        monkeypatch.setattr(provider._tokenizer, "apply_chat_template", spy_apply_chat_template)

        async for _ in provider.generate(
            [ChatMessage(role="user", content="hi")], GenerationOptions(max_tokens=4)
        ):
            pass

        assert captured["messages"][0] == {"role": "system", "content": "banana"}
    finally:
        provider.unload()


async def test_temperature_zero_uses_greedy_decoding(tiny_model_dir, monkeypatch):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        captured_kwargs = {}
        original_generate = provider._model.generate

        def spy_generate(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return original_generate(*args, **kwargs)

        monkeypatch.setattr(provider._model, "generate", spy_generate)

        async for _ in provider.generate(
            [ChatMessage(role="user", content="hi")], GenerationOptions(max_tokens=4, temperature=0.0)
        ):
            pass

        assert captured_kwargs["do_sample"] is False
        assert "temperature" not in captured_kwargs
    finally:
        provider.unload()


async def test_positive_temperature_uses_sampling(tiny_model_dir, monkeypatch):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        captured_kwargs = {}
        original_generate = provider._model.generate

        def spy_generate(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return original_generate(*args, **kwargs)

        monkeypatch.setattr(provider._model, "generate", spy_generate)

        async for _ in provider.generate(
            [ChatMessage(role="user", content="hi")], GenerationOptions(max_tokens=4, temperature=0.7)
        ):
            pass

        assert captured_kwargs["do_sample"] is True
        assert captured_kwargs["temperature"] == 0.7
    finally:
        provider.unload()


async def test_default_max_new_tokens_is_512_when_unspecified(tiny_model_dir, monkeypatch):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        captured_kwargs = {}
        original_generate = provider._model.generate

        def spy_generate(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return original_generate(*args, **kwargs)

        monkeypatch.setattr(provider._model, "generate", spy_generate)

        async for _ in provider.generate(
            [ChatMessage(role="user", content="hi")], GenerationOptions(max_tokens=None)
        ):
            pass

        assert captured_kwargs["max_new_tokens"] == 512
    finally:
        provider.unload()


async def test_abandoning_generation_stops_the_inner_generate_thread(tiny_model_dir):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        gen = provider.generate(
            [ChatMessage(role="user", content="hello world")], GenerationOptions(max_tokens=64)
        )
        await gen.__anext__()
        await gen.aclose()

        inner_threads = [
            t for t in threading.enumerate() if t.name == "transformers-generate-inner" and t.is_alive()
        ]
        assert inner_threads == [], "inner model.generate() thread is still alive after aclose()"
    finally:
        provider.unload()


async def test_abandoning_generation_releases_the_lock_promptly(tiny_model_dir):
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        first_gen = provider.generate(
            [ChatMessage(role="user", content="hello world")], GenerationOptions(max_tokens=64)
        )
        await first_gen.__anext__()
        await first_gen.aclose()

        second_deltas = []
        async for delta in provider.generate(
            [ChatMessage(role="user", content="hi")], GenerationOptions(max_tokens=4)
        ):
            second_deltas.append(delta)
        assert len(second_deltas) >= 1
    finally:
        provider.unload()


@pytest.mark.timeout(10)
async def test_exception_inside_generate_propagates_instead_of_hanging(tiny_model_dir, monkeypatch):
    # Regression test: threading.Thread silently swallows exceptions raised
    # in its target. Without generate()'s own capture-and-reraise, this
    # would hang forever (the streamer never gets its terminating sentinel)
    # instead of raising -- see Task 3's "Update" note above. @pytest.mark.timeout(10)
    # (from pytest-timeout, added in Step 1 above) makes a regression back to
    # that hang fail this one test after 10s instead of hanging the whole
    # test run indefinitely.
    provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        def broken_generate(*args, **kwargs):
            raise RuntimeError("simulated generation failure")

        monkeypatch.setattr(provider._model, "generate", broken_generate)

        with pytest.raises(RuntimeError, match="simulated generation failure"):
            async for _ in provider.generate(
                [ChatMessage(role="user", content="hi")], GenerationOptions(max_tokens=4)
            ):
                pass
    finally:
        provider.unload()


@pytest.mark.timeout(60)
def test_fp8_quantization_config_in_checkpoint_is_auto_detected(tmp_path_factory, caplog):
    """Confirms the design doc's core claim for the FP8 addendum: that
    AutoModelForCausalLM.from_pretrained() auto-detects a checkpoint's own
    baked-in quantization_config with zero extra kwargs from
    TransformersLLMProvider's own code. weight_block_size is set to (16, 16)
    to match this tiny fixture's n_embd=16 -- the real-world default is
    (128, 128), which this tiny model's dimensions can't satisfy.

    ACTUAL OUTCOME OBSERVED (recorded here for Task 2, on this machine:
    transformers 5.12.1, torch 2.12.1, CPU-only -- no CUDA, no XPU, no
    triton, no compressed_tensors; accelerate IS installed):

    from_pretrained() does NOT raise. It logs a warning via transformers'
    own logger --

        "Using FP8 quantized models requires a GPU or XPU, we will default
        to dequantizing the model to bf16 since no GPU or XPU is available"

    -- and then dequantizes on the fly. This is the FineGrainedFP8HfQuantizer
    (transformers/quantizers/quantizer_finegrained_fp8.py)'s validate_environment():
    for a checkpoint that is *already* pre_quantized (our case -- the
    quantization_config comes from the checkpoint's own config.json, not
    from an explicit kwarg), a missing/inadequate GPU never raises -- it only
    warns once (logger.warning_once) and sets quantization_config.dequantize
    = True. Traced through transformers/quantizers/base.py: postprocess_model()
    then calls remove_quantization_config(), which does `del model.hf_quantizer`,
    `del model.config.quantization_config`, `del model.quantization_method`,
    and `model.is_quantized = False` -- by design, once a checkpoint has been
    fully dequantized back to plain floats there is no remaining quantization
    state to expose, so model.config.quantization_config being absent
    afterward is NOT a silent-ignore bug (design doc's stated risk): the
    checkpoint's quantization_config WAS read, recognized by name ("FP8"
    appears in the warning text), and deliberately acted on -- the action
    taken was a documented graceful CPU fallback, not silent pass-through.
    Confirmed this is deliberate by reading the library source (see file
    paths above), not by guessing from behavior alone.

    Practical implication for Task 2: on hardware with no CUDA/XPU (this
    dev machine, and most CI runners), loading an FP8-quantized checkpoint
    through this provider will NOT raise -- it will silently succeed as a
    dequantized (plain-float) model, merely emitting a transformers-internal
    log warning our provider does not currently surface anywhere. Task 2's
    error-wrapping code should NOT expect a catchable hardware exception for
    this path; if surfacing this to the user matters, it would need to
    inspect logs or model.is_quantized after load, not a try/except around
    from_pretrained(). The `RuntimeError("No GPU or XPU found. A GPU or XPU
    is needed for FP8 quantization.")` in the same source function only
    fires for the non-pre_quantized (on-the-fly quantization request) case,
    which this provider never triggers -- it always loads existing
    checkpoints. The triton / compressed-tensors dependency question (the
    plan's Outcome 2) could not be exercised on this CPU-only machine since
    that code path requires an actual CUDA/XPU device to be reached first;
    this remains unverified and would need real GPU hardware to confirm.
    """
    target_dir = tmp_path_factory.mktemp("tiny_fp8_model")
    build_tiny_transformers_model_dir(
        target_dir,
        fp8_quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [16, 16],
            "activation_scheme": "dynamic",
        },
    )

    from transformers import AutoModelForCausalLM

    with caplog.at_level(logging.WARNING):
        model = AutoModelForCausalLM.from_pretrained(str(target_dir))

    try:
        # Loading must not raise on this hardware (see docstring): confirm it
        # didn't silently produce a model indistinguishable from an
        # unquantized load by checking for the FP8-specific log warning that
        # proves transformers actually recognized and acted on the
        # checkpoint's quantization_config, rather than ignoring it outright.
        fp8_warnings = [
            record.message for record in caplog.records if "FP8" in record.message
        ]
        assert fp8_warnings, (
            "Expected transformers to log an FP8-specific warning when loading "
            "a pre-quantized FP8 checkpoint on hardware without CUDA/XPU (its "
            "documented graceful-dequantize fallback). No such warning was "
            "observed -- this means the checkpoint's quantization_config may "
            "have been silently ignored with zero indication, which would be "
            "the real bug the design doc's risk section warns about. If you "
            "see this failure, stop and report back rather than proceeding "
            "to Task 2."
        )
        assert "GPU or XPU" in fp8_warnings[0]

        # On this graceful-dequantize path, transformers deliberately removes
        # the quantization_config from the loaded model's config (see
        # docstring) -- so its absence here is expected, not a bug.
        assert getattr(model.config, "quantization_config", None) is None
        assert getattr(model, "is_quantized", None) is not True
    finally:
        del model


def test_warns_when_fp8_checkpoint_is_silently_dequantized(tmp_path_factory, caplog):
    """Companion to Task 1's test_fp8_quantization_config_in_checkpoint_is_auto_detected
    -- that test confirmed the underlying transformers behavior (silent
    dequantize-to-full-precision on hardware without a qualifying GPU/XPU,
    with only an internal transformers log line marking it). This test
    confirms TransformersLLMProvider itself surfaces a clear, provider-level
    warning about it, instead of relying on that easily-missed internal log
    line as the only trace."""
    target_dir = tmp_path_factory.mktemp("tiny_fp8_model_for_warning_test")
    build_tiny_transformers_model_dir(
        target_dir,
        fp8_quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [16, 16],
            "activation_scheme": "dynamic",
        },
    )

    with caplog.at_level(logging.WARNING, logger="comfyui_realtime"):
        provider = TransformersLLMProvider(model_path=str(target_dir), device="cpu")
    try:
        matching = [
            r for r in caplog.records
            if "quantization_config" in r.getMessage() and "dequantiz" in r.getMessage().lower()
        ]
        assert matching, (
            f"expected a warning naming quantization_config + dequantization, "
            f"got: {[r.getMessage() for r in caplog.records]}"
        )
    finally:
        provider.unload()


def test_no_warning_for_a_normal_non_quantized_model(tiny_model_dir, caplog):
    """Regression guard: the new warning must not fire for the ordinary,
    non-quantized fixture every other test in this file uses -- it should
    only fire when config.json actually declared quantization_config and
    the loaded model doesn't have it."""
    with caplog.at_level(logging.WARNING, logger="comfyui_realtime"):
        provider = TransformersLLMProvider(model_path=str(tiny_model_dir), device="cpu")
    try:
        matching = [r for r in caplog.records if "quantization_config" in r.getMessage()]
        assert matching == []
    finally:
        provider.unload()
