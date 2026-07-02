"""Builds a tiny, real (not mocked) transformers causal-LM + tokenizer +
chat_template entirely locally -- no network calls, no real checkpoint
download. Used by test_transformers_llm_provider.py and
test_transformers_llm_provider_node.py as a fast, offline stand-in for a
real HF model directory (this repo's testing convention for the other two
LLM/TTS providers requires a real downloaded model for their tests; this
provider's format is small enough to construct genuinely from scratch,
so no download is needed at all)."""
from __future__ import annotations

import json
import pathlib

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedTokenizerFast

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant:{% endif %}"
)

_VOCAB = {
    "[UNK]": 0,
    "[PAD]": 1,
    "[BOS]": 2,
    "[EOS]": 3,
    "hello": 4,
    "world": 5,
    "system": 6,
    "user": 7,
    "assistant": 8,
    ":": 9,
    "say": 10,
    "hi": 11,
    "banana": 12,
    "fruit": 13,
    "favorite": 14,
    "your": 15,
    "what": 16,
    "is": 17,
    "one": 18,
    "word": 19,
    "in": 20,
    "exactly": 21,
}


def build_tiny_transformers_model_dir(
    target_dir: pathlib.Path,
    *,
    include_chat_template: bool = True,
    fp8_quantization_config: dict | None = None,
) -> None:
    """Save a tiny 2-layer GPT-2-architecture model + matching tokenizer to
    target_dir via save_pretrained(), suitable for AutoModelForCausalLM/
    AutoTokenizer.from_pretrained() to load back. include_chat_template=False
    produces a directory with no chat_template, for testing the
    "chat template required" construction error. fp8_quantization_config, if
    given, is written directly into the saved config.json's
    "quantization_config" key after save_pretrained() -- this produces a
    directory that LOOKS like a pre-quantized checkpoint from
    AutoModelForCausalLM.from_pretrained()'s perspective (config.json says
    quantized) without the tiny model's weights actually having been
    through any real quantization process. This is deliberately checking
    whether transformers' auto-detection of a checkpoint's own
    quantization_config actually fires -- not whether FP8 compute itself
    produces correct numerical results (that needs real H100-class hardware,
    out of scope for this fixture)."""
    tokenizer_backend = Tokenizer(WordLevel(vocab=_VOCAB, unk_token="[UNK]"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    if include_chat_template:
        tokenizer.chat_template = CHAT_TEMPLATE

    config = GPT2Config(
        vocab_size=len(_VOCAB),
        n_positions=64,
        n_embd=16,
        n_layer=2,
        n_head=2,
        bos_token_id=2,
        eos_token_id=3,
    )
    model = AutoModelForCausalLM.from_config(config)

    target_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(target_dir)
    model.save_pretrained(target_dir)

    if fp8_quantization_config is not None:
        config_path = target_dir / "config.json"
        config_dict = json.loads(config_path.read_text())
        config_dict["quantization_config"] = fp8_quantization_config
        config_path.write_text(json.dumps(config_dict))
