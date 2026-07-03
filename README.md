# ComfyUI-Realtime

An OpenAI Realtime API-compatible WebSocket endpoint, implemented as a ComfyUI custom node. Wire up provider nodes (LLM, VAD, STT, TTS) in a ComfyUI workflow, register a named pipeline, and any client speaking the OpenAI Realtime protocol -- the official SDK, or a custom client -- can connect to it over WebSocket for real-time voice/text conversation. 

A bundled "Realtime" sidebar tab inside ComfyUI itself doubles as the reference client implementation: it talks to this project exclusively through the same public REST and WebSocket API any third-party client uses, with no privileged access of its own (see "Reference client" below).

**Status:** Core feature set is complete -- full speech-to-speech with barge-in, pipeline modularity, the REST API surface, and reference-counted provider lifecycle management.

## Requirements

- ComfyUI installed and runnable locally.
- Python 3.10+ (this package is tested on 3.13).
- At minimum one provider per pipeline stage you want to use -- see "Models" below for what to download.

## Installation

**Via ComfyUI-Manager:** Install through the manager default s2s providers automatically.

**Manual:**

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/uttermyth/ComfyUI-Realtime comfyui-realtime
cd comfyui-realtime
pip install ".[default]"
```

The `[default]` extra installs everything needed for a full speech-to-speech pipeline: Silero VAD, Whisper.cpp STT, llama.cpp LLM, Pocket TTS, and the audio resampler.

**Picking specific providers:**

Install individual dependencies instead of or alongside `[default]`:

| Dependecy | Provider node | Notes |
|---|---|---|
| `silero` | `SileroVADProviderNode` | Included in `[default]`. Bundles its own model. |
| `whisper-cpp` | `WhisperCppSTTProviderNode` | Included in `[default]` |
| `llama-cpp` | `LlamaCppLLMProviderNode` | Included in `[default]` |
| `resample` | *(engine-internal)* | Included in `[default]`. Required for sample-rate conversion. |
| `faster-whisper` | `FasterWhisperSTTProviderNode` | Alternative STT — lighter on some platforms |
| `pocket-tts` | `PocketTTSProviderNode` | Included in `[default]` |
| `piper` | `PiperTTSProviderNode` |  GPL-3.0 licensed |
| `transformers` | `TransformersLLMProviderNode` | Alternative LLM provider — loads HuggingFace transformers-format safetensors models instead of GGUF. Not included in `[default]` (pulls in `torch`). |
| `vllm` | `VLLMProviderNode` | Alternative LLM provider — loads the same HuggingFace transformers-format directory as `TransformersLLMProviderNode`, but via vLLM's `AsyncLLMEngine` for quantized-format coverage (NVFP4/modelopt, FP8, AWQ, GPTQ — auto-detected from the checkpoint) and continuous-batching performance. Not included in `[default]` (pulls in `vllm`/`torch`, requires a CUDA GPU). |

To install individual dependencies:

```bash
pip install ".[faster-whisper]"   # drop-in STT alternative to whisper-cpp
pip install ".[piper]"            # TTS (GPL-3.0)
pip install ".[transformers]"     # LLM alternative to llama.cpp -- HF safetensors models
pip install ".[vllm]"             # LLM alternative via vLLM -- same HF safetensors models, GPU-only, quantized-format support
```

Restart ComfyUI after installing or changing anything in this package.

## Models

Every model/voice file this project uses is gitignored -- none of them ship in this repo. Download what each provider you want to use needs:

- **LLM** (`LlamaCppLLMProviderNode`): any GGUF-format model compatible with `llama-cpp-python`. Place in `models/llm/`.
- **LLM** (`TransformersLLMProviderNode`): a local HuggingFace transformers-format chat model directory (`config.json` + `.safetensors` weights + tokenizer files) with a tokenizer chat template. Place the whole model directory in `models/llm/transformers/<model-dir>/`. FP8-quantized checkpoints (`quant_method: "fp8"`) are supported automatically via the checkpoint's own `config.json`.
- **LLM** (`VLLMProviderNode`): the same local HuggingFace transformers-format model directory as `TransformersLLMProviderNode` — place it in `models/llm/transformers/<model-dir>/`. Requires a CUDA GPU. Quantized checkpoints (NVFP4/`modelopt`, FP8, AWQ, GPTQ) are supported automatically via the checkpoint's own `config.json`.
- **STT** (`WhisperCppSTTProviderNode`): a whisper.cpp `ggml-*.bin` model. Place in `models/stt/whisper_cpp/`.
- **STT** (`FasterWhisperSTTProviderNode`): a CTranslate2 model directory. Place in `models/stt/faster_whisper/`.
- **TTS** (`PiperTTSProviderNode`): a Piper voice (`.onnx` + matching `.onnx.json` config). Place in `assets/piper_voices/` or `models/tts/piper_voices/`.
- **TTS** (`PocketTTSProviderNode`): run `python scripts/fetch_pocket_tts_model.py <language>` once per language to fetch `model_without_voice_cloning.safetensors` and `tokenizer.model` into `models/tts/pocket_tts/<language>/` or download them manually. Voices are a local reference audio clip (`.wav`/`.mp3`, ~5-30s of clean speech; raw-audio voice cloning specifically requires the gated `kyutai/pocket-tts` weights -- accept terms at https://huggingface.co/kyutai/pocket-tts and download manually while logged in) or a pre-exported `.safetensors` voice embedding (works with the default, non-gated weights). Place voice files in `assets/pocket_tts_voices/` or `models/tts/pocket_tts_voices/`.
- **VAD** (`SileroVADProviderNode`): no download needed -- bundles its own model.

## Configuration: building a pipeline

In ComfyUI's graph editor, wire provider nodes into a `RealtimePipelineNode`:

- `LlamaCppLLMProviderNode` -> `llm` input
- `TransformersLLMProviderNode` -> `llm` input
- `VLLMProviderNode` -> `llm` input
- `SileroVADProviderNode` -> `vad` input
- `WhisperCppSTTProviderNode` -> `stt` input
- `PocketTTSProviderNode` -> `tts` input

All four are optional -- the pipeline shape you get depends on which ones you connect (see "Example workflows" below). `RealtimePipelineNode` itself takes a `pipeline_name` (what clients connect with via `?model=<name>` on the WebSocket URL), plus `voice`, `instructions`, and `temperature`.

Queue the workflow once to register the pipeline. Re-queuing the same `pipeline_name` replaces the registration; deleting it (via `DELETE /realtime/pipelines/{name}`, see "REST API" below) or queuing a different `pipeline_name` makes a different pipeline.

Connect a client to `ws://<comfyui-host>:<port>/v1/realtime?model=<pipeline_name>`.

## Example workflows

`workflows/` has one example for each pipeline shape this project supports. `workflows/api/` contains prompt-API payloads for use with `curl`; `workflows/ui/` contains graph format for loading directly in the ComfyUI UI. Each uses placeholder model names (`your-llm.gguf`, `your-whisper-model.bin`) -- replace these with the actual filenames of models you've placed in the appropriate `models/` subdirectory.

| File | Shape | Providers |
|---|---|---|
| `full-speech-to-speech.json` | Full speech-to-speech | VAD + STT + LLM + TTS |
| `text-to-speech.json` | Text-to-speech | LLM + TTS |
| `speech-to-text.json` | Speech-to-text | VAD + STT + LLM |
| `text-to-text.json` | Text-to-text | LLM only |
| `continuous-transcription.json` | Continuous transcription (live captioning, no response generation) | VAD + STT only |
| `shared-multi-voice.json` | One shared pipeline with multiple TTS voices, demonstrating per-session voice selection via `session.update` | LLM + TTS (3 voices) |

Queue one with ComfyUI's `/prompt` API, e.g.:

```bash
curl -X POST http://127.0.0.1:8188/prompt -H "Content-Type: application/json" -d @workflows/api/full-speech-to-speech.json
```
Alternatively, you can import the ui compatible workflows via the ComfyUI web interface.

## Reference client

A "Realtime" sidebar tab ships bundled with this custom node (via ComfyUI's `WEB_DIRECTORY` mechanism) -- open it in ComfyUI's UI to converse with any registered pipeline without installing anything separately. It's also the reference implementation of how to actually use this project's API: every request it makes is `GET /realtime/pipelines` or the `/v1/realtime` WebSocket, the exact same public surface documented below, nothing more.

Open the tab, pick a registered pipeline from the dropdown, and connect. The UI adapts to whatever the pipeline actually supports -- a microphone control and live transcript for pipelines with audio input, a text box for text-only ones, and a response area that only appears once the pipeline actually generates a response (some shapes, like continuous transcription, never will by design). When a connected pipeline has more than one TTS voice loaded, the client displays a voice selector dropdown; selecting a voice sends `session.update` over the WebSocket.

Source lives in `ui/` (React + TypeScript + Vite); the built bundle in `dist/` is committed, so cloning this repo and restarting ComfyUI is enough to use it -- no Node.js toolchain required unless you're modifying the client itself. If you are:

```bash
cd ui
npm install
npm run build    # one-shot build into ../dist
npm run watch     # rebuild on change while developing
npm test          # run the client's own test suite
```

## REST API

- `GET /realtime/health` -- status, registered pipeline count, active session count.
- `GET /realtime/pipelines` -- list all registered pipelines (name, modalities, providers, registration time, voices).
- `GET /realtime/pipelines/{name}` -- details for one pipeline.
- `DELETE /realtime/pipelines/{name}` -- unregister a pipeline.
- `GET /realtime/sessions` -- list active WebSocket sessions (id, pipeline, uptime, state).
- `GET /v1/models` -- OpenAI-compatible model list; each registered pipeline appears as a model entry.

The `GET /realtime/pipelines` response includes a `voices` field for each pipeline: a list of voice IDs the pipeline's TTS provider has loaded (empty if the pipeline has no TTS provider).

`/v1/models` can be disabled (e.g. to avoid colliding with another custom node's route) by setting `COMFYUI_REALTIME_DISABLE_MODELS_ROUTE=1` before starting ComfyUI.

## Testing

Python backend:

```bash
pip install -e ".[dev]"
pytest tests/ -m "not integration"
```

Tests marked `integration` require a live ComfyUI server with real models loaded -- see `tests/conftest.py` for the environment variables they read (`COMFYUI_ROOT`, `COMFYUI_URL`).

Reference client (`ui/`):

```bash
cd ui
npm install
npm test
```

## License

MIT -- see `LICENSE`.
