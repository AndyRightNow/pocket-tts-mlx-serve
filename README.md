# pocket-tts-mlx-serve

A thin [FastAPI](https://fastapi.tiangolo.com/) server layer for
[pocket-tts-mlx](https://github.com/jishnuvenugopal/pocket-tts-mlx).

It exposes the same `POST /tts` endpoint as `pocket-tts serve` from
[pocket-tts](https://github.com/kyutai-labs/pocket-tts) but runs the MLX
backend on Apple Silicon.

## Requirements

- Python 3.10+
- Apple Silicon Mac (M1/M2/M3/M4)
- [uv](https://docs.astral.sh/uv/)

## Install

```bash
uv sync
```

## Run

The quickest way to run the server is with [`uvx`](https://docs.astral.sh/uv/guides/tools/):

```bash
uvx --from git+https://github.com/AndyRightNow/pocket-tts-mlx-serve.git pocket-tts-mlx-serve --help
```

Or, after cloning the repository and installing dependencies with `uv sync`:

```bash
uv run pocket-tts-mlx-serve
```

Options:

```bash
pocket-tts-mlx-serve \
  --host 0.0.0.0 \
  --port 8080 \
  --default-voice https://huggingface.co/.../voice.wav \
  --normalize-voice \
  --voice-cache-size 4
```

| Flag | Description |
|------|-------------|
| `--host` | Host to bind to (default: `localhost`) |
| `--port` | Port to bind to (default: `8000`) |
| `--config` | Model variant or custom config YAML |
| `--default-voice` | Default voice URL when none is supplied |
| `--normalize-voice` | Normalize voice prompts before cloning |
| `--voice-cache-size` | Number of voice states to cache, `0` disables caching (default: `2`) |

## API

### `POST /tts`

Generate speech from text. The response is a streaming `audio/wav` file.

**Form fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | yes | Text to synthesise |
| `voice_url` | string | no | Remote audio URL (`http://`, `https://`, `hf://`) or `.safetensors` embedding URL |
| `voice_wav` | file | no | Uploaded voice audio (mutually exclusive with `voice_url`) |
| `max_tokens` | int | no | Max tokens per chunk (default: `50`) |
| `frames_after_eos` | int | no | Frames to generate after end-of-sentence |

**Examples**

Clone from a remote audio file:

```bash
curl -X POST http://localhost:8000/tts \
  -d "text=Hello from MLX!" \
  -d "voice_url=https://huggingface.co/.../voice.wav" \
  -o output.wav
```

Use a pre-computed `.safetensors` voice embedding:

```bash
curl -X POST http://localhost:8000/tts \
  -d "text=Hello from MLX!" \
  -d "voice_url=hf://kyutai/pocket-tts-without-voice-cloning/embeddings/alba.safetensors" \
  -o output.wav
```

Upload a voice file:

```bash
curl -X POST http://localhost:8000/tts \
  -F "text=Hello from MLX!" \
  -F "voice_wav=@voice.wav" \
  -o output.wav
```

### `GET /health`

Health check.

### `GET /`

Simple landing page with links.

## Development

```bash
uv run --reload pocket-tts-mlx-serve --reload
```

### Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check .
```

Auto-format with:

```bash
uv run ruff format .
uv run ruff check . --fix
```

### Pre-commit

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```
