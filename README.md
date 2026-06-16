# pocket-tts-mlx-serve

A thin [FastAPI](https://fastapi.tiangolo.com/) server layer for
[pocket-tts-mlx-serve](https://github.com/andyrightnow/pocket-tts-mlx-serve).

It exposes the same `POST /tts` endpoint as the `pocket-tts serve` command
from [pocket-tts](https://github.com/kyutai-labs/pocket-tts) but runs the
MLX backend on Apple Silicon.

## Requirements

- Python 3.10+
- Apple Silicon Mac (M1/M2/M3/M4)
- [uv](https://docs.astral.sh/uv/)

## Install

```bash
uv sync
```

## Run

```bash
# Using uv
uv run pocket-tts-mlx-serve

# Or after installing the package
pocket-tts-mlx-serve
```

Options:

```bash
pocket-tts-mlx-serve --host 0.0.0.0 --port 8080 --default-voice marius
```

## API

### `POST /tts`

Generate speech from text. The response is a streaming `audio/wav` file.

**Form fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | yes | Text to synthesise |
| `voice_url` | string | no | Built-in voice name (`alba`, `marius`, ...) or remote URL (`http://`, `https://`, `hf://`) |
| `voice_wav` | file | no | Uploaded voice audio (mutually exclusive with `voice_url`) |
| `max_tokens` | int | no | Max tokens per chunk (default: `50`) |
| `frames_after_eos` | int | no | Frames to generate after end-of-sentence |

**Example**

```bash
curl -X POST http://localhost:8000/tts \
  -d "text=Hello from MLX!" \
  -d "voice_url=marius" \
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
