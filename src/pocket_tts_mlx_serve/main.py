"""FastAPI server and CLI for pocket-tts-mlx.

Mirrors the ``POST /tts`` endpoint from ``pocket-tts serve`` while running the
MLX backend on Apple Silicon.
"""

import argparse
import logging
import os
import tempfile
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Queue
from typing import Any

import uvicorn
from anyio.to_thread import run_sync
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pocket_tts_mlx import TTSModel
from pocket_tts_mlx.default_parameters import DEFAULT_AUDIO_PROMPT
from pocket_tts_mlx.utils.weight_conversion import PREDEFINED_VOICES
from typing_extensions import Annotated

from pocket_tts_mlx_serve.audio_streaming import stream_audio_chunks_to_queue

logger = logging.getLogger(__name__)

# Model configuration set by ``serve`` and consumed by the generation worker.
_config: str | None = None
# Default voice used when neither ``voice_url`` nor ``voice_wav`` is provided.
default_voice: str = DEFAULT_AUDIO_PROMPT
# Queue used to submit jobs to the dedicated MLX generation thread.
_task_queue: Queue | None = None


class _JobFailed:
    """Marker placed on a response queue when a generation job fails."""

    def __init__(self, message: str):
        self.message = message


def _is_builtin_or_remote_voice(voice_url: str) -> bool:
    return (
        voice_url.startswith("http://")
        or voice_url.startswith("https://")
        or voice_url.startswith("hf://")
        or voice_url in PREDEFINED_VOICES
    )


def _generation_worker(ready_event: threading.Event) -> None:
    """Dedicated thread that owns the MLX model and processes generation jobs."""
    logger.info("Loading pocket-tts-mlx model in generation worker...")
    model = TTSModel.load_model(config=_config) if _config else TTSModel.load_model()
    logger.info("Model loaded in worker.")

    # Small voice-state cache, matching the upstream serve behaviour.
    voice_cache: OrderedDict[str, dict] = OrderedDict()
    voice_cache_max_size = 2

    ready_event.set()

    assert _task_queue is not None

    while True:
        item = _task_queue.get()
        if item is None:
            break

        response_queue, text, voice_kind, voice_value, max_tokens, frames_after_eos = item
        try:
            if voice_kind == "url":
                if voice_value in voice_cache:
                    model_state = voice_cache[voice_value]
                    voice_cache.move_to_end(voice_value)
                else:
                    model_state = model.get_state_for_audio_prompt(voice_value)
                    voice_cache[voice_value] = model_state
                    if len(voice_cache) > voice_cache_max_size:
                        voice_cache.popitem(last=False)
            else:
                model_state = model.get_state_for_audio_prompt(Path(voice_value), truncate=True)
        except Exception as exc:
            logger.exception("Failed to load voice state")
            response_queue.put(_JobFailed(str(exc)))
            response_queue.put(None)
            continue

        try:
            audio_chunks = model.generate_audio_stream(
                model_state=model_state,
                text_to_generate=text,
                max_tokens=max_tokens,
                frames_after_eos=frames_after_eos,
            )
            stream_audio_chunks_to_queue(response_queue, audio_chunks, model.sample_rate)
        except Exception:
            logger.exception("TTS generation failed")
        finally:
            response_queue.put(None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the MLX generation worker on startup and shut it down cleanly."""
    global _task_queue
    _task_queue = Queue()
    ready_event = threading.Event()
    worker = threading.Thread(
        target=_generation_worker,
        args=(ready_event,),
        daemon=True,
        name="mlx-generation-worker",
    )
    worker.start()
    await run_sync(ready_event.wait)
    logger.info("Server ready; MLX generation worker is running.")

    try:
        yield
    finally:
        _task_queue.put(None)
        worker.join(timeout=10)
        logger.info("MLX generation worker stopped.")


web_app = FastAPI(
    title="pocket-tts-mlx-serve",
    description="FastAPI server layer for pocket-tts-mlx.",
    version="0.1.0",
    lifespan=lifespan,
)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/", response_class=HTMLResponse)
async def root() -> str:
    """Serve a minimal landing page."""
    return f"""<!doctype html>
<html>
<head><title>pocket-tts-mlx-serve</title></head>
<body>
<h1>pocket-tts-mlx-serve</h1>
<p>FastAPI server layer for <a href="https://github.com/andyrightnow/pocket-tts-mlx-serve">pocket-tts-mlx-serve</a>.</p>
<ul>
  <li><code>POST /tts</code> - generate speech (mirrors <code>pocket-tts serve</code>)</li>
  <li><code>GET /health</code> - health check</li>
  <li><a href="/docs">OpenAPI docs</a></li>
</ul>
<p>Default voice: <strong>{default_voice}</strong></p>
</body>
</html>
"""


@web_app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


def _iter_response_queue(response_queue: Queue) -> Any:
    """Yield WAV bytes from a response queue, raising on job failure."""
    first_item = response_queue.get()
    if isinstance(first_item, _JobFailed):
        # Drain the terminating None so the queue does not leak.
        response_queue.get()
        raise HTTPException(status_code=500, detail=first_item.message)
    if first_item is not None:
        yield first_item

    while True:
        data = response_queue.get()
        if data is None:
            break
        yield data


def _submit_job(
    text: str,
    voice_kind: str,
    voice_value: str,
    max_tokens: int,
    frames_after_eos: int | None,
) -> Any:
    """Submit a generation job and return a generator over the WAV bytes."""
    if _task_queue is None:
        raise HTTPException(status_code=503, detail="TTS model is not loaded")

    response_queue: Queue = Queue()
    _task_queue.put((response_queue, text, voice_kind, voice_value, max_tokens, frames_after_eos))
    return _iter_response_queue(response_queue)


def _start_streaming_response(iterator: Any) -> StreamingResponse:
    """Advance ``iterator`` once to validate the job before committing the response."""
    try:
        first_chunk = next(iterator)
    except StopIteration as exc:
        raise HTTPException(status_code=500, detail="No audio generated") from exc

    def _with_first_chunk():
        yield first_chunk
        yield from iterator

    return StreamingResponse(
        _with_first_chunk(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=generated_speech.wav",
            "Transfer-Encoding": "chunked",
        },
    )


@web_app.post("/tts")
def text_to_speech(
    text: Annotated[str, Form()],
    voice_url: Annotated[str | None, Form()] = None,
    voice_wav: Annotated[UploadFile | None, File()] = None,
    max_tokens: Annotated[int, Form()] = 50,
    frames_after_eos: Annotated[int | None, Form()] = None,
) -> StreamingResponse:
    """
    Generate speech from text.

    Args:
        text: Text to convert to speech.
        voice_url: Optional built-in voice name (e.g. ``alba``), or a remote URL
            starting with ``http://``, ``https://``, or ``hf://``.
        voice_wav: Optional uploaded voice file (mutually exclusive with ``voice_url``).
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if voice_url is None and voice_wav is None:
        voice_url = default_voice

    if voice_url is not None and voice_wav is not None:
        raise HTTPException(status_code=400, detail="Cannot provide both voice_url and voice_wav")

    if voice_url is not None:
        if not _is_builtin_or_remote_voice(voice_url):
            raise HTTPException(
                status_code=400,
                detail="voice_url must be a predefined voice or start with http://, https://, or hf://",
            )
        iterator = _submit_job(text, "url", voice_url, max_tokens, frames_after_eos)
        return _start_streaming_response(iterator)

    # voice_wav path
    assert voice_wav is not None
    suffix = Path(voice_wav.filename).suffix if voice_wav.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(voice_wav.file.read())
        temp_file.flush()
        temp_path = temp_file.name

    def _stream_file_response() -> Any:
        iterator = _submit_job(text, "file", temp_path, max_tokens, frames_after_eos)
        try:
            yield from iterator
        finally:
            os.unlink(temp_path)

    return _start_streaming_response(_stream_file_response())


def serve(
    host: str = "localhost",
    port: int = 8000,
    reload: bool = False,
    config: str | None = None,
    default_voice_arg: str | None = None,
) -> None:
    """Configure and start the FastAPI server."""
    global _config, default_voice

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _config = config
    if default_voice_arg is not None:
        default_voice = default_voice_arg

    uvicorn.run(
        "pocket_tts_mlx_serve.main:web_app",
        host=host,
        port=port,
        reload=reload,
    )


def serve_cli() -> None:
    """Entry-point for the ``pocket-tts-mlx-serve`` CLI."""
    parser = argparse.ArgumentParser(
        prog="pocket-tts-mlx-serve",
        description="Start a FastAPI server for pocket-tts-mlx.",
    )
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Model variant or path to a custom config YAML. "
            "Defaults to the pocket-tts-mlx default variant."
        ),
    )
    parser.add_argument(
        "--default-voice",
        default=DEFAULT_AUDIO_PROMPT,
        help="Voice used when no voice_url/voice_wav is supplied",
    )

    args = parser.parse_args()
    serve(
        host=args.host,
        port=args.port,
        reload=args.reload,
        config=args.config,
        default_voice_arg=args.default_voice,
    )


if __name__ == "__main__":
    serve_cli()
