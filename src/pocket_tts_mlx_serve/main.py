"""FastAPI server and CLI for pocket-tts-mlx.

Mirrors the ``POST /tts`` endpoint from ``pocket-tts serve`` while running the
MLX backend on Apple Silicon.
"""

import argparse
import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Queue
from typing import Any

import uvicorn
from anyio.to_thread import run_sync
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from typing_extensions import Annotated

from pocket_tts_mlx_serve.worker import _JobFailed, generation_worker, is_builtin_or_remote_voice

logger = logging.getLogger(__name__)

# Model configuration set by ``serve`` and consumed by the generation worker.
_config: str | None = None
# Default voice URL used when neither ``voice_url`` nor ``voice_wav`` is provided.
default_voice: str | None = None
# Queue used to submit jobs to the dedicated MLX generation thread.
_task_queue: Queue | None = None
# Whether to normalize voice prompts to a consistent peak level.
_normalize_voice: bool = False
# Maximum number of voice states to keep cached in the generation worker.
_voice_cache_size: int = 2


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the MLX generation worker on startup and shut it down cleanly."""
    global _task_queue
    _task_queue = Queue()
    ready_event = threading.Event()
    worker = threading.Thread(
        target=generation_worker,
        args=(ready_event, _config, _task_queue, _normalize_voice, _voice_cache_size),
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
<p>Default voice: <strong>{default_voice or "none (must be provided per request)"}</strong></p>
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
        voice_url: Remote URL starting with ``http://``, ``https://``, or ``hf://``,
            or a path to a ``.safetensors`` voice embedding.
        voice_wav: Optional uploaded voice file (mutually exclusive with ``voice_url``).
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if voice_url is None and voice_wav is None:
        if default_voice is None:
            raise HTTPException(
                status_code=400,
                detail="voice_url or voice_wav is required",
            )
        voice_url = default_voice

    if voice_url is not None and voice_wav is not None:
        raise HTTPException(status_code=400, detail="Cannot provide both voice_url and voice_wav")

    if voice_url is not None:
        if not is_builtin_or_remote_voice(voice_url):
            raise HTTPException(
                status_code=400,
                detail="voice_url must be a remote URL or a .safetensors path",
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
    normalize_voice: bool = False,
    voice_cache_size: int = 2,
) -> None:
    """Configure and start the FastAPI server."""
    global _config, default_voice, _normalize_voice, _voice_cache_size

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _config = config
    _normalize_voice = normalize_voice
    _voice_cache_size = voice_cache_size
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
        default=None,
        help="Default voice URL used when no voice_url/voice_wav is supplied",
    )
    parser.add_argument(
        "--normalize-voice",
        action="store_true",
        help="Normalize voice prompts to a consistent peak level before cloning",
    )
    parser.add_argument(
        "--voice-cache-size",
        type=int,
        default=2,
        help="Number of voice states to cache (0 disables caching)",
    )

    args = parser.parse_args()
    serve(
        host=args.host,
        port=args.port,
        reload=args.reload,
        config=args.config,
        default_voice_arg=args.default_voice,
        normalize_voice=args.normalize_voice,
        voice_cache_size=args.voice_cache_size,
    )


if __name__ == "__main__":
    serve_cli()
