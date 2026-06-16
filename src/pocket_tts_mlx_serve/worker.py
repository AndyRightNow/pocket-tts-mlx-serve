"""MLX generation worker for the pocket-tts-mlx-serve server.

The worker owns the ``TTSModel`` and runs in a dedicated thread so that MLX
arrays and streams stay bound to the thread that loaded the model.
"""

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from queue import Queue
from typing import Any

import mlx.core as mx
import numpy as np
from pocket_tts_mlx import TTSModel
from pocket_tts_mlx.data.audio import audio_read
from pocket_tts_mlx.models.tts_model import convert_audio
from pocket_tts_mlx.modules.stateful_module import init_states
from pocket_tts_mlx.utils.utils import download_if_necessary
from pocket_tts_mlx.utils.weight_conversion import load_safetensors_to_mlx

from pocket_tts_mlx_serve.audio_streaming import stream_audio_chunks_to_queue

logger = logging.getLogger(__name__)

# Target peak level when normalizing voice prompts.
_NORMALIZATION_PEAK = 0.95


def _normalize_audio(audio: np.ndarray, target_peak: float = _NORMALIZATION_PEAK) -> np.ndarray:
    """Scale audio so its absolute peak matches ``target_peak``."""
    peak = float(np.abs(audio).max())
    if peak == 0:
        return audio
    return (audio / peak * target_peak).astype(np.float32)


def _state_from_audio(
    model: TTSModel,
    audio: np.ndarray,
    sample_rate: int,
    normalize: bool,
    truncate: bool,
) -> dict:
    """Build a flow-lm state from a loaded audio prompt."""
    if truncate:
        max_samples = int(30 * sample_rate)
        if audio.shape[-1] > max_samples:
            audio = audio[..., :max_samples]
            logger.info("Audio truncated to 30 seconds")

    if normalize:
        audio = _normalize_audio(audio)

    audio_conditioning = convert_audio(audio, sample_rate, model.config.mimi.sample_rate, 1)
    prompt = model._encode_audio(mx.array(audio_conditioning)[None, ...])
    model_state = init_states(model.flow_lm, batch_size=1, sequence_length=prompt.shape[1])
    model._run_flow_lm_and_increment_step(model_state=model_state, audio_conditioning=prompt)
    return model_state


def _load_voice_embedding(url: str) -> mx.array:
    """Load an ``audio_prompt`` embedding from a safetensors file."""
    path = download_if_necessary(url)
    weights = load_safetensors_to_mlx(path, key_filter="audio_prompt")
    if "audio_prompt" not in weights:
        raise KeyError(f"'audio_prompt' not found in voice embedding {url}")
    return weights["audio_prompt"]


class _JobFailed:
    """Marker placed on a response queue when a generation job fails."""

    def __init__(self, message: str):
        self.message = message


def is_builtin_or_remote_voice(voice_url: str) -> bool:
    """Return ``True`` if ``voice_url`` can be handled without uploading a file."""
    return voice_url.startswith(("http://", "https://", "hf://")) or voice_url.endswith(
        ".safetensors"
    )


def generation_worker(
    ready_event: threading.Event,
    config: str | None,
    task_queue: Queue,
    normalize_voice: bool = False,
    voice_cache_size: int = 2,
) -> None:
    """Dedicated thread that owns the MLX model and processes generation jobs."""
    logger.info("Loading pocket-tts-mlx model in generation worker...")
    model = TTSModel.load_model(config=config) if config else TTSModel.load_model()
    logger.info("Model loaded in worker.")

    # Patch the audio encoder so voice cloning works with arbitrary downloaded
    # audio files.  The bundled implementation transposes with only two axes,
    # which fails for the batched [B, C, T] tensors returned by Mimi.
    def _encode_audio_fixed(audio: mx.array) -> mx.array:
        encoded = model.mimi.encode_to_latent(audio)
        latents = mx.transpose(encoded, (0, 2, 1)).astype(mx.float32)
        return mx.matmul(latents, model.flow_lm.speaker_proj_weight.T)

    model._encode_audio = _encode_audio_fixed

    # Small voice-state cache, matching the upstream serve behaviour.
    voice_cache: OrderedDict[str, dict] | None = None
    if voice_cache_size > 0:
        voice_cache = OrderedDict()

    ready_event.set()

    while True:
        item = task_queue.get()
        if item is None:
            break

        (
            response_queue,
            text,
            voice_kind,
            voice_value,
            max_tokens,
            frames_after_eos,
            trim_start_ms,
            fade_in_ms,
            warmup_frames,
        ) = item
        try:
            if voice_kind == "url" and voice_cache is not None and voice_value in voice_cache:
                model_state = voice_cache[voice_value]
                voice_cache.move_to_end(voice_value)
            else:
                if voice_value.endswith(".safetensors"):
                    embedding = _load_voice_embedding(voice_value)
                    model_state = init_states(
                        model.flow_lm,
                        batch_size=1,
                        sequence_length=embedding.shape[1],
                    )
                    model._run_flow_lm_and_increment_step(
                        model_state=model_state,
                        audio_conditioning=embedding,
                    )
                elif voice_kind == "url":
                    # Generic audio URL: download and clone the voice.
                    audio_path = download_if_necessary(voice_value)
                    audio, sample_rate = audio_read(audio_path)
                    model_state = _state_from_audio(
                        model,
                        audio,
                        sample_rate,
                        normalize=normalize_voice,
                        truncate=False,
                    )
                else:
                    audio, sample_rate = audio_read(Path(voice_value))
                    model_state = _state_from_audio(
                        model,
                        audio,
                        sample_rate,
                        normalize=normalize_voice,
                        truncate=True,
                    )

                if voice_cache is not None and voice_kind == "url":
                    voice_cache[voice_value] = model_state
                    if len(voice_cache) > voice_cache_size:
                        voice_cache.popitem(last=False)
        except Exception as exc:
            logger.exception("Failed to load voice state")
            response_queue.put(_JobFailed(str(exc)))
            response_queue.put(None)
            continue

        try:
            if trim_start_ms or fade_in_ms:
                audio = model.generate_audio(
                    model_state=model_state,
                    text_to_generate=text,
                    max_tokens=max_tokens,
                    frames_after_eos=frames_after_eos,
                    trim_start_ms=trim_start_ms,
                    fade_in_ms=fade_in_ms,
                    warmup_frames=warmup_frames,
                )
                audio_chunks: Any = iter([audio])
            else:
                audio_chunks = model.generate_audio_stream(
                    model_state=model_state,
                    text_to_generate=text,
                    max_tokens=max_tokens,
                    frames_after_eos=frames_after_eos,
                    warmup_frames=warmup_frames,
                )
            stream_audio_chunks_to_queue(response_queue, audio_chunks, model.sample_rate)
        except Exception:
            logger.exception("TTS generation failed")
        finally:
            response_queue.put(None)


def iter_response_queue(response_queue: Queue) -> Any:
    """Yield WAV bytes from a response queue, raising on job failure."""
    first_item = response_queue.get()
    if isinstance(first_item, _JobFailed):
        # Drain the terminating None so the queue does not leak.
        response_queue.get()
        raise RuntimeError(first_item.message)
    if first_item is not None:
        yield first_item

    while True:
        data = response_queue.get()
        if data is None:
            break
        yield data
