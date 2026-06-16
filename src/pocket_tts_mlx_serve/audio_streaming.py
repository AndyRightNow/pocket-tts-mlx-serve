"""WAV streaming helpers for MLX-generated audio chunks."""

import io
import logging
import wave
from queue import Queue
from typing import Any, Iterator

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


class _QueueFileLike(io.IOBase):
    """Minimal file-like object that pushes bytes into a queue."""

    def __init__(self, queue: Queue):
        self.queue = queue

    def write(self, data: bytes) -> int:
        self.queue.put(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.queue.put(None)


class StreamingWAVWriter:
    """Streaming WAV writer that emits int16 PCM chunks."""

    def __init__(self, output_stream: Any, sample_rate: int):
        self.output_stream = output_stream
        self.sample_rate = sample_rate
        self.wave_writer: wave.Wave_write | None = None

    def write_header(self) -> None:
        self.wave_writer = wave.open(self.output_stream, "wb")
        self.wave_writer.setnchannels(1)
        self.wave_writer.setsampwidth(2)
        self.wave_writer.setframerate(self.sample_rate)
        # Large placeholder frame count; real count is patched on close for seekable files.
        self.wave_writer.setnframes(1_000_000_000)

    def write_pcm_data(self, audio_chunk: mx.array | np.ndarray) -> None:
        if self.wave_writer is None:
            raise RuntimeError("WAV header has not been written")

        arr = np.array(audio_chunk)
        if arr.ndim > 1:
            arr = arr.reshape(-1)

        int16 = np.clip(arr, -1.0, 1.0) * 32767.0
        chunk_bytes = int16.astype(np.int16).tobytes()
        self.wave_writer.writeframesraw(chunk_bytes)

    def finalize(self) -> None:
        if self.wave_writer is None:
            return

        # Append a short silence tail so players do not clip the last frame.
        silence_samples = int(self.sample_rate * 0.2)
        self.wave_writer.writeframesraw(bytes(silence_samples * 2))

        # Avoid calling _patchheader for non-seekable streams.
        setattr(self.wave_writer, "_patchheader", lambda: None)
        self.wave_writer.close()


def stream_audio_chunks_to_queue(
    queue: Queue,
    audio_chunks: Iterator[mx.array],
    sample_rate: int,
) -> None:
    """Write an iterable of MLX audio chunks to a WAV stream backed by ``queue``."""

    file_like = _QueueFileLike(queue)
    writer = StreamingWAVWriter(file_like, sample_rate)
    writer.write_header()

    try:
        for chunk in audio_chunks:
            writer.write_pcm_data(chunk)
    finally:
        writer.finalize()
        file_like.close()
