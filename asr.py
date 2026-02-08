"""Small ASR helpers for Wyoming pipeline."""

import logging

import numpy as np

_LOGGER = logging.getLogger(__name__)


def transcribe_audio(recognizer, audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """Transcribe signed 16-bit mono PCM bytes into text."""
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float = audio_array.astype(np.float32) / np.iinfo(np.int16).max

    _LOGGER.debug(
        "Transcribing %d samples (%.2fs)",
        len(audio_float),
        len(audio_float) / sample_rate,
    )

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio_float)

    # Small tail padding helps model finalize trailing words.
    tail_padding = np.random.rand(int(sample_rate * 0.3)).astype(np.float32) * 0.01
    stream.accept_waveform(sample_rate, tail_padding)
    stream.input_finished()

    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    text = recognizer.get_result(stream)
    recognizer.reset(stream)
    return text.strip()
