"""
Wyoming protocol handler for kiirkirjutaja.

Handles incoming audio from Home Assistant and returns transcripts.
"""

import asyncio
import logging
from typing import Optional
from functools import partial

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger(__name__)

# Expected audio format
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit
CHANNELS = 1  # mono


class KiirkirjutajaEventHandler(AsyncEventHandler):
    """
    Wyoming event handler for kiirkirjutaja ASR.

    Receives audio chunks via Wyoming protocol, buffers them,
    and returns transcripts when audio stream ends.
    """

    def __init__(
        self,
        wyoming_info: Info,
        transcribe_func,
        model_lock: asyncio.Lock,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.transcribe_func = transcribe_func
        self.model_lock = model_lock

        # Audio state
        self.audio_buffer = bytes()
        self.audio_started = False
        self.audio_rate: Optional[int] = None
        self.audio_width: Optional[int] = None
        self.audio_channels: Optional[int] = None

    async def handle_event(self, event: Event) -> bool:
        """Handle incoming Wyoming events."""

        if Describe.is_type(event.type):
            # Client wants service info
            _LOGGER.debug("Received Describe request")
            await self.write_event(self.wyoming_info.event())

        elif AudioStart.is_type(event.type):
            await self._handle_audio_start(AudioStart.from_event(event))

        elif AudioChunk.is_type(event.type):
            await self._handle_audio_chunk(AudioChunk.from_event(event))

        elif AudioStop.is_type(event.type):
            await self._handle_audio_stop()

        elif Transcribe.is_type(event.type):
            # Client requesting transcription info
            _LOGGER.debug("Received Transcribe request")

        return True

    async def _handle_audio_start(self, audio_start: AudioStart) -> None:
        """Handle start of audio stream."""
        _LOGGER.debug(
            f"Audio start: rate={audio_start.rate}, "
            f"width={audio_start.width}, channels={audio_start.channels}"
        )

        # Validate audio format
        if audio_start.rate != SAMPLE_RATE:
            _LOGGER.warning(
                f"Expected {SAMPLE_RATE}Hz, got {audio_start.rate}Hz. "
                "Transcription quality may suffer."
            )

        if audio_start.width != SAMPLE_WIDTH:
            _LOGGER.warning(
                f"Expected {SAMPLE_WIDTH}-byte samples, got {audio_start.width}. "
                "Transcription may fail."
            )

        if audio_start.channels != CHANNELS:
            _LOGGER.warning(
                f"Expected {CHANNELS} channel(s), got {audio_start.channels}. "
                "Transcription quality may suffer."
            )

        # Store format info
        self.audio_rate = audio_start.rate
        self.audio_width = audio_start.width
        self.audio_channels = audio_start.channels

        # Reset buffer
        self.audio_buffer = bytes()
        self.audio_started = True

    async def _handle_audio_chunk(self, chunk: AudioChunk) -> None:
        """Handle incoming audio chunk."""
        if not self.audio_started:
            _LOGGER.warning("Received AudioChunk before AudioStart, ignoring")
            return

        self.audio_buffer += chunk.audio
        _LOGGER.debug(f"Received audio chunk, buffer size: {len(self.audio_buffer)}")

    async def _handle_audio_stop(self) -> None:
        """Handle end of audio stream, perform transcription."""
        if not self.audio_started:
            _LOGGER.warning("Received AudioStop before AudioStart, ignoring")
            return

        _LOGGER.info(
            f"Audio stop, total buffer size: {len(self.audio_buffer)} bytes "
            f"({len(self.audio_buffer) / SAMPLE_RATE / SAMPLE_WIDTH:.2f} seconds)"
        )

        self.audio_started = False

        if len(self.audio_buffer) == 0:
            _LOGGER.warning("Empty audio buffer, sending empty transcript")
            await self.write_event(Transcript(text="").event())
            return

        try:
            # Transcribe with lock to prevent concurrent model access
            async with self.model_lock:
                # Run sync transcription in executor to not block event loop
                loop = asyncio.get_event_loop()
                transcript = await loop.run_in_executor(
                    None,
                    self.transcribe_func,
                    self.audio_buffer
                )

            _LOGGER.info(f"Transcript: {transcript}")
            await self.write_event(Transcript(text=transcript).event())

        except Exception as e:
            _LOGGER.error(f"Transcription failed: {e}", exc_info=True)
            await self.write_event(Transcript(text="").event())

        finally:
            # Clear buffer
            self.audio_buffer = bytes()


def get_wyoming_info() -> Info:
    """Return Wyoming service info for kiirkirjutaja."""
    return Info(
        asr=[
            AsrProgram(
                name="kiirkirjutaja-local-int8",
                attribution=Attribution(
                    name="Tanel Alumae / TalTech",
                    url="https://github.com/alumae/kiirkirjutaja"
                ),
                installed=True,
                description="Estonian real-time speech recognition (INT8, local)",
                version="1.0.0-int8",
                models=[
                    AsrModel(
                        name="streaming-zipformer-int8",
                        attribution=Attribution(
                            name="TalTech",
                            url="https://huggingface.co/TalTechNLP/streaming-zipformer.et-en"
                        ),
                        installed=True,
                        description="Estonian streaming ASR model (INT8 quantized)",
                        version="1.0.0",
                        languages=["et"],
                    )
                ],
            )
        ]
    )


async def run_wyoming_server(
    uri: str,
    transcribe_func,
) -> None:
    """
    Run Wyoming protocol server.

    Args:
        uri: Server URI, e.g., "tcp://0.0.0.0:10300"
        transcribe_func: Function that takes audio bytes and returns transcript string
    """
    wyoming_info = get_wyoming_info()
    model_lock = asyncio.Lock()

    server = AsyncServer.from_uri(uri)
    _LOGGER.info(f"Starting Wyoming server on {uri}")

    await server.run(
        partial(
            KiirkirjutajaEventHandler,
            wyoming_info,
            transcribe_func,
            model_lock,
        )
    )
