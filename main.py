#!/usr/bin/env python3
"""Minimal Wyoming server entrypoint for local INT8 Kiirkirjutaja ASR."""

import argparse
import asyncio
import logging
import sys

import sherpa_onnx

from asr import transcribe_audio
from wyoming_handler import run_wyoming_server

_LOGGER = logging.getLogger(__name__)


def create_recognizer(model_dir: str) -> sherpa_onnx.OnlineRecognizer:
    """Load INT8 transducer recognizer from model directory."""
    _LOGGER.info("Loading INT8 ASR model from %s", model_dir)
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{model_dir}/tokens.txt",
        encoder=f"{model_dir}/encoder.int8.onnx",
        decoder=f"{model_dir}/decoder.int8.onnx",
        joiner=f"{model_dir}/joiner.int8.onnx",
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=5.0,
        rule2_min_trailing_silence=2.0,
        rule3_min_utterance_length=300,
        decoding_method="modified_beam_search",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kiirkirjutaja Wyoming server (INT8, minimal runtime)",
    )
    parser.add_argument(
        "--wyoming-uri",
        required=True,
        help="Wyoming URI, e.g. tcp://0.0.0.0:10300",
    )
    parser.add_argument(
        "--model-dir",
        default="models/sherpa-int8",
        help="Directory containing encoder/decoder/joiner/tokens files",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
        level=logging.INFO,
    )

    args = parse_args()
    recognizer = create_recognizer(args.model_dir)

    def transcribe_func(audio_bytes: bytes) -> str:
        return transcribe_audio(recognizer, audio_bytes)

    _LOGGER.info("Starting Wyoming server on %s", args.wyoming_uri)
    asyncio.run(run_wyoming_server(args.wyoming_uri, transcribe_func))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        _LOGGER.info("Server stopped")
        raise SystemExit(0)
