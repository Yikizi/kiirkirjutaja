#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="kiirkirjutaja-int8-local"
DEVICE=""
NO_BUILD="false"
NO_COLIMA="false"
LIST_ONLY="false"

usage() {
  cat <<'EOF'
Run live Estonian transcription from Mac microphone through Docker.

Usage:
  ./run_mic_live.sh [options]

Options:
  -d, --device <id>      AVFoundation audio device id (e.g. 1 or :1)
  -i, --image <name>     Docker image name (default: kiirkirjutaja-int8-local)
      --no-build         Skip docker build
      --no-colima        Do not auto-start Colima
      --list-devices     List microphone devices and exit
  -h, --help             Show help

Examples:
  ./run_mic_live.sh
  ./run_mic_live.sh --device 1
  ./run_mic_live.sh --device :0 --no-build
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: missing command '$1'" >&2
    exit 1
  fi
}

list_devices() {
  echo "Available AVFoundation audio devices:"
  ffmpeg -f avfoundation -list_devices true -i "" 2>&1 \
    | sed -n '/AVFoundation audio devices:/,/AVFoundation video devices:/p'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--device)
      DEVICE="${2:-}"
      shift 2
      ;;
    -i|--image)
      IMAGE_NAME="${2:-}"
      shift 2
      ;;
    --no-build)
      NO_BUILD="true"
      shift
      ;;
    --no-colima)
      NO_COLIMA="true"
      shift
      ;;
    --list-devices)
      LIST_ONLY="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

require_cmd ffmpeg
require_cmd docker

if [[ "$NO_COLIMA" != "true" ]]; then
  require_cmd colima
  if ! colima status >/dev/null 2>&1; then
    echo "Starting Colima..."
    colima start
  fi
  docker context use colima >/dev/null 2>&1 || true
fi

if [[ "$LIST_ONLY" == "true" ]]; then
  list_devices
  exit 0
fi

if [[ "$NO_BUILD" != "true" ]]; then
  echo "Building image: $IMAGE_NAME"
  docker build -t "$IMAGE_NAME" .
fi

if [[ -z "$DEVICE" ]]; then
  list_devices
  echo
  read -r -p "Enter audio device id (e.g. 1): " DEVICE
fi

if [[ -z "$DEVICE" ]]; then
  echo "Error: audio device id is required" >&2
  exit 1
fi

if [[ "$DEVICE" != :* ]]; then
  DEVICE=":$DEVICE"
fi

echo

echo "Streaming from microphone device $DEVICE"
echo "Press Ctrl+C to stop"

auto_python_script=$(cat <<'PY'
import sys
import numpy as np
import sherpa_onnx

SAMPLE_RATE = 16000
CHUNK = 1600  # 0.1s

rec = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens="models/sherpa-int8/tokens.txt",
    encoder="models/sherpa-int8/encoder.int8.onnx",
    decoder="models/sherpa-int8/decoder.int8.onnx",
    joiner="models/sherpa-int8/joiner.int8.onnx",
    num_threads=2,
    sample_rate=SAMPLE_RATE,
    feature_dim=80,
    enable_endpoint_detection=True,
    rule1_min_trailing_silence=2.4,
    rule2_min_trailing_silence=1.2,
    rule3_min_utterance_length=300,
)

stream = rec.create_stream()
last = ""
print("Listening...", file=sys.stderr)

while True:
    b = sys.stdin.buffer.read(CHUNK * 2)
    if not b or len(b) < CHUNK * 2:
        break

    s = np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0
    stream.accept_waveform(SAMPLE_RATE, s)

    while rec.is_ready(stream):
        rec.decode_stream(stream)

    txt = rec.get_result(stream)
    endpoint = rec.is_endpoint(stream)

    if txt and txt != last:
        print(f"\r{txt}", end="", flush=True)
        last = txt

    if endpoint and txt:
        print(f"\n✓ {txt}", flush=True)
        rec.reset(stream)
        last = ""
PY
)

set +e
ffmpeg -f avfoundation -i "$DEVICE" \
  -ar 16000 -ac 1 -f s16le -acodec pcm_s16le -loglevel error - | \
docker run --rm -i "$IMAGE_NAME" python -u -c "$auto_python_script"
status=$?
set -e

if [[ $status -ne 0 ]]; then
  echo
  echo "Transcription pipeline exited with code $status" >&2
  echo "Tip: check microphone permissions and correct device id." >&2
  exit $status
fi
