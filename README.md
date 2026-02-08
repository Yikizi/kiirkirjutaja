# Kiirkirjutaja (Minimal Wyoming INT8 Runtime)

This repo is intentionally trimmed for one use case:

- local Estonian ASR in Home Assistant
- Wyoming protocol server
- INT8 sherpa-onnx transducer model

## What Is Included

- `main.py`: Wyoming server entrypoint
- `asr.py`: audio-bytes -> transcript helper
- `wyoming_handler.py`: Wyoming event handling
- `Dockerfile`: minimal container runtime
- `requirements.txt`: minimal Python dependencies

## Run With Docker

```bash
docker build -t kiirkirjutaja-int8 .
docker run --rm -p 10300:10300 kiirkirjutaja-int8
```

The server listens on `tcp://0.0.0.0:10300`.

Model files are downloaded automatically at build time from:

- `https://huggingface.co/TalTechNLP/streaming-zipformer.et-en`

Optional custom model source:

```bash
docker build \
  --build-arg MODEL_BASE_URL="https://your-host/path" \
  -t kiirkirjutaja-int8 .
```

## Home Assistant

Add Wyoming integration with:

- Host: `127.0.0.1` (or your container host)
- Port: `10300`

## Notes

- This is not the full upstream Kiirkirjutaja feature set.
- Legacy pipeline code (presenters, LID, VAD, speaker-change stack) was removed to keep maintenance simple.
- If needed, you can restore advanced features from upstream history.
