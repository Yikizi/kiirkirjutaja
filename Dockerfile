FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /opt/kiirkirjutaja

ARG MODEL_BASE_URL="https://huggingface.co/TalTechNLP/streaming-zipformer.et-en/resolve/main"
RUN mkdir -p /opt/kiirkirjutaja/models/sherpa-int8 \
    && for f in encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt; do \
         curl -fL --retry 3 --retry-delay 2 "$MODEL_BASE_URL/$f" \
           -o "/opt/kiirkirjutaja/models/sherpa-int8/$f"; \
       done

COPY main.py /opt/kiirkirjutaja/
COPY asr.py /opt/kiirkirjutaja/
COPY wyoming_handler.py /opt/kiirkirjutaja/

EXPOSE 10300

CMD ["python", "main.py", "--wyoming-uri", "tcp://0.0.0.0:10300"]
