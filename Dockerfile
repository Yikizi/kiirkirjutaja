FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /opt/kiirkirjutaja

COPY models/sherpa-int8 /opt/kiirkirjutaja/models/sherpa-int8
COPY main.py /opt/kiirkirjutaja/
COPY asr.py /opt/kiirkirjutaja/
COPY wyoming_handler.py /opt/kiirkirjutaja/

EXPOSE 10300

CMD ["python", "main.py", "--wyoming-uri", "tcp://0.0.0.0:10300"]
