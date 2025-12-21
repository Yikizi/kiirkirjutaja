FROM python:3.9-slim-buster

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt 

RUN apt-get update && apt-get install  -y --no-install-recommends git ffmpeg

COPY models /opt/models

RUN echo '2022-01-31_16:24' >/dev/null

RUN git clone https://github.com/alumae/online_speaker_change_detector.git /opt/online-speaker-change-detector

RUN mkdir /opt/kiirkirjutaja \
    && cd /opt/kiirkirjutaja && ln -s ../models
    
COPY *.py /opt/kiirkirjutaja/

ENV PYTHONPATH="/opt/online-speaker-change-detector"

WORKDIR /opt/kiirkirjutaja

CMD ["/bin/bash"] 
