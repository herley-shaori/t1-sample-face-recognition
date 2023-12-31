# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:22.04

MAINTAINER Amazon AI <sage-learner@amazon.com>

RUN apt-get update &&\
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN #apt-get -y update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
#RUN apt-get -y update
RUN apt install python3.12 -y
RUN apt install curl -y
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN apt-get update -y
RUN apt install -y dos2unix
RUN pip install --only-binary opencv-python-headless opencv-python-headless

COPY . /opt/program
WORKDIR /opt/program
RUN mkdir -p inference_input
RUN pip install  --ignore-installed  -r requirements.txt

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

RUN dos2unix train
RUN dos2unix predictor.py
RUN dos2unix serve
RUN dos2unix wsgi.py