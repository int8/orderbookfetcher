FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y python3.6 python3.6-dev python3-pip

WORKDIR /app
ADD requirements.txt .
RUN pip3 install -r requirements.txt
ADD . .