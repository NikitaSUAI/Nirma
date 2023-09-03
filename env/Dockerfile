# syntax=docker/dockerfile:1
FROM ubuntu:22.04.22
COPY ./env/requirements.txt /usr/local
WORKDIR /usr/local
RUN pip install -U -r requirements.txt
