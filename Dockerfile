FROM ubuntu:22.04

RUN apt update && apt install -y git gcc g++ make libtbb2-dev && apt clean

RUN git clone --branch devel https://github.com/nim-lang/Nim.git --depth 1 /opt/Nim

WORKDIR /opt/Nim
RUN sh ./build_all.sh 

RUN ./bin/nim c koch
RUN ./koch boot -d:release
RUN ./koch tools

RUN mkdir -p /root/.nimble/bin

ENV PATH="/opt/Nim/bin/:/root/.nimble/bin:${PATH}"
