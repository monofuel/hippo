FROM ubuntu:22.04

RUN apt update && apt install -y git gdb gcc g++ make libtbb2 libtbb2-dev nvidia-cuda-toolkit && apt clean

# sudo wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
#   echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.1 jammy main" | sudo tee --append /etc/apt/sources.list.d/rocm.list
#   echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | sudo tee /etc/apt/preferences.d/rocm-pin-600
#   sudo apt-get update
#   # rocm takes up too much space for github ci runner
#   #sudo apt-get install -y hipcc rocm nvidia-cuda-toolkit
#   sudo apt-get install -y hipcc rocm-device-libs rocm-hip-runtime nvidia-cuda-toolkit
#   echo "/opt/rocm/bin" >> $GITHUB_PATH
#   echo "/usr/local/cuda/bin" >> $GITHUB_PATH


RUN git clone --branch devel https://github.com/nim-lang/Nim.git --depth 1 /opt/Nim

WORKDIR /opt/Nim
RUN sh ./build_all.sh 

RUN ./bin/nim c koch
RUN ./koch boot -d:release
RUN ./koch tools

RUN mkdir -p /root/.nimble/bin

ENV PATH="/opt/Nim/bin/:/root/.nimble/bin:/usr/local/cuda/bin:${PATH}"
