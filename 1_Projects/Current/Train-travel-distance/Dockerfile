FROM ros:humble

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get -y install \
    vim wget curl \
    libopencv-dev \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics,video