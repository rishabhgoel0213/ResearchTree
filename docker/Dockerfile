ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG SYSTEM_PACKAGES=""
ARG NODE_MAJOR=22
ARG CODEX_NPM_SPEC=@openai/codex@0.118.0

ENV PIXI_HOME=/opt/pixi
ENV PATH=${PIXI_HOME}/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        bubblewrap \
        ca-certificates \
        curl \
        git \
        git-lfs \
        gnupg \
        procps \
        python3 \
        python3-pip \
        python3-tomli \
        python3-venv \
        tmux \
        ${SYSTEM_PACKAGES} \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g ${CODEX_NPM_SPEC} \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | sh

WORKDIR /workspace

CMD ["bash"]
