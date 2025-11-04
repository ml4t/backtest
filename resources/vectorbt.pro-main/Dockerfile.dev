FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake git wget sudo vim && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
    tar -xzf ta-lib-0.6.4-src.tar.gz && \
    cd ta-lib-0.6.4/ && \
    ./configure --prefix=/usr && \
    make && \
    make install
RUN rm -R ta-lib-0.6.4 ta-lib-0.6.4-src.tar.gz

COPY ./vectorbtpro ./vectorbtpro
COPY pyproject.toml LICENSE README.md ./
RUN uv pip install --no-cache-dir ".[all]"
RUN uv pip install --no-cache-dir --no-deps "universal-portfolios"

RUN groupadd -g 1000 devuser && \
    useradd -u 1000 -g devuser -m -s /bin/bash devuser && \
    echo 'devuser ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    install -d -o devuser -g devuser /workspace
USER devuser
WORKDIR /workspace

ENTRYPOINT ["bash"]
