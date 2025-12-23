FROM nvcr.io/nvidia/pytorch:25.05-py3

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:/root/.cargo/bin:${PATH}"

WORKDIR /workspace/polyrl

# Copy the local workspace into the image
COPY . .

# System deps, Rust (for rollout manager), and Python installer tooling
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        ninja-build \
        numactl \
        python3-dev \
        python3-venv && \
    rm -rf /var/lib/apt/lists/* && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Python dependencies (uv, sglang, flash-attn, verl, PolyRL)
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip uv && \
    uv pip install cmake && \
    uv pip install sglang==0.5.5 && \
    uv pip install flash-attn --no-build-isolation && \
    cd 3rdparty/verl && uv pip install -e . && \
    cd /workspace/polyrl && uv pip install -e .

# Pre-download OpenR1 data needed by PolyRL
RUN mkdir -p /data/openr1 && \
    python examples/data_preprocess/openr1.py --local_dir /data/openr1

CMD ["/bin/bash"]
