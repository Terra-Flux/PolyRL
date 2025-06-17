# Install

## Install locally (recommended)

### Create conda environment

```bash
conda create -n polyrl python=3.12
conda activate polyrl
```

### Install sglang

```bash
cd src/sglang
pip install -e "python[all]"
```

### Install verl

```bash
cd ../verl
pip install -e .
```

### Install other dependencies

Install Cargo for rust.
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
```

Install 
```bash
conda install -c conda-forge libcurl # for mooncake-transfer-engine

pip cache purge # (recommended) make sure flash attn is rebuilt and compatible with torch

pip install flash_attn==2.7.4.post1 --no-build-isolation
pip install rpyc==6.0.2 mooncake-transfer-engine==0.3.2.post1 --no-build-isolation
```

## Install in Docker

This is a demo Dockerfile for the development purpose. You have to run training and rollout instances in the container. Advanced examples supporting multi-container orchestration are on the way.

```bash
docker build -t polyrl -f docker/Dockerfile.demo .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it polyrl
```

## Known Issue
1. Older version Debian<=11 or Ubuntu<=20.04 don't have `GLIBC_2.34`. If users encounter problems such as missing lib*.so, they should uninstall this package by pip uninstall mooncake-transfer-engine, and build the binaries manually.
