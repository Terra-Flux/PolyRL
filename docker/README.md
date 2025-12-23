# PolyRL Trainer Container

This image builds from `nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04`, copies the local workspace into `/workspace/polyrl`, installs Rust, and installs PolyRL and its dependencies with `uv`.

## Prerequisites

- NVIDIA Container Toolkit installed so `--gpus all` works. Install via the NVIDIA apt guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian, then restart Docker afterward with `sudo systemctl restart docker`.

## Build

Run from the repo root (Dockerfiles are under `docker/`):

- Trainer image (includes Rust, Verl, and OpenR1 data):
  ```bash
  docker build -t polyrl:trainer -f docker/Dockerfile .
  ```
- Rollout engine image (no Rust/Verl/OpenR1 data):
  ```bash
  docker build -t polyrl:rollout -f docker/Dockerfile.rollout .
  ```

## Run (one-shot)

Trainer example (GPU, large shm, privileged):

```bash
docker run \
  --name polyrl-trainer \
  --gpus all \
  --privileged \
  -u 0 \
  --network=host \
  --userns=host \
  --cap-add=IPC_LOCK \
  --shm-size=2048g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it --rm polyrl:trainer bash
```

Notes:
- Requires a large shared memory segment (2048GB recommended `--shm-size=2048g`).
- For maximal IPC sharing, you can add `--ipc=host`.

Rollout engine example (no Rust/Verl/OpenR1 baked in):

```bash
docker run \
  --name polyrl-rollout \
  --gpus all \
  --privileged \
  -u 0 \
  --network=host \
  --userns=host \
  --cap-add=IPC_LOCK \
  --shm-size=2048g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it --rm polyrl:rollout bash
```

## Run with docker-compose (Deprecated)

Also from repo root:

```bash
docker compose -f docker/docker-compose.yml up -d polyrl
```

The compose file bakes in the same options as the one-shot command: GPUs, privileged, host network/userns, shm-size, ulimits, and a bind mount to `/workspace/polyrl`. Use `docker compose down` to stop it.

## Dataset (OpenR1)

PolyRL expects the OpenR1 dataset to be available. The trainer Docker build downloads it automatically to `/data/openr1` via `examples/data_preprocess/openr1.py`. If you prefer a different location, edit `docker/Dockerfile` to change the `--local_dir` path. The rollout image does not download this dataset.
