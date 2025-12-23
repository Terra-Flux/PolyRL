# Reproduce

Step-by-step guide to reproduce the paper results.


### Clone repo

```bash
git clone https://github.com/Terra-Flux/PolyRL.git --recursive polyrl
cd polyrl
# disable logging in veRL as it will impact performance
sed -i 's/"log_level": "info",/# "log_level": "info",/' 3rdparty/verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py
git checkout artifact
```

### Hardware

Assume at least **two H100x8 instances**:
- Machine A: trainer.
- Machine B: emulates four rollout engines on preemptible instances.
- Network: 200+ Gbps egress host network on Machine A. This is common for Cloud providers like AWS and GCP.

### Environment Setup

#### Docker (recommended)

Install Docker and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian).
For explanations, refer to [polyrl-docker](docker/README.md).

Build images:
```bash
# machine A
docker build -t polyrl:trainer -f docker/Dockerfile .

# machine B
docker build -t polyrl:rollout -f docker/Dockerfile.rollout .
```

Run containers:
> **NOTE:** The following command maps host dir to `/workspace/polyrl` inside the container, any change you made will reflect on the host side.

```bash
# machine A
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
  -v "$(pwd)":/workspace/polyrl \
  -it --rm polyrl:trainer bash

# machine B
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
  -v "$(pwd)":/workspace/polyrl \
  -it --rm polyrl:rollout bash
```

> **NOTE:** All necessities for rollout is also included in the trainer image, but it is recommended to use a separate container for rollout.

Python modules are installed in a virtual environment, enter venv by `source /opt/venv/bin/activate`.

#### Local environment

See [INSTALL.md](INSTALL.md) to install from source.

### Dataset

- Trainer Docker image downloads OpenR1 to `/workspace/polyrl/data/openr1` during build.
- For a source install, download OpenR1 with `python examples/data_preprocess/openr1.py --local_dir ~/data/openr1`.


### VeRL Training (baseline)

Launch VeRL training:
```bash
# machine A
source /opt/venv/bin/activate
cd /workspace/polyrl  # already set when using the Docker image
bash examples/scripts/run_sync_grpo8b_default.sh
```

You are expected to see `perf/throughput` is around 1,300 token/s. 
Multiplying by 8 for throughput of all GPUs, the overall throughput of VeRL is ~10,000 token/s.


### FluidRL Training

Train `Qwen3-8B` on OpenR1 with GRPO:
```bash
# machine A
source /opt/venv/bin/activate
cd /workspace/polyrl  # already set when using the Docker image
bash examples/scripts/run_async_grpo8b_pipeline.sh
```

Trainer initialization takes time on first run. 
When the rollout manager is ready ( <font color="red"> when seeing red prompts on the trainer</font>), 
start rollout workers on machine B. 
Default rollout manager port is 5000, replace `<MACHINE_A_IP>` with the IP address of machine A:
```bash
# machine B
source /opt/venv/bin/activate
bash examples/scripts/launch_sglang_step.sh <MACHINE_A_IP> 5000
```

The script launches one rollout worker every 1800 seconds (30 minutes); after four are up, 
it retires the oldest every 1800 seconds. 
Rollout workers join/leave seamlessly, which is handled by the rollout manager of FluidRL.

You are expected to see `perf/throughput_all_gpus` (#rollout workers → throughput):
- 1 → ~12,000 token/s
- 2 → ~14,000 token/s
- 3 → ~16,000 token/s
- 4 → ~18,000 token/s

The average cost of standard H100x8 instance is $83.79 per hour, while spot ones are $21.28 per hour.
The improvement of cost efficiency (token per dollar) is around 60%.

### Demonstrate Preemption

FluidRL handles preemptible resources. Besides `launch_sglang_step.sh`, you can start rollout workers manually:
```bash
# machine B
source /opt/venv/bin/activate
bash examples/scripts/launch_sglang_8b.sh <MACHINE_A_IP> 5000 <API_PORT> <HANDSHAKE_PORT>
```

Use unique `API_PORT` per worker; set `HANDSHAKE_PORT` offsets of 1000 per worker to avoid conflicts. Example:
```bash
CUDA_VISIBLE_DEVICES=0,1 bash examples/scripts/launch_sglang_8b.sh 192.168.0.1 5000 40001 20000
CUDA_VISIBLE_DEVICES=2,3 bash examples/scripts/launch_sglang_8b.sh 192.168.0.1 5000 40002 21000
# launch a third worker later
CUDA_VISIBLE_DEVICES=4,5 bash examples/scripts/launch_sglang_8b.sh 192.168.0.1 5000 40003 22000
...
```

Use `Ctrl+C` to stop a rollout worker when it is still generating responses; the rollout manager will automatically migrate requests to remaining workers.

> **IMPORTANT:** This version of FluidRL cannot handle zero rollout workers; that support is planned for v0.1.0.


### Optimizations

1. To use high-speed network interface to transfer weights, specify the range of IP addresses of the rollout workers in `allowed_sender_ips` in [`rollout-manager/config.toml`](rollout-manager/config.toml).
2. If the baseline throughput is lower than expected, comment out `"log_level": "info",` in [sglang_rollout.py#L461](https://github.com/volcengine/verl/blob/0eb50ec4a33cda97e05ed8caab9c7f17a30c05a9/verl/workers/rollout/sglang_rollout/sglang_rollout.py#L461)


### Known issues
1. When encounter OOM issue, clean up `dev/shm/*`.
2. When encounter `libcuda.so not found`, it usually due to ldconfig is searching a wrong pass. Find the correct path by `find /usr -name 'libcuda.so'` and run `ldconfig <parent dir of libcuda.so>` to add the path.
