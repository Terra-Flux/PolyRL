# Instructions

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

## Install SGLang on GPU node

```bash
# commit d39ed074cf11ae9247dbcb04170e6921ca727559
sudo apt update && sudo apt install python3-dev
uv venv --python 3.12 && source .venv/bin/activate
cd sglang/
uv pip install -e "python[all]"
```

## Install SGLang-jax on TPU node

```bash
# commit 3ec442ae616201e7069bbc19345e9fcff3535668
uv venv --python 3.12 && source .venv/bin/activate
cd sglang-jax/
uv pip install -e "python[all]"
```

## Install utrans-engine
```bash
cd weight-transfer
uv pip install -e ".[tpu]" # on TPU node
uv pip install -e ".[gpu]" # on GPU node
```

## Important Insights
- Current implementation of `update_weights_from_disk` on TPU will force pause and retract all running request. The retract includes a kv cache clearance, so we should skip flush_cache after the weight updates (flush_cache fails on sglang-jax if pending request is not empty). After weight updates, it will resume generation.
- To avoid materialize whole model on single chip, the patch call device put with sharding, however, putting tensors one by one cannot fully utilize the PCIE bandwidth.

# Plan

## Finished
- Separate weight transfer agent from rlboost in to a proxy
- GPU proxy direct `update_weights_from_tensor`, tensor swapping
- SGLang-JAX patch with `update_weights_from_disk` to load weights, with pause/resume generation
- TPU proxy call `update_weights_from_disk` and share weights via shm. 
- 

## Next
- Proxy back to `health/` check if rollout engine connection breaks
- Performance profile on weight updates
- SHM to TPU memory copy optimization
- TPU network bandwidth optimization
- Proxy fetch engine info (tp size) automatically
- Single proxy for multiple engines on a shared host
- 