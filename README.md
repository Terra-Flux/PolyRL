# PolyRL

## Download

```bash
git clone https://github.com/Terra-Flux/PolyRL.git --recursive polyrl
cd polyrl
git checkout artifact
# disable logging in veRL as it will impact performance
sed -i 's/"log_level": "info",/# "log_level": "info",/' 3rdparty/verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py
```

## Getting Started Instructions

Please refer to [MINIMUM.md](MINIMUM.md) for instructions to get started with FluidRL.

## Detailed Instructions to Reproduce FluidRL

Please refer to [REPRODUCE.md](REPRODUCE.md) for instructions to reproduce the FluidRL main results.

