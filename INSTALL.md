# Install

## Create conda environment

```bash
conda create -n polyrl python=3.12
conda activate polyrl
conda install -c conda-forge libcurl numactl 
```

## Install sglang

```bash
cd src/sglang
pip install -e "python[all]"
pip install flash-attn --no-build-isolation
```

## Install verl

```bash
cd ../verl
pip install -e .
```

## Install RLBoost
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cd ../rlboost
pip install -e .
pip install mooncake-transfer-engine # optional
```
