# Install

## Create conda environment

```bash
conda create -n polyrl python=3.12
conda activate polyrl
```

## Install sglang

```bash
cd src/sglang
pip install -e "python[all]"
```

## Install verl

```bash
cd ../verl
pip install -e .
```

## Install other dependencies

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

conda install -c conda-forge libcurl numactl 

pip install flash_attn rpyc mooncake-transfer-engine --no-build-isolation
```

