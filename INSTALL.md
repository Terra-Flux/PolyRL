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

conda install -c conda-forge libcurl # for mooncake-transfer-engine

pip install flash_attn==2.7.4.post1 --no-build-isolation
pip install rpyc==6.0.2 mooncake-transfer-engine==0.3.2.post1
```

