# GKE setup guide for PolyRL

## Preliminaries

### VPC setup (Recommend)

TODO: add management network setup instructions

### Build and Push Docker Image

```bash
cd docker
```

#### Create a Repository on GCloud to Save Docker Images

```bash
gcloud artifacts repositories create "<your repo name>" \
    --project="<your project name>" \
    --repository-format="docker" \
    --location="<your project region>" \
    --description="Docker repository for PolyRL workloads"

gcloud auth configure-docker "<your project region>-docker.pkg.dev" --project="<your project name>"
```

#### Build and push image

We provide a docker file to test functionality of Kubernetes instance.
You may build and push the repo following
```bash
docker build -t "<your image name>-test:v1" -f "docker/test.dockerfile" .
docker push "<your image name>-test:v1"

docker build -t "<your image name>-rollout:v1" -f "docker/rollout.dockerfile" .
docker push "<your image name>-rollout:v1"
```

## Create a Kubernetes Cluster

Use the Python CLI to create the cluster from a TOML config (e.g., `config/example.toml`):
```bash
python cluster_setup/gke_setup.py --config example create-cluster
```
It initializes a cluster with the `default_pool` settings and optionally connects `kubectl` automatically (add `--skip-connect` to skip).

## Create Node Pools

Create worker pool(s) based on `[[worker_pools]]` entries in your TOML:
```bash
# Create all worker pools defined in the config
python cluster_setup/gke_setup.py --config example create-worker-pool -1

# Create only the first worker pool (index 0)
python cluster_setup/gke_setup.py --config example create-worker-pool 0
```
Pools are named from machine type and provisioning, with an index suffix (e.g., `a3-highgpu-2g-spot-gpu-pool-0`). You can override the base name with `--pool-name`, and autoscaling bounds with `--min-nodes`, `--max-nodes`.

If you want to update an existing pool, add `--delete-first`:
```bash
python cluster_setup/gke_setup.py --config example create-worker-pool 0 --delete-first
```

For a dry-run to preview commands:
```bash
python cluster_setup/gke_setup.py --dry-run --config example create-worker-pool -1
```

## Run PolyRL with Spot Instances

### Start Trainer of PolyRL 
Please refer to `examples/scripts/README.md`.

### Start Rollout Deployment

Please refer to `kube/rust/README.md`.