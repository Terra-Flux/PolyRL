## GKE RL Scaler (Rust)

### Overview
This service exposes a minimal HTTP API to inspect Kubernetes workloads and scale Deployments. It connects to the current kube context (same as `kubectl`) and writes pod logs locally grouped by the pod's `app` label.

### Prerequisites
- Rust toolchain installed (`cargo`)
- `kubectl` configured to point at your cluster
- Cluster and pools can be created with the Python helper: `./cluster_setup/gke_setup.py --config example create-cluster` and `create-rollout-pool`
- Update the `ROLLOUT_MGR` value `http://<your rollout manager address>:5000` in `kube/deployment/kube-rollout-manifest.yaml` to your rollout manager address.
- Update the `<your image name>-test:v1` and `<your image name>-rollout:v1` in `kube/deployment/kube-test-manifest.yaml` and `kube/deployment/kube-rollout-manifest.yaml` 

### Build
```bash
cd rust
cargo build --release
```

### Run
You can run with a short config name (resolved under `config/<name>.toml`) or an absolute path.

```bash
# From repo root
./rust/target/release/gke-rl --config example

# Or using cargo while developing
cargo run -- --config example

# Or pass an explicit path
cargo run -- --config /Users/you/path/to/config/example.toml
```

Environment overrides (optional):
- `BIND_ADDR` (default `0.0.0.0:5000`)
- `LOGS_DIR` (default `pod-logs`)
- `NAMESPACE` (default `default`)

Example with overrides:
```bash
BIND_ADDR=0.0.0.0:5001 LOGS_DIR=logs/example NAMESPACE=default cargo run -- --config example
```

If you specify `[deployments]` in the `example.toml`, it will automatically apply manifest before start the API service.

### API
Base URL: `http://<host>:<port>` (default `http://0.0.0.0:5000`)

- Health
```bash
curl localhost:5000/health
```

- List Deployments (in configured namespace)
```bash
curl localhost:5000/deployments
```

- List Pods (in configured namespace)
```bash
curl localhost:5000/pods
```

- Scale a Deployment
```bash
curl -X POST localhost:5000/scale \
  -H 'content-type: application/json' \
  -d '{"app":"rollout-app","replicas":2}'
```
Replace `rollout-app` with the Deployment name you want to scale (e.g., the deployment defined in `deployment/kube-rollout-manifest.yaml`).

### Logs
The service tails logs from running pods in the configured namespace and stores them under `LOGS_DIR/<app>/<pod>.log`.

Notes:
- `<app>` is derived from the pod label `app`. If missing, logs are written under `unknown/`.
- The directory structure is created automatically.

### Configuration
When `--config <name-or-path>` is provided, the service reads TOML and uses the following fields:
- `[general].namespace`
- `[general].bind_addr`
- `[general].logs_dir`

Example: `config/example.toml` includes these keys under `[general]`.

### Troubleshooting
- Ensure `kubectl get pods` works in the same shell; the scaler uses the same kube context.
- If you see RBAC errors, make sure your user/service account has permissions to list pods/deployments and patch deployments in the target namespace.

