# PolyRL

> **PolyRL** is a polymorphic reinforcement learning (RL) framework for large language models (LLM). Its unique poly-architecture disaggregates the rollout and update stages, strategically fitting them into heterogeneous hardware configurations to maximize cost efficiency. By utilizing elastic control, PolyRL dynamically scales resources, embodying its adaptable nature. Our goal is to create a portable and affordable fine-tuning platform accessible to everyone.

## Installation

Please refer to [INSTALL.md](INSTALL.md) for installation instructions.

## Usage

Please refer to [examples/scripts/README.md](examples/scripts/README.md) for usage instructions.

## Roadmap (subject to change)

### Release-v0: Basic disaggregated RL system

Goal: 
1. Rollout instances running on dynamic independent hardware. 
2. Rollout instances stream results to training engine via async-efficient manager process. 
3. Weight update via TCP&RDMA aggregated channel.

- Rollout
  - Decouple rollout and update
    - [x] Training engine send requests to SGLang server via API.
    - [x] Rank-zero send all generation requests and skip local generation.
    - [x] Wrap rollout results streaming into an iterator.
  - Multiple rollout instances management
    - Rollout manager in Rust
      - [x] Rollout instance register request via API.
      - [x] Relay requests from training engine to rollout instances.
      - [x] Interface of algorithm-driven request scheduling.
  - Rollout instances dynamic in-n-out
    - [x] Active new rollout instance register during runtime.
    - [x] Shutdown connection when rollout instance goes down. 
- Training
  - From batch process to stream process 
    - [x] Align micro-batch size along rollout-actor-critic.
    - [x] Reward, KL, advantage in micro-batch.
    - [x] Critic and actor for/backward in micro-batch and update in mini-batch.
    - [ ] Align runtime metric collection with micro-batch processing.
- Weight transfer
  - TCP&RDMA aggregated interface
    - [x] Integrate Mooncake transfer engine.
    - [x] Weight transfer agent for each instance.
  - Model weight gather and re-shard
    - [x] Rank-zero gathers weights from FSDP ranks and call agent to transfer.
    - [x] Support weight resharding on TP>1 rollout instances.
  - Compression interface
    - [x] Encode and decode weight interface to support compressed weight update.
    - [ ] Asynchronous full weight update interface.

### Release v1: Elastic and auto-balanced resource management
Goal:
1. Maximize the utilization of heterogeneous resources.
2. Adaptive resource allocation to align the extending rollout time.
3. Improve robustness to handle dynamic availability.

- Intra-stage (rollout) balance
  - Rollout workload balance, start from homogenous rollout instances.
    - [ ] Round-robin request assignment.
    - [ ] Per-sample tracking and workload balance (need to check if affect the training progress)
    - [ ] Decouple data plane and control plane (replicate requests to all rollout instances and send control message during rollout).  
- Inter-stage (rollout vs. training) balance
  - Rollout buffer zone
    - [ ] Estimate the time gap between update finish and rollout ready.
    - [ ] Training engine rollout locally before receiving streamed batches.
  - Dynamic rollout instances allocation
    - [ ] Add rollout instances at runtime when rollout time extend.
- Pack sequences in rollout manager
  - [ ] Send rollout prompts to manager in a batch.
  - [ ] Rollout manager decompose into per-sample requests and send to rollout instances. 
  - [ ] Manage the order of rollout results and pack into micro-batches.
  - [ ] Dynamic batch size with a lower bound (block if under; return all when asked).
- Fault tolerance 
  - Handle multiple failure cases
    - [ ] Spot instance preemption
    - [ ] Failure during weight transfer
    - [ ] Failure during rollout
- Weight compression
  - [ ] Quantization+lossless compression to reduce the size of weight before transfer.

## Known issues

