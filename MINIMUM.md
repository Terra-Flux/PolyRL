# Minimum Setup

Step-by-step guide to validate basic funtionality of FluidRL. 
> NOTE: this is just a minimum toy example, its performance doesn't reflect the real performance of FluidRL.

### Clone repo

```bash
git clone https://github.com/Terra-Flux/PolyRL.git --recursive polyrl
cd polyrl
git checkout artifact
# disable logging in veRL as it will impact performance
sed -i 's/"log_level": "info",/# "log_level": "info",/' 3rdparty/verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py
```

### Hardware

Assume at least **a 4 GPUs instance** with >=40GB VRAM each.

### Environment Setup

#### Docker

Install Docker and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian).
For explanations, refer to [docker](docker/README.md).

Build images:
```bash
docker build -t polyrl:trainer -f docker/Dockerfile .
```

Run containers:

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
  -d -it --rm polyrl:trainer
```

Enter container:
```bash
docker exec -it polyrl-trainer bash
```

Python modules are installed in a virtual environment, enter venv by `source /opt/venv/bin/activate`.

### Installation Validation

You can validate the installation by running
```bash
source /opt/venv/bin/activate
bash examples/scripts/launch_sglang_test.sh
```

If you see `The server is fired up and ready to roll!`, the installation is successful.

### Launch Experiment

Launch training engine:
```bash
bash examples/scripts/run_async_grpo_pipeline_mini.sh
```

When you see the following prompt in red 🔴: 
```bash
[SGLangRollout] Rollout manager is ready at http://xxx.xxx.xxx.xxx:5000
```
Start rollout engines.

Launch the first rollout engine:
```bash
CUDA_VISIBLE_DEVICES=2,3 bash examples/scripts/launch_sglang_mini.sh 4000 20000
```

### Expected Ouput

#### Initialization
On the rollout engine, you will observe outputs like 
```bash
- rlboost.sglang.autopatch - INFO - Auto-applying rlboost patches to sglang...
- rlboost.sglang.autopatch - INFO - Successfully applied patch: ServerArgsPatch
...
```

When the rollout engine is ready, it will receive a request to update weight, the output looks like 
```bash
INFO receiver_agent.py:88: Initialized TCP engine with 8 parallel streams
INFO receiver_agent.py:111: Successfully bound ZMQ socket to tcp://0.0.0.0:41911
INFO receiver_agent.py:155: [Weight receiver] ZMQ listener thread started successfully on port 41911.
INFO transfer_engine.py:54: TCPTransferEngine registered buffer: ptr=139601060065280, length=8126959616
INFO transfer_engine.py:88: TCPTransferEngine started 8 listeners on ports [38631, 39329, 42735, 35571, 33513, 33677, 34615, 38987]
INFO receiver_agent.py:286: Transfer buffer allocated and placed in output queue.
INFO receiver_agent.py:192: Attempting to connect to sender RPyC server at xxx:18861...
...
Successfully started receiver transfer agent process.
INFO:     xxx:xxx - "POST /update_weights_from_agent HTTP/1.1" 200 OK
INFO transfer_engine.py:139: Received 1015869952 bytes at offset 5079349760 from xxx:xxx-L5
INFO transfer_engine.py:139: Received 1015869952 bytes at offset 7111089664 from xxx:xxx-L7
INFO transfer_engine.py:139: Received 1015869952 bytes at offset 2031739904 from xxx:xxx-L2
INFO transfer_engine.py:139: Received 1015869952 bytes at offset 0 from xxx:xxx-L0
INFO transfer_engine.py:139: Received 1015869952 bytes at offset 1015869952 from xxx:xxx-L1
INFO transfer_engine.py:139: Received 1015869952 bytes at offset 3047609856 from xxx:xxx-L3
INFO transfer_engine.py:139: Received 1015869952 bytes at offset 6095219712 from xxx:xxx-L6
INFO transfer_engine.py:139: Received 1015869952 bytes at offset 4063479808 from xxx:xxx-L4
INFO:     xxx:xxx - "POST /update_weights_from_agent HTTP/1.1" 200 OK
```

On the trainer side, you will also observe output like:
```bash
Received registration request for instance: http://xxx.xxx.xxx.xxx:4000 (id: xxx)
Starting health check for instance: http://xxx.xxx.xxx.xxx:4000
Health check passed for instance: http://xxx.xxx.xxx.xxx:4000
Instance http://xxx.xxx.xxx.xxx:4000 is now ready and added to instances list (id: xxx) with weight sender xxx:xxx
Updating weights for 1 instances to version 1
Weight update successful for instance http://xxx.xxx.xxx.xxx:4000
Weight update completed successfully for all instances
Updating weights for 1 instances to version 1
Weight update successful for instance http://xxx.xxx.xxx.xxx:4000
Adding 1 instances to active pool
Weight update completed successfully for all instances
```
During the weight update process, the trainer will do rollout to maximize resource utilization.

#### Training Process
After weight update is done, the rollout manager will start assigning requests to the rollout engine, and the rollout engine will print log messages like
```bash
- "POST /generate HTTP/1.1" 200 OK
- "POST /generate HTTP/1.1" 200 OK
- "POST /generate HTTP/1.1" 200 OK
```
which means the rollout engine has started to work on generation.

At some point, you will see a bunch of messages like,
```bash
Stream failed on instance http://xxx.xxx.xxx.xxx:30000 due to stream_abort [ignore stream_abort], removing instance and attempting to continue with another instance
```

Don't panic, that means the trainer has switched to training mode, and its unfinished requests will be continued on the active rollout engines.

The rollout results are streamed to the trainer, you will see outputs like
```bash
--- Received Batch #0 with 17 prompt_responses ---
[SGLangRollout] Accumulated responses len(acc_responses)=136 > stream_size=16, process 128 responses
--- Received Batch #1 with 1 prompt_responses ---
[SGLangRollout] Accumulated responses len(acc_responses)=16 > stream_size=16, process 16 responses
```
This is because the training are done in micro-batches.

#### Model Update 
After a step of training is done, the trainer will output training progress as
```bash
step:1 - actor/entropy:0.28172561526298523 - ... - perf/total_num_tokens:1345358 - perf/time_per_step:1065.5507174460217 - perf/throughput:631.2970269611557 - perf/throughput_all_gpus:1262.5940539223113
```
The `perf/throughput_all_gpus` is the average throughput of the last training step in tokens/s.

#### Next Step

If you have more GPUs, you can continue to add more rollout engines by running 
```bash
CUDA_VISIBLE_DEVICES=4,5 bash examples/scripts/launch_sglang_mini.sh 4001 21000
```
It will go through the same weight update process and start to generate sequences.



