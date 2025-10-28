# Start from the official NVIDIA base image, default WORKDIR is /workspace
FROM nvcr.io/nvidia/pytorch:25.05-py3

# Set up the environment to be non-interactive and disable Python's output buffering
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Layer 1: System dependencies and cleanup
# This layer is cached unless you need to add more apt packages.
RUN apt-get update && \
    # Fix: Corrected typo from ldconfg to ldconfig
    ldconfig && \
    # Best Practice: Clean up apt cache to reduce image size
    rm -rf /var/lib/apt/lists/*

# Layer 3: Clone the repository and install the project-specific dependencies
# This layer will only be re-run if the git repo or project requirements change.
RUN git clone -b weight-transfer https://github.com/Terra-Flux/PolyRL.git && \
    cd PolyRL/src/sglang && \
    pip install -e "python[all]"

# Layer 2: Install standalone Python packages
# This is cached unless you change these specific dependencies.
RUN pip install flash_attn rpyc mooncake-transfer-engine --no-build-isolation && \
    pip cache purge

# Layer 4: Copy local files
# This is one of the last steps, so changes to the script don't invalidate previous layers.
COPY launch_sglang.sh .
COPY launch_rollout.sh .
