# Start from the official NVIDIA base image
FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Set up the environment to be non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# *** THE FIX: Add this environment variable to disable output buffering ***
ENV PYTHONUNBUFFERED=1

# Install Python, pip
RUN apt-get update && apt-get install -y \
    python3-pip python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Create an /app directory for our code
WORKDIR /app

# Install PyTorch for CUDA 12.8 in a single, efficient layer
RUN python3 -m venv llm && \
    llm/bin/pip3 install torch torchvision

# Copy the Python script into the image
COPY test_script.py .
