# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import os

print("Launching CUDA test")

import torch

# Check if CUDA is available
if not torch.cuda.is_available():
    print('!!! FATAL: CUDA is not available to PyTorch. Exiting.')
    exit(1)

# Set the device to the first GPU
device = torch.device('cuda:0')
pod_name = os.getenv('HOSTNAME', 'unknown-pod')

print(f'🚀 CUDA test running on pod: {pod_name}')
print(f'--> Using device: {torch.cuda.get_device_name(0)}')

while True:
    try:
        # Create two random matrices on the GPU
        a = torch.randn(4096, 4096, device=device)
        b = torch.randn(4096, 4096, device=device)
        
        # Perform matrix multiplication
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Performing 4096x4096 matrix multiplication...')
        
        start_time = time.time()
        c = torch.matmul(a, b)
        
        # Synchronize to get accurate timing
        torch.cuda.synchronize()
        end_time = time.time()
        
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Computation complete in {(end_time - start_time):.4f} seconds.')
    
    except Exception as e:
        print(f'An error occurred: {e}')
    
    # Wait for 10 seconds before the next computation
    print("...sleeping for 10 seconds...")
    time.sleep(10)
