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
from setuptools import setup, find_packages

# Core dependencies required by all modes
install_requires = [
    "rpyc",
    "requests",
    "aiohttp",
    "numpy",
    "psutil",
    "pybase64",
    "transformers"
]

# Optional dependencies
extras_require = {
    "gpu": ["torch", "accelerate"],
    "tpu": ["jax", "jaxlib", "flax", "wrapt", "tpu-info"],
}

# 'all' includes everything
extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup(
    name="utrans-engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
    description="Unified Transfer Engine for high-speed weight synchronization between GPU and TPU.",
    author="PolyRL Team",
    python_requires=">=3.12",
)
