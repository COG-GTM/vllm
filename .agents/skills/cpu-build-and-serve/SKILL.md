---
name: cpu-build-and-serve
description: How to build vLLM with CPU support and run the OpenAI-compatible backend server on a CPU-only machine.
---

# vLLM CPU Build & Serve

## System Dependencies

The CPU backend requires gcc >= 12.3 for x86_64. Install these system packages:

```bash
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install -y gcc-13 g++-13 libnuma-dev ninja-build
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 --slave /usr/bin/g++ g++ /usr/bin/g++-13
```

Also ensure cmake >= 3.26 is available:
```bash
uv pip install cmake>=3.26
```

## Building CPU Extensions

The default precompiled install (`VLLM_USE_PRECOMPILED=1`) only ships CUDA extensions. To build CPU C extensions from source:

```bash
source .venv/bin/activate
CC=gcc-13 CXX=g++-13 VLLM_TARGET_DEVICE=cpu python setup.py develop
```

This compiles `_C.abi3.so`, `_C_AVX512.abi3.so`, and `_C_AVX2.abi3.so` into the `vllm/` directory.

## Starting the Server

```bash
VLLM_TARGET_DEVICE=cpu vllm serve facebook/opt-125m \
  --dtype float32 \
  --max-model-len 256 \
  --enforce-eager \
  --port 8000 \
  --gpu-memory-utilization 0.5
```

### Key Flags

- `VLLM_TARGET_DEVICE=cpu` — required env var to select CPU backend
- `--dtype float32` — CPU does not support float16/bfloat16 on all platforms
- `--enforce-eager` — disables `torch.compile` (avoids Triton/CUDA dependency)
- `--gpu-memory-utilization 0.5` — controls CPU memory fraction; default 0.92 may OOM on VMs with < 16GB
- `--max-model-len 256` — reduce context length to lower memory usage

### Recommended Test Model

`facebook/opt-125m` (~350MB) is small enough for CPU inference demos.

## Verifying the Server

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Text completion
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "facebook/opt-125m", "prompt": "Hello, my name is", "max_tokens": 20}'

# Version
curl http://localhost:8000/version
```

## Notes

- Chat completions (`/v1/chat/completions`) requires a model with a chat template (opt-125m does not have one)
- CPU inference is significantly slower than GPU — expect seconds per request
- The server exposes an OpenAI-compatible API on the configured port
