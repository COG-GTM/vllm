#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
DEST="${1:-/home/ubuntu/work/vllm-cpu-verify}"
if [[ -e "$DEST" ]]; then echo "Destination already exists: $DEST" >&2; exit 1; fi
mkdir -p "$(dirname "$DEST")"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
sudo -n apt-get update
sudo -n apt-get install -y gcc-12 g++-12 libnuma-dev git curl
git clone https://git-manager.devin.ai/proxy/github.com/COG-GTM/vllm "$DEST"
cd "$DEST"
git checkout 98661fe012c5c467252d4df8411d2f46190e9268
uv venv --python 3.12
export VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" CC=gcc-12 CXX=g++-12 MAX_JOBS=4 OMP_NUM_THREADS=4
uv pip install -r requirements/lint.txt
.venv/bin/pre-commit install
uv pip install -e . --torch-backend=cpu
uv pip install -r requirements/test/cuda.in --torch-backend=cpu
uv pip install -e tests/vllm_test_utils
.venv/bin/python -c 'import vllm, torch, pytest, pytest_timeout; from vllm.platforms import current_platform; current_platform.import_kernels(); print("CPU environment ready:", vllm.__version__, torch.__version__, current_platform.device_type)'
.venv/bin/python -m pytest tests/utils_/test_collection_utils.py -q --timeout=300 -p no:cacheprovider
