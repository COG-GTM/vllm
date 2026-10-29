#!/usr/bin/env bash
# Build the evaluated Python/CPU environment from a fresh freeze-base clone.
# Observed on this 8-core Linux VM, with warm uv/download caches:
# uv venv: <1s; locked dependencies: 0.312s; editable vLLM: 28.745s;
# native CPU kernels: 12.832s (two unchanged freeze-base ops, no GPU build).
# Full fresh-venv rerun of this script with final pins: 46.910s (warm caches).
# Cold-download timings were not measured. Requires uv 0.13.6, a C++17
# compiler with OpenMP, and an AVX2 x86_64 Linux host.
set -euo pipefail

HARNESS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE=98661fe012c5c467252d4df8411d2f46190e9268
if [[ "$(git rev-parse HEAD)" != "$BASE" ]]; then
    echo "Run from a fresh clone checked out at $BASE" >&2
    exit 1
fi
command -v uv >/dev/null
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python \
    -r "$HARNESS_DIR/cpu-requirements.txt" --torch-backend=auto
VLLM_TARGET_DEVICE=cpu VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_COMMIT="$BASE" \
    uv pip install --python .venv/bin/python -e . --torch-backend=auto --no-deps
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python "$HARNESS_DIR/build_cpu_kernels.py"

# Test runs set these too; setup may download packages/wheels, tests may not.
VLLM_TARGET_DEVICE=cpu HF_HUB_OFFLINE=1 .venv/bin/python - <<'PY'
import torch
import vllm

assert torch.version.cuda is None
assert not torch.cuda.is_available()
print(f"CPU environment ready: torch={torch.__version__}, vllm={vllm.__version__}")
PY
