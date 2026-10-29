# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind two unchanged freeze-base CPU kernels; no replacement implementation."""

import os
from pathlib import Path

from torch.utils.cpp_extension import load

repo = Path.cwd()
build = repo / ".venv" / "cpu-kernels"
build.mkdir(exist_ok=True)
os.environ.setdefault("MAX_JOBS", "2")
load(
    name="vllm_eval_cpu_kernels",
    sources=[
        str(Path(__file__).with_name("cpu_kernels.cpp")),
        str(repo / "csrc/cpu/spec_decode_utils.cpp"),
    ],
    extra_cflags=["-O2", "-mavx2", "-mfma", "-mf16c", "-fopenmp"],
    extra_ldflags=["-fopenmp"],
    build_directory=str(build),
    is_python_module=False,
    verbose=True,
)
