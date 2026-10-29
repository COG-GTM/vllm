# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise the real launch adapters and compiled freeze-base CPU kernels."""

import sys
from pathlib import Path

import pytest
import torch

import vllm.utils.cpu_triton_utils as cpu


@pytest.fixture(scope="module", autouse=True)
def native_kernels():
    path = Path(sys.prefix) / "cpu-kernels/vllm_eval_cpu_kernels.so"
    assert path.is_file(), "Run harness/setup_cpu_env.sh before evaluation"
    torch.ops.load_library(str(path))


def common():
    return (
        torch.full((2, 3), -1, dtype=torch.int32),
        torch.tensor([2, 4], dtype=torch.int32),
        torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        torch.tensor([[5], [6]], dtype=torch.int32),
        torch.tensor([True, True]),
    )


@pytest.mark.parametrize("new_call", [False, True], ids=["legacy", "shared-sampler"])
def test_greedy_dispatch_and_copyback(new_call):
    out, cu, draft, bonus, is_greedy = common()
    target = torch.tensor([1, 2, 0, 0], dtype=torch.int32)
    args = (out, cu, draft, target, bonus, is_greedy, 2)
    if new_call:
        cpu.rejection_greedy_sample_kernel[(2,)](
            *args, None, None, SYNTHETIC_MODE=False
        )
    else:
        cpu.rejection_greedy_sample_kernel[(2,)](*args)
    assert out.dtype == torch.int32
    torch.testing.assert_close(
        out, torch.tensor([[1, 2, 5], [0, -1, -1]], dtype=out.dtype)
    )


@pytest.mark.parametrize("new_call", [False, True], ids=["legacy", "shared-sampler"])
@pytest.mark.parametrize(
    "no_draft", [False, True], ids=["draft-probs", "no-draft-probs"]
)
def test_random_dispatch_and_copyback(new_call, no_draft):
    out, cu, draft, bonus, is_greedy = common()
    is_greedy.zero_()
    target_probs = torch.full((4, 8), 0.125)
    draft_probs = None if no_draft else target_probs.clone()
    recovered = torch.tensor([2, 3, 4, 5], dtype=torch.int32)
    uniform = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    args = (
        out,
        cu,
        draft,
        draft_probs,
        target_probs,
        bonus,
        recovered,
        uniform,
        is_greedy,
        2,
        8,
    )
    if new_call:
        cpu.rejection_random_sample_kernel[(2,)](
            *args, None, NO_DRAFT_PROBS=no_draft, SYNTHETIC_MODE=False
        )
    else:
        cpu.rejection_random_sample_kernel[(2,)](*args, NO_DRAFT_PROBS=no_draft)
    assert out.dtype == torch.int32
    expected = [[1, 3, -1], [4, -1, -1]] if no_draft else [[1, 2, 5], [3, 4, 6]]
    torch.testing.assert_close(out, torch.tensor(expected, dtype=out.dtype))


@pytest.mark.parametrize("kind", ["greedy", "random"])
def test_synthetic_acceptance_is_explicitly_unsupported(kind):
    out, cu, draft, bonus, is_greedy = common()
    with pytest.raises(
        (AssertionError, RuntimeError, ValueError, NotImplementedError),
        match="(?i)synthetic",
    ):
        if kind == "greedy":
            cpu.rejection_greedy_sample_kernel[(2,)](
                out,
                cu,
                draft,
                draft,
                bonus,
                is_greedy,
                2,
                torch.full((4,), 0.5),
                torch.ones(2),
                SYNTHETIC_MODE=True,
            )
        else:
            cpu.rejection_random_sample_kernel[(2,)](
                out,
                cu,
                draft,
                None,
                torch.full((4, 8), 0.125),
                bonus,
                draft,
                torch.full((4,), 0.5),
                is_greedy,
                2,
                8,
                torch.ones(2),
                NO_DRAFT_PROBS=True,
                SYNTHETIC_MODE=True,
            )
    torch.testing.assert_close(out, torch.full_like(out, -1))
