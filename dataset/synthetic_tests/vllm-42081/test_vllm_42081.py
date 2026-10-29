# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU regression: real Kimi projector orchestration and projector forward.

Only the distributed linear-layer construction/kernel boundary is substituted.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.model_executor.models.kimi_k25_vit as kimi


class CPULinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, value):
        return self.linear(value), None


@pytest.fixture
def projector(monkeypatch):
    monkeypatch.setattr(kimi, "ReplicatedLinear", CPULinear)
    with torch.random.fork_rng():
        torch.manual_seed(144)
        return kimi.KimiK25MultiModalProjector(
            SimpleNamespace(hidden_size=8, merge_kernel_size=(2, 2), mm_hidden_size=16)
        ).eval()


@pytest.mark.parametrize(
    "input_dtype,parameter_dtype",
    [
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float64, torch.float32),
    ],
    ids=["fp32_to_bf16", "bf16_to_fp32", "fp64_to_fp32"],
)
def test_projector_accepts_vision_dtype(projector, input_dtype, parameter_dtype):
    projector.to(parameter_dtype)
    features = [
        torch.arange(n * 32).reshape(n, 4, 8).to(input_dtype) / 64 for n in (3, 2)
    ]
    originals = [x.clone() for x in features]
    weights = {n: p.detach().clone() for n, p in projector.named_parameters()}
    with torch.inference_mode():
        expected = [projector(x.to(parameter_dtype)) for x in features]
        actual = kimi.mm_projector_forward(projector, features)
    assert [tuple(x.shape) for x in actual] == [(3, 16), (2, 16)]
    for got, want in zip(actual, expected):
        assert got.dtype == parameter_dtype
        torch.testing.assert_close(got, want)
    for before, after in zip(originals, features):
        assert torch.equal(before, after)
    for name, parameter in projector.named_parameters():
        assert parameter.dtype == parameter_dtype
        assert torch.equal(parameter, weights[name])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_matching_dtype_keeps_batched_storage(projector, monkeypatch, dtype):
    projector.to(dtype)
    features = [torch.ones(n, 4, 8, dtype=dtype) for n in (3, 2)]
    batched = []
    received = []
    cat = torch.cat

    def record_cat(*args, **kwargs):
        result = cat(*args, **kwargs)
        batched.append(result)
        return result

    monkeypatch.setattr(kimi.torch, "cat", record_cat)
    hook = projector.register_forward_pre_hook(
        lambda module, args: received.append(args[0])
    )
    try:
        output = kimi.mm_projector_forward(projector, features)
    finally:
        hook.remove()
    # Accept either batched or per-image forwarding; do not prescribe the
    # batching strategy or number of projector calls for an equivalent fix.
    allowed_storage = {x.data_ptr() for x in features + batched}
    assert received
    assert all(x.data_ptr() in allowed_storage for x in received)
    assert [tuple(x.shape) for x in output] == [(3, 16), (2, 16)]
    assert all(x.dtype == dtype for x in output)
