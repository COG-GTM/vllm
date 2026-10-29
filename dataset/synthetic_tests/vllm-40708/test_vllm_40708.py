# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real Gemma4Model.load_weights with a small parameter-tree boundary double."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.gemma4 import Gemma4Model


class ParameterTree(nn.Module):
    def __init__(self, parameter_names):
        super().__init__()
        self.config = SimpleNamespace(num_experts=2)
        self.quant_config = None
        self.calls: list[tuple] = []
        layer = nn.Module()
        layer.moe = nn.Module()
        layer.moe.experts = nn.Module()
        self.layers = nn.ModuleList([layer])
        for name in parameter_names:
            parameter = nn.Parameter(torch.zeros(1), requires_grad=False)

            def record(
                param,
                value,
                mapped_name=None,
                shard_id=None,
                expert_id=None,
                *,
                key=name,
            ):
                self.calls.append((key, param, value, mapped_name, shard_id, expert_id))

            parameter.weight_loader = record
            setattr(layer.moe.experts, name, parameter)


@pytest.mark.parametrize("expert", [0, 1])
@pytest.mark.parametrize(
    "projection,fused,shard",
    [
        ("gate_proj", "w13", "w1"),
        ("up_proj", "w13", "w3"),
        ("down_proj", "w2", "w2"),
    ],
)
@pytest.mark.parametrize(
    "checkpoint_suffix,parameter_suffix",
    [
        ("_packed", "weight_packed"),
        ("_scale", "weight_scale"),
        ("_zero_point", "weight_zero_point"),
        (".qweight", "qweight"),
        (".scales", "scales"),
        (".qzeros", "qzeros"),
        (".weight", "weight"),
        ("", "weight"),
    ],
    ids=[
        "packed",
        "scale",
        "zero_point",
        "qweight",
        "scales",
        "qzeros",
        "weight",
        "bare",
    ],
)
def test_checkpoint_expert_routes_to_parameter_and_shard(
    expert, projection, fused, shard, checkpoint_suffix, parameter_suffix
):
    parameter_name = f"{fused}_{parameter_suffix}"
    model = ParameterTree([parameter_name])
    value = torch.tensor([expert + 1.25])
    checkpoint_name = f"layers.0.moe.experts.{expert}.{projection}{checkpoint_suffix}"
    mapped_name = f"layers.0.moe.experts.{parameter_name}"
    loaded = Gemma4Model.load_weights(model, iter([(checkpoint_name, value)]))
    assert mapped_name in loaded
    assert len(model.calls) == 1
    key, param, received, name, received_shard, received_expert = model.calls[0]
    assert key == parameter_name
    assert param is dict(model.named_parameters())[mapped_name]
    torch.testing.assert_close(received, value)
    assert name == mapped_name
    assert received_shard == shard
    assert received_expert == expert


def test_nonexpert_unquantized_parameter_still_loads():
    model = ParameterTree([])
    model.norm = nn.LayerNorm(2)
    value = torch.tensor([1.5, 2.5])
    loaded = Gemma4Model.load_weights(model, iter([("norm.weight", value)]))
    assert "norm.weight" in loaded
    torch.testing.assert_close(model.norm.weight, value)
