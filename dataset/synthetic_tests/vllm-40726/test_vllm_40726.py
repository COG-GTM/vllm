# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Execute the real FX split/codegen pipeline without imposing its strategy."""

import io

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from vllm.compilation.backends import split_graph
from vllm.compilation.codegen import compile_execution_fn, generate_execution_code


def compile_graph(graph, *, cached=False):
    artifact = generate_execution_code(graph)
    if cached:
        # Simulates serialization of an execution artifact without inspecting
        # its representation.  Both the original 2-part and extended artifacts
        # are accepted; only execution behavior is asserted.
        buffer = io.BytesIO()
        torch.save(artifact, buffer)
        buffer.seek(0)
        artifact = torch.load(buffer, weights_only=False)
    callables = dict(graph.named_children())
    return compile_execution_fn(artifact[0], callables, *artifact[1:])


def conversion(x):
    return x.to(device=torch.device("cpu"), dtype=torch.float16)


def allocation(x):
    return torch.zeros(x.shape, device=torch.device("cpu"), dtype=torch.float16) + x


def repeated_conversion(x):
    return x.to(device="cpu", dtype=torch.float16) + (2 * x).to(
        device="cpu", dtype=torch.float16
    )


def split_conversion(x):
    half = x.to(device="cpu", dtype=torch.float16)
    return half.relu().to(dtype=torch.float32).to(dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "model,split_ops",
    [
        (conversion, ["aten::_to_copy.default"]),
        (allocation, []),
        (repeated_conversion, []),
        (split_conversion, ["aten::relu.default"]),
    ],
    ids=["conversion", "allocation", "repeated_objects", "split_submodules"],
)
@pytest.mark.parametrize("cached", [False, True], ids=["fresh", "serialized"])
def test_generated_execution_matches_eager(model, split_ops, cached):
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3) - 2
    graph, _ = split_graph(make_fx(model)(x), split_ops)
    compiled = compile_graph(graph, cached=cached)
    # Reuse the callable with different values, not only the trace example.
    for value in [x, x + 1]:
        expected = model(value)
        actual = compiled(value)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device


@pytest.mark.parametrize("cached", [False, True])
def test_nested_nonprimitive_graph_outputs(cached):
    from torch.distributed.tensor import Replicate, Shard

    # Exercise nested argument containers through the real code generator,
    # rather than invoking or asserting the format of a private ref helper.
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    cpu = torch.device("cpu")
    graph.output(
        (x, [cpu, 1], (cpu,), {"dtype": torch.float16}, [Shard(0), Replicate()])
    )
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    result = compile_graph(gm, cached=cached)(torch.ones(2))
    torch.testing.assert_close(result[0], torch.ones(2))
    assert result[1] == [cpu, 1]
    assert result[2] == (cpu,)
    assert result[3] == {"dtype": torch.float16}
    assert result[4] == [Shard(0), Replicate()]


@pytest.mark.parametrize("cached", [False, True])
def test_primitive_graph_still_executes(cached):
    def model(x):
        return x.transpose(0, 1).relu() + 2

    x = torch.arange(6, dtype=torch.float32).reshape(2, 3) - 2
    gm, _ = split_graph(make_fx(model)(x), [])
    torch.testing.assert_close(compile_graph(gm, cached=cached)(x), model(x))


def test_non_graphmodule_child_keeps_callable_behavior():
    def model(x):
        return x.relu().sigmoid()

    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *args, **kwargs):
            return self.inner(*args, **kwargs)

    x = torch.tensor([-2.0, 1.0])
    gm, _ = split_graph(make_fx(model)(x), ["aten::relu.default"])
    name, child = next(iter(gm.named_children()))
    gm.add_module(name, Wrapper(child))
    torch.testing.assert_close(compile_graph(gm)(x), model(x))


def test_legacy_artifact_without_special_objects_still_loads():
    code = (
        "import torch\n"
        "def execution_fn(x, *, __vllm_submods__):\n"
        "    return __vllm_submods__[0](x) + 1\n"
    )
    fn = compile_execution_fn(code, {"sub": torch.nn.ReLU()}, ["sub"])
    torch.testing.assert_close(fn(torch.tensor([-2.0, 2.0])), torch.tensor([1.0, 3.0]))
