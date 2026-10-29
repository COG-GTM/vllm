# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Public pooling conversion interface plus actual embedding response building."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.responses import JSONResponse, ORJSONResponse

import vllm.entrypoints.pooling.utils as utils
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding


def output(tensor):
    return SimpleNamespace(
        outputs=SimpleNamespace(data=tensor), prompt_token_ids=[1, 2]
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"]
)
def test_orjson_float_endpoint_preserves_schema_and_uses_fast_representation(
    monkeypatch, dtype
):
    tensors = [
        torch.tensor([1.0, 2.5], dtype=dtype),
        torch.tensor([-3.0, 0.25], dtype=dtype),
    ]
    seen = []
    render = ORJSONResponse.render

    def record(self, content):
        seen.append(content)
        return render(self, content)

    monkeypatch.setattr(ORJSONResponse, "render", record)
    server = ServingEmbedding.__new__(ServingEmbedding)
    server.json_response_cls = ORJSONResponse
    response = server._openai_json_response(
        [output(t) for t in tensors],
        "embd-test",
        123,
        "m",
        "float",
        "float32",
        "native",
    )
    body = json.loads(response.body)
    assert body["id"] == "embd-test" and body["created"] == 123 and body["model"] == "m"
    assert body["object"] == "list"
    assert body["usage"]["prompt_tokens"] == 4 and body["usage"]["total_tokens"] == 4
    assert body["data"] == [
        dict(object="embedding", index=i, embedding=t.tolist())
        for i, t in enumerate(tensors)
    ]
    for item, tensor in zip(seen[0]["data"], tensors):
        if dtype == torch.float32:
            assert isinstance(item["embedding"], np.ndarray)
            assert np.shares_memory(item["embedding"], tensor.numpy())
        else:
            assert item["embedding"] == tensor.tolist()


def test_standard_json_float_response_is_unchanged():
    tensor = torch.tensor([1.0, -2.5, 3.25])
    server = ServingEmbedding.__new__(ServingEmbedding)
    server.json_response_cls = JSONResponse
    response = server._openai_json_response(
        [output(tensor)], "embd-test", 123, "m", "float", "float32", "native"
    )
    assert json.loads(response.body)["data"][0]["embedding"] == tensor.tolist()
    assert utils.encode_pooling_output_float(output(tensor)) == tensor.tolist()


def test_base64_response_unchanged_for_orjson():
    tensor = torch.tensor([1.0, -2.5, 3.25])
    server = ServingEmbedding.__new__(ServingEmbedding)
    server.json_response_cls = ORJSONResponse
    response = server._openai_json_response(
        [output(tensor)], "embd-test", 123, "m", "base64", "float32", "native"
    )
    import pybase64 as base64

    encoded = json.loads(response.body)["data"][0]["embedding"]
    assert base64.b64decode(encoded) == tensor.numpy().tobytes()
