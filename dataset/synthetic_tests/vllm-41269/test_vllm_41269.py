# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Call real public endpoint methods; substitute engine/model/renderer boundaries."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from starlette.requests import Request

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

KV = {
    "do_remote_prefill": True,
    "remote_engine_id": "prefill-node",
    "remote_block_ids": [3, 9],
}
ENDPOINTS = ["completion", "chat", "responses"]


@pytest.fixture(autouse=True)
def fixed_validation_environment(monkeypatch):
    monkeypatch.setenv("VLLM_SKIP_MODEL_NAME_VALIDATION", "0")
    monkeypatch.setenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "0")


def endpoint(kind, kv=KV, has_connector=True):
    cls, request_cls, method, fields = {
        "completion": (
            OpenAIServingCompletion,
            CompletionRequest,
            "create_completion",
            {"prompt": "hi"},
        ),
        "chat": (
            OpenAIServingChat,
            ChatCompletionRequest,
            "create_chat_completion",
            {"messages": [{"role": "user", "content": "hi"}]},
        ),
        "responses": (
            OpenAIServingResponses,
            ResponsesRequest,
            "create_responses",
            {"input": "hi"},
        ),
    }[kind]
    engine = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=32, is_encoder_decoder=False),
        renderer=SimpleNamespace(tokenizer=object()),
        input_processor=None,
        vllm_config=SimpleNamespace(
            kv_transfer_config=object() if has_connector else None
        ),
        notify_kv_transfer_request_rejected=AsyncMock(),
        errored=False,
        is_tracing_enabled=AsyncMock(return_value=False),
        generate=Mock(),
    )
    models = SimpleNamespace(
        is_base_model=Mock(return_value=False),
        lora_requests={},
        model_name=Mock(return_value="m"),
    )
    server = cls.__new__(cls)
    # Real base initialization discovers connector support from engine configuration.
    OpenAIServing.__init__(server, engine, models, request_logger=None)
    server.chat_template = None
    server.chat_template_content_format = "auto"
    server.default_chat_template_kwargs = {}
    server.reasoning_parser_cls = None
    request = request_cls(
        model="m", request_id="original-id", kv_transfer_params=kv, **fields
    )
    raw = Request({"type": "http", "headers": [(b"x-data-parallel-rank", b"2")]})
    return server, request, raw, getattr(server, method)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ENDPOINTS)
async def test_rejected_remote_prefill_notifies_with_original_identity(kind):
    server, request, raw, call = endpoint(kind)
    result = await call(request, raw)
    assert isinstance(result, ErrorResponse) and result.error.code == 404
    assert "does not exist" in result.error.message
    server.engine_client.notify_kv_transfer_request_rejected.assert_awaited_once_with(
        "original-id", KV, data_parallel_rank=2
    )
    server.engine_client.generate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ENDPOINTS)
async def test_original_validation_exception_survives_cleanup(kind):
    server, request, raw, call = endpoint(kind)
    error = ValueError("model registry failed")
    server.models.is_base_model.side_effect = error
    with pytest.raises(ValueError) as caught:
        await call(request, raw)
    assert caught.value is error
    server.engine_client.notify_kv_transfer_request_rejected.assert_awaited_once_with(
        "original-id", KV, data_parallel_rank=2
    )
    server.engine_client.generate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ENDPOINTS)
@pytest.mark.parametrize("mode", ["no-connector", "no-metadata", "local-prefill"])
async def test_ordinary_rejections_do_not_notify(kind, mode):
    kv = (
        None
        if mode == "no-metadata"
        else dict(KV, do_remote_prefill=mode != "local-prefill")
    )
    server, request, raw, call = endpoint(kind, kv, mode != "no-connector")
    result = await call(request, raw)
    assert isinstance(result, ErrorResponse) and result.error.code == 404
    server.engine_client.notify_kv_transfer_request_rejected.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_completion_admission_does_not_notify():
    server, request, raw, call = endpoint("completion")
    request.stream = True
    server.models.is_base_model.return_value = True
    server.default_sampling_params = {}
    server.override_max_tokens = None
    server.openai_serving_render = SimpleNamespace(
        render_completion=AsyncMock(
            return_value=[{"type": "token", "prompt_token_ids": [1, 2]}]
        )
    )

    async def generated():
        if False:
            yield None

    server.engine_client.generate.return_value = generated()
    response = await call(request, raw)
    assert not isinstance(response, ErrorResponse)
    server.engine_client.generate.assert_called_once()
    server.engine_client.notify_kv_transfer_request_rejected.assert_not_awaited()
    await response.aclose()
    await server.engine_client.generate.return_value.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ENDPOINTS)
async def test_cleanup_failure_does_not_replace_client_error(kind):
    server, request, raw, call = endpoint(kind)
    server.engine_client.notify_kv_transfer_request_rejected.side_effect = RuntimeError(
        "cleanup unavailable"
    )
    result = await call(request, raw)
    assert isinstance(result, ErrorResponse) and result.error.code == 404
