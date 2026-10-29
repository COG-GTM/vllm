# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU session orchestration through real Worker, LLM, AsyncLLM and HTTP APIs."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

import vllm.model_executor.model_loader.reload as reload
from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest
from vllm.entrypoints.llm import LLM
from vllm.entrypoints.serve.rlhf.api_router import attach_router
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.worker.gpu_worker import Worker


class TinyModel(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        self.b = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        self.events = events

    def load_weights(self, weights):
        self.events.append("model-load")
        for name, value in weights:
            self.get_parameter(name).copy_(value)


class MemoryTransport:
    """External trainer transport substitute: no session logic belongs here."""

    def __init__(self):
        self.initialized = []
        self.received = []

    def parse_init_info(self, info):
        return info

    def init_transfer_engine(self, info):
        self.initialized.append(info)

    def parse_update_info(self, info):
        # BASE expects this old backend flag. It must not mask missing session errors.
        return SimpleNamespace(weights=info, is_checkpoint_format=True)

    def receive_weights(self, info, load_weights):
        self.received.append(info.weights)
        load_weights(
            [(name, torch.tensor(value)) for name, value in info.weights.items()]
        )


@pytest.fixture
def worker(monkeypatch):
    fields = [
        "model_config",
        "cache_config",
        "lora_config",
        "load_config",
        "scheduler_config",
        "device_config",
        "speculative_config",
        "observability_config",
        "kv_transfer_config",
        "compilation_config",
    ]
    cfg = SimpleNamespace(
        **{name: SimpleNamespace() for name in fields},
        parallel_config=SimpleNamespace(),
        weight_transfer_config=None,
        profiler_config=SimpleNamespace(profiler=None),
    )
    precision = torch.get_float32_matmul_precision()
    instance = Worker(cfg, local_rank=0, rank=0, distributed_init_method="unused")
    instance.device = torch.device("cpu")
    events: list[object] = []
    instance.model_runner = SimpleNamespace(model=TinyModel(events))
    instance.weight_transfer_engine = MemoryTransport()
    # Layerwise reload and accelerator sync are external device/model boundaries;
    # the production Worker still owns their scheduling, callbacks and session state.
    monkeypatch.setattr(
        reload, "initialize_layerwise_reload", lambda model: events.append("initialize")
    )
    monkeypatch.setattr(
        reload,
        "finalize_layerwise_reload",
        lambda model, config: events.append("finalize"),
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    yield instance, events
    torch.set_float32_matmul_precision(precision)


def async_client(worker):
    client = AsyncLLM.__new__(AsyncLLM)

    async def dispatch(method, timeout=None, args=(), kwargs=None):
        return [getattr(worker, method)(*args, **(kwargs or {}))]

    client.engine_core = SimpleNamespace(
        collective_rpc_async=AsyncMock(side_effect=dispatch), shutdown=Mock()
    )
    return client


def http_client(monkeypatch, engine, enabled=True):
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "1" if enabled else "0")
    app = FastAPI()
    app.state.engine_client = engine
    attach_router(app)
    return TestClient(app)


@pytest.fixture
def api(request, worker, monkeypatch):
    instance, events = worker
    path = request.param
    if path == "worker":
        client = instance
    elif path == "offline":
        client = LLM.__new__(LLM)

        def dispatch(method, args=(), kwargs=None, **unused):
            return [getattr(instance, method)(*args, **(kwargs or {}))]

        client.llm_engine = SimpleNamespace(collective_rpc=Mock(side_effect=dispatch))
    else:
        engine = async_client(instance)
        client = engine if path == "async" else http_client(monkeypatch, engine)

    async def invoke(operation, payload=None):
        if path == "http":
            body = (
                {}
                if payload is None
                else (
                    {"update_info": payload}
                    if operation == "update_weights"
                    else {"is_checkpoint_format": payload}
                )
            )
            response = client.post("/" + operation, json=body)
            assert response.status_code == 200, response.text
            return
        method = getattr(client, operation)
        if operation == "update_weights":
            arg = (
                payload
                if path == "worker"
                else (
                    {"update_info": payload}
                    if path == "offline"
                    else WeightTransferUpdateRequest(update_info=payload)
                )
            )
            result = method(arg)
        else:
            result = (
                method() if payload is None else method(is_checkpoint_format=payload)
            )
        if path == "async":
            await result

    yield invoke, instance, events
    if path == "http":
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["worker", "offline", "async", "http"], indirect=True)
async def test_checkpoint_session_wraps_multiple_chunks_once(api):
    invoke, worker, events = api
    await invoke("start_weight_update")
    assert events == ["initialize"]
    await invoke("update_weights", {"a": [1.0, 2.0]})
    await invoke("update_weights", {"b": [3.0, 4.0]})
    assert events.count("initialize") == 1
    assert "finalize" not in events
    torch.testing.assert_close(worker.model_runner.model.a, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(worker.model_runner.model.b, torch.tensor([3.0, 4.0]))
    await invoke("finish_weight_update")
    assert events.count("finalize") == 1
    assert events.count("model-load") == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["worker", "offline", "async", "http"], indirect=True)
async def test_kernel_session_direct_copy_then_reusable_checkpoint_session(api):
    invoke, worker, events = api
    await invoke("start_weight_update", False)
    await invoke("update_weights", {"a": [5.0, 6.0]})
    await invoke("update_weights", {"b": [7.0, 8.0]})
    await invoke("finish_weight_update")
    assert (
        events == []
    )  # No checkpoint loading or layerwise processing for kernel format.
    torch.testing.assert_close(worker.model_runner.model.a, torch.tensor([5.0, 6.0]))
    torch.testing.assert_close(worker.model_runner.model.b, torch.tensor([7.0, 8.0]))
    # Default format is checkpoint again; zero-chunk sessions are legal.
    await invoke("start_weight_update")
    await invoke("finish_weight_update")
    assert events == ["initialize", "finalize"]


@pytest.mark.parametrize("operation", ["update_weights", "finish_weight_update"])
def test_no_active_session_rejected_before_any_transport_or_reload(worker, operation):
    instance, events = worker
    with pytest.raises(RuntimeError) as caught:
        if operation == "update_weights":
            instance.update_weights({"a": [1.0, 2.0]})
        else:
            instance.finish_weight_update()
    assert str(caught.value)
    assert not instance.weight_transfer_engine.received
    assert events == []


def test_double_start_rejected_without_reinitializing(worker):
    instance, events = worker
    instance.start_weight_update()
    with pytest.raises(RuntimeError) as caught:
        instance.start_weight_update()
    assert str(caught.value)
    assert events == ["initialize"]
    instance.finish_weight_update()
    assert events == ["initialize", "finalize"]


@pytest.mark.parametrize("operation", ["update_weights", "finish_weight_update"])
def test_completed_session_is_not_active(worker, operation):
    instance, events = worker
    instance.start_weight_update(False)
    instance.finish_weight_update()
    with pytest.raises(RuntimeError):
        if operation == "update_weights":
            instance.update_weights({"a": [1.0, 2.0]})
        else:
            instance.finish_weight_update()
    assert events == []


def test_transport_initialization_is_unchanged(worker):
    instance, events = worker
    instance.init_weight_transfer_engine({"trainer": "local"})
    assert instance.weight_transfer_engine.initialized == [{"trainer": "local"}]
    assert events == []


def test_unconfigured_transport_still_errors(worker):
    instance, events = worker
    instance.weight_transfer_engine = None
    with pytest.raises(RuntimeError):
        instance.init_weight_transfer_engine({})
    assert events == []


def test_rlhf_routes_stay_hidden_when_dev_mode_disabled(monkeypatch):
    engine = SimpleNamespace(start_weight_update=AsyncMock())
    with http_client(monkeypatch, engine, enabled=False) as client:
        assert client.post("/start_weight_update", json={}).status_code == 404
        assert client.post("/update_weights", json={}).status_code == 404
        assert client.post("/finish_weight_update", json={}).status_code == 404
    engine.start_weight_update.assert_not_called()


def test_http_missing_update_metadata_still_returns_bad_request(monkeypatch):
    engine = SimpleNamespace(update_weights=AsyncMock())
    with http_client(monkeypatch, engine) as client:
        assert client.post("/update_weights", json={}).status_code == 400
    engine.update_weights.assert_not_called()
