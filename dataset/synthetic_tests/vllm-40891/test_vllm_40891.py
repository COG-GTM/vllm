# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise actual executor initialization and RPC/Future behavior on CPU."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm.v1.executor.uniproc_executor as uniproc
from vllm.v1.outputs import AsyncModelRunnerOutput


class Output(AsyncModelRunnerOutput):
    def __init__(self, fail=False):
        self.calls = []
        self.value = object()
        self.error = RuntimeError("output retrieval failed") if fail else None

    def get_output(self):
        self.calls.append(threading.current_thread())
        if self.error is not None:
            raise self.error
        return self.value


@pytest.fixture
def executor_factory(monkeypatch):
    # Hardware/model/network initialization are external collaborators. Neither
    # executor initialization nor the changed async resolution path is mocked.
    monkeypatch.setenv("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", "0")
    monkeypatch.setattr(
        uniproc.UniProcExecutor,
        "_distributed_args",
        lambda self: ("unused://local", 0, 0),
    )
    monkeypatch.setattr(
        uniproc.current_platform, "update_block_size_for_backend", lambda cfg: None
    )
    pools = []
    pool_init = ThreadPoolExecutor.__init__

    def record_pool(self, *args, **kwargs):
        pool_init(self, *args, **kwargs)
        pools.append(self)

    monkeypatch.setattr(ThreadPoolExecutor, "__init__", record_pool)
    workers = []

    def make(output, async_scheduling):
        worker = MagicMock()
        worker.execute_model.return_value = output
        workers.append(worker)
        monkeypatch.setattr(uniproc, "WorkerWrapperBase", lambda **kwargs: worker)
        config = SimpleNamespace(
            **{
                name: None
                for name in (
                    "model_config",
                    "cache_config",
                    "lora_config",
                    "load_config",
                    "parallel_config",
                    "device_config",
                    "speculative_config",
                    "observability_config",
                )
            },
            scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        )
        return uniproc.UniProcExecutor(config), pools

    yield make
    # These are exclusively pools constructed by this test, never shared workers.
    for pool in pools:
        pool.shutdown(wait=True)
    for worker in workers:
        worker.shutdown()


@pytest.mark.parametrize(
    "async_scheduling", [False, True], ids=["sync_scheduler", "async_scheduler"]
)
@pytest.mark.parametrize("single_value", [False, True], ids=["list", "single"])
def test_async_output_is_lazy_cached_and_on_result_callers_thread(
    executor_factory, async_scheduling, single_value
):
    output = Output()
    executor, _ = executor_factory(output, async_scheduling)
    future = executor.collective_rpc(
        "execute_model", non_block=True, single_value=single_value
    )
    assert output.calls == [], "Output must not be fetched before result()"
    first = future.result()
    second = future.result()
    assert output.calls == [threading.current_thread()]
    assert first is second
    if single_value:
        assert first is output.value
    else:
        assert first == [output.value]


@pytest.mark.parametrize(
    "async_scheduling", [False, True], ids=["sync_scheduler", "async_scheduler"]
)
def test_output_exception_is_lazy_and_cached(executor_factory, async_scheduling):
    output = Output(fail=True)
    executor, _ = executor_factory(output, async_scheduling)
    future = executor.collective_rpc("execute_model", non_block=True, single_value=True)
    assert output.calls == []
    for _ in range(2):
        with pytest.raises(RuntimeError) as error:
            future.result()
        assert error.value is output.error
    assert output.calls == [threading.current_thread()]


@pytest.mark.parametrize(
    "async_scheduling", [False, True], ids=["sync_scheduler", "async_scheduler"]
)
def test_initialization_creates_no_output_thread_pool(
    executor_factory, async_scheduling
):
    _, pools = executor_factory("immediate", async_scheduling)
    assert pools == [], "UniProcExecutor must not construct a background thread pool"


@pytest.mark.parametrize("non_block", [False, True], ids=["blocking", "nonblocking"])
@pytest.mark.parametrize("single_value", [False, True], ids=["list", "single"])
def test_nonasync_output_shape_is_unchanged(executor_factory, non_block, single_value):
    executor, _ = executor_factory("immediate", True)
    result = executor.collective_rpc(
        "execute_model", non_block=non_block, single_value=single_value
    )
    if non_block:
        result = result.result()
    assert result == ("immediate" if single_value else ["immediate"])
