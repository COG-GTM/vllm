# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Actor host resolution/propagation without starting Ray or touching sockets."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import ray
import ray.util.scheduling_strategies

from vllm.v1.engine.core import EngineCoreActorMixin
from vllm.v1.engine.utils import CoreEngineActorManager, EngineZmqAddresses

HOST = "VLLM_NIXL_SIDE_CHANNEL_HOST"


def config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_size_local=1,
            enable_elastic_ep=False,
            world_size=1,
        ),
        model_config=SimpleNamespace(is_moe=False),
        kv_transfer_config=None,
    )


def addresses():
    return EngineZmqAddresses(
        inputs=["tcp://127.0.0.1:12345"], outputs=["tcp://127.0.0.1:12346"]
    )


class DeviceFreeActor(EngineCoreActorMixin):
    def _set_visible_devices(self, *args):
        # Substitute only GPU construction; host initialization stays real.
        pass


@pytest.mark.parametrize(
    "explicit", [False, True], ids=["local-default", "explicit-node-override"]
)
def test_actor_initialization_resolves_its_own_host(monkeypatch, explicit):
    monkeypatch.setattr(ray.util, "get_node_ip_address", lambda: "10.0.0.22")
    if explicit:
        monkeypatch.setenv(HOST, "actor-configured-host")
    else:
        monkeypatch.delenv(HOST, raising=False)
    actor = DeviceFreeActor(config(), addresses())
    assert os.environ.get(HOST) == (
        "actor-configured-host" if explicit else "10.0.0.22"
    )
    assert actor.addresses.inputs == ["tcp://127.0.0.1:12345"]


def test_host_does_not_affect_compile_factors(monkeypatch):
    from vllm import envs

    monkeypatch.setenv(HOST, "10.0.0.11")
    first = envs.compile_factors()
    monkeypatch.setenv(HOST, "10.0.0.22")
    second = envs.compile_factors()
    assert HOST not in first and HOST not in second
    assert first == second


def test_worker_environment_excludes_host_but_keeps_ordinary_variables(monkeypatch):
    from vllm.ray.ray_env import get_env_vars_to_copy
    from vllm.v1.executor.ray_utils import WORKER_SPECIFIC_ENV_VARS

    monkeypatch.setenv(HOST, "driver-only-host")
    monkeypatch.setenv("VLLM_HOST_IP", "10.0.0.11")
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "INFO")
    forwarded = get_env_vars_to_copy(exclude_vars=WORKER_SPECIFIC_ENV_VARS)
    assert HOST not in forwarded
    assert "VLLM_HOST_IP" not in forwarded
    assert "VLLM_LOGGING_LEVEL" in forwarded


def test_driver_host_not_forwarded_in_actor_runtime_environment(monkeypatch):
    monkeypatch.setenv(HOST, "driver-only-host")
    monkeypatch.setenv("VLLM_HOST_IP", "10.0.0.11")
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "INFO")
    actor = Mock()
    factory = Mock()
    factory.options.return_value = factory
    factory.remote.return_value = actor
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "remote", lambda cls: factory)
    monkeypatch.setattr(ray, "get", lambda refs: None)
    monkeypatch.setattr(
        ray.util.scheduling_strategies,
        "PlacementGroupSchedulingStrategy",
        lambda **kwargs: None,
    )
    # The real manager builds the runtime env; only external Ray scheduling is fake.
    manager = CoreEngineActorManager(
        config(),
        addresses(),
        object,
        False,
        placement_groups=[object()],
        local_dp_ranks=[0],
    )
    propagated = factory.options.call_args.kwargs["runtime_env"]["env_vars"]
    assert HOST not in propagated
    assert "VLLM_HOST_IP" not in propagated
    assert propagated["VLLM_LOGGING_LEVEL"] == "INFO"
    assert os.environ[HOST] == "driver-only-host"
    assert manager.local_engine_actors == [actor]
