# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate allocator compatibility through public VllmConfig construction."""

import pytest
from transformers import GPT2Config

from vllm.config import KVTransferConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.platforms import current_platform


@pytest.fixture
def local_model(tmp_path, monkeypatch):
    # Only device capability is mocked: no CUDA allocations, model weights,
    # network access, private validation-hook invocation, or validator mocks.
    monkeypatch.setattr(current_platform, "is_sleep_mode_available", lambda: True)
    GPT2Config(
        architectures=["GPT2LMHeadModel"],
        n_layer=1,
        n_head=2,
        n_embd=16,
        n_positions=64,
        vocab_size=128,
    ).save_pretrained(tmp_path)
    return str(tmp_path)


def build(connector=None, *, model=None, sleep=False):
    transfer = (
        KVTransferConfig(kv_connector=connector, kv_role="kv_both")
        if connector is not None
        else None
    )
    model_config = (
        ModelConfig(
            model=model,
            enable_sleep_mode=sleep,
            enforce_eager=True,
            max_model_len=32,
            skip_tokenizer_init=True,
            generation_config="vllm",
        )
        if model
        else None
    )
    model_kwargs = {"model_config": model_config} if model_config is not None else {}
    return VllmConfig(
        **model_kwargs,
        kv_transfer_config=transfer,
        scheduler_config=SchedulerConfig(
            max_model_len=32, is_encoder_decoder=False, async_scheduling=False
        ),
    )


@pytest.mark.parametrize(
    "connector", ["NixlConnector", "MooncakeConnectorV1", "SomeOOTConnector"]
)
@pytest.mark.parametrize(
    "allocator",
    ["expandable_segments:True", "max_split_size_mb:512,expandable_segments:True"],
)
def test_rejects_unsafe_connector_during_construction(
    monkeypatch, connector, allocator
):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", allocator)
    with pytest.raises(ValueError, match="expandable_segments"):
        build(connector)


def test_sleep_mode_allows_connector(monkeypatch, local_model):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    config = build("NixlConnector", model=local_model, sleep=True)
    assert config.model_config.enable_sleep_mode is True
    assert config.kv_transfer_config.kv_connector == "NixlConnector"


@pytest.mark.parametrize(
    "allocator",
    [None, "max_split_size_mb:512", "max_split_size_mb:512,expandable_segments:False"],
)
def test_unrelated_allocator_options_allowed(monkeypatch, allocator):
    if allocator is None:
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    else:
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", allocator)
    config = build("SomeOOTConnector")
    assert config.kv_transfer_config.kv_connector == "SomeOOTConnector"


def test_no_connector_allows_expandable_segments(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    assert build().kv_transfer_config is None
