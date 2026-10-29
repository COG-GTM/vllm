# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise prefetch through CLI/config and the real default-loader I/O path."""

import builtins
import os
import threading
import time
from types import SimpleNamespace
from typing import TypedDict

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.utils.argparse_utils import FlexibleArgumentParser


class PrefetchState(TypedDict):
    active: int
    peak: int
    closed: set[str]
    sizes: list[int]
    bytes: dict[str, int]


def test_cli_options_reach_load_config():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--safetensors-prefetch-num-threads",
            "3",
            "--safetensors-prefetch-block-size",
            "1M",
        ]
    )
    config = EngineArgs.from_cli_args(args).create_load_config()
    assert config.safetensors_prefetch_num_threads == 3
    assert config.safetensors_prefetch_block_size == 1024 * 1024


def test_public_defaults():
    config = LoadConfig()
    assert config.safetensors_prefetch_num_threads == 8
    assert config.safetensors_prefetch_block_size == 16 * 1024 * 1024


@pytest.mark.parametrize(
    "field", ["safetensors_prefetch_num_threads", "safetensors_prefetch_block_size"]
)
@pytest.mark.parametrize("value", [0, -1])
def test_config_rejects_nonpositive(field, value):
    # First prove the field is recognized: rejecting an unknown keyword is not
    # evidence of the requested numeric validation.
    accepted = LoadConfig(**{field: 1})
    assert getattr(accepted, field) == 1
    with pytest.raises(ValueError):
        LoadConfig(**{field: value})


def exercise_prefetch(tmp_path, monkeypatch, config, *, block_size, num_threads):
    paths = set()
    expected_names = set()
    for i in range(6):
        path = tmp_path / f"model-{i}.safetensors"
        name = f"weight_{i}"
        save_file({name: torch.arange(3000, dtype=torch.float32)}, str(path))
        paths.add(str(path))
        expected_names.add(name)

    condition = threading.Condition()
    state: PrefetchState = {
        "active": 0,
        "peak": 0,
        "closed": set(),
        "sizes": [],
        "bytes": {},
    }
    real_open = builtins.open

    class ObservedFile:
        def __init__(self, file, path):
            self.file = file
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            try:
                self.file.close()
            finally:
                with condition:
                    state["active"] -= 1
                    state["closed"].add(self.path)
                    condition.notify_all()

        def __getattr__(self, name):
            return getattr(self.file, name)

        def read(self, size=-1):
            data = self.file.read(size)
            with condition:
                state["sizes"].append(size)
                state["bytes"][self.path] = state["bytes"].get(self.path, 0) + len(data)
            # Widen overlap, without requiring any minimum degree of parallelism.
            time.sleep(0.003)
            return data

    def observed_open(file, mode="r", *args, **kwargs):
        opened = real_open(file, mode, *args, **kwargs)
        path = os.fspath(file) if isinstance(file, (str, os.PathLike)) else None
        if path in paths and mode == "rb":
            with condition:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            return ObservedFile(opened, path)
        return opened

    # Instrument only external file I/O. The loader, iterator, prefetch helper,
    # thread pool, and reads all execute unchanged, with real local tiny files.
    monkeypatch.setattr(builtins, "open", observed_open)
    loader = DefaultModelLoader(config)
    weights = dict(
        loader.get_all_weights(
            SimpleNamespace(model=str(tmp_path), revision=None), torch.nn.Module()
        )
    )
    with condition:
        finished = condition.wait_for(lambda: state["closed"] == paths, timeout=10)
    assert finished, "all local files must be prefetched before the test completes"
    assert set(weights) == expected_names
    assert all(
        torch.equal(value, torch.arange(3000, dtype=torch.float32))
        for value in weights.values()
    )
    assert state["active"] == 0
    assert 1 <= state["peak"] <= num_threads
    assert state["sizes"] and set(state["sizes"]) == {block_size}
    assert state["bytes"] == {path: os.path.getsize(path) for path in paths}


@pytest.mark.parametrize("threads,block", [(1, 1024), (2, 4096)])
def test_loader_honors_prefetch_settings(tmp_path, monkeypatch, threads, block):
    config = LoadConfig(
        load_format="safetensors",
        safetensors_load_strategy="prefetch",
        use_tqdm_on_load=False,
        safetensors_prefetch_num_threads=threads,
        safetensors_prefetch_block_size=block,
    )
    exercise_prefetch(
        tmp_path, monkeypatch, config, block_size=block, num_threads=threads
    )


def test_default_loader_io_unchanged(tmp_path, monkeypatch):
    config = LoadConfig(
        load_format="safetensors",
        safetensors_load_strategy="prefetch",
        use_tqdm_on_load=False,
    )
    exercise_prefetch(
        tmp_path, monkeypatch, config, block_size=16 * 1024 * 1024, num_threads=8
    )


def test_cli_without_new_flags_still_constructs():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    config = EngineArgs.from_cli_args(parser.parse_args([])).create_load_config()
    assert config.load_format == "auto"
