# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Spawn-decorator contract without xfail or prescribed exception wrapper types."""

import json
import os
from pathlib import Path

import pytest

from tests.utils import spawn_new_process_for_each_test


@spawn_new_process_for_each_test
def failing_body():
    raise ValueError("child_failure_sentinel")


@spawn_new_process_for_each_test
def skipped_body():
    pytest.skip("child_skip_sentinel")


@spawn_new_process_for_each_test
def passing_body():
    assert 1 + 1 == 2


@spawn_new_process_for_each_test
def argument_body(destination, value, *, suffix):
    Path(destination).write_text(
        json.dumps(
            {
                "value": value,
                "suffix": suffix,
                "pid": os.getpid(),
            }
        )
    )


def test_child_failure_reaches_parent():
    with pytest.raises(Exception, match="child_failure_sentinel"):
        failing_body()


def test_child_skip_reaches_parent():
    # Catch the public pytest skip exception so this regression is scored as
    # PASSED rather than a skipped test that cannot be an F2P observation.
    with pytest.raises(pytest.skip.Exception, match="child_skip_sentinel"):
        skipped_body()


@pytest.mark.parametrize("value,suffix", [(3, "three"), (0, "zero")])
def test_arguments_and_process_isolation(tmp_path, value, suffix):
    destination = tmp_path / "child.json"
    argument_body(str(destination), value, suffix=suffix)
    assert destination.exists(), "decorated body must actually execute in the child"
    observed = json.loads(destination.read_text())
    assert observed["value"] == value and observed["suffix"] == suffix
    assert observed["pid"] != os.getpid()


def test_passing_body_returns_normally():
    assert passing_body() is None
