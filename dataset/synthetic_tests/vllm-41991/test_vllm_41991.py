# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Behavioral additions to the unchanged Gemma4 CPU parser coverage."""

import pytest

from vllm.tool_parsers.gemma4_tool_parser import (
    _parse_gemma4_args,
    _parse_gemma4_array,
)


@pytest.mark.timeout(5)
def test_malformed_array_terminates_without_prescribing_recovery():
    # The statement permits either partial output or safe rejection.  It does
    # not prescribe the exact partial list chosen after a stray bracket.
    result = _parse_gemma4_array("42,]trailing")
    assert result is None or isinstance(result, list)


@pytest.mark.timeout(5)
def test_malformed_arguments_terminate_without_prescribing_recovery():
    result = _parse_gemma4_args(":[t:[]")
    assert result is None or isinstance(result, dict)


@pytest.mark.timeout(5)
@pytest.mark.parametrize("value", ["a[b", "a]b", "[a]", "plain"])
def test_quoted_brackets_are_literal(value):
    result = _parse_gemma4_array(f'[<|"|>{value}<|"|>],<|"|>tail<|"|>')
    assert result == [[value], "tail"]
