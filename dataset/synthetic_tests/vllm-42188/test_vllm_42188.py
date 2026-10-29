# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Render the shipped template, without a tokenizer or checkpoint download."""

from pathlib import Path

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment


@pytest.fixture(scope="module")
def template():
    path = (
        Path(__file__).resolve().parents[2] / "examples/tool_chat_template_gemma4.jinja"
    )
    return ImmutableSandboxedEnvironment().from_string(path.read_text())


def assistant():
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_weather",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "SF"}},
            }
        ],
    }


def render(template, messages):
    return template.render(
        messages=[{"role": "user", "content": "weather?"}] + messages,
        bos_token="<bos>",
        add_generation_prompt=True,
    )


@pytest.mark.parametrize(
    "tool_message",
    [
        {"role": "tool", "tool_call_id": "unmatched", "content": "sunny"},
        {"role": "tool", "content": "sunny"},
        {"role": "tool", "tool_call_id": None, "name": None, "content": "sunny"},
    ],
    ids=["unmatched_id", "missing_id", "null_id_and_name"],
)
def test_unresolvable_openai_tool_name(template, tool_message):
    output = render(template, [assistant(), tool_message])
    assert "response:unknown{" in output
    assert "sunny" in output


@pytest.mark.parametrize(
    "response",
    [
        {"name": None, "response": {"temperature": 20}},
        {"response": {"temperature": 20}},
    ],
    ids=["null_name", "missing_name"],
)
def test_legacy_unresolvable_name(template, response):
    message = assistant()
    message["tool_responses"] = [response]
    output = render(template, [message])
    assert "response:unknown{" in output
    assert "temperature" in output
    assert "20" in output


def test_matching_tool_id_retains_function_name(template):
    output = render(
        template,
        [
            assistant(),
            {"role": "tool", "tool_call_id": "call_weather", "content": "sunny"},
        ],
    )
    assert "response:get_weather{" in output
    assert "sunny" in output
    assert "response:unknown{" not in output


def test_named_legacy_response_retains_name_and_content(template):
    message = assistant()
    message["tool_responses"] = [
        {"name": "get_weather", "response": {"temperature": 20}}
    ]
    output = render(template, [message])
    assert "response:get_weather{" in output
    assert "temperature" in output
    assert "20" in output


def test_plain_conversation_still_renders(template):
    output = render(template, [{"role": "assistant", "content": "Hello there"}])
    assert "weather?" in output
    assert "Hello there" in output
    assert "response:unknown{" not in output
