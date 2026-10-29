# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline behavioral checks for Responses output-item resubmission."""

from collections.abc import Iterable, Mapping

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


@pytest.mark.parametrize(
    "item",
    [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "id": "msg_existing",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hello"}],
        },
        {"type": "reasoning", "summary": []},
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Consider the evidence."}],
        },
    ],
    ids=[
        "assistant-missing-metadata",
        "assistant-missing-annotations",
        "reasoning-empty-summary",
        "reasoning-summary-content",
    ],
)
def test_output_items_can_be_resubmitted(item):
    request = ResponsesRequest(model="test", input=[item])
    normalized = request.model_dump(mode="json")["input"][0]
    assert normalized["type"] == item["type"]
    if "id" in item:
        assert normalized["id"] == item["id"]
    if item["type"] == "message":
        assert normalized["role"] == "assistant"
        assert normalized["content"][0]["type"] == "output_text"
        assert normalized["content"][0]["text"] == "hello"
    else:
        assert normalized["summary"] == item["summary"]


def test_explicit_metadata_is_preserved():
    item = {
        "type": "message",
        "role": "assistant",
        "id": "msg_kept",
        "status": "incomplete",
        "content": [{"type": "output_text", "text": "partial", "annotations": []}],
    }
    request = ResponsesRequest(model="test", input=[item])
    normalized = request.model_dump(mode="json")["input"][0]
    assert normalized["id"] == "msg_kept"
    assert normalized["status"] == "incomplete"
    assert normalized["content"][0]["text"] == "partial"
    assert normalized["content"][0]["annotations"] == []
    reasoning = ResponsesRequest(
        model="test", input=[{"type": "reasoning", "id": "rs_kept", "summary": []}]
    )
    assert reasoning.model_dump(mode="json")["input"][0]["id"] == "rs_kept"


def test_function_call_and_ordinary_text_remain_valid():
    item = {
        "type": "function_call",
        "name": "lookup",
        "call_id": "call_1",
        "arguments": "{}",
    }
    request = ResponsesRequest(model="test", input=[item])
    normalized = request.model_dump(mode="json")["input"][0]
    assert normalized["name"] == "lookup"
    assert normalized["call_id"] == "call_1"
    assert normalized["arguments"] == "{}"
    assert ResponsesRequest(model="test", input="hello").input == "hello"


@pytest.mark.parametrize(
    "item",
    [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text"}]},
        {"type": "reasoning", "summary": [{"type": "summary_text"}]},
        {"type": "not_a_real_response_item", "payload": "anything"},
        {"type": "function_call", "name": "lookup", "arguments": "{}"},
    ],
    ids=[
        "missing-output-text",
        "missing-summary-text",
        "unknown-item-kind",
        "missing-call-id",
    ],
)
def test_substantive_invalid_content_still_rejected(item):
    # OpenAI input TypedDicts may contain lazily validated Iterable fields.
    # Accept eager or deferred rejection: do not mandate validation timing.
    def consume(value):
        if isinstance(value, Mapping):
            for child in value.values():
                consume(child)
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            for child in value:
                consume(child)

    with pytest.raises(ValidationError):
        request = ResponsesRequest(model="test", input=[item])
        consume(request.model_dump())
