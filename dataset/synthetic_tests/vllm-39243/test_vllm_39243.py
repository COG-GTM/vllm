# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Behavioral LFM2 contract; no implementation-specific regex or JSON formatting."""

import json
import time

import pytest
from transformers import AutoTokenizer

from tests.tool_parsers.utils import run_tool_extraction, run_tool_extraction_streaming
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers import ToolParserManager

START = "<|tool_call_start|>"
END = "<|tool_call_end|>"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(
        "LiquidAI/LFM2.5-1.2B-Instruct", local_files_only=True
    )


def parser(tokenizer):
    return ToolParserManager.get_tool_parser("lfm2")(tokenizer)


def decoded(calls):
    assert all(call.type == "function" for call in calls)
    return [(call.function.name, json.loads(call.function.arguments)) for call in calls]


CASES = [
    ("get_weather()", [("get_weather", {})]),
    ("lookup(city='Tokyo')", [("lookup", {"city": "Tokyo"})]),
    (
        (
            "store(name='John', age=37, ratio=1.5, flag=True, missing=None, "
            "address={'city': 'SF'}, aliases=['J', 'John'])"
        ),
        [
            (
                "store",
                {
                    "name": "John",
                    "age": 37,
                    "ratio": 1.5,
                    "flag": True,
                    "missing": None,
                    "address": {"city": "SF"},
                    "aliases": ["J", "John"],
                },
            )
        ],
    ),
    ("save(data={}, steps=[])", [("save", {"data": {}, "steps": []})]),
    (
        r"lookup(city='Martha\'s Vineyard', metric='\"cool units\"')",
        [("lookup", {"city": "Martha's Vineyard", "metric": '"cool units"'})],
    ),
    (
        (
            "a.b.c(items=[{'name': 'apple', 'quantity': 2}, "
            "{'name': 'bread', 'quantity': 1}])"
        ),
        [
            (
                "a.b.c",
                {
                    "items": [
                        {"name": "apple", "quantity": 2},
                        {"name": "bread", "quantity": 1},
                    ]
                },
            )
        ],
    ),
    ("first(x=1), second(ok=False)", [("first", {"x": 1}), ("second", {"ok": False})]),
]


@pytest.mark.parametrize(
    "body,expected",
    CASES,
    ids=[
        "parameterless",
        "string",
        "nested_types",
        "empty_collections",
        "escaping",
        "dotted_nested",
        "parallel_calls",
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_argument_semantics(tokenizer, body, expected, streaming):
    content, calls = run_tool_extraction(
        parser(tokenizer),
        f"{START}[{body}]{END}",
        streaming=streaming,
        assert_one_tool_per_delta=False,
    )
    assert decoded(calls) == expected
    assert not content


@pytest.mark.parametrize("streaming", [False, True])
def test_plain_content(tokenizer, streaming):
    text = "How can I help you today?"
    content, calls = run_tool_extraction(parser(tokenizer), text, streaming=streaming)
    assert content == text
    assert calls == []


@pytest.mark.parametrize("layout", ["block", "trailing", "leading", "both"])
@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_streaming_arbitrary_chunks(tokenizer, layout, chunk_size):
    lead = "Let me check. " if layout in ("leading", "both") else ""
    tail = "\nDone." if layout in ("trailing", "both") else ""
    text = f"{lead}{START}[a.b.c(items=[{{'x': 1}}, {{'x': 2}}])]{END}{tail}"
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    result = run_tool_extraction_streaming(
        parser(tokenizer), chunks, assert_one_tool_per_delta=False
    )
    assert decoded(result.tool_calls) == [("a.b.c", {"items": [{"x": 1}, {"x": 2}]})]
    if lead:
        assert lead.strip() in result.other_content
    if tail:
        assert tail.strip() in result.other_content
    assert START not in result.other_content
    assert END not in result.other_content
    assert "<|" not in result.other_content, "partial tool markers must not leak either"
    assert "a.b.c" not in result.other_content
    if not lead and not tail:
        assert not result.other_content


def test_nonstreaming_surrounding_content(tokenizer):
    content, calls = run_tool_extraction(
        parser(tokenizer), f"Checking. {START}[lookup(city='Tokyo')]{END}\nDone."
    )
    assert decoded(calls) == [("lookup", {"city": "Tokyo"})]
    assert "Checking." in content and "Done." in content
    assert START not in content and END not in content


@pytest.mark.parametrize("with_tools", [False, True])
def test_request_preserves_special_markers(tokenizer, with_tools):
    tools = (
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        if with_tools
        else None
    )
    request = ChatCompletionRequest(messages=[], model="test", tools=tools)
    adjusted = parser(tokenizer).adjust_request(request)
    assert adjusted.skip_special_tokens is (not with_tools)


def test_pathological_input_finishes_without_crashing(tokenizer):
    # Real malformed input, rather than requiring a particular regex attribute,
    # invocation count, timeout implementation, or fallback content policy.
    instance = parser(tokenizer)
    text = f"{START}[A(A=" + "\t)A(A=,\t" * 250 + f"]{END}"
    start = time.monotonic()
    result = instance.extract_tool_calls(
        text, ChatCompletionRequest(messages=[], model="test")
    )
    assert time.monotonic() - start < 10
    assert not result.tools_called and not result.tool_calls


def test_existing_pythonic_parser_still_parses(tokenizer):
    instance = ToolParserManager.get_tool_parser("pythonic")(tokenizer)
    content, calls = run_tool_extraction(instance, "[lookup(city='Tokyo')]")
    assert decoded(calls) == [("lookup", {"city": "Tokyo"})]
    assert not content
