# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise Mistral parser state used by the existing streaming serving layer."""

import json
from unittest.mock import Mock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.mistral_tool_parser import MistralToolParser

BOT = "[TOOL_CALLS]"
REQUEST = ChatCompletionRequest(messages=[], model="test")


def parser_for(old):
    # Only tokenizer construction is mocked.  Both formats are selected by
    # the real parser's existing vocabulary-based detection.
    tokenizer = Mock()
    vocab = {BOT: 1}
    if not old:
        vocab["[ARGS]"] = 2
    tokenizer.get_vocab.return_value = vocab
    return MistralToolParser(tokenizer)


def render(old, calls):
    if old:
        return BOT + json.dumps(
            [{"name": name, "arguments": args} for name, args in calls],
            ensure_ascii=False,
        )
    return "".join(
        BOT + name + json.dumps(args, ensure_ascii=False) for name, args in calls
    )


def stream(parser, text, chunk_size):
    if chunk_size is None:
        chunks = [text]
    else:
        # Special tokens arrive atomically; ordinary text can be arbitrarily
        # fragmented.  There is no real tokenizer or network dependency.
        chunks = []
        pieces = text.split(BOT)
        for index, piece in enumerate(pieces):
            if index:
                chunks.append(BOT)
            chunks.extend(
                piece[i : i + chunk_size] for i in range(0, len(piece), chunk_size)
            )
    previous = ""
    previous_ids: list[int] = []
    for chunk in chunks:
        ids = [1] if BOT in chunk else [3]
        current = previous + chunk
        current_ids = previous_ids + ids
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=chunk,
            previous_token_ids=previous_ids,
            current_token_ids=current_ids,
            delta_token_ids=ids,
            request=REQUEST,
        )
        previous, previous_ids = current, current_ids
        if delta is not None:
            yield delta


# GT still has old-format multi-tool fragment/one-chunk decoding limitations.
# Character streaming isolates the old-format state-tracking change.
@pytest.mark.parametrize(
    "old,chunk_size",
    [(False, 1), (False, 7), (False, None), (True, 1)],
    ids=["v11-characters", "v11-fragments", "v11-one_chunk", "pre_v11-characters"],
)
@pytest.mark.parametrize(
    "calls",
    [
        [("add", {"a": 3, "b": 4})],
        [("add", {"a": "é", "b": "4"}), ("weather", {"city": "Paris"})],
    ],
    ids=["one_tool", "independent_tools"],
)
def test_streaming_tool_state_is_usable_by_serving(old, chunk_size, calls):
    text = render(old, calls)
    expected = parser_for(old).extract_tool_calls(text, REQUEST)
    parser = parser_for(old)
    names: dict[int, str] = {}
    arguments: dict[int, str] = {}
    for delta in stream(parser, text, chunk_size):
        for call in delta.tool_calls or []:
            names[call.index] = names.get(call.index, "") + (call.function.name or "")
            arguments[call.index] = arguments.get(call.index, "") + (
                call.function.arguments or ""
            )
    assert [names[i] for i in range(len(calls))] == [name for name, _ in calls]
    assert [json.loads(arguments[i]) for i in range(len(calls))] == [
        args for _, args in calls
    ]
    assert [call.function.name for call in expected.tool_calls] == list(names.values())
    assert [json.loads(call.function.arguments) for call in expected.tool_calls] == [
        json.loads(arguments[i]) for i in range(len(calls))
    ]
    # These two attributes already exist on ToolParser in BASE and are read
    # by chat_completion/serving.py when finishing a stream.  Do not impose
    # new helper names or a string-vs-dict representation on the saved args.
    assert len(parser.prev_tool_call_arr) == len(calls)
    assert len(parser.streamed_args_for_tool) == len(calls)
    for index, (name, args) in enumerate(calls):
        assert parser.prev_tool_call_arr[index]["name"] == name
        saved = parser.prev_tool_call_arr[index]["arguments"]
        assert (json.loads(saved) if isinstance(saved, str) else saved) == args
        assert parser.streamed_args_for_tool[index] == arguments[index]


@pytest.mark.parametrize("old", [False, True], ids=["v11", "pre_v11"])
def test_ordinary_text_has_no_tool_state(old):
    text = "A normal assistant response."
    parser = parser_for(old)
    deltas = list(stream(parser, text, 3))
    assert "".join(delta.content or "" for delta in deltas) == text
    assert all(not delta.tool_calls for delta in deltas)
    assert parser.prev_tool_call_arr == []
    assert parser.streamed_args_for_tool == []
    extracted = parser.extract_tool_calls(text, REQUEST)
    assert not extracted.tools_called
    assert extracted.content == text


@pytest.mark.parametrize("old", [False, True], ids=["v11", "pre_v11"])
def test_partial_tool_name_is_safe(old):
    text = BOT + ('[{"name": "wea' if old else "wea")
    deltas = list(stream(parser_for(old), text, 1))
    assert all(not delta.tool_calls for delta in deltas)


@pytest.mark.parametrize("old", [False, True], ids=["v11", "pre_v11"])
def test_nonstreaming_functions_remain_compatible(old):
    calls = [("add", {"a": 1}), ("weather", {"city": "Paris"})]
    result = parser_for(old).extract_tool_calls(render(old, calls), REQUEST)
    assert result.tools_called
    assert [call.function.name for call in result.tool_calls] == [
        name for name, _ in calls
    ]
    assert [json.loads(call.function.arguments) for call in result.tool_calls] == [
        args for _, args in calls
    ]
