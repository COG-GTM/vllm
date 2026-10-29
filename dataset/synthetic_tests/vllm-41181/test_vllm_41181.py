# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline concurrency through the existing HfRenderer tokenizer interface."""

import io
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

import vllm.renderers.base as renderer_base
from vllm.renderers.hf import HfRenderer


def make_fast_tokenizer():
    backend = Tokenizer(
        WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "[PAD]": 3}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{message['content']}} {% endfor %}"
    )
    return tokenizer


@pytest.fixture
def renderer_factory(monkeypatch):
    # The model registry is a collaborator: no model or processor is needed to
    # exercise the real renderer initialization and real fast-tokenizer calls.
    monkeypatch.setattr(
        renderer_base.mm_registry, "supports_multimodal_inputs", lambda _: False
    )
    renderers = []

    def create(tokenizer):
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                renderer_num_workers=8,
                hf_config=SimpleNamespace(),
                enable_prompt_embeds=False,
            ),
            parallel_config=SimpleNamespace(_api_process_rank=0),
        )
        renderer = HfRenderer(config, tokenizer)
        renderers.append(renderer)
        return renderer

    yield create
    for renderer in renderers:
        renderer._executor.shutdown(wait=True)


def test_shared_fast_tokenizer_batches_match_sequential(renderer_factory):
    tokenizer = renderer_factory(make_fast_tokenizer()).get_tokenizer()
    texts = ["hello world " * 30] * 128
    lengths = [24, 32, 40, 48, 56, 64, 72, 80]
    expected = {
        length: dict(
            tokenizer(texts, padding="max_length", max_length=length, truncation=True)
        )
        for length in lengths
    }
    barrier = Barrier(8)

    def encode(worker):
        for iteration in range(8):
            length = lengths[(worker + iteration) % len(lengths)]
            barrier.wait(timeout=15)
            # Do not abort early on a race: all workers must reach the next
            # barrier. Return errors for an explicit assertion after joining.
            try:
                actual = dict(
                    tokenizer(
                        texts, padding="max_length", max_length=length, truncation=True
                    )
                )
                yield actual == expected[length]
            except Exception as exc:
                yield repr(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda worker: list(encode(worker)), range(8)))
    assert results == [[True] * 8 for _ in range(8)]


def test_encode_decode_and_chat_remain_compatible(renderer_factory):
    original = make_fast_tokenizer()
    tokenizer = renderer_factory(original).get_tokenizer()
    messages = [{"role": "user", "content": "hello world"}]

    def operations(tok):
        return (
            tok.encode("hello world"),
            tok.decode([1, 2]),
            tok.batch_decode([[1], [1, 2]]),
            tok.convert_tokens_to_ids(["hello", "world"]),
            tok.convert_ids_to_tokens([1, 2]),
            tok.apply_chat_template(messages, tokenize=False),
            tok.apply_chat_template(messages, tokenize=True),
        )

    expected = operations(original)
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert (
            list(pool.map(lambda _: operations(tokenizer), range(32)))
            == [expected] * 32
        )


def test_renderer_tokenizer_serialization(renderer_factory):
    tokenizer = renderer_factory(make_fast_tokenizer()).get_tokenizer()
    buffer = io.BytesIO()
    torch.save(tokenizer, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    assert restored.encode("hello world") == tokenizer.encode("hello world")
    assert restored.decode([1, 2]) == tokenizer.decode([1, 2])


class TinySlowTokenizer(PreTrainedTokenizer):
    """Minimal local slow-tokenizer implementation, no external vocabulary."""

    vocab = {"[UNK]": 0, "hello": 1, "world": 2}

    def get_vocab(self):
        return dict(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text):
        return text.lower().split()

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, 0)

    def _convert_id_to_token(self, index):
        return {value: key for key, value in self.vocab.items()}.get(index, "[UNK]")


def test_slow_tokenizer_behavior_is_preserved(renderer_factory):
    original = TinySlowTokenizer(unk_token="[UNK]")
    tokenizer = renderer_factory(original).get_tokenizer()
    assert not tokenizer.is_fast
    assert tokenizer.encode("Hello world") == original.encode("Hello world")
    assert tokenizer.decode(tokenizer.encode("Hello world")) == original.decode(
        original.encode("Hello world")
    )
