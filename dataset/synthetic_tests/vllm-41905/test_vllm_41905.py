# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real MiMo MTP construction, wrapper and predictor routing; tiny layer doubles."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.model_executor.models.mimo_v2_mtp as mimo


@pytest.fixture
def model_factory(monkeypatch):
    calls = []

    class Layer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, embeddings, positions, previous):
            calls.append((embeddings, positions, previous))
            return previous + embeddings + positions.unsqueeze(-1)

    class Head(nn.Linear):
        def __init__(self, vocab, hidden, **kwargs):
            super().__init__(hidden, vocab, bias=False)
            with torch.no_grad():
                self.weight.copy_(
                    torch.arange(vocab * hidden).reshape(vocab, hidden) / 32
                )

    class Logits(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, head, hidden):
            return head(hidden)

    def embedding(vocab, hidden, **kwargs):
        result = nn.Embedding(vocab, hidden)
        with torch.no_grad():
            result.weight.copy_(
                torch.arange(vocab * hidden).reshape(vocab, hidden) / 16
            )
        return result

    # Only heavyweight model/embedding/kernel collaborators are substituted.
    monkeypatch.setattr(mimo, "MiMoV2MTPLayer", Layer)
    monkeypatch.setattr(mimo, "VocabParallelEmbedding", embedding)
    monkeypatch.setattr(mimo, "ParallelLMHead", Head)
    monkeypatch.setattr(mimo, "LogitsProcessor", Logits)

    def make(num_tokens=1):
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(vocab_size=8, hidden_size=4)
            ),
            speculative_config=SimpleNamespace(num_speculative_tokens=num_tokens),
            quant_config=None,
        )
        return mimo.MiMoV2MTP(vllm_config=config), calls

    return make


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 8])
def test_construction_accepts_positive_speculative_token_counts(
    model_factory, num_tokens
):
    model, calls = model_factory(num_tokens)
    assert isinstance(model, nn.Module)
    assert calls == []


@pytest.mark.parametrize("step", [0, 1, 2, 5])
@pytest.mark.parametrize(
    "explicit_embeddings", [False, True], ids=["input_ids", "inputs_embeds"]
)
def test_hidden_state_path_reuses_available_layer(
    model_factory, step, explicit_embeddings
):
    model, calls = model_factory()
    ids = torch.tensor([1, 3])
    positions = torch.tensor([2, 4])
    previous = torch.ones(2, 4)
    embeddings = torch.full((2, 4), 0.75) if explicit_embeddings else None
    expected_embeddings = (
        embeddings if embeddings is not None else model.embed_input_ids(ids)
    )
    output = model(
        ids, positions, previous, inputs_embeds=embeddings, spec_step_idx=step
    )
    torch.testing.assert_close(
        output, previous + expected_embeddings + positions.unsqueeze(-1)
    )
    assert len(calls) == 1
    received_embeddings, received_positions, received_previous = calls[0]
    torch.testing.assert_close(received_embeddings, expected_embeddings)
    torch.testing.assert_close(received_positions, positions)
    torch.testing.assert_close(received_previous, previous)
    assert output.shape == previous.shape


@pytest.mark.parametrize("step", [0, 1, 2, 5])
def test_logits_path_accepts_later_speculative_steps(model_factory, step):
    model, _ = model_factory()
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    actual = model.compute_logits(hidden, spec_step_idx=step)
    expected = model.lm_head(hidden)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 8)


def test_constructed_multitoken_model_can_chain_predictions(model_factory):
    model, calls = model_factory(3)
    ids = torch.tensor([1, 3])
    positions = torch.tensor([2, 4])
    state = torch.zeros(2, 4)
    increment = model.embed_input_ids(ids) + positions.unsqueeze(-1)
    for step in range(3):
        state = model(ids, positions, state, spec_step_idx=step)
        torch.testing.assert_close(state, (step + 1) * increment)
        logits = model.compute_logits(state, spec_step_idx=step)
        torch.testing.assert_close(logits, model.lm_head(state))
    assert len(calls) == 3
