# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real model weight routing, with CPU parameter sinks instead of model construction."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.nano_nemotron_vl import NemotronH_Nano_VL_V2


class Sink:
    def __init__(self):
        self.loaded = {}

    def load_weights(self, weights):
        for name, weight in weights:
            # Like parameter.copy_: consume the tensor before requesting another.
            self.loaded[name] = weight.clone()


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(4))


def make_model(multimodal=True, sound=False):
    model = object.__new__(NemotronH_Nano_VL_V2)
    torch.nn.Module.__init__(model)
    mm = SimpleNamespace(get_limit_per_prompt=lambda modality: int(multimodal))
    object.__setattr__(model, "model_config", SimpleNamespace(multimodal_config=mm))
    object.__setattr__(model, "language_model", Sink())
    object.__setattr__(model, "vision_model", Sink())
    object.__setattr__(model, "sound_encoder", Sink() if sound else None)
    model.mlp1 = Adapter()
    return model


def weights(reused, sound=False):
    names = [
        "language_model.a",
        "vision_model.radio_model.v",
        "language_model.b",
        "mlp1.w",
        "language_model.c",
    ]
    if sound:
        names += ["sound_encoder.w", "language_model.d"]
    buffer = torch.empty(4)
    for i, name in enumerate(names, 1):
        value = buffer if reused else torch.empty(4)
        value.fill_(i)
        yield name, value


@pytest.mark.parametrize("reused", [False, True], ids=["independent", "reused-buffer"])
@pytest.mark.parametrize("sound", [False, True], ids=["vision-only", "with-sound"])
def test_values_survive_streaming(reused, sound):
    model = make_model(sound=sound)
    model.load_weights(weights(reused, sound))
    expected = {"a": 1, "b": 3, "c": 5}
    if sound:
        expected["d"] = 7
    assert set(model.language_model.loaded) == set(expected)
    for name, value in expected.items():
        torch.testing.assert_close(
            model.language_model.loaded[name], torch.full((4,), float(value))
        )
    torch.testing.assert_close(
        model.vision_model.loaded["radio_model.v"], torch.full((4,), 2.0)
    )
    torch.testing.assert_close(model.mlp1.w, torch.full((4,), 4.0))
    if sound:
        torch.testing.assert_close(
            model.sound_encoder.loaded["sound_encoder.w"], torch.full((4,), 6.0)
        )


def test_language_weights_are_consumed_before_iterator_advances():
    model = make_model()

    def stream():
        yield "language_model.a", torch.ones(4)
        assert "a" in model.language_model.loaded, (
            "language checkpoint was eagerly buffered"
        )
        yield "language_model.b", torch.full((4,), 2.0)

    model.load_weights(stream())
    assert set(model.language_model.loaded) == {"a", "b"}


def test_text_only_skips_multimodal_weights_including_sound():
    model = make_model(multimodal=False)
    model.load_weights(weights(False, sound=True))
    assert set(model.language_model.loaded) == {"a", "b", "c", "d"}
    assert model.vision_model.loaded == {}
    torch.testing.assert_close(model.mlp1.w, torch.zeros(4))


def test_sound_weights_require_sound_encoder():
    model = make_model()
    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        model.load_weights(iter([("sound_encoder.w", torch.ones(4))]))
