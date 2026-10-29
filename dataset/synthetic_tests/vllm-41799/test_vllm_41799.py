# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gemma4 budget checks using real imported planning and prompt methods."""

from types import SimpleNamespace

import pytest
from PIL import Image

from vllm.model_executor.models.gemma4_mm import (
    Gemma4MultiModalProcessor,
    Gemma4ProcessingInfo,
)
from vllm.multimodal.parse import MultiModalDataParser

BUDGETS = [
    ({}, 280),
    ({"max_soft_tokens": 70}, 70),
    ({"max_soft_tokens": 140}, 140),
    ({"max_soft_tokens": 280}, 280),
    ({"max_soft_tokens": 560}, 560),
    ({"max_soft_tokens": 1120}, 1120),
    ({"images_kwargs": {"max_soft_tokens": 560}}, 560),
    ({"images_kwargs": None}, 280),
    ({"images_kwargs": "not-a-dict"}, 280),
    ({"max_soft_tokens": 70, "images_kwargs": {"max_soft_tokens": 560}}, 70),
]
IDS = [
    "default",
    "top-70",
    "top-140",
    "top-280",
    "top-560",
    "top-1120",
    "nested-560",
    "nested-none",
    "nested-malformed",
    "top-overrides-nested",
]


def make_info(kwargs, audio=False):
    # Only model/config and HF processor construction are substituted. Budget
    # lookup, image sizing, and token replacement remain real vLLM methods.
    config = SimpleNamespace(
        vision_config=SimpleNamespace(
            default_output_length=280, patch_size=16, pooling_kernel_size=2
        ),
        audio_config=SimpleNamespace() if audio else None,
        boi_token_id=100,
        eoi_token_id=101,
    )
    hf_processor = SimpleNamespace(
        image_token_id=102, image_token="<image>", audio_seq_length=777
    )
    ctx = SimpleNamespace(
        get_hf_config=lambda *args, **kw: config,
        get_hf_processor=lambda *args, **kw: hf_processor,
        get_merged_mm_kwargs=lambda overrides: {**kwargs, **overrides},
        model_config=SimpleNamespace(
            get_multimodal_config=lambda: SimpleNamespace(limit_per_prompt={})
        ),
    )
    return Gemma4ProcessingInfo(ctx)


@pytest.mark.parametrize("kwargs,expected", BUDGETS, ids=IDS)
def test_encoder_planning_respects_effective_budget(kwargs, expected):
    info = make_info(kwargs)
    result = info.get_mm_max_tokens_per_item(4096, {"image": 1, "video": 1})
    assert result == {"image": expected, "video": 32 * (70 + 2 + 6)}


@pytest.mark.parametrize("kwargs,expected", BUDGETS, ids=IDS)
def test_image_prompt_uses_effective_budget(kwargs, expected):
    info = make_info(kwargs)
    processor = Gemma4MultiModalProcessor.__new__(Gemma4MultiModalProcessor)
    processor.info = info
    items = MultiModalDataParser().parse_mm_data(
        {"image": Image.new("RGB", (1000, 1000))}
    )
    updates = processor._get_prompt_updates(items, {}, {})
    assert len(updates) == 1
    replacement = updates[0].resolve(0).content.full
    expected_replacement = info.get_image_repl(
        image_width=1000,
        image_height=1000,
        processor=info.get_hf_processor(),
        max_soft_tokens=expected,
    ).full
    assert replacement == expected_replacement
    assert replacement[0] == 100 and replacement[-1] == 101
    assert 0 < replacement.count(102) <= expected


def test_audio_and_video_accounting_are_independent_of_image_budget():
    info = make_info({"max_soft_tokens": 70}, audio=True)
    result = info.get_mm_max_tokens_per_item(4096, {"audio": 1, "video": 1})
    assert result["audio"] == 777
    assert result["video"] == 32 * (70 + 2 + 6)
