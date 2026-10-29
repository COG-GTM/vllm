# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cancellation behavior through the real speech-to-text orchestration."""

import asyncio
import io
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from vllm.entrypoints.openai.speech_to_text import speech_to_text as stt
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranscriptionResponse,
    TranslationResponse,
)
from vllm.sampling_params import SamplingParams


class Engine:
    def __init__(self, blocking=True):
        self.errored = False
        self.blocking = blocking
        self.submitted = []
        self.started = []
        self.closed = []
        self.first_started = asyncio.Event()
        self.abort = AsyncMock()
        self.is_tracing_enabled = AsyncMock(return_value=False)

    def generate(self, prompt, params, request_id, **kwargs):
        self.submitted.append(request_id)
        index = len(self.submitted) - 1

        async def result():
            self.started.append(request_id)
            self.first_started.set()
            try:
                if self.blocking:
                    await asyncio.Event().wait()
                yield SimpleNamespace(
                    finished=True,
                    outputs=[SimpleNamespace(text=f"chunk{index}", token_ids=[1])],
                )
            finally:
                self.closed.append(request_id)

        return result()


def setup(monkeypatch, count, task_type, *, language="en", blocking=True, stream=False):
    # Only model/engine construction is substituted; audio I/O is real.
    chunks = [np.zeros(8, dtype=np.float32) for _ in range(count)]
    audio = io.BytesIO()
    with wave.open(audio, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * (16000 * (30 * (count - 1) + 1)))
    engine = Engine(blocking)
    server = stt.OpenAISpeechToText.__new__(stt.OpenAISpeechToText)
    server.engine_client = engine
    server.task_type = task_type
    server.models = SimpleNamespace(
        is_base_model=lambda name: True, lora_requests={}, model_name=lambda: "audio"
    )
    server.model_config = SimpleNamespace(max_model_len=1024, is_encoder_decoder=False)
    server.model_cls = SimpleNamespace(
        no_space_languages=set(),
        validate_language=lambda lang: lang,
        supports_explicit_language_detection=True,
        get_language_detection_prompt=Mock(return_value={"prompt": "detect"}),
        get_language_token_ids=Mock(return_value=[1]),
        parse_language_detection_output=Mock(return_value="en"),
        get_generation_prompt=Mock(return_value={"prompt": "audio"}),
        post_process_output=lambda text: text,
    )
    server.tokenizer = Mock()
    server.renderer = SimpleNamespace(
        render_cmpl_async=AsyncMock(return_value=[{"prompt": "audio"} for _ in chunks])
    )
    server.default_sampling_params = {}
    server.asr_config = SimpleNamespace(
        sample_rate=16000,
        allow_audio_chunking=True,
        max_audio_clip_s=30,
        min_energy_split_window_size=160,
        overlap_chunk_second=1,
    )
    server.max_audio_filesize_mb = 25
    server.request_logger = None
    request = SimpleNamespace(
        model="audio",
        response_format="json",
        stream=stream,
        use_beam_search=False,
        max_completion_tokens=None,
        language=language,
        to_language=None,
        prompt="",
        build_stt_params=Mock(return_value=SimpleNamespace()),
        to_sampling_params=Mock(return_value=SamplingParams(max_tokens=4)),
    )
    raw_request = SimpleNamespace(
        headers={"X-Request-Id": "outer-request"}, state=SimpleNamespace()
    )
    stream_callback = Mock(return_value=object())
    coroutine = server._create_speech_to_text(
        audio_data=audio.getvalue(),
        request=request,
        raw_request=raw_request,
        response_class=TranscriptionResponse
        if task_type == "transcribe"
        else TranslationResponse,
        stream_generator_method=stream_callback,
    )
    return engine, coroutine, stream_callback


def aborted_ids(engine):
    # EngineClient.abort permits one id or a batch: don't require batching,
    # suffix choices, order of abort calls, or exactly one abort invocation.
    ids: list[str] = []
    for call in engine.abort.await_args_list:
        value = call.args[0] if call.args else call.kwargs["request_id"]
        ids.extend([value] if isinstance(value, str) else value)
    return ids


async def cancel_and_drain(task):
    if not task.done():
        task.cancel()
    try:
        return await task
    except asyncio.CancelledError:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("task_type", ["transcribe", "translate"])
@pytest.mark.parametrize("count", [1, 3], ids=["single_chunk", "multiple_chunks"])
async def test_cancellation_aborts_exact_submitted_ids_and_propagates(
    monkeypatch, count, task_type
):
    engine, coroutine, _ = setup(monkeypatch, count, task_type)
    task = asyncio.create_task(coroutine)
    try:
        await asyncio.wait_for(engine.first_started.wait(), timeout=2)
        # Yield to already-ready chunk collectors, without wall-clock sleeps.
        for _ in range(5):
            await asyncio.sleep(0)
        assert len(engine.submitted) == count
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert set(aborted_ids(engine)) == set(engine.submitted)
        assert set(engine.started) == set(engine.submitted)
        assert set(engine.closed) == set(engine.started)
    finally:
        await cancel_and_drain(task)


@pytest.mark.asyncio
async def test_all_chunk_generators_are_advanced_and_cleaned_up(monkeypatch):
    engine, coroutine, _ = setup(monkeypatch, 3, "transcribe")
    task = asyncio.create_task(coroutine)
    try:
        await asyncio.wait_for(engine.first_started.wait(), timeout=2)
        for _ in range(5):
            await asyncio.sleep(0)
        assert len(engine.submitted) == 3
        assert set(engine.started) == set(engine.submitted)
    finally:
        await cancel_and_drain(task)
    assert set(engine.closed) == set(engine.started)


@pytest.mark.asyncio
@pytest.mark.parametrize("task_type", ["transcribe", "translate"])
async def test_language_detection_outer_cancellation_aborts_submitted_id(
    monkeypatch, task_type
):
    engine, coroutine, _ = setup(monkeypatch, 3, task_type, language=None)
    task = asyncio.create_task(coroutine)
    try:
        await asyncio.wait_for(engine.first_started.wait(), timeout=2)
        assert len(engine.submitted) == 1  # detection is before chunk generation
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert set(aborted_ids(engine)) == set(engine.submitted)
        assert set(engine.closed) == set(engine.started)
    finally:
        await cancel_and_drain(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("task_type", ["transcribe", "translate"])
@pytest.mark.parametrize("count", [1, 3])
async def test_success_preserves_order_and_never_aborts(monkeypatch, count, task_type):
    engine, coroutine, _ = setup(monkeypatch, count, task_type, blocking=False)
    response = await coroutine
    assert response.text == " ".join(f"chunk{i}" for i in range(count))
    if task_type == "transcribe":
        assert response.usage.seconds == 30 * (count - 1) + 1
    engine.abort.assert_not_awaited()
    assert set(engine.closed) == set(engine.submitted)


@pytest.mark.asyncio
@pytest.mark.parametrize("task_type", ["transcribe", "translate"])
async def test_streaming_still_delegates_without_collecting_or_aborting(
    monkeypatch, task_type
):
    engine, coroutine, callback = setup(monkeypatch, 2, task_type, stream=True)
    response = await coroutine
    assert response is callback.return_value
    callback.assert_called_once()
    assert len(engine.submitted) == 2
    assert engine.started == []
    engine.abort.assert_not_awaited()
    for iterator in callback.call_args.args[1]:
        await iterator.aclose()
