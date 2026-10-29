# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise real HTTP parsing over local Unix IPC, with no TCP/DNS or mocks."""

import asyncio
import tempfile
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

from vllm.benchmarks.serve import fetch_spec_decode_metrics


def fetch(body):
    async def run():
        async def metrics(request):
            return web.Response(text=body)

        app = web.Application()
        app.router.add_get("/metrics", metrics)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = str(Path(directory) / "metrics.sock")
                await web.UnixSite(runner, path).start()
                connector = aiohttp.UnixConnector(path=path)
                async with aiohttp.ClientSession(connector=connector) as session:
                    return await fetch_spec_decode_metrics("http://local", session)
        finally:
            await runner.cleanup()

    return asyncio.run(run())


def values(metrics):
    assert metrics is not None
    return (
        metrics.num_drafts,
        metrics.num_draft_tokens,
        metrics.num_accepted_tokens,
        metrics.accepted_per_pos,
    )


TOTALS = """# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{engine="0"} 60.0
vllm:spec_decode_num_drafts_total{engine="1"} 40
vllm:spec_decode_num_draft_tokens_total 300
vllm:spec_decode_num_accepted_tokens_total 150
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0",engine="0"} 30
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",position="0"} 50
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 70
"""


@pytest.mark.parametrize("suffix", ["created", "sum", "gauge"])
def test_only_counter_totals_contribute(suffix):
    ignored = "\n".join(
        [
            f'vllm:spec_decode_num_drafts_{suffix}{{engine="0"}} 1.7769e9',
            f"vllm:spec_decode_num_draft_tokens_{suffix} 1.7769e9",
            f"vllm:spec_decode_num_accepted_tokens_{suffix} 1.7769e9",
            (
                f"vllm:spec_decode_num_accepted_tokens_per_pos_{suffix}"
                '{position="0"} 1.7e9'
            ),
            (
                f"vllm:spec_decode_num_accepted_tokens_per_pos_{suffix}"
                '{position="1"} 1.7e9'
            ),
        ]
    )
    assert values(fetch(TOTALS + ignored)) == (100, 300, 150, {0: 80, 1: 70})


def test_labels_do_not_change_counter_identity():
    body = (
        """vllm:spec_decode_num_drafts_total 100
vllm:spec_decode_num_draft_tokens_total{model_name="num_drafts"} 300
vllm:spec_decode_num_accepted_tokens_total{model_name="num_draft_tokens"} 150
"""
        "vllm:spec_decode_num_accepted_tokens_per_pos_total"
        '{model_name="num_drafts",position="0"} 80\n'
        """
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 70
"""
    )
    assert values(fetch(body)) == (100, 300, 150, {0: 80, 1: 70})


def test_totals_only_and_multiengine_aggregation():
    assert values(fetch(TOTALS)) == (100, 300, 150, {0: 80, 1: 70})


def test_endpoint_without_speculative_metrics():
    assert fetch("# a comment\nvllm:request_success_total 5\n") is None
