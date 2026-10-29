#!/usr/bin/env python3
"""Prefetch the pinned Hugging Face files needed by offline evaluation."""

import json
import sys
import time
from pathlib import Path

import huggingface_hub
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.file_download import repo_folder_name


MAX_RETRIES = 5


def download_file(asset, filename):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return huggingface_hub.hf_hub_download(
                repo_id=asset["repo_id"],
                filename=filename,
                revision=asset["revision"],
                repo_type=asset["repo_type"],
            )
        except HfHubHTTPError as error:
            response = getattr(error, "response", None)
            status_code = getattr(response, "status_code", None)
            retryable = status_code == 429 or (
                status_code is not None and 500 <= status_code < 600
            )
            if not retryable or attempt == MAX_RETRIES:
                raise
            delay = 2**attempt
            print(
                f"HTTP {status_code} fetching {asset['repo_id']}/{filename}; "
                f"retrying in {delay}s ({attempt + 1}/{MAX_RETRIES})",
                file=sys.stderr,
            )
            time.sleep(delay)


def pin_default_revision(asset):
    cache = Path(huggingface_hub.constants.HF_HUB_CACHE)
    repo_dir = cache / repo_folder_name(
        repo_id=asset["repo_id"],
        repo_type=asset["repo_type"],
    )
    ref = repo_dir / "refs" / huggingface_hub.constants.DEFAULT_REVISION
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(asset["revision"], encoding="utf-8")


def main():
    assets_path = Path(__file__).with_name("hf_assets.json")
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    total_bytes = 0
    file_count = 0

    for asset in assets:
        for filename in asset["files"]:
            path = download_file(asset, filename)
            total_bytes += Path(path).stat().st_size
            file_count += 1
        pin_default_revision(asset)

    elapsed = time.perf_counter() - started
    total_mb = total_bytes / 1_000_000
    print(f"Prefetched {file_count} files ({total_mb:.2f} MB) in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
