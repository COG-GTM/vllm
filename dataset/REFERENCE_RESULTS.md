# Reference results

One base and three GT runs per task, using the supplied scorer unchanged. Counts are exact manifest node selectors (parameter cases are separate where listed). T alone does not determine eligibility: inspect exclusions and pytest return codes. All GT patches exclude tests/.

| Task | F2P | P2P | Base T | GT 1 | GT 2 | GT 3 | Synthetic | Statement changed | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| vllm-41991 | 6 | 48 | 0 | 1 | 1 | 1 | Yes | No | Yes |
| vllm-40951 | 2 | 70 | 0 | 1 | 1 | 1 | No | No | Yes |
| vllm-41795 | 2 | 11 | 0 | 1 | 1 | 1 | No | No | Yes |
| vllm-41730 | 8 | 6 | 0 | 1 | 1 | 1 | Yes | Yes | Yes |
| vllm-40726 | 10 | 4 | 0 | 1 | 1 | 1 | Yes | No | Yes |
| vllm-41266 | 7 | 6 | 0 | 1 | 1 | 1 | Yes | No | Yes |
| vllm-41801 | 5 | 52 | 0 | 1 | 1 | 1 | No | No | Yes |
| vllm-40749 | 1 | 6 | 0 | 1 | 1 | 1 | No | No | Yes |
| vllm-41181 | 1 | 2 | 0 | 1 | 1 | 1 | Yes | No | No |
| vllm-42162 | 2 | 0 | 0 | 1 | 1 | 1 | No | Yes | Yes |
| vllm-41355 | 4 | 6 | 0 | 1 | 1 | 1 | Yes | No | Yes |
| vllm-41269 | 6 | 13 | 0 | 1 | 1 | 1 | Yes | No | Yes |
| vllm-41681 | 1 | 3 | 0 | 1 | 1 | 1 | Yes | No | Yes |
| vllm-41799 | 7 | 14 | 0 | 1 | 1 | 1 | Yes | No | Yes |
| vllm-42244 | 3 | 4 | 0 | 1 | 1 | 1 | Yes | No | Yes |
| vllm-41932 | 5 | 3 | 0 | 1 | 1 | 1 | Yes | No | Yes |
| vllm-41806 | 4 | 1 | 0 | 1 | 1 | 1 | Yes | No | Yes |
| vllm-39212 | 13 | 4 | 0 | 1 | 1 | 1 | Yes | Yes | Yes |
| vllm-42081 | 3 | 2 | 0 | 1 | 1 | 1 | Yes | No | Yes |
| vllm-39243 | 24 | 1 | 0 | 1 | 1 | 1 | Yes | No | No |
| vllm-41423 | 0 | 1 | 0 | 0 | 0 | 0 | Yes | No | No |
| vllm-42026 | 2 | 60 | 0 | 1 | 1 | 1 | No | No | Yes |
| vllm-41499 | 8 | 2 | 0 | 1 | 1 | 1 | Yes | No | Yes |
| vllm-41965 | 2 | 3 | 0 | 1 | 1 | 1 | No | No | Yes |
| vllm-41916 | 4 | 2 | 0 | 1 | 1 | 1 | Yes | No | Yes |
| vllm-42188 | 4 | 4 | 0 | 1 | 1 | 1 | Yes | No | Yes |
| vllm-40891 | 7 | 5 | 0 | 1 | 1 | 1 | Yes | No | Yes |
| vllm-40708 | 18 | 31 | 0 | 1 | 1 | 1 | Yes | No | Yes |
| vllm-41237 | 6 | 5 | 0 | 1 | 1 | 1 | Yes | No | Yes |
| vllm-41905 | 13 | 4 | 0 | 1 | 1 | 1 | Yes | No | Yes |

## Exclusions

- **vllm-41181**: GT fails task-required existing tokenizer serialization compatibility: pickle.loads(pickle.dumps(renderer.get_tokenizer())) returns None, causing AttributeError on encode. BASE passes the same real public-entrypoint round-trip. The GT new reducer calls a pool initializer that has no return on the fast-tokenizer path. Preserve this behavioral compatibility test rather than deleting it to force a clean GT. Not an import/fixture problem or scorer defect.
- **vllm-39243**: Fair real-parser tests expose a GT semantic gap: streaming character-by-character and seven-character chunks emit incomplete tool marker prefixes as assistant content, e.g. <|tool_call_start| and <|tool_call_st. The task explicitly requires arbitrary chunking without leaking markup. Retaining those assertions yields a nonzero GT pytest return code, so this task is excluded rather than weakening tests to make GT green.
- **vllm-41423**: Required GT definition excludes the entire implementation change in tests/utils.py. Both BASE and restricted GT execute the old decorator: two child exception/skip assertions fail and two argument/PID side-effect assertions fail identically. No eligible F2P node exists. Overlaying tests/utils.py would leak the implementation and is deliberately not done.

## Cold cache + pinned HF prefetch re-validation

`harness/setup_cpu_env.sh` prefetched nine pinned snapshot files (27.83 MB; 1.75
seconds on this VM) before the final CPU assertion. Tests continued to run
offline.

| Repository | Revision | Files |
| --- | --- | --- |
| `deepseek-ai/DeepSeek-V3.2` | `a7e62ac04ecb2c0a54d736dc46601c5606cf10a6` | `config.json`, `tokenizer.json`, `tokenizer_config.json` |
| `facebook/opt-125m` | `27dcfa74d334bc871f3234de431e71c6eeba5dd6` | `config.json` |
| `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | `config.json` |
| `zai-org/GLM-4.5` | `cbb2c7cfb52fa128a9660cb1a7a78e017899e115` | `chat_template.jinja`, `config.json`, `tokenizer.json`, `tokenizer_config.json` |

With the cache cleared before prefetch, all 27 eligible tasks scored T=0.0 in
base and T=1.0 in GT; all GT F2P cases passed, with no P2P regressions. The
original P2P lists from `0c13c421d` were restored and no P2P IDs were removed.
The previously cold-cache-failing HF-dependent cases in `vllm-40951`,
`vllm-41499`, `vllm-41801`, and `vllm-42026` passed after prefetch.
