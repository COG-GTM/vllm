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

## Cold Hugging Face cache re-validation

Runs kept `~/.cache/huggingface` absent and used the scorer's offline settings.

- `vllm-40951` `tests/engine/test_arg_utils.py::test_attention_config`: offline Hugging Face cache miss (`huggingface_hub` error).
- `vllm-40951` `tests/engine/test_arg_utils.py::test_ir_op_priority`: offline Hugging Face cache miss (`huggingface_hub` error).
- `vllm-40951` `tests/engine/test_arg_utils.py::test_numa_bind_args`: offline Hugging Face cache miss (`huggingface_hub` error).
- `vllm-40951` `tests/engine/test_arg_utils.py::test_prefix_cache_default`: offline Hugging Face cache miss (`huggingface_hub` error).
- `vllm-41801` `tests/tool_parsers/test_deepseekv32_tool_parser.py::test_convert_param_value_checked_helper`: pytest short summary reported `ERROR` without an exception reason in the captured output.
- `vllm-41801` `tests/tool_parsers/test_deepseekv32_tool_parser.py::test_convert_param_value_edge_cases`: pytest short summary reported `ERROR` without an exception reason in the captured output.
- `vllm-41801` `tests/tool_parsers/test_deepseekv32_tool_parser.py::test_convert_param_value_multi_typed_values`: pytest short summary reported `ERROR` without an exception reason in the captured output.
- `vllm-41801` `tests/tool_parsers/test_deepseekv32_tool_parser.py::test_convert_param_value_single_types`: pytest short summary reported `ERROR` without an exception reason in the captured output.
- `vllm-41801` `tests/tool_parsers/test_deepseekv32_tool_parser.py::test_convert_param_value_stricter_type_checking`: pytest short summary reported `ERROR` without an exception reason in the captured output.

The scorer still executes the complete overlaid test file, including P2P cases removed from the scoring list. Both GT2 runs reached T=1.0, but pytest still returned 1 for those files.
