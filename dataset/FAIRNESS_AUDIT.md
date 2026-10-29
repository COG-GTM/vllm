# STEP 3 fairness audit

All 16 originally stock-pytest tasks were audited. The sections below list original and replacement F2P contracts, every statement addition, and test changes. All 14 originally non-stock tasks have real pytest files; additional synthetic replacements remove overspecified stock contracts.

No historical production implementation is overlaid as a test. Public-interface paragraphs name required integration contracts, not an implementation strategy. P2P membership is restricted to actually PASSED cases in the overlaid files in both states; skipped and xfailed cases are not selected.

The real native CPU rejection-sampling kernels are compiled from the unchanged freeze base, with registration-only bindings in harness/. Model/device construction is kept small; changed control flow and changed parsing/loading/cleanup behavior run normally. Metrics use Unix IPC and audio uses locally generated WAV data.

## vllm-41991

Originally stock pytest: True. Synthetic overlay: True.

### Changes

- Retain all 46 pre-existing stock cases and two fair stock F2P cases. Deselect only TestParseGemma4Array.test_stray_closing_bracket because its exact [42] recovery is narrower than the stated partial-result-or-safe-rejection contract.
- Add bounded malformed args/array checks accepting either a partial container or None and parameterized exact bracket-content checks; every potentially hanging synthetic case has a 5s timeout.

### Contract audit

- `tests/tool_parsers/test_gemma4_tool_parser.py::TestParseGemma4Args::test_malformed_partial_array`: Existing BASE helper, bounded completion on malformed array; asserts dict type only, not exact recovery payload. Statement covers bounded parsing.
- `tests/tool_parsers/test_gemma4_tool_parser.py::TestParseGemma4Array::test_string_element_with_closing_bracket`: Existing BASE helper; exact well-formed string payload is public parsing semantics explicitly stated.
- `tests/tool_parsers/test_gemma4_tool_parser.py::TestParseGemma4Array::test_stray_closing_bracket`: Audited stock F2P, replaced because exact partial output is overspecified; behavioral replacement accepts safe None or list within timeout.
- `tests/eval_hidden/test_vllm_41991.py::test_malformed_arguments_terminate_without_prescribing_recovery`: Calls real BASE-existing parser helpers; malformed input checks bounded return with permissive recovery; well-formed bracket strings check exact semantic payload. No new private names or implementation strategy.
- `tests/eval_hidden/test_vllm_41991.py::test_malformed_array_terminates_without_prescribing_recovery`: Calls real BASE-existing parser helpers; malformed input checks bounded return with permissive recovery; well-formed bracket strings check exact semantic payload. No new private names or implementation strategy.
- `tests/eval_hidden/test_vllm_41991.py::test_quoted_brackets_are_literal[a[b]`: Calls real BASE-existing parser helpers; malformed input checks bounded return with permissive recovery; well-formed bracket strings check exact semantic payload. No new private names or implementation strategy.
- `tests/eval_hidden/test_vllm_41991.py::test_quoted_brackets_are_literal[a]b]`: Calls real BASE-existing parser helpers; malformed input checks bounded return with permissive recovery; well-formed bracket strings check exact semantic payload. No new private names or implementation strategy.
- `tests/eval_hidden/test_vllm_41991.py::test_quoted_brackets_are_literal[[a]]`: Calls real BASE-existing parser helpers; malformed input checks bounded return with permissive recovery; well-formed bracket strings check exact semantic payload. No new private names or implementation strategy.
- `tests/eval_hidden/test_vllm_41991.py::test_quoted_brackets_are_literal[plain]`: Calls real BASE-existing parser helpers; malformed input checks bounded return with permissive recovery; well-formed bracket strings check exact semantic payload. No new private names or implementation strategy.

## vllm-40951

Originally stock pytest: True. Synthetic overlay: False.

### Changes

- Retain whole 72-case GT CLI/config test file; no GPU restriction needed.

### Contract audit

- `tests/engine/test_arg_utils.py::test_get_kwargs`: Existing get_kwargs API and bool|string|None config fields. String type is explicit in statement; nargs=? and const=True implement documented optional-value flag semantics, no BooleanOptionalAction is compatible with both bare True and explicit strings. Dummy field and ModelConfig.hf_token exercise general and canonical cases. Base fails missing string type; CLI roundtrips already pass on base because of special-case hf-token and are p2p.
- `tests/engine/test_arg_utils.py::test_hf_token_get_kwargs`: Existing get_kwargs API and bool|string|None config fields. String type is explicit in statement; nargs=? and const=True implement documented optional-value flag semantics, no BooleanOptionalAction is compatible with both bare True and explicit strings. Dummy field and ModelConfig.hf_token exercise general and canonical cases. Base fails missing string type; CLI roundtrips already pass on base because of special-case hf-token and are p2p.

## vllm-41795

Originally stock pytest: True. Synthetic overlay: False.

### Changes

- Retain entire GT regression test file; exact individual IDs classified from supplied plugin; no implementation helpers overlaid.

### Contract audit

- `tests/v1/kv_offload/cpu/test_manager.py::TestARCPolicy::test_basic`: Existing CPUOffloadingManager prepare_store/lookup contract; in-flight None is explicitly stated for both LRU and ARC. Base failure is False is None, not collection/import failure. Existing store/load/event assertions preserved.
- `tests/v1/kv_offload/cpu/test_manager.py::test_cpu_manager`: Existing CPUOffloadingManager prepare_store/lookup contract; in-flight None is explicitly stated for both LRU and ARC. Base failure is False is None, not collection/import failure. Existing store/load/event assertions preserved.

## vllm-41730

Originally stock pytest: True. Synthetic overlay: True.

### Statement addition

Interface requirements: The existing serving integration reads prev_tool_call_arr and streamed_args_for_tool for each streamed tool. Keep these pre-existing parser state collections aligned and complete so serving can finish with no IndexError or duplicate argument suffix; arguments may use any representation already supported by serving (JSON string or parsed mapping).

### Changes

- Replace stock file requiring unavailable offline Mistral tokenizer assets with 14 normal-import tests using a tokenizer vocabulary boundary mock.
- Exercise actual MistralToolParser streaming and nonstreaming v11 and pre-v11 paths, one/two functions, Unicode arguments, per-character output and v11 fragment/one-chunk output.
- Compare semantic parsed JSON, tool names and independent per-call state, accepting string or mapping arguments supported by existing serving. Retain ordinary text, malformed partial tool-name and nonstreaming P2P.
- Remove three originally proposed pre-v11 large-chunk combinations after observing GT JSONDecodeError/missing second tool. Keep legacy per-character streaming to isolate the targeted state repair; this is not claimed as universal chunking coverage.

### Contract audit

- `tests/tool_parsers/test_mistral_tool_parser.py::test_extract_tool_calls_streaming[multiple_tools]`: Stock helper asserts exact string state representation and streamed-args equality. Existing serving reads both arrays, but permits mapping or JSON string arguments. Replace with semantic JSON/name/state alignment checks on actual parser; do not require a new helper or force string storage.
- `tests/tool_parsers/test_mistral_tool_parser.py::test_extract_tool_calls_streaming[single_tool_add]`: Stock helper asserts exact string state representation and streamed-args equality. Existing serving reads both arrays, but permits mapping or JSON string arguments. Replace with semantic JSON/name/state alignment checks on actual parser; do not require a new helper or force string storage.
- `tests/tool_parsers/test_mistral_tool_parser.py::test_extract_tool_calls_streaming[single_tool_add_strings]`: Stock helper asserts exact string state representation and streamed-args equality. Existing serving reads both arrays, but permits mapping or JSON string arguments. Replace with semantic JSON/name/state alignment checks on actual parser; do not require a new helper or force string storage.
- `tests/eval_hidden/test_vllm_41730.py::test_streaming_tool_state_is_usable_by_serving[independent_tools-pre_v11-characters]`: Normal real parser import; tokenizer construction alone mocked. Existing state arrays are read by BASE serving at chat_completion/serving.py:965-1013. Names and argument JSON checked against request and nonstreaming interpretation; dict/string state both accepted, no ID spelling or new private helper requirement.
- `tests/eval_hidden/test_vllm_41730.py::test_streaming_tool_state_is_usable_by_serving[independent_tools-v11-characters]`: Normal real parser import; tokenizer construction alone mocked. Existing state arrays are read by BASE serving at chat_completion/serving.py:965-1013. Names and argument JSON checked against request and nonstreaming interpretation; dict/string state both accepted, no ID spelling or new private helper requirement.
- `tests/eval_hidden/test_vllm_41730.py::test_streaming_tool_state_is_usable_by_serving[independent_tools-v11-fragments]`: Normal real parser import; tokenizer construction alone mocked. Existing state arrays are read by BASE serving at chat_completion/serving.py:965-1013. Names and argument JSON checked against request and nonstreaming interpretation; dict/string state both accepted, no ID spelling or new private helper requirement.
- `tests/eval_hidden/test_vllm_41730.py::test_streaming_tool_state_is_usable_by_serving[independent_tools-v11-one_chunk]`: Normal real parser import; tokenizer construction alone mocked. Existing state arrays are read by BASE serving at chat_completion/serving.py:965-1013. Names and argument JSON checked against request and nonstreaming interpretation; dict/string state both accepted, no ID spelling or new private helper requirement.
- `tests/eval_hidden/test_vllm_41730.py::test_streaming_tool_state_is_usable_by_serving[one_tool-pre_v11-characters]`: Normal real parser import; tokenizer construction alone mocked. Existing state arrays are read by BASE serving at chat_completion/serving.py:965-1013. Names and argument JSON checked against request and nonstreaming interpretation; dict/string state both accepted, no ID spelling or new private helper requirement.
- `tests/eval_hidden/test_vllm_41730.py::test_streaming_tool_state_is_usable_by_serving[one_tool-v11-characters]`: Normal real parser import; tokenizer construction alone mocked. Existing state arrays are read by BASE serving at chat_completion/serving.py:965-1013. Names and argument JSON checked against request and nonstreaming interpretation; dict/string state both accepted, no ID spelling or new private helper requirement.
- `tests/eval_hidden/test_vllm_41730.py::test_streaming_tool_state_is_usable_by_serving[one_tool-v11-fragments]`: Normal real parser import; tokenizer construction alone mocked. Existing state arrays are read by BASE serving at chat_completion/serving.py:965-1013. Names and argument JSON checked against request and nonstreaming interpretation; dict/string state both accepted, no ID spelling or new private helper requirement.
- `tests/eval_hidden/test_vllm_41730.py::test_streaming_tool_state_is_usable_by_serving[one_tool-v11-one_chunk]`: Normal real parser import; tokenizer construction alone mocked. Existing state arrays are read by BASE serving at chat_completion/serving.py:965-1013. Names and argument JSON checked against request and nonstreaming interpretation; dict/string state both accepted, no ID spelling or new private helper requirement.
- `tests/eval_hidden/test_vllm_41730.py::test_nonstreaming_functions_remain_compatible[pre_v11]`: Normal real parser import; tokenizer construction alone mocked. Existing state arrays are read by BASE serving at chat_completion/serving.py:965-1013. Names and argument JSON checked against request and nonstreaming interpretation; dict/string state both accepted, no ID spelling or new private helper requirement.
- `tests/eval_hidden/test_vllm_41730.py::test_nonstreaming_functions_remain_compatible[v11]`: Normal real parser import; tokenizer construction alone mocked. Existing state arrays are read by BASE serving at chat_completion/serving.py:965-1013. Names and argument JSON checked against request and nonstreaming interpretation; dict/string state both accepted, no ID spelling or new private helper requirement.
- `tests/eval_hidden/test_vllm_41730.py::test_ordinary_text_has_no_tool_state[pre_v11]`: Normal real parser import; tokenizer construction alone mocked. Existing state arrays are read by BASE serving at chat_completion/serving.py:965-1013. Names and argument JSON checked against request and nonstreaming interpretation; dict/string state both accepted, no ID spelling or new private helper requirement.
- `tests/eval_hidden/test_vllm_41730.py::test_ordinary_text_has_no_tool_state[v11]`: Normal real parser import; tokenizer construction alone mocked. Existing state arrays are read by BASE serving at chat_completion/serving.py:965-1013. Names and argument JSON checked against request and nonstreaming interpretation; dict/string state both accepted, no ID spelling or new private helper requirement.
- `tests/eval_hidden/test_vllm_41730.py::test_partial_tool_name_is_safe[pre_v11]`: Normal real parser import; tokenizer construction alone mocked. Existing state arrays are read by BASE serving at chat_completion/serving.py:965-1013. Names and argument JSON checked against request and nonstreaming interpretation; dict/string state both accepted, no ID spelling or new private helper requirement.
- `tests/eval_hidden/test_vllm_41730.py::test_partial_tool_name_is_safe[v11]`: Normal real parser import; tokenizer construction alone mocked. Existing state arrays are read by BASE serving at chat_completion/serving.py:965-1013. Names and argument JSON checked against request and nonstreaming interpretation; dict/string state both accepted, no ID spelling or new private helper requirement.

## vllm-40726

Originally stock pytest: True. Synthetic overlay: True.

### Changes

- Replace complete stock file with 14 normal-import behavioral tests of real split_graph/generate_execution_code/compile_execution_fn.
- Avoid const-pool layout, dedup slot count, generated variable names, new private helper contracts, exact artifact tuple arity and regex source assertions.
- Add fresh/serialized artifacts, two input values, mixed dtypes, nested containers including distributed placement values without a process group, non-GraphModule child and legacy artifact P2P.

### Contract audit

- `tests/compile/test_codegen.py::test_consts_ordering_deterministic`: Exact const-list ordering is private caching strategy; round-trip artifact through pickle and execute instead.
- `tests/compile/test_codegen.py::test_consts_shared_across_split_submods`: Const-slot identity and regex of internal function bodies overfit codegen strategy; test split-submodule execution behavior instead.
- `tests/compile/test_codegen.py::test_del_emitted_for_intermediate_values`: Emitted del source spelling/liveness strategy is unrelated to the stated NameError repair; not graded. Primitive multi-op execution stays covered.
- `tests/compile/test_codegen.py::test_distinct_dtypes_get_distinct_slots`: Exact const slots are not stated; replace with mixed-dtype split execution and dtype/content comparisons.
- `tests/compile/test_codegen.py::test_dtype_singleton_deduped`: Identity-based const-slot count and literal emitted reference are implementation details; replace with repeated-object execution for fresh/serialized artifacts.
- `tests/compile/test_codegen.py::test_legacy_code_without_consts`: Legacy artifact compatibility is stated and retained as actual callable execution.
- `tests/compile/test_codegen.py::test_node_ref_recurses_through_containers`: Direct helper signature/exact source strings/const slots overfit implementation. Replace with actual graph outputs containing nested list/tuple/dict and device/dtype/placement objects.
- `tests/compile/test_codegen.py::test_non_graphmodule_submod_uses_indexed_callable`: Internal emitted variable spelling/indexed-call syntax is not the behavioral contract; preserve wrapped non-GraphModule callable execution.
- `tests/compile/test_codegen.py::test_non_primitive_kwargs_lifted_to_consts[aten::_to_copy.default]`: Stock requires triple unpacking, const-pool membership and source exclusion. Replace with real traced conversion/allocation execution and eager comparison; permit any valid nonprimitive emission.
- `tests/compile/test_codegen.py::test_non_primitive_kwargs_lifted_to_consts[aten::empty.memory_format]`: Stock requires triple unpacking, const-pool membership and source exclusion. Replace with real traced conversion/allocation execution and eager comparison; permit any valid nonprimitive emission.
- `tests/compile/test_codegen.py::test_primitive_args_inlined`: An empty const list and inlining choice are not the semantic contract; preserve primitive execution in fresh/serialized P2P cases.
- `tests/eval_hidden/test_vllm_40726.py::test_generated_execution_matches_eager[fresh-allocation]`: Normal imports and actual generated callable execution. Compare tensor value/dtype/device or returned nested values; does not require the implementation to lift constants, deduplicate identity, or name private helpers. Pickle simulates execution-artifact serialization, not disk cache integration.
- `tests/eval_hidden/test_vllm_40726.py::test_generated_execution_matches_eager[fresh-conversion]`: Normal imports and actual generated callable execution. Compare tensor value/dtype/device or returned nested values; does not require the implementation to lift constants, deduplicate identity, or name private helpers. Pickle simulates execution-artifact serialization, not disk cache integration.
- `tests/eval_hidden/test_vllm_40726.py::test_generated_execution_matches_eager[fresh-repeated_objects]`: Normal imports and actual generated callable execution. Compare tensor value/dtype/device or returned nested values; does not require the implementation to lift constants, deduplicate identity, or name private helpers. Pickle simulates execution-artifact serialization, not disk cache integration.
- `tests/eval_hidden/test_vllm_40726.py::test_generated_execution_matches_eager[fresh-split_submodules]`: Normal imports and actual generated callable execution. Compare tensor value/dtype/device or returned nested values; does not require the implementation to lift constants, deduplicate identity, or name private helpers. Pickle simulates execution-artifact serialization, not disk cache integration.
- `tests/eval_hidden/test_vllm_40726.py::test_generated_execution_matches_eager[serialized-allocation]`: Normal imports and actual generated callable execution. Compare tensor value/dtype/device or returned nested values; does not require the implementation to lift constants, deduplicate identity, or name private helpers. Pickle simulates execution-artifact serialization, not disk cache integration.
- `tests/eval_hidden/test_vllm_40726.py::test_generated_execution_matches_eager[serialized-conversion]`: Normal imports and actual generated callable execution. Compare tensor value/dtype/device or returned nested values; does not require the implementation to lift constants, deduplicate identity, or name private helpers. Pickle simulates execution-artifact serialization, not disk cache integration.
- `tests/eval_hidden/test_vllm_40726.py::test_generated_execution_matches_eager[serialized-repeated_objects]`: Normal imports and actual generated callable execution. Compare tensor value/dtype/device or returned nested values; does not require the implementation to lift constants, deduplicate identity, or name private helpers. Pickle simulates execution-artifact serialization, not disk cache integration.
- `tests/eval_hidden/test_vllm_40726.py::test_generated_execution_matches_eager[serialized-split_submodules]`: Normal imports and actual generated callable execution. Compare tensor value/dtype/device or returned nested values; does not require the implementation to lift constants, deduplicate identity, or name private helpers. Pickle simulates execution-artifact serialization, not disk cache integration.
- `tests/eval_hidden/test_vllm_40726.py::test_nested_nonprimitive_graph_outputs[False]`: Normal imports and actual generated callable execution. Compare tensor value/dtype/device or returned nested values; does not require the implementation to lift constants, deduplicate identity, or name private helpers. Pickle simulates execution-artifact serialization, not disk cache integration.
- `tests/eval_hidden/test_vllm_40726.py::test_nested_nonprimitive_graph_outputs[True]`: Normal imports and actual generated callable execution. Compare tensor value/dtype/device or returned nested values; does not require the implementation to lift constants, deduplicate identity, or name private helpers. Pickle simulates execution-artifact serialization, not disk cache integration.
- `tests/eval_hidden/test_vllm_40726.py::test_legacy_artifact_without_special_objects_still_loads`: Normal imports and actual generated callable execution. Compare tensor value/dtype/device or returned nested values; does not require the implementation to lift constants, deduplicate identity, or name private helpers. Pickle simulates execution-artifact serialization, not disk cache integration.
- `tests/eval_hidden/test_vllm_40726.py::test_non_graphmodule_child_keeps_callable_behavior`: Normal imports and actual generated callable execution. Compare tensor value/dtype/device or returned nested values; does not require the implementation to lift constants, deduplicate identity, or name private helpers. Pickle simulates execution-artifact serialization, not disk cache integration.
- `tests/eval_hidden/test_vllm_40726.py::test_primitive_graph_still_executes[False]`: Normal imports and actual generated callable execution. Compare tensor value/dtype/device or returned nested values; does not require the implementation to lift constants, deduplicate identity, or name private helpers. Pickle simulates execution-artifact serialization, not disk cache integration.
- `tests/eval_hidden/test_vllm_40726.py::test_primitive_graph_still_executes[True]`: Normal imports and actual generated callable execution. Compare tensor value/dtype/device or returned nested values; does not require the implementation to lift constants, deduplicate identity, or name private helpers. Pickle simulates execution-artifact serialization, not disk cache integration.

## vllm-41266

Originally stock pytest: True. Synthetic overlay: True.

### Changes

- Replace stock orchestration tests with 13 normal-import tests executing real OpenAISpeechToText preprocessing/language-detection/chunk collection.
- Mock only audio I/O, model/config/renderer and engine boundaries; never patch cancellation, chunk iteration, ID construction, preprocessing or language-detection helper behavior.
- Check transcribe and translate with one/three chunks, propagation of outer CancelledError, exact submitted/aborted ID-set equality, all generators started and closed, language detection, successful ordered response and streaming delegation.
- Use generated PCM WAV bytes, actual soundfile decoding, duration measurement, and real audio splitting. Only model/engine construction remains substituted.

### Contract audit

- `tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py::test_language_detection_cancel_aborts_engine_request`: Stock cancellation behavior is statement-covered, but stock stubs preprocessing/model paths and fixes internal request suffixes/abort batching; replace with real orchestration comparing submitted ID sets, permitting batch or per-ID abort, and propagating outer-task cancellation. Language detection runs through actual preprocessing rather than calling a newly introduced helper.
- `tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py::test_non_streaming_cancel_aborts_engine_requests[engine_inputs0-expected_request_ids0]`: Stock cancellation behavior is statement-covered, but stock stubs preprocessing/model paths and fixes internal request suffixes/abort batching; replace with real orchestration comparing submitted ID sets, permitting batch or per-ID abort, and propagating outer-task cancellation. Language detection runs through actual preprocessing rather than calling a newly introduced helper.
- `tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py::test_non_streaming_cancel_aborts_engine_requests[engine_inputs1-expected_request_ids1]`: Stock cancellation behavior is statement-covered, but stock stubs preprocessing/model paths and fixes internal request suffixes/abort batching; replace with real orchestration comparing submitted ID sets, permitting batch or per-ID abort, and propagating outer-task cancellation. Language detection runs through actual preprocessing rather than calling a newly introduced helper.
- `tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py::test_non_streaming_cancel_advances_all_chunk_generators`: Stock cancellation behavior is statement-covered, but stock stubs preprocessing/model paths and fixes internal request suffixes/abort batching; replace with real orchestration comparing submitted ID sets, permitting batch or per-ID abort, and propagating outer-task cancellation. Language detection runs through actual preprocessing rather than calling a newly introduced helper.
- `tests/eval_hidden/test_vllm_41266.py::test_all_chunk_generators_are_advanced_and_cleaned_up`: Real speech-to-text code with deterministic engine events, no model/native audio/network. IDs compared to engine submissions, no literal suffix/order/batching constraints. Existing base entry method is used; no newly introduced helper name. Generator cleanup, cancellation, success and streaming requirements are explicit in task.
- `tests/eval_hidden/test_vllm_41266.py::test_cancellation_aborts_exact_submitted_ids_and_propagates[multiple_chunks-transcribe]`: Real speech-to-text code with deterministic engine events, no model/native audio/network. IDs compared to engine submissions, no literal suffix/order/batching constraints. Existing base entry method is used; no newly introduced helper name. Generator cleanup, cancellation, success and streaming requirements are explicit in task.
- `tests/eval_hidden/test_vllm_41266.py::test_cancellation_aborts_exact_submitted_ids_and_propagates[multiple_chunks-translate]`: Real speech-to-text code with deterministic engine events, no model/native audio/network. IDs compared to engine submissions, no literal suffix/order/batching constraints. Existing base entry method is used; no newly introduced helper name. Generator cleanup, cancellation, success and streaming requirements are explicit in task.
- `tests/eval_hidden/test_vllm_41266.py::test_cancellation_aborts_exact_submitted_ids_and_propagates[single_chunk-transcribe]`: Real speech-to-text code with deterministic engine events, no model/native audio/network. IDs compared to engine submissions, no literal suffix/order/batching constraints. Existing base entry method is used; no newly introduced helper name. Generator cleanup, cancellation, success and streaming requirements are explicit in task.
- `tests/eval_hidden/test_vllm_41266.py::test_cancellation_aborts_exact_submitted_ids_and_propagates[single_chunk-translate]`: Real speech-to-text code with deterministic engine events, no model/native audio/network. IDs compared to engine submissions, no literal suffix/order/batching constraints. Existing base entry method is used; no newly introduced helper name. Generator cleanup, cancellation, success and streaming requirements are explicit in task.
- `tests/eval_hidden/test_vllm_41266.py::test_language_detection_outer_cancellation_aborts_submitted_id[transcribe]`: Real speech-to-text code with deterministic engine events, no model/native audio/network. IDs compared to engine submissions, no literal suffix/order/batching constraints. Existing base entry method is used; no newly introduced helper name. Generator cleanup, cancellation, success and streaming requirements are explicit in task.
- `tests/eval_hidden/test_vllm_41266.py::test_language_detection_outer_cancellation_aborts_submitted_id[translate]`: Real speech-to-text code with deterministic engine events, no model/native audio/network. IDs compared to engine submissions, no literal suffix/order/batching constraints. Existing base entry method is used; no newly introduced helper name. Generator cleanup, cancellation, success and streaming requirements are explicit in task.
- `tests/eval_hidden/test_vllm_41266.py::test_streaming_still_delegates_without_collecting_or_aborting[transcribe]`: Real speech-to-text code with deterministic engine events, no model/native audio/network. IDs compared to engine submissions, no literal suffix/order/batching constraints. Existing base entry method is used; no newly introduced helper name. Generator cleanup, cancellation, success and streaming requirements are explicit in task.
- `tests/eval_hidden/test_vllm_41266.py::test_streaming_still_delegates_without_collecting_or_aborting[translate]`: Real speech-to-text code with deterministic engine events, no model/native audio/network. IDs compared to engine submissions, no literal suffix/order/batching constraints. Existing base entry method is used; no newly introduced helper name. Generator cleanup, cancellation, success and streaming requirements are explicit in task.
- `tests/eval_hidden/test_vllm_41266.py::test_success_preserves_order_and_never_aborts[1-transcribe]`: Real speech-to-text code with deterministic engine events, no model/native audio/network. IDs compared to engine submissions, no literal suffix/order/batching constraints. Existing base entry method is used; no newly introduced helper name. Generator cleanup, cancellation, success and streaming requirements are explicit in task.
- `tests/eval_hidden/test_vllm_41266.py::test_success_preserves_order_and_never_aborts[1-translate]`: Real speech-to-text code with deterministic engine events, no model/native audio/network. IDs compared to engine submissions, no literal suffix/order/batching constraints. Existing base entry method is used; no newly introduced helper name. Generator cleanup, cancellation, success and streaming requirements are explicit in task.
- `tests/eval_hidden/test_vllm_41266.py::test_success_preserves_order_and_never_aborts[3-transcribe]`: Real speech-to-text code with deterministic engine events, no model/native audio/network. IDs compared to engine submissions, no literal suffix/order/batching constraints. Existing base entry method is used; no newly introduced helper name. Generator cleanup, cancellation, success and streaming requirements are explicit in task.
- `tests/eval_hidden/test_vllm_41266.py::test_success_preserves_order_and_never_aborts[3-translate]`: Real speech-to-text code with deterministic engine events, no model/native audio/network. IDs compared to engine submissions, no literal suffix/order/batching constraints. Existing base entry method is used; no newly introduced helper name. Generator cleanup, cancellation, success and streaming requirements are explicit in task.

## vllm-41801

Originally stock pytest: True. Synthetic overlay: False.

### Changes

- Overlay only complete GT regression test files; no implementation/test utility overlays.
- Classify exact individual plugin node IDs; retain only PASSED-both p2p.

### Contract audit

- {'node_id': 'tests/tool_parsers/test_deepseekv32_tool_parser.py::TestExtractToolCalls::test_arguments_wrapper_repaired', 'check': 'Existing public parser extraction interface; assertion observes literal string preservation or wrapper unwrapping expressly required by statement. BASE fails with wrong JSON arguments, not an import/collection failure. No new private helper or new argument is required.'}
- {'node_id': 'tests/tool_parsers/test_deepseekv32_tool_parser.py::TestExtractToolCalls::test_input_wrapper_repaired', 'check': 'Existing public parser extraction interface; assertion observes literal string preservation or wrapper unwrapping expressly required by statement. BASE fails with wrong JSON arguments, not an import/collection failure. No new private helper or new argument is required.'}
- {'node_id': 'tests/tool_parsers/test_deepseekv32_tool_parser.py::TestExtractToolCalls::test_string_attr_true_preserves_literal_despite_schema', 'check': 'Existing public parser extraction interface; assertion observes literal string preservation or wrapper unwrapping expressly required by statement. BASE fails with wrong JSON arguments, not an import/collection failure. No new private helper or new argument is required.'}
- {'node_id': 'tests/tool_parsers/test_deepseekv32_tool_parser.py::TestExtractToolCallsStreaming::test_string_attr_true_preserves_literal_in_streaming', 'check': 'Existing public parser extraction interface; assertion observes literal string preservation or wrapper unwrapping expressly required by statement. BASE fails with wrong JSON arguments, not an import/collection failure. No new private helper or new argument is required.'}
- {'node_id': 'tests/tool_parsers/test_deepseekv4_tool_parser.py::test_extract_tool_calls_arguments_wrapper', 'check': 'Existing public parser extraction interface; assertion observes literal string preservation or wrapper unwrapping expressly required by statement. BASE fails with wrong JSON arguments, not an import/collection failure. No new private helper or new argument is required.'}

## vllm-40749

Originally stock pytest: True. Synthetic overlay: False.

### Changes

- Overlay only complete GT regression test files; no implementation/test utility overlays.
- Classify exact individual plugin node IDs; retain only PASSED-both p2p.

### Contract audit

- {'node_id': 'tests/v1/worker/test_gpu_model_runner.py::test_sample_tokens_receives_pp_sampled_ids_only_on_non_last_rank[2-True-0]', 'check': 'Existing GPUModelRunner.sample_tokens and existing pipeline receiver; construction and pipeline-group boundary are stubbed only. BASE performs one receive on the last rank instead of zero. No new symbols/arguments/errors required.'}

## vllm-41181

Originally stock pytest: False. Synthetic overlay: True.

### Changes

- Replace optional new maybe_make_thread_pool helper call with real HfRenderer construction and existing get_tokenizer API. No new helper symbol appears in synthetic tests.
- Use an unmodified local Rust WordLevel tokenizer; barrier-synchronized real batch tokenization with changing padding/truncation lengths is compared with every corresponding sequential result. No artificial borrow guard or changed-behavior mocks.
- Add real concurrent encode/decode/chat, standard pickle round-trip, and slow-tokenizer compatibility checks.
- Fix a fixture issue: Transformers 5 BertTokenizer is fast, so use a tiny local subclass of the actual slow PreTrainedTokenizer base instead.

### Contract audit

- {'node_id': 'tests/eval_hidden/test_vllm_41181.py::test_shared_fast_tokenizer_batches_match_sequential', 'check': 'Existing renderer/tokenizer interfaces only; BASE shows concrete differing tokenization results during simultaneous varied-length batches. GT passes concurrency; an equivalent lock-based implementation could also pass. All values/types are compared with sequential calls, not wrapper classes/pool internals.'}

### Exclusion

GT fails task-required existing tokenizer serialization compatibility: pickle.loads(pickle.dumps(renderer.get_tokenizer())) returns None, causing AttributeError on encode. BASE passes the same real public-entrypoint round-trip. The GT new reducer calls a pool initializer that has no return on the fast-tokenizer path. Preserve this behavioral compatibility test rather than deleting it to force a clean GT. Not an import/fixture problem or scorer defect.

## vllm-42162

Originally stock pytest: True. Synthetic overlay: False.

### Statement addition

Interface requirements: Preserve build_flat_image_bool_length(image_grids, hf_config), returning flattened token IDs and per-image lengths. Accept image_use_col_tokens (bool, default True), use_single_crop_col_tokens (bool or None, default None inheriting image_use_col_tokens), and use_single_crop_start_token (bool, default True) so processor settings can be supplied independently.

### Changes

- Overlay only complete GT regression test files; no implementation/test utility overlays.
- Classify exact individual plugin node IDs; retain only PASSED-both p2p.
- Parent shortened the original prose without changing the symptom/constraints so the new public-interface paragraph keeps the full statement within 250 words.

### Contract audit

- {'node_id': 'tests/models/multimodal/processing/test_molmo2.py::test_build_flat_image_bool_length_matches_molmoweb_processor_tokens', 'check': 'Existing public module function, but three keyword arguments are NEW and absent from original statement: image_use_col_tokens, use_single_crop_col_tokens, use_single_crop_start_token. BASE fails TypeError at call, not collection. Proposed Interface requirements paragraph makes this contract explicit; assertions check task-implied token counts/structural IDs, not private helper names. Inclusion requires parent to apply paragraph.'}
- {'node_id': 'tests/models/multimodal/processing/test_molmo2.py::test_build_flat_image_bool_length_respects_disabled_col_tokens', 'check': 'Existing public module function, but three keyword arguments are NEW and absent from original statement: image_use_col_tokens, use_single_crop_col_tokens, use_single_crop_start_token. BASE fails TypeError at call, not collection. Proposed Interface requirements paragraph makes this contract explicit; assertions check task-implied token counts/structural IDs, not private helper names. Inclusion requires parent to apply paragraph.'}

## vllm-41355

Originally stock pytest: False. Synthetic overlay: True.

### Changes

- Create normal-import ResponsesRequest pytest suite; no new validator/helper names or mock validators.
- Assert acceptance and preservation of meaningful content via JSON-mode model_dump, without requiring a specific dict-vs-model internal representation, generated-ID prefix, default status, or synthesized annotations.
- Cover assistant messages missing metadata/annotations, reasoning summaries, explicit metadata preservation, function calls and ordinary text.
- Keep invalid substantive-content checks, but consume lazily validated Iterable fields so equivalent eager or deferred validation is allowed. Initial constructor-only checks incorrectly assumed eager validation; these were corrected, not excluded.
- Remove an unnecessary input-dictionary immutability assertion; statement only requires preserved resulting content.

### Contract audit

- {'node_id': 'tests/eval_hidden/test_vllm_41355.py::test_output_items_can_be_resubmitted[assistant-missing-annotations]', 'check': 'Existing ResponsesRequest constructor and standard Pydantic JSON serialization; task explicitly requires round-trip assistant/reasoning inputs without output metadata to be accepted. BASE raises missing-field ValidationError. No exact new helper/model name, generated ID format, or implementation-dependent representation is asserted.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41355.py::test_output_items_can_be_resubmitted[assistant-missing-metadata]', 'check': 'Existing ResponsesRequest constructor and standard Pydantic JSON serialization; task explicitly requires round-trip assistant/reasoning inputs without output metadata to be accepted. BASE raises missing-field ValidationError. No exact new helper/model name, generated ID format, or implementation-dependent representation is asserted.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41355.py::test_output_items_can_be_resubmitted[reasoning-empty-summary]', 'check': 'Existing ResponsesRequest constructor and standard Pydantic JSON serialization; task explicitly requires round-trip assistant/reasoning inputs without output metadata to be accepted. BASE raises missing-field ValidationError. No exact new helper/model name, generated ID format, or implementation-dependent representation is asserted.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41355.py::test_output_items_can_be_resubmitted[reasoning-summary-content]', 'check': 'Existing ResponsesRequest constructor and standard Pydantic JSON serialization; task explicitly requires round-trip assistant/reasoning inputs without output metadata to be accepted. BASE raises missing-field ValidationError. No exact new helper/model name, generated ID format, or implementation-dependent representation is asserted.'}

## vllm-41269

Originally stock pytest: False. Synthetic overlay: True.

### Changes

- Replaced four oracle-private wrapper calls and implementation-specific wire flag tests with real public completion/chat/responses endpoint calls.
- Use real model validation and DP-header parsing with mocked external model registry and engine.
- Added nine no-cleanup controls, admitted completion control and cleanup-failure preservation cases.

### Contract audit

- Six F2P cases (three returned model errors and three escaping registry errors) assert cleanup via the async engine-client method explicitly named in the task, preserving request ID, KV metadata, DP rank and original error.
- No tests name the new private cleanup wrapper, private endpoint bodies or abort_immediately field; initialization discovers connector support through real base constructor.
- All endpoint/request types, registry collaboration, rendering boundary and header convention already exist on BASE.
- P2P cases check ordinary rejections, successful streaming admission, and client errors when cleanup is unavailable.

## vllm-41681

Originally stock pytest: True. Synthetic overlay: True.

### Changes

- Replaced three stock tests with equivalent per-test helper access so BASE yields individual failures rather than collection failure.
- Replaced fabricated unsupported-tensor mock with a real CPU bfloat16 tensor.
- Added zero-copy storage assertion and real embedding response/serializer boundary checks for float32, bfloat16, ordinary JSON and base64.
- Parent fairness correction: remove all three direct tests of the GT-only encode_pooling_output_float_or_ndarray helper. Score only the existing endpoint/serializer path, not a newly invented helper name.
- Remove direct tests of the GT-only encode_pooling_output_float_or_ndarray helper; exercise the existing public pooling response path instead.

### Contract audit

- Stock test_encode_pooling_output_float_or_ndarray_returns_numpy_array: new public helper is explicitly specified by the current task statement; name and return types are fair. Synthetic equivalent also verifies the stated zero-copy property.
- Stock test_orjson_serializes_numpy_array: serialization through real ORJSONResponse uses the same named public helper; only numerical values/schema matter.
- Stock test_encode_pooling_output_float_or_ndarray_falls_back_to_list: fake object assumed .numpy/.tolist implementation calls; replaced with real bfloat16 fallback to avoid imposing duck-typed implementation details.
- Synthetic helper F2P tests access the stated public function inside each test, never at collection.
- Synthetic endpoint float32 F2P uses existing BASE _openai_json_response (discoverable existing interface), real response building and ORJSON renderer; only the serialization boundary is spied on. No new private helpers or implementation details are required.
- Entire synthetic CPU file is run; no GPU selection or flaky/skip/xfail evidence used.
- {'check': 'No newly introduced helper symbols remain in the executable test. Real existing response construction and serializer inputs establish the float32 fast-path and bfloat16/JSON/base64 compatibility.'}

## vllm-41799

Originally stock pytest: False. Synthetic overlay: True.

### Changes

- Replace AST-extracted probe with normal import of complete Gemma4 module.
- Use real Gemma4ProcessingInfo planning and Gemma4MultiModalProcessor prompt construction; mock only model/config and HF-processor boundaries.
- Exercise all five supported numeric budgets, nested kwargs, malformed nested values, top-level precedence, prompt agreement, audio and fixed video accounting.
- Use only existing BASE interfaces, including the existing _get_prompt_updates hook, not the new private budget helper.

### Contract audit

- {'node_id': 'tests/eval_hidden/test_vllm_41799.py::test_encoder_planning_respects_effective_budget[nested-560]', 'check': 'Existing get_mm_max_tokens_per_item API, configured budget explicitly named in statement. BASE reports 280 instead of configured limit; no fixture/import failure.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41799.py::test_encoder_planning_respects_effective_budget[top-1120]', 'check': 'Existing get_mm_max_tokens_per_item API, configured budget explicitly named in statement. BASE reports 280 instead of configured limit; no fixture/import failure.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41799.py::test_encoder_planning_respects_effective_budget[top-140]', 'check': 'Existing get_mm_max_tokens_per_item API, configured budget explicitly named in statement. BASE reports 280 instead of configured limit; no fixture/import failure.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41799.py::test_encoder_planning_respects_effective_budget[top-560]', 'check': 'Existing get_mm_max_tokens_per_item API, configured budget explicitly named in statement. BASE reports 280 instead of configured limit; no fixture/import failure.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41799.py::test_encoder_planning_respects_effective_budget[top-70]', 'check': 'Existing get_mm_max_tokens_per_item API, configured budget explicitly named in statement. BASE reports 280 instead of configured limit; no fixture/import failure.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41799.py::test_encoder_planning_respects_effective_budget[top-overrides-nested]', 'check': 'Existing get_mm_max_tokens_per_item API, configured budget explicitly named in statement. BASE reports 280 instead of configured limit; no fixture/import failure.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41799.py::test_image_prompt_uses_effective_budget[nested-560]', 'check': 'Existing prompt replacement hook and get_image_repl API; BASE constructs 256 image placeholders instead of 529 for nested budget 560 on a square image. Tests compare against real pre-existing runtime image-sizing method, not fix helper or implementation recipe.'}

## vllm-42244

Originally stock pytest: False. Synthetic overlay: True.

### Changes

- Replaced script probe with seven deterministic real-module pytest cases.
- Added independent-storage, sound encoder, text-only and eager-consumption coverage.

### Contract audit

- The two reused-buffer cases assert named task behavior: final parameter values for language, vision, adapter and sound components.
- The streaming-consumption case enforces the explicit prohibition on buffering the full language checkpoint; it names no new helper.
- Existing load_weights entrypoint and existing checkpoint key prefixes are used; only model construction/submodule loaders are substituted.
- Stock tests were previously non-discriminating; none are used as F2P evidence.

## vllm-41932

Originally stock pytest: False. Synthetic overlay: True.

### Changes

- Converted embedded probe into eight deterministic real-wrapper tests.
- Compare all native argument positions, integer/probability conversions, output copyback and both draft-probability modes.
- Added legacy-call P2P tests and accepted multiple clear error types instead of oracle AssertionError alone.
- Replace recording native-op substitutes with the real compiled freeze-base CPU kernels. Verify actual token outputs and int32 copyback, not internal forwarding identities.

### Contract audit

- Existing BASE rejection_greedy_sample_kernel and rejection_random_sample_kernel launch adapters are used via normal imports; no new private symbols are named.
- Greedy shared-sampler F2P uses positional uniform/rate arguments and SYNTHETIC_MODE keyword copied from existing shared caller.
- Both random shared-sampler F2P cases use the existing caller convention including NO_DRAFT_PROBS; assert unchanged native positional contract.
- Two synthetic-mode F2P cases require a descriptive rejection mentioning synthetic acceptance, not an exact exception class/message; TypeError from signature mismatch is not accepted.
- Only external native op boundaries are mocked, not wrapper behavior; deterministic tensor values replace random probe inputs.

## vllm-41806

Originally stock pytest: True. Synthetic overlay: True.

### Changes

- Retained equivalent compile-factor and worker-env stock assertions in isolated synthetic file.
- Replaced stock real-Ray cluster with real CoreEngineActorManager runtime-env construction against mocked Ray scheduling; exercised real EngineCoreActorMixin initialization with only device setup stubbed.
- Added explicit actor override P2P and ordinary env propagation control.

### Contract audit

- Stock test_nixl_side_channel_host_is_not_compile_factor: fair direct public compile_factors contract; retained and extended across host changes.
- Stock TestExclusion.test_worker_specific_host_vars_are_excluded: fair existing environment-selection API and constant; retained with ordinary-variable control.
- Stock test_driver_nixl_side_channel_host_does_not_leak_to_engine_core_actor: fair behavior, but actual ray.init, placement group, and remote processes exceed this deterministic CPU evaluation scope; replaced scheduling boundary only, not the changed env selection or actor initialization.
- No assertion on newly introduced _set_nixl_side_channel_host private helper. Existing_set_visible_devices overridden only to avoid device construction as in stock test.
- No Ray cluster is started, no network or model construction. Entire synthetic suite selected without skip/xfail.

## vllm-39212

Originally stock pytest: False. Synthetic overlay: True.

### Statement addition

Interface requirements: Expose start_weight_update(is_checkpoint_format: bool = True) and finish_weight_update() across Worker, LLM, and AsyncLLM/EngineClient, alongside update_weights. Expose dev-mode HTTP POST /start_weight_update with optional is_checkpoint_format (default true), and POST /finish_weight_update; keep /update_weights and update_info. Invalid session transitions raise RuntimeError.

### Changes

- Added real CPU pytest cases through normal imports, real Worker constructor and real tiny torch model parameters.
- Parameterized checkpoint/kernel chunked sessions across Worker, LLM, AsyncLLM, and in-process HTTP -> AsyncLLM -> Worker; external transport, accelerator synchronization, layerwise reload implementation and RPC execution boundaries substituted.
- Added invalid-state and transport/HTTP legacy P2P controls.

### Contract audit

- No upstream pytest file was added by this implementation commit; original evaluation was a script probe.
- Public session names, boolean default, HTTP paths and RuntimeError type are explicitly stated in proposed statement addition, rather than hidden requirements.
- No AST extraction, source execution, new private helper assertions or private session flag access. Production worker manages session lifecycle, format dispatch and callbacks; external reload helper calls are observed only at start/finish boundaries.
- Real CPU parameters are checked after two chunks; both checkpoint and kernel formats exercised end-to-end. No native kernel math, NCCL group, GPU, checkpoint or network needed.
- The complete synthetic file is selected; no skip/xfail or dependency failure is used as a signal.

## vllm-42081

Originally stock pytest: False. Synthetic overlay: True.

### Changes

- Added 3 input/parameter dtype mismatch cases through normally imported mm_projector_forward and real KimiK25MultiModalProjector.forward.
- Replaced only ReplicatedLinear construction/kernel boundary with CPU torch.nn.Linear; no distributed setup or new helper assumptions.
- Asserted output shapes/dtypes/numerical equivalence, unchanged parameters and input tensors; added 2 matched-dtype P2P storage-identity checks.
- Relaxed no-copy storage guard to permit per-image as well as batched forwarding; no projector-call-count assumption.

### Contract audit

- {'node_id': 'tests/eval_hidden/test_vllm_42081.py::test_projector_accepts_vision_dtype[bf16_to_fp32]', 'check': 'Existing mm_projector_forward and projector class are present in BASE. Statement explicitly requires accepting vision dtype and outputting parameter dtype; BASE fails inside real LayerNorm/Linear on dtype mismatch, not an import or new-symbol failure.'}
- {'node_id': 'tests/eval_hidden/test_vllm_42081.py::test_projector_accepts_vision_dtype[fp32_to_bf16]', 'check': 'Existing mm_projector_forward and projector class are present in BASE. Statement explicitly requires accepting vision dtype and outputting parameter dtype; BASE fails inside real LayerNorm/Linear on dtype mismatch, not an import or new-symbol failure.'}
- {'node_id': 'tests/eval_hidden/test_vllm_42081.py::test_projector_accepts_vision_dtype[fp64_to_fp32]', 'check': 'Existing mm_projector_forward and projector class are present in BASE. Statement explicitly requires accepting vision dtype and outputting parameter dtype; BASE fails inside real LayerNorm/Linear on dtype mismatch, not an import or new-symbol failure.'}

## vllm-39243

Originally stock pytest: True. Synthetic overlay: True.

### Changes

- Replace stock LFM2 suite with semantic JSON-decoded assertions, preserving argument types, dotted names, parallel calls, content handling, chunking and request adjustment.
- Replace TOOL_CALL_REGEX monkeypatch with real malformed-input bounded-completion behavior; omit unstated echo-deletion heuristic.
- Add a P2P regression for the pre-existing pythonic parser, in the overlaid synthetic file itself.
- Strengthen the chunking check to reject partial as well as complete leaked tool markers, as explicitly required by the statement.

### Contract audit

- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_no_tool_call[False]', 'verdict': 'fair_public_parser_behavior_replaced_for_consistency', 'note': 'Uses the existing ToolParserManager.get_tool_parser interface with the lfm2 selection name explicitly specified in the statement. Non-tool passthrough, chunking, function arguments and request special-token preservation are stated requirements. Replacement keeps those observable behaviors, without prescribing JSON serialization whitespace.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_no_tool_call[True]', 'verdict': 'fair_public_parser_behavior_replaced_for_consistency', 'note': 'Uses the existing ToolParserManager.get_tool_parser interface with the lfm2 selection name explicitly specified in the statement. Non-tool passthrough, chunking, function arguments and request special-token preservation are stated requirements. Replacement keeps those observable behaviors, without prescribing JSON serialization whitespace.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_adjust_request_disables_skip_special_tokens', 'verdict': 'fair_public_parser_behavior_replaced_for_consistency', 'note': 'Uses the existing ToolParserManager.get_tool_parser interface with the lfm2 selection name explicitly specified in the statement. Non-tool passthrough, chunking, function arguments and request special-token preservation are stated requirements. Replacement keeps those observable behaviors, without prescribing JSON serialization whitespace.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_streaming_char_by_char_multi_dict_list', 'verdict': 'fair_public_parser_behavior_replaced_for_consistency', 'note': 'Uses the existing ToolParserManager.get_tool_parser interface with the lfm2 selection name explicitly specified in the statement. Non-tool passthrough, chunking, function arguments and request special-token preservation are stated requirements. Replacement keeps those observable behaviors, without prescribing JSON serialization whitespace.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_streaming_dotted_name_in_single_delta', 'verdict': 'fair_public_parser_behavior_replaced_for_consistency', 'note': 'Uses the existing ToolParserManager.get_tool_parser interface with the lfm2 selection name explicitly specified in the statement. Non-tool passthrough, chunking, function arguments and request special-token preservation are stated requirements. Replacement keeps those observable behaviors, without prescribing JSON serialization whitespace.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_streaming_full_block_and_trailing_in_single_delta', 'verdict': 'fair_public_parser_behavior_replaced_for_consistency', 'note': 'Uses the existing ToolParserManager.get_tool_parser interface with the lfm2 selection name explicitly specified in the statement. Non-tool passthrough, chunking, function arguments and request special-token preservation are stated requirements. Replacement keeps those observable behaviors, without prescribing JSON serialization whitespace.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_streaming_leading_block_and_trailing_in_single_delta', 'verdict': 'fair_public_parser_behavior_replaced_for_consistency', 'note': 'Uses the existing ToolParserManager.get_tool_parser interface with the lfm2 selection name explicitly specified in the statement. Non-tool passthrough, chunking, function arguments and request special-token preservation are stated requirements. Replacement keeps those observable behaviors, without prescribing JSON serialization whitespace.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_streaming_leading_content_and_full_block_in_single_delta', 'verdict': 'fair_public_parser_behavior_replaced_for_consistency', 'note': 'Uses the existing ToolParserManager.get_tool_parser interface with the lfm2 selection name explicitly specified in the statement. Non-tool passthrough, chunking, function arguments and request special-token preservation are stated requirements. Replacement keeps those observable behaviors, without prescribing JSON serialization whitespace.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_streaming_tool_call_with_large_steps', 'verdict': 'fair_public_parser_behavior_replaced_for_consistency', 'note': 'Uses the existing ToolParserManager.get_tool_parser interface with the lfm2 selection name explicitly specified in the statement. Non-tool passthrough, chunking, function arguments and request special-token preservation are stated requirements. Replacement keeps those observable behaviors, without prescribing JSON serialization whitespace.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_tool_call[simple_streaming]', 'verdict': 'replaced_exact_json_serialization_requirement', 'note': 'Tool-call semantics and selection name are stated, but FunctionCall equality compares a fixed JSON string, unnecessarily requiring json.dumps default spacing/order. Replacement json.loads arguments before comparison so equivalent encodings are accepted.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_tool_call[simple_nonstreaming]', 'verdict': 'replaced_exact_json_serialization_requirement', 'note': 'Tool-call semantics and selection name are stated, but FunctionCall equality compares a fixed JSON string, unnecessarily requiring json.dumps default spacing/order. Replacement json.loads arguments before comparison so equivalent encodings are accepted.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_tool_call[more_types_streaming]', 'verdict': 'replaced_exact_json_serialization_requirement', 'note': 'Tool-call semantics and selection name are stated, but FunctionCall equality compares a fixed JSON string, unnecessarily requiring json.dumps default spacing/order. Replacement json.loads arguments before comparison so equivalent encodings are accepted.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_tool_call[more_types_nonstreaming]', 'verdict': 'replaced_exact_json_serialization_requirement', 'note': 'Tool-call semantics and selection name are stated, but FunctionCall equality compares a fixed JSON string, unnecessarily requiring json.dumps default spacing/order. Replacement json.loads arguments before comparison so equivalent encodings are accepted.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_tool_call[parameterless_streaming]', 'verdict': 'replaced_exact_json_serialization_requirement', 'note': 'Tool-call semantics and selection name are stated, but FunctionCall equality compares a fixed JSON string, unnecessarily requiring json.dumps default spacing/order. Replacement json.loads arguments before comparison so equivalent encodings are accepted.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_tool_call[parameterless_nonstreaming]', 'verdict': 'replaced_exact_json_serialization_requirement', 'note': 'Tool-call semantics and selection name are stated, but FunctionCall equality compares a fixed JSON string, unnecessarily requiring json.dumps default spacing/order. Replacement json.loads arguments before comparison so equivalent encodings are accepted.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_tool_call[empty_dict_streaming]', 'verdict': 'replaced_exact_json_serialization_requirement', 'note': 'Tool-call semantics and selection name are stated, but FunctionCall equality compares a fixed JSON string, unnecessarily requiring json.dumps default spacing/order. Replacement json.loads arguments before comparison so equivalent encodings are accepted.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_tool_call[empty_dict_nonstreaming]', 'verdict': 'replaced_exact_json_serialization_requirement', 'note': 'Tool-call semantics and selection name are stated, but FunctionCall equality compares a fixed JSON string, unnecessarily requiring json.dumps default spacing/order. Replacement json.loads arguments before comparison so equivalent encodings are accepted.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_tool_call[empty_list_streaming]', 'verdict': 'replaced_exact_json_serialization_requirement', 'note': 'Tool-call semantics and selection name are stated, but FunctionCall equality compares a fixed JSON string, unnecessarily requiring json.dumps default spacing/order. Replacement json.loads arguments before comparison so equivalent encodings are accepted.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_tool_call[empty_list_nonstreaming]', 'verdict': 'replaced_exact_json_serialization_requirement', 'note': 'Tool-call semantics and selection name are stated, but FunctionCall equality compares a fixed JSON string, unnecessarily requiring json.dumps default spacing/order. Replacement json.loads arguments before comparison so equivalent encodings are accepted.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_tool_call[escaped_string_streaming]', 'verdict': 'replaced_exact_json_serialization_requirement', 'note': 'Tool-call semantics and selection name are stated, but FunctionCall equality compares a fixed JSON string, unnecessarily requiring json.dumps default spacing/order. Replacement json.loads arguments before comparison so equivalent encodings are accepted.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_tool_call[escaped_string_nonstreaming]', 'verdict': 'replaced_exact_json_serialization_requirement', 'note': 'Tool-call semantics and selection name are stated, but FunctionCall equality compares a fixed JSON string, unnecessarily requiring json.dumps default spacing/order. Replacement json.loads arguments before comparison so equivalent encodings are accepted.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_tool_call[parallel_calls_streaming]', 'verdict': 'replaced_exact_json_serialization_requirement', 'note': 'Tool-call semantics and selection name are stated, but FunctionCall equality compares a fixed JSON string, unnecessarily requiring json.dumps default spacing/order. Replacement json.loads arguments before comparison so equivalent encodings are accepted.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_tool_call[parallel_calls_nonstreaming]', 'verdict': 'replaced_exact_json_serialization_requirement', 'note': 'Tool-call semantics and selection name are stated, but FunctionCall equality compares a fixed JSON string, unnecessarily requiring json.dumps default spacing/order. Replacement json.loads arguments before comparison so equivalent encodings are accepted.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_tool_call[content_after_tool_call_nonstreaming]', 'verdict': 'replaced_exact_json_serialization_requirement', 'note': 'Tool-call semantics and selection name are stated, but FunctionCall equality compares a fixed JSON string, unnecessarily requiring json.dumps default spacing/order. Replacement json.loads arguments before comparison so equivalent encodings are accepted.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_tool_call[dotted_name_streaming]', 'verdict': 'replaced_exact_json_serialization_requirement', 'note': 'Tool-call semantics and selection name are stated, but FunctionCall equality compares a fixed JSON string, unnecessarily requiring json.dumps default spacing/order. Replacement json.loads arguments before comparison so equivalent encodings are accepted.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_tool_call[dotted_name_nonstreaming]', 'verdict': 'replaced_exact_json_serialization_requirement', 'note': 'Tool-call semantics and selection name are stated, but FunctionCall equality compares a fixed JSON string, unnecessarily requiring json.dumps default spacing/order. Replacement json.loads arguments before comparison so equivalent encodings are accepted.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_echoed_tool_call_body_not_leaked_to_content', 'verdict': 'omitted_underspecified_echo_suppression', 'note': 'This test additionally requires deleting an unmarked repeated Python call body after an end marker. Surrounding content is otherwise to be preserved; that special echo-deletion policy is not explicitly stated. Omitted rather than prescribing this implementation-specific heuristic.'}
- {'test_id': 'tests/tool_parsers/test_lfm2_tool_parser.py::test_regex_timeout_handling[False]', 'verdict': 'replaced_new_private_regex_attribute', 'note': 'patch.object requires new TOOL_CALL_REGEX and exactly one match() call. Equivalent parsers need not use a regex. Replacement sends real malformed input through the actual parser and checks bounded completion/no tool calls, without prescribing a regex or fallback-content policy.'}

### Exclusion

Fair real-parser tests expose a GT semantic gap: streaming character-by-character and seven-character chunks emit incomplete tool marker prefixes as assistant content, e.g. <|tool_call_start| and <|tool_call_st. The task explicitly requires arbitrary chunking without leaking markup. Retaining those assertions yields a nonzero GT pytest return code, so this task is excluded rather than weakening tests to make GT green.

## vllm-41423

Originally stock pytest: True. Synthetic overlay: True.

### Changes

- Replace xfail/skip stock outcomes with ordinary assertions on exception/skip propagation and subprocess side effects, plus a passing-body P2P case.
- Use importlib pytest import mode so the test module has a child-importable package path; observed failures are missing exceptions/files, not import errors.
- Do not overlay tests/utils.py: it is the changed implementation, and runner intentionally omits all tests/ implementation diffs from GT.
- Exclude: git diff COMMIT^ COMMIT --stat -- :!tests is empty, so the required GT has the same broken decorator as BASE.

### Contract audit

- {'test_id': 'tests/utils_/test_spawn_decorator.py::test_spawn_decorator_failure_is_caught', 'verdict': 'replaced_strict_xfail_and_wrapper_type_coupling', 'note': 'Original xfail(raises=RuntimeError, strict=True) is an XPASS/FAILED on broken BASE and XFAIL on the fixed upstream tree, not a PASSED-in-GT test. The task does not prescribe RuntimeError as wrapper type. Synthetic assertion accepts any ordinary Exception carrying the child sentinel, and tests skips and actual child side effects without xfail.'}

### Exclusion

Required GT definition excludes the entire implementation change in tests/utils.py. Both BASE and restricted GT execute the old decorator: two child exception/skip assertions fail and two argument/PID side-effect assertions fail identically. No eligible F2P node exists. Overlaying tests/utils.py would leak the implementation and is deliberately not done.

## vllm-42026

Originally stock pytest: True. Synthetic overlay: False.

### Changes

- Overlay both complete GT GLM parser regression files unchanged; run all CPU tests rather than restricting to the new cases.
- Keep both whitespace F2P nodes and all 60 PASSED-both nodes from these overlaid files.

### Contract audit

- {'test_id': 'tests/tool_parsers/test_glm4_moe_tool_parser.py::test_whitespace_preserved_in_arg_values', 'verdict': 'fair_existing_interface', 'note': 'Glm4MoeModelToolParser, ChatCompletionRequest, tools schema and extract_tool_calls all exist in BASE. Assertion decodes JSON and compares the exact string whitespace explicitly required by the statement; no new symbol/argument/error coupling.'}
- {'test_id': 'tests/tool_parsers/test_glm47_moe_tool_parser.py::TestGlm47ExtractToolCalls::test_whitespace_preserved_in_arg_values', 'verdict': 'fair_existing_interface', 'note': 'Existing GLM4.7 subclass and request fixture exercise the same public extract_tool_calls interface. Decoded city string must retain two leading/trailing spaces, explicitly required for both variants. No new private symbols.'}

## vllm-41499

Originally stock pytest: False. Synthetic overlay: True.

### Changes

- Materialize the originally non-stock probe as a deterministic real-module pytest suite.
- Replace direct private prefetch-helper invocation/new keyword coupling with DefaultModelLoader.get_all_weights over six tiny locally generated safetensors files.
- Observe real open/read/close calls, complete-byte coverage, configured read block sizes, and a peak-open-file upper bound for one and two workers; execute the real loader/iterator/prefetch/thread pool unchanged.
- Verify CLI human-readable 1M, explicit public defaults, and zero/negative config validation; first prove positive values are accepted so unknown-keyword rejection cannot accidentally pass validation tests.
- Add P2P default-loader I/O and no-new-flags CLI cases.

### Contract audit

- {'test_id': 'original audit_probe', 'verdict': 'replaced_new_helper_kwargs', 'note': 'Original probe calls_prefetch_all_checkpoints(num_prefetch_threads=..., block_size=...), new implementation-specific parameters not named by the statement. Replacement uses only the explicitly specified LoadConfig fields/CLI flags and pre-existing DefaultModelLoader.get_all_weights, with I/O observation. No direct helper-name or signature requirement.'}

## vllm-41965

Originally stock pytest: True. Synthetic overlay: False.

### Changes

- Overlay GT compressed-tensors regression file only, never implementation helpers under tests.
- Restrict the large GPU/checkpoint-centric file with -k 'find_matched_target_returns_none or get_scheme_dict_returns_none or lm_head': all five selected tests are CPU-only matching/scheme/config tests; 19 checkpoint/kernel/GPU tests are deliberately deselected.
- Keep two unmatched-target F2P and three existing lm_head P2P nodes.

### Contract audit

- {'test_id': 'tests/quantization/test_compressed_tensors.py::test_find_matched_target_returns_none_on_no_match', 'verdict': 'fair_existing_interface', 'note': 'find_matched_target and all four arguments are already present in BASE (utils.py:113). Returning None is the no-match result compatible with the existing scheme-resolution API, not a newly introduced helper name. The base ValueError is exactly the unmatched-target failure described in the statement.'}
- {'test_id': 'tests/quantization/test_compressed_tensors.py::test_get_scheme_dict_returns_none_on_no_match', 'verdict': 'fair_existing_interface', 'note': 'CompressedTensorsConfig.get_scheme_dict exists in BASE (compressed_tensors.py:797). Existing ignore handling returns None; requiring the same no-scheme result for an unspecified layer follows the task semantics. Only the Linear collaborator is mocked; real matching and scheme selection execute.'}

## vllm-41916

Originally stock pytest: False. Synthetic overlay: True.

### Changes

- Added 3 suffix-filtering cases and labelled metric identity case using real fetch_spec_decode_metrics.
- Mocked only session/HTTP response boundary, so no network is opened.
- Added totals-only multiengine aggregation and absent-metrics P2P guards.
- Use real aiohttp server/client over Unix-domain IPC: no HTTP response doubles, TCP, DNS, internet, or metric-parser substitution.

### Contract audit

- {'node_id': 'tests/eval_hidden/test_vllm_41916.py::test_labels_do_not_change_counter_identity', 'check': 'Existing scraper API; label values must not redefine metric identity. BASE counts metric keywords inside model_name labels and returns (480,150,0,{1:70}) rather than (100,300,150,{0:80,1:70}).'}
- {'node_id': 'tests/eval_hidden/test_vllm_41916.py::test_only_counter_totals_contribute[created]', 'check': 'Existing scraper API and return fields; statement requires ignoring all non-counter samples. BASE includes 1.7769e9 created/sum/gauge values in counts.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41916.py::test_only_counter_totals_contribute[gauge]', 'check': 'Existing scraper API and return fields; statement requires ignoring all non-counter samples. BASE includes 1.7769e9 created/sum/gauge values in counts.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41916.py::test_only_counter_totals_contribute[sum]', 'check': 'Existing scraper API and return fields; statement requires ignoring all non-counter samples. BASE includes 1.7769e9 created/sum/gauge values in counts.'}

## vllm-42188

Originally stock pytest: False. Synthetic overlay: True.

### Changes

- Added 3 missing/unmatched/null OpenAI tool-name cases and null/missing legacy tool-response names using the complete shipped template in ImmutableSandboxedEnvironment.
- Asserted required unknown fallback and retained content; matched id, named legacy response, ordinary conversation and legacy absent-name cases are P2P.
- No tokenizer/checkpoint or source extraction; template path is the pre-existing examples file named by the task, loaded from the tested checkout.

### Contract audit

- {'node_id': 'tests/eval_hidden/test_vllm_42188.py::test_legacy_unresolvable_name[null_name]', 'check': 'Task explicitly requires unknown placeholder and content for missing/unresolvable names in this pre-existing shipped template. BASE raises the reported NoneType concatenation TypeError during actual sandboxed rendering. No new helper, undocumented error text or interface required.'}
- {'node_id': 'tests/eval_hidden/test_vllm_42188.py::test_unresolvable_openai_tool_name[missing_id]', 'check': 'Task explicitly requires unknown placeholder and content for missing/unresolvable names in this pre-existing shipped template. BASE raises the reported NoneType concatenation TypeError during actual sandboxed rendering. No new helper, undocumented error text or interface required.'}
- {'node_id': 'tests/eval_hidden/test_vllm_42188.py::test_unresolvable_openai_tool_name[null_id_and_name]', 'check': 'Task explicitly requires unknown placeholder and content for missing/unresolvable names in this pre-existing shipped template. BASE raises the reported NoneType concatenation TypeError during actual sandboxed rendering. No new helper, undocumented error text or interface required.'}
- {'node_id': 'tests/eval_hidden/test_vllm_42188.py::test_unresolvable_openai_tool_name[unmatched_id]', 'check': 'Task explicitly requires unknown placeholder and content for missing/unresolvable names in this pre-existing shipped template. BASE raises the reported NoneType concatenation TypeError during actual sandboxed rendering. No new helper, undocumented error text or interface required.'}

## vllm-40891

Originally stock pytest: False. Synthetic overlay: True.

### Changes

- Replaced source-inspecting prior probe with actual UniProcExecutor construction and actual _init_executor/collective_rpc paths, mocking only worker/model/platform/network boundaries.
- Did not instantiate or assert on the newly introduced Future subclass/helper name; tests exercise existing executor RPC and Future result APIs only.
- Wrapped the real stdlib ThreadPoolExecutor constructor to record pool creation and safely shut down only test-owned pools, rather than inferring behavior from source.
- Covered lazy retrieval, caller-thread identity, cached value/error and single/list shape with both scheduling modes; preserved 4 immediate-output paths and no-pool sync initialization as P2P.

### Contract audit

- {'node_id': 'tests/eval_hidden/test_vllm_40891.py::test_async_output_is_lazy_cached_and_on_result_callers_thread[list-async_scheduler]', 'check': 'Existing collective_rpc and Future.result APIs; statement explicitly requires lazy, cached single/list results on result caller thread. BASE fetches eagerly in sync mode or on WorkerAsyncOutput_0 in async mode; GT fetches only at result on caller. No sleeps, source introspection or dependency on a new helper.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40891.py::test_async_output_is_lazy_cached_and_on_result_callers_thread[list-sync_scheduler]', 'check': 'Existing collective_rpc and Future.result APIs; statement explicitly requires lazy, cached single/list results on result caller thread. BASE fetches eagerly in sync mode or on WorkerAsyncOutput_0 in async mode; GT fetches only at result on caller. No sleeps, source introspection or dependency on a new helper.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40891.py::test_async_output_is_lazy_cached_and_on_result_callers_thread[single-async_scheduler]', 'check': 'Existing collective_rpc and Future.result APIs; statement explicitly requires lazy, cached single/list results on result caller thread. BASE fetches eagerly in sync mode or on WorkerAsyncOutput_0 in async mode; GT fetches only at result on caller. No sleeps, source introspection or dependency on a new helper.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40891.py::test_async_output_is_lazy_cached_and_on_result_callers_thread[single-sync_scheduler]', 'check': 'Existing collective_rpc and Future.result APIs; statement explicitly requires lazy, cached single/list results on result caller thread. BASE fetches eagerly in sync mode or on WorkerAsyncOutput_0 in async mode; GT fetches only at result on caller. No sleeps, source introspection or dependency on a new helper.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40891.py::test_initialization_creates_no_output_thread_pool[async_scheduler]', 'check': 'Statement explicitly prohibits executor background pools. Actual BASE initialization constructs ThreadPoolExecutor for async scheduling; real GT initialization does not. Spy wraps genuine constructor and does not mock the changed behavior.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40891.py::test_output_exception_is_lazy_and_cached[async_scheduler]', 'check': 'Existing collective_rpc and Future.result APIs; statement requires lazy retrieval and exception propagation. BASE fetches before result (caller thread with sync scheduling, WorkerAsyncOutput_0 with async scheduling); GT defers, caches the same error and invokes once on result caller. No exact error message or new class name required.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40891.py::test_output_exception_is_lazy_and_cached[sync_scheduler]', 'check': 'Existing collective_rpc and Future.result APIs; statement requires lazy retrieval and exception propagation. BASE fetches before result (caller thread with sync scheduling, WorkerAsyncOutput_0 with async scheduling); GT defers, caches the same error and invokes once on result caller. No exact error message or new class name required.'}

## vllm-40708

Originally stock pytest: False. Synthetic overlay: True.

### Changes

- Replaced GPU/checkpoint-dependent stock AWQ regression with real Gemma4Model.load_weights on an nn.Module parameter-tree double.
- Expanded original 2-case probe to all gate/up/down projections and 2 experts, with underscore packed/scale/zero_point metadata and dot qweight/scales/qzeros/weight plus bare weights.
- Asserted exact parameter object, source tensor values, mapped name, shard and expert routing through recording loader boundary; kept unrelated nonexpert unquantized weight loading as P2P.
- Did not overlay tests/models/quantization/test_awq.py or implementation helpers; entire synthetic suite runs on CPU without downloads.
- Fairness review relaxed source tensor identity to numerical equality, allowing equivalent fixes to pass a value-preserving tensor view/copy to the loader.

### Contract audit

- {'node_id': 'tests/eval_hidden/test_vllm_40708.py::test_checkpoint_expert_routes_to_parameter_and_shard[packed-down_proj-w2-w2-0]', 'check': 'Existing load_weights API and BASE fused expert parameter names/shard convention are discoverable; task explicitly requests both suffix conventions, metadata, all projections and multiple experts. BASE raises KeyError for this underscore-suffix source name, while GT routes to the intended existing fused parameter. No dependency on the newly used mapping helper or recipe for implementing the fix.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40708.py::test_checkpoint_expert_routes_to_parameter_and_shard[packed-down_proj-w2-w2-1]', 'check': 'Existing load_weights API and BASE fused expert parameter names/shard convention are discoverable; task explicitly requests both suffix conventions, metadata, all projections and multiple experts. BASE raises KeyError for this underscore-suffix source name, while GT routes to the intended existing fused parameter. No dependency on the newly used mapping helper or recipe for implementing the fix.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40708.py::test_checkpoint_expert_routes_to_parameter_and_shard[packed-gate_proj-w13-w1-0]', 'check': 'Existing load_weights API and BASE fused expert parameter names/shard convention are discoverable; task explicitly requests both suffix conventions, metadata, all projections and multiple experts. BASE raises KeyError for this underscore-suffix source name, while GT routes to the intended existing fused parameter. No dependency on the newly used mapping helper or recipe for implementing the fix.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40708.py::test_checkpoint_expert_routes_to_parameter_and_shard[packed-gate_proj-w13-w1-1]', 'check': 'Existing load_weights API and BASE fused expert parameter names/shard convention are discoverable; task explicitly requests both suffix conventions, metadata, all projections and multiple experts. BASE raises KeyError for this underscore-suffix source name, while GT routes to the intended existing fused parameter. No dependency on the newly used mapping helper or recipe for implementing the fix.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40708.py::test_checkpoint_expert_routes_to_parameter_and_shard[packed-up_proj-w13-w3-0]', 'check': 'Existing load_weights API and BASE fused expert parameter names/shard convention are discoverable; task explicitly requests both suffix conventions, metadata, all projections and multiple experts. BASE raises KeyError for this underscore-suffix source name, while GT routes to the intended existing fused parameter. No dependency on the newly used mapping helper or recipe for implementing the fix.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40708.py::test_checkpoint_expert_routes_to_parameter_and_shard[packed-up_proj-w13-w3-1]', 'check': 'Existing load_weights API and BASE fused expert parameter names/shard convention are discoverable; task explicitly requests both suffix conventions, metadata, all projections and multiple experts. BASE raises KeyError for this underscore-suffix source name, while GT routes to the intended existing fused parameter. No dependency on the newly used mapping helper or recipe for implementing the fix.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40708.py::test_checkpoint_expert_routes_to_parameter_and_shard[scale-down_proj-w2-w2-0]', 'check': 'Existing load_weights API and BASE fused expert parameter names/shard convention are discoverable; task explicitly requests both suffix conventions, metadata, all projections and multiple experts. BASE raises KeyError for this underscore-suffix source name, while GT routes to the intended existing fused parameter. No dependency on the newly used mapping helper or recipe for implementing the fix.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40708.py::test_checkpoint_expert_routes_to_parameter_and_shard[scale-down_proj-w2-w2-1]', 'check': 'Existing load_weights API and BASE fused expert parameter names/shard convention are discoverable; task explicitly requests both suffix conventions, metadata, all projections and multiple experts. BASE raises KeyError for this underscore-suffix source name, while GT routes to the intended existing fused parameter. No dependency on the newly used mapping helper or recipe for implementing the fix.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40708.py::test_checkpoint_expert_routes_to_parameter_and_shard[scale-gate_proj-w13-w1-0]', 'check': 'Existing load_weights API and BASE fused expert parameter names/shard convention are discoverable; task explicitly requests both suffix conventions, metadata, all projections and multiple experts. BASE raises KeyError for this underscore-suffix source name, while GT routes to the intended existing fused parameter. No dependency on the newly used mapping helper or recipe for implementing the fix.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40708.py::test_checkpoint_expert_routes_to_parameter_and_shard[scale-gate_proj-w13-w1-1]', 'check': 'Existing load_weights API and BASE fused expert parameter names/shard convention are discoverable; task explicitly requests both suffix conventions, metadata, all projections and multiple experts. BASE raises KeyError for this underscore-suffix source name, while GT routes to the intended existing fused parameter. No dependency on the newly used mapping helper or recipe for implementing the fix.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40708.py::test_checkpoint_expert_routes_to_parameter_and_shard[scale-up_proj-w13-w3-0]', 'check': 'Existing load_weights API and BASE fused expert parameter names/shard convention are discoverable; task explicitly requests both suffix conventions, metadata, all projections and multiple experts. BASE raises KeyError for this underscore-suffix source name, while GT routes to the intended existing fused parameter. No dependency on the newly used mapping helper or recipe for implementing the fix.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40708.py::test_checkpoint_expert_routes_to_parameter_and_shard[scale-up_proj-w13-w3-1]', 'check': 'Existing load_weights API and BASE fused expert parameter names/shard convention are discoverable; task explicitly requests both suffix conventions, metadata, all projections and multiple experts. BASE raises KeyError for this underscore-suffix source name, while GT routes to the intended existing fused parameter. No dependency on the newly used mapping helper or recipe for implementing the fix.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40708.py::test_checkpoint_expert_routes_to_parameter_and_shard[zero_point-down_proj-w2-w2-0]', 'check': 'Existing load_weights API and BASE fused expert parameter names/shard convention are discoverable; task explicitly requests both suffix conventions, metadata, all projections and multiple experts. BASE raises KeyError for this underscore-suffix source name, while GT routes to the intended existing fused parameter. No dependency on the newly used mapping helper or recipe for implementing the fix.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40708.py::test_checkpoint_expert_routes_to_parameter_and_shard[zero_point-down_proj-w2-w2-1]', 'check': 'Existing load_weights API and BASE fused expert parameter names/shard convention are discoverable; task explicitly requests both suffix conventions, metadata, all projections and multiple experts. BASE raises KeyError for this underscore-suffix source name, while GT routes to the intended existing fused parameter. No dependency on the newly used mapping helper or recipe for implementing the fix.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40708.py::test_checkpoint_expert_routes_to_parameter_and_shard[zero_point-gate_proj-w13-w1-0]', 'check': 'Existing load_weights API and BASE fused expert parameter names/shard convention are discoverable; task explicitly requests both suffix conventions, metadata, all projections and multiple experts. BASE raises KeyError for this underscore-suffix source name, while GT routes to the intended existing fused parameter. No dependency on the newly used mapping helper or recipe for implementing the fix.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40708.py::test_checkpoint_expert_routes_to_parameter_and_shard[zero_point-gate_proj-w13-w1-1]', 'check': 'Existing load_weights API and BASE fused expert parameter names/shard convention are discoverable; task explicitly requests both suffix conventions, metadata, all projections and multiple experts. BASE raises KeyError for this underscore-suffix source name, while GT routes to the intended existing fused parameter. No dependency on the newly used mapping helper or recipe for implementing the fix.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40708.py::test_checkpoint_expert_routes_to_parameter_and_shard[zero_point-up_proj-w13-w3-0]', 'check': 'Existing load_weights API and BASE fused expert parameter names/shard convention are discoverable; task explicitly requests both suffix conventions, metadata, all projections and multiple experts. BASE raises KeyError for this underscore-suffix source name, while GT routes to the intended existing fused parameter. No dependency on the newly used mapping helper or recipe for implementing the fix.'}
- {'node_id': 'tests/eval_hidden/test_vllm_40708.py::test_checkpoint_expert_routes_to_parameter_and_shard[zero_point-up_proj-w13-w3-1]', 'check': 'Existing load_weights API and BASE fused expert parameter names/shard convention are discoverable; task explicitly requests both suffix conventions, metadata, all projections and multiple experts. BASE raises KeyError for this underscore-suffix source name, while GT routes to the intended existing fused parameter. No dependency on the newly used mapping helper or recipe for implementing the fix.'}

## vllm-41237

Originally stock pytest: True. Synthetic overlay: True.

### Changes

- Replace all six original private-hook-coupled tests with actual VllmConfig initialization.
- Test three connector names with bare and combined allocator options, producing six genuine rejection F2P cases.
- Add five P2P exemptions: sleep mode, unset allocator, unrelated allocator option, explicit False, and no connector.
- Generate a tiny local GPT2 configuration for the sleep-mode case, with no model weights; only mock the device sleep-mode availability boundary.
- Use BASE-valid SchedulerConfig required arguments and omit model_config instead of explicitly passing None (pydantic rejects explicit None).
- No new hook names or mocked validation behavior remain in the pytest suite.

### Contract audit

- {'test_id': 'tests/v1/kv_connector/unit/test_config.py::test_kv_connector_rejects_expandable_segments[NixlConnector]', 'verdict': 'replaced_unfair_new_private_hook', 'note': 'The expected ValueError mentioning expandable_segments is specified, but_build_config directly calls the new private_verify_kv_transfer_compat hook, absent in BASE. Replaced by real public VllmConfig construction; arbitrary connector names are explicitly covered by the statement.'}
- {'test_id': 'tests/v1/kv_connector/unit/test_config.py::test_kv_connector_rejects_expandable_segments[MooncakeConnectorV1]', 'verdict': 'replaced_unfair_new_private_hook', 'note': 'The expected ValueError mentioning expandable_segments is specified, but_build_config directly calls the new private_verify_kv_transfer_compat hook, absent in BASE. Replaced by real public VllmConfig construction; arbitrary connector names are explicitly covered by the statement.'}
- {'test_id': 'tests/v1/kv_connector/unit/test_config.py::test_kv_connector_rejects_expandable_segments[SomeOOTConnector]', 'verdict': 'replaced_unfair_new_private_hook', 'note': 'The expected ValueError mentioning expandable_segments is specified, but_build_config directly calls the new private_verify_kv_transfer_compat hook, absent in BASE. Replaced by real public VllmConfig construction; arbitrary connector names are explicitly covered by the statement.'}
- {'test_id': 'tests/v1/kv_connector/unit/test_config.py::test_kv_connector_allows_expandable_segments_with_sleep_mode', 'verdict': 'replaced_spurious_f2p_private_hook', 'note': 'Original BASE failure is only AttributeError for the new private hook, not incorrect behavior. Replaced with construction-level test; these exemption behaviors now pass in both BASE and GT and are P2P.'}
- {'test_id': 'tests/v1/kv_connector/unit/test_config.py::test_kv_connector_allows_other_alloc_conf', 'verdict': 'replaced_spurious_f2p_private_hook', 'note': 'Original BASE failure is only AttributeError for the new private hook, not incorrect behavior. Replaced with construction-level test; these exemption behaviors now pass in both BASE and GT and are P2P.'}
- {'test_id': 'tests/v1/kv_connector/unit/test_config.py::test_no_kv_connector_ignores_expandable_segments', 'verdict': 'replaced_spurious_f2p_private_hook', 'note': 'Original BASE failure is only AttributeError for the new private hook, not incorrect behavior. Replaced with construction-level test; these exemption behaviors now pass in both BASE and GT and are P2P.'}

## vllm-41905

Originally stock pytest: False. Synthetic overlay: True.

### Changes

- Replaced fake-self/unbound-only probe with actual MiMoV2MTP and MiMoV2MultiTokenPredictor construction, forward and compute_logits.
- Only embedding, head/logits-kernel and MTP layer construction collaborators are small nn.Module doubles; actual changed wrapper/predictor behavior is never mocked.
- Covered positive token counts 1/2/3/8, prediction steps 0/1/2/5, input-id and explicit-embedding paths, forward argument forwarding, output shapes and separately asserted logits.
- Added 3-step chained hidden-state/logit checks and one-step construction/forward/logit P2P guards; no assertions on private constants, new helper names or number of newly invented parameters.
- Fairness review checks forwarded position/state tensor values, not Python object identity, to allow equivalent value-preserving routing implementations.

### Contract audit

- {'node_id': 'tests/eval_hidden/test_vllm_41905.py::test_constructed_multitoken_model_can_chain_predictions', 'check': 'Existing MiMoV2MTP constructor/configuration, explicitly named in task. BASE raises its num_speculative_tokens=1 restriction for token counts greater than one; GT constructs successfully. Chaining additionally observes 3 forward/logit steps through actual wrapper/predictor methods.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41905.py::test_construction_accepts_positive_speculative_token_counts[2]', 'check': 'Existing MiMoV2MTP constructor/configuration, explicitly named in task. BASE raises its num_speculative_tokens=1 restriction for token counts greater than one; GT constructs successfully. Chaining additionally observes 3 forward/logit steps through actual wrapper/predictor methods.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41905.py::test_construction_accepts_positive_speculative_token_counts[3]', 'check': 'Existing MiMoV2MTP constructor/configuration, explicitly named in task. BASE raises its num_speculative_tokens=1 restriction for token counts greater than one; GT constructs successfully. Chaining additionally observes 3 forward/logit steps through actual wrapper/predictor methods.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41905.py::test_construction_accepts_positive_speculative_token_counts[8]', 'check': 'Existing MiMoV2MTP constructor/configuration, explicitly named in task. BASE raises its num_speculative_tokens=1 restriction for token counts greater than one; GT constructs successfully. Chaining additionally observes 3 forward/logit steps through actual wrapper/predictor methods.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41905.py::test_hidden_state_path_reuses_available_layer[input_ids-1]', 'check': 'Existing forward(inputs_embeds=..., spec_step_idx=...) API in BASE; task requires later-step reuse and preserving arguments/shapes. BASE wrapper rejects nonzero steps; GT routes through its available layer and preserves positions/previous states/embeddings. Equivalent routing implementations need no new helper names.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41905.py::test_hidden_state_path_reuses_available_layer[input_ids-2]', 'check': 'Existing forward(inputs_embeds=..., spec_step_idx=...) API in BASE; task requires later-step reuse and preserving arguments/shapes. BASE wrapper rejects nonzero steps; GT routes through its available layer and preserves positions/previous states/embeddings. Equivalent routing implementations need no new helper names.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41905.py::test_hidden_state_path_reuses_available_layer[input_ids-5]', 'check': 'Existing forward(inputs_embeds=..., spec_step_idx=...) API in BASE; task requires later-step reuse and preserving arguments/shapes. BASE wrapper rejects nonzero steps; GT routes through its available layer and preserves positions/previous states/embeddings. Equivalent routing implementations need no new helper names.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41905.py::test_hidden_state_path_reuses_available_layer[inputs_embeds-1]', 'check': 'Existing forward(inputs_embeds=..., spec_step_idx=...) API in BASE; task requires later-step reuse and preserving arguments/shapes. BASE wrapper rejects nonzero steps; GT routes through its available layer and preserves positions/previous states/embeddings. Equivalent routing implementations need no new helper names.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41905.py::test_hidden_state_path_reuses_available_layer[inputs_embeds-2]', 'check': 'Existing forward(inputs_embeds=..., spec_step_idx=...) API in BASE; task requires later-step reuse and preserving arguments/shapes. BASE wrapper rejects nonzero steps; GT routes through its available layer and preserves positions/previous states/embeddings. Equivalent routing implementations need no new helper names.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41905.py::test_hidden_state_path_reuses_available_layer[inputs_embeds-5]', 'check': 'Existing forward(inputs_embeds=..., spec_step_idx=...) API in BASE; task requires later-step reuse and preserving arguments/shapes. BASE wrapper rejects nonzero steps; GT routes through its available layer and preserves positions/previous states/embeddings. Equivalent routing implementations need no new helper names.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41905.py::test_logits_path_accepts_later_speculative_steps[1]', 'check': 'Existing compute_logits(spec_step_idx=...) API in BASE; nonzero indices are explicitly requested. BASE wrapper asserts step==0; GT returns expected head logits for the same hidden state. No new symbol, error message or private layer-store requirement.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41905.py::test_logits_path_accepts_later_speculative_steps[2]', 'check': 'Existing compute_logits(spec_step_idx=...) API in BASE; nonzero indices are explicitly requested. BASE wrapper asserts step==0; GT returns expected head logits for the same hidden state. No new symbol, error message or private layer-store requirement.'}
- {'node_id': 'tests/eval_hidden/test_vllm_41905.py::test_logits_path_accepts_later_speculative_steps[5]', 'check': 'Existing compute_logits(spec_step_idx=...) API in BASE; nonzero indices are explicitly requested. BASE wrapper asserts step==0; GT returns expected head logits for the same hidden state. No new symbol, error message or private layer-store requirement.'}
