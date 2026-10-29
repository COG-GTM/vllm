# vLLM historical-commit evaluation tasks

Freeze/base: `98661fe012c5c467252d4df8411d2f46190e9268` (2026-05-05). Pool: the 200 squash merges in `98661fe012c5c467252d4df8411d2f46190e9268..d7af6b34d83fc691ad69347ee4d066231e5678ab` (head 2026-05-11). Repository: [COG-GTM/vllm](https://github.com/COG-GTM/vllm).

Ranks 1–20 are primary; ranks 21–30 are held-out/alternates. Primary difficulty mix: **6 easy / 9 medium / 5 hard**. Ranking prioritizes CPU discrimination and base independence, then area coverage and behavioral signal; difficulty reflects reasoning/integration risk, not just diff size. This is an oracle dataset: give an eval agent **only `task_statement`**, never this document or the remaining JSON fields.

## Verification and scoring contract

All selected patches passed a full binary-diff `git apply --cached --check` against the freeze tree using an isolated index. Earlier file overlaps are recorded per task. `applies_cleanly_to_base` means mechanical application, not whole-vLLM runtime certification. Ground-truth tests were copied into the parent without implementation changes before parent testing; parent API-absence collection errors are labeled rather than presented as assertion failures. Separate base-plus-patch results below establish additional portability where run.

`cpu_verified: true` means the listed behavioral CPU test/probe actually ran and discriminated the commit from its parent. It does **not** mean the upstream suite, model inference, GPU path or multi-node system was verified. `stock_cpu_verified` and `verification_level` distinguish unmodified stock pytest, normally imported real code with mocks, and AST/source-isolated probes. Synthetic audit probes are evidence, not ready-made implementation-neutral grading suites; author/adapt the hidden tests next. Never require an internal helper merely because the original PR introduced one.

### Environment and reproduction

Audits ran on CPU-only Linux with Python 3.12 managed by uv, a precompiled editable vLLM install and PyTorch 2.11.0+cpu. The coordinator independently confirmed `torch.cuda.is_available() == False`. Test-only packages installed as needed included pytest-asyncio, pytest-timeout, tblib, ray and orjson. Parser/model-processor tests may fetch tokenizer/config assets, not weights. Upstream remains read-only.

```bash
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install pytest pytest-asyncio pytest-timeout tblib ray orjson
```

Run commands from the tested source-worktree root with its `.venv` (or substitute an absolute shared uv venv). For parent comparisons, overlay only commit-added/changed regression test files onto the parent. For eval-base comparisons, overlay those tests on the freeze base, then compare base+tests with base+full-patch. Do not expose future source history or oracle metadata to eval agents. Some original commands retain audit-VM absolute paths in `audit_executed_command` for provenance; portable stock commands and proposed hidden-test commands are separate fields. Where `audit_probe.source` is present in JSON, materialize it verbatim under `audit-probes/` before using `probe_command`; missing probe sources mean author the described test next, not that a nonexistent pytest has run.

## Ranked task table

| Rank | Task / upstream context (read-only) | Area | Difficulty | Diff (+/−) | CPU evidence |
| --- | --- | --- | --- | --- | --- |
| 1 | [vllm-41991](https://github.com/vllm-project/vllm/pull/41991) — [Bugfix][Gemma4] Fix infinite loop and array boundary issues in tool parser | tool/reasoning parsers | easy | 34/0 | stock_pytest |
| 2 | [vllm-40951](https://github.com/vllm-project/vllm/pull/40951) — Handle optional bool-or-string CLI args in get_kwargs | config/args | easy | 42/11 | stock_pytest |
| 3 | [vllm-41795](https://github.com/vllm-project/vllm/pull/41795) — [KV Offload] Return None from lookup() for in-flight blocks | kv_offload | easy | 11/7 | stock_pytest |
| 4 | [vllm-41730](https://github.com/vllm-project/vllm/pull/41730) — [BUGFIX] Support streamed_args_for_tool in MistralToolParser (#41730) | tool parsers / Mistral streaming | medium | 55/15 | stock_pytest |
| 5 | [vllm-40726](https://github.com/vllm-project/vllm/pull/40726) — [Bugfix] Fix codegen for unqualified names (#40726) | compilation / FX code generation and cache | hard | 450/32 | stock_pytest |
| 6 | [vllm-41266](https://github.com/vllm-project/vllm/pull/41266) — [Frontend][Bugfix] Abort ASR engine requests on cancellation | API / OpenAI entrypoints (speech-to-text) | medium | 284/54 | stock_pytest |
| 7 | [vllm-41801](https://github.com/vllm-project/vllm/pull/41801) — [Bugfix] DeepSeekV32/v4: respect string='true/false' attribute and unwrap arguments/input wrapper | tool parsers | medium | 224/10 | stock_pytest |
| 8 | [vllm-40749](https://github.com/vllm-project/vllm/pull/40749) — [Bugfix] Skip PP sampled-token receive on last rank during async scheduling (#40749) | core execution / pipeline-parallel async scheduling | easy | 54/1 | stock_pytest |
| 9 | [vllm-41181](https://github.com/vllm-project/vllm/pull/41181) — [Bugfix] Fix `RuntimeError: Already borrowed` by adding thread-safe Hugging Face fast-tokenizer wrappers (#41181) | tokenizers / concurrent rendering | hard | 107/35 | real_module_cpu_probe_with_mocks |
| 10 | [vllm-42162](https://github.com/vllm-project/vllm/pull/42162) — Fix Molmo2 image token metadata | multimodal processing | medium | 94/10 | stock_pytest |
| 11 | [vllm-41355](https://github.com/vllm-project/vllm/pull/41355) — [Frontend] Supports resubmitting output items with missing fields in Responses API (#41355) | API / Responses request validation | medium | 60/10 | real_module_cpu_probe_with_mocks |
| 12 | [vllm-41269](https://github.com/vllm-project/vllm/pull/41269) — [Bugfix][KV Transfer][NIXL] Notify P node on pre-admission rejection to free stranded KV blocks | KV transfer / engine core + API | hard | 156/2 | real_module_cpu_probe_with_mocks |
| 13 | [vllm-41681](https://github.com/vllm-project/vllm/pull/41681) — [Perf] Use numpy zero-copy path for embedding float response serialization | API / pooling entrypoints | medium | 105/0 | stock_pytest |
| 14 | [vllm-41799](https://github.com/vllm-project/vllm/pull/41799) — [MM][Gemma4] Respect max_soft_tokens in encoder budget (#41799) | multimodal processing / encoder budget | medium | 91/11 | isolated_cpu_probe |
| 15 | [vllm-42244](https://github.com/vllm-project/vllm/pull/42244) — Avoid silent weights corruption when loading Nemotron Nano VL with reusable-buffer loaders like runai distributed streaming | Model loading / multimodal model weights | medium | 55/28 | real_module_cpu_probe_with_mocks |
| 16 | [vllm-41932](https://github.com/vllm-project/vllm/pull/41932) — [CPU] Fix spec decode kernel signatures for synthetic mode compatibility | speculative decoding / sampling (CPU) | easy | 15/0 | real_python_wrappers_with_stubbed_native_ops |
| 17 | [vllm-41806](https://github.com/vllm-project/vllm/pull/41806) — fix nixl side-channel host selection | KV transfer / distributed env propagation (Ray, engine-core actors, env compile factors) | medium | 179/2 | stock_pytest |
| 18 | [vllm-39212](https://github.com/vllm-project/vllm/pull/39212) — [feat] Add explicit /start_weight_update and /finish_weight_update APIs for weight transfer | API / RLHF weight transfer | hard | 288/54 | real_module_cpu_probe_with_mocks |
| 19 | [vllm-42081](https://github.com/vllm-project/vllm/pull/42081) — [Bug] Fix kimi dtype issue with `mm_projector_forward` | Multimodal model (Kimi-K2.5/K2.6 vision projector) | easy | 3/0 | real_module_cpu_probe_with_mocks |
| 20 | [vllm-39243](https://github.com/vllm-project/vllm/pull/39243) — add: LFM2/2.5 Tool Parser | tool/reasoning parsers | hard | 864/5 | stock_pytest |
| 21 | [vllm-41423](https://github.com/vllm-project/vllm/pull/41423) — [Bugfix] Fix spawn_new_process_for_each_test silently swallowing test failures | utils/test_infra | medium | 306/245 | stock_pytest |
| 22 | [vllm-42026](https://github.com/vllm-project/vllm/pull/42026) — [Bugfix] Preserve leading/trailing whitespace in GLM non-streaming tool parser | tool_parsers | easy | 40/3 | stock_pytest |
| 23 | [vllm-41499](https://github.com/vllm-project/vllm/pull/41499) — [Performance] Make safetensors checkpoint prefetch settings configurable | model loading / config | medium | 87/15 | real_module_cpu_probe_with_mocks |
| 24 | [vllm-41965](https://github.com/vllm-project/vllm/pull/41965) — [Compressed Tensors] Allow configs with non-explicit ignores | model loading/quantization config | easy | 34/19 | stock_pytest |
| 25 | [vllm-41916](https://github.com/vllm-project/vllm/pull/41916) — Fix spec decode benchmark metrics | benchmarks / spec decode metrics | easy | 23/19 | real_module_cpu_probe_with_mocks |
| 26 | [vllm-42188](https://github.com/vllm-project/vllm/pull/42188) — [Bugfix] Gemma 4 chat template crash with missing tool name and tool id | tool calling / chat templates | easy | 2/2 | real_module_cpu_probe_with_mocks |
| 27 | [vllm-40891](https://github.com/vllm-project/vllm/pull/40891) — [Core] Avoid using extra thread in `UniProcExecutor` | core/executor | medium | 21/16 | real_module_cpu_probe_with_mocks |
| 28 | [vllm-40708](https://github.com/vllm-project/vllm/pull/40708) — [BugFix] Fix Gemma4 'layers.0.moe.experts.0.down_proj_packed' KeyError issue | model loading | medium | 55/18 | real_module_cpu_probe_with_mocks |
| 29 | [vllm-41237](https://github.com/vllm-project/vllm/pull/41237) — [Bugfix][KV Transfer] Reject NixlConnector + expandable_segments:True | config/KV transfer | easy | 98/0 | stock_pytest |
| 30 | [vllm-41905](https://github.com/vllm-project/vllm/pull/41905) — [SpecDecoding] extend mtp support for mimo 2.5 | spec_decode/models | easy | 3/10 | real_module_cpu_probe_with_mocks |

## Rejected categories

Of the **200 commits**, **30 were selected** and **170 were excluded** under mutually exclusive primary reasons: **36 low-signal/trivial/test-or-example-only/lower-ranked**; **57 gpu/native/hardware/model-dependent**; **7 dependency/optional-package coupling**; **18 refactors/renames/removals**; **32 ci/build/repository infrastructure**; **3 non-discriminating cpu evidence**; **12 documentation only**; **5 base/dependency incompatibility**. These counts include the 17 audited candidates not selected, not only the initial 153 rejects; 14 were diversity/ranking alternatives, two failed independent-base suitability, and one had non-discriminating CPU evidence. Native CPU kernels were also deprioritized because they require compiled-kernel rather than Python-level verification. No docs-only task was selected. Reverts and pure dependency bumps were not promoted. The ledger at the end makes the accounting auditable.

## 1. vllm-41991 — [Bugfix][Gemma4] Fix infinite loop and array boundary issues in tool parser

**Set:** primary; **area:** tool/reasoning parsers; **difficulty:** easy.

**Ground truth:** [dbd86a67e3ee7ba95ce90292bb21006b9761faee](https://github.com/COG-GTM/vllm/commit/dbd86a67e3ee7ba95ce90292bb21006b9761faee); **parent:** `2c6b59b80771ac3bfa1c789abe3cb27c379bf3a1`.

**Files:** 2; **lines:** +34/−0.

- `tests/tool_parsers/test_gemma4_tool_parser.py`
- `vllm/tool_parsers/gemma4_tool_parser.py`

### Agent task statement

The Gemma4 tool-call parser can hang a request on malformed or adversarial output. An incomplete bare value inside an array, such as key:[ followed by truncated content, or a stray closing bracket where an element is expected, can leave parsing running indefinitely. Separately, array strings containing literal square brackets are sometimes split or truncated. For example, with the Gemma4 string delimiter, [<|"|>a]b<|"|>] should represent one string element whose value is a]b. Please make parsing terminate in bounded time even on malformed fragments, returning a best-effort partial result or safely rejecting content rather than hanging. Well-formed strings containing either bracket must retain their exact contents. Existing tool-call behavior, streaming and non-streaming output, and the public response format must stay compatible. Demonstrate the malformed-input cases without loading a model or requiring an accelerator.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** True; **level:** `stock_pytest`.

**Ground-truth test files:** `tests/tool_parsers/test_gemma4_tool_parser.py`

**Recorded test IDs:**

- `tests/tool_parsers/test_gemma4_tool_parser.py::TestParseGemma4Args::test_malformed_partial_array`
- `tests/tool_parsers/test_gemma4_tool_parser.py::TestParseGemma4Array::test_string_element_with_closing_bracket`
- `tests/tool_parsers/test_gemma4_tool_parser.py::TestParseGemma4Array::test_stray_closing_bracket`

**Pytest / probe command and status:**

Executed stock pytest; normalized worktree and venv locations. Parent/base runs overlay only the ground-truth regression tests, never implementation.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest -q -p no:cacheprovider tests/tool_parsers/test_gemma4_tool_parser.py
```

**Observed ground truth:** STOCK pytest, full file collected and run: 49 passed, 14 warnings

**Observed parent:** STOCK pytest with commit's test file copied onto parent (impl not applied): 3 failed, 46 passed. Failures = the 3 new tests; 2x 'Failed: Timeout (>5.0s) from pytest-timeout' (infinite loop) + 1 failure on string-with-bracket case

Executed on CPU (torch 2.11.0+cpu). New tests carry @pytest.mark.timeout(5); requires pytest-timeout installed or parent hangs instead of failing. Logs: /home/ubuntu/wt/logs/tests/gemma4_{parent,commit}.log

**Coordinator freeze-base rerun:**

```json
[
  {
    "pr_number": 41991,
    "mode": "fix",
    "exit_code": 0,
    "summary": [
      "49 passed in 10.30s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/tool_parsers/test_gemma4_tool_parser.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  },
  {
    "pr_number": 41991,
    "mode": "tests",
    "exit_code": 1,
    "summary": [
      "3 failed, 46 passed in 22.28s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/tool_parsers/test_gemma4_tool_parser.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  }
]
```

**Synthetic test idea:** Not required for the demonstrated stock CPU regression; retain equivalent behavioral coverage.

### Rubric

- Parser terminates on every malformed input (no zero-progress loops) in both args and array parsing paths
- Brackets inside <|"|>-delimited strings are not counted for array nesting
- Malformed fragments degrade gracefully (partial/skip) rather than raising
- Existing well-formed parsing and streaming behavior unchanged (46 pre-existing tests stay green)
- Fix is localized; no interface/output-format change

### Freeze-base suitability

**Applies cleanly:** True. No overlap: neither changed file touched in 98661fe01..dbd86a67e^. In disposable worktree at 98661fe01: `git diff dbd86a67e^ dbd86a67e | git apply --check` -> OK; `git cherry-pick --no-commit dbd86a67e` -> OK, 0 conflicts. Test only imports module-private helpers already present in base.

Earlier file overlaps: `[]`.

**Why high signal:** Real hang/correctness bug in a pure-Python parser with crisp, fast, deterministic regression tests that fail on parent (timeouts) and pass at fix.

## 2. vllm-40951 — Handle optional bool-or-string CLI args in get_kwargs

**Set:** primary; **area:** config/args; **difficulty:** easy.

**Ground truth:** [0d382ecde8c6771fe07854eaad3af73c1353e031](https://github.com/COG-GTM/vllm/commit/0d382ecde8c6771fe07854eaad3af73c1353e031); **parent:** `1029e5ef282ec4f705de99b91418e956c835c315`.

**Files:** 2; **lines:** +42/−11.

- `tests/engine/test_arg_utils.py`
- `vllm/engine/arg_utils.py`

### Agent task statement

Engine CLI options whose config field is typed as `bool | str | None` are not exposed correctly on the command line. The canonical example is `--hf-token`: users expect to be able to (a) omit the flag entirely, (b) pass it bare to mean "use the token from the local Hugging Face login", or (c) pass an explicit token string. Today the argparse kwargs generated for such a field do not support this, so the bare form and/or the explicit-value form do not round-trip into `EngineArgs` the way the type annotation implies.

Expected behavior after the fix, for any dataclass config field annotated `bool | str | None` that is turned into a CLI argument:

- flag absent -> value stays `None`
- bare flag (`--hf-token`) -> `True`
- flag with a value (`--hf-token hf_abc`) -> the string `"hf_abc"`
- the literal value `None` given on the CLI is treated as an ordinary string, not converted to Python `None`.

Constraints: the generated argparse kwargs should declare a string type for such fields; existing behavior for plain `bool`, `bool | None`, `str | None`, literal/choice and other union types must be unchanged, and the parser must still work through `EngineArgs.add_cli_args` / `FlexibleArgumentParser`.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** True; **level:** `stock_pytest`.

**Ground-truth test files:** `tests/engine/test_arg_utils.py`

**Recorded test IDs:**

- `tests/engine/test_arg_utils.py::test_hf_token_get_kwargs`
- `tests/engine/test_arg_utils.py::test_hf_token_cli_arg`

**Pytest / probe command and status:**

Executed stock pytest; normalized worktree and venv locations. Parent/base runs overlay only the ground-truth regression tests, never implementation.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest -q -p no:cacheprovider tests/engine/test_arg_utils.py
```

**Observed ground truth:** stock pytest, full file: 72 passed, 14 warnings in 33.06s (EXIT 0)

**Observed parent:** stock test file copied into parent 0d382ecde^ without implementation: 2 failed, 70 passed (new hf_token tests fail with `KeyError: 'type'`)

Stock commit test file, CPU torch 2.11.0+cpu. Base matrix at 98661fe01: patch+tests 72 passed; tests-only 2 failed, 70 passed. Logs: /home/ubuntu/audit/logs/0d382ecde-{gt,par}.log, base-0d382ecde-{fix,tst}.log. Note test_hf_token_cli_arg is parametrized; the 2 parent failures are the get_kwargs assertion and the bare/explicit CLI cases.

**Coordinator freeze-base rerun:**

```json
[
  {
    "pr_number": 40951,
    "mode": "fix",
    "exit_code": 0,
    "summary": [
      "72 passed, 14 warnings in 33.46s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/engine/test_arg_utils.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  },
  {
    "pr_number": 40951,
    "mode": "tests",
    "exit_code": 1,
    "summary": [
      "2 failed, 70 passed, 14 warnings in 13.76s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/engine/test_arg_utils.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  }
]
```

**Synthetic test idea:** Not required for the demonstrated stock CPU regression; retain equivalent behavioral coverage.

### Rubric

- Bare `--hf-token` parses to True and `--hf-token X` to 'X'; absent flag stays None.
- Generated kwargs for `bool | str | None` declare type=str (and optional-value nargs) rather than a bool action.
- The string 'None' is not coerced to Python None.
- No regressions for other Optional/union/Literal field handling in get_kwargs (full test_arg_utils.py passes).
- Change is generic over the type hint, not special-cased to the hf_token field name.

### Freeze-base suitability

**Applies cleanly:** True. Mechanical: base-matrix script /home/ubuntu/audit/basematrix.sh created detached worktree at 98661fe01 and ran `git -C /home/ubuntu/repos/vllm diff 0d382ecde^ 0d382ecde | git apply` -> applied with no errors. No overlapping files in 98661fe01..0d382ecde^ (git log per changed file empty). Stock test file meaningful on base (tests-only fails 2, with fix passes 72).

Earlier file overlaps: `[]`.

**Why high signal:** Small, generic type-hint-to-argparse logic bug with crisp, user-visible CLI semantics and a discriminating stock CPU test.

## 3. vllm-41795 — [KV Offload] Return None from lookup() for in-flight blocks

**Set:** primary; **area:** kv_offload; **difficulty:** easy.

**Ground truth:** [f39bcf1e30693f2e620e3a2ec732c5f7192408fb](https://github.com/COG-GTM/vllm/commit/f39bcf1e30693f2e620e3a2ec732c5f7192408fb); **parent:** `6467213a9f118fcf5ebf65d5e1ee8d64d2e101e4`.

**Files:** 2; **lines:** +11/−7.

- `tests/v1/kv_offload/cpu/test_manager.py`
- `vllm/v1/kv_offload/cpu/manager.py`

### Agent task statement

The CPU KV-cache offloading manager exposes a lookup() method that the offloading connector uses to decide whether a block's data can be loaded from CPU memory. The abstract manager contract defines three outcomes: the block is present and ready, the block is absent, or the block exists but is not yet usable and the caller should retry later. The CPU manager currently collapses the last two cases: a block whose store (write to CPU) has been started but not completed is reported exactly like a block that was never offloaded. As a result, a scheduler that sees a hit-miss cannot distinguish 'not cached' from 'cached soon', so it recomputes prefixes that are about to become available instead of waiting. Repro: create a CPU offloading manager with any eviction policy (LRU or ARC), call prepare_store() for some block hashes and do not call complete_store(); lookup() of those keys returns False. Expected: lookup() returns True for stored-and-completed blocks, a distinct retry-later value (None, per the base-class contract) for blocks whose store is still in flight, and False for unknown blocks. Constraints: behavior must be identical across eviction policies, the return type must match the abstract interface, and existing prepare_store/complete_store/touch semantics must not change.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** True; **level:** `stock_pytest`.

**Ground-truth test files:** `tests/v1/kv_offload/cpu/test_manager.py`

**Recorded test IDs:**

- `tests/v1/kv_offload/cpu/test_manager.py::test_cpu_manager`
- `tests/v1/kv_offload/cpu/test_manager.py::TestARCPolicy::test_basic`

**Pytest / probe command and status:**

Executed stock pytest; normalized worktree and venv locations. Parent/base runs overlay only the ground-truth regression tests, never implementation.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest -q -p no:cacheprovider tests/v1/kv_offload/cpu/test_manager.py
```

**Observed ground truth:** 13 passed in 6.79s (stock pytest, CPU)

**Observed parent:** 2 failed, 11 passed in 3.29s with commit's test file copied into parent worktree; FAILED test_cpu_manager and TestARCPolicy::test_basic (AssertionError: lookup of in-flight block returned False, expected None)

Stock pytest, real CPUOffloadingManager exercised on CPU; clean parent/commit discrimination. Test file modifies existing assertions (no new test functions) so hidden tests are the two edited tests.

**Coordinator freeze-base rerun:**

```json
[
  {
    "pr_number": 41795,
    "mode": "fix",
    "exit_code": 0,
    "summary": [
      "13 passed in 1.86s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/v1/kv_offload/cpu/test_manager.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  },
  {
    "pr_number": 41795,
    "mode": "tests",
    "exit_code": 1,
    "summary": [
      "2 failed, 11 passed in 2.03s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/v1/kv_offload/cpu/test_manager.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  }
]
```

**Synthetic test idea:** Not required for the demonstrated stock CPU regression; retain equivalent behavioral coverage.

### Rubric

- Returns None (not False) for keys whose store is in flight, True only when ready
- Unknown keys still return False
- Return annotation matches base OffloadingManager (bool | None)
- Works for both LRU and ARC policies without policy-specific hacks
- No change to prepare_store/complete_store bookkeeping

### Freeze-base suitability

**Applies cleanly:** True. No earlier commits in 98661fe01..f39bcf1e3^ touch the changed files; git apply --check at base: CLEAN.

Earlier file overlaps: `[]`.

**Why high signal:** Small but semantically precise tri-state contract bug with a stock CPU regression that fails on parent and passes on the fix.

## 4. vllm-41730 — [BUGFIX] Support streamed_args_for_tool in MistralToolParser (#41730)

**Set:** primary; **area:** tool parsers / Mistral streaming; **difficulty:** medium.

**Ground truth:** [c6235ed1803e31bafedd90d53766e5705c570780](https://github.com/COG-GTM/vllm/commit/c6235ed1803e31bafedd90d53766e5705c570780); **parent:** `628c43630155b9fa8a8ec8871eb90c06bf8b36e6`.

**Files:** 2; **lines:** +55/−15.

- `tests/tool_parsers/test_mistral_tool_parser.py`
- `vllm/tool_parsers/mistral_tool_parser.py`

### Agent task statement

Mistral tool-call streaming can fail after the parser has already emitted a valid function name or an initial fragment of JSON arguments. Affected clients see an IndexError in the chat-completion streaming path rather than a completed tool response. Reproduce with incremental Mistral tool-call output containing one or several functions, including arguments split over multiple deltas and output ending immediately after a tool call. The final function names and argument strings must agree with the non-streaming interpretation, and each tool must have complete, independent state as streaming progresses. Ordinary assistant text, no-tool output, older and newer tokenizer formats, and malformed partial fragments must retain their existing behavior. The repair should be local to parsing and must not require loading model weights or changing the external response schema.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** True; **level:** `stock_pytest`.

**Ground-truth test files:** `tests/tool_parsers/test_mistral_tool_parser.py`

**Recorded test IDs:**

- `tests/tool_parsers/test_mistral_tool_parser.py::test_extract_tool_calls_streaming_pre_v11_tokenizer`
- `tests/tool_parsers/test_mistral_tool_parser.py::test_extract_tool_calls_streaming`
- `tests/tool_parsers/test_mistral_tool_parser.py::test_extract_tool_calls_streaming_v11_no_tools`
- `tests/tool_parsers/test_mistral_tool_parser.py::test_extract_tool_calls_streaming_one_chunk`
- `tests/tool_parsers/test_mistral_tool_parser.py::test_extract_tool_calls_streaming_exception_returns_none`

**Pytest / probe command and status:**

Executed stock pytest; normalized worktree and venv locations. Parent/base runs overlay only the ground-truth regression tests, never implementation.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest -q -p no:cacheprovider tests/tool_parsers/test_mistral_tool_parser.py -k extract_tool_calls_streaming
```

**Observed ground truth:** {"deselected": 59, "passed": 27}; exit 0

**Observed parent:** {"deselected": 59, "failed": 10, "passed": 17}; exit 1

- The commit-modified test file was copied unchanged into the parent. Focused stock pytest -k extract_tool_calls_streaming executed 27 cases: commit 27 passed; parent 10 failed and 17 passed.
- Full-file attempts were also executed: commit 14 failed/72 passed; parent 24 failed/62 passed. Both have grammar-factory failures including TypeError: GrammarFactory.get_lark_from_jinja() missing required argument force_tool_names. The focused streaming result is not a claim that the full file passes.
- The ten additional parent failures include missing streamed-argument records and accumulated argument/name state checks; no inference server or GPU was involved.
- Exact commands below include revision-specific PYTHONPATH/cwd. Parent implementation is untouched; copied upstream changed test files: ['tests/tool_parsers/test_mistral_tool_parser.py']. A shared synthetic file explicitly skips other candidates, so skipped unrelated probes are not claimed as coverage.
- Downloadable evidence archive: [audit evidence bundle; source embedded below where applicable] Contains the actual probe source, candidate/parent/base JSON logs, diffs and inspected upstream metadata. Local log paths below correspond to same-named archive members.

Additional stock attempts (not necessarily discriminating):

- ground_truth: {"failed": 14, "passed": 72}; exit 1
- parent: {"failed": 24, "passed": 62}; exit 1

**Audit base-plus-patch:** {"deselected": 59, "passed": 27}; exit 0

**Coordinator freeze-base rerun:**

```json
[
  {
    "pr_number": 41730,
    "mode": "fix",
    "exit_code": 0,
    "summary": [
      "27 passed, 59 deselected in 9.12s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/tool_parsers/test_mistral_tool_parser.py -k extract_tool_calls_streaming",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  },
  {
    "pr_number": 41730,
    "mode": "tests",
    "exit_code": 1,
    "summary": [
      "10 failed, 17 passed, 59 deselected in 7.30s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/tool_parsers/test_mistral_tool_parser.py -k extract_tool_calls_streaming",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  }
]
```

**Synthetic test idea:** Not required for the demonstrated stock CPU regression; retain equivalent behavioral coverage.

### Rubric

- Maintain one correctly populated streaming record per tool, including accumulated arguments and function names.
- Handle multi-tool output and arbitrary partial argument boundaries without leaking state between calls.
- Retain ordinary text/no-tool and pre-v11/v11+ parser behavior.
- Demonstrate differential CPU failures using the copied upstream regression tests, not only mocked expected JSON.

### Freeze-base suitability

**Applies cleanly:** True. Inspected earlier history using git log --format=%H 98661fe01..c6235ed1803e31bafedd90d53766e5705c570780^ -- <each changed file>; overlap list is []. git -C /home/ubuntu/vllm-audit/base apply --check /home/ubuntu/vllm-audit/41730.diff returned exit 0 with '' output. In an independent detached 98661fe01 worktree, applied only this patch and actually reran the relevant CPU checks: exit 0, counts {'passed': 27, 'deselected': 59}. All non-test changed files were also mechanically byte-compared between base and candidate parent and are identical: {'vllm/tool_parsers/mistral_tool_parser.py': True}. Existing base ToolParser/tokenizer protocol dependencies suffice. The changed test file and parser patch apply at base; the filtered real stock suite runs successfully there without any earlier commit.

Earlier file overlaps: `[]`.

**Why high signal:** Streamed tool bookkeeping has concrete externally reported IndexError symptoms and ten additional parent failures in actual CPU parser tests.

## 5. vllm-40726 — [Bugfix] Fix codegen for unqualified names (#40726)

**Set:** primary; **area:** compilation / FX code generation and cache; **difficulty:** hard.

**Ground truth:** [213f10bfdd43422d2936a8f5e218000050572678](https://github.com/COG-GTM/vllm/commit/213f10bfdd43422d2936a8f5e218000050572678); **parent:** `e87e09a50a41db3ddd1cc0403e8d1ebe0b0f522c`.

**Files:** 4; **lines:** +450/−32.

- `tests/compile/test_codegen.py`
- `vllm/compilation/backends.py`
- `vllm/compilation/caching.py`
- `vllm/compilation/codegen.py`

### Agent task statement

A compiled execution graph can fail on its first inference call with NameError when an operator argument is a non-primitive Python object. For example, tracing a CPU tensor conversion or allocation with an explicit device can produce generated code referring to an undefined device name. Similar failures affect distributed placement objects whose string representations are not valid in the generated execution environment. Reproduce by tracing and compiling small tensor functions that include explicit device and dtype arguments, then executing the generated callable and comparing it with eager execution. Supported graph arguments must retain their meaning through graph splitting, nested submodules and cached execution. Primitive arguments should remain valid, repeated objects should be handled consistently, and execution artifacts without these special objects must remain loadable. The reproduction should not depend on an accelerator or a distributed process group.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** True; **level:** `stock_pytest`.

**Ground-truth test files:** `tests/compile/test_codegen.py`

**Recorded test IDs:**

- `tests/compile/test_codegen.py::test_dtype_singleton_deduped`
- `tests/compile/test_codegen.py::test_distinct_dtypes_get_distinct_slots`
- `tests/compile/test_codegen.py::test_consts_ordering_deterministic`
- `tests/compile/test_codegen.py::test_primitive_args_inlined`
- `tests/compile/test_codegen.py::test_consts_shared_across_split_submods`
- `tests/compile/test_codegen.py::test_non_graphmodule_submod_uses_indexed_callable`
- `tests/compile/test_codegen.py::test_node_ref_recurses_through_containers`
- `tests/compile/test_codegen.py::test_legacy_code_without_consts`

**Pytest / probe command and status:**

Executed stock pytest; normalized worktree and venv locations. Parent/base runs overlay only the ground-truth regression tests, never implementation.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest -q -p no:cacheprovider tests/compile/test_codegen.py
```

**Observed ground truth:** {"passed": 12, "skipped": 1}; exit 0

**Observed parent:** {"failed": 11, "passed": 1, "skipped": 1}; exit 1

- The added upstream tests/compile/test_codegen.py was copied to the parent without implementation changes. Stock commit: 12 passed/1 skipped; stock parent: 11 failed/1 passed/1 skipped. Several parent failures are expected new-API mismatches rather than direct NameError reproductions.
- The separate API-adaptive real FX/execution probe confirms both parent behavioral failures are NameError: name device is not defined, while the commit executes both functions successfully.
- The PR body links <https://github.com/pytorch/torchtitan/issues/3067>. It was not fetched because the hard read-access rule permits only vllm-project/vllm APIs; the symptom is documented in the inspected vLLM PR body itself. No vLLM issue is linked.
- Exact commands below include revision-specific PYTHONPATH/cwd. Parent implementation is untouched; copied upstream changed test files: ['tests/compile/test_codegen.py']. A shared synthetic file explicitly skips other candidates, so skipped unrelated probes are not claimed as coverage.
- Downloadable evidence archive: [audit evidence bundle; source embedded below where applicable] Contains the actual probe source, candidate/parent/base JSON logs, diffs and inspected upstream metadata. Local log paths below correspond to same-named archive members.

**Audit base-plus-patch:** {"passed": 12, "skipped": 1}; exit 0

**Coordinator freeze-base rerun:**

```json
[
  {
    "pr_number": 40726,
    "mode": "fix",
    "exit_code": 0,
    "summary": [
      "12 passed, 1 skipped, 14 warnings in 4.02s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/compile/test_codegen.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  },
  {
    "pr_number": 40726,
    "mode": "tests",
    "exit_code": 1,
    "summary": [
      "11 failed, 1 passed, 1 skipped, 14 warnings in 3.03s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/compile/test_codegen.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  }
]
```

**Synthetic test idea:** Executed an API-adaptive behavioral probe using actual make_fx, split_graph, generate_execution_code and compile_execution_fn for a CPU copy and allocation. It accepts either the old two-element or new three-element codegen return shape so parent failures are semantic NameError: device is not defined rather than tuple-unpacking failures. Commit 2 passed, parent 2 failed. DTensor placement objects are a further CPU-only extension.

### Rubric

- Execute actual generated code containing device/dtype or equivalent non-primitive graph arguments and compare values with eager CPU execution.
- Handle nested containers and recursively inlined submodules without losing object identity or deterministic argument ordering.
- Preserve ordinary primitive/operator code generation and callable submodule behavior.
- Keep the code-generation, compilation and cache call sites compatible with one another.
- Retain compatibility with old execution artifacts that contain no new special constants.

### Freeze-base suitability

**Applies cleanly:** True. Inspected earlier history using git log --format=%H 98661fe01..213f10bfdd43422d2936a8f5e218000050572678^ -- <each changed file>; overlap list is []. git -C /home/ubuntu/vllm-audit/base apply --check /home/ubuntu/vllm-audit/40726.diff returned exit 0 with '' output. In an independent detached 98661fe01 worktree, applied only this patch and actually reran the relevant CPU checks: exit 0, counts {'passed': 12, 'skipped': 1}. All non-test changed files were also mechanically byte-compared between base and candidate parent and are identical: {'vllm/compilation/backends.py': True, 'vllm/compilation/caching.py': True, 'vllm/compilation/codegen.py': True}. The newly added test file is absent at base until supplied by the candidate patch. Its real split_graph and compilation dependencies import successfully after applying only this patch at base; the full stock file then gives 12 passed/1 skipped. New codegen return values and cache fields are implemented together in this patch, not supplied by an earlier commit.

Earlier file overlaps: `[]`.

**Why high signal:** Real FX tracing/code generation executes on CPU and reproduces the unqualified-device NameError; tests cover constants and cache compatibility rather than hardware execution.

## 6. vllm-41266 — [Frontend][Bugfix] Abort ASR engine requests on cancellation

**Set:** primary; **area:** API / OpenAI entrypoints (speech-to-text); **difficulty:** medium.

**Ground truth:** [a5d0a5afba521e875454a9cd920734bef73d404f](https://github.com/COG-GTM/vllm/commit/a5d0a5afba521e875454a9cd920734bef73d404f); **parent:** `f2840120f632c55a972dc2ec482241db89b6fc0f`.

**Files:** 2; **lines:** +284/−54.

- `tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py`
- `vllm/entrypoints/openai/speech_to_text/speech_to_text.py`

### Agent task statement

When a client disconnects or the server cancels a non-streaming `/v1/audio/transcriptions` or `/v1/audio/translations` request, the generation work already submitted to the engine keeps running. Long audio is split into several chunks, each submitted as its own internal engine request with its own request id, and automatic language detection may submit an additional short engine request; none of these internal requests are aborted when the outer HTTP handler task is cancelled. The result is wasted GPU time and engine requests that outlive their client.

Expected behavior: if the outer non-streaming speech-to-text coroutine is cancelled at any point (while waiting on language detection or while collecting chunk outputs), every internal engine request that was started on behalf of that call — all chunk requests and the language-detection request — is aborted via the engine client using the exact request ids that were submitted, and `asyncio.CancelledError` still propagates to the caller. All chunk result generators should be advanced/closed so none is left pending. Normal (non-cancelled) transcription/translation results and the streaming code path must be unchanged. Should be testable on CPU with a mocked engine client whose generate() blocks.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** True; **level:** `stock_pytest`.

**Ground-truth test files:** `tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py`

**Recorded test IDs:**

- `tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py::test_non_streaming_cancel_aborts_engine_requests`
- `tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py::test_non_streaming_cancel_advances_all_chunk_generators`
- `tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py::test_language_detection_cancel_aborts_engine_request`

**Pytest / probe command and status:**

Executed stock pytest; normalized worktree and venv locations. Parent/base runs overlay only the ground-truth regression tests, never implementation.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest -q -p no:cacheprovider tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py
```

**Observed ground truth:** stock pytest: 4 passed, 14 warnings in 1.58s (one test parametrized)

**Observed parent:** test file copied into parent: 4 failed — `DID NOT RAISE CancelledError` / aborted ids did not match expected chunk request ids

Stock commit test (new file), mocked engine client, CPU only. Base matrix: fix 4 passed / tests-only 4 failed. Hidden tests mock some private helpers of OpenAISpeechToText; check test compatibility with alternate valid implementations before using as primary. Logs: /home/ubuntu/audit/logs/a5d0a5afb-{gt,par}.log

**Coordinator freeze-base rerun:**

```json
[
  {
    "pr_number": 41266,
    "mode": "fix",
    "exit_code": 0,
    "summary": [
      "4 passed in 1.01s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  },
  {
    "pr_number": 41266,
    "mode": "tests",
    "exit_code": 1,
    "summary": [
      "4 failed in 0.95s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/entrypoints/openai/speech_to_text/test_speech_to_text_cancellation.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  }
]
```

**Synthetic test idea:** Not required for the demonstrated stock CPU regression; retain equivalent behavioral coverage.

### Rubric

- All chunk request ids and the language-detection request id are aborted on outer cancellation.
- CancelledError is re-raised, not swallowed or converted to an error response.
- Abort uses the same internal ids as submitted to engine_client.generate.
- No abort on successful completion; streaming path unaffected.
- Generators/tasks are cleaned up (no pending tasks/warnings).

### Freeze-base suitability

**Applies cleanly:** True. Mechanical: base-matrix script /home/ubuntu/audit/basematrix.sh created detached worktree at 98661fe01 and ran `git -C /home/ubuntu/repos/vllm diff a5d0a5afb^ a5d0a5afb | git apply` -> applied with no errors. No overlapping files in 98661fe01..a5d0a5afb^. Tests meaningful on base.

Earlier file overlaps: `[]`.

**Why high signal:** Async cancellation/resource-leak bug across multi-chunk requests requires careful asyncio reasoning and is fully CPU-testable with mocks.

## 7. vllm-41801 — [Bugfix] DeepSeekV32/v4: respect string='true|false' attribute and unwrap arguments/input wrapper

**Set:** primary; **area:** tool parsers; **difficulty:** medium.

**Ground truth:** [95582868efd4db0b120e3640bbc61dcfce20d59f](https://github.com/COG-GTM/vllm/commit/95582868efd4db0b120e3640bbc61dcfce20d59f); **parent:** `50acdc5b5cc00f10408d8f98b21fc97efc615173`.

**Files:** 3; **lines:** +224/−10.

- `tests/tool_parsers/test_deepseekv32_tool_parser.py`
- `tests/tool_parsers/test_deepseekv4_tool_parser.py`
- `vllm/tool_parsers/deepseekv32_tool_parser.py`

### Agent task statement

The DeepSeek V3.2 / V4 tool-call parser reads DSML-formatted tool calls where each parameter tag carries a `string="true"` or `string="false"` attribute. Two problems show up with real model output. First, the attribute is ignored: a parameter marked string="true" is still converted according to the tool's JSON schema, so a literal such as "123" or "true" is turned into a number/boolean even though the model explicitly said it is a string; conversely string="false" values should still be schema-converted/JSON-decoded. Second, the model sometimes wraps all arguments in a single `arguments` or `input` parameter (a JSON object or JSON string) even though the tool schema has no such parameter, producing tool calls with the wrong shape. Expected: string="true" values are preserved verbatim as strings regardless of schema; string="false" values are converted as before; when the only emitted parameter is `arguments`/`input`, that name is not a declared parameter, and its (possibly JSON-encoded) content is an object whose keys are declared parameters, the inner object becomes the arguments. Behavior must be the same for non-streaming and streaming extraction and for both V3.2 and V4 parsers; tools that genuinely declare `arguments`/`input` must not be unwrapped.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** True; **level:** `stock_pytest`.

**Ground-truth test files:** `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv4_tool_parser.py`

**Recorded test IDs:**

- `tests/tool_parsers/test_deepseekv32_tool_parser.py::TestExtractToolCalls::test_string_attr_true_preserves_literal_despite_schema`
- `tests/tool_parsers/test_deepseekv32_tool_parser.py::TestExtractToolCalls::test_string_attr_false_allows_schema_conversion`
- `tests/tool_parsers/test_deepseekv32_tool_parser.py::TestExtractToolCalls::test_arguments_wrapper_repaired`
- `tests/tool_parsers/test_deepseekv32_tool_parser.py::TestExtractToolCalls::test_input_wrapper_repaired`
- `tests/tool_parsers/test_deepseekv32_tool_parser.py::TestExtractToolCallsStreaming::test_string_attr_true_preserves_literal_in_streaming`
- `tests/tool_parsers/test_deepseekv4_tool_parser.py::test_extract_tool_calls_arguments_wrapper`

**Pytest / probe command and status:**

Executed stock pytest; normalized worktree and venv locations. Parent/base runs overlay only the ground-truth regression tests, never implementation.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest -q -p no:cacheprovider tests/tool_parsers/test_deepseekv32_tool_parser.py tests/tool_parsers/test_deepseekv4_tool_parser.py
```

**Observed ground truth:** 57 passed in 14.55s (stock pytest, CPU)

**Observed parent:** 5 failed, 52 passed in 9.74s with commit's tests copied into parent; FAILED: test_string_attr_true_preserves_literal_despite_schema, test_arguments_wrapper_repaired, test_input_wrapper_repaired, TestExtractToolCallsStreaming::test_string_attr_true_preserves_literal_in_streaming, deepseekv4::test_extract_tool_calls_arguments_wrapper

Stock pytest, real parser on CPU; strongly discriminating. test_string_attr_false_allows_schema_conversion passes on both (guard test).

**Coordinator freeze-base rerun:**

```json
[
  {
    "pr_number": 41801,
    "mode": "fix",
    "exit_code": 0,
    "summary": [
      "57 passed in 10.88s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/tool_parsers/test_deepseekv32_tool_parser.py tests/tool_parsers/test_deepseekv4_tool_parser.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  },
  {
    "pr_number": 41801,
    "mode": "tests",
    "exit_code": 1,
    "summary": [
      "5 failed, 52 passed in 9.13s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/tool_parsers/test_deepseekv32_tool_parser.py tests/tool_parsers/test_deepseekv4_tool_parser.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  }
]
```

**Synthetic test idea:** Not required for the demonstrated stock CPU regression; retain equivalent behavioral coverage.

### Rubric

- string="true" values never schema-coerced
- string="false" values still schema/JSON converted
- Wrapper unwrap only when sole param, undeclared, and inner keys subset of schema
- JSON-string wrapper content is decoded; invalid JSON left untouched
- Streaming and non-streaming agree; V4 parser inherits fix

### Freeze-base suitability

**Applies cleanly:** True. No earlier commits in range touch the changed files; git apply --check at base: CLEAN.

Earlier file overlaps: `[]`.

**Why high signal:** Two concrete parsing-correctness behaviors with precise CPU regression tests that fail 5/6 on parent.

## 8. vllm-40749 — [Bugfix] Skip PP sampled-token receive on last rank during async scheduling (#40749)

**Set:** primary; **area:** core execution / pipeline-parallel async scheduling; **difficulty:** easy.

**Ground truth:** [b53c507bc91f87e28b03e9b54bbff7c76e97d58b](https://github.com/COG-GTM/vllm/commit/b53c507bc91f87e28b03e9b54bbff7c76e97d58b); **parent:** `2d7d6cf765d0c796070feaa34fcfcc26494d5c31`.

**Files:** 2; **lines:** +54/−1.

- `tests/v1/worker/test_gpu_model_runner.py`
- `vllm/v1/worker/gpu_model_runner.py`

### Agent task statement

Asynchronous pipeline-parallel generation can crash when a pipeline stage has no execution state for the current scheduling step. In a two-stage deployment the final stage already owns the sampled token IDs, but the empty-step path may still attempt work intended for another stage and trigger an internal pipeline assertion. Reproduce the scheduling decision with asynchronous operation enabled, an empty current execution state, and each pipeline rank in turn. The final stage must complete this path safely, while earlier stages still obtain the previous token decisions needed to continue generation. A single-stage deployment must remain valid. With asynchronous scheduling disabled, this otherwise empty path should remain independent of pipeline-group initialization. Preserve the existing return value and behavior of unrelated execution paths without requiring a real distributed cluster for the regression test.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** True; **level:** `stock_pytest`.

**Ground-truth test files:** `tests/v1/worker/test_gpu_model_runner.py`

**Recorded test IDs:**

- `tests/v1/worker/test_gpu_model_runner.py::test_sample_tokens_receives_pp_sampled_ids_only_on_non_last_rank`
- `tests/v1/worker/test_gpu_model_runner.py::test_sample_tokens_skips_pp_group_lookup_without_async_scheduling`

**Pytest / probe command and status:**

Executed stock pytest; normalized worktree and venv locations. Parent/base runs overlay only the ground-truth regression tests, never implementation.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest -q -p no:cacheprovider tests/v1/worker/test_gpu_model_runner.py::test_sample_tokens_receives_pp_sampled_ids_only_on_non_last_rank tests/v1/worker/test_gpu_model_runner.py::test_sample_tokens_skips_pp_group_lookup_without_async_scheduling
```

**Observed ground truth:** {"passed": 4}; exit 0

**Observed parent:** {"failed": 1, "passed": 3}; exit 1

- Copied the commit-modified upstream worker test file into the parent. The exact two selected test functions expand to four CPU cases: commit 4 passed; parent 1 failed/3 passed. The failing parent case is world_size=2, is_last_rank=True.
- These are real stock tests invoking GPUModelRunner.sample_tokens on a **new**-constructed runner with a stub pipeline group; no CUDA state is initialized. The test file name does not imply a GPU was required.
- An additional AST-isolated method probe gave the same 4-pass versus 1-fail/3-pass result, but the primary evidence is stock pytest. No linked issue was present in the PR body.
- Exact commands below include revision-specific PYTHONPATH/cwd. Parent implementation is untouched; copied upstream changed test files: ['tests/v1/worker/test_gpu_model_runner.py']. A shared synthetic file explicitly skips other candidates, so skipped unrelated probes are not claimed as coverage.
- Downloadable evidence archive: [audit evidence bundle; source embedded below where applicable] Contains the actual probe source, candidate/parent/base JSON logs, diffs and inspected upstream metadata. Local log paths below correspond to same-named archive members.

**Audit base-plus-patch:** {"passed": 4}; exit 0

**Coordinator freeze-base rerun:**

```json
[
  {
    "pr_number": 40749,
    "mode": "fix",
    "exit_code": 0,
    "summary": [
      "4 passed, 14 warnings in 1.43s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/v1/worker/test_gpu_model_runner.py::test_sample_tokens_receives_pp_sampled_ids_only_on_non_last_rank tests/v1/worker/test_gpu_model_runner.py::test_sample_tokens_skips_pp_group_lookup_without_async_scheduling",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  },
  {
    "pr_number": 40749,
    "mode": "tests",
    "exit_code": 1,
    "summary": [
      "1 failed, 3 passed, 14 warnings in 1.46s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/v1/worker/test_gpu_model_runner.py::test_sample_tokens_receives_pp_sampled_ids_only_on_non_last_rank tests/v1/worker/test_gpu_model_runner.py::test_sample_tokens_skips_pp_group_lookup_without_async_scheduling",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  }
]
```

**Synthetic test idea:** Not required for the demonstrated stock CPU regression; retain equivalent behavioral coverage.

### Rubric

- Cover the final rank, non-final rank and single-rank asynchronous cases.
- Preserve the necessary token-receive action for non-final ranks only.
- Avoid any pipeline-group lookup in the disabled-async control case.
- Exercise the real empty-state method on CPU and keep its return/side-effect contract unchanged.

### Freeze-base suitability

**Applies cleanly:** True. Inspected earlier history using git log --format=%H 98661fe01..b53c507bc91f87e28b03e9b54bbff7c76e97d58b^ -- <each changed file>; overlap list is []. git -C /home/ubuntu/vllm-audit/base apply --check /home/ubuntu/vllm-audit/40749.diff returned exit 0 with '' output. In an independent detached 98661fe01 worktree, applied only this patch and actually reran the relevant CPU checks: exit 0, counts {'passed': 4}. All non-test changed files were also mechanically byte-compared between base and candidate parent and are identical: {'vllm/v1/worker/gpu_model_runner.py': True}. GPUModelRunner and the relevant pipeline APIs already exist at base; SimpleNamespace is already available to the test file. Applying only this candidate at base permits all four original added tests to run on CPU.

Earlier file overlaps: `[]`.

**Why high signal:** Small but high-signal distributed control-flow bug with four genuine CPU tests; the last-rank case fails on the parent while non-last and non-async behavior remain covered.

## 9. vllm-41181 — [Bugfix] Fix `RuntimeError: Already borrowed` by adding thread-safe Hugging Face fast-tokenizer wrappers (#41181)

**Set:** primary; **area:** tokenizers / concurrent rendering; **difficulty:** hard.

**Ground truth:** [20dcd984f9a49b8dc69c400486eed50953cb16cf](https://github.com/COG-GTM/vllm/commit/20dcd984f9a49b8dc69c400486eed50953cb16cf); **parent:** `6fca51815706fbb0735849b95c0dafa05da20f9f`.

**Files:** 5; **lines:** +107/−35.

- `vllm/multimodal/processing/context.py`
- `vllm/renderers/base.py`
- `vllm/renderers/hf.py`
- `vllm/tokenizers/__init__.py`
- `vllm/tokenizers/hf.py`

### Agent task statement

Concurrent requests sharing a Hugging Face fast tokenizer can fail with RuntimeError: Already borrowed, or interfere with one another when tokenization options differ. This can occur when rendering requests from multiple workers or processing sampling options that tokenize text. Reproduce the problem by sharing one fast tokenizer across several threads and repeatedly encoding batches with different padding and truncation lengths; compare every result with the corresponding sequential call. Concurrent encode, decode and chat-template operations should remain usable and return the same types and values as ordinary single-threaded tokenization. Existing slow-tokenizer behavior and tokenizer serialization must remain compatible. The solution must not require model weights, an accelerator, or callers to create a separate tokenizer for every request. Propagating tokenizer mutations during active requests is not a new guarantee.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** False; **level:** `real_module_cpu_probe_with_mocks`.

**Ground-truth test files:** None in the commit; synthetic evidence only.

**Recorded test IDs:**

- No CPU-discriminating stock IDs established; see the synthetic probe and next-stage test idea.

Audit probe IDs (not upstream test names):

- `audit-probes/test_audit_cpu.py::test_tokenizer_pool_real_encode_concurrency`
- `audit-probes/test_audit_cpu.py::test_real_fast_tokenizer_concurrent_padding`

**Pytest / probe command and status:**

Executed audit probe; not a stock test-suite pass. Build a behavior-oriented hidden pytest from synthetic_test_idea before scoring agent PRs.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu AUDIT_TREE="$PWD" AUDIT_PR=41181 TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest audit-probes/test_audit_cpu.py --confcutdir=audit-probes -q
```

**Observed ground truth:** {"passed": 2, "skipped": 23}; exit 0

**Observed parent:** {"failed": 2, "skipped": 23}; exit 1

- No regression tests are added by this commit. The upstream suggested stock file tests/models/multimodal/processing/test_common.py was attempted at both revisions with a 90-second limit; both attempts timed out, so no stock-file pass is claimed.
- Synthetic tests import the real revision-specific module and run real Rust tokenization on CPU. The deterministic guard models contention; the second test uses an unmodified real fast tokenizer and found a padding/truncation race in the parent.
- cpu_verified refers to successful real tokenizer/wrapper behavior on CPU, not a full multi-process HTTP server or all renderer integration paths.
- Exact commands below include revision-specific PYTHONPATH/cwd. Parent implementation is untouched; copied upstream changed test files: []. A shared synthetic file explicitly skips other candidates, so skipped unrelated probes are not claimed as coverage.
- Downloadable evidence archive: [audit evidence bundle; source embedded below where applicable] Contains the actual probe source, candidate/parent/base JSON logs, diffs and inspected upstream metadata. Local log paths below correspond to same-named archive members.

Additional stock attempts (not necessarily discriminating):

- ground_truth: {}; exit 124
- parent: {}; exit 124

**Audit base-plus-patch:** {"passed": 2, "skipped": 23}; exit 0

**Synthetic test idea:** Executed: build a local WordLevel Rust tokenizer, share a PreTrainedTokenizerFast across eight threads, and vary max_length between 32 and 40 for 32 batches of 256 inputs. Check every sequence length and token prefix. A separate subclass adds a deterministic per-instance contention guard while still executing real tokenization. The real-backend parent failure was incorrect output length; the guard parent failure was an injected Already borrowed error. Extend this with concurrent decode/chat-template calls and pickle round trips.

### Rubric

- Preserve per-request encoding correctness when concurrent calls use different padding/truncation options.
- Avoid overlapping unsafe backend access across the supported public tokenizer methods.
- Maintain ordinary return types, slow-tokenizer compatibility and serialization behavior.
- Exercise contention and repeated reuse without downloading a model; do not treat the deterministic guard as a reproduction of a Rust exception.

### Freeze-base suitability

**Applies cleanly:** True. Inspected earlier history using git log --format=%H 98661fe01..20dcd984f9a49b8dc69c400486eed50953cb16cf^ -- <each changed file>; overlap list is []. git -C /home/ubuntu/vllm-audit/base apply --check /home/ubuntu/vllm-audit/41181.diff returned exit 0 with '' output. In an independent detached 98661fe01 worktree, applied only this patch and actually reran the relevant CPU checks: exit 0, counts {'passed': 2, 'skipped': 23}. All non-test changed files were also mechanically byte-compared between base and candidate parent and are identical: {'vllm/multimodal/processing/context.py': True, 'vllm/renderers/base.py': True, 'vllm/renderers/hf.py': True, 'vllm/tokenizers/**init**.py': True, 'vllm/tokenizers/hf.py': True}. The new thread-safe mixin/helper is supplied by the patch itself; standard-library queue/contextlib and installed Transformers/tokenizers suffice for the probes. Full renderer and multimodal integration is not established by the isolated tokenizer run.

Earlier file overlaps: `[]`.

**Why high signal:** Thread-safe fast-tokenizer borrowing is a substantive concurrency task; a local real Rust-tokenizer workload exposes parent padding-state races without downloading a model.

## 10. vllm-42162 — Fix Molmo2 image token metadata

**Set:** primary; **area:** multimodal processing; **difficulty:** medium.

**Ground truth:** [879a8c318032ed32716e5e0b10af355a4ddbced2](https://github.com/COG-GTM/vllm/commit/879a8c318032ed32716e5e0b10af355a4ddbced2); **parent:** `1b57eb41f24166d0bee2031c53847b188089d77d`.

**Files:** 2; **lines:** +94/−10.

- `tests/models/multimodal/processing/test_molmo2.py`
- `vllm/model_executor/models/molmo2.py`

### Agent task statement

Serving MolmoWeb (a Molmo2-architecture checkpoint, e.g. `allenai/MolmoWeb-4B`) with image inputs crashes with a CUDA device-side assert deep in attention. With launch blocking enabled, the first failing invariant is the multimodal embedding merge: the prompt contains more image placeholder tokens than vLLM produced image embeddings (reported example: 550 placeholder tokens vs 536 embeddings, a difference equal to the number of rows in the low-resolution/global crop).

The per-image token metadata vLLM computes (which positions are image patches vs. structural tokens such as column separators and start tokens, and the total length per image) does not follow the Hugging Face Molmo2 processor semantics when the processor configuration controls column tokens for the full-image grid and for the single (low-res) crop independently, and whether a start token is emitted for the single crop.

Expected: for any combination of these processor settings, the flattened per-image length and patch mask computed by vLLM must exactly match the token sequence the HF processor emits, so the number of embeddings equals the number of placeholder tokens. Existing defaults (column tokens on) must keep producing the same result as before for the original Molmo2 checkpoints. The check must be possible on CPU with a tiny synthetic config and image grid, without loading weights.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** True; **level:** `stock_pytest`.

**Ground-truth test files:** `tests/models/multimodal/processing/test_molmo2.py`

**Recorded test IDs:**

- `tests/models/multimodal/processing/test_molmo2.py::test_build_flat_image_bool_length_matches_molmoweb_processor_tokens`
- `tests/models/multimodal/processing/test_molmo2.py::test_build_flat_image_bool_length_respects_disabled_col_tokens`

**Pytest / probe command and status:**

Executed stock pytest; normalized worktree and venv locations. Parent/base runs overlay only the ground-truth regression tests, never implementation.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest -q -p no:cacheprovider tests/models/multimodal/processing/test_molmo2.py
```

**Observed ground truth:** stock pytest: 2 passed, 14 warnings in 1.36s

**Observed parent:** test file copied into parent: 2 failed — `TypeError: build_flat_image_bool_length() got an unexpected keyword argument 'image_use_col_tokens'`

Stock commit test (new file), CPU only, no weights. Base matrix: fix 2 passed / tests-only 2 failed. Caveat: hidden tests call `build_flat_image_bool_length(image_grids, hf_config, image_use_col_tokens=..., use_single_crop_col_tokens=..., use_single_crop_start_token=...)` — the task statement intentionally does not name this; grader should either expose that signature as required interface or add an end-to-end processor-count test. Logs: /home/ubuntu/audit/logs/879a8c318-{gt,par}.log, base-879a8c318-*.log

**Coordinator freeze-base rerun:**

```json
[
  {
    "pr_number": 42162,
    "mode": "fix",
    "exit_code": 0,
    "summary": [
      "2 passed, 14 warnings in 1.43s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/models/multimodal/processing/test_molmo2.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  },
  {
    "pr_number": 42162,
    "mode": "tests",
    "exit_code": 1,
    "summary": [
      "2 failed, 14 warnings in 1.45s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/models/multimodal/processing/test_molmo2.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  }
]
```

**Synthetic test idea:** Interface-agnostic variant: build a tiny Molmo2 processor/config with image_use_col_tokens=False & use_single_crop_col_tokens=True, run vLLM's prompt-replacement path and assert placeholder count == sum of per-image feature lengths.

### Rubric

- Per-image token length matches HF Molmo2 processor for MolmoWeb settings (550 vs old 536 example).
- Global-crop and single-crop column-token flags handled independently; single-crop start token configurable.
- Patch mask (bool) positions remain consistent with lengths.
- Defaults preserve original Molmo2 behavior.
- Settings read from processor/HF config and plumbed to the metadata computation rather than hard-coded per model.

### Freeze-base suitability

**Applies cleanly:** True. Mechanical: base-matrix script /home/ubuntu/audit/basematrix.sh created detached worktree at 98661fe01 and ran `git -C /home/ubuntu/repos/vllm diff 879a8c318^ 879a8c318 | git apply` -> applied with no errors. No overlapping files in 98661fe01..879a8c318^. Tests meaningful on base.

Earlier file overlaps: `[]`.

**Why high signal:** Real crash traced to a subtle token-count invariant in multimodal processing, verifiable on CPU with synthetic grids.

## 11. vllm-41355 — [Frontend] Supports resubmitting output items with missing fields in Responses API (#41355)

**Set:** primary; **area:** API / Responses request validation; **difficulty:** medium.

**Ground truth:** [c7aa186d67b6f051680831418e957c67f34ba7a2](https://github.com/COG-GTM/vllm/commit/c7aa186d67b6f051680831418e957c67f34ba7a2); **parent:** `f6537612521df1156d8ac13f524e427aca908322`.

**Files:** 1; **lines:** +60/−10.

- `vllm/entrypoints/openai/responses/protocol.py`

### Agent task statement

Clients building a follow-up Responses request from a previous model response can receive validation errors for output items that omit server-generated metadata. In particular, assistant output messages and reasoning items may contain their meaningful content but lack identifiers, completion status, or output-text annotations. Reproduce by constructing a request with an assistant message containing output_text and with a reasoning item containing a summary, omitting these optional round-trip fields. Such items should be accepted and represented as valid response-input items, while explicitly supplied identifiers and content remain intact. Existing function-call input handling and ordinary text requests must continue to work. Invalid substantive content should still be validated normally; this change should not indiscriminately accept arbitrary dictionaries or require the client to manufacture internal output metadata.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** False; **level:** `real_module_cpu_probe_with_mocks`.

**Ground-truth test files:** `tests/entrypoints/openai/responses/test_protocol.py`

**Recorded test IDs:**

- No CPU-discriminating stock IDs established; see the synthetic probe and next-stage test idea.

Audit probe IDs (not upstream test names):

- `audit-probes/test_audit_cpu.py::test_responses_output_item_resubmission`

**Pytest / probe command and status:**

Executed audit probe; not a stock test-suite pass. Build a behavior-oriented hidden pytest from synthetic_test_idea before scoring agent PRs.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu AUDIT_TREE="$PWD" AUDIT_PR=41355 TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest audit-probes/test_audit_cpu.py --confcutdir=audit-probes -q
```

**Observed ground truth:** {"passed": 4, "skipped": 21}; exit 0

**Observed parent:** {"failed": 3, "passed": 1, "skipped": 21}; exit 1

- No tests are changed by this commit. Existing stock test_protocol.py runs 2 passed on both commit and parent and does not discriminate this regression.
- The CPU probe imports the complete real ResponsesRequest and exercises its actual Pydantic validators. Commit 4 passed; parent 3 failed and 1 passed with missing-field ValidationError. The passing parent case is the existing function-call compatibility control.
- No linked issue was present in the inspected upstream PR body.
- Exact commands below include revision-specific PYTHONPATH/cwd. Parent implementation is untouched; copied upstream changed test files: []. A shared synthetic file explicitly skips other candidates, so skipped unrelated probes are not claimed as coverage.
- Downloadable evidence archive: [audit evidence bundle; source embedded below where applicable] Contains the actual probe source, candidate/parent/base JSON logs, diffs and inspected upstream metadata. Local log paths below correspond to same-named archive members.

Additional stock attempts (not necessarily discriminating):

- ground_truth: {"passed": 2}; exit 0
- parent: {"passed": 2}; exit 0

**Audit base-plus-patch:** {"passed": 4, "skipped": 21}; exit 0

**Synthetic test idea:** Executed four real ResponsesRequest constructions: minimal assistant output; assistant output with an existing id/status but missing annotations; reasoning with summary and no id; function-call input as a compatibility control. Assert typed metadata, text annotations and call fields. Add malformed-content and explicitly provided metadata cases for broader coverage.

### Rubric

- Accept minimal assistant output messages and reasoning items as follow-up inputs.
- Fill round-trip metadata consistently without overwriting explicit identifiers or text.
- Preserve the pre-existing function-call and normal text request behavior.
- Keep Pydantic validation for genuinely malformed content; verify the resulting typed objects, not just absence of an exception.

### Freeze-base suitability

**Applies cleanly:** True. Inspected earlier history using git log --format=%H 98661fe01..c7aa186d67b6f051680831418e957c67f34ba7a2^ -- <each changed file>; overlap list is []. git -C /home/ubuntu/vllm-audit/base apply --check /home/ubuntu/vllm-audit/41355.diff returned exit 0 with '' output. In an independent detached 98661fe01 worktree, applied only this patch and actually reran the relevant CPU checks: exit 0, counts {'passed': 4, 'skipped': 21}. All non-test changed files were also mechanically byte-compared between base and candidate parent and are identical: {'vllm/entrypoints/openai/responses/protocol.py': True}. The request model and all imported OpenAI/Pydantic types exist at base. The candidate introduces its own item-normalization validator; no missing earlier-commit symbol was needed by the base-patched CPU probes.

Earlier file overlaps: `[]`.

**Why high signal:** Request-model normalization is pure CPU API behavior with a direct Pydantic resubmission repro; the existing two tests alone do not discriminate, so dedicated probes are needed.

## 12. vllm-41269 — [Bugfix][KV Transfer][NIXL] Notify P node on pre-admission rejection to free stranded KV blocks

**Set:** primary; **area:** KV transfer / engine core + API; **difficulty:** hard.

**Ground truth:** [3f5bd482f5c1a5dbdffbbf68d624e20bb7032013](https://github.com/COG-GTM/vllm/commit/3f5bd482f5c1a5dbdffbbf68d624e20bb7032013); **parent:** `fb1ac806c55a6dc96fe92261b80c8550e9c39d2f`.

**Files:** 11; **lines:** +156/−2.

- `tests/v1/kv_connector/unit/test_nixl_connector.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py`
- `vllm/engine/protocol.py`
- `vllm/entrypoints/openai/chat_completion/serving.py`
- `vllm/entrypoints/openai/completion/serving.py`
- `vllm/entrypoints/openai/engine/serving.py`
- `vllm/entrypoints/openai/responses/serving.py`
- `vllm/v1/engine/__init__.py`
- `vllm/v1/engine/async_llm.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/request.py`

### Agent task statement

In disaggregated prefill/decode with the NIXL connector, a request arrives at the decode (D) node with `kv_transfer_params` that say `do_remote_prefill=True`; the prefill (P) node has already computed KV and is holding those blocks until D pulls them. If the D node's OpenAI frontend rejects the request before it ever reaches the engine scheduler (validation error, bad model name, length check, exception while rendering, etc.), nothing ever tells P that the transfer will not happen, so P's blocks stay pinned until `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` (default 480 s) expires — under load this exhausts P's KV cache.

Expected behavior: whenever a chat completions, completions, or responses request that carries remote-prefill KV transfer params is rejected pre-admission (an error response is returned or an exception escapes), the frontend must notify the engine (via a new async engine-client method `notify_kv_transfer_request_rejected(request_id, kv_transfer_params, *, data_parallel_rank=None)`) so the engine can immediately abort it and the connector frees/notifies the remote P blocks. Requests without a KV connector, without remote prefill, or that are admitted successfully must not trigger the notification, and original exceptions must still propagate.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** False; **level:** `real_module_cpu_probe_with_mocks`.

**Ground-truth test files:** `tests/v1/kv_connector/unit/test_nixl_connector.py`

**Recorded test IDs:**

- `tests/v1/kv_connector/unit/test_nixl_connector.py::test_abort_immediately_remote_prefill_enqueues_empty_recv`

Audit probe IDs (not upstream test names):

- `audit-probes/test_synth_kv_reject_cleanup.py::test_error_response_triggers_notify`
- `audit-probes/test_synth_kv_reject_cleanup.py::test_exception_triggers_notify_and_reraises`
- `audit-probes/test_synth_kv_reject_cleanup.py::test_success_does_not_notify`
- `audit-probes/test_synth_kv_reject_cleanup.py::test_no_notify_without_remote_prefill`
- `audit-probes/test_synth_kv_reject_cleanup.py::test_engine_core_request_abort_immediately_roundtrip`
- `audit-probes/test_synth_kv_reject_cleanup.py::test_completion_endpoint_rejection_notifies_engine`

**Pytest / probe command and status:**

Executed audit probe; not a stock test-suite pass. Build a behavior-oriented hidden pytest from synthetic_test_idea before scoring agent PRs.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest audit-probes/test_synth_kv_reject_cleanup.py --confcutdir=audit-probes -q
```

**Observed ground truth:** SYNTHETIC: 8 passed, 14 warnings in 1.32s. STOCK targeted test_abort_immediately_remote_prefill_enqueues_empty_recv: 1 passed in 35.77s.

**Observed parent:** SYNTHETIC copied into parent: 8 failed (AttributeError: no `_with_kv_transfer_rejection_cleanup` / missing notify call / missing abort_immediately). STOCK targeted test: 1 passed in 37.89s at parent — NOT discriminating.

Synthetic file at /home/ubuntu/audit/synth/test_synth_kv_reject_cleanup.py exercises real OpenAIServing / OpenAIServingCompletion.create_completion with mocked engine_client plus real EngineCoreRequest msgspec roundtrip. 4 of its tests call the private wrapper `_with_kv_transfer_rejection_cleanup` (implementation-coupled; use only test_completion_endpoint_rejection_notifies_engine and engine-level tests as hidden tests, or name the wrapper in the spec). Stock NIXL unit test does not discriminate (parent already enqueues empty recv on abort). Required `ray` install for nixl test collection. Full stock test_nixl_connector.py at base 98661fe01 is NOT usable: with patch 32 failed/31 passed, tests-only 40 failed/23 passed due to pre-existing base/test drift (`KeyError: 'remote_engine'`, `NixlAgentMetadata.__init__() got an unexpected keyword argument 'physical_blocks_per_logical_kv_block'`, `Cannot access accelerator device`). Synthetic on base: fix 8/8 pass, tests-only 8 failed. Logs: /home/ubuntu/audit/logs/3f5bd482f-*.log, base-3f5bd482f-*.log

**Synthetic test idea:** Already implemented (see evidence_notes); a stronger hidden test would drive chat/responses endpoints too with a mocked_check_model error and assert notify awaited once with the request id and DP rank.

### Rubric

- Rejected remote-prefill requests trigger cleanup before admission for returned errors and raised exceptions on chat, completions and Responses.
- Preserve client-facing errors, exceptions, original request identity and data-parallel routing.
- Successful requests, absent transfer metadata and clients without KV-transfer cleanup support remain unchanged.
- Remote cleanup releases stranded prefiller blocks without initiating normal generation; keep the wire representation compatible.
- Use endpoint/engine observations for grading, not the oracle private wrapper name; four audit wrapper tests are implementation-coupled.
- CPU mocks establish notification and serialization only; no claim of NIXL network or NIC-memory behavior.

### Freeze-base suitability

**Applies cleanly:** True. Mechanical: base-matrix script /home/ubuntu/audit/basematrix.sh created detached worktree at 98661fe01 and ran `git -C /home/ubuntu/repos/vllm diff 3f5bd482f^ 3f5bd482f | git apply` -> applied with no errors. 11 overlapping (file,commit) pairs in 98661fe01..3f5bd482f^ (e.g. df8e63f4e test_nixl_connector.py, e3b65a5ba069b350120ca7a614a010787d2de867 engine/protocol.py, 8189a1591/8eb401134/d5b31c954/6467213a9 chat/completion serving.py) yet git apply succeeded. Stock NIXL test file is broken on base independent of the fix (see evidence); synthetic tests remain meaningful.

Earlier file overlaps: `[{"file": "tests/v1/kv_connector/unit/test_nixl_connector.py", "earlier_commits": ["df8e63f4edabc06e51c10fe479fc9b95c303542f"]}, {"file": "vllm/engine/protocol.py", "earlier_commits": ["e3b65a5ba069b350120ca7a614a010787d2de867"]}, {"file": "vllm/entrypoints/openai/chat_completion/serving.py", "earlier_commits": ["8189a15914ca48461acf106f126c58ef7e41c9ee", "8eb401134e750781a202c0b6dc4059616cdb4954", "d5b31c954d511ecdb486e84c71d52d2a5c28ade4", "6467213a9f118fcf5ebf65d5e1ee8d64d2e101e4"]}, {"file": "vllm/entrypoints/openai/completion/serving.py", "earlier_commits": ["8189a15914ca48461acf106f126c58ef7e41c9ee", "d5b31c954d511ecdb486e84c71d52d2a5c28ade4"]}, {"file": "vllm/entrypoints/openai/engine/serving.py", "earlier_commits": ["6467213a9f118fcf5ebf65d5e1ee8d64d2e101e4"]}, {"file": "vllm/entrypoints/openai/responses/serving.py", "earlier_commits": ["d5b31c954d511ecdb486e84c71d52d2a5c28ade4"]}, {"file": "vllm/v1/engine/async_llm.py", "earlier_commits": ["e3b65a5ba069b350120ca7a614a010787d2de867"]}]`.

**Why high signal:** Cross-layer distributed resource-leak fix spanning frontend, engine protocol, engine core and connector, with behavior checkable on CPU via mocks.

## 13. vllm-41681 — [Perf] Use numpy zero-copy path for embedding float response serialization

**Set:** primary; **area:** API / pooling entrypoints; **difficulty:** medium.

**Ground truth:** [cd58e30872c25319e43cc530d124539adfee0c2b](https://github.com/COG-GTM/vllm/commit/cd58e30872c25319e43cc530d124539adfee0c2b); **parent:** `1d694e78c954027a42c909cd3488274e4c5ccd9c`.

**Files:** 3; **lines:** +105/−0.

- `tests/entrypoints/pooling/test_utils.py`
- `vllm/entrypoints/pooling/embed/serving.py`
- `vllm/entrypoints/pooling/utils.py`

### Agent task statement

Serving `/v1/embeddings` with `encoding_format="float"` is slow for large batches and high-dimensional embeddings: when the ORJSON response class is active, each embedding tensor is converted to a Python list element by element before serialization, which dominates response-building time.

Add a fast path for the float encoding format: when ORJSON is used, pooled float32 CPU outputs should be handed to the serializer as a zero-copy NumPy view (ORJSON can serialize NumPy arrays natively) instead of a Python list. The conversion must fall back to the existing list-based path whenever a NumPy view is not possible (e.g. bfloat16/float8 dtypes or non-CPU tensors). The JSON produced must be numerically identical to the current output. Base64/bytes encodings, Cohere-style endpoints, non-ORJSON responses, and model execution are out of scope and must be unchanged. Expose the conversion as a reusable helper in the pooling utilities, alongside the existing pooling-output encoders, named `encode_pooling_output_float_or_ndarray`, that takes a pooling output and returns either an ndarray or a list of floats.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** True; **level:** `stock_pytest`.

**Ground-truth test files:** `tests/entrypoints/pooling/test_utils.py`

**Recorded test IDs:**

- `tests/entrypoints/pooling/test_utils.py::test_encode_pooling_output_float_or_ndarray_returns_numpy_array`
- `tests/entrypoints/pooling/test_utils.py::test_orjson_serializes_numpy_array`
- `tests/entrypoints/pooling/test_utils.py::test_encode_pooling_output_float_or_ndarray_falls_back_to_list`

**Pytest / probe command and status:**

Executed stock pytest; normalized worktree and venv locations. Parent/base runs overlay only the ground-truth regression tests, never implementation.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest -q -p no:cacheprovider tests/entrypoints/pooling/test_utils.py
```

**Observed ground truth:** STOCK pytest: 3 passed, 15 warnings (after installing orjson; before orjson: 2 passed, 1 skipped)

**Observed parent:** STOCK pytest with commit's test file on parent: collection ERROR (1 error) - ImportError: cannot import name 'encode_pooling_output_float_or_ndarray' from 'vllm.entrypoints.pooling.utils'

Parent fails at collection (interface absent), not behavior; hidden tests depend on helper name, so the task statement intentionally names the helper interface (but not file/implementation). Serving fast path itself not exercised end-to-end. Logs: /home/ubuntu/wt/logs/tests/embed2_{parent,commit}.log

**Coordinator freeze-base rerun:**

```json
[
  {
    "pr_number": 41681,
    "mode": "fix",
    "exit_code": 0,
    "summary": [
      "3 passed, 1 warning in 0.60s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/entrypoints/pooling/test_utils.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  },
  {
    "pr_number": 41681,
    "mode": "tests",
    "exit_code": 2,
    "summary": [
      "ERROR tests/entrypoints/pooling/test_utils.py",
      "!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!",
      "1 error in 0.09s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/entrypoints/pooling/test_utils.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  }
]
```

**Synthetic test idea:** Not required for the demonstrated stock CPU regression; retain equivalent behavioral coverage.

### Rubric

- Preserve float embedding response schema and numerical values while reducing per-element Python serialization work.
- Use a zero-copy CPU float32 serialization representation when compatible; verify shared storage rather than timing alone.
- Retain a safe value-equivalent fallback for unsupported tensor dtypes; do not require accelerator execution to test it.
- Apply the optimization only to the optional ORJSON float path; ordinary JSON, other formats and endpoints remain compatible.
- Do not require a particular private helper name in the grader; new-helper stock tests need adaptation for equivalent solutions.

### Freeze-base suitability

**Applies cleanly:** True. No overlap with 98661fe01..cd58e3087^. At base: git apply --check OK; cherry-pick --no-commit OK, 0 conflicts. orjson needed for one test (skips otherwise).

Earlier file overlaps: `[]`.

**Why high signal:** Self-contained API perf feature with CPU-runnable unit tests covering both fast path and fallback; the parent signal is weaker because it fails at import, so the statement must expose the helper name.

## 14. vllm-41799 — [MM][Gemma4] Respect max_soft_tokens in encoder budget (#41799)

**Set:** primary; **area:** multimodal processing / encoder budget; **difficulty:** medium.

**Ground truth:** [242afc6bf40d6d088b7b97eda1dd7f00ddcdfa21](https://github.com/COG-GTM/vllm/commit/242afc6bf40d6d088b7b97eda1dd7f00ddcdfa21); **parent:** `5d0fd87038b123a11dd9c85a05ce2d258e27ce7b`.

**Files:** 2; **lines:** +91/−11.

- `tests/models/multimodal/processing/test_gemma4.py`
- `vllm/model_executor/models/gemma4_mm.py`

### Agent task statement

Gemma 4 image processing supports explicit max_soft_tokens budgets of 70, 140, 280, 560 and 1120, but the encoder planning estimate remains at the default 280 even when a different budget is configured. Small budgets therefore reserve excessive memory, while larger budgets can understate encoder requirements. Reproduce by configuring an explicit image budget and comparing the reported maximum image-token count with the configured runtime limit. Include both the ordinary processor-kwargs form and the nested image-kwargs form, as well as conflicting settings to check precedence. Planning and prompt construction should agree on the effective numeric budget. Missing or malformed nested kwargs must not crash budget lookup, and the existing default, audio behavior and fixed video-frame accounting should remain intact. Automatic budget selection is a separate feature and is not required here.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** False; **level:** `isolated_cpu_probe`.

**Ground-truth test files:** `tests/models/multimodal/processing/test_gemma4.py`

**Recorded test IDs:**

- `tests/models/multimodal/processing/test_gemma4.py::test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens`
- `tests/models/multimodal/processing/test_gemma4.py::test_get_prompt_updates_respects_nested_max_soft_tokens`

Audit probe IDs (not upstream test names):

- `audit-probes/test_audit_cpu.py::test_gemma_encoder_budget`

**Pytest / probe command and status:**

Executed audit probe; not a stock test-suite pass. Build a behavior-oriented hidden pytest from synthetic_test_idea before scoring agent PRs.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu AUDIT_TREE="$PWD" AUDIT_PR=41799 TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest audit-probes/test_audit_cpu.py --confcutdir=audit-probes -q
```

**Observed ground truth:** {"passed": 9, "skipped": 16}; exit 0

**Observed parent:** {"failed": 5, "passed": 4, "skipped": 16}; exit 1

- Both added stock test selectors were attempted at commit and parent: all 8 parameterized cases failed during model configuration with Transformers AmbiguousGlobalPerLayerAttributeError for head_dim. No stock processing pass is claimed.
- The exact-source isolated CPU budgeting probe passes 9 cases on the commit versus 5 failures/4 passes on the parent. Its real changed helper/method are compiled unchanged; the context and config are stubs. Unrelated candidates are explicitly skipped by the shared probe file.
- This confirms numeric budget accounting, not the full image processor, nested prompt replacement, audio processing or a Gemma inference run. The PR cites #40599 as separate auto-budget work; its issue endpoint/body was read, and it is a related PR rather than a bug issue.
- Exact commands below include revision-specific PYTHONPATH/cwd. Parent implementation is untouched; copied upstream changed test files: ['tests/models/multimodal/processing/test_gemma4.py']. A shared synthetic file explicitly skips other candidates, so skipped unrelated probes are not claimed as coverage.
- Downloadable evidence archive: [audit evidence bundle; source embedded below where applicable] Contains the actual probe source, candidate/parent/base JSON logs, diffs and inspected upstream metadata. Local log paths below correspond to same-named archive members.

Additional stock attempts (not necessarily discriminating):

- ground_truth: {"failed": 8}; exit 1
- parent: {"failed": 8}; exit 1

**Audit base-plus-patch:** {"passed": 9, "skipped": 16}; exit 0

**Synthetic test idea:** Executed the exact source of Gemma4ProcessingInfo.get_mm_max_tokens_per_item plus the candidate helper through AST extraction, with a minimal context/config and real configuration dictionaries. Nine cases cover top-level budgets, nested 560, None/non-mapping nested options, defaults and conflicting precedence; assert unchanged video accounting. For fuller coverage, run the added actual-image nested prompt-update test with a compatible Transformers Gemma 4 config or an isolated real prompt-update object graph.

### Rubric

- Report every supported explicit image budget, including values above and below the default.
- Respect top-level versus nested-image configuration precedence consistently.
- Keep image prompt expansion aligned with encoder planning rather than silently falling back to 280.
- Handle absent/non-mapping nested options and preserve default/video/audio behavior.
- Keep auto budget selection out of scope.

### Freeze-base suitability

**Applies cleanly:** True. Inspected earlier history using git log --format=%H 98661fe01..242afc6bf40d6d088b7b97eda1dd7f00ddcdfa21^ -- <each changed file>; overlap list is []. git -C /home/ubuntu/vllm-audit/base apply --check /home/ubuntu/vllm-audit/41799.diff returned exit 0 with '' output. In an independent detached 98661fe01 worktree, applied only this patch and actually reran the relevant CPU checks: exit 0, counts {'passed': 9, 'skipped': 16}. All non-test changed files were also mechanically byte-compared between base and candidate parent and are identical: {'vllm/model_executor/models/gemma4_mm.py': True}. Gemma4ProcessingInfo already exists at base; the helper/constants needed by the changed method are supplied by this patch. Actual source-probe imports require only Python standard types. Full stock test setup remains blocked by the installed Transformers configuration API, equally at parent and candidate; no extra earlier commit was needed for the nine base-patched budget probes.

Earlier file overlaps: `[]`.

**Why high signal:** Multimodal budget and kwargs precedence are pure Python and CPU-testable despite the GPU model module; exact source probes distinguish five incorrect parent budgets.

## 15. vllm-42244 — Avoid silent weights corruption when loading Nemotron Nano VL with reusable-buffer loaders like runai distributed streaming

**Set:** primary; **area:** Model loading / multimodal model weights; **difficulty:** medium.

**Ground truth:** [ac062147fae75e482e1fd0b2930746678253d7bb](https://github.com/COG-GTM/vllm/commit/ac062147fae75e482e1fd0b2930746678253d7bb); **parent:** `617239b70c46d3d53fc16619718da1c347c43fb6`.

**Files:** 2; **lines:** +55/−28.

- `tests/models/multimodal/test_nano_nemotron_vl.py`
- `vllm/model_executor/models/nano_nemotron_vl.py`

### Agent task statement

Loading Nemotron Nano VL / Nemotron-H VL checkpoints with a streaming loader that reuses tensor buffers between yields (e.g. Run:ai distributed streaming) silently corrupts weights: the model loads without error but produces empty or garbage output. The same checkpoint loads correctly with the default safetensors loader.

The weight-iteration contract for such loaders is that a yielded tensor is only valid until the next item is requested. The model's weight loading currently materializes or buffers tensors from the iterator before they are consumed, so many entries end up aliasing the same reused buffer and contain the last value written.

Expected: loading produces identical parameters regardless of whether the weight iterator reuses buffers. Language-model weights should be consumed as they are produced, and any vision/adapter/sound weights that must be held until later are safely retained. Existing behavior must be preserved: text-only mode skips multimodal weights, vision weights load without a sound encoder, and sound weights still require a sound encoder.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** False; **level:** `real_module_cpu_probe_with_mocks`.

**Ground-truth test files:** `tests/models/multimodal/test_nano_nemotron_vl.py`

**Recorded test IDs:**

- No CPU-discriminating stock IDs established; see the synthetic probe and next-stage test idea.

Audit probe IDs (not upstream test names):

- Standalone Python probe, not a pytest; `probe_command` is authoritative.

**Pytest / probe command and status:**

Executed audit probe; not a stock test-suite pass. Build a behavior-oriented hidden pytest from synthetic_test_idea before scoring agent PRs.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" .venv/bin/python audit-probes/probe_42244.py
```

**Observed ground truth:** stock: 3 passed; synthetic probe: PROBE PASS (at ac062147f and at base+cherry-pick)

**Observed parent:** stock (commit's test copied into parent 617239b70): 3 passed — non-discriminating; synthetic probe: AssertionError, all parent language weights equal last buffer value instead of 1.0/3.0/5.0 (also fails at base)

Synthetic probe calls the real NemotronH_Nano_VL_V2.load_weights on an object built with stub language_model/vision submodules (same pattern as stock test fakes) fed by a generator that mutates and re-yields one tensor. Stock tests run on CPU but do not discriminate. Logs: /home/ubuntu/audit/logs/c42244.{gt,parent}.log, /home/ubuntu/audit/logs/probe_42244.*.log.

**Synthetic test idea:** Author a behavior-oriented CPU regression from the executed probe and evidence below, mocking device/model construction but exercising the real changed callable.

### Rubric

- Loading yields identical parameter values for independent-storage and reusable-buffer tensor iterators.
- Do not eagerly materialize the full language-model checkpoint in memory.
- Retained vision/sound tensors remain valid after the source iterator advances.
- Preserve absent-sound-encoder handling and text-only loading.
- Use the reused-buffer probe for grading; the three stock tests pass on the parent and do not discriminate.

### Freeze-base suitability

**Applies cleanly:** True. No earlier commits in 98661fe01..ac062147f^ touch these files. `git apply --check` at base: OK; cherry-pick at base: probe pass; unpatched base: probe fail.

Earlier file overlaps: `[]`.

**Why high signal:** Silent-corruption bug requiring understanding of iterator/buffer-lifetime semantics, discriminable with a small CPU reused-buffer test.

## 16. vllm-41932 — [CPU] Fix spec decode kernel signatures for synthetic mode compatibility

**Set:** primary; **area:** speculative decoding / sampling (CPU); **difficulty:** easy.

**Ground truth:** [a54f0d1049c3acf18047f21e157d6452b7305957](https://github.com/COG-GTM/vllm/commit/a54f0d1049c3acf18047f21e157d6452b7305957); **parent:** `48698b1b9b30071952084ce12e8904b219193545`.

**Files:** 1; **lines:** +15/−0.

- `vllm/utils/cpu_triton_utils.py`

### Agent task statement

Speculative decoding on the CPU backend crashes in the rejection sampler. On CPU, the Triton rejection-sampling kernels are replaced by Python fallback wrappers that forward to C++ ops, but the GPU rejection sampler has since started passing extra arguments (for its synthetic-acceptance mode used for benchmarking, and a draft-probs flag passed by keyword) when launching the greedy and random rejection kernels. The CPU fallbacks do not accept the new calling convention, so every spec-decode step on CPU raises a `TypeError` (unexpected keyword argument / multiple values for an argument).

Expected: the CPU fallback kernels must accept exactly the arguments the shared rejection sampler passes for both greedy and random sampling and continue to dispatch to the existing CPU C++ ops with the same positional arguments as before. Synthetic acceptance mode is not implemented on CPU; if it is requested, fail loudly with a clear error rather than silently producing wrong results. Builds on CPU spec decode support from #32662.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** False; **level:** `real_python_wrappers_with_stubbed_native_ops`.

**Ground-truth test files:** None in the commit; synthetic evidence only.

**Recorded test IDs:**

- No CPU-discriminating stock IDs established; see the synthetic probe and next-stage test idea.

Audit probe IDs (not upstream test names):

- `audit-probes/test_synth_cpu_rejection_sig.py::test_greedy_kernel_accepts_sampler_calling_convention`
- `audit-probes/test_synth_cpu_rejection_sig.py::test_random_kernel_accepts_sampler_calling_convention`
- `audit-probes/test_synth_cpu_rejection_sig.py::test_synthetic_mode_rejected_on_cpu`

**Pytest / probe command and status:**

Executed audit probe; not a stock test-suite pass. Build a behavior-oriented hidden pytest from synthetic_test_idea before scoring agent PRs.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest audit-probes/test_synth_cpu_rejection_sig.py --confcutdir=audit-probes -q
```

**Observed ground truth:** SYNTHETIC: 3 passed, 14 warnings in 1.37s

**Observed parent:** parent: 3 failed — unexpected keyword argument 'SYNTHETIC_MODE' / got multiple values for argument 'NO_DRAFT_PROBS'

Commit adds no tests. Probe calls the real CPU wrappers in vllm/utils/cpu_triton_utils.py with the calling convention used by vllm/v1/sample/rejection_sampler.py; torch.ops._C replaced by a recording fake because the precompiled CUDA wheel lacks CPU C++ kernels (limitation: dispatch-to-C++ correctness not exercised). test_synthetic_mode_rejected_on_cpu checks the ground-truth assertion behavior — confirm intended semantics before using as hidden test. Base matrix: fix 3 passed / tests-only 3 failed. Logs: /home/ubuntu/audit/logs/a54f0d104-synth-{gt,par}.log

**Synthetic test idea:** On a CPU build with vllm._C CPU kernels: run RejectionSampler forward with small random draft/target probs on CPU and assert no TypeError and output shape/accept semantics.

### Rubric

- Greedy and random CPU fallbacks accept the sampler's new args (synthetic rates, SYNTHETIC_MODE, NO_DRAFT_PROBS keyword).
- C++ op called with unchanged argument order.
- Synthetic mode on CPU raises a clear error.
- No change to GPU Triton kernels.

### Freeze-base suitability

**Applies cleanly:** True. Mechanical: base-matrix script /home/ubuntu/audit/basematrix.sh created detached worktree at 98661fe01 and ran `git -C /home/ubuntu/repos/vllm diff a54f0d104^ a54f0d104 | git apply` -> applied with no errors. No overlapping files in 98661fe01..a54f0d104^.

Earlier file overlaps: `[]`.

**Why high signal:** Signature-drift crash between shared sampler and CPU fallback; small but requires reading the caller to infer the exact contract.

## 17. vllm-41806 — fix nixl side-channel host selection

**Set:** primary; **area:** KV transfer / distributed env propagation (Ray, engine-core actors, env compile factors); **difficulty:** medium.

**Ground truth:** [f9f770ca0b02f8b83ae92997a68efa797611ce36](https://github.com/COG-GTM/vllm/commit/f9f770ca0b02f8b83ae92997a68efa797611ce36); **parent:** `5cba6839e6cb8ede0c81947557b76617468d0d61`.

**Files:** 8; **lines:** +179/−2.

- `tests/test_envs.py`
- `tests/test_ray_env.py`
- `tests/v1/engine/test_core_engine_actor_manager.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py`
- `vllm/envs.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/engine/utils.py`
- `vllm/v1/executor/ray_utils.py`

### Agent task statement

In multi-node disaggregated prefill/decode deployments using the NIXL KV connector with Ray, workers on non-head nodes fail to start their NIXL handshake side-channel listener. The driver process sets VLLM_NIXL_SIDE_CHANNEL_HOST (explicitly or from its own IP), and that value is propagated verbatim into engine-core and worker actors that run on other machines, so they try to bind to the head node's address and the handshake cannot be established. The variable also currently participates in the environment-derived compile/cache factors even though it is a per-host network setting.

Expected behavior: a side-channel host chosen for the driver must not leak into actors scheduled on other nodes; each engine-core actor and worker should resolve its own host locally unless the user explicitly configured it in that actor's environment, and the handshake listener should bind to that actor-local host. VLLM_NIXL_SIDE_CHANNEL_HOST must be treated like other worker-specific host variables (excluded from Ray env propagation) and must not affect compile/cache factors. Constraints: keep existing behavior for single-node deployments and for users who set the variable explicitly on each node; do not require a GPU or a real Ray cluster to validate the behavior.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** True; **level:** `stock_pytest`.

**Ground-truth test files:** `tests/test_envs.py`, `tests/test_ray_env.py`, `tests/v1/engine/test_core_engine_actor_manager.py`

**Recorded test IDs:**

- `tests/test_envs.py::test_nixl_side_channel_host_is_not_compile_factor`
- `tests/test_ray_env.py::TestExclusion::test_worker_specific_host_vars_are_excluded`
- `tests/v1/engine/test_core_engine_actor_manager.py::test_driver_nixl_side_channel_host_does_not_leak_to_engine_core_actor`

**Pytest / probe command and status:**

Executed stock pytest; normalized worktree and venv locations. Parent/base runs overlay only the ground-truth regression tests, never implementation.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_envs.py::test_nixl_side_channel_host_is_not_compile_factor tests/test_ray_env.py::TestExclusion::test_worker_specific_host_vars_are_excluded tests/v1/engine/test_core_engine_actor_manager.py
```

**Observed ground truth:** stock pytest on CPU: 3 passed (at f9f770ca0); also base 98661fe01 + cherry-pick: 3 passed

**Observed parent:** stock pytest, tests copied from commit into parent 5cba6839e without implementation: 3 failed; also base 98661fe01 + copied tests only: 3 failed

Stock pytest, CPU-only torch 2.11.0+cpu, Python 3.12 uv venv. Parent failures: 'assert VLLM_NIXL_SIDE_CHANNEL_HOST not in compile_factors()', 'assert VLLM_NIXL_SIDE_CHANNEL_HOST not in {...excluded vars}', and actor test 'driver-only-nixl-host-<uuid>' == leaked value. Logs: /home/ubuntu/audit/logs/c41806.{gt,parent,base,base+fix}.log. No Ray cluster needed (actor manager test mocks Ray).

**Coordinator freeze-base rerun:**

```json
[
  {
    "pr_number": 41806,
    "mode": "fix",
    "exit_code": 0,
    "summary": [
      "3 passed in 9.51s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_envs.py::test_nixl_side_channel_host_is_not_compile_factor tests/test_ray_env.py::TestExclusion::test_worker_specific_host_vars_are_excluded tests/v1/engine/test_core_engine_actor_manager.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  },
  {
    "pr_number": 41806,
    "mode": "tests",
    "exit_code": 1,
    "summary": [
      "3 failed in 9.94s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_envs.py::test_nixl_side_channel_host_is_not_compile_factor tests/test_ray_env.py::TestExclusion::test_worker_specific_host_vars_are_excluded tests/v1/engine/test_core_engine_actor_manager.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  }
]
```

**Synthetic test idea:** Not required for the demonstrated stock CPU regression; retain equivalent behavioral coverage.

### Rubric

- Driver-chosen side-channel host is not propagated to engine-core/worker actors on other nodes
- Explicit per-actor/per-node configuration is still honored
- Handshake listener binds to the actor-local resolved host
- Variable excluded from Ray env propagation and from compile/cache factors
- No regression for single-node or non-NIXL paths

### Freeze-base suitability

**Applies cleanly:** True. Overlapping earlier commits in 98661fe01..f9f770ca0^: 3f5bd482f (#NIXL pre-admission rejection) and 44e6b44a2 (Elastic EP test fix). `git format-patch -1 f9f770ca0 | git apply --check` at 98661fe01: OK. Actual cherry-pick --no-commit at base succeeded and the 3 tests pass there; tests-only at base fail 3/3, so tests stay meaningful on base. No missing prerequisites observed.

Earlier file overlaps: `[{"file": "vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py", "earlier_commits": ["3f5bd482f5c1a5dbdffbbf68d624e20bb7032013"]}, {"file": "vllm/v1/engine/core.py", "earlier_commits": ["3f5bd482f5c1a5dbdffbbf68d624e20bb7032013"]}, {"file": "vllm/v1/engine/utils.py", "earlier_commits": ["44e6b44a213ca52706c208c639da3fe20d867728"]}]`.

**Why high signal:** Real multi-node production bug with a crisp, CPU-runnable, discriminating 3-test suite spanning env handling, Ray env propagation, and actor startup.

## 18. vllm-39212 — [feat] Add explicit /start_weight_update and /finish_weight_update APIs for weight transfer

**Set:** primary; **area:** API / RLHF weight transfer; **difficulty:** hard.

**Ground truth:** [e3b65a5ba069b350120ca7a614a010787d2de867](https://github.com/COG-GTM/vllm/commit/e3b65a5ba069b350120ca7a614a010787d2de867); **parent:** `30f519e947d4dbea0bceef002f2f567faa8c8457`.

**Files:** 19; **lines:** +288/−54.

- `docs/training/weight_transfer/README.md`
- `docs/training/weight_transfer/base.md`
- `docs/training/weight_transfer/ipc.md`
- `docs/training/weight_transfer/nccl.md`
- `examples/rl/rlhf_async_new_apis.py`
- `examples/rl/rlhf_http_ipc.py`
- `examples/rl/rlhf_http_nccl.py`
- `examples/rl/rlhf_ipc.py`
- `examples/rl/rlhf_nccl.py`
- `examples/rl/rlhf_nccl_fsdp_ep.py`
- `tests/entrypoints/openai/test_openai_schema.py`
- `tests/entrypoints/weight_transfer/test_weight_transfer_llm.py`
- `vllm/distributed/weight_transfer/base.py`
- `vllm/distributed/weight_transfer/ipc_engine.py`
- `vllm/engine/protocol.py`
- `vllm/entrypoints/llm.py`
- `vllm/entrypoints/serve/rlhf/api_router.py`
- `vllm/v1/engine/async_llm.py`
- `vllm/v1/worker/gpu_worker.py`

### Agent task statement

vLLM's weight-transfer API for RL training loops currently treats every `update_weights` call as a complete, self-contained update. That makes chunked transfers impossible: trainers that want to bound memory by sending weights in several chunks (e.g. packed CUDA-IPC buffers), or that have multiple senders when the trainer uses pipeline parallelism, need the per-update setup and finalization (such as layerwise reload processing for checkpoint-format weights) to happen exactly once around many partial transfers. Please add an explicit update session to the weight-transfer API: a way to start an update (indicating whether incoming weights are in checkpoint format or already in kernel format), any number of weight-update calls that transfer subsets of weights, and a way to finish the update that performs final processing. This must be exposed consistently across the offline `LLM` API, the async engine client, and the HTTP RLHF endpoints. Misuse should fail clearly: updating or finishing without an active session, or starting a second session while one is active, should raise an error.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** False; **level:** `real_module_cpu_probe_with_mocks`.

**Ground-truth test files:** `tests/entrypoints/weight_transfer/test_weight_transfer_llm.py`, `tests/entrypoints/openai/test_openai_schema.py`

**Recorded test IDs:**

- `tests/entrypoints/weight_transfer/test_weight_transfer_llm.py::test_update_weights_calls_engine`
- `tests/entrypoints/weight_transfer/test_weight_transfer_llm.py::test_full_weight_transfer_flow`

Audit probe IDs (not upstream test names):

- `audit-probes/test_probe_weight_update.py::test_update_requires_start`
- `audit-probes/test_probe_weight_update.py::test_chunked_checkpoint_flow_single_init_finalize`
- `audit-probes/test_probe_weight_update.py::test_double_start_rejected`
- `audit-probes/test_probe_weight_update.py::test_kernel_format_skips_layerwise`
- `audit-probes/test_probe_weight_update.py::test_llm_and_router_expose_new_apis`

**Pytest / probe command and status:**

Executed audit probe; not a stock test-suite pass. Build a behavior-oriented hidden pytest from synthetic_test_idea before scoring agent PRs.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest audit-probes/test_probe_weight_update.py --confcutdir=audit-probes -q
```

**Observed ground truth:** synthetic pytest (executed): 5 passed

**Observed parent:** synthetic pytest (executed): 5 failed; AttributeError: 'Worker' object has no attribute 'start_weight_update' / type object 'LLM' has no attribute 'start_weight_update'

Harness: detached worktrees /home/ubuntu/wt/c_<sha> (commit) and /home/ubuntu/wt/p_<sha> (parent + commit's test files copied verbatim, byte-identical, no impl). Python: /home/ubuntu/wt/base/.venv/bin/python (uv, py3.12, VLLM_USE_PRECOMPILED=1 editable install, torch 2.11.0+cpu), run from worktree root with PYTHONPATH=$PWD so the worktree source is imported. Synthetic harness instantiates the REAL vllm.v1.worker.gpu_worker.Worker via **new** (no device init), with MagicMock weight_transfer_engine and model, monkeypatched initialize/finalize_layerwise_reload and torch.accelerator.synchronize; calls real start_weight_update/update_weights/finish_weight_update; also calls real LLM.start/finish_weight_update on a fake self and inspects real FastAPI router routes. Parent failures are API-absence (AttributeError) rather than behavioral, as expected for a feature. Async engine/protocol paths not exercised.  Import mode: normal package imports of the worktree (no AST extraction); stubs only for collaborators (see SUMMARY.md). BASE-ONLY: base+patch synthetic 5 passed; base+tests-only 5 failed. WARNING: stock test_weight_transfer_llm.py reports 5 PASSED on CPU at both base+patch and base+tests-only because pytest.skip('Need at least 1 GPU') inside create_new_process_for_each_test fork is reported as pass -> stock file is non-discriminating on CPU; use synthetic probe.

**Stock-test caveat:** Additional CPU attempt of test_weight_transfer_llm.py on freeze base: 5 reported passed with patch AND with tests only. These are false positives from pytest.skip(Need at least 1 GPU) inside a forked subprocess, not CPU model/update coverage. Use the executed five-test real-worker probe instead.

**Synthetic test idea:** Implemented; could add AsyncLLM.start_weight_update test with mocked engine_core.collective_rpc_async.

### Rubric

- Session state: start/update*/finish, with RuntimeError on misuse (update/finish without start, double start)
- Layerwise reload initialized once at start and finalized once at finish for checkpoint-format; skipped for kernel format
- LLM, AsyncLLM/EngineClient protocol and HTTP router expose matching start/finish APIs
- Existing update_weights callers/examples updated to new flow
- Hidden tests are synthetic against Worker method names; interface names must be given in task interface

### Freeze-base suitability

**Applies cleanly:** True. Mechanical: `git apply --check` of full commit patch on detached worktree at base 98661fe01 -> OK; `git cherry-pick -n` -> OK (no conflicts). Overlapping earlier commits: 52458b60a (#42042, examples/rl/rlhf_async_new_apis.py) and 1c5887661 (#41344, gpu_worker.py); patch still applies/cherry-picks cleanly at base. Independent `git apply` of only this patch on pristine base: OK; tests green on base+patch, no missing-API/import errors from earlier-window commits.

Earlier file overlaps: `[{"file": "examples/rl/rlhf_async_new_apis.py", "earlier_commits": ["52458b60a88310a7f730f067e5a431a55cd34afd"]}, {"file": "vllm/v1/worker/gpu_worker.py", "earlier_commits": ["1c58876618dc7275520297cbbd1244a64f551952"]}]`.

**Why high signal:** Cross-layer API feature (worker state machine + LLM + HTTP) with real CPU-testable state-machine behavior.

## 19. vllm-42081 — [Bug] Fix kimi dtype issue with `mm_projector_forward`

**Set:** primary; **area:** Multimodal model (Kimi-K2.5/K2.6 vision projector); **difficulty:** easy.

**Ground truth:** [3f9c0c25b3312c6b9e9fa70c4b9849eb53a335ba](https://github.com/COG-GTM/vllm/commit/3f9c0c25b3312c6b9e9fa70c4b9849eb53a335ba); **parent:** `a2e776d7168e1a75fe32438aea4254542602c012`.

**Files:** 1; **lines:** +3/−0.

- `vllm/model_executor/models/kimi_k25_vit.py`

### Agent task statement

Serving moonshotai/Kimi-K2.6 with image inputs crashes during encoder profiling / first multimodal request when batch-invariant mode or data-parallel encoder mode is enabled: the worker raises a dtype mismatch inside the multimodal projector (layer norm / linear receiving activations whose dtype differs from the projector parameters, e.g. float32 activations vs bfloat16 weights). Text-only requests are unaffected.

Expected: the vision-to-language projector accepts vision tower outputs regardless of the dtype they arrive in and produces outputs in the projector's parameter dtype, so profiling and image requests succeed. Constraints: no change in behavior or extra copies when dtypes already match; do not change the projector's weights or model dtype configuration.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** False; **level:** `real_module_cpu_probe_with_mocks`.

**Ground-truth test files:** None in the commit; synthetic evidence only.

**Recorded test IDs:**

- No CPU-discriminating stock IDs established; see the synthetic probe and next-stage test idea.

Audit probe IDs (not upstream test names):

- Standalone Python probe, not a pytest; `probe_command` is authoritative.

**Pytest / probe command and status:**

Executed audit probe; not a stock test-suite pass. Build a behavior-oriented hidden pytest from synthetic_test_idea before scoring agent PRs.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" .venv/bin/python audit-probes/probe_42081.py
```

**Observed ground truth:** synthetic probe: PROBE PASS (outputs [(3,16),(2,16)] torch.bfloat16) at 3f9c0c25b and base+cherry-pick

**Observed parent:** synthetic probe: RuntimeError: mixed dtype (CPU): expect parameter to have scalar type of Float (torch.layer_norm) at parent and at base

Probe builds the real projector under VllmConfig with single-rank gloo init_distributed_environment + initialize_model_parallel(1,1), casts to bf16, runs quant_method.process_weights_after_loading (needed on CPU so ReplicatedLinear has cpu_linear), then calls the real mm_projector_forward. First attempts failed in both versions with 'tensor model parallel group is not initialized' and then "'ReplicatedLinear' object has no attribute 'cpu_linear'" — harness issues, fixed. Logs: /home/ubuntu/audit/logs/probe_42081.*.log.

**Synthetic test idea:** Author a behavior-oriented CPU regression from the executed probe and evidence below, mocking device/model construction but exercising the real changed callable.

### Rubric

- Projector input cast to projector parameter dtype
- No-op when dtypes match
- Outputs in projector dtype with correct shapes

### Freeze-base suitability

**Applies cleanly:** True. No earlier commits in range touch kimi_k25_vit.py. `git apply --check` at base: OK; probe pass on base+cherry-pick, fail on base.

Earlier file overlaps: `[]`.

**Why high signal:** Real user crash from linked issue with deterministic CPU repro, though fix is small.

## 20. vllm-39243 — add: LFM2/2.5 Tool Parser

**Set:** primary; **area:** tool/reasoning parsers; **difficulty:** hard.

**Ground truth:** [50f2db25554a3a4b9aa4151ec8f6bc0b65bc878d](https://github.com/COG-GTM/vllm/commit/50f2db25554a3a4b9aa4151ec8f6bc0b65bc878d); **parent:** `09a7cc5ba94c5c1b1c908a29f28ed41c793f1e9f`.

**Files:** 4; **lines:** +864/−5.

- `tests/tool_parsers/test_lfm2_tool_parser.py`
- `vllm/tool_parsers/__init__.py`
- `vllm/tool_parsers/lfm2_tool_parser.py`
- `vllm/tool_parsers/utils.py`

### Agent task statement

LiquidAI LFM2 / LFM2.5 models emit tool calls as pythonic call lists wrapped in special tokens, e.g. `<|tool_call_start|>[get_current_temperature(location="Tokyo")]<|tool_call_end|>`. vLLM has no tool parser for this format, so with `--enable-auto-tool-choice` the raw markup is returned in `message.content` and `tool_calls` is empty.

Add a tool parser selectable as `--tool-call-parser lfm2` that: extracts one or more calls from each `<|tool_call_start|>...<|tool_call_end|>` block into OpenAI-style tool calls with JSON-encoded arguments (supporting strings, numbers, booleans, None, nested lists/dicts, and dotted function names like `a.b.c`); returns surrounding text (before/after blocks) as content and never leaks tool-call markup into content; works in streaming mode regardless of how deltas are chunked (character-by-character, a full block plus trailing text in one delta, leading content plus block, etc.); ensures special tokens are not stripped from output so the markers are visible to the parser; and handles pathological input without catastrophic regex backtracking. Non-tool output must pass through unchanged.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** True; **level:** `stock_pytest`.

**Ground-truth test files:** `tests/tool_parsers/test_lfm2_tool_parser.py`

**Recorded test IDs:**

- `tests/tool_parsers/test_lfm2_tool_parser.py::test_no_tool_call`
- `tests/tool_parsers/test_lfm2_tool_parser.py::test_tool_call`
- `tests/tool_parsers/test_lfm2_tool_parser.py::test_streaming_tool_call_with_large_steps`
- `tests/tool_parsers/test_lfm2_tool_parser.py::test_streaming_full_block_and_trailing_in_single_delta`
- `tests/tool_parsers/test_lfm2_tool_parser.py::test_streaming_leading_content_and_full_block_in_single_delta`
- `tests/tool_parsers/test_lfm2_tool_parser.py::test_streaming_leading_block_and_trailing_in_single_delta`
- `tests/tool_parsers/test_lfm2_tool_parser.py::test_echoed_tool_call_body_not_leaked_to_content`
- `tests/tool_parsers/test_lfm2_tool_parser.py::test_streaming_char_by_char_multi_dict_list`
- `tests/tool_parsers/test_lfm2_tool_parser.py::test_streaming_dotted_name_in_single_delta`
- `tests/tool_parsers/test_lfm2_tool_parser.py::test_adjust_request_disables_skip_special_tokens`
- `tests/tool_parsers/test_lfm2_tool_parser.py::test_regex_timeout_handling`

**Pytest / probe command and status:**

Executed stock pytest; normalized worktree and venv locations. Parent/base runs overlay only the ground-truth regression tests, never implementation.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest -q -p no:cacheprovider tests/tool_parsers/test_lfm2_tool_parser.py
```

**Observed ground truth:** STOCK pytest: 46 passed (28 lfm2 + 18 pythonic regression), 14 warnings

**Observed parent:** STOCK pytest with commit's test file on parent: 28 failed, 18 passed; all 28 lfm2 tests fail with KeyError: "Tool parser 'lfm2' not found."; pythonic tests pass on both

Tokenizer fixture downloaded from HF hub successfully in this env. Pythonic parser tests included to guard shared-utils change. Logs: /home/ubuntu/wt/logs/tests/lfm2_{parent,commit}.log

**Coordinator freeze-base rerun:**

```json
[
  {
    "pr_number": 39243,
    "mode": "fix",
    "exit_code": 0,
    "summary": [
      "46 passed in 11.66s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/tool_parsers/test_lfm2_tool_parser.py tests/tool_parsers/test_pythonic_tool_parser.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  },
  {
    "pr_number": 39243,
    "mode": "tests",
    "exit_code": 1,
    "summary": [
      "28 failed, 18 passed in 9.77s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/tool_parsers/test_lfm2_tool_parser.py tests/tool_parsers/test_pythonic_tool_parser.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  }
]
```

**Synthetic test idea:** Not required for the demonstrated stock CPU regression; retain equivalent behavioral coverage.

### Rubric

- Parser registered under name 'lfm2' and selectable from CLI
- Correct non-streaming extraction incl. multiple calls, nested args, dotted names
- Streaming robust to arbitrary delta chunking; no markup leaked into content
- adjust_request disables skip_special_tokens
- Regex timeout / pathological input handled; existing pythonic parser unaffected

### Freeze-base suitability

**Applies cleanly:** True. No overlap with 98661fe01..50f2db255^. At base: git apply --check OK; cherry-pick --no-commit OK, 0 conflicts.

Earlier file overlaps: `[]`.

**Why high signal:** Substantial new-feature parser task with a rich hidden suite (streaming edge cases) that is entirely CPU-runnable and fails cleanly on parent.

## 21. vllm-41423 — [Bugfix] Fix spawn_new_process_for_each_test silently swallowing test failures

**Set:** held_out; **area:** utils/test_infra; **difficulty:** medium.

**Ground truth:** [ee38750a7565ee7158ddc78ff5abd12e9c0d3733](https://github.com/COG-GTM/vllm/commit/ee38750a7565ee7158ddc78ff5abd12e9c0d3733); **parent:** `27e0057aeda6bc443069c20fdf2f3cc95ed892f3`.

**Files:** 4; **lines:** +306/−245.

- `tests/utils.py`
- `tests/utils_/test_spawn_decorator.py`
- `tests/v1/cudagraph/test_cudagraph_dispatch.py`
- `tests/v1/logits_processors/test_custom_online.py`

### Agent task statement

vLLM's test utilities provide a decorator that runs each decorated test in a freshly spawned Python process, used for tests that need CUDA/process isolation. Reported in issue #41415: the decorator is broken - when the decorated test body raises (an assertion error or any exception), the outer pytest test still reports PASSED. Failures in the child process are never propagated, so entire suites using the decorator can silently mask regressions. Skips raised inside the child also are not reported faithfully, and parametrized arguments must still reach the test body. Repro: decorate a test whose body is `assert False` with the spawn decorator and run pytest; it passes. Expected: a test body that fails in the child must make the parent test fail with an error that includes the child's exception/traceback; a passing body passes; pytest.skip inside the child results in a skip; parametrized/keyword arguments are delivered to the child unchanged. Constraints: must keep using a separate spawned process per test (no fork), must not depend on the test being importable by module path re-execution tricks only, and must work for module-level test functions with arguments.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** True; **level:** `stock_pytest`.

**Ground-truth test files:** `tests/utils_/test_spawn_decorator.py`

**Recorded test IDs:**

- `tests/utils_/test_spawn_decorator.py::test_spawn_decorator_passing`
- `tests/utils_/test_spawn_decorator.py::test_spawn_decorator_failure_is_caught`
- `tests/utils_/test_spawn_decorator.py::test_spawn_decorator_skip`
- `tests/utils_/test_spawn_decorator.py::test_spawn_decorator_parametrized`

**Pytest / probe command and status:**

Executed stock pytest; normalized worktree and venv locations. Parent/base runs overlay only the ground-truth regression tests, never implementation.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest -q -p no:cacheprovider tests/utils_/test_spawn_decorator.py
```

**Observed ground truth:** 4 passed, 1 xfailed (stock pytest, CPU); test_spawn_decorator_failure_is_caught is an expected-failure test that xfails because the child failure now propagates

**Observed parent:** 4 passed, 1 failed, 14 warnings with commit's test file copied into parent; FAILED test_spawn_decorator_failure_is_caught - [XPASS(strict)]: the child failure was swallowed so the test marked xfail(raises=RuntimeError, strict=True) unexpectedly passed

Real tests/utils.py decorator exercised with real spawned subprocesses on CPU. Discriminating. Commit also edits two GPU test files (cudagraph, custom logits processors) that are not part of hidden tests.

**Coordinator freeze-base rerun:**

```json
[
  {
    "pr_number": 41423,
    "mode": "fix",
    "exit_code": 0,
    "summary": [
      "4 passed, 1 xfailed, 14 warnings in 4.22s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/utils_/test_spawn_decorator.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  },
  {
    "pr_number": 41423,
    "mode": "tests",
    "exit_code": 1,
    "summary": [
      "1 failed, 4 passed, 14 warnings in 42.89s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/utils_/test_spawn_decorator.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  }
]
```

**Synthetic test idea:** Not required for the demonstrated stock CPU regression; retain equivalent behavioral coverage.

### Rubric

- Child exceptions make the parent test fail and include the child traceback/message
- pytest.skip in child surfaces as a skip, not pass/fail
- Parametrized and keyword args reach the child
- Still spawns a new process (no fork-based shortcut)
- Does not break existing decorated tests

### Freeze-base suitability

**Applies cleanly:** True. No earlier commits in range touch tests/utils.py or other changed files; git apply --check at base: CLEAN. Requires cloudpickle and tblib (present in test requirements).

Earlier file overlaps: `[]`.

**Why high signal:** Real infrastructure bug (false-green tests) with a crisp CPU-only subprocess regression that discriminates parent from fix.

## 22. vllm-42026 — [Bugfix] Preserve leading/trailing whitespace in GLM non-streaming tool parser

**Set:** held_out; **area:** tool_parsers; **difficulty:** easy.

**Ground truth:** [f6490a28413526adefae8dd7af32906d060e0436](https://github.com/COG-GTM/vllm/commit/f6490a28413526adefae8dd7af32906d060e0436); **parent:** `a2812becd6952ab66b1ee54856bb855a0551353b`.

**Files:** 3; **lines:** +40/−3.

- `tests/tool_parsers/test_glm47_moe_tool_parser.py`
- `tests/tool_parsers/test_glm4_moe_tool_parser.py`
- `vllm/tool_parsers/glm4_moe_tool_parser.py`

### Agent task statement

When serving GLM-4.5/4.6/4.7 MoE models with tool calling enabled, string-typed tool arguments lose meaningful whitespace in non-streaming responses. For example, if the model emits a tool call whose argument value is `indented code` (as commonly happens with code-editing tools, diff/patch application, or text-replacement tools), the parsed tool call arguments returned to the client contain `"indented code"` instead of the original value with its leading and trailing spaces. This silently corrupts code edits and diffs because indentation is significant. Non-string arguments (integers, booleans, objects, arrays) are expected to keep being parsed correctly even if the model surrounds them with incidental whitespace or newlines. Please fix the GLM non-streaming tool-call extraction so that values of string-typed parameters (per the request's tool JSON schema) are returned exactly as generated, while typed/non-string values continue to be deserialized correctly. Both the GLM-4 MoE and GLM-4.7 variants of the parser should behave consistently.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** True; **level:** `stock_pytest`.

**Ground-truth test files:** `tests/tool_parsers/test_glm4_moe_tool_parser.py`, `tests/tool_parsers/test_glm47_moe_tool_parser.py`

**Recorded test IDs:**

- `tests/tool_parsers/test_glm4_moe_tool_parser.py::test_whitespace_preserved_in_arg_values`
- `tests/tool_parsers/test_glm47_moe_tool_parser.py::TestGlm47ExtractToolCalls::test_whitespace_preserved_in_arg_values`

**Pytest / probe command and status:**

Executed stock pytest; normalized worktree and venv locations. Parent/base runs overlay only the ground-truth regression tests, never implementation.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest -q -p no:cacheprovider tests/tool_parsers/test_glm4_moe_tool_parser.py tests/tool_parsers/test_glm47_moe_tool_parser.py
```

**Observed ground truth:** stock pytest (executed): 62 passed, 0 failed (two files, full collection)

**Observed parent:** stock pytest (executed, commit tests copied into parent): 2 failed, 60 passed; failures are exactly the two whitespace regression tests (AssertionError on stripped value)

Harness: detached worktrees /home/ubuntu/wt/c_<sha> (commit) and /home/ubuntu/wt/p_<sha> (parent + commit's test files copied verbatim, byte-identical, no impl). Python: /home/ubuntu/wt/base/.venv/bin/python (uv, py3.12, VLLM_USE_PRECOMPILED=1 editable install, torch 2.11.0+cpu), run from worktree root with PYTHONPATH=$PWD so the worktree source is imported. Logs: <bundle>/log_f6490a284.txt. Tokenizer for GLM tests was available (HF cache/network) on CPU; run reproduced twice with identical counts. Import mode: normal package imports of the worktree (no AST extraction); stubs only for collaborators (see SUMMARY.md).

**Coordinator freeze-base rerun:**

```json
[
  {
    "pr_number": 42026,
    "mode": "fix",
    "exit_code": 0,
    "summary": [
      "62 passed in 14.41s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/tool_parsers/test_glm47_moe_tool_parser.py tests/tool_parsers/test_glm4_moe_tool_parser.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  },
  {
    "pr_number": 42026,
    "mode": "tests",
    "exit_code": 1,
    "summary": [
      "2 failed, 60 passed in 12.20s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/tool_parsers/test_glm47_moe_tool_parser.py tests/tool_parsers/test_glm4_moe_tool_parser.py",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  }
]
```

**Synthetic test idea:** Not required for the demonstrated stock CPU regression; retain equivalent behavioral coverage.

### Rubric

- Strips only non-string values; string-typed args returned verbatim (spaces/newlines preserved)
- Uses request tool schema to decide string type; unknown params behavior unchanged
- Non-string (int/bool/json) values with surrounding whitespace still deserialize
- Fix applies to GLM-4 MoE and GLM-4.7 (subclass) paths
- No regression in remaining 60 GLM parser tests

### Freeze-base suitability

**Applies cleanly:** True. Mechanical: `git apply --check` of full commit patch on detached worktree at base 98661fe01 -> OK; `git cherry-pick -n` -> OK (no conflicts). No earlier commit in 98661fe01..f6490a284^ touches its 3 files. Tests depend only on code present at base.

Earlier file overlaps: `[]`.

**Why high signal:** Small, real user-facing correctness bug with a crisp fail-to-pass stock test that runs on CPU in seconds.

## 23. vllm-41499 — [Performance] Make safetensors checkpoint prefetch settings configurable

**Set:** held_out; **area:** model loading / config; **difficulty:** medium.

**Ground truth:** [21943d4c258983c4b8eb56d50029aca4f18e4629](https://github.com/COG-GTM/vllm/commit/21943d4c258983c4b8eb56d50029aca4f18e4629); **parent:** `f396bee56fb516195a3669e5499129021ced4443`.

**Files:** 4; **lines:** +87/−15.

- `vllm/config/load.py`
- `vllm/engine/arg_utils.py`
- `vllm/model_executor/model_loader/default_loader.py`
- `vllm/model_executor/model_loader/weight_utils.py`

### Agent task statement

vLLM can prefetch safetensors checkpoint files into the OS page cache before loading weights (useful on network filesystems). The prefetch currently uses a hard-coded worker-thread count (8) and a hard-coded read block size (16 MiB), which is far from optimal on some storage backends (e.g. high-latency object-store mounts want more threads; others want larger reads).

Add two load-config options, exposed as engine CLI flags `--safetensors-prefetch-num-threads` and `--safetensors-prefetch-block-size` and as `LoadConfig.safetensors_prefetch_num_threads` / `LoadConfig.safetensors_prefetch_block_size`. Defaults must remain 8 threads and 16 MiB so existing behavior is unchanged. Both must be positive integers (reject 0/negative at config validation); the block size should accept human-readable sizes on the CLI like other size flags (e.g. `1M`). The configured values must actually be honored when the default model loader prefetches checkpoint files: no more than the configured number of files read concurrently, and reads issued in chunks of the configured size. The prefetch helper should also reject non-positive values if called directly.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** False; **level:** `real_module_cpu_probe_with_mocks`.

**Ground-truth test files:** None in the commit; synthetic evidence only.

**Recorded test IDs:**

- No CPU-discriminating stock IDs established; see the synthetic probe and next-stage test idea.

Audit probe IDs (not upstream test names):

- `audit-probes/test_synth_safetensors_prefetch.py::test_cli_args_flow_into_load_config`
- `audit-probes/test_synth_safetensors_prefetch.py::test_defaults_preserved`
- `audit-probes/test_synth_safetensors_prefetch.py::test_non_positive_rejected`
- `audit-probes/test_synth_safetensors_prefetch.py::test_prefetch_uses_configured_block_size_and_threads`
- `audit-probes/test_synth_safetensors_prefetch.py::test_prefetch_rejects_invalid_values`

**Pytest / probe command and status:**

Executed audit probe; not a stock test-suite pass. Build a behavior-oriented hidden pytest from synthetic_test_idea before scoring agent PRs.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest audit-probes/test_synth_safetensors_prefetch.py --confcutdir=audit-probes -q
```

**Observed ground truth:** SYNTHETIC: 6 passed, 14 warnings in 2.14s

**Observed parent:** SYNTHETIC copied into parent: 4 failed, 2 passed — argparse SystemExit: 2 (unknown flags), AttributeError: 'LoadConfig' object has no attribute 'safetensors_prefetch_num_threads', TypeError:_prefetch_all_checkpoints() got an unexpected keyword argument 'num_prefetch_threads'

Commit adds no tests. Synthetic probe /home/ubuntu/audit/synth/test_synth_safetensors_prefetch.py uses real EngineArgs/FlexibleArgumentParser, LoadConfig pydantic validation, and the real weight_utils._prefetch_all_checkpoints over tmp files with a wrapped builtins.open measuring read sizes and concurrency. Earlier probe revision had 2 bad expectations (assumed `1M`=1_000_000 but vLLM human_readable_int gives 1048576; broken **exit** instrumentation gave peak=6) — corrected and rerun. Helper kwarg names (num_prefetch_threads, block_size) are implementation-coupled; hidden tests should test via LoadConfig + DefaultModelLoader or the spec must name the helper signature. Base matrix: fix 6 passed / tests-only 4 failed, 2 passed. Logs: /home/ubuntu/audit/logs/st2-{gt,par}.log

**Synthetic test idea:** Interface-agnostic: construct DefaultModelLoader with LoadConfig(safetensors_prefetch_num_threads=2, safetensors_prefetch_block_size=4096, enable prefetch) over a tmp dir of small .safetensors files and assert read sizes/peak concurrency.

### Rubric

- Two new LoadConfig fields with defaults 8 and 16 MiB and positive-int validation.
- CLI flags registered; block size accepts human-readable values.
- DefaultModelLoader passes the configured values to prefetch; prefetch honors concurrency and block size.
- Default behavior unchanged when flags not set.

### Freeze-base suitability

**Applies cleanly:** True. Mechanical: base-matrix script /home/ubuntu/audit/basematrix.sh created detached worktree at 98661fe01 and ran `git -C /home/ubuntu/repos/vllm diff 21943d4c2^ 21943d4c2 | git apply` -> applied with no errors. One overlapping commit in 98661fe01..21943d4c2^: 0d382ecde (vllm-40951) on vllm/engine/arg_utils.py; patch still applied cleanly at base. Synthetic tests meaningful on base.

Earlier file overlaps: `[{"file": "vllm/engine/arg_utils.py", "earlier_commits": ["0d382ecde8c6771fe07854eaad3af73c1353e031"]}]`.

**Why high signal:** End-to-end config plumbing (CLI -> config validation -> loader -> threaded I/O) with checkable runtime effect on CPU.

## 24. vllm-41965 — [Compressed Tensors] Allow configs with non-explicit ignores

**Set:** held_out; **area:** model loading/quantization config; **difficulty:** easy.

**Ground truth:** [c1819ca283021b65ac1ea71901d99f26b1103efe](https://github.com/COG-GTM/vllm/commit/c1819ca283021b65ac1ea71901d99f26b1103efe); **parent:** `969fbfb4a985ad763a6109851bad9dfc94fea289`.

**Files:** 3; **lines:** +34/−19.

- `tests/quantization/test_compressed_tensors.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/utils.py`

### Agent task statement

Loading a compressed-tensors checkpoint fails when its configuration lists only the layers to be quantized and does not explicitly enumerate every other layer in an ignore list. For example, a config group targeting attention projections can leave an unrelated linear layer unmatched and trigger an Unable to find matching target error during initialization. Checkpoint producers expect unspecified layers to remain unquantized. Please accept these configs so unmatched layers behave as unquantized layers, whether or not they were explicitly ignored. Existing explicitly ignored layers must keep that behavior. Layers that do match must still receive their intended quantization schemes, including regular-expression targets and fused-module mappings. Preserve support for existing fully specified configurations and avoid suppressing genuine invalid-config errors indiscriminately. Exercise configuration and scheme selection on CPU rather than relying on a downloaded quantized checkpoint.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** True; **level:** `stock_pytest`.

**Ground-truth test files:** `tests/quantization/test_compressed_tensors.py`

**Recorded test IDs:**

- `tests/quantization/test_compressed_tensors.py::test_find_matched_target_returns_none_on_no_match`
- `tests/quantization/test_compressed_tensors.py::test_get_scheme_dict_returns_none_on_no_match`

**Pytest / probe command and status:**

Executed stock pytest; normalized worktree and venv locations. Parent/base runs overlay only the ground-truth regression tests, never implementation.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest -q -p no:cacheprovider tests/quantization/test_compressed_tensors.py -k 'find_matched_target_returns_none or get_scheme_dict_returns_none or lm_head'
```

**Observed ground truth:** STOCK pytest, subset via -k: 5 passed, 19 deselected, 14 warnings

**Observed parent:** STOCK pytest with commit's test file on parent: 2 failed, 3 passed, 19 deselected; the 2 new tests fail with ValueError (unable to find matching target)

Full file NOT run: remaining 19 tests load real quantized checkpoints via vllm_runner/GPU and were deselected; only the selected subset was executed on CPU. Logs: /home/ubuntu/wt/logs/tests/ct_{parent,commit}.log

**Coordinator freeze-base rerun:**

```json
[
  {
    "pr_number": 41965,
    "mode": "fix",
    "exit_code": 0,
    "summary": [
      "5 passed, 19 deselected, 14 warnings in 2.39s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/quantization/test_compressed_tensors.py -k 'find_matched_target_returns_none or get_scheme_dict_returns_none or lm_head'",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  },
  {
    "pr_number": 41965,
    "mode": "tests",
    "exit_code": 1,
    "summary": [
      "2 failed, 3 passed, 19 deselected, 14 warnings in 2.13s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/quantization/test_compressed_tensors.py -k 'find_matched_target_returns_none or get_scheme_dict_returns_none or lm_head'",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  }
]
```

**Synthetic test idea:** Not required for the demonstrated stock CPU regression; retain equivalent behavioral coverage.

### Rubric

- Unmatched layers return a no-match sentinel rather than raising
- Scheme resolution yields no scheme for unmatched layers so they load unquantized
- Matched/fused/regex target behavior unchanged
- Type hints updated consistently for optional result; callers handle no-match

### Freeze-base suitability

**Applies cleanly:** True. No overlap with 98661fe01..c1819ca28^. At base: git apply --check OK; cherry-pick --no-commit OK, 0 conflicts.

Earlier file overlaps: `[]`.

**Why high signal:** Small, well-scoped config-semantics change with direct unit tests that flip from ValueError to pass, exercising real quantization-config code on CPU.

## 25. vllm-41916 — Fix spec decode benchmark metrics

**Set:** held_out; **area:** benchmarks / spec decode metrics; **difficulty:** easy.

**Ground truth:** [06a60d3dd08b689f381ffd6384b8123371b2f250](https://github.com/COG-GTM/vllm/commit/06a60d3dd08b689f381ffd6384b8123371b2f250); **parent:** `2a16ece2d342c0c154a4949ad317b521f8c04ec4`.

**Files:** 1; **lines:** +23/−19.

- `vllm/benchmarks/serve.py`

### Agent task statement

`vllm bench serve` can report speculative-decoding acceptance statistics by scraping the server's Prometheus /metrics endpoint before and after the run. The reported numbers are wildly wrong: draft counts, draft-token counts, accepted-token counts and per-position acceptance come out around 1.7 billion. The Prometheus exposition for each counter includes both the `<name>_total` sample and a `<name>_created` sample whose value is a Unix timestamp; the scraper matches lines by metric-name prefix and sums both, so timestamps are added to the counts. Labelled samples (e.g. per-position counters with `{position="0"}`) must be handled too. Repro: serve a /metrics page containing `vllm:spec_decode_num_drafts_total 100` and `vllm:spec_decode_num_drafts_created 1.7769e9` (and similar for draft tokens, accepted tokens, and per-position accepted tokens) and call the benchmark's spec-decode metric fetch. Expected: only counter values are aggregated (num_drafts=100, etc., per-position {0: 80, 1: 70}); `_created` samples and other non-counter samples are ignored; summing across label sets still works. Constraints: no new dependencies; same return type.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** False; **level:** `real_module_cpu_probe_with_mocks`.

**Ground-truth test files:** None in the commit; synthetic evidence only.

**Recorded test IDs:**

- No CPU-discriminating stock IDs established; see the synthetic probe and next-stage test idea.

Audit probe IDs (not upstream test names):

- Standalone Python probe, not a pytest; `probe_command` is authoritative.

**Pytest / probe command and status:**

Executed audit probe; not a stock test-suite pass. Build a behavior-oriented hidden pytest from synthetic_test_idea before scoring agent PRs.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" .venv/bin/python audit-probes/specbench_probe.py
```

**Observed ground truth:** RESULT SpecDecodeMetrics(num_drafts=100, num_draft_tokens=300, num_accepted_tokens=150, accepted_per_pos={0: 80, 1: 70}) PASS

**Observed parent:** RESULT SpecDecodeMetrics(num_drafts=1776924494, num_draft_tokens=1776924694, num_accepted_tokens=1776924544, accepted_per_pos={0: 1700000080, 1: 1700000070}) FAIL

Commit adds no tests. Probe calls the real vllm.benchmarks.serve.fetch_spec_decode_metrics over HTTP against a local aiohttp server serving realistic Prometheus text; discriminating. Hidden test must be authored from this probe (pytest-asyncio + aiohttp test server).

**Synthetic test idea:** pytest-asyncio test starting aiohttp web app serving /metrics with *_total and*_created lines (incl. labelled per-position counters); assert fetch_spec_decode_metrics(base_url, session) returns counts from *_total only.

### Rubric

- Ignores *_created (and other non-_total) samples
- Parses metric name before label braces
- Per-position labelled counters aggregated correctly
- Return structure unchanged
- No new dependency

### Freeze-base suitability

**Applies cleanly:** True. No earlier commits in range touch vllm/benchmarks/serve.py; git apply --check at base: CLEAN.

Earlier file overlaps: `[]`.

**Why high signal:** Self-contained text-parsing bug with dramatic, easily asserted wrong output; real function exercised on CPU.

## 26. vllm-42188 — [Bugfix] Gemma 4 chat template crash with missing tool name and tool id

**Set:** held_out; **area:** tool calling / chat templates; **difficulty:** easy.

**Ground truth:** [b1687527b83649771a543baef1f33aa07b219855](https://github.com/COG-GTM/vllm/commit/b1687527b83649771a543baef1f33aa07b219855); **parent:** `171019ab192362c2c9f4ce9897f0a35ddcaff506`.

**Files:** 1; **lines:** +2/−2.

- `examples/tool_chat_template_gemma4.jinja`

### Agent task statement

Rendering a conversation with the Gemma 4 tool-calling chat template shipped in the examples crashes with `TypeError: can only concatenate str (not "NoneType") to str` when a tool-result message cannot be associated with a tool name. This happens with OpenAI-style `role: "tool"` messages that have no `tool_call_id`, or whose `tool_call_id` does not match any id in the preceding assistant message's `tool_calls`, and with legacy `tool_responses` entries whose `name` is missing or null. Clients that omit tool ids (several popular agent frameworks do) therefore get a 500 from `/v1/chat/completions`.

Expected: the template must never crash on a missing/unresolvable tool name; it should fall back to a placeholder name (`unknown`) for the tool-response block and still render the response content. When the id does match an earlier tool call, the real function name must still be used exactly as before. Rendering must work in a sandboxed Jinja2 environment as used by vLLM.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** False; **level:** `real_module_cpu_probe_with_mocks`.

**Ground-truth test files:** `tests/renderers/test_gemma4_chat_template.py`

**Recorded test IDs:**

- No CPU-discriminating stock IDs established; see the synthetic probe and next-stage test idea.

Audit probe IDs (not upstream test names):

- `audit-probes/test_synth_gemma4_tool_name.py::test_openai_tool_message_without_resolvable_name`
- `audit-probes/test_synth_gemma4_tool_name.py::test_legacy_tool_responses_with_none_name`
- `audit-probes/test_synth_gemma4_tool_name.py::test_matching_id_still_resolves_name`

**Pytest / probe command and status:**

Executed audit probe; not a stock test-suite pass. Build a behavior-oriented hidden pytest from synthetic_test_idea before scoring agent PRs.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest audit-probes/test_synth_gemma4_tool_name.py --confcutdir=audit-probes -q tests/renderers/test_gemma4_chat_template.py
```

**Observed ground truth:** SYNTHETIC + existing stock Gemma4 template tests: 19 passed, 14 warnings in 6.01s

**Observed parent:** parent: 4 failed, 15 passed — `TypeError: can only concatenate str (not "NoneType") to str` on the missing-name cases

Commit adds no tests. Synthetic probe renders the real examples/tool_chat_template_gemma4.jinja with jinja2 ImmutableSandboxedEnvironment. Base matrix (synthetic only): fix 5 passed / tests-only 4 failed, 1 passed. Logs: /home/ubuntu/audit/logs/b1687527b-synth-{gt,par}.log

**Synthetic test idea:** Author a behavior-oriented CPU regression from the executed probe and evidence below, mocking device/model construction but exercising the real changed callable.

### Rubric

- No crash for tool messages without/with unmatched tool_call_id.
- No crash for legacy tool_responses with None name.
- Fallback name 'unknown' used; content still rendered.
- Matching ids still render the real tool name.

### Freeze-base suitability

**Applies cleanly:** True. Mechanical: base-matrix script /home/ubuntu/audit/basematrix.sh created detached worktree at 98661fe01 and ran `git -C /home/ubuntu/repos/vllm diff b1687527b^ b1687527b | git apply` -> applied with no errors. No overlapping files in 98661fe01..b1687527b^.

Earlier file overlaps: `[]`.

**Why high signal:** Concrete 500-error crash in tool-calling rendering with deterministic CPU repro; Jinja rather than Python, so diversity pick at easy tier.

## 27. vllm-40891 — [Core] Avoid using extra thread in `UniProcExecutor`

**Set:** held_out; **area:** core/executor; **difficulty:** medium.

**Ground truth:** [10ebb40d62e024116c1a4473f8e357a3e72761ed](https://github.com/COG-GTM/vllm/commit/10ebb40d62e024116c1a4473f8e357a3e72761ed); **parent:** `54f548e9e58087f0155e4e164e416ad7efdfde6d`.

**Files:** 1; **lines:** +21/−16.

- `vllm/v1/executor/uniproc_executor.py`

### Agent task statement

With async scheduling enabled, the single-process (`uni`) executor resolves asynchronous model-runner outputs on a dedicated background worker thread: each `execute_model` returns a Future that a thread pool completes by blocking on the async output. This adds a thread hop and context switching on every step and measurably hurts latency, especially at low concurrency and high request rates on a single GPU.

Change the executor so no extra thread is used: when the model runner returns an async output, the executor should return a Future-compatible object whose result is resolved lazily, on the thread that calls `result()`, by fetching the async output at that time (not eagerly at submission). The returned object must behave like a normal `concurrent.futures.Future` for callers: `result()` returns the same shape as before (a single value for single-value calls, otherwise a one-element list), repeated `result()` calls return the same value, and exceptions raised while fetching the output propagate from `result()`. Non-async outputs must behave as before. No background thread pool should be created by the executor.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** False; **level:** `real_module_cpu_probe_with_mocks`.

**Ground-truth test files:** None in the commit; synthetic evidence only.

**Recorded test IDs:**

- No CPU-discriminating stock IDs established; see the synthetic probe and next-stage test idea.

Audit probe IDs (not upstream test names):

- `audit-probes/test_probe_uniproc_async.py::test_async_output_resolved_lazily_on_caller_thread`
- `audit-probes/test_probe_uniproc_async.py::test_async_output_exception_propagates`

**Pytest / probe command and status:**

Executed audit probe; not a stock test-suite pass. Build a behavior-oriented hidden pytest from synthetic_test_idea before scoring agent PRs.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest audit-probes/test_probe_uniproc_async.py --confcutdir=audit-probes -q
```

**Observed ground truth:** SYNTHETIC pytest at commit: 5 passed

**Observed parent:** SYNTHETIC pytest at parent: 4 failed, 1 passed (output resolved on 'WorkerAsyncOutput_0' thread / eagerly, not on MainThread)

Upstream commit adds no tests. Probe drives real UniProcExecutor future-wrapping code with a fake AsyncModelRunnerOutput (records calling thread) - not a reimplementation. No GPU model run. Logs: /home/ubuntu/wt/logs/tests/uniproc_probe_{parent,commit}.log

**Synthetic test idea:** Hidden test: construct a fake AsyncModelRunnerOutput whose get_output() records threading.current_thread().name and call count; pass through the executor's async-output future path; assert get_output not called before result(), called on MainThread, single_value vs list shape, exception propagation, and that no ThreadPoolExecutor attribute/thread is created.

### Rubric

- No worker thread pool created for async outputs
- Lazy resolution on caller thread at result() time
- Future semantics preserved (shape, idempotent result, exceptions)
- Non-async path unchanged

### Freeze-base suitability

**Applies cleanly:** True. No overlap with 98661fe01..10ebb40d6^. At base: git apply --check OK; cherry-pick --no-commit OK, 0 conflicts.

Earlier file overlaps: `[]`.

**Why high signal:** Core concurrency design change whose correctness (laziness, thread, Future semantics) is precisely checkable on CPU with a discriminating probe.

## 28. vllm-40708 — [BugFix] Fix Gemma4 'layers.0.moe.experts.0.down_proj_packed' KeyError issue

**Set:** held_out; **area:** model loading; **difficulty:** medium.

**Ground truth:** [25abddc1a5cb5dbe51d629aada9360579acb8acf](https://github.com/COG-GTM/vllm/commit/25abddc1a5cb5dbe51d629aada9360579acb8acf); **parent:** `171d59ae8d1b8d6718ec3cd6558334ce8bbc0eb4`.

**Files:** 2; **lines:** +55/−18.

- `tests/models/quantization/test_awq.py`
- `vllm/model_executor/models/gemma4.py`

### Agent task statement

Some quantized Gemma 4 MoE checkpoints fail during startup with a KeyError such as layers.0.moe.experts.0.down_proj_packed. Reports include compressed-tensors four-bit variants of Gemma 4 26B and AWQ variants. Expert tensors may use a dot before a quantization suffix, such as experts.0.down_proj.qweight, or an underscore, such as experts.0.down_proj_packed and gate_proj_scale. Please make loading accept both supported checkpoint naming conventions and produce the intended expert weights and associated quantization metadata. Cover gate, up, and down projections, multiple experts, and the already-working naming convention so it does not regress. Unquantized checkpoints and unrelated models should be unaffected. A small fake parameter tree and recording weight loaders should demonstrate correct parameter, shard, and expert selection on CPU without downloading a full checkpoint.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** False; **level:** `real_module_cpu_probe_with_mocks`.

**Ground-truth test files:** `tests/models/quantization/test_awq.py`

**Recorded test IDs:**

- `tests/models/quantization/test_awq.py::test_awq_load`

Audit probe IDs (not upstream test names):

- `audit-probes/test_probe_gemma4_load.py::test_compressed_tensors_underscore_suffix_expert_weights`
- `audit-probes/test_probe_gemma4_load.py::test_dot_suffix_awq_expert_weights_still_load`

**Pytest / probe command and status:**

Executed audit probe; not a stock test-suite pass. Build a behavior-oriented hidden pytest from synthetic_test_idea before scoring agent PRs.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest audit-probes/test_probe_gemma4_load.py --confcutdir=audit-probes -q
```

**Observed ground truth:** synthetic pytest (executed): 2 passed

**Observed parent:** synthetic pytest (executed): 1 failed, 1 passed; underscore-suffix test fails with KeyError: 'layers.0.moe.experts.0.down_proj_packed' at gemma4.py:1487 (same symptom as issue #40591); dot-suffix test passes at parent (regression guard)

Harness: detached worktrees /home/ubuntu/wt/c_<sha> (commit) and /home/ubuntu/wt/p_<sha> (parent + commit's test files copied verbatim, byte-identical, no impl). Python: /home/ubuntu/wt/base/.venv/bin/python (uv, py3.12, VLLM_USE_PRECOMPILED=1 editable install, torch 2.11.0+cpu), run from worktree root with PYTHONPATH=$PWD so the worktree source is imported. Synthetic harness calls the REAL Gemma4Model.load_weights (unbound) with a fake nn.Module tree exposing fused expert params (w13_weight_packed, w2_weight_packed, w13_weight_scale, w2_weight_scale, w2_qweight, w13_scales) whose weight_loader records (param, name, shard_id, expert_id). Stock test_awq_load requires CUDA and two HF checkpoints; not run.  Import mode: normal package imports of the worktree (no AST extraction); stubs only for collaborators (see SUMMARY.md). BASE-ONLY: base+patch 2 passed; base(no impl) 1 failed (KeyError)/1 passed; stock test_awq_load collects 2 tests on both (not run: GPU+HF).

**Synthetic test idea:** Implemented: see probe; extend with zero-point (_zero_point / .qzeros) names for both styles.

### Rubric

- Underscore-suffix (compressed-tensors) expert names load into fused params with correct shard_id and expert_id
- Dot-suffix (AWQ) expert names still load
- No KeyError fallthrough to params_dict for expert weights
- Unquantized checkpoints unaffected
- Reuses existing fused-MoE expert mapping helper rather than hard-coding names (quality)

### Freeze-base suitability

**Applies cleanly:** True. Mechanical: `git apply --check` of full commit patch on detached worktree at base 98661fe01 -> OK; `git cherry-pick -n` -> OK (no conflicts). Overlapping earlier commit on vllm/model_executor/models/gemma4.py: 90f145aaf (#40588, hidden_act variants); patch still applies/cherry-picks cleanly at base. Independent `git apply` of only this patch on pristine base: OK; tests green on base+patch, no missing-API/import errors from earlier-window commits.

Earlier file overlaps: `[{"file": "vllm/model_executor/models/gemma4.py", "earlier_commits": ["90f145aaf724194ccffeb3ea6a68e9457ff00169"]}]`.

**Why high signal:** Real user-reported startup crash (two linked issues) whose discriminating loader logic is testable on CPU with the real load_weights.

## 29. vllm-41237 — [Bugfix][KV Transfer] Reject NixlConnector + expandable_segments:True

**Set:** held_out; **area:** config/KV transfer; **difficulty:** easy.

**Ground truth:** [60851b1d22749de50da36287301fe994118941c2](https://github.com/COG-GTM/vllm/commit/60851b1d22749de50da36287301fe994118941c2); **parent:** `8bcd8a260cd19cc534f1773ba241c3a718764452`.

**Files:** 2; **lines:** +98/−0.

- `tests/v1/kv_connector/unit/test_config.py`
- `vllm/config/vllm.py`

### Agent task statement

Running disaggregated prefill/decode with a KV connector (e.g. NixlConnector, MooncakeConnector) while `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set silently corrupts inter-node KV transfers. The connector registers KV-cache memory with the NIC once at startup, but PyTorch's expandable-segments (CUDA VMM) allocator may later remap those virtual addresses to different physical pages, leaving the registration stale. The failure only appears on the first request with opaque errors such as `IBV_WC_REM_ACCESS_ERR`, `remote agent ... was invalidated`, or `NIXL_ERR_REMOTE_DISCONNECT`, making it very hard to diagnose. vLLM should detect this incompatible combination up front during configuration validation and fail fast with a clear, actionable error mentioning expandable_segments. The check should apply to any configured KV connector (including out-of-tree ones), but must not fire when no KV connector is configured, when the allocator config contains other unrelated options, or when sleep mode is enabled (sleep mode routes KV allocations through vLLM's own pool where expandable segments are disabled).

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** True; **level:** `stock_pytest`.

**Ground-truth test files:** `tests/v1/kv_connector/unit/test_config.py`

**Recorded test IDs:**

- `tests/v1/kv_connector/unit/test_config.py::test_kv_connector_rejects_expandable_segments`
- `tests/v1/kv_connector/unit/test_config.py::test_kv_connector_allows_expandable_segments_with_sleep_mode`
- `tests/v1/kv_connector/unit/test_config.py::test_kv_connector_allows_other_alloc_conf`
- `tests/v1/kv_connector/unit/test_config.py::test_no_kv_connector_ignores_expandable_segments`

**Pytest / probe command and status:**

Executed stock pytest; normalized worktree and venv locations. Parent/base runs overlay only the ground-truth regression tests, never implementation.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest -q -p no:cacheprovider tests/v1/kv_connector/unit/test_config.py -k 'expandable_segments or alloc_conf'
```

**Observed ground truth:** stock pytest (executed): 6 passed, 6 deselected

**Observed parent:** stock pytest (executed, tests copied): 6 failed, 6 deselected; all fail with AttributeError at test_config.py:83 (`_verify_kv_transfer_compat` missing on VllmConfig)

Harness: detached worktrees /home/ubuntu/wt/c_<sha> (commit) and /home/ubuntu/wt/p_<sha> (parent + commit's test files copied verbatim, byte-identical, no impl). Python: /home/ubuntu/wt/base/.venv/bin/python (uv, py3.12, VLLM_USE_PRECOMPILED=1 editable install, torch 2.11.0+cpu), run from worktree root with PYTHONPATH=$PWD so the worktree source is imported. Logs: <bundle>/log_60851b1d2.txt. Caveat: the 3 'allows'/'ignores' tests fail at parent only because the private hook name is missing (not behavior), so true fail-to-pass behavior signal is the 3 rejection cases; the hidden tests are coupled to the private method name. Other 6 tests in the file (deselected) were not run here. Import mode: normal package imports of the worktree (no AST extraction); stubs only for collaborators (see SUMMARY.md). BASE-ONLY: base+patch -k subset 6 passed, whole file 12 passed; base+tests-only 6 failed (whole file 6 failed/6 passed).

**Coordinator freeze-base rerun:**

```json
[
  {
    "pr_number": 41237,
    "mode": "fix",
    "exit_code": 0,
    "summary": [
      "6 passed, 6 deselected in 1.43s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/v1/kv_connector/unit/test_config.py -k 'expandable_segments or other_alloc_conf'",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  },
  {
    "pr_number": 41237,
    "mode": "tests",
    "exit_code": 1,
    "summary": [
      "6 failed, 6 deselected in 0.85s"
    ],
    "command": "/home/ubuntu/repos/vllm/.venv/bin/python -m pytest -q -p no:cacheprovider tests/v1/kv_connector/unit/test_config.py -k 'expandable_segments or other_alloc_conf'",
    "environment": "From the indicated freeze-base worktree: PYTHONPATH=$PWD VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=1; shared uv Python."
  }
]
```

**Synthetic test idea:** Not required for the demonstrated stock CPU regression; retain equivalent behavioral coverage.

### Rubric

- Raises ValueError for any configured kv_connector + expandable_segments:True
- Exempts enable_sleep_mode, unrelated PYTORCH_CUDA_ALLOC_CONF values, and no-connector configs
- Error message is actionable (mentions expandable_segments and remedies)
- Validation runs during VllmConfig post-init (fail fast, not at first request)
- Grader: hidden tests invoke VllmConfig._verify_kv_transfer_compat; expose that hook name in the task interface or adapt tests

### Freeze-base suitability

**Applies cleanly:** True. Mechanical: `git apply --check` of full commit patch on detached worktree at base 98661fe01 -> OK; `git cherry-pick -n` -> OK (no conflicts). Overlapping earlier commits on vllm/config/vllm.py: 8189a1591 (#39917) and 7a08b34fb (#35520); patch still applies and cherry-picks cleanly at base. Test uses only base-present APIs (KVTransferConfig, VllmConfig). Independent `git apply` of only this patch on pristine base: OK; tests green on base+patch, no missing-API/import errors from earlier-window commits.

Earlier file overlaps: `[{"file": "vllm/config/vllm.py", "earlier_commits": ["8189a15914ca48461acf106f126c58ef7e41c9ee", "7a08b34fbf17a491ccf1f7e2cff3b8184a9e4454"]}]`.

**Why high signal:** Fail-fast config validation for a silent data-corruption failure mode, with deterministic CPU-only tests driven by env vars.

## 30. vllm-41905 — [SpecDecoding] extend mtp support for mimo 2.5

**Set:** held_out; **area:** spec_decode/models; **difficulty:** easy.

**Ground truth:** [2ee8c2a56e41fbd00b4fb52f29464fb7fca48dba](https://github.com/COG-GTM/vllm/commit/2ee8c2a56e41fbd00b4fb52f29464fb7fca48dba); **parent:** `cd74911d92841102c1cbbc8ed6d544518bf68223`.

**Files:** 1; **lines:** +3/−10.

- `vllm/model_executor/models/mimo_v2_mtp.py`

### Agent task statement

MiMo-V2.5 ships multi-token-prediction (MTP) layers, but vLLM refuses to use more than one speculative token with it. Launching e.g. `vllm serve XiaomiMiMo/MiMo-V2.5 --speculative_config '{"method":"mtp","num_speculative_tokens":3}'` fails during model construction with an error saying MiMo-V2 MTP only supports `num_speculative_tokens=1`, and even if construction is bypassed, the draft model's forward and logits paths reject any speculative step index other than 0. Other MTP models in vLLM support chaining several speculative steps by reusing their available MTP layer(s) for subsequent steps. Please enable `num_speculative_tokens > 1` for MiMo-V2 MTP: construction should succeed for any positive number of speculative tokens, and each speculative step should run through an MTP layer (cycling through the available layers when the step index exceeds the number of MTP layers) for both hidden-state computation and logits, so that acceptance-rate-preserving multi-step drafting works.

### Hidden verification (oracle only)

**CPU verified:** True; **stock CPU verified:** False; **level:** `real_module_cpu_probe_with_mocks`.

**Ground-truth test files:** None in the commit; synthetic evidence only.

**Recorded test IDs:**

- No CPU-discriminating stock IDs established; see the synthetic probe and next-stage test idea.

Audit probe IDs (not upstream test names):

- `audit-probes/test_probe_mimo_mtp.py::test_forward_multi_step_reuses_layer`
- `audit-probes/test_probe_mimo_mtp.py::test_compute_logits_multi_step`
- `audit-probes/test_probe_mimo_mtp.py::test_wrapper_forward_and_logits_multi_step`
- `audit-probes/test_probe_mimo_mtp.py::test_init_accepts_num_speculative_tokens_gt_1`

**Pytest / probe command and status:**

Executed audit probe; not a stock test-suite pass. Build a behavior-oriented hidden pytest from synthetic_test_idea before scoring agent PRs.

```bash
PYTHONPATH="$PWD" VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest audit-probes/test_probe_mimo_mtp.py --confcutdir=audit-probes -q
```

**Observed ground truth:** synthetic pytest (executed): 8 passed

**Observed parent:** synthetic pytest (executed): 7 failed, 1 passed (only step 0 passes); failures: AssertionError 'MiMo-V2 MTP only supports one speculative token.' (mimo_v2_mtp.py:258) and ValueError 'MiMo-V2 MTP in vLLM only supports num_speculative_tokens=1.' (mimo_v2_mtp.py:174)

Harness: detached worktrees /home/ubuntu/wt/c_<sha> (commit) and /home/ubuntu/wt/p_<sha> (parent + commit's test files copied verbatim, byte-identical, no impl). Python: /home/ubuntu/wt/base/.venv/bin/python (uv, py3.12, VLLM_USE_PRECOMPILED=1 editable install, torch 2.11.0+cpu), run from worktree root with PYTHONPATH=$PWD so the worktree source is imported. Synthetic harness: calls the REAL unbound methods MiMoV2MultiTokenPredictor.forward/compute_logits, MiMoV2MTP.forward/compute_logits and MiMoV2MultiTokenPredictor.**init** (with monkeypatched submodule construction) using fake self objects with a 1-layer dict of recording layers; asserts layer selection and outputs. Not stock upstream tests (commit added none). First run had a probe bug (positional arg bound to inputs_embeds) which was fixed to keyword spec_step_idx before final run; log <bundle>/log_probe_2ee8c2a56.txt. Import mode: normal package imports of the worktree (no AST extraction); stubs only for collaborators (see SUMMARY.md). BASE-ONLY: base+patch 8 passed; base(no impl) 7 failed/1 passed.

**Synthetic test idea:** Already implemented: drive real MiMo MTP forward/compute_logits with fake layers and spec_step_idx in {1,2,3}; assert no exception and layer index == step % num_layers; instantiate MiMoV2MTP with speculative_config.num_speculative_tokens=3 using monkeypatched heavy submodules.

### Rubric

- Allow more than one configured MiMo-V2 speculative token and nonzero prediction steps.
- Use the intended model prediction sequence consistently for hidden-state and logits paths, including steps beyond the available layer count.
- Preserve one-step behavior, forward arguments and output shape; do not invent extra model parameters.
- Test real routing with lightweight layer doubles; no numerical acceptance-rate or full-checkpoint claim.

### Freeze-base suitability

**Applies cleanly:** True. Mechanical: `git apply --check` of full commit patch on detached worktree at base 98661fe01 -> OK; `git cherry-pick -n` -> OK (no conflicts). No earlier commit in 98661fe01..2ee8c2a56^ touches vllm/model_executor/models/mimo_v2_mtp.py. Independent `git apply` of only this patch on pristine base: OK; tests green on base+patch, no missing-API/import errors from earlier-window commits.

Earlier file overlaps: `[]`.

**Why high signal:** Small spec-decode capability gap with clear discriminating behavior exercised via real model methods on CPU.

## Exclusion ledger

| PR | Commit | Primary reason |
| --- | --- | --- |
| [#41465](https://github.com/vllm-project/vllm/pull/41465) | `6fca51815706fbb0735849b95c0dafa05da20f9f` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#38099](https://github.com/vllm-project/vllm/pull/38099) | `b786ec8e744b0c1a48b58873a14adb4e4f30eec3` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#41729](https://github.com/vllm-project/vllm/pull/41729) | `84bd8a3c1eb7560a810ff4d99a476708a6f9f2f2` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#40815](https://github.com/vllm-project/vllm/pull/40815) | `2228fe68687d98fb796e46e83de770ade483d73a` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#40871](https://github.com/vllm-project/vllm/pull/40871) | `628c43630155b9fa8a8ec8871eb90c06bf8b36e6` | GPU/native/hardware/model-dependent |
| [#39213](https://github.com/vllm-project/vllm/pull/39213) | `48954de2378d19bc0da771930c1d7a202b5e58e0` | GPU/native/hardware/model-dependent |
| [#41571](https://github.com/vllm-project/vllm/pull/41571) | `79246b5ea6e0ede64a7fe18f4538d59d33dd4792` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#39755](https://github.com/vllm-project/vllm/pull/39755) | `8c57b6e7bc461715eb9d74412aa694547b940633` | Dependency/optional-package coupling |
| [#41744](https://github.com/vllm-project/vllm/pull/41744) | `01b9b5af67b77715fd1c33f19598f213ae4fec6a` | Refactors/renames/removals |
| [#40415](https://github.com/vllm-project/vllm/pull/40415) | `13338644082bdab10c4ff017de1a97c03ba62524` | CI/build/repository infrastructure |
| [#41575](https://github.com/vllm-project/vllm/pull/41575) | `4a8ae26e532c4b36acc1dd2f5e7c672d73832b4e` | CI/build/repository infrastructure |
| [#41453](https://github.com/vllm-project/vllm/pull/41453) | `f6537612521df1156d8ac13f524e427aca908322` | CI/build/repository infrastructure |
| [#41658](https://github.com/vllm-project/vllm/pull/41658) | `16e336491e96041946e2cd379a80034fcac439bd` | Dependency/optional-package coupling |
| [#40575](https://github.com/vllm-project/vllm/pull/40575) | `aee190ac37cd98ed164d870763a1fec6041e3cb4` | CI/build/repository infrastructure |
| [#41387](https://github.com/vllm-project/vllm/pull/41387) | `e47c98ef7a38792996e452ef53914e21e41928e9` | GPU/native/hardware/model-dependent |
| [#37243](https://github.com/vllm-project/vllm/pull/37243) | `91740ca5eaedbcdd960c3629206087d20fb43a00` | CI/build/repository infrastructure |
| [#41752](https://github.com/vllm-project/vllm/pull/41752) | `2d7d6cf765d0c796070feaa34fcfcc26494d5c31` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#41186](https://github.com/vllm-project/vllm/pull/41186) | `809b98e5b71482e52ee0ecbf140084aae49f3652` | GPU/native/hardware/model-dependent |
| [#41318](https://github.com/vllm-project/vllm/pull/41318) | `e87e09a50a41db3ddd1cc0403e8d1ebe0b0f522c` | GPU/native/hardware/model-dependent |
| [#40759](https://github.com/vllm-project/vllm/pull/40759) | `51c1ee9b7c8acbba4899a8ebffd390685d171946` | Refactors/renames/removals |
| [#41344](https://github.com/vllm-project/vllm/pull/41344) | `1c58876618dc7275520297cbbd1244a64f551952` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#40686](https://github.com/vllm-project/vllm/pull/40686) | `66d1cc0c77628e36df8e5e245c116e6aac3045fb` | GPU/native/hardware/model-dependent |
| [#41794](https://github.com/vllm-project/vllm/pull/41794) | `e43a7912847ef335337476138e5e863850a4ae0a` | CI/build/repository infrastructure |
| [#40390](https://github.com/vllm-project/vllm/pull/40390) | `2e777d21a82c232ee8bc315c13400cd9d9afd489` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#41734](https://github.com/vllm-project/vllm/pull/41734) | `d8deb5b7ade403ae36feda6f71898a61ac0bda6e` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#40569](https://github.com/vllm-project/vllm/pull/40569) | `5d0fd87038b123a11dd9c85a05ce2d258e27ce7b` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#40731](https://github.com/vllm-project/vllm/pull/40731) | `df8e63f4edabc06e51c10fe479fc9b95c303542f` | Refactors/renames/removals |
| [#40148](https://github.com/vllm-project/vllm/pull/40148) | `6467213a9f118fcf5ebf65d5e1ee8d64d2e101e4` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#41745](https://github.com/vllm-project/vllm/pull/41745) | `27e0057aeda6bc443069c20fdf2f3cc95ed892f3` | GPU/native/hardware/model-dependent |
| [#41800](https://github.com/vllm-project/vllm/pull/41800) | `d5b31c954d511ecdb486e84c71d52d2a5c28ade4` | Non-discriminating CPU evidence |
| [#38296](https://github.com/vllm-project/vllm/pull/38296) | `22a3cbe1520bc8a3b19ace0abe497c514b3129ea` | GPU/native/hardware/model-dependent |
| [#41133](https://github.com/vllm-project/vllm/pull/41133) | `27702f6d0879312c011e0180ffd48cbc034380d2` | Non-discriminating CPU evidence |
| [#39324](https://github.com/vllm-project/vllm/pull/39324) | `38e16678ba7ec8417ce87f7010a03d203ee64b2b` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#41844](https://github.com/vllm-project/vllm/pull/41844) | `ca3e62d3363d3f8f5470b351ebfc48fa1826e687` | Dependency/optional-package coupling |
| [#41857](https://github.com/vllm-project/vllm/pull/41857) | `f3f8efa73a49ec87d6a22fe268d0abde503497d7` | CI/build/repository infrastructure |
| [#41832](https://github.com/vllm-project/vllm/pull/41832) | `deb737e323b3c2bf7986b2225ba76e32b4b097f2` | Documentation only |
| [#40820](https://github.com/vllm-project/vllm/pull/40820) | `50acdc5b5cc00f10408d8f98b21fc97efc615173` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#41665](https://github.com/vllm-project/vllm/pull/41665) | `80d5e7d103ee181f66389d0e2580841e2a2f1db5` | GPU/native/hardware/model-dependent |
| [#41840](https://github.com/vllm-project/vllm/pull/41840) | `7a576e2c724e135236c0dc005dce114c7e4d911e` | CI/build/repository infrastructure |
| [#38920](https://github.com/vllm-project/vllm/pull/38920) | `5a0a8fc1ea7542394ff315138bd5677b7b53bca1` | Documentation only |
| [#40549](https://github.com/vllm-project/vllm/pull/40549) | `20cac26b197a2e9530a787a5c1dbf5163533b2ff` | GPU/native/hardware/model-dependent |
| [#41025](https://github.com/vllm-project/vllm/pull/41025) | `51f22dcfd068fe8f1e3192da2a1e825b930223cf` | GPU/native/hardware/model-dependent |
| [#41314](https://github.com/vllm-project/vllm/pull/41314) | `713b28bd0bf4d7f8226ee20940e3ae171fabd09b` | GPU/native/hardware/model-dependent |
| [#40264](https://github.com/vllm-project/vllm/pull/40264) | `8a4888be21219bf081b5bc7e01f8ecf2af9057a8` | GPU/native/hardware/model-dependent |
| [#41845](https://github.com/vllm-project/vllm/pull/41845) | `6e6d182d18dbe5c81c9c9058a5571241f649ae7b` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#41713](https://github.com/vllm-project/vllm/pull/41713) | `d4b00484040c9a0bbd4ee2d55983df7a50ab1fd3` | GPU/native/hardware/model-dependent |
| [#41919](https://github.com/vllm-project/vllm/pull/41919) | `b20731d0aea48c426441880b0ebe28c5d0cefece` | CI/build/repository infrastructure |
| [#41889](https://github.com/vllm-project/vllm/pull/41889) | `9c0812ffd0ea66cd50109019b289e39194e17a16` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#41901](https://github.com/vllm-project/vllm/pull/41901) | `ffee74162609612c3aae3113b8fa3cbca5eeb1b9` | Refactors/renames/removals |
| [#41924](https://github.com/vllm-project/vllm/pull/41924) | `b3945cc3160119111927816841950715c5410708` | GPU/native/hardware/model-dependent |
| [#41837](https://github.com/vllm-project/vllm/pull/41837) | `f650ace6de5acdf0616c70282b0b7f426bb52a61` | Base/dependency incompatibility |
| [#41755](https://github.com/vllm-project/vllm/pull/41755) | `75f0d516c43cc664bec51409e0f0ddccef5d53e6` | GPU/native/hardware/model-dependent |
| [#38206](https://github.com/vllm-project/vllm/pull/38206) | `805e9f7b777b6bf39e18674f3db7c950bcbd9db1` | GPU/native/hardware/model-dependent |
| [#41808](https://github.com/vllm-project/vllm/pull/41808) | `2a84da3b17a9d0b6b10f7c0347fe9529307596ec` | GPU/native/hardware/model-dependent |
| [#41534](https://github.com/vllm-project/vllm/pull/41534) | `003159d98b2545872a215a2648f2ee8d1a4b498d` | GPU/native/hardware/model-dependent |
| [#41741](https://github.com/vllm-project/vllm/pull/41741) | `2a16ece2d342c0c154a4949ad317b521f8c04ec4` | Dependency/optional-package coupling |
| [#35520](https://github.com/vllm-project/vllm/pull/35520) | `7a08b34fbf17a491ccf1f7e2cff3b8184a9e4454` | Refactors/renames/removals |
| [#41805](https://github.com/vllm-project/vllm/pull/41805) | `9d6500b89d12f4d62ef9f2ab7292823b4255217c` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#41876](https://github.com/vllm-project/vllm/pull/41876) | `8eb401134e750781a202c0b6dc4059616cdb4954` | Refactors/renames/removals |
| [#39917](https://github.com/vllm-project/vllm/pull/39917) | `8189a15914ca48461acf106f126c58ef7e41c9ee` | Refactors/renames/removals |
| [#41835](https://github.com/vllm-project/vllm/pull/41835) | `c936548ce6b0f6ef2f71968315c13213898ecd31` | GPU/native/hardware/model-dependent |
| [#41972](https://github.com/vllm-project/vllm/pull/41972) | `3af561ec0a00aabd407cf782a60f17e86c1a253f` | GPU/native/hardware/model-dependent |
| [#41880](https://github.com/vllm-project/vllm/pull/41880) | `969fbfb4a985ad763a6109851bad9dfc94fea289` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#41646](https://github.com/vllm-project/vllm/pull/41646) | `54f548e9e58087f0155e4e164e416ad7efdfde6d` | GPU/native/hardware/model-dependent |
| [#41770](https://github.com/vllm-project/vllm/pull/41770) | `09a7cc5ba94c5c1b1c908a29f28ed41c793f1e9f` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#41953](https://github.com/vllm-project/vllm/pull/41953) | `5f6a02812ae127679d407ac44852baaeac7822a6` | CI/build/repository infrastructure |
| [#41940](https://github.com/vllm-project/vllm/pull/41940) | `57c2f724c1215a2c27382248e017d74f3ec97bb1` | CI/build/repository infrastructure |
| [#41084](https://github.com/vllm-project/vllm/pull/41084) | `1d694e78c954027a42c909cd3488274e4c5ccd9c` | Refactors/renames/removals |
| [#41434](https://github.com/vllm-project/vllm/pull/41434) | `989c176c0a14e1adc5a9ba33cb5c3a39fceec3d3` | GPU/native/hardware/model-dependent |
| [#41486](https://github.com/vllm-project/vllm/pull/41486) | `01b0f3adab12f883bf8a4f2460134fa8839b54be` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#41990](https://github.com/vllm-project/vllm/pull/41990) | `baf068d8be09b22db636ae8c6a2082033e2b6911` | GPU/native/hardware/model-dependent |
| [#40850](https://github.com/vllm-project/vllm/pull/40850) | `0b999713522099cfdf0de57d42972369dddff053` | Refactors/renames/removals |
| [#41895](https://github.com/vllm-project/vllm/pull/41895) | `1acd67a795ebccdf9b9db7697ae9082058301657` | CI/build/repository infrastructure |
| [#40711](https://github.com/vllm-project/vllm/pull/40711) | `ed582b6a4cae6f58c952768716a45b92a8ac1f3f` | GPU/native/hardware/model-dependent |
| [#41082](https://github.com/vllm-project/vllm/pull/41082) | `77b13b9602d32e91d2df71a633e4e3f3c01632ae` | Documentation only |
| [#41984](https://github.com/vllm-project/vllm/pull/41984) | `445d747434542741340e62ff8f1c3a1b2b3d71c2` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#41634](https://github.com/vllm-project/vllm/pull/41634) | `f9b9bf3bbb36d84336bc88e077ee98b39246ee9a` | CI/build/repository infrastructure |
| [#41943](https://github.com/vllm-project/vllm/pull/41943) | `160858cba47aa5999b57ef9445b9dfa946a09458` | CI/build/repository infrastructure |
| [#42039](https://github.com/vllm-project/vllm/pull/42039) | `36b2c79d4b234481f1a4b0ff29155bb21f369fcd` | CI/build/repository infrastructure |
| [#42010](https://github.com/vllm-project/vllm/pull/42010) | `19df11f5d1610e51915d89034e169acb3dfeb7ea` | CI/build/repository infrastructure |
| [#41926](https://github.com/vllm-project/vllm/pull/41926) | `630820a59b94eb31e2797349085b5a64113660ce` | Dependency/optional-package coupling |
| [#42042](https://github.com/vllm-project/vllm/pull/42042) | `52458b60a88310a7f730f067e5a431a55cd34afd` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#42013](https://github.com/vllm-project/vllm/pull/42013) | `f2bbd575e2be8d901f65e304fcedd05ca7afa31a` | CI/build/repository infrastructure |
| [#42066](https://github.com/vllm-project/vllm/pull/42066) | `4140faa4a51d42cb9618949bee28fd47682f611c` | Documentation only |
| [#40588](https://github.com/vllm-project/vllm/pull/40588) | `90f145aaf724194ccffeb3ea6a68e9457ff00169` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#41792](https://github.com/vllm-project/vllm/pull/41792) | `44e6b44a213ca52706c208c639da3fe20d867728` | GPU/native/hardware/model-dependent |
| [#39280](https://github.com/vllm-project/vllm/pull/39280) | `2c6b59b80771ac3bfa1c789abe3cb27c379bf3a1` | GPU/native/hardware/model-dependent |
| [#42057](https://github.com/vllm-project/vllm/pull/42057) | `f47ccc8b1c211e2639cee07baa2825f262577b14` | CI/build/repository infrastructure |
| [#42054](https://github.com/vllm-project/vllm/pull/42054) | `f4dd5c116cba84f5bba80a4e84908fa8766f5026` | CI/build/repository infrastructure |
| [#42055](https://github.com/vllm-project/vllm/pull/42055) | `d2f22dfc9f3551a19988101d742d08b0b925d813` | CI/build/repository infrastructure |
| [#42059](https://github.com/vllm-project/vllm/pull/42059) | `0c2e9d48927f8122cba25f9a649a72f131faffb3` | CI/build/repository infrastructure |
| [#41802](https://github.com/vllm-project/vllm/pull/41802) | `6881c754e1ce4b154ad48e8532d4680e3a54096c` | GPU/native/hardware/model-dependent |
| [#41868](https://github.com/vllm-project/vllm/pull/41868) | `8a2fc80b848af8f1e902d4b2b14afe378d9d4d89` | GPU/native/hardware/model-dependent |
| [#42089](https://github.com/vllm-project/vllm/pull/42089) | `8bcd8a260cd19cc534f1773ba241c3a718764452` | GPU/native/hardware/model-dependent |
| [#42053](https://github.com/vllm-project/vllm/pull/42053) | `30f519e947d4dbea0bceef002f2f567faa8c8457` | CI/build/repository infrastructure |
| [#40356](https://github.com/vllm-project/vllm/pull/40356) | `be0dcc29dcfa83659e6857dd73cb527b5986a7f9` | GPU/native/hardware/model-dependent |
| [#42121](https://github.com/vllm-project/vllm/pull/42121) | `b1728c1e660d35f0443605c779cf0643e7737cec` | Refactors/renames/removals |
| [#42073](https://github.com/vllm-project/vllm/pull/42073) | `236bf9d152136f3f199f5bc23d283551c8604576` | Documentation only |
| [#42077](https://github.com/vllm-project/vllm/pull/42077) | `a43bc34baf2a2e367cdbbdaec44de8cf5de92d1a` | Documentation only |
| [#41911](https://github.com/vllm-project/vllm/pull/41911) | `4f6fa6341da0789714b240c7d6baf53a5106f5e2` | Documentation only |
| [#42038](https://github.com/vllm-project/vllm/pull/42038) | `845ca327ce140d71f2829c95aee49c39f8723269` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#42037](https://github.com/vllm-project/vllm/pull/42037) | `e934e459e6bfdff0de6bcf255013a499434c3606` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#42130](https://github.com/vllm-project/vllm/pull/42130) | `97cc7685c4866cce51aeb59b1fff5466ca6f7c2a` | CI/build/repository infrastructure |
| [#42040](https://github.com/vllm-project/vllm/pull/42040) | `df2636a9d8139dd1d53ec1f1e666abe66fb76a3c` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#42061](https://github.com/vllm-project/vllm/pull/42061) | `e8f9038ebd2b72e8fea24642ca8b4c3500264145` | GPU/native/hardware/model-dependent |
| [#42078](https://github.com/vllm-project/vllm/pull/42078) | `a2812becd6952ab66b1ee54856bb855a0551353b` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#40380](https://github.com/vllm-project/vllm/pull/40380) | `d6563d693c817ec762ccd0bb6c44c56b4f3dc4a3` | CI/build/repository infrastructure |
| [#40867](https://github.com/vllm-project/vllm/pull/40867) | `ecd0b60aad2f4e28dd00ababfc1402690d88cbed` | GPU/native/hardware/model-dependent |
| [#42113](https://github.com/vllm-project/vllm/pull/42113) | `34ab4f2565572a1525865a06d2ba19cc5fcb4f00` | Dependency/optional-package coupling |
| [#41428](https://github.com/vllm-project/vllm/pull/41428) | `530d37130278d38b07b089f936850537aa1ea5e6` | GPU/native/hardware/model-dependent |
| [#42076](https://github.com/vllm-project/vllm/pull/42076) | `adb6d96516cb0cd6d1dec985cb5e5cdc20ed418b` | GPU/native/hardware/model-dependent |
| [#42070](https://github.com/vllm-project/vllm/pull/42070) | `3dda9aeb54cc15c9ecc6d1498a42ceb372d4472b` | GPU/native/hardware/model-dependent |
| [#41957](https://github.com/vllm-project/vllm/pull/41957) | `171d59ae8d1b8d6718ec3cd6558334ce8bbc0eb4` | Base/dependency incompatibility |
| [#41706](https://github.com/vllm-project/vllm/pull/41706) | `cd74911d92841102c1cbbc8ed6d544518bf68223` | Refactors/renames/removals |
| [#41769](https://github.com/vllm-project/vllm/pull/41769) | `7a2b59698275f30d6168dbac1f2b3c5c3cd5fe94` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#41392](https://github.com/vllm-project/vllm/pull/41392) | `f80aa53c9dc2273a19a6855092069db7e1306fff` | Refactors/renames/removals |
| [#39832](https://github.com/vllm-project/vllm/pull/39832) | `ea0e501bb18c12b80acc05ff8c7f013db515ba80` | Refactors/renames/removals |
| [#42148](https://github.com/vllm-project/vllm/pull/42148) | `006af4b9562d0b60de0f1b11ab2fb33dea646cde` | GPU/native/hardware/model-dependent |
| [#41882](https://github.com/vllm-project/vllm/pull/41882) | `bc5fdc1e6a718aec4b76b47b69ec5e7cb3c2412a` | GPU/native/hardware/model-dependent |
| [#41846](https://github.com/vllm-project/vllm/pull/41846) | `dcb3135af7f65c82b11e78b180fce2627ede3f7a` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#33322](https://github.com/vllm-project/vllm/pull/33322) | `0b272a6e01ab49274ce392dc1684d52143d5fd8e` | GPU/native/hardware/model-dependent |
| [#42176](https://github.com/vllm-project/vllm/pull/42176) | `1029e5ef282ec4f705de99b91418e956c835c315` | CI/build/repository infrastructure |
| [#39306](https://github.com/vllm-project/vllm/pull/39306) | `00b0618a03916f826e7da384a67ddf72f2302d97` | GPU/native/hardware/model-dependent |
| [#42180](https://github.com/vllm-project/vllm/pull/42180) | `27d3bac272597bf1cbbd070ae0b81c0f68e13788` | Documentation only |
| [#42169](https://github.com/vllm-project/vllm/pull/42169) | `986edc858a13d92aa9bd2731710bd95e9e5c672d` | GPU/native/hardware/model-dependent |
| [#41573](https://github.com/vllm-project/vllm/pull/41573) | `fb1ac806c55a6dc96fe92261b80c8550e9c39d2f` | CI/build/repository infrastructure |
| [#41313](https://github.com/vllm-project/vllm/pull/41313) | `f2840120f632c55a972dc2ec482241db89b6fc0f` | CI/build/repository infrastructure |
| [#42197](https://github.com/vllm-project/vllm/pull/42197) | `efd0e7789d879188a1e64a07b47a8d20fd97129e` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#42200](https://github.com/vllm-project/vllm/pull/42200) | `301305c09311a5103f9b1be2acac00bd898e59b0` | CI/build/repository infrastructure |
| [#42160](https://github.com/vllm-project/vllm/pull/42160) | `a2c9d548d756f300fd5218d95d57414949a36b91` | Documentation only |
| [#42196](https://github.com/vllm-project/vllm/pull/42196) | `84f7a55340601ddc77b850025ea1ca03f6b1fd82` | CI/build/repository infrastructure |
| [#38502](https://github.com/vllm-project/vllm/pull/38502) | `0a309b5ee9480d0d78e7f00946a418ffd7e3baf1` | GPU/native/hardware/model-dependent |
| [#37912](https://github.com/vllm-project/vllm/pull/37912) | `48698b1b9b30071952084ce12e8904b219193545` | GPU/native/hardware/model-dependent |
| [#41366](https://github.com/vllm-project/vllm/pull/41366) | `e175192d33fdeefcd6689c25a1f7dbe1a3f043ef` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#41617](https://github.com/vllm-project/vllm/pull/41617) | `215e2f7990d9bb8788555a49036002e69ce14eaa` | GPU/native/hardware/model-dependent |
| [#41694](https://github.com/vllm-project/vllm/pull/41694) | `f396bee56fb516195a3669e5499129021ced4443` | GPU/native/hardware/model-dependent |
| [#41979](https://github.com/vllm-project/vllm/pull/41979) | `1b57eb41f24166d0bee2031c53847b188089d77d` | Refactors/renames/removals |
| [#41536](https://github.com/vllm-project/vllm/pull/41536) | `171019ab192362c2c9f4ce9897f0a35ddcaff506` | GPU/native/hardware/model-dependent |
| [#41119](https://github.com/vllm-project/vllm/pull/41119) | `7f95e66a11f5e9d30abb8862e8471daf24e590ba` | GPU/native/hardware/model-dependent |
| [#41188](https://github.com/vllm-project/vllm/pull/41188) | `5536fc0c019db9b1f42afcc18fc38f4666b65e23` | Refactors/renames/removals |
| [#42052](https://github.com/vllm-project/vllm/pull/42052) | `581b5e9afc98b2711c920699326f3e15f928eea9` | Base/dependency incompatibility |
| [#42266](https://github.com/vllm-project/vllm/pull/42266) | `05d610e5cdb710bffdccdda630b4eb0f79afd76d` | CI/build/repository infrastructure |
| [#42163](https://github.com/vllm-project/vllm/pull/42163) | `5cba6839e6cb8ede0c81947557b76617468d0d61` | Documentation only |
| [#38895](https://github.com/vllm-project/vllm/pull/38895) | `b1b59720b252b422368deb99aadc888fb397db40` | GPU/native/hardware/model-dependent |
| [#42280](https://github.com/vllm-project/vllm/pull/42280) | `9efdddca283cc0eb7c37fa49c4a9d1c9bf59ec4e` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#41383](https://github.com/vllm-project/vllm/pull/41383) | `770e9bd6b3519e8a6fa5738877974af274ec946e` | Base/dependency incompatibility |
| [#40364](https://github.com/vllm-project/vllm/pull/40364) | `5672d100ed66490b84ea54b030b0785a6d20d111` | Non-discriminating CPU evidence |
| [#42293](https://github.com/vllm-project/vllm/pull/42293) | `17ed5e61f51ffa3bd14a5eb202439fb87892e421` | CI/build/repository infrastructure |
| [#42246](https://github.com/vllm-project/vllm/pull/42246) | `27ae6763647920119dcd50d147ca365fdc138945` | Dependency/optional-package coupling |
| [#42272](https://github.com/vllm-project/vllm/pull/42272) | `617239b70c46d3d53fc16619718da1c347c43fb6` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#41928](https://github.com/vllm-project/vllm/pull/41928) | `8415bf2cdbdd00c1786abb766df568d5ae19e2c0` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#40392](https://github.com/vllm-project/vllm/pull/40392) | `a51376b3f05a2f74eac6ceeed7e52598b871a0fb` | GPU/native/hardware/model-dependent |
| [#42236](https://github.com/vllm-project/vllm/pull/42236) | `724ed2fc352bb388aea7269ca5c6690637c30ce7` | GPU/native/hardware/model-dependent |
| [#41942](https://github.com/vllm-project/vllm/pull/41942) | `5f1b313900fd5375857bdb7f709e1c5aad54443c` | GPU/native/hardware/model-dependent |
| [#41993](https://github.com/vllm-project/vllm/pull/41993) | `4b64fc2cbf2a10c0027f7c80681af0ede0c3799a` | Refactors/renames/removals |
| [#41727](https://github.com/vllm-project/vllm/pull/41727) | `4955990f1bd91e5d6070a56ad3ed9684a23e5146` | Refactors/renames/removals |
| [#42181](https://github.com/vllm-project/vllm/pull/42181) | `a2e776d7168e1a75fe32438aea4254542602c012` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#40408](https://github.com/vllm-project/vllm/pull/40408) | `0d453e23364701f6fe2f20deafbb372b28714a86` | GPU/native/hardware/model-dependent |
| [#41812](https://github.com/vllm-project/vllm/pull/41812) | `7863fff6e591161a03a7df68b0879b9fa61cfc0b` | GPU/native/hardware/model-dependent |
| [#42202](https://github.com/vllm-project/vllm/pull/42202) | `9af6a5ed75bbb095c3e7e78b807ca0fe29ef90bb` | Base/dependency incompatibility |
| [#42190](https://github.com/vllm-project/vllm/pull/42190) | `5497ffbf7c078550c18566cc76ab9bae86ad0496` | Documentation only |
| [#41987](https://github.com/vllm-project/vllm/pull/41987) | `cf0d2791420eefab4a3416551bb3f4a1288f0aea` | Documentation only |
| [#42201](https://github.com/vllm-project/vllm/pull/42201) | `6fdb49392e5183c310c4703b883e7edbf25f6a96` | GPU/native/hardware/model-dependent |
| [#41825](https://github.com/vllm-project/vllm/pull/41825) | `a721315488429e728e25adabb6d94c35a4770b94` | GPU/native/hardware/model-dependent |
| [#41877](https://github.com/vllm-project/vllm/pull/41877) | `639cbfd27428632f2851b46a9d70779e73bccbb3` | CI/build/repository infrastructure |
| [#39846](https://github.com/vllm-project/vllm/pull/39846) | `56e5810ff168bb13695d35fba2c7401256afdda4` | GPU/native/hardware/model-dependent |
| [#42274](https://github.com/vllm-project/vllm/pull/42274) | `a0dc7a0f368a3f212d37e39ba4dd7b682e378fe2` | Refactors/renames/removals |
| [#42287](https://github.com/vllm-project/vllm/pull/42287) | `53181384e036cc660ff6754c8a94a8e6d0b60562` | GPU/native/hardware/model-dependent |
| [#41429](https://github.com/vllm-project/vllm/pull/41429) | `bbee5329880bfe45a8b0cf7753d17501760e62c9` | Low-signal/trivial/test-or-example-only/lower-ranked |
| [#41761](https://github.com/vllm-project/vllm/pull/41761) | `d7af6b34d83fc691ad69347ee4d066231e5678ab` | Low-signal/trivial/test-or-example-only/lower-ranked |
