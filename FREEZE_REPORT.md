# vLLM eval freeze report

## 1. Freeze point
- Freeze commit: `98661fe012c5c467252d4df8411d2f46190e9268` = `main~200` (first-parent) on COG-GTM/vllm. "[Bugfix][KVConnector] Support DCP/PCP in OffloadingConnector (#41549)", 2026-05-05.
- Task pool: the 200 commits `98661fe01..d7af6b34d` (`git rev-list --count` = 200).
- **Branch `eval-base` is pushed** to COG-GTM/vllm. `git ls-remote origin refs/heads/eval-base` → `98661fe012c5c467252d4df8411d2f46190e9268`. `main` was not touched.

## 2. Leak-free mirror `COG-GTM/vllm-eval`: BLOCKED (access)
- First `gh repo create COG-GTM/vllm-eval --private --description ...` exited with no output.
- After that, `gh repo view` / `gh api repos/COG-GTM/vllm-eval` return `Not Found (HTTP 404)`.
- Running create again: `GraphQL: Name already exists on this account (createRepository)`. REST `POST orgs/COG-GTM/repos` gives `HTTP 422 ... "name already exists on this account"`.
- So the repo was most likely created, but this session's GitHub token can't reach it. `gh api installation/repositories` lists only `COG-GTM/vllm` and `COG-GTM/edelmanfinancialenginesios`. This is a per-session scoped token, even though the org's `devin-ai-integration` app installation has `repository_selection: all`.
- Devin git proxy: `git ls-remote https://git-manager.devin.ai/proxy/github.com/COG-GTM/vllm-eval` → `The requested URL returned error: 403`. Retried later in the session and still got 403.
- Per your instruction, nothing was deleted or recreated, and **nothing has been pushed to the mirror yet**.
- Once access exists, run:
  ```bash
  cd vllm && git push https://git-manager.devin.ai/proxy/github.com/COG-GTM/vllm-eval 98661fe012c5c467252d4df8411d2f46190e9268:refs/heads/main
  git clone https://git-manager.devin.ai/proxy/github.com/COG-GTM/vllm-eval /tmp/ve && cd /tmp/ve
  git log -1 --format='%H %cd'        # expect 98661fe01..., 2026-05-05
  git cat-file -e d7af6b34d && echo LEAK || echo OK
  git branch -a; git tag | wc -l      # expect only main, 0 tags
  ```

## 3. Agent-instruction surface at the freeze point
Files found in tree `98661fe01`. Searched for AGENTS.md, CLAUDE.md, SKILL.md, GEMINI.md, .cursorrules, copilot-instructions.md, .claude/, .cursor/, .agents/, .devin/, .gemini/, .codex/, .windsurf, .github/{instructions,prompts,agents,skills}:
| File | Notes |
|---|---|
| `AGENTS.md` | ~110 lines: contribution policy (duplicate checks, no busywork, accountability), uv/.venv workflow, `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto`, test deps `requirements/test/cuda.{in,txt}`, pre-commit/mypy, commit trailers, links to one domain guide |
| `CLAUDE.md` | just `@AGENTS.md` |
| `docs/contributing/editing-agent-instructions.md` | meta-rules: AGENTS.md < 200 lines, domain guides < 300, when NOT to add rules, checklist |
| `.gemini/config.yaml` | Gemini Code Assist review config only (HIGH severity threshold, no summary) |
- There are **no** skills (`.agents/`, `.claude/`, SKILL.md), no `.cursor/`, no `.devin/`, no copilot instructions, and no nested AGENTS.md.
- `git diff 98661fe01 d7af6b34d -- AGENTS.md CLAUDE.md docs/contributing/editing-agent-instructions.md .gemini` is **empty (0 bytes)**. The instructions did not change during the task window.
- Snapshots: `agent_surface_98661fe01/`. Upstream copies: `agent_surface_upstream_main_ce775739e9/` (kept locally, can attach on request).

### Upstream vllm-project/vllm evolution after 2026-05-11 (read-only fetch, upstream main `ce775739e9`, 2026-10-28)
About 50 commits touched agent files. The notable additions below are candidate hill-climbing hypotheses:
1. **Test-writing rules** in AGENTS.md "When adding tests", plus a new `tests/AGENTS.md` (Oct 21, #62601; Jul 7, #47784):
   - Design before writing: module purpose, I/O contract, the failure being guarded, the cheapest level that reproduces it.
   - Reuse existing files and fixtures. Test behavior through public APIs.
   - Use real objects rather than SimpleNamespace/MagicMock.
   - Show the test failing before the fix.
   - "Never buy a green CI": no xfail, skip, or loosened tolerances.
   - Deterministic, exact asserts.
   - Make sure a `.buildkite/test_areas/*.yaml` step actually selects the test.
2. **Coding-style section** (#45301): match existing style, minimal comments, assume the reader knows vLLM. Also an 88-char line limit and Google-style docstrings.
3. **Domain guides for parsers**: `vllm/tool_parsers/AGENTS.md`, `vllm/reasoning/AGENTS.md`, `vllm/parser/AGENTS.md` (Sep 16, #56883).
   - Layout map and "keep adapters thin".
   - Don't change the legacy `ToolParser`/`ReasoningParser` interfaces.
   - Put new models in `vllm/parser/` as a `ParserEngine`. Registration keys are breaking.
   - Very relevant here: tool_parsers and reasoning are CPU-testable areas.
4. **Skills under `.agents/skills/`**, symlinked into `.claude/skills/`:
   - `pr-checklist`: design fit, core-component impact, complexity budget by LOC, correctness tracing across backends. Required before PRs.
   - `check-api-compat`: check the OpenAI/Anthropic spec rather than guessing from vLLM code.
   - `ci-fails-buildkite`, `transformers-upgrade`, `model-recipes`, `model-microbenchmark`.
   - GPU-only: `kernel-microbenchmark`, `triton-kernel-writing`, `debug-ima`.
5. **Docs requirement**: user-visible changes (CLI args, config, env vars, defaults) must update docs in the same PR (#62599).
6. **Pre-commit `tools/pre_commit/check_agent_files.py`** (#59657) validates skill frontmatter and AGENTS.md budget rules. `editing-agent-instructions.md` gained a "Skills" section (agentskills.io spec, symlinks, SKILL.md < 500 lines).
7. Removed: `.gemini/` and `CLAUDE.md` (#58541), and Python 3.10 support. Added: incremental-build guidance for C++ and security-doc pointers. AGENTS.md grew to 163 lines.
8. Also new: `rust/AGENTS.md` and `rust/src/bench/AGENTS.md` (Rust frontend, not present at the freeze).

## 4. CPU-only environment
Host: 8 cores (AVX2, no AVX-512), 31 GB RAM, Ubuntu 22.04, no GPU.

What worked (verified end to end in a fresh clone with `setup_cpu_env.sh`: **792 s ≈ 13 min**, with warm uv/apt caches; the clone itself took 6.5 s):
```bash
sudo apt-get install -y gcc-12 g++-12 libnuma-dev
uv venv --python 3.12
export VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" CC=gcc-12 CXX=g++-12 MAX_JOBS=4 OMP_NUM_THREADS=4
uv pip install -r requirements/lint.txt && .venv/bin/pre-commit install
uv pip install -e . --torch-backend=cpu                 # CPU source build (~most of the time)
uv pip install -r requirements/test/cuda.in --torch-backend=cpu
uv pip install -e tests/vllm_test_utils                 # needed by tests/utils_ and others
```
Gotchas:
- `VLLM_USE_PRECOMPILED=1 ... --torch-backend=auto` (the AGENTS.md path) installs, but it pulls the CUDA wheel. There is no CPU native extension, so it is not usable for CPU.
- The CPU source build needs **GCC/G++ ≥ 12.3** and `libnuma-dev`. The default gcc-11 fails.
- `requirements/test/cuda.txt` (pinned) conflicts with CPU torch. Use `cuda.in` with `--torch-backend=cpu`.
- A bare `import vllm._C` crashes on this AVX2-only host. Use the supported dispatch, `current_platform.import_kernels()`, which works. Tests are unaffected.
- HF Hub: many tests download tokenizers/configs. Gated meta-llama repos return **401** without `HF_TOKEN`, and anonymous use hits Qwen **429** rate limits. **Give eval sessions an `HF_TOKEN`** and/or pre-warm `~/.cache/huggingface`.
- E2E scheduler tests ask for `gpu_memory_utilization`-style 0.92 of RAM (≈29 GiB). They error on a 31 GB box when other work is running.

### CPU test matrix (at 98661fe01; `pytest -q --timeout=300 -p no:cacheprovider`, 600 s wall cap)
| Path | P | F | S | E | sec | Main failure cause |
|---|---:|---:|---:|---:|---:|---|
| tests/v1/core | 291 | 0 | 0 | 2 | 424 | 2 E2E setup errors: CPU memory request > available |
| tests/tokenizers_ (was tests/tokenization) | 374* | 0 | 120 | 70 | 600 | **hit cap**, partial; HF downloads/gated |
| tests/entrypoints/{test_chat_utils,test_context,test_utils}.py | 95 | 0 | 0 | 0 | 138 | all pass |
| tests/tool_parsers | 676 | 14 | 0 | 41 | 354 | mistral_common API mismatch (14); HF 429 (41); 41 xfail |
| tests/reasoning | 434 | 11 | 0 | 0 | 248 | GLM4/SeedOSS assertion mismatches (cause unverified) |
| tests/config | 51 | 3 | 0 | 0 | 287 | gated Llama 401 |
| tests/test_config.py | 124 | 5 | 1 | 0 | 253 | CPU async_scheduling/IR-op expectations; gated 401 |
| tests/v1/structured_output | 20 | 0 | 0 | 0 | 26 | all pass |
| tests/multimodal | 323 | 12 | 0 | 0 | 294 | external MP4 URL 403 |
| tests/utils_ | 164 | 3 | 1 | 0 | 44 | CUDA memory-profiling tests |
| tests/v1/kv_connector/unit | 320 | 19 | 20 | 0 | 443 | CUDA-only paths, Ray GPU, NIXL, gated 401 |
| tests/lora/{test_lora_utils,test_peft_helper,test_resolver,test_utils}.py | 31 | 0 | 0 | 0 | 22 | all pass |
| tests/transformers_utils | 35 | 1 | 0 | 0 | 15 | gated Llama 401 |

\* observed progress at the cap, not a pytest summary. Note that `tests/tool_parsers` lives at top level at this commit, not under `tests/entrypoints/openai/`.

**Recommendation for eval grading:**
- Score tasks with the specific test files each answer commit touches, not whole directories. Most failures above are environmental (HF auth, CUDA) and are stable across base and answer.
- Grade as "fail-before / pass-after" on those targeted tests, and run the same tests at the base commit as a baseline.
- The most CPU-friendly areas: v1/core (scheduler and KV cache manager), structured_output, tool_parsers, reasoning, config, multimodal processing, utils_, LoRA utils, and entrypoint unit tests.
- With an `HF_TOKEN`, most of the 401 failures should go away. This has not been tested.

Full attempt log, exact errors, and timings are in the appendix (cpu_env_notes.md).

---
# Appendix: cpu_env_notes.md
# Frozen vLLM CPU evaluation environment

## Scope and machine

Only the fork `COG-GTM/vllm` was cloned, at detached commit
`98661fe012c5c467252d4df8411d2f46190e9268`. Primary checkout:
`/home/ubuntu/work/vllm-freeze`. No source edits, commits, pushes, PRs, or upstream
repository operations. Fresh-directory setup replays are sibling checkouts under
`/home/ubuntu/work/`; all reports/logs/scripts are in `out/`.

Ubuntu 22.04.5, Linux x86_64, 8 vCPUs, 31.34 GiB physical RAM; Intel Xeon exposes
AVX2 but not AVX512 or AMX. No GPU. uv was missing from the initial PATH and from
`~/.local/bin`; installed using the documented Astral installer (uv 0.13.6).
All Python commands used uv or a checkout's `.venv/bin/python` (CPython 3.12.16).

## Setup attempts, chronological

Wall timings are `/usr/bin/time -p` real seconds unless identified otherwise.
Raw stdout/stderr is retained in the named logs.

| Step / exact command | Seconds | Result / evidence |
|---|---:|---|
| `git clone https://git-manager.devin.ai/proxy/github.com/COG-GTM/vllm /home/ubuntu/work/vllm-freeze` | 6.51 | Success; then `git checkout 98661fe012c5c467252d4df8411d2f46190e9268` detached |
| `curl -LsSf https://astral.sh/uv/install.sh \| sh` | not separately timed (~1s tool wall) | Installed uv; no preexisting environment to activate |
| `uv venv --python 3.12` | 0.76 | Downloaded CPython and created `.venv` |
| `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto` | 33.69 | Success, CPU torch 2.11.0+cpu; `install_precompiled.log` |
| `.venv/bin/python -c 'import vllm; import torch; print(vllm.__version__, torch.__version__); import vllm._C'` | 4.80 | `import vllm` succeeds, but native extension absent: `ModuleNotFoundError: No module named 'vllm._C'`; `import_initial.log` |
| `uv pip install -r requirements/lint.txt` | 0.25 | Success; `.venv/bin/pre-commit install` installed hooks, no commits made |
| `uv pip install -r requirements/test/cuda.txt --torch-backend=cpu` | 1.04 | Resolution fails: `Because there is no version of torch==2.11.0+cu130 and you require torch==2.11.0+cu130, we can conclude that your requirements are unsatisfiable.`; `test_deps_cuda.log` |
| `VLLM_TARGET_DEVICE=cpu uv pip install -e . --torch-backend=cpu` | 8.90 | Build fails: `X86 backend requires gcc/g++ >= 12.3`; default GCC was 11; `install_cpu.log` |
| `uv pip install -r requirements/test/cuda.in --torch-backend=cpu --dry-run` | 8.18 | Resolves successfully; `test_deps_source.log` |
| `uv pip install -r requirements/test/cuda.in --torch-backend=cpu` | 10.75 | Installs full source test requirements; `test_deps_source_install.log` |
| `sudo -n apt-get update` | not separately timed | Success, `apt_update.log` |
| `sudo -n apt-get install -y gcc-13 g++-13 libnuma-dev` | 0.25 | `E: Unable to locate package gcc-13`; `apt_cpu.log` |
| `sudo -n apt-get install -y gcc-12 g++-12 libnuma-dev` | 6.54 | Success, GCC 12.3.0; `apt_cpu12.log` |
| `CC=gcc-12 CXX=g++-12 MAX_JOBS=4 VLLM_TARGET_DEVICE=cpu uv pip install -e . --torch-backend=cpu` | 754.70 | Native CPU build succeeds; `install_cpu12.log` |
| CPU native dispatch/import + pytest availability smoke | 1.56 | Success; `import_cpu_dispatch.log` also verifies selected CPU platform/native kernel dispatch |
| `uv pip install -e tests/vllm_test_utils` | 0.61 | Needed for utils test collection; `test_utils_install.log` |

The sum of the measured initial setup/diagnostic steps above is approximately
14 minutes, excluding untimed installer/update/checkout steps and interactive
investigation. The definitive complete-script replay time is recorded below.
The CPU wheel build compiled all x86 variants and fetched oneDNN v3.10; this is
why setup needs a C++ compiler, NUMA headers, GitHub access, RAM and ~13 minutes.

### Dependency caveats

The first precompiled attempt was **not** blocked by CUDA libraries: host detection
selected CPU dependencies, but no usable CPU extension was installed. Explicit
CPU source build was necessary. The pinned CUDA test lock file is unusable with
the CPU torch index. The unpinned `requirements/test/cuda.in` successfully resolves
on CPU, including substantial optional test packages; its Torch 2.11.0 constraint
accepts 2.11.0+cpu. It downgrades Transformers 5.21.0 to 5.5.3, Tokenizers 0.23.3 to
0.22.2 and torchaudio 2.11.1+cpu to 2.11.0+cpu. These are deliberate requirements,
not manual edits to shared files. Exact installed versions are saved in
`cpu_env_freeze.txt`; editable paths in that diagnostic freeze refer to the primary
checkout. Fresh setup script resolves requirements anew, so it is not a fully
hermetic dependency lock.

### Native import trap and setup replays

Do **not** test AVX2 hosts using direct `import vllm._C`: `_C` is the AMX/AVX512
variant. A first fresh-directory replay completed installation but our overly
strict smoke command crashed with `Illegal instruction (core dumped)` (790.79s,
`setup_verify.log`). Retrying after importing torch first still crashed, so import
ordering did not explain it (`verify_import_retry.log`). CPU feature detection in
`vllm/platforms/cpu.py:381-409` dispatches this machine to `_C_AVX2`; the runtime
call `current_platform.import_kernels()` succeeds (`import_cpu_dispatch.log`).

A corrected fresh-directory replay (`vllm-cpu-verify-final`) completed in 818.74s,
printing `CPU environment ready: 0.1.dev16316+g98661fe01 2.11.0+cpu cpu` and
`10 passed in 2.63s` (`setup_verify_final.log`). A final replay of the delivered
script, also including local test-utils installation, is recorded in
`setup_verified.log`: **792.46 seconds (13m12.46s) total**, including clone,
detached checkout, venv creation, OS package checks, CPU compilation, all test
dependencies, helper installation, import/native dispatch verification, and a
10-test smoke run. Its final output was:

```text
CPU environment ready: 0.1.dev16316+g98661fe01 2.11.0+cpu cpu
10 passed in 1.26s
real 792.46
user 2796.75
sys 131.56
```

Command: `/usr/bin/time -p bash /home/ubuntu/work/out/setup_cpu_env.sh /home/ubuntu/work/vllm-cpu-verified`.
Each replay uses a new clone and new venv, **but shared uv,
Python-download and OS package caches are warm**, not a pristine machine image.
Only the initial clone download was cold. Sudo without a password and access to
the private fork proxy are prerequisites for the copy-paste script.

## Reproduction

Run the delivered script from a CPU Ubuntu Linux session:

```bash
bash /home/ubuntu/work/out/setup_cpu_env.sh /home/ubuntu/work/new-cpu-checkout
```

The working installation core (inside the frozen checkout) is:

```bash
uv venv --python 3.12
export VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES=""
export CC=gcc-12 CXX=g++-12 MAX_JOBS=4 OMP_NUM_THREADS=4
uv pip install -r requirements/lint.txt
.venv/bin/pre-commit install
uv pip install -e . --torch-backend=cpu
uv pip install -r requirements/test/cuda.in --torch-backend=cpu
uv pip install -e tests/vllm_test_utils
.venv/bin/python -c 'import vllm, torch, pytest, pytest_timeout; from vllm.platforms import current_platform; current_platform.import_kernels(); print(vllm.__version__, torch.__version__, current_platform.device_type)'
```

## Test methodology and interpretation

Each matrix row ran:

```bash
VLLM_TARGET_DEVICE=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 \
HF_HUB_DOWNLOAD_TIMEOUT=30 HF_HUB_ETAG_TIMEOUT=15 \
timeout 600 .venv/bin/python -m pytest <path(s)> -q --timeout=300 -p no:cacheprovider
```

No `-x`, no custom CPU mocks, no source changes, no blanket skip filters. Seconds
in JSON/table are subprocess wall seconds including process startup; exact pytest
elapsed time is retained in each row's notes and raw `test_00.log` through
`test_12.log`. Existing tests can opt into pytest-rerunfailures themselves;
multimodal reports 9 such reruns. Xfail is not counted as skip: tool parsers have
41 xfails and multimodal 1, in addition to the requested columns. Pytest emits
an unset `asyncio_default_fixture_loop_scope` deprecation warning; this is test
configuration, not an unloaded venv. No config was edited to silence warnings.

`tests/tokenization` moved to `tests/tokenizers_`; `tests/test_utils.py` is absent
and `tests/utils_` is the current location. Tool parsers are at `tests/tool_parsers`.
Both `tests/config` and `tests/test_config.py` exist and were separately run.
Entry points deliberately cover only `test_chat_utils.py`, `test_context.py`, and
`test_utils.py`. LoRA deliberately covers only `test_lora_utils.py`,
`test_peft_helper.py`, `test_resolver.py`, and `test_utils.py`, excluding GPU/model
kernel suites. Do not extrapolate these subset counts to their full directories.

The tokenizer run hit the 600s wall cap. Its numeric fields are **partial observed
progress symbols** (374 pass / 120 skip / 70 error), not a final pytest summary;
906 items were collected and 564 outcomes were visible before termination. There
are no retained traceback details for its errors because pytest had not printed
the final failure report. Do not treat the unexecuted/unreported remainder as
passing or skipped. HF downloads are implicated by test fixtures, but exact error
causes are unverified for this capped row.

A runner bug initially passed no path for the nonexistent `tests/tokenization`,
accidentally triggering whole-repository collection; it aborted in CUDA platform
import after 41.61s. This is **not** a matrix row. The alternate path was fixed and
run explicitly; the final log/row is the proper tokenizer run. An existence guard
is now in `run_matrix.py`. Initial utils collection also failed with
`ModuleNotFoundError: No module named 'vllm_test_utils'` in 10.28s; local helper
installation fixed collection and the matrix contains its completed rerun.

### Observed failure categories

- Core: 291 pass; two scheduler E2E fixture failures. The worker explicitly logged
  `Available memory on node 0 (22.63/31.34 GiB) on startup is less than desired CPU
  memory utilization (0.92, 28.84 GiB)`. Compilation replay was concurrent, so these
  E2E memory-sensitive times/counts are not isolated performance measurements.
  A lower CPU memory utilization setting might help; not applied to tests.
- Tool parsers: 14 Mistral API failures (`GrammarFactory.get_lark_from_jinja()
  missing 1 required positional argument: 'force_tool_names'`); 41 setup errors
  include Qwen/Qwen3-32B Hugging Face API HTTP 429, followed by inaccessible
  vocabulary. These are observed API/network failures, not CUDA failures.
- Reasoning: 11 GLM4/SeedOSS reasoning/content assertion mismatches. No evidence
  establishes whether a specific dependency revision or source behavior is the
  underlying issue; no unsupported claim that this is preexisting/flaky.
- Config: three gated Meta Llama model-config HTTP 401 failures. Root config file:
  five failures comprise three gated-model HTTP 401s, an async scheduling expectation
  (CPU platform explicitly forces it off at `cpu.py:135`), and an IR op priority
  expectation (`['native']` vs `['vllm_c', 'native']`).
- Multimodal: all 12 failures are requests to the external bogotobogo MP4 URL
  returning `403 Client Error: Forbidden`. No changes to media source URLs.
- Utils: after helper installation, 3 CUDA memory profiling/snapshot failures
  (`libcudart is not loaded` / `Cannot access accelerator device when none is available`).
- KV connector units: CUDA-dependent HF3FS/Mooncake/pinned memory paths, gated
  Llama config, Ray requesting 1 GPU when `CUDA_VISIBLE_DEVICES=[]`, and NIXL absent;
  320 other tests pass and 20 skip.
- Transformers utils: sole failure is a gated Llama-3.2-1B-Instruct config HTTP 401.

### Downloads and eval recommendations

HF internet is required for tokenizers, parser tokenizer fixtures, reasoning,
model-config tests, transformers-utils, some connector and core E2E tests. Examples
visible in caches/logs: Qwen/Qwen3-32B, Qwen/Qwen3-0.6B, ByteDance-Seed/Seed-OSS-36B-Instruct,
EleutherAI/pythia-70m, GPT2 and multiple Meta Llama configs. Model cache directories
are not proof full weights downloaded: many tests fetch only tokenizers/configs.
HF cache grew to about 15 GiB. No HF token was provisioned; gated-model tests cannot
succeed anonymously. Shared anonymous IP hit the 500 API requests/300s limit.
Prefetch the eval's exact public tokenizer/config revisions and use authenticated
access for gated models; do not rely on this network-dependent run as a stable baseline.

Best verified all-pass CPU eval candidates: the three entrypoint unit files (95),
`tests/v1/structured_output` (20), and the four listed LoRA unit files (31).
`tests/utils_` except `test_mem_utils.py`, core excluding scheduler E2E, and public
non-gated transformers/config units are promising but **those exact filtered sets
were not separately run**, so do not report them as all-pass. Parser/reasoning suites
have many passing CPU tests but need dependency/network control and known-failure
selection. Do not use whole GPU-oriented connector or LoRA/kernel directories
without curated selectors on this host. This is environment feasibility/count
measurement, not a CPU throughput benchmark; some setup compilation and tokenizer
download runs overlapped, and no inference performance claim is made.
