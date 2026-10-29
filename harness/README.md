# CPU evaluation harness

This orphan branch contains private evaluator assets, not vLLM source. Keep it
and the GT history out of the agent checkout. Give the agent only its task
statement and a source checkout at `98661fe012c5c467252d4df8411d2f46190e9268`.

## Setup

From a fresh base source checkout, run the absolute path to
`harness/setup_cpu_env.sh` in this separate harness checkout. It creates `.venv`,
installs the pinned CPU environment, installs vLLM editable, and builds two real
CPU kernels from unchanged base C++ source. Requires Linux x86_64 with AVX2,
a C++17 compiler/OpenMP, and uv. Initial setup needs package/wheel downloads;
test execution does not. Timings are warm-cache observations, not cold-install
estimates. No GPU, model weights, or Hugging Face downloads are required.

## Run

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/path/to/checkout/.venv/bin/python /path/to/harness/harness/score_tests.py \
  --repo /path/to/checkout --harness /path/to/harness \
  --gt-repo /path/to/private-full-history-vllm --task vllm-41991 \
  --out /path/to/result.json --timeout 90
```

Use separate, clean source checkouts per task/state: overlays modify files in
place. The private full-history checkout supplies only `gt:` test overlays;
synthetic overlays live on this branch. For GT references apply exactly
`git diff <commit>^ <commit> --binary -- ':!tests'` to the freeze base before
running the scorer. Do not cherry-pick future test implementations into base.

The supplied scorer and plugin are unchanged. **Consumers must filter manifest
entries with `excluded: true` before aggregating benchmark scores**: the scorer
does not skip them. A GT score of 1 alone is insufficient: the scorer grades
only listed selectors, not every executed test. Eligibility also requires
successful pytest exit and the recorded contract audit. In particular, two
excluded tasks have GT T=1 but pytest exits 1 because
additional overlaid contract cases fail. Their failures are intentionally kept.

See `dataset/FAIRNESS_AUDIT.md` for interface additions and exclusions, and
`dataset/REFERENCE_RESULTS.md` for the per-task table. `reference_results.json`
contains one base and three GT runs per task without `output_tail`.
