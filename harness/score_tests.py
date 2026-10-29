#!/usr/bin/env python3
"""Deterministic hidden-test scorer for one vllm-eval run.

Overlays the task's hidden tests (ground-truth test files and/or synthetic
tests) onto a checkout of the solver's code, runs pytest, and reports
fail-to-pass (F2P) and pass-to-pass (P2P) outcomes.

usage: score_tests.py --repo DIR --harness DIR --gt-repo DIR --task ID [--out FILE]
  --repo     checkout of the code under test (solver head, base, or GT); must have .venv
  --harness  checkout of COG-GTM/vllm@eval-harness (dataset/manifest.json, dataset/synthetic_tests/)
  --gt-repo  checkout of COG-GTM/vllm containing the ground-truth commits
"""
import argparse, json, os, subprocess, sys, tempfile

ENV = {"VLLM_TARGET_DEVICE": "cpu", "CUDA_VISIBLE_DEVICES": "", "TOKENIZERS_PARALLELISM": "false",
       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
HERE = os.path.dirname(os.path.abspath(__file__))


def overlay(task, repo, harness, gt_repo):
    for o in task["overlay"]:
        kind, src = o["src"].split(":", 1)
        if kind == "gt":
            data = subprocess.run(["git", "-C", gt_repo, "show", f"{task['commit_sha']}:{src}"],
                                  check=True, capture_output=True).stdout
        elif kind == "synthetic":
            data = open(os.path.join(harness, src), "rb").read()
        else:
            raise ValueError(o["src"])
        dst = os.path.join(repo, o["dst"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "wb").write(data)


def run_pytest(task, repo, timeout):
    files = sorted({o["dst"] for o in task["overlay"]} | set(task.get("extra_test_files", [])))
    outcomes_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    cmd = [os.path.join(repo, ".venv/bin/python"), "-m", "pytest", "-p", "no:cacheprovider",
           "-p", "score_outcomes_plugin", "-rA", "-q", "--color=no", *task.get("pytest_args", []), *files]
    env = {**os.environ, **ENV, "PYTHONPATH": os.pathsep.join([repo, HERE]),
           "SCORE_OUTCOMES_FILE": outcomes_file}
    try:
        p = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=timeout)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        raw = e.stdout or ""
        out, rc = (raw.decode(errors="replace") if isinstance(raw, bytes) else raw), "timeout"
    try:
        outcomes = json.load(open(outcomes_file))
    except (OSError, ValueError):
        outcomes = {}
    return cmd, rc, outcomes, out


def matches(test_id, outcomes):
    return [v for k, v in outcomes.items() if k == test_id or k.startswith(test_id + "[")]


def passed(test_id, outcomes):
    r = matches(test_id, outcomes)
    return bool(r) and all(v in ("PASSED", "XFAIL") for v in r)


def main():
    ap = argparse.ArgumentParser()
    for a in ("--repo", "--harness", "--gt-repo", "--task"):
        ap.add_argument(a, required=True)
    ap.add_argument("--out")
    ap.add_argument("--timeout", type=int, default=1200)
    a = ap.parse_args()
    manifest = json.load(open(os.path.join(a.harness, "dataset/manifest.json")))
    task = next(t for t in manifest if t["task_id"] == a.task)
    overlay(task, a.repo, a.harness, a.gt_repo)
    cmd, rc, outcomes, out = run_pytest(task, a.repo, a.timeout)
    f2p = {t: passed(t, outcomes) for t in task["f2p"]}
    p2p = {t: passed(t, outcomes) for t in task.get("p2p", [])}
    f2p_rate = sum(f2p.values()) / len(f2p) if f2p else 0.0
    p2p_regressions = [t for t, ok in p2p.items() if not ok]
    T = f2p_rate * (0.5 if p2p_regressions else 1.0)
    res = {"task_id": a.task, "T": round(T, 4), "f2p_rate": round(f2p_rate, 4), "f2p": f2p,
           "p2p_total": len(p2p), "p2p_regressions": p2p_regressions, "pytest_returncode": rc,
           "command": " ".join(cmd), "output_tail": out[-4000:]}
    s = json.dumps(res, indent=1)
    if a.out:
        open(a.out, "w").write(s)
    print(s)


if __name__ == "__main__":
    sys.exit(main())
