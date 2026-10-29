"""pytest plugin: record per-nodeid outcomes to $SCORE_OUTCOMES_FILE as JSON."""
import json
import os

_res = {}


def pytest_runtest_logreport(report):
    nid = report.nodeid
    if report.when == "call":
        if hasattr(report, "wasxfail"):
            outcome = "XFAIL" if report.skipped else "XPASS"
        else:
            outcome = "PASSED" if report.passed else "SKIPPED" if report.skipped else "FAILED"
        if _res.get(nid) != "ERROR":
            _res[nid] = outcome
    elif report.failed:
        _res[nid] = "ERROR"
    elif report.when == "setup" and report.skipped:
        _res[nid] = "SKIPPED"


def pytest_collectreport(report):
    if report.failed:
        _res[report.nodeid or "<collection>"] = "ERROR"


def pytest_sessionfinish(session):
    path = os.environ.get("SCORE_OUTCOMES_FILE")
    if path:
        with open(path, "w") as f:
            json.dump(_res, f, indent=1, sort_keys=True)
