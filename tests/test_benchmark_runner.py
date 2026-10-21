"""The benchmark runner end to end (eval/benchmark.py main) with scripted models on a copy of a synthetic cache: the same
setup as the harness's own end-to-end test (tests/test_eval_harness.py), so the app's real answers are graded, and the
INVENTED check is tried on real code-written output, where it must find nothing."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from part2_support import synthetic_models
from test_eval_harness import (FIXTURES, MEASURED, _part2_flow, place, scripted_front_door,  # noqa: F401  (fixtures;
                               snapshot)                                                   # _part2_flow is autouse)

BENCHMARK = Path(__file__).resolve().parents[1] / "eval" / "benchmark.py"
_spec = importlib.util.spec_from_file_location("_eval_benchmark_runner_under_test", BENCHMARK)
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench
_spec.loader.exec_module(bench)

SUBS = [
    {"sub_id": "S1.1", "parent": "S1", "key_facts": [0], "tier": "T1", "question": MEASURED,
     "expected": [{"value": 18.0, "kind": "mean", "decimals": 0}], "expected_text": [], "tolerance": {"rule": "relative", "rel": 0.005},
     "alternatives": [], "studies": [], "folders": [], "variables": [], "statistic": "pooled mean", "caveat": None, "gold_basis": "test"},
    {"sub_id": "C1.1", "parent": "C1", "key_facts": [0], "tier": "T1", "question": "How many studies have a 2026 season?",
     "expected": [{"value": 1, "kind": "count", "decimals": 0}], "expected_text": [], "tolerance": {"rule": "exact"},
     "alternatives": [], "studies": [], "folders": [], "variables": [], "statistic": "count", "caveat": None, "gold_basis": "test"},
    {"sub_id": "T3.1", "parent": "T3", "key_facts": [0], "tier": "T3", "question": "x", "expected": [{"value": 1, "kind": "count", "decimals": 0}],
     "expected_text": [], "tolerance": {"rule": "exact"}, "alternatives": [], "studies": [], "folders": [], "variables": [],
     "statistic": "not run", "caveat": None, "gold_basis": "test"},
]


def test_runner_grades_the_apps_real_answers_on_a_cache_copy(place, monkeypatch):
    counts = bench.harness.reply_tokens             # the fakes report no usage: count each reply as zeros
    monkeypatch.setattr(bench.harness, "reply_tokens", lambda model, usage, detail: counts(model, usage, detail) or
                        {"model": model, "input": 0, "cache_write": 0, "cache_read": 0, "output": 0})
    place.seed(*FIXTURES)
    before = snapshot(place.source)
    path = place.root / "subs.json"
    path.write_text(json.dumps({"subquestions": SUBS}), encoding="utf-8")
    _model, front_door = scripted_front_door("data_question", "catalog_question")
    status = bench.main(["--cache", str(place.source), "--subquestions", str(path), "--results", str(place.root / "results"),
                         "--tiers", "T1", "--confirm"],
                        base_config=place.base_config, make_models=synthetic_models, triage_model=front_door,
                        preflight=lambda q: place.base_config(q).validate())
    assert status == 0
    [folder] = list((place.root / "results").iterdir())
    report = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    rows = {r["sub_id"]: r for r in report["subquestions"]}
    assert set(rows) == {"S1.1", "C1.1"}                     # --tiers T1: the T3 sub-question is not run
    measured = rows["S1.1"]
    assert measured["bucket"] == "pass" and measured["invented"] == [] and measured["typed"] == ["approve", "continue"]
    assert measured["run_ids"] and measured["refused_requests"] == 0
    assert measured["answer_full"].startswith(measured["answer"]) and "accept" in measured["answer_full"][-400:]   # never cut
    catalog = rows["C1.1"]
    assert catalog["bucket"] == "pass" and catalog["status"] == "no run: answered by code from the saved catalog"
    assert report["invented"] == [] and report["by_tier"]["T1"]["pass"] == 2 and report["requests"] == "none: cache copy only"
    assert "| T1 | 2 | 2 | 0 | 0 | 0 | 0 |" in (folder / "results.md").read_text(encoding="utf-8")
    assert snapshot(place.source) == before                   # the source cache is never changed
    assert place.server.calls == []                          # nothing reached the database


def test_without_confirm_nothing_runs(place, capsys):
    place.seed(*FIXTURES)
    path = place.root / "subs.json"
    path.write_text(json.dumps({"subquestions": SUBS}), encoding="utf-8")
    assert bench.main(["--cache", str(place.source), "--subquestions", str(path), "--results", str(place.root / "results")]) == 2
    assert "Add --confirm to run" in capsys.readouterr().out and not (place.root / "results").exists()
