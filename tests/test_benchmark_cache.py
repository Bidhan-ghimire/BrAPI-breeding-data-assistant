"""The benchmark cache (eval/benchmark_cache.py; coach Task 1 step 3): built ONCE, through the app's own data tools
(get_observation_units / get_observations -> BrapiClient -> the cache), under one approval made by code for exactly the
studies and variables the sub-questions name. It records each study's unit count and whether the app's limits hold it
(one page per fetch; RunConfig.max_fetches fetch calls per question). It changes no limit and never the source cache."""
import importlib.util
import sys
from pathlib import Path

import pytest

from cache_store import CacheStore
from part2_support import seed_fixture
from test_eval_harness import LIVE_BASE, FakeLiveServer, snapshot


class StudyServer(FakeLiveServer):
    """The harness test's fake database, plus one study's own record (a run's Retriever asks for it: get_study)."""
    REPLIES = {**FakeLiveServer.REPLIES, "/studies/S1": "synthetic_study_S1.json"}

MODULE = Path(__file__).resolve().parents[1] / "eval" / "benchmark_cache.py"
_spec = importlib.util.spec_from_file_location("_eval_benchmark_cache_under_test", MODULE)
cache_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cache_mod
_spec.loader.exec_module(cache_mod)


def sub(sub_id, studies, variables):
    return {"sub_id": sub_id, "parent": sub_id.split(".")[0], "tier": "T1", "question": "q",
            "studies": [{"studyDbId": s, "name": f"study {s}"} for s in studies],
            "variables": [{"name": name, "term": term} for name, term in variables]}


SUBS = [sub("S1.1", ["S1"], [("fresh root yield", None)]), sub("S1.2", ["S1"], []),
        sub("S1.3", ["S1"], [("no such trait", "CO_999:0000001")])]


@pytest.fixture
def live(tmp_path, monkeypatch):
    import brapi_client as bc

    monkeypatch.setenv("BRAPI_MODE", "live")
    monkeypatch.setenv("BRAPI_BASE_URL", LIVE_BASE)
    monkeypatch.setenv("PART2_IGNORE_DOTENV", "true")
    source = tmp_path / "source"
    for name in ("synthetic_studies.json", "synthetic_variables.json"):
        seed_fixture(CacheStore(source), LIVE_BASE, name)
    server = StudyServer()
    monkeypatch.setattr(bc, "RequestsTransport", lambda: server)
    return source, server


def test_built_once_through_the_apps_tools_with_counts_and_limits(live, tmp_path):
    source, server = live
    before = snapshot(source)
    report = cache_mod.build(SUBS, source=source, dest=tmp_path / "bench", out_dir=tmp_path / "out", max_fetches=1)
    # the study's record, its roster and one variable: nothing else (review 10: get_study was missing, so cache-only runs
    # met a refused request)
    assert sorted(server.calls) == ["/observations", "/observationunits", "/studies/S1"]
    study = report["studies"]["S1"]
    assert study["units"] == {"status": "complete", "complete": True, "returned": 6, "reported_total": 6}
    assert study["one_page"] is True and study["observations"]["V1"]["returned"] == 6 and study["study_record"] == "ok"
    assert study["seconds"] >= 0                                                  # request time, against the run's budget
    rows = report["subquestions"]
    # run.required_fetches: one fetch per (study, variable); a roster only for a design question (review 10)
    assert rows["S1.1"]["fetch_calls"] == 1 and rows["S1.1"]["fetch_calls_with_rosters"] == 2
    assert rows["S1.1"]["over_max_fetches"] is False
    assert rows["S1.2"]["fetch_calls"] == 1 and rows["S1.2"]["over_max_fetches"] is False
    assert report["unresolved_variables"] == [{"sub_id": "S1.3", "name": "no such trait", "term": "CO_999:0000001"}]
    assert snapshot(source) == before                                            # the source cache is never changed
    assert len(snapshot(tmp_path / "bench")) > len(before)                      # the new cache holds the plot data


def test_a_cache_is_built_once(live, tmp_path):
    source, _server = live
    (tmp_path / "bench").mkdir()
    (tmp_path / "bench" / "x").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="already"):
        cache_mod.build(SUBS, source=source, dest=tmp_path / "bench", out_dir=tmp_path / "out", max_fetches=5)


def test_no_catalog_in_the_source_is_refused(live, tmp_path):
    with pytest.raises(ValueError, match="catalog"):
        cache_mod.build(SUBS, source=tmp_path / "empty", dest=tmp_path / "bench2", out_dir=tmp_path / "out", max_fetches=5)


def test_over_the_fetch_limit_and_over_one_page_are_listed(live, tmp_path, monkeypatch):
    """A sub-question needing more fetch calls than one question allows is listed; so is a study whose observations do
    not come back complete within one page (review 10: one_page looked at the roster only)."""
    source, _server = live
    two = [sub("S1.4", ["S1"], [("fresh root yield", None), ("dry matter content", None)])]
    report = cache_mod.build(two, source=source, dest=tmp_path / "bench", out_dir=tmp_path / "out", max_fetches=1)
    assert report["subquestions"]["S1.4"]["fetch_calls"] == 2 and report["over_max_fetches"] == ["S1.4"]
    real = cache_mod.srv.dispatch

    def short_page(ctx, name, arguments):                     # as the server would report a 1500-row variable
        result = real(ctx, name, arguments)
        if name == "get_observations":
            result.data.update(complete=False, status="incomplete", reported_total=1500)
        return result
    monkeypatch.setattr(cache_mod.srv, "dispatch", short_page)
    again = cache_mod.build(two, source=source, dest=tmp_path / "bench2", out_dir=tmp_path / "out2", max_fetches=5)
    assert again["studies"]["S1"]["one_page"] is False and again["over_one_page"] == ["S1"]
