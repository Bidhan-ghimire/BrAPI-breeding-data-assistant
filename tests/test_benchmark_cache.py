"""The benchmark cache (eval/benchmark_cache.py; coach Task 1 step 3): built ONCE, through the app's own data tools
(get_observation_units / get_observations -> BrapiClient -> the cache), under one approval made by code for exactly the
studies and variables the sub-questions name. It records each study's unit count and whether the app's limits hold it
(the app's pages per fetch; RunConfig.max_fetches fetch calls per question). It changes no limit and never the source cache."""
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
    assert study["app_pages"] is True and study["observations"]["V1"]["returned"] == 6 and study["study_record"] == "ok"
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


def test_over_the_fetch_limit_and_over_the_apps_pages_are_listed(live, tmp_path, monkeypatch):
    """A sub-question needing more fetch calls than one question allows is listed; so is a study whose observations do
    not come back complete within the app's pages per fetch (review 10: the check looked at the roster only)."""
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
    assert again["studies"]["S1"]["app_pages"] is False and again["over_app_pages"] == ["S1"]


def test_the_fetches_use_the_apps_own_run_settings(live, tmp_path, monkeypatch):
    """2026-10-21: built with the client's bare default (30 s read), every SweetPotatoBase /observations request timed
    out; a run uses app.live_config's settings (APP_READ_TIMEOUT, default 90 s; 600 s per question) and gets the data in
    about 40 s. The cache is built with the same settings a run uses, and says how long each fetch took against the
    run's budget."""
    source, _server = live
    monkeypatch.setenv("APP_READ_TIMEOUT", "77")
    report = cache_mod.build(SUBS[:2], source=source, dest=tmp_path / "bench", out_dir=tmp_path / "out")
    assert report["read_timeout"] == 77.0 and report["time_budget_seconds"] == 600 and report["max_fetches"] == 5
    row = report["subquestions"]["S1.1"]
    assert row["fetch_seconds"] >= 0 and row["over_time_budget"] is False
    assert report["studies"]["S1"]["observations"]["V1"]["seconds"] >= 0


class TwoPageStudyServer(StudyServer):
    """The measurements and the plot list as 3 + 3 rows over two pages, as a study larger than one page; the rest as StudyServer."""

    def __call__(self, req):
        import json

        from brapi_client import TransportResponse
        from part2_support import load_fixture

        endpoint = req.url[len(LIVE_BASE):]
        if endpoint not in ("/observations", "/observationunits"):
            return super().__call__(req)
        page = int(req.params.get("page", 0))
        self.calls.append(f"{endpoint} page {page}")
        _, body = load_fixture(self.REPLIES[endpoint])
        payload = json.loads(body)
        rows = payload["result"]["data"]
        payload["result"]["data"] = rows[:3] if page == 0 else rows[3:]
        payload["metadata"]["pagination"].update(currentPage=page, pageSize=3, totalCount=len(rows), totalPages=2)
        return TransportResponse(200, {"Content-Type": "application/json"}, json.dumps(payload).encode("utf-8"))


def test_the_cache_holds_both_pages_a_run_reads(live, tmp_path, monkeypatch):
    """Item 14 (coach 2026-10-23 21:48 UTC) and its review: the cache is filled as a run reads - the app's own pages per fetch
    (app.live_config, RunConfig's default 2, never a literal) - so a study over one page is saved whole and counted as held; the
    report names the number; the code-made approval allows three attempts per request and 3 hours per page at that many pages."""
    from datetime import timedelta

    import app
    import brapi_client as bc
    import run

    server = TwoPageStudyServer()
    monkeypatch.setattr(bc, "RequestsTransport", lambda: server)
    source, _ = live
    built = []
    real = cache_mod.srv.build_context

    def recording(*args, **kwargs):
        built.append(real(*args, **kwargs))
        return built[-1]

    monkeypatch.setattr(cache_mod.srv, "build_context", recording)
    two = [sub("S1.4", ["S1"], [("fresh root yield", None), ("dry matter content", None)])]
    report = cache_mod.build(two, source=source, dest=tmp_path / "bench", out_dir=tmp_path / "out", max_fetches=5)
    pages = app.live_config("q").max_observation_pages
    assert pages == run.RunConfig(question="q").max_observation_pages == 2 and report["pages_per_fetch"] == pages
    assert {"/observationunits page 1", "/observations page 1"} <= set(server.calls)
    assert report["studies"]["S1"]["app_pages"] is True and report["over_app_pages"] == []
    approval = built[0].approval
    assert approval.max_http_attempts == 3 * (1 + pages * (1 + 2))                   # one study record; a roster and two variables
    assert approval.expires_at_utc - approval.issued_at_utc == timedelta(hours=3 * pages, seconds=1)
