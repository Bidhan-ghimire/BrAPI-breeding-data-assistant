"""
tests/test_benchmark_budgets.py - the benchmark-only budgets (Bidhan, 2026-10-21 ~17:58 UTC, answering Q1-Q3 of the baseline
report): the harness passes 24 fetches, 10 pages and 30 plan steps per run; the app's own defaults stay 5 / 1 / 12. Pages may
go to 20 since Bidhan's decision of 2026-10-23 (item 19), benchmark only; the app's own pages are 2 since the same decision
(item 14), so its defaults are 5 / 2 / 12.

The fetch count was already a run setting (RunConfig.max_fetches). Pages per plot-record fetch and steps per plan were fixed
numbers (one page; the plan contract's 12 steps), so they become run settings with the old values as defaults; the plan
contract keeps an absolute cap of 30. The HTTP attempt cap of the approval follows the fetches and pages in the benchmark.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

import agents.coordinator as co
import app
import brapi_mcp_server as srv
import contracts as c
import run as ctl
from part2_support import SYNTHETIC_BASE, synthetic_context, synthetic_plan_draft

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import benchmark as bench  # noqa: E402

NOW = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)


# --- 1. the app keeps its own limits; the benchmark passes its budgets -------------------------------------------

def test_the_apps_own_limits_are_5_2_12(monkeypatch):
    """Bidhan, 2026-10-23 (coach's DECISION entry 14:01 UTC, item 14): the app's own pages per plot-record fetch 1 -> 2 (the app,
    the Space and every run that names none); fetches and plan steps unchanged."""
    import app_space

    monkeypatch.delenv("APP_MAX_HTTP_ATTEMPTS", raising=False)                 # the default, whatever the shell sets (review of 23)

    live = app.live_config("How many plots does study S1 have?")
    assert (live.max_fetches, live.max_observation_pages, live.max_plan_steps) == (5, 2, 12)
    hosted = app_space.space_config("How many plots does study S1 have?")                   # the Space (review: unpinned)
    assert (hosted.max_fetches, hosted.max_observation_pages, hosted.max_plan_steps) == (5, 2, 12)
    default = ctl.RunConfig(question="q")
    assert (default.max_fetches, default.max_observation_pages, default.max_plan_steps) == (5, 2, 12)
    # Bidhan, 2026-10-24 (coach's DECISION entry 03:43 UTC, item 23): the app's HTTP attempts 10 -> 15 = 5 fetches x 2 pages +
    # 5 lookups (the review of item 14: two-page fetches used the whole cap of 10); the Space takes the same default
    assert live.max_http_attempts == 15 and hosted.max_http_attempts == 15
    # its reason too (review of item 23): every fetch's pages and the 5 lookups fit, so a change of pages or fetches raises it
    assert live.max_http_attempts >= live.max_fetches * live.max_observation_pages + 5


def test_a_valid_attempt_setting_is_read(monkeypatch):
    """Review of item 23: only bad values (0, text) were tried; a valid APP_MAX_HTTP_ATTEMPTS is the cap of the app and the Space,
    from 1 to 100."""
    import app_space

    for value in (12, 40):
        monkeypatch.setenv("APP_MAX_HTTP_ATTEMPTS", str(value))
        assert app.live_config("q").max_http_attempts == value and app_space.space_config("q").max_http_attempts == value
    monkeypatch.setenv("APP_MAX_HTTP_ATTEMPTS", "101")
    with pytest.raises(ValueError):
        app.live_config("q")


def test_the_benchmark_passes_its_budgets_per_run_and_only_when_asked():
    question = "How many plots does study S1 have?"
    assert bench.budgeted(app.live_config)(question) == app.live_config(question)
    config = bench.budgeted(app.live_config, max_fetches=24, max_pages=10, max_plan_steps=30)(question)
    assert (config.max_fetches, config.max_observation_pages, config.max_plan_steps) == (24, 10, 30)
    assert config.max_http_attempts == 240                                        # 24 fetches x 10 pages
    budget_problems = [p for p in config.validate() if any(k in p for k in ("max_fetches", "max_observation_pages", "max_plan_steps",
                                                                             "max_http_attempts"))]
    assert budget_problems == []
    assert config.max_elapsed_seconds == app.live_config(question).max_elapsed_seconds   # the time budget is not raised
    steps_only = bench.budgeted(app.live_config, max_plan_steps=30)(question)                 # review: one budget moves only itself
    assert (steps_only.max_fetches, steps_only.max_observation_pages, steps_only.max_http_attempts) == (5, 2, 15)
    pages_only = bench.budgeted(app.live_config, max_pages=4)(question)           # 5 x 4 = 20 is above the app's own 15
    assert (pages_only.max_fetches, pages_only.max_observation_pages, pages_only.max_http_attempts) == (5, 4, 20)
    assert app.live_config(question).max_fetches == 5 and app.live_config(question).max_http_attempts == 15
    assert bench.budgeted(app.live_config, max_fetches=3)(question).max_http_attempts == 15      # never below the app's own (3 x 2 = 6)


def test_twenty_pages_pass_the_harness_and_the_runs_own_check():
    """Bidhan, 2026-10-23 (coach's DECISION entry 14:01 UTC, item 19): benchmark-only pages 10 -> 20, the app unchanged. The
    run's own range (run.MAX_OBSERVATION_PAGES) follows: without it every 20-page run stopped at its configuration check."""
    question = "How many plots does study S1 have?"
    config = bench.budgeted(app.live_config, max_fetches=24, max_pages=20, max_plan_steps=30)(question)
    assert (config.max_fetches, config.max_observation_pages, config.max_plan_steps) == (24, 20, 30)
    assert config.max_http_attempts == 480                                        # 24 fetches x 20 pages
    assert [p for p in config.validate() if "max_observation_pages" in p or "max_http_attempts" in p] == []
    assert bench.parse_args(["--cache", "x", "--max-pages", "20"]).max_pages == 20
    assert app.live_config(question).max_observation_pages == 2 and ctl.RunConfig(question="q").max_observation_pages == 2   # (14)


@pytest.mark.parametrize("budget", [{"max_fetches": 25}, {"max_pages": 21}, {"max_plan_steps": 31}, {"max_fetches": 0},
                                    {"max_pages": 0}, {"max_plan_steps": 0}])
def test_the_benchmark_refuses_budgets_beyond_what_was_allowed(budget):
    with pytest.raises(ValueError):
        bench.budgeted(app.live_config, **budget)


def test_the_command_line_names_the_budgets_and_records_them():
    args = bench.parse_args(["--cache", "x", "--max-fetches", "24", "--max-pages", "10", "--max-plan-steps", "30"])
    assert (args.max_fetches, args.max_pages, args.max_plan_steps) == (24, 10, 30)
    plain = bench.parse_args(["--cache", "x"])
    assert (plain.max_fetches, plain.max_pages, plain.max_plan_steps) == (None, None, None)
    made = bench.budgeted(app.live_config, max_fetches=24, max_pages=10, max_plan_steps=30)
    assert bench.budgets_record(args, made) == {"max_fetches": 24, "max_observation_pages": 10, "max_plan_steps": 30, "max_http_attempts": 240}
    assert bench.budgets_record(plain, app.live_config) == {}


@pytest.mark.parametrize("change", [{"max_observation_pages": 0}, {"max_observation_pages": 21}, {"max_plan_steps": 0},
                                    {"max_plan_steps": c.MAX_PLAN_STEPS + 1}, {"max_plan_steps": 12.0}])
def test_a_run_setting_outside_its_range_is_refused(change):
    import dataclasses

    config = dataclasses.replace(ctl.RunConfig(question="q"), **change)
    assert any("max_observation_pages" in p or "max_plan_steps" in p for p in config.validate())


# --- 2. plan steps: the run's own limit, under the contract's absolute cap ------------------------------------------

def _draft(steps: int) -> co.PlanDraft:
    draft = synthetic_plan_draft("S1", "V1")
    extra = [{"step_id": f"step_{n}", "agent": "analyst", "action": "table_info", "depends_on": ["step_1"]} for n in range(5, steps + 1)]
    draft["steps"] = draft["steps"] + extra
    return co.PlanDraft.model_validate(draft)


def _assemble(steps: int, **kw):
    meta = co.KnownMetadata.from_rows([{"studyDbId": "S1", "studyName": "SYNTHETIC trial"}],
                                      [{"observationVariableDbId": "V1", "observationVariableName": "fresh root yield"}],
                                      base_url=SYNTHETIC_BASE, complete=True)
    return co.assemble_plan("What is the mean fresh root yield in study S1, per clone?", _draft(steps), meta, base_url=SYNTHETIC_BASE,
                            now=NOW, plan_id="plan_budget", **kw)


def test_a_plan_longer_than_the_runs_limit_is_refused():
    _plan, problems, _ = _assemble(13)
    assert any("13 steps" in p and "at most 12" in p for p in problems), problems           # the default stays 12
    plan, problems, _ = _assemble(12)
    assert problems == [] and len(plan.steps) == 12


def test_a_longer_run_limit_lets_a_longer_plan_through_up_to_the_contracts_cap():
    plan, problems, _ = _assemble(30, max_plan_steps=30)
    assert problems == [] and len(plan.steps) == 30
    _plan, problems, _ = _assemble(31, max_plan_steps=30)
    assert any("31 steps" in p and "at most 30" in p for p in problems), problems
    assert c.MAX_PLAN_STEPS == 30


def test_the_controller_passes_its_plan_limit_to_the_planner(tmp_path, monkeypatch):
    from part2_support import part2_runtime

    part2_runtime(monkeypatch, server=SYNTHETIC_BASE)                       # as tests/test_run.py does for every run
    seen = []
    original = ctl.draft_plan

    async def recording(*args, **kwargs):
        seen.append(kwargs.get("max_plan_steps"))
        return await original(*args, **kwargs)

    monkeypatch.setattr(ctl, "draft_plan", recording)
    from test_run import cfg, go

    go(cfg(tmp_path, max_plan_steps=30), answers=["approve", "approve", "approve"])
    assert seen and set(seen) == {30}
    seen.clear()
    go(cfg(tmp_path / "default"), answers=["approve", "approve", "approve"])                    # review: the app's own run hands 12
    assert seen and set(seen) == {12}


# --- 3. pages per plot-record fetch ---------------------------------------------------------------------------------

def test_plot_record_fetches_page_only_when_the_run_allows_it(tmp_path, monkeypatch):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_pages")
    seen = []
    for name in ("observations", "observation_units"):
        original = getattr(ctx.client, name)

        def recording(*args, _original=original, _name=name, **kwargs):
            seen.append((_name, kwargs.get("paging", False), kwargs.get("max_pages", 1)))
            return _original(*args, **kwargs)

        monkeypatch.setattr(ctx.client, name, recording)
    srv.dispatch(ctx, "get_observations", {"study_db_id": "S1", "variable_db_id": "V1"})
    srv.dispatch(ctx, "get_observation_units", {"study_db_id": "S1"})
    assert seen == [("observations", False, 1), ("observation_units", False, 1)]                # one page, as before
    ctx.observation_max_pages = 3
    srv.dispatch(ctx, "get_observations", {"study_db_id": "S1", "variable_db_id": "V1"})
    srv.dispatch(ctx, "get_observation_units", {"study_db_id": "S1"})
    assert seen[2:] == [("observations", True, 3), ("observation_units", True, 3)]


def test_the_controller_gives_its_tools_the_runs_page_limit(tmp_path):
    from test_run import cfg

    default = ctl.Controller(cfg(tmp_path))._build_ctx(tmp_path / "o1")
    paged = ctl.Controller(cfg(tmp_path, max_observation_pages=10))._build_ctx(tmp_path / "o2")
    assert (default.observation_max_pages, paged.observation_max_pages) == (2, 10)                 # the app's own 2 (item 14)


def test_main_runs_every_question_with_the_budgets_and_writes_them_down(tmp_path, monkeypatch):
    import json

    seen = {}

    def fake_run(subs, *, base_config, **_kwargs):
        seen["config"] = base_config("How many plots does study S1 have?")
        return {"subquestions": [], "by_tier": {}, "invented": [], "cost_usd": 0.0}

    monkeypatch.setattr(bench, "run_subquestions", fake_run)
    monkeypatch.setattr(bench, "results_markdown", lambda report: "")
    cache = tmp_path / "cache"
    cache.mkdir()
    status = bench.main(["--cache", str(cache), "--results", str(tmp_path / "results"), "--only", "A35.2", "--confirm",
                         "--max-fetches", "24", "--max-pages", "10", "--max-plan-steps", "30"],
                        base_config=app.live_config, make_models=lambda: None, triage_model=lambda: None)
    config = seen["config"]
    assert (config.max_fetches, config.max_observation_pages, config.max_plan_steps, config.max_http_attempts) == (24, 10, 30, 240)
    [folder] = list((tmp_path / "results").iterdir())
    record = json.loads((folder / "results.json").read_text(encoding="utf-8"))["budgets"]
    assert record == {"max_fetches": 24, "max_observation_pages": 10, "max_plan_steps": 30, "max_http_attempts": 240} and status in (0, None)


def test_draft_plan_hands_its_limit_to_the_plan_check(tmp_path):
    import json

    from llm import Budget, FakeModelClient, reply_text

    draft = synthetic_plan_draft("S1", "V1")
    draft["steps"] += [{"step_id": f"step_{n}", "agent": "analyst", "action": "table_info", "depends_on": ["step_1"]} for n in range(5, 15)]
    meta = co.KnownMetadata.from_rows([{"studyDbId": "S1", "studyName": "SYNTHETIC trial"}],
                                      [{"observationVariableDbId": "V1", "observationVariableName": "fresh root yield"}],
                                      base_url=SYNTHETIC_BASE, complete=True)

    def plan(**kw):
        model = FakeModelClient([reply_text(json.dumps({"status": "completed", "payload": draft}))])
        return asyncio.run(co.draft_plan("What is the mean fresh root yield in study S1, per clone?", model=model, metadata=meta,
                                         base_url=SYNTHETIC_BASE, budget=Budget(), run_id="run_budget", log_dir=tmp_path, now=NOW,
                                         max_replans=0, **kw))

    longer = plan(max_plan_steps=30)
    assert longer.plan is not None and len(longer.plan.steps) == 14
    default = plan()
    assert default.plan is None and any("14 steps" in n and "at most 12" in n for n in default.notes), default.notes
