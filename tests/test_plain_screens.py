"""
tests/test_plain_screens.py — every approval, review and question screen starts with one plain sentence.

What these tests prove, in plain words:
* The first line is one plain sentence, written by code from the request or from the hash-checked tables:
  no technical words, no markdown, and its counts match the request (studies, traits, tables, rows).
* Every fact the screen showed before is still below it: source, allowed data, limits, IDs and row counts.
* "APPROVAL NEEDED" still appears exactly once on each approval screen.
* The four question prompts speak as "I" instead of naming the planner, coordinator, retriever or analyst.
Synthetic objects only: no model and no network.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import app
import chat_presenter as cp
import run as ctl
from artifacts import ArtifactRegistry
from contracts import FetchManifest, MeasurementMeta, Plan, PlanStep, ResolvedScope, RetrievalReport
from coordination_runtime import SUPERVISOR_PROMPT
from retrieval_review import review_snapshot

BASE = "https://plain-test.invalid/brapi/v2"
TECHNICAL = re.compile(r"HTTP|manifest|sha256|endpoint|observationvariables|observationunits|BrAPI|APPROVAL|CATALOG|[*#|=`_]")


def plain_first_line(text: str) -> str:
    line = text.split("\n", 1)[0]
    assert line.endswith(".") and not TECHNICAL.search(line), line
    return line


def data_request(*, observations: bool = True, units: bool = False, studies: int = 3) -> tuple[Plan, FetchManifest]:
    ids = [f"S{i}" for i in range(1, studies + 1)]
    steps = []
    if observations:
        steps.append(PlanStep(step_id="step_1", agent="retriever", action="get_observations"))
    if units:
        steps.append(PlanStep(step_id="step_2", agent="retriever", action="get_observation_units"))
    if not steps:
        steps.append(PlanStep(step_id="step_1", agent="retriever", action="export_metadata", inputs={"entity": "studies"}))
    measured = observations or units
    plan = Plan(plan_id="plan_plain", question="Which studies have yield?", interpretation="Review studies", status="ready",
                scope=ResolvedScope(base_url=BASE, study_ids=ids, variable_ids=["V1"] if observations else []),
                steps=steps, created_at_utc=datetime.now(timezone.utc))
    families = ["studies", "observationvariables", *(["observations"] if observations else []), *(["observationunits"] if units else [])]
    manifest = FetchManifest(run_id="run_plain", base_url=BASE, origin="live", endpoint_families=families,
                             observation_study_ids=ids if measured else [], observation_variable_ids=["V1"] if observations else [],
                             max_http_attempts=20, max_observation_studies=len(ids) if measured else 0)
    return plan, manifest


@pytest.mark.parametrize("request_kind, lead", [
    (dict(), "I plan to fetch measurements of 1 trait in 3 studies from plain-test.invalid. Nothing is fetched until you approve."),
    (dict(studies=1), "I plan to fetch measurements of 1 trait in 1 study from plain-test.invalid. Nothing is fetched until you approve."),
    (dict(units=True, studies=1), "I plan to fetch measurements of 1 trait in 1 study and the plot list of 1 study from plain-test.invalid. "
                                  "Nothing is fetched until you approve."),
    (dict(observations=False, units=True, studies=1),
     "I plan to fetch the plot list of 1 study from plain-test.invalid. Nothing is fetched until you approve."),
    (dict(observations=False, units=True, studies=2),
     "I plan to fetch the plot lists of 2 studies from plain-test.invalid. Nothing is fetched until you approve."),
    (dict(observations=False),
     "This question needs only public catalog records of studies and traits from plain-test.invalid, no measurements. "
     "Nothing is requested until you approve."),
])
def test_the_chat_approval_screen_starts_plain_and_keeps_every_fact(request_kind, lead):
    plan, manifest = data_request(**request_kind)
    view = SimpleNamespace(plan=plan, manifest=manifest, metadata=None, config=SimpleNamespace(question=plan.question, max_fetches=5),
                           budget=SimpleNamespace(max_model_calls=30), catalog_fetched_at="2026-09-29T12:00:00+00:00", catalog_setup=None)
    shown = cp.approval_summary(view)
    assert plain_first_line(shown) == lead
    for fact in ("**Review this plan before approving**", f"Source: {BASE}.", "previously cached; fetched 2026-09-29T12:00:00+00:00",
                 "Allowed data: " + ", ".join(manifest.endpoint_families) + ".",
                 f"Limits: 20 HTTP attempts, {manifest.max_observation_studies} observation studies, 5 data fetch calls, and 30 model calls.",
                 "does not accept the eventual answer"):
        assert fact in shown
    for study_id in manifest.observation_study_ids:
        assert f"| {study_id} |" in shown


def test_the_full_approval_screen_starts_plain_and_keeps_every_fact(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAPI_BASE_URL", BASE)
    plan, manifest = data_request()
    controller = ctl.Controller(ctl.RunConfig(question=plan.question, out_dir=tmp_path / "out"))
    controller.plan, controller.manifest, controller.catalog_fetched_at = plan, manifest, "2026-09-29T12:00:00+00:00"
    screen = controller.approval_screen()
    assert plain_first_line(screen) == ("I plan to fetch measurements of 1 trait in 3 studies from plain-test.invalid. "
                                        "Nothing is fetched until you approve.")
    assert screen.count("APPROVAL NEEDED") == 1
    for fact in ("APPROVAL NEEDED - no observation data has been fetched yet", f"source server  : {BASE}",
                 "database access: studies, observationvariables, observations", "study IDs      : ['S1', 'S2', 'S3']",
                 "max 20 HTTP attempts", f"manifest sha256: {manifest.sha256()}", "cached copy fetched 2026-09-29T12:00:00+00:00"):
        assert fact in screen


def test_the_catalog_setup_screen_starts_plain_and_keeps_every_fact(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAPI_BASE_URL", BASE)
    controller = ctl.Controller(ctl.RunConfig(question="Which studies have a 2026 season?", out_dir=tmp_path / "out"))
    screen = controller.catalog_approval_screen()
    assert plain_first_line(screen) == ("First I need the public list of studies and traits from plain-test.invalid. "
                                        "No measurements are fetched in this step.")
    assert screen.count("APPROVAL NEEDED") == 1
    for fact in ("CATALOG SETUP APPROVAL NEEDED — complete cached catalogs are missing.", f"Source: {BASE}",
                 "at most 20 HTTP attempts in total and 20 pages per catalog", "No observation data or model calls",
                 "does not grant approval for the later question", "public-data cache"):
        assert fact in screen
    content, finished = app.render(SimpleNamespace(controller=controller), ("prompt", ctl.CATALOG_PROMPT))
    assert content.startswith(screen) and "Reply **approve** to prepare these catalogs" in content and not finished
    assert not (tmp_path / "out").exists()


@pytest.fixture
def fetched(tmp_path):
    registry = ArtifactRegistry(tmp_path / "out", "run_plain", base_url=BASE)
    observations = [{"studyDbId": "S1", "observationUnitDbId": f"P{i}", "observationVariableDbId": "V1", "value": str(i)}
                    for i in range(1, 7)]
    plots = [{"studyDbId": "S1", "observationUnitDbId": f"P{i}"} for i in range(1, 4)]
    tables = [registry.save_table(observations, kind="observations", columns={key: "synthetic" for key in observations[0]},
                                  complete=True, study_ids=["S1"], variable_ids=["V1"]),
              registry.save_table(plots, kind="observation_units", columns={key: "synthetic" for key in plots[0]},
                                  complete=True, study_ids=["S1"])]
    plan, _ = data_request(studies=1)
    ctx = SimpleNamespace(registry=registry, variable_measurements={"V1": MeasurementMeta(trait="root yield", unit="t/ha")})
    return SimpleNamespace(ctx=ctx, plan=plan, tables=tables)


@pytest.mark.parametrize("count, lead", [
    (1, "Your data are ready: 1 table with 6 rows from plain-test.invalid. Nothing is analyzed until you check it."),
    (2, "Your data are ready: 2 tables with 9 rows in total from plain-test.invalid. Nothing is analyzed until you check them."),
])
def test_the_data_review_starts_plain_and_keeps_every_fact(fetched, count, lead):
    report = RetrievalReport(status="completed", complete=True, artifacts=fetched.tables[:count])
    markdown, fingerprints = review_snapshot(fetched.ctx, report, fetched.plan, [{"origin": "live"}, {"origin": "cache"}])
    assert plain_first_line(markdown) == lead
    assert sum(f["row_count"] for f in fingerprints) == (6 if count == 1 else 9)       # the counts come from the checked tables
    for fact in ("## Review the retrieved data", "Analysis has not started", f"**Source:** {BASE}", "**Studies:** S1",
                 "**HTTP attempts:** 1 · **Cached responses:** 1", "observations · 6 rows", "This sample is not the full table.",
                 "does not accept an answer or authorize additional retrieval"):
        assert fact in markdown


def waiting_controller() -> SimpleNamespace:
    return SimpleNamespace(
        coordination=SimpleNamespace(questions=["Which study is meant?"]), metadata=None, plan=None, approval_revisions=0,
        config=SimpleNamespace(question="Tell me about a study.", max_clarification_rounds=3),
        analysis=SimpleNamespace(caveats=["clarification needed: Use the seasons column instead?"]), analyst_answers=0,
        retriever_questions=["Which label should be used?"], retriever_answers=0,
        supervisor_questions=["Should the plan include 2025?"])


@pytest.mark.parametrize("prompt, lead, kept", [
    (ctl.CLARIFY_PROMPT, "I need one more detail before I can make a plan (answer 1 of 3):",
     ["Which study is meant?", "Your answer is added to your question", "reply **cancel**"]),
    (ctl.ANALYST_PROMPT, "I have the data and need one more detail before I can answer (answer 1 of 3):",
     ["Use the seasons column instead?", "nothing new is fetched", "reply **cancel**"]),
    (ctl.RETRIEVER_PROMPT, "I need one more detail while preparing your data (answer 1 of 3):",
     ["Which label should be used?", "within the approved scope", "**cancel** and start a new question"]),
    (SUPERVISOR_PROMPT, "I need one more detail before choosing the next step:",
     ["Should the plan include 2025?", "still needs your approval"]),
])
def test_the_question_prompts_speak_plainly_as_i(prompt, lead, kept):
    content, finished = app.render(SimpleNamespace(controller=waiting_controller()), ("prompt", prompt))
    assert content.split("\n", 1)[0] == lead and not finished
    for words in kept:
        assert words in content
    assert not re.search(r"\b(coordinator|planner|retriever|analyst)\b", content, re.I)
