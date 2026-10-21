"""
tests/test_unfetchable_plans.py - the plan check sends back a READY plan whose fetch list cannot be built.

Benchmark baseline 2026-10-21: 23 of 149 failing sub-questions ended "Could not complete: unexpected error in state resolve",
because the fetch-list guard (contracts.FetchManifest) refused the plan there. Measurements need exact traits, and plot data
needs exact studies. The guard stays as it is. The plan check (agents.coordinator.assemble_plan) now builds the same fetch
list for a ready plan (plan_permissions.fetch_manifest, which run.Controller.resolve_manifest also uses) and sends a plan it
cannot hold back to the planner, with a reason it can act on.

For the data-plan shapes below, the plan check accepts a ready plan exactly when the controller's resolve step can build
its fetch list. A plan still waiting for the person's answer is never refused for its fetch list: the person answers first,
and the plan is made again (review 2026-10-21). Everything is SYNTHETIC.
"""
from __future__ import annotations

import itertools
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

import agents.coordinator as co
import run as ctl
from contracts import Plan, PlanStep, ResolvedScope
from part2_support import part2_config

BASE = "https://live.invalid/brapi/v2"
NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
META = co.KnownMetadata(base_url=BASE, complete=True, studies=[co.KnownStudy(study_id="S1", name="SYNTHETIC trial S1")],
                        variables=[co.KnownVariable(variable_id="V1", name="fresh root yield")])
# how the question names its studies: by ID; as a whole set whose filter matches S1, or matches nothing; or not at all
STUDIES = {"named": ["S1"], "whole set, match": ["S1"], "whole set, no match": [], "not named": []}
FILTER = {"whole set, match": "SYNTHETIC", "whole set, no match": "Genetic Gain"}


def _question(studies: str, trait: bool) -> str:
    where = {"named": "in study S1", "not named": "here"}.get(studies, f"across all studies named '{FILTER.get(studies)}'")
    return f"How many {'V1 values' if trait else 'plots'} are there {where}?"


def _steps(measurements: bool, roster: bool) -> list[dict]:
    steps = ([{"step_id": "step_1", "agent": "retriever", "action": "get_observations", "inputs": {}}] if measurements else []) + \
            ([{"step_id": "step_2", "agent": "retriever", "action": "get_observation_units", "inputs": {}}] if roster else [])
    return steps + [{"step_id": "step_3", "agent": "analyst", "action": "count_records", "depends_on": [s["step_id"] for s in steps]}]


def _draft(steps: list[dict], studies: str, trait: bool, statistic: dict) -> co.PlanDraft:
    scope = {"study_ids": ["S1"] if studies == "named" else [], "all_known_studies": studies.startswith("whole set"),
             "variable_ids": ["V1"] if trait else []}
    if studies in FILTER:
        scope["filters"] = {"name_contains": FILTER[studies]}
    return co.PlanDraft(interpretation="count", scope=co.DraftScope(**scope), steps=[co.DraftStep(**s) for s in steps], statistic=statistic)


def _fetch_list_holds(tmp_path, steps: list[dict], studies: list[str], variables: list[str]) -> bool:
    """The controller's own resolve step (run.Controller.resolve_manifest) on the same plan: does the guard accept it?"""
    controller = ctl.Controller(part2_config(question="q", offline=False, mock_model=False, direct=False, auto=False, fixture="synthetic",
                                             cache_dir=tmp_path / "cache", out_dir=tmp_path / "out"))
    controller.plan = Plan(plan_id="plan_shape", question="q", interpretation="count", status="ready",
                           scope=ResolvedScope(base_url=BASE, study_ids=studies, variable_ids=variables),
                           steps=[PlanStep(**s) for s in steps], created_at_utc=NOW)
    try:
        controller.resolve_manifest()
    except ValidationError:
        return False
    return True


COUNT = {"kind": "count", "denominator": "rows", "grouping": []}
SHAPES = [(m, r, s, t) for m, r, s, t in itertools.product((True, False), (True, False), STUDIES, (True, False)) if m or r]


@pytest.mark.parametrize("measurements, roster, studies, trait", SHAPES,
                         ids=[f"{'obs' if m else '-'}+{'roster' if r else '-'}|{s}|{'trait' if t else 'no trait'}" for m, r, s, t in SHAPES])
def test_a_ready_plan_is_accepted_exactly_when_its_fetch_list_can_be_built(tmp_path, measurements, roster, studies, trait):
    steps = _steps(measurements, roster)
    plan, problems, _notes = co.assemble_plan(_question(studies, trait), _draft(steps, studies, trait, COUNT), META, base_url=BASE, now=NOW,
                                              plan_id="plan_shape")
    holds = _fetch_list_holds(tmp_path, steps, STUDIES[studies], ["V1"] if trait else [])
    assert (plan is not None) == holds and (problems == []) == holds, problems
    if holds:
        assert plan.status == "ready" and plan.scope.study_ids == STUDIES[studies]
    elif not STUDIES[studies]:                                 # no study: the reason says why there is none
        reason = ("(filters {'name_contains': 'Genetic Gain'}) matches none of the 1 known studies" if studies == "whole set, no match"
                  else "the plan names no study (a question about all studies sets all_known_studies, narrowed by its filters)")
        assert problems[0].startswith(f"{steps[0]['step_id']}: {steps[0]['action']} fetches plot data of exact studies") and reason in problems[0]
    else:                                                      # a study, but measurements with no trait
        assert problems[0].startswith("step_1: get_observations fetches the measurements of exact traits, but scope.variable_ids is empty")


@pytest.mark.parametrize("studies", ["named", "whole set, no match", "not named"])
def test_a_plan_waiting_for_an_answer_is_never_refused_for_its_fetch_list(tmp_path, studies):
    """A mean with no trait: code asks which trait is meant (a required clarification), so the plan is not ready and never
    reaches resolve, even though its fetch list could not be built yet."""
    steps = _steps(True, False)
    mean = {"kind": "mean", "denominator": "valid values", "grouping": []}
    plan, problems, _notes = co.assemble_plan(f"What is the mean starch content {_question(studies, False).split(' are there ')[1]}",
                                              _draft(steps, studies, False, mean), META, base_url=BASE, now=NOW, plan_id="plan_shape")
    assert not _fetch_list_holds(tmp_path, steps, STUDIES[studies], [])
    assert problems == [] and plan is not None and plan.status == "needs_clarification"
    assert any("names no known trait" in c.question for c in plan.clarifications)


@pytest.mark.parametrize("scope, said", [
    ({"season_ids": ["2019"], "filters": {"name_contains": "SYNTHETIC"}}, "(filters {'name_contains': 'SYNTHETIC'}, season_ids ['2019'])"),
    ({"location_ids": ["L2"], "filters": {"name_contains": "SYNTHETIC"}}, "(filters {'name_contains': 'SYNTHETIC'}, location_ids ['L2'])"),
    ({"filters": {"location_name": "Ibadan", "study_type": "Advanced Yield Trial"}}, "(filters {'study_type': 'Advanced Yield Trial'})"),
], ids=["a season", "a location", "a name filter that narrows nothing"])
def test_matches_none_names_every_input_that_narrowed_the_studies(scope, said):
    """Reviews 2 and 3 of 2026-10-21: a name filter that alone matches S1, combined with a season or location S1 does not have,
    matched nothing; the reason named only the filter, which then looked false. It names the filters, locations and seasons
    that narrowed the list together, and never location_name or season_name, which _study_matches does not apply."""
    meta = META.model_copy(update={"studies": [co.KnownStudy(study_id="S1", name="SYNTHETIC trial S1", location_id="L1", seasons=["2026"],
                                                              study_type="Synthetic Yield Trial")],
                                   "location_ids": ["L1", "L2"], "season_ids": ["2019", "2026"]})
    draft = co.PlanDraft(interpretation="count", statistic=COUNT, scope=co.DraftScope(all_known_studies=True, variable_ids=["V1"], **scope),
                         steps=[co.DraftStep(**s) for s in _steps(True, False)])
    plan, problems, _notes = co.assemble_plan("How many V1 values are there across all studies of that kind?", draft, meta,
                                              base_url=BASE, now=NOW, plan_id="plan_shape")
    assert plan is None and len(problems) == 1
    assert f"{said} matches none of the 1 known studies" in problems[0] and "Ibadan" not in problems[0]


def _plan(steps: list[tuple[str, dict]], studies: list[str], variables: list[str]) -> Plan:
    return Plan(plan_id="plan_pin", question="q", interpretation="count", status="ready",
                scope=ResolvedScope(base_url=BASE, study_ids=studies, variable_ids=variables), created_at_utc=NOW,
                steps=[PlanStep(step_id=f"step_{i}", agent="retriever", action=a, inputs=inputs) for i, (a, inputs) in enumerate(steps, 1)])


@pytest.mark.parametrize("steps, studies, variables, families, obs_studies, obs_variables, most", [
    ([("get_observations", {}), ("get_observation_units", {})], ["S1", "S2"], ["V1"],
     ["studies", "observationvariables", "observations", "observationunits"], ["S1", "S2"], ["V1"], 2),
    ([("get_observation_units", {})], ["S1"], ["V1"], ["studies", "observationvariables", "observationunits"], ["S1"], [], 1),
    ([("get_observation_units", {"study_db_id": "S2"})], ["S1", "S2"], [], ["studies", "observationunits"], ["S1", "S2"], [], 2),
    ([("export_metadata", {"entity": "studies"})], [], [], ["studies"], [], [], 0),
    ([("get_study", {"study_db_id": "S1"})], ["S1"], [], ["studies"], [], [], 0),
], ids=["measurements and plot lists", "plot list only: no trait listed", "one explicit plot list", "catalog export only",
        "a study record only: no plot data listed"])
def test_the_fetch_list_builder_is_pinned_by_hand_written_values(steps, studies, variables, families, obs_studies, obs_variables, most):
    """Review 2 of 2026-10-21: comparing the plan check with resolve_manifest is circular now that both call fetch_manifest.
    These values are written out by hand from the pre-change run.Controller.resolve_manifest, so a change to the shared
    builder (listing a trait without measurements, a wrong study ceiling) fails here."""
    manifest = co.fetch_manifest(_plan(steps, studies, variables), run_id="run_pin", base_url="https://other.invalid/brapi/v2",
                                 max_http_attempts=7)
    assert manifest.endpoint_families == families and manifest.observation_study_ids == obs_studies
    assert manifest.observation_variable_ids == obs_variables and manifest.max_observation_studies == most
    assert manifest.max_http_attempts == 7 and manifest.origin == "live" and manifest.run_id == "run_pin"
    assert manifest.base_url == "https://other.invalid/brapi/v2"           # the server it is given, not the plan's (review 3)


@pytest.mark.parametrize("steps, studies, variables, error", [
    ([("get_observations", {})], ["S1"], [], ValidationError),
    ([("get_observations", {}), ("get_observation_units", {})], ["S1"], [], ValidationError),
    ([("get_observations", {})], [], ["V1"], ValidationError),
    ([("get_observation_units", {})], [], [], ValidationError),
    ([("export_metadata", {})], [], [], ValueError),
], ids=["measurements, no trait", "measurements and plot list, no trait", "measurements, no study", "plot list, no study",
        "export without an entity"])
def test_the_fetch_list_builder_refuses_what_the_guard_refuses(steps, studies, variables, error):
    """Review 3 of 2026-10-21: the shape test compares two callers of the same builder, so a builder that accepted these plans
    would pass it. Written out by hand: these plans cannot be fetched (FetchManifest, or the families of an export step)."""
    with pytest.raises(error):
        co.fetch_manifest(_plan(steps, studies, variables), run_id="run_pin", base_url=BASE, max_http_attempts=1)


def test_an_unforeseen_refusal_gets_the_guards_own_words():
    """The two plain reasons cover every refusal a valid plan can meet today; anything else is passed on as the guard says it."""
    plan = _plan([("get_observations", {})], ["S1"], ["V1"])
    with pytest.raises(ValidationError) as caught:
        ctl.FetchManifest(run_id="run_pin", base_url=BASE, origin="snapshot", endpoint_families=["observations"],
                          observation_study_ids=["S1"], observation_variable_ids=["V1"], max_http_attempts=1, max_observation_studies=1)
    reason = co._unfetchable(plan, caught.value, filled_from_filters=False, known=1, narrowing={})
    assert reason.startswith("the fetch list cannot hold this plan:") and "snapshot" in reason
