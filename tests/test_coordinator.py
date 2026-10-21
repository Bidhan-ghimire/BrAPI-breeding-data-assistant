"""
tests/test_coordinator.py — the Coordinator drafts plans with NO tools, from validated metadata only; code
assembles and checks them (IDs from metadata, counts computed or unknown, only needed clarifications, bounded
re-plans); the numeric table is rendered deterministically from typed claims; prose must cite the claims it
repeats and is removed otherwise.

Fake model replies are built here; the guide lists no fixture folder for this task. Everything is SYNTHETIC.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import anyio
import pytest
from pydantic import ValidationError

import agents.coordinator as co
import brapi_mcp_server as srv
from contracts import AnalysisReport, ArtifactManifest, Claim, Denominator, Exclusion, Plan, RetrievalReport, StudyStatusEntry
from part2_support import SYNTHETIC_BASE
from llm import Budget, FakeModelClient, reply_text, reply_tools
from part2_support import synthetic_context

NOW = datetime(2026, 9, 27, 13, 0, 0, tzinfo=timezone.utc)
Q_AB = "Give the mean fresh root yield per clone in synthetic study S1, with n."


# --- helpers -------------------------------------------------------------------------------

@pytest.fixture
def metadata(tmp_path) -> co.KnownMetadata:
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_meta")
    studies = srv.dispatch(ctx, "export_metadata", {"entity": "studies"}).artifact_ids[0]
    variables = srv.dispatch(ctx, "export_metadata", {"entity": "variables"}).artifact_ids[0]
    return co.KnownMetadata.from_registry(ctx.registry, base_url=SYNTHETIC_BASE, studies_artifact=studies, variables_artifact=variables)


def draft(**overrides) -> dict:
    base = {
        "interpretation": "pooled and per-clone descriptive means of fresh root yield (V1) in study S1",
        "clarifications": [],
        "scope": {"study_ids": ["S1"], "variable_ids": ["V1"]},
        "statistic": {"kind": "mean", "denominator": "valid plot-level values", "grouping": ["germplasmDbId"]},
        "steps": [
            {"step_id": "step_1", "agent": "retriever", "action": "get_observations", "inputs": {"study_db_id": "S1", "variable_db_id": "V1"}},
            {"step_id": "step_2", "agent": "retriever", "action": "get_observation_units", "inputs": {"study_db_id": "S1"}},
            {"step_id": "step_3", "agent": "analyst", "action": "group_stats", "inputs": {"by": "germplasmDbId"}, "depends_on": ["step_1", "step_2"]},
        ],
    }
    base.update(overrides)
    return base


def completed(payload: dict):
    return reply_text(json.dumps({"status": "completed", "payload": payload}), model="mock-model")


def clarify(question: str):
    return reply_text(json.dumps({"status": "needs_clarification", "question": question}), model="mock-model")


def fake(*replies) -> FakeModelClient:
    return FakeModelClient(list(replies), reported_model="mock-model")


def plan_with(tmp_path, question: str, model: FakeModelClient, metadata: co.KnownMetadata | None, **kw) -> co.CoordinatorResult:
    async def go():
        return await co.draft_plan(question, model=model, metadata=metadata, base_url=SYNTHETIC_BASE, budget=Budget(), run_id="run_co",
                                   log_dir=tmp_path / "logs", now=NOW, **kw)

    return anyio.run(go)


# --- 1. validated metadata and what the question points at ---------------------------------------

def test_metadata_is_built_from_the_exports_not_from_a_model(metadata):
    assert metadata.complete is True and metadata.base_url == SYNTHETIC_BASE
    assert [(s.study_id, s.name, s.location_id, s.seasons) for s in metadata.studies] == [("S1", "SYNTHETIC yield trial S1", "L1", ["2026"])]
    assert [(v.variable_id, v.name, v.unit) for v in metadata.variables] == [("V1", "fresh root yield", "t/ha"), ("V2", "dry matter content", "%")]
    assert metadata.location_ids == ["L1"] and metadata.season_ids == ["2026"]
    assert "S1 'SYNTHETIC yield trial S1'" in metadata.summary_for_model() and "V2 'dry matter content' unit=%" in metadata.summary_for_model()


def test_a_real_size_catalog_is_summarized_to_what_the_question_names(metadata):
    """SweetPotatoBase has 2,029 studies and 486 variables; listing them all is ~40k tokens per planning call, more than a local
    model can read. With a question, the summary lists only the named studies, the best-matching variables, the mentioned
    locations and the totals; without a question, or for a small catalog, the whole listing stays as before."""
    studies = [co.KnownStudy(study_id=str(100000 + i), name=f"trial_{i:04d}", location_id=str(i % 40), location_name=f"Site {i % 40}" if i % 40 else "Namulonge-NaCRRI",
                             seasons=[str(2000 + i % 25)], study_type="Advanced Yield Trial") for i in range(2029)]
    studies[491] = studies[491].model_copy(update={"name": "07DISE189HFLNO7", "location_id": "61", "location_name": "Horticultural Field Laboratory"})
    variables = [co.KnownVariable(variable_id=str(70000 + i), name=f"Reaction to disease number {i} estimating 1-9|CO_331:{i:07d}", trait=f"disease {i}") for i in range(486)]
    variables[36] = variables[36].model_copy(update={"name": "Reaction to Fusarium Wilt estimating 0-5|CO_331:0000789", "trait": "Reaction to Fusarium Wilt estimating 0-5"})
    big = co.KnownMetadata(base_url=SYNTHETIC_BASE, complete=True, studies=studies, variables=variables,
                           location_ids=sorted({s.location_id for s in studies}), season_ids=sorted({x for s in studies for x in s.seasons}))
    question = "In study 07DISE189HFLNO7, what is the pooled mean of the variable 'Reaction to Fusarium Wilt estimating 0-5', with valid-value and independent-unit counts?"
    full, short = big.summary_for_model(), big.summary_for_model(question)
    assert len(full) > 120_000 and len(short) < 3_000                                   # the whole catalog vs. what the question names
    assert "2029 studies, 486 variables" in short and "1 of 2029" in short and "100491 '07DISE189HFLNO7'" in short
    assert "70036 'Reaction to Fusarium Wilt estimating 0-5|CO_331:0000789'" in short.split("Variables whose names")[1].split(";")[0]   # best match first
    assert short.count("Reaction to disease number") <= co.SUMMARY_MAX_LISTED - 1 and "100000 'trial_0000'" not in short
    assert "trial_0001" not in short and "Locations the question mentions (0 of" in short and f"Seasons: {big.season_ids}" in short
    # a location the question mentions is listed with its ID; a whole-set question names no study and says what to do instead
    whole = big.summary_for_model("How many studies have the resolved Namulonge-NaCRRI location and the resolved 2018 season?")
    assert "Studies the question names (0 of 2029): none" in whole and "all_known_studies=true" in whole and "0 'Namulonge-NaCRRI'" in whole
    # the small synthetic catalog is still listed whole, question or not, so every existing behaviour is unchanged
    assert metadata.summary_for_model(Q_AB) == metadata.summary_for_model() and "Studies: S1 'SYNTHETIC yield trial S1'" in metadata.summary_for_model(Q_AB)
    assert co._user_message(question, big, []).count("Reaction to disease number") <= co.SUMMARY_MAX_LISTED - 1


def test_candidates_come_from_literal_matching(metadata):
    c = co.find_candidates(Q_AB, metadata)
    assert [s.study_id for s in c.studies] == ["S1"] and [v.variable_id for v in c.variables] == ["V1"] and c.decision_words == []
    c = co.find_candidates("Which trial has the best yield?", metadata)
    assert c.studies == [] and [v.variable_id for v in c.variables] == ["V1"] and c.decision_words == ["best"]
    assert [v.variable_id for v in co.find_candidates("dry matter for S1", metadata).variables] == ["V2"]
    assert [v.variable_id for v in co.find_candidates("Compare yield and dry matter in S1", metadata).variables] == ["V1", "V2"]
    assert co.find_candidates("mean starch content in S1", metadata).variables == []
    assert [v.variable_id for v in co.find_candidates("mean of V2 in S1", metadata).variables] == ["V2"]     # exact ID also counts
    assert co.find_candidates(Q_AB, None).studies == []


# --- 2. drafting: no tools, IDs from metadata, counts by code -------------------------------------

def test_valid_draft_becomes_a_ready_plan_with_code_counted_scope(tmp_path, metadata):
    model = fake(completed(draft()))
    result = plan_with(tmp_path, Q_AB, model, metadata)
    assert result.status == "completed" and result.attempts == 1 and result.questions == []
    plan = result.plan
    assert isinstance(plan, Plan) and plan.status == "ready" and plan.plan_id.startswith("plan_run_co_")
    assert plan.scope.study_ids == ["S1"] and plan.scope.variable_ids == ["V1"] and plan.scope.matching_study_count == 1
    assert [s.step_id for s in plan.steps] == ["step_1", "step_2", "step_3"] and plan.statistic.denominator == "valid plot-level values"
    assert model.requests[0]["tools"] == []                                                # nothing to call, ever
    assert any("counted by code" in n for n in result.notes)
    assert result.agents[0].log_path == "run_co/llm_coordinator_1.jsonl"


def test_a_tool_request_is_refused_and_the_plan_still_arrives(tmp_path, metadata):
    model = fake(reply_tools([("search_studies", "{}")]), completed(draft()))
    result = plan_with(tmp_path, Q_AB, model, metadata)
    assert result.status == "completed" and result.agents[0].tool_calls == 1
    tool_msgs = [m for m in result.agents[0].messages if m.role == "tool"]
    assert len(tool_msgs) == 1 and json.loads(tool_msgs[0].content)["error"]["code"] == "invalid_argument"


def test_model_estimates_and_old_fields_are_refused_then_replanned(tmp_path, metadata):
    with_count = draft(scope={"study_ids": ["S1"], "variable_ids": ["V1"], "matching_study_count": 6})
    model = fake(completed(with_count), completed(draft(estimated_fetches=3)), completed(draft()))
    result = plan_with(tmp_path, Q_AB, model, metadata)
    assert result.status == "completed" and result.attempts == 3
    assert any("matching_study_count" in n for n in result.notes) and any("estimated_fetches" in n for n in result.notes)
    assert result.plan.scope.matching_study_count == 1
    second_prompt = model.requests[1]["messages"][-1]["content"]
    assert "rejected" in second_prompt and "matching_study_count" in second_prompt          # the reason went back to the model


def test_invented_ids_are_sent_back_and_bounded(tmp_path, metadata):
    bad = draft(scope={"study_ids": ["S9"], "variable_ids": ["V1"]},
                steps=[{"step_id": "step_1", "agent": "retriever", "action": "get_observations", "inputs": {"study_db_id": "S9", "variable_db_id": "V1"}}])
    model = fake(completed(bad), completed(draft()))
    result = plan_with(tmp_path, Q_AB, model, metadata)
    assert result.status == "completed" and result.attempts == 2
    assert any("not in the validated metadata: ['S9']" in n for n in result.notes)
    stubborn = fake(completed(bad), completed(bad), completed(bad), completed(draft()))
    result = plan_with(tmp_path, Q_AB, stubborn, metadata)
    assert result.status == "failed" and result.plan is None and result.attempts == 3
    assert any("no acceptable plan after 3 attempts" in n for n in result.notes) and len(stubborn._replies) == 1   # the fourth reply was never asked for
    result = plan_with(tmp_path, Q_AB, fake(completed(bad), completed(bad), completed(draft())), metadata, max_replans=1)
    assert result.status == "failed" and result.attempts == 2


def test_steps_are_checked_against_scope_and_agent_menus(tmp_path, metadata):
    off_scope = draft(steps=[{"step_id": "step_1", "agent": "retriever", "action": "get_observations", "inputs": {"study_db_id": "S2", "variable_db_id": "V1"}}])
    wrong_menu = draft(steps=[{"step_id": "step_1", "agent": "analyst", "action": "get_observations", "inputs": {"study_db_id": "S1", "variable_db_id": "V1"}}])
    orphan = draft(steps=[{"step_id": "step_1", "agent": "analyst", "action": "numeric_summary"}])
    model = fake(completed(off_scope), completed(wrong_menu), completed(orphan))
    result = plan_with(tmp_path, Q_AB, model, metadata, max_replans=2)
    assert result.status == "failed"
    notes = "\n".join(result.notes)
    assert "outside the plan's scope" in notes and "is not a analyst action" in notes and "plan contract rejected" in notes


# Benchmark baseline 2026-10-21: 23 of 149 failing sub-questions crashed at resolve ("unexpected error ... FetchManifest"):
# the plan asked for measurements with no trait, or for plot data with no study. The plan check now builds the same fetch list
# (plan_permissions.fetch_manifest) for a READY plan and sends one it cannot hold back to the planner.
PLOTS_Q = "How many plots does synthetic study S1 have, and how many for each clone?"
ROSTER = {"step_id": "step_2", "agent": "retriever", "action": "get_observation_units", "inputs": {"study_db_id": "S1"}}
COUNT_PLOTS = {"kind": "count", "denominator": "plots in the plot list", "grouping": ["germplasmDbId"]}
PER_CLONE = {"step_id": "step_3", "agent": "analyst", "action": "count_records", "inputs": {"group_by": "germplasmDbId"}}


def test_measurements_without_a_trait_are_sent_back_with_what_to_plan_instead(tmp_path, metadata):
    """A plot question planned as get_observations with no trait: the fetch list holds measurements only of exact traits.
    The planner is told what it can do instead (the plot list alone; the trait the question names; or ask) and re-plans.
    The reason promises no replicate or block column: the plot list has none (review 2026-10-21)."""
    no_trait = draft(scope={"study_ids": ["S1"], "variable_ids": []}, statistic=COUNT_PLOTS,
                     steps=[{"step_id": "step_1", "agent": "retriever", "action": "get_observations", "inputs": {"study_db_id": "S1"}}, ROSTER,
                            {**PER_CLONE, "depends_on": ["step_1", "step_2"]}])
    roster_only = draft(scope={"study_ids": ["S1"], "variable_ids": []}, statistic=COUNT_PLOTS, steps=[ROSTER, {**PER_CLONE, "depends_on": ["step_2"]}])
    model = fake(completed(no_trait), completed(roster_only))
    result = plan_with(tmp_path, PLOTS_Q, model, metadata)
    assert result.status == "completed" and result.attempts == 2 and result.plan.status == "ready"
    assert [s.action for s in result.plan.steps] == ["get_observation_units", "count_records"]
    told = model.requests[1]["messages"][-1]["content"]
    assert "step_1: get_observations fetches the measurements of exact traits, but scope.variable_ids is empty" in told
    assert "plan get_observation_units alone, then count_records on that plot list (rule 7)" in told and "ask which trait is meant" in told
    assert "replicate" not in told and "block" not in told
    assert "get_observation_units alone" in co.COORDINATOR_SYSTEM_PROMPT        # rule 7 says the same (review 2 of 2026-10-21)
    # review 3: count_records with group_by states plots per clone and the clone count, never the plot total; the recipe asks for both
    assert "then count_records() for the number of plots and" in co.COORDINATOR_SYSTEM_PROMPT


def test_a_plan_waiting_for_the_persons_answer_is_not_refused_for_its_fetch_list(tmp_path, metadata):
    """Review 2026-10-21: a measurements plan with no trait that ASKS which trait is meant never reaches the fetch list (the
    person answers first, then the plan is made again). Refusing it turned an honest question into a failed run."""
    question = "What is the mean starch content in synthetic study S1?"
    asking = draft(scope={"study_ids": ["S1"], "variable_ids": []},
                   clarifications=[{"question": "Which variable is starch content?", "required": True}],
                   steps=[{"step_id": "step_1", "agent": "retriever", "action": "get_observations", "inputs": {"study_db_id": "S1"}},
                          {"step_id": "step_2", "agent": "analyst", "action": "numeric_summary", "depends_on": ["step_1"]}])
    result = plan_with(tmp_path, question, fake(completed(asking)), metadata)
    assert result.status == "needs_clarification" and result.attempts == 1 and result.plan.status == "needs_clarification"
    assert result.questions and not any("fetches the measurements" in n for n in result.notes)


def test_plot_data_with_no_study_is_sent_back_and_a_question_is_the_way_out(tmp_path, metadata):
    """'All studies in folder X': the folder's name matched no study name, so the measurements step had no study at all."""
    question = "Across all studies of the folder 'Genetic Gain 2016-19', how many V1 values are there?"
    none_match = draft(scope={"study_ids": [], "all_known_studies": True, "variable_ids": ["V1"], "filters": {"name_contains": "Genetic Gain"}},
                       statistic={"kind": "count", "denominator": "valid values", "grouping": []},
                       steps=[{"step_id": "step_1", "agent": "retriever", "action": "get_observations", "inputs": {"variable_db_id": "V1"}},
                              {"step_id": "step_2", "agent": "analyst", "action": "count_records", "depends_on": ["step_1"]}])
    model = fake(completed(none_match), clarify("No known study name contains 'Genetic Gain'. Which studies does that folder hold?"))
    result = plan_with(tmp_path, question, model, metadata)
    assert result.status == "needs_clarification" and result.attempts == 2 and "Genetic Gain" in result.questions[0]
    told = model.requests[1]["messages"][-1]["content"]
    assert "step_1: get_observations fetches plot data of exact studies" in told and "matches none of the 1 known studies" in told
    assert "(filters {'name_contains': 'Genetic Gain'})" in told                         # the narrowing that matched nothing is named


def test_no_match_is_said_only_when_code_looked_for_matching_studies(tmp_path, metadata):
    """Review 2026-10-21: when another rule refuses the all-studies switch (here: the question names S1), code never looked
    for matching studies, so 'the filters match none' would be false. Only that rule's reason goes back."""
    switched = draft(scope={"study_ids": [], "all_known_studies": True, "variable_ids": ["V1"]},
                     steps=[{"step_id": "step_1", "agent": "retriever", "action": "get_observations", "inputs": {"variable_db_id": "V1"}},
                            {"step_id": "step_2", "agent": "analyst", "action": "numeric_summary", "depends_on": ["step_1"]}])
    result = plan_with(tmp_path, Q_AB, fake(completed(switched), completed(draft())), metadata)
    assert result.status == "completed" and result.attempts == 2
    assert any("all_known_studies must be false when a study is named" in n for n in result.notes)
    assert not any("match none" in n or "names no study" in n for n in result.notes)


def test_a_variable_given_as_a_study_input_is_outside_the_scope(tmp_path, metadata):
    """Review 2026-10-21: the scope check accepted any ID of the scope for either input, so get_observation_units(study_db_id="V1")
    passed and its fetch list held no study. A study input must be a study of the scope, a variable input a variable of it."""
    crossed = draft(steps=[{"step_id": "step_1", "agent": "retriever", "action": "get_observations", "inputs": {"study_db_id": "S1", "variable_db_id": "V1"}},
                           {"step_id": "step_2", "agent": "retriever", "action": "get_observation_units", "inputs": {"study_db_id": "V1"}},
                           {"step_id": "step_3", "agent": "analyst", "action": "group_stats", "inputs": {"by": "germplasmDbId"}, "depends_on": ["step_1", "step_2"]}])
    result = plan_with(tmp_path, Q_AB, fake(completed(crossed), completed(draft())), metadata)
    assert result.status == "completed" and result.attempts == 2
    assert any("step_2: input study_db_id='V1' is outside the plan's scope" in n for n in result.notes)


def test_an_export_without_a_known_entity_is_sent_back(tmp_path, metadata):
    """An export_metadata step whose entity is missing failed at resolve too (endpoint_families_for_plan); now it goes back."""
    question = "How many studies are there per location?"
    no_entity = draft(interpretation="count studies per location", scope={"study_ids": [], "all_known_studies": True, "variable_ids": []},
                      statistic={"kind": "count", "denominator": "rows in the complete studies table", "grouping": ["locationDbId"]},
                      steps=[{"step_id": "step_1", "agent": "retriever", "action": "export_metadata", "inputs": {}},
                             {"step_id": "step_2", "agent": "analyst", "action": "count_records", "inputs": {"group_by": "locationDbId"}, "depends_on": ["step_1"]}])
    fixed = draft(**{**no_entity, "steps": [{**no_entity["steps"][0], "inputs": {"entity": "studies"}}, no_entity["steps"][1]]})
    model = fake(completed(no_entity), completed(fixed))
    result = plan_with(tmp_path, question, model, metadata)
    assert result.status == "completed" and result.attempts == 2
    assert "export_metadata takes entity = one of studies, variables, locations, programs, seasons" in model.requests[1]["messages"][-1]["content"]


# --- 3. clarifications: only the needed ones -------------------------------------------------------

def test_irrelevant_clarification_for_a_uniquely_named_study_is_dropped(tmp_path, metadata):
    asking = draft(clarifications=[{"question": "Which location and season should be used?"}])
    result = plan_with(tmp_path, Q_AB, fake(completed(asking)), metadata)
    assert result.status == "completed" and result.plan.status == "ready" and result.plan.clarifications == []
    assert any(n.startswith("dropped clarification") and "uniquely named" in n for n in result.notes)


def test_two_matching_traits_force_a_clarification(tmp_path, metadata):
    question = "Compare yield and dry matter in synthetic study S1."
    result = plan_with(tmp_path, question, fake(completed(draft())), metadata)          # the model just picked V1
    assert result.status == "needs_clarification" and result.plan.status == "needs_clarification"
    assert len(result.questions) == 1 and "More than one trait matches" in result.questions[0] and "V2 (dry matter content)" in result.questions[0]
    asked = plan_with(tmp_path, question, fake(clarify("Do you want fresh root yield (V1) or dry matter content (V2)?")), metadata)
    assert asked.status == "needs_clarification" and asked.plan is None and asked.questions == ["Do you want fresh root yield (V1) or dry matter content (V2)?"]


def test_missing_trait_cannot_be_papered_over(tmp_path, metadata):
    question = "What is the mean starch content in synthetic study S1?"
    stubborn = fake(completed(draft()), completed(draft()), completed(draft()))
    result = plan_with(tmp_path, question, stubborn, metadata)
    assert result.status == "failed" and any("not what the question names" in n for n in result.notes)
    honest = fake(completed(draft()), clarify("No variable named starch content exists; the known variables are V1 fresh root yield and V2 dry matter content. Which one?"))
    result = plan_with(tmp_path, question, honest, metadata)
    assert result.status == "needs_clarification" and result.attempts == 2 and "starch" in result.questions[0]
    empty_scope = draft(scope={"study_ids": ["S1"], "variable_ids": []}, steps=[{"step_id": "step_1", "agent": "retriever", "action": "get_study", "inputs": {"study_db_id": "S1"}}])
    result = plan_with(tmp_path, question, fake(completed(empty_scope)), metadata)
    assert result.status == "needs_clarification" and any("names no known trait" in q for q in result.questions)


def test_breeding_judgement_words_force_a_clarification(tmp_path, metadata):
    whole = draft(scope={"study_ids": [], "all_known_studies": True, "variable_ids": ["V1"]})       # "which trial" = over all studies
    result = plan_with(tmp_path, "Which trial has the best yield?", fake(completed(whole)), metadata)
    assert result.status == "needs_clarification" and result.plan is not None and result.plan.scope.study_ids == ["S1"]
    assert "breeding judgement" in result.questions[0] and "'best'" in result.questions[0]
    assert any(n.startswith("added required clarification") for n in result.notes)


def test_a_study_the_question_does_not_name_is_never_chosen_for_the_user(tmp_path, metadata):
    """ADV03 with a real model on 2026-09-28: 'In study 19ayt18highYldIB ...' came back with S1's numbers. Code now refuses that draft:
    a study enters the scope only when the question names it by ID or exact name, and the whole-set switch needs whole-set wording."""
    question = "In study 19ayt18highYldIB, what is the pooled mean fresh root yield, with valid-value and independent-plot counts?"
    assert co.find_candidates(question, metadata).studies == [] and not co.WHOLE_SET_WORDS.search(question)
    stubborn = fake(completed(draft()), completed(draft()), completed(draft()))                     # keeps choosing S1
    result = plan_with(tmp_path, question, stubborn, metadata)
    assert result.status == "failed" and result.plan is None and result.attempts == 3
    assert any("not named in the question" in n and "['S1']" in n and "ask which study is meant" in n for n in result.notes)
    dodge = draft(scope={"study_ids": [], "all_known_studies": True, "variable_ids": ["V1"]})       # the whole-set switch as a loophole
    result = plan_with(tmp_path, question, fake(completed(dodge), clarify("Which study is meant? The metadata knows S1 (SYNTHETIC yield trial S1).")), metadata)
    assert result.status == "needs_clarification" and result.plan is None and result.questions[0].startswith("Which study is meant?")
    assert any("all_known_studies is only for a question about several studies" in n for n in result.notes)


def test_a_variable_the_question_names_outright_is_the_only_candidate(tmp_path, metadata):
    """Real data, 2026-09-28 (Q4/Q10 on SweetPotatoBase): the question quoted one variable, gpt-4.1 chose it, and the CODE still
    demanded 'which trait?' because word overlap made dozens of variables candidates ('estimating' is in hundreds of names).
    Exact naming now wins: ID, full name, name before the ontology suffix, or trait name written out in the question."""
    named = [co.KnownVariable(variable_id="76736", name="Reaction to Fusarium Wilt estimating 0-5|CO_331:0000789", trait="Reaction to Fusarium Wilt estimating 0-5"),
             co.KnownVariable(variable_id="76740", name="Reaction to Fusarium root rot estimating 1-9|CO_331:0000742", trait="Reaction to Fusarium root rot estimating 1-9"),
             co.KnownVariable(variable_id="76984", name="Alternaria symptoms estimating 1-9|CO_331:0000811", trait="Alternaria symptoms estimating 1-9")]
    study = co.KnownStudy(study_id="4491", name="07DISE189HFLNO7", location_id="61", location_name="Horticultural Field Laboratory", seasons=["2007"], study_type="phenotyping_trial")
    real = co.KnownMetadata(base_url=SYNTHETIC_BASE, complete=True, studies=[study], variables=named, location_ids=["61"], season_ids=["2007"])
    quoted = "In study 07DISE189HFLNO7, what is the pooled mean of the variable 'Reaction to Fusarium Wilt estimating 0-5', with valid-value and independent-unit counts?"
    assert [v.variable_id for v in co.find_candidates(quoted, real).variables] == ["76736"]
    assert [v.variable_id for v in co.find_candidates("mean of 76984 in study 07DISE189HFLNO7", real).variables] == ["76984"]           # an ID token
    assert [v.variable_id for v in co.find_candidates("mean Fusarium reaction estimating in study 07DISE189HFLNO7", real).variables] == ["76736", "76740", "76984"]   # nothing named: overlap
    # the plan for the quoted question is ready: no 'more than one trait' clarification is added by code
    plan = draft(interpretation="pooled mean of the Fusarium wilt score in study 4491", scope={"study_ids": ["4491"], "variable_ids": ["76736"]},
                 steps=[{"step_id": "step_1", "agent": "retriever", "action": "get_observations", "inputs": {"study_db_id": "4491", "variable_db_id": "76736"}},
                        {"step_id": "step_2", "agent": "retriever", "action": "get_observation_units", "inputs": {"study_db_id": "4491"}},
                        {"step_id": "step_3", "agent": "analyst", "action": "numeric_summary", "inputs": {"column": "value"}, "depends_on": ["step_1", "step_2"]}])
    result = plan_with(tmp_path, quoted, fake(completed(plan)), real)
    assert result.status == "completed" and result.plan.status == "ready" and result.plan.scope.variable_ids == ["76736"]
    assert not any("More than one trait" in n for n in result.notes)
    # the synthetic behaviour is unchanged: 'fresh root yield' is named outright, 'yield' alone is overlap, two named traits still ask
    assert [v.variable_id for v in co.find_candidates(Q_AB, metadata).variables] == ["V1"]
    assert [v.variable_id for v in co.find_candidates("Compare fresh root yield and dry matter content in S1", metadata).variables] == ["V1", "V2"]


def test_rejections_tell_the_model_what_would_be_accepted(tmp_path, metadata):
    """Real data, 2026-09-28 (Q1-Q3, Q5, Q7, Q8): the model put objects into step inputs and lists into scope.filters; the raw
    contract messages named the place but not the remedy, and three attempts ran out. Now the remedy is spelled out."""
    with pytest.raises(ValidationError) as exc:
        co.PlanDraft.model_validate(draft(steps=[{"step_id": "step_1", "agent": "retriever", "action": "export_metadata", "inputs": {"filters": {"studyType": "AYT"}}}]))
    text = co._explain_rejection(exc.value)
    assert text.startswith("steps.0.inputs.filters") and "simple values" in text and "min_independent_n=2" in text
    assert _unknown_arguments(text) == [] and ("filter_rows", "op") in _named_tool_arguments(text)      # the remedy uses the tools' own names
    with pytest.raises(ValidationError) as exc:
        co.PlanDraft.model_validate(draft(scope={"study_ids": [], "all_known_studies": True, "variable_ids": [], "filters": {"location_ids": ["L1"]}}))
    text = co._explain_rejection(exc.value)
    assert text.startswith("scope.filters.location_ids") and "may hold only location_id, season_id, study_type, name_contains" in text
    with pytest.raises(ValidationError) as exc:
        co.PlanDraft.model_validate(draft(scope={"study_ids": ["S1"], "variable_ids": ["V1"], "studyType": "AYT"}))
    text = co._explain_rejection(exc.value)
    assert "'studyType' is not a field here" in text and "scope has only study_ids, all_known_studies" in text
    # end to end: a rejected draft is explained in the notes and the model's second, corrected draft is accepted
    bad = draft(steps=[{"step_id": "step_1", "agent": "retriever", "action": "get_observations", "inputs": {"study_db_id": "S1", "variable_db_id": "V1", "filters": {"x": "y"}}}])
    result = plan_with(tmp_path, Q_AB, fake(completed(bad), completed(draft())), metadata)
    assert result.status == "completed" and result.attempts == 2 and any("simple values" in n for n in result.notes)
    # a filter key the contract does not know is explained the same way when the plan is assembled
    extra = draft(scope={"study_ids": [], "all_known_studies": True, "variable_ids": ["V1"], "filters": {"studyType": "Synthetic Yield Trial"}})
    result = plan_with(tmp_path, "Among all synthetic studies, which study has the highest pooled mean of V1?", fake(completed(extra), completed(draft(
        interpretation="pooled mean of V1 per study", scope={"study_ids": [], "all_known_studies": True, "variable_ids": ["V1"]}))), metadata)
    assert result.status == "completed" and any("'studyType' is not a field here" in n and "filters only location_id" in n for n in result.notes)


def test_a_chain_of_analyst_steps_is_accepted_and_an_orphan_step_is_told_what_to_add(tmp_path, metadata):
    """Real data, 2026-09-29 (Q7, Q8, Q3): gpt-4.1 writes the natural chain export -> filter -> count, where the count depends on the
    filter step; the old contract wanted a DIRECT retriever dependency and rejected it three times a run, remedy or not. The chain is
    now accepted (contracts.Plan traces evidence through earlier analyst steps); a step that depends on nothing still gets the remedy.
    Q7's plan used list_variables, a preview that leaves the Analyst with no table: the prompt says catalog counts need export_metadata."""
    chained = draft(steps=[{"step_id": "step_1", "agent": "retriever", "action": "get_observations", "inputs": {"study_db_id": "S1", "variable_db_id": "V1"}},
                           {"step_id": "step_2", "agent": "analyst", "action": "table_info", "inputs": {}, "depends_on": ["step_1"]},
                           {"step_id": "step_3", "agent": "analyst", "action": "numeric_summary", "inputs": {"column": "value"}, "depends_on": ["step_2"]}])
    result = plan_with(tmp_path, Q_AB, fake(completed(chained)), metadata)
    assert result.status == "completed" and result.attempts == 1 and [s.step_id for s in result.plan.steps] == ["step_1", "step_2", "step_3"]
    orphan = draft(steps=[{"step_id": "step_1", "agent": "retriever", "action": "get_observations", "inputs": {"study_db_id": "S1", "variable_db_id": "V1"}},
                          {"step_id": "step_2", "agent": "analyst", "action": "numeric_summary", "inputs": {"column": "value"}, "depends_on": []}])
    result = plan_with(tmp_path, Q_AB, fake(completed(orphan), completed(draft())), metadata)
    assert result.status == "completed" and result.attempts == 2
    assert any("must depend on at least one retriever step" in n and "leads back to a retriever step" in n for n in result.notes)
    assert "previews only, never a table" in co.COORDINATOR_SYSTEM_PROMPT and "export_metadata step first" in co.COORDINATOR_SYSTEM_PROMPT


_TOOL_CALL = re.compile(r"\b([a-z_]+)\(([^()]*)\)")
_ARG_NAME = re.compile(r"\b([a-z_]+)=")


def _named_tool_arguments(text: str) -> list[tuple[str, str]]:
    """(tool, argument) for every tool(argument=...) written in a text; words that are not tools are skipped."""
    import analyst_tools as at
    import brapi_mcp_server as srv
    tools = {**{n: s.args_model for n, s in srv.TOOLS.items()}, **{n: s.args_model for n, s in at.ANALYST_TOOLS.items()}}
    return [(name, arg) for name, args in _TOOL_CALL.findall(text) if name in tools for arg in _ARG_NAME.findall(args)]


def _unknown_arguments(text: str) -> list[tuple[str, str]]:
    import analyst_tools as at
    import brapi_mcp_server as srv
    tools = {**{n: s.args_model for n, s in srv.TOOLS.items()}, **{n: s.args_model for n, s in at.ANALYST_TOOLS.items()}}
    return [(name, arg) for name, arg in _named_tool_arguments(text) if arg not in tools[name].model_fields]


def test_every_tool_argument_the_prompt_names_is_the_tools_own():
    """Real data, 2026-09-29, round 5: the prompt told the planner count_records with "by", but the tool's argument is group_by;
    the earlier test only checked that MY words were in the prompt, not that they were the TOOL's words. This test reads every
    tool(argument=...) in the prompt and checks it against the tool's own schema, so a wrong name fails here, not on real data."""
    prompt = co.COORDINATOR_SYSTEM_PROMPT
    named = _named_tool_arguments(prompt)
    assert _unknown_arguments(prompt) == []
    for pair in [("count_records", "group_by"), ("count_records", "distinct_by"), ("group_stats", "by"), ("group_stats", "min_independent_n"),
                 ("group_stats", "top_n"), ("filter_rows", "op"), ("filter_rows", "value"), ("export_metadata", "entity"),
                 ("get_observations", "study_db_id")]:
        assert pair in named, pair                                               # the rules are written as checkable calls, not prose
    assert "FILTERED studies table" in prompt and "controller ranks, the plan does not pick" in prompt and "count_records counts rows, never plots" in prompt
    # round 6 (Q7): the variables case was a sentence, not a call, and the planner copied only calls; now both are written out
    assert 'count_records(group_by="observationVariableDbId", distinct_by="observationVariableDbId")' in prompt
    # the checker itself catches the round-5 mistake
    assert _unknown_arguments('count_records(by="studyDbId") and group_stats(min_n=2)') == [("count_records", "by"), ("group_stats", "min_n")]


def test_a_placeholder_study_id_for_a_whole_set_question_is_refused_with_the_rule(tmp_path, metadata):
    """Real data, 2026-09-29, round 5 (Q10): for 'the resolved 2007 trials at ...' the planner wrote study_db_id='*from_step_1_each*'
    three times and ran out of attempts. The refusal now says to leave study_db_id out; the retriever gets the resolved study list."""
    question = "Among all synthetic studies, which study has the highest pooled mean of V1, and which studies lack usable values?"
    def whole(study_input):
        inputs = {"variable_db_id": "V1", **study_input}
        return draft(interpretation="pooled mean of V1 per study", scope={"study_ids": [], "all_known_studies": True, "variable_ids": ["V1"]},
                     steps=[{"step_id": "step_1", "agent": "retriever", "action": "get_observations", "inputs": inputs},
                            {"step_id": "step_2", "agent": "analyst", "action": "group_stats", "inputs": {"by": "studyDbId"}, "depends_on": ["step_1"]}])
    result = plan_with(tmp_path, question, fake(completed(whole({"study_db_id": "*from_step_1_each*"})), completed(whole({}))), metadata)
    assert result.status == "completed" and result.attempts == 2 and result.plan.scope.study_ids == ["S1"]
    assert any("outside the plan's scope" in n and "leave study_db_id out" in n for n in result.notes)
    assert "leave study_db_id OUT" in co.COORDINATOR_SYSTEM_PROMPT


def test_a_catalog_question_that_needs_no_study_is_told_so(tmp_path, metadata):
    """Real data, 2026-09-28 (Q9): 'how many seasons and variables exist' names no study and has no whole-set wording; the model
    set all_known_studies, was refused, and then asked 'which study?'. The refusal now says what such a question needs."""
    question = "How many distinct season IDs and observation-variable IDs are present on this snapshot?"
    assert not co.WHOLE_SET_WORDS.search(question) and co.find_candidates(question, metadata).studies == []
    wrong = draft(interpretation="count seasons and variables", scope={"study_ids": [], "all_known_studies": True, "variable_ids": []},
                  statistic={"kind": "count", "denominator": "catalog entries", "grouping": []},
                  steps=[{"step_id": "step_1", "agent": "retriever", "action": "export_metadata", "inputs": {"entity": "seasons"}},
                         {"step_id": "step_2", "agent": "retriever", "action": "export_metadata", "inputs": {"entity": "variables"}},
                         {"step_id": "step_3", "agent": "analyst", "action": "count_records", "inputs": {"by": "seasonDbId"}, "depends_on": ["step_1", "step_2"]}])
    right = dict(wrong, scope={"study_ids": [], "all_known_studies": False, "variable_ids": []})
    result = plan_with(tmp_path, question, fake(completed(wrong), completed(right)), metadata)
    assert result.status == "completed" and result.plan.status == "ready" and result.plan.scope.study_ids == [] and result.attempts == 2
    assert any("catalog question that needs no study at all" in n and "export_metadata" in n for n in result.notes)
    assert "catalog question that needs no study" in co.COORDINATOR_SYSTEM_PROMPT and "SIMPLE values only" in co.COORDINATOR_SYSTEM_PROMPT


def test_when_the_whole_set_rule_refuses_every_attempt_the_person_is_asked(tmp_path, metadata):
    """Real data, 2026-09-29 (the chat): 'list ids of sweetpotato study done in 2026' was planned as all studies three times and
    refused three times ('study', not 'studies'). The rule stays as it is; the person is asked, and 'all studies from 2026' passes."""
    question = "list ids of sweetpotato study done in 2026"
    assert not co.WHOLE_SET_WORDS.search(question) and co.find_candidates(question, metadata).studies == []
    listing = draft(interpretation="IDs of the studies with a 2026 season",
                    scope={"study_ids": [], "all_known_studies": True, "variable_ids": [], "filters": {"season_id": "2026"}},
                    statistic={"kind": "count", "denominator": "catalog entries", "grouping": []},
                    steps=[{"step_id": "step_1", "agent": "retriever", "action": "export_metadata", "inputs": {"entity": "studies"}},
                           {"step_id": "step_2", "agent": "analyst", "action": "count_records",
                            "inputs": {"group_by": "studyDbId", "distinct_by": "studyDbId"}, "depends_on": ["step_1"]}])
    result = plan_with(tmp_path, question, fake(completed(listing), completed(listing), completed(listing)), metadata)
    assert result.status == "needs_clarification" and result.plan is None and result.questions == [co.ONE_OR_SEVERAL]
    assert any("no acceptable plan after 3 attempts" in n for n in result.notes) and any(co.WHOLE_SET_REJECTION in n for n in result.notes)
    answered = co.add_clarification(question, "all studies from 2026")
    result = plan_with(tmp_path, answered, fake(completed(listing)), metadata)
    assert result.status == "completed" and result.plan.scope.study_ids == ["S1"] and result.plan.question == answered   # S1 has season 2026
    stubborn = fake(completed(draft()), completed(draft()), completed(draft()))                         # any other refusal still fails
    assert plan_with(tmp_path, "In study 19ayt18highYldIB, what is the mean?", stubborn, metadata).status == "failed"


def test_a_person_reviewed_plan_may_cover_several_studies_without_the_word_studies(tmp_path, metadata):
    """Bidhan, 2026-09-29: 'study', 'studies' or even 'stdy' - the model should judge, because nothing is fetched before a person
    approves the scope. With a person (the chat; the command line without --auto) the planner's judgment stands; without one
    (--auto, the evaluation) the wording rule stays strict. A study the question names still wins."""
    question = "list ids of sweetpotato stdy done in 2026"
    listing = draft(interpretation="IDs of the studies with a 2026 season",
                    scope={"study_ids": [], "all_known_studies": True, "variable_ids": [], "filters": {"season_id": "2026"}},
                    statistic={"kind": "count", "denominator": "catalog entries", "grouping": []},
                    steps=[{"step_id": "step_1", "agent": "retriever", "action": "export_metadata", "inputs": {"entity": "studies"}},
                           {"step_id": "step_2", "agent": "analyst", "action": "count_records",
                            "inputs": {"group_by": "studyDbId", "distinct_by": "studyDbId"}, "depends_on": ["step_1"]}])
    strict = plan_with(tmp_path, question, fake(completed(listing), completed(listing), completed(listing)), metadata)
    assert strict.status == "needs_clarification" and strict.questions == [co.ONE_OR_SEVERAL]
    judged = plan_with(tmp_path, question, fake(completed(listing)), metadata, person_reviews_scope=True)
    assert judged.status == "completed" and judged.plan.scope.study_ids == ["S1"] and co.SCOPE_JUDGED_NOTE in judged.notes
    named = draft(scope={"study_ids": [], "all_known_studies": True, "variable_ids": ["V1"]})
    result = plan_with(tmp_path, Q_AB, fake(completed(named), completed(draft())), metadata, person_reviews_scope=True)
    assert result.status == "completed" and result.attempts == 2 and result.plan.scope.study_ids == ["S1"]


def test_the_planner_sees_its_earlier_questions_but_the_rules_read_only_the_persons_words(tmp_path, metadata):
    """Run 203842 (2026-09-29): 'yes' meant nothing to a planner that never saw its own question. Its earlier questions are now
    context in its message and, with a person at the approval screen, it is told that a person checks the scope. The rules
    still read only the person's words: the planner's 'Do you mean study S1?' does not name S1."""
    earlier = ["Do you mean all studies from the 2026 season? Please confirm."]
    model = fake(clarify("Which study?"))
    plan_with(tmp_path, co.add_clarification("tell me about the study of 2026", "yes"), model, metadata,
              earlier_questions=earlier, person_reviews_scope=True)
    told = model.requests[0]["messages"][-1]["content"]
    assert f"your question 1: {earlier[0]}" in told and co.PERSON_REVIEWS_NOTE in told
    plain = fake(clarify("Which study?"))
    plan_with(tmp_path, "tell me about the study of 2026", plain, metadata)                      # --auto, or a first question
    told = plain.requests[0]["messages"][-1]["content"]
    assert "your question 1" not in told and co.PERSON_REVIEWS_NOTE not in told
    stubborn = fake(completed(draft()), completed(draft()), completed(draft()))                 # plans S1, which only the planner named
    result = plan_with(tmp_path, co.add_clarification("What is the mean yield?", "yes"), stubborn, metadata,
                       earlier_questions=["Do you mean study S1?"], person_reviews_scope=True)
    assert result.status == "failed" and any("not named in the question" in n for n in result.notes)


def test_answers_are_added_and_only_the_persons_words_count(metadata):
    """The controller adds each answer after a label that code writes. The label never names anything (a study whose ID is 1 is
    not named by 'Clarification 1'), and decision words are read in the newest answer only: 'best' was already asked about."""
    q = co.add_clarification(co.add_clarification("Which trial has the best yield?", "  S1   please "), "descriptive means")
    assert q == "Which trial has the best yield?\nClarification 1 from the person: S1 please\nClarification 2 from the person: descriptive means"
    assert co.clarifications_given(q) == 2 and co.latest_words(q) == "descriptive means" and "Clarification" not in co.person_words(q)
    cands = co.find_candidates(q, metadata)
    assert [s.study_id for s in cands.studies] == ["S1"] and cands.decision_words == []
    assert co.find_candidates("Which trial has the best yield?", metadata).decision_words == ["best"]
    one = co.KnownStudy(study_id="1", name="a study whose ID is 1")
    tiny = co.KnownMetadata(base_url=SYNTHETIC_BASE, complete=True, studies=[one], variables=[])
    assert co.find_candidates(co.add_clarification("Tell me about a study.", "yes"), tiny).studies == []
    assert co.find_candidates("Tell me about study 1.", tiny).studies == [one]                         # the person's own '1' counts
    assert not co.WHOLE_SET_WORDS.search(co.person_words(co.add_clarification("list ids of sweetpotato study", "yes")))


def test_all_known_studies_is_filled_by_code_for_a_question_about_several_studies(tmp_path, metadata):
    question = "Among all synthetic studies, which study has the highest pooled mean of V1, and which studies lack usable values?"
    assert co.WHOLE_SET_WORDS.search(question) and co.find_candidates(question, metadata).studies == []
    whole = draft(interpretation="pooled mean of V1 per study over every known study", scope={"study_ids": [], "all_known_studies": True, "variable_ids": ["V1"]})
    result = plan_with(tmp_path, question, fake(completed(whole)), metadata)
    assert result.status == "completed" and result.plan.scope.study_ids == ["S1"] and result.plan.scope.matching_study_count == 1
    assert any("set by code to the 1 of 1 known studies" in n for n in result.notes)
    listed = draft(interpretation="pooled mean of V1 per study", scope={"study_ids": ["S1"], "all_known_studies": True, "variable_ids": ["V1"]})
    result = plan_with(tmp_path, question, fake(completed(listed)), metadata)                        # the list is code's, not the model's
    assert result.status == "completed" and result.plan.scope.study_ids == ["S1"]
    named = draft(scope={"study_ids": [], "all_known_studies": True, "variable_ids": ["V1"]})
    result = plan_with(tmp_path, Q_AB, fake(completed(named), completed(draft())), metadata)          # Q_AB names S1: the switch is refused, then fixed
    assert result.status == "completed" and result.plan.scope.study_ids == ["S1"] and result.attempts == 2
    assert any("all_known_studies must be false when a study is named" in n for n in result.notes)
    for text, expected in (("Which trial has the best yield?", True), ("How many studies have the Ibadan location?", True),
                           ("In study 19ayt18highYldIB, what is the mean across all plots?", False), ("What is the mean yield of S1?", False)):
        assert bool(co.WHOLE_SET_WORDS.search(text)) is expected, text


def test_all_known_studies_is_narrowed_by_the_plans_filters(tmp_path, metadata):
    """On a live server 'all known studies' is thousands; the plan's exact filters (location, season, type, name) narrow the list
    in code before any study is in scope. The synthetic metadata has one study at L1, season 2026, type 'Synthetic Yield Trial'."""
    question = "Among all studies at the resolved location in the 2026 season, which study has the highest pooled mean of V1?"
    kept = draft(scope={"study_ids": [], "all_known_studies": True, "variable_ids": ["V1"], "filters": {"location_id": "L1", "season_id": "2026"}})
    result = plan_with(tmp_path, question, fake(completed(kept)), metadata)
    assert result.status == "completed" and result.plan.scope.study_ids == ["S1"] and result.plan.scope.matching_study_count == 1
    assert any("the 1 of 1 known studies that match" in n for n in result.notes)
    dropped = draft(interpretation="pooled mean of V1 per Advanced Yield Trial", scope={"study_ids": [], "all_known_studies": True, "variable_ids": ["V1"],
                    "filters": {"study_type": "Advanced Yield Trial"}},
                    steps=[{"step_id": "step_1", "agent": "retriever", "action": "export_metadata", "inputs": {"entity": "studies"}}])
    result = plan_with(tmp_path, question, fake(completed(dropped)), metadata)
    assert result.status == "completed" and result.plan.scope.study_ids == [] and result.plan.scope.matching_study_count == 0
    assert any("the 0 of 1 known studies that match" in n for n in result.notes)
    s1 = metadata.studies[0]
    assert co._study_matches(s1, co.DraftScope(all_known_studies=True, filters={"name_contains": "yield trial s1"}))
    assert not co._study_matches(s1, co.DraftScope(all_known_studies=True, season_ids=["2019"]))
    assert not co._study_matches(s1, co.DraftScope(all_known_studies=True, location_ids=["L2"]))


def test_an_explicit_list_is_accepted_only_when_it_equals_the_code_computed_whole_set(tmp_path, metadata):
    """A model may say the whole set aloud instead of using the switch, but only the exact set code would fill: a subset could
    drop a study silently, and a study the question does not name is still refused when the question is about one study."""
    question = "Among all synthetic studies, which study has the highest pooled mean of V1, and which studies lack usable values?"
    said = draft(interpretation="pooled mean of V1 per study", scope={"study_ids": ["S1"], "variable_ids": ["V1"]})
    result = plan_with(tmp_path, question, fake(completed(said)), metadata)
    assert result.status == "completed" and result.plan.scope.study_ids == ["S1"] and result.attempts == 1
    assert any("accepted for a question about all studies" in n for n in result.notes)
    two = metadata.model_copy(update={"studies": [*metadata.studies, co.KnownStudy(study_id="S2", name="SYNTHETIC yield trial S2", location_id="L1",
                                                                                  seasons=["2026"], study_type="Synthetic Yield Trial")]})
    subset = plan_with(tmp_path, question, fake(completed(said), completed(said), completed(said)), two)
    assert subset.status == "failed" and any("['S1'] are not named in the question" in n for n in subset.notes)
    both = draft(interpretation="pooled mean of V1 per study", scope={"study_ids": ["S2", "S1"], "variable_ids": ["V1"]},
                 steps=[{"step_id": "step_1", "agent": "retriever", "action": "get_observations", "inputs": {"study_db_id": "S1", "variable_db_id": "V1"}},
                        {"step_id": "step_2", "agent": "retriever", "action": "get_observations", "inputs": {"study_db_id": "S2", "variable_db_id": "V1"}},
                        {"step_id": "step_3", "agent": "analyst", "action": "numeric_summary", "depends_on": ["step_1", "step_2"]}])
    result = plan_with(tmp_path, question, fake(completed(both)), two)
    assert result.status == "completed" and result.plan.scope.study_ids == ["S1", "S2"] and result.plan.scope.matching_study_count == 2
    one_study = "In study 19ayt18highYldIB, what is the pooled mean fresh root yield, with valid-value and independent-plot counts?"
    result = plan_with(tmp_path, one_study, fake(completed(said), completed(said), completed(said)), metadata)
    assert result.status == "failed"                                                        # no whole-set wording: still refused


def test_sentence_end_punctuation_does_not_hide_an_id(metadata):
    assert [s.study_id for s in co.find_candidates("Compare yield and dry matter in synthetic study S1.", metadata).studies] == ["S1"]
    assert [v.variable_id for v in co.find_candidates("Is V2, or V1?", metadata).variables] == ["V1", "V2"]


# --- 4. metadata-only questions and discovery proposals --------------------------------------------

def test_metadata_only_question_plans_a_count_without_asking_for_a_trait(tmp_path, metadata):
    catalog = draft(interpretation="count studies per location from the complete studies catalog", scope={"study_ids": [], "all_known_studies": True, "variable_ids": []},
                    statistic={"kind": "count", "denominator": "rows in the complete studies table", "grouping": ["locationDbId"]},
                    steps=[{"step_id": "step_1", "agent": "retriever", "action": "export_metadata", "inputs": {"entity": "studies"}},
                           {"step_id": "step_2", "agent": "analyst", "action": "count_records", "inputs": {"group_by": "locationDbId"}, "depends_on": ["step_1"]}])
    result = plan_with(tmp_path, "How many studies are there per location?", fake(completed(catalog)), metadata)
    assert result.status == "completed" and result.plan.status == "ready" and result.plan.statistic.kind == "count"
    assert result.plan.scope.variable_ids == [] and result.plan.clarifications == [] and result.plan.scope.matching_study_count == 1


def test_without_metadata_the_plan_must_propose_discovery(tmp_path):
    discovery = draft(interpretation="discover studies and variables about yield, then plan the fetch",
                      scope={"study_ids": [], "variable_ids": [], "discovery_needed": True, "discovery_families": ["studies", "observationvariables"],
                             "filters": {"name_contains": "yield"}},
                      steps=[{"step_id": "step_1", "agent": "retriever", "action": "search_studies", "inputs": {"name_contains": "yield"}},
                             {"step_id": "step_2", "agent": "retriever", "action": "list_variables", "inputs": {"name_contains": "yield"}}])
    result = plan_with(tmp_path, "Mean yield of the yield trials?", fake(completed(discovery)), None)
    assert result.status == "completed" and result.plan.scope.discovery_needed is True and result.plan.scope.matching_study_count is None
    assert result.plan.scope.filters.name_contains == "yield" and result.plan.scope.discovery_families == ["studies", "observationvariables"]
    result = plan_with(tmp_path, "Mean yield of the yield trials?", fake(completed(draft()), completed(draft()), completed(draft())), None)
    assert result.status == "failed" and any("must set discovery_needed=true" in n for n in result.notes)
    with_obs = draft(scope={"study_ids": [], "variable_ids": [], "discovery_needed": True, "discovery_families": ["observations"]},
                     steps=[{"step_id": "step_1", "agent": "retriever", "action": "search_studies"}])
    result = plan_with(tmp_path, "Mean yield?", fake(completed(with_obs), completed(with_obs), completed(with_obs)), None)
    assert result.status == "failed" and any("metadata families only" in n for n in result.notes)


def test_discovery_is_refused_when_complete_metadata_exists(tmp_path, metadata):
    redundant = draft(scope={"study_ids": [], "variable_ids": [], "discovery_needed": True, "discovery_families": ["studies"]},
                      steps=[{"step_id": "step_1", "agent": "retriever", "action": "search_studies"}])
    result = plan_with(tmp_path, Q_AB, fake(completed(redundant), completed(draft())), metadata)
    assert result.status == "completed" and any("discovery is not needed" in n for n in result.notes)


def test_agent_failures_carry_through(tmp_path, metadata):
    result = plan_with(tmp_path, Q_AB, fake(*[reply_text("", model="mock-model")] * 3), metadata)   # empty until repairs run out
    assert result.status == "failed" and result.plan is None and any("empty_reply" in n for n in result.notes)
    result = plan_with(tmp_path, Q_AB, fake(reply_text("not json", model="mock-model")) if False else fake(completed({"nonsense": True}), completed(draft())), metadata)
    assert result.status == "completed" and result.attempts == 2


# --- 5. deterministic rendering ------------------------------------------------------------------

def _claim(cid, kind, value, *, filters=None, unit=None, n=4, units=4, reason=None, excl=()):
    return Claim(claim_id=cid, kind=kind, value=value, unit=unit, missing_reason=reason, denominator=Denominator(name="valid plot-level values", n=n),
                 n_independent_units=units, source_artifact_hashes=["a" * 64], filters=filters or {}, grouping=list((filters or {}).keys()),
                 transformation="pooled mean of valid values", exclusions=[Exclusion(reason=r, count=k) for r, k in excl])


def _artifact(aid, rows, sha="a" * 64):
    return ArtifactManifest(artifact_id=aid, run_id="run_r", kind="observations", relative_path=f"run_r/artifacts/{aid}.csv", sha256=sha,
                            schema_version="1", row_count=rows, complete=True, study_ids=["S1"], variable_ids=["V1"], created_at_utc=NOW)


def test_numeric_table_is_rendered_from_claims_and_keeps_no_data_studies():
    claims = [_claim("clm_x_0001_mean_germplasmDbId_GB", "mean", 24.0, filters={"germplasmDbId": "GB"}, n=2, units=2),
              _claim("clm_x_0001_mean", "mean", 18.0, excl=(("raw-missing value token", 1), ("invalid or non-finite value token", 1))),
              _claim("clm_x_0001_sd", "sd", None, reason="fewer than 2 valid values", n=1, units=1)]
    analysis = AnalysisReport(status="completed", claims=claims, caveats=["means are descriptive"])
    a1, a2 = _artifact("art_x_0001", 6), _artifact("art_x_0002", 0, "b" * 64)
    retrieval = RetrievalReport(status="blocked", complete=False, artifacts=[a1, a2], warnings=["S9 blocked before dispatch"],
                                study_ledger=[StudyStatusEntry(study_id="S9", status="blocked", reason="not on the approval"),
                                              StudyStatusEntry(study_id="S1", status="complete", artifact_id="art_x_0001"),
                                              StudyStatusEntry(study_id="S2", status="empty", artifact_id="art_x_0002")])
    text = co.render_numeric_table(analysis, retrieval)
    lines = text.split("\n")
    assert lines[2].startswith("| clm_x_0001_mean |") and lines[3].startswith("| clm_x_0001_mean_germplasmDbId_GB |")   # sorted by claim id
    assert "| 18 | not stated | valid plot-level values (4) | 4 | all rows | raw-missing value token x1; invalid or non-finite value token x1 |" in lines[2]
    assert "| null (fewer than 2 valid values) |" in lines[4] and "germplasmDbId=GB" in lines[3]
    assert "| S1 | complete | 6 |  |" in text and "| S2 | empty | 0 |  |" in text and "| S9 | blocked | - | not on the approval |" in text
    assert "- means are descriptive" in text and "- S9 blocked before dispatch" in text
    again = co.render_numeric_table(AnalysisReport(status="completed", claims=list(reversed(claims)), caveats=["means are descriptive"]), retrieval)
    assert again == text                                                                         # order of input does not matter
    empty = co.render_numeric_table(AnalysisReport(status="failed", claims=[]), None)
    assert "(no claims)" in empty


# --- 6. prose must cite the claims it repeats ---------------------------------------------------

CLAIMS = [_claim("clm_x_0001_mean", "mean", 18.0, excl=(("raw-missing value token", 1), ("invalid or non-finite value token", 1))),
          _claim("clm_x_0001_mean_germplasmDbId_GB", "mean", 24.0, filters={"germplasmDbId": "GB"}, n=2, units=2),
          _claim("clm_x_0001_sd", "sd", 7.831560082980487),
          _claim("clm_x_0001_count", "count", 4)]


def test_fabricated_values_are_removed_and_flagged():
    prose = ("The pooled mean fresh root yield was 18 from 4 valid plots [clm_x_0001_mean]. "
             "Clone GB reached 25 [clm_x_0001_mean_germplasmDbId_GB]. "
             "About 20 plots were sown. "
             "The sample SD was 7.83 [clm_x_0001_sd]. "
             "Plot P4 belongs to clone GB in study S1, measured on 2026-06-01T00:00:00Z [clm_x_0001_mean]. "
             "Six plots were measured and 4 were valid [clm_x_0001_count]. "
             "Clone GB is the best clone for release [clm_x_0001_mean_germplasmDbId_GB]. "
             "See [clm_nope_0001] for 3 more details.")
    kept, flags = co.sanitize_prose(prose, CLAIMS)
    assert kept == ("The pooled mean fresh root yield was 18 from 4 valid plots [clm_x_0001_mean]. The sample SD was 7.83 [clm_x_0001_sd]. "
                    "Plot P4 belongs to clone GB in study S1, measured on 2026-06-01T00:00:00Z [clm_x_0001_mean]. "
                    "Six plots were measured and 4 were valid [clm_x_0001_count].")
    assert [f.split(":")[0] for f in flags] == ["removed (number 25 not in cited claims)", "removed (numbers without a claim citation)",
                                                "removed (breeding judgement)", "removed (cites unknown claim clm_nope_0001)"]


def test_prose_number_matching_is_by_claim_not_by_digit_string():
    assert co.sanitize_prose("Clone GB (plots P4, P5, P6) averaged 24 [clm_x_0001_mean_germplasmDbId_GB].", CLAIMS)[1] == []
    assert co.sanitize_prose("Clone GB averaged 24.0 over 2 plots [clm_x_0001_mean_germplasmDbId_GB].", CLAIMS)[1] == []
    kept, flags = co.sanitize_prose("The mean was 18.3 [clm_x_0001_mean].", CLAIMS)
    assert kept == "" and "18.3" in flags[0]                                                     # 18.3 is not 18.0 at one decimal
    kept, flags = co.sanitize_prose("One value was missing and one was invalid [clm_x_0001_mean]. Nothing numeric here.", CLAIMS)
    assert flags == [] and kept.endswith("Nothing numeric here.")
    kept, flags = co.sanitize_prose("", CLAIMS)
    assert kept == "" and flags == []
