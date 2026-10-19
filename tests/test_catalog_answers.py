"""Bounded catalog counts use exact export evidence, scope and fixed tools; no model/network."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import catalog_answers as ca
from agents.retriever import ToolRecord
from artifacts import ArtifactRegistry
from contracts import Plan, PlanStep, ResolvedScope, RetrievalReport, ToolResult
from part2_support import SYNTHETIC_BASE
from llm import Budget
from part2_support import async_returning, part2_config, part2_runtime


@pytest.fixture
def registry(tmp_path):
    return ArtifactRegistry(tmp_path / "out", "run_catalog_counts", base_url=SYNTHETIC_BASE)


def plan(entity="locations", *, filters=None, studies=(), question="How many locations are in the catalog?", scope_season=None):
    steps = [PlanStep(step_id="step_export", agent="retriever", action="export_metadata", inputs={"entity": entity})]
    for index, args in enumerate(filters or []):
        steps.append(PlanStep(step_id=f"step_filter{index}", agent="analyst", action="filter_rows", inputs=args,
                              depends_on=[steps[-1].step_id]))
    identity = ca.ENTITY_IDS[entity]
    steps.append(PlanStep(step_id="step_count", agent="analyst", action="count_records", depends_on=[steps[-1].step_id],
                          inputs={"group_by": identity, "distinct_by": identity}))
    scope = ResolvedScope(base_url=SYNTHETIC_BASE, study_ids=list(studies), season_ids=[scope_season] if scope_season else [],
                          filters={"season_id": scope_season}, matching_study_count=len(studies))
    return Plan(plan_id="plan_catalog_counts", question=question, interpretation="Count complete catalog records", status="ready", scope=scope,
                statistic={"kind": "count", "denominator": "complete catalog rows", "grouping": [identity]},
                steps=steps, created_at_utc=datetime.now(timezone.utc))


def saved(registry, rows, *, kind="locations", complete=True):
    if kind == "studies":
        columns = {"studyDbId": "ID", "studyName": "name", "seasons": "JSON season IDs"}
    else:
        columns = {"locationDbId": "ID", "locationName": "name"}
    return registry.save_table(rows, kind=kind, columns=columns, complete=complete)


def record(artifact, args=None, *, complete=True):
    return ToolRecord("export_metadata", args or {"entity": artifact.kind},
                      ToolResult(ok=True, complete=complete, artifact_ids=[artifact.artifact_id]))


def run(registry, proposed, artifacts, records=None, **kwargs):
    retrieval = RetrievalReport(status="completed", complete=True, artifacts=artifacts)
    sources = ca.capture_catalog_sources(proposed, records if records is not None else [record(a) for a in artifacts])
    return ca.run_catalog_plan(proposed, retrieval, registry, export_sources=sources, **kwargs)


def season_plan(studies=("S1", "S2"), value="2026", **kwargs):
    return plan("studies", filters=[{"column": "seasonDbId", "op": "eq", "value": value}], studies=studies,
                scope_season=value, question=kwargs.pop("question", f"Which studies have a {value} season?"), **kwargs)


def season_rows():
    return [{"studyDbId": "S1", "studyName": "one", "seasons": '["2026"]'},
            {"studyDbId": "S2", "studyName": "two", "seasons": '["2025", "2026"]'},
            {"studyDbId": "S3", "studyName": "three", "seasons": '["12026"]'},
            {"studyDbId": "S4", "studyName": "four", "seasons": '["year 2026"]'}]


def test_locations_count_needs_two_fixed_tool_calls_and_no_model(registry):
    artifact = saved(registry, [{"locationDbId": "007"}, {"locationDbId": "7"}, {"locationDbId": "80"}])
    budget = Budget(tool_calls=1)
    result = run(registry, plan(), [artifact], budget=budget)
    assert result.status == "completed" and budget.tool_calls == 3 and budget.model_calls == 0
    assert next(c for c in result.claims if c.transformation.startswith("number of distinct")).value == 3
    assert all(c.source_artifact_hashes == [artifact.sha256] for c in result.claims)


def test_season_alias_uses_exact_json_membership_and_derived_lineage(registry):
    source = saved(registry, season_rows(), kind="studies")
    budget = Budget()
    answer = run(registry, season_plan(), [source], budget=budget)
    assert answer.status == "completed" and budget.tool_calls == 3
    ids = {c.filters["studyDbId"] for c in answer.claims if "studyDbId" in c.filters}
    assert ids == {"S1", "S2"}
    assert next(c.value for c in answer.claims if c.transformation.startswith("number of distinct")) == 2
    assert len(answer.result_artifact_ids) == 1
    derived, rows = registry.load(answer.result_artifact_ids[0])
    assert {r["studyDbId"] for r in rows} == {"S1", "S2"}
    assert all(c.source_artifact_hashes == [derived.sha256] for c in answer.claims)
    assert any(source.sha256 in note for note in derived.quality_notes)


@pytest.mark.parametrize("value", ["2026"])
def test_the_season_filter_in_its_text_form_uses_the_same_exact_membership(registry, value):
    """Live 2026-10-19 (Opus 5.5): the planner wrote "seasons contains 2026" instead of "seasonDbId eq 2026"; the run
    stopped and asked the person ("did not apply every approved scope filter: season_id"). The text form now goes through
    the same exact check: "12026" and "year 2026" contain the text but are not the season."""
    source = saved(registry, season_rows(), kind="studies")
    proposed = plan("studies", filters=[{"column": "seasons", "op": "contains", "value": value}], studies=("S1", "S2"),
                    scope_season=value, question=f"Which studies have a {value} season?")
    answer = run(registry, proposed, [source], budget=Budget())
    assert answer.status == "completed"
    assert next(c.value for c in answer.claims if c.transformation.startswith("number of distinct")) == 2
    derived, rows = registry.load(answer.result_artifact_ids[0])
    assert {r["studyDbId"] for r in rows} == {"S1", "S2"}
    assert any("seasons text filter as exact JSON seasons membership" in m for m in answer.methods)


def test_a_season_text_filter_that_is_not_the_approved_season_still_stops(registry):
    source = saved(registry, season_rows(), kind="studies")
    proposed = plan("studies", filters=[{"column": "seasons", "op": "contains", "value": "202"}], studies=("S1", "S2"),
                    scope_season="2026", question="Which studies have a 2026 season?")
    answer = run(registry, proposed, [source], budget=Budget())
    assert answer.status != "completed" and not answer.claims


def test_narrower_first_export_cannot_satisfy_approved_full_export(registry):
    narrow = saved(registry, season_rows()[:2], kind="studies")
    full = saved(registry, season_rows(), kind="studies")
    records = [record(narrow, {"entity": "studies", "filters": {"season_id": "2026"}}), record(full)]
    proposed = season_plan()
    bindings = ca.capture_catalog_sources(proposed, records)
    assert bindings == {"step_export": [full.artifact_id]}
    result = run(registry, proposed, [narrow, full], records)
    assert result.status == "completed"
    assert full.artifact_id in result.methods[0] and narrow.artifact_id not in result.methods[0]


def test_different_sources_with_the_same_export_arguments_are_ambiguous(registry):
    first = saved(registry, [{"locationDbId": "L1"}])
    other = saved(registry, [{"locationDbId": "L2"}])
    budget = Budget()
    answer = run(registry, plan(), [first, other], budget=budget)
    assert answer.status == "needs_clarification" and not answer.claims and budget.tool_calls == 0
    assert "ambiguous" in " ".join(answer.caveats)


def test_identical_repeat_exports_deduplicate_by_hash(registry):
    one = saved(registry, [{"locationDbId": "L1"}])
    repeat = saved(registry, [{"locationDbId": "L1"}])
    answer = run(registry, plan(), [one, repeat])
    assert answer.status == "completed"
    assert next(c.value for c in answer.claims if c.transformation.startswith("number of distinct")) == 1


def test_complete_empty_catalog_reports_zero(registry):
    artifact = saved(registry, [])
    answer = run(registry, plan(), [artifact])
    assert answer.status == "completed" and len(answer.claims) == 1 and answer.claims[0].value == 0


@pytest.mark.parametrize("failure", ["partial", "wrong_hash", "failed_record", "missing_binding"])
def test_unproven_export_never_produces_count(registry, failure):
    artifact = saved(registry, [{"locationDbId": "L1"}], complete=failure != "partial")
    evidence = artifact.model_copy(update={"sha256": "f" * 64}) if failure == "wrong_hash" else artifact
    records = [record(artifact)]
    if failure == "failed_record":
        records = [ToolRecord("export_metadata", {"entity": "locations"}, ToolResult(ok=False, error={"code": "timeout", "message": "SYNTHETIC failure"}))]
    if failure == "missing_binding":
        records = [record(artifact, {"entity": "studies"})]
    answer = run(registry, plan(), [evidence], records)
    assert answer.status != "completed" and not answer.claims


@pytest.mark.parametrize("column", ["seasonMagic", "locationLabelInvented"])
def test_arbitrary_missing_filter_column_is_never_ignored(registry, column):
    artifact = saved(registry, [{"locationDbId": "L1"}])
    proposed = plan(filters=[{"column": column, "op": "eq", "value": "L1"}])
    answer = run(registry, proposed, [artifact])
    assert answer.status == "needs_clarification" and not answer.claims


@pytest.mark.parametrize("fault", ["scope_ids", "scope_season", "question", "clarification_label", "partial_json", "malformed_json"])
def test_season_alias_refuses_unverified_scope_or_membership(registry, fault):
    rows = season_rows()
    proposed = season_plan()
    if fault == "scope_ids":
        proposed = season_plan(studies=("S1",))
    elif fault == "scope_season":
        proposed = proposed.model_copy(update={"scope": proposed.scope.model_copy(update={"season_ids": ["2025"]})})
    elif fault == "question":
        proposed = season_plan(question="Which studies have a recent season?")
    elif fault == "clarification_label":
        rows = [{"studyDbId": "S1", "seasons": '["1"]'}]
        proposed = season_plan(studies=("S1",), value="1", question="Which studies?\nClarification 1 from the person: recent ones")
    elif fault == "partial_json":
        rows[-1]["seasons"] = "[]"
    else:
        rows[-1]["seasons"] = '"2026"'
    artifact = saved(registry, rows, kind="studies")
    answer = run(registry, proposed, [artifact])
    assert answer.status != "completed" and not answer.claims


def test_unapplied_scope_filter_is_not_silently_ignored(registry):
    artifact = saved(registry, season_rows(), kind="studies")
    proposed = plan("studies", studies=("S1", "S2"), scope_season="2026")
    answer = run(registry, proposed, [artifact])
    assert answer.status == "needs_clarification" and "every approved scope filter" in " ".join(answer.caveats)


def test_ambiguous_step_dependencies_stop_before_count(registry):
    artifact = saved(registry, [{"locationDbId": "L1"}])
    proposed = plan()
    proposed = proposed.model_copy(update={"steps": [proposed.steps[0], proposed.steps[-1].model_copy(update={"depends_on": []})]})
    budget = Budget()
    answer = run(registry, proposed, [artifact], budget=budget)
    assert answer.status == "needs_clarification" and budget.tool_calls == 0


@pytest.mark.parametrize("limited", ["tools", "time"])
def test_shared_budget_blocks_additional_catalog_operations(registry, limited):
    artifact = saved(registry, [{"locationDbId": "L1"}])
    budget = Budget(max_tool_calls=1, tool_calls=1) if limited == "tools" else Budget(started=1.0, max_elapsed_seconds=10.0)
    before = budget.tool_calls
    answer = run(registry, plan(), [artifact], budget=budget, clock=lambda: 11.0)
    assert answer.status == "limit_reached" and budget.tool_calls == before and not answer.claims


def test_numeric_or_complex_plan_keeps_existing_analyst_path(registry):
    proposed = plan()
    proposed = proposed.model_copy(update={"steps": [proposed.steps[0], proposed.steps[-1].model_copy(update={"action": "numeric_summary"})]})
    artifact = saved(registry, [{"locationDbId": "L1"}])
    assert not ca.catalog_count_plan(proposed)
    assert run(registry, proposed, [artifact]) is None


def test_validated_export_filter_can_apply_the_scope_without_a_second_filter(registry):
    artifact = saved(registry, season_rows()[:2], kind="studies")
    proposed = plan("studies", studies=("S1", "S2"), scope_season="2026")
    export_args = {"entity": "studies", "filters": {"season_id": "2026"}}
    # Test the executor's structured export binding; PlanStep's current public schema
    # disallows nested dictionaries, so normal planners cannot create this shape yet.
    proposed = proposed.model_copy(update={"steps": [proposed.steps[0].model_copy(update={"inputs": export_args}), proposed.steps[-1]]})
    answer = run(registry, proposed, [artifact], [record(artifact, export_args)])
    assert answer.status == "completed"
    assert next(c.value for c in answer.claims if c.transformation.startswith("number of distinct")) == 2


def test_export_filter_must_be_true_of_every_returned_row(registry):
    artifact = saved(registry, season_rows(), kind="studies")
    proposed = plan("studies", studies=("S1", "S2"), scope_season="2026")
    export_args = {"entity": "studies", "filters": {"season_id": "2026"}}
    proposed = proposed.model_copy(update={"steps": [proposed.steps[0].model_copy(update={"inputs": export_args}), proposed.steps[-1]]})
    answer = run(registry, proposed, [artifact], [record(artifact, export_args)])
    assert answer.status == "needs_clarification" and not answer.claims


def test_known_zero_scope_is_not_reinterpreted_as_all_studies(registry):
    artifact = saved(registry, season_rows(), kind="studies")
    answer = run(registry, plan("studies"), [artifact])
    assert answer.status == "needs_clarification" and not answer.claims
    assert "scope count" in " ".join(answer.caveats)


def test_trait_scope_cannot_be_ignored_for_location_count(registry):
    artifact = saved(registry, [{"locationDbId": "L1"}])
    proposed = plan()
    proposed = proposed.model_copy(update={"scope": proposed.scope.model_copy(update={"variable_ids": ["V1"]})})
    answer = run(registry, proposed, [artifact])
    assert answer.status == "needs_clarification" and not answer.claims
    assert "another entity" in " ".join(answer.caveats)


@pytest.mark.parametrize("direct", [False])
def test_controller_catalog_plan_completes_without_an_analyst_model(tmp_path, monkeypatch, direct):
    """Both direct and MCP paths bind actual retrieved artifacts and leave acceptance pending."""
    import asyncio
    import run as ctl
    from agents.coordinator import CoordinatorResult, KnownMetadata, KnownStudy
    from llm import FakeModelClient, reply_text, reply_tools

    proposed = plan("studies", studies=("S1",), question="Which studies are in the synthetic catalog?")
    metadata = KnownMetadata(base_url=SYNTHETIC_BASE, complete=True, studies=[KnownStudy(study_id="S1", name="SYNTHETIC trial")])
    monkeypatch.setattr(ctl.Controller, "_load_metadata_mcp", async_returning(metadata))
    part2_runtime(monkeypatch, server=SYNTHETIC_BASE)

    async def coordinator(*args, **kwargs):
        return CoordinatorResult(status="completed", plan=proposed)

    monkeypatch.setattr(ctl, "draft_plan", coordinator)
    retriever = FakeModelClient([
        reply_tools([("export_metadata", json.dumps({"entity": "studies"}))], model="mock-model"),
        reply_text(json.dumps({"status": "completed", "payload": {"resolved": [], "studies": [], "notes": []}}), model="mock-model"),
    ], reported_model="mock-model")

    def forbidden_analyst(prefix):
        raise AssertionError("A supported catalog count must not instantiate the Analyst model")

    config = part2_config(question=proposed.question, offline=False, mock_model=False, direct=direct, auto=False,
                           fixture="synthetic", cache_dir=tmp_path / "cache", out_dir=tmp_path / "out")
    controller = ctl.Controller(config, models=ctl.Models(retriever=retriever, analyst_factory=forbidden_analyst))
    result = asyncio.run(controller.run())
    assert result.execution_status == "completed", result.notes
    assert result.human_acceptance == "pending" and result.approval_id is not None
    assert result.retrieval.complete and len(result.retrieval.artifacts) == 1
    assert result.plan.scope.study_ids == ["S1"] and result.plan.scope.matching_study_count == 1
    artifact = result.retrieval.artifacts[0]
    assert controller.catalog_sources == {"step_export": [artifact.artifact_id]}
    assert artifact.kind == "studies" and artifact.row_count == 1
    assert {c.filters["studyDbId"] for c in result.analysis.claims if "studyDbId" in c.filters} == {"S1"}
    assert all(c.source_artifact_hashes == [artifact.sha256] for c in result.analysis.claims)
    assert controller.budget.tool_calls == 3 and len(retriever.requests) == 2
    assert all(not (agent.log_path or "").endswith("llm_analyst.jsonl") for agent in controller.agents)
