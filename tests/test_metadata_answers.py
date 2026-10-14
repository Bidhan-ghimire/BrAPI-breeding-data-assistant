"""Synthetic metadata regressions. Run only under the project's offline conftest guards."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import brapi_mcp_server as srv
import metadata_answers as ma
from agents.analyst import AnalysisInput, precheck
from agents.retriever import ToolRecord
from contracts import AnalysisReport, Plan, PlanStep, ResolvedScope, RetrievalReport, RunResult, ToolResult
from part2_support import SYNTHETIC_BASE
from part2_support import synthetic_context


@pytest.fixture
def ctx(tmp_path):
    return synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_metadata_test")


def plan(*, studies=(), variables=(), action="get_study", question="Details of SYNTHETIC trial one", analyst=None):
    inputs = {"entity": "variables" if variables else "studies"} if action == "export_metadata" else {}
    if action == "get_study" and studies:
        inputs = {"study_db_id": studies[0]}
    steps = [PlanStep(step_id="step_1", agent="retriever", action=action, inputs=inputs)]
    if analyst:
        steps.append(PlanStep(step_id="step_2", agent="analyst", action=analyst, depends_on=["step_1"]))
    return Plan(plan_id="plan_metadata_test", question=question, interpretation="Read recorded metadata", status="ready",
                scope=ResolvedScope(base_url=SYNTHETIC_BASE, study_ids=list(studies), variable_ids=list(variables)),
                steps=steps, created_at_utc=datetime.now(timezone.utc))


def save(ctx, records, *, kind="studies", complete=True, legacy_columns=None):
    shape, columns, _ = srv.EXPORT_SHAPES[kind]
    return ctx.registry.save_table([shape(r) for r in records], kind=kind, columns=legacy_columns or columns,
                                   complete=complete, source_request_ids=["req_metadata_1"])


def retrieval(*artifacts, complete=True):
    return RetrievalReport(status="completed" if complete else "incomplete", artifacts=list(artifacts), complete=complete,
                           request_ids=["req_metadata_1"])


def facts_by_field(report):
    return {f.field: f for f in report.metadata_facts}


def study(identifier="S1", **extra):
    return {"studyDbId": identifier, "studyName": "SYNTHETIC trial one" if identifier == "S1" else "SYNTHETIC other trial",
            "experimentalDesign": {"description": "RCBD", "PUI": "RCBD"}, **extra}


def test_lookup_materializes_source_rows_without_changing_numeric_discovery(ctx):
    original = ToolResult(ok=True, data={"study": study()}, request_ids=["req_metadata_1"])
    tools = SimpleNamespace(records=[ToolRecord("get_study", {"study_db_id": "S1"}, original)])
    ma.materialize_metadata(plan(studies=("S1",), analyst="numeric_summary"), tools, ctx)
    assert tools.records[0].result.artifact_ids == [] and ctx.registry.artifact_ids() == []
    ma.materialize_metadata(plan(studies=("S1",)), tools, ctx)
    result = tools.records[0].result
    assert result.ok and result.complete and len(result.artifact_ids) == 1
    manifest = ctx.registry.manifest(result.artifact_ids[0])
    answer = ma.analyze_metadata(plan(studies=("S1",)), retrieval(manifest), ctx.registry,
                                 requests=[{"request_id": "req_metadata_1", "fetched_at_utc": "2026-09-30T02:41:27Z"}])
    assert answer.status == "completed" and answer.claims == []
    fields = facts_by_field(answer)
    assert fields["studyDbId"].value == "S1" and fields["experimentalDesignDescription"].value == "RCBD"
    assert all(f.source_artifact_hash == manifest.sha256 and f.source_request_ids == ["req_metadata_1"] for f in answer.metadata_facts)
    assert fields["studyDbId"].source_fetched_at_utc[0].year == 2026


def test_study_identity_lookup_uses_exact_saved_name_even_without_planner_scope(ctx):
    rows = [srv._study_row(study())]
    result = ToolResult(ok=True, complete=True, request_ids=["req_metadata_1"],
                        data={"candidates": rows, "returned": 1, "total_matches": 1, "offset": 0, "next_offset": None})
    saved = srv.materialize_metadata_result(ctx, "search_studies", {"name_contains": "SYNTHETIC trial one"}, result)
    answer = ma.analyze_metadata(plan(action="search_studies"), retrieval(ctx.registry.manifest(saved.artifact_ids[0])), ctx.registry)
    assert answer.status == "completed" and facts_by_field(answer)["studyDbId"].value == "S1"


@pytest.mark.parametrize("offset,next_offset,total", [(1, None, 1), (0, 1, 2), (0, None, 2)])
def test_paged_or_truncated_search_cannot_prove_complete_metadata(ctx, offset, next_offset, total):
    result = ToolResult(ok=True, complete=True, request_ids=["req_metadata_1"],
                        data={"candidates": [srv._study_row(study())], "returned": 1, "total_matches": total,
                              "offset": offset, "next_offset": next_offset})
    saved = srv.materialize_metadata_result(ctx, "search_studies", {"offset": offset}, result)
    manifest = ctx.registry.manifest(saved.artifact_ids[0])
    assert not saved.complete and not manifest.complete and manifest.row_count == 1
    answer = ma.analyze_metadata(plan(studies=("S1",)), retrieval(manifest, complete=False), ctx.registry)
    assert answer.status == "incomplete" and answer.metadata_facts == []


def test_requested_study_scope_excludes_other_exported_rows(ctx):
    artifact = save(ctx, [study(), study("S2")])
    answer = ma.analyze_metadata(plan(studies=("S1",), action="export_metadata"), retrieval(artifact), ctx.registry)
    assert answer.status == "completed" and {f.entity_id for f in answer.metadata_facts} == {"S1"}


def test_variable_method_and_unit_missingness_survives_percent_in_source_text(ctx):
    variable = {"observationVariableDbId": "V1", "observationVariableName": "Dry matter computing percent",
                "trait": {"traitDescription": "The dry matter content in percent"}, "method": {}, "scale": {"units": None}}
    artifact = save(ctx, [variable, {"observationVariableDbId": "V2", "observationVariableName": "Other trait"}], kind="variables")
    proposed = plan(variables=("V1",), action="export_metadata")
    answer = ma.analyze_metadata(proposed, retrieval(artifact), ctx.registry)
    fields = facts_by_field(answer)
    assert answer.status == "completed" and {f.entity_id for f in answer.metadata_facts} == {"V1"}
    assert fields["traitDescription"].value == "The dry matter content in percent"
    assert fields["units"].value is None and fields["methodDescription"].value is None
    assert fields["units"].missing_reason == "not stated in the retrieved record"
    assert precheck(AnalysisInput(question="Mean V1", artifact_ids=[artifact.artifact_id], requested_variable_ids=["V1"]), ctx) is not None
    assert not ma.metadata_only_plan(plan(variables=("V1",), action="export_metadata", analyst="numeric_summary"))


def test_explicit_method_and_units_are_preserved(ctx):
    artifact = save(ctx, [{"observationVariableDbId": "V1", "method": {"methodName": "Recorded method", "description": "Recorded protocol"},
                          "scale": {"units": "g", "scaleName": "Mass"}}], kind="variables")
    answer = ma.analyze_metadata(plan(variables=("V1",), action="export_metadata"), retrieval(artifact), ctx.registry)
    fields = facts_by_field(answer)
    assert fields["units"].value == "g" and fields["methodDescription"].value == "Recorded protocol"


def test_dates_keep_source_labels_and_do_not_invent_planting_or_harvest(ctx):
    artifact = save(ctx, [study(startDate="2008-06-19", endDate="2008-10-20")])
    answer = ma.analyze_metadata(plan(studies=("S1",)), retrieval(artifact), ctx.registry)
    fields = facts_by_field(answer)
    assert fields["startDate"].value == "2008-06-19" and fields["endDate"].value == "2008-10-20"
    assert fields["plantingDate"].value is None and fields["harvestDate"].value is None
    rendered = "\n".join(ma.render_metadata_facts(answer.metadata_facts))
    assert "Study start date: 2008-06-19" in rendered and "Planting date: not stated in the retrieved record" in rendered


def test_conflicting_design_sources_are_not_silently_selected(ctx):
    first = save(ctx, [study()])
    second = save(ctx, [study(experimentalDesign={"description": "CRD"})])
    answer = ma.analyze_metadata(plan(studies=("S1",)), retrieval(first, second), ctx.registry)
    assert answer.status == "needs_clarification" and answer.metadata_facts == []
    assert any("disagree" in c for c in answer.caveats)


def test_missing_and_ambiguous_entity_never_become_absence_proof(ctx):
    artifact = save(ctx, [study()])
    answer = ma.analyze_metadata(plan(studies=("S9",)), retrieval(artifact), ctx.registry)
    assert answer.status == "needs_clarification" and answer.metadata_facts == []
    assert "not evidence" in " ".join(answer.caveats)
    ambiguous = save(ctx, [study(), study("S2", studyName="SYNTHETIC trial one")])
    answer = ma.analyze_metadata(plan(action="search_studies"), retrieval(ambiguous), ctx.registry)
    assert answer.status == "needs_clarification" and "More than one" in " ".join(answer.caveats)


def test_legacy_table_columns_are_not_claimed_missing_from_database(ctx):
    artifact = save(ctx, [study()], legacy_columns={"studyDbId": "ID", "studyName": "name"})
    answer = ma.analyze_metadata(plan(studies=("S1",)), retrieval(artifact), ctx.registry)
    assert answer.status == "completed"
    assert {f.field for f in answer.metadata_facts} == {"studyDbId", "studyName"}
    assert "absent from the supplied table schema" in " ".join(answer.caveats)


def test_metadata_values_are_revalidated_against_hash_checked_rows(ctx):
    artifact = save(ctx, [study()])
    answer = ma.analyze_metadata(plan(studies=("S1",)), retrieval(artifact), ctx.registry)
    fact = facts_by_field(answer)["experimentalDesignDescription"]
    for forged in (fact.model_copy(update={"value": "invented"}), fact.model_copy(update={"source_artifact_hash": "f" * 64})):
        with pytest.raises(ValueError):
            ma.validate_metadata_facts([forged], ctx.registry)
    with pytest.raises(ValidationError, match="at least one claim"):
        AnalysisReport(status="completed")
    assert AnalysisReport(status="completed", metadata_facts=[fact]).status == "completed"


def test_run_contract_requires_metadata_provenance_to_match_retrieval(ctx):
    artifact = save(ctx, [study()])
    proposed = plan(studies=("S1",))
    fetched = retrieval(artifact)
    answer = ma.analyze_metadata(proposed, fetched, ctx.registry)
    base = dict(run_id="run_metadata_test", plan=proposed, retrieval=fetched, analysis=answer,
                artifacts=[artifact], execution_status="completed", started_at_utc=datetime.now(timezone.utc))
    assert RunResult(**base).analysis.metadata_facts
    fact = answer.metadata_facts[0]
    for change in ({"source_artifact_hash": "f" * 64}, {"source_request_ids": ["req_other"]}, {"field": "inventedField"}):
        bad = answer.model_copy(update={"metadata_facts": [fact.model_copy(update=change)]})
        with pytest.raises(ValidationError):
            RunResult(**{**base, "analysis": bad})
    incomplete = artifact.model_copy(update={"complete": False})
    with pytest.raises(ValidationError):
        RunResult(**{**base, "artifacts": [incomplete], "retrieval": retrieval(incomplete)})


def test_clarification_number_cannot_supply_a_requested_entity_id(ctx):
    artifact = save(ctx, [{"studyDbId": "1", "studyName": "Unmentioned trial"}])
    proposed = plan(action="search_studies", question="Please explain this\nClarification 1 from the person: anything")
    answer = ma.analyze_metadata(proposed, retrieval(artifact), ctx.registry)
    assert answer.status == "needs_clarification" and not answer.metadata_facts
    assert answer.caveats[0].startswith("analyst asks:")


def test_mismatched_retrieval_manifest_is_not_used_for_metadata(ctx):
    artifact = save(ctx, [study()])
    forged = artifact.model_copy(update={"sha256": "e" * 64})
    answer = ma.analyze_metadata(plan(studies=("S1",)), retrieval(forged), ctx.registry)
    assert answer.status == "incomplete" and not answer.metadata_facts
