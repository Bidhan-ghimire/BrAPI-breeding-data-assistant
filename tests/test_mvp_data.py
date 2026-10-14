"""MVP evidence regressions; synthetic in-memory models and temporary artifacts only."""
from __future__ import annotations

import pytest

import agents.analyst as analyst
import agents.retriever as retriever
import analyst_tools as at
import brapi_mcp_server as srv
from artifacts import ArtifactRegistry
from contracts import AgentResult, CollectionResult, MeasurementMeta, ToolResult
from part2_support import SYNTHETIC_BASE
from part2_support import synthetic_context


def completed_agent():
    return AgentResult(status="completed", payload={"resolved": [], "studies": [], "notes": []},
                       model_requested="synthetic-script", model_calls=0, tool_calls=0, elapsed_seconds=0.0)


@pytest.fixture
def context(tmp_path):
    return synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out",
                             run_id="run_mvp_data")


def recorded_fetch(context, tools, study, variable=None, *, complete=True, records=None, artifact_study=None):
    name = "get_observation_units" if variable is None else "get_observations"
    kind = "observation_units" if variable is None else "observations"
    artifact = context.registry.save_table(records or [], kind=kind, columns={"value": "SYNTHETIC value"},
                                          complete=complete, study_ids=[artifact_study or study],
                                          variable_ids=[] if variable is None else [variable])
    args = {"study_db_id": study}
    if variable is not None:
        args["variable_db_id"] = variable
    tools.records.append(retriever.ToolRecord(name, args, ToolResult(ok=True, complete=complete,
                                                                   artifact_ids=[artifact.artifact_id])))
    return artifact


def test_missing_planned_study_cannot_be_omitted_by_model(context):
    tools = retriever.ApprovedTools(None, context, None)
    recorded_fetch(context, tools, "S1", "V1")
    report = retriever.build_report(completed_agent(), tools, context,
                                   expected_observation_pairs={("S1", "V1"), ("S2", "V1")})
    assert report.status == "incomplete" and not report.complete
    assert any("study='S2', variable='V1'" in w for w in report.warnings)


def test_variable_union_does_not_prove_each_requested_study_pair(context):
    tools = retriever.ApprovedTools(None, context, None)
    recorded_fetch(context, tools, "S1", "V1")
    recorded_fetch(context, tools, "S2", "V2")
    expected = {(s, v) for s in ("S1", "S2") for v in ("V1", "V2")}
    report = retriever.build_report(completed_agent(), tools, context, expected_observation_pairs=expected)
    assert report.status == "incomplete"
    warning = next(w for w in report.warnings if "planned fetches" in w)
    assert "study='S1', variable='V2'" in warning and "study='S2', variable='V1'" in warning


def test_complete_empty_fetch_covers_scope_but_missing_roster_does_not(context):
    tools = retriever.ApprovedTools(None, context, None)
    recorded_fetch(context, tools, "S1", "V1")
    expected = {"expected_observation_pairs": {("S1", "V1")}, "expected_unit_studies": {"S1"}}
    assert retriever.build_report(completed_agent(), tools, context, **expected).status == "incomplete"
    recorded_fetch(context, tools, "S1")
    report = retriever.build_report(completed_agent(), tools, context, **expected)
    assert report.complete and report.status == "completed"
    assert all(a.row_count == 0 for a in report.artifacts)


def test_request_label_does_not_override_wrong_artifact_scope(context):
    tools = retriever.ApprovedTools(None, context, None)
    recorded_fetch(context, tools, "S1", "V1", artifact_study="S2")
    report = retriever.build_report(completed_agent(), tools, context, expected_observation_pairs={("S1", "V1")})
    assert report.status == "incomplete"


def test_incomplete_request_does_not_cover_expected_pair(context):
    tools = retriever.ApprovedTools(None, context, None)
    recorded_fetch(context, tools, "S1", "V1", complete=False)
    report = retriever.build_report(completed_agent(), tools, context, expected_observation_pairs={("S1", "V1")})
    assert report.status == "incomplete" and not report.complete


def test_catalog_plan_has_no_observation_requirements(context):
    tools = retriever.ApprovedTools(None, context, None)
    result = srv.dispatch(context, "export_metadata", {"entity": "studies"})
    tools.records.append(retriever.ToolRecord("export_metadata", {"entity": "studies"}, result))
    report = retriever.build_report(completed_agent(), tools, context, expects_observations=False)
    assert report.complete and report.status == "completed"


def test_known_units_enter_manifest_and_claim_without_metadata_request(context):
    context.variable_measurements["V1"] = MeasurementMeta(trait="SYNTHETIC yield", unit="t/ha")
    result = srv.dispatch(context, "get_observations", {"study_db_id": "S1", "variable_db_id": "V1"})
    assert result.ok
    manifest = context.registry.manifest(result.artifact_ids[0])
    assert manifest.measurement.unit == "t/ha"
    summary = at.dispatch_analyst(context.registry, "numeric_summary", {"artifact_id": manifest.artifact_id})
    assert summary.ok
    assert all(c["unit"] == "t/ha" for c in summary.data["claims"] if c["kind"] != "count")
    assert [r.endpoint for r in context.client.provenance()] == ["/observations"]


def test_other_variable_cannot_supply_unit(context):
    context.variable_measurements["V2"] = MeasurementMeta(unit="%")
    result = srv.dispatch(context, "get_observations", {"study_db_id": "S1", "variable_db_id": "V1"})
    assert context.registry.manifest(result.artifact_ids[0]).measurement.unit is None


def test_empty_observations_still_keep_known_unit(context, monkeypatch):
    context.variable_measurements["V1"] = MeasurementMeta(unit="t/ha")
    monkeypatch.setattr(context.client, "observations", lambda *a, **k: CollectionResult(
        records=[], returned_count=0, reported_total=0, complete=True, status="empty"))
    result = srv.dispatch(context, "get_observations", {"study_db_id": "S1", "variable_db_id": "V1"})
    manifest = context.registry.manifest(result.artifact_ids[0])
    assert result.complete and manifest.row_count == 0 and manifest.measurement.unit == "t/ha"


def test_scale_name_is_not_invented_as_measurement_unit():
    row = srv._variable_row({"observationVariableDbId": "V-ordinal", "scale": {"scaleName": "Ordinal 0-5"}})
    assert row["scaleName"] == "Ordinal 0-5" and row["units"] in (None, "")
    explicit = srv._variable_row({"scale": {"scaleName": "Root yield", "units": "t/ha"}})
    assert explicit["units"] == "t/ha"


@pytest.mark.parametrize("left,right", [
    (("group", "A/B"), ("group", "A?B")),
    (("group", "α"), ("group", "β")),
    (("a_b", "c"), ("a", "b_c")),
    (("group", "A/B"), ("group", "x0412f42")),
    (("group", ""), ("group", "x0")),
])
def test_claim_components_cannot_collide(left, right):
    assert at._claim_id("art_123abc_0001", "count", *left) != at._claim_id("art_123abc_0001", "count", *right)


def test_plain_synthetic_claim_ids_remain_compatible():
    assert at._claim_id("art_123abc_0001", "mean", "germplasmDbId", "GA") == "clm_123abc_0001_mean_germplasmDbId_GA"
    assert at._claim_id("art_123abc_0001", "mean") == "clm_123abc_0001_mean"


def test_punctuation_groups_survive_analyst_harvest(tmp_path):
    registry = ArtifactRegistry(tmp_path / "out", "run_collision", base_url=SYNTHETIC_BASE)
    art = registry.save_table([{"group": value} for value in ("A/B", "A?B", "x0412f42", "α", "β")],
                              kind="studies", columns={"group": "SYNTHETIC group"}, complete=True)
    result = at.dispatch_analyst(registry, "count_records", {"artifact_id": art.artifact_id, "group_by": "group"})
    assert result.ok and len(result.data["claims"]) == 5
    tools = analyst.GuardedAnalystTools(at.AnalystTools(registry), [art.artifact_id])
    tools.records.append(analyst.ToolRecord("count_records", {"artifact_id": art.artifact_id}, result))
    claims, *_ = analyst._harvest(tools, {art.artifact_id})
    assert len(claims) == 5 and {c.filters["group"] for c in claims} == {"A/B", "A?B", "x0412f42", "α", "β"}


def test_frequency_and_distinct_operations_have_separate_claim_ids(tmp_path):
    registry = ArtifactRegistry(tmp_path / "out", "run_operation", base_url=SYNTHETIC_BASE)
    art = registry.save_table([{"distinct": "field", "field": "S1"}], kind="studies",
                              columns={"distinct": "SYNTHETIC text", "field": "SYNTHETIC ID"}, complete=True)
    result = at.dispatch_analyst(registry, "count_records", {"artifact_id": art.artifact_id,
                                                          "group_by": "distinct", "distinct_by": "field"})
    assert result.ok
    ids = [c["claim_id"] for c in result.data["claims"]]
    assert len(ids) == len(set(ids)) == 2


def test_alternate_numeric_column_does_not_reuse_default_claim_id(tmp_path):
    registry = ArtifactRegistry(tmp_path / "out", "run_column", base_url=SYNTHETIC_BASE)
    art = registry.save_table([{"value": "2", "other": "7", "observationUnitDbId": "P1"}], kind="observations",
                              columns={"value": "SYNTHETIC value", "other": "SYNTHETIC other value", "observationUnitDbId": "unit"},
                              complete=True, variable_ids=["V1"])
    first = at.dispatch_analyst(registry, "numeric_summary", {"artifact_id": art.artifact_id})
    second = at.dispatch_analyst(registry, "numeric_summary", {"artifact_id": art.artifact_id, "value": "other"})
    assert first.ok and second.ok
    assert {c["claim_id"] for c in first.data["claims"]}.isdisjoint(c["claim_id"] for c in second.data["claims"])


def test_conflicting_same_id_is_withheld_and_later_repeat_cannot_restore_it(tmp_path):
    registry = ArtifactRegistry(tmp_path / "out", "run_conflict", base_url=SYNTHETIC_BASE)
    art = registry.save_table([{"group": "A"}, {"group": "B"}], kind="studies",
                              columns={"group": "SYNTHETIC group"}, complete=True)
    result = at.dispatch_analyst(registry, "count_records", {"artifact_id": art.artifact_id, "group_by": "group"})
    assert result.ok
    first = result.data["claims"][0]
    disagreement = result.model_copy(update={"data": {**result.data, "claims": [{**first, "value": 99}]}})
    tools = analyst.GuardedAnalystTools(at.AnalystTools(registry), [art.artifact_id])
    for evidence in (result, disagreement, result):
        tools.records.append(analyst.ToolRecord("count_records", {"artifact_id": art.artifact_id}, evidence))
    claims, _, caveats, _, incomplete = analyst._harvest(tools, {art.artifact_id})
    assert incomplete and len(claims) == 1 and claims[0].claim_id != first["claim_id"]
    assert any("conflicting tool claims" in text for text in caveats)
    agent = completed_agent().model_copy(update={"payload": {"methods": [], "caveats": [], "key_claim_ids": []}})
    report, _ = analyst.build_report(agent, tools, analyst.AnalysisInput(question="SYNTHETIC count", artifact_ids=[art.artifact_id]))
    assert report.status == "incomplete" and all(c.claim_id != first["claim_id"] for c in report.claims)


def test_identical_claim_repetition_remains_a_completed_answer(tmp_path):
    registry = ArtifactRegistry(tmp_path / "out", "run_same_claim", base_url=SYNTHETIC_BASE)
    art = registry.save_table([{"group": "A"}], kind="studies", columns={"group": "SYNTHETIC group"}, complete=True)
    result = at.dispatch_analyst(registry, "count_records", {"artifact_id": art.artifact_id})
    assert result.ok
    tools = analyst.GuardedAnalystTools(at.AnalystTools(registry), [art.artifact_id])
    tools.records = [analyst.ToolRecord("count_records", {"artifact_id": art.artifact_id}, result)] * 2
    claims, _, caveats, _, incomplete = analyst._harvest(tools, {art.artifact_id})
    assert not incomplete and len(claims) == 1 and not caveats
    agent = completed_agent().model_copy(update={"payload": {"methods": [], "caveats": [], "key_claim_ids": []}})
    report, _ = analyst.build_report(agent, tools, analyst.AnalysisInput(question="SYNTHETIC count", artifact_ids=[art.artifact_id]))
    assert report.status == "completed" and len(report.claims) == 1
