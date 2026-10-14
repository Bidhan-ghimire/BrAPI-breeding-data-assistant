"""Catalog completion regressions: synthetic tables and scripted models, no live calls.

Draft only. Run with the project tests/conftest.py safety rails after integration.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

import agents.retriever as rt
import brapi_mcp_server as srv
import run as ctl
from agents.coordinator import CoordinatorResult, KnownMetadata, KnownStudy
from contracts import AgentResult, Plan, PlanStep, ResolvedScope, ToolResult
from part2_support import SYNTHETIC_BASE
from llm import FakeModelClient, reply_text, reply_tools
from mcp_bridge import McpTools
from part2_support import async_returning, part2_config, part2_runtime, synthetic_approval, synthetic_context


@pytest.fixture(autouse=True)
def _part2_flow(monkeypatch):
    """Ported from part2: run the release flow the way part2's mock mode did (see part2_support.part2_runtime)."""
    part2_runtime(monkeypatch, server=SYNTHETIC_BASE)



EXPORT = {"entity": "studies"}
ROWS = [{"studyDbId": f"S{i + 1}", "locationDbId": location}
        for i, location in enumerate(["12"] * 8 + ["80"] * 4 + ["13"] * 2)]
COLUMNS = {"studyDbId": "SYNTHETIC study ID", "locationDbId": "SYNTHETIC location ID"}


def agent(status="completed"):
    payload = {"resolved": [], "studies": [], "notes": []} if status == "completed" else {"question": "Which catalog?"}
    return AgentResult(status=status, payload=payload, model_requested="mock-model",
                       model_calls=0, tool_calls=0, elapsed_seconds=0.0,
                       errors=[{"code": "synthetic_failure", "message": "SYNTHETIC failure"}] if status == "failed" else [])


def final_reply():
    return reply_text(json.dumps({"status": "completed", "payload": agent().payload}), model="mock-model")


@pytest.fixture
def context(tmp_path):
    return synthetic_context(cache_dir=tmp_path / "cache",
                             out_dir=tmp_path / "out", run_id="run_catalog_completion")


def table_result(context, *, kind="studies", complete=True, rows=None):
    records = ROWS if rows is None else rows
    manifest = context.registry.save_table(records, kind=kind, columns=COLUMNS, complete=complete)
    return ToolResult(ok=True, complete=complete, artifact_ids=[manifest.artifact_id],
                      data={"preview_rows": records[:5], "row_count": len(records)})


def report(context, tools, expected=None):
    return rt.build_report(agent(), tools, context, expects_observations=False,
                           expected_catalog_exports=[EXPORT] if expected is None else expected)


def test_catalog_keys_validate_and_canonicalize_optional_filters():
    key = rt.catalog_export_key(EXPORT)
    assert rt.catalog_export_key({"entity": "studies", "filters": None}) == key
    assert rt.catalog_export_key({"entity": "studies", "filters": {}}) == key
    assert rt.catalog_export_key({"entity": "studies", "filters": {"location_id": None}}) == key
    left = {"entity": "studies", "filters": {"location_id": "12", "season_id": "2026"}}
    right = {"filters": {"season_id": "2026", "location_id": "12"}, "entity": "studies"}
    assert rt.catalog_export_key(left) == rt.catalog_export_key(right)
    assert rt.catalog_export_key(left) != key
    for invalid in ({"entity": "invented"}, {"entity": "studies", "extra": True},
                    {"entity": "studies", "filters": {"location_id": 12}}):
        with pytest.raises(ValidationError):
            rt.catalog_export_key(invalid)


@pytest.mark.parametrize("defect", ["missing", "wrong_entity", "wrong_filter", "fabricated_artifact",
                                    "wrong_artifact_kind", "incomplete_result", "incomplete_artifact", "failed"])
def test_catalog_report_requires_exact_successful_registered_export(context, defect):
    tools = rt.ApprovedTools(None, context, None)
    if defect != "missing":
        args = dict(EXPORT)
        if defect == "wrong_entity":
            args = {"entity": "variables"}
        if defect == "wrong_filter":
            args = {"entity": "studies", "filters": {"location_id": "12"}}
        if defect == "fabricated_artifact":
            result = ToolResult(ok=True, complete=True, artifact_ids=["art_abcdef_9999"])
        elif defect == "failed":
            result = ToolResult(ok=False, error={"code": "timeout", "message": "SYNTHETIC timeout"})
        else:
            result = table_result(context, kind="variables" if defect in ("wrong_entity", "wrong_artifact_kind") else "studies",
                                  complete=defect != "incomplete_artifact")
            if defect == "incomplete_result":
                result = result.model_copy(update={"complete": False})
            elif defect == "incomplete_artifact":
                result = result.model_copy(update={"complete": True})
        tools.records.append(rt.ToolRecord("export_metadata", args, result))
    outcome = report(context, tools)
    assert outcome.status == "incomplete" and not outcome.complete
    if defect in ("missing", "fabricated_artifact", "failed"):
        assert outcome.artifacts == []
    else:
        assert outcome.artifacts
    assert any("catalog" in note.lower() or "incomplete artifact" in note.lower() for note in outcome.warnings)


def test_catalog_report_explains_a_missing_catalog_without_claiming_zero(context):
    outcome = report(context, rt.ApprovedTools(None, context, None))
    notes = " ".join(outcome.warnings)
    assert "studies" in notes and "No catalog count was produced" in notes
    assert "no complete artifact is eligible" not in notes


def test_complete_empty_catalog_is_valid_and_every_planned_export_is_required(context):
    tools = rt.ApprovedTools(None, context, None)
    tools.records.append(rt.ToolRecord("export_metadata", {"entity": "studies", "filters": {}}, table_result(context, rows=[])))
    outcome = report(context, tools)
    assert outcome.status == "completed" and outcome.complete
    assert outcome.artifacts[0].row_count == 0
    assert report(context, tools, [EXPORT, {"entity": "variables"}]).status == "incomplete"


def config(tmp_path, **overrides):
    values = dict(question="Which locations have the most studies?", offline=False, mock_model=False,
                  direct=False, auto=False, fixture="synthetic", out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    values.update(overrides)
    return part2_config(**values)


def catalog_plan():
    return Plan(plan_id="plan_catalog_completion", question="Which locations have the most studies?",
                interpretation="Count all study rows by location ID", status="ready",
                scope=ResolvedScope(base_url=SYNTHETIC_BASE, study_ids=[r["studyDbId"] for r in ROWS], matching_study_count=len(ROWS)),
                steps=[PlanStep(step_id="step_1", agent="retriever", action="export_metadata", inputs=EXPORT),
                       PlanStep(step_id="step_2", agent="analyst", action="count_records",
                                inputs={"group_by": "locationDbId"}, depends_on=["step_1"])],
                created_at_utc=datetime.now(timezone.utc))


def install_catalog_scenario(monkeypatch, *, export_failure=None):
    """Replace only discovery/model planning and the export reply; real controller/registry/analyst remain."""
    metadata = KnownMetadata(base_url=SYNTHETIC_BASE, complete=True,
                             studies=[KnownStudy(study_id=r["studyDbId"], location_id=r["locationDbId"]) for r in ROWS])
    monkeypatch.setattr(ctl.Controller, "_load_metadata_mcp", async_returning(metadata))

    async def draft(*args, **kwargs):
        return CoordinatorResult(status="completed", plan=catalog_plan())

    monkeypatch.setattr(ctl, "draft_plan", draft)
    original = McpTools.call
    calls = []

    async def export(self, name, args):
        if name != "export_metadata":
            return await original(self, name, args)
        calls.append(dict(args))
        if export_failure == "failed":
            return ToolResult(ok=False, error={"code": "timeout", "message": "SYNTHETIC timeout"})
        return table_result(self.ctx, complete=export_failure != "incomplete")

    monkeypatch.setattr(McpTools, "call", export)
    return calls


def count_model(prefix):
    handle = "art_" + prefix + "_0001"
    return FakeModelClient([
        reply_tools([("table_info", json.dumps({"artifact_id": handle}))], model="mock-model"),
        reply_tools([("count_records", json.dumps({"artifact_id": handle, "group_by": "locationDbId"}))], model="mock-model"),
        reply_text(json.dumps({"status": "completed", "payload": {"methods": ["Count every fetched study row by location ID"],
                                                                   "narrative": "Counts are from the full fetched table."}}), model="mock-model")
    ], reported_model="mock-model")


def test_controller_recovers_skipped_export_and_counts_the_full_table(tmp_path, monkeypatch):
    calls = install_catalog_scenario(monkeypatch)
    model = FakeModelClient([final_reply()], reported_model="mock-model")
    controller = ctl.Controller(config(tmp_path), models=ctl.Models(retriever=model, analyst_factory=count_model))
    result = asyncio.run(controller.run())
    assert result.execution_status == "completed", result.notes
    assert calls == [EXPORT]
    assert result.retrieval.complete and len(result.retrieval.artifacts) == 1
    assert result.retrieval.artifacts[0].row_count == 14
    frequencies = {c.filters["locationDbId"]: c.value for c in result.analysis.claims if "locationDbId" in c.filters}
    assert frequencies == {"12": 8, "80": 4, "13": 2}  # preview has only five rows, all at location 12
    assert all(c.denominator.n == 14 for c in result.analysis.claims)
    assert any("skipped planned export_metadata" in note for note in result.notes)
    assert controller.budget.tool_calls == 3  # recovered export + inspection + grouped count
    assert len(model.requests) == 1  # no second Retriever/model pass


def test_cancel_at_approval_performs_no_export_or_retriever_call(tmp_path, monkeypatch):
    calls = install_catalog_scenario(monkeypatch)
    model = FakeModelClient([final_reply()], reported_model="mock-model")
    controller = ctl.Controller(config(tmp_path, auto=False), models=ctl.Models(retriever=model), answer_fn=lambda prompt: "cancel")
    result = asyncio.run(controller.run())
    assert result.execution_status == "canceled"
    assert controller.approval is None and result.retrieval is None
    assert calls == [] and model.requests == [] and controller.budget.tool_calls == 0


@pytest.mark.parametrize("failure", ["failed", "incomplete"])
def test_an_attempted_unsuccessful_export_is_not_retried(tmp_path, monkeypatch, failure):
    calls = install_catalog_scenario(monkeypatch, export_failure=failure)
    model = FakeModelClient([reply_tools([("export_metadata", json.dumps(EXPORT))], model="mock-model"), final_reply()],
                            reported_model="mock-model")
    controller = ctl.Controller(config(tmp_path), models=ctl.Models(retriever=model, analyst_factory=count_model))
    result = asyncio.run(controller.run())
    assert calls == [EXPORT] and controller.budget.tool_calls == 1
    assert result.execution_status == "incomplete" and result.analysis is None
    assert not result.retrieval.complete
    assert not any("controller executed skipped" in note for note in result.notes)


class RecordingInner:
    def __init__(self, context):
        self.context = context
        self.calls = []

    async def call(self, name, args):
        self.calls.append((name, dict(args)))
        return table_result(self.context)


def recovery_setup(tmp_path, context):
    controller = ctl.Controller(config(tmp_path))
    controller.approval = synthetic_approval(run_id=controller.run_id)
    inner = RecordingInner(context)
    tools = rt.ApprovedTools(inner, context, controller.approval)
    return controller, inner, tools


@pytest.mark.parametrize("status", ["blocked", "needs_clarification", "failed", "limit_reached"])
def test_catalog_recovery_never_continues_after_noncompletion(tmp_path, context, status):
    controller, inner, tools = recovery_setup(tmp_path, context)
    asyncio.run(controller._finish_catalog_exports(tools, agent(status), [EXPORT]))
    assert inner.calls == [] and tools.records == [] and controller.budget.tool_calls == 0


@pytest.mark.parametrize("constraint", ["no_approval", "expired_approval", "tool_budget", "time_budget"])
def test_catalog_recovery_obeys_approval_and_remaining_budgets(tmp_path, context, monkeypatch, constraint):
    controller, inner, tools = recovery_setup(tmp_path, context)
    if constraint == "no_approval":
        controller.approval = None
    elif constraint == "expired_approval":
        controller.now = lambda: datetime.now(timezone.utc) + timedelta(days=1)
    elif constraint == "tool_budget":
        controller.budget.tool_calls = controller.budget.max_tool_calls
    else:
        controller.budget.started = 1.0
        monkeypatch.setattr(ctl, "_clock", lambda: 1.0 + controller.budget.max_elapsed_seconds)
    before = controller.budget.tool_calls
    asyncio.run(controller._finish_catalog_exports(tools, agent(), [EXPORT]))
    assert inner.calls == [] and tools.records == [] and controller.budget.tool_calls == before
    assert report(context, tools).status == "incomplete"
    assert any("recovery stopped" in note for note in controller.notes)


def test_equivalent_duplicate_planned_exports_are_dispatched_once(tmp_path, context):
    controller, inner, tools = recovery_setup(tmp_path, context)
    asyncio.run(controller._finish_catalog_exports(tools, agent(), [EXPORT, {"entity": "studies", "filters": {}}]))
    assert inner.calls == [("export_metadata", EXPORT)]
    assert controller.budget.tool_calls == 1 and len(tools.records) == 1


def test_wrong_filter_attempt_does_not_suppress_the_planned_full_export(tmp_path, context):
    controller, inner, tools = recovery_setup(tmp_path, context)
    narrowed = {"entity": "studies", "filters": {"location_id": "12"}}
    tools.records.append(rt.ToolRecord("export_metadata", narrowed, table_result(context, rows=ROWS[:8])))
    asyncio.run(controller._finish_catalog_exports(tools, agent(), [EXPORT]))
    assert inner.calls == [("export_metadata", EXPORT)]
    assert report(context, tools).complete


def test_skipped_catalog_export_is_recovered_through_offline_mcp_child(tmp_path, monkeypatch):
    """Child-owned registry is reloaded after closure; Analyst reads the recovered artifact."""
    metadata = KnownMetadata(base_url=SYNTHETIC_BASE, complete=True,
                             studies=[KnownStudy(study_id="S1", location_id="L1")])
    proposed = catalog_plan().model_copy(update={"scope": ResolvedScope(base_url=SYNTHETIC_BASE,
                                                                       study_ids=["S1"], matching_study_count=1)})
    monkeypatch.setattr(ctl.Controller, "_load_metadata_mcp", async_returning(metadata))

    async def draft(*args, **kwargs):
        return CoordinatorResult(status="completed", plan=proposed)

    monkeypatch.setattr(ctl, "draft_plan", draft)
    model = FakeModelClient([final_reply()], reported_model="mock-model")
    controller = ctl.Controller(config(tmp_path, direct=False),
                                models=ctl.Models(retriever=model, analyst_factory=count_model))
    result = asyncio.run(controller.run())
    assert result.execution_status == "completed", result.notes
    assert result.retrieval.complete and len(result.retrieval.artifacts) == 1
    manifest = result.retrieval.artifacts[0]
    assert manifest.kind == "studies" and manifest.row_count == 1
    assert controller.ctx is not None
    assert controller.ctx.registry.manifest(manifest.artifact_id).sha256 == manifest.sha256
    assert [(c.filters.get("locationDbId"), c.value) for c in result.analysis.claims] == [("L1", 1)]
    assert all(c.source_artifact_hashes == [manifest.sha256] for c in result.analysis.claims)
    assert controller.budget.tool_calls == 3 and len(model.requests) == 1
    assert sum("controller executed skipped planned export_metadata" in note for note in result.notes) == 1
