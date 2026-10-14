"""Controller metadata flows on synthetic cached records; scripted models only.

Run after integration under tests/conftest.py. The MCP cases deliberately exercise
the ordinary offline child transport and its registry reload.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

import chat_presenter as presenter
import run as ctl
from agents.coordinator import CoordinatorResult
from contracts import Plan, PlanStep, ResolvedScope, RunResult, ToolResult
from part2_support import SYNTHETIC_BASE
from llm import FakeModelClient, reply_text, reply_tools
from mcp_bridge import McpTools
from part2_support import part2_config, part2_runtime


@pytest.fixture(autouse=True)
def _part2_flow(monkeypatch):
    """Ported from part2: run the release flow the way part2's mock mode did (see part2_support.part2_runtime)."""
    part2_runtime(monkeypatch, server=SYNTHETIC_BASE)



def config(tmp_path, question, *, direct=True):
    return part2_config(question=question, offline=False, mock_model=False, direct=direct, auto=False,
                         fixture="synthetic", out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")


def study_plan(*, action="get_study"):
    args = {"study_db_id": "S1"} if action == "get_study" else {"name_contains": "SYNTHETIC yield trial S1"}
    return Plan(plan_id="plan_metadata_study", question="Show metadata for study S1.",
                interpretation="Read the recorded fields for exact study S1", status="ready",
                scope=ResolvedScope(base_url=SYNTHETIC_BASE, study_ids=["S1"], matching_study_count=1),
                steps=[PlanStep(step_id="step_lookup", agent="retriever", action=action, inputs=args)],
                created_at_utc=datetime.now(timezone.utc))


def variable_plan():
    return Plan(plan_id="plan_metadata_variable", question="Show the recorded method and units for variable V1.",
                interpretation="Read V1 metadata from the complete variables catalog", status="ready",
                scope=ResolvedScope(base_url=SYNTHETIC_BASE, variable_ids=["V1"]),
                steps=[PlanStep(step_id="step_export", agent="retriever", action="export_metadata", inputs={"entity": "variables"}),
                       PlanStep(step_id="step_exact", agent="analyst", action="filter_rows", depends_on=["step_export"],
                                inputs={"column": "observationVariableDbId", "op": "eq", "value": "V1"})],
                created_at_utc=datetime.now(timezone.utc))


def install_plan(monkeypatch, proposed):
    async def draft(*args, **kwargs):
        return CoordinatorResult(status="completed", plan=proposed)

    monkeypatch.setattr(ctl, "draft_plan", draft)


def retriever(*calls):
    replies = [reply_tools([(name, json.dumps(args))], model="mock-model") for name, args in calls]
    replies.append(reply_text(json.dumps({"status": "completed", "payload": {"resolved": [], "studies": [], "notes": []}}),
                              model="mock-model"))
    return FakeModelClient(replies, reported_model="mock-model")


def execute(tmp_path, proposed, model, *, direct=True):
    analyst_calls = []

    def forbidden_analyst(prefix):
        analyst_calls.append(prefix)
        pytest.fail("A metadata-only controller flow must not instantiate the numeric Analyst model")

    controller = ctl.Controller(config(tmp_path, proposed.question, direct=direct),
                                models=ctl.Models(retriever=model, analyst_factory=forbidden_analyst))
    result = asyncio.run(controller.run())
    assert analyst_calls == []
    return controller, result


def assert_saved_roundtrip(controller, result):
    paths = controller.output_paths()
    assert all(path.is_file() for path in paths.values())
    loaded = RunResult.model_validate_json(paths["answer_json"].read_text(encoding="utf-8"))
    assert loaded == result
    assert paths["answer_md"].read_text(encoding="utf-8") == controller.answer_markdown()


def assert_exact_sources(controller, result, *, kind, identifier):
    facts = result.analysis.metadata_facts
    assert facts and {(f.entity_kind, f.entity_id) for f in facts} == {(kind, identifier)}
    sources = {f.source_artifact_id: controller.ctx.registry.load(f.source_artifact_id) for f in facts}
    id_column = "studyDbId" if kind == "study" else "observationVariableDbId"
    for fact in facts:
        manifest, rows = sources[fact.source_artifact_id]
        assert manifest.complete and fact.source_artifact_hash == manifest.sha256
        assert fact.source_request_ids == manifest.source_request_ids
        matches = [row for row in rows if row[id_column] == identifier]
        assert len(matches) == 1 and fact.value == (matches[0].get(fact.field) or None)
    assert result.analysis.claims == [] and result.counts.live_attempts == 0
    assert not ({"/observations", "/observationunits"} & {r["endpoint"] for r in controller.requests})
    return {fact.field: fact for fact in facts}


@pytest.mark.parametrize("direct", [False], ids=["offline_mcp"])
def test_get_study_metadata_reaches_saved_and_chat_answers_without_an_analyst(tmp_path, monkeypatch, direct):
    proposed = study_plan()
    install_plan(monkeypatch, proposed)
    model = retriever(("get_study", {"study_db_id": "S1"}))
    controller, result = execute(tmp_path, proposed, model, direct=direct)
    assert result.execution_status == "completed", result.notes
    assert result.retrieval.complete and result.analysis.status == "completed"
    fields = assert_exact_sources(controller, result, kind="study", identifier="S1")
    assert fields["studyDbId"].value == "S1"
    assert fields["studyName"].value == "SYNTHETIC yield trial S1"
    assert fields["locationName"].value == "SYNTHETIC location one"
    assert fields["programName"].value == "SYNTHETIC program"
    assert fields["plantingDate"].value is None and fields["plantingDate"].missing_reason
    assert all(f.source_request_ids for f in fields.values())
    assert any(r["endpoint"] == "/studies/S1" for r in controller.requests)
    for shown in (controller.answer_markdown(), presenter.answer_summary(controller)):
        assert "Database ID: S1" in shown
        assert "Study name: SYNTHETIC yield trial S1" in shown
        assert "Location: SYNTHETIC location one" in shown
        assert "Planting date: not stated in the retrieved record" in shown
        assert "no complete artifact" not in shown
    assert len(model.requests) == 2
    assert_saved_roundtrip(controller, result)


@pytest.mark.parametrize("direct", [False], ids=["offline_mcp"])
def test_variable_export_exact_filter_avoids_numeric_precheck_and_excludes_v2(tmp_path, monkeypatch, direct):
    proposed = variable_plan()
    install_plan(monkeypatch, proposed)
    model = retriever(("export_metadata", {"entity": "variables"}))
    controller, result = execute(tmp_path, proposed, model, direct=direct)
    assert result.execution_status == "completed", result.notes
    fields = assert_exact_sources(controller, result, kind="variable", identifier="V1")
    assert fields["observationVariableDbId"].value == "V1"
    assert fields["observationVariableName"].value == "fresh root yield"
    assert fields["methodName"].value == "weigh all roots per plot"
    assert fields["units"].value == "t/ha"
    assert fields["methodDescription"].value is None and fields["methodDescription"].missing_reason
    # The exported source really has both variables. Selection happens against full source rows.
    _, rows = controller.ctx.registry.load(fields["units"].source_artifact_id)
    assert {row["observationVariableDbId"] for row in rows} == {"V1", "V2"}
    for shown in (controller.answer_markdown(), presenter.answer_summary(controller)):
        assert "Variable name: fresh root yield" in shown
        assert "Method name: weigh all roots per plot" in shown
        assert "Units (structured field): t/ha" in shown
        assert "dry matter content" not in shown and "oven drying" not in shown
        assert "which was not retrieved" not in shown
    assert len(model.requests) == 2
    assert_saved_roundtrip(controller, result)


@pytest.mark.parametrize("defect", ["skipped_lookup", "failed_lookup", "explicitly_incomplete", "wrong_study", "paged_search"])
def test_missing_or_incomplete_metadata_never_becomes_a_completed_answer(tmp_path, monkeypatch, defect):
    proposed = study_plan(action="search_studies" if defect == "paged_search" else "get_study")
    install_plan(monkeypatch, proposed)
    original = McpTools.call

    async def faulty(self, name, args):
        if name not in {"get_study", "search_studies"}:
            return await original(self, name, args)
        if defect == "failed_lookup":
            return ToolResult(ok=False, complete=False, error={"code": "timeout", "message": "SYNTHETIC lookup timeout"})
        row = {"studyDbId": "S2" if defect == "wrong_study" else "S1", "studyName": "SYNTHETIC yield trial S1"}
        if defect == "paged_search":
            return ToolResult(ok=True, complete=True, data={"candidates": [row], "returned": 1, "total_matches": 2,
                                                          "offset": 0, "next_offset": 1})
        return ToolResult(ok=True, complete=False if defect == "explicitly_incomplete" else None, data={"study": row})

    monkeypatch.setattr(McpTools, "call", faulty)
    step = proposed.steps[0]
    model = retriever() if defect == "skipped_lookup" else retriever((step.action, dict(step.inputs)))
    controller, result = execute(tmp_path, proposed, model)
    assert result.execution_status in {"incomplete", "blocked", "failed", "needs_clarification"}
    assert result.analysis is None or (result.analysis.status != "completed" and not result.analysis.metadata_facts)
    assert "Study name: SYNTHETIC yield trial S1" not in presenter.answer_summary(controller)
    reasons = [*result.notes, *(result.retrieval.warnings if result.retrieval else []),
               *(result.analysis.caveats if result.analysis else [])]
    assert any(note.strip() for note in reasons)
    assert result.counts.live_attempts == 0
    assert_saved_roundtrip(controller, result)
