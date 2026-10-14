"""Fixed catalog counts traverse real Analyst MCP, with exact lineage and no model."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import anyio
import pytest

import catalog_answers as ca
from brapi_mcp_server import build_context
from contracts import Plan, PlanStep, ResolvedScope, RetrievalReport, ToolResult
from part2_support import SYNTHETIC_BASE
from llm import Budget
from mcp_bridge import McpTools
from part2_support import analysis_service, synthetic_context


@pytest.fixture
def ctx(tmp_path):
    return synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out")


def _plan(*, season=False):
    entity = "studies" if season else "locations"
    identity = ca.ENTITY_IDS[entity]
    steps = [PlanStep(step_id="step_export", agent="retriever", action="export_metadata", inputs={"entity": entity})]
    if season:
        steps.append(PlanStep(step_id="step_filter", agent="analyst", action="filter_rows",
                              inputs={"column": "seasonDbId", "op": "eq", "value": "2026"},
                              depends_on=["step_export"]))
    steps.append(PlanStep(step_id="step_count", agent="analyst", action="count_records",
                          inputs={"group_by": identity, "distinct_by": identity}, depends_on=[steps[-1].step_id]))
    scope = ResolvedScope(base_url=SYNTHETIC_BASE, study_ids=["S1", "S2"] if season else [],
                          season_ids=["2026"] if season else [], filters={"season_id": "2026" if season else None},
                          matching_study_count=2 if season else None)
    return Plan(plan_id="plan_catalog_mcp", question="How many studies have a 2026 season?" if season else "How many locations?",
                interpretation="Count the complete catalog", status="ready", scope=scope, steps=steps,
                statistic={"kind": "count", "denominator": "complete catalog rows", "grouping": [identity]},
                created_at_utc=datetime.now(timezone.utc))


def _save(ctx, rows=None, *, season=False, complete=True):
    if season:
        rows = rows if rows is not None else [
            {"studyDbId": "S1", "studyName": "one", "seasons": '["2026"]'},
            {"studyDbId": "S2", "studyName": "two", "seasons": '["2025", "2026"]'},
            {"studyDbId": "S3", "studyName": "three", "seasons": '["12026"]'},
        ]
        columns = {"studyDbId": "ID", "studyName": "name", "seasons": "JSON season IDs"}
    else:
        rows = rows if rows is not None else [{"locationDbId": "L1", "locationName": "one"},
                                               {"locationDbId": "L2", "locationName": "two"}]
        columns = {"locationDbId": "ID", "locationName": "name"}
    return ctx.registry.save_table(rows, kind="studies" if season else "locations", columns=columns, complete=complete)


def _retrieval(artifacts):
    return RetrievalReport(status="completed", complete=True, artifacts=artifacts)


def _mcp(ctx, proposed, artifacts, *, budget=None, export_sources=None):
    """The only executed math lives behind actual initialize/list/call protocol messages."""
    ids = [a.artifact_id for a in artifacts]
    budget = budget if budget is not None else Budget()

    async def scenario():
        service = await analysis_service(ctx, ids)
        async with McpTools.memory(ctx, server_factory=lambda owned: service.server) as tools:
            await tools.schemas()
            report = await ca.run_catalog_plan_async(proposed, _retrieval(artifacts), ctx.registry, tools=tools,
                export_sources=export_sources if export_sources is not None else {"step_export": ids}, budget=budget)
        return report, tools.protocol_events, budget

    return anyio.run(scenario)


def _calls(events):
    return [event["tool"] for event in events if event["method"] == "tools/call"]


def _no_direct_fallback(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Catalog client must not dispatch a local Analyst tool")
    # Replace the client's alias only: the server still owns the real dispatcher.
    monkeypatch.setattr(ca, "at", SimpleNamespace(dispatch_analyst=forbidden))


def test_locations_count_uses_analyst_mcp_and_never_direct_fallback(ctx, monkeypatch):
    source = _save(ctx)
    _no_direct_fallback(monkeypatch)
    answer, events, budget = _mcp(ctx, _plan(), [source])
    assert answer.status == "completed", answer.caveats
    assert [e["method"] for e in events] == ["initialize", "tools/list", "tools/call", "tools/call"]
    assert _calls(events) == ["table_info", "count_records"]
    assert next(c.value for c in answer.claims if c.transformation.startswith("number of distinct")) == 2
    assert all(c.source_artifact_hashes == [source.sha256] for c in answer.claims)
    assert budget.tool_calls == 2 and budget.model_calls == 0
    assert not ctx.client.provenance()


def test_every_equivalent_export_is_inspected_before_mcp_count(ctx):
    one, repeat = _save(ctx), _save(ctx)
    assert one.artifact_id != repeat.artifact_id and one.sha256 == repeat.sha256
    answer, events, budget = _mcp(ctx, _plan(), [one, repeat])
    assert answer.status == "completed", answer.caveats
    assert _calls(events) == ["table_info", "table_info", "count_records"]
    assert all(any(handle in method for method in answer.methods[:2]) for handle in [one.artifact_id, repeat.artifact_id])
    assert budget.tool_calls == 3 and budget.model_calls == 0


def test_equivalent_source_inspections_consume_shared_budget(ctx):
    one, repeat = _save(ctx), _save(ctx)
    answer, events, budget = _mcp(ctx, _plan(), [one, repeat], budget=Budget(max_tool_calls=2))
    assert answer.status == "limit_reached" and not answer.claims
    assert _calls(events) == ["table_info", "table_info"] and budget.tool_calls == 2


def test_exact_season_filter_and_derived_hashes_survive_mcp(ctx):
    source = _save(ctx, season=True)
    answer, events, budget = _mcp(ctx, _plan(season=True), [source])
    assert answer.status == "completed", answer.caveats
    assert _calls(events) == ["table_info", "filter_rows", "count_records"]
    assert budget.tool_calls == 3 and budget.model_calls == 0
    derived, rows = ctx.registry.load(answer.result_artifact_ids[0])
    assert {row["studyDbId"] for row in rows} == {"S1", "S2"}
    assert all(c.source_artifact_hashes == [derived.sha256] for c in answer.claims)
    assert any(source.sha256 in note for note in derived.quality_notes)


@pytest.mark.parametrize("fault", ["ambiguous", "missing_binding", "wrong_hash", "incomplete"])
def test_unproven_catalog_stops_before_any_analyst_mcp_call(ctx, fault):
    first = _save(ctx, complete=fault != "incomplete")
    artifacts = [first]
    sources = {"step_export": [first.artifact_id]}
    if fault == "ambiguous":
        other = _save(ctx, [{"locationDbId": "different", "locationName": "other"}])
        artifacts.append(other)
        sources["step_export"].append(other.artifact_id)
    elif fault == "missing_binding":
        sources = {}
    elif fault == "wrong_hash":
        artifacts = [first.model_copy(update={"sha256": "f" * 64})]
    answer, events, budget = _mcp(ctx, _plan(), artifacts, export_sources=sources)
    assert answer.status != "completed" and not answer.claims
    assert _calls(events) == [] and budget.tool_calls == 0 and budget.model_calls == 0


def test_async_tool_refusal_does_not_switch_to_local_dispatch(ctx, monkeypatch):
    source = _save(ctx)
    _no_direct_fallback(monkeypatch)
    calls = []

    class RefusedTools:
        async def call(self, name, args):
            calls.append(name)
            return ToolResult(ok=False, error={"code": "not_authorized", "message": "SYNTHETIC MCP refusal"})

    async def scenario():
        return await ca.run_catalog_plan_async(_plan(), _retrieval([source]), ctx.registry, tools=RefusedTools(),
                                               export_sources={"step_export": [source.artifact_id]})
    answer = anyio.run(scenario)
    assert answer.status == "needs_clarification" and not answer.claims
    assert calls == ["table_info"] and "SYNTHETIC MCP refusal" in " ".join(answer.caveats)


def test_async_incomplete_tool_result_cannot_become_a_count(ctx):
    source = _save(ctx)
    calls = []

    class IncompleteTools:
        async def call(self, name, args):
            calls.append(name)
            return ToolResult(ok=True, complete=False)

    async def scenario():
        return await ca.run_catalog_plan_async(_plan(), _retrieval([source]), ctx.registry, tools=IncompleteTools(),
                                               export_sources={"step_export": [source.artifact_id]})
    answer = anyio.run(scenario)
    assert answer.status != "completed" and not answer.claims and calls == ["table_info"]


@pytest.mark.parametrize("failure", [TimeoutError, ValueError, RuntimeError, asyncio.CancelledError])
def test_async_driver_preserves_failure_and_cancellation_without_fallback(ctx, monkeypatch, failure):
    source = _save(ctx)
    _no_direct_fallback(monkeypatch)
    budget = Budget()
    calls = []

    class BrokenTools:
        async def call(self, name, args):
            calls.append(name)
            raise failure("SYNTHETIC failure")

    async def scenario():
        return await ca.run_catalog_plan_async(_plan(), _retrieval([source]), ctx.registry, tools=BrokenTools(),
                                               export_sources={"step_export": [source.artifact_id]}, budget=budget)
    if failure in (RuntimeError, asyncio.CancelledError):
        with pytest.raises(failure):
            asyncio.run(scenario())
    else:
        answer = asyncio.run(scenario())
        assert answer.status == ("limit_reached" if failure is TimeoutError else "needs_clarification")
        assert not answer.claims
    assert calls == ["table_info"] and budget.tool_calls == 1 and budget.model_calls == 0
