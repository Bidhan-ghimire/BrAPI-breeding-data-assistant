"""Shared local service: real MCP round trips, synthetic data and fake HTTP only."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timedelta, timezone

import anyio
import pytest
from mcp import ClientSession

import agents.analyst as analyst
import analyst_tools as at
import brapi_mcp_server as brapi
from synthetic_support import synthetic_approval, synthetic_inputs, context
from brapi_client import Settings, TransportResponse
from breeding_mcp_server import BreedingMcpService
from contracts import FetchApproval
from mcp_bridge import McpTools

BASE = "https://synthetic.invalid/brapi/v2"




def catalog(ctx):
    return ctx.registry.save_table([
        {"locationDbId": "L1", "locationName": "SYNTHETIC one"},
        {"locationDbId": "L2", "locationName": "SYNTHETIC two"},
    ], kind="locations", columns={"locationDbId": "text", "locationName": "text"}, complete=True).artifact_id


def memory(ctx, service):
    return McpTools.memory(ctx, server_factory=lambda _: service.server)


def test_same_mcp_session_changes_discovery_and_enforces_each_stage(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    service = BreedingMcpService(ctx)
    methods = []
    original = ClientSession.send_request

    async def spy(self, request, *args, **kwargs):
        methods.append(request.root.method)
        return await original(self, request, *args, **kwargs)

    monkeypatch.setattr(ClientSession, "send_request", spy)

    async def scenario():
        async with memory(ctx, service) as raw:
            assert await raw.schemas() == brapi.tool_schemas()
            fetched = await raw.call("export_metadata", {"entity": "studies"})
            assert fetched.ok and fetched.artifact_ids
            handle = fetched.artifact_ids[0]
            denied = await raw.call("table_info", {"artifact_id": handle})
            assert not denied.ok and denied.error.code == "not_authorized"
            await service.suspend()
            assert service.phase == "suspended" and await raw.schemas() == []
            assert not (await raw.call("get_study", {"study_db_id": "S1"})).ok
            assert not (await raw.call("table_info", {"artifact_id": handle})).ok
            await service.enable_analysis([handle])
            assert await raw.schemas() == at.analyst_tool_schemas()
            # Stale client knowledge must never reopen retrieval.
            denied = await raw.call("get_study", {"study_db_id": "S1"})
            assert not denied.ok and denied.error.code == "not_authorized"
            denied = await raw.call("count_records", {"artifact_id": handle})
            assert not denied.ok and "table_info" in denied.error.message
            assert (await raw.call("table_info", {"artifact_id": handle})).ok
            counted = await raw.call("count_records", {"artifact_id": handle})
            assert counted.ok and counted.data["claims"][0]["value"] == 1
            assert not counted.data["claims"][0]["claim_id"].startswith("clm_ref_")
            await service.close()
            assert service.phase == "closed" and await raw.schemas() == []
            assert not (await raw.call("table_info", {"artifact_id": handle})).ok
        assert methods.count("initialize") == 1
        assert methods.count("tools/list") == 4 and methods.count("tools/call") == 9

    anyio.run(scenario)


@pytest.mark.parametrize("bad", ["missing", "expired", "other_server", "other_run", "family", "study", "variable"])
def test_raw_retrieval_cannot_bypass_approval_even_on_cached_data(tmp_path, bad):
    ctx = context(tmp_path)
    name, args = "get_study", {"study_db_id": "S1"}
    old = ctx.approval.model_dump()
    if bad == "missing":
        ctx.approval = None
    else:
        if bad == "expired":
            old.update(issued_at_utc=datetime.now(timezone.utc)-timedelta(hours=2),
                       expires_at_utc=datetime.now(timezone.utc)-timedelta(hours=1))
        elif bad == "other_server":
            old["base_url"] = "https://other-synthetic.invalid/brapi/v2"
        elif bad == "other_run":
            old["run_id"] = "run_elsewhere"
        elif bad == "family":
            old.update(endpoint_families=["locations"], observation_study_ids=[],
                       observation_variable_ids=[], max_observation_studies=0)
        elif bad in ("study", "variable"):
            name = "get_observations"
            args = {"study_db_id": "S2" if bad == "study" else "S1",
                    "variable_db_id": "V2" if bad == "variable" else "V1"}
        ctx.approval = FetchApproval(**old)
    service = BreedingMcpService(ctx)

    async def scenario():
        async with memory(ctx, service) as raw:
            denied = await raw.call(name, args)
            assert not denied.ok and denied.error.code == "not_authorized"
        assert not ctx.client.provenance() and not ctx.registry.artifact_ids()

    anyio.run(scenario)


@pytest.mark.parametrize("name,args", [
    ("get_study", {"study_db_id": "S1", "phase": "analysis"}),
    ("get_study", {"study_db_id": "S1", "catalog_only": True}),
    ("get_study", {"study_db_id": "S1", "approval": True}),
    ("get_study", {"study_db_id": 1}),
    ("enable_analysis", {"artifact_ids": []}),
    ("suspend", {}),
    ("read_file", {"path": "../../secret.env"}),
])
def test_model_arguments_cannot_change_service_authority(tmp_path, name, args):
    ctx = context(tmp_path)
    service = BreedingMcpService(ctx)

    async def scenario():
        async with memory(ctx, service) as raw:
            denied = await raw.call(name, args)
            assert not denied.ok and denied.error.code == "invalid_argument"
            assert service.phase == "retrieval"
        assert not ctx.client.provenance()

    anyio.run(scenario)


def test_analysis_is_one_way_and_only_registered_handles_are_granted(tmp_path):
    ctx = context(tmp_path)
    handle = catalog(ctx)
    outside = catalog(ctx)
    service = BreedingMcpService(ctx)

    async def scenario():
        with pytest.raises(RuntimeError, match="suspended"):
            await service.enable_analysis([handle])
        await service.suspend()
        await service.suspend()  # safe idempotent pause
        for invalid in (["../../secret.env"], ["art_unknown"], [handle, handle], [123], handle):
            with pytest.raises(ValueError):
                await service.enable_analysis(invalid)
            assert service.phase == "suspended"
        await service.enable_analysis([handle])
        with pytest.raises(RuntimeError):
            await service.enable_analysis([outside])
        with pytest.raises(RuntimeError):
            await service.suspend()
        async with memory(ctx, service) as raw:
            denied = await raw.call("table_info", {"artifact_id": outside})
            assert not denied.ok and denied.error.code == "not_authorized"
        await service.close()
        with pytest.raises(RuntimeError):
            await service.enable_analysis([handle])

    anyio.run(scenario)


def test_all_inputs_require_inspection_and_derivations_stay_in_run(tmp_path):
    ctx = context(tmp_path)
    first, second = catalog(ctx), catalog(ctx)
    service = BreedingMcpService(ctx)

    async def scenario():
        await service.suspend()
        await service.enable_analysis([first, second])
        async with memory(ctx, service) as raw:
            assert (await raw.call("table_info", {"artifact_id": first})).ok
            denied = await raw.call("count_records", {"artifact_id": first})
            assert not denied.ok and second in denied.error.message
            assert (await raw.call("table_info", {"artifact_id": second})).ok
            filtered = await raw.call("filter_rows", {"artifact_id": first, "column": "locationDbId", "op": "eq", "value": "L1"})
            assert filtered.ok
            counted = await raw.call("count_records", {"artifact_id": filtered.data["artifact_id"]})
            assert counted.ok and counted.data["claims"][0]["value"] == 1

    anyio.run(scenario)


def test_canonical_numeric_claims_match_existing_fixed_calculator(tmp_path):
    ctx = context(tmp_path)
    inputs = synthetic_inputs(ctx, "Describe synthetic yield.")
    initial_requests = list(ctx.client.provenance())
    expected = at.dispatch_analyst(ctx.registry, "numeric_summary", {"artifact_id": inputs.artifact_ids[0]})
    assert expected.ok
    service = BreedingMcpService(ctx)

    async def scenario():
        await service.suspend()
        await service.enable_analysis(inputs.artifact_ids)
        async with memory(ctx, service) as raw:
            for handle in inputs.artifact_ids:
                assert (await raw.call("table_info", {"artifact_id": handle})).ok
            result = await raw.call("numeric_summary", {"artifact_id": inputs.artifact_ids[0]})
            assert result.ok and result.data["claims"] == expected.data["claims"]
            mean = next(c for c in result.data["claims"] if c["kind"] == "mean")
            assert mean["value"] == 18.0 and mean["denominator"]["n"] == 4
            assert not mean["claim_id"].startswith("clm_ref_")
        assert ctx.client.provenance() == initial_requests

    anyio.run(scenario)


def test_two_contexts_cannot_share_stages_artifacts_or_inspection(tmp_path):
    first = context(tmp_path / "one", "run_one")
    second = context(tmp_path / "two", "run_two")
    a, b = catalog(first), catalog(second)
    one, two = BreedingMcpService(first), BreedingMcpService(second)

    async def scenario():
        await one.suspend()
        await one.enable_analysis([a])
        async with memory(first, one) as raw_one, memory(second, two) as raw_two:
            assert (await raw_one.call("table_info", {"artifact_id": a})).ok
            assert (await raw_one.call("count_records", {"artifact_id": a})).ok
            assert two.phase == "retrieval"
            assert not (await raw_two.call("table_info", {"artifact_id": b})).ok
            await two.suspend()
            with pytest.raises(ValueError):
                await two.enable_analysis([a])
            await two.enable_analysis([b])
            denied = await raw_two.call("count_records", {"artifact_id": b})
            assert not denied.ok and "table_info" in denied.error.message
            assert not (await raw_two.call("table_info", {"artifact_id": a})).ok

    anyio.run(scenario)


def fake_live_context(tmp_path, *, attempts=2):
    ctx = context(tmp_path)
    ctx.client.settings = Settings(base_url=BASE, mode="live", cache_dir=tmp_path / "empty_cache")
    ctx.approval = ctx.approval.model_copy(update={"max_http_attempts": attempts})
    calls = []

    def transport(request):
        calls.append(request)
        body = {"metadata": {}, "result": {"studyDbId": request.url.rsplit("/", 1)[-1], "studyName": "SYNTHETIC"}}
        return TransportResponse(status=200, headers={}, body=json.dumps(body).encode())

    ctx.client._transport = transport
    return ctx, calls, transport


def test_shared_service_does_not_double_count_http_budget(tmp_path):
    ctx, calls, _ = fake_live_context(tmp_path, attempts=1)
    service = BreedingMcpService(ctx)

    async def scenario():
        async with memory(ctx, service) as raw:
            assert (await raw.call("get_study", {"study_db_id": "UNSEEDED1"})).ok
            denied = await raw.call("get_study", {"study_db_id": "UNSEEDED2"})
            assert not denied.ok and denied.error.code == "not_authorized"
        assert len(calls) == len(ctx.client.provenance()) == 1

    anyio.run(scenario)


def test_suspend_waits_for_active_call_and_refuses_later_retrieval(tmp_path):
    ctx, calls, original = fake_live_context(tmp_path)
    started, finished = threading.Event(), threading.Event()

    def slow(request):
        started.set()
        time.sleep(0.12)
        result = original(request)
        finished.set()
        return result

    ctx.client._transport = slow
    service = BreedingMcpService(ctx)

    async def scenario():
        async with memory(ctx, service) as raw:
            fetch = asyncio.create_task(raw.call("get_study", {"study_db_id": "UNSEEDED1"}))
            with anyio.fail_after(1):
                while not started.is_set():
                    await asyncio.sleep(0.001)
            pause = asyncio.create_task(service.suspend())
            await asyncio.sleep(0.01)
            assert not pause.done()
            assert (await fetch).ok
            await pause
            assert finished.is_set() and service.phase == "suspended"
            denied = await raw.call("get_study", {"study_db_id": "UNSEEDED2"})
            assert not denied.ok and denied.error.code == "not_authorized"
        assert len(calls) == 1

    asyncio.run(scenario())


def test_catalog_only_allows_cached_planning_metadata_without_granting_retrieval(tmp_path):
    ctx = context(tmp_path)
    ctx.approval = None
    service = BreedingMcpService(ctx, catalog_only=True)

    async def scenario():
        async with memory(ctx, service) as raw:
            assert {s["name"] for s in await raw.schemas()} == {"export_metadata", "request_log"}
            for entity in ("studies", "variables"):
                assert (await raw.call("export_metadata", {"entity": entity})).ok
            assert not (await raw.call("export_metadata", {"entity": "locations"})).ok
            assert not (await raw.call("get_study", {"study_db_id": "S1"})).ok
            assert not (await raw.call("numeric_summary", {"artifact_id": "art_unknown"})).ok
            with pytest.raises(RuntimeError):
                await service.suspend()
            with pytest.raises(RuntimeError):
                await service.enable_analysis(ctx.registry.artifact_ids())
        assert all(record.origin == "cache" for record in ctx.client.provenance())

    anyio.run(scenario)


def test_live_catalog_without_approval_never_reaches_transport(tmp_path):
    ctx, calls, _ = fake_live_context(tmp_path)
    ctx.approval = None
    service = BreedingMcpService(ctx, catalog_only=True)

    async def scenario():
        async with memory(ctx, service) as raw:
            denied = await raw.call("export_metadata", {"entity": "studies"})
            assert not denied.ok and denied.error.code == "not_authorized"
        assert not calls and not ctx.client.provenance()

    anyio.run(scenario)
