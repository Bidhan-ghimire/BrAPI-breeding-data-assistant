"""Real local MCP round trips; every database/model response is synthetic."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import asyncio
import json
import threading
import time

import anyio
import pytest
from mcp import ClientSession

import brapi_mcp_server as srv
from brapi_client import BrapiClientError, Settings, TransportResponse
from contracts import FetchApproval
from mcp_bridge import BridgeClosed, BridgeError, BridgeTimeout, McpTools
from part2_support import part2_memory_default, synthetic_approval, synthetic_context


@pytest.fixture(autouse=True)
def _part2_memory_server(monkeypatch):
    """Ported from part2: McpTools.memory(ctx) serves the release's shared service (see part2_support)."""
    part2_memory_default(monkeypatch)


BASE = "https://synthetic.invalid/brapi/v2"


def approval(*, families=None, study_ids=None, variable_ids=None, **overrides):
    now = datetime.now(timezone.utc)
    values = dict(approval_id="appr_embedded", run_id="run_embedded", base_url=BASE,
                  endpoint_families=families or ["studies"],
                  observation_study_ids=study_ids or [], observation_variable_ids=variable_ids or [],
                  max_http_attempts=4, max_observation_studies=len(study_ids or []),
                  issued_at_utc=now-timedelta(minutes=1), expires_at_utc=now+timedelta(minutes=10))
    return FetchApproval(**(values | overrides))


def context(tmp_path, *, approved=True):
    ctx = srv.build_context(offline=False, settings=Settings(base_url=BASE, mode="live", read_timeout=90,
                            operation_deadline=180), cache_dir=tmp_path / "cache", out_dir=tmp_path / "out",
                            run_id="run_embedded", approval=approval() if approved else None)
    calls = []

    def transport(req):
        calls.append(req)
        return response({"studyDbId": req.url.rsplit("/", 1)[-1], "studyName": "SYNTHETIC"})

    ctx.client._transport = transport
    return ctx, calls


def response(result, *, pagination=None, status=200, headers=None):
    body = {"metadata": {"pagination": pagination} if pagination is not None else {}, "result": result}
    return TransportResponse(status=status, headers=headers or {}, body=json.dumps(body).encode())


def test_memory_exchanges_real_mcp_messages_and_matches_direct_schemas_and_results(tmp_path, monkeypatch):
    methods = []
    original = ClientSession.send_request

    async def recorded(self, request, *args, **kwargs):
        methods.append(request.root.method)
        return await original(self, request, *args, **kwargs)

    monkeypatch.setattr(ClientSession, "send_request", recorded)
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out")
    ctx.approval = synthetic_approval(run_id=ctx.client.run_id)   # the release serves retrieval tools only under an approval

    async def scenario():
        async with McpTools.direct(ctx) as direct:
            schemas = await direct.schemas()
            expected = await direct.call("get_study", {"study_db_id": "S1"})
        async with McpTools.memory(ctx) as bridge:
            assert await bridge.schemas() == schemas
            actual = await bridge.call("get_study", {"study_db_id": "S1"})
            assert actual.ok and actual.data["study"] == expected.data["study"]
            assert actual.data["provenance"]["response_sha256"] == expected.data["provenance"]["response_sha256"]
            assert bridge.protocol_events == [{"method": "initialize", "outcome": "ok"},
                {"method": "tools/list", "outcome": "ok"}, {"method": "tools/call", "outcome": "ok", "tool": "get_study"}]
        assert methods == ["initialize", "tools/list", "tools/call"]
        with pytest.raises(BridgeClosed):
            await bridge.call("server_info", {})
        assert ctx.client.remaining_time is None

    anyio.run(scenario)


@pytest.mark.parametrize("invalid", ["missing", "expired", "wrong_server", "wrong_family", "wrong_study", "wrong_trait", "smuggled"])
def test_real_mcp_refuses_invalid_authority_before_http(tmp_path, invalid):
    ctx, calls = context(tmp_path, approved=False)
    name, args = "get_study", {"study_db_id": "S1"}
    if invalid == "expired":
        now = datetime.now(timezone.utc)
        ctx.approval = approval(issued_at_utc=now-timedelta(hours=2), expires_at_utc=now-timedelta(hours=1))
    elif invalid == "wrong_server":
        ctx.approval = approval(base_url="https://other-synthetic.invalid/brapi/v2")
    elif invalid == "wrong_family":
        ctx.approval = approval(families=["locations"])
    elif invalid in ("wrong_study", "wrong_trait"):
        ctx.approval = approval(families=["observations"], study_ids=["S1"], variable_ids=["V1"])
        name = "get_observations"
        args = {"study_db_id": "S2" if invalid == "wrong_study" else "S1",
                "variable_db_id": "V2" if invalid == "wrong_trait" else "V1"}
    elif invalid == "smuggled":
        args["approval"] = {"allow_live": True}

    async def scenario():
        async with McpTools.memory(ctx) as bridge:
            denied = await bridge.call(name, args)
            assert not denied.ok
            # Collection tools report a failed collection; single-record/schema tools return an error.
            if denied.error:
                assert denied.error.code in ("not_authorized", "invalid_argument", "upstream_error")
            assert not calls and not ctx.client.provenance()

    anyio.run(scenario)


def test_real_mcp_preserves_attempt_budget_and_does_not_leak_approval(tmp_path):
    ctx, calls = context(tmp_path / "first")
    ctx.approval = approval(max_http_attempts=1)
    second, second_calls = context(tmp_path / "second", approved=False)

    async def scenario():
        async with McpTools.memory(ctx, call_timeout=180, remaining_time=lambda: 150) as bridge:
            assert (await bridge.call("get_study", {"study_db_id": "S1"})).ok
            denied = await bridge.call("get_study", {"study_db_id": "S2"})
            assert not denied.ok and denied.error.code == "not_authorized"
        assert len(calls) == len(ctx.client.provenance()) == 1
        assert calls[0].read_timeout == 90 and 0 < calls[0].deadline_remaining <= 150
        async with McpTools.memory(second) as bridge:
            denied = await bridge.call("get_study", {"study_db_id": "S1"})
            assert not denied.ok and not second_calls

    anyio.run(scenario)


def test_memory_initialization_timeout_closes_and_poison_session(tmp_path, monkeypatch):
    ctx, calls = context(tmp_path)
    server = srv.build_server(ctx)

    async def stalled_server(*args, **kwargs):
        await anyio.sleep(1)

    monkeypatch.setattr(server, "run", stalled_server)
    monkeypatch.setattr(srv, "build_server", lambda ctx: server)
    bridge = McpTools.memory(ctx, call_timeout=0.02)

    async def scenario():
        with pytest.raises(BridgeTimeout):
            async with bridge:
                pytest.fail("Initialization exceeded its allowance")
        with pytest.raises(BridgeClosed):
            await bridge.schemas()
        with pytest.raises(BridgeClosed):
            async with bridge:
                pass
        assert not calls and bridge._stack is None
        assert bridge.protocol_events == [{"method": "initialize", "outcome": "timeout"}]

    anyio.run(scenario)


def test_memory_listing_is_bounded_by_remaining_work_and_cannot_fallback(tmp_path, monkeypatch):
    ctx, calls = context(tmp_path)
    server = srv.build_server(ctx)

    @server.list_tools()
    async def slow_list():
        await anyio.sleep(1)

    monkeypatch.setattr(srv, "build_server", lambda ctx: server)

    async def scenario():
        async with McpTools.memory(ctx, remaining_time=lambda: 0.03) as bridge:
            with pytest.raises(BridgeTimeout):
                await bridge.schemas()
            with pytest.raises(BridgeClosed):
                await bridge.call("get_study", {"study_db_id": "S1"})
        assert not calls and bridge.protocol_events[-1] == {"method": "tools/list", "outcome": "timeout"}

    anyio.run(scenario)


def test_asyncio_task_cancellation_joins_worker_before_return(tmp_path):
    ctx, calls = context(tmp_path)
    started, finished = threading.Event(), threading.Event()

    def slow_response(req):
        calls.append(req)
        started.set()
        time.sleep(0.15)
        finished.set()
        return response({"studyDbId": "S1", "studyName": "SYNTHETIC"})

    ctx.client._transport = slow_response

    async def work():
        async with McpTools.memory(ctx) as bridge:
            await bridge.call("get_study", {"study_db_id": "S1"})

    async def scenario():
        task = asyncio.create_task(work())
        async with asyncio.timeout(1):
            while not started.is_set():
                await asyncio.sleep(0.002)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert len(calls) == len(ctx.client.provenance()) == 1

    asyncio.run(scenario())


def test_depleted_initialization_has_timeout_evidence(tmp_path):
    ctx, calls = context(tmp_path)
    bridge = McpTools.memory(ctx, remaining_time=lambda: 0)

    async def scenario():
        with pytest.raises(BridgeTimeout):
            async with bridge:
                pass
        assert not calls
        assert bridge.protocol_events == [{"method": "initialize", "outcome": "timeout"}]

    anyio.run(scenario)


@pytest.mark.parametrize("finish", ["timeout", "cancel"])
def test_interrupt_waits_for_worker_retains_provenance_and_stops_later_pages(tmp_path, finish):
    ctx, calls = context(tmp_path)
    started, finished = threading.Event(), threading.Event()

    def slow_page(req):
        calls.append(req)
        started.set()
        time.sleep(0.15)
        finished.set()
        return response({"data": [{"studyDbId": "S1", "studyName": "SYNTHETIC"}]},
                        pagination={"currentPage": 0, "pageSize": 1, "totalPages": 2, "totalCount": 2})

    ctx.client._transport = slow_page

    async def scenario():
        async with McpTools.memory(ctx) as bridge:
            if finish == "timeout":
                bridge.call_timeout = 0.03
                with pytest.raises(BridgeTimeout):
                    await bridge.call("search_studies", {})
            else:
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(bridge.call, "search_studies", {})
                    with anyio.fail_after(1):
                        while not started.is_set():
                            await anyio.sleep(0.002)
                    tasks.cancel_scope.cancel()
            with pytest.raises(BridgeClosed):
                await bridge.call("get_study", {"study_db_id": "S2"})
        assert started.is_set() and finished.is_set(), "Context exit must join the blocking worker"
        assert len(calls) == 1, "No page two or later request may run after stop"
        assert len(ctx.client.provenance()) == 1
        assert ctx.client.provenance()[0].http_status == 200
        assert bridge.protocol_events[-1]["outcome"] == ("timeout" if finish == "timeout" else "cancelled")
        with pytest.raises(BrapiClientError, match="cancelled"):
            ctx.client.study("S2", approval=ctx.approval)

    anyio.run(scenario)


def test_cancel_wakes_retry_wait_and_preserves_failed_attempt(tmp_path):
    ctx, calls = context(tmp_path)

    def throttled(req):
        calls.append(req)
        return response({}, status=429, headers={"Retry-After": "10"})

    ctx.client._transport = throttled

    async def scenario():
        async with McpTools.memory(ctx) as bridge:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(bridge.call, "get_study", {"study_db_id": "S1"})
                with anyio.fail_after(1):
                    while not ctx.client.provenance():
                        await anyio.sleep(0.002)
                tasks.cancel_scope.cancel()
        assert len(calls) == len(ctx.client.provenance()) == 1
        assert ctx.client.provenance()[0].http_status == 429

    started = time.monotonic()
    anyio.run(scenario)
    assert time.monotonic() - started < 2, "Cancellation must wake the 10-second retry sleep"


def test_remaining_work_caps_both_network_timeouts(tmp_path):
    ctx, calls = context(tmp_path)

    async def scenario():
        async with McpTools.memory(ctx, remaining_time=lambda: 3) as bridge:
            assert (await bridge.call("get_study", {"study_db_id": "S1"})).ok
        assert 0 < calls[0].connect_timeout <= 3
        assert 0 < calls[0].read_timeout <= 3
        assert 0 < calls[0].deadline_remaining <= 3

    anyio.run(scenario)


def test_depleted_work_budget_fails_closed_before_any_tool(tmp_path):
    ctx, calls = context(tmp_path)
    remaining = [5.0]

    async def scenario():
        async with McpTools.memory(ctx, remaining_time=lambda: remaining[0]) as bridge:
            remaining[0] = 0
            with pytest.raises(BridgeTimeout):
                await bridge.call("get_study", {"study_db_id": "S1"})
            with pytest.raises(BridgeClosed):
                await bridge.schemas()
        assert not calls

    anyio.run(scenario)
