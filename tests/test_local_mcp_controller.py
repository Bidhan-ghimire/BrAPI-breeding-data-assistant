"""Actual chat/Controller/MCP integration with SYNTHETIC approvals, model and HTTP.

No browser/server is launched and no request leaves the process. These tests do
not grant or record approval for real data; all records live in pytest temp dirs.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

import app
import brapi_client as bc
import run as ctl
from agents.coordinator import CoordinatorResult
from brapi_client import Settings, TransportResponse
from cache_store import CacheStore
from contracts import Plan, PlanStep, ResolvedScope
from part2_support import load_fixture, seed_fixture
from llm import FakeModelClient, reply_text
from mcp_bridge import McpTools
from part2_support import part2_runtime, synthetic_models


@pytest.fixture(autouse=True)
def _part2_flow(monkeypatch):
    """Ported from part2: run the release flow the way part2's mock mode did (see part2_support.part2_runtime)."""
    part2_runtime(monkeypatch)


BASE = "https://local-mcp-test.invalid/brapi/v2"
QUESTION = "For synthetic study S1, give mean yield and clone means with n."


class SyntheticHTTP:
    replies = {"/studies": "synthetic_studies.json", "/variables": "synthetic_variables.json",
               "/observations": "synthetic_observations_S1_V1.json",
               "/observationunits": "synthetic_observationunits_S1.json"}

    def __init__(self):
        self.calls = []
        self.partial_endpoint = None

    def __call__(self, request):
        endpoint = request.url.removeprefix(BASE)
        self.calls.append((endpoint, dict(request.params), request.read_timeout))
        assert endpoint in self.replies, f"Unexpected synthetic HTTP endpoint: {endpoint}"
        _, body = load_fixture(self.replies[endpoint])
        if endpoint == self.partial_endpoint:
            payload = json.loads(body)
            payload["metadata"]["pagination"].update(totalPages=2, totalCount=2)
            body = json.dumps(payload).encode("utf-8")
        return TransportResponse(200, {"Content-Type": "application/json"}, body)


@pytest.fixture
def chat(tmp_path, monkeypatch):
    """Actual Job thread and MCP session; only external services are substituted."""
    server = SyntheticHTTP()
    cache = CacheStore(tmp_path / "cache")
    notes = tmp_path / "reviewed.txt"
    notes.write_text("Reviewed by Bidhan on: 2026-09-28\n", encoding="utf-8")
    settings = Settings(base_url=BASE, mode="live", cache_dir=tmp_path / "cache")
    monkeypatch.setattr(ctl, "load_settings", lambda: settings)
    monkeypatch.setattr(bc, "RequestsTransport", lambda: server)

    def no_direct(*args, **kwargs):
        raise AssertionError("Local MCP must not use Controller direct dispatch")

    monkeypatch.setattr(ctl, "dispatch", no_direct, raising=False)
    jobs = []
    original_start, original_next = app.Job.start, app.Job.next_event

    def track_start(job):
        jobs.append(job)
        return original_start(job)

    monkeypatch.setattr(app.Job, "start", track_start)
    monkeypatch.setattr(app.Job, "next_event", lambda job, timeout=15: original_next(job, timeout=15))

    def config(question):
        return ctl.RunConfig(question=question, offline=False, mock_model=False,
                             direct=False, mcp_transport="memory", auto=False,
                             out_dir=tmp_path / "out", cache_dir=tmp_path / "cache", access_notes=notes,
                             bootstrap_catalogs=True, read_timeout=90, max_elapsed_seconds=600,
                             max_clarification_rounds=app.CHAT_CLARIFICATION_ROUNDS)

    def warm():
        for name in ("synthetic_studies.json", "synthetic_variables.json"):
            seed_fixture(cache, BASE, name)

    kwargs = {"make_config": config, "make_models": synthetic_models,
              "preflight": lambda question: config(question).validate()}
    yield SimpleNamespace(server=server, config=config, warm=warm, kwargs=kwargs, jobs=jobs)
    for job in jobs:
        if job.thread.is_alive():
            job.close()
        job.thread.join(timeout=15)
        assert not job.thread.is_alive(), "Synthetic chat job did not stop"


def turn(chat, text, history=None, state=None):
    events = list(app.respond(text, history or [], state or {}, **chat.kwargs))
    assert events and events[-1][2] == ""
    return events[-1][0], events[-1][1]


def saved(job):
    job.thread.join(timeout=15)
    assert not job.thread.is_alive(), "Synthetic chat did not finish writing its record"
    result = ctl.RunResult.model_validate_json((job.controller.run_dir / "answer.json").read_text(encoding="utf-8"))
    manifest = json.loads((job.controller.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["transport"] == "mcp"
    assert manifest["tool_transport"] == "memory"
    return result, manifest


def protocol_session(manifest, phase, *, outcome="ok"):
    matches = [session for session in manifest["mcp_sessions"] if session["phase"] == phase]
    assert len(matches) == 1, manifest["mcp_sessions"]
    session = matches[0]
    assert session["protocol"] == "mcp" and session["transport"] == "memory"
    events = session["events"]
    assert {"method": "initialize", "outcome": "ok"} in events
    assert {"method": "tools/list", "outcome": "ok"} in events
    return [event.get("tool") for event in events if event["method"] == "tools/call" and event["outcome"] == outcome]


def test_warm_chat_waits_for_approval_without_http_and_cancel_does_not_fetch(chat):
    chat.warm()
    history, state = turn(chat, QUESTION)
    job = state["job"]
    assert job.waiting_on.startswith("approve / edit / cancel"), history[-1]
    assert chat.server.calls == [] and job.controller.approval is None
    history, state = turn(chat, "cancel", history, state)
    result, manifest = saved(job)
    assert result.execution_status == "canceled", result.notes
    assert result.approval_id is None and result.counts.live_attempts == 0 and chat.server.calls == []
    assert protocol_session(manifest, "catalog_cache") == ["export_metadata", "export_metadata"]
    assert [session["phase"] for session in manifest["mcp_sessions"]] == ["catalog_cache"]


def test_canceling_cold_catalog_setup_calls_neither_http_nor_models(chat):
    history, state = turn(chat, QUESTION)
    job = state["job"]
    assert job.waiting_on == ctl.CATALOG_PROMPT
    turn(chat, "cancel", history, state)
    result, manifest = saved(job)
    assert result.execution_status == "canceled", result.notes
    assert result.counts.model_calls == result.counts.live_attempts == 0
    assert chat.server.calls == [] and result.catalog_approval is None and result.approval_id is None
    assert manifest["catalog_setup"]["status"] == "canceled"
    assert protocol_session(manifest, "catalog_cache", outcome="rejected") == ["export_metadata", "export_metadata"]


def test_partial_catalog_stops_before_model_and_preserves_its_approved_http_evidence(chat, monkeypatch):
    chat.server.partial_endpoint = "/studies"
    monkeypatch.setattr(ctl, "CATALOG_MAX_PAGES", 1)
    history, state = turn(chat, QUESTION)
    job = state["job"]
    history, state = turn(chat, "approve", history, state)
    result, manifest = saved(job)
    assert result.execution_status == "incomplete", result.notes
    assert result.counts.model_calls == 0 and result.analysis is None
    assert result.catalog_approval is not None and result.approval_id is None
    assert job.controller.metadata is None
    assert [call[0] for call in chat.server.calls] == ["/studies"]
    assert result.counts.live_attempts == 1 and len(result.catalog_requests) == 1
    assert manifest["catalog_setup"]["status"] == "incomplete"
    assert protocol_session(manifest, "catalog_setup", outcome="rejected") == ["export_metadata"]


def test_two_chat_states_keep_approval_separate_even_with_shared_cache(chat):
    chat.warm()
    first_history, first_state = turn(chat, QUESTION)
    second_history, second_state = turn(chat, QUESTION)
    first, second = first_state["job"], second_state["job"]
    assert first.controller.run_id != second.controller.run_id
    assert first.controller.approval is second.controller.approval is None
    assert chat.server.calls == []
    first_history, first_state = turn(chat, "approve", first_history, first_state)
    assert "Reply **accept**" in first_history[-1]["content"], first.controller.notes
    assert second.controller.approval is None and second.waiting_on.startswith("approve / edit / cancel")
    assert [call[0] for call in chat.server.calls] == ["/observations", "/observationunits"]
    turn(chat, "cancel", second_history, second_state)
    second_result, second_manifest = saved(second)
    assert second_result.execution_status == "canceled", second_result.notes
    assert second_result.approval_id is None and second_result.counts.live_attempts == 0
    assert [session["phase"] for session in second_manifest["mcp_sessions"]] == ["catalog_cache"]
    turn(chat, "accept", first_history, first_state)
    first_result, _ = saved(first)
    assert first_result.execution_status == "completed", first_result.notes
    assert first_result.approval_id is not None
    assert len(chat.server.calls) == 2


@pytest.mark.parametrize("direct, transport, problem", [
    (False, "memory", None),
])
def test_runconfig_live_transport_validation_matrix(chat, direct, transport, problem):
    config = replace(chat.config(QUESTION), direct=direct, mcp_transport=transport)
    problems = config.validate()
    if problem is None:
        assert problems == []
    else:
        assert any(problem in value for value in problems), problems
