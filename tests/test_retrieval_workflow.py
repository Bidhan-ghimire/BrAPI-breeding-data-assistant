"""Real Job/MCP workflow with synthetic HTTP, models, and human replies only.

The tests exercise retained retrieval evidence, bounded clarification, unchanged
approval scope, and a mandatory data-review gate before any analysis. No browser
is launched and the project's network guard blocks external connections.
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
from llm import FakeModelClient, reply_text, reply_tools
from part2_support import part2_runtime, synthetic_models


@pytest.fixture(autouse=True)
def _part2_flow(monkeypatch):
    """Ported from part2: run the release flow the way part2's mock mode did (see part2_support.part2_runtime)."""
    part2_runtime(monkeypatch, review=None)


BASE = "https://workflow-test.invalid/brapi/v2"
QUESTION = "For synthetic study S1, give mean yield and clone means with n."
FOLLOWUP = "Keep the original S1/V1 plot-level values?"
ANSWER = "Yes, keep the approved S1/V1 values."


def tool(name, **args):
    return reply_tools([(name, json.dumps(args))], model="mock-model")


def clarification(question=FOLLOWUP):
    return reply_text(json.dumps({"status": "needs_clarification", "question": question}), model="mock-model")


def completed(studies=None):
    return reply_text(json.dumps({"status": "completed", "payload": {
        "resolved": [], "studies": ["S1"] if studies is None else studies, "notes": []}}), model="mock-model")


def scripted(*middle):
    return FakeModelClient([
        tool("get_observations", study_db_id="S1", variable_db_id="V1"),
        *middle,
        tool("get_observation_units", study_db_id="S1"), completed(),
    ], reported_model="mock-model")


class SyntheticHTTP:
    replies = {"/studies": "synthetic_studies.json", "/variables": "synthetic_variables.json",
               "/observations": "synthetic_observations_S1_V1.json",
               "/observationunits": "synthetic_observationunits_S1.json",
               "/locations": "synthetic_locations.json"}

    def __init__(self):
        self.calls = []

    def __call__(self, request):
        endpoint = request.url.removeprefix(BASE)
        self.calls.append((endpoint, dict(request.params)))
        assert endpoint in self.replies, f"Unexpected synthetic HTTP endpoint: {endpoint}"
        _, body = load_fixture(self.replies[endpoint])
        if endpoint == "/locations":
            payload = json.loads(body)
            first = payload["result"]["data"][0]
            payload["result"]["data"] = [{**first, "locationDbId": f"L{n}",
                                              "locationName": f"SYNTHETIC location {n}"} for n in (1, 2, 3)]
            payload["metadata"]["pagination"].update(totalCount=3, totalPages=1)
            body = json.dumps(payload).encode("utf-8")
        return TransportResponse(200, {"Content-Type": "application/json"}, body)


@pytest.fixture

def chat(tmp_path, monkeypatch):
    """Only the external services are replaced; app Job and memory MCP are real."""
    notes = tmp_path / "reviewed.txt"
    notes.write_text("Reviewed by Bidhan on: 2026-09-28\n", encoding="utf-8")
    settings = Settings(base_url=BASE, mode="live", cache_dir=tmp_path / "cache")
    server = SyntheticHTTP()
    cache = CacheStore(settings.cache_dir)
    for name in ("synthetic_studies.json", "synthetic_variables.json"):
        seed_fixture(cache, BASE, name)
    monkeypatch.setattr(ctl, "load_settings", lambda: settings)
    monkeypatch.setattr(bc, "RequestsTransport", lambda: server)

    def no_direct(*args, **kwargs):
        raise AssertionError("Memory MCP workflow must not use Controller direct dispatch")

    monkeypatch.setattr(ctl, "dispatch", no_direct, raising=False)
    rig = SimpleNamespace(server=server, jobs=[], analysis_entries=[], retriever=scripted(),
                          clarification_rounds=3, review=True)
    original_start, original_next = app.Job.start, app.Job.next_event
    original_analyze = ctl.Controller.analyze_phase

    def start(job):
        rig.jobs.append(job)
        job.controller.answers_simulated = True  # Scripted fixture replies, never real human approval.
        return original_start(job)

    async def analyze(controller):
        rig.analysis_entries.append(controller.run_id)
        return await original_analyze(controller)

    monkeypatch.setattr(app.Job, "start", start)
    monkeypatch.setattr(app.Job, "next_event", lambda job, timeout=15: original_next(job, timeout=15))
    monkeypatch.setattr(ctl.Controller, "analyze_phase", analyze)

    def config(question):
        return ctl.RunConfig(question=question, offline=False, mock_model=False,
                             direct=False, mcp_transport="memory", auto=False,
                             out_dir=tmp_path / "out", cache_dir=tmp_path / "cache", access_notes=notes,
                             bootstrap_catalogs=True, read_timeout=90, max_elapsed_seconds=600,
                             max_clarification_rounds=rig.clarification_rounds,
                             review_retrieved_data=rig.review)

    rig.config = config
    rig.kwargs = {"make_config": config,
                  "make_models": lambda: replace(synthetic_models(), retriever=rig.retriever),
                  "preflight": lambda question: config(question).validate()}
    yield rig
    for job in rig.jobs:
        if job.thread.is_alive():
            job.close()
        job.thread.join(timeout=15)
        assert not job.thread.is_alive(), "Synthetic workflow job did not stop"


def turn(chat, message, history=None, state=None):
    events = list(app.respond(message, history or [], state or {}, **chat.kwargs))
    assert events and events[-1][2] == ""
    return events[-1][0], events[-1][1]


def start_approved(chat):
    history, state = turn(chat, QUESTION)
    job = state["job"]
    assert job.waiting_on.startswith("approve / edit / cancel"), history[-1]
    assert chat.server.calls == [] and chat.analysis_entries == []
    history, state = turn(chat, "approve", history, state)
    return history, state, job


def saved(job):
    job.thread.join(timeout=15)
    assert not job.thread.is_alive(), "Synthetic workflow did not finish writing its record"
    result = ctl.RunResult.model_validate_json((job.controller.run_dir / "answer.json").read_text(encoding="utf-8"))
    manifest = json.loads((job.controller.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tool_transport"] == "memory"
    return result, manifest


def assert_review_wait(chat, history, state, job):
    assert state["job"] is job and job.waiting_on == ctl.DATA_REVIEW_PROMPT, history[-1]
    assert job.controller.data_review["status"] == "pending"
    assert job.controller.analysis is None and chat.analysis_entries == []
    assert "continue" in history[-1]["content"].lower(), history[-1]
    assert app.reply_buttons(state)[1] is True  # Cancellation remains possible at this gate.


def retrieval_session(manifest):
    matches = [session for session in manifest["mcp_sessions"] if session["phase"] == "retrieval"]
    assert len(matches) == 1, manifest["mcp_sessions"]
    events = matches[0]["events"]
    assert sum(event["method"] == "initialize" for event in events) == 1
    return events


def test_approve_button_at_review_means_continue_and_does_not_reapprove_fetches(chat):
    history, state, job = start_approved(chat)
    assert_review_wait(chat, history, state, job)
    assert app.reply_buttons(state)[0] is True
    approved_id = job.controller.approval.approval_id
    fetched = list(chat.server.calls)
    history, state = turn(chat, "approve", history, state)
    assert "Reply **accept**" in history[-1]["content"]
    assert job.controller.approval.approval_id == approved_id and chat.server.calls == fetched
    turn(chat, "accept", history, state)
    result, manifest = saved(job)
    assert result.execution_status == "completed"
    assert manifest["data_review"]["status"] == "approved"


@pytest.mark.parametrize("reply, status", [("cancel", "canceled"), ("reject", "canceled"),
                                           ("yes", "blocked"), ("accept", "blocked"), ("go ahead", "blocked")])

def test_data_review_denial_or_unsupported_reply_never_starts_analysis(chat, reply, status):
    history, state, job = start_approved(chat)
    assert_review_wait(chat, history, state, job)
    fetched = list(chat.server.calls)
    history, state = turn(chat, reply, history, state)
    result, manifest = saved(job)
    assert result.execution_status == status, result.notes
    assert result.analysis is None and chat.analysis_entries == [] and "job" not in state
    assert result.retrieval.complete is True and result.retrieval.artifacts
    assert chat.server.calls == fetched and result.counts.live_attempts == 2
    assert manifest["data_review"]["status"] == status
    assert manifest["data_review"]["artifacts"]


def test_data_review_eof_stops_without_analysis_and_keeps_fetch_receipts(chat):
    history, state = turn(chat, QUESTION)
    job = state["job"]
    job.reply_timeout_s = 0.03  # Synthetic missing user reply only at the next prompt.
    history, state = turn(chat, "approve", history, state)
    result, manifest = saved(job)
    assert result.execution_status == "canceled", result.notes
    assert result.analysis is None and chat.analysis_entries == []
    assert result.counts.live_attempts == 2 and result.retrieval.artifacts
    assert manifest["data_review"]["status"] == "canceled"


def test_changed_artifact_after_preview_blocks_analysis_instead_of_accepting_stale_review(chat):
    history, state, job = start_approved(chat)
    assert_review_wait(chat, history, state, job)
    registry = job.controller.ctx.registry
    handle = job.controller.data_review["artifacts"][0]["artifact_id"]
    artifact = registry.manifest(handle)
    path = registry.out_root / artifact.relative_path
    with path.open("ab") as target:
        target.write(b"\nSYNTHETIC tampered row\n")
    turn(chat, "continue", history, state)
    result, manifest = saved(job)
    assert result.execution_status == "blocked", result.notes
    assert result.analysis is None and chat.analysis_entries == []
    assert manifest["data_review"]["status"] == "blocked"
    assert result.counts.live_attempts == 2
    assert any("hash" in note.lower() or "chang" in note.lower() for note in result.notes)

