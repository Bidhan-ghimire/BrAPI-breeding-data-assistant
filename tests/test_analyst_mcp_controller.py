"""Local memory MCP controller behavior with synthetic cache/model/human replies."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json

import pytest
from mcp import ClientSession

import agents.analyst as an
import run as ctl
from agents.coordinator import CoordinatorResult
from contracts import Plan, PlanStep, ResolvedScope, RunResult
from part2_support import SYNTHETIC_BASE
from llm import FakeModelClient, reply_text, reply_tools
from mcp_bridge import BridgeError, McpTools
from part2_support import part2_config, part2_runtime, synthetic_models


@pytest.fixture(autouse=True)
def _part2_flow(monkeypatch, request):
    """Ported from part2: run the release flow the way part2's mock mode did (see part2_support.part2_runtime)."""
    part2_runtime(monkeypatch, server=SYNTHETIC_BASE,               # the data-review tests answer the review themselves
                  review=None if "data_review" in request.node.name else "continue")


QUESTION = "For synthetic study S1, give mean yield and clone means with n."


def config(tmp_path, **changes):
    return replace(part2_config(question=QUESTION, offline=False, mock_model=False, direct=False,
        mcp_transport="memory", fixture="synthetic", auto=False,
        cache_dir=tmp_path / "cache", out_dir=tmp_path / "out"), **changes)


def answers(*values):
    pending = iter(values)

    def answer(prompt):
        try:
            return next(pending)
        except StopIteration:
            raise EOFError from None

    return answer


def execute(tmp_path, *, models=None, answer_fn=None, **changes):
    controller = ctl.Controller(config(tmp_path, **changes), models=models or synthetic_models(), answer_fn=answer_fn)
    result = asyncio.run(controller.run())
    paths = controller.output_paths()
    assert all(path.is_file() for path in paths.values())
    assert RunResult.model_validate_json(paths["answer_json"].read_text(encoding="utf-8")) == result
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    return controller, result, manifest


def analysis_session(manifest):
    matches = [item for item in manifest["mcp_sessions"] if item["phase"] == "analysis"]
    assert len(matches) == 1, manifest["mcp_sessions"]
    session = matches[0]
    assert session["protocol"] == "mcp" and session["transport"] == "memory"
    assert {"method": "initialize", "outcome": "ok"} in session["events"]
    assert {"method": "tools/list", "outcome": "ok"} in session["events"]
    return session


def analysis_calls(manifest):
    return [event.get("tool") for event in analysis_session(manifest)["events"]
            if event["method"] == "tools/call" and event["outcome"] == "ok"]


@pytest.mark.parametrize("choice", ["cancel"])
def test_fetched_data_review_precedes_any_analyst_session(tmp_path, choice):
    prompted = []
    holder = {}

    def answer(prompt):
        controller = holder["controller"]
        prompted.append(prompt)
        if prompt == ctl.DATA_REVIEW_PROMPT:
            assert controller.analysis is None
            assert not any(item["phase"] == "analysis" for item in controller.mcp_sessions)
            return choice
        if prompt.startswith("approve / edit / cancel"):
            return "approve"
        return "accept"

    controller = ctl.Controller(config(tmp_path, auto=False, review_retrieved_data=True),
        models=synthetic_models(), answer_fn=answer)
    holder["controller"] = controller
    result = asyncio.run(controller.run())
    manifest = json.loads(controller.output_paths()["manifest_json"].read_text(encoding="utf-8"))
    assert ctl.DATA_REVIEW_PROMPT in prompted
    if choice == "continue":
        assert result.execution_status == "completed", result.notes
        assert analysis_calls(manifest)
    else:
        assert result.execution_status == "canceled" and result.analysis is None
        assert not any(item["phase"] == "analysis" for item in manifest["mcp_sessions"])


def test_exact_metadata_facts_do_not_claim_numeric_mcp_execution(tmp_path, monkeypatch):
    question = "Show metadata for study S1."
    plan = Plan(plan_id="plan_mcp_metadata", question=question, interpretation="Read exact recorded fields", status="ready",
        scope=ResolvedScope(base_url=SYNTHETIC_BASE, study_ids=["S1"], matching_study_count=1),
        steps=[PlanStep(step_id="step_lookup", agent="retriever", action="get_study", inputs={"study_db_id": "S1"})],
        created_at_utc=datetime.now(timezone.utc))

    async def draft(*args, **kwargs):
        return CoordinatorResult(status="completed", plan=plan)

    def no_analyst(prefix):
        raise AssertionError("Exact metadata facts do not require Analyst model tools")

    retriever = FakeModelClient([
        reply_tools([("get_study", json.dumps({"study_db_id": "S1"}))]),
        reply_text(json.dumps({"status": "completed", "payload": {"resolved": [], "studies": [], "notes": []}})),
    ])
    monkeypatch.setattr(ctl, "draft_plan", draft)
    _, result, manifest = execute(tmp_path, question=question,
        models=ctl.Models(retriever=retriever, analyst_factory=no_analyst, requested="mock-model"))
    assert result.execution_status == "completed", result.notes
    assert result.analysis.metadata_facts and result.analysis.claims == []
    assert not any(item["phase"] == "analysis" for item in manifest["mcp_sessions"])
    assert manifest["analysis_execution"] == {"route": "metadata_facts", "tool_transport": None}
