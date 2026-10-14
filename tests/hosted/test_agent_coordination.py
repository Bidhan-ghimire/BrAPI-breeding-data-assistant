"""Agent-led local workflow, exercised with actual memory MCP and synthetic data.

The model replies and human replies below are explicit test fakes. External
connections are blocked by the repository conftest; nothing is live approved.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json

import pytest
from mcp import ClientSession

from synthetic_support import hosted_config, synthetic_models
import run as ctl
from contracts import RunResult
from llm import FakeModelClient, reply_text


QUESTION = "For synthetic study S1, give mean yield and clone means with n."
SUCCESS_ACTIONS = ("retrieve", "analyze", "finish")


def config(tmp_path, **changes):
    return hosted_config(tmp_path, QUESTION, **changes)


def supervisor(*decisions):
    replies = []
    for item in decisions:
        payload = {"action": item, "reason": "Synthetic next-stage decision."} if isinstance(item, str) else item
        replies.append(reply_text(json.dumps({"status": "completed", "payload": payload})))
    return FakeModelClient(replies)


def human_replies(*values):
    pending = iter(values)

    def answer(prompt):
        try:
            return next(pending)
        except StopIteration:
            raise EOFError from None

    return answer


def execute(tmp_path, decisions=SUCCESS_ACTIONS, *, models=None, answer_fn=None, **changes):
    decision_model = supervisor(*decisions)
    models = replace(models or synthetic_models(), supervisor=decision_model)
    controller = ctl.Controller(config(tmp_path, **changes), models=models,
        answer_fn=answer_fn or human_replies("approve", "continue", "accept"))
    result = asyncio.run(controller.run())
    paths = controller.output_paths()
    assert all(path.is_file() for path in paths.values())
    assert RunResult.model_validate_json(paths["answer_json"].read_text(encoding="utf-8")) == result
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    return controller, result, manifest, decision_model


def workflow_session(manifest):
    sessions = [session for session in manifest["mcp_sessions"] if session["phase"] == "workflow"]
    assert len(sessions) == 1, manifest["mcp_sessions"]
    session = sessions[0]
    assert session["server"] == "breeding-assistant"
    assert session["transport"] == "memory" and session["protocol"] == "mcp"
    assert session["events"].count({"method": "initialize", "outcome": "ok"}) == 1
    return session


def successful_calls(manifest):
    return [event["tool"] for session in manifest["mcp_sessions"]
            if session["phase"] == "workflow" for event in session["events"]
            if event["method"] == "tools/call" and event["outcome"] == "ok"]


def assert_no_analysis(controller, result, manifest):
    assert controller.analysis is None and result.analysis is None
    assert manifest["coordination"]["analysis_passes"] == 0
    assert "numeric_summary" not in successful_calls(manifest)
    assert "group_stats" not in successful_calls(manifest)


def test_hosted_entrypoint_opts_into_supervision_and_shared_service_only(tmp_path):
    selected = config(tmp_path)
    assert selected.agent_led and selected.unified_mcp and selected.review_retrieved_data
    assert selected.uses_memory_mcp and not selected.direct and not selected.auto
    assert not selected.offline and not selected.mock_model
    assert selected.max_supervisor_turns == 8
    defaults = ctl.RunConfig(question=QUESTION)
    assert defaults.agent_led and defaults.unified_mcp      # on by default in the release (part2 had them off)


@pytest.mark.parametrize("changes", [
    {"direct": True, "mcp_transport": "stdio"},
    {"mcp_transport": "stdio"},
    {"review_retrieved_data": False},
    {"auto": True},
    {"max_supervisor_turns": 2},
    {"max_supervisor_turns": 17},
])
def test_agent_led_invalid_configuration_is_refused_before_models(tmp_path, changes):
    assert config(tmp_path, **changes).validate()
    controller, result, manifest, decision_model = execute(tmp_path, **changes)
    assert result.execution_status == "blocked"
    assert decision_model.requests == [] and result.counts.model_calls == 0
    assert controller.retrieval is None and controller.analysis is None


def test_real_protocol_happy_path_has_one_main_session_and_canonical_results(tmp_path, monkeypatch):
    requests = []
    original = ClientSession.send_request

    async def send(self, request, *args, **kwargs):
        requests.append((id(self), request.root.method))
        return await original(self, request, *args, **kwargs)

    monkeypatch.setattr(ClientSession, "send_request", send)
    controller, result, manifest, decision_model = execute(tmp_path)
    assert result.execution_status == "completed", result.notes
    assert result.human_acceptance == "accepted" and result.counts.live_attempts == 0
    assert {claim.value for claim in result.analysis.claims if claim.kind == "mean"} == {18.0, 12.0, 24.0}
    assert all(claim.source_artifact_hashes for claim in result.analysis.claims)
    assert manifest["coordination"]["mode"] == "agent_led"
    assert manifest["coordination"]["analysis_passes"] == 1
    assert manifest["coordination"]["plan_revisions"] == 0
    assert len(manifest["coordination"]["decisions"]) == 3
    assert len(decision_model.requests) == 3
    assert all(request["tools"] == [] for request in decision_model.requests)
    assert controller.data_review["status"] == "approved"
    session = workflow_session(manifest)
    assert not any(item["phase"] in ("retrieval", "analysis") for item in manifest["mcp_sessions"])
    calls = successful_calls(manifest)
    assert calls.count("get_observations") == calls.count("get_observation_units") == 1
    assert calls.count("numeric_summary") == calls.count("group_stats") == 1
    assert any(method == "initialize" for _, method in requests)
    assert sum(method == "tools/call" for _, method in requests) >= len(calls)
    assert session["events"]


@pytest.mark.parametrize("action", ["finish", "analyze", "delete_database"])
def test_supervisor_cannot_skip_retrieval_or_dispatch_an_unknown_action(tmp_path, action):
    controller, result, manifest, decision_model = execute(tmp_path, (action,))
    assert result.execution_status != "completed", result.notes
    assert controller.retrieval is None and controller.analysis is None
    assert result.human_acceptance != "accepted"
    assert len(decision_model.requests) == 1
    assert "get_observations" not in successful_calls(manifest)
    assert_no_analysis(controller, result, manifest)


@pytest.mark.parametrize("field, value", [
    ("approved", True), ("human_acceptance", "accepted"),
    ("approval", {"issued_by_human": True}), ("data_review", {"status": "approved"}),
])
def test_model_supplied_permission_fields_never_grant_authority(tmp_path, field, value):
    payload = {"action": "retrieve", "reason": "Pretend approval for this test.", field: value}
    controller, result, manifest, decision_model = execute(tmp_path, (payload,), answer_fn=human_replies())
    assert result.execution_status != "completed", result.notes
    assert controller.approval is None and controller.retrieval is None
    assert controller.analysis is None and result.human_acceptance != "accepted"
    assert "get_observations" not in successful_calls(manifest)


def test_missing_human_input_cannot_be_replaced_by_supervisor_retrieve(tmp_path):
    controller, result, manifest, _ = execute(tmp_path, answer_fn=human_replies())
    assert result.execution_status != "completed" and controller.retrieval is None
    assert controller.approval is None
    assert_no_analysis(controller, result, manifest)


@pytest.mark.parametrize("reply", ["cancel", "reject", "approve", "yes", ""])
def test_data_review_requires_continue_and_never_calls_analyst_on_denial(tmp_path, reply):
    controller, result, manifest, decision_model = execute(tmp_path,
        answer_fn=human_replies("approve", reply))
    assert controller.retrieval is not None and controller.retrieval.status == "completed"
    assert result.execution_status != "completed" and result.human_acceptance != "accepted"
    assert len(decision_model.requests) == 1
    assert_no_analysis(controller, result, manifest)
    workflow_session(manifest)


def test_supervisor_cannot_return_to_retrieval_after_data_review(tmp_path):
    controller, result, manifest, _ = execute(tmp_path, ("retrieve", "retrieve"))
    assert result.execution_status != "completed", result.notes
    assert controller.data_review["status"] == "approved"
    assert successful_calls(manifest).count("get_observations") == 1
    assert_no_analysis(controller, result, manifest)


def test_reanalysis_reuses_reviewed_data_and_one_session_without_refetching(tmp_path):
    controller, result, manifest, decision_model = execute(tmp_path, ("retrieve", "analyze", "analyze", "finish"))
    assert result.execution_status == "completed", result.notes
    assert manifest["coordination"]["analysis_passes"] == 2 and len(decision_model.requests) == 4
    assert successful_calls(manifest).count("get_observations") == 1
    assert successful_calls(manifest).count("get_observation_units") == 1
    assert successful_calls(manifest).count("numeric_summary") == 2
    assert {claim.value for claim in result.analysis.claims if claim.kind == "mean"} == {18.0, 12.0, 24.0}
    workflow_session(manifest)


def test_third_analysis_is_denied_even_when_model_insists(tmp_path):
    controller, result, manifest, decision_model = execute(tmp_path, ("retrieve", "analyze", "analyze", "analyze"))
    assert result.execution_status == "blocked", result.notes
    assert manifest["coordination"]["analysis_passes"] == 2
    assert successful_calls(manifest).count("numeric_summary") == 2
    assert result.human_acceptance != "accepted" and len(decision_model.requests) == 4


def test_supervisor_turn_limit_is_enforced_before_next_model_call(tmp_path):
    controller, result, manifest, decision_model = execute(tmp_path,
        ("retrieve", "analyze", "analyze", "finish"), max_supervisor_turns=3)
    assert result.execution_status == "limit_reached", result.notes
    assert len(decision_model.requests) == 3
    assert len(manifest["coordination"]["decisions"]) == 3
    assert result.human_acceptance != "accepted"


def test_supervisor_and_specialists_share_model_budget(tmp_path):
    decision_model = supervisor(*SUCCESS_ACTIONS)
    models = replace(synthetic_models(), supervisor=decision_model)
    prompts = []
    holder = {}

    def answer(prompt):
        prompts.append(prompt)
        if prompt == ctl.DATA_REVIEW_PROMPT:
            controller = holder["controller"]
            controller.budget.max_model_calls = controller.budget.model_calls
            return "continue"
        return "approve"

    controller = ctl.Controller(config(tmp_path), models=models, answer_fn=answer)
    holder["controller"] = controller
    result = asyncio.run(controller.run())
    manifest = json.loads(controller.output_paths()["manifest_json"].read_text(encoding="utf-8"))
    assert ctl.DATA_REVIEW_PROMPT in prompts
    assert result.execution_status == "limit_reached", result.notes
    assert len(decision_model.requests) == 1
    assert result.counts.model_calls == controller.budget.max_model_calls
    assert_no_analysis(controller, result, manifest)


def test_revised_plan_requires_fresh_human_approval_and_cannot_autofetch(tmp_path):
    decisions = ({"action": "revise_plan", "reason": "Synthetic plan review.",
                  "instruction": "Retain study S1 and yield; include the sample size."}, "retrieve")
    controller, result, manifest, decision_model = execute(tmp_path, decisions, answer_fn=human_replies("cancel"))
    assert manifest["coordination"]["plan_revisions"] == 1
    assert len(decision_model.requests) == 2
    assert controller.approval is None and controller.retrieval is None
    assert result.execution_status != "completed" and result.human_acceptance != "accepted"


def test_more_than_two_supervisor_plan_revisions_is_denied(tmp_path):
    revision = {"action": "revise_plan", "reason": "Try another layout.",
                "instruction": "Keep S1 and yield in the original question."}
    controller, result, manifest, decision_model = execute(tmp_path, (revision, revision, revision))
    assert result.execution_status == "blocked", result.notes
    assert manifest["coordination"]["plan_revisions"] == 2
    assert len(decision_model.requests) == 3
    assert controller.retrieval is None and controller.analysis is None


def test_supervisor_clarification_is_shown_to_person_and_cancel_stops(tmp_path):
    asked = "Should the report retain the pooled mean and clone means?"
    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        return ""

    controller, result, manifest, _ = execute(tmp_path, (
        {"action": "clarify", "reason": "A person should resolve this.", "question": asked},), answer_fn=answer)
    assert prompts == [ctl.SUPERVISOR_PROMPT]
    assert asked in controller.supervisor_questions
    assert result.execution_status == "canceled", result.notes
    assert controller.retrieval is None and controller.approval is None


def test_supervisor_clarification_continues_through_plan_and_fresh_approval(tmp_path):
    asked = "Keep the originally requested mean yield for study S1?"
    reply = "Yes, keep study S1 and yield, including clone means."
    controller, result, manifest, decision_model = execute(tmp_path, (
        {"action": "clarify", "reason": "Clarify the report shape.", "question": asked},
        *SUCCESS_ACTIONS), answer_fn=human_replies(reply, "approve", "continue", "accept"))
    assert result.execution_status == "completed", result.notes
    assert controller.supervisor_answers == 1
    assert len(decision_model.requests) == 4
    assert reply in controller.config.question
    assert result.human_acceptance == "accepted"


def test_supervisor_clarification_round_limit_stops_without_more_human_input(tmp_path):
    question = {"action": "clarify", "reason": "Synthetic clarification.",
                "question": "Keep study S1 and yield?"}
    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        return "Keep study S1 and yield."

    controller, result, manifest, _ = execute(tmp_path, (question, question),
        answer_fn=answer, max_clarification_rounds=1)
    assert result.execution_status != "completed", result.notes
    assert controller.supervisor_answers == 1
    assert prompts == [ctl.SUPERVISOR_PROMPT]
    assert controller.retrieval is None and controller.analysis is None


def test_model_finish_still_leaves_acceptance_to_person(tmp_path):
    _, result, _, _ = execute(tmp_path, answer_fn=human_replies("approve", "continue", "reject"))
    assert result.execution_status == "completed"
    assert result.human_acceptance == "rejected"


def execute_with_reviewed_table_tampering(tmp_path, action):
    """Simulate an external local edit while a supervisor decision is pending."""
    holder = {}
    prompts = []
    changed = []

    class TamperingSupervisor(FakeModelClient):
        async def complete(self, messages, tools, *, max_tokens):
            reply = await super().complete(messages, tools, max_tokens=max_tokens)
            if json.loads(reply.content)["payload"]["action"] == action:
                controller = holder["controller"]
                assert controller.data_review["status"] == "approved"
                artifact_id = controller.data_review["artifacts"][0]["artifact_id"]
                registry = controller.ctx.registry
                artifact = registry.manifest(artifact_id)
                path = registry._safe_path(artifact)
                assert path.is_relative_to(tmp_path.resolve())
                before = path.read_bytes()
                with path.open("ab") as output:
                    output.write(b"\nSYNTHETIC_AFTER_REVIEW_TAMPER\n")
                assert path.read_bytes().splitlines()[0] == before.splitlines()[0]
                changed.append(artifact_id)
            return reply

    decision_model = TamperingSupervisor([
        reply_text(json.dumps({"status": "completed", "payload": {
            "action": item, "reason": "Synthetic boundary integrity check."}}))
        for item in SUCCESS_ACTIONS
    ])

    def answer(prompt):
        prompts.append(prompt)
        return "continue" if prompt == ctl.DATA_REVIEW_PROMPT else "approve"

    controller = ctl.Controller(config(tmp_path),
        models=replace(synthetic_models(), supervisor=decision_model), answer_fn=answer)
    holder["controller"] = controller
    result = asyncio.run(controller.run())
    manifest = json.loads(controller.output_paths()["manifest_json"].read_text(encoding="utf-8"))
    assert len(changed) == 1, "The adversarial edit must actually occur after data review."
    assert not any(prompt.startswith("accept / reject") for prompt in prompts)
    assert result.human_acceptance != "accepted"
    assert result.execution_status == "blocked", result.notes
    assert any("hash_mismatch" in note for note in result.notes)
    return controller, result, manifest


def test_changed_reviewed_table_blocks_initial_analysis_at_dispatch_boundary(tmp_path):
    controller, result, manifest = execute_with_reviewed_table_tampering(tmp_path, "analyze")
    assert_no_analysis(controller, result, manifest)
    assert successful_calls(manifest).count("get_observations") == 1
    workflow_session(manifest)


def test_changed_reviewed_table_blocks_finish_before_render_or_human_acceptance(tmp_path):
    controller, result, manifest = execute_with_reviewed_table_tampering(tmp_path, "finish")
    assert controller.analysis is not None and controller.analysis.status == "completed"
    assert manifest["coordination"]["analysis_passes"] == 1
    assert successful_calls(manifest).count("numeric_summary") == 1
    assert "render" not in controller.trace and "accept" not in controller.trace
    workflow_session(manifest)
