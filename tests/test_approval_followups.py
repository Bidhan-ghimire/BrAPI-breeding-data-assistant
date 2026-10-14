"""Approval follow-ups request a bounded new plan, never permission to fetch."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

import app
import run as ctl
from llm import FakeModelClient, reply_text
from part2_support import SYNTHETIC_BASE, part2_config, part2_runtime, synthetic_models, synthetic_plan_draft


@pytest.fixture(autouse=True)
def _part2_flow(monkeypatch):
    """Ported from part2: run the release flow the way part2's mock mode did (see part2_support.part2_runtime)."""
    part2_runtime(monkeypatch, server=SYNTHETIC_BASE)



Q = "For synthetic study S1, give mean yield and clone means with n."


def config(tmp_path, **overrides):
    values = dict(question=Q, offline=False, mock_model=False, direct=False, auto=False,
                  fixture="synthetic", out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    values.update(overrides)
    return part2_config(**values)


def ready(variable="V1"):
    return reply_text(json.dumps({"status": "completed", "payload": synthetic_plan_draft("S1", variable)}), model="mock-model")


def planner_models(*replies):
    planner = FakeModelClient(list(replies), reported_model="mock-model")
    return planner, replace(synthetic_models(), coordinator=planner)


@pytest.mark.parametrize("answer", ["5817,", "details of this study 5822", "no, only study 5817", "approve study 5817"])
def test_scope_text_is_a_revision_not_permission(answer):
    assert ctl.is_approval_revision(answer)


@pytest.mark.parametrize("answer", ["no", "no thanks", "yes", "YES!", "done", "approve", "approve!", "edit", "cancel", "?", "maybe"])
def test_control_words_are_not_replanned(answer):
    assert not ctl.is_approval_revision(answer)


def test_refinement_rebuilds_scope_and_clears_stale_approval_before_planning(tmp_path):
    planner, models = planner_models(ready(), ready("V2"))
    controller = ctl.Controller(config(tmp_path), models=models)
    scopes = []
    planning_calls = 0
    original_plan_phase = controller.plan_phase

    async def plan_phase():
        nonlocal planning_calls
        if planning_calls:
            assert controller.approval is None and controller.manifest is None
            assert controller.ctx is None and controller.requests == []
        planning_calls += 1
        await original_plan_phase()

    def answer(prompt):
        assert prompt == "approve / edit / cancel: "
        assert controller.ctx is None and controller.requests == []
        scopes.append(list(controller.plan.scope.variable_ids))
        if len(scopes) == 1:
            # Exercise the invalidation invariant even if future callers arrive with an old approval.
            controller.approval = ctl.build_approval(controller.manifest, issued_by_human=True, now=controller.now())
            return "Use dry matter content V2 instead"
        assert controller.approval is None
        return "cancel"

    controller.plan_phase = plan_phase
    controller.answer_fn = answer
    result = asyncio.run(controller.run())
    assert scopes == [["V1"], ["V2"]] and planning_calls == 2
    assert len(planner.requests) == 2 and result.execution_status == "canceled"
    assert result.approval_id is None and result.retrieval is None and result.counts.tool_calls == 0
    assert "retrieve" not in controller.trace


def test_only_revised_manifest_is_approved(tmp_path):
    """Exercise the approval phases without running the V1-only synthetic retrieval script."""
    planner, models = planner_models(ready(), ready("V2"))
    controller = ctl.Controller(config(tmp_path), models=models)
    seen = []
    replies = iter(["Use dry matter content V2 instead", "approve"])

    def answer(prompt):
        assert prompt == "approve / edit / cancel: "
        assert controller.approval is None and controller.ctx is None
        seen.append(controller.manifest)
        return next(replies)

    async def approve_changed_scope():
        controller.metadata = await controller._load_metadata_mcp()
        await controller.plan_phase()
        controller.manifest = controller.resolve_manifest()
        assert not controller.approve_phase()
        assert controller.approval is None and controller.manifest is None
        revision = controller.pending_approval_revision
        assert revision == "Use dry matter content V2 instead"
        controller.pending_approval_revision = None
        controller.config = replace(controller.config, question=ctl.add_clarification(controller.config.question, revision))
        await controller.plan_phase()
        controller.manifest = controller.resolve_manifest()
        assert controller.approve_phase()

    controller.answer_fn = answer
    asyncio.run(approve_changed_scope())
    assert len(planner.requests) == 2 and len(seen) == 2
    assert seen[0].observation_variable_ids == ["V1"]
    assert seen[1].observation_variable_ids == ["V2"]
    assert seen[0].sha256() != seen[1].sha256()
    assert controller.manifest is seen[1]
    assert controller.approval is not None
    assert controller.approval.observation_variable_ids == ["V2"]
    assert controller.ctx is None and controller.requests == []
    assert controller.budget.tool_calls == 0


def test_revision_limit_never_enters_retrieval(tmp_path):
    planner, models = planner_models(ready(), ready())
    replies = iter(["S1,", "details of this study S1"])
    controller = ctl.Controller(config(tmp_path, max_clarification_rounds=1), models=models,
                                answer_fn=lambda prompt: next(replies))
    result = asyncio.run(controller.run())
    assert result.execution_status == "needs_clarification"
    assert controller.approval is None and controller.manifest is None
    assert result.retrieval is None and result.counts.tool_calls == 0
    assert len(planner.requests) == 2 and "retrieve" not in controller.trace
    assert any("approval revision limit reached" in note for note in result.notes)


@pytest.mark.parametrize("reply", ["no thanks", "yes", "done"])
def test_existing_nonapproving_replies_stay_blocked(tmp_path, reply):
    planner, models = planner_models(ready())
    controller = ctl.Controller(config(tmp_path), models=models, answer_fn=lambda prompt: reply)
    result = asyncio.run(controller.run())
    assert result.execution_status == "blocked" and result.approval_id is None
    assert result.retrieval is None and len(planner.requests) == 1
    assert controller.approval_revisions == 0 and result.counts.tool_calls == 0


def test_revised_question_can_clarify_without_fetching(tmp_path):
    question = "Which study and trait should be described?"
    planner, models = planner_models(ready(), reply_text(json.dumps({"status": "needs_clarification", "question": question}), model="mock-model"))
    replies = iter(["use the other study instead", ""])
    controller = ctl.Controller(config(tmp_path), models=models, answer_fn=lambda prompt: next(replies))
    result = asyncio.run(controller.run())
    assert result.execution_status == "canceled" and result.approval_id is None
    assert result.retrieval is None and controller.manifest is None
    assert "clarify" in controller.trace and "retrieve" not in controller.trace
    assert len(planner.requests) == 2


@pytest.mark.parametrize("study_ids, show_notice", [([], True), (["S1"], False)])
def test_new_question_hint_is_only_shown_without_resolved_study_scope(tmp_path, study_ids, show_notice):
    question = "Which study and trait does this summary describe?"
    controller = SimpleNamespace(config=config(tmp_path, question="What supports this summary?"),
                                 coordination=SimpleNamespace(questions=[question]), metadata=None,
                                 plan=SimpleNamespace(scope=SimpleNamespace(study_ids=study_ids)))
    job = SimpleNamespace(controller=controller)
    before = controller.config.question
    rendered, finished = app.render(job, ("prompt", ctl.CLARIFY_PROMPT))
    assert ("Each new question starts a separate analysis" in rendered) is show_notice
    assert "Your answer is added to your question" in rendered and question in rendered
    assert not finished and controller.config.question == before
    assert controller.coordination.questions == [question]


def test_approval_revision_does_not_increment_planner_answer_label(tmp_path):
    controller = SimpleNamespace(config=config(tmp_path, question=ctl.add_clarification(Q, "S1,")),
                                 coordination=SimpleNamespace(questions=["Which trait?"]), metadata=None,
                                 plan=None, approval_revisions=1)
    rendered, _ = app.render(SimpleNamespace(controller=controller), ("prompt", ctl.CLARIFY_PROMPT))
    assert "answer 1 of 1" in rendered
