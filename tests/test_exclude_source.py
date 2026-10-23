"""
tests/test_exclude_source.py - the change-set "exclude_source" (coach 2026-10-23 04:49 UTC, NEXT 2).

F4: anova_trial leaves out only entries the person named. A runtime gate, not a prompt line (Bidhan's rule: a prompt is not a
    boundary): each exclude name must be written in the person's own words (case-insensitive, whole name), or the call is
    refused before the tool runs and the analyst asks, as for an unknown name. threshold_count's own recipe line stays.
F5: cells="single_only" or "mean" passes without asking only when the person's own words state that rule (C28.1-C28.6 say "on
    the progenies that have exactly one yield value in every replicate"); otherwise it is refused and the app keeps asking.
C33.2: a t-test asked as "A vs B" or "A compared with B" takes the differences A - B (Luna 2026-10-23: "(irrigated vs
    drought)", the plan made the irrigated study the base and t came out -4.1145 against the gold's 4.11).

The person's words are person_words(question): the question and the person's answers, without the labels code writes. The real
wordings come from the benchmark's own rows; the tables are SYNTHETIC and the models scripted. No network, no model.
"""
from __future__ import annotations

import json
from pathlib import Path

import anyio

import agents.analyst as an
import agents.coordinator as co
from contracts import ToolResult
from llm import Budget, FakeModelClient, reply_text, reply_tools
from part2_support import synthetic_context

ROWS = {r["sub_id"]: r["question"] for r in json.loads(
    (Path(__file__).resolve().parents[1] / "eval" / "benchmark" / "spb_subquestions.json").read_text(encoding="utf-8"))["subquestions"]}
C28_3 = ROWS["C28.3"]                                    # "... (all entries except Beauregard, Tanzania and CIP199062.1) ..."
C28_RULE = " on the progenies that have exactly one yield value in every replicate"


class _Recorder:
    """An inner tool interface that records every call it gets: a refused call never reaches it."""

    def __init__(self):
        self.calls = []

    async def schemas(self):
        return []

    async def call(self, name, args):
        self.calls.append((name, dict(args)))
        return ToolResult(ok=True, data={}, artifact_ids=[])


def guarded(question: str | None) -> an.GuardedAnalystTools:
    tools = an.GuardedAnalystTools(_Recorder(), ["art_1"], question=question)
    tools.inspected.add("art_1")
    return tools


def call(tools: an.GuardedAnalystTools, name: str = "anova_trial", **args) -> ToolResult:
    return anyio.run(tools.call, name, {"artifact_id": "art_1", **args})


# --- F4: the exclude names come from the person's words ------------------------------------------------------------------------

def test_names_the_person_wrote_reach_the_tool_whatever_their_case():
    tools = guarded(C28_3)
    r = call(tools, exclude=["beauregard", "TANZANIA", "CIP199062.1"])
    assert r.ok and tools.inner.calls == [("anova_trial", {"artifact_id": "art_1", "exclude": ["beauregard", "TANZANIA", "CIP199062.1"]})]


def test_a_name_the_person_never_wrote_is_refused_before_the_tool_runs():
    """A "without the checks" question names none: check names the app found itself (plot_design's entryType, a recipe line)
    are refused, as an unknown name is, and the analyst asks. Every name is checked, not only the first."""
    for question, exclude in ((C28_3, ["Beauregard", "Resisto"]), ("The ANOVA of yield in PEP2016ICA-ST03 without the checks", ["Beauregard"])):
        tools = guarded(question)
        r = call(tools, exclude=exclude)
        assert not r.ok and r.error.code == "invalid_argument" and tools.inner.calls == [], (question, exclude)
        assert f"exclude {exclude[-1]!r} is not written in the person's words" in r.error.message and "needs_clarification" in r.error.message
        assert tools.records[-1].refused_here


def test_part_of_a_longer_name_is_not_the_name():
    """Whole names only: MUSG15033 is not written in MUSG15033-8, nor PH17.001 in PH17.0012, nor 8 in MUSG15033-8; the full stop
    that ends a sentence is not part of a name."""
    question = "The ANOVA of yield leaving out MUSG15033-8, PH17.0012 and CIP199062.1."
    for name in ("MUSG15033", "PH17.001", "CIP199062", "USG15033-8", "8", "0012"):
        assert not call(guarded(question), exclude=[name]).ok, name
    for name in ("MUSG15033-8", "ph17.0012", "CIP199062.1", "PH17.0012 "):
        assert call(guarded(question), exclude=[name]).ok, name


def test_a_blank_name_and_the_labels_code_writes_are_not_the_persons_words():
    """The person's answer counts; the label code writes before it ("Clarification 1 from the person: ") does not."""
    question = co.add_clarification("The ANOVA of yield in PEP2016ICA-ST03.", "leave out Beauregard")
    assert call(guarded(question), exclude=["Beauregard"]).ok
    for name in ("person", "Clarification", "Clarification 1", " ", ""):
        assert not call(guarded(question), exclude=[name]).ok, name
    for name in (" ", ""):       # deliberate bug X9 survived the line above: its sentence has no ", " where an empty name matches
        assert not call(guarded(C28_3), exclude=[name]).ok, repr(name)


def test_only_anova_trial_is_gated_and_only_where_the_question_is_known():
    """The coach's scope: threshold_count's own line ("exclude names the study's other checks") stays, so its exclude is not
    checked here; the MCP server's own guard has no question (None) and checks neither."""
    assert call(guarded("How many entries are above Beauregard's mean?"), "threshold_count", exclude=["Covington"]).ok
    assert call(guarded(None), exclude=["Resisto"], cells="mean").ok


def test_the_analyst_loop_hands_the_guard_the_persons_question(tmp_path):
    """run_analyst gives the guard the question: a scripted analyst that leaves out an entry the question never names is refused
    by the guard (the tool never runs); a name the question writes reaches the tool, which refuses it here only because the
    synthetic S1 has no such entry. The analysis ends with a question to the person."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_exclude_source")
    inputs = an.synthetic_inputs(ctx, "SYNTHETIC: the analysis of variance of V1 in S1 without Beauregard")
    observations = next(a for a in inputs.artifact_ids if ctx.registry.manifest(a).kind == "observations")
    ask = json.dumps({"status": "needs_clarification", "question": "Which entries should be left out?"})
    model = FakeModelClient([reply_tools([("table_info", json.dumps({"artifact_id": a})) for a in inputs.artifact_ids]),
                             reply_tools([("anova_trial", json.dumps({"artifact_id": observations, "exclude": ["Resisto"]}))]),
                             reply_tools([("anova_trial", json.dumps({"artifact_id": observations, "exclude": ["Beauregard"]}))]),
                             reply_text(ask)])

    async def go():
        return await an.run_analyst(inputs, model=model, ctx=ctx, budget=Budget(), run_id="run_exclude_source",
                                    log_dir=ctx.registry.out_root)

    report, agent, _narrative = anyio.run(go)
    refused, reached = [json.loads(m.content) for m in agent.messages if m.role == "tool" and m.name == "anova_trial"]
    assert "exclude 'Resisto' is not written in the person's words" in refused["error"]["message"]
    assert "exclude 'Beauregard' is not an entry ID or name" in reached["error"]["message"]
    assert report.status == "needs_clarification" and model._replies == []


# --- F5: the cells rule only when the person's words state it ------------------------------------------------------------------

def test_the_c28_wording_states_single_only_and_only_single_only():
    """C28.1-C28.6: single_only passes without asking; mean is another rule and is refused."""
    assert C28_RULE in C28_3 and call(guarded(C28_3), cells="single_only").ok
    r = call(guarded(C28_3), cells="mean")
    assert not r.ok and r.error.code == "invalid_argument" and "cells='mean'" in r.error.message and "needs_clarification" in r.error.message


def test_a_rule_the_persons_words_do_not_state_is_refused_and_the_app_asks():
    """Decision (6) and the recorded C28 call hold: without the person's words the analyst never names a rule. Real wordings that
    state none: C28.3 without its clause, A38.7 ("the mean of their single-plot values") and "entry-mean heritability"."""
    plain = C28_3.replace(C28_RULE, "")
    assert plain != C28_3 and "single-plot values" in ROWS["A38.7"] and "entry-mean" in plain
    said = "Fit the ANOVA on the mean of their single-plot values, over offspring plots with a value."   # A38.7's phrase, as a statement
    for question in (plain, ROWS["A38.7"], ROWS["C31.1"], said):
        for rule in ("mean", "single_only"):
            tools = guarded(question)
            r = call(tools, cells=rule)
            assert not r.ok and tools.inner.calls == [] and f"cells={rule!r} is a rule the person's words do not state" in r.error.message
    assert call(guarded(plain), cells="refuse").ok and call(guarded(plain)).ok
    assert call(guarded(plain), cells=["mean"]).ok            # malformed: the tool's own argument check refuses it, the guard does not fail


def test_a_rule_inside_a_question_sentence_states_no_rule():
    """Review of this set (finding 3): a design question or a request for rep means is not the person choosing a cells rule.
    The statement must stand in a sentence that is not a question; C28's clause ends in "." and still counts."""
    for question, rule in (("Does every entry have exactly one plot in every rep in this trial?", "single_only"),
                           ("What is the mean of the yield values in each replicate?", "mean"),
                           ("Fit the ANOVA. Is there only one plot per entry in each rep? Then give H2.", "single_only")):
        r = call(guarded(question), cells=rule)
        assert not r.ok and f"cells={rule!r} is a rule the person's words do not state" in r.error.message, question
    asked_then_said = ("Does every entry have exactly one plot in every rep? If not, use only the entries with exactly one plot in "
                       "every rep.")
    assert call(guarded(asked_then_said), cells="single_only").ok and call(guarded(C28_3), cells="single_only").ok


def test_the_persons_answer_in_the_rules_own_words_counts():
    """Asked about a doubled cell, the person can answer with one of the two rules the analyst names; the code-written label
    before the answer is not part of it."""
    plain = C28_3.replace(C28_RULE, "")
    mean = co.add_clarification(plain, "use the mean of the plots in each rep")
    single = co.add_clarification(plain, "only the entries with exactly one plot in every rep")
    assert call(guarded(mean), cells="mean").ok and not call(guarded(mean), cells="single_only").ok
    assert call(guarded(single), cells="single_only").ok and not call(guarded(single), cells="mean").ok
    assert not call(guarded(co.add_clarification(plain, "mean")), cells="mean").ok     # one word states no rule: asked again


def test_the_prompts_say_where_the_names_and_the_rule_come_from():
    analyst, planner = " ".join(an.ANALYST_SYSTEM_PROMPT.split()), " ".join(co.COORDINATOR_SYSTEM_PROMPT.split())
    assert "only with entries the person named, as they wrote them" in analyst
    assert "only when the person's own words state the rule" in analyst
    assert '"the mean of the plots in each rep" or "only the entries with exactly one plot in every rep"' in analyst
    assert "for the check entries - a count, never anova_trial's exclude" in planner
    assert "to leave out, as the person wrote them" in planner
    assert "exclude names the study's other checks" in planner          # threshold_count's own line stays


def test_the_persons_answer_to_the_analysts_question_reaches_the_guard(tmp_path, monkeypatch):
    """Found while building this set: the person's answer to the ANALYST's question went back only as a note, and the analyst ran
    again with the plan's question, so the guard never saw it and a cells rule or a name given there was refused until the rounds
    ran out (the app and the CLI share this loop). The answer is now added to the analyst's question, as the planner's answers
    are; the note stays."""
    import asyncio
    import dataclasses

    import run as ctl
    from part2_support import SYNTHETIC_BASE, part2_config, part2_runtime, synthetic_models

    part2_runtime(monkeypatch, server=SYNTHETIC_BASE)
    seen, real = [], ctl.run_analyst

    async def spy(inputs, **kwargs):
        seen.append(inputs)
        return await real(inputs, **kwargs)

    monkeypatch.setattr(ctl, "run_analyst", spy)
    made = []

    def analyst(prefix):
        made.append(prefix)
        if len(made) == 1:                                       # first the analyst asks about a doubled cell ...
            return FakeModelClient([reply_text(json.dumps({"status": "needs_clarification", "question": "Two plots of one entry in "
                                                           "one rep: the mean of the plots in each rep, or only the entries with "
                                                           "exactly one plot in every rep?"}), model="mock-model")], reported_model="mock-model")
        return synthetic_models().analyst_factory(prefix)        # ... then it answers (the usual script)

    answer = "use the mean of the plots in each rep"
    config = part2_config(question="For synthetic study S1, give mean yield and clone means with n.", offline=False, mock_model=False,
                          direct=False, auto=False, fixture="synthetic", out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    queue = ["approve", answer, "accept"]
    controller = ctl.Controller(config, models=dataclasses.replace(synthetic_models(), analyst_factory=analyst),
                                answer_fn=lambda prompt: queue.pop(0))
    result = asyncio.run(controller.run())
    assert result.execution_status == "completed" and len(seen) == 2 and seen[0].question == controller.plan.question
    assert seen[1].question == co.add_clarification(controller.plan.question, answer)
    assert an.CELLS_STATED["mean"].search(co.person_words(seen[1].question))
    assert any(answer in note for note in seen[1].notes)         # the note that says which question it answers stays


# --- C33.2: "A vs B" ------------------------------------------------------------------------------------------------------------

def test_an_a_vs_b_t_test_takes_the_differences_a_minus_b():
    """C33.2: the coordinator picks the base in its plan step, the analyst passes it on; both read the line. The sign label
    ("t (base - other)" / "(other - base)") stays. "B = base" only when no percent or difference is asked (review of this set):
    C27.3 ("control vs drought ... by how many percentage points did dry matter change from control to drought") and D14.2
    ("irrigated vs drought ... the percent change from the irrigated to the drought arm") keep A as the base."""
    line = ('a t-test asked as "A vs B" or "A compared with B" takes the differences A - B when no percent or difference is asked '
            '(B\'s table as the base, percent="change"); a percent or a difference asked keeps the base its own wording gives (a '
            'change from A to B: A\'s table; "by what percent is B lower than A": A\'s table, percent="reduction"), and t keeps its '
            'sign label')
    assert line in " ".join(an.ANALYST_SYSTEM_PROMPT.split()) and line in " ".join(co.COORDINATOR_SYSTEM_PROMPT.split())
