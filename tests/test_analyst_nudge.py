"""
tests/test_analyst_nudge.py — one nudge when the Analyst finishes with no claim (analyst_no_claim_20261019).

Eval 20261019T203241Z: Luna's analyst called table_info and missing_report, wrote the right count in its narrative and
finished "completed" with no claim, so the run ended "Could not complete this question". The fix: when a "completed"
final arrives and no claim-making tool has returned a claim, the loop asks ONCE more with a fixed message, drawing one
repair from the shared pool. It computes nothing: code still builds every claim from tool records. No nudge after a
claim, on needs_clarification, with the repair pool used up, or with no model call or tool call left.

Everything runs offline on SYNTHETIC artifacts with SCRIPTED model replies.
"""
from __future__ import annotations

import json

import anyio

import agents.analyst as an
import brapi_mcp_server as srv
import llm
from contracts import MeasurementMeta
from llm import Budget, FakeModelClient, reply_text, reply_tools
from part2_support import synthetic_context

RUN = "run_nudge"
# The exact wording measured on gpt-6-luna (analyst_no_claim_20261019); pinned here, not read from the code.
TESTED_WORDING = ("No tool has made a claim yet, so nothing can be reported. table_info and missing_report show counts for checking, "
                  "but they make no claims. Only numeric_summary, group_stats and count_records make claims: numeric_summary's count "
                  "claim is the number of valid values it used (one per unit under duplicate_policy=mean_per_unit); count_records "
                  "counts rows (all rows, or the rows kept by filter_rows, e.g. op is_valid_number keeps each row with a valid value). "
                  "Call the tool that states the number the question asks for, then finish again with the JSON object. If no tool "
                  "can state it, finish with needs_clarification.")
NOT_JSON = "The count is 6."                       # a plain-text final: the loop's existing non-JSON repair, not the nudge
NO_CLAIM_FINAL = json.dumps({"status": "completed", "payload": {"methods": ["missing_report on the value column"], "caveats": [],
                                                                "narrative": "The table has 6 valid values.", "key_claim_ids": []}})
CLAIMED_FINAL = json.dumps({"status": "completed", "payload": {"methods": ["numeric_summary on the value column"], "caveats": [],
                                                               "narrative": "numeric_summary states the count.", "key_claim_ids": []}})
CLARIFY = json.dumps({"status": "needs_clarification", "question": "Which variable is meant?"})


# --- helpers ---------------------------------------------------------------------------------------

def make_ctx(tmp_path):
    return synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id=RUN)


def s1_inputs(ctx) -> an.AnalysisInput:
    return an.synthetic_inputs(ctx, "SYNTHETIC: how many valid values does V1 have in S1?")


def inspect_all(inputs: an.AnalysisInput):
    return reply_tools([("table_info", json.dumps({"artifact_id": a})) for a in inputs.artifact_ids])


def call(name: str, artifact_id: str):
    return reply_tools([(name, json.dumps({"artifact_id": artifact_id}))])


def run(ctx, inputs: an.AnalysisInput, replies: list, *, budget: Budget | None = None):
    model = FakeModelClient(replies)
    budget = budget or Budget()

    async def go():
        return await an.run_analyst(inputs, model=model, ctx=ctx, budget=budget, run_id=RUN, log_dir=ctx.registry.out_root)

    report, agent, narrative = anyio.run(go)
    return report, agent, model, budget


def sent_nudges(model: FakeModelClient) -> list[str]:
    """User messages after the question in the conversation the model saw last, without the non-JSON repair's own message."""
    users = [m["content"] for m in model.requests[-1]["messages"] if m["role"] == "user"]
    return [u for u in users[1:] if not u.startswith("Your reply was not a valid JSON object")]


def repair_events(ctx) -> list[dict]:
    lines = (ctx.registry.out_root / RUN / "llm_analyst.jsonl").read_text(encoding="utf-8").splitlines()
    return [e for e in map(json.loads, lines) if e["event"] == "repair"]


# --- 1. the nudge recovers a run that would end with no claim -------------------------------------

def test_a_no_claim_finish_gets_one_nudge_and_the_next_tool_call_makes_the_claim(tmp_path):
    ctx = make_ctx(tmp_path)
    inputs = s1_inputs(ctx)
    obs = inputs.artifact_ids[0]
    report, agent, model, budget = run(ctx, inputs, [inspect_all(inputs), call("missing_report", obs), reply_text(NO_CLAIM_FINAL),
                                                     call("numeric_summary", obs), reply_text(CLAIMED_FINAL)])
    assert report.status == "completed" and model._replies == []
    count = next(c for c in report.claims if c.kind == "count")
    assert count.transformation.startswith("count of valid values") and count.value == count.denominator.n   # a tool's claim
    assert sent_nudges(model) == [TESTED_WORDING]
    # the model's own final stays in the conversation, followed by the nudge, as in the replays
    assert model.requests[3]["messages"][-2:] == [{"role": "assistant", "content": NO_CLAIM_FINAL},
                                                  {"role": "user", "content": TESTED_WORDING}]
    assert budget.repairs == 1 and agent.model_calls == 5
    events = repair_events(ctx)
    assert len(events) == 1 and events[0]["reason"].startswith("completion check")


def test_the_nudge_is_sent_at_most_once_and_a_second_no_claim_finish_still_fails(tmp_path):
    ctx = make_ctx(tmp_path)
    inputs = s1_inputs(ctx)
    budget = Budget()            # two repairs: a second nudge would fit the pool, so "once" is the loop's own rule
    report, agent, model, _ = run(ctx, inputs, [inspect_all(inputs), reply_text(NO_CLAIM_FINAL), reply_text(NO_CLAIM_FINAL),
                                               reply_text(NO_CLAIM_FINAL)], budget=budget)
    assert report.status == "failed" and report.claims == []
    assert any("no tool produced a claim" in c for c in report.caveats)
    assert sent_nudges(model) == [TESTED_WORDING] and len(model._replies) == 1 and agent.model_calls == 3
    assert budget.repairs == 1 and len(repair_events(ctx)) == 1


# --- 2. where the nudge must NOT fire ---------------------------------------------------------------

def test_no_nudge_once_a_tool_has_made_a_claim(tmp_path):
    ctx = make_ctx(tmp_path / "claimed")
    inputs = s1_inputs(ctx)
    report, agent, model, budget = run(ctx, inputs, [inspect_all(inputs), call("numeric_summary", inputs.artifact_ids[0]),
                                                     reply_text(CLAIMED_FINAL)])
    assert report.status == "completed" and sent_nudges(model) == [] and agent.model_calls == 3 and budget.repairs == 0
    # the same final with no claim-making call before it gets the nudge: the boundary is the claim, not the final's words
    ctx2 = make_ctx(tmp_path / "unclaimed")
    inputs2 = s1_inputs(ctx2)
    report2, _, model2, _ = run(ctx2, inputs2, [inspect_all(inputs2), reply_text(CLAIMED_FINAL),
                                                call("numeric_summary", inputs2.artifact_ids[0]), reply_text(CLAIMED_FINAL)])
    assert report2.status == "completed" and sent_nudges(model2) == [TESTED_WORDING]


def test_no_nudge_on_needs_clarification(tmp_path):
    ctx = make_ctx(tmp_path / "clarify")
    inputs = s1_inputs(ctx)
    report, agent, model, budget = run(ctx, inputs, [inspect_all(inputs), reply_text(CLARIFY)])
    assert report.status == "needs_clarification" and sent_nudges(model) == [] and agent.model_calls == 2 and budget.repairs == 0
    ctx2 = make_ctx(tmp_path / "completed")
    inputs2 = s1_inputs(ctx2)
    _, _, model2, _ = run(ctx2, inputs2, [inspect_all(inputs2), reply_text(NO_CLAIM_FINAL), reply_text(CLARIFY)])
    assert sent_nudges(model2) == [TESTED_WORDING]            # the nudge also allows a question as the answer


def test_no_nudge_when_the_repair_pool_is_used_up(tmp_path):
    ctx = make_ctx(tmp_path / "empty")
    inputs = s1_inputs(ctx)
    report, agent, model, budget = run(ctx, inputs, [inspect_all(inputs), reply_text(NO_CLAIM_FINAL)], budget=Budget(max_repairs=0))
    assert report.status == "failed" and any("no tool produced a claim" in c for c in report.caveats)     # exactly as before
    assert sent_nudges(model) == [] and agent.model_calls == 2 and budget.repairs == 0
    # one pool for every repair: a non-JSON final before used the only repair, so no nudge is left for the no-claim final
    ctx2 = make_ctx(tmp_path / "shared")
    inputs2 = s1_inputs(ctx2)
    report2, agent2, model2, budget2 = run(ctx2, inputs2, [inspect_all(inputs2), reply_text(NOT_JSON), reply_text(NO_CLAIM_FINAL)],
                                           budget=Budget(max_repairs=1))
    assert report2.status == "failed" and any("no tool produced a claim" in c for c in report2.caveats)
    assert sent_nudges(model2) == [] and agent2.model_calls == 3 and budget2.repairs == 1
    ctx3 = make_ctx(tmp_path / "left")
    inputs3 = s1_inputs(ctx3)
    _, _, model3, budget3 = run(ctx3, inputs3, [inspect_all(inputs3), reply_text(NOT_JSON), reply_text(NO_CLAIM_FINAL),
                                                reply_text(NO_CLAIM_FINAL)], budget=Budget(max_repairs=2))
    assert sent_nudges(model3) == [TESTED_WORDING] and budget3.repairs == 2


def test_no_nudge_when_no_model_call_is_left(tmp_path):
    ctx = make_ctx(tmp_path / "last")
    inputs = s1_inputs(ctx)
    report, agent, model, budget = run(ctx, inputs, [inspect_all(inputs), reply_text(NO_CLAIM_FINAL), reply_text(NO_CLAIM_FINAL)],
                                       budget=Budget(max_model_calls=2))
    assert report.status == "failed" and any("no tool produced a claim" in c for c in report.caveats)     # not "limit reached"
    assert sent_nudges(model) == [] and agent.model_calls == 2 and budget.repairs == 0 and len(model._replies) == 1
    ctx2 = make_ctx(tmp_path / "one_more")
    inputs2 = s1_inputs(ctx2)
    _, agent2, model2, _ = run(ctx2, inputs2, [inspect_all(inputs2), reply_text(NO_CLAIM_FINAL), reply_text(NO_CLAIM_FINAL)],
                               budget=Budget(max_model_calls=3))
    assert sent_nudges(model2) == [TESTED_WORDING] and agent2.model_calls == 3


def test_no_nudge_when_no_tool_call_is_left(tmp_path):
    ctx = make_ctx(tmp_path / "spent")
    inputs = s1_inputs(ctx)                                    # two supplied tables: two table_info calls
    report, _, model, budget = run(ctx, inputs, [inspect_all(inputs), reply_text(NO_CLAIM_FINAL), reply_text(NO_CLAIM_FINAL)],
                                   budget=Budget(max_tool_calls=2))
    assert report.status == "failed" and any("no tool produced a claim" in c for c in report.caveats)     # not "limit reached"
    assert sent_nudges(model) == [] and budget.tool_calls == 2 and budget.repairs == 0 and len(model._replies) == 1
    ctx2 = make_ctx(tmp_path / "one_more")
    inputs2 = s1_inputs(ctx2)
    _, _, model2, _ = run(ctx2, inputs2, [inspect_all(inputs2), reply_text(NO_CLAIM_FINAL), reply_text(NO_CLAIM_FINAL)],
                          budget=Budget(max_tool_calls=3))
    assert sent_nudges(model2) == [TESTED_WORDING]


# --- 3. the message is fixed text: it states no number --------------------------------------------

def test_the_nudge_is_the_tested_wording_and_states_no_number(tmp_path):
    sent = []
    ctx = make_ctx(tmp_path / "s1")
    inputs = s1_inputs(ctx)
    _, _, model, _ = run(ctx, inputs, [inspect_all(inputs), reply_text(NO_CLAIM_FINAL), reply_text(NO_CLAIM_FINAL)])
    sent += sent_nudges(model)
    # a different table (other rows, other values): the same words
    ctx2 = make_ctx(tmp_path / "other")
    t = json.loads((an.PART2_DIR / "tests" / "fixtures" / "analysis" / "tied_means.json").read_text(encoding="utf-8"))["tables"]["three_clones"]
    columns = {c: srv.OBSERVATION_COLUMNS.get(c, "text") for c in t["columns"]}
    art = ctx2.registry.save_table(t["rows"], kind=t["kind"], columns=columns, complete=t["complete"], study_ids=t["study_ids"],
                                   variable_ids=t["variable_ids"], measurement=MeasurementMeta(**t["measurement"])).artifact_id
    inputs2 = an.AnalysisInput(question="SYNTHETIC: how many valid values?", artifact_ids=[art], requested_variable_ids=t["variable_ids"])
    _, _, model2, _ = run(ctx2, inputs2, [inspect_all(inputs2), reply_text(NO_CLAIM_FINAL), reply_text(NO_CLAIM_FINAL)])
    sent += sent_nudges(model2)
    assert sent == [TESTED_WORDING, TESTED_WORDING]
    assert not any(ch.isdigit() for ch in sent[0])                                   # it names tools, never a value


# --- 4. the loop hook itself (llm.run_agent_loop) ---------------------------------------------------

class NoTools:
    async def schemas(self):
        return []

    async def call(self, name, args):              # pragma: no cover - never reached: these scripts call no tool
        raise AssertionError("no tool call expected")


FINAL = json.dumps({"status": "completed", "payload": {"answer": 5}})


def run_loop(model, tmp_path, *, budget=None, **over):
    async def go():
        return await llm.run_agent_loop(model, NoTools(), agent_name="t", system_prompt="be exact", user_message="add 2 and 3",
                                        budget=budget or Budget(), log_dir=tmp_path / "out", run_id="run_t",
                                        model_requested="fake-model", clock=lambda: 0.0, **over)

    return anyio.run(go)


def test_the_loop_hook_asks_once_more_only_for_a_completed_final(tmp_path):
    seen: list[dict] = []

    def check(payload):
        seen.append(payload)
        return "Please call a tool first."

    plain = FakeModelClient([reply_text(FINAL)])                                   # no hook (every other agent): ends at once
    assert run_loop(plain, tmp_path / "plain").status == "completed" and len(plain.requests) == 1
    # Claude's blocks (thinking included) go back with the final, as for the non-JSON repair
    blocks = ({"type": "thinking", "thinking": "SYNTHETIC", "signature": "SIG"}, {"type": "text", "text": FINAL})
    claude_final = llm.ModelReply(content=FINAL, tool_calls=[], finish_reason="stop", model="fake-model", usage=None,
                                  provider_content=blocks)
    model = FakeModelClient([claude_final, reply_text(FINAL), reply_text(FINAL)])
    budget = Budget()
    result = run_loop(model, tmp_path / "hook", budget=budget, check_completed=check)
    assert result.status == "completed" and result.payload == {"answer": 5}
    assert len(model.requests) == 2 and len(model._replies) == 1 and budget.repairs == 1     # asked once more, never twice
    assert model.requests[1]["messages"][-2:] == [{"role": "assistant", "content": FINAL, "provider_content": blocks},
                                                  {"role": "user", "content": "Please call a tool first."}]
    assert seen == [{"answer": 5}]                                                          # consulted once, not again
    seen.clear()
    clarify = FakeModelClient([reply_text(CLARIFY)])
    assert run_loop(clarify, tmp_path / "clarify", check_completed=check).status == "needs_clarification"
    assert seen == [] and len(clarify.requests) == 1                                        # never consulted for a question
    spent = FakeModelClient([reply_text(FINAL)])
    result = run_loop(spent, tmp_path / "spent", budget=Budget(max_repairs=0), check_completed=check)
    assert result.status == "completed" and len(spent.requests) == 1                        # no repair left: finishes as before
    quiet = FakeModelClient([reply_text(FINAL)])
    result = run_loop(quiet, tmp_path / "quiet", check_completed=lambda payload: None)
    assert result.status == "completed" and len(quiet.requests) == 1
