"""
tests/test_app.py — the chat window (app.py), driven turn by turn without a browser.

What these tests prove, in plain words:
* A question shows the approval screen and NOTHING is fetched; "approve" fetches exactly the approved study/variable
  and shows the answer with its numbers; "accept" finishes the run, recorded as a person's acceptance (not simulated).
* "cancel" ends the run with nothing fetched. A precondition that fails (BRAPI_MODE not live) is said in the chat
  before any model call. Each chat state carries its own run.
* A vague question shows the planner's own question in the chat (on 2026-09-29 it reached only the PowerShell window);
  "cancel" there ends the run, and so does closing the tab. A short answer is ADDED to the first question and is enough;
  after three answers the run ends with the planner's question still shown. The three count for the whole run, also after
  the coordinator asks for a new plan or asks its own question, and the last message lists every round.
* The Analyst's question is shown too; the answer goes back to it with the data already fetched (nothing fetched again).
  "New question" ends a waiting run and clears the chat; while the app works the input is dimmed (locked/unlocked).
  The header is Bidhan's wording: the server, approve, accept, descriptive numbers only.
* The page builds (Gradio) without being launched; a Hugging Face Space refuses to start without a password.
Everything runs against a FAKE live server; the conftest network guard would stop any real socket.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

import app
import run as ctl
from brapi_client import TransportResponse
from cache_store import CacheStore
from llm import FakeModelClient, reply_text
from part2_support import load_fixture, seed_fixture
from part2_support import part2_runtime, synthetic_models, synthetic_plan_draft


@pytest.fixture(autouse=True)
def _part2_flow(monkeypatch):
    """Ported from part2: run the release flow the way part2's mock mode did (see part2_support.part2_runtime)."""
    part2_runtime(monkeypatch)


LIVE_BASE = "https://live.invalid/brapi/v2"
Q = "For synthetic study S1, give mean yield and clone means with n."
VAGUE = "Tell me about a study."                     # names no study: the scripted planner asks "Which study is meant?"


class FakeLiveServer:
    REPLIES = {"/observations": "synthetic_observations_S1_V1.json", "/observationunits": "synthetic_observationunits_S1.json"}

    def __init__(self):
        self.calls = []

    def __call__(self, req):
        endpoint = req.url[len(LIVE_BASE):]
        self.calls.append(endpoint)
        _, body = load_fixture(self.REPLIES[endpoint])
        return TransportResponse(200, {"Content-Type": "application/json"}, body)


@pytest.fixture
def live(tmp_path, monkeypatch):
    """Live settings for this test only, the catalogs cached, a fake server, and factories the app uses."""
    import brapi_client as bc

    monkeypatch.setenv("BRAPI_MODE", "live")
    monkeypatch.setenv("BRAPI_BASE_URL", LIVE_BASE)
    cache = CacheStore(tmp_path / "live_cache")
    for name in ("synthetic_studies.json", "synthetic_variables.json"):
        seed_fixture(cache, LIVE_BASE, name)
    notes = tmp_path / "ACCESS_NOTES.md"
    notes.write_text("Reviewed by Bidhan on: 2026-09-28\n", encoding="utf-8")
    server = FakeLiveServer()
    monkeypatch.setattr(bc, "RequestsTransport", lambda: server)

    def make_config(question):
        return ctl.RunConfig(question=question, offline=False, mock_model=False, direct=False, auto=False,
                             out_dir=tmp_path / "out", cache_dir=tmp_path / "live_cache", access_notes=notes,
                             max_clarification_rounds=app.CHAT_CLARIFICATION_ROUNDS)

    kwargs = dict(make_config=make_config, make_models=synthetic_models, preflight=lambda q: make_config(q).validate())
    return server, kwargs, tmp_path


def turn(message, history, state, kwargs):
    """Run one chat turn to its end and return the last (history, state)."""
    out = list(app.respond(message, history, state, **kwargs))
    assert out[-1][2] == ""                                                       # the text box is cleared
    return out[-1][0], out[-1][1]


def test_approve_then_accept_in_the_chat(live):
    server, kwargs, tmp_path = live
    history, state = turn(Q, [], {}, kwargs)
    assert "Review this plan before approving" in history[-1]["content"] and "Reply **approve**" in history[-1]["content"] and "job" in state
    assert "| S1 | SYNTHETIC yield trial S1 |" in history[-1]["content"]                # the scope, in words, on the screen
    assert server.calls == []                                                     # the question alone fetches nothing
    history, state = turn("approve", history, state, kwargs)
    assert server.calls == ["/observations", "/observationunits"]
    assert "Numbers (from typed claims only)" not in history[-1]["content"] and "Reply **accept**" in history[-1]["content"]
    assert "Numbers (from typed claims only)" in app.full_report(state)  # complete audit evidence remains accessible
    assert "GA · clone A" in history[-1]["content"] and "GB · clone B" in history[-1]["content"]
    assert "| Mean | 12 t/ha |" in history[-1]["content"] and "| Mean | 24 t/ha |" in history[-1]["content"]
    history, state = turn("accept", history, state, kwargs)
    assert "**Accepted.**" in history[-1]["content"] and "job" not in state
    assert "**Human acceptance:** accepted" in app.full_report(state)
    run_id = next((tmp_path / "out").glob("run_*/answer.json")).parent.name
    result = ctl.RunResult.model_validate_json((tmp_path / "out" / run_id / "answer.json").read_text(encoding="utf-8"))
    assert result.human_acceptance == "accepted" and result.manifest.origin == "live" and "_human_" in result.approval_id
    assert any(n == "acceptance recorded: accept" for n in result.notes)        # a person typed it: not labelled simulated
    assert [m["role"] for m in history] == ["user", "assistant"] * 3


def test_cancel_fetches_nothing_and_ends_the_run(live):
    server, kwargs, _ = live
    history, state = turn(Q, [], {}, kwargs)
    history, state = turn("cancel", history, state, kwargs)
    assert "canceled" in history[-1]["content"] and server.calls == [] and "job" not in state


def test_a_failed_precondition_is_said_before_any_model_call(live, monkeypatch):
    server, kwargs, _ = live
    monkeypatch.setenv("BRAPI_MODE", "offline")
    history, state = turn(Q, [], {}, kwargs)
    assert "cannot run yet" in history[-1]["content"] and "BRAPI_MODE is not 'live'" in history[-1]["content"]
    assert state == {} and server.calls == []
    assert turn("   ", [], {}, kwargs) == ([], {})                                # an empty message does nothing


def test_each_chat_state_has_its_own_run(live):
    _, kwargs, _ = live
    _, first = turn(Q, [], {}, kwargs)
    _, second = turn(Q, [], {}, kwargs)
    assert first["job"] is not second["job"] and first["job"].controller.run_id != second["job"].controller.run_id
    for state in (first, second):                                                 # the tabs go away: their runs end
        app._close_job(state)
        kind, result = state["job"].next_event(timeout=60)
        assert kind == "done" and result.execution_status == "canceled"


def test_the_planners_question_is_shown_and_cancel_ends_the_run(live):
    server, kwargs, _ = live
    history, state = turn(VAGUE, [], {}, kwargs)
    shown = history[-1]["content"]
    assert "Which study is meant?" in shown and "(answer 1 of 3)" in shown and "reply **cancel**" in shown and "job" in state
    history, state = turn("cancel", history, state, kwargs)                      # there 'cancel' is the controller's empty answer
    assert "canceled" in history[-1]["content"] and "job" not in state and server.calls == []
    _, closing = turn(VAGUE, [], {}, kwargs)                                      # a tab closed at this prompt cancels the run;
    app._close_job(closing)                                                       # it does not plan the word 'cancel'
    kind, result = closing["job"].next_event(timeout=60)
    assert kind == "done" and result.execution_status == "canceled"
    assert app._shorten("short") == "short" and app._shorten("a" * 700).startswith("a" * 600) and "shortened" in app._shorten("a" * 700)


def test_a_short_answer_is_added_to_the_first_question(live):
    server, kwargs, tmp_path = live
    history, state = turn(VAGUE, [], {}, kwargs)
    history, state = turn("S1, mean yield per clone", history, state, kwargs)    # a few words; the first question stays
    assert "Review this plan before approving" in history[-1]["content"] and server.calls == []
    history, state = turn("cancel", history, state, kwargs)
    run_id = next((tmp_path / "out").glob("run_*/answer.json")).parent.name
    result = ctl.RunResult.model_validate_json((tmp_path / "out" / run_id / "answer.json").read_text(encoding="utf-8"))
    assert result.plan.question == f"{VAGUE}\nClarification 1 from the person: S1, mean yield per clone"
    assert any(n.startswith("clarification 1: the planner asked: Which study is meant?") and n.endswith("answer: S1, mean yield per clone")
               for n in result.notes)                                             # a person's answer: not labelled simulated


def test_three_answers_at_most_and_the_question_stays_visible(live):
    server, kwargs, _ = live
    history, state = turn(VAGUE, [], {}, kwargs)
    for number in (2, 3):
        history, state = turn(VAGUE, history, state, kwargs)                     # still vague: asked again, one fewer try left
        assert f"(answer {number} of 3)" in history[-1]["content"] and "job" in state
    history, state = turn(VAGUE, history, state, kwargs)
    last = history[-1]["content"]
    assert "More detail is needed" in last and "still not ready after 3 clarification rounds" in last and "Which study is meant?" in last
    assert "job" not in state and server.calls == []


ASKS = {"status": "needs_clarification", "question": "Which study is meant?"}
READY = {"status": "completed", "payload": synthetic_plan_draft("S1", "V1")}


def scripted(*replies: dict) -> FakeModelClient:
    return FakeModelClient([reply_text(json.dumps(r), model="mock-model") for r in replies], reported_model="mock-model")


def last_run(tmp_path: Path) -> ctl.RunResult:
    answer = next((tmp_path / "out").glob("run_*/answer.json"))
    return ctl.RunResult.model_validate_json(answer.read_text(encoding="utf-8"))


def test_the_cap_holds_for_the_whole_run_and_the_last_message_lists_every_round(live):
    """Run 20261015T165633: after two answers the plan was ready, the coordinator asked for a new plan, and the planner then
    asked three more times ("answer 4 of 3", "answer 5 of 3"); the last message showed only two rounds."""
    server, kwargs, tmp_path = live
    planner = scripted(ASKS, ASKS, READY, ASKS, ASKS, ASKS)
    supervisor = scripted({"status": "completed", "payload": {"action": "revise_plan", "reason": "Scripted: a new plan.",
                                                              "instruction": "Reframe the request and ask which study is meant."}})
    kw = {**kwargs, "make_models": lambda: dataclasses.replace(synthetic_models(), coordinator=planner, supervisor=supervisor)}
    history, state = turn(VAGUE, [], {}, kw)
    shown = [history[-1]["content"]]
    answers = ("a yield study", "S1, mean yield per clone", "list all traits")
    for answer in answers:
        history, state = turn(answer, history, state, kw)
        shown.append(history[-1]["content"])
    for number in (1, 2, 3):                        # the third prompt comes after the coordinator's new plan: still "3 of 3"
        assert f"(answer {number} of 3)" in shown[number - 1]
    assert not any("answer 4 of 3" in text for text in shown) and "job" not in state and server.calls == []
    last = shown[-1]
    assert "More detail is needed" in last and "still not ready after 3 clarification rounds" in last
    for number, answer in enumerate(answers, 1):    # every round, not only the last ones
        assert f"clarification {number}: the planner asked: Which study is meant? - answer: {answer}" in last
    result = last_run(tmp_path)
    assert [n.split(":", 1)[0] for n in result.notes if n.startswith("clarification ")] == [f"clarification {n}" for n in (1, 2, 3)]
    assert result.execution_status == "needs_clarification" and len(planner.requests) == 5


def test_the_coordinators_own_question_counts_toward_the_same_cap(live):
    server, kwargs, tmp_path = live
    planner = scripted(ASKS, READY, ASKS, ASKS)
    supervisor = scripted({"status": "completed", "payload": {"action": "clarify", "reason": "Scripted: one question first.",
                                                              "question": "Should the plan include 2025?"}})
    kw = {**kwargs, "make_models": lambda: dataclasses.replace(synthetic_models(), coordinator=planner, supervisor=supervisor)}
    history, state = turn(VAGUE, [], {}, kw)                                      # answer 1 of 3: the planner's question
    history, state = turn("S1, mean yield per clone", history, state, kw)
    assert "Should the plan include 2025?" in history[-1]["content"]             # answer 2: the coordinator's question
    history, state = turn("no, only 2026", history, state, kw)
    assert "(answer 3 of 3)" in history[-1]["content"]
    history, state = turn("list all traits", history, state, kw)
    last = history[-1]["content"]
    assert "still not ready after 3 clarification rounds" in last and "job" not in state and server.calls == []
    assert "clarification 2: the coordinator asked: Should the plan include 2025? - answer: no, only 2026" in last
    assert "clarification 3: the planner asked: Which study is meant? - answer: list all traits" in last


@pytest.mark.parametrize("answers, offered", [(2, True), (3, False)])
def test_the_coordinator_may_ask_only_while_answers_are_left(answers, offered):
    from types import SimpleNamespace

    from coordination_runtime import _allowed

    c = SimpleNamespace(plan=SimpleNamespace(status="ready"), retrieval=None, supervision={"plan_revisions": 0},
                        clarification_answers=answers, supervisor_answers=0, config=SimpleNamespace(max_clarification_rounds=3))
    assert ("clarify" in _allowed(c, "planned")) is offered


def test_the_analysts_question_is_shown_and_the_answer_reuses_the_fetched_data(live):
    """Run 193941 (2026-09-29): the Analyst stopped with a question the chat never showed. Now it is shown, and the answer goes
    back to the Analyst with the SAME artifacts: nothing is fetched again."""
    server, kwargs, tmp_path = live
    made = []

    def analyst(prefix):
        made.append(prefix)
        if len(made) == 1:                                                        # first the Analyst asks ...
            return FakeModelClient([reply_text(json.dumps({"status": "needs_clarification", "question": "Use the seasons column instead?"}),
                                               model="mock-model")], reported_model="mock-model")
        return synthetic_models().analyst_factory(prefix)                     # ... then it answers (the usual script)

    kw = {**kwargs, "make_models": lambda: dataclasses.replace(synthetic_models(), analyst_factory=analyst)}
    history, state = turn(Q, [], {}, kw)
    history, state = turn("approve", history, state, kw)
    shown = history[-1]["content"]
    assert "Use the seasons column instead?" in shown and "(answer 1 of 3)" in shown and "nothing new is fetched" in shown
    fetched = list(server.calls)
    history, state = turn("yes, use seasons", history, state, kw)
    assert "| Mean |" in history[-1]["content"] and server.calls == fetched      # the same data, no new fetch
    history, state = turn("accept", history, state, kw)
    run_id = next((tmp_path / "out").glob("run_*/answer.json")).parent.name
    result = ctl.RunResult.model_validate_json((tmp_path / "out" / run_id / "answer.json").read_text(encoding="utf-8"))
    assert result.execution_status == "completed" and result.human_acceptance == "accepted" and len(made) == 2
    assert any(n.startswith("analyst clarification 1: the analyst asked: Use the seasons column instead?")
               and n.endswith("answer: yes, use seasons") for n in result.notes)


def test_the_reply_buttons_follow_what_the_run_waits_for(live):
    """2026-09-29: '1032', 'done' and 'accept' were typed where the app waited for approve / cancel. The buttons offer the
    replies that fit: Approve and Cancel at the approval screen, Accept and Reject after the answer, Cancel at a question."""
    _, kwargs, _ = live
    history, state = turn(Q, [], {}, kwargs)
    assert app.reply_buttons(state) == (True, True, False, False)
    history, state = turn("approve", history, state, kwargs)
    assert app.reply_buttons(state) == (False, False, True, True)
    assert [u["visible"] for u in app.after_step(state)[2:]] == [False, False, True, True]
    history, state = turn("accept", history, state, kwargs)
    assert app.reply_buttons(state) == (False, False, False, False) and "job" not in state
    _, asking = turn(VAGUE, [], {}, kwargs)
    assert app.reply_buttons(asking) == (False, True, False, False)                # the planner asks: Cancel only
    app._close_job(asking)
    assert asking["job"].next_event(timeout=60)[0] == "done"


def test_new_question_ends_a_waiting_run_and_clears_the_chat(live):
    server, kwargs, _ = live
    history, state = turn(Q, [], {}, kwargs)                                     # the run waits at the approval screen
    job = state["job"]
    assert app.start_over(state) == ([], {}, "")
    kind, result = job.next_event(timeout=60)
    assert kind == "done" and result.execution_status == "canceled" and server.calls == []
    assert app.start_over({}) == ([], {}, "")                                    # nothing waiting: it only clears


def test_the_page_builds_and_a_space_needs_a_password(live, monkeypatch):
    import gradio as gr

    _, kwargs, _ = live
    ui = app.build_ui(**kwargs)
    assert isinstance(ui, gr.Blocks)
    assert isinstance(app.build_theme(), gr.themes.Soft)
    tabs = [block.label for block in ui.blocks.values() if isinstance(block, gr.Tab)]
    assert tabs == ["Assistant", "FAQ & How to use"]
    assert not any(getattr(block, "elem_id", None) == "brapi-context" for block in ui.blocks.values())
    source = app.source_markdown()
    assert f"Current data source: `{LIVE_BASE}`" in source and "does not edit the database" in source
    assert [u["interactive"] for u in app.locked()] == [False] * 6                     # box, New question and the four reply buttons
    after = app.after_step({})
    assert [u["interactive"] for u in after] == [True] * 6 and [u["visible"] for u in after[2:]] == [False] * 4   # nothing waits
    long = "| a |\n|---|\n" + "\n".join(f"| {i} |" for i in range(40))
    assert app._shorten_tables(long).count("\n| ") == app.TABLE_ROWS_SHOWN and "25 more rows: all of them are in answer.md" in app._shorten_tables(long)
    assert app.live_config("any question").max_clarification_rounds == app.CHAT_CLARIFICATION_ROUNDS == 3
    monkeypatch.setenv("SPACE_ID", "someone/brapi-assistant")
    monkeypatch.delenv("APP_USERNAME", raising=False)
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("APP_ALLOW_OPEN", raising=False)
    with pytest.raises(SystemExit, match="without APP_USERNAME and APP_PASSWORD"):
        app.launch_options()
