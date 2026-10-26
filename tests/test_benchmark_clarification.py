"""The benchmark answers one clarification by itself (coach, 2026-10-22 15:53 UTC, harness-only): the planner's "More than one
trait matches the question: ..." with the variable(s) the sub-question names, once per question, and says so in its row and
table. 28 of 42 tool-1 answers and 21 of 45 tool-2 answers had stopped there. The app and its defaults are unchanged; the
scripted approver still never answers any other question. Everything here is SYNTHETIC: fake chat turns, no model, no request."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BENCHMARK = Path(__file__).resolve().parents[1] / "eval" / "benchmark.py"
_spec = importlib.util.spec_from_file_location("_eval_benchmark_clarification_under_test", BENCHMARK)
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench
_spec.loader.exec_module(bench)
harness = bench.harness

BETA = "Content of beta-carotene in dry weight basis in raw storage roots measuring mg per 100g|CO_331:0000289"
DRY = "Storage root dry matter content computing percent|CO_331:0000297"
ASKED = ("I need one more detail before I can make a plan (answer 1 of 3):\n\n> More than one trait matches the question: "
         f"76619 ({BETA}), 76605 ({DRY}), 78652 (Storage root dry matter content computing percent|day 1|after harvest|COMP:0000102). "
         "Which one?\n\nReply with your answer.")


# --- 1. which answer -----------------------------------------------------------------------------------------------------

def test_the_answer_is_the_listed_variable_the_sub_question_names_by_its_term():
    assert bench.trait_answer(f"What is the mean of '{BETA}' in study 4610?", ASKED) == "76619"


def test_two_named_variables_are_both_given_in_the_listed_order():
    assert bench.trait_answer(f"Correlate '{DRY}' with '{BETA}' across trials.", ASKED) == "76619 and 76605"


def test_the_term_decides_between_two_variables_of_the_same_name():
    shown = ("> More than one trait matches the question: 76605 (Storage root dry matter content computing percent|CO_331:0000297), "
             "99999 (Storage root dry matter content computing percent|CO_331:0009999). Which one?")
    assert bench.trait_answer(f"What is the mean of '{DRY}' in 4610?", shown) == "76605"


def test_a_name_without_its_term_is_matched_whole_and_in_any_case():
    assert bench.trait_answer("What is the mean storage root DRY MATTER CONTENT COMPUTING PERCENT in 4610?", ASKED) == "76605"


@pytest.mark.parametrize("question", ["What is the mean yield in 4610?", "What is the mean of 'Content of beta-carotene' in 4610?"])
def test_no_answer_when_the_sub_question_names_none_of_the_listed_variables(question):
    assert bench.trait_answer(question, ASKED) is None


@pytest.mark.parametrize("shown", ["The question names no known trait. Which variable is meant: 76619 (x|CO_331:0000289)?",
                                   "I have the data and need one more detail before I can answer (answer 1 of 3): which study?", ""])
def test_no_answer_to_any_other_question(shown):
    assert bench.trait_answer(f"What is the mean of '{BETA}'?", shown) is None


# --- 2. the scripted chat: one answer, once, recorded ----------------------------------------------------------------------

class FakeJob:
    def __init__(self, tmp_path):
        self.waiting_on = harness.CLARIFY_PROMPT
        self.closed = False
        self.thread = SimpleNamespace(is_alive=lambda: False, join=lambda timeout=None: None)
        self.controller = SimpleNamespace(config=SimpleNamespace(out_dir=tmp_path), run_id="run_fake", run_dir=tmp_path / "run_fake")

    def close(self):
        self.closed = True


def scripted_turns(monkeypatch, job, screens):
    """screens: what the chat shows after each turn, with what the job then waits on (None = the run ended)."""
    typed = []

    def turn(message, history, state, chat):
        typed.append(message)
        text, waiting = screens[len(typed) - 1]
        job.waiting_on = waiting
        history = [*history, {"role": "user", "content": message}, {"role": "assistant", "content": text}]
        return history, {"job": job if waiting is not None else None}

    monkeypatch.setattr(harness, "_turn", turn)
    return typed


def test_the_harness_answers_the_trait_question_once_then_approves_as_before(monkeypatch, tmp_path):
    job = FakeJob(tmp_path)
    typed = scripted_turns(monkeypatch, job, [(ASKED, harness.CLARIFY_PROMPT), ("Approval screen", "approve / edit / cancel: "),
                                               ("Data review", harness.DATA_REVIEW_PROMPT), ("**Summary** Mean 6.78", None)])
    question = f"What is the mean of '{BETA}' in study 4610?"
    outcome = harness.ask(question, {}, answer_clarification=lambda shown: bench.trait_answer(question, shown))
    assert typed == [question, "76619", "approve", "continue"] and outcome.typed == ["76619", "approve", "continue"]
    assert outcome.answered == ["clarification answered by harness: 76619"] and outcome.status == "run ended"
    assert outcome.shown == "**Summary** Mean 6.78" and not job.closed


def test_the_same_question_asked_again_is_not_answered_twice(monkeypatch, tmp_path):
    job = FakeJob(tmp_path)
    scripted_turns(monkeypatch, job, [(ASKED, harness.CLARIFY_PROMPT), (ASKED, harness.CLARIFY_PROMPT)])
    question = f"Correlate '{DRY}' with '{BETA}'."
    outcome = harness.ask(question, {}, answer_clarification=lambda shown: bench.trait_answer(question, shown))
    assert outcome.typed == ["76619 and 76605"] and outcome.answered == ["clarification answered by harness: 76619 and 76605"]
    assert outcome.status == "stopped: the app asked a question (the harness never answers one)" and job.closed


def test_without_the_hook_the_harness_still_never_answers(monkeypatch, tmp_path):
    job = FakeJob(tmp_path)
    scripted_turns(monkeypatch, job, [(ASKED, harness.CLARIFY_PROMPT)])
    outcome = harness.ask(f"What is the mean of '{BETA}'?", {})
    assert outcome.typed == [] and outcome.answered == [] and job.closed
    assert outcome.status == "stopped: the app asked a question (the harness never answers one)"


@pytest.mark.parametrize("prompt", ["ANALYST_PROMPT", "RETRIEVER_PROMPT", "NEXT_STEP_PROMPT"])
def test_only_the_planners_question_is_answered_never_the_analysts_retrievers_or_next_steps(monkeypatch, tmp_path, prompt):
    job = FakeJob(tmp_path)
    scripted_turns(monkeypatch, job, [(ASKED, getattr(harness, prompt))])
    outcome = harness.ask(f"What is the mean of '{BETA}'?", {}, answer_clarification=lambda shown: "76619")
    assert outcome.typed == [] and outcome.answered == [] and job.closed


# --- 3. the runner passes the sub-question and reports the answer -----------------------------------------------------------

SUB = {"sub_id": "X1.1", "parent": "X1", "key_facts": [0], "tier": "T1", "question": f"What is the mean of '{BETA}' in study 4610?",
       "expected": [{"value": 6.78, "kind": "mean", "decimals": 2, "abs": 0.005}], "expected_text": [], "alternatives": [],
       "tolerance": {"rule": "relative", "rel": 0.005},
       "studies": [], "folders": [], "statistic": "mean", "caveat": None, "gold_basis": "test"}


def test_the_runner_answers_for_its_sub_question_and_the_row_and_table_say_so(monkeypatch, tmp_path):
    seen = {}

    def fake_ask(question, chat, *, wait=120.0, answer_clarification=None):
        seen["answer"] = answer_clarification(ASKED) if answer_clarification else None
        return harness.Outcome(shown="**Summary**\n\n| Mean | 6.78 |", status="run ended", typed=["76619", "approve", "continue"],
                               answered=["clarification answered by harness: 76619"])

    monkeypatch.setattr(harness, "ask", fake_ask)
    report = bench.run_subquestions([SUB], cache_copy=tmp_path, out_dir=tmp_path, base_config=lambda q: None, make_models=lambda: None,
                                    triage_model=None, preflight=None, allow_live=False)
    [row] = report["subquestions"]
    assert seen["answer"] == "76619" and row["answered_clarifications"] == ["clarification answered by harness: 76619"]
    markdown = bench.results_markdown({**report, "started_utc": "now", "models": "m", "cache_copy": "c"})
    assert "| Sub | Tier | Bucket | Why | Got | Answered clarifications | Model calls | Cost ($) | Seconds | Status |" in markdown
    assert "| X1.1 | T1 | " in markdown and "| 76619 | 0 | " in markdown


def test_a_row_without_an_answered_clarification_shows_an_empty_cell(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, "ask", lambda question, chat, *, wait=120.0, answer_clarification=None:
                        harness.Outcome(shown="**Summary**\n\n| Mean | 6.78 |", status="run ended"))
    report = bench.run_subquestions([SUB], cache_copy=tmp_path, out_dir=tmp_path, base_config=lambda q: None, make_models=lambda: None,
                                    triage_model=None, preflight=None, allow_live=False)
    [row] = report["subquestions"]
    assert row["answered_clarifications"] == []
    line = next(l for l in bench.results_markdown({**report, "started_utc": "now", "models": "m", "cache_copy": "c"}).splitlines()
                if l.startswith("| X1.1 |"))
    assert "|  | 0 | " in line                                          # the empty cell, then the model calls
