"""The benchmark grades no number of a reply code wrote (coach 2026-10-24 12:50 UTC, answer (b)): C29.6 was graded "invented"
for the 2026 of the front door's fixed example question (triage.EXAMPLE_QUESTION), in triage.out_of_scope_line written when
no run started. Narrowest form: only a reply BYTE-IDENTICAL to triage.out_of_scope_line(host), the host the app names
(app._source_name), is "no run: code-written reply"; one character off or one digit added is graded as before. The report
always says both counts. Everything here is SYNTHETIC: fake chat turns, no model, no request."""
import importlib.util
import sys
from pathlib import Path

import pytest

BENCHMARK = Path(__file__).resolve().parents[1] / "eval" / "benchmark.py"
_spec = importlib.util.spec_from_file_location("_eval_benchmark_code_reply_under_test", BENCHMARK)
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench
_spec.loader.exec_module(bench)
harness = bench.harness

import triage  # noqa: E402  (the app's directory is on the path once the benchmark is loaded)
from llm import reply_text  # noqa: E402
from run import Models, RunConfig  # noqa: E402
from test_triage import chat, decision, model_factory, no_runs, talk  # noqa: E402,F401  (the scripted front door)

HOST = "sweetpotatobase.org"
LINE = triage.out_of_scope_line(HOST)
QUESTION = ("The Pearson correlation of progeny means between two trials is 0.67, and the repeatabilities are 0.74 and 0.64. "
            "What is the disattenuated correlation r / sqrt(H2a x H2b)?")
SUB = {"sub_id": "X1.1", "parent": "X1", "key_facts": [0], "tier": "T3", "question": QUESTION,
       "expected": [{"value": 0.97, "kind": "other", "decimals": 2, "abs": 0.005}], "expected_text": [],
       "tolerance": {"rule": "display", "abs": 0.005}, "alternatives": [], "studies": [], "folders": [], "variables": [],
       "statistic": "disattenuated correlation", "caveat": None, "gold_basis": "test"}


def grade(shown, code_replies=(LINE,)):
    return bench.grade(SUB, shown=shown, status=bench.MODEL_REPLY_STATUS, claims=[], run=False, code_replies=code_replies)


def test_the_line_holds_the_example_year_that_was_graded_invented():
    assert "2026" in LINE and LINE.endswith(f"For example: “{triage.EXAMPLE_QUESTION}”")
    old = bench.grade(SUB, shown=LINE, status=bench.MODEL_REPLY_STATUS, claims=[], run=False)     # no code replies given
    assert old["bucket"] == "invented" and [i["value"] for i in old["invented"]] == [2026]


def test_the_byte_identical_line_is_a_code_written_reply_and_none_of_its_numbers_is_graded():
    g = grade(LINE)
    assert g["bucket"] == "no_number" and g["invented"] == [] and g["code_written_reply"] is True
    assert g["why"].startswith("no run: code-written reply")


@pytest.mark.parametrize("shown, flagged", [
    (LINE + " 42", [2026, 42]),                                  # one number added
    (LINE.replace("2026", "2027"), [2027]),                      # one digit off
    (LINE + " ", [2026]),                                        # one character added
    (LINE[:-1], [2026]),                                         # one character missing
    (LINE.replace(HOST, "cassavabase.org"), [2026]),             # another host's line is not this app's reply
    ("Sure: " + LINE, [2026]),                                   # the line inside a longer reply
    ("**" + LINE + "**", [2026]),                               # the same words in markdown: not the bytes code wrote
    ("\n" + LINE, [2026]),                                       # a line break before (review: lstrip survived)
    (LINE + "\n", [2026]),                                       # a line break after (review: rstrip survived)
    (LINE.lower(), [2026]),                                      # other letters (review: casefold survived)
    (LINE + "\n> 42", [2026]),                                   # a quoted line after it (review: comparing without quotes)
])
def test_anything_but_the_exact_line_is_graded_as_before(shown, flagged):
    g = grade(shown)
    assert g["bucket"] == "invented" and [i["value"] for i in g["invented"]] == flagged
    assert not g.get("code_written_reply")


def test_the_exception_holds_only_for_a_no_run_reply():
    g = bench.grade(SUB, shown=LINE, status="run ended", claims=[], run=True, code_replies=(LINE,))
    assert not g.get("code_written_reply") and g["why"] != bench.CODE_REPLY_WHY
    assert [i["value"] for i in g["invented"]] == [2026]                 # a run's answer: nothing backs 2026


def test_the_exact_texts_are_the_apps_own_out_of_scope_line_for_its_host(monkeypatch):
    monkeypatch.setattr(harness.app, "_source_name", lambda: HOST)
    assert bench.code_written_texts() == (LINE,)
    monkeypatch.setattr(harness.app, "_source_name", lambda: "the breeding database")
    assert bench.code_written_texts() == (triage.out_of_scope_line("the breeding database"),)


@pytest.mark.parametrize("shown", ["2", "a", ""])
def test_a_bare_string_given_as_the_texts_is_one_text_not_its_characters(shown):
    """Review NIT: tuple(LINE) is its characters, so a reply "2" would have been exempt."""
    g = bench.grade(SUB, shown=shown, status=bench.MODEL_REPLY_STATUS, claims=[], run=False, code_replies=LINE)
    assert not g.get("code_written_reply")
    assert grade(LINE, code_replies=LINE)["code_written_reply"] is True


def run(monkeypatch, tmp_path, shown):
    monkeypatch.setattr(harness.app, "_source_name", lambda: HOST)
    monkeypatch.setattr(harness, "ask", lambda question, chat, *, wait=120.0, answer_clarification=None:
                        harness.Outcome(shown=shown, status=bench.MODEL_REPLY_STATUS))
    return bench.run_subquestions([SUB], cache_copy=tmp_path, out_dir=tmp_path, base_config=lambda q: None,
                                  make_models=lambda: None, triage_model=None, preflight=None, allow_live=False)


def test_the_runner_grades_the_apps_line_as_code_written_and_the_report_says_both_counts(monkeypatch, tmp_path):
    report = run(monkeypatch, tmp_path, LINE)
    [row] = report["subquestions"]
    assert row["bucket"] == "no_number" and row["code_written_reply"] is True
    assert report["invented"] == [] and report["code_written_replies"] == ["X1.1"]
    assert bench.invented_summary(report) == "none (1 code-written reply, numbers not graded: X1.1)"
    markdown = bench.results_markdown({**report, "started_utc": "now", "models": "m", "cache_copy": "c"})
    assert "- INVENTED (red flag): none (1 code-written reply, numbers not graded: X1.1)" in markdown


def test_the_runner_still_flags_a_line_with_one_number_added(monkeypatch, tmp_path):
    report = run(monkeypatch, tmp_path, LINE + " 42")
    [row] = report["subquestions"]
    assert row["bucket"] == "invented" and report["invented"] == ["X1.1"] and report["code_written_replies"] == []
    assert bench.invented_summary(report) == "X1.1 (0 code-written replies, numbers not graded)"


def test_the_report_says_both_counts_when_there_is_neither():
    assert bench.invented_summary({"invented": [], "code_written_replies": []}) == "none (0 code-written replies, numbers not graded)"
    assert bench.invented_summary({"invented": []}) == "none (0 code-written replies, numbers not graded)"     # an older report
    markdown = bench.results_markdown({"invented": [], "code_written_replies": [], "by_tier": {}, "subquestions": [],
                                       "started_utc": "now", "models": "m", "cache_copy": "c", "requests": "r", "cost_usd": 0.0})
    assert "- INVENTED (red flag): none (0 code-written replies, numbers not graded)" in markdown


def test_a_no_run_reply_in_the_models_words_without_numbers_is_not_counted_as_code_written(monkeypatch, tmp_path):
    report = run(monkeypatch, tmp_path, "I can look up studies and traits for you.")
    [row] = report["subquestions"]
    assert row["bucket"] == "no_number" and not row.get("code_written_reply") and report["code_written_replies"] == []


def test_the_printed_summary_line_says_both_counts(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bench, "run_subquestions", lambda subs, **_kwargs: {"subquestions": [], "by_tier": {}, "invented": [],
                                                                         "code_written_replies": ["X1.1"], "cost_usd": 0.0})
    monkeypatch.setattr(bench, "results_markdown", lambda report: "")          # its own test is above
    cache = tmp_path / "cache"
    cache.mkdir()
    bench.main(["--cache", str(cache), "--results", str(tmp_path / "results"), "--only", "A35.2", "--confirm"],
               base_config=harness.app.live_config, make_models=lambda: None, triage_model=lambda: None)
    assert "pass of 0; invented: none (1 code-written reply, numbers not graded: X1.1) -> " in capsys.readouterr().out
    monkeypatch.setattr(bench, "run_subquestions", lambda subs, **_kwargs: {"subquestions": [], "by_tier": {}, "invented": [],
                                                                         "code_written_replies": [], "cost_usd": 0.0})
    bench.main(["--cache", str(cache), "--results", str(tmp_path / "results2"), "--only", "A35.2", "--confirm"],
               base_config=harness.app.live_config, make_models=lambda: None, triage_model=lambda: None)
    assert "pass of 0; invented: none (0 code-written replies, numbers not graded) -> " in capsys.readouterr().out


# --- the real front door (review SHOULD-FIX 1): harness.ask with a scripted triage model, no run, no request ---------------

def front_door(chat, monkeypatch, *replies):
    monkeypatch.setenv("BRAPI_BASE_URL", f"https://{HOST}/brapi/v2")
    _, _, factory = model_factory(*replies)
    return bench.run_subquestions([SUB], cache_copy=chat.cache, out_dir=chat.out,
                                  base_config=lambda q: RunConfig(question=q, out_dir=chat.out, cache_dir=chat.cache),
                                  make_models=lambda: Models(), triage_model=factory, preflight=lambda q: [], allow_live=False)


def test_the_front_doors_own_out_of_scope_reply_is_the_exempt_text(chat, no_runs, monkeypatch):
    """out_of_scope and no usable reply from the model: the front door writes the line (app.py, triage.front_door_reply)."""
    report = front_door(chat, monkeypatch, reply_text(decision("out_of_scope")), reply_text("not json"), reply_text("not json"))
    [row] = report["subquestions"]
    assert row["answer_full"] == LINE == bench.code_written_texts()[0] and row["status"] == bench.MODEL_REPLY_STATUS
    assert row["bucket"] == "no_number" and row["code_written_reply"] is True and report["code_written_replies"] == ["X1.1"]


def test_the_models_own_out_of_scope_words_are_graded_as_before(chat, no_runs, monkeypatch):
    report = front_door(chat, monkeypatch, reply_text(decision("out_of_scope")),
                        reply_text(talk("out_of_scope", "Sorry, that is outside what I can do.")))
    [row] = report["subquestions"]
    assert row["bucket"] == "no_number" and not row.get("code_written_reply") and report["code_written_replies"] == []


def test_the_greetings_question_guide_holds_2026_too_and_stays_graded(chat, no_runs, monkeypatch):
    """The narrowest form: the guide (app_help) is code-written as well, but it is not the out_of_scope line (review 3)."""
    report = front_door(chat, monkeypatch, reply_text(decision("greeting_or_smalltalk")), reply_text("not json"),
                        reply_text("not json"))
    [row] = report["subquestions"]
    assert row["bucket"] == "invented" and [i["value"] for i in row["invented"]] == [2026] and report["code_written_replies"] == []
