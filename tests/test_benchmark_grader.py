"""The benchmark grader (eval/benchmark.py; coach Task 1, 2026-10-21): code-based, like the harness. Only the code-written
part of the last chat message counts (model notes are left out). A sub-question passes only if every expected value is
found within its tolerance. Buckets: pass / wrong / no_number (an honest "cannot compute" or a question back) /
blocked (a gate, a limit, an error) / invented (a number in the code-written part that no claim, run text, table or the
question backs: the red flag)."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

BENCHMARK = Path(__file__).resolve().parents[1] / "eval" / "benchmark.py"
_spec = importlib.util.spec_from_file_location("_eval_benchmark_under_test", BENCHMARK)
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench
_spec.loader.exec_module(bench)


def sub(**over):
    base = {"sub_id": "A35.1", "parent": "A35", "key_facts": [0], "tier": "T1",
            "question": "How many plots does PEP2016CSR-ST03 (studyDbId 4610) have?",
            "expected": [{"value": 960, "kind": "count", "decimals": 0}], "expected_text": [],
            "tolerance": {"rule": "exact"}, "alternatives": [], "studies": [{"studyDbId": "4610", "name": "PEP2016CSR-ST03"}],
            "folders": [], "variables": [], "statistic": "number of plots", "caveat": None, "gold_basis": "answer"}
    return {**base, **over}


def claim(value, *, n=None, units=None, transformation="row count of a complete table", filters=None):
    return {"kind": "count", "value": value, "denominator": {"name": "rows", "n": n if n is not None else value},
            "n_independent_units": units, "exclusions": [], "filters": filters or {}, "grouping": [],
            "transformation": transformation}


ANSWER = "**Counts**\n\n- **960** rows in the complete observation\\_units table\n\nFull report contains the complete methods."
MEAN = ("**Summary**\n\n| Statistic | Result |\n| --- | --- |\n| Mean | 12.3412345 t/ha |\n| Count | 840 |\n\n"
        "Based on valid numeric values, one per observation unit: 840. Recorded analysis units: 840.")
MEAN_CLAIMS = [{"kind": "mean", "value": 12.3412345, "denominator": {"name": "valid numeric values", "n": 840},
                "n_independent_units": 840, "exclusions": [], "filters": {}, "grouping": [], "transformation": "mean of valid values"},
               claim(840, transformation="count of valid values; as recorded")]


def test_a_count_shown_and_backed_by_its_claim_passes():
    g = bench.grade(sub(), shown=ANSWER, status="final review reached", claims=[claim(960)])
    assert g["bucket"] == "pass" and g["invented"] == []


@pytest.mark.parametrize("expected, tolerance, bucket", [
    ({"value": 12.34, "kind": "mean", "decimals": 2}, {"rule": "display", "abs": 0.005}, "pass"),      # 12.3412 is 12.34
    ({"value": 12.36, "kind": "mean", "decimals": 2}, {"rule": "display", "abs": 0.005}, "wrong"),
    ({"value": 12.40, "kind": "mean", "decimals": 2}, {"rule": "relative", "rel": 0.005}, "pass"),     # within 0.5 %
    ({"value": 12.50, "kind": "mean", "decimals": 2}, {"rule": "relative", "rel": 0.005}, "wrong"),
], ids=["display-pass", "display-wrong", "relative-pass", "relative-wrong"])
def test_tolerance_rules(expected, tolerance, bucket):
    g = bench.grade(sub(expected=[expected], tolerance=tolerance), shown=MEAN, status="final review reached", claims=MEAN_CLAIMS)
    assert g["bucket"] == bucket


def test_every_expected_value_and_name_must_be_found():
    two = sub(expected=[{"value": 960, "kind": "count", "decimals": 0}, {"value": 3, "kind": "count", "decimals": 0}])
    assert bench.grade(two, shown=ANSWER, status="run ended", claims=[claim(960)])["bucket"] == "wrong"   # 3 is missing
    named = sub(expected_text=["Dagga"])
    assert bench.grade(named, shown=ANSWER, status="run ended", claims=[claim(960)])["bucket"] == "wrong"
    shown = ANSWER + "\n\n| ID | Germplasm |\n| --- | --- |\n| 1 | Dagga |"
    assert bench.grade(named, shown=shown, status="run ended", claims=[claim(960)], backed_extra=[1])["bucket"] == "pass"


def test_an_alternative_from_the_grading_notes_passes():
    alt = sub(expected=[{"value": 961, "kind": "count", "decimals": 0}],
              alternatives=[{"expected": [{"value": 960, "kind": "count", "decimals": 0}], "expected_text": []}])
    g = bench.grade(alt, shown=ANSWER, status="run ended", claims=[claim(960)])
    assert g["bucket"] == "pass" and "alternative" in g["why"]


def test_a_number_only_in_the_model_notes_neither_passes_nor_counts_as_invented():
    shown = ("**Counts**\n\n- **5** rows in the complete studies table\n\n**Additional model notes (unverified)**\n\n"
             "- The trial has 960 plots and 17 blocks.")
    g = bench.grade(sub(), shown=shown, status="run ended", claims=[claim(5)])
    assert g["bucket"] == "wrong" and g["invented"] == []


def test_a_code_written_number_that_nothing_backs_is_invented():
    shown = ANSWER + "\n\nThe trial has 17 blocks."
    g = bench.grade(sub(), shown=shown, status="run ended", claims=[claim(960)])
    assert g["bucket"] == "invented" and [i["value"] for i in g["invented"]] == [17]
    assert "17 blocks" in g["invented"][0]["context"]


def test_a_rounded_claim_value_and_the_questions_own_numbers_are_backed():
    shown = "Mean: 2.38 (n = 140)\n\nStudy 4610, season 2016."
    claims = [{"kind": "mean", "value": 2.37857142857, "denominator": {"name": "values", "n": 140}, "n_independent_units": 140,
               "exclusions": [], "filters": {}, "grouping": [], "transformation": "mean of valid values"}]
    g = bench.grade(sub(expected=[{"value": 2.38, "kind": "mean", "decimals": 2}], tolerance={"rule": "display", "abs": 0.005},
                        question="Mean in PEP2016CSR-ST03 (studyDbId 4610), 2016?"),
                    shown=shown, status="run ended", claims=claims)
    assert g["bucket"] == "pass" and g["invented"] == []


def test_the_apps_own_structural_numbers_are_not_invented():
    shown = (ANSWER + "\n\n**Traits: 1 groups · source 2**\n\nShowing 30 of 36 entries; 6 additional IDs are not displayed here."
             "\n\nI need one more detail (answer 1 of 3):")
    g = bench.grade(sub(), shown=shown, status="run ended", claims=[claim(960)])
    assert g["invented"] == []


@pytest.mark.parametrize("shown, status", [
    ("**Question blocked.**\n\nThe plan needs 8 study plot rosters.", "run ended"),
    ("**A request limit was reached.**\n\nThe shared time budget was exhausted.", "run ended"),
    ("**Could not complete this question.**\n\nthe model declared completion but no tool produced a claim", "run ended"),
    ("Approve fetching the study and trait catalogs?", "stopped: the catalog setup needs an approval (the harness never approves it)"),
    ("Some screen", "stopped at a step the harness does not answer"),
    ("", "error: RuntimeError: boom"),
], ids=["blocked", "limit", "error", "catalog-gate", "unknown-step", "harness-error"])
def test_gates_limits_and_errors_are_blocked(shown, status):
    # a run's own notes back the numbers of its stop message ("8 study plot rosters"): run_numbers reads them
    assert bench.grade(sub(), shown=shown, status=status, claims=[], backed_extra=[8])["bucket"] == "blocked"


def test_a_stop_message_with_an_unbacked_number_is_still_the_red_flag():
    """The invented check comes first: a number that nothing backs is reported even in a stop message."""
    g = bench.grade(sub(), shown="**Question blocked.**\n\nThe plan needs 8 study plot rosters.", status="run ended", claims=[])
    assert g["bucket"] == "invented" and [i["value"] for i in g["invented"]] == [8]


@pytest.mark.parametrize("shown, status", [
    ("**More detail is needed.**\n\nWhich trial do you mean?", "run ended"),
    ("I need one more detail before I can make a plan (answer 1 of 3):\n\n> Which study?",
     "stopped: the app asked a question (the harness never answers one)"),
    ("Do you want me to look this up?", "no run: the app asked whether to look it up (the harness never says yes)"),
    ("**From the saved catalog** (no records were looked up):\n\n- It has many studies.", "no run: answered by code from the saved catalog"),
], ids=["more-detail", "planner-question", "lookup-offer", "catalog-summary"])
def test_an_honest_answer_without_the_number_is_no_number(shown, status):
    assert bench.grade(sub(), shown=shown, status=status, claims=[], run=False)["bucket"] == "no_number"


def test_load_subquestions_checks_the_shape(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"subquestions": [sub()]}), encoding="utf-8")
    assert [s["sub_id"] for s in bench.load_subquestions(good)] == ["A35.1"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"subquestions": [{k: v for k, v in sub().items() if k != "tolerance"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="tolerance"):
        bench.load_subquestions(bad)
    tier = tmp_path / "tier.json"
    tier.write_text(json.dumps({"subquestions": [sub(tier="T9")]}), encoding="utf-8")
    with pytest.raises(ValueError, match="tier"):
        bench.load_subquestions(tier)


# --------------------------------------------------------------------------
# Review 10 (2026-10-21): each defect it found, pinned
# --------------------------------------------------------------------------

def test_a_catalog_list_written_by_code_is_not_the_red_flag():
    """D1: 27 of 58 real run answers (catalog lists) were flagged INVENTED: their per-item counts come from the saved
    catalog by code, not from claims about measurements."""
    shown = "**Seasons in the catalog: 24.** Newest first:\n\n- 2026 · 10 studies\n- 2025 · 62 studies\n- 2024 · 137 studies"
    two = sub(expected=[{"value": 62, "kind": "count", "decimals": 0}])
    g = bench.grade(two, shown=shown, status="run ended", claims=[claim(24)], code_catalog=True)
    assert g["bucket"] == "pass" and g["invented"] == []


def test_ontology_terms_with_or_without_an_underscore_are_no_numbers():
    shown = ANSWER + "\n\nVariables: Root yield|COMP:0000107; Total storage roots yield|CO_331:0000296."
    assert bench.grade(sub(), shown=shown, status="run ended", claims=[claim(960)])["invented"] == []


@pytest.mark.parametrize("reply, bucket", [
    ("PEP2016CSR-ST03 is a 2016 trial with 960 plots.", "invented"),     # 960 is not in the question
    ("I can look up study 4610 for you.", "no_number"),                  # only the question's own number: never scored
])
def test_a_reply_in_the_models_words_never_passes_and_holds_only_the_questions_numbers(reply, bucket):
    """D2: such a reply passed with "960 plots"; it was never checked for invention."""
    g = bench.grade(sub(question="How many plots does study 4610 have in 2016?"), shown=reply,
                    status="no run: the app answered without starting one", claims=[], run=False)
    assert g["bucket"] == bucket
    assert [i["value"] for i in g["invented"]] == ([960] if bucket == "invented" else [])


def test_a_distinct_count_in_the_catalog_header_counts_for_the_pass():
    """D3: the header "Studies: 18 distinct IDs" was blanked before the pass check, so the right count failed."""
    label = 'rows matching seasons contains "2026" in the complete studies table'      # the claim's own denominator name
    distinct = {**claim(18, transformation="number of distinct non-empty exact text values"), "denominator": {"name": label, "n": 40}}
    shown = f"**Studies: 18 distinct IDs**\n\nBased on {label}: 40."
    g = bench.grade(sub(expected=[{"value": 18, "kind": "count", "decimals": 0}]), shown=shown, status="run ended", claims=[distinct])
    assert g["bucket"] == "pass" and g["invented"] == []


def test_a_number_in_scientific_notation_is_read_whole():
    """D6: the presenter writes small values as 1.5e-05; it was read as 1 and 5."""
    mean = [{"kind": "mean", "value": 1.5e-05, "denominator": {"name": "values", "n": 12}, "n_independent_units": 12,
             "exclusions": [], "filters": {}, "grouping": [], "transformation": "mean of valid values"}]
    g = bench.grade(sub(expected=[{"value": 1.5e-05, "kind": "mean", "decimals": 6}], tolerance={"rule": "relative", "rel": 0.005}),
                    shown="| Mean | 1.5e-05 |\n\nBased on values: 12.", status="run ended", claims=mean)
    assert g["bucket"] == "pass" and g["invented"] == []


def test_the_apps_quoted_questions_are_left_out_like_model_notes():
    """D6: a clarification screen quotes the model's question; its numbers were flagged INVENTED."""
    shown = "I need one more detail before I can make a plan (answer 1 of 3):\n\n> Which of the 2 studies, 4610 or 4612?"
    g = bench.grade(sub(), shown=shown, status="stopped: the app asked a question (the harness never answers one)", claims=[])
    assert g["bucket"] == "no_number" and g["invented"] == []


def test_a_pass_records_where_each_value_was_found():
    """D4: a pass is any number within tolerance anywhere; the result keeps where, so a reader can check it."""
    g = bench.grade(sub(), shown=ANSWER, status="final review reached", claims=[claim(960)])
    assert g["bucket"] == "pass" and len(g["matched"]) == 1 and "960" in g["matched"][0]


def test_an_ending_the_harness_never_scores_is_no_number():
    g = bench.grade(sub(), shown="Something else.", status="no run: something the harness does not know", claims=[], run=False)
    assert g["bucket"] == "no_number"


@pytest.mark.parametrize("over, message", [
    ({"expected": [], "expected_text": []}, "nothing to check"),
    ({"tolerance": {"rule": "display"}}, "abs"),
    ({"alternatives": [{"expected": [], "expected_text": []}]}, "alternative"),
], ids=["empty", "display-without-abs", "empty-alternative"])
def test_the_loader_refuses_specs_that_would_grade_wrongly(tmp_path, over, message):
    """D4: an empty expected list passed anything; "display" without abs silently became exact."""
    path = tmp_path / "subs.json"
    path.write_text(json.dumps({"subquestions": [sub(**over)]}), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        bench.load_subquestions(path)


def test_a_sub_question_marked_not_checkable_may_have_nothing_to_check(tmp_path):
    path = tmp_path / "subs.json"
    path.write_text(json.dumps({"subquestions": [sub(expected=[], expected_text=[], checkable=False)]}), encoding="utf-8")
    assert bench.load_subquestions(path)[0]["checkable"] is False


# --------------------------------------------------------------------------
# The split's conventions (eval/benchmark/spb_subquestions.json "notes")
# --------------------------------------------------------------------------

def test_each_number_may_carry_its_own_display_tolerance_and_counts_stay_exact():
    two = sub(expected=[{"value": 12.3, "kind": "mean", "decimals": 1, "abs": 0.05},
                        {"value": 840, "kind": "count", "decimals": 0, "abs": None}],
              tolerance={"rule": "display", "abs": 0.005})
    assert bench.grade(two, shown=MEAN, status="run ended", claims=MEAN_CLAIMS)["bucket"] == "pass"   # 12.341 is 12.3 at 1 decimal
    # a count stays exact even when the sub-question's display tolerance is wide (a mean shown in whole tonnes: abs 5)
    off = sub(expected=[{"value": 841, "kind": "count", "decimals": 0, "abs": None}], tolerance={"rule": "display", "abs": 5})
    assert bench.grade(off, shown=MEAN, status="run ended", claims=MEAN_CLAIMS)["bucket"] == "wrong"


def test_a_name_written_a_or_b_is_found_by_either():
    named = sub(expected_text=["CIP199062.1|Dagga"])
    shown = ANSWER + "\n\n| ID | Germplasm |\n| --- | --- |\n| 1 | Dagga |"
    assert bench.grade(named, shown=shown, status="run ended", claims=[claim(960)], backed_extra=[1])["bucket"] == "pass"
    missing = sub(expected_text=["CIP199062.1|Dagga"])
    assert bench.grade(missing, shown=ANSWER, status="run ended", claims=[claim(960)])["bucket"] == "wrong"


# --------------------------------------------------------------------------
# Baseline 2026-10-21: 45 INVENTED flags, all on code-written text (checked one by one)
# --------------------------------------------------------------------------

PLANNER_QUESTION = ("I need one more detail before I can make a plan (answer 1 of 3):\n\n> Which location do you mean?\n\n"
                    "- 12 · Clinton - Horticultural Crops Research · 426 studies\n- 80 · Cunningham Research Station · 184 studies\n\n"
                    "Your answer is added to your question, so a few words are enough, for example **study 4501** or **all studies "
                    "from 2026**. Or reply **cancel**.")


@pytest.mark.parametrize("shown, status", [
    (PLANNER_QUESTION, "stopped: the app asked a question (the harness never answers one)"),
    (PLANNER_QUESTION, "no run: the app asked whether to look it up"),
    ("More detail is needed.\n\nThe plot roster of study 4501 lists 412 units.", "run ended"),
], ids=["planner-question", "lookup-offer", "more-detail"])
def test_the_apps_question_screen_is_not_checked_for_invention(shown, status):
    """44 flags: the planner's clarification screen holds the app's own hint ("for example study 4501 or all studies from
    2026") and location choices with study counts from the saved catalog, written by code. It answers nothing; nor do the
    lookup offer and a run's "More detail is needed." screen (both pinned after the fresh review of 2026-10-21)."""
    g = bench.grade(sub(), shown=shown, status=status, claims=[])
    assert g["bucket"] == "no_number" and g["invented"] == []


def test_a_timestamp_with_fractions_and_a_timezone_is_no_number():
    """1 flag: "Source fetched: 2026-10-21T12:22:33.472516+00:00" was read as 33.472516."""
    shown = ANSWER + "\n\nSource fetched: 2026-10-21T12:22:33.472516+00:00"
    g = bench.grade(sub(), shown=shown, status="final review reached", claims=[claim(960)])
    assert g["bucket"] == "pass" and g["invented"] == []


def _run_dir(root, ledger):
    run = root / "run_x"
    (run / "artifacts").mkdir(parents=True)
    (run / "answer.json").write_text(json.dumps({"notes": [], "retrieval": {"warnings": [], "study_ledger": ledger}}), encoding="utf-8")
    return run


def _refused(study, reason=None):
    """A ledger entry as agents/retriever.py writes one when the approval check refuses a fetch (record_study(..., "blocked"))."""
    return {"study_id": study, "status": "blocked", "artifact_id": None, "reason": reason or f"study {study!r} is not on the FetchApproval",
            "request_ids": []}


REFUSED_LINE = "**A request limit was reached.**\n\nStudy {}: a fetch was refused by the approval check; the Full report has the reason."


def test_a_refused_study_in_the_runs_own_ledger_backs_its_id(tmp_path):
    """Variance run of 2026-10-21 (C33.5): "Study 4777: a fetch was refused by the approval check" was flagged invented. 4777
    is in the run's own study ledger: the retriever asked for it outside the plan, and the approval check refused it."""
    shown = REFUSED_LINE.format(4777)
    ledger = [{"study_id": "4775", "status": "complete", "artifact_id": "art_8e7a85_0001", "reason": None, "request_ids": []}, _refused("4777")]
    g = bench.grade(sub(), shown=shown, status="run ended", claims=[], backed_extra=bench.run_numbers([_run_dir(tmp_path / "a", ledger)]))
    assert g["bucket"] == "blocked" and g["invented"] == []
    other = bench.run_numbers([_run_dir(tmp_path / "b", ledger[:1])])            # a ledger without 4777
    flagged = bench.grade(sub(), shown=shown, status="run ended", claims=[], backed_extra=other)
    assert flagged["bucket"] == "invented" and [i["value"] for i in flagged["invented"]] == [4777]


def test_model_text_in_the_ledger_backs_nothing(tmp_path):
    """Review 2026-10-21: a ledger entry's study_id is the model's own argument and its reason quotes the model's arguments, so
    a number smuggled in through them ("mean=12.5" as a study ID, "mean=3.7" as a variable) must stay the red flag."""
    run = _run_dir(tmp_path, [_refused("mean=12.5"), _refused("4610", reason="variable 'mean=3.7' is not on the FetchApproval")])
    backed = bench.run_numbers([run])
    assert backed == [4610.0]                                                  # only the all-digit study ID
    shown = "**Question blocked.**\n\nThe mean is 12.5 t/ha; the other study gives 3.7."
    g = bench.grade(sub(), shown=shown, status="run ended", claims=[], backed_extra=backed)
    assert g["bucket"] == "invented" and sorted(i["value"] for i in g["invented"]) == [3.7, 12.5]


def _with_report_lines(run, *, warnings=(), caveats=(), notes=()):
    doc = json.loads((run / "answer.json").read_text(encoding="utf-8"))
    doc["retrieval"]["warnings"], doc["analysis"], doc["notes"] = list(warnings), {"caveats": list(caveats)}, list(notes)
    (run / "answer.json").write_text(json.dumps(doc), encoding="utf-8")
    return run


def test_an_id_the_model_only_named_stays_invented(tmp_path):
    """Coach 2026-10-21 (20:50, answer 2): an ID that reaches the run's record only because the model NAMED it is an invented
    ID's path. The study ledger holds only IDs passed to a fetch (fetched, or refused by the approval check), but report lines
    quote IDs the model only named, and until now they backed them: the retriever's "studies named by the model but never
    fetched: [...]" and its siblings, the analyst's unknown claim IDs, a rejected plan draft's study IDs (review 2026-10-22)."""
    run = _with_report_lines(_run_dir(tmp_path, []), warnings=[
        "studies named by the model but never fetched: ['99991']",
        "studies named by the model for a catalog plan (no observation step; nothing to fetch for them): ['99992']",
        "model declared IDs that no tool returned: ['study 99993']",
        "model listed resolved entries of no entity kind, dropped: [\"'studies' '99994'\"]"],
        caveats=["model named claim IDs that no tool produced: ['99995']"],
        notes=["coordinator: attempt 1 rejected: scope.study_ids names IDs that are not in the validated metadata: ['99996']"])
    backed = bench.run_numbers([run])
    assert backed == [1.0]                                                     # the attempt number is the code's own
    shown = "**Incomplete result.**\n\nStudies 99991, 99992, 99993, 99994, 99995 and 99996 were named but never fetched."
    g = bench.grade(sub(), shown=shown, status="run ended", claims=[], backed_extra=backed)
    assert g["bucket"] == "invented" and sorted(i["value"] for i in g["invented"]) == [99991, 99992, 99993, 99994, 99995, 99996]


def test_report_lines_that_quote_the_model_back_nothing_and_code_lines_still_do(tmp_path):
    """The model's notes, its rejected tool arguments, the agents' questions (with any answer: the harness never answers) and
    the supervisor's stop reason are model text too. A line the code wrote still backs its numbers, in warnings, caveats and
    notes alike; a model-quoting phrase inside a code line does not silence it (prefixes, not substrings)."""
    run = _with_report_lines(_run_dir(tmp_path, []),
                             warnings=["clarification needed: Is it study 4612 or 4614?", "model note: Fetched 952 rows.",
                                       "agent error malformed_arguments: 'get_observations': 99993 is not of type 'string'",
                                       "get_observation_units: stopped after 3 page(s)"],
                             caveats=["clarification needed: The artifact has 40 plot records; which 12?", "model: The mean is 12.5.",
                                      "group_stats: Measurement dates are not recorded for 951 of 951 valid source rows"],
                             notes=["coordinator stopped the run: The analysis omits 26 claims.",
                                    "clarification needed but --auto never asks: Which of 2 traits?",
                                    "clarification 1: the planner asked: Which of 2 traits? - answer: 76581",
                                    "retriever clarification 1: asked Is it study 4612?; answer: 4614",
                                    "analyst clarification 1: the analyst asked: Which 13 plots? - answer: all 40",
                                    "coordinator: matching_study_count=7 was counted by code from the scope, not taken from the model",
                                    "coordinator: dropped clarification 'Which 5 studies?': study 4610 is uniquely named and its "
                                    "metadata supplies location and season",
                                    "coordinator: attempt 2 rejected: plan contract rejected the draft at steps: List should have "
                                    "at most 12 items after validation, not 14"])
    assert sorted(set(bench.run_numbers([run]))) == [2.0, 3.0, 7.0, 12.0, 14.0, 951.0, 4610.0]


def test_a_model_quoting_line_backs_nothing_in_any_of_the_runs(tmp_path):
    """Review 2026-10-22: every run folder of a sub-question is read the same way, whichever comes first."""
    named = _with_report_lines(_run_dir(tmp_path / "a", []), warnings=["studies named by the model but never fetched: ['99991']"])
    code = _with_report_lines(_run_dir(tmp_path / "b", []), warnings=["get_observation_units: stopped after 3 page(s)"])
    assert bench.run_numbers([named, code]) == bench.run_numbers([code, named]) == [3.0]


def test_the_lines_the_app_writes_back_nothing(tmp_path):
    """Review 2026-10-22: the prefixes are pinned against the lines the app itself writes, so a reworded warning fails here
    instead of quietly backing a model-named ID again."""
    from types import SimpleNamespace

    import agents.analyst as an
    import test_retriever as tr
    from contracts import AgentResult
    from llm import Budget, FakeModelClient, reply_text, reply_tools

    def payload_model(payload):
        return FakeModelClient([reply_tools([("get_observation_units", json.dumps({"study_db_id": "S1"}))]),
                                reply_text(json.dumps({"status": "completed", "payload": payload}))], reported_model="mock-model")

    lines = []
    for n, payload in enumerate(({"resolved": [], "studies": ["S1", "99991"], "notes": []},                    # named, never fetched
                                 {"resolved": [{"kind": "study", "id": "99993"}], "studies": ["S1"], "notes": []},   # never returned
                                 {"resolved": [{"kind": "studies", "id": "99994"}], "studies": ["S1"], "notes": []})):  # no such kind
        report, *_ = tr.run_script(tmp_path / f"r{n}", "valid_request.json", model=payload_model(payload), request="How many plots?")
        lines += report.warnings
    malformed = FakeModelClient([reply_tools([("get_observations", json.dumps({"study_db_id": 99995, "variable_db_id": "V1"}))])],
                                reported_model="mock-model")
    report, *_ = tr.run_script(tmp_path / "r3", "valid_request.json", model=malformed, budget=Budget(max_repairs=0))
    lines += report.warnings
    agent = AgentResult(status="completed", payload={"methods": [], "caveats": [], "narrative": "n", "key_claim_ids": ["99992"]},
                        model_requested="m", model_calls=1, tool_calls=0, elapsed_seconds=0.0)
    caveats = an.build_report(agent, SimpleNamespace(records=[], claim_refs={}), SimpleNamespace(artifact_ids=[]))[0].caveats
    for value in ("99991", "99993", "99994", "99995"):
        assert any(value in line for line in lines), value                    # each was written by the app's own code
    assert any("99992" in line for line in caveats)
    run = _with_report_lines(_run_dir(tmp_path / "graded", []), warnings=lines, caveats=caveats)
    assert not {99991.0, 99992.0, 99993.0, 99994.0, 99995.0} & set(bench.run_numbers([run]))


def test_an_endless_number_backs_nothing(tmp_path):
    """Review 2026-10-22 (round 2, R4): an all-digit ID of 309 digits or more reads as inf, and inf backed every number shown
    (abs(inf - x) <= inf). A model may put such an ID in a refused fetch, and the ledger then holds it; a table cell could too."""
    run = _run_dir(tmp_path, [_refused("9" * 400)])
    with (run / "artifacts" / "t.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("value\n" + "8" * 400 + "\n-" + "7" * 400 + "\n")                         # +inf and -inf
    backed = bench.run_numbers([run])
    assert backed and not any(math.isfinite(value) for value in backed)                  # read, and endless
    shown = "**Question blocked.**\n\nThe mean is 12.5 t/ha; the other study gives 3.7."
    g = bench.grade(sub(), shown=shown, status="run ended", claims=[], backed_extra=backed)
    assert g["bucket"] == "invented" and sorted(i["value"] for i in g["invented"]) == [3.7, 12.5]
    assert not bench._backed(12.5, 1, [math.inf]) and bench._backed(12.5, 1, [math.inf, 12.5])
