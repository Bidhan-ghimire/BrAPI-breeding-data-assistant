"""The benchmark grader (eval/benchmark.py; coach Task 1, 2026-10-21): code-based, like the harness. Only the code-written
part of the last chat message counts (model notes are left out). A sub-question passes only if every expected value is
found within its tolerance. Buckets: pass / wrong / no_number (an honest "cannot compute" or a question back) /
blocked (a gate, a limit, an error) / invented (a number in the code-written part that no claim, run text, table or the
question backs: the red flag)."""
import importlib.util
import json
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
