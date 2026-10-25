"""Decision (31)(a), rule (3) (Bidhan, coach 2026-10-25 12:20 UTC, with the coach's 02:52 lean): a number in the model's notes (the
"Additional model notes (unverified)" bullets, the "Unverified model note:" paragraphs) gets the backing check a number of the
checked reply gets (the claims' numbers, with their filters and texts; the question's; the run's own tables and code-written
fields), and the checked reply's own numbers back it too; it is read at the precision it is written with; notes never count
toward the pass match. Real-looking inputs from the Sonnet runs of 24-25 Oct (B26.4, C30.1, A38.2)."""
import importlib.util
import sys
from pathlib import Path

from test_note_numbers import B264, B264_ANSWER, B264_CELLS, B264_CLAIMS

BENCHMARK = Path(__file__).resolve().parents[1] / "eval" / "benchmark.py"
_spec = importlib.util.spec_from_file_location("_eval_benchmark_note_grading_under_test", BENCHMARK)
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench
_spec.loader.exec_module(bench)

NOTES = "\n\n**Additional model notes (unverified)**\n\n"
PLOTS = {"sub_id": "A35.1", "parent": "A35", "tier": "T1", "question": "How many plots does PEP2016CSR-ST03 (studyDbId 4610) have?",
         "expected": [{"value": 960, "kind": "count", "decimals": 0}], "expected_text": [], "tolerance": {"rule": "exact"},
         "alternatives": []}
PLOTS_ANSWER = ("**Counts**\n\n- **960** rows in the complete observation\\_units table\n\nShowing 30 of 960 entries; 930 additional "
                "IDs are not displayed here.")
PLOTS_CLAIMS = [{"kind": "count", "value": 960, "denominator": {"name": "rows", "n": 960}, "n_independent_units": None,
                 "exclusions": [], "filters": {}, "grouping": [], "transformation": "row count of a complete table"}]


def noted(g):
    return [(i["value"], i.get("in")) for i in g["invented"]]


def test_b264_the_40_is_a_real_block_code_so_the_grade_stays_wrong():
    """B26.4 (Sonnet, 24 and 25 Oct): "40 blocks per replicate" is wrong, but 40 is a block code in a claim's filter and in the
    plot table, so rule (3) backs it; the row stays wrong (the design word "alpha" is missing), not invented."""
    g = bench.grade(B264, shown=B264_ANSWER, status="run ended", claims=B264_CLAIMS, backed_extra=B264_CELLS)
    assert g["bucket"] == "wrong" and g["invented"] == []
    # with no table cell 40, the claim's filter {"block": "40"} backs it alone, as it would back a 40 in the checked reply
    g = bench.grade(B264, shown=B264_ANSWER, status="run ended", claims=B264_CLAIMS, backed_extra=[c for c in B264_CELLS if c != 40.0])
    assert g["bucket"] == "wrong" and g["invented"] == []


def test_a_note_number_nothing_backs_makes_the_row_invented():
    """The same answer with block 40 in no claim and no table cell: the note's 40 is backed by nothing, so the row is invented,
    and the reason says it is in the model's notes."""
    claims = [c for c in B264_CLAIMS if c["filters"].get("block") != "40"]
    cells = [c for c in B264_CELLS if c != 40.0]
    g = bench.grade(B264, shown=B264_ANSWER, status="run ended", claims=claims, backed_extra=cells)
    assert g["bucket"] == "invented" and noted(g) == [(40, "notes")]
    assert g["why"] == "1 number(s) in the model's notes that nothing backs" and "40 blocks per replicate" in g["invented"][0]["context"]


def test_a_number_the_checked_reply_shows_backs_a_note_number():
    """930 is shown in the checked reply ("930 additional IDs", the app's own phrase, which the invented check does not read);
    no claim and no table holds it, and it backs the note's 930; a 931 is backed by nothing."""
    g = bench.grade(PLOTS, shown=PLOTS_ANSWER + NOTES + "- The other 930 rows are not displayed.", status="run ended", claims=PLOTS_CLAIMS)
    assert g["bucket"] == "pass" and g["invented"] == []
    g = bench.grade(PLOTS, shown=PLOTS_ANSWER + NOTES + "- The other 931 rows are not displayed.", status="run ended", claims=PLOTS_CLAIMS)
    assert g["bucket"] == "invented" and noted(g) == [(931, "notes")]


def test_the_questions_numbers_back_a_note_number():
    g = bench.grade(PLOTS, shown=PLOTS_ANSWER + NOTES + "- Study 4610 is the only study read.", status="run ended", claims=PLOTS_CLAIMS)
    assert g["bucket"] == "pass" and g["invented"] == []


def test_c301_a_note_number_is_read_at_the_precision_it_is_written_with():
    """C30.1 (Sonnet, 24 Oct): "p about 5.6e-7" against the claim's 5.603e-07 is a match at two figures, as written (the
    checked reply's numbers are read at four, as the app writes them); 5.8e-7 is not."""
    sub = {**PLOTS, "expected": [{"value": 19, "kind": "count", "decimals": 0}], "question": "Compare the 90-day and 120-day trials."}
    shown = "| Paired t | 7.3644 | 19 | 5.603e-07 |\n\nWelch's p 0.0664."
    claims = [{"kind": "test", "value": v, "filters": {}, "transformation": t}
              for v, t in ((7.3644, "paired t"), (19, "paired t: df"), (5.603e-07, "paired t: p"), (0.0664, "Welch t: p"))]
    note = NOTES + "- The paired t on entries is 7.36 (df 19, p about {p}), while Welch's p is 0.066."
    g = bench.grade(sub, shown=shown + note.format(p="5.6e-7"), status="run ended", claims=claims)
    assert g["bucket"] == "pass" and g["invented"] == []
    g = bench.grade(sub, shown=shown + note.format(p="5.8e-7"), status="run ended", claims=claims)
    assert g["bucket"] == "invented" and noted(g) == [(5.8e-07, "notes")]
    assert [(v, d) for v, d, *_ in bench._numbers("5.6e-7", written=True)] == [(5.6e-07, 8)]
    assert [(v, d) for v, d, *_ in bench._numbers("5.6e-7")] == [(5.6e-07, 10)]        # the checked reply's reading, unchanged


def test_notes_never_count_toward_the_pass_match():
    """A38.2 (Sonnet, 24 Oct): 3292 in the checked reply, "each with 1 plot" only in the notes: the 1 is backed (a claim counts
    one plot), and the row is still not a pass."""
    sub = {**PLOTS, "sub_id": "A38.2", "question": "In PEP2017CAN-OT01 (studyDbId 4354), how many offspring clones (germplasm names "
           "starting with 'PH17.') are there, and how many plots does each offspring clone occupy?",
           "expected": [{"value": 3292, "kind": "count", "decimals": 0}, {"value": 1, "kind": "count", "decimals": 0}]}
    shown = ("**Germplasm: 3292 distinct IDs**\n\nBased on rows matching germplasmName contains 'PH17.' in the complete "
             "observation\\_units table: 3292." + NOTES + "- The preview shows only the first five clones, each with 1 plot.")
    claims = [{"kind": "count", "value": 3292, "filters": {}, "transformation": "row count"},
              {"kind": "count", "value": 1, "filters": {"germplasmDbId": "397805"}, "transformation": "frequency of exact text values"}]
    g = bench.grade(sub, shown=shown, status="run ended", claims=claims)
    assert g["bucket"] == "wrong" and g["invented"] == []


def test_the_question_screens_and_catalog_runs_stay_unchecked():
    """As for the checked reply: the app's question screens and the code's catalog answers are not checked for invention."""
    note = NOTES + "- 17 blocks."
    asked = bench.grade(PLOTS, shown="Which trait did you mean?" + note, status="stopped: the app asked a question", claims=[])
    catalog = bench.grade(PLOTS, shown=PLOTS_ANSWER + note, status="run ended", claims=[], code_catalog=True)
    assert asked["bucket"] == "no_number" and asked["invented"] == [] and catalog["invented"] == []


def test_a_code_like_scientific_token_is_no_number_and_stops_nothing():
    """Review 2026-10-25 S2, now that notes are graded: "0E0326" (a plot code) overflowed _backed's precision and an exponent of
    5,000 digits could not be read; both are no number, in the checked text and in the notes alike."""
    assert bench._numbers("the plot code 0E0326 was not found") == [] and bench._numbers("0e" + "9" * 5000) == []
    assert bench._numbers("0e400 and 0e301") == [] and [v for v, *_ in bench._numbers("1.5e300")] == [1.5e300]
    shown = (PLOTS_ANSWER + "\n\nPlot code 0E0326." + NOTES + "- the plot code 0E0326 was not found; nor 0e400; nor 0e"
             + "9" * 5000 + ".")
    g = bench.grade(PLOTS, shown=shown, status="run ended", claims=PLOTS_CLAIMS)
    assert g["bucket"] == "pass" and g["invented"] == []


def test_the_apps_own_phrases_are_read_in_a_note():
    """"81 groups" is the app's phrase in the checked reply (not checked there); in a note it is the model's own count, read and
    checked."""
    g = bench.grade(PLOTS, shown=PLOTS_ANSWER + NOTES + "- The table has 81 groups.", status="run ended", claims=PLOTS_CLAIMS)
    assert g["bucket"] == "invented" and noted(g) == [(81, "notes")]


def test_review_of_31a_a_sentence_end_a_sign_a_bare_mantissa_and_a_note_paragraph():
    """Review 2026-10-25 of (31)(a): "5.6e-7." at a sentence's end is one number (the period ends the sentence); "-2.5" is not
    backed by 2.5; "6e-7" is matched at its own precision (half of 1e-7: 5.603e-07 rounds to it, and not to "5e-7"); the
    "Unverified model note:" paragraphs are graded too;
    a 3-digit negative exponent ("6.8e-101", a p-value) is read, a positive one past 300 is not."""
    sub = {**PLOTS, "expected": [{"value": 19, "kind": "count", "decimals": 0}], "question": "Compare the two trials."}
    shown = "| Paired t | 7.3644 | 19 | 5.603e-07 |"
    claims = [{"kind": "test", "value": v, "filters": {}, "transformation": t}
              for v, t in ((7.3644, "paired t"), (19, "paired t: df"), (5.603e-07, "paired t: p"), (2.5, "mean difference"))]
    for note, flagged in (("- The p-value is about 5.6e-7.", []), ("- The change is -2.5 t/ha.", [(-2.5, "notes")]),
                          ("- p is about 6e-7, and 19 df.", []), ("- p is about 5e-7.", [(5e-07, "notes")])):
        assert noted(bench.grade(sub, shown=shown + NOTES + note, status="run ended", claims=claims)) == flagged, note
    assert [(v, d) for v, d, *_ in bench._numbers("about 5.6e-7.")] == [(5.6e-07, 10)]
    assert [v for v, *_ in bench._numbers("p 6.8e-101 and 1.234e-301")] == [6.8e-101, 1.234e-301] and bench._numbers("1e305") == []
    paragraph = bench.grade(PLOTS, shown=PLOTS_ANSWER + "\n\nUnverified model note: 18 rows are checks.", status="run ended",
                            claims=PLOTS_CLAIMS)
    assert noted(paragraph) == [(18, "notes")] and type(paragraph["invented"][0]["value"]) is int


def test_the_persons_quoted_words_back_no_note_number():
    """A quoted line ("> ...", the person's own earlier words) is left out of the checked reply, so it backs no note number."""
    shown = PLOTS_ANSWER + "\n\n> In 45 plots, please." + NOTES + "- 45 plots."
    assert noted(bench.grade(PLOTS, shown=shown, status="run ended", claims=PLOTS_CLAIMS)) == [(45, "notes")]


def test_checked_and_note_numbers_nothing_backs_are_both_reported():
    shown = PLOTS_ANSWER + "\n\nThe trial has 17 blocks." + NOTES + "- 18 rows are checks."
    g = bench.grade(PLOTS, shown=shown, status="run ended", claims=PLOTS_CLAIMS)
    assert g["bucket"] == "invented" and noted(g) == [(17, None), (18, "notes")]
    assert g["why"] == "1 number(s) in the code-written answer and 1 in the model's notes that nothing backs"
