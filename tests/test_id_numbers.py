"""S2, identifiers are not quantities (decision of Bidhan, coach 2026-10-25 14:35 UTC; the rule as named in the builder's 14:32
entry). (1) A sign before a zero-padded digit run is no minus sign: "-00595", the abbreviated ending of the plot name
PEP2016CSR-ST03-00595, reads 595, and 595 still needs backing; a hyphen glued to a letter or digit was never a sign
("MZP2019CHL-ST01-00595" reads 595). A real negative keeps its sign ("-3.2 t/ha", "-0.5", "-12"). (2) The digits inside a name
(a word of capital letters and digits that holds both: 2016 in MZP2016UMB-ST01, 1 in ST01) back a number of that value wherever
the name is a source of the invented-number check: the question, the claims' texts, the run's tables and code-written fields,
and, for a note, the checked reply; they are no numbers of the text that holds them, and they do not make a shown number a claim
number (wrong against no number). (3) "p<0.001" is unchanged: 0.001 stays unbacked unless a source gives it. The exact cases are
the Sonnet rows of 25 Oct: A36.3 (plot endings), E18.2 (study-name years), C30.2 (p<0.001)."""
import importlib.util
import json
import sys
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parents[1] / "eval" / "benchmark.py"
_spec = importlib.util.spec_from_file_location("_eval_benchmark_id_numbers_under_test", BENCHMARK)
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench
_spec.loader.exec_module(bench)

NOTES = "\n\n**Additional model notes (unverified)**\n\n"
A36_LINE = "There are no duplicate plot values and no missing values. The 3 plots are PEP2016CSR-ST03-00287, -00595 and -00858."
A36 = {"sub_id": "A36.3", "parent": "A36", "tier": "T1", "alternatives": [], "expected_text": [],
       "question": "In PEP2016CSR-ST03 (studyDbId 4610; folder '2016-17 BxT', id 4619; Peru-CIP, San Ramon 2016), what is the mean "
                   "total storage root yield of the check Dagga (CIP199062.1), variable 'Total storage roots yield computation tons "
                   "per ha|CO_331:0000296' (arithmetic mean of its 3 plot values)?",
       "expected": [{"value": 26.32, "kind": "mean", "decimals": 2, "abs": 0.005}], "tolerance": {"rule": "display", "abs": 0.005}}
A36_ANSWER = "**Summary**\n\n| Statistic | Result |\n| --- | --- |\n| Mean | 26.31851852 (unit not stated) |\n| Count | 3 |" + NOTES + "- " + A36_LINE
A36_CLAIMS = [{"kind": "mean", "value": 26.31851852, "denominator": {"name": "valid numeric values", "n": 3}, "filters": {},
               "grouping": [], "transformation": "mean of valid values"},
              {"kind": "count", "value": 3, "denominator": {"name": "valid numeric values", "n": 3}, "filters": {}, "grouping": [],
               "transformation": "count of valid values"}]
A36_TABLE = ("observationUnitName,germplasmName,value\nPEP2016CSR-ST03-00287,CIP199062.1,20.11111111\n"
             "PEP2016CSR-ST03-00595,CIP199062.1,27.65555556\nPEP2016CSR-ST03-00858,CIP199062.1,31.18888889\n")
E18_LINE = ("Plot IDs and names differ between the studies (for example, 00001 is Jonathan in 2016 and Esther in 2017). No replicate "
            "or block is recorded.")
E18 = {"sub_id": "E18.2", "parent": "E18", "tier": "T2", "alternatives": [], "expected_text": [],
       "question": "Compare MZP2017UMB-ST01 (studyDbId 4788) with MZP2016UMB-ST01 (studyDbId 4777): for how many of the 30 genotypes "
                   "present in both are the genotype means identical, and how many of the 48 beta-carotene plot values are identical?",
       "expected": [{"value": 18, "kind": "count", "decimals": 0, "abs": None}, {"value": 33, "kind": "count", "decimals": 0, "abs": None}],
       "tolerance": {"rule": "exact"}}
E18_ANSWER = "**Counts**\n\n- **18** entries with identical means (difference 0.0)" + NOTES + "- " + E18_LINE
E18_CLAIMS = [{"kind": "count", "value": 18, "denominator": {"name": "rows in the complete analysis_result table", "n": 33},
               "filters": {"difference": "0.0"}, "grouping": ["difference"], "transformation": "frequency of exact text values"}]
C30_LINE = "This is a descriptive comparison, not adjusted for trial design. The paired t is 6.08 (p<0.001, 19 df)."
C30 = {"sub_id": "C30.2", "parent": "C30", "tier": "T2", "alternatives": [], "expected_text": [],
       "question": "In folder '2016-19 Genetic Gain' (4631), Huaral 2016: PEP2016HUA-ST01 (studyDbId 4650, 90-day harvest) and "
                   "PEP2016HUA-ST02 (studyDbId 4652, 120-day harvest) hold the same 20 entries x 2 reps. What are the trial means?",
       "expected": [{"value": 4.24, "kind": "mean", "decimals": 2, "abs": 0.005}], "tolerance": {"rule": "display", "abs": 0.005}}
C30_ANSWER = ("**Paired comparison**\n\n| Trial | Study | Mean |\n| --- | --- | --- |\n| Base | 4650 | 4.2428 |\n\n| Test | t | df | p |\n"
              "| --- | --- | --- | --- |\n| Paired t | 6.0773 | 19 | 7.624e-06 |" + NOTES + "- " + C30_LINE)
C30_CLAIMS = [{"kind": "mean", "value": 4.242777777525, "filters": {}, "transformation": "mean of the base trial's entry means"},
              {"kind": "test", "value": 6.077262934884576, "filters": {}, "transformation": "paired t"},
              {"kind": "test", "value": 7.623845808868657e-06, "filters": {}, "transformation": "paired t: p"},
              {"kind": "test", "value": 19, "filters": {}, "transformation": "paired t: df"}]


def values(text, written=False):
    return [v for v, *_ in bench._numbers(text, written=written)]


def noted(g):
    return [(i["value"], i.get("in")) for i in g["invented"]]


def table_dir(tmp_path, text):
    run_dir = tmp_path / "run_20261025T013700_abcdef"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "artifacts" / "0001_observations.csv").write_text(text, encoding="utf-8")
    return run_dir


def test_a_zero_padded_ending_is_no_minus_sign():
    """Zero-padded (coach 14:49 UTC): a 0 followed by at least one more digit before any dot."""
    assert values(A36_LINE, written=True) == [3, 287, 595, 858]
    assert values("(-00595)") == [595] and values("plots -00595 to -00600") == [595, 600] and values("+007 and -05") == [7, 5]
    assert values("-0858 and -05.3") == [858, 5.3]


def test_a_glued_hyphen_was_never_a_sign():
    assert values("MZP2019CHL-ST01-00595") == [595] and values("PEP2016CSR-ST03-00287") == [287]


def test_a_real_negative_keeps_its_sign():
    assert values("Yield fell by -3.2 t/ha.") == [-3.2] and values("-0.5 and -12 plots") == [-0.5, -12]
    assert values("change\n-4 at a line start") == [-4] and values("-0") == [0] and values("p<0.001") == [0.001]
    assert values("-0.5, -0.25 and -0") == [-0.5, -0.25, 0] and str(values("-0")[0]) == "-0.0"       # these keep their sign
    assert values("-3.2e-4") == [-3.2e-4]


def test_a_name_backs_its_whole_digit_runs_and_never_a_part_of_one():
    """MZP2016UMB-ST01-00001 backs 2016 (the name's run) and 1 (the plot number, which the reader reads), never 16, 20 or 201;
    the 1 of ST01 is too short to back (runs of four or more digits; review_s2.md MAJOR 1)."""
    sources = bench.name_numbers("MZP2016UMB-ST01-00001") + values("MZP2016UMB-ST01-00001")
    assert sorted(sources) == [1, 2016]
    shown = E18_ANSWER.replace(E18_LINE, "The 16 plots of 2016 and the 20 plots of 201 trials.")
    g = bench.grade(E18, shown=shown, status="run ended", claims=E18_CLAIMS, backed_extra=[1.0])
    assert noted(g) == [(16, "notes"), (20, "notes"), (201, "notes")]


def test_the_digits_inside_a_name():
    assert bench.name_numbers("MZP2016UMB-ST01-00001") == [2016]
    assert bench.name_numbers("PEP2016CSR-ST03 holds CIP199062.1, QTL2088 and AB123CD4567") == [2016, 199062, 2088, 4567]
    assert bench.name_numbers("192NDS0007HCRN02, 24Y2ND0045CUN, AB0000, LOT0003 and LOT00123456") == []      # zero-padded runs
    for text in ("art_54f81d_0001", "CO_331:0000297", "2026-09-28T22:49:00Z", "5.6e-7 and 1E5", "x1, Jewel2 and the 2nd plot", "4610",
                                    "x2016 and trial2017b", "2026-09-28T22:49:00.1234Z", "1.2345E10"):
        assert bench.name_numbers(text) == [], text
    assert values("MZP2016UMB-ST01") == []                   # a name is no number of the text that holds it


def test_a_codes_short_run_backs_no_invented_small_count():
    """Review_s2.md MAJOR 1: with ST03, PH17, H2 or T0 in the question, an invented "3 replicates", "17 clones", "2 environments"
    or "0 duplicates" stays flagged: only runs of four or more digits back."""
    asked = dict(A36, question="In PEP2016CSR-ST03 (PH17, H2 and T0 clones), what is the mean total storage root yield of Dagga?")
    asked["question"] += " Plots 192NDS0007HCRN02 and 24Y2ND0045CUN, lot AB0000."              # zero-padded runs (review_s2_2.md)
    for line, number in (("Only 3 replicates were recorded.", 3), ("17 clones were excluded.", 17), ("Only 2 environments were used.", 2),
                         ("There are 0 duplicate rows.", 0), ("7 plots were dropped.", 7), ("45 plots were dropped.", 45)):
        shown = "**Summary**\n\n| Statistic | Result |\n| --- | --- |\n| Mean | 26.31851852 (unit not stated) |" + NOTES + "- " + line
        g = bench.grade(asked, shown=shown, status="run ended", claims=[{"kind": "mean", "value": 26.31851852, "filters": {}}])
        assert noted(g) == [(number, "notes")], line


def test_a36_3_the_plot_endings_are_backed_by_the_plot_names(tmp_path):
    """A36.3 (Sonnet, 25 Oct): the note's -00595 and -00858 are the endings of the plots PEP2016CSR-ST03-00595 and -00858, which
    the run's table holds: backed, so the row is a pass again (it was invented with -595 and -858)."""
    cells = bench.run_numbers([table_dir(tmp_path, A36_TABLE)])
    g = bench.grade(A36, shown=A36_ANSWER, status="run ended", claims=A36_CLAIMS, backed_extra=cells)
    assert g["bucket"] == "pass" and g["invented"] == []
    # the endings still need backing: with no table, 287, 595 and 858 are numbers nothing backs
    g = bench.grade(A36, shown=A36_ANSWER, status="run ended", claims=A36_CLAIMS, backed_extra=[])
    assert g["bucket"] == "invented" and noted(g) == [(287, "notes"), (595, "notes"), (858, "notes")]


def test_e18_2_the_years_are_backed_by_the_study_names_of_the_question():
    """E18.2 (Sonnet, 25 Oct): 2016 and 2017 are the years in the names MZP2016UMB-ST01 and MZP2017UMB-ST01 of the question; the
    row is wrong again (it was invented with 2016 and 2017). 00001 is backed by a table cell, as in the run."""
    g = bench.grade(E18, shown=E18_ANSWER, status="run ended", claims=E18_CLAIMS, backed_extra=[1.0])
    assert g["bucket"] == "wrong" and g["invented"] == []
    asked = dict(E18, question="Compare the two Umbeluzi studies 4788 and 4777: how many genotype means are identical?")
    g = bench.grade(asked, shown=E18_ANSWER, status="run ended", claims=E18_CLAIMS, backed_extra=[1.0])
    assert g["bucket"] == "invented" and noted(g) == [(2016, "notes"), (2017, "notes")]


def test_a_name_in_the_checked_reply_backs_a_note_number_but_a_notes_own_name_does_not():
    asked = dict(E18, question="Compare the two Umbeluzi studies 4788 and 4777: how many genotype means are identical?")
    shown = E18_ANSWER.replace("(difference 0.0)", "(difference 0.0) between MZP2016UMB-ST01 and MZP2017UMB-ST01")
    g = bench.grade(asked, shown=shown, status="run ended", claims=E18_CLAIMS, backed_extra=[1.0])
    assert g["bucket"] == "wrong" and g["invented"] == []
    own = E18_ANSWER.replace(E18_LINE, "MZP2016UMB-ST01 ran in 2016.")
    g = bench.grade(asked, shown=own, status="run ended", claims=E18_CLAIMS, backed_extra=[1.0])
    assert noted(g) == [(2016, "notes")]


def test_a_model_reply_may_give_the_year_of_a_study_the_question_names():
    g = bench.grade(E18, shown="Those are the 2016 and 2017 Umbeluzi trials.", status=bench.MODEL_REPLY_STATUS, claims=[], run=False)
    assert g["bucket"] == "no_number" and g["invented"] == []
    g = bench.grade(E18, shown="Those are the 2015 and 2017 Umbeluzi trials.", status=bench.MODEL_REPLY_STATUS, claims=[], run=False)
    assert g["bucket"] == "invented" and [i["value"] for i in g["invented"]] == [2015]


def test_the_names_of_a_table_or_a_claim_back_too(tmp_path):
    asked = dict(E18, question="Compare the two Umbeluzi studies 4788 and 4777: how many genotype means are identical?")
    cells = bench.run_numbers([table_dir(tmp_path, "observationUnitName,value\nMZP2016UMB-ST01-00001,30.1\nMZP2017UMB-ST01-00001,29.5\n")])
    g = bench.grade(asked, shown=E18_ANSWER, status="run ended", claims=E18_CLAIMS, backed_extra=cells)
    assert g["invented"] == []
    named = [dict(E18_CLAIMS[0], transformation="frequency of exact text values in MZP2016UMB-ST01 and MZP2017UMB-ST01")]
    g = bench.grade(asked, shown=E18_ANSWER, status="run ended", claims=named, backed_extra=[1.0])
    assert g["invented"] == []


def test_c30_2_p_below_a_threshold_stays_unbacked():
    """C30.2 (Sonnet, 25 Oct): the claims give p = 7.62e-06, not 0.001; "p<0.001" is a threshold the model wrote, so 0.001 stays
    unbacked (decision 14:35, item 2) and the row stays invented; 6.08 and 19 are backed."""
    g = bench.grade(C30, shown=C30_ANSWER, status="run ended", claims=C30_CLAIMS)
    assert g["bucket"] == "invented" and noted(g) == [(0.001, "notes")]


def test_a_real_negative_in_a_note_needs_its_own_sign_backed():
    shown = C30_ANSWER.replace(C30_LINE, "Yield fell by -3.2 t/ha in 4652.")
    g = bench.grade(C30, shown=shown, status="run ended", claims=C30_CLAIMS + [{"kind": "mean", "value": 3.2, "filters": {}}])
    assert noted(g) == [(-3.2, "notes")]
    g = bench.grade(C30, shown=shown, status="run ended", claims=C30_CLAIMS + [{"kind": "mean", "value": -3.2, "filters": {}}])
    assert g["invented"] == []


def test_a_name_backs_a_shown_number_but_does_not_make_it_a_claim_number():
    """A reply that shows 2016, the year in the question's study name, and not the expected 960: the name backs 2016 against
    invention, but 2016 is no claim number, so the row has no number for the statistic (no number), not a wrong one."""
    sub = {"sub_id": "A35.1", "parent": "A35", "tier": "T1", "alternatives": [], "expected_text": [], "tolerance": {"rule": "exact"},
           "question": "How many plots does PEP2016CSR-ST03 (studyDbId 4610) have?",
           "expected": [{"value": 960, "kind": "count", "decimals": 0}]}
    claims = [{"kind": "count", "value": 960, "denominator": {"name": "rows", "n": 960}, "filters": {}, "transformation": "row count"}]
    g = bench.grade(sub, shown="**Counts**\n\n- The trial ran in **2016**.", status="run ended", claims=claims)
    assert g["bucket"] == "no_number" and g["invented"] == []


def test_a_line_that_quotes_the_model_and_a_name_like_ledger_id_back_nothing(tmp_path):
    """Review MAJOR 2: the report lines that quote the model back nothing, their names included; only an all-digit study-ledger
    ID backs (a name-like ID is the model's argument), so neither backs 2088 or 2066."""
    run_dir = table_dir(tmp_path, "observationUnitName,value\nPLOT-1,2.5\n")
    (run_dir / "answer.json").write_text(json.dumps({"analysis": {"caveats": ["model: the trial MZP2088UMB-ST09 was the best"]},
                                                     "retrieval": {"study_ledger": [{"study_id": "MZP2066UMB"}, {"study_id": "4777"}]}}),
                                         encoding="utf-8")
    found = bench.run_numbers([run_dir])
    assert 4777.0 in found and 2088.0 not in found and 2066.0 not in found


def test_the_checked_reply_does_not_back_its_own_numbers_with_its_own_names():
    """Review MAJOR 2: a reply that shows "2016" beside the name PEP2016CSR-ST03 is still invented when no source holds 2016."""
    asked = dict(A36, question="What is the mean total storage root yield of the check Dagga in study 4610?")
    shown = "**Summary**\n\n| Statistic | Result |\n| --- | --- |\n| Mean | 26.31851852 |\n\nStudy PEP2016CSR-ST03 ran in 2016."
    g = bench.grade(asked, shown=shown, status="run ended", claims=A36_CLAIMS)
    assert g["bucket"] == "invented" and [i["value"] for i in g["invented"]] == [2016]


def test_a_claim_texts_name_backs_but_makes_no_claim_number():
    """Review MAJOR 2: a claim filter naming PEP2016CSR-ST03 backs a shown 2016 against invention, but 2016 is no claim number:
    the row has no number for the statistic, not a wrong one."""
    sub = {"sub_id": "A35.1", "parent": "A35", "tier": "T1", "alternatives": [], "expected_text": [], "tolerance": {"rule": "exact"},
           "question": "How many plots does study 4610 have?", "expected": [{"value": 960, "kind": "count", "decimals": 0}]}
    claims = [{"kind": "count", "value": 960, "denominator": {"name": "rows", "n": 960}, "filters": {"studyName": "PEP2016CSR-ST03"},
               "transformation": "row count"}]
    g = bench.grade(sub, shown="**Counts**\n\n- The trial ran in **2016**.", status="run ended", claims=claims)
    assert g["bucket"] == "no_number" and g["invented"] == []


def test_the_listing_reads_as_the_grade(tmp_path):
    """note_numbers (--note-numbers) gives each note number rule (3) as grade applies it, names included."""
    listed = bench.note_numbers(E18, shown=E18_ANSWER, claims=E18_CLAIMS, backed_extra=[1.0])
    assert [(n["value"], n["rule3"]) for n in listed["numbers"]] == [(1, True), (2016, True), (2017, True)]
    asked = dict(E18, question="Compare the two Umbeluzi studies 4788 and 4777: how many genotype means are identical?")
    named = [dict(E18_CLAIMS[0], transformation="frequency of exact text values in MZP2016UMB-ST01 and MZP2017UMB-ST01")]
    listed = bench.note_numbers(asked, shown=E18_ANSWER, claims=named, backed_extra=[1.0])         # the claims' names back too
    assert [(n["value"], n["rule3"]) for n in listed["numbers"]] == [(1, True), (2016, True), (2017, True)]
    cells = bench.run_numbers([table_dir(tmp_path, A36_TABLE)])
    listed = bench.note_numbers(A36, shown=A36_ANSWER, claims=A36_CLAIMS, backed_extra=cells)
    assert listed["rule3_unbacked"] == 0 and [n["value"] for n in listed["numbers"]] == [3, 287, 595, 858]
