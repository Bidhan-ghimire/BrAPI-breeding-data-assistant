"""The note-number listing (coach 2026-10-25 02:52 UTC, NEXT 2, while Bidhan's rule for note numbers, item (31), is open): every
number in what the app marks as model text (the "Additional model notes (unverified)" bullets and the "Unverified model note:"
paragraphs, which the grader leaves out) is listed with its backing under two readings, and counted, for each row, in results.json
and in results.md, and for a saved results folder. A measurement only: no bucket, grade or shown text depends on it. Real-looking
inputs from the Sonnet runs of 24-25 Oct: B26.4's note, whose 40 is a real block code (backed under rule (3)) but no claim's value, and
C30.1's p-value written to two figures."""
import importlib.util
import json
import sys
from pathlib import Path

from part2_support import synthetic_models
from test_eval_harness import FIXTURES, MEASURED, _part2_flow, place, scripted_front_door  # noqa: F401  (fixtures; _part2_flow is autouse)

BENCHMARK = Path(__file__).resolve().parents[1] / "eval" / "benchmark.py"
_spec = importlib.util.spec_from_file_location("_eval_benchmark_notes_under_test", BENCHMARK)
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench
_spec.loader.exec_module(bench)
harness = bench.harness

B264 = {"sub_id": "B26.4", "parent": "B26", "tier": "T1",
        "question": "What experimental design does SweetPotatoBase record for PEP2016ICA-ST04 (studyDbId 4608; folder '2016-17 BxT', id "
                    "4619; Peru-CIP, Ica 2016), and how many replicates, plots per replicate, plots in total, and incomplete blocks of "
                    "what size per replicate does it have?",
        "expected": [{"value": v, "kind": "count", "decimals": 0} for v in (2, 320, 640, 80, 4)], "expected_text": ["alpha"],
        "tolerance": {"rule": "exact"}, "alternatives": []}
DESIGN_NOTE = ("- The supplied plot list has no field that names the design type, so I cannot confirm the label SweetPotatoBase "
               "records, such as alpha-lattice. The structure is consistent with a resolvable incomplete-block design, but that is my "
               "reading of the counts.")
BLOCK_NOTE = ("- I did not assign blocks to replicates in a separate tool step. The 40 blocks per replicate comes from 80 blocks of 8 "
              "plots split across 2 equal replicates of 320 plots.")
B264_ANSWER = "\n".join([
    "**rep: 2 distinct IDs**", "", "| ID | Records |", "| --- | --- |", "| 1 | 320 |", "| 2 | 320 |", "",
    "Based on rows in the complete observation\\_units table: 640.", "",
    "**block: 80 groups**", "", "| ID | Records |", "| --- | --- |", "| 1 | 8 |", "| 10 | 8 |", "",
    "Showing 2 of 80 entries; 78 additional IDs are not displayed here. Full report records the counts and source evidence.", "",
    "**Additional model notes (unverified)**", "", DESIGN_NOTE, BLOCK_NOTE, "",
    "Full report contains the complete methods and evidence."])
B264_CLAIMS = ([{"kind": "count", "value": 2, "filters": {}, "transformation": "number of distinct non-empty exact text values"}]
               + [{"kind": "count", "value": 320, "filters": {"rep": r}, "transformation": "frequency of exact text values"} for r in "12"]
               + [{"kind": "count", "value": 8, "filters": {"block": str(b)}, "transformation": "frequency of exact text values"}
                  for b in range(1, 81)])
B264_CELLS = [float(b) for b in range(1, 81)] + [1.0, 2.0] + [float(p) for p in range(1, 641)]   # the plot table's block, rep, plot cells


def flags(listing):
    return [(n["value"], n["rule3"], n["claim_value"]) for n in listing["numbers"]]


def test_model_notes_are_exactly_the_lines_code_written_leaves_out():
    """The reader of model text is the grader's own split: every line of the plain answer is in the checked text or in the
    notes, never both (the heading and blank lines aside); a paragraph that starts "Unverified model note:" is a note too."""
    answer = (B264_ANSWER + "\n\nUnverified model note: The keyword search covered 9 terms.\n\n"
              "Reply **accept** if this answers your question.")
    notes = harness.model_notes(answer)
    kept = [line for line in harness.code_written(answer).split("\n") if line.strip()]
    every = [line for line in harness.plain(answer).split("\n") if line.strip() and line.strip() != "Additional model notes (unverified)"]
    assert notes == [harness.plain(DESIGN_NOTE), harness.plain(BLOCK_NOTE), "Unverified model note: The keyword search covered 9 terms."]
    assert sorted(kept + notes) == sorted(every) and not set(kept) & set(notes)


def test_the_notes_block_ends_at_the_first_line_that_is_no_bullet_and_survives_a_blank_line():
    """Review 2026-10-25 S4-2, S4-3: a blank line between two note bullets keeps the block open; the first line that is neither
    blank nor a bullet closes it, so a later "- " line is the app's again."""
    answer = ("Mean: 2.19\n\n**Additional model notes (unverified)**\n\n- Note one, 3 plots.\n\n- Note two, 4 plots.\n"
              "Full report contains the complete methods and evidence.\n\n- Distinct IDs alone do not establish independence, 5.")
    assert harness.model_notes(answer) == ["- Note one, 3 plots.", "- Note two, 4 plots."]
    assert "- Distinct IDs alone do not establish independence, 5." in harness.code_written(answer)
    # review 2026-10-25, round 2: "-5 ..." is no bullet (it ends the block), and a line that only mentions the note marker is
    # the app's
    other = ("**Additional model notes (unverified)**\n- Note one, 3 plots.\n-5 is the lowest value.\n- a later bullet, 6.\n"
             "The text after Unverified model note: is the model's.")
    assert harness.model_notes(other) == ["- Note one, 3 plots."]
    assert harness.code_written(other).split("\n") == ["-5 is the lowest value.", "- a later bullet, 6.",
                                                       "The text after Unverified model note: is the model's."]


def test_every_number_of_a_note_is_read_the_apps_phrases_too():
    """Review 2026-10-25 S1: "80 groups", "12 distinct IDs" and "source 3" are blanked in the checked text, which the app writes;
    in a note they are the model's own words, so the listing reads them (as grade reads a reply with no run)."""
    note = "**Additional model notes (unverified)**\n\n- The table holds 80 groups of 4 plots, and 12 distinct IDs; see source 3."
    listing = bench.note_numbers({"question": "x"}, shown=note, claims=[], backed_extra=[])
    assert [n["value"] for n in listing["numbers"]] == [80, 4, 12, 3] and all(type(n["value"]) is int for n in listing["numbers"])


def test_a_number_the_reader_cannot_hold_makes_an_error_not_a_stopped_run(monkeypatch):
    """Review 2026-10-25 S2: a listing that raises is an error, and the row and the run go on. "0E0326", which overflowed
    _backed, is no number since decision (31)(a) (the reader skips an exponent past 300), so a raising _backed stands in."""
    note = "**Additional model notes (unverified)**\n\n- the plot code 0E0326 was not found in the table, 7 times."
    assert [n["value"] for n in bench.listed_notes({"question": "x"}, shown=note, claims=[], backed_extra=[])["numbers"]] == [7]

    def overflow(*args):
        raise OverflowError("int too large to convert to float")
    monkeypatch.setattr(bench, "_backed", overflow)
    listing = bench.listed_notes({"question": "x"}, shown=note, claims=[], backed_extra=[])
    assert listing["count"] == 0 and listing["error"].startswith("OverflowError")
    assert bench.note_summary([{"sub_id": "Z1.1", "note_numbers": listing}])["rows_error"] == ["Z1.1"]

    def unreadable(*args):
        raise ValueError("Exceeds the limit (4300 digits)\nfor integer string conversion")
    monkeypatch.setattr(bench, "_backed", unreadable)
    huge = bench.listed_notes({"question": "x"}, shown=note, claims=[], backed_extra=[])
    assert huge["error"] == "ValueError: Exceeds the limit (4300 digits) for integer string conversion"     # any error, on one line


def test_b264_the_40_is_backed_under_rule3_but_by_no_claim_value():
    """B26.4 (Sonnet, 24 and 25 Oct): "40 blocks per replicate" is wrong (80 blocks of 4 in each rep), yet 40 is a real block code,
    in a claim's filter and in the plot table: rule (3) backs it; no claim has the value 40 (nor 80)."""
    listing = bench.note_numbers(B264, shown=B264_ANSWER, claims=B264_CLAIMS, backed_extra=B264_CELLS)
    assert flags(listing) == [(40, True, False), (80, True, False), (8, True, True), (2, True, True), (320, True, True)]
    assert (listing["lines"], listing["count"], listing["rule3_unbacked"], listing["claim_unbacked"]) == (2, 5, 0, 2)
    assert "40 blocks per replicate" in listing["numbers"][0]["context"]


def test_rule3_reads_the_claims_texts_the_runs_tables_and_the_checked_reply():
    """Rule (3), as graded (decision (31)(a)): without the per-block claims and the plot table, the checked reply's own 80
    ("block: 80 groups") and 8 (its table) still back the note's, and nothing backs the 40; the table's cells back a number on
    their own."""
    bare = [c for c in B264_CLAIMS if "block" not in c["filters"]]
    listing = bench.note_numbers(B264, shown=B264_ANSWER, claims=bare, backed_extra=[])
    assert flags(listing) == [(40, False, False), (80, True, False), (8, True, False), (2, True, True), (320, True, True)]
    assert listing["rule3_unbacked"] == 1
    assert flags(bench.note_numbers(B264, shown=B264_ANSWER, claims=bare, backed_extra=[40.0]))[:1] == [(40, True, False)]


def test_c301_a_p_value_written_to_two_figures_is_read_at_the_precision_written():
    """C30.1 (Sonnet, 24 Oct): "p about 5.6e-7" is read at the two figures it is written with (rule (3), as graded), so the
    claim's 5.603e-07 backs it under both readings, as it backs 7.36, 19 and 0.066."""
    sub = {"question": "How do the 90-day and 120-day San Ramon trials compare?"}
    note = "- The comparison is descriptive. The paired t on entries is 7.36 (df 19, p about 5.6e-7), while Welch's p is 0.066."
    claims = [{"kind": "test", "value": v, "filters": {}, "transformation": t}
              for v, t in ((7.3644, "paired t"), (19, "paired t: df"), (5.603e-07, "paired t: p"), (0.0664, "Welch t: p"))]
    listing = bench.note_numbers(sub, shown="**Additional model notes (unverified)**\n\n" + note, claims=claims, backed_extra=[])
    assert flags(listing) == [(7.36, True, True), (19, True, True), (5.6e-07, True, True), (0.066, True, True)]
    assert (listing["count"], listing["rule3_unbacked"], listing["claim_unbacked"]) == (4, 0, 0)


def test_the_questions_numbers_back_rule3_only():
    sub = {"question": "Compare the 90-day and 120-day trials."}
    note = "**Additional model notes (unverified)**\n\n- The 90-day and 120-day labels come from the question, not from the metadata."
    assert flags(bench.note_numbers(sub, shown=note, claims=[], backed_extra=[])) == [(90, True, False), (120, True, False)]


def test_a_reply_with_no_run_is_backed_by_the_question_alone():
    """As in the invented check: with no run, only the question's numbers back a number under rule (3) (the run's numbers are not read);
    the question's 4608 still does."""
    note = "**Additional model notes (unverified)**\n\n- 40 blocks in study 4608."
    assert flags(bench.note_numbers(B264, shown=note, claims=B264_CLAIMS, backed_extra=B264_CELLS, run=False)) == [
        (40, False, False), (4608, True, False)]


def test_no_notes_no_numbers():
    assert bench.note_numbers(B264, shown="Mean: 2.19 (unit not stated)", claims=B264_CLAIMS, backed_extra=B264_CELLS) == {
        "lines": 0, "numbers": [], "count": 0, "rule3_unbacked": 0, "claim_unbacked": 0}
    notes_only = "**Additional model notes (unverified)**\n\n- The unit is not stated in the metadata."
    assert bench.note_numbers(B264, shown=notes_only, claims=[], backed_extra=[])["lines"] == 1


def test_claim_values_are_the_numeric_fields_and_claim_numbers_still_reads_the_texts():
    """Review 2026-10-25 S4-8, S4-9: claim_numbers reads every text it read before (filters, grouping, transformation, missing
    reason, denominator name, exclusion reasons), from a list or an iterator alike; claim_values reads none of them."""
    claims = [{"kind": "count", "value": 278, "filters": {"studyDbId": "4620"}, "grouping": ["season 2016"],
               "denominator": {"n": 317, "name": "entries of 318 with a value"}, "n_independent_units": 317,
               "exclusions": [{"count": 1, "reason": "plots below 25 left out"}], "missing_reason": "7 plots not measured",
               "transformation": "count of entries above the threshold 10.2685"}]
    assert sorted(bench.claim_values(claims)) == [1.0, 278.0, 317.0, 317.0]
    assert {4620.0, 2016.0, 318.0, 25.0, 7.0, 10.2685} <= set(bench.claim_numbers(claims))
    assert sorted(bench.claim_numbers(iter(claims))) == sorted(bench.claim_numbers(claims))
    listing = bench.note_numbers(B264, shown=B264_ANSWER, claims=iter(B264_CLAIMS), backed_extra=B264_CELLS)
    assert flags(listing) == [(40, True, False), (80, True, False), (8, True, True), (2, True, True), (320, True, True)]


def test_the_grade_does_not_read_the_listing():
    """The listing is a measurement: grading B26.4's answer gives the same result before and after listing its notes, and the
    graded text never holds a note line."""
    before = bench.grade(B264, shown=B264_ANSWER, status="run ended", claims=B264_CLAIMS, backed_extra=B264_CELLS)
    bench.note_numbers(B264, shown=B264_ANSWER, claims=B264_CLAIMS, backed_extra=B264_CELLS)
    after = bench.grade(B264, shown=B264_ANSWER, status="run ended", claims=B264_CLAIMS, backed_extra=B264_CELLS)
    assert before == after and before["bucket"] == "wrong"
    assert not any(line in harness.code_written(B264_ANSWER) for line in harness.model_notes(B264_ANSWER))


def test_the_totals_count_rows_with_numbers_and_name_the_rows_with_unbacked_ones():
    rows = [{"sub_id": "A1.1", "note_numbers": {"lines": 2, "count": 3, "rule3_unbacked": 1, "claim_unbacked": 2}},
            {"sub_id": "A1.2", "note_numbers": {"lines": 1, "count": 0, "rule3_unbacked": 0, "claim_unbacked": 0}},
            {"sub_id": "A1.3", "note_numbers": {"lines": 1, "count": 2, "rule3_unbacked": 0, "claim_unbacked": 1}}, {"sub_id": "A1.4"}]
    summary = bench.note_summary(rows)
    assert summary == {"count": 5, "rows_with_notes": 3, "rows_with_numbers": 2, "rule3_unbacked": 1, "claim_unbacked": 3,
                       "rows_rule3_unbacked": ["A1.1"], "rows_claim_unbacked": ["A1.1", "A1.3"], "rows_error": []}
    assert bench.note_line(summary) == ("- Note numbers (the model's notes): 5 in 2 rows, of 3 rows with notes; no backing under "
                                        "rule (3): 1 (A1.1); no claim value: 3 (in 2 rows)")
    one = bench.note_summary(rows[:1])
    assert bench.note_line(one).endswith("3 in 1 row, of 1 row with notes; no backing under rule (3): 1 (A1.1); "
                                         "no claim value: 2 (in 1 row)")


def test_the_runner_writes_the_listing_and_its_counts(place, monkeypatch):
    """End to end (the runner test's setup): every row carries its listing, the report its totals, results.md a line; the
    buckets are the ones the runner test pins. The listing reads what the grade reads: the same answer, claims and run
    numbers (the same objects), and the grade still gets the answer as shown."""
    counts = bench.harness.reply_tokens
    monkeypatch.setattr(bench.harness, "reply_tokens", lambda model, usage, detail: counts(model, usage, detail) or
                        {"model": model, "input": 0, "cache_write": 0, "cache_read": 0, "output": 0})
    calls, real_grade, real_notes = {}, bench.grade, bench.note_numbers
    monkeypatch.setattr(bench, "grade", lambda sub, **kw: calls.setdefault("grade", kw) and real_grade(sub, **kw))
    monkeypatch.setattr(bench, "note_numbers", lambda sub, **kw: calls.setdefault("notes", kw) and real_notes(sub, **kw))
    place.seed(*FIXTURES)
    subs = [{"sub_id": "S1.1", "parent": "S1", "key_facts": [0], "tier": "T1", "question": MEASURED,
             "expected": [{"value": 18.0, "kind": "mean", "decimals": 0}], "expected_text": [],
             "tolerance": {"rule": "relative", "rel": 0.005}, "alternatives": [], "studies": [], "folders": [], "variables": [],
             "statistic": "pooled mean", "caveat": None, "gold_basis": "test"}]
    path = place.root / "subs.json"
    path.write_text(json.dumps({"subquestions": subs}), encoding="utf-8")
    _model, front_door = scripted_front_door("data_question")
    assert bench.main(["--cache", str(place.source), "--subquestions", str(path), "--results", str(place.root / "results"), "--tiers", "T1",
                       "--confirm"], base_config=place.base_config, make_models=synthetic_models, triage_model=front_door,
                      preflight=lambda q: place.base_config(q).validate()) == 0
    [folder] = list((place.root / "results").iterdir())
    report = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    [row] = report["subquestions"]
    assert row["bucket"] == "pass" and set(row["note_numbers"]) == {"lines", "numbers", "count", "rule3_unbacked", "claim_unbacked"}
    listing = row["note_numbers"]
    assert listing["count"] == len(listing["numbers"]) and listing["rule3_unbacked"] == sum(not n["rule3"] for n in listing["numbers"])
    assert report["note_numbers"] == {"count": listing["count"], "rows_with_notes": int(listing["lines"] > 0),
                                      "rows_with_numbers": int(listing["count"] > 0),
                                      "rule3_unbacked": listing["rule3_unbacked"], "claim_unbacked": listing["claim_unbacked"],
                                      "rows_rule3_unbacked": ["S1.1"] if listing["rule3_unbacked"] else [],
                                      "rows_claim_unbacked": ["S1.1"] if listing["claim_unbacked"] else [], "rows_error": []}
    assert "- Note numbers (the model's notes): " in (folder / "results.md").read_text(encoding="utf-8")
    assert calls["grade"]["shown"] == calls["notes"]["shown"] == row["answer_full"]
    assert calls["notes"]["backed_extra"] is calls["grade"]["backed_extra"] and calls["notes"]["claims"] is calls["grade"]["claims"]
    assert calls["notes"]["run"] is calls["grade"]["run"] is True and calls["grade"]["backed_extra"]


def test_a_listing_that_fails_does_not_stop_the_run(place, monkeypatch):
    """Review 2026-10-25 S2, end to end: when the listing raises, the row keeps its grade, its listing says why, and the run
    writes its results."""
    counts = bench.harness.reply_tokens
    monkeypatch.setattr(bench.harness, "reply_tokens", lambda model, usage, detail: counts(model, usage, detail) or
                        {"model": model, "input": 0, "cache_write": 0, "cache_read": 0, "output": 0})

    def broken(sub, **kwargs):
        raise OverflowError("int too large to convert to float")
    monkeypatch.setattr(bench, "note_numbers", broken)
    place.seed(*FIXTURES)
    subs = [{"sub_id": "S1.1", "parent": "S1", "key_facts": [0], "tier": "T1", "question": MEASURED,
             "expected": [{"value": 18.0, "kind": "mean", "decimals": 0}], "expected_text": [],
             "tolerance": {"rule": "relative", "rel": 0.005}, "alternatives": [], "studies": [], "folders": [], "variables": [],
             "statistic": "pooled mean", "caveat": None, "gold_basis": "test"}]
    path = place.root / "subs.json"
    path.write_text(json.dumps({"subquestions": subs}), encoding="utf-8")
    _model, front_door = scripted_front_door("data_question")
    assert bench.main(["--cache", str(place.source), "--subquestions", str(path), "--results", str(place.root / "results"), "--tiers", "T1",
                       "--confirm"], base_config=place.base_config, make_models=synthetic_models, triage_model=front_door,
                      preflight=lambda q: place.base_config(q).validate()) == 0
    [folder] = list((place.root / "results").iterdir())
    report = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    [row] = report["subquestions"]
    assert row["bucket"] == "pass" and row["note_numbers"]["error"] == "OverflowError: int too large to convert to float"
    assert report["note_numbers"]["rows_error"] == ["S1.1"] and "listing failed: S1.1" in (folder / "results.md").read_text(encoding="utf-8")


def test_a_saved_folder_is_listed_without_changing_its_results(tmp_path):
    """--note-numbers <folder>: a saved run's notes listed from its own files (the full answer, the full claims of every run's
    answer.json, the cells of its saved tables), written to note_numbers.json and note_numbers.md beside results.json, which is
    not touched. Review 2026-10-25 S3, S4-5, S4-6: the full answer before the cut one, every run's claims, and a listing that
    says when it read a cut answer or missed a run folder."""
    folder = tmp_path / "benchmark_20261025T020426Z"
    first, second = folder / "out" / "run_20261025T020430_abcdef", folder / "out" / "run_20261025T020431_abcdeg"
    (first / "artifacts").mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "answer.json").write_text(json.dumps({"analysis": {"claims": B264_CLAIMS[:1]}}), encoding="utf-8")
    (second / "answer.json").write_text(json.dumps({"analysis": {"claims": B264_CLAIMS[1:]}}), encoding="utf-8")
    (first / "artifacts" / "0001_observation_units.csv").write_text("observationUnitDbId,rep,block\n" + "".join(
        f"{1276868 + i},{1 + i // 320},{1 + (i % 320) // 4}\n" for i in range(640)), encoding="utf-8")
    empty = folder / "out" / "run_20261025T020432_abcdeh"         # a run folder that holds no answer.json
    empty.mkdir()
    answer = B264_ANSWER + "\n\nUnverified model note: The first plot is 1276868."     # backed by a table cell alone
    cut = (answer + "\n" + "x" * harness.GOT_CHARS)[:harness.GOT_CHARS]                # as harness.GOT_CHARS cuts it
    rows = [{**B264, "bucket": "wrong", "status": "run ended", "answer": answer[:300], "answer_full": answer,
             "run_ids": [first.name, second.name]},
            {**B264, "sub_id": "B26.5", "bucket": "wrong", "status": "run ended", "answer": cut, "run_ids": ["run_gone"]},
            {**B264, "sub_id": "B26.6", "bucket": "wrong", "status": "run ended", "answer": answer, "run_ids": [first.name, second.name]},
            {**B264, "sub_id": "B26.7", "bucket": "wrong", "status": "run ended", "answer_full": answer, "run_ids": [empty.name]}]
    (folder / "results.json").write_text(json.dumps({"subquestions": rows}), encoding="utf-8")
    saved = (folder / "results.json").read_bytes()
    assert bench.main(["--note-numbers", str(folder)]) == 0
    written = json.loads((folder / "note_numbers.json").read_text(encoding="utf-8"))
    listing = written["rows"]["B26.4"]
    assert flags(listing) == [(40, True, False), (80, True, False), (8, True, True), (2, True, True), (320, True, True),
                              (1276868, True, False)]
    read = {sid: (n["answer_source"], n["runs_missing"], n["answers_missing"]) for sid, n in written["rows"].items()}
    assert read == {"B26.4": ("answer_full", 0, 0), "B26.5": (f"answer (cut at {harness.GOT_CHARS} characters)", 1, 0),
                    "B26.6": ("answer", 0, 0), "B26.7": ("answer_full", 0, 1)}
    assert flags(written["rows"]["B26.6"]) == flags(listing)            # an answer under the cut is the whole answer
    summary = written["summary"]
    assert (summary["rows_cut_answer"], summary["rows_runs_missing"], summary["rows_answers_missing"]) == (["B26.5"], ["B26.5"], ["B26.7"])
    assert (folder / "results.json").read_bytes() == saved
    md = (folder / "note_numbers.md").read_text(encoding="utf-8").splitlines()
    assert md[2] == bench.note_line(summary)
    assert md[3] == "- Read from a cut answer: B26.5; run folders missing: B26.5; runs without answer.json: B26.7"
    assert "- B26.4: 6 numbers; no backing under rule (3): none; no claim value: 3" in md


def test_a_saved_folder_warns_on_one_kind_of_gap_and_keeps_failed_and_repeated_rows(tmp_path, monkeypatch):
    """One gap alone still gives the warning line; a row whose listing fails says so in its md line; a sub-question listed twice
    keeps both listings; "." names the folder in the heading."""
    folder = tmp_path / "benchmark_20261025T030000Z"
    run = folder / "out" / "run_20261025T030001_abcdef"
    run.mkdir(parents=True)
    note = "**Additional model notes (unverified)**\n\n- Only 2 plots."
    rows = [{**B264, "answer_full": note, "run_ids": [run.name]}, {**B264, "answer_full": note, "run_ids": [run.name]}]
    (folder / "results.json").write_text(json.dumps({"subquestions": rows}), encoding="utf-8")
    monkeypatch.chdir(folder)
    assert bench.main(["--note-numbers", "."]) == 0
    md = (folder / "note_numbers.md").read_text(encoding="utf-8").splitlines()
    assert md[0] == f"# Note numbers, {folder.name}" and "- B26.4: 1 number; no backing under rule (3): 2; no claim value: 1" in md
    assert md[3] == "- Read from a cut answer: none; run folders missing: none; runs without answer.json: B26.4, B26.4 #2"
    assert set(json.loads((folder / "note_numbers.json").read_text(encoding="utf-8"))["rows"]) == {"B26.4", "B26.4 #2"}

    def broken(sub, **kwargs):
        raise OverflowError("boom\nat the second line")
    monkeypatch.setattr(bench, "note_numbers", broken)
    assert bench.main(["--note-numbers", "."]) == 0
    md = (folder / "note_numbers.md").read_text(encoding="utf-8")
    assert ("- B26.4: 0 numbers; no backing under rule (3): none; no claim value: 0; listing failed: OverflowError: boom at the "
            "second line") in md
    assert "listing failed: B26.4, B26.4 #2" in md and ": 0 in 0 rows, of 2 rows with notes;" in md     # a failed row still has notes


def test_a_folder_without_benchmark_results_says_so(tmp_path, capsys):
    assert bench.main(["--note-numbers", str(tmp_path)]) == 2 and "no results.json" in capsys.readouterr().out
    (tmp_path / "results.json").write_text(json.dumps({"rows": []}), encoding="utf-8")       # eval/harness.py's own format
    assert bench.main(["--note-numbers", str(tmp_path)]) == 2 and "no benchmark rows" in capsys.readouterr().out
