"""Decision (31)(b), rule (3) (Bidhan, coach 2026-10-25 12:20 UTC): the answer shown drops a model-note line that holds a number
nothing backs the way the benchmark backs a note number (eval/benchmark.py grade, c5f19af: the claims' numbers with their filters
and texts, the question's numbers, the run's own tables and code-written fields, the checked reply's own numbers, at the
precision written); note lines without numbers, or whose numbers are all backed, stay under the "Additional model notes
(unverified)" label; the person's view changes only by the dropped lines; the run's manifest records them. note_backing copies
the grader's reader and sources, since the app does not ship eval/: the first tests tie the two."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import chat_presenter as cp
import note_backing as nb
import run as ctl
from contracts import AnalysisReport, Claim, Exclusion, MetadataFact, RetrievalReport, StudyStatusEntry
from part2_support import part2_config, synthetic_models
from test_eval_harness import FIXTURES, MEASURED, _part2_flow, place, scripted_front_door  # noqa: F401  (fixtures; _part2_flow is autouse)
from test_note_numbers import B264, B264_ANSWER, B264_CELLS, B264_CLAIMS

BENCHMARK = Path(__file__).resolve().parents[1] / "eval" / "benchmark.py"
_spec = importlib.util.spec_from_file_location("_eval_benchmark_note_backing_under_test", BENCHMARK)
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench
_spec.loader.exec_module(bench)

TEXTS = ["5.6e-7", "about 5.6e-7.", "p 6.8e-101 and 1.234e-301", "1e305", "the plot code 0E0326 was not found", "0e" + "9" * 5000,
         "2,039 rows", "2.38 t/ha", "-2.5 t/ha", "2026-09-28 22:49 and 2026-09-28T22:49:00Z", "CO_331:0000296 and CO_331:0001016",
         "PEP2016CSR-ST03-00287, -00595 and -00858", "MZP2019CHL-ST01-00595", "(-00595)", "+007 and -05", "-0", "-0.5 and -0.25", "-3.2e-4",
         "MZP2016UMB-ST01-00001 and CIP199062.1", "x1, Jewel2 and the 2nd plot", "art_54f81d_0001",
         "x2016 and trial2017b", "2026-09-28T22:49:00.1234Z", "192NDS0007HCRN02, 24Y2ND0045CUN and LOT00123456",
         "PH123 and AB123CD4567", "-05.3", "MZP_2017_UMB and MZP2017_UMB", "LOT0003",
         "84.7 % of 314", "(df 19, p about 5.6e-7)", "1.5e300", "at 12:30", "x1.5", "3.",
         ".5", "+7", "1e5.", "1e5.5", "all 80 groups tie", "12 distinct IDs; source 3", "1,234.5 and 1,2345", "observation\\_units: 640"]
CLAIMS = [{"kind": "count", "value": 278, "unit": None, "filters": {"studyDbId": "4620", "block": "40", "studyName": "MZP2016UMB-ST01"},
           "grouping": ["season 2016"],
           "denominator": {"n": 317, "name": "entries of 318 with a value"}, "n_independent_units": 317,
           "exclusions": [{"count": 1, "reason": "plots below 25 left out"}], "missing_reason": "7 plots not measured",
           "transformation": "count of entries above the threshold 10.2685 in MZP2017UMB-ST01"},
          {"kind": "test", "value": 5.603e-07, "filters": {}, "transformation": "paired t: p"}, {"kind": "mean", "value": None,
           "missing_reason": "no valid value", "filters": {}, "transformation": "mean"}]


def test_the_reader_is_the_graders():
    for text in TEXTS:
        for written in (False, True):
            assert nb.numbers(text, written=written) == bench._numbers(text, written=written), (text, written)
        assert nb.name_numbers(text) == bench.name_numbers(text), text
    assert nb.plain("a **bold** observation\\_units &amp; 5") == bench.harness.plain("a **bold** observation\\_units &amp; 5")


def test_backed_and_claim_numbers_are_the_graders():
    for value, decimals, sources in ((5.6e-07, 8, [5.603e-07]), (5.6e-07, 10, [5.603e-07]), (-2.5, 1, [2.5]), (40, 0, [40.0]),
                                     (0.066, 3, [0.0664]), (5, 0, [float("inf")]), (1e300, -297, [1.2e300])):
        assert nb.backed(value, decimals, sources) == bench._backed(value, decimals, sources), (value, decimals, sources)
    assert sorted(nb.claim_numbers(CLAIMS)) == sorted(bench.claim_numbers(CLAIMS))
    assert sorted(nb.claim_numbers(iter(CLAIMS))) == sorted(bench.claim_numbers(CLAIMS))
    assert sorted(nb.claim_names(CLAIMS)) == sorted(bench.claim_names(CLAIMS)) == [2016, 2017]       # S2: a filter's and a method's names


def test_the_checked_reply_backs_but_not_the_lines_it_quotes():
    """The checked reply's own numbers back a note number; a line the app quotes ("> ...": the person's words) backs nothing,
    as the grade's _note_sources over _without_quotes reads it."""
    checked = "> In the 77 plots?\nMean: 2.5 over 12 plots."
    sources = nb.sources(claims=[], question="", extra=[], checked_text=checked)
    assert sorted(sources) == sorted(bench._note_sources([], bench._without_quotes(checked)))
    assert 12.0 in sources and 2.5 in sources and 77.0 not in sources


def test_the_sources_are_the_graders_whole_backing_of_a_note_number():
    """note_backing.sources is grade's backing of a note number: the claims' numbers and names, the question's numbers and names,
    the run's own numbers, and the checked reply's numbers and names without the lines it quotes (S2 included)."""
    question = "Compare MZP2017UMB-ST01 (studyDbId 4788) with MZP2016UMB-ST01?"
    checked = "> In MZP2015UMB-ST09 only?\nMean: 2.5 over the PEP2014CSR-ST03 plots."
    asked = [n for n, *_ in bench._numbers(question)] + bench.name_numbers(question)
    graded = bench._note_sources(bench.claim_numbers(CLAIMS) + bench.claim_names(CLAIMS) + asked + [7.0], bench._without_quotes(checked))
    sources = nb.sources(claims=CLAIMS, question=question, extra=[7.0], checked_text=checked)
    assert sorted(sources) == sorted(graded) and {2014.0, 2016.0, 2017.0} <= set(sources) and 2015.0 not in sources


def test_the_sources_take_the_claims_once():
    """Review MINOR 6: claims given as a one-shot iterable keep their names (sources reads them once)."""
    assert sorted(nb.sources(claims=iter(CLAIMS), question="", extra=[], checked_text="")) == sorted(
        nb.sources(claims=CLAIMS, question="", extra=[], checked_text=""))


def test_run_numbers_are_the_graders(tmp_path):
    """The run's own tables (every cell) and code-written fields (caveats, warnings, notes, the study ledger's IDs), with the
    model's quoted words left out and a planner draft's quoted parts blanked, as eval/benchmark.py run_numbers reads them."""
    run_dir = tmp_path / "run_20261025T140000_abcdef"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "artifacts" / "0001_observation_units.csv").write_text(
        "observationUnitDbId,observationUnitName,rep,block\n1276868,PEP2016CSR-ST03-00287,1,40\n1276869,PEP2016CSR-ST03-00595,2,7\n",
        encoding="utf-8")
    (run_dir / "answer.json").write_text(json.dumps({
        "analysis": {"caveats": ["numeric_summary: 951 of 960 values of MZP2017UMB-ST01 kept", "model: MZP2088UMB has 4143.48 of squares"]},
        "retrieval": {"warnings": ["get_observations: 960 units read", "model note: 12 studies were looked at"],
                      "study_ledger": [{"study_id": "4610"}, {"study_id": "mean=12.5"}, {"study_id": "MZP2066UMB"}]},
        "notes": ["coordinator: attempt 2 rejected: at most 12 items, not 14 ['99991']", "a plain note of 3 plots"]}), encoding="utf-8")
    assert sorted(nb.run_numbers(run_dir)) == sorted(bench.run_numbers([run_dir]))
    assert 4143.48 not in nb.run_numbers(run_dir) and 99991.0 not in nb.run_numbers(run_dir) and nb.run_numbers(tmp_path / "none") == []
    assert {2016.0, 2017.0, 287.0, 595.0} <= set(nb.run_numbers(run_dir))        # S2: the names in a cell and in a code-written field
    assert not {2088.0, 2066.0} & set(nb.run_numbers(run_dir))      # a model-quoted line's names, a name-like ledger ID: nothing
    assert nb.field_numbers(caveats=["model: the trial MZP2088UMB-ST09 was best"], warnings=["model note: MZP2077UMB"], notes=[],
                            ledger=[{"study_id": "MZP2066UMB"}, SimpleNamespace(study_id="MZP2055UMB")]) == []


def test_the_decisions_are_the_graders_on_real_looking_answers():
    """B26.4, C30.1, A36.3, D16.3 and A38.2 of 24-25 Oct: a note line the app drops is a line whose numbers the grade flags, and
    the reverse (eval/benchmark.py _note_sources and _backed at the precision written)."""
    no_40 = ([c for c in B264_CLAIMS if c["filters"].get("block") != "40"], [c for c in B264_CELLS if c != 40])
    cases = [(B264["question"], B264_ANSWER, B264_CLAIMS, B264_CELLS), (B264["question"], B264_ANSWER, *no_40),
             ("Compare the 90-day and 120-day trials.", "| Paired t | 7.3644 | 19 | 5.603e-07 |\n\n**Additional model notes "
              "(unverified)**\n\n- The paired t on entries is 7.36 (df 19, p about 5.6e-7), while Welch's p is 0.066.\n- p is about 5.8e-7.",
              [{"kind": "test", "value": v, "filters": {}, "transformation": t}
               for v, t in ((7.3644, "paired t"), (19, "df"), (5.603e-07, "p"), (0.0664, "Welch p"))], []),
             ("Mean yield of CIP199062.1 in PEP2016CSR-ST03 (studyDbId 4610)?", "Mean: 26.3185 over 3 plots.\n\n**Additional model notes "
              "(unverified)**\n\n- The 3 plots are PEP2016CSR-ST03-00287, -00595 and -00858.",
              [{"kind": "mean", "value": 26.3185185, "filters": {}, "transformation": "mean"}, {"kind": "count", "value": 3, "filters": {}}],
              []),
             ("Partition yield over the 5 regimes.", "Shares: reps within regime 0.2207 %, pooled error 14.3854 %.\n\n**Additional model "
              "notes (unverified)**\n\n- Added to pooled error it gives a single error term of 14.61%.\n- The 5 regimes are environments.",
              [{"kind": "ratio", "value": 0.2207, "filters": {}}, {"kind": "ratio", "value": 14.3854, "filters": {}}], []),
             ("How many PH17. clones, and how many plots each?", "Germplasm: 3292 distinct IDs.\n\n**Additional model notes (unverified)**"
              "\n\n- The preview shows only the first five clones, each with 1 plot.",
              [{"kind": "count", "value": 3292, "filters": {}}, {"kind": "count", "value": 1, "filters": {"germplasmDbId": "397805"}}], [])]
    for question, shown, claims, cells in cases:
        checked = bench._without_quotes(bench.harness.code_written(shown))
        asked = [n for n, *_ in bench._numbers(question)]
        graded = bench._note_sources(bench.claim_numbers(claims) + asked + [float(c) for c in cells], checked)
        sources = nb.sources(claims=claims, question=question, extra=[float(c) for c in cells], checked_text=checked)
        for line in bench.harness.model_notes(shown):
            grade_keeps = all(bench._backed(v, d, graded) for v, d, *_ in bench._numbers(line, written=True))
            assert nb.keeps(line, sources) is grade_keeps, line


MEAN_QUESTION = "In study 4501, what is the pooled mean of variable 76736?"


def mean():
    return Claim(claim_id="clm_presenter_mean", kind="mean", value=2.3785714285714286, unit=None,
                 denominator={"name": "valid numeric values, one per observation unit", "n": 140},
                 n_independent_units=140, source_artifact_hashes=["a" * 64],
                 transformation="mean of valid values; valid values used as recorded (one row per observation unit)")


def view(claims=None, caveats=(), *, status="completed", run_dir=None, facts=None):
    return SimpleNamespace(execution_status=status, run_dir=run_dir,             # as the controller: the question is the plan's
                           analysis=AnalysisReport(status=status, claims=claims or [], caveats=list(caveats),
                                                   metadata_facts=facts or []),
                           retrieval=RetrievalReport(status=status, complete=status == "completed"),
                           plan=SimpleNamespace(question=MEAN_QUESTION, statistic=SimpleNamespace(kind="mean")), ctx=None, notes=[])


KEPT = ["model: Source annotations require checking.", "model: The mean rests on 140 plots.", "model: Study 4501 only.",
        "model: The mean is about 2.38."]
BAD = "model: Growth stage differs across sites by 120 days."


def test_a_note_line_nothing_backs_is_dropped_and_the_view_changes_only_by_it():
    """140 is the claim's denominator, 4501 is the question's, 2.38 is the claim's mean at two decimals; 120 is backed by nothing."""
    with_bad, without = cp.answer_summary(view([mean()], [*KEPT, BAD])), cp.answer_summary(view([mean()], KEPT))
    assert with_bad == without
    assert "Additional model notes (unverified)" in with_bad and "140 plots" in with_bad and "Study 4501 only" in with_bad
    assert "120 days" not in with_bad and "about 2.38" in with_bad and "Source annotations require checking" in with_bad
    assert cp.dropped_notes(view([mean()], [*KEPT, BAD])) == [{"line": "Growth stage differs across sites by 120 days.", "numbers": [120]}]
    assert cp.dropped_notes(view([mean()], KEPT)) == []
    assert cp.dropped_notes(view([mean()], ["model: 140 plots over 120 days."])) == [{"line": "140 plots over 120 days.", "numbers": [120]}]


def test_when_every_note_line_is_dropped_the_notes_block_goes_with_them():
    shown = cp.answer_summary(view([mean()], [BAD, "model: 77 plots were checked."]))
    assert "Additional model notes" not in shown and shown == cp.answer_summary(view([mean()], []))


def test_the_runs_own_tables_and_the_checked_reply_back_a_note_number(tmp_path):
    run_dir = tmp_path / "run_20261025T140000_abcdef"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "artifacts" / "0001_observations.csv").write_text("observationUnitDbId,value\n77,2.3\n", encoding="utf-8")
    note = "model: 77 plots were checked."
    assert "77 plots were checked" in cp.answer_summary(view([mean()], [note], run_dir=run_dir))
    assert "77 plots were checked" not in cp.answer_summary(view([mean()], [note]))
    limit, counted = "numeric_summary: 9 values were left out as non-numeric.", "model: The 9 values were left out."
    assert "The 9 values were left out" in cp.answer_summary(view([mean()], [limit, counted]))   # the answer's limits show 9
    assert "The 9 values were left out" not in cp.answer_summary(view([mean()], [counted]))


def test_a_claims_own_text_backs_a_note_number():
    """10.2685 is in the claim's method text only (the chat shows no method): it backs "10.27"."""
    claim = mean().model_copy(update={"transformation": "mean of valid values above the threshold 10.2685"})
    note = "model: The threshold is 10.27."
    assert "The threshold is 10.27" in cp.answer_summary(view([claim], [note]))
    assert "The threshold is 10.27" not in cp.answer_summary(view([mean()], [note]))


def test_identifier_endings_and_the_years_of_named_studies_keep_their_note_lines(tmp_path):
    """S2 (decision 2026-10-25 14:35 UTC): A36.3's line stays when the run's table holds the plots; E18.2's line stays when the
    question names the studies; with neither, both are dropped, with the numbers nothing backs."""
    run_dir = tmp_path / "run_20261025T013700_abcdef"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "artifacts" / "0001_observations.csv").write_text(
        "observationUnitName,value\nPEP2016CSR-ST03-00287,20.1\nPEP2016CSR-ST03-00595,27.7\nPEP2016CSR-ST03-00858,31.2\n", encoding="utf-8")
    plots = "model: The 140 plots include PEP2016CSR-ST03-00287, -00595 and -00858."
    assert cp.dropped_notes(view([mean()], [plots], run_dir=run_dir)) == []
    assert cp.dropped_notes(view([mean()], [plots])) == [{"line": plots[len("model: "):], "numbers": [287, 595, 858]}]
    years = "model: Plot names differ between the studies (00001 is Jonathan in 2016 and Esther in 2017)."
    plots = tmp_path / "run_20261025T040304_bcd548"                   # as in the run: its tables hold the plot MZP2016UMB-ST01-00001
    (plots / "artifacts").mkdir(parents=True)
    (plots / "artifacts" / "0001_observations.csv").write_text("observationUnitName,value\nMZP2016UMB-ST01-00001,30.1\n", encoding="utf-8")
    named = view([mean()], [years], run_dir=plots)
    named.plan.question = "Compare MZP2017UMB-ST01 (studyDbId 4788) with MZP2016UMB-ST01 (studyDbId 4777)."
    assert "Esther in 2017" in cp.answer_summary(named) and cp.dropped_notes(named) == []
    assert cp.dropped_notes(view([mean()], [years])) == [{"line": years[len("model: "):], "numbers": [1, 2016, 2017]}]


def test_an_incomplete_answer_drops_its_unverified_note_paragraphs_too():
    controller = view([], ["no complete artifact is eligible for analysis", "model: 99 rows were cut.", "model: The study has no plots."],
                      status="incomplete")
    shown = cp.answer_summary(controller)
    assert "Unverified model note: The study has no plots." in shown and "99 rows" not in shown
    assert cp.dropped_notes(controller) == [{"line": "99 rows were cut.", "numbers": [99]}]


def test_a_model_service_failure_drops_them_too():
    controller = view([], ["model: 99 rows were cut.", "model: The study has no plots."], status="failed")
    controller.notes = ["agent error provider_quota: The provider's quota is exhausted"]
    shown = cp.answer_summary(controller)
    assert "Model service unavailable" in shown and "Unverified model note: The study has no plots." in shown and "99 rows" not in shown
    assert cp.dropped_notes(controller) == [{"line": "99 rows were cut.", "numbers": [99]}]


def test_a_metadata_answer_drops_them_too():
    facts = [MetadataFact(entity_kind="study", entity_id="S1", entity_label="SYNTHETIC trial S1", field="studyDescription",
                          value="A synthetic trial.", source_artifact_id="art_meta_0001", source_artifact_hash="b" * 64)]
    controller = view([], ["model: The trial ran 120 days.", "model: The study record names no end date."], facts=facts)
    shown = cp.answer_summary(controller)
    assert "names no end date" in shown and "120 days" not in shown
    assert cp.dropped_notes(controller) == [{"line": "The trial ran 120 days.", "numbers": [120]}]


def test_the_manifest_records_the_dropped_lines():
    """run's record writes them into manifest.json as "dropped_note_lines" (decision (31)(b); coach 13:48 UTC): the count, and
    each dropped line with the numbers nothing backs, as dropped_notes gives them."""
    record = ctl.dropped_notes_record(view([mean()], [*KEPT, BAD]))
    assert record == {"count": 1, "lines": [{"line": "Growth stage differs across sites by 120 days.", "numbers": [120]}]}
    assert json.dumps(record["lines"][0]["numbers"]) == "[120]"
    assert ctl.dropped_notes_record(view([mean()], KEPT)) == {"count": 0, "lines": []}


def test_a_check_that_cannot_run_leaves_out_what_it_cannot_back(monkeypatch):
    """Fail closed (review 2026-10-25, finding 5): the claims, the question and the answer cannot be read, so a line with a
    number is left out unless the run's own numbers back it; a line without a number stays; the record names the error."""
    def broken(*args, **kwargs):
        raise OSError("claims unreadable")
    monkeypatch.setattr(cp.note_backing, "sources", broken)
    shown = cp.answer_summary(view([mean()], [*KEPT, BAD]))
    assert "Source annotations require checking" in shown and "140 plots" not in shown and "120 days" not in shown
    record = ctl.dropped_notes_record(view([mean()], [*KEPT, BAD]))
    assert record["count"] == 4 and record["error"] == "OSError: claims unreadable" and record["lines"][-1]["numbers"] == [120]


def test_when_the_runs_own_numbers_cannot_be_read_only_the_lines_they_would_back_are_left_out(monkeypatch, tmp_path):
    def broken(run_dir):
        raise OSError("table unreadable")
    monkeypatch.setattr(cp.note_backing, "table_numbers", broken)
    controller = view([mean()], [BAD, *KEPT, "model: 77 plots were checked."], run_dir=tmp_path)   # the read fails at the first line
    shown = cp.answer_summary(controller)
    assert "140 plots" in shown and "about 2.38" in shown and "77 plots" not in shown and "120 days" not in shown
    record = ctl.dropped_notes_record(controller)
    assert [line["numbers"] for line in record["lines"]] == [[120], [77]] and record["error"] == "OSError: table unreadable"


def test_a_line_the_reader_cannot_read_is_left_out(monkeypatch):
    real = cp.note_backing.unbacked
    monkeypatch.setattr(cp.note_backing, "unbacked", lambda note, backing: 1 / 0 if "120" in note else real(note, backing))
    record = ctl.dropped_notes_record(view([mean()], [*KEPT, BAD]))
    assert record["lines"] == [{"line": "Growth stage differs across sites by 120 days.", "numbers": None}]
    assert record["count"] == 1 and record["error"] == "ZeroDivisionError: division by zero"


def test_a_number_out_of_float_range_is_recorded_as_written():
    """Review finding 1: a run of 310 digits reads as inf; the record keeps it as written, and stays valid JSON."""
    huge = "1" + "0" * 309
    record = ctl.dropped_notes_record(view([mean()], [f"model: Plot code {huge} was seen."]))
    assert record == {"count": 1, "lines": [{"line": f"Plot code {huge} was seen.", "numbers": [huge]}]}
    json.dumps(record, allow_nan=False)


def test_a_record_that_would_not_be_json_is_the_error_form(monkeypatch):
    """The record is checked to be JSON inside its guard: a value no JSON holds turns it into the error form."""
    monkeypatch.setattr(cp, "dropped_notes", lambda controller: [{"line": "x", "numbers": [float("nan")]}])
    record = ctl.dropped_notes_record(view([mean()], [BAD]))
    assert record["count"] is None and record["lines"] == [] and record["error"].startswith("ValueError: Out of range float")


def test_numbers_no_float_holds_are_recorded_as_written():
    """Review 2, findings 5 and 7: -inf (a negative run of 310 digits) and 1e-400 (read as 0) keep their written form; 1.5e300 is a
    float, not a 301-digit integer."""
    minus = "-1" + "0" * 309
    record = ctl.dropped_notes_record(view([mean()], [f"model: Offsets {minus}, p 1e-400 and sum 1.5e300."]))
    assert record["lines"][0]["numbers"] == [minus, "1e-400", 1.5e300]
    assert json.dumps(record["lines"][0]["numbers"][1:], allow_nan=False) == '["1e-400", 1.5e+300]'      # a float, not 301 digits


def test_every_report_field_is_read_from_memory_as_the_grader_reads_it_from_the_file():
    """Review 2, finding 5: the caveats, warnings, notes and study ledger (records here, dicts in answer.json) read from the
    controller give the numbers note_backing.field_numbers gives for answer.json."""
    controller = view([mean()], ["numeric_summary: 951 of 960 values kept", "model: 4143.48 is a sum of squares"])
    controller.retrieval = RetrievalReport(status="incomplete", complete=False,
                                           warnings=["get_observations: 77 units read", "model note: 12 studies were looked at"],
                                           study_ledger=[StudyStatusEntry(study_id="4610", status="blocked", reason="not approved"),
                                                         StudyStatusEntry(study_id="mean=12.5", status="blocked", reason="not approved")])
    controller.notes = ["coordinator: attempt 2 rejected: at most 12 items, not 14 ['99991']", "a plain note of 3 plots"]
    memory = cp._field_numbers(controller)
    saved = json.loads(json.dumps({"analysis": controller.analysis.model_dump(mode="json"),
                                   "retrieval": controller.retrieval.model_dump(mode="json"), "notes": controller.notes}))
    from_file = nb.field_numbers(caveats=saved["analysis"]["caveats"], warnings=saved["retrieval"]["warnings"], notes=saved["notes"],
                                 ledger=saved["retrieval"]["study_ledger"])
    assert sorted(memory) == sorted(from_file) == [2, 3, 12, 14, 77, 951, 960, 4610]


def test_an_unreadable_table_leaves_the_report_fields(monkeypatch, tmp_path):
    """Review 2, finding 4: the tables and the report fields are read apart; a table that cannot be read leaves the fields."""
    def broken(run_dir):
        raise OSError("table unreadable")
    monkeypatch.setattr(cp.note_backing, "table_numbers", broken)
    controller = view([mean()], ["model: Study 4620 was also read."], run_dir=tmp_path)
    controller.notes = ["coordinator: scope.study_ids 4620"]
    assert "Study 4620 was also read" in cp.answer_summary(controller)
    assert ctl.dropped_notes_record(controller) == {"count": 0, "lines": [], "error": "OSError: table unreadable"}


def test_every_label_is_taken_out_and_every_word_of_the_person_stays():
    seven = mean().model_copy(update={"value": 7.25})          # the mean 2.3786 would back a "2" written to no decimals
    controller = view([seven], ["model: 2 answers were used.", "model: The 2025 values were added.", "model: Variable 76619 was used."])
    controller.plan.question = ("Give the mean yield of the trial.\nClarification 1 from the person: variable 76619\nClarification 2 "
                                "from the person: the whole trial\nFollow-up from the person: and in 2025?")
    assert cp.dropped_notes(controller) == [{"line": "2 answers were used.", "numbers": [2]}]


def test_the_record_joins_every_error_and_keeps_the_line_as_the_person_reads_it(monkeypatch, tmp_path):
    def broken_sources(*args, **kwargs):
        raise OSError("claims unreadable")

    def broken_tables(run_dir):
        raise OSError("table unreadable")
    monkeypatch.setattr(cp.note_backing, "sources", broken_sources)
    monkeypatch.setattr(cp.note_backing, "table_numbers", broken_tables)
    record = ctl.dropped_notes_record(view([mean()], ["model: Growth stage of plot_7 and *row* 3 differs by 120 days."], run_dir=tmp_path))
    assert record["error"] == "OSError: claims unreadable; OSError: table unreadable"
    assert record["lines"] == [{"line": "Growth stage of plot_7 and *row* 3 differs by 120 days.", "numbers": [3, 120]}]


def full_report(tmp_path, caveats):
    controller = ctl.Controller(part2_config(question=MEAN_QUESTION, offline=True, mock_model=True, direct=True, fixture="synthetic",
                                             out_dir=tmp_path / "out", cache_dir=tmp_path / "cache"))
    controller.analysis = AnalysisReport(status="completed", claims=[mean()], caveats=caveats)
    controller.retrieval = RetrievalReport(status="completed", complete=True)
    controller.render_phase()
    return controller, controller.answer_markdown().split("## Raw model notes (unverified)", 1)[1].split("## Notes", 1)[0]


def test_the_full_report_keeps_every_raw_note_and_marks_the_ones_the_answer_leaves_out(tmp_path):
    """Coach 2026-10-25 15:48 UTC: the Full report (and the console, which prints it) keeps every model note under "Raw model
    notes (unverified)"; each line the answer leaves out says so, with its numbers. The answer never shows it."""
    controller, raw = full_report(tmp_path, ["numeric_summary: the growth stage is not stated.", "model: The mean rests on 140 plots.", BAD,
                                               "model: plot_7 had 3 rows."])
    assert f"- Analyst note: {BAD} (left out of the answer: 120 unbacked)" in raw
    assert "- Analyst note: model: plot_7 had 3 rows. (left out of the answer: 3 unbacked)" in raw
    assert "- Analyst note: model: The mean rests on 140 plots.\n" in raw.replace("\r\n", "\n")
    assert raw.count("left out of the answer") == 2 and "120 days" not in cp.answer_summary(controller)


def test_the_full_report_says_when_it_cannot_tell(tmp_path, monkeypatch):
    real = cp.note_backing.unbacked
    monkeypatch.setattr(cp.note_backing, "unbacked", lambda note, backing: 1 / 0 if "120" in note else real(note, backing))
    _controller, raw = full_report(tmp_path, ["model: The mean rests on 140 plots.", BAD])
    assert f"- Analyst note: {BAD} (left out of the answer: its numbers could not be checked)" in raw
    def broken(controller):
        raise RuntimeError("no view")
    monkeypatch.setattr(cp, "dropped_notes", broken)
    _controller, raw = full_report(tmp_path / "again", ["model: The mean rests on 140 plots.", BAD])
    assert raw.count("(whether the answer shows it could not be determined)") == 2


def test_the_persons_words_back_a_note_number_and_the_code_labels_do_not():
    """Review finding 4: the plan's question carries the code's "Clarification N from the person:" labels; their N backs
    nothing, the person's own answer does."""
    controller = view([mean()], ["model: 1 trial was excluded as an outlier.", "model: Variable 76619 was used."])
    controller.plan.question = "Give the mean yield of the trial.\nClarification 1 from the person: variable 76619"
    assert cp.dropped_notes(controller) == [{"line": "1 trial was excluded as an outlier.", "numbers": [1]}]


def test_the_runs_report_fields_back_a_note_number_before_answer_json_exists(tmp_path):
    """Review finding 2: the answer is shown before record() writes answer.json; the run's code-written fields are read from the
    controller, so the chat and the record decide alike."""
    controller = view([mean()], ["model: Study 4620 was also read."], run_dir=tmp_path)
    controller.notes = ["coordinator: scope.study_ids 4620"]
    assert not (tmp_path / "answer.json").exists()
    assert "Study 4620 was also read" in cp.answer_summary(controller)
    assert ctl.dropped_notes_record(controller) == {"count": 0, "lines": []}
    controller.notes = []
    assert "Study 4620 was also read" not in cp.answer_summary(controller)


def test_the_limits_of_a_metadata_answer_and_the_reasons_of_an_unfinished_one_back_too():
    """Numbers only the answer's own text shows (no claim, question, table or report field holds them): an analysis-level
    exclusion's count in a metadata answer's limits, and a refused study's ID in an unfinished answer's reasons."""
    facts = [MetadataFact(entity_kind="study", entity_id="S1", entity_label="SYNTHETIC trial S1", field="studyDescription",
                          value="A synthetic trial.", source_artifact_id="art_meta_0001", source_artifact_hash="b" * 64)]
    controller = view([], ["model: 7 rows had no value."], facts=facts)
    controller.analysis = controller.analysis.model_copy(update={"exclusions": [Exclusion(reason="rows without a value", count=7)]})
    shown = cp.answer_summary(controller)
    assert "Excluded: 7" in shown and "7 rows had no value" in shown
    assert "7 rows had no value" not in cp.answer_summary(view([], ["model: 7 rows had no value."], facts=facts))
    controller = view([], ["model: The fetch for study 123 was refused."], status="incomplete")
    controller.retrieval = RetrievalReport(status="incomplete", complete=False,
                                           study_ledger=[StudyStatusEntry(study_id="S-123", status="blocked", reason="not approved")])
    shown = cp.answer_summary(controller)
    assert "Study S" in shown and "Unverified model note: The fetch for study 123 was refused." in shown
    assert "study 123 was refused" not in cp.answer_summary(view([], ["model: The fetch for study 123 was refused."], status="incomplete"))


def test_the_checked_text_is_read_as_the_person_reads_it():
    assert 2.5 in nb.sources(claims=[], question="", extra=[], checked_text="Mean: 2\\.5 \\(unit not stated\\)")


def test_a_failing_answer_leaves_the_record_its_error(monkeypatch):
    def broken(controller):
        raise RuntimeError("no view")
    monkeypatch.setattr(cp, "dropped_notes", broken)
    assert ctl.dropped_notes_record(view([mean()], [BAD])) == {"count": None, "lines": [], "error": "RuntimeError: no view"}


def test_record_writes_the_dropped_lines_into_the_manifest(place, monkeypatch):
    """The record step itself, end to end on the synthetic pipeline (the benchmark runner's setup; no provider, no database):
    every run's manifest.json carries "dropped_note_lines", as dropped_notes_record gives it for that run."""
    counts = bench.harness.reply_tokens
    monkeypatch.setattr(bench.harness, "reply_tokens", lambda model, usage, detail: counts(model, usage, detail) or
                        {"model": model, "input": 0, "cache_write": 0, "cache_read": 0, "output": 0})
    groups = cp._caveat_groups        # the scripted analyst writes no note: one is added, so the answer has a line to leave out
    monkeypatch.setattr(cp, "_caveat_groups", lambda controller, claims: (lambda found: (found[0], [*found[1], BAD[len("model: "):]]))(
        groups(controller, claims)))
    build = ctl.Controller.build_result     # review 2, finding 2: a note build_result adds after the answer was shown backs nothing
    monkeypatch.setattr(ctl.Controller, "build_result", lambda self: self.notes.append("RunResult contract refused: 120 checks") or build(self))
    seen = []
    real = ctl.dropped_notes_record
    monkeypatch.setattr(ctl, "dropped_notes_record", lambda controller: seen.append(controller) or real(controller))
    place.seed(*FIXTURES)
    sub = {"sub_id": "S1.1", "parent": "S1", "key_facts": [0], "tier": "T1", "question": MEASURED,
           "expected": [{"value": 18.0, "kind": "mean", "decimals": 0}], "expected_text": [], "tolerance": {"rule": "relative", "rel": 0.005},
           "alternatives": [], "studies": [], "folders": [], "variables": [], "statistic": "pooled mean", "caveat": None, "gold_basis": "test"}
    path = place.root / "subs.json"
    path.write_text(json.dumps({"subquestions": [sub]}), encoding="utf-8")
    _model, front_door = scripted_front_door("data_question")
    assert bench.main(["--cache", str(place.source), "--subquestions", str(path), "--results", str(place.root / "results"), "--tiers", "T1",
                       "--confirm"], base_config=place.base_config, make_models=synthetic_models, triage_model=front_door,
                      preflight=lambda q: place.base_config(q).validate()) == 0
    [manifest] = list((place.root / "results").glob("benchmark_*/out/run_*/manifest.json"))
    recorded = json.loads(manifest.read_text(encoding="utf-8"))["dropped_note_lines"]
    assert seen and recorded == {"count": 1, "lines": [{"line": BAD[len("model: "):], "numbers": [120]}]}
    [results] = list((place.root / "results").glob("benchmark_*/results.json"))
    [row] = json.loads(results.read_text(encoding="utf-8"))["subquestions"]
    assert "Growth stage differs" not in row["answer_full"] and "18" in row["answer_full"]      # what the chat showed: the line is out
