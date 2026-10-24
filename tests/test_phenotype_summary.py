"""
tests/test_phenotype_summary.py - numeric_summary with the columns of Breedbase's Phenotype Summary (coach 2026-10-22 22:50 UTC,
NEXT 3; research report: "extend the existing entry/trial means with min, max, sample SD, raw CV (SD/mean x 100) and percent
missing"). Mean, min, max, sample SD and count were there; added: the CV of the values and, with the study's plot list, the
percent of its plots without a valid value. A measurements table holds only recorded values (/observations and the plot records
alike), so percent missing is never computed from the table alone. SYNTHETIC tables; no network, no model.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agents.analyst as an
import agents.coordinator as co
import analyst_tools as at
import brapi_mcp_server as srv
import chat_presenter as cp
from artifacts import ArtifactRegistry
from contracts import AnalysisReport, Claim, MeasurementMeta
from part2_support import SYNTHETIC_BASE, synthetic_context

CV_FOUR = 43.508667127669376          # values 10, 14, 20, 28: mean 18, sample SD 7.831560082980487
# REAL data stays git-ignored (ACCESS_NOTES: until the redistribution question is answered): snapshots/, absent from a clone
FIXTURE_4501 = Path(__file__).resolve().parents[1] / "snapshots" / "spb_4501_fusarium_20261022.json"


def real_4501() -> dict:
    if not FIXTURE_4501.is_file():
        pytest.skip("the REAL snapshot is git-ignored and not in this copy (ACCESS_NOTES)")
    return json.loads(FIXTURE_4501.read_text(encoding="utf-8"))


@pytest.fixture
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "out", "run_ps", base_url=SYNTHETIC_BASE)


def measurements(registry: ArtifactRegistry, values, *, study="S1", complete=True) -> str:
    """values: (plot ID, value text[, date]); a plot may repeat (several values of one plot, or one per date)."""
    rows = [{"observationDbId": f"O{i}", "observationUnitDbId": item[0], "observationUnitName": f"plot {item[0]}", "germplasmDbId": "G1",
             "germplasmName": "clone G1", "studyDbId": study, "observationVariableDbId": "V1", "observationVariableName": "SYNTHETIC yield",
             "observationTimeStamp": item[2] if len(item) > 2 else "", "value": item[1]} for i, item in enumerate(values)]
    return registry.save_table(rows, kind="observations", columns=dict(srv.OBSERVATION_COLUMNS), complete=complete, study_ids=[study],
                               variable_ids=["V1"], measurement=MeasurementMeta(trait="SYNTHETIC yield", unit="t/ha"),
                               quality_notes=["SYNTHETIC"]).artifact_id


def plot_list(registry: ArtifactRegistry, units, *, study="S1", complete=True, kind="observation_units") -> str:
    """units: plot IDs (level plot), or (unit ID, level name) pairs."""
    rows = []
    for item in units:
        unit, level = (item, "plot") if isinstance(item, str) else item
        rows.append({"observationUnitDbId": unit, "observationUnitName": f"unit {unit}", "germplasmDbId": "G1", "germplasmName": "clone G1",
                     "studyDbId": study, "levelName": level, "levelCode": unit, "observationUnitPosition": ""})
    return registry.save_table(rows, kind=kind, columns=dict(srv.UNIT_COLUMNS), complete=complete, study_ids=[study],
                               quality_notes=["SYNTHETIC"]).artifact_id


def summary(registry, artifact_id, **args):
    return at.dispatch_analyst(registry, "numeric_summary", {"artifact_id": artifact_id, **args})


def claims(r) -> list[Claim]:
    return [Claim.model_validate(k) for k in r.data["claims"]]


def cv_claim(r) -> Claim:
    return next(k for k in claims(r) if k.kind == "ratio" and k.transformation.startswith("cv of valid values"))


def missing_claims(r) -> list[Claim]:
    return [k for k in claims(r) if k.kind == "ratio" and k.transformation.startswith("percent missing")]


FOUR = [("P1", "10"), ("P2", "14"), ("P3", "20"), ("P4", "28")]


# --- 1. CV = 100 x sample SD / mean -----------------------------------------------------------------

def test_cv_is_the_sample_sd_over_the_mean_times_100(registry):
    r = summary(registry, measurements(registry, FOUR))
    assert r.ok, r.error
    k = cv_claim(r)
    mean = next(c for c in claims(r) if c.kind == "mean")
    assert k.value == pytest.approx(CV_FOUR, rel=1e-12) and k.unit == "%"
    assert k.denominator == mean.denominator and k.n_independent_units == mean.n_independent_units and k.exclusions == mean.exclusions
    assert r.data["cv_percent"] == pytest.approx(CV_FOUR, rel=1e-12) and set(r.data["stats"]) == {"mean", "median", "min", "max", "sd"}


def test_cv_uses_the_values_the_other_statistics_use(registry):
    """Raw-missing and invalid tokens are left out; under mean_per_unit one value per plot (the plot's mean)."""
    r = summary(registry, measurements(registry, [*FOUR, ("P5", ""), ("P6", "bad")]))
    assert cv_claim(r).value == pytest.approx(CV_FOUR, rel=1e-12)
    rows = [("P1", "8"), ("P1", "12"), ("P2", "14"), ("P3", "20"), ("P4", "28")]
    assert not summary(registry, measurements(registry, rows)).ok
    r = summary(registry, measurements(registry, rows), duplicate_policy="mean_per_unit")
    assert r.ok and cv_claim(r).value == pytest.approx(CV_FOUR, rel=1e-12) and cv_claim(r).denominator.n == 4


def test_cv_is_not_available_without_a_positive_mean_or_two_values(registry):
    for rows, reason in (([("P1", "-1"), ("P2", "1")], "the mean is not positive"), ([("P1", "-3"), ("P2", "-1")], "the mean is not positive"),
                         ([("P1", "5")], "fewer than 2 valid values"), ([("P1", ""), ("P2", "x")], "no valid numeric values")):
        k = cv_claim(summary(registry, measurements(registry, rows)))
        assert k.value is None and reason in k.missing_reason


# --- 2. percent missing needs the plot list --------------------------------------------------------------

def test_without_the_plot_list_no_percent_missing_is_made(registry):
    r = summary(registry, measurements(registry, FOUR))
    assert r.ok and missing_claims(r) == [] and r.data["percent_missing"] is None


def test_percent_missing_counts_the_plot_lists_plots_without_a_valid_value(registry):
    """8 plots listed: 5 with a valid value, P6 with an empty value, P7 and P8 absent from the table -> 3 of 8 = 37.5 %."""
    obs = measurements(registry, [*FOUR, ("P5", "12"), ("P6", "")])
    r = summary(registry, obs, roster_artifact_id=plot_list(registry, [f"P{i}" for i in range(1, 9)]))
    assert r.ok, r.error
    (k,) = missing_claims(r)
    assert k.value == pytest.approx(37.5, rel=1e-12) and k.unit == "%" and k.denominator.n == 8 and "plot list" in k.denominator.name
    pm = r.data["percent_missing"]
    assert (pm["plots_listed"], pm["with_valid_value"], pm["without_valid_value"], pm["rows_not_on_list"]) == (8, 5, 3, 0)
    assert pm["value"] == pytest.approx(37.5, rel=1e-12) and pm["reason"] is None
    assert next(c for c in claims(r) if c.kind == "count").value == 5          # Count stays the valid values used


def test_percent_missing_through_the_real_pipeline_matches_missing_report(tmp_path):
    """The teaching study S1: 6 plots, 4 with a valid value (one empty, one 'bad') -> 2 of 6."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_ps_s1")
    obs = srv.dispatch(ctx, "get_observations", {"study_db_id": "S1", "variable_db_id": "V1"}).artifact_ids[0]
    units = srv.dispatch(ctx, "get_observation_units", {"study_db_id": "S1"}).artifact_ids[0]
    r = summary(ctx.registry, obs, roster_artifact_id=units)
    report = at.dispatch_analyst(ctx.registry, "missing_report", {"artifact_id": obs, "roster_artifact_id": units})
    assert r.ok and report.ok
    expected = report.data["expected_units"]
    without = expected - report.data["units"]["with_valid_value"]
    (k,) = missing_claims(r)
    assert (expected, without) == (6, 2) and k.value == pytest.approx(100 * without / expected, rel=1e-12)


def test_a_plot_with_several_values_counts_once(registry):
    rows = [("P1", "8"), ("P1", "12"), ("P2", "14"), ("P3", "20")]
    r = summary(registry, measurements(registry, rows), duplicate_policy="mean_per_unit", roster_artifact_id=plot_list(registry, ["P1", "P2", "P3", "P4"]))
    assert r.ok and missing_claims(r)[0].value == pytest.approx(25.0, rel=1e-12)


def test_a_plot_measured_on_two_dates_counts_once(registry):
    """Two dates are not duplicates, so numeric_summary keeps both rows; the plot still counts once for percent missing."""
    rows = [("P1", "8", "2026-01-05"), ("P1", "12", "2026-02-05"), ("P2", "14", ""), ("P3", "20", "")]
    r = summary(registry, measurements(registry, rows), roster_artifact_id=plot_list(registry, ["P1", "P2", "P3", "P4"]))
    assert r.ok and r.data["counts"]["values_used"] == 4 and r.data["percent_missing"]["with_valid_value"] == 3
    assert missing_claims(r)[0].value == pytest.approx(25.0, rel=1e-12)


def test_a_plot_off_the_list_makes_percent_missing_not_available(registry):
    """The coach's guard (2026-10-22 23:50) and Bidhan's (15)(a): a measured plot that is not on the plot list means the list is
    not the table's, so percent missing is not available; the row stays in the other statistics. The warning counts valid
    values only (review finding 8: 'NA' and 'bad' are in no statistic)."""
    r = summary(registry, measurements(registry, [*FOUR, ("Q9", "16"), ("Q8", "NA"), ("Q7", "bad")]),
                roster_artifact_id=plot_list(registry, ["P1", "P2", "P3", "P4", "P5"]))
    (k,) = missing_claims(r)
    assert r.ok and k.value is None and "not on the plot list" in k.missing_reason and r.data["percent_missing"]["value"] is None
    assert (r.data["percent_missing"]["rows_not_on_list"], r.data["percent_missing"]["valid_values_not_on_list"]) == (3, 1)
    assert next(c for c in claims(r) if c.kind == "count").value == 5
    assert [w for w in r.warnings if "not on the plot list" in w] == [
        f"1 valid value(s) on plots not on the plot list {r.artifact_ids[1]}: kept in the statistics; "
        "percent missing not available"]


def test_only_the_level_of_the_measured_units_counts(registry):
    """A plot list with plots and plants: values on plots -> the plots are the denominator; values on both -> not available."""
    units = ["P1", "P2", "P3", "P4", ("P1-a", "plant"), ("P1-b", "plant")]
    r = summary(registry, measurements(registry, [("P1", "10"), ("P2", "14"), ("P3", "20")]), roster_artifact_id=plot_list(registry, units))
    (k,) = missing_claims(r)
    assert k.value == pytest.approx(25.0, rel=1e-12) and k.denominator.n == 4 and "plot" in k.denominator.name
    r = summary(registry, measurements(registry, [("P1", "10"), ("P1-a", "14"), ("P2", "20")]), roster_artifact_id=plot_list(registry, units))
    (k,) = missing_claims(r)
    assert k.value is None and "several levels" in k.missing_reason


def test_percent_missing_is_not_available_when_its_count_of_plots_is_unknown(registry):
    obs = measurements(registry, FOUR)
    (k,) = missing_claims(summary(registry, obs, roster_artifact_id=plot_list(registry, ["P1", "P2", "P3", "P4", "P5"], complete=False)))
    assert k.value is None and "incomplete" in k.missing_reason
    (k,) = missing_claims(summary(registry, obs, roster_artifact_id=plot_list(registry, ["X1", "X2"])))
    assert k.value is None and "not on the plot list" in k.missing_reason
    r = summary(registry, measurements(registry, [("P1", ""), ("P2", "bad")]), roster_artifact_id=plot_list(registry, ["P1", "P2", "P3"]))
    (k,) = missing_claims(r)
    assert k.value == pytest.approx(100.0, rel=1e-12)                        # no valid value at all: every listed plot is missing


def test_a_wrong_plot_list_is_refused(registry):
    obs = measurements(registry, FOUR)
    r = summary(registry, obs, roster_artifact_id=plot_list(registry, ["P1"], study="S2"))
    assert not r.ok and r.error.code == "invalid_argument" and "S2" in r.error.message
    r = summary(registry, obs, roster_artifact_id=obs)
    assert not r.ok and r.error.code == "invalid_argument" and "observation_units" in r.error.message


def test_missing_report_keeps_its_checks(registry):
    """The plot-list checks moved to one helper; missing_report's refusals stay as they were (an incomplete list is refused
    there, where numeric_summary says the number of plots is unknown)."""
    obs = measurements(registry, FOUR)

    def report(roster):
        return at.dispatch_analyst(registry, "missing_report", {"artifact_id": obs, "roster_artifact_id": roster})

    r = report(plot_list(registry, ["P1", "P2", "P3", "P4", "P5"], complete=False))
    assert not r.ok and r.error.code == "incomplete_data" and "not a complete collection" in r.error.message
    r = report(plot_list(registry, ["P1"], study="S2"))
    assert not r.ok and r.error.code == "invalid_argument" and "differ from the table's" in r.error.message
    r = report(obs)
    assert not r.ok and r.error.code == "invalid_argument" and "not observation_units" in r.error.message
    r = report(plot_list(registry, ["P1", "P2", "P3", "P4", "P5"]))
    assert r.ok and r.data["expected_units"] == 5 and r.data["units"]["with_valid_value"] == 4


def test_study_4501_matches_sweetpotatobases_phenotype_summary(registry):
    """REAL data: study 4501's Fusarium wilt scores (177 of its 178 plots scored, 37 of them 0) against the Phenotype Summary row
    SweetPotatoBase shows for the trial: mean 1.88, min 0.00, max 4.00, SD 1.20, count 177 and percent missing 0.56% match at its
    rounding. Its CV, 63.96%, divides the SD by the mean already rounded to 1.88; the exact CV is 63.92%."""
    doc = real_4501()
    row = doc["breedbase_phenotype_summary"]["row"]
    obs = measurements(registry, [tuple(plot) for plot in doc["plots"]], study="4501")
    r = summary(registry, obs, roster_artifact_id=plot_list(registry, doc["plot_list"], study="4501"))
    assert r.ok, r.error
    s, pm = r.data["stats"], r.data["percent_missing"]
    assert [f"{s[k]:.2f}" for k in ("mean", "min", "max", "sd")] == row[1:5] and r.data["counts"]["values_used"] == row[6] == 177
    assert f"{pm['value']:.2f}%" == row[7] and (pm["plots_listed"], pm["without_valid_value"]) == (178, 1)
    assert f"{r.data['cv_percent']:.2f}%" == "63.92%" and f"{100 * s['sd'] / round(s['mean'], 2):.2f}%" == row[5] == "63.96%"

# --- 3. the answer and the wiring ---------------------------------------------------------------------------

def answer(ctx, r, question="") -> str:
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claims(r), caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[],
                           plan=SimpleNamespace(statistic=SimpleNamespace(kind="mean"), question=question))
    return cp.answer_summary(view)


def test_the_answer_shows_the_cv_row_and_the_plots_without_a_value(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_ps_answer")
    obs = measurements(ctx.registry, [*FOUR, ("P5", "")])
    r = summary(ctx.registry, obs, roster_artifact_id=plot_list(ctx.registry, ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]))
    shown = answer(ctx, r, "Summarize yield in this trial")
    assert "| Sample standard deviation | 7.83156008298 t/ha |\n| CV \\(sample SD as % of the mean\\) | 43.5086671277 % |\n| Count | 4 |" in shown
    # two sources (the measurements, and the plot list for percent missing), so each block names its source
    assert "**Summary · source 1**" in shown and "**Plots without a value · source 2**" in shown and "| Percent missing | 50 % |" in shown
    assert "Based on plots on the study's plot list: 8. Recorded analysis units: 4." in shown and "\\(4 of 8;" in shown


def test_a_question_for_the_mean_and_the_cv_keeps_the_cv(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_ps_focus")
    r = summary(ctx.registry, measurements(ctx.registry, FOUR))
    assert "43.5086671277 %" in answer(ctx, r, "What is the mean and the CV of yield?")
    assert "43.5086671277 %" in answer(ctx, r, "What is the mean yield and the percent missing?")
    assert "43.5086671277 %" not in answer(ctx, r, "What is the mean yield?")


def test_the_prompts_and_the_schema_name_the_plot_list():
    schema = next(s for s in at.analyst_tool_schemas() if s["name"] == "numeric_summary")
    assert "roster_artifact_id" in schema["inputSchema"]["properties"]
    assert "Percent missing" in an.ANALYST_SYSTEM_PROMPT and "numeric_summary(artifact_id=" in co.COORDINATOR_SYSTEM_PROMPT
    assert "roster_artifact_id=<the plot list>" in co.COORDINATOR_SYSTEM_PROMPT


def test_the_prompts_and_the_schema_ask_for_the_table_as_fetched():
    """Bidhan's (22): percent missing needs the measurements table and the plot list as fetched; the prompts and the schema say
    so instead of "the unfiltered table"."""
    schema = next(s for s in at.analyst_tool_schemas() if s["name"] == "numeric_summary")
    for text in (an.ANALYST_SYSTEM_PROMPT, co.COORDINATOR_SYSTEM_PROMPT, schema["inputSchema"]["properties"]["roster_artifact_id"]["description"]):
        assert "not a filtered, stacked or joined table" in text and "unfiltered" not in text


def test_the_prompts_say_which_cv_and_one_study_only():
    """Review findings 5 and 6: this CV is the CV of the values, not the trial's error CV (anova_trial's); the recipe is for ONE
    study, never plot lists for several (each plot list is a fetch under the fetch cap)."""
    assert "not the trial's error CV" in an.ANALYST_SYSTEM_PROMPT and "not the trial's error CV" in co.COORDINATOR_SYSTEM_PROMPT
    assert "ONE study only, never plot lists for several studies" in co.COORDINATOR_SYSTEM_PROMPT


# --- 4. not available in every unsafe case (Bidhan's (15)(a), 2026-10-23) --------------------------------------------

def test_a_filtered_table_gives_no_percent_missing(registry):
    """Review finding 1: a filtered table no longer holds the plots it left out, so they counted as without a value (synthetic:
    75 % for a table filtered to two measured plots). Not available, whatever the filter."""
    obs = measurements(registry, [*FOUR, ("P5", "0")])
    kept = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": obs, "column": "value", "op": "gt", "value": "0"})
    assert kept.ok, kept.error
    r = summary(registry, kept.artifact_ids[0], roster_artifact_id=plot_list(registry, ["P1", "P2", "P3", "P4", "P5"]))
    (k,) = missing_claims(r)
    assert r.ok and k.value is None and "filtered" in k.missing_reason and k.n_independent_units is None
    assert r.data["stats"]["mean"] == 18.0                                     # the other statistics are the filtered table's, as before


def test_study_4501_filtered_to_scored_plots_gives_no_percent_missing(registry):
    """REAL data, the review's WRONG NUMBER (rule 11): study 4501 filtered with value > 0 gave 21.35 % (38 of 178) where the
    trial's percent missing is 0.56 % (1 of 178). Now not available; the whole table still gives 0.56 %."""
    doc = real_4501()
    obs = measurements(registry, [tuple(plot) for plot in doc["plots"]], study="4501")
    roster = plot_list(registry, doc["plot_list"], study="4501")
    kept = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": obs, "column": "value", "op": "gt", "value": "0"})
    assert kept.ok and kept.data["rows_out"] == 140
    (k,) = missing_claims(summary(registry, kept.artifact_ids[0], roster_artifact_id=roster))
    assert k.value is None and "filtered" in k.missing_reason
    (k,) = missing_claims(summary(registry, obs, roster_artifact_id=roster))
    assert f"{k.value:.2f}" == "0.56"


def made_from(registry, tool, args) -> str:
    result = at.dispatch_analyst(registry, tool, args)
    assert result.ok, result.error
    return result.artifact_ids[-1]                                 # plot_design lists its inputs first, then the table it made


def test_a_stacked_or_joined_table_gives_no_percent_missing(registry):
    """Bidhan's (22), 2026-10-24: percent missing only for a measurements table and a plot list straight from the server (no
    lineage note); a table made from other tables is not available, whatever made it. The review of item 15: concat_tables and
    plot_design dropped their sources' filter notes, so a stack of two filtered tables gave 40 % here where the table as fetched
    gives 20 %, and 38.20 % and 21.35 % on REAL data against 0.56 %."""
    obs = measurements(registry, FOUR)
    roster = plot_list(registry, ["P1", "P2", "P3", "P4", "P5"])
    part = {op: made_from(registry, "filter_rows", {"artifact_id": obs, "column": "value", "op": op, "value": value})
            for op, value in (("gt", "15"), ("lt", "12"), ("le", "15"))}
    made = [("concat_tables", made_from(registry, "concat_tables", {"artifact_ids": [part["gt"], part["lt"]]})),    # the 40 %
            ("concat_tables", made_from(registry, "concat_tables", {"artifact_ids": [part["gt"], part["le"]]})),    # all four plots
            ("plot_design", made_from(registry, "plot_design", {"roster_artifact_id": roster, "artifact_id": obs})),
            ("plot_design", made_from(registry, "plot_design", {"roster_artifact_id": roster, "artifact_id": part["gt"]}))]
    for tool, table in made:
        r = summary(registry, table, roster_artifact_id=roster)
        (k,) = missing_claims(r)
        assert r.ok and k.value is None and f"made by {tool}" in k.missing_reason and k.n_independent_units is None, (tool, k)
    (k,) = missing_claims(summary(registry, obs, roster_artifact_id=roster))
    assert k.value == 20.0                                                 # the table as fetched: P5 of the five has no value


def test_a_filtered_plot_list_gives_no_percent_missing(registry):
    """The review of item 15, finding 3: a plot list cut down by filter_rows was taken as the study's plot list (here 3 of its 7
    plots without a value, 42.86 %, where the study's list of 8 gives 50 %). Not available: the plot list must be as fetched."""
    full = plot_list(registry, [f"P{i}" for i in range(1, 9)])
    cut = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": full, "column": "observationUnitDbId", "op": "ne", "value": "P8"})
    assert cut.ok and cut.data["rows_out"] == 7
    obs = measurements(registry, FOUR)
    (k,) = missing_claims(summary(registry, obs, roster_artifact_id=cut.artifact_ids[0]))
    assert k.value is None and "made by filter_rows" in k.missing_reason and k.n_independent_units is None
    (k,) = missing_claims(summary(registry, obs, roster_artifact_id=full))
    assert k.value == 50.0


def test_study_4501_stacked_or_joined_gives_no_percent_missing(registry):
    """REAL data, the review's two WRONG NUMBERs of item 15 (rule 11): concat_tables of 4501 filtered with value > 2 (73 rows) and
    with value < 1 (37) gave 38.20 % (68 of 178), and plot_design of 4501 filtered with value > 0 gave 21.35 % (38 of 178); the
    trial's percent missing is 0.56 % (1 of 178). Both tables are made from other tables: not available (Bidhan's (22))."""
    doc = real_4501()
    obs = measurements(registry, [tuple(plot) for plot in doc["plots"]], study="4501")
    roster = plot_list(registry, doc["plot_list"], study="4501")

    def cut(op, value, rows):
        r = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": obs, "column": "value", "op": op, "value": value})
        assert r.ok and r.data["rows_out"] == rows, r.error
        return r.artifact_ids[0]

    made = [("concat_tables", made_from(registry, "concat_tables", {"artifact_ids": [cut("gt", "2", 73), cut("lt", "1", 37)]})),
            ("plot_design", made_from(registry, "plot_design", {"roster_artifact_id": roster, "artifact_id": cut("gt", "0", 140)}))]
    for tool, table in made:
        (k,) = missing_claims(summary(registry, table, roster_artifact_id=roster))
        assert k.value is None and f"made by {tool}" in k.missing_reason, (tool, k)
    (k,) = missing_claims(summary(registry, obs, roster_artifact_id=roster))
    assert f"{k.value:.2f}" == "0.56"


def test_an_incomplete_measurements_table_gives_no_percent_missing(registry):
    """Review finding 2(a): plots never fetched counted as without a value (50 %); only the method line said INCOMPLETE."""
    obs = measurements(registry, FOUR, complete=False)
    r = summary(registry, obs, allow_incomplete=True, roster_artifact_id=plot_list(registry, [f"P{i}" for i in range(1, 9)]))
    (k,) = missing_claims(r)
    assert r.ok and k.value is None and "incomplete" in k.missing_reason


def test_a_valid_value_without_a_plot_id_gives_no_percent_missing(registry):
    """Review finding 2(b): 4 valid values, 2 of them on rows with no plot ID, and a 4-plot list gave 50 % beside Count 4."""
    r = summary(registry, measurements(registry, [("P1", "10"), ("P2", "14"), ("", "20"), ("", "28")]),
                roster_artifact_id=plot_list(registry, ["P1", "P2", "P3", "P4"]))
    (k,) = missing_claims(r)
    assert r.ok and k.value is None and "without a plot ID" in k.missing_reason


def test_percent_missing_is_for_one_study(registry):
    """The coach's guard: the plot list and the measurements table are the same SINGLE study."""
    rows = [{"observationDbId": f"O{i}", "observationUnitDbId": unit, "observationUnitName": unit, "germplasmDbId": "G1",
             "germplasmName": "clone G1", "studyDbId": study, "observationVariableDbId": "V1", "observationVariableName": "SYNTHETIC yield",
             "observationTimeStamp": "", "value": "10"} for i, (unit, study) in enumerate((("P1", "S1"), ("Q1", "S2")))]
    obs = registry.save_table(rows, kind="observations", columns=dict(srv.OBSERVATION_COLUMNS), complete=True, study_ids=["S1", "S2"],
                              variable_ids=["V1"], measurement=MeasurementMeta(trait="SYNTHETIC yield", unit="t/ha"),
                              quality_notes=["SYNTHETIC"]).artifact_id
    units = [{"observationUnitDbId": unit, "observationUnitName": unit, "germplasmDbId": "G1", "germplasmName": "clone G1", "studyDbId": study,
              "levelName": "plot", "levelCode": unit, "observationUnitPosition": ""} for unit, study in (("P1", "S1"), ("P2", "S1"), ("Q1", "S2"))]
    roster = registry.save_table(units, kind="observation_units", columns=dict(srv.UNIT_COLUMNS), complete=True, study_ids=["S1", "S2"],
                                 quality_notes=["SYNTHETIC"]).artifact_id
    (k,) = missing_claims(summary(registry, obs, roster_artifact_id=roster))
    assert k.value is None and "one single study" in k.missing_reason


def test_the_ids_are_read_as_written_and_the_levels_without_case(registry):
    """Review finding 10: 'Plot' and 'plot' are one level; a unit listed under two levels, or units without a stated level beside
    stated ones, give no number; unit IDs are exact text, as in missing_report (' P1' is not 'P1')."""
    values = [("P1", "10"), ("P2", "14")]
    (k,) = missing_claims(summary(registry, measurements(registry, values),
                                  roster_artifact_id=plot_list(registry, ["P1", ("P2", "Plot"), ("P3", "PLOT"), "P4"])))
    assert k.value == pytest.approx(50.0, rel=1e-12) and k.denominator.n == 4
    for units, reason in ((["P1", "P2", "P3", ("P3", "plant")], "listed under two levels"),
                          (["P1", "P2", ("P3", ""), ("P4", "")], "without a stated level")):
        (k,) = missing_claims(summary(registry, measurements(registry, values), roster_artifact_id=plot_list(registry, units)))
        assert k.value is None and reason in k.missing_reason, units
    (k,) = missing_claims(summary(registry, measurements(registry, [(" P1", "10"), ("P2", "14")]),
                                  roster_artifact_id=plot_list(registry, ["P1", "P2", "P3"])))
    assert k.value is None and "not on the plot list" in k.missing_reason


def test_the_claim_names_its_plot_list_and_the_method_says_fillers_count(registry):
    """Review finding 9: one table with two copies of the plot list gave two claims with one ID, and the analysis was withheld.
    Finding 10: every listed plot counts, fillers included, as on Breedbase, and the method says so."""
    obs = measurements(registry, FOUR)
    first, second = (missing_claims(summary(registry, obs, roster_artifact_id=plot_list(registry, ["P1", "P2", "P3", "P4", "P5"])))[0]
                     for _ in range(2))
    assert first.value == second.value == pytest.approx(20.0, rel=1e-12) and first.claim_id != second.claim_id
    assert "every listed plot counts, fillers included, as on Breedbase" in first.transformation
    assert first.n_independent_units == 4 and first.denominator.n == 5             # finding 3: the plots with a value


def test_a_percent_missing_not_available_shows_no_counts_in_the_answer(tmp_path):
    """Review finding 3: 'not available' came with 'Based on plots on the study's plot list: 5. Recorded analysis units: 5.' and
    '(1 of 5)'. Now the reason alone, and the method."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_ps_na")
    obs = measurements(ctx.registry, FOUR)
    r = summary(ctx.registry, obs, roster_artifact_id=plot_list(ctx.registry, ["P1", "P2", "P3", "P4", "P5"], complete=False))
    shown = answer(ctx, r, "Summarize yield in this trial")
    block = shown[shown.index("**Plots without a value"):]
    block = block[:block.index("\n\n**") if "\n\n**" in block else len(block)]
    assert "| Percent missing | Not available: the plot list" in block and "Based on" not in block and " of 5" not in block


def test_a_mean_only_answer_adds_no_unrequested_cv_or_percent_missing_note(tmp_path):
    """Review finding 4: a mean-only answer gained 'An additional statistic could not be calculated: the mean is not above 0'
    (an unrequested CV); nor is an unrequested percent missing noted there."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_ps_mean")
    obs = measurements(ctx.registry, [("P1", "-3"), ("P2", "-1"), ("P3", "1")])
    r = summary(ctx.registry, obs, roster_artifact_id=plot_list(ctx.registry, ["P1", "P2", "P3", "P4"], complete=False))
    shown = answer(ctx, r, "What is the mean yield in this trial?")
    assert "**Mean: " in shown and "could not be calculated" not in shown and "not positive" not in shown and "not above" not in shown


def test_the_coordinators_view_keeps_the_cv_and_percent_missing_when_there_is_room(registry):
    """The CV and percent missing come after the other calls' headline claims in the coordinator's 20-claim view
    (test_anova_trial: test_mixed_calls_keep_each_trials_mean_squares_in_view); alone they are in view."""
    import coordination_runtime as cr
    r = summary(registry, measurements(registry, FOUR), roster_artifact_id=plot_list(registry, ["P1", "P2", "P3", "P4", "P5"]))
    shown = cr._view_claims(claims(r), set())
    last = [k.transformation for k in shown][-2:]
    assert len(shown) == len(claims(r)) and last[0].startswith("cv of valid values") and last[1].startswith("percent missing = ")


def test_the_mean_stays_alone_unless_the_cv_or_percent_missing_is_asked(tmp_path):
    """Review finding 7: 'percent' or 'missing' alone widened a mean question to the full table (SweetPotatoBase's trait
    'Storage root dry matter content computing percent'); only the CV and 'percent missing' / 'missing values' do."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_ps_words")
    r = summary(ctx.registry, measurements(ctx.registry, FOUR))
    for question in ("What is the average dry matter percent in this trial?", "What is the mean yield, ignoring missing plots?"):
        assert "43.5086671277 %" not in answer(ctx, r, question), question
    for question in ("What is the mean yield and its missing values?", "What is the mean yield and the coefficient of variation?"):
        assert "43.5086671277 %" in answer(ctx, r, question), question


def test_the_cv_and_percent_missing_stay_in_view_beside_an_anova_call(registry):
    """The review of item 15, SHOULD-FIX 5: after all the calls' turns, one anova_trial call (more than 20 claims) and one
    numeric_summary with its plot list left the CV and percent missing out of the coordinator's 20 claims. They take the fifth
    turn now, after each call's first four claims (Bidhan's (22): the small view fix)."""
    import coordination_runtime as cr
    from test_anova_trial import RICE, anova, plots, save

    table = save(registry, plots(RICE))
    r = summary(registry, table, roster_artifact_id=plot_list(registry, [plot[0] for plot in plots(RICE)]))
    every = [*claims(anova(registry, table)), *claims(r)]
    shown = cr._view_claims(every, set())
    assert len(every) > cr.VIEW_CLAIMS and cv_claim(r) in shown and missing_claims(r)[0] in shown
