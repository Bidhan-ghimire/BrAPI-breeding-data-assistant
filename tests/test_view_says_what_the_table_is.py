"""The Coordinator's view says what the table is (coach 2026-10-24 12:50 UTC, NEXT 3; Luna 24 Oct, four right analyses stopped).

GAP 1 (A42.4): a mean of 315 genotype means (group_stats by germplasmDbId, then filter_rows x 3) was written "valid values used as
recorded (one row per observation unit)", and the coordinator stopped it as a mean over plot rows. The transformation text now
names the table's row unit: "one row per germplasmDbId; values are group means" for a group_stats table (carried through
filter_rows and concat_tables), "one row per observation unit" only when the rows are units, else that it is not recorded.
GAP 2 (E13.4): "entries only in y", 1 was the check CIP199062.1, and no claim named it. Exclusions now name their entries (all of
them up to ten; else the first ten and "and N more").
GAP 3 (A38.2): "claims_omitted": 3273 was read as missing results. A sentence beside it says what it is.
Everything here is SYNTHETIC: made-up tables, no network, no model."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import analyst_tools as at
import coordination_runtime as cr
from contracts import AnalysisReport, Claim, Denominator
from test_correlate_entry_means import claims, registry, row, save, save_rows  # noqa: F401  (the registry fixture)


def run(registry, tool, **args):
    r = at.dispatch_analyst(registry, tool, args)
    assert r.ok, (tool, r.error)
    return r


def plots(registry, entries=(("A", "Alpha", "10"), ("B", "Beta", "20"), ("C", "Gamma", "30"), ("D", "Delta", "40"), ("K", "Check", "50")),
          study="S1"):
    """One plot table, two plots per entry (values v and v + 2)."""
    rows = [row(f"u{study}{g}{j}", g, name, str(float(v) + 2 * j), study=study, obs=f"o{study}{g}{j}") for g, name, v in entries for j in (0, 1)]
    return save_rows(registry, rows, [study])


def group_table(registry, plot_id, by="germplasmDbId"):
    return run(registry, "group_stats", artifact_id=plot_id, by=by, value="value").data["result_artifact_id"]


def summary_text(registry, artifact_id, value):
    return claims(run(registry, "numeric_summary", artifact_id=artifact_id, value=value))[0].transformation


# --- GAP 1: the row unit -------------------------------------------------------------------------------------------------

def test_a_summary_of_filtered_group_means_says_one_row_per_entry_not_per_unit(registry):
    kept = run(registry, "filter_rows", artifact_id=group_table(registry, plots(registry)), column="label", op="ne", value="Check")
    text = summary_text(registry, kept.artifact_ids[0], "mean")
    assert "one row per germplasmDbId; values are group means" in text and "observation unit" not in text


def test_the_group_table_itself_and_a_median_column_say_so(registry):
    groups = group_table(registry, plots(registry))
    assert "one row per germplasmDbId; values are group means" in summary_text(registry, groups, "mean")
    assert "one row per germplasmDbId; values are group medians" in summary_text(registry, groups, "median")


def test_a_summary_of_the_plot_table_still_says_one_row_per_observation_unit(registry):
    text = summary_text(registry, plots(registry), "value")
    assert "valid values used as recorded (one row per observation unit)" in text


def test_two_stacked_group_tables_keep_the_row_unit(registry):
    first = group_table(registry, plots(registry))
    second = group_table(registry, plots(registry, (("E", "Eps", "5"), ("F", "Phi", "6"), ("G", "Chi", "7"))))
    stacked = run(registry, "concat_tables", artifact_ids=[first, second]).artifact_ids[0]
    assert "one row per germplasmDbId in each of the 2 stacked tables; values are group means" in summary_text(registry, stacked, "mean")


def test_a_group_table_stacked_with_another_kind_says_the_rows_differ(registry):
    by_entry = group_table(registry, plots(registry))
    by_study = group_table(registry, plots(registry), by="studyDbId")
    stacked = run(registry, "concat_tables", artifact_ids=[by_entry, by_study]).artifact_ids[0]
    text = summary_text(registry, stacked, "mean")
    assert "stacked rows of different kinds" in text and "group_stats by germplasmDbId" in text and "group_stats by studyDbId" in text
    assert "values are group means" not in text and "one row per observation unit" not in text


def test_a_derived_table_with_no_recorded_row_unit_does_not_claim_units(registry):
    rows = [{"group_id": g, "label": g, "mean": v} for g, v in (("A", "1"), ("B", "2"), ("C", "3"))]
    made = registry.save_table(rows, kind="analysis_result", columns={"group_id": "id", "label": "name", "mean": "number"}, complete=True,
                               study_ids=["S1"], variable_ids=["V1"], quality_notes=["SYNTHETIC"]).artifact_id
    text = summary_text(registry, made, "mean")
    assert "what one row stands for is not recorded" in text and "observation unit" not in text


# --- GAP 2: the excluded and unpaired entries are named ------------------------------------------------------------------

def reasons(r):
    return {e.reason: e.count for e in claims(r)[0].exclusions}


def test_a_correlation_names_the_entries_it_could_not_pair(registry):
    x = save(registry, [("A", "Alpha", ["1"]), ("B", "Beta", ["2"]), ("C", "Gamma", ["4"]), ("D", "Delta", ["5"]), ("K", "Check", ["9"])])
    y = save(registry, [("A", "Alpha", ["2"]), ("B", "Beta", ["3"]), ("C", "Gamma", ["7"]), ("D", "Delta", ["4"]), ("Q", "Queen", ["1"])],
             study="S2")
    got = reasons(run(registry, "correlate_entry_means", artifact_ids=[x, y], exclude=["Gamma"]))
    assert got == {"entries only in x (Check)": 1, "entries only in y (Queen)": 1, "entries left out: excluded (Gamma)": 1}


def test_a_correlation_names_the_entries_not_in_every_study(registry):
    """E13.4: 265 entries had flesh colour in all five trials; the 265th, the check CIP199062.1, lacked one trial's other trait."""
    entries = [("A", "Alpha"), ("B", "Beta"), ("C", "Gamma"), ("K", "CIP199062.1")]
    rows_x = [row(f"x{s}{g}", g, n, str(i + 1), study=s, obs=f"ox{s}{g}") for s in ("S1", "S2") for i, (g, n) in enumerate(entries)
              if not (g == "K" and s == "S2")]                                   # the check has no value in S2
    rows_y = [row(f"y{s}{g}", g, n, str(2 * i + 1), study=s, obs=f"oy{s}{g}", variable="V2") for s in ("S1", "S2") for i, (g, n) in enumerate(entries)]
    x = save_rows(registry, rows_x, ["S1", "S2"])
    y = save_rows(registry, rows_y, ["S1", "S2"], variable="V2")
    got = reasons(run(registry, "correlate_entry_means", artifact_ids=[x, y], entries="in_every_study", min_pairs=3))
    assert got["entries not in every study of x (CIP199062.1)"] == 1 and got["entries only in y (CIP199062.1)"] == 1


def test_more_than_ten_entries_are_named_ten_and_counted(registry):
    many = [(f"E{n:02d}", f"Name{n:02d}", [str(n)]) for n in range(15)]
    x = save(registry, many[:3] + many[3:])                                      # 15 entries
    y = save(registry, many[:3], study="S2")                                     # 3 of them
    got = reasons(run(registry, "correlate_entry_means", artifact_ids=[x, y], min_pairs=3))
    shown = ", ".join(f"Name{n:02d}" for n in range(3, 13))
    assert got == {f"entries only in x ({shown} and 2 more)": 12}


def test_a_paired_comparison_names_the_entries_left_out_and_the_unpaired(registry):
    base = save(registry, [("A", "Alpha", ["1"]), ("B", "Beta", ["2"]), ("C", "Gamma", ["4"]), ("D", "Delta", ["5"]), ("K", "Beauregard", ["9"])])
    other = save(registry, [("A", "Alpha", ["2"]), ("B", "Beta", ["3"]), ("C", "Gamma", ["7"]), ("Q", "Queen", ["1"]), ("K", "Beauregard", ["8"])],
                 study="S2")
    got = reasons(run(registry, "compare_trials_paired", base_artifact_id=base, other_artifact_id=other, by="germplasmDbId",
                      exclude=["Beauregard"], percent="change"))
    assert got == {"entries only in the base trial (Delta)": 1, "entries only in the other trial (Queen)": 1,
                   "entries left out: excluded (Beauregard)": 1}


def test_stress_indices_name_the_entries_left_out(registry):
    base = save(registry, [("A", "Alpha", ["10"]), ("B", "Beta", ["20"]), ("C", "Gamma", ["40"]), ("D", "Delta", ["50"]), ("K", "Beauregard", ["9"])])
    other = save(registry, [("A", "Alpha", ["5"]), ("B", "Beta", ["8"]), ("C", "Gamma", ["30"]), ("K", "Beauregard", ["8"])], study="S2")
    got = reasons(run(registry, "stress_indices", nonstress_artifact_id=base, stress_artifact_id=other, by="germplasmDbId",
                      exclude=["Beauregard"]))
    assert got["entries only in the non-stress trial (Delta)"] == 1 and got["entries left out: excluded (Beauregard)"] == 1


# --- GAP 3: the claims the view leaves out ------------------------------------------------------------------------------

def view(n_claims, status="completed"):
    made = [Claim(claim_id=f"clm_many_{n:03d}", kind="count", value=1, denominator=Denominator(name="rows", n=3),
                  source_artifact_hashes=["a" * 64], filters={"germplasmDbId": f"G{n:03d}"}, grouping=["germplasmDbId"],
                  transformation="frequency of exact text values") for n in range(n_claims)]
    plain = SimpleNamespace(plan=None, retrieval=None, data_review={"status": "approved"},
                            analysis=AnalysisReport(status=status, claims=made),
                            delegation={"decisions": [], "plan_revisions": 0, "analysis_passes": 1})
    return cr._state(plain, "analyzed")["analysis"]


def test_the_view_says_the_claims_it_leaves_out_are_in_the_saved_result():
    v = view(25)
    assert v["claims_omitted"] == 5 and len(v["claims"]) == 20
    assert v["claims_not_shown"] == "5 further claims not shown in this view; the analysis is complete and all claims are in the saved result"


def test_an_incomplete_analysis_is_not_called_complete():
    v = view(25, status="incomplete")
    assert v["claims_not_shown"] == "5 further claims not shown in this view; all claims are in the saved result"


def test_no_sentence_when_every_claim_is_shown():
    v = view(3)
    assert v["claims_omitted"] == 0 and "claims_not_shown" not in v


def test_compare_means_on_rows_without_unit_ids_does_not_claim_units(registry):
    def made(study, values):
        rows = [{"germplasmDbId": g, "germplasmName": f"clone {g}", "studyDbId": study, "observationVariableDbId": "V1", "value": v}
                for g, v in values]
        return registry.save_table(rows, kind="observations", columns={c: "text" for c in rows[0]}, complete=True, study_ids=[study],
                                   variable_ids=["V1"], quality_notes=["SYNTHETIC"]).artifact_id
    r = run(registry, "compare_means", base_artifact_id=made("S1", [("A", "10"), ("B", "20"), ("C", "30")]),
            other_artifact_id=made("S2", [("A", "8"), ("B", "18"), ("C", "24")]), base_study_db_id="S1", percent="reduction")
    texts = " | ".join(k.transformation for k in claims(r))
    assert "what one row stands for is not recorded" in texts and "one row per observation unit" not in texts


def test_one_claim_left_out_is_said_in_the_singular():
    assert view(21)["claims_not_shown"] == "1 further claim not shown in this view; the analysis is complete and all claims are in the saved result"


def test_an_excluded_entry_in_one_trial_only_is_named_once_as_excluded(registry):
    """The names agree with the counts: Delta is left out by name and is only in the base trial; it is not also unpaired."""
    base = save(registry, [("A", "Alpha", ["1"]), ("B", "Beta", ["2"]), ("C", "Gamma", ["4"]), ("D", "Delta", ["5"]), ("K", "Beauregard", ["9"])])
    other = save(registry, [("A", "Alpha", ["2"]), ("B", "Beta", ["3"]), ("C", "Gamma", ["7"]), ("Q", "Queen", ["1"]), ("K", "Beauregard", ["8"])],
                 study="S2")
    r = run(registry, "compare_trials_paired", base_artifact_id=base, other_artifact_id=other, by="germplasmDbId", exclude=["Delta"],
            percent="change")
    assert reasons(r) == {"entries only in the other trial (Queen)": 1, "entries left out: excluded (Delta)": 1}
    assert r.data["unpaired"] == {"only in base": 0, "only in other": 1}


# --- round 2 (review of NEXT 3: NOT SAFE) ------------------------------------------------------------------------------

def test_stacked_group_tables_of_the_same_clones_say_one_row_per_entry_in_each_table(registry):
    """MUST-FIX: two studies' group tables of the same clones A, B, C, stacked: 6 rows, 3 distinct IDs. "one row per
    germplasmDbId" was untrue; it is one row per germplasmDbId in each stacked table."""
    clones = (("A", "Alpha", "10"), ("B", "Beta", "20"), ("C", "Gamma", "30"))
    first, second = group_table(registry, plots(registry, clones)), group_table(registry, plots(registry, clones, study="S2"))
    stacked = run(registry, "concat_tables", artifact_ids=[first, second]).artifact_ids[0]
    text = summary_text(registry, stacked, "mean")
    assert "one row per germplasmDbId in each of the 2 stacked tables; values are group means" in text
    assert "one row per germplasmDbId;" not in text
    kept = run(registry, "filter_rows", artifact_id=stacked, column="label", op="ne", value="Gamma").artifact_ids[0]
    assert "in each of the 2 stacked tables" in summary_text(registry, kept, "mean")          # carried by filter_rows


def test_a_unit_with_several_rows_is_not_called_one_row_per_unit(registry):
    """SHOULD-FIX: one plot measured at two times gives 3 rows for 2 units; the text said one row per observation unit."""
    rows = [row("u1", "A", "Alpha", "1", obs="o1"), row("u1", "A", "Alpha", "2", obs="o2"), row("u2", "B", "Beta", "3", obs="o3")]
    rows[0]["observationTimeStamp"], rows[1]["observationTimeStamp"] = "2019-01-01T00:00:00Z", "2019-02-01T00:00:00Z"
    text = summary_text(registry, save_rows(registry, rows, ["S1"]), "value")
    assert "one row per observation; some observation units have several" in text and "one row per observation unit" not in text


def test_a_long_name_is_cut_in_the_reason(registry):
    """SHOULD-FIX: a 400-character name made a 450-character reason on every claim."""
    x = save(registry, [("A", "Alpha", ["1"]), ("B", "Beta", ["2"]), ("C", "Gamma", ["4"]), ("L", "L" * 400, ["5"])])
    y = save(registry, [("A", "Alpha", ["2"]), ("B", "Beta", ["3"]), ("C", "Gamma", ["7"])], study="S2")
    [reason] = list(reasons(run(registry, "correlate_entry_means", artifact_ids=[x, y])))
    assert reason == "entries only in x (" + "L" * 79 + "\u2026)"


def many(n_only_x):
    entries = [(f"E{n:02d}", f"Name{n:02d}", [str(n + 1)]) for n in range(3 + n_only_x)]
    return entries, entries[:3]


def test_exactly_ten_names_have_no_more_clause_and_eleven_say_one_more(registry):
    ten_x, ten_y = many(10)
    got = reasons(run(registry, "correlate_entry_means", artifact_ids=[save(registry, ten_x), save(registry, ten_y, study="S2")]))
    assert got == {"entries only in x (" + ", ".join(f"Name{n:02d}" for n in range(3, 13)) + ")": 10}
    eleven_x, eleven_y = many(11)
    got = reasons(run(registry, "correlate_entry_means", artifact_ids=[save(registry, eleven_x), save(registry, eleven_y, study="S2")]))
    assert got == {"entries only in x (" + ", ".join(f"Name{n:02d}" for n in range(3, 13)) + " and 1 more)": 11}


def test_names_are_sorted_and_an_entry_without_a_name_is_shown_by_its_id(registry):
    x = save(registry, [("A", "Alpha", ["1"]), ("B", "Beta", ["2"]), ("C", "Gamma", ["4"]), ("Z1", "Zeta", ["5"]), ("A9", "Omega", ["6"]),
                        ("N7", "", ["7"])])
    y = save(registry, [("A", "Alpha", ["2"]), ("B", "Beta", ["3"]), ("C", "Gamma", ["7"])], study="S2")
    assert reasons(run(registry, "correlate_entry_means", artifact_ids=[x, y])) == {"entries only in x (N7, Omega, Zeta)": 3}


def test_the_entries_outside_the_name_prefix_are_named(registry):
    x = save(registry, [("A", "BT-1", ["1"]), ("B", "BT-2", ["2"]), ("C", "BT-3", ["4"]), ("K", "Check", ["9"])])
    y = save(registry, [("A", "BT-1", ["2"]), ("B", "BT-2", ["3"]), ("C", "BT-3", ["7"]), ("K", "Check", ["8"])], study="S2")
    got = reasons(run(registry, "correlate_entry_means", artifact_ids=[x, y], name_prefix="BT-"))
    assert got == {"entries left out: outside the name prefix (Check)": 1}
    paired = reasons(run(registry, "compare_trials_paired", base_artifact_id=x, other_artifact_id=y, by="germplasmDbId", name_prefix="BT-",
                         percent="change"))
    assert paired == {"entries left out: outside the name prefix (Check)": 1}


def test_an_entry_outside_the_prefix_in_one_trial_only_is_named_once(registry):
    base = save(registry, [("A", "BT-1", ["1"]), ("B", "BT-2", ["2"]), ("C", "BT-3", ["4"]), ("K", "Check", ["9"])])
    other = save(registry, [("A", "BT-1", ["2"]), ("B", "BT-2", ["3"]), ("C", "BT-3", ["7"])], study="S2")
    got = reasons(run(registry, "compare_trials_paired", base_artifact_id=base, other_artifact_id=other, by="germplasmDbId",
                      name_prefix="BT-", percent="change"))
    assert got == {"entries left out: outside the name prefix (Check)": 1}                # not also "only in the base trial"


def test_an_excluded_entry_without_a_mean_is_not_named_as_left_out(registry):
    base = save(registry, [("A", "Alpha", ["1"]), ("B", "Beta", ["2"]), ("C", "Gamma", ["4"]), ("D", "Delta", ["NA"])])
    other = save(registry, [("A", "Alpha", ["2"]), ("B", "Beta", ["3"]), ("C", "Gamma", ["7"])], study="S2")
    r = run(registry, "compare_trials_paired", base_artifact_id=base, other_artifact_id=other, by="germplasmDbId", exclude=["Delta"],
            percent="change")
    assert not any(reason.startswith("entries left out: excluded") for reason in reasons(r))


def test_stress_indices_name_the_entries_not_ranked(registry):
    base = save(registry, [("A", "Alpha", ["10"]), ("B", "Beta", ["20"]), ("C", "Gamma", ["0"]), ("D", "Delta", ["30"])])
    other = save(registry, [("A", "Alpha", ["5"]), ("B", "Beta", ["8"]), ("C", "Gamma", ["3"]), ("D", "Delta", ["20"])], study="S2")
    got = reasons(run(registry, "stress_indices", nonstress_artifact_id=base, stress_artifact_id=other, by="germplasmDbId", rank_by="YSI"))
    assert got["entries not ranked: YSI undefined (Gamma)"] == 1


def test_each_group_statistic_is_named(registry):
    groups = group_table(registry, plots(registry))
    for value, words in (("min", "group minima"), ("max", "group maxima"), ("sd", "group SDs"), ("n_rows", "group statistics")):
        assert f"one row per germplasmDbId; values are {words}" in summary_text(registry, groups, value)


def test_a_group_table_summarised_again_by_group_stats_says_so(registry):
    groups = group_table(registry, plots(registry))
    again = run(registry, "group_stats", artifact_id=groups, by="label", value="mean")
    assert again.data["transformation"] == "valid values used as recorded (one row per germplasmDbId; values are group means)"


def test_the_grader_backs_the_numbers_in_an_exclusion_reason():
    """The derived "and N more" is claim text the answer shows (benchmark claim_numbers): never graded invented."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("_bench_view_gaps", Path(__file__).resolve().parents[1] / "eval" / "benchmark.py")
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    claim = {"kind": "correlation", "value": 0.91, "denominator": {"name": "pairs", "n": 264}, "filters": {}, "grouping": [],
             "transformation": "Pearson correlation of entry means", "missing_reason": None,
             "exclusions": [{"reason": "entries only in x (Name03, Name04 and 43 more)", "count": 45}]}
    assert 43.0 in bench.claim_numbers([claim])


def test_the_full_reports_numeric_table_escapes_a_bar_in_a_name():
    from contracts import Exclusion
    import agents.coordinator as co
    c = Claim(claim_id="clm_bar", kind="count", value=1, denominator=Denominator(name="rows", n=3), source_artifact_hashes=["a" * 64],
              filters={}, grouping=[], transformation="count", exclusions=[Exclusion(reason="entries only in x (A|B)", count=1)])
    table = co.render_numeric_table(AnalysisReport(status="completed", claims=[c]))
    assert "entries only in x (A\\|B) x1" in table


# --- review 2 (SAFE; SHOULD-FIX 1-2) -------------------------------------------------------------------------------------

def test_blank_unit_ids_are_not_one_unit(registry):
    """SHOULD-FIX 1: two valid rows with an empty observationUnitDbId were read as one unit with several rows."""
    rows = [row("", "A", "Alpha", "1", obs="o1"), row("", "B", "Beta", "2", obs="o2"), row("u3", "C", "Gamma", "3", obs="o3")]
    text = summary_text(registry, save_rows(registry, rows, ["S1"]), "value")
    assert "one row per observation; some rows name no observation unit" in text
    assert "have several" not in text and "one row per observation unit" not in text


def test_the_summary_denominator_agrees_with_the_row_unit(registry):
    """SHOULD-FIX 2: one plot measured at two times (3 values, 2 units) kept "valid numeric values, one per observation unit"."""
    rows = [row("u1", "A", "Alpha", "1", obs="o1"), row("u1", "A", "Alpha", "2", obs="o2"), row("u2", "B", "Beta", "3", obs="o3")]
    rows[0]["observationTimeStamp"], rows[1]["observationTimeStamp"] = "2019-01-01T00:00:00Z", "2019-02-01T00:00:00Z"
    [first] = claims(run(registry, "numeric_summary", artifact_id=save_rows(registry, rows, ["S1"]), value="value"))[:1]
    assert first.denominator.name == "valid numeric values (rows; some observation units have several)"
    one_each = [row("u1", "A", "Alpha", "1", obs="o1"), row("u2", "B", "Beta", "3", obs="o3")]
    [plain] = claims(run(registry, "numeric_summary", artifact_id=save_rows(registry, one_each, ["S1"]), value="value"))[:1]
    assert plain.denominator.name == "valid numeric values, one per observation unit"
