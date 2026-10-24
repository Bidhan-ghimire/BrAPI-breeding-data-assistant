"""Texts true (coach 2026-10-24 15:49 UTC, NEXT 2): four small texts that were not quite true after NEXT 3 and NEXT 4.
(a) the answer's "Left out:" line removed every "left out: " in a reason, so an entry named "Gamma left out: z" read "Gamma z":
    only the note's own "left out: " at its start goes; (b) a stack of group tables filtered down to some of its sources still
    said "in each of the N stacked tables": it names the one table, or says k of the N or none, read from every row of the
    table, and a stack of threshold_count tables says the same;
    (c) rank_entries and threshold_count tables wrote no rows note, so a summary of them said only that a row's meaning is not
    recorded: they say what a row is, a rank_entries table its basis too (on study_means its mean is the unweighted mean of
    per-study means, the rest pooled: review 2026-10-24); (d) "N/A" and "none" name no program, as "NA" does. No program ID or
    name in the benchmark cache is "N/A" or "none" (count_all_program_tokens.txt), so no number from the cached data changes;
    a study whose program is "N/A" or "none" now has none. Everything here is SYNTHETIC."""
from __future__ import annotations

from types import SimpleNamespace

import agents.coordinator as co
import analyst_tools as at
import brapi_client
import brapi_mcp_server as srv
import chat_presenter as cp
from artifacts import ArtifactRegistry
from contracts import AnalysisReport
from part2_support import SYNTHETIC_BASE, synthetic_context
from test_compare_means import BASE, OTHER, by_study, compare, save as save_means
from test_correlate_entry_means import claims, registry, row, save, save_rows  # noqa: F401  (the registry fixture)
from test_program_from_additional_info import catalog
from test_view_says_what_the_table_is import group_table, plots, run, summary_text

ODD = "Gamma left out: z"                                  # a name holding the words the "Left out:" line used to strip
CLONES = (("A", "Alpha", "10"), ("B", "Beta", "20"), ("C", "Gamma", "30"))


def answer(ctx, r):
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claims(r), caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    return cp.answer_summary(view)


def four(registry, study, odd=False):
    entries = [("A", "Alpha", ["10"]), ("B", "Beta", ["20"]), ("C", "Delta", ["30"]), ("D", "Eps", ["40"])]
    return save(registry, entries + ([("G", ODD, ["50"])] if odd else []), study=study)


def stack(registry, *studies):
    """Group tables of the same clones, one per study, stacked: (the stacked table, the group tables)."""
    groups = [group_table(registry, plots(registry, CLONES, study=s)) for s in studies]
    return run(registry, "concat_tables", artifact_ids=groups).artifact_ids[0], groups


def two_studies(registry):
    """A has S1 [10, 10, 10, 10] and S2 [20]: pooled mean 12, mean of its per-study means 15 (review 2026-10-24, MUST-FIX 1)."""
    values = {("A", "S1"): [10, 10, 10, 10], ("A", "S2"): [20], ("B", "S1"): [30], ("B", "S2"): [30]}
    rows = [row(f"u{s}{g}{j}", g, {"A": "Alpha", "B": "Beta"}[g], str(v), study=s, obs=f"o{s}{g}{j}")
            for (g, s), vs in values.items() for j, v in enumerate(vs)]
    return save_rows(registry, rows, ["S1", "S2"])


def means(registry, artifact_id):
    return {r["group_id"]: float(r["mean"]) for r in at._load(registry, artifact_id).rows}


# --- (a) "left out: " goes once, at the start of the note --------------------------------------------------------------

def test_a_correlation_keeps_a_name_that_holds_left_out(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_texts_a1")
    shown = answer(ctx, run(ctx.registry, "correlate_entry_means", artifact_ids=[four(ctx.registry, "S1", odd=True), four(ctx.registry, "S2")]))
    assert ODD in shown and "Gamma z" not in shown


def test_a_paired_comparison_keeps_a_name_that_holds_left_out(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_texts_a2")
    shown = answer(ctx, run(ctx.registry, "compare_trials_paired", base_artifact_id=four(ctx.registry, "S1", odd=True),
                            other_artifact_id=four(ctx.registry, "S2"), by="germplasmDbId", percent="change"))
    assert ODD in shown and "Gamma z" not in shown


def test_stress_indices_keep_a_name_that_holds_left_out(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_texts_a3")
    shown = answer(ctx, run(ctx.registry, "stress_indices", nonstress_artifact_id=four(ctx.registry, "S1", odd=True),
                            stress_artifact_id=four(ctx.registry, "S2"), by="germplasmDbId"))
    assert ODD in shown and "Gamma z" not in shown


def test_the_notes_own_left_out_still_goes():
    assert cp._left_out_text("entries left out: excluded (Beauregard)") == "excluded (Beauregard)"
    assert cp._left_out_text("entries only in x (Gamma left out: z)") == "only in x (Gamma left out: z)"
    assert cp._left_out_text("entries left out: excluded (A left out: b)") == "excluded (A left out: b)"


# --- (b) a stack filtered down to some of its sources says which -------------------------------------------------------

def test_a_stack_filtered_to_one_source_names_the_one_table(registry):
    stacked, (first, _) = stack(registry, "S1", "S2")
    one = run(registry, "filter_rows", artifact_id=stacked, column="source_artifact_id", op="eq", value=first).artifact_ids[0]
    text = summary_text(registry, one, "mean")
    assert f"one row per germplasmDbId in the one table from {first}; values are group means" in text and "stacked tables" not in text
    both = run(registry, "filter_rows", artifact_id=stacked, column="label", op="ne", value="Gamma").artifact_ids[0]
    assert "one row per germplasmDbId in each of the 2 stacked tables; values are group means" in summary_text(registry, both, "mean")


def test_a_stack_filtered_to_some_of_its_sources_says_how_many(registry):
    """Review 2026-10-24, SHOULD-FIX 1: source_artifact_id ne <third> on a stack of 3 said "in each of the 3 stacked tables"."""
    stacked, (_, _, third) = stack(registry, "S1", "S2", "S3")
    two = run(registry, "filter_rows", artifact_id=stacked, column="source_artifact_id", op="ne", value=third).artifact_ids[0]
    assert "one row per germplasmDbId in each of 2 of the 3 stacked tables; values are group means" in summary_text(registry, two, "mean")
    assert "one row per germplasmDbId in each of the 3 stacked tables; values are group means" in summary_text(registry, stacked, "mean")


def test_the_sources_are_read_from_every_row_not_the_valid_values(registry):
    """Review 2026-10-24, SHOULD-FIX 2: with no valid SD in the second table (one plot per clone), the SD summary named "the one
    table from" the first and the mean summary "each of the 2"; the table's rows are from both either way."""
    first = group_table(registry, plots(registry, CLONES))
    single = save_rows(registry, [row(f"uS2{g}", g, name, v, study="S2", obs=f"oS2{g}") for g, name, v in CLONES], ["S2"])
    stacked = run(registry, "concat_tables", artifact_ids=[first, group_table(registry, single)]).artifact_ids[0]
    assert "one row per germplasmDbId in each of the 2 stacked tables; values are group means" in summary_text(registry, stacked, "mean")
    assert "one row per germplasmDbId in each of the 2 stacked tables; values are group SDs" in summary_text(registry, stacked, "sd")


def test_a_row_that_names_no_source_table_is_said(registry):
    """A hand-made table (concat_tables always names the source): a blank source_artifact_id is no known table, so rows of one
    named table and one blank row are not "the one table from" it."""
    stacked, (first, _) = stack(registry, "S1", "S2")
    table = at._load(registry, stacked)
    rows = [dict(r) for r in table.rows if r["source_artifact_id"] == first]
    rows[0]["source_artifact_id"] = ""
    m = table.manifest
    hand = registry.save_table(rows, kind=m.kind, columns=m.columns, complete=m.complete, study_ids=list(m.study_ids),
                               variable_ids=list(m.variable_ids), measurement=m.measurement, quality_notes=list(m.quality_notes)).artifact_id
    text = summary_text(registry, hand, "mean")
    assert "one row per germplasmDbId in each stacked table; some rows name no source table; values are group means" in text


def test_a_stack_filtered_to_no_row_says_no_row_is_left(registry):
    """Review 2 of 2026-10-24, SHOULD-FIX 1: a filter that keeps no row said "in each of the 3 stacked tables" beside a count
    of 0."""
    stacked, _ = stack(registry, "S1", "S2", "S3")
    none = run(registry, "filter_rows", artifact_id=stacked, column="source_artifact_id", op="eq", value="no such table").artifact_ids[0]
    text = summary_text(registry, none, "mean")
    assert "one row per germplasmDbId in none of the 3 stacked tables: no row is left; values are group means" in text


# --- (c) rank_entries and threshold_count tables say what a row is -----------------------------------------------------

def test_a_rank_entries_table_says_one_row_per_entry(registry):
    ranked = run(registry, "rank_entries", artifact_id=plots(registry), by="germplasmDbId", value="value").data["result_artifact_id"]
    assert "one row per germplasmDbId; values are group means" in summary_text(registry, ranked, "mean")


def test_a_study_means_rank_table_says_what_its_mean_is(registry):
    """Review 2026-10-24, MUST-FIX 1: on basis study_means the saved mean is the unweighted mean of per-study means (A: 15, not
    the pooled 12), but the rows note said "values are group means"; median, min, max and sd stay pooled."""
    table = two_studies(registry)
    pooled = run(registry, "rank_entries", artifact_id=table, by="germplasmDbId", value="value").data["result_artifact_id"]
    per_study = run(registry, "rank_entries", artifact_id=table, by="germplasmDbId", value="value",
                    basis="study_means").data["result_artifact_id"]
    assert means(registry, pooled)["A"] == 12.0 and means(registry, per_study)["A"] == 15.0
    assert "one row per germplasmDbId; values are group means" in summary_text(registry, pooled, "mean")
    text = summary_text(registry, per_study, "mean")
    assert "one row per germplasmDbId; values are the unweighted means of each group's per-study means" in text
    assert "values are group means" not in text
    for column, word in (("median", "medians"), ("min", "minima"), ("max", "maxima"), ("sd", "SDs")):     # review 2: each column
        assert f"one row per germplasmDbId; values are group {word}, pooled over the studies" in summary_text(registry, per_study, column)
    assert ("one row per germplasmDbId; values are group statistics (on basis study_means, mean is the unweighted mean of per-study "
            "means)") in summary_text(registry, per_study, "rank")
    assert "one row per germplasmDbId; values are group statistics)" in summary_text(registry, pooled, "rank")


def test_a_stack_of_pooled_and_study_means_rank_tables_says_they_differ(registry):
    """Review 2026-10-24, MUST-FIX 1: a pooled table stacked with a study_means one said "values are group means" of both."""
    table = two_studies(registry)
    ranked = [run(registry, "rank_entries", artifact_id=table, by="germplasmDbId", value="value", basis=b).data["result_artifact_id"]
              for b in ("pooled", "study_means")]
    text = summary_text(registry, run(registry, "concat_tables", artifact_ids=ranked).artifact_ids[0], "mean")
    assert ("stacked rows of different kinds (rank_entries by germplasmDbId, basis pooled | rank_entries by germplasmDbId, "
            "basis study_means)") in text and "values are group means" not in text


def test_stacked_rank_entries_tables_say_each_stacked_table(registry):
    ranked = [run(registry, "rank_entries", artifact_id=plots(registry, CLONES, study=s), by="germplasmDbId", value="value")
              .data["result_artifact_id"] for s in ("S1", "S2")]
    stacked = run(registry, "concat_tables", artifact_ids=ranked).artifact_ids[0]
    assert "one row per germplasmDbId in each of the 2 stacked tables; values are group means" in summary_text(registry, stacked, "mean")


def test_a_threshold_count_table_says_its_rows_in_each_mode(registry):
    entries = run(registry, "threshold_count", artifact_id=plots(registry), by="germplasmDbId", value="value", count="entries", op="above",
                  threshold=25).data["result_artifact_id"]
    text = summary_text(registry, entries, "value")
    assert "threshold_count result: one row per studyDbId and germplasmDbId (value: the entry mean)" in text and "value used" not in text
    by_plot = run(registry, "threshold_count", artifact_id=plots(registry), by="germplasmDbId", value="value", count="plots", op="above",
                  threshold=25).data["result_artifact_id"]
    assert ("threshold_count result: one row per studyDbId and germplasmDbId (value: the entry mean), but a counted entry has one row per "
            "value used (its plot in unit; a value with no plot ID is not counted)") in summary_text(registry, by_plot, "value")


def test_a_counted_value_with_no_plot_id_is_a_row_but_not_counted(registry):
    """Review 2026-10-24, SHOULD-FIX 3: "one row per plot for each counted entry" was not true of a value whose row names no
    plot: it is a row of the saved table, and the count leaves it out."""
    rows = [row("uA0", "A", "Alpha", "30", obs="oA0"), row("", "A", "Alpha", "40", obs="oA1"), row("uB0", "B", "Beta", "10", obs="oB0")]
    r = run(registry, "threshold_count", artifact_id=save_rows(registry, rows, ["S1"]), by="germplasmDbId", value="value", count="plots",
            op="above", threshold=25)
    assert sorted(x["unit"] for x in at._load(registry, r.data["result_artifact_id"]).rows if x["entry"] == "A") == ["", "uA0"]
    count = next(c for c in claims(r) if c.kind == "count")
    assert count.value == 1 and {e.reason: e.count for e in count.exclusions}["plots left out: no plot ID"] == 1


def test_stacked_threshold_tables_say_which_tables_their_rows_are_in(registry):
    """Review 2026-10-24: the stack suffix was tested on group and rank notes only; review 2, SHOULD-FIX 2: a threshold stack
    filtered to one source kept "from 2 stacked tables"."""
    counted = [run(registry, "threshold_count", artifact_id=plots(registry, study=s), by="germplasmDbId", value="value", op="above",
                   threshold=25).data["result_artifact_id"] for s in ("S1", "S2")]
    stacked = run(registry, "concat_tables", artifact_ids=counted).artifact_ids[0]
    assert ("threshold_count result: one row per studyDbId and germplasmDbId (value: the entry mean), in each of the 2 stacked tables"
            in summary_text(registry, stacked, "value"))
    one = run(registry, "filter_rows", artifact_id=stacked, column="source_artifact_id", op="eq", value=counted[0]).artifact_ids[0]
    text = summary_text(registry, one, "value")
    assert f"(value: the entry mean), in the one table from {counted[0]}" in text and "stacked tables" not in text


def test_the_notes_name_the_by_column(registry):
    """Review 2026-10-24: every test ranked and counted by germplasmDbId, so a note that always named it passed."""
    table = plots(registry)
    ranked = run(registry, "rank_entries", artifact_id=table, by="observationUnitDbId", value="value").data["result_artifact_id"]
    assert "one row per observationUnitDbId; values are group means" in summary_text(registry, ranked, "mean")
    counted = run(registry, "threshold_count", artifact_id=table, by="observationUnitDbId", value="value", op="above",
                  threshold=25).data["result_artifact_id"]
    assert "one row per studyDbId and observationUnitDbId (value: the entry mean)" in summary_text(registry, counted, "value")


# --- the row unit reads the rows it is given (review 2026-10-24: two callers untested) ---------------------------------

def test_a_summary_reads_the_units_of_the_values_it_used(registry):
    """A row with no plot and no valid value is not among the values used, which are one row per observation unit; reading every
    row of the table would say "some rows name no observation unit"."""
    rows = [row("u1", "A", "Alpha", "10", obs="o1"), row("", "A", "Alpha", "NA", obs="o2"), row("u2", "B", "Beta", "20", obs="o3")]
    assert "valid values used as recorded (one row per observation unit)" in summary_text(registry, save_rows(registry, rows, ["S1"]), "value")


def test_a_compare_means_rule_reads_the_units_of_its_rows(registry):
    """With no rows given, the rule would say "one row per observation unit" of a row that names none."""
    c = by_study(compare(registry, save_means(registry, BASE, units={("G1", 1): ""}), save_means(registry, OTHER, study="S2")))
    assert "valid values used as recorded (one row per observation; some rows name no observation unit)" in c["S1"].transformation


# --- (d) "N/A" and "none" name no program ------------------------------------------------------------------------------

def test_n_a_and_none_name_no_program():
    for token in ("N/A", "n/a", "none", "None", " NONE "):
        assert at.unknown_metadata(token)
        record = {"studyDbId": "S1", "studyName": "SYNTHETIC trial", "additionalInfo": {"programDbId": token, "programName": token}}
        exported = srv._study_row(record)
        assert (exported["programDbId"], exported["programName"], exported["programSource"]) == (None, None, "")
    assert not at.unknown_metadata("Nonesuch") and not at.unknown_metadata("NA program")


def test_n_a_and_none_are_no_program_in_every_reader(tmp_path):
    """Review 2026-10-24, SHOULD-FIX 4: (d) was tested through the export only; count_records, filter_rows, the offline study
    filter and the Coordinator's metadata read the same rule."""
    rows = [{"studyDbId": "S1", "studyName": "SYNTHETIC trial 1", "programDbId": "N/A", "programName": "N/A"},
            {"studyDbId": "S2", "studyName": "SYNTHETIC trial 2", "programDbId": "none", "programName": "None"},
            {"studyDbId": "S3", "studyName": "SYNTHETIC trial 3", "programDbId": "5017", "programName": "DR&SS"}]
    registry = ArtifactRegistry(tmp_path / "out", "run_texts_d", base_url=SYNTHETIC_BASE)
    table = catalog(registry, rows)
    r = at.dispatch_analyst(registry, "count_records", {"artifact_id": table, "distinct_by": "programName"})
    assert not r.ok and r.error.code == "incomplete_data" and "2 of 3" in r.error.message
    r = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": table, "column": "programName", "op": "eq", "value": "DR&SS"})
    assert r.ok and r.complete is False and r.warnings[0].startswith("2 of 3 catalog rows have no programName; 1 known matches")
    assert brapi_client._study_program({"studyDbId": "S4", "programDbId": "none", "programName": "N/A",
                                        "additionalInfo": {"programDbId": "5017"}}) == "5017"
    assert co._program_field({"programDbId": "N/A", "programName": "None", "additionalInfoProgramDbId": "5017",
                              "additionalInfoProgramName": "DR&SS"}, "programName") == "DR&SS"
    assert co._program_field({"programDbId": "5017", "programName": "none"}, "programName") is None       # review 2: the field kept
    assert brapi_client._study_program({"studyDbId": "S5", "programDbId": "N/A", "programName": "Alpha"}) is None
