"""
tests/test_named_without_value.py - coach 2026-10-23 09:51 UTC, NEXT 1:
(ii) (5) from the plot records get_observations already read: an entry the study's plot records hold without a value of the trait
     (kept by identity on the table's manifest, entries_without_value) is "named, no <trait> value" in compare_trials_paired and
     stress_indices, with no plot list, no new request and no recipe line; "unknown" stays = absent from the trial (F4 unchanged);
(iv) anova_trial's exclude in the same words: a named entry without a value in the trial is named apart, not refused;
(v) the correlation table shows the claims of correlate_entry_means alone (an allow-list): the T8 review's finding 1 printed another
     tool's R2 as a "Pearson r" under a false heading.
SYNTHETIC tables. No network, no model.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

import analyst_tools as at
import brapi_client as bc
import brapi_mcp_server as srv
import chat_presenter as cp
from artifacts import ArtifactRegistry
from contracts import AnalysisReport, Claim, MeasurementMeta
from part2_support import SYNTHETIC_BASE, synthetic_context

ENTRIES = [f"E{i}" for i in range(8)]
NAMES = {**{f"E{i}": f"PH17.00{i}" for i in range(6)}, "E6": "Beauregard", "E7": "Tanzania"}
WITHOUT = {"E6": "Beauregard", "E7": "Tanzania"}             # in the plot records, no value of the trait (C27.3's two checks)
TOOLS = {"compare_trials_paired": ("base_artifact_id", "other_artifact_id"), "stress_indices": ("nonstress_artifact_id", "stress_artifact_id")}


@pytest.fixture
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "out", "run_named_without_value", base_url=SYNTHETIC_BASE)


def trial(level: float, *, seed: float = 0.0, entries=ENTRIES[:6], plots: int = 2) -> dict[str, list[float]]:
    return {e: [level * (1.0 + 0.3 * math.cos(1.1 * i + seed)) + 0.2 * math.sin(2.3 * i + 1.7 * k + 3.1 * seed) for k in range(plots)]
            for i, e in enumerate(ENTRIES) if e in entries}


def save(registry, study: str, values: dict[str, list[float]], *, without=None, empty=()) -> str:
    """A table as get_observations saves it from the plot records: the rows of the trait, and the study's entries the plot records
    hold without a value of it (manifest.entries_without_value)."""
    rows = [{"observationDbId": f"O{study}-{e}-{k}", "observationUnitDbId": f"{study}-{e}-{k}", "observationUnitName": f"plot {k}",
             "germplasmDbId": e, "germplasmName": NAMES[e], "studyDbId": study, "observationVariableDbId": "V1",
             "observationVariableName": "SYNTHETIC yield", "observationTimeStamp": "", "value": "" if e in empty else repr(float(v))}
            for e, plot_values in values.items() for k, v in enumerate(plot_values, 1)]
    collection = bc.PlotRecordsResult(records=rows, returned_count=len(rows), complete=True, status="complete",
                                      entries_without_value=dict(WITHOUT if without is None else without))
    columns = {col: srv.OBSERVATION_COLUMNS.get(col, "text") for col in rows[0]}
    return registry.save_table(rows, kind="observations", columns=columns, collection=collection, study_ids=[study], variable_ids=["V1"],
                               measurement=MeasurementMeta(trait="SYNTHETIC yield", unit="t/ha"), quality_notes=["SYNTHETIC"]).artifact_id


def tables(registry, tool: str, **kw) -> dict[str, str]:
    first, second = TOOLS[tool]
    return {first: save(registry, "S1", trial(10.0), **kw), second: save(registry, "S2", trial(4.0, seed=0.4), **kw)}


def claims(r) -> list[Claim]:
    return [Claim.model_validate(k) for k in r.data["claims"]]


def answer(ctx, claim_list) -> str:
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claim_list, caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    return cp.answer_summary(view)


# --- (ii) compare_trials_paired and stress_indices --------------------------------------------------------------------------------

@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_an_entry_the_plot_records_hold_without_a_value_is_named_with_no_plot_list(registry, tool):
    """Luna C27.3: Beauregard and Tanzania are in the trials' plot records with no dry-matter value, so no table row shows them.
    get_observations keeps them by identity on the table's manifest; the tool finds them there with no plot list and no new
    request, names them apart, and every number is that of leaving out PH17.005 alone."""
    t = tables(registry, tool)
    r = at.dispatch_analyst(registry, tool, {**t, "exclude": ["Beauregard", "Tanzania", "PH17.005"]})
    assert r.ok, r.error
    want = "left out by name: PH17.005 (1 entry); named, no SYNTHETIC yield value in either trial: Beauregard, Tanzania"
    assert all(want in k.transformation for k in claims(r))
    alone = at.dispatch_analyst(registry, tool, {**t, "exclude": ["PH17.005"]})
    assert alone.ok and [k.value for k in claims(r)] == [k.value for k in claims(alone)]


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_one_trials_plot_records_are_enough_and_an_id_is_found_like_a_name(registry, tool):
    first, second = TOOLS[tool]
    t = {first: save(registry, "S1", trial(10.0)), second: save(registry, "S2", trial(4.0, seed=0.4), without={})}
    r = at.dispatch_analyst(registry, tool, {**t, "exclude": ["E7"]})
    assert r.ok, r.error
    assert all("left out by name: no entry with a mean; named, no SYNTHETIC yield value in either trial: E7" in k.transformation
               for k in claims(r))


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_a_name_absent_from_the_trial_is_refused_exactly_as_before(registry, tool):
    """"Unknown" stays = absent from the trial: a name in no table and not among the plot records' entries without a value is
    refused with the message it had before (no plot list named)."""
    r = at.dispatch_analyst(registry, tool, {**tables(registry, tool), "exclude": ["Resisto"]})
    assert not r.ok and r.error.code == "invalid_argument"
    assert r.error.message.startswith("exclude 'Resisto' is not an entry ID or name in art_") and "plot list" not in r.error.message


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_two_entries_with_one_name_and_a_name_the_tables_hold(registry, tool):
    """A name the plot records give to two entries without a value is refused, never guessed; a name a table holds is resolved
    there, whatever the plot records say (here they also list another entry under it)."""
    first, second = TOOLS[tool]
    twins = {"E6": "Twin", "E7": "Twin"}
    t = {first: save(registry, "S1", trial(10.0), without=twins), second: save(registry, "S2", trial(4.0, seed=0.4), without=twins)}
    r = at.dispatch_analyst(registry, tool, {**t, "exclude": ["Twin"]})
    assert not r.ok and r.error.code == "invalid_argument" and "is the name of 2 entries" in r.error.message
    shadow = {"X9": "PH17.005"}
    t = {first: save(registry, "S1", trial(10.0), without=shadow), second: save(registry, "S2", trial(4.0, seed=0.4), without=shadow)}
    r = at.dispatch_analyst(registry, tool, {**t, "exclude": ["PH17.005"]})
    assert r.ok and all("left out by name: PH17.005 (1 entry)" in k.transformation and "named, no" not in k.transformation
                        for k in claims(r))


# --- (iv) anova_trial ------------------------------------------------------------------------------------------------------------

def test_anova_trial_names_an_entry_without_a_value_in_the_trial(registry):
    """anova_trial's exclude in T4's and T5's words: Tanzania has plots but no valid value, Beauregard is only among the plot
    records' entries without a value; both are named apart, and the analysis is that of leaving out PH17.005 alone. A name absent
    from the trial is refused as before."""
    aid = save(registry, "S1", {**trial(10.0), "E7": [0.0, 0.0]}, without={"E6": "Beauregard"}, empty=("E7",))
    r = at.dispatch_analyst(registry, "anova_trial", {"artifact_id": aid, "exclude": ["Beauregard", "Tanzania", "PH17.005"]})
    assert r.ok, r.error
    want = "left out by name: PH17.005 (2 plots); named, no SYNTHETIC yield value in the trial: Beauregard, Tanzania"
    assert all(want in k.transformation for k in claims(r))
    alone = at.dispatch_analyst(registry, "anova_trial", {"artifact_id": aid, "exclude": ["PH17.005"]})
    assert alone.ok and [k.value for k in claims(r)] == [k.value for k in claims(alone)]
    only = at.dispatch_analyst(registry, "anova_trial", {"artifact_id": aid, "exclude": ["Beauregard"]})
    assert only.ok and all("left out by name: no entry with a value; named, no SYNTHETIC yield value in the trial: Beauregard"
                           in k.transformation for k in claims(only))
    r = at.dispatch_analyst(registry, "anova_trial", {"artifact_id": aid, "exclude": ["Resisto"]})
    assert not r.ok and "exclude 'Resisto' is not an entry ID or name" in r.error.message


# --- the review of this set (findings 1, 3, 4) ----------------------------------------------------------------------------------

def plot_list(registry, study: str, entries) -> str:
    """The study's plot list as get_observation_units saves it, one row per plot of the measurements above (no design fields)."""
    rows = [{"observationUnitDbId": f"{study}-{e}-{k}", "observationUnitName": f"plot {k}", "germplasmDbId": e, "germplasmName": NAMES[e],
             "studyDbId": study, "levelName": "plot", "levelCode": str(k), "observationUnitPosition": "{}"} for e in entries for k in (1, 2)]
    return registry.save_table(rows, kind="observation_units", columns=dict(srv.UNIT_COLUMNS), complete=True, study_ids=[study],
                               variable_ids=[], quality_notes=["SYNTHETIC"]).artifact_id


def test_plot_design_keeps_the_record_so_anova_trial_on_its_table_names_the_entry(registry):
    """Finding 1: anova_trial reads plot_design's table (the plots with their reps), which had no record, so Beauregard was refused
    there. plot_design keeps every row of the measurements and now carries the record; a name absent from the trial is still refused."""
    measured = save(registry, "S1", trial(10.0), without={"E6": "Beauregard"})
    designed = at.dispatch_analyst(registry, "plot_design", {"roster_artifact_id": plot_list(registry, "S1", ENTRIES[:6]), "artifact_id": measured})
    assert designed.ok, designed.error
    table = designed.data["artifact_id"]
    assert registry.manifest(table).entries_without_value == {"E6": "Beauregard"}
    r = at.dispatch_analyst(registry, "anova_trial", {"artifact_id": table, "exclude": ["Beauregard", "PH17.005"]})
    assert r.ok, r.error
    assert all("left out by name: PH17.005 (2 plots); named, no SYNTHETIC yield value in the trial: Beauregard" in k.transformation
               for k in claims(r))
    r = at.dispatch_analyst(registry, "anova_trial", {"artifact_id": table, "exclude": ["Resisto"]})
    assert not r.ok and "exclude 'Resisto' is not an entry ID or name" in r.error.message


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_a_filtered_table_records_nothing_and_one_trials_record_never_speaks_for_both(registry, tool):
    """Finding 3: filter_rows can drop the rows of an entry that has values in the trial; its table records nothing (None), and with
    one table unrecorded no name is looked up in the other's record: Beauregard is refused, as before this change-set."""
    first, second = TOOLS[tool]
    base = save(registry, "S1", trial(10.0))
    filtered = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": base, "column": "observationUnitName", "op": "eq", "value": "plot 1"})
    assert filtered.ok, filtered.error
    narrow = filtered.data["artifact_id"] if "artifact_id" in filtered.data else filtered.artifact_ids[-1]
    assert registry.manifest(narrow).entries_without_value is None
    r = at.dispatch_analyst(registry, tool, {first: narrow, second: save(registry, "S2", trial(4.0, seed=0.4)), "exclude": ["Beauregard"]})
    assert not r.ok and "exclude 'Beauregard' is not an entry ID or name" in r.error.message


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_n_kept_counts_the_entries_the_prefix_keeps_with_a_mean_in_either_trial(registry, tool):
    """Finding 4, the coach's form "M entries outside PH17.* (N kept)": N counts the prefix's entries with a mean in either trial,
    as M does; PH17.005 has a mean in the base trial only, so N is 6 while 5 entries are paired."""
    first, second = TOOLS[tool]
    t = {first: save(registry, "S1", trial(10.0, entries=ENTRIES), without={}),
         second: save(registry, "S2", trial(4.0, seed=0.4, entries=[e for e in ENTRIES if e != "E5"]), without={})}
    r = at.dispatch_analyst(registry, tool, {**t, "name_prefix": "PH17."})
    assert r.ok, r.error
    assert r.data["n_pairs"] == 5 and all("left out by prefix: 2 entries outside PH17.* (6 kept)" in k.transformation for k in claims(r))


# --- (v) the correlation table's allow-list --------------------------------------------------------------------------------------

def test_the_correlation_table_shows_the_claims_of_correlate_entry_means_alone(tmp_path):
    """The T8 review (finding 1): the correlation table took every claim of kind "correlation" but compare_trials_paired's, so
    another tool's R2 was printed as a Pearson r under "4 entries in both tables". It now allows correlate_entry_means' claims
    alone: the same claims under another tool's namespace are not shown there."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_cor_allow")
    x, y = save(ctx.registry, "S1", trial(10.0), without={}), save(ctx.registry, "S2", trial(4.0, seed=0.4), without={})
    r = at.dispatch_analyst(ctx.registry, "correlate_entry_means", {"artifact_ids": [x, y]})
    assert r.ok, r.error
    mine = claims(r)
    assert "Correlation of entry means" in answer(ctx, mine)
    other = [c.model_copy(update={"claim_id": c.claim_id.replace("_cor-", "_zzz-")}) for c in mine]
    assert all(c.claim_id != o.claim_id for c, o in zip(mine, other))
    assert "Correlation of entry means" not in answer(ctx, other)
