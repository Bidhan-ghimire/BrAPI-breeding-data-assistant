"""
tests/test_compare_means.py - R2 tool 1 (coach, 2026-10-21): the percent change (or reduction) between two studies' means.

14 benchmark sub-questions check a percent between two studies' means of one trait: "mean of genotype means on the N
genotypes common to both; % reduction = (control - drought)/control x 100" (C27.1, C27.2, C33.1, C34.1, D14.1, D14.2,
D14.6, D16.1, D16.2) or "gain % = (mean120 - mean90)/mean90 x 100" (C30.1-C30.4, C31.1). compare_means states both means and
the percent, the formula named in the claim. Known small SYNTHETIC tables, every number worked out by hand in the comments.
No network, no model, no real data. Strengthened after the reviews of 2026-10-22 (round 0: the analyst's table guard, claim
IDs per comparison, the base study, the incomplete mark, the exclusions, the edges; round 2, coach 03:48 UTC: the plan binds
the base study and the percent, each excluded entry counted once, the percent shown to one decimal, the planner's wording,
one test per surviving bug).
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from types import SimpleNamespace

import anyio
import pytest

import analyst_tools as at
import brapi_mcp_server as srv
from artifacts import ArtifactRegistry
from contracts import Claim, MeasurementMeta
from part2_support import SYNTHETIC_BASE

# entry ID, name, plot values.  Base S1 (the control): G1 11, G2 21, G3 16, G4 31 (only here).
# Other S2 (the drought arm): G1 6, G2 11, G3 9, G7 2 (only here).  Shared: G1, G2, G3.
# Base mean of the shared entry means 16; other 26/3 = 8.6667; change (8.6667 - 16) / 16 = -45.8333 %; reduction +45.8333 %.
BASE = [("G1", "Alpha", ["10", "12"]), ("G2", "Beta", ["20", "22"]), ("G3", "Gamma", ["15", "17"]), ("G4", "Check-1", ["30", "32"])]
OTHER = [("G1", "Alpha", ["5", "7"]), ("G2", "Beta", ["10", "12"]), ("G3", "Gamma", ["9", "9"]), ("G7", "Omega", ["1", "3"])]


def save(registry: ArtifactRegistry, entries, *, study="S1", variable="V1", complete=True, unit="t/ha", units=None, studies=None) -> str:
    rows = [{"observationDbId": f"O{study}{g}{i}", "observationUnitDbId": (units or {}).get((g, i), f"P{study}{g}{i}"),
             "observationUnitName": f"plot {g}-{i}", "germplasmDbId": g, "germplasmName": name, "studyDbId": study,
             "observationVariableDbId": variable, "observationVariableName": "SYNTHETIC yield", "observationTimeStamp": "", "value": value}
            for g, name, values in entries for i, value in enumerate(values)]
    columns = {c: srv.OBSERVATION_COLUMNS.get(c, "text") for c in rows[0]}
    return registry.save_table(rows, kind="observations", columns=columns, complete=complete, study_ids=studies or [study],
                               variable_ids=[variable], measurement=MeasurementMeta(trait="SYNTHETIC yield", unit=unit)).artifact_id


@pytest.fixture
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "out", "run_compare", base_url=SYNTHETIC_BASE)


def compare(registry, base, other, **args):
    """The base table's own study is the plan's base study, and the percent a change, unless a test says otherwise."""
    args.setdefault("base_study_db_id", registry.manifest(base).study_ids[0])
    args.setdefault("percent", "change")
    return at.dispatch_analyst(registry, "compare_means", {"base_artifact_id": base, "other_artifact_id": other, **args})


def by_study(result) -> dict[str, Claim]:
    claims = [Claim.model_validate(c) for c in result.data["claims"]]
    return {("percent" if c.kind == "ratio" else c.filters.get("studyDbId")): c for c in claims}


def exclusions(claim: Claim) -> dict[str, int]:
    return {e.reason: e.count for e in claim.exclusions}


def entry_exclusions(claim: Claim) -> int:
    return sum(e.count for e in claim.exclusions if "entries" in e.reason)


# --- 1. the numbers ------------------------------------------------------------------------------

def test_two_study_means_over_the_shared_entries_and_the_percent_change(registry):
    r = compare(registry, save(registry, BASE), save(registry, OTHER, study="S2"))
    assert r.ok and r.complete and r.data["entries_used"] == ["G1", "G2", "G3"] and r.data["only_in_base"] == ["G4"]
    assert r.data["only_in_other"] == ["G7"]
    c = by_study(r)
    assert set(c) == {"S1", "S2", "percent"}
    assert c["S1"].kind == "mean" and c["S1"].value == 16 and c["S1"].unit == "t/ha" and c["S1"].denominator.n == 3
    assert c["S2"].kind == "mean" and math.isclose(c["S2"].value, 26 / 3) and c["S2"].denominator.n == 3
    assert "mean of entry means over the 3 entries used: those with a mean in both tables" in c["S1"].transformation
    p = c["percent"]
    assert p.kind == "ratio" and p.unit == "%" and math.isclose(p.value, (26 / 3 - 16) / 16 * 100) and p.denominator.n == 3
    assert "percent change = (other - base) / base x 100" in p.transformation and "base S1, other S2" in p.transformation
    assert p.filters == {"studyDbId": "S1 vs S2"}
    assert sorted(p.source_artifact_hashes) == sorted([c["S1"].source_artifact_hashes[0], c["S2"].source_artifact_hashes[0]])
    assert "descriptive" in p.transformation and "not a recommendation" in p.transformation


def test_a_reduction_is_the_same_drop_as_a_positive_percent(registry):
    """'% reduction = (control - drought)/control x 100' (C27.1, C33.1): the drop is stated as a positive reduction."""
    p = by_study(compare(registry, save(registry, BASE), save(registry, OTHER, study="S2"), percent="reduction"))["percent"]
    assert math.isclose(p.value, (16 - 26 / 3) / 16 * 100) and p.value > 0
    assert "percent reduction = (base - other) / base x 100" in p.transformation


def test_every_entry_of_each_table_when_asked(registry):
    """shared_entries_only=False: base (11+21+16+31)/4 = 19.75, other (6+11+9+2)/4 = 7, reduction (19.75-7)/19.75 = 64.557 %.
    With the other table one entry shorter (no G7) the percent still counts the base table's entries."""
    r = compare(registry, save(registry, BASE), save(registry, OTHER, study="S2"), percent="reduction", shared_entries_only=False)
    c = by_study(r)
    assert c["S1"].value == 19.75 and c["S1"].denominator.n == 4 and c["S2"].value == 7 and c["S2"].denominator.n == 4
    assert math.isclose(c["percent"].value, (19.75 - 7) / 19.75 * 100)
    assert "mean of entry means over the 4 entries of this table" in c["S1"].transformation
    shorter = by_study(compare(registry, save(registry, BASE, study="S3"), save(registry, OTHER[:3], study="S4"), shared_entries_only=False))
    assert shorter["S4"].denominator.n == 3 and shorter["percent"].denominator.n == 4 and shorter["percent"].denominator.name == "entries of the base table"


def test_left_out_entries_by_id_or_name_and_said_so(registry):
    """exclude Beta (G2): base (11+16)/2 = 13.5, other (6+9)/2 = 7.5, reduction 44.444 %; the text says one was left out.
    Each claim names its study in every excluded line (round 3 (B))."""
    base, other = save(registry, BASE), save(registry, OTHER, study="S2")
    for exclude in (["G2"], ["Beta"]):
        c = by_study(compare(registry, base, other, percent="reduction", exclude=exclude))
        assert c["S1"].value == 13.5 and c["S2"].value == 7.5 and math.isclose(c["percent"].value, (13.5 - 7.5) / 13.5 * 100)
        assert "over the 2 entries used: those with a mean in both tables, after 1 left out" in c["S1"].transformation
        assert exclusions(c["S1"]) == {"S1: entries left out before comparing": 1, "S1: entries without a usable mean in S2": 1}
        assert exclusions(c["S2"]) == {"S2: entries left out before comparing": 1, "S2: entries without a usable mean in S1": 1}
        assert exclusions(c["percent"]) == {"S1 vs S2: entries left out before comparing": 1,
                                            "S1 vs S2: entries with a mean in only one of S1 and S2": 2}
    refused = compare(registry, base, other, exclude=["Zeta"])
    assert not refused.ok and refused.error.code == "invalid_argument"


def test_the_mean_of_plot_values_instead_of_entry_means(registry):
    """mean_of='values' over the shared entries' plots: base (10+12+20+22+15+17)/6 = 16, other (5+7+10+12+9+9)/6 = 8.6667."""
    c = by_study(compare(registry, save(registry, BASE), save(registry, OTHER, study="S2"), mean_of="values"))
    assert c["S1"].value == 16 and c["S1"].denominator.n == 6 and math.isclose(c["S2"].value, 52 / 6) and c["S2"].denominator.n == 6
    assert "mean of the valid values of the 3 entries used: those with a mean in both tables" in c["S1"].transformation
    unequal = [("G1", "Alpha", ["10", "12", "14"]), ("G2", "Beta", ["20"])]          # entry means 12 and 20; values mean 14
    c = by_study(compare(registry, save(registry, unequal, study="S3"), save(registry, unequal[:1] + [("G2", "Beta", ["21"])], study="S4"),
                         mean_of="values"))
    assert c["S3"].value == 14
    c = by_study(compare(registry, save(registry, unequal, study="S5"), save(registry, unequal[:1] + [("G2", "Beta", ["21"])], study="S6")))
    assert c["S5"].value == 16


def test_entries_with_too_few_units_are_left_out_of_both(registry):
    """min_independent_n=2: G3 has one plot in the other table, so it is not shared: base (11+21)/2 = 16, other (6+11)/2 = 8.5."""
    other = [("G1", "Alpha", ["5", "7"]), ("G2", "Beta", ["10", "12"]), ("G3", "Gamma", ["9"])]
    r = compare(registry, save(registry, BASE), save(registry, other, study="S2"), percent="reduction", min_independent_n=2)
    c = by_study(r)
    assert r.data["entries_used"] == ["G1", "G2"] and c["S1"].value == 16 and c["S2"].value == 8.5
    assert math.isclose(c["percent"].value, (16 - 8.5) / 16 * 100)
    assert any(e["entry"] == "G3" and "fewer than 2" in e["reason"] for e in r.data["not_used"])
    assert {e["entry"]: e["reason"] for e in r.data["not_used"]}["G4"] == "only in the base table"
    assert exclusions(c["S2"])["S2: entries under min_independent_n=2"] == 1


def test_duplicate_plots_are_refused_unless_a_rule_is_named(registry):
    """Two rows for one plot (P1) in the base table: refused by default; mean_per_unit averages them first (G1: (10+12)/2 = 11
    for P1, 14 for P2, entry mean 12.5)."""
    duplicated = save(registry, [("G1", "Alpha", ["10", "12", "14"]), ("G2", "Beta", ["20", "22"])], units={("G1", 1): "PS1G10"})
    other = save(registry, [("G1", "Alpha", ["5", "7"]), ("G2", "Beta", ["10", "12"])], study="S2")
    assert not compare(registry, duplicated, other).ok
    c = by_study(compare(registry, duplicated, other, duplicate_policy="mean_per_unit"))
    assert c["S1"].value == (12.5 + 21) / 2 and "duplicate_policy=mean_per_unit" in c["S1"].transformation


# --- 2. the base study, the tables, the edges ---------------------------------------------------------

def test_the_base_table_must_be_the_plans_base_study(registry):
    """base_study_db_id (the plan's reference study) is required: code checks the analyst's choice of base table, so a
    swapped pair is refused instead of turning a reduction around."""
    base, other = save(registry, BASE), save(registry, OTHER, study="S2")
    assert compare(registry, base, other, base_study_db_id="S1").ok
    swapped = compare(registry, other, base, base_study_db_id="S1")
    assert not swapped.ok and swapped.error.code == "invalid_argument" and "not of the plan's base study 'S1'" in swapped.error.message
    for wrong in ("S9", "V1"):
        assert not compare(registry, base, other, base_study_db_id=wrong).ok, wrong
    missing = at.dispatch_analyst(registry, "compare_means", {"base_artifact_id": base, "other_artifact_id": other, "percent": "change"})
    assert not missing.ok and missing.error.code == "invalid_argument" and "base_study_db_id" in missing.error.message
    again = save(registry, OTHER, study="S1")                                             # another table of the base study
    refused = compare(registry, base, again, base_study_db_id="S1")
    assert not refused.ok and "also a table of the base study" in refused.error.message
    two = save(registry, BASE, study="S1", studies=["S1", "S5"])                         # a base table of two studies
    assert not compare(registry, two, other, base_study_db_id="S1").ok


def test_the_tables_must_measure_the_same_variable_in_the_same_unit_and_differ(registry):
    base = save(registry, BASE)
    for other in (save(registry, OTHER, study="S2", variable="V2"), save(registry, OTHER, study="S3", unit="kg/plot")):
        refused = compare(registry, base, other)
        assert not refused.ok and refused.error.code == "invalid_argument"
    same = compare(registry, base, base)
    assert not same.ok and same.error.code == "invalid_argument"                                  # one table against itself
    manifest, rows = registry.load(base)
    twin = registry.save_table(rows, kind="observations", columns=dict(manifest.columns), complete=True, study_ids=["S2"],
                               variable_ids=["V1"], measurement=manifest.measurement).artifact_id
    copied = compare(registry, base, twin)
    assert not copied.ok and copied.error.code == "invalid_argument"                           # the same data under two handles
    named = compare(registry, base, save(registry, OTHER, study="S4"), by="germplasmName")
    assert not named.ok and named.error.code == "invalid_argument"


def test_no_percent_of_a_base_that_is_zero_or_negative_and_none_without_shared_entries(registry):
    for n, (values, why) in enumerate(((["0", "0"], "counts as zero"), (["-2", "-2"], "is negative"))):
        p = by_study(compare(registry, save(registry, [("G1", "Alpha", values), ("G2", "Beta", values)], study=f"S1{n}"),
                             save(registry, OTHER, study=f"S2{n}")))["percent"]
        assert p.value is None and f"the base mean {why}" in p.missing_reason and "a percent of it is not defined" in p.missing_reason
    apart = compare(registry, save(registry, [("G8", "Theta", ["1", "2"])], study="S7"), save(registry, OTHER, study="S8"))
    assert not apart.ok and apart.error.code == "incomplete_data"


def test_an_incomplete_table_is_refused_unless_allowed_and_marks_everything_it_shapes(registry):
    """Review 2026-10-22: with shared entries, which entries the base mean uses depends on the other table, so an incomplete
    other table marks the base mean too; the result, the saved table and the percent are all incomplete."""
    base, partial = save(registry, BASE), save(registry, OTHER, study="S2", complete=False)
    assert not compare(registry, base, partial).ok
    allowed = compare(registry, base, partial, allow_incomplete=True)
    c = by_study(allowed)
    assert allowed.ok and not allowed.complete and not registry.manifest(allowed.data["result_artifact_id"]).complete
    assert all("INCOMPLETE" in c[k].transformation for k in ("S1", "S2", "percent"))
    own = by_study(compare(registry, base, partial, allow_incomplete=True, shared_entries_only=False))
    assert "INCOMPLETE" not in own["S1"].transformation and "INCOMPLETE" in own["S2"].transformation   # each its own entries


def test_each_comparison_has_its_own_claims_and_the_entry_table_is_saved(registry):
    """Review 2026-10-22: the same base against two other tables (one optimum, two drought arms) gave colliding claim IDs."""
    base, other, third = save(registry, BASE), save(registry, OTHER, study="S2"), save(registry, OTHER[:2], study="S3")
    ids = [sorted(c["claim_id"] for c in compare(registry, base, b, **args).data["claims"])
           for b, args in ((other, {}), (other, {"percent": "reduction"}), (other, {"exclude": ["G2"]}), (other, {"mean_of": "values"}),
                           (other, {"shared_entries_only": False}), (other, {"min_independent_n": 2}), (third, {}))]
    assert len({i for group in ids for i in group}) == 3 * len(ids)                       # no two comparisons share a claim ID
    r = compare(registry, base, other)
    manifest, rows = registry.load(r.data["result_artifact_id"])
    assert manifest.kind == "analysis_result" and {row["entry"] for row in rows} == {"G1", "G2", "G3", "G4", "G7"}
    used = {row["entry"]: row["used"] for row in rows}
    assert used == {"G1": "true", "G2": "true", "G3": "true", "G4": "false", "G7": "false"}
    assert {row["entry"]: row["reason"] for row in rows if row["used"] == "false"} == {"G4": "only in the base table",
                                                                                       "G7": "only in the other table"}


# --- 3. the wiring ------------------------------------------------------------------------------

def test_it_is_one_of_the_analysts_tools():
    assert "compare_means" in at.ANALYST_TOOL_NAMES and len(at.ANALYST_TOOL_NAMES) == 19     # the nineteenth, after T2-T7a (item 12)
    schema = next(s for s in at.analyst_tool_schemas() if s["name"] == "compare_means")
    assert "percent" in schema["description"].lower() and "not a recommendation" in schema["description"]
    percent = schema["inputSchema"]["properties"]["percent"]["description"]
    assert "change: (other - base) / base x 100" in percent and "reduction: (base - other) / base x 100" in percent
    assert {"base_study_db_id", "percent"} <= set(schema["inputSchema"]["required"])          # round 3 (A): no silent default


def test_the_planner_and_the_analyst_know_the_tool():
    import agents.analyst as an
    import agents.coordinator as co

    prompt = " ".join(co.COORDINATOR_SYSTEM_PROMPT.split())
    assert "compare_means" in co.ANALYST_ACTIONS and "compare_means(base_study_db_id=" in prompt and "shared_entries_only=false" in prompt
    assert "the change on the entries in both stays compare_trials_paired" in prompt                 # T4 keeps the paired change
    assert 'statistic kind "ratio"' in prompt
    assert ('percent="reduction" when the question asks for a reduction, a loss, a drop, "lower" or "below": (base - other) / base '
            'x 100') in prompt
    assert ('percent="change" when it asks for a change, a gain, an increase, "higher" or "above": (other - base) / base x 100'
            in prompt)
    assert "compare_means" in an.NUMERIC_TOOLS and "compare_means" in an.DERIVING_TOOLS and "compare_means" in an.NO_CLAIM_NUDGE
    analyst = " ".join(an.ANALYST_SYSTEM_PROMPT.split())
    assert "nineteen fixed tools" in analyst and "the plan's base_study_db_id and percent, and that study's table as base_artifact_id" in analyst


def test_the_analysts_guard_refuses_a_table_never_supplied(registry):
    """Review 2026-10-22: the guard read three argument names, so base/other_artifact_id could read a table never supplied
    (and, once read, use it anywhere). Every *artifact_id argument is checked now."""
    import agents.analyst as an

    base, other, stray = save(registry, BASE), save(registry, OTHER, study="S2"), save(registry, OTHER, study="S3", complete=False)
    tools = an.GuardedAnalystTools(at.AnalystTools(registry), [base, other])

    async def go():
        for handle in (base, other):
            assert (await tools.call("table_info", {"artifact_id": handle})).ok
        refused = await tools.call("compare_means", {"base_artifact_id": base, "other_artifact_id": stray, "base_study_db_id": "S1",
                                                     "percent": "change", "allow_incomplete": True})
        later = await tools.call("group_stats", {"artifact_id": stray, "allow_incomplete": True})
        fine = await tools.call("compare_means", {"base_artifact_id": base, "other_artifact_id": other, "base_study_db_id": "S1",
                                                  "percent": "change"})
        return refused, later, fine

    refused, later, fine = anyio.run(go)
    assert not refused.ok and refused.error.code == "not_authorized" and stray in refused.error.message
    assert not later.ok and later.error.code == "not_authorized"
    assert fine.ok


META_ROWS = ([{"studyDbId": "S1", "studyName": "SYNTHETIC control"}, {"studyDbId": "S2", "studyName": "SYNTHETIC drought"}],
             [{"observationVariableDbId": "V1", "observationVariableName": "fresh root yield"}])
QUESTION = "What is the percent reduction in fresh root yield in study S2 compared with study S1?"
NOW = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)


def _plan(inputs):
    import agents.coordinator as co

    meta = co.KnownMetadata.from_rows(*META_ROWS, base_url=SYNTHETIC_BASE, complete=True)
    fetch = [co.DraftStep(step_id=f"step_{s}", agent="retriever", action="get_observations", inputs={"study_db_id": s, "variable_db_id": "V1"})
             for s in ("S1", "S2")]
    draft = co.PlanDraft(interpretation="percent reduction of fresh root yield, S2 against S1",
                         statistic={"kind": "ratio", "denominator": "entries in both studies", "grouping": []},
                         scope=co.DraftScope(study_ids=["S1", "S2"], variable_ids=["V1"]),
                         steps=[*fetch, co.DraftStep(step_id="step_compare", agent="analyst", action="compare_means",
                                                     inputs=inputs, depends_on=["step_S1", "step_S2"])])
    return co.assemble_plan(QUESTION, draft, meta, base_url=SYNTHETIC_BASE, now=NOW, plan_id="plan_compare")


def test_the_base_study_must_be_one_of_the_plans_studies():
    _plan_, problems, _ = _plan({"base_study_db_id": "S9", "percent": "reduction"})
    assert any("input base_study_db_id='S9' is outside the plan's scope" in p for p in problems)
    ready, problems, _ = _plan({"base_study_db_id": "S1", "percent": "reduction"})
    assert problems == [] and ready.status == "ready" and ready.statistic.kind == "ratio"


def test_the_answer_labels_the_percent_and_shows_both_means(registry):
    import chat_presenter as cp
    from contracts import AnalysisReport

    base, other = save(registry, BASE), save(registry, OTHER, study="S2")
    for percent, label, shown_percent in (("reduction", "Percent reduction", "45.8"), ("change", "Percent change", "-45.8")):
        r = compare(registry, base, other, percent=percent)
        claims = [Claim.model_validate(c) for c in r.data["claims"]]
        view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claims, caveats=[]),
                               retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=SimpleNamespace(registry=registry), notes=[])
        shown = cp.answer_summary(view)
        assert f"| {label} | {shown_percent} % |" in shown and "Ratio" not in shown
        assert "| Mean | 16 t/ha |" in shown and "| Mean | " + cp._number(26 / 3) + " t/ha |" in shown
    assert "compare_means" not in cp._human_note("compare_means: the base mean counts as zero")


def test_a_check_in_one_table_only_can_be_left_out_but_an_ambiguous_name_cannot(registry):
    """exclude Check-1 (G4, only in base) with every entry of each table: base (11+21+16)/3 = 16, other 7 as before."""
    c = by_study(compare(registry, save(registry, BASE), save(registry, OTHER, study="S2"), shared_entries_only=False,
                         exclude=["Check-1"]))
    assert c["S1"].value == 16 and c["S1"].denominator.n == 3 and c["S2"].value == 7
    twins = [("G8", "Twin", ["1", "2"]), ("G9", "Twin", ["3", "4"]), ("G1", "Alpha", ["5", "6"])]
    refused = compare(registry, save(registry, twins, study="S9"), save(registry, [("G8", "Twin", ["1", "2"]), ("G1", "Alpha", ["2", "3"])],
                                                                    study="S10"), exclude=["Twin"])
    assert not refused.ok and refused.error.code == "invalid_argument" and "name of 2 entries" in refused.error.message


# --- 4. round 3 (coach 2026-10-22 03:48 UTC) -----------------------------------------------------------
# (A) the plan binds the base study and the percent. Review of round 1 (A): the analyst swapped base and other CONSISTENTLY
# (base_study_db_id "S2" with the S2 table as base, while the plan said S1) and the answer read "Percent reduction -84.6 %".

@pytest.mark.parametrize("inputs, missing", [({}, ("base_study_db_id", "percent")), ({"base_study_db_id": "S1"}, ("percent",)),
                                             ({"percent": "reduction"}, ("base_study_db_id",)),
                                             ({"base_study_db_id": "S1", "percent": "banana"}, ("percent",)),
                                             ({"base_study_db_id": "", "percent": "change"}, ("base_study_db_id",))])
def test_a_compare_means_step_must_name_its_base_study_and_its_percent(inputs, missing):
    _plan_, problems, _ = _plan(inputs)
    for name in missing:
        assert any(p.startswith(f"step_compare: compare_means needs {name}") for p in problems), (name, problems)


def test_the_analyst_must_call_compare_means_as_the_plan_says(registry):
    import agents.analyst as an

    base, other = save(registry, BASE), save(registry, OTHER, study="S2")

    def guard(planned):
        return an.GuardedAnalystTools(at.AnalystTools(registry), [base, other], inspected={base, other}, planned_comparisons=planned)

    tools = guard([("S1", "reduction")])
    calls = {"swapped": {"base_artifact_id": other, "other_artifact_id": base, "base_study_db_id": "S2", "percent": "reduction"},
             "turned": {"base_artifact_id": base, "other_artifact_id": other, "base_study_db_id": "S1", "percent": "change"},
             "unsaid": {"base_artifact_id": base, "other_artifact_id": other, "base_study_db_id": "S1"}}
    for name, args in calls.items():
        refused = anyio.run(tools.call, "compare_means", args)
        assert not refused.ok and refused.error.code == "not_authorized" and tools.records[-1].refused_here, name
        assert "base_study_db_id='S1', percent='reduction'" in refused.error.message, name
    planned = anyio.run(tools.call, "compare_means", {**calls["turned"], "percent": "reduction"})
    assert planned.ok and by_study(planned)["percent"].value > 0
    none_planned = anyio.run(guard([]).call, "compare_means", {**calls["turned"], "percent": "reduction"})
    assert not none_planned.ok and "the plan has none" in none_planned.error.message
    assert anyio.run(guard(None).call, "compare_means", calls["turned"]).ok          # no plan to bind (the MCP server's own guard)


def test_the_plans_comparisons_reach_the_analysts_guard(registry, tmp_path):
    import agents.analyst as an
    import run
    from llm import Budget, FakeModelClient, reply_text, reply_tools
    from part2_support import synthetic_context

    plan, problems, _ = _plan({"base_study_db_id": "S1", "percent": "reduction"})
    assert problems == []
    inputs = run.analysis_input(plan, ["art_x_0001"], notes=["n"])
    assert [(c.base_study_db_id, c.percent) for c in inputs.planned_comparisons] == [("S1", "reduction")]

    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out2", run_id="run_bind")
    base, other = save(ctx.registry, BASE), save(ctx.registry, OTHER, study="S2")
    swapped = {"base_artifact_id": other, "other_artifact_id": base, "base_study_db_id": "S2", "percent": "reduction"}
    model = FakeModelClient([reply_tools([("table_info", json.dumps({"artifact_id": base})), ("table_info", json.dumps({"artifact_id": other}))]),
                             reply_tools([("compare_means", json.dumps(swapped))]),
                             reply_tools([("compare_means", json.dumps({**swapped, "base_artifact_id": base, "other_artifact_id": other,
                                                                        "base_study_db_id": "S1"}))]),
                             reply_text(json.dumps({"status": "completed", "payload": {"methods": [], "caveats": [], "narrative": "",
                                                                                       "key_claim_ids": []}}))], reported_model="mock-model")
    inputs = run.analysis_input(plan, [base, other], notes=[])
    report, _agent, _ = anyio.run(lambda: an.run_analyst(inputs, model=model, ctx=ctx, budget=Budget(), run_id="run_bind",
                                                         log_dir=ctx.registry.out_root))
    percent = next(c for c in report.claims if c.kind == "ratio")
    assert report.status == "completed" and percent.value > 0 and percent.filters == {"studyDbId": "S1 vs S2"}
    assert any("compare_means must follow the plan" in c for c in report.caveats)


# (B) each excluded entry counted once, only what that mean used, the study named.

def test_a_left_out_entry_counts_only_in_the_table_that_holds_it(registry):
    """Review of round 1 (B): exclude G7 (only in the other table) with every entry of each table: the base mean (19.75 over 4)
    said "after 1 left out" but listed nothing."""
    c = by_study(compare(registry, save(registry, BASE), save(registry, OTHER, study="S2"), shared_entries_only=False, exclude=["G7"]))
    assert c["S1"].value == 19.75 and "left out" not in c["S1"].transformation and exclusions(c["S1"]) == {}
    assert math.isclose(c["S2"].value, 26 / 3) and "after 1 left out" in c["S2"].transformation
    assert exclusions(c["S2"]) == {"S2: entries left out before comparing": 1}
    assert exclusions(c["percent"]) == {"S1 vs S2: entries left out before comparing": 1}


def test_an_entry_is_counted_once_in_each_claim(registry):
    """Review of round 1 (B): exclude G2 with min_independent_n=2 while G2 has one plot in the other table: "left out 1" and
    "under min_independent_n 1" were one entry, and the percent listed 4 entries for 3."""
    other = [("G1", "Alpha", ["5", "7"]), ("G2", "Beta", ["10"]), ("G3", "Gamma", ["9", "9"]), ("G7", "Omega", ["1", "3"])]
    r = compare(registry, save(registry, BASE), save(registry, other, study="S2"), percent="reduction", exclude=["G2"], min_independent_n=2)
    c = by_study(r)
    assert c["S1"].value == 13.5 and c["S2"].value == 7.5
    assert exclusions(c["S1"]) == {"S1: entries left out before comparing": 1, "S1: entries without a usable mean in S2": 1}
    assert exclusions(c["S2"]) == {"S2: entries left out before comparing": 1, "S2: entries without a usable mean in S1": 1}
    assert exclusions(c["percent"]) == {"S1 vs S2: entries left out before comparing": 1,
                                        "S1 vs S2: entries with a mean in only one of S1 and S2": 2}


def test_a_mean_lists_only_the_rows_and_the_duplicates_of_the_entries_it_used(registry):
    """Review of round 1 (B): G4 (not shared, no valid value) holds NA, NA, "bad", "x"; the mean over G1-G3 listed raw-missing 2
    and invalid 2. With duplicates only in G4 its text said "9 valid rows averaged within 8 observation units" beside n=3."""
    empty_g4 = BASE[:3] + [("G4", "Check-1", ["NA", "NA", "bad", "x"])]
    c = by_study(compare(registry, save(registry, empty_g4), save(registry, OTHER, study="S2")))
    assert c["S1"].value == 16 and exclusions(c["S1"]) == {"S1: entries under min_independent_n=1": 1}
    one_missing = [("G1", "Alpha", ["10", "NA", "12"])] + BASE[1:]
    c = by_study(compare(registry, save(registry, one_missing, study="S3"), save(registry, OTHER, study="S4")))
    assert exclusions(c["S3"])["S3: raw-missing value token (entries used)"] == 1
    doubled_g4 = BASE[:3] + [("G4", "Check-1", ["30", "32", "34"])]
    c = by_study(compare(registry, save(registry, doubled_g4, study="S5", units={("G4", 1): "PS5G40"}), save(registry, OTHER, study="S6"),
                         duplicate_policy="mean_per_unit"))
    assert c["S5"].value == 16 and "averaged within" not in c["S5"].transformation
    assert "valid values used as recorded (one row per observation unit)" in c["S5"].transformation


@pytest.mark.parametrize("args", [{}, {"shared_entries_only": False}, {"exclude": ["G2"]}, {"exclude": ["G7"], "shared_entries_only": False},
                                  {"min_independent_n": 2}, {"exclude": ["G2"], "min_independent_n": 2},
                                  {"exclude": ["G4", "G7"], "min_independent_n": 2, "shared_entries_only": False}])
def test_the_excluded_entries_add_up_to_the_entries_not_used(registry, args):
    """Each claim's excluded entries, counted once each, are exactly the entries it did not use."""
    other = [("G1", "Alpha", ["5", "7"]), ("G2", "Beta", ["10"]), ("G3", "Gamma", ["9", "9"]), ("G7", "Omega", ["1", "3"])]
    r = compare(registry, save(registry, BASE), save(registry, other, study="S2"), **args)
    c = by_study(r)
    used = set(r.data["entries_used"])
    tables = {"S1": {"G1", "G2", "G3", "G4"}, "S2": {"G1", "G2", "G3", "G7"}}
    for study, here in tables.items():                              # a mean of entry means: n is the entries it used
        assert entry_exclusions(c[study]) == len(here) - c[study].denominator.n, (study, exclusions(c[study]))
    assert entry_exclusions(c["percent"]) == len(tables["S1"] | tables["S2"]) - len(used), exclusions(c["percent"])


def test_the_chat_names_the_study_on_every_excluded_line(registry):
    import chat_presenter as cp
    from contracts import AnalysisReport

    r = compare(registry, save(registry, BASE), save(registry, OTHER, study="S2"), percent="reduction", exclude=["G2"])
    claims = [Claim.model_validate(c) for c in r.data["claims"]]
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claims, caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=SimpleNamespace(registry=registry), notes=[])
    excluded = [line for line in cp.answer_summary(view).splitlines() if "Excluded:" in line]
    assert len(excluded) == 6 and all("S1" in line or "S2" in line for line in excluded), excluded


# (C) the percent to one decimal, never "-0.0"; a base under 1e-9 counts as zero.

def test_the_percent_is_shown_to_one_decimal_and_never_as_minus_zero(registry):
    import chat_presenter as cp
    from contracts import AnalysisReport

    base = save(registry, BASE)
    for other, percent, shown in ((save(registry, BASE, study="S2"), "reduction", "0.0"), (save(registry, OTHER, study="S3"), "change", "-45.8")):
        r = compare(registry, base, other, percent=percent)
        assert r.data["display"]["percent"] == shown
        claims = [Claim.model_validate(c) for c in r.data["claims"]]
        view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claims, caveats=[]),
                               retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=SimpleNamespace(registry=registry), notes=[])
        text = cp.answer_summary(view)
        assert f"{shown} %" in text and "-0.0 %" not in text and not re.search(r"\d[eE][+-]\d", text)


def test_a_base_mean_under_1e_9_counts_as_zero(registry):
    """Review of round 1 (C): base entry means 0.1, 0.2, -0.3 gave a base mean of about 1e-17 and "Percent change 1.08e+19 %"."""
    near_zero = [("G1", "Alpha", ["0.1", "0.1"]), ("G2", "Beta", ["0.2", "0.2"]), ("G3", "Gamma", ["-0.3", "-0.3"])]
    r = compare(registry, save(registry, near_zero), save(registry, OTHER, study="S2"))
    base_mean, p = by_study(r)["S1"].value, by_study(r)["percent"]
    assert abs(base_mean) < 1e-9 and p.value is None and "counts as zero" in p.missing_reason
    assert r.data["display"]["percent"] is None


# (D) the planner's words.

def test_the_planner_reads_lower_and_below_as_a_reduction_and_takes_the_base_after_than():
    import agents.coordinator as co

    prompt = " ".join(co.COORDINATOR_SYSTEM_PROMPT.split())
    assert 'The reference study is the control, the earlier harvest, or Y in "X higher/lower than Y".' in prompt
    assert '"lower" or "below": (base - other) / base x 100' in prompt and '"higher" or "above": (other - base) / base x 100' in prompt
    assert "Both inputs are required." in prompt


# (E) one test per surviving bug of the round-1 review (G1 and G2, the guard reading lists, are tests/test_analyst.py's
# test_every_table_argument_of_every_tool_is_checked_before_the_tool_runs and test_a_table_in_any_argument_place_or_list_place_is_checked).

def test_h2_an_incomplete_comparison_makes_the_analysis_incomplete(registry):
    import agents.analyst as an
    from contracts import AgentResult

    base, partial = save(registry, BASE), save(registry, OTHER, study="S2", complete=False)
    args = {"base_artifact_id": base, "other_artifact_id": partial, "base_study_db_id": "S1", "percent": "change", "allow_incomplete": True}
    result = at.dispatch_analyst(registry, "compare_means", args)
    assert result.ok and result.complete is False
    tools = an.GuardedAnalystTools(at.AnalystTools(registry), [base, partial])
    tools.records.append(an.ToolRecord("compare_means", args, result))
    agent = AgentResult(status="completed", payload={"methods": [], "caveats": [], "narrative": "", "key_claim_ids": []},
                        model_requested="m", model_calls=1, tool_calls=1, elapsed_seconds=0.0)
    report, _ = an.build_report(agent, tools, an.AnalysisInput(question="q", artifact_ids=[base, partial]))
    assert report.status == "incomplete"


def test_i1_comparisons_by_another_column_have_their_own_claims(registry):
    base, other = save(registry, BASE), save(registry, OTHER, study="S2")
    by_entry = {c["claim_id"] for c in compare(registry, base, other, shared_entries_only=False).data["claims"]}
    by_plot = {c["claim_id"] for c in compare(registry, base, other, shared_entries_only=False, by="observationUnitDbId").data["claims"]}
    assert by_entry and by_plot and not by_entry & by_plot


def test_c1_c2_an_incomplete_base_table_marks_the_result_incomplete(registry):
    partial, other = save(registry, BASE, complete=False), save(registry, OTHER, study="S2")
    r = compare(registry, partial, other, allow_incomplete=True)
    assert r.ok and r.complete is False and not registry.manifest(r.data["result_artifact_id"]).complete
    assert all("INCOMPLETE" in claim.transformation for claim in by_study(r).values())


def test_v1_plot_values_averaged_per_unit_when_asked(registry):
    """G1 has two rows for plot PS1G10 (10, 12) and one for PS1G12 (14): per unit 11 and 14; G2 20, 22: (11+14+20+22)/4 = 16.75,
    not (10+12+14+20+22)/5 = 15.6."""
    duplicated = save(registry, [("G1", "Alpha", ["10", "12", "14"]), ("G2", "Beta", ["20", "22"])], units={("G1", 1): "PS1G10"})
    other = save(registry, [("G1", "Alpha", ["5", "7"]), ("G2", "Beta", ["10", "12"])], study="S2")
    c = by_study(compare(registry, duplicated, other, mean_of="values", duplicate_policy="mean_per_unit"))
    assert c["S1"].value == 16.75 and c["S1"].denominator.n == 4 and "observation units" in c["S1"].denominator.name


def test_v2_plot_values_of_every_entry_when_asked(registry):
    """Every entry of the base table: (10+12+20+22+15+17+30+32)/8 = 19.75, not the shared entries' 16."""
    c = by_study(compare(registry, save(registry, BASE), save(registry, OTHER, study="S2"), mean_of="values", shared_entries_only=False))
    assert c["S1"].value == 19.75 and c["S1"].denominator.n == 8


def test_e1_the_percent_lists_the_entries_under_min_independent_n(registry):
    other = [("G1", "Alpha", ["5", "7"]), ("G2", "Beta", ["10", "12"]), ("G3", "Gamma", ["9"])]
    p = by_study(compare(registry, save(registry, BASE), save(registry, other, study="S2"), min_independent_n=2))["percent"]
    assert sum(e.count for e in p.exclusions if "min_independent_n=2" in e.reason) == 1


def test_e2_a_mean_lists_the_rows_with_no_entry_id(registry):
    with_blank = BASE + [("", "", ["13"])]
    c = by_study(compare(registry, save(registry, with_blank), save(registry, OTHER, study="S2")))
    assert c["S1"].value == 16 and sum(e.count for e in c["S1"].exclusions if "empty germplasmDbId" in e.reason) == 1


def test_l1_the_saved_table_says_which_entries_were_compared(registry):
    r = compare(registry, save(registry, BASE), save(registry, OTHER, study="S2"), shared_entries_only=False)
    notes = registry.manifest(r.data["result_artifact_id"]).quality_notes
    assert any("shared_entries_only=False" in n for n in notes) and not any("shared_entries_only=True" in n for n in notes)


def test_w1_the_timing_warnings_are_passed_on(registry):
    r = compare(registry, save(registry, BASE), save(registry, OTHER, study="S2"))
    assert any("Measurement dates are not recorded for 8 of 8 valid source rows" in w for w in r.warnings)
    assert sum("Growth stage or timepoint is not stated" in w for w in r.warnings) == 2


def test_b2_an_other_table_that_also_holds_the_base_study_is_refused(registry):
    mixed = save(registry, OTHER, study="S2", studies=["S2", "S1"])
    refused = compare(registry, save(registry, BASE), mixed)
    assert not refused.ok and "also a table of the base study" in refused.error.message


def test_r2_the_planners_example_is_a_reduction():
    import agents.coordinator as co

    prompt = " ".join(co.COORDINATOR_SYSTEM_PROMPT.split())
    assert ('compare_means(base_study_db_id="<the reference study>", percent="reduction", shared_entries_only=false, or '
            'mean_of="values")') in prompt


def test_z1_an_empty_base_study_is_refused(registry):
    refused = compare(registry, save(registry, BASE), save(registry, OTHER, study="S2"), base_study_db_id="")
    assert not refused.ok and refused.error.code == "invalid_argument"


# --- round 4 (DECISION (Bidhan) 2026-10-22 12:40 UTC, item 8: points 1-3 of the round-3 stop) ---------------------

def test_the_zero_rule_is_relative_to_the_values_the_base_mean_averaged(registry):
    """Review of round 3: base values near 1e8 that cancel to an exact 0 leave a float residue of 9.9e-9, above the absolute 1e-9,
    so a percent of 1e10 was stated. A base mean under 1e-9 of the largest absolute plot value behind it counts as zero too."""
    big = [("G1", "A", ["-100000000.1"]), ("G2", "B", ["-200000000.2"]), ("G3", "C", ["300000000.3"])]
    other = [("G1", "A", ["1"]), ("G2", "B", ["2"]), ("G3", "C", ["3"])]
    r = compare(registry, save(registry, big), save(registry, other, study="S2"))
    base_mean, p = by_study(r)["S1"].value, by_study(r)["percent"]
    assert 1e-9 < abs(base_mean) < 1e-7 and p.value is None and "counts as zero" in p.missing_reason and "largest absolute plot value" in p.missing_reason
    small = [("G1", "A", ["0.00000001"]), ("G2", "B", ["0.00000001"]), ("G3", "C", ["0.00000001"])]     # 1e-8 made of values of 1e-8: a mean
    assert by_study(compare(registry, save(registry, small), save(registry, other, study="S2")))["percent"].value is not None
    seven = [("G1", "A", ["1000"]), ("G2", "B", ["-1000"]), ("G3", "C", ["0.0003"])]                   # 1e-4 of values up to 1000: 1e-7 relative, a mean
    assert by_study(compare(registry, save(registry, seven), save(registry, other, study="S2")))["percent"].value is not None
    one_small = [("G1", "A", ["100000000"]), ("G2", "B", ["-100000000"]), ("G3", "C", ["0.000001"])]  # 3.3e-7 of values up to 1e8: zero
    assert by_study(compare(registry, save(registry, one_small), save(registry, other, study="S2")))["percent"].value is None


def test_the_1e_9_threshold_is_pinned(registry):
    for value, defined in (("0.0000000005", False), ("0.000000002", True)):                    # 5e-10 counts as zero; 2e-9 does not
        base = [("G1", "A", [value]), ("G2", "B", [value])]
        r = compare(registry, save(registry, base), save(registry, [("G1", "A", ["1"]), ("G2", "B", ["1"])], study="S2"))
        assert (by_study(r)["percent"].value is not None) == defined


def test_a_tiny_negative_percent_is_shown_as_zero_but_kept_in_the_claim(registry):
    base, other = [("G1", "A", ["100"]), ("G2", "B", ["100"])], [("G1", "A", ["100.04"]), ("G2", "B", ["100.04"])]
    r = compare(registry, save(registry, base), save(registry, other, study="S2"), percent="reduction")   # (100 - 100.04) / 100 = -0.04 %
    assert by_study(r)["percent"].value == pytest.approx(-0.04) and r.data["display"]["percent"] == "0.0"
    assert at._one_decimal(-0.04) == "0.0" and at._one_decimal(-0.06) == "-0.1" and at._one_decimal(0.0) == "0.0"


def test_the_percents_n_is_the_entries_used_not_the_union(registry):
    r = compare(registry, save(registry, BASE), save(registry, OTHER, study="S2"))              # base G1-G4, other G1-G3 + G7: 3 shared of 5
    assert by_study(r)["percent"].denominator.n == 3


def test_an_int_base_study_in_a_plan_step_is_a_problem_not_a_crash():
    _plan_, problems, _ = _plan({"base_study_db_id": 4577, "percent": "reduction"})
    assert any("needs base_study_db_id" in p for p in problems)


def test_missing_and_invalid_tokens_are_counted_apart(registry):
    base = [("G1", "A", ["10", "NA", ""]), ("G2", "B", ["20", "abc"]), ("G3", "C", ["15"])]        # 2 missing, 1 invalid among the entries used
    ex = exclusions(by_study(compare(registry, save(registry, base), save(registry, OTHER, study="S2")))["S1"])
    assert ex["S1: raw-missing value token (entries used)"] == 2 and ex["S1: invalid or non-finite value token (entries used)"] == 1


def test_after_n_left_out_counts_only_the_entries_of_that_table(registry):
    r = compare(registry, save(registry, BASE), save(registry, OTHER, study="S2"), exclude=["Check-1"])    # G4 is only in the base table
    assert "after 1 left out" in by_study(r)["S1"].transformation and "left out" not in by_study(r)["S2"].transformation


def test_every_planned_comparison_binds_not_only_the_first(registry):
    import agents.analyst as an
    import agents.coordinator as co

    pairs = [("S1", "reduction"), ("S2", "change")]
    steps = [co.DraftStep(step_id=f"step_{i}", agent="analyst", action="compare_means", inputs={"base_study_db_id": b, "percent": p},
                          depends_on=[]) for i, (b, p) in enumerate(pairs)]
    assert [(c.base_study_db_id, c.percent) for c in an.planned_comparisons(steps)] == pairs
    base, other = save(registry, BASE), save(registry, OTHER, study="S2")
    guard = an.GuardedAnalystTools(at.AnalystTools(registry), [base, other], inspected={base, other}, planned_comparisons=pairs)
    call = {"base_artifact_id": other, "other_artifact_id": base, "base_study_db_id": "S2"}
    assert anyio.run(guard.call, "compare_means", {**call, "percent": "change"}).ok                   # the second planned pair
    assert not anyio.run(guard.call, "compare_means", {**call, "percent": "reduction"}).ok              # not a planned pair


def test_the_full_report_shows_the_percent_to_one_decimal_and_no_unit_after_not_available(registry):
    import run

    def report(result):
        return "\n".join(run.readable_answer([Claim.model_validate(c) for c in result.data["claims"]], registry))

    text = report(compare(registry, save(registry, BASE), save(registry, OTHER, study="S3")))        # change -45.8333 %
    assert "**-45.8 %**" in text and "-45.8333" not in text
    near_zero = [("G1", "A", ["0.1", "0.1"]), ("G2", "B", ["0.2", "0.2"]), ("G3", "C", ["-0.3", "-0.3"])]
    text = report(compare(registry, save(registry, near_zero), save(registry, OTHER, study="S2")))
    assert "not available (" in text and ") %**" not in text                                        # no stray unit after the reason
    tiny = compare(registry, save(registry, [("G1", "A", ["100"]), ("G2", "B", ["100"])]),
                   save(registry, [("G1", "A", ["100.04"]), ("G2", "B", ["100.04"])], study="S2"), percent="reduction")
    assert "-0.0" not in report(tiny) and "**0.0 %**" in report(tiny)


def test_two_comparisons_with_the_same_counts_get_their_own_tables_and_labels(registry):
    import chat_presenter as cp
    from contracts import AnalysisReport

    base, other = save(registry, BASE), save(registry, OTHER, study="S2")
    one = compare(registry, base, other, percent="reduction", exclude=["Alpha"])                  # G2, G3 used: 44.4 %? no: 2 entries each
    two = compare(registry, base, other, percent="reduction", exclude=["Beta"])                   # G1, G3 used
    claims = [Claim.model_validate(c) for r in (one, two) for c in r.data["claims"]]
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claims, caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=SimpleNamespace(registry=registry), notes=[])
    text = cp.answer_summary(view)
    assert "Percent reduction \\(left out: G1\\)" in text and "Percent reduction \\(left out: G2\\)" in text   # the chat escapes ( )
    assert text.count("| Statistic | Result |") == 6                                                  # three claims per call, never merged


def test_a_share_that_reaches_the_numeric_block_is_labelled_share_not_percent_change():
    import chat_presenter as cp
    from contracts import Denominator

    share = Claim(claim_id="clm_v2_000000_0001_thr-0000000000_ratio_studyDbId_S1", kind="ratio", value=100 / 3, unit="%",
                  denominator=Denominator(name="entries with a valid value in study S1", n=3), source_artifact_hashes=["a" * 64],
                  filters={"studyDbId": "S1"}, grouping=["studyDbId"],
                  transformation="share of the entries with a valid value, in percent; count of entries above the threshold 11 t/ha; threshold = x; n")
    assert any("| Share |" in line for line in cp._numeric_blocks([share], cp._Sources(None), {}))


# --- round 5 (Bidhan 2026-10-23, item 12): the round-4 review's findings and gaps ------------------------------------------

def _full_report(registry, result) -> str:
    import run
    return "\n".join(run.readable_answer([Claim.model_validate(c) for c in result.data["claims"]], registry))


def test_r5_the_full_report_says_and_n_more_only_for_the_groups_it_did_not_list(registry):
    """Round 4 review, Q2 (a wrong number on real data, in round 4's own code): the grouped list reused the name of the shown
    claims for one line's text, so 12 groups all listed read '... and 8 more' and 31 groups '... and 24 more' (not 'and 1 more')."""
    for n, more in ((12, None), (31, "and 1 more;")):
        table = save(registry, [(f"G{i}", f"name {i}", ["1", "2"]) for i in range(n)], study=f"S{n}")
        result = at.dispatch_analyst(registry, "group_stats", {"artifact_id": table, "by": "germplasmDbId", "top_n": n})
        assert result.ok, result.error
        text = _full_report(registry, result)
        assert ("more; all of them are in the numbers table" in text) == (more is not None) and (more is None or more in text), n


def test_r5_values_that_cancel_inside_one_entry_count_as_zero_for_entry_means_too(registry):
    """Round 4 review, Finding A (made-up input): with mean_of=entry_means the zero rule's scale was the largest ENTRY MEAN, so
    values cancelling inside one entry (decimal sum exactly 0) left a residue of 2.5e-9 that got a percent of 40265318300 %.
    The scale is now the largest plot value behind the mean in both modes (T4's rule)."""
    base = save(registry, [("G1", "Alpha", ["-86011353.6", "-28734791.6", "114746145.2"])])
    other = save(registry, [("G1", "Alpha", ["1"])], study="S2")
    for mean_of in ("entry_means", "values"):
        k = by_study(compare(registry, base, other, mean_of=mean_of))["percent"]
        assert k.value is None and "counts as zero" in k.missing_reason, mean_of


def test_r5_the_zero_scale_is_the_plot_values_of_the_entries_used_only(registry):
    """Gap M2: an entry left out is not in the scale - its 1e9 must not make the base's genuine 1e-6 a zero."""
    base = save(registry, [("G1", "Alpha", ["0.000001"]), ("G2", "Beta", ["1000000000"])])
    other = save(registry, [("G1", "Alpha", ["1"]), ("G2", "Beta", ["1"])], study="S2")
    k = by_study(compare(registry, base, other, exclude=["G2"]))["percent"]
    assert k.value is not None and k.missing_reason is None


def test_r5_the_zero_rule_edges(registry):
    """Gaps M4 and M5: 1e-9 itself is not under 1e-9; a mean of 2.5e-8 beside plot values of 10 is 2.5e-9 of the scale, above
    the relative 1e-9, so it is a mean with a percent."""
    other = save(registry, [("G1", "Alpha", ["1"])], study="S2")
    exactly = by_study(compare(registry, save(registry, [("G1", "Alpha", ["0.000000001"])]), other))["percent"]
    assert exactly.value is not None and exactly.missing_reason is None
    relative = by_study(compare(registry, save(registry, [("G1", "Alpha", ["10", "-9.99999995"])], study="S3"), other))["percent"]
    assert relative.value is not None and relative.missing_reason is None


def test_r5_the_scale_is_the_largest_absolute_plot_value_a_negative_one_too(registry):
    """Gap M27: the scale is the largest ABSOLUTE plot value, here -4e9; a base mean of 2 is under 1e-9 of it and counts as zero
    (with the largest signed value, 1e9 + 10, it would not). Made-up input: real tables hold no negative value (round-4 review)."""
    base = save(registry, [("G1", "Alpha", ["-4000000000", "1000000000", "1000000000", "1000000000", "1000000010"])])
    k = by_study(compare(registry, base, save(registry, [("G1", "Alpha", ["1"])], study="S2")))["percent"]
    assert by_study(compare(registry, base, save(registry, [("G1", "Alpha", ["1"])], study="S3")))["S1"].value == 2.0
    assert k.value is None and "counts as zero" in k.missing_reason


def test_r5_without_shared_entries_n_is_the_base_tables_entries(registry):
    """Gap M22: with shared_entries_only=false the percent's n is the base table's entries (4), not every entry of both (5)."""
    k = by_study(compare(registry, save(registry, BASE), save(registry, OTHER, study="S2"), shared_entries_only=False))["percent"]
    assert (k.denominator.name, k.denominator.n) == ("entries of the base table", 4)


def test_r5_a_share_in_the_full_report_never_reads_a_partial_count_as_0_or_100():
    """Gaps M11 and M12: no test read a share in the Full report. One decimal, more only when one would read 0 or 100."""
    import run

    def share(value):
        return run._percent_text(SimpleNamespace(value=value, transformation="share of the entries with a valid value, in percent"))

    assert [share(v) for v in (100 / 3, 100 / 2001, 200000 / 2001, 0.0, 100.0)] == ["33.3", "0.05", "99.95", "0.0", "100.0"]


def test_r5_every_entry_left_out_is_named_in_the_label(registry):
    """Gaps M7 and M9: two entries left out are both named, after the comma too."""
    import chat_presenter as cp
    from contracts import AnalysisReport

    result = compare(registry, save(registry, BASE), save(registry, OTHER, study="S2"), percent="reduction", exclude=["Alpha", "Beta"])
    view = SimpleNamespace(execution_status="completed",
                           analysis=AnalysisReport(status="completed", claims=[Claim.model_validate(c) for c in result.data["claims"]], caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=SimpleNamespace(registry=registry), notes=[])
    assert "Percent reduction \\(left out: G1, G2\\)" in cp.answer_summary(view)


def test_r5_a_zero_percent_keeps_its_unit_and_not_available_has_none(registry):
    """Gaps M10 and M25: a percent of exactly 0 is '0.0 %'; a grouped value that is not available has no unit after its reason."""
    equal = compare(registry, save(registry, [("G1", "Alpha", ["10"])]), save(registry, [("G1", "Alpha", ["10"])], study="S2"))
    assert "**0.0 %**" in _full_report(registry, equal)
    import run
    from contracts import Denominator

    grouped = [Claim(claim_id=f"clm_000000_0001_mean_{g}", kind="mean", value=value, unit="t/ha", missing_reason=reason,
                     denominator=Denominator(name="valid numeric values", n=n), source_artifact_hashes=["a" * 64],
                     filters={"germplasmDbId": g}, grouping=["germplasmDbId"], transformation="group mean of valid values")
               for g, value, reason, n in (("G1", 11.0, None, 2), ("G2", None, "no valid numeric values", 0))]
    assert "  - G2: not available (no valid numeric values) (n = 0)" in run.readable_answer(grouped, registry)


def test_r5_other_tools_percents_keep_their_digits(registry):
    """Round 5 review, SHOULD-FIX 1 and 2: the one-decimal rule is compare_means' (and a share's in the Full report); another
    tool's percent (T6's heterosis, 0.04321 %) read "0.0 %" there and in the chat's number block. It keeps dev's digits again."""
    import run
    import chat_presenter as cp
    from contracts import Denominator

    def claim(value, transformation):
        return Claim(claim_id="clm_v2_000000_0001_het-0000000000_ratio_mph", kind="ratio", value=value, unit="%",
                     denominator=Denominator(name="sites", n=2), source_artifact_hashes=["a" * 64], filters={}, grouping=[],
                     transformation=transformation)

    other = claim(0.04321, "mid-parent heterosis of group means, in percent")
    assert "**0.0432 %**" in "\n".join(run.readable_answer([other], registry))
    assert any("0.04321 %" in line for line in cp._numeric_blocks([other], cp._Sources(None), {}))
    third = claim(100 / 3, "share of the entries with a valid value, in percent")
    assert "**33.3 %**" in "\n".join(run.readable_answer([third], registry))   # the share's own rule (plain: 33.3333)
    share = claim(100 / 2001, "share of the entries with a valid value, in percent")
    assert any("0.0499750124938 %" in line for line in cp._numeric_blocks([share], cp._Sources(None), {}))
