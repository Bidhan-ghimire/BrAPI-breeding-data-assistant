"""
tests/test_compare_trials_paired.py - T4 "compare_trials_paired" (coach 2026-10-23 01:51 UTC, NEXT 4; research build order T4):
ONE trait in two studies (trials), on the entries with a mean in both: the two trial means (the mean of the entry means), the
mean difference and the percent change (other - base) or reduction (base - other) of the base mean, the paired t on the
entry-mean differences (n - 1 df) with its p, Welch's t with the Welch-Satterthwaite df as the alternative that ignores the
pairing, and Pearson r of the entry means between the trials. Descriptive numbers, never a decision.

Fixtures: SYNTHETIC tables checked against an independent numpy computation (numpy only in this test); the p values against
pvalues on the same t and df. The metan data_ge check (E3 against E4, research notes Q4) runs outside the repo (item (16)).
No network, no model.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

import agents.analyst as an
import agents.coordinator as co
import analyst_tools as at
import brapi_mcp_server as srv
import chat_presenter as cp
import pvalues
from artifacts import ArtifactRegistry
from contracts import AnalysisReport, Claim, MeasurementMeta, Statistic
from part2_support import SYNTHETIC_BASE, synthetic_context

ENTRIES = [f"E{i}" for i in range(8)]


def trial(shift: float = 0.0, *, reps: int = 2, entries=ENTRIES, seed: float = 0.0) -> dict[str, list[float]]:
    """entry -> plot values: an entry effect, a shift between the trials and an irregular plot error (no randomness)."""
    return {e: [10.0 + 2.0 * math.cos(1.1 * i + seed) + shift + 0.4 * math.sin(2.3 * i + 1.7 * k + 3.1 * seed) for k in range(reps)]
            for i, e in enumerate(entries)}


@pytest.fixture
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "out", "run_pair", base_url=SYNTHETIC_BASE)


def rows_of(study: str, values: dict[str, list[float]], *, names=None, variable: str = "V1") -> list[dict[str, str]]:
    """Measurement rows of one study, one plot per value (rep k + 1 for the k-th)."""
    rows = []
    for entry, plot_values in values.items():
        for k, value in enumerate(plot_values, 1):
            unit = f"{study}-{entry}-{k}"
            rows.append({"observationDbId": f"O{unit}", "observationUnitDbId": unit, "observationUnitName": f"plot {unit}",
                         "germplasmDbId": entry, "germplasmName": (names or {}).get(entry, f"clone {entry}"), "studyDbId": study,
                         "observationVariableDbId": variable, "observationVariableName": "SYNTHETIC yield", "observationTimeStamp": "",
                         "value": repr(float(value))})
    return rows


def save(registry: ArtifactRegistry, rows, *, complete=True) -> str:
    columns = {col: srv.OBSERVATION_COLUMNS.get(col, "text") for col in rows[0]}
    return registry.save_table(rows, kind="observations", columns=columns, complete=complete, study_ids=sorted({r["studyDbId"] for r in rows}),
                               variable_ids=sorted({r["observationVariableDbId"] for r in rows}),
                               measurement=MeasurementMeta(trait="SYNTHETIC yield", unit="t/ha"), quality_notes=["SYNTHETIC"]).artifact_id


def pair(registry, base, other, **args):
    return at.dispatch_analyst(registry, "compare_trials_paired", {"base_artifact_id": base, "other_artifact_id": other, **args})


def claims(r) -> list[Claim]:
    return [Claim.model_validate(k) for k in r.data["claims"]]


def named(r) -> dict[str, Claim]:
    return {k.transformation.split(" | ")[0]: k for k in claims(r)}


def per_entry(registry, r) -> dict[str, dict[str, str]]:
    return {row["entry"]: row for row in at._load(registry, r.data["result_artifact_id"]).rows}


def expected(a, b, *, reduction=False) -> dict[str, float]:
    """The comparison by an independent numpy computation, on the entries in both trials."""
    np = pytest.importorskip("numpy")
    paired = sorted(set(a) & set(b))
    x = np.array([np.mean(a[e]) for e in paired])
    y = np.array([np.mean(b[e]) for e in paired])
    d = x - y if reduction else y - x
    g = len(paired)
    va, vb = x.var(ddof=1) / g, y.var(ddof=1) / g
    return {"g": g, "mean_base": float(x.mean()), "mean_other": float(y.mean()), "dbar": float(d.mean()), "sd": float(d.std(ddof=1)),
            "se": float(d.std(ddof=1) / np.sqrt(g)), "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(g))), "df": g - 1,
            "welch_t": float(d.mean() / np.sqrt(va + vb)), "welch_df": float((va + vb) ** 2 / ((va ** 2 + vb ** 2) / (g - 1))),
            "r": _r(np, x, y), "percent": float(100 * d.mean() / x.mean())}


def _r(np, x, y) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):          # a constant side gives nan, as the tool gives no r
        return float(np.corrcoef(x, y)[0, 1])


def two(registry, a, b, **args):
    return pair(registry, save(registry, rows_of("S1", a)), save(registry, rows_of("S2", b)), **args)


# --- 1. the numbers against numpy ------------------------------------------------------------------------------------

def test_the_change_matches_an_independent_numpy_computation(registry):
    a, b = trial(), trial(1.5, seed=0.4)
    r = two(registry, a, b)
    assert r.ok, r.error
    x, d = expected(a, b), r.data
    assert d["n_pairs"] == 8 and d["percent_rule"] == "change" and (d["base_study"], d["other_study"]) == ("S1", "S2")
    for key in ("mean_base", "mean_other"):
        assert d[key] == pytest.approx(x[key], rel=1e-12)
    assert d["mean_difference"] == pytest.approx(x["dbar"], rel=1e-12) and d["percent"] == pytest.approx(x["percent"], rel=1e-12)
    assert d["sd_difference"] == pytest.approx(x["sd"], rel=1e-12) and d["se_difference"] == pytest.approx(x["se"], rel=1e-12)
    assert d["paired"]["t"] == pytest.approx(x["t"], rel=1e-12) and d["paired"]["df"] == 7
    assert d["paired"]["p"] == pytest.approx(pvalues.t_two_sided_p(x["t"], 7), rel=1e-12)
    assert d["welch"]["t"] == pytest.approx(x["welch_t"], rel=1e-12) and d["welch"]["df"] == pytest.approx(x["welch_df"], rel=1e-12)
    assert d["welch"]["p"] == pytest.approx(pvalues.t_two_sided_p(x["welch_t"], x["welch_df"]), rel=1e-12)
    assert d["pearson_r"] == pytest.approx(x["r"], rel=1e-12)


def test_every_number_is_a_claim_with_its_formula_headline_first(registry):
    a, b = trial(), trial(1.5, seed=0.4)
    r = two(registry, a, b)
    k, x = named(r), expected(a, b)
    base = k["mean of the base trial's entry means"]
    assert base.kind == "mean" and base.unit == "t/ha" and base.value == pytest.approx(x["mean_base"], rel=1e-12)
    percent = k["percent change of the mean"]
    assert percent.kind == "ratio" and percent.unit == "%" and "100 x (mean other - mean base) / mean base" in percent.transformation
    assert k["mean difference other - base"].value == pytest.approx(x["dbar"], rel=1e-12)
    assert k["paired t"].kind == "test" and k["paired t: p"].kind == "test" and k["paired t: df"].value == 7
    assert "the entries as the sampling units" in k["paired t"].transformation                 # what the paired t leaves out
    assert k["Welch t"].kind == "test" and "ignores the pairing" in k["Welch t"].transformation
    assert k["Pearson r of the entry means between the trials"].kind == "correlation"
    assert k["SE of the mean difference"].kind == "precision" and k["SD of the differences"].kind == "sd"
    assert k["entries in both trials"].kind == "count" and k["entries in both trials"].value == 8
    assert all(c.denominator.n == 8 for c in claims(r)) and all(at.PAIR_NOTE in c.transformation for c in claims(r))
    assert [c.transformation.split(" | ")[0] for c in claims(r)][:5] == [
        "mean of the base trial's entry means", "mean of the other trial's entry means", "percent change of the mean",
        "mean difference other - base", "paired t"]


def test_reduction_turns_the_differences_around(registry):
    a, b = trial(3.0), trial(0.0, seed=0.4)
    base, other = save(registry, rows_of("S1", a)), save(registry, rows_of("S2", b))
    change, reduction = pair(registry, base, other), pair(registry, base, other, percent="reduction")
    x, d = expected(a, b, reduction=True), reduction.data
    assert d["percent_rule"] == "reduction" and d["mean_difference"] > 0 and d["mean_difference"] == pytest.approx(x["dbar"], rel=1e-12)
    assert d["percent"] == pytest.approx(x["percent"], rel=1e-12) and d["percent"] == pytest.approx(-change.data["percent"], rel=1e-12)
    assert d["paired"]["t"] == pytest.approx(x["t"], rel=1e-12) and d["paired"]["t"] == pytest.approx(-change.data["paired"]["t"], rel=1e-12)
    assert d["paired"]["p"] == pytest.approx(change.data["paired"]["p"], rel=1e-12)
    assert d["welch"]["t"] == pytest.approx(-change.data["welch"]["t"], rel=1e-12)
    assert d["pearson_r"] == pytest.approx(change.data["pearson_r"], rel=1e-12)
    k = named(reduction)
    assert "100 x (mean base - mean other) / mean base" in k["percent reduction of the mean"].transformation
    assert "mean difference base - other" in k
    assert {c.claim_id for c in claims(change)}.isdisjoint({c.claim_id for c in claims(reduction)})


@pytest.mark.parametrize("scale", [1e180, 1e-160])
def test_huge_and_tiny_values_give_the_same_t_df_r_and_percent(registry, scale):
    """Squares of 1e180 overflow and of 1e-160 underflow: every sum of squares runs on values divided by their scale."""
    a, b = trial(), trial(1.5, seed=0.4)
    plain = two(registry, a, b)
    scaled = pair(registry, save(registry, rows_of("S3", {e: [v * scale for v in vs] for e, vs in a.items()})),
                  save(registry, rows_of("S4", {e: [v * scale for v in vs] for e, vs in b.items()})))
    assert scaled.ok, scaled.error
    for key in ("percent", "pearson_r"):
        assert scaled.data[key] == pytest.approx(plain.data[key], rel=1e-9)
    for test in ("paired", "welch"):
        assert scaled.data[test]["t"] == pytest.approx(plain.data[test]["t"], rel=1e-9)
    assert scaled.data["welch"]["df"] == pytest.approx(plain.data["welch"]["df"], rel=1e-9)
    assert scaled.data["mean_difference"] == pytest.approx(plain.data["mean_difference"] * scale, rel=1e-9)
    assert scaled.data["sd_difference"] == pytest.approx(plain.data["sd_difference"] * scale, rel=1e-9)


# --- 2. which entries are paired -------------------------------------------------------------------------------------

def test_only_entries_in_both_trials_are_paired_and_the_rest_are_counted(registry):
    a = trial(entries=ENTRIES[:7])                                  # E7 only in the other trial
    b = trial(1.0, entries=ENTRIES[1:], seed=0.4)                   # E0 only in the base trial
    r = two(registry, a, b)
    x = expected(a, b)
    assert r.data["n_pairs"] == 6 and x["g"] == 6 and r.data["paired"]["t"] == pytest.approx(x["t"], rel=1e-12)
    assert r.data["unpaired"] == {"only in base": 1, "only in other": 1}
    reasons = {e.reason: e.count for e in claims(r)[0].exclusions}
    assert reasons["entries only in the base trial"] == 1 and reasons["entries only in the other trial"] == 1
    rows = per_entry(registry, r)
    assert rows["E0"]["paired"] == "false" and rows["E0"]["left_out"] == "only in base" and rows["E0"]["other_mean"] == ""
    a3, b3 = sum(a["E3"]) / 2, sum(b["E3"]) / 2
    assert rows["E3"]["paired"] == "true" and float(rows["E3"]["difference"]) == pytest.approx(b3 - a3, rel=1e-12)
    assert float(rows["E3"]["percent"]) == pytest.approx(100 * (b3 - a3) / a3, rel=1e-12)


def test_the_per_entry_table_follows_the_direction(registry):
    """Deliberate bug M24 survived: the saved per-entry difference stayed other - base under percent='reduction'."""
    a, b = trial(3.0), trial(0.0, seed=0.4)
    rows = per_entry(registry, two(registry, a, b, percent="reduction"))
    a3, b3 = sum(a["E3"]) / 2, sum(b["E3"]) / 2
    assert float(rows["E3"]["difference"]) == pytest.approx(a3 - b3, rel=1e-12) and a3 > b3
    assert float(rows["E3"]["percent"]) == pytest.approx(100 * (a3 - b3) / a3, rel=1e-12)


def test_exclude_and_name_prefix_leave_entries_out_by_name(registry):
    names = {e: (f"PH17.{e}" if i < 5 else f"check {e}") for i, e in enumerate(ENTRIES)}
    a, b = trial(), trial(1.0, seed=0.4)
    r = pair(registry, save(registry, rows_of("S1", a, names=names)), save(registry, rows_of("S2", b, names=names)),
             exclude=["check E5"], name_prefix="PH17.")
    assert r.ok, r.error
    x = expected({e: a[e] for e in ENTRIES[:5]}, {e: b[e] for e in ENTRIES[:5]})
    assert r.data["n_pairs"] == 5 and r.data["paired"]["t"] == pytest.approx(x["t"], rel=1e-12)
    assert r.data["left_out"] == {"excluded": 1, "outside the name prefix": 2} and r.data["excluded_named"] == ["E5"]


def test_fewer_than_three_shared_entries_are_refused(registry):
    r = two(registry, trial(entries=ENTRIES[:4]), trial(1.0, entries=ENTRIES[2:], seed=0.4))     # E2 and E3 in both
    assert not r.ok and r.error.code == "unsupported" and "2 entries" in r.error.message and "at least 3" in r.error.message


# --- 3. numbers that are not defined -------------------------------------------------------------------------------

def test_entry_means_of_both_signs_that_sum_to_0_give_no_percent(registry):
    """compare_means round 4, review Q1: a base mean of exactly 0 that float arithmetic left as 2e-9 gave 40265318300 %. Since round 2
    E2's -0.3 withholds the percent first (no ratio scale); the zero route is tested with cancelling plots below (review 2, F4)."""
    a = {"E0": [0.1], "E1": [0.2], "E2": [-0.3]}                    # exactly 0 on paper; the float sum is 2.8e-17
    b = {"E0": [0.5], "E1": [0.1], "E2": [0.9]}
    r = two(registry, a, b)
    assert r.ok, r.error
    assert r.data["percent"] is None and "negative" in r.data["percent_missing_reason"]       # E2's -0.3 (round 2: not a ratio scale)
    percent = named(r)["percent change of the mean"]
    assert percent.value is None and percent.missing_reason
    assert r.data["mean_difference"] == pytest.approx(0.5, rel=1e-12)
    rows = per_entry(registry, r)
    assert all(row["percent"] == "" and "negative" in row["percent_note"] for row in rows.values())     # the rows follow the headline


def test_a_negative_base_mean_gives_no_percent(registry):
    a, b = trial(-15.0), trial(-14.0, seed=0.4)                     # means below 0 (a deviation, a log ratio)
    r = two(registry, a, b)
    assert r.data["mean_base"] < 0 and r.data["percent"] is None and "negative" in r.data["percent_missing_reason"]
    assert r.data["paired"]["t"] == pytest.approx(expected(a, b)["t"], rel=1e-12)


def test_the_same_change_in_every_entry_leaves_t_undefined_and_says_why(registry):
    a = {e: [0.1 * i, 0.1 * i + 0.2] for i, e in enumerate(ENTRIES)}
    b = {e: [v + 0.3 for v in a[e]] for e in ENTRIES}              # every entry + 0.3; the differences carry float residue
    r = two(registry, a, b)
    assert r.ok, r.error
    assert r.data["mean_difference"] == pytest.approx(0.3, rel=1e-9)
    assert r.data["paired"]["t"] is None and r.data["paired"]["p"] is None and "same amount" in r.data["paired"]["missing_reason"]
    k = named(r)
    assert k["paired t"].value is None and k["paired t"].missing_reason and k["paired t: p"].value is None
    assert r.data["welch"]["t"] == pytest.approx(expected(a, b)["welch_t"], rel=1e-9)


def test_r_is_not_stated_when_one_trial_has_equal_entry_means(registry):
    a, b = trial(), {e: [7.0, 7.0] for e in ENTRIES}
    r = two(registry, a, b)
    assert r.ok, r.error
    assert r.data["pearson_r"] is None and "equal" in r.data["pearson_missing_reason"]
    assert named(r)["Pearson r of the entry means between the trials"].value is None
    assert r.data["paired"]["t"] == pytest.approx(expected(a, b)["t"], rel=1e-12)
    assert r.data["welch"]["df"] == pytest.approx(7.0, rel=1e-12)


def test_a_p_below_what_a_double_holds_is_said(registry):
    entries = [f"E{i}" for i in range(200)]
    a = trial(entries=entries)
    b = {e: [v + 5.0 + 0.01 * math.sin(3.7 * i) for v in vs] for i, (e, vs) in enumerate(a.items())}
    k = named(two(registry, a, b))["paired t: p"]
    assert k.value == 0 and "p below 1e-300" in k.transformation


# --- 4. refusals ---------------------------------------------------------------------------------------------------

def test_the_same_table_or_the_same_rows_twice_are_refused(registry):
    rows = rows_of("S1", trial())
    tid = save(registry, rows)
    same = pair(registry, tid, tid)
    assert not same.ok and same.error.code == "invalid_argument"
    twin = pair(registry, tid, save(registry, [dict(x) for x in rows]))
    assert not twin.ok and "same rows" in twin.error.message


def test_each_side_is_one_study_and_the_two_studies_differ(registry):
    both = save(registry, rows_of("S1", trial()) + rows_of("S2", trial(1.0, seed=0.4)))
    r = pair(registry, both, save(registry, rows_of("S3", trial(2.0, seed=0.8))))
    assert not r.ok and r.error.code == "unsupported" and "2 studies" in r.error.message and "filter_rows" in r.error.message
    same = pair(registry, save(registry, rows_of("S1", trial())), save(registry, rows_of("S1", trial(1.0, seed=0.4))))
    assert not same.ok and "same study" in same.error.message


def test_two_variables_are_refused(registry):
    r = pair(registry, save(registry, rows_of("S1", trial())), save(registry, rows_of("S2", trial(1.0, seed=0.4), variable="V2")))
    assert not r.ok and r.error.code == "unsupported" and "V1" in r.error.message and "V2" in r.error.message


def test_several_values_of_one_plot_need_mean_per_unit(registry):
    rows = rows_of("S1", trial())
    rows.append(dict(rows[0], observationDbId="Oextra", value="99.0"))
    base, other = save(registry, rows), save(registry, rows_of("S2", trial(1.0, seed=0.4)))
    refused = pair(registry, base, other)
    assert not refused.ok and refused.error.code == "unsupported"
    averaged = pair(registry, base, other, duplicate_policy="mean_per_unit")
    assert averaged.ok, averaged.error
    a = trial()
    a["E0"] = [(a["E0"][0] + 99.0) / 2, a["E0"][1]]
    assert averaged.data["mean_base"] == pytest.approx(expected(a, trial(1.0, seed=0.4))["mean_base"], rel=1e-12)


def test_an_incomplete_table_needs_allow_incomplete_and_is_labelled(registry):
    base = save(registry, rows_of("S1", trial()), complete=False)
    other = save(registry, rows_of("S2", trial(1.0, seed=0.4)))
    refused = pair(registry, base, other)
    assert not refused.ok and refused.error.code == "incomplete_data"
    labelled = pair(registry, base, other, allow_incomplete=True)
    assert labelled.ok and not labelled.complete and all("INCOMPLETE" in c.transformation for c in claims(labelled))


def test_the_same_call_gives_the_same_claim_ids(registry):
    base, other = save(registry, rows_of("S1", trial())), save(registry, rows_of("S2", trial(1.0, seed=0.4)))
    assert [c.claim_id for c in claims(pair(registry, base, other))] == [c.claim_id for c in claims(pair(registry, base, other))]


# --- 5. the answer and the wiring ------------------------------------------------------------------------------------

def answer(ctx, claim_list) -> str:
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claim_list, caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    return cp.answer_summary(view)


def test_the_answer_shows_the_means_the_change_the_tests_and_the_sentence(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_pair_answer")
    r = pair(ctx.registry, save(ctx.registry, rows_of("S1", trial(3.0))), save(ctx.registry, rows_of("S2", trial(0.0, seed=0.4))),
             percent="reduction")
    shown = answer(ctx, claims(r))
    assert "**Paired comparison of entry means, 8 entries in both trials**" in shown
    assert "| Trial | Study | Mean of the entry means |" in shown and "| Base | S1 |" in shown and "| Other | S2 |" in shown
    assert "| Test | t (base - other) | df | p |" in shown and "| Paired t: the entries as pairs |" in shown
    assert "| Welch t: two independent samples |" in shown
    assert "Mean difference (base - other) " in shown and "Percent reduction of the base mean " in shown
    shown.encode("cp1252")                                    # round 2: the Windows console's code page holds every character
    assert "Pearson r of the entry means between the trials " in shown
    assert cp.PAIR_SENTENCE in shown and "| Statistic | Result |" not in shown and "**Correlation of entry means" not in shown


def test_an_undefined_number_is_shown_with_its_reason(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_pair_undefined")
    a = {"E0": [0.1], "E1": [0.2], "E2": [-0.3]}
    r = pair(ctx.registry, save(ctx.registry, rows_of("S1", a)), save(ctx.registry, rows_of("S2", {"E0": [0.5], "E1": [0.1], "E2": [0.9]})))
    shown = answer(ctx, claims(r))
    assert "Percent change of the base mean not available (" in shown


def test_every_number_the_answer_shows_is_backed_by_a_claim(tmp_path):
    """As T3 review 1: a number the answer shows that no claim carries is graded invented."""
    import importlib.util
    import sys
    from pathlib import Path
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_pair_backed")
    r = pair(ctx.registry, save(ctx.registry, rows_of("S1", trial(3.0))), save(ctx.registry, rows_of("S2", trial(0.0, seed=0.4))),
             percent="reduction")
    shown = answer(ctx, claims(r))
    spec = importlib.util.spec_from_file_location("_bench_for_pair", Path(at.__file__).resolve().parent / "eval" / "benchmark.py")
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    flagged = [n for n in bench._numbers_for_invention(bench.harness.code_written(shown))
               if not bench._backed(n[0], n[1], bench.claim_numbers([k.model_dump(mode="json") for k in claims(r)]))]
    assert flagged == []


def test_every_menu_and_prompt_names_the_tool():
    assert "compare_trials_paired" in at.ANALYST_TOOL_NAMES and len(at.ANALYST_TOOL_NAMES) == 17
    assert "compare_trials_paired" in an.NUMERIC_TOOLS and "compare_trials_paired" in an.DERIVING_TOOLS
    assert "seventeen fixed tools" in an.ANALYST_SYSTEM_PROMPT and "compare_trials_paired(base_artifact_id=" in an.ANALYST_SYSTEM_PROMPT
    assert "compare_trials_paired" in co.ANALYST_ACTIONS and "compare_trials_paired(base_artifact_id=" in co.COORDINATOR_SYSTEM_PROMPT
    schema = next(s for s in at.analyst_tool_schemas() if s["name"] == "compare_trials_paired")
    assert schema["inputSchema"]["required"] == ["base_artifact_id", "other_artifact_id"] and "paired t" in schema["description"]
    assert Statistic(kind="test", denominator="entries in both trials").kind == "test"


# --- 6. round 2: the review's findings and the builder's own probe ------------------------------------------------------------

def test_plot_values_that_cancel_inside_an_entry_give_no_percent_t_or_r(registry):
    """Builder's probe on round 1: the zero rules measured residue against the entry means, which are residue themselves when an
    entry's plot values cancel (0.1 + 0.2 - 0.3). Every entry mean is 0 on paper, yet round 1 stated percent -200 %, t -1.0,
    Welch t -1.41 and r -1.0 from float noise. Residue is now measured against the largest plot value behind the means."""
    a = {"E0": [0.1, 0.2, -0.3], "E1": [0.0, 0.0, 0.0], "E2": [0.0, 0.0, 0.0]}
    b = {"E0": [0.3, -0.1, -0.2], "E1": [0.0, 0.0, 0.0], "E2": [0.0, 0.0, 0.0]}
    r = two(registry, a, b)
    assert r.ok, r.error
    d = r.data
    assert d["percent"] is None and "is 0" in d["percent_missing_reason"]
    assert d["paired"]["t"] is None and d["paired"]["p"] is None and "same amount" in d["paired"]["missing_reason"] and d["sd_difference"] == 0
    assert d["welch"]["t"] is None and d["welch"]["df"] is None and "all equal" in d["welch"]["missing_reason"]
    assert d["pearson_r"] is None and "equal" in d["pearson_missing_reason"]
    assert all(row["percent"] == "" for row in per_entry(registry, r).values())


def test_a_base_with_negative_entry_means_gets_no_percent_even_when_its_mean_is_positive(registry):
    """Builder's probe on round 1: entry means of both signs are not on a ratio scale; their small positive mean gave 64900 %."""
    a, b = {"E0": [-1.0], "E1": [1.002], "E2": [0.0]}, {"E0": [-0.5], "E1": [1.5], "E2": [0.3]}
    r = two(registry, a, b)
    assert r.ok, r.error
    assert r.data["mean_base"] > 0 and r.data["percent"] is None and "(1 in the base trial, 1 in the other)" in r.data["percent_missing_reason"]   # the other's E0 is -0.5
    assert r.data["paired"]["t"] == pytest.approx(expected(a, b)["t"], rel=1e-12)


def test_the_t_claims_name_their_direction(registry):
    """Review F1: the Welch text said "difference of the means" with no direction; under percent='reduction' both t's are base - other."""
    a, b = trial(3.0), trial(0.0, seed=0.4)
    k = named(two(registry, a, b, percent="reduction"))
    assert "mean difference base - other" in k["paired t"].transformation and "(mean base - mean other)" in k["Welch t"].transformation
    k = named(two(registry, a, b))
    assert "mean difference other - base" in k["paired t"].transformation and "(mean other - mean base)" in k["Welch t"].transformation


def test_a_valid_value_without_an_entry_id_is_counted(registry):
    """Review F2 (inherited from _entry_means): such a value was dropped without a count; correlate_entry_means shares the function."""
    rows = rows_of("S1", trial())
    rows[0]["germplasmDbId"] = ""
    base, other = save(registry, rows), save(registry, rows_of("S2", trial(1.0, seed=0.4)))
    reasons = {e.reason: e.count for e in claims(pair(registry, base, other))[0].exclusions}
    assert reasons["base: rows with an empty germplasmDbId"] == 1
    cor = at.dispatch_analyst(registry, "correlate_entry_means", {"artifact_ids": [base, other]})
    assert cor.ok, cor.error
    assert {e["reason"]: e["count"] for e in cor.data["claims"][0]["exclusions"]}["x: rows with an empty germplasmDbId"] == 1


def test_the_refusal_counts_only_excluded_entries_that_had_a_mean(registry):
    """Review F3: the refusal said "excluded: 1" for a named entry without a valid value; the result counts only entries with a mean."""
    rows = rows_of("S1", trial(entries=ENTRIES[:2])) + [dict(rows_of("S1", {"E7": [1.0]})[0], value="NA")]
    r = pair(registry, save(registry, rows), save(registry, rows_of("S2", trial(1.0, entries=ENTRIES[:2], seed=0.4))), exclude=["E7"])
    assert not r.ok and "excluded: 0" in r.error.message


# --- 7. round 3: review 2 ------------------------------------------------------------------------------------------------------

def one_plot_rows(study: str, values: dict[str, list[float]]) -> list[dict[str, str]]:
    """Each entry's values as repeated measurements of ONE plot (one unit, distinct observation IDs)."""
    return [{"observationDbId": f"O{study}-{entry}-{k}", "observationUnitDbId": f"{study}-{entry}", "observationUnitName": f"plot {study}-{entry}",
             "germplasmDbId": entry, "germplasmName": f"clone {entry}", "studyDbId": study, "observationVariableDbId": "V1",
             "observationVariableName": "SYNTHETIC yield", "observationTimeStamp": "", "value": repr(float(value))}
            for entry, plot_values in values.items() for k, value in enumerate(plot_values, 1)]


def test_values_that_cancel_inside_one_plot_give_no_numbers_under_mean_per_unit(registry):
    """Review 2, F1: the residue scale came from the plot values after mean_per_unit averaged them, so a plot whose repeated values
    cancel was judged against its own residue: round 1's percent -200 %, t -1.0, Welch -1.41 and r -1.0 came back."""
    a = {"E0": [0.1, 0.2, -0.3], "E1": [0.0, 0.0, 0.0], "E2": [0.0, 0.0, 0.0]}
    b = {"E0": [0.3, -0.1, -0.2], "E1": [0.0, 0.0, 0.0], "E2": [0.0, 0.0, 0.0]}
    r = pair(registry, save(registry, one_plot_rows("S1", a)), save(registry, one_plot_rows("S2", b)), duplicate_policy="mean_per_unit")
    assert r.ok, r.error
    d = r.data
    assert d["percent"] is None and "is 0" in d["percent_missing_reason"]
    assert d["paired"]["t"] is None and d["welch"]["t"] is None and d["pearson_r"] is None
    assert all(row["percent"] == "" for row in per_entry(registry, r).values())


def test_a_plot_that_cancels_beside_real_entries_gets_no_per_entry_percent(registry):
    """Review 2, F1 (A3): the headline was right, but the cancelling entry's row stated -200 %."""
    a = {"E0": [0.1, 0.2, -0.3], "E1": [5.0, 5.0, 5.0], "E2": [7.0, 7.0, 7.0]}
    b = {"E0": [0.3, -0.1, -0.2], "E1": [6.0, 6.0, 6.0], "E2": [8.0, 8.0, 8.0]}
    r = pair(registry, save(registry, one_plot_rows("S1", a)), save(registry, one_plot_rows("S2", b)), duplicate_policy="mean_per_unit")
    assert r.ok, r.error
    rows = per_entry(registry, r)
    assert rows["E0"]["percent"] == "" and rows["E0"]["percent_note"] == "base mean is 0: no percent"
    assert float(rows["E1"]["percent"]) == pytest.approx(20.0, rel=1e-12)


def test_a_negative_entry_mean_on_either_side_gives_no_percent_anywhere(registry):
    """Review 2, F2: the rule looked at the base only (other -1, 2, 3 gave -33.33 % and a row of -200 %), and under reduction with a
    negative base the rows still stated -25 % and -16.67 %."""
    r = two(registry, {"E0": [1.0], "E1": [2.0], "E2": [3.0]}, {"E0": [-1.0], "E1": [2.0], "E2": [3.0]})
    assert r.data["percent"] is None and "(0 in the base trial, 1 in the other)" in r.data["percent_missing_reason"]
    assert all(row["percent"] == "" and "negative" in row["percent_note"] for row in per_entry(registry, r).values())
    r = two(registry, {"E0": [-1.0], "E1": [2.0], "E2": [3.0]}, {"E0": [1.0], "E1": [1.5], "E2": [2.5]}, percent="reduction")
    assert r.data["percent"] is None and all(row["percent"] == "" for row in per_entry(registry, r).values())
