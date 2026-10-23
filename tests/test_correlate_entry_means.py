"""
tests/test_correlate_entry_means.py - R2 tool "Pearson/Spearman of entry means" (coach 2026-10-22 10:52 UTC next 2): the
correlation between the entry means of two tables, joined by entry: one trait in two studies, or two traits in the same
studies. Measured first: 15 benchmark sub-questions need it (r2_need_20261022/NEED.md). A descriptive correlation of
recorded entry means, not a recommendation.

The tool on known small SYNTHETIC tables, the contract's new claim kind, the menus that name it, and the answer's block and
sentence. No network, no model, no real data.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import agents.analyst as an
import agents.coordinator as co
import analyst_tools as at
import brapi_mcp_server as srv
import chat_presenter as cp
import contracts as c
from artifacts import ArtifactRegistry
from contracts import AnalysisReport, Claim, MeasurementMeta
from part2_support import SYNTHETIC_BASE, synthetic_context

# (entry ID, name, plot values) per study. X: A 1, B 2, C 3, D 4, E 5, F 6 (F only here). Y: A 1, B 3, C 2, D 4, E 50, G 7 (G only here).
# Pearson of the 5 shared: r = 99 / sqrt(10 x 1810) = 0.73584; Spearman (ranks 1-5 vs 1,3,2,4,5) = 0.9.
X = [("A", "Alpha", ["1", "1"]), ("B", "Beta", ["2", "2"]), ("C", "Gamma", ["3", "3"]), ("D", "Delta", ["4", "4"]), ("E", "Eps", ["5", "5"]),
     ("F", "Only-x", ["6"])]
Y = [("A", "Alpha", ["1", "1"]), ("B", "Beta", ["3", "3"]), ("C", "Gamma", ["2", "2"]), ("D", "Delta", ["4", "4"]), ("E", "Eps", ["50"]),
     ("G", "Only-y", ["7"]), ("I", "Only-y-too", ["8"])]
PEARSON, SPEARMAN = 99 / (10 * 1810) ** 0.5, 0.9


def save(registry: ArtifactRegistry, entries, *, study="S1", variable="V1", trait="SYNTHETIC yield", unit="t/ha", complete=True, studies=None) -> str:
    """One observations table; `studies` gives every row's study per entry index when the table spans several."""
    rows = []
    for i, (g, name, values) in enumerate(entries):
        s = studies[i] if studies else study
        rows += [{"observationDbId": f"O{s}{g}{j}", "observationUnitDbId": f"P{s}{g}{j}", "observationUnitName": f"plot {s}-{g}-{j}", "germplasmDbId": g,
                  "germplasmName": name, "studyDbId": s, "observationVariableDbId": variable, "observationVariableName": trait,
                  "observationTimeStamp": "", "value": value} for j, value in enumerate(values)]
    columns = {col: srv.OBSERVATION_COLUMNS.get(col, "text") for col in rows[0]}
    return registry.save_table(rows, kind="observations", columns=columns, complete=complete, study_ids=sorted({r["studyDbId"] for r in rows}),
                               variable_ids=[variable], measurement=MeasurementMeta(trait=trait, unit=unit), quality_notes=["SYNTHETIC"]).artifact_id


def save_rows(registry: ArtifactRegistry, rows, study_ids, *, variable="V1", trait="SYNTHETIC yield", unit="t/ha", complete=True) -> str:
    columns = {col: srv.OBSERVATION_COLUMNS.get(col, "text") for col in rows[0]}
    return registry.save_table(rows, kind="observations", columns=columns, complete=complete, study_ids=list(study_ids), variable_ids=[variable],
                               measurement=MeasurementMeta(trait=trait, unit=unit), quality_notes=["SYNTHETIC"]).artifact_id


def row(unit, entry, name, value, *, study="S1", obs=None, variable="V1"):
    return {"observationDbId": obs or f"O{unit}{value}", "observationUnitDbId": unit, "observationUnitName": f"plot {unit}", "germplasmDbId": entry,
            "germplasmName": name, "studyDbId": study, "observationVariableDbId": variable, "observationVariableName": "SYNTHETIC yield",
            "observationTimeStamp": "", "value": value}


@pytest.fixture
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "out", "run_cor", base_url=SYNTHETIC_BASE)


def correlate(registry, x, y, **args):
    return at.dispatch_analyst(registry, "correlate_entry_means", {"artifact_ids": [x, y], **args})


def claims(r) -> list[Claim]:
    return [Claim.model_validate(k) for k in r.data["claims"]]


def by_method(r) -> dict[str, Claim]:
    return {k.transformation.split(" ")[0].lower(): k for k in claims(r)}


# --- 1. the numbers ------------------------------------------------------------------------------

def test_pearson_and_spearman_of_the_shared_entries(registry):
    r = correlate(registry, save(registry, X), save(registry, Y, study="S2"))
    assert r.ok and r.data["n_pairs"] == 5 and r.data["pearson"] == pytest.approx(PEARSON) and r.data["spearman"] == pytest.approx(SPEARMAN)
    got = by_method(r)
    assert set(got) == {"pearson", "spearman"} and all(k.kind == "correlation" and k.unit is None and k.denominator.n == 5 for k in got.values())
    assert got["pearson"].value == pytest.approx(PEARSON) and got["spearman"].value == pytest.approx(SPEARMAN)
    assert len(got["pearson"].source_artifact_hashes) == 2 and "not a recommendation" in got["pearson"].transformation
    assert r.data["unpaired"] == {"only in x": 1, "only in y": 2}                       # F; G and I: never dropped silently


def test_one_method_gives_one_claim(registry):
    x, y = save(registry, X), save(registry, Y, study="S2")
    assert list(by_method(correlate(registry, x, y, method="pearson"))) == ["pearson"]
    assert list(by_method(correlate(registry, x, y, method="spearman"))) == ["spearman"]


def test_spearman_gives_tied_values_their_average_rank(registry):
    tied = [("A", "A", ["1"]), ("B", "B", ["2"]), ("C", "C", ["2"]), ("D", "D", ["4"]), ("E", "E", ["5"])]
    r = correlate(registry, save(registry, X[:5]), save(registry, tied, study="S2"), method="spearman")
    assert r.ok and r.data["spearman"] == pytest.approx((9.5 / 10) ** 0.5)                 # ranks 1, 2.5, 2.5, 4, 5


def test_the_order_of_the_two_tables_does_not_matter(registry):
    x, y = save(registry, X), save(registry, Y, study="S2")
    one, two = correlate(registry, x, y), correlate(registry, y, x)
    assert one.data["pearson"] == pytest.approx(two.data["pearson"]) and one.data["spearman"] == pytest.approx(two.data["spearman"])


def test_two_traits_of_the_same_study_are_joined_by_entry(registry):
    r = correlate(registry, save(registry, X), save(registry, Y, variable="V2", trait="SYNTHETIC dry matter", unit="%"))
    assert r.ok and r.data["n_pairs"] == 5 and r.data["pearson"] == pytest.approx(PEARSON)
    assert [t["trait"] for t in r.data["tables"]] == ["SYNTHETIC yield", "SYNTHETIC dry matter"]


def test_an_entry_mean_over_several_studies_is_the_unweighted_mean_of_its_study_means(registry):
    # X spans S1 and S2: A is 1 in S1 and 3 in S2 (mean 2); B 2 and 4 (3); C 3 and 5 (4); D 4 and 6 (5); E 5 and 7 (6); H only in S1.
    both = [("A", "A", ["1"]), ("B", "B", ["2"]), ("C", "C", ["3"]), ("D", "D", ["4"]), ("E", "E", ["5"]), ("H", "H", ["9"]),
            ("A", "A", ["3", "3"]), ("B", "B", ["4"]), ("C", "C", ["5"]), ("D", "D", ["6"]), ("E", "E", ["7"])]
    x = save(registry, both, studies=["S1"] * 6 + ["S2"] * 5)
    y = save(registry, Y, study="S3")
    r = correlate(registry, x, y)
    assert r.ok and r.data["n_pairs"] == 5 and r.data["pearson"] == pytest.approx(PEARSON)       # means 2..6 keep the same shape
    assert r.data["tables"][0]["entries_not_in_every_study"] == 1                                 # H
    manifest, rows = registry.load(r.data["result_artifact_id"])
    assert manifest.kind == "analysis_result" and {row["entry"]: row["x_mean"] for row in rows if row["paired"] == "true"}["A"] == "2.0"
    any_study = correlate(registry, x, save(registry, Y + [("H", "H", ["1"])], study="S3"), entries="in_any_study")
    assert any_study.ok and any_study.data["n_pairs"] == 6                                       # H's one-study mean counts


def test_excluded_entries_and_a_name_prefix_narrow_the_pairs(registry):
    x, y = save(registry, X), save(registry, Y, study="S2")
    without = correlate(registry, x, y, exclude=["Eps"])
    assert without.data["n_pairs"] == 4 and without.data["left_out"]["excluded"] == 1 and without.data["excluded_named"] == ["E"]
    named =[("P.1", "P.1", ["1"]), ("P.2", "P.2", ["2"]), ("P.3", "P.3", ["3"]), ("Q.1", "Q.1", ["4"]), ("Chk", "Chk", ["5"]), ("XP.4", "XP.4", ["7"])]
    other = [("P.1", "P.1", ["2"]), ("P.2", "P.2", ["1"]), ("P.3", "P.3", ["3"]), ("Q.1", "Q.1", ["9"]), ("Chk", "Chk", ["5"]), ("XP.4", "XP.4", ["8"])]
    r = correlate(registry, save(registry, named), save(registry, other, study="S2"), name_prefix="P.")
    assert r.ok and r.data["n_pairs"] == 3 and r.data["left_out"] == {"excluded": 0, "outside the name prefix": 3}   # XP.4 only contains "P."


def test_correlations_equal_at_the_shown_precision_keep_full_precision_in_the_claim(registry):
    r = correlate(registry, save(registry, X), save(registry, Y, study="S2"), method="pearson")
    assert by_method(r)["pearson"].value == r.data["pearson"] and r.data["display_pearson"] == "0.7359"     # 0.73586 at 4 decimals


# --- 2. refusals -----------------------------------------------------------------------------------

def test_too_few_pairs_a_constant_table_and_the_same_table_twice_are_refused(registry):
    x = save(registry, X)
    two = save(registry, Y[:2], study="S2")
    refused = correlate(registry, x, two)
    assert not refused.ok and refused.error.code == "unsupported" and "2 shared" in refused.error.message
    flat = save(registry, [("A", "A", ["3"]), ("B", "B", ["3"]), ("C", "C", ["3"]), ("D", "D", ["3"]), ("E", "E", ["3"])], study="S2")
    constant = correlate(registry, x, flat)
    assert not constant.ok and "all equal" in constant.error.message and "undefined" in constant.error.message
    same = correlate(registry, x, x)
    assert not same.ok and "same" in same.error.message


def test_an_incomplete_table_two_variables_in_one_table_and_a_name_column_are_refused(registry):
    x = save(registry, X)
    incomplete = save(registry, Y, study="S2", complete=False)
    refused = correlate(registry, x, incomplete)
    assert not refused.ok and refused.error.code == "incomplete_data"
    labelled = correlate(registry, x, incomplete, allow_incomplete=True)
    assert labelled.ok and any("INCOMPLETE" in w for w in labelled.warnings)
    assert not correlate(registry, x, save(registry, Y, study="S2"), by="germplasmName").ok


def test_an_unknown_excluded_name_is_refused_with_the_closest(registry):
    refused = correlate(registry, save(registry, X), save(registry, Y, study="S2"), exclude=["Alpah"])
    assert not refused.ok and "closest" in refused.error.message


# --- 3. the contract, the menus and the answer -----------------------------------------------------

def test_correlation_is_a_claim_kind_and_a_plan_statistic_kind():
    assert "correlation" in c.ClaimKind.__args__
    assert c.Statistic(kind="correlation", denominator="entries with a mean in both tables").kind == "correlation"


def test_every_menu_and_prompt_names_the_tool():
    assert "correlate_entry_means" in at.ANALYST_TOOL_NAMES and len(at.ANALYST_TOOL_NAMES) == 17
    assert "correlate_entry_means" in an.NUMERIC_TOOLS and "correlate_entry_means" in an.DERIVING_TOOLS
    assert "seventeen fixed tools" in an.ANALYST_SYSTEM_PROMPT and "correlate_entry_means" in an.ANALYST_SYSTEM_PROMPT
    assert "correlate_entry_means" in co.ANALYST_ACTIONS and "correlate_entry_means(artifact_ids=[" in co.COORDINATOR_SYSTEM_PROMPT
    schema = next(s for s in at.analyst_tool_schemas() if s["name"] == "correlate_entry_means")
    assert schema["inputSchema"]["required"] == ["artifact_ids"] and "Spearman" in schema["description"]


def test_other_settings_give_other_claim_ids(registry):
    x, y = save(registry, X), save(registry, Y, study="S2")
    one = {k.claim_id for k in claims(correlate(registry, x, y))}
    two = {k.claim_id for k in claims(correlate(registry, x, y, exclude=["Eps"]))}
    assert one and two and not one & two


def test_the_answer_shows_the_correlation_table_the_tables_and_the_descriptive_sentence(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_cor_answer")
    r = correlate(ctx.registry, save(ctx.registry, X), save(ctx.registry, Y, study="S2"))
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claims(r), caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    shown = cp.answer_summary(view)
    assert "| Method | Correlation | n |" in shown and "| Pearson r | 0.7359 | 5 |" in shown and "| Spearman rho | 0.9 | 5 |" in shown
    assert "x: SYNTHETIC yield \\(t/ha\\) in study S1; 6 entries with a mean; y: SYNTHETIC yield \\(t/ha\\) in study S2; 7 entries with a mean; " \
           "entries: in every study of each table." in shown                                      # which tables, which rule
    assert "Left out: only in x 1; only in y 2." in shown
    assert "Descriptive numbers only: a correlation of recorded entry means, not a recommendation." in shown
    assert "| Statistic | Result |" not in shown                                                 # not shown again as plain statistics


def test_the_analyst_loop_keeps_the_correlation_claims(tmp_path):
    import json

    import anyio

    from llm import Budget, FakeModelClient, reply_text, reply_tools

    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_cor_loop")
    inputs = an.synthetic_inputs(ctx, "SYNTHETIC: how do the entry means of V1 in S1 correlate with those of V2 in S1?")
    x, y = save(ctx.registry, X), save(ctx.registry, Y, variable="V2", trait="SYNTHETIC dry matter", unit="%")
    inputs = inputs.model_copy(update={"artifact_ids": [*inputs.artifact_ids, x, y]})
    final = json.dumps({"status": "completed", "payload": {"methods": ["correlate_entry_means on the two tables"], "caveats": [],
                                                           "narrative": "correlate_entry_means states the correlation.", "key_claim_ids": []}})
    model = FakeModelClient([reply_tools([("table_info", json.dumps({"artifact_id": a})) for a in inputs.artifact_ids]),
                             reply_tools([("correlate_entry_means", json.dumps({"artifact_ids": [x, y]}))]), reply_text(final)])

    async def go():
        return await an.run_analyst(inputs, model=model, ctx=ctx, budget=Budget(), run_id="run_cor_loop", log_dir=ctx.registry.out_root)

    report, _agent, _narrative = anyio.run(go)
    assert report.status == "completed" and model._replies == []
    assert {k.kind for k in report.claims} == {"correlation"} and report.result_artifact_ids


# --- 4. after the review of 2026-10-22 -----------------------------------------------------------------

def test_a_decimal_zero_mean_ties_with_an_exact_zero(registry):
    """Review: 0.1 + 0.2 - 0.3 averages to 9.25e-18, which did not tie with the exact 0 of other entries, so rho was 0.81 instead of 0.94."""
    x = [("E1", "E1", ["0.1", "0.2", "-0.3"]), ("E2", "E2", ["0"]), ("E3", "E3", ["0"]), ("E4", "E4", ["1"]), ("E5", "E5", ["2"]), ("E6", "E6", ["3"])]
    y = [(e, e, [str(i)]) for i, (e, _, _) in enumerate(x, start=1)]
    r = correlate(registry, save(registry, x), save(registry, y, study="S2"), method="spearman")
    assert r.ok and r.data["spearman"] == pytest.approx(15.5 / (15.5 * 17.5) ** 0.5)                 # ranks 2, 2, 2, 4, 5, 6 vs 1..6


@pytest.mark.parametrize("factor", ["1e155", "1e-200"], ids=["huge", "tiny"])
def test_huge_and_tiny_values_correlate_like_ordinary_ones(registry, factor):
    """Review: values near 1e155 overflowed inside the correlation to r = 0 with no refusal; values near 1e-200 crashed."""
    scaled = [(g, name, [f"{v}{factor[1:]}" if v != "0" else "0" for v in values]) for g, name, values in X]   # "1e155" -> "1e155", "2e155", ...
    r = correlate(registry, save(registry, scaled), save(registry, Y, study="S2"))
    assert r.ok and r.data["pearson"] == pytest.approx(PEARSON) and r.data["spearman"] == pytest.approx(SPEARMAN)


def test_values_beyond_1e200_are_refused_not_crashed(registry):
    refused = correlate(registry, save(registry, [("A", "A", ["1e300"]), ("B", "B", ["2e300"]), ("C", "C", ["3e300"])]), save(registry, Y, study="S2"))
    assert not refused.ok and refused.error.code == "unsupported" and "1e200" in refused.error.message


def test_the_same_rows_under_two_handles_are_refused(registry):
    one, two = save(registry, X), save(registry, X)
    assert one != two
    refused = correlate(registry, one, two)
    assert not refused.ok and "same rows" in refused.error.message


def test_partly_overlapping_study_sets_are_warned_equal_or_disjoint_ones_are_not(registry):
    both = [("A", "A", ["1"]), ("B", "B", ["2"]), ("C", "C", ["3"]), ("D", "D", ["4"]), ("E", "E", ["5"]),
            ("A", "A", ["3"]), ("B", "B", ["4"]), ("C", "C", ["5"]), ("D", "D", ["6"]), ("E", "E", ["7"])]
    x = save(registry, both, studies=["S1"] * 5 + ["S2"] * 5)
    partly = correlate(registry, x, save(registry, Y))                                             # y: S1 only
    assert partly.ok and any(w.startswith("x covers studies S1, S2 and y covers S1:") for w in partly.warnings)
    assert "x: SYNTHETIC yield (t/ha) in studies S1 and S2;" in claims(partly)[0].transformation    # two studies read "S1 and S2"
    assert not any("different study sets" in w for w in correlate(registry, x, save(registry, Y, study="S3")).warnings)
    assert not any("different study sets" in w for w in correlate(registry, save(registry, X), save(registry, Y)).warnings)


def test_entries_not_in_every_study_are_saved_with_their_reason(registry):
    both = [("A", "A", ["1"]), ("B", "B", ["2"]), ("C", "C", ["3"]), ("D", "D", ["4"]), ("E", "E", ["5"]), ("H", "H", ["9"]),
            ("A", "A", ["3"]), ("B", "B", ["4"]), ("C", "C", ["5"]), ("D", "D", ["6"]), ("E", "E", ["7"])]
    r = correlate(registry, save(registry, both, studies=["S1"] * 6 + ["S2"] * 5), save(registry, Y, study="S3"))
    rows = {row["entry"]: row for row in registry.load(r.data["result_artifact_id"])[1]}
    assert rows["H"]["paired"] == "false" and rows["H"]["left_out"] == "not in every study of x" and rows["H"]["x_mean"] == ""


def test_a_manifest_study_with_no_rows_is_said(registry):
    x = save_rows(registry, [row(f"u{g}", g, g, v) for g, v in (("A", "1"), ("B", "2"), ("C", "3"), ("D", "4"))], ["S1", "S9"])
    r = correlate(registry, x, save(registry, Y, study="S2"))
    assert r.ok and any(w.startswith("x: ") and "S9" in w and "no row" in w for w in r.warnings) and r.data["tables"][0]["study_ids"] == ["S1"]
    swapped = correlate(registry, save(registry, Y, study="S2"), x)                                 # the same table as y
    assert any(w.startswith("y: ") and "S9" in w for w in swapped.warnings)


def test_the_across_study_mean_is_the_mean_of_three_study_means_not_their_median(registry):
    three = [("A", "A", ["1"]), ("B", "B", ["5"]), ("C", "C", ["3"]), ("D", "D", ["4"]),
             ("A", "A", ["2"]), ("B", "B", ["5"]), ("C", "C", ["3"]), ("D", "D", ["4"]),
             ("A", "A", ["9"]), ("B", "B", ["5"]), ("C", "C", ["3"]), ("D", "D", ["4"])]
    r = correlate(registry, save(registry, three, studies=["S1"] * 4 + ["S2"] * 4 + ["S3"] * 4), save(registry, Y, study="S4"))
    rows = {row["entry"]: row for row in registry.load(r.data["result_artifact_id"])[1]}
    assert rows["A"]["x_mean"] == "4.0" and rows["A"]["x_n_studies"] == "3"                        # (1 + 2 + 9) / 3, not 2


def test_a_side_equal_at_the_shown_precision_is_constant(registry):
    flat = [("A", "A", ["0.1", "0.2"]), ("B", "B", ["0.3"]), ("C", "C", ["0.15", "0.15"]), ("D", "D", ["0.3"])]     # all 0.15 or 0.3? no: A 0.15, B 0.3
    constant = [("A", "A", ["0.1", "0.2"]), ("B", "B", ["0.15", "0.15"]), ("C", "C", ["0.05", "0.25"])]            # 0.15000000000000002, 0.15, 0.15
    refused = correlate(registry, save(registry, constant), save(registry, Y, study="S2"))
    assert not refused.ok and "all equal" in refused.error.message
    assert correlate(registry, save(registry, flat), save(registry, Y, study="S2")).ok


def test_names_from_either_table_serve_the_prefix_and_the_exclude(registry):
    x = save_rows(registry, [row("u1", "G", "Other", "1"), row("u2", "H", "H", "2"), row("u3", "I", "I", "3"), row("u4", "J", "J", "4")], ["S1"])
    y = save_rows(registry, [row("v1", "G", "P.g", "2", study="S2"), row("v2", "H", "P.h", "1", study="S2"), row("v3", "I", "P.i", "3", study="S2"),
                             row("v4", "J", "Chk-y", "9", study="S2")], ["S2"])
    r = correlate(registry, x, y, name_prefix="P.")
    assert r.ok and r.data["n_pairs"] == 3                                                          # G, H, I by their y names; J is outside
    assert correlate(registry, x, y, exclude=["Chk-y"]).data["n_pairs"] == 3                        # a name that exists only in y


def test_two_calls_make_two_blocks_in_the_answer(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_cor_two")
    x, y, z = save(ctx.registry, X), save(ctx.registry, Y, study="S2"), save(ctx.registry, Y, variable="V2", trait="SYNTHETIC dry matter", unit="%")
    both = claims(correlate(ctx.registry, x, y)) + claims(correlate(ctx.registry, x, z))
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=both, caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    shown = cp.answer_summary(view)
    assert shown.count("**Correlation of entry means") == 2 and shown.count("| Pearson r |") == 2


def test_the_entries_rule_changes_the_claim_ids(registry):
    both = [("A", "A", ["1"]), ("B", "B", ["2"]), ("C", "C", ["3"]), ("D", "D", ["4"]), ("E", "E", ["5"]), ("H", "H", ["9"]),
            ("A", "A", ["3"]), ("B", "B", ["4"]), ("C", "C", ["5"]), ("D", "D", ["6"]), ("E", "E", ["7"])]
    x, y = save(registry, both, studies=["S1"] * 6 + ["S2"] * 5), save(registry, Y, study="S3")
    every = {k.claim_id for k in claims(correlate(registry, x, y))}
    any_study = {k.claim_id for k in claims(correlate(registry, x, y, entries="in_any_study"))}
    assert every and any_study and not every & any_study


def test_repeated_measurements_need_a_rule_and_are_averaged_per_plot_first(registry):
    rows = [row("u1", "A", "A", "8", obs="d1"), row("u1", "A", "A", "14", obs="d2"), row("u5", "A", "A", "20"), row("u2", "B", "B", "2"),
            row("u3", "C", "C", "3"), row("u4", "D", "D", "4")]
    x, y = save_rows(registry, rows, ["S1"]), save(registry, Y, study="S2")
    refused = correlate(registry, x, y)
    assert not refused.ok and "duplicate_policy" in refused.error.message
    averaged = correlate(registry, x, y, duplicate_policy="mean_per_unit")
    assert averaged.ok and {r["entry"]: r["x_mean"] for r in registry.load(averaged.data["result_artifact_id"])[1]}["A"] == "15.5"   # not 14


def test_rows_without_a_study_are_not_a_study(registry):
    rows = [row("u1", "A", "A", "1"), row("u2", "B", "B", "2"), row("u3", "C", "C", "3"), row("u4", "D", "D", "4"), row("u5", "E", "E", "5", study="")]
    r = correlate(registry, save_rows(registry, rows, ["S1"]), save(registry, Y, study="S2"))
    assert r.ok and r.data["tables"][0]["study_ids"] == ["S1"] and r.data["n_pairs"] == 4
    assert any(e.reason == "x: rows with an empty studyDbId" and e.count == 1 for e in claims(r)[0].exclusions)


# --- 5. published fixtures (research_stats_tools_20261022/REPORT.md, the fixture table) ---------------------

def test_the_scipy_spearman_example_and_its_tie_ranks(registry):
    """scipy.stats.spearmanr([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]) = 0.8207826816681233; ties on [1, 2, 2, 4, 5, 5, 5, 8] rank to
    [1, 2.5, 2.5, 4, 6, 6, 6, 8]."""
    x = [(f"G{i}", f"G{i}", [str(v)]) for i, v in enumerate([1, 2, 3, 4, 5], start=1)]
    y = [(f"G{i}", f"G{i}", [str(v)]) for i, v in enumerate([5, 6, 7, 8, 7], start=1)]
    r = correlate(registry, save(registry, x), save(registry, y, study="S2"), method="spearman")
    assert r.ok and r.data["spearman"] == pytest.approx(0.8207826816681233, abs=1e-12)
    assert at._average_ranks([v / 8 for v in [1, 2, 2, 4, 5, 5, 5, 8]]) == [1, 2.5, 2.5, 4, 6, 6, 6, 8]


def test_the_metan_data_ge_genotype_means_fixture(registry):
    """metan data_ge, genotype means over 14 environments x 3 reps, GY vs HM (G1..G10): Pearson r -0.1046076 and Spearman rho
    -0.2121212 (cross-checked against scipy in the research notes). Two traits of the same genotypes, one value per genotype."""
    gy = [2.60368, 2.74415, 2.95534, 2.64192, 2.53734, 2.53403, 2.74091, 3.00364, 2.51019, 2.47122]
    hm = [47.0798, 46.6598, 47.6038, 48.0319, 49.2976, 48.7314, 47.9662, 49.1024, 47.9036, 48.5064]
    x = save(registry, [(f"G{i}", f"G{i}", [str(v)]) for i, v in enumerate(gy, start=1)], trait="GY", unit="t/ha")
    y = save(registry, [(f"G{i}", f"G{i}", [str(v)]) for i, v in enumerate(hm, start=1)], variable="V2", trait="HM", unit="%")
    r = correlate(registry, x, y)
    assert r.ok and r.data["n_pairs"] == 10
    # the notes print the means to 5 decimals: Pearson from them is -0.10460777 (1.7e-7 off the full-precision value); ranks are exact
    assert r.data["pearson"] == pytest.approx(-0.1046076, abs=5e-7) and r.data["spearman"] == pytest.approx(-0.2121212, abs=5e-8)


# --- 6. after the second review of 2026-10-22 ------------------------------------------------------------

def test_a_negative_side_keeps_its_sign(registry):
    """Review 2: scaling by the signed maximum would flip an all-negative side (r = +1 for -1)."""
    x = [(f"G{i}", f"G{i}", [str(-i)]) for i in range(1, 6)]
    y = [(f"G{i}", f"G{i}", [str(i)]) for i in range(1, 6)]
    r = correlate(registry, save(registry, x), save(registry, y, study="S2"))
    assert r.ok and r.data["pearson"] == pytest.approx(-1.0) and r.data["spearman"] == pytest.approx(-1.0)


def test_means_differing_at_the_eighth_digit_rank_apart(registry):
    x = [(f"G{i}", f"G{i}", [f"1.0000000{i}"]) for i in range(1, 6)]                                 # 1.00000001 .. 1.00000005
    y = [(f"G{i}", f"G{i}", [str(i)]) for i in range(1, 6)]
    r = correlate(registry, save(registry, x), save(registry, y, study="S2"))
    assert r.ok and r.data["spearman"] == pytest.approx(1.0) and r.data["pearson"] == pytest.approx(1.0, abs=1e-6)


def test_means_spanning_more_than_12_orders_of_magnitude_are_refused_not_tied(registry):
    """Review 2: with x = (1e13, 2, 3, 4, 5, 6) the small means tied falsely and rho was -0.17 for an exact 0.14."""
    y = [(f"G{i}", f"G{i}", [str(i)]) for i in range(1, 7)]
    wide = [("G1", "G1", ["1e13"]), *[(f"G{i}", f"G{i}", [str(i)]) for i in range(2, 7)]]
    refused = correlate(registry, save(registry, wide), save(registry, y, study="S2"))
    assert not refused.ok and "orders of magnitude" in refused.error.message
    eleven = [("G1", "G1", ["1e11"]), *[(f"G{i}", f"G{i}", [str(i)]) for i in range(2, 7)]]        # 11 orders: ranks 6, 1, 2, 3, 4, 5
    r = correlate(registry, save(registry, eleven), save(registry, y, study="S2"), method="spearman")
    assert r.ok and r.data["spearman"] == pytest.approx(2.5 / 17.5)


def test_values_of_exactly_1e200_are_refused_also_under_mean_per_unit(registry):
    for token in ("1e200", "-1e200"):
        huge = [(f"G{i}", f"G{i}", [token]) for i in range(1, 4)]
        refused = correlate(registry, save(registry, huge), save(registry, Y, study="S2"))
        assert not refused.ok and "1e200" in refused.error.message
    rows = [row("u1", "A", "A", "1.7e308", obs="d1"), row("u1", "A", "A", "1.7e308", obs="d2"), row("u2", "B", "B", "2"), row("u3", "C", "C", "3")]
    doubled = correlate(registry, save_rows(registry, rows, ["S1"]), save(registry, Y, study="S2"), duplicate_policy="mean_per_unit")
    assert not doubled.ok and doubled.error.code == "unsupported" and "1e200" in doubled.error.message       # refused before the averaging


def test_excluded_names_without_a_mean_are_named_but_not_counted_as_removed(registry):
    rows = [row("u1", "A", "A", "1"), row("u2", "B", "B", "2"), row("u3", "C", "C", "3"), row("u4", "D", "D", "4"), row("u5", "Z", "Zed", "NA")]
    r = correlate(registry, save_rows(registry, rows, ["S1"]), save(registry, Y, study="S2"), exclude=["Zed"])
    assert r.ok and r.data["left_out"]["excluded"] == 0 and r.data["excluded_named"] == ["Z"]
    assert not any(e.reason == "entries left out: excluded" for e in claims(r)[0].exclusions)              # the claim agrees: none removed
    only_y = correlate(registry, save(registry, X), save(registry, Y, study="S2"), exclude=["Only-y"])      # G has a mean in y only
    assert only_y.ok and only_y.data["left_out"]["excluded"] == 1 and only_y.data["unpaired"] == {"only in x": 1, "only in y": 1}
    assert any(e.reason == "entries left out: excluded" and e.count == 1 for e in claims(only_y)[0].exclusions)


def test_the_dropped_reason_names_the_table_that_lacks_the_entry(registry):
    both = [("A", "A", ["1"]), ("B", "B", ["2"]), ("C", "C", ["3"]), ("D", "D", ["4"]), ("E", "E", ["5"]), ("H", "H", ["9"]), ("K", "K", ["8"]),
            ("A", "A", ["3"]), ("B", "B", ["4"]), ("C", "C", ["5"]), ("D", "D", ["6"]), ("E", "E", ["7"])]
    x = save(registry, both, studies=["S1"] * 7 + ["S2"] * 5)                                       # H and K miss S2 in x
    y_rows = [*[(g, g, [v]) for g, v in (("A", "1"), ("B", "3"), ("C", "2"), ("D", "4"), ("E", "50"), ("K", "1"), ("M", "2"))],
              *[(g, g, [v]) for g, v in (("A", "1"), ("B", "3"), ("C", "2"), ("D", "4"), ("E", "50"), ("H", "1"))]]
    y = save(registry, y_rows, studies=["S3"] * 7 + ["S4"] * 6)                                     # K misses S4 and H misses S3 in y
    r = correlate(registry, x, y)
    rows = {row["entry"]: row["left_out"] for row in registry.load(r.data["result_artifact_id"])[1]}
    assert rows["H"] == "not in every study of x and y" and rows["K"] == "not in every study of x and y" and rows["M"] == "not in every study of y"


# --- T4's residue rule (coach 2026-10-23 03:48 UTC, NEXT 2 part c) ------------------------------------------------------------

def test_entry_means_that_are_float_residue_give_no_correlation(registry):
    """Plot values that cancel inside an entry (0.1 + 0.2 - 0.3) leave a mean of float residue. Judged against the means themselves
    that residue looked like spread: r -1.0 and rho -1.0 from noise. It is now judged against the largest plot value."""
    x = save(registry, [("E0", "a", ["0.1", "0.2", "-0.3"]), ("E1", "b", ["0.0", "0.0", "0.0"]), ("E2", "c", ["0.0", "0.0", "0.0"])])
    y = save(registry, [("E0", "a", ["0.3", "-0.1", "-0.2"]), ("E1", "b", ["0.0", "0.0", "0.0"]), ("E2", "c", ["0.0", "0.0", "0.0"])],
             study="S2")
    r = correlate(registry, x, y)
    assert not r.ok and r.error.code == "unsupported" and "all equal" in r.error.message


def test_the_residue_scale_comes_from_the_raw_values_under_mean_per_unit(registry):
    """T4 review 2: averaging a plot's repeated values first must not hide the scale of those values."""
    x = save_rows(registry, [row("P0", "E0", "a", v, obs=f"O0{i}") for i, v in enumerate(["0.1", "0.2", "-0.3"])]
                  + [row(f"P{k}", f"E{k}", n, "0.0") for k, n in ((1, "b"), (2, "c"))], ["S1"])
    y = save_rows(registry, [row("Q0", "E0", "a", v, study="S2", obs=f"Q0{i}") for i, v in enumerate(["0.3", "-0.1", "-0.2"])]
                  + [row(f"Q{k}", f"E{k}", n, "0.0", study="S2") for k, n in ((1, "b"), (2, "c"))], ["S2"])
    r = correlate(registry, x, y, duplicate_policy="mean_per_unit")
    assert not r.ok and r.error.code == "unsupported" and "all equal" in r.error.message
