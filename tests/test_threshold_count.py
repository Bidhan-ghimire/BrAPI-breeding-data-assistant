"""
tests/test_threshold_count.py - R2 tool "counts above a threshold" (coach 2026-10-22 10:52 UTC): how many entries (by their
entry mean) or plots (by their own value) are above or below a number, or above the mean of named entries, in each study.
Measured first: 5 benchmark sub-questions need it (A36.5, A42.5, C31.3, C32.3, C32.4; r2_need_20261022/NEED.md), and in all
five the threshold is named entries' mean, so the tool computes that mean itself.

The tool on a known small SYNTHETIC table, the menus that name it, and the answer's block and sentence. No network, no
model, no real data.
"""
from __future__ import annotations

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

# study, entry ID, entry name, plot values.
#   S1: Ref 11, Chk 31, P.1 13, P.2 11 (equal to Ref), P.3 6, P.4 no value, Q.1 21
#   S2: Ref 20, Chk 40, P.1 25, P.2 20 (equal to Ref)
ENTRIES = [("S1", "G1", "Ref", ["10", "12"]), ("S1", "G2", "Chk", ["30", "32"]), ("S1", "G3", "P.1", ["12", "14"]),
           ("S1", "G4", "P.2", ["11", "11"]), ("S1", "G5", "P.3", ["5", "7"]), ("S1", "G6", "P.4", ["", ""]),
           ("S1", "G7", "Q.1", ["20", "22"]),
           ("S2", "G1", "Ref", ["20", "20"]), ("S2", "G2", "Chk", ["40"]), ("S2", "G3", "P.1", ["25"]), ("S2", "G4", "P.2", ["19", "21"])]


def save(registry: ArtifactRegistry, entries=ENTRIES, *, complete: bool = True, variable: str = "V1", study_names=None) -> str:
    rows = [{"observationDbId": f"O{s}{g}{i}", "observationUnitDbId": f"P{s}{g}{i}", "observationUnitName": f"plot {s}-{g}-{i}",
             "germplasmDbId": g, "germplasmName": name, "studyDbId": s, **({"studyName": study_names.get(s, "")} if study_names else {}),
             "observationVariableDbId": variable, "observationVariableName": "SYNTHETIC yield", "observationTimeStamp": "", "value": value}
            for s, g, name, values in entries for i, value in enumerate(values)]
    return save_rows(registry, rows, sorted({e[0] for e in entries}), complete=complete, variable=variable)


def save_rows(registry: ArtifactRegistry, rows, study_ids, *, complete: bool = True, variable: str = "V1") -> str:
    columns = {c: srv.OBSERVATION_COLUMNS.get(c, "text") for c in rows[0]}
    return registry.save_table(rows, kind="observations", columns=columns, complete=complete, study_ids=list(study_ids),
                               variable_ids=[variable], measurement=MeasurementMeta(trait="SYNTHETIC yield", unit="t/ha"),
                               quality_notes=["SYNTHETIC threshold table"]).artifact_id


def row(unit, entry, name, value, *, study="S1", obs=None, when=""):
    return {"observationDbId": obs or f"O{unit}{when}", "observationUnitDbId": unit, "observationUnitName": f"plot {unit}", "germplasmDbId": entry,
            "germplasmName": name, "studyDbId": study, "observationVariableDbId": "V1", "observationVariableName": "SYNTHETIC yield",
            "observationTimeStamp": when, "value": value}


@pytest.fixture
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "out", "run_threshold", base_url=SYNTHETIC_BASE)


def count(registry, art, **args):
    return at.dispatch_analyst(registry, "threshold_count", {"artifact_id": art, **args})


def result(r, study="S1") -> tuple[int, int, float, float]:
    s = next(s for s in r.data["studies"] if s["studyDbId"] == study)
    return s["n_counted"], s["n_with_value"], round(s["share_percent"], 6), s["threshold"]


def claims(r) -> list[Claim]:
    return [Claim.model_validate(c) for c in r.data["claims"]]


# --- 1. the counts -------------------------------------------------------------------------------

def test_entries_above_a_named_entrys_mean_leave_out_the_named_and_excluded_entries(registry):
    r = count(registry, save(registry), count="entries", op="above", threshold_entries=["Ref"], exclude=["Chk"])
    assert r.ok and result(r) == (2, 4, 50.0, 11.0)                        # P.1 13 and Q.1 21 of P.1, P.2, P.3, Q.1; P.4 has no value
    s1 = next(s for s in r.data["studies"] if s["studyDbId"] == "S1")
    assert s1["threshold_entries"] == [{"entry": "G1", "label": "Ref", "mean": 11.0, "n_valid_values": 2}]
    assert s1["left_out"] == {"threshold entries": 1, "excluded": 1, "outside the name prefix": 0, "no valid value": 1, "no plot ID": 0}


def test_a_name_prefix_counts_only_those_entries(registry):
    r = count(registry, save(registry), count="entries", op="above", threshold_entries=["Ref"], name_prefix="P.")
    assert r.ok and result(r) == (1, 3, round(100 / 3, 6), 11.0)          # P.1 of P.1, P.2, P.3 (Q.1 and Chk are not "P.")


def test_the_name_prefix_is_an_exact_start_case_kept(registry):
    entries = [("S1", "G1", "Ref", ["10"]), ("S1", "G2", "P.1", ["20"]), ("S1", "G3", "XP.1", ["30"]), ("S1", "G4", "p.2", ["40"])]
    r = count(registry, save(registry, entries), count="entries", op="above", threshold_entries=["Ref"], name_prefix="P.")
    assert r.ok and result(r)[:2] == (1, 1)                                  # P.1 only: XP.1 contains "P.", p.2 differs in case
    assert next(s for s in r.data["studies"])["left_out"]["outside the name prefix"] == 2


def test_equal_is_not_above_but_is_at_least(registry):
    art = save(registry)
    above = count(registry, art, count="entries", op="above", threshold_entries=["Ref"], name_prefix="P.")
    at_least = count(registry, art, count="entries", op="at_least", threshold_entries=["Ref"], name_prefix="P.")
    assert result(above)[0] == 1 and result(at_least)[0] == 2                # P.2's mean 11 equals Ref's 11
    below = count(registry, art, count="entries", op="below", threshold_entries=["Ref"], name_prefix="P.")
    at_most = count(registry, art, count="entries", op="at_most", threshold_entries=["Ref"], name_prefix="P.")
    assert result(below)[0] == 1 and result(at_most)[0] == 2                 # P.3 6; P.3 and P.2


def test_each_study_gets_its_own_threshold(registry):
    r = count(registry, save(registry), count="entries", op="above", threshold_entries=["Ref"], name_prefix="P.")
    assert result(r, "S2") == (1, 2, 50.0, 20.0)                             # S2: Ref 20; P.1 25 above, P.2 20 equal
    assert [s["studyDbId"] for s in r.data["studies"]] == ["S1", "S2"]


def test_two_named_entries_give_the_unweighted_mean_of_their_entry_means(registry):
    r = count(registry, save(registry), count="entries", op="at_least", threshold_entries=["Ref", "Chk"])
    assert r.ok and result(r) == (1, 4, 25.0, 21.0)                         # (11 + 31) / 2 = 21; only Q.1 (21) reaches it
    assert result(r, "S2") == (0, 2, 0.0, 30.0)                              # (20 + 40) / 2 = 30; P.1 25 and P.2 20 are below it


def test_plots_are_counted_by_their_own_values(registry):
    r = count(registry, save(registry), count="plots", op="above", threshold_entries=["Ref"], name_prefix="P.")
    assert r.ok and result(r) == (2, 6, round(100 / 3, 6), 11.0)          # plots 12 and 14 of P.1; 11, 11, 5, 7 are not above 11
    assert result(r, "S2") == (2, 3, round(200 / 3, 6), 20.0)             # S2: plots 25 and 21 are above Ref's 20; 19 is not


def test_a_number_from_the_question_is_a_threshold_too(registry):
    r = count(registry, save(registry), count="entries", op="above", threshold=12.5, exclude=["Ref", "Chk"])
    assert r.ok and result(r)[:2] == (2, 4) and result(r)[3] == 12.5        # P.1 13 and Q.1 21
    assert not any(c.kind == "mean" for c in claims(r))                       # a stated number is not a computed claim


def test_means_equal_at_the_shown_precision_are_equal(registry):
    """0.1 and 0.2 average to 0.15000000000000002; a threshold entry of 0.15 and 0.15 gives 0.15. Equal, so not above."""
    entries = [("S1", "R", "Ref", ["0.15", "0.15"]), ("S1", "A", "A", ["0.1", "0.2"]), ("S1", "B", "B", ["0.3", "0.3"])]
    r = count(registry, save(registry, entries), count="entries", op="above", threshold_entries=["Ref"])
    assert r.ok and result(r)[:2] == (1, 2)                                  # only B


# --- 2. refusals -----------------------------------------------------------------------------------

@pytest.mark.parametrize("args, words", [
    ({}, "exactly one of threshold"),
    ({"threshold": 5, "threshold_entries": ["Ref"]}, "exactly one of threshold"),
    ({"threshold_entries": ["Reff"]}, "closest"),
    ({"threshold_entries": ["Ref"], "name_prefix": "Z."}, "no entry"),
], ids=["no-threshold", "both", "unknown-name", "prefix-matches-nothing"])
def test_a_threshold_or_population_that_cannot_be_read_is_refused(registry, args, words):
    r = count(registry, save(registry), count="entries", op="above", **args)
    assert not r.ok and words in r.error.message


def test_a_threshold_entry_without_a_value_in_a_study_is_refused(registry):
    entries = [e for e in ENTRIES if not (e[0] == "S2" and e[2] == "Ref")]     # Ref is not in S2
    r = count(registry, save(registry, entries), count="entries", op="above", threshold_entries=["Ref"])
    assert not r.ok and "S2" in r.error.message and "Ref" in r.error.message


def test_a_non_finite_threshold_an_incomplete_table_and_two_variables_are_refused(registry):
    art = save(registry)
    infinite = count(registry, art, count="entries", op="above", threshold=float("inf"))
    assert not infinite.ok and "finite" in infinite.error.message
    incomplete = save(registry, complete=False)
    refused = count(registry, incomplete, count="entries", op="above", threshold=1)
    assert not refused.ok and refused.error.code == "incomplete_data"
    labelled = count(registry, incomplete, count="entries", op="above", threshold=1, allow_incomplete=True)
    assert labelled.ok and any("INCOMPLETE" in w for w in labelled.warnings)


# --- 3. claims and the saved table -----------------------------------------------------------------

def test_each_study_gets_a_count_a_share_and_its_threshold_as_claims(registry):
    r = count(registry, save(registry), count="entries", op="above", threshold_entries=["Ref"], name_prefix="P.")
    by = {(c.kind, c.filters.get("studyDbId")): c for c in claims(r)}
    assert set(by) == {("count", "S1"), ("ratio", "S1"), ("mean", "S1"), ("count", "S2"), ("ratio", "S2"), ("mean", "S2")}
    assert by[("count", "S1")].value == 1 and by[("count", "S1")].denominator.n == 3 and by[("count", "S1")].unit is None
    assert by[("ratio", "S1")].unit == "%" and abs(by[("ratio", "S1")].value - 100 / 3) < 1e-9 and by[("ratio", "S1")].denominator.n == 3
    assert by[("mean", "S1")].value == 11.0 and by[("mean", "S1")].unit == "t/ha" and by[("mean", "S1")].denominator.n == 1
    assert "not a recommendation" in by[("count", "S1")].transformation and "above" in by[("count", "S1")].transformation


def test_other_settings_give_other_claim_ids(registry):
    art = save(registry)
    one = {c.claim_id for c in claims(count(registry, art, count="entries", op="above", threshold_entries=["Ref"]))}
    two = {c.claim_id for c in claims(count(registry, art, count="entries", op="above", threshold_entries=["Ref"], exclude=["Chk"]))}
    assert one and two and not one & two


def test_every_entry_or_plot_is_saved_with_its_role(registry):
    r = count(registry, save(registry), count="entries", op="above", threshold_entries=["Ref"], exclude=["Chk"])
    manifest, rows = registry.load(r.data["result_artifact_id"])
    assert manifest.kind == "analysis_result"
    roles = {(row["studyDbId"], row["entry"]): row["role"] for row in rows}
    assert roles[("S1", "G1")] == "threshold entry" and roles[("S1", "G2")] == "excluded" and roles[("S1", "G6")] == "no valid value"
    assert roles[("S1", "G3")] == "counted" and {row["side"] for row in rows if row["role"] == "counted"} <= {"above", "not above"}


# --- 4. the menus, the prompts and the answer --------------------------------------------------------

def test_every_menu_and_prompt_names_the_tool():
    assert "threshold_count" in at.ANALYST_TOOL_NAMES and len(at.ANALYST_TOOL_NAMES) == 16      # with correlate_entry_means and plot_design (R2)
    assert "threshold_count" in an.NUMERIC_TOOLS and "threshold_count" in an.DERIVING_TOOLS
    assert "sixteen fixed tools" in an.ANALYST_SYSTEM_PROMPT and "threshold_count" in an.ANALYST_SYSTEM_PROMPT
    assert "threshold_count" in co.ANALYST_ACTIONS and "threshold_count(" in co.COORDINATOR_SYSTEM_PROMPT
    schema = next(s for s in at.analyst_tool_schemas() if s["name"] == "threshold_count")
    assert "threshold_entries" in schema["inputSchema"]["properties"] and "named entries" in schema["description"]


def test_the_answer_shows_the_count_the_share_the_threshold_and_the_descriptive_sentence(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_threshold_answer")
    r = count(ctx.registry, save(ctx.registry), count="entries", op="above", threshold_entries=["Ref"], name_prefix="P.")
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claims(r), caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    shown = cp.answer_summary(view)
    assert "| Counted | Above the threshold | With a value | Share |" in shown
    assert "| entries | 1 | 3 | 33.3 % |" in shown and "| entries | 1 | 2 | 50 % |" in shown
    assert "11 t/ha" in shown and "Ref" in shown                              # the threshold and whose mean it is
    assert "Descriptive numbers only: a count of recorded values against a threshold, not a recommendation." in shown


def test_the_analyst_loop_keeps_the_threshold_claims(tmp_path):
    import json

    import anyio

    from llm import Budget, FakeModelClient, reply_text, reply_tools

    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_threshold_loop")
    inputs = an.synthetic_inputs(ctx, "SYNTHETIC: how many clones had a mean of V1 above 0 in S1?")
    observations = next(a for a in inputs.artifact_ids if ctx.registry.manifest(a).kind == "observations")
    final = json.dumps({"status": "completed", "payload": {"methods": ["threshold_count on the value column"], "caveats": [],
                                                           "narrative": "threshold_count states the count.", "key_claim_ids": []}})
    model = FakeModelClient([reply_tools([("table_info", json.dumps({"artifact_id": a})) for a in inputs.artifact_ids]),
                             reply_tools([("threshold_count", json.dumps({"artifact_id": observations, "count": "entries", "op": "above",
                                                                          "threshold": 0}))]), reply_text(final)])

    async def go():
        return await an.run_analyst(inputs, model=model, ctx=ctx, budget=Budget(), run_id="run_threshold_loop", log_dir=ctx.registry.out_root)

    report, _agent, _narrative = anyio.run(go)
    assert report.status == "completed" and model._replies == []
    assert {"count", "ratio"} <= {c.kind for c in report.claims} and report.result_artifact_ids


# --- 5. after the review of 2026-10-22 ------------------------------------------------------------------

def test_rows_without_an_entry_id_are_counted_and_stated(registry):
    """Review: a row with no germplasmDbId vanished with no exclusion line."""
    rows = [row("u1", "R", "PH.ref", "10"), row("u2", "A", "PH.a", "12"), row("u3", "", "PH.noid", "99"), row("u4", "B", "B", "13"),
            row("v1", "R", "PH.ref", "10", study="S2"), row("v2", "A", "PH.a", "12", study="S2")]
    r = count(registry, save_rows(registry, rows, ["S1", "S2"]), count="entries", op="above", threshold_entries=["PH.ref"], name_prefix="PH.")
    assert r.ok and r.data["rows_with_an_empty_entry_id"] == 1 and [s["rows_with_an_empty_entry_id"] for s in r.data["studies"]] == [1, 0]
    by = {c.filters["studyDbId"]: c for c in claims(r) if c.kind == "count"}
    assert any(e.reason == "rows with an empty germplasmDbId" and e.count == 1 for e in by["S1"].exclusions)
    assert not any("empty germplasmDbId" in e.reason for e in by["S2"].exclusions)


def test_plots_mode_refuses_a_plot_measured_at_two_times(registry):
    """Review: the same plot at two timestamps counted as two plots (duplicate_policy=reject sees two different times)."""
    rows = [row("uR", "R", "Ref", "10", when="t1"), row("uA", "A", "A", "12", when="t1"), row("uA", "A", "A", "12", when="t2"),
            row("uB", "B", "B", "5", when="t1")]
    art = save_rows(registry, rows, ["S1"])
    refused = count(registry, art, count="plots", op="above", threshold_entries=["Ref"])
    assert not refused.ok and refused.error.code == "unsupported" and "1 plot" in refused.error.message and "each plot once" in refused.error.message
    entries = count(registry, art, count="entries", op="above", threshold_entries=["Ref"])          # entries mode averages the two times
    assert entries.ok and result(entries)[:2] == (1, 2)


def test_missing_and_invalid_tokens_are_counted_per_study(registry):
    rows = [row("u1", "R", "Ref", "10"), row("u2", "A", "A", "12"), row("u3", "A", "A", "NA", obs="O3"), row("u5", "A", "A", "", obs="O5"),
            row("u4", "B", "B", "abc"), row("v1", "R", "Ref", "10", study="S2"), row("v2", "A", "A", "12", study="S2")]
    r = count(registry, save_rows(registry, rows, ["S1", "S2"]), count="entries", op="above", threshold_entries=["Ref"])
    by = {c.filters["studyDbId"]: c for c in claims(r) if c.kind == "count"}
    assert {(e.reason, e.count) for e in by["S1"].exclusions if "token" in e.reason} == {("raw-missing value token", 2), ("invalid or non-finite value token", 1)}
    assert not any("token" in e.reason for e in by["S2"].exclusions)


def test_the_share_never_reads_a_partial_count_as_0_or_100():
    assert cp._share(100 * 2500 / 2501) == "99.96 %" and cp._share(100 / 2501) == "0.04 %"
    assert cp._share(0.0) == "0 %" and cp._share(100.0) == "100 %" and cp._share(100 / 3) == "33.3 %" and cp._share(None) == "Not available"
    assert cp._share(100 * 999999 / 1000000) == "99.9999 %" and cp._share(100 * 9999999 / 10000000) == "over 99.9999 %"


def test_a_study_id_that_is_a_digit_but_not_decimal_renders(tmp_path):
    """Review: "²".isdigit() is true and int("²") fails, so the chat's study order crashed."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_threshold_digit")
    entries = [("²", g, name, values) for _, g, name, values in ENTRIES if _ == "S1"]
    r = count(ctx.registry, save(ctx.registry, entries), count="entries", op="above", threshold_entries=["Ref"])
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claims(r), caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    assert "| ² | entries |" in cp.answer_summary(view) and cp._sort_id("²") == (float("inf"), "²") and cp._sort_id("10") == (10, "10")


def test_an_entry_with_two_names_keeps_the_whole_threshold_line(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_threshold_names")
    rows = [row("u1", "R", "Ref", "10"), row("u2", "R", "Ref-alt", "12"), row("u3", "A", "A", "12"), row("u4", "B", "B", "5")]
    r = count(ctx.registry, save_rows(ctx.registry, rows, ["S1"]), count="entries", op="above", threshold_entries=["Ref"])
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claims(r), caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    shown = cp.answer_summary(view)
    assert "the entry mean of R \\(Ref; Ref-alt\\)\\)." in shown and "Left out: threshold entries 1." in shown


def test_a_threshold_written_as_text_works_because_plan_steps_carry_no_floats(registry):
    from pydantic import TypeAdapter, ValidationError

    import contracts as c

    with pytest.raises(ValidationError):
        TypeAdapter(c.PlanInputValue).validate_python(12.5)
    assert TypeAdapter(c.PlanInputValue).validate_python("12.5") == "12.5"
    r = count(registry, save(registry), count="entries", op="above", threshold="12.5", exclude=["Ref", "Chk"])
    assert r.ok and result(r)[:2] == (2, 4) and result(r)[3] == 12.5
    import agents.coordinator as co
    assert 'threshold="12.5"' in co.COORDINATOR_SYSTEM_PROMPT and "threshold=12.5" not in co.COORDINATOR_SYSTEM_PROMPT


def test_three_threshold_entries_give_their_mean_not_their_median(registry):
    entries = [("S1", "T1", "T1", ["1"]), ("S1", "T2", "T2", ["2"]), ("S1", "T3", "T3", ["9"]), ("S1", "A", "A", ["3"]), ("S1", "B", "B", ["5"])]
    r = count(registry, save(registry, entries), count="entries", op="above", threshold_entries=["T1", "T2", "T3"])
    assert r.ok and result(r) == (1, 2, 50.0, 4.0) and {c.kind: c.value for c in claims(r)}["mean"] == 4.0        # (1 + 2 + 9) / 3


def test_the_mean_claim_is_the_threshold_for_two_entries(registry):
    r = count(registry, save(registry), count="entries", op="at_least", threshold_entries=["Ref", "Chk"])
    assert {(c.kind, c.filters["studyDbId"]): c.value for c in claims(r)}[("mean", "S1")] == 21.0


def test_an_entry_with_several_names_matches_the_prefix_by_any_of_them(registry):
    rows = [row("u1", "R", "Ref", "10"), row("u2", "G", "Other", "12"), row("u3", "G", "P.g", "14"), row("u4", "H", "H", "20")]
    r = count(registry, save_rows(registry, rows, ["S1"]), count="entries", op="above", threshold_entries=["Ref"], name_prefix="P.")
    assert r.ok and result(r)[:2] == (1, 1)                                                              # G counts; H is outside


def test_values_differing_past_the_sixth_digit_are_not_equal(registry):
    entries = [("S1", "R", "Ref", ["1.0000001"]), ("S1", "A", "A", ["1.0000002"]), ("S1", "B", "B", ["1.0000001"])]
    r = count(registry, save(registry, entries), count="entries", op="above", threshold_entries=["Ref"])
    assert r.ok and result(r)[:2] == (1, 2)


def test_a_threshold_with_float_residue_is_equal_at_the_shown_precision(registry):
    entries = [("S1", "R", "Ref", ["0.1", "0.2"]), ("S1", "A", "A", ["0.15"]), ("S1", "B", "B", ["0.14"])]      # Ref = 0.15000000000000002
    r = count(registry, save(registry, entries), count="entries", op="at_least", threshold_entries=["Ref"])
    assert r.ok and result(r)[:2] == (1, 2)                                                              # A (0.15) reaches it


def test_op_and_count_change_the_claim_ids(registry):
    art = save(registry)
    settings = [dict(count="entries", op="above"), dict(count="entries", op="at_least"), dict(count="plots", op="above"),
                dict(count="entries", op="above", duplicate_policy="mean_per_unit"), dict(count="entries", op="above", allow_incomplete=True)]
    ids = [{c.claim_id for c in claims(count(registry, art, threshold_entries=["Ref"], **s))} for s in settings]
    assert all(ids) and all(not (ids[i] & ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids)))


def test_excluded_entries_stay_out_when_a_prefix_is_set(registry):
    r = count(registry, save(registry), count="entries", op="above", threshold_entries=["Ref"], name_prefix="P.", exclude=["P.3"])
    assert r.ok and result(r)[:2] == (1, 2) and result(count(registry, save(registry), count="entries", op="above", threshold_entries=["Ref"],
                                                             name_prefix="P."))[:2] == (1, 3)


def test_repeated_measurements_need_a_rule_and_are_averaged_per_plot_first(registry):
    rows = [row("u1", "R", "Ref", "15"), row("u2", "A", "A", "8", obs="O2a"), row("u2", "A", "A", "14", obs="O2b"), row("u5", "A", "A", "20"),
            row("u3", "B", "B", "5")]
    art = save_rows(registry, rows, ["S1"])
    refused = count(registry, art, count="entries", op="above", threshold_entries=["Ref"])
    assert not refused.ok and "duplicate_policy" in refused.error.message
    averaged = count(registry, art, count="entries", op="above", threshold_entries=["Ref"], duplicate_policy="mean_per_unit")
    assert averaged.ok and result(averaged)[:2] == (1, 2)      # A = mean(mean(8, 14), 20) = 15.5 > 15; the raw mean 14 would not be


def test_plots_mode_counts_plots_without_a_valid_value_by_row(registry):
    rows = [row("u1", "R", "Ref", "10"), row("u2", "A", "A", "12"), row("u3", "A", "A", "NA"), row("u4", "B", "B", "5")]
    art = save_rows(registry, rows, ["S1"])
    plots = count(registry, art, count="plots", op="above", threshold_entries=["Ref"])
    entries = count(registry, art, count="entries", op="above", threshold_entries=["Ref"])
    assert plots.data["studies"][0]["left_out"]["no valid value"] == 1 and entries.data["studies"][0]["left_out"]["no valid value"] == 0
    assert result(plots)[:2] == (1, 2) and result(entries)[:2] == (1, 2)


def test_an_incomplete_table_is_labelled_in_the_claims_and_at_least_is_named(registry):
    labelled = count(registry, save(registry, complete=False), count="entries", op="at_least", threshold=1, allow_incomplete=True)
    first = claims(labelled)[0]
    assert "INCOMPLETE" in first.transformation and "at least" in first.transformation


def test_the_answer_orders_studies_numerically_and_shows_study_names(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_threshold_order")
    entries = [(s, g, name, values) for s in ("9", "10") for _, g, name, values in ENTRIES if _ == "S1"]
    r = count(ctx.registry, save(ctx.registry, entries, study_names={"9": "Nine", "10": "Ten"}), count="entries", op="above", threshold_entries=["Ref"])
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claims(r), caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    shown = cp.answer_summary(view)
    assert shown.index("| 9 · Nine | entries |") < shown.index("| 10 · Ten | entries |")


def test_two_calls_make_two_tables_in_the_answer(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_threshold_two")
    art = save(ctx.registry)
    both = claims(count(ctx.registry, art, count="entries", op="above", threshold_entries=["Ref"])) + \
        claims(count(ctx.registry, art, count="entries", op="below", threshold_entries=["Ref"]))
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=both, caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    shown = cp.answer_summary(view)
    assert shown.count("**Counts against a threshold**") == 2 and "| Above the threshold |" in shown and "| Below the threshold |" in shown


def test_a_requested_mean_never_hides_the_threshold_block(tmp_path):
    """The chat may show only a clearly requested mean; a threshold answer with one study has exactly one mean claim (the threshold),
    and nothing of it may be hidden or flagged as "other statistics"."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_threshold_focus")
    entries = [e for e in ENTRIES if e[0] == "S1"]
    r = count(ctx.registry, save(ctx.registry, entries), count="entries", op="above", threshold_entries=["Ref"])
    plan = SimpleNamespace(statistic=SimpleNamespace(kind="mean"), question="How many entries have a mean SYNTHETIC yield above Ref's mean in S1?")
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claims(r), caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[], plan=plan)
    shown = cp.answer_summary(view)
    assert "**Counts against a threshold**" in shown and "Other calculated statistics remain" not in shown


# --- 6. after the second review of 2026-10-22 ---------------------------------------------------------

def test_plots_without_a_plot_id_are_left_out_and_stated_not_refused(registry):
    """Review 2: valid rows with a blank observationUnitDbId collapsed into one "plot" and plots mode refused with a false message."""
    rows = [row("uR", "R", "Ref", "10"), row("uA", "A", "A", "12"), row("", "A", "A", "14", obs="x1"), row("", "A", "A", "16", obs="x2"),
            row("uB", "B", "B", "5")]
    art = save_rows(registry, rows, ["S1"])
    plots = count(registry, art, count="plots", op="above", threshold_entries=["Ref"])
    assert plots.ok and result(plots)[:2] == (1, 2) and plots.data["studies"][0]["left_out"]["no plot ID"] == 2
    assert any(e.reason == "plots left out: no plot ID" and e.count == 2 for e in claims(plots)[0].exclusions)
    entries = count(registry, art, count="entries", op="above", threshold_entries=["Ref"])
    assert entries.ok and result(entries)[:2] == (1, 2) and entries.data["studies"][0]["left_out"]["no plot ID"] == 0   # A = 14 counts


def test_a_negative_text_threshold_keeps_its_sign(registry):
    entries = [("S1", "A", "A", ["-3"]), ("S1", "B", "B", ["-2"]), ("S1", "C", "C", ["-1"])]
    r = count(registry, save(registry, entries), count="entries", op="above", threshold="-2.5")
    assert r.ok and result(r) == (2, 3, round(200 / 3, 6), -2.5)


def test_two_repeated_plots_are_named_in_the_refusal_and_only_the_population_is_checked(registry):
    rows = [row("uR", "R", "Ref", "10", when="t1"), row("uR", "R", "Ref", "12", when="t2"), row("uA", "A", "A", "12", when="t1"),
            row("uA", "A", "A", "13", when="t2"), row("uB", "B", "B", "5", when="t1"), row("uB", "B", "B", "6", when="t2"), row("uC", "C", "C", "20")]
    art = save_rows(registry, rows, ["S1"])
    refused = count(registry, art, count="plots", op="above", threshold_entries=["Ref"])
    assert not refused.ok and "2 plots" in refused.error.message
    narrowed = count(registry, art, count="plots", op="above", threshold_entries=["Ref"], exclude=["A", "B"])    # only Ref repeats, outside the population
    assert narrowed.ok and result(narrowed)[:2] == (1, 1) and narrowed.data["studies"][0]["threshold"] == 11.0


def test_repeated_observation_ids_in_plots_mode_merge_under_mean_per_unit(registry):
    rows = [row("uR", "R", "Ref", "10"), row("uA", "A", "A", "12", obs="d1"), row("uA", "A", "A", "14", obs="d2"), row("uB", "B", "B", "5")]
    art = save_rows(registry, rows, ["S1"])
    assert not count(registry, art, count="plots", op="above", threshold_entries=["Ref"]).ok                       # needs a rule
    merged = count(registry, art, count="plots", op="above", threshold_entries=["Ref"], duplicate_policy="mean_per_unit")
    assert merged.ok and result(merged)[:2] == (1, 2)                                                            # uA once, at 13


def test_rows_without_a_study_or_an_entry_are_counted_once_each_everywhere(registry):
    rows = [row("u1", "R", "Ref", "10"), row("u2", "A", "A", "12"), row("u3", "B", "B", "5"),
            row("u4", "C", "C", "7", study=""), row("u5", "", "", "8", study=""), row("u7", "D", "D", "6", study=""), row("u8", "", "", "NA")]
    art = save_rows(registry, rows, ["S1"])
    for kind in ("entries", "plots"):
        r = count(registry, art, count=kind, op="above", threshold_entries=["Ref"])
        assert r.ok and result(r)[:2] == (1, 2)                                   # A and B; the u4, u5, u7, u8 rows are left out, never counted
        assert r.data["rows_with_an_empty_study_id"] == 3 and r.data["rows_with_an_empty_entry_id"] == 2     # u4, u5, u7; u5, u8
        first = claims(r)[0]
        assert any(e.reason == "rows with an empty studyDbId" and e.count == 3 for e in first.exclusions)
        assert any(e.reason == "rows with an empty germplasmDbId" and e.count == 1 for e in first.exclusions)   # u8, the one in S1
        assert r.data["studies"][0]["left_out"]["no valid value"] == 0            # u8's NA belongs to no entry of the population
