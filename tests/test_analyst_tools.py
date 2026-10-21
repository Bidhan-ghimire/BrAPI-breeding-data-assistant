"""
tests/test_analyst_tools.py — the eight fixed calculation tools: right arithmetic on the six-row SYNTHETIC
example, honest counts (missing and invalid are never zero), named denominators, duplicates refused unless a
rule is named, full precision kept, ties reported, incomplete artifacts refused, paths refused.

Everything runs on SYNTHETIC artifacts in temp folders. No network, no model, no real data.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import anyio
import pytest

import analyst_tools as at
import brapi_mcp_server as srv
from artifacts import ArtifactRegistry
from contracts import Claim, MeasurementMeta
from part2_support import SYNTHETIC_BASE
from part2_support import synthetic_context

PART2 = Path(__file__).resolve().parents[1]
FIXTURES = PART2 / "tests" / "fixtures" / "analysis"
SIX_VALUES = ["10", "14", "", "20", "bad", "28"]


# --- helpers ------------------------------------------------------------------------------------

def call(registry, name, **args):
    return at.dispatch_analyst(registry, name, args)


def register(registry: ArtifactRegistry, fixture: str, table: str) -> str:
    """Register one SYNTHETIC fixture table as an artifact and return its handle."""
    doc = json.loads((FIXTURES / fixture).read_text(encoding="utf-8"))
    assert doc["_synthetic"] is True and "SYNTHETIC" in doc["_note"]
    t = doc["tables"][table]
    columns = {c: srv.OBSERVATION_COLUMNS.get(c, "text") for c in t["columns"]}
    manifest = registry.save_table(t["rows"], kind=t["kind"], columns=columns, complete=t["complete"], study_ids=t["study_ids"],
                                   variable_ids=t["variable_ids"], measurement=MeasurementMeta(**t["measurement"]),
                                   quality_notes=[f"SYNTHETIC fixture {fixture}:{table}"])
    return manifest.artifact_id


@pytest.fixture
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "out", "run_an", base_url=SYNTHETIC_BASE)


@pytest.fixture
def s1(tmp_path):
    """The teaching fixture through the real pipeline: registry, observations artifact, unit roster artifact, context."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_an")
    obs = srv.dispatch(ctx, "get_observations", {"study_db_id": "S1", "variable_db_id": "V1"})
    units = srv.dispatch(ctx, "get_observation_units", {"study_db_id": "S1"})
    assert obs.ok and units.ok
    return ctx.registry, obs.artifact_ids[0], units.artifact_ids[0], ctx


def claims_of(result) -> dict[str, Claim]:
    return {c["kind"]: Claim.model_validate(c) for c in result.data["claims"]}


# --- 1. the six-row example, exactly as the guide states it ----------------------------------------

def test_six_row_example_pooled_summary(s1):
    registry, obs, _, _ = s1
    r = call(registry, "numeric_summary", artifact_id=obs)
    assert r.ok, r.error
    assert r.data["counts"] == {"total_rows": 6, "valid": 4, "raw_missing": 1, "invalid_or_nonfinite": 1, "values_used": 4}
    assert r.data["stats"]["mean"] == (10 + 14 + 20 + 28) / 4 == 18.0
    assert r.data["stats"]["median"] == (14 + 20) / 2 and r.data["stats"]["min"] == 10.0 and r.data["stats"]["max"] == 28.0
    assert r.data["stats"]["sd"] == pytest.approx(math.sqrt(((10 - 18) ** 2 + (14 - 18) ** 2 + (20 - 18) ** 2 + (28 - 18) ** 2) / 3))
    assert r.data["display"] == {"mean": "18", "median": "17", "min": "10", "max": "28", "sd": "7.8316"}
    assert r.data["n_independent_units"] == 4 and r.data["variable_id"] == "V1"
    claims = claims_of(r)
    mean = claims["mean"]
    assert mean.value == 18.0 and mean.denominator.n == 4 and mean.denominator.name == "valid numeric values, one per observation unit"
    assert mean.n_independent_units == 4 and mean.source_artifact_hashes == [registry.manifest(obs).sha256]
    assert [(e.reason, e.count) for e in mean.exclusions] == [("raw-missing value token", 1), ("invalid or non-finite value token", 1)]
    assert claims["count"].value == 4 and claims["count"].unit is None and claims["sd"].value == pytest.approx(7.8316, abs=1e-4)
    assert mean.unit is None and any("unit not stated" in w for w in r.warnings)      # the observation rows carry no unit
    assert r.data["value_policy"]["missing_tokens"] == sorted(at.MISSING_TOKENS)


def test_six_row_example_group_means_and_ranking(s1):
    registry, obs, _, _ = s1
    r = call(registry, "group_stats", artifact_id=obs, by="germplasmDbId")
    assert r.ok, r.error
    shown = {g["group_id"]: g for g in r.data["shown"]}
    assert shown["GA"]["mean"] == 12.0 and shown["GA"]["n_valid_values"] == 2 and shown["GA"]["n_independent_units"] == 2
    assert shown["GB"]["mean"] == 24.0 and shown["GB"]["n_valid_values"] == 2 and shown["GB"]["n_independent_units"] == 2
    assert shown["GA"]["label"] == "clone A" and shown["GB"]["label"] == "clone B"            # names carried as labels
    assert shown["GA"]["n_raw_missing"] == 1 and shown["GB"]["n_invalid"] == 1                # kept per group, never zeroed
    assert [g["group_id"] for g in r.data["shown"]] == ["GB", "GA"] and shown["GB"]["rank"] == 1 and shown["GA"]["rank"] == 2
    assert r.data["ties"] == [] and r.data["groups_included"] == 2 and r.data["groups_excluded"] == 0
    claims = {c["filters"]["germplasmDbId"]: Claim.model_validate(c) for c in r.data["claims"]}
    assert claims["GA"].value == 12.0 and claims["GA"].denominator.n == 2 and claims["GA"].grouping == ["germplasmDbId"]
    assert "not evidence of genetic superiority" in r.data["analysis_target"]
    manifest, rows = registry.load(r.data["result_artifact_id"])
    assert manifest.kind == "analysis_result" and [row["group_id"] for row in rows] == ["GA", "GB"]
    assert rows[0]["mean"] == "12.0" and rows[0]["included"] == "true" and rows[0]["rank"] == "2"
    assert any(n.startswith("lineage: group_stats by germplasmDbId on " + obs) for n in manifest.quality_notes)


def test_min_independent_n_is_an_exclusion_threshold_not_a_correction(s1):
    registry, obs, _, _ = s1
    r = call(registry, "group_stats", artifact_id=obs, by="germplasmDbId", min_independent_n=3)
    assert r.ok and r.data["groups_included"] == 0 and r.data["shown"] == [] and r.data["claims"] == []
    assert {g["group_id"]: g["reason"] for g in r.data["excluded"]} == {"GA": "fewer than 3 independent units: 2", "GB": "fewer than 3 independent units: 2"}
    _, rows = registry.load(r.data["result_artifact_id"])
    assert len(rows) == 2 and all(row["included"] == "false" and row["exclusion_reason"] for row in rows)   # excluded groups stay in the full result


def test_group_by_a_name_column_is_refused(s1):
    registry, obs, _, _ = s1
    r = call(registry, "group_stats", artifact_id=obs, by="germplasmName")
    assert not r.ok and r.error.code == "invalid_argument" and "ID column" in r.error.message


# --- 2. duplicates: detected, refused, or aggregated by an explicit rule -----------------------------

def test_repeated_measurements_need_an_explicit_rule(registry):
    art = register(registry, "duplicate_plots.json", "repeated_measurements")
    info = call(registry, "table_info", artifact_id=art)
    assert info.data["duplicates"]["repeated_measurements_per_unit"] == 1 and info.data["duplicates"]["repeated_observation_ids"] == 0
    refused = call(registry, "numeric_summary", artifact_id=art)
    assert not refused.ok and refused.error.code == "unsupported" and "duplicate_policy='mean_per_unit'" in refused.error.message
    r = call(registry, "numeric_summary", artifact_id=art, duplicate_policy="mean_per_unit")
    assert r.ok and r.data["counts"]["valid"] == 5 and r.data["counts"]["values_used"] == 4      # P1 = mean(10, 12) = 11
    assert r.data["stats"]["mean"] == (11 + 14 + 20 + 28) / 4 and "averaged within 4 observation units" in r.data["transformation"]
    assert r.data["n_independent_units"] == 4 and claims_of(r)["mean"].denominator.n == 4
    g = call(registry, "group_stats", artifact_id=art, by="germplasmDbId", duplicate_policy="mean_per_unit")
    assert g.ok and {x["group_id"]: x["mean"] for x in g.data["shown"]} == {"GA": (11 + 14) / 2, "GB": 24.0}


def test_repeated_observation_ids_are_detected(registry):
    art = register(registry, "duplicate_plots.json", "repeated_observation_id")
    info = call(registry, "table_info", artifact_id=art)
    assert info.data["duplicates"]["repeated_observation_ids"] == 1 and info.data["duplicates"]["repeated_measurements_per_unit"] == 1
    refused = call(registry, "numeric_summary", artifact_id=art)
    assert not refused.ok and "1 repeated observation ID" in refused.error.message
    r = call(registry, "numeric_summary", artifact_id=art, duplicate_policy="mean_per_unit")
    assert r.ok and r.data["stats"]["mean"] == (10 + 14 + 20 + 28) / 4 and r.data["counts"]["values_used"] == 4


# --- 3. tokens: infinity, odd numbers, leading zeros -----------------------------------------------

def test_infinity_and_odd_tokens_are_invalid_never_measurements(registry):
    assert at.classify_value("inf") == ("invalid", None) and at.classify_value("-inf") == ("invalid", None)
    assert at.classify_value("nan") == ("invalid", None) and at.classify_value("1e400") == ("invalid", None)
    assert at.classify_value("1_000") == ("invalid", None) and at.classify_value("0x10") == ("invalid", None)
    assert at.classify_value(" 7 ") == ("valid", 7.0) and at.classify_value("2e3") == ("valid", 2000.0) and at.classify_value("-3.5") == ("valid", -3.5)
    assert at.classify_value("") == ("missing", None) and at.classify_value("NA") == ("missing", None) and at.classify_value(None) == ("missing", None)
    art = register(registry, "infinity.json", "nonfinite")
    r = call(registry, "numeric_summary", artifact_id=art)
    assert r.ok and r.data["counts"] == {"total_rows": 8, "valid": 2, "raw_missing": 0, "invalid_or_nonfinite": 6, "values_used": 2}
    assert r.data["stats"]["mean"] == (2000 + 7) / 2 and r.data["stats"]["max"] == 2000.0
    assert [(e.reason, e.count) for e in claims_of(r)["mean"].exclusions] == [("invalid or non-finite value token", 6)]


def test_leading_zero_ids_stay_distinct(registry):
    art = register(registry, "leading_zero_ids.json", "clones")
    g = call(registry, "group_stats", artifact_id=art, by="germplasmDbId", min_independent_n=1)
    assert g.ok and {x["group_id"]: x["mean"] for x in g.data["shown"]} == {"007": 11.0, "7": 20.0, "0007": 30.0}
    c = call(registry, "count_records", artifact_id=art, distinct_by="germplasmDbId")
    assert c.ok and c.data["distinct_count"] == 3 and c.data["distinct_values_preview"] == ["0007", "007", "7"]
    f = call(registry, "filter_rows", artifact_id=art, column="germplasmDbId", op="eq", value="007")
    assert f.ok and f.data["rows_out"] == 2 and all(row["germplasmDbId"] == "007" for row in registry.load(f.data["artifact_id"])[1])


# --- 4. one variable, compatible units, lineage ----------------------------------------------------

def test_two_variables_inside_one_table_are_refused(registry):
    art = register(registry, "mixed_units.json", "two_variables_inside")
    for name, extra in (("numeric_summary", {}), ("group_stats", {"by": "germplasmDbId"})):
        r = call(registry, name, artifact_id=art, **extra)
        assert not r.ok and r.error.code == "unsupported" and "more than one variable" in r.error.message, name


def test_concat_compatible_tables_keeps_lineage(registry):
    a = register(registry, "mixed_units.json", "yield_t_ha")
    b = register(registry, "mixed_units.json", "compatible_s2")
    r = call(registry, "concat_tables", artifact_ids=[a, b])
    assert r.ok, r.error
    assert r.data["row_count"] == 4 and r.data["study_ids"] == ["S1", "S2"] and r.data["variable_ids"] == ["V1"] and r.data["complete"] is True
    manifest, rows = registry.load(r.data["artifact_id"])
    assert manifest.kind == "analysis_result" and "source_artifact_id" in manifest.columns
    assert [row["source_artifact_id"] for row in rows] == [a, a, b, b] and manifest.measurement.unit == "t/ha"
    assert any(n.startswith(f"lineage: concat_tables of {a} sha256=") for n in manifest.quality_notes)
    s = call(registry, "numeric_summary", artifact_id=r.data["artifact_id"])
    assert s.ok and s.data["stats"]["mean"] == 25.0 and claims_of(s)["mean"].unit == "t/ha" and s.data["n_independent_units"] == 4


def test_concat_refuses_incompatible_tables(registry):
    t_ha = register(registry, "mixed_units.json", "yield_t_ha")
    kg = register(registry, "mixed_units.json", "yield_kg_plot")
    dm = register(registry, "mixed_units.json", "dry_matter")
    later = register(registry, "mixed_units.json", "later_timepoint")
    r = call(registry, "concat_tables", artifact_ids=[t_ha, kg])
    assert not r.ok and r.error.code == "unsupported" and "incompatible measurement metadata (unit)" in r.error.message
    r = call(registry, "concat_tables", artifact_ids=[t_ha, dm])
    assert not r.ok and r.error.code == "unsupported"                                # different variable, unit and trait
    r = call(registry, "concat_tables", artifact_ids=[t_ha, later])
    assert not r.ok and "timepoint" in r.error.message
    r = call(registry, "concat_tables", artifact_ids=[t_ha, t_ha])
    assert not r.ok and r.error.code == "invalid_argument" and "twice" in r.error.message
    r = call(registry, "concat_tables", artifact_ids=[t_ha])
    assert not r.ok and r.error.code == "invalid_argument"                          # the schema demands two or more
    assert registry.artifact_ids() == sorted([t_ha, kg, dm, later])                  # nothing was saved by a refusal


def test_concat_refuses_different_schemas(s1):
    registry, obs, units, _ = s1
    r = call(registry, "concat_tables", artifact_ids=[obs, units])
    assert not r.ok and r.error.code == "unsupported" and "incompatible schemas" in r.error.message


# --- 5. ties, all-missing data, odd group IDs, incomplete artifacts ---------------------------------

def test_group_stats_can_state_every_group_of_a_real_size_study(registry):
    """Real data, 2026-09-29, round 5 (Q6): study 4491 has 157 clones; the old ceiling top_n <= 100 left 57 clones without a claim,
    so 'how many plots does each clone have' could not be answered. The ceiling is now 1000; claims are still best-first and top_n
    still limits what is stated."""
    rows = [{"observationDbId": f"O{i}", "observationUnitDbId": f"P{i}", "observationUnitName": f"plot {i}", "germplasmDbId": f"G{i:03d}",
             "germplasmName": f"clone {i}", "studyDbId": "S1", "observationVariableDbId": "V1", "observationVariableName": "SYNTHETIC score",
             "observationTimeStamp": "", "value": str(i % 6)} for i in range(157)]
    columns = {c: srv.OBSERVATION_COLUMNS.get(c, "text") for c in rows[0]}
    art = registry.save_table(rows, kind="observations", columns=columns, complete=True, study_ids=["S1"], variable_ids=["V1"],
                              measurement=MeasurementMeta(trait="SYNTHETIC score"), quality_notes=["SYNTHETIC 157-clone table"]).artifact_id
    every = call(registry, "group_stats", artifact_id=art, by="germplasmDbId", min_independent_n=1, top_n=1000)
    assert every.ok and len(every.data["claims"]) == 157
    assert {c["filters"]["germplasmDbId"] for c in every.data["claims"]} == {f"G{i:03d}" for i in range(157)}
    assert all(c["n_independent_units"] == 1 for c in every.data["claims"])
    capped = call(registry, "group_stats", artifact_id=art, by="germplasmDbId", min_independent_n=1, top_n=100)
    assert capped.ok and len(capped.data["claims"]) == 100
    refused = call(registry, "group_stats", artifact_id=art, by="germplasmDbId", top_n=1001)
    assert not refused.ok and refused.error.code == "invalid_argument"


def test_tied_means_share_a_rank_deterministically(registry):
    art = register(registry, "tied_means.json", "three_clones")
    r = call(registry, "group_stats", artifact_id=art, by="germplasmDbId")
    assert r.ok and [(g["group_id"], g["mean"], g["rank"]) for g in r.data["shown"]] == [("A", 15.0, 1), ("B", 15.0, 1), ("C", 12.0, 3)]
    assert r.data["ties"] == [["A", "B"]] and "share a rank" in r.data["tie_rule"]
    up = call(registry, "group_stats", artifact_id=art, by="germplasmDbId", ascending=True)
    assert [(g["group_id"], g["rank"]) for g in up.data["shown"]] == [("C", 1), ("A", 2), ("B", 2)]
    again = call(registry, "group_stats", artifact_id=art, by="germplasmDbId")
    assert [g["group_id"] for g in again.data["shown"]] == [g["group_id"] for g in r.data["shown"]]      # a rerun gives the same order


def test_all_missing_data_gives_null_claims_with_reasons(registry):
    art = register(registry, "all_missing.json", "nothing_valid")
    r = call(registry, "numeric_summary", artifact_id=art)
    assert r.ok and r.data["counts"] == {"total_rows": 4, "valid": 0, "raw_missing": 3, "invalid_or_nonfinite": 1, "values_used": 0}
    claims = claims_of(r)
    assert claims["mean"].value is None and claims["mean"].missing_reason == "no valid numeric values" and claims["count"].value == 0
    assert claims["sd"].value is None and r.data["stats"] == {"mean": None, "median": None, "min": None, "max": None, "sd": None}
    g = call(registry, "group_stats", artifact_id=art, by="germplasmDbId")
    assert g.ok and g.data["groups_included"] == 0 and len(registry.load(g.data["result_artifact_id"])[1]) == 2
    one = register(registry, "all_missing.json", "one_valid")
    r = call(registry, "numeric_summary", artifact_id=one)
    assert r.ok and claims_of(r)["mean"].value == 5.0 and claims_of(r)["sd"].value is None
    assert claims_of(r)["sd"].missing_reason == "fewer than 2 valid values: sample SD undefined"


def test_string_and_empty_group_ids(registry):
    art = register(registry, "string_null_groups.json", "odd_group_ids")
    r = call(registry, "group_stats", artifact_id=art, by="germplasmDbId", min_independent_n=1)
    assert r.ok and r.data["rows_without_group_id"] == 1
    assert {g["group_id"]: g["label"] for g in r.data["shown"]} == {"GA": "clone GA", "None": "literally None", "null": "literally null"}
    assert any(e["reason"] == "rows with an empty germplasmDbId" and e["count"] == 1 for e in r.data["exclusions"])
    m = call(registry, "missing_report", artifact_id=art, by="germplasmDbId")
    assert m.ok and {g["group_id"]: g["total_rows"] for g in m.data["groups"]} == {at.MISSING_GROUP: 1, "GA": 2, "None": 1, "null": 1}


def test_incomplete_artifact_is_refused_unless_explicitly_allowed(registry):
    art = register(registry, "incomplete.json", "partial_fetch")
    for name, extra in (("numeric_summary", {}), ("group_stats", {"by": "germplasmDbId"}), ("count_records", {})):
        r = call(registry, name, artifact_id=art, **extra)
        assert not r.ok and r.error.code == "incomplete_data", name
    r = call(registry, "numeric_summary", artifact_id=art, allow_incomplete=True)
    assert r.ok and r.complete is False and r.data["transformation"].startswith("INCOMPLETE ARTIFACT")
    assert claims_of(r)["mean"].transformation.startswith("mean of valid values; INCOMPLETE ARTIFACT") and r.warnings


# --- 6. handles, not paths ----------------------------------------------------------------------

@pytest.mark.parametrize("bad,code", [
    ("../x.csv", "invalid_argument"), ("C:\\Users\\someone\\secret.csv", "invalid_argument"), ("/etc/passwd", "invalid_argument"),
    ("out/run_an/artifacts/0001_observations.csv", "invalid_argument"), ("art_abcdef_0001", "not_found"),
])
def test_paths_and_unknown_handles_are_refused(registry, bad, code):
    for name, extra in (("table_info", {}), ("numeric_summary", {}), ("count_records", {})):
        r = call(registry, name, artifact_id=bad, **extra)
        assert not r.ok and r.error.code == code, (name, bad)
    r = call(registry, "table_info", artifact_id=5)
    assert not r.ok and r.error.code == "invalid_argument"


def test_module_makes_no_forbidden_calls():
    source = (PART2 / "analyst_tools.py").read_text(encoding="utf-8")
    body = source.split('"""', 2)[2]                                   # skip the module docstring, which names the rules
    for token in ("eval(", "exec(", "subprocess", "os.system", "requests", "urllib", "httpx", "socket", "openai", "open(", "Path("):
        assert token not in body, token


# --- 7. filter_rows, missing_report, count_records, table_info, plot_bar -----------------------------

def test_filter_rows_operations_are_literal_and_typed(s1):
    registry, obs, _, _ = s1
    r = call(registry, "filter_rows", artifact_id=obs, column="germplasmName", op="contains", value="CLONE a")
    assert r.ok and r.data["rows_out"] == 3 and r.complete is True
    manifest, rows = registry.load(r.data["artifact_id"])
    assert manifest.kind == "observations" and list(manifest.columns) == list(registry.manifest(obs).columns)
    assert [row["observationUnitDbId"] for row in rows] == ["P1", "P2", "P3"] and rows[2]["value"] == ""     # the missing token survives
    assert any(n.startswith(f"lineage: filter_rows: germplasmName contains 'CLONE a' on {obs}") for n in manifest.quality_notes)
    assert call(registry, "filter_rows", artifact_id=obs, column="germplasmName", op="contains", value=".").data["rows_out"] == 0   # literal, not regex
    gt = call(registry, "filter_rows", artifact_id=obs, column="value", op="gt", value="12")
    assert gt.ok and gt.data["rows_out"] == 3 and gt.data["dropped_non_numeric"] == 2
    assert call(registry, "filter_rows", artifact_id=obs, column="value", op="is_missing").data["rows_out"] == 1
    assert call(registry, "filter_rows", artifact_id=obs, column="value", op="is_valid_number").data["rows_out"] == 4
    assert call(registry, "filter_rows", artifact_id=obs, column="value", op="gt", value="abc").error.code == "invalid_argument"
    assert call(registry, "filter_rows", artifact_id=obs, column="nope", op="eq", value="x").error.code == "invalid_argument"
    assert call(registry, "filter_rows", artifact_id=obs, column="value", op="regex", value=".*").error.code == "invalid_argument"


def test_absent_study_gives_an_empty_table_not_an_error(s1):
    registry, obs, _, _ = s1
    none = call(registry, "filter_rows", artifact_id=obs, column="studyDbId", op="eq", value="S9")
    assert none.ok and none.data["rows_out"] == 0
    s = call(registry, "numeric_summary", artifact_id=none.data["artifact_id"])
    assert s.ok and s.data["counts"]["total_rows"] == 0 and claims_of(s)["mean"].value is None and claims_of(s)["count"].value == 0
    c = call(registry, "count_records", artifact_id=none.data["artifact_id"])
    assert c.ok and c.data["total_rows"] == 0 and claims_of(c)["count"].value == 0


def test_missing_report_with_and_without_a_verified_roster(s1):
    registry, obs, units, _ = s1
    r = call(registry, "missing_report", artifact_id=obs, by="germplasmDbId", roster_artifact_id=units)
    assert r.ok, r.error
    assert r.data["overall"] == {"total_rows": 6, "valid": 4, "raw_missing": 1, "invalid_or_nonfinite": 1}
    assert r.data["expected_units"] == 6 and r.data["denominators"] == {"rows": 6, "expected_units": 6}
    assert r.data["units"] == {"with_valid_value": 4, "present_without_valid_value": 2, "absent_from_table": 0, "rows_with_unit_not_on_roster": 0, "absent_ids": []}
    assert {g["group_id"]: (g["valid"], g["raw_missing"], g["invalid_or_nonfinite"]) for g in r.data["groups"]} == {"GA": (2, 1, 0), "GB": (2, 0, 1)}
    plain = call(registry, "missing_report", artifact_id=obs)
    assert plain.ok and plain.data["expected_units"] is None and "no verified unit roster" in plain.data["expected_units_note"]
    subset = call(registry, "filter_rows", artifact_id=obs, column="germplasmDbId", op="eq", value="GA").data["artifact_id"]
    r = call(registry, "missing_report", artifact_id=subset, roster_artifact_id=units)
    assert r.ok and r.data["units"]["absent_from_table"] == 3 and r.data["units"]["absent_ids"] == ["P4", "P5", "P6"]
    assert call(registry, "missing_report", artifact_id=obs, roster_artifact_id=obs).error.code == "invalid_argument"   # not a unit roster


def test_missing_report_refuses_an_incomplete_or_foreign_roster(registry):
    obs = register(registry, "mixed_units.json", "yield_t_ha")
    unit_columns = dict(srv.UNIT_COLUMNS)
    incomplete = registry.save_table([{"observationUnitDbId": "P1", "studyDbId": "S1"}], kind="observation_units", columns=unit_columns,
                                     complete=False, study_ids=["S1"]).artifact_id
    foreign = registry.save_table([{"observationUnitDbId": "Q1", "studyDbId": "S2"}], kind="observation_units", columns=unit_columns,
                                  complete=True, study_ids=["S2"]).artifact_id
    assert call(registry, "missing_report", artifact_id=obs, roster_artifact_id=incomplete).error.code == "incomplete_data"
    assert call(registry, "missing_report", artifact_id=obs, roster_artifact_id=foreign).error.code == "invalid_argument"


def test_count_records_counts_complete_metadata_exactly(s1):
    registry, _, _, ctx = s1
    studies = srv.dispatch(ctx, "export_metadata", {"entity": "studies"}).artifact_ids[0]
    r = call(registry, "count_records", artifact_id=studies)
    assert r.ok and r.data["total_rows"] == 1 and claims_of(r)["count"].value == 1 and claims_of(r)["count"].denominator.name == "rows in the complete studies table"
    r = call(registry, "count_records", artifact_id=studies, group_by="studyType", distinct_by="locationDbId")
    assert r.ok and r.data["frequencies"] == [{"value": "Synthetic Yield Trial", "count": 1, "rank": 1}] and r.data["distinct_count"] == 1
    variables = srv.dispatch(ctx, "export_metadata", {"entity": "variables"}).artifact_ids[0]
    r = call(registry, "count_records", artifact_id=variables, group_by="units")
    assert r.ok and r.data["frequencies"][0]["count"] == 1 and len(r.data["frequencies"]) == 2 and r.data["ties"] == [["%", "t/ha"]]
    with_empty = registry.save_table([{"studyDbId": "S1", "locationDbId": "L1"}, {"studyDbId": "S2", "locationDbId": ""}, {"studyDbId": "S3", "locationDbId": "L1"}],
                                     kind="studies", columns={"studyDbId": "id", "locationDbId": "id"}, complete=True).artifact_id
    r = call(registry, "count_records", artifact_id=with_empty, group_by="locationDbId", distinct_by="locationDbId")
    assert r.ok and r.data["frequencies"] == [{"value": "L1", "count": 2, "rank": 1}, {"value": at.MISSING_GROUP, "count": 1, "rank": 2}]
    assert r.data["distinct_count"] == 1 and r.data["empty_cells"] == 1 and "never dropped" in r.data["null_group_policy"]


def test_table_info_describes_without_calculating(s1):
    registry, obs, units, _ = s1
    r = call(registry, "table_info", artifact_id=obs)
    assert r.ok and r.data["kind"] == "observations" and r.data["row_count"] == 6 and r.data["complete"] is True
    assert r.data["study_ids"] == ["S1"] and r.data["variable_ids"] == ["V1"] and r.data["measurement"]["trait"] == "fresh root yield"
    assert r.data["value_counts"] == {"total_rows": 6, "valid": 4, "raw_missing": 1, "invalid_or_nonfinite": 1}
    assert r.data["n_independent_units"] == 4 and r.data["duplicates"]["note"] == "none detected"
    assert r.data["preview"]["displayed_rows"] == 5 and r.data["preview"]["total_rows"] == 6 and [x["value"] for x in r.data["preview"]["rows"]] == SIX_VALUES[:5]
    assert "stats" not in r.data and "claims" not in r.data
    u = call(registry, "table_info", artifact_id=units)
    assert u.ok and u.data["kind"] == "observation_units" and "value_counts" not in u.data


def test_plot_bar_is_unavailable_and_installs_nothing(s1):
    registry, obs, _, _ = s1
    r = call(registry, "plot_bar", artifact_id=obs, x="germplasmDbId", y="value")
    assert not r.ok and r.error.code == "unsupported" and "nothing was installed" in r.error.message
    assert r.data["plotting_enabled"] is False and "matplotlib" not in sys.modules and at.PLOTTING_ENABLED is False


# --- 8. precision, schemas, dispatch, async interface -----------------------------------------------

def test_full_precision_is_kept_and_only_display_is_rounded(registry):
    rows = [{"observationDbId": f"O{i}", "observationUnitDbId": f"P{i}", "germplasmDbId": "GA", "studyDbId": "S1", "observationVariableDbId": "V1", "value": v}
            for i, v in enumerate(["1", "2", "2"], start=1)]
    columns = {c: "text" for c in rows[0]}
    art = registry.save_table(rows, kind="observations", columns=columns, complete=True, study_ids=["S1"], variable_ids=["V1"]).artifact_id
    r = call(registry, "numeric_summary", artifact_id=art)
    assert r.ok and r.data["stats"]["mean"] == statistics.fmean([1, 2, 2]) == 5 / 3 and r.data["display"]["mean"] == "1.6667"
    assert claims_of(r)["mean"].value == 5 / 3                                            # the claim keeps full precision
    close = [{"observationDbId": "O1", "observationUnitDbId": "P1", "germplasmDbId": "A", "studyDbId": "S1", "observationVariableDbId": "V1", "value": "1.00004"},
             {"observationDbId": "O2", "observationUnitDbId": "P2", "germplasmDbId": "B", "studyDbId": "S1", "observationVariableDbId": "V1", "value": "1.00003"}]
    art2 = registry.save_table(close, kind="observations", columns=columns, complete=True, study_ids=["S1"], variable_ids=["V1"]).artifact_id
    g = call(registry, "group_stats", artifact_id=art2, by="germplasmDbId", min_independent_n=1)
    assert [(x["group_id"], x["rank"], x["display_mean"]) for x in g.data["shown"]] == [("A", 1, "1"), ("B", 2, "1")]   # sorted at full precision, displayed rounded
    assert g.data["ties"] == []


def test_schemas_and_dispatch_guardrails(registry):
    schemas = at.analyst_tool_schemas()
    assert [s["name"] for s in schemas] == at.ANALYST_TOOL_NAMES and len(schemas) == 9       # rank_entries added (R1, 2026-10-21)
    assert all(s["inputSchema"]["additionalProperties"] is False and s["description"] for s in schemas)
    assert call(registry, "drop_table", artifact_id="art_abcdef_0001").error.code == "invalid_argument"
    assert call(registry, "numeric_summary", artifact_id="art_abcdef_0001", path="x.csv").error.code == "invalid_argument"
    assert call(registry, "numeric_summary", artifact_id="art_abcdef_0001", duplicate_policy="average").error.code == "invalid_argument"
    assert at.dispatch_analyst(registry, "table_info", ["art_abcdef_0001"]).error.code == "invalid_argument"
    assert at.dispatch_analyst(registry, "table_info", None).error.code == "invalid_argument"


def test_async_interface_matches_dispatch(s1):
    registry, obs, _, _ = s1
    tools = at.AnalystTools(registry)

    async def scenario():
        schemas = await tools.schemas()
        result = await tools.call("numeric_summary", {"artifact_id": obs})
        return schemas, result

    schemas, result = anyio.run(scenario)
    assert [s["name"] for s in schemas] == at.ANALYST_TOOL_NAMES and result.ok and result.data["stats"]["mean"] == 18.0 and tools.calls_made == 1


def test_every_claim_from_every_tool_validates_and_names_its_evidence(s1):
    registry, obs, units, ctx = s1
    studies = srv.dispatch(ctx, "export_metadata", {"entity": "studies"}).artifact_ids[0]
    results = [call(registry, "numeric_summary", artifact_id=obs), call(registry, "group_stats", artifact_id=obs, by="germplasmDbId"),
               call(registry, "count_records", artifact_id=studies, group_by="studyType")]
    for r in results:
        assert r.ok
        ids = [c["claim_id"] for c in r.data["claims"]]
        assert len(ids) == len(set(ids))
        for raw in r.data["claims"]:
            claim = Claim.model_validate(raw)
            assert claim.denominator.name and claim.source_artifact_hashes and claim.transformation
            assert all(len(h) == 64 for h in claim.source_artifact_hashes)
