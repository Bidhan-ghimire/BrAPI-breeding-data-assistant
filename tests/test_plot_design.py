"""
tests/test_plot_design.py - R2 tool 2 (coach, 2026-10-22 03:48 UTC, step 6): the roster<->measurements join.

A fetched plot list (observation_units) carries each plot's design only inside a JSON text column, observationUnitPosition:
entryType (check / test), observationLevelRelationships (rep, block, plot with their codes) and the grid coordinates. So the
analyst could not count replicates or plots per replicate (A35.2, A37.1: "no replicate field") or find the check entries
(B29.2, B29.3: "no entryType column"). plot_design writes those fields into columns, on the plot list itself or on a
measurements table of the same study (each row gets its plot's fields). Only what the position states is read; nothing is
inferred from names or numbers. Known small SYNTHETIC tables, plus two position objects copied verbatim from SweetPotatoBase.
"""
from __future__ import annotations

import json

import pytest

import analyst_tools as at
import brapi_mcp_server as srv
from artifacts import ArtifactRegistry
from contracts import Claim, MeasurementMeta
from part2_support import SYNTHETIC_BASE

# unit, germplasm, name, rep, block, plot, entryType, grid column (X), grid row (Y): 2 reps x 2 blocks x 2 plots, G3 the check.
PLOTS = [("P1", "G1", "Alpha", "1", "1", "1", "test", 1, 1), ("P2", "G2", "Beta", "1", "1", "2", "test", 2, 1),
         ("P3", "G3", "Check-A", "1", "2", "3", "check", 3, 1), ("P4", "G4", "Delta", "1", "2", "4", "test", 4, 1),
         ("P5", "G1", "Alpha", "2", "1", "5", "test", 1, 2), ("P6", "G2", "Beta", "2", "1", "6", "test", 2, 2),
         ("P7", "G3", "Check-A", "2", "2", "7", "check", 3, 2), ("P8", "G4", "Delta", "2", "2", "8", "test", 4, 2)]
# Two position objects exactly as SweetPotatoBase returns them (benchmark run 2026-10-21, study 4612's plot list).
REAL = ['{"entryType": "test", "geoCoordinates": null, "observationLevel": {"levelCode": "1", "levelName": "plot", "levelOrder": 2}, '
        '"observationLevelRelationships": [{"levelCode": "1", "levelName": "rep", "levelOrder": 0}, {"levelCode": "1", "levelName": '
        '"block", "levelOrder": 1}, {"levelCode": "1", "levelName": "plot", "levelOrder": 2}], "positionCoordinateX": 1, '
        '"positionCoordinateXType": "GRID_COL", "positionCoordinateY": 1, "positionCoordinateYType": "GRID_ROW"}',
        '{"entryType": "check", "geoCoordinates": null, "observationLevel": {"levelCode": "2", "levelName": "plot", "levelOrder": 2}, '
        '"observationLevelRelationships": [{"levelCode": "1", "levelName": "rep", "levelOrder": 0}, {"levelCode": "1", "levelName": '
        '"block", "levelOrder": 1}, {"levelCode": "2", "levelName": "plot", "levelOrder": 2}], "positionCoordinateX": 2, '
        '"positionCoordinateXType": "GRID_COL", "positionCoordinateY": 1, "positionCoordinateYType": "GRID_ROW"}']


def position(rep=None, block=None, plot=None, entry=None, x=None, y=None, *, x_type="GRID_COL", y_type="GRID_ROW", more=()):
    levels = [{"levelCode": code, "levelName": name, "levelOrder": order}
              for order, (name, code) in enumerate((("rep", rep), ("block", block), ("plot", plot))) if code is not None]
    body = {"entryType": entry, "observationLevelRelationships": [*levels, *more], "positionCoordinateX": x, "positionCoordinateY": y,
            "positionCoordinateXType": x_type if x is not None else None, "positionCoordinateYType": y_type if y is not None else None,
            "observationLevel": {"levelCode": plot, "levelName": "plot", "levelOrder": 2} if plot is not None else None}
    return json.dumps({k: v for k, v in body.items() if v is not None})


def roster(registry, plots=PLOTS, *, study="S1", complete=True, positions=None, studies=None, levels=None, requests=(), drop=()) -> str:
    rows = [{"observationUnitDbId": u, "observationUnitName": f"{study}-plot-{p}", "germplasmDbId": g, "germplasmName": n,
             "studyDbId": study, "levelName": (levels or {}).get(u, "plot"), "levelCode": p,
             "observationUnitPosition": (positions or {}).get(u, position(r, b, p, e, x, y))}
            for u, g, n, r, b, p, e, x, y in plots]
    rows = [{k: v for k, v in row.items() if k not in drop} for row in rows]
    columns = {k: v for k, v in srv.UNIT_COLUMNS.items() if k not in drop}
    return registry.save_table(rows, kind="observation_units", columns=columns, complete=complete, study_ids=studies or [study],
                               variable_ids=[], source_request_ids=list(requests)).artifact_id


def measurements(registry, values, *, study="S1", complete=True, row_studies=None, requests=(), drop=(), extra=None) -> str:
    rows = [{"observationDbId": f"O{u}", "observationUnitDbId": u, "observationUnitName": f"{study}-plot-{u}", "germplasmDbId": g,
             "germplasmName": g, "studyDbId": (row_studies or {}).get(u, study), "observationVariableDbId": "V1",
             "observationVariableName": "SYNTHETIC yield", "observationTimeStamp": "", "value": v, **(extra or {})} for u, g, v in values]
    rows = [{k: v for k, v in row.items() if k not in drop} for row in rows]
    columns = {c: srv.OBSERVATION_COLUMNS.get(c, "text") for c in rows[0]}
    return registry.save_table(rows, kind="observations", columns=columns, complete=complete, study_ids=[study], variable_ids=["V1"],
                               measurement=MeasurementMeta(trait="SYNTHETIC yield", unit="t/ha"), source_request_ids=list(requests)).artifact_id


@pytest.fixture
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "out", "run_design", base_url=SYNTHETIC_BASE)


def design(registry, **args):
    return at.dispatch_analyst(registry, "plot_design", args)


def rows_of(registry, result):
    return registry.load(result.data["artifact_id"])[1]


def count(registry, artifact, **args):
    r = at.dispatch_analyst(registry, "count_records", {"artifact_id": artifact, **args})
    assert r.ok, r.error
    if "group_by" in args:
        return {c["filters"][args["group_by"]]: c["value"] for c in r.data["claims"] if args["group_by"] in (c["filters"] or {})}
    return {"all": next(c["value"] for c in r.data["claims"] if not c["filters"])}


# --- 1. the plot list ---------------------------------------------------------------------------------

def test_the_plot_list_gets_its_design_columns(registry):
    r = design(registry, roster_artifact_id=roster(registry))
    assert r.ok and r.complete and "claims" not in r.data                    # it derives a table; it states no number
    got = {row["observationUnitDbId"]: (row["rep"], row["block"], row["plot"], row["entryType"], row["row"], row["col"]) for row in rows_of(registry, r)}
    assert got == {u: (rep, b, p, e, str(y), str(x)) for u, _, _, rep, b, p, e, x, y in PLOTS}
    manifest = registry.manifest(r.data["artifact_id"])
    assert manifest.kind == "observation_units" and manifest.study_ids == ["S1"] and manifest.complete
    assert any(note.startswith("lineage: plot_design of ") for note in manifest.quality_notes)
    assert r.data["stated"] == {"rep": 8, "block": 8, "plot": 8, "entryType": 8, "row": 8, "col": 8}


def test_count_records_then_answers_replicates_plots_per_replicate_and_checks(registry):
    """A35.2 / A37.1 (replicates, plots per replicate) and B29.2 / B29.3 (check entries), on the plot list with its columns."""
    plots = design(registry, roster_artifact_id=roster(registry)).data["artifact_id"]
    assert count(registry, plots, distinct_by="rep") == {"all": 2}
    assert count(registry, plots, group_by="rep") == {"1": 4, "2": 4}
    checks = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": plots, "column": "entryType", "op": "eq", "value": "check"})
    assert checks.ok and count(registry, checks.data["artifact_id"], distinct_by="germplasmDbId") == {"all": 1}


def test_real_sweetpotatobase_positions_are_read(registry):
    plots = [("U1", "G1", "A", None, None, None, None, None, None), ("U2", "G2", "B", None, None, None, None, None, None)]
    r = design(registry, roster_artifact_id=roster(registry, plots, positions={"U1": REAL[0], "U2": REAL[1]}))
    got = [(row["rep"], row["block"], row["plot"], row["entryType"], row["row"], row["col"]) for row in rows_of(registry, r)]
    assert got == [("1", "1", "1", "test", "1", "1"), ("1", "1", "2", "check", "1", "2")]


# --- 2. only what the position states ----------------------------------------------------------------------

def test_a_field_the_position_does_not_state_stays_empty(registry):
    plots = [("P1", "G1", "A", "1", None, "1", None, None, None), ("P2", "G2", "B", None, None, None, "test", 3, None)]
    r = design(registry, roster_artifact_id=roster(registry, plots))
    got = [(row["rep"], row["block"], row["plot"], row["entryType"], row["row"], row["col"]) for row in rows_of(registry, r)]
    assert got == [("1", "", "1", "", "", ""), ("", "", "", "test", "", "3")]
    assert r.data["stated"] == {"rep": 1, "block": 0, "plot": 1, "entryType": 1, "row": 0, "col": 1}


def test_the_plot_code_of_the_plots_own_level_and_level_names_in_any_case_are_read(registry):
    """A position whose relationships name no plot but whose own observationLevel is the plot; levels written Rep / BLOCK."""
    own_level_only = json.dumps({"observationLevel": {"levelCode": "7", "levelName": "plot", "levelOrder": 2},
                                 "observationLevelRelationships": [{"levelCode": "2", "levelName": "Rep", "levelOrder": 0},
                                                                   {"levelCode": "3", "levelName": "BLOCK", "levelOrder": 1}]})
    plots = [("P1", "G1", "A", None, None, None, None, None, None)]
    row = rows_of(registry, design(registry, roster_artifact_id=roster(registry, plots, positions={"P1": own_level_only})))[0]
    assert (row["rep"], row["block"], row["plot"]) == ("2", "3", "7")


def test_nothing_is_read_from_names_or_from_other_level_names(registry):
    """A unit named like a plot of rep 2, a level called "replicate" and a coordinate of another type state no design field."""
    plots = [("P1", "G1", "A", None, None, None, None, None, None)]
    odd = position(more=[{"levelCode": "2", "levelName": "replicate", "levelOrder": 0}], x=5, y=6, x_type="LONGITUDE", y_type="LATITUDE")
    r = design(registry, roster_artifact_id=roster(registry, plots, positions={"P1": odd}))
    row = rows_of(registry, r)[0]
    assert (row["rep"], row["block"], row["plot"], row["row"], row["col"]) == ("", "", "", "", "")


def test_a_level_stated_twice_or_a_plot_listed_twice_is_refused(registry):
    twice = position("1", "1", "1", "test", 1, 1, more=[{"levelCode": "2", "levelName": "rep", "levelOrder": 0}])
    refused = design(registry, roster_artifact_id=roster(registry, PLOTS[:1], positions={"P1": twice}))
    assert not refused.ok and refused.error.code == "unsupported" and "rep" in refused.error.message and "P1" in refused.error.message
    doubled = design(registry, roster_artifact_id=roster(registry, PLOTS[:2] + PLOTS[:1], study="S2"))
    assert not doubled.ok and "P1" in doubled.error.message and "twice" in doubled.error.message


def test_conflicting_plot_codes_or_grid_axes_are_refused(registry):
    conflicting = json.loads(position("1", "1", "1", "test", 1, 1))
    conflicting["observationLevel"]["levelCode"] = "9"
    refused = design(registry, roster_artifact_id=roster(registry, PLOTS[:1], positions={"P1": json.dumps(conflicting)}))
    assert not refused.ok and refused.error.code == "unsupported" and "plot" in refused.error.message
    two_rows = position("1", "1", "1", "test", 1, 2, x_type="GRID_ROW", y_type="GRID_ROW")
    refused = design(registry, roster_artifact_id=roster(registry, PLOTS[:1], study="S2", positions={"P1": two_rows}))
    assert not refused.ok and refused.error.code == "unsupported" and "GRID_ROW" in refused.error.message


def test_an_unreadable_or_empty_position_leaves_empty_fields_and_is_counted(registry):
    plots = PLOTS[:3]
    r = design(registry, roster_artifact_id=roster(registry, plots, positions={"P1": "{not json", "P2": ""}))
    got = {row["observationUnitDbId"]: row["rep"] for row in rows_of(registry, r)}
    assert got == {"P1": "", "P2": "", "P3": "1"}
    assert r.data["unreadable_positions"] == 1 and r.data["empty_positions"] == 1
    assert any("1 plot position could not be read" in w for w in r.warnings)


# --- 3. measurements get their plot's fields ------------------------------------------------------------------

def test_measurements_get_their_plots_design_columns_and_unknown_plots_are_counted(registry):
    values = [(u, g, str(10 + n)) for n, (u, g, *_rest) in enumerate(PLOTS)] + [("P99", "G1", "30")]
    r = design(registry, roster_artifact_id=roster(registry), artifact_id=measurements(registry, values))
    assert r.ok and r.complete
    joined = {row["observationUnitDbId"]: (row["rep"], row["entryType"], row["value"]) for row in rows_of(registry, r)}
    assert joined["P1"] == ("1", "test", "10") and joined["P7"] == ("2", "check", "16") and joined["P99"] == ("", "", "30")
    assert r.data["rows_without_a_plot_in_the_roster"] == 1 and any("1 measurement row" in w for w in r.warnings)
    manifest = registry.manifest(r.data["artifact_id"])
    assert manifest.kind == "observations" and manifest.variable_ids == ["V1"] and manifest.measurement.unit == "t/ha"
    stats = at.dispatch_analyst(registry, "group_stats", {"artifact_id": r.data["artifact_id"], "by": "rep"})
    means = {Claim.model_validate(c).filters.get("rep"): Claim.model_validate(c).value for c in stats.data["claims"]
             if Claim.model_validate(c).kind == "mean"}
    assert means["1"] == (10 + 11 + 12 + 13) / 4 and means["2"] == (14 + 15 + 16 + 17) / 4


def test_another_studys_roster_or_a_roster_of_several_studies_is_refused(registry):
    values = [(u, g, "10") for u, g, *_rest in PLOTS]
    other = design(registry, roster_artifact_id=roster(registry, study="S2"), artifact_id=measurements(registry, values))
    assert not other.ok and other.error.code == "invalid_argument" and "S2" in other.error.message
    several = design(registry, roster_artifact_id=roster(registry, study="S3", studies=["S1", "S3"]), artifact_id=measurements(registry, values))
    assert not several.ok and several.error.code == "invalid_argument"
    alone = design(registry, roster_artifact_id=roster(registry, study="S4", studies=["S4", "S5"]))      # counts would mix studies
    assert not alone.ok and alone.error.code == "invalid_argument" and "one study" in alone.error.message


def test_an_incomplete_roster_or_table_is_refused_unless_allowed_and_then_marked(registry):
    partial = roster(registry, complete=False)
    refused = design(registry, roster_artifact_id=partial)
    assert not refused.ok and refused.error.code == "incomplete_data"
    allowed = design(registry, roster_artifact_id=partial, allow_incomplete=True)
    assert allowed.ok and allowed.complete is False and not registry.manifest(allowed.data["artifact_id"]).complete
    assert any("INCOMPLETE" in w for w in allowed.warnings)
    values = [(u, g, "10") for u, g, *_rest in PLOTS]
    half = design(registry, roster_artifact_id=roster(registry, study="S1"), artifact_id=measurements(registry, values, complete=False))
    assert not half.ok and half.error.code == "incomplete_data"


def test_the_roster_must_be_a_plot_list_without_design_columns_and_the_table_measurements(registry):
    values = [(u, g, "10") for u, g, *_rest in PLOTS]
    table = measurements(registry, values)
    not_plots = design(registry, roster_artifact_id=table)
    assert not not_plots.ok and not_plots.error.code == "invalid_argument" and "observation_units" in not_plots.error.message
    done = design(registry, roster_artifact_id=roster(registry)).data["artifact_id"]
    again = design(registry, roster_artifact_id=done)
    assert not again.ok and again.error.code == "invalid_argument" and "already" in again.error.message
    plots = roster(registry, study="S1")
    not_measurements = design(registry, roster_artifact_id=plots, artifact_id=plots)
    assert not not_measurements.ok and not_measurements.error.code == "invalid_argument"


# --- 4. the wiring ---------------------------------------------------------------------------------------------

def test_it_is_one_of_the_analysts_tools_and_the_planner_knows_it():
    import agents.analyst as an
    import agents.coordinator as co

    assert "plot_design" in at.ANALYST_TOOL_NAMES and len(at.ANALYST_TOOL_NAMES) == 16
    assert "plot_design" in an.DERIVING_TOOLS and "plot_design" not in an.NUMERIC_TOOLS
    schema = next(s for s in at.analyst_tool_schemas() if s["name"] == "plot_design")
    assert schema["inputSchema"]["required"] == ["roster_artifact_id"] and "replicate" in schema["description"].lower()
    prompt = " ".join(co.COORDINATOR_SYSTEM_PROMPT.split())
    assert "plot_design" in co.ANALYST_ACTIONS and "get_observation_units, then plot_design(roster_artifact_id=" in prompt
    analyst = " ".join(an.ANALYST_SYSTEM_PROMPT.split())
    assert "sixteen fixed tools" in analyst and "plot_design" in analyst


def test_the_analysts_guard_refuses_a_plot_list_never_supplied(registry):
    import anyio

    import agents.analyst as an

    supplied, stray = roster(registry), roster(registry, study="S2")
    tools = an.GuardedAnalystTools(at.AnalystTools(registry), [supplied], inspected={supplied})
    refused = anyio.run(tools.call, "plot_design", {"roster_artifact_id": stray})
    assert not refused.ok and refused.error.code == "not_authorized" and stray in refused.error.message
    assert anyio.run(tools.call, "plot_design", {"roster_artifact_id": supplied}).ok


# --- 5. after the review of 2026-10-22 (round 1) ----------------------------------------------------------------

def one_plot(registry, text, *, study="S9"):
    plots = [("P1", "G1", "A", None, None, None, None, None, None)]
    return design(registry, roster_artifact_id=roster(registry, plots, study=study, positions={"P1": text}))


@pytest.mark.parametrize("text", ["[" * 30000 + "]" * 30000, "[]", '"text"', '{"entryType": "test", "entryType": "check"}',
                                  '{"observationLevelRelationships": ' + "[" * 5000 + "]" * 5000 + "}"],
                         ids=["deep", "list", "string", "doubled-key", "deep-inside"])
def test_a_deeply_nested_odd_or_doubled_key_position_counts_as_unreadable(registry, text):
    """Review: one position nested about 1,000 levels deep ended the whole call (RecursionError); a key written twice in one
    position let the last value win silently."""
    r = one_plot(registry, text)
    assert r.ok and r.data["unreadable_positions"] == 1 and rows_of(registry, r)[0]["entryType"] == ""


def test_codes_are_kept_as_the_position_writes_them(registry):
    """Review: 1e2 came out as "100.0"; NaN as "NaN". Numbers stay as written, a NaN or a null states nothing (a true/false makes
    the position unreadable since round 4), codes are trimmed, a code 0 is a code, entryType is written in lower case (round 3),
    and a grid type is read in any case."""
    text = ('{"observationLevelRelationships": [{"levelName": "rep", "levelCode": 1e2}, {"levelName": "block", "levelCode": 1.0}, '
            '{"levelName": "plot", "levelCode": 0}, {"levelName": "rep", "levelCode": 1e2}], "entryType": " Check ", '
            '"positionCoordinateX": NaN, "positionCoordinateXType": "grid_col", "positionCoordinateY": null, "positionCoordinateYType": "Grid_Row"}')
    # round 4 (B): a relation that is not an object, or a true/false where a code is read, makes the position unreadable instead
    row = rows_of(registry, one_plot(registry, text))[0]
    assert (row["rep"], row["block"], row["plot"], row["entryType"], row["row"], row["col"]) == ("1e2", "1.0", "0", "check", "", "")
    spaced = '{"observationLevelRelationships": [{"levelName": "rep", "levelCode": " 2 "}], "positionCoordinateX": 3, "positionCoordinateXType": "grid_col"}'
    row = rows_of(registry, one_plot(registry, spaced, study="S8"))[0]
    assert (row["rep"], row["col"]) == ("2", "3")


def test_a_level_stated_twice_with_one_code_is_one_level_and_any_level_with_two_codes_is_refused(registry):
    same = position("1", "1", "1", "test", 1, 1, more=[{"levelCode": "1", "levelName": "rep", "levelOrder": 0}])
    assert rows_of(registry, one_plot(registry, same))[0]["rep"] == "1"
    two_blocks = position("1", "1", "1", "test", 1, 1, more=[{"levelCode": "2", "levelName": "block", "levelOrder": 1}])
    refused = one_plot(registry, two_blocks, study="S8")
    assert not refused.ok and refused.error.code == "unsupported" and "block" in refused.error.message


def test_units_that_are_not_plots_are_refused(registry):
    """Review: a plant-level unit list was counted as plots ("plots per rep = 3" for 3 plants in one plot)."""
    plants = design(registry, roster_artifact_id=roster(registry, levels={"P1": "plant", "P2": "plant"}))
    assert not plants.ok and plants.error.code == "unsupported" and "plant" in plants.error.message
    unstated = design(registry, roster_artifact_id=roster(registry, study="S2", levels={"P1": "", "P2": "PLOT"}))
    assert unstated.ok


def test_measurement_rows_of_another_study_are_refused(registry):
    """Review: only the tables' manifests were compared, never the rows' own studyDbId."""
    values = [(u, g, "10") for u, g, *_rest in PLOTS]
    refused = design(registry, roster_artifact_id=roster(registry), artifact_id=measurements(registry, values, row_studies={"P3": "S2"}))
    assert not refused.ok and refused.error.code == "invalid_argument" and "S2" in refused.error.message
    unstated = measurements(registry, values, study="S2", drop=("studyDbId",))        # rows that name no study: the tables' own
    other = design(registry, roster_artifact_id=roster(registry, study="S1"), artifact_id=unstated)
    assert not other.ok and other.error.code == "invalid_argument" and "S2" in other.error.message


def test_the_join_keeps_the_order_and_exact_ids_and_counts_every_row_without_a_plot(registry):
    values = [("P3", "G3", "13"), ("P1", "G1", "11"), ("P1 ", "G1", "12"), ("", "G2", "14"), ("P2", "G2", "15")]
    r = design(registry, roster_artifact_id=roster(registry), artifact_id=measurements(registry, values))
    rows = rows_of(registry, r)
    assert [(row["observationUnitDbId"], row["rep"]) for row in rows] == [("P3", "1"), ("P1", "1"), ("P1 ", ""), ("", ""), ("P2", "1")]
    assert r.data["rows_without_a_plot_in_the_roster"] == 2 and r.data["stated"]["rep"] == 3 and r.data["rows"] == 5


def test_the_lineage_and_the_result_name_every_source(registry):
    plots = roster(registry, requests=["req_roster"])
    table = measurements(registry, [(u, g, "10") for u, g, *_rest in PLOTS], requests=["req_values"])
    r = design(registry, roster_artifact_id=plots, artifact_id=table)
    manifest = registry.manifest(r.data["artifact_id"])
    assert r.artifact_ids == [plots, table, r.data["artifact_id"]]
    assert r.data["roster_artifact_id"] == plots and r.data["source_artifact_id"] == table
    assert manifest.source_request_ids == ["req_roster", "req_values"]
    lineage = manifest.quality_notes[0]
    assert registry.manifest(plots).sha256 in lineage and registry.manifest(table).sha256 in lineage


def test_the_columns_it_needs_and_the_columns_it_must_not_find(registry):
    no_position = design(registry, roster_artifact_id=roster(registry, drop=("observationUnitPosition",)))
    assert not no_position.ok and no_position.error.code == "invalid_argument" and "observationUnitPosition" in no_position.error.message
    values = [(u, g, "10") for u, g, *_rest in PLOTS]
    no_unit = design(registry, roster_artifact_id=roster(registry, study="S1"), artifact_id=measurements(registry, values, drop=("observationUnitDbId",)))
    assert not no_unit.ok and no_unit.error.code == "invalid_argument" and "observationUnitDbId" in no_unit.error.message
    has_rep = design(registry, roster_artifact_id=roster(registry, study="S1"), artifact_id=measurements(registry, values, extra={"rep": "9"}))
    assert not has_rep.ok and "already" in has_rep.error.message


def test_incomplete_measurements_allowed_make_the_result_incomplete(registry):
    values = [(u, g, "10") for u, g, *_rest in PLOTS]
    r = design(registry, roster_artifact_id=roster(registry), artifact_id=measurements(registry, values, complete=False), allow_incomplete=True)
    assert r.ok and r.complete is False and not registry.manifest(r.data["artifact_id"]).complete
    assert any("INCOMPLETE" in w for w in r.warnings)


def test_a_plot_without_a_position_is_warned(registry):
    r = design(registry, roster_artifact_id=roster(registry, PLOTS[:2], positions={"P1": ""}))
    assert r.data["empty_positions"] == 1 and any("1 plot has no position object" in w for w in r.warnings)


# --- 6. round 3 (coach 2026-10-22 06:52 UTC: points 1-3) --------------------------------------------------------

def test_entry_types_are_written_in_lower_case_so_one_filter_finds_every_check(registry):
    """Review 2: the BrAPI standard writes CHECK; SweetPotatoBase writes check. Kept as written, filter_rows eq "check" found
    0 checks on a standard server and half of them on a mixed one. plot_design writes entryType in lower case."""
    cases = {"P1": "CHECK", "P2": "Check", "P3": "check", "P4": "TEST"}
    positions = {u: position("Rep-A", "1", "Plot-A" + u[1:], e, "Col-B", "Row-C") for u, e in cases.items()}
    plots = [(u, f"G{u[1:]}", f"N{u[1:]}", None, None, None, None, None, None) for u in cases]
    r = design(registry, roster_artifact_id=roster(registry, plots, positions=positions))
    written = rows_of(registry, r)
    assert [row["entryType"] for row in written] == ["check", "check", "check", "test"]
    # entryType only: the codes and coordinates stay as written
    assert {(row["rep"], row["plot"][:6], row["col"], row["row"]) for row in written} == {("Rep-A", "Plot-A", "Col-B", "Row-C")}
    checks = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": r.data["artifact_id"], "column": "entryType", "op": "eq", "value": "check"})
    assert checks.data["rows_out"] == 3
    schema = next(s for s in at.analyst_tool_schemas() if s["name"] == "plot_design")
    assert "lower case" in at.DESIGN_COLUMNS["entryType"] and "lower case" in schema["description"]
    import agents.coordinator as co
    assert "plot_design writes entryType in lower case" in " ".join(co.COORDINATOR_SYSTEM_PROMPT.split())


@pytest.mark.parametrize("text", ['{"entryType": "\\ud800", "observationLevelRelationships": [{"levelName": "rep", "levelCode": "1"}]}',
                                  '{"observationLevelRelationships": [{"levelName": "rep", "levelCode": "1"}, '
                                  '{"levelName": "rep", "levelCode": "\\udfff"}]}',
                                  '{"observationLevel": {"levelName": "plot", "levelCode": "\\ud800"}}',
                                  '{"positionCoordinateX": "\\udc00", "positionCoordinateXType": "GRID_ROW"}'],
                         ids=["entryType", "levelCode", "own-plot-code", "grid-coordinate"])
def test_a_code_that_cannot_be_saved_counts_as_unreadable(registry, text):
    """Review 2: a lone surrogate escape (\\ud800) in a code ended the whole call when the table was saved. Read code by code,
    so a second code that cannot be saved is unreadable too, not a level with two codes."""
    assert "\\u" in text and text.isascii()                                 # the escape itself, so the plot list saves
    r = one_plot(registry, text)
    assert r.ok and r.data["unreadable_positions"] == 1 and any("1 plot position could not be read" in w for w in r.warnings)
    assert all(rows_of(registry, r)[0][column] == "" for column in at.DESIGN_COLUMNS)


# One test per bug that survived review 2 (its N numbers).

def test_n1_a_row_without_a_study_is_not_of_another_study(registry):
    values = [(u, g, "10") for u, g, *_rest in PLOTS]
    r = design(registry, roster_artifact_id=roster(registry), artifact_id=measurements(registry, values, row_studies={"P2": ""}))
    assert r.ok


def test_n4_every_unit_of_the_plot_list_must_be_a_plot(registry):
    refused = design(registry, roster_artifact_id=roster(registry, levels={"P8": "plant"}))
    assert not refused.ok and "plant" in refused.error.message


def test_n5_a_level_first_unstated_then_stated_has_two_codes(registry):
    text = '{"observationLevelRelationships": [{"levelName": "rep", "levelCode": null}, {"levelName": "rep", "levelCode": "1"}]}'
    refused = one_plot(registry, text)
    assert not refused.ok and refused.error.code == "unsupported" and "rep" in refused.error.message


def test_n7_a_key_written_twice_inside_a_level_is_unreadable(registry):
    text = '{"observationLevelRelationships": [{"levelName": "rep", "levelCode": "1", "levelCode": "2"}]}'
    r = one_plot(registry, text)
    assert r.ok and r.data["unreadable_positions"] == 1 and rows_of(registry, r)[0]["rep"] == ""


def test_n8_every_measurement_row_without_a_plot_is_counted(registry):
    values = [(u, g, "10") for u, g, *_rest in PLOTS[:2]] + [("P99", "G9", "1"), ("P99", "G9", "2")]
    r = design(registry, roster_artifact_id=roster(registry), artifact_id=measurements(registry, values))
    assert r.data["rows_without_a_plot_in_the_roster"] == 2


def test_n10_a_grid_type_must_be_exactly_grid_row_or_grid_col(registry):
    text = '{"positionCoordinateX": 3, "positionCoordinateXType": "SUBGRID_COL", "positionCoordinateY": 4, "positionCoordinateYType": "NOT_GRID_ROW"}'
    row = rows_of(registry, one_plot(registry, text))[0]
    assert (row["row"], row["col"]) == ("", "")


def test_n11_the_plot_code_of_the_plots_own_level_is_trimmed(registry):
    text = '{"observationLevel": {"levelName": "plot", "levelCode": " 7 "}}'
    assert rows_of(registry, one_plot(registry, text))[0]["plot"] == "7"


def test_n13_a_blank_position_is_empty_not_unreadable(registry):
    r = one_plot(registry, "   ")
    assert (r.data["empty_positions"], r.data["unreadable_positions"]) == (1, 0)


def test_n18_a_plot_listed_twice_is_refused_even_with_two_positions(registry):
    twice = [PLOTS[0], ("P1", "G1", "Alpha", "2", "2", "9", "test", 9, 9)]
    refused = design(registry, roster_artifact_id=roster(registry, twice))
    assert not refused.ok and "P1" in refused.error.message and "twice" in refused.error.message


def test_n20_a_plot_list_row_of_another_study_is_refused(registry):
    plots = roster(registry)
    manifest, rows = registry.load(plots)
    rows[3]["studyDbId"] = "S2"
    mixed = registry.save_table(rows, kind="observation_units", columns=dict(manifest.columns), complete=True, study_ids=["S1"],
                                variable_ids=[]).artifact_id
    refused = design(registry, roster_artifact_id=mixed)
    assert not refused.ok and refused.error.code == "invalid_argument" and "S2" in refused.error.message


def test_n25_only_the_level_names_rep_block_and_plot_count(registry):
    text = ('{"observationLevelRelationships": [{"levelName": "replicate", "levelCode": "1"}, {"levelName": "replicate", "levelCode": "2"}, '
            '{"levelName": "plots", "levelCode": "5"}]}')
    r = one_plot(registry, text)
    row = rows_of(registry, r)[0]
    assert r.ok and (row["rep"], row["plot"]) == ("", "") and "replicate" not in row


def test_n29_plot_ids_differing_only_in_case_are_two_plots(registry):
    plots = [("P1", "G1", "A", "1", "1", "1", "test", 1, 1), ("p1", "G2", "B", "2", "1", "2", "test", 2, 1)]
    assert design(registry, roster_artifact_id=roster(registry, plots)).ok


def test_n30_a_measurement_joins_only_the_plot_id_written_exactly_alike(registry):
    plots = [("P1", "G1", "A", "1", "1", "1", "test", 1, 1)]
    joined = design(registry, roster_artifact_id=roster(registry, plots, study="S2"),
                    artifact_id=measurements(registry, [("p1", "G1", "10")], study="S2"))
    assert joined.data["rows_without_a_plot_in_the_roster"] == 1 and rows_of(registry, joined)[0]["rep"] == ""


# --- 7. round 4 (DECISION (Bidhan) 2026-10-22 12:40 UTC, item 10: points A-F of the round-3 stop) -------------------

def test_a_an_entry_type_that_is_not_check_test_or_filler_is_kept_and_warned(registry):
    """Review 3: an entryType with an invisible character (a zero-width space before "check") was kept quietly, so a filter on
    "check" missed that plot with no sign. The value stays (trimmed, case-folded) and the call names it as written."""
    cases = {"P1": "\u200bcheck", "P2": "CHECK", "P3": "Filler", "P4": "chk"}
    positions = {u: position("1", "1", u[1:], e, 1, int(u[1:])) for u, e in cases.items()}
    plots = [(u, f"G{u[1:]}", f"N{u[1:]}", None, None, None, None, None, None) for u in cases]
    r = design(registry, roster_artifact_id=roster(registry, plots, positions=positions))
    assert r.ok and [row["entryType"] for row in rows_of(registry, r)] == ["\u200bcheck", "check", "filler", "chk"]
    assert r.data["entry_types_not_standard"] == {"\u200bcheck": 1, "chk": 1}
    assert any("2 plots have an entryType that is not check, test or filler" in w and "\\u200bcheck" in w and "'chk'" in w for w in r.warnings)


@pytest.mark.parametrize("text", ['{"entryType": true}', '{"entryType": ["check"]}',
                                  '{"observationLevelRelationships": [{"levelName": "rep", "levelCode": [1]}]}',
                                  '{"observationLevelRelationships": {"levelName": "rep", "levelCode": "1"}}',
                                  '{"observationLevelRelationships": ["rep"]}', '{"observationLevel": "plot"}',
                                  '{"positionCoordinateX": {"v": 3}, "positionCoordinateXType": "GRID_COL"}',
                                  '{"positionCoordinateX": 3, "positionCoordinateXType": false}',
                                  '{"observationLevelRelationships": [{"levelName": true, "levelCode": "1"}]}'],
                         ids=["entryType-bool", "entryType-list", "code-list", "relations-object", "relation-string", "level-string",
                              "coordinate-object", "type-bool", "name-bool"])
def test_b_a_value_of_the_wrong_json_type_makes_the_position_unreadable(registry, text):
    """Review 3: true, a list or an object where a code, an entryType or a coordinate is read came out blank with no warning."""
    r = one_plot(registry, text)
    assert r.ok and r.data["unreadable_positions"] == 1 and any("1 plot position could not be read" in w for w in r.warnings)


def test_b_null_still_means_not_stated(registry):
    r = one_plot(registry, '{"entryType": null, "observationLevelRelationships": [{"levelName": "rep", "levelCode": null}], "observationLevel": null}')
    assert r.ok and r.data["unreadable_positions"] == 0 and r.data["empty_positions"] == 0 and rows_of(registry, r)[0]["rep"] == ""


def test_c_a_design_value_longer_than_1000_characters_is_unreadable(registry):
    """Review 3: a 70,000-character entryType made the saved table too big to load back (internal_error)."""
    too_long = one_plot(registry, json.dumps({"observationLevelRelationships": [{"levelName": "rep", "levelCode": "x" * 1001}]}))
    assert too_long.ok and too_long.data["unreadable_positions"] == 1
    fine = one_plot(registry, json.dumps({"observationLevelRelationships": [{"levelName": "rep", "levelCode": "x" * 1000}]}), study="S8")
    assert fine.ok and fine.data["unreadable_positions"] == 0 and rows_of(registry, fine)[0]["rep"] == "x" * 1000
    folded = one_plot(registry, json.dumps({"entryType": "ß" * 501}), study="S7")             # 501 sharp s fold to 1,002 characters
    assert folded.ok and folded.data["unreadable_positions"] == 1


def test_d_plot_list_rows_without_a_plot_id_are_counted_and_warned(registry):
    plots = [*PLOTS[:2], ("", "G9", "NoId", "1", "1", "9", "test", 9, 9)]
    r = design(registry, roster_artifact_id=roster(registry, plots))
    assert r.ok and r.data["rows_without_a_plot_id"] == 1 and any("1 plot-list row has no plot ID" in w for w in r.warnings)
    assert rows_of(registry, r)[2]["rep"] == "1"                                                      # its fields are still read


def test_e_whitespace_around_an_entry_type_is_trimmed_and_sharp_s_is_case_folded(registry):
    r = one_plot(registry, json.dumps({"entryType": " CHECK\t"}))
    assert rows_of(registry, r)[0]["entryType"] == "check" and r.data["entry_types_not_standard"] == {}
    folded = one_plot(registry, json.dumps({"entryType": "STRAßE"}), study="S8")
    assert rows_of(registry, folded)[0]["entryType"] == "strasse" and folded.data["entry_types_not_standard"] == {"strasse": 1}   # round 5: the stored value


def test_e_the_planners_example_filters_on_check_in_lower_case():
    import agents.coordinator as co
    assert 'filter_rows(column="entryType", op="eq", value="check")' in " ".join(co.COORDINATOR_SYSTEM_PROMPT.split())


def test_e_grid_types_and_level_names_match_whole_not_by_prefix_or_suffix(registry):
    text = ('{"observationLevelRelationships": [{"levelName": "subplot", "levelCode": "5"}, {"levelName": "rep", "levelCode": "2"}], '
            '"positionCoordinateX": 3, "positionCoordinateXType": "GRID_ROW_X", "positionCoordinateY": 4, "positionCoordinateYType": "GRID_COL"}')
    row = rows_of(registry, one_plot(registry, text))[0]
    assert (row["plot"], row["rep"], row["row"], row["col"]) == ("", "2", "", "4")


def test_e_a_plot_list_row_without_a_study_is_not_of_another_study(registry):
    plots = roster(registry)
    manifest, rows = registry.load(plots)
    rows[1]["studyDbId"] = ""
    blank = registry.save_table(rows, kind="observation_units", columns=dict(manifest.columns), complete=True, study_ids=["S1"],
                                variable_ids=[]).artifact_id
    assert design(registry, roster_artifact_id=blank).ok


def test_e_the_check_entries_are_counted_as_distinct_entries(registry):
    """The planner's steps end to end: plot_design, filter_rows eq check, count_records distinct_by germplasmDbId. Two check plots
    of one entry are one check entry."""
    plots = [("P1", "G1", "Chk", None, None, None, None, None, None), ("P2", "G1", "Chk", None, None, None, None, None, None),
             ("P3", "G3", "Chk-B", None, None, None, None, None, None), ("P4", "G4", "Test", None, None, None, None, None, None)]
    positions = {"P1": position("1", "1", "1", "CHECK", 1, 1), "P2": position("2", "1", "2", "Check", 1, 2), "P3": position("1", "1", "3", "check", 1, 3),
                 "P4": position("1", "1", "4", "TEST", 1, 4)}
    r = design(registry, roster_artifact_id=roster(registry, plots, positions=positions))
    checks = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": r.data["artifact_id"], "column": "entryType", "op": "eq", "value": "check"})
    distinct = at.dispatch_analyst(registry, "count_records", {"artifact_id": checks.data["artifact_id"], "distinct_by": "germplasmDbId"})
    assert checks.data["rows_out"] == 3 and distinct.data["distinct_count"] == 2


def test_f_the_texts_say_case_folded_and_what_makes_a_position_unreadable():
    assert "case-folded" in at.DESIGN_COLUMNS["entryType"]
    schema = next(s for s in at.analyst_tool_schemas() if s["name"] == "plot_design")
    assert "wrong JSON type" in schema["description"] and "1,000" in schema["description"] and "not check, test or filler" in schema["description"]
    assert "kept as written" not in at._stated_fields.__doc__


# --- 8. round 5 (review 4 of 2026-10-22: residuals F1-F4, the blank-ID join, the test gaps, the texts) ---------------

def test_r5_codes_equal_as_numbers_but_written_differently_are_counted_apart_and_named(registry):
    """F1: a rep written 1, 1.0 and 01 is three replicates to count_records; the codes stay as written, and the call says so."""
    positions = {"P1": position("1", "1", "1", "test", 1, 1), "P2": position("1.0", "1", "2", "test", 1, 2),
                 "P3": position("01", "1", "3", "test", 1, 3), "P4": position("2", "1", "4", "test", 1, 4)}
    plots = [(u, f"G{u[1:]}", f"N{u[1:]}", None, None, None, None, None, None) for u in positions]
    r = design(registry, roster_artifact_id=roster(registry, plots, positions=positions))
    assert r.ok and [row["rep"] for row in rows_of(registry, r)] == ["1", "1.0", "01", "2"]
    assert r.data["codes_equal_as_numbers"] == {"rep": [["01", "1", "1.0"]]}
    assert any("rep codes written differently but equal as numbers: '01', '1', '1.0'" in w and "counted apart" in w for w in r.warnings)
    assert count(registry, r.data["artifact_id"], distinct_by="rep") == {"all": 4}


def test_r5_level_names_and_coordinate_types_not_read_are_counted_and_named(registry):
    """F2: a level name or a grid type with an invisible character ("rep\u200b", "GRID_COL\u200b") left the field blank with no
    sign; the call now names what it did not read, as written (ascii), so the invisible character shows."""
    text = ('{"observationLevelRelationships": [{"levelName": "rep\u200b", "levelCode": "2"}, {"levelName": "plant", "levelCode": "7"}, '
            '{"levelName": "block", "levelCode": "3"}], "positionCoordinateX": 4, "positionCoordinateXType": "GRID_COL\u200b", '
            '"positionCoordinateY": 5, "positionCoordinateYType": "GRID_ROW"}')
    r = one_plot(registry, text)
    row = rows_of(registry, r)[0]
    assert (row["rep"], row["block"], row["col"], row["row"]) == ("", "3", "", "5")
    assert r.data["level_names_not_read"] == {"plant": 1, "rep\u200b": 1} and r.data["coordinate_types_not_read"] == {"GRID_COL\u200b": 1}
    assert any("level names the tool does not read" in w and "'rep\\u200b'" in w and "'plant'" in w for w in r.warnings)
    assert any("coordinate types the tool does not read" in w and "'GRID_COL\\u200b'" in w for w in r.warnings)
    clean = one_plot(registry, json.dumps({"observationLevelRelationships": [{"levelName": "rep", "levelCode": "2"}]}), study="S8")
    assert clean.data["level_names_not_read"] == {} and clean.data["coordinate_types_not_read"] == {} and len(clean.warnings) == 0


def test_r5_a_blank_or_invisible_plot_id_is_no_plot_id_and_never_joins(registry):
    """F3 + the reviewer's Q12: a plot ID that is only spaces or a zero-width space is no ID; a measurement row with a blank ID
    takes no plot's fields and counts as a row without a plot in the plot list."""
    plots = [*PLOTS[:2], ("   ", "G9", "Blank", "2", "1", "9", "test", 9, 9), ("\u200b", "G8", "Zw", "3", "1", "8", "test", 8, 8)]
    r = design(registry, roster_artifact_id=roster(registry, plots))
    assert r.ok and r.data["rows_without_a_plot_id"] == 2 and any("2 plot-list rows have no plot ID" in w for w in r.warnings)
    joined = design(registry, roster_artifact_id=roster(registry, plots),
                    artifact_id=measurements(registry, [("P1", "G1", "5"), ("   ", "G9", "6"), ("", "G7", "7")]))
    assert {row["value"]: row["rep"] for row in rows_of(registry, joined)} == {"5": "1", "6": "", "7": ""}
    assert joined.data["rows_without_a_plot_in_the_roster"] == 2


def test_r5_odd_entry_types_are_keyed_by_the_stored_value_and_counted_per_plot(registry):
    """Review 4 (Q6): "Chk" and "CHK" were listed apart although both are stored as "chk"; the result and the warning now name
    the stored value (trimmed, case-folded), which is what filter_rows sees. Invisible characters still show through ascii()."""
    positions = {"P1": position("1", "1", "1", "chk", 1, 1), "P2": position("1", "1", "2", " Chk ", 1, 2), "P3": position("1", "1", "3", "CHK", 1, 3),
                 "P4": position("1", "1", "4", "\u200bcheck", 1, 4)}
    plots = [(u, f"G{u[1:]}", f"N{u[1:]}", None, None, None, None, None, None) for u in positions]
    r = design(registry, roster_artifact_id=roster(registry, plots, positions=positions))
    assert r.data["entry_types_not_standard"] == {"\u200bcheck": 1, "chk": 3} and r.data["entry_types_not_standard_distinct"] == 2
    w = next(w for w in r.warnings if "not check, test or filler" in w)
    assert "4 plots have" in w and "'chk'" in w and "'\\u200bcheck'" in w and "kept as written" not in w and "case-folded" in w


def test_r5_many_odd_entry_types_are_capped_in_the_result_and_shortened_in_the_warning(registry):
    """F4: 30 distinct 996-character entry types made a 2 MB result and a 50 KB warning. The result keeps at most 20 values and
    says how many there were; the warning shows at most 5, each cut at 60 characters."""
    kinds = [f"kind{n:02d}" + "x" * 990 for n in range(30)]
    positions = {f"P{n}": position("1", "1", str(n), kinds[n], 1, n) for n in range(30)}
    plots = [(u, f"G{u[1:]}", f"N{u[1:]}", None, None, None, None, None, None) for u in positions]
    r = design(registry, roster_artifact_id=roster(registry, plots, positions=positions))
    odd = r.data["entry_types_not_standard"]
    assert len(odd) == 20 and r.data["entry_types_not_standard_distinct"] == 30 and all(v == 1 for v in odd.values())
    w = next(w for w in r.warnings if "not check, test or filler" in w)
    assert "30 plots have" in w and w.count("kind") == 5 and "and 25 more" in w and "…" in w and len(w) < 600


def test_r5_the_length_limit_applies_to_coordinates_and_the_plots_own_code_after_trimming(registry):
    long = "x" * 1001
    assert one_plot(registry, json.dumps({"positionCoordinateX": long, "positionCoordinateXType": "GRID_COL"})).data["unreadable_positions"] == 1
    assert one_plot(registry, json.dumps({"observationLevel": {"levelName": "plot", "levelCode": long}}), study="S8").data["unreadable_positions"] == 1
    padded = one_plot(registry, json.dumps({"observationLevelRelationships": [{"levelName": "rep", "levelCode": "  " + "x" * 1000 + "  "}]}), study="S7")
    assert padded.data["unreadable_positions"] == 0 and rows_of(registry, padded)[0]["rep"] == "x" * 1000


@pytest.mark.parametrize("text, unreadable", [('{"observationLevelRelationships": [null]}', 1), ('{"observationLevel": ""}', 1),
                                              ('{"observationLevel": 0}', 1), ('{"observationLevelRelationships": ""}', 1),
                                              ('{"observationLevelRelationships": []}', 0),
                                              ('{"positionCoordinateX": {"v": 1}, "positionCoordinateXType": "LATITUDE"}', 0)],
                         ids=["null-relation", "level-empty-string", "level-zero", "relations-empty-string", "relations-empty-list",
                              "object-under-a-type-not-read"])
def test_r5_falsy_and_unused_values_are_pinned(registry, text, unreadable):
    r = one_plot(registry, text)
    assert r.ok and r.data["unreadable_positions"] == unreadable
    if "LATITUDE" in text:
        assert r.data["coordinate_types_not_read"] == {"LATITUDE": 1}


def test_r5_texts_name_what_is_stored_and_what_states_nothing():
    schema = next(s for s in at.analyst_tool_schemas() if s["name"] == "plot_design")
    assert "null" in schema["description"] and "NaN" in schema["description"] and "JSON text" in schema["description"]
    assert "other than check, test or filler" in at.DESIGN_COLUMNS["entryType"] and "kept as written" not in at.tool_plot_design.__doc__


# --- 9. round 5b (review 5 of 2026-10-22: guards and pins for its made-up findings; rule (11): no further review round) -------

def plots_with(registry, positions, **kw):
    plots = [(u, f"G{u[1:]}", f"N{u[1:]}", None, None, None, None, None, None) for u in positions]
    return design(registry, roster_artifact_id=roster(registry, plots, positions=positions, **kw))


def test_r5b_a_lone_surrogate_in_a_level_name_or_type_never_breaks_the_result(registry):
    """Review 5: a lone surrogate became a raw key of data["level_names_not_read"], and ToolResult.model_dump_json() raised."""
    text = ('{"observationLevelRelationships": [{"levelName": "rep\\ud800", "levelCode": "1"}], '
            '"positionCoordinateX": 1, "positionCoordinateXType": "GRID\\udfff"}')
    r = one_plot(registry, text)
    assert r.ok and r.model_dump_json()
    assert r.data["level_names_not_read"] == {"rep\\ud800": 1} and r.data["coordinate_types_not_read"] == {"GRID\\udfff": 1}


def test_r5b_long_level_names_are_cut_in_data_and_counted_by_their_full_value(registry):
    """Review 5: 25 level names of 130,000 characters made data 3.25 MB. Data keys are cut at 60 characters with an ellipsis;
    the distinct count is of the full values."""
    text = json.dumps({"observationLevelRelationships": [{"levelName": "n" * 5000 + str(k), "levelCode": "1"} for k in range(3)]})
    r = one_plot(registry, text)
    assert r.data["level_names_not_read"] == {"n" * 60 + "…": 1} and r.data["level_names_not_read_distinct"] == 3
    assert r.data["coordinate_types_not_read_distinct"] == 0 and len(json.dumps(r.data["level_names_not_read"])) < 100


def test_r5b_names_and_types_are_counted_once_per_plot(registry):
    """Review 5: counts were per statement (two "plant" relations of one plot counted 2), entry types per plot."""
    text = json.dumps({"observationLevelRelationships": [{"levelName": "plant", "levelCode": "1"}, {"levelName": "plant", "levelCode": "2"}],
                       "positionCoordinateX": 1, "positionCoordinateXType": "LATITUDE", "positionCoordinateY": 2, "positionCoordinateYType": "LATITUDE"})
    r = one_plot(registry, text)
    assert r.data["level_names_not_read"] == {"plant": 1} and r.data["coordinate_types_not_read"] == {"LATITUDE": 1}


def test_r5b_the_plots_own_level_name_is_counted_when_the_tool_does_not_read_it(registry):
    """Review 5 (older than round 5): observationLevel {"levelName": "plot\u200b"} or "plots" left plot blank with no count."""
    for k, name in enumerate(("plots", "plot\u200b")):
        r = one_plot(registry, json.dumps({"observationLevel": {"levelName": name, "levelCode": "9"}}), study=f"S{k + 5}")
        assert rows_of(registry, r)[0]["plot"] == "" and r.data["level_names_not_read"] == {name: 1}
        assert any("level names the tool does not read" in w and ascii(name) in w for w in r.warnings)
    read = one_plot(registry, json.dumps({"observationLevel": {"levelName": " Plot ", "levelCode": "9"}}), study="S7")
    assert rows_of(registry, read)[0]["plot"] == "9" and read.data["level_names_not_read"] == {}


def test_r5b_only_plain_numerals_are_compared_as_numbers(registry):
    """Review 5: "1_0" was named equal to "10" (Python's Decimal accepts underscores). Only numerals are compared: digits, one
    point, an exponent; no underscore, hex, Infinity or NaN. Codes compare in full ("1000" and "1001" differ)."""
    codes = ["10", "1e1", "1_0", "Infinity", "NaN", "sNaN", "0x1", "1000", "1000.0", "1001"]
    r = plots_with(registry, {f"P{n}": position(code, "1", str(n), "test", 1, n) for n, code in enumerate(codes, 1)})
    assert r.ok and r.data["codes_equal_as_numbers"] == {"rep": [["10", "1e1"], ["1000", "1000.0"]]}
    assert r.data["codes_equal_as_numbers_groups"] == {"rep": 2}


def test_r5b_codes_equal_as_numbers_are_capped_and_the_rest_is_counted(registry):
    """Review 5: 1,000 spellings of one rep made a 504,594-character warning; 12 groups showed 5 in data and 3 in the warning
    with no word of the rest. Data and warning now show the same 5 groups of at most 5 spellings each, and say how many more."""
    codes = ["0" * i + "1" for i in range(8)] + [c for k in range(2, 8) for c in (str(k), f"{k}.0")]
    r = plots_with(registry, {f"P{n}": position(code, "1", str(n), "test", 1, n) for n, code in enumerate(codes, 1)})
    groups = r.data["codes_equal_as_numbers"]["rep"]
    assert len(groups) == 5 and all(len(g) <= 5 for g in groups) and r.data["codes_equal_as_numbers_groups"] == {"rep": 7}
    assert groups[0] == ["00000001", "0000001", "000001", "00001", "0001"]
    w = next(w for w in r.warnings if "equal as numbers" in w)
    assert "and 3 more" in w and "and 2 more groups" in w and w.count("'") == 2 * (5 + 4 * 2)


def test_r5b_entry_types_are_never_compared_as_numbers(registry):
    r = plots_with(registry, {"P1": position("1", "1", "1", "1", 1, 1), "P2": position("1", "1", "2", "01", 1, 2)})
    assert r.data["codes_equal_as_numbers"] == {} and r.data["entry_types_not_standard"] == {"01": 1, "1": 1}


def test_r5b_the_cap_keeps_the_most_frequent_values(registry):
    positions = {f"P{n}": position("1", "1", str(n), f"odd{n:02d}", 1, n) for n in range(1, 22)}
    positions.update({f"P{n}": position("1", "1", str(n), "often", 1, n) for n in range(22, 25)})
    r = plots_with(registry, positions)
    odd = r.data["entry_types_not_standard"]
    assert len(odd) == 20 and list(odd)[0] == "often" and odd["often"] == 3 and r.data["entry_types_not_standard_distinct"] == 22


def test_r5b_the_warning_cuts_after_60_characters_and_says_more_only_when_there_is_more(registry):
    r = plots_with(registry, {"P1": position("1", "1", "1", "a" * 60, 1, 1), "P2": position("1", "1", "2", "b" * 61, 1, 2)})
    w = next(w for w in r.warnings if "not check, test or filler" in w)
    assert "'" + "a" * 60 + "'," in w and "'" + "b" * 60 + "'…" in w and "more" not in w
    five = plots_with(registry, {f"P{n}": position("1", "1", str(n), f"kind{n}", 1, n) for n in range(1, 6)}, study="S2")
    assert "more" not in next(w for w in five.warnings if "not check, test or filler" in w)


def test_r5b_a_padded_plot_id_joins_only_to_the_same_padded_id(registry):
    plots = [(" P1 ", "G1", "Alpha", "1", "1", "1", "test", 1, 1), ("P2", "G2", "Beta", "2", "1", "2", "test", 2, 1)]
    joined = design(registry, roster_artifact_id=roster(registry, plots),
                    artifact_id=measurements(registry, [("P1", "G1", "5"), (" P1 ", "G1", "6"), ("P2", "G2", "7")]))
    assert {row["value"]: row["rep"] for row in rows_of(registry, joined)} == {"5": "", "6": "1", "7": "2"}


def test_r5b_the_description_names_infinity_trimming_and_the_plots_own_level():
    schema = next(s for s in at.analyst_tool_schemas() if s["name"] == "plot_design")
    text = " ".join(schema["description"].split())
    assert "Infinity" in text and "trimmed" in text and "plot's own level" in text
    assert "invisible" not in at._has_text.__doc__
