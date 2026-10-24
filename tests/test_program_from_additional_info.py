"""A study's program from additionalInfo (coach 2026-10-24 12:50 UTC, NEXT 4; Luna B29.4, D15.2). SweetPotatoBase keeps a
study's program in additionalInfo (studies 5073, 5074, 5106: programDbId 5017 "DR&SS"; 4354: 1774 "Peru-CIP"); the top-level
programDbId and programName are empty. brapi_mcp_server._study_row exports both pairs, but KnownMetadata.from_rows read only
the top-level pair, so the program guard counted every study unknown and every program question stopped at a question. Now
from_rows (and from_registry, which uses it) falls back to the additionalInfo pair; a row with neither stays unknown.
Everything here is SYNTHETIC: made-up rows, no network, no model."""
from datetime import datetime, timezone

import pytest

import agents.coordinator as co
import analyst_tools as at
import brapi_mcp_server as srv
import catalog_lists
import catalog_summary
from artifacts import ArtifactRegistry
from part2_support import SYNTHETIC_BASE
from test_missing_metadata import assemble, proposed
from test_catalog_lists import TABLES as LIST_TABLES


def study(sid, *, top=("", ""), extra=None):
    """A study record shaped like SweetPotatoBase's /studies, through the export's own row builder."""
    record = {"studyDbId": sid, "studyName": f"SYNTHETIC trial {sid}", "programDbId": top[0], "programName": top[1]}
    if extra is not None:
        record["additionalInfo"] = {"programDbId": extra[0], "programName": extra[1]}
    return srv._study_row(record)


def known(*rows):
    return co.KnownMetadata.from_rows(list(rows), [{"observationVariableDbId": "V1", "observationVariableName": "SYNTHETIC yield"}],
                                      base_url=SYNTHETIC_BASE, complete=True)


def test_a_study_with_the_program_only_in_additional_info_has_that_program():
    [s] = known(study("S1", extra=("5017", "DR&SS"))).studies
    assert (s.program_id, s.program_name) == ("5017", "DR&SS")


def test_the_top_level_program_comes_first_when_both_are_recorded():
    [s] = known(study("S1", top=("P1", "Top program"), extra=("5017", "DR&SS"))).studies
    assert (s.program_id, s.program_name) == ("P1", "Top program")


def test_a_half_recorded_standard_program_is_not_mixed_with_additional_info():
    """Review 2, SHOULD-FIX 1: per field, (P1, "") and additionalInfo (5017, DR&SS) made the pair (P1, DR&SS): an ID and a name of
    two programs. additionalInfo is read only when BOTH standard fields name nothing."""
    [s] = known(study("S1", top=("P1", ""), extra=("5017", "DR&SS"))).studies
    assert (s.program_id, s.program_name) == ("P1", None)
    [t] = known(study("S2", top=("", "Alpha"), extra=("5017", "DR&SS"))).studies
    assert (t.program_id, t.program_name) == (None, "Alpha")


def test_a_study_with_neither_stays_unknown():
    [s] = known(study("S1"), ).studies
    assert (s.program_id, s.program_name) == (None, None)
    [t] = known(study("S2", extra=("", " "))).studies                          # empty or blank additionalInfo is no program
    assert (t.program_id, t.program_name) == (None, None)


def test_the_guard_counts_additional_info_programs_known():
    with_programs = known(study("S1", extra=("5017", "DR&SS")), study("S2", extra=("5017", "DR&SS")))
    plan = assemble("Which studies belong to the DR&SS program?", proposed(column="programName", value="DR&SS"), with_programs)
    assert not any("Program membership is unknown" in c.question for c in plan.clarifications)
    assert co.study_program_metadata_question("Which studies belong to the DR&SS program?",
                                              proposed(column="programName", value="DR&SS"), with_programs) is None


def test_the_guard_still_asks_when_a_study_has_neither():
    mixed = known(study("S1", extra=("5017", "DR&SS")), study("S2"))
    plan = assemble("Which studies belong to the DR&SS program?", proposed(column="programName", value="DR&SS"), mixed)
    question = " ".join(c.question for c in plan.clarifications)
    assert plan.status == "needs_clarification" and "unknown for 1 of 2" in question and "1 have the needed" in question


def test_the_catalog_summary_counts_them(tmp_path):
    summary = known(study("S1", extra=("5017", "DR&SS")), study("S2")).summary_for_model()
    assert "Program membership metadata: 1 of 2 studies have a program ID or name" in summary


def test_the_registry_path_falls_back_too(tmp_path):
    registry = ArtifactRegistry(tmp_path / "out", "run_program", base_url=SYNTHETIC_BASE)
    rows = [study("S1", extra=("5017", "DR&SS")), study("S2")]
    studies = registry.save_table(rows, kind="studies", columns={k: "study field (text)" for k in rows[0]}, complete=True,
                                  study_ids=["S1", "S2"], variable_ids=[], quality_notes=["SYNTHETIC"])
    variables = registry.save_table([{"observationVariableDbId": "V1", "observationVariableName": "SYNTHETIC yield"}], kind="variables",
                                    columns={"observationVariableDbId": "id", "observationVariableName": "name"}, complete=True,
                                    study_ids=[], variable_ids=["V1"], quality_notes=["SYNTHETIC"])
    meta = co.KnownMetadata.from_registry(registry, base_url=SYNTHETIC_BASE, studies_artifact=studies.artifact_id,
                                          variables_artifact=variables.artifact_id)
    assert [(s.program_id, s.program_name) for s in meta.studies] == [("5017", "DR&SS"), (None, None)]


# --- fix (c) (coach 2026-10-24 14:49 UTC): the export fills the standard fields from additionalInfo; count_records refuses unknown
# program cells as filter_rows does; one rule for an unknown field. The review of the first round (NOT SAFE): with programs known,
# a program COUNT read the empty top-level column and said 0 of 2 (check_program_count.py, kept verbatim below). ----------------

SPB = [("5017", "DR&SS"), ("5017", "DR&SS"), ("5017", "DR&SS"), ("1774", "Peru-CIP")]


def spb_rows():
    """check_program_count.py's rows: SweetPotatoBase-shaped, the program only in additionalInfo."""
    return [srv._study_row({"studyDbId": f"S{i}", "studyName": f"SYNTHETIC trial {i}", "additionalInfo": {"programDbId": p, "programName": n}})
            for i, (p, n) in enumerate(SPB, 1)]


def catalog(registry, rows):
    return registry.save_table(rows, kind="studies", columns={k: "study field (text)" for k in rows[0]}, complete=True,
                               study_ids=[r["studyDbId"] for r in rows], variable_ids=[], quality_notes=["SYNTHETIC"]).artifact_id


def counted(r):
    assert r.ok, r.error
    return [c["value"] for c in r.data["claims"]]


def test_the_export_fills_the_standard_fields_and_says_where_from():
    [row] = spb_rows()[:1]
    assert (row["programDbId"], row["programName"]) == ("5017", "DR&SS")
    assert (row["additionalInfoProgramDbId"], row["additionalInfoProgramName"]) == ("5017", "DR&SS")       # left as they are
    assert row["programSource"] == "additionalInfo (programDbId, programName)"


def test_a_top_level_program_is_kept_and_its_source_is_the_standard_fields():
    row = study("S1", top=("P1", "Top program"), extra=("5017", "DR&SS"))
    assert (row["programDbId"], row["programName"], row["programSource"]) == ("P1", "Top program", "standard fields")
    half = study("S2", top=("P1", ""), extra=("5017", "DR&SS"))
    assert (half["programDbId"], half["programName"], half["programSource"]) == ("P1", "", "standard fields")


@pytest.mark.parametrize("token", ["", "[]", "{}", "NA", "na", "null", "NULL", " NA "])
def test_an_unknown_token_is_no_program_anywhere(token):
    row = study("S1", extra=(token, token))
    assert (row["programDbId"], row["programName"], row["programSource"]) == ("", "", "")
    [s] = known(row).studies
    assert (s.program_id, s.program_name) == (None, None)
    assert at.unknown_metadata(token) and not at.unknown_metadata("NAM") and not at.unknown_metadata("5017")


def test_the_reviewers_program_count_is_two_programs(tmp_path):
    """check_program_count.py verbatim: "How many different programs do the studies belong to?" was 0; it is 2."""
    rows = spb_rows()
    meta = known(*rows)
    question = "How many different programs do the studies belong to?"
    draft = co.PlanDraft.model_validate({
        "interpretation": "SYNTHETIC program count", "scope": {"all_known_studies": True, "filters": {}},
        "statistic": {"kind": "count", "denominator": "programs", "grouping": ["programName"]},
        "steps": [{"step_id": "step_fetch", "agent": "retriever", "action": "export_metadata", "inputs": {"entity": "studies"}},
                  {"step_id": "step_count", "agent": "analyst", "action": "count_records", "inputs": {"distinct_by": "programName"},
                   "depends_on": ["step_fetch"]}]})
    plan, problems, _ = co.assemble_plan(question, draft, meta, base_url=SYNTHETIC_BASE, now=datetime(2026, 10, 24, tzinfo=timezone.utc),
                                         plan_id="plan_check", person_reviews_scope=True)
    assert problems == [] and plan.status == "ready"
    registry = ArtifactRegistry(tmp_path / "out", "run_program_c", base_url=SYNTHETIC_BASE)
    assert counted(at.dispatch_analyst(registry, "count_records", {"artifact_id": catalog(registry, rows), "distinct_by": "programName"})) == [2]


def test_the_studies_of_program_5017_are_three(tmp_path):
    registry = ArtifactRegistry(tmp_path / "out", "run_program_c", base_url=SYNTHETIC_BASE)
    kept = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": catalog(registry, spb_rows()), "column": "programDbId", "op": "eq",
                                                         "value": "5017"})
    assert kept.ok and not kept.warnings                                   # every program known: no unknown-cell warning
    assert counted(at.dispatch_analyst(registry, "count_records", {"artifact_id": kept.artifact_ids[0], "distinct_by": "studyDbId"})) == [3]
    plan = assemble("How many studies belong to program 5017?", proposed(column="programDbId", value="5017", filters={"program_id": "5017"}),
                    known(*spb_rows()))
    assert plan.status == "ready" and plan.scope.matching_study_count == 3


@pytest.mark.parametrize("mode", ["distinct_by", "group_by"])
@pytest.mark.parametrize("column", ["programName", "programDbId"])
def test_a_table_of_unknown_programs_gives_no_number(tmp_path, mode, column):
    rows = [study(f"S{i}", top=(token, token)) for i, token in enumerate(("[]", "NA", "{}", "null"), 1)]
    registry = ArtifactRegistry(tmp_path / "out", "run_program_c", base_url=SYNTHETIC_BASE)
    r = at.dispatch_analyst(registry, "count_records", {"artifact_id": catalog(registry, rows), mode: column})
    assert not r.ok and r.error.code == "incomplete_data" and "4 of 4" in r.error.message and "not zero" in r.error.message


def test_one_unknown_program_among_known_ones_gives_no_number(tmp_path):
    rows = spb_rows() + [study("S9", top=("NA", "NA"))]
    registry = ArtifactRegistry(tmp_path / "out", "run_program_c", base_url=SYNTHETIC_BASE)
    r = at.dispatch_analyst(registry, "count_records", {"artifact_id": catalog(registry, rows), "distinct_by": "programName"})
    assert not r.ok and r.error.code == "incomplete_data" and "1 of 5" in r.error.message


def test_filter_rows_shares_the_unknown_rule(tmp_path):
    rows = [study(f"S{i}", top=(token, token)) for i, token in enumerate(("{}", "NA", "null"), 1)]
    registry = ArtifactRegistry(tmp_path / "out", "run_program_c", base_url=SYNTHETIC_BASE)
    r = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": catalog(registry, rows), "column": "programName", "op": "eq", "value": "DR&SS"})
    assert not r.ok and r.error.code == "incomplete_data" and "missing for all 3" in r.error.message


def test_the_catalog_summary_says_where_programs_come_from():
    summary = catalog_summary.summarize(known(*spb_rows(), study("S9")), host="synthetic.invalid", fetched_at=None).text
    assert "Programs: 4 of 5 studies name a program (standard or additionalInfo)." in summary


def test_the_program_list_names_additional_info_programs_and_its_note_is_true():
    meta = known(*spb_rows())
    shown = catalog_lists.chat_message(catalog_lists.build(catalog_lists.ListRequest("programs"), meta, LIST_TABLES), in_report=True)
    assert "DR&amp;SS" in shown and "Peru-CIP" in shown and "does not use that field" not in shown       # the list escapes &



def test_the_offline_catalog_filter_reads_the_same_program():
    """brapi_client: a study search by program on the cached catalog (offline) compared the empty standard field and kept none."""
    from types import SimpleNamespace

    import brapi_client
    from contracts import CollectionResult
    records = [{"studyDbId": f"S{i}", "studyName": f"SYNTHETIC trial {i}", "additionalInfo": {"programDbId": p, "programName": n}}
               for i, (p, n) in enumerate(SPB, 1)]
    records += [{"studyDbId": "S5", "programDbId": "5017"},                                       # the standard field, as other servers
                {"studyDbId": "S6", "programDbId": "NA", "additionalInfo": {"programDbId": "5017"}},   # an unknown standard field
                {"studyDbId": "S7", "additionalInfo": {"programDbId": "[]"}}]                       # neither
    whole = CollectionResult(records=records, returned_count=len(records), reported_total=len(records), complete=True, status="complete",
                             request_ids=[], warnings=[])
    fake = SimpleNamespace(get_all=lambda path, params, approval=None, max_pages=1: whole)
    got = brapi_client.BrapiClient._filter_cached_catalog(fake, {"programDbId": "5017"}, approval=None, max_pages=1)
    assert sorted(r["studyDbId"] for r in got.records) == ["S1", "S2", "S3", "S5", "S6"] and got.complete
    assert brapi_client._study_program(records[-1]) is None and brapi_client._study_program(records[4]) == "5017"



def test_the_refusal_is_for_catalog_tables_only_as_in_filter_rows(tmp_path):
    """filter_rows judges unknown cells on catalog tables only; count_records does the same: a derived table is counted as before."""
    registry = ArtifactRegistry(tmp_path / "out", "run_program_c", base_url=SYNTHETIC_BASE)
    rows = [{"group_id": f"G{i}", "programName": name} for i, name in enumerate(("NA", "DR&SS", "DR&SS"), 1)]
    made = registry.save_table(rows, kind="analysis_result", columns={"group_id": "id", "programName": "text"}, complete=True,
                               study_ids=["S1"], variable_ids=[], quality_notes=["SYNTHETIC"]).artifact_id
    assert counted(at.dispatch_analyst(registry, "count_records", {"artifact_id": made, "distinct_by": "programName"})) == [2]



# --- review 2 (SAFE; SHOULD-FIX 1-7) -------------------------------------------------------------------------------------

def test_an_old_export_row_still_falls_back():
    """A saved export from before the fill (no programSource, standard pair empty): from_rows reads the additionalInfo pair."""
    old = {"studyDbId": "S1", "studyName": "SYNTHETIC trial", "programDbId": "", "programName": "",
           "additionalInfoProgramDbId": "5017", "additionalInfoProgramName": "DR&SS"}
    [s] = known(old).studies
    assert (s.program_id, s.program_name) == ("5017", "DR&SS")


def test_a_junk_standard_program_is_replaced_by_additional_info():
    row = study("S1", top=("NA", "null"), extra=("5017", "DR&SS"))
    assert (row["programDbId"], row["programName"], row["programSource"]) == ("5017", "DR&SS", "additionalInfo (programDbId, programName)")


def test_a_json_number_is_a_program_id():
    """Review 2, SHOULD-FIX 3: {"programDbId": 5017} was kept by the offline filter on base and dropped by the change."""
    import brapi_client
    assert not at.unknown_metadata(5017) and at.unknown_metadata(None) and at.unknown_metadata([]) and at.unknown_metadata({})
    assert brapi_client._study_program({"studyDbId": "S8", "programDbId": 5017}) == 5017
    assert brapi_client._study_program({"studyDbId": "S9", "programName": "Alpha", "additionalInfo": {"programDbId": "5017"}}) is None


def junk_programs(registry):
    rows = [study(f"S{i}", top=(tok, tok)) for i, tok in enumerate(("DR&SS", "DR&SS", "NA", "{}", "null", ""), 1)]
    for row in rows[:2]:
        row["programDbId"] = "5017"
    return catalog(registry, rows)


def test_filter_rows_does_not_match_unknown_program_cells(tmp_path):
    """Review 2, SHOULD-FIX 2: `ne "DR&SS"` kept the NA, {} and null rows as known non-matches."""
    registry = ArtifactRegistry(tmp_path / "out", "run_program_c", base_url=SYNTHETIC_BASE)
    r = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": junk_programs(registry), "column": "programName", "op": "ne", "value": "DR&SS"})
    assert r.ok and r.data["rows_out"] == 0
    assert any("4 of 6 catalog rows have no programName; 0 known matches" in w for w in r.warnings)


def test_count_records_refuses_unknown_additional_info_program_cells(tmp_path):
    """Review 2, SHOULD-FIX 4: distinct additionalInfoProgramName counted "{}" as a program (3 for 2)."""
    rows = spb_rows() + [study("S9", extra=("{}", "{}"))]
    registry = ArtifactRegistry(tmp_path / "out", "run_program_c", base_url=SYNTHETIC_BASE)
    r = at.dispatch_analyst(registry, "count_records", {"artifact_id": catalog(registry, rows), "distinct_by": "additionalInfoProgramName"})
    assert not r.ok and r.error.code == "incomplete_data" and "1 of 5" in r.error.message


def test_count_records_checks_group_by_and_distinct_by_together(tmp_path):
    registry = ArtifactRegistry(tmp_path / "out", "run_program_c", base_url=SYNTHETIC_BASE)
    r = at.dispatch_analyst(registry, "count_records", {"artifact_id": junk_programs(registry), "group_by": "studyType", "distinct_by": "programName"})
    assert not r.ok and r.error.code == "incomplete_data"


def test_a_missing_program_column_is_an_invalid_argument(tmp_path):
    registry = ArtifactRegistry(tmp_path / "out", "run_program_c", base_url=SYNTHETIC_BASE)
    rows = [{"studyDbId": "S1", "studyName": "SYNTHETIC trial"}]
    made = registry.save_table(rows, kind="studies", columns={"studyDbId": "id", "studyName": "name"}, complete=True, study_ids=["S1"],
                               variable_ids=[], quality_notes=["SYNTHETIC"]).artifact_id
    r = at.dispatch_analyst(registry, "count_records", {"artifact_id": made, "distinct_by": "programName"})
    assert not r.ok and r.error.code == "invalid_argument"


def test_other_catalog_columns_keep_their_rule(tmp_path):
    """Review 2, SHOULD-FIX 6: SweetPotatoBase location 120 is named "NA"; a location question that worked must keep working."""
    rows = [study(f"S{i}", top=("5017", "DR&SS")) for i in range(1, 4)]
    for row, name in zip(rows, ("Kampala", "NA", "Kampala")):
        row["locationName"] = name
    registry = ArtifactRegistry(tmp_path / "out", "run_program_c", base_url=SYNTHETIC_BASE)
    r = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": catalog(registry, rows), "column": "locationName", "op": "contains",
                                                      "value": "Kampala"})
    assert r.ok and r.data["rows_out"] == 2 and not r.warnings


def test_the_study_facts_say_where_the_program_came_from():
    """Review 2, SHOULD-FIX 5: a study's facts showed "Program ID: 5017" with no word of where it came from."""
    import metadata_answers
    assert metadata_answers.STUDY_FIELDS["programSource"] == "Program source"
    assert "programSource" in srv.EXPORT_SHAPES["studies"][1]


def test_a_large_catalog_with_every_program_named_stays_compact():
    """Review 2: with 2039 of 2039 programs named the Programs line made the summary 1501 characters (the cap is 1500)."""
    studies = [co.KnownStudy(study_id=f"S{i}", name=f"Yield trial number {i} at the station", location_id=f"L{i % 100}",
                             location_name=f"Research station {i % 100}, NC", seasons=[str(2003 + i % 24)], study_type=f"Trial type {i % 12}",
                             program_id="5017", program_name="DR&SS") for i in range(2039)]
    variables = [co.KnownVariable(variable_id=f"V{i}", name=f"trait {i}") for i in range(486)]
    summary = catalog_summary.summarize(co.KnownMetadata(base_url=SYNTHETIC_BASE, complete=True, studies=studies, variables=variables),
                                        host="sweetpotatobase.org", fetched_at=datetime(2026, 6, 1, tzinfo=timezone.utc))   # as that test
    assert "Programs: 2039 of 2039 studies name a program" in summary.text and len(summary.text) <= 1500
