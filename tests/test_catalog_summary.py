"""
tests/test_catalog_summary.py — the saved catalog's summary that the front-door model may read (catalog_summary.py), offline.

What these tests prove, in plain words:
* The summary is read from the cache only: an offline client, no database request (the fake transport records none), no run
  folder and no new file anywhere in the test's folder. A catalog that is not completely cached gives no summary.
* It is the same catalog a run loads: the rows read here give exactly the metadata a run builds from its saved tables.
* Code writes every count: the totals; every location named by a study, most studies first; every season, newest first; the
  study types; how many studies name a program; a few study names; when the catalogs were fetched.
* The answer to a catalog question is written by code only (answer): the totals; the locations, study types and seasons the
  message names with their number of studies (whole items; the rest are counted); when it names two kinds of them, the
  studies that match them together (a place and a year with no study together says 0); when it names none, the locations
  and study types with the most studies; that the catalog stores location names only (always said); then the source.
  Database names are escaped.
* Which locations a message "mentions" is decided by code, word by word, ignoring case and accents: 3 or more letters (not
  common words such as "station" or "state"), numbers of 2 or more digits, and 2-letter words only in capitals in both and
  not everyday words ("NC" yes; "me", "OK", "ID" no; none in a message written all in capitals), or the whole name as
  written ("Wa"). A location ID without a name is counted but never mentioned. A year names every season that has it as a
  word ("2025" names "Fall 2025"; a year inside an ID names nothing). A study type is named by its whole name or a word.
* The summary is kept in memory for a while (no second read of the cache), and forget() drops it.
Synthetic catalogs only; the conftest guard blocks every real socket.
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import catalog_summary
from agents.coordinator import KnownMetadata, KnownStudy, KnownVariable
from brapi_mcp_server import ExportMetadataArgs, ToolContext, build_context, tool_export_metadata
from cache_store import CacheStore
from part2_support import seed_fixture
from run import RunConfig

BASE = "https://summary.invalid/brapi/v2"
FETCHED = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)          # part2_support.FIXTURE_FETCHED_AT
STUDIES = [
    {"studyDbId": "S1", "studyName": "Clinton yield trial 2025", "locationDbId": "L1", "locationName": "Clinton, NC",
     "seasons": ["2025"], "studyType": "Advanced Yield Trial"},
    {"studyDbId": "S2", "studyName": "Clinton yield trial 2024", "locationDbId": "L1", "locationName": "Clinton, NC",
     "seasons": ["2024"], "studyType": "Advanced Yield Trial"},
    {"studyDbId": "S3", "studyName": "Kinston observation 2025", "locationDbId": "L2", "locationName": "Kinston, NC",
     "seasons": ["2025"], "studyType": "Preliminary Yield Trial", "programDbId": "PG1", "programName": "Sweetpotato breeding"},
    {"studyDbId": "S4", "studyName": "Namulonge clonal trial", "locationDbId": "L3", "locationName": "Namulonge, Uganda",
     "seasons": ["2019"], "studyType": "Preliminary Yield Trial"},
    {"studyDbId": "S5", "studyName": "Unplaced germplasm study", "seasons": ["Fall season"]},
]
VARIABLES = [{"observationVariableDbId": "V1", "observationVariableName": "fresh root yield", "trait": {"traitName": "fresh root yield"},
              "scale": {"units": "t/ha"}},
             {"observationVariableDbId": "V2", "observationVariableName": "dry matter content", "trait": {"traitName": "dry matter content"},
              "scale": {"units": "%"}}]


def write_fixture(folder: Path, name: str, endpoint: str, records: list[dict]) -> None:
    body = {"_note": "SYNTHETIC catalog-summary fixture. Invented data for tests. Not real breeding data.", "_synthetic": True,
            "_request": {"endpoint": endpoint, "params": {"page": 0, "pageSize": 1000}},
            "metadata": {"pagination": {"currentPage": 0, "pageSize": 1000, "totalCount": len(records), "totalPages": 1},
                         "synthetic": True},
            "result": {"data": [{"commonCropName": "sweetpotato", **r} for r in records]}}
    (folder / name).write_text(json.dumps(body), encoding="utf-8")


def seed_catalog(cache_dir: Path, base: str = BASE, *, studies: bool = True, variables: bool = True) -> None:
    """The catalogs as a run's cache holds them after the catalog setup (or a saved copy)."""
    folder = cache_dir.parent / "summary_fixtures"
    folder.mkdir(exist_ok=True)
    write_fixture(folder, "summary_studies.json", "/studies", STUDIES)
    write_fixture(folder, "summary_variables.json", "/variables", VARIABLES)
    cache = CacheStore(cache_dir)
    for wanted, name in ((studies, "summary_studies.json"), (variables, "summary_variables.json")):
        if wanted:
            seed_fixture(cache, base, name, folder)


@pytest.fixture
def place(tmp_path, monkeypatch):
    """The settings a front-door read sees, a transport that records any request, and a clean summary memory."""
    import brapi_client as bc

    monkeypatch.setenv("BRAPI_BASE_URL", BASE)
    monkeypatch.setenv("BRAPI_MODE", "live")                    # even with live settings, the summary reads offline
    requests: list[object] = []
    monkeypatch.setattr(bc, "RequestsTransport", lambda: requests.append)
    catalog_summary.forget()
    yield dataclasses.make_dataclass("Place", ["root", "cache", "requests", "config"])(
        tmp_path, tmp_path / "cache", requests, RunConfig(question="tell me about Florida", out_dir=tmp_path / "out",
                                                         cache_dir=tmp_path / "cache"))
    catalog_summary.forget()


def files_in(root: Path) -> set[Path]:
    return {p for p in root.rglob("*") if p.is_file()}


def test_the_summary_is_read_from_the_cache_only_no_request_no_run_folder_no_file(place):
    seed_catalog(place.cache)
    before = files_in(place.root)
    summary = catalog_summary.from_cache(place.config)
    assert summary is not None and place.requests == []                       # nothing was requested
    assert files_in(place.root) == before and not (place.root / "out").exists()   # no run folder, no file written
    assert summary.host == "summary.invalid" and summary.fetched_at == FETCHED
    text = summary.text
    assert text.startswith("Saved catalog of summary.invalid, fetched 2026-06-01 00:00 UTC.")
    assert ("Totals: 5 studies, 2 traits, 3 locations and 4 seasons named by studies, 2 study types. Studies that name no "
            "location: 1; no season: 0.") in text
    assert "Locations with the most studies (studies in brackets): Clinton, NC (2); Kinston, NC (1); Namulonge, Uganda (1)" in text
    assert "Seasons named by studies, newest first (studies in brackets): 2025 (2); 2024 (1); 2019 (1); Fall season (1)" in text
    assert "Study types with the most studies (studies in brackets): Advanced Yield Trial (2); Preliminary Yield Trial (2)" in text
    assert "Programs: 1 of 5 studies name a program (standard or additionalInfo)." in text     # NEXT 4: additionalInfo is read
    assert ("A few study names, newest season first: “Clinton yield trial 2025” (ID S1; Clinton, NC; 2025); “Kinston observation "
            "2025” (ID S3; Kinston, NC; 2025); “Clinton yield trial 2024” (ID S2; Clinton, NC; 2024)") in text
    assert "Florida" not in text
    assert summary.for_message("tell me about Florida") == text + (
        "\nLocations the message mentions: none (no distinctive word of the message is in any of the 3 location names; common "
        "words such as 'station' are not compared, and the catalog stores location names only, without towns, states or "
        "countries).")
    assert summary.for_message("what about clinton?").endswith(
        "Locations the message mentions (checked by code; number of studies in brackets): Clinton, NC (2).")
    assert [name for name, _ in summary.mentioned("stations in NC?")] == ["Clinton, NC", "Kinston, NC"]
    assert summary.mentioned("stations in nc?") == []                          # a 2-letter word counts only in capitals
    assert summary.mentioned("the research station farm") == []                  # common words name no location
    source = ("*From the saved catalog of summary.invalid, fetched 2026-06-01 00:00 UTC. Nothing new was requested; ask a data "
              "question to look up records.*")
    places_note = ("The catalog stores only location names (often a station's name), without towns, states or countries, so a "
                   "place may also be there under another name; ask **list all locations** to see every name.")
    assert summary.answer("tell me about Florida") == "\n".join([
        "**From the saved catalog** (no records were looked up):", "",
        "- It has 5 studies, 2 traits and 3 locations named by studies, with seasons from 2019 to 2025.",
        "- Your message names no location, study type or season of the catalog. The locations with the most studies: Clinton, "
        "NC (2 studies); Kinston, NC (1 study); Namulonge, Uganda (1 study). The study types with the most studies: Advanced "
        f"Yield Trial (2 studies); Preliminary Yield Trial (2 studies). {places_note}", "", source])
    assert summary.answer("anything at Clinton in 2025 or 2018?") == "\n".join([
        "**From the saved catalog** (no records were looked up):", "",
        "- It has 5 studies, 2 traits and 3 locations named by studies, with seasons from 2019 to 2025.",
        f"- Locations whose names contain a word of your message: Clinton, NC (2 studies). {places_note}",
        "- Seasons you named: 2025 (2 studies).", "- No study names a season with 2018.",
        "- Each count above is for that one name; together, studies at Clinton, NC in 2025: 1 study.", "", source])
    assert summary.seasons_mentioned("in the fall season?") == ([("Fall season", 1)], [])


@pytest.mark.parametrize("cached", [{"studies": False}, {"variables": False}, {"studies": False, "variables": False}],
                         ids=["no-studies", "no-traits", "nothing"])
def test_a_catalog_that_is_not_completely_cached_gives_no_summary(place, cached):
    seed_catalog(place.cache, **cached)
    assert catalog_summary.from_cache(place.config) is None and place.requests == []


def test_another_servers_cache_is_not_read(place):
    seed_catalog(place.cache, base="https://another.invalid/brapi/v2")
    assert catalog_summary.from_cache(place.config) is None and place.requests == []


def test_the_rows_give_the_same_metadata_a_run_builds_from_its_saved_tables(place):
    seed_catalog(place.cache)
    ctx = build_context(offline=True, cache_dir=place.cache, out_dir=place.root / "run_out", run_id="run_compare")
    studies = tool_export_metadata(ctx, ExportMetadataArgs(entity="studies"))
    variables = tool_export_metadata(ctx, ExportMetadataArgs(entity="variables"))
    from_tables = KnownMetadata.from_registry(ctx.registry, base_url=BASE, studies_artifact=studies.artifact_ids[0],
                                              variables_artifact=variables.artifact_ids[0])
    settings = dataclasses.replace(ctx.client.settings, mode="live")
    from_rows, fetched = catalog_summary._read(settings, place.cache)
    assert from_rows == from_tables and fetched == FETCHED


def test_the_summary_reads_as_many_pages_as_a_runs_cache_read():
    assert catalog_summary.EXPORT_MAX_PAGES == ToolContext.__dataclass_fields__["export_max_pages"].default


def test_the_summary_is_kept_for_a_while_and_forget_drops_it(place, monkeypatch):
    seed_catalog(place.cache)
    reads: list[Path] = []
    read = catalog_summary._read
    monkeypatch.setattr(catalog_summary, "_read", lambda settings, cache_dir: reads.append(cache_dir) or read(settings, cache_dir))
    first = catalog_summary.from_cache(place.config)
    assert catalog_summary.from_cache(place.config) is first and len(reads) == 1
    catalog_summary.forget()
    assert catalog_summary.from_cache(place.config) == first and len(reads) == 2
    clock = [0.0]
    monkeypatch.setattr(catalog_summary, "_clock", lambda: clock[0])
    catalog_summary.forget()
    catalog_summary.from_cache(place.config)
    clock[0] = catalog_summary.CACHE_SECONDS + 1                             # too old: read again
    catalog_summary.from_cache(place.config)
    assert len(reads) == 4


def test_a_missing_catalog_is_not_kept_so_a_later_setup_is_seen(place):
    assert catalog_summary.from_cache(place.config) is None
    seed_catalog(place.cache)                                               # e.g. the first run's approved catalog setup
    assert catalog_summary.from_cache(place.config) is not None


def test_a_summary_never_stops_the_chat(place, monkeypatch):
    def broken(settings, cache_dir):
        raise OSError("the cache folder cannot be read")
    monkeypatch.setattr(catalog_summary, "_read", broken)
    assert catalog_summary.from_cache(place.config) is None


SMALL = KnownMetadata(base_url=BASE, complete=True,
                      studies=[KnownStudy(study_id="S1", name="A", location_id="L1", location_name="Clinton, NC", seasons=["2025"]),
                               KnownStudy(study_id="S2", name="B", location_id="L1", location_name="Clinton, NC", seasons=["2025"]),
                               KnownStudy(study_id="S3", name="C", location_id="L2", location_name="Kinston, NC", seasons=["2024"])],
                      variables=[KnownVariable(variable_id="V1", name="fresh root yield")])


def test_the_answer_lists_whole_items_and_counts_the_rest():
    studies = [KnownStudy(study_id=f"S{i}_{k}", location_id=f"L{i}", location_name=f"Clinton plot {i:02d} " + "y" * 50,
                          seasons=[f"Fall {2000 + i}"]) for i in range(14) for k in range(2)]
    summary = catalog_summary.summarize(KnownMetadata(base_url=BASE, complete=True, studies=studies), host="h", fetched_at=None)
    answer = summary.answer("anything at Clinton in 2005 or 2030?")
    assert answer.count("(2 studies)") == 11 and "; and 4 more." in answer       # 10 locations + 1 season; the rest counted
    assert "- Seasons you named: Fall 2005 (2 studies)." in answer and "- No study names a season with 2030." in answer
    assert "y" * 61 not in answer and "…" in answer                              # names are cut whole, never inside a count


def test_database_names_in_the_answer_are_escaped():
    summary = catalog_summary.summarize(KnownMetadata(base_url=BASE, complete=True, studies=[
        KnownStudy(study_id="S1", location_id="L1", location_name="Clinton [x](http://x.invalid) <b>bold</b>")]),
        host="h", fetched_at=None)
    answer = summary.answer("clinton?")
    assert "<b>" not in answer and "](" not in answer and "&lt;b&gt;" in answer


def test_every_count_the_person_sees_is_written_by_code_for_the_places_the_message_names():
    studies = [KnownStudy(study_id=f"S{i}_{k}", location_id=f"L{i}", location_name=f"Big station {i}", seasons=["2025"])
               for i in range(12) for k in range(3)]                          # 12 locations with 3 studies each
    studies.append(KnownStudy(study_id="X1", location_id="LX", location_name="Kinston, NC", seasons=["2024"]))
    summary = catalog_summary.summarize(KnownMetadata(base_url=BASE, complete=True, studies=studies), host="h", fetched_at=None)
    assert "Kinston" not in summary.text                                       # not among the 10 with the most studies
    assert "Kinston, NC (1)" in summary.for_message("what about Kinston?")      # the model's line, for this message
    answer = summary.answer("what about Kinston in 2024?")
    assert "- Locations whose names contain a word of your message: Kinston, NC (1 study). The catalog stores" in answer
    assert "- Seasons you named: 2024 (1 study)." in answer and "It has 37 studies, 0 traits and 13 locations" in answer
    assert "together, studies at Kinston, NC in 2024: 1 study." in answer


@pytest.mark.parametrize("message, name, found", [
    ("tell me about Bogota", "Bogotá", True), ("anything in NC?", "Clinton, NC", True), ("ho ho, any trials?", "Ho", False),
    ("which state has the most?", "Mississippi State", False), ("OK, and what about Clinton?", "Stillwater, OK", False),
    ("give me the study ID", "Aberdeen, ID", False), ("anything in Wa?", "Wa", True), ("ANYTHING IN NC?", "Clinton, NC", False),
    ("OH, AND WHAT ABOUT CLINTON?", "Wooster, OH", False),
], ids=["accents", "capitals", "two-letter-name-lowercase", "common-state", "everyday-OK", "everyday-ID", "two-letter-name-as-written",
        "all-capitals", "all-capitals-OH"])
def test_accents_capitals_and_everyday_words(message, name, found):
    summary = catalog_summary.summarize(KnownMetadata(base_url=BASE, complete=True, studies=[
        KnownStudy(study_id="S1", location_id="L1", location_name=name)]), host="h", fetched_at=None)
    assert bool(summary.mentioned(message)) is found


@pytest.mark.parametrize("message, name", [
    ("tell me about Florida", "Orono, ME"), ("yield or height?", "Corvallis, OR"), ("what trait id is it?", "Aberdeen, ID"),
    ("is there anything in the field?", "Field 7"),
], ids=["me", "or", "id", "common-word"])
def test_short_or_common_words_do_not_mention_a_location(message, name):
    summary = catalog_summary.summarize(KnownMetadata(base_url=BASE, complete=True, studies=[
        KnownStudy(study_id="S1", location_id="L1", location_name=name)]), host="h", fetched_at=None)
    assert summary.mentioned(message) == []


def test_a_location_id_without_a_name_is_counted_but_never_mentioned():
    summary = catalog_summary.summarize(KnownMetadata(base_url=BASE, complete=True, studies=[
        KnownStudy(study_id="S1", location_id="L7"), KnownStudy(study_id="S2", location_id="L8", location_name="Clinton, NC")]),
        host="h", fetched_at=None)
    assert summary.locations == (("Clinton, NC", 1),) and summary.unnamed_locations == 1 and summary.location_count == 2
    assert "Totals: 2 studies, 0 traits, 2 locations" in summary.text and "and 1 location IDs without a name" in summary.text
    assert summary.mentioned("no name, ID L7?") == [] and "2 locations named by studies" in summary.answer("L7")


def test_large_counts_are_written_whole():
    big = KnownMetadata(base_url=BASE, complete=True, studies=[KnownStudy(study_id=f"S{i}") for i in range(2039)],
                        variables=[KnownVariable(variable_id="V007")])
    summary = catalog_summary.summarize(big, host="summary.invalid", fetched_at=None)
    assert "Totals: 2039 studies" in summary.text and "fetched at a time that was not recorded" in summary.text
    assert "- It has 2039 studies, 1 trait and 0 locations named by studies." in summary.answer("hello")


def test_the_summary_stays_compact_for_a_catalog_as_large_as_sweetpotatobase():
    """About 2,000 studies at 100 locations in 24 seasons with 12 study types: the text that goes with every label-deciding
    call stays near 1,500 characters (the coach's note), while every location can still be found for a message."""
    studies = [KnownStudy(study_id=f"S{i}", name=f"Yield trial number {i} at the station",
                          location_id=f"L{i % 100}", location_name=f"Research station {i % 100}, NC",
                          seasons=[str(2003 + i % 24)], study_type=f"Trial type {i % 12}") for i in range(2039)]
    variables = [KnownVariable(variable_id=f"V{i}", name=f"trait {i}") for i in range(486)]
    summary = catalog_summary.summarize(KnownMetadata(base_url=BASE, complete=True, studies=studies, variables=variables),
                                        host="sweetpotatobase.org", fetched_at=FETCHED)
    message = summary.for_message("tell me about station 77")
    assert len(summary.text) <= 1500 and len(message) <= 1700
    assert "Research station 77, NC (20)" in message and len(summary.locations) == 100


def test_long_names_are_cut_and_long_lists_are_counted():
    studies = [KnownStudy(study_id=f"S{i}", location_id=f"L{i}", location_name=f"Station {i} " + "x" * 200) for i in range(305)]
    summary = catalog_summary.summarize(KnownMetadata(base_url=BASE, complete=True, studies=studies), host="h", fetched_at=None)
    assert "x" * 61 not in summary.text and "; and 295 more" in summary.text
    assert [name.split()[1] for name, _ in summary.mentioned("what about station 12?")] == ["12"]


GURUE = KnownMetadata(base_url=BASE, complete=True, studies=[
    *[KnownStudy(study_id=f"G{i}", location_id="L1", location_name="Gurue", seasons=["2024"], study_type="Advanced Yield Trial")
      for i in range(3)],
    *[KnownStudy(study_id=f"K{i}", location_id="L2", location_name="Kinston station", seasons=["2025"], study_type="Variety Release Trial")
      for i in range(2)],
    KnownStudy(study_id="P1", location_id="L3", location_name="Pucallpa, Peru", seasons=["2025"], study_type="Advanced Yield Trial"),
    KnownStudy(study_id="P2", location_id="L4", location_name="Satipo", seasons=["Fall 2025"], study_type="Advanced Yield Trial")])


def test_a_place_and_a_season_named_together_get_the_count_of_both_even_when_it_is_zero():
    """Review finding (2026-10-16): "any trials at Gurue in 2025?" showed Gurue's 144 studies and 2025's 64, never that no
    study is at Gurue in 2025."""
    summary = catalog_summary.summarize(GURUE, host="h", fetched_at=None)
    answer = summary.answer("any trials at Gurue in 2025?")
    assert "- Locations whose names contain a word of your message: Gurue (3 studies)." in answer
    assert "- Seasons you named: 2025 (3 studies); Fall 2025 (1 study)." in answer
    assert "- Each count above is for that one name; together, studies at Gurue in 2025 or Fall 2025: 0 studies." in answer
    together = summary.answer("advanced yield trials at Satipo in 2025?")
    assert "together, studies at Satipo of type Advanced Yield Trial in 2025 or Fall 2025: 1 study." in together
    assert "Each count above" not in summary.answer("tell me about Gurue")     # one kind named: no combined count


def test_the_place_note_is_given_even_when_a_location_matches():
    """Review finding: "how many studies are in Peru?" named only "Pucallpa, Peru" and dropped the note that other Peruvian
    stations may be there under other names (Satipo)."""
    answer = catalog_summary.summarize(GURUE, host="h", fetched_at=None).answer("how many studies are in Peru?")
    assert "Pucallpa, Peru (1 study). The catalog stores only location names" in answer and "under another name" in answer


def test_study_types_are_answered_by_their_whole_name_or_a_word():
    """Review finding: "how many Variety Release Trials are there?" got a location's count, not the study type's."""
    summary = catalog_summary.summarize(GURUE, host="h", fetched_at=None)
    answer = summary.answer("how many Variety Release Trials are there?")
    assert "- Study types you named: Variety Release Trial (2 studies)." in answer and "Locations whose names" not in answer
    with_station = catalog_summary.summarize(GURUE.model_copy(update={"studies": [*GURUE.studies, KnownStudy(
        study_id="W1", location_id="L9", location_name="Gwebi Variety Testing Center", seasons=["2025"])]}), host="h", fetched_at=None)
    assert with_station.mentioned("how many Variety Release Trials are there?") == []   # the type's words name no place
    assert with_station.mentioned("anything at the Variety Testing Center?") == [("Gwebi Variety Testing Center", 1)]
    assert "- Study types whose names contain a word of your message: Advanced Yield Trial (5 studies)." in summary.answer(
        "anything advanced?")
    assert summary.types == (("Advanced Yield Trial", 5), ("Variety Release Trial", 2))


def test_a_year_inside_an_id_names_no_season_and_a_range_is_said_to_be_counted_alone():
    summary = catalog_summary.summarize(GURUE, host="h", fetched_at=None)
    assert summary.seasons_mentioned("is MZP2025UMB-PT01 there?") == ([], [])
    answer = summary.answer("studies since 2024?")
    assert "- Seasons you named: 2024 (3 studies)." in answer and "- Each season is counted alone: a range of seasons is not added up here." in answer
    assert "counted alone" not in summary.answer("studies in 2024?")


NAMED = KnownMetadata(base_url=BASE, complete=True, studies=[
    KnownStudy(study_id="A1", location_id="L1", location_name="Gurue", seasons=["2025"], study_type="Advanced Yield Trial"),
    KnownStudy(study_id="Y1", location_id="L1", location_name="Gurue", seasons=["2025"], study_type="Yield Trial"),
    KnownStudy(study_id="G1", location_id="L2", location_name="Gwebi Variety Testing Center", seasons=["2024"], study_type="Specialty trial"),
    KnownStudy(study_id="V1", location_id="L3", location_name="Clinton, NC", seasons=["2024"], study_type="Variety Release Trial")])


def test_a_count_without_a_named_place_says_it_is_for_all_locations():
    """Review finding (third round): "what about Uganda in 2024?" showed only 2024's count, which read as Uganda's."""
    answer = catalog_summary.summarize(NAMED, host="h", fetched_at=None).answer("what about Uganda in 2024?")
    assert ("- No location name contains a distinctive word of your message, so every count here is for all locations. The "
            "catalog stores only location names") in answer and "- Seasons you named: 2024 (2 studies)." in answer


def test_a_place_name_names_no_study_type_and_a_type_guessed_from_one_word_is_never_combined():
    """Review finding: "anything at Gwebi Variety Testing Center?" also named the type "Variety Release Trial" by its word
    "variety", and combined the two (0 studies)."""
    summary = catalog_summary.summarize(NAMED, host="h", fetched_at=None)
    answer = summary.answer("anything at Gwebi Variety Testing Center?")
    assert "Gwebi Variety Testing Center (1 study)" in answer and "Study types" not in answer and "together" not in answer
    guessed = summary.answer("advanced trials at Gurue?")                        # "advanced": a word, not a whole type name
    assert "Study types whose names contain a word of your message: Advanced Yield Trial (1 study)." in guessed
    assert "together" not in guessed


def test_a_type_inside_a_longer_type_is_not_also_named():
    """Review finding: with "Advanced Yield Trial" and "Yield Trial", "advanced yield trials" named both."""
    summary = catalog_summary.summarize(NAMED, host="h", fetched_at=None)
    assert summary.types_mentioned("advanced yield trials at Gurue in 2025?") == ([("Advanced Yield Trial", 1)], True)
    assert "together, studies at Gurue of type Advanced Yield Trial in 2025: 1 study." in summary.answer(
        "advanced yield trials at Gurue in 2025?")
    assert summary.types_mentioned("yield trials and advanced yield trials")[0] == [("Advanced Yield Trial", 1), ("Yield Trial", 1)]


@pytest.mark.parametrize("message", ["NC?", "NC 2025", "NC मा कति अध्ययन छन्?", "advanced yield trials in NC?"],
                         ids=["code-alone", "code-and-year", "other-script", "after-a-type-name"])
def test_a_message_without_lower_case_letters_is_not_taken_as_shouting(message):
    """Review finding: "NC?" or a message in another script no longer named "Clinton, NC"."""
    assert catalog_summary.summarize(NAMED, host="h", fetched_at=None).mentioned(message) == [("Clinton, NC", 1)]
