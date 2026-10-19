"""
tests/test_catalog_lists.py — "list all traits" (and locations, seasons, programs, study types), answered by code from the
cached catalog.

What these tests prove, in plain words:
* The five commands are found by code, with an optional filter word ("list traits with yield"); longer requests ("list traits
  for study 4501") are not commands and go to the planner as before.
* At the front door a command starts a short run: no front-door model call, no planner or other model call, no web request
  when the catalog is cached, no measurements. The chat shows the first 30 names and the total; the Full report and the
  run's answer.md keep every name. The run is recorded as completed with a fixed plan, the catalog table as its evidence,
  and acceptance pending (it asks no accept or reject).
* The filter word keeps only the names containing it, ignoring case. An unfiltered list longer than 30 says how to narrow
  it; a filtered or short list does not.
* Every list says its source and when that catalog was fetched (a saved copy, or fetched after the person approved the
  catalog setup). A list that could not get its evidence is not shown as a list.
* A filter word is matched against the names only (never an added ID); the heading says so. A suggested narrowing word
  always narrows the list.
* While a run waits at the planner's question, the same command is answered from that run's catalog: nothing reaches the
  run, it keeps waiting, and the hint under a trait question now names the command. A saved copy's cache reads are kept
  as the list run's request records, so the fetch time shown is backed by the record.
* A fifth list, study types, each with its number of studies, most first ("list all study types"; the help tab names it).
* The front door's catalog_list label (eval baseline 20261016T150241Z, q04-q06) starts the same list run for a question in
  other words: one label call, no planner, no talk call, no model-written answer. A label whose list or word fails the
  code check (the word must be one word of the person's own message) carries on as without a decision: the planner runs.
Fake server and scripted models only; the conftest guard would stop any real socket.
"""
from __future__ import annotations

import dataclasses
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

import app
import catalog_lists
import run as ctl
import triage
from agents.coordinator import KnownMetadata, KnownStudy, KnownVariable
from app_help import faq_items
from brapi_client import TransportResponse
from cache_store import CacheStore
from llm import FakeModelClient, reply_text
from part2_support import load_fixture, seed_fixture

LIVE = "https://lists.invalid/brapi/v2"
FETCHED = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)          # part2_support.FIXTURE_FETCHED_AT: when the seeded replies were "fetched"
SAVED_SOURCE = "Source: lists.invalid, trait catalog fetched 2026-06-01 00:00 UTC (a saved copy; nothing new was requested). No measurements were used."
YIELD_TRAITS = [f"{'Root Yield' if i % 2 else 'root yield'} {i:02d}" for i in range(1, 13)]     # 12, in both cases
HEIGHT_TRAITS = [f"Plant height {i:02d}" for i in range(1, 34)]                               # 33
ODD_TRAIT = "Weight [kg](http://x.invalid) <b>bold</b>"                                        # database text with markup
TRAITS = YIELD_TRAITS + HEIGHT_TRAITS + [ODD_TRAIT]                                          # 46


# --------------------------------------------------------------------------
# A synthetic catalog: 46 traits; 40 studies at 35 locations, in 37 seasons, in 2 programs
# --------------------------------------------------------------------------

def study_record(i: int) -> dict:
    location = i if i <= 35 else i - 35                                     # L01-L05 have two studies each
    record = {"studyDbId": f"S{i:03d}", "studyName": f"SYNTHETIC study {i}", "studyType": "Synthetic Yield Trial",
              "commonCropName": "sweetpotato", "active": False, "locationDbId": f"L{location:02d}",
              "locationName": f"Station {location:02d}",
              "seasons": [str(1990 + i)] if i <= 36 else (["Fall season"] if i == 37 else [])}
    if i <= 30:
        record.update(programDbId="PG1" if i <= 20 else "PG2", programName="Sweetpotato breeding" if i <= 20 else "Cassava program")
    return record


def variable_record(i: int, name: str) -> dict:
    return {"observationVariableDbId": f"V{i:03d}", "observationVariableName": name, "commonCropName": "sweetpotato",
            "status": "SYNTHETIC", "trait": {"traitDbId": f"T{i:03d}", "traitName": name},
            "method": {"methodDbId": f"M{i:03d}", "methodName": "synthetic method"},
            "scale": {"scaleDbId": f"SC{i:03d}", "scaleName": "unit", "units": "unit", "dataType": "Numerical"}}


def write_fixture(folder: Path, name: str, endpoint: str, records: list[dict]) -> None:
    body = {"_note": "SYNTHETIC catalog-list fixture. Invented data for tests. Not real breeding data.", "_synthetic": True,
            "_request": {"endpoint": endpoint, "params": {"page": 0, "pageSize": 1000}},
            "metadata": {"pagination": {"currentPage": 0, "pageSize": 1000, "totalCount": len(records), "totalPages": 1},
                         "synthetic": True},
            "result": {"data": records}}
    (folder / name).write_text(json.dumps(body), encoding="utf-8")


class CatalogServer:
    """Serves only the two catalogs (for the catalog setup); a measurement request would fail the test."""

    REPLIES = {"/studies": "list_studies.json", "/variables": "list_variables.json"}

    def __init__(self, folder: Path):
        self.folder, self.calls = folder, []

    def __call__(self, req):
        endpoint = req.url[len(LIVE):]
        self.calls.append(endpoint)
        _, body = load_fixture(self.REPLIES[endpoint], self.folder)
        return TransportResponse(200, {"Content-Type": "application/json"}, body)


class Models:
    """Scripted models that must never be asked anything by a list run; each records every request."""

    def __init__(self, planner_replies: tuple[dict, ...] = ()):
        self.planner = FakeModelClient([reply_text(json.dumps(r), model="mock-model") for r in planner_replies])
        self.retriever, self.supervisor, self.analysts = FakeModelClient([]), FakeModelClient([]), []

    def __call__(self) -> ctl.Models:
        return ctl.Models(coordinator=self.planner, retriever=self.retriever, supervisor=self.supervisor,
                          analyst_factory=lambda prefix: self.analysts.append(prefix) or FakeModelClient([]), requested="mock-model")

    def asked(self) -> int:
        return len(self.planner.requests) + len(self.retriever.requests) + len(self.supervisor.requests) + len(self.analysts)


@pytest.fixture
def lists(tmp_path, monkeypatch):
    import brapi_client as bc

    monkeypatch.setenv("BRAPI_MODE", "live")
    monkeypatch.setenv("BRAPI_BASE_URL", LIVE)
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    write_fixture(fixtures, "list_studies.json", "/studies", [study_record(i) for i in range(1, 41)])
    write_fixture(fixtures, "list_variables.json", "/variables", [variable_record(i, name) for i, name in enumerate(TRAITS, 1)])
    server = CatalogServer(fixtures)
    monkeypatch.setattr(bc, "RequestsTransport", lambda: server)
    notes = tmp_path / "ACCESS_NOTES.md"
    notes.write_text("Reviewed by Bidhan on: 2026-09-28\n", encoding="utf-8")
    models = Models()
    front_door: list[object] = []

    def make_config(question, *, bootstrap=False):
        return ctl.RunConfig(question=question, offline=False, mock_model=False, direct=False, auto=False, out_dir=tmp_path / "out",
                             cache_dir=tmp_path / "live_cache", access_notes=notes, bootstrap_catalogs=bootstrap,
                             max_clarification_rounds=app.CHAT_CLARIFICATION_ROUNDS)

    def cache_the_catalog():
        cache = CacheStore(tmp_path / "live_cache")
        for name in ("list_studies.json", "list_variables.json"):
            seed_fixture(cache, LIVE, name, fixtures)

    def turn(message, history=(), state=None, *, bootstrap=False, models_=None, triage_model=None):
        config = lambda q: make_config(q, bootstrap=bootstrap)
        out = list(app.respond(message, list(history), dict(state or {}), make_config=config, make_models=models_ or models,
                               preflight=lambda q: config(q).validate(),
                               triage_model=triage_model or (lambda: front_door.append(1))))
        assert out[-1][2] == ""
        return out[-1][0], out[-1][1]

    def runs() -> list[ctl.RunResult]:
        return [ctl.RunResult.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted((tmp_path / "out").glob("run_*/answer.json"))]

    return dataclasses.make_dataclass("Lists", ["turn", "server", "models", "front_door", "runs", "cache_the_catalog", "out"])(
        turn, server, models, front_door, runs, cache_the_catalog, tmp_path / "out")


def shown_items(content: str) -> list[str]:
    return [line[2:] for line in content.splitlines() if line.startswith("- ")]


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------

@pytest.mark.parametrize("typed, entity, word", [
    ("list all traits", "traits", None), ("List ALL traits!", "traits", None), ("list traits", "traits", None),
    ("list traits with yield", "traits", "yield"), ("List traits with 'YIELD'?", "traits", "yield"),
    ("show me all the locations", "locations", None), ("can you list every season please", "seasons", None),
    ("please list the programs", "programs", None), ("could you please show programmes with sweet", "programs", "sweet"),
    ("show all variables", "traits", None), ("list all trait names", "traits", None), ("list traits containing dry-matter", "traits", "dry-matter"),
    ("list locations that contain n.c.", "locations", "n.c"), ("list seasons with 2025", "seasons", "2025"),
    ("list the available traits in the catalog", "traits", None), ("show me a list of all traits", "traits", None),
    ("give me a list of traits with yield", "traits", "yield"), ("a list of all locations", "locations", None),
    ("can you give me the list of seasons?", "seasons", None), ("list all study types", "study_types", None),
    ("show me the study types with yield", "study_types", "yield"), ("list study type names", "study_types", None),
])
def test_the_five_commands_and_the_filter_word_are_found_by_code(typed, entity, word):
    assert catalog_lists.parse_request(typed) == catalog_lists.ListRequest(entity, word)


@pytest.mark.parametrize("typed", ["the yield trait", "list traits for study 4501", "list traits with yield in Clinton",
                                   "list traits with yield and maturity", "list all studies", "which traits are there?", "list",
                                   "list traits with", "show me the yield trait", "list the locations in Uganda",
                                   "traits", "list traits with " + "y" * 61, "I want to list all traits"])
def test_anything_else_is_not_a_list_command(typed):
    assert catalog_lists.parse_request(typed) is None


@pytest.mark.parametrize("entity, word, problem", [
    ("plots", None, "catalog_list must be one of traits, locations, seasons, programs, study_types, studies"),
    ("studies", None, "catalog_list studies needs catalog_list_word"),
    ("traits", "two words", "catalog_list_word must be one word"), ("traits", "", "catalog_list_word must be one word"),
    (None, "yield", "catalog_list_word needs catalog_list")])
def test_a_list_run_names_one_list_and_at_most_one_word(tmp_path, entity, word, problem):
    config = ctl.RunConfig(question="list", catalog_list=entity, catalog_list_word=word)
    assert any(p.startswith(problem) for p in config.validate())
    assert not any("catalog_list" in p for p in ctl.RunConfig(question="list", catalog_list="traits", catalog_list_word="yield").validate())


# --------------------------------------------------------------------------
# At the front door: a short run of its own
# --------------------------------------------------------------------------

def test_list_all_traits_shows_30_names_the_total_how_to_narrow_and_the_source(lists):
    lists.cache_the_catalog()
    history, state = lists.turn("list all traits")
    content = history[-1]["content"]
    assert content.startswith("**Traits in the catalog: 46.** The first 30, A to Z:")
    names = shown_items(content)
    assert len(names) == 31 and names[-1] == "… and 16 more in the Full report below."
    assert names[:30] == sorted(TRAITS, key=str.casefold)[:30]                  # A to Z, ignoring case
    assert "To see fewer, add a word: **list traits with yield** shows only the names that contain “yield”." in content
    assert content.endswith(SAVED_SOURCE)
    assert lists.server.calls == [] and lists.front_door == [] and lists.models.asked() == 0   # no request, no model at all
    assert "job" not in state and [m["role"] for m in history] == ["user", "assistant"]
    report = app.full_report(state)
    for name in TRAITS:                                                           # every name, with its ID, in the report
        assert f"- {name} · V" in report
    [result] = lists.runs()
    assert result.execution_status == "completed" and result.human_acceptance == "pending"
    assert result.counts.model_calls == 0 and result.counts.live_attempts == 0 and result.manifest is None and result.approval_id is None
    assert [(s.agent, s.action) for s in result.plan.steps] == [("retriever", "export_metadata"), ("analyst", "list_catalog_names")]
    assert [a.kind for a in result.retrieval.artifacts] == ["variables"] and result.analysis.claims[0].value == 46
    assert result.analysis.claims[0].source_artifact_hashes == [result.retrieval.artifacts[0].sha256]
    assert "catalog list: 46 of 46 traits, read by code from the trait catalog of https://lists.invalid/brapi/v2" in result.notes[0]
    assert "acceptance is not asked for a catalog list; it stays pending" in result.notes
    record = (lists.out / result.run_id / "answer.md").read_text(encoding="utf-8")
    assert "## Catalog list (written by code from the cached catalog)" in record and "- Plant height 33 · V045" in record
    manifest = json.loads((lists.out / result.run_id / "manifest.json").read_text(encoding="utf-8"))
    saved = {r["request_id"]: r for r in manifest["requests"]}                   # the cache reads behind the shown fetch time
    assert set(result.retrieval.request_ids) <= set(saved) and all(r["origin"] == "cache" for r in saved.values())
    assert {saved[i]["fetched_at_utc"] for i in result.retrieval.request_ids} == {"2026-06-01T00:00:00Z"}
    assert result.counts.cached_requests == len(saved) and manifest["request_ids"] == list(saved)   # counted once


def test_a_filter_word_keeps_only_the_names_containing_it_ignoring_case(lists):
    lists.cache_the_catalog()
    history, _ = lists.turn("List traits with YIELD")
    content = history[-1]["content"]
    assert content.startswith("**Trait names containing “yield”: 12 of 46.** A to Z:")
    assert shown_items(content) == sorted(YIELD_TRAITS, key=str.casefold)       # "Root Yield" and "root yield" alike
    assert "To see fewer" not in content and "more" not in content and content.endswith(SAVED_SOURCE)
    [result] = lists.runs()
    assert result.analysis.claims[0].value == 12 and result.analysis.claims[0].filters == {"name contains": "yield"}
    assert result.plan.steps[1].inputs == {"entity": "traits", "contains": "yield"} and lists.models.asked() == 0


def test_locations_seasons_and_programs_come_from_the_study_catalog_with_study_counts(lists):
    lists.cache_the_catalog()
    history, _ = lists.turn("show me all the locations")
    locations = history[-1]["content"]
    assert locations.startswith("**Locations in the catalog: 35.** The first 30, A to Z:")
    assert shown_items(locations)[:2] == ["Station 01 — 2 studies", "Station 02 — 2 studies"] and "- Station 06 — 1 study" in locations
    assert "To see fewer, add a word: **list locations with** and a word from the names." in locations   # "station" is in all 35
    assert "study catalog fetched 2026-06-01 00:00 UTC (a saved copy; nothing new was requested)" in locations
    history, _ = lists.turn("list seasons")
    seasons = history[-1]["content"]
    assert seasons.startswith("**Seasons in the catalog: 37.** The first 30, newest first:")
    assert shown_items(seasons)[:3] == ["2026 — 1 study", "2025 — 1 study", "2024 — 1 study"]
    assert "3 studies name no season." in seasons and "**list seasons with 2026**" in seasons            # 2026 narrows: 1 of 37
    history, _ = lists.turn("list programs")
    programs = history[-1]["content"]
    assert shown_items(programs) == ["Cassava program — 10 studies", "Sweetpotato breeding — 20 studies"]
    assert "10 studies name no program." in programs and "To see fewer" not in programs
    history, _ = lists.turn("list programs with SWEET")
    assert shown_items(history[-1]["content"]) == ["Sweetpotato breeding — 20 studies"]
    history, _ = lists.turn("list locations with yield")                     # names only: not "locations with yield data"
    assert history[-1]["content"].startswith("**Location names containing “yield”: none of 35.** Only the names are searched; "
                                             "try another word, or ask **list all locations**.")
    assert lists.server.calls == [] and lists.models.asked() == 0 and len(lists.runs()) == 5


def test_without_a_cached_catalog_the_catalog_setup_comes_first_then_the_list_says_when_it_was_fetched(lists):
    history, state = lists.turn("list traits with yield", bootstrap=True)
    assert "CATALOG SETUP APPROVAL NEEDED" in history[-1]["content"] and state["job"].waiting_on == ctl.CATALOG_PROMPT
    history, state = lists.turn("list traits with yield", history, state, bootstrap=True)    # no catalog yet: a new question
    assert history[-1]["content"].startswith("That sounds like a new question: “list traits with yield”. Reply **drop**")
    history, state = lists.turn("keep", history, state, bootstrap=True)
    assert "CATALOG SETUP APPROVAL NEEDED" in history[-1]["content"] and lists.server.calls == []
    history, state = lists.turn("approve", history, state, bootstrap=True)
    content = history[-1]["content"]
    assert content.startswith("**Trait names containing “yield”: 12 of 46.**") and "job" not in state
    assert re.search(r"Source: lists\.invalid, trait catalog fetched \d{4}-\d\d-\d\d \d\d:\d\d UTC \(fetched after you approved the "
                     r"catalog setup\)\.", content)
    assert lists.server.calls == ["/studies", "/variables"]                     # the two catalogs only: no measurements
    [result] = lists.runs()
    assert result.execution_status == "completed" and result.catalog_approval is not None and result.counts.model_calls == 0
    assert lists.models.asked() == 0 and lists.front_door == []


# --------------------------------------------------------------------------
# While a run waits: from that run's own catalog
# --------------------------------------------------------------------------

def test_at_the_planners_trait_question_the_hint_works_and_the_run_keeps_waiting(lists):
    lists.cache_the_catalog()
    models = Models(({"status": "needs_clarification", "question": "Which trait is meant?"},))
    history, state = lists.turn("Describe the trial at Station 01", models_=models)
    asked = history[-1]["content"]
    assert "Which trait is meant?" in asked and catalog_lists.TRAIT_HINT in asked and state["job"].waiting_on == ctl.CLARIFY_PROMPT
    job = state["job"]
    history, state = lists.turn("list traits with yield", history, state, models_=models)
    content = history[-1]["content"]
    assert content.startswith("**Trait names containing “yield”: 12 of 46.** A to Z:") and SAVED_SOURCE in content
    assert content.endswith("I'm still waiting for your answer to my question (or reply **cancel**): “Which trait is meant?”")
    assert state["job"] is job and job.waiting_on == ctl.CLARIFY_PROMPT and job.controller.clarification_answers == 0
    assert lists.front_door == [1] and len(models.planner.requests) == 1          # the question asked the front door once; the list asked no model
    history, state = lists.turn("list all traits", history, state, models_=models)
    assert "- … and 16 more." in history[-1]["content"] and "Full report" not in history[-1]["content"]   # this run's report is not a list
    history, state = lists.turn("cancel", history, state, models_=models)
    [result] = lists.runs()
    assert result.execution_status == "canceled" and "list traits" not in json.dumps(result.notes)   # never an answer
    assert result.counts.model_calls == 1 and lists.server.calls == [] and result.plan.question == "Describe the trial at Station 01"


# --------------------------------------------------------------------------
# The list itself (pure functions)
# --------------------------------------------------------------------------

def table(export: str, *, fetched: datetime | None = FETCHED, live: bool = False) -> catalog_lists.CatalogTable:
    return catalog_lists.CatalogTable(export=export, artifact_id="art_abcdef_0001" if export == "studies" else "art_abcdef_0002",
                                      sha256="a" * 64, rows=3, base_url=LIVE, fetched_at=fetched, fetched_in_this_run=live)


TABLES = {"studies": table("studies"), "variables": table("variables")}
SMALL = KnownMetadata(base_url=LIVE, complete=True,
                      studies=[KnownStudy(study_id="S1", location_id="L1", location_name="Clinton", seasons=["2025"]),
                               KnownStudy(study_id="S2", location_id="L2", location_name="Clinton", seasons=["2024", "2025"]),
                               KnownStudy(study_id="S3", location_name="Kinston", program_name="NCSU")],
                      variables=[KnownVariable(variable_id="V2", name="Fresh root yield"), KnownVariable(variable_id="V1", name="fresh root yield"),
                                 KnownVariable(variable_id="V3", name="", trait="dry matter"), KnownVariable(variable_id="V4")])


@pytest.mark.parametrize("entity", catalog_lists.ENTITIES)
@pytest.mark.parametrize("word", [None, "root", "nothing-matches"])
@pytest.mark.parametrize("in_report", [True, False])
def test_every_list_answer_states_the_source_and_when_the_catalog_was_fetched(entity, word, in_report):
    message = catalog_lists.chat_message(catalog_lists.build(catalog_lists.ListRequest(entity, word), SMALL, TABLES), in_report=in_report)
    catalog = "trait catalog" if entity == "traits" else "study catalog"
    assert message.endswith(f"Source: lists.invalid, {catalog} fetched 2026-06-01 00:00 UTC (a saved copy; nothing new was "
                            "requested). No measurements were used.")
    report = "\n".join(catalog_lists.report_lines(catalog_lists.build(catalog_lists.ListRequest(entity, word), SMALL, TABLES)))
    assert f"Source: {LIVE}, {catalog}" in report and "fetched 2026-06-01 00:00 UTC" in report and "sha256 " + "a" * 64 in report


def test_the_source_line_says_when_the_catalog_was_fetched_after_the_setup_approval_or_that_the_time_is_unknown():
    request = catalog_lists.ListRequest("traits")
    fresh = catalog_lists.chat_message(catalog_lists.build(request, SMALL, {"variables": table("variables", live=True)}), in_report=True)
    assert fresh.endswith("trait catalog fetched 2026-06-01 00:00 UTC (fetched after you approved the catalog setup). "
                          "No measurements were used.")              # also right when another run's setup fetched it
    unknown = catalog_lists.chat_message(catalog_lists.build(request, SMALL, {"variables": table("variables", fetched=None)}), in_report=True)
    assert "trait catalog fetched at a time that was not recorded (a saved copy" in unknown


def test_same_names_stay_apart_by_id_and_a_missing_name_says_so():
    traits = catalog_lists.build(catalog_lists.ListRequest("traits"), SMALL, TABLES)
    assert [i.label for i in traits.items] == ["(no name) · V4", "dry matter", "fresh root yield · V1", "Fresh root yield · V2"]
    locations = catalog_lists.build(catalog_lists.ListRequest("locations"), SMALL, TABLES)
    assert [(i.label, i.studies) for i in locations.items] == [("Clinton · L1", 1), ("Clinton · L2", 1), ("Kinston", 1)]
    seasons = catalog_lists.build(catalog_lists.ListRequest("seasons"), SMALL, TABLES)
    assert [(i.label, i.studies) for i in seasons.items] == [("2025", 2), ("2024", 1)] and seasons.unrecorded == 1
    programs = catalog_lists.build(catalog_lists.ListRequest("programs"), SMALL, TABLES)
    assert [i.label for i in programs.items] == ["NCSU"] and programs.unrecorded == 2
    assert "2 studies name no program." in catalog_lists.chat_message(programs, in_report=True)


def test_a_long_filtered_list_shows_30_and_the_rest_but_no_narrowing_line():
    metadata = KnownMetadata(base_url=LIVE, complete=True, variables=[KnownVariable(variable_id=f"V{i}", name=f"Plant height {i:02d}")
                                                                      for i in range(1, 41)])
    message = catalog_lists.chat_message(catalog_lists.build(catalog_lists.ListRequest("traits", "plant"), metadata, TABLES), in_report=False)
    assert message.startswith("**Trait names containing “plant”: 40 of 40.** The first 30, A to Z:")
    assert len(shown_items(message)) == 31 and "- … and 10 more." in message and "To see fewer" not in message
    short = catalog_lists.chat_message(catalog_lists.build(catalog_lists.ListRequest("traits"), SMALL, TABLES), in_report=True)
    assert "To see fewer" not in short and "more" not in short and len(shown_items(short)) == 4
    none = catalog_lists.chat_message(catalog_lists.build(catalog_lists.ListRequest("traits", "zzz"), SMALL, TABLES), in_report=True)
    assert none.startswith("**Trait names containing “zzz”: none of 4.** Only the names are searched; try another word, or ask "
                           "**list all traits**.")


def test_database_names_and_the_persons_word_are_shown_as_text():
    metadata = KnownMetadata(base_url=LIVE, complete=True, variables=[KnownVariable(variable_id="V1", name=ODD_TRAIT)])
    message = catalog_lists.chat_message(catalog_lists.build(catalog_lists.ListRequest("traits", "<b>"), metadata, TABLES), in_report=True)
    assert "<b>" not in message and "](" not in message and "&lt;b&gt;" in message
    whole = catalog_lists.chat_message(catalog_lists.build(catalog_lists.ListRequest("traits"), metadata, TABLES), in_report=True)
    assert "](http" not in whole and "&#91;kg&#93;" in whole


def test_if_the_catalog_tables_cannot_be_read_a_list_says_so_and_other_questions_run_as_before(lists, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("synthetic: the provenance could not be read")

    monkeypatch.setattr(catalog_lists, "catalog_tables", broken)
    lists.cache_the_catalog()
    history, state = lists.turn("list all traits")
    assert history[-1]["content"].startswith("**Could not complete this question.**") and "job" not in state
    [result] = lists.runs()
    assert result.execution_status == "failed" and lists.models.asked() == 0
    assert "catalog lists unavailable in this run: OSError: synthetic: the provenance could not be read" in result.notes
    assert "catalog list: the catalog tables behind the metadata are unknown; no list was written" in result.notes
    models = Models(({"status": "needs_clarification", "question": "Which trait is meant?"},))
    history, state = lists.turn("Describe the trial at Station 01", models_=models)     # a data question still runs as before
    assert "Which trait is meant?" in history[-1]["content"] and state["job"].waiting_on == ctl.CLARIFY_PROMPT
    history, state = lists.turn("list traits with yield", history, state, models_=models)
    assert history[-1]["content"].startswith("That sounds like a new question")        # no tables: a new question, as in 2a
    waiting = state["job"]
    lists.turn("drop", history, state, models_=models)                                   # the waiting run ends, the list run starts
    waiting.thread.join(timeout=60)                                                       # its record is written by its own thread
    assert len(lists.runs()) == 3 and lists.server.calls == []


def test_study_based_lists_say_what_they_cover_and_empty_program_fields_are_explained():
    """SweetPotatoBase (offline smoke on the real cache, 2026-10-15): all 2039 studies leave the standard program fields
    empty (the names sit in additionalInfo, which the app does not use for programs); 101 locations are named by studies."""
    no_programs = KnownMetadata(base_url=LIVE, complete=True, studies=[KnownStudy(study_id=f"S{i}", location_id="L1",
                                                                                  location_name="Clinton") for i in range(3)])
    programs = catalog_lists.chat_message(catalog_lists.build(catalog_lists.ListRequest("programs"), no_programs, TABLES), in_report=True)
    assert programs.startswith("**Programs in the catalog: none.**\n\n3 studies name no program. Some servers keep the program only "
                               "in each study's additional information (additionalInfo); this app does not use that field for programs.")
    assert "Only programs named" not in programs and programs.endswith("No measurements were used.")
    locations = catalog_lists.chat_message(catalog_lists.build(catalog_lists.ListRequest("locations"), no_programs, TABLES), in_report=True)
    assert "- Clinton — 3 studies\n\nOnly locations named by at least one study are listed.\n\nSource:" in locations
    report = "\n".join(catalog_lists.report_lines(catalog_lists.build(catalog_lists.ListRequest("programs"), no_programs, TABLES)))
    assert "Programs come from the study records. 3 studies name no program. Some servers keep the program only" in report
    assert "(no matching names)" in report


def test_a_list_without_its_evidence_is_not_shown_as_a_list(lists, monkeypatch):
    """Review 2026-10-15: the list was set before its evidence, so a failed run still showed it as if it had worked."""
    def broken(*args, **kwargs):
        raise ValueError("synthetic: the catalog table behind this list changed")

    monkeypatch.setattr(catalog_lists, "evidence", broken)
    lists.cache_the_catalog()
    history, state = lists.turn("list all traits")
    content = history[-1]["content"]
    assert content.startswith("**Could not complete this question.**") and "Traits in the catalog" not in content
    [result] = lists.runs()
    assert result.execution_status == "failed" and "job" not in state
    assert "## Catalog list" not in app.full_report(state)                    # nor in the failed run's report (answer.md)
    assert "## Catalog list" not in (lists.out / result.run_id / "answer.md").read_text(encoding="utf-8")
    from types import SimpleNamespace                                          # and a list whose record failed later on
    listing = catalog_lists.build(catalog_lists.ListRequest("traits"), SMALL, TABLES)
    controller = SimpleNamespace(catalog_list=listing, execution_status="failed", analysis=None, retrieval=None, notes=["x failed"])
    shown, finished = app.render(SimpleNamespace(controller=controller, answer_shown=False),
                                 ("done", SimpleNamespace(execution_status="failed", human_acceptance="pending")))
    assert finished and "Traits in the catalog" not in shown and shown.startswith("**Could not complete this question.**")


def test_the_filter_word_is_matched_against_the_name_only_never_an_added_id():
    """Review 2026-10-15: duplicate names get their ID added for display, and a nameless trait shows "(no name)"; neither
    may match a filter word, or the count would claim names that do not contain it."""
    metadata = KnownMetadata(base_url=LIVE, complete=True, variables=[
        KnownVariable(variable_id="77001", name="Plant height"), KnownVariable(variable_id="77002", name="plant height"),
        KnownVariable(variable_id="77003", name="Root yield"), KnownVariable(variable_id="77004")])
    assert [i.label for i in catalog_lists.build(catalog_lists.ListRequest("traits"), metadata, TABLES).items] == [
        "(no name) · 77004", "Plant height · 77001", "plant height · 77002", "Root yield"]
    for word in ("770", "name", "·"):
        assert catalog_lists.build(catalog_lists.ListRequest("traits", word), metadata, TABLES).items == ()
    height = catalog_lists.build(catalog_lists.ListRequest("traits", "height"), metadata, TABLES)
    assert [i.key for i in height.items] == ["77001", "77002"] and height.total == 4


def test_the_suggested_narrowing_word_always_narrows_the_list():
    """Review 2026-10-15: 35 locations all named "Station NN" got "list locations with station", which keeps all 35."""
    studies = [KnownStudy(study_id=f"S{i}", location_id=f"L{i}", location_name=f"Station {i:02d}") for i in range(35)]
    studies += [KnownStudy(study_id=f"T{i}", location_id="L99", location_name="Clinton Research Station") for i in range(5)]
    metadata = KnownMetadata(base_url=LIVE, complete=True, studies=studies)
    message = catalog_lists.chat_message(catalog_lists.build(catalog_lists.ListRequest("locations"), metadata, TABLES), in_report=True)
    assert "To see fewer, add a word: **list locations with clinton** shows only the names that contain “clinton”." in message
    narrowed = catalog_lists.build(catalog_lists.ListRequest("locations", "clinton"), metadata, TABLES)
    assert 0 < len(narrowed.items) < narrowed.total
    alike = KnownMetadata(base_url=LIVE, complete=True, studies=studies[:35])                 # no word keeps fewer than all
    plain = catalog_lists.chat_message(catalog_lists.build(catalog_lists.ListRequest("locations"), alike, TABLES), in_report=True)
    assert "To see fewer, add a word: **list locations with** and a word from the names." in plain


# --------------------------------------------------------------------------
# Study types: a fifth list (eval baseline 20261016T150241Z, q06: "How many studies are there of each study type?")
# --------------------------------------------------------------------------

TYPED = KnownMetadata(base_url=LIVE, complete=True, studies=[
    *[KnownStudy(study_id=f"A{i}", study_type="Advanced Yield Trial") for i in range(3)],
    *[KnownStudy(study_id=f"C{i}", study_type="Clonal Evaluation") for i in range(2)],
    *[KnownStudy(study_id=f"T{i}", study_type="Specialty Trial") for i in range(2)],
    KnownStudy(study_id="t1", study_type="Specialty trial"), KnownStudy(study_id="N1")])


def test_study_types_are_listed_most_studies_first_each_name_as_the_catalog_writes_it():
    listing = catalog_lists.build(catalog_lists.ListRequest("study_types"), TYPED, TABLES)
    assert [(i.label, i.studies) for i in listing.items] == [("Advanced Yield Trial", 3), ("Clonal Evaluation", 2),
                                                             ("Specialty Trial", 2), ("Specialty trial", 1)]
    assert listing.total == 4 and listing.unrecorded == 1
    message = catalog_lists.chat_message(listing, in_report=True)
    assert message.startswith("**Study types in the catalog: 4.** Most studies first:\n\n- Advanced Yield Trial — 3 studies\n"
                              "- Clonal Evaluation — 2 studies\n- Specialty Trial — 2 studies\n- Specialty trial — 1 study\n")
    assert "Only study types named by at least one study are listed. 1 study names no study type." in message
    assert message.endswith("Source: lists.invalid, study catalog fetched 2026-06-01 00:00 UTC (a saved copy; nothing new was "
                            "requested). No measurements were used.")
    trial = catalog_lists.build(catalog_lists.ListRequest("study_types", "trial"), TYPED, TABLES)
    assert [i.label for i in trial.items] == ["Advanced Yield Trial", "Specialty Trial", "Specialty trial"]
    assert catalog_lists.chat_message(trial, in_report=True).startswith("**Study type names containing “trial”: 3 of 4.**")
    report = "\n".join(catalog_lists.report_lines(listing))
    assert "Study types in the catalog: 4, most studies first." in report and "- Specialty trial — 1 study" in report
    assert "Study types come from the study records." in report
    assert "study types" in dict(faq_items("x"))["What makes a good question?"]          # the help tab names the list


def test_list_all_study_types_typed_at_the_front_door_is_a_short_run_with_no_model(lists):
    lists.cache_the_catalog()
    history, state = lists.turn("list all study types")
    content = history[-1]["content"]
    assert content.startswith("**Study types in the catalog: 1.** Most studies first:\n\n- Synthetic Yield Trial — 40 studies")
    assert lists.front_door == [] and lists.models.asked() == 0 and lists.server.calls == [] and "job" not in state
    [result] = lists.runs()
    assert result.execution_status == "completed" and result.analysis.claims[0].value == 1
    assert result.plan.steps[1].inputs == {"entity": "study_types"}


# --------------------------------------------------------------------------
# The front door's catalog_list label: the same list run for a question in other words
# --------------------------------------------------------------------------

def list_label(entity, word="") -> dict:
    return {"label": "catalog_list", "reason": "scripted for this test", "reply": "", "list": entity, "word": word}


def front_door_model(*decisions: dict):
    """The front door's model, scripted: one JSON reply per call, and every request it was sent."""
    model = FakeModelClient([reply_text(json.dumps(d), model="mock-model") for d in decisions], reported_model="mock-model")
    return model, (lambda: model)


@pytest.mark.parametrize("question, decision, heading, catalog", [
    ("Which seasons are in the catalog?", list_label("seasons"),
     "**Seasons in the catalog: 37.** The first 30, newest first:\n\n- 2026 — 1 study", "study catalog"),
    ("How many studies are there of each study type?", list_label("study types"),
     "**Study types in the catalog: 1.** Most studies first:\n\n- Synthetic Yield Trial — 40 studies", "study catalog"),
    ("How many trait names contain the word yield?", list_label("traits", "yield"),
     "**Trait names containing “yield”: 12 of 46.** A to Z:", "trait catalog"),
], ids=["q05-seasons", "q06-study-types", "q04-trait-names-with-yield"])
def test_the_three_baseline_questions_reach_the_lists_written_by_code(lists, question, decision, heading, catalog):
    lists.cache_the_catalog()
    model, factory = front_door_model(decision)
    history, state = lists.turn(question, triage_model=factory)
    content = history[-1]["content"]
    assert content.startswith(heading) and f"Source: lists.invalid, {catalog} fetched 2026-06-01 00:00 UTC (a saved copy" in content
    assert len(model.requests) == 1 and lists.models.asked() == 0 and lists.server.calls == []    # one label call, nothing else
    assert "job" not in state and [m["role"] for m in history] == ["user", "assistant"]
    [result] = lists.runs()
    assert result.execution_status == "completed" and result.plan.question == question and result.counts.model_calls == 0
    assert [(s.agent, s.action) for s in result.plan.steps] == [("retriever", "export_metadata"), ("analyst", "list_catalog_names")]
    prompt = model.requests[0]["messages"][0]["content"]
    assert triage.CATALOG_LIST_LABEL in prompt and "Saved catalog of lists.invalid" in prompt


@pytest.mark.parametrize("decision", [
    list_label("plots"),                                # not one of the six lists
    list_label("studies"),                              # the studies list without its season
    list_label("traits", "potato"),                     # a word the person did not write (coach, 2026-10-16)
    list_label("traits", "yields"),                     # not as a whole word: the message says "yield"
    list_label("traits", "trait names"),                # two words
    list_label("traits", "yield?"),                     # not one word as the typed command reads one
    {**list_label("traits", "yield"), "list": None},    # no list named
], ids=["not-a-list", "studies-without-a-season", "word-not-in-the-message", "not-a-whole-word", "two-words", "not-one-word", "no-list"])
def test_a_list_label_whose_fields_fail_the_check_carries_on_as_without_a_decision(lists, decision):
    lists.cache_the_catalog()
    models = Models(({"status": "needs_clarification", "question": "Which trait is meant?"},))
    model, factory = front_door_model(decision)
    history, state = lists.turn("How many trait names contain the word yield?", models_=models, triage_model=factory)
    job = state["job"]                                  # the run started as before: its planner asks a question
    assert job.waiting_on == ctl.CLARIFY_PROMPT and "Which trait is meant?" in history[-1]["content"]
    assert job.controller.config.catalog_list is None and len(models.planner.requests) == 1 and len(model.requests) == 1
    lists.turn("cancel", history, state, models_=models)
    job.thread.join(timeout=60)
    assert [r.execution_status for r in lists.runs()] == ["canceled"] and lists.server.calls == []


# --------------------------------------------------------------------------
# The studies of one season (coach check 3, 2026-10-19): exact season membership, written by code
# --------------------------------------------------------------------------

@pytest.mark.parametrize("typed, season", [
    ("list studies in season 2026", "2026"), ("show me all studies with a 2026 season", "2026"),
    ("list the studies from 2026", "2026"), ("list studies with season wet-2025", "wet-2025"),
    ("can you list studies that have season fall?", "fall"), ("List Studies In Season 2026 please", "2026"),
    ("give me a list of studies for the season 2019", "2019"),
])
def test_the_studies_of_one_season_are_a_list_command(typed, season):
    assert catalog_lists.parse_request(typed) == catalog_lists.ListRequest("studies", season)


@pytest.mark.parametrize("typed", ["list studies with yield", "list all studies", "list studies in Clinton", "list studies from season",
                                   "list studies in season 2026 at Clinton", "list studies in 2026 and 2025"])
def test_a_studies_list_needs_a_season_named_as_one(typed):
    assert catalog_lists.parse_request(typed) is None


SEASONED = KnownMetadata(base_url=LIVE, complete=True, studies=[
    KnownStudy(study_id="S1", name="Beta trial", seasons=["2026"]),
    KnownStudy(study_id="S2", name="Alpha trial", seasons=["2025", "2026"]),
    KnownStudy(study_id="S3", name="Not this one", seasons=["12026"]),
    KnownStudy(study_id="S4", name="Nor this", seasons=["2026 rainy"]),
    KnownStudy(study_id="S5", name="No season"),
    KnownStudy(study_id="S6", name="Alpha trial", seasons=["2026"]),
    KnownStudy(study_id="S7", name="Fall one", seasons=["Fall"])])


def studies_of(season, metadata=SEASONED):
    return catalog_lists.build(catalog_lists.ListRequest("studies", season), metadata, TABLES)


def test_a_study_is_listed_only_when_one_of_its_seasons_is_exactly_that_season():
    listing = studies_of("2026")
    assert [item.key for item in listing.items] == ["S2", "S6", "S1"]           # A to Z; "12026" and "2026 rainy" are other seasons
    assert listing.total == 7 and listing.unrecorded == 1 and listing.source == TABLES["studies"]
    fall = studies_of("fall")                                                     # the word, read in lower case, finds "Fall"
    assert [item.key for item in fall.items] == ["S7"] and fall.season == "Fall"
    assert catalog_lists.chat_message(fall, in_report=True).startswith("**Studies with season “Fall”: 1 of the 7")
    message = catalog_lists.chat_message(listing, in_report=True)
    assert message.startswith("**Studies with season “2026”: 3 of the 7 studies in the catalog.** A to Z:\n\n"
                              "- Alpha trial · S2\n- Alpha trial · S6\n- Beta trial\n")
    assert "A study is listed when one of its recorded seasons is exactly “2026”. 1 study names no season." in message
    report = "\n".join(catalog_lists.report_lines(listing))
    assert 'Studies with season "2026": 3 of 7, A to Z.' in report and "- Beta trial · S1" in report


def test_no_exact_season_offers_the_season_names_that_contain_the_word():
    message = catalog_lists.chat_message(studies_of("202"), in_report=True)
    assert message.startswith("**Studies with season “202”: none of the 7 studies in the catalog.**\n\n")
    assert "Season names that contain “202”: 2026, 2026 rainy, 2025, 12026." in message     # newest first, as the seasons list
    assert "No season name contains it" in catalog_lists.chat_message(studies_of("1880"), in_report=False)
    assert "(no matching studies)" in catalog_lists.report_lines(studies_of("1880"))


def test_the_list_and_the_count_of_one_season_agree_on_the_same_rows():
    """Review 2026-10-19: the list ignored case and spaces, the count (catalog_answers) matches the recorded text exactly.
    Now the list matches one recorded name exactly too; the same rows give the same studies."""
    import json as _json
    import catalog_answers
    rows = [{"studyDbId": f"S{i}", "studyName": f"T{i}", "seasons": _json.dumps(seasons)} for i, seasons in enumerate(
        [["2026"], ["2025", "2026"], [" 2026"], ["12026"], ["Fall"], ["FALL"], ["fall"], ["2026 rainy"], ["Wet"]])]
    metadata = KnownMetadata.from_rows(rows, [], base_url=LIVE, complete=True)
    for season in ("2026", "2025", "Wet", " 2026"):
        listed = {item.key for item in studies_of(season.strip().lower(), metadata).items}
        recorded = catalog_lists.season_names(metadata, season.strip().lower())
        counted = catalog_answers._exact_season_ids(rows, recorded[0]) if len(recorded) == 1 else set()
        assert listed == counted, season
    assert {item.key for item in studies_of("2026", metadata).items} == {"S0", "S1"}      # " 2026" is another recorded text


def test_seasons_that_differ_only_in_capitals_are_named_and_not_merged():
    metadata = KnownMetadata(base_url=LIVE, complete=True, studies=[
        KnownStudy(study_id="S1", name="One", seasons=["Fall"]), KnownStudy(study_id="S2", name="Two", seasons=["FALL"])])
    listing = studies_of("fall", metadata)
    assert listing.items == () and listing.hints == ("FALL", "Fall")
    message = catalog_lists.chat_message(listing, in_report=True)
    assert "More than one recorded season is written this way: FALL, Fall." in message and "- One" not in message


@pytest.mark.parametrize("typed", ["list studies with a season", "list studies in the season", "list studies in season please",
                                   "list studies with season the", "list studies with any season"])
def test_a_filler_word_is_never_read_as_the_season(typed):
    assert catalog_lists.parse_request(typed) is None


def test_a_long_season_list_shows_30_in_the_chat_and_every_study_in_the_report():
    many = KnownMetadata(base_url=LIVE, complete=True,
                         studies=[KnownStudy(study_id=f"S{i:02d}", name=f"Trial {i:02d}", seasons=["2026"]) for i in range(1, 36)])
    listing = studies_of("2026", many)
    message = catalog_lists.chat_message(listing, in_report=True)
    assert "35 of the 35 studies in the catalog.** The first 30, A to Z:" in message
    assert len(shown_items(message)) == 31 and "- … and 5 more in the Full report below." in message
    assert "To see fewer" not in message
    assert sum(line.startswith("- Trial ") for line in catalog_lists.report_lines(listing)) == 35


def test_the_studies_list_claim_and_plan_name_the_season():
    listing = studies_of("2026")
    plan = catalog_lists.fixed_plan(listing.request, "Which studies have a 2026 season?", LIVE, FETCHED)
    assert plan.steps[1].inputs == {"entity": "studies", "season": "2026"} and "'2026' exactly" in plan.interpretation
    note = catalog_lists.run_note(listing)
    assert note.startswith("catalog list: 3 of 7 studies with season '2026'")


def test_list_studies_in_a_season_typed_at_the_front_door_is_a_short_run_with_no_model(lists):
    lists.cache_the_catalog()
    history, state = lists.turn("list studies in season 2026")
    content = history[-1]["content"]
    assert content.startswith("**Studies with season “2026”: 1 of the 40 studies in the catalog.** A to Z:\n\n- SYNTHETIC study 36\n")
    assert lists.front_door == [] and lists.models.asked() == 0 and lists.server.calls == [] and "job" not in state
    [result] = lists.runs()
    assert result.execution_status == "completed" and result.analysis.claims[0].value == 1
    assert result.analysis.claims[0].filters == {"season": "2026"} and result.analysis.claims[0].denominator.n == 40
    assert result.plan.steps[1].inputs == {"entity": "studies", "season": "2026"}
    [report] = lists.out.glob("run_*/answer.md")
    assert "- SYNTHETIC study 36 · S036" in report.read_text(encoding="utf-8")


def test_which_studies_have_a_season_reaches_the_studies_list_through_the_label(lists):
    lists.cache_the_catalog()
    model, factory = front_door_model(list_label("studies", "2026"))
    history, state = lists.turn("Which studies have a 2026 season?", triage_model=factory)
    assert history[-1]["content"].startswith("**Studies with season “2026”: 1 of the 40 studies in the catalog.**")
    assert len(model.requests) == 1 and lists.models.asked() == 0 and lists.server.calls == [] and "job" not in state
    assert triage.catalog_list_request(triage.TriageDecision.model_validate(list_label("studies", "2027")),
                                       "Which studies have a 2026 season?") is None              # a season they did not write


@pytest.mark.parametrize("message, accepted", [
    ("Which studies have a 2026 season?", True), ("which studies are in season 2026?", True),
    ("Show me the studies from the 2026 season", True), ("Can you list all studies that had a 2026 season, please?", True),
    ("Which studies do not have a 2026 season?", False), ("Which studies don't have a 2026 season?", False),
    ("Which studies had a 2026 season in Clinton?", False), ("Which studies have a 2025 or 2026 season?", False),
    ("Which 2026 studies measure yield?", False), ("Which studies without a 2026 season?", False),
])
def test_the_studies_label_is_taken_only_for_the_plain_list_question(message, accepted):
    """Review 2026-10-19: the label check only looked for the season word, so a negation, a place or a second season
    reached the plain list (a wrong list). Now every other word must be a plain list word."""
    request = triage.catalog_list_request(triage.TriageDecision.model_validate(list_label("studies", "2026")), message)
    assert (request == catalog_lists.ListRequest("studies", "2026")) is accepted
