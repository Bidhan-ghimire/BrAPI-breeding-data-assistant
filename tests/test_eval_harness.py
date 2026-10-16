"""
tests/test_eval_harness.py — the answer-quality harness (eval/harness.py, overnight Item 4, 2026-10-16), offline.

What these tests prove, in plain words:
* Without --confirm nothing runs: no results folder, no model, no request.
* The expected answers are computed by code from the raw catalog records and the cached measurements (seasons as words,
  "Fall 2025" counts for 2025; trait names without their ontology suffix; ties; a study or trait the catalog does not have
  gives no expected answer, so that question is skipped, never run).
* compare(): a whole number in the answer or a typed claim; "2018" is not 18; "2,039" is 2039; names ignore case and
  markdown escapes; a mean rounds within the decimals shown, and an integer alone never passes for a mean.
* End to end, with scripted models and the app's own pipeline: a catalog question answered by code passes; a measurement
  question passes through "approve" (data request) and "continue" (data review) only, and its run is left pending, never
  accepted; an "unsure" message fails (the harness never says yes); results.json and results.md are written; runs are
  recorded under the results folder, on a copy of the cache: the source cache is unchanged and nothing was requested.
* The scripted approver never approves a catalog setup and never answers a question the app asks.
* By default a request for anything not in the cache copy is refused and counted; --allow-live lets the app fetch it.
Fake transport and scripted models only; the conftest guard would stop any real socket.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import app
import catalog_summary
import run as ctl
from brapi_client import TransportResponse
from cache_store import CacheStore
from llm import FakeModelClient, reply_text
from part2_support import load_fixture, part2_runtime, seed_fixture, synthetic_models

HARNESS = Path(__file__).resolve().parents[1] / "eval" / "harness.py"
_spec = importlib.util.spec_from_file_location("_eval_harness_under_test", HARNESS)
harness = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = harness                      # dataclasses read their module while the file is loaded
_spec.loader.exec_module(harness)

LIVE_BASE = "https://eval.invalid/brapi/v2"
MEASURED = "For synthetic study S1, give mean yield and clone means with n."
FIXTURES = ("synthetic_studies.json", "synthetic_variables.json", "synthetic_observations_S1_V1.json",
            "synthetic_observationunits_S1.json")


@pytest.fixture(autouse=True)
def _part2_flow(monkeypatch):
    part2_runtime(monkeypatch, review=None)              # the data review is shown in the chat: the harness types continue


class FakeLiveServer:
    """Answers like the database would; records every request (the default cache-only mode must make none)."""

    REPLIES = {"/observations": "synthetic_observations_S1_V1.json", "/observationunits": "synthetic_observationunits_S1.json"}

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, req):
        endpoint = req.url[len(LIVE_BASE):]
        self.calls.append(endpoint)
        _, body = load_fixture(self.REPLIES[endpoint])
        return TransportResponse(200, {"Content-Type": "application/json"}, body)


@pytest.fixture
def place(tmp_path, monkeypatch):
    """Live settings for this test only, a source cache holding the synthetic catalogs and measurements, reviewed access
    notes, a fake server that records requests, and the app's own settings for a question."""
    import brapi_client as bc

    monkeypatch.setenv("BRAPI_MODE", "live")
    monkeypatch.setenv("BRAPI_BASE_URL", LIVE_BASE)
    source = tmp_path / "source_cache"

    def seed(*names: str) -> None:
        for name in names:
            seed_fixture(CacheStore(source), LIVE_BASE, name)

    notes = tmp_path / "ACCESS_NOTES.md"
    notes.write_text("Reviewed by Bidhan on: 2026-09-28\n", encoding="utf-8")
    server = FakeLiveServer()
    monkeypatch.setattr(bc, "RequestsTransport", lambda: server)

    def base_config(question):     # as the app's settings could be: the harness turns off the standing approval and the age limit
        return ctl.RunConfig(question=question, offline=False, mock_model=False, direct=False, auto=False,
                             out_dir=tmp_path / "never_out", cache_dir=tmp_path / "never_cache", access_notes=notes,
                             max_clarification_rounds=app.CHAT_CLARIFICATION_ROUNDS, bootstrap_catalogs=True,
                             catalog_standing_approval=True, catalog_max_age_hours=24)

    catalog_summary.forget()
    yield type("Place", (), dict(root=tmp_path, source=source, seed=staticmethod(seed), server=server, base_config=staticmethod(base_config)))
    catalog_summary.forget()


def scripted_front_door(*labels: str):
    """The front door's model, scripted: one decision per label, in order."""
    model = FakeModelClient([reply_text(json.dumps({"label": label, "reason": "scripted", "reply": ""}), model="mock-model")
                             for label in labels], reported_model="mock-model")
    return model, (lambda: model)


def write_questions(folder: Path, questions: list[dict]) -> Path:
    path = folder / "questions.json"
    path.write_text(json.dumps({"questions": questions}), encoding="utf-8")
    return path


def snapshot(folder: Path) -> dict[str, bytes]:
    return {str(p.relative_to(folder)): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


QUESTIONS = [
    {"id": "c1", "question": "How many studies have a 2026 season?", "expect": {"kind": "studies_with_season", "season": "2026"}},
    {"id": "m1", "question": MEASURED, "expect": {"kind": "pooled_mean", "study": "SYNTHETIC yield trial S1", "variable": "fresh root yield"}},
    {"id": "x1", "question": "In study NOPE, what is the mean fresh root yield?",
     "expect": {"kind": "pooled_mean", "study": "NOPE", "variable": "fresh root yield"}},
    {"id": "u1", "question": "what about the thing?", "expect": {"kind": "study_count"}},
]


# --------------------------------------------------------------------------
# Nothing runs without --confirm
# --------------------------------------------------------------------------

def test_without_confirm_nothing_runs(place, capsys):
    place.seed(*FIXTURES)
    asked = []
    status = harness.main(["--cache", str(place.source), "--questions", str(write_questions(place.root, QUESTIONS)),
                           "--results", str(place.root / "results")],
                          base_config=place.base_config, make_models=lambda: asked.append("models"),
                          triage_model=lambda: asked.append("front door"), preflight=lambda q: [])
    assert status == 2 and "Nothing was run; add --confirm to run." in capsys.readouterr().out
    assert not (place.root / "results").exists() and asked == [] and place.server.calls == []


# --------------------------------------------------------------------------
# End to end with scripted models
# --------------------------------------------------------------------------

def test_end_to_end_with_scripted_models_on_a_copy_of_the_cache(place):
    place.seed(*FIXTURES)
    before = snapshot(place.source)
    model, front_door = scripted_front_door("catalog_question", "data_question", "unsure")
    status = harness.main(["--cache", str(place.source), "--questions", str(write_questions(place.root, QUESTIONS)),
                           "--results", str(place.root / "results"), "--confirm"],
                          base_config=place.base_config, make_models=synthetic_models, triage_model=front_door,
                          preflight=lambda q: place.base_config(q).validate())
    assert status == 0
    [folder] = list((place.root / "results").iterdir())
    report = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in report["questions"]}
    assert (report["passed"], report["failed"], report["skipped"]) == (2, 1, 1) and report["models"] == "scripted"
    assert report["requests"] == "none: cache copy only" and report["catalog_fetched"] == ["2026-06-01 00:00 UTC"]

    c1 = rows["c1"]                                       # answered by code from the saved catalog: no run
    assert c1["passed"] and c1["expected"] == 1 and c1["status"] == "no run: answered by code from the saved catalog"
    assert c1["typed"] == [] and c1["run_ids"] == [] and c1["model_calls"] == c1["front_door_calls"] == 1

    m1 = rows["m1"]                                       # a run: approve, continue, then left pending
    assert m1["passed"] and m1["expected"] == pytest.approx(18.0) and m1["why"] == "an ungrouped mean claim of the run has the expected value"
    assert m1["typed"] == ["approve", "continue"] and m1["acceptance"] == ["pending"]
    assert m1["status"] == "final review reached; the answer was left pending"
    assert m1["run_model_calls"] > 0 and m1["model_calls"] == m1["run_model_calls"] + 1
    assert m1["live_attempts"] == 0 and m1["refused_requests"] == 0 and any(c["kind"] == "mean" for c in m1["claims"])
    [run_id] = m1["run_ids"]
    result = ctl.RunResult.model_validate_json((folder / "out" / run_id / "answer.json").read_text(encoding="utf-8"))
    assert result.human_acceptance == "pending" and result.approval_id is not None    # approved data request, never accepted
    recorded = json.loads((folder / "out" / run_id / "manifest.json").read_text(encoding="utf-8"))["config"]
    assert Path(recorded["cache_dir"]) == folder / "cache" and recorded["catalog_standing_approval"] is False
    assert recorded["catalog_max_age_hours"] is None            # the copy's catalogs (fetched 2026-06-01) are used as they are

    x1 = rows["x1"]                                       # no expected answer: skipped, never run
    assert x1["passed"] is None and x1["status"].startswith("skipped: the expected answer cannot be computed")

    u1 = rows["u1"]                                       # "look it up?" is never answered yes
    assert u1["passed"] is False and u1["status"] == "no run: the app asked whether to look it up (the harness never says yes)"

    markdown = (folder / "results.md").read_text(encoding="utf-8")
    assert "| # | Question | Expected | Got | Pass | Model calls | Seconds | How it ended |" in markdown
    assert "Result: 2 passed, 1 failed, 1 skipped" in markdown and "| m1 |" in markdown and "| PASS |" in markdown
    assert len(model.requests) == 3                       # one front-door call per question that ran
    assert snapshot(place.source) == before and (folder / "cache").is_dir()           # the source cache is never changed
    assert place.server.calls == [] and not (place.root / "never_out").exists()        # nothing requested; runs under results


# --------------------------------------------------------------------------
# What the scripted approver never does
# --------------------------------------------------------------------------

def chat_for(place, cache_dir: Path, *labels: str) -> dict:
    _, front_door = scripted_front_door(*labels)
    make_config = harness.eval_config(place.base_config, cache_dir, place.root / "runs")
    return dict(make_config=make_config, make_models=synthetic_models, preflight=lambda q: make_config(q).validate(),
                triage_model=front_door)


def test_the_approver_never_approves_a_catalog_setup(place):
    outcome = harness.ask(MEASURED, chat_for(place, place.root / "empty_cache", "data_question"))
    assert outcome.status == "stopped: the catalog setup needs an approval (the harness never approves it)"
    assert outcome.typed == [] and place.server.calls == []
    [result] = outcome.results
    assert result.execution_status == "canceled" and result.catalog_approval is None and result.approval_id is None


def test_the_approver_never_answers_a_question_the_app_asks(place):
    place.seed(*FIXTURES)
    outcome = harness.ask("what about yield?", chat_for(place, place.source, "data_question"))
    assert outcome.status == "stopped: the app asked a question (the harness never answers one)"
    assert outcome.typed == [] and place.server.calls == []
    [result] = outcome.results
    assert result.approval_id is None and result.human_acceptance == "pending"


def test_every_question_runs_cache_only_not_just_the_first(place):
    """Review finding (2026-10-16): with the swap working only for the first question, every test still passed."""
    place.seed("synthetic_studies.json", "synthetic_variables.json", "synthetic_observations_S1_V1.json")   # no units
    _, front_door = scripted_front_door("data_question", "data_question")
    copy = place.root / "copy"
    import shutil
    shutil.copytree(place.source, copy)
    report = harness.run_questions([QUESTIONS[1], {**QUESTIONS[1], "id": "m2"}], cache_copy=copy, out_dir=place.root / "runs",
                                   base_config=place.base_config, make_models=synthetic_models, triage_model=front_door,
                                   preflight=lambda q: place.base_config(q).validate(), allow_live=False)
    assert place.server.calls == [] and all(row["refused_requests"] >= 1 for row in report["questions"])
    assert report["runs_still_going"] == 0


def test_a_run_the_chat_stopped_waiting_for_is_ended_and_requests_stay_refused_while_it_runs(place, monkeypatch):
    """Review finding: when the chat's step wait ran out, the run went on after the harness put the real transport back, and
    its next request reached the server. Now every run the chat starts is known, ended, and waited for; while one still
    goes on, requests stay refused."""
    import threading
    import brapi_client as bc
    place.seed(*FIXTURES)
    release = threading.Event()
    real_next_event, real_run = app.Job.next_event, app.Job._run

    def impatient(self, timeout=None):                # the chat waits far less than the run's first step takes
        return real_next_event(self, timeout=0.2)

    def slow_run(self):
        release.wait(10)
        real_run(self)

    monkeypatch.setattr(app.Job, "next_event", impatient)
    monkeypatch.setattr(app.Job, "_run", slow_run)
    _, front_door = scripted_front_door("data_question")
    copy = place.root / "copy"
    import shutil
    shutil.copytree(place.source, copy)
    report = harness.run_questions([QUESTIONS[1]], cache_copy=copy, out_dir=place.root / "runs", base_config=place.base_config,
                                   make_models=synthetic_models, triage_model=front_door,
                                   preflight=lambda q: place.base_config(q).validate(), allow_live=False, wait=0.5)
    [row] = report["questions"]
    assert row["status"] == "run ended" and report["runs_still_going"] == 1 and app.Job is not harness.TrackedJob
    assert bc.RequestsTransport is harness.RefusingTransport                     # a run is still going: requests stay refused
    release.set()                                                                 # it goes on; it was closed, so it ends
    [thread] = [t for t in threading.enumerate() if t.name.startswith("job-") and t.is_alive()]
    thread.join(timeout=30)
    assert not thread.is_alive() and place.server.calls == []                    # it ended at its approval: nothing fetched
    monkeypatch.setattr(bc, "RequestsTransport", lambda: place.server)          # as the fixture set it, for the other tests


@pytest.mark.parametrize("allow_live", [False, True], ids=["cache-only", "allow-live"])
def test_a_request_the_copy_cannot_answer_is_refused_unless_allowed(place, allow_live):
    place.seed("synthetic_studies.json", "synthetic_variables.json", "synthetic_observations_S1_V1.json")   # no units
    _, front_door = scripted_front_door("data_question")
    copy = place.root / "copy"
    import shutil
    shutil.copytree(place.source, copy)
    report = harness.run_questions([QUESTIONS[1]], cache_copy=copy, out_dir=place.root / "runs", base_config=place.base_config,
                                   make_models=synthetic_models, triage_model=front_door,
                                   preflight=lambda q: place.base_config(q).validate(), allow_live=allow_live)
    [row] = report["questions"]
    if allow_live:
        assert place.server.calls == ["/observationunits"] and row["live_attempts"] == 1 and row["refused_requests"] == 0
    else:
        assert place.server.calls == [] and row["refused_requests"] >= 1
    import brapi_client as bc
    assert bc.RequestsTransport is not harness.RefusingTransport                     # the transport is put back


# --------------------------------------------------------------------------
# Expected answers and the comparison
# --------------------------------------------------------------------------

def catalog(studies: list[dict], variables: list[dict] | None = None) -> "harness.Catalog":
    return harness.Catalog(studies=studies, variables=variables or [], cache_dir=Path("."), base_url=LIVE_BASE, fetched_at=[])


STUDIES = [
    {"studyDbId": "1", "studyName": "A", "locationDbId": "L1", "locationName": "Clinton - Horticultural", "seasons": ["2025"],
     "studyType": "Advanced Yield Trial"},
    {"studyDbId": "2", "studyName": "B", "locationDbId": "L1", "locationName": "Clinton - Horticultural", "seasons": ["Fall 2025"],
     "studyType": "Advanced Yield Trial"},
    {"studyDbId": "3", "studyName": "C", "locationDbId": "L2", "locationName": "Kinston", "seasons": ["2024", "MZP20255"],
     "studyType": "Clonal Evaluation"},
    {"studyDbId": "4", "studyName": "D", "locationDbId": "L3", "locationName": "Gurue", "seasons": [{"seasonDbId": "2019"}],
     "studyType": "Clonal Evaluation"},
    {"studyDbId": "5", "studyName": "E", "seasons": []},
]
VARIABLES = [{"observationVariableDbId": "V1", "observationVariableName": "fresh root yield|CO_331:0000296"},
             {"observationVariableDbId": "V2", "observationVariableName": "Yield of vines|CO_331:0000300"},
             {"observationVariableDbId": "V3", "observationVariableName": "dry matter content|CO_331:0000297"}]


@pytest.mark.parametrize("spec, expected", [
    ({"kind": "study_count"}, 5), ({"kind": "trait_count"}, 3), ({"kind": "location_count"}, 3),
    ({"kind": "studies_with_season", "season": "2025"}, 2), ({"kind": "studies_with_season", "season": "2019"}, 1),
    ({"kind": "studies_at_location", "word": "clinton"}, 2), ({"kind": "traits_containing", "word": "yield"}, 2),
    ({"kind": "studies_of_type", "type": "clonal evaluation"}, 2), ({"kind": "top_location"}, {"Clinton - Horticultural": 2}),
    ({"kind": "top_study_type"}, {"Advanced Yield Trial": 2, "Clonal Evaluation": 2}),
    ({"kind": "seasons_present"}, ["2019", "2024", "2025", "Fall 2025", "MZP20255"]),
    ({"kind": "studies_per_type"}, {"Advanced Yield Trial": 2, "Clonal Evaluation": 2}),
], ids=lambda value: value["kind"] if isinstance(value, dict) and "kind" in value else None)
def test_expected_answers_are_counted_by_code_from_the_raw_records(spec, expected):
    value, text = harness.expected_answer(spec, catalog(STUDIES, VARIABLES))
    assert value == expected and text


def test_a_measurement_answer_needs_exactly_one_study_and_trait_with_cached_values():
    with pytest.raises(ValueError, match="names 0 catalog rows"):
        harness.expected_answer({"kind": "pooled_mean", "study": "NOPE", "variable": "fresh root yield"}, catalog(STUDIES, VARIABLES))
    assert harness._valid_values([{"value": "10"}, {"value": " 14 "}, {"value": ""}, {"value": "NA"}, {"value": "bad"},
                                  {"value": "inf"}, {"value": None}, {"value": "2e1"}]) == [10.0, 14.0, 20.0]
    assert harness._variable_name({"observationVariableName": "Reaction to Fusarium Wilt estimating 0-5|CO_331:0000789"}) == (
        "Reaction to Fusarium Wilt estimating 0-5")


@pytest.mark.parametrize("kind, expected, shown, claims, passed", [
    ("study_count", 18, "There are 18 studies.", [], True),
    ("study_count", 18, "Seasons: 2018 (4 studies).", [], False),                 # 2018 is not 18
    ("study_count", 18, "Study S18 has 4 plots; version 1.18.", [], False),       # nor S18 or 1.18
    ("study_count", 2039, "It has 2,039 studies.", [], True),
    ("study_count", 18, "Seasons: 64,18 studies", [], True),                      # a list written without spaces
    ("valid_count", 140, "There are 140.0 valid values.", [], True),
    ("study_count", 7, "No number here.", [{"kind": "count", "value": 7}], True),
    ("valid_count", 140, "No number here.", [{"kind": "median", "value": 140}], False),   # only a count claim counts
    ("valid_count", 0, "No study names a season with 2025.", [], False),          # zero said in words is not counted
    ("top_location", {"Clinton - Horticultural": 426}, "Most: Clinton \\- Horticultural (426 studies)", [], True),
    ("top_location", {"Gurue": 144}, "Most: Clinton.", [], False),
    ("top_study_type", {"Advanced Yield Trial": 454},                             # named, but not with its own count
     "Preliminary Yield Trial has the most (454). Advanced Yield Trial is second (403).", [], False),
    ("top_study_type", {"Advanced Yield Trial": 454}, "Advanced Yield Trial (454 studies); Preliminary Yield Trial (403 studies)",
     [], True),
    ("seasons_present", ["2024", "2025"], "Seasons: 2025; 2024.", [], True),
    ("seasons_present", ["2024", "2025"], "Seasons: 20245.", [], False),
    ("studies_per_type", {"Clonal Evaluation": 2}, "- Clonal Evaluation (2 studies)", [], True),
    ("studies_per_type", {"Clonal Evaluation": 2}, "- Clonal Evaluation (3 studies), 2 more", [], False),   # its own count only
    ("studies_per_type", {"Advanced Yield Trial": 454}, "Advanced Yield Trial (450), Preliminary Yield Trial (454)", [], False),
    ("studies_per_type", {"Specialty trial": 68, "Specialty Trial": 61}, "Specialty trial (68); Specialty Trial (61)", [], True),
    ("studies_per_type", {"Clonal Evaluation": 2, "Advanced Yield Trial": 9}, "- Clonal Evaluation (2 studies)", [], False),
    ("pooled_mean", 2.3785714, "The pooled mean is 2.38 (n = 140).", [], True),
    ("pooled_mean", 2.3785714, "The pooled mean is 2.37.", [], False),
    ("pooled_mean", 2.3785714, "The pooled mean is 2.", [], False),                 # an integer alone never passes for a mean
    ("pooled_mean", 2.3785714, "Pooled mean 2.95; clone A mean 2.4.", [], False),  # another mean that happens to round
    ("pooled_mean", 2.3785714, "Mean 2.75, SD 2.38.", [], False),                  # the SD is not the mean
    ("pooled_mean", 18.0, "", [{"kind": "mean", "value": 18.0}], True),
    ("pooled_mean", 18.0, "", [{"kind": "mean", "value": 18.0, "grouping": ["germplasmDbId"]}], False),   # a clone's mean
    ("pooled_mean", 18.0, "", [{"kind": "median", "value": 18.0}], False),
], ids=["whole-number", "not-inside-a-year", "not-inside-a-name", "thousands", "list-without-spaces", "count-with-zero-decimal",
        "count-claim", "median-claim-is-not-a-count", "zero-in-words", "name-with-its-count", "wrong-name", "name-without-its-count",
        "top-with-its-count", "seasons", "seasons-inside-a-number", "type-count", "type-count-is-its-own", "type-count-of-the-next-type",
        "types-differing-in-case", "a-type-missing", "mean-rounded", "mean-off", "mean-integer", "another-mean", "sd-is-not-mean",
        "mean-claim", "grouped-mean-claim", "median-is-not-mean"])
def test_compare(kind, expected, shown, claims, passed):
    assert harness.compare(kind, expected, shown, claims)[0] is passed


def test_the_question_file_is_checked_and_the_real_one_loads():
    questions = harness.load_questions(harness.QUESTIONS)
    assert [q["id"] for q in questions] == [f"q{n:02d}" for n in range(1, 15)]
    assert sum(q["expect"]["kind"] in ("pooled_mean", "valid_count") for q in questions) == 2
    assert all("05DISE178HFLNO7" in q["question"] for q in questions if q["expect"]["kind"] in ("pooled_mean", "valid_count"))


@pytest.mark.parametrize("questions, problem", [
    ([{"id": "a", "question": "x", "expect": {"kind": "study_count"}}] * 2, "unique"),
    ([{"id": "a", "question": "x", "expect": {"kind": "best_clone"}}], "expect.kind"),
    ([{"id": "a", "question": " ", "expect": {"kind": "study_count"}}], "expect.kind"),
], ids=["duplicate-id", "unknown-kind", "empty-question"])
def test_a_bad_question_file_is_refused(tmp_path, questions, problem):
    with pytest.raises(ValueError, match=problem):
        harness.load_questions(write_questions(tmp_path, questions))
