"""
tests/test_eval_harness.py — the answer-quality harness (eval/harness.py, overnight Item 4, 2026-10-16), offline.

What these tests prove, in plain words:
* Without --confirm nothing runs: no results folder, no model, no request.
* The expected answers are computed by code from the raw catalog records and the cached measurements (seasons as words,
  "Fall 2025" counts for 2025; trait names without their ontology suffix; ties; a study or trait the catalog does not have
  gives no expected answer, so that question is skipped, never run).
* compare() scores only the answer the person sees: the chat's last message, without what the app marks as model text
  ("Additional model notes (unverified)", "Unverified model note: ..."). Numbers recorded inside the run no longer count
  (the baseline's q04 passed on a recorded 22 while the chat said "Question blocked"). A whole number in the answer; "2018"
  is not 18; "2,039" is 2039; names ignore case and markdown escapes; a mean is the number right after "Mean", within the
  decimals shown, and a whole number counts only as the exact value. Dates and times (the catalog's fetch time "22:49")
  are not numbers. A run that stopped ("Question blocked." ...), a question of the app, a lookup offer or a model's reply
  shows no answer, so it fails whatever numbers it holds (review, 2026-10-16).
* End to end, with scripted models and the app's own pipeline: a catalog question answered by code passes; a measurement
  question passes through "approve" (data request) and "continue" (data review) only, and its run is left pending, never
  accepted; an "unsure" message fails (the harness never says yes); results.json and results.md are written; runs are
  recorded under the results folder, on a copy of the cache: the source cache is unchanged and nothing was requested.
* The scripted approver never approves a catalog setup and never answers a question the app asks.
* By default a request for anything not in the cache copy is refused and counted; --allow-live lets the app fetch it.
* Tokens and cost (2026-10-17): every reply's counts, from the runs' model logs and from the front door's replies as they
  pass (its role and effort unchanged); Claude's four counts priced apart; a dated model name has its family's price; a
  model without a price gives "?"; a call without counts makes the cost "at least"; the table shows tokens in / out and
  cost, and the report adds them up.
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
    """The front door's model, scripted: one decision per label, in order. "unsure>talk" is the label call's unsure decision
    followed by the talk call's reply (Step 2c), which keeps the "Reply yes to look up" line."""
    replies = []
    for label in labels:
        decided = label.split(">")[0]
        replies.append(reply_text(json.dumps({"label": decided, "reason": "scripted", "reply": ""}), model="mock-model"))
        if label.endswith(">talk"):
            replies.append(reply_text(json.dumps({"label": decided, "reason": "scripted", "reply": "What would you like to know?",
                                                  "offer_lookup": True}), model="mock-model"))
    model = FakeModelClient(replies, reported_model="mock-model")
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

def test_end_to_end_with_scripted_models_on_a_copy_of_the_cache(place, monkeypatch):
    counts = harness.reply_tokens                 # the fakes report no usage: count each reply as zeros, so the join is seen
    monkeypatch.setattr(harness, "reply_tokens", lambda model, usage, detail: counts(model, usage, detail) or
                        {"model": model, "input": 0, "cache_write": 0, "cache_read": 0, "output": 0})
    place.seed(*FIXTURES)
    before = snapshot(place.source)
    model, front_door = scripted_front_door("catalog_question", "data_question", "unsure>talk")
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
    assert m1["passed"] and m1["expected"] == pytest.approx(18.0)
    assert m1["why"] == "the mean given, 18, is the expected value"      # from the answer's "| Mean | 18 t/ha |", not a claim
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
    assert u1["front_door_calls"] == 2                    # Step 2c: the label call, then the talk call (its reply, with the
    assert u1["answer"].startswith("What would you like to know?") and "Reply **yes**" in u1["answer"]   # yes line by code)

    markdown = (folder / "results.md").read_text(encoding="utf-8")
    assert ("| # | Question | Expected | Got | Pass | Model calls | Tokens in / out | Cost ($) | Seconds | How it ended |"
            in markdown)                                  # two columns added 2026-10-17 (tokens, cost)
    assert m1["tokens"]["calls_with_counts"] == m1["model_calls"] and m1["tokens"]["calls_without_counts"] == 0   # every
    assert m1["tokens"]["unpriced_models"] == ["mock-model"] and "| ? |" in markdown   # reply of the run's logs and front door
    assert "Result: 2 passed, 1 failed, 1 skipped" in markdown and "| m1 |" in markdown and "| PASS |" in markdown
    assert len(model.requests) == 4                       # one front-door call per question that ran, and u1's talk call
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


# Shaped like the eval baseline's answers (results 20261016T150241Z): q13's code-written mean with model notes under it, and
# q04's blocked run with a model note (its number here is invented for the test: only the model note has it).
Q13_ANSWER = ("**Mean: 2.37857142857 \\(unit not stated\\)**\n\nn = 140 (valid numeric values, one per observation unit).\n\n"
              "**Additional model notes (unverified)**\n\n- Fetched 140 observations. The pooled mean was not calculated.\n\n"
              "Other calculated statistics remain in Full report, together with the complete methods and evidence.")
Q04_SHAPED = ("**Question blocked.**\n\nCounts describe the retrieved catalog; its records may change after retrieval.\n\n"
              "Unverified model note: 22 trait names contain yield.\n\nSee Full report for the complete reason and evidence.")
NOTES_ONLY = "**Additional model notes (unverified)**\n\n- 7 rows were read\n\nn = 140 (valid numeric values)."
CATALOG_SUMMARY = ("**From the saved catalog** (no records were looked up):\n\n- It has 2029 studies, 486 traits and 101 locations "
                   "named by studies, with seasons from 2003 to 2026.\n\n*From the saved catalog of sweetpotatobase.org, "
                   "fetched 2026-09-28 22:49 UTC. Nothing new was requested; ask a data question to look up records.*")
YIELD_LIST = ("**Trait names containing “yield”: 22 of 486.** A to Z:\n\n- Biomass yield\n\nSource: sweetpotatobase.org, trait "
              "catalog fetched 2026-09-28 22:49 UTC (a saved copy; nothing new was requested). No measurements were used.")


@pytest.mark.parametrize("kind, expected, shown, passed", [
    ("study_count", 18, "There are 18 studies.", True),
    ("study_count", 18, "Seasons: 2018 (4 studies).", False),                 # 2018 is not 18
    ("study_count", 18, "Study S18 has 4 plots; version 1.18.", False),       # nor S18 or 1.18
    ("study_count", 2039, "It has 2,039 studies.", True),
    ("study_count", 18, "Seasons: 64,18 studies", True),                      # a list written without spaces
    ("valid_count", 140, "There are 140.0 valid values.", True),
    ("valid_count", 0, "No study names a season with 2025.", False),          # zero said in words is not counted
    ("top_location", {"Clinton - Horticultural": 426}, "Most: Clinton \\- Horticultural (426 studies)", True),
    ("top_location", {"Gurue": 144}, "Most: Clinton.", False),
    ("top_study_type", {"Advanced Yield Trial": 454},                         # named, but not with its own count
     "Preliminary Yield Trial has the most (454). Advanced Yield Trial is second (403).", False),
    ("top_study_type", {"Advanced Yield Trial": 454}, "Advanced Yield Trial (454 studies); Preliminary Yield Trial (403 studies)",
     True),
    ("seasons_present", ["2024", "2025"], "Seasons: 2025; 2024.", True),
    ("seasons_present", ["2024", "2025"], "Seasons: 20245.", False),
    ("studies_per_type", {"Clonal Evaluation": 2}, "- Clonal Evaluation (2 studies)", True),
    ("studies_per_type", {"Clonal Evaluation": 2}, "- Clonal Evaluation (3 studies), 2 more", False),   # its own count only
    ("studies_per_type", {"Advanced Yield Trial": 454}, "Advanced Yield Trial (450), Preliminary Yield Trial (454)", False),
    ("studies_per_type", {"Specialty trial": 68, "Specialty Trial": 61}, "Specialty trial (68); Specialty Trial (61)", True),
    ("studies_per_type", {"Clonal Evaluation": 2, "Advanced Yield Trial": 9}, "- Clonal Evaluation (2 studies)", False),
    ("studies_per_type", {"Advanced Yield Trial": 454, "Clonal Evaluation": 299},                       # the list run's lines
     "- Advanced Yield Trial — 454 studies\n- Clonal Evaluation — 299 studies", True),
    ("pooled_mean", 2.3785714, "**Mean: 2.38 (unit not stated)**", True),
    ("pooled_mean", 2.3785714, "| Statistic | Result |\n| --- | --- |\n| Mean | 2.3786 t/ha |", True),  # a table row
    ("pooled_mean", 2.3785714, "The pooled mean is 2.38 (n = 140).", False),  # not right after "Mean"
    ("pooled_mean", 2.3785714, "Mean: 2.37.", False),
    ("pooled_mean", 2.3785714, "Mean: 2.", False),                            # a whole number is not a rounding ...
    ("pooled_mean", 18.0, "| Mean | 18 t/ha |", True),                        # ... but the exact value counts
    ("pooled_mean", 2.3785714, "Pooled mean 2.95; clone A mean 2.4.", False), # another mean that happens to round
    ("pooled_mean", 2.3785714, "Mean 2.75, SD 2.38.", False),                 # the SD is not the mean
    ("pooled_mean", 2.3785714285714286, Q13_ANSWER, True),                    # the baseline's q13 still passes
    ("study_count", 22, Q04_SHAPED, False),                                   # a number only in a model note
    ("pooled_mean", 2.3785714, "**Summary**\n\nNo mean was calculated.\n\n**Additional model notes (unverified)**\n\n- Mean: 2.38",
     False),                                                                  # a mean only in the model notes
    ("valid_count", 140, NOTES_ONLY, True),                                   # code text after the notes still counts
    ("valid_count", 7, NOTES_ONLY, False),                                    # a number only in the notes does not
    ("traits_containing", 22, CATALOG_SUMMARY, False),                        # the fetch time 22:49 is not 22 (review)
    ("traits_containing", 22, YIELD_LIST, True),                              # the list's own count
    ("study_count", 22, "**Question blocked.**\n\ncoordinator stopped the run: 22 trait names were counted.\n\nSee Full "
     "report for the complete reason and evidence.", False),                 # a run that stopped shows no answer
    ("valid_count", 7, "**Additional model notes (unverified)**\n\n- 7 rows\u2028were read\n\nn = 140.", False),   # U+2028
    ("seasons_present", ["2024", "2025", "2026"], "Seasons: 2025; 2024.\n\n*fetched 2026-09-28 22:49 UTC*", False),   # not 2026
    ("study_count", 22, "This question had already ended, so your message was not sent to it.\n\n**Question canceled.**\n\n"
     "coordinator stopped the run: 22 trait names.", False),                 # the run's own message after that line
], ids=["whole-number", "not-inside-a-year", "not-inside-a-name", "thousands", "list-without-spaces", "count-with-zero-decimal",
        "zero-in-words", "name-with-its-count", "wrong-name", "name-without-its-count", "top-with-its-count", "seasons",
        "seasons-inside-a-number", "type-count", "type-count-is-its-own", "type-count-of-the-next-type", "types-differing-in-case",
        "a-type-missing", "type-list-lines", "mean-label", "mean-table-row", "mean-not-right-after", "mean-off", "mean-integer",
        "mean-exact-whole-number", "another-mean", "sd-is-not-mean", "baseline-q13", "number-only-in-a-model-note",
        "mean-only-in-the-model-notes", "code-text-after-the-notes", "number-only-in-the-notes", "a-fetch-time-is-no-number",
        "the-list-count", "a-stopped-run-is-no-answer", "a-line-separator-in-a-model-note", "a-fetch-date-is-no-season",
        "already-ended-then-stopped"])
def test_compare(kind, expected, shown, passed):
    assert harness.compare(kind, expected, shown)[0] is passed


def test_numbers_recorded_inside_the_run_no_longer_count(place, monkeypatch):
    """Eval baseline 20261016T150241Z, q04: the run recorded a count claim of 22, but the person saw "Question blocked", and
    the harness scored a pass. Now only the answer the person sees counts; the claims stay in results.json for reading."""
    from types import SimpleNamespace
    import shutil
    from contracts import Claim, Denominator

    place.seed(*FIXTURES)
    copy = place.root / "copy"
    shutil.copytree(place.source, copy)

    def claim(kind, value):
        return Claim(claim_id=f"clm_recorded_{kind}", kind=kind, value=value, denominator=Denominator(name="valid values", n=4),
                     source_artifact_hashes=["a" * 64], transformation=f"{kind} of valid values")
    recorded = SimpleNamespace(analysis=SimpleNamespace(claims=[claim("count", 1), claim("mean", 18.0)]), run_id="run_recorded",
                               counts=SimpleNamespace(model_calls=3, live_attempts=0), human_acceptance="pending",
                               model_reported="mock-model")
    shown = "**Summary**\n\nThe Full report keeps the methods and evidence."      # an answer, but without the values
    monkeypatch.setattr(harness, "ask", lambda question, chat, wait=120.0: harness.Outcome(shown=shown, status="run ended",
                                                                                           results=[recorded]))
    report = harness.run_questions(QUESTIONS[:2], cache_copy=copy, out_dir=place.root / "runs", base_config=place.base_config,
                                   make_models=synthetic_models, triage_model=None, preflight=None, allow_live=False)
    rows = {r["id"]: r for r in report["questions"]}
    assert rows["c1"]["expected"] == 1 and rows["m1"]["expected"] == pytest.approx(18.0)        # the claims hold both values
    assert (rows["c1"]["passed"], rows["m1"]["passed"], report["passed"], report["failed"]) == (False, False, 0, 2)
    assert rows["c1"]["why"] == "1 does not appear in the answer"
    assert rows["m1"]["why"] == 'the answer gives no number right after "Mean"'
    assert [c["value"] for c in rows["c1"]["claims"]] == [1, 18.0]                                # recorded, for reading


@pytest.mark.parametrize("status", ["incomplete", "failed", "blocked", "canceled", "needs_clarification", "limit_reached",
                                    "unknown"])
def test_every_answer_of_a_run_that_did_not_complete_is_no_answer(status):
    """chat_presenter.answer_summary for a run that did not complete: its reasons can carry a model's words without a mark
    ("coordinator stopped the run: ..."; review, 2026-10-16), so nothing in it is scored."""
    from types import SimpleNamespace
    from chat_presenter import answer_summary

    shown = answer_summary(SimpleNamespace(execution_status=status, analysis=None, retrieval=None,
                                           notes=["coordinator stopped the run: the total is 22 trait names"]))
    assert "22" in shown and harness.no_answer(shown) is not None
    assert harness.compare("traits_containing", 22, shown)[:2] == (False, f"no answer was shown: {harness.no_answer(shown)}")


@pytest.mark.parametrize("status, shown", [
    ("stopped: the app asked a question (the harness never answers one)",
     "I need one more detail before I can make a plan (answer 1 of 3):\n\n> Do you mean the 1 study in 2026?"),
    ("no run: the app asked whether to look it up (the harness never says yes)",
     "Is that 1 study?\n\nReply **yes** to look up “How many studies have a 2026 season?” in eval.invalid as you typed it."),
    ("no run: the app answered without starting one", "I can count 1 study for you."),
], ids=["a-question-of-the-app", "a-lookup-offer", "a-model-reply"])
def test_a_question_that_ended_without_an_answer_written_by_code_is_not_scored(place, monkeypatch, status, shown):
    """Review (2026-10-16): the app's own question, a lookup offer and a model's reply were scored like an answer."""
    import shutil

    place.seed(*FIXTURES)
    copy = place.root / "copy"
    shutil.copytree(place.source, copy)
    monkeypatch.setattr(harness, "ask", lambda question, chat, wait=120.0: harness.Outcome(shown=shown, status=status))
    report = harness.run_questions(QUESTIONS[:1], cache_copy=copy, out_dir=place.root / "runs", base_config=place.base_config,
                                   make_models=synthetic_models, triage_model=None, preflight=None, allow_live=False)
    [row] = report["questions"]
    assert row["expected"] == 1 and row["passed"] is False and row["why"] == "no answer was shown to score"
    assert harness.compare("studies_with_season", 1, shown)[0] is True       # read as an answer, it would have passed


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


# --------------------------------------------------------------------------
# Tokens and cost
# --------------------------------------------------------------------------

def test_a_reply_is_priced_by_the_model_it_names():
    assert harness.price_of("gpt-6-luna") == harness.price_of("gpt-6-luna-2026-09-30") == harness.PRICES["gpt-6-luna"]
    assert harness.price_of("claude-opus-5-5") == harness.PRICES["claude-opus-5-5"]
    assert harness.price_of("gpt-6-lunar") is None and harness.price_of(None) is None and harness.price_of("claude-opus-5") is None
    claude = harness.reply_tokens("claude-opus-5-5", {"prompt_tokens": 1130, "completion_tokens": 200, "total_tokens": 1330},
                                  {"input_tokens": 1000, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 30,
                                   "output_tokens": 200})
    luna = harness.reply_tokens("gpt-6-luna-2026-09-30", {"prompt_tokens": 2000, "completion_tokens": 100, "total_tokens": 2100}, None)
    assert claude == {"model": "claude-opus-5-5", "input": 1000, "cache_write": 100, "cache_read": 30, "output": 200}
    assert harness.reply_tokens("m", None, None) is None
    totals = harness.tally([claude, luna], calls=3)
    expected = (1000 * 4.00 + 100 * 5.00 + 30 * 0.20 + 200 * 20.00 + 2000 * 0.10 + 100 * 0.50) / 1e6
    assert totals["cost_usd"] == pytest.approx(expected) and totals["input"] == 3000 and totals["output"] == 300
    assert totals["calls_with_counts"] == 2 and totals["calls_without_counts"] == 1 and totals["unpriced_models"] == []
    assert harness.tally([claude, luna], calls=1)["calls_without_counts"] == 0      # never fewer calls than counted replies
    unknown = harness.tally([{**luna, "model": "some-other-model"}], calls=1)
    assert unknown["cost_usd"] is None and unknown["unpriced_models"] == ["some-other-model"]


def test_the_runs_model_logs_are_read_one_reply_per_assistant_line(tmp_path):
    run = tmp_path / "run_1"
    run.mkdir()
    lines = [{"event": "start"}, {"event": "assistant", "model": "claude-opus-5-5", "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                                   "usage_detail": {"input_tokens": 5, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 2,
                                                    "output_tokens": 3}},
             {"event": "provider_error", "code": "timeout"},
             {"event": "assistant", "model": "gpt-6-luna", "usage": {"prompt_tokens": 10, "completion_tokens": 4}},
             {"event": "assistant", "model": "gpt-6-luna", "usage": None}]
    (run / "llm_retriever.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\nnot json\n", encoding="utf-8")
    (run / "llm_analyst.jsonl").write_text(json.dumps({"event": "assistant", "model": "gpt-6-luna",
                                                       "usage": {"prompt_tokens": 1, "completion_tokens": 1}}) + "\n", encoding="utf-8")
    (run / "notes.jsonl").write_text(json.dumps({"event": "assistant", "usage": {"prompt_tokens": 99}}) + "\n", encoding="utf-8")
    replies = harness.run_replies([run, tmp_path / "missing"])
    assert [(r["model"], r["input"], r["cache_read"], r["output"]) for r in replies] == [
        ("gpt-6-luna", 1, 0, 1), ("claude-opus-5-5", 5, 2, 3), ("gpt-6-luna", 10, 0, 4)]


def test_the_front_doors_replies_are_counted_without_changing_its_request(tmp_path, monkeypatch):
    import anthropic
    import httpx2
    import llm
    import triage

    for name in ("SPACE_ID", *llm.ANTHROPIC_REFUSED_ENV):
        monkeypatch.delenv(name, raising=False)
    seen = []

    def answer(request):
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json={"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                                          "content": [{"type": "text", "text": "Hi"}], "stop_reason": "end_turn",
                                          "stop_sequence": None, "usage": {"input_tokens": 120, "output_tokens": 30,
                                                                            "cache_read_input_tokens": 10}})

    settings = llm.load_llm_settings(env_file=tmp_path / "none.env", environ={
        "LLM_PROVIDER": "anthropic", "LLM_ALLOW_REMOTE": "true", "LLM_MODEL": "claude-opus-5-5", "ANTHROPIC_API_KEY": "sk-ant-test"})
    counter = harness.CountingFactory(lambda: llm.AnthropicClient(
        settings, http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(answer))))
    import anyio

    reply = anyio.run(triage._complete, counter(), "PROMPT", "hello")
    assert reply.content == "Hi" and seen[0]["output_config"] == {"effort": "low"}       # the front door's role is kept
    assert counter.calls == 1 and counter.replies == [{"model": "claude-opus-5-5", "input": 120, "cache_write": 0,
                                                        "cache_read": 10, "output": 30}]
    fake = FakeModelClient([reply_text("ok")])
    assert anyio.run(triage._complete, harness.CountingFactory(lambda: fake)(), "P", "x").content == "ok"   # a fake works too
    assert len(fake.requests) == 1


def test_the_table_shows_tokens_and_cost_and_the_report_adds_them_up():
    def row(qid, tokens):
        return {"id": qid, "question": "q", "expected_text": "", "got": "", "passed": True, "model_calls": 1, "seconds": 1.0,
                "status": "run ended", "tokens": tokens}

    priced = {"input": 46_000, "cache_write": 0, "cache_read": 821, "output": 2_189, "calls_with_counts": 10,
              "calls_without_counts": 0, "cost_usd": 0.231, "unpriced_models": []}
    short = {**priced, "calls_without_counts": 1, "cost_usd": 0.01}
    unpriced = {**priced, "cost_usd": None, "unpriced_models": ["mystery"]}
    total = {"input": 1, "cache_write": 2, "cache_read": 3, "output": 4, "calls_with_counts": 1, "calls_without_counts": 1,
             "cost_usd": None, "unpriced_models": ["mystery"]}
    report = {"cache_copy": "c", "base_url": "b", "catalog_fetched": [], "requests": "none", "passed": 3, "failed": 0, "skipped": 0,
              "questions": [row("a", priced), row("b", short), row("c", unpriced)], "tokens": total}
    markdown = harness.results_markdown(report, started="now", models="m")
    assert "| 46,821 / 2,189 | 0.231000 |" in markdown and "| at least 0.010000 |" in markdown and "| ? |" in markdown
    assert "- Tokens: 6 in (cache writes 2, cache reads 3), 4 out (thinking included); cost: ? US dollars" in markdown
    assert "- Model calls without token counts (a provider error or a timeout): 1." in markdown
    assert "- No price for: mystery." in markdown


def test_a_front_door_replys_counts_reach_its_row_and_the_total(place):
    from contracts import Usage

    place.seed(*FIXTURES)
    reply = reply_text(json.dumps({"label": "catalog_question", "reason": "scripted", "reply": ""}), model="gpt-6-luna-2026-09-30",
                       usage=Usage(prompt_tokens=1000, completion_tokens=40, total_tokens=1040))
    model = FakeModelClient([reply], reported_model="gpt-6-luna-2026-09-30")
    status = harness.main(["--cache", str(place.source), "--questions", str(write_questions(place.root, [QUESTIONS[0]])),
                           "--results", str(place.root / "results"), "--confirm"],
                          base_config=place.base_config, make_models=synthetic_models, triage_model=lambda: model,
                          preflight=lambda q: place.base_config(q).validate())
    [folder] = list((place.root / "results").iterdir())
    report = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    [row] = report["questions"]
    assert status == 0 and row["id"] == "c1" and row["model_calls"] == 1
    assert row["tokens"] == {"input": 1000, "cache_write": 0, "cache_read": 0, "output": 40, "calls_with_counts": 1,
                             "calls_without_counts": 0, "cost_usd": pytest.approx((1000 * 0.10 + 40 * 0.50) / 1e6),
                             "unpriced_models": []}
    assert report["tokens"]["input"] == 1000 and report["tokens"]["cost_usd"] == row["tokens"]["cost_usd"]
    assert "| 1,000 / 40 | 0.000120 |" in (folder / "results.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# A6 (coach 2026-10-22 17:47 UTC): on a server whose measurements the app reads from the plot records (brapi_client.
# PLOT_RECORD_HOSTS, whose /observations leaves out values recorded as 0), the expected values come from the plot records too
# --------------------------------------------------------------------------

SPB_BASE = "https://sweetpotatobase.org/brapi/v2"
S1_YIELD = {"study": "SYNTHETIC yield trial S1", "variable": "fresh root yield"}


def measured_copy(tmp_path, monkeypatch, base_url: str, *, plot_records: bool = True) -> "harness.Catalog":
    """A cache copy of server base_url holding study S1's V1 values from /observations (10, 14, 20 and 28 are valid) and, unless
    plot_records is False, S1's plot records with one more plot whose V1 value was recorded as 0: the value such a server's
    /observations leaves out."""
    monkeypatch.setenv("BRAPI_BASE_URL", base_url)
    cache_dir = tmp_path / "copy"
    seed_fixture(CacheStore(cache_dir), base_url, "synthetic_observations_S1_V1.json")
    if plot_records:
        parsed, _ = load_fixture("synthetic_plotrecords_S1.json")
        plot = json.loads(json.dumps(parsed["result"]["data"][0]))
        plot.update(observationUnitDbId="P7", observationUnitName="plot P7")
        plot["observationUnitPosition"]["observationLevel"]["levelCode"] = "P7"
        plot["observations"] = [dict(plot["observations"][0], observationDbId="O7", value="0")]
        parsed["result"]["data"].append(plot)
        parsed["metadata"]["pagination"]["totalCount"] = len(parsed["result"]["data"])
        folder = tmp_path / "fixtures"
        folder.mkdir()
        (folder / "synthetic_plotrecords_S1_with_a_zero.json").write_text(json.dumps(parsed), encoding="utf-8")
        seed_fixture(CacheStore(cache_dir), base_url, "synthetic_plotrecords_S1_with_a_zero.json", fixture_dir=folder)
    return harness.Catalog(studies=[{"studyDbId": "S1", "studyName": "SYNTHETIC yield trial S1"}],
                           variables=[{"observationVariableDbId": "V1", "observationVariableName": "fresh root yield"}],
                           cache_dir=cache_dir, base_url=base_url, fetched_at=[])


def test_on_a_plot_record_server_the_expected_values_come_from_the_plot_records(tmp_path, monkeypatch):
    copy = measured_copy(tmp_path, monkeypatch, SPB_BASE)
    assert harness.expected_answer({"kind": "valid_count", **S1_YIELD}, copy) == (5, "5")
    mean, text = harness.expected_answer({"kind": "pooled_mean", **S1_YIELD}, copy)
    assert mean == pytest.approx(14.4) and text == "14.4000 (from 5 valid values)"


def test_on_any_other_server_the_expected_values_still_come_from_observations(tmp_path, monkeypatch):
    copy = measured_copy(tmp_path, monkeypatch, LIVE_BASE)            # the plot records are in the copy and are not read
    assert harness.expected_answer({"kind": "valid_count", **S1_YIELD}, copy) == (4, "4")
    mean, text = harness.expected_answer({"kind": "pooled_mean", **S1_YIELD}, copy)
    assert mean == pytest.approx(18.0) and text == "18.0000 (from 4 valid values)"


def test_a_plot_record_server_whose_copy_lacks_the_plot_records_gives_no_expected_value(tmp_path, monkeypatch):
    """The question is skipped with this reason; its expected value is never computed from the /observations without zeros."""
    copy = measured_copy(tmp_path, monkeypatch, SPB_BASE, plot_records=False)
    for kind in ("pooled_mean", "valid_count"):
        with pytest.raises(ValueError, match="its plot records are not completely in the cache copy"):
            harness.expected_answer({"kind": kind, **S1_YIELD}, copy)
