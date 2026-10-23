"""
tests/test_fresh_catalogs.py — fresh catalogs (Item 5): saved catalogs older than APP_CATALOG_MAX_AGE_HOURS, and "refresh
catalogs", offline.

What these tests prove, in plain words:
* Only WHEN the catalogs are downloaded changes, never what or how. Saved catalogs older than the limit (24 hours by
  default) count as missing: the visitor sees the catalog screen, which says why, and nothing is requested until they type
  approve; then the same two catalog requests are sent again (the old saved replies are kept, the new ones are added) and
  the question goes on to its own approval. Saved catalogs within the limit, or with no limit, are used as before: no
  screen, no request.
* "refresh catalogs" (also "please update the catalog") is a command code finds: no model call. It starts a catalog setup
  run that needs the catalog approval (the visitor's, or the operator's standing approval) and then says what the
  catalogs hold now and when they were fetched. Canceled, it requests nothing. Typed while a question waits, it is a new
  question (drop or keep).
* The client never sends a request because of a refresh without an approval, and offline it always uses the saved reply.
* The approval screen says when the catalogs were fetched.
Fake server and scripted models only; the conftest guard would stop any real socket.
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import app
import catalog_lists
import catalog_summary
import run as ctl
from brapi_client import BrapiClient, BrapiClientError, Settings, TransportResponse
from cache_store import CacheIdentity, CacheStore
from part2_support import FIXTURE_FETCHED_AT, load_fixture, part2_runtime, seed_fixture, synthetic_models

LIVE_BASE = "https://refresh.invalid/brapi/v2"
Q = "For synthetic study S1, give mean yield and clone means with n."
CATALOGS = ["/studies", "/variables"]


@pytest.fixture(autouse=True)
def _part2_flow(monkeypatch):
    part2_runtime(monkeypatch)


class FakeLiveServer:
    REPLIES = {"/studies": "synthetic_studies.json", "/variables": "synthetic_variables.json",
               "/observations": "synthetic_observations_S1_V1.json", "/observationunits": "synthetic_observationunits_S1.json"}

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, req):
        endpoint = req.url[len(LIVE_BASE):]
        self.calls.append(endpoint)
        _, body = load_fixture(self.REPLIES[endpoint])
        return TransportResponse(200, {"Content-Type": "application/json"}, body)


@pytest.fixture
def place(tmp_path, monkeypatch):
    """Live settings, the catalogs SAVED (their replies say they were fetched 2026-06-01), a fake server, reviewed notes."""
    import brapi_client as bc

    monkeypatch.setenv("BRAPI_MODE", "live")
    monkeypatch.setenv("BRAPI_BASE_URL", LIVE_BASE)
    server = FakeLiveServer()
    monkeypatch.setattr(bc, "RequestsTransport", lambda: server)
    notes = tmp_path / "ACCESS_NOTES.md"
    notes.write_text("Reviewed by Bidhan on: 2026-09-28\n", encoding="utf-8")
    cache = CacheStore(tmp_path / "live_cache")
    for name in ("synthetic_studies.json", "synthetic_variables.json"):
        seed_fixture(cache, LIVE_BASE, name)

    def kwargs(max_age: float | None = 24, standing: bool = False):
        def make_config(question):
            return ctl.RunConfig(question=question, offline=False, mock_model=False, direct=False, auto=False,
                                 out_dir=tmp_path / "out", cache_dir=tmp_path / "live_cache", access_notes=notes,
                                 max_clarification_rounds=app.CHAT_CLARIFICATION_ROUNDS, bootstrap_catalogs=True,
                                 catalog_standing_approval=standing, catalog_max_age_hours=max_age)
        return dict(make_config=make_config, make_models=synthetic_models, preflight=lambda q: make_config(q).validate())

    catalog_summary.forget()
    yield dataclasses.make_dataclass("Place", ["server", "root", "cache", "kwargs"])(server, tmp_path, cache, kwargs)
    catalog_summary.forget()


def turn(message, history, state, kwargs):
    out = list(app.respond(message, history, state, **kwargs))
    assert out[-1][2] == ""
    return out[-1][0], out[-1][1]


def versions(cache: CacheStore, endpoint: str) -> list[int]:
    return cache.versions(CacheIdentity(base_url=LIVE_BASE, endpoint=endpoint, params={"page": 0, "pageSize": 1000}))


def saved_run(root: Path, run_id: str) -> tuple[ctl.RunResult, dict]:
    folder = root / "out" / run_id
    return (ctl.RunResult.model_validate_json((folder / "answer.json").read_text(encoding="utf-8")),
            json.loads((folder / "manifest.json").read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# Saved catalogs older than the limit
# --------------------------------------------------------------------------

def test_old_catalogs_are_set_aside_and_downloaded_again_only_after_approve(place):
    kw = place.kwargs(24)
    history, state = turn(Q, [], {}, kw)
    shown = history[-1]["content"]
    assert ("The saved list of studies and traits needs to be downloaded again from refresh.invalid (the saved catalogs were "
            "fetched 2026-06-01 00:00 UTC, more than 24 hours ago).") in shown
    assert "Reply **approve** to prepare these catalogs" in shown and place.server.calls == []     # nothing before approve
    job = state["job"]
    history, state = turn("approve", history, state, kw)
    assert place.server.calls == CATALOGS and "Review this plan before approving" in history[-1]["content"]
    assert versions(place.cache, "/studies") == [1, 2] and versions(place.cache, "/variables") == [1, 2]   # old ones kept
    history, state = turn("cancel", history, state, kw)
    result, manifest = saved_run(place.root, job.controller.run_id)
    assert manifest["catalog_setup"]["status"] == "completed" and result.counts.live_attempts == 2
    assert any("saved catalogs set aside: the saved catalogs were fetched 2026-06-01 00:00 UTC" in n for n in result.notes)
    assert place.server.calls == CATALOGS                                                     # no measurement either


@pytest.mark.parametrize("max_age", [8760, None], ids=["within-a-year", "no-limit"])
def test_catalogs_within_the_limit_or_without_one_are_used_as_before(place, max_age):
    history, state = turn(Q, [], {}, place.kwargs(max_age))
    assert "Review this plan before approving" in history[-1]["content"] and place.server.calls == []
    assert "previously cached; fetched 2026-06-01T00:00:00+00:00" in history[-1]["content"]      # the screen gives the date
    assert "catalogs       : cached copy fetched 2026-06-01T00:00:00+00:00 (not re-fetched)" in app.full_report(state)
    app._close_job(state)


def test_the_age_is_measured_from_the_oldest_saved_reply(place, monkeypatch):
    now = FIXTURE_FETCHED_AT.replace(hour=23)                             # 23 hours after the replies were saved
    config = place.kwargs(24)["make_config"](Q)
    controller = ctl.Controller(config, models=synthetic_models(), answer_fn=lambda p: "cancel", now=lambda: now)
    controller.catalog_fetched_at = FIXTURE_FETCHED_AT.isoformat()
    assert controller._why_set_aside() is None
    controller.now = lambda: FIXTURE_FETCHED_AT.replace(day=2, hour=1)    # 25 hours after
    assert controller._why_set_aside() == ("the saved catalogs were fetched 2026-06-01 00:00 UTC, more than 24 hours ago")


# --------------------------------------------------------------------------
# "refresh catalogs"
# --------------------------------------------------------------------------

@pytest.mark.parametrize("typed, found", [
    ("refresh catalogs", True), ("Refresh the catalogs!", True), ("please update the catalog", True),
    ("reload catalogs now", True), ("re-download the saved catalogs", True), ("refresh the study and trait catalogs", True),
    ("refresh catalogs for study 4501", False), ("what is a catalog?", False), ("list all traits", False),
])
def test_the_refresh_command_is_found_by_code(typed, found):
    assert catalog_lists.is_refresh_request(typed) is found


def test_refresh_asks_the_visitor_and_then_downloads_the_catalogs_again(place):
    model_calls: list[object] = []
    kw = {**place.kwargs(None), "triage_model": lambda: model_calls.append(1)}             # never asked
    history, state = turn("refresh catalogs", [], {}, kw)
    shown = history[-1]["content"]
    assert "needs to be downloaded again from refresh.invalid (you asked to refresh the catalogs)" in shown
    assert place.server.calls == [] and model_calls == []
    job = state["job"]
    history, state = turn("approve", history, state, kw)
    assert place.server.calls == CATALOGS and "job" not in state
    now = datetime.now(timezone.utc)
    assert history[-1]["content"].startswith("**The study and trait catalogs were downloaded again.** The study catalog has "
                                             "1 study and the trait catalog 2 traits, from refresh.invalid, fetched ")
    assert f"fetched {now:%Y-%m-%d}" in history[-1]["content"] and "No measurements were fetched." in history[-1]["content"]
    result, manifest = saved_run(place.root, job.controller.run_id)
    assert result.execution_status == "completed" and result.counts.model_calls == 0 and result.approval_id is None
    assert manifest["catalog_setup"]["issued_by"] == ctl.PERSON_ISSUER and manifest["config"]["catalog_refresh"] is True
    [past] = state["memory"].turns
    assert past.kind == "catalog_setup" and not past.continuable


def test_refresh_canceled_requests_nothing(place):
    history, state = turn("refresh catalogs", [], {}, place.kwargs(None))
    history, state = turn("cancel", history, state, place.kwargs(None))
    assert place.server.calls == [] and "job" not in state
    assert versions(place.cache, "/studies") == [1]


def test_refresh_with_the_operator_standing_approval_needs_no_screen(place):
    history, state = turn("refresh catalogs", [], {}, place.kwargs(None, standing=True))
    assert history[-1]["content"].startswith("**The study and trait catalogs were downloaded again.**")
    assert place.server.calls == CATALOGS and "job" not in state


def test_refresh_typed_while_a_question_waits_is_a_new_question(place):
    kw = place.kwargs(None)
    history, state = turn(Q, [], {}, kw)
    job = state["job"]
    history, state = turn("refresh catalogs", history, state, kw)
    assert history[-1]["content"].startswith("That sounds like a new question: “refresh catalogs”")
    assert state["job"] is job and job.waiting_on.startswith("approve / edit / cancel") and place.server.calls == []
    app._close_job(state)


# --------------------------------------------------------------------------
# The client: a refresh never sends a request without an approval; offline, the saved reply is always used
# --------------------------------------------------------------------------

def test_a_set_aside_reply_is_requested_again_only_live_and_only_with_an_approval(place):
    calls: list[str] = []

    def transport(req):
        calls.append(req.url)
        raise AssertionError("no request may be sent here")

    live = BrapiClient(Settings(base_url=LIVE_BASE, mode="live", cache_dir=place.root / "live_cache"), transport=transport,
                       cache=place.cache)
    live.refresh_cached_before = datetime.now(timezone.utc)
    with pytest.raises(BrapiClientError):
        live.get("/studies", {"page": 0, "pageSize": 1000})                # no approval: refused before any request
    assert calls == []
    offline = BrapiClient(Settings(base_url=LIVE_BASE, mode="offline", cache_dir=place.root / "live_cache"), transport=transport,
                          cache=place.cache)
    offline.refresh_cached_before = datetime.now(timezone.utc)
    envelope, record = offline.get("/studies", {"page": 0, "pageSize": 1000})   # offline: the saved reply, as always
    assert record.origin == "cache" and calls == []


@pytest.mark.parametrize("value, problem", [("24", None), ("0", "APP_CATALOG_MAX_AGE_HOURS"), ("lots", "APP_CATALOG_MAX_AGE_HOURS")])
def test_the_age_setting_is_read_and_checked(monkeypatch, value, problem):
    monkeypatch.setenv("APP_CATALOG_MAX_AGE_HOURS", value)
    if problem is None:
        assert app.live_config("any question").catalog_max_age_hours == 24
    else:
        with pytest.raises(ValueError, match=problem):
            app.live_config("any question")


def test_the_default_limit_is_24_hours(monkeypatch):
    monkeypatch.delenv("APP_CATALOG_MAX_AGE_HOURS", raising=False)
    assert app.live_config("any question").catalog_max_age_hours == 24


@pytest.mark.parametrize("changes, problem", [
    (dict(catalog_max_age_hours=0.5), "catalog_max_age_hours must be a number of hours from 1 to 8760"),
    (dict(catalog_max_age_hours=float("inf")), "catalog_max_age_hours must be a number of hours from 1 to 8760"),
    (dict(catalog_refresh=True), "a catalog refresh is a catalog setup run"),
])
def test_a_run_refuses_a_bad_age_or_a_refresh_that_is_not_a_setup_run(changes, problem):
    assert problem in ctl.RunConfig(question=Q, **changes).validate()


def test_startup_downloads_saved_catalogs_older_than_the_limit_again(place):
    """Status doc, other open (coach 2026-10-23 12:49 UTC, NEXT 2): at startup (app.prepare_catalogs, the operator's standing
    approval on) a saved catalog counted as saved whatever its age, so these 2026-06-01 catalogs were kept with a 24-hour limit
    until a visitor's question set them aside. Startup applies the limit as the front door and a run do: they are downloaded again
    under the standing approval, and the next startup finds them within the limit and requests nothing."""
    make_config = place.kwargs(24, standing=True)["make_config"]
    line = app.prepare_catalogs(make_config, make_models=synthetic_models)
    assert line.startswith("catalog standing approval: on; catalog setup completed (run run_") and place.server.calls == CATALOGS
    result, _ = saved_run(place.root, line.rsplit("(run ", 1)[1].rstrip(")"))
    assert any("set aside before: the saved catalogs were fetched 2026-06-01" in note for note in result.notes)
    again = app.prepare_catalogs(make_config, make_models=synthetic_models)
    assert again == "catalog standing approval: on; the study and trait catalogs are already saved, nothing was requested"
    assert place.server.calls == CATALOGS


def test_startup_without_a_limit_keeps_the_saved_catalogs_as_before(place):
    line = app.prepare_catalogs(place.kwargs(None, standing=True)["make_config"], make_models=synthetic_models)
    assert line == "catalog standing approval: on; the study and trait catalogs are already saved, nothing was requested"
    assert place.server.calls == []


def test_saved_catalogs_with_an_unknown_date_are_kept_at_startup_and_at_the_front_door(place, monkeypatch):
    """As in a run (Controller._why_set_aside), saved catalogs whose fetch date was not recorded count as fresh under a limit."""
    make_config = place.kwargs(24, standing=True)["make_config"]
    saved = catalog_summary.from_cache(make_config(Q))
    assert saved is not None and saved.fetched_at is not None
    unknown = dataclasses.replace(saved, fetched_at=None)
    monkeypatch.setattr(catalog_summary, "from_cache", lambda config: unknown)
    line = app.prepare_catalogs(make_config, make_models=synthetic_models)
    assert line == "catalog standing approval: on; the study and trait catalogs are already saved, nothing was requested"
    assert app._saved_catalog(make_config, "tell me about the synthetic location") is unknown
    assert place.server.calls == []


def test_saved_catalogs_exactly_at_the_limit_are_kept_as_in_a_run(place, monkeypatch):
    """Review finding (2026-10-23): no test sat on the limit itself, so `>=` for `>` passed. As in a run (set aside only when
    OLDER than the limit), catalogs exactly 24 hours old are kept; one microsecond more counts as missing."""
    now = datetime(2026, 6, 2, tzinfo=timezone.utc)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(app, "datetime", Clock)
    config = place.kwargs(24)["make_config"](Q)
    saved = catalog_summary.from_cache(config)
    assert not app._older_than_limit(dataclasses.replace(saved, fetched_at=now - timedelta(hours=24)), config)
    assert app._older_than_limit(dataclasses.replace(saved, fetched_at=now - timedelta(hours=24, microseconds=1)), config)


def test_old_saved_catalogs_are_not_requested_at_startup_without_the_standing_approval_or_its_review_date(place):
    """Review finding (2026-10-23): no test held old saved catalogs with the setting off or with no review date. Both are
    checked before the catalogs' age: off, nothing is requested; on without a review date, the app refuses to start."""
    line = app.prepare_catalogs(place.kwargs(24, standing=False)["make_config"], make_models=synthetic_models)
    assert line.startswith("catalog standing approval: off") and place.server.calls == []
    (place.root / "ACCESS_NOTES.md").write_text("Not reviewed yet.\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        app.prepare_catalogs(place.kwargs(24, standing=True)["make_config"], make_models=synthetic_models)
    assert place.server.calls == []


# --------------------------------------------------------------------------
# Review findings (2026-10-16)
# --------------------------------------------------------------------------

def put_reply(cache: CacheStore, name: str, fetched_at: datetime) -> None:
    """A saved catalog reply with its own fetch time (a new version; earlier versions are kept)."""
    parsed, body = load_fixture(name)
    request = parsed["_request"]
    identity = CacheIdentity(base_url=LIVE_BASE, endpoint=request["endpoint"], params=dict(request.get("params", {})))
    cache.put(identity, body, http_status=200, fetched_at_utc=fetched_at, source_url_redacted=f"{LIVE_BASE}{request['endpoint']}",
              content_type="application/json")


def kwargs_for(place, cache_dir: Path, max_age: float | None = 24, **extra):
    notes = place.root / "ACCESS_NOTES.md"

    def make_config(question):
        return ctl.RunConfig(question=question, offline=False, mock_model=False, direct=False, auto=False,
                             out_dir=place.root / "out", cache_dir=cache_dir, access_notes=notes,
                             max_clarification_rounds=app.CHAT_CLARIFICATION_ROUNDS, bootstrap_catalogs=True,
                             catalog_max_age_hours=max_age)
    return dict(make_config=make_config, make_models=synthetic_models, preflight=lambda q: make_config(q).validate(), **extra)


def front_door(*labels: str):
    from llm import FakeModelClient, reply_text
    model = FakeModelClient([reply_text(json.dumps({"label": label, "reason": "scripted", "reply": ""}), model="mock-model")
                             for label in labels], reported_model="mock-model")
    return model, (lambda: model)


def test_the_oldest_saved_reply_decides_and_exactly_the_limit_is_not_older(place):
    """Review finding: no test failed when the age was taken from the newest reply, or when "older than" became "at least"."""
    put_reply(place.cache, "synthetic_variables.json", datetime.now(timezone.utc))      # the trait catalog is new, the study one old
    history, state = turn(Q, [], {}, place.kwargs(24))
    assert "needs to be downloaded again" in history[-1]["content"] and place.server.calls == []
    app._close_job(state)
    config = place.kwargs(24)["make_config"](Q)
    controller = ctl.Controller(config, models=synthetic_models(), answer_fn=lambda p: "cancel",
                                now=lambda: FIXTURE_FETCHED_AT + timedelta(hours=24))
    controller.catalog_fetched_at = FIXTURE_FETCHED_AT.isoformat()
    assert controller._why_set_aside() is None                                          # exactly 24 hours: not older
    controller.now = lambda: FIXTURE_FETCHED_AT + timedelta(hours=24, seconds=1)
    assert controller._why_set_aside() is not None


def test_a_refresh_with_a_partly_saved_catalog_downloads_the_whole_set_again(place):
    """Review finding: with the study catalog saved and the trait catalog missing, a refresh fetched only the traits, kept the
    old studies, and still said both were downloaded again."""
    partial = place.root / "partial_cache"
    seed_fixture(CacheStore(partial), LIVE_BASE, "synthetic_studies.json")
    kw = kwargs_for(place, partial, max_age=None)
    history, state = turn("refresh catalogs", [], {}, kw)
    assert "needs to be downloaded again" in history[-1]["content"] and place.server.calls == []
    history, state = turn("approve", history, state, kw)
    assert place.server.calls == CATALOGS                                               # both, the saved one too
    assert history[-1]["content"].startswith("**The study and trait catalogs were downloaded again.**")


def test_a_refresh_that_requested_nothing_does_not_say_it_downloaded(place):
    """Review finding: saved replies dated after now were reused by a refresh, and the chat still said "downloaded again"."""
    future = place.root / "future_cache"
    for name in ("synthetic_studies.json", "synthetic_variables.json"):
        put_reply(CacheStore(future), name, datetime.now(timezone.utc) + timedelta(days=1))
    kw = kwargs_for(place, future, max_age=None)
    history, state = turn("refresh catalogs", [], {}, kw)
    history, state = turn("approve", history, state, kw)
    assert place.server.calls == []
    assert history[-1]["content"].startswith("**The study and trait catalogs were already saved; nothing new was requested.**")


def test_the_front_door_does_not_answer_from_catalogs_older_than_the_limit(place):
    """Review finding: a catalog question was answered from 2026-06-01 catalogs with a 24-hour limit, with no screen."""
    model, factory = front_door("catalog_question")
    history, state = turn("tell me about the synthetic location", [], {}, {**place.kwargs(24), "triage_model": factory})
    assert "Saved catalog" not in model.requests[0]["messages"][0]["content"]          # not offered: it counts as missing
    assert history[-1]["content"] == app.triage.unsure_line("refresh.invalid") and place.server.calls == []
    model, factory = front_door("catalog_question")
    history, _ = turn("tell me about the synthetic location", [], {}, {**place.kwargs(None), "triage_model": factory})
    assert history[-1]["content"].startswith("**From the saved catalog**")              # no limit: answered as before


def test_after_a_question_downloads_the_catalogs_again_the_front_door_reads_the_new_ones(place):
    """Review finding: only a refresh run made the front door forget its kept summary; after a question's own catalog setup it
    kept answering from the old catalogs for up to 10 minutes."""
    model, factory = front_door("data_question", "catalog_question")
    kw = {**place.kwargs(24), "triage_model": factory}
    history, state = turn(Q, [], {}, kw)                                                 # the catalog screen (old catalogs)
    history, state = turn("approve", history, state, kw)                                 # downloaded again now
    history, state = turn("cancel", history, state, kw)
    history, state = turn("tell me about the synthetic location", history, state, kw)
    shown = history[-1]["content"]
    assert shown.startswith("**From the saved catalog**") and "fetched 2026-06-01" not in shown


def test_a_set_aside_question_still_reads_its_measurements_from_the_saved_copy(place):
    """Review finding: no test ran a set-aside question past the plan, so a refresh mark leaking to the question's own data
    request (re-fetching saved measurements) was not caught."""
    for name in ("synthetic_observations_S1_V1.json", "synthetic_observationunits_S1.json"):
        seed_fixture(place.cache, LIVE_BASE, name)
    kw = place.kwargs(24)
    history, state = turn(Q, [], {}, kw)
    history, state = turn("approve", history, state, kw)                                 # the catalogs, downloaded again
    history, state = turn("approve", history, state, kw)                                 # the data request
    assert "Reply **accept**" in history[-1]["content"] and place.server.calls == CATALOGS   # measurements: the saved copy
    turn("reject", history, state, kw)


def test_a_bad_age_setting_is_named_in_the_settings_message(monkeypatch):
    monkeypatch.setenv("APP_CATALOG_MAX_AGE_HOURS", "a-secret-value")
    [problem] = app.live_preflight("any question")
    assert "APP_CATALOG_MAX_AGE_HOURS" in problem and "secret" not in problem


def test_a_refresh_with_nothing_saved_is_a_first_download(place):
    """Review finding: with nothing saved, a refresh said the saved catalogs were set aside and "downloaded again"."""
    empty = place.root / "empty_cache"
    kw = kwargs_for(place, empty, max_age=None)
    history, state = turn("refresh catalogs", [], {}, kw)
    shown = history[-1]["content"]
    assert shown.startswith("First I need the public list of studies and traits") and "downloaded again" not in shown
    history, state = turn("approve", history, state, kw)
    assert place.server.calls == CATALOGS
    assert history[-1]["content"].startswith("**The study and trait catalogs were downloaded.**")
