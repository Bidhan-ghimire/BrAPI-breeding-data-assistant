"""
tests/test_standing_approval.py — the operator's standing approval for the two public catalogs (Part D, Step 3), offline.

What these tests prove, in plain words:
* Setting off (the default): with no saved catalogs, the visitor sees the catalog screen exactly as before, nothing is
  requested until they type approve, and the run records that a person approved it.
* Setting on: the visitor never sees the catalog screen. The catalog setup is approved by code from the operator's setting,
  recorded as such (issued_by in catalog_setup, an "operator" approval ID, the contract's operator_config), and only the
  study and trait catalogs are requested. The data request still shows its own approval screen and needs the visitor's
  typed approve: cancel fetches no measurement; approve fetches them under a person's approval.
* Without a valid review date in the access notes (a real date, not in the future) the setting is refused: a run is
  blocked before any request, and the app refuses to start, before it launches. Who approved is recorded only with an
  approval. A catalog list says the operator approved its catalog setup. The contract itself lets an operator approval
  cover only the two catalogs, never observations.
* At startup (app.prepare_catalogs): off, nothing happens; on, missing catalogs are prepared once, as a recorded run with no
  planner, no model call and no measurements; saved catalogs are not requested again.
* With the setting on, the help page says when the saved catalogs were downloaded and that the operator approves a new
  download, read again when the page opens; with it off, the help page reads exactly as before. A bad setting stops the
  startup only when the standing approval is on (or is itself the bad setting).
Fake server and scripted models only; the conftest guard would stop any real socket.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import app
import catalog_summary
import run as ctl
from brapi_client import TransportResponse
from cache_store import CacheStore
from contracts import FetchApproval, FetchManifest
from part2_support import load_fixture, part2_runtime, seed_fixture, synthetic_models

LIVE_BASE = "https://standing.invalid/brapi/v2"
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
    """Live settings for this test only, an EMPTY catalog cache, a fake server, reviewed access notes and the app's factories."""
    import brapi_client as bc

    monkeypatch.setenv("BRAPI_MODE", "live")
    monkeypatch.setenv("BRAPI_BASE_URL", LIVE_BASE)
    server = FakeLiveServer()
    monkeypatch.setattr(bc, "RequestsTransport", lambda: server)
    notes = tmp_path / "ACCESS_NOTES.md"
    notes.write_text("Reviewed by Bidhan on: 2026-09-28\n", encoding="utf-8")

    def config_for(standing: bool, notes_path: Path = notes):
        def make_config(question):
            return ctl.RunConfig(question=question, offline=False, mock_model=False, direct=False, auto=False,
                                 out_dir=tmp_path / "out", cache_dir=tmp_path / "live_cache", access_notes=notes_path,
                                 max_clarification_rounds=app.CHAT_CLARIFICATION_ROUNDS, bootstrap_catalogs=True,
                                 catalog_standing_approval=standing)
        return make_config

    def kwargs(standing: bool, notes_path: Path = notes):
        make_config = config_for(standing, notes_path)
        return dict(make_config=make_config, make_models=synthetic_models, preflight=lambda q: make_config(q).validate())

    catalog_summary.forget()
    yield dataclasses.make_dataclass("Place", ["server", "root", "notes", "config_for", "kwargs"])(
        server, tmp_path, notes, config_for, kwargs)
    catalog_summary.forget()


def turn(message, history, state, kwargs):
    out = list(app.respond(message, history, state, **kwargs))
    assert out[-1][2] == ""
    return out[-1][0], out[-1][1]


def saved_run(root: Path, run_id: str) -> tuple[ctl.RunResult, dict]:
    folder = root / "out" / run_id
    return (ctl.RunResult.model_validate_json((folder / "answer.json").read_text(encoding="utf-8")),
            json.loads((folder / "manifest.json").read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# A visitor's question with no saved catalogs
# --------------------------------------------------------------------------

def test_off_the_visitor_sees_the_catalog_screen_exactly_as_before(place):
    history, state = turn(Q, [], {}, place.kwargs(False))
    assert "Reply **approve** to prepare these catalogs, or **cancel**." in history[-1]["content"]
    assert state["job"].waiting_on == ctl.CATALOG_PROMPT and place.server.calls == []          # nothing until approve
    job = state["job"]
    history, state = turn("approve", history, state, place.kwargs(False))
    assert place.server.calls == CATALOGS and "Review this plan before approving" in history[-1]["content"]
    history, state = turn("cancel", history, state, place.kwargs(False))
    result, manifest = saved_run(place.root, job.controller.run_id)
    assert manifest["catalog_setup"]["issued_by"] == ctl.PERSON_ISSUER
    assert "_human_" in result.catalog_approval.approval_id and result.catalog_approval.issued_by == "human_controller"
    assert result.approval_id is None and place.server.calls == CATALOGS                        # canceled: no measurement


def test_on_the_operator_approves_the_two_catalogs_and_the_visitor_sees_only_the_data_request(place):
    history, state = turn(Q, [], {}, place.kwargs(True))
    shown = history[-1]["content"]
    assert "prepare these catalogs" not in shown and "Review this plan before approving" in shown   # no catalog screen
    assert place.server.calls == CATALOGS                                                       # the two catalogs only
    job = state["job"]
    assert job.waiting_on.startswith("approve / edit / cancel") and job.controller.approval is None
    history, state = turn("cancel", history, state, place.kwargs(True))
    result, manifest = saved_run(place.root, job.controller.run_id)
    setup = manifest["catalog_setup"]
    assert setup["issued_by"] == ctl.OPERATOR_ISSUER and setup["status"] == "completed"
    assert setup["manifest"]["endpoint_families"] == ["studies", "observationvariables"]
    assert setup["manifest"]["max_observation_studies"] == 0 and setup["approval"]["observation_study_ids"] == []
    assert "_operator_" in setup["approval"]["approval_id"] and setup["approval"]["issued_by"] == "operator_config"
    assert result.catalog_approval.issued_by == "operator_config" and result.approval_id is None
    assert any("standing approval (APP_CATALOG_STANDING_APPROVAL)" in note for note in result.notes)
    assert place.server.calls == CATALOGS and result.counts.live_attempts == 2                  # no measurement requested


def test_on_measurements_still_need_the_visitors_typed_approve(place):
    history, state = turn(Q, [], {}, place.kwargs(True))
    job = state["job"]
    history, state = turn("approve", history, state, place.kwargs(True))                        # the visitor's own approve
    assert place.server.calls == CATALOGS + ["/observations", "/observationunits"]
    assert "Reply **accept**" in history[-1]["content"]
    history, state = turn("reject", history, state, place.kwargs(True))
    result, _ = saved_run(place.root, job.controller.run_id)
    assert "_human_" in result.approval_id and result.catalog_approval.issued_by == "operator_config"


def test_on_without_the_review_date_nothing_is_requested_and_the_app_refuses_to_start(place):
    unreviewed = place.root / "UNREVIEWED.md"
    unreviewed.write_text("Not reviewed yet.\n", encoding="utf-8")
    history, state = turn(Q, [], {}, place.kwargs(True, unreviewed))
    assert "cannot run yet" in history[-1]["content"] and "APP_CATALOG_STANDING_APPROVAL needs the access notes reviewed" in history[-1]["content"]
    assert place.server.calls == [] and state == {}
    with pytest.raises(SystemExit, match="refusing to start: APP_CATALOG_STANDING_APPROVAL is on"):
        app.prepare_catalogs(place.config_for(True, unreviewed))
    assert place.server.calls == [] and not (place.root / "out").exists()


def test_the_bootstrap_itself_checks_the_review_date_again(place, monkeypatch):
    """A run whose notes lose their review line after the check at the start is still refused at the catalog setup."""
    config = place.config_for(True)(Q)
    controller = ctl.Controller(config, models=synthetic_models(), answer_fn=lambda prompt: "cancel")
    import asyncio

    monkeypatch.setattr(ctl, "access_notes_review_date", lambda path=None: "2026-09-28")   # passes the start check ...
    original = ctl.Controller._bootstrap_metadata_mcp

    async def lose_the_review(self):
        monkeypatch.setattr(ctl, "access_notes_review_date", lambda path=None: None)      # ... then the line is gone
        return await original(self)
    monkeypatch.setattr(ctl.Controller, "_bootstrap_metadata_mcp", lose_the_review)
    result = asyncio.run(controller.run())
    assert result.execution_status == "blocked" and place.server.calls == []
    assert controller.catalog_setup["status"] == "blocked" and controller.catalog_setup["approval"] is None
    assert controller.catalog_setup["issued_by"] is None                          # no approval: nobody is recorded as approving


# --------------------------------------------------------------------------
# The contract and the approval itself
# --------------------------------------------------------------------------

def catalog_manifest(run_id: str = "run_contract", **over) -> FetchManifest:
    values = dict(run_id=run_id, base_url=LIVE_BASE, origin="live", endpoint_families=["studies", "observationvariables"],
                  max_http_attempts=ctl.CATALOG_MAX_HTTP_ATTEMPTS, max_observation_studies=0)
    values.update(over)
    return FetchManifest(**values)


def test_an_operator_approval_covers_only_the_two_catalogs():
    approval = ctl.build_approval(catalog_manifest(), issued_by_human=False, operator=True)
    assert approval.issued_by == "operator_config" and approval.approval_id.startswith("appr_run_contract_operator_")
    assert ctl.scope_mismatches(catalog_manifest(), approval) == []
    wider = catalog_manifest(endpoint_families=["studies", "observationvariables", "observations", "observationunits"],
                             observation_study_ids=["S1"], observation_variable_ids=["V1"], max_observation_studies=1)
    with pytest.raises(ValidationError, match="covers only the study and trait catalogs"):
        ctl.build_approval(wider, issued_by_human=False, operator=True)
    with pytest.raises(ValidationError, match="covers only the study and trait catalogs"):
        FetchApproval(**{**approval.model_dump(), "endpoint_families": ["studies"]})
    human = ctl.build_approval(wider, issued_by_human=True)                       # a person's approval is unchanged
    assert human.issued_by == "human_controller" and "_human_" in human.approval_id


@pytest.mark.parametrize("value, expected", [(None, False), ("", False), ("false", False), ("FALSE", False), ("true", True),
                                             (" True ", True)])
def test_the_setting_is_read_as_true_or_false(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("APP_CATALOG_STANDING_APPROVAL", raising=False)
    else:
        monkeypatch.setenv("APP_CATALOG_STANDING_APPROVAL", value)
    assert app.live_config("any question").catalog_standing_approval is expected


def test_a_bad_setting_is_refused_without_echoing_it(monkeypatch):
    monkeypatch.setenv("APP_CATALOG_STANDING_APPROVAL", "yes-please-secret")
    with pytest.raises(ValueError) as caught:
        app.live_config("any question")
    assert str(caught.value) == "APP_CATALOG_STANDING_APPROVAL must be true or false" and "secret" not in str(caught.value)
    assert any("App settings are invalid" in p for p in app.live_preflight("any question"))
    with pytest.raises(SystemExit, match="refusing to start: APP_CATALOG_STANDING_APPROVAL must be true or false"):
        app.prepare_catalogs()


# --------------------------------------------------------------------------
# At startup
# --------------------------------------------------------------------------

def test_startup_with_the_setting_off_does_nothing(place):
    assert app.prepare_catalogs(place.config_for(False)).startswith("catalog standing approval: off")
    assert place.server.calls == [] and not (place.root / "out").exists()


def test_startup_prepares_missing_catalogs_once_as_a_recorded_run(place):
    models = []

    def make_models():
        made = synthetic_models()
        models.append(made)
        return made

    line = app.prepare_catalogs(place.config_for(True), make_models=make_models)
    assert line.startswith("catalog standing approval: on; catalog setup completed (run run_") and place.server.calls == CATALOGS
    run_id = line.rsplit("(run ", 1)[1].rstrip(")")
    result, manifest = saved_run(place.root, run_id)
    assert result.execution_status == "completed" and result.counts.model_calls == 0 and result.approval_id is None
    assert manifest["catalog_setup"]["issued_by"] == ctl.OPERATOR_ISSUER and manifest["state_trace"][-2:] == ["catalog_setup", "record"]
    assert [c.value for c in result.analysis.claims] == [1, 2]                     # 1 study, 2 traits: counted by code
    assert "no planner" in result.plan.interpretation and result.human_acceptance == "pending"
    again = app.prepare_catalogs(place.config_for(True), make_models=make_models)
    assert again == "catalog standing approval: on; the study and trait catalogs are already saved, nothing was requested"
    assert place.server.calls == CATALOGS and len(list((place.root / "out").glob("run_*"))) == 1


def test_after_the_startup_setup_a_question_goes_straight_to_its_own_approval(place):
    app.prepare_catalogs(place.config_for(True))
    history, state = turn(Q, [], {}, place.kwargs(True))
    assert "Review this plan before approving" in history[-1]["content"] and place.server.calls == CATALOGS
    job = state["job"]
    turn("cancel", history, state, place.kwargs(True))
    _, manifest = saved_run(place.root, job.controller.run_id)
    assert manifest["catalog_setup"] is None                                       # the saved catalogs were used


def test_a_catalog_setup_run_is_refused_for_anything_else(place):
    config = place.config_for(True)(Q)
    assert any("lists nothing and continues no question" in p
               for p in dataclasses.replace(config, catalog_setup_only=True, catalog_list="traits").validate())
    assert any("needs bootstrap_catalogs" in p
               for p in dataclasses.replace(config, catalog_setup_only=True, bootstrap_catalogs=False).validate())


# --------------------------------------------------------------------------
# The help page
# --------------------------------------------------------------------------

RULE = ("When they need downloading, the app's operator approves it (a standing approval); measurements always need your "
        "approval. Type **refresh catalogs** to download them again.")


def save_catalogs(place) -> None:
    cache = CacheStore(place.root / "live_cache")
    for name in ("synthetic_studies.json", "synthetic_variables.json"):
        seed_fixture(cache, LIVE_BASE, name)
    catalog_summary.forget()


def test_the_help_page_says_when_the_catalogs_were_downloaded_and_who_approves_only_when_on(place):
    assert app.catalog_source_line(place.config_for(False)) == ""                  # off: the help page reads as before
    assert app.source_details(place.config_for(False)) == app.source_markdown()
    assert app.catalog_source_line(place.config_for(True)) == f"The study and trait catalogs are not saved yet. {RULE}"
    save_catalogs(place)
    assert app.catalog_source_line(place.config_for(True)) == (
        f"The saved study and trait catalogs were downloaded 2026-06-01 00:00 UTC. {RULE}")
    assert app.catalog_source_line(place.config_for(False)) == ""
    details = app.source_details(place.config_for(True))
    assert details.startswith(app.source_markdown()) and details.endswith("Type **refresh catalogs** to download them again.")
    assert app.catalog_source_line(lambda q: (_ for _ in ()).throw(ValueError("bad setting"))) == ""


def test_the_help_page_reads_the_catalog_line_again_when_the_page_opens(place):
    pytest.importorskip("gradio")
    make_config = place.config_for(True)
    ui = app.build_ui(make_config=make_config, make_models=synthetic_models, preflight=lambda q: [])
    [load] = [f for f in ui.fns.values() if any(target[1] == "load" for target in f.targets)]
    [shown] = load.outputs
    assert shown.value == app.source_details(make_config) and "not saved yet" in shown.value
    assert app.SOURCE_FAQ in [question for question, _ in app.faq_items("x")]
    save_catalogs(place)                                                            # e.g. the startup setup, after the build
    assert load.fn() == app.source_details(make_config) and "downloaded 2026-06-01 00:00 UTC" in load.fn()
    off = app.build_ui(make_config=place.config_for(False), make_models=synthetic_models, preflight=lambda q: [])
    assert not [f for f in off.fns.values() if any(target[1] == "load" for target in f.targets)]   # off: no page-load event


def test_who_approved_is_recorded_only_with_an_approval(place):
    history, state = turn(Q, [], {}, place.kwargs(False))
    job = state["job"]
    turn("cancel", history, state, place.kwargs(False))                            # the visitor cancels the catalog screen
    _, manifest = saved_run(place.root, job.controller.run_id)
    setup = manifest["catalog_setup"]
    assert (setup["status"], setup["approval"], setup["issued_by"]) == ("canceled", None, None) and place.server.calls == []


@pytest.mark.parametrize("line", ["Reviewed by Bidhan on: 2099-12-31", "Reviewed by Bidhan on: 2026-13-45",
                                  "Reviewed by Bidhan on: 0000-00-00"], ids=["future", "no-such-month", "zeros"])
def test_the_standing_approval_needs_a_real_review_date_that_is_not_in_the_future(place, line):
    """Review finding (2026-10-16): any YYYY-MM-DD counted as reviewed, and startup then fetched the catalogs."""
    notes = place.root / "ODD_NOTES.md"
    notes.write_text(line + "\n", encoding="utf-8")
    assert any("with a real date that is not in the future" in p for p in place.config_for(True, notes)(Q).validate())
    with pytest.raises(SystemExit, match="no valid 'Reviewed by Bidhan on"):
        app.prepare_catalogs(place.config_for(True, notes))
    assert place.server.calls == [] and not (place.root / "out").exists()
    assert not any("APP_CATALOG_STANDING_APPROVAL" in p for p in place.config_for(False, notes)(Q).validate())   # off: as before


def test_a_review_date_counts_from_its_day_on(tmp_path):
    from datetime import date

    notes = tmp_path / "ACCESS_NOTES.md"
    notes.write_text("Reviewed by Bidhan on: 2026-09-28\n", encoding="utf-8")
    assert ctl.standing_review_date(notes, today=date(2026, 9, 28)) == date(2026, 9, 28)
    assert ctl.standing_review_date(notes, today=date(2026, 9, 27)) is None


def test_a_bad_setting_stops_startup_only_when_the_standing_approval_is_on_or_is_the_bad_setting(monkeypatch):
    """Review finding: with the setting off, a bad APP_READ_TIMEOUT stopped the Space; before, it started and each question
    said the settings were invalid."""
    monkeypatch.setenv("APP_READ_TIMEOUT", "not-a-number")
    monkeypatch.delenv("APP_CATALOG_STANDING_APPROVAL", raising=False)
    line = app.prepare_catalogs()
    assert line.startswith("catalog standing approval: off") and "another app setting is invalid" in line
    assert any("APP_CATALOG_STANDING_APPROVAL" in p for p in app.live_preflight("any question"))   # named among the settings
    monkeypatch.setenv("APP_CATALOG_STANDING_APPROVAL", "true")
    with pytest.raises(SystemExit, match="refusing to start: APP_READ_TIMEOUT must be a number"):
        app.prepare_catalogs()


def test_on_a_catalog_list_says_the_operator_approved_its_catalog_setup(place):
    """Review finding: "list all traits" with the setting on said "(fetched after you approved the catalog setup)", though the
    visitor approved nothing."""
    history, state = turn("list all traits", [], {}, place.kwargs(True))
    shown = history[-1]["content"]
    assert "fetched after the app's operator approved the catalog setup (standing approval)" in shown
    assert "you approved" not in shown and place.server.calls == CATALOGS and "job" not in state


def test_the_space_refuses_to_start_before_it_launches(monkeypatch, tmp_path):
    from test_app_space import load_launcher

    module, calls, _ = load_launcher(monkeypatch)
    unreviewed = tmp_path / "UNREVIEWED.md"
    unreviewed.write_text("Not reviewed yet.\n", encoding="utf-8")
    monkeypatch.setattr(module, "space_config", lambda q: ctl.RunConfig(question=q, offline=False, mock_model=False, direct=False,
                                                                        auto=False, out_dir=tmp_path / "out", access_notes=unreviewed,
                                                                        bootstrap_catalogs=True, catalog_standing_approval=True))
    with pytest.raises(SystemExit, match="refusing to start"):
        module.main()
    assert not any(name == "launch" for name, _ in calls) and not (tmp_path / "out").exists()
