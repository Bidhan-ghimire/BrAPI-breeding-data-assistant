"""
tests/test_plot_records.py — item (3) of 2026-10-22: plot records (observationunits?includeObservations=true) as the
measurement source. SweetPotatoBase's /observations leaves out every value recorded as 0 (2026-10-21: 7 of the 388 yield
values of study 4577); the plot records carry them. The client method fetches GET /observationunits with
includeObservations=true under the same allowlist, approval, timeouts, page cap and 429 rule as every other collection,
flattens each unit's observations into /observations-shaped rows, and keeps ONLY the approved variable's values: no other
trait's value ever leaves the client. Everything here uses the fake transport of test_brapi_client.py; zero real HTTP.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import brapi_client as bc
import brapi_mcp_server as srv
import chat_presenter as cp
import run as ctl
from brapi_client import BrapiClientError, TransportError
from cache_store import CacheIdentity, CacheStore
from contracts import FetchApproval, Plan
from part2_support import SYNTHETIC_BASE, synthetic_context
from plan_permissions import fetch_manifest
from test_brapi_client import BASE, T0, FakeTransport, approval, make_client, ok, page_body

SPB = "https://sweetpotatobase.org/brapi/v2"
CASSAVA = "https://cassavabase.org/brapi/v2"

PARAMS = {"studyDbId": "S1", "includeObservations": "true"}
PAGE = {**PARAMS, "page": 0, "pageSize": 250}
COLUMNS = ("observationDbId", "observationUnitDbId", "observationUnitName", "germplasmDbId", "germplasmName", "studyDbId",
           "observationVariableDbId", "observationVariableName", "observationTimeStamp", "value")


def unit(n, observations, **more):
    return {"observationUnitDbId": f"P{n}", "observationUnitName": f"S1-plot-{n}", "germplasmDbId": f"G{n}", "germplasmName": f"Clone {n}",
            "studyDbId": "S1", "observationUnitPosition": {"entryType": "test"}, "observations": observations, **more}


def obs(i, variable, value, **more):
    names = {"V1": "yield", "V2": "dry matter"}
    return {"observationDbId": f"O{i}", "observationVariableDbId": variable, "observationVariableName": names[variable], "value": value, **more}


def plot_approval(**kw):
    """The measurement approval a plan with get_observations gets on any server: the observations family, the study, the
    variable. Since the safety review of 2026-10-22 (C6) plot records are fetched under it, and nothing adds observationunits."""
    return approval(families=(), studies=["S1"], variables=["V1"], **kw)


# --- 1. the rows: one per observation of the approved variable, in the /observations shape -----------------------------

def test_plot_records_keep_only_the_approved_variable_in_observation_shaped_rows(tmp_path):
    page = [unit(1, [obs(1, "V1", "0", observationTimeStamp="2016-08-01"), obs(2, "V2", "31.5")]), unit(2, [obs(3, "V1", "12.5")]), unit(3, [])]
    transport = FakeTransport().add("/observationunits", PAGE, ok(page_body(page, total_count=3, page=0, page_size=250)))
    client = make_client(tmp_path, transport)
    result = client.plot_records("S1", "V1", approval=plot_approval())
    assert result.status == "complete" and result.complete and result.returned_count == 2 and len(transport.calls) == 1
    assert transport.calls[0].params == PAGE                                                     # GET, allowlisted path, 250 per page
    assert result.records[0] == {"observationDbId": "O1", "observationUnitDbId": "P1", "observationUnitName": "S1-plot-1", "germplasmDbId": "G1",
                                 "germplasmName": "Clone 1", "studyDbId": "S1", "observationVariableDbId": "V1", "observationVariableName": "yield",
                                 "observationTimeStamp": "2016-08-01", "value": "0"}            # the zero is kept
    assert [r["observationDbId"] for r in result.records] == ["O1", "O3"] and set(result.records[1]) == set(COLUMNS)
    text = json.dumps(result.records)
    assert "V2" not in text and "31.5" not in text and "dry matter" not in text                  # the other trait never leaves the client
    assert "observationUnitPosition" not in text


def test_plot_records_with_observations_keys_but_none_of_the_variable_are_empty(tmp_path):
    transport = FakeTransport().add("/observationunits", PAGE, ok(page_body([unit(1, [obs(1, "V2", "3")]), unit(2, [])], total_count=2)))
    client = make_client(tmp_path, transport)
    result = client.plot_records("S1", "V1", approval=plot_approval())
    assert result.status == "empty" and result.complete and result.returned_count == 0 and result.records == []


def test_plot_records_without_any_observations_key_are_unsupported_not_empty(tmp_path):
    """A server that ignores includeObservations returns units without an observations key: that is not "no values"."""
    bare = [{k: v for k, v in unit(n, []).items() if k != "observations"} for n in (1, 2)]
    transport = FakeTransport().add("/observationunits", PAGE, ok(page_body(bare, total_count=2)))
    client = make_client(tmp_path, transport)
    result = client.plot_records("S1", "V1", approval=plot_approval())
    assert result.status == "unsupported" and not result.complete and result.returned_count == 0 and result.records == []
    assert any("includeObservations" in w for w in result.warnings)


def test_no_units_at_all_is_an_empty_result(tmp_path):
    transport = FakeTransport().add("/observationunits", PAGE, ok(page_body([], total_count=0)))
    result = make_client(tmp_path, transport).plot_records("S1", "V1", approval=plot_approval())
    assert result.status == "empty" and result.complete and result.records == []


# --- 2. the gates: approval (study AND variable AND the observationunits family), plain IDs, page cap, timeouts ------------

def test_plot_records_need_the_measurement_approval_of_the_study_and_the_variable(tmp_path):
    """Safety review of 2026-10-22 (C6): plot records are values, so they need what /observations needs (the observations
    family, the study, the variable); an approval for plain plot lists of the study is not enough."""
    transport = FakeTransport().add("/observationunits", PAGE, ok(page_body([], total_count=0)))
    client = make_client(tmp_path, transport)
    other_study = client.plot_records("S2", "V1", approval=plot_approval())
    assert other_study.status == "failed" and transport.calls == [] and "not_authorized" in other_study.warnings[0]
    other_variable = client.plot_records("S1", "V9", approval=plot_approval())
    assert other_variable.status == "failed" and transport.calls == [] and "V9" in other_variable.warnings[0]
    lists_only = FetchApproval(approval_id="appr_lists", run_id="run_t", base_url=BASE, endpoint_families=["observationunits"],
                               observation_study_ids=["S1"], observation_variable_ids=[], max_http_attempts=5, max_observation_studies=1,
                               issued_at_utc=T0 - timedelta(minutes=1), expires_at_utc=T0 + timedelta(hours=1))
    plot_lists_only = client.plot_records("S1", "V1", approval=lists_only)
    assert plot_lists_only.status == "failed" and transport.calls == [] and "'observations'" in plot_lists_only.warnings[0]
    assert client.plot_records("S1", "V1", approval=plot_approval()).status == "empty" and len(transport.calls) == 1


def test_the_measurement_approval_never_opens_plain_plot_lists(tmp_path):
    """Safety review of 2026-10-22, C6 (blocking): the first batch 2 added observationunits to every get_observations approval
    on a listed server, so get_observation_units(S1) — not in the plan — sent a request and saved a plot list. Now the
    measurement approval reads the plot records and still refuses a plain plot list, with no request."""
    transport = FakeTransport().add("/observationunits", PAGE, ok(page_body([unit(1, [obs(1, "V1", "5")])], total_count=1)))
    client = make_client(tmp_path, transport)
    appr = plot_approval()
    assert client.plot_records("S1", "V1", approval=appr).complete and len(transport.calls) == 1
    plain = client.observation_units("S1", approval=appr)
    assert plain.status == "failed" and len(transport.calls) == 1 and "'observationunits'" in plain.warnings[0]


def test_only_a_plot_record_fetch_may_be_authorized_as_measurements(tmp_path):
    client = make_client(tmp_path, FakeTransport())
    for endpoint in ("/studies", "/observations", "/variables"):
        with pytest.raises(BrapiClientError):
            client.get_all(endpoint, {}, approval=plot_approval(), authorize_as="observations")
    with pytest.raises(BrapiClientError):
        client.get_all("/observationunits", {"studyDbId": "S1"}, approval=plot_approval(), authorize_as="studies")


def test_plot_records_refuse_ids_that_are_not_plain(tmp_path):
    client = make_client(tmp_path, FakeTransport())
    for study, variable in (("S1/x", "V1"), ("S 1", "V1"), ("", "V1"), ("S1", "V1/x"), ("S1", "")):
        with pytest.raises(BrapiClientError):
            client.plot_records(study, variable, approval=plot_approval())


def test_plot_records_first_take_one_page_and_stop_when_more_exist(tmp_path):
    rows = [unit(i, [obs(i, "V1", str(i))]) for i in range(250)]
    transport = FakeTransport().add("/observationunits", PAGE, ok(page_body(rows, total_count=750, page=0, page_size=250, total_pages=3)))
    result = make_client(tmp_path, transport).plot_records("S1", "V1", approval=plot_approval())
    assert result.status == "incomplete" and not result.complete and result.returned_count == 250 and len(transport.calls) == 1


def test_plot_records_paging_is_bounded_by_max_pages_and_stays_incomplete(tmp_path):
    transport = FakeTransport()
    for page in (0, 1, 2):
        rows = [unit(page * 250 + i, [obs(page * 250 + i, "V1", str(i))]) for i in range(250)]
        transport.add("/observationunits", {**PARAMS, "page": page, "pageSize": 250},
                      ok(page_body(rows, total_count=750, page=page, page_size=250, total_pages=3)))
    result = make_client(tmp_path, transport).plot_records("S1", "V1", approval=plot_approval(), paging=True, max_pages=2)
    assert result.status == "incomplete" and not result.complete and result.returned_count == 500 and len(transport.calls) == 2


def test_plot_records_paging_to_the_end_is_complete(tmp_path):
    transport = FakeTransport()
    for page, count in ((0, 250), (1, 100)):
        rows = [unit(page * 250 + i, [obs(page * 250 + i, "V1", "1"), obs(10000 + page * 250 + i, "V2", "2")]) for i in range(count)]
        transport.add("/observationunits", {**PARAMS, "page": page, "pageSize": 250},
                      ok(page_body(rows, total_count=350, page=page, page_size=250, total_pages=2)))
    result = make_client(tmp_path, transport).plot_records("S1", "V1", approval=plot_approval(), paging=True, max_pages=10)
    assert result.status == "complete" and result.complete and result.returned_count == 350 and len(transport.calls) == 2
    assert all(r["observationVariableDbId"] == "V1" for r in result.records)


def test_plot_record_timeouts_are_never_retried(tmp_path):
    transport = FakeTransport().add("/observationunits", PAGE, TransportError("timeout", "slow", retryable=True), ok(page_body([], total_count=0)))
    client = make_client(tmp_path, transport, max_attempts=3)
    result = client.plot_records("S1", "V1", approval=plot_approval())
    assert result.status == "failed" and len(transport.calls) == 1


# --- 3. the switch: on a server that leaves zeros out of /observations, get_observations reads the plot records ------------

def seed_plot_records(cache_dir, units):
    body = json.dumps({"metadata": {"pagination": {"currentPage": 0, "pageSize": 250, "totalCount": len(units), "totalPages": 1}},
                       "result": {"data": units}}).encode("utf-8")
    identity = CacheIdentity(base_url=SYNTHETIC_BASE, endpoint="/observationunits", params={**PARAMS, "page": 0, "pageSize": 250})
    CacheStore(cache_dir).put(identity, body, http_status=200, fetched_at_utc=datetime(2026, 6, 1, tzinfo=timezone.utc),
                              source_url_redacted=f"{SYNTHETIC_BASE}/observationunits", content_type="application/json")


def obs_plan(base, variables=("V1",)):
    return Plan.model_validate({
        "plan_id": "plan_plot_records", "question": "Synthetic plot-record check", "interpretation": "Traits of one study",
        "status": "ready", "created_at_utc": datetime.now(timezone.utc),
        "scope": {"base_url": base, "study_ids": ["S1"], "variable_ids": list(variables)},
        "steps": [{"step_id": f"step_{i}", "agent": "retriever", "action": "get_observations",
                   "inputs": {"study_db_id": "S1", "variable_db_id": v}} for i, v in enumerate(variables, 1)]})


@pytest.mark.parametrize("url, expected", [(SPB, True), ("https://www.sweetpotatobase.org/brapi/v2", True),
                                           ("https://SweetPotatoBase.org:443/brapi/v2", True), (CASSAVA, False),
                                           ("https://notsweetpotatobase.org/brapi/v2", False),
                                           ("https://sweetpotatobase.org.evil.invalid/brapi/v2", False), (SYNTHETIC_BASE, False), ("", False)])
def test_plot_records_are_read_only_on_the_listed_servers(url, expected):
    assert bc.reads_plot_records(url) is expected


def test_get_observations_reads_the_plot_records_on_a_listed_server(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "PLOT_RECORD_HOSTS", ("synthetic.invalid",))
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_pr")
    seed_plot_records(tmp_path / "cache", [unit(1, [obs(1, "V1", "0"), obs(2, "V2", "31.5")]), unit(2, [obs(3, "V1", "12.5")])])
    r = srv.dispatch(ctx, "get_observations", {"study_db_id": "S1", "variable_db_id": "V1"})
    assert r.ok and r.complete and r.data["row_count"] == 2
    manifest, rows = ctx.registry.load(r.data["artifact_id"])
    assert [row["value"] for row in rows] == ["0", "12.5"] and manifest.kind == "observations" and manifest.variable_ids == ["V1"]
    assert any(note.startswith(bc.PLOT_RECORDS_NOTE) for note in manifest.quality_notes)
    saved = "".join(p.read_text(encoding="utf-8") for p in (tmp_path / "out").rglob("*.csv"))
    assert "31.5" not in saved and "dry matter" not in saved                                  # the other trait was never saved


def test_get_observations_keeps_the_observations_service_on_other_servers(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_obs")
    r = srv.dispatch(ctx, "get_observations", {"study_db_id": "S1", "variable_db_id": "V1"})
    manifest = ctx.registry.manifest(r.data["artifact_id"])
    assert r.ok and r.data["row_count"] == 6 and not any(note.startswith(bc.PLOT_RECORDS_NOTE) for note in manifest.quality_notes)


def test_the_approval_of_a_listed_server_is_the_same_as_elsewhere():
    """Safety review of 2026-10-22 (C6): the plot records are read under the measurement approval, so a listed server's
    approval asks for exactly what another server's does — no plot-list family is added."""
    listed = fetch_manifest(obs_plan(SPB), run_id="run_pr", base_url=SPB, max_http_attempts=10)
    other = fetch_manifest(obs_plan(CASSAVA), run_id="run_pr", base_url=CASSAVA, max_http_attempts=10)
    assert listed.endpoint_families == other.endpoint_families and "observationunits" not in listed.endpoint_families
    assert listed.observation_study_ids == ["S1"] and listed.observation_variable_ids == ["V1"]


def test_the_approval_screen_says_other_traits_are_carried_and_dropped_on_a_listed_server():
    """Safety review of 2026-10-22 (C8): the screen said "other observations are excluded" while the plot records bring every
    trait's values; on a listed server it now says they are dropped, elsewhere it is unchanged."""
    def screen(base):
        plan = obs_plan(base)
        manifest = fetch_manifest(plan, run_id="run_pr", base_url=base, max_http_attempts=10)
        return cp.approval_summary(SimpleNamespace(plan=plan, manifest=manifest, metadata=None, catalog_fetched_at="2026-10-22",
                                                   catalog_setup=None, config=SimpleNamespace(question=plan.question, max_fetches=5),
                                                   budget=SimpleNamespace(max_model_calls=30)))
    listed, other = screen(SPB), screen(CASSAVA)
    assert "other observations are excluded" in other and "other observations are excluded" not in listed
    assert ("the plot records also carry other traits' values, which are dropped before any table, preview or answer (the raw "
            "pages stay in the local request cache)") in listed and "before anything is saved" not in listed


def test_the_approval_text_says_the_values_come_from_the_plot_records():
    plan = obs_plan(SPB)
    text = ctl.approval_lead(fetch_manifest(plan, run_id="run_pr", base_url=SPB, max_http_attempts=10), plan)
    assert "read from the plot records, which keep values recorded as zero" in text and "recorded as 0" not in text
    assert "the plot records also carry other traits' values; only the approved trait is used" in text and "is kept" not in text
    two = obs_plan(SPB, ("V1", "V2"))
    assert "only the 2 approved traits are used" in ctl.approval_lead(fetch_manifest(two, run_id="run_pr", base_url=SPB, max_http_attempts=10), two)
    plan = obs_plan(CASSAVA)
    assert "plot records" not in ctl.approval_lead(fetch_manifest(plan, run_id="run_pr", base_url=CASSAVA, max_http_attempts=10), plan)


def test_the_zeros_note_is_left_out_for_tables_read_from_the_plot_records():
    def controller(*notes_per_table):
        tables = [SimpleNamespace(kind="observations", quality_notes=list(notes)) for notes in notes_per_table]
        return SimpleNamespace(base_url=SPB, retrieval=SimpleNamespace(artifacts=tables))
    from_plot_records = [bc.PLOT_RECORDS_NOTE + ": 3 units read, 2 values of variable V1 kept"]
    assert not cp._zeros_left_out(controller(from_plot_records))
    assert cp._zeros_left_out(controller(from_plot_records, []))                         # one table still from /observations
    assert cp._zeros_left_out(SimpleNamespace(base_url=SPB, retrieval=SimpleNamespace(artifacts=[SimpleNamespace(kind="observations")])))


def test_one_list_of_servers_for_the_switch_and_the_note(monkeypatch):
    monkeypatch.setattr(bc, "PLOT_RECORD_HOSTS", ("example.invalid",))
    other = SimpleNamespace(base_url="https://example.invalid/brapi/v2", retrieval=SimpleNamespace(artifacts=[SimpleNamespace(kind="observations")]))
    assert bc.reads_plot_records(other.base_url) and cp._zeros_left_out(other) and not bc.reads_plot_records(SPB)


# --- 4. completeness on made-up pages (safety review of 2026-10-22, C11 and C12.1; none of it seen on real data) -----------

def plot_result(tmp_path, units, **page):
    transport = FakeTransport().add("/observationunits", PAGE, ok(page_body(units, **({"total_count": len(units)} | page))))
    return make_client(tmp_path, transport).plot_records("S1", "V1", approval=plot_approval())


def test_a_plot_without_observations_among_plots_with_them_is_not_complete(tmp_path):
    bare = {k: v for k, v in unit(2, []).items() if k != "observations"}
    r = plot_result(tmp_path, [unit(1, [obs(1, "V1", "5")]), bare, unit(3, [obs(3, "V1", "7")])])
    assert not r.complete and r.status == "unknown" and r.returned_count == 2
    assert any("1 of 3 plots came without readable observations" in w for w in r.warnings)


@pytest.mark.parametrize("bad", [{"O1": "5"}, "5", None])
def test_observations_that_are_not_a_list_are_not_read_as_no_values(tmp_path, bad):
    r = plot_result(tmp_path, [unit(1, bad), unit(2, bad)])
    assert not r.complete and r.status in ("unsupported", "unknown") and r.records == []


def test_one_observation_listed_under_two_plots_is_not_complete(tmp_path):
    r = plot_result(tmp_path, [unit(1, [obs(1, "V1", "5")]), unit(2, [obs(1, "V1", "5")])])
    assert not r.complete and r.status == "unknown" and any("listed twice" in w for w in r.warnings)


def test_a_plot_of_another_study_is_dropped_and_the_result_is_not_complete(tmp_path):
    r = plot_result(tmp_path, [unit(1, [obs(1, "V1", "5")]), unit(2, [obs(2, "V1", "9")], studyDbId="S9")])
    assert not r.complete and r.status == "unknown" and [row["value"] for row in r.records] == ["5"]
    assert any("1 plot of another study" in w for w in r.warnings)


def test_a_failed_later_page_is_never_turned_into_unsupported(tmp_path):
    bare = [{k: v for k, v in unit(n, []).items() if k != "observations"} for n in range(250)]
    transport = FakeTransport().add("/observationunits", PAGE, ok(page_body(bare, total_count=400, page=0, page_size=250, total_pages=2)))
    transport.add("/observationunits", {**PARAMS, "page": 1, "pageSize": 250}, TransportError("timeout", "slow", retryable=True))
    r = make_client(tmp_path, transport).plot_records("S1", "V1", approval=plot_approval(), paging=True, max_pages=2)
    assert r.status == "failed" and not r.complete


# --- 5. after the second safety review (2026-10-22, REVIEW_round2.md; none of it seen on real data) ------------------------

def test_a_plot_record_fetch_must_ask_for_the_observations(tmp_path):
    """R2: the measurement check looked at the endpoint only, so a plain plot list could have been sent under it."""
    client = make_client(tmp_path, FakeTransport())
    for params in ({"studyDbId": "S1"}, {"studyDbId": "S1", "includeObservations": "false"}):
        with pytest.raises(BrapiClientError):
            client.get_all("/observationunits", params, approval=plot_approval(), authorize_as="observations", study_id="S1", variable_id="V1")


@pytest.mark.parametrize("entry", ["5", 7, None, {"observationDbId": "O9", "value": "5"}, {"observationDbId": "O9", "observationVariableDbId": "", "value": "5"}],
                         ids=["string", "number", "null", "no-variable", "blank-variable"])
def test_an_observation_that_cannot_be_read_makes_the_result_unknown(tmp_path, entry):
    r = plot_result(tmp_path, [unit(1, [obs(1, "V1", "5"), entry])])
    assert not r.complete and r.status == "unknown" and [row["value"] for row in r.records] == ["5"]
    assert any("could not be read" in w for w in r.warnings)


def test_a_value_without_an_observation_id_makes_the_result_unknown(tmp_path):
    """R4 (b): two values without an observationDbId could be one value listed twice; that cannot be seen, so no proof."""
    r = plot_result(tmp_path, [unit(1, [obs(1, "V1", "5", observationDbId=None)]), unit(2, [obs(2, "V1", "5", observationDbId=None)])])
    assert not r.complete and r.status == "unknown" and any("without an observation ID" in w for w in r.warnings)


def test_the_warning_says_where_the_other_traits_go(tmp_path):
    r = plot_result(tmp_path, [unit(1, [obs(1, "V1", "5"), obs(2, "V2", "31.5")])])
    note = next(w for w in r.warnings if w.startswith(bc.PLOT_RECORDS_NOTE))
    assert "before any table, preview or answer" in note and "local request cache" in note and "before anything was saved" not in note
