"""
tests/test_brapi_client.py — the client goes online only when allowed, never
follows a model-supplied path, and never calls a collection complete without
the server's own counts to prove it.

Everything here uses a FAKE transport: a small in-memory "server" that
returns scripted replies. Zero real HTTP happens (conftest blocks it anyway).
All data is SYNTHETIC.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

import brapi_client as bc
from brapi_client import (
    BrapiClient,
    BrapiClientError,
    NotAuthorized,
    OfflineCacheMiss,
    Settings,
    TransportError,
    TransportResponse,
    records_to_df,
)
from cache_store import CacheIdentity, CacheStore
from contracts import FetchApproval, StudyFilters

BASE = "https://cassavabase.org/brapi/v2"
T0 = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)


# --- a scripted fake server ---------------------------------------------------------

def page_body(data, *, total_count=None, page=0, page_size=None, total_pages=None, result_null=False,
              pagination=True) -> bytes:
    meta = {}
    if pagination:
        pag = {"currentPage": page}
        if page_size is not None:
            pag["pageSize"] = page_size
        if total_count is not None:
            pag["totalCount"] = total_count
        if total_pages is not None:
            pag["totalPages"] = total_pages
        meta["pagination"] = pag
    envelope = {"metadata": meta, "result": None if result_null else {"data": data}}
    return json.dumps(envelope).encode("utf-8")


def ok(body: bytes, **headers) -> TransportResponse:
    return TransportResponse(200, {"Content-Type": "application/json", **headers}, body)


class FakeTransport:
    """Scripted replies keyed by (endpoint, params). Each key holds a queue of replies.

    A reply is a TransportResponse, an exception to raise, or a callable(req) -> reply.
    """

    def __init__(self, clock=None):
        self.queues: dict[tuple, list] = {}
        self.calls: list[bc.TransportRequest] = []
        self.clock = clock

    @staticmethod
    def key(endpoint: str, params: dict | None = None):
        return (endpoint, tuple(sorted((params or {}).items())))

    def add(self, endpoint: str, params: dict | None, *replies):
        self.queues.setdefault(self.key(endpoint, params), []).extend(replies)
        return self

    def __call__(self, req: bc.TransportRequest) -> TransportResponse:
        self.calls.append(req)
        endpoint = req.url[len(BASE):]
        queue = self.queues.get(self.key(endpoint, req.params))
        if not queue:
            raise AssertionError(f"fake server has no reply for {endpoint} {req.params}")
        reply = queue.pop(0)
        if callable(reply) and not isinstance(reply, TransportResponse):
            reply = reply(req)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_client(tmp_path, transport=None, *, mode="live", clock=None, sleeps=None, **settings_over):
    settings = Settings(base_url=BASE, mode=mode, cache_dir=tmp_path / "cache", **settings_over)
    clock = clock or FakeClock()
    sleeps = sleeps if sleeps is not None else []
    return BrapiClient(settings, transport=transport, cache=CacheStore(settings.cache_dir), run_id="run_t",
                       clock=clock, now_utc=lambda: T0, sleeper=sleeps.append)


def approval(*, families=("studies", "observationvariables", "locations", "programs", "seasons"),
             studies=(), variables=(), max_attempts=20, max_studies=0, expires_hours=1, base=BASE):
    families = list(families)
    if studies and "observations" not in families:
        families.append("observations")
    return FetchApproval(approval_id="appr_t", run_id="run_t", base_url=base, endpoint_families=families,
                         observation_study_ids=list(studies), observation_variable_ids=list(variables),
                         max_http_attempts=max_attempts, max_observation_studies=max_studies or len(studies),
                         issued_at_utc=T0 - timedelta(minutes=1), expires_at_utc=T0 + timedelta(hours=expires_hours))


def studies_page(start: int, count: int, **kw) -> bytes:
    return page_body([{"studyDbId": f"S{i}", "studyName": f"study {i}"} for i in range(start, start + count)], **kw)


# --- 1. settings: offline by default, loaded relative to this file --------------------

def test_settings_default_to_offline_and_ignore_the_caller_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                       # the caller's folder is irrelevant
    settings = bc.load_settings(env_file=tmp_path / "no-such.env", environ={})
    assert settings.mode == "offline" and settings.base_url == BASE
    assert settings.cache_dir == bc.PART2_DIR / "cache"
    env_file = tmp_path / "custom.env"
    env_file.write_text("BRAPI_BASE_URL=https://test-server.brapi.org/brapi/v2/\nBRAPI_MODE=live\n", encoding="utf-8")
    loaded = bc.load_settings(env_file=env_file, environ={})
    assert loaded.base_url == "https://test-server.brapi.org/brapi/v2" and loaded.mode == "live"
    overridden = bc.load_settings(env_file=env_file, environ={"BRAPI_MODE": "offline"})   # real env wins
    assert overridden.mode == "offline"
    with pytest.raises(BrapiClientError):
        bc.load_settings(env_file=env_file, environ={"BRAPI_MODE": "maybe"})


# --- 2. offline: cache or nothing ------------------------------------------------------

def test_offline_cache_miss_is_an_error_and_transport_is_never_touched(tmp_path):
    transport = FakeTransport()
    client = make_client(tmp_path, transport, mode="offline")
    with pytest.raises(OfflineCacheMiss):
        client.get("/studies", {"page": 0, "pageSize": 10})
    assert transport.calls == [] and client.request_log == []


def test_offline_cache_hit_is_served_with_a_cache_request_record(tmp_path):
    client = make_client(tmp_path, FakeTransport(), mode="offline")
    identity = CacheIdentity(base_url=BASE, endpoint="/studies", params={"page": 0, "pageSize": 1000})
    client.cache.put(identity, studies_page(1, 2, total_count=2, page_size=1000), http_status=200,
                     fetched_at_utc=T0 - timedelta(days=2), source_url_redacted=f"{BASE}/studies?page=0&pageSize=1000")
    result = client.studies()
    assert result.complete and result.returned_count == 2 and result.status == "complete"
    record = client.request_log[0]
    assert record.origin == "cache" and record.fetched_at_utc == T0 - timedelta(days=2)
    assert record.request_id == result.request_ids[0] == "req_run_t_0001"


CATALOG = [{"studyDbId": "S1", "studyName": "19ayt A", "locationDbId": "L1", "seasons": ["2019"], "studyType": "Advanced Yield Trial", "programDbId": "P1"},
           {"studyDbId": "S2", "studyName": "19pyt B", "locationDbId": "L1", "seasons": ["2019"], "studyType": "Preliminary Yield Trial", "programDbId": "P1"},
           {"studyDbId": "S3", "studyName": "20ayt C", "locationDbId": "L1", "seasons": ["2020"], "studyType": "advanced yield trial", "programDbId": "P2"},
           {"studyDbId": "S4", "studyName": "19ayt D", "locationDbId": "L2", "seasons": ["2019", "2020"], "studyType": "Advanced Yield Trial", "programDbId": "P1"}]


def _seed_catalog(client, body: bytes) -> None:
    identity = CacheIdentity(base_url=BASE, endpoint="/studies", params={"page": 0, "pageSize": 1000})
    client.cache.put(identity, body, http_status=200, fetched_at_utc=T0 - timedelta(days=2), source_url_redacted=f"{BASE}/studies?page=0&pageSize=1000")


def test_offline_filtered_study_request_is_answered_from_the_cached_complete_catalog(tmp_path):
    """Real data, 2026-09-29 (Q1-Q3): a snapshot holds the complete catalog, never the server-filtered request, so offline
    'studies at L1 in 2019 of type AYT' missed the cache and the run failed. The exact filters are now applied here to the cached
    catalog, with the server's meaning, and the subset is complete only because the catalog is."""
    transport = FakeTransport()
    client = make_client(tmp_path, transport, mode="offline")
    _seed_catalog(client, page_body(CATALOG, total_count=4, page_size=1000))
    result = client.studies(StudyFilters(location_id="L1", season_id="2019", study_type="Advanced Yield Trial"))
    assert [r["studyDbId"] for r in result.records] == ["S1"] and result.complete and result.status == "complete" and result.reported_total == 1
    assert any("applied client-side" in w and "complete" in w for w in result.warnings) and transport.calls == []
    assert client.studies(StudyFilters(location_id="L1", season_id="2019")).returned_count == 2                 # S1, S2
    assert [r["studyDbId"] for r in client.studies(StudyFilters(season_id="2020")).records] == ["S3", "S4"]    # a season among several
    assert [r["studyDbId"] for r in client.studies(StudyFilters(study_type="ADVANCED YIELD TRIAL")).records] == ["S1", "S3", "S4"]   # type ignores case
    assert [r["studyDbId"] for r in client.studies(StudyFilters(program_id="P2")).records] == ["S3"]
    empty = client.studies(StudyFilters(location_id="L9"))
    assert empty.status == "empty" and empty.complete and empty.returned_count == 0
    named = client.studies(StudyFilters(location_id="L1", name_contains="19ayt"))                                # exact filters, then name_contains
    assert [r["studyDbId"] for r in named.records] == ["S1"] and named.complete
    # the cache still has no filtered key: nothing was invented into it
    assert client.cache.get(CacheIdentity(base_url=BASE, endpoint="/studies", params={"locationDbId": "L1", "page": 0, "pageSize": 1000})) is None


def test_offline_filtered_subset_of_an_incomplete_catalog_is_not_called_complete(tmp_path):
    client = make_client(tmp_path, FakeTransport(), mode="offline")
    _seed_catalog(client, page_body(CATALOG, total_count=10, page_size=1000, total_pages=3))                     # 4 of 10 cached
    result = client.studies(StudyFilters(location_id="L1"))
    assert not result.complete and result.status == "incomplete" and result.reported_total is None and result.returned_count == 3
    assert any("cannot be called complete" in w for w in result.warnings)
    # with no catalog cached at all the filtered request still fails honestly, and no filter magic applies to a live client
    bare = make_client(tmp_path / "bare", FakeTransport(), mode="offline")
    failed = bare.studies(StudyFilters(location_id="L1"))
    assert failed.status == "failed" and any("neither is the study catalog" in w for w in failed.warnings)
    transport = FakeTransport().add("/studies", {"locationDbId": "L1", "page": 0, "pageSize": 1000}, ok(page_body(CATALOG[:2], total_count=2, page_size=1000)))
    live = make_client(tmp_path / "live", transport)
    live_result = live.studies(StudyFilters(location_id="L1"), approval=approval())
    assert live_result.returned_count == 2 and len(transport.calls) == 1 and transport.calls[0].params["locationDbId"] == "L1"


# --- 3. live: approval before every attempt --------------------------------------------

def test_live_get_without_approval_is_refused_before_any_http(tmp_path):
    transport = FakeTransport().add("/studies", {}, ok(studies_page(1, 1, total_count=1)))
    client = make_client(tmp_path, transport)
    with pytest.raises(NotAuthorized):
        client.get("/studies")
    assert transport.calls == []


@pytest.mark.parametrize("bad", [
    dict(base="https://test-server.brapi.org/brapi/v2"),   # other server
    dict(expires_hours=-1),                                 # expired
    dict(families=("locations",)),                          # family not covered
])
def test_live_get_rejects_wrong_approvals(tmp_path, bad):
    transport = FakeTransport().add("/studies", {}, ok(studies_page(1, 1, total_count=1)))
    client = make_client(tmp_path, transport)
    if "expires_hours" in bad:
        appr = FetchApproval(**{**approval().model_dump(), "issued_at_utc": T0 - timedelta(hours=3),
                                "expires_at_utc": T0 - timedelta(hours=1)})
    else:
        appr = approval(**bad)
    with pytest.raises(NotAuthorized):
        client.get("/studies", approval=appr)
    assert transport.calls == []


def test_live_success_is_cached_and_logged_then_served_from_cache(tmp_path):
    transport = FakeTransport().add("/studies", {"page": 0, "pageSize": 1000}, ok(studies_page(1, 3, total_count=3, page_size=1000)))
    client = make_client(tmp_path, transport)
    first = client.studies(approval=approval())
    second = client.studies(approval=approval())
    assert first.records == second.records and first.complete and second.complete
    assert len(transport.calls) == 1                                  # second call never hit the server
    assert [r.origin for r in client.request_log] == ["live", "cache"]
    live = client.request_log[0]
    assert live.http_status == 200 and live.response_sha256 and live.attempt == 1
    assert live.source_url_redacted == f"{BASE}/studies?page=0&pageSize=1000"
    assert transport.calls[0].read_timeout <= client.settings.read_timeout


def test_attempt_budget_counts_failed_attempts_and_blocks_the_next_request(tmp_path):
    transport = (FakeTransport()
                 .add("/studies", {"page": 0, "pageSize": 1000}, TransportError("timeout", "slow", retryable=True),
                      TransportError("timeout", "slow", retryable=True))
                 .add("/locations", {"page": 0, "pageSize": 1000}, ok(page_body([{"locationDbId": "L1"}], total_count=1))))
    client = make_client(tmp_path, transport, max_attempts=2)
    appr = approval(max_attempts=2)
    failed = client.studies(approval=appr)
    assert failed.status == "failed" and not failed.complete and len(failed.request_ids) == 2
    assert [r.error.code for r in client.request_log] == ["timeout", "timeout"]
    with pytest.raises(NotAuthorized):                       # budget of 2 is spent, even though both failed
        client.get("/locations", {"page": 0, "pageSize": 1000}, approval=appr)
    assert len(transport.calls) == 2


# --- 4. the allowlist: no model-supplied URLs or paths -----------------------------------

@pytest.mark.parametrize("endpoint", [
    "https://evil.example/brapi/v2/studies", "/admin", "/studies/../secret", "/studies//S1",
    "/studies?studyDbId=1", "studies", "/germplasm", "/studies/S1/extra",
])
def test_endpoints_off_the_allowlist_are_refused(tmp_path, endpoint):
    client = make_client(tmp_path, FakeTransport())
    with pytest.raises(BrapiClientError) as info:
        client.get(endpoint, approval=approval())
    assert info.value.code == "invalid_argument"


@pytest.mark.parametrize("study_id", ["..", "S1/../S2", "S 1", "", "S1?x=1"])
def test_ids_with_slashes_dots_or_spaces_are_refused(tmp_path, study_id):
    client = make_client(tmp_path, FakeTransport())
    with pytest.raises(BrapiClientError):
        client.study(study_id, approval=approval())


# --- 5. transport outcomes ------------------------------------------------------------------

def test_redirects_are_refused_not_followed(tmp_path):
    transport = FakeTransport().add("/studies", {"page": 0, "pageSize": 1000},
                                    TransportResponse(302, {"Location": "https://elsewhere.example/x"}, b""))
    client = make_client(tmp_path, transport)
    result = client.studies(approval=approval())
    assert result.status == "failed" and len(transport.calls) == 1
    assert "redirect" in client.request_log[0].error.message


@pytest.mark.parametrize("body", [b"<html>not json</html>", b'{"result": {"data": []}}', b'{"metadata": {}, "result": [1, 2]}'])
def test_invalid_json_or_non_brapi_replies_are_errors_not_retried_and_not_cached(tmp_path, body):
    transport = FakeTransport().add("/studies", {"page": 0, "pageSize": 1000}, ok(body))
    client = make_client(tmp_path, transport)
    result = client.studies(approval=approval())
    assert result.status == "failed" and len(transport.calls) == 1
    assert client.request_log[0].error.code in ("invalid_json", "not_brapi")
    assert client.cache.get(CacheIdentity(base_url=BASE, endpoint="/studies", params={"page": 0, "pageSize": 1000})) is None


def test_oversized_reply_is_an_error(tmp_path):
    transport = FakeTransport().add("/studies", {"page": 0, "pageSize": 1000}, TransportError("too_large", "cap exceeded"))
    client = make_client(tmp_path, transport, max_attempts=3)
    result = client.studies(approval=approval())
    assert result.status == "failed" and len(transport.calls) == 1 and client.request_log[0].error.code == "too_large"


def test_metadata_timeouts_are_retried_within_bounds_and_each_attempt_is_logged(tmp_path):
    sleeps = []
    transport = (FakeTransport()
                 .add("/locations", {"page": 0, "pageSize": 1000},
                      TransportError("timeout", "slow", retryable=True),
                      TransportError("connection", "reset", retryable=True),
                      ok(page_body([{"locationDbId": "L1", "locationName": "Ibadan"}], total_count=1))))
    client = make_client(tmp_path, transport, sleeps=sleeps, max_attempts=3)
    result = client.locations(approval=approval())
    assert result.complete and result.returned_count == 1
    assert [r.attempt for r in client.request_log] == [1, 2, 3]
    assert [r.error.code if r.error else "ok" for r in client.request_log] == ["timeout", "connection", "ok"]
    assert result.request_ids == [client.request_log[2].request_id]       # the successful attempt
    assert sleeps == [0.5, 1.0]                                          # bounded back-off, no real sleeping


def test_observation_timeout_is_never_retried(tmp_path):
    transport = FakeTransport().add("/observations", {"studyDbId": "S1", "observationVariableDbId": "V1", "page": 0, "pageSize": 1000},
                                    TransportError("timeout", "slow", retryable=True), ok(page_body([], total_count=0)))
    client = make_client(tmp_path, transport, max_attempts=3)
    result = client.observations("S1", "V1", approval=approval(studies=["S1"], variables=["V1"]))
    assert result.status == "failed" and len(transport.calls) == 1


def test_rate_limit_honours_retry_after_within_budget(tmp_path):
    sleeps = []
    transport = (FakeTransport()
                 .add("/studies", {"page": 0, "pageSize": 1000},
                      TransportResponse(429, {"Retry-After": "2"}, b""),
                      ok(studies_page(1, 1, total_count=1))))
    client = make_client(tmp_path, transport, sleeps=sleeps)
    result = client.studies(approval=approval())
    assert result.complete and sleeps == [2.0] and len(transport.calls) == 2
    assert client.request_log[0].http_status == 429 and client.request_log[0].error.retryable


def test_rate_limit_beyond_the_remaining_budget_stops(tmp_path):
    sleeps = []
    transport = FakeTransport().add("/studies", {"page": 0, "pageSize": 1000},
                                    TransportResponse(429, {"Retry-After": "600"}, b""), ok(studies_page(1, 1, total_count=1)))
    client = make_client(tmp_path, transport, sleeps=sleeps, operation_deadline=60.0)
    result = client.studies(approval=approval())
    assert result.status == "failed" and sleeps == [] and len(transport.calls) == 1
    assert "exceeds remaining budget" in result.warnings[-1]


def test_stalled_reads_are_cut_by_the_read_timeout_and_the_operation_deadline(tmp_path):
    clock = FakeClock()

    def stalled_read(req):
        clock.advance(req.read_timeout)         # the server goes silent; the read times out after read_timeout
        raise TransportError("timeout", "read stalled", retryable=True)

    transport = FakeTransport().add("/studies", {"page": 0, "pageSize": 1000}, stalled_read, stalled_read, stalled_read)
    client = make_client(tmp_path, transport, clock=clock, operation_deadline=60.0, read_timeout=30.0, max_attempts=3)
    result = client.studies(approval=approval())
    assert result.status == "failed"
    assert len(transport.calls) == 2                                   # 30 s + 30 s = the whole 60 s budget; no third try
    assert [c.read_timeout for c in transport.calls] == [30.0, 30.0]
    assert transport.calls[0].deadline_remaining == 60.0 and transport.calls[1].deadline_remaining == 30.0
    assert [r.duration_seconds for r in client.request_log] == [30.0, 30.0]   # each stalled read, measured
    assert all(r.error.code == "timeout" for r in client.request_log)
    assert any("deadline" in w for w in result.warnings)
    assert clock.now == 1060.0                                         # exactly the budget, never more


def test_read_timeout_never_exceeds_the_remaining_budget(tmp_path):
    clock = FakeClock()
    transport = FakeTransport().add("/studies", {"page": 0, "pageSize": 1000}, ok(studies_page(1, 1, total_count=1)))
    client = make_client(tmp_path, transport, clock=clock, operation_deadline=5.0, read_timeout=30.0)
    client.studies(approval=approval())
    assert transport.calls[0].read_timeout == 5.0


# --- 6. pagination and completeness ------------------------------------------------------------

def test_2500_records_served_1000_at_a_time_are_complete(tmp_path):
    transport = FakeTransport()
    for page in range(3):
        count = 1000 if page < 2 else 500
        transport.add("/studies", {"page": page, "pageSize": 1000},
                      ok(studies_page(page * 1000 + 1, count, total_count=2500, page=page, page_size=1000, total_pages=3)))
    client = make_client(tmp_path, transport)
    result = client.studies(approval=approval(), max_pages=5)
    assert result.status == "complete" and result.complete
    assert result.returned_count == 2500 and result.reported_total == 2500 and len(result.request_ids) == 3
    assert len(transport.calls) == 3 and result.records[-1]["studyDbId"] == "S2500"


def test_max_pages_exhaustion_is_incomplete_not_complete(tmp_path):
    transport = FakeTransport()
    for page in range(2):
        transport.add("/studies", {"page": page, "pageSize": 1000},
                      ok(studies_page(page * 1000 + 1, 1000, total_count=2500, page=page, page_size=1000, total_pages=3)))
    client = make_client(tmp_path, transport)
    result = client.studies(approval=approval(), max_pages=2)
    assert result.status == "incomplete" and not result.complete
    assert result.returned_count == 2000 and result.reported_total == 2500
    assert any("max_pages=2" in w for w in result.warnings)


def test_default_is_one_page_and_more_pages_means_incomplete(tmp_path):
    transport = FakeTransport().add("/studies", {"page": 0, "pageSize": 1000},
                                    ok(studies_page(1, 1000, total_count=1500, page=0, page_size=1000)))
    client = make_client(tmp_path, transport)
    result = client.studies(approval=approval())
    assert result.status == "incomplete" and result.returned_count == 1000 and len(transport.calls) == 1


def test_null_result_is_empty_only_when_the_metadata_says_zero(tmp_path):
    transport = (FakeTransport()
                 .add("/locations", {"page": 0, "pageSize": 1000}, ok(page_body([], total_count=0, result_null=True)))
                 .add("/programs", {"page": 0, "pageSize": 1000}, ok(page_body([], total_count=5, result_null=True))))
    client = make_client(tmp_path, transport)
    empty = client.locations(approval=approval())
    assert empty.status == "empty" and empty.complete and empty.returned_count == 0
    odd = client.programs(approval=approval())
    assert odd.status == "unknown" and not odd.complete and odd.reported_total == 5
    assert any("result is null" in w for w in odd.warnings)


def test_missing_pagination_cannot_prove_completeness(tmp_path):
    transport = FakeTransport().add("/seasons", {"page": 0, "pageSize": 1000},
                                    ok(page_body([{"seasonDbId": "2019"}], pagination=False)))
    client = make_client(tmp_path, transport)
    result = client.seasons(approval=approval())
    assert result.status == "unknown" and not result.complete and result.returned_count == 1
    assert any("no totalCount" in w for w in result.warnings)


def test_repeated_page_is_detected_and_stops(tmp_path):
    same = studies_page(1, 1000, total_count=2000, page=0, page_size=1000, total_pages=2)
    transport = (FakeTransport()
                 .add("/studies", {"page": 0, "pageSize": 1000}, ok(same))
                 .add("/studies", {"page": 1, "pageSize": 1000}, ok(same)))   # server ignores the page number
    client = make_client(tmp_path, transport)
    result = client.studies(approval=approval(), max_pages=3)
    assert not result.complete and result.returned_count == 1000
    assert any("repeats page" in w for w in result.warnings)


def test_count_change_during_paging_marks_uncertainty(tmp_path):
    transport = (FakeTransport()
                 .add("/studies", {"page": 0, "pageSize": 1000}, ok(studies_page(1, 1000, total_count=1500, page=0, page_size=1000, total_pages=2)))
                 .add("/studies", {"page": 1, "pageSize": 1000}, ok(studies_page(1001, 501, total_count=1501, page=1, page_size=1000, total_pages=2))))
    client = make_client(tmp_path, transport)
    result = client.studies(approval=approval(), max_pages=3)
    assert result.status == "unknown" and not result.complete and result.returned_count == 1501
    assert any("not an atomic snapshot" in w for w in result.warnings)


def test_duplicates_across_pages_are_kept_with_a_warning(tmp_path):
    first = studies_page(1, 1000, total_count=1001, page=0, page_size=1000, total_pages=2)
    second = studies_page(1000, 1, total_count=1001, page=1, page_size=1000, total_pages=2)   # S1000 again
    transport = FakeTransport().add("/studies", {"page": 0, "pageSize": 1000}, ok(first)).add("/studies", {"page": 1, "pageSize": 1000}, ok(second))
    client = make_client(tmp_path, transport)
    result = client.studies(approval=approval(), max_pages=3)
    assert result.returned_count == 1001                       # nothing silently deleted
    assert not result.complete and any("duplicate" in w for w in result.warnings)


def test_page_and_page_size_are_reserved(tmp_path):
    client = make_client(tmp_path, FakeTransport())
    with pytest.raises(BrapiClientError):
        client.get_all("/studies", {"pageSize": 5}, approval=approval())


def test_failed_page_returns_what_was_fetched_as_failed(tmp_path):
    transport = (FakeTransport()
                 .add("/studies", {"page": 0, "pageSize": 1000}, ok(studies_page(1, 1000, total_count=2000, page=0, page_size=1000, total_pages=2)))
                 .add("/studies", {"page": 1, "pageSize": 1000}, TransportResponse(500, {}, b""), TransportResponse(500, {}, b"")))
    client = make_client(tmp_path, transport, max_attempts=2)
    result = client.studies(approval=approval(), max_pages=3)
    assert result.status == "failed" and result.returned_count == 1000 and not result.complete
    assert len(result.request_ids) == 3                       # 1 good page + 2 failed attempts, all traceable


# --- 7. observations: one authorized page, then stop ---------------------------------------------

def test_observations_first_tries_one_page_and_stops_when_more_exist(tmp_path):
    params = {"studyDbId": "S1", "observationVariableDbId": "V1"}
    transport = FakeTransport().add("/observations", {**params, "page": 0, "pageSize": 1000},
                                    ok(page_body([{"observationDbId": f"O{i}", "value": "10"} for i in range(1000)],
                                                 total_count=1400, page=0, page_size=1000)))
    client = make_client(tmp_path, transport)
    result = client.observations("S1", "V1", approval=approval(studies=["S1"], variables=["V1"]))
    assert result.status == "incomplete" and result.returned_count == 1000 and len(transport.calls) == 1


def test_observations_paging_mode_is_bounded_and_separate(tmp_path):
    params = {"studyDbId": "S1", "observationVariableDbId": "V1"}
    transport = FakeTransport()
    for page, count in ((0, 1000), (1, 400)):
        transport.add("/observations", {**params, "page": page, "pageSize": 1000},
                      ok(page_body([{"observationDbId": f"O{page * 1000 + i}", "value": "10"} for i in range(count)],
                                   total_count=1400, page=page, page_size=1000, total_pages=2)))
    client = make_client(tmp_path, transport)
    result = client.observations("S1", "V1", approval=approval(studies=["S1"], variables=["V1"]), paging=True, max_pages=5)
    assert result.complete and result.returned_count == 1400 and len(transport.calls) == 2


def test_observations_need_the_exact_study_and_variable_in_the_approval(tmp_path):
    transport = FakeTransport().add("/observations", {"studyDbId": "S2", "observationVariableDbId": "V1", "page": 0, "pageSize": 1000},
                                    ok(page_body([], total_count=0)))
    client = make_client(tmp_path, transport)
    appr = approval(studies=["S1"], variables=["V1"])
    blocked = client.observations("S2", "V1", approval=appr)          # S2 is not approved
    assert blocked.status == "failed" and transport.calls == [] and "not_authorized" in blocked.warnings[0]
    blocked_var = client.observations("S1", "V9", approval=appr)      # V9 is not approved
    assert blocked_var.status == "failed" and transport.calls == []


def test_tenth_study_is_blocked_by_the_distinct_study_limit(tmp_path):
    transport = FakeTransport()
    ids = [f"S{i}" for i in range(1, 11)]
    for sid in ids:
        transport.add("/observations", {"studyDbId": sid, "observationVariableDbId": "V1", "page": 0, "pageSize": 1000},
                      ok(page_body([{"observationDbId": f"O{sid}", "value": "1"}], total_count=1)))
    client = make_client(tmp_path, transport, max_attempts=1)
    appr = approval(studies=ids[:9], variables=["V1"], max_studies=9, max_attempts=50)
    for sid in ids[:9]:
        assert client.observations(sid, "V1", approval=appr).complete
    tenth = client.observations("S10", "V1", approval=appr)
    assert tenth.status == "failed" and len(transport.calls) == 9
    assert "not in the approval" in tenth.warnings[0]


# --- 7b. a saved reply is served only within the approval (item 20, Bidhan 2026-10-23) -----------------------------

ONE_PAGE = {"page": 0, "pageSize": 1000}


def _seed(client, endpoint: str, params: dict, body: bytes) -> None:
    identity = CacheIdentity(base_url=BASE, endpoint=endpoint, params=params)
    client.cache.put(identity, body, http_status=200, fetched_at_utc=T0 - timedelta(days=2), source_url_redacted=f"{BASE}{endpoint}")


def test_a_saved_reply_outside_the_approval_is_refused_like_a_live_request(tmp_path):
    """Item (20): live, with an approval, a saved reply is checked against it as a live request is (study, variable, family,
    server, time); before, a cache hit was returned before _authorize ran. Nothing is sent either way."""
    transport = FakeTransport()
    client = make_client(tmp_path, transport)
    for study, variable in (("S1", "V1"), ("S2", "V1"), ("S1", "V9")):
        _seed(client, "/observations", {"studyDbId": study, "observationVariableDbId": variable, **ONE_PAGE},
              page_body([{"observationDbId": f"O{study}{variable}", "value": "1"}], total_count=1))
    appr = approval(studies=["S1"], variables=["V1"])
    assert client.observations("S1", "V1", approval=appr).complete                 # within the approval: served as before
    for study, variable in (("S2", "V1"), ("S1", "V9")):                            # the study, then the variable, not approved
        refused = client.observations(study, variable, approval=appr)
        assert refused.status == "failed" and "not_authorized" in refused.warnings[0]
    _seed(client, "/locations", ONE_PAGE, page_body([{"locationDbId": "L1"}], total_count=1))
    with pytest.raises(NotAuthorized, match="family"):
        client.get("/locations", ONE_PAGE, approval=approval(families=("studies",)))
    _seed(client, "/studies", ONE_PAGE, studies_page(1, 1, total_count=1, page_size=1000))
    with pytest.raises(NotAuthorized, match="different server"):
        client.get("/studies", ONE_PAGE, approval=approval(base="https://test-server.brapi.org/brapi/v2"))
    expired = FetchApproval(**{**approval().model_dump(), "issued_at_utc": T0 - timedelta(hours=3),
                               "expires_at_utc": T0 - timedelta(hours=1)})
    with pytest.raises(NotAuthorized, match="expired"):
        client.get("/studies", ONE_PAGE, approval=expired)
    assert transport.calls == []
    assert [r.origin for r in client.request_log] == ["cache"]                      # a refused saved reply is not logged as used
    assert client._counter == 1                                                     # nor does it use up a request number (review)


def test_a_saved_reply_within_the_approval_is_served_after_the_attempt_budget_is_used_up(tmp_path):
    """Item (20): the HTTP attempt budget counts live attempts; a saved reply sends nothing, so a used-up budget does not stop
    it (hit=True). The distinct-study limit counts live fetches too (next test)."""
    transport = FakeTransport().add("/observations", {"studyDbId": "S1", "observationVariableDbId": "V1", **ONE_PAGE},
                                    ok(page_body([{"observationDbId": "O1", "value": "1"}], total_count=1)))
    client = make_client(tmp_path, transport, max_attempts=1)
    _seed(client, "/observations", {"studyDbId": "S2", "observationVariableDbId": "V1", **ONE_PAGE},
          page_body([{"observationDbId": "O2", "value": "2"}], total_count=1))
    appr = approval(studies=["S1", "S2"], variables=["V1"], max_attempts=1)
    assert client.observations("S1", "V1", approval=appr).complete                 # the one live attempt
    saved = client.observations("S2", "V1", approval=appr)                          # the budget is used up; S2 is saved
    assert saved.complete and len(transport.calls) == 1
    assert [r.origin for r in client.request_log] == ["live", "cache"]


def test_a_saved_reply_of_an_approved_study_is_served_when_another_approval_filled_the_study_limit(tmp_path):
    """Review of item (20): the distinct-study limit counts the client's live fetches, whoever approved them (_studies_fetched is
    per client, not per approval): a saved reply of a study the CURRENT approval lists is still served (hit=True)."""
    transport = FakeTransport().add("/observations", {"studyDbId": "S1", "observationVariableDbId": "V1", **ONE_PAGE},
                                    ok(page_body([{"observationDbId": "O1", "value": "1"}], total_count=1)))
    client = make_client(tmp_path, transport)
    _seed(client, "/observations", {"studyDbId": "S2", "observationVariableDbId": "V1", **ONE_PAGE},
          page_body([{"observationDbId": "O2", "value": "2"}], total_count=1))
    first = approval(studies=["S1"], variables=["V1"], max_studies=1, max_attempts=50)
    second = approval(studies=["S2"], variables=["V1"], max_studies=1, max_attempts=50)
    assert client.observations("S1", "V1", approval=first).complete                 # the one study of the limit is now used
    assert client.observations("S2", "V1", approval=second).complete                # saved, in the second approval: served
    assert len(transport.calls) == 1


def test_offline_and_approval_less_saved_replies_are_served_as_before(tmp_path):
    """Item (20) changes only a live read WITH an approval: offline (with any approval or none) and live without one, a saved
    reply is served as before (the catalogs, the front door's summary and the eval read so)."""
    for n, (mode, appr) in enumerate((("offline", None), ("offline", approval(families=("locations",))), ("live", None))):
        client = make_client(tmp_path / str(n), FakeTransport(), mode=mode)
        _seed(client, "/studies", ONE_PAGE, studies_page(1, 2, total_count=2, page_size=1000))
        assert client.studies(approval=appr).complete, (mode, appr)


# --- 8. named endpoints and typed returns --------------------------------------------------------

def test_studies_filters_map_to_server_params_and_name_contains_is_client_side(tmp_path):
    transport = FakeTransport().add("/studies", {"locationDbId": "L1", "seasonDbId": "2019", "studyType": "Advanced Yield Trial",
                                                 "page": 0, "pageSize": 1000},
                                    ok(page_body([{"studyDbId": "S1", "studyName": "19ayt18highYldIB"},
                                                  {"studyDbId": "S2", "studyName": "19pyt20lowIB"}], total_count=2)))
    client = make_client(tmp_path, transport)
    filters = StudyFilters(location_id="L1", season_id="2019", study_type="Advanced Yield Trial", name_contains="AYT")
    result = client.studies(filters, approval=approval())
    assert [r["studyDbId"] for r in result.records] == ["S1"] and result.complete and result.reported_total == 1
    with pytest.raises(BrapiClientError):
        client.studies(StudyFilters(location_name="Ibadan"), approval=approval())


def test_study_returns_one_record_and_its_receipt(tmp_path):
    body = json.dumps({"metadata": {}, "result": {"studyDbId": "S1", "studyName": "synthetic one"}}).encode()
    transport = FakeTransport().add("/studies/S1", {}, ok(body))
    client = make_client(tmp_path, transport)
    record, receipt = client.study("S1", approval=approval())
    assert record["studyName"] == "synthetic one" and receipt.endpoint == "/studies/S1" and receipt.origin == "live"


def test_find_variables_is_literal_case_insensitive_discovery(tmp_path):
    transport = FakeTransport().add("/variables", {"page": 0, "pageSize": 1000},
                                    ok(page_body([{"observationVariableDbId": "V1", "observationVariableName": "Fresh root yield"},
                                                  {"observationVariableDbId": "V2", "observationVariableName": "Dry Matter content"},
                                                  {"observationVariableDbId": "V3", "observationVariableName": "dry-matter (NIRS)"}],
                                                 total_count=3)))
    client = make_client(tmp_path, transport)
    found = client.find_variables("dry matter", approval=approval())
    assert [r["observationVariableDbId"] for r in found.records] == ["V2"]    # literal: the hyphen does not match
    assert found.complete and found.reported_total == 1
    none = client.find_variables("plant height", approval=approval())
    assert none.status == "empty" and none.complete


def test_records_to_df_keeps_every_value_as_text(tmp_path):
    transport = FakeTransport().add("/observations", {"studyDbId": "S1", "observationVariableDbId": "V1", "page": 0, "pageSize": 1000},
                                    ok(page_body([{"observationDbId": "007", "value": "10", "extra": {"a": 1}, "flag": True, "gone": None},
                                                  {"observationDbId": "008", "value": "", "extra": [1, 2], "flag": False, "gone": "x"}],
                                                 total_count=2)))
    client = make_client(tmp_path, transport)
    result = client.observations("S1", "V1", approval=approval(studies=["S1"], variables=["V1"]))
    frame = records_to_df(result)
    assert list(frame["observationDbId"]) == ["007", "008"]               # leading zero preserved
    assert list(frame["value"]) == ["10", ""]                             # empty stays empty, not NaN
    assert list(frame["extra"]) == ['{"a": 1}', "[1, 2]"] and list(frame["flag"]) == ["true", "false"]
    assert frame["gone"][0] is None and frame["gone"][1] == "x"
    assert all(frame[col].dtype == object for col in frame.columns)
    subset = records_to_df(result, columns=["observationDbId", "value"])
    assert list(subset.columns) == ["observationDbId", "value"]


def test_provenance_lists_every_attempt_and_cache_use_in_order(tmp_path):
    transport = (FakeTransport()
                 .add("/studies", {"page": 0, "pageSize": 1000}, TransportError("timeout", "slow", retryable=True),
                      ok(studies_page(1, 1, total_count=1))))
    client = make_client(tmp_path, transport)
    client.studies(approval=approval())
    client.studies(approval=approval())
    log = client.provenance()
    assert [(r.attempt, r.origin, r.error.code if r.error else "ok") for r in log] == [(1, "live", "timeout"), (2, "live", "ok"), (1, "cache", "ok")]
    assert all(r.run_id == "run_t" for r in log) and [r.request_id for r in log] == ["req_run_t_0001", "req_run_t_0002", "req_run_t_0003"]
    assert not any("token" in r.source_url_redacted for r in log)


def test_no_real_http_happened_in_this_module():
    """The conftest guard would have raised on any real socket use; this documents the intent."""
    assert bc.RequestsTransport is not None   # exists, but was never instantiated by these tests
