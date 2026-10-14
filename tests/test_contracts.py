"""
tests/test_contracts.py — every contract accepts good synthetic records and
REJECTS bad ones (it never "fixes" them quietly).

All data here is invented (SYNTHETIC). No network, no model, no files.

Reading tip: `accepts` means building the record works; `rejects` means
pydantic raises ValidationError. Each rejected case names the rule it hits.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

import contracts as c

T0 = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)
SHA_A = "a" * 64
SHA_B = "b" * 64


def rejects(record_type, **fields):
    """Assert that building this record fails, and hand back the error text for checking."""
    with pytest.raises(ValidationError) as info:
        record_type(**fields)
    return str(info.value)


# --- builders for valid synthetic records (each test tweaks one thing) -----------

def good_request(**over):
    base = dict(
        request_id="req_0001", run_id="run_0001",
        base_url="https://cassavabase.org/brapi/v2", endpoint="/studies",
        params={"pageSize": 5, "page": 0},
        requested_at_utc=T0, fetched_at_utc=T0 + timedelta(seconds=1),
        origin="live", http_status=200, attempt=1, duration_seconds=0.8,
        response_sha256=SHA_A,
        source_url_redacted="https://cassavabase.org/brapi/v2/studies?pageSize=5&page=0",
    )
    base.update(over)
    return base


def good_collection(**over):
    base = dict(records=[{"studyDbId": "s1"}, {"studyDbId": "s2"}], returned_count=2,
                reported_total=2, complete=True, status="complete", request_ids=["req_0001"])
    base.update(over)
    return base


def good_manifest(**over):
    base = dict(artifact_id="art_0001", run_id="run_0001", kind="observations",
                relative_path="out/run_0001/observations_S1_V1.csv", sha256=SHA_A,
                schema_version="1", row_count=6, complete=True,
                source_request_ids=["req_0001"], study_ids=["S1"], variable_ids=["V1"],
                measurement=c.MeasurementMeta(trait="fresh root yield", unit="t/ha", timepoint="harvest"),
                columns={"value": "raw observation value as text"}, created_at_utc=T0)
    base.update(over)
    return base


def good_approval(**over):
    base = dict(approval_id="appr_0001", run_id="run_0001",
                base_url="https://cassavabase.org/brapi/v2",
                endpoint_families=["studies", "observations"],
                observation_study_ids=["S1"], observation_variable_ids=["V1"],
                max_http_attempts=10, max_observation_studies=1,
                issued_at_utc=T0, expires_at_utc=T0 + timedelta(hours=1))
    base.update(over)
    return base


def good_claim(**over):
    base = dict(claim_id="clm_0001", kind="mean", value=18.0, unit="t/ha",
                denominator=c.Denominator(name="valid plot-level values", n=4),
                n_independent_units=4, source_artifact_hashes=[SHA_A],
                filters={"study": "S1", "variable": "V1"}, grouping=[],
                transformation="pooled mean of valid values; missing and invalid excluded",
                exclusions=[c.Exclusion(reason="raw value empty", count=1),
                            c.Exclusion(reason="raw value not numeric", count=1)])
    base.update(over)
    return base


def good_agent(**over):
    base = dict(status="completed", payload={"answer": "S1 resolved"},
                messages=[c.NormalizedMessage(role="user", content="find S1"),
                          c.NormalizedMessage(role="assistant", content="done", correlation_id="m1")],
                model_requested="gpt-oss:20b", model_reported="gpt-oss:20b",
                model_calls=2, tool_calls=1,
                usage=c.Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                elapsed_seconds=3.5, log_path="logs/run_0001/retriever.jsonl")
    base.update(over)
    return base


# --- 0. rules shared by every record --------------------------------------------

ALL_SIMPLE_RECORDS = [c.StudyFilters, c.MeasurementMeta, c.Denominator, c.Exclusion]


def test_contract_version_is_fixed_to_1():
    assert c.StudyFilters().contract_version == "1"
    assert "Input should be '1'" in rejects(c.StudyFilters, contract_version="2")


def test_extra_fields_are_rejected_even_authority_looking_ones():
    assert "Extra inputs are not permitted" in rejects(c.StudyFilters, approved=True)
    assert "Extra inputs are not permitted" in rejects(c.ToolResult, ok=True, fetch_approval={"x": 1})


def test_ids_must_be_strings_not_integers():
    assert "valid string" in rejects(c.StudyFilters, location_id=12)
    assert "valid string" in rejects(c.RequestRecord, **good_request(request_id=1))


def test_records_are_frozen():
    filters = c.StudyFilters(location_id="loc_1")
    with pytest.raises(ValidationError):
        filters.location_id = "loc_2"


def test_strict_mode_does_not_convert_types():
    assert "valid integer" in rejects(c.Denominator, name="plots", n="4")     # "4" is not 4
    assert "valid integer" in rejects(c.Denominator, name="plots", n=True)    # True is not 1
    assert "valid boolean" in rejects(c.CollectionResult, **good_collection(complete="yes"))


def test_negative_counts_are_rejected():
    assert "greater than or equal to 0" in rejects(c.Denominator, name="plots", n=-1)
    assert "greater than or equal to 0" in rejects(c.Exclusion, reason="x", count=-3)
    assert "greater than or equal to 0" in rejects(c.CollectionResult, **good_collection(returned_count=-2))


# --- 1. StudyFilters --------------------------------------------------------------

def test_study_filters_exact_and_discovery():
    exact = c.StudyFilters(location_id="loc_23", season_id="2019", study_type="Advanced Yield Trial")
    assert exact.is_exact
    discovery = c.StudyFilters(location_name="Ibadan", name_contains="ayt")
    assert not discovery.is_exact


def test_study_filters_reject_id_and_name_for_the_same_thing():
    text = rejects(c.StudyFilters, location_id="loc_23", location_name="Ibadan")
    assert "resolve the name to an ID first" in text
    assert "resolve the name" in rejects(c.StudyFilters, season_id="2019", season_name="2019 wet")


def test_study_filters_reject_blank_text():
    assert rejects(c.StudyFilters, name_contains="   ")


# --- 2. RequestRecord -------------------------------------------------------------

def test_request_record_accepts_live_success_and_cache_hit():
    live = c.RequestRecord(**good_request())
    assert live.ok and live.origin == "live"
    cached = c.RequestRecord(**good_request(request_id="req_0002", origin="cache", duration_seconds=0.0))
    assert cached.ok and cached.origin == "cache"


def test_request_record_failure_needs_structured_error():
    failed = c.RequestRecord(**good_request(http_status=404, response_sha256=None,
                                            error=c.RequestError(code="http_error", message="HTTP 404")))
    assert not failed.ok and failed.error.code == "http_error"
    assert "valid dictionary" in rejects(c.RequestRecord, **good_request(error="something broke"))


def test_request_record_success_must_have_hash_time_and_2xx():
    assert "response_sha256" in rejects(c.RequestRecord, **good_request(response_sha256=None))
    assert "fetched_at_utc" in rejects(c.RequestRecord, **good_request(fetched_at_utc=None))
    assert "2xx" in rejects(c.RequestRecord, **good_request(http_status=500))
    assert "cannot finish before" in rejects(c.RequestRecord, **good_request(fetched_at_utc=T0 - timedelta(seconds=5)))


def test_request_record_cache_hit_keeps_the_original_earlier_fetch_time():
    # A cache hit is asked for today but the server answered days ago: allowed for origin="cache" only.
    hit = c.RequestRecord(**good_request(origin="cache", fetched_at_utc=T0, requested_at_utc=T0 + timedelta(days=3)))
    assert hit.fetched_at_utc < hit.requested_at_utc


def test_request_record_times_must_be_utc():
    naive = datetime(2026, 9, 25, 12, 0, 0)
    assert "timezone-aware UTC" in rejects(c.RequestRecord, **good_request(requested_at_utc=naive))
    eastern = datetime(2026, 9, 25, 8, 0, 0, tzinfo=timezone(timedelta(hours=-4)))
    assert "timezone-aware UTC" in rejects(c.RequestRecord, **good_request(requested_at_utc=eastern))


@pytest.mark.parametrize("bad_url", [
    "cassavabase.org/brapi/v2",               # no scheme
    "https://user:pw@cassavabase.org/brapi/v2",  # credentials
    "https://cassavabase.org/brapi/v2/",     # trailing slash
    "https://cassavabase.org/brapi/v2?x=1",  # query string
    "HTTPS://CassavaBase.org/brapi/v2",      # not lower-case
    "ftp://cassavabase.org/brapi/v2",        # wrong scheme
])
def test_request_record_rejects_unnormalized_base_url(bad_url):
    assert rejects(c.RequestRecord, **good_request(base_url=bad_url))


@pytest.mark.parametrize("bad_endpoint", ["studies", "/studies?x=1", "/studies/../secret", "//studies"])
def test_request_record_rejects_bad_endpoint(bad_endpoint):
    assert rejects(c.RequestRecord, **good_request(endpoint=bad_endpoint))


def test_request_record_redacted_url_may_not_leak_secrets():
    assert "secret-looking" in rejects(c.RequestRecord, **good_request(
        source_url_redacted="https://cassavabase.org/brapi/v2/studies?token=abc123"))
    assert "credentials" in rejects(c.RequestRecord, **good_request(
        source_url_redacted="https://me:pw@cassavabase.org/brapi/v2/studies"))


def test_request_record_json_round_trip():
    original = c.RequestRecord(**good_request())
    text = c.dump_json(original)
    assert "NaN" not in text and "Infinity" not in text
    assert c.RequestRecord.model_validate_json(text) == original


# --- 3. CollectionResult ----------------------------------------------------------

def test_collection_complete_and_empty_are_the_only_trusted_states():
    complete = c.CollectionResult(**good_collection())
    assert complete.complete
    empty = c.CollectionResult(records=[], returned_count=0, reported_total=0, complete=True, status="empty")
    assert empty.complete
    incomplete = c.CollectionResult(records=[{"studyDbId": "s1"}], returned_count=1, reported_total=40,
                                    complete=False, status="incomplete", warnings=["39 more records not fetched"])
    assert not incomplete.complete


def test_collection_rejects_inconsistent_counts():
    assert "!= len(records)" in rejects(c.CollectionResult, **good_collection(returned_count=3))
    assert "exceeds reported_total" in rejects(c.CollectionResult, **good_collection(reported_total=1))
    assert "reported_total != returned_count" in rejects(c.CollectionResult, **good_collection(reported_total=5, complete=True))


def test_collection_rejects_false_completeness_claims():
    assert "only allowed with status complete/empty" in rejects(c.CollectionResult, **good_collection(status="incomplete"))
    assert "requires complete=True" in rejects(c.CollectionResult, **good_collection(complete=False))
    assert "use status empty" in rejects(c.CollectionResult, records=[], returned_count=0, complete=True, status="complete")
    assert "non-zero total" in rejects(c.CollectionResult, records=[], returned_count=0, reported_total=3, complete=True, status="empty")


def test_collection_rejects_unknown_status_word():
    assert "Input should be 'complete'" in rejects(c.CollectionResult, **good_collection(status="done"))


def test_collection_rejects_nan_inside_a_record_but_allows_any_server_key():
    assert "NaN or infinity" in rejects(c.CollectionResult, **good_collection(
        records=[{"studyDbId": "s1"}, {"value": float("nan")}]))
    # Server data may contain any key name; the authority scan is for MODEL payloads only.
    ok = c.CollectionResult(**good_collection(records=[{"approved": True}, {"studyDbId": "s2"}]))
    assert ok.returned_count == 2


# --- 4. ArtifactManifest ----------------------------------------------------------

def test_manifest_accepts_managed_path_and_empty_table():
    manifest = c.ArtifactManifest(**good_manifest())
    assert manifest.measurement.unit == "t/ha"
    empty = c.ArtifactManifest(**good_manifest(row_count=0, complete=True, quality_notes=["zero rows"]))
    assert empty.row_count == 0


@pytest.mark.parametrize("bad_path", [
    "C:/Users/someone/secret.csv", "/etc/passwd", "out/../../.env", "out\\run\\x.csv", "", "out//x.csv",
])
def test_manifest_rejects_unmanaged_paths(bad_path):
    assert rejects(c.ArtifactManifest, **good_manifest(relative_path=bad_path))


def test_manifest_rejects_bad_ids_hashes_and_duplicates():
    assert "match pattern" in rejects(c.ArtifactManifest, **good_manifest(artifact_id="0001"))
    assert "match pattern" in rejects(c.ArtifactManifest, **good_manifest(sha256="abc"))
    assert "duplicates" in rejects(c.ArtifactManifest, **good_manifest(study_ids=["S1", "S1"]))
    assert "Input should be" in rejects(c.ArtifactManifest, **good_manifest(kind="spreadsheet"))


# --- 5. ToolResult ----------------------------------------------------------------

def test_tool_result_ok_or_coded_error_never_both():
    ok = c.ToolResult(ok=True, data={"count": 3}, request_ids=["req_0001"], complete=True)
    assert ok.data["count"] == 3
    err = c.ToolResult(ok=False, error=c.ToolError(code="not_found", message="no study S9"))
    assert err.error.code == "not_found"
    assert "requires a structured error" in rejects(c.ToolResult, ok=False)
    assert "cannot carry an error" in rejects(c.ToolResult, ok=True, error=c.ToolError(code="timeout", message="x"))


def test_tool_result_rejects_plain_string_error_and_unknown_code():
    assert rejects(c.ToolResult, ok=False, error="failed")
    assert "Input should be" in rejects(c.ToolError, code="oops", message="x")


def test_tool_result_rejects_nan_hidden_in_data():
    assert "NaN or infinity" in rejects(c.ToolResult, ok=True, data={"rows": [{"v": math.inf}]})
    assert "NaN or infinity" in rejects(c.ToolResult, ok=True, data=[1.0, [2.0, float("nan")]])


# --- 6. FetchApproval -------------------------------------------------------------

def test_approval_accepts_and_answers_permission_questions():
    approval = c.FetchApproval(**good_approval())
    assert approval.allows_family("studies") and not approval.allows_family("locations")
    assert approval.allows_study("S1") and not approval.allows_study("S2")
    assert approval.allows_variable("V1")
    assert approval.is_valid_at(T0 + timedelta(minutes=30))
    assert not approval.is_valid_at(T0 + timedelta(hours=2))


def test_approval_rejects_scope_that_contradicts_its_own_limits():
    assert "max_observation_studies" in rejects(c.FetchApproval, **good_approval(observation_study_ids=["S1", "S2"]))
    assert "after issued_at" in rejects(c.FetchApproval, **good_approval(expires_at_utc=T0))
    assert "exact observation_study_ids" in rejects(c.FetchApproval, **good_approval(observation_study_ids=[]))
    assert "without an observation endpoint" in rejects(c.FetchApproval, **good_approval(endpoint_families=["studies"]))
    assert "observation_variable_ids" in rejects(c.FetchApproval, **good_approval(observation_variable_ids=[]))
    assert "duplicates" in rejects(c.FetchApproval, **good_approval(endpoint_families=["studies", "studies", "observations"]))
    assert "greater than or equal to 1" in rejects(c.FetchApproval, **good_approval(max_http_attempts=0))


def test_approval_cannot_claim_a_model_issued_it_and_cannot_be_edited():
    assert "human_controller" in rejects(c.FetchApproval, **good_approval(issued_by="model"))
    approval = c.FetchApproval(**good_approval())
    with pytest.raises(ValidationError):
        approval.max_http_attempts = 1000


def test_agent_payload_cannot_smuggle_an_approval():
    text = rejects(c.AgentResult, **good_agent(payload={"plan": {"steps": [{"fetch_approval": {"max_http_attempts": 99}}]}}))
    assert "authority field 'fetch_approval'" in text
    assert "authority field 'APPROVED'" in rejects(c.AgentResult, **good_agent(payload={"APPROVED": True}))


# --- 7. Claim ---------------------------------------------------------------------

def test_claim_carries_value_unit_and_named_denominator():
    claim = c.Claim(**good_claim())
    assert claim.value == 18.0 and claim.denominator.name == "valid plot-level values" and claim.denominator.n == 4
    count = c.Claim(**good_claim(claim_id="clm_0002", kind="count", value=51, unit=None,
                                 denominator=c.Denominator(name="studies in complete metadata", n=51)))
    assert count.value == 51


def test_claim_missing_value_needs_a_reason_and_never_nan():
    missing = c.Claim(**good_claim(value=None, missing_reason="all six raw values were empty or invalid"))
    assert missing.value is None
    assert "non-empty missing_reason" in rejects(c.Claim, **good_claim(value=None))
    assert "only allowed when value is null" in rejects(c.Claim, **good_claim(value=18.0, missing_reason="x"))
    assert "finite" in rejects(c.Claim, **good_claim(value=float("nan")))
    assert "finite" in rejects(c.Claim, **good_claim(value=float("inf")))
    assert "valid number" in rejects(c.Claim, **good_claim(value="18"))


def test_claim_needs_evidence_and_rejects_unit_on_a_count():
    assert "at least 1 item" in rejects(c.Claim, **good_claim(source_artifact_hashes=[]))
    assert "duplicates" in rejects(c.Claim, **good_claim(source_artifact_hashes=[SHA_A, SHA_A]))
    assert "count has no unit" in rejects(c.Claim, **good_claim(kind="count", value=3, unit="t/ha"))
    assert "Input should be" in rejects(c.Claim, **good_claim(kind="average"))


# --- 8. AgentResult ---------------------------------------------------------------

def test_agent_result_accepts_and_keeps_usage_nullable():
    done = c.AgentResult(**good_agent())
    assert done.usage.total_tokens == 120
    unknown_usage = c.AgentResult(**good_agent(usage=None))
    assert unknown_usage.usage is None            # unknown, not zero


def test_agent_result_status_rules():
    assert "requires a payload" in rejects(c.AgentResult, **good_agent(payload=None))
    assert "structured error" in rejects(c.AgentResult, **good_agent(status="failed", payload=None))
    failed = c.AgentResult(**good_agent(status="failed", payload=None,
                                        errors=[c.StructuredError(code="model_timeout", message="no reply in 60s", stage="retriever")]))
    assert failed.errors[0].code == "model_timeout"
    assert "Input should be" in rejects(c.AgentResult, **good_agent(status="success"))
    assert "total_tokens" in rejects(c.Usage, prompt_tokens=1, completion_tokens=2, total_tokens=4)
    assert "greater than or equal to 0" in rejects(c.AgentResult, **good_agent(tool_calls=-1))


def test_agent_result_log_path_is_managed_and_payload_finite():
    assert "relative, not absolute" in rejects(c.AgentResult, **good_agent(log_path="C:/temp/run.log"))
    assert "NaN or infinity" in rejects(c.AgentResult, **good_agent(payload={"mean": float("nan")}))


# --- 9. RetrievalReport -----------------------------------------------------------

def test_retrieval_report_keeps_zero_row_and_failed_studies_on_the_ledger():
    manifest = c.ArtifactManifest(**good_manifest())
    empty_manifest = c.ArtifactManifest(**good_manifest(artifact_id="art_0002", study_ids=["S2"], row_count=0,
                                                         relative_path="out/run_0001/observations_S2_V1.csv", sha256=SHA_B))
    report = c.RetrievalReport(
        status="incomplete", complete=False,
        resolved=[c.ResolvedEntity(kind="study", id="S1", label="Synthetic study one", source_request_ids=["req_0001"])],
        artifacts=[manifest, empty_manifest],
        study_ledger=[
            c.StudyStatusEntry(study_id="S1", status="complete", artifact_id="art_0001"),
            c.StudyStatusEntry(study_id="S2", status="empty", artifact_id="art_0002"),
            c.StudyStatusEntry(study_id="S3", status="failed", reason="timeout after 1 attempt"),
            c.StudyStatusEntry(study_id="S4", status="blocked", reason="not in FetchApproval"),
        ],
        request_ids=["req_0001"], warnings=["S3 and S4 have no data"],
    )
    assert [e.study_id for e in report.study_ledger] == ["S1", "S2", "S3", "S4"]


def test_retrieval_report_rejects_lies_about_completeness():
    manifest = c.ArtifactManifest(**good_manifest())
    good = c.StudyStatusEntry(study_id="S1", status="complete", artifact_id="art_0001")
    bad = c.StudyStatusEntry(study_id="S3", status="failed", reason="timeout")
    assert "not complete/empty" in rejects(c.RetrievalReport, status="completed", complete=True,
                                            artifacts=[manifest], study_ledger=[good, bad])
    assert "requires complete=True" in rejects(c.RetrievalReport, status="completed", complete=False,
                                                artifacts=[manifest], study_ledger=[good])
    assert "unknown artifact" in rejects(c.RetrievalReport, status="completed", complete=True, study_ledger=[good])
    assert "duplicates" in rejects(c.RetrievalReport, status="completed", complete=True,
                                    artifacts=[manifest], study_ledger=[good, good])


def test_study_status_entry_rules():
    assert "point at its artifact" in rejects(c.StudyStatusEntry, study_id="S1", status="complete")
    assert "needs a reason" in rejects(c.StudyStatusEntry, study_id="S1", status="blocked")
    assert "Input should be" in rejects(c.StudyStatusEntry, study_id="S1", status="done", reason="x")


# --- 10. AnalysisReport -----------------------------------------------------------

def test_analysis_report_with_the_six_row_synthetic_example():
    pooled = c.Claim(**good_claim())
    clone_a = c.Claim(**good_claim(claim_id="clm_0002", value=12.0, grouping=["germplasmDbId"],
                                   filters={"study": "S1", "variable": "V1", "clone": "A"},
                                   denominator=c.Denominator(name="valid plot-level values for clone A", n=2),
                                   n_independent_units=2, exclusions=[c.Exclusion(reason="raw value empty", count=1)]))
    clone_b = c.Claim(**good_claim(claim_id="clm_0003", value=24.0, grouping=["germplasmDbId"],
                                   filters={"study": "S1", "variable": "V1", "clone": "B"},
                                   denominator=c.Denominator(name="valid plot-level values for clone B", n=2),
                                   n_independent_units=2, exclusions=[c.Exclusion(reason="raw value not numeric", count=1)]))
    report = c.AnalysisReport(status="completed", claims=[pooled, clone_a, clone_b],
                              methods=["pooled descriptive mean of valid plot-level values"],
                              exclusions=[c.Exclusion(reason="raw value empty", count=1),
                                          c.Exclusion(reason="raw value not numeric", count=1)],
                              caveats=["descriptive means only; not adjusted genetic performance"],
                              result_artifact_ids=["art_0009"])
    assert [cl.value for cl in report.claims] == [18.0, 12.0, 24.0]
    round_trip = c.AnalysisReport.model_validate_json(c.dump_json(report))
    assert round_trip == report


def test_analysis_report_rejects_empty_completed_and_duplicate_claims():
    assert "at least one claim" in rejects(c.AnalysisReport, status="completed")
    claim = c.Claim(**good_claim())
    assert "duplicates" in rejects(c.AnalysisReport, status="completed", claims=[claim, claim])
    assert "match pattern" in rejects(c.AnalysisReport, status="completed", claims=[claim], result_artifact_ids=["table1"])


# --- serialization ----------------------------------------------------------------

def test_dump_json_produces_plain_json_without_nan_tokens():
    text = c.dump_json(c.ToolResult(ok=True, data={"mean": 18.0, "n": 4}, complete=True))
    parsed = json.loads(text)
    assert parsed["data"] == {"mean": 18.0, "n": 4} and parsed["contract_version"] == "1"
    assert "NaN" not in text and "Infinity" not in text


def test_json_input_is_strict_too():
    # From JSON, an integer where a string ID belongs is still rejected.
    with pytest.raises(ValidationError):
        c.StudyFilters.model_validate_json('{"location_id": 12}')
    # An ISO-8601 UTC string is the JSON form of a datetime and is accepted.
    text = c.dump_json(c.RequestRecord(**good_request()))
    assert c.RequestRecord.model_validate_json(text).requested_at_utc == T0
