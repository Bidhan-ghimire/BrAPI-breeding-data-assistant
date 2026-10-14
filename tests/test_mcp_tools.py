"""
tests/test_mcp_tools.py — the DIRECT route: call the twelve tools through dispatch()
in this process, on the SYNTHETIC fixtures, with no MCP transport involved.

What this route proves: the tools' logic, argument validation and the ToolResult
contract. What it cannot prove: that the tools work over a real MCP connection
(that is tests/test_mcp_stdio.py).
"""
from __future__ import annotations

import json

import pytest

import brapi_mcp_server as srv
from artifacts import ArtifactRegistry
from brapi_client import BrapiClient, Settings
from cache_store import CacheStore
from contracts import ToolResult
from part2_support import MALFORMED_FIXTURES, SYNTHETIC_BASE, seed_fixture
from part2_support import synthetic_context

EXPECTED_ORDER = [
    "server_info", "search_studies", "study_types", "get_study", "list_variables", "get_observations",
    "list_locations", "list_programs", "list_seasons", "request_log", "get_observation_units", "export_metadata",
]


@pytest.fixture
def ctx(tmp_path):
    return synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_direct")


def call(ctx, name, args=None) -> ToolResult:
    result = srv.dispatch(ctx, name, args)
    assert isinstance(result, ToolResult)
    json.loads(srv.dump_json(result))            # every result serializes cleanly (no NaN, no objects)
    return result


# --- 1. one registry, twelve tools, strict schemas ------------------------------------------

def test_exactly_twelve_tools_in_the_documented_order():
    assert srv.TOOL_NAMES == EXPECTED_ORDER
    schemas = srv.tool_schemas()
    assert [s["name"] for s in schemas] == EXPECTED_ORDER
    for schema in schemas:
        assert schema["description"], schema["name"]                     # every tool documented
        assert schema["inputSchema"]["type"] == "object"
        assert schema["inputSchema"]["additionalProperties"] is False    # unknown fields refused by the schema too
    for word in ("Limits", "complete", "not"):
        assert word in srv.TOOLS["search_studies"].description


def test_schema_and_dispatch_come_from_the_same_model(ctx):
    spec = srv.TOOLS["search_studies"]
    assert set(spec.input_schema()["properties"]) == set(spec.args_model.model_fields)
    assert spec.input_schema()["properties"]["limit"]["maximum"] == 100


@pytest.mark.parametrize("name,args,fragment", [
    ("get_study", {"study_db_id": 5}, "study_db_id"),                      # int where a string ID belongs
    ("get_study", {}, "study_db_id"),                                       # missing required
    ("search_studies", {"limit": 0}, "limit"),                              # below bound
    ("search_studies", {"limit": 101}, "limit"),                            # above bound
    ("search_studies", {"offset": -1}, "offset"),
    ("search_studies", {"approval": {"max_http_attempts": 99}}, "approval"),  # authority smuggling
    ("search_studies", {"base_url": "https://evil.example"}, "base_url"),  # cannot change the server
    ("get_observations", {"study_db_id": "S1", "variable_db_id": "V1", "endpoint": "/admin"}, "endpoint"),
    ("export_metadata", {"entity": "germplasm"}, "entity"),                 # not one of the five entities
    ("list_variables", {"name_contains": 7}, "name_contains"),
])
def test_bad_arguments_are_rejected_with_invalid_argument(ctx, name, args, fragment):
    result = call(ctx, name, args)
    assert not result.ok and result.error.code == "invalid_argument" and fragment in result.error.message


def test_unknown_tool_and_non_object_arguments(ctx):
    assert call(ctx, "run_shell", {}).error.code == "invalid_argument"
    assert call(ctx, "get_study", ["S1"]).error.code == "invalid_argument"
    assert "known" in call(ctx, "eval", {}).error.message


# --- 2. the twelve tools on the synthetic fixture -------------------------------------------

def test_server_info_identifies_the_synthetic_server(ctx):
    r = call(ctx, "server_info", {})
    assert r.ok and r.data["server_name"] == "SYNTHETIC fixture server" and r.data["base_url"] == SYNTHETIC_BASE
    assert "studies" in r.data["advertised_calls"] and r.request_ids and r.complete is None


def test_search_studies_pages_candidates_and_states_completeness(ctx):
    r = call(ctx, "search_studies", {"name_contains": "SYNTHETIC", "limit": 1})
    assert r.ok and r.data["total_matches"] == 1 and r.data["returned"] == 1 and r.data["next_offset"] is None
    assert r.data["candidates"][0]["studyDbId"] == "S1" and r.complete and r.data["complete"]
    none = call(ctx, "search_studies", {"name_contains": "nonexistent"})       # discovery filter, applied client-side
    assert none.ok and none.data["total_matches"] == 0 and none.data["status"] == "empty"
    # an exact server filter is a NEW request the offline cache does not hold; since 2026-09-29 the client applies it to the cached
    # complete catalog instead of failing, says so in a warning, and the honest answer here is "empty, complete"
    uncached = call(ctx, "search_studies", {"study_type": "Nonexistent Trial"})
    assert uncached.ok and uncached.data["total_matches"] == 0 and uncached.data["status"] == "empty" and uncached.complete
    assert any("applied client-side" in w for w in uncached.warnings)
    typed = call(ctx, "search_studies", {"study_type": "Synthetic Yield Trial"})
    assert typed.ok and typed.data["total_matches"] == 1 and typed.data["candidates"][0]["studyDbId"] == "S1" and typed.complete


def test_search_studies_refuses_human_names_for_exact_filters(ctx):
    r = call(ctx, "search_studies", {"location_name": "Ibadan"})     # not a field: resolve to location_id first
    assert not r.ok and r.error.code == "invalid_argument"


def test_study_types_counts_only_a_complete_collection(ctx):
    r = call(ctx, "study_types", {})
    assert r.ok and r.data["counts"] == {"Synthetic Yield Trial": 1} and r.complete


def test_get_study_returns_record_and_provenance(ctx):
    r = call(ctx, "get_study", {"study_db_id": "S1"})
    assert r.ok and r.data["study"]["studyName"].startswith("SYNTHETIC")
    assert r.data["provenance"]["origin"] == "cache" and r.data["provenance"]["response_sha256"]
    missing = call(ctx, "get_study", {"study_db_id": "S404"})
    assert not missing.ok and missing.error.code in ("unsupported", "not_found")   # offline: not cached


def test_list_variables_flattens_trait_method_scale(ctx):
    r = call(ctx, "list_variables", {"name_contains": "fresh"})
    assert r.ok and r.data["total_matches"] == 1
    v1 = r.data["variables"][0]
    assert v1["observationVariableDbId"] == "V1" and v1["units"] == "t/ha" and v1["methodName"] == "weigh all roots per plot"
    assert v1["timepoint"] is None                                       # not stated by the server -> None, not invented
    everything = call(ctx, "list_variables", {"limit": 1})
    assert everything.data["total_matches"] == 2 and everything.data["next_offset"] == 1


def test_get_observations_returns_a_handle_and_preview_never_the_table(ctx):
    r = call(ctx, "get_observations", {"study_db_id": "S1", "variable_db_id": "V1"})
    assert r.ok and r.complete and r.data["row_count"] == 6 and r.data["displayed_rows"] == 5
    assert len(r.data["preview_rows"]) == 5 and r.artifact_ids == [r.data["artifact_id"]]
    assert r.data["preview_rows"][2]["value"] == "" and r.data["preview_rows"][4]["value"] == "bad"   # raw tokens kept
    assert "records" not in r.data and "rows" not in r.data
    assert r.data["ledger"] == [{"contract_version": "1", "study_id": "S1", "status": "complete",
                                 "artifact_id": r.data["artifact_id"], "reason": None, "request_ids": r.request_ids}]
    manifest, rows = ctx.registry.load(r.data["artifact_id"])
    assert manifest.row_count == 6 and [row["value"] for row in rows] == ["10", "14", "", "20", "bad", "28"]


def test_get_observations_preserves_incomplete_and_unknown_status(tmp_path):
    cache = CacheStore(tmp_path / "cache")
    seed_fixture(cache, SYNTHETIC_BASE, MALFORMED_FIXTURES["no_pagination"])
    client = BrapiClient(Settings(base_url=SYNTHETIC_BASE, mode="offline", cache_dir=tmp_path / "cache"), cache=cache, run_id="run_bad")
    ctx = srv.ToolContext(client=client, registry=ArtifactRegistry(tmp_path / "out", "run_bad", base_url=SYNTHETIC_BASE))
    r = call(ctx, "get_observations", {"study_db_id": "S1", "variable_db_id": "V1"})
    assert r.ok and r.complete is False and r.data["status"] == "unknown" and r.data["row_count"] == 6
    assert r.data["ledger"][0]["status"] == "unknown" and r.data["ledger"][0]["reason"]


def test_get_observations_offline_miss_is_a_coded_error_with_the_study_on_the_ledger(ctx):
    r = call(ctx, "get_observations", {"study_db_id": "S9", "variable_db_id": "V1"})
    assert not r.ok and r.error.code == "unsupported" and "offline" in r.error.message
    assert r.data["row_count"] == 0 and r.data["ledger"][0]["study_id"] == "S9" and r.data["ledger"][0]["status"] == "failed"


def test_list_locations_programs_seasons(ctx):
    loc = call(ctx, "list_locations", {"name_contains": "synthetic"})
    assert loc.ok and loc.data["total_matches"] == 1 and loc.data["candidates"][0]["locationDbId"] == "L1" and loc.complete
    assert call(ctx, "list_locations", {"name_contains": "ibadan"}).data["total_matches"] == 0
    prog = call(ctx, "list_programs", {})
    assert prog.ok and prog.data["programs"][0]["programDbId"] == "PG1" and prog.complete
    sea = call(ctx, "list_seasons", {})
    assert sea.ok and sea.data["seasons"] == [{"seasonDbId": "2026", "seasonName": "2026", "year": "2026"}] and sea.complete


def test_request_log_is_per_run_and_redacted(ctx):
    call(ctx, "server_info", {})
    call(ctx, "get_study", {"study_db_id": "S1"})
    r = call(ctx, "request_log", {"run_id": "run_direct"})
    assert r.ok and r.data["count"] == 2 and [rec["endpoint"] for rec in r.data["records"]] == ["/serverinfo", "/studies/S1"]
    assert all(rec["origin"] == "cache" for rec in r.data["records"])
    assert "token" not in json.dumps(r.data)
    other = call(ctx, "request_log", {"run_id": "some_other_run"})
    assert not other.ok and other.error.code == "not_found"


def test_get_observation_units_keeps_levels(ctx):
    r = call(ctx, "get_observation_units", {"study_db_id": "S1"})
    assert r.ok and r.complete and r.data["row_count"] == 6 and r.data["levels"] == {"plot": 6}
    _, rows = ctx.registry.load(r.data["artifact_id"])
    assert rows[0]["levelName"] == "plot" and rows[0]["levelCode"] == "P1" and '"levelName": "plot"' in rows[0]["observationUnitPosition"]


def test_export_metadata_saves_complete_collections_only(ctx):
    r = call(ctx, "export_metadata", {"entity": "studies"})
    assert r.ok and r.complete and r.data["row_count"] == 1 and r.data["entity"] == "studies"
    _, rows = ctx.registry.load(r.data["artifact_id"])
    assert rows[0]["studyDbId"] == "S1" and rows[0]["seasons"] == '["2026"]'
    bad = call(ctx, "export_metadata", {"entity": "seasons", "filters": {"location_id": "L1"}})
    assert not bad.ok and bad.error.code == "invalid_argument"


def test_incomplete_export_is_refused_not_saved(tmp_path):
    cache = CacheStore(tmp_path / "cache")
    seed_fixture(cache, SYNTHETIC_BASE, MALFORMED_FIXTURES["no_pagination"])   # observations only; no /variables cached
    client = BrapiClient(Settings(base_url=SYNTHETIC_BASE, mode="offline", cache_dir=tmp_path / "cache"), cache=cache, run_id="run_x")
    ctx = srv.ToolContext(client=client, registry=ArtifactRegistry(tmp_path / "out", "run_x", base_url=SYNTHETIC_BASE))
    r = call(ctx, "export_metadata", {"entity": "variables"})
    assert not r.ok and r.error.code in ("unsupported", "incomplete_data") and ctx.registry.artifact_ids() == []


# --- 3. the self-test and the CLI ------------------------------------------------------------

def test_context_is_offline_even_if_the_environment_says_live(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAPI_MODE", "live")
    settings = Settings(base_url="https://cassavabase.org/brapi/v2", mode="live", cache_dir=tmp_path / "cache")
    ctx = srv.build_context(offline=True, settings=settings, cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_env")
    assert ctx.client.settings.mode == "offline" and ctx.approval is None
