"""Plan-derived approval families, checked with cold synthetic MCP reads.

All approvals below are fabricated for an invalid test hostname. No real human
approval is supplied and no external network/model call is made.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import anyio
import pytest

import brapi_mcp_server as srv
from brapi_client import Settings, TransportResponse
from contracts import FetchApproval, Plan
from synthetic_support import load_fixture
from breeding_mcp_server import BreedingMcpService
from mcp_bridge import McpTools
from plan_permissions import endpoint_families_for_plan


BASE = "https://plan-permissions-test.invalid/brapi/v2"


def make_plan(*steps, scope=None):
    return Plan.model_validate({
        "plan_id": "plan_permissions", "question": "Synthetic permission check",
        "interpretation": "Only the stated catalog or observation operation",
        "status": "ready", "created_at_utc": datetime.now(timezone.utc),
        "scope": {"base_url": BASE, **(scope or {})},
        "steps": [{"step_id": f"step_{i}", "agent": "retriever", "action": action,
                   "inputs": inputs} for i, (action, inputs) in enumerate(steps, 1)],
    })


@pytest.mark.parametrize("entity,family", [
    ("studies", "studies"), ("variables", "observationvariables"),
    ("locations", "locations"), ("programs", "programs"), ("seasons", "seasons"),
])
def test_catalog_question_has_only_its_explicit_family(entity, family):
    plan = make_plan(("export_metadata", {"entity": entity}))
    assert endpoint_families_for_plan(plan) == [family]


@pytest.mark.parametrize("action,family", [
    ("server_info", "serverinfo"), ("search_studies", "studies"),
    ("study_types", "studies"), ("get_study", "studies"),
    ("list_variables", "observationvariables"), ("get_observations", "observations"),
    ("get_observation_units", "observationunits"), ("list_locations", "locations"),
    ("list_programs", "programs"), ("list_seasons", "seasons"),
])
def test_each_known_database_action_maps_to_its_family(action, family):
    assert endpoint_families_for_plan(make_plan((action, {}))) == [family]


def test_observation_plan_keeps_its_id_verification_families():
    plan = make_plan(("get_observations", {"study_db_id": "S1", "variable_db_id": "V1"}),
                     ("get_observation_units", {"study_db_id": "S1"}),
                     scope={"study_ids": ["S1"], "variable_ids": ["V1"]})
    assert endpoint_families_for_plan(plan) == ["studies", "observationvariables", "observations", "observationunits"]


def test_location_season_and_program_filters_allow_required_identity_checks():
    plan = make_plan(("export_metadata", {"entity": "studies"}), scope={
        "location_ids": ["L1"], "season_ids": ["Y1"],
        "filters": {"program_id": "P1"},
    })
    assert endpoint_families_for_plan(plan) == ["studies", "locations", "programs", "seasons"]
    by_input = make_plan(("search_studies", {"location_id": "L1", "season_id": "Y1", "program_id": "P1"}))
    assert endpoint_families_for_plan(by_input) == endpoint_families_for_plan(plan)


def test_named_location_and_season_discovery_is_allowed_without_observation_permission():
    plan = make_plan(("search_studies", {}), scope={
        "filters": {"location_name": "SYNTHETIC location", "season_name": "2026"},
        "discovery_needed": True, "discovery_families": ["locations", "seasons", "studies"],
    })
    assert endpoint_families_for_plan(plan) == ["studies", "locations", "seasons"]


@pytest.mark.parametrize("action,inputs", [
    ("run_shell", {}), ("get_everything", {}),
    ("export_metadata", {}), ("export_metadata", {"entity": "germplasm"}),
    ("export_metadata", {"entity": ["locations", "programs"]}),
])
def test_unknown_operation_or_export_entity_fails_closed(action, inputs):
    with pytest.raises(ValueError, match="unsupported"):
        endpoint_families_for_plan(make_plan((action, inputs)))


def test_local_evidence_does_not_grant_any_database_family():
    assert endpoint_families_for_plan(make_plan(("request_log", {"run_id": "run_permissions"}))) == []


def test_repeated_actions_do_not_duplicate_permissions_or_depend_on_step_order():
    steps = [("list_seasons", {}), ("export_metadata", {"entity": "locations"}),
             ("list_locations", {})]
    expected = ["locations", "seasons"]
    assert endpoint_families_for_plan(make_plan(*steps)) == expected
    assert endpoint_families_for_plan(make_plan(*reversed(steps))) == expected


def test_cold_locations_export_succeeds_over_mcp_but_unrelated_reads_remain_blocked(tmp_path):
    plan = make_plan(("export_metadata", {"entity": "locations"}))
    now = datetime.now(timezone.utc)
    approval = FetchApproval(
        approval_id="appr_permissions", run_id="run_permissions", base_url=BASE,
        endpoint_families=endpoint_families_for_plan(plan),
        max_http_attempts=3, max_observation_studies=0,
        issued_at_utc=now-timedelta(minutes=1), expires_at_utc=now+timedelta(minutes=10),
    )
    ctx = srv.build_context(offline=False, settings=Settings(base_url=BASE, mode="live"),
                            cache_dir=tmp_path / "cold", out_dir=tmp_path / "out",
                            run_id="run_permissions", approval=approval)
    calls = []

    def transport(request):
        calls.append(request.url.removeprefix(BASE))
        assert calls[-1] == "/locations", "Unapproved endpoint reached synthetic HTTP"
        _, body = load_fixture("synthetic_locations.json")
        return TransportResponse(200, {"Content-Type": "application/json"}, body)

    ctx.client._transport = transport

    async def scenario():
        async with McpTools.memory(ctx, server_factory=lambda _: BreedingMcpService(ctx).server) as tools:
            assert await tools.schemas()
            result = await tools.call("export_metadata", {"entity": "locations"})
            assert result.ok and result.complete and result.data["row_count"] == 1
            assert result.artifact_ids
            for entity in ("studies", "variables", "programs", "seasons"):
                refused = await tools.call("export_metadata", {"entity": entity})
                assert not refused.ok and not refused.complete
            refused = await tools.call("get_observations", {"study_db_id": "S1", "variable_db_id": "V1"})
            assert not refused.ok and not refused.complete

    anyio.run(scenario)
    assert calls == ["/locations"]
