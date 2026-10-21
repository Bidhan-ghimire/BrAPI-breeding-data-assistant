"""Derive the database families a human is asked to approve for a typed plan, and the fetch list itself.

This describes proposed access, never grants it. The controller still binds a
human's approval to the complete manifest; observation IDs and all request
budgets remain enforced by the existing controller/client guards. The plan
check (agents.coordinator.assemble_plan) builds the same fetch list, so a plan
the fetch list cannot hold goes back to the planner instead of failing later.
"""
from __future__ import annotations

from typing import get_args

from contracts import EndpointFamily, FetchManifest, Plan


_ACTION_FAMILIES: dict[str, EndpointFamily | None] = {
    "server_info": "serverinfo",
    "search_studies": "studies",
    "study_types": "studies",
    "get_study": "studies",
    "list_variables": "observationvariables",
    "get_observations": "observations",
    "get_observation_units": "observationunits",
    "list_locations": "locations",
    "list_programs": "programs",
    "list_seasons": "seasons",
    "request_log": None,  # Reads this run's local evidence, never the database.
}
_EXPORT_FAMILIES: dict[str, EndpointFamily] = {
    "studies": "studies",
    "variables": "observationvariables",  # BrAPI family differs from artifact name.
    "locations": "locations",
    "programs": "programs",
    "seasons": "seasons",
}
_INPUT_LOOKUPS: dict[str, EndpointFamily] = {
    "study_db_id": "studies",
    "variable_db_id": "observationvariables",
    "location_id": "locations",
    "season_id": "seasons",
    "program_id": "programs",
}


def endpoint_families_for_plan(plan: Plan) -> list[EndpointFamily]:
    """Return only planned read families and their required ID/name lookups.

    Catalog-only questions do not inherit permission for studies or traits.
    Observation plans still allow their study/trait lookups: the Retriever must
    verify IDs from tool results in this run. The same requirement applies to
    a plan's already-resolved location/season IDs, including study filters.
    Unknown actions/entities fail closed instead of receiving broad access.
    """
    families: set[EndpointFamily] = set(plan.scope.discovery_families)
    for step in plan.steps:
        if step.agent != "retriever":
            continue
        if step.action == "export_metadata":
            entity = step.inputs.get("entity")
            if not isinstance(entity, str) or entity not in _EXPORT_FAMILIES:
                raise ValueError(f"unsupported export_metadata entity in {step.step_id}: {entity!r}")
            families.add(_EXPORT_FAMILIES[entity])
        elif step.action in _ACTION_FAMILIES:
            family = _ACTION_FAMILIES[step.action]
            if family is not None:
                families.add(family)
        else:
            raise ValueError(f"unsupported retriever action in {step.step_id}: {step.action!r}")
        for key, family in _INPUT_LOOKUPS.items():
            if step.inputs.get(key):
                families.add(family)

    scope = plan.scope
    if scope.study_ids:
        families.add("studies")
    if scope.variable_ids:
        families.add("observationvariables")
    if scope.location_ids or scope.filters.location_id or scope.filters.location_name:
        families.add("locations")
    if scope.season_ids or scope.filters.season_id or scope.filters.season_name:
        families.add("seasons")
    if scope.filters.program_id:
        families.add("programs")
    # Stable ordering keeps the approval display/hash deterministic. No fallback
    # family: a plan with only local evidence reads grants no database access.
    return [family for family in get_args(EndpointFamily) if family in families]


def required_fetches(plan: Plan, manifest: FetchManifest | None = None) -> tuple[set[tuple[str, str]], set[str]]:
    """Required observation coverage is the full scope, even if a model omitted a step for one pair.

    A manifest is authoritative after approval. Unit rosters are required only when the plan uses them.
    This is a minimum call estimate, not a promise about additional metadata calls or pagination."""
    studies = set(manifest.observation_study_ids if manifest is not None else plan.scope.study_ids)
    variables = set(manifest.observation_variable_ids if manifest is not None else plan.scope.variable_ids)
    observation_plan = any(s.agent == "retriever" and s.action == "get_observations" for s in plan.steps)
    pairs = {(study, variable) for study in studies for variable in variables} if observation_plan else set()
    units: set[str] = set()
    for step in plan.steps:
        if step.agent == "retriever" and step.action == "get_observation_units":
            explicit = step.inputs.get("study_db_id")
            units.update(({explicit} & studies) if isinstance(explicit, str) else studies)
    return pairs, units


def fetch_manifest(plan: Plan, *, run_id: str, base_url: str, max_http_attempts: int,
                   study_ids: list[str] | None = None) -> FetchManifest:
    """The exact fetch list a human approves for this plan (run.Controller.resolve_manifest). Raises when the plan cannot
    be fetched: ValueError for an unsupported retriever step, pydantic's ValidationError when FetchManifest refuses it."""
    studies = list(plan.scope.study_ids if study_ids is None else study_ids)
    variables = list(plan.scope.variable_ids)
    families = endpoint_families_for_plan(plan)
    pairs, units = required_fetches(plan)
    return FetchManifest(run_id=run_id, base_url=base_url, origin="live",
                         endpoint_families=families,   # type: ignore[arg-type]
                         observation_study_ids=studies if (pairs or units) else [], observation_variable_ids=variables if pairs else [],
                         max_http_attempts=max_http_attempts, max_observation_studies=len(studies) if (pairs or units) else 0)
