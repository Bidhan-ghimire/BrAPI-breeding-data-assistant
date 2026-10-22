"""Test helpers from the laptop copy (part2) that the release removed from its app modules.

The code of each block below marked "from part2/..." is copied verbatim from that file (commit 7e40634
archive of CassavaBase/part2). The blocks marked "new" are test setup written for the release:
`synthetic_context` reproduces the removed build_context(fixture="synthetic") branch, and the
"part2 run flow" hooks let a ported test run the release's agent-led flow the way part2's mock mode did.
Nothing here is imported by the application.
"""
from __future__ import annotations

import dataclasses
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import run as ctl
from agents.analyst import FIXTURE_DIR as ANALYST_FIXTURES, load_analyst_script
from agents.coordinator import KnownMetadata, find_candidates
from agents.retriever import load_model_script
from brapi_client import PART2_DIR, Settings
from brapi_mcp_server import ToolContext, build_context
from cache_store import CacheEntry, CacheIdentity, CacheStore, sha256_bytes
from contracts import FetchApproval
from llm import FakeModelClient, reply_text
from run import Models

# --- from part2/explore_cassavabase.py ---------------------------------------------------
SYNTHETIC_BASE = "https://synthetic.invalid/brapi/v2"      # .invalid is reserved: it can never resolve
FIXTURE_DIR = PART2_DIR / "tests" / "fixtures" / "brapi"
FIXTURE_FETCHED_AT = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)   # a fixed, fake fetch time

NORMAL_FIXTURES = [
    "synthetic_serverinfo.json",
    "synthetic_studies.json",
    "synthetic_study_S1.json",          # GET /studies/S1 — the same record as the S1 row of the list
    "synthetic_variables.json",
    "synthetic_observations_S1_V1.json",
    "synthetic_observationunits_S1.json",
    "synthetic_plotrecords_S1.json",    # the same plots with their values (item (3), 2026-10-22): read on a server that drops zeros
    "synthetic_locations.json",
    "synthetic_programs.json",
    "synthetic_seasons.json",
]
MALFORMED_FIXTURES = {
    "null_result_with_count": "malformed_observations_null_result_with_count.json",
    "no_pagination": "malformed_observations_no_pagination.json",
    "duplicate_id": "malformed_observations_duplicate_id.json",
}


def load_fixture(name: str, fixture_dir: Path = FIXTURE_DIR) -> tuple[dict[str, Any], bytes]:
    """Return (parsed JSON, exact bytes). Refuses a fixture that is not labelled SYNTHETIC."""
    path = fixture_dir / name
    body = path.read_bytes()
    parsed = json.loads(body.decode("utf-8"))
    if parsed.get("_synthetic") is not True or "SYNTHETIC" not in str(parsed.get("_note", "")):
        raise ValueError(f"{name} is not labelled SYNTHETIC; refusing to use it as a fixture")
    if "_request" not in parsed or "metadata" not in parsed or "result" not in parsed:
        raise ValueError(f"{name} lacks _request/metadata/result")
    return parsed, body


def seed_fixture(cache: CacheStore, base_url: str, name: str, fixture_dir: Path = FIXTURE_DIR) -> CacheEntry:
    """Put one fixture into the cache under the request it stands for. Idempotent: same bytes -> reuse."""
    parsed, body = load_fixture(name, fixture_dir)
    request = parsed["_request"]
    identity = CacheIdentity(base_url=base_url, endpoint=request["endpoint"], params=dict(request.get("params", {})))
    existing = cache.get(identity)
    if existing is not None and existing[0].response_sha256 == sha256_bytes(body):
        return existing[0]
    query = "&".join(f"{k}={identity.params[k]}" for k in sorted(identity.params))
    redacted = f"{base_url}{identity.endpoint}" + (f"?{query}" if query else "")
    return cache.put(identity, body, http_status=200, fetched_at_utc=FIXTURE_FETCHED_AT,
                     source_url_redacted=redacted, content_type="application/json")


def seed_synthetic(cache: CacheStore, base_url: str = SYNTHETIC_BASE,
                   names: list[str] | None = None, fixture_dir: Path = FIXTURE_DIR) -> list[CacheEntry]:
    return [seed_fixture(cache, base_url, name, fixture_dir) for name in (names or NORMAL_FIXTURES)]


# --- from part2/agents/retriever.py ------------------------------------------------------

def synthetic_approval(*, run_id: str, studies: list[str] | None = None, variables: list[str] | None = None,
                       max_studies: int | None = None, base_url: str = SYNTHETIC_BASE) -> FetchApproval:
    """An explicit approval for the SYNTHETIC server, built by harness/controller code — never by a model."""
    now = datetime.now(timezone.utc)
    studies = ["S1"] if studies is None else list(studies)
    variables = ["V1"] if variables is None else list(variables)
    return FetchApproval(
        approval_id=f"appr_synth_{uuid.uuid4().hex[:8]}", run_id=run_id, base_url=base_url,
        endpoint_families=["serverinfo", "studies", "observationvariables", "locations", "programs", "seasons",
                           "observations", "observationunits"],
        observation_study_ids=studies, observation_variable_ids=variables,
        max_http_attempts=50, max_observation_studies=len(studies) if max_studies is None else max_studies,
        issued_at_utc=now - timedelta(seconds=1), expires_at_utc=now + timedelta(hours=1),
    )


# --- from part2/run.py ---------------------------------------------------------------------

def synthetic_plan_draft(study_id: str, variable_id: str) -> dict[str, Any]:
    """The A/B plan for the teaching fixture, as a Coordinator draft (no counts, no budgets, no approvals)."""
    return {
        "interpretation": f"pooled and per-clone descriptive means of variable {variable_id} in study {study_id}, with named n",
        "clarifications": [],
        "scope": {"study_ids": [study_id], "variable_ids": [variable_id]},
        "statistic": {"kind": "mean", "denominator": "valid plot-level values", "grouping": ["germplasmDbId"]},
        "steps": [
            {"step_id": "step_1", "agent": "retriever", "action": "get_observations", "inputs": {"study_db_id": study_id, "variable_db_id": variable_id}},
            {"step_id": "step_2", "agent": "retriever", "action": "get_observation_units", "inputs": {"study_db_id": study_id}},
            {"step_id": "step_3", "agent": "analyst", "action": "numeric_summary", "depends_on": ["step_1", "step_2"]},
            {"step_id": "step_4", "agent": "analyst", "action": "group_stats", "inputs": {"by": "germplasmDbId"}, "depends_on": ["step_1", "step_2"]},
        ],
    }


def mock_coordinator_model(question: str, metadata: KnownMetadata | None) -> FakeModelClient:
    """A scripted Coordinator that models ambiguity: a valid plan ONLY for a question naming one study and one trait."""
    cands = find_candidates(question, metadata)
    if metadata is not None and len(cands.studies) == 1 and len(cands.variables) == 1 and not cands.decision_words:
        payload = synthetic_plan_draft(cands.studies[0].study_id, cands.variables[0].variable_id)
        return FakeModelClient([reply_text(json.dumps({"status": "completed", "payload": payload}), model="mock-model")], reported_model="mock-model")
    if cands.decision_words:
        question_text = (f"The question asks for {', '.join(cands.decision_words)}, a breeding judgement this assistant does not make. "
                         "Which study and which trait should be described with plain means and named n?")
    elif metadata is None:
        question_text = "No validated metadata is available offline; which study and trait should be discovered?"
    elif not cands.studies:
        question_text = f"Which study is meant? Known studies: {', '.join(s.study_id + ' ' + repr(s.name) for s in metadata.studies)}."
    else:
        question_text = f"Which trait is meant? Known variables: {', '.join(v.variable_id + ' ' + repr(v.name) for v in metadata.variables)}."
    return FakeModelClient([reply_text(json.dumps({"status": "needs_clarification", "question": question_text}), model="mock-model")], reported_model="mock-model")


def synthetic_models() -> Models:
    return Models(coordinator=None, retriever=load_model_script("valid_request.json"),
                  analyst_factory=lambda prefix: load_analyst_script("valid_ab.json", ANALYST_FIXTURES, prefix=prefix), requested="mock-model")


# --- new: part2's build_context(fixture="synthetic") branch, on the release's build_context -------

def synthetic_context(*, offline: bool = True, cache_dir: Path | None = None, out_dir: Path | None = None,
                      run_id: str | None = None, approval: FetchApproval | None = None) -> ToolContext:
    """Same as part2's build_context(fixture="synthetic", ...): the invented server, its cache seeded with the fixtures."""
    cache_dir = Path(cache_dir) if cache_dir else PART2_DIR / "cache" / "synthetic"
    settings = Settings(base_url=SYNTHETIC_BASE, mode="offline", cache_dir=cache_dir)
    seed_synthetic(CacheStore(cache_dir), SYNTHETIC_BASE)      # includes the GET /studies/S1 reply (synthetic_study_S1.json)
    return build_context(offline=True, cache_dir=cache_dir, out_dir=out_dir, run_id=run_id, approval=approval, settings=settings)


# --- new: part2's run flow on the release's runtime (setup for ported controller and chat tests) ----------
# The release refuses part2's offline/mock/direct/--auto modes (run.RunConfig.validate). A ported test runs the
# release flow instead and gets, through these hooks, what part2's mock mode supplied: the scripted planner above,
# a supervisor that always takes the next step, the person's "continue" at the data review the release added, and
# for tests written for --auto (no answer function) a scripted person who approves the data request and nothing else.

PART2_STAGE_ACTIONS = {"planned": "retrieve", "data_reviewed": "analyze", "analyzed": "finish"}


class StageSupervisor:
    """Always delegates the next step (plan -> retrieve -> analyze -> finish), like the release's own scripted
    supervisor in coordination_runtime._choose (its mock_model branch)."""

    reported_model = "mock-model"

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def complete(self, messages, tools, *, max_tokens):
        self.requests.append({"messages": [dict(m) for m in messages], "max_tokens": max_tokens})
        stage = json.loads(messages[1]["content"])["state"]["stage"]
        return reply_text(json.dumps({"status": "completed", "payload": {
            "action": PART2_STAGE_ACTIONS[stage], "reason": "Synthetic offline supervisor script."}}), model="mock-model")


def part2_runtime(monkeypatch, *, server: str | None = None, review: str | None = "continue") -> None:
    """Install the hooks for one test. server: also select the live mode on that (synthetic) server."""
    if server is not None:
        monkeypatch.setenv("BRAPI_MODE", "live")
        monkeypatch.setenv("BRAPI_BASE_URL", server)

    async def model_for_coordinator(self):
        if self.models.coordinator is not None:
            return self.models.coordinator
        return mock_coordinator_model(self.config.question, self.metadata)

    post_init = ctl.Controller.__post_init__

    def with_supervisor(self) -> None:
        post_init(self)
        if self.models.supervisor is None:
            self.models = dataclasses.replace(self.models, supervisor=StageSupervisor())

    ask = ctl.Controller._ask

    def ask_like_part2(self, prompt: str):
        if review is not None and prompt == ctl.DATA_REVIEW_PROMPT:
            return review
        if self.answer_fn is None:                        # written for --auto: approve the data request, nothing else
            return "approve" if prompt.startswith("approve / edit / cancel") else None
        return ask(self, prompt)

    monkeypatch.setattr(ctl.Controller, "_model_for_coordinator", model_for_coordinator)
    monkeypatch.setattr(ctl.Controller, "__post_init__", with_supervisor)
    monkeypatch.setattr(ctl.Controller, "_ask", ask_like_part2)


def part2_config(**values: Any) -> "ctl.RunConfig":
    """part2's RunConfig keywords on the release's RunConfig: fixture="synthetic" became a seeded cache."""
    fixture = values.pop("fixture", None)
    if fixture == "synthetic":
        seed_synthetic(CacheStore(values["cache_dir"]), SYNTHETIC_BASE)
    elif fixture is not None:
        raise ValueError(f"unknown fixture set {fixture!r}; only 'synthetic' exists")
    return ctl.RunConfig(**values)


def async_returning(value: Any):
    """An async method that returns value: part2 tests replaced the sync Controller.load_metadata; the release reads
    the catalogs in the async Controller._load_metadata_mcp."""
    async def method(self):
        return value
    return method


def part2_memory_default(monkeypatch) -> None:
    """part2's McpTools.memory(ctx) served brapi_mcp_server.build_server(ctx). The release requires a server_factory;
    this names the release's shared service through a patchable brapi_mcp_server.build_server, as part2 had."""
    import brapi_mcp_server as srv
    from breeding_mcp_server import BreedingMcpService
    from mcp_bridge import McpTools

    monkeypatch.setattr(srv, "build_server", lambda ctx: BreedingMcpService(ctx).server, raising=False)
    memory = McpTools.memory.__func__

    def memory_with_default(cls, ctx, *args, server_factory=None, **kwargs):
        return memory(cls, ctx, *args, server_factory=server_factory or (lambda owned: srv.build_server(owned)), **kwargs)

    monkeypatch.setattr(McpTools, "memory", classmethod(memory_with_default))


async def analysis_service(ctx: ToolContext, artifact_ids: list[str]):
    """part2's build_analyst_server(registry, ids): the release's shared service, moved to its analysis stage."""
    from breeding_mcp_server import BreedingMcpService

    service = BreedingMcpService(ctx)
    await service.suspend()
    await service.enable_analysis(list(artifact_ids))
    return service
