"""Synthetic data and scripted models kept outside the published application."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json, re, uuid
from dataclasses import replace
from typing import Any
from brapi_client import Settings
from brapi_mcp_server import ToolContext, build_context, dispatch
from cache_store import CacheEntry, CacheIdentity, CacheStore, sha256_bytes
from contracts import FetchApproval
from agents.analyst import AnalysisInput
from llm import FakeModelClient, reply_text, reply_tools
import run as ctl
SYNTHETIC_BASE = "https://synthetic.invalid/brapi/v2"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE_DIR = FIXTURES / "brapi"
FIXTURE_FETCHED_AT = datetime(2026,6,1,tzinfo=timezone.utc)
NORMAL_FIXTURES = sorted(p.name for p in FIXTURE_DIR.glob("synthetic_*.json"))


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

def synthetic_inputs(ctx: ToolContext, question: str, *, requested_variable_ids: list[str] | None = None) -> AnalysisInput:
    """The teaching artifacts through the real pipeline: S1/V1 observations plus the S1 unit roster."""
    obs = dispatch(ctx, "get_observations", {"study_db_id": "S1", "variable_db_id": "V1"})
    units = dispatch(ctx, "get_observation_units", {"study_db_id": "S1"})
    if not (obs.ok and units.ok):
        raise RuntimeError("synthetic fixture did not load")
    return AnalysisInput(question=question, artifact_ids=[obs.artifact_ids[0], units.artifact_ids[0]],
                         requested_variable_ids=["V1"] if requested_variable_ids is None else list(requested_variable_ids),
                         notes=["variable V1 = fresh root yield (t/ha per the variables list); study S1 = SYNTHETIC yield trial S1",
                                f"{units.artifact_ids[0]} is the observation-unit roster (plots) of S1"])

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

def load_analyst_script(name: str, fixture_dir: Path = FIXTURES / "model_analyst", *, prefix: str) -> FakeModelClient:
    """A fixture is a JSON list of scripted replies labelled SYNTHETIC.

    Artifact handles are only known at run time, so a script writes {{ART:3}} for the third handle this run's
    registry issues (art_<prefix>_0003) and {{PREFIX}} for the six-character registry prefix (used in claim IDs).
    """
    text = (fixture_dir / name).read_text(encoding="utf-8")
    text = re.sub(r"\{\{ART:(\d+)\}\}", lambda m: f"art_{prefix}_{int(m.group(1)):04d}", text).replace("{{PREFIX}}", prefix)
    doc = json.loads(text)
    if doc.get("_synthetic") is not True:
        raise ValueError(f"{name} is not labelled SYNTHETIC")
    replies = []
    for step in doc["replies"]:
        kind = step.get("type")
        if kind == "tool_calls":
            replies.append(reply_tools([(c["name"], json.dumps(c["arguments"])) for c in step["calls"]], model="mock-model"))
        elif kind == "final":
            replies.append(reply_text(json.dumps(step["content"]), model="mock-model"))
        elif kind == "text":
            replies.append(reply_text(step["content"], model="mock-model"))
        else:
            raise ValueError(f"{name}: unknown reply type {kind!r}")
    return FakeModelClient(replies, reported_model="mock-model")

def load_model_script(name: str, fixture_dir: Path = FIXTURES / "model_retriever") -> FakeModelClient:
    """A fixture is a JSON list of scripted replies; every fixture is labelled SYNTHETIC."""
    doc = json.loads((fixture_dir / name).read_text(encoding="utf-8"))
    if doc.get("_synthetic") is not True:
        raise ValueError(f"{name} is not labelled SYNTHETIC")
    replies = []
    for step in doc["replies"]:
        kind = step.get("type")
        if kind == "tool_calls":
            replies.append(reply_tools([(c["name"], json.dumps(c["arguments"])) for c in step["calls"]], model="mock-model"))
        elif kind == "final":
            replies.append(reply_text(json.dumps(step["content"]), model="mock-model"))
        elif kind == "text":
            replies.append(reply_text(step["content"], model="mock-model"))
        else:
            raise ValueError(f"{name}: unknown reply type {kind!r}")
    return FakeModelClient(replies, reported_model="mock-model")

def synthetic_models():
    plan = synthetic_plan_draft("S1", "V1")
    planner = FakeModelClient([reply_text(json.dumps({"status":"completed","payload":plan})) for _ in range(12)])
    return ctl.Models(coordinator=planner, retriever=load_model_script("valid_request.json"),
                      analyst_factory=lambda prefix: load_analyst_script("valid_ab.json", prefix=prefix),
                      requested="synthetic-test-model")

def context(tmp_path, run_id="run_shared"):
    cache_dir=tmp_path / "cache"
    seed_synthetic(CacheStore(cache_dir))
    ctx=build_context(offline=True,settings=Settings(base_url=SYNTHETIC_BASE,mode="offline",cache_dir=cache_dir),
                      cache_dir=cache_dir,out_dir=tmp_path / "out",run_id=run_id)
    ctx.approval=synthetic_approval(run_id=run_id)
    return ctx

def hosted_config(tmp_path, question, **changes):
    import app_space
    seed_synthetic(CacheStore(tmp_path / "cache"))
    # The seeded replies carry a fixed fetch time (2026-06-01), which the app's catalog age limit (APP_CATALOG_MAX_AGE_HOURS,
    # default 24) would set aside; these tests are not about catalog age (tests/test_fresh_catalogs.py is).
    config = replace(app_space.space_config(question), cache_dir=tmp_path / "cache", out_dir=tmp_path / "out",
                     catalog_max_age_hours=None)
    return replace(config, **changes)

