"""MVP controller regressions: synthetic metadata, fake transport/models, no live traffic."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import run as ctl
from agents.coordinator import CoordinatorResult, KnownMetadata, KnownStudy
from brapi_client import Settings, TransportError, TransportResponse
from contracts import Claim, Plan, PlanStep, ResolvedScope
from part2_support import SYNTHETIC_BASE, async_returning, part2_config, part2_runtime, synthetic_context, synthetic_models


@pytest.fixture(autouse=True)
def _part2_flow(monkeypatch):
    """Ported from part2: run the release flow the way part2's mock mode did (see part2_support.part2_runtime)."""
    part2_runtime(monkeypatch, server=SYNTHETIC_BASE)



BASE = "https://live.invalid/brapi/v2"


def config(tmp_path, **over):
    values = dict(question="Which studies have yield observations?", offline=False, mock_model=False,
                  direct=False, auto=False, fixture="synthetic", cache_dir=tmp_path / "cache", out_dir=tmp_path / "out")
    values.update(over)
    return part2_config(**values)


def plan(studies=("S1",), variables=("V1",), units=False, explicit=False, catalog=False):
    inputs = {"study_db_id": studies[0], "variable_db_id": variables[0]} if explicit else {}
    steps = [PlanStep(step_id="step_1", agent="retriever", action="export_metadata" if catalog else "get_observations",
                      inputs={"entity": "studies"} if catalog else inputs)]
    if units:
        steps.append(PlanStep(step_id="step_2", agent="retriever", action="get_observation_units"))
    steps.append(PlanStep(step_id="step_3", agent="analyst", action="count_records", depends_on=[s.step_id for s in steps]))
    return Plan(plan_id="plan_mvp_test", question="Which studies have yield observations?", interpretation="Count the matching observations",
                status="ready", scope=ResolvedScope(base_url=BASE, study_ids=list(studies), variable_ids=list(variables)),
                steps=steps, created_at_utc=datetime.now(timezone.utc))


@pytest.mark.parametrize("value", [0, 601, float("nan"), float("inf"), -float("inf")])
def test_read_timeout_rejects_nonfinite_and_out_of_range(tmp_path, value):
    assert any("read_timeout must be finite" in problem for problem in config(tmp_path, read_timeout=value).validate())


@pytest.mark.parametrize("value", [59, 3601, float("nan"), float("inf")])
def test_work_budget_rejects_nonfinite_and_out_of_range(tmp_path, value):
    assert any("max_elapsed_seconds must be finite" in problem for problem in config(tmp_path, max_elapsed_seconds=value).validate())


@pytest.mark.parametrize("value", [1, 90, 600])
def test_read_timeout_valid_bounds(tmp_path, value):
    assert config(tmp_path, read_timeout=value).validate() == []


def test_live_timeout_and_work_budget_reach_the_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(ctl, "load_settings", lambda: Settings(base_url=BASE, mode="live", cache_dir=tmp_path / "cache"))
    controller = ctl.Controller(config(tmp_path, offline=False, fixture=None, auto=False, read_timeout=90, max_elapsed_seconds=600))
    ctx = controller._build_ctx(tmp_path / "out")
    assert ctx.client.settings.read_timeout == 90
    assert ctx.client.settings.operation_deadline >= 180
    assert controller.budget.max_elapsed_seconds == 600
    assert controller._build_ctx(tmp_path / "metadata", cache_only=True).client.settings.mode == "offline"


def test_preflight_counts_every_scoped_pair_even_when_a_step_omits_studies():
    proposed = plan(studies=("S1", "S2", "S3"), variables=("V1", "V2"), explicit=True)
    pairs, units = ctl.required_fetches(proposed)
    assert pairs == {(s, v) for s in ("S1", "S2", "S3") for v in ("V1", "V2")}
    assert not units
    assert "at least 6 fetch calls" in ctl.plan_too_wide(proposed, 5)


def test_preflight_counts_required_unit_rosters_and_leaves_catalogs_alone():
    proposed = plan(studies=("S1", "S2", "S3"), units=True)
    assert "3 study/trait observation requests and 3 study plot rosters" in ctl.plan_too_wide(proposed, 5)
    assert ctl.plan_too_wide(plan(studies=("S1", "S2"), units=True), 5) is None
    assert ctl.plan_too_wide(plan(studies=tuple(f"S{i}" for i in range(100)), variables=(), catalog=True), 5) is None
    assert ctl.plan_too_wide(plan(studies=(), variables=()), 5) is None
    assert ctl.plan_too_wide(plan(studies=("S1",), variables=()), 5) is None


@pytest.fixture
def live(tmp_path, monkeypatch):
    import brapi_client
    from part2_support import load_fixture

    monkeypatch.setattr(ctl, "load_settings", lambda: Settings(base_url=BASE, mode="live", cache_dir=tmp_path / "cache"))
    notes = tmp_path / "reviewed.txt"
    notes.write_text("Reviewed by Bidhan on: 2026-09-28\n", encoding="utf-8")
    calls = []
    failure = {"endpoint": None}
    replies = {"/studies": "synthetic_studies.json", "/variables": "synthetic_variables.json"}

    def transport(req):
        endpoint = req.url[len(BASE):]
        calls.append((endpoint, dict(req.params), req.read_timeout))
        if endpoint == failure["endpoint"]:
            raise TransportError("timeout", "synthetic transport timeout", retryable=False)
        _, body = load_fixture(replies[endpoint])
        return TransportResponse(200, {"Content-Type": "application/json"}, body)

    monkeypatch.setattr(brapi_client, "RequestsTransport", lambda: transport)
    cfg = config(tmp_path, offline=False, fixture=None, auto=False, access_notes=notes, bootstrap_catalogs=True,
                 read_timeout=90, max_elapsed_seconds=600)
    return cfg, calls, failure


def test_live_too_wide_asks_to_narrow_before_approval_or_dispatch(live, monkeypatch):
    cfg, calls, _ = live
    proposed = plan(studies=("S1", "S2", "S3"), variables=("V1", "V2"))
    metadata = KnownMetadata(base_url=BASE, complete=True, studies=[KnownStudy(study_id=s) for s in proposed.scope.study_ids])
    monkeypatch.setattr(ctl.Controller, "_load_metadata_mcp", async_returning(metadata))

    async def draft(*args, **kwargs):
        return CoordinatorResult(status="completed", plan=proposed)

    monkeypatch.setattr(ctl, "draft_plan", draft)
    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        return ""

    controller = ctl.Controller(cfg, answer_fn=answer)
    result = asyncio.run(controller.run())
    assert result.execution_status == "canceled"
    assert prompts == [ctl.CLARIFY_PROMPT]
    assert controller.approval is None and controller.manifest is None and not calls
    assert "at least 6 fetch calls" in controller.coordination.questions[0]


def test_scope_edit_narrows_plan_manifest_and_retriever_instructions(tmp_path):
    controller = ctl.Controller(config(tmp_path, auto=False))
    answers = iter(["edit", "S1", "approve"])
    controller.answer_fn = lambda prompt: next(answers)
    controller.plan = plan(studies=("S1", "S2"), units=True)
    controller.metadata = KnownMetadata(base_url=BASE, complete=True,
                                         studies=[KnownStudy(study_id="S1"), KnownStudy(study_id="S2")])
    controller.manifest = controller.resolve_manifest()
    assert controller.approve_phase()
    assert controller.plan.scope.study_ids == controller.manifest.observation_study_ids == ["S1"]
    assert controller.plan.scope.matching_study_count == 1
    assert ctl.required_fetches(controller.plan, controller.manifest) == ({("S1", "V1")}, {"S1"})
    assert "Authoritative approved observation scope: studies ['S1']" in controller._retriever_request()
    assert "studies ['S1', 'S2']" not in controller._retriever_request()
    assert controller.approval.observation_study_ids == ["S1"]


def test_scope_edit_rejects_expansion_and_removes_explicit_removed_steps():
    proposed = plan(studies=("S1", "S2"), units=True)
    with pytest.raises(ValueError, match="only narrow"):
        ctl.narrowed_plan(proposed, ["S1", "S3"])
    data = proposed.model_dump()
    data["steps"][0]["inputs"] = {"study_db_id": "S2", "variable_db_id": "V1"}
    data["steps"].insert(1, {"step_id": "step_extra", "agent": "retriever", "action": "get_observations",
                             "inputs": {"study_db_id": "S1", "variable_db_id": "V1"}})
    data["steps"][-1]["depends_on"].append("step_extra")
    narrowed = ctl.narrowed_plan(Plan.model_validate(data), ["S1"])
    assert all(step.inputs.get("study_db_id") != "S2" for step in narrowed.steps)
    assert all("step_1" not in step.depends_on for step in narrowed.steps)


def test_scope_edit_blocks_nested_stale_ids():
    proposed = plan(studies=("S1", "S2"))
    data = proposed.model_dump()
    data["steps"][-1]["inputs"] = {"study_ids": ["S1", "S2"]}
    with pytest.raises(ValueError, match="must be replanned"):
        ctl.narrowed_plan(Plan.model_validate(data), ["S1"])


def test_controller_blocks_analysis_when_model_omits_requested_study(tmp_path, monkeypatch):
    proposed = plan(studies=("S1", "S2"), units=True)

    async def draft(*args, **kwargs):
        return CoordinatorResult(status="completed", plan=proposed)

    monkeypatch.setattr(ctl, "draft_plan", draft)
    controller = ctl.Controller(config(tmp_path), models=synthetic_models())
    result = asyncio.run(controller.run())
    assert result.execution_status == "incomplete"
    assert result.analysis is None
    assert result.retrieval.status == "incomplete"
    assert "S2" in " ".join(result.retrieval.warnings)


def test_count_only_answer_lists_ids_with_names_from_its_fingerprinted_table(tmp_path):
    import analyst_tools as at
    import brapi_mcp_server as srv

    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_mvp_ids")
    artifact = srv.dispatch(ctx, "export_metadata", {"entity": "studies"}).artifact_ids[0]
    result = at.tool_count_records(ctx.registry, at.CountRecordsArgs(artifact_id=artifact, distinct_by="studyDbId"))
    claims = [Claim.model_validate(c) for c in result.data["claims"]]
    assert "  - S1 · SYNTHETIC yield trial S1 · SYNTHETIC location one · 2026" in ctl.readable_answer(claims, ctx.registry)


def test_distinct_list_is_numeric_sorted_capped_and_group_labels_use_each_claim_source():
    def claim(hash_value, key=None):
        return SimpleNamespace(kind="count", grouping=["studyDbId"], filters={} if key is None else {"studyDbId": key},
                               transformation="number of distinct non-empty exact text values", value=32 if key is None else 1,
                               missing_reason=None, unit=None, denominator=SimpleNamespace(name="rows", n=32), source_artifact_hashes=[hash_value])

    tables = {"a": [{"studyDbId": str(i), "studyName": f"Trial {i}"} for i in reversed(range(1, 33))],
              "b": [{"studyDbId": "2", "studyName": "Other source name"}]}
    registry = SimpleNamespace(artifact_ids=lambda: list(tables), manifest=lambda aid: SimpleNamespace(sha256=aid),
                               load=lambda aid: (None, tables[aid]))
    lines = ctl.readable_answer([claim("a")], registry)
    assert lines[1:3] == ["  - 1 · Trial 1", "  - 2 · Trial 2"]
    assert "  - 30 · Trial 30" in lines and "  - 31 · Trial 31" not in lines
    assert lines[-1] == "  - … and 2 more in the source table"
    lines = ctl.readable_answer([claim("a", "1"), claim("b", "2")], registry)
    assert "  - 1 · Trial 1" in lines and "  - 2 · Other source name" in lines


@pytest.mark.parametrize("answer", ["cancel", "approve", ""])
def test_bootstrap_requires_its_distinct_approval_and_records_cancellation(live, answer):
    cfg, calls, _ = live
    prompts = []

    def reply(prompt):
        prompts.append(prompt)
        assert not calls
        return answer

    controller = ctl.Controller(cfg, answer_fn=reply)
    result = asyncio.run(controller.run())
    saved = json.loads((controller.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert prompts == [ctl.CATALOG_PROMPT] and result.execution_status == "canceled"
    assert result.counts.model_calls == 0 and not calls
    assert saved["catalog_setup"]["approval"] is None
    assert saved["catalog_setup"]["status"] == "canceled"


def test_bootstrap_records_separate_approval_and_requests_then_requires_observation_approval(live):
    cfg, calls, _ = live
    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        if prompt == ctl.CATALOG_PROMPT:
            assert not calls
            return "approve catalog"
        assert prompt == "approve / edit / cancel: "
        assert [call[0] for call in calls] == ["/studies", "/variables"]
        return "cancel"

    # A named study avoids a separate planner clarification unrelated to bootstrap.
    cfg = ctl.replace(cfg, question="For synthetic study S1, give mean yield and clone means with n.")
    controller = ctl.Controller(cfg, answer_fn=answer)
    result = asyncio.run(controller.run())
    saved = json.loads((controller.run_dir / "manifest.json").read_text(encoding="utf-8"))
    setup = saved["catalog_setup"]
    assert result.execution_status == "canceled" and controller.approval is None
    assert prompts == [ctl.CATALOG_PROMPT, "approve / edit / cancel: "]
    assert setup["status"] == "completed"
    assert setup["manifest"]["endpoint_families"] == ["studies", "observationvariables"]
    assert setup["approval"]["max_http_attempts"] == ctl.CATALOG_MAX_HTTP_ATTEMPTS
    assert setup["approval"]["observation_study_ids"] == []
    assert result.counts.live_attempts == 2 and all(call[2] == 90 for call in calls)
    assert result.catalog_approval is not None and result.catalog_manifest is not None
    assert result.approved_catalog_manifest_sha256 == result.catalog_manifest.sha256()
    assert result.approval_id is None and len(result.catalog_requests) == 2
    assert ctl.RunResult.model_validate_json((controller.run_dir / "answer.json").read_text(encoding="utf-8")) == result
    assert len(saved["request_ids"]) == len(set(saved["request_ids"])) == 2


def test_bootstrap_failure_keeps_approval_and_attempt_evidence_and_never_calls_model(live):
    cfg, calls, failure = live
    failure["endpoint"] = "/variables"
    controller = ctl.Controller(cfg, answer_fn=lambda prompt: "approve catalog")
    result = asyncio.run(controller.run())
    saved = json.loads((controller.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert result.execution_status == "incomplete" and result.counts.model_calls == 0
    assert controller.metadata is None and controller.approval is None
    assert saved["catalog_setup"]["approval"] is not None
    assert saved["catalog_setup"]["status"] == "incomplete"
    assert [call[0] for call in calls] == ["/studies", "/variables"]
    assert saved["requests"][-1]["error"]["code"] == "timeout"


def catalog_result_fields():
    from contracts import FetchManifest, RequestRecord, RunCounts

    now = datetime.now(timezone.utc)
    manifest = FetchManifest(run_id="run_catalogcontract_catalog", base_url=BASE, origin="live",
                             endpoint_families=["studies", "observationvariables"], max_http_attempts=20, max_observation_studies=0)
    approval = ctl.build_approval(manifest, issued_by_human=True, now=now)
    record = RequestRecord(request_id="req_catalog_1", run_id=manifest.run_id, base_url=BASE, endpoint="/studies",
                           requested_at_utc=now, origin="live", attempt=1, duration_seconds=1,
                           source_url_redacted=BASE + "/studies", error={"code": "timeout", "message": "synthetic timeout"})
    return dict(run_id="run_catalogcontract", plan=plan(), execution_status="canceled", started_at_utc=now,
                counts=RunCounts(live_attempts=1), catalog_manifest=manifest, catalog_approval=approval,
                approved_catalog_manifest_sha256=manifest.sha256(), catalog_requests=[record])


def test_catalog_only_result_retains_total_attempts_without_observation_approval():
    result = ctl.RunResult(**catalog_result_fields())
    assert result.counts.live_attempts == 1 and result.approval_id is None
    assert result.catalog_approval is not None and len(result.catalog_requests) == 1


@pytest.mark.parametrize("fault, expected", [
    ("missing_approval", "no catalog approval"),
    ("missing_manifest", "without a catalog manifest"),
    ("bad_hash", "catalog manifest changed"),
    ("wrong_approval_run", "catalog approval does not match"),
    ("wider_approval", "catalog approval does not match"),
    ("wider_manifest_and_approval", "metadata-only"),
    ("observation_request", "metadata-only approved endpoint"),
    ("wrong_request_server", "request run_id or server differs"),
    ("unrecorded_live_attempt", "no approval is recorded for non-catalog"),
    ("missing_request_receipts", "no approval is recorded for non-catalog"),
    ("understated_total", "exceed the run's total"),
    ("duplicate_receipts", "duplicates"),
    ("catalog_budget_exceeded", "approved HTTP attempt budget"),
    ("approval_expired", "approval validity period"),
])
def test_catalog_contract_rejects_missing_mismatched_or_widened_authority(fault, expected):
    fields = catalog_result_fields()
    if fault == "missing_approval":
        fields.update(catalog_approval=None, approved_catalog_manifest_sha256=None)
    elif fault == "missing_manifest":
        fields["catalog_manifest"] = None
    elif fault == "bad_hash":
        fields["approved_catalog_manifest_sha256"] = "0" * 64
    elif fault == "wrong_approval_run":
        fields["catalog_approval"] = fields["catalog_approval"].model_copy(update={"run_id": "run_other_catalog"})
    elif fault == "wider_approval":
        fields["catalog_approval"] = fields["catalog_approval"].model_copy(update={"endpoint_families": ["studies", "observationvariables", "locations"]})
    elif fault == "wider_manifest_and_approval":
        manifest = fields["catalog_manifest"].model_copy(update={"endpoint_families": ["studies", "observations"],
            "observation_study_ids": ["S1"], "observation_variable_ids": ["V1"], "max_observation_studies": 1})
        fields.update(catalog_manifest=manifest, catalog_approval=ctl.build_approval(manifest, issued_by_human=True),
                      approved_catalog_manifest_sha256=manifest.sha256())
    elif fault == "observation_request":
        fields["catalog_requests"] = [fields["catalog_requests"][0].model_copy(update={"endpoint": "/observations"})]
    elif fault == "wrong_request_server":
        fields["catalog_requests"] = [fields["catalog_requests"][0].model_copy(update={"base_url": "https://other.invalid/brapi/v2"})]
    elif fault == "unrecorded_live_attempt":
        fields["counts"] = fields["counts"].model_copy(update={"live_attempts": 2})
    elif fault == "missing_request_receipts":
        fields["catalog_requests"] = []
    elif fault == "understated_total":
        fields["counts"] = fields["counts"].model_copy(update={"live_attempts": 0})
    elif fault == "duplicate_receipts":
        fields["catalog_requests"] *= 2
        fields["counts"] = fields["counts"].model_copy(update={"live_attempts": 2})
    elif fault == "catalog_budget_exceeded":
        manifest = fields["catalog_manifest"].model_copy(update={"max_http_attempts": 1})
        fields.update(catalog_manifest=manifest, catalog_approval=ctl.build_approval(manifest, issued_by_human=True),
                      approved_catalog_manifest_sha256=manifest.sha256())
        fields["catalog_requests"].append(fields["catalog_requests"][0].model_copy(update={"request_id": "req_catalog_2"}))
        fields["counts"] = fields["counts"].model_copy(update={"live_attempts": 2})
    elif fault == "approval_expired":
        from datetime import timedelta

        record = fields["catalog_requests"][0]
        fields["catalog_requests"] = [record.model_copy(update={"requested_at_utc": record.requested_at_utc + timedelta(hours=2)})]
    with pytest.raises(ValueError, match=expected):
        ctl.RunResult(**fields)
