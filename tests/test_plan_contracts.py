"""
tests/test_plan_contracts.py — Stage 5 contracts: a Plan that can only name allowed agents, only depend on
earlier steps and only carry typed inputs; a FetchManifest whose hash an approval binds to; a RunResult that
links plan, manifest, approval, retrieval, analysis and artifacts, with execution and acceptance kept apart.

All values are SYNTHETIC. No network, no model, no files.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

import contracts as c

NOW = datetime(2026, 9, 27, 12, 0, 0, tzinfo=timezone.utc)
BASE = "https://synthetic.invalid/brapi/v2"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


# --- builders (each returns a VALID record; tests mutate one thing at a time) --------------------

def step(step_id="step_1", agent="retriever", action="get_observations", depends_on=(), **inputs) -> dict:
    return {"step_id": step_id, "agent": agent, "action": action, "inputs": inputs, "depends_on": list(depends_on)}


def scope(**overrides) -> dict:
    base = {"base_url": BASE, "study_ids": ["S1"], "variable_ids": ["V1"]}
    base.update(overrides)
    return base


def plan(**overrides) -> dict:
    base = {
        "plan_id": "plan_synth_001", "question": "Mean fresh root yield per clone in synthetic study S1?",
        "interpretation": "pooled and per-clone descriptive means of V1 in S1", "status": "ready", "scope": scope(),
        "statistic": {"kind": "mean", "denominator": "valid plot-level values", "grouping": ["germplasmDbId"]},
        "steps": [step("step_1", "retriever", "get_observations", study_db_id="S1", variable_db_id="V1"),
                  step("step_2", "analyst", "group_stats", ("step_1",), by="germplasmDbId")],
        "created_at_utc": NOW,
    }
    base.update(overrides)
    return base


def manifest(**overrides) -> c.FetchManifest:
    base = {"run_id": "run_5a", "base_url": BASE, "origin": "live", "endpoint_families": ["studies", "observations"],
            "observation_study_ids": ["S1"], "observation_variable_ids": ["V1"], "max_http_attempts": 20, "max_observation_studies": 1}
    base.update(overrides)
    return c.FetchManifest(**base)


def approval_for(m: c.FetchManifest, **overrides) -> c.FetchApproval:
    base = {"approval_id": "appr_synth_1", "run_id": m.run_id, "base_url": m.base_url, "endpoint_families": list(m.endpoint_families),
            "observation_study_ids": list(m.observation_study_ids), "observation_variable_ids": list(m.observation_variable_ids),
            "max_http_attempts": m.max_http_attempts, "max_observation_studies": m.max_observation_studies,
            "issued_at_utc": NOW, "expires_at_utc": NOW + timedelta(hours=1)}
    base.update(overrides)
    return c.FetchApproval(**base)


def artifact(artifact_id="art_abc123_0001", sha=SHA_A, kind="observations", complete=True) -> c.ArtifactManifest:
    return c.ArtifactManifest(artifact_id=artifact_id, run_id="run_5a", kind=kind, relative_path=f"run_5a/artifacts/{artifact_id}.csv",
                              sha256=sha, schema_version="1", row_count=6, complete=complete, study_ids=["S1"], variable_ids=["V1"],
                              created_at_utc=NOW)


def retrieval(art: c.ArtifactManifest, status="completed") -> c.RetrievalReport:
    return c.RetrievalReport(status=status, artifacts=[art], complete=status == "completed",
                             study_ledger=[c.StudyStatusEntry(study_id="S1", status="complete", artifact_id=art.artifact_id)])


def claim(claim_id="clm_abc123_0001_mean", kind="mean", value=18.0, hashes=(SHA_A,), unit=None) -> c.Claim:
    return c.Claim(claim_id=claim_id, kind=kind, value=value, unit=unit, denominator=c.Denominator(name="valid plot-level values", n=4),
                   n_independent_units=4, source_artifact_hashes=list(hashes), transformation="pooled mean of valid values")


def analysis(*claims: c.Claim, status="completed", result_ids=()) -> c.AnalysisReport:
    return c.AnalysisReport(status=status, claims=list(claims), methods=["numeric_summary"], result_artifact_ids=list(result_ids))


def run_result(**overrides) -> c.RunResult:
    art = artifact()
    base = {"run_id": "run_5a", "plan": c.Plan(**plan()), "retrieval": retrieval(art), "analysis": analysis(claim()), "artifacts": [art],
            "execution_status": "completed", "started_at_utc": NOW, "finished_at_utc": NOW + timedelta(seconds=30)}
    base.update(overrides)
    return c.RunResult(**base)


def rejects(build, *fragments: str) -> None:
    with pytest.raises(ValidationError) as info:
        build()
    text = str(info.value)
    for fragment in fragments:
        assert fragment in text, f"{fragment!r} not in:\n{text}"


# --- 1. Plan: shape, agents, steps -------------------------------------------------------------

def test_valid_plan_round_trips_and_is_frozen():
    p = c.Plan(**plan())
    assert p.plan_schema == "plan.v1" and p.status == "ready" and [s.agent for s in p.steps] == ["retriever", "analyst"]
    assert c.Plan.model_validate_json(c.dump_json(p)) == p
    with pytest.raises(ValidationError):
        p.status = "draft"          # frozen


@pytest.mark.parametrize("agent", ["coordinator", "shell", "controller", "Retriever", ""])
def test_only_retriever_and_analyst_may_appear_in_a_plan(agent):
    rejects(lambda: c.Plan(**plan(steps=[step("step_1", agent, "anything")])), "agent")


def test_wrong_types_are_rejected_not_converted():
    rejects(lambda: c.Plan(**plan(steps=json.dumps(plan()["steps"]))), "steps")                    # a JSON string is not a list
    rejects(lambda: c.Plan(**plan(question=42)), "question")
    rejects(lambda: c.PlanStep(step_id=1, agent="retriever", action="x"), "step_id")
    rejects(lambda: c.PlanStep(step_id="step_1", agent="retriever", action="x", inputs={"study": {"id": "S1"}}), "inputs")   # nested object is not a typed input
    rejects(lambda: c.PlanStep(step_id="step_1", agent="retriever", action="x", inputs={"n": 2.5}), "inputs")
    rejects(lambda: c.Plan(**plan(status="done")), "status")
    assert c.PlanStep(step_id="step_1", agent="retriever", action="x", inputs={"limit": "5"}).inputs["limit"] == "5"   # stays text, not converted


def test_a_string_that_parses_as_json_is_not_an_input():
    blob = json.dumps({"study_db_id": "S1", "approval": {"max_http_attempts": 99}})
    rejects(lambda: c.PlanStep(step_id="step_1", agent="retriever", action="get_observations", inputs={"args": blob}), "JSON-looking string")
    rejects(lambda: c.PlanStep(step_id="step_1", agent="retriever", action="x", inputs={"ids": ['["S1","S2"]']}), "JSON-looking string")
    rejects(lambda: c.Plan(**plan(interpretation='{"status": "approved"}')), "JSON-looking string")
    ok = c.PlanStep(step_id="step_1", agent="retriever", action="x", inputs={"name_contains": "yield [2026]", "ids": ["S1", "S2"], "limit": 5, "paged": False})
    assert ok.inputs["ids"] == ["S1", "S2"]                        # plain text with brackets inside is fine


def test_negative_or_zero_budgets_are_rejected():
    rejects(lambda: c.PlanStep(step_id="step_1", agent="retriever", action="x", max_tool_calls=0), "max_tool_calls")
    rejects(lambda: c.PlanStep(step_id="step_1", agent="retriever", action="x", max_tool_calls=-3), "max_tool_calls")
    rejects(lambda: c.PlanStep(step_id="step_1", agent="retriever", action="x", max_tool_calls=31), "max_tool_calls")
    rejects(lambda: c.Statistic(kind="mean", denominator="valid values", min_independent_n=0), "min_independent_n")
    rejects(lambda: manifest(max_http_attempts=0), "max_http_attempts")
    rejects(lambda: manifest(max_observation_studies=-1), "max_observation_studies")
    rejects(lambda: c.RunCounts(live_attempts=-1), "live_attempts")
    rejects(lambda: c.ResolvedScope(base_url=BASE, matching_study_count=-1), "matching_study_count")


def test_step_dependencies_must_be_earlier_steps():
    rejects(lambda: c.Plan(**plan(steps=[step("step_1", depends_on=("step_1",))])), "depends on itself")
    rejects(lambda: c.Plan(**plan(steps=[step("step_1", depends_on=("step_2",)), step("step_2", "analyst", "group_stats", ("step_1",))])), "later step")
    rejects(lambda: c.Plan(**plan(steps=[step("step_1", depends_on=("step_9",))])), "unknown step")
    cyclic = [step("step_1", depends_on=("step_2",)), step("step_2", depends_on=("step_1",))]
    rejects(lambda: c.Plan(**plan(steps=cyclic)), "later step")                                    # a cycle needs a forward edge, which is refused
    rejects(lambda: c.Plan(**plan(steps=[step("step_1"), step("step_1")])), "duplicates")
    rejects(lambda: c.PlanStep(step_id="step_2", agent="analyst", action="x", depends_on=["step_1", "step_1"]), "duplicates")


def test_plans_are_bounded_and_analysts_need_evidence():
    too_many = [step(f"step_{i}") for i in range(1, c.MAX_PLAN_STEPS + 2)]
    # The contract's absolute cap (30 since 2026-10-22: the benchmark may pass up to 30 plan steps). Each run's own limit, 12 in
    # the app, is checked when a plan is assembled: tests/test_benchmark_budgets.py test_a_plan_longer_than_the_runs_limit_is_refused.
    rejects(lambda: c.Plan(**plan(steps=too_many)), f"at most {c.MAX_PLAN_STEPS}")
    rejects(lambda: c.Plan(**plan(steps=[])), "at least 1")
    rejects(lambda: c.Plan(**plan(steps=[step("step_1", "analyst", "numeric_summary")])), "must depend on at least one retriever step")
    # a chain of analyst steps traces back to the retriever step it started from (export -> filter -> count): accepted since 2026-09-29;
    # on real questions gpt-4.1 wrote exactly this chain and the old rule (a DIRECT retriever dependency) rejected it three times a run
    chain = [step("step_1"), step("step_2", "analyst", "numeric_summary", ("step_1",)), step("step_3", "analyst", "group_stats", ("step_2",))]
    assert [s.step_id for s in c.Plan(**plan(steps=chain)).steps] == ["step_1", "step_2", "step_3"]
    # but a chain that never reaches a retriever step is still thin air
    orphan = [step("step_1"), step("step_2", "analyst", "numeric_summary"), step("step_3", "analyst", "group_stats", ("step_2",))]
    rejects(lambda: c.Plan(**plan(steps=orphan)), "step_2 must depend on at least one retriever step")


def test_clarifications_drive_the_plan_status():
    asked = [{"question": "Which trait: fresh root yield or dry matter?"}]
    rejects(lambda: c.Plan(**plan(status="ready", clarifications=asked)), "status must be needs_clarification")
    waiting = c.Plan(**plan(status="needs_clarification", clarifications=asked))
    assert waiting.clarifications[0].answer is None
    answered = c.Plan(**plan(status="ready", clarifications=[{**asked[0], "answer": "fresh root yield"}]))
    assert answered.status == "ready"
    optional = c.Plan(**plan(status="ready", clarifications=[{"question": "Any season preference?", "required": False}]))
    assert optional.status == "ready"


def test_scope_says_unknown_instead_of_estimating():
    open_scope = c.ResolvedScope(base_url=BASE, discovery_needed=True, discovery_families=["studies", "locations"])
    assert open_scope.matching_study_count is None and open_scope.study_ids == []
    rejects(lambda: c.ResolvedScope(base_url=BASE, discovery_needed=True), "requires the endpoint families")
    rejects(lambda: c.ResolvedScope(base_url=BASE, discovery_families=["studies"]), "although discovery_needed is False")
    rejects(lambda: c.ResolvedScope(base_url=BASE, discovery_needed=True, discovery_families=["observations"]), "metadata only")
    rejects(lambda: c.ResolvedScope(base_url=BASE, discovery_needed=True, discovery_families=["studies"], matching_study_count=6), "cannot be known")
    rejects(lambda: c.Plan(**plan(status="ready", scope=scope(study_ids=[], discovery_needed=True, discovery_families=["studies"]),
                                  steps=[step("step_1", "analyst", "count_records")])), "retriever")


def test_injected_authority_and_old_shapes_are_rejected():
    rejects(lambda: c.PlanStep(step_id="step_1", agent="retriever", action="x", inputs={"approval": "granted"}), "authority field")
    rejects(lambda: c.PlanStep(step_id="step_1", agent="retriever", action="x", inputs={"max_http_attempts": 99}), "authority field")
    rejects(lambda: c.Plan(**plan(approved=True)), "extra")
    rejects(lambda: c.Plan(**plan(fetch_approval={"max_http_attempts": 99})), "extra")
    rejects(lambda: c.Plan(**plan(estimated_fetches=3)), "extra")                                # the original guide's guessed field
    rejects(lambda: c.Plan(**plan(plan_schema="plan.v0")), "plan_schema")
    rejects(lambda: c.RunResult(**{**run_result().model_dump(), "run_schema": "run_result.v0"}), "run_schema")
    rejects(lambda: c.AgentResult(status="completed", payload={"plan": {"approval_id": "appr_x"}}, model_requested="m", model_calls=1,
                                  tool_calls=0, elapsed_seconds=0.1), "authority field")          # FetchApproval can never arrive in model JSON


# --- 2. FetchManifest and approvals ----------------------------------------------------------------

def test_manifest_hash_is_stable_and_changes_with_scope():
    m = manifest()
    again = manifest()
    assert m.sha256() == again.sha256() and len(m.sha256()) == 64
    assert manifest(observation_study_ids=["S1", "S2"], max_observation_studies=2).sha256() != m.sha256()
    assert manifest(max_http_attempts=21).sha256() != m.sha256()
    assert c.FetchManifest.model_validate_json(c.dump_json(m)).sha256() == m.sha256()


def test_manifest_rules_mirror_the_approval_rules():
    rejects(lambda: manifest(origin="snapshot"), "snapshot_id")
    rejects(lambda: manifest(snapshot_id="snap_1"), "live manifest cannot name")
    rejects(lambda: manifest(observation_study_ids=["S1", "S2"]), "max_observation_studies")
    rejects(lambda: manifest(observation_study_ids=[]), "exact observation_study_ids")
    rejects(lambda: manifest(observation_variable_ids=[]), "exact observation_variable_ids")
    rejects(lambda: manifest(endpoint_families=["studies"]), "without an observation endpoint family")
    rejects(lambda: manifest(endpoint_families=["studies", "studies", "observations"]), "duplicates")
    snap = manifest(origin="snapshot", snapshot_id="snap_20260901")
    assert snap.origin == "snapshot"


def test_approval_must_match_the_manifest_exactly():
    m = manifest()
    assert c.approval_mismatches(m, approval_for(m)) == []
    changed = manifest(observation_study_ids=["S1", "S2"], max_observation_studies=2)          # scope changed after approval
    problems = c.approval_mismatches(changed, approval_for(m))
    assert any("observation_study_ids differ" in p for p in problems) and any("max_observation_studies differ" in p for p in problems)
    wider = approval_for(m, max_http_attempts=99)
    assert c.approval_mismatches(m, wider) == ["max_http_attempts differ: approval 99 vs manifest 20"]     # wider is also a mismatch
    assert any("base_url differs" in p for p in c.approval_mismatches(m, approval_for(m, base_url="https://other.invalid/brapi/v2")))
    assert any("run_id differs" in p for p in c.approval_mismatches(m, approval_for(m, run_id="run_other")))
    assert any("observation_variable_ids differ" in p for p in c.approval_mismatches(m, approval_for(m, observation_variable_ids=["V2"])))
    snap = manifest(origin="snapshot", snapshot_id="snap_1")
    assert c.approval_mismatches(snap, approval_for(m))[0].startswith("manifest origin is 'snapshot'")


# --- 3. RunResult: links and the two statuses --------------------------------------------------------

def test_valid_run_result_links_everything_and_leaves_acceptance_pending():
    r = run_result()
    assert r.execution_status == "completed" and r.human_acceptance == "pending"                  # evaluation: finished, not accepted
    assert r.analysis.claims[0].source_artifact_hashes == [r.artifacts[0].sha256]
    assert c.RunResult.model_validate_json(c.dump_json(r)) == r
    accepted = run_result(human_acceptance="accepted")
    assert accepted.human_acceptance == "accepted"
    rejected = run_result(execution_status="failed", analysis=None, human_acceptance="rejected")
    assert rejected.human_acceptance == "rejected"


def test_acceptance_needs_a_completed_execution():
    rejects(lambda: run_result(execution_status="incomplete", human_acceptance="accepted"), "only accept a run whose execution completed")
    rejects(lambda: run_result(human_acceptance="approved"), "human_acceptance")
    rejects(lambda: run_result(execution_status="completed", analysis=None), "requires a completed analysis")
    rejects(lambda: run_result(execution_status="completed", retrieval=None, analysis=None), "requires a completed retrieval")
    canceled = run_result(execution_status="canceled", retrieval=None, analysis=None, artifacts=[])
    assert canceled.execution_status == "canceled" and canceled.human_acceptance == "pending"


def test_claims_must_cite_real_run_artifacts():
    rejects(lambda: run_result(analysis=analysis(claim(hashes=(SHA_B,)))), "cites artifact hashes that no run artifact has")
    rejects(lambda: run_result(analysis=analysis(claim(), result_ids=["art_abc123_0099"])), "not among the run's artifacts")
    other = artifact(sha=SHA_C)                                                                  # same id, different bytes
    rejects(lambda: run_result(artifacts=[other]), "different hash")
    rejects(lambda: run_result(retrieval=None), "an analysis without a retrieval has no evidence")
    rejects(lambda: run_result(artifacts=[artifact(), artifact()]), "duplicates")


def test_count_claim_from_metadata_is_a_legitimate_claim():
    studies = artifact(artifact_id="art_abc123_0002", sha=SHA_B, kind="studies")
    count = c.Claim(claim_id="clm_abc123_0002_count_rows", kind="count", value=1, denominator=c.Denominator(name="rows in the complete studies table", n=1),
                    source_artifact_hashes=[SHA_B], transformation="row count of a complete table")
    r = run_result(retrieval=retrieval(studies), analysis=analysis(count), artifacts=[studies])
    assert r.analysis.claims[0].kind == "count" and r.analysis.claims[0].unit is None and r.analysis.claims[0].n_independent_units is None


def test_manifest_and_approval_links_in_a_run():
    m = manifest()
    r = run_result(manifest=m, approval_id="appr_synth_1", approved_manifest_sha256=m.sha256(), counts=c.RunCounts(live_attempts=3, distinct_studies_fetched=1))
    assert r.approved_manifest_sha256 == m.sha256()
    changed = manifest(max_http_attempts=21)
    rejects(lambda: run_result(manifest=changed, approval_id="appr_synth_1", approved_manifest_sha256=m.sha256()), "manifest changed after approval")
    rejects(lambda: run_result(manifest=m, approval_id="appr_synth_1"), "must be given together")
    rejects(lambda: run_result(approval_id="appr_synth_1", approved_manifest_sha256=m.sha256()), "no manifest")
    rejects(lambda: run_result(counts=c.RunCounts(live_attempts=1)), "live attempts were counted but no approval")
    rejects(lambda: run_result(manifest=manifest(run_id="run_other")), "manifest.run_id differs")
    rejects(lambda: run_result(finished_at_utc=NOW - timedelta(seconds=1)), "before started_at_utc")
    rejects(lambda: run_result(started_at_utc=datetime(2026, 9, 27, 12, 0, 0)), "UTC")


def test_existing_report_shapes_are_unchanged():
    art = artifact()
    assert retrieval(art).status == "completed" and analysis(claim()).status == "completed"
    rejects(lambda: c.RetrievalReport(status="completed", complete=True, human_acceptance="accepted"), "extra")
    rejects(lambda: c.AnalysisReport(status="completed", claims=[claim()], execution_status="completed"), "extra")
    rejects(lambda: c.Claim(**{**claim().model_dump(), "plan_id": "plan_x"}), "extra")
