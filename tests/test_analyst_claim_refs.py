"""Offline synthetic checks for short model references and full canonical evidence."""
from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import anyio
import pytest

import agents.analyst as an
from agents.coordinator import render_numeric_table
from analyst_tools import AnalystTools
from artifacts import ArtifactRegistry
from contracts import AgentResult
from llm import Budget, FakeModelClient, compact_tool_result, reply_text, reply_tools


def catalog(tmp_path, count=8):
    registry = ArtifactRegistry(tmp_path / "out", "synthetic_claim_refs", base_url="https://synthetic.invalid/brapi/v2")
    names = [f"SYNTHETIC trait {i:04d}: leaf pigmentation estimating 1-9 with exact label" for i in range(count)]
    artifact = registry.save_table(
        [{"observationVariableDbId": f"V{i:04d}", "traitName": name} for i, name in enumerate(names)],
        kind="variables", columns={"observationVariableDbId": "text", "traitName": "text"}, complete=True,
    ).artifact_id
    inputs = an.AnalysisInput(question="List all traits.", artifact_ids=[artifact])
    return registry, artifact, inputs, names


def counted(registry, artifact):
    tools = an.GuardedAnalystTools(AnalystTools(registry), [artifact])

    async def go():
        await tools.call("table_info", {"artifact_id": artifact})
        return await tools.call("count_records", {"artifact_id": artifact, "group_by": "traitName", "distinct_by": "traitName"})

    return tools, anyio.run(go)


def completed(keys):
    return AgentResult(status="completed", payload={"key_claim_ids": keys}, model_requested="synthetic-model",
                       model_calls=1, tool_calls=2, elapsed_seconds=0.0)


def test_saved_486_trait_shape_keeps_every_name_and_short_exact_model_references(tmp_path):
    registry, artifact, inputs, names = catalog(tmp_path, 486)
    model = FakeModelClient([
        reply_tools([("table_info", json.dumps({"artifact_id": artifact}))]),
        reply_tools([("count_records", json.dumps({"artifact_id": artifact, "group_by": "traitName", "distinct_by": "traitName"}))]),
        reply_text(json.dumps({"status": "completed", "payload": {"key_claim_ids": ["clm_ref_000001", "clm_ref_000005"]}})),
    ])

    async def go():
        return await an.run_analyst(inputs, model=model, ctx=SimpleNamespace(registry=registry), budget=Budget(),
                                    run_id=registry.run_id, log_dir=registry.out_root)

    report, agent, _ = anyio.run(go)
    assert report.status == "completed" and agent.status == "completed"
    assert len(report.claims) == 487
    assert {c.filters["traitName"] for c in report.claims if "traitName" in c.filters} == set(names)
    assert all(not c.claim_id.startswith("clm_ref_") for c in report.claims)
    assert any(c.value == 486 and not c.filters for c in report.claims)
    count_message = next(json.loads(m.content) for m in agent.messages if m.role == "tool" and m.name == "count_records")
    assert count_message["data"]["claim_count"] == 487
    shown = count_message["data"]["claims"]
    assert [c["claim_id"] for c in shown[:-1]] == [f"clm_ref_{i:06d}" for i in range(1, 6)]
    assert shown[-1] == {"omitted_items": 482}
    assert "all original tool claims are retained" in count_message["data"]["claim_selection_note"]
    assert len(json.dumps(count_message)) < 6000
    full_table = render_numeric_table(report)
    for name in names:
        assert full_table.count(f"traitName={name} |") == 1


def test_repeated_claims_reuse_references_without_mutating_recorded_evidence(tmp_path):
    registry, artifact, inputs, _ = catalog(tmp_path)
    tools, first = counted(registry, artifact)
    original = deepcopy(tools.records[-1].result.model_dump())
    second = anyio.run(tools.call, "count_records", {"artifact_id": artifact, "group_by": "traitName", "distinct_by": "traitName"})
    assert first.data["claims"] == second.data["claims"]
    assert len(tools.claim_refs) == 9 and len(set(tools.claim_refs.values())) == 9
    assert set(tools.claim_refs.values()) == {c["claim_id"] for c in original["data"]["claims"]}
    first.data["claims"][0]["filters"]["traitName"] = "changed only in the returned copy"
    assert tools.records[1].result.model_dump() == original
    report, _ = an.build_report(completed([second.data["claims"][0]["claim_id"]]), tools, inputs)
    assert report.status == "completed" and len(report.claims) == 9


@pytest.mark.parametrize("bad", ["clm_ref_999999", "clm_ref_000...", "clm_made_up_0001_mean"])
def test_unknown_and_abbreviated_references_still_fail(tmp_path, bad):
    registry, artifact, inputs, _ = catalog(tmp_path)
    tools, _ = counted(registry, artifact)
    report, narrative = an.build_report(completed([bad]), tools, inputs)
    assert report.status == "failed" and report.claims == [] and narrative == ""
    assert any(f"no tool produced: ['{bad}']" in c for c in report.caveats)


def test_abbreviated_original_id_still_fails_and_exact_original_id_remains_compatible(tmp_path):
    registry, artifact, inputs, _ = catalog(tmp_path)
    tools, _ = counted(registry, artifact)
    canonical = tools.records[-1].result.data["claims"][0]["claim_id"]
    bad, _ = an.build_report(completed([canonical[:60] + "..."]), tools, inputs)
    good, _ = an.build_report(completed([canonical]), tools, inputs)
    assert bad.status == "failed" and bad.claims == []
    assert good.status == "completed" and good.claims[0].claim_id == canonical


def test_references_are_local_to_the_guard_that_issued_them(tmp_path):
    registry, artifact, inputs, _ = catalog(tmp_path)
    tools, model_result = counted(registry, artifact)
    other = an.GuardedAnalystTools(AnalystTools(registry), [artifact])
    other.records = list(tools.records)  # original evidence, but no issued model reference in this guard
    assert other.claim_refs == {}
    report, _ = an.build_report(completed([model_result.data["claims"][0]["claim_id"]]), other, inputs)
    assert report.status == "failed" and report.claims == []


def test_reference_cannot_restore_a_conflicting_canonical_claim(tmp_path):
    registry, artifact, inputs, _ = catalog(tmp_path)
    tools, model_result = counted(registry, artifact)
    original = tools.records[-1]
    disputed = original.result.model_copy(deep=True)
    disputed.data["claims"][0]["value"] = 2
    tools.records.append(an.ToolRecord(original.name, dict(original.args), disputed))
    selected, _ = an.build_report(completed([model_result.data["claims"][0]["claim_id"]]), tools, inputs)
    assert selected.status == "failed" and selected.claims == []
    unselected, _ = an.build_report(completed([]), tools, inputs)
    assert unselected.status == "incomplete" and len(unselected.claims) == 8
    assert any("conflicting tool claims" in c for c in unselected.caveats)


def test_empty_selection_keeps_all_claims_even_when_preview_omits_them(tmp_path):
    registry, artifact, inputs, names = catalog(tmp_path, 40)
    tools, model_result = counted(registry, artifact)
    assert compact_tool_result(model_result)["data"]["claims"][-1] == {"omitted_items": 36}
    report, _ = an.build_report(completed([]), tools, inputs)
    assert report.status == "completed" and len(report.claims) == 41
    assert {c.filters["traitName"] for c in report.claims if "traitName" in c.filters} == set(names)
