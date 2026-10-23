"""
tests/test_run.py — the controller is a state machine: every branch of the decision flow is exercised
offline on the SYNTHETIC fixture with scripted models. --auto never asks, never grants live access, never
accepts; denial and cancel leave no denied work behind (the saved request log proves it); every run,
including failures and cancellations, leaves answer.md, answer.json and manifest.json.
"""
from __future__ import annotations

import asyncio
import builtins
import json
import subprocess
import sys
from pathlib import Path

import pytest

import run as ctl
from agents.retriever import ApprovedTools
from contracts import FetchManifest, RunResult, ToolResult
from llm import FakeModelClient, ModelProviderError, reply_text, reply_tools
from mcp_bridge import McpTools
from part2_support import SYNTHETIC_BASE, part2_config, part2_runtime, synthetic_context, synthetic_models, synthetic_plan_draft


@pytest.fixture(autouse=True)
def _part2_flow(monkeypatch):
    """Ported from part2: run the release flow the way part2's mock mode did (see part2_support.part2_runtime)."""
    part2_runtime(monkeypatch, server=SYNTHETIC_BASE)


PART2 = Path(__file__).resolve().parents[1]
Q = "For synthetic study S1, give mean yield and clone means with n."
VAGUE = "Which trial has the best yield?"


# --- helpers -------------------------------------------------------------------------------

def cfg(tmp_path, question=Q, **over) -> ctl.RunConfig:
    base = dict(question=question, offline=False, mock_model=False, direct=False, auto=False, fixture="synthetic",
                out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    base.update(over)
    return part2_config(**base)


def scripted(answers):
    queue = list(answers)

    def answer(prompt: str) -> str:
        if not queue:
            raise EOFError
        return queue.pop(0)

    return answer


def go(config, models=None, answers=None):
    controller = ctl.Controller(config, models=models or synthetic_models(), answer_fn=scripted(answers) if answers is not None else None)
    result = asyncio.run(controller.run())
    return controller, result


def claim_values(result: RunResult) -> dict[str, float | int | None]:
    return {c.claim_id.split("_", 2)[2]: c.value for c in (result.analysis.claims if result.analysis else [])}


def saved(controller) -> tuple[RunResult, dict, str]:
    paths = controller.output_paths()
    assert all(p.is_file() for p in paths.values()), paths
    return (RunResult.model_validate_json(paths["answer_json"].read_text(encoding="utf-8")),
            json.loads(paths["manifest_json"].read_text(encoding="utf-8")), paths["answer_md"].read_text(encoding="utf-8"))


def endpoints(manifest: dict) -> list[str]:
    return [r["endpoint"] for r in manifest["requests"]]


def retriever_model(*replies) -> FakeModelClient:
    return FakeModelClient(list(replies), reported_model="mock-model")


def obs(study="S1", variable="V1") -> tuple[str, str]:
    return ("get_observations", json.dumps({"study_db_id": study, "variable_db_id": variable}))


FINAL_S1 = reply_text(json.dumps({"status": "completed", "payload": {"resolved": [{"kind": "study", "id": "S1"}, {"kind": "variable", "id": "V1"}],
                                                                    "studies": ["S1"], "notes": []}}), model="mock-model")


# --- 1. the synthetic A/B flow under --auto -------------------------------------------------------

def test_the_analyst_is_told_the_plans_own_analysis_steps(tmp_path):
    """Real data, 2026-09-29, round 5: the Retriever always saw the plan, the Analyst never did, so on real questions it counted a
    table without the ID grouping the plan asked for. The plan's analyst steps are now in the Analyst's first message."""
    import dataclasses

    made = []
    models = synthetic_models()

    def capture(prefix):
        model = synthetic_models().analyst_factory(prefix)
        made.append(model)
        return model

    controller, result = go(cfg(tmp_path), models=dataclasses.replace(models, analyst_factory=capture))
    assert result.execution_status == "completed" and len(made) == 1
    told = next(m["content"] for m in made[0].requests[0]["messages"] if m["role"] == "user")
    assert "planned analysis steps" in told and "step_3 numeric_summary {}" in told and 'step_4 group_stats {"by": "germplasmDbId"}' in told
    assert "V1 = fresh root yield (unit t/ha)" in told                                        # the variable note is still there
    assert controller.analysis_notes()[-1].startswith("planned analysis steps")


# --- live runs (added 2026-09-29): a FAKE live server; the conftest network guard would stop any real socket -----------------

LIVE_BASE = "https://live.invalid/brapi/v2"


def _reviewed_notes(tmp_path: Path) -> Path:
    path = tmp_path / "ACCESS_NOTES.md"
    path.write_text("# notes\n\nReviewed by Bidhan on: 2026-09-28\n", encoding="utf-8")
    return path


class _FakeLiveServer:
    """Serves the synthetic observation and unit replies at a live-looking address and records every request it receives."""

    REPLIES = {"/observations": "synthetic_observations_S1_V1.json", "/observationunits": "synthetic_observationunits_S1.json"}

    def __init__(self):
        self.calls = []

    def __call__(self, req):
        from brapi_client import TransportResponse
        from part2_support import load_fixture
        endpoint = req.url[len(LIVE_BASE):]
        self.calls.append((endpoint, dict(req.params)))
        parsed, body = load_fixture(self.REPLIES[endpoint])
        assert parsed["_request"]["params"] == dict(req.params), (endpoint, req.params)
        return TransportResponse(200, {"Content-Type": "application/json"}, body)


def _live_cfg(tmp_path: Path, **over) -> ctl.RunConfig:
    base = dict(question=Q, offline=False, mock_model=False, direct=False, auto=False, fixture=None, out_dir=tmp_path / "out",
                cache_dir=tmp_path / "live_cache", access_notes=_reviewed_notes(tmp_path))
    base.update(over)
    return part2_config(**base)


def test_a_live_run_sends_only_what_the_human_approved_after_the_approve(tmp_path, monkeypatch, capsys):
    import brapi_client as bc
    from cache_store import CacheStore
    from part2_support import seed_fixture

    monkeypatch.setenv("BRAPI_MODE", "live")
    monkeypatch.setenv("BRAPI_BASE_URL", LIVE_BASE)
    cache = CacheStore(tmp_path / "live_cache")
    for name in ("synthetic_studies.json", "synthetic_variables.json"):                              # the complete catalogs, cached earlier
        seed_fixture(cache, LIVE_BASE, name)
    server = _FakeLiveServer()
    monkeypatch.setattr(bc, "RequestsTransport", lambda: server)
    answers, asked = ["approve", "accept"], []

    def answer(prompt):
        asked.append((prompt, len(server.calls)))
        return answers.pop(0)

    controller = ctl.Controller(_live_cfg(tmp_path), models=synthetic_models(), answer_fn=answer)
    result = asyncio.run(controller.run())
    assert result.execution_status == "completed", result.notes
    assert asked[0] == ("approve / edit / cancel: ", 0)                                            # nothing left before the approve
    assert server.calls == [("/observations", {"observationVariableDbId": "V1", "page": 0, "pageSize": 1000, "studyDbId": "S1"}),
                            ("/observationunits", {"page": 0, "pageSize": 1000, "studyDbId": "S1"})]   # the catalogs came from the cache
    assert result.manifest.origin == "live" and result.manifest.base_url == LIVE_BASE and "_human_" in result.approval_id
    origins = {r["endpoint"]: r["origin"] for r in controller.requests}
    assert origins["/observations"] == "live" and origins["/observationunits"] == "live" and origins["/studies"] == "cache"
    assert claim_values(result)["0001_mean"] == 18.0                                                # the same numbers, now fetched live
    screen = capsys.readouterr().out
    assert "cached copy fetched 2026-06-01" in screen and f"only the study/variable requests above are sent to {LIVE_BASE}" in screen


def test_mcp_child_transport_gives_the_same_numbers(tmp_path):
    controller, result = go(cfg(tmp_path, direct=False))
    assert result.execution_status == "completed"
    values = claim_values(result)
    assert values["0001_mean"] == 18.0 and values["0001_mean_germplasmDbId_GA"] == 12.0 and values["0001_mean_germplasmDbId_GB"] == 24.0
    _, manifest, _ = saved(controller)
    assert manifest["model"]["transport"] == "mcp" and endpoints(manifest) == ["/studies", "/variables", "/observations", "/observationunits"]
    assert len(result.artifacts) == 3 and result.counts.distinct_studies_fetched == 1


def test_prose_is_kept_only_when_it_cites_the_claims_it_repeats(tmp_path):
    controller, result = go(cfg(tmp_path))
    _, _, markdown = saved(controller)
    prefix = result.artifacts[0].artifact_id[4:10]
    assert f"Pooled mean fresh root yield is 18 from 4 valid plot values, with 1 missing and 1 invalid token [clm_{prefix}_0001_mean]." in markdown
    assert "Removed from the prose:" not in markdown and controller.prose_flags == []
    fabricating = synthetic_models()
    fabricating.analyst_factory = lambda prefix: FakeModelClient([
        reply_tools([("table_info", json.dumps({"artifact_id": f"art_{prefix}_0001"})), ("table_info", json.dumps({"artifact_id": f"art_{prefix}_0002"}))]),
        reply_tools([("numeric_summary", json.dumps({"artifact_id": f"art_{prefix}_0001"}))]),
        reply_text(json.dumps({"status": "completed", "payload": {"methods": [], "caveats": [], "key_claim_ids": [],
                                                                  "narrative": f"The pooled mean was 18 [clm_{prefix}_0001_mean]. Clone B reached 25 in the second plot."}}), model="mock-model"),
    ], reported_model="mock-model")
    controller, result = go(cfg(tmp_path / "fab"), models=fabricating)
    _, _, markdown = saved(controller)
    prefix = result.artifacts[0].artifact_id[4:10]                                            # a new run, a new prefix
    assert result.execution_status == "completed" and f"The pooled mean was 18 [clm_{prefix}_0001_mean]." in markdown
    assert "Clone B reached 25" not in markdown.split("Removed from the prose:")[0]           # the fabricated 25 never reaches the reader as prose
    assert "Removed from the prose:" in markdown and any("numbers without a claim citation" in f for f in controller.prose_flags)


# --- 2. --auto never asks, never grants live, never accepts ----------------------------------------------

# --- 3. the human at the screen: clarify, approve, edit, deny, cancel, accept -----------------------------

def test_interactive_clarification_round_then_completion(tmp_path):
    controller, result = go(cfg(tmp_path, question=VAGUE, auto=False), answers=[Q, "approve", "accept"])
    assert result.execution_status == "completed" and result.human_acceptance == "accepted"
    assert result.plan.question == f"{VAGUE}\nClarification 1 from the person: {Q}"                    # added, not put in its place
    assert any(n.startswith("clarification 1: the planner asked:") and "breeding judgement" in n and n.endswith(f"answer (simulated): {Q}")
               for n in result.notes)                                                                  # what was asked is in the evidence
    assert controller.trace[:5] == ["config", "metadata", "plan", "clarify", "resolve"]


def test_an_answer_is_added_to_the_question_so_a_short_answer_is_enough(tmp_path):
    """2026-09-29: a restated question replaced the first one, so an answer such as '4501' would have reached the planner alone.
    Now the first words stay: 'S1, descriptive means' is enough, and the trait (yield) comes from the FIRST question."""
    _, result = go(cfg(tmp_path, question=VAGUE, auto=False), answers=["S1, descriptive means", "approve", "accept"])
    assert result.execution_status == "completed" and result.plan.question == f"{VAGUE}\nClarification 1 from the person: S1, descriptive means"
    assert result.plan.scope.study_ids == ["S1"] and result.plan.scope.variable_ids == ["V1"]
    _, result = go(cfg(tmp_path / "one", question=VAGUE, auto=False), answers=["no idea", "S1"])       # the command line: one answer
    assert result.execution_status == "needs_clarification" and any("still not ready after 1 clarification round:" in n for n in result.notes)
    _, result = go(cfg(tmp_path / "two", question=VAGUE, auto=False, max_clarification_rounds=2), answers=["no idea", "S1", "approve", "accept"])
    assert result.execution_status == "completed" and result.plan.question.count("\nClarification ") == 2
    for rounds in (0, 6):
        assert any("max_clarification_rounds must be between 1 and 5" in p for p in cfg(tmp_path, max_clarification_rounds=rounds).validate())


def test_waiting_for_a_person_does_not_use_the_runs_time_budget(tmp_path, monkeypatch):
    """A chess clock (2026-09-29, run 192524): the 5-minute budget counts the run's own work, not a person's thinking time."""
    controller = ctl.Controller(cfg(tmp_path, auto=False), answer_fn=lambda prompt: "approve")
    controller.budget.started = 100.0                                           # the run's work began at t = 100 s
    ticks = iter([160.0, 760.0])                                                # asked at t = 160 s, answered 10 minutes later
    monkeypatch.setattr(ctl, "_clock", lambda: next(ticks))
    assert controller._ask("approve / edit / cancel: ") == "approve"
    assert controller.budget.started == 700.0                                   # moved by exactly the 600 s of waiting
    assert controller.budget.elapsed(lambda: 780.0) == 80.0 and controller.budget.exceeded(lambda: 780.0) is None   # 60 s + 20 s of work


def test_the_readable_answer_names_each_id_from_the_fetched_table(tmp_path):
    """Runs 204101 and 204423 (2026-09-29): the 10 study IDs and the 136 locations were only in the claims table, and no name was
    shown. The readable answer is written by code: each ID with the names beside it in the SAME fetched table."""
    import analyst_tools as at
    import brapi_mcp_server as srv
    from contracts import Claim

    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_readable")
    studies = srv.dispatch(ctx, "export_metadata", {"entity": "studies"}).artifact_ids[0]
    counted = at.tool_count_records(ctx.registry, at.CountRecordsArgs(artifact_id=studies, group_by="studyDbId", distinct_by="studyDbId"))
    lines = ctl.readable_answer([Claim.model_validate(c) for c in counted.data["claims"]], ctx.registry)
    assert "- **1 studyDbId value**, each once:" in lines
    assert "  - S1 · SYNTHETIC yield trial S1 · SYNTHETIC location one · 2026" in lines           # name, location, season: from the table
    assert any(line.startswith("- number of distinct non-empty exact text values — studyDbId: **1**") for line in lines)
    controller, _ = go(cfg(tmp_path / "ab"))                                                      # a whole run: answer.md has it too
    markdown = saved(controller)[2]
    assert "## Answer (written by code from the fetched table)" in markdown and "  - GA · clone A: 12" in markdown
    assert markdown.index("## Answer (written by code") < markdown.index("## Numbers (from typed claims only)")


def test_the_planner_sees_its_earlier_question_when_the_person_answers(tmp_path):
    """Run 203842 (2026-09-29): the planner offered 'studies from the 2026 season?', the person said yes, and the planner, which
    never saw its own question, asked again three times. The next plan call now carries the planner's earlier question."""
    import dataclasses

    planner = FakeModelClient([reply_text(json.dumps({"status": "needs_clarification", "question": "Do you mean study S1?"}), model="mock-model"),
                               reply_text(json.dumps({"status": "completed", "payload": synthetic_plan_draft("S1", "V1")}), model="mock-model")],
                              reported_model="mock-model")
    models = dataclasses.replace(synthetic_models(), coordinator=planner)
    _, result = go(cfg(tmp_path, question="What is the mean yield?", auto=False), models=models, answers=["yes, study S1", "cancel"])
    second = planner.requests[1]["messages"][-1]["content"]
    assert "your question 1: Do you mean study S1?" in second and "A person sees the approval screen" in second
    assert "your question" not in planner.requests[0]["messages"][-1]["content"]                 # the first call had nothing to recall
    assert result.execution_status == "canceled"                                                 # the plan was ready; the person canceled


def test_the_approval_screen_asks_the_person_to_check_a_judged_scope():
    """Bidhan, 2026-09-29: 'study', 'studies' or 'stdy' - with a person at the approval screen the planner's judgment stands, and
    the screen asks the person to check a several-study scope that the question never spelled out."""
    from agents.coordinator import KnownMetadata, KnownStudy

    two = KnownMetadata(base_url="https://live.invalid/brapi/v2", complete=True,
                        studies=[KnownStudy(study_id="S1", name="first"), KnownStudy(study_id="S2", name="second")])
    assert ctl.scope_needs_a_check("list ids of sweetpotato stdy done in 2026", ["S1", "S2"], two)
    assert not ctl.scope_needs_a_check("Which studies ran in 2026?", ["S1", "S2"], two)                  # the person said 'studies'
    assert not ctl.scope_needs_a_check("list ids of sweetpotato stdy done in 2026", ["S1"], two)         # one study only
    assert not ctl.scope_needs_a_check("compare study S1 with study S2", ["S1", "S2"], two)               # the person named them


def test_simulated_approve_and_accept_are_labelled_simulated(tmp_path):
    _, result = go(cfg(tmp_path, auto=False), answers=["approve", "accept"])
    assert result.execution_status == "completed" and result.human_acceptance == "accepted" and "_human_" in result.approval_id
    assert any("SIMULATED by a scripted answer" in n for n in result.notes) and any("typed approve at the screen (simulated" in n for n in result.notes)
    _, result = go(cfg(tmp_path / "b", auto=False), answers=["approve", "reject"])
    assert result.execution_status == "completed" and result.human_acceptance == "rejected"


def test_edit_invalidates_the_earlier_approval_and_reshows_the_screen(tmp_path, capsys):
    controller, result = go(cfg(tmp_path, auto=False), answers=["edit", "S9", "edit", "S1", "approve", "accept"])
    out = capsys.readouterr().out
    assert result.execution_status == "completed" and result.human_acceptance == "accepted"
    assert out.count("APPROVAL NEEDED") == 3 and "unknown study IDs ['S9']" in out
    assert any("edited study IDs to ['S1']; any earlier approval is void" in n for n in result.notes)
    assert result.approved_manifest_sha256 == controller.manifest.sha256()


def test_cancel_and_eof_at_the_screen_are_saved_as_canceled(tmp_path):
    controller, result = go(cfg(tmp_path, auto=False), answers=["cancel"])
    assert result.execution_status == "canceled" and ctl.EXIT_CODES["canceled"] == 130
    _, manifest, _ = saved(controller)
    assert manifest["requests"] == []
    controller, result = go(cfg(tmp_path / "eof", auto=False), answers=[])
    assert result.execution_status == "canceled" and any("(no keyboard: EOF)" in n for n in result.notes)
    controller, result = go(cfg(tmp_path / "clar", question=VAGUE, auto=False), answers=[])
    assert result.execution_status == "canceled" and any("canceled at the clarification prompt" in n for n in result.notes)


def test_too_many_edits_block_without_work(tmp_path):
    _, result = go(cfg(tmp_path, auto=False), answers=["edit", "", "edit", "", "edit", "", "edit", ""])
    assert result.execution_status == "blocked" and any("too many edits" in n for n in result.notes) and result.retrieval is None


# --- 4. cancellation while working: the child is closed and the run is saved -----------------------------

class _Interrupting:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.requests: list = []

    async def complete(self, messages, tools, *, max_tokens):
        self.requests.append(len(messages))
        raise self.exc


@pytest.fixture
def captured_processes(monkeypatch):
    import mcp.client.stdio as stdio_mod

    real = stdio_mod._create_platform_compatible_process
    processes = []

    async def wrapper(*args, **kwargs):
        process = await real(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(stdio_mod, "_create_platform_compatible_process", wrapper)
    return processes


def _finished(process) -> bool:
    popen = getattr(process, "popen", None)
    return (popen.poll() is not None) if popen is not None else getattr(process, "returncode", None) is not None


def test_task_cancellation_is_saved_as_canceled_too(tmp_path):
    models = ctl.Models(retriever=_Interrupting(asyncio.CancelledError()), requested="mock-model")
    controller, result = go(cfg(tmp_path), models=models)
    assert result.execution_status == "canceled" and controller.output_paths()["answer_json"].is_file()


# --- 5. the approval gate in the flow: 9 approved, the 10th blocked; ceilings; multi-call turns ---------------

def test_nine_approved_studies_and_the_tenth_is_blocked_before_dispatch(tmp_path):
    controller = ctl.Controller(cfg(tmp_path, offline=True))   # part2's base was offline: a cache-only context
    nine = [f"S{i}" for i in range(1, 10)]
    manifest = FetchManifest(run_id=controller.run_id, base_url="https://synthetic.invalid/brapi/v2", origin="snapshot", snapshot_id="fixture_synthetic",
                             endpoint_families=["studies", "observationvariables", "observations", "observationunits"], observation_study_ids=nine,
                             observation_variable_ids=["V1"], max_http_attempts=20, max_observation_studies=9)
    approval = ctl.build_approval(manifest, issued_by_human=False)
    assert approval.max_observation_studies == 9 and ctl.scope_mismatches(manifest, approval) == []
    ctx = controller._build_ctx(tmp_path / "out")
    ctx.approval = approval

    async def scenario():
        async with McpTools.direct(ctx) as raw:
            gate = ApprovedTools(raw, ctx, approval)
            results = [await gate.call("get_observations", {"study_db_id": s, "variable_db_id": "V1"}) for s in nine + ["S10"]]
            log = await raw.call("request_log", {"run_id": controller.run_id})
        return results, log

    results, log = asyncio.run(scenario())
    assert results[0].ok and all(not r.ok for r in results[1:9])                           # S1 cached; S2..S9 approved but not in the offline cache
    assert all(r.error.code == "unsupported" for r in results[1:9])                          # a cache miss offline, not a permission problem
    assert not results[9].ok and results[9].error.code == "not_authorized" and "S10" in results[9].error.message
    ledger = {e.study_id: e.status for e in ctx.registry.ledger()}
    assert ledger["S1"] == "complete" and ledger["S10"] == "blocked" and all(ledger[s] == "failed" for s in nine[1:])
    attempted = [(r["endpoint"], r["params"].get("studyDbId")) for r in log.data["records"]]
    assert attempted == [("/observations", "S1")]      # only the cached study produced a request; offline misses make no attempt; S10 never left the gate


def test_multiple_tool_calls_in_one_turn_are_dispatched_in_order(tmp_path):
    model = retriever_model(
        reply_tools([("search_studies", json.dumps({"name_contains": "S1"})), ("list_variables", json.dumps({"name_contains": "fresh"})), obs()]),
        reply_tools([("get_observation_units", json.dumps({"study_db_id": "S1"}))]),
        FINAL_S1,
    )
    controller, result = go(cfg(tmp_path), models=ctl.Models(retriever=model, analyst_factory=synthetic_models().analyst_factory))
    assert result.execution_status == "completed"
    _, manifest, _ = saved(controller)
    assert endpoints(manifest) == ["/studies", "/variables", "/observations", "/observationunits"]
    assert next(a for a in controller.agents if a.log_path.endswith("llm_retriever.jsonl")).model_calls >= 3


def test_retrieval_of_an_unapproved_study_blocks_the_run_before_analysis(tmp_path):
    model = retriever_model(reply_tools([("search_studies", json.dumps({"name_contains": "S1"}))]), reply_tools([obs("S10")]),
                            reply_text(json.dumps({"status": "completed", "payload": {"resolved": [], "studies": ["S10"], "notes": []}}), model="mock-model"))
    controller, result = go(cfg(tmp_path), models=ctl.Models(retriever=model))
    assert result.execution_status == "blocked" and result.analysis is None
    assert {e.study_id: e.status for e in result.retrieval.study_ledger} == {"S10": "blocked"}
    _, manifest, _ = saved(controller)
    assert "/observations" not in endpoints(manifest) and controller.trace[-3:] == ["approve", "retrieve", "record"]


# --- 6. failures are saved; configuration is checked; the CLI -----------------------------------------------

def test_failed_runs_are_saved_with_evidence(tmp_path):
    class Failing:
        async def complete(self, messages, tools, *, max_tokens):
            raise ModelProviderError("connection", "the model endpoint is unreachable")

    controller, result = go(cfg(tmp_path), models=ctl.Models(retriever=Failing(), requested="mock-model"))
    assert result.execution_status == "failed" and ctl.EXIT_CODES["failed"] == 1
    reloaded, manifest, markdown = saved(controller)
    assert reloaded.execution_status == "failed" and "provider_connection" in " ".join(result.retrieval.warnings)
    assert manifest["state_trace"][-1] == "record" and "Retrieval ended with status failed" in markdown


# --- the console's code page (Luna 2026-10-23, the T4 pass) -------------------------------------------------------------

def cp1252_console(monkeypatch):
    """stdout as a Windows console redirected to a file: cp1252, strict."""
    import io
    buffer = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(buffer, encoding="cp1252", errors="strict", newline="\n", write_through=True))
    return buffer


def test_an_answer_with_a_character_the_console_cannot_hold_still_completes(tmp_path, monkeypatch):
    """6 of 15 Luna runs had a right analysis and failed at accept: the model's prose held U+2212 and print() raised on cp1252."""
    original = ctl.Controller.answer_markdown
    monkeypatch.setattr(ctl.Controller, "answer_markdown", lambda self: original(self) + "\nt = \u22124.1145")
    buffer = cp1252_console(monkeypatch)
    controller, result = go(cfg(tmp_path, auto=False), answers=["approve", "accept"])
    assert result.execution_status == "completed" and result.human_acceptance == "accepted"
    assert "t = ?4.1145" in buffer.getvalue().decode("cp1252")
    assert "t = \u22124.1145" in controller.output_paths()["answer_md"].read_text(encoding="utf-8")    # the saved answer keeps it


def test_every_console_message_survives_the_code_page(monkeypatch):
    buffer = cp1252_console(monkeypatch)
    ctl._say("THE ANALYST ASKS: is \u2212 or \u2264 the sign?")
    assert buffer.getvalue().decode("cp1252") == "THE ANALYST ASKS: is ? or ? the sign?\n"


def test_the_run_prints_only_through_console_say():
    """Every console message of a run goes through console.say; a bare print() of model or person text could bring the crash back.
    Review of the fix: the coordinator's question (coordination_runtime.py) was still printed bare."""
    import coordination_runtime
    import console
    for module in (ctl, coordination_runtime):
        assert "print(" not in Path(module.__file__).read_text(encoding="utf-8"), module.__name__
    assert Path(console.__file__).read_text(encoding="utf-8").count("print(") == 1


def test_the_coordinators_question_with_a_character_the_console_cannot_hold(tmp_path, monkeypatch):
    """Review of the fix: an agent-led run without --auto (the app, the CLI, the harness) whose coordinator asks a question holding
    U+2265 failed in state supervise, the same crash one file over. The question now reaches the console with a "?"."""
    from dataclasses import replace
    from llm import FakeModelClient, reply_text
    from synthetic_support import hosted_config
    for key, value in (("BRAPI_MODE", "live"), ("BRAPI_BASE_URL", "https://synthetic.invalid/brapi/v2"), ("LLM_ALLOW_REMOTE", "true"),
                       ("LLM_API_KEY", "synthetic-regression-placeholder"), ("PART2_IGNORE_DOTENV", "true")):
        monkeypatch.setenv(key, value)
    buffer = cp1252_console(monkeypatch)
    question = "Do you mean yield \u2265 20 t/ha, or every plot?"
    decisions = [{"action": "clarify", "reason": "Two readings.", "question": question}, "retrieve", "analyze", "finish"]
    supervisor = FakeModelClient([reply_text(json.dumps({"status": "completed", "payload": item if isinstance(item, dict) else
                                                         {"action": item, "reason": "Synthetic next-stage decision."}})) for item in decisions])
    config = hosted_config(tmp_path, Q, auto=False)
    assert config.agent_led and not config.auto
    controller = ctl.Controller(config, models=replace(synthetic_models(), supervisor=supervisor),
                                answer_fn=scripted(["every plot", "approve", "continue", "accept"]))
    result = asyncio.run(controller.run())
    assert not any("UnicodeEncodeError" in note for note in result.notes) and result.execution_status != "failed"
    assert "THE COORDINATOR ASKS: Do you mean yield ? 20 t/ha" in buffer.getvalue().decode("cp1252")


def test_the_answer_files_are_the_runs_texts_in_utf_8_whatever_the_console(tmp_path, monkeypatch):
    """Coach 2026-10-23 04:49 UTC: the console fix must not change answer.md or answer.json. Under a cp1252 console they are byte
    for byte the run's own texts in UTF-8 (each "\\n" written as the platform's line end), the U+2212 included."""
    import os
    original = ctl.Controller.answer_markdown
    monkeypatch.setattr(ctl.Controller, "answer_markdown", lambda self: original(self) + "\nt = \u22124.1145")
    cp1252_console(monkeypatch)
    controller, result = go(cfg(tmp_path, auto=False), answers=["approve", "accept"])
    paths = controller.output_paths()
    assert paths["answer_md"].read_bytes() == controller.answer_markdown().replace("\n", os.linesep).encode("utf-8")
    assert paths["answer_json"].read_bytes() == ctl.dump_json(result).replace("\n", os.linesep).encode("utf-8")
    assert "\u2212".encode("utf-8") in paths["answer_md"].read_bytes()
