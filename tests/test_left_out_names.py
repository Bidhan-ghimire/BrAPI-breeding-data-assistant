"""
tests/test_left_out_names.py - the change-set "left_out_names" (coach 2026-10-23 06:49 UTC, NEXT 1):
(a) compare_trials_paired (T4) and stress_indices (T5) name every entry the person left out, on every claim and in the answer,
    in anova_trial's words: "left out by name: ..." plus the count (Luna C27.1: the coordinator saw "excluded 3" and stopped a
    right answer, 34.26 / 20.84, because it could not verify the BT progeny restriction);
(b) the F4 gate covers both tools: exclude names and name_prefix must be written in the person's words, else refused and asked;
(c) HM and RDI carry the sources the research notes record (Bidinger et al. 1987; Fischer and Wood 1979).
SYNTHETIC tables; the real C27.1 question for the gate. No network, no model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest

import agents.analyst as an
import analyst_tools as at
import brapi_mcp_server as srv
import chat_presenter as cp
from artifacts import ArtifactRegistry
from contracts import AnalysisReport, Claim, MeasurementMeta, ToolResult
from part2_support import SYNTHETIC_BASE, synthetic_context

ENTRIES = [f"E{i}" for i in range(8)]
C27_1 = next(r["question"] for r in json.loads((Path(__file__).resolve().parents[1] / "eval" / "benchmark" / "spb_subquestions.json")
                                               .read_text(encoding="utf-8"))["subquestions"] if r["sub_id"] == "C27.1")


@pytest.fixture
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "out", "run_left_out", base_url=SYNTHETIC_BASE)


def trial(level: float, *, seed: float = 0.0) -> dict[str, list[float]]:
    return {e: [level * (1.0 + 0.3 * math.cos(1.1 * i + seed)) + 0.2 * math.sin(2.3 * i + 1.7 * k + 3.1 * seed) for k in range(2)]
            for i, e in enumerate(ENTRIES)}


def save(registry, study: str, values: dict[str, list[float]], *, empty=()) -> str:
    """empty: entries whose plots hold no value (the entry is in the table, without a mean)."""
    rows = [{"observationDbId": f"O{study}-{e}-{k}", "observationUnitDbId": f"{study}-{e}-{k}", "observationUnitName": f"plot {k}",
             "germplasmDbId": e, "germplasmName": f"clone {e}", "studyDbId": study, "observationVariableDbId": "V1",
             "observationVariableName": "SYNTHETIC yield", "observationTimeStamp": "", "value": "" if e in empty else repr(float(v))}
            for e, plot_values in values.items() for k, v in enumerate(plot_values, 1)]
    columns = {col: srv.OBSERVATION_COLUMNS.get(col, "text") for col in rows[0]}
    return registry.save_table(rows, kind="observations", columns=columns, complete=True, study_ids=[study], variable_ids=["V1"],
                               measurement=MeasurementMeta(trait="SYNTHETIC yield", unit="t/ha"), quality_notes=["SYNTHETIC"]).artifact_id


TOOLS = {"compare_trials_paired": ("base_artifact_id", "other_artifact_id"), "stress_indices": ("nonstress_artifact_id", "stress_artifact_id")}


def run(registry, tool: str, **args) -> ToolResult:
    first, second = TOOLS[tool]
    return at.dispatch_analyst(registry, tool, {first: save(registry, "S1", trial(10.0)), second: save(registry, "S2", trial(4.0, seed=0.4)),
                                                **args})


def claims(r) -> list[Claim]:
    return [Claim.model_validate(k) for k in r.data["claims"]]


def answer(ctx, claim_list) -> str:
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claim_list, caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    return cp.answer_summary(view)


# --- (a) the left-out entries by name --------------------------------------------------------------------------------------

@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_every_claim_names_the_entries_the_person_left_out(registry, tool):
    """Luna C27.1: the claims said "entries left out: excluded 3" and the coordinator could not see which. Now every claim carries
    the names as the person gave them, grouped per entry (an entry named by its ID and its name is one), and the count."""
    r = run(registry, tool, exclude=["clone E5", "E6", "E5"])
    assert r.ok, r.error
    assert all("left out by name: clone E5 / E5, E6 (2 entries)" in k.transformation for k in claims(r))
    plain = run(registry, tool)
    assert plain.ok and not any("left out by name" in k.transformation for k in claims(plain))


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_a_named_entry_without_a_mean_is_named_apart(registry, tool):
    """Deliberate bug L4 showed the mismatch (E7 has no valid value in either table, listed beside E5 with a count of 1); the review
    of this set (finding 1) showed that dropping E7 hid it from the coordinator's check. It is named apart: the count is still the
    entries left out, and every name the person gave is on the claims."""
    first, second = TOOLS[tool]
    r = at.dispatch_analyst(registry, tool, {first: save(registry, "S1", trial(10.0), empty=("E7",)),
                                             second: save(registry, "S2", trial(4.0, seed=0.4), empty=("E7",)),
                                             "exclude": ["clone E5", "clone E7"]})
    assert r.ok, r.error
    # coach 07:48 UTC (5): the category's words name the trait ("named, no <trait> value in either trial")
    assert all("left out by name: clone E5 (1 entry); named, no SYNTHETIC yield value in either trial: clone E7" in k.transformation
               for k in claims(r))


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_the_answer_prints_the_names(tmp_path, tool):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id=f"run_left_out_{tool}")
    r = run(ctx.registry, tool, exclude=["clone E5"])
    assert r.ok, r.error
    shown = answer(ctx, claims(r))
    assert "Left out by name: clone E5 \\(1 entry\\)." in shown


# --- (b) the F4 gate on both tools -----------------------------------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.calls = []

    async def schemas(self):
        return []

    async def call(self, name, args):
        self.calls.append((name, dict(args)))
        return ToolResult(ok=True, data={}, artifact_ids=[])


def guarded(question):
    tools = an.GuardedAnalystTools(_Recorder(), ["art_1", "art_2"], question=question)
    tools.inspected.update({"art_1", "art_2"})
    return tools


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_exclude_and_name_prefix_come_from_the_persons_words(tool):
    """Coach 06:49 UTC: F4 extended to both tools. C27.1's real wording passes ("all entries except Beauregard, Tanzania and
    CIP199062.1"); a name or a prefix the person never wrote is refused before the tool runs, and the analyst asks."""
    first, second = TOOLS[tool]
    base = {first: "art_1", second: "art_2"}
    tools = guarded(C27_1)
    assert anyio.run(tools.call, tool, {**base, "exclude": ["Beauregard", "Tanzania", "CIP199062.1"]}).ok and len(tools.inner.calls) == 1
    for extra, word in (({"exclude": ["Beauregard", "Resisto"]}, "exclude 'Resisto'"), ({"name_prefix": "PH17."}, "name_prefix 'PH17.'")):
        tools = guarded(C27_1)
        r = anyio.run(tools.call, tool, {**base, **extra})
        assert not r.ok and r.error.code == "invalid_argument" and f"{word} is not written in the person's words" in r.error.message, extra
        assert tools.inner.calls == [] and "needs_clarification" in r.error.message
    family = guarded("Compare yield between the two trials for the PH17 family only.")
    assert anyio.run(family.call, tool, {**base, "name_prefix": "PH17."}).ok            # the prefix's stem is written


def test_threshold_count_and_the_mcp_servers_guard_are_unchanged():
    """threshold_count's own line stays (coach 04:49 UTC); the MCP server's guard has no question."""
    assert anyio.run(guarded("How many entries are above Beauregard's mean?").call, "threshold_count",
                     {"artifact_id": "art_1", "exclude": ["Covington"]}).ok
    tools = an.GuardedAnalystTools(_Recorder(), ["art_1", "art_2"], question=None)
    tools.inspected.update({"art_1", "art_2"})
    assert anyio.run(tools.call, "stress_indices", {"nonstress_artifact_id": "art_1", "stress_artifact_id": "art_2", "exclude": ["Resisto"]}).ok


# --- (c) the sources of HM and RDI -----------------------------------------------------------------------------------------

def test_hm_and_rdi_carry_the_sources_the_research_notes_record(tmp_path):
    """The research notes (NOTES_multi_trial_gxe_stress.md, Q4, the iPASTIC reference list): Harmonic Mean - Bidinger et al. 1987;
    Relative Drought Index - Fischer and Wood 1979."""
    assert at.STRESS_FORMULAS["HM"].endswith("(Bidinger et al. 1987)") and at.STRESS_FORMULAS["RDI"].endswith("(Fischer and Wood 1979)")
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_left_out_sources")
    shown = answer(ctx, claims(run(ctx.registry, "stress_indices", rank_by="RDI")))
    assert "(Fischer and Wood 1979)" in shown


# --- review of this set (findings 2, 3) and of T6 (finding 2: the prefix rule) ----------------------------------------------

@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_cells_is_checked_for_anova_trial_alone(tool):
    """Finding 2: the two tools have no cells argument; the guard refused one with anova_trial's clarification text. It passes it
    on, and the tool's own argument check refuses it."""
    first, second = TOOLS[tool]
    tools = guarded(C27_1)
    assert anyio.run(tools.call, tool, {first: "art_1", second: "art_2", "cells": "mean"}).ok


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_a_name_with_a_bar_keeps_the_line_whole(tmp_path, tool):
    """Finding 3: a name holding " | " cut the names line where the claim's parts are split."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id=f"run_left_out_bar_{tool}")
    first, second = TOOLS[tool]

    def barred(study, level, seed=0.0):
        tid = save(ctx.registry, study, trial(level, seed=seed))
        rows = [{**row, "germplasmName": "check | E5" if row["germplasmDbId"] == "E5" else row["germplasmName"]}
                for row in at._load(ctx.registry, tid).rows]
        columns = {col: srv.OBSERVATION_COLUMNS.get(col, "text") for col in rows[0]}
        return ctx.registry.save_table(rows, kind="observations", columns=columns, complete=True, study_ids=[study], variable_ids=["V1"],
                                       measurement=MeasurementMeta(trait="SYNTHETIC yield", unit="t/ha"), quality_notes=["SYNTHETIC"]).artifact_id

    r = at.dispatch_analyst(ctx.registry, tool, {first: barred("S1", 10.0), second: barred("S2", 4.0, 0.4), "exclude": ["check | E5"]})
    assert r.ok, r.error
    assert all("left out by name: check / E5 (1 entry)" in k.transformation for k in claims(r))
    assert "Left out by name: check / E5 \\(1 entry\\)." in answer(ctx, claims(r))


def test_a_bare_word_is_a_prefix_only_with_its_star():
    """Review of T6, finding 2: the stem rule admitted any whole word as a prefix, and a name as a prefix ("Isabel" took Isabela). A
    prefix counts when written as given ("PH17.*", "PH17."), or through its stem when it ends with . - or _ ("PZ14 parents" for
    PZ14.); a bare word only when the person wrote its star ("Isabel*")."""
    words = "Founders (PJ05.*, Isabel, CIP105103.1) at Satipo; PZ14 parents likewise; the PH17.* offspring."
    for prefix, ok in (("PJ05.", True), ("PJ05.*", True), ("PZ14.", True), ("PH17.", True), ("PH17.*", True), ("Isabel", False),
                       ("Isabel*", False), ("Satipo*", False), ("PZ", False), ("PZ14", False), ("CIP105103.", False)):
        assert an.prefix_named(prefix, words) is ok, prefix
    assert an.prefix_named("Isabel", "the Isabel* family")
    tools = guarded("Compare yield in the two trials for Isabel and the PH17 family.")
    r = anyio.run(tools.call, "compare_trials_paired", {"base_artifact_id": "art_1", "other_artifact_id": "art_2", "name_prefix": "Isabel"})
    assert not r.ok and "name_prefix 'Isabel' is not written in the person's words" in r.error.message
