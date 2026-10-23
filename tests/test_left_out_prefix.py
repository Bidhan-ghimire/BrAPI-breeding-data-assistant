"""
tests/test_left_out_prefix.py - the change-set (4)+(5) (coach 2026-10-23 07:48 UTC, NEXT 2) for compare_trials_paired (T4) and
stress_indices (T5):
(4) every claim names the prefix the person kept, as the tool applied it: "left out by prefix: M entries outside PH17.* (N kept)" (the
    coach's form of 09:51 UTC; before: "outside PH17.* (N entries)"), M = the
    entries with a mean the prefix left out; the answer prints it;
(5) a name the person gave that is an entry of the trial (its plot list, from get_observation_units) but has no value of the trait
    goes in "named, no <trait> value in either trial: ..." - it is not an unknown name. A name absent from the trial is still
    refused, and the F4 gate is unchanged. Luna C27.3: Beauregard and Tanzania have no dry-matter plot, so they were in neither
    table, were refused as unknown, and the claims named CIP199062.1 alone beside the question's three names.
SYNTHETIC tables; the real C27.3 question for the gate. No network, no model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest

import agents.analyst as an
import agents.coordinator as co
import analyst_tools as at
import brapi_mcp_server as srv
import chat_presenter as cp
from artifacts import ArtifactRegistry
from contracts import AnalysisReport, Claim, MeasurementMeta, ToolResult
from part2_support import SYNTHETIC_BASE, synthetic_context

ENTRIES = [f"E{i}" for i in range(8)]
NAMES = {**{f"E{i}": f"PH17.00{i}" for i in range(6)}, "E6": "Beauregard", "E7": "Tanzania"}
C27_3 = next(r["question"] for r in json.loads((Path(__file__).resolve().parents[1] / "eval" / "benchmark" / "spb_subquestions.json")
                                               .read_text(encoding="utf-8"))["subquestions"] if r["sub_id"] == "C27.3")
TOOLS = {"compare_trials_paired": ("base_artifact_id", "other_artifact_id"), "stress_indices": ("nonstress_artifact_id", "stress_artifact_id")}


@pytest.fixture
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "out", "run_left_out_prefix", base_url=SYNTHETIC_BASE)


def trial(level: float, *, seed: float = 0.0, entries=ENTRIES) -> dict[str, list[float]]:
    return {e: [level * (1.0 + 0.3 * math.cos(1.1 * i + seed)) + 0.2 * math.sin(2.3 * i + 1.7 * k + 3.1 * seed) for k in range(2)]
            for i, e in enumerate(ENTRIES) if e in entries}


def save(registry, study: str, values: dict[str, list[float]], *, names=NAMES, empty=(), complete=True) -> str:
    """The measurements of one study; empty: entries whose plots hold no value (in the table, without a mean)."""
    rows = [{"observationDbId": f"O{study}-{e}-{k}", "observationUnitDbId": f"{study}-{e}-{k}", "observationUnitName": f"plot {k}",
             "germplasmDbId": e, "germplasmName": names[e], "studyDbId": study, "observationVariableDbId": "V1",
             "observationVariableName": "SYNTHETIC yield", "observationTimeStamp": "", "value": "" if e in empty else repr(float(v))}
            for e, plot_values in values.items() for k, v in enumerate(plot_values, 1)]
    columns = {col: srv.OBSERVATION_COLUMNS.get(col, "text") for col in rows[0]}
    return registry.save_table(rows, kind="observations", columns=columns, complete=complete, study_ids=[study], variable_ids=["V1"],
                               measurement=MeasurementMeta(trait="SYNTHETIC yield", unit="t/ha"), quality_notes=["SYNTHETIC"]).artifact_id


def plot_list(registry, *studies: str, entries=ENTRIES, names=NAMES, drop=(), complete=True, row_study=None) -> str:
    """The trial's plot list as get_observation_units saves it: every plot with its entry, whether or not it holds a value."""
    columns = {col: meaning for col, meaning in srv.UNIT_COLUMNS.items() if col not in drop}
    rows = [{col: value for col, value in {"observationUnitDbId": f"{study}-{e}-{k}", "observationUnitName": f"plot {k}", "germplasmDbId": e,
                                           "germplasmName": names[e], "studyDbId": row_study or study, "levelName": "plot", "levelCode": str(k),
                                           "observationUnitPosition": "{}"}.items() if col in columns}
            for study in studies for e in entries for k in (1, 2)]
    return registry.save_table(rows, kind="observation_units", columns=columns, complete=complete, study_ids=list(studies), variable_ids=[],
                               quality_notes=["SYNTHETIC"]).artifact_id


def tables(registry, tool: str, *, entries=ENTRIES, names=NAMES, empty=()) -> dict[str, str]:
    first, second = TOOLS[tool]
    return {first: save(registry, "S1", trial(10.0, entries=entries), names=names, empty=empty),
            second: save(registry, "S2", trial(4.0, seed=0.4, entries=entries), names=names, empty=empty)}


def claims(r) -> list[Claim]:
    return [Claim.model_validate(k) for k in r.data["claims"]]


def lineage(registry, r) -> list[str]:
    return at._load(registry, r.data["result_artifact_id"]).manifest.quality_notes


def answer(ctx, claim_list) -> str:
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claim_list, caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    return cp.answer_summary(view)


# --- (4) the prefix on every claim ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_every_claim_names_the_prefix_the_person_kept(tmp_path, tool):
    """(4): name_prefix was checked against the person's words (F4) but never named on the claims, so the coordinator could not
    see the family restriction. Every claim now says which prefix was kept and how many entries with a mean it left out
    (Beauregard and Tanzania here); "outside" because the prefix keeps the family. The answer prints it."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id=f"run_prefix_{tool}")
    r = at.dispatch_analyst(ctx.registry, tool, {**tables(ctx.registry, tool), "name_prefix": "PH17."})
    assert r.ok, r.error
    assert all("left out by prefix: 2 entries outside PH17.* (6 kept)" in k.transformation for k in claims(r))
    assert not any("left out by name" in k.transformation for k in claims(r))
    assert cp._text("Left out by prefix: 2 entries outside PH17.* (6 kept).") in answer(ctx, claims(r))
    plain = at.dispatch_analyst(ctx.registry, tool, tables(ctx.registry, tool))
    assert plain.ok and not any("left out by prefix" in k.transformation for k in claims(plain))
    assert "Left out by prefix" not in answer(ctx, claims(plain))


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_the_prefix_count_leaves_the_named_entries_to_the_names_and_says_zero(registry, tool):
    """The two counts never overlap: an entry the person named is counted by name, not again by the prefix. A prefix that keeps
    every entry still says so, with 0: the restriction was applied."""
    r = at.dispatch_analyst(registry, tool, {**tables(registry, tool), "name_prefix": "PH17.", "exclude": ["Tanzania"]})
    assert r.ok, r.error
    assert all("left out by name: Tanzania (1 entry)" in k.transformation and "left out by prefix: 1 entry outside PH17.* (6 kept)" in k.transformation
               for k in claims(r))
    every = at.dispatch_analyst(registry, tool, {**tables(registry, tool, entries=ENTRIES[:6]), "name_prefix": "PH17."})
    assert every.ok, every.error
    assert all("left out by prefix: 0 entries outside PH17.* (6 kept)" in k.transformation for k in claims(every))


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_a_bar_in_the_prefix_keeps_the_part_whole(registry, tool):
    """Deliberate bug M26: " | " splits a claim's parts, so a prefix holding it cut the line (as a name did, left_out_names finding
    3); it is written " / "."""
    names = {**{f"E{i}": f"fam | {i}" for i in range(6)}, "E6": "Beauregard", "E7": "Tanzania"}
    r = at.dispatch_analyst(registry, tool, {**tables(registry, tool, names=names), "name_prefix": "fam | "})
    assert r.ok, r.error
    assert all("left out by prefix: 2 entries outside fam / * (6 kept)" in k.transformation.split(" | ") for k in claims(r))


# --- (5) a named entry of the trial without a value of the trait --------------------------------------------------------------

@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_a_named_entry_of_the_trial_without_a_value_is_named_not_refused(tmp_path, tool):
    """(5), Luna C27.3: Beauregard and Tanzania are in the trials' plot lists but have no value of the trait, so they are in neither
    table. With the plot lists they are named apart, in the coach's words, beside the entry left out with a mean; the numbers are
    those of leaving out that entry alone; the lineage names the plot lists."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id=f"run_roster_{tool}")
    t = tables(ctx.registry, tool, entries=ENTRIES[:6])
    rosters = [plot_list(ctx.registry, "S1"), plot_list(ctx.registry, "S2")]
    r = at.dispatch_analyst(ctx.registry, tool, {**t, "exclude": ["Beauregard", "Tanzania", "PH17.005"], "roster_artifact_ids": rosters})
    assert r.ok, r.error
    want = "left out by name: PH17.005 (1 entry); named, no SYNTHETIC yield value in either trial: Beauregard, Tanzania"
    assert all(want in k.transformation for k in claims(r))
    assert cp._text(f"L{want[1:]}.") in answer(ctx, claims(r))
    alone = at.dispatch_analyst(ctx.registry, tool, {**t, "exclude": ["PH17.005"]})
    assert alone.ok and [k.value for k in claims(r)] == [k.value for k in claims(alone)]
    assert any(note == f"plot lists: {rosters[0]}, {rosters[1]}" for note in lineage(ctx.registry, r))
    assert not any(note.startswith("plot lists") for note in lineage(ctx.registry, alone))


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_a_name_absent_from_the_trial_is_still_refused(registry, tool):
    """'Unknown' stays = absent from the trial's entries (coach 07:48 UTC): a name in neither table nor plot list is refused, with
    the plot lists named; without the plot lists a name in neither table is refused exactly as before."""
    t = tables(registry, tool, entries=ENTRIES[:6])
    rosters = [plot_list(registry, "S1"), plot_list(registry, "S2")]
    r = at.dispatch_analyst(registry, tool, {**t, "exclude": ["Resisto"], "roster_artifact_ids": rosters})
    assert not r.ok and r.error.code == "invalid_argument"
    assert "exclude 'Resisto' is not an entry ID or name in" in r.error.message and f"nor in the plot lists {rosters[0]}, {rosters[1]}" in r.error.message
    r = at.dispatch_analyst(registry, tool, {**t, "exclude": ["Beauregard"]})
    assert not r.ok and "exclude 'Beauregard' is not an entry ID or name in" in r.error.message and "plot lists" not in r.error.message


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_a_name_a_table_knows_is_resolved_there_as_before(registry, tool):
    """A name a trait table knows never reaches the plot lists: one that names two entries there is refused as before, even when
    the plot list gives it one entry. An entry's ID resolves in the plot list as its name does; one plot list is enough."""
    names = {**NAMES, "E1": "PH17.000"}                                  # E0 and E1 share a name in the tables
    t = tables(registry, tool, entries=ENTRIES[:6], names=names)
    rosters = [plot_list(registry, "S1"), plot_list(registry, "S2")]     # where E0 alone is PH17.000
    r = at.dispatch_analyst(registry, tool, {**t, "exclude": ["PH17.000"], "roster_artifact_ids": rosters})
    assert not r.ok and "is the name of 2 entries" in r.error.message
    r = at.dispatch_analyst(registry, tool, {**t, "exclude": ["E7"], "roster_artifact_ids": rosters[1:]})
    assert r.ok, r.error
    assert all("left out by name: no entry with a mean; named, no SYNTHETIC yield value in either trial: E7" in k.transformation
               for k in claims(r))


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_an_entry_with_plots_but_no_valid_value_takes_the_same_words(registry, tool):
    """The existing category (an entry whose plots hold no valid value) is the one the coach named: same words, with the trait."""
    r = at.dispatch_analyst(registry, tool, {**tables(registry, tool, empty=("E7",)), "exclude": ["PH17.005", "Tanzania"]})
    assert r.ok, r.error
    assert all("left out by name: PH17.005 (1 entry); named, no SYNTHETIC yield value in either trial: Tanzania" in k.transformation
               for k in claims(r))


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_plot_lists_are_checked(registry, tool):
    """A plot list is the plot list of one of the two studies compared, one per study, with the entry column; anything else is
    refused, never guessed."""
    first, _ = TOOLS[tool]
    t = tables(registry, tool, entries=ENTRIES[:6])
    s1 = plot_list(registry, "S1")
    for rosters, words in (([t[first]], "is observations, not observation_units"),
                           ([plot_list(registry, "S3")], "is of study S3, not of a study compared (S1, S2)"),
                           ([s1, plot_list(registry, "S1", entries=ENTRIES[:7])], "two plot lists of study S1"),
                           ([plot_list(registry, "S1", "S2")], "is of ['S1', 'S2']; pass one plot list per study"),
                           ([plot_list(registry, "S1", drop=("germplasmDbId",))], "has no column 'germplasmDbId'")):
        r = at.dispatch_analyst(registry, tool, {**t, "exclude": ["Beauregard"], "roster_artifact_ids": rosters})
        assert not r.ok and r.error.code == "invalid_argument" and words in r.error.message, (words, r.error)
    assert at.dispatch_analyst(registry, tool, {**t, "exclude": ["Beauregard"], "roster_artifact_ids": [s1, s1]}).ok   # the same list twice is one


# --- the review of this set (SAFE; findings 1, 2, 3, 5 and the answer of 8) ------------------------------------------------------

@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_a_plot_list_name_for_an_entry_the_tables_hold_is_refused(registry, tool):
    """Finding 1: the plot lists named E3 "Zapallo" where the tables name it "PH17.003"; the name resolved through the plot list and
    E3 was left out with its mean (n 8 -> 7). The plot lists name only entries no table holds: this one is refused, with the
    tables' name for it, never guessed."""
    other_names = {**NAMES, "E3": "Zapallo"}
    rosters = [plot_list(registry, "S1", names=other_names), plot_list(registry, "S2", names=other_names)]
    r = at.dispatch_analyst(registry, tool, {**tables(registry, tool), "exclude": ["Zapallo"], "roster_artifact_ids": rosters})
    assert not r.ok and r.error.code == "invalid_argument"
    assert "exclude 'Zapallo' is entry E3 in the plot lists, which the tables hold as 'PH17.003'" in r.error.message


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_a_name_missing_from_an_incomplete_plot_list_is_not_called_absent_from_the_trial(registry, tool):
    """Finding 2: an incomplete plot list was taken as the trial's whole entry list. The refusal now says only the plots fetched
    were searched; a name the incomplete list does hold is still named."""
    t = tables(registry, tool, entries=ENTRIES[:6])
    short = [plot_list(registry, "S1", entries=ENTRIES[:7], complete=False), plot_list(registry, "S2", entries=ENTRIES[:7])]   # one is enough
    r = at.dispatch_analyst(registry, tool, {**t, "exclude": ["Tanzania"], "roster_artifact_ids": short})
    assert not r.ok and "nor in the plot lists" in r.error.message and "(incomplete: only the plots fetched were searched)" in r.error.message
    r = at.dispatch_analyst(registry, tool, {**t, "exclude": ["Beauregard"], "roster_artifact_ids": short})
    assert r.ok, r.error
    assert all("named, no SYNTHETIC yield value in either trial: Beauregard" in k.transformation for k in claims(r))
    full = at.dispatch_analyst(registry, tool, {**t, "exclude": ["Resisto"], "roster_artifact_ids": [plot_list(registry, "S1")]})
    assert not full.ok and "incomplete" not in full.error.message


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_the_words_say_when_the_tables_are_incomplete(registry, tool):
    """Finding 3: with incomplete tables (allow_incomplete) "no value in either trial" is said of the rows fetched."""
    first, second = TOOLS[tool]
    t = {first: save(registry, "S1", trial(10.0, entries=ENTRIES[:6]), complete=False), second: save(registry, "S2", trial(4.0, seed=0.4, entries=ENTRIES[:6]))}
    rosters = [plot_list(registry, "S1"), plot_list(registry, "S2")]
    r = at.dispatch_analyst(registry, tool, {**t, "exclude": ["Tanzania"], "roster_artifact_ids": rosters, "allow_incomplete": True})
    assert r.ok, r.error
    assert all("named, no SYNTHETIC yield value in either trial (in the rows fetched): Tanzania" in k.transformation for k in claims(r))


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_a_plot_list_whose_rows_are_of_another_study_is_refused(registry, tool):
    """Finding 5: the checks read the manifest's study only; a list labelled S1 whose rows say S9 resolved names."""
    t = tables(registry, tool, entries=ENTRIES[:6])
    r = at.dispatch_analyst(registry, tool, {**t, "exclude": ["Beauregard"], "roster_artifact_ids": [plot_list(registry, "S1", row_study="S9")]})
    assert not r.ok and r.error.code == "invalid_argument" and "is of study S1 but holds rows of ['S9']" in r.error.message
    r = at.dispatch_analyst(registry, tool, {**t, "exclude": ["Beauregard"], "roster_artifact_ids": [plot_list(registry, "S1", drop=("studyDbId",))]})
    assert not r.ok and "has no column 'studyDbId'" in r.error.message


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_the_answer_prints_the_names_then_the_prefix(tmp_path, tool):
    """Finding 8: the answer with both lines, names first."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id=f"run_both_{tool}")
    r = at.dispatch_analyst(ctx.registry, tool, {**tables(ctx.registry, tool), "name_prefix": "PH17.", "exclude": ["Tanzania"]})
    assert r.ok, r.error
    shown = answer(ctx, claims(r))
    by_name, by_prefix = cp._text("Left out by name: Tanzania (1 entry)."), cp._text("Left out by prefix: 1 entry outside PH17.* (6 kept).")
    assert by_name in shown and by_prefix in shown and shown.index(by_name) < shown.index(by_prefix)


# --- the F4 gate, the prompts and the schema -----------------------------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.calls = []

    async def schemas(self):
        return []

    async def call(self, name, args):
        self.calls.append((name, dict(args)))
        return ToolResult(ok=True, data={}, artifact_ids=[])


def guarded(question):
    tools = an.GuardedAnalystTools(_Recorder(), ["art_1", "art_2", "art_3", "art_4"], question=question)
    tools.inspected.update({"art_1", "art_2", "art_3", "art_4"})
    return tools


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_the_f4_gate_is_unchanged_with_plot_lists(tool):
    """C27.3's real wording passes with all three names and the plot lists; a name the person never wrote is refused before the
    tool runs, plot lists or not; a plot list never supplied is refused like any table."""
    first, second = TOOLS[tool]
    base = {first: "art_1", second: "art_2", "roster_artifact_ids": ["art_3", "art_4"]}
    tools = guarded(C27_3)
    assert anyio.run(tools.call, tool, {**base, "exclude": ["Beauregard", "Tanzania", "CIP199062.1"]}).ok and len(tools.inner.calls) == 1
    tools = guarded(C27_3)
    r = anyio.run(tools.call, tool, {**base, "exclude": ["Beauregard", "Resisto"]})
    assert not r.ok and "exclude 'Resisto' is not written in the person's words" in r.error.message and tools.inner.calls == []
    tools = guarded(C27_3)
    r = anyio.run(tools.call, tool, {**base, "roster_artifact_ids": ["art_3", "art_9"], "exclude": ["Beauregard"]})
    assert not r.ok and r.error.code == "not_authorized" and "art_9" in r.error.message and tools.inner.calls == []


def test_on_a_plot_record_server_the_planner_is_told_to_skip_the_plot_lists(tmp_path):
    """Coach 2026-10-23 11:50 UTC: the recipe line stays for the other servers (test_the_recipes_fetch_and_pass_the_plot_lists keeps
    its assertion); on a server whose plot records are read, get_observations already keeps the entries without a value
    (manifest.entries_without_value), so the planner's message says to skip those plot lists: one fetch per study saved."""
    from llm import Budget, FakeModelClient, reply_text
    question = "Compare dry matter between 4614 and 4608 for all entries except Beauregard and Tanzania."
    on = co._user_message(question, None, [], base_url="https://sweetpotatobase.org/brapi/v2")
    off = co._user_message(question, None, [], base_url="https://cassavabase.org/brapi/v2")
    assert co.PLOT_RECORDS_PLANNER_NOTE in on and co.PLOT_RECORDS_PLANNER_NOTE not in off
    # the words (review of this change, finding 1): the link to the prompt, the scope, and every other plot-list use kept (finding 2)
    note = co.PLOT_RECORDS_PLANNER_NOTE
    assert note.startswith("This server's plot records are read") and "already keeps the entries a study has without a value" in note
    assert "for compare_trials_paired or stress_indices only, plan no get_observation_units for the entries the question names" in note
    assert "Every other get_observation_units step stays exactly as its recipe says" in note
    assert all(use in note for use in ("plot_design for anova_trial", "the plot or entry counts of a study", "the check entries",
                                        "plot_basis_h2's checks"))
    prompt = " ".join(co.COORDINATOR_SYSTEM_PROMPT.split())       # the exception sits in the compare_trials_paired recipe, after its line
    assert ("also get_observation_units of each study and pass the plot lists as roster_artifact_ids=[\"<each study's plot list>\"] "
            "(a named entry with no value of the trait is then named, not refused), unless the message says this server's plot "
            "records are read") in prompt
    model = FakeModelClient([reply_text(json.dumps({"status": "needs_clarification", "question": "Which trait?"}), model="mock-model")],
                            reported_model="mock-model")

    async def go():
        return await co.draft_plan(question, model=model, metadata=None, base_url="https://sweetpotatobase.org/brapi/v2",
                                   budget=Budget(), run_id="run_rc", log_dir=tmp_path / "logs")

    anyio.run(go)
    assert co.PLOT_RECORDS_PLANNER_NOTE in json.dumps(model.requests[0]["messages"])      # draft_plan hands the server on


def test_the_recipes_fetch_and_pass_the_plot_lists():
    """The coordinator plans each study's plot list when the question names entries to leave out of T4 or T5, and the analyst
    passes them; the argument is in both tools' schemas."""
    for prompt in (co.COORDINATOR_SYSTEM_PROMPT, an.ANALYST_SYSTEM_PROMPT):
        flat = " ".join(prompt.split())
        assert "roster_artifact_ids" in flat and "get_observation_units" in flat
    flat = " ".join(co.COORDINATOR_SYSTEM_PROMPT.split())
    assert "when the question names entries to leave out, also get_observation_units of each study" in flat
    for tool in TOOLS:
        assert "roster_artifact_ids" in at.ANALYST_TOOLS[tool].args_model.model_json_schema()["properties"]
