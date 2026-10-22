"""
tests/test_rank_entries.py - R1 (Bidhan, 2026-10-21): the highest, the lowest, the top k, or the rank of a named entry, for
one measured trait, is a fact the app may state, as an ordering of recorded values and not a recommendation. "Best",
"should I select", "advance" and "recommend" stay refused.

The rank_entries tool on a known small SYNTHETIC table, the menus that name it, the answer's ranking table and its
sentence, and the front door's wording. No network, no model, no real data.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import agents.analyst as an
import agents.coordinator as co
import analyst_tools as at
import brapi_mcp_server as srv
import chat_presenter as cp
import triage
from artifacts import ArtifactRegistry
from contracts import AnalysisReport, Claim, MeasurementMeta
from part2_support import SYNTHETIC_BASE, synthetic_context

# entry ID, name, two plot values. Means: G1 11, G2 21, G3 16, G4 31, G5 21 (tied with G2), G6 none (no valid value).
ENTRIES = [("G1", "Alpha", ["10", "12"]), ("G2", "Beta", ["20", "22"]), ("G3", "Gamma", ["15", "17"]),
           ("G4", "Check-1", ["30", "32"]), ("G5", "Delta", ["21", "21"]), ("G6", "Epsilon", ["", ""])]


def save(registry: ArtifactRegistry, *, complete: bool = True) -> str:
    rows = [{"observationDbId": f"O{g}{i}", "observationUnitDbId": f"P{g}{i}", "observationUnitName": f"plot {g}-{i}", "germplasmDbId": g,
             "germplasmName": name, "studyDbId": "S1", "observationVariableDbId": "V1", "observationVariableName": "SYNTHETIC yield",
             "observationTimeStamp": "", "value": value} for g, name, values in ENTRIES for i, value in enumerate(values)]
    columns = {c: srv.OBSERVATION_COLUMNS.get(c, "text") for c in rows[0]}
    return registry.save_table(rows, kind="observations", columns=columns, complete=complete, study_ids=["S1"], variable_ids=["V1"],
                               measurement=MeasurementMeta(trait="SYNTHETIC yield", unit="t/ha"),
                               quality_notes=["SYNTHETIC ranking table"]).artifact_id


@pytest.fixture
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "out", "run_rank", base_url=SYNTHETIC_BASE)


def rank(registry, art, **args):
    return at.dispatch_analyst(registry, "rank_entries", {"artifact_id": art, **args})


def stated(result) -> list[tuple[str, int, float]]:
    return [(e["entry"], e["rank"], e["mean"]) for e in result.data["stated"]]


def claims(result) -> list[Claim]:
    return [Claim.model_validate(c) for c in result.data["claims"]]


# --- 1. the ordering ---------------------------------------------------------------------------

def test_the_highest_entry_gets_a_mean_claim_and_a_rank_claim(registry):
    r = rank(registry, save(registry))
    assert r.ok and stated(r) == [("G4", 1, 31.0)] and r.data["entries_ranked"] == 5
    by_kind = {c.kind: c for c in claims(r)}
    assert set(by_kind) == {"mean", "rank"} and by_kind["mean"].value == 31.0 and by_kind["mean"].unit == "t/ha"
    ranked = by_kind["rank"]
    assert ranked.value == 1 and ranked.unit is None and ranked.denominator.n == 5 and ranked.filters == {"germplasmDbId": "G4"}
    assert "highest first" in ranked.transformation and "ties share a rank" in ranked.transformation
    assert "descriptive ordering of recorded values, not a recommendation" in ranked.transformation
    # G6 has no valid value: said plainly (review: "entries under min_independent_n=1" read as if they had too few plots)
    assert any(e.reason == "entries not ranked: fewer than 1 independent units" and e.count == 1 for e in ranked.exclusions)


def test_a_tie_at_the_cut_states_every_tied_entry(registry):
    r = rank(registry, save(registry), exclude=["Check-1"])            # Beta and Delta share the top mean, 21
    assert r.ok and stated(r) == [("G2", 1, 21.0), ("G5", 1, 21.0)] and r.data["entries_ranked"] == 4
    assert r.data["left_out"] == [{"entry": "G4", "label": "Check-1"}]
    rank_claims = [c for c in claims(r) if c.kind == "rank"]
    assert all(c.denominator.n == 4 for c in rank_claims)
    assert any(e.reason == "entries left out before ranking" and e.count == 1 for e in rank_claims[0].exclusions)


def test_the_lowest_first_and_top_k(registry):
    r = rank(registry, save(registry), order="lowest", top_k=2)
    assert r.ok and stated(r) == [("G1", 1, 11.0), ("G3", 2, 16.0)] and "lowest first" in claims(r)[0].transformation + claims(r)[1].transformation


def test_a_named_entry_gets_its_rank_wherever_it_falls(registry):
    art = save(registry)
    by_name = rank(registry, art, named=["Gamma"])
    assert by_name.ok and stated(by_name) == [("G4", 1, 31.0), ("G3", 4, 16.0)]   # 31, 21, 21, 16: competition ranking
    by_id = rank(registry, art, named=["G3"])
    assert stated(by_id) == stated(by_name)


def test_unknown_or_conflicting_names_are_refused_with_the_closest_names(registry):
    art = save(registry)
    typo = rank(registry, art, named=["Gama"])
    assert not typo.ok and typo.error.code == "invalid_argument" and "Gamma" in typo.error.message
    both = rank(registry, art, named=["Gamma"], exclude=["G3"])
    assert not both.ok and both.error.code == "invalid_argument" and "both named and left out" in both.error.message


def test_a_named_entry_without_a_valid_value_is_reported_not_ranked(registry):
    r = rank(registry, save(registry), named=["Epsilon"])
    assert r.ok and stated(r) == [("G4", 1, 31.0)]
    assert r.data["named_not_ranked"] == [{"entry": "G6", "label": "Epsilon", "reason": "fewer than 1 independent units: 0"}]
    assert not any(c.filters.get("germplasmDbId") == "G6" for c in claims(r)) and any("Epsilon" in w for w in r.warnings)


def test_a_name_column_and_an_incomplete_table_are_refused(registry):
    assert rank(registry, save(registry), by="germplasmName").error.code == "invalid_argument"
    partial = save(registry, complete=False)
    assert rank(registry, partial).error.code == "incomplete_data"
    allowed = rank(registry, partial, allow_incomplete=True)
    assert allowed.ok and all(c.transformation.find("INCOMPLETE ARTIFACT") >= 0 for c in claims(allowed))


def test_the_full_ranking_is_saved_with_every_entry(registry):
    r = rank(registry, save(registry), exclude=["Check-1"])
    rows = {row["group_id"]: row for row in registry.load(r.data["result_artifact_id"])[1]}
    assert set(rows) == {"G1", "G2", "G3", "G4", "G5", "G6"}
    assert (rows["G2"]["rank"], rows["G5"]["rank"], rows["G3"]["rank"], rows["G1"]["rank"]) == ("1", "1", "3", "4")
    assert rows["G4"]["left_out"] == "true" and rows["G4"]["rank"] == "" and rows["G6"]["rank"] == ""


# --- 2. the menus and prompts that must name it ----------------------------------------------------

def test_every_menu_and_prompt_names_the_tool():
    assert "rank_entries" in at.ANALYST_TOOL_NAMES and len(at.ANALYST_TOOL_NAMES) == 12     # threshold_count, correlate_entry_means, plot_design (R2, 2026-10-22)
    assert "rank_entries" in co.ANALYST_ACTIONS and "rank_entries(" in co.COORDINATOR_SYSTEM_PROMPT
    assert "twelve fixed tools" in an.ANALYST_SYSTEM_PROMPT and "rank_entries" in an.ANALYST_SYSTEM_PROMPT
    assert "rank_entries" in an.NO_CLAIM_NUDGE


# --- 3. the answer -------------------------------------------------------------------------------

def test_the_answer_shows_one_ranking_table_and_the_descriptive_sentence(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_rank_answer")
    r = rank(ctx.registry, save(ctx.registry), named=["Gamma"])
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claims(r), caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    shown = cp.answer_summary(view)
    assert "| Rank | Entry | Mean | n |" in shown
    top, gamma = shown.index("| 1 | G4 · Check-1 | 31 t/ha | 2 |"), shown.index("| 4 | G3 · Gamma | 16 t/ha | 2 |")
    assert top < gamma and "of 5 entries" in shown
    assert "Descriptive numbers only: an ordering of recorded values, not a recommendation." in shown
    assert "unit not stated" not in shown and shown.count("31 t/ha") == 1          # each mean once, inside the ranking table


# --- 4. the front door ---------------------------------------------------------------------------

def test_the_front_door_treats_an_ordering_of_recorded_values_as_a_data_question():
    labels = triage._LABELS
    assert "A request to recommend or choose varieties" in labels and "recommend, rank or choose" not in labels
    assert "where a named entry ranks" in labels
    assert "does not recommend" in triage.OUT_OF_SCOPE_RULE and "does not rank" not in triage.OUT_OF_SCOPE_RULE


def test_a_second_ranking_with_other_entries_left_out_gets_its_own_claim_ids(registry):
    """Leaving the checks out changes the ranks, so the two rankings must not share claim IDs (a report keeps both)."""
    art = save(registry)
    plain, without = rank(registry, art, named=["Beta"]), rank(registry, art, named=["Beta"], exclude=["Check-1"])
    first = {c.claim_id: c.value for c in claims(plain) if c.kind == "rank"}
    second = {c.claim_id: c.value for c in claims(without) if c.kind == "rank"}
    assert not set(first) & set(second)
    assert sorted(first.values()) == [1, 2] and sorted(second.values()) == [1, 1]     # Beta is 2nd with the check, 1st (tied) without


def test_a_ranking_is_never_hidden_behind_a_requested_mean(tmp_path):
    """The chat may show only a clearly requested mean; a 'highest mean' question has a mean and a rank, and both stay."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_rank_focus")
    r = rank(ctx.registry, save(ctx.registry))
    plan = SimpleNamespace(statistic=SimpleNamespace(kind="mean"), question="Which entry had the highest mean SYNTHETIC yield in study S1?")
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claims(r), caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[], plan=plan)
    shown = cp.answer_summary(view)
    assert "| 1 | G4 · Check-1 | 31 t/ha | 2 |" in shown and "Other calculated statistics remain" not in shown


# --- 5. review of 2026-10-21 ----------------------------------------------------------------------

def _table(registry, rows_spec, note):
    """rows_spec: (entry ID, entry name, plot ID, value) per observation."""
    rows = [{"observationDbId": f"O{i}", "observationUnitDbId": plot, "observationUnitName": f"plot {plot}", "germplasmDbId": g,
             "germplasmName": name, "studyDbId": "S1", "observationVariableDbId": "V1", "observationVariableName": "SYNTHETIC score",
             "observationTimeStamp": "", "value": value} for i, (g, name, plot, value) in enumerate(rows_spec)]
    columns = {c: srv.OBSERVATION_COLUMNS.get(c, "text") for c in rows[0]}
    return registry.save_table(rows, kind="observations", columns=columns, complete=True, study_ids=["S1"], variable_ids=["V1"],
                               measurement=MeasurementMeta(trait="SYNTHETIC score"), quality_notes=[f"SYNTHETIC {note}"]).artifact_id


def test_the_analyst_loop_keeps_the_ranking_claims(tmp_path):
    """Review: rank_entries was missing from the Analyst's NUMERIC_TOOLS and DERIVING_TOOLS, so a real run that ranked
    ended "failed" with no claim (and got the no-claim nudge). Through the real loop, with scripted model replies."""
    import json

    import anyio

    from llm import Budget, FakeModelClient, reply_text, reply_tools

    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_rank_loop")
    inputs = an.synthetic_inputs(ctx, "SYNTHETIC: which clone had the highest mean of V1 in S1?")
    observations = next(a for a in inputs.artifact_ids if ctx.registry.manifest(a).kind == "observations")
    final = json.dumps({"status": "completed", "payload": {"methods": ["rank_entries on the value column"], "caveats": [],
                                                           "narrative": "rank_entries states the order.", "key_claim_ids": []}})
    model = FakeModelClient([reply_tools([("table_info", json.dumps({"artifact_id": a})) for a in inputs.artifact_ids]),
                             reply_tools([("rank_entries", json.dumps({"artifact_id": observations, "top_k": 1}))]), reply_text(final)])

    async def go():
        return await an.run_analyst(inputs, model=model, ctx=ctx, budget=Budget(), run_id="run_rank_loop", log_dir=ctx.registry.out_root)

    report, _agent, _narrative = anyio.run(go)
    assert report.status == "completed" and model._replies == []                 # no nudge was needed
    assert sorted({c.kind for c in report.claims}) == ["mean", "rank"] and any(c.kind == "rank" and c.value == 1 for c in report.claims)
    assert report.result_artifact_ids                                          # the full ranking is a result artifact


def test_means_equal_at_the_shown_precision_share_a_rank(registry):
    """Review: 0.1 and 0.2 average to 0.15000000000000002, 0.15 and 0.15 to 0.15; both show as 0.15 and must share a rank."""
    art = _table(registry, [("A", "A", "P1", "0.1"), ("A", "A", "P2", "0.2"), ("B", "B", "P3", "0.15"), ("B", "B", "P4", "0.15"),
                            ("C", "C", "P5", "0.1"), ("C", "C", "P6", "0.1")], "float tie")
    r = rank(registry, art)
    assert r.ok and [(e["entry"], e["rank"]) for e in r.data["stated"]] == [("A", 1), ("B", 1)]


def test_an_id_that_is_also_another_entrys_name_is_refused(registry):
    art = _table(registry, [("G1", "G2", "P1", "5"), ("G2", "Two", "P2", "6"), ("G3", "Same", "P3", "7"), ("G4", "Same", "P4", "8")], "names")
    clash = rank(registry, art, named=["G2"])                         # G2 is an ID, and the name of G1
    assert not clash.ok and "the ID of one entry and the name of another" in clash.error.message
    shared = rank(registry, art, named=["Same"])                      # one name, two entries
    assert not shared.ok and "the name of 2 entries" in shared.error.message


def test_repeated_plots_need_a_rule_and_min_independent_n_leaves_thin_entries_out(registry):
    art = _table(registry, [("A", "A", "PA", "4"), ("A", "A", "PA", "6"), ("B", "B", "PB", "9")], "repeats")
    assert not rank(registry, art).ok                                 # plot PA was measured twice: refused without a rule
    averaged = rank(registry, art, duplicate_policy="mean_per_unit", top_k=2)
    assert averaged.ok and [(e["entry"], e["rank"], e["mean"]) for e in averaged.data["stated"]] == [("B", 1, 9.0), ("A", 2, 5.0)]
    thin = rank(registry, art, duplicate_policy="mean_per_unit", min_independent_n=2, top_k=2)
    assert thin.ok and thin.data["entries_ranked"] == 0 and {e["entry"] for e in thin.data["not_ranked"]} == {"A", "B"}


def test_a_mean_already_in_the_ranking_is_not_shown_again(tmp_path):
    """Review: with group_stats on the same table, the same entry mean was shown twice (ranking table and a mean block)."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_rank_dupe")
    art = save(ctx.registry)
    grouped = at.dispatch_analyst(ctx.registry, "group_stats", {"artifact_id": art, "by": "germplasmDbId", "min_independent_n": 1, "top_n": 1})
    both = claims(rank(ctx.registry, art)) + [Claim.model_validate(c) for c in grouped.data["claims"]]
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=both, caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    shown = cp.answer_summary(view)
    assert shown.count("31 t/ha") == 1 and "| 1 | G4 · Check-1 | 31 t/ha | 2 |" in shown


def test_judgement_words_stay_refused_and_advance_joins_them():
    """Reviews 1 and 2 of 2026-10-21: "should I advance" is refused by the planner too (R1). "highest-yielding" stays a judgement
    word in both checks: without it, advice riding on an ordering passed ("Which highest-yielding clone should we cross?",
    "...the highest-yielding clone, so it should be advanced"). Whether a plain ordering phrased that way may pass is Bidhan's."""
    assert co.find_candidates("Which clone should I advance from study S1?", None).decision_words == ["should i advance"]
    assert co.find_candidates("Which clone is best in study S1?", None).decision_words == ["best"]
    for advice in ("Which highest-yielding clone should we cross?", "What are the highest-yielding clones I should select?",
                   "Pick the highest-yielding clones to cross."):
        assert co.find_candidates(advice, None).decision_words, advice
    assert co.find_candidates("Which entry had the highest mean yield in study S1?", None).decision_words == []    # a plain ordering
    assert an._SUPERIORITY.search("Clone GB is the highest-yielding clone, so it should be advanced.")
    assert an._SUPERIORITY.search("G4 is the best clone.") and an._SUPERIORITY.search("G4 outperformed the checks.")
    assert "rank_entries, missing_report" in " ".join(co.COORDINATOR_SYSTEM_PROMPT.split())      # rule 4's list of analyst actions


def test_claims_keep_full_precision_and_ties_follow_the_shown_precision(registry):
    """Review 2: the 12-significant-digit tie rule must neither round the claims nor tie means the answer shows apart."""
    art = _table(registry, [("A", "A", "P1", "0.1"), ("A", "A", "P2", "0.2"), ("B", "B", "P3", "0.15"), ("B", "B", "P4", "0.15"),
                            ("C", "C", "P5", "1.0000000000001"), ("D", "D", "P6", "1"),                 # equal at 12 digits
                            ("E", "E", "P7", "1.000000001"), ("F", "F", "P8", "1.000000002")], "precision")   # apart at 12 digits
    low = rank(registry, art, order="lowest", top_k=2)
    assert [(e["entry"], e["rank"]) for e in low.data["stated"]] == [("A", 1), ("B", 1)]     # a float-noise tie, listed by entry ID
    means = {c.filters["germplasmDbId"]: c.value for c in claims(low) if c.kind == "mean"}
    assert means == {"A": 0.15000000000000002, "B": 0.15}                                    # the claims keep full precision
    by_entry = {e["entry"]: e["rank"] for e in rank(registry, art, named=["C", "D", "E", "F"]).data["stated"]}
    assert by_entry["C"] == by_entry["D"] and by_entry["F"] < by_entry["E"]


def test_the_tools_prefix_is_not_shown_in_the_data_limits():
    note = "rank_entries: Measurement dates are not recorded for 4 of 4 valid source rows."
    assert cp._human_note(note) == "Measurement dates are not recorded for 4 of 4 valid source rows."
