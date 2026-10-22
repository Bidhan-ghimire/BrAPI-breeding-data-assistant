"""
tests/test_names_in_claims.py - what the Coordinator sees of the claims: entry names from the fetched tables.

Benchmark 2026-10-21 (Luna): the Coordinator saw at most 20 claims, carrying entry IDs but no names, and stopped A41.3 ("does
not identify Chingova by name ... 26 claims were omitted") and C33.5 ("do not identify the top genotype by name"). Coach
2026-10-22 00:50 UTC (answer 2): names come only from fetched tables (the plot records or the plot list in the run's ledger),
never from the model or the question; claims about entries named in the question come first; a name absent from the fetched
tables is never shown. Review of 2026-10-22: the named claims must not push the overall totals out, names are matched and
shown whole, IDs are exact text. SYNTHETIC tables only.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import brapi_mcp_server as srv
import coordination_runtime as cr
from artifacts import ArtifactRegistry
from contracts import AnalysisReport, Claim, Denominator, MeasurementMeta, RetrievalReport
from part2_support import SYNTHETIC_BASE

QUESTION = "In study S1, what was the mean yield of the check Chingova, and where did it rank among the genotype means?"


def table(registry, entries, *, study="S1", kind="observations", extra=None):
    rows = [{"observationDbId": f"O{study}{g}{i}", "observationUnitDbId": f"P{study}{g}{i}", "observationUnitName": f"plot {g}-{i}",
             "germplasmDbId": g, "germplasmName": name, "studyDbId": study, "observationVariableDbId": "V1",
             "observationVariableName": "SYNTHETIC yield", "observationTimeStamp": "", "value": "10", **(extra or {})}
            for g, name in entries for i in range(2)]
    columns = {c: srv.OBSERVATION_COLUMNS.get(c, "text") for c in rows[0]}
    return registry.save_table(rows, kind=kind, columns=columns, complete=True, study_ids=[study], variable_ids=["V1"],
                               measurement=MeasurementMeta(trait="SYNTHETIC yield", unit="t/ha"))


def mean(number, entry, sha, *, column="germplasmDbId", more=None):
    return Claim(claim_id=f"clm_names_{number:03d}", kind="mean", value=10.0 + number, unit="t/ha",
                 denominator=Denominator(name="plots", n=2), source_artifact_hashes=[sha], filters={column: entry, **(more or {})},
                 grouping=[column], transformation="mean of valid values")


def total(number, sha):
    return Claim(claim_id=f"clm_total_{number:03d}", kind="mean", value=12.0, unit="t/ha", denominator=Denominator(name="plots", n=50),
                 source_artifact_hashes=[sha], transformation="mean of valid values")


def controller(registry, fetched, claims, *, question=QUESTION):
    return SimpleNamespace(plan=None, config=SimpleNamespace(question=question), ctx=SimpleNamespace(registry=registry),
                           retrieval=RetrievalReport(status="completed", artifacts=fetched, complete=True),
                           data_review={"status": "approved"}, analysis=AnalysisReport(status="completed", claims=claims),
                           supervision={"decisions": [], "plan_revisions": 0, "analysis_passes": 1})


def view(registry, fetched, claims, **kw):
    return cr._state(controller(registry, fetched, claims, **kw), "analyzed")["analysis"]


def names_of(v):
    return {next(iter(c["filters"].values())) if c["filters"] else "total": c.get("names_from_fetched_tables") for c in v["claims"]}


@pytest.fixture
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "out", "run_names", base_url=SYNTHETIC_BASE)


# --- 1. the names come from the fetched tables only -------------------------------------------------------------

def test_the_coordinator_sees_the_names_beside_the_ids_in_the_fetched_tables(registry):
    fetched = table(registry, [("G1", "Alpha"), ("G2", "Beta"), ("G3", "Chingova")])
    v = view(registry, [fetched], [mean(n, g, fetched.sha256) for n, g in enumerate(("G1", "G2", "G3"))])
    assert names_of(v) == {"G1": {"germplasmDbId G1": "Alpha"}, "G2": {"germplasmDbId G2": "Beta"}, "G3": {"germplasmDbId G3": "Chingova"}}


def test_each_id_column_shows_the_name_column_beside_it(registry):
    """The five pairs (review: dropping one pair, or pairing a unit ID with the germplasm name, went unnoticed)."""
    extra = {"locationDbId": "L1", "locationName": "Ica"}
    fetched = table(registry, [("G1", "Alpha")], extra=extra)
    pairs = {"germplasmDbId": ("G1", "Alpha"), "observationUnitDbId": ("PS1G10", "plot G1-0"), "studyDbId": ("S1", None),
             "locationDbId": ("L1", "Ica"), "observationVariableDbId": ("V1", "SYNTHETIC yield")}
    claims = [mean(n, key, fetched.sha256, column=column) for n, (column, (key, _)) in enumerate(pairs.items())]
    shown = {c["grouping"][0]: c.get("names_from_fetched_tables") for c in view(registry, [fetched], claims)["claims"]}
    assert shown == {column: ({f"{column} {key}": name} if name else None) for column, (key, name) in pairs.items()}


def test_a_name_absent_from_the_fetched_tables_is_never_shown(registry):
    """The question names Chingova, but no fetched table holds it. Tables the Retriever did not return are not read: a derived
    table, and a registered OBSERVATIONS table outside the retrieval (review: only the derived kind was tested). An ID no
    fetched table holds gets no name."""
    fetched = table(registry, [("G1", "Alpha"), ("G2", "Beta")])
    derived = table(registry, [("G1", "Fake")], kind="analysis_result")
    stray = table(registry, [("G9", "Chingova"), ("G2", "Stray")], study="S7")
    claims = [mean(1, "G1", fetched.sha256), mean(2, "G2", fetched.sha256), mean(3, "G9", derived.sha256), mean(4, "G9", stray.sha256)]
    v = view(registry, [fetched], claims)
    shown = json.dumps(v["claims"])
    assert "Chingova" not in shown and "Fake" not in shown and "Stray" not in shown
    assert names_of(v) == {"G1": {"germplasmDbId G1": "Alpha"}, "G2": {"germplasmDbId G2": "Beta"}, "G9": None}


def test_ids_are_exact_text_and_names_are_trimmed(registry):
    """Review: "G1" and "G1 " (a trailing space) are two IDs; a name cell " Alpha " is shown trimmed."""
    fetched = table(registry, [("G1", " Alpha "), ("G1 ", "Beta")])
    v = view(registry, [fetched], [mean(1, "G1", fetched.sha256), mean(2, "G1 ", fetched.sha256)])
    assert [c["names_from_fetched_tables"] for c in v["claims"]] == [{"germplasmDbId G1": "Alpha"}, {"germplasmDbId G1 ": "Beta"}]


def test_a_claim_with_two_ids_shows_the_name_each_has(registry):
    fetched = table(registry, [("G1", "Alpha")])
    v = view(registry, [fetched], [mean(1, "G1", fetched.sha256, more={"locationDbId": "L9"})])
    assert v["claims"][0]["names_from_fetched_tables"] == {"germplasmDbId G1": "Alpha"}


def test_names_are_shown_whole_and_the_rest_counted(registry):
    """Review: the 200-character cut went through a name. Whole names up to the limit, then "(+N more)"; a single name longer
    than the limit is cut to it."""
    first, second = table(registry, [("G1", "Alpha")]), table(registry, [("G1", "Alpha-2" + "x" * 300), ("G2", "Y" * 300)], study="S2")
    v = view(registry, [first, second], [mean(1, "G1", first.sha256), mean(2, "G2", second.sha256)])
    assert v["claims"][0]["names_from_fetched_tables"] == {"germplasmDbId G1": "Alpha (+1 more)"}
    assert v["claims"][1]["names_from_fetched_tables"] == {"germplasmDbId G2": "Y" * cr.NAME_LIMIT}


# --- 2. claims about entries the question names come first -------------------------------------------------------

def test_claims_about_entries_the_question_names_come_first(registry):
    """25 genotype means and a total: the usual 20 cut Chingova (G23) off. It comes first now, then the usual 20 (the total
    first); the omitted count says what is left."""
    entries = [(f"G{n:02d}", f"Clone {n:02d}") for n in range(25)]
    entries[23] = ("G23", "Chingova")
    fetched = table(registry, entries)
    claims = [mean(n, g, fetched.sha256) for n, (g, _) in enumerate(entries)] + [total(1, fetched.sha256)]
    v = view(registry, [fetched], claims, question=QUESTION.replace("Chingova", "chingova"))
    order = [next(iter(c["filters"].values()), "total") if c["filters"] else "total" for c in v["claims"]]
    assert order[:3] == ["G23", "total", "G00"] and len(order) == 21 and order.count("G23") == 1 and v["claims_omitted"] == 5


def test_naming_many_entries_never_pushes_the_overall_totals_out(registry):
    """Review: ten named entries with three claims each pushed the total out of one list of 20."""
    entries = [(f"G{n:02d}", f"Clone {n:02d}") for n in range(12)]
    fetched = table(registry, entries)
    claims = [mean(10 * n + k, g, fetched.sha256) for n, (g, _) in enumerate(entries) for k in range(3)] + [total(1, fetched.sha256)]
    question = "Compare " + ", ".join(name for _, name in entries[:10]) + "."
    v = view(registry, [fetched], claims, question=question)
    shown = [c["filters"].get("germplasmDbId", "total") for c in v["claims"]]
    assert shown[20] == "total" and len(shown) == 37 and v["claims_omitted"] == 0     # 20 named, then the other 17, the total first
    assert set(shown[:20]) <= {g for g, _ in entries[:10]}


def test_every_fetched_name_of_an_id_counts_when_the_question_names_it(registry):
    """Review: with 11 names on one ID the cut dropped Chingova from the matching too; the second name of an ID counts."""
    many = [("G7", f"Aa{n:02d}") for n in range(10)] + [("G7", "Chingova")]
    fetched = table(registry, [("G1", "Alpha")] + many)
    v = view(registry, [fetched], [mean(1, "G1", fetched.sha256), mean(2, "G7", fetched.sha256)])
    assert [c["filters"]["germplasmDbId"] for c in v["claims"]][0] == "G7"


@pytest.mark.parametrize("question, first", [("How did Ana yield?", "G02"), ("How did Chingovas yield?", "G01"),
                                             ("How did xChingova yield?", "G01"), ("How did Ex yield?", "G01"),
                                             ("How did G23 yield?", "G01"), ("How did Chingova yield?", "G23")])
def test_only_a_whole_fetched_name_of_three_letters_or_more_puts_a_claim_first(registry, question, first):
    """A name inside a longer word, on either side; a name shorter than three letters; an ID typed in the question: nothing moves.
    A three-letter name does."""
    fetched = table(registry, [("G01", "Alpha"), ("G02", "Ana"), ("G03", "Ex"), ("G23", "Chingova")])
    claims = [mean(n, g, fetched.sha256) for n, g in enumerate(("G01", "G02", "G03", "G23"))]
    assert view(registry, [fetched], claims, question=question)["claims"][0]["filters"]["germplasmDbId"] == first


def test_a_name_holding_a_semicolon_is_matched_whole(registry):
    """Review: a name "zzz; Chingova" was split, and its fragment matched the question."""
    fetched = table(registry, [("G1", "Alpha"), ("G2", "zzz; Chingova")])
    v = view(registry, [fetched], [mean(1, "G1", fetched.sha256), mean(2, "G2", fetched.sha256)])
    assert [c["filters"]["germplasmDbId"] for c in v["claims"]] == ["G1", "G2"]


def test_no_fetched_tables_no_names_and_the_order_as_before():
    claims = [Claim(claim_id=f"clm_plain_{n}", kind="count", value=1, denominator=Denominator(name="rows", n=3),
                    source_artifact_hashes=["a" * 64], filters={"germplasmDbId": f"G{n}"}, grouping=["germplasmDbId"],
                    transformation="frequency of exact text values") for n in range(3)]
    plain = SimpleNamespace(plan=None, retrieval=None, data_review={"status": "approved"},
                            analysis=AnalysisReport(status="completed", claims=claims),
                            supervision={"decisions": [], "plan_revisions": 0, "analysis_passes": 1})
    v = cr._state(plain, "analyzed")["analysis"]
    assert [c["claim_id"] for c in v["claims"]] == ["clm_plain_0", "clm_plain_1", "clm_plain_2"] and v["claims_omitted"] == 0
    assert not any("names_from_fetched_tables" in c for c in v["claims"])


# --- 3. after the second review (2026-10-22): pins of what the code already does -------------------------------------

def test_an_unreadable_fetched_table_is_skipped_and_the_others_are_still_read(registry):
    from artifacts import ArtifactError

    good, bad = table(registry, [("G1", "Alpha")]), table(registry, [("G2", "Beta")], study="S2")
    load = registry.load
    registry.load = lambda aid: (_ for _ in ()).throw(ArtifactError("cannot read")) if aid == bad.artifact_id else load(aid)
    v = view(registry, [bad, good], [mean(1, "G1", good.sha256), mean(2, "G2", bad.sha256)])
    assert names_of(v) == {"G1": {"germplasmDbId G1": "Alpha"}, "G2": None}


def test_an_incomplete_fetched_table_still_names_what_it_holds(registry):
    rows = [{"observationDbId": "O1", "observationUnitDbId": "P1", "observationUnitName": "plot 1", "germplasmDbId": "G1",
             "germplasmName": "Alpha", "studyDbId": "S1", "observationVariableDbId": "V1", "observationVariableName": "SYNTHETIC yield",
             "observationTimeStamp": "", "value": "10"}]
    partial = registry.save_table(rows, kind="observations", columns={c: "text" for c in rows[0]}, complete=False, study_ids=["S1"],
                                  variable_ids=["V1"], measurement=MeasurementMeta(trait="SYNTHETIC yield", unit="t/ha"))
    assert names_of(view(registry, [partial], [mean(1, "G1", partial.sha256)])) == {"G1": {"germplasmDbId G1": "Alpha"}}


def test_a_large_table_and_a_long_question_are_read_to_their_end(registry):
    entries = [(f"G{n:04d}", f"Clone {n:04d}") for n in range(800)]
    entries[790] = ("G0790", "Chingova")
    fetched = table(registry, entries)                                   # 1,600 rows
    question = "Please compare " + "the trial means, " * 20 + "and tell me about Chingova."
    assert len(question) > 300
    v = view(registry, [fetched], [mean(n, g, fetched.sha256) for n, (g, _) in enumerate(entries[:5])] + [mean(9, "G0790", fetched.sha256)],
             question=question)
    assert v["claims"][0]["filters"]["germplasmDbId"] == "G0790" and v["claims"][0]["names_from_fetched_tables"] == {"germplasmDbId G0790": "Chingova"}


def test_a_name_with_a_bar_or_a_comma_is_shown_whole_and_a_blank_id_gets_none(registry):
    extra = {"observationVariableName": "Total storage roots yield computation tons per ha|CO_331:0000296"}
    fetched = table(registry, [("G1", "Clone, A"), ("   ", "Blank")], extra=extra)
    claims = [mean(1, "V1", fetched.sha256, column="observationVariableDbId"), mean(2, "G1", fetched.sha256), mean(3, "   ", fetched.sha256)]
    shown = [c.get("names_from_fetched_tables") for c in view(registry, [fetched], claims)["claims"]]
    assert shown == [{"observationVariableDbId V1": "Total storage roots yield computation tons per ha|CO_331:0000296"},
                     {"germplasmDbId G1": "Clone, A"}, None]


def test_names_that_fill_the_limit_exactly_are_shown_whole():
    names = {"A" * 100, "B" * 98}                                         # 100 + "; " + 98 = 200 characters
    assert cr._shown_names(names) == "A" * 100 + "; " + "B" * 98
    assert cr._shown_names(names | {"C"}) == "A" * 100 + "; " + "B" * 98 + " (+1 more)"


@pytest.mark.parametrize("seed", range(40))
def test_the_view_always_holds_the_old_view_and_lists_named_claims_first_in_the_usual_order(seed):
    """What the Coordinator saw before this change (the first 20 in the usual order) is always still listed; nothing twice."""
    import random

    rng = random.Random(seed)
    claims = []
    for n in range(rng.randrange(1, 120)):
        filters = {} if rng.random() < 0.1 else {"germplasmDbId": f"G{rng.randrange(60)}",
                                                  **({"observationVariableDbId": "V1"} if rng.random() < 0.3 else {})}
        claims.append(Claim(claim_id=f"clm_prop_{seed}_{n:03d}", kind="mean", value=float(n), unit="t/ha",
                            denominator=Denominator(name="plots", n=2), source_artifact_hashes=["a" * 64], filters=filters,
                            grouping=list(filters), transformation="mean of valid values"))
    named = {("germplasmDbId", f"G{k}") for k in rng.sample(range(60), rng.randrange(0, 30))}
    shown = cr._view_claims(claims, named)
    assert all(any(old is new for new in shown) for old in cr._overall_first(claims)[:cr.VIEW_CLAIMS])
    assert len({id(c) for c in shown}) == len(shown) <= 2 * cr.VIEW_CLAIMS
    about = [c for c in cr._overall_first(claims) if any(item in named for item in c.filters.items())][:cr.VIEW_CLAIMS]
    assert shown[:len(about)] == about
