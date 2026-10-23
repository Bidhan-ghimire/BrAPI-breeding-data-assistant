"""
tests/test_stress_indices.py - T5 "stress_indices" (coach 2026-10-23 03:48 UTC, NEXT 3; research build order T5): ONE trait in a
non-stress (Yp) and a stress (Ys) study, on the entries with a mean in both: per entry TOL, MP, GMP, HM, SSI, YSI, YI, STI and
RDI, the population means and the stress intensity SI, and the entries ranked by one index. Descriptive numbers, never a choice.

Fixtures: SYNTHETIC tables checked against an independent numpy computation (numpy only in this test). The research notes'
data_ge E3 / E11 table runs outside the repo (item (16)). No network, no model.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

import agents.analyst as an
import agents.coordinator as co
import analyst_tools as at
import brapi_mcp_server as srv
import chat_presenter as cp
from artifacts import ArtifactRegistry
from contracts import AnalysisReport, Claim, MeasurementMeta, Statistic
from part2_support import SYNTHETIC_BASE, synthetic_context

ENTRIES = [f"E{i}" for i in range(8)]


def trial(level: float, *, reps: int = 2, entries=ENTRIES, seed: float = 0.0) -> dict[str, list[float]]:
    """entry -> plot values, all positive: a level, an entry effect and an irregular plot error (no randomness)."""
    return {e: [level * (1.0 + 0.3 * math.cos(1.1 * i + seed)) + 0.2 * math.sin(2.3 * i + 1.7 * k + 3.1 * seed) for k in range(reps)]
            for i, e in enumerate(entries)}


@pytest.fixture
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "out", "run_stress", base_url=SYNTHETIC_BASE)


def rows_of(study: str, values: dict[str, list[float]], *, names=None, variable: str = "V1") -> list[dict[str, str]]:
    return [{"observationDbId": f"O{study}-{entry}-{k}", "observationUnitDbId": f"{study}-{entry}-{k}", "observationUnitName": f"plot {k}",
             "germplasmDbId": entry, "germplasmName": (names or {}).get(entry, f"clone {entry}"), "studyDbId": study,
             "observationVariableDbId": variable, "observationVariableName": "SYNTHETIC yield", "observationTimeStamp": "",
             "value": repr(float(value))}
            for entry, plot_values in values.items() for k, value in enumerate(plot_values, 1)]


def save(registry, rows, *, complete=True) -> str:
    columns = {col: srv.OBSERVATION_COLUMNS.get(col, "text") for col in rows[0]}
    return registry.save_table(rows, kind="observations", columns=columns, complete=complete, study_ids=sorted({r["studyDbId"] for r in rows}),
                               variable_ids=sorted({r["observationVariableDbId"] for r in rows}),
                               measurement=MeasurementMeta(trait="SYNTHETIC yield", unit="t/ha"), quality_notes=["SYNTHETIC"]).artifact_id


def pair(registry, nonstress, stress, **args):
    return at.dispatch_analyst(registry, "stress_indices", {"nonstress_artifact_id": nonstress, "stress_artifact_id": stress, **args})


def two(registry, p, s, **args):
    return pair(registry, save(registry, rows_of("S1", p)), save(registry, rows_of("S2", s)), **args)


def claims(r) -> list[Claim]:
    return [Claim.model_validate(k) for k in r.data["claims"]]


def per_entry(registry, r) -> dict[str, dict[str, str]]:
    return {row["entry"]: row for row in at._load(registry, r.data["result_artifact_id"]).rows}


def expected(p, s):
    """Every index by an independent numpy computation, on the entries in both trials."""
    np = pytest.importorskip("numpy")
    paired = sorted(set(p) & set(s))
    yp = np.array([np.mean(p[e]) for e in paired])
    ys = np.array([np.mean(s[e]) for e in paired])
    mp, ms = yp.mean(), ys.mean()
    si = 1 - ms / mp
    values = {"TOL": yp - ys, "MP": (yp + ys) / 2, "GMP": np.sqrt(yp * ys), "HM": 2 * yp * ys / (yp + ys), "SSI": (1 - ys / yp) / si,
              "YSI": ys / yp, "YI": ys / ms, "STI": yp * ys / mp ** 2, "RDI": (ys / yp) / (ms / mp)}
    return paired, dict(zip(paired, yp)), dict(zip(paired, ys)), float(mp), float(ms), float(si), \
        {index: {e: float(v) for e, v in zip(paired, column)} for index, column in values.items()}


# --- 1. the numbers against numpy ------------------------------------------------------------------------------------

def test_every_index_matches_an_independent_numpy_computation(registry):
    p, s = trial(10.0), trial(4.0, seed=0.4)
    r = two(registry, p, s)
    assert r.ok, r.error
    paired, yp, ys, mp, ms, si, idx = expected(p, s)
    d = r.data
    assert d["n_pairs"] == 8 and d["mean_yp"] == pytest.approx(mp, rel=1e-12) and d["mean_ys"] == pytest.approx(ms, rel=1e-12)
    assert d["si"] == pytest.approx(si, rel=1e-12) and (d["nonstress_study"], d["stress_study"]) == ("S1", "S2")
    rows = per_entry(registry, r)
    for index in at.STRESS_INDICES:
        for e in paired:
            assert float(rows[e][index]) == pytest.approx(idx[index][e], rel=1e-12), (index, e)
    assert all(float(rows[e]["yp"]) == pytest.approx(yp[e], rel=1e-12) and float(rows[e]["ys"]) == pytest.approx(ys[e], rel=1e-12) for e in paired)


@pytest.mark.parametrize("scale", [1e180, 1e-160])
def test_huge_and_tiny_values_neither_overflow_nor_underflow(registry, scale):
    """Yp x Ys of 1e180 values would overflow and of 1e-160 values underflow: the ratios come out the same, the rest scale."""
    p, s = trial(10.0), trial(4.0, seed=0.4)
    plain = per_entry(registry, two(registry, p, s))
    r = pair(registry, save(registry, rows_of("S3", {e: [v * scale for v in vs] for e, vs in p.items()})),
             save(registry, rows_of("S4", {e: [v * scale for v in vs] for e, vs in s.items()})))
    assert r.ok, r.error
    big = per_entry(registry, r)
    for e in ENTRIES:
        for index in ("STI", "YSI", "YI", "SSI", "RDI"):
            assert float(big[e][index]) == pytest.approx(float(plain[e][index]), rel=1e-9), (index, e)
        for index in ("GMP", "MP", "HM", "TOL"):
            assert float(big[e][index]) == pytest.approx(float(plain[e][index]) * scale, rel=1e-9), (index, e)


# --- 2. the ranking -------------------------------------------------------------------------------------------------

def test_the_highest_sti_entries_are_stated_with_their_means_and_ranks(registry):
    p, s = trial(10.0), trial(4.0, seed=0.4)
    r = two(registry, p, s)
    paired, yp, ys, mp, ms, si, idx = expected(p, s)
    top = sorted(paired, key=lambda e: -idx["STI"][e])[:3]
    assert [x["entry"] for x in r.data["stated"]] == top and [x["rank"] for x in r.data["stated"]] == [1, 2, 3]
    assert r.data["stated"][0]["value"] == pytest.approx(idx["STI"][top[0]], rel=1e-12)
    heads = {c.transformation.split(" | ")[0]: c for c in claims(r) if c.filters == {"germplasmDbId": top[0]}}
    assert heads["STI of the entry"].kind == "ratio" and heads["STI of the entry"].value == pytest.approx(idx["STI"][top[0]], rel=1e-12)
    assert "STI = Yp x Ys / (mean Yp)^2 (Fernandez 1992)" in heads["STI of the entry"].transformation
    assert heads["rank by STI, highest first"].kind == "rank" and heads["rank by STI, highest first"].value == 1
    assert heads["the entry's non-stress mean Yp"].value == pytest.approx(yp[top[0]], rel=1e-12) and heads["the entry's non-stress mean Yp"].unit == "t/ha"
    assert heads["the entry's stress mean Ys"].value == pytest.approx(ys[top[0]], rel=1e-12)
    assert all(at.STRESS_NOTE in c.transformation for c in claims(r))
    assert claims(r)[0].filters == {"germplasmDbId": top[0]}                    # the top entry first: the coordinator sees 20 claims


def test_tied_entries_share_a_rank_and_are_stated_together(registry):
    r = two(registry, {"E0": [10.0], "E1": [10.0], "E2": [8.0], "E3": [6.0]}, {"E0": [5.0], "E1": [5.0], "E2": [4.0], "E3": [3.0]}, top_k=1)
    assert r.ok, r.error
    assert [(x["entry"], x["rank"]) for x in r.data["stated"]] == [("E0", 1), ("E1", 1)]


def test_another_index_lowest_first(registry):
    p, s = trial(10.0), trial(4.0, seed=0.4)
    r = two(registry, p, s, rank_by="TOL", order="lowest", top_k=2)
    paired, yp, ys, mp, ms, si, idx = expected(p, s)
    assert [x["entry"] for x in r.data["stated"]] == sorted(paired, key=lambda e: idx["TOL"][e])[:2]
    tol = next(c for c in claims(r) if c.transformation.startswith("TOL of the entry"))
    assert tol.kind == "mean" and tol.unit == "t/ha" and "TOL = Yp - Ys (Rosielle and Hamblin 1981)" in tol.transformation
    assert {c.claim_id for c in claims(r)}.isdisjoint({c.claim_id for c in claims(two(registry, p, s))})


# --- 3. numbers that are not defined --------------------------------------------------------------------------------

def test_an_entry_whose_non_stress_mean_is_0_has_no_ratio_to_it(registry):
    p, s = {"E0": [0.0], "E1": [10.0], "E2": [8.0], "E3": [6.0]}, {"E0": [2.0], "E1": [5.0], "E2": [4.0], "E3": [3.0]}
    r = two(registry, p, s, rank_by="YSI")
    assert r.ok, r.error
    rows = per_entry(registry, r)
    assert rows["E0"]["YSI"] == "" and rows["E0"]["SSI"] == "" and rows["E0"]["RDI"] == "" and "Yp is 0" in rows["E0"]["note"]
    assert float(rows["E0"]["STI"]) == 0.0 and float(rows["E0"]["GMP"]) == 0.0 and rows["E0"]["rank"] == ""
    assert "E0" not in [x["entry"] for x in r.data["stated"]]
    assert r.data["not_ranked"] == [{"entry": "E0", "label": "clone E0", "reason": "YSI is undefined: Yp is 0"}]


def test_equal_population_means_leave_ssi_undefined(registry):
    p, s = {"E0": [10.0], "E1": [6.0], "E2": [8.0]}, {"E0": [6.0], "E1": [10.0], "E2": [8.0]}         # both means 8.0
    r = two(registry, p, s)
    assert r.ok, r.error
    assert r.data["si"] == 0 and all(row["SSI"] == "" for row in per_entry(registry, r).values())
    refused = two(registry, p, s, rank_by="SSI")
    assert not refused.ok and refused.error.code == "unsupported" and "SSI" in refused.error.message and "SI is 0" in refused.error.message


def test_a_mean_of_float_residue_counts_as_0(registry):
    """T4's residue rule: plot values that cancel (0.1 + 0.2 - 0.3) leave a mean of 1e-17, which is 0, not a tiny Yp."""
    p = {"E0": [0.1, 0.2, -0.3], "E1": [10.0, 10.0, 10.0], "E2": [8.0, 8.0, 8.0]}
    s = {"E0": [2.0, 2.0, 2.0], "E1": [5.0, 5.0, 5.0], "E2": [4.0, 4.0, 4.0]}
    rows = per_entry(registry, two(registry, p, s))
    assert rows["E0"]["YSI"] == "" and float(rows["E0"]["STI"]) == 0.0 and float(rows["E0"]["GMP"]) == 0.0


def test_a_population_mean_at_float_noise_counts_as_0(registry):
    """Deliberate bug S17 survived: the non-stress population mean is judged against the largest plot value, not against exact
    0. One entry's 1e-15 beside entries whose plots cancel to 0 (1.0 and -1.0) gives a mean Yp of 3.3e-16, which is 0 at the
    data's precision: STI, RDI and SI are undefined with the reason. (An entry mean is judged against the entry's own plots, so
    E0's own ratios such as YSI stay numbers: review of T5, finding 5, made-up input.)"""
    p = {"E0": [1e-15, 1e-15], "E1": [1.0, -1.0], "E2": [1.0, -1.0]}
    s = {"E0": [1.0, 1.0], "E1": [2.0, 2.0], "E2": [3.0, 3.0]}
    r = two(registry, p, s, rank_by="TOL")
    assert r.ok, r.error
    rows = per_entry(registry, r)
    assert r.data["si"] is None and all(rows[e]["STI"] == "" and rows[e]["RDI"] == "" for e in p)
    assert all("STI: mean Yp is 0" in rows[e]["note"] for e in p)


def test_negative_entry_means_are_refused(registry):
    r = two(registry, {"E0": [-1.0], "E1": [2.0], "E2": [3.0]}, {"E0": [1.0], "E1": [1.5], "E2": [2.5]})
    assert not r.ok and r.error.code == "unsupported" and "ratio scale" in r.error.message


# --- 4. the pairing is compare_trials_paired's ----------------------------------------------------------------------

def test_the_pairing_rules_are_compare_trials_paireds(registry):
    names = {e: (f"PH17.{e}" if i < 5 else f"check {e}") for i, e in enumerate(ENTRIES)}
    r = pair(registry, save(registry, rows_of("S1", trial(10.0), names=names)), save(registry, rows_of("S2", trial(4.0, seed=0.4), names=names)),
             exclude=["check E5"], name_prefix="PH17.")
    assert r.ok, r.error
    assert r.data["n_pairs"] == 5 and r.data["left_out"] == {"excluded": 1, "outside the name prefix": 2} and r.data["excluded_named"] == ["E5"]
    few = two(registry, trial(10.0, entries=ENTRIES[:4]), trial(4.0, entries=ENTRIES[2:]))
    assert not few.ok and "2 entries have a mean in both trials" in few.error.message and "stress indices need at least 3" in few.error.message
    tid = save(registry, rows_of("S1", trial(10.0)))
    assert not pair(registry, tid, tid).ok
    both = save(registry, rows_of("S1", trial(10.0)) + rows_of("S2", trial(4.0)))
    two_studies = pair(registry, both, save(registry, rows_of("S3", trial(4.0))))
    assert not two_studies.ok and "the non-stress table" in two_studies.error.message and "filter_rows" in two_studies.error.message


# --- 5. the answer and the wiring -----------------------------------------------------------------------------------

def answer(ctx, claim_list) -> str:
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claim_list, caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    return cp.answer_summary(view)


def test_the_answer_shows_the_top_entries_by_name_the_population_and_the_sentence(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_stress_answer")
    r = pair(ctx.registry, save(ctx.registry, rows_of("S1", trial(10.0))), save(ctx.registry, rows_of("S2", trial(4.0, seed=0.4))))
    shown = answer(ctx, claims(r))
    top = r.data["stated"][0]["entry"]
    assert "**Stress indices of entry means, 8 entries in both trials**" in shown
    assert "| Rank | Entry | STI | Yp | Ys |" in shown and f"| 1 | {top} · clone {top} |" in shown
    assert "Non-stress (Yp): study S1" in shown and "stress intensity SI " in shown and "STI = Yp x Ys / (mean Yp)^2 (Fernandez 1992)" in shown
    assert cp.STRESS_SENTENCE in shown and "**Ranking by entry mean" not in shown and "| Statistic | Result |" not in shown
    shown.encode("cp1252")


def test_every_number_the_answer_shows_is_backed_by_a_claim(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_stress_backed")
    r = pair(ctx.registry, save(ctx.registry, rows_of("S1", trial(10.0))), save(ctx.registry, rows_of("S2", trial(4.0, seed=0.4))))
    shown = answer(ctx, claims(r))
    spec = importlib.util.spec_from_file_location("_bench_for_stress", Path(at.__file__).resolve().parent / "eval" / "benchmark.py")
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    flagged = [n for n in bench._numbers_for_invention(bench.harness.code_written(shown))
               if not bench._backed(n[0], n[1], bench.claim_numbers([k.model_dump(mode="json") for k in claims(r)]))]
    assert flagged == []


def test_every_menu_and_prompt_names_the_tool():
    assert "stress_indices" in at.ANALYST_TOOL_NAMES and len(at.ANALYST_TOOL_NAMES) == 17
    assert "stress_indices" in an.NUMERIC_TOOLS and "stress_indices" in an.DERIVING_TOOLS
    assert "seventeen fixed tools" in an.ANALYST_SYSTEM_PROMPT and "stress_indices(nonstress_artifact_id=" in an.ANALYST_SYSTEM_PROMPT
    assert "stress_indices" in co.ANALYST_ACTIONS and "stress_indices(nonstress_artifact_id=" in co.COORDINATOR_SYSTEM_PROMPT
    schema = next(s for s in at.analyst_tool_schemas() if s["name"] == "stress_indices")
    assert schema["inputSchema"]["required"] == ["nonstress_artifact_id", "stress_artifact_id"] and "STI" in schema["description"]
    assert Statistic(kind="ratio", denominator="entries in both trials").kind == "ratio"


# --- 6. review of T5 (2026-10-23): the answer's cells, the rank claims, the undefined rules, float range, units -------------

def _answer_rows(shown: str) -> list[str]:
    lines = shown[shown.index("| Rank | Entry |"):].splitlines()
    rows = []
    for line in lines[2:]:
        if not line.startswith("|"):
            break
        rows.append(line)
    return rows


@pytest.mark.parametrize("order", ["highest", "lowest"])
def test_every_answer_row_shows_its_own_rank_index_yp_and_ys(tmp_path, order):
    """Review finding 2: mutants that swapped the Yp and Ys columns, showed Ys as the index, gave every rank claim 1 or started
    lowest-first ranks at 2 all passed. Each shown row is now checked whole against the independent numpy computation."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id=f"run_stress_rows_{order}")
    p, s = trial(10.0), trial(4.0, seed=0.4)
    r = pair(ctx.registry, save(ctx.registry, rows_of("S1", p)), save(ctx.registry, rows_of("S2", s)), order=order, top_k=4)
    assert r.ok, r.error
    paired, yp, ys, _mp, _ms, _si, values = expected(p, s)
    top = sorted(paired, key=lambda e: (-values["STI"][e] if order == "highest" else values["STI"][e], e))[:4]
    want = [f"| {k} | {e} · clone {e} | {cp._anova_number(values['STI'][e])} | {cp._anova_number(yp[e])} | {cp._anova_number(ys[e])} |"
            for k, e in enumerate(top, 1)]
    assert _answer_rows(answer(ctx, claims(r))) == want
    assert sorted(k.value for k in claims(r) if k.kind == "rank") == [1, 2, 3, 4]


def test_the_undefined_rules_of_hm_yi_and_rdi(registry):
    """Review finding 2: HM 0 for 0 / 0, YI undefined when mean Yp (not mean Ys) is 0, and RDI dividing by a mean Ys of 0 all
    passed the tests. A stress trial of zeros: YI and RDI undefined, SI 1, SSI 1, STI 0."""
    rows = per_entry(registry, two(registry, {"E0": [0.0], "E1": [6.0], "E2": [8.0]}, {"E0": [0.0], "E1": [3.0], "E2": [5.0]}, rank_by="TOL"))
    assert rows["E0"]["HM"] == "" and "HM: Yp + Ys is 0" in rows["E0"]["note"] and float(rows["E1"]["HM"]) == pytest.approx(4.0)
    no_stress_yield = two(registry, {"E0": [4.0], "E1": [6.0], "E2": [8.0]}, {"E0": [0.0], "E1": [0.0], "E2": [0.0]}, rank_by="TOL")
    assert no_stress_yield.ok, no_stress_yield.error
    assert no_stress_yield.data["si"] == 1.0
    rows = per_entry(registry, no_stress_yield)
    for e in ("E0", "E1", "E2"):
        assert rows[e]["YI"] == "" and "YI: mean Ys is 0" in rows[e]["note"] and rows[e]["RDI"] == "" and "RDI: mean Ys is 0" in rows[e]["note"]
        assert float(rows[e]["SSI"]) == 1.0 and float(rows[e]["STI"]) == 0.0
    no_base_yield = two(registry, {"E0": [0.0], "E1": [0.0], "E2": [0.0]}, {"E0": [1.0], "E1": [2.0], "E2": [3.0]}, rank_by="TOL")
    assert no_base_yield.ok, no_base_yield.error
    rows = per_entry(registry, no_base_yield)
    assert [float(rows[e]["YI"]) for e in ("E0", "E1", "E2")] == pytest.approx([0.5, 1.0, 1.5])


def test_top_k_above_the_ranked_entries_states_them_all(registry):
    """Review finding 2: top_k beyond the entries ranked."""
    r = two(registry, trial(10.0, entries=ENTRIES[:4]), trial(4.0, entries=ENTRIES[:4], seed=0.4), top_k=50)
    assert r.ok, r.error
    assert len(r.data["stated"]) == 4


def test_a_ratio_beyond_float_range_is_undefined_with_the_reason(registry):
    """Review finding 4 (made-up input; rule (11): guard + test): Ys / Yp of 1e199 / 1e-199 wrote 'inf' to the saved table with no
    reason, and rank_by="YSI" ended in an internal error. Such a ratio is undefined with the reason; the other entries still rank."""
    p, s = {"E0": [1e-199], "E1": [6.0], "E2": [8.0], "E3": [7.0]}, {"E0": [1e199], "E1": [3.0], "E2": [5.0], "E3": [4.0]}
    r = two(registry, p, s)
    assert r.ok, r.error
    rows = per_entry(registry, r)
    for index in ("YSI", "SSI", "RDI"):
        assert rows["E0"][index] == "" and f"{index}: beyond the float range" in rows["E0"]["note"], index
    by_ysi = two(registry, p, s, rank_by="YSI")
    assert by_ysi.ok, by_ysi.error
    assert per_entry(registry, by_ysi)["E0"]["rank"] == "" and sorted(e["entry"] for e in by_ysi.data["stated"]) == ["E1", "E2", "E3"]


def test_two_tables_in_different_units_are_refused(registry):
    """Review finding 6 (made-up input: the retriever saves one variable in one unit): t/ha beside kg/ha gave an STI of 617 and the
    non-stress unit on the stress mean. Both tools that pair two trials refuse it; the same unit written differently is one unit."""
    def in_unit(study, values, unit):
        rows = rows_of(study, values)
        columns = {col: srv.OBSERVATION_COLUMNS.get(col, "text") for col in rows[0]}
        return registry.save_table(rows, kind="observations", columns=columns, complete=True, study_ids=[study], variable_ids=["V1"],
                                   measurement=MeasurementMeta(trait="SYNTHETIC yield", unit=unit), quality_notes=["SYNTHETIC"]).artifact_id
    tonnes, kilos = in_unit("S1", trial(10.0), "t/ha"), in_unit("S2", trial(4000.0, seed=0.4), "kg/ha")
    for tool, args in (("stress_indices", {"nonstress_artifact_id": tonnes, "stress_artifact_id": kilos}),
                       ("compare_trials_paired", {"base_artifact_id": tonnes, "other_artifact_id": kilos})):
        r = at.dispatch_analyst(registry, tool, args)
        assert not r.ok and r.error.code == "unsupported" and "t/ha" in r.error.message and "kg/ha" in r.error.message, tool
    same = at.dispatch_analyst(registry, "stress_indices", {"nonstress_artifact_id": tonnes, "stress_artifact_id": in_unit("S3", trial(4.0), " T/HA ")})
    assert same.ok, same.error
