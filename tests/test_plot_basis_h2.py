"""
tests/test_plot_basis_h2.py - T7a "plot_basis_h2" (coach 2026-10-23 06:49 UTC, T7; research build order tool 8, Q8): the plot-basis
heritability of an unreplicated offspring population, per site: h2 = 1 - s2e / s2P, with s2e the pooled within-entry variance of
the replicated entries (or of the checks only) and s2P the sample variance of the offspring's plots; the mean of the site h2.
A convention (V_P = V_G + V_E) with no published fixture: the research notes' toy (s2e 0.0914286 on 7 df, s2P 4.3266667,
h2 0.97887) is self-consistent only. SYNTHETIC tables otherwise, checked against plain arithmetic. No network, no model.
"""
from __future__ import annotations

import json
import statistics
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

E15 = {r["sub_id"]: r["question"] for r in json.loads((Path(__file__).resolve().parents[1] / "eval" / "benchmark" / "spb_subquestions.json")
                                                     .read_text(encoding="utf-8"))["subquestions"] if r["sub_id"].startswith("E15.")}
# the research notes' toy (Q8): two checks replicated, ten single offspring plots
TOY = {"C1": ([10.2, 9.8, 10.5, 9.9], "check"), "C2": ([12.1, 11.6, 12.4, 11.9, 12.0], "check"),
       **{f"OFF-{i}": ([v], "test") for i, v in enumerate([8.5, 11.2, 13.4, 9.9, 12.8, 10.1, 14.2, 7.9, 11.7, 12.3], 1)}}


@pytest.fixture
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "out", "run_plot_h2", base_url=SYNTHETIC_BASE)


def rows_of(study: str, entries: dict, *, entry_type: bool = True, variable: str = "V1") -> list[dict[str, str]]:
    rows = []
    for name, (values, kind) in entries.items():
        for k, value in enumerate(values, 1):
            row = {"observationDbId": f"O{study}-{name}-{k}", "observationUnitDbId": f"{study}-{name}-{k}", "observationUnitName": f"plot {k}",
                   "germplasmDbId": f"G{name}", "germplasmName": name, "studyDbId": study, "observationVariableDbId": variable,
                   "observationVariableName": "SYNTHETIC trait", "observationTimeStamp": "", "value": repr(float(value))}
            if entry_type:
                row["entryType"] = kind
            rows.append(row)
    return rows


def save(registry, rows, *, unit: str = "mg/100g", complete: bool = True) -> str:
    columns = {col: srv.OBSERVATION_COLUMNS.get(col, "text") for col in rows[0]}
    return registry.save_table(rows, kind="observations", columns=columns, complete=complete, study_ids=sorted({r["studyDbId"] for r in rows} - {""}),
                               variable_ids=sorted({r["observationVariableDbId"] for r in rows}),
                               measurement=MeasurementMeta(trait="SYNTHETIC trait", unit=unit), quality_notes=["SYNTHETIC"]).artifact_id


def h2(registry, ids, **args) -> ToolResult:
    return at.dispatch_analyst(registry, "plot_basis_h2", {"artifact_ids": ids, **args})


def claims(r) -> list[Claim]:
    return [Claim.model_validate(k) for k in r.data["claims"]]


def heads(r) -> dict[tuple[str, str | None], Claim]:
    return {(k.transformation.split(" | ")[0], k.filters.get("studyDbId")): k for k in claims(r)}


def pooled(groups):
    """sum (n_i - 1) s2_i / sum (n_i - 1) over the groups with 2 values or more, by plain arithmetic."""
    groups = [g for g in groups if len(g) >= 2]
    df = sum(len(g) - 1 for g in groups)
    return sum((len(g) - 1) * statistics.variance(g) for g in groups) / df, df


# --- 1. the numbers ----------------------------------------------------------------------------------------------------------

def test_the_research_notes_toy(registry):
    r = h2(registry, [save(registry, rows_of("S1", TOY))], offspring_prefixes=["OFF-"])
    assert r.ok, r.error
    k = heads(r)
    assert k[("error variance s2e", "S1")].value == pytest.approx(0.0914286, abs=1e-7) and k[("error variance s2e", "S1")].denominator.n == 7
    assert k[("phenotypic variance s2P", "S1")].value == pytest.approx(4.3266667, abs=1e-7) and k[("phenotypic variance s2P", "S1")].denominator.n == 10
    assert k[("plot-basis heritability h2", "S1")].value == pytest.approx(0.97887, abs=1e-5)
    assert k[("genetic variance s2g", "S1")].value == pytest.approx(4.3266667 - 0.0914286, abs=1e-7)
    assert k[("plot-basis heritability h2", "S1")].kind == "heritability" and k[("error variance s2e", "S1")].kind == "variance"
    # coach 07:48 UTC: per site, which error pool and the plots behind s2P; and the convention
    said = k[("plot-basis heritability h2", "S1")].transformation
    assert "error pool in study S1: 2 entries with at least 2 plots outside the offspring (df 7); s2P from 10 offspring plots" in said
    assert "a convention" in said and "no published fixture" in said


def two_sites():
    s2 = {"C1": ([10.0, 11.0, 12.0], "check"), "C2": ([20.0, 22.0], "check"), "P1": ([5.0, 6.0, 8.0], "test"),
          **{f"OFF-{i}": ([v], "test") for i, v in enumerate([3.0, 9.0, 14.0, 6.0, 11.0, 7.5], 1)}}
    return {"S1": TOY, "S2": s2}


def test_two_sites_each_on_its_own_and_the_mean_of_the_two(registry):
    sites = two_sites()
    r = h2(registry, [save(registry, rows_of(s, v)) for s, v in sites.items()], offspring_prefixes=["OFF-"])
    assert r.ok, r.error
    k, want = heads(r), {}
    for site, entries in sites.items():
        s2e, df = pooled([vals for name, (vals, _) in entries.items() if not name.startswith("OFF-")])
        s2p = statistics.variance([vals[0] for name, (vals, _) in entries.items() if name.startswith("OFF-")])
        want[site] = 1 - s2e / s2p
        assert k[("error variance s2e", site)].value == pytest.approx(s2e, rel=1e-12) and k[("error variance s2e", site)].denominator.n == df
        assert k[("plot-basis heritability h2", site)].value == pytest.approx(want[site], rel=1e-12)
    assert k[("plot-basis heritability h2, the mean of the sites", None)].value == pytest.approx((want["S1"] + want["S2"]) / 2, rel=1e-12)
    assert [c.transformation.split(" | ")[0] for c in claims(r)][0] == "plot-basis heritability h2, the mean of the sites"


def test_check_only_error_takes_the_entries_whose_entry_type_is_check(registry):
    """E15.3: "use only the two checks' within-entry variance as the error"; P1 (a replicated parent) is left out of s2e."""
    sites = two_sites()
    r = h2(registry, [save(registry, rows_of("S2", sites["S2"]))], offspring_prefixes=["OFF-"], error_from="checks")
    assert r.ok, r.error
    s2e, df = pooled([[10.0, 11.0, 12.0], [20.0, 22.0]])
    assert "error pool in study S2: 2 check entries with at least 2 plots (df 3)" in heads(r)[("plot-basis heritability h2", "S2")].transformation
    assert heads(r)[("error variance s2e", "S2")].value == pytest.approx(s2e, rel=1e-12) and heads(r)[("error variance s2e", "S2")].denominator.n == df
    no_type = h2(registry, [save(registry, rows_of("S2", sites["S2"], entry_type=False))], offspring_prefixes=["OFF-"], error_from="checks")
    assert not no_type.ok and "entryType" in no_type.error.message and "plot_design" in no_type.error.message


def test_an_offspring_entry_with_two_plots_stays_out_of_the_error(registry):
    entries = {**TOY, "OFF-1": ([8.5, 30.0], "test")}
    r = h2(registry, [save(registry, rows_of("S1", entries))], offspring_prefixes=["OFF-"])
    assert r.ok, r.error
    assert heads(r)[("error variance s2e", "S1")].value == pytest.approx(0.0914286, abs=1e-7)
    assert heads(r)[("phenotypic variance s2P", "S1")].denominator.n == 11            # every offspring plot counts in s2P


def test_h2_is_set_to_0_and_said_when_s2e_is_above_s2p(registry):
    entries = {"C1": ([1.0, 9.0], "check"), **{f"OFF-{i}": ([v], "test") for i, v in enumerate([5.0, 5.5, 6.0], 1)}}
    r = h2(registry, [save(registry, rows_of("S1", entries))], offspring_prefixes=["OFF-"])
    assert r.ok, r.error
    k = heads(r)[("plot-basis heritability h2", "S1")]
    assert k.value == 0.0 and "set to 0: s2e is above s2P" in k.transformation
    g = heads(r)[("genetic variance s2g", "S1")]
    assert g.value == 0.0 and "set to 0: s2e is above s2P" in g.transformation                 # never a negative variance


def test_an_s2p_of_0_leaves_h2_undefined(registry):
    entries = {"C1": ([1.0, 2.0], "check"), **{f"OFF-{i}": ([4.0], "test") for i in range(1, 4)}}
    r = h2(registry, [save(registry, rows_of("S1", entries))], offspring_prefixes=["OFF-"])
    assert r.ok, r.error
    k = heads(r)[("plot-basis heritability h2", "S1")]
    assert k.value is None and "s2P is 0" in k.missing_reason


# --- 2. refusals --------------------------------------------------------------------------------------------------------------

def test_what_cannot_be_computed_is_refused(registry):
    few = {"C1": ([1.0, 2.0], "check"), "OFF-1": ([4.0], "test")}
    no_error = {"C1": ([1.0], "check"), **{f"OFF-{i}": ([float(i)], "test") for i in range(1, 4)}}
    for entries, args, word in ((few, {"offspring_prefixes": ["OFF-"]}, "2 plots"),
                                (no_error, {"offspring_prefixes": ["OFF-"]}, "no entry with 2 plots"),
                                (TOY, {"offspring_prefixes": ["NONE-"]}, "'NONE-'"),
                                (TOY, {}, "offspring")):
        r = h2(registry, [save(registry, rows_of("S1", entries))], **args)
        assert not r.ok and word in r.error.message, (args, r.error.message)
    t = save(registry, rows_of("S1", TOY))
    for other, word in ((save(registry, rows_of("S2", TOY), unit="g/100g"), "g/100g"), (save(registry, rows_of("S2", TOY, variable="V2")), "variable")):
        r = h2(registry, [t, other], offspring_prefixes=["OFF-"])
        assert not r.ok and word in r.error.message, word
    assert not h2(registry, [save(registry, rows_of("S3", TOY), complete=False)], offspring_prefixes=["OFF-"]).ok


# --- 3. the person's words ----------------------------------------------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.calls = []

    async def schemas(self):
        return []

    async def call(self, name, args):
        self.calls.append((name, dict(args)))
        return ToolResult(ok=True, data={}, artifact_ids=[])


def test_the_offspring_must_be_written_in_the_persons_words():
    """E15.1 writes "offspring (PH17.*)": PH17. and PH17.* pass; a prefix the person never wrote is refused before the tool runs."""
    def guarded():
        tools = an.GuardedAnalystTools(_Recorder(), ["art_1"], question=E15["E15.1"])
        tools.inspected.add("art_1")
        return tools
    for prefix in ("PH17.", "PH17.*"):
        assert anyio.run(guarded().call, "plot_basis_h2", {"artifact_ids": ["art_1"], "offspring_prefixes": [prefix]}).ok, prefix
    tools = guarded()
    r = anyio.run(tools.call, "plot_basis_h2", {"artifact_ids": ["art_1"], "offspring_prefixes": ["PJ14."]})
    assert not r.ok and "'PJ14.' is not written in the person's words" in r.error.message and tools.inner.calls == []


# --- 4. the answer and the wiring ---------------------------------------------------------------------------------------------

def answer(ctx, claim_list) -> str:
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claim_list, caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    return cp.answer_summary(view)


def test_the_answer_shows_each_site_the_mean_and_the_convention(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_plot_h2_answer")
    r = h2(ctx.registry, [save(ctx.registry, rows_of(s, v)) for s, v in two_sites().items()], offspring_prefixes=["OFF-"])
    shown = answer(ctx, claims(r))
    assert "**Plot-basis heritability, 2 sites**" in shown and "| Site | h2 | s2e (df) | s2P (plots) | s2g |" in shown
    assert f"| S1 | {cp._anova_number(heads(r)[('plot-basis heritability h2', 'S1')].value)} |" in shown
    assert "Mean of the site h2" in shown and cp.PLOT_H2_SENTENCE in shown and "convention" in shown
    shown.encode("cp1252")


def test_every_number_the_answer_shows_is_backed_by_a_claim(tmp_path):
    import importlib.util
    import sys
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_plot_h2_backed")
    r = h2(ctx.registry, [save(ctx.registry, rows_of(s, v)) for s, v in two_sites().items()], offspring_prefixes=["OFF-"])
    shown = answer(ctx, claims(r))
    spec = importlib.util.spec_from_file_location("_bench_for_plot_h2", Path(at.__file__).resolve().parent / "eval" / "benchmark.py")
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    flagged = [n for n in bench._numbers_for_invention(bench.harness.code_written(shown))
               if not bench._backed(n[0], n[1], bench.claim_numbers([k.model_dump(mode="json") for k in claims(r)]))]
    assert flagged == []


def test_every_menu_and_prompt_names_the_tool():
    assert "plot_basis_h2" in at.ANALYST_TOOL_NAMES and len(at.ANALYST_TOOL_NAMES) == 18
    assert "plot_basis_h2" in an.NUMERIC_TOOLS and "plot_basis_h2" in an.DERIVING_TOOLS
    assert "eighteen fixed tools" in an.ANALYST_SYSTEM_PROMPT and "plot_basis_h2(artifact_ids=" in an.ANALYST_SYSTEM_PROMPT
    assert "plot_basis_h2" in co.ANALYST_ACTIONS and "plot_basis_h2(artifact_ids=" in co.COORDINATOR_SYSTEM_PROMPT


# --- 5. review of T7a (2026-10-23) --------------------------------------------------------------------------------------------

OFFS = {f"OFF-{i}": ([v], "test") for i, v in enumerate([3.0, 9.0, 14.0, 6.0], 1)}


def test_a_check_is_a_check_in_its_own_site(registry):
    """Finding 1 (a wrong number on made-up input): X is a check in S1 and a test entry in S2; it joined S2's check-only pool
    (s2e 1666.8 instead of 0.5). An entry's type is read per site; one entry with two types in one site is refused."""
    s1 = {"C1": ([10.0, 11.0], "check"), "X": ([20.0, 22.0], "check"), **OFFS}
    s2 = {"C1": ([10.0, 11.0], "check"), "X": ([100.0, 200.0], "test"), **OFFS}
    r = h2(registry, [save(registry, rows_of("S1", s1)), save(registry, rows_of("S2", s2))], offspring_prefixes=["OFF-"], error_from="checks")
    assert r.ok, r.error
    assert heads(r)[("error variance s2e", "S2")].value == pytest.approx(0.5) and heads(r)[("error variance s2e", "S2")].denominator.n == 1
    mixed = rows_of("S3", {"C1": ([10.0, 11.0], "check"), **OFFS})
    mixed[1]["entryType"] = "test"
    r = h2(registry, [save(registry, mixed)], offspring_prefixes=["OFF-"], error_from="checks")
    assert not r.ok and "GC1" in r.error.message and "study S3" in r.error.message


def test_rows_without_an_entry_or_a_study_are_counted(registry):
    """Finding 2: such rows left df and n silently (df 6 of 7, n 9 of 10)."""
    rows = rows_of("S1", TOY)
    rows[0]["germplasmDbId"] = ""                                          # a C1 plot
    rows[-1]["studyDbId"] = ""                                             # an offspring plot
    r = h2(registry, [save(registry, rows)], offspring_prefixes=["OFF-"])
    assert r.ok, r.error
    left = {e.reason: e.count for e in claims(r)[0].exclusions}
    assert left["rows with an empty germplasmDbId"] == 1 and left["rows with an empty studyDbId"] == 1
    assert any("empty" in w for w in r.warnings)


def test_check_only_error_must_be_written_in_the_persons_words():
    """Finding 3: E15.1 with error_from="checks" passed the guard. "checks" passes only where the person's words ask for it (E15.3:
    "use only the two checks' within-entry variance as the error (check-only error)")."""
    def guarded(question):
        tools = an.GuardedAnalystTools(_Recorder(), ["art_1"], question=question)
        tools.inspected.add("art_1")
        return tools
    args = {"artifact_ids": ["art_1"], "offspring_prefixes": ["PH17."], "error_from": "checks"}
    assert anyio.run(guarded(E15["E15.3"]).call, "plot_basis_h2", args).ok
    r = anyio.run(guarded(E15["E15.1"]).call, "plot_basis_h2", args)
    assert not r.ok and "error_from 'checks' is not stated in the person's words" in r.error.message
    assert anyio.run(guarded(E15["E15.1"]).call, "plot_basis_h2", {**args, "error_from": "replicated"}).ok


def test_entries_in_no_role_counts_neither_offspring_nor_error(registry):
    """Finding 4: the error entries were counted too (the toy gave 2)."""
    r = h2(registry, [save(registry, rows_of("S1", {**TOY, "LONE": ([5.0], "test")}))], offspring_prefixes=["OFF-"])
    assert r.ok and r.data["entries_in_no_role"] == 1


def test_the_mean_of_the_sites_says_a_site_h2_was_set_to_0(registry):
    """Finding 5: the mean averaged a clamped 0 without saying so (and the clamped value is the one averaged)."""
    low = {"C1": ([1.0, 9.0], "check"), **{f"OFF-{i}": ([v], "test") for i, v in enumerate([5.0, 5.5, 6.0], 1)}}
    r = h2(registry, [save(registry, rows_of("S1", TOY)), save(registry, rows_of("S2", low))], offspring_prefixes=["OFF-"])
    assert r.ok, r.error
    k = heads(r)
    mean = k[("plot-basis heritability h2, the mean of the sites", None)]
    assert mean.value == pytest.approx((k[("plot-basis heritability h2", "S1")].value + 0.0) / 2, rel=1e-12)
    assert "includes a site h2 set to 0" in mean.transformation


def test_a_bar_in_the_trait_keeps_the_answer_whole(tmp_path):
    """Finding 6: the block reads the claim's parts by position; a trait holding " | " shifted the Groups line."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_plot_h2_bar")
    rows = rows_of("S1", TOY)
    columns = {col: srv.OBSERVATION_COLUMNS.get(col, "text") for col in rows[0]}
    tid = ctx.registry.save_table(rows, kind="observations", columns=columns, complete=True, study_ids=["S1"], variable_ids=["V1"],
                                  measurement=MeasurementMeta(trait="trait | with a bar", unit="mg/100g"), quality_notes=["SYNTHETIC"]).artifact_id
    shown = answer(ctx, claims(h2(ctx.registry, [tid], offspring_prefixes=["OFF-"])))
    assert "Groups: offspring: OFF-\\*; error: every entry with at least 2 plots outside the offspring." in shown


def test_the_four_refusals_the_review_found_untested(registry):
    """Finding 7: a site the manifest lists without a row, two tables sharing a plot, and a bare prefix that is a whole name and
    starts a longer one were each refused without a test."""
    rows = rows_of("S1", TOY)
    columns = {col: srv.OBSERVATION_COLUMNS.get(col, "text") for col in rows[0]}
    two_sites_one_empty = registry.save_table(rows, kind="observations", columns=columns, complete=True, study_ids=["S1", "S9"], variable_ids=["V1"],
                                              measurement=MeasurementMeta(trait="SYNTHETIC trait", unit="mg/100g"), quality_notes=["SYNTHETIC"]).artifact_id
    r = h2(registry, [two_sites_one_empty], offspring_prefixes=["OFF-"])
    assert not r.ok and "study S9" in r.error.message
    shared = rows_of("S1", TOY)
    r = h2(registry, [save(registry, shared), save(registry, shared[:2] + rows_of("S2", TOY))], offspring_prefixes=["OFF-"])
    assert not r.ok and "is in both" in r.error.message
    r = h2(registry, [save(registry, rows_of("S1", {**TOY, "OFF": ([5.0], "test")}))], offspring_prefixes=["OFF"])
    assert not r.ok and "whole name" in r.error.message
