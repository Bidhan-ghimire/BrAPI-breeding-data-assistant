"""
tests/test_heterosis_population.py - T6 "heterosis_population" (coach 2026-10-23 05:49 UTC, "Then T6"; research build order tool 7):
population-level mid-parent heterosis of ONE trait from group means of entry means - the offspring against the mean of two parent
groups (or one), per site and on the averages over the sites - with high-parent heterosis and the response of the parents over
the founders. The groups are name prefixes and names the person wrote (BrAPI carries no grouping). Descriptive numbers, never a
choice of parents.
Fixtures: Fehr, Principles of Cultivar Development, ch. 9 (F1 90, parents 60 and 80: mid-parent heterosis 28.6 %, high-parent
12.5 %); the research notes' toy groups (parent means 10.1 and 12.0, offspring 11.2: +1.357 %); SYNTHETIC tables otherwise,
checked against plain arithmetic. No network, no model.
"""
from __future__ import annotations

import statistics
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

E14 = {r["sub_id"]: r["question"] for r in __import__("json").loads(
    (__import__("pathlib").Path(__file__).resolve().parents[1] / "eval" / "benchmark" / "spb_subquestions.json").read_text(encoding="utf-8"))[
    "subquestions"] if r["sub_id"].startswith("E14.")}

# two sites; each entry's plot values. Offspring PH17.*, parents PJ14.* and PZ14.*, founders PJ05.* and Isabel, one check.
SITES = {
    "S1": {"PH17.001": [10.0, 12.0], "PH17.002": [14.0], "PJ14.01": [8.0, 8.0], "PJ14.02": [9.0], "PZ14.01": [10.0],
           "PJ05.10": [6.0], "Isabel": [7.0, 9.0], "Covington": [30.0]},
    "S2": {"PH17.001": [20.0], "PH17.002": [16.0, 18.0], "PJ14.01": [12.0], "PJ14.02": [13.0, 15.0], "PZ14.01": [16.0],
           "PJ05.10": [10.0], "Isabel": [11.0], "Covington": [31.0]},
}
GROUPS = {"offspring_prefixes": ["PH17."], "parent_a_prefixes": ["PJ14."], "parent_b_prefixes": ["PZ14."],
          "founder_prefixes": ["PJ05."], "founder_names": ["Isabel"]}
MEMBERS = {"offspring": ("PH17.001", "PH17.002"), "parent A": ("PJ14.01", "PJ14.02"), "parent B": ("PZ14.01",),
           "founder": ("PJ05.10", "Isabel")}


@pytest.fixture
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "out", "run_heterosis", base_url=SYNTHETIC_BASE)


def rows_of(study: str, values: dict[str, list[float]], *, variable: str = "V1") -> list[dict[str, str]]:
    """One plot per value; the entry ID is the name in upper case without dots (names and IDs differ, as in a real table)."""
    return [{"observationDbId": f"O{study}-{name}-{k}", "observationUnitDbId": f"{study}-{name}-{k}", "observationUnitName": f"plot {k}",
             "germplasmDbId": f"G{name.replace('.', '').upper()}", "germplasmName": name, "studyDbId": study,
             "observationVariableDbId": variable, "observationVariableName": "SYNTHETIC yield", "observationTimeStamp": "",
             "value": repr(float(value))}
            for name, plot_values in values.items() for k, value in enumerate(plot_values, 1)]


def save(registry, rows, *, unit: str = "t/ha", complete: bool = True) -> str:
    columns = {col: srv.OBSERVATION_COLUMNS.get(col, "text") for col in rows[0]}
    return registry.save_table(rows, kind="observations", columns=columns, complete=complete, study_ids=sorted({r["studyDbId"] for r in rows}),
                               variable_ids=sorted({r["observationVariableDbId"] for r in rows}),
                               measurement=MeasurementMeta(trait="SYNTHETIC yield", unit=unit), quality_notes=["SYNTHETIC"]).artifact_id


def het(registry, ids, **args) -> ToolResult:
    return at.dispatch_analyst(registry, "heterosis_population", {"artifact_ids": ids, **args})


def two_sites(registry, **over) -> ToolResult:
    return het(registry, [save(registry, rows_of(s, v)) for s, v in SITES.items()], **{**GROUPS, **over})


def claims(r) -> list[Claim]:
    return [Claim.model_validate(k) for k in r.data["claims"]]


def heads(r) -> dict[tuple[str, str | None], Claim]:
    """(the claim's head, its site or None for the averages over the sites) -> the claim."""
    return {(k.transformation.split(" | ")[0], k.filters.get("studyDbId")): k for k in claims(r)}


def expected(sites=SITES):
    """Every number by plain arithmetic: entry means, group means (unweighted over entries), MP, heterosis, high-parent, response."""
    out = {}
    for site, values in [*sites.items(), (None, None)]:
        if site is None:
            g = {name: statistics.fmean(out[s]["groups"][name] for s in sites) for name in MEMBERS}
        else:
            g = {name: statistics.fmean(statistics.fmean(values[e]) for e in members) for name, members in MEMBERS.items()}
        mp = (g["parent A"] + g["parent B"]) / 2
        hp = max(g["parent A"], g["parent B"])
        out[site] = {"groups": g, "MP": mp, "MPH": g["offspring"] - mp, "MPH%": 100 * (g["offspring"] - mp) / mp,
                     "HPH%": 100 * (g["offspring"] - hp) / hp, "response%": 100 * (mp - g["founder"]) / g["founder"]}
    return out


# --- 1. published and computed fixtures ---------------------------------------------------------------------------------

def test_fehrs_published_example(registry):
    """Fehr ch. 9: F1 90, parents 60 and 80 (MP 70, HP 80): (90 - 70) / 70 = 28.6 %; high-parent (90 - 80) / 80 = 12.5 %."""
    t = save(registry, rows_of("S1", {"HYB-1": [90.0], "PA-1": [60.0], "PB-1": [80.0]}))
    r = het(registry, [t], offspring_names=["HYB-1"], parent_a_names=["PA-1"], parent_b_names=["PB-1"])
    assert r.ok, r.error
    k = heads(r)
    assert k[("mid-parent mean", "S1")].value == 70 and k[("mid-parent heterosis", "S1")].value == 20
    assert k[("mid-parent heterosis %", "S1")].value == pytest.approx(100 * 20 / 70, rel=1e-12)          # 28.5714
    assert k[("high-parent heterosis %", "S1")].value == pytest.approx(12.5, rel=1e-12)
    assert not any(site is None for _, site in k)                                                        # one site: no averages


def test_the_research_notes_toy_groups(registry):
    """Parent-group means 10.1 and 12.0 (MP 11.05), offspring 11.2: +1.357 %; each group mean is the unweighted mean of its
    entries' means, whatever their plot counts."""
    values = {"OFF-1": [11.0], "OFF-2": [11.4, 11.4, 11.4], "PA-1": [10.0], "PA-2": [10.2, 10.2], "PB-1": [11.5], "PB-2": [12.5]}
    r = het(registry, [save(registry, rows_of("S1", values))], offspring_prefixes=["OFF-"], parent_a_prefixes=["PA-"], parent_b_prefixes=["PB-"])
    assert r.ok, r.error
    k = heads(r)
    assert k[("mid-parent mean", "S1")].value == pytest.approx(11.05, rel=1e-12)
    assert k[("mid-parent heterosis %", "S1")].value == pytest.approx(100 * 0.15 / 11.05, rel=1e-9)        # 1.3575
    assert k[("offspring mean", "S1")].value == pytest.approx(11.2, rel=1e-12) and k[("offspring mean", "S1")].denominator.n == 2


def test_every_number_per_site_and_on_the_site_averages(registry):
    r = two_sites(registry)
    assert r.ok, r.error
    k, want = heads(r), expected()
    for site in ("S1", "S2", None):
        on = " on the site averages" if site is None else ""
        over = " over the sites" if site is None else ""
        assert k[(f"mid-parent heterosis %{on}", site)].value == pytest.approx(want[site]["MPH%"], rel=1e-12)
        assert k[(f"mid-parent heterosis{on}", site)].value == pytest.approx(want[site]["MPH"], rel=1e-12)
        assert k[(f"high-parent heterosis %{on}", site)].value == pytest.approx(want[site]["HPH%"], rel=1e-12)
        assert k[(f"response of the parents over the founders %{on}", site)].value == pytest.approx(want[site]["response%"], rel=1e-12)
        assert k[(f"mid-parent mean{over}", site)].value == pytest.approx(want[site]["MP"], rel=1e-12)
        for group in MEMBERS:
            assert k[(f"{group} mean{over}", site)].value == pytest.approx(want[site]["groups"][group], rel=1e-12)
    # E14.1's rule: the heterosis on the averages is not the mean of the per-site percents
    assert k[("mid-parent heterosis % on the site averages", None)].value != pytest.approx(
        (want["S1"]["MPH%"] + want["S2"]["MPH%"]) / 2, rel=1e-6)
    assert all(c.unit == "%" and c.kind == "ratio" for c in claims(r) if c.transformation.split(" | ")[0].endswith(("%", "% on the site averages")))


def test_one_table_of_both_sites_gives_the_same_numbers(registry):
    both = save(registry, rows_of("S1", SITES["S1"]) + rows_of("S2", SITES["S2"]))
    one, split = het(registry, [both], **GROUPS), two_sites(registry)
    assert one.ok and split.ok
    assert {h: c.value for h, c in heads(one).items()} == {h: c.value for h, c in heads(split).items()}


def test_one_parent_group_is_its_own_mid_parent(registry):
    r = two_sites(registry, parent_b_prefixes=[])
    assert r.ok, r.error
    k = heads(r)
    assert k[("mid-parent mean", "S1")].value == pytest.approx(8.5) and ("high-parent heterosis %", "S1") not in k
    assert ("parent B mean", "S1") not in k


def test_the_headline_numbers_come_first_for_the_coordinators_view(registry):
    """The coordinator sees 20 claims, the fewest filters first: the averages over the sites (no filter) lead, then each site's
    heterosis, mid-parent mean and response."""
    order = [c.transformation.split(" | ")[0] for c in claims(two_sites(registry))]
    assert order[:4] == ["mid-parent heterosis % on the site averages", "mid-parent heterosis on the site averages",
                         "response of the parents over the founders % on the site averages", "high-parent heterosis % on the site averages"]
    first_site = [h for h, c in zip(order, claims(two_sites(registry))) if c.filters.get("studyDbId") == "S1"]
    assert first_site[:3] == ["mid-parent heterosis %", "mid-parent mean", "mid-parent heterosis"]


# --- 2. edges -------------------------------------------------------------------------------------------------------------

def test_a_mid_parent_mean_of_0_leaves_the_percent_undefined(registry):
    values = {"OFF-1": [3.0], "PA-1": [0.0], "PB-1": [0.0]}
    r = het(registry, [save(registry, rows_of("S1", values))], offspring_prefixes=["OFF-"], parent_a_prefixes=["PA-"], parent_b_prefixes=["PB-"])
    assert r.ok, r.error
    k = heads(r)
    assert k[("mid-parent heterosis %", "S1")].value is None and "the mid-parent mean is 0" in k[("mid-parent heterosis %", "S1")].missing_reason
    assert k[("mid-parent heterosis", "S1")].value == 3.0


def test_a_mean_of_float_residue_counts_as_0(registry):
    """T4's residue rule: plot values that cancel (0.1 + 0.2 - 0.3) leave 1e-17 beside plots of size 1: that mean is 0."""
    values = {"OFF-1": [3.0], "PA-1": [0.1, 0.2, -0.3], "PB-1": [0.0]}
    r = het(registry, [save(registry, rows_of("S1", values))], offspring_prefixes=["OFF-"], parent_a_prefixes=["PA-"], parent_b_prefixes=["PB-"])
    assert r.ok, r.error
    assert heads(r)[("mid-parent heterosis %", "S1")].value is None
    assert heads(r)[("parent A mean", "S1")].value == 0.0                 # deliberate bug H11: 9.25e-18 was shown as the mean


def test_negative_group_means_are_refused(registry):
    values = {"OFF-1": [3.0], "PA-1": [-2.0], "PB-1": [1.0]}
    r = het(registry, [save(registry, rows_of("S1", values))], offspring_prefixes=["OFF-"], parent_a_prefixes=["PA-"], parent_b_prefixes=["PB-"])
    assert not r.ok and r.error.code == "unsupported" and "ratio scale" in r.error.message


def test_an_entry_in_two_groups_is_refused(registry):
    r = two_sites(registry, parent_a_prefixes=["PJ"])                     # PJ also takes the founder PJ05.10
    assert not r.ok and r.error.code == "invalid_argument" and "PJ05.10" in r.error.message and "founder" in r.error.message


def test_a_group_with_no_value_in_a_site_is_refused(registry):
    s2 = {k: v for k, v in SITES["S2"].items() if not k.startswith("PZ14.")}
    r = het(registry, [save(registry, rows_of("S1", SITES["S1"])), save(registry, rows_of("S2", s2))], **GROUPS)
    assert not r.ok and r.error.code == "unsupported" and "parent B" in r.error.message and "S2" in r.error.message


def test_a_prefix_or_name_that_matches_no_entry_is_refused(registry):
    for over, word in (({"founder_names": ["Isabell"]}, "'Isabell'"), ({"offspring_prefixes": ["PH18."]}, "'PH18.'")):
        r = two_sites(registry, **over)
        assert not r.ok and r.error.code == "invalid_argument" and word in r.error.message, over


def test_a_trailing_star_is_the_wildcard_the_person_wrote(registry):
    r = two_sites(registry, offspring_prefixes=["PH17.*"])
    assert r.ok, r.error
    assert heads(r)[("offspring mean", "S1")].value == pytest.approx(expected()["S1"]["groups"]["offspring"])


def test_an_offspring_and_a_parent_group_are_needed(registry):
    r = two_sites(registry, parent_a_prefixes=[], parent_b_prefixes=[])
    assert not r.ok and r.error.code == "invalid_argument" and "parent" in r.error.message


def test_tables_that_do_not_fit_together_are_refused(registry):
    s1 = save(registry, rows_of("S1", SITES["S1"]))
    for other, word in ((save(registry, rows_of("S2", SITES["S2"]), unit="kg/ha"), "kg/ha"),
                        (save(registry, rows_of("S2", SITES["S2"], variable="V2")), "variable"),
                        (s1, "same rows")):
        r = het(registry, [s1, other], **GROUPS)
        assert not r.ok and word in r.error.message, word


def test_two_tables_that_share_plots_are_refused(registry):
    """Two different tables holding the same plot would count its value twice in its entry's mean."""
    s1 = rows_of("S1", SITES["S1"])
    r = het(registry, [save(registry, s1), save(registry, s1[:3] + rows_of("S2", SITES["S2"]))], **GROUPS)
    assert not r.ok and r.error.code == "unsupported" and "is in both" in r.error.message


def test_an_incomplete_table_needs_allow_incomplete(registry):
    t = save(registry, rows_of("S1", SITES["S1"]), complete=False)
    assert not het(registry, [t], **GROUPS).ok
    assert het(registry, [t], **GROUPS, allow_incomplete=True).ok


def test_the_entries_in_no_group_are_counted(registry):
    r = two_sites(registry)
    left = {e.reason: e.count for e in claims(r)[0].exclusions}
    assert left["entries in no group the person named"] == 1                                            # Covington
    assert r.data["groups"]["S1"]["founder"]["entries"] == 2


# --- 3. the person's words: the groups are theirs (the F4 gate, extended to this tool) ---------------------------------------

class _Recorder:
    def __init__(self):
        self.calls = []

    async def schemas(self):
        return []

    async def call(self, name, args):
        self.calls.append((name, dict(args)))
        return ToolResult(ok=True, data={}, artifact_ids=[])


def guarded(question):
    tools = an.GuardedAnalystTools(_Recorder(), ["art_1"], question=question)
    tools.inspected.add("art_1")
    return tools


def test_every_group_prefix_and_name_must_be_written_in_the_persons_words():
    """The grouping is the person's (BrAPI has none): each prefix and name is checked against their words before the tool runs,
    the same whole-name test as anova_trial's exclude. E14.1's real wording passes; it writes "PZ14 parents likewise", so the
    prefix PZ14. counts through its stem, while PZ or PH1 (parts of longer names) do not."""
    question = E14["E14.1"]
    assert "PZ14 parents likewise" in question and "PZ14." not in question
    args = {"artifact_ids": ["art_1"], "offspring_prefixes": ["PH17.*"], "parent_a_prefixes": ["PJ14."], "parent_b_prefixes": ["PZ14."],
            "founder_prefixes": ["PJ05.", "PZ06."], "founder_names": ["Isabel", "CIP105103.1"]}
    tools = guarded(question)
    assert anyio.run(tools.call, "heterosis_population", args).ok and len(tools.inner.calls) == 1
    for key, extra in (("founder_names", "Tanzania"), ("parent_b_prefixes", "PZ"), ("offspring_prefixes", "PH1")):
        tools = guarded(question)
        r = anyio.run(tools.call, "heterosis_population", {**args, key: [*args[key], extra]})
        assert not r.ok and r.error.code == "invalid_argument" and f"{extra!r} is not written in the person's words" in r.error.message, key
        assert tools.inner.calls == []


# --- 4. the answer and the wiring -----------------------------------------------------------------------------------------

def answer(ctx, claim_list) -> str:
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claim_list, caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    return cp.answer_summary(view)


def test_the_answer_shows_the_group_means_the_heterosis_and_the_sentence(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_heterosis_answer")
    r = two_sites(ctx.registry)
    shown = answer(ctx, claims(r))
    want = expected()
    assert "**Population heterosis of group means, 2 sites**" in shown
    assert "| Group | S1 | S2 | Mean of the sites |" in shown
    row = f"| Mid-parent | {cp._anova_number(want['S1']['MP'])} | {cp._anova_number(want['S2']['MP'])} | {cp._anova_number(want[None]['MP'])} |"
    assert row in shown
    assert f"on the site averages {cp._anova_number(want[None]['MPH'])} t/ha ({cp._anova_number(want[None]['MPH%'])} %)" in shown
    assert cp.HETEROSIS_SENTENCE in shown and "| Statistic | Result |" not in shown
    shown.encode("cp1252")


def test_every_number_the_answer_shows_is_backed_by_a_claim(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_heterosis_backed")
    r = two_sites(ctx.registry)
    shown = answer(ctx, claims(r))
    spec = importlib.util.spec_from_file_location("_bench_for_heterosis", Path(at.__file__).resolve().parent / "eval" / "benchmark.py")
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    flagged = [n for n in bench._numbers_for_invention(bench.harness.code_written(shown))
               if not bench._backed(n[0], n[1], bench.claim_numbers([k.model_dump(mode="json") for k in claims(r)]))]
    assert flagged == []


def test_every_menu_and_prompt_names_the_tool():
    assert "heterosis_population" in at.ANALYST_TOOL_NAMES and len(at.ANALYST_TOOL_NAMES) == 19
    assert "heterosis_population" in an.NUMERIC_TOOLS and "heterosis_population" in an.DERIVING_TOOLS
    assert "nineteen fixed tools" in an.ANALYST_SYSTEM_PROMPT and "heterosis_population(artifact_ids=" in an.ANALYST_SYSTEM_PROMPT
    assert "heterosis_population" in co.ANALYST_ACTIONS and "heterosis_population(artifact_ids=" in co.COORDINATOR_SYSTEM_PROMPT
    schema = next(s for s in at.analyst_tool_schemas() if s["name"] == "heterosis_population")
    assert schema["inputSchema"]["required"] == ["artifact_ids"] and "mid-parent" in schema["description"]


# --- 5. review of T6 (2026-10-23) ----------------------------------------------------------------------------------------------

def blank_rows(study: str, names) -> list[dict[str, str]]:
    """Plots of the trait without a value (a study that holds the variable but recorded nothing)."""
    rows = rows_of(study, {name: [0.0] for name in names})
    return [{**row, "value": ""} for row in rows]


def test_a_site_without_any_valid_value_is_refused_not_dropped(registry):
    """Review finding 1: S2's plots carry the trait but no value; the site vanished and a two-site question got a one-site answer.
    The site is one of the tables' studies, so the group without a value there is refused, naming it."""
    for tables in ([save(registry, rows_of("S1", SITES["S1"])), save(registry, blank_rows("S2", SITES["S2"]))],
                   [save(registry, rows_of("S1", SITES["S1"]) + blank_rows("S2", SITES["S2"]))]):
        r = het(registry, tables, **GROUPS)
        assert not r.ok and r.error.code == "unsupported" and "study S2" in r.error.message and "no entry with a valid value" in r.error.message


def test_the_site_averages_weight_each_site_equally(registry):
    """Review finding 4 (R1 survived): every group had the same entries in both sites, so entry-weighted and site-weighted averages
    coincided. S2 now has one more offspring entry and one founder fewer."""
    s2 = {**SITES["S2"], "PH17.003": [40.0]}
    s2.pop("PJ05.10")
    r = het(registry, [save(registry, rows_of("S1", SITES["S1"])), save(registry, rows_of("S2", s2))], **GROUPS)
    assert r.ok, r.error
    k = heads(r)
    o1, o2 = statistics.fmean([11.0, 14.0]), statistics.fmean([20.0, 17.0, 40.0])
    assert k[("offspring mean over the sites", None)].value == pytest.approx((o1 + o2) / 2, rel=1e-12)
    assert k[("offspring mean over the sites", None)].value != pytest.approx(statistics.fmean([11.0, 14.0, 20.0, 17.0, 40.0]), rel=1e-6)
    f1, f2 = statistics.fmean([6.0, 8.0]), 11.0
    assert k[("founder mean over the sites", None)].value == pytest.approx((f1 + f2) / 2, rel=1e-12)


def test_a_name_is_matched_whole_never_as_a_prefix(registry):
    """Review finding 4 (R6 survived): with Isabela beside Isabel, founder_names=["Isabel"] takes Isabel alone."""
    sites = {s: {**v, "Isabela": [50.0]} for s, v in SITES.items()}
    r = het(registry, [save(registry, rows_of(s, v)) for s, v in sites.items()], **GROUPS)
    assert r.ok, r.error
    assert r.data["groups"]["S1"]["founder"]["entries"] == 2 and r.data["entries_in_no_group"] == 2      # Covington, Isabela


def test_a_prefix_that_is_a_whole_entry_name_and_starts_a_longer_one_is_refused(registry):
    """Review finding 2 (tool side): founder_prefixes=["Isabel"] took Isabela too (founder mean 7.0 -> 38.0); a name passed as a
    prefix that also starts a longer name is refused loudly: pass it as a name, or the prefix the person wrote."""
    sites = {s: {**v, "Isabela": [50.0]} for s, v in SITES.items()}
    r = het(registry, [save(registry, rows_of(s, v)) for s, v in sites.items()], **{**GROUPS, "founder_names": [], "founder_prefixes": ["PJ05.", "Isabel"]})
    assert not r.ok and r.error.code == "invalid_argument" and "'Isabel'" in r.error.message and "Isabela" in r.error.message


def test_the_block_names_its_trait_and_a_percent_trait_differs_in_percentage_points(tmp_path):
    """Review finding 3: E14.4's four blocks did not say which trait each was; a dry-matter difference read "-0.5568 %"."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_heterosis_trait")
    shown = answer(ctx, claims(two_sites(ctx.registry)))
    assert "SYNTHETIC yield \\(t/ha\\) in studies S1, S2." in shown
    percent = het(ctx.registry, [save(ctx.registry, rows_of(s, v), unit="%") for s, v in SITES.items()], **GROUPS)
    shown = answer(ctx, claims(percent))
    assert f"on the site averages {cp._anova_number(expected()[None]['MPH'])} percentage points (" in shown


def test_a_table_without_a_unit_beside_one_with_a_unit_is_refused(registry):
    """Review finding 5: t/ha beside no unit passed and the answer showed t/ha for both."""
    r = het(registry, [save(registry, rows_of("S1", SITES["S1"])), save(registry, rows_of("S2", SITES["S2"]), unit="")], **GROUPS)
    assert not r.ok and r.error.code == "unsupported" and "unit" in r.error.message


def test_each_warning_is_said_once(registry):
    """Review finding 6: two tables repeated the same timing warning."""
    r = two_sites(registry)
    assert r.ok and len(r.warnings) == len(set(r.warnings))


def test_on_e14_1s_words_a_bare_word_is_no_prefix_and_a_name_has_no_stem():
    """Review of T6, finding 2 (gate side): '80*', 'Satipo*', 'Isabel*' and the bare 'Isabel' passed as prefixes because each is a
    whole word of the question; a prefix now counts only as written, or through its stem when it ends with . - or _, and a bare
    word only with its star. A name is matched as written: 'PZ14.' is no name of E14.1's."""
    question = E14["E14.1"]
    args = {"artifact_ids": ["art_1"], "offspring_prefixes": ["PH17.*"], "parent_a_prefixes": ["PJ14."], "parent_b_prefixes": ["PZ14."],
            "founder_prefixes": ["PJ05.", "PZ06."], "founder_names": ["Isabel", "CIP105103.1"]}
    assert anyio.run(guarded(question).call, "heterosis_population", args).ok
    for key, value in (("founder_prefixes", ["PJ05.", "PZ06.", "Isabel"]), ("offspring_prefixes", ["Satipo*"]), ("offspring_prefixes", ["80*"]),
                       ("founder_prefixes", ["Isabel*"]), ("founder_names", ["Isabel", "PZ14."])):
        tools = guarded(question)
        r = anyio.run(tools.call, "heterosis_population", {**args, key: value})
        assert not r.ok and f"{value[-1]!r} is not written in the person's words" in r.error.message and tools.inner.calls == [], value
