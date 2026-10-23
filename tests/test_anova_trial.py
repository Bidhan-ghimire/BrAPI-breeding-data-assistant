"""
tests/test_anova_trial.py - T2 "anova_trial" (coach 2026-10-22 20:48 UTC, NEXT 5; research build order, tool 2): the analysis of
variance of ONE trial's plot values for one trait - an RCBD (entries + replicates) when every plot states its rep (plot_design),
one-way (entries only) when none does - with the grand mean, CV, SE, SED, LSD(5%), the variance components and the entry-mean
heritability. Missing plots: the least-squares ("adjusted") analysis, replicates first. Descriptive numbers, never a decision.

Fixtures (literal numbers, research notes NOTES_plain_python_implementation.md Q1, Q2, Q11; values computed in
Claude outputs/t2_anova_trial_20261022/fixture_values.py): agricolae `sweetpotato` (one-way, published MSerror 22.48917, CV
17.1666, LSD 8.928965, mean 27.625), the NDSU rice RCBD (published table), the NDSU missing-plot example (published adjusted
treatment SS 54.0417, error 26.375 on 5 df, F 3.415, p 0.1097). Unbalanced designs are also checked against an independent
numpy least-squares fit (numpy only in this test). SYNTHETIC tables otherwise; no network, no model.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import agents.analyst as an
import agents.coordinator as co
import analyst_tools as at
import brapi_mcp_server as srv
import chat_presenter as cp
import contracts as c
from artifacts import ArtifactRegistry
from contracts import AnalysisReport, Claim, MeasurementMeta
from part2_support import SYNTHETIC_BASE, synthetic_context

SWEETPOTATO = {"cc": [28.5, 21.7, 23.0], "fc": [14.9, 10.6, 13.1], "ff": [41.8, 39.2, 28.0], "oo": [38.2, 40.4, 32.1]}   # agricolae, virus
RICE = {"25": [5.1, 5.4, 5.3, 4.7], "50": [5.3, 6.0, 4.7, 4.3], "75": [5.3, 5.7, 5.5, 4.7], "100": [5.2, 4.8, 5.0, 4.4],
        "125": [4.8, 4.8, 4.4, 4.7], "150": [5.3, 4.5, 4.9, 4.1]}                                                    # NDSU, rates x reps 1-4
MISSING_PLOT = {"A": [9, 8, 7], "B": [11, 13, 12], "C": [3, 5, 8], "D": [7, 10, 4]}                                  # NDSU, reps 1-3


@pytest.fixture
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "out", "run_anova", base_url=SYNTHETIC_BASE)


def plots(table: dict[str, list[float]], *, reps: bool = True, drop: tuple[tuple[str, int], ...] = ()) -> list[tuple[str, str, str | None, str]]:
    """(plot ID, entry, rep or None, value text) for an entries x reps table; `drop` leaves out (entry, rep) plots."""
    return [(f"P{entry}-{j}", entry, str(j) if reps else None, repr(float(v)))
            for entry, values in table.items() for j, v in enumerate(values, 1) if (entry, j) not in drop]


def save(registry: ArtifactRegistry, rows_in, *, study="S1", variable="V1", unit="t/ha", complete=True, rep_column=True, study_of=None,
         blocks=None) -> str:
    """rows_in: (plot ID, entry, rep or None, value text[, date]); blocks: {plot ID: block code} adds plot_design's block column."""
    rows = []
    for item in rows_in:
        unit_id, entry, rep, value = item[:4]
        row = {"observationDbId": f"O{unit_id}-{len(rows)}", "observationUnitDbId": unit_id, "observationUnitName": f"plot {unit_id}",
               "germplasmDbId": entry, "germplasmName": f"clone {entry}", "studyDbId": study_of(unit_id) if study_of else study,
               "observationVariableDbId": variable,
               "observationVariableName": "SYNTHETIC yield", "observationTimeStamp": item[4] if len(item) > 4 else "", "value": value}
        if rep_column:
            row["rep"] = rep or ""
        if blocks is not None:
            row["block"] = blocks.get(unit_id, "")
        rows.append(row)
    columns = {col: srv.OBSERVATION_COLUMNS.get(col, "text") for col in rows[0]}
    return registry.save_table(rows, kind="observations", columns=columns, complete=complete, study_ids=sorted({r["studyDbId"] for r in rows}),
                               variable_ids=[variable], measurement=MeasurementMeta(trait="SYNTHETIC yield", unit=unit),
                               quality_notes=["SYNTHETIC"]).artifact_id


def anova(registry, artifact_id, **args):
    return at.dispatch_analyst(registry, "anova_trial", {"artifact_id": artifact_id, **args})


def table(r) -> dict[str, dict]:
    return {row["source"]: row for row in r.data["anova"]}


def claims(r) -> list[Claim]:
    return [Claim.model_validate(k) for k in r.data["claims"]]


# --- 1. the numbers: published fixtures ------------------------------------------------------------

def test_one_way_matches_agricolae_sweetpotato(registry):
    """No rep column: one-way. Published by agricolae: Mean 27.625, CV 17.1666, MSerror 22.48917, LSD 8.928965."""
    r = anova(registry, save(registry, plots(SWEETPOTATO, reps=False), rep_column=False))
    assert r.ok, r.error
    d, t = r.data, table(r)
    assert d["design"] == "one-way" and d["balanced"] is True and set(t) == {"entry", "error", "total"}
    assert (t["entry"]["df"], t["error"]["df"], t["total"]["df"]) == (3, 8, 11)
    assert t["entry"]["ss"] == pytest.approx(1170.209167, rel=1e-9) and t["error"]["ss"] == pytest.approx(179.9133333, rel=1e-9)
    assert t["error"]["ms"] == pytest.approx(22.48916667, rel=1e-9) and t["entry"]["f"] == pytest.approx(17.34478329, rel=1e-9)
    assert t["entry"]["p"] == pytest.approx(0.0007334171662, rel=1e-9)
    assert d["grand_mean"] == pytest.approx(27.625, rel=1e-12) and d["cv_percent"] == pytest.approx(17.16660423, rel=1e-9)
    assert d["se"] == pytest.approx(2.737953412, rel=1e-9) and d["sed"] == pytest.approx(3.872050849, rel=1e-9)
    assert d["lsd_5pct"] == pytest.approx(8.92896527, rel=1e-9)
    assert d["sigma2_g"] == pytest.approx(122.5268519, rel=1e-9) and d["sigma2_e"] == pytest.approx(22.48916667, rel=1e-9)
    assert d["h2"] == pytest.approx(0.9423457772, rel=1e-9) and d["h2_set_to_zero"] is False
    # agricolae's printout, 7 significant digits
    assert t["error"]["ms"] == pytest.approx(22.48917, rel=5e-7) and d["lsd_5pct"] == pytest.approx(8.928965, rel=5e-7)
    assert d["cv_percent"] == pytest.approx(17.1666, rel=5e-6)


def test_one_way_with_unequal_plot_counts_is_exact(registry):
    """The sweetpotato data without the last oo plot (research note Q1): df error 7, MS error 20.764762."""
    rows = [p for p in plots(SWEETPOTATO, reps=False) if p[0] != "Poo-3"]
    r = anova(registry, save(registry, rows, rep_column=False))
    d, t = r.data, table(r)
    assert d["design"] == "one-way" and d["balanced"] is False and t["error"]["df"] == 7
    assert t["error"]["ms"] == pytest.approx(20.764762, rel=1e-6) and d["replication"] == pytest.approx(4 / (1 / 3 * 3 + 1 / 2), rel=1e-12)


def test_rcbd_matches_the_ndsu_rice_table(registry):
    r = anova(registry, save(registry, plots(RICE)))
    assert r.ok, r.error
    d, t = r.data, table(r)
    assert d["design"] == "RCBD" and d["balanced"] is True and set(t) == {"rep", "entry", "error", "total"}
    assert (t["rep"]["df"], t["entry"]["df"], t["error"]["df"], t["total"]["df"]) == (3, 5, 15, 23)
    assert t["rep"]["ss"] == pytest.approx(1.964583333, rel=1e-9) and t["entry"]["ss"] == pytest.approx(1.267083333, rel=1e-9)
    assert t["error"]["ss"] == pytest.approx(1.787916667, rel=1e-9) and t["total"]["ss"] == pytest.approx(5.019583333, rel=1e-9)
    assert t["entry"]["f"] == pytest.approx(2.126077837, rel=1e-9) and t["entry"]["p"] == pytest.approx(0.1183664559, rel=1e-9)
    assert t["rep"]["f"] == pytest.approx(5.494057329, rel=1e-9) and t["rep"]["p"] == pytest.approx(0.009487716041, rel=1e-9)
    assert d["cv_percent"] == pytest.approx(6.968790264, rel=1e-9) and d["lsd_5pct"] == pytest.approx(0.5203410222, rel=1e-9)
    assert d["sigma2_g"] == pytest.approx(0.03355555556, rel=1e-9) and d["h2"] == pytest.approx(0.5296503343, rel=1e-9)
    # the published NDSU table, computed there from rounded intermediates: SS to 4 decimals, MS error 0.1192, CV 6.97 %
    assert t["rep"]["ss"] == pytest.approx(1.9650, abs=5e-4) and t["entry"]["ss"] == pytest.approx(1.2675, abs=5e-4)
    assert t["error"]["ss"] == pytest.approx(1.7875, abs=5e-4) and t["error"]["ms"] == pytest.approx(0.1192, abs=5e-5)
    assert d["cv_percent"] == pytest.approx(6.97, abs=5e-3)


def test_complete_and_one_missing_plot_match_the_ndsu_missing_plot_example(registry):
    full = anova(registry, save(registry, plots(MISSING_PLOT)))
    t = table(full)
    assert full.data["balanced"] is True and t["rep"]["ss"] == pytest.approx(5.166666667, rel=1e-9)
    assert t["entry"]["ss"] == pytest.approx(72.25, rel=1e-12) and t["error"]["ss"] == pytest.approx(29.5, rel=1e-12)
    assert t["entry"]["f"] == pytest.approx(4.898305085, rel=1e-9) and t["entry"]["p"] == pytest.approx(0.04712986676, rel=1e-9)
    gap = anova(registry, save(registry, plots(MISSING_PLOT, drop=(("C", 2),))))
    assert gap.ok, gap.error
    d, t = gap.data, table(gap)
    assert d["design"] == "RCBD" and d["balanced"] is False and d["adjusted"] is True
    assert t["entry"]["ss"] == pytest.approx(54.04166667, rel=1e-9) and t["error"]["ss"] == pytest.approx(26.375, rel=1e-9)
    assert t["error"]["df"] == 5 and t["entry"]["f"] == pytest.approx(3.41495524, rel=1e-9)
    assert t["entry"]["p"] == pytest.approx(0.109709418, rel=1e-8)
    assert t["entry"]["f"] == pytest.approx(3.415, abs=5e-4) and t["entry"]["p"] == pytest.approx(0.1097, abs=5e-5)   # published
    assert "harmonic mean" in d["replication_basis"] and d["balance"]["empty_cells"] == 1
    assert d["replication"] == pytest.approx(8 / 3, rel=1e-12)                 # 4 / (1/3 + 1/3 + 1/2 + 1/3): C has 2 plots


def test_unbalanced_designs_match_an_independent_least_squares_fit(registry):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(20261022)
    checked = 0
    while checked < 25:
        t, r = int(rng.integers(3, 8)), int(rng.integers(2, 5))
        table_in = {f"E{i}": [float(round(rng.normal(10, 3), 3)) for _ in range(r)] for i in range(t)}
        cells = [(e, j) for e in table_in for j in range(1, r + 1)]
        drop = tuple(cells[k] for k in rng.choice(len(cells), size=int(rng.integers(1, max(2, len(cells) // 4))), replace=False))
        kept = [p for p in plots(table_in, drop=drop)]
        if len({p[1] for p in kept}) != t or len({p[2] for p in kept}) != r:
            continue
        y = np.array([float(p[3]) for p in kept])
        entries, reps = sorted({p[1] for p in kept}), sorted({p[2] for p in kept})
        x_rep = np.column_stack([np.ones(len(kept))] + [[1.0 if p[2] == q else 0.0 for p in kept] for q in reps[1:]])
        x_full = np.column_stack([x_rep] + [[1.0 if p[1] == e else 0.0 for p in kept] for e in entries[1:]])
        rss = lambda m: float(np.sum((y - m @ np.linalg.lstsq(m, y, rcond=None)[0]) ** 2))
        if np.linalg.matrix_rank(x_full) != x_full.shape[1]:
            continue                                               # disconnected: the tool refuses those (tested below)
        result = anova(registry, save(registry, kept))
        assert result.ok, result.error
        tab = table(result)
        assert tab["entry"]["ss"] == pytest.approx(rss(x_rep) - rss(x_full), abs=1e-9)
        assert tab["error"]["ss"] == pytest.approx(rss(x_full), abs=1e-9) and tab["error"]["df"] == len(kept) - t - r + 1
        checked += 1


# --- 2. variance components, H2 and its flags -----------------------------------------------------

def test_a_negative_genotype_variance_is_set_to_zero_and_flagged(registry):
    """Entries that differ less than plots within an entry: MS entry < MS error, so the estimate of sigma2 g is negative."""
    r = anova(registry, save(registry, plots({"A": [1, 9, 5], "B": [2, 8, 5], "C": [9, 1, 5]})))
    d = r.data
    assert table(r)["entry"]["ms"] < table(r)["error"]["ms"]
    assert d["sigma2_g"] == 0 and d["h2"] == 0 and d["h2_set_to_zero"] is True
    h2 = next(k for k in claims(r) if k.kind == "heritability")
    assert h2.value == 0 and "set to 0" in h2.transformation


# --- 3. what is refused, never guessed ---------------------------------------------------------------

def test_two_plots_of_one_entry_in_one_rep_are_not_an_rcbd(registry):
    rows = plots(RICE) + [("P25-extra", "25", "1", "5.0")]
    r = anova(registry, save(registry, rows))
    assert not r.ok and r.error.code == "unsupported" and "entry 25 has 2 plots in rep 1" in r.error.message


def test_some_plots_without_a_rep_are_refused(registry):
    rows = [(u, e, None if u == "P25-1" else rep, v) for u, e, rep, v in plots(RICE)]
    r = anova(registry, save(registry, rows))
    assert not r.ok and r.error.code == "unsupported" and "1 plot(s) state no rep" in r.error.message


def test_a_disconnected_design_is_refused(registry):
    """Entries A and B only in reps 1-2, C and D only in reps 3-4: rep and entry effects cannot be told apart."""
    rows = [(f"p{e}{j}", e, str(j), str(float(k + j))) for k, (e, js) in enumerate({"A": (1, 2), "B": (1, 2), "C": (3, 4), "D": (3, 4)}.items())
            for j in js]
    r = anova(registry, save(registry, rows))
    assert not r.ok and r.error.code == "unsupported" and "not connected" in r.error.message


def test_a_table_of_two_studies_is_refused(registry):
    r = anova(registry, save(registry, plots(RICE), study_of=lambda unit: "S2" if unit.endswith("-4") else "S1"))
    assert not r.ok and r.error.code == "unsupported" and "one study" in r.error.message


def test_an_incomplete_table_is_refused_unless_allowed_and_then_labelled(registry):
    aid = save(registry, plots(RICE), complete=False)
    r = anova(registry, aid)
    assert not r.ok and r.error.code == "incomplete_data"
    r = anova(registry, aid, allow_incomplete=True)
    assert r.ok and all("INCOMPLETE ARTIFACT" in k.transformation for k in claims(r))


def test_missing_and_invalid_values_are_excluded_and_counted(registry):
    rows = [(u, e, rep, "NA" if u == "P25-1" else "bad" if u == "P50-2" else v) for u, e, rep, v in plots(RICE)]
    r = anova(registry, save(registry, rows))
    assert r.ok, r.error
    assert r.data["balanced"] is False and r.data["plots_used"] == 22
    exclusions = {e.reason: e.count for e in claims(r)[0].exclusions}
    assert exclusions == {"raw-missing value token": 1, "invalid or non-finite value token": 1}


def test_too_few_entries_or_reps_are_refused(registry):
    r = anova(registry, save(registry, plots({"A": [1, 2, 3]})))
    assert not r.ok and "at least 2 entries" in r.error.message
    r = anova(registry, save(registry, plots({"A": [1], "B": [2], "C": [3]})))
    assert not r.ok and "at least 2" in r.error.message


# --- 4. the claims, the menus, the contract and the answer ----------------------------------------------

def test_every_number_is_a_claim_with_its_kind_source_and_plots(registry):
    aid = save(registry, plots(RICE))
    r = anova(registry, aid)
    ks = claims(r)
    sha = registry.load(aid)[0].sha256
    assert {k.kind for k in ks} == {"anova", "mean", "precision", "variance", "heritability"}
    assert all(k.source_artifact_hashes == [sha] and k.denominator.n == 24 for k in ks)
    anova_claims = {k.transformation.split(" | ")[0]: k.value for k in ks if k.kind == "anova"}
    assert anova_claims["RCBD ANOVA: df of entry"] == 5 and anova_claims["RCBD ANOVA: p of entry"] == pytest.approx(0.1183664559, rel=1e-9)
    precision = {k.transformation.split(" | ")[0]: k for k in ks if k.kind == "precision"}
    assert precision["CV"].unit == "%" and precision["LSD (5%)"].unit == "t/ha" and precision["LSD (5%)"].value == pytest.approx(0.5203410222, rel=1e-9)
    assert len({k.claim_id for k in ks}) == len(ks)
    texts = [k.transformation.replace(at.ANOVA_NOTE, "").lower() for k in ks]          # the disclaimer itself says "not a recommendation"
    assert all(at.ANOVA_NOTE in k.transformation for k in ks)
    assert not any(word in text for text in texts for word in ("best", "select", "recommend", "superior"))


def test_every_menu_and_prompt_names_the_tool():
    assert "anova_trial" in at.ANALYST_TOOL_NAMES and len(at.ANALYST_TOOL_NAMES) == 15
    assert "anova_trial" in an.NUMERIC_TOOLS and "anova_trial" in an.DERIVING_TOOLS
    assert "fifteen fixed tools" in an.ANALYST_SYSTEM_PROMPT and "anova_trial" in an.ANALYST_SYSTEM_PROMPT
    assert "anova_trial" in co.ANALYST_ACTIONS and "anova_trial(artifact_id=" in co.COORDINATOR_SYSTEM_PROMPT
    schema = next(s for s in at.analyst_tool_schemas() if s["name"] == "anova_trial")
    assert schema["inputSchema"]["required"] == ["artifact_id"] and "RCBD" in schema["description"]


def test_the_contract_knows_the_new_claim_kinds():
    for kind in ("anova", "precision", "variance", "heritability"):
        assert kind in c.ClaimKind.__args__
    assert c.Statistic(kind="anova", denominator="plots in the analysis").kind == "anova"


def test_the_answer_shows_the_anova_table_the_precision_lines_and_the_descriptive_sentence(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_anova_answer")
    r = anova(ctx.registry, save(ctx.registry, plots(RICE)))
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claims(r), caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    shown = cp.answer_summary(view)
    assert "| Source | df | SS | MS | F | p |" in shown
    assert "| Entry | 5 | 1.2671 | 0.2534 | 2.1261 | 0.1184 |" in shown and "| Error | 15 | 1.7879 | 0.1192 |  |  |" in shown
    assert "CV 6.9688 %" in shown and "LSD (5%) 0.5203 t/ha" in shown and "Entry-mean heritability (H²) 0.5297" in shown
    assert cp.ANOVA_SENTENCE in shown and "| Statistic | Result |" not in shown


# --- 5. the guards of the first review (2026-10-22) -----------------------------------------------------

def answer(ctx, r) -> str:
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claims(r), caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    return cp.answer_summary(view)


def test_identical_plots_within_entries_give_no_f_and_say_why(tmp_path):
    """MS error is 0 (in floating point -2e-14, or a tiny positive number): no crash, no F or p, and the reason is shown."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_anova_zero")
    for means in ((19.5, 33.6, 8.0, 15.1, 41.18, 23.64, 35.71, 19.75), (12.19, 24.74, 13.1, 26.68, 7.9)):
        values = {f"E{i}": [m] * 6 for i, m in enumerate(means)}
        r = anova(ctx.registry, save(ctx.registry, plots(values, reps=False), rep_column=False))
        assert r.ok, r.error
        t = table(r)
        assert t["error"]["ss"] == 0 and t["entry"]["f"] is None and t["entry"]["p"] is None
        f = next(k for k in claims(r) if k.transformation.startswith("one-way ANOVA: f of entry"))
        lsd = next(k for k in claims(r) if k.transformation.startswith("LSD (5%)"))
        assert f.value is None and "error mean square is 0" in f.missing_reason
        assert lsd.value is None and "error mean square is 0" in lsd.missing_reason
    shown = answer(ctx, r)
    assert "| Entry | 4 |" in shown and "| n/a | n/a |" in shown and "F and p not available: the error mean square is 0" in shown


def test_an_exactly_additive_rcbd_gives_no_f(registry):
    values = {e: [base + offset for offset in (0.0, 1.5, 3.25)] for e, base in (("A", 10.0), ("B", 12.5), ("C", 9.75), ("D", 14.0))}
    r = anova(registry, save(registry, plots(values)))
    assert r.ok, r.error
    t = table(r)
    assert t["error"]["ss"] == 0 and t["entry"]["f"] is None and t["rep"]["f"] is None


def test_several_dates_of_one_plot_are_refused_or_averaged_per_plot(registry):
    """Each plot measured on 3 dates: 18 rows are 6 plots, not 18 (error df 3, not 15)."""
    rows = [(f"P{e}{j}", e, None, repr(v + day / 10), f"2025-0{day}-01") for e, vs in {"A": (10.0, 12.0), "B": (14.0, 15.0), "C": (9.0, 11.0)}.items()
            for j, v in enumerate(vs, 1) for day in (1, 2, 3)]
    aid = save(registry, rows, rep_column=False)
    r = anova(registry, aid)
    assert not r.ok and r.error.code == "unsupported" and "has 3 values" in r.error.message and "mean_per_unit" in r.error.message
    r = anova(registry, aid, duplicate_policy="mean_per_unit")
    assert r.ok, r.error
    assert r.data["plots_used"] == 6 and table(r)["error"]["df"] == 3
    assert all("averaged over its dates" in k.transformation and "used as recorded" not in k.transformation for k in claims(r))
    # per-plot means 10.2, 12.2 / 14.2, 15.2 / 9.2, 11.2: error SS 2 + 0.5 + 2 = 4.5 on 3 df
    assert table(r)["error"]["ss"] == pytest.approx(4.5, rel=1e-9) and table(r)["error"]["ms"] == pytest.approx(1.5, rel=1e-9)


def test_one_plot_named_with_two_entries_is_refused(registry):
    rows = plots(RICE)
    rows = [(u, "50" if (u == "P25-1" and k == 1) else e, rep, v, f"2025-0{k + 1}-01") for u, e, rep, v in rows for k in ((0, 1) if u == "P25-1" else (0,))]
    r = anova(registry, save(registry, rows), duplicate_policy="mean_per_unit")
    assert not r.ok and "plot P25-1" in r.error.message


def test_equal_plot_counts_with_empty_cells_are_labelled_approximate(registry):
    values = {e: [10.0 + i + 0.7 * j + (0.3 if (i + j) % 2 else -0.2) for j in range(4)] for i, e in enumerate("ABCD")}
    r = anova(registry, save(registry, plots(values, drop=(("A", 1), ("B", 2), ("C", 3), ("D", 4)))))
    assert r.ok, r.error
    d = r.data
    assert d["adjusted"] is True and d["replication"] == 3 and "approximate" in d["replication_basis"]
    assert any("approximate" in w for w in r.warnings)
    lsd = next(k for k in claims(r) if k.transformation.startswith("LSD (5%)"))
    assert "approximate" in lsd.transformation


def test_small_values_keep_their_digits_in_the_answer(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_anova_small")
    small = {e: [v / 1000 for v in vs] for e, vs in RICE.items()}
    shown = answer(ctx, anova(ctx.registry, save(ctx.registry, plots(small))))
    assert "| Entry | 5 | 1.267e-06 | 2.534e-07 | 2.1261 | 0.1184 |" in shown
    assert "genotype variance 3.356e-08" in shown


def test_blocks_within_reps_are_named_as_not_modelled(registry):
    rows = [(f"P{e}-{j}", e, str(j), repr(float(v))) for e, vs in RICE.items() for j, v in enumerate(vs[:2], 1)]       # 6 entries x 2 reps
    entries = list(RICE)
    blocks = {f"P{e}-{j}": f"{j}-{entries.index(e) // 2 + 1}" for e in entries for j in (1, 2)}                          # 3 blocks of 2 per rep
    r = anova(registry, save(registry, rows, blocks=blocks))
    assert r.ok, r.error
    assert any("incomplete blocks" in w for w in r.warnings)
    assert all("incomplete blocks" in k.transformation for k in claims(r))
    one_way = [(u, e, None, v) for u, e, _, v in rows]
    r = anova(registry, save(registry, one_way, blocks=blocks))
    assert r.ok and r.data["design"] == "one-way" and any("blocks" in w and "not modelled" in w for w in r.warnings)


def test_rep_codes_equal_as_numbers_but_written_differently_are_refused(registry):
    rows = [(u, e, "1.0" if (u == "P25-1") else rep, v) for u, e, rep, v in plots(RICE)]
    r = anova(registry, save(registry, rows))
    assert not r.ok and r.error.code == "unsupported" and "'1' and '1.0'" in r.error.message


def test_an_undefined_cv_is_shown_with_its_reason(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_anova_cv")
    below = {e: [v - 10 for v in vs] for e, vs in RICE.items()}                                                            # grand mean below 0
    shown = answer(ctx, anova(ctx.registry, save(ctx.registry, plots(below))))
    assert "CV not available (the grand mean is not above 0)" in shown


# --- 6. the second review's findings (2026-10-22) ------------------------------------------------------

def test_a_large_offset_keeps_a_real_error_variance(registry):
    """One entry 1e6 above the rest: the error SS is 6e-13 of SS total, real, not floating residue (round 2 zeroed it)."""
    np = pytest.importorskip("numpy")
    noise = [[0.3, -0.5, 0.2], [-0.1, 0.4, -0.3], [0.25, -0.2, -0.05], [0.6, -0.4, -0.2], [-0.35, 0.1, 0.25]]
    values = {f"E{i}": [(1e6 if i == 0 else 0.0) + 10.0 + i + j + noise[i][j] for j in range(3)] for i in range(5)}
    rows = plots(values)
    r = anova(registry, save(registry, rows))
    assert r.ok, r.error
    y = np.array([float(p[3]) for p in rows])
    x = np.column_stack([np.ones(len(rows))] + [[1.0 if p[2] == q else 0.0 for p in rows] for q in ("2", "3")]
                        + [[1.0 if p[1] == e else 0.0 for p in rows] for e in list(values)[1:]])
    residual = float(np.sum((y - x @ np.linalg.lstsq(x, y, rcond=None)[0]) ** 2))
    assert residual > 0.1 and table(r)["error"]["ss"] == pytest.approx(residual, rel=1e-6) and table(r)["entry"]["f"] is not None


def test_a_large_offset_keeps_a_real_error_variance_one_way(registry):
    """The same offset without reps: the error SS is the plots' spread around their entry means, not total SS minus entry SS."""
    np = pytest.importorskip("numpy")
    noise = [[0.3, -0.5, 0.2], [-0.1, 0.4, -0.3], [0.25, -0.2, -0.05], [0.6, -0.4, -0.2], [-0.35, 0.1, 0.25]]
    values = {f"E{i}": [(1e6 if i == 0 else 0.0) + 10.0 + i + j + noise[i][j] for j in range(3)] for i in range(5)}
    r = anova(registry, save(registry, plots(values, reps=False), rep_column=False))
    assert r.ok, r.error
    within = float(sum(np.sum((np.array(v) - np.mean(v)) ** 2) for v in values.values()))
    assert r.data["design"] == "one-way" and within > 1 and table(r)["error"]["ss"] == pytest.approx(within, rel=1e-8)

def test_rep_codes_are_compared_as_the_plot_list_compares_them(registry):
    rows = [(u, e, "01" if u == "P25-1" else rep, v) for u, e, rep, v in plots(RICE)]
    r = anova(registry, save(registry, rows))
    assert not r.ok and "'01' and '1'" in r.error.message
    long_codes = {"1": "100000000000000000000000000001", "2": "100000000000000000000000000002", "3": "3", "4": "4"}
    r = anova(registry, save(registry, [(u, e, long_codes[rep], v) for u, e, rep, v in plots(RICE)]))
    assert r.ok and r.data["balance"]["reps"] == 4                          # 30 digits differing in the last: two reps
    r = anova(registry, save(registry, [(u, e, "1E+99999999" if rep == "1" else rep, v) for u, e, rep, v in plots(RICE)]))
    assert r.ok and r.data["balance"]["reps"] == 4                          # a huge exponent is a code, not an overflow
    r = anova(registry, save(registry, [(u, e, "1_0" if rep == "1" else "10" if rep == "2" else rep, v) for u, e, rep, v in plots(RICE)]))
    assert r.ok and r.data["balance"]["reps"] == 4                          # "1_0" is no numeral: not the number 10



def test_a_small_p_is_shown_with_its_digits(tmp_path):
    """T3 review 1 and Luna A40.4 (2026-10-23): "< 0.0001" is a number no claim carries; the p itself is shown instead."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_anova_p")
    values = {"A": [10.0, 10.1, 9.9], "B": [20.0, 20.1, 19.9], "C": [30.0, 30.2, 29.8], "D": [40.0, 39.9, 40.1]}
    r = anova(ctx.registry, save(ctx.registry, plots(values, reps=False), rep_column=False))
    p = table(r)["entry"]["p"]
    shown = answer(ctx, r)
    assert p < 1e-4 and "< 0.0001" not in shown and f"| {cp._anova_number(p)} |" in shown



def test_a_p_below_what_a_double_holds_is_shown_as_a_bound(tmp_path):
    """T3 review 2: an F tail under 1e-308 is 0.0; the table showed "0". 300 entries x 3 plots, entries far apart."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_anova_p0")
    values = {f"E{i:03d}": [float(i), i + 0.01, i - 0.01] for i in range(300)}
    r = anova(ctx.registry, save(ctx.registry, plots(values, reps=False), rep_column=False))
    p = next(k for k in claims(r) if k.transformation.startswith("one-way ANOVA: p of entry"))
    assert p.value == 0.0 and "p below 1e-300" in p.transformation and "| &lt; 1e-300 |" in answer(ctx, r)



# --- 7. Luna 2026-10-23 (A40.5, E16.1-E16.5): headline claims first, the coordinator's view, the label ---------------

HEADLINE = ("entry-mean heritability H2", "genotype variance sigma2 g", "error variance sigma2 e", "RCBD ANOVA: f of entry",
            "RCBD ANOVA: p of entry", "RCBD ANOVA: ms of entry", "RCBD ANOVA: ms of error")


def test_the_coordinators_view_holds_the_headline_numbers_of_one_trial(registry):
    """A40.5: the coordinator sees 20 claims; H2 was claim 23 of 23 and never reached it, so the run stopped."""
    import coordination_runtime as cr
    r = anova(registry, save(registry, plots(RICE, drop=(("50", 2),))))
    assert len(claims(r)) > cr.VIEW_CLAIMS
    seen = {k.transformation.split(" | ")[0] for k in cr._view_claims(claims(r), set())}
    assert set(HEADLINE) <= seen, set(HEADLINE) - seen


def test_the_coordinators_view_holds_every_trials_headline_numbers(registry):
    """E16.1-E16.5: two or more anova_trial calls; the view held only the first study's claims, so the run stopped."""
    import coordination_runtime as cr
    first = anova(registry, save(registry, plots(RICE, drop=(("50", 2),)), study="S1"))
    second = anova(registry, save(registry, plots(RICE, drop=(("75", 3),)), study="S2"))
    third = anova(registry, save(registry, plots(RICE), study="S3"))
    view = cr._view_claims(claims(first) + claims(second) + claims(third), set())
    for r in (first, second, third):
        ids = {k.claim_id for k in claims(r) if k.transformation.split(" | ")[0] in ("entry-mean heritability H2", "error variance sigma2 e")}
        assert ids <= {k.claim_id for k in view}


def test_missing_plots_are_labelled_as_the_sequential_ols_fit(tmp_path):
    """E16.1/E16.2: "adjusted" was read as not the requested sequential OLS fit; it is that fit, reps first."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_anova_label")
    r = anova(ctx.registry, save(ctx.registry, plots(RICE, drop=(("50", 2),))))
    shown = answer(ctx, r)
    assert "Sequential least squares, reps first \\(the OLS fit\\): 1 entry x rep cell\\(s\\) without a plot." in shown and "Adjusted:" not in shown
    assert any("sequential least squares, reps first (the OLS fit)" in w for w in r.warnings)
    assert all("sequential least squares, reps first (the OLS fit)" in k.transformation for k in claims(r))



# --- 9. the fresh review of fixes (1)-(4), 2026-10-23 ----------------------------------------------------------------

def test_mixed_calls_keep_each_trials_mean_squares_in_view(registry):
    """Review: E16.2 (2 anova_trial + 2 numeric_summary calls) lost MS of entry from the coordinator's view."""
    import coordination_runtime as cr
    all_claims = []
    for study, drop in (("S1", (("50", 2),)), ("S2", (("75", 3),))):
        tid = save(registry, plots(RICE, drop=drop), study=study)
        all_claims += claims(anova(registry, tid))
        all_claims += [Claim.model_validate(k) for k in at.dispatch_analyst(registry, "numeric_summary", {"artifact_id": tid}).data["claims"]]
    seen = [(k.source_artifact_hashes[0], k.transformation.split(" | ")[0]) for k in cr._view_claims(all_claims, set())]
    for name in ("entry-mean heritability H2", "RCBD ANOVA: ms of entry", "RCBD ANOVA: ms of error", "genotype variance sigma2 g"):
        assert sum(1 for _, n in seen if n == name) == 2, name


def test_missing_plots_speak_of_least_squares_means(registry):
    """Review: "adjusted entry means" in the precision note kept the word the coordinator misread."""
    drop = tuple(zip(RICE, (1, 2, 3, 4, 1, 2)))                           # every entry 3 plots, 6 cells empty: the equal-count note
    r = anova(registry, save(registry, plots(RICE, drop=drop)))
    assert all("least-squares entry means" in k.transformation and "adjusted entry means" not in k.transformation for k in claims(r))



# --- 8. the C28 cell rule (coach 2026-10-23 00:49 UTC): default refuse, two opt-ins only with the person's answer -------

def doubled_rice(*, drop=()):
    """RICE with two cells holding two plots: entry 50 in rep 2 (6.0 and 5.9), entry 75 in rep 1 (5.3 and 5.1)."""
    rows = plots(RICE, drop=drop)
    return rows + [("P50-2b", "50", "2", "5.9"), ("P75-1b", "75", "1", "5.1")]


def test_two_plots_in_one_cell_are_still_refused_by_default_and_the_opt_ins_are_named(registry):
    r = anova(registry, save(registry, doubled_rice()))
    assert not r.ok and "entry 50 has 2 plots in rep 2" in r.error.message
    assert "cells='mean'" in r.error.message and "cells='single_only'" in r.error.message and "if the person asks" in r.error.message


def test_cells_mean_averages_a_cells_plots(registry):
    r = anova(registry, save(registry, doubled_rice()), cells="mean")
    assert r.ok, r.error
    averaged = {"50": [5.3, 5.95, 4.7, 4.3], "75": [5.2, 5.7, 5.5, 4.7]}
    reference = anova(registry, save(registry, plots({**RICE, **averaged})))
    for source in ("rep", "entry", "error"):
        for quantity in ("ss", "ms"):
            assert table(r)[source][quantity] == pytest.approx(table(reference)[source][quantity], rel=1e-12)
    assert r.data["plots_used"] == 24 and r.data["cells_rule"] == "mean"
    assert all("cells='mean': the plots of 2 entry x rep cell(s) averaged (4 plots into one value each)" in k.transformation for k in claims(r))
    assert any("cells='mean'" in w for w in r.warnings)


def test_cells_single_only_keeps_the_entries_with_one_plot_in_every_rep(registry):
    """50 has two plots in rep 2, 75 misses rep 3. Not the C28 gold's set: a named check with one plot per rep stays (review)."""
    rows = [row for row in doubled_rice() if row[0] not in ("P75-1b", "P75-3")]
    r = anova(registry, save(registry, rows), cells="single_only")
    assert r.ok, r.error
    kept = {e: v for e, v in RICE.items() if e not in ("50", "75")}
    reference = anova(registry, save(registry, plots(kept)))
    for source in ("rep", "entry", "error"):
        assert table(r)[source]["ss"] == pytest.approx(table(reference)[source]["ss"], rel=1e-12)
    note = ("cells='single_only': the entries with exactly one plot in every rep, 4 of 6 (1 with several plots in a rep and 1 with a rep "
            "missing left out)")
    assert all(note in k.transformation for k in claims(r)) and r.data["plots_used"] == 16
    assert all(k.n_independent_units == 16 for k in claims(r))
    left = {e.reason: e.count for e in claims(r)[0].exclusions}
    assert left["plots left out by cells='single_only': entries with several plots in a rep"] == 5
    assert left["plots left out by cells='single_only': entries with a rep missing"] == 3


def test_a_rule_that_changes_nothing_keeps_the_default_ids_and_one_that_acts_has_its_own(registry):
    """Review: on a table without a doubled cell a rule changed nothing but issued a second set of claim IDs."""
    tid = save(registry, plots(RICE))
    plain = {k.claim_id for k in claims(anova(registry, tid))}
    for rule in ("refuse", "mean", "single_only"):
        assert plain == {k.claim_id for k in claims(anova(registry, tid, cells=rule))}, rule
    doubled = save(registry, doubled_rice())
    assert {k.claim_id for k in claims(anova(registry, doubled, cells="mean"))}.isdisjoint(
        {k.claim_id for k in claims(anova(registry, doubled, cells="single_only"))})


def test_the_analyst_never_picks_a_cell_rule_itself():
    assert "cells=\"mean\"" in an.ANALYST_SYSTEM_PROMPT and "cells=\"single_only\"" in an.ANALYST_SYSTEM_PROMPT
    assert "never choose one yourself" in an.ANALYST_SYSTEM_PROMPT



def test_cells_mean_counts_cells_and_keeps_the_plot_units(registry):
    """Review: under mean the replication note said "plot counts" for cells, and n_independent_units was the cell count."""
    r = anova(registry, save(registry, doubled_rice()), cells="mean")
    assert "every entry has 4 cell(s)" in r.data["replication_basis"] and "plot(s)" not in r.data["replication_basis"]
    assert all(k.n_independent_units == 26 for k in claims(r))                  # 26 plots, 24 cells
    assert "the error mean square mixes single plots and cell means: approximate" in r.data["cells_note"]
    lineage = at._load(registry, r.data["result_artifact_id"]).manifest.quality_notes
    assert any("cells='mean'" in note for note in lineage)


def test_the_prompt_keeps_the_cells_rule_for_the_person():
    """Review: rule 4 told the analyst to name the aggregation rule a refusal asks for."""
    assert "anova_trial's cells rule is never yours to name" in an.ANALYST_SYSTEM_PROMPT
