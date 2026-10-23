"""
tests/test_gxe_partition.py - T3 "gxe_partition" (coach 2026-10-22 22:50 / 23:50 UTC, NEXT 3; research build order, tool 5): the
genotype x environment partition of ONE trait's entry means across studies (each study one environment), on the entries present
in every study: environment, entry and entry x environment sums of squares with df, MS and their shares, the F for entries
tested against entry x environment with its p, and the across-environment entry-mean heritability H2 = 1 - MS_GE / MS_G. With
plots that state their rep in complete RCBDs of equal r: the pooled within-trial error and the variance components
sigma2 e, sigma2 ge = MS_GE - MS_e / r and sigma2 g = (MS_G - MS_GE) / e on the means scale, and r_B = sigma2 g / (sigma2 g +
sigma2 ge). Also rank_entries' basis "study_means" (the unweighted mean of an entry's per-study means). Descriptive numbers,
never a decision.

Fixtures: SYNTHETIC tables checked against an independent numpy computation (the additive two-way least-squares fit of the
means table, and each study's RCBD fit for the pooled error; numpy only in this test). The metan data_ge values (research notes
Q3: means-table SS 93.191184 / 4.331681 / 10.406522, H2 0.8151984) are added once the raw data can be read. No network, no model.
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
import pvalues
from artifacts import ArtifactRegistry
from contracts import AnalysisReport, Claim, MeasurementMeta
from part2_support import SYNTHETIC_BASE, synthetic_context

STUDIES = ["S1", "S2", "S3", "S4"]
ENTRIES = ["E0", "E1", "E2", "E3", "E4"]
G = [0.0, 1.5, -1.0, 2.5, 0.5]
ENV = [0.0, 3.0, -2.0, 5.0]


def trial(*, r: int = 3, g_scale: float = 1.0, ge_scale: float = 1.0, entries=ENTRIES, studies=STUDIES) -> dict[str, dict[str, list[float]]]:
    """study -> entry -> plot values (rep k + 1 for the k-th): fixed effects, a pure (double-centred) G x E term and a fixed
    irregular plot error."""
    raw = [[math.cos(1.3 * i + 0.7 * j) for j in range(len(studies))] for i in range(len(entries))]
    row = [sum(x) / len(x) for x in raw]
    col = [sum(raw[i][j] for i in range(len(entries))) / len(entries) for j in range(len(studies))]
    ge = [[raw[i][j] - row[i] - col[j] + sum(row) / len(row) for j in range(len(studies))] for i in range(len(entries))]
    return {s: {e: [10.0 + g_scale * G[i % 5] + ENV[j % 4] + ge_scale * 1.2 * ge[i][j] + 0.25 * k * (j - 1.5)
                    + 0.3 * math.sin(1.7 * i + 2.3 * j + 3.1 * k) for k in range(r)]
                for i, e in enumerate(entries)}
            for j, s in enumerate(studies)}


@pytest.fixture
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "out", "run_gxe", base_url=SYNTHETIC_BASE)


def rows_of(values, *, reps=True, drop=(), names=None) -> list[dict[str, str]]:
    """Measurement rows of every study; drop: (study, entry, rep) plots left out; names: entry -> name."""
    rows = []
    for study, by_entry in values.items():
        for entry, plot_values in by_entry.items():
            for k, value in enumerate(plot_values, 1):
                if (study, entry, k) in drop:
                    continue
                unit = f"{study}-{entry}-{k}"
                row = {"observationDbId": f"O{unit}", "observationUnitDbId": unit, "observationUnitName": f"plot {unit}",
                       "germplasmDbId": entry, "germplasmName": (names or {}).get(entry, f"clone {entry}"), "studyDbId": study,
                       "observationVariableDbId": "V1", "observationVariableName": "SYNTHETIC yield", "observationTimeStamp": "",
                       "value": repr(float(value))}
                if reps:
                    row["rep"] = str(k)
                rows.append(row)
    return rows


def save(registry: ArtifactRegistry, rows, *, complete=True, variable="V1") -> str:
    for row in rows:
        row["observationVariableDbId"] = variable
    columns = {col: srv.OBSERVATION_COLUMNS.get(col, "text") for col in rows[0]}
    return registry.save_table(rows, kind="observations", columns=columns, complete=complete, study_ids=sorted({r["studyDbId"] for r in rows}),
                               variable_ids=[variable], measurement=MeasurementMeta(trait="SYNTHETIC yield", unit="t/ha"),
                               quality_notes=["SYNTHETIC"]).artifact_id


def per_study(registry, rows) -> list[str]:
    return [save(registry, [dict(r) for r in rows if r["studyDbId"] == s]) for s in sorted({r["studyDbId"] for r in rows})]


def gxe(registry, artifact_ids, **args):
    return at.dispatch_analyst(registry, "gxe_partition", {"artifact_ids": artifact_ids, **args})


def table(r) -> dict[str, dict]:
    return {row["source"]: row for row in r.data["anova"]}


def claims(r) -> list[Claim]:
    return [Claim.model_validate(k) for k in r.data["claims"]]


def expected(values, *, with_error=True) -> dict[str, float]:
    """The partition and the pooled error by an independent numpy computation."""
    np = pytest.importorskip("numpy")
    studies = sorted(values)
    entries = sorted(values[studies[0]])
    m = np.array([[np.mean(values[s][e]) for s in studies] for e in entries])
    g, e = m.shape
    grand = m.mean()
    out = {"ss_env": g * float(((m.mean(axis=0) - grand) ** 2).sum()), "ss_entry": e * float(((m.mean(axis=1) - grand) ** 2).sum()),
           "ss_ge": float(((m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + grand) ** 2).sum()), "grand": float(grand),
           "g": g, "e": e}
    x = np.column_stack([np.ones(g * e)] + [[1.0 if i == a else 0.0 for i in range(g) for _ in range(e)] for a in range(1, g)]
                        + [[1.0 if j == b else 0.0 for _ in range(g) for j in range(e)] for b in range(1, e)])
    y = m.reshape(-1)
    out["ss_ge_lstsq"] = float(((y - x @ np.linalg.lstsq(x, y, rcond=None)[0]) ** 2).sum())
    if with_error:
        ss_e = df_e = 0.0
        for s in studies:
            plots = [(i, k, v) for i, en in enumerate(entries) for k, v in enumerate(values[s][en])]
            r = 1 + max(k for _, k, _ in plots)
            xs = np.column_stack([np.ones(len(plots))] + [[1.0 if i == a else 0.0 for i, _, _ in plots] for a in range(1, g)]
                                 + [[1.0 if k == b else 0.0 for _, k, _ in plots] for b in range(1, r)])
            ys = np.array([v for _, _, v in plots])
            ss_e += float(((ys - xs @ np.linalg.lstsq(xs, ys, rcond=None)[0]) ** 2).sum())
            df_e += len(plots) - g - r + 1
        out["ms_e"] = ss_e / df_e
        out["r"] = r
    return out


# --- 1. the partition of a complete means table ------------------------------------------------------

def test_the_partition_matches_an_independent_numpy_fit(registry):
    values = trial()
    r = gxe(registry, [save(registry, rows_of(values, reps=False))])
    assert r.ok, r.error
    x, t = expected(values, with_error=False), table(r)
    assert x["ss_ge"] == pytest.approx(x["ss_ge_lstsq"], rel=1e-9)                      # the closed form is the least-squares residual
    assert (t["environment"]["df"], t["entry"]["df"], t["entry x environment"]["df"], t["total"]["df"]) == (3, 4, 12, 19)
    assert t["environment"]["ss"] == pytest.approx(x["ss_env"], rel=1e-9) and t["entry"]["ss"] == pytest.approx(x["ss_entry"], rel=1e-9)
    assert t["entry x environment"]["ss"] == pytest.approx(x["ss_ge"], rel=1e-9)
    assert t["total"]["ss"] == pytest.approx(x["ss_env"] + x["ss_entry"] + x["ss_ge"], rel=1e-9)
    ms_g, ms_ge = x["ss_entry"] / 4, x["ss_ge"] / 12
    assert t["entry"]["f"] == pytest.approx(ms_g / ms_ge, rel=1e-9) and t["entry"]["p"] == pytest.approx(pvalues.f_upper_p(ms_g / ms_ge, 4, 12), rel=1e-12)
    assert t["environment"]["f"] is None and t["entry x environment"]["f"] is None                # only entries are tested, against G x E
    shares = r.data["shares"]
    assert sum(shares.values()) == pytest.approx(100.0, rel=1e-12)
    assert shares["environment"] == pytest.approx(100 * x["ss_env"] / (x["ss_env"] + x["ss_entry"] + x["ss_ge"]), rel=1e-9)
    assert r.data["h2"] == pytest.approx(1 - ms_ge / ms_g, rel=1e-12) and r.data["grand_mean"] == pytest.approx(x["grand"], rel=1e-12)
    assert r.data["studies"] == STUDIES and r.data["entries_kept"] == 5 and r.data["entries_dropped"] == 0
    assert r.data["variance_components"]["available"] is False and "rep" in r.data["variance_components"]["reason"]


def test_one_table_per_study_gives_the_same_partition(registry):
    rows = rows_of(trial(), reps=False)
    stacked = gxe(registry, [save(registry, [dict(x) for x in rows])])
    split = gxe(registry, per_study(registry, rows))
    assert stacked.ok and split.ok, (stacked.error, split.error)
    for source in ("environment", "entry", "entry x environment", "total"):
        assert table(split)[source]["ss"] == pytest.approx(table(stacked)[source]["ss"], rel=1e-12)
    assert split.data["h2"] == pytest.approx(stacked.data["h2"], rel=1e-12)


# --- 2. with reps: the pooled error and the variance components -----------------------------------------

def test_with_reps_the_pooled_error_and_the_variance_components_match_numpy(registry):
    values = trial()
    r = gxe(registry, per_study(registry, rows_of(values)))
    assert r.ok, r.error
    x, vc = expected(values), r.data["variance_components"]
    ms_g, ms_ge = x["ss_entry"] / 4, x["ss_ge"] / 12
    assert vc["available"] is True and vc["r"] == 3 and vc["df_error"] == 4 * (15 - 5 - 3 + 1)
    assert vc["sigma2_e"] == pytest.approx(x["ms_e"], rel=1e-9)
    assert vc["sigma2_ge"] == pytest.approx(ms_ge - x["ms_e"] / 3, rel=1e-9) and vc["sigma2_g"] == pytest.approx((ms_g - ms_ge) / 4, rel=1e-9)
    assert vc["r_b"] == pytest.approx(vc["sigma2_g"] / (vc["sigma2_g"] + vc["sigma2_ge"]), rel=1e-12)
    long_form = vc["sigma2_g"] / (vc["sigma2_g"] + vc["sigma2_ge"] / 4 + vc["sigma2_e"] / 12)
    assert long_form == pytest.approx(r.data["h2"], rel=1e-12) and vc["h2_long_form"] == pytest.approx(long_form, rel=1e-12)


def test_unequal_reps_or_a_missing_plot_give_no_variance_components(registry):
    values = trial()
    values["S2"] = {e: v[:2] for e, v in values["S2"].items()}                    # S2 has 2 reps, the others 3
    r = gxe(registry, per_study(registry, rows_of(values)))
    assert r.ok and r.data["variance_components"]["available"] is False and "mixed model" in r.data["variance_components"]["reason"]
    assert next(k for k in claims(r) if k.kind == "variance").value is None
    r = gxe(registry, per_study(registry, rows_of(trial(), drop=(("S3", "E1", 2),))))
    assert r.ok and r.data["variance_components"]["available"] is False and "mixed model" in r.data["variance_components"]["reason"]
    assert r.data["entries_kept"] == 5                                            # the means table is still complete


def test_a_negative_genotype_variance_is_set_to_zero_and_flagged(registry):
    values = trial(g_scale=0.0, ge_scale=4.0)                                     # no entry effect, strong G x E
    r = gxe(registry, per_study(registry, rows_of(values)))
    assert r.ok, r.error
    t = table(r)
    assert t["entry"]["ms"] < t["entry x environment"]["ms"]
    assert r.data["h2"] == 0.0 and r.data["h2_set_to_zero"] is True
    vc = r.data["variance_components"]
    assert vc["sigma2_g"] == 0.0 and vc["sigma2_g_set_to_zero"] is True and vc["r_b"] == 0.0
    assert any("set to 0" in k.transformation for k in claims(r) if k.kind == "heritability")


def test_an_exactly_additive_means_table_gives_no_f_and_says_why(registry):
    values = {s: {e: [1.0 + i + 0.25 * j] for i, e in enumerate(ENTRIES)} for j, s in enumerate(STUDIES)}
    r = gxe(registry, [save(registry, rows_of(values, reps=False))])
    assert r.ok, r.error
    t = table(r)
    assert t["entry x environment"]["ss"] == 0.0 and t["entry"]["f"] is None and t["entry"]["p"] is None
    f_claim = next(k for k in claims(r) if k.kind == "anova" and k.transformation.startswith("means-table ANOVA: f of entry"))
    assert f_claim.value is None and "entry x environment mean square is 0" in f_claim.missing_reason


def test_an_additive_table_with_floating_residue_still_gives_no_f(registry):
    """Additive means that are not exact in binary (0.1 steps): the residue (about 1e-30) is not interaction."""
    values = {s: {e: [12.19 + 0.1 * i + 2.3 * j] for i, e in enumerate(ENTRIES)} for j, s in enumerate(STUDIES)}
    r = gxe(registry, [save(registry, rows_of(values, reps=False))])
    assert r.ok and table(r)["entry x environment"]["ss"] == 0.0 and table(r)["entry"]["f"] is None and r.data["h2"] == 1.0


def test_two_dates_of_one_plot_are_refused_or_averaged(registry):
    rows = rows_of(trial(), reps=False)
    extra = [dict(row, observationDbId=row["observationDbId"] + "-d2", observationTimeStamp="2026-03-01", value=repr(float(row["value"]) + 1.0))
             for row in rows if row["studyDbId"] == "S1"]
    for row in rows:
        if row["studyDbId"] == "S1":
            row["observationTimeStamp"] = "2026-02-01"
    r = gxe(registry, [save(registry, rows + extra)])
    assert not r.ok and "has 2 values" in r.error.message and "duplicate_policy='mean_per_unit'" in r.error.message
    r = gxe(registry, [save(registry, [dict(x) for x in rows + extra])], duplicate_policy="mean_per_unit")
    assert r.ok and "averaged over its dates (30 values in 15 plots)" in next(k for k in claims(r) if k.kind == "anova").transformation
    plain = gxe(registry, [save(registry, rows_of(trial(), reps=False))])
    s1 = table(r)["environment"]["ss"] != table(plain)["environment"]["ss"]          # S1 means are 0.5 higher: environment SS moves
    assert s1 and table(r)["entry"]["ss"] == pytest.approx(table(plain)["entry"]["ss"], rel=1e-9)


def test_two_tables_holding_the_same_plot_are_refused(registry):
    rows = rows_of(trial())
    s1 = [dict(x) for x in rows if x["studyDbId"] == "S1"]
    both = [dict(x) for x in rows if x["studyDbId"] in ("S1", "S2")]
    r = gxe(registry, [save(registry, s1), save(registry, both)])
    assert not r.ok and "is in both" in r.error.message


def test_rep_codes_written_two_ways_give_no_variance_components(registry):
    rows = rows_of(trial())
    for row in rows:
        if row["studyDbId"] == "S2" and row["rep"] == "1" and row["germplasmDbId"] in ("E0", "E1"):
            row["rep"] = "1.0"
    r = gxe(registry, per_study(registry, rows))
    assert r.ok and r.data["variance_components"]["available"] is False and "'1' and '1.0'" in r.data["variance_components"]["reason"]

# --- 3. which entries: every study, exclusions, a family -----------------------------------------------------

def test_entries_missing_from_a_study_are_dropped_and_counted(registry):
    values = trial()
    del values["S4"]["E3"]                                                        # E3 is not in S4
    r = gxe(registry, per_study(registry, rows_of(values)))
    assert r.ok and r.data["entries_kept"] == 4 and r.data["entries_dropped"] == 1 and r.data["dropped"][0]["entry"] == "E3"
    counts = {k.transformation.split(" | ")[0]: k.value for k in claims(r) if k.kind == "count"}
    assert counts["entries with a mean in every study"] == 4 and counts["entries dropped: not in every study"] == 1
    kept = {s: {e: v for e, v in values[s].items() if e != "E3"} for s in values}
    assert table(r)["entry"]["ss"] == pytest.approx(expected(kept, with_error=False)["ss_entry"], rel=1e-9)


def test_exclude_and_name_prefix_choose_the_entries(registry):
    names = {"E0": "Beauregard", "E1": "PH17.1", "E2": "PH17.2", "E3": "PH17.3", "E4": "Tanzania"}
    rows = rows_of(trial(), names=names)
    r = gxe(registry, per_study(registry, rows), exclude=["Beauregard"], name_prefix="PH17.")
    assert r.ok and r.data["entries_kept"] == 3 and {d["entry"] for d in r.data["dropped"]} == {"E0", "E4"}
    sub = {s: {e: v for e, v in trial()[s].items() if e in ("E1", "E2", "E3")} for s in STUDIES}
    assert table(r)["entry"]["ss"] == pytest.approx(expected(sub, with_error=False)["ss_entry"], rel=1e-9)


def test_refusals(registry):
    rows = rows_of(trial())
    one = save(registry, [dict(x) for x in rows if x["studyDbId"] == "S1"])
    r = gxe(registry, [one])
    assert not r.ok and "at least 2 studies" in r.error.message
    r = gxe(registry, [one, one])
    assert not r.ok and "same rows" in r.error.message
    first, second = save(registry, [dict(x) for x in rows if x["studyDbId"] == "S1"]), save(registry, [dict(x) for x in rows if x["studyDbId"] == "S2"], variable="V2")
    r = gxe(registry, [first, second])
    assert not r.ok and "one variable" in r.error.message
    lonely = {s: {("E0" if j < 2 else "E1"): [1.0, 2.0, 3.0], "E2": [4.0, 5.0, 6.0]} for j, s in enumerate(STUDIES)}
    r = gxe(registry, [save(registry, rows_of(lonely))])
    assert not r.ok and "at least 2 entries" in r.error.message
    twice = rows_of(trial(), reps=False) + [dict(rows_of(trial(), reps=False)[0], observationDbId="Oextra", value="99")]
    r = gxe(registry, [save(registry, twice)])
    assert not r.ok and "duplicate_policy='mean_per_unit'" in r.error.message
    r = gxe(registry, [save(registry, twice)], duplicate_policy="mean_per_unit")
    assert r.ok and "averaged" in next(k for k in claims(r) if k.kind == "anova").transformation


def test_every_number_is_a_claim(registry):
    r = gxe(registry, per_study(registry, rows_of(trial())))
    kinds = {k.kind for k in claims(r)}
    assert kinds == {"anova", "ratio", "mean", "heritability", "variance", "count"}
    for k in claims(r):
        assert k.claim_id.split("_")[4].startswith("gxe-") and "4 studies" in k.transformation and at.GXE_NOTE in k.transformation
    shares = [k for k in claims(r) if k.kind == "ratio" and k.transformation.startswith("share of the means-table SS")]
    assert len(shares) == 3 and all(k.unit == "%" for k in shares)
    assert next(k for k in claims(r) if k.kind == "mean").unit == "t/ha"



def test_the_coordinators_view_holds_the_headline_numbers(registry):
    """The coordinator sees 20 claims (VIEW_CLAIMS); on Luna, anova_trial's H2 was claim 25 and never reached it (A40.5)."""
    import coordination_runtime as cr
    r = gxe(registry, per_study(registry, rows_of(trial())))
    seen = {k.transformation.split(" | ")[0] for k in cr._view_claims(claims(r), set())}
    assert len(claims(r)) > cr.VIEW_CLAIMS
    for name in ("across-environment entry-mean heritability H2", "genotype variance sigma2 g", "genotype x environment variance sigma2 ge",
                 "error variance sigma2 e (pooled within trials)", "share of the means-table SS: entry x environment",
                 "means-table ANOVA: f of entry", "means-table ANOVA: p of entry", "means-table ANOVA: ms of entry"):
        assert name in seen, name

# --- 4. rank_entries: the mean of per-study means ---------------------------------------------------------------

def test_rank_entries_on_per_study_means_weights_each_study_equally(registry):
    """A misses its second plot in the better study: pooled B leads (34.5 vs 33.33); per-study means A leads (35 vs 34.5)."""
    rows = []
    for study, entry, rep, value in (("S1", "A", 1, 30), ("S1", "A", 2, 30), ("S2", "A", 1, 40), ("S1", "B", 1, 33), ("S1", "B", 2, 33),
                                     ("S2", "B", 1, 36), ("S2", "B", 2, 36), ("S1", "C", 1, 50)):
        rows.append({"observationDbId": f"O{study}{entry}{rep}", "observationUnitDbId": f"{study}-{entry}-{rep}", "observationUnitName": "p",
                     "germplasmDbId": entry, "germplasmName": f"clone {entry}", "studyDbId": study, "observationVariableDbId": "V1",
                     "observationVariableName": "SYNTHETIC yield", "observationTimeStamp": "", "value": str(value)})
    tid = save(registry, rows)
    pooled = at.dispatch_analyst(registry, "rank_entries", {"artifact_id": tid, "top_k": 3})
    assert [s["entry"] for s in pooled.data["stated"]] == ["C", "B", "A"]
    by_study = at.dispatch_analyst(registry, "rank_entries", {"artifact_id": tid, "top_k": 3, "basis": "study_means"})
    assert by_study.ok, by_study.error
    assert [(s["entry"], s["mean"]) for s in by_study.data["stated"]] == [("A", 35.0), ("B", 34.5)]
    assert by_study.data["not_ranked"] == [{"entry": "C", "label": "clone C", "reason": "not in every study of the table (1 of 2)"}]
    mean = next(Claim.model_validate(k) for k in by_study.data["claims"] if k["kind"] == "mean" and k["filters"]["germplasmDbId"] == "A")
    assert mean.denominator.n == 2 and "per-study means" in mean.transformation and "each study weighted equally" in mean.transformation
    assert {k["claim_id"] for k in by_study.data["claims"]}.isdisjoint({k["claim_id"] for k in pooled.data["claims"]})


def test_rank_entries_keeps_pooled_as_the_default(registry):
    rows = rows_of(trial(), reps=False)
    tid = save(registry, rows)
    r = at.dispatch_analyst(registry, "rank_entries", {"artifact_id": tid, "top_k": 5})
    pooled = {e: sum(v) / len(v) for e in ENTRIES for v in [[float(x["value"]) for x in rows if x["germplasmDbId"] == e]]}
    assert all(s["mean"] == pytest.approx(pooled[s["entry"]], rel=1e-12) for s in r.data["stated"])
    assert "per-study" not in next(k for k in r.data["claims"] if k["kind"] == "mean")["transformation"]


# --- 5. the answer and the wiring -------------------------------------------------------------------------------------

def answer(ctx, claim_list) -> str:
    view = SimpleNamespace(execution_status="completed", analysis=AnalysisReport(status="completed", claims=claim_list, caveats=[]),
                           retrieval=SimpleNamespace(status="completed", warnings=[]), ctx=ctx, notes=[])
    return cp.answer_summary(view)


def test_the_answer_shows_the_partition_the_shares_h2_and_the_sentence(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_gxe_answer")
    r = gxe(ctx.registry, per_study(ctx.registry, rows_of(trial())))
    shown = answer(ctx, claims(r))
    assert "**Genotype × environment partition of entry means**" in shown and "| Source | df | SS | MS | F | p |" in shown
    assert "| Entry × environment | 12 |" in shown and "| Environment | 3 |" in shown and "| Total | 19 |" in shown
    assert "Shares of the means-table sum of squares: environment " in shown and "Across-environment entry-mean heritability (H²) " in shown
    assert "F for entries is tested against entry × environment" in shown and "genotype × environment variance " in shown
    assert cp.GXE_SENTENCE in shown and "| Statistic | Result |" not in shown


def test_the_answer_names_the_ranking_basis(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_gxe_rank")
    tid = save(ctx.registry, rows_of(trial(), reps=False))
    r = at.dispatch_analyst(ctx.registry, "rank_entries", {"artifact_id": tid, "top_k": 2, "basis": "study_means"})
    shown = answer(ctx, [Claim.model_validate(k) for k in r.data["claims"]])
    assert "**Ranking by the mean of per-study entry means, highest first, of 5 entries**" in shown
    assert "each study weighted equally; n = studies averaged" in shown


def test_every_menu_and_prompt_names_the_tool():
    assert "gxe_partition" in at.ANALYST_TOOL_NAMES and len(at.ANALYST_TOOL_NAMES) == 15
    assert "gxe_partition" in an.NUMERIC_TOOLS and "gxe_partition" in an.DERIVING_TOOLS
    assert "fifteen fixed tools" in an.ANALYST_SYSTEM_PROMPT and "gxe_partition" in an.ANALYST_SYSTEM_PROMPT
    assert "basis=\"study_means\"" in an.ANALYST_SYSTEM_PROMPT
    assert "gxe_partition" in co.ANALYST_ACTIONS and "gxe_partition(artifact_ids=" in co.COORDINATOR_SYSTEM_PROMPT
    assert "basis=\"study_means\"" in co.COORDINATOR_SYSTEM_PROMPT
    schema = next(s for s in at.analyst_tool_schemas() if s["name"] == "gxe_partition")
    assert schema["inputSchema"]["required"] == ["artifact_ids"] and "environment" in schema["description"]
    assert "basis" in next(s for s in at.analyst_tool_schemas() if s["name"] == "rank_entries")["inputSchema"]["properties"]


# --- 6. the first review's findings (2026-10-23) ------------------------------------------------------------------

def test_a_small_p_is_shown_with_its_digits_and_graded_as_backed(tmp_path):
    """Review 1: "< 0.0001" is a number no claim carries; on the real E17 data 7 of 8 rows were graded invented."""
    import importlib.util
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_gxe_p")
    values = trial(g_scale=6.0)
    r = gxe(ctx.registry, [save(ctx.registry, rows_of(values, reps=False))])
    p = table(r)["entry"]["p"]
    assert p < 1e-4
    shown = answer(ctx, claims(r))
    assert "< 0.0001" not in shown and f"| {cp._anova_number(p)} |" in shown
    import sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("_bench_for_gxe", Path(at.__file__).resolve().parent / "eval" / "benchmark.py")
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    flagged = [n for n in bench._numbers_for_invention(bench.harness.code_written(shown))
               if not bench._backed(n[0], n[1], bench.claim_numbers([k.model_dump(mode="json") for k in claims(r)]))]
    assert flagged == []


def test_rank_entries_names_why_an_entry_is_not_ranked(registry):
    """Review 1: under study_means an entry missing from a study was said to have "fewer than 1 independent units"."""
    rows = rows_of(trial(), reps=False)
    rows = [x for x in rows if not (x["germplasmDbId"] == "E2" and x["studyDbId"] == "S3")]
    r = at.dispatch_analyst(registry, "rank_entries", {"artifact_id": save(registry, rows), "top_k": 5, "basis": "study_means"})
    reasons = {e["reason"]: e["count"] for e in r.data["exclusions"]}
    assert reasons.get("entries not ranked: not in every study of the table") == 1
    assert not any("independent units" in reason for reason in reasons)


def test_a_study_without_a_valid_value_is_not_counted(registry):
    """Review 1: a study whose rows are all NA made every entry "not in every study"."""
    rows = rows_of(trial(), reps=False)
    rows += [dict(x, studyDbId="S9", observationUnitDbId=x["observationUnitDbId"] + "-9", observationDbId=x["observationDbId"] + "-9", value="NA")
             for x in rows if x["studyDbId"] == "S1"]
    r = at.dispatch_analyst(registry, "rank_entries", {"artifact_id": save(registry, rows), "top_k": 5, "basis": "study_means"})
    assert r.ok and len(r.data["stated"]) == 5 and r.data["studies"] == STUDIES


def test_entries_not_ranked_keep_the_study_means_basis_in_the_saved_table(registry):
    rows = [x for x in rows_of(trial(), reps=False) if not (x["germplasmDbId"] == "E2" and (x["studyDbId"] == "S3" or x["observationUnitDbId"] == "S1-E2-3"))]
    r = at.dispatch_analyst(registry, "rank_entries", {"artifact_id": save(registry, rows), "top_k": 5, "basis": "study_means"})
    saved = {row["group_id"]: row for row in at._load(registry, r.data["result_artifact_id"]).rows}
    values = trial()
    expected = (sum(values["S1"]["E2"][:2]) / 2 + sum(values["S2"]["E2"]) / 3 + sum(values["S4"]["E2"]) / 3) / 3   # S1 has 2 plots: pooled differs
    assert saved["E2"]["rank"] == "" and float(saved["E2"]["mean"]) == pytest.approx(expected, rel=1e-12) and saved["E2"]["n_studies"] == "3"


def test_each_reason_for_no_variance_components_is_named(registry):
    """Review 1: unreplicated trials and a doubled cell both read "unequal replication or missing plots"."""
    one_rep = gxe(registry, per_study(registry, rows_of(trial(r=1))))
    assert one_rep.ok and "one rep" in one_rep.data["variance_components"]["reason"]
    rows = rows_of(trial())
    rows.append(dict(rows[0], observationDbId="Oextra", observationUnitDbId="S1-E0-extra"))
    doubled = gxe(registry, per_study(registry, rows))
    assert "entry E0 has 2 plots in rep 1 of study S1" in doubled.data["variance_components"]["reason"]
    values = trial()
    values["S2"] = {e: v[:2] for e, v in values["S2"].items()}
    unequal = gxe(registry, per_study(registry, rows_of(values)))
    assert "different numbers of reps" in unequal.data["variance_components"]["reason"]
    missing = gxe(registry, per_study(registry, rows_of(trial(), drop=(("S3", "E1", 2),))))
    assert "1 entry x rep cell(s) have no plot (study S3: 1)" in missing.data["variance_components"]["reason"]


def test_rep_codes_are_compared_within_each_study(registry):
    """Review 1: S1 writing reps 1/2/3 and S2 writing 01/02/03 is no clash: the reps belong to their own study."""
    rows = rows_of(trial())
    for row in rows:
        if row["studyDbId"] == "S2":
            row["rep"] = "0" + row["rep"]
    r = gxe(registry, per_study(registry, rows))
    assert r.ok and r.data["variance_components"]["available"] is True


def test_blocks_within_reps_are_named_as_not_modelled(registry):
    rows = rows_of(trial())
    for row in rows:
        row["block"] = f"{row['rep']}-{'a' if row['germplasmDbId'] in ('E0', 'E1') else 'b'}"
    r = gxe(registry, per_study(registry, rows))
    assert r.ok and r.data["variance_components"]["available"] is True
    variance = [k for k in claims(r) if k.kind == "variance"]
    assert all("incomplete blocks not modelled" in k.transformation for k in variance)
    assert any("incomplete blocks not modelled" in w for w in r.warnings)


def test_the_method_text_keeps_every_table_note(registry):
    """Review 1: a table averaging a repeat (mean_per_unit) beside a table with several dates lost its note."""
    rows = rows_of(trial(), reps=False)
    s1 = [dict(x) for x in rows if x["studyDbId"] == "S1"]
    s1.append(dict(s1[0], observationDbId="Orepeat"))                                    # the same plot and date twice
    s2 = [dict(x) for x in rows if x["studyDbId"] == "S2"]
    s2 += [dict(x, observationDbId=x["observationDbId"] + "-d2", observationTimeStamp="2026-03-01") for x in s2]
    for x in s2[:15]:
        x["observationTimeStamp"] = "2026-02-01"
    rest = [dict(x) for x in rows if x["studyDbId"] in ("S3", "S4")]
    r = gxe(registry, [save(registry, s2), save(registry, s1), save(registry, rest)], duplicate_policy="mean_per_unit")   # dates first
    assert r.ok, r.error
    text = next(k for k in claims(r) if k.kind == "anova").transformation
    assert "duplicate_policy=mean_per_unit" in text and "averaged over its dates" in text




# --- 7. the second review's findings (2026-10-23) -----------------------------------------------------------------

def test_a_positive_heritability_carries_no_set_to_zero_note(tmp_path):
    """Review 2: the H2 text itself said "set to 0", so every answer read "(set to 0: MS entry is below ...)"."""
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_gxe_h2")
    r = gxe(ctx.registry, per_study(ctx.registry, rows_of(trial())))
    assert 0 < r.data["h2"] < 1 and "(set to 0" not in answer(ctx, claims(r))
    negative = gxe(ctx.registry, per_study(ctx.registry, rows_of(trial(g_scale=0.0, ge_scale=4.0))))
    shown = answer(ctx, claims(negative))
    assert "(H²) 0 (set to 0: MS entry is below MS entry × environment)" in shown and "genotype variance 0 (set to 0)" in shown


def test_reasons_count_every_study(registry):
    """Review 2: "6 cells of study 4775" when four studies missed 12 cells."""
    drop = (("S1", "E1", 2), ("S3", "E0", 1), ("S3", "E4", 3))
    r = gxe(registry, per_study(registry, rows_of(trial(), drop=drop)))
    assert "3 entry x rep cell(s) have no plot (study S1: 1; study S3: 2)" in r.data["variance_components"]["reason"]
    one = gxe(registry, per_study(registry, rows_of(trial(r=1))))
    assert "4 of 4 studies have one rep (S1, S2, S3, S4)" in one.data["variance_components"]["reason"]


def test_a_p_below_what_a_double_holds_is_shown_as_a_bound_the_claim_carries(tmp_path, monkeypatch):
    monkeypatch.setattr(at.pvalues, "f_upper_p", lambda f, d1, d2: 0.0)
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_gxe_p0")
    r = gxe(ctx.registry, [save(ctx.registry, rows_of(trial(), reps=False))])
    p = next(k for k in claims(r) if k.transformation.startswith("means-table ANOVA: p of entry"))
    assert p.value == 0.0 and "p below 1e-300" in p.transformation and "| &lt; 1e-300 |" in answer(ctx, claims(r))


def test_the_heritability_text_says_when_the_long_form_holds():
    assert "no variance component is set to 0" in at.GXE_H2_TEXT


# --- T4's residue rule (coach 2026-10-23 03:48 UTC, NEXT 2 part c) ------------------------------------------------------------

CANCELLING = {"S1": {"E0": [0.1, 0.2, -0.3], "E1": [0.0, 0.0, 0.0], "E2": [0.0, 0.0, 0.0]},
              "S2": {"E0": [0.3, -0.1, -0.2], "E1": [0.0, 0.0, 0.0], "E2": [0.0, 0.0, 0.0]}}


def test_a_means_table_of_float_residue_gives_no_shares_f_or_h2(registry):
    """Entry means that are float residue of cancelling plots gave shares of 33 / 0 / 67 % and F 0.0 from noise; the table now
    counts as constant at the plots' precision."""
    r = gxe(registry, per_study(registry, rows_of(CANCELLING, reps=False)))
    assert r.ok, r.error
    assert all(share is None for share in r.data["shares"].values()) and table(r)["entry"]["f"] is None and r.data["h2"] is None
    assert table(r)["total"]["ss"] == 0                     # the total row too: residue is 0 (deliberate bug W3 survived)


def test_the_gxe_residue_scale_comes_from_the_raw_values_under_mean_per_unit(registry):
    """T4 review 2: an entry's three values as repeated measurements of ONE plot, averaged first, must keep their scale."""
    rows = rows_of(CANCELLING, reps=False)
    for x in rows:
        x["observationUnitDbId"] = x["observationUnitDbId"].rsplit("-", 1)[0]       # an entry's values in a study: one plot
    r = gxe(registry, per_study(registry, rows), duplicate_policy="mean_per_unit")
    assert r.ok, r.error
    assert all(share is None for share in r.data["shares"].values()) and r.data["h2"] is None


def test_an_entry_effect_of_float_residue_gives_no_h2(registry):
    """The one review of NEXT 2, F1: entries equal on paper in each environment (0.15 from different plots) left an entry SS of
    3e-31, and H2 = 1 - 0 / MS entry came out 1.0 from residue."""
    values = {"S1": {"E0": [0.1, 0.2], "E1": [0.15], "E2": [0.3, 0.0]}, "S2": {"E0": [5.0], "E1": [5.0], "E2": [5.0]}}
    r = gxe(registry, per_study(registry, rows_of(values, reps=False)))
    assert r.ok, r.error
    t = table(r)
    assert t["entry"]["ss"] == 0 and t["entry x environment"]["ss"] == 0 and r.data["h2"] is None and t["entry"]["f"] is None
    assert r.data["shares"]["environment"] == pytest.approx(100.0) and t["total"]["ss"] == pytest.approx(t["environment"]["ss"], rel=1e-12)


def test_a_g_x_e_of_float_residue_gives_no_f(registry):
    """F2: the same three entries in both studies (1e4 and 1e-7 apart) left residue SS for environment and G x E, and F 1.8e10."""
    values = {s: {"E0": [10000.0], "E1": [10000.0000001], "E2": [10000.0000002]} for s in ("S1", "S2")}
    r = gxe(registry, per_study(registry, rows_of(values, reps=False)))
    assert r.ok, r.error
    t = table(r)
    assert t["environment"]["ss"] == 0 and t["entry x environment"]["ss"] == 0 and t["entry"]["f"] is None and t["entry"]["p"] is None
    assert t["total"]["ss"] == t["entry"]["ss"]          # the total is the sum of its parts, residue left out (deliberate bug W3 survived)


def test_a_real_small_g_x_e_beside_a_large_environment_effect_is_kept(registry):
    """F3: G x E deviations of 1e-7 are real at values of 1e3 (residue there is near 1e-13), but the old rule judged them against the
    environment-dominated total and set them to 0 (F None, H2 1.0; numpy F 163.86, H2 0.9939)."""
    values = {"S1": {"E0": [1000.0], "E1": [1000.000001], "E2": [1000.000002]},
              "S2": {"E0": [2000.0], "E1": [2000.0000012], "E2": [2000.0000019]}}
    r = gxe(registry, per_study(registry, rows_of(values, reps=False)))
    assert r.ok, r.error
    x, t = expected(values, with_error=False), table(r)
    f = (x["ss_entry"] / 2) / (x["ss_ge"] / 2)
    assert t["entry"]["f"] == pytest.approx(f, rel=1e-4) and r.data["h2"] == pytest.approx(1 - 1 / f, rel=1e-4)
