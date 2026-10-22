"""T1 of the statistics tools (research report of 2026-10-22, Claude outputs\\research_stats_tools_20261022): p-values and the t
quantile in plain Python, no scipy. Every later tool that states a p-value or an LSD uses this module. The fixtures are the
report's published numbers: scipy's spearmanr example, metan's data_ge correlations and t tests, agricolae's sweetpotato
one-way ANOVA, metan's joint ANOVA (a p-value of 2.19e-19, compared on a log scale), and t(0.975, 8) for an LSD."""
import math

import pytest

import pvalues as pv


def t_of_r(r: float, n: int) -> float:
    return r * math.sqrt((n - 2) / (1 - r * r))


# --- 1. published numbers ------------------------------------------------------------------------------------------------

def test_the_t_quantile_for_an_lsd_matches_scipy():
    assert pv.t_quantile(0.975, 8) == pytest.approx(2.3060041352041667, abs=1e-12)          # agricolae's LSD uses t(0.975, 8)


@pytest.mark.parametrize("prob, df, expected", [(0.975, 1, 12.706204736174694), (0.975, 30, 2.0422724563012378),
                                                (0.95, 10, 1.8124611228116754), (0.995, 5, 4.032142983555228)])
def test_t_quantiles_match_scipy(prob, df, expected):
    """scipy 1.17.1 t.ppf (review: two of the first expected values were wrong in the 12th digit, hidden by a 1e-9 tolerance)."""
    assert pv.t_quantile(prob, df) == pytest.approx(expected, abs=1e-13)


def test_the_spearman_p_value_of_scipys_example():
    """scipy.stats.spearmanr([1,2,3,4,5],[5,6,7,8,7]) -> 0.8207826816681233, p 0.08858700531354381 (t with n - 2 df)."""
    assert pv.t_two_sided_p(t_of_r(0.8207826816681233, 5), 3) == pytest.approx(0.08858700531354381, rel=1e-9)


@pytest.mark.parametrize("r, expected", [(-0.1046076, 0.7736586), (-0.2121212, 0.5563058)], ids=["pearson", "spearman"])
def test_data_ge_correlations_of_entry_means(r, expected):
    """metan data_ge genotype means GY vs HM (10 genotypes): r and rho printed to 7 decimals, so the p agrees to about 1e-6."""
    assert pv.t_two_sided_p(t_of_r(r, 10), 8) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("t, df, expected", [(2.9521990, 9, 0.01616307), (2.2893950, 17.121246, 0.03502448)], ids=["paired", "welch"])
def test_data_ge_t_tests(t, df, expected):
    """E3 vs E4: a paired t test (9 df) and a Welch test with non-integer df."""
    assert pv.t_two_sided_p(t, df) == pytest.approx(expected, abs=1e-7)


def test_agricolae_sweetpotato_one_way_anova():
    assert pv.f_upper_p(17.344783, 3, 8) == pytest.approx(7.334e-4, abs=5e-8)                     # as printed (4 digits)
    assert pv.f_upper_p(17.344783, 3, 8) == pytest.approx(7.334172082275658e-4, rel=1e-9)          # unrounded (scipy 1.17.1)


def test_metan_joint_anova_tiny_p_value_on_a_log_scale():
    """GEN F = 14.93 on (9, 252) df, printed p 2.19e-19: compared as log10 within 0.02 (the printed F is rounded)."""
    assert abs(math.log10(pv.f_upper_p(14.93, 9, 252)) - math.log10(2.19e-19)) < 0.01


# --- 2. identities and edges -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("a, b", [(0.5, 0.5), (1, 1), (2.5, 7), (50, 0.5), (500, 4.5), (0.5, 126)])
@pytest.mark.parametrize("x", [2.0 ** -30, 2.0 ** -7, 0.3125, 0.5, 0.75, 1 - 2.0 ** -10])      # dyadic: 1 - x is exact in binary
def test_the_incomplete_beta_is_symmetric(a, b, x):
    assert pv.regularized_beta(a, b, x) + pv.regularized_beta(b, a, 1 - x) == pytest.approx(1.0, abs=1e-13)


def test_known_closed_forms_of_the_incomplete_beta():
    for x in (0.1, 0.5, 0.9):
        assert pv.regularized_beta(1, 1, x) == pytest.approx(x, abs=1e-15)                    # uniform
        assert pv.regularized_beta(1, 3, x) == pytest.approx(1 - (1 - x) ** 3, abs=1e-14)
    assert pv.regularized_beta(2, 3, 0.0) == 0.0 and pv.regularized_beta(2, 3, 1.0) == 1.0


def test_p_values_at_the_edges():
    assert pv.t_two_sided_p(0.0, 5) == 1.0 and pv.f_upper_p(0.0, 2, 10) == 1.0
    assert pv.t_two_sided_p(-2.5, 7) == pv.t_two_sided_p(2.5, 7)                              # two-sided
    assert 0.0 <= pv.t_two_sided_p(1e6, 3) < 1e-15 and 0.0 <= pv.f_upper_p(1e9, 3, 30) < 1e-15
    ps = [pv.t_two_sided_p(t, 12) for t in (0.5, 1, 2, 3, 5)]
    assert ps == sorted(ps, reverse=True)                                                     # a larger t is a smaller p


def test_quantile_and_p_value_agree():
    for df in (1, 3, 8, 30, 252):
        for prob in (0.9, 0.975, 0.995):
            q = pv.t_quantile(prob, df)
            assert pv.t_two_sided_p(q, df) == pytest.approx(2 * (1 - prob), rel=1e-12)
    assert pv.t_quantile(0.025, 8) == pytest.approx(-2.3060041352041667, abs=1e-13)          # the lower tail by symmetry


@pytest.mark.parametrize("call", [lambda: pv.t_two_sided_p(2.0, 0), lambda: pv.t_two_sided_p(2.0, -3), lambda: pv.t_two_sided_p(math.nan, 5),
                                  lambda: pv.t_two_sided_p(2.0, math.inf), lambda: pv.f_upper_p(2.0, 0, 5), lambda: pv.f_upper_p(math.nan, 2, 5),
                                  lambda: pv.f_upper_p(2.0, 2, math.nan), lambda: pv.t_quantile(1.0, 5), lambda: pv.t_quantile(0.0, 5),
                                  lambda: pv.t_quantile(0.975, 0), lambda: pv.regularized_beta(0, 1, 0.5), lambda: pv.regularized_beta(1, 1, 1.5),
                                  lambda: pv.regularized_beta(1, 1, math.nan)],
                         ids=["t-df0", "t-df-neg", "t-nan", "t-df-inf", "f-df0", "f-nan", "f-df-nan", "q-1", "q-0", "q-df0", "beta-a0", "beta-x", "beta-nan"])
def test_impossible_inputs_are_refused_not_answered(call):
    with pytest.raises(ValueError):
        call()


def test_a_fraction_that_does_not_converge_is_refused_not_returned(monkeypatch):
    monkeypatch.setattr(pv, "_MAX_ITERATIONS", 2)
    with pytest.raises(ArithmeticError):
        pv.regularized_beta(50, 60, 0.45)


def test_a_tiny_t_with_huge_degrees_of_freedom_keeps_its_precision():
    """scipy cross-check (scipy_cross_check.txt): df / (df + t^2) rounds to exactly 1.0 for t = 1e-6, df = 1e5, which gave p = 1.0
    instead of 0.99999920; the tail is now computed from t^2 / (df + t^2), which keeps every digit."""
    assert pv.t_two_sided_p(1e-6, 100000) == pytest.approx(0.9999992021174339, abs=1e-12)
    assert pv.f_upper_p(1e-12, 2, 1e6) == pytest.approx(0.999999999999, abs=1e-14)



# --- 3. after the review of 2026-10-22 (REVIEW.md): exact closed forms, the complement branch, limits and inputs -------------

@pytest.mark.parametrize("t", [1e-9, 0.3, 1.0, 2.5, 40.0])
def test_closed_forms_for_one_and_two_degrees_of_freedom(t):
    assert pv.t_two_sided_p(t, 1) == pytest.approx(1 - 2 / math.pi * math.atan(t), abs=1e-15)       # Cauchy; a few units in the last place
    assert pv.t_two_sided_p(t, 2) == pytest.approx(1 - t / math.sqrt(2 + t * t), abs=1e-15)


@pytest.mark.parametrize("f, df2", [(0.01, 7), (0.7, 3), (3.0, 40), (25.0, 9)])
def test_closed_form_of_the_f_tail_with_two_numerator_df(f, df2):
    assert pv.f_upper_p(f, 2, df2) == pytest.approx((1 + 2 * f / df2) ** (-df2 / 2), rel=1e-13)


@pytest.mark.parametrize("f, df1, df2, expected", [(0.5, 3, 8, 0.6926248669438465), (0.2, 4, 20, 0.9353965414511061),
                                                   (0.05, 10, 120, 0.9999923505285832)])
def test_the_f_tail_in_its_complement_branch_matches_scipy(f, df1, df2, expected):
    """Review: every F fixture sat in the direct branch, so the y = df1 f / (df2 + df1 f) side was untested."""
    assert pv.f_upper_p(f, df1, df2) == pytest.approx(expected, abs=1e-13)


def test_quantiles_far_in_either_tail_keep_their_digits():
    """Review: a quantile above 1e7 came back as exactly 1e7, and a lower tail of 1e-8 lost digits through 1 - (1 - p)."""
    assert pv.t_quantile(1 - 2.0 ** -40, 1) == pytest.approx(2.0 ** 40 / math.pi, rel=1e-12)       # Cauchy: cot(pi * tail)
    assert pv.t_quantile(2.0 ** -40, 1) == pytest.approx(-(2.0 ** 40) / math.pi, rel=1e-12)
    assert pv.t_quantile(1e-10, 8) == pytest.approx(-39.12956297338265, rel=1e-10)                 # scipy t.ppf(1e-10, 8)


def test_degrees_of_freedom_above_a_million_are_refused_not_guessed():
    """Review: lgamma(a + b) - lgamma(a) - lgamma(b) cancels for huge df (t = 1 on 1e12 df gave 0.31744, not 0.3173105).
    No breeding trial has a million error df; above that the module refuses."""
    assert pv.t_two_sided_p(1.0, 1e6) == pytest.approx(0.3173107, abs=1e-7)
    for call in (lambda: pv.t_two_sided_p(1.0, 1e6 + 1), lambda: pv.f_upper_p(1.0, 3, 2e6), lambda: pv.f_upper_p(1.0, 2e6, 3),
                 lambda: pv.t_quantile(0.975, 1e7), lambda: pv.regularized_beta(6e5, 1.0, 0.5)):
        with pytest.raises(ValueError):
            call()


def test_numbers_of_any_numeric_type_are_read_and_booleans_or_overflows_refused():
    from fractions import Fraction
    import numpy as np
    assert pv.t_two_sided_p(Fraction(5, 2), np.int64(8)) == pv.t_two_sided_p(2.5, 8.0)
    assert pv.f_upper_p(np.float32(2.0), 3, 8) == pytest.approx(pv.f_upper_p(2.0, 3, 8), rel=1e-15)
    for call in (lambda: pv.t_two_sided_p(True, 5), lambda: pv.t_two_sided_p(2.0, True), lambda: pv.t_two_sided_p(10 ** 400, 5),
                 lambda: pv.t_quantile(0.975, 10 ** 400), lambda: pv.f_upper_p("2", 3, 8)):
        with pytest.raises(ValueError):
            call()
