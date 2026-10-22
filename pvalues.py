"""
pvalues.py — p-values and the t quantile in plain Python (no scipy), for the statistics tools.

The regularized incomplete beta I_x(a, b) from math.lgamma and the Numerical Recipes continued fraction (Lentz's method, with
the symmetry switch at x > (a + 1) / (a + b + 2)); from it the two identities every tool needs:
  two-sided t p-value      p = I_{v / (v + t^2)}(v / 2, 1 / 2)
  F upper tail             P(F' > F) = I_{v2 / (v2 + v1 F)}(v2 / 2, v1 / 2)
and the t quantile (for an LSD) by bisection on the t tail. Checked against scipy (the research report of 2026-10-22, Claude
outputs/research_stats_tools_20261022, NOTES_plain_python_implementation.md, Q0; and the builder's and a reviewer's own grids,
Claude outputs/t1_pvalues_20261022): about 1e-12 or better in p-values up to 1e5 degrees of freedom, and to the last few digits of
the exact closed forms for 1 and 2 degrees of freedom. Non-integer degrees of freedom (Welch) pass straight through.

A number that cannot be computed is refused (ValueError), never guessed: degrees of freedom must be finite, above 0 and at most
1,000,000 (above that the log-gamma differences cancel and the p-value would be wrong in its 8th digit or worse); a statistic
must be a finite number (not a bool); a probability strictly between 0 and 1. A continued fraction that does not converge raises
ArithmeticError.
"""
from __future__ import annotations

import math
import numbers

_EPS = 3e-16            # the convergence tolerance of the continued fraction (the report's)
_FPMIN = 1e-300         # Lentz's guard against division by zero
_MAX_ITERATIONS = 10_000
MAX_DF = 1_000_000      # degrees of freedom above this are refused (review of 2026-10-22: lgamma differences cancel there)


def _check_finite(value: float, name: str) -> float:
    """A finite real number of any numeric type (int, float, Fraction, numpy), as a float; a bool, a string, NaN, an infinity
    or an int too large for a float is refused."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be a finite number, got an int of {len(str(abs(int(value))))} digits") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def _check_positive(value: float, name: str, limit: float = math.inf) -> float:
    number = _check_finite(value, name)
    if number <= 0 or number > limit:
        raise ValueError(f"{name} must be above 0" + (f" and at most {limit:,.0f}" if math.isfinite(limit) else "") + f", got {value!r}")
    return number


def _continued_fraction(a: float, b: float, x: float) -> float:
    """The continued fraction of I_x(a, b) (Numerical Recipes betacf), evaluated by Lentz's method."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (_FPMIN if abs(d) < _FPMIN else d)
    h = d
    for m in range(1, _MAX_ITERATIONS + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))                     # even step
        d = 1.0 + aa * d
        d = 1.0 / (_FPMIN if abs(d) < _FPMIN else d)
        c = 1.0 + aa / c
        c = _FPMIN if abs(c) < _FPMIN else c
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))           # odd step
        d = 1.0 + aa * d
        d = 1.0 / (_FPMIN if abs(d) < _FPMIN else d)
        c = 1.0 + aa / c
        c = _FPMIN if abs(c) < _FPMIN else c
        step = d * c
        h *= step
        if abs(step - 1.0) < _EPS:
            return h
    raise ArithmeticError(f"the incomplete beta did not converge for a={a}, b={b}, x={x}")


def _ibeta(a: float, b: float, x: float, y: float) -> float:
    """I_x(a, b) given x AND y = 1 - x, each computed by the caller without cancellation: near x = 1 the tail comes from y, so a
    t of 1e-6 on 100,000 df, where df / (df + t^2) rounds to exactly 1.0, keeps its p of 0.99999920 (scipy cross-check)."""
    if x <= 0.0:
        return 0.0
    if y <= 0.0:
        return 1.0
    log_x = math.log1p(-y) if y < 0.5 else math.log(x)
    log_y = math.log1p(-x) if x < 0.5 else math.log(y)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * log_x + b * log_y)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _continued_fraction(a, b, x) / a
    return 1.0 - front * _continued_fraction(b, a, y) / b


def regularized_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b), the regularized incomplete beta function, for a, b > 0 and 0 <= x <= 1."""
    a, b = _check_positive(a, "a", MAX_DF / 2), _check_positive(b, "b", MAX_DF / 2)
    x = _check_finite(x, "x")
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"x must be between 0 and 1, got {x!r}")
    return _ibeta(a, b, x, 1.0 - x)


def t_two_sided_p(t: float, df: float) -> float:
    """P(|T| >= |t|) for Student's t with df degrees of freedom (df may be non-integer, as in Welch's test)."""
    t, df = _check_finite(t, "t"), _check_positive(df, "df", MAX_DF)
    return _ibeta(df / 2.0, 0.5, df / (df + t * t), t * t / (df + t * t))


def f_upper_p(f: float, df1: float, df2: float) -> float:
    """P(F' >= f) for the F distribution with (df1, df2) degrees of freedom; 1.0 for f <= 0."""
    f, df1, df2 = _check_finite(f, "F"), _check_positive(df1, "df1", MAX_DF), _check_positive(df2, "df2", MAX_DF)
    if f <= 0.0:
        return 1.0
    return _ibeta(df2 / 2.0, df1 / 2.0, df2 / (df2 + df1 * f), df1 * f / (df2 + df1 * f))


def t_quantile(prob: float, df: float) -> float:
    """The t value with P(T <= t) = prob, for 0 < prob < 1 (t_quantile(0.975, 8) = 2.30600413520417 to 15 digits, the LSD's
    t). The tail beyond it (1 - prob above the median, prob itself below it, so a small lower-tail probability keeps its digits)
    is bracketed by doubling and then halved 200 times; a quantile beyond 1e300 is refused."""
    prob = _check_finite(prob, "prob")
    df = _check_positive(df, "df", MAX_DF)
    if not 0.0 < prob < 1.0:
        raise ValueError(f"prob must be strictly between 0 and 1, got {prob!r}")
    if prob == 0.5:
        return 0.0
    tail = 1.0 - prob if prob > 0.5 else prob                              # P(T > |t|), half of the two-sided p
    lo, hi = 0.0, 1.0
    while 0.5 * t_two_sided_p(hi, df) > tail:                              # review: a fixed [0, 1e7] silently capped the answer
        lo, hi = hi, hi * 2.0
        if hi > 1e300:
            raise ValueError(f"the t quantile for prob={prob!r} on {df!r} df is beyond 1e300")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * t_two_sided_p(mid, df) > tail:
            lo = mid
        else:
            hi = mid
    quantile = 0.5 * (lo + hi)
    return quantile if prob > 0.5 else -quantile
