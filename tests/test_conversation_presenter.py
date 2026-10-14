"""Focused conversation presentation checks using synthetic equivalents of the saved 4501 case."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import chat_presenter as cp
from contracts import AnalysisReport, Claim, Exclusion, Plan, PlanStep, ResolvedScope, Statistic


def make_view(question="In synthetic study T4501, what is the pooled mean of wilt reaction?", *, statistic="mean", caveats=None):
    values = {"mean": 2.3785714285714286, "median": 3.0, "min": 1.0, "max": 4.0, "sd": 0.7998907941907057, "count": 140}
    claims = [Claim(claim_id=f"clm_polish_{kind}", kind=kind, value=value, unit=None,
                    denominator={"name": "valid numeric values, one per observation unit", "n": 140}, n_independent_units=140,
                    source_artifact_hashes=["a" * 64], transformation=f"{kind} of valid values; valid values used as recorded (one row per observation unit)")
              for kind, value in values.items()]
    default_notes = [
        "numeric_summary: unit not stated in the artifact's measurement metadata; claims carry unit=null",
        "analysis target: descriptive means of valid values as recorded; not adjusted for trial design, environment, missingness or clone composition; not evidence of genetic superiority",
        "model: Unit is not stated in the artifact's metadata, so the mean value is reported with unit=null.",
        "model: The mean is a simple descriptive average of all valid recorded values in the study; it does not account for trial design, environment, or genetic factors.",
        "key claims (model's selection): ['clm_polish_mean']",
    ]
    plan = Plan(plan_id="plan_polish", question=question, interpretation="Summarize the recorded observations", status="ready",
                scope=ResolvedScope(base_url="https://synthetic.invalid/brapi/v2", study_ids=["T4501"], variable_ids=["V1"]),
                statistic=Statistic(kind=statistic, denominator="valid plot-level values") if statistic else None,
                steps=[PlanStep(step_id="step_fetch", agent="retriever", action="get_observations")],
                created_at_utc=datetime.now(timezone.utc))
    return SimpleNamespace(plan=plan, analysis=AnalysisReport(status="completed", claims=claims,
        caveats=default_notes if caveats is None else caveats), retrieval=SimpleNamespace(status="completed", warnings=[]),
        execution_status="completed", ctx=None, notes=[])


def test_saved_4501_shape_leads_with_exact_mean_and_n_without_extra_metric_table():
    shown = cp.answer_summary(make_view())
    assert shown.startswith("**Mean: 2.37857142857")
    assert "n = 140" in shown and "Recorded analysis units: 140" in shown
    assert "| Median |" not in shown and "| Minimum |" not in shown and "| Maximum |" not in shown
    assert "| Sample standard deviation |" not in shown and "| Count |" not in shown
    assert "Other calculated statistics remain in Full report" in shown
    assert "unit not stated" in shown and shown.count("unit not stated") == 1
    assert "numeric_summary" not in shown and "numeric\\_summary" not in shown
    assert "unit=null" not in shown and "model:" not in shown and "analysis target:" not in shown
    assert shown.count("trial design") == 1
    assert "missingness or clone composition" in shown and "not evidence of genetic superiority" in shown
    assert shown.count("Distinct IDs alone do not establish statistical independence") == 1


@pytest.mark.parametrize("question", [
    "What are the mean and median in synthetic study T4501?",
    "Show mean, standard deviation, and n for T4501.",
    "What are the mean and count?",
    "Give a summary including the mean.",
    "Summarize the observations in T4501.",
])
def test_explicit_multiple_metrics_or_general_summary_keeps_supplied_statistics(question):
    shown = cp.answer_summary(make_view(question))
    assert "| Mean |" in shown and "| Median |" in shown and "| Sample standard deviation |" in shown
    assert "2.37857142857" in shown
    assert "Other calculated statistics remain" not in shown


def test_mean_and_n_focuses_without_hiding_denominator_or_unknown_independence():
    view = make_view("Give the mean and n for T4501.")
    view.analysis.claims[0] = view.analysis.claims[0].model_copy(update={"n_independent_units": None})
    shown = cp.answer_summary(view)
    assert shown.startswith("**Mean:") and "n = 140" in shown
    assert "Recorded analysis units: unknown" in shown
    assert "statistical independence" in shown


@pytest.mark.parametrize("kind", [None, "median"])
def test_mean_focus_requires_the_typed_plan_kind(kind):
    shown = cp.answer_summary(make_view(statistic=kind))
    assert "| Median |" in shown and "| Sample standard deviation |" in shown


def test_exclusions_and_incomplete_warning_survive_focusing_and_duplicate_prefixes():
    view = make_view(caveats=["model: Two observations are missing.", "Two observations are missing.",
                              "Incomplete coverage of the planned environments."])
    view.analysis.claims[0] = view.analysis.claims[0].model_copy(update={"exclusions": [
        Exclusion(reason="missing values", count=2)]})
    shown = cp.answer_summary(view)
    assert shown.count("Two observations are missing") == 1
    assert "Incomplete coverage" in shown and "Excluded: 2" in shown
    assert "model:" not in shown and "n = 140" in shown


def test_compound_unit_warning_does_not_lose_missingness_when_technical_prefix_is_removed():
    shown = cp.answer_summary(make_view(caveats=["numeric_summary: unit=null; 4 observations are missing and coverage is incomplete."]))
    assert "unit=null" not in shown and "numeric\\_summary" not in shown
    assert "4 observations are missing and coverage is incomplete" in shown


def test_unknown_extra_warning_is_not_treated_as_a_duplicate_standard_disclaimer():
    view = make_view()
    view.analysis.caveats.append("model: The mean is a simple descriptive average of all valid recorded values in the study; it does not account for trial design, environment, or genetic factors. Extra warning: treatment allocation is unknown.")
    shown = cp.answer_summary(view)
    assert "Extra warning: treatment allocation is unknown" in shown


def test_same_clause_unit_warning_preserves_unknown_warning_text():
    shown = cp.answer_summary(make_view(caveats=["numeric_summary: unit=null and treatment allocation is unknown."]))
    assert "treatment allocation is unknown" in shown and "unit=null" not in shown


def test_unavailable_hidden_statistic_keeps_its_substantive_reason():
    view = make_view()
    view.analysis.claims[4] = view.analysis.claims[4].model_copy(update={"value": None, "missing_reason": "fewer than two valid independent units"})
    shown = cp.answer_summary(view)
    assert "fewer than two valid independent units" in shown
    assert "| Sample standard deviation |" not in shown


def test_no_complete_table_explains_the_known_reason_and_offers_retry_without_inventing_a_timeout():
    view = make_view()
    view.execution_status = "blocked"
    view.analysis = AnalysisReport(status="blocked", caveats=["no complete artifact is eligible for analysis"])
    shown = cp.answer_summary(view)
    assert "No complete data table was available for analysis" in shown
    assert "Retry the question" in shown
    assert "review the proposed scope and approve" in shown
    assert "artifact" not in shown and "timeout" not in shown and "server failed" not in shown
    assert "2.378571" not in shown and "Full report" in shown


def test_failure_retains_recorded_fetch_reason_alongside_plain_language_no_table_message():
    view = make_view()
    view.execution_status = "incomplete"
    view.analysis = AnalysisReport(status="blocked", caveats=["no complete artifact is eligible for analysis"])
    view.retrieval = SimpleNamespace(status="incomplete", warnings=["The request timed out after 90 seconds."])
    shown = cp.answer_summary(view)
    assert "request timed out after 90 seconds" in shown
    assert "No complete data table" in shown and "Retry the question" in shown


def test_multiple_means_keep_comparison_tables():
    view = make_view("Give the pooled mean and clone means with n.")
    view.analysis.claims.append(view.analysis.claims[0].model_copy(update={
        "claim_id": "clm_polish_clone_mean", "value": 3.0,
        "filters": {"germplasmDbId": "G1"}, "grouping": ["germplasmDbId"]}))
    shown = cp.answer_summary(view)
    assert "| Mean |" in shown and "G1" in shown
    assert "2.37857142857" in shown and "3" in shown
    assert not shown.startswith("**Mean:")


def test_warning_deduplication_preserves_numeric_signs_and_comparison_operators():
    warnings = ["Calibration offset: -0.1 units.", "Calibration offset: +0.1 units.",
                "Required sample size: <2.", "Required sample size: >2."]
    shown = cp.answer_summary(make_view(caveats=warnings))
    assert "-0.1 units" in shown and "+0.1 units" in shown
    assert "&lt;2" in shown and "&gt;2" in shown
