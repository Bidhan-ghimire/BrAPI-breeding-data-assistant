"""Synthetic presenter checks: no model calls, live requests, or authority changes."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import analyst_tools as at
import brapi_mcp_server as srv
import chat_presenter as cp
from contracts import AnalysisReport, Claim, FetchManifest, Plan, PlanStep, ResolvedScope
from part2_support import synthetic_context


@pytest.fixture
def ctx(tmp_path):
    return synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_presenter")


def counted(ctx, artifact, **options):
    result = at.tool_count_records(ctx.registry, at.CountRecordsArgs(artifact_id=artifact, **options))
    return [Claim.model_validate(claim) for claim in result.data["claims"]]


def controller(ctx, claims, *, caveats=(), status="completed", warnings=()):
    return SimpleNamespace(execution_status=status, analysis=AnalysisReport(status="completed", claims=claims, caveats=list(caveats)),
                           retrieval=SimpleNamespace(status="completed", warnings=list(warnings)), ctx=ctx, notes=[],
                           prose="INVENTED MODEL CLAIM: yield doubled", prose_flags=["unsupported 999"],
                           trace=["private debugging trace"])


def save_studies(ctx, rows):
    return ctx.registry.save_table(rows, kind="studies", complete=True,
        columns={"studyDbId": "ID", "studyName": "Name", "locationName": "Location", "seasons": "Season"}).artifact_id


def test_catalog_duplicate_count_and_frequency_make_one_named_table(ctx):
    aid = srv.dispatch(ctx, "export_metadata", {"entity": "studies"}).artifact_ids[0]
    claims = counted(ctx, aid, group_by="studyDbId", distinct_by="studyDbId")
    shown = cp.answer_summary(controller(ctx, claims))
    assert shown.count("| ID |") == 1
    assert shown.count("SYNTHETIC yield trial S1") == 1
    assert "SYNTHETIC location one" in shown and "2026" in shown
    assert "1 distinct IDs" in shown
    assert "INVENTED" not in shown and "unsupported 999" not in shown and "clm_" not in shown


def test_different_source_hashes_are_never_merged(ctx):
    aid1 = save_studies(ctx, [{"studyDbId": "1", "studyName": "First source"}])
    aid2 = save_studies(ctx, [{"studyDbId": "1", "studyName": "Second source"}])
    claims = counted(ctx, aid1, group_by="studyDbId", distinct_by="studyDbId") + counted(ctx, aid2, group_by="studyDbId", distinct_by="studyDbId")
    shown = cp.answer_summary(controller(ctx, claims))
    assert shown.count("| ID |") == 2
    assert "First source" in shown and "Second source" in shown
    assert "source 1" in shown and "source 2" in shown


def test_different_count_statistics_are_not_suppressed(ctx):
    aid = save_studies(ctx, [{"studyDbId": "1", "studyName": "Trial"}, {"studyDbId": "1", "studyName": "Trial"}])
    claims = counted(ctx, aid, group_by="studyDbId", distinct_by="studyDbId") + counted(ctx, aid)
    shown = cp.answer_summary(controller(ctx, claims))
    assert "1 distinct IDs" in shown and "Records" in shown
    assert "row count of a complete table" in shown
    assert "| Count | 2 |" in shown


def test_location_frequency_uses_descending_study_count_and_hides_catalog_internal_notes(ctx):
    rows = [{"locationDbId": location, "locationName": name} for location, name, count in
            [("1", "Fewest", 1), ("2", "Most", 4), ("3", "Middle", 2)] for _ in range(count)]
    aid = ctx.registry.save_table(rows, kind="studies", complete=True,
        columns={"locationDbId": "Location ID", "locationName": "Name"}).artifact_id
    claims = counted(ctx, aid, group_by="locationDbId")
    shown = cp.answer_summary(controller(ctx, claims, caveats=["analysis target: descriptive means",
                        "model: Study names were found in the artifact preview", "Missing locations were excluded."]))
    assert shown.index("| 2 |") < shown.index("| 3 |") < shown.index("| 1 |")
    assert "| Studies |" in shown
    assert "analysis target" not in shown and "artifact preview" not in shown
    assert "Missing locations were excluded" in shown


@pytest.mark.parametrize("count", [30, 32])
def test_catalog_limit_is_honest_and_numeric_order_is_preserved(ctx, count):
    aid = save_studies(ctx, [{"studyDbId": str(i), "studyName": f"Trial {i}"} for i in reversed(range(1, count + 1))])
    shown = cp.answer_summary(controller(ctx, counted(ctx, aid, distinct_by="studyDbId")))
    assert shown.index("| 2 |") < shown.index("| 10 |")
    assert "| 30 |" in shown
    if count > 30:
        assert "| 31 |" not in shown
        assert "Showing 30 of 32 entries" in shown and "2 additional IDs are not displayed here" in shown
        assert "Full report" in shown
    else:
        assert "not displayed" not in shown


def numeric_claim(kind="mean", **over):
    values = dict(claim_id=f"clm_presenter_{kind}", kind=kind, value=2.10625, unit="t/ha",
                  denominator={"name": "valid plot-level values", "n": 4}, n_independent_units=3,
                  source_artifact_hashes=["a" * 64], transformation=f"{kind} of valid values; one value per plot",
                  exclusions=[{"reason": "missing values", "count": 2}])
    values.update(over)
    return Claim(**values)


def test_numeric_precision_units_denominator_independence_and_caveats_survive(ctx):
    claim = numeric_claim()
    shown = cp.answer_summary(controller(ctx, [claim], caveats=["This is a partial description, not a study-level result.",
                            "Independence was not established across environments.", "key claims (model): clm_private_123"]))
    assert "2.10625 t/ha" in shown
    assert "Distinct IDs alone do not establish statistical independence" in shown
    assert "valid plot-level values: 4" in shown and "Recorded analysis units: 3" in shown
    assert "Excluded: 2" in shown and "missing values" in shown
    assert "partial description, not a study-level result" in shown
    assert "Independence was not established" in shown and "key claims" not in shown
    assert "clm_" not in shown and "art_" not in shown


def test_missing_units_unknown_independence_and_missing_value_are_explicit(ctx):
    claim = numeric_claim(unit=None, n_independent_units=None)
    missing = numeric_claim("sd", value=None, missing_reason="fewer than 2 independent units", unit=None, n_independent_units=None)
    shown = cp.answer_summary(controller(ctx, [claim, missing]))
    assert "unit not stated" in shown and "Recorded analysis units: unknown" in shown
    assert "Not available: fewer than 2 independent units" in shown
    assert "| Mean |" in shown and "| Sample standard deviation |" in shown


@pytest.mark.parametrize("status", ["incomplete", "failed", "blocked", "canceled", "limit_reached", "unknown"])
def test_noncomplete_status_never_shows_success_and_preserves_reason(ctx, status):
    view = controller(ctx, [numeric_claim()], status=status)
    view.retrieval = SimpleNamespace(status="incomplete", warnings=["Study S2 is missing; observations were not retrieved."])
    shown = cp.answer_summary(view)
    assert "Study S2 is missing" in shown and "Full report" in shown
    assert "2.10625" not in shown and "| Statistic |" not in shown


def test_completed_flag_cannot_hide_incomplete_analysis_or_local_failure(ctx):
    view = controller(ctx, [numeric_claim()])
    view.analysis = SimpleNamespace(status="failed", caveats=[], exclusions=[])
    view.notes = [r"unexpected error: C:\private workspace\out\data.csv could not be read because its hash changed"]
    shown = cp.answer_summary(view)
    assert "No verified answer" in shown and "hash changed" in shown
    assert "C:" not in shown and "private workspace" not in shown
    assert "2.10625" not in shown


def test_markup_in_database_names_is_displayed_as_data(ctx):
    aid = save_studies(ctx, [{"studyDbId": "1", "studyName": "<script>alert(1)</script> [click](https://bad.invalid)"}])
    shown = cp.answer_summary(controller(ctx, counted(ctx, aid, distinct_by="studyDbId")))
    assert "<script>" not in shown and "&lt;script&gt;" in shown
    assert "[click](https://bad.invalid)" not in shown


def approval_controller(observations=True):
    base = "https://synthetic.invalid/brapi/v2"
    ids = [f"S{i}" for i in range(1, 13)]
    plan = Plan(plan_id="plan_presenter", question="Which studies have yield?", interpretation="Review studies", status="ready",
                scope=ResolvedScope(base_url=base, study_ids=ids, variable_ids=["V1"] if observations else []),
                steps=[PlanStep(step_id="step_1", agent="retriever", action="get_observations" if observations else "export_metadata")],
                created_at_utc=datetime.now(timezone.utc))
    manifest = FetchManifest(run_id="run_presenter", base_url=base, origin="live",
        endpoint_families=["studies", "observationvariables", *(["observations"] if observations else [])],
        observation_study_ids=ids if observations else [], observation_variable_ids=["V1"] if observations else [],
        max_http_attempts=20, max_observation_studies=len(ids) if observations else 0)
    return SimpleNamespace(plan=plan, manifest=manifest, metadata=None, config=SimpleNamespace(question=plan.question, max_fetches=15),
                           budget=SimpleNamespace(max_model_calls=30), catalog_fetched_at="2026-09-29T12:00:00+00:00", catalog_setup=None)


def test_approval_preserves_exact_observation_scope_source_limits_and_semantics():
    view = approval_controller()
    shown = cp.approval_summary(view)
    assert "| S12 |" in shown and "V1" in shown
    assert "https://synthetic.invalid/brapi/v2" in shown and "previously cached" in shown
    assert "2026-09-29T12:00:00+00:00" in shown
    assert "20 HTTP attempts" in shown and "12 observation studies" in shown and "15 data fetch calls" in shown
    assert "does not accept the eventual answer" in shown
    assert "manifest sha256" not in shown and "run_presenter" not in shown


def test_catalog_approval_does_not_imply_observation_permission():
    view = approval_controller(observations=False)
    view.catalog_setup = {"status": "completed"}
    shown = cp.approval_summary(view)
    assert "no observation requests" in shown and "prepared during this question" in shown
    assert "Matched studies: 12" in shown and "2 additional matched IDs are not shown here" in shown
    assert "0 observation studies" in shown and "**Observation scope**" not in shown
