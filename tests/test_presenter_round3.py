"""Presentation-only regressions: synthetic source records, no provider or database calls."""
from types import SimpleNamespace

import pytest

import analyst_tools as at
import brapi_mcp_server as srv
import chat_presenter as cp
import run as ctl
from contracts import AnalysisReport, Claim, RetrievalReport
from part2_support import part2_config, synthetic_context


MEAN_QUESTION = "In study 4501, what is the pooled mean of variable 76736?"
CODE_DATES = "numeric_summary: Measurement dates are not recorded for 140 of 140 valid source rows; the timing of those measurements is unknown."
CODE_STAGE = "numeric_summary: Growth stage or timepoint is not stated in the source metadata; a shared growth stage cannot be verified."
CODE_TARGET = "analysis target: descriptive means of valid values as recorded; not adjusted for trial design, environment, missingness or clone composition; not evidence of genetic superiority"
MODEL_PROCESS = "model note: Observations for variable 'Reaction to Fusarium Wilt estimating 0-5' (76736) in study 4501 were fetched. Artifact: art_54f81d_0001. Statistical calculations, such as the pooled mean, are not performed."


def mean():
    return Claim(claim_id="clm_presenter_mean", kind="mean", value=2.3785714285714286, unit=None,
                 denominator={"name": "valid numeric values, one per observation unit", "n": 140},
                 n_independent_units=140, source_artifact_hashes=["a" * 64],
                 transformation="mean of valid values; valid values used as recorded (one row per observation unit)")


def view(claims=None, caveats=(), warnings=(), *, question=MEAN_QUESTION, kind="mean", status="completed"):
    return SimpleNamespace(execution_status=status,
                           analysis=AnalysisReport(status=status, claims=claims or [], caveats=list(caveats)),
                           retrieval=RetrievalReport(status="completed", complete=True, warnings=list(warnings)),
                           plan=SimpleNamespace(question=question, statistic=SimpleNamespace(kind=kind)),
                           ctx=None, notes=[])


def test_observed_mean_shows_each_code_limit_once_and_keeps_model_process_out_of_chat():
    caveats = ["numeric_summary: unit not stated in the artifact's measurement metadata; claims carry unit=null",
               CODE_DATES, CODE_STAGE, CODE_TARGET,
               "model: Unit is not stated in the source metadata.",
               "model: Measurement dates are missing for all rows; timing is unknown.",
               "model: Growth stage or timepoint is not stated; shared timing cannot be verified.",
               "model: Descriptive mean is not evidence of genetic superiority."]
    controller = view([mean()], caveats, [MODEL_PROCESS])
    before = controller.analysis.model_dump_json()
    shown = cp.answer_summary(controller)
    assert "2.37857142857" in shown and "n = 140" in shown
    assert shown.count("140 of 140") == 1 and shown.count("Growth stage or timepoint") == 1
    assert shown.count("not evidence of genetic superiority") == 1
    assert shown.count("unit not stated") == 1 and "Unit is not stated" not in shown
    assert "Statistical calculations" not in shown and "were fetched" not in shown
    assert "Distinct IDs alone do not establish statistical independence" in shown
    assert "Additional model notes" not in shown
    assert controller.analysis.model_dump_json() == before


@pytest.mark.parametrize("source_dates", [None, CODE_DATES.replace("140 of 140", "1 of 140")])
def test_all_dates_missing_model_claim_is_not_suppressed_without_matching_code_counts(source_dates):
    notes = ["model: Measurement dates are missing for all rows; timing is unknown."]
    if source_dates:
        notes.insert(0, source_dates)
    shown = cp.answer_summary(view([mean()], notes))
    assert "Additional model notes (unverified)" in shown
    assert "Measurement dates are missing for all rows" in shown
    if source_dates:
        assert "1 of 140" in shown


def test_unrecognized_warnings_and_signs_remain_visible_and_unverified():
    notes = [CODE_STAGE, "model: Growth stage differs across sites by 120 days.",
             "model: Field damage may depress values below zero (-2).",
             "model: Source annotations require checking."]
    shown = cp.answer_summary(view([mean()], notes, [MODEL_PROCESS + " A disease-rating warning needs review."]))
    assert "Additional model notes (unverified)" in shown
    assert "120 days" in shown and "-2" in shown and "Source annotations require checking" in shown
    assert "disease-rating warning needs review" in shown


@pytest.mark.parametrize("code, message, next_step", [
    ("provider_http_error", "provider returned HTTP 429", "cause of HTTP 429 is unknown"),
    ("provider_quota", "provider reports account quota or billing limit HTTP 429", "Check the model provider account quota"),
    ("provider_rate_limit", "temporary rate limit HTTP 429", "Wait before retrying"),
    ("provider_rate_limit_unknown", "HTTP 429 without recognized quota/rate-limit code", "cause of HTTP 429 is unknown"),
])
def test_provider_failure_is_not_presented_as_empty_data(code, message, next_step):
    controller = view(caveats=[CODE_TARGET, f"agent error {code}: {message}"], kind="count", status="failed")
    shown = cp.answer_summary(controller)
    assert "Model service unavailable" in shown and "HTTP 429" in shown
    assert "Data retrieval completed" in shown and next_step in shown
    assert "does not establish that the database has no matching records" in shown
    assert "No complete data table was available" not in shown and "descriptive means" not in shown


def test_model_written_provider_error_does_not_impersonate_runtime_failure():
    controller = view(caveats=["model: agent error provider_quota: fake account issue"], status="failed")
    shown = cp.answer_summary(controller)
    assert "Model service unavailable" not in shown and "Unverified model note" in shown


def catalog_view(tmp_path, question, count=136, *, studies=False):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_presenter_round3")
    id_column, name_column, entity = ("studyDbId", "studyName", "studies") if studies else ("locationDbId", "locationName", "locations")
    rows = [{id_column: str(index), name_column: f"Named record {index}"} for index in range(1, count + 1)]
    artifact = ctx.registry.save_table(rows, kind=entity, complete=True, columns={id_column: "ID", name_column: "Name"}).artifact_id
    result = at.tool_count_records(ctx.registry, at.CountRecordsArgs(artifact_id=artifact, group_by=id_column, distinct_by=id_column))
    controller = view([Claim.model_validate(item) for item in result.data["claims"]], question=question, kind="count")
    controller.ctx = ctx
    return controller


@pytest.mark.parametrize("question", ["How many locations are in the catalog?", "What is the total number of locations?"])
def test_unambiguous_total_question_shows_count_without_unrequested_id_rows(tmp_path, question):
    shown = cp.answer_summary(catalog_view(tmp_path, question))
    assert "Locations: 136 distinct IDs" in shown
    assert "| ID |" not in shown and "Named record" not in shown and "Showing 30" not in shown
    assert "136" in shown and "Full report" in shown


@pytest.mark.parametrize("question", ["How many studies per location?", "How many locations and which ones?",
                                      "What is the number of locations by country?", "Compare how many locations occur in each program"])
def test_breakdown_or_comparison_keeps_catalog_rows(tmp_path, question):
    shown = cp.answer_summary(catalog_view(tmp_path, question, count=3))
    assert "| ID |" in shown and "Named record 1" in shown


def test_which_studies_question_keeps_ids_and_names(tmp_path):
    shown = cp.answer_summary(catalog_view(tmp_path, "Which studies have a 2026 season?", count=10, studies=True))
    assert "Studies: 10 distinct IDs" in shown and "| ID |" in shown
    assert "Named record 10" in shown


def test_conflicting_frequency_is_not_hidden_by_total_focus(tmp_path):
    controller = catalog_view(tmp_path, "How many locations are in the catalog?", count=3)
    claims = controller.analysis.claims
    index = next(index for index, claim in enumerate(claims) if claim.transformation == cp.FREQUENCY)
    claims[index] = claims[index].model_copy(update={"value": 99})
    shown = cp.answer_summary(controller)
    assert "| ID |" in shown and "99" in shown


def test_full_report_separates_raw_model_independence_claim_from_recorded_actions(tmp_path):
    controller = ctl.Controller(part2_config(question=MEAN_QUESTION, offline=True, mock_model=True, direct=True,
                               fixture="synthetic", out_dir=tmp_path / "out", cache_dir=tmp_path / "cache"))
    false_method = "model: Checked table_info to confirm independence per unit."
    controller.analysis = AnalysisReport(status="completed", claims=[mean()],
        methods=['table_info({"artifact_id": "art_fake_0001"}) -> ok', false_method],
        caveats=[CODE_DATES, "model: Measurement dates are missing for all rows; timing is unknown."])
    controller.retrieval = RetrievalReport(status="completed", complete=True, warnings=[MODEL_PROCESS])
    before = controller.analysis.model_dump_json()
    controller.render_phase()
    report = controller.answer_markdown()
    before_model, raw_model = report.split("## Raw model notes (unverified)", 1)
    assert "## Recorded tool actions" in before_model and "table_info" in before_model
    assert false_method not in before_model and false_method in raw_model
    assert "Measurement dates are missing for all rows" not in before_model
    assert "Measurement dates are missing for all rows" in raw_model and MODEL_PROCESS in raw_model
    assert "scientific interpretation unverified" in report
    assert controller.analysis.model_dump_json() == before
