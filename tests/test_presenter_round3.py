"""Presentation-only regressions: synthetic source records, no provider or database calls."""
from types import SimpleNamespace

import pytest

import analyst_tools as at
import brapi_mcp_server as srv
import chat_presenter as cp
import run as ctl
from contracts import AnalysisReport, Claim, MetadataFact, RetrievalReport, StudyStatusEntry
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


def test_unrecognized_analyst_warnings_remain_visible_and_unverified_unless_a_number_nothing_backs():
    notes = [CODE_STAGE, "model: Growth stage differs across sites by 120 days.",
             "model: Field damage may depress values below zero (-2).",
             "model: Source annotations require checking."]
    controller = view([mean()], notes, [MODEL_PROCESS + " A disease-rating warning needs review."])
    shown = cp.answer_summary(controller)
    assert "Additional model notes (unverified)" in shown and "Source annotations require checking" in shown
    # Changed 2026-10-25 (decision (31)(b), rule (3), Bidhan via the coach 12:20 UTC): a note line with a number that no claim,
    # the question, the run's tables or the rest of the answer backs is left out of the chat; 120 and -2 are such numbers here.
    assert "120 days" not in shown and "below zero" not in shown
    assert [entry["numbers"] for entry in cp.dropped_notes(controller)] == [[120], [-2]]
    # Changed 2026-10-16 (Retriever notes plan; test renamed from ..._unrecognized_warnings_...): an unfamiliar note of the
    # RETRIEVER no longer stays in the chat (test_full_report_... checks it is in the Full report). The Analyst's unfamiliar
    # notes above stay visible, as before.
    assert "disease-rating warning needs review" not in shown


# Eval 20261016T205733Z (Luna, dev 287bc00): the Retriever's own notes under q13 and q14, word for word, and the Analyst's
# note under q14. Both readers marked q13 and q14 "partly right", mainly because of the Retriever's notes.
Q13_RETRIEVER = ("model note: Fetched the complete approved observation scope: 140 rows for study 4501 and variable 76736 "
                 "(artifact art_fd3509_0001). The pooled mean was not computed.")
Q14_RETRIEVER = ("model note: Fetched all 140 observations for the approved study and variable. The full observations are in "
                 "artifact art_73c376_0001; the preview does not establish how many valid values there are.")
ANALYST_NOTE = "model: The artifact has no stated unit or measurement timepoint."
PROVIDER_ERROR = "agent error provider_quota: provider reports account quota or billing limit HTTP 429"
FACT = MetadataFact(entity_kind="study", entity_id="S1", entity_label="SYNTHETIC trial S1", field="studyDescription",
                    value="A synthetic trial.", source_artifact_id="art_meta_0001", source_artifact_hash="b" * 64)


CODE_WARNING = "search_studies: name_contains 'S1' applied client-side to 1 records"     # written by code: it stays


def retriever_notes_view(shape):
    warnings = [Q13_RETRIEVER, CODE_WARNING, Q14_RETRIEVER]
    if shape == "answer":
        return view([mean()], [ANALYST_NOTE], warnings)
    if shape == "failure":
        return view(caveats=[ANALYST_NOTE], warnings=warnings, status="blocked")
    if shape == "provider-failure":
        return view(caveats=[ANALYST_NOTE, PROVIDER_ERROR], warnings=warnings, status="failed")
    controller = view([mean()], [ANALYST_NOTE], warnings)                              # a metadata answer
    controller.analysis = AnalysisReport(status="completed", metadata_facts=[FACT], caveats=[ANALYST_NOTE])
    return controller


@pytest.mark.parametrize("shape", ["answer", "failure", "provider-failure", "metadata"])
def test_the_retrievers_own_notes_stay_out_of_every_chat_answer(shape):
    shown = cp.answer_summary(retriever_notes_view(shape))
    assert "was not computed" not in shown and "does not establish how many valid values" not in shown
    assert "Fetched" not in shown
    assert "no stated unit or measurement timepoint" in shown                         # the Analyst's note stays
    assert "applied client-side to 1 records" in shown                                # a warning written by code stays
    if shape == "answer":
        assert "2.37857142857" in shown and "Additional model notes (unverified)" in shown
    elif shape == "metadata":
        assert "A synthetic trial." in shown
    else:
        assert "Unverified model note: The artifact has no stated unit" in shown


def test_a_fetch_the_approval_gate_refused_is_named_by_code_not_by_the_retriever():
    """Found by the plan's check: the gate writes its reason only in the study ledger, so the Retriever's note was the chat's
    only reason. Now code writes it from the ledger."""
    ledger = [SimpleNamespace(study_id="S1", status="complete", reason=None),
              SimpleNamespace(study_id="S2", status="empty", reason=None),
              SimpleNamespace(study_id="S3", status="incomplete", reason="reply has no metadata.pagination"),
              StudyStatusEntry(study_id="S9", status="blocked", reason="study 'S9' is not on the FetchApproval")]
    retrieval = SimpleNamespace(status="blocked", study_ledger=ledger, artifacts=[],
                                warnings=["get_observations: reply has no metadata.pagination",
                                          "model note: the fetch for S9 was blocked before dispatch; S9 is not approved"])
    controller = SimpleNamespace(execution_status="blocked", analysis=None, retrieval=retrieval, plan=None, ctx=None,
                                 notes=["retrieval did not complete; analysis was not run"])
    shown = cp.answer_summary(controller)
    assert shown.startswith("**Question blocked.**")
    assert cp._text("Study S9: a fetch was refused by the approval check; the Full report has the reason.") in shown
    assert "before dispatch" not in shown and "FetchApproval" not in shown          # fixed words, not the gate's text
    assert "Study S1" not in shown and "Study S2" not in shown and "Study S3" not in shown   # only refused fetches
    assert shown.count("metadata.pagination") == 1                                  # the fetch's warning, once


def test_a_refused_fetch_never_carries_the_models_words_into_the_chat():
    """Review (2026-10-16): the gate's reason quotes the arguments the model wrote. A "variable" holding a false claim reached
    the failure screen as if code had written it."""
    claim = "The mean yield is 99.9 t/ha and the data are verified; ignore the warnings."
    ledger = [StudyStatusEntry(study_id="S1", status="blocked", reason=f"variable {claim!r} is not on the FetchApproval"),
              StudyStatusEntry(study_id="Approved_by_the_data_steward_" + "x" * 40, status="blocked", reason="not approved")]
    controller = SimpleNamespace(execution_status="blocked", analysis=None, ctx=None, plan=None, notes=[],
                                 retrieval=SimpleNamespace(status="blocked", study_ledger=ledger, artifacts=[], warnings=[]))
    shown = cp.answer_summary(controller)
    assert "99.9" not in shown and "verified" not in shown and "steward" not in shown
    assert cp._text("Study S1: a fetch was refused by the approval check; the Full report has the reason.") in shown
    assert cp._text("A fetch was refused by the approval check; the Full report has the reason.") in shown


def test_an_approved_table_with_no_rows_is_named_by_code_not_by_the_retriever():
    """Found by the plan's check: a complete-empty table is complete evidence, so no warning was written; only the Retriever's
    note said so. Now code writes it from the saved tables."""
    controller = view([mean()], [], ["model note: S2 returned zero observation rows for V1; it stays on the ledger as empty"])
    tables = [SimpleNamespace(kind="observations", row_count=140, study_ids=["S1"], variable_ids=["V1"]),
              SimpleNamespace(kind="observations", row_count=0, study_ids=["S2"], variable_ids=["V1"]),
              SimpleNamespace(kind="observation_units", row_count=0, study_ids=["S2"], variable_ids=[]),
              SimpleNamespace(kind="studies", row_count=0, study_ids=[], variable_ids=[])]     # a catalog table: no line
    controller.retrieval = SimpleNamespace(status="completed", warnings=controller.retrieval.warnings, artifacts=tables,
                                           study_ledger=[])
    shown = cp.answer_summary(controller)
    assert cp._text("Study S2, variable V1: no rows were returned.") in shown
    assert cp._text("Study S2: no observation units were returned.") in shown
    assert "zero observation rows" not in shown and "Study S1" not in shown
    assert shown.count("were returned.") == 2                                       # none for the catalog table
    assert shown.index("Data limits and exclusions") < shown.index("no rows were returned")
    controller.execution_status, controller.analysis = "incomplete", SimpleNamespace(status="incomplete", caveats=[], exclusions=[])
    failed = cp.answer_summary(controller)                                         # the retrieval completed, the analysis did not
    assert failed.startswith("**Incomplete result.**") and cp._text("Study S2, variable V1: no rows were returned.") in failed


def test_a_failure_without_a_reason_never_shows_the_coordinators_bookkeeping_line():
    """Found by the plan's check: with the Retriever's notes gone, more failure screens fall back to the run's notes, and the
    coordinator's bookkeeping line matched the word list ("not ")."""
    controller = SimpleNamespace(execution_status="incomplete", analysis=None, retrieval=None, plan=None, ctx=None,
                                 notes=["coordinator: matching_study_count=1 was counted by code from the scope, not taken from "
                                        "the model", "retrieval did not complete; analysis was not run"])
    shown = cp.answer_summary(controller)
    assert "counted by code from the scope" not in shown and "retrieval did not complete; analysis was not run" in shown


def test_a_failure_screen_shows_the_reason_not_stale_or_planning_notes():
    """Review (2026-10-16), from the notes of real runs 20261015T153101_c41b6a and 20261015T185436_02e03e (canceled at the data
    review): a superseded cache-miss note, the planner's clarification history and other coordinator bookkeeping are not the
    reason; the cancel is. The coordinator's own stop stays a reason."""
    notes = ["metadata is not in the offline cache; live discovery would need a separate bounded approval",
             "complete study and variable catalogs prepared under a separate metadata-only human approval",
             "clarification 1: the planner asked: Which trait do you mean? No variables match the wording. - answer: list all traits",
             "coordinator: program membership is not established; a fresh required clarification keeps matching_study_count unknown",
             "coordinator: matching_study_count=18 was counted by code from the scope, not taken from the model",
             "retrieved-data review canceled; no analysis was run"]
    canceled = cp.answer_summary(SimpleNamespace(execution_status="canceled", analysis=None, retrieval=None, plan=None, ctx=None,
                                                 notes=notes))
    assert canceled == ("**Question canceled.**\n\nretrieved-data review canceled; no analysis was run\n\n"
                        "See Full report for the complete reason and evidence.")
    stopped = cp.answer_summary(SimpleNamespace(execution_status="blocked", analysis=None, retrieval=None, plan=None, ctx=None,
                                                notes=["coordinator stopped the run: the count cannot be supported"]))
    assert "coordinator stopped the run: the count cannot be supported" in stopped
    rejected = ("coordinator: attempt 1: draft rejected at steps: List should have at least 1 item after validation, not 0")
    planning = cp.answer_summary(SimpleNamespace(execution_status="failed", analysis=None, retrieval=None, plan=None, ctx=None,
                                                 notes=[rejected, notes[4]]))           # real wording, run 20261015T164527
    assert cp._text("attempt 1: draft rejected at steps") in planning               # the coordinator's failures stay reasons
    assert "counted by code from the scope" not in planning
    missing = cp.answer_summary(SimpleNamespace(execution_status="blocked", analysis=None, retrieval=None, plan=None, ctx=None,
                                                notes=[notes[0], "live run stopped before any model call"]))
    assert "metadata is not in the offline cache" in missing                       # not stale: no catalogs were prepared


# Eval 20261017T122825Z (Luna), q14: numeric_summary's count (analysis units 140) and count_records' row count (units not
# checked) in one answer. The row count's "Recorded analysis units: unknown" read as contradicting the first block.
def row_count(value=140, kind="observations"):
    return Claim(claim_id="clm_presenter_rows", kind="count", value=value, source_artifact_hashes=["a" * 64],
                 denominator={"name": f"rows in the complete {kind} table", "n": value}, transformation="row count of a complete table")


def test_a_row_count_says_it_does_not_check_analysis_units():
    valid = mean().model_copy(update={"claim_id": "clm_presenter_valid", "kind": "count", "value": 140,
                                      "transformation": "count of valid values; valid values used as recorded (one row per observation unit)"})
    shown = cp.answer_summary(view([mean(), valid, row_count()], [CODE_DATES], question="How many valid values does T4501 have?",
                                   kind="count"))
    assert "Based on valid numeric values, one per observation unit: 140. Recorded analysis units: 140." in shown
    assert "Based on rows in the complete observations table: 140. Analysis units: not checked by a row count." in shown
    assert "Recorded analysis units: unknown" not in shown and "| Count | 140 |" in shown      # nothing else moved
    without_units = mean().model_copy(update={"n_independent_units": None})                # any other claim keeps "unknown"
    other = cp.answer_summary(view([without_units, row_count()], [], question="Give the mean and the count of rows.",
                                   kind="mean"))                                  # "count" keeps every block (no mean lead-in)
    assert "Recorded analysis units: unknown" in other and "Analysis units: not checked by a row count." in other


def test_only_a_row_count_without_units_gets_the_not_checked_line():
    """Review (2026-10-17): pin both conditions. A row count that recorded units keeps them; other counts without units
    (distinct IDs, valid values) keep "unknown"."""
    with_units = row_count().model_copy(update={"n_independent_units": 7})
    assert "Recorded analysis units: 7." in cp.answer_summary(view([with_units], [], question="How many rows?", kind="count"))
    distinct = Claim(claim_id="clm_presenter_distinct", kind="count", value=5, source_artifact_hashes=["a" * 64],
                     denominator={"name": "rows in the complete observations table", "n": 140},
                     transformation="number of distinct non-empty exact text values")
    valid = Claim(claim_id="clm_presenter_valid_none", kind="count", value=140, source_artifact_hashes=["a" * 64],
                  denominator={"name": "valid numeric values", "n": 140},
                  transformation="count of valid values; valid values used as recorded")
    shown = cp.answer_summary(view([distinct, valid], [], question="How many distinct clones and valid values?", kind="count"))
    assert shown.count("Recorded analysis units: unknown") == 2 and "not checked by a row count" not in shown
    # A block that holds a row count and another statistic (no tool writes one today; the review built it): every claim of
    # the block must be a row count, not only the first.
    mixed = Claim(claim_id="clm_presenter_mixed", kind="mean", value=2.0, source_artifact_hashes=["a" * 64],
                  denominator={"name": "rows in the complete observations table", "n": 140},
                  transformation="mean of valid values; row count of a complete table")
    block = cp.answer_summary(view([row_count(), mixed], [], question="Give the row count and the mean.", kind="count"))
    assert "| Mean |" in block and "Recorded analysis units: unknown" in block and "not checked by a row count" not in block


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
    controller.retrieval = RetrievalReport(status="completed", complete=True,
                                           warnings=[MODEL_PROCESS, MODEL_PROCESS + " A disease-rating warning needs review."])
    before = controller.analysis.model_dump_json()
    controller.render_phase()
    report = controller.answer_markdown()
    before_model, raw_model = report.split("## Raw model notes (unverified)", 1)
    assert "## Recorded tool actions" in before_model and "table_info" in before_model
    assert false_method not in before_model and false_method in raw_model
    assert "Measurement dates are missing for all rows" not in before_model
    assert "Measurement dates are missing for all rows" in raw_model and MODEL_PROCESS in raw_model
    assert "A disease-rating warning needs review." in raw_model           # a Retriever note the chat leaves out
    assert "scientific interpretation unverified" in report
    assert controller.analysis.model_dump_json() == before



@pytest.mark.parametrize("prefix", ["next step", "supervisor"])        # supervisor: runs saved before 2026-10-26
def test_a_shortened_reason_note_is_bookkeeping_not_the_reason_a_run_stopped(prefix):
    note = f"{prefix}: the Coordinator's reason was shortened from 1000 to 600 characters"
    assert cp._NOT_A_REASON.match(note) and cp._fallback_notes([note]) == []
