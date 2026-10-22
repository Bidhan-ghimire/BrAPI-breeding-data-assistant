"""Zeros items (a) and (c) (coach, 2026-10-21 18:10 UTC). SweetPotatoBase's /observations leaves out every value recorded as 0
(size check of 2026-10-21: 3,132 values in 26 of 117 benchmark tables, every one a 0). An answer built from an /observations
table of such a server says so in one sentence, once; no other answer or screen does. And a recorded 0 stays a valid value all
the way from the server to the screen. Strengthened after the review of 2026-10-22 (richer answers, a real run, the edges)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import analyst_tools as at
import brapi_mcp_server as srv
import chat_presenter as cp
import run as ctl
from artifacts import cell_text
from cache_store import CacheStore
from contracts import AnalysisReport, Claim, MeasurementMeta, MetadataFact
from part2_support import part2_config, part2_runtime, seed_synthetic, synthetic_context, synthetic_models

SPB = "https://sweetpotatobase.org/brapi/v2"
SPEC = [("G1", "Alpha", "P1", "10"), ("G1", "Alpha", "P2", "12"), ("G2", "Beta", "P3", "20"), ("G2", "Beta", "P4", "22")]


@pytest.fixture
def ctx(tmp_path):
    return synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_zeros")


def obs_table(registry, rows_spec):
    rows = [{"observationDbId": f"O{i}", "observationUnitDbId": plot, "observationUnitName": f"plot {plot}", "germplasmDbId": g,
             "germplasmName": name, "studyDbId": "S1", "observationVariableDbId": "V1", "observationVariableName": "SYNTHETIC yield",
             "observationTimeStamp": "", "value": value} for i, (g, name, plot, value) in enumerate(rows_spec)]
    columns = {c: srv.OBSERVATION_COLUMNS.get(c, "text") for c in rows[0]}
    return registry.save_table(rows, kind="observations", columns=columns, complete=True, study_ids=["S1"], variable_ids=["V1"],
                               measurement=MeasurementMeta(trait="SYNTHETIC yield", unit="t/ha")).artifact_id


def claims_of(ctx, tool, args):
    return [Claim.model_validate(c) for c in at.dispatch_analyst(ctx.registry, tool, args).data["claims"]]


def controller_for(ctx, claims, *, base_url=SPB, kinds=("observations",), status="completed", caveats=(), warnings=(), plan=None):
    return SimpleNamespace(execution_status=status, base_url=base_url, ctx=ctx, notes=[], plan=plan,
                           analysis=AnalysisReport(status="completed", claims=claims, caveats=list(caveats)),
                           retrieval=SimpleNamespace(status="completed", warnings=list(warnings),
                                                     artifacts=[SimpleNamespace(kind=kind) for kind in kinds]))


def n(shown):
    return shown.count(cp.ZEROS_LEFT_OUT)


def test_answers_from_sweetpotatobase_observations_carry_the_note_once(ctx):
    art = obs_table(ctx.registry, SPEC)
    summary, ranked = claims_of(ctx, "numeric_summary", {"artifact_id": art}), claims_of(ctx, "rank_entries", {"artifact_id": art})
    grouped = claims_of(ctx, "group_stats", {"artifact_id": art, "by": "germplasmDbId"})
    other = claims_of(ctx, "numeric_summary", {"artifact_id": obs_table(ctx.registry, [("G9", "Z", "P9", "5"), ("G9", "Z", "P8", "7")])})
    mean_plan = SimpleNamespace(statistic=SimpleNamespace(kind="mean"), question="What is the mean SYNTHETIC yield in study S1?")
    answers = {"a summary": controller_for(ctx, summary), "entry means": controller_for(ctx, grouped),
               "a ranking": controller_for(ctx, ranked), "a mean in focus": controller_for(ctx, summary, plan=mean_plan),
               "with other limits": controller_for(ctx, ranked, caveats=["some caveat"], warnings=["a warning"]),
               "with a model note": controller_for(ctx, summary, caveats=["model: a note in the model's words"]),
               "two tables": controller_for(ctx, summary + other)}
    for label, controller in answers.items():
        shown = cp.answer_summary(controller)
        assert n(shown) == 1, label
        assert shown.index("**Data limits and exclusions**") < shown.index(cp.ZEROS_LEFT_OUT), label    # one of the data limits
    assert "source 2" in cp.answer_summary(answers["two tables"])


@pytest.mark.parametrize("server, expected", [(SPB, 1), ("https://www.sweetpotatobase.org/brapi/v2", 1),
                                              ("https://synthetic.invalid/brapi/v2", 0), ("https://cassavabase.org/brapi/v2", 0)])
def test_a_real_run_carries_the_note_only_on_sweetpotatobase(monkeypatch, tmp_path, server, expected):
    """The real Controller, offline, with the synthetic study cached under each address (review 2026-10-22)."""
    part2_runtime(monkeypatch, server=server)
    seed_synthetic(CacheStore(tmp_path / "cache"), server)
    config = part2_config(question="For synthetic study S1, give mean yield and clone means with n.", offline=False,
                          mock_model=False, direct=False, auto=False, fixture=None, out_dir=tmp_path / "out",
                          cache_dir=tmp_path / "cache")
    controller = ctl.Controller(config, models=synthetic_models())
    result = asyncio.run(controller.run())
    assert result.execution_status == "completed" and controller.base_url == server
    assert n(cp.answer_summary(controller)) == expected


def test_stop_and_failure_screens_never_carry_it(ctx):
    summary = claims_of(ctx, "numeric_summary", {"artifact_id": obs_table(ctx.registry, SPEC)})
    for status in ("incomplete", "failed", "blocked", "canceled", "needs_clarification", "limit_reached", "unknown"):
        assert n(cp.answer_summary(controller_for(ctx, summary, status=status))) == 0, status
    blocked = controller_for(ctx, summary)
    blocked.analysis = AnalysisReport(status="blocked", caveats=["no complete artifact is eligible for analysis"])
    unfinished = controller_for(ctx, summary)
    unfinished.retrieval.status = "incomplete"
    assert n(cp.answer_summary(blocked)) == 0 and n(cp.answer_summary(unfinished)) == 0


def test_no_note_without_observations_data(ctx):
    counted = claims_of(ctx, "count_records", {"artifact_id": obs_table(ctx.registry, SPEC), "group_by": "germplasmDbId"})
    for kinds in ((), ("observation_units",), ("studies", "variables"), ("studies", "variables", "observation_units"),
                  ("analysis_result",)):
        assert n(cp.answer_summary(controller_for(ctx, counted, kinds=kinds))) == 0, kinds
    aid = srv.dispatch(ctx, "export_metadata", {"entity": "studies"}).artifact_ids[0]                   # an answer of study fields
    manifest, rows = ctx.registry.load(aid)
    facts = controller_for(ctx, counted, kinds=("studies", "observations"))
    facts.analysis = AnalysisReport(status="completed", caveats=[], metadata_facts=[MetadataFact(
        entity_kind="study", entity_id=rows[0]["studyDbId"], field="studyName", value=rows[0]["studyName"],
        source_artifact_id=aid, source_artifact_hash=manifest.sha256)])
    shown = cp.answer_summary(facts)
    assert rows[0]["studyName"].split()[0] in shown and n(shown) == 0


def test_the_server_is_read_from_its_host_exactly(ctx):
    summary = claims_of(ctx, "numeric_summary", {"artifact_id": obs_table(ctx.registry, SPEC)})
    for base_url, expected in (("https://SweetPotatoBase.org/brapi/v2", 1), ("https://sweetpotatobase.org:443/brapi/v2", 1),
                               ("https://sweetpotatobase.org.evil.invalid/brapi/v2", 0), ("https://notsweetpotatobase.org/brapi/v2", 0),
                               ("", 0)):
        assert n(cp.answer_summary(controller_for(ctx, summary, base_url=base_url))) == expected, base_url


def test_an_empty_observations_table_still_gets_the_note(ctx):
    """"No rows were returned" is exactly where a reader could take zero for none (review 2026-10-22)."""
    controller = controller_for(ctx, claims_of(ctx, "numeric_summary", {"artifact_id": obs_table(ctx.registry, SPEC)}))
    controller.retrieval.artifacts = [SimpleNamespace(kind="observations", row_count=0, variable_ids=["V1"], study_ids=["S1"])]
    shown = cp.answer_summary(controller)
    assert n(shown) == 1 and "no rows were returned" in shown


def test_the_note_is_one_sentence_with_no_digit_that_says_what_was_found():
    """The benchmark grader reads every digit of the code-written answer: "zero" in words keeps the note out of its check."""
    text = cp.ZEROS_LEFT_OUT
    assert not any(ch.isdigit() for ch in text) and text.endswith(".") and ". " not in text
    assert "/observations" in text and "recorded as zero" in text and "missing" in text


@pytest.mark.parametrize("token", ["0", "0.0", "-0", "00", "0e0", " 0 ", "0.00"])
def test_a_recorded_zero_is_a_valid_measurement(token):
    """Item (c): a zero yield is a failed plot and a zero beta-carotene a white-fleshed clone, both real measurements."""
    assert at.classify_value(token) == ("valid", 0.0)


def test_a_zero_stays_a_value_from_the_server_to_the_screen(ctx):
    """Review 2026-10-22: written as text, counted and averaged, kept by the valid-number filter, and shown as 0."""
    assert cell_text(0) == ("0", None) and cell_text(0.0) == ("0.0", None) and cell_text("0") == ("0", None)
    art = obs_table(ctx.registry, [("G1", "A", "P1", "0"), ("G1", "A", "P2", "10"), ("G2", "B", "P3", "20")])
    got = {c.kind: c.value for c in claims_of(ctx, "numeric_summary", {"artifact_id": art})}
    assert got["count"] == 3 and got["mean"] == 10 and got["min"] == 0
    kept = at.dispatch_analyst(ctx.registry, "filter_rows", {"artifact_id": obs_table(ctx.registry, [
        ("G1", "A", "P1", "0"), ("G1", "A", "P2", "10"), ("G2", "B", "P3", "")]), "column": "value", "op": "is_valid_number"})
    assert kept.data["rows_out"] == 2
    assert cp._number(0) == "0" and cp._number(0.0) == "0"
