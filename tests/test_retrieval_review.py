"""Human review renders real managed tables and binds consent to their bytes.

Synthetic local artifacts only; neither a model nor a network service supplies
these expectations. The ordinary test guard blocks external sockets.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
from types import SimpleNamespace

import pytest

from artifacts import ArtifactError, ArtifactRegistry
from contracts import MeasurementMeta, Plan, RetrievalReport
from retrieval_review import _cell, review_snapshot, verify_reviewed_tables


BASE = "https://review-test.invalid/brapi/v2"


@pytest.fixture
def review(tmp_path):
    registry = ArtifactRegistry(tmp_path / "out", "run_review", base_url=BASE)
    columns = {key: "synthetic text" for key in (
        "studyDbId", "germplasmDbId", "germplasmName", "observationVariableDbId",
        "observationUnitDbId", "value", "unit", "locationDbId", "private_extra", "other_extra",
    )}
    rows = [{"studyDbId": "S1", "germplasmDbId": f"G{i}", "germplasmName": f"Row {i}",
             "observationVariableDbId": "V1", "observationUnitDbId": f"P{i}", "value": str(i * 10),
             "unit": "t/ha", "locationDbId": "L1", "private_extra": "UNSHOWN COLUMN VALUE",
             "other_extra": "also outside preview"} for i in range(1, 7)]
    artifact = registry.save_table(rows, kind="observations", columns=columns, complete=True,
                                   study_ids=["S1"], variable_ids=["V1"])
    plan = Plan.model_validate({
        "plan_id": "plan_review", "question": "Mean of V1 in S1", "interpretation": "Synthetic data review",
        "status": "ready", "created_at_utc": datetime.now(timezone.utc),
        "scope": {"base_url": BASE, "study_ids": ["S1"], "variable_ids": ["V1"]},
        "steps": [{"step_id": "step_1", "agent": "retriever", "action": "get_observations",
                   "inputs": {"study_db_id": "S1", "variable_db_id": "V1"}}],
    })
    report = RetrievalReport(status="completed", complete=True, artifacts=[artifact])
    ctx = SimpleNamespace(registry=registry,
                          variable_measurements={"V1": MeasurementMeta(trait="root yield", unit="t/ha")})
    return SimpleNamespace(ctx=ctx, report=report, plan=plan, artifact=artifact, rows=rows)


def test_review_uses_verified_table_counts_and_explicit_bounded_preview(review):
    markdown, fingerprints = review_snapshot(review.ctx, review.report, review.plan,
                                              [{"origin": "live"}, {"origin": "cache"}, {"origin": "cache"}])
    assert "observations · 6 rows" in markdown
    assert "Preview: 5 of 6 rows; 8 of 10 columns." in markdown
    assert "This sample is not the full table." in markdown
    assert "| Row 1 |" in markdown and "| Row 5 |" in markdown
    assert "Row 6" not in markdown and "UNSHOWN COLUMN VALUE" not in markdown
    assert "**HTTP attempts:** 1" in markdown and "**Cached responses:** 2" in markdown
    assert "root yield" in markdown and "t/ha" in markdown and BASE in markdown
    assert fingerprints == [{"artifact_id": review.artifact.artifact_id, "sha256": review.artifact.sha256,
                             "row_count": 6, "complete": True}]
    assert verify_reviewed_tables(review.ctx, review.report, fingerprints) is None


def test_review_explains_analysis_has_not_started_and_consent_is_limited(review):
    markdown, _ = review_snapshot(review.ctx, review.report, review.plan, [])
    assert "Analysis has not started" in markdown
    assert "analysis only" in markdown
    assert "does not accept an answer or authorize additional retrieval" in markdown


def test_review_escapes_html_pipes_backticks_and_newlines_from_table_and_warnings(review):
    dangerous = '<img src=x onerror="alert(1)"> | `code`\nnext'
    artifact = review.ctx.registry.save_table([{"unsafe|<name>": dangerous}], kind="locations",
                                              columns={"unsafe|<name>": "synthetic"}, complete=True)
    report = RetrievalReport(status="completed", complete=True, artifacts=[artifact], warnings=[
        '<script>bad()</script> | `warning`', "model note: Unsupported confidence claim",
    ])
    markdown, _ = review_snapshot(review.ctx, report, review.plan, [])
    assert "<img" not in markdown and "<script>" not in markdown
    assert "&lt;img" in markdown and "&lt;script&gt;" in markdown
    assert "unsafe&#124;&lt;name&gt;" in markdown
    assert "&#124; &#96;code&#96; next" in markdown
    assert "Unsupported confidence claim" not in markdown
    assert "**Data limitations:**" in markdown


def test_long_cells_are_bounded_and_truncation_is_explicit(review):
    token = "z" * 220
    artifact = review.ctx.registry.save_table([{"locationName": token}], kind="locations",
                                              columns={"locationName": "label"}, complete=True)
    report = RetrievalReport(status="completed", complete=True, artifacts=[artifact])
    markdown, _ = review_snapshot(review.ctx, report, review.plan, [])
    assert token not in markdown and "z" * 180 + "…" in markdown
    assert "Cells longer than 180 characters are shortened" in markdown


def test_complete_empty_table_is_reviewable_and_does_not_invent_rows(review):
    artifact = review.ctx.registry.save_table([], kind="locations", columns={"locationDbId": "ID"}, complete=True)
    report = RetrievalReport(status="completed", complete=True, artifacts=[artifact])
    markdown, fingerprints = review_snapshot(review.ctx, report, review.plan, [])
    assert "locations · 0 rows" in markdown and "Preview: 0 of 0 rows" in markdown
    assert fingerprints[0]["row_count"] == 0
    verify_reviewed_tables(review.ctx, report, fingerprints)


@pytest.mark.parametrize("status", ["incomplete", "failed", "needs_clarification", "blocked"])
def test_incomplete_report_cannot_enter_review(review, status):
    report = RetrievalReport(status=status, complete=False, artifacts=[review.artifact])
    with pytest.raises(ArtifactError):
        review_snapshot(review.ctx, report, review.plan, [])


def test_complete_report_without_tables_is_refused(review):
    report = RetrievalReport(status="completed", complete=True, artifacts=[])
    with pytest.raises(ArtifactError):
        review_snapshot(review.ctx, report, review.plan, [])


def test_incomplete_declared_or_registered_table_is_refused(review):
    incomplete = review.artifact.model_copy(update={"complete": False})
    report = RetrievalReport(status="completed", complete=True, artifacts=[incomplete])
    with pytest.raises(ArtifactError):
        review_snapshot(review.ctx, report, review.plan, [])
    review.ctx.registry._manifests[review.artifact.artifact_id] = incomplete
    with pytest.raises(ArtifactError):
        review_snapshot(review.ctx, review.report, review.plan, [])


def test_mismatched_declared_hash_is_refused_before_review(review):
    different = review.artifact.model_copy(update={"sha256": "0" * 64})
    report = RetrievalReport(status="completed", complete=True, artifacts=[different])
    with pytest.raises(ArtifactError):
        review_snapshot(review.ctx, report, review.plan, [])


@pytest.mark.parametrize("when", ["before", "after"])
def test_file_tampering_is_refused_before_or_after_human_review(review, when):
    fingerprints = []
    if when == "after":
        _, fingerprints = review_snapshot(review.ctx, review.report, review.plan, [])
    path = review.ctx.registry.out_root / review.artifact.relative_path
    path.write_bytes(path.read_bytes().replace(b"Row 1", b"BAD 1"))
    with pytest.raises(ArtifactError, match="hash_mismatch"):
        if when == "before":
            review_snapshot(review.ctx, review.report, review.plan, [])
        else:
            verify_reviewed_tables(review.ctx, review.report, fingerprints)


def test_changed_file_with_updated_registry_hash_is_still_not_the_reviewed_bytes(review):
    _, fingerprints = review_snapshot(review.ctx, review.report, review.plan, [])
    path = review.ctx.registry.out_root / review.artifact.relative_path
    changed_bytes = path.read_bytes().replace(b"Row 1", b"BAD 1")
    path.write_bytes(changed_bytes)
    changed = review.artifact.model_copy(update={"sha256": hashlib.sha256(changed_bytes).hexdigest()})
    review.ctx.registry._manifests[changed.artifact_id] = changed
    assert review.ctx.registry.load(changed.artifact_id)[0].sha256 == changed.sha256
    with pytest.raises(ArtifactError):
        verify_reviewed_tables(review.ctx, review.report, fingerprints)


@pytest.mark.parametrize("change", ["added", "removed", "replaced"])
def test_changed_artifact_set_requires_another_review(review, change):
    _, fingerprints = review_snapshot(review.ctx, review.report, review.plan, [])
    new = review.ctx.registry.save_table([{"locationDbId": "L1"}], kind="locations",
                                         columns={"locationDbId": "ID"}, complete=True)
    artifacts = {"added": [review.artifact, new], "removed": [], "replaced": [new]}[change]
    changed = RetrievalReport(status="completed", complete=True, artifacts=artifacts)
    with pytest.raises(ArtifactError):
        verify_reviewed_tables(review.ctx, changed, fingerprints)


def test_report_losing_completeness_during_review_blocks_analysis(review):
    _, fingerprints = review_snapshot(review.ctx, review.report, review.plan, [])
    changed = RetrievalReport(status="incomplete", complete=False, artifacts=[review.artifact])
    with pytest.raises(ArtifactError):
        verify_reviewed_tables(review.ctx, changed, fingerprints)


def test_table_losing_completeness_during_review_blocks_analysis(review):
    _, fingerprints = review_snapshot(review.ctx, review.report, review.plan, [])
    changed = review.artifact.model_copy(update={"complete": False})
    review.ctx.registry._manifests[changed.artifact_id] = changed
    with pytest.raises(ArtifactError):
        verify_reviewed_tables(review.ctx, review.report, fingerprints)


def test_mismatched_declared_row_count_is_refused_before_review(review):
    changed = review.artifact.model_copy(update={"row_count": 999})
    report = RetrievalReport(status="completed", complete=True, artifacts=[changed])
    with pytest.raises(ArtifactError):
        review_snapshot(review.ctx, report, review.plan, [])


@pytest.mark.parametrize("update", [{"sha256": "0" * 64}, {"row_count": 999}, {"complete": False}])
def test_changed_report_declaration_is_refused_even_when_registry_bytes_are_unchanged(review, update):
    _, fingerprints = review_snapshot(review.ctx, review.report, review.plan, [])
    changed = review.artifact.model_copy(update=update)
    report = RetrievalReport(status="completed", complete=True, artifacts=[changed])
    assert review.ctx.registry.load(changed.artifact_id)[0] == review.artifact
    with pytest.raises(ArtifactError):
        verify_reviewed_tables(review.ctx, report, fingerprints)


def test_database_markdown_cannot_add_images_links_or_formatting_to_review(review):
    dangerous = r"![pixel](https://image.invalid/x) [click](https://link.invalid) *bold* _em_ \\path #heading ~strike~ 'quote'"
    artifact = review.ctx.registry.save_table([{"locationName": dangerous}], kind="locations",
                                              columns={"locationName": "label"}, complete=True)
    report = RetrievalReport(status="completed", complete=True, artifacts=[artifact])
    markdown, _ = review_snapshot(review.ctx, report, review.plan, [])
    assert "![pixel]" not in markdown and "[click]" not in markdown
    assert "&#33;&#91;pixel&#93;&#40;" in markdown
    assert "&#91;click&#93;&#40;" in markdown
    assert "*bold*" not in markdown and "_em_" not in markdown
    assert "| locationName |" in markdown and "| --- |" in markdown
    # Literal source wording remains readable after the Markdown renderer decodes
    # entities; quotes/ampersands are not corrupted by a second escape pass.
    assert html.unescape(_cell(dangerous)) == dangerous
