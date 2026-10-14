"""Synthetic regressions: absent metadata must not become zero matching memberships."""
from datetime import datetime, timezone

import pytest

import agents.coordinator as co
import analyst_tools as at
from artifacts import ArtifactRegistry
from part2_support import SYNTHETIC_BASE


def metadata(*programs):
    return co.KnownMetadata(base_url=SYNTHETIC_BASE, complete=True, studies=[
        co.KnownStudy(study_id=f"S{i}", name=f"SYNTHETIC trial {i}", program_id=program_id, program_name=program_name)
        for i, (program_id, program_name) in enumerate(programs, 1)])


def proposed(*, column="studyName", value="NCSU", filters=None):
    return co.PlanDraft.model_validate({
        "interpretation": "SYNTHETIC catalog study query", "scope": {"all_known_studies": True, "filters": filters or {}},
        "statistic": {"kind": "count", "denominator": "studies", "grouping": ["studyDbId"]},
        "steps": [
            {"step_id": "step_fetch", "agent": "retriever", "action": "export_metadata", "inputs": {"entity": "studies"}},
            {"step_id": "step_filter", "agent": "analyst", "action": "filter_rows", "inputs": {"column": column, "op": "contains", "value": value}, "depends_on": ["step_fetch"]},
            {"step_id": "step_count", "agent": "analyst", "action": "count_records", "inputs": {"distinct_by": "studyDbId"}, "depends_on": ["step_filter"]},
        ]})


def assemble(question, draft, known):
    plan, problems, _ = co.assemble_plan(question, draft, known, base_url=SYNTHETIC_BASE,
                                        now=datetime(2026, 9, 29, tzinfo=timezone.utc), plan_id="plan_missing",
                                        person_reviews_scope=True)
    assert problems == [] and plan is not None
    return plan


def test_saved_ncsu_plan_shape_cannot_claim_zero_from_study_names():
    plan = assemble("Which public studies belong to the NCSU breeding program?",
                    proposed(filters={"name_contains": "NCSU"}), metadata((None, None), (None, None)))
    assert plan.status == "needs_clarification" and plan.scope.matching_study_count is None
    question = " ".join(c.question for c in plan.clarifications)
    assert "unknown for 2 of 2" in question and "does not mean zero" in question


def test_missing_program_guard_is_not_specific_to_ncsu():
    plan = assemble("Which trials belong to the OTHER breeding programme?", proposed(value="OTHER"), metadata((None, None)))
    assert plan.status == "needs_clarification" and plan.scope.matching_study_count is None


def test_partial_missing_programs_are_qualified_not_claimed_complete():
    plan = assemble("Which studies belong to NCSU program?", proposed(column="programName"), metadata(("P1", "NCSU"), (None, None)))
    question = " ".join(c.question for c in plan.clarifications)
    assert plan.status == "needs_clarification" and plan.scope.matching_study_count is None
    assert "unknown for 1 of 2" in question and "1 have the needed" in question


def test_name_substitution_is_refused_even_when_program_metadata_exists():
    plan = assemble("Which studies belong to NCSU program?", proposed(), metadata(("P1", "NCSU")))
    assert plan.status == "needs_clarification"
    assert any("searching study names does not establish" in c.question for c in plan.clarifications)


def test_known_exact_program_ids_can_have_legitimate_zero_matches():
    plan = assemble("Which studies belong to program P9?", proposed(column="programDbId", value="P9", filters={"program_id": "P9"}),
                    metadata(("P1", "NCSU"), ("P2", "OTHER")))
    assert plan.status == "ready" and plan.scope.study_ids == [] and plan.scope.matching_study_count == 0
    assert co._study_matches(metadata(("P1", "NCSU")).studies[0], co.DraftScope(filters={"program_id": "P1"}))


def test_an_explicit_literal_study_name_search_is_allowed():
    question = "List study names containing the word program."
    plan = assemble(question, proposed(value="program", filters={"name_contains": "program"}), metadata((None, None)))
    assert plan.status == "ready" and plan.scope.matching_study_count == 0


def test_yes_clarification_cannot_supply_missing_program_metadata():
    question = co.add_clarification("Which studies belong to NCSU program?", "yes")
    plan = assemble(question, proposed(), metadata((None, None)))
    assert plan.status == "needs_clarification" and plan.scope.matching_study_count is None


@pytest.fixture
def registry(tmp_path):
    return ArtifactRegistry(tmp_path / "out", "run_missing_metadata", base_url=SYNTHETIC_BASE)


def catalog(registry, values, column="programName", *, complete=True):
    return registry.save_table([{"studyDbId": f"S{i}", column: value} for i, value in enumerate(values, 1)],
                               kind="studies", columns={"studyDbId": "SYNTHETIC ID", column: "SYNTHETIC metadata"},
                               complete=complete).artifact_id


@pytest.mark.parametrize("column", ["programName", "locationName", "studyType"])
@pytest.mark.parametrize("op", ["eq", "ne", "contains"])
def test_all_missing_catalog_metadata_refuses_a_matching_count(registry, column, op):
    artifact = catalog(registry, [None, "", "   "], column)
    before = registry.artifact_ids()
    result = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": artifact, "column": column, "op": op, "value": "NCSU"})
    assert not result.ok and result.error.code == "incomplete_data"
    assert "unknown, not zero" in result.error.message and registry.artifact_ids() == before


def test_partial_unknown_membership_cannot_become_an_exact_total(registry):
    artifact = catalog(registry, ["NCSU", "OTHER", None])
    filtered = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": artifact, "column": "programName", "op": "eq", "value": "NCSU"})
    assert filtered.ok and filtered.complete is False and filtered.data["rows_out"] == 1
    assert filtered.data["rows_with_unknown_filter_metadata"] == 1
    assert any("1 known matches" in w and "full matching count is unknown" in w for w in filtered.warnings)
    counted = at.dispatch_analyst(registry, "count_records", {"artifact_id": filtered.artifact_ids[0]})
    assert not counted.ok and counted.error.code == "incomplete_data"


def test_negative_filter_does_not_classify_unknown_as_known_nonmatch(registry):
    artifact = catalog(registry, ["NCSU", "OTHER", None])
    result = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": artifact, "column": "programName", "op": "ne", "value": "NCSU"})
    assert result.ok and result.complete is False and result.data["rows_out"] == 1
    _, rows = registry.load(result.artifact_ids[0])
    assert [row["programName"] for row in rows] == ["OTHER"]


def test_known_nonmatching_catalog_allows_zero(registry):
    artifact = catalog(registry, ["OTHER", "ANOTHER"])
    filtered = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": artifact, "column": "programName", "op": "eq", "value": "NCSU"})
    assert filtered.ok and filtered.complete and filtered.data["rows_out"] == 0
    counted = at.dispatch_analyst(registry, "count_records", {"artifact_id": filtered.artifact_ids[0]})
    assert counted.ok and counted.data["claims"][0]["value"] == 0


def test_complete_empty_catalog_allows_zero(registry):
    artifact = catalog(registry, [])
    filtered = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": artifact, "column": "programName", "op": "eq", "value": "NCSU"})
    assert filtered.ok and filtered.complete and filtered.data["rows_out"] == 0


def test_explicit_missingness_question_stays_answerable(registry):
    artifact = catalog(registry, ["", "NCSU"])
    filtered = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": artifact, "column": "programName", "op": "is_missing"})
    assert filtered.ok and filtered.complete and filtered.data["rows_out"] == 1
    counted = at.dispatch_analyst(registry, "count_records", {"artifact_id": filtered.artifact_ids[0]})
    assert counted.ok and counted.data["claims"][0]["value"] == 1


def test_filter_cannot_upgrade_an_incomplete_catalog(registry):
    artifact = catalog(registry, ["NCSU"], complete=False)
    filtered = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": artifact, "column": "programName", "op": "eq", "value": "NCSU"})
    assert filtered.ok and filtered.complete is False


def test_named_study_does_not_require_unrelated_program_metadata():
    known = metadata(("P1", "NCSU"), (None, None))
    draft = proposed(column="programName").model_copy(update={"scope": co.DraftScope(study_ids=["S1"])})
    plan = assemble("Which program does study S1 belong to?", draft, known)
    assert plan.status == "ready" and plan.scope.study_ids == ["S1"] and plan.scope.matching_study_count == 1


def test_person_can_explicitly_replace_program_scope_with_a_recorded_field():
    known = metadata((None, None)).model_copy(update={"studies": [
        co.KnownStudy(study_id="S1", name="SYNTHETIC trial", seasons=["2026"])], "season_ids": ["2026"]})
    question = co.add_clarification("Which studies belong to NCSU program?", "Use season 2026 instead")
    plan = assemble(question, proposed(column="seasons", value="2026", filters={"season_id": "2026"}), known)
    assert plan.status == "ready" and plan.scope.matching_study_count == 1


@pytest.mark.parametrize("existing", [
    co.DraftClarification(question="Program membership was already discussed?", required=False),
    # Adversarial typed input: even an existing answered contract cannot discharge the code guard.
    co.Clarification(question="Program membership was already discussed?", required=True, answer="NCSU"),
])
def test_model_clarification_cannot_discharge_missing_metadata_guard(existing):
    draft = proposed().model_copy(update={"clarifications": [existing]})
    plan = assemble("Which studies belong to NCSU program?", draft, metadata((None, None)))
    assert plan.status == "needs_clarification" and plan.scope.matching_study_count is None
    questions = [c for c in plan.clarifications if "Program membership" in c.question]
    assert len(questions) == 1 and "unknown for 1 of 1" in questions[0].question
    assert questions[0].required is True and questions[0].answer is None


def test_partial_location_catalog_counts_recorded_matches_and_keeps_warning(registry):
    artifact = catalog(registry, ["L1", "L2", None], "locationDbId")
    filtered = at.dispatch_analyst(registry, "filter_rows", {"artifact_id": artifact, "column": "locationDbId", "op": "eq", "value": "L1"})
    assert filtered.ok and filtered.complete and filtered.data["rows_out"] == 1
    assert any("Counts describe recorded matches only" in warning for warning in filtered.warnings)
    counted = at.dispatch_analyst(registry, "count_records", {"artifact_id": filtered.artifact_ids[0]})
    assert counted.ok and counted.data["claims"][0]["value"] == 1
