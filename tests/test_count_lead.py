"""A count answer leads with its counts (plan v3, lead_with_count_plan_20261020; coach package, 2026-10-20).

q14 ("how many valid values does ... have?") passed on both models on 2026-10-19, but on Luna the 140 was the last row of
a six-row table (mean first) and on Opus it sat inside a "Traits: 1 groups" table. When the plan's own statistic is a
count, the answer now opens with "**Counts**": every count, number first, with exactly what it counted, in a fixed order.
Nothing is picked or hidden; the blocks below are unchanged."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import analyst_tools as at
import chat_presenter as cp
from artifacts import ArtifactRegistry
from contracts import AnalysisReport, Claim, Denominator
from part2_support import SYNTHETIC_BASE
from test_analyst_tools import call, register, s1  # noqa: F401
from test_count_labels import counted, filtered, studies

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "count_lead"
STUDIES_7 = [{"studyDbId": f"S{i}", "studyName": f"T{i}", "seasons": '["2026"]', "locationDbId": location}
             for i, location in enumerate(["L1", "L1", "L1", "L1", "L2", "L2", ""], 1)]


@pytest.fixture
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "out", "run_lead", base_url=SYNTHETIC_BASE)


def view(claims, *, kind="count", registry=None, question="How many valid values are there?"):
    plan = SimpleNamespace(statistic=None if kind is None else SimpleNamespace(kind=kind), question=question)
    return SimpleNamespace(execution_status="completed", ctx=SimpleNamespace(registry=registry), notes=[], plan=plan,
                           analysis=AnalysisReport(status="completed", claims=list(claims)),
                           retrieval=SimpleNamespace(status="completed", warnings=[]))


def lead(content: str) -> list[str]:
    """The lines of the Counts section, or [] when the answer has none."""
    lines = content.splitlines()
    if not lines or lines[0] != "**Counts**":
        return []
    return [line for line in lines[2:lines.index("", 2)]]


def fixture_claims(name):
    doc = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    assert doc["statistic"]["kind"] == "count"
    return [Claim.model_validate_json(json.dumps(c)) for c in doc["claims"]], doc["question"]


# --------------------------------------------------------------------------
# q14 as the two models answered it on 2026-10-19 (the words their claims store)
# --------------------------------------------------------------------------

def test_lunas_q14_leads_with_both_counts_valid_values_first():
    claims, question = fixture_claims("q14_luna")
    content = cp.answer_summary(view(claims, question=question))
    assert lead(content) == ["- **140** " + cp._text("valid numeric values, one per observation unit"),
                             "- **140** " + cp._text("rows matching value is_valid_number '' in the complete observations table")]
    assert "| Mean |" in content and "| Count | 140 |" in content                 # the blocks below are unchanged


def test_opus_q14_frequency_of_one_group_gets_a_line():
    """The stored claim, drawn without its table (the registry of that run is not in the repository)."""
    claims, question = fixture_claims("q14_opus")
    content = cp.answer_summary(view(claims, question=question))
    assert lead(content) == ["- **140** " + cp._text("rows with observationVariableDbId 76736, of 140 rows matching value "
                                                     "is_valid_number '' in the complete observations table")]


def test_a_one_group_frequency_table_gets_its_line_above_the_unchanged_table(registry):
    """Opus's q14 shape with its table (review7): today the answer opens "**Traits: 1 groups**" with 140 in the table."""
    rows = [{"observationDbId": f"O{i}", "observationVariableDbId": "76736",
             "observationVariableName": "Reaction to Fusarium Wilt estimating 0-5", "value": "2"} for i in range(140)]
    source = registry.save_table(rows, kind="observations", complete=True,
                                 columns={"observationDbId": "id", "observationVariableDbId": "id",
                                          "observationVariableName": "text", "value": "number"}).artifact_id
    claims = counted(registry, filtered(registry, source, "value", "is_valid_number", ""), group_by="observationVariableDbId")
    content = cp.answer_summary(view(claims, registry=registry))
    assert lead(content) == ["- **140** " + cp._text("rows with observationVariableDbId 76736, of 140 rows matching value "
                                                     "is a valid number in the complete observations table")]
    assert content.splitlines()[4] == "**Traits: 1 groups**" and "| 76736 |" in content     # the table below, as today


# --------------------------------------------------------------------------
# Today's tools on a real table (the synthetic S1 study: 6 rows, 4 valid values)
# --------------------------------------------------------------------------

def q14_claims(registry, obs, *, rows_first: bool):
    kept = filtered(registry, obs, "value", "is_valid_number", "")
    numeric = [Claim.model_validate(c) for c in call(registry, "numeric_summary", artifact_id=obs).data["claims"]]
    rows = counted(registry, kept) + counted(registry, obs)
    return rows + numeric if rows_first else numeric + rows


@pytest.mark.parametrize("rows_first", [False, True])
def test_the_order_is_fixed_whatever_order_the_model_called_its_tools(s1, rows_first):
    registry, obs, _, _ = s1
    content = cp.answer_summary(view(q14_claims(registry, obs, rows_first=rows_first), registry=registry))
    lines = lead(content)
    assert [line.rsplit(" · source ", 1)[0] for line in lines] == [
        "- **4** " + cp._text("valid numeric values, one per observation unit"),
        "- **4** " + cp._text("rows matching value is a valid number in the complete observations table"),
        "- **6** " + cp._text("rows in the complete observations table")]
    # two tables (the 6 rows and the 4 kept): each line names its source, as its block below does
    assert lines[0][-10:] == lines[2][-10:] != lines[1][-10:] and all(" · source " in line for line in lines)


def test_values_averaged_per_unit_lead_as_units(registry):
    art = register(registry, "duplicate_plots.json", "repeated_measurements")
    claims = [Claim.model_validate(c) for c in call(registry, "numeric_summary", artifact_id=art,
                                                     duplicate_policy="mean_per_unit").data["claims"]]
    assert lead(cp.answer_summary(view(claims))) == ["- **4** " + cp._text("observation units (valid values averaged per unit first)")]


# --------------------------------------------------------------------------
# Only for a count; no repeat of a count a block already leads with
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["median", "mean", None])
def test_no_section_unless_the_plans_statistic_is_a_count(s1, kind):
    registry, obs, _, _ = s1
    content = cp.answer_summary(view(q14_claims(registry, obs, rows_first=False), kind=kind, registry=registry,
                                     question="What is the median value?"))
    assert lead(content) == [] and not content.startswith("**Counts**")


def test_a_distinct_count_in_a_catalog_title_is_not_repeated_but_one_drawn_as_a_statistic_is(registry):
    source = registry.save_table(STUDIES_7, kind="studies", complete=True,
                                 columns={"studyDbId": "id", "studyName": "text", "seasons": "text", "locationDbId": "id"}).artifact_id
    claims = counted(registry, source, distinct_by="locationDbId")
    in_title = cp.answer_summary(view(claims, registry=registry, question="How many locations are there?"))
    assert "distinct IDs" in in_title.splitlines()[0] and lead(in_title) == []
    as_statistic = cp.answer_summary(view(claims, registry=None, question="How many locations are there?"))   # no table to draw
    assert lead(as_statistic) == ["- **2** " + cp._text("distinct locationDbId values, in 7 rows in the complete studies table")]


@pytest.mark.parametrize("question", ["How many studies have a 2026 season?", "Which studies have a 2026 season?"])
def test_a_one_match_catalog_count_is_not_repeated_above_its_title(registry, question):
    """Review 7: the catalog route counts group_by and distinct_by studyDbId on the filtered table. With one match, the
    title "Studies: 1 distinct IDs" leads, and the one-row frequency must not add "1 rows with studyDbId S1" above it."""
    source = registry.save_table(STUDIES_7[:1], kind="studies", complete=True,
                                 columns={"studyDbId": "id", "studyName": "text", "seasons": "text", "locationDbId": "id"}).artifact_id
    claims = counted(registry, source, group_by="studyDbId", distinct_by="studyDbId")
    content = cp.answer_summary(view(claims, registry=registry, question=question))
    assert lead(content) == [] and content.splitlines()[0] == "**Studies: 1 distinct IDs**"


def test_a_count_from_an_incomplete_table_gets_no_line():
    claim = Claim(claim_id="clm_partial", kind="count", value=12, source_artifact_hashes=["b" * 64],
                  denominator=Denominator(name="valid numeric values, one per observation unit", n=12),
                  transformation="count of valid values; INCOMPLETE ARTIFACT: describes the rows fetched, not the study; "
                                 "valid values used as recorded (one row per observation unit)")
    assert lead(cp.answer_summary(view([claim]))) == []


def test_a_frequency_table_of_several_groups_adds_no_line_and_one_group_adds_one(registry):
    source = registry.save_table(STUDIES_7, kind="studies", complete=True,
                                 columns={"studyDbId": "id", "studyName": "text", "seasons": "text", "locationDbId": "id"}).artifact_id
    several = counted(registry, source, group_by="locationDbId")                  # L1: 4, L2: 2, empty: 1
    assert len(several) == 3 and lead(cp.answer_summary(view(several, registry=registry, question="Which location has the most studies?"))) == []
    one = counted(registry, filtered(registry, source, "locationDbId", "eq", "L2"), group_by="locationDbId")
    assert lead(cp.answer_summary(view(one, registry=registry))) == [
        "- **2** " + cp._text("rows with locationDbId L2, of 2 rows matching locationDbId eq 'L2' in the complete studies table")]


def test_the_empty_group_reads_no_column(registry):
    source = registry.save_table([{"studyDbId": "S1", "locationDbId": ""}], kind="studies", complete=True,
                                 columns={"studyDbId": "id", "locationDbId": "id"}).artifact_id
    [claim] = counted(registry, source, group_by="locationDbId")
    assert claim.filters == {"locationDbId": at.MISSING_GROUP}
    assert lead(cp.answer_summary(view([claim], registry=None))) == [
        "- **1** " + cp._text("rows with no locationDbId, of 1 rows in the complete studies table")]


# --------------------------------------------------------------------------
# Two sources, a count with no value, an answer without counts
# --------------------------------------------------------------------------

def test_two_source_tables_each_line_names_its_source(registry):
    first = studies(registry)
    second = registry.save_table(STUDIES_7, kind="studies", complete=True,
                                 columns={"studyDbId": "id", "studyName": "text", "seasons": "text", "locationDbId": "id"}).artifact_id
    claims = counted(registry, first) + counted(registry, second)
    assert lead(cp.answer_summary(view(claims, registry=registry))) == [
        "- **3** " + cp._text("rows in the complete studies table") + " · source 1",
        "- **7** " + cp._text("rows in the complete studies table") + " · source 2"]


def test_a_count_with_no_value_says_why():
    claim = Claim(claim_id="clm_none", kind="count", value=None, missing_reason="the table could not be read",
                  denominator=Denominator(name="rows in the complete studies table", n=0), source_artifact_hashes=["a" * 64],
                  transformation=cp.ROW_COUNT)
    assert lead(cp.answer_summary(view([claim]))) == [
        "- **Not available**: " + cp._text("the table could not be read (rows in the complete studies table)")]


def test_no_heading_when_no_count_claim_remains(s1):
    registry, obs, _, _ = s1
    means = [Claim.model_validate(c) for c in call(registry, "group_stats", artifact_id=obs, by="germplasmDbId",
                                                   min_independent_n=1).data["claims"]]
    assert means and all(c.kind == "mean" for c in means)
    assert lead(cp.answer_summary(view(means, registry=registry))) == []
