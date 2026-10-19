"""Every count says exactly what it counted (coach 2026-10-17, count_layout_plan RECOMMENDATION item 2; open item 6).

Before: count_records called its count "rows in the complete studies table" also after a filter cut the table down ("How
many studies have a 2026 season?" -> "Count: 22 (rows in the complete studies table)"; zero matches read as an empty
catalog); with each unit's values averaged first, the "valid values" count was a count of units; a distinct-ID count drawn
as an ordinary statistic was labelled "Count" beside the table's row count. One test per claim shape."""
from types import SimpleNamespace

import pytest

import analyst_tools as at
import chat_presenter as cp
from artifacts import ArtifactRegistry
from contracts import AnalysisReport, Claim
from part2_support import SYNTHETIC_BASE
from test_analyst_tools import call, claims_of, register

STUDIES = [{"studyDbId": "S1", "studyName": "A", "seasons": '["2026"]', "locationDbId": "L1"},
           {"studyDbId": "S2", "studyName": "B", "seasons": '["2025", "2026"]', "locationDbId": "L2"},
           {"studyDbId": "S3", "studyName": "C", "seasons": '["2024"]', "locationDbId": "L1"}]


@pytest.fixture
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "out", "run_labels", base_url=SYNTHETIC_BASE)


def studies(registry) -> str:
    return registry.save_table(STUDIES, kind="studies", complete=True,
                               columns={"studyDbId": "id", "studyName": "text", "seasons": "text", "locationDbId": "id"}).artifact_id


def filtered(registry, artifact_id, column, op, value) -> str:
    result = call(registry, "filter_rows", artifact_id=artifact_id, column=column, op=op, value=value)
    assert result.ok
    return result.data["artifact_id"]


def counted(registry, artifact_id, **options) -> list[Claim]:
    result = call(registry, "count_records", artifact_id=artifact_id, **options)
    assert result.ok
    return [Claim.model_validate(c) for c in result.data["claims"]]


def test_a_row_count_of_a_source_table_is_labelled_as_before(registry):
    [claim] = counted(registry, studies(registry))
    assert (claim.value, claim.denominator.name, claim.denominator.n) == (3, "rows in the complete studies table", 3)
    assert claim.transformation == cp.ROW_COUNT                              # the method text the chat matches is unchanged


def test_a_row_count_after_a_filter_names_the_filter(registry):
    [claim] = counted(registry, filtered(registry, studies(registry), "seasons", "contains", "2026"))
    assert claim.value == 2 and claim.denominator.n == 2
    assert claim.denominator.name == "rows matching seasons contains '2026' in the complete studies table"


def test_zero_matches_read_as_zero_matches_not_an_empty_catalog(registry):
    [claim] = counted(registry, filtered(registry, studies(registry), "seasons", "contains", "1999"))
    assert claim.value == 0 and claim.denominator.name == "rows matching seasons contains '1999' in the complete studies table"


def test_the_catalog_season_filter_shows_its_quoted_value_once(registry):
    """Replay of run_20261019T140336_71b191 (Opus 5.5): catalog_answers filters on the quoted JSON text."""
    [claim] = counted(registry, filtered(registry, studies(registry), "seasons", "contains", '"2026"'))
    assert claim.value == 2 and claim.denominator.name == 'rows matching seasons contains "2026" in the complete studies table'


def test_the_tests_that_take_no_value_are_named_in_words(registry):
    """Eval 20261019T170352Z (Opus, q14): "rows matching value is_valid_number '' in the complete observations table"."""
    art = register(registry, "duplicate_plots.json", "repeated_measurements")
    [valid] = counted(registry, filtered(registry, art, "value", "is_valid_number", ""))
    [missing] = counted(registry, filtered(registry, art, "value", "is_missing", ""))
    assert valid.denominator.name == "rows matching value is a valid number in the complete observations table"
    assert missing.denominator.name == "rows matching value is missing in the complete observations table"


def test_a_filter_on_a_filtered_table_names_both_oldest_first(registry):
    first = filtered(registry, studies(registry), "seasons", "contains", "2026")
    [claim] = counted(registry, filtered(registry, first, "locationDbId", "eq", "L1"))
    assert claim.value == 1
    assert claim.denominator.name == ("rows matching seasons contains '2026' and locationDbId eq 'L1' in the complete "
                                      "studies table")


def test_frequency_and_distinct_counts_share_the_row_label_so_a_single_total_still_leads(registry):
    source = filtered(registry, studies(registry), "seasons", "contains", "2026")
    claims = counted(registry, source, group_by="studyDbId", distinct_by="studyDbId")
    assert {c.denominator.name for c in claims} == {"rows matching seasons contains '2026' in the complete studies table"}
    distinct = next(c for c in claims if c.transformation == cp.DISTINCT)
    assert distinct.value == 2 and distinct.grouping == ["studyDbId"]
    plan = SimpleNamespace(statistic=SimpleNamespace(kind="count"), question="How many studies have a 2026 season?")
    assert cp._count_focus(SimpleNamespace(plan=plan), claims)               # the labels still agree, as the chat requires


def test_a_distinct_count_drawn_as_a_statistic_is_not_called_a_count_of_rows(registry):
    source = studies(registry)
    claims = counted(registry, source, distinct_by="locationDbId") + counted(registry, source)
    lines = "\n".join(cp._numeric_blocks(claims, cp._Sources(registry), {}))
    assert "| Distinct locationDbId values | 2 |" in lines and "| Rows | 3 |" in lines and "| Count |" not in lines
    assert "Based on rows in the complete studies table: 3." in lines


def test_a_summary_of_values_as_recorded_counts_values(registry):
    art = register(registry, "duplicate_plots.json", "repeated_measurements")
    _, rows = registry.load(art)
    once = registry.save_table([r for i, r in enumerate(rows) if i != 1], kind="observations", columns=dict(registry.manifest(art).columns),
                               complete=True, measurement=registry.manifest(art).measurement).artifact_id
    r = call(registry, "numeric_summary", artifact_id=once)
    assert r.ok and claims_of(r)["count"].denominator.name == "valid numeric values, one per observation unit"


def test_a_summary_with_values_averaged_per_unit_counts_units(registry):
    art = register(registry, "duplicate_plots.json", "repeated_measurements")
    r = call(registry, "numeric_summary", artifact_id=art, duplicate_policy="mean_per_unit")
    count = claims_of(r)["count"]
    assert (count.value, count.denominator.n) == (4, 4) and r.data["counts"]["valid"] == 5      # 5 values, 4 units
    assert count.denominator.name == "observation units (valid values averaged per unit first)"
    g = call(registry, "group_stats", artifact_id=art, by="germplasmDbId", duplicate_policy="mean_per_unit")
    names = {Claim.model_validate(c).filters["germplasmDbId"]: Claim.model_validate(c).denominator.name for c in g.data["claims"]}
    assert names == {"GA": "observation units in group GA (valid values averaged per unit first)",
                     "GB": "observation units in group GB (valid values averaged per unit first)"}


def test_a_summary_of_a_filtered_table_names_the_filter(registry):
    art = register(registry, "duplicate_plots.json", "repeated_measurements")
    kept = filtered(registry, art, "germplasmDbId", "eq", "GB")
    r = call(registry, "numeric_summary", artifact_id=kept)
    assert r.ok and claims_of(r)["count"].denominator.name == ("valid numeric values, one per observation unit, from rows "
                                                               "matching germplasmDbId eq 'GB'")


def test_the_full_report_and_the_chat_show_the_exact_label(registry):
    source = filtered(registry, studies(registry), "seasons", "contains", "2026")
    claims = counted(registry, source, distinct_by="studyDbId")
    view = SimpleNamespace(execution_status="completed", ctx=SimpleNamespace(registry=registry), notes=[],
                           analysis=AnalysisReport(status="completed", claims=claims),
                           retrieval=SimpleNamespace(status="completed", warnings=[]))
    shown = cp.answer_summary(view)
    assert "Based on rows matching seasons contains '2026' in the complete studies table: 2." in shown
