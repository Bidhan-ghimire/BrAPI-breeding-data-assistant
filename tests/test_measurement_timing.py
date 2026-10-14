"""Timing caveats describe saved evidence; zero values remain valid measurements."""
import pytest

from analyst_tools import dispatch_analyst
from artifacts import ArtifactRegistry
from contracts import MeasurementMeta


def sample(tmp_path, rows, *, timepoint=None, kind="observations"):
    registry = ArtifactRegistry(tmp_path / "out", "run_timing", base_url="https://synthetic.invalid/brapi/v2")
    records = [dict(observationDbId=f"O{i}", observationUnitDbId=f"U{i}",
                    germplasmDbId="G1", observationVariableDbId="V1", value=value,
                    observationTimeStamp=stamp) for i, (value, stamp) in enumerate(rows)]
    artifact = registry.save_table(records, kind=kind, columns={k: "synthetic" for k in records[0]},
                                   complete=True, variable_ids=["V1"],
                                   measurement=MeasurementMeta(unit="g", timepoint=timepoint))
    return registry, artifact.artifact_id


@pytest.mark.parametrize("tool,extra", [("numeric_summary", {}), ("group_stats", {"by": "germplasmDbId"})])
def test_missing_dates_and_stage_are_disclosed_without_changing_numbers(tmp_path, tool, extra):
    registry, aid = sample(tmp_path, [("0", ""), ("2", "")])
    result = dispatch_analyst(registry, tool, {"artifact_id": aid, **extra})
    assert result.ok
    assert any("2 of 2 valid source rows" in warning for warning in result.warnings)
    assert any("shared growth stage cannot be verified" in warning for warning in result.warnings)
    mean = next(c for c in result.data["claims"] if c["kind"] == "mean")
    assert mean["value"] == 1 and mean["denominator"]["n"] == 2


def test_missing_dates_count_only_valid_source_values(tmp_path):
    registry, aid = sample(tmp_path, [("0", ""), ("2", "2026-01-02"), ("", ""), ("bad", "")], timepoint="30 days")
    result = dispatch_analyst(registry, "numeric_summary", {"artifact_id": aid})
    assert result.ok
    assert any("1 of 2 valid source rows" in warning for warning in result.warnings)
    assert not any("growth stage" in warning for warning in result.warnings)


def test_recorded_dates_and_stage_do_not_get_missing_timing_warning(tmp_path):
    registry, aid = sample(tmp_path, [("0", "2026-01-02"), ("2", "2026-01-03")], timepoint="30 days")
    result = dispatch_analyst(registry, "numeric_summary", {"artifact_id": aid})
    assert result.ok
    assert not any("dates are not recorded" in warning or "growth stage" in warning for warning in result.warnings)


def test_no_valid_measurements_does_not_invent_timing_denominator(tmp_path):
    registry, aid = sample(tmp_path, [("", ""), ("bad", "")])
    result = dispatch_analyst(registry, "numeric_summary", {"artifact_id": aid})
    assert result.ok
    assert not any("valid source rows" in warning for warning in result.warnings)
