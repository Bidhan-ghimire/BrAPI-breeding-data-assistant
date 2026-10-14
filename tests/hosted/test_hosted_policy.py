"""Checks for the moved access gate and the supported hosted execution mode."""
from dataclasses import replace

import pytest

import access_policy
from synthetic_support import hosted_config


def test_extracted_access_review_reader_preserves_existing_contract(tmp_path):
    notes = tmp_path / "ACCESS_NOTES.md"
    assert access_policy.access_notes_review_date(notes) is None
    notes.write_text("# Policy\nNo recorded operator review.\n", encoding="utf-8")
    assert access_policy.access_notes_review_date(notes) is None
    notes.write_text("# Policy\nReviewed by Bidhan on: 2026-09-28\n", encoding="utf-8")
    assert access_policy.access_notes_review_date(notes) == "2026-09-28"


def test_missing_access_review_still_blocks_hosted_configuration(tmp_path):
    config = hosted_config(tmp_path, "Find study S1", access_notes=tmp_path / "missing-policy.md")
    assert any("review" in problem.lower() for problem in config.validate())


@pytest.mark.parametrize("change", [
    {"offline": True}, {"mock_model": True}, {"direct": True}, {"auto": True},
    {"mcp_transport": "stdio"}, {"unified_mcp": False}, {"review_retrieved_data": False},
])
def test_retired_execution_modes_fail_closed(tmp_path, change):
    config = hosted_config(tmp_path, "Find study S1", **change)
    assert config.validate(), "A retired or unreviewed path must be explicitly refused."


def test_hosted_configuration_has_no_snapshot_or_fixture_fields(tmp_path):
    config = hosted_config(tmp_path, "Find study S1")
    assert config.validate() == []
    for name in ("fixture", "snapshot", "snapshots_dir"):
        assert not hasattr(config, name)
