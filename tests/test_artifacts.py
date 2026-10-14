"""
tests/test_artifacts.py — artifacts are reached only through handles issued by this
registry; files are text-only, hashed, never overwritten, and never resolved from
paths a caller (or a model) typed.

All data is SYNTHETIC; every file is written under pytest's temporary folder.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

import artifacts as art
from artifacts import ArtifactError, ArtifactRegistry
from contracts import CollectionResult, MeasurementMeta

BASE = "https://synthetic.invalid/brapi/v2"
OTHER = "https://test-server.brapi.org/brapi/v2"

SIX_ROWS = [
    {"observationDbId": "O1", "observationUnitDbId": "P1", "germplasmName": "clone A", "value": "10"},
    {"observationDbId": "O2", "observationUnitDbId": "P2", "germplasmName": "clone A", "value": "14"},
    {"observationDbId": "O3", "observationUnitDbId": "P3", "germplasmName": "clone A", "value": ""},
    {"observationDbId": "O4", "observationUnitDbId": "P4", "germplasmName": "clone B", "value": "20"},
    {"observationDbId": "O5", "observationUnitDbId": "P5", "germplasmName": "clone B", "value": "bad"},
    {"observationDbId": "O6", "observationUnitDbId": "P6", "germplasmName": "clone B", "value": "28"},
]
COLUMNS = {"observationDbId": "observation ID (text)", "observationUnitDbId": "plot ID (text)",
           "germplasmName": "clone label", "value": "raw value token, unconverted"}


def collection(records, **over):
    base = dict(records=records, returned_count=len(records), reported_total=len(records), complete=True,
                status="complete" if records else "empty", request_ids=["req_run_a_0001"])
    base.update(over)
    return CollectionResult(**base)


def registry(tmp_path, run_id="run_a", base=BASE):
    return ArtifactRegistry(tmp_path / "out", run_id, base_url=base)


def save_six(reg, **over):
    args = dict(kind="observations", columns=COLUMNS, collection=collection(SIX_ROWS), study_ids=["S1"],
                variable_ids=["V1"], measurement=MeasurementMeta(trait="fresh root yield", unit="t/ha"))
    args.update(over)
    return reg.save_table(SIX_ROWS, **args)


# --- 1. saving: generated names, opaque handles, text-only cells ---------------------------

def test_save_table_registers_a_handle_and_writes_text_only_csv(tmp_path):
    reg = registry(tmp_path)
    manifest = save_six(reg)
    assert manifest.artifact_id.startswith("art_") and manifest.run_id == "run_a"
    assert manifest.relative_path == "run_a/artifacts/0001_observations.csv"      # generated, not "S1"
    assert "S1" not in manifest.relative_path
    assert manifest.row_count == 6 and manifest.complete and manifest.kind == "observations"
    assert manifest.columns == COLUMNS and manifest.source_request_ids == ["req_run_a_0001"]
    assert manifest.measurement.unit == "t/ha" and manifest.study_ids == ["S1"]
    path = tmp_path / "out" / manifest.relative_path
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "observationDbId,observationUnitDbId,germplasmName,value"
    assert text.splitlines()[3] == "O3,P3,clone A,"                                  # '' stays empty
    assert text.splitlines()[5] == "O5,P5,clone B,bad"                               # 'bad' stays 'bad'
    assert path.with_suffix(".manifest.json").is_file()
    assert (tmp_path / "out" / "run_a" / "registry.json").is_file()


def test_load_returns_every_cell_as_text_with_tokens_preserved(tmp_path):
    reg = registry(tmp_path)
    rows = [{"observationDbId": "007", "observationUnitDbId": "P1", "germplasmName": "x", "value": "10.50"},
            {"observationDbId": "008", "observationUnitDbId": "P2", "germplasmName": "y", "value": "1e3"}]
    manifest = reg.save_table(rows, kind="observations", columns=COLUMNS, complete=True)
    loaded_manifest, loaded = reg.load(manifest.artifact_id)
    assert loaded_manifest == manifest
    assert [r["observationDbId"] for r in loaded] == ["007", "008"]                 # leading zero kept
    assert [r["value"] for r in loaded] == ["10.50", "1e3"]                          # no numeric conversion
    assert all(isinstance(v, str) for r in loaded for v in r.values())


def test_nonfinite_and_nested_values_become_null_or_json_with_a_recorded_reason(tmp_path):
    reg = registry(tmp_path)
    rows = [{"id": "1", "mean": float("nan"), "extra": {"a": 1}, "flag": True, "n": 4},
            {"id": "2", "mean": float("inf"), "extra": [1, 2], "flag": False, "n": None}]
    cols = {"id": "id", "mean": "mean", "extra": "nested", "flag": "bool", "n": "count"}
    manifest = reg.save_table(rows, kind="analysis_result", columns=cols, complete=True)
    _, loaded = reg.load(manifest.artifact_id)
    assert loaded[0]["mean"] == "" and loaded[1]["mean"] == ""                      # null cells
    assert any("row 0 column 'mean'" in n and "nan" in n for n in manifest.quality_notes)
    assert any("row 1 column 'mean'" in n and "inf" in n for n in manifest.quality_notes)
    assert loaded[0]["extra"] == '{"a": 1}' and loaded[1]["extra"] == "[1, 2]"
    assert loaded[0]["flag"] == "true" and loaded[0]["n"] == "4" and loaded[1]["n"] == ""
    assert art.NULL_POLICY in manifest.quality_notes


def test_fields_outside_the_declared_columns_are_reported_not_silently_lost(tmp_path):
    reg = registry(tmp_path)
    manifest = reg.save_table([{"id": "1", "surprise": "x"}], kind="studies", columns={"id": "id"}, complete=True)
    assert any("surprise" in n for n in manifest.quality_notes)


# --- 2. empty tables and the study ledger -----------------------------------------------------

def test_empty_table_still_has_columns_and_a_manifest_and_the_study_stays_on_the_ledger(tmp_path):
    reg = registry(tmp_path)
    manifest = reg.save_table([], kind="observations", columns=COLUMNS, collection=collection([]), study_ids=["S2"])
    assert manifest.row_count == 0 and manifest.complete and list(manifest.columns) == list(COLUMNS)
    _, rows = reg.load(manifest.artifact_id)
    assert rows == []
    preview = reg.preview(manifest.artifact_id)
    assert preview.total_rows == 0 and preview.displayed_rows == 0 and preview.columns == list(COLUMNS)
    reg.record_study("S2", "empty", artifact_id=manifest.artifact_id)
    reg.record_study("S3", "failed", reason="timeout after 1 attempt")
    reg.record_study("S4", "blocked", reason="not in the FetchApproval")
    ledger = reg.ledger()
    assert [(e.study_id, e.status) for e in ledger] == [("S2", "empty"), ("S3", "failed"), ("S4", "blocked")]
    with pytest.raises(ArtifactError):
        reg.record_study("S5", "complete", artifact_id="art_000000_9999")           # unknown artifact
    with pytest.raises(ValidationError):
        reg.record_study("S6", "failed")                                              # a failure needs a reason


def test_table_without_columns_or_without_a_completeness_statement_is_refused(tmp_path):
    reg = registry(tmp_path)
    with pytest.raises(ArtifactError):
        reg.save_table([], kind="observations", columns={}, complete=True)
    with pytest.raises(ArtifactError):
        reg.save_table(SIX_ROWS, kind="observations", columns=COLUMNS)                # neither collection nor complete
    with pytest.raises(ArtifactError):
        reg.save_table(SIX_ROWS[:3], kind="observations", columns=COLUMNS, collection=collection(SIX_ROWS))


# --- 3. previews -------------------------------------------------------------------------------

def test_preview_shows_a_few_rows_but_reports_the_true_total(tmp_path):
    reg = registry(tmp_path)
    manifest = save_six(reg)
    preview = reg.preview(manifest.artifact_id, max_rows=2)
    assert preview.total_rows == 6 and preview.displayed_rows == 2 and len(preview.rows) == 2
    assert preview.rows[0] == SIX_ROWS[0] and preview.complete and preview.artifact_id == manifest.artifact_id
    big = reg.preview(manifest.artifact_id, max_rows=50)
    assert big.displayed_rows == 6 and big.total_rows == 6


def test_incomplete_collection_is_saved_as_incomplete(tmp_path):
    reg = registry(tmp_path)
    partial = collection(SIX_ROWS[:4], reported_total=6, complete=False, status="incomplete")
    manifest = reg.save_table(SIX_ROWS[:4], kind="observations", columns=COLUMNS, collection=partial)
    assert manifest.complete is False and reg.preview(manifest.artifact_id).complete is False


# --- 4. handles only: no paths, URLs, traversal, absolute paths --------------------------------

@pytest.mark.parametrize("bad", [
    "run_a/artifacts/0001_observations.csv", "C:\\Windows\\win.ini", "/etc/passwd", "../../.env",
    "https://cassavabase.org/brapi/v2/studies", "art_../x", "art_0001", "", 7, None, "ART_3F9A2C_0001",
])
def test_anything_but_a_handle_is_rejected(tmp_path, bad):
    reg = registry(tmp_path)
    save_six(reg)
    with pytest.raises(ArtifactError) as info:
        reg.load(bad)
    assert info.value.code == "invalid_handle"


def test_unknown_but_well_formed_handle_is_rejected(tmp_path):
    reg = registry(tmp_path)
    with pytest.raises(ArtifactError) as info:
        reg.preview("art_abcdef_0001")
    assert info.value.code == "invalid_handle"


def test_handles_from_another_run_do_not_resolve_without_an_explicit_import(tmp_path):
    reg_a = registry(tmp_path, "run_a")
    reg_b = registry(tmp_path, "run_b")
    manifest = save_six(reg_a)
    with pytest.raises(ArtifactError):
        reg_b.load(manifest.artifact_id)
    imported = reg_b.import_run("run_a")
    assert imported == [manifest.artifact_id]
    assert reg_b.load(manifest.artifact_id)[1][0]["observationDbId"] == "O1"


# --- 5. immutability, hashes, tampering, traversal, symlinks ---------------------------------------

def test_two_runs_with_the_same_study_id_never_collide_or_overwrite(tmp_path):
    reg_a = registry(tmp_path, "run_a")
    reg_b = registry(tmp_path, "run_b")
    a = save_six(reg_a)
    b = save_six(reg_b)
    assert a.artifact_id != b.artifact_id and a.relative_path != b.relative_path
    assert a.study_ids == b.study_ids == ["S1"]
    assert (tmp_path / "out" / a.relative_path).is_file() and (tmp_path / "out" / b.relative_path).is_file()
    second_in_a = save_six(reg_a)
    assert second_in_a.artifact_id.endswith("_0002") and second_in_a.relative_path.endswith("0002_observations.csv")


def test_existing_file_is_never_overwritten(tmp_path):
    reg = registry(tmp_path)
    (reg.artifact_dir / "0001_observations.csv").write_text("someone else's file", encoding="utf-8")
    with pytest.raises(ArtifactError) as info:
        save_six(reg)
    assert info.value.code == "exists"
    assert (reg.artifact_dir / "0001_observations.csv").read_text(encoding="utf-8") == "someone else's file"


def test_tampered_file_fails_the_hash_check(tmp_path):
    reg = registry(tmp_path)
    manifest = save_six(reg)
    path = tmp_path / "out" / manifest.relative_path
    path.write_text(path.read_text(encoding="utf-8").replace("O5,P5,clone B,bad", "O5,P5,clone B,25"), encoding="utf-8")
    with pytest.raises(ArtifactError) as info:
        reg.load(manifest.artifact_id)
    assert info.value.code == "hash_mismatch"


def test_manifest_pointing_outside_the_run_root_is_rejected(tmp_path):
    reg = registry(tmp_path)
    manifest = save_six(reg)
    secret = tmp_path / "secret.csv"
    secret.write_text("observationDbId,observationUnitDbId,germplasmName,value\n", encoding="utf-8")
    # The contract itself refuses '..' and absolute paths in a managed path ...
    for bad_path in ("../secret.csv", str(secret), "/etc/passwd"):
        with pytest.raises(ValidationError):
            type(manifest).model_validate({**manifest.model_dump(), "relative_path": bad_path})
    # ... and a registry entry rewritten to a sibling run dir is caught by the hash/containment checks.
    forged = manifest.model_copy(update={"relative_path": "run_z/artifacts/0001_observations.csv"})
    reg._manifests[manifest.artifact_id] = forged
    with pytest.raises(ArtifactError) as info:
        reg.load(manifest.artifact_id)
    assert info.value.code in ("missing", "escape")


def test_symlink_escape_is_rejected(tmp_path):
    reg = registry(tmp_path)
    manifest = save_six(reg)
    real = tmp_path / "out" / manifest.relative_path
    outside = tmp_path / "outside.csv"
    outside.write_bytes(real.read_bytes())
    real.unlink()
    try:
        os.symlink(outside, real)
    except (OSError, NotImplementedError):
        pytest.skip("this account cannot create symlinks (Windows needs Developer Mode or admin)")
    with pytest.raises(ArtifactError) as info:
        reg.load(manifest.artifact_id)
    assert info.value.code == "escape"


# --- 6. server separation and registry persistence ------------------------------------------------

def test_import_refuses_a_run_from_another_server(tmp_path):
    reg_cassava = registry(tmp_path, "run_c", base=OTHER)
    save_six(reg_cassava)
    reg_synthetic = registry(tmp_path, "run_s", base=BASE)
    with pytest.raises(ArtifactError) as info:
        reg_synthetic.import_run("run_c")
    assert info.value.code == "server_mismatch"
    with pytest.raises(ArtifactError):
        reg_synthetic.import_run("../run_c")


def test_import_verifies_hashes_and_refuses_missing_runs(tmp_path):
    reg_a = registry(tmp_path, "run_a")
    manifest = save_six(reg_a)
    (tmp_path / "out" / manifest.relative_path).write_text("tampered\n", encoding="utf-8")
    reg_b = registry(tmp_path, "run_b")
    with pytest.raises(ArtifactError) as info:
        reg_b.import_run("run_a")
    assert info.value.code == "hash_mismatch" and reg_b.artifact_ids() == []
    with pytest.raises(ArtifactError):
        reg_b.import_run("run_missing")


def test_registry_reopens_from_disk_with_ledger_and_imports(tmp_path):
    reg_a = registry(tmp_path, "run_a")
    a = save_six(reg_a)
    reg_b = registry(tmp_path, "run_b")
    b = save_six(reg_b)
    reg_b.record_study("S1", "complete", artifact_id=b.artifact_id)
    reg_b.import_run("run_a")
    reopened = registry(tmp_path, "run_b")
    assert set(reopened.artifact_ids()) == {a.artifact_id, b.artifact_id}
    assert [e.study_id for e in reopened.ledger()] == ["S1"]
    third = save_six(reopened)
    assert third.artifact_id.endswith("_0002")                      # numbering continues, no reuse
    with pytest.raises(ArtifactError):
        ArtifactRegistry(tmp_path / "out", "run_b", base_url=OTHER)   # reopening under another server is refused


def test_registry_file_is_plain_json_listing_only_this_runs_artifacts(tmp_path):
    reg_a = registry(tmp_path, "run_a")
    a = save_six(reg_a)
    reg_b = registry(tmp_path, "run_b")
    reg_b.import_run("run_a")
    data = json.loads((tmp_path / "out" / "run_b" / "registry.json").read_text(encoding="utf-8"))
    assert data["base_url"] == BASE and data["imported_runs"] == ["run_a"]
    assert data["artifacts"] == {} and a.artifact_id not in data["artifacts"]


def test_run_id_must_be_a_plain_identifier(tmp_path):
    for bad in ("..", "../x", "run/one", "", "a" * 81):
        with pytest.raises(ArtifactError):
            ArtifactRegistry(tmp_path / "out", bad, base_url=BASE)
