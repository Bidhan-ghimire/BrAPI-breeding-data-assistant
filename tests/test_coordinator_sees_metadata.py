"""
tests/test_coordinator_sees_metadata.py - the Coordinator's view of a finished analysis holds its metadata facts (coach 2026-10-24
09:49 UTC, NEXT 1). Luna 2026-10-24, B26.1 and B30.1 ("What does the SweetPotatoBase study description say ... tests"): the
metadata path read the answer from the study record (studyDescription "Specialty trial with 200 genotypes and 2 replications,
late season drought"), but coordination_runtime._state showed the Coordinator only claims and caveats, so it chose stop ("The
completed analysis contains no claims ..."). Every run since 22 Oct with metadata facts and no claims ended "blocked".
"""
from __future__ import annotations

from types import SimpleNamespace

import coordination_runtime as cr
from agents.coordinator import SUPERVISOR_SYSTEM_PROMPT
from contracts import AnalysisReport, MetadataFact

DESCRIPTION = "Specialty trial with 200 genotypes and 2 replications, late season drought"


def fact(field, value=None, *, missing=None, entity="4579"):
    return MetadataFact(entity_kind="study", entity_id=entity, entity_label="MZP2019CHL-ST02", field=field, value=value,
                        missing_reason=missing, source_artifact_id="art_0adf38_0001", source_artifact_hash="a" * 64)


def view(facts, claims=()):
    controller = SimpleNamespace(plan=None, config=SimpleNamespace(question="What does the study description say it tests?"),
                                 ctx=None, retrieval=None, data_review={"status": "approved"},
                                 analysis=AnalysisReport(status="completed", claims=list(claims), metadata_facts=facts),
                                 supervision={"decisions": [], "plan_revisions": 0, "analysis_passes": 1})
    return cr._state(controller, "analyzed")["analysis"]


def test_the_coordinator_sees_the_metadata_facts():
    """B30.1's shape: the facts the metadata path read, each with its entity, field, and value or missing reason."""
    v = view([fact("studyDescription", DESCRIPTION), fact("plantingDate", missing="not stated in the retrieved record")])
    assert v["claims"] == [] and v["metadata_facts_omitted"] == 0
    assert v["metadata_facts"] == [
        {"entity": "study 4579", "field": "studyDescription", "value": DESCRIPTION, "missing_reason": None},
        {"entity": "study 4579", "field": "plantingDate", "value": None, "missing_reason": "not stated in the retrieved record"}]


def test_the_facts_in_view_are_bounded():
    """Bounded like the claims: at most 60 facts, the rest counted; a value longer than 300 characters is cut and marked."""
    v = view([fact(f"field{i:02d}", ("x" * 400) if i == 0 else f"value {i}") for i in range(65)])
    assert len(v["metadata_facts"]) == 60 and v["metadata_facts_omitted"] == 5
    assert v["metadata_facts"][0]["value"] == "x" * 300 and v["metadata_facts"][0]["value_cut"] is True
    assert v["metadata_facts"][1] == {"entity": "study 4579", "field": "field01", "value": "value 1", "missing_reason": None}


def test_several_studies_all_show():
    """Review of this change, SHOULD-FIX 1: one study has about 20 fields, so a cap of 20 showed only the first study. The
    entities take turns: three studies of 20 fields all show; with four, each still shows its first fields."""
    three = view([fact(f"field{i:02d}", f"v{i}", entity=e) for e in ("S1", "S2", "S3") for i in range(20)])
    assert len(three["metadata_facts"]) == 60 and {f["entity"] for f in three["metadata_facts"]} == {"study S1", "study S2", "study S3"}
    four = view([fact(f"field{i:02d}", f"v{i}", entity=e) for e in ("S1", "S2", "S3", "S4") for i in range(20)])
    shown = [f["entity"] for f in four["metadata_facts"]]
    assert four["metadata_facts_omitted"] == 20 and all(shown.count(f"study {e}") == 15 for e in ("S1", "S2", "S3", "S4"))
    assert four["metadata_facts"][:4] == [{"entity": f"study {e}", "field": "field00", "value": "v0", "missing_reason": None}
                                          for e in ("S1", "S2", "S3", "S4")]


def test_a_long_value_is_cut_at_a_space_and_marked():
    """Review of this change, SHOULD-FIX 2: a cut inside a word or a number misleads ("202-208oF" became "202-2"); the cut falls
    at the last space before 300 characters, and the fact says it was cut. The person reads the whole value either way."""
    long = "Temperatures were " + "x" * 277 + " 202-208oF at noon"        # characters 296-304 are "202-208oF": a cut at 300 splits it
    (shown,) = view([fact("studyDescription", long)])["metadata_facts"]
    assert shown["value_cut"] is True and len(shown["value"]) <= 300 and long.startswith(shown["value"])
    assert long[len(shown["value"])] == " " and not shown["value"].endswith(" ") and "202" not in shown["value"]


def test_claims_and_facts_both_show_and_a_report_without_facts_shows_none():
    """Review of this change (surviving mutants): an analysis with claims and facts shows both; an analysis summary without
    a metadata_facts field (a stand-in) shows an empty list, not an error."""
    from contracts import Claim, Denominator

    claim = Claim(claim_id="clm_meta_001", kind="count", value=2.0, unit=None, denominator=Denominator(name="studies", n=2),
                  source_artifact_hashes=["a" * 64], transformation="rows in the complete studies table")
    both = view([fact("studyDescription", DESCRIPTION)], claims=[claim])
    assert len(both["claims"]) == 1 and both["metadata_facts"][0]["value"] == DESCRIPTION
    bare = SimpleNamespace(plan=None, config=SimpleNamespace(question="q"), ctx=None, retrieval=None, data_review={"status": "approved"},
                           analysis=SimpleNamespace(status="completed", claims=[], caveats=[]),
                           supervision={"decisions": [], "plan_revisions": 0, "analysis_passes": 1})
    state = cr._state(bare, "analyzed")["analysis"]
    assert state["metadata_facts"] == [] and state["metadata_facts_omitted"] == 0


def test_the_finish_rule_names_the_metadata_facts():
    """The Coordinator's rule for finish counts a record's fields read by code, not only claims."""
    text = " ".join(SUPERVISOR_SYSTEM_PROMPT.split())
    assert "finish: only after completed, code-validated analysis supports an answer: its claims, or" in text
    assert "for a question about what a record states" in text
    assert "its metadata_facts (fields read by code from the retrieved records)" in text
