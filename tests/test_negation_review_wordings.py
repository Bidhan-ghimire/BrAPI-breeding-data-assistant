"""A12 (coach 2026-10-22 20:48 UTC): the regression fixture of P3's gate (G1). Every wording of the two reviews of the negation rule
A9 that holds a word of negation or exclusion (132 of their 134) gives exactly dev ba2d297's candidates, exact IDs and unmatched
terms: with such a word, ontology terms name nothing, as before P3. The reviews found silent wrong plans in these wordings under
the earlier designs; here they behave as dev. REAL metadata: the 486 SweetPotatoBase variables of the benchmark cache of
2026-10-21. The fixture records dev's candidates by count and sha256 (the sorted IDs joined by spaces), and lists them when 10 or
fewer."""
import hashlib
import json
from pathlib import Path

import pytest

import agents.coordinator as co

_ROOT = Path(__file__).resolve().parents[1]
SPB = co.KnownMetadata(base_url="https://sweetpotatobase.org/brapi/v2", complete=True, variables=[
    co.KnownVariable(variable_id=i, name=n, trait=t or None)
    for i, n, t in json.loads((_ROOT / "tests" / "fixtures" / "spb_variables_20261021.json").read_text(encoding="utf-8"))["variables"]])
CASES = json.loads((_ROOT / "tests" / "fixtures" / "negation_review_wordings_20261022.json").read_text(encoding="utf-8"))["cases"]


def test_the_fixture_holds_the_reviews_gate_word_wordings_only():
    assert len(CASES) == 132
    assert all(co._TERMS_NAME_NOTHING.search(co.person_words(case["wording"]).lower()) for case in CASES)


@pytest.mark.parametrize("case", CASES, ids=[f"wording{i:03d}" for i in range(len(CASES))])
def test_a_wording_with_a_gate_word_plans_exactly_as_on_dev(case):
    c = co.find_candidates(case["wording"], SPB)
    ids = sorted(v.variable_id for v in c.variables)
    dev = case["dev"]
    assert len(ids) == dev["count"] and hashlib.sha256(" ".join(ids).encode()).hexdigest() == dev["sha256"]
    assert ids == dev.get("candidates", ids)
    assert sorted(c.exact_variable_ids) == dev["exact"] and c.unmatched_terms == dev["unmatched"]
