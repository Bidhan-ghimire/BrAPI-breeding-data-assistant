"""Location suggestions are explicitly alternatives, never implied geographic matches."""
from copy import deepcopy
from types import SimpleNamespace

import app
from agents.coordinator import KnownMetadata, KnownStudy
from run import CLARIFY_PROMPT, RunConfig


def test_unknown_place_clarification_labels_popular_locations_as_alternatives():
    metadata = KnownMetadata(
        base_url="https://synthetic.invalid/brapi/v2", complete=True,
        studies=[KnownStudy(study_id="S1", location_id="L1", location_name="SYNTHETIC location one")],
    )
    questions = ["Which database location do you mean for Kansas?"]
    original_metadata = deepcopy(metadata)
    controller = SimpleNamespace(
        metadata=metadata, coordination=SimpleNamespace(questions=questions),
        config=RunConfig(question="List studies in Kansas", max_clarification_rounds=3),
    )
    shown, finished = app.render(SimpleNamespace(controller=controller), ("prompt", CLARIFY_PROMPT))

    assert "Locations with the most studies:" in shown
    assert "popular database locations, not confirmed matches for the place you requested" in shown
    assert "Choosing one changes the location scope of your question." in shown
    assert shown.index("not confirmed matches") < shown.index("- L1")
    assert "- L1 · SYNTHETIC location one — 1 study" in shown
    assert finished is False
    assert controller.config.question == "List studies in Kansas"
    assert questions == ["Which database location do you mean for Kansas?"]
    assert metadata == original_metadata


def test_location_alternative_warning_preserves_ranked_choices_and_exact_counts():
    metadata = KnownMetadata(
        base_url="https://synthetic.invalid/brapi/v2", complete=True,
        studies=[
            KnownStudy(study_id="S1", location_id="L2", location_name="SYNTHETIC second"),
            KnownStudy(study_id="S2", location_id="L1", location_name="SYNTHETIC first"),
            KnownStudy(study_id="S3", location_id="L1", location_name="SYNTHETIC first"),
        ],
    )
    shown = app.clarification_choices(["Choose a location"], metadata)
    assert shown.splitlines()[2:] == [
        "- L1 · SYNTHETIC first — 2 studies",
        "- L2 · SYNTHETIC second — 1 study",
    ]
    assert app.clarification_choices(["Choose a year"], metadata) == ""
    assert app.clarification_choices(["Choose a location"], None) == ""
