"""Display hints and chat settings, using synthetic metadata only."""
from types import SimpleNamespace
import pytest
import app
from agents.coordinator import KnownMetadata, KnownStudy
from run import CATALOG_PROMPT, CLARIFY_PROMPT, RunConfig


def test_chat_defaults_fit_slow_server(monkeypatch):
    monkeypatch.delenv("APP_READ_TIMEOUT", raising=False)
    monkeypatch.delenv("APP_MAX_HTTP_ATTEMPTS", raising=False)
    config = app.live_config("one study")
    assert config.read_timeout == 90
    assert config.max_elapsed_seconds == 600
    assert config.bootstrap_catalogs is True


@pytest.mark.parametrize("key,value", [("APP_READ_TIMEOUT", "nan"), ("APP_READ_TIMEOUT", "inf"),
                                      ("APP_READ_TIMEOUT", "0"), ("APP_READ_TIMEOUT", "601"),
                                      ("APP_MAX_HTTP_ATTEMPTS", "0"), ("APP_MAX_HTTP_ATTEMPTS", "not-a-number")])
def test_bad_chat_settings_explain_without_echoing_value(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    problems = app.live_preflight("one study")
    assert problems and "App settings" in problems[0]
    if value == "not-a-number":
        assert value not in problems[0]


def test_location_hint_is_display_only_and_no_traits_hint_while_waiting():
    metadata = KnownMetadata(base_url="https://synthetic.invalid/brapi/v2", complete=True,
                             studies=[KnownStudy(study_id="S1", location_id="L1", location_name="SYNTHETIC location one")])
    questions = ["Which location and trait do you mean?"]
    controller = SimpleNamespace(metadata=metadata, coordination=SimpleNamespace(questions=questions),
                                 config=RunConfig(question="Describe a trial", max_clarification_rounds=3))
    job = SimpleNamespace(controller=controller)
    text, finished = app.render(job, ("prompt", CLARIFY_PROMPT))
    assert "L1 · SYNTHETIC location one — 1 study" in text
    assert "list all traits" not in text and not finished          # a run waits here: a new question cannot start
    assert controller.config.question == "Describe a trial"
    assert questions == ["Which location and trait do you mean?"]


def test_location_choices_are_bounded_and_count_distinct_studies():
    studies = [KnownStudy(study_id=f"S{i}", location_id=f"L{i}", location_name=f"Location {i}") for i in range(12)]
    studies += [studies[0], KnownStudy(study_id="S99", location_id="L0", location_name="Location 0")]
    metadata = KnownMetadata(base_url="https://synthetic.invalid/brapi/v2", complete=True, studies=studies)
    text = app.clarification_choices(["Choose a location"], metadata)
    assert text.count("\n- ") == 10
    assert text.splitlines()[2] == "- L0 · Location 0 — 2 studies"
    assert app.clarification_choices(["Choose a year"], metadata) == ""


def test_catalog_prompt_offers_correct_human_controls():
    job = SimpleNamespace(waiting_on=CATALOG_PROMPT,
                          controller=SimpleNamespace(catalog_approval_screen=lambda: "Catalog scope: studies and variables"))
    assert app.reply_buttons({"job": job}) == (True, True, False, False)
    text, finished = app.render(job, ("prompt", CATALOG_PROMPT))
    assert "separate approval" in text and not finished


def test_catalog_approve_button_maps_only_at_catalog_prompt():
    import queue
    job = app.Job.__new__(app.Job)
    job.inbox = queue.Queue()
    job.waiting_on = CATALOG_PROMPT
    job.reply("approve")
    assert job.inbox.get_nowait() == "approve catalog"
    job.waiting_on = "approve / edit / cancel: "
    job.reply("approve")
    assert job.inbox.get_nowait() == "approve"
