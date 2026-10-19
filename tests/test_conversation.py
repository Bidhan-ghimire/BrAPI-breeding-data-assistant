"""
tests/test_conversation.py — a message typed while a run is waiting for a reply (conversation.py and app.respond), offline.

What these tests prove, in plain words:
* At every waiting prompt (the planner's, the coordinator's, the Retriever's and the Analyst's questions, the plan approval,
  the catalog setup, the data review and the final review), a side question gets the model's short reply and one line
  written by code that repeats what the run waits for; NOTHING reaches the run. An out-of-scope message ("which country
  has the largest population?", "i hate you") is handled the same way, never as a new question (no drop or keep); its
  reply says what the app covers, written by code when the model gives none, with no example question: the waiting
  question is what should be answered.
* The prompt's own words (approve, edit, cancel, continue, accept, reject) reach the run with no model call, also as
  "Approve." or " CANCEL "; the model's words never do, whatever label it gives and even when it writes "approve".
* "list all traits" (or locations, seasons, programs, study types, with an optional word) typed mid-run is answered by code from the
  waiting run's own catalog, then what the run waits for is repeated; nothing reaches the run and no model is asked, also
  at the "study IDs" prompt after edit. Before the run has a catalog (its catalog setup), it is a new question. When the
  waiting question is itself about listing ("Should I list the seasons?"), the reply may be its answer: the model decides. A new data question (one the model labels,
  or a request to list another part of the catalog) asks drop or keep and starts nothing until answered: keep shows the
  waiting question again; drop ends the waiting run and starts the new one.
* Unsure asks "should I send it as your answer?": "yes" sends the person's own words, "no" drops them. Only what the run
  itself can take is ever sent: an answer to a question, or a described change at the plan approval ("yes", "ok" or
  "Uganda" alone would end it as a denial). Elsewhere the run's choices are repeated. No model, a failed call or an
  invalid reply: an answer still goes to the run as before; anything the run could not take gets its choices again.
* A run that has ended meanwhile (15 minutes without a reply) gets nothing: the chat says so and shows how it ended.
* The planner's location list is shown once per run; the hint to ask "list all traits" is shown under a planner's question
  about a trait, only where it works (the run has its catalog tables; the question is not about listing). The run's
  buttons are hidden while the tab asks its own question.
Fake runs and scripted models only: no thread, no model provider and no network (the conftest guard blocks sockets).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import app
import conversation
from app_help import faq_items
from llm import FakeModelClient, LlmConfigError, ModelProviderError, reply_text, reply_tools
from run import (ANALYST_PROMPT, CATALOG_PROMPT, CLARIFY_PROMPT, DATA_REVIEW_PROMPT, RETRIEVER_PROMPT, SUPERVISOR_PROMPT,
                 RunConfig)
from agents.coordinator import KnownMetadata, KnownStudy, KnownVariable
import catalog_lists

SOURCE = "https://waiting.invalid/brapi/v2"
APPROVAL = "approve / edit / cancel: "
FINAL_REVIEW = "accept / reject (anything else = leave pending): "
EDIT_IDS = "study IDs, comma separated (empty = keep): "
QUESTIONS = {CLARIFY_PROMPT: "Which study is meant?", SUPERVISOR_PROMPT: "Should the plan include 2025?",
             RETRIEVER_PROMPT: "Which label should be used?", ANALYST_PROMPT: "Use the seasons column instead?"}
STILL_WAITING = {APPROVAL: "The plan above is still waiting: reply **approve**",
                 CATALOG_PROMPT: "The catalog setup above is still waiting: reply **approve** or **cancel**.",
                 DATA_REVIEW_PROMPT: "The data review above is still waiting: reply **continue**",
                 FINAL_REVIEW: "The answer above is still waiting for your review: reply **accept** or **reject**.",
                 **{prompt: f"I'm still waiting for your answer to my question (or reply **cancel**): “{q}”" for prompt, q in QUESTIONS.items()}}
EVERY_PROMPT = list(STILL_WAITING)
IDS = {CLARIFY_PROMPT: "planner", SUPERVISOR_PROMPT: "coordinator", RETRIEVER_PROMPT: "retriever", ANALYST_PROMPT: "analyst",
       APPROVAL: "plan-approval", CATALOG_PROMPT: "catalog", DATA_REVIEW_PROMPT: "data-review", FINAL_REVIEW: "final-review"}
DECIDED_BY_TYPED_WORDS = [CATALOG_PROMPT, DATA_REVIEW_PROMPT, FINAL_REVIEW]


def waiting_controller(**changes) -> SimpleNamespace:
    values = dict(coordination=SimpleNamespace(questions=[QUESTIONS[CLARIFY_PROMPT]]), metadata=None, plan=None,
                  approval_revisions=0, clarification_answers=0,
                  config=SimpleNamespace(question="Tell me about a study.", max_clarification_rounds=3),
                  analysis=SimpleNamespace(caveats=[f"clarification needed: {QUESTIONS[ANALYST_PROMPT]}"]), analyst_answers=0,
                  retriever_questions=[QUESTIONS[RETRIEVER_PROMPT]], retriever_answers=0,
                  supervisor_questions=[QUESTIONS[SUPERVISOR_PROMPT]],
                  catalog_approval_screen=lambda: "Catalog scope: studies and variables", answer_markdown=lambda: "full report")
    values.update(changes)
    return SimpleNamespace(**values)


class WaitingJob:
    """A run that waits on one prompt. It records every reply that reaches it; it has no thread, model or request."""

    def __init__(self, waiting_on: str, controller: SimpleNamespace | None = None):
        self.waiting_on = waiting_on
        self.controller = controller or waiting_controller()
        self.answer_shown = waiting_on == FINAL_REVIEW
        self.locations_listed = False
        self.replies: list[str] = []
        self.closed = False

    def reply(self, text):
        self.replies.append(text)

    def close(self):
        self.closed = True
        self.reply("cancel")

    def next_event(self, timeout=None):
        self.waiting_on = "the next step: "              # any later prompt; the tests only look at what reached the run
        return "prompt", "the next step: "


class RecordingJob:
    """Stands in for app.Job when a new question starts: it records the question and waits."""

    started: list[str] = []

    def __init__(self, question, *, make_config, make_models):
        RecordingJob.started.append(question)
        self.controller = SimpleNamespace(run_id="recorded-run", answer_markdown=lambda: "recorded report")
        self.answer_shown = False
        self.waiting_on = ""
        self.locations_listed = False

    def start(self):
        return self

    def next_event(self, timeout=None):
        self.waiting_on = "the recorded run waits: "
        return "prompt", "the recorded run waits: "


def decision(label: str, reply: str = "") -> str:
    return json.dumps({"label": label, "reason": "scripted for this test", "reply": reply})


def scripted(*replies):
    """A model factory for the chat: the same scripted model each time, and a list of every time it was built."""
    model = FakeModelClient(list(replies))
    built: list[FakeModelClient] = []

    def factory():
        built.append(model)
        return model
    return model, built, factory


def unusable_settings():
    raise LlmConfigError("LLM_MODEL is blank")


@pytest.fixture
def chat(tmp_path, monkeypatch):
    import brapi_client as bc

    monkeypatch.setenv("BRAPI_BASE_URL", SOURCE)
    requests: list[object] = []
    monkeypatch.setattr(bc, "RequestsTransport", lambda: requests.append)     # a database request would be recorded here
    monkeypatch.setattr(app, "Job", RecordingJob)
    RecordingJob.started = []

    def turn(message, job, state=None, **options):
        state = {"job": job} if state is None else state
        out = list(app.respond(message, [], state, make_config=lambda q: RunConfig(question=q, out_dir=tmp_path / "out"),
                               make_models=lambda: None, preflight=lambda q: [], **options))
        assert out[-1][2] == ""
        return out[-1][0], out[-1][1]

    return SimpleNamespace(turn=turn, requests=requests, out=tmp_path / "out")


# --------------------------------------------------------------------------
# A side question: answered, and the run keeps waiting
# --------------------------------------------------------------------------

@pytest.mark.parametrize("waiting_on", EVERY_PROMPT, ids=[IDS[p] for p in EVERY_PROMPT])
def test_a_side_question_is_answered_and_nothing_reaches_the_run(chat, waiting_on):
    job = WaitingJob(waiting_on)
    model, _, factory = scripted(reply_text(decision("side_question", "I am the Breeding data assistant.")))
    history, state = chat.turn("what's your name?", job, triage_model=factory)
    content = history[-1]["content"]
    assert content.startswith("I am the Breeding data assistant.\n\n") and STILL_WAITING[waiting_on] in content
    assert job.replies == [] and not job.closed and state["job"] is job                # untouched, still waiting
    assert [m["role"] for m in history] == ["user", "assistant"] and chat.requests == [] and not chat.out.exists()
    [request] = model.requests                                                          # one call, no tools
    assert request["tools"] == [] and request["messages"][1] == {"role": "user", "content": "what's your name?"}
    prompt = request["messages"][0]["content"]
    assert "while the Breeding data assistant was waiting for their reply" in prompt and "waiting.invalid" in prompt
    if waiting_on in QUESTIONS:                                                         # the model saw the waiting question
        assert QUESTIONS[waiting_on] in prompt and "answer_to_pending" in prompt
    if waiting_on in DECIDED_BY_TYPED_WORDS:                                            # there only typed words decide
        assert "answer_to_pending" not in prompt


OUT_OF_SCOPE_REPLY = "I can help only with public breeding records, such as studies, traits and measurements."
FRONT_DOOR_REPLY = OUT_OF_SCOPE_REPLY + " For example: “Which studies have a 2026 season?”"   # only the front door has an example


@pytest.mark.parametrize("waiting_on", EVERY_PROMPT, ids=[IDS[p] for p in EVERY_PROMPT])
def test_out_of_scope_is_answered_like_a_side_question_never_drop_or_keep(chat, waiting_on):
    """2026-10-15 live: "which country has the largest population?" typed mid-run was taken as a new data question
    (drop or keep), and after drop the front door offered to look it up."""
    job = WaitingJob(waiting_on)
    model, _, factory = scripted(reply_text(decision("out_of_scope", OUT_OF_SCOPE_REPLY)))
    history, state = chat.turn("which country has the largest population?", job, triage_model=factory)
    content = history[-1]["content"]
    assert content.startswith(OUT_OF_SCOPE_REPLY + "\n\n") and STILL_WAITING[waiting_on] in content
    assert "drop" not in content and "look up" not in content
    prompt = model.requests[0]["messages"][0]["content"]                           # here: no example question
    assert conversation.OUT_OF_SCOPE_RULE in prompt and "then offer one example question" not in prompt
    assert job.replies == [] and not job.closed and state == {"job": job}          # no drop or keep, nothing sent
    assert app.reply_buttons(state) == app.reply_buttons({"job": job})              # the run's own buttons stay
    assert RecordingJob.started == [] and chat.requests == [] and len(model.requests) == 1


def test_after_drop_an_out_of_scope_question_gets_no_lookup_offer(chat):
    """The live chain of 2026-10-15: mid-run the model said new_data_question, the person chose drop, and the front door
    then offered to look the question up. Now the front door's out_of_scope gets the reply only."""
    job = WaitingJob(CLARIFY_PROMPT)
    model, _, factory = scripted(reply_text(decision("new_data_question")), reply_text(decision("out_of_scope")),
                                 reply_text(json.dumps({"label": "out_of_scope", "reason": "scripted", "reply": FRONT_DOOR_REPLY})))
    _, state = chat.turn("which country has the largest population?", job, triage_model=factory)
    assert state["new_question"] == "which country has the largest population?"
    history, state = chat.turn("drop", job, state, triage_model=factory)
    assert job.closed and history[-1]["content"] == FRONT_DOOR_REPLY and "look up" not in history[-1]["content"]
    assert RecordingJob.started == [] and len(model.requests) == 3                     # label + talk call; no run
    assert state == {}                                                                  # no waiting "yes", no job


def test_an_out_of_scope_message_without_model_text_gets_the_line_written_by_code(chat):
    job = WaitingJob(CLARIFY_PROMPT)
    _, _, factory = scripted(reply_text(decision("out_of_scope", "")))
    history, _ = chat.turn("i hate you", job, triage_model=factory)
    assert history[-1]["content"] == ("I can help only with public breeding records in waiting.invalid: studies, trials, traits, "
                                      "locations, seasons and recorded measurements.\n\n" + STILL_WAITING[CLARIFY_PROMPT])
    assert job.replies == [] and "For example" not in history[-1]["content"]     # the waiting question, not an example


def test_the_side_reply_is_escaped_and_cut_like_database_text(chat):
    job = WaitingJob(CLARIFY_PROMPT)
    _, _, factory = scripted(reply_text(decision("side_question", "<img src=x onerror=alert(1)> [x](http://x.invalid) " + "a" * 900)))
    history, _ = chat.turn("tell me a joke", job, triage_model=factory)
    content = history[-1]["content"]
    assert "<img" not in content and "&lt;img" in content and "](" not in content and "a" * 601 not in content
    assert job.replies == []


def test_an_empty_side_reply_still_says_what_the_run_waits_for(chat):
    job = WaitingJob(SUPERVISOR_PROMPT)
    _, _, factory = scripted(reply_text(decision("side_question", "")))
    history, _ = chat.turn("hmm", job, triage_model=factory)
    assert history[-1]["content"] == STILL_WAITING[SUPERVISOR_PROMPT] and job.replies == []


# --------------------------------------------------------------------------
# The prompt's own words go to the run; the model's words never do
# --------------------------------------------------------------------------

@pytest.mark.parametrize("waiting_on, typed, sent", [
    (CLARIFY_PROMPT, "cancel", "cancel"), (SUPERVISOR_PROMPT, " CANCEL ", "cancel"), (RETRIEVER_PROMPT, "Cancel.", "cancel"),
    (ANALYST_PROMPT, "cancel", "cancel"), (APPROVAL, "Approve.", "approve"), (APPROVAL, "edit", "edit"), (APPROVAL, "cancel", "cancel"),
    (CATALOG_PROMPT, "approve", "approve"), (CATALOG_PROMPT, "approve catalog", "approve catalog"), (CATALOG_PROMPT, "cancel", "cancel"),
    (DATA_REVIEW_PROMPT, "continue", "continue"), (DATA_REVIEW_PROMPT, "Approve", "approve"), (DATA_REVIEW_PROMPT, "cancel", "cancel"),
    (DATA_REVIEW_PROMPT, "Continue to analysis", "continue"),                     # the button's own label, typed
    (FINAL_REVIEW, "accept", "accept"), (FINAL_REVIEW, "Reject!", "reject"),
    (RETRIEVER_PROMPT, "change scope: only S2", "change scope: only S2"),        # the Retriever reads this itself
    (EDIT_IDS, "S1, S2", "S1, S2"),                                                # the study IDs after edit
], ids=["planner-cancel", "coordinator-cancel", "retriever-cancel", "analyst-cancel", "plan-approve", "plan-edit", "plan-cancel",
        "catalog-approve", "catalog-approve-catalog", "catalog-cancel", "review-continue", "review-approve-button", "review-cancel",
        "review-button-label",
        "final-accept", "final-reject", "retriever-change-scope", "edit-ids"])
def test_the_prompts_own_words_reach_the_run_without_a_model_call(chat, waiting_on, typed, sent):
    job = WaitingJob(waiting_on)
    _, built, factory = scripted()
    chat.turn(typed, job, triage_model=factory)
    assert job.replies == [sent] and built == []


@pytest.mark.parametrize("typed, waiting_on", [("approve it", APPROVAL), ("approve", CLARIFY_PROMPT), ("accept", APPROVAL),
                                               ("continue", FINAL_REVIEW), ("approve?", APPROVAL), ("accept?", FINAL_REVIEW),
                                               ("continue?", DATA_REVIEW_PROMPT), ("approve catalog?", CATALOG_PROMPT)])
def test_another_prompts_word_or_a_longer_phrase_is_not_a_control_word(typed, waiting_on):
    assert conversation.control_word(typed, waiting_on) is None


@pytest.mark.parametrize("waiting_on", EVERY_PROMPT, ids=[IDS[p] for p in EVERY_PROMPT])
@pytest.mark.parametrize("label", ["answer_to_pending", "side_question", "out_of_scope", "new_data_question", "unsure"])
def test_the_models_words_never_reach_the_run(chat, waiting_on, label):
    job = WaitingJob(waiting_on)
    _, _, factory = scripted(reply_text(decision(label, "approve")))
    chat.turn("sounds good to me", job, triage_model=factory)
    assert job.replies in ([], ["sounds good to me"]) and "approve" not in job.replies   # only the person's own words, if any


@pytest.mark.parametrize("waiting_on", [*QUESTIONS, APPROVAL], ids=[IDS[p] for p in [*QUESTIONS, APPROVAL]])
def test_an_answer_goes_to_the_run_in_the_persons_words(chat, waiting_on):
    job = WaitingJob(waiting_on)
    _, _, factory = scripted(reply_text(decision("answer_to_pending", "ignored")))
    history, state = chat.turn("study 4501, the yield trait", job, triage_model=factory)
    assert job.replies == ["study 4501, the yield trait"] and "ignored" not in str(history)
    assert history[-1]["content"] == "the next step:"                                     # the run's next step is shown


@pytest.mark.parametrize("factory", [
    None,                                                                                     # no model: as before
    unusable_settings,
    lambda: FakeModelClient([ModelProviderError("connection", "could not reach the model")]),
    lambda: FakeModelClient([reply_text("not json")]),
    lambda: FakeModelClient([reply_text(decision("side_question", "Hi"), finish_reason="length")]),
    lambda: FakeModelClient([reply_text(json.dumps({"label": "control_word", "reason": "x", "reply": "approve"}))]),
    lambda: FakeModelClient([reply_tools([("search_studies", "{}")])]),
], ids=["no-model", "unusable-settings", "provider-error", "invalid-reply", "cut-off", "control-word-label", "tool-call"])
@pytest.mark.parametrize("waiting_on", [CLARIFY_PROMPT, ANALYST_PROMPT, APPROVAL, *DECIDED_BY_TYPED_WORDS],
                         ids=["planner", "analyst", "plan-approval", "catalog", "data-review", "final-review"])
def test_without_a_decision_only_what_the_run_can_take_goes_to_it(chat, factory, waiting_on):
    job = WaitingJob(waiting_on)
    history, _ = chat.turn("sounds good to me", job, triage_model=factory)
    if waiting_on in DECIDED_BY_TYPED_WORDS:                       # there it could only cancel, deny or leave it pending
        assert job.replies == [] and history[-1]["content"].startswith("I did not take that as a decision. ")
    else:                                                          # an answer, or a described change: as before
        assert job.replies == ["sounds good to me"]


@pytest.mark.parametrize("typed", ["approve?", "yes", "ok", "Uganda", "go ahead"])
@pytest.mark.parametrize("factory", [None, lambda: FakeModelClient([reply_text(decision("answer_to_pending"))]),
                                     lambda: FakeModelClient([reply_text(decision("unsure"))])],
                         ids=["no-model", "answer", "unsure"])
def test_at_the_plan_words_the_run_would_take_as_a_denial_are_never_sent(chat, typed, factory):
    job = WaitingJob(APPROVAL)
    history, state = chat.turn(typed, job, triage_model=factory)
    assert job.replies == [] and "unsent_reply" not in state
    assert history[-1]["content"] == ("I did not take that as a decision. The plan above is still waiting: reply **approve** to "
                                      "use it or **cancel**, or describe a change.")


def test_at_the_plan_a_described_change_is_sent_or_offered(chat):
    job = WaitingJob(APPROVAL)
    _, _, factory = scripted(reply_text(decision("answer_to_pending")), reply_text(decision("unsure")))
    chat.turn("only 2025 please", job, triage_model=factory)
    assert job.replies == ["only 2025 please"]
    job = WaitingJob(APPROVAL)
    _, state = chat.turn("add study 4501", job, triage_model=factory)
    assert job.replies == [] and state["unsent_reply"] == "add study 4501"


# --------------------------------------------------------------------------
# A new data question mid-run: drop or keep
# --------------------------------------------------------------------------

CATALOG = KnownMetadata(base_url=SOURCE, complete=True,
                        studies=[KnownStudy(study_id="S1", location_id="L1", location_name="SYNTHETIC location one", seasons=["2026"])],
                        variables=[KnownVariable(variable_id="V1", name="fresh root yield"), KnownVariable(variable_id="V2", name="dry matter")])
FETCHED = "2026-06-01 00:00 UTC"


def catalog_tables() -> dict[str, catalog_lists.CatalogTable]:
    from datetime import datetime, timezone

    return {export: catalog_lists.CatalogTable(export=export, artifact_id=f"art_abcdef_000{n}", sha256="b" * 64, rows=2, base_url=SOURCE,
                                               fetched_at=datetime(2026, 6, 1, tzinfo=timezone.utc), fetched_in_this_run=False)
            for n, export in enumerate(("studies", "variables"), 1)}


@pytest.mark.parametrize("waiting_on", EVERY_PROMPT, ids=[IDS[p] for p in EVERY_PROMPT])
def test_a_list_mid_run_is_shown_from_the_runs_catalog_and_the_run_keeps_waiting(chat, waiting_on):
    job = WaitingJob(waiting_on, waiting_controller(metadata=CATALOG, catalog_tables=catalog_tables()))
    _, built, factory = scripted()
    history, state = chat.turn("list traits with YIELD", job, triage_model=factory)
    content = history[-1]["content"]
    assert content.startswith("**Trait names containing “yield”: 1 of 2.** A to Z:\n\n- fresh root yield\n")
    assert f"Source: waiting.invalid, trait catalog fetched {FETCHED} (a saved copy; nothing new was requested)." in content
    assert content.rsplit("\n\n", 1)[1].startswith(STILL_WAITING[waiting_on])                # then what the run waits for
    assert job.replies == [] and not job.closed and built == [] and state == {"job": job}     # nothing sent, no model, no new question
    assert chat.requests == [] and not chat.out.exists()                                    # no request and no new run


def test_a_list_at_the_study_ids_prompt_is_shown_and_never_taken_as_study_ids(chat):
    job = WaitingJob(EDIT_IDS, waiting_controller(metadata=CATALOG, catalog_tables=catalog_tables()))
    _, built, factory = scripted()
    history, state = chat.turn("list all traits", job, triage_model=factory)
    assert history[-1]["content"].startswith("**Traits in the catalog: 2.** A to Z:") and job.replies == [] and built == []
    chat.turn("S1, S2", job, state, triage_model=factory)                       # the IDs themselves still go straight to it
    assert job.replies == ["S1, S2"]


@pytest.mark.parametrize("pending, typed", [("Should I list the seasons or only count them?", "list the seasons"),
                                            ("Should I show all traits measured in study 4501, or only yield?", "show all traits")])
def test_when_the_waiting_question_is_about_listing_a_list_reply_goes_to_the_model(chat, pending, typed):
    """Review 2026-10-15: there the reply may be the answer; showing the catalog list would swallow it."""
    controller = waiting_controller(metadata=CATALOG, catalog_tables=catalog_tables(),
                                    analysis=SimpleNamespace(caveats=[f"clarification needed: {pending}"]))
    job = WaitingJob(ANALYST_PROMPT, controller)
    model, _, factory = scripted(reply_text(decision("answer_to_pending")))
    history, _ = chat.turn(typed, job, triage_model=factory)
    assert job.replies == [typed] and len(model.requests) == 1 and "in the catalog" not in history[-1]["content"]


@pytest.mark.parametrize("waiting_on", EVERY_PROMPT, ids=[IDS[p] for p in EVERY_PROMPT])
def test_a_list_before_the_run_has_a_catalog_asks_drop_or_keep_without_a_model_call(chat, waiting_on):
    job = WaitingJob(waiting_on)                                    # no catalog yet: in a real run only during its catalog setup
    _, built, factory = scripted()
    history, state = chat.turn("list all traits", job, triage_model=factory)
    assert history[-1]["content"].startswith("That sounds like a new question: “list all traits”. Reply **drop**")
    assert job.replies == [] and built == [] and state["new_question"] == "list all traits" and state["job"] is job
    assert app.reply_buttons(state) == (False, False, False, False)                         # the tab asks; no run buttons


@pytest.mark.parametrize("typed", ["list all traits", "Show me the locations", "please list the seasons", "list studies",
                                   "can you please list all traits", "show me every trait", "list every study", "list all traits please"])
def test_requests_to_list_the_catalog_are_new_questions(typed):
    assert conversation.is_list_request(typed, [QUESTIONS[CLARIFY_PROMPT]])


@pytest.mark.parametrize("typed", ["the yield trait", "all of them", "study 4501", "show me the yield trait", "list",
                                   "show the locations too", "list studies separately"])
def test_answers_are_not_list_requests(typed):
    assert not conversation.is_list_request(typed, [QUESTIONS[CLARIFY_PROMPT]])


def test_when_the_waiting_question_is_about_listing_the_model_decides(chat):
    job = WaitingJob(ANALYST_PROMPT, waiting_controller(analysis=SimpleNamespace(
        caveats=["clarification needed: Should I list the trials or only count them?"])))
    model, _, factory = scripted(reply_text(decision("answer_to_pending")))
    chat.turn("list the trials", job, triage_model=factory)
    assert job.replies == ["list the trials"] and len(model.requests) == 1


def test_a_new_data_question_the_model_finds_asks_drop_or_keep_too(chat):
    job = WaitingJob(RETRIEVER_PROMPT)
    _, _, factory = scripted(reply_text(decision("new_data_question")))
    history, state = chat.turn("which studies have a 2025 season?", job, triage_model=factory)
    assert "“which studies have a 2025 season?”" in history[-1]["content"] and "**keep**" in history[-1]["content"]
    assert job.replies == [] and state["new_question"] == "which studies have a 2025 season?"


def test_keep_shows_the_waiting_question_again_and_sends_nothing(chat):
    job = WaitingJob(CLARIFY_PROMPT)
    _, state = chat.turn("list all traits", job)
    history, state = chat.turn("something else", job, state)                 # neither drop nor keep: asked again
    assert history[-1]["content"].startswith("That sounds like a new question") and job.replies == []
    history, state = chat.turn("Keep", job, state)
    content = history[-1]["content"]
    assert "I need one more detail before I can make a plan (answer 1 of 3)" in content and "Which study is meant?" in content
    assert job.replies == [] and "new_question" not in state and state["job"] is job and RecordingJob.started == []


@pytest.mark.parametrize("waiting_on", [ANALYST_PROMPT, FINAL_REVIEW], ids=["analyst", "final-review"])
def test_drop_ends_the_waiting_run_and_starts_the_new_question(chat, waiting_on):
    job = WaitingJob(waiting_on)
    _, state = chat.turn("list all traits", job, {"job": job, "guide_shown": True, "full_report": "old report"})
    history, state = chat.turn("drop", job, state)
    assert job.closed and job.replies == ["cancel"]                          # ended by the app's usual cancel; it is recorded
    assert RecordingJob.started == ["list all traits"] and state["job"] is not job
    assert "new_question" not in state and "full_report" not in state                  # the old run's report is not kept
    assert state["guide_shown"] is True and history[-1]["content"] == "the recorded run waits:"


# --------------------------------------------------------------------------
# Unsure: ask first
# --------------------------------------------------------------------------

def test_unsure_at_a_question_asks_before_sending_and_yes_sends_the_persons_words(chat):
    job = WaitingJob(ANALYST_PROMPT)
    _, built, factory = scripted(reply_text(decision("unsure")))
    history, state = chat.turn("hmm maybe the other one", job, triage_model=factory)
    assert history[-1]["content"] == ("Should I send “hmm maybe the other one” as your answer? Reply **yes** to send it, or say it "
                                      "another way. My question was: “Use the seasons column instead?”")
    assert job.replies == [] and state["unsent_reply"] == "hmm maybe the other one"
    assert app.reply_buttons(state) == (False, False, False, False)
    history, state = chat.turn("yes", job, state, triage_model=factory)
    assert job.replies == ["hmm maybe the other one"] and "unsent_reply" not in state and len(built) == 1


@pytest.mark.parametrize("no", ["no", "No.", "nope", "never mind"])
def test_no_drops_the_unsent_answer_without_a_model_call(chat, no):
    job = WaitingJob(SUPERVISOR_PROMPT)
    model, _, factory = scripted(reply_text(decision("unsure")))
    _, state = chat.turn("hmm maybe", job, triage_model=factory)
    history, state = chat.turn(no, job, state, triage_model=factory)
    assert history[-1]["content"] == STILL_WAITING[SUPERVISOR_PROMPT] and job.replies == [] and "unsent_reply" not in state
    assert len(model.requests) == 1


def test_a_run_that_has_ended_meanwhile_gets_nothing(chat):
    import threading

    ended = waiting_controller(execution_status="canceled", analysis=None, retrieval=None,
                               notes=["no clarification answer; run canceled at the clarification prompt"])
    job = WaitingJob(CLARIFY_PROMPT, ended)
    job.thread = threading.Thread(target=lambda: None)
    job.thread.start()
    job.thread.join()                                                  # the run's thread has finished
    job.next_event = lambda timeout=None: ("done", SimpleNamespace(execution_status="canceled", human_acceptance="pending"))
    _, built, factory = scripted()
    history, state = chat.turn("keep", job, {"job": job, "new_question": "list all traits"}, triage_model=factory)
    content = history[-1]["content"]
    assert content.startswith("This question had already ended, so your message was not sent to it.") and "Question canceled" in content
    assert job.replies == [] and built == [] and "job" not in state and "new_question" not in state
    assert state["full_report"] == "full report"


def test_unsure_then_something_else_is_sorted_as_a_new_message(chat):
    job = WaitingJob(CLARIFY_PROMPT)
    _, _, factory = scripted(reply_text(decision("unsure")))
    _, state = chat.turn("hmm", job, triage_model=factory)
    history, state = chat.turn("cancel", job, state, triage_model=factory)
    assert job.replies == ["cancel"] and "unsent_reply" not in state


def test_unsure_at_the_plan_asks_whether_it_is_a_change(chat):
    job = WaitingJob(APPROVAL)
    _, _, factory = scripted(reply_text(decision("unsure")))
    history, state = chat.turn("hmm, 2025 too?", job, triage_model=factory)
    assert history[-1]["content"].startswith("Should I send “hmm, 2025 too?” as a change to the plan above? Reply **yes**")
    assert job.replies == [] and state["unsent_reply"] == "hmm, 2025 too?"


@pytest.mark.parametrize("waiting_on", DECIDED_BY_TYPED_WORDS, ids=[IDS[p] for p in DECIDED_BY_TYPED_WORDS])
@pytest.mark.parametrize("label", ["unsure", "answer_to_pending"])
def test_where_only_typed_words_decide_the_choices_are_repeated(chat, waiting_on, label):
    job = WaitingJob(waiting_on)
    _, _, factory = scripted(reply_text(decision(label)))
    history, state = chat.turn("looks fine I guess", job, triage_model=factory)
    assert history[-1]["content"].startswith("I did not take that as a decision. " + STILL_WAITING[waiting_on])
    assert job.replies == [] and "unsent_reply" not in state


# --------------------------------------------------------------------------
# What the waiting prompts show
# --------------------------------------------------------------------------

def location_controller(question: str, **changes) -> SimpleNamespace:
    metadata = KnownMetadata(base_url=SOURCE, complete=True,
                             studies=[KnownStudy(study_id="S1", location_id="L1", location_name="SYNTHETIC location one")])
    return waiting_controller(**{"metadata": metadata, "coordination": SimpleNamespace(questions=[question]),
                                 "catalog_tables": catalog_tables(), **changes})


def test_the_location_list_is_shown_once_per_run():
    job = WaitingJob(CLARIFY_PROMPT, location_controller("Which location do you mean?"))
    first, _ = app.render(job, ("prompt", CLARIFY_PROMPT))
    again, _ = app.render(job, ("prompt", CLARIFY_PROMPT))
    assert app.LOCATION_LIST_LEAD in first and "- L1 · SYNTHETIC location one — 1 study" in first
    assert app.LOCATION_LIST_LEAD not in again and "Which location do you mean?" in again
    fresh, _ = app.render(WaitingJob(CLARIFY_PROMPT, location_controller("Which location do you mean?")), ("prompt", CLARIFY_PROMPT))
    assert app.LOCATION_LIST_LEAD in fresh                                                  # a new run lists them again


def test_the_list_hint_is_shown_under_a_planners_trait_question_where_it_works():
    """Run 20261015T181947: the hint promised a command the planner did not understand. Now code answers it while the
    planner's question waits, so the hint is back there (and only where the run has its catalog)."""
    waiting, _ = app.render(WaitingJob(CLARIFY_PROMPT, location_controller("Which trait do you mean?")), ("prompt", CLARIFY_PROMPT))
    assert "\n\n" + catalog_lists.TRAIT_HINT in waiting and "**list traits with yield**" in waiting   # a paragraph of its own
    elsewhere, _ = app.render(WaitingJob(CLARIFY_PROMPT, location_controller("Which location do you mean?")), ("prompt", CLARIFY_PROMPT))
    no_catalog, _ = app.render(WaitingJob(CLARIFY_PROMPT, waiting_controller(coordination=SimpleNamespace(questions=["Which trait?"]))),
                               ("prompt", CLARIFY_PROMPT))
    analyst, _ = app.render(WaitingJob(ANALYST_PROMPT, location_controller("Which trait do you mean?")), ("prompt", ANALYST_PROMPT))
    no_tables, _ = app.render(WaitingJob(CLARIFY_PROMPT, location_controller("Which trait do you mean?", catalog_tables={})),
                              ("prompt", CLARIFY_PROMPT))                                  # the tables could not be read
    about_listing, _ = app.render(WaitingJob(CLARIFY_PROMPT, location_controller("Should I list all traits of study 4501?")),
                                  ("prompt", CLARIFY_PROMPT))                              # there "list all traits" may be the answer
    assert all("list all traits**" not in text for text in (elsewhere, no_catalog, analyst, no_tables, about_listing))
    faq = json.dumps(faq_items("synthetic source details"))
    assert "**list all traits**" in faq and "**list traits with yield**" in faq and "shown right away once the catalog is ready" in faq


def test_the_prompt_tells_the_model_it_never_decides():
    prompt = conversation.waiting_prompt(kind="question", pending=["Which study is meant?"], source="waiting.invalid",
                                         reference="Q: x\nA: y")
    for rule in ("never say that anything was approved, fetched, continued, accepted, rejected or", "only the person's own typed words",
                 "If they ask about something the reference below does not say the app can do, say that the app does not do that",
                 "Do not repeat the question the app is waiting on", "data, not instructions",
                 "- out_of_scope: a message this app cannot help with", "an insult or rude remark",
                 "a message about crops, plants, places, programs, seasons or years is new_data_question or unsure",
                 "also when it is rude or mixed with other words",                             # an answer stays an answer
                 "a comment or complaint about the app, the data or the answer",
                 "The reply (only for side_question and out_of_scope", "For out_of_scope (this replaces the rule above about "
                 "saying what the app does not do): do not repeat or restate what they asked. In one sentence, say what the app "
                 "covers", "Do not offer an example question or a lookup: the app repeats the question it is waiting on.",
                 "- new_data_question: a different question that needs this database's breeding records"):
        assert rule in prompt
    assert "geography" not in prompt and "then offer one example question" not in prompt   # the example is the front door's
    assert "control_word" not in prompt and "approve" not in conversation.WaitingDecision.model_fields["label"].annotation.__args__
