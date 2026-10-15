"""
tests/test_triage.py — the chat's front door (triage.py and app.respond), offline.

What these tests prove, in plain words:
* The code rules sort a fixed table of messages. A short number on its own is not a data word; a number that looks
  like an ID (4 or more digits, or letters mixed with digits) is, and so is any number next to a data word.
* A message the rules cannot place costs exactly one scripted model call with no tools. A missing, failing, slow or
  invalid model reply means the run starts exactly as before this front door existed.
* Greetings and app questions make ZERO model calls: the sorting model is never built, no agent model is made,
  no run starts, no database request is made and no run folder appears.
* "unsure" asks first and starts nothing; a later "yes" starts the original question; New question forgets it.
* Replies to a run that is waiting are never sorted, and the app's checks run before the sorting model is asked.
The conftest network guard blocks every real socket; the models are scripted.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import time
from types import SimpleNamespace

import pytest

import app
import triage
from app_help import faq_items
from llm import FakeModelClient, LlmConfigError, ModelProviderError, reply_text, reply_tools
from run import Models, RunConfig

SOURCE = "https://front-door.invalid/brapi/v2"


def decision(label: str) -> str:
    return json.dumps({"label": label, "reason": "scripted for this test"})


# --------------------------------------------------------------------------
# The code rules
# --------------------------------------------------------------------------

@pytest.mark.parametrize("message, label", [
    ("hello how are you?", "greeting_or_smalltalk"),
    ("Thanks!", "greeting_or_smalltalk"),
    ("Good morning", "greeting_or_smalltalk"),
    ("what can you do?", "app_question"),
    ("help", "app_question"),
    ("What does approve mean?", "app_question"),                  # 'mean' here is about the app, not a statistic
    ("What is the meaning of accept?", "app_question"),
    ("Show the mean yield for study 4501 and continue", "data_question"),
    ("Where do the data come from?", "app_question"),
    ("Which studies have a 2026 season?", "data_question"),
    ("How many locations are in the catalog?", "data_question"),
    ("In study 05DISE178HFLNO7, what is the pooled mean of 'Reaction to Fusarium Wilt estimating 0-5'?", "data_question"),
    ("What is the mean yield?", "data_question"),
    ("Can you help me find the mean yield?", "data_question"),     # the data word decides before the word "help"
    ("Happy new year to you!", "greeting_or_smalltalk"),          # 'year' is a data word, but this message is only a greeting
    ("hello, which studies have a 2026 season?", "data_question"),  # a greeting in front of a question is still a question
    ("tell me something interesting", None),                       # the rules cannot tell: the model decides
    ("question", None),
])
def test_rules_sort_a_fixed_table_of_messages(message, label):
    sorted_message = triage.sort_by_rules(message)
    assert (sorted_message.label if sorted_message else None) == label
    assert sorted_message is None or sorted_message.decided_by == "rules"


@pytest.mark.parametrize("message, label", [
    ("42", None),                    # a bare number alone is not a data word: the model decides
    ("7", None),
    ("and 25?", None),
    ("2 + 2", None),
    ("4501", "data_question"),       # 4 or more digits looks like an ID
    ("2026", "data_question"),
    ("05DISE178HFLNO7", "data_question"),  # letters mixed with digits looks like an ID
    ("S1", "data_question"),
    ("plot 12", "data_question"),    # a short number next to a data word: the data word decides
    ("study 7", "data_question"),
])
def test_a_bare_number_is_not_a_data_word_but_an_id_is(message, label):
    sorted_message = triage.sort_by_rules(message)
    assert (sorted_message.label if sorted_message else None) == label


@pytest.mark.parametrize("token, looks_like_an_id", [
    ("7", False), ("42", False), ("123", False), ("4501", True), ("0000192", True),
    ("05DISE178HFLNO7", True), ("S1", True), ("abc", False),
])
def test_id_like_means_four_digits_or_letters_with_digits(token, looks_like_an_id):
    assert triage.id_like(token) is looks_like_an_id


@pytest.mark.parametrize("message, first_line", [
    ("hello how are you?", "Hello! I can help you explore public breeding data."),
    ("Thanks!", "You're welcome!"),
    ("no", "I can help you explore public breeding data."),
])
def test_the_friendly_line_fits_the_message_and_the_examples_follow(message, first_line):
    reply = triage.greeting_reply(message)
    assert reply.split("\n", 1)[0] == first_line and "Questions to try" in reply


@pytest.mark.parametrize("message", [
    "thanks so much " * 24 + "x",           # took 14 s with the first draft's single regular expression
    "thanks so much " * 2000 + "x",
    "approve " * 5000 + "x",
    "hello " * 10000,
], ids=["thanks-x24", "thanks-x2000", "approve-x5000", "hello-x10000"])   # short names: pytest puts the name in an environment variable
def test_long_or_tricky_messages_are_sorted_quickly(message):
    started = time.perf_counter()
    triage.sort_by_rules(message)
    assert time.perf_counter() - started < 1.0


def test_every_app_answer_names_a_question_of_the_help_tab():
    titles = dict(faq_items("synthetic source details"))
    assert set(triage.FAQ_TITLES.values()) <= set(titles)


# --------------------------------------------------------------------------
# The one model call
# --------------------------------------------------------------------------

@pytest.mark.parametrize("label", ["greeting_or_smalltalk", "app_question", "data_question", "follow_up", "unsure"])
def test_the_model_is_asked_once_without_tools(label):
    model = FakeModelClient([reply_text(decision(label))])
    result = triage.ask_model("tell me something interesting", lambda: model)
    assert result is not None and result.label == label
    [request] = model.requests                                     # exactly one call
    assert request["tools"] == [] and request["max_tokens"] == triage.TRIAGE_MAX_TOKENS
    assert request["messages"] == [{"role": "system", "content": triage.SYSTEM_PROMPT},
                                   {"role": "user", "content": "tell me something interesting"}]


@pytest.mark.parametrize("reply", [
    reply_text("not json"),
    reply_text(json.dumps({"label": "maybe", "reason": "an unknown label"})),
    reply_text(json.dumps({"label": "unsure", "reason": "x", "extra": "a field the decision does not have"})),
    reply_text(json.dumps({"label": "unsure"})),                  # the reason is required
    reply_text("```json\n" + decision("unsure") + "\n```"),     # only the bare JSON object counts
    reply_text(decision("unsure"), finish_reason="length"),      # a reply cut off at the limit is no decision
    reply_tools([("search_studies", "{}")]),                      # the sorter has no tools
    ModelProviderError("http_error", "provider returned HTTP 500"),
])
def test_an_invalid_or_failed_reply_is_no_decision(reply):
    model = FakeModelClient([reply])
    assert triage.ask_model("tell me something interesting", lambda: model) is None
    assert len(model.requests) == 1                                # asked once, never retried


class OpensAndCloses:
    """Like the real client: usable only inside `async with`, which opens and closes its connection."""

    def __init__(self):
        self.opened = self.closed = 0
        self.inner = FakeModelClient([reply_text(decision("app_question"))])

    async def __aenter__(self):
        self.opened += 1
        return self.inner

    async def __aexit__(self, *exc):
        self.closed += 1


def test_a_client_that_needs_async_with_is_opened_and_closed_once():
    client = OpensAndCloses()
    result = triage.ask_model("tell me something interesting", lambda: client)
    assert result is not None and result.label == "app_question"
    assert client.opened == client.closed == 1 and len(client.inner.requests) == 1


def test_no_model_unusable_settings_or_a_slow_model_give_no_decision(monkeypatch):
    assert triage.ask_model("tell me something interesting", None) is None

    def unusable():
        raise LlmConfigError("LLM_MODEL is blank")
    assert triage.ask_model("tell me something interesting", unusable) is None

    class Slow:
        calls = 0

        async def complete(self, messages, tools, *, max_tokens):
            Slow.calls += 1
            await asyncio.sleep(30)

    monkeypatch.setattr(triage, "TRIAGE_TIMEOUT_S", 0.05)
    assert triage.ask_model("tell me something interesting", Slow) is None and Slow.calls == 1


# --------------------------------------------------------------------------
# In the chat
# --------------------------------------------------------------------------

class RecordingJob:
    """Stands in for app.Job: records the question a run would start with and waits at the answer (no thread, no run)."""

    started: list[str] = []

    def __init__(self, question, *, make_config, make_models):
        RecordingJob.started.append(question)
        self.controller = SimpleNamespace(run_id="recorded-run", answer_markdown=lambda: "recorded report")
        self.answer_shown = False
        self.waiting_on = ""
        self.replies: list[str] = []

    def start(self):
        return self

    def reply(self, text):
        self.replies.append(text)

    def next_event(self, timeout=None):
        self.waiting_on = "accept / reject"
        return "prompt", "accept / reject"


@pytest.fixture
def chat(tmp_path, monkeypatch):
    """Factories that count what they are asked for, a database transport that records any request, and a run folder."""
    import brapi_client as bc

    monkeypatch.setenv("BRAPI_BASE_URL", SOURCE)
    requests: list[object] = []
    monkeypatch.setattr(bc, "RequestsTransport", lambda: requests.append)
    counts = {"models": 0, "preflight": 0}

    def make_config(question):
        return RunConfig(question=question, out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")

    def make_models():
        counts["models"] += 1
        return Models()

    def preflight(question):
        counts["preflight"] += 1
        return []

    def turn(message, history=None, state=None, **options):
        out = list(app.respond(message, history or [], state or {},
                               **{"make_config": make_config, "make_models": make_models, "preflight": preflight, **options}))
        assert out[-1][2] == ""
        return out[-1][0], out[-1][1]

    RecordingJob.started = []
    return SimpleNamespace(turn=turn, counts=counts, requests=requests, out=tmp_path / "out")


def sorter(*replies):
    """A sorting-model factory that records whether it was ever called."""
    model = FakeModelClient(list(replies))
    built: list[FakeModelClient] = []

    def factory():
        built.append(model)
        return model
    return model, built, factory


@pytest.mark.parametrize("message, shown", [
    ("hello how are you?", "Questions to try"),
    ("Thanks!", "You're welcome!"),
    ("what can you do?", "What can I use this assistant for?"),
    ("help", "Your first question"),
    ("What does approve mean?", "**Approve** allows the specific database request"),
    ("Where do the data come from?", SOURCE),
])
def test_greetings_and_app_questions_make_zero_model_calls(chat, message, shown):
    model, built, factory = sorter(reply_text(decision("data_question")))
    history, state = chat.turn(message, triage_model=factory)     # the real app.Job: a wrong path would make models and a run
    assert shown in history[-1]["content"]
    assert built == [] and model.requests == []                   # the sorting model was never even built
    assert chat.counts["models"] == 0                             # no run, so no agent model either
    assert chat.counts["preflight"] == 0 and chat.requests == []  # no run checks, no database request
    assert not chat.out.exists() and "job" not in state           # no run folder, nothing waiting
    assert [m["role"] for m in history] == ["user", "assistant"]


def test_a_number_alone_asks_the_model_once_but_an_id_starts_the_run_directly(chat, monkeypatch):
    monkeypatch.setattr(app, "Job", RecordingJob)
    model, built, factory = sorter(reply_text(decision("data_question")))
    history, state = chat.turn("42", triage_model=factory)
    assert len(model.requests) == 1 and RecordingJob.started == ["42"] and "job" in state
    assert [m["role"] for m in history] == ["user", "assistant"]  # one reply: the progress line was replaced
    history, state = chat.turn("4501", triage_model=factory)
    assert len(model.requests) == 1 and RecordingJob.started == ["42", "4501"]


@pytest.mark.parametrize("label, shown", [
    ("greeting_or_smalltalk", "Questions to try"),
    ("app_question", "Your first question"),
])
def test_model_labels_without_data_get_a_code_written_reply_and_start_nothing(chat, monkeypatch, label, shown):
    monkeypatch.setattr(app, "Job", RecordingJob)
    model, _, factory = sorter(reply_text(decision(label)))
    history, state = chat.turn("tell me something interesting", triage_model=factory)
    assert shown in history[-1]["content"] and RecordingJob.started == [] and "job" not in state
    assert len(model.requests) == 1 and chat.requests == []


@pytest.mark.parametrize("label", ["data_question", "follow_up"])
def test_data_and_follow_up_labels_start_the_run_with_the_message(chat, monkeypatch, label):
    monkeypatch.setattr(app, "Job", RecordingJob)
    _, _, factory = sorter(reply_text(decision(label)))
    history, state = chat.turn("and in that trial?", triage_model=factory)
    assert RecordingJob.started == ["and in that trial?"] and "job" in state
    assert "Reply **accept**" in history[-1]["content"]


def unusable_settings():
    raise LlmConfigError("LLM_MODEL is blank")


@pytest.mark.parametrize("factory", [
    None,                                                                       # no sorting model: today's behaviour
    unusable_settings,                                                          # the model cannot even be built
    lambda: FakeModelClient([ModelProviderError("connection", "could not reach the model")]),
    lambda: FakeModelClient([reply_text("not json")]),
], ids=["no-model", "unusable-settings", "provider-error", "invalid-reply"])
def test_without_a_decision_the_run_starts_as_before(chat, monkeypatch, factory):
    monkeypatch.setattr(app, "Job", RecordingJob)
    history, state = chat.turn("tell me something interesting", triage_model=factory)
    assert RecordingJob.started == ["tell me something interesting"] and "job" in state
    assert [m["role"] for m in history] == ["user", "assistant"] and "Reply **accept**" in history[-1]["content"]


def test_a_run_that_cannot_start_after_sorting_leaves_one_clear_reply(chat, monkeypatch):
    def cannot_start(*args, **kwargs):
        raise ValueError("secret-that-must-not-appear")
    monkeypatch.setattr(app, "Job", cannot_start)
    _, _, factory = sorter(reply_text(decision("data_question")))
    history, state = chat.turn("tell me something interesting", triage_model=factory)
    assert [m["role"] for m in history] == ["user", "assistant"] and "could not start" in history[-1]["content"]
    assert "secret-that-must-not-appear" not in str(history) and state == {}


def test_the_page_hands_the_sorting_model_to_the_chat(monkeypatch):
    assert inspect.signature(app.build_ui).parameters["triage_model"].default is app.default_triage_model  # on by default
    monkeypatch.setattr(app, "source_markdown", lambda: "Synthetic source details")
    monkeypatch.setattr(app, "hosted_notice", lambda: "")
    monkeypatch.setattr(app, "Job", RecordingJob)
    RecordingJob.started = []
    model, built, factory = sorter(reply_text(decision("unsure")))
    ui = app.build_ui(preflight=lambda q: [], triage_model=factory)
    [submit] = [f for f in ui.fns.values() if f.name == "on_submit"]
    history, state, box = list(submit.fn("tell me something interesting", [], {}))[-1]
    assert built == [model] and len(model.requests) == 1
    assert state["pending_question"] == "tell me something interesting" and RecordingJob.started == [] and box == ""


def test_unsure_asks_first_and_yes_starts_the_original_question(chat, monkeypatch):
    monkeypatch.setattr(app, "Job", RecordingJob)
    model, _, factory = sorter(reply_text(decision("unsure")))
    message = "tell me something interesting"
    history, state = chat.turn(message, triage_model=factory)
    assert "Do you want me to look this up in front-door.invalid? Reply **yes**" in history[-1]["content"]
    assert state["pending_question"] == message and RecordingJob.started == [] and chat.requests == []
    history, state = chat.turn("yes", history, state, triage_model=factory)
    assert RecordingJob.started == [message] and "job" in state and "pending_question" not in state
    assert len(model.requests) == 1                                # "yes" itself is never sent to the model


def test_new_question_forgets_a_waiting_question(chat, monkeypatch):
    monkeypatch.setattr(app, "Job", RecordingJob)
    _, _, factory = sorter(reply_text(decision("unsure")))
    _, state = chat.turn("tell me something interesting", triage_model=factory)
    assert app.start_over(state) == ([], {}, "")
    history, state = chat.turn("yes", [], {}, triage_model=factory)
    assert RecordingJob.started == [] and "Questions to try" in history[-1]["content"]


def test_another_message_replaces_a_waiting_question(chat, monkeypatch):
    monkeypatch.setattr(app, "Job", RecordingJob)
    _, _, factory = sorter(reply_text(decision("unsure")))
    _, state = chat.turn("tell me something interesting", triage_model=factory)
    history, state = chat.turn("Which studies have a 2026 season?", [], state, triage_model=factory)
    assert RecordingJob.started == ["Which studies have a 2026 season?"] and "pending_question" not in state


def test_the_app_checks_run_before_the_sorting_model_is_asked(chat):
    model, built, factory = sorter(reply_text(decision("data_question")))
    history, state = chat.turn("tell me something interesting", triage_model=factory, preflight=lambda q: ["BRAPI_MODE is not 'live'"])
    assert "cannot run yet" in history[-1]["content"] and built == [] and model.requests == [] and state == {}


def test_replies_to_a_waiting_run_are_never_sorted(chat):
    job = RecordingJob("For synthetic study S1, give the mean yield.", make_config=None, make_models=None)
    model, built, factory = sorter(reply_text(decision("greeting_or_smalltalk")))
    history, state = chat.turn("hello", [], {"job": job}, triage_model=factory)
    assert job.replies == ["hello"] and built == [] and "Questions to try" not in history[-1]["content"]
