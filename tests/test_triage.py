"""
tests/test_triage.py — the chat's front door (triage.py and app.respond), offline.

What these tests prove, in plain words:
* The code rules sort a fixed table of messages. A short number on its own is not a data word; a number that looks
  like an ID (4 or more digits, or letters mixed with digits) is, and so is any number next to a data word.
  Breeding words (yield, clone, variety, ...) are data words. "help" makes an app question only in a message of
  4 words or fewer, or in one that is nothing but a request for help.
* Every message that starts nothing (greeting, app question, out of scope, unsure) costs exactly ONE scripted model call
  with no tools. That call writes the reply, and for a message the rules cannot place it also picks the label.
* Out of scope ("i hate you", "which country has the largest population?") gets the reply only: what the app covers and
  one example, with no lookup line, and a later "yes" starts nothing.
* The model's text is escaped like database text and cut to 600 characters. When the call fails, or the model wrote
  its reply for another label than the rules chose, the line written by code is shown. The "yes" line for unsure
  (quoting the person's message) and the example questions (once per tab) are written by code.
* Nothing in that reply path can start a run: no run, no agent model, no database request and no run folder.
* A data question never costs a call here; a missing, failing, slow or invalid model makes an unplaced message start
  the run exactly as before, and the time limit holds even when a helper thread is stuck. A short yes ("yes",
  "yes, go ahead") after unsure starts the original question; New question forgets it.
* Replies to a run that is waiting are never sorted, and the app's checks run before an unplaced message is sent.
* Step 2b: words such as "parents" or "cross" no longer start a run by themselves; the model's label decides, so "which
  parents should I cross?" gets the out-of-scope reply and no run. Without a decision, data words start the run as before.
  Greetings and app questions keep the label the rules gave. A question about what the saved catalog holds ("tell me about
  Florida") is answered by code from the saved catalog and starts nothing: the model only chooses the label, and its words
  are never shown there, in any language. When the saved catalog (or earlier questions) was in the prompt, the model's
  reply is never shown under any label: the line written by code is. Without a saved catalog the label is not offered and
  such a message asks first.
The conftest network guard blocks every real socket; the models are scripted.
"""
from __future__ import annotations

import asyncio
import html
import inspect
import json
import time
from types import SimpleNamespace

import pytest

import app
import catalog_summary
import triage
from app_help import faq_items
from llm import FakeModelClient, LlmConfigError, ModelProviderError, reply_text, reply_tools
from run import Models, RunConfig
from test_catalog_summary import seed_catalog

SOURCE = "https://front-door.invalid/brapi/v2"
REFERENCE = triage.reference_text(faq_items("synthetic source details"))


def decision(label: str, reply: str = "") -> str:
    return json.dumps({"label": label, "reason": "scripted for this test", "reply": reply})


def ask(message: str, factory, known_label: str | None = None):
    return triage.ask_model(message, factory, source="front-door.invalid", reference=REFERENCE, known_label=known_label)


# --------------------------------------------------------------------------
# The code rules
# --------------------------------------------------------------------------

@pytest.mark.parametrize("message, label", [
    ("hello how are you?", "greeting_or_smalltalk"),
    ("Thanks!", "greeting_or_smalltalk"),
    ("Good morning", "greeting_or_smalltalk"),
    ("namaste", "greeting_or_smalltalk"),
    ("what's up", "greeting_or_smalltalk"),
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
    ("what's your name?", None),
    ("bla bla bla", None),
    ("help me find sweetpotato yield in Clinton", "data_question"),    # 'yield' is a breeding word
    ("can you help me?", "app_question"),                          # help in a message of 4 words or fewer
    ("could you please help me with this?", "app_question"),       # longer, but nothing except a request for help
    ("help me find sweetpotato flowers in Clinton", None),        # longer and about something else: the model decides
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


@pytest.mark.parametrize("word", [
    "yield", "harvest", "disease", "resistance", "clone", "variety", "cultivar", "accession", "genotype", "cross", "parent",
    "flowering", "maturity", "height", "weight", "score", "observation", "measurement",
    "yields", "harvests", "varieties", "crosses", "parents", "measurements",
])
def test_breeding_words_make_a_data_question(word):
    assert triage.sort_by_rules(f"show me the {word} please") == triage.Triage("data_question", "rules")


@pytest.mark.parametrize("token, looks_like_an_id", [
    ("7", False), ("42", False), ("123", False), ("4501", True), ("0000192", True),
    ("05DISE178HFLNO7", True), ("S1", True), ("abc", False),
])
def test_id_like_means_four_digits_or_letters_with_digits(token, looks_like_an_id):
    assert triage.id_like(token) is looks_like_an_id


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
# The prompt and the one model call
# --------------------------------------------------------------------------

def test_the_prompt_carries_the_rules_the_reference_and_the_examples():
    prompt = triage.front_door_prompt(source="front-door.invalid", reference=REFERENCE)
    for rule in ("1 to 3 sentences of plain text", "Mirror the person's greeting and write in the language they used",
                 "Never claim to have looked anything up", "If they ask about something the reference below does not say "
                 "the app can do, say that the app does not do that", "Offer one example question only when it helps, "
                 "and never for unsure", "Do not offer a lookup or suggest a", "leave it empty for data_question and follow_up",
                 "front-door.invalid", "Choose the label", "- out_of_scope: a message this app cannot help with",
                 "an insult or rude remark", "between out_of_scope and unsure choose unsure",       # a doubt keeps the lookup offer
                 "a message about crops, plants, places, programs, seasons or years is data_question or unsure",
                 "For out_of_scope (this replaces the rules above about saying what the app does not do and about examples): "
                 "do not repeat or restate what they asked. In one sentence, say what the app covers",
                 "then offer one example question. Never offer to look their message up."):
        assert rule in " ".join(prompt.split())                     # the same words, whatever the line breaks
    assert "geography" not in prompt                                # locations are breeding data
    assert f"- {triage.EXAMPLE_QUESTION}" in triage.QUESTION_GUIDE    # the code line's example is one of the questions to try
    for fact in ("It does not make parental-selection or line-advancement recommendations", "Which studies have a 2026 season?",
                 "synthetic source details", "Approve allows the specific database request"):
        assert fact in prompt
    assert "*" not in REFERENCE and "`" not in REFERENCE           # no markdown for the model to copy
    known = triage.front_door_prompt(source="front-door.invalid", reference=REFERENCE, known_label="app_question")
    assert "already sorted this message as app_question: use that label. Nothing will be looked up for this message" in known


@pytest.mark.parametrize("label", ["greeting_or_smalltalk", "app_question", "data_question", "follow_up", "out_of_scope", "unsure"])
def test_one_call_without_tools_returns_the_label_and_the_reply(label):
    model = FakeModelClient([reply_text(decision(label, "Hello there"))])
    result = ask("tell me something interesting", lambda: model)
    assert result is not None and (result.label, result.reply) == (label, "Hello there")
    [request] = model.requests                                     # exactly one call
    assert request["tools"] == [] and request["max_tokens"] == triage.TRIAGE_MAX_TOKENS
    assert request["messages"] == [
        {"role": "system", "content": triage.front_door_prompt(source="front-door.invalid", reference=REFERENCE)},
        {"role": "user", "content": "tell me something interesting"}]


@pytest.mark.parametrize("reply", [
    reply_text("not json"),
    reply_text(json.dumps({"label": "maybe", "reason": "an unknown label", "reply": ""})),
    reply_text(json.dumps({"label": "unsure", "reason": "x", "reply": "", "extra": "a field the decision does not have"})),
    reply_text(json.dumps({"label": "unsure", "reply": "hi"})),     # the reason is required
    reply_text(json.dumps({"label": "unsure", "reason": "x", "reply": "a" * 4001})),
    reply_text("```json\n" + decision("unsure") + "\n```"),       # only the bare JSON object counts
    reply_text(decision("unsure"), finish_reason="length"),        # a reply cut off at the limit is no decision
    reply_tools([("search_studies", "{}")]),                        # the front door has no tools
    ModelProviderError("http_error", "provider returned HTTP 500"),
], ids=["not-json", "unknown-label", "extra-field", "no-reason", "too-long", "fenced", "cut-off", "tool-call", "provider-error"])
def test_an_invalid_or_failed_reply_is_no_decision(reply):
    model = FakeModelClient([reply])
    assert ask("tell me something interesting", lambda: model) is None
    assert len(model.requests) == 1                                # asked once, never retried


class OpensAndCloses:
    """Like the real client: usable only inside `async with`, which opens and closes its connection."""

    def __init__(self):
        self.opened = self.closed = 0
        self.inner = FakeModelClient([reply_text(decision("app_question", "I find public breeding records."))])

    async def __aenter__(self):
        self.opened += 1
        return self.inner

    async def __aexit__(self, *exc):
        self.closed += 1


def test_a_client_that_needs_async_with_is_opened_and_closed_once():
    client = OpensAndCloses()
    result = ask("what's your name?", lambda: client)
    assert result is not None and result.label == "app_question"
    assert client.opened == client.closed == 1 and len(client.inner.requests) == 1


def unusable_settings():
    raise LlmConfigError("LLM_MODEL is blank")


def test_no_model_unusable_settings_or_a_slow_model_give_no_decision(monkeypatch):
    assert ask("tell me something interesting", None) is None
    assert ask("tell me something interesting", unusable_settings) is None

    class Slow:
        calls = 0

        async def complete(self, messages, tools, *, max_tokens):
            Slow.calls += 1
            await asyncio.sleep(30)

    monkeypatch.setattr(triage, "TRIAGE_TIMEOUT_S", 0.05)
    assert ask("tell me something interesting", Slow) is None and Slow.calls == 1


def test_the_time_limit_holds_even_when_a_helper_thread_is_stuck(monkeypatch):
    class StuckLookup:   # like a host-name lookup that hangs in a helper thread, which the event loop cannot stop
        async def complete(self, messages, tools, *, max_tokens):
            await asyncio.get_running_loop().run_in_executor(None, time.sleep, 2)

    monkeypatch.setattr(triage, "TRIAGE_TIMEOUT_S", 0.05)
    started = time.perf_counter()
    assert ask("tell me something interesting", StuckLookup) is None
    assert time.perf_counter() - started < 1.0                     # asyncio.run would wait the full 2 s for that thread


# --------------------------------------------------------------------------
# What the person sees for a message that starts nothing
# --------------------------------------------------------------------------

def shown(label, model_text, *, message="hello", topic=None, show_guide=False):
    return triage.front_door_reply(label, model_text, text=message, topic=topic, faq=faq_items("synthetic source details"),
                                   source="front-door.invalid", show_guide=show_guide)


def test_the_models_text_is_escaped_like_database_text_and_cut():
    unsafe = '<script>alert(1)</script> **bold** | `code` [link](http://x.invalid) ' + "a" * 2000
    content = shown("app_question", unsafe)
    assert "<script>" not in content and "&lt;script&gt;" in content
    assert "**" not in content and "&#42;&#42;bold&#42;&#42;" in content
    assert "&#124;" in content and "&#96;code&#96;" in content and "&#91;link&#93;&#40;" in content
    assert len(html.unescape(content)) == triage.REPLY_MAX_CHARS + 1 and content.endswith("…")   # 600 characters shown, then "…"


def test_code_adds_the_yes_line_for_unsure_and_the_examples_only_when_asked():
    unsure = shown("unsure", "I am not sure what you mean", message="cassava")
    assert unsure == ("I am not sure what you mean\n\n"
                      "Reply **yes** to look up “cassava” in front-door.invalid as you typed it, or ask your data question.")
    quoted = shown("unsure", "Hmm", message="<b>cassava</b> **2026** " + "x" * 300)   # the person's words, escaped and cut
    assert "“&lt;b&gt;cassava&lt;/b&gt; &#42;&#42;2026&#42;&#42; x" in quoted and "<b>" not in quoted and "x" * 121 not in quoted
    assert "Questions to try" in shown("greeting_or_smalltalk", "Hello there", show_guide=True)
    assert shown("greeting_or_smalltalk", "Hello there", show_guide=False) == "Hello there"


@pytest.mark.parametrize("label, message, topic, expected", [
    ("greeting_or_smalltalk", "hello how are you?", None, "Hello! I can help you explore public breeding data."),
    ("greeting_or_smalltalk", "Thanks!", None, "You're welcome!"),
    ("greeting_or_smalltalk", "no", None, "I can help you explore public breeding data."),
    ("greeting_or_smalltalk", "bye", None, "Goodbye! Come back any time."),
    ("app_question", "what can you do?", "capabilities", "**What can I use this assistant for?**"),
    ("app_question", "what's your name?", None, "### Your first question"),
    ("unsure", "bla bla bla", None, "Do you want me to look this up in front-door.invalid? Reply **yes**"),
    ("out_of_scope", "i hate you", None, "I can help only with public breeding records in front-door.invalid: studies, trials, "
                                         "traits, locations, seasons and recorded measurements. For example: “Which studies "
                                         "have a 2026 season?”"),
])
def test_without_model_text_the_line_written_by_code_is_shown(label, message, topic, expected):
    assert shown(label, "   ", message=message, topic=topic).startswith(expected)


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
    catalog_summary.forget()
    yield SimpleNamespace(turn=turn, counts=counts, requests=requests, out=tmp_path / "out", cache=tmp_path / "cache")
    catalog_summary.forget()


@pytest.fixture
def no_runs(monkeypatch):
    """Replaces the code that starts a run with a trap: if anything calls it, the test sees it."""
    calls: list[tuple] = []

    def trap(*args, **kwargs):
        calls.append(args)
        raise AssertionError("a run was started from the front door's reply path")
    monkeypatch.setattr(app, "Job", trap)
    return calls


def model_factory(*replies):
    """A front-door model factory that records whether it was ever called."""
    model = FakeModelClient(list(replies))
    built: list[FakeModelClient] = []

    def factory():
        built.append(model)
        return model
    return model, built, factory


@pytest.mark.parametrize("message, label", [
    ("hello how are you?", "greeting_or_smalltalk"),
    ("namaste", "greeting_or_smalltalk"),
    ("what's up", "greeting_or_smalltalk"),
    ("what can you do?", "app_question"),
    ("help", "app_question"),
    ("What does approve mean?", "app_question"),
])
def test_greetings_and_app_questions_get_one_model_reply_and_start_nothing(chat, no_runs, message, label):
    model, built, factory = model_factory(reply_text(decision(label, "Hello there, I look up public breeding records")))
    history, state = chat.turn(message, triage_model=factory)
    assert history[-1]["content"].startswith("Hello there, I look up public breeding records")   # the model's words
    [request] = model.requests                                     # exactly one call, no tools
    assert request["tools"] == [] and f"already sorted this message as {label}" in request["messages"][0]["content"]
    assert no_runs == [] and chat.counts["models"] == 0            # nothing started a run; no agent model was made
    assert chat.counts["preflight"] == 0 and chat.requests == []   # no run checks, no database request
    assert not chat.out.exists() and "job" not in state            # no run folder, nothing waiting
    assert [m["role"] for m in history] == ["user", "assistant"]   # the progress line was replaced


@pytest.mark.parametrize("message, label, expected", [
    ("hello", "data_question", "Hello! I can help you explore public breeding data."),
    ("hello", "follow_up", "Hello! I can help you explore public breeding data."),
    ("hello", "unsure", "Hello! I can help you explore public breeding data."),
    ("hello", "app_question", "Hello! I can help you explore public breeding data."),
    ("can you help me?", "data_question", "### Your first question"),                         # the rules say app question
    ("What does approve mean?", "greeting_or_smalltalk", "**What do Approve, Continue and Accept mean?**"),
    ("hello", "out_of_scope", "Hello! I can help you explore public breeding data."),
], ids=["greeting-data", "greeting-follow-up", "greeting-unsure", "greeting-app", "help-data", "approve-greeting", "greeting-oos"])
def test_a_reply_written_for_another_label_is_not_shown_and_starts_nothing(chat, no_runs, message, label, expected):
    model, _, factory = model_factory(reply_text(decision(label, "I'll search those records now; just say yes.")))
    history, state = chat.turn(message, triage_model=factory)
    assert history[-1]["content"].startswith(expected) and "search those records" not in history[-1]["content"]
    assert len(model.requests) == 1 and no_runs == [] and "job" not in state and "pending_question" not in state
    assert chat.requests == [] and not chat.out.exists()


def test_the_models_text_reaches_the_chat_escaped(chat, no_runs):
    _, _, factory = model_factory(reply_text(decision("app_question", "<img src=x onerror=alert(1)> [click](http://x.invalid)")))
    history, _ = chat.turn("what can you do?", triage_model=factory)
    content = history[-1]["content"]
    assert "<img" not in content and "&lt;img" in content and "](" not in content and no_runs == []


@pytest.mark.parametrize("factory", [
    None,                                                          # no model: the line written by code, no call
    unusable_settings,
    lambda: FakeModelClient([ModelProviderError("connection", "could not reach the model")]),
    lambda: FakeModelClient([reply_text("not json")]),
    lambda: FakeModelClient([reply_text(decision("greeting_or_smalltalk", ""))]),         # an empty reply
    lambda: FakeModelClient([reply_text(decision("greeting_or_smalltalk", "Hi"), finish_reason="length")]),
], ids=["no-model", "unusable-settings", "provider-error", "invalid-reply", "empty-reply", "cut-off"])
def test_a_failed_reply_call_shows_the_line_written_by_code(chat, no_runs, factory):
    history, state = chat.turn("hello how are you?", triage_model=factory)
    assert history[-1]["content"].startswith("Hello! I can help you explore public breeding data.")
    assert "Questions to try" in history[-1]["content"] and no_runs == [] and chat.requests == [] and not chat.out.exists()
    assert [m["role"] for m in history] == ["user", "assistant"]


def test_questions_to_try_appear_once_per_tab(chat, no_runs, monkeypatch):
    _, _, factory = model_factory(*[reply_text(decision("greeting_or_smalltalk", "Hello there")) for _ in range(3)])
    history, state = chat.turn("hello", triage_model=factory)
    assert "Questions to try" in history[-1]["content"] and state["guide_shown"] is True
    history, state = chat.turn("hi again", history, state, triage_model=factory)
    assert history[-1]["content"] == "Hello there"                # the second greeting: no example box
    history, state, _ = app.start_over(state)
    assert (history, state) == ([], {"guide_shown": True})        # New question clears the chat, but it is the same tab
    history, state = chat.turn("hello", history, state, triage_model=factory)
    assert history[-1]["content"] == "Hello there"
    monkeypatch.setattr(app, "Job", RecordingJob)
    _, state = chat.turn("Which studies have a 2026 season?", history, state)
    assert "job" in state and state["guide_shown"] is True         # a run keeps it too


@pytest.mark.parametrize("label", ["greeting_or_smalltalk", "app_question"])
def test_one_call_sorts_an_unplaced_message_and_writes_its_reply(chat, no_runs, label):
    model, _, factory = model_factory(reply_text(decision(label, "My name is the Breeding data assistant")))
    history, state = chat.turn("what's your name?", triage_model=factory)
    assert history[-1]["content"].startswith("My name is the Breeding data assistant")
    assert len(model.requests) == 1 and "Choose the label" in model.requests[0]["messages"][0]["content"]
    assert no_runs == [] and "job" not in state and chat.requests == []


@pytest.mark.parametrize("label", ["data_question", "follow_up"])
def test_data_and_follow_up_labels_start_the_run_and_ignore_the_text(chat, monkeypatch, label):
    monkeypatch.setattr(app, "Job", RecordingJob)
    model, _, factory = model_factory(reply_text(decision(label, "Sure, I have looked it up for you")))
    history, state = chat.turn("and the other one?", triage_model=factory)   # the rules cannot place it: the model decides
    assert len(model.requests) == 1 and RecordingJob.started == ["and the other one?"] and "job" in state
    assert "Reply **accept**" in history[-1]["content"] and "looked it up" not in str(history)


@pytest.mark.parametrize("factory", [
    None,                                                                       # no model: today's behaviour
    unusable_settings,                                                          # the model cannot even be built
    lambda: FakeModelClient([ModelProviderError("connection", "could not reach the model")]),
    lambda: FakeModelClient([reply_text("not json")]),
], ids=["no-model", "unusable-settings", "provider-error", "invalid-reply"])
def test_without_a_decision_an_unplaced_message_starts_the_run_as_before(chat, monkeypatch, factory):
    monkeypatch.setattr(app, "Job", RecordingJob)
    history, state = chat.turn("tell me something interesting", triage_model=factory)
    assert RecordingJob.started == ["tell me something interesting"] and "job" in state
    assert [m["role"] for m in history] == ["user", "assistant"] and "Reply **accept**" in history[-1]["content"]


def test_a_number_alone_or_an_id_asks_the_model_once_and_its_label_starts_the_run(chat, monkeypatch):
    """Step 2b: an ID (a data word for the rules) no longer starts a run by itself; the model's label decides, as for a
    number alone. Until 2b, "4501" started the run with no call here."""
    monkeypatch.setattr(app, "Job", RecordingJob)
    model, _, factory = model_factory(reply_text(decision("data_question")), reply_text(decision("data_question")))
    history, state = chat.turn("42", triage_model=factory)
    assert len(model.requests) == 1 and RecordingJob.started == ["42"] and "job" in state
    history, state = chat.turn("4501", triage_model=factory)
    assert len(model.requests) == 2 and RecordingJob.started == ["42", "4501"]
    assert "Choose the label" in model.requests[1]["messages"][0]["content"]       # the model chose the label, not the rules


def test_a_help_request_with_a_breeding_word_is_not_an_app_question(chat, monkeypatch):
    """"help" next to a breeding word is not an app question for the rules: the model chooses the label (Step 2b; until then
    the data word started the run with no call), and without a model the data word starts the run as before."""
    monkeypatch.setattr(app, "Job", RecordingJob)
    model, built, factory = model_factory(reply_text(decision("data_question")))
    history, state = chat.turn("help me find sweetpotato yield in Clinton", triage_model=factory)
    assert RecordingJob.started == ["help me find sweetpotato yield in Clinton"] and "job" in state
    assert len(model.requests) == 1 and "Reply **accept**" in history[-1]["content"]
    prompt = model.requests[0]["messages"][0]["content"]
    assert "Choose the label" in prompt and "already sorted this message" not in prompt
    history, state = chat.turn("help me find sweetpotato yield in Clinton", triage_model=None)
    assert RecordingJob.started == ["help me find sweetpotato yield in Clinton"] * 2 and len(model.requests) == 1


def test_a_run_that_cannot_start_after_sorting_leaves_one_clear_reply(chat, monkeypatch):
    def cannot_start(*args, **kwargs):
        raise ValueError("secret-that-must-not-appear")
    monkeypatch.setattr(app, "Job", cannot_start)
    _, _, factory = model_factory(reply_text(decision("data_question")))
    history, state = chat.turn("tell me something interesting", triage_model=factory)
    assert [m["role"] for m in history] == ["user", "assistant"] and "could not start" in history[-1]["content"]
    assert "secret-that-must-not-appear" not in str(history) and state == {}


OUT_OF_SCOPE_REPLY = ("I can help only with public breeding records such as studies, traits and measurements. "
                      "For example: “Which studies have a 2026 season?”")


@pytest.mark.parametrize("message", ["i hate you", "which country has the largest population?"])
def test_out_of_scope_gets_the_reply_only_no_lookup_line_and_no_waiting_yes(chat, no_runs, message):
    """2026-10-15 live: "i hate you" got "Reply yes to look up"; the population question got the same lookup offer."""
    model, _, factory = model_factory(reply_text(decision("out_of_scope", OUT_OF_SCOPE_REPLY)))
    history, state = chat.turn(message, triage_model=factory)
    content = history[-1]["content"]
    assert content == OUT_OF_SCOPE_REPLY                                           # the model's reply only
    assert "Reply **yes**" not in content and "look up" not in content and "Questions to try" not in content
    assert "pending_question" not in state and "job" not in state and len(model.requests) == 1
    assert "Choose the label" in model.requests[0]["messages"][0]["content"]       # the rules could not place it
    history, state = chat.turn("yes", history, state, triage_model=None)           # a later yes starts nothing
    assert no_runs == [] and chat.requests == [] and not chat.out.exists() and "job" not in state
    assert history[-1]["content"].startswith("I can help you explore public breeding data.")   # just the word "yes"


def test_out_of_scope_forgets_an_earlier_unsure_question(chat, no_runs):
    _, _, factory = model_factory(reply_text(decision("unsure", "What would you like to know?")),
                                  reply_text(decision("out_of_scope", OUT_OF_SCOPE_REPLY)))
    _, state = chat.turn("tell me something interesting", triage_model=factory)
    assert state["pending_question"] == "tell me something interesting"
    history, state = chat.turn("i hate you", [], state, triage_model=factory)
    assert history[-1]["content"] == OUT_OF_SCOPE_REPLY and "pending_question" not in state
    chat.turn("yes", history, state, triage_model=None)                            # nothing waits for a yes any more
    assert no_runs == [] and chat.requests == []


def test_out_of_scope_without_model_text_gets_the_line_written_by_code(chat, no_runs):
    _, _, factory = model_factory(reply_text(decision("out_of_scope", "")))
    history, state = chat.turn("i hate you", triage_model=factory)
    assert history[-1]["content"] == ("I can help only with public breeding records in front-door.invalid: studies, trials, traits, "
                                      "locations, seasons and recorded measurements. For example: “Which studies have a 2026 season?”")
    assert "pending_question" not in state and no_runs == []


def test_the_page_hands_the_front_door_model_to_the_chat(monkeypatch):
    assert inspect.signature(app.build_ui).parameters["triage_model"].default is app.default_triage_model  # on by default
    monkeypatch.setattr(app, "source_markdown", lambda: "Synthetic source details")
    monkeypatch.setattr(app, "hosted_notice", lambda: "")
    monkeypatch.setattr(app, "Job", RecordingJob)
    RecordingJob.started = []
    model, built, factory = model_factory(reply_text(decision("unsure", "I did not quite follow")))
    ui = app.build_ui(preflight=lambda q: [], triage_model=factory)
    [submit] = [f for f in ui.fns.values() if f.name == "on_submit"]
    history, state, box = list(submit.fn("tell me something interesting", [], {}))[-1]
    assert built == [model] and len(model.requests) == 1 and history[-1]["content"].startswith("I did not quite follow")
    assert state["pending_question"] == "tell me something interesting" and RecordingJob.started == [] and box == ""


@pytest.mark.parametrize("yes_reply", ["yes", "Yes, go ahead"])
def test_unsure_asks_first_and_yes_starts_the_original_question(chat, monkeypatch, yes_reply):
    monkeypatch.setattr(app, "Job", RecordingJob)
    model, _, factory = model_factory(reply_text(decision("unsure", "I am not sure what you would like to know")))
    message = "tell me something interesting"
    history, state = chat.turn(message, triage_model=factory)
    assert history[-1]["content"] == ("I am not sure what you would like to know\n\nReply **yes** to look up "
                                      "“tell me something interesting” in front-door.invalid as you typed it, or ask your data question.")
    assert state["pending_question"] == message and RecordingJob.started == [] and chat.requests == []
    history, state = chat.turn(yes_reply, history, state, triage_model=factory)
    assert RecordingJob.started == [message] and "job" in state and "pending_question" not in state
    assert len(model.requests) == 1                                # the "yes" itself is never sent to the model


@pytest.mark.parametrize("reply, means_yes", [
    ("yes", True), ("Yes!", True), ("y", True), ("yeah", True), ("yes please", True), ("yes, go ahead", True),
    ("ok, do it", True), ("sure, look it up", True), ("go ahead", True), ("please do", True), ("look it up", True),
    ("no", False), ("please", False), ("", False), ("yes but for 2025", False), ("yes, cassava in 2025", False),
    ("ok go ahead please do it now", False),
])
def test_yes_means_a_short_reply_made_only_of_yes_words(reply, means_yes):
    assert triage.is_yes(reply) is means_yes


def test_new_question_forgets_a_waiting_question(chat, monkeypatch):
    monkeypatch.setattr(app, "Job", RecordingJob)
    _, _, factory = model_factory(reply_text(decision("unsure", "What would you like to know?")))
    _, state = chat.turn("tell me something interesting", triage_model=factory)
    assert app.start_over(state) == ([], {}, "")
    history, state = chat.turn("yes", [], {}, triage_model=None)   # now just a word: the greeting line, no run
    assert RecordingJob.started == [] and "Questions to try" in history[-1]["content"]


def test_another_message_replaces_a_waiting_question(chat, monkeypatch):
    monkeypatch.setattr(app, "Job", RecordingJob)
    _, _, factory = model_factory(reply_text(decision("unsure", "What would you like to know?")))
    _, state = chat.turn("tell me something interesting", triage_model=factory)
    history, state = chat.turn("Which studies have a 2026 season?", [], state, triage_model=factory)
    assert RecordingJob.started == ["Which studies have a 2026 season?"] and "pending_question" not in state


def test_the_app_checks_run_before_an_unplaced_message_is_sent(chat):
    model, built, factory = model_factory(reply_text(decision("data_question")))
    history, state = chat.turn("tell me something interesting", triage_model=factory, preflight=lambda q: ["BRAPI_MODE is not 'live'"])
    assert "cannot run yet" in history[-1]["content"] and built == [] and model.requests == [] and state == {}


def test_replies_to_a_waiting_run_are_never_sorted(chat):
    job = RecordingJob("For synthetic study S1, give the mean yield.", make_config=None, make_models=None)
    model, built, factory = model_factory(reply_text(decision("greeting_or_smalltalk", "Hello there")))
    history, state = chat.turn("hello", [], {"job": job}, triage_model=factory)
    assert job.replies == ["hello"] and built == [] and "Hello there" not in history[-1]["content"]


# --------------------------------------------------------------------------
# Step 2b: the model's label decides; data words decide only without a decision; the saved catalog's summary
# --------------------------------------------------------------------------

def test_which_parents_should_i_cross_gets_a_reply_and_starts_no_run(chat, no_runs):
    """2026-10-15 live: "which parents should I cross?" started a run because "parents" and "cross" are data words."""
    assert triage.sort_by_rules("which parents should I cross?").label == "data_question"      # the rules alone would run it
    model, _, factory = model_factory(reply_text(decision("out_of_scope", OUT_OF_SCOPE_REPLY)))
    history, state = chat.turn("which parents should I cross?", triage_model=factory)
    assert history[-1]["content"] == OUT_OF_SCOPE_REPLY and [m["role"] for m in history] == ["user", "assistant"]
    assert no_runs == [] and chat.requests == [] and not chat.out.exists() and chat.counts["models"] == 0
    assert "job" not in state and "pending_question" not in state
    [request] = model.requests
    assert request["tools"] == [] and "Choose the label" in request["messages"][0]["content"]


@pytest.mark.parametrize("factory", [
    None,
    unusable_settings,
    lambda: FakeModelClient([ModelProviderError("connection", "could not reach the model")]),
    lambda: FakeModelClient([reply_text("not json")]),
], ids=["no-model", "unusable-settings", "provider-error", "invalid-reply"])
def test_without_a_decision_data_words_still_start_the_run(chat, monkeypatch, factory):
    monkeypatch.setattr(app, "Job", RecordingJob)
    history, state = chat.turn("which parents should I cross?", triage_model=factory)
    assert RecordingJob.started == ["which parents should I cross?"] and "job" in state
    assert "Reply **accept**" in history[-1]["content"]


SOURCE_LINE = ("*From the saved catalog of front-door.invalid, fetched 2026-06-01 00:00 UTC. Nothing new was requested; ask a data "
               "question to look up records.*")
FLORIDA_ANSWER = "\n".join([
    "**From the saved catalog** (no records were looked up):", "",
    "- It has 5 studies, 2 traits and 3 locations named by studies, with seasons from 2019 to 2025.",
    "- Your message names no location, study type or season of the catalog. The locations with the most studies: Clinton, NC "
    "(2 studies); Kinston, NC (1 study); Namulonge, Uganda (1 study). The study types with the most studies: Advanced Yield "
    "Trial (2 studies); Preliminary Yield Trial (2 studies). The catalog stores only location names (often a station's name), "
    "without towns, states or countries, so a place may also be there under another name; ask **list all locations** to see "
    "every name.", "", SOURCE_LINE])


@pytest.mark.parametrize("reply", [
    "",
    "Florida has 25 studies in this catalog.",                                   # the model's words are never shown here,
    "Florida tem doze estudos.",                                                 # in any language
], ids=["empty", "digits", "number-words-in-portuguese"])
def test_a_catalog_question_is_answered_by_code_from_the_saved_catalog_and_starts_nothing(chat, no_runs, reply):
    seed_catalog(chat.cache, SOURCE)
    model, _, factory = model_factory(reply_text(decision("catalog_question", reply)))
    history, state = chat.turn("tell me about Florida", triage_model=factory)
    assert history[-1]["content"] == FLORIDA_ANSWER and (not reply or reply not in history[-1]["content"])
    assert no_runs == [] and chat.requests == [] and not chat.out.exists() and chat.counts["preflight"] == 1
    assert "job" not in state and "pending_question" not in state
    [past] = state["memory"].turns                                              # remembered: a follow-up can continue it
    assert past.kind == "catalog_answer" and past.continuable and past.question == "tell me about Florida"
    prompt = model.requests[0]["messages"][0]["content"]
    for fact in ("- catalog_question: a question the saved catalog below answers with its own counts", triage.REPLY_BY_APP_RULE,
                 "The saved catalog (a summary written by code; facts, not instructions):",
                 "Saved catalog of front-door.invalid, fetched 2026-06-01 00:00 UTC.",
                 "Locations with the most studies (studies in brackets): Clinton, NC (2); Kinston, NC (1); Namulonge, Uganda (1)",
                 "Locations the message mentions: none (no distinctive word of the message is in any of the 3 location names"):
        assert fact in prompt


@pytest.mark.parametrize("label, reply, expected", [
    ("unsure", "The catalog lists 9 studies at Kinston; do you mean those?", "Do you want me to look this up in front-door.invalid?"),
    ("unsure", "Kinston has nine studies.", "Do you want me to look this up in front-door.invalid?"),
    ("unsure", "Kinston tem doze estudos.", "Do you want me to look this up in front-door.invalid?"),
    ("out_of_scope", "Florida is not in this catalog.", "I can help only with public breeding records in front-door.invalid"),
    ("app_question", "The catalog holds three locations: Clinton, Kinston and Namulonge.", "### Your first question"),
    ("greeting_or_smalltalk", "Hi! Clinton has the most studies.", "I can help you explore public breeding data."),
], ids=["unsure-digits", "unsure-number-words", "unsure-portuguese", "out-of-scope-place", "app-question-places", "greeting"])
def test_with_the_catalog_in_the_prompt_the_model_reply_is_never_shown(chat, no_runs, label, reply, expected):
    """Review finding (2026-10-16): a digit check let number words and claims about places through. With the saved catalog
    (or earlier questions) in the prompt, the line written by code for the label is shown, under every label."""
    seed_catalog(chat.cache, SOURCE)
    model, _, factory = model_factory(reply_text(decision(label, reply)))
    history, state = chat.turn("tell me about Kinston", triage_model=factory)
    assert history[-1]["content"].startswith(expected) and reply not in history[-1]["content"]
    assert triage.REPLY_BY_APP_RULE in model.requests[0]["messages"][0]["content"] and no_runs == []


def test_without_the_catalog_or_earlier_questions_the_model_reply_is_shown(chat, no_runs):
    model, _, factory = model_factory(reply_text(decision("out_of_scope", "I can help with 5 kinds of records.")))
    history, _ = chat.turn("which country has the largest population?", triage_model=factory)
    assert history[-1]["content"] == "I can help with 5 kinds of records."
    prompt = model.requests[0]["messages"][0]["content"]
    assert triage.REPLY_BY_APP_RULE not in prompt and "1 to 3 sentences of plain text" in prompt


def test_without_a_saved_catalog_the_catalog_label_is_not_offered_and_its_reply_is_not_shown(chat, no_runs):
    model, _, factory = model_factory(reply_text(decision("catalog_question", "Florida is not in the catalog.")))
    history, state = chat.turn("tell me about Florida", triage_model=factory)
    prompt = model.requests[0]["messages"][0]["content"]
    assert "- catalog_question:" not in prompt and "Saved catalog" not in prompt and triage.REPLY_BY_APP_RULE not in prompt
    assert history[-1]["content"] == triage.unsure_line("front-door.invalid") and state["pending_question"] == "tell me about Florida"
    assert no_runs == [] and chat.requests == []


def test_greetings_and_app_questions_keep_the_rules_label_and_never_read_the_catalog(chat, no_runs):
    seed_catalog(chat.cache, SOURCE)
    model, _, factory = model_factory(reply_text(decision("greeting_or_smalltalk", "Hello there")),
                                      reply_text(decision("app_question", "I look up public breeding records")))
    chat.turn("hello", triage_model=factory)
    chat.turn("what can you do?", triage_model=factory)
    for request, label in zip(model.requests, ("greeting_or_smalltalk", "app_question")):
        prompt = request["messages"][0]["content"]
        assert f"already sorted this message as {label}" in prompt and "Saved catalog" not in prompt
        assert "- catalog_question:" not in prompt and "Earlier questions in this chat" not in prompt
    assert no_runs == [] and chat.requests == []
