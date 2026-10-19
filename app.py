"""Shared chat interface for BreedAPI Chat Agent on Hugging Face Spaces.

app_space.py selects agent-led coordination, the shared MCP service, and human
review before retrieval and analysis. Each browser session has its own run;
the Full report retains sources, calculations, answers and review decisions.
A new message first passes the front door (triage.py): greetings and questions
about the app are answered without starting a run, and the model's label decides
whether any other message starts one; a message that starts nothing is answered
by a model call that never sees the saved catalog (Step 2c), or by code. A
message typed while a run waits is
sorted by conversation.py before anything reaches the run. "list all traits" (and
locations, seasons, programs, study types) is answered by code from the cached
catalog (catalog_lists.py): as a short run of its own, or, while a run waits, from
that run's catalog; the front door's catalog_list label starts the same short run
for a question in other words ("Which seasons are in the catalog?"). The tab
remembers its earlier questions (chat_memory.py), so a follow-up such as "and in
2025?" continues the last one with its own approval, and a question about what
the catalog holds is answered from the saved catalog's summary
(catalog_summary.py) without a run.
"""
from __future__ import annotations

import asyncio
import dataclasses
from datetime import datetime, timedelta, timezone
import os
import math
import re
import queue
import threading
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import urlsplit

import catalog_lists
import catalog_summary
import chat_memory
import conversation
import triage
from app_style import CSS, HERO_HTML, build_theme
from app_help import QUICK_START, QUESTION_GUIDE, faq_items
from chat_presenter import answer_summary, approval_summary
from brapi_client import load_settings
from llm import ModelProviderError, ProviderConfigError
from run import (ANALYST_PROMPT, CLARIFY_PROMPT, CATALOG_PROMPT, CATALOG_SETUP_QUESTION, DATA_REVIEW_PROMPT, RETRIEVER_PROMPT, SUPERVISOR_PROMPT,
                 Controller, Models, RunConfig, analyst_questions, source_host, standing_review_date)

__all__ = ["Job", "live_config", "live_preflight", "respond", "render", "build_ui", "REPLY_TIMEOUT_S", "locked", "after_step",
           "reply_buttons", "start_over", "default_triage_model", "prepare_catalogs", "catalog_source_line"]

REPLY_TIMEOUT_S = 15 * 60          # how long a run waits for your reply in the chat before it cancels itself
STEP_TIMEOUT_S = 15 * 60           # how long the page waits for the run's next step (model calls, fetches)
WORKING = "Working on your question…"
PLACEHOLDER = ("**Start with a question.**\n\nFind a study, count locations, or explore a trait in one trial.")
CHAT_CLARIFICATION_ROUNDS = 3      # how many times a person may answer the planner in the chat (the command line allows 1)
SHOWN_QUESTION_CHARS = 600         # a planner question can list hundreds of variables; the chat shows the start, the notes keep all
CHAT_CONCURRENCY_ID = "chat_steps"
CHAT_QUEUE_SIZE = 20
TABLE_ROWS_SHOWN = 15              # rows of a long table shown in the chat; answer.md keeps every row
EXAMPLES = (                       # the first was answered live on 2026-09-29 (run_20260929T183240_3aec99); the others are catalog questions
    "In study 05DISE178HFLNO7, what is the pooled mean of the variable 'Reaction to Fusarium Wilt estimating 0-5'?",
    "Which studies have a 2026 season?",
    "How many locations are in the catalog?",
)
Message = dict[str, str]           # {"role": "user" | "assistant", "content": markdown}


class Job:
    """One question: its controller runs in a background thread and talks to the chat through two queues.

    inbox  : the person's replies, typed in the chat ("approve", "cancel", "accept", ...)
    outbox : what the run needs next — ("prompt", text), ("done", RunResult) or ("error", message)
    """

    def __init__(self, question: str, *, make_config: Callable[[str], RunConfig], make_models: Callable[[], Models],
                 reply_timeout_s: float = REPLY_TIMEOUT_S) -> None:
        self.inbox: queue.Queue[str] = queue.Queue()
        self.outbox: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.reply_timeout_s = reply_timeout_s
        self.answer_shown = False
        self.locations_listed = False            # the planner's location list is shown once per run, not on every re-ask
        self.waiting_on = ""                     # the prompt the run is waiting on; set by the run's thread before it is shown
        self.controller = Controller(make_config(question), models=make_models(), answer_fn=self._ask, answers_from_human=True)
        self.thread = threading.Thread(target=self._run, name=f"job-{self.controller.run_id}", daemon=True)

    def _ask(self, prompt: str) -> str:
        self.waiting_on = prompt
        self.outbox.put(("prompt", prompt))
        try:
            return self.inbox.get(timeout=self.reply_timeout_s)
        except queue.Empty:
            raise EOFError("no reply in the chat within the time limit") from None   # the controller treats it as no answer

    def _run(self) -> None:
        try:
            self.outbox.put(("done", asyncio.run(self.controller.run())))
        except BaseException as exc:  # noqa: BLE001 - every ending reaches the page, none is lost
            self.outbox.put(("error", f"{type(exc).__name__}: {exc}"))

    def start(self) -> "Job":
        self.thread.start()
        return self

    def reply(self, text: str) -> None:
        if self.waiting_on == CATALOG_PROMPT and text.strip().lower() == "approve":
            text = "approve catalog"
        if self.waiting_on == DATA_REVIEW_PROMPT and text.strip().lower() == "approve":
            text = "continue"               # the first quick-reply button is labelled Continue to analysis here
        if text.strip().lower() == "cancel" and self.waiting_on in (CLARIFY_PROMPT, ANALYST_PROMPT, RETRIEVER_PROMPT, SUPERVISOR_PROMPT):
            text = ""                            # there the controller's rule is 'empty = cancel'; a chat box cannot send an empty line
        self.inbox.put(text)

    def next_event(self, timeout: float = STEP_TIMEOUT_S) -> tuple[str, Any]:
        try:
            return self.outbox.get(timeout=timeout)
        except queue.Empty:
            return ("error", f"no progress within {timeout:.0f} s; the run keeps its evidence under out/{self.controller.run_id}")

    def close(self) -> None:
        """The tab went away: release a waiting prompt so the run ends (as canceled or pending) instead of waiting."""
        self.reply("cancel")


# --------------------------------------------------------------------------
# What a live question needs, and how its steps are shown
# --------------------------------------------------------------------------

def _app_number(name: str, default: str, low: float, high: float, *, integer: bool = False):
    """Refuse bad deployment settings without echoing their values."""
    try:
        value = int(os.environ.get(name, default)) if integer else float(os.environ.get(name, default))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a {'whole number' if integer else 'number'} from {low:g} to {high:g}") from None
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a {'whole number' if integer else 'number'} from {low:g} to {high:g}")
    return value


def _app_flag(name: str) -> bool:
    """An on/off deployment setting: "true" or "false" (any case); unset or empty is off. Anything else is refused, and its
    value is not echoed."""
    value = os.environ.get(name, "").strip().lower()
    if value in ("", "false"):
        return False
    if value == "true":
        return True
    raise ValueError(f"{name} must be true or false")


def live_config(question: str) -> RunConfig:
    return RunConfig(question=question, offline=False, mock_model=False, direct=False, auto=False,
                     max_fetches=5, max_http_attempts=_app_number("APP_MAX_HTTP_ATTEMPTS", "10", 1, 100, integer=True),
                     max_clarification_rounds=CHAT_CLARIFICATION_ROUNDS,
                     read_timeout=_app_number("APP_READ_TIMEOUT", "90", 1, 600), max_elapsed_seconds=600,
                     bootstrap_catalogs=True, catalog_standing_approval=_app_flag("APP_CATALOG_STANDING_APPROVAL"),
                     catalog_max_age_hours=_app_number("APP_CATALOG_MAX_AGE_HOURS", "24", 1, 8760))


def _never_asked(prompt: str) -> str:
    raise RuntimeError("a catalog setup at startup never asks a person")


def prepare_catalogs(make_config: Callable[[str], RunConfig] = live_config, *,
                     make_models: Callable[[], Models] | None = None) -> str:
    """At app startup (app_space.main), with APP_CATALOG_STANDING_APPROVAL on (Part D): refuse to start while the access
    notes carry no review date; when the two catalogs are not completely saved, prepare them now under the operator's
    standing approval, as a recorded run with no planner, no model call and no measurements. With the setting off nothing
    happens here: a visitor approves the catalog setup when a question needs it, as before. Returns one line for the log."""
    off = "catalog standing approval: off (a visitor approves the catalog setup when a question needs it)"
    try:
        config = make_config(CATALOG_SETUP_QUESTION)
    except ValueError as exc:                                   # a bad setting: say which, never its value
        try:
            standing = _app_flag("APP_CATALOG_STANDING_APPROVAL")
        except ValueError:
            standing = True                                     # the setting itself is bad
        if standing:
            raise SystemExit(f"refusing to start: {exc}") from None
        return off + "; another app setting is invalid, so each question will say so, as before"
    if not config.catalog_standing_approval:
        return off
    if standing_review_date(config.access_notes) is None:
        raise SystemExit("refusing to start: APP_CATALOG_STANDING_APPROVAL is on, but the access notes have no valid "
                         "'Reviewed by Bidhan on: YYYY-MM-DD' line (a real date, not in the future); review them first, "
                         "or turn the setting off")
    if catalog_summary.from_cache(config) is not None:
        return "catalog standing approval: on; the study and trait catalogs are already saved, nothing was requested"
    controller = Controller(dataclasses.replace(config, catalog_setup_only=True, bootstrap_catalogs=True),
                            models=(make_models or default_models)(), answer_fn=_never_asked)
    result = asyncio.run(controller.run())
    catalog_summary.forget()                                     # the next summary reads the new catalogs
    return f"catalog standing approval: on; catalog setup {result.execution_status} (run {controller.run_id})"


def default_models() -> Models:
    return Models(requested="configured model")


def default_triage_model():
    """The configured model for the front door's one call. Built only for a message that does not go straight to a run."""
    from llm import load_llm_settings, make_model_client

    return make_model_client(load_llm_settings())


def live_preflight(question: str, make_config: Callable[[str], RunConfig] = live_config) -> list[str]:
    """The command line's refusals, said in the chat before any model call or request."""
    try:
        problems = make_config(question).validate()
    except (TypeError, ValueError, OverflowError):
        return ["App settings are invalid. Check APP_MAX_HTTP_ATTEMPTS, APP_READ_TIMEOUT, APP_CATALOG_STANDING_APPROVAL, "
                "APP_CATALOG_MAX_AGE_HOURS and the BrAPI settings."]
    try:
        from llm import is_loopback, load_llm_settings

        settings = load_llm_settings()
        if not settings.model:
            problems.append("LLM_MODEL is missing; set the model name before starting a question")
        if (getattr(settings, "provider", "openai") == "openai" and not is_loopback(settings.base_url)
                and settings.api_key.strip().lower() in ("", "ollama")):           # Claude's key is checked with its settings
            problems.append("LLM_API_KEY must be set to a real key for this remote-model chat")
    except ProviderConfigError as exc:  # a provider setting's own reason: it names settings, never their values
        problems.append(str(exc))
    except Exception:  # a model endpoint that cannot be used is a stated reason, not a crash
        problems.append("the model endpoint cannot be used: check LLM_BASE_URL, LLM_MODEL, LLM_API_KEY and LLM_ALLOW_REMOTE")
    return problems


LOCATION_LIST_LEAD = "Locations with the most studies:"
SOURCE_FAQ = "Where do the data come from?"          # the help answer that carries source_markdown


def clarification_choices(questions: list[str], metadata: Any, *, list_locations: bool = True, lists_work: bool = False) -> str:
    """Hints from validated catalogs, only displayed; never added to the person's words. Under a question about a trait, the
    hint to ask "list all traits", where code answers it on the spot (lists_work: the run has its catalog tables; and a
    question that is itself about listing leaves such a reply to the model, so it gets no hint)."""
    text = " ".join(questions).lower()
    lines = []
    if list_locations and re.search(r"\blocations?\b", text) and metadata is not None:
        locations = {}
        for study in metadata.studies:
            if not study.location_id:
                continue
            item = locations.setdefault(study.location_id, {"names": set(), "studies": set()})
            item["studies"].add(study.study_id)
            if study.location_name:
                item["names"].add(study.location_name)
        ranked = sorted(locations.items(), key=lambda item: (-len(item[1]["studies"]), item[0]))[:10]
        if ranked:
            lines += ["", f"{LOCATION_LIST_LEAD} These are popular database locations, not confirmed matches "
                      "for the place you requested. Choosing one changes the location scope of your question."]
            for location_id, item in ranked:
                count = len(item["studies"])
                name = " / ".join(sorted(item["names"])) or "(name not stated)"
                lines.append(f"- {location_id} · {name} — {count} {'study' if count == 1 else 'studies'}")
    if (lists_work and metadata is not None and re.search(r"\b(traits?|variables?)\b", text)
            and not conversation.asks_about_listing(questions)):
        lines += ["", "", catalog_lists.TRAIT_HINT]           # its own paragraph, not inside the quoted question above
    return "\n".join(lines)


def render(job: Job, event: tuple[str, Any]) -> tuple[str, bool]:
    """(markdown for the chat, finished?) for one step of the run."""
    kind, payload = event
    controller = job.controller
    if kind == "prompt" and payload == CATALOG_PROMPT:
        return (controller.catalog_approval_screen() + "\n\nReply **approve** to prepare these catalogs, or **cancel**. "
                "Measurements require a separate approval afterward."), False
    if kind == "prompt" and str(payload).startswith("approve / edit / cancel"):
        return (approval_summary(controller) + "\n\nReply **approve** to use this scope, or **cancel**. "
                "You can also type **edit** to change the study IDs, or describe a change in your own words. "
                "A change produces a new plan and asks for approval again before fetching."), False
    if kind == "prompt" and str(payload).startswith("accept / reject"):
        job.answer_shown = True
        return (answer_summary(controller) + "\n\n---\nReply **accept** if this answers your question, or **reject**. "
                "The Full report below keeps the complete evidence."), False
    if kind == "prompt" and payload == DATA_REVIEW_PROMPT:
        return (controller.data_review_screen() + "\n\nReply **continue** to analyze these data, or **cancel**. "
                "Continuing allows analysis; you will review and accept the final answer separately."), False
    if kind == "prompt" and payload == SUPERVISOR_PROMPT:
        asked = "\n".join("> " + _shorten(" ".join(q.split())) for q in controller.supervisor_questions)
        return ("I need one more detail before choosing the next step:\n\n" + asked +
                "\n\nReply in the chat, or cancel. Any revised data request still needs your approval."), False
    if kind == "prompt" and payload == RETRIEVER_PROMPT:
        asked = [_shorten(" ".join(q.split())) for q in controller.retriever_questions if q and q.strip()]
        shown = "\n".join(f"> {q}" for q in asked) or "> (no question was given; reply cancel to stop)"
        number = controller.retriever_answers + 1
        return (f"I need one more detail while preparing your data (answer {number} of "
                f"{controller.config.max_clarification_rounds}):\n\n{shown}\n\n"
                "Your answer resumes retrieval within the approved scope. To change that scope, "
                "reply **cancel** and start a new question."), False
    if kind == "prompt" and payload in (CLARIFY_PROMPT, ANALYST_PROMPT):
        planner = payload == CLARIFY_PROMPT
        questions = (controller.coordination.questions if controller.coordination else []) if planner else analyst_questions(controller.analysis)
        asked = [_shorten(" ".join(q.split())) for q in questions if q and q.strip()]
        shown = "\n".join(f"> {q}" for q in asked) or "> (no question was given; say in a few more words what you mean)"
        number = (getattr(controller, "clarification_answers", 0) if planner else controller.analyst_answers) + 1  # the cap's own count
        if planner:
            who = "I need one more detail before I can make a plan"
            hint = "Your answer is added to your question, so a few words are enough, for example **study 4501** or **all studies from 2026**."
            scope = getattr(getattr(controller, "plan", None), "scope", None)
            if not getattr(scope, "study_ids", []):
                hint += (" Each new question starts a separate analysis. If you mean an earlier study or summary, "
                         "name its study and trait, or paste the relevant summary. Replies here clarify this question.")
        else:
            who = "I have the data and need one more detail before I can answer"
            hint = "Your answer is used with the data I already have; nothing new is fetched."
        if planner:
            choices = clarification_choices(questions, controller.metadata, list_locations=not getattr(job, "locations_listed", False),
                                            lists_work=bool(getattr(controller, "catalog_tables", None)))
            if LOCATION_LIST_LEAD in choices:
                job.locations_listed = True
            shown += choices
        return f"{who} (answer {number} of {controller.config.max_clarification_rounds}):\n\n{shown}\n\n{hint} Or reply **cancel**.", False
    if kind == "prompt":
        return str(payload).strip(), False
    if kind == "done":
        result = payload
        listing = getattr(controller, "catalog_list", None)
        if listing is not None and getattr(result, "execution_status", None) == "completed":   # a list run: the list itself;
            return catalog_lists.chat_message(listing, in_report=True), True                     # no accept or reject is asked
        if (getattr(getattr(controller, "config", None), "catalog_setup_only", False)          # a refresh: what the catalogs
                and getattr(result, "execution_status", None) == "completed"):                  # hold now
            fetched_now = any(r.get("origin") == "live" and not r.get("error") for r in controller.catalog_requests)
            return catalog_lists.setup_message(controller.catalog_tables, fetched_now=fetched_now,
                                               again=bool(getattr(controller, "catalog_set_aside", None))), True
        if not job.answer_shown:
            return answer_summary(controller), True
        review = {"accepted": "**Accepted.** Your review is recorded.",
                  "rejected": "**Rejected.** Your review is recorded; you can start a new question.",
                  "pending": "**Review pending.** This answer has not been accepted."}
        return review[result.human_acceptance] + " The Full report is available below.", True
    return f"The run stopped: {payload}", True


def _shorten_tables(markdown: str, keep: int | None = None) -> str:
    """The chat shows the first rows of a long table (136 location rows were one screen after another, 2026-09-29); answer.md
    keeps every row. A table here is a run of lines that start with '|': a header, a separator, then data rows."""
    keep = TABLE_ROWS_SHOWN if keep is None else keep
    out: list[str] = []
    rows = hidden = 0
    in_table = False
    for line in markdown.splitlines() + [""]:
        if line.startswith("|"):
            if not in_table:
                in_table, rows, hidden = True, -2, 0                      # the header and the separator are not data rows
            rows += 1
            if rows <= keep:
                out.append(line)
            else:
                hidden += 1
            continue
        if in_table and hidden:
            out += ["", f"*… and {hidden} more rows: all of them are in answer.md and the Full report panel below.*"]
        in_table = False
        out.append(line)
    return "\n".join(out[:-1])


def _shorten(text: str) -> str:
    if len(text) <= SHOWN_QUESTION_CHARS:
        return text
    return text[:SHOWN_QUESTION_CHARS].rstrip() + " … (shortened here; the run's notes keep the whole question)"


def _faq() -> list[tuple[str, str]]:
    try:
        return faq_items(source_markdown(), data_sharing_policy=hosted_notice())
    except Exception:  # a settings problem must not break a help answer
        return faq_items("The assistant reads public records from the configured breeding database.")


def _source_name() -> str:
    try:
        return source_host(load_settings().base_url) or "the breeding database"
    except Exception:
        return "the breeding database"


def _ask_front_door(text: str, triage_model: Callable[[], Any], *, catalog: catalog_summary.CatalogSummary | None = None,
                    earlier: str = "") -> triage.TriageDecision | None:
    return triage.ask_model(text, triage_model, source=_source_name(), reference=triage.reference_text(_faq()),
                            catalog=catalog.for_message(text) if catalog is not None else "", earlier=earlier)


def _said(content: Any) -> str | None:
    """The words of one chat message: a string, or a list of text blocks, which is how Gradio hands the chat's history back
    ([{"type": "text", "text": "hello"}]). None for anything else (a file, an image)."""
    if isinstance(content, str):
        return content
    blocks = content if isinstance(content, (list, tuple)) else [content]
    texts = [block.get("text") for block in blocks if isinstance(block, dict) and block.get("type") == "text"]
    texts = [words for words in texts if isinstance(words, str)]
    return " ".join(texts) if texts else None


def _earlier_words(history: list[Message], text: str) -> list[str]:
    """The person's own earlier messages in this tab, oldest first. Not the last one (the message just typed) and no copy of
    the message being answered: after "drop" that is the new question, typed before it (compared without spacing, which
    Gradio changes: tabs become spaces, indents go). The app's answers are never included."""
    said = [_said(m.get("content")) for m in history if isinstance(m, dict) and m.get("role") == "user"]
    answered = " ".join(text.split())
    return [words for words in said[:-1] if words is not None and " ".join(words.split()) != answered]


def _ask_to_talk(text: str, triage_model: Callable[[], Any], label: str, history: list[Message],
                 state: dict[str, Any]) -> triage.TalkReply | None:
    """The talk call: the reply to every message that starts nothing and has a talk label (sorted by the rules or by the
    label call, with or without a saved catalog or earlier questions: "front door, one path", 2026-10-19), written with no
    saved catalog, none of the app's earlier questions and no answer of the app in view; only the message, the person's
    own last messages, the help reference, whether the tab already said what the app covers, and whether a reply of the
    app is already in this chat (_introduced). None: the line written by code is shown (triage.ask_to_talk says when)."""
    return triage.ask_to_talk(text, triage_model, label=label, source=_source_name(), reference=triage.reference_text(_faq()),
                              earlier_words=_earlier_words(history, text), coverage_said=bool(state.get("coverage_said")),
                              introduced=_introduced(history))


def _introduced(history: list[Message]) -> bool:
    """Whether a reply of the app is already in this chat (front-door replies, catalog answers, lists, a run's screens): read
    from the chat itself, so "New question" (which clears it) starts again and a drop (which keeps it) does not. The
    progress line shown while the model is asked is not a reply. Live chat 2026-10-19: replies kept re-introducing the app."""
    return any(isinstance(m, dict) and m.get("role") == "assistant" and _said(m.get("content")) != WORKING for m in history)


def _saved_catalog(config_for: Callable[[str], RunConfig], text: str) -> catalog_summary.CatalogSummary | None:
    """The saved catalog's summary for the front-door model: read from the cache a run would use (nothing is requested).
    None when it is older than APP_CATALOG_MAX_AGE_HOURS: then it counts as missing here too, so a question about it starts
    a run, which asks for the catalog download first."""
    try:
        config = config_for(text)
        summary = catalog_summary.from_cache(config)
    except Exception:  # a bad setting only means no summary; the message is still sorted
        return None
    hours = getattr(config, "catalog_max_age_hours", None)
    if (summary is not None and hours and summary.fetched_at is not None    # an unknown date counts as fresh, as in a run
            and datetime.now(timezone.utc) - summary.fetched_at > timedelta(hours=hours)):
        return None
    return summary


def _list_config(make_config: Callable[[str], RunConfig], listing: catalog_lists.ListRequest) -> Callable[[str], RunConfig]:
    """A list run's settings: the app's own, plus the fixed list (catalog_lists.py). The typed command ("list all seasons")
    and the front door's catalog_list label (triage.catalog_list_request) start the same run."""
    return lambda q: dataclasses.replace(make_config(q), catalog_list=listing.entity, catalog_list_word=listing.word)


def _memory(state: dict[str, Any]) -> chat_memory.ConversationMemory:
    memory = state.get("memory")
    return memory if isinstance(memory, chat_memory.ConversationMemory) else chat_memory.ConversationMemory()


def _remember(state: dict[str, Any], job: Job, *, status: str, acceptance: str) -> dict[str, Any]:
    """Add the question that just ended to the tab's memory (chat_memory.py). A run that cannot be summarized is not
    remembered: memory is a convenience, never a reason to fail a turn."""
    try:
        turn = chat_memory.past_turn(job.controller, status=status, acceptance=acceptance)
    except Exception:  # noqa: BLE001
        return state
    return {**state, "memory": _memory(state).add(turn)}


def _remember_result(state: dict[str, Any], job: Job, event: tuple[str, Any]) -> dict[str, Any]:
    kind, result = event
    if kind != "done":
        return state
    return _remember(state, job, status=str(getattr(result, "execution_status", "unknown")),
                     acceptance=str(getattr(result, "human_acceptance", "pending")))


KEPT_FOR_A_NEW_QUESTION = ("guide_shown", "memory", "coverage_said")    # what a new question keeps of the tab's state
CATALOG_ANSWER_ID = "a catalog answer, no run"         # a remembered catalog answer has no run ID


def _reply_without_run(label: str, topic: str | None, text: str, state: dict[str, Any], *,
                       catalog: catalog_summary.CatalogSummary | None = None,
                       talk: triage.TalkReply | None = None) -> tuple[str, dict[str, Any]]:
    """The answer to a message that starts nothing, and the tab's state after it. It only returns text: no run, no request.
    A catalog question is answered by code from the saved catalog (CatalogSummary.answer; the model's text is not used) and
    remembered, so a follow-up can continue it; without a saved catalog the person is asked whether to look it up, as for
    unsure (the line written by code). The label call's reply is never shown, under any label ("front door, one path",
    2026-10-19): the talk call's reply is (talk: written without the saved catalog or the earlier questions in view and
    checked by triage.ask_to_talk), or, without one, the line written by code for the label. For unsure,
    talk.offer_lookup false (words that mean nothing, no lookup, advice or a ranking) drops the "yes" line, so nothing
    waits for a yes. coverage_said: set once a shown reply said what the app covers (the talk reply's said_coverage, or
    the out_of_scope line written by code), so later replies in the tab do not repeat it."""
    if label == "catalog_question":
        if catalog is not None:
            turn = chat_memory.PastTurn(question=text, run_id=CATALOG_ANSWER_ID, status="answered from the saved catalog by code",
                                        acceptance="not asked", scope=None, answer=catalog.mentions_line(text),
                                        kind="catalog_answer")
            return catalog.answer(text), {**state, "memory": _memory(state).add(turn)}
        label = "unsure"
    talk = talk if talk is not None and talk.label == label else None
    model_text, offer_lookup = (talk.reply, talk.offer_lookup) if talk is not None else ("", True)
    show_guide = label == "greeting_or_smalltalk" and not state.get("guide_shown")     # "Questions to try" once per tab
    said = bool(state.get("coverage_said"))
    content = triage.front_door_reply(label, model_text, text=text, topic=topic, faq=_faq(), source=_source_name(),
                                      show_guide=show_guide, offer_lookup=offer_lookup, coverage_said=said)
    if show_guide:
        state = {**state, "guide_shown": True}
    if not said and ((talk is not None and talk.said_coverage) or (talk is None and label == "out_of_scope")):
        state = {**state, "coverage_said": True}                                       # what the app covers: said once
    if triage.offers_lookup(label, model_text, offer_lookup):
        state = {**state, "pending_question": text}                                    # a later "yes" starts it
    return content, state


OPEN_QUESTIONS = ("new_question", "unsent_reply")   # the tab's own open question while a run waits: drop or keep / send it?


def _without(state: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: v for k, v in state.items() if k not in keys}


def _waiting_on(job: Job) -> str:
    return getattr(job, "waiting_on", "") or ""          # unknown: the message goes straight to the run, as before


def _pending_questions(job: Job) -> list[str]:
    """The question the run is waiting on, as the person saw it: the planner's, the coordinator's, the Retriever's or the
    Analyst's. Empty for the approval, data review and final review prompts."""
    controller, waiting_on = job.controller, _waiting_on(job)
    if waiting_on == CLARIFY_PROMPT:
        return list(getattr(controller.coordination, "questions", None) or [])
    if waiting_on == SUPERVISOR_PROMPT:
        return list(controller.supervisor_questions)
    if waiting_on == RETRIEVER_PROMPT:
        return list(controller.retriever_questions)
    if waiting_on == ANALYST_PROMPT:
        return list(analyst_questions(controller.analysis))
    return []


def _run_has_ended(job: Job) -> bool:
    """The run's thread has finished while the chat went on (it waits 15 minutes for a reply, then records itself canceled)."""
    thread = getattr(job, "thread", None)
    return thread is not None and thread.ident is not None and not thread.is_alive()


def _sort_waiting_message(text: str, job: Job, state: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Code first, for a message typed while the run waits: (route, payload, state). route "run": payload goes to the run;
    "reply": payload is shown and nothing reaches the run; "drop": payload is the new question; "ask_model": the model labels it."""
    kind, pending = conversation.waiting_kind(_waiting_on(job)), _pending_questions(job)
    if "new_question" in state:                                         # the drop-or-keep question is open
        choice = conversation.drop_or_keep(text)
        if choice == "drop":
            return "drop", state["new_question"], _without(state, *OPEN_QUESTIONS)
        if choice == "keep":
            content, _ = render(job, ("prompt", _waiting_on(job)))      # the waiting question again, as it was shown
            return "reply", content, _without(state, *OPEN_QUESTIONS)
        return "reply", conversation.drop_or_keep_question(state["new_question"]), state
    if "unsent_reply" in state:                                         # "should I send it as your answer?" is open
        unsent, state = state["unsent_reply"], _without(state, "unsent_reply")
        if triage.is_yes(text):
            return "run", unsent, state
        if conversation.is_no(text):
            return "reply", conversation.still_waiting_line(kind, pending), state
    if catalog_lists.is_refresh_request(text):                         # "refresh catalogs": a run of its own, so a new question
        return "reply", conversation.drop_or_keep_question(text), {**state, "new_question": text}
    listing = catalog_lists.parse_request(text)
    if listing is not None and not conversation.asks_about_listing(pending):   # "list all traits": from the run's own catalog
        shown = _list_from_the_runs_catalog(listing, job)                         # (never a study ID or a reply the run reads)
        if shown is not None:
            return "reply", f"{shown}\n\n{conversation.still_waiting_line(kind, pending)}", state
        return "reply", conversation.drop_or_keep_question(text), {**state, "new_question": text}   # no catalog yet: a new run
    if conversation.goes_straight_to_the_run(text, _waiting_on(job)):
        return "run", text, state
    word = conversation.control_word(text, _waiting_on(job))
    if word is not None:
        return "run", word, state                                       # the prompt's exact word, found by code
    if conversation.is_list_request(text, pending):
        return "reply", conversation.drop_or_keep_question(text), {**state, "new_question": text}
    return "ask_model", text, state


def _list_from_the_runs_catalog(listing: catalog_lists.ListRequest, job: Job) -> str | None:
    """The list, read by code from the catalog the waiting run has loaded; None before it has one (the catalog setup)."""
    controller = job.controller
    metadata, tables = getattr(controller, "metadata", None), getattr(controller, "catalog_tables", None)
    if metadata is None or not tables:
        return None
    return catalog_lists.chat_message(catalog_lists.build(listing, metadata, tables), in_report=False)


WAITING_KEPT = "Your message was not sent; the question is still waiting."
NOTHING_STARTED = "No question was started."
DROPPED_NOTHING_STARTED = "The earlier question was dropped as you asked; no new question was started."


def budget_reply(error: Exception, after: str = "") -> str:
    """A front-door model call found the model budget used up (the account's credit, spend, quota or billing limit;
    triage passes only that error up): said plainly, with no second request, and `after` says what did not happen.
    Nothing is started, and while a question waits, the message is not sent to it."""
    return " ".join(part for part in ("**The model service cannot answer right now.**", getattr(error, "message", ""), after) if part)


def _ask_while_waiting(text: str, job: Job, triage_model: Callable[[], Any]) -> conversation.WaitingDecision | None:
    prompt = conversation.waiting_prompt(kind=conversation.waiting_kind(_waiting_on(job)), pending=_pending_questions(job),
                                         source=_source_name(), reference=triage.reference_text(_faq()))
    return conversation.ask_model(text, triage_model, prompt=prompt)


def _route_waiting_decision(decision: conversation.WaitingDecision | None, text: str, job: Job,
                            state: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """The model's label, applied by code. Only the person's own text ever goes to the run, never the model's words, and only
    where the run can take it as a reply (conversation.sendable)."""
    kind = conversation.waiting_kind(_waiting_on(job))
    pending = _pending_questions(job)
    sendable = conversation.sendable(kind, text)
    if decision is None:                                                # no model or no decision: what the run can take goes
        return ("run", text, state) if sendable else ("reply", conversation.not_understood(kind, pending), state)
    if decision.label in ("side_question", "out_of_scope"):          # answered; the run keeps waiting (never drop or keep)
        fallback = triage.out_of_scope_line(_source_name(), example=False) if decision.label == "out_of_scope" else ""
        return "reply", conversation.side_reply(decision.reply, kind, pending, fallback=fallback), state
    if decision.label == "new_data_question":
        return "reply", conversation.drop_or_keep_question(text), {**state, "new_question": text}
    if decision.label == "answer_to_pending" and sendable:
        return "run", text, state
    if decision.label == "unsure" and sendable:                         # ask before anything is sent
        return "reply", conversation.confirm_question(kind, text, pending), {**state, "unsent_reply": text}
    return "reply", conversation.not_understood(kind, pending), state  # the run could not take it: its choices again


def respond(message: str, history: list[Message], state: dict[str, Any], *, make_config: Callable[[str], RunConfig] = live_config,
            make_models: Callable[[], Models] = default_models,
            preflight: Callable[[str], list[str]] | None = None,
            triage_model: Callable[[], Any] | None = None) -> Iterator[tuple[list[Message], dict[str, Any], str]]:
    """One chat turn. A message typed while a run waits is sorted first (conversation.py): the prompt's exact control words
    go to the run; a catalog list ("list all traits") is shown from the run's own catalog; any other message gets one
    triage_model call that labels it, and only an answer to the waiting question reaches the run (a side question or an
    out-of-scope message is answered while the run keeps waiting; a new data question asks drop or keep; without a
    decision the message goes to the run as before). A new message passes the front door: a catalog list starts a short
    list run (catalog_lists.py: no rule, model or planner); greetings and app questions found by the rules start nothing and
    get one triage_model call, the talk call, that writes the reply (the line written by code when there is no model or the
    call fails). Every other message, with data words or not, gets the label call with the saved catalog's summary and the
    tab's earlier questions in view (when there are any), and its label decides (Step 2b): a data question starts a run; a
    follow-up starts a run that continues the last question; a catalog list starts the short list run, once code accepts
    its list and word (a refused one counts as no decision); a catalog question, an out-of-scope or an unsure message starts
    nothing. The label call's reply is never shown: the reply to a message that starts nothing (except a catalog question,
    which code answers) comes from the talk call, which sees neither the catalog nor the earlier questions but the person's
    own last messages ("front door, one path", 2026-10-19), or else from code. Without a decision, the message starts a run
    as before. A question that ends is added to the tab's memory. Yields (history, state, box)."""
    text = (message or "").strip()
    if not text:
        yield history, state, ""
        return
    history = list(history) + [{"role": "user", "content": text}]
    job: Job | None = state.get("job")
    working = False                          # the progress line is already shown (while the front door asked the model)
    dropped = False                          # the waiting question was dropped for this message
    if job is not None and _run_has_ended(job):                                    # it ended while the chat went on
        event = job.next_event(timeout=5)
        content, _ = render(job, event)
        history.append({"role": "assistant", "content": "This question had already ended, so your message was not sent to it.\n\n"
                        + content})
        state = {**_without(state, "job", *OPEN_QUESTIONS), "full_report": job.controller.answer_markdown()}
        state = _remember_result(state, job, event)
        yield history, state, ""
        return
    if job is not None:
        route, payload, state = _sort_waiting_message(text, job, state)            # code only: no model call
        if route == "ask_model":
            if triage_model is None:                                               # no model: what the run can take goes, as before
                route, payload, state = _route_waiting_decision(None, text, job, state)
            else:
                history.append({"role": "assistant", "content": WORKING})
                yield history, state, ""
                try:
                    decision = _ask_while_waiting(text, job, triage_model)
                except ModelProviderError as exc:                              # the model budget is used up
                    history[-1] = {"role": "assistant", "content": budget_reply(exc, WAITING_KEPT)}
                    yield history, state, ""
                    return
                history = history[:-1]
                route, payload, state = _route_waiting_decision(decision, text, job, state)
        if route == "reply":                                                       # nothing reaches the run: it keeps waiting
            history.append({"role": "assistant", "content": payload})
            yield history, state, ""
            return
        if route == "drop":
            dropped = True
            state = _remember(state, job, status="dropped for a new question",
                              acceptance=str(getattr(job.controller, "acceptance", "pending")))
            if (getattr(job.controller, "catalog_setup", None) or {}).get("status") == "completed":
                catalog_summary.forget()                                           # the front door reads the new catalogs
            job.close()                                                            # the waiting run ends and is recorded ...
            job, text = None, payload                                              # ... and the new question starts below
            state = {k: v for k, v in state.items() if k in KEPT_FOR_A_NEW_QUESTION}
        else:
            job.reply(payload)                                                     # the person's own words, or the exact word
    if job is None:
        pending = state.get("pending_question")
        state = {k: v for k, v in state.items() if k != "pending_question"}      # this message answers it either way
        question, by_rules, from_pending, listing, refresh = text, None, False, None, False
        if pending and triage.is_yes(text):
            question, from_pending = pending, True                               # "yes" starts the earlier question as typed
        elif catalog_lists.is_refresh_request(text):                             # "refresh catalogs": code decides, no model
            question, refresh = CATALOG_SETUP_QUESTION, True
        elif (listing := catalog_lists.parse_request(text)) is None:            # a catalog list: code decides, no rule or model
            by_rules = triage.sort_by_rules(text)
            if by_rules is not None and by_rules.label not in triage.PIPELINE_LABELS:
                talk, budget_note = None, ""
                if triage_model is not None:                                     # one call, the talk call, writes the reply
                    history.append({"role": "assistant", "content": WORKING})
                    yield history, state, ""
                    try:
                        talk = _ask_to_talk(text, triage_model, by_rules.label, history, state)
                    except ModelProviderError as exc:          # the model budget is used up: the line written by code, and why
                        talk, budget_note = None, budget_reply(exc)
                    history = history[:-1]
                content, state = _reply_without_run(by_rules.label, by_rules.topic, text, state, talk=talk)
                history.append({"role": "assistant", "content": content + ("\n\n" + budget_note if budget_note else "")})
                yield history, state, ""                                         # no run and no database request
                return
        config_for = make_config if listing is None else _list_config(make_config, listing)   # a list run: plus the list
        if refresh:                                                   # a refresh: a catalog setup run, with its own approval
            config_for = lambda q: dataclasses.replace(make_config(q), catalog_setup_only=True, catalog_refresh=True,
                                                       bootstrap_catalogs=True)
        try:
            problems = (preflight or (lambda q: live_preflight(q, config_for)))(question)
        except Exception:  # a bad setting must not leave the chat locked or echo its value
            problems = ["App configuration could not be checked. Ask the operator to review the app and model settings."]
        if problems:
            history.append({"role": "assistant", "content": "This question cannot run yet:\n" + "\n".join(f"- {p}" for p in problems)})
            yield history, state, ""
            return
        if listing is None and not from_pending and not refresh and triage_model is not None:   # data words, or unplaced:
            history.append({"role": "assistant", "content": WORKING})            # the model's label decides (Step 2b)
            working = True
            yield history, state, ""
            memory = _memory(state)
            catalog = _saved_catalog(config_for, text)
            earlier_text = chat_memory.for_model(memory)
            try:
                decision = _ask_front_door(text, triage_model, catalog=catalog, earlier=earlier_text)
            except ModelProviderError as exc:                                  # the model budget is used up: no run
                after = DROPPED_NOTHING_STARTED if dropped else NOTHING_STARTED
                history[-1] = {"role": "assistant", "content": budget_reply(exc, after)}
                yield history, state, ""
                return
            listed = triage.catalog_list_request(decision, text)                  # a whole list or one word, checked by code
            if listed is None and decision is not None and decision.label == "catalog_list":
                decision = None                                                     # refused fields: as without a decision
            if listed is not None:                                                  # the list run the typed command starts:
                config_for = _list_config(make_config, listed)                      # no planner, no model-written answer
            elif decision is not None and decision.label not in triage.PIPELINE_LABELS:   # None: no decision, so the run starts
                try:
                    talk = (_ask_to_talk(text, triage_model, decision.label, history, state)   # a second call writes the reply,
                            if decision.label in triage.TALK_LABELS else None)  # with or without a catalog or earlier questions
                except ModelProviderError as exc:                              # the model budget is used up
                    after = DROPPED_NOTHING_STARTED if dropped else NOTHING_STARTED
                    history[-1] = {"role": "assistant", "content": budget_reply(exc, after)}
                    yield history, state, ""
                    return
                content, state = _reply_without_run(decision.label, None, text, state, catalog=catalog, talk=talk)
                history[-1] = {"role": "assistant", "content": content}
                yield history, state, ""
                return
            earlier = memory.last
            if (decision is not None and decision.label == "follow_up" and earlier is not None
                    and earlier.continuable):                                    # continue the last question (not a list)
                question = chat_memory.follow_up_question(earlier, text)
                plain = config_for
                config_for = lambda q: dataclasses.replace(plain(q), follow_up_of=earlier.run_id, carried_over=earlier.question)
        try:
            job = Job(question, make_config=config_for, make_models=make_models).start()
        except Exception:
            failed = {"role": "assistant", "content": "This question could not start. Ask the operator to check the app settings and output-folder access."}
            if working:
                history[-1] = failed
            else:
                history.append(failed)
            yield history, state, ""
            return
        state = {"job": job, **{k: state[k] for k in KEPT_FOR_A_NEW_QUESTION if k in state}}   # it clears the old report
    if not working:
        history.append({"role": "assistant", "content": WORKING})
    yield history, state, ""
    event = job.next_event()
    content, finished = render(job, event)
    history[-1] = {"role": "assistant", "content": content}
    if job.answer_shown or finished:
        state = {**state, "full_report": job.controller.answer_markdown()}
    elif job.waiting_on.startswith("approve / edit / cancel"):
        scope_ids = ", ".join(job.controller.plan.scope.study_ids)
        state = {**state, "full_report": job.controller.approval_screen() + "\n\nAll matched study IDs: " + (scope_ids or "none")}
    elif job.waiting_on == CATALOG_PROMPT:
        state = {**state, "full_report": job.controller.catalog_approval_screen()}
    elif job.waiting_on == DATA_REVIEW_PROMPT:
        state = {**state, "full_report": job.controller.data_review_screen()}
    if finished:
        state = _remember_result({k: v for k, v in state.items() if k != "job"}, job, event)
        if (getattr(getattr(job.controller, "config", None), "catalog_setup_only", False)
                or (getattr(job.controller, "catalog_setup", None) or {}).get("status") == "completed"):
            catalog_summary.forget()                                               # the front door reads the new catalogs
    yield history, state, ""


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------

def source_markdown() -> str:
    """Data-source details belong in the FAQ, not above the conversation."""
    settings = load_settings()
    return ("The assistant reads public records from the configured breeding database through **BrAPI**, "
            "a standard way to access breeding data. It does not edit the database.\n\n"
            f"Current data source: `{settings.base_url}`. Results depend on the records available there.")


def source_details(make_config: Callable[[str], RunConfig]) -> str:
    """The help page's answer about the data source: source_markdown, then when the saved catalogs were downloaded and who
    approves a catalog download (catalog_source_line), with the page's own settings."""
    line = catalog_source_line(make_config)
    return source_markdown() + (f" {line}" if line else "")


def catalog_source_line(make_config: Callable[[str], RunConfig]) -> str:
    """With APP_CATALOG_STANDING_APPROVAL on: when the saved study and trait catalogs were downloaded (read from the cache,
    nothing requested), and that the app's operator approves a new catalog download. With the setting off: nothing, so the
    help page reads exactly as before."""
    try:
        config = make_config(CATALOG_SETUP_QUESTION)
        if not config.catalog_standing_approval:
            return ""
        summary = catalog_summary.from_cache(config)
    except Exception:  # a settings problem must not break the help page
        return ""
    when = (f"The saved study and trait catalogs were downloaded {summary.fetched_at:%Y-%m-%d %H:%M} UTC."
            if summary is not None and summary.fetched_at is not None else
            "The study and trait catalogs are saved." if summary is not None else "The study and trait catalogs are not saved yet.")
    hours = getattr(config, "catalog_max_age_hours", None)
    age = (f" Saved catalogs older than {hours:g} hours are downloaded again; type **refresh catalogs** to do it now."
           if hours else " Type **refresh catalogs** to download them again.")
    return (f"{when} When they need downloading, the app's operator approves it (a standing approval); measurements always "
            f"need your approval.{age}")


def hosted_notice(environ: Mapping[str, str] | None = None) -> str:
    """Data-sharing policy for hosted demos, without displaying credentials or endpoint URLs."""
    values = os.environ if environ is None else environ
    if not values.get("SPACE_ID"):
        return ""
    try:
        host = (urlsplit(values.get("LLM_BASE_URL", "")).hostname or "").lower()
    except ValueError:
        host = ""
    provider = "OpenAI" if host == "api.openai.com" else "the configured model provider"
    return (f"Your messages and relevant catalog/data excerpts are sent to **{provider}**. "
            "The planner uses the model before database-fetch approval. Run records are saved on this demo's server; "
            "copy your full report before closing the page. Approval allows database requests; acceptance records your review.")


def full_report(state: dict[str, Any] | None) -> str:
    """Only text from this browser session, never a client-supplied path or another run's directory."""
    return str((state or {}).get("full_report", ""))


def _close_job(state: dict[str, Any]) -> None:
    job = (state or {}).get("job")
    if job is not None:
        job.close()


def reply_buttons(state: dict[str, Any] | None) -> tuple[bool, bool, bool, bool]:
    """Which quick replies fit what the run waits for: (Approve, Cancel, Accept, Reject). On 2026-09-29 '1032', 'done' and
    'accept' were typed where the app waited for approve / cancel; buttons show the valid replies (typing still works).
    While the tab asks its own question (drop or keep, or "is that your answer?"), the run's buttons are hidden."""
    job = (state or {}).get("job")
    asking = any(key in (state or {}) for key in OPEN_QUESTIONS)
    waiting = job.waiting_on if job is not None and not asking else ""
    approval = waiting.startswith("approve / edit / cancel") or waiting in (CATALOG_PROMPT, DATA_REVIEW_PROMPT)
    answer = waiting.startswith("accept / reject")
    question = waiting in (CLARIFY_PROMPT, ANALYST_PROMPT, RETRIEVER_PROMPT, SUPERVISOR_PROMPT)
    return approval, approval or question, answer, answer


def first_button_word(state: dict[str, Any] | None) -> str:
    """The word the first quick-reply button sends: the one its screen asks for. At the data review the button reads
    "Continue to analysis" and the screen says "Reply continue", so it sends "continue" (2026-10-19: it sent "approve",
    which code also accepts there, Job.reply; the words now match). Elsewhere it reads "Approve" and sends "approve"."""
    job = (state or {}).get("job")
    return "continue" if job is not None and getattr(job, "waiting_on", "") == DATA_REVIEW_PROMPT else "approve"


def locked():
    """While the app works: the input box (with its arrow), New question and the reply buttons are dimmed, so a message
    cannot overtake another. Order: box, New question, Approve, Cancel, Accept, Reject."""
    import gradio as gr

    return tuple(gr.update(interactive=False) for _ in range(6))


def after_step(state: dict[str, Any] | None):
    """The step is done: the box and New question work again, and only the reply buttons that fit are shown."""
    import gradio as gr

    job = (state or {}).get("job")
    review = job is not None and job.waiting_on == DATA_REVIEW_PROMPT
    buttons = reply_buttons(state)
    return (gr.update(interactive=True), gr.update(interactive=True),
            gr.update(interactive=True, visible=buttons[0], value="Continue to analysis" if review else "Approve"),
            *(gr.update(interactive=True, visible=show) for show in buttons[1:]))


def start_over(state: dict[str, Any] | None) -> tuple[list[Message], dict[str, Any], str]:
    """New question: a run that waits for a reply is ended (cancel; it is still recorded) and the chat is cleared, and so is
    the tab's memory of earlier questions. The tab keeps only one thing: that it has already shown the example questions."""
    _close_job(state or {})
    return [], {k: v for k, v in (state or {}).items() if k == "guide_shown"}, ""


def build_ui(*, make_config: Callable[[str], RunConfig] = live_config, make_models: Callable[[], Models] = default_models,
             preflight: Callable[[str], list[str]] | None = None, triage_model: Callable[[], Any] | None = default_triage_model):
    import gradio as gr

    with gr.Blocks(title="Breeding data assistant", fill_height=False) as demo:
        gr.HTML(HERO_HTML, elem_id="brapi-hero")
        with gr.Tabs(selected="assistant", elem_id="brapi-tabs"):
            with gr.Tab("Assistant", id="assistant", elem_id="brapi-assistant"):
                chat = gr.Chatbot(height="min(62vh, 620px)", min_height=350, show_label=False, layout="bubble",
                                  buttons=["copy"], placeholder=PLACEHOLDER, elem_id="brapi-chat")
                box = gr.Textbox(placeholder="Ask about a study, trait, year or location…", show_label=False,
                                 autofocus=False, lines=1, max_lines=5, scale=0, submit_btn=True, elem_id="brapi-composer")
                with gr.Row(elem_id="brapi-actions"):
                    approve = gr.Button("Approve", variant="primary", size="sm", scale=0, min_width=110, visible=False)
                    cancel = gr.Button("Cancel", variant="secondary", size="sm", scale=0, min_width=110, visible=False)
                    accept = gr.Button("Accept", variant="primary", size="sm", scale=0, min_width=110, visible=False)
                    reject = gr.Button("Reject", variant="secondary", size="sm", scale=0, min_width=110, visible=False)
                    new = gr.Button("New question", variant="secondary", size="sm", scale=0, min_width=140)
                gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[box], label="Try a question", elem_id="brapi-examples")
                with gr.Accordion("Full report · sources & methods", open=False, elem_id="brapi-report"):
                    report = gr.Textbox(label="Full report", lines=12, max_lines=30, interactive=False, buttons=["copy"])
            with gr.Tab("FAQ & How to use", id="help", elem_id="brapi-help"):
                gr.Markdown("## A little guidance, when you need it.\n\n"
                            "Get started with a focused question, then review the evidence behind the answer.",
                            elem_id="brapi-help-intro")
                with gr.Row(elem_id="brapi-help-cards"):
                    with gr.Column(scale=3, min_width=280):
                        gr.Markdown(QUICK_START, elem_classes=["brapi-help-card"])
                    with gr.Column(scale=2, min_width=240):
                        gr.Markdown(QUESTION_GUIDE, elem_classes=["brapi-help-card"])
                source_answer = None
                for question, answer in faq_items(source_details(make_config), data_sharing_policy=hosted_notice()):
                    with gr.Accordion(question, open=False, elem_classes=["brapi-faq-item"]):
                        shown = gr.Markdown(answer)
                    if question == SOURCE_FAQ:
                        source_answer = shown
                if source_answer is not None and catalog_source_line(make_config):   # setting on: the catalogs' date as it
                    demo.load(lambda: source_details(make_config), None, source_answer,  # is when the page opens (off: no
                              api_visibility="private", queue=False, show_progress="hidden")   # page-load event, as before)
        state = gr.State({}, time_to_live=REPLY_TIMEOUT_S + 60, delete_callback=_close_job)
        controls = [box, new, approve, cancel, accept, reject]              # the order locked() and after_step() use

        def on_submit(message, history, current):
            yield from respond(message, history or [], current or {}, make_config=make_config, make_models=make_models, preflight=preflight,
                               triage_model=triage_model)

        def pressing(word: str | Callable[[dict[str, Any] | None], str]):
            def on_press(history, current):                                # a button is the same as typing its word ...
                if not (current or {}).get("job"):                         # ... unless nothing waits any more (e.g. after 15 minutes)
                    yield (list(history or []) + [{"role": "assistant", "content": "Nothing is waiting for that reply now; ask a new question."}],
                           current or {}, "")
                    return
                said = word(current) if callable(word) else word            # the first button: the word its screen asks for
                yield from respond(said, history or [], current or {}, make_config=make_config, make_models=make_models, preflight=preflight)
            return on_press

        # One queue group for state-changing callbacks; hidden API endpoints add no authentication.
        event_options = {"concurrency_id": CHAT_CONCURRENCY_ID, "concurrency_limit": 1, "api_visibility": "private"}
        (box.submit(locked, None, controls, **event_options)
            .then(on_submit, inputs=[box, chat, state], outputs=[chat, state, box], **event_options)
            .then(full_report, inputs=[state], outputs=[report], **event_options)
            .then(after_step, inputs=[state], outputs=controls, **event_options))
        for button, word in ((approve, first_button_word), (cancel, "cancel"), (accept, "accept"), (reject, "reject")):
            (button.click(locked, None, controls, **event_options)
                .then(pressing(word), inputs=[chat, state], outputs=[chat, state, box], **event_options)
                .then(full_report, inputs=[state], outputs=[report], **event_options)
                .then(after_step, inputs=[state], outputs=controls, **event_options))
        (new.click(start_over, inputs=[state], outputs=[chat, state, box], **event_options)
            .then(full_report, inputs=[state], outputs=[report], **event_options)
            .then(after_step, inputs=[state], outputs=controls, **event_options))
    return demo


def launch_options(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Validate hosting options without launching a server or exposing credentials in messages."""
    values = os.environ if environ is None else environ
    user, password = values.get("APP_USERNAME", ""), values.get("APP_PASSWORD", "")
    has_user, has_password = bool(user.strip()), bool(password.strip())
    if has_user != has_password:
        raise SystemExit("APP_USERNAME and APP_PASSWORD must both be set, or both be absent")
    is_space = bool(values.get("SPACE_ID"))
    open_on_purpose = values.get("APP_ALLOW_OPEN", "").strip().lower() == "true"
    if is_space and not (has_user and has_password) and not open_on_purpose:
        raise SystemExit("refusing to serve a Hugging Face Space without APP_USERNAME and APP_PASSWORD: anyone could spend the model key "
                         "(set APP_ALLOW_OPEN=true only after you verify that the Space is private)")
    host = values.get("GRADIO_SERVER_NAME", "").strip() or ("0.0.0.0" if is_space else "127.0.0.1")
    return {"server_name": host, "auth": (user, password) if has_user and has_password else None}
