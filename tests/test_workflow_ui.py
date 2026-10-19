"""The extra human handoffs in the chat; no server, model or browser is launched."""
from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

import app
from run import CATALOG_PROMPT, DATA_REVIEW_PROMPT, RETRIEVER_PROMPT


def waiting_job(prompt: str) -> app.Job:
    """Use the real reply/render methods without starting a controller thread."""
    job = app.Job.__new__(app.Job)
    job.waiting_on = prompt
    job.inbox = queue.Queue()
    job.outbox = queue.Queue()
    job.answer_shown = False
    job.controller = SimpleNamespace(
        retriever_questions=["The reply contains two seasons. Which label should be used?"],
        retriever_answers=1,
        config=SimpleNamespace(max_clarification_rounds=3),
        data_review_screen=lambda: "## Review fetched data\n\n6 rows in the approved study.\n\n| Value |\n|---|\n| 12 |",
    )
    return job


def test_retriever_question_is_visible_with_a_safe_scope_change_route():
    job = waiting_job(RETRIEVER_PROMPT)
    content, finished = app.render(job, ("prompt", RETRIEVER_PROMPT))
    assert not finished and not job.answer_shown
    assert "Which label should be used?" in content
    assert "answer 2 of 3" in content
    assert "within the approved scope" in content
    assert "**cancel** and start a new question" in content
    assert app.reply_buttons({"job": job}) == (False, True, False, False)


@pytest.mark.parametrize("reply", ["Use season 2026", "change scope: include study S2"])
def test_retriever_replies_reach_the_controller_without_becoming_approval(reply):
    job = waiting_job(RETRIEVER_PROMPT)
    job.reply(reply)
    assert job.inbox.get_nowait() == reply
    assert job.inbox.empty()


def test_cancel_and_closing_a_retriever_question_end_the_wait():
    job = waiting_job(RETRIEVER_PROMPT)
    job.reply("  CANCEL  ")
    assert job.inbox.get_nowait() == ""
    job.close()
    assert job.inbox.get_nowait() == ""


def test_review_shows_data_and_requires_continuation_separate_from_answer_acceptance():
    job = waiting_job(DATA_REVIEW_PROMPT)
    content, finished = app.render(job, ("prompt", DATA_REVIEW_PROMPT))
    assert "6 rows in the approved study" in content and "| 12 |" in content
    assert "**continue**" in content and "**cancel**" in content
    assert "accept the final answer separately" in content
    assert not finished and not job.answer_shown
    assert app.reply_buttons({"job": job}) == (True, True, False, False)


def test_continue_button_uses_the_review_reply_and_returns_to_approve_label():
    job = waiting_job(DATA_REVIEW_PROMPT)
    controls = app.after_step({"job": job})
    assert len(controls) == 6
    assert controls[2]["value"] == "Continue to analysis"
    assert [item["visible"] for item in controls[2:]] == [True, True, False, False]
    job.reply("approve")  # the existing first-button callback; its label changes above
    assert job.inbox.get_nowait() == "continue"
    job.waiting_on = "approve / edit / cancel: "
    assert app.after_step({"job": job})[2]["value"] == "Approve"
    job.reply("approve")
    assert job.inbox.get_nowait() == "approve"
    job.waiting_on = CATALOG_PROMPT
    job.reply("approve")
    assert job.inbox.get_nowait() == "approve catalog"


@pytest.mark.parametrize("waiting_on, word", [(DATA_REVIEW_PROMPT, "continue"), ("approve / edit / cancel: ", "approve"),
                                              (CATALOG_PROMPT, "approve")])
def test_the_first_button_sends_the_word_its_screen_asks_for(waiting_on, word):
    """2026-10-19: at the data review the "Continue to analysis" button sent "approve" (code accepted it as continue). Now
    each screen's button sends the word on that screen; a typed "approve" at the review is still accepted (Job.reply)."""
    job = waiting_job(waiting_on)
    assert app.first_button_word({"job": job}) == word and word in waiting_on
    job.reply(word)
    assert job.inbox.get_nowait() == {"approve": "approve catalog" if waiting_on == CATALOG_PROMPT else "approve"}.get(word, word)
    assert app.first_button_word({}) == "approve"                                   # nothing waits: as before
    assert app.after_step({})[2]["value"] == "Approve"


def test_accept_cannot_act_as_a_data_review_continue_and_cancel_stays_cancel():
    job = waiting_job(DATA_REVIEW_PROMPT)
    job.reply("accept")
    assert job.inbox.get_nowait() == "accept"  # only the controller decides validity; UI cannot turn it into consent
    job.reply("cancel")
    assert job.inbox.get_nowait() == "cancel"


def test_full_report_updates_to_the_reviewed_data_instead_of_the_old_plan():
    job = waiting_job("approve / edit / cancel: ")

    def fetched_data_ready():
        job.waiting_on = DATA_REVIEW_PROMPT
        return "prompt", DATA_REVIEW_PROMPT

    job.next_event = fetched_data_ready
    state = {"job": job, "full_report": "Old approval scope"}
    events = list(app.respond("approve", [], state))
    history, updated, box = events[-1]
    assert box == "" and updated["job"] is job
    assert "Reply **continue**" in history[-1]["content"]
    assert app.full_report(updated) == job.controller.data_review_screen()
    assert "Old approval scope" not in app.full_report(updated)
    assert job.inbox.get_nowait() == "approve"  # the earlier fetch approval did not also continue analysis
