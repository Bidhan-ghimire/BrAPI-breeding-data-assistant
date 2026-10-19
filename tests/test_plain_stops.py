"""The chat shows a plain line for every stop that code writes (coach check 2b, 2026-10-19).

Live 2026-10-19 the chat showed a breeder "analyst asks: The bounded plan did not apply every approved scope filter:
season_id". Every stop message in catalog_answers.run_catalog_plan and metadata_answers._clarify is read from the source
here, so a new stop without a plain line fails this file."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import catalog_answers
import chat_presenter as cp
import metadata_answers
from contracts import AnalysisReport

ROOT = Path(__file__).resolve().parents[1]
ENGINEER_WORDS = ("artifact", "lineage", "hash", "export", "bounded", "entity", "metadata", "json", "scope", "args",
                  "catalog", "registered", "_id", "analyst asks", "handle", "claims")


def _sample(node: ast.AST) -> str | None:
    """A message the call can write: the literal text, with "7" for each inserted value; None for str(exc) and for an
    error text taken from a tool result (those reach the chat as the general plain line)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(part.value if isinstance(part, ast.Constant) else "7" for part in node.values)
    if isinstance(node, ast.BinOp):
        left = _sample(node.left)
        return left + ("season_id" if "scope filter" in left else "7")
    if isinstance(node, (ast.Call, ast.IfExp)):
        return None
    raise AssertionError(f"unexpected stop message form: {ast.dump(node)}")


def _messages(module: str, *helpers: str) -> list[str]:
    tree = ast.parse((ROOT / module).read_text(encoding="utf-8"))
    found = [_sample(call.args[0]) for call in ast.walk(tree)
             if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in helpers and call.args]
    return [message for message in found if message is not None]


CATALOG = _messages("catalog_answers.py", "stop", "ValueError", "TimeoutError")   # errors reach stop(str(exc))
METADATA = _messages("metadata_answers.py", "_clarify")


def test_every_stop_message_was_found():
    assert len(CATALOG) == 33 and len(METADATA) == 9    # a new stop: give it a plain line, then raise the number


@pytest.mark.parametrize("message", CATALOG)
def test_every_catalog_stop_has_a_plain_line(message):
    plain = catalog_answers.plain_reason(message)
    assert plain and not any(word in plain.lower() for word in ENGINEER_WORDS), (message, plain)


@pytest.mark.parametrize("message", METADATA)
def test_every_metadata_stop_has_a_plain_line(message):
    plain = metadata_answers.plain_reason(message)
    assert plain and not any(word in plain.lower() for word in ENGINEER_WORDS), (message, plain)


def test_filter_names_are_words_and_unknown_text_is_not_matched():
    assert catalog_answers.plain_reason("The bounded plan did not apply every approved scope filter: program_id, season_id") == (
        "The plan did not apply the program and season filter you approved, so no count was made.")
    assert catalog_answers.plain_reason("something else") is None and metadata_answers.plain_reason("something else") is None
    assert metadata_answers.plain_reason("More than one variable record matches the supplied name. Please choose an exact "
                                         "database ID.") == "More than one trait has that name. Please give its database ID."


def stopped(caveat):
    analysis = AnalysisReport(status="needs_clarification", caveats=["Rows read: 18.", caveat])
    return SimpleNamespace(execution_status="needs_clarification", analysis=analysis, notes=[],
                           retrieval=SimpleNamespace(status="completed", warnings=[]))


def test_the_chat_shows_the_plain_line_and_the_report_keeps_the_exact_check():
    exact = "analyst asks: The bounded plan did not apply every approved scope filter: season_id"
    view = stopped(exact)
    shown = cp.answer_summary(view)
    assert "The plan did not apply the season filter you approved, so no count was made." in shown
    assert "bounded" not in shown and "season_id" not in shown and "analyst asks" not in shown
    assert "Full report" in shown and view.analysis.caveats[-1] == exact      # the exact check is kept for the report


def test_a_stop_without_a_plain_line_gets_the_general_one():
    """A tool's own error text (catalog_answers: stop(str(exc))) is not written for a breeder; the report keeps it."""
    shown = cp.answer_summary(stopped("analyst asks: distinct_by column studyDbX is not in the artifact"))
    assert "The app's own checks stopped this answer before a number was given." in shown
    assert "studyDbX" not in shown and "analyst asks" not in shown
