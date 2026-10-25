"""The sweetpotato benchmark's single-statistic sub-questions through the app (coach Task 1, 2026-10-21).

eval/benchmark/spb_subquestions.json holds the 33 benchmark items computed from SweetPotatoBase plot data, split into
sub-questions of one statistic each, with the gold values (sweetpotato_breeding_benchmark.json, built 16-17 Oct 2026
independently of this app). Each sub-question is asked as a new chat, exactly as eval/harness.py asks its questions (the
same scripted approver: "approve" at the data request, "continue" at the data review, nothing else), on a copy of a
cache folder; with --allow-live the app may fetch what the copy lacks.

Grading is code-based, like the harness: only the code-written part of the chat's last message counts toward a pass
(harness.code_written: model notes are left out); the model's notes are checked for invented numbers (decision (31)(a)). Buckets:
  pass       every expected value (and name) is found within its tolerance (or every value of one listed alternative)
  wrong      the answer shows backed numbers, but not the expected ones
  no_number  an honest answer without the number: "cannot compute", a question back, a lookup offer, a catalog summary
  blocked    a gate, a limit or an error stopped the question (blocked / limit / could not complete / the catalog
             approval the harness never gives / a harness error)
  invented   a number in the code-written part that no claim, no run text, no table of the run and not the question
             backs, or a number in the model's notes that none of these and no number of the code-written part backs,
             read at the precision written (rule (3), Bidhan, coach 2026-10-25 12:20 UTC): the red flag, reported first
             whatever else the answer holds (except on the app's question screens: they answer nothing, so they are not
             checked)
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))           # the harness beside this file
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))    # the app
import harness  # noqa: E402
import triage  # noqa: E402
from console import say  # noqa: E402

SUBQUESTIONS = Path(__file__).resolve().parent / "benchmark" / "spb_subquestions.json"
RESULTS = harness.RESULTS
TIERS = ("T1", "T2", "T3")
BUCKETS = ("pass", "wrong", "no_number", "blocked", "invented")
REQUIRED = ("sub_id", "parent", "question", "expected", "expected_text", "tolerance", "alternatives", "tier")
RULES = ("exact", "display", "relative")
# Ontology terms in variable names ("CO_331:0000296", "COMP:0000107") are never a statistic.
_ONTOLOGY = re.compile(r"\b[A-Z][A-Z0-9_]*:\d+\b")
# Numbers the app writes around an answer by itself (a source label, a list cut, a clarification count, a catalog block's
# group or ID count): left out of the invention check only; the pass check still reads them ("Studies: 18 distinct IDs"
# is how the app shows a distinct count; review 10).
_STRUCTURAL = re.compile(r"source \d+|Showing \d+ of \d+ entries; \d+ additional|answer \d+ of \d+|\d+ (?:groups|distinct IDs)\b")
# A full ISO timestamp ("2026-10-21T12:22:33.472516+00:00": a record's fetch time): no number (baseline 2026-10-21).
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?")
# A number in scientific notation, as the presenter writes a small value ("1.5e-05"; chat_presenter._number uses .12g).
_SCIENTIFIC = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?[eE][+-]?\d+(?!\w|\.\w)")     # a full stop after it ends a sentence
# S2 (decision of Bidhan, coach 2026-10-25 14:35 UTC; pinned 14:49): identifiers are not quantities. A sign before a zero-padded
# number (a 0 followed by at least one more digit before any dot) is no minus sign: "-00595", the abbreviated ending of the plot
# name PEP2016CSR-ST03-00595, reads 595 and still needs backing; "-0.5", "-0" and "-0.25" keep their sign (a hyphen glued to a
# letter or digit was never a sign). A name is a word of capital letters and digits that holds both (MZP2016UMB, CIP199062): its
# whole digit runs of four or more digits that start with 1-9 (a year, a long ID) are no numbers of its text, but as a source they
# back a number of that value (name_numbers), never a part of one. A code's short or zero-padded run (1 in ST01, 3 in ST03, 0045 in
# 24Y2ND0045CUN) backs nothing: it would back an invented small count (review_s2.md and review_s2_2.md, MAJOR 1; the narrowing
# was put to the coach in the builder's 16:54 UTC entry).
_ZERO_PADDED = re.compile(r"[+-]0\d+(?:\.\d+)?")
_WORD = re.compile(r"[A-Za-z0-9]+")
_ASKED_FIRST_LINES = ("More detail is needed.",)
_ASKED_STATUS = ("stopped: the app asked a question", "no run: the app asked whether to look it up")
_BLOCKED_STATUS = ("error", "stopped: the catalog setup needs an approval", "stopped at a step", "stopped after")
# A reply in the model's own words, with no run (the front door's talk reply): never an answer to score; the app's own
# backstop allows only numbers from the person's words in it, so any other number is the red flag (review 10).
MODEL_REPLY_STATUS = "no run: the app answered without starting one"
# A no-run reply that code wrote, not the model (coach 2026-10-24 12:50 UTC, (b)): C29.6 was "invented" for the 2026 of the
# front door's example question. Only a reply byte-identical to one of code_written_texts() is exempt; one character
# off, or one digit added, is graded as before.
CODE_REPLY_WHY = "no run: code-written reply (triage.out_of_scope_line), numbers not graded"
# Report lines that quote the model, not what code did: its notes; the IDs it named that nothing fetched, returned or produced
# (agents/retriever.py build_report and _drop_unknown_kinds, agents/analyst.py build_report); its rejected tool arguments; the
# agents' questions, with any answer to them (run.py's clarification notes; the harness never answers); the supervisor's stop
# reason. They back no number: an ID on the run's record only because the model named it is an invented ID's path (coach,
# 2026-10-21 20:50 UTC, answer 2; review 2026-10-22).
# An agent error quotes the model when it reports the model's refusal, its malformed tool call or its unreadable final reply
# (llm.py add_error). The retriever and the analyst write it as "agent error ...", the planner's arrives as "coordinator: agent
# error ..." and the supervisor's as "supervisor: supervisor error ..." (review 2026-10-22, round 2, R1). A budget or provider
# error is the code's own and still backs its numbers.
_MODEL_ERRORS = tuple(f"{spelling} {code}" for spelling in ("agent error", "coordinator: agent error", "supervisor: supervisor error")
                      for code in ("model_refusal", "malformed_arguments", "invalid_json"))
_MODEL_QUOTED = ("model:", "model note:", "model declared ids", "model listed resolved entries", "model named claim ids",
                 "studies named by the model", "clarification ", "retriever clarification ", "analyst clarification ",
                 *_MODEL_ERRORS, "coordinator stopped the run:")
# Coordinator notes that code writes about the planner's draft: a rejected attempt ("coordinator: attempt 2 rejected: ... at most
# 12 items ..., not 14") and a dropped clarification. The draft's own IDs, names and questions sit in quotes or brackets
# ("scope.study_ids names IDs that are not in the validated metadata: ['99991']", "dropped clarification 'Which 5 studies?'"):
# those are blanked, the code's own numbers stay (review 2026-10-22).
_DRAFT_LINES = ("coordinator: attempt", "coordinator: dropped clarification")
# A single quote opens a quoted part only where no word runs into it ("the plan's narrowing" holds none), and an escaped quote
# does not close it: repr writes a question with both quote kinds as '... the "S1" study\'s ...' (review 2026-10-22, round 2, R2).
# An escaped quote never opens one either, so a long run of them is read once, not once per quote (review of round 3).
_QUOTED_PART = re.compile(r"(?<![\w\\])'(?:\\.|[^'\\])*'(?!\w)|(?<!\\)\"(?:\\.|[^\"\\])*\"|\{[^}]*\}|\[[^\]]*\]")
# Runs whose answer is written by code from the saved catalog, not from claims about measurements: a list run.
CATALOG_ACTIONS = ("list_catalog_names", "count_catalog_rows")


def load_subquestions(path: Path) -> list[dict[str, Any]]:
    """The sub-questions, each checked for what the grader needs: something to check, a known tier and a complete
    tolerance (review 10: an empty expected list, or "display" without "abs", graded wrongly)."""
    subs = json.loads(Path(path).read_text(encoding="utf-8"))["subquestions"]
    for s in subs:
        missing = [key for key in REQUIRED if key not in s]
        if missing:
            raise ValueError(f"{s.get('sub_id', '?')}: missing {', '.join(missing)}")
        if s["tier"] not in TIERS:
            raise ValueError(f"{s['sub_id']}: tier must be one of {TIERS}, not {s['tier']!r}")
        rule = s["tolerance"].get("rule")
        if rule not in RULES:
            raise ValueError(f"{s['sub_id']}: tolerance rule must be one of {RULES}")
        if rule == "display" and not float(s["tolerance"].get("abs") or 0) > 0:
            raise ValueError(f"{s['sub_id']}: a display tolerance needs abs > 0")
        if s.get("checkable", True) and not (s["expected"] or s["expected_text"]):
            raise ValueError(f"{s['sub_id']}: nothing to check (no expected value or name)")
        for alternative in s["alternatives"]:
            if not (alternative.get("expected") or alternative.get("expected_text")):
                raise ValueError(f"{s['sub_id']}: an alternative with nothing to check")
    return subs


def _numbers(text: str, *, written: bool = False) -> list[tuple[float, int, int, int]]:
    """Every number in the text (harness.numbers_in), with numbers in scientific notation read whole and ontology terms
    left out (blanked at the same length, so positions stay); full ISO timestamps are no numbers. written=True reads a number
    in scientific notation at the precision it is written with, as for a model's note ("5.6e-7" to two figures), not at the
    four figures the app writes (rule (3), decision (31)(a)). A sign before a zero-padded digit run is no minus sign (S2:
    "-00595" reads 595)."""
    blank = lambda m: " " * len(m.group())                                              # noqa: E731
    text = _TIMESTAMP.sub(blank, text)
    found = []
    for match in _SCIENTIFIC.finditer(text):
        mantissa, _, exponent = match.group(0).lower().partition("e")
        digits = exponent.lstrip("+-")
        if len(digits) > 3 or (not exponent.startswith("-") and int(digits) > 300):
            continue          # "0E0326" (a plot code), "0e999": no number; their precision would overflow _backed (review 2026-10-25)
        value = float(match.group(0))
        if not math.isfinite(value):
            continue                                          # "92E0326" inside a name reads as inf: no number (T3 review 2)
        # read at 4 significant digits at least: the app writes .4g, which drops trailing zeros ("1e+07" is 1.000e+07), so the last
        # digit steps by 10^(exponent - 3); "1.9e-07" is read as 1.900e-07 (T3 reviews 1 and 2, 2026-10-23)
        places = len(mantissa.split(".")[1]) if "." in mantissa else 0
        decimals = (places if written else max(places, 3)) - int(exponent)
        found.append((value, decimals, match.start(), match.end()))
    found += harness.numbers_in(_ONTOLOGY.sub(blank, _SCIENTIFIC.sub(blank, text)))
    found = [(abs(v), d, start, end) if _ZERO_PADDED.fullmatch(text, start, end) else (v, d, start, end) for v, d, start, end in found]
    return sorted(found, key=lambda n: n[2])


def name_numbers(text: str) -> list[float]:
    """The whole digit runs of four or more digits that start with 1-9 inside the names in the text (S2): a name is a word of
    capital letters and digits that holds both (2016 in MZP2016UMB-ST01, 199062 in CIP199062.1; the 1 of ST01 is too short, the
    0045 of 24Y2ND0045CUN is zero-padded); full timestamps, scientific numbers and ontology terms are blanked first. They are no
    numbers of the text (the reader leaves them out); as a source they back a number of the same value."""
    blank = lambda m: " " * len(m.group())                                              # noqa: E731
    text = _ONTOLOGY.sub(blank, _SCIENTIFIC.sub(blank, _TIMESTAMP.sub(blank, text)))
    return [float(run) for word in _WORD.findall(text) if word.isupper() and any(ch.isdigit() for ch in word)
            for run in re.findall(r"(?<!\d)[1-9]\d{3,}", word)]


def _numbers_for_invention(text: str) -> list[tuple[float, int, int, int]]:
    return _numbers(_STRUCTURAL.sub(lambda m: " " * len(m.group()), text))


def _without_quotes(text: str) -> str:
    """The text without the lines the app quotes ("> ..."): a model's question shown on a clarification screen, or the
    person's own earlier words. They are attributed, not answers: left out like the model notes (review 10)."""
    return "\n".join(line for line in text.split("\n") if not line.lstrip().startswith(">"))


def claim_values(claims: Iterable[dict[str, Any]]) -> list[float]:
    """The numbers a run's claims hold as numbers: values, denominators, independent units and exclusion counts (claim_numbers
    without the numbers in the claims' texts): the second reading of a note number (note_numbers)."""
    found: list[float] = []
    for c in claims:
        denominator = c.get("denominator") or {}
        for v in (c.get("value"), denominator.get("n"), c.get("n_independent_units"),
                  *[e.get("count") for e in c.get("exclusions") or []]):
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                found.append(float(v))
    return found


def claim_numbers(claims: Iterable[dict[str, Any]]) -> list[float]:
    """Every number a run's claims carry: values, denominators, units, exclusion counts, and numbers in their filters,
    grouping, method, missing-reason, denominator and exclusion-reason texts (a reason the answer shows is claim text; T3
    review 2; an exclusion names its entries and says "and N more", coach 2026-10-24 12:50 UTC, NEXT 3)."""
    claims = list(claims)
    found = claim_values(claims)
    for c in claims:
        for text in _claim_texts(c):
            found += [n for n, *_ in _numbers(str(text))]
    return found


def _claim_texts(c: dict[str, Any]) -> tuple[Any, ...]:
    """A claim's texts: filters, grouping, method, missing reason, denominator name and exclusion reasons."""
    denominator = c.get("denominator") or {}
    return (*(c.get("filters") or {}).values(), *(c.get("grouping") or []), c.get("transformation") or "", c.get("missing_reason") or "",
            denominator.get("name") or "", *(e.get("reason") or "" for e in c.get("exclusions") or []))


def claim_names(claims: Iterable[dict[str, Any]]) -> list[float]:
    """The digits inside the names in the claims' texts (S2, name_numbers). They back a number against invention; they make no
    shown number a claim number (grade's wrong against no number reads claim_numbers alone)."""
    return [n for c in claims for text in _claim_texts(c) for n in name_numbers(str(text))]


def _backed(value: float, decimals: int, sources: list[float]) -> bool:
    """A shown number is backed when some source rounds to it at the shown precision ("2.38" backs 2.37857). An endless source
    backs nothing: an all-digit ID of 309 digits or more reads as inf, and abs(inf - x) <= inf (review 2026-10-22, round 2)."""
    half = 0.5 * 10 ** -decimals
    # float noise is relative: a fixed 1e-9 floor let any tiny claim back any tiny shown number (review of fixes (1)-(4))
    return any(math.isfinite(s) and abs(s - value) <= half + 1e-9 * abs(s) for s in sources)


def _match(expected: dict[str, Any], tolerance: dict[str, Any], shown: list[tuple[float, int, int, int]]) -> tuple[float, int, int, int] | None:
    """The first shown number within tolerance of one expected value. Under "display" each number may carry its own "abs"
    (half a unit of its own last decimal); a count without one is exact (the split's notes)."""
    target = float(expected["value"])
    rule = tolerance.get("rule")
    if rule == "display" and expected.get("abs") is None and expected.get("kind") == "count":
        rule = "exact"
    for number in shown:
        value = number[0]
        if rule == "exact" and value == target:
            return number
        if rule == "display" and abs(value - target) <= float(expected.get("abs") or tolerance["abs"]) + 1e-9:
            return number
        if rule == "relative" and abs(value - target) <= float(tolerance.get("rel", 0.005)) * abs(target) + 1e-12:
            return number
    return None


def _found_all(expected: list[dict[str, Any]], names: list[str], tolerance: dict[str, Any], text: str,
               shown: list[tuple[float, int, int, int]]) -> list[str] | None:
    """Where each expected value and name was found (for the reader to check), or None when one is missing."""
    lowered = text.casefold()
    where = []
    for e in expected:
        hit = _match(e, tolerance, shown)
        if hit is None:
            return None
        where.append(f"{expected_label(e)}: {harness._around(text, hit[2], hit[3], 50)}")
    for name in names:                                    # "A|B": either synonym (the split's notes)
        found = [option for option in name.split("|") if option.strip() and option.strip().casefold() in lowered]
        if not found:
            return None
        start = lowered.index(found[0].strip().casefold())
        where.append(f"{name}: {harness._around(text, start, start + len(found[0].strip()), 50)}")
    return where


def _note_sources(sources: list[float], text: str) -> list[float]:
    """Rule (3) (decision (31)(a)): a note number's sources are a checked-reply number's (sources) and the checked reply's own
    numbers (text: the checked reply as grade reads it) with the digits inside its names (S2). grade and note_numbers both take
    them from here."""
    return sources + [n for n, *_ in _numbers(text)] + name_numbers(text)


def note_invented(shown: str, sources: list[float]) -> list[dict[str, Any]]:
    """The numbers of the model's notes (harness.model_notes, each line read at the precision written) that no source backs, as
    grade reports an invented number, marked "in": "notes". Rule (3) (decision (31)(a)): grade gives as sources the checked
    reply's own backing and the checked reply's numbers."""
    found = []
    for line in harness.model_notes(shown):
        for v, d, start, end in _numbers(line, written=True):
            if not _backed(v, d, sources):
                found.append({"value": int(v) if float(v).is_integer() else v, "context": harness._around(line, start, end, 60),
                              "in": "notes"})
    return found


def expected_label(e: dict[str, Any]) -> str:
    return f"{e['value']:g}" if isinstance(e["value"], float) else str(e["value"])


def grade(sub: dict[str, Any], *, shown: str, status: str, claims: list[dict[str, Any]],
          backed_extra: Iterable[float] = (), run: bool = True, code_catalog: bool = False,
          code_replies: Iterable[str] = ()) -> dict[str, Any]:
    """One sub-question's bucket, from the chat's last message (shown), how the question ended (status, as harness.ask
    writes it), the run's claims and other numbers the run's own code or data wrote (backed_extra).
    run=False: no run answered. code_catalog=True: a list run's answer, written by code from the saved catalog.
    Neither is checked for invention against claims (there are none to check against); a reply in the model's own words
    (MODEL_REPLY_STATUS) is never scored and may hold only numbers from the question, unless it is byte-identical to one of
    code_replies (code_written_texts(): the line code writes then), whose numbers are not graded."""
    text = _without_quotes(harness.code_written(shown))
    asked = [n for n, *_ in _numbers(sub.get("question", ""))] + name_numbers(sub.get("question", ""))      # with its names' digits (S2)
    result: dict[str, Any] = {"bucket": "", "why": "", "invented": [], "matched": []}

    def flagged(numbers: list[tuple[float, int, int, int]], sources: list[float]) -> list[dict[str, Any]]:
        return [{"value": int(v) if float(v).is_integer() else v, "context": harness._around(text, start, end, 60)}
                for v, d, start, end in numbers if not _backed(v, d, sources)]

    if status.startswith(MODEL_REPLY_STATUS):
        if shown in ((code_replies,) if isinstance(code_replies, str) else tuple(code_replies)):
            return {**result, "bucket": "no_number", "why": CODE_REPLY_WHY, "code_written_reply": True}
        invented = flagged(_numbers(text), asked)
        if invented:
            return {**result, "bucket": "invented", "invented": invented,
                    "why": f"a reply in the model's words with {len(invented)} number(s) not in the question"}
        return {**result, "bucket": "no_number", "why": "a reply in the model's words (no run): never scored"}
    first = harness.no_answer(shown)
    question_back = status.startswith(_ASKED_STATUS) or (first is not None and first.startswith(_ASKED_FIRST_LINES))
    # The app's question screens answer nothing: their hint and choices (catalog counts) are written by code from the saved
    # catalog, the model's question is quoted and left out (baseline 2026-10-21: 44 false flags on them).
    if run and not code_catalog and not question_back:
        sources = claim_numbers(claims) + claim_names(claims) + asked + [float(x) for x in backed_extra]
        invented = flagged(_numbers_for_invention(text), sources)
        noted = note_invented(shown, _note_sources(sources, text))          # rule (3): the notes too
        if invented or noted:
            why = (f"{len(invented)} number(s) in the code-written answer and {len(noted)} in the model's notes that nothing backs"
                   if invented and noted else f"{len(noted)} number(s) in the model's notes that nothing backs" if noted
                   else f"{len(invented)} number(s) in the code-written answer that nothing backs")
            return {**result, "bucket": "invented", "invented": invented + noted, "why": why}
    if question_back:
        return {**result, "bucket": "no_number", "why": first or status}
    if status.startswith(_BLOCKED_STATUS) or first is not None:
        return {**result, "bucket": "blocked", "why": first or status}
    if not status.startswith(harness.SCORED_ENDINGS):
        return {**result, "bucket": "no_number", "why": f"no answer to score: {status}"}
    numbers = _numbers(text)
    tolerance = sub["tolerance"]
    where = _found_all(sub["expected"], sub.get("expected_text") or [], tolerance, text, numbers)
    if where is not None:
        return {**result, "bucket": "pass", "why": "every expected value found", "matched": where}
    for n, alternative in enumerate(sub.get("alternatives") or [], 1):
        where = _found_all(alternative.get("expected") or [], alternative.get("expected_text") or [],
                           alternative.get("tolerance") or tolerance, text, numbers)
        if where is not None:
            return {**result, "bucket": "pass", "why": f"alternative {n} from the grading notes found", "matched": where}
    known = claim_numbers(claims) if run and not code_catalog else []                  # once, not once per shown number
    backed_shown = [v for v, d, *_ in numbers if _backed(v, d, known)] if run and not code_catalog else numbers
    if backed_shown:
        return {**result, "bucket": "wrong", "why": "numbers shown, but not every expected value"}
    return {**result, "bucket": "no_number", "why": "no number for the statistic was shown"}


def code_written_texts() -> tuple[str, ...]:
    """The one no-run reply code writes whose numbers are not graded (coach 2026-10-24 12:50 UTC, the narrowest form): the
    front door's out_of_scope line with its example question (app.py front door -> triage.front_door_reply ->
    triage.out_of_scope_line), for the host the app names (app._source_name). Other replies code writes without a run also
    hold numbers (the greeting's "Questions to try" guide, an app question's answer) and stay graded."""
    return (triage.out_of_scope_line(harness.app._source_name()),)


def note_numbers(sub: dict[str, Any], *, shown: str, claims: list[dict[str, Any]], backed_extra: Iterable[float] = (),
                 run: bool = True) -> dict[str, Any]:
    """Every number in what the app marks as model text (harness.model_notes: the "Additional model notes (unverified)" bullets
    and the "Unverified model note:" paragraphs, which grade leaves out), read with grade's reader (_numbers: the app's own
    phrases, "80 groups" or "source 3", are the model's words in a note, so they are read too), each with its backing under
    two readings, both at the precision written: "rule3", rule (3) as grade applies it (decision (31)(a), Bidhan, coach
    2026-10-25 12:20 UTC: _note_sources, the checked reply's backing (claim_numbers, the question's numbers and backed_extra,
    the run's own tables and code-written fields; with no run, the question's alone) and the checked reply's own numbers), and
    "claim_value", the claims' numeric fields alone (claim_values). A row grade does not check for invention (a question screen,
    a catalog answer) is still listed."""
    claims = list(claims)
    asked = [n for n, *_ in _numbers(sub.get("question", ""))] + name_numbers(sub.get("question", ""))      # as grade reads it (S2)
    checked = claim_numbers(claims) + claim_names(claims) + asked + [float(x) for x in backed_extra] if run else asked
    rule3 = _note_sources(checked, _without_quotes(harness.code_written(shown)))
    values = claim_values(claims)
    numbers = []
    lines = harness.model_notes(shown)
    for line in lines:
        for v, d, start, end in _numbers(line, written=True):
            numbers.append({"value": int(v) if float(v).is_integer() else v, "context": harness._around(line, start, end, 60),
                            "rule3": _backed(v, d, rule3), "claim_value": _backed(v, d, values)})
    return {"lines": len(lines), "numbers": numbers, "count": len(numbers), "rule3_unbacked": sum(not n["rule3"] for n in numbers),
            "claim_unbacked": sum(not n["claim_value"] for n in numbers)}


def listed_notes(sub: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """note_numbers that never stops a run: a number the reader cannot hold ("0E0326" overflows _backed's precision) makes this
    row's listing an error, and the run goes on, since results.json is written only when a run ends (review 2026-10-25, S2)."""
    try:
        return note_numbers(sub, **kwargs)
    except Exception as exc:  # noqa: BLE001 - a measurement must not end a paid run
        return {"lines": len(harness.model_notes(str(kwargs.get("shown") or ""))), "numbers": [], "count": 0, "rule3_unbacked": 0,
                "claim_unbacked": 0, "error": " ".join(f"{type(exc).__name__}: {exc}".split())[:200]}


def note_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The listing's totals over a report's rows (a row without a listing counts as none)."""
    listed = [(r["sub_id"], r.get("note_numbers") or {}) for r in rows]
    return {"count": sum(n.get("count", 0) for _, n in listed), "rows_with_notes": sum(1 for _, n in listed if n.get("lines")),
            "rows_with_numbers": sum(1 for _, n in listed if n.get("count")),
            "rule3_unbacked": sum(n.get("rule3_unbacked", 0) for _, n in listed),
            "claim_unbacked": sum(n.get("claim_unbacked", 0) for _, n in listed),
            "rows_rule3_unbacked": [sid for sid, n in listed if n.get("rule3_unbacked")],
            "rows_claim_unbacked": [sid for sid, n in listed if n.get("claim_unbacked")],
            "rows_error": [sid for sid, n in listed if n.get("error")]}


def _rows(n: int) -> str:
    return f"{n} row" if n == 1 else f"{n} rows"


def note_line(summary: dict[str, Any]) -> str:
    """The listing's line in results.md and note_numbers.md."""
    return (f"- Note numbers (the model's notes): {summary['count']} in {_rows(summary['rows_with_numbers'])}, "
            f"of {_rows(summary.get('rows_with_notes', 0))} with notes; no backing under rule (3): {summary['rule3_unbacked']}"
            + (f" ({', '.join(summary['rows_rule3_unbacked'])})" if summary["rows_rule3_unbacked"] else "")
            + f"; no claim value: {summary['claim_unbacked']} (in {_rows(len(summary['rows_claim_unbacked']))})"
            + (f"; listing failed: {', '.join(summary['rows_error'])}" if summary.get("rows_error") else ""))


def invented_summary(report: dict[str, Any]) -> str:
    """Both counts, always (coach 2026-10-24 12:50 UTC): the invented rows, then the code-written replies not graded."""
    code = report.get("code_written_replies") or []
    replies = f"{len(code)} code-written {'reply' if len(code) == 1 else 'replies'}, numbers not graded"
    return f"{', '.join(report['invented']) or 'none'} ({replies}{': ' + ', '.join(code) if code else ''})"


def run_numbers(run_dirs: Iterable[Path]) -> list[float]:
    """Numbers the runs' own code and data wrote: every cell of their saved tables (artifacts/*.csv), the numbers in their
    code-written report fields (caveats, warnings, notes, exclusions) and the plain-digit study IDs of the study ledger (a
    refused fetch is reported as "Study 4777: ..."; variance run of 2026-10-21). Model text is not among them: a ledger
    entry's reason quotes the model's arguments, and its study_id is the model's own argument, so only an all-digit ID counts
    (review 2026-10-21: "mean=12.5" as a study ID must stay unbacked); a report line that quotes the model (_MODEL_QUOTED)
    backs nothing, and the planner draft's quoted IDs and questions in coordinator notes (_DRAFT_LINES) are blanked first.
    The digits inside the names a cell or a field holds are among them (S2, name_numbers)."""
    found: list[float] = []
    for run_dir in run_dirs:
        for table in sorted((Path(run_dir) / "artifacts").glob("*.csv")):
            with table.open(encoding="utf-8", newline="") as handle:
                for row in csv.reader(handle):
                    for cell in row:
                        found += [n for n, *_ in _numbers(cell)] + name_numbers(cell)
        answer = Path(run_dir) / "answer.json"
        if answer.is_file():
            doc = json.loads(answer.read_text(encoding="utf-8"))
            texts = [*((doc.get("analysis") or {}).get("caveats") or []), *((doc.get("retrieval") or {}).get("warnings") or []),
                     *(doc.get("notes") or [])]
            found += [float(entry["study_id"]) for entry in (doc.get("retrieval") or {}).get("study_ledger") or []
                      if re.fullmatch(r"[0-9]+", str(entry.get("study_id") or ""))]
            for text in map(str, texts):
                if text.lower().startswith(_MODEL_QUOTED):
                    continue
                if text.lower().startswith(_DRAFT_LINES):
                    text = _QUOTED_PART.sub(" ", text)
                found += [n for n, *_ in _numbers(text)] + name_numbers(text)
    return found


_TRAIT_QUESTION = "More than one trait matches the question:"
_TRAIT_OPTION = re.compile(r"(\d+) \(([^()|]+)\|(CO_\d+:\d+)\)")       # "76619 (Content of beta-carotene ...|CO_331:0000289)"


def trait_answer(question: str, shown: str) -> str | None:
    """The benchmark's answer to the planner's "More than one trait matches the question: ..." (coach, 2026-10-22 15:53 UTC;
    harness-only, the app is unchanged): the listed variable(s) the sub-question names, by their ontology term (|CO_331:...),
    else by their whole name in any case, in the listed order and joined by "and". None when the chat shows another question
    or the sub-question names none of the listed variables: then the run stops there, as before."""
    if _TRAIT_QUESTION not in shown:
        return None
    options = _TRAIT_OPTION.findall(shown.split(_TRAIT_QUESTION, 1)[1].split("Which one?", 1)[0])
    chosen = [vid for vid, _name, term in options if term in question]
    if not chosen:
        lowered = question.lower()
        chosen = [vid for vid, name, _term in options if name.strip().lower() in lowered]
    return " and ".join(dict.fromkeys(chosen)) or None


def run_subquestions(subs: list[dict[str, Any]], *, cache_copy: Path, out_dir: Path,
                     base_config: Callable[[str], Any], make_models: Callable[[], Any], triage_model: Callable[[], Any] | None,
                     preflight: Callable[[str], list[str]] | None, allow_live: bool, wait: float = 120.0) -> dict[str, Any]:
    make_config = harness.eval_config(base_config, cache_copy, out_dir)
    code_replies = code_written_texts()
    saved_transport, saved_job = harness.brapi_client.RequestsTransport, harness.app.Job
    harness.app.Job = harness.TrackedJob
    if not allow_live:
        harness.brapi_client.RequestsTransport = harness.RefusingTransport
    rows, threads = [], []
    try:
        for s in subs:
            counter = harness.CountingFactory(triage_model) if triage_model is not None else None
            harness.RefusingTransport.refused = 0
            started = time.monotonic()
            try:
                outcome = harness.ask(s["question"], dict(make_config=make_config, make_models=make_models, preflight=preflight,
                                                          triage_model=counter), wait=wait,
                                      answer_clarification=lambda shown, question=s["question"]: trait_answer(question, shown))
            except Exception as exc:  # noqa: BLE001 - one broken question must not stop the others
                outcome = harness.Outcome(status=f"error: {type(exc).__name__}: {exc}")
            seconds = time.monotonic() - started
            threads += outcome.threads
            claims = [c.model_dump(mode="json") for r in outcome.results if r.analysis for c in r.analysis.claims]
            ran = bool(outcome.results)
            catalog_run = any(step.action in CATALOG_ACTIONS for r in outcome.results if r.plan for step in r.plan.steps)
            extra = run_numbers(outcome.run_dirs) if ran else ()
            graded = grade(s, shown=outcome.shown, status=outcome.status, claims=claims, backed_extra=extra, run=ran,
                           code_catalog=catalog_run, code_replies=code_replies)
            run_calls = sum(r.counts.model_calls for r in outcome.results)
            tokens = harness.tally((counter.replies if counter else []) + harness.run_replies(outcome.run_dirs),
                                   (counter.calls if counter else 0) + run_calls)
            rows.append({"sub_id": s["sub_id"], "parent": s["parent"], "tier": s["tier"], "question": s["question"],
                         "expected": s["expected"], "expected_text": s.get("expected_text") or [], **graded,
                         "status": outcome.status, "answer": outcome.shown[:harness.GOT_CHARS], "answer_full": outcome.shown,
                         "typed": outcome.typed, "answered_clarifications": list(outcome.answered),
                         "claims": [{k: c.get(k) for k in ("kind", "value", "unit", "filters", "transformation")} for c in claims],
                         "model_calls": (counter.calls if counter else 0) + run_calls,
                         "live_attempts": sum(r.counts.live_attempts for r in outcome.results),
                         "refused_requests": 0 if allow_live else harness.RefusingTransport.refused,
                         "run_ids": [r.run_id for r in outcome.results], "tokens": tokens, "seconds": round(seconds, 2),
                         "caveat": s.get("caveat"),
                         "note_numbers": listed_notes(s, shown=outcome.shown, claims=claims, backed_extra=extra, run=ran)})
            say(f"{s['sub_id']:8} {s['tier']} {graded['bucket']:9} {seconds:6.1f}s  {graded['why'][:80]}", flush=True)
    finally:
        harness.app.Job = saved_job
        if allow_live or not [th for th in threads if th.is_alive()]:
            harness.brapi_client.RequestsTransport = saved_transport
    costs = [r["tokens"]["cost_usd"] for r in rows]
    by_tier = {t: {b: sum(1 for r in rows if r["tier"] == t and r["bucket"] == b) for b in BUCKETS} for t in TIERS}
    return {"cost_usd": None if None in costs else round(sum(costs), 6), "by_tier": by_tier,
            "invented": [r["sub_id"] for r in rows if r["bucket"] == "invented"],
            "code_written_replies": [r["sub_id"] for r in rows if r.get("code_written_reply")],
            "note_numbers": note_summary(rows), "subquestions": rows,
            "requests": "allowed after the scripted approval" if allow_live else "none: cache copy only"}


def results_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Benchmark run, {report['started_utc']}", "", f"- Models: {report['models']}",
             f"- Cache copy: `{report['cache_copy']}`; database requests: {report['requests']}",
             f"- Cost: {report['cost_usd']} US dollars at the prices in eval/harness.py PRICES (an upper bound for OpenAI)",
             f"- INVENTED (red flag): {invented_summary(report)}",
             *([note_line(report["note_numbers"])] if report.get("note_numbers") else []), "",
             "| Tier | Sub-questions | pass | wrong | no number | blocked | invented |", "|---|---|---|---|---|---|---|"]
    for tier, counts in report["by_tier"].items():
        total = sum(counts.values())
        if total:
            lines.append(f"| {tier} | {total} | " + " | ".join(str(counts[b]) for b in BUCKETS) + " |")
    lines += ["", "A pass is any number within tolerance anywhere in the code-written answer: read Got before trusting it.", "",
              "| Sub | Tier | Bucket | Why | Got | Answered clarifications | Model calls | Cost ($) | Seconds | Status |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in report["subquestions"]:
        cost = r["tokens"]["cost_usd"]
        got = "; ".join(r.get("matched") or []) or "; ".join(f"{i['value']} in \"{i['context']}\"" for i in r.get("invented") or [])
        answered = "; ".join(a.removeprefix("clarification answered by harness: ") for a in r.get("answered_clarifications") or [])
        lines.append(f"| {r['sub_id']} | {r['tier']} | {r['bucket']} | {harness._cell(r['why'], 90)} | {harness._cell(got, 120)} | "
                     f"{harness._cell(answered, 40)} | {r['model_calls']} | {'' if cost is None else f'{cost:.6f}'} | {r['seconds']} | "
                     f"{harness._cell(r['status'], 60)} |")
    skipped = report.get("not_checkable") or []
    if skipped:
        lines += ["", "Not checkable (not run): " + ", ".join(skipped)]
    return "\n".join(lines) + "\n"


# The budgets the benchmark may pass per run, above the app's own (5 fetches, 2 pages per plot-record fetch, 12 plan steps).
# Bidhan, 2026-10-21 ~17:58 UTC (answering Q1-Q3 of the baseline report): "yes, as benchmark-only settings the harness passes per
# run (24 fetches, 10 pages, 30 steps); the app's defaults stay 5 / 1 / 12". Pages 10 -> 20: Bidhan, 2026-10-23 (coach's DECISION
# entry 14:01 UTC, item 19), benchmark only. The HTTP attempt cap of the approval follows them: fetches x pages, never below the
# app's own. The time budget is not raised.
BUDGET_LIMITS = {"max_fetches": 24, "max_pages": 20, "max_plan_steps": 30}


def budgeted(base_config: Callable[[str], Any], *, max_fetches: int | None = None, max_pages: int | None = None,
             max_plan_steps: int | None = None) -> Callable[[str], Any]:
    """The run settings with the benchmark's budgets, or the settings unchanged when none is given. A budget above what Bidhan
    allowed, or below 1, is refused."""
    budgets = {"max_fetches": max_fetches, "max_pages": max_pages, "max_plan_steps": max_plan_steps}
    for name, value in budgets.items():
        if value is not None and not 1 <= value <= BUDGET_LIMITS[name]:
            raise ValueError(f"{name} must be between 1 and {BUDGET_LIMITS[name]}, not {value}")
    if all(value is None for value in budgets.values()):
        return base_config

    def make(question: str) -> Any:
        config = base_config(question)
        fetches = max_fetches or config.max_fetches
        pages = max_pages or config.max_observation_pages
        return dataclasses.replace(config, max_fetches=fetches, max_observation_pages=pages,
                                   max_plan_steps=max_plan_steps or config.max_plan_steps,
                                   max_http_attempts=max(config.max_http_attempts, fetches * pages))
    return make


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the benchmark's sub-questions through the app and grade them by code.")
    parser.add_argument("--cache", type=Path, help="the cache folder to copy (it is never changed); needed for a run")
    parser.add_argument("--note-numbers", type=Path, metavar="FOLDER",
                        help="no run: list the model-note numbers of a saved results folder into note_numbers.json and .md")
    parser.add_argument("--subquestions", type=Path, default=SUBQUESTIONS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--tiers", default="T1,T2,T3", help="comma-separated tiers, run in this order")
    parser.add_argument("--only", default="", help="comma-separated sub_ids")
    parser.add_argument("--allow-live", action="store_true", help="let the app fetch what the copy lacks, after the scripted approval")
    parser.add_argument("--confirm", action="store_true", help="really run (uses the configured model; costs model calls)")
    parser.add_argument("--max-fetches", type=int, default=None, help="benchmark budget: fetch calls per question (app: 5; at most 24)")
    parser.add_argument("--max-pages", type=int, default=None, help="benchmark budget: pages per plot-record fetch (app: 2; at most 20)")
    parser.add_argument("--max-plan-steps", type=int, default=None, help="benchmark budget: steps per plan (app: 12; at most 30)")
    return parser.parse_args(argv)


def budgets_record(args: argparse.Namespace, make_config: Callable[[str], Any]) -> dict[str, int]:
    """What results.json says about the budgets the runs had ({} = the app's own settings), read from the same settings the
    runs got (review of 2026-10-22: not recomputed from the live settings)."""
    if args.max_fetches is None and args.max_pages is None and args.max_plan_steps is None:
        return {}
    config = make_config("budget record")
    return {"max_fetches": config.max_fetches, "max_observation_pages": config.max_observation_pages,
            "max_plan_steps": config.max_plan_steps, "max_http_attempts": config.max_http_attempts}


def list_note_numbers(folder: Path) -> int:
    """--note-numbers: the listing for a saved results folder, from its own files (each row's question and full answer, the full
    claims of its runs' answer.json, run_numbers of its run folders), written to note_numbers.json and note_numbers.md beside
    results.json, which is left as it is. No run, no model call, no request. Each listing says what it read: "answer_source"
    (an old row without answer_full has only the answer cut at harness.GOT_CHARS, and notes come last) and "runs_missing" (run
    folders not found: their claims and tables cannot back anything) (review 2026-10-25, S3)."""
    if not (folder / "results.json").is_file():
        print(f"no results.json in {folder}")
        return 2
    report = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(report.get("subquestions"), list):
        print(f"no benchmark rows in {folder / 'results.json'}")
        return 2
    listed: dict[str, Any] = {}
    for row in report["subquestions"]:
        ids = row.get("run_ids") or []
        run_dirs = [folder / "out" / rid for rid in ids if (folder / "out" / rid).is_dir()]
        claims: list[dict[str, Any]] = []
        answers = [run_dir / "answer.json" for run_dir in run_dirs if (run_dir / "answer.json").is_file()]
        for answer in answers:
            claims += (json.loads(answer.read_text(encoding="utf-8")).get("analysis") or {}).get("claims") or []
        full, shown = row.get("answer_full"), row.get("answer") or ""
        source = "answer_full" if full is not None else (f"answer (cut at {harness.GOT_CHARS} characters)"
                                                         if len(shown) >= harness.GOT_CHARS else "answer")
        sid = row["sub_id"]
        key = sid if sid not in listed else f"{sid} #{sum(k.split(' #')[0] == sid for k in listed) + 1}"      # a repeated sub-question
        listed[key] = {**listed_notes(row, shown=full if full is not None else shown, claims=claims,
                                      backed_extra=run_numbers(run_dirs) if ids else (), run=bool(ids)),
                       "answer_source": source, "runs_missing": len(ids) - len(run_dirs), "answers_missing": len(run_dirs) - len(answers)}
    summary = {**note_summary([{"sub_id": sid, "note_numbers": n} for sid, n in listed.items()]),
               "rows_cut_answer": [sid for sid, n in listed.items() if n["answer_source"].startswith("answer (cut")],
               "rows_runs_missing": [sid for sid, n in listed.items() if n["runs_missing"]],
               "rows_answers_missing": [sid for sid, n in listed.items() if n["answers_missing"]]}
    (folder / "note_numbers.json").write_text(json.dumps({"summary": summary, "rows": listed}, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    gaps = (summary["rows_cut_answer"], summary["rows_runs_missing"], summary["rows_answers_missing"])
    warning = (f"- Read from a cut answer: {', '.join(gaps[0]) or 'none'}; run folders missing: {', '.join(gaps[1]) or 'none'}; "
               f"runs without answer.json: {', '.join(gaps[2]) or 'none'}" if any(gaps) else "")
    lines = [f"# Note numbers, {folder.resolve().name}", "", note_line(summary)] + ([warning] if warning else []) + [""]
    lines += [f"- {sid}: {n['count']} number{'' if n['count'] == 1 else 's'}; no backing under rule (3): "
              f"{', '.join(str(x['value']) for x in n['numbers'] if not x['rule3']) or 'none'}; no claim value: {n['claim_unbacked']}"
              + (f"; listing failed: {n['error']}" if n.get("error") else "") for sid, n in listed.items() if n["count"] or n.get("error")]
    (folder / "note_numbers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{note_line(summary)}{' ' + warning if warning else ''} -> {folder / 'note_numbers.md'}")
    return 0


def main(argv: list[str] | None = None, *, base_config: Callable[[str], Any] | None = None,
         make_models: Callable[[], Any] | None = None, triage_model: Callable[[], Any] | None = None,
         preflight: Callable[[str], list[str]] | None = None) -> int:
    args = parse_args(argv)
    if args.note_numbers is not None:
        return list_note_numbers(args.note_numbers)
    if args.cache is None:
        print("--cache is needed for a run")
        return 2
    subs = load_subquestions(args.subquestions)
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    not_checkable = [s["sub_id"] for s in subs if not s.get("checkable", True)]
    subs = sorted((s for s in subs if s["tier"] in tiers and s.get("checkable", True)), key=lambda s: tiers.index(s["tier"]))
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        subs = [s for s in subs if s["sub_id"] in wanted]
    if not args.cache.is_dir():
        print(f"no cache folder at {args.cache}")
        return 2
    scripted = make_models is not None
    if not args.confirm:
        print(f"Would run {len(subs)} sub-questions ({', '.join(tiers)}) on a copy of {args.cache}; database requests: "
              f"{'allowed after the scripted approval' if args.allow_live else 'none (cache copy only)'}. Add --confirm to run.")
        return 2
    started = datetime.now(timezone.utc)
    folder = args.results / ("benchmark_" + started.strftime("%Y%m%dT%H%M%SZ"))
    folder.mkdir(parents=True, exist_ok=False)
    cache_copy = folder / "cache"
    shutil.copytree(args.cache, cache_copy)
    config = budgeted(base_config or harness.app.live_config, max_fetches=args.max_fetches, max_pages=args.max_pages,
                      max_plan_steps=args.max_plan_steps)
    report = run_subquestions(subs, cache_copy=cache_copy, out_dir=folder / "out", base_config=config,
                              make_models=make_models or harness.app.default_models,
                              triage_model=triage_model if scripted else harness.app.default_triage_model,
                              preflight=preflight, allow_live=args.allow_live)
    models = "scripted" if scripted else "configured model: " + (", ".join(sorted({m for r in report["subquestions"]
                                                                                    for m in r["tokens"].get("models", [])})) or "see run logs")
    report = {"started_utc": started.isoformat(), "models": models, "cache_copy": str(cache_copy), "not_checkable": not_checkable,
              "budgets": budgets_record(args, config), **report}
    (folder / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (folder / "results.md").write_text(results_markdown(report), encoding="utf-8")
    print(f"{sum(c['pass'] for c in report['by_tier'].values())} pass of {len(report['subquestions'])}; invented: "
          f"{invented_summary(report)} -> {folder / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
