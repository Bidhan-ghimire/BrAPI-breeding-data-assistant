"""
eval/harness.py — the answer-quality harness (overnight Item 4, 2026-10-16).

What it does, in plain words:
* Reads a question file (eval/questions.json): real breeder questions, each with the KIND of answer it expects (for
  example "how many studies name the season 2026"). The file never stores an answer.
* Copies the given cache folder into a new results folder (eval/results/<UTC time>/cache) and computes every expected
  answer BY CODE from that copy: from the raw study and trait catalog records, and from the cached measurements of a
  study (on a server whose measurements the app reads from the plot records, brapi_client.PLOT_RECORD_HOSTS:
  sweetpotatobase.org and cassavabase.org, from its cached plot records, which keep the values recorded as 0). Nothing is requested for this. A question whose expected answer cannot be computed from the copy (for example,
  its measurements are not in it) is skipped, not run.
* Feeds each question, as a new chat, through the app's own pipeline (app.respond: the front door, the planner, the
  approvals, the retrieval, the analysis) on that copy, with a scripted approver that types only two words: "approve" at
  the data request's approval screen and "continue" at the data review. It never approves a catalog setup, never answers
  a question the app asks, never says "yes" to "look it up?", and never accepts or rejects: at the final review it ends
  the run the way the app ends a waiting run, so the review stays "pending".
* Compares the answer the person sees (the chat's last message for the question, its code-written part only) with the
  expected answer, and writes results.json and results.md: question, expected, got, pass or fail, model calls, seconds,
  tokens and cost.
* Tokens and cost: every model reply's token counts, from the runs' own logs (llm_*.jsonl, one line per reply) and, for
  the front door, which logs nothing, from its replies as they pass. Input counts everything the model read (Claude:
  input + cache writes + cache reads, kept apart for pricing; OpenAI: the prompt, whose cached part the app does not keep,
  so its cost is an upper bound); output includes thinking. Cost = tokens x PRICES, by the model each reply names. A call
  with no counts (a provider error, a timeout) is counted in "calls without counts" and makes the cost a lower bound
  ("at least"); a model with no price gives no cost ("?").
* Requests: by default nothing goes to the database. A request for anything not in the cache copy is refused and counted
  (--allow-live lets the app fetch after the scripted approval, as it would in the chat). Models: the app's configured
  model, so a real run is Bidhan's own command with --confirm. Without --confirm the harness only prints what it would
  do. The tests inject scripted models and run it end to end offline.

"pass" means the expected value appears in the answer as the person sees it (a name with its count; a mean as the number
right after "Mean"). Only the chat's last message for the question counts, and only its code-written part: what the app
marks as model text ("Additional model notes (unverified)" and its bullets, "Unverified model note: ...") is left out, and
numbers recorded inside the run (its typed claims) do not count; they stay in results.json for reading. Eval baseline
20261016T150241Z: q04 passed on a recorded 22 while the person saw "Question blocked". Only an answer written by code is
scored (a run's answer, a list, a catalog answer): a question that ended with the app asking something, offering a lookup,
replying in a model's words, or showing that a run stopped ("Question blocked.", "Incomplete result." ...) showed no answer
and fails. Dates and times, such as the catalog's fetch time "2026-09-28 22:49", are not read as numbers (review,
2026-10-16: its 22 was q04's expected 22). Read the "got" column before trusting a pass: a number can appear for another
reason.
"""
from __future__ import annotations

import argparse
import dataclasses
import html
import json
import math
import re
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import app  # noqa: E402
import brapi_client  # noqa: E402
import llm  # noqa: E402
from brapi_client import BrapiClient, TransportError, load_settings  # noqa: E402
from cache_store import CacheStore  # noqa: E402
from run import (ANALYST_PROMPT, CATALOG_PROMPT, CLARIFY_PROMPT, DATA_REVIEW_PROMPT, RETRIEVER_PROMPT,  # noqa: E402
                 SUPERVISOR_PROMPT, RunConfig, RunResult)

QUESTIONS = Path(__file__).resolve().parent / "questions.json"
RESULTS = Path(__file__).resolve().parent / "results"
MAX_PAGES = 20             # catalog and measurement pages read from the copy (as a run's cache read)
MAX_REPLIES = 6            # the most words the scripted approver types for one question
GOT_CHARS = 1500           # the most of an answer kept in results.json (results.md shows less)
MISSING = frozenset({"", "NA", "N/A", "NULL", "null", "None", "."})
_PLAIN_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_NUMBER_IN_TEXT = re.compile(r"(?<![\w.])[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.])[+-]?\d+(?:\.\d+)?(?![\w])")
_MARKDOWN_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~<])")
_MODEL_NOTES_HEADING = "Additional model notes (unverified)"    # chat_presenter._limit_blocks: model text, as bullets under it
_MODEL_NOTE = "Unverified model note:"                         # chat_presenter._caveats: one model note per paragraph
_MEAN_LABEL = re.compile(r"\bmean\b[ \t:=|]*", re.IGNORECASE)    # "Mean: 2.38", or a table row "| Mean | 2.38 ..."
_DATE_TIME = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?|(?<![\w.:])\d{1,2}:\d{2}(?::\d{2})?(?![\w:])")
# The first line of an app message that gives no answer (chat_presenter.answer_summary for a run that did not complete;
# app.render and app.respond when a run stops or cannot start): whatever numbers it carries, it answers nothing.
NO_ANSWER = ("Incomplete result.", "Could not complete this question", "Question blocked.", "Question canceled.",
             "More detail is needed.", "A request limit was reached.", "No verified answer is available.", "The run stopped:",
             "This question cannot run yet:", "This question could not start.")
# How a question ended when the chat's last message was written by code for it: an answer to score. Anything else (the app
# asked a question, offered a lookup or replied in a model's words; the run stopped at a step the harness never takes)
# showed no answer.
SCORED_ENDINGS = ("final review reached", "run ended", "no run: answered by code from the saved catalog")
INTEGER_KINDS = ("study_count", "trait_count", "location_count", "studies_with_season", "studies_at_location",
                 "traits_containing", "studies_of_type", "valid_count")
NAME_KINDS = ("top_location", "top_study_type")
KINDS = INTEGER_KINDS + NAME_KINDS + ("seasons_present", "studies_per_type", "pooled_mean")


# --------------------------------------------------------------------------
# Expected answers, computed by code from the cache copy (raw records; not the app's own summaries)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Catalog:
    studies: list[dict[str, Any]]
    variables: list[dict[str, Any]]
    cache_dir: Path
    base_url: str
    fetched_at: list[str]


def _settings(cache_dir: Path) -> Any:
    return dataclasses.replace(load_settings(), mode="offline", cache_dir=cache_dir)


def read_catalog(cache_dir: Path) -> Catalog | None:
    """The raw study and trait records of the cache copy, read by an offline client (never a request). None unless both
    catalogs are completely cached."""
    settings = _settings(cache_dir)
    client = BrapiClient(settings, cache=CacheStore(cache_dir), run_id="eval_expected")
    studies = client.studies(None, approval=None, max_pages=MAX_PAGES)
    variables = client.variables(approval=None, max_pages=MAX_PAGES)
    if any(result.status == "failed" or not result.complete for result in (studies, variables)):
        return None
    fetched = sorted({r.fetched_at_utc.strftime("%Y-%m-%d %H:%M UTC") for r in client.provenance() if r.fetched_at_utc})
    return Catalog(list(studies.records), list(variables.records), cache_dir, settings.base_url, fetched)


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _season_names(study: dict[str, Any]) -> list[str]:
    names = []
    for season in study.get("seasons") or []:
        if isinstance(season, dict):
            season = season.get("seasonName") or season.get("season") or season.get("seasonDbId") or season.get("year")
        if _text(season):
            names.append(_text(season))
    return names


def _variable_name(variable: dict[str, Any]) -> str:
    """The trait's name as breeders read it: the variable name before its ontology suffix ("...|CO_331:0000789")."""
    return _text(str(variable.get("observationVariableName") or "").split("|")[0])


def _studies_by_location(catalog: Catalog) -> dict[str, tuple[set[str], set[str]]]:
    """location key -> (its names, its study IDs)."""
    places: dict[str, tuple[set[str], set[str]]] = {}
    for study in catalog.studies:
        key, name = _text(study.get("locationDbId")), _text(study.get("locationName"))
        if not key and not name:
            continue
        names, ids = places.setdefault(key or f"name:{name}", (set(), set()))
        if name:
            names.add(name)
        ids.add(_text(study.get("studyDbId")))
    return places


def _types(catalog: Catalog) -> dict[str, int]:
    counts: dict[str, int] = {}
    for study in catalog.studies:
        kind = _text(study.get("studyType"))
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def _top(counts: dict[str, int]) -> list[str]:
    best = max(counts.values(), default=0)
    return sorted(name for name, count in counts.items() if count == best) if best else []


def _whole_word(word: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(word.strip())}(?!\w)", re.IGNORECASE)


def _valid_values(rows: list[dict[str, Any]]) -> list[float]:
    """The values that read as finite plain numbers; missing tokens and text are left out."""
    values = []
    for row in rows:
        text = str(row.get("value") if row.get("value") is not None else "").strip()
        if text in MISSING or not _PLAIN_NUMBER.match(text):
            continue
        number = float(text)
        if math.isfinite(number):
            values.append(number)
    return values


def _measurements(catalog: Catalog, study_name: str, variable_name: str) -> list[float] | str:
    """The valid values of one variable in one study, from the cached measurements; or why they cannot be read. On a server
    whose measurements the app reads from the plot records (brapi_client.PLOT_RECORD_HOSTS, whose /observations leaves out the
    values recorded as 0) they come from the cached plot records, as the app's get_observations reads them (A6, coach
    2026-10-22 17:47 UTC); a copy without them gives no expected value, never one from the /observations without the zeros."""
    studies = [s for s in catalog.studies if _text(s.get("studyName")).casefold() == _text(study_name).casefold()]
    variables = [v for v in catalog.variables if _variable_name(v).casefold() == _text(variable_name).casefold()]
    if len({_text(s.get("studyDbId")) for s in studies}) != 1:
        return f"study {study_name!r} names {len(studies)} catalog rows, not exactly one"
    if len({_text(v.get("observationVariableDbId")) for v in variables}) != 1:
        return f"variable {variable_name!r} names {len(variables)} catalog rows, not exactly one"
    client = BrapiClient(_settings(catalog.cache_dir), cache=CacheStore(catalog.cache_dir), run_id="eval_expected")
    plot_records = brapi_client.reads_plot_records(catalog.base_url)
    fetch = client.plot_records if plot_records else client.observations
    result = fetch(_text(studies[0].get("studyDbId")), _text(variables[0].get("observationVariableDbId")),
                   approval=None, paging=True, max_pages=MAX_PAGES)
    if result.status == "failed" or not result.complete:
        return f"its {'plot records are' if plot_records else 'measurements are'} not completely in the cache copy"
    return _valid_values(list(result.records))


def expected_answer(spec: dict[str, Any], catalog: Catalog) -> tuple[Any, str]:
    """(the expected value, how it reads in results.md). Raises ValueError when the cache copy cannot give it."""
    kind = spec["kind"]
    if kind == "study_count":
        value = len({_text(s.get("studyDbId")) for s in catalog.studies})
    elif kind == "trait_count":
        value = len({_text(v.get("observationVariableDbId")) for v in catalog.variables})
    elif kind == "location_count":
        value = len(_studies_by_location(catalog))
    elif kind == "studies_with_season":
        year = _whole_word(spec["season"])
        value = len({_text(s.get("studyDbId")) for s in catalog.studies if any(year.search(n) for n in _season_names(s))})
    elif kind == "studies_at_location":
        word = _whole_word(spec["word"])
        value = len({_text(s.get("studyDbId")) for s in catalog.studies if word.search(_text(s.get("locationName")))})
    elif kind == "traits_containing":
        word = spec["word"].casefold()
        value = len({_text(v.get("observationVariableDbId")) for v in catalog.variables if word in _variable_name(v).casefold()})
    elif kind == "studies_of_type":
        value = sum(count for name, count in _types(catalog).items() if name.casefold() == spec["type"].casefold())
    elif kind == "top_location":
        counts = {" / ".join(sorted(names)): len(ids) for names, ids in _studies_by_location(catalog).values() if names}
        value = {name: counts[name] for name in _top(counts)}
        return value, " or ".join(f"{name} ({count} studies)" for name, count in value.items())
    elif kind == "top_study_type":
        counts = _types(catalog)
        value = {name: counts[name] for name in _top(counts)}
        return value, " or ".join(f"{name} ({count} studies)" for name, count in value.items())
    elif kind == "seasons_present":
        value = sorted({n for s in catalog.studies for n in _season_names(s)})
        return value, ", ".join(value)
    elif kind == "studies_per_type":
        value = dict(sorted(_types(catalog).items(), key=lambda t: (-t[1], t[0].casefold())))
        return value, "; ".join(f"{name}: {count}" for name, count in value.items())
    elif kind in ("pooled_mean", "valid_count"):
        values = _measurements(catalog, spec["study"], spec["variable"])
        if isinstance(values, str):
            raise ValueError(values)
        if kind == "valid_count":
            return len(values), str(len(values))
        if not values:
            raise ValueError("no valid values in the cached measurements")
        mean = statistics.fmean(values)
        return mean, f"{mean:.4f} (from {len(values)} valid values)"
    else:
        raise ValueError(f"unknown expectation kind {kind!r}")
    return value, str(value)


# --------------------------------------------------------------------------
# Comparing what the chat showed with the expected answer
# --------------------------------------------------------------------------

def plain(markdown: str) -> str:
    """The answer as a reader sees it: markdown escapes and HTML entities undone, bold markers removed."""
    return _MARKDOWN_ESCAPE.sub(r"\1", html.unescape(markdown or "")).replace("**", "")


def _model_text_split(markdown: str) -> tuple[list[str], list[str]]:
    """(kept, notes): the plain answer's lines, split where the app marks model text. The "Additional model notes
    (unverified)" heading opens a block of bullet lines, which blank lines do not end and the first other line does; every
    paragraph that starts "Unverified model note:" is a note too. The heading and the block's blank lines are in neither."""
    kept: list[str] = []
    notes: list[str] = []
    in_notes = False
    for line in plain(markdown).split("\n"):        # as chat_presenter writes lines (not splitlines: U+2028 is no new line)
        text = line.strip()
        if text == _MODEL_NOTES_HEADING:
            in_notes = True
            continue
        if in_notes and (not text or text.startswith("- ")):
            if text:
                notes.append(line)
            continue
        in_notes = False
        (notes if text.startswith(_MODEL_NOTE) else kept).append(line)
    return kept, notes


def code_written(markdown: str) -> str:
    """The answer as a reader sees it (plain), without what the app marks as model text: the "Additional model notes
    (unverified)" heading with the bullet lines under it, and every paragraph that starts "Unverified model note:"."""
    return "\n".join(_model_text_split(markdown)[0])


def model_notes(markdown: str) -> list[str]:
    """The lines code_written leaves out as model text, in order (_model_text_split)."""
    return _model_text_split(markdown)[1]


def no_answer(markdown: str) -> str | None:
    """The first line of a message that gives no answer (NO_ANSWER), or None. The line app.respond puts first when the run
    had already ended ("This question had already ended, ...") is passed over: the run's own message follows it."""
    lines = [line.strip() for line in plain(markdown).split("\n") if line.strip()]
    if lines and lines[0].startswith("This question had already ended"):
        lines = lines[1:]
    first = lines[0] if lines else ""
    return first if first.startswith(NO_ANSWER) else None


def _without_dates(text: str) -> str:
    """The text with every date and time blanked out (same length): "fetched 2026-09-28" names no season 2026."""
    return _DATE_TIME.sub(lambda match: " " * len(match.group()), text)


def numbers_in(text: str) -> list[tuple[float, int, int, int]]:
    """Every number in the text: (value, decimals, start, end). "2,039" is 2039; "2.38" has 2 decimals. A date or a time
    ("2026-09-28 22:49") is not a number here."""
    found = []
    dates = [match.span() for match in _DATE_TIME.finditer(text)]
    for match in _NUMBER_IN_TEXT.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in dates):
            continue
        token = match.group(0).replace(",", "")
        found.append((float(token), len(token.split(".")[1]) if "." in token else 0, match.start(), match.end()))
    return found


def _around(text: str, start: int, end: int, width: int = 90) -> str:
    """The words around a match, on one line: what the "got" column shows."""
    left, right = max(0, start - width), min(len(text), end + width)
    return ("…" if left else "") + " ".join(text[left:right].split()) + ("…" if right < len(text) else "")


def _first_number_after(text: str, start: int) -> tuple[float, int, int, int] | None:
    """The first number after a position, on the same line."""
    end = text.find("\n", start)
    found = numbers_in(text[start:end if end >= 0 else len(text)])
    return (found[0][0], found[0][1], start + found[0][2], start + found[0][3]) if found else None


def compare(kind: str, expected: Any, shown: str) -> tuple[bool, str, str]:
    """(pass?, why, the part of the answer that shows it) for the answer the person sees: the code-written part of the
    chat's last message (code_written). Numbers recorded inside the run do not count, nor anything in a message that gives
    no answer (no_answer)."""
    stopped = no_answer(shown)
    if stopped is not None:
        return False, f"no answer was shown: {stopped}", _around(plain(shown), 0, 0, 200)
    text = code_written(shown)
    start_of_answer = _around(text, 0, 0, 200)
    if kind in INTEGER_KINDS:
        hits = [(s, e) for v, d, s, e in numbers_in(text) if v == expected]
        if hits:
            return True, f"{expected} appears in the answer", _around(text, *hits[0])
        return False, f"{expected} does not appear in the answer", start_of_answer
    if kind in NAME_KINDS:                    # the name, with its number of studies as the first number after it
        for name, count in expected.items():
            for at in re.finditer(re.escape(" ".join(name.split())), text, re.IGNORECASE):
                after = _first_number_after(text, at.end())
                if after is not None and after[0] == count:
                    return True, f"the answer gives {name} with {count} studies", _around(text, at.start(), after[3])
        return False, "the answer gives none of these with its count: " + "; ".join(f"{n} ({c})" for n, c in expected.items()), \
            start_of_answer
    if kind == "seasons_present":
        missing = [s for s in expected if not _whole_word(s).search(_without_dates(text))]
        return (not missing, "every season appears" if not missing
                else f"{len(missing)} seasons do not appear: " + ", ".join(missing[:10]), start_of_answer)
    if kind == "studies_per_type":
        lines = [" ".join(line.split()) for line in text.splitlines()]

        def shown_with_count(name: str, count: int) -> bool:          # its own count: the first number after the name
            return any((first := numbers_in(line[at.end():])[:1]) and first[0][0] == count
                       for line in lines for at in re.finditer(re.escape(name), line, re.IGNORECASE))
        missing = [name for name, count in expected.items() if not shown_with_count(name, count)]
        return (not missing, "every study type appears with its count" if not missing
                else f"{len(missing)} of {len(expected)} study types do not appear with their count: " + ", ".join(missing[:8]),
                start_of_answer)
    if kind == "pooled_mean":                 # the number right after "Mean": the first "Mean" that a number follows
        for label in _MEAN_LABEL.finditer(text):
            after = _first_number_after(text, label.end())
            if after is None or after[2] != label.end():
                continue
            exact = abs(after[0] - expected) <= 1e-12
            rounded = after[1] >= 1 and abs(after[0] - expected) <= 0.5 * 10 ** -after[1] + 1e-12
            where = _around(text, label.start(), after[3])
            if exact or rounded:              # a whole number counts only as the exact value: 2 is not 2.38 rounded
                return True, f"the mean given, {after[0]:g}, is the expected value" + (
                    "" if exact else f" rounded to {after[1]} decimals"), where
            return False, f"the mean given, {after[0]:g}, does not round from {expected:.4f}", where
        return False, "the answer gives no number right after \"Mean\"", start_of_answer
    raise ValueError(f"unknown expectation kind {kind!r}")


# --------------------------------------------------------------------------
# Feeding one question through the app's own pipeline, with the scripted approver
# --------------------------------------------------------------------------

# US dollars per million tokens (Anthropic's prices as given in the claude-api skill and the coach's research note,
# checked 2026-10-17; OpenAI's gpt-6-luna from the research note). Cache writes: 1.25 x input, the 5-minute cache the app marks
# on the Anthropic path (llm.CACHE_MARK, 2026-10-24); Sonnet 5.5's cache reads are $0.20 (shared/models.md; the table had
# $0.10). Haiku 5.5 costs 5x for a request whose prompt is over 100k tokens; the app's requests stay far below. A reply
# from a model not listed here has no cost.
PRICES = {
    "gpt-6-luna": {"input": 0.10, "cache_write": 0.125, "cache_read": 0.01, "output": 0.50},
    "claude-opus-5-5": {"input": 4.00, "cache_write": 5.00, "cache_read": 0.20, "output": 20.00},
    "claude-sonnet-5-5": {"input": 2.00, "cache_write": 2.50, "cache_read": 0.20, "output": 10.00},
    "claude-haiku-5-5": {"input": 0.10, "cache_write": 0.125, "cache_read": 0.01, "output": 0.50},
}
TOKEN_KINDS = ("input", "cache_write", "cache_read", "output")


def price_of(model: str | None) -> dict[str, float] | None:
    """The price of the model a reply names; a dated name ("gpt-6-luna-2026-09-30") has its family's price."""
    names = [name for name in PRICES if model and (model == name or model.startswith(name + "-"))]
    return PRICES[max(names, key=len)] if names else None


def reply_tokens(model: str | None, usage: dict[str, Any] | None, detail: dict[str, Any] | None) -> dict[str, Any] | None:
    """One reply's counts by kind: Claude's four counts when given, else the prompt as input and the completion as output."""
    if detail:
        return {"model": model, "input": detail.get("input_tokens", 0), "cache_write": detail.get("cache_creation_input_tokens", 0),
                "cache_read": detail.get("cache_read_input_tokens", 0), "output": detail.get("output_tokens", 0)}
    if usage:
        return {"model": model, "input": usage.get("prompt_tokens", 0), "cache_write": 0, "cache_read": 0,
                "output": usage.get("completion_tokens", 0)}
    return None


def tally(replies: list[dict[str, Any]], calls: int) -> dict[str, Any]:
    """The tokens and cost of one question: `replies` are the counted replies, `calls` every model call it made."""
    calls = max(calls, len(replies))              # a run with no result still logged its replies: never fewer calls than replies
    totals = {kind: sum(int(reply[kind]) for reply in replies) for kind in TOKEN_KINDS}
    unpriced = sorted({str(reply["model"]) for reply in replies if price_of(reply["model"]) is None})
    cost = None if unpriced else round(sum(reply[kind] * price_of(reply["model"])[kind] for reply in replies
                                           for kind in TOKEN_KINDS) / 1e6, 6)
    return {**totals, "calls_with_counts": len(replies), "calls_without_counts": max(0, calls - len(replies)),
            "cost_usd": cost, "unpriced_models": unpriced}


def run_replies(run_dirs: list[Path]) -> list[dict[str, Any]]:
    """Every counted model reply the runs logged (llm.run_agent_loop writes one "assistant" line per reply)."""
    replies = []
    for folder in run_dirs:
        for log in sorted(Path(folder).glob("llm_*.jsonl")):
            for line in log.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and event.get("event") == "assistant":
                    counted = reply_tokens(event.get("model"), event.get("usage"), event.get("usage_detail"))
                    if counted is not None:
                        replies.append(counted)
    return replies


class _Recorded:
    """A front-door model client that notes each reply's token counts (the front door logs nothing). It opens, closes and
    takes a role (llm.for_role) as the client it wraps does, and changes nothing it sends or returns."""

    def __init__(self, inner: Any, replies: list[dict[str, Any]]) -> None:
        self._inner, self._replies, self._opened = inner, replies, None

    async def __aenter__(self) -> "_Recorded":
        if hasattr(self._inner, "__aenter__"):
            self._opened = await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if hasattr(self._inner, "__aexit__"):
            await self._inner.__aexit__(*exc)

    def for_role(self, role: str) -> "_Recorded":
        return _Recorded(llm.for_role(self._opened or self._inner, role), self._replies)

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, max_tokens: int) -> Any:
        reply = await (self._opened or self._inner).complete(messages, tools, max_tokens=max_tokens)
        usage = getattr(reply, "usage", None)
        counted = reply_tokens(getattr(reply, "model", None), usage.model_dump() if usage is not None else None,
                               getattr(reply, "usage_detail", None))
        if counted is not None:
            self._replies.append(counted)
        return reply


class CountingFactory:
    """Counts the model clients the chat asks for: one per front-door (or waiting-message) model call, and keeps the
    token counts of their replies."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self.factory, self.calls, self.replies = factory, 0, []

    def __call__(self) -> Any:
        self.calls += 1
        return _Recorded(self.factory(), self.replies)


class RefusingTransport:
    """--cache-only (the default): any request for something not in the cache copy is refused here and counted; nothing
    is sent. The app records it as a failed request, as it would a network error."""

    refused = 0

    def __call__(self, request: Any) -> Any:
        RefusingTransport.refused += 1
        raise TransportError("blocked", "the eval harness answers from its cache copy only; this request is not in it")


class TrackedJob(app.Job):
    """app.Job, recording each run it starts: the chat drops a run without closing it when its own step wait runs out,
    and the harness must still end that run (and keep requests refused while it goes on)."""

    started: list[Any] = []

    def start(self) -> "TrackedJob":
        TrackedJob.started.append(self)
        return super().start()


@dataclass
class Outcome:
    shown: str = ""                                   # the chat's last message for this question
    status: str = ""                                  # how the question ended
    typed: list[str] = field(default_factory=list)    # what the scripted approver typed
    results: list[RunResult] = field(default_factory=list)
    threads: list[Any] = field(default_factory=list)   # the runs' threads (one still alive keeps requests refused)
    run_dirs: list[Path] = field(default_factory=list)  # the runs' folders, where their model logs are
    answered: list[str] = field(default_factory=list)   # the planner's question the benchmark answered, with what (2026-10-22)


def _turn(message: str, history: list, state: dict, chat: dict[str, Any]) -> tuple[list, dict]:
    out = list(app.respond(message, history, state, **chat))
    return out[-1][0], out[-1][1]


def _last(history: list) -> str:
    replies = [m["content"] for m in history if m.get("role") == "assistant"]
    return replies[-1] if replies else ""


def _stopped_at(waiting: str) -> str:
    if waiting == CATALOG_PROMPT:
        return "stopped: the catalog setup needs an approval (the harness never approves it)"
    if waiting in (CLARIFY_PROMPT, ANALYST_PROMPT, RETRIEVER_PROMPT, SUPERVISOR_PROMPT):
        return "stopped: the app asked a question (the harness never answers one)"
    return "stopped at a step the harness does not answer"


def _without_a_run(state: dict[str, Any]) -> str:
    if state.get("pending_question"):
        return "no run: the app asked whether to look it up (the harness never says yes)"
    last = getattr(state.get("memory"), "last", None)
    if getattr(last, "kind", None) == "catalog_answer":
        return "no run: answered by code from the saved catalog"
    return "no run: the app answered without starting one"


def _result_of(job: Any, wait: float) -> RunResult | None:
    """The run's record: its last event, or its answer.json when the chat already read that event."""
    if job.thread.is_alive():
        kind, payload = job.next_event(timeout=wait)
        if kind == "done" and isinstance(payload, RunResult):
            return payload
    job.thread.join(timeout=wait)
    path = Path(job.controller.config.out_dir) / job.controller.run_id / "answer.json"
    try:
        return RunResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def ask(question: str, chat: dict[str, Any], *, wait: float = 120.0,
        answer_clarification: Callable[[str], str | None] | None = None) -> Outcome:
    """One question as a new chat. The scripted approver types "approve" at the data request's approval screen and
    "continue" at the data review; anything else ends the question (a waiting run is ended, never accepted). With
    answer_clarification (the benchmark's, coach 2026-10-22 15:53 UTC), the planner's question (CLARIFY_PROMPT) is answered
    ONCE per question with what it returns for the chat's last message, and recorded in Outcome.answered; it returns None for
    any question it does not answer. The analyst's, retriever's and supervisor's questions are never answered."""
    outcome = Outcome()
    first = len(TrackedJob.started)
    history, state = _turn(question, [], {}, chat)
    jobs: list[Any] = []
    for _ in range(MAX_REPLIES + 1):
        job = state.get("job")
        if job is not None and all(job is not seen for seen in jobs):
            jobs.append(job)
        if job is None:
            outcome.status = ""                                          # decided below, once every run is known
            break
        waiting = job.waiting_on or ""
        reply = (answer_clarification(_last(history)) if waiting == CLARIFY_PROMPT and answer_clarification is not None
                 and not outcome.answered else None)
        if waiting.startswith("approve / edit / cancel"):
            word = "approve"
        elif waiting == DATA_REVIEW_PROMPT:
            word = "continue"
        elif reply:
            word = reply
            outcome.answered.append(f"clarification answered by harness: {reply}")
        else:
            outcome.status = ("final review reached; the answer was left pending" if waiting.startswith("accept / reject")
                              else _stopped_at(waiting))
            job.close()                                  # the app's own way to end a waiting run: never accept or reject
            break
        outcome.typed.append(word)
        history, state = _turn(word, history, state, chat)
    else:
        outcome.status = f"stopped after {MAX_REPLIES} replies"
        if state.get("job") is not None:
            state["job"].close()
    outcome.shown = _last(history)
    jobs += [job for job in TrackedJob.started[first:] if all(job is not seen for seen in jobs)]
    if not outcome.status:
        outcome.status = "run ended" if jobs else _without_a_run(state)
    for job in jobs:                         # a run the chat stopped waiting for (its step limit) is ended too, never left going
        if job.thread.is_alive():
            job.close()
    outcome.results = [r for r in (_result_of(job, wait) for job in jobs) if r is not None]
    outcome.threads = [job.thread for job in jobs]
    outcome.run_dirs = [Path(job.controller.run_dir) for job in jobs]
    return outcome


# --------------------------------------------------------------------------
# The whole evaluation: copy, expected answers, questions, results files
# --------------------------------------------------------------------------

def load_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data["questions"] if isinstance(data, dict) else data
    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("question ids must be unique")
    for q in questions:
        if not str(q.get("question", "")).strip() or q.get("expect", {}).get("kind") not in KINDS:
            raise ValueError(f"question {q.get('id')!r}: a question and an expect.kind from {', '.join(KINDS)} are required")
    return questions


def eval_config(base_config: Callable[[str], RunConfig], cache_dir: Path, out_dir: Path) -> Callable[[str], RunConfig]:
    """The app's own settings for a question, on the cache copy: runs are recorded under the results folder; no catalog
    is ever prepared under the operator's standing approval (the copy must already hold it); and the copy is used as it
    is, however old (APP_CATALOG_MAX_AGE_HOURS is not applied): the expected answers come from the same copy."""
    def make(question: str) -> RunConfig:
        return dataclasses.replace(base_config(question), cache_dir=cache_dir, out_dir=out_dir, catalog_standing_approval=False,
                                   catalog_max_age_hours=None)
    return make


def run_questions(questions: list[dict[str, Any]], *, cache_copy: Path, out_dir: Path,
                  base_config: Callable[[str], RunConfig], make_models: Callable[[], Any], triage_model: Callable[[], Any] | None,
                  preflight: Callable[[str], list[str]] | None, allow_live: bool, wait: float = 120.0) -> dict[str, Any]:
    catalog = read_catalog(cache_copy)
    make_config = eval_config(base_config, cache_copy, out_dir)
    saved_transport, saved_job = brapi_client.RequestsTransport, app.Job
    app.Job = TrackedJob                                            # every run the chat starts is known to the harness
    if not allow_live:
        brapi_client.RequestsTransport = RefusingTransport          # nothing reaches the database from this process
    rows, threads = [], []
    try:
        for q in questions:
            row: dict[str, Any] = {"id": q["id"], "question": q["question"], "kind": q["expect"]["kind"]}
            try:
                if catalog is None:
                    raise ValueError("the study and trait catalogs are not completely in the cache copy")
                expected, expected_text = expected_answer(q["expect"], catalog)
            except (ValueError, KeyError) as exc:
                rows.append({**row, "expected": None, "expected_text": "", "passed": None,
                             "status": f"skipped: the expected answer cannot be computed ({exc})"})
                continue
            counter = CountingFactory(triage_model) if triage_model is not None else None
            RefusingTransport.refused = 0
            started = time.monotonic()
            try:
                outcome = ask(q["question"], dict(make_config=make_config, make_models=make_models, preflight=preflight,
                                                  triage_model=counter), wait=wait)
            except Exception as exc:  # noqa: BLE001 - one broken question must not stop the others
                outcome = Outcome(status=f"error: {type(exc).__name__}: {exc}")
            seconds = time.monotonic() - started
            threads += outcome.threads
            claims = [c.model_dump(mode="json") for r in outcome.results if r.analysis for c in r.analysis.claims]
            if outcome.status.startswith(SCORED_ENDINGS):                                  # the claims are kept, not scored
                passed, why, where = compare(q["expect"]["kind"], expected, outcome.shown)
            else:                                     # a question, a lookup offer or a model's reply: no answer to score
                passed, why, where = False, "no answer was shown to score", _around(plain(outcome.shown), 0, 0, 200)
            run_calls = sum(r.counts.model_calls for r in outcome.results)
            tokens = tally((counter.replies if counter else []) + run_replies(outcome.run_dirs),
                           (counter.calls if counter else 0) + run_calls)
            rows.append({**row, "expected": expected, "expected_text": expected_text, "passed": passed, "why": why,
                         "status": outcome.status, "got": where, "answer": outcome.shown[:GOT_CHARS], "typed": outcome.typed,
                         "claims": [{k: c.get(k) for k in ("kind", "value", "unit", "filters", "transformation")} for c in claims],
                         "model_calls": (counter.calls if counter else 0) + run_calls,
                         "front_door_calls": counter.calls if counter else 0, "run_model_calls": run_calls,
                         "live_attempts": sum(r.counts.live_attempts for r in outcome.results),
                         "refused_requests": 0 if allow_live else RefusingTransport.refused,
                         "run_ids": [r.run_id for r in outcome.results],
                         "acceptance": [r.human_acceptance for r in outcome.results],
                         "models_reported": sorted({r.model_reported for r in outcome.results if r.model_reported}),
                         "tokens": tokens, "seconds": round(seconds, 2)})
    finally:
        app.Job = saved_job
        still_running = [th for th in threads if th.is_alive()]
        if allow_live or not still_running:
            brapi_client.RequestsTransport = saved_transport
        # else: a run did not end within the wait; requests stay refused in this process (main exits right after)
    answered = [r for r in rows if r["passed"] is not None]
    counted = [r["tokens"] for r in rows if "tokens" in r]
    total = {kind: sum(t[kind] for t in counted) for kind in (*TOKEN_KINDS, "calls_with_counts", "calls_without_counts")}
    costs = [t["cost_usd"] for t in counted]
    total["cost_usd"] = None if None in costs else round(sum(costs), 6)
    total["unpriced_models"] = sorted({m for t in counted for m in t["unpriced_models"]})
    return {"tokens": total, "cache_copy": str(cache_copy), "base_url": catalog.base_url if catalog else None,
            "runs_still_going": len([th for th in threads if th.is_alive()]),
            "catalog_fetched": catalog.fetched_at if catalog else [], "requests": "allowed after the scripted approval"
            if allow_live else "none: cache copy only", "passed": sum(1 for r in answered if r["passed"]),
            "failed": sum(1 for r in answered if not r["passed"]), "skipped": len(rows) - len(answered), "questions": rows}


def _cell(text: Any, limit: int = 160) -> str:
    text = " ".join(str(text if text is not None else "").split()).replace("|", "\\|")
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def tokens_cell(tokens: dict[str, Any] | None) -> str:
    """Input (everything read) / output, as whole numbers."""
    if not tokens:
        return ""
    read = tokens["input"] + tokens["cache_write"] + tokens["cache_read"]
    return f"{read:,} / {tokens['output']:,}"


def cost_cell(tokens: dict[str, Any] | None) -> str:
    """Dollars; "?" when a model has no price; "at least" when a call had no counts."""
    if not tokens:
        return ""
    if tokens["cost_usd"] is None:
        return "?"
    return ("at least " if tokens["calls_without_counts"] else "") + f"{tokens['cost_usd']:.6f}"


def results_markdown(report: dict[str, Any], *, started: str, models: str) -> str:
    lines = [f"# Answer-quality run, {started}", "",
             f"- Models: {models}", f"- Cache copy: `{report['cache_copy']}` (server {report['base_url']}; catalogs fetched "
             f"{', '.join(report['catalog_fetched']) or 'unknown'})", f"- Database requests: {report['requests']}",
             f"- Result: {report['passed']} passed, {report['failed']} failed, {report['skipped']} skipped",
             "- A pass means the expected value appears in the answer the person sees (the chat's last message, without "
             "what the app marks as model notes); numbers recorded inside the run do not count. Read \"got\" before "
             "trusting it. The scripted approver typed only \"approve\" (data request) and \"continue\" (data review); no "
             "answer was accepted.",
             *_totals_lines(report.get("tokens")), "",
             "| # | Question | Expected | Got | Pass | Model calls | Tokens in / out | Cost ($) | Seconds | How it ended |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in report["questions"]:
        verdict = "skipped" if r["passed"] is None else ("PASS" if r["passed"] else "FAIL")
        lines.append(f"| {r['id']} | {_cell(r['question'], 120)} | {_cell(r['expected_text'], 80)} | "
                     f"{_cell(r.get('got', ''), 200)} | {verdict} | {r.get('model_calls', '')} | {tokens_cell(r.get('tokens'))} | "
                     f"{cost_cell(r.get('tokens'))} | {r.get('seconds', '')} | "
                     f"{_cell(r['status'] + ('; ' + r['why'] if r.get('why') else ''), 160)} |")
    return "\n".join(lines) + "\n"


def _totals_lines(total: dict[str, Any] | None) -> list[str]:
    if not total:
        return []
    lines = [f"- Tokens: {total['input'] + total['cache_write'] + total['cache_read']:,} in (cache writes {total['cache_write']:,}, "
             f"cache reads {total['cache_read']:,}), {total['output']:,} out (thinking included); cost: {cost_cell(total)} US dollars "
             "at the prices in eval/harness.py PRICES (the app does not keep OpenAI's cached-input count, so its cost is an upper bound)."]
    if total["calls_without_counts"]:
        lines.append(f"- Model calls without token counts (a provider error or a timeout): {total['calls_without_counts']}.")
    if total["unpriced_models"]:
        lines.append(f"- No price for: {', '.join(total['unpriced_models'])}.")
    return lines


def main(argv: list[str] | None = None, *, base_config: Callable[[str], RunConfig] | None = None,
         make_models: Callable[[], Any] | None = None, triage_model: Callable[[], Any] | None = None,
         preflight: Callable[[str], list[str]] | None = None, models_label: str | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the breeder questions through the app and compare its answers with "
                                     "answers computed by code from a cache copy.")
    parser.add_argument("--cache", required=True, type=Path, help="the cache folder to copy (it is never changed)")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--results", type=Path, default=RESULTS, help="a folder per run is made inside it")
    parser.add_argument("--only", default="", help="comma-separated question ids")
    parser.add_argument("--allow-live", action="store_true", help="let the app fetch what the copy lacks, after the scripted approval")
    parser.add_argument("--confirm", action="store_true", help="really run (uses the configured model; costs model calls)")
    args = parser.parse_args(argv)
    questions = load_questions(args.questions)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        questions = [q for q in questions if q["id"] in wanted]
    if not args.cache.is_dir():
        print(f"no cache folder at {args.cache}")
        return 2
    scripted = make_models is not None
    if not args.confirm:
        print(f"Would run {len(questions)} questions on a copy of {args.cache} with "
              f"{'the scripted models given' if scripted else 'the configured model (real model calls)'}; database requests: "
              f"{'allowed after the scripted approval' if args.allow_live else 'none (cache copy only)'}. Results under "
              f"{args.results}. Nothing was run; add --confirm to run.")
        return 2
    started = datetime.now(timezone.utc)
    folder = args.results / started.strftime("%Y%m%dT%H%M%SZ")
    folder.mkdir(parents=True, exist_ok=False)
    cache_copy = folder / "cache"
    shutil.copytree(args.cache, cache_copy)
    report = run_questions(questions, cache_copy=cache_copy, out_dir=folder / "out", base_config=base_config or app.live_config,
                           make_models=make_models or app.default_models,
                           triage_model=triage_model if scripted else app.default_triage_model,
                           preflight=preflight, allow_live=args.allow_live)
    models = models_label or ("scripted" if scripted else "configured model: " +
                              (", ".join(sorted({m for r in report["questions"] for m in r.get("models_reported", [])})) or "unknown"))
    report = {"started_utc": started.isoformat(), "models": models, **report}
    (folder / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (folder / "results.md").write_text(results_markdown(report, started=started.strftime("%Y-%m-%d %H:%M UTC"), models=models),
                                       encoding="utf-8")
    print(f"{report['passed']} passed, {report['failed']} failed, {report['skipped']} skipped -> {folder / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
