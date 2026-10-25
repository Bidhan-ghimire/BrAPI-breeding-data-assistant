"""Which model-note lines the answer may show (decision (31)(b), rule (3); Bidhan, coach 2026-10-25 12:20 UTC).

A line of the model's notes (the "Additional model notes (unverified)" bullets, the "Unverified model note:" paragraphs) is shown
only when every number in it is backed the way the benchmark backs a note number (eval/benchmark.py grade, c5f19af): by the
claims' numbers, with the numbers in their filters and texts; by the question's numbers; by the run's own tables and
code-written fields; or by a number the checked reply itself shows; each read at the precision it is written with. A line with
a number nothing backs is dropped; lines without numbers stay. Identifiers are not quantities (S2, decision of 2026-10-25 14:35
UTC): "-00595", a plot name's ending, reads 595; the digits inside a name (2016 in MZP2016UMB-ST01) back a number of that value.

The reader and the sources are copies of eval/benchmark.py (_numbers, name_numbers, _backed, claim_numbers, claim_names,
run_numbers) and eval/harness.py
(plain, numbers_in): the app does not ship eval/, and tests/test_note_backing.py checks that the copies agree with them.
"""
from __future__ import annotations

import csv
import html
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

# eval/harness.py
_MARKDOWN_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~<])")
_NUMBER_IN_TEXT = re.compile(r"(?<![\w.])[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.])[+-]?\d+(?:\.\d+)?(?![\w])")
_DATE_TIME = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?|(?<![\w.:])\d{1,2}:\d{2}(?::\d{2})?(?![\w:])")
# eval/benchmark.py
_ONTOLOGY = re.compile(r"\b[A-Z][A-Z0-9_]*:\d+\b")
_SCIENTIFIC = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?[eE][+-]?\d+(?!\w|\.\w)")
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
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?")
_MODEL_ERRORS = tuple(f"{spelling} {code}" for spelling in ("agent error", "coordinator: agent error", "supervisor: supervisor error")
                      for code in ("model_refusal", "malformed_arguments", "invalid_json"))
_MODEL_QUOTED = ("model:", "model note:", "model declared ids", "model listed resolved entries", "model named claim ids",
                 "studies named by the model", "clarification ", "retriever clarification ", "analyst clarification ",
                 *_MODEL_ERRORS, "coordinator stopped the run:")
_DRAFT_LINES = ("coordinator: attempt", "coordinator: dropped clarification")
_QUOTED_PART = re.compile(r"(?<![\w\\])'(?:\\.|[^'\\])*'(?!\w)|(?<!\\)\"(?:\\.|[^\"\\])*\"|\{[^}]*\}|\[[^\]]*\]")


def plain(markdown: str) -> str:
    """The text as a reader sees it: markdown escapes and HTML entities undone, bold markers removed."""
    return _MARKDOWN_ESCAPE.sub(r"\1", html.unescape(markdown or "")).replace("**", "")


def _numbers_in(text: str) -> list[tuple[float, int, int, int]]:
    found = []
    dates = [match.span() for match in _DATE_TIME.finditer(text)]
    for match in _NUMBER_IN_TEXT.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in dates):
            continue
        token = match.group(0).replace(",", "")
        found.append((float(token), len(token.split(".")[1]) if "." in token else 0, match.start(), match.end()))
    return found


def numbers(text: str, *, written: bool = False) -> list[tuple[float, int, int, int]]:
    """Every number in the text: (value, decimals, start, end); numbers in scientific notation read whole (at the precision
    written, or at the four figures the app writes), ontology terms and full timestamps left out; a scientific token whose
    exponent has more than three digits or is positive past 300 is no number; a sign before a zero-padded digit run is no minus
    sign ("-00595" reads 595)."""
    blank = lambda m: " " * len(m.group())                                              # noqa: E731
    text = _TIMESTAMP.sub(blank, text)
    found = []
    for match in _SCIENTIFIC.finditer(text):
        mantissa, _, exponent = match.group(0).lower().partition("e")
        digits = exponent.lstrip("+-")
        if len(digits) > 3 or (not exponent.startswith("-") and int(digits) > 300):
            continue
        value = float(match.group(0))
        if not math.isfinite(value):
            continue
        places = len(mantissa.split(".")[1]) if "." in mantissa else 0
        decimals = (places if written else max(places, 3)) - int(exponent)
        found.append((value, decimals, match.start(), match.end()))
    found += _numbers_in(_ONTOLOGY.sub(blank, _SCIENTIFIC.sub(blank, text)))
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


def backed(value: float, decimals: int, sources: list[float]) -> bool:
    """A number is backed when some source rounds to it at its precision ("2.38" backs 2.37857)."""
    half = 0.5 * 10 ** -decimals
    return any(math.isfinite(s) and abs(s - value) <= half + 1e-9 * abs(s) for s in sources)


def claim_numbers(claims: Iterable[dict[str, Any]]) -> list[float]:
    """Every number the claims carry: values, denominators, units, exclusion counts, and the numbers in their filters,
    grouping, method, missing-reason, denominator and exclusion-reason texts."""
    claims = list(claims)
    found: list[float] = []
    for c in claims:
        denominator = c.get("denominator") or {}
        for v in (c.get("value"), denominator.get("n"), c.get("n_independent_units"),
                  *[e.get("count") for e in c.get("exclusions") or []]):
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                found.append(float(v))
    for c in claims:
        for text in _claim_texts(c):
            found += [n for n, *_ in numbers(str(text))]
    return found


def _claim_texts(c: dict[str, Any]) -> tuple[Any, ...]:
    """A claim's texts: filters, grouping, method, missing reason, denominator name and exclusion reasons."""
    denominator = c.get("denominator") or {}
    return (*(c.get("filters") or {}).values(), *(c.get("grouping") or []), c.get("transformation") or "", c.get("missing_reason") or "",
            denominator.get("name") or "", *(e.get("reason") or "" for e in c.get("exclusions") or []))


def claim_names(claims: Iterable[dict[str, Any]]) -> list[float]:
    """The digits inside the names in the claims' texts (S2, name_numbers)."""
    return [n for c in claims for text in _claim_texts(c) for n in name_numbers(str(text))]


def table_numbers(run_dir: Path | str | None) -> list[float]:
    """Every number in the cells of the run's saved tables (artifacts/*.csv), with the digits inside the names they hold (S2)."""
    if not run_dir:
        return []
    found: list[float] = []
    for table in sorted((Path(run_dir) / "artifacts").glob("*.csv")):
        with table.open(encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle):
                for cell in row:
                    found += [n for n, *_ in numbers(cell)] + name_numbers(cell)
    return found


def field_numbers(*, caveats: Iterable[Any] | None, warnings: Iterable[Any] | None, notes: Iterable[Any] | None,
                  ledger: Iterable[Any] | None) -> list[float]:
    """The numbers the run's code wrote in its report fields: the all-digit study IDs of its study ledger (entries as records or
    as answer.json's dicts) and the numbers in its caveats, warnings and notes; a line that quotes the model backs nothing, and
    a planner draft's quoted parts are blanked; with the digits inside the names they hold (S2)."""
    ids = [str((entry.get("study_id") if isinstance(entry, dict) else getattr(entry, "study_id", None)) or "") for entry in ledger or []]
    found = [float(study_id) for study_id in ids if re.fullmatch(r"[0-9]+", study_id)]
    for text in map(str, [*(caveats or []), *(warnings or []), *(notes or [])]):
        if text.lower().startswith(_MODEL_QUOTED):
            continue
        if text.lower().startswith(_DRAFT_LINES):
            text = _QUOTED_PART.sub(" ", text)
        found += [n for n, *_ in numbers(text)] + name_numbers(text)
    return found


def run_numbers(run_dir: Path | str | None) -> list[float]:
    """The numbers the run's own code and data wrote, as a finished run's folder records them (the grader's reading): its
    tables (table_numbers) and the report fields answer.json holds (field_numbers)."""
    if not run_dir:
        return []
    found = table_numbers(run_dir)
    answer = Path(run_dir) / "answer.json"
    if answer.is_file():
        doc = json.loads(answer.read_text(encoding="utf-8"))
        analysis, retrieval = doc.get("analysis") or {}, doc.get("retrieval") or {}
        found += field_numbers(caveats=analysis.get("caveats"), warnings=retrieval.get("warnings"), notes=doc.get("notes"),
                               ledger=retrieval.get("study_ledger"))
    return found


def sources(*, claims: Iterable[dict[str, Any]], question: str, extra: Iterable[float], checked_text: str) -> list[float]:
    """Rule (3)'s sources for a note number: the claims' numbers, the question's, the run's own (extra: run_numbers) and the
    numbers of the checked reply (checked_text: the answer without the model's notes; lines the app quotes, "> ...", are left
    out, as the person's own words back nothing). Each with the digits inside its names (S2)."""
    checked = "\n".join(line for line in plain(checked_text).split("\n") if not line.lstrip().startswith(">"))
    claims = list(claims)                                     # read twice below: a one-shot iterable keeps its names
    return (claim_numbers(claims) + claim_names(claims) + [n for n, *_ in numbers(question or "")] + name_numbers(question or "")
            + [float(x) for x in extra] + [n for n, *_ in numbers(checked)] + name_numbers(checked))


def unbacked(note: str, backing: list[float]) -> list[float | int | str]:
    """The numbers of the note line (read at the precision written) that nothing in backing backs, as a record keeps them
    (_recorded)."""
    text = plain(note)
    return [_recorded(v, text[start:end]) for v, d, start, end in numbers(text, written=True) if not backed(v, d, backing)]


def _recorded(value: float, written: str) -> float | int | str:
    """A number as a record keeps it: an integer when it is one (below 10^15), else the float; the number as written when no
    float holds it (a run of 310 digits reads as inf, "1e-400" as 0)."""
    if not math.isfinite(value) or (value == 0 and re.search(r"[1-9]", written.lower().partition("e")[0])):
        return written
    return int(value) if value.is_integer() and abs(value) < 1e15 else value


def keeps(note: str, backing: list[float]) -> bool:
    """True when every number of the note line (read at the precision written) is backed: unbacked finds none."""
    return not unbacked(note, backing)
