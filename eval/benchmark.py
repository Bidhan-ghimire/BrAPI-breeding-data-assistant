"""The sweetpotato benchmark's single-statistic sub-questions through the app (coach Task 1, 2026-10-21).

eval/benchmark/spb_subquestions.json holds the 33 benchmark items computed from SweetPotatoBase plot data, split into
sub-questions of one statistic each, with the gold values (sweetpotato_breeding_benchmark.json, built 16-17 Oct 2026
independently of this app). Each sub-question is asked as a new chat, exactly as eval/harness.py asks its questions (the
same scripted approver: "approve" at the data request, "continue" at the data review, nothing else), on a copy of a
cache folder; with --allow-live the app may fetch what the copy lacks.

Grading is code-based, like the harness: only the code-written part of the chat's last message counts (harness.code_written:
model notes are left out). Buckets:
  pass       every expected value (and name) is found within its tolerance (or every value of one listed alternative)
  wrong      the answer shows backed numbers, but not the expected ones
  no_number  an honest answer without the number: "cannot compute", a question back, a lookup offer, a catalog summary
  blocked    a gate, a limit or an error stopped the question (blocked / limit / could not complete / the catalog
             approval the harness never gives / a harness error)
  invented   a number in the code-written part that no claim, no run text, no table of the run and not the question
             backs: the red flag, reported first whatever else the answer holds (except on the app's question screens:
             they answer nothing, so they are not checked)
"""
from __future__ import annotations

import argparse
import csv
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
_SCIENTIFIC = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?[eE][+-]?\d+(?![\w.])")
_ASKED_FIRST_LINES = ("More detail is needed.",)
_ASKED_STATUS = ("stopped: the app asked a question", "no run: the app asked whether to look it up")
_BLOCKED_STATUS = ("error", "stopped: the catalog setup needs an approval", "stopped at a step", "stopped after")
# A reply in the model's own words, with no run (the front door's talk reply): never an answer to score; the app's own
# backstop allows only numbers from the person's words in it, so any other number is the red flag (review 10).
MODEL_REPLY_STATUS = "no run: the app answered without starting one"
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


def _numbers(text: str) -> list[tuple[float, int, int, int]]:
    """Every number in the text (harness.numbers_in), with numbers in scientific notation read whole and ontology terms
    left out (blanked at the same length, so positions stay); full ISO timestamps are no numbers."""
    blank = lambda m: " " * len(m.group())                                              # noqa: E731
    text = _TIMESTAMP.sub(blank, text)
    found = []
    for match in _SCIENTIFIC.finditer(text):
        mantissa = match.group(0).lower().split("e")[0]
        found.append((float(match.group(0)), len(mantissa.split(".")[1]) if "." in mantissa else 0, match.start(), match.end()))
    found += harness.numbers_in(_ONTOLOGY.sub(blank, _SCIENTIFIC.sub(blank, text)))
    return sorted(found, key=lambda n: n[2])


def _numbers_for_invention(text: str) -> list[tuple[float, int, int, int]]:
    return _numbers(_STRUCTURAL.sub(lambda m: " " * len(m.group()), text))


def _without_quotes(text: str) -> str:
    """The text without the lines the app quotes ("> ..."): a model's question shown on a clarification screen, or the
    person's own earlier words. They are attributed, not answers: left out like the model notes (review 10)."""
    return "\n".join(line for line in text.split("\n") if not line.lstrip().startswith(">"))


def claim_numbers(claims: Iterable[dict[str, Any]]) -> list[float]:
    """Every number a run's claims carry: values, denominators, units, exclusion counts, and numbers in their filters,
    grouping, method and denominator texts."""
    found: list[float] = []
    for c in claims:
        denominator = c.get("denominator") or {}
        for v in (c.get("value"), denominator.get("n"), c.get("n_independent_units"),
                  *[e.get("count") for e in c.get("exclusions") or []]):
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                found.append(float(v))
        for text in (*(c.get("filters") or {}).values(), *(c.get("grouping") or []), c.get("transformation") or "",
                     denominator.get("name") or ""):
            found += [n for n, *_ in _numbers(str(text))]
    return found


def _backed(value: float, decimals: int, sources: list[float]) -> bool:
    """A shown number is backed when some source rounds to it at the shown precision ("2.38" backs 2.37857)."""
    half = 0.5 * 10 ** -decimals
    return any(abs(s - value) <= half + 1e-9 * max(1.0, abs(s)) for s in sources)


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


def expected_label(e: dict[str, Any]) -> str:
    return f"{e['value']:g}" if isinstance(e["value"], float) else str(e["value"])


def grade(sub: dict[str, Any], *, shown: str, status: str, claims: list[dict[str, Any]],
          backed_extra: Iterable[float] = (), run: bool = True, code_catalog: bool = False) -> dict[str, Any]:
    """One sub-question's bucket, from the chat's last message (shown), how the question ended (status, as harness.ask
    writes it), the run's claims and other numbers the run's own code or data wrote (backed_extra).
    run=False: no run answered. code_catalog=True: a list run's answer, written by code from the saved catalog.
    Neither is checked for invention against claims (there are none to check against); a reply in the model's own words
    (MODEL_REPLY_STATUS) is never scored and may hold only numbers from the question."""
    text = _without_quotes(harness.code_written(shown))
    asked = [n for n, *_ in _numbers(sub.get("question", ""))]
    result: dict[str, Any] = {"bucket": "", "why": "", "invented": [], "matched": []}

    def flagged(numbers: list[tuple[float, int, int, int]], sources: list[float]) -> list[dict[str, Any]]:
        return [{"value": int(v) if float(v).is_integer() else v, "context": harness._around(text, start, end, 60)}
                for v, d, start, end in numbers if not _backed(v, d, sources)]

    if status.startswith(MODEL_REPLY_STATUS):
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
        invented = flagged(_numbers_for_invention(text), claim_numbers(claims) + asked + [float(x) for x in backed_extra])
        if invented:
            return {**result, "bucket": "invented", "invented": invented,
                    "why": f"{len(invented)} number(s) in the code-written answer that nothing backs"}
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
    backed_shown = [v for v, d, *_ in numbers if _backed(v, d, claim_numbers(claims))] if run and not code_catalog else numbers
    if backed_shown:
        return {**result, "bucket": "wrong", "why": "numbers shown, but not every expected value"}
    return {**result, "bucket": "no_number", "why": "no number for the statistic was shown"}


def run_numbers(run_dirs: Iterable[Path]) -> list[float]:
    """Numbers the runs' own code and data wrote: every cell of their saved tables (artifacts/*.csv), the numbers in their
    code-written report fields (caveats, warnings, notes, exclusions) and the plain-digit study IDs of the study ledger (a
    refused fetch is reported as "Study 4777: ..."; variance run of 2026-10-21). Model text is not among them: a ledger
    entry's reason quotes the model's arguments, and its study_id is the model's own argument, so only an all-digit ID counts
    (review 2026-10-21: "mean=12.5" as a study ID must stay unbacked)."""
    found: list[float] = []
    for run_dir in run_dirs:
        for table in sorted((Path(run_dir) / "artifacts").glob("*.csv")):
            with table.open(encoding="utf-8", newline="") as handle:
                for row in csv.reader(handle):
                    for cell in row:
                        found += [n for n, *_ in _numbers(cell)]
        answer = Path(run_dir) / "answer.json"
        if answer.is_file():
            doc = json.loads(answer.read_text(encoding="utf-8"))
            texts = [*((doc.get("analysis") or {}).get("caveats") or []), *((doc.get("retrieval") or {}).get("warnings") or []),
                     *(doc.get("notes") or [])]
            found += [float(entry["study_id"]) for entry in (doc.get("retrieval") or {}).get("study_ledger") or []
                      if re.fullmatch(r"[0-9]+", str(entry.get("study_id") or ""))]
            found += [n for t in texts if not str(t).lower().startswith(("model:", "model note:"))
                      for n, *_ in _numbers(str(t))]
    return found


def run_subquestions(subs: list[dict[str, Any]], *, cache_copy: Path, out_dir: Path,
                     base_config: Callable[[str], Any], make_models: Callable[[], Any], triage_model: Callable[[], Any] | None,
                     preflight: Callable[[str], list[str]] | None, allow_live: bool, wait: float = 120.0) -> dict[str, Any]:
    make_config = harness.eval_config(base_config, cache_copy, out_dir)
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
                                                          triage_model=counter), wait=wait)
            except Exception as exc:  # noqa: BLE001 - one broken question must not stop the others
                outcome = harness.Outcome(status=f"error: {type(exc).__name__}: {exc}")
            seconds = time.monotonic() - started
            threads += outcome.threads
            claims = [c.model_dump(mode="json") for r in outcome.results if r.analysis for c in r.analysis.claims]
            ran = bool(outcome.results)
            catalog_run = any(step.action in CATALOG_ACTIONS for r in outcome.results if r.plan for step in r.plan.steps)
            graded = grade(s, shown=outcome.shown, status=outcome.status, claims=claims,
                           backed_extra=run_numbers(outcome.run_dirs) if ran else (), run=ran, code_catalog=catalog_run)
            run_calls = sum(r.counts.model_calls for r in outcome.results)
            tokens = harness.tally((counter.replies if counter else []) + harness.run_replies(outcome.run_dirs),
                                   (counter.calls if counter else 0) + run_calls)
            rows.append({"sub_id": s["sub_id"], "parent": s["parent"], "tier": s["tier"], "question": s["question"],
                         "expected": s["expected"], "expected_text": s.get("expected_text") or [], **graded,
                         "status": outcome.status, "answer": outcome.shown[:harness.GOT_CHARS], "answer_full": outcome.shown,
                         "typed": outcome.typed,
                         "claims": [{k: c.get(k) for k in ("kind", "value", "unit", "filters", "transformation")} for c in claims],
                         "model_calls": (counter.calls if counter else 0) + run_calls,
                         "live_attempts": sum(r.counts.live_attempts for r in outcome.results),
                         "refused_requests": 0 if allow_live else harness.RefusingTransport.refused,
                         "run_ids": [r.run_id for r in outcome.results], "tokens": tokens, "seconds": round(seconds, 2),
                         "caveat": s.get("caveat")})
            print(f"{s['sub_id']:8} {s['tier']} {graded['bucket']:9} {seconds:6.1f}s  {graded['why'][:80]}", flush=True)
    finally:
        harness.app.Job = saved_job
        if allow_live or not [th for th in threads if th.is_alive()]:
            harness.brapi_client.RequestsTransport = saved_transport
    costs = [r["tokens"]["cost_usd"] for r in rows]
    by_tier = {t: {b: sum(1 for r in rows if r["tier"] == t and r["bucket"] == b) for b in BUCKETS} for t in TIERS}
    return {"cost_usd": None if None in costs else round(sum(costs), 6), "by_tier": by_tier,
            "invented": [r["sub_id"] for r in rows if r["bucket"] == "invented"], "subquestions": rows,
            "requests": "allowed after the scripted approval" if allow_live else "none: cache copy only"}


def results_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Benchmark run, {report['started_utc']}", "", f"- Models: {report['models']}",
             f"- Cache copy: `{report['cache_copy']}`; database requests: {report['requests']}",
             f"- Cost: {report['cost_usd']} US dollars at the prices in eval/harness.py PRICES (an upper bound for OpenAI)",
             f"- INVENTED (red flag): {', '.join(report['invented']) or 'none'}", "",
             "| Tier | Sub-questions | pass | wrong | no number | blocked | invented |", "|---|---|---|---|---|---|---|"]
    for tier, counts in report["by_tier"].items():
        total = sum(counts.values())
        if total:
            lines.append(f"| {tier} | {total} | " + " | ".join(str(counts[b]) for b in BUCKETS) + " |")
    lines += ["", "A pass is any number within tolerance anywhere in the code-written answer: read Got before trusting it.", "",
              "| Sub | Tier | Bucket | Why | Got | Model calls | Cost ($) | Seconds | Status |", "|---|---|---|---|---|---|---|---|---|"]
    for r in report["subquestions"]:
        cost = r["tokens"]["cost_usd"]
        got = "; ".join(r.get("matched") or []) or "; ".join(f"{i['value']} in \"{i['context']}\"" for i in r.get("invented") or [])
        lines.append(f"| {r['sub_id']} | {r['tier']} | {r['bucket']} | {harness._cell(r['why'], 90)} | {harness._cell(got, 120)} | "
                     f"{r['model_calls']} | {'' if cost is None else f'{cost:.6f}'} | {r['seconds']} | {harness._cell(r['status'], 60)} |")
    skipped = report.get("not_checkable") or []
    if skipped:
        lines += ["", "Not checkable (not run): " + ", ".join(skipped)]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, *, base_config: Callable[[str], Any] | None = None,
         make_models: Callable[[], Any] | None = None, triage_model: Callable[[], Any] | None = None,
         preflight: Callable[[str], list[str]] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the benchmark's sub-questions through the app and grade them by code.")
    parser.add_argument("--cache", required=True, type=Path, help="the cache folder to copy (it is never changed)")
    parser.add_argument("--subquestions", type=Path, default=SUBQUESTIONS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--tiers", default="T1,T2,T3", help="comma-separated tiers, run in this order")
    parser.add_argument("--only", default="", help="comma-separated sub_ids")
    parser.add_argument("--allow-live", action="store_true", help="let the app fetch what the copy lacks, after the scripted approval")
    parser.add_argument("--confirm", action="store_true", help="really run (uses the configured model; costs model calls)")
    args = parser.parse_args(argv)
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
    report = run_subquestions(subs, cache_copy=cache_copy, out_dir=folder / "out", base_config=base_config or harness.app.live_config,
                              make_models=make_models or harness.app.default_models,
                              triage_model=triage_model if scripted else harness.app.default_triage_model,
                              preflight=preflight, allow_live=args.allow_live)
    models = "scripted" if scripted else "configured model: " + (", ".join(sorted({m for r in report["subquestions"]
                                                                                    for m in r["tokens"].get("models", [])})) or "see run logs")
    report = {"started_utc": started.isoformat(), "models": models, "cache_copy": str(cache_copy), "not_checkable": not_checkable, **report}
    (folder / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (folder / "results.md").write_text(results_markdown(report), encoding="utf-8")
    print(f"{sum(c['pass'] for c in report['by_tier'].values())} pass of {len(report['subquestions'])}; invented: "
          f"{', '.join(report['invented']) or 'none'} -> {folder / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
