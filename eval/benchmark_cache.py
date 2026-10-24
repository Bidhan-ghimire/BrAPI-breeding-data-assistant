"""Build the benchmark cache ONCE (coach Task 1 step 3, 2026-10-21; live GETs to SweetPotatoBase pre-approved by
Bidhan's standing grant).

The source cache (with the study and trait catalogs) is copied to a new folder, never changed. Then, for every study the
sub-questions name, its own record (get_study), its plot roster (get_observation_units) and, for every variable a
sub-question names with that study, its observations (get_observations) are fetched through the app's own data tools (brapi_mcp_server.dispatch ->
BrapiClient -> the new cache), under one approval made by code for exactly those studies and variables, exactly as a
run's Retriever would request them. Benchmark runs then use a copy of this cache (eval/benchmark.py --cache).

It records each study's unit count, its request time, and whether the app's limits hold it: the app's pages per fetch
(app.live_config; a roster or a variable the server reports as larger comes back incomplete, as in a run) and
RunConfig.max_fetches fetch calls per question, counted as run.required_fetches counts them (one per study and variable; a roster per study only for
a question without a variable, a design question). It changes no limit.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))           # the harness and benchmark beside this file
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))    # the app
import harness  # noqa: E402
import brapi_mcp_server as srv  # noqa: E402
from brapi_client import load_settings  # noqa: E402
from contracts import FetchApproval  # noqa: E402
from run import RunConfig  # noqa: E402

MAX_FETCHES = RunConfig.__dataclass_fields__["max_fetches"].default     # RunConfig's default (app.live_config sets its own)
_FIELDS = ("status", "complete", "returned", "reported_total")


def variable_ids(catalog: Any, name: str, term: str | None) -> list[str]:
    """The trait catalog's variable IDs for a sub-question's variable: by its ontology term ("…|CO_331:0000296"), else by
    its whole name (with or without the "|term" suffix), ignoring case."""
    wanted = (name or "").strip().casefold()
    hits = []
    for row in catalog.variables:
        full = str(row.get("observationVariableName") or "")
        plain = full.split("|", 1)[0].strip().casefold()
        if (term and full.endswith("|" + term)) or (not term and wanted and wanted in (full.casefold(), plain)):
            hits.append(str(row.get("observationVariableDbId")))
    return hits


def build(subs: list[dict[str, Any]], *, source: Path, dest: Path, out_dir: Path, max_fetches: int | None = None) -> dict[str, Any]:
    """max_fetches: the app's run limit (app.live_config) unless given. The fetches use the settings a run uses:
    app.live_config's read timeout (APP_READ_TIMEOUT, default 90 s) and its operation deadline (run.Controller
    ._live_settings). 2026-10-21: with the client's bare 30 s default every SweetPotatoBase /observations request timed out,
    while a run gets them in about 40 s."""
    cfg = harness.app.live_config("benchmark cache")
    max_fetches = cfg.max_fetches if max_fetches is None else max_fetches
    pages = cfg.max_observation_pages                         # the app's own pages per fetch, as a run reads them (item 14)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError(f"{dest} is already built; a benchmark cache is built once")
    catalog = harness.read_catalog(source) if source.is_dir() else None
    if catalog is None:
        raise ValueError("the source cache holds no complete study and trait catalog")
    pairs: dict[str, set[str]] = {}
    per_sub: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []
    for s in subs:
        studies = [str(x["studyDbId"]) for x in s.get("studies") or []]
        found: list[str] = []
        for v in s.get("variables") or []:
            ids = variable_ids(catalog, v.get("name") or "", v.get("term"))
            if ids:
                found.append(ids[0])
            else:
                unresolved.append({"sub_id": s["sub_id"], "name": v.get("name"), "term": v.get("term")})
        for study in studies:
            pairs.setdefault(study, set()).update(found)
        calls = len(studies) * len(found) if found else len(studies)          # run.required_fetches (review 10)
        per_sub[s["sub_id"]] = {"studies": studies, "variables": found, "fetch_calls": calls,
                                "fetch_calls_with_rosters": len(studies) * (1 + len(found)), "over_max_fetches": calls > max_fetches}
    shutil.copytree(source, dest, dirs_exist_ok=True)
    base = load_settings()
    settings = dataclasses.replace(base, read_timeout=cfg.read_timeout,
                                   operation_deadline=max(base.operation_deadline, 2 * cfg.read_timeout))
    now = datetime.now(timezone.utc)
    run_id = f"benchmark_cache_{now:%Y%m%dT%H%M%S}"
    variables = sorted({v for vs in pairs.values() for v in vs})
    approval = FetchApproval(
        approval_id=f"appr_{run_id}_human_{uuid.uuid4().hex[:6]}", run_id=run_id, base_url=settings.base_url,
        endpoint_families=["observations", "observationunits", "studies"], observation_study_ids=sorted(pairs),
        observation_variable_ids=variables,
        # three attempts per request: a study record, and each roster and variable at that many pages; 3 hours per page
        max_http_attempts=max(1, 3 * (len(pairs) + pages * (len(pairs) + sum(len(v) for v in pairs.values())))),
        max_observation_studies=len(pairs), issued_at_utc=now - timedelta(seconds=1), expires_at_utc=now + timedelta(hours=3 * pages))
    ctx = srv.build_context(offline=False, cache_dir=dest, out_dir=out_dir, run_id=run_id, approval=approval, settings=settings)
    ctx.observation_max_pages = pages                          # as a run's tools read (run.Controller._build_ctx)
    studies_report: dict[str, Any] = {}
    for study in sorted(pairs):
        before = len(list(ctx.client.provenance()))
        record = srv.dispatch(ctx, "get_study", {"study_db_id": study})
        started = time.monotonic()
        units = srv.dispatch(ctx, "get_observation_units", {"study_db_id": study})
        roster_seconds = round(time.monotonic() - started, 3)
        row: dict[str, Any] = {"study_record": "ok" if record.ok else str(record.error),
                               "units": {k: (units.data or {}).get(k) for k in _FIELDS}, "roster_seconds": roster_seconds,
                               "error": None if units.ok else str(units.error), "observations": {}}
        for variable in sorted(pairs[study]):
            started = time.monotonic()
            got = srv.dispatch(ctx, "get_observations", {"study_db_id": study, "variable_db_id": variable})
            row["observations"][variable] = {**{k: (got.data or {}).get(k) for k in _FIELDS},
                                             "seconds": round(time.monotonic() - started, 3),
                                             "error": None if got.ok else str(got.error)}
        # the app's pages hold the study only when its roster and every variable came back complete (review 10)
        row["app_pages"] = bool(row["units"]["complete"]) and all(o["complete"] for o in row["observations"].values())
        records = list(ctx.client.provenance())[before:]
        row["seconds"] = round(sum(r.duration_seconds or 0.0 for r in records), 3)    # request time, vs the run's budget
        studies_report[study] = row
    for row in per_sub.values():           # a question's fetch time: its own fetches, as a run would make them, vs the budget
        if row["variables"]:
            seconds = [studies_report[s]["observations"].get(v, {}).get("seconds", 0) for s in row["studies"] for v in row["variables"]]
        else:
            seconds = [studies_report[s]["roster_seconds"] for s in row["studies"] if s in studies_report]
        row["fetch_seconds"] = round(sum(seconds), 1)
        row["over_time_budget"] = row["fetch_seconds"] > cfg.max_elapsed_seconds
    return {"built_utc": now.isoformat(), "source": str(source), "dest": str(dest), "base_url": settings.base_url,
            "catalog_fetched": catalog.fetched_at, "max_fetches": max_fetches, "page_size": settings.page_size, "pages_per_fetch": pages,
            "read_timeout": settings.read_timeout, "time_budget_seconds": cfg.max_elapsed_seconds,
            "over_time_budget": sorted(k for k, r in per_sub.items() if r["over_time_budget"]),
            "studies": studies_report, "subquestions": per_sub, "unresolved_variables": unresolved,
            "over_app_pages": sorted(s for s, r in studies_report.items() if not r["app_pages"]),
            "over_max_fetches": sorted(k for k, r in per_sub.items() if r["over_max_fetches"])}


def main(argv: list[str] | None = None) -> int:
    import benchmark  # noqa: PLC0415 - the sub-question loader beside this file

    parser = argparse.ArgumentParser(description="Build the benchmark cache once through the app's own data tools (live GETs).")
    parser.add_argument("--source", required=True, type=Path, help="a cache with the study and trait catalogs (copied, never changed)")
    parser.add_argument("--dest", required=True, type=Path, help="the new benchmark cache folder (must not exist or be empty)")
    parser.add_argument("--subquestions", type=Path, default=benchmark.SUBQUESTIONS)
    parser.add_argument("--report", required=True, type=Path, help="where to write the report (JSON)")
    parser.add_argument("--confirm", action="store_true", help="really fetch (live GETs to the database)")
    args = parser.parse_args(argv)
    subs = benchmark.load_subquestions(args.subquestions)
    if not args.confirm:
        print(f"Would build {args.dest} from {args.source} for {len(subs)} sub-questions (live GETs). Add --confirm to run.")
        return 2
    report = build(subs, source=args.source, dest=args.dest, out_dir=args.report.parent / "build_out")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"{len(report['studies'])} studies; over {report['pages_per_fetch']} pages per fetch: {report['over_app_pages'] or 'none'}; "
          f"over {report['max_fetches']} "
          f"fetch calls: {len(report['over_max_fetches'])} sub-questions; unresolved variables: {len(report['unresolved_variables'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
