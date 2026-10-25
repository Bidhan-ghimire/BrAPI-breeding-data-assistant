"""Agent-led delegation with controller-owned permissions and bounded execution.

The supervisor chooses work. These functions, not its text, determine which
actions can run. New data scope after retrieval requires a new approved question.
"""
from __future__ import annotations

from dataclasses import replace
import json
import re
from typing import Any

from agents.coordinator import add_clarification, choose_next
from analyst_tools import CV_OF_VALUES, PERCENT_MISSING
from artifacts import ArtifactError
from console import say
from llm import FakeModelClient, reply_text
from retrieval_review import verify_reviewed_tables

MAX_PLAN_REVISIONS = 2
MAX_ANALYSIS_PASSES = 2
SUPERVISOR_PROMPT = "Answer the coordinator's question before delegation (empty = cancel): "


def _call_of(claim: Any) -> tuple[str, ...]:
    """The tool call a claim came from, read from its ID: clm_v2_<artifact>_<namespace>_..., or clm_<artifact>_... for the tools
    without a namespace."""
    parts = claim.claim_id.split("_")
    return tuple(parts[:5]) if parts[:2] == ["clm", "v2"] else tuple(parts[:3])


SUMMARY_EXTRA_TURN = 4             # numeric_summary's CV and percent missing come after each call's first four claims (item 22)


def _overall_first(claims: list[Any]) -> list[Any]:
    """The claims with overall totals before per-item counts: the fewest filters first, otherwise in their own order. The
    view below keeps 20 claims; run_20261016T150253_016a3c counted 22 trait names, one count per name and the total last,
    so the coordinator never saw the total and stopped the run. Among claims with as many filters the calls take turns, each in
    its own order: the analysis tools put their headline numbers first, and on Luna (2026-10-23, E16.1-E16.5) two or more
    anova_trial calls left only the first call's claims in view, so the coordinator stopped. One call keeps its order. Claims
    without a namespace (numeric_summary, count_records, group_stats: their "call" is only the table) keep their old place, before
    the calls that take turns (review: C27.1's distinct-entry totals left the view), except numeric_summary's CV and percent
    missing, which take the fifth turn among the calls (SUMMARY_EXTRA_TURN): before them, one more claim per numeric_summary call
    pushed two anova_trial calls' variance components out of the view (2026-10-23, item 15), and after all the turns one
    anova_trial call left them out (its review; Bidhan's (22), 2026-10-24). The 20 stays, so it is a trade (review of item 22):
    two numeric_summary calls with plot lists beside one anova_trial call push its error variance, F and p of entry and CV out."""
    by_filters = sorted(claims, key=lambda claim: len(claim.filters))
    turn: dict[int, int] = {}
    seen: dict[tuple[Any, ...], int] = {}
    order: dict[tuple[str, ...], int] = {}
    for claim in by_filters:
        call = _call_of(claim)
        order.setdefault(call, len(order))
        key = (len(claim.filters), call)
        turn[id(claim)] = seen.get(key, 0) if call[:2] == ("clm", "v2") else 0
        seen[key] = seen.get(key, 0) + 1
    # claims without a namespace first, in their original order; then the calls take turns
    return sorted(by_filters, key=lambda claim: (len(claim.filters), 1, turn[id(claim)], order[_call_of(claim)])
                  if _call_of(claim)[:2] == ("clm", "v2") else (len(claim.filters), 1, SUMMARY_EXTRA_TURN, order[_call_of(claim)])
                  if _summary_extra(claim) else (len(claim.filters), 0, 0, 0))


def _summary_extra(claim: Any) -> bool:
    """numeric_summary's CV and percent missing (Breedbase's Phenotype Summary columns beside the mean, min, max, SD, count)."""
    return claim.kind == "ratio" and claim.transformation.startswith((CV_OF_VALUES, PERCENT_MISSING))


# Names for the Coordinator's view of the claims. Benchmark 2026-10-21: it saw entry IDs only and stopped A41.3 ("does not
# identify Chingova by name") and C33.5. Coach 2026-10-22 00:50 UTC (answer 2): names only from fetched tables, never from the
# model or the question; claims about entries the question names first.
NAME_BESIDE = {"germplasmDbId": "germplasmName", "observationUnitDbId": "observationUnitName", "studyDbId": "studyName",
               "locationDbId": "locationName", "observationVariableDbId": "observationVariableName"}
NAME_LIMIT = 200                           # characters of names shown per ID: database text, bounded like the caveats
VIEW_CLAIMS = 20                           # claims in the usual order; claims about entries the question names come on top
# The metadata path's facts (fields read by code from the retrieved records), bounded like the claims. Luna 2026-10-24, B26.1 and
# B30.1: the view held only claims and caveats, so a question about what a study description says always looked unanswered and
# the Coordinator stopped (every run since 22 Oct with facts and no claims ended "blocked"; coach 09:49 UTC, NEXT 1). One study has
# about 20 fields, so the entities take turns (review of this change: a cap of 20 showed only the first study).
VIEW_FACTS, VIEW_FACT_CHARS = 60, 300


def _view_facts(facts: list[Any]) -> list[dict[str, Any]]:
    """The metadata facts the Coordinator sees: each entity's fields in their own order, the entities taking turns, at most
    VIEW_FACTS. A value longer than VIEW_FACT_CHARS is cut at its last space before that and marked value_cut (review of this
    change: "202-208oF" reached it as "202-2"); the person's answer keeps the whole value."""
    groups: dict[tuple[str, str], list[Any]] = {}
    for fact in facts:
        groups.setdefault((fact.entity_kind, fact.entity_id), []).append(fact)
    turns: list[Any] = []
    for depth in range(max((len(group) for group in groups.values()), default=0)):
        turns.extend(group[depth] for group in groups.values() if depth < len(group))
    shown = []
    for fact in turns[:VIEW_FACTS]:
        value, cut = fact.value, fact.value is not None and len(fact.value) > VIEW_FACT_CHARS
        if cut:
            head = value[:VIEW_FACT_CHARS]
            value = head.rsplit(" ", 1)[0].rstrip() if " " in head else head
        shown.append({"entity": f"{fact.entity_kind} {fact.entity_id}", "field": fact.field, "value": value,
                      **({"value_cut": True} if cut else {}),
                      "missing_reason": None if fact.missing_reason is None else fact.missing_reason[:200]})
    return shown


def _fetched_names(c: Any) -> dict[tuple[str, str], set[str]]:
    """(ID column, ID) -> the names beside that ID in this run's fetched tables: every table the Retriever's tools returned
    (plot records, plot lists, catalog exports), never one an analysis derived. IDs are exact text; names are trimmed."""
    retrieval, registry = getattr(c, "retrieval", None), getattr(getattr(c, "ctx", None), "registry", None)
    found: dict[tuple[str, str], set[str]] = {}
    for artifact in (retrieval.artifacts if retrieval is not None and registry is not None else []):
        try:
            rows = registry.load(artifact.artifact_id)[1]
        except Exception:  # noqa: BLE001 - a table that cannot be read only means no names
            continue
        for row in rows:
            for column, name_column in NAME_BESIDE.items():
                key, name = row.get(column) or "", (row.get(name_column) or "").strip()
                if key.strip() and name:
                    found.setdefault((column, key), set()).add(name)
    return found


def _shown_names(names: set[str]) -> str:
    """The names, each whole, up to NAME_LIMIT characters (a first name longer than that is cut), then how many are left out."""
    ordered, shown = sorted(names), []
    for name in ordered:
        if shown and len("; ".join([*shown, name])) > NAME_LIMIT:
            break
        shown.append(name[:NAME_LIMIT])
    left = len(ordered) - len(shown)
    return "; ".join(shown) + (f" (+{left} more)" if left else "")


def _named_in(question: str, names: dict[tuple[str, str], set[str]]) -> set[tuple[str, str]]:
    """The IDs one of whose fetched names (three letters or more, any case) stands in the question as a whole word, each name
    whole. It only orders the claims; the names shown come from the fetched tables. A plain substring test finds the
    candidates first, so the word test runs on a few names, not on every one."""
    folded = question.casefold()
    return {key for key, found in names.items() for name in found
            if len(name) >= 3 and name.casefold() in folded and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", question, re.IGNORECASE)}


def _view_claims(claims: list[Any], named: set[tuple[str, str]]) -> list[Any]:
    """Claims about an entry the question names first (at most VIEW_CLAIMS), then the first VIEW_CLAIMS of the usual order,
    overall totals first, without repeats. Review of 2026-10-22: one list of 20 with the named claims first let a question
    naming seven entries push the overall totals out, as on 2026-10-16."""
    ordered = _overall_first(claims)
    about = [claim for claim in ordered if any(item in named for item in claim.filters.items())][:VIEW_CLAIMS]
    taken = {id(claim) for claim in about}
    return about + [claim for claim in ordered if id(claim) not in taken][:VIEW_CLAIMS]


def _with_names(claim: Any, names: dict[tuple[str, str], set[str]]) -> dict[str, Any]:
    view = claim.model_dump(mode="json")
    beside = {f"{column} {key}": _shown_names(names[(column, key)]) for column, key in claim.filters.items() if (column, key) in names}
    if beside:
        view["names_from_fetched_tables"] = beside
    return view


def _not_shown(analysis: Any, omitted: int) -> dict[str, str]:
    """The claims the view leaves out, in words (coach 2026-10-24 12:50 UTC, GAP 3; Luna A38.2: "claims_omitted": 3273 was read
    as results the analysis omitted, and a right count was stopped)."""
    if omitted <= 0:
        return {}
    where = ("the analysis is complete and all claims are in the saved result" if analysis.status == "completed"
             else "all claims are in the saved result")
    return {"claims_not_shown": f"{omitted} further claim{'' if omitted == 1 else 's'} not shown in this view; {where}"}


def _state(c: Any, stage: str) -> dict[str, Any]:
    """Only bounded evidence summaries; never authority supplied by model text."""
    retrieval = c.retrieval
    analysis = c.analysis
    names = _fetched_names(c) if analysis is not None else {}
    shown = [] if analysis is None else _view_claims(analysis.claims,
                                                     _named_in(getattr(getattr(c, "config", None), "question", "") or "", names))
    return {
        "stage": stage,
        "plan": c.plan.model_dump(mode="json") if c.plan else None,
        "retrieval": None if retrieval is None else {
            "status": retrieval.status,
            "tables": [{"id": a.artifact_id, "rows": a.row_count, "complete": a.complete}
                       for a in retrieval.artifacts[:20]],
            "tables_omitted": max(0, len(retrieval.artifacts) - 20),
        },
        "data_review": c.data_review["status"],
        "analysis": None if analysis is None else {
            "status": analysis.status,
            "claims": [_with_names(claim, names) for claim in shown],
            "claims_omitted": len(analysis.claims) - len(shown),
            **_not_shown(analysis, len(analysis.claims) - len(shown)),
            "caveats": [v[:400] for v in analysis.caveats[:10]],
            "metadata_facts": _view_facts(list(getattr(analysis, "metadata_facts", None) or [])),
            "metadata_facts_omitted": max(0, len(getattr(analysis, "metadata_facts", None) or []) - VIEW_FACTS),
        },
        "history": c.supervision["decisions"],
        "plan_revisions": c.supervision["plan_revisions"],
        "analysis_passes": c.supervision["analysis_passes"],
    }


def _allowed(c: Any, stage: str) -> list[str]:
    # The phase is controller-owned, never parsed from model-provided state.
    if stage == "planned" and c.plan is not None and c.plan.status == "ready" and c.retrieval is None:
        allowed = ["retrieve", "stop"]
        if c.supervision["plan_revisions"] < MAX_PLAN_REVISIONS:
            allowed.append("revise_plan")
            if c.clarification_answers < c.config.max_clarification_rounds:   # one cap for planner and coordinator questions
                allowed.append("clarify")
        return allowed
    reviewed = (c.retrieval is not None and c.retrieval.status == "completed"
                and c.data_review["status"] == "approved" and c.approval is not None)
    if stage == "data_reviewed" and reviewed:
        return ["analyze", "stop"]
    if stage == "analyzed" and reviewed and c.analysis is not None and c.analysis.status == "completed":
        return (["finish", "analyze", "stop"] if c.supervision["analysis_passes"] < MAX_ANALYSIS_PASSES
                else ["finish", "stop"])
    return ["stop"]


async def _choose(c: Any, stage: str):
    if len(c.supervision["decisions"]) >= c.config.max_supervisor_turns:
        c.execution_status = "limit_reached"
        c.notes.append("coordinator delegation decision limit reached")
        return None
    allowed = _allowed(c, stage)
    c._enter("supervise")
    if c.models.supervisor is not None:
        model = c.models.supervisor
    elif c.config.mock_model:
        action = {"planned": "retrieve", "data_reviewed": "analyze", "analyzed": "finish"}[stage]
        model = FakeModelClient([reply_text(json.dumps({"status": "completed", "payload": {
            "action": action, "reason": "Synthetic offline supervisor script."}}))])
    else:
        model = await c._real_model()
    choice = await choose_next(model=model, question=c.config.question, state=_state(c, stage),
                               allowed_actions=allowed, budget=c.budget, run_id=c.run_id,
                               log_dir=c.run_dir.parent, model_requested=c.models.requested,
                               turn=len(c.supervision["decisions"]) + 1, clock=c._supervisor_clock)
    c.agents.append(choice.agent)
    c.notes.extend("supervisor: " + note for note in choice.notes)
    decision = choice.decision
    c.supervision["decisions"].append({"stage": stage, "allowed_actions": allowed,
        "status": choice.status, "decision": decision.model_dump(mode="json") if decision else None})
    if choice.status != "completed" or decision is None:
        c.execution_status = choice.status if choice.status in ("failed", "blocked", "limit_reached", "needs_clarification") else "failed"
        return None
    # Recheck at the execution boundary, independently of decision parsing.
    if decision.action not in _allowed(c, stage):
        c.execution_status = "blocked"
        c.notes.append("supervisor requested an action not permitted by the current execution state")
        return None
    if decision.action == "stop":
        c.execution_status = "blocked"
        c.notes.append("coordinator stopped the run: " + decision.reason)
        return None
    return decision


async def run_agent_led(c: Any) -> bool:
    """A bounded delegation loop; Python owns every approval and data gate."""
    while True:
        c._enter("plan")
        await c.plan_phase()
        if c.plan is None or c.plan.status != "ready":
            c._enter("clarify")
            if not await c.clarify_phase():
                return False
        c._enter("resolve")
        c.manifest = c.resolve_manifest()
        choice = await _choose(c, "planned")
        if choice is None:
            return False
        if choice.action in ("revise_plan", "clarify"):
            c.supervision["plan_revisions"] += 1
            c.supervision["plan_history"].append(c.plan.model_dump(mode="json"))
            c.approval = None
            c.manifest = None
            if choice.action == "revise_plan":
                c._supervisor_revision = choice.instruction
            else:
                c.supervisor_questions = [choice.question]
                if c.config.auto:
                    c.execution_status = "needs_clarification"
                    return False
                say("\nTHE COORDINATOR ASKS: " + choice.question)
                answer = c._ask(SUPERVISOR_PROMPT)
                if not answer or answer.lower() == "cancel":
                    c.execution_status = "canceled"
                    c.notes.append("coordinator clarification canceled; no retrieval was performed")
                    return False
                c.supervisor_answers += 1
                c.clarification_answers += 1
                c.notes.append(f"clarification {c.clarification_answers}: the coordinator asked: {choice.question} - "
                               f"answer{' (simulated)' if c.answers_simulated else ''}: {answer}")
                c.planner_questions.append(choice.question)
                c.config = replace(c.config, question=add_clarification(c.config.question, answer))
                c.supervision["clarifications"].append({"question": choice.question, "answer": answer,
                                                       "simulated": c.answers_simulated})
                c._supervisor_revision = None
            c.plan = None
            continue
        # 'retrieve' is a delegation request, never an approval.
        c._enter("approve")
        if not c.approve_phase():
            if c.pending_approval_revision is None:
                return False
            revision = c.pending_approval_revision
            c.pending_approval_revision = None
            c.config = replace(c.config, question=add_clarification(c.config.question, revision))
            c._supervisor_revision = None
            continue
        break

    c._enter("retrieve")
    await c.retrieve_phase()
    if c.retrieval is None or c.retrieval.status != "completed":
        c.execution_status = "canceled" if c.retrieval_canceled else (c.retrieval.status if c.retrieval else "failed")
        c.notes.append("retrieval did not complete; analysis was not run")
        return False
    c._enter("review_data")
    if not c.review_data_phase():
        return False
    stage = "data_reviewed"
    while True:
        choice = await _choose(c, stage)
        if choice is None:
            return False
        if choice.action == "finish":
            try:
                verify_reviewed_tables(c.ctx, c.retrieval, c.data_review["artifacts"])
            except ArtifactError as exc:
                c.execution_status = "blocked"
                c.notes.append("reviewed data changed before final rendering: " + str(exc))
                return False
            c._enter("render")
            c.render_phase()
            c.execution_status = "completed"
            c._enter("accept")
            c.accept_phase()
            return True
        # A supervisor can ask for another calculation pass on the SAME reviewed
        # evidence, but cannot fetch new data or supply a new artifact allowlist.
        try:
            verify_reviewed_tables(c.ctx, c.retrieval, c.data_review["artifacts"])
        except ArtifactError as exc:
            c.execution_status = "blocked"
            c.notes.append("reviewed data changed before analysis: " + str(exc))
            return False
        c._supervisor_analysis_instruction = choice.instruction
        c.supervision["analysis_passes"] += 1
        c._enter("analyze")
        await c.analyze_phase()
        if c.analysis is not None:
            c.supervision["analysis_history"].append(c.analysis.model_dump(mode="json"))
        if c.analysis is None or c.analysis.status != "completed":
            c.execution_status = "canceled" if c.analysis_canceled else (c.analysis.status if c.analysis else "failed")
            return False
        stage = "analyzed"
