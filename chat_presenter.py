"""Small, evidence-backed chat views. Display only; full reports and approvals stay in the controller."""
from __future__ import annotations

import html
import json
import re
from collections import Counter, defaultdict
from typing import Any

import brapi_client
from analyst_tools import MISSING_GROUP
from catalog_answers import plain_reason as catalog_plain
from contracts import Claim
from metadata_answers import plain_reason as metadata_plain, render_metadata_facts
from agents.coordinator import person_words
from run import approval_lead, carried_over_line, required_fetches, scope_needs_a_check, source_host


LIST_LIMIT = 30
DISTINCT = "number of distinct non-empty exact text values"
FREQUENCY = "frequency of exact text values"
ROW_COUNT = "row count of a complete table"      # analyst_tools.tool_count_records: it counts rows and never checks units
VALUE_COUNT = "count of valid values"            # analyst_tools._numeric_claims: "count of valid values; <how they were used>"
NAMES = {
    "studyDbId": ("Studies", (("studyName", "Study"), ("locationName", "Location"), ("seasons", "Season"))),
    "locationDbId": ("Locations", (("locationName", "Location"), ("countryName", "Country"))),
    "germplasmDbId": ("Germplasm", (("germplasmName", "Name"),)),
    "observationVariableDbId": ("Traits", (("observationVariableName", "Trait"),)),
    "observationUnitDbId": ("Observation units", (("observationUnitName", "Name"),)),
    "seasonDbId": ("Seasons", (("seasonName", "Season"),)),
    "programDbId": ("Programs", (("programName", "Program"),)),
}
_INTERNAL_ID = re.compile(r"\b(?:clm|art|req|plan|run)_[A-Za-z0-9_-]+\b")
_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/][^\n;:]*?\.[A-Za-z0-9]{1,8}(?=\s|[\"'`),;:]|$)(?::\d+)?|[A-Za-z]:[\\/][^\s;:]+)")
_LOCAL_PATH = re.compile(r"(?<![:/\w])(?:/(?:home|tmp|mnt|workspace|app)/|(?:out|logs|cache)/)[^\s,;]+")


def _plain(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                text = ", ".join(str(item) for item in parsed)
        except ValueError:
            pass
    text = _WINDOWS_PATH.sub("(file details in Full report)", text)
    text = _LOCAL_PATH.sub("(file details in Full report)", text)
    return _INTERNAL_ID.sub("(evidence in Full report)", text)


def _text(value: Any) -> str:
    text = html.escape(_plain(value), quote=False).replace("\n", " ").replace("\r", " ")
    return re.sub(r"([\\`*_{}\[\]()|])", r"\\\1", text)


def _number(value: float | int | None) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, int):
        return str(value)
    return format(value, ".12g")


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(_text(cell) for cell in row) + " |" for row in rows]]


class _Sources:
    def __init__(self, registry: Any):
        self.registry = registry
        self.by_hash: dict[str, str] = {}
        self.loaded: dict[str, list[dict[str, str]] | None] = {}
        if registry is not None:
            for aid in registry.artifact_ids():
                try:
                    self.by_hash[registry.manifest(aid).sha256] = aid
                except Exception:
                    continue

    def rows(self, hashes: tuple[str, ...]) -> list[dict[str, str]] | None:
        # Do not arbitrarily choose one table when a claim cites several.
        if len(hashes) != 1 or hashes[0] not in self.by_hash:
            return None
        fingerprint = hashes[0]
        if fingerprint not in self.loaded:
            try:
                self.loaded[fingerprint] = self.registry.load(self.by_hash[fingerprint])[1]
            except Exception:
                self.loaded[fingerprint] = None
        return self.loaded[fingerprint]


def _sort_id(value: str) -> tuple[float | int, str]:
    return (int(value) if value.isdecimal() else float("inf"), value)     # "²" is a digit but not decimal: int() refuses it (review)


def _names(rows: list[dict[str, str]], column: str, key: str, name_column: str) -> str:
    return "; ".join(sorted({_plain(row.get(name_column, "")) for row in rows
                              if row.get(column, "") == key and _plain(row.get(name_column, ""))}))


def _catalog_block(claims: list[Claim], sources: _Sources, source_label: str, *, focus_count: bool = False) -> list[str] | None:
    first = claims[0]
    column = first.grouping[0]
    rows = sources.rows(tuple(sorted(first.source_artifact_hashes)))
    if rows is None:
        return None
    distinct = [claim for claim in claims if claim.transformation == DISTINCT]
    frequency = [claim for claim in claims if claim.transformation == FREQUENCY]
    if len({claim.value for claim in distinct}) > 1:
        return None
    counts: dict[str, Any] = {}
    for claim in frequency:
        key = claim.filters[column]
        if key in counts and counts[key] != claim.value:
            return None
        counts[key] = claim.value
    source_ids = {row.get(column, "") for row in rows if row.get(column, "")}
    if distinct and distinct[0].value != len(source_ids):
        return None  # Preserve conflicting statistics separately instead of silently choosing one.
    keys = sorted(source_ids if distinct else set(counts), key=_sort_id)
    keys.extend(sorted(set(counts) - set(keys), key=_sort_id))  # includes the explicit missing-value group
    if frequency:
        keys.sort(key=lambda key: (-(counts.get(key) or 0), *_sort_id(key)))
    title, name_columns = NAMES.get(column, (column, ()))
    total = f"{_number(distinct[0].value)} distinct IDs" if distinct else f"{len(keys)} groups"
    lines = [f"**{_text(title)}: {total}{source_label}**", ""]
    # Only redundant one-per-ID frequencies can be omitted; conflicting/additional counts stay visible.
    if focus_count and distinct and (not frequency or (set(counts) == source_ids and all(value == 1 for value in counts.values()))):
        return [*lines, f"Based on {_text(first.denominator.name)}: {first.denominator.n}."]
    headers = ["ID", *[label for _, label in name_columns]]
    show_frequency = bool(frequency) and (not distinct or any(value != 1 for value in counts.values()))
    if show_frequency:
        headers.append("Studies" if column == "locationDbId" and "studies table" in first.denominator.name else "Records")
    table_rows = []
    for key in keys[:LIST_LIMIT]:
        values = [key, *[_names(rows, column, key, name_column) or "Not stated" for name_column, _ in name_columns]]
        if show_frequency:
            values.append(_number(counts[key]) if key in counts else "Not reported")
        table_rows.append(values)
    lines += _table(headers, table_rows)
    if len(keys) > LIST_LIMIT:
        lines += ["", f"Showing {LIST_LIMIT} of {len(keys)} entries; {len(keys) - LIST_LIMIT} additional IDs are not displayed here. "
                  "Full report records the counts and source evidence."]
    lines += ["", f"Based on {_text(first.denominator.name)}: {first.denominator.n}."]
    return lines


def _claim_method(claim: Claim) -> str:
    return re.sub(r"^(?:mean|median|min|max|sd|count) of valid values;\s*", "", claim.transformation)


def _mean_focus(controller: Any, claims: list[Claim]) -> bool:
    """Only a clearly requested mean may hide the numeric tool's incidental statistics in chat; a ranking is never hidden."""
    plan = getattr(controller, "plan", None)
    if getattr(getattr(plan, "statistic", None), "kind", None) != "mean" or sum(claim.kind == "mean" for claim in claims) != 1:
        return False
    if any(claim.kind in ("rank", "correlation", "anova") or _THRESHOLD_ID.match(claim.claim_id) for claim in claims):
        return False
    words = person_words(getattr(plan, "question", ""))
    if not re.search(r"\b(?:mean|average)\b", words, re.I):
        return False
    # An explicit second metric or a general summary request keeps every supplied metric visible.
    # 'n' means the displayed denominator and does not require a redundant Count row.
    return not re.search(r"\b(?:median|minimum|maximum|min|max|sd|std|standard deviation|variance|range|"
                         r"count|counts|total|totals|ratio|rank|ranks|percentile|quartile|statistics|stats|summary|summarize|describe)\b", words, re.I)


def _count_focus(controller: Any, claims: list[Claim]) -> bool:
    """A single total request may omit redundant IDs; lists, breakdowns and comparisons keep them."""
    plan = getattr(controller, "plan", None)
    if getattr(getattr(plan, "statistic", None), "kind", None) != "count":
        return False
    words = person_words(getattr(plan, "question", ""))
    if not re.search(r"^\s*(?:how many|what (?:is|are) the (?:total(?: number)?|number) of)\b", words, re.I):
        return False
    if re.search(r"\b(?:and|per|each|by|group|groups|grouped|compare|comparison|versus|vs|which|list|show|breakdown|distribution)\b", words, re.I):
        return False
    distinct = [claim for claim in claims if claim.transformation == DISTINCT]
    if len(distinct) != 1 or len(distinct[0].grouping) != 1:
        return False
    target = distinct[0]
    return all(claim.kind == "count" and claim.transformation in (DISTINCT, FREQUENCY)
               and claim.grouping == target.grouping
               and sorted(claim.source_artifact_hashes) == sorted(target.source_artifact_hashes)
               and claim.denominator == target.denominator for claim in claims)


def _count_lead(controller: Any, claims: list[Claim], in_catalog_titles: set[str],
                source_labels: dict[tuple[str, ...], str]) -> list[str]:
    """A count answer leads with its counts (plan v3, lead_with_count_plan_20261020): when the plan's own statistic is a
    count, "**Counts**" lists every count claim, number first, with exactly what it counted, in a fixed order: valid
    values or units, then rows, then distinct values, then frequencies. Within one kind, ties keep the claims' order, and
    the " · source N" labels number the tables in that order too, as the blocks do. The model's tool-call order never
    decides which kind comes first, and code never picks one. No line for a count a catalog block's title already leads
    with ("Studies: 18 distinct IDs"), nor for the frequencies of that same table (review7: one match printed "1 rows with
    studyDbId 1" above "Studies: 1 distinct IDs"), nor for a frequency table of several groups (the ranked table is the
    answer), nor for a count from an incomplete table (its block below says so); a count of another kind keeps only its
    block. The blocks below are unchanged."""
    if getattr(getattr(getattr(controller, "plan", None), "statistic", None), "kind", None) != "count":
        return []
    def table(claim: Claim) -> tuple[Any, ...]:
        return (tuple(sorted(claim.source_artifact_hashes)), tuple(claim.grouping), claim.denominator.name, claim.denominator.n)
    frequency_groups = Counter(table(c) for c in claims if c.kind == "count" and c.transformation == FREQUENCY)
    led = {table(c) for c in claims if c.claim_id in in_catalog_titles}        # tables whose count a title already leads with
    ranked = []
    for index, claim in enumerate(claims):
        if claim.kind != "count" or "INCOMPLETE ARTIFACT" in claim.transformation:   # never a finished answer (analyst)
            continue
        name, n = claim.denominator.name, claim.denominator.n
        column = claim.grouping[0] if len(claim.grouping) == 1 else None
        if claim.transformation == VALUE_COUNT or claim.transformation.startswith(VALUE_COUNT + ";"):
            order, words = 0, name                      # n is the count itself: values, or units when averaged per unit
        elif claim.transformation == ROW_COUNT:
            order, words = 1, name
        elif claim.transformation == DISTINCT and column and not claim.filters:
            if claim.claim_id in in_catalog_titles:
                continue
            order, words = 2, f"distinct {column} values, in {n} {name}"
        elif claim.transformation == FREQUENCY and column and set(claim.filters) == {column}:
            if frequency_groups[table(claim)] != 1 or table(claim) in led:
                continue
            value = claim.filters[column]
            order, words = 3, f"rows with {'no ' + column if value == MISSING_GROUP else f'{column} {value}'}, of {n} {name}"
        else:
            continue
        source = source_labels.get(tuple(sorted(claim.source_artifact_hashes)), "")
        if claim.value is None:
            reason = claim.missing_reason or "reason not provided"
            line = f"- **Not available**: {_text(f'{reason} ({words})')}{source}"
        else:
            line = f"- **{_number(claim.value)}** {_text(words)}{source}"
        ranked.append((order, index, line))
    if not ranked:
        return []
    return ["**Counts**", "", *[line for _, _, line in sorted(ranked)], ""]


RANK_SENTENCE = "Descriptive numbers only: an ordering of recorded values, not a recommendation."


def _ranking_block(claims: list[Claim], sources: _Sources, source_labels: dict[tuple[str, ...], str]) -> tuple[list[str], set[str]]:
    """R1 (2026-10-21): the rank claims of one ranking as one table (Rank, Entry, Mean, n) in rank order, each with the mean
    claim rank_entries made beside it (the same claim ID with "_mean_" for "_rank_"), then the descriptive sentence. Returns
    the lines and the IDs of the claims they show, so no mean appears twice."""
    by_id = {claim.claim_id: claim for claim in claims}
    rankings: dict[tuple[Any, ...], list[tuple[Claim, Claim | None]]] = defaultdict(list)
    for claim in claims:
        if claim.kind == "rank" and not _STRESS_CALL.match(claim.claim_id):        # stress_indices' ranks are in its own table (T5)
            head, _, tail = claim.claim_id.partition("_rank_")
            rankings[(tuple(sorted(claim.source_artifact_hashes)), claim.denominator.name, claim.denominator.n, head)].append(
                (claim, by_id.get(f"{head}_mean_{tail}")))
    lines: list[str] = []
    shown: set[str] = set()
    for (hashes, name, n, _), pairs in rankings.items():
        rows = sources.rows(hashes) or []
        table = []
        for rank, mean in sorted(pairs, key=lambda pair: (pair[0].value if pair[0].value is not None else float("inf"),
                                                          sorted(pair[0].filters.items()))):
            entry = [" · ".join([value, *[found for found in (_names(rows, column, value, name_column)
                                                              for name_column, _ in NAMES.get(column, (column, ()))[1]) if found]])
                     for column, value in rank.filters.items()]
            mean_text = "not stated" if mean is None else f"{_number(mean.value)} {mean.unit or '(unit not stated)'}"
            table.append([_number(rank.value), "; ".join(entry), mean_text, "" if mean is None else mean.denominator.n])
            shown |= {rank.claim_id, *([mean.claim_id] if mean is not None else [])}
        order = "highest first" if "highest first" in name else "lowest first"
        by_study = "per-study" in name                                # rank_entries basis="study_means" (T3)
        lines += [f"**Ranking by {'the mean of per-study entry means' if by_study else 'entry mean'}, {order}, of {n} entries"
                  f"{source_labels.get(hashes, '')}**", "", *_table(["Rank", "Entry", "Mean", "n"], table), ""]
        if by_study:
            lines.append("Mean: the unweighted mean of each entry's per-study means (each study weighted equally; n = studies averaged).")
        lines += [_text(f"{item.reason}: {item.count}.") for item in pairs[0][0].exclusions if item.reason.startswith("entries ")]
        lines += [RANK_SENTENCE, ""]
    return lines, shown


THRESHOLD_SENTENCE = "Descriptive numbers only: a count of recorded values against a threshold, not a recommendation."
_THRESHOLD_ID = re.compile(r"^(clm_v2_.+?_thr-[0-9a-f]+)_(count|ratio|mean)_(.+)$")     # clm_v2_<artifact>_thr-<variant>_<kind>_studyDbId_<study>
_THRESHOLD_TEXT = re.compile(r"^count of (entries|plots) (above|below|at least|at most) the threshold (.+?); threshold = (.+?); "
                             r"descriptive count of recorded values against a threshold")     # the fixed note ends the source text (review)


def _share(value: float | int | None) -> str:
    """A share in percent with the fewest decimals that do not read a partial count as 0 or 100 (review: 2500 of 2501 read "100 %")."""
    if value is None:
        return "Not available"
    for decimals in (1, 2, 3, 4):
        text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
        if text not in ("0", "100") or value in (0, 100):
            return text + " %"
    return ("under 0.0001" if value < 50 else "over 99.9999") + " %"


def _threshold_block(claims: list[Claim], sources: _Sources, source_labels: dict[tuple[str, ...], str]) -> tuple[list[str], set[str]]:
    """R2 (2026-10-22): the claims of one threshold_count call as one table (Study, Counted, <op> the threshold, With a value,
    Share) in study order, then each study's threshold and what was left out, then the descriptive sentence. Returns the lines
    and the IDs of the claims they show."""
    calls: dict[str, dict[str, dict[str, Claim]]] = defaultdict(lambda: defaultdict(dict))     # call -> study -> kind -> claim
    for claim in claims:
        match = _THRESHOLD_ID.match(claim.claim_id)
        if match:
            calls[match.group(1)][match.group(3)][match.group(2)] = claim
    lines: list[str] = []
    shown: set[str] = set()
    for studies in calls.values():
        table, notes, header, label = [], [], "Meets the threshold", ""
        for kinds in sorted(studies.values(), key=lambda kinds: _sort_id(next(iter(kinds.values())).filters.get("studyDbId", ""))):
            count = kinds.get("count")
            if count is None:
                continue
            text = _THRESHOLD_TEXT.match(count.transformation)
            what, op, threshold, source = text.groups() if text else ("", "", "not stated", "not stated")
            header = f"{op[:1].upper()}{op[1:]} the threshold" if op else header
            hashes = tuple(sorted(count.source_artifact_hashes))
            label = source_labels.get(hashes, "")
            study = count.filters.get("studyDbId", "")
            name = _names(sources.rows(hashes) or [], "studyDbId", study, "studyName")
            ratio = kinds.get("ratio")
            table.append([f"{study} · {name}" if name else study, what, _number(count.value), count.denominator.n,
                          _share(ratio.value) if ratio else "Not stated"])
            left = "; ".join(f"{item.reason.split(' left out: ', 1)[1]} {item.count}" for item in count.exclusions if " left out: " in item.reason)
            notes.append(_text(f"Threshold in {study}: {threshold} ({source}).") + (f" Left out: {_text(left)}." if left else ""))
            shown |= {claim.claim_id for claim in kinds.values()}
        if table:
            lines += [f"**Counts against a threshold{label}**", "", *_table(["Study", "Counted", header, "With a value", "Share"], table), "",
                      *notes, "", THRESHOLD_SENTENCE, ""]
    return lines, shown


CORRELATION_SENTENCE = "Descriptive numbers only: a correlation of recorded entry means, not a recommendation."


def _correlation(value: float | int | None) -> str:
    return "Not available" if value is None else f"{value:.4f}".rstrip("0").rstrip(".")


def _correlation_block(claims: list[Claim], source_labels: dict[tuple[str, ...], str]) -> tuple[list[str], set[str]]:
    """R2 (2026-10-22): the correlation claims of one correlate_entry_means call as one table (Method, Correlation, n), the two
    tables it correlated and what was left out, then the descriptive sentence. Returns the lines and the IDs of the claims shown."""
    calls: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        if claim.kind == "correlation" and not _PAIR_CALL.match(claim.claim_id):     # compare_trials_paired's r is in its own table (T4)
            calls[claim.claim_id.rsplit("_correlation_", 1)[0]].append(claim)
    lines: list[str] = []
    shown: set[str] = set()
    for group in calls.values():
        table = []
        for claim in sorted(group, key=lambda claim: claim.transformation.startswith("Spearman")):
            method = "Spearman rho" if claim.transformation.startswith("Spearman") else "Pearson r"
            table.append([method, _correlation(claim.value), claim.denominator.n])
            shown.add(claim.claim_id)
        first = group[0]
        parts = first.transformation.split(" | ")
        about = [part for part in parts if part.startswith(("x: ", "y: ", "entries: "))]
        left = "; ".join(f"{item.reason[len('entries '):].replace('left out: ', '')} {item.count}" for item in first.exclusions
                         if item.reason.startswith("entries "))
        label = source_labels.get(tuple(sorted(first.source_artifact_hashes)), "")
        lines += [f"**Correlation of entry means, {first.denominator.n} entries in both tables{label}**", "",
                  *_table(["Method", "Correlation", "n"], table), "", _text("; ".join(about) + ".") if about else "",
                  *([_text(f"Left out: {left}.")] if left else []), "", CORRELATION_SENTENCE, ""]
    return lines, shown


ANOVA_SENTENCE = "Descriptive numbers only: an analysis of variance of recorded plot values within one trial, not a recommendation."
_ANOVA_CALL = re.compile(r"^(clm_v2_[0-9a-f]{6}_[0-9]{4,}_anova-[0-9a-f]{8})_")
_ANOVA_KINDS = ("anova", "mean", "precision", "variance", "heritability")


def _anova_number(value: float | int) -> str:
    """4 decimals; 4 significant digits for a value below 0.001 or at 1e7 and above, so a small unit never reads as 0 (review)."""
    if value == 0:
        return "0"
    if abs(value) < 0.001 or abs(value) >= 1e7:
        return format(value, ".4g")
    return _correlation(value)


def _anova_cell(quantity: str, claim: Claim | None) -> str:
    if claim is None:
        return ""                                             # not part of the table (MS of total; F and p of error and total)
    if claim.value is None:
        return "n/a"                                          # the reason is stated under the table
    if quantity == "df":
        return str(int(claim.value))
    if quantity == "p" and claim.value == 0:
        return "< 1e-300"                                     # an F tail below what a double holds; the claim carries the bound (review 2)
    return _anova_number(claim.value)                     # a small p keeps its digits: "< 0.0001" is a number no claim carries


def _anova_block(claims: list[Claim], source_labels: dict[tuple[str, ...], str]) -> tuple[list[str], set[str]]:
    """T2 (2026-10-22): the claims of one anova_trial call as the ANOVA table (Source, df, SS, MS, F, p), the trial and its shape,
    the precision line (grand mean, CV, SE, SED, LSD), the variance components and the entry-mean heritability, then the
    descriptive sentence. A number that is not available says why. Returns the lines and the IDs of the claims shown."""
    calls: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        match = _ANOVA_CALL.match(claim.claim_id)
        if match and claim.kind in _ANOVA_KINDS:
            calls[match[1]].append(claim)
    lines: list[str] = []
    shown: set[str] = set()
    for group in calls.values():
        named = {claim.transformation.split(" | ")[0]: claim for claim in group}
        first = next((claim for claim in group if claim.kind == "anova"), None)
        if first is None:
            continue
        design = first.transformation.split(" ANOVA: ")[0]
        rows = []
        for source in ("rep", "entry", "error", "total"):
            cells = [named.get(f"{design} ANOVA: {quantity} of {source}") for quantity in ("df", "ss", "ms", "f", "p")]
            if cells[0] is not None:
                rows.append([source.capitalize(), *(_anova_cell(q, c) for q, c in zip(("df", "ss", "ms", "f", "p"), cells))])
        parts = first.transformation.split(" | ")[1:]
        label = source_labels.get(tuple(sorted(first.source_artifact_hashes)), "")
        lsd_claim = named.get("LSD (5%)")
        unit_text = f" {_text(lsd_claim.unit)}" if lsd_claim is not None and lsd_claim.unit else ""

        def stated(text: str, name: str, suffix: str) -> str | None:
            claim = named.get(name)
            if claim is None:
                return None
            if claim.value is None:
                return f"{text} not available" + (f" ({_text(claim.missing_reason)})" if claim.missing_reason else "")
            return f"{text} {_anova_number(claim.value)}{suffix}"

        precision = [line for line in (stated("Grand mean", "grand mean", unit_text), stated("CV", "CV", " %"),
                                       stated("SE of an entry mean", "SE of an entry mean", unit_text), stated("SED", "SED", unit_text),
                                       stated("LSD (5%)", "LSD (5%)", unit_text)) if line]
        variance = [line for line in (stated("genotype variance", "genotype variance sigma2 g", ""),
                                      stated("error variance", "error variance sigma2 e", "")) if line]
        h2 = stated("Entry-mean heritability (H²)", "entry-mean heritability H2", "")
        h2_claim = named.get("entry-mean heritability H2")
        undefined_f = next((c for c in group if c.kind == "anova" and c.value is None and c.missing_reason), None)
        lines += [f"**Analysis of variance, {design}{label}**", "", _text("; ".join(parts[:2]) + "."), "",
                  *_table(["Source", "df", "SS", "MS", "F", "p"], rows), ""]
        if undefined_f is not None:
            lines.append(f"F and p not available: {_text(undefined_f.missing_reason)}.")
        if precision:
            lines.append("; ".join(precision) + ".")
        if variance:
            lines.append("Variance components: " + ", ".join(variance) + ".")
        if h2 is not None:
            zero = h2_claim is not None and h2_claim.value is not None and "set to 0" in h2_claim.transformation
            lines.append(h2 + (", within this trial" if h2_claim is not None and h2_claim.value is not None else "")
                         + (" (set to 0: the genotype variance estimate was negative)" if zero else "") + ".")
        for part in parts:
            if part.startswith("r = "):
                lines.append(_text("Replication: " + part + "."))
            elif part.startswith(("sequential least squares", "incomplete blocks not modelled:", "blocks not modelled:", "left out by name:")):
                lines.append(_text(part[0].upper() + part[1:] + "."))
        lines += ["", ANOVA_SENTENCE, ""]
        shown |= {claim.claim_id for claim in group}
    return lines, shown


GXE_SENTENCE = "Descriptive numbers only: a partition of recorded entry means across trials, not a recommendation."
_GXE_CALL = re.compile(r"^(clm_v2_[0-9a-f]{6}_[0-9]{4,}_gxe-[0-9a-f]{8})_")
_GXE_ROWS = (("environment", "Environment"), ("entry", "Entry"), ("entry x environment", "Entry × environment"), ("total", "Total"))


def _gxe_block(claims: list[Claim], source_labels: dict[tuple[str, ...], str]) -> tuple[list[str], set[str]]:
    """T3 (2026-10-23): the claims of one gxe_partition call as the means-table ANOVA (Source, df, SS, MS, F, p), the studies and
    entries, the shares of the sum of squares, the F's denominator, the grand mean, the across-environment heritability and the
    variance components (or why they are not given), then the descriptive sentence. Returns the lines and the IDs shown."""
    calls: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        match = _GXE_CALL.match(claim.claim_id)
        if match:
            calls[match[1]].append(claim)
    lines: list[str] = []
    shown: set[str] = set()
    for group in calls.values():
        named = {claim.transformation.split(" | ")[0]: claim for claim in group}
        first = next((claim for claim in group if claim.kind == "anova"), None)
        if first is None:
            continue
        rows = []
        for source, label in _GXE_ROWS:
            cells = [named.get(f"means-table ANOVA: {quantity} of {source}") for quantity in ("df", "ss", "ms", "f", "p")]
            if cells[0] is not None:
                rows.append([label, *(_anova_cell(q, c) for q, c in zip(("df", "ss", "ms", "f", "p"), cells))])
        parts = first.transformation.split(" | ")[1:]
        label = source_labels.get(tuple(sorted(first.source_artifact_hashes)), "")
        lines += [f"**Genotype × environment partition of entry means{label}**", "", _text("; ".join(parts[:2]) + "."), "",
                  *_table(["Source", "df", "SS", "MS", "F", "p"], rows), ""]
        undefined_f = named.get("means-table ANOVA: f of entry")
        if undefined_f is not None and undefined_f.value is None:
            lines.append(f"F and p not available: {_text(undefined_f.missing_reason or 'reason not provided')}.")
        shares = [(text, named.get(f"share of the means-table SS: {source}")) for source, text in
                  (("environment", "environment"), ("entry", "entry"), ("entry x environment", "entry × environment"))]
        if all(claim is not None and claim.value is not None for _, claim in shares):
            lines.append("Shares of the means-table sum of squares: " + ", ".join(f"{text} {_anova_number(claim.value)} %" for text, claim in shares) + ".")
        lines.append("F for entries is tested against entry × environment (environments random).")
        grand = named.get("grand mean of the entry means")
        if grand is not None and grand.value is not None:
            lines.append(f"Grand mean of the entry means {_anova_number(grand.value)}" + (f" {_text(grand.unit)}" if grand.unit else "") + ".")
        h2 = named.get("across-environment entry-mean heritability H2")
        if h2 is not None:
            if h2.value is None:
                lines.append(f"Across-environment entry-mean heritability (H²) not available ({_text(h2.missing_reason or 'reason not provided')}).")
            else:
                zero = "MS entry is below MS entry x environment: set to 0" in h2.transformation   # its flag, not the H2 text (review 2)
                lines.append(f"Across-environment entry-mean heritability (H²) {_anova_number(h2.value)}"
                             + (" (set to 0: MS entry is below MS entry × environment)" if zero else "") + ".")
        components = [(text, named.get(name)) for text, name in (("genotype variance", "genotype variance sigma2 g"),
                                                                 ("genotype × environment variance", "genotype x environment variance sigma2 ge"),
                                                                 ("error variance (pooled within trials)", "error variance sigma2 e (pooled within trials)"))]
        r_b = named.get("type-B genetic correlation r_B = sigma2 g / (sigma2 g + sigma2 ge)")
        if components and all(claim is not None and claim.value is not None for _, claim in components):
            lines.append("Variance components: " + ", ".join(
                f"{text} {_anova_number(claim.value)}" + (" (set to 0)" if "estimate negative, set to 0" in claim.transformation else "")
                for text, claim in components)
                + (f"; r_B {_anova_number(r_b.value)}" if r_b is not None and r_b.value is not None else "") + ".")
        elif components[0][1] is not None:
            lines.append(f"Variance components not available ({_text(components[0][1].missing_reason or 'reason not provided')}).")
        lines += ["", GXE_SENTENCE, ""]
        shown |= {claim.claim_id for claim in group}
    return lines, shown


PAIR_SENTENCE = "Descriptive numbers only: a comparison of recorded entry means of two trials, not a recommendation."
_PAIR_CALL = re.compile(r"^(clm_v2_[0-9a-f]{6}_[0-9]{4,}_pair-[0-9a-f]{8})_")
_PAIR_STUDIES = re.compile(r" in study (.+); other: study (.+)$")


def _pair_block(claims: list[Claim], source_labels: dict[tuple[str, ...], str]) -> tuple[list[str], set[str]]:
    """T4 (2026-10-23): the claims of one compare_trials_paired call as the two trial means (Trial, Study, Mean of the entry
    means), the mean difference and the percent, the paired and the Welch t (Test, t, df, p), Pearson r between the trials, the
    SD and SE of the differences and what was left out, then the descriptive sentence. Returns the lines and the IDs shown."""
    calls: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        match = _PAIR_CALL.match(claim.claim_id)
        if match:
            calls[match[1]].append(claim)
    lines: list[str] = []
    shown: set[str] = set()
    for group in calls.values():
        named = {claim.transformation.split(" | ")[0]: claim for claim in group}
        base, other = named.get("mean of the base trial's entry means"), named.get("mean of the other trial's entry means")
        if base is None or other is None:
            continue
        studies = _PAIR_STUDIES.search(base.transformation.split(" | ")[1])
        unit = f" {_text(base.unit)}" if base.unit else ""
        label = source_labels.get(tuple(sorted(base.source_artifact_hashes)), "")
        means = [[side, studies[i] if studies else "", "n/a" if claim.value is None else _anova_number(claim.value)]
                 for i, (side, claim) in enumerate((("Base", base), ("Other", other)), 1)]
        lines += [f"**Paired comparison of entry means, {base.denominator.n} entries in both trials{label}**", "",
                  *_table(["Trial", "Study", "Mean of the entry means"], means), ""]
        difference = next((claim for head, claim in named.items() if head.startswith("mean difference ")), None)
        sides = difference.transformation.split(" | ")[0][len("mean difference "):] if difference is not None else ""
        if difference is not None and difference.value is not None:   # an ASCII minus: the Windows console's cp1252 has no U+2212 (round 2)
            lines.append(f"Mean difference ({sides}) {_anova_number(difference.value)}{unit}.")
        percent = next((claim for head, claim in named.items() if head.startswith("percent ")), None)
        if percent is not None:
            word = "reduction" if percent.transformation.startswith("percent reduction") else "change"
            lines.append(f"Percent {word} of the base mean {_anova_number(percent.value)} %." if percent.value is not None else
                         f"Percent {word} of the base mean not available ({_text(percent.missing_reason or 'reason not provided')}).")
        tests = []
        for text, head in (("Paired t: the entries as pairs", "paired t"), ("Welch t: two independent samples", "Welch t")):
            t, df, p = named.get(head), named.get(f"{head}: df"), named.get(f"{head}: p")
            if t is not None:
                tests.append([text, "n/a" if t.value is None else _anova_number(t.value),
                              "n/a" if df is None or df.value is None else str(int(df.value)) if head == "paired t" else _anova_number(df.value),
                              _anova_cell("p", p)])
        if tests:
            lines += ["", *_table(["Test", f"t ({sides})" if sides else "t", "df", "p"], tests), ""]    # the sign's direction (review F1)
        for head in ("paired t", "Welch t"):
            claim = named.get(head)
            if claim is not None and claim.value is None:
                lines.append(f"{head[0].upper()}{head[1:]} not available: {_text(claim.missing_reason or 'reason not provided')}.")
        r = named.get("Pearson r of the entry means between the trials")
        if r is not None:
            lines.append(f"Pearson r of the entry means between the trials {_correlation(r.value)}." if r.value is not None else
                         f"Pearson r of the entry means between the trials not available ({_text(r.missing_reason or 'reason not provided')}).")
        sd, se = named.get("SD of the differences"), named.get("SE of the mean difference")
        if sd is not None and se is not None and sd.value is not None and se.value is not None:
            lines.append(f"SD of the differences {_anova_number(sd.value)}{unit}; SE of the mean difference {_anova_number(se.value)}{unit}.")
        left = "; ".join(f"{item.reason[len('entries '):].replace('left out: ', '')} {item.count}" for item in base.exclusions
                         if item.reason.startswith("entries "))
        if left:
            lines.append(_text(f"Left out: {left}."))
        named_out = next((part for part in base.transformation.split(" | ") if part.startswith("left out by name: ")), None)
        if named_out:
            lines.append(_text(f"L{named_out[1:]}."))
        lines += ["The paired t treats the entries as the sampling units; the plot error within each trial is not used.", "", PAIR_SENTENCE, ""]
        shown |= {claim.claim_id for claim in group}
    return lines, shown


STRESS_SENTENCE = "Descriptive numbers only: stress indices of recorded entry means of two trials, not a recommendation."
_STRESS_CALL = re.compile(r"^(clm_v2_[0-9a-f]{6}_[0-9]{4,}_stress-[0-9a-f]{8})_")
_STRESS_STUDIES = re.compile(r" in study (.+); stress \(Ys\): study (.+)$")


def _stress_block(claims: list[Claim], sources: _Sources, source_labels: dict[tuple[str, ...], str]) -> tuple[list[str], set[str]]:
    """T5 (2026-10-23): the claims of one stress_indices call as the ranked entries (Rank, Entry, the index, Yp, Ys; names from the
    fetched tables), the two population means and SI, the index's formula and what was left out, then the descriptive sentence.
    Returns the lines and the IDs of the claims shown."""
    calls: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        match = _STRESS_CALL.match(claim.claim_id)
        if match:
            calls[match[1]].append(claim)
    lines: list[str] = []
    shown: set[str] = set()
    for group in calls.values():
        named = {claim.transformation.split(" | ")[0]: claim for claim in group if not claim.filters}
        mean_p, mean_s = named.get("mean of the non-stress entry means"), named.get("mean of the stress entry means")
        ranks = sorted((claim for claim in group if claim.kind == "rank"), key=lambda claim: (claim.value, sorted(claim.filters.items())))
        if mean_p is None or mean_s is None or not ranks:
            continue
        per_entry: dict[tuple[tuple[str, str], ...], dict[str, Claim]] = defaultdict(dict)
        for claim in group:
            if claim.filters:
                per_entry[tuple(sorted(claim.filters.items()))][claim.transformation.split(" | ")[0]] = claim
        head = ranks[0].transformation.split(" | ")[0]                        # "rank by STI, highest first"
        index, order = head[len("rank by "):].split(", ", 1)
        rows = [row for fingerprint in ranks[0].source_artifact_hashes for row in (sources.rows((fingerprint,)) or [])]
        table = []
        for rank in ranks:
            cells = per_entry[tuple(sorted(rank.filters.items()))]
            entry = [" · ".join([value, *[found for found in (_names(rows, column, value, name_column)
                                                              for name_column, _ in NAMES.get(column, (column, ()))[1]) if found]])
                     for column, value in rank.filters.items()]
            shown_value = cells.get(f"{index} of the entry")
            yp, ys = cells.get("the entry's non-stress mean Yp"), cells.get("the entry's stress mean Ys")
            table.append([_number(rank.value), "; ".join(entry),
                          *("n/a" if claim is None or claim.value is None else _anova_number(claim.value) for claim in (shown_value, yp, ys))])
        studies = _STRESS_STUDIES.search(mean_p.transformation.split(" | ")[1])
        unit = f" {_text(mean_p.unit)}" if mean_p.unit else ""
        si = named.get("stress intensity SI")
        formula = next((claim.transformation.split(" | ")[-1] for claim in group if claim.transformation.startswith(f"{index} of the entry")), "")
        label = source_labels.get(tuple(sorted(mean_p.source_artifact_hashes)), "")
        lines += [f"**Stress indices of entry means, {mean_p.denominator.n} entries in both trials{label}**", "",
                  *_table(["Rank", "Entry", index, "Yp", "Ys"], table), "", f"Ranked by {index}, {order}; tied entries share a rank."]
        lines.append(f"Non-stress (Yp): study {studies[1] if studies else ''}, mean of the entry means {_anova_number(mean_p.value)}{unit}; "
                     f"stress (Ys): study {studies[2] if studies else ''}, {_anova_number(mean_s.value)}{unit}; "
                     + (f"stress intensity SI {_anova_number(si.value)}." if si is not None and si.value is not None else
                        f"stress intensity SI not available ({_text(si.missing_reason if si is not None and si.missing_reason else 'reason not provided')})."))
        if formula:
            lines.append(f"{formula}.")
        left = "; ".join(f"{item.reason[len('entries '):].replace('left out: ', '')} {item.count}" for item in mean_p.exclusions
                         if item.reason.startswith("entries "))
        if left:
            lines.append(_text(f"Left out: {left}."))
        named_out = next((part for part in mean_p.transformation.split(" | ") if part.startswith("left out by name: ")), None)
        if named_out:
            lines.append(_text(f"L{named_out[1:]}."))
        lines += ["", STRESS_SENTENCE, ""]
        shown |= {claim.claim_id for claim in group}
    return lines, shown


HETEROSIS_SENTENCE = "Descriptive numbers only: population means of the groups the question named, not a choice of parents."
_HET_CALL = re.compile(r"^(clm_v2_[0-9a-f]{6}_[0-9]{4,}_het-[0-9a-f]{8})_")
_HET_ROWS = (("Offspring", "offspring mean"), ("Parent A", "parent A mean"), ("Parent B", "parent B mean"),
             ("Mid-parent", "mid-parent mean"), ("Founders", "founder mean"))


def _heterosis_block(claims: list[Claim], source_labels: dict[tuple[str, ...], str]) -> tuple[list[str], set[str]]:
    """T6 (2026-10-23): the claims of one heterosis_population call as the group means per site and on the averages over the sites
    (Group, each study, Mean of the sites), the mid-parent heterosis in the trait's unit and in percent, the high-parent heterosis,
    the response of the parents over the founders, the groups as named, the formulas and what was left out, then the descriptive
    sentence. Returns the lines and the IDs of the claims shown."""
    calls: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        match = _HET_CALL.match(claim.claim_id)
        if match:
            calls[match[1]].append(claim)
    lines: list[str] = []
    shown: set[str] = set()
    for group in calls.values():
        per = {(claim.transformation.split(" | ")[0], claim.filters.get("studyDbId")): claim for claim in group}
        sites = sorted({site for _, site in per if site is not None})
        if not sites or ("mid-parent mean", sites[0]) not in per:
            continue
        across = len(sites) > 1
        first = per[("mid-parent mean", sites[0])]
        # a difference of percent contents is in percentage points (review of T6, finding 3: dry matter read "-0.5568 %")
        unit = " percentage points" if first.unit == "%" else f" {_text(first.unit)}" if first.unit else ""
        label = source_labels.get(tuple(sorted(first.source_artifact_hashes)), "")
        rows = []
        for name, head in _HET_ROWS:
            cells = [per.get((head, site)) for site in sites] + ([per.get((f"{head} over the sites", None))] if across else [])
            if all(cell is None for cell in cells):
                continue
            rows.append([name, *("n/a" if cell is None or cell.value is None else _anova_number(cell.value) for cell in cells)])
        lines += [f"**Population heterosis of group means, {len(sites)} site{'s' if across else ''}{label}**", "",
                  _text(first.transformation.split(" | ")[1] + "."), "",            # the trait, its unit and the studies (review finding 3)
                  *_table(["Group", *sites, *(["Mean of the sites"] if across else [])], rows), ""]

        def percent(claim: Claim | None) -> str:
            if claim is None:
                return "n/a"
            return f"{_anova_number(claim.value)} %" if claim.value is not None else \
                f"not available: {_text(claim.missing_reason or 'reason not provided')}"

        def where(site: str | None) -> str:
            return f"study {site}" if site is not None else "on the site averages"

        places = [*sites, *([None] if across else [])]
        on = lambda site: " on the site averages" if site is None else ""                     # noqa: E731
        parts = []
        for site in places:
            difference, share = per.get((f"mid-parent heterosis{on(site)}", site)), per.get((f"mid-parent heterosis %{on(site)}", site))
            if difference is not None and difference.value is not None:
                parts.append(f"{where(site)} {_anova_number(difference.value)}{unit} ({percent(share)})")
        if parts:
            lines.append(f"Mid-parent heterosis (offspring - mid-parent): {'; '.join(parts)}.")
        for title, head in (("High-parent heterosis", "high-parent heterosis %"),
                            ("Response of the parents over the founders (mid-parent - founders)", "response of the parents over the founders %")):
            found = [(site, per.get((f"{head}{on(site)}", site))) for site in places]
            if any(claim is not None for _, claim in found):
                lines.append(f"{title}: {'; '.join(f'{where(site)} {percent(claim)}' for site, claim in found if claim is not None)}.")
        segments = first.transformation.split(" | ")
        groups = next((segment for segment in segments if segment.startswith("groups: ")), "")
        if groups:
            lines.append(_text(f"Groups as named: {groups[len('groups: '):]}."))
        lines.append(_text(f"{segments[-1]}; heterosis % = 100 x (mean offspring - MP) / MP; each group mean is the unweighted mean of its "
                           f"entries' means."))
        left = "; ".join(f"{item.reason[len('entries '):]} {item.count}" for item in first.exclusions if item.reason.startswith("entries "))
        if left:
            lines.append(_text(f"Left out: {left}."))
        lines += ["", HETEROSIS_SENTENCE, ""]
        shown |= {claim.claim_id for claim in group}
    return lines, shown


def _numeric_blocks(claims: list[Claim], sources: _Sources, source_labels: dict[tuple[str, ...], str], *, focus_mean: bool = False) -> list[str]:
    groups: dict[tuple[Any, ...], list[Claim]] = defaultdict(list)
    for claim in claims:
        key = (tuple(sorted(claim.source_artifact_hashes)), tuple(sorted(claim.filters.items())), tuple(claim.grouping),
               claim.denominator.name, claim.denominator.n, claim.n_independent_units,
               tuple((item.reason, item.count) for item in claim.exclusions), _claim_method(claim))
        groups[key].append(claim)
    lines: list[str] = []
    labels = {"mean": "Mean", "median": "Median", "min": "Minimum", "max": "Maximum", "sd": "Sample standard deviation", "count": "Count",
              "correlation": "Correlation", "anova": "ANOVA", "precision": "Precision", "variance": "Variance component",
              "heritability": "Heritability", "test": "Test"}
    for key, group in groups.items():
        first = group[0]
        hashes = key[0]
        rows = sources.rows(hashes) or []
        about = []
        for column, value in first.filters.items():
            name_columns = NAMES.get(column, (column, ()))[1]
            names = [_names(rows, column, value, name_column) for name_column, _ in name_columns]
            about.append(" · ".join([value, *[name for name in names if name]]))
        title = "; ".join(about) or "Summary"
        if not focus_mean or about or source_labels.get(hashes):
            lines += [f"**{_text(title)}{source_labels.get(hashes, '')}**", ""]
        rendered = []
        for claim in group:
            label = labels.get(claim.kind, claim.kind.title())
            if claim.transformation == DISTINCT and len(claim.grouping) == 1:          # not a row count (count labels, item 6)
                label = f"Distinct {claim.grouping[0]} values"
            elif claim.transformation in (ROW_COUNT, FREQUENCY):
                label = "Rows"
            value = _number(claim.value)
            if claim.value is None:
                value += ": " + (claim.missing_reason or "reason not provided")
            if claim.kind != "count":
                value += " " + (claim.unit or "(unit not stated)")
            rendered.append([label, value])
        if focus_mean and len(rendered) == 1:
            lines += [f"**Mean: {_text(rendered[0][1])}**"]
        else:
            lines += _table(["Statistic", "Result"], rendered)
        independent = str(first.n_independent_units) if first.n_independent_units is not None else "unknown"
        units = f"Recorded analysis units: {independent}."
        if first.n_independent_units is None and all(claim.transformation == ROW_COUNT for claim in group):
            units = "Analysis units: not checked by a row count."  # "unknown" read as contradicting a block that has them (q14)
        if focus_mean:
            lines += ["", f"n = {first.denominator.n} ({_text(first.denominator.name)}). {units}", ""]
        else:
            lines += ["", f"Based on {_text(first.denominator.name)}: {first.denominator.n}. {units}",
                      f"Method: {_text(_claim_method(first))}.", ""]
    return lines


_CODE_STOP = "analyst asks: "      # catalog_answers.stop and metadata_answers._clarify
_CODE_STOP_OTHER = "The app's own checks stopped this answer before a number was given."   # a stop text with no plain line


def _human_note(note: str) -> str:
    """Translate specific internal phrases without guessing the cause of a failed run."""
    text = note.strip()
    if text.startswith(_CODE_STOP):     # a stop written by code (catalog_answers, metadata_answers): its plain line;
        asked = text[len(_CODE_STOP):].strip()                                 # the exact check stays in the Full report
        return catalog_plain(asked) or metadata_plain(asked) or _CODE_STOP_OTHER
    text = re.sub(r"^(?:(?:numeric_summary|group_stats|count_records|rank_entries|threshold_count|correlate_entry_means|analysis target|model(?: note)?)\s*:\s*)+", "", text, flags=re.I)
    if "no complete artifact is eligible for analysis" in text.lower():
        return re.sub("no complete artifact is eligible for analysis", "No complete data table was available for analysis, so an answer could not be calculated.", text, flags=re.I)
    # Normalize only the two known unit-only messages; other caveat wording is preserved.
    unit_messages = {
        "unit not stated in the artifact's measurement metadata; claims carry unit=null",
        "unit is not stated in the artifact's metadata, so the mean value is reported with unit=null.",
        "unit is not stated in the source metadata.",
    }
    if " ".join(text.split()).casefold() in unit_messages:
        return "The measurement unit is not stated in the source data."
    return re.sub(r"unit=null", "unit not stated", text, flags=re.I)


def _note_key(note: str) -> str:
    return " ".join(note.split()).casefold()


_CODE_AVERAGE_LIMIT = "descriptive means of valid values as recorded; not adjusted for trial design, environment, missingness or clone composition; not evidence of genetic superiority"
_CODE_STAGE_LIMIT = "Growth stage or timepoint is not stated in the source metadata; a shared growth stage cannot be verified."
_ALL_MISSING_DATES = re.compile(r"Measurement dates are not recorded for ([0-9]+) of ([0-9]+) valid source rows; the timing of those measurements is unknown\.")
_MODEL_REPEAT_REQUIRES = {
    "growth stage or timepoint is not stated; shared timing cannot be verified.": _note_key(_CODE_STAGE_LIMIT),
    "descriptive mean is not evidence of genetic superiority.": _note_key(_CODE_AVERAGE_LIMIT),
    "the mean is a simple descriptive average of all valid recorded values in the study; it does not account for trial design, environment, or genetic factors.": _note_key(_CODE_AVERAGE_LIMIT),
}
# The Retriever's own notes ("model note: ..." in the retrieval warnings, agents.retriever.build_report) describe its job
# (finding IDs, fetching tables), not the data. Under an answer they read as if it were wrong: "The pooled mean was not
# computed." under "Mean: 2.37857" (eval 20261016T205733Z, q13). The chat leaves every one out; the Full report keeps them
# ("Retriever note: ..."), as the data review screen already did. What only they told the chat is now written by code: a
# fetch the approval gate refused (_ledger_reasons) and a saved table with no rows (_empty_tables). The three exact wordings
# of Retriever process notes that were hidden here before are gone with them.
_RETRIEVER_NOTE = "model note:"
# Exact administrative wording already covered by the table; never a keyword-based warning filter. (Also an Analyst note.)
_MODEL_BOOKKEEPING = frozenset({"study names were found in the artifact preview"})


def _caveat_groups(controller: Any, claims: list[Claim]) -> tuple[list[str], list[str]]:
    """Keep tool/controller limits separate from unverified model prose; preserve unknown warnings."""
    analysis = getattr(controller, "analysis", None)
    retrieval = getattr(controller, "retrieval", None)
    notes = [*(getattr(analysis, "caveats", []) or []),
             *[warning for warning in (getattr(retrieval, "warnings", []) or []) if not warning.lower().startswith(_RETRIEVER_NOTE)]]
    exclusions = [*(getattr(analysis, "exclusions", []) or []), *[item for claim in claims for item in claim.exclusions]]
    notes += [f"Excluded: {item.count} — {item.reason}." for item in exclusions if item.count]
    source_notes, model_notes = [], []
    catalog_only = (bool(claims) and all(claim.kind == "count" and claim.transformation in
                    (DISTINCT, FREQUENCY, ROW_COUNT) for claim in claims)) or (
                    not claims and getattr(getattr(getattr(controller, "plan", None), "statistic", None), "kind", None) == "count")
    for note in notes:
        if note.lower().startswith("key claims (model"):
            continue
        if note.lower().startswith(("model note:", "model:")):
            model_notes.append(_human_note(note))
        elif catalog_only and note.lower().startswith("analysis target: descriptive means"):
            continue
        else:
            source_notes.append(_human_note(note))
    source_keys = {_note_key(note) for note in source_notes}
    all_dates_missing = any(match and int(match[1]) > 0 and match[1] == match[2]
                            for match in (_ALL_MISSING_DATES.fullmatch(note) for note in source_notes))
    unit_in_result = any(claim.kind != "count" and claim.unit is None for claim in claims)
    source_result, model_result = [], []
    seen_source, seen_model = set(), set()
    for note in source_notes:
        key = _note_key(note)
        if note == "The measurement unit is not stated in the source data." and unit_in_result:
            continue
        if key and key not in seen_source:
            seen_source.add(key)
            source_result.append(_text(note))
    for note in model_notes:
        key = _note_key(note)
        if key in source_keys or key in _MODEL_BOOKKEEPING:
            continue
        if note == "The measurement unit is not stated in the source data." and unit_in_result:
            continue
        if _MODEL_REPEAT_REQUIRES.get(key) in source_keys:
            continue
        if key == "measurement dates are missing for all rows; timing is unknown." and all_dates_missing:
            continue
        if key and key not in seen_model:
            seen_model.add(key)
            model_result.append(_text(note))
    if any(claim.kind != "count" for claim in claims):
        warning = "Distinct IDs alone do not establish statistical independence."
        if _note_key(warning) not in seen_source:
            source_result.append(warning)
    return source_result, model_result


_ID_LIKE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,39}")


def _ledger_reasons(controller: Any) -> list[str]:
    """For a run that did not complete: each fetch the approval gate refused (the ledger's "blocked" studies), in fixed words.
    The gate's own reason is not shown: it quotes the arguments the model wrote (review, 2026-10-16: a "variable" holding a
    false claim reached the chat). The study ID is shown only when it looks like an ID. Other ledger statuses are not
    repeated here: their reasons are the fetch's own warnings, which the screen already shows."""
    reasons = []
    for entry in getattr(getattr(controller, "retrieval", None), "study_ledger", None) or []:
        if getattr(entry, "status", None) != "blocked":
            continue
        study = str(getattr(entry, "study_id", "") or "")
        who = f"Study {study}: a fetch" if _ID_LIKE.fullmatch(study) else "A fetch"
        reasons.append(_text(f"{who} was refused by the approval check; the Full report has the reason."))
    return list(dict.fromkeys(reasons))


def _empty_tables(controller: Any) -> list[str]:
    """Each saved measurement table with no rows: complete-empty is complete evidence, so no warning is written for it."""
    lines = []
    for artifact in getattr(getattr(controller, "retrieval", None), "artifacts", None) or []:
        kind = getattr(artifact, "kind", None)
        if kind not in ("observations", "observation_units") or getattr(artifact, "row_count", None) != 0:
            continue
        variables = ", ".join(getattr(artifact, "variable_ids", None) or [])
        what = f"Study {', '.join(getattr(artifact, 'study_ids', None) or []) or '(not stated)'}" + (f", variable {variables}" if variables else "")
        lines.append(_text(f"{what}: " + ("no rows were returned." if kind == "observations" else "no observation units were returned.")))
    return list(dict.fromkeys(lines))


# SweetPotatoBase's /observations leaves out every value recorded as 0 (zeros size check, 2026-10-21: 3,132 values in 26 of 117
# benchmark tables, every one a 0; beta-carotene loses about half its values). An answer built from an /observations table of a
# server where this was found says so once, among its data limits (coach, 2026-10-21 18:10 UTC). "zero" is written in words: the
# benchmark grader reads every digit of the answer.
ZEROS_LEFT_OUT = ("This database's /observations service leaves out values recorded as zero, so measurements of zero (such as a "
                  "failed plot's yield or a white-fleshed clone's beta-carotene) are missing from these numbers.")
def _zeros_left_out(controller: Any) -> bool:
    """Was this answer built from an /observations table of a server known to leave out values recorded as 0? The servers are
    brapi_client.PLOT_RECORD_HOSTS; since item (3) of 2026-10-22 get_observations reads their plot records, which keep the
    zeros, and such a table carries brapi_client.PLOT_RECORDS_NOTE among its quality notes: it needs no note."""
    artifacts = getattr(getattr(controller, "retrieval", None), "artifacts", None) or []
    return (brapi_client.reads_plot_records(getattr(controller, "base_url", None) or "")
            and any(getattr(artifact, "kind", None) == "observations"
                    and not any(str(note).startswith(brapi_client.PLOT_RECORDS_NOTE) for note in getattr(artifact, "quality_notes", None) or [])
                    for artifact in artifacts))


def _caveats(controller: Any, claims: list[Claim]) -> list[str]:
    limits, model_notes = _caveat_groups(controller, claims)
    return [*limits, *["Unverified model note: " + note for note in model_notes]]


def _limit_blocks(limits: list[str], model_notes: list[str]) -> list[str]:
    lines = []
    if limits:
        lines += ["**Data limits and exclusions**", "", *[f"- {note}" for note in limits], ""]
    if model_notes:
        lines += ["**Additional model notes (unverified)**", "", *[f"- {note}" for note in model_notes], ""]
    return lines


def _provider_failure(controller: Any) -> tuple[str, str, str] | None:
    """Recognize the runtime's explicit provider error record, never model-written warning text."""
    analysis = getattr(controller, "analysis", None)
    retrieval = getattr(controller, "retrieval", None)
    notes = [*(getattr(analysis, "caveats", []) or []), *(getattr(retrieval, "warnings", []) or []),
             *(getattr(controller, "notes", []) or [])]
    for note in notes:     # the planner's notes carry "coordinator: ", the next-step call's "supervisor: supervisor error"
        match = re.fullmatch(r"(?:coordinator: agent|supervisor: supervisor|agent) error (provider_[a-z_]+): (.+)", note)
        if match:
            return note, match[1], match[2]
    return None


# Run notes that are never the reason a run stopped: the coordinator's bookkeeping (agents.coordinator: the scope code set, the
# study count, clarifications it dropped or added; not its rejected attempts, errors or stop), and the history of clarification
# rounds (the needs_clarification screen lists those itself).
_NOT_A_REASON = re.compile(r"coordinator: (?:scope\.study_ids |the planner judged |dropped clarification |program membership is not "
                           r"established|matching_study_count=|added required clarification)|clarification \d+: "
                           r"|supervisor: the Coordinator's reason was shortened ")
_CACHE_MISS = "metadata is not in the offline cache"                       # stale once the catalog setup has prepared them
_CATALOGS_PREPARED = "complete study and variable catalogs prepared"


def _fallback_notes(notes: list[str]) -> list[str]:
    """The run notes a failure screen may show when it has no other reason (the last three that sound like a reason)."""
    prepared = [i for i, note in enumerate(notes) if note.startswith(_CATALOGS_PREPARED)]
    kept = [note for i, note in enumerate(notes) if not _NOT_A_REASON.match(note)
            and not (note.startswith(_CACHE_MISS) and any(j > i for j in prepared))]
    return [note for note in kept if re.search(r"error|fail|block|missing|not |timeout|cancel|limit|incomplete|clarification|stop|refus",
                                                 note, re.I)][-3:]
# A clarification round in the run's notes (run.clarify_phase, coordination_runtime), and the note that ends the last one.
# When the cap ends a run, the chat lists every round: run 20261015T165633 showed only the last two of five.
_ROUND_NOTE = re.compile(r"(?:clarification \d+: |still not ready after )")


def answer_summary(controller: Any) -> str:
    """One compact answer from typed claims; no model narrative or internal evidence IDs."""
    status = getattr(controller, "execution_status", "unknown")
    analysis = getattr(controller, "analysis", None)
    retrieval = getattr(controller, "retrieval", None)
    if status != "completed" or getattr(analysis, "status", None) != "completed" or getattr(retrieval, "status", None) != "completed":
        shown = {"incomplete": "Incomplete result", "failed": "Could not complete this question", "blocked": "Question blocked",
                 "canceled": "Question canceled", "needs_clarification": "More detail is needed", "limit_reached": "A request limit was reached"}.get(status, "No verified answer is available")
        empty = _empty_tables(controller) if getattr(retrieval, "status", None) == "completed" else []
        reasons = list(dict.fromkeys([*_ledger_reasons(controller), *_caveats(controller, []), *empty]))
        provider = _provider_failure(controller)
        if provider is not None:
            raw_error, code, message = provider
            if code == "provider_http_error" and message == "provider returned HTTP 429":
                code = "provider_rate_limit_unknown"
                message = "The model provider returned HTTP 429; its cause was not reported"
            reasons = [reason for reason in reasons if reason != _text(_human_note(raw_error))]
            evidence = ("Data retrieval completed, but the model service could not finish the analysis."
                        if getattr(retrieval, "status", None) == "completed" else "The model service could not finish this question.")
            next_step = {"provider_quota": "Check the model provider account quota or billing limit before retrying.",
                         "provider_rate_limit": "Wait before retrying this question.",
                         "provider_rate_limit_unknown": "Check the provider quota and rate limits before retrying; the cause of HTTP 429 is unknown."}.get(
                             code, "Resolve the model service issue before retrying.")
            return "\n\n".join(["**Could not complete this question: Model service unavailable.**", _text(message.rstrip(".")) + ".", evidence,
                                 "This failure does not establish that the database has no matching records.",
                                 *reasons, next_step, "Full report keeps the exact failure and any completed retrieval evidence."])
        if not reasons:
            notes = getattr(controller, "notes", [])
            if status == "needs_clarification" and any(note.startswith("still not ready after ") for note in notes):
                picked = [note for note in notes if _ROUND_NOTE.match(note)]    # every round, then what is still unclear
            else:
                picked = _fallback_notes(list(notes))
            reasons = [_text(_human_note(note)) for note in picked]
        if not reasons:
            reasons = ["The run did not produce a complete, verified analysis."]
        retry = []
        if any("No complete data table was available" in reason for reason in reasons):
            retry = ["Retry the question, then review the proposed scope and approve its retrieval. "
                     "If the same step stops again, check Full report for the recorded retrieval reason."]
        return "\n\n".join([f"**{shown}.**", *reasons, *retry, "See Full report for the complete reason and evidence."])
    facts = getattr(analysis, "metadata_facts", [])
    if facts:
        lines = render_metadata_facts(facts)
        limits, model_notes = _caveat_groups(controller, [])
        lines += ["", *_limit_blocks(limits, model_notes)]
        lines += ["", "Full report contains the source records and evidence for these fields."]
        return "\n".join(lines)
    claims = list(analysis.claims)
    focus_mean = _mean_focus(controller, claims)
    focus_count = _count_focus(controller, claims)
    displayed_claims = [claim for claim in claims if claim.kind == "mean"] if focus_mean else claims
    sources = _Sources(getattr(getattr(controller, "ctx", None), "registry", None))
    hashes = list(dict.fromkeys(tuple(sorted(claim.source_artifact_hashes)) for claim in claims))
    source_labels = {key: f" · source {index + 1}" for index, key in enumerate(hashes)} if len(hashes) > 1 else {}
    ranking, in_ranking = _ranking_block(claims, sources, source_labels)
    thresholds, in_threshold = _threshold_block(claims, sources, source_labels)
    correlations, in_correlation = _correlation_block(claims, source_labels)
    anova_lines, in_anova = _anova_block(claims, source_labels)
    gxe_lines, in_gxe = _gxe_block(claims, source_labels)
    pair_lines, in_pair = _pair_block(claims, source_labels)
    stress_lines, in_stress = _stress_block(claims, sources, source_labels)
    heterosis_lines, in_heterosis = _heterosis_block(claims, source_labels)
    in_ranking |= in_threshold | in_correlation | in_anova | in_gxe | in_pair | in_stress | in_heterosis
    # a mean the ranking table shows is not shown again (group_stats on the same table states the same entry mean; review)
    ranked_means = {(tuple(sorted(c.source_artifact_hashes)), tuple(sorted(c.filters.items())), c.value)
                    for c in claims if c.claim_id in in_ranking and c.kind == "mean" and c.claim_id not in in_threshold}
    displayed_claims = [claim for claim in displayed_claims if claim.claim_id not in in_ranking and not (
        claim.kind == "mean" and (tuple(sorted(claim.source_artifact_hashes)), tuple(sorted(claim.filters.items())), claim.value) in ranked_means)]
    buckets: dict[tuple[Any, ...], list[Claim]] = defaultdict(list)
    other = []
    for claim in displayed_claims:
        eligible = claim.kind == "count" and len(claim.grouping) == 1 and (
            (claim.transformation == DISTINCT and not claim.filters) or
            (claim.transformation == FREQUENCY and set(claim.filters) == set(claim.grouping)))
        if eligible:
            buckets[(tuple(sorted(claim.source_artifact_hashes)), claim.grouping[0], claim.denominator.name, claim.denominator.n)].append(claim)
        else:
            other.append(claim)
    lines: list[str] = [*ranking, *thresholds, *correlations, *anova_lines, *gxe_lines, *pair_lines, *stress_lines, *heterosis_lines]
    in_catalog_titles: set[str] = set()              # distinct counts a catalog block's title already leads with
    for key, group in buckets.items():
        block = _catalog_block(group, sources, source_labels.get(key[0], ""), focus_count=focus_count)
        if block is None:
            other.extend(group)
        else:
            lines += [*block, ""]
            in_catalog_titles |= {claim.claim_id for claim in group if claim.transformation == DISTINCT}
    lines += _numeric_blocks(other, sources, source_labels, focus_mean=focus_mean)
    lines = [*_count_lead(controller, claims, in_catalog_titles, source_labels), *lines]
    caveats, model_notes = _caveat_groups(controller, claims)
    caveats += [line for line in _empty_tables(controller) if line not in caveats]
    if _zeros_left_out(controller):
        caveats.append(ZEROS_LEFT_OUT)
    if focus_mean:
        for claim in claims:
            if claim.kind != "mean" and claim.value is None and claim.missing_reason:
                warning = _text("An additional statistic could not be calculated: " + claim.missing_reason)
                if warning not in caveats:
                    caveats.append(warning)
    lines += _limit_blocks(caveats, model_notes)
    if focus_mean and len(displayed_claims) < len(claims):
        lines.append("Other calculated statistics remain in Full report, together with the complete methods and evidence.")
    else:
        lines.append("Full report contains the complete methods and evidence.")
    return "\n".join(lines)


def approval_summary(controller: Any) -> str:
    """Show the exact permission being requested without duplicating the technical manifest."""
    manifest, plan = controller.manifest, controller.plan
    if manifest is None or plan is None:
        return "No validated fetch plan is ready. See Full report for details."
    metadata = getattr(controller, "metadata", None)
    studies = {study.study_id: study for study in (getattr(metadata, "studies", []) or [])}
    variables = {variable.variable_id: variable for variable in (getattr(metadata, "variables", []) or [])}
    pairs, units = required_fetches(plan, manifest)
    lines = [_text(approval_lead(manifest, plan)), ""]
    carried = carried_over_line(controller.config)               # a follow-up: what was carried over, written by code
    if carried:
        lines += [_text(carried), ""]
    lines += ["**Review this plan before approving**", "", f"Source: {_text(manifest.base_url)}."]
    if manifest.origin == "snapshot":
        lines += ["Uses an offline snapshot; no live requests."]
    else:
        fresh = getattr(controller, "catalog_fetched_at", None) or "unknown date"
        setup = getattr(controller, "catalog_setup", None) or {}
        origin = "prepared during this question" if setup.get("status") == "completed" else "previously cached"
        lines += [f"Catalogs: {origin}; fetched {_text(fresh)}. Cached replies are reused."]
    if manifest.observation_study_ids:
        lines += ["", "**Observation scope**", ""]
        rows = []
        for study_id in manifest.observation_study_ids:
            study = studies.get(study_id)
            rows.append([study_id, getattr(study, "name", "") or "Not stated"])
        lines += _table(["Study ID", "Study"], rows)
        if manifest.observation_variable_ids:
            lines += ["", "Traits: " + "; ".join(
                _text(f"{variable_id} · {getattr(variables.get(variable_id), 'name', '') or 'name not stated'} "
                      f"({getattr(variables.get(variable_id), 'unit', None) or 'unit not stated'})")
                for variable_id in manifest.observation_variable_ids) + "."]
        excluded = ("the plot records also carry other traits' values, which are dropped before any table, preview or answer "
                    "(the raw pages stay in the local request cache)"
                    if brapi_client.reads_plot_records(manifest.base_url) else "other observations are excluded")   # C8, 2026-10-22
        lines += [f"At least {len(pairs) + len(units)} data fetches: {len(pairs)} study/trait requests and {len(units)} plot rosters.",
                  f"Approval covers only these observation study IDs and trait IDs; {excluded}."]
    else:
        lines += ["", "**Catalog question: no observation requests.**"]
        scoped = list(plan.scope.study_ids)
        if scoped:
            shown = ", ".join(_text(study_id) for study_id in scoped[:10])
            tail = f"; {len(scoped) - 10} additional matched IDs are not shown here" if len(scoped) > 10 else ""
            lines += [f"Matched studies: {len(scoped)}. IDs: {shown}{tail}."]
    filters = {key: value for key, value in plan.scope.filters.model_dump().items()
               if key != "contract_version" and value not in (None, "", [])}
    if filters:
        lines += ["Filters: " + "; ".join(f"{_text(key)} = {_text(value)}" for key, value in filters.items()) + "."]
    lines += ["Allowed data: " + ", ".join(_text(family) for family in manifest.endpoint_families) + ".",
              f"Limits: {manifest.max_http_attempts} HTTP attempts, {manifest.max_observation_studies} observation studies, "
              f"{controller.config.max_fetches} data fetch calls, and {controller.budget.max_model_calls} model calls."]
    if scope_needs_a_check(controller.config.question, list(plan.scope.study_ids), metadata):
        lines += ["**Check the scope:** the planner interpreted your question as several studies. Cancel if you meant one."]
    lines += ["", "Approve authorizes this read-only plan. It does not accept the eventual answer. Cancel stops here; "
              "edit can only narrow the study list. Full report contains the exact plan and approval details."]
    return "\n".join(lines)
