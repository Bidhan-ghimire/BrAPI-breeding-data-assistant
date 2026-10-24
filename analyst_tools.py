"""
analyst_tools.py — the FIXED calculation tools for the Analyst.

Plain-words summary:
* The Analyst never writes Python. It picks one of these functions and passes a
  registered artifact handle (art_xxxxxx_0001). The function reads the table through
  artifacts.py (hash-checked, path-safe), computes with plain arithmetic, and hands back
  typed claims plus the counts behind them. No HTTP, no eval/exec, no subprocess, no model.
* Every cell is TEXT until this module classifies it. A value token is one of three things:
    valid    — a plain decimal number ('10', '14.5', '-3', '2e3') that is finite
    missing  — an empty cell or an agreed missing token ('', 'NA', 'N/A', 'null', 'None', '.')
    invalid  — anything else, including 'bad', 'inf', '-inf', 'nan', '1e400' (overflows)
  Missing and invalid are counted, never turned into zero, and the original tokens survive.
* n needs a name. n_valid_values (how many numbers were averaged) is reported separately from
  n_independent_units (how many distinct plots/plants they came from). When the table cannot
  establish independence, the tool says unknown rather than calling rows "plots".
* Repeated measurements of one unit (same unit, variable, timepoint) or repeated observation
  IDs are detected. The default is to refuse; the caller must name an aggregation rule
  (duplicate_policy='mean_per_unit') and the result says that it was applied.
* Full precision is kept for sorting and for the `value` fields; `display` strings are rounded.
  Ties are reported, and ordering inside a tie is by group ID, so a rerun gives the same rank.
* An incomplete artifact (the source collection was not proven complete) yields no study-level
  claim: numeric tools refuse it unless allow_incomplete=True, and then every claim carries
  the caveat in its transformation text.
* Full results (every group, including excluded ones) are saved as analysis_result artifacts
  with their lineage in the manifest notes; a message only ever shows a preview.

Everyday example: a pocket calculator with a few labelled buttons, glued to a filing cabinet.
You may press a button and name a drawer; you may not rewire the calculator, open other
cabinets, or dictate the answer.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Iterable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from artifacts import ArtifactError, ArtifactRegistry
from contracts import ArtifactId, ArtifactManifest, Claim, Denominator, Exclusion, ToolError, ToolResult, unknown_metadata
import pvalues

__all__ = [
    "MISSING_TOKENS", "PLOTTING_ENABLED", "AnalystError", "classify_value", "ANALYST_TOOLS", "ANALYST_TOOL_NAMES",
    "analyst_tool_schemas", "dispatch_analyst", "AnalystTools",
]

# -- the value policy: explicit, visible in every numeric result -------------------------------
MISSING_TOKENS = frozenset({"", "NA", "N/A", "NULL", "null", "None", "."})
_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")     # plain decimal numbers only; no '1_000', no '0x10'
VALUE_POLICY = {
    "missing_tokens": sorted(MISSING_TOKENS),
    "valid": "a plain decimal number (optional sign, digits, one decimal point, optional exponent) that is finite",
    "invalid": "every other token, including inf, -inf, nan and numbers that overflow to infinity",
    "ids": "IDs are exact text: '007' and '7' are different; only an empty cell is a missing ID",
}
PLOTTING_ENABLED = False          # matplotlib is not an approved dependency; plot_bar reports unavailable
DISPLAY_DECIMALS = 4

# column conventions of the observation tables written by brapi_mcp_server
VALUE_COL, UNIT_COL, OBS_ID_COL, VARIABLE_COL, TIMEPOINT_COL = (
    "value", "observationUnitDbId", "observationDbId", "observationVariableDbId", "observationTimeStamp")
LABEL_FOR = {"germplasmDbId": "germplasmName", "observationUnitDbId": "observationUnitName", "studyDbId": "studyName",
             "locationDbId": "locationName", "programDbId": "programName", "seasonDbId": "seasonName",
             "observationVariableDbId": "observationVariableName"}
MISSING_GROUP = "(missing group id)"
_HANDLE = re.compile(r"^art_[0-9a-f]{6}_[0-9]{4,}$")               # same shape artifacts.py issues
ANALYSIS_TARGET = ("descriptive means of valid values as recorded; not adjusted for trial design, environment, "
                   "missingness or clone composition; not evidence of genetic superiority")


class AnalystError(Exception):
    """A refusal with a fixed code (invalid_argument, not_found, incomplete_data, unsupported)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# --------------------------------------------------------------------------
# Reading and classifying
# --------------------------------------------------------------------------

def classify_value(token: str | None) -> tuple[Literal["valid", "missing", "invalid"], float | None]:
    """One text token -> (class, number-or-None). Never guesses; the original token is left to the caller."""
    if token is None:
        return "missing", None
    text = token.strip()
    if text in MISSING_TOKENS:
        return "missing", None
    if not _NUMBER.match(text):
        return "invalid", None
    number = float(text)
    if not math.isfinite(number):
        return "invalid", None
    return "valid", number


@dataclass
class Table:
    manifest: ArtifactManifest
    rows: list[dict[str, str]]

    @property
    def columns(self) -> list[str]:
        return list(self.manifest.columns)


def _load(registry: ArtifactRegistry, artifact_id: Any) -> Table:
    if not isinstance(artifact_id, str) or not _HANDLE.match(artifact_id):
        raise AnalystError("invalid_argument", f"{artifact_id!r} is not an artifact handle (expected art_xxxxxx_0001); paths are not accepted")
    if artifact_id not in registry.artifact_ids():
        raise AnalystError("not_found", f"{artifact_id} is not registered in this run")
    manifest, rows = registry.load(artifact_id)
    return Table(manifest, rows)


def _require_column(table: Table, column: str, what: str = "column") -> None:
    if column not in table.manifest.columns:
        raise AnalystError("invalid_argument", f"{what} {column!r} is not a column of {table.manifest.artifact_id}; columns: {table.columns}")


def _single_variable(table: Table) -> str:
    """The one variable this table measures, or a refusal. Mixed variables are never averaged."""
    ids = set(table.manifest.variable_ids)
    if VARIABLE_COL in table.manifest.columns:
        ids |= {r.get(VARIABLE_COL, "") for r in table.rows if r.get(VARIABLE_COL, "")}
    if len(ids) > 1:
        raise AnalystError("unsupported", f"table holds more than one variable {sorted(ids)}; numeric tools work on exactly one")
    if not ids:
        raise AnalystError("unsupported", "cannot establish which variable the table measures (no variable_ids and no observationVariableDbId column)")
    return next(iter(ids))


def _display(number: float | None) -> str | None:
    if number is None:
        return None
    text = f"{number:.{DISPLAY_DECIMALS}f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _one_decimal(number: float | None) -> str | None:
    """A percent to one decimal, never "-0.0" (compare_means; coach 2026-10-22 03:48 UTC, round 3 (C))."""
    if number is None:
        return None
    text = f"{number:.1f}"
    return "0.0" if text == "-0.0" else text


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "sd": None}
    return {"mean": statistics.fmean(values), "median": statistics.median(values), "min": min(values), "max": max(values),
            "sd": statistics.stdev(values) if len(values) >= 2 else None}


CV_OF_VALUES = "cv of valid values"          # numeric_summary's CV (Breedbase's Phenotype Summary CV), not anova_trial's error CV
PERCENT_MISSING = "percent missing"           # numeric_summary's share of the plot list's plots without a valid value


def _cv(stats: dict[str, float | None]) -> tuple[float | None, str | None]:
    """CV = 100 x sample SD / mean of the same values (Breedbase's Phenotype Summary CV), or None with the reason."""
    if stats["mean"] is None:
        return None, "no valid numeric values"
    if stats["sd"] is None:
        return None, "fewer than 2 valid values: sample SD undefined"
    if stats["mean"] <= 0:
        return None, "the mean is not positive, so SD / mean is not a CV"
    return 100 * stats["sd"] / stats["mean"], None


def _claim_component(part: str) -> str:
    # Plain components have no separator and may not occupy the encoding namespace.
    # UTF-8 hex is injective; punctuation, Unicode, underscores and empty text stay distinct.
    if re.fullmatch(r"[A-Za-z0-9-]+", part) and not part.startswith("x0"):
        return part
    return "x0" + part.encode("utf-8").hex()


def _claim_id(artifact_id: str, kind: str, *parts: str, namespace: str | None = None) -> str:
    tail = "_".join(_claim_component(p) for p in parts)
    # v2 cannot be a registry's six-hex prefix, so this namespace cannot overlap legacy IDs.
    head = (f"clm_v2_{artifact_id[4:]}_{_claim_component(namespace)}_{kind}" if namespace is not None
            else f"clm_{artifact_id[4:]}_{kind}")
    return head + (f"_{tail}" if tail else "")


def _duplicate_key(row: Mapping[str, str], columns: list[str]) -> tuple[str, ...] | None:
    unit = row.get(UNIT_COL, "")
    if not unit:
        return None
    return (unit, row.get(VARIABLE_COL, "") if VARIABLE_COL in columns else "", row.get(TIMEPOINT_COL, "") if TIMEPOINT_COL in columns else "")


@dataclass
class Classified:
    """Rows of one table sorted into valid / missing / invalid, with independence and duplicate facts."""

    valid: list[tuple[dict[str, str], float]]
    n_missing: int
    n_invalid: int
    n_rows: int
    units_known: bool                    # every valid row names an observation unit
    repeated_observation_ids: int        # extra copies of an observationDbId
    repeated_measurements: int           # extra rows for a (unit, variable, timepoint)
    duplicate_note: str | None

    @property
    def n_valid(self) -> int:
        return len(self.valid)

    @property
    def n_independent_units(self) -> int | None:
        if not self.units_known:
            return None
        return len({r[UNIT_COL] for r, _ in self.valid})


def _classify(table: Table, value_col: str) -> Classified:
    valid: list[tuple[dict[str, str], float]] = []
    n_missing = n_invalid = 0
    for row in table.rows:
        kind, number = classify_value(row.get(value_col))
        if kind == "valid":
            valid.append((row, number))      # type: ignore[arg-type]
        elif kind == "missing":
            n_missing += 1
        else:
            n_invalid += 1
    has_units = UNIT_COL in table.manifest.columns
    units_known = has_units and all(r.get(UNIT_COL, "") for r, _ in valid)
    rep_obs = 0
    if OBS_ID_COL in table.manifest.columns:
        counts = Counter(r.get(OBS_ID_COL, "") for r in table.rows if r.get(OBS_ID_COL, ""))
        rep_obs = sum(c - 1 for c in counts.values() if c > 1)
    rep_meas = 0
    if has_units:
        keys = Counter(k for k in (_duplicate_key(r, table.columns) for r in table.rows) if k is not None)
        rep_meas = sum(c - 1 for c in keys.values() if c > 1)
    note = None
    if rep_obs or rep_meas:
        note = (f"{rep_obs} repeated observation ID(s) and {rep_meas} repeated measurement(s) per unit/variable/timepoint; "
                "duplicate_policy='mean_per_unit' averages each unit first, duplicate_policy='reject' refuses")
    return Classified(valid, n_missing, n_invalid, len(table.rows), units_known, rep_obs, rep_meas, note)


ROWS_NOTE = "rows: "    # what one row of a derived table is (coach 2026-10-24 12:50 UTC, GAP 1): group_stats, rank_entries and
                        # threshold_count write it (15:49 UTC), filter_rows and concat_tables carry it, _row_unit reads it
GROUP_VALUE_WORDS = {"mean": "means", "median": "medians", "min": "minima", "max": "maxima", "sd": "SDs"}
NAMED_ENTRIES = 10      # GAP 2: an exclusion names up to ten entries; above that the first ten and "and N more"
NAME_CHARS = 80         # ... each name cut to 80 characters (review: a 400-character name made every reason 450 long)
_GROUP_ROWS = re.compile(r"(?:group_stats|rank_entries) by ([^,\s]+)(?:, basis (pooled|study_means))?(?:, from (\d+) stacked tables)?")
_STACKED_NOTE = re.compile(r"(.+), from (\d+) stacked tables")      # another rows note of a stack (threshold_count's; review 2)


def _in_stacked(table: Table, stacked: str) -> str:
    """Which of the N stacked tables a stack's rows are in, read from every row's source_artifact_id, not only the rows with a
    valid value (coach 2026-10-24 15:49 UTC; reviews 1 and 2 of 2026-10-24: a filter keeps the rows of one source, of some, or
    of none)."""
    sources = {row.get("source_artifact_id", "") for row in table.rows}
    named = sorted(sources - {""})
    if "" in sources:                                # a row naming no source table: which tables the rows are in is unknown
        return "in each stacked table; some rows name no source table"
    if not named:                                    # a filter that keeps no row (review 2)
        return f"in none of the {stacked} stacked tables: no row is left"
    if len(named) == 1:                              # a stack filtered down to one source (coach 2026-10-24 15:49 UTC)
        return f"in the one table from {named[0]}"
    if len(named) < int(stacked):                    # ... or to some of them (review 1)
        return f"in each of {len(named)} of the {stacked} stacked tables"
    return f"in each of the {stacked} stacked tables"


def _row_unit(table: Table, value: str | None = None, rows: list[dict[str, str]] | None = None) -> str:
    """What one row of the table is, for the transformation text (Luna A42.4: a mean of 315 genotype means read "one row per
    observation unit", and the coordinator stopped it as a mean over plot rows). A group_stats or rank_entries table, also
    filtered: "one row per <by>; values are group means" (medians, ... by the value column; on rank_entries' basis study_means
    a mean is the unweighted mean of the group's per-study means and the rest are pooled: review 2026-10-24); stacked from
    several such tables: "one row per <by> in each of the N stacked tables" (review: two studies' tables of the same clones are
    6 rows of 3 IDs), or which of them its rows are in (_in_stacked); another note (threshold_count's) as written, a stack's
    with which of its tables; rows that are units: "one row per observation unit", when no unit has several of the rows used
    (review: one plot measured twice); any other table: not recorded, never units."""
    noted = list(dict.fromkeys(n[len(ROWS_NOTE):] for n in table.manifest.quality_notes if n.startswith(ROWS_NOTE)))
    group = _GROUP_ROWS.fullmatch(noted[0]) if len(noted) == 1 else None
    if group:
        by, basis, stacked, word = group[1], group[2], group[3], GROUP_VALUE_WORDS.get(value or "")
        if basis == "study_means":                   # only mean is per study; median, min, max and sd are pooled (review)
            values = ("values are the unweighted means of each group's per-study means" if value == "mean"
                      else f"values are group {word}, pooled over the studies" if word
                      else "values are group statistics (on basis study_means, mean is the unweighted mean of per-study means)")
        else:
            values = f"values are group {word}" if word else "values are group statistics"
        return f"one row per {by} {_in_stacked(table, stacked)}; {values}" if stacked else f"one row per {by}; {values}"
    if noted:                                        # another note as written; a stack's says which of its tables (review 2)
        stacks = [_STACKED_NOTE.fullmatch(note) for note in noted]
        return "; ".join(f"{s[1]}, {_in_stacked(table, s[2])}" if s else note for note, s in zip(noted, stacks))
    if UNIT_COL in table.manifest.columns:
        units = [row.get(UNIT_COL, "") for row in rows] if rows is not None else []
        if not all(units):                           # review 2: blank IDs are unknown units, not one unit
            return "one row per observation; some rows name no observation unit"
        if len(set(units)) < len(units):
            return "one row per observation; some observation units have several"
        return "one row per observation unit"
    return "what one row stands for is not recorded"


def _short(name: str) -> str:
    return name if len(name) <= NAME_CHARS else name[:NAME_CHARS - 1] + "\u2026"


def _named_reason(reason: str, entries: Iterable[str], label: dict[str, str]) -> str:
    """An exclusion's reason with the entries it counts named, by their names in the tables (else their IDs), so the reader
    can see which went (coach 2026-10-24 12:50 UTC, GAP 2; Luna E13.4: "entries only in y", 1 was the check CIP199062.1)."""
    names = sorted(_short(label.get(entry) or entry) for entry in entries)
    more = f" and {len(names) - NAMED_ENTRIES} more" if len(names) > NAMED_ENTRIES else ""
    return f"{reason} ({', '.join(names[:NAMED_ENTRIES])}{more})"


PROGRAM_COLUMNS = ("programDbId", "programName", "additionalInfoProgramDbId", "additionalInfoProgramName")


def _unknown_cell(column: str, cell: str) -> bool:
    """A catalog cell that names nothing. Program columns use the shared rule (contracts.unknown_metadata: "", "[]", "{}", "NA",
    "N/A", "none", "null"; coach 2026-10-24 14:49 / 15:49 UTC, NEXT 4); every other column keeps "" and "[]" (review 2:
    SweetPotatoBase's location 120 is named "NA")."""
    return unknown_metadata(cell) if column in PROGRAM_COLUMNS else (not cell.strip() or cell.strip() == "[]")


def _apply_duplicate_policy(table: Table, classified: Classified, policy: str,
                            value: str | None = None) -> tuple[list[tuple[dict[str, str], float]], str]:
    """Return (rows-with-values to analyse, transformation text). Refuses duplicates unless a rule is named. value: the value
    column, which names what a group_stats table's values are."""
    if not (classified.repeated_observation_ids or classified.repeated_measurements):
        return classified.valid, f"valid values used as recorded ({_row_unit(table, value, [row for row, _ in classified.valid])})"
    if policy == "reject":
        raise AnalystError("unsupported", f"{table.manifest.artifact_id}: {classified.duplicate_note}")
    if not classified.units_known:
        raise AnalystError("unsupported", "mean_per_unit needs an observation unit ID on every valid row; independence is unknown here")
    groups: dict[tuple[str, ...], list[tuple[dict[str, str], float]]] = {}
    for row, number in classified.valid:
        groups.setdefault(_duplicate_key(row, table.columns), []).append((row, number))   # type: ignore[arg-type]
    merged = [(rows[0][0], statistics.fmean(n for _, n in rows)) for rows in groups.values()]
    return merged, (f"duplicate_policy=mean_per_unit: {len(classified.valid)} valid rows averaged within "
                    f"{len(merged)} observation units before pooling")


def _completeness_gate(table: Table, allow_incomplete: bool) -> str | None:
    if table.manifest.complete:
        return None
    if not allow_incomplete:
        raise AnalystError("incomplete_data", f"{table.manifest.artifact_id} is not a complete collection; no study-level figure can be "
                                              "claimed from it (pass allow_incomplete=True for a clearly labelled partial description)")
    return "INCOMPLETE ARTIFACT: describes the rows fetched, not the study"


def _exclusions(classified: Classified) -> list[Exclusion]:
    out = []
    if classified.n_missing:
        out.append(Exclusion(reason="raw-missing value token", count=classified.n_missing))
    if classified.n_invalid:
        out.append(Exclusion(reason="invalid or non-finite value token", count=classified.n_invalid))
    return out


# Count labels say exactly what n counts (coach, 2026-10-17, count_layout_plan RECOMMENDATION item 2). The method texts
# (transformation) stay as they are: chat_presenter matches them. Before: "rows in the complete studies table" also after a
# filter cut the table down (zero matches read as an empty catalog), and "valid numeric values" also when each unit's
# values were averaged first (n then counts units).
_FILTER_LINEAGE = re.compile(r"lineage: filter_rows: (.+) on \S+ sha256=\S+")
PER_UNIT = "valid values averaged per unit first"


def _filters_applied(manifest: Any) -> list[str]:
    """The filter_rows filters that made this table, oldest first ("seasons contains '2026'"); [] for a source table. A
    quoted JSON value (the catalog season filter, '"2026"') is shown once in double quotes: seasons contains "2026"."""
    found = [m[1] for m in (_FILTER_LINEAGE.fullmatch(note) for note in manifest.quality_notes) if m]
    # the two tests that take no value, in words (eval 20261019T170352Z, q14: "value is_valid_number ''")
    found = [re.sub(r"^(\S+) is_valid_number '.*'$", r"\1 is a valid number", re.sub(r"^(\S+) is_missing '.*'$", r"\1 is missing", text))
             for text in found]
    return [re.sub(r"""'"([^'"]*)"'$""", r'"\1"', text) for text in found]


def _made_by(manifest: Any) -> list[str]:
    """The tools that made this table from other tables, read from its lineage notes (filter_rows, concat_tables, plot_design,
    ...); [] for a table as fetched from the server, which has none."""
    return list(dict.fromkeys(m[1] for m in (re.match(r"lineage: (\w+)", note) for note in manifest.quality_notes) if m))


def _rows_label(manifest: Any) -> str:
    """What a row count of this table counts: its rows, and the filters that kept them."""
    filters = _filters_applied(manifest)
    complete = "complete " if manifest.complete else ""
    if not filters:
        return f"rows in the {complete}{manifest.kind} table"
    return f"rows matching {' and '.join(filters)} in the {complete}{manifest.kind} table"


def _numeric_claims(table: Table, classified: Classified, used: list[tuple[dict[str, str], float]], transformation: str,
                    *, grouping: list[str], filters: dict[str, str], unit: str | None, denominator_name: str,
                    label_parts: tuple[str, ...] = (), claim_namespace: str | None = None) -> tuple[dict[str, float | None], list[Claim]]:
    values = [n for _, n in used]
    stats = _stats(values)
    n_units = len({r[UNIT_COL] for r, _ in used}) if classified.units_known else None
    denominator = Denominator(name=denominator_name, n=len(values))
    exclusions = _exclusions(classified)
    hashes = [table.manifest.sha256]
    claims: list[Claim] = []
    for kind in ("mean", "median", "min", "max", "sd"):
        value = stats[kind]
        reason = None
        if value is None:
            reason = "fewer than 2 valid values: sample SD undefined" if kind == "sd" and values else "no valid numeric values"
        claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, kind, *label_parts, namespace=claim_namespace), kind=kind, value=value, unit=unit,
                            missing_reason=reason, denominator=denominator, n_independent_units=n_units,
                            source_artifact_hashes=hashes, filters=filters, grouping=grouping,
                            transformation=f"{kind} of valid values; {transformation}", exclusions=exclusions))
    cv, cv_reason = _cv(stats)
    claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, "ratio", "cv", *label_parts, namespace=claim_namespace), kind="ratio",
                        value=cv, unit="%", missing_reason=cv_reason, denominator=denominator, n_independent_units=n_units,
                        source_artifact_hashes=hashes, filters=filters, grouping=grouping,
                        transformation=f"{CV_OF_VALUES}; {transformation}", exclusions=exclusions))
    claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, "count", *label_parts, namespace=claim_namespace), kind="count", value=len(values),
                        denominator=denominator, n_independent_units=n_units, source_artifact_hashes=hashes, filters=filters,
                        grouping=grouping, transformation=f"count of valid values; {transformation}", exclusions=exclusions))
    return stats, claims


def _claims_json(claims: list[Claim]) -> list[dict[str, Any]]:
    return [c.model_dump(mode="json") for c in claims]


# --------------------------------------------------------------------------
# Argument models = the schemas (strict, extra keys refused)
# --------------------------------------------------------------------------

class ToolArgs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


DuplicatePolicy = Literal["reject", "mean_per_unit"]
def _handle() -> Any:
    return Field(description="a registered artifact handle such as art_3f9a2c_0001 (never a path)")


class TableInfoArgs(ToolArgs):
    artifact_id: ArtifactId = _handle()


class NumericSummaryArgs(ToolArgs):
    artifact_id: ArtifactId = _handle()
    value: str = Field(default=VALUE_COL, description="the value column")
    duplicate_policy: DuplicatePolicy = Field(default="reject", description="what to do with repeated measurements of one unit")
    allow_incomplete: bool = Field(default=False, description="describe an incomplete artifact anyway, clearly labelled")
    roster_artifact_id: ArtifactId | None = Field(default=None, description="the study's plot list (get_observation_units) for Percent "
                                                  "missing: its plots without a valid value, over its plots, x 100; ONE study, "
                                                  "the measurements table as fetched (not a filtered, stacked or joined table)")


class GroupStatsArgs(ToolArgs):
    artifact_id: ArtifactId = _handle()
    by: str = Field(description="a stable ID column to group by (e.g. germplasmDbId); names are carried as labels")
    value: str = Field(default=VALUE_COL)
    top_n: int = Field(default=10, ge=1, le=1000,
                       description="how many included groups to show AND state as claims (best first); the artifact holds all. A question "
                                   "about every group (plots per clone) needs top_n at least the number of groups")
    ascending: bool = Field(default=False)
    min_independent_n: int = Field(default=2, ge=1, description="exclusion threshold on independent units; not a design correction")
    duplicate_policy: DuplicatePolicy = Field(default="reject")
    allow_incomplete: bool = Field(default=False)


FilterOp = Literal["eq", "ne", "contains", "gt", "ge", "lt", "le", "is_missing", "is_valid_number"]


class FilterRowsArgs(ToolArgs):
    artifact_id: ArtifactId = _handle()
    column: str
    op: FilterOp = Field(description="eq/ne/contains compare text (contains = literal, case-insensitive); gt/ge/lt/le compare valid numbers only")
    value: str = Field(default="", description="the comparison text; ignored by is_missing / is_valid_number")


class ConcatTablesArgs(ToolArgs):
    artifact_ids: list[ArtifactId] = Field(min_length=2, max_length=50, description="two or more compatible artifacts")


class MissingReportArgs(ToolArgs):
    artifact_id: ArtifactId = _handle()
    by: str | None = Field(default=None, description="optional ID column for per-group counts")
    value: str = Field(default=VALUE_COL)
    roster_artifact_id: ArtifactId | None = Field(default=None, description="an observation_units artifact for the same study: enables the expected-unit denominator")


class PlotBarArgs(ToolArgs):
    artifact_id: ArtifactId = _handle()
    x: str
    y: str
    top_n: int = Field(default=10, ge=1, le=100)


class CountRecordsArgs(ToolArgs):
    artifact_id: ArtifactId = _handle()
    group_by: str | None = Field(default=None, description="column whose value frequencies to count")
    distinct_by: str | None = Field(default=None, description="column whose distinct non-empty values to count")


class RankEntriesArgs(ToolArgs):
    artifact_id: ArtifactId = _handle()
    by: str = Field(default="germplasmDbId", description="the entry ID column to rank (names are carried as labels)")
    value: str = Field(default=VALUE_COL)
    order: Literal["highest", "lowest"] = Field(default="highest", description="which end of the order comes first")
    top_k: int = Field(default=1, ge=1, le=1000, description="how many entries to state from the top of the order; entries tied "
                                                              "with the last one are stated too")
    named: list[str] = Field(default_factory=list, max_length=50,
                             description="entries (exact ID or exact name) whose rank and mean to state wherever they fall")
    exclude: list[str] = Field(default_factory=list, max_length=50,
                               description="entries (exact ID or exact name) left out before ranking, for example the checks")
    min_independent_n: int = Field(default=1, ge=1, description="exclusion threshold on independent units; not a design correction")
    duplicate_policy: DuplicatePolicy = Field(default="reject")
    allow_incomplete: bool = Field(default=False)
    basis: Literal["pooled", "study_means"] = Field(default="pooled", description="pooled: the mean of all the entry's valid values; "
                                                    "study_means: the unweighted mean of its per-study means, over the entries present in "
                                                    "every study of the table (each study weighted equally)")


class ThresholdCountArgs(ToolArgs):
    artifact_id: ArtifactId = _handle()
    count: Literal["entries", "plots"] = Field(default="entries", description="entries: each entry by its mean over its plots; plots: each plot by its own value")
    op: Literal["above", "below", "at_least", "at_most"] = Field(default="above", description="above / below are strict (equal does not count); "
                                                                                              "at_least / at_most include equal")
    threshold: float | None = Field(default=None, description="a number from the question (in a plan step, written as text: \"12.5\"); "
                                                              "exactly one of threshold / threshold_entries")
    threshold_entries: list[str] = Field(default_factory=list, max_length=50,
                                         description="entries (exact ID or exact name) whose entry means, averaged unweighted within each study, are "
                                                     "the threshold, for example a check; they are left out of the count")
    exclude: list[str] = Field(default_factory=list, max_length=50,
                               description="entries (exact ID or exact name) left out of the count, for example the other checks")
    name_prefix: str | None = Field(default=None, min_length=1, description="count only the entries whose name starts with this exact text, for example PH17.")
    by: str = Field(default="germplasmDbId", description="the entry ID column (names are carried as labels)")
    value: str = Field(default=VALUE_COL)
    duplicate_policy: DuplicatePolicy = Field(default="reject")
    allow_incomplete: bool = Field(default=False)

    @field_validator("threshold", mode="before")
    @classmethod
    def _number_text(cls, value: Any) -> Any:
        """A plan step carries no floats (PlanInputValue), so a threshold may arrive as text: "12.5" is the number 12.5."""
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError as error:
                raise ValueError(f"threshold {value!r} is not a number") from error
        return value

    @model_validator(mode="after")
    def _one_threshold(self) -> ThresholdCountArgs:
        if (self.threshold is None) == (not self.threshold_entries):
            raise ValueError("give exactly one of threshold (a number from the question) and threshold_entries (named entries whose mean is the threshold)")
        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("threshold must be a finite number")
        return self


class CorrelateEntryMeansArgs(ToolArgs):
    artifact_ids: list[ArtifactId] = Field(min_length=2, max_length=2,
                                           description="the two measurement tables whose entry means are correlated: one trait in two studies, or two "
                                                       "traits of the same studies (the order does not matter)")
    by: str = Field(default="germplasmDbId", description="the entry ID column both tables share (names are carried as labels)")
    value: str = Field(default=VALUE_COL)
    method: Literal["pearson", "spearman", "both"] = Field(default="both", description="Pearson r of the entry means, Spearman rho of their ranks "
                                                                                       "(tied means share the average rank), or both")
    entries: Literal["in_every_study", "in_any_study"] = Field(default="in_every_study",
                                                               description="for a table of several studies, an entry's mean is the unweighted mean of its "
                                                                           "study means: over the entries present in every study of the table (default), or in any")
    exclude: list[str] = Field(default_factory=list, max_length=50, description="entries (exact ID or exact name) left out, for example the checks")
    name_prefix: str | None = Field(default=None, min_length=1, description="keep only the entries whose name starts with this exact text, for example PH17.")
    min_pairs: int = Field(default=3, ge=3, description="the fewest shared entries a correlation is stated for")
    duplicate_policy: DuplicatePolicy = Field(default="reject")
    allow_incomplete: bool = Field(default=False)


class PlotDesignArgs(ToolArgs):
    roster_artifact_id: ArtifactId = Field(description="the plot list: a COMPLETE observation_units artifact of one study, from "
                                                       "get_observation_units (a registered handle such as art_3f9a2c_0001)")
    artifact_id: ArtifactId | None = Field(default=None, description="optional: a measurements (observations) artifact of the same "
                                                                     "study; each of its rows gets its plot's design columns")
    allow_incomplete: bool = Field(default=False)


class AnovaTrialArgs(ToolArgs):
    artifact_id: ArtifactId = Field(description="the plot values of ONE study and ONE variable: for an RCBD the table plot_design saved "
                                                "with each plot's rep column (a registered handle such as art_3f9a2c_0001)")
    value: str = Field(default=VALUE_COL, description="the value column")
    entry: str = Field(default="germplasmDbId", description="the entry (genotype) column")
    rep: str = Field(default="rep", description="the replicate column (plot_design's rep); when the table has none, or no plot states "
                                                "one, the analysis is one-way (entries only)")
    duplicate_policy: DuplicatePolicy = Field(default="reject", description="what to do with repeated measurements of one plot")
    allow_incomplete: bool = Field(default=False, description="analyse an incomplete artifact anyway, clearly labelled")
    cells: Literal["refuse", "mean", "single_only"] = Field(default="refuse", description="an entry with several plots in one rep: refuse "
                                                            "(default), mean = the mean of the cell's plots, single_only = only the entries "
                                                            "with exactly one plot in every rep; a rule other than refuse only when the "
                                                            "person asked for it")
    exclude: list[str] = Field(default_factory=list, max_length=50, description="entries (exact ID or exact name) the person named to leave "
                                                                                "out, for example the checks; never entries chosen by the app")


class GxePartitionArgs(ToolArgs):
    artifact_ids: list[ArtifactId] = Field(min_length=1, max_length=40,
                                           description="the measurements of ONE trait: one table per study, or one table holding several "
                                                       "studies (concat_tables); each study is one environment")
    by: str = Field(default="germplasmDbId", description="the entry ID column (names are carried as labels)")
    value: str = Field(default=VALUE_COL)
    rep: str = Field(default="rep", description="the replicate column plot_design adds: when every plot states one and every study is a "
                                                "complete RCBD with the same number of reps, the pooled error and the variance components "
                                                "are given too")
    exclude: list[str] = Field(default_factory=list, max_length=50, description="entries (exact ID or exact name) left out, for example the checks")
    name_prefix: str | None = Field(default=None, min_length=1, description="keep only the entries whose name starts with this exact text, for example PH17.")
    duplicate_policy: DuplicatePolicy = Field(default="reject", description="what to do with repeated measurements of one plot")
    allow_incomplete: bool = Field(default=False)


PLOT_LISTS_TEXT = ("the plot lists (get_observation_units) of the two studies, when the person names entries to leave out: a name in "
                   "neither table but in a plot list is an entry of the trial without a value of the trait, named apart; a name in "
                   "neither is refused")


class CompareTrialsPairedArgs(ToolArgs):
    base_artifact_id: ArtifactId = Field(description="the measurements of the reference trial: ONE study (a registered handle such as "
                                                     "art_3f9a2c_0001); the change is stated relative to its mean")
    other_artifact_id: ArtifactId = Field(description="the measurements of the trial compared with it: ONE other study, the same variable")
    by: str = Field(default="germplasmDbId", description="the entry ID column both tables share (names are carried as labels)")
    value: str = Field(default=VALUE_COL)
    percent: Literal["change", "reduction"] = Field(default="change", description="change: the differences other - base and the percent "
                                                    "100 x (mean other - mean base) / mean base; reduction: base - other and 100 x (mean "
                                                    "base - mean other) / mean base; as the question words it")
    exclude: list[str] = Field(default_factory=list, max_length=50, description="entries (exact ID or exact name) left out, for example the checks")
    name_prefix: str | None = Field(default=None, min_length=1, description="keep only the entries whose name starts with this exact text, for example PH17.")
    roster_artifact_ids: list[ArtifactId] = Field(default_factory=list, max_length=2, description=PLOT_LISTS_TEXT)
    duplicate_policy: DuplicatePolicy = Field(default="reject", description="what to do with repeated measurements of one plot")
    allow_incomplete: bool = Field(default=False)


class StressIndicesArgs(ToolArgs):
    nonstress_artifact_id: ArtifactId = Field(description="the measurements of the non-stress trial (Yp): ONE study (a registered handle "
                                                          "such as art_3f9a2c_0001)")
    stress_artifact_id: ArtifactId = Field(description="the measurements of the stress trial (Ys): ONE other study, the same variable")
    by: str = Field(default="germplasmDbId", description="the entry ID column both tables share (names are carried as labels)")
    value: str = Field(default=VALUE_COL)
    rank_by: Literal["STI", "GMP", "MP", "HM", "YI", "YSI", "TOL", "SSI", "RDI"] = Field(default="STI", description="the index the "
                                                                                          "entries are ranked by")
    order: Literal["highest", "lowest"] = Field(default="highest", description="which end of the order comes first")
    top_k: int = Field(default=3, ge=1, le=50, description="how many entries to state from the top of the order; entries tied with the "
                                                           "last one are stated too")
    exclude: list[str] = Field(default_factory=list, max_length=50, description="entries (exact ID or exact name) the person named to leave "
                                                                                "out, for example the checks")
    name_prefix: str | None = Field(default=None, min_length=1, description="keep only the entries whose name starts with this exact text, for example PH17.")
    roster_artifact_ids: list[ArtifactId] = Field(default_factory=list, max_length=2, description=PLOT_LISTS_TEXT)
    duplicate_policy: DuplicatePolicy = Field(default="reject", description="what to do with repeated measurements of one plot")
    allow_incomplete: bool = Field(default=False)


class HeterosisPopulationArgs(ToolArgs):
    artifact_ids: list[ArtifactId] = Field(min_length=1, max_length=20,
                                           description="the measurements of ONE trait: one table per site, or one table of several sites "
                                                       "(concat_tables); each study is one site")
    offspring_prefixes: list[str] = Field(default_factory=list, max_length=20, description="the offspring group: exact name prefixes the "
                                          "person wrote, for example PH17. (a trailing * is the wildcard)")
    offspring_names: list[str] = Field(default_factory=list, max_length=200, description="the offspring group: exact entry names or IDs "
                                       "the person wrote")
    parent_a_prefixes: list[str] = Field(default_factory=list, max_length=20, description="the first parent group: name prefixes the "
                                         "person wrote")
    parent_a_names: list[str] = Field(default_factory=list, max_length=200, description="the first parent group: entry names or IDs")
    parent_b_prefixes: list[str] = Field(default_factory=list, max_length=20, description="the second parent group; leave it and "
                                         "parent_b_names empty for one parent group (the mid-parent is then its mean)")
    parent_b_names: list[str] = Field(default_factory=list, max_length=200, description="the second parent group: entry names or IDs")
    founder_prefixes: list[str] = Field(default_factory=list, max_length=20, description="the founders, for the response of the parents "
                                        "over the founders (optional)")
    founder_names: list[str] = Field(default_factory=list, max_length=200, description="the founders: entry names or IDs")
    by: str = Field(default="germplasmDbId", description="the entry ID column (names are carried as labels)")
    value: str = Field(default=VALUE_COL)
    duplicate_policy: DuplicatePolicy = Field(default="reject", description="what to do with repeated measurements of one plot")
    allow_incomplete: bool = Field(default=False)


class PlotBasisH2Args(ToolArgs):
    artifact_ids: list[ArtifactId] = Field(min_length=1, max_length=20,
                                           description="the measurements of ONE trait: one table per site, or one table of several sites "
                                                       "(concat_tables); each study is one site")
    offspring_prefixes: list[str] = Field(default_factory=list, max_length=20, description="the unreplicated offspring: exact name prefixes "
                                          "the person wrote, for example PH17. (a trailing * is the wildcard)")
    offspring_names: list[str] = Field(default_factory=list, max_length=200, description="the offspring: exact entry names or IDs the "
                                       "person wrote")
    error_from: Literal["replicated", "checks"] = Field(default="replicated", description="the error variance s2e: replicated = every "
                                                        "entry with at least 2 plots outside the offspring (parents, founders, checks); "
                                                        "checks = only the entries whose entryType is check (plot_design's column)")
    by: str = Field(default="germplasmDbId", description="the entry ID column (names are carried as labels)")
    value: str = Field(default=VALUE_COL)
    entry_type: str = Field(default="entryType", description="the entry type column plot_design adds (check / test)")
    duplicate_policy: DuplicatePolicy = Field(default="reject", description="what to do with repeated measurements of one plot")
    allow_incomplete: bool = Field(default=False)


class CompareMeansArgs(ToolArgs):
    base_artifact_id: ArtifactId = Field(description="the reference table (the control, the earlier harvest, the first site): a "
                                                     "registered artifact handle such as art_3f9a2c_0001 (never a path)")
    other_artifact_id: ArtifactId = Field(description="the compared table, a registered artifact handle (never a path)")
    base_study_db_id: str = Field(min_length=1, description=("the reference study's ID as the plan names it: the base table must be "
                                                             "that one study's table and the other table not (a swapped pair would "
                                                             "turn a reduction around)"))
    by: str = Field(default="germplasmDbId", description="the entry ID column (names are carried as labels)")
    value: str = Field(default=VALUE_COL)
    percent: Literal["change", "reduction"] = Field(description=(
        "as the plan names it; change: (other - base) / base x 100, a drop is negative; reduction: (base - other) / base x 100, "
        "a drop is positive"))
    mean_of: Literal["entry_means", "values"] = Field(default="entry_means", description=(
        "entry_means: each study's mean is the mean of its entry means; values: the mean of the valid values of its entries"))
    shared_entries_only: bool = Field(default=True, description="use only the entries with a mean in both tables")
    exclude: list[str] = Field(default_factory=list, max_length=50,
                               description="entries (exact ID or exact name) left out of both means, for example the checks")
    min_independent_n: int = Field(default=1, ge=1, description="exclusion threshold on independent units; not a design correction")
    duplicate_policy: DuplicatePolicy = Field(default="reject")
    allow_incomplete: bool = Field(default=False)


# --------------------------------------------------------------------------
# The thirteen tools
# --------------------------------------------------------------------------

def tool_table_info(registry: ArtifactRegistry, a: TableInfoArgs) -> ToolResult:
    """What is in a registered table: kind, row count, completeness, study/variable IDs, measurement metadata
    (trait, unit, timepoint when stated), column meanings, quality notes, value-token counts (valid / raw-missing /
    invalid), duplicate and independence facts, and a five-row labelled preview. Start here before any calculation."""
    table = _load(registry, a.artifact_id)
    m = table.manifest
    data: dict[str, Any] = {
        "artifact_id": m.artifact_id, "kind": m.kind, "row_count": m.row_count, "complete": m.complete, "sha256": m.sha256,
        "study_ids": list(m.study_ids), "variable_ids": list(m.variable_ids), "measurement": m.measurement.model_dump(),
        "columns": dict(m.columns), "quality_notes": [n for n in m.quality_notes][:20], "source_request_count": len(m.source_request_ids),
        "value_policy": VALUE_POLICY,
    }
    if VALUE_COL in m.columns:
        c = _classify(table, VALUE_COL)
        data["value_counts"] = {"total_rows": c.n_rows, "valid": c.n_valid, "raw_missing": c.n_missing, "invalid_or_nonfinite": c.n_invalid}
        data["n_independent_units"] = c.n_independent_units
        data["independence"] = ("every valid row names an observation unit" if c.units_known
                                else "unknown: no observation unit ID on every valid row; rows are not plots")
        data["duplicates"] = {"repeated_observation_ids": c.repeated_observation_ids, "repeated_measurements_per_unit": c.repeated_measurements,
                              "note": c.duplicate_note or "none detected"}
    preview = registry.preview(m.artifact_id, 5)
    data["preview"] = {"displayed_rows": preview.displayed_rows, "total_rows": preview.total_rows, "rows": preview.rows}
    return ToolResult(ok=True, data=data, artifact_ids=[m.artifact_id], complete=m.complete)


def _timing_warnings(table: Table, value_column: str) -> list[str]:
    """Describe missing timing on valid source rows, without assigning dates or growth stages."""
    if table.manifest.kind != "observations":
        return []
    valid = [row for row in table.rows if classify_value(row.get(value_column))[0] == "valid"]
    if not valid:
        return []
    missing = sum(not row.get("observationTimeStamp", "").strip() for row in valid)
    warnings = []
    if missing:
        warnings.append(f"Measurement dates are not recorded for {missing} of {len(valid)} valid source rows; "
                        "the timing of those measurements is unknown.")
    if not table.manifest.measurement.timepoint:
        warnings.append("Growth stage or timepoint is not stated in the source metadata; a shared growth stage cannot be verified.")
    return warnings


def tool_numeric_summary(registry: ArtifactRegistry, a: NumericSummaryArgs) -> ToolResult:
    """Counts (total, valid, raw-missing, invalid/non-finite) and mean, median, min, max, sample SD and CV (100 x SD / mean)
    of the valid values of ONE variable, as typed claims with a named denominator, n_independent_units (or unknown) and the
    exclusions; with the study's plot list (roster_artifact_id) also Percent missing, the share of its plots without a valid
    value - the columns of Breedbase's Phenotype Summary. Refuses mixed variables, unresolved duplicates and incomplete
    artifacts (unless allow_incomplete). SD with fewer than 2 valid values is null. Values keep full precision; `display` is
    rounded. On a group_stats or rank_entries result (one row per group or entry), value="mean" gives the mean of the group
    means, each group weighted equally (a mean of genotype means). Those tables keep every group: included=false marks one
    under min_independent_n (or not ranked), left_out=true one that rank_entries left out; filter_rows on those columns first
    when they must not count."""
    table = _load(registry, a.artifact_id)
    _require_column(table, a.value, "value column")
    caveat = _completeness_gate(table, a.allow_incomplete)
    variable = _single_variable(table)
    classified = _classify(table, a.value)
    used, transformation = _apply_duplicate_policy(table, classified, a.duplicate_policy, value=a.value)
    if caveat:
        transformation = f"{caveat}; {transformation}"
    unit = table.manifest.measurement.unit
    averaged = len(used) != len(classified.valid)                       # duplicate_policy=mean_per_unit merged rows
    repeated_units = classified.units_known and len({row.get(UNIT_COL, "") for row, _ in used}) < len(used)    # review 2
    denominator_name = (f"observation units ({PER_UNIT})" if averaged else
                        "valid numeric values (rows; some observation units have several)" if repeated_units else
                        "valid numeric values, one per observation unit" if classified.units_known else "valid numeric values (rows; independence unknown)")
    filters_note = _filters_applied(table.manifest)
    if filters_note:
        denominator_name += f", from rows matching {' and '.join(filters_note)}"
    stats, claims = _numeric_claims(table, classified, used, transformation, grouping=[], filters={}, unit=unit,
                                   denominator_name=denominator_name,
                                   label_parts=() if a.value == VALUE_COL else (a.value,),
                                   claim_namespace=None if a.value == VALUE_COL else "numeric-value")
    warnings = [w for w in (classified.duplicate_note if a.duplicate_policy != "reject" else None, caveat) if w]
    if unit is None:
        warnings.append("unit not stated in the artifact's measurement metadata; claims carry unit=null")
    warnings.extend(_timing_warnings(table, a.value))
    percent_missing = None
    if a.roster_artifact_id is not None:
        roster = _plot_list(registry, a.roster_artifact_id, table, require_complete=False)
        percent_missing, missing_claim = _percent_missing(table, roster, used, transformation,
                                                          label_parts=() if a.value == VALUE_COL else (a.value,),
                                                          namespace=None if a.value == VALUE_COL else "numeric-value")
        claims.append(missing_claim)
        if percent_missing["valid_values_not_on_list"]:                   # valid values only: 'NA' is in no statistic
            warnings.append(f"{percent_missing['valid_values_not_on_list']} valid value(s) on plots not on the plot list "
                            f"{roster.manifest.artifact_id}: kept in the statistics; percent missing not available")
    data = {
        "artifact_id": table.manifest.artifact_id, "sha256": table.manifest.sha256, "variable_id": variable, "unit": unit,
        "value_column": a.value, "counts": {"total_rows": classified.n_rows, "valid": classified.n_valid, "raw_missing": classified.n_missing,
                                            "invalid_or_nonfinite": classified.n_invalid, "values_used": len(used)},
        "n_independent_units": claims[0].n_independent_units, "stats": stats, "display": {k: _display(v) for k, v in stats.items()},
        "transformation": transformation, "exclusions": [e.model_dump() for e in _exclusions(classified)],
        "claims": _claims_json(claims), "value_policy": VALUE_POLICY, "analysis_target": ANALYSIS_TARGET,
        "cv_percent": _cv(stats)[0], "percent_missing": percent_missing,
    }
    return ToolResult(ok=True, data=data, artifact_ids=[table.manifest.artifact_id] + ([a.roster_artifact_id] if a.roster_artifact_id else []),
                      complete=table.manifest.complete, warnings=warnings)


GROUP_RESULT_COLUMNS = {
    "group_id": "exact group ID text", "label": "name carried from the label column, if any", "included": "true/false",
    "exclusion_reason": "why the group was excluded ('' if included)", "rank": "rank among included groups (ties share a rank)",
    "n_rows": "source rows in the group", "n_valid_values": "valid numeric values used", "n_raw_missing": "raw-missing tokens",
    "n_invalid": "invalid or non-finite tokens", "n_independent_units": "distinct observation units among valid rows ('' = unknown)",
    "mean": "full precision", "median": "full precision", "min": "full precision", "max": "full precision", "sd": "sample SD, full precision ('' if undefined)",
}


def _group_results(table: Table, by: str, value: str, classified: Classified, used: list[tuple[dict[str, str], float]],
                   min_independent_n: int, label_col: str | None) -> tuple[list[dict[str, Any]], int]:
    """Per-group statistics of the used values (group_stats and rank_entries): every source row is attributed to a group first,
    so missing and invalid counts are per group too. Returns the groups sorted by ID, and the rows with an empty group ID."""
    per_group_rows: dict[str, list[dict[str, str]]] = {}
    for row in table.rows:
        per_group_rows.setdefault(row.get(by, ""), []).append(row)
    per_group_used: dict[str, list[tuple[dict[str, str], float]]] = {}
    for row, number in used:
        per_group_used.setdefault(row.get(by, ""), []).append((row, number))

    results: list[dict[str, Any]] = []
    no_group_rows = len(per_group_rows.get("", []))
    for group_id in sorted(per_group_rows):
        if group_id == "":
            continue
        rows = per_group_rows[group_id]
        counts = Counter(classify_value(r.get(value))[0] for r in rows)
        values = [n for _, n in per_group_used.get(group_id, [])]
        n_units = len({r[UNIT_COL] for r, _ in per_group_used.get(group_id, [])}) if classified.units_known else None
        basis = n_units if n_units is not None else len(values)
        reason = ""
        if basis < min_independent_n:
            what = "independent units" if n_units is not None else "valid values (independence unknown)"
            reason = f"fewer than {min_independent_n} {what}: {basis}"
        labels = {r.get(label_col, "") for r in rows if r.get(label_col, "")} if label_col else set()
        results.append({"group_id": group_id, "label": "; ".join(sorted(labels)), "included": reason == "", "exclusion_reason": reason,
                        "n_rows": len(rows), "n_valid_values": len(values), "n_raw_missing": counts["missing"], "n_invalid": counts["invalid"],
                        "n_independent_units": n_units, **_stats(values)})
    return results, no_group_rows


def tool_group_stats(registry: ArtifactRegistry, a: GroupStatsArgs) -> ToolResult:
    """Per-group statistics of ONE variable, grouped by a stable ID column with names carried as labels. Each group
    reports n_valid_values and n_independent_units separately. Groups under min_independent_n are EXCLUDED with a
    reason (an exclusion threshold, not a design correction); rows with an empty group ID are excluded and counted.
    The FULL result (every group, included or not) is saved as an analysis_result artifact; the message shows top_n
    included groups ranked by mean at full precision, ties shared and listed. Descriptive only: not adjusted, not
    evidence of superiority. For the mean of all the group means in the saved result (each group weighted equally), call
    numeric_summary on the result_artifact_id with value="mean"; the result keeps every group, included or not, so
    filter_rows on it first (included eq true, or on the label) leaves groups out."""
    table = _load(registry, a.artifact_id)
    _require_column(table, a.by, "group column")
    _require_column(table, a.value, "value column")
    if a.by.endswith("Name"):
        raise AnalystError("invalid_argument", f"group by the ID column, not {a.by!r}; names are carried as labels")
    caveat = _completeness_gate(table, a.allow_incomplete)
    variable = _single_variable(table)
    classified = _classify(table, a.value)
    used, transformation = _apply_duplicate_policy(table, classified, a.duplicate_policy, value=a.value)
    if caveat:
        transformation = f"{caveat}; {transformation}"
    unit = table.manifest.measurement.unit
    label_col = LABEL_FOR.get(a.by)
    label_col = label_col if label_col in table.manifest.columns else None
    results, no_group_rows = _group_results(table, a.by, a.value, classified, used, a.min_independent_n, label_col)

    included = [g for g in results if g["included"]]
    included.sort(key=lambda g: ((g["mean"] if a.ascending else -g["mean"]), g["group_id"]))
    for g in included:                                            # competition ranking: ties share a rank
        g["rank"] = 1 + sum(1 for o in included if (o["mean"] < g["mean"] if a.ascending else o["mean"] > g["mean"]))
    for g in results:
        g.setdefault("rank", None)
    tie_groups = [sorted(x["group_id"] for x in included if x["mean"] == v) for v in sorted({g["mean"] for g in included})]
    ties = [t for t in tie_groups if len(t) > 1]

    lineage = [f"lineage: group_stats by {a.by} on {table.manifest.artifact_id} sha256={table.manifest.sha256}",
               f"{ROWS_NOTE}group_stats by {a.by}",                             # GAP 1: one row per group
               f"transformation: {transformation}", f"min_independent_n={a.min_independent_n} (exclusion threshold)",
               f"rows with an empty {a.by}: {no_group_rows} (excluded, not dropped silently)", f"analysis target: {ANALYSIS_TARGET}"]
    text_rows = [{**g, "included": "true" if g["included"] else "false",
                  "n_independent_units": "" if g["n_independent_units"] is None else str(g["n_independent_units"]),
                  "rank": "" if g["rank"] is None else str(g["rank"]),
                  **{k: ("" if g[k] is None else repr(g[k])) for k in ("mean", "median", "min", "max", "sd")}} for g in results]
    manifest = registry.save_table(text_rows, kind="analysis_result", columns=GROUP_RESULT_COLUMNS, complete=table.manifest.complete,
                                   source_request_ids=list(table.manifest.source_request_ids), study_ids=list(table.manifest.study_ids),
                                   variable_ids=list(table.manifest.variable_ids), measurement=table.manifest.measurement, quality_notes=lineage)

    exclusions = _exclusions(classified)
    if no_group_rows:
        exclusions.append(Exclusion(reason=f"rows with an empty {a.by}", count=no_group_rows))
    excluded_groups = [g for g in results if not g["included"]]
    if excluded_groups:
        exclusions.append(Exclusion(reason=f"groups under min_independent_n={a.min_independent_n}", count=len(excluded_groups)))
    shown = included[:a.top_n]
    claims: list[Claim] = []
    for g in shown:
        averaged = len(used) != len(classified.valid)
        denominator = Denominator(name=(f"observation units in group {g['group_id']} ({PER_UNIT})" if averaged else
                                        f"valid numeric values in group {g['group_id']}" + ("" if classified.units_known else " (rows; independence unknown)")),
                                  n=g["n_valid_values"])
        claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, "mean", a.by, g["group_id"],
                                                 *(() if a.value == VALUE_COL else (a.value,)),
                                                 namespace=None if a.value == VALUE_COL else "group-value"), kind="mean", value=g["mean"], unit=unit,
                            denominator=denominator, n_independent_units=g["n_independent_units"], source_artifact_hashes=[table.manifest.sha256],
                            filters={a.by: g["group_id"]}, grouping=[a.by], transformation=f"group mean of valid values; {transformation}",
                            exclusions=exclusions))
    data = {
        "artifact_id": table.manifest.artifact_id, "result_artifact_id": manifest.artifact_id, "variable_id": variable, "unit": unit, "by": a.by,
        "label_column": label_col, "groups_total": len(results), "groups_included": len(included), "groups_excluded": len(excluded_groups),
        "rows_without_group_id": no_group_rows, "order": "ascending" if a.ascending else "descending",
        "shown": [{**g, "display_mean": _display(g["mean"])} for g in shown],
        "excluded": [{"group_id": g["group_id"], "label": g["label"], "reason": g["exclusion_reason"]} for g in excluded_groups],
        "ties": ties, "tie_rule": "equal means share a rank; order inside a tie is by group ID",
        "transformation": transformation, "min_independent_n": a.min_independent_n, "claims": _claims_json(claims),
        "exclusions": [e.model_dump() for e in exclusions], "analysis_target": ANALYSIS_TARGET, "value_policy": VALUE_POLICY,
    }
    warnings = [w for w in (caveat, classified.duplicate_note if a.duplicate_policy != "reject" else None) if w]
    if not classified.units_known:
        warnings.append("independence unknown: min_independent_n was applied to valid values, not to units")
    warnings.extend(_timing_warnings(table, a.value))
    return ToolResult(ok=True, data=data, artifact_ids=[table.manifest.artifact_id, manifest.artifact_id], complete=table.manifest.complete, warnings=warnings)


def tool_filter_rows(registry: ArtifactRegistry, a: FilterRowsArgs) -> ToolResult:
    """Keep the rows where one column satisfies an allowlisted typed test and save them as a NEW artifact (same
    columns, lineage recorded). eq/ne compare exact text; contains is a literal case-insensitive substring (never a
    regular expression); gt/ge/lt/le compare valid numbers only and report how many rows had no valid number to
    compare; is_missing / is_valid_number test the value policy."""
    table = _load(registry, a.artifact_id)
    _require_column(table, a.column)
    needle = a.value
    catalog_comparison = (table.manifest.kind in {"studies", "variables", "locations", "programs", "seasons"}
                          and a.op in {"eq", "ne", "contains"} and bool(needle.strip()))
    unknown_cells = sum(_unknown_cell(a.column, row.get(a.column, "")) for row in table.rows) if catalog_comparison else 0
    if table.rows and unknown_cells == len(table.rows):
        raise AnalystError("incomplete_data", f"{a.column} is missing for all {unknown_cells} catalog rows; "
                           "matching membership is unknown, not zero. Ask for a recorded field or more metadata.")
    threshold: float | None = None
    if a.op in ("gt", "ge", "lt", "le"):
        kind, threshold = classify_value(needle)
        if kind != "valid":
            raise AnalystError("invalid_argument", f"{a.op} needs a valid number to compare with, not {needle!r}")
    kept: list[dict[str, str]] = []
    dropped_non_numeric = 0
    for row in table.rows:
        cell = row.get(a.column, "")
        if catalog_comparison and _unknown_cell(a.column, cell):
            continue                    # absent metadata is neither a match nor a known non-match
        if a.op == "eq":
            keep = cell == needle
        elif a.op == "ne":
            keep = cell != needle
        elif a.op == "contains":
            keep = needle.lower() in cell.lower()
        elif a.op == "is_missing":
            keep = classify_value(cell)[0] == "missing"
        elif a.op == "is_valid_number":
            keep = classify_value(cell)[0] == "valid"
        else:
            kind, number = classify_value(cell)
            if kind != "valid":
                dropped_non_numeric += 1
                continue
            keep = {"gt": number > threshold, "ge": number >= threshold, "lt": number < threshold, "le": number <= threshold}[a.op]  # type: ignore[operator]
        if keep:
            kept.append(row)
    description = f"filter_rows: {a.column} {a.op} {needle!r}"
    earlier = [note for note in table.manifest.quality_notes if _FILTER_LINEAGE.fullmatch(note)]   # a filter on a filtered table
    lineage = [*earlier, f"lineage: {description} on {table.manifest.artifact_id} sha256={table.manifest.sha256}",
               f"rows in: {len(table.rows)}, rows kept: {len(kept)}, rows without a valid number to compare: {dropped_non_numeric}",
               *[note for note in table.manifest.quality_notes if note.startswith(ROWS_NOTE)]]      # GAP 1: the same row unit
    membership_incomplete = bool(unknown_cells and a.column in PROGRAM_COLUMNS)
    warnings = ([f"{unknown_cells} of {len(table.rows)} catalog rows have no {a.column}; {len(kept)} known matches were found, "
                 "but the full matching count is unknown. Counts describe recorded matches only."
                 + (" The filtered table is incomplete." if membership_incomplete else "")] if unknown_cells else [])
    lineage.extend(warnings)
    manifest = registry.save_table(kept, kind=table.manifest.kind, columns=dict(table.manifest.columns),
                                   complete=table.manifest.complete and not membership_incomplete,
                                   source_request_ids=list(table.manifest.source_request_ids), study_ids=list(table.manifest.study_ids),
                                   variable_ids=list(table.manifest.variable_ids), measurement=table.manifest.measurement, quality_notes=lineage)
    preview = registry.preview(manifest.artifact_id, 5)
    data = {"source_artifact_id": table.manifest.artifact_id, "artifact_id": manifest.artifact_id, "filter": description,
            "rows_in": len(table.rows), "rows_out": len(kept), "dropped_non_numeric": dropped_non_numeric,
            "rows_with_unknown_filter_metadata": unknown_cells,
            "complete": manifest.complete, "preview": {"displayed_rows": preview.displayed_rows, "rows": preview.rows},
            "note": "counts describe recorded matches; incomplete program membership is blocked"}
    return ToolResult(ok=True, data=data, artifact_ids=[manifest.artifact_id], complete=manifest.complete, warnings=warnings)


def tool_concat_tables(registry: ArtifactRegistry, a: ConcatTablesArgs) -> ToolResult:
    """Stack two or more artifacts that are provably compatible: same columns, same kind, ONE shared variable, same
    measurement metadata (unit, method, scale, timepoint). Adds a source_artifact_id column so every row keeps its
    lineage; study IDs are the union. Refuses anything else with the reason. Complete only if every source is."""
    ids = list(a.artifact_ids)
    if len(set(ids)) != len(ids):
        raise AnalystError("invalid_argument", "the same artifact is listed twice")
    tables = [_load(registry, i) for i in ids]
    first = tables[0]
    variables: set[str] = set()
    for t in tables:
        if t.manifest.columns != first.manifest.columns:
            raise AnalystError("unsupported", f"incompatible schemas: {t.manifest.artifact_id} columns differ from {first.manifest.artifact_id}")
        if t.manifest.kind != first.manifest.kind:
            raise AnalystError("unsupported", f"incompatible kinds: {t.manifest.kind} vs {first.manifest.kind}")
        if t.manifest.measurement != first.manifest.measurement:
            mine, theirs = t.manifest.measurement.model_dump(), first.manifest.measurement.model_dump()
            differing = sorted(k for k in mine if mine[k] != theirs[k])
            raise AnalystError("unsupported", f"incompatible measurement metadata ({', '.join(differing)}): "
                                              f"{t.manifest.artifact_id} {mine} vs {first.manifest.artifact_id} {theirs}")
        variables |= set(t.manifest.variable_ids)
        if VARIABLE_COL in t.manifest.columns:
            variables |= {r.get(VARIABLE_COL, "") for r in t.rows if r.get(VARIABLE_COL, "")}
    if len(variables) > 1:
        raise AnalystError("unsupported", f"different variables cannot be stacked: {sorted(variables)}")
    if "source_artifact_id" in first.manifest.columns:
        raise AnalystError("unsupported", "the tables already carry a source_artifact_id column; stacking twice is not supported")
    rows: list[dict[str, str]] = []
    for t in tables:
        rows.extend({**r, "source_artifact_id": t.manifest.artifact_id} for r in t.rows)
    columns = {**first.manifest.columns, "source_artifact_id": "the artifact each row came from (lineage)"}
    study_ids = list(dict.fromkeys(s for t in tables for s in t.manifest.study_ids))
    request_ids = list(dict.fromkeys(r for t in tables for r in t.manifest.source_request_ids))
    lineage = [f"lineage: concat_tables of {t.manifest.artifact_id} sha256={t.manifest.sha256} rows={t.manifest.row_count} complete={t.manifest.complete}" for t in tables]
    kinds = [tuple(n for n in t.manifest.quality_notes if n.startswith(ROWS_NOTE)) for t in tables]
    if len(set(kinds)) == 1:                         # GAP 1: one row unit in every table (none written: the columns say it)
        lineage += [f"{note}, from {len(tables)} stacked tables" if len(tables) > 1 else note
                    for note in kinds[0]]            # review: per-table rows of the same entries repeat their IDs
    else:
        described = [" / ".join(n[len(ROWS_NOTE):] for n in k) or ("observation units" if UNIT_COL in t.manifest.columns else "rows not recorded")
                     for t, k in zip(tables, kinds)]
        lineage.append(f"{ROWS_NOTE}stacked rows of different kinds ({' | '.join(dict.fromkeys(described))})")
    repeated = 0
    if OBS_ID_COL in columns:
        counts = Counter(r.get(OBS_ID_COL, "") for r in rows if r.get(OBS_ID_COL, ""))
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        if repeated:
            lineage.append(f"{repeated} repeated observation ID(s) across sources; numeric tools will require a duplicate rule")
    complete = all(t.manifest.complete for t in tables)
    manifest = registry.save_table(rows, kind="analysis_result", columns=columns, complete=complete, source_request_ids=request_ids,
                                   study_ids=study_ids, variable_ids=sorted(variables), measurement=first.manifest.measurement, quality_notes=lineage)
    data = {"artifact_id": manifest.artifact_id, "sources": ids, "row_count": manifest.row_count, "complete": complete, "study_ids": study_ids,
            "variable_ids": sorted(variables), "measurement": first.manifest.measurement.model_dump(), "repeated_observation_ids": repeated,
            "lineage": lineage}
    warnings = [] if complete else ["at least one source is incomplete; the stacked table is incomplete too"]
    return ToolResult(ok=True, data=data, artifact_ids=[manifest.artifact_id], complete=complete, warnings=warnings)


def _plot_list(registry: ArtifactRegistry, roster_id: str, table: Table, *, require_complete: bool) -> Table:
    """The study's plot list (an observation_units table of the table's study), for missing_report and numeric_summary's
    percent missing. An incomplete list is refused when require_complete; otherwise the caller says the number of plots is
    unknown."""
    roster = _load(registry, roster_id)
    if roster.manifest.kind != "observation_units":
        raise AnalystError("invalid_argument", f"roster {roster.manifest.artifact_id} is {roster.manifest.kind}, not observation_units")
    if require_complete and not roster.manifest.complete:
        raise AnalystError("incomplete_data", f"roster {roster.manifest.artifact_id} is not a complete collection; expected units are unknown")
    if set(roster.manifest.study_ids) != set(table.manifest.study_ids):
        raise AnalystError("invalid_argument", f"roster studies {sorted(roster.manifest.study_ids)} differ from the table's {sorted(table.manifest.study_ids)}")
    _require_column(roster, UNIT_COL, "roster unit column")
    _require_column(table, UNIT_COL, "unit column")
    return roster


def _percent_missing(table: Table, roster: Table, used: list[tuple[dict[str, str], float]], transformation: str, *,
                     label_parts: tuple[str, ...], namespace: str | None) -> tuple[dict[str, Any], Claim]:
    """Breedbase's Percent missing: the plots on the study's plot list without a valid value, over the plots on the list, x 100.
    A measurements table holds only recorded values, so only the plot list knows the plots; every listed plot counts, fillers
    included, as on Breedbase. A plot with several values counts once. A list with units of several levels (levelName plot,
    plant, ...; read without case) counts only the level the values are recorded on. Unit IDs are exact text, as in
    missing_report. Not available, with the reason and no count, whenever the table cannot show which listed plots have a value
    (Bidhan's (15)(a), 2026-10-23; the coach's guard of 2026-10-22 23:50): a filtered table, a table or plot list made from other
    tables (any lineage note: concat_tables, plot_design, filter_rows, ...; Bidhan's (22), 2026-10-24: a number only for the two
    tables as fetched), an incomplete table or plot list, more than one study, a valid value on a row without a plot ID, a row
    whose plot is not on the list, or levels that do not settle on one."""
    level_of: dict[str, str] = {}
    listed_twice = False
    for r in roster.rows:
        unit = r.get(UNIT_COL, "")
        if unit:
            level = (r.get("levelName") or "").strip().casefold()
            listed_twice = listed_twice or level_of.get(unit, level) != level
            level_of[unit] = level
    valued = {row.get(UNIT_COL, "") for row, _ in used} - {""}
    without_id = sum(1 for row, _ in used if not row.get(UNIT_COL, ""))
    off_list = sum(1 for r in table.rows if r.get(UNIT_COL, "") and r.get(UNIT_COL, "") not in level_of)
    off_list_valid = sum(1 for row, _ in used if row.get(UNIT_COL, "") and row.get(UNIT_COL, "") not in level_of)
    levels = set(level_of.values())
    measured = {level_of[u] for u in valued if u in level_of}
    listed = set(level_of) if len(levels) <= 1 else {u for u, level in level_of.items() if level in measured}
    made, list_made = _made_by(table.manifest), _made_by(roster.manifest)
    reason = None
    if _filters_applied(table.manifest):
        reason = "the table is a filtered subset: the plots it left out would count as without a value"
    elif made:
        reason = (f"the table was made by {', '.join(made)} from other tables, not fetched from the server: percent missing needs "
                  "the measurements table as fetched")
    elif list_made:
        reason = (f"the plot list {roster.manifest.artifact_id} was made by {', '.join(list_made)}, not fetched from the server: "
                  "percent missing needs the study's plot list as fetched")
    elif not table.manifest.complete:
        reason = "the measurements table is incomplete: plots never fetched would count as without a value"
    elif not roster.manifest.complete:
        reason = f"the plot list {roster.manifest.artifact_id} is incomplete, so the number of plots is unknown"
    elif len(set(table.manifest.study_ids) | set(roster.manifest.study_ids)) != 1:
        reason = "the table and the plot list are not one single study"
    elif not level_of:
        reason = f"the plot list {roster.manifest.artifact_id} holds no unit"
    elif without_id:
        reason = "valid values on rows without a plot ID cannot be matched to the plot list"
    elif off_list:
        reason = "the table holds plots that are not on the plot list, so the list is not the table's"
    elif listed_twice:
        reason = "a unit is listed under two levels on the plot list"
    elif "" in levels and len(levels) > 1:
        reason = "the plot list holds units without a stated level beside units with one"
    elif len(levels) > 1 and len(measured) != 1:
        reason = (f"the plot list holds units of several levels ({', '.join(sorted(levels))}) and the values "
                  f"{'are recorded on several of them' if measured else 'show none of them'}: percent missing needs one level")
    with_value = len(valued & listed)
    without = len(listed) - with_value
    value = None if reason else 100.0 * without / len(listed)
    level = next(iter(measured)) if len(levels) > 1 and len(measured) == 1 else None
    method = (f"{PERCENT_MISSING} = plots on the plot list {roster.manifest.artifact_id} without a valid value / plots on the list x 100 "
              f"({without} of {len(listed)}; every listed plot counts, fillers included, as on Breedbase)" if reason is None
              else f"{PERCENT_MISSING} with the plot list {roster.manifest.artifact_id}: not available")
    claim = Claim(claim_id=_claim_id(table.manifest.artifact_id, "ratio", "missing", "list-" + roster.manifest.artifact_id[4:].replace("_", "-"),
                                     *label_parts, namespace=namespace), kind="ratio", value=value, unit="%", missing_reason=reason,
                  denominator=Denominator(name=("plots on the study's plot list" + (f" (level {level})" if level else "")) if reason is None
                                          else "plots read from the plot list", n=len(listed)),
                  n_independent_units=None if reason else with_value, source_artifact_hashes=[table.manifest.sha256, roster.manifest.sha256],
                  filters={}, grouping=[], transformation=f"{method}; {transformation}", exclusions=[])
    data = {"value": value, "plots_listed": len(listed), "with_valid_value": with_value, "without_valid_value": without,
            "rows_not_on_list": off_list, "valid_values_not_on_list": off_list_valid, "rows_without_plot_id": without_id, "level": level,
            "reason": reason, "roster_artifact_id": roster.manifest.artifact_id}
    return data, claim


def tool_missing_report(registry: ArtifactRegistry, a: MissingReportArgs) -> ToolResult:
    """Raw-missing, invalid and valid counts for the value column, overall and per group. The denominator is rows
    unless a verified observation_units roster for the same study is supplied; then it also reports expected units,
    units with a valid value, units with only missing/invalid values, units absent from the table, and rows whose
    unit is not on the roster. Missing is never turned into zero."""
    table = _load(registry, a.artifact_id)
    _require_column(table, a.value, "value column")
    c = _classify(table, a.value)
    data: dict[str, Any] = {
        "artifact_id": table.manifest.artifact_id, "complete": table.manifest.complete,
        "overall": {"total_rows": c.n_rows, "valid": c.n_valid, "raw_missing": c.n_missing, "invalid_or_nonfinite": c.n_invalid},
        "denominators": {"rows": c.n_rows}, "value_policy": VALUE_POLICY,
    }
    if a.by is not None:
        _require_column(table, a.by, "group column")
        groups: dict[str, Counter] = {}
        for row in table.rows:
            groups.setdefault(row.get(a.by, "") or MISSING_GROUP, Counter())[classify_value(row.get(a.value))[0]] += 1
        data["by"] = a.by
        data["null_group_policy"] = f"rows with an empty {a.by} are counted under {MISSING_GROUP!r}, never dropped"
        data["groups"] = [{"group_id": g, "total_rows": sum(k.values()), "valid": k["valid"], "raw_missing": k["missing"], "invalid_or_nonfinite": k["invalid"]}
                          for g, k in sorted(groups.items())]
    if a.roster_artifact_id is None:
        data["expected_units"] = None
        data["expected_units_note"] = "no verified unit roster supplied; the only denominator is rows"
    else:
        roster = _plot_list(registry, a.roster_artifact_id, table, require_complete=True)
        expected = {r.get(UNIT_COL, "") for r in roster.rows if r.get(UNIT_COL, "")}
        units_valid = {r.get(UNIT_COL, "") for r, _ in c.valid if r.get(UNIT_COL, "")}
        units_present = {r.get(UNIT_COL, "") for r in table.rows if r.get(UNIT_COL, "")}
        off_roster_rows = sum(1 for r in table.rows if r.get(UNIT_COL, "") and r.get(UNIT_COL, "") not in expected)
        data["expected_units"] = len(expected)
        data["units"] = {"with_valid_value": len(units_valid & expected), "present_without_valid_value": len((units_present - units_valid) & expected),
                         "absent_from_table": len(expected - units_present), "rows_with_unit_not_on_roster": off_roster_rows,
                         "absent_ids": sorted(expected - units_present)[:20]}
        data["denominators"]["expected_units"] = len(expected)
        data["roster_artifact_id"] = roster.manifest.artifact_id
    return ToolResult(ok=True, data=data, artifact_ids=[table.manifest.artifact_id] + ([a.roster_artifact_id] if a.roster_artifact_id else []),
                      complete=table.manifest.complete)


def tool_plot_bar(registry: ArtifactRegistry, a: PlotBarArgs) -> ToolResult:
    """Optional bar chart of a registered result. Plotting is NOT enabled in this build: matplotlib is not an approved
    dependency, so this tool returns unavailable and installs nothing. Use group_stats / count_records tables instead."""
    if not PLOTTING_ENABLED:
        return ToolResult(ok=False, error=ToolError(code="unsupported", message="plotting is unavailable: matplotlib is not an approved "
                                                                                "dependency of part2 and nothing was installed"),
                          data={"plotting_enabled": False, "requested": {"artifact_id": a.artifact_id, "x": a.x, "y": a.y, "top_n": a.top_n}})
    raise AnalystError("unsupported", "plotting is enabled by flag but not implemented in this build")   # pragma: no cover


def tool_count_records(registry: ArtifactRegistry, a: CountRecordsArgs) -> ToolResult:
    """Exact counts from a COMPLETE table (an incomplete one is refused): total rows, or value frequencies of one
    column (group_by), or the number of distinct non-empty values of one column (distinct_by). IDs are compared as
    exact text ('007' is not '7'). Empty cells form their own explicit group and are never dropped."""
    table = _load(registry, a.artifact_id)
    if not table.manifest.complete:
        raise AnalystError("incomplete_data", f"{table.manifest.artifact_id} is not a complete collection; a count from it would not be a count of the catalog")
    for column in dict.fromkeys(c for c in (a.group_by, a.distinct_by) if c in PROGRAM_COLUMNS and c in table.manifest.columns):
        if table.manifest.kind in {"studies", "variables", "locations", "programs", "seasons"}:   # as filter_rows (NEXT 4 review)
            unknown = sum(_unknown_cell(column, row.get(column, "")) for row in table.rows)
            if unknown:
                raise AnalystError("incomplete_data", f"{unknown} of {len(table.rows)} catalog rows have no {column}; a count of programs "
                                   "from them is not known (missing is not zero). Ask for a recorded field or more metadata.")
    hashes = [table.manifest.sha256]
    denominator = Denominator(name=_rows_label(table.manifest), n=table.manifest.row_count)   # rows, after any filter
    claims: list[Claim] = []
    data: dict[str, Any] = {"artifact_id": table.manifest.artifact_id, "kind": table.manifest.kind, "total_rows": table.manifest.row_count,
                            "null_group_policy": f"empty cells are counted under {MISSING_GROUP!r}, never dropped", "value_policy": VALUE_POLICY["ids"]}
    if a.group_by is None and a.distinct_by is None:
        claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, "count", "rows"), kind="count", value=table.manifest.row_count,
                            denominator=denominator, source_artifact_hashes=hashes, transformation="row count of a complete table"))
    if a.group_by is not None:
        _require_column(table, a.group_by, "group_by column")
        counts = Counter(r.get(a.group_by, "") or MISSING_GROUP for r in table.rows)
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        data["group_by"] = a.group_by
        data["frequencies"] = [{"value": k, "count": n, "rank": 1 + sum(1 for _, m in ordered if m > n)} for k, n in ordered]
        ties = [sorted(k for k, m in ordered if m == n) for n in sorted({m for _, m in ordered})]
        data["ties"] = [t for t in ties if len(t) > 1]
        data["tie_rule"] = "equal counts share a rank; order inside a tie is by value text"
        for k, n in ordered:
            claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, "count", a.group_by, k, namespace="frequency"), kind="count", value=n, denominator=denominator,
                                source_artifact_hashes=hashes, filters={a.group_by: k}, grouping=[a.group_by], transformation="frequency of exact text values"))
    if a.distinct_by is not None:
        _require_column(table, a.distinct_by, "distinct_by column")
        values = [r.get(a.distinct_by, "") for r in table.rows]
        distinct = sorted({v for v in values if v})
        data["distinct_by"] = a.distinct_by
        data["distinct_count"] = len(distinct)
        data["empty_cells"] = sum(1 for v in values if not v)
        data["distinct_values_preview"] = distinct[:20]
        claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, "count", "distinct", a.distinct_by), kind="count", value=len(distinct),
                            denominator=denominator, source_artifact_hashes=hashes, grouping=[a.distinct_by],
                            transformation="number of distinct non-empty exact text values",
                            exclusions=[Exclusion(reason="empty cells", count=data["empty_cells"])] if data["empty_cells"] else []))
    data["claims"] = _claims_json(claims)
    return ToolResult(ok=True, data=data, artifact_ids=[table.manifest.artifact_id], complete=True)


# R1 (Bidhan, 2026-10-21): the highest, the lowest, the top k or the rank of a named entry, for one measured trait, is a fact the
# app may state, as an ordering of recorded values; "best", "select", "advance" and "recommend" stay refused elsewhere.
ORDERING_NOTE = "descriptive ordering of recorded values, not a recommendation"
RANK_RESULT_COLUMNS = {**GROUP_RESULT_COLUMNS, "rank": "rank among the ranked entries (ties share a rank; '' = not ranked)",
                       "left_out": "true when the entry was left out before ranking (exclude)",
                       "basis": "what mean is: pooled = the mean of the entry's valid values; study_means = the unweighted mean of its "
                                "per-study means (median, min, max and sd stay over its values)",
                       "n_studies": "studies behind the mean under study_means ('' under pooled)"}


def _entries(table: Table, by: str, label_col: str | None, wanted: list[str], what: str) -> list[str]:
    """Exact entry IDs for entries given by exact ID or exact name; an unknown name is refused with the closest ones."""
    ids = sorted({row.get(by, "") for row in table.rows} - {""})
    by_label: dict[str, set[str]] = {}
    if label_col:
        for row in table.rows:
            if row.get(label_col) and row.get(by):
                by_label.setdefault(row[label_col], set()).add(row[by])
    found = []
    for name in wanted:
        if name in ids and by_label.get(name, {name}) - {name}:
            raise AnalystError("invalid_argument", f"{what} {name!r} is the ID of one entry and the name of another "
                                                   f"({sorted(by_label[name] - {name})}); give an ID that is not also a name")
        if name in ids:
            found.append(name)
        elif len(by_label.get(name, ())) == 1:
            found.append(next(iter(by_label[name])))
        elif name in by_label:
            raise AnalystError("invalid_argument", f"{what} {name!r} is the name of {len(by_label[name])} entries "
                                                   f"({sorted(by_label[name])}); give the entry ID")
        else:
            close = difflib.get_close_matches(name, ids + sorted(by_label), n=3, cutoff=0.6)
            raise AnalystError("invalid_argument", f"{what} {name!r} is not an entry ID or name in {table.manifest.artifact_id}"
                                                   + (f"; the closest are {close}" if close else ""))
    return list(dict.fromkeys(found))


def tool_rank_entries(registry: ArtifactRegistry, a: RankEntriesArgs) -> ToolResult:
    """The order of entries by their mean for ONE trait: the highest (or lowest) top_k entries, and the rank of each named entry
    wherever it falls, after leaving out the excluded entries (for example the checks). Each stated entry gets a mean claim and
    a rank claim (n = the entries ranked; ties share a rank, order inside a tie by entry ID; entries tied with the last of the
    top_k are stated too). Entries are given by exact ID or exact name; an unknown name is refused with the closest ones. The
    full ranking (every entry, ranked or not) is saved as an analysis_result artifact. A descriptive ordering of recorded
    values, not a recommendation."""
    table = _load(registry, a.artifact_id)
    _require_column(table, a.by, "entry column")
    _require_column(table, a.value, "value column")
    if a.by.endswith("Name"):
        raise AnalystError("invalid_argument", f"rank by the ID column, not {a.by!r}; names are carried as labels")
    caveat = _completeness_gate(table, a.allow_incomplete)
    variable = _single_variable(table)
    label_col = LABEL_FOR.get(a.by)
    label_col = label_col if label_col in table.manifest.columns else None
    left_out = set(_entries(table, a.by, label_col, a.exclude, "exclude"))
    named = _entries(table, a.by, label_col, a.named, "named")
    both = sorted(left_out & set(named))
    if both:
        raise AnalystError("invalid_argument", f"{both} are both named and left out; an entry is either ranked or left out")
    classified = _classify(table, a.value)
    used, transformation = _apply_duplicate_policy(table, classified, a.duplicate_policy, value=a.value)
    if caveat:
        transformation = f"{caveat}; {transformation}"
    unit = table.manifest.measurement.unit
    results, no_group_rows = _group_results(table, a.by, a.value, classified, used, a.min_independent_n, label_col)
    studies: list[str] = []
    if a.basis == "study_means":                                  # coach 2026-10-22 23:50 UTC (T3): each study weighted equally
        _require_column(table, "studyDbId", "study column")
        studies = sorted({row.get("studyDbId", "") for row, _ in used} - {""})      # studies with a valid value (review 1)
        per_entry: dict[str, dict[str, list[float]]] = {}
        for row, number in used:
            if row.get(a.by, "") and row.get("studyDbId", ""):
                per_entry.setdefault(row[a.by], {}).setdefault(row["studyDbId"], []).append(number)
        for g in results:
            study_means = per_entry.get(g["group_id"], {})
            g["n_studies"] = len(study_means)
            if study_means:                                       # the saved table keeps one basis for every entry (review 1)
                g["mean"] = statistics.fmean(statistics.fmean(values) for values in study_means.values())
            if g["included"] and len(study_means) < len(studies):
                g["included"] = False
                g["exclusion_reason"] = f"not in every study of the table ({len(study_means)} of {len(studies)})"
    by_study = a.basis == "study_means"
    measure = "the mean of per-study entry means" if by_study else "entry mean"

    first = "highest" if a.order == "highest" else "lowest"
    ranked = [g for g in results if g["included"] and g["group_id"] not in left_out]
    # means equal at the precision the answer shows are equal: 0.1 and 0.2 average to 0.15000000000000002, 0.15 and 0.15 to
    # 0.15, and both entries must share a rank (review 2026-10-21)
    tie = {g["group_id"]: float(format(g["mean"], ".12g")) for g in ranked}
    ranked.sort(key=lambda g: ((-tie[g["group_id"]] if a.order == "highest" else tie[g["group_id"]]), g["group_id"]))
    for g in ranked:                                              # competition ranking: ties share a rank
        g["rank"] = 1 + sum(1 for o in ranked if (tie[o["group_id"]] > tie[g["group_id"]] if a.order == "highest"
                                                  else tie[o["group_id"]] < tie[g["group_id"]]))
    for g in results:
        g.setdefault("rank", None)
    cut = ranked[min(a.top_k, len(ranked)) - 1]["rank"] if ranked else 0
    stated = [g for g in ranked if g["rank"] <= cut or g["group_id"] in named]
    by_id = {g["group_id"]: g for g in results}
    named_not_ranked = [{"entry": n, "label": by_id[n]["label"], "reason": by_id[n]["exclusion_reason"]} for n in named if by_id[n]["rank"] is None]
    not_ranked = [g for g in results if not g["included"] and g["group_id"] not in left_out]

    lineage = [f"lineage: rank_entries by {a.by} on {table.manifest.artifact_id} sha256={table.manifest.sha256}",
               f"{ROWS_NOTE}rank_entries by {a.by}, basis {a.basis}",         # one row per entry; the basis says what mean is (review)
               f"transformation: {transformation}", f"order: {first} {measure} first; ties share a rank", f"basis: {a.basis}",
               f"left out before ranking: {sorted(left_out) or 'none'}", f"min_independent_n={a.min_independent_n} (exclusion threshold)",
               f"rows with an empty {a.by}: {no_group_rows} (excluded, not dropped silently)", f"analysis target: {ANALYSIS_TARGET}"]
    text_rows = [{**g, "included": "true" if g["included"] else "false", "left_out": "true" if g["group_id"] in left_out else "false",
                  "basis": a.basis, "n_studies": "" if g.get("n_studies") is None else str(g["n_studies"]),
                  "n_independent_units": "" if g["n_independent_units"] is None else str(g["n_independent_units"]),
                  "rank": "" if g["rank"] is None else str(g["rank"]),
                  **{k: ("" if g[k] is None else repr(g[k])) for k in ("mean", "median", "min", "max", "sd")}} for g in results]
    manifest = registry.save_table(text_rows, kind="analysis_result", columns=RANK_RESULT_COLUMNS, complete=table.manifest.complete,
                                   source_request_ids=list(table.manifest.source_request_ids), study_ids=list(table.manifest.study_ids),
                                   variable_ids=list(table.manifest.variable_ids), measurement=table.manifest.measurement, quality_notes=lineage)

    exclusions = _exclusions(classified)
    if no_group_rows:
        exclusions.append(Exclusion(reason=f"rows with an empty {a.by}", count=no_group_rows))
    order_exclusions = list(exclusions)
    if left_out:
        order_exclusions.append(Exclusion(reason="entries left out before ranking", count=len(left_out)))
    absent = sum(1 for g in not_ranked if g["exclusion_reason"].startswith("not in every study"))     # each reason counted (review 1)
    for reason, count in ((f"entries not ranked: fewer than {a.min_independent_n} independent units", len(not_ranked) - absent),
                          ("entries not ranked: not in every study of the table", absent)):
        if count:
            order_exclusions.append(Exclusion(reason=reason, count=count))
    # one namespace per ranking: a second call with other entries left out ranks differently, so its claims must not share IDs
    variant = hashlib.sha256(json.dumps([a.order, sorted(left_out), a.min_independent_n, a.duplicate_policy, a.allow_incomplete,
                                         a.value] + ([a.basis] if by_study else [])).encode("utf-8")).hexdigest()[:10]
    averaged = len(used) != len(classified.valid)
    claims: list[Claim] = []
    for g in stated:
        if by_study:
            denominator = Denominator(name=f"study means of entry {g['group_id']} (each the mean of its valid values in one study)", n=g["n_studies"])
            mean_text = (f"entry mean = the unweighted mean of its per-study means over the {len(studies)} studies of the table (each study "
                         f"weighted equally; entries in every study); {transformation}")
        else:
            denominator = Denominator(name=(f"observation units in entry {g['group_id']} ({PER_UNIT})" if averaged else
                                            f"valid numeric values in entry {g['group_id']}" + ("" if classified.units_known else " (rows; independence unknown)")),
                                      n=g["n_valid_values"])
            mean_text = f"entry mean of valid values; {transformation}"
        claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, "mean", a.by, g["group_id"], namespace=f"rank-{variant}"),
                            kind="mean", value=g["mean"], unit=unit, denominator=denominator, n_independent_units=g["n_independent_units"],
                            source_artifact_hashes=[table.manifest.sha256], filters={a.by: g["group_id"]}, grouping=[a.by],
                            transformation=mean_text, exclusions=exclusions))
        claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, "rank", a.by, g["group_id"], namespace=f"rank-{variant}"),
                            kind="rank", value=g["rank"], denominator=Denominator(name=f"entries ranked by {measure}, {first} first", n=len(ranked)),
                            source_artifact_hashes=[table.manifest.sha256], filters={a.by: g["group_id"]}, grouping=[a.by],
                            transformation=(f"rank among {len(ranked)} entries by {measure}, {first} first; ties share a rank; "
                                            f"{ORDERING_NOTE}; {transformation}"), exclusions=order_exclusions))
    data = {
        "artifact_id": table.manifest.artifact_id, "result_artifact_id": manifest.artifact_id, "variable_id": variable, "unit": unit,
        "by": a.by, "label_column": label_col, "order": f"{first} first", "entries_ranked": len(ranked), "basis": a.basis, "studies": studies,
        "left_out": [{"entry": e, "label": by_id[e]["label"]} for e in sorted(left_out)],
        "not_ranked": [{"entry": g["group_id"], "label": g["label"], "reason": g["exclusion_reason"]} for g in not_ranked],
        "named_not_ranked": named_not_ranked,
        "stated": [{"entry": g["group_id"], "label": g["label"], "rank": g["rank"], "mean": g["mean"], "display_mean": _display(g["mean"]),
                    "n_valid_values": g["n_valid_values"], "n_independent_units": g["n_independent_units"]} for g in stated],
        "tie_rule": "equal means share a rank; order inside a tie is by entry ID; entries tied with the last of top_k are stated",
        "transformation": transformation, "min_independent_n": a.min_independent_n, "claims": _claims_json(claims),
        "exclusions": [e.model_dump() for e in order_exclusions], "analysis_target": ANALYSIS_TARGET, "ordering": ORDERING_NOTE,
        "value_policy": VALUE_POLICY,
    }
    warnings = [w for w in (caveat, classified.duplicate_note if a.duplicate_policy != "reject" else None) if w]
    if not classified.units_known:
        warnings.append("independence unknown: min_independent_n was applied to valid values, not to units")
    warnings.extend(f"named entry {n['entry']} ({n['label'] or 'no name'}) is not ranked: {n['reason']}" for n in named_not_ranked)
    warnings.extend(_timing_warnings(table, a.value))
    return ToolResult(ok=True, data=data, artifact_ids=[table.manifest.artifact_id, manifest.artifact_id], complete=table.manifest.complete,
                      warnings=warnings)


# R2 (Bidhan 2026-10-21; coach 2026-10-22 10:52 UTC): how many entries or plots are above or below a threshold, in each study.
# Measured first (r2_need_20261022): five benchmark sub-questions need it (A36.5, A42.5, C31.3, C32.3, C32.4), and in all five the
# threshold is the mean of named entries (Beauregard; a site's Dagga; two checks averaged), so the tool computes that mean itself:
# a rounded number copied by the model could put an entry on the wrong side of the line.
THRESHOLD_NOTE = "descriptive count of recorded values against a threshold, not a recommendation"
THRESHOLD_NAMESPACE = "thr"
_THRESHOLD_OPS: dict[str, tuple[str, Callable[[float, float], bool]]] = {
    "above": ("above", lambda x, t: x > t), "below": ("below", lambda x, t: x < t),
    "at_least": ("at least", lambda x, t: x >= t), "at_most": ("at most", lambda x, t: x <= t)}
THRESHOLD_RESULT_COLUMNS = {
    "studyDbId": "the study", "entry": "exact entry ID", "label": "name carried from the label column, if any",
    "unit": "the plot (observation unit) of a counted plot row, when plots are counted; '' on an entry row",
    "role": "threshold entry / excluded / outside the name prefix / no valid value / counted",
    "side": "for a counted row: the op word when it meets the threshold, else 'not ' + the op word ('' otherwise)",
    "value": "the entry mean, full precision ('' when none); on a counted plot row, that plot's own value",
    "n_valid_values": "valid values behind the entry mean (1 on a plot row)", "threshold": "the study's threshold, full precision",
}


def _shown(number: float) -> float:
    """Numbers equal at the precision the answer shows are equal (rank_entries' tie rule): 0.1 and 0.2 average to
    0.15000000000000002, which must not count as above 0.15."""
    return float(format(number, ".12g"))


def tool_threshold_count(registry: ArtifactRegistry, a: ThresholdCountArgs) -> ToolResult:
    """How many entries (each by its mean over its plots) or plots (each by its own value) of ONE trait are above, below, at
    least or at most a threshold, in each study of the table, and the share they are of those with a valid value. The threshold
    is a number from the question, or the mean of named entries (exact ID or exact name: a check, or two checks averaged
    unweighted), computed here for each study; the named entries and the excluded ones are left out of the count, and
    name_prefix keeps one family only (PH17.). A value equal to the threshold is not above it. Each study gets a count claim,
    a share claim (%) and, for named entries, their mean as the threshold claim; every entry (and every counted plot, when
    plots are counted) is saved with its role as an analysis_result artifact. Plots mode counts each plot once: a plot with
    values at several times is refused, and a row without a plot ID is left out and counted. A descriptive count of recorded
    values, not a recommendation."""
    table = _load(registry, a.artifact_id)
    _require_column(table, a.by, "entry column")
    _require_column(table, a.value, "value column")
    _require_column(table, "studyDbId", "study column")
    if a.by.endswith("Name"):
        raise AnalystError("invalid_argument", f"count by the ID column, not {a.by!r}; names are carried as labels")
    if a.count == "plots":
        _require_column(table, UNIT_COL, "plot column")
    caveat = _completeness_gate(table, a.allow_incomplete)
    variable = _single_variable(table)
    label_col = LABEL_FOR.get(a.by)
    label_col = label_col if label_col in table.manifest.columns else None
    if a.name_prefix is not None and label_col is None:
        raise AnalystError("invalid_argument", f"name_prefix needs a name column beside {a.by!r}; {table.manifest.artifact_id} has none")
    threshold_ids = _entries(table, a.by, label_col, a.threshold_entries, "threshold entry")
    excluded = set(_entries(table, a.by, label_col, a.exclude, "exclude"))
    both = sorted(excluded & set(threshold_ids))
    if both:
        raise AnalystError("invalid_argument", f"{both} are both threshold entries and left out; give each entry one role")
    classified = _classify(table, a.value)
    used, transformation = _apply_duplicate_policy(table, classified, a.duplicate_policy, value=a.value)
    if caveat:
        transformation = f"{caveat}; {transformation}"
    unit = table.manifest.measurement.unit
    op_words, meets = _THRESHOLD_OPS[a.op]
    names: dict[str, set[str]] = {}                              # entry -> its names anywhere in the table
    for row in table.rows:
        if row.get(a.by, ""):
            names.setdefault(row[a.by], set())
            if label_col and row.get(label_col, ""):
                names[row[a.by]].add(row[label_col])
    label = {entry: "; ".join(sorted(found)) for entry, found in names.items()}
    rows_by_study: dict[str, list[dict[str, str]]] = {}
    for row in table.rows:
        rows_by_study.setdefault(row.get("studyDbId", ""), []).append(row)
    used_by_study: dict[str, list[tuple[dict[str, str], float]]] = {}
    for row, number in used:
        used_by_study.setdefault(row.get("studyDbId", ""), []).append((row, number))
    no_study_rows = len(rows_by_study.pop("", []))
    used_by_study.pop("", None)
    if not rows_by_study:
        raise AnalystError("invalid_argument", f"{table.manifest.artifact_id} has no row with a studyDbId")
    no_entry_rows = sum(1 for row in table.rows if not row.get(a.by, ""))           # review: such rows vanished unreported
    base_exclusions = [Exclusion(reason="rows with an empty studyDbId", count=no_study_rows)] if no_study_rows else []
    # one namespace per setting: another threshold, population or op counts differently, so its claims must not share IDs
    variant = hashlib.sha256(json.dumps([a.count, a.op, a.threshold, threshold_ids, sorted(excluded), a.name_prefix, a.by, a.value,
                                         a.duplicate_policy, a.allow_incomplete]).encode("utf-8")).hexdigest()[:10]
    namespace = f"{THRESHOLD_NAMESPACE}-{variant}"
    hashes = [table.manifest.sha256]
    studies: list[dict[str, Any]] = []
    result_rows: list[dict[str, str]] = []
    claims: list[Claim] = []
    for study in sorted(rows_by_study):
        present = list(dict.fromkeys(row[a.by] for row in rows_by_study[study] if row.get(a.by, "")))
        values: dict[str, list[tuple[dict[str, str], float]]] = {}
        for row, number in used_by_study.get(study, []):
            if row.get(a.by, ""):
                values.setdefault(row[a.by], []).append((row, number))
        means = {entry: statistics.fmean(number for _, number in pairs) for entry, pairs in values.items()}
        if a.threshold is not None:
            threshold, source, threshold_entries = a.threshold, "stated in the question", []
        else:
            missing = [t for t in threshold_ids if t not in means]
            if missing:
                raise AnalystError("invalid_argument", f"threshold entry {missing[0]} ({label.get(missing[0]) or 'no name'}) has no valid value "
                                                       f"in study {study}; the threshold needs its mean in every study of the table")
            threshold = statistics.fmean(means[t] for t in threshold_ids)
            threshold_entries = [{"entry": t, "label": label[t], "mean": means[t], "n_valid_values": len(values[t])} for t in threshold_ids]
            described = " and ".join(f"{t} ({label[t] or 'no name'})" for t in threshold_ids)
            source = f"the entry mean of {described}" if len(threshold_ids) == 1 else f"the unweighted mean of the entry means of {described}"
        outside = {e for e in present if a.name_prefix is not None and e not in threshold_ids and e not in excluded
                   and not any(n.startswith(a.name_prefix) for n in names[e])}
        population = [e for e in present if e not in threshold_ids and e not in excluded and e not in outside]
        if a.name_prefix is not None and not population:
            raise AnalystError("invalid_argument", f"no entry in study {study} has a name starting with {a.name_prefix!r}")
        counted_set = set(population)
        n_no_unit = 0
        if a.count == "entries":
            with_value = [(e, "", means[e], len(values[e])) for e in population if e in means]
            n_no_value = sum(1 for e in population if e not in means)
        else:
            plots = {e: [(row, number) for row, number in values.get(e, []) if row.get(UNIT_COL, "")] for e in population}
            n_no_unit = sum(len(values.get(e, [])) - len(plots[e]) for e in population)     # review 2: a row without a plot ID is not a plot
            repeated = sum(1 for e in population for _, n in Counter(row.get(UNIT_COL, "") for row, _ in plots[e]).items() if n > 1)
            if repeated:                                    # review: a plot measured at two times counted as two plots
                raise AnalystError("unsupported", f"{repeated} plot{'' if repeated == 1 else 's'} of the counted population in study {study} "
                                                  f"{'has' if repeated == 1 else 'have'} more than one valid value (several times); plots mode counts "
                                                  f"each plot once: use count=\"entries\", or narrow the table to one time with filter_rows on {TIMEPOINT_COL}")
            with_value = [(e, row.get(UNIT_COL, ""), number, 1) for e in population for row, number in plots[e]]
            n_no_value = sum(1 for row in rows_by_study[study]
                             if row.get(a.by, "") in counted_set and classify_value(row.get(a.value))[0] != "valid")
        cut = _shown(threshold)
        counted = [item for item in with_value if meets(_shown(item[2]), cut)]
        n_with_value, n_counted = len(with_value), len(counted)
        share = 100.0 * n_counted / n_with_value if n_with_value else None
        for e in present:
            role = ("threshold entry" if e in threshold_ids else "excluded" if e in excluded else "outside the name prefix" if e in outside
                    else "no valid value" if e not in means else "counted")
            if role == "counted" and a.count == "plots":
                items = [(row.get(UNIT_COL, ""), number, 1) for row, number in values[e]]
            else:
                items = [("", means.get(e), len(values.get(e, [])))]
            for plot, number, n_values in items:
                side = "" if role != "counted" else (op_words if meets(_shown(number), cut) else f"not {op_words}")
                result_rows.append({"studyDbId": study, "entry": e, "label": label[e], "unit": plot, "role": role, "side": side,
                                    "value": "" if number is None else repr(number), "n_valid_values": str(n_values), "threshold": repr(threshold)})
        left_out = {"threshold entries": sum(t in present for t in threshold_ids), "excluded": sum(e in present for e in excluded),
                    "outside the name prefix": len(outside), "no valid value": n_no_value, "no plot ID": n_no_unit}
        tokens = Counter(classify_value(row.get(a.value))[0] for row in rows_by_study[study])      # this study's rows only (review)
        exclusions = list(base_exclusions)
        if tokens["missing"]:
            exclusions.append(Exclusion(reason="raw-missing value token", count=tokens["missing"]))
        if tokens["invalid"]:
            exclusions.append(Exclusion(reason="invalid or non-finite value token", count=tokens["invalid"]))
        study_no_entry = sum(1 for row in rows_by_study[study] if not row.get(a.by, ""))
        if study_no_entry:
            exclusions.append(Exclusion(reason=f"rows with an empty {a.by}", count=study_no_entry))
        for reason, n in left_out.items():
            if n:
                exclusions.append(Exclusion(reason=f"{a.count if reason in ('no valid value', 'no plot ID') else 'entries'} left out: {reason}", count=n))
        units = len({row.get(UNIT_COL, "") for e in population for row, _ in values.get(e, [])}) if classified.units_known else None
        filters = {"studyDbId": study}
        text = (f"count of {a.count} {op_words} the threshold {_display(threshold)} {unit or '(unit not stated)'}; threshold = {source}; "
                f"{THRESHOLD_NOTE}; each {'entry mean' if a.count == 'entries' else 'value'} is compared with the threshold at 12 significant "
                f"digits (equal is not above); {transformation}")
        denominator = Denominator(name=f"{a.count} with a valid value in study {study}", n=n_with_value)
        claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, "count", "studyDbId", study, namespace=namespace), kind="count",
                            value=n_counted, denominator=denominator, n_independent_units=units, source_artifact_hashes=hashes, filters=filters,
                            grouping=["studyDbId"], transformation=text, exclusions=exclusions))
        claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, "ratio", "studyDbId", study, namespace=namespace), kind="ratio",
                            value=share, unit="%", missing_reason=None if share is not None else f"no {a.count} with a valid value",
                            denominator=denominator, n_independent_units=units, source_artifact_hashes=hashes, filters=filters,
                            grouping=["studyDbId"], transformation=f"share of the {a.count} with a valid value, in percent; {text}", exclusions=exclusions))
        if threshold_entries:
            claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, "mean", "studyDbId", study, namespace=namespace), kind="mean",
                                value=threshold, unit=unit, source_artifact_hashes=hashes, filters=filters, grouping=["studyDbId"],
                                denominator=Denominator(name="threshold entries whose entry means were averaged (unweighted)", n=len(threshold_ids)),
                                transformation=f"threshold = {source}; entry mean of valid values; {transformation}", exclusions=base_exclusions))
        studies.append({"studyDbId": study, "threshold": threshold, "display_threshold": _display(threshold), "threshold_source": source,
                        "threshold_entries": threshold_entries, "n_counted": n_counted, "n_with_value": n_with_value, "share_percent": share,
                        "left_out": left_out, "rows_with_an_empty_entry_id": study_no_entry,
                        "counted_preview": [item[0] if a.count == "entries" else item[1] for item in counted[:20]]})
    lineage = [f"lineage: threshold_count ({a.count} {a.op}) on {table.manifest.artifact_id} sha256={table.manifest.sha256}",
               f"transformation: {transformation}",
               "threshold: " + (f"the number {a.threshold!r}" if a.threshold is not None else f"the mean of the entries {threshold_ids}, per study"),
               f"left out: threshold entries {threshold_ids}, excluded {sorted(excluded)}, name_prefix {a.name_prefix!r}",
               f"rows with an empty studyDbId: {no_study_rows}; rows with an empty {a.by}: {no_entry_rows} (excluded, not dropped silently)",
               f"analysis target: {ANALYSIS_TARGET}",
               f"{ROWS_NOTE}threshold_count result: one row per studyDbId and {a.by} (value: the entry mean)"   # coach 15:49 UTC
               + (", but a counted entry has one row per value used (its plot in unit; a value with no plot ID is not counted)"
                  if a.count == "plots" else "")]
    manifest = registry.save_table(result_rows, kind="analysis_result", columns=THRESHOLD_RESULT_COLUMNS, complete=table.manifest.complete,
                                   source_request_ids=list(table.manifest.source_request_ids), study_ids=list(table.manifest.study_ids),
                                   variable_ids=list(table.manifest.variable_ids), measurement=table.manifest.measurement, quality_notes=lineage)
    data = {"artifact_id": table.manifest.artifact_id, "result_artifact_id": manifest.artifact_id, "variable_id": variable, "unit": unit,
            "count": a.count, "op": a.op, "by": a.by, "label_column": label_col, "studies": studies, "claims": _claims_json(claims),
            "rows_with_an_empty_entry_id": no_entry_rows, "rows_with_an_empty_study_id": no_study_rows,
            "transformation": transformation, "analysis_target": ANALYSIS_TARGET, "note": THRESHOLD_NOTE, "value_policy": VALUE_POLICY}
    warnings = [w for w in (caveat, classified.duplicate_note if a.duplicate_policy != "reject" else None) if w]
    if not classified.units_known:
        warnings.append("independence unknown: some valid rows name no observation unit")
    warnings.extend(_timing_warnings(table, a.value))
    return ToolResult(ok=True, data=data, artifact_ids=[table.manifest.artifact_id, manifest.artifact_id], complete=table.manifest.complete,
                      warnings=warnings)


# R2 (Bidhan 2026-10-21; coach 2026-10-22 10:52 UTC next 2): the Pearson / Spearman correlation of two tables' entry means, joined
# by entry. Measured first (r2_need_20261022): 15 benchmark sub-questions need it, 8 as one trait in two studies (C27.4, C29.2-5,
# C31.5, C33.3, C34.2) and 7 as two traits of the same studies (E13.1-4, E18.1, E18.3, E18.4). Entry means first: per study, then
# the unweighted mean over the studies, as those questions define them. No p-value is stated.
CORRELATION_NOTE = "descriptive correlation of recorded entry means, not a recommendation"
CORRELATION_RESULT_COLUMNS = {
    "entry": "exact entry ID", "label": "name carried from the label column, if any",
    "paired": "true when the entry has a mean in both tables and is correlated",
    "left_out": "why an entry is not paired ('' when paired): only in x, only in y, excluded, outside the name prefix, not in every study of x / y",
    "x_mean": "entry mean in the first table, full precision ('' = none)", "x_n_values": "valid values behind x_mean", "x_n_studies": "studies x_mean averages",
    "y_mean": "entry mean in the second table, full precision ('' = none)", "y_n_values": "valid values behind y_mean", "y_n_studies": "studies y_mean averages",
    "x_rank": "rank of x_mean among the paired entries, lowest first; tied means share the average rank ('' when not paired)",
    "y_rank": "rank of y_mean among the paired entries, the same way",
}


@dataclass
class _EntryMeans:
    table: Table
    variable: str
    means: dict[str, tuple[float, int, int]]        # entry -> (mean, valid values behind it, studies it averages)
    labels: dict[str, set[str]]                     # entry -> its names in this table
    studies: list[str]
    not_in_every_study: int
    no_study_rows: int
    transformation: str
    caveat: str | None
    classified: Classified
    dropped: set[str]                               # entries without a mean under entries=in_every_study (review: saved and named)
    manifest_only: list[str]                        # studies the manifest lists that have no row (review: said, not ignored)
    peaks: dict[str, float]                         # entry -> the largest absolute RAW value behind its mean, before any averaging (T4)
    no_entry_values: int                            # valid values with a study but no entry ID (T4 review F2: counted, not dropped)


def _entry_means(table: Table, a: CorrelateEntryMeansArgs, label_col: str | None) -> _EntryMeans:
    """Each entry's mean in one table: the mean of its valid values per study, then the unweighted mean over its studies (all of
    the table's studies under entries=in_every_study; otherwise those where it has a value)."""
    _require_column(table, a.by, "entry column")
    _require_column(table, a.value, "value column")
    _require_column(table, "studyDbId", "study column")
    caveat = _completeness_gate(table, a.allow_incomplete)
    variable = _single_variable(table)
    classified = _classify(table, a.value)
    big = max((abs(number) for _, number in classified.valid), default=0.0)
    if big >= 1e200:                                      # review: sums of such values overflow inside a mean (also when duplicates are averaged); refuse, never crash
        raise AnalystError("unsupported", f"{table.manifest.artifact_id} holds a value of 1e200 or more in absolute value ({big!r}); "
                                          "entry means of such values are not computed")
    used, transformation = _apply_duplicate_policy(table, classified, a.duplicate_policy, value=a.value)
    if caveat:
        transformation = f"{caveat}; {transformation}"
    labels: dict[str, set[str]] = {}
    studies: set[str] = set()
    no_study_rows = 0
    for row in table.rows:
        if not row.get("studyDbId", ""):
            no_study_rows += 1
            continue
        studies.add(row["studyDbId"])
        if row.get(a.by, ""):
            labels.setdefault(row[a.by], set())
            if label_col and row.get(label_col, ""):
                labels[row[a.by]].add(row[label_col])
    per_study: dict[str, dict[str, list[float]]] = {}
    peaks: dict[str, float] = {}
    no_entry_values = 0
    for row, number in used:
        if row.get(a.by, "") and row.get("studyDbId", ""):
            per_study.setdefault(row[a.by], {}).setdefault(row["studyDbId"], []).append(number)
        elif row.get("studyDbId", ""):
            no_entry_values += 1
    for row, number in classified.valid:                     # the raw values: mean_per_unit's plot averages can be residue (T4 review 2, F1)
        if row.get(a.by, "") and row.get("studyDbId", ""):
            peaks[row[a.by]] = max(peaks.get(row[a.by], 0.0), abs(number))
    means: dict[str, tuple[float, int, int]] = {}
    not_in_every_study = 0
    dropped: set[str] = set()
    for entry in sorted(per_study):
        study_means = {study: statistics.fmean(values) for study, values in per_study[entry].items()}
        if getattr(a, "entries", "in_every_study") == "in_every_study" and len(study_means) < len(studies):   # T4: one study a side
            not_in_every_study += 1
            dropped.add(entry)
            continue
        means[entry] = (statistics.fmean(study_means.values()), sum(len(v) for v in per_study[entry].values()), len(study_means))
    manifest_only = sorted(set(table.manifest.study_ids) - studies)
    return _EntryMeans(table, variable, means, labels, sorted(studies), not_in_every_study, no_study_rows, transformation, caveat, classified,
                       dropped, manifest_only, peaks, no_entry_values)


def _tie_key(scaled: float) -> float:
    """The value two entry means are compared by, after each side was divided by its largest absolute value: anything under 1e-13 of
    that scale is a float residue and counts as 0 (0.1 + 0.2 - 0.3 averages to 9e-18, which must tie with an exact 0; review), and
    the rest tie at 12 significant digits (the project's rule)."""
    return 0.0 if abs(scaled) < 1e-13 else float(format(scaled, ".12g"))


def _average_ranks(values: list[float]) -> list[float]:
    """Ranks from 1 (the lowest) of values already divided by the side's largest absolute value; values with the same tie key share
    the average of their ranks."""
    keyed = [_tie_key(v) for v in values]
    order = sorted(range(len(values)), key=lambda i: keyed[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and keyed[order[j + 1]] == keyed[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _table_text(side: str, e: _EntryMeans) -> str:
    unit = e.table.manifest.measurement.unit
    trait = e.table.manifest.measurement.trait or e.variable
    listed = " and ".join(e.studies) if len(e.studies) <= 2 else ", ".join(e.studies)
    return f"{side}: {trait}{f' ({unit})' if unit else ''} in {'study' if len(e.studies) == 1 else 'studies'} {listed}; {len(e.means)} entries with a mean"


def tool_correlate_entry_means(registry: ArtifactRegistry, a: CorrelateEntryMeansArgs) -> ToolResult:
    """The Pearson correlation (r) and the Spearman rank correlation (rho) between the entry means of two tables, joined by
    entry: ONE trait in two studies, or TWO traits of the same studies. An entry's mean in a table is the mean of its valid
    values per study, then the unweighted mean over the studies (over the entries present in every study of the table unless
    entries=in_any_study). Named entries can be left out (exclude) and one family kept (name_prefix). n = the entries with a
    mean in both tables. Refused, never guessed: fewer than min_pairs; a side whose means are all equal at 12 significant digits
    (undefined); means spanning more than 12 orders of magnitude; a value of 1e200 or more; the same table, or the same rows
    under two handles.
    One claim per method (kind correlation; no p-value); every entry and why it was or was not paired is saved as an
    analysis_result artifact. A descriptive correlation of recorded entry means, not a recommendation."""
    x_table, y_table = (_load(registry, handle) for handle in a.artifact_ids)
    if x_table.manifest.artifact_id == y_table.manifest.artifact_id or x_table.manifest.sha256 == y_table.manifest.sha256:
        raise AnalystError("invalid_argument", f"{x_table.manifest.artifact_id} and {y_table.manifest.artifact_id} hold the same rows; "
                                               "correlate two different tables")
    if a.by.endswith("Name"):
        raise AnalystError("invalid_argument", f"join by the ID column, not {a.by!r}; names are carried as labels")
    label_col = LABEL_FOR.get(a.by)
    label_col = label_col if label_col in x_table.manifest.columns and label_col in y_table.manifest.columns else None
    if a.name_prefix is not None and label_col is None:
        raise AnalystError("invalid_argument", f"name_prefix needs a name column beside {a.by!r} in both tables")
    x, y = _entry_means(x_table, a, label_col), _entry_means(y_table, a, label_col)
    names = {entry: x.labels.get(entry, set()) | y.labels.get(entry, set()) for entry in set(x.labels) | set(y.labels)}
    label = {entry: "; ".join(sorted(found)) for entry, found in names.items()}
    excluded: set[str] = set()
    for wanted in a.exclude:
        try:
            excluded |= set(_entries(x_table, a.by, label_col, [wanted], "exclude"))
        except AnalystError as not_in_x:
            try:
                excluded |= set(_entries(y_table, a.by, label_col, [wanted], "exclude"))
            except AnalystError:
                raise not_in_x from None
    candidates = sorted(set(x.means) | set(y.means))
    outside = {e for e in candidates if a.name_prefix is not None and e not in excluded and not any(n.startswith(a.name_prefix) for n in names.get(e, ()))}
    kept = [e for e in candidates if e not in excluded and e not in outside]
    paired = [e for e in kept if e in x.means and e in y.means]
    only_x = sum(1 for e in kept if e in x.means and e not in y.means)
    only_y = sum(1 for e in kept if e in y.means and e not in x.means)
    if len(paired) < a.min_pairs:
        raise AnalystError("unsupported", f"{len(paired)} shared entries with a mean in both tables; a correlation needs at least {a.min_pairs} "
                                          f"(only in x: {only_x}, only in y: {only_y}, excluded: {len(excluded)}, outside the name prefix: {len(outside)})")
    xs, ys = [x.means[e][0] for e in paired], [y.means[e][0] for e in paired]
    # Each side is divided by its largest absolute value first: a correlation does not change under that scaling, and the sums
    # inside it can then neither overflow (values near 1e155 gave r = 0) nor underflow (values near 1e-200 crashed) (review).
    # The scale is the largest RAW plot value behind the paired means, not the means: an entry whose plot values cancel (0.1 + 0.2
    # - 0.3) has a mean of float residue, and against the means' own scale that residue looked like spread (r -1.0 from noise; T4,
    # coach 2026-10-23 03:48 UTC).
    scaled = {side: [v / scale if scale else 0.0 for v in values]
              for side, values, scale in (("x", xs, max(x.peaks[e] for e in paired)), ("y", ys, max(y.peaks[e] for e in paired)))}
    for side, values in (("x", xs), ("y", ys)):
        keys = {_tie_key(v) for v in scaled[side]}
        if len(keys) == 1:
            raise AnalystError("unsupported", f"the {side} entry means are all equal at 12 significant digits ({_display(values[0])}) over the "
                                              f"{len(paired)} shared entries; a correlation is undefined")
        nonzero = [abs(k) for k in keys if k]
        if nonzero and max(nonzero) / min(nonzero) > 1e12:    # review 2: means 13 orders of magnitude apart tied falsely; say so instead
            smallest = min(abs(v) for v, k in zip(values, scaled[side]) if _tie_key(k))
            raise AnalystError("unsupported", f"the {side} entry means span more than 12 orders of magnitude (from {smallest!r} to "
                                              f"{max(abs(v) for v in values)!r} in absolute value); their ranks and correlation are not computed")
    x_ranks, y_ranks = _average_ranks(scaled["x"]), _average_ranks(scaled["y"])
    results: dict[str, float] = {}
    if a.method in ("pearson", "both"):
        results["pearson"] = statistics.correlation(scaled["x"], scaled["y"])
    if a.method in ("spearman", "both"):
        if len(set(x_ranks)) == 1 or len(set(y_ranks)) == 1:
            raise AnalystError("unsupported", "every shared entry ties on one side; the Spearman correlation is undefined")
        results["spearman"] = statistics.correlation(x_ranks, y_ranks)
    overlap = set(x.studies) & set(y.studies)
    warnings: list[str] = []
    if overlap and set(x.studies) != set(y.studies):        # review: neither the same studies nor two different ones, and nothing said
        warnings.append(f"x covers {'study' if len(x.studies) == 1 else 'studies'} {', '.join(x.studies)} and y covers {', '.join(y.studies)}: "
                        "the correlation pairs entry means from different study sets")
    for side, e in (("x", x), ("y", y)):
        if e.manifest_only:
            warnings.append(f"{side}: the table's manifest lists {'a study' if len(e.manifest_only) == 1 else 'studies'} with no row: {', '.join(e.manifest_only)}")

    variant = hashlib.sha256(json.dumps([x_table.manifest.sha256, y_table.manifest.sha256, a.by, a.value, a.entries, sorted(excluded), a.name_prefix,
                                         a.min_pairs, a.duplicate_policy, a.allow_incomplete]).encode("utf-8")).hexdigest()[:10]
    hashes = [x_table.manifest.sha256, y_table.manifest.sha256]
    exclusions: list[Exclusion] = []
    for side, e in (("x", x), ("y", y)):
        exclusions += [Exclusion(reason=f"{side}: {item.reason}", count=item.count) for item in _exclusions(e.classified)]
        if e.no_study_rows:
            exclusions.append(Exclusion(reason=f"{side}: rows with an empty studyDbId", count=e.no_study_rows))
        if e.no_entry_values:                                 # T4 review F2: a valid value without an entry ID is counted
            exclusions.append(Exclusion(reason=f"{side}: rows with an empty {a.by}", count=e.no_entry_values))
        if e.not_in_every_study:
            exclusions.append(Exclusion(reason=_named_reason(f"entries not in every study of {side}", e.dropped, label),
                                        count=e.not_in_every_study))
    removed = sum(1 for e in excluded if e in x.means or e in y.means)              # named entries that had a mean to lose (review)
    for reason, entries in (("entries only in x", [e for e in kept if e in x.means and e not in y.means]),
                            ("entries only in y", [e for e in kept if e in y.means and e not in x.means]),
                            ("entries left out: excluded", [e for e in excluded if e in x.means or e in y.means]),
                            ("entries left out: outside the name prefix", sorted(outside))):
        if entries:                                  # GAP 2: the entries named, not only counted
            exclusions.append(Exclusion(reason=_named_reason(reason, entries, label), count=len(entries)))
    rule = "over the entries present in every study of the table" if a.entries == "in_every_study" else "over the studies where the entry has a value"
    rule_text = "entries: in every study of each table" if a.entries == "in_every_study" else "entries: in any study"
    basis = (f"entry mean = the mean of the entry's valid values per study, then the unweighted mean over its studies ({rule}); "
             f"x: {x.transformation}; y: {y.transformation}")
    denominator = Denominator(name="entries with an entry mean in both tables (pairs)", n=len(paired))
    claims = [Claim(claim_id=_claim_id(x_table.manifest.artifact_id, "correlation", method, namespace=f"cor-{variant}"), kind="correlation",
                    value=value, denominator=denominator, source_artifact_hashes=hashes, grouping=[a.by],
                    transformation=f"{'Pearson' if method == 'pearson' else 'Spearman'} correlation of entry means | {_table_text('x', x)} | "
                                   f"{_table_text('y', y)} | {rule_text} | {CORRELATION_NOTE} | {basis}"
                                   + (" | ranks: ties share the average rank" if method == "spearman" else ""),
                    exclusions=exclusions)
              for method, value in results.items()]

    rank_of = {e: (rx, ry) for e, rx, ry in zip(paired, x_ranks, y_ranks)}
    rows = []
    for e in sorted(set(candidates) | excluded | x.dropped | y.dropped):
        missing = [side for side, table_means in (("x", x), ("y", y)) if e in table_means.dropped]
        why = ("excluded" if e in excluded else "outside the name prefix" if e in outside else "" if e in rank_of
               else f"not in every study of {' and '.join(missing)}" if missing else "only in x" if e in x.means else "only in y")
        xm, ym = x.means.get(e), y.means.get(e)
        rows.append({"entry": e, "label": label.get(e, ""), "paired": "true" if e in rank_of else "false", "left_out": why,
                     "x_mean": "" if xm is None else repr(xm[0]), "x_n_values": "" if xm is None else str(xm[1]), "x_n_studies": "" if xm is None else str(xm[2]),
                     "y_mean": "" if ym is None else repr(ym[0]), "y_n_values": "" if ym is None else str(ym[1]), "y_n_studies": "" if ym is None else str(ym[2]),
                     "x_rank": repr(rank_of[e][0]) if e in rank_of else "", "y_rank": repr(rank_of[e][1]) if e in rank_of else ""})
    lineage = [f"lineage: correlate_entry_means of {x_table.manifest.artifact_id} sha256={x_table.manifest.sha256} and "
               f"{y_table.manifest.artifact_id} sha256={y_table.manifest.sha256}", basis, f"left out: excluded {sorted(excluded)}, name_prefix {a.name_prefix!r}",
               f"analysis target: {ANALYSIS_TARGET}"]
    complete = bool(x_table.manifest.complete and y_table.manifest.complete)
    manifest = registry.save_table(rows, kind="analysis_result", columns=CORRELATION_RESULT_COLUMNS, complete=complete,
                                   source_request_ids=list(dict.fromkeys([*x_table.manifest.source_request_ids, *y_table.manifest.source_request_ids])),
                                   study_ids=list(dict.fromkeys([*x_table.manifest.study_ids, *y_table.manifest.study_ids])),
                                   variable_ids=list(dict.fromkeys([*x_table.manifest.variable_ids, *y_table.manifest.variable_ids])), quality_notes=lineage)
    tables = [{"artifact_id": e.table.manifest.artifact_id, "trait": e.table.manifest.measurement.trait, "unit": e.table.manifest.measurement.unit,
               "variable_id": e.variable, "study_ids": e.studies, "entries_with_a_mean": len(e.means), "entries_not_in_every_study": e.not_in_every_study,
               "transformation": e.transformation} for e in (x, y)]
    data = {"x_artifact_id": x_table.manifest.artifact_id, "y_artifact_id": y_table.manifest.artifact_id, "result_artifact_id": manifest.artifact_id,
            "n_pairs": len(paired), "pearson": results.get("pearson"), "spearman": results.get("spearman"),
            "display_pearson": _display(results.get("pearson")), "display_spearman": _display(results.get("spearman")), "tables": tables,
            "unpaired": {"only in x": only_x, "only in y": only_y}, "left_out": {"excluded": removed, "outside the name prefix": len(outside)},
            "excluded_named": sorted(excluded), "entries_rule": rule, "claims": _claims_json(claims), "note": CORRELATION_NOTE,
            "analysis_target": ANALYSIS_TARGET, "value_policy": VALUE_POLICY}
    warnings += [w for e in (x, y) for w in (e.caveat, e.classified.duplicate_note if a.duplicate_policy != "reject" else None) if w]
    for side, e in (("x", x), ("y", y)):
        if not e.classified.units_known:
            warnings.append(f"{side}: independence unknown: some valid rows name no observation unit")
        warnings.extend(f"{side}: {w}" for w in _timing_warnings(e.table, a.value))
    return ToolResult(ok=True, data=data, artifact_ids=[x_table.manifest.artifact_id, y_table.manifest.artifact_id, manifest.artifact_id],
                      complete=complete, warnings=warnings)


# The design fields a plot's position object states (R2 tool 2, coach 2026-10-22 03:48 UTC, step 6): the plot list carries them
# only inside one JSON text column, so the analyst could not count replicates (A35.2, A37.1) or find the checks (B29.2, B29.3).
POSITION_COL = "observationUnitPosition"
DESIGN_COLUMNS = {
    "rep": "replicate code the plot's position states (observationLevelRelationships, levelName rep; '' = not stated)",
    "block": "block code the plot's position states (levelName block; '' = not stated)",
    "plot": "plot code the plot's position states (levelName plot, or its own observationLevel; '' = not stated)",
    "entryType": "the position's entryType in lower case (trimmed and case-folded), for example check or test (CHECK, Check and "
                 "check are all check; '' = not stated); any value other than check, test or filler is kept as stored and the call names it",
    "row": "grid row: the position coordinate whose type is GRID_ROW ('' = not stated)",
    "col": "grid column: the position coordinate whose type is GRID_COL ('' = not stated)",
}
_DESIGN_LEVELS = ("rep", "block", "plot")
_GRID_AXES = {"GRID_ROW": "row", "GRID_COL": "col"}


class _Unreadable(Exception):
    """A position object that cannot be read as design fields (round 4, review 3 of 2026-10-22): a value of the wrong JSON type
    where text is read, a code longer than 1,000 characters, or one the table cannot save. Counted and warned, never guessed."""


ENTRY_TYPES = ("check", "test", "filler")      # the BrAPI entry types, after trimming and case-folding
DESIGN_VALUE_LIMIT = 1000


def _design_text(value: Any) -> str:
    """A code as the position states it, trimmed. Numbers are read as the JSON text they were written in (1e2 stays "1e2");
    NaN and null state nothing; true / false, a list or an object where text is read make the position unreadable."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise _Unreadable(f"{type(value).__name__} where text is read")
    return value.strip()


def _saved_text(value: Any) -> str:
    """_design_text for a value plot_design writes into its table. A lone surrogate escape ("\\ud800") parses, but the table
    cannot save it as UTF-8 (review 2 of 2026-10-22); a value longer than 1,000 characters would not load back (review 3): both
    make the position unreadable."""
    text = _design_text(value)
    text.encode("utf-8")
    if len(text) > DESIGN_VALUE_LIMIT:
        raise _Unreadable("a value longer than 1,000 characters")
    return text


def _no_repeated_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("a key is written twice in one object")     # the last value would win silently
    return dict(pairs)


def _has_text(value: str) -> bool:
    """True when an ID has at least one character that str.isprintable() accepts and that is not a space: an ID made only of
    spaces, control or format characters (a zero-width space) is no ID (review 4 of 2026-10-22). A lone combining mark or a
    blank-looking letter such as U+3164 still counts; it joins only to the identical string."""
    return any(ch.isprintable() and not ch.isspace() for ch in value)


def _shown_key(value: str, width: int = 60) -> str:
    """A value as a key of the tool's data: cut at `width` characters with an ellipsis, and a lone surrogate written as \\udXXX so
    the result always serializes (review 5 of 2026-10-22: a 130,000-character level name, and a lone surrogate that made
    ToolResult.model_dump_json() raise)."""
    cut = value[:width] + ("…" if len(value) > width else "")
    return cut.encode("utf-8", "backslashreplace").decode("utf-8")


def _plot_fields(unit: str, text: str) -> tuple[dict[str, str], str, str | None, set[str], set[str]]:
    """The design fields one plot's position states ('' where it states none); why there are none ("empty", "unreadable", else
    ""); the stored entryType when it is not check, test or filler (else None); and the level names and coordinate types the
    position states but the tool does not read, as written. Nothing is read from names or numbers outside the position; a level
    stated with two codes, or two plot codes, is refused. Review of 2026-10-22: a position nested 1,000 levels deep, or with a
    key written twice, is unreadable, not a failure; review 2: so is one with a code the table cannot save; review 3 (round 4):
    so is a value of the wrong JSON type, or one longer than 1,000 characters."""
    fields = dict.fromkeys(DESIGN_COLUMNS, "")
    nothing: tuple[set[str], set[str]] = (set(), set())
    if not text.strip():
        return fields, "empty", None, *nothing
    try:
        position = json.loads(text, parse_int=str, parse_float=str, parse_constant=lambda _name: None, object_pairs_hook=_no_repeated_keys)
    except (ValueError, RecursionError):
        return fields, "unreadable", None, *nothing
    if not isinstance(position, dict):
        return fields, "unreadable", None, *nothing
    try:
        fields, levels_not_read, types_not_read = _stated_fields(unit, position)
    except (UnicodeEncodeError, _Unreadable):
        return dict.fromkeys(DESIGN_COLUMNS, ""), "unreadable", None, *nothing
    odd = fields["entryType"] if fields["entryType"] and fields["entryType"] not in ENTRY_TYPES else None
    return fields, "", odd, levels_not_read, types_not_read


def _stated_fields(unit: str, position: dict[str, Any]) -> tuple[dict[str, str], set[str], set[str]]:
    """The design fields a readable position object states, with the level names (of its relations and of its own level) and
    the coordinate types it states but the tool does not read, as written, once per plot (review 5 of 2026-10-22). Codes are
    trimmed and stored as written; the entryType is trimmed and case-folded, because the BrAPI standard writes CHECK and
    SweetPotatoBase writes check, and a filter on "check" must find both (review 2). Raises _Unreadable for a value of the wrong
    JSON type, a list of relations that is not a list, or a relation that is not an object."""
    fields = dict.fromkeys(DESIGN_COLUMNS, "")
    levels_not_read: set[str] = set()
    types_not_read: set[str] = set()
    relations = position.get("observationLevelRelationships")
    if relations is not None and not isinstance(relations, list):
        raise _Unreadable("observationLevelRelationships is not a list")
    seen: set[str] = set()
    for relation in relations or []:
        if not isinstance(relation, dict):
            raise _Unreadable("a level relation is not an object")
        name_text = _design_text(relation.get("levelName"))
        name = name_text.lower()
        if name not in _DESIGN_LEVELS:
            if name_text:
                levels_not_read.add(name_text)                                    # round 5: "rep\u200b" or "plant" is named, not dropped quietly
            continue
        code = _saved_text(relation.get("levelCode"))
        if name in seen and code != fields[name]:
            raise AnalystError("unsupported", f"plot {unit}: its position states the level {name!r} with two codes "
                                              f"({fields[name]!r} and {code!r}); a design field needs one value")
        seen.add(name)
        fields[name] = code
    level = position.get("observationLevel")
    if level is not None and not isinstance(level, dict):
        raise _Unreadable("observationLevel is not an object")
    level_name = _design_text(level.get("levelName")) if isinstance(level, dict) else ""
    if level_name.lower() == "plot":
        own = _saved_text(level.get("levelCode"))
        if fields["plot"] and own and own != fields["plot"]:
            raise AnalystError("unsupported", f"plot {unit}: its position states two plot codes ({fields['plot']!r} and {own!r})")
        fields["plot"] = fields["plot"] or own
    elif level_name:
        levels_not_read.add(level_name)                                           # round 5b: the plot's own level, "plots" or "plot\u200b"
    fields["entryType"] = _saved_text(position.get("entryType")).casefold()
    if len(fields["entryType"]) > DESIGN_VALUE_LIMIT:                             # case-folding can lengthen a value (sharp s -> ss)
        raise _Unreadable("an entryType longer than 1,000 characters once case-folded")
    axes: dict[str, str] = {}
    for coordinate in ("positionCoordinateX", "positionCoordinateY"):
        kind_text = _design_text(position.get(coordinate + "Type"))
        kind = kind_text.upper()                                                  # in any case, like the level names
        axis = _GRID_AXES.get(kind)
        if axis is None:
            if kind_text:
                types_not_read.add(kind_text)                                     # round 5: LATITUDE or "GRID_COL\u200b" is named
            continue
        if axis in axes:
            raise AnalystError("unsupported", f"plot {unit}: both grid coordinates are of type {kind}; one row and one column are needed")
        axes[axis] = _saved_text(position.get(coordinate))
    fields.update(axes)
    return fields, levels_not_read, types_not_read


_NUMERAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")     # digits, one point, an exponent: no "_", hex, Infinity, NaN


def _codes_equal_as_numbers(per_row: list[dict[str, str]]) -> dict[str, list[list[str]]]:
    """Per design column, every group of distinct codes written differently but equal as numbers ("01", "1", "1.0"), sorted:
    they stay as written and count apart, so the call names them (review 4 of 2026-10-22). Only numerals are compared (review
    5: Python's Decimal reads "1_0" as 10, and "sNaN" or "Infinity" are not codes to compare). Data and the warning show at most
    5 groups of at most 5 spellings (_same_number_data, _same_number_text)."""
    out: dict[str, list[list[str]]] = {}
    for column in DESIGN_COLUMNS:
        if column == "entryType":
            continue
        groups: dict[Decimal, set[str]] = {}
        for code in {row[column] for row in per_row if row[column]}:
            if not _NUMERAL.fullmatch(code):
                continue
            try:
                number = Decimal(code)
            except InvalidOperation:
                continue
            if number.is_finite():
                groups.setdefault(number, set()).add(code)
        found = sorted(sorted(codes) for codes in groups.values() if len(codes) > 1)
        if found:
            out[column] = found
    return out


def _same_number_data(groups: list[list[str]]) -> list[list[str]]:
    """At most 5 groups of at most 5 spellings, each cut at 60 characters (review 5: 1,000 spellings of one rep made a
    504,594-character warning and data of 505 KB)."""
    return [[_shown_key(code) for code in group[:5]] for group in groups[:5]]


def _same_number_text(column: str, groups: list[list[str]]) -> str:
    """The warning for one column's codes equal as numbers: the same groups and spellings as data, each group saying how many
    spellings it has beyond the 5 shown, then how many groups beyond the 5 shown (review 5: the rest was not counted)."""
    shown = "; ".join(", ".join(ascii(code[:60]) + ("…" if len(code) > 60 else "") for code in group[:5])
                      + (f" and {len(group) - 5} more" if len(group) > 5 else "") for group in groups[:5])
    return (f"{column} codes written differently but equal as numbers: {shown}"
            + (f"; and {len(groups) - 5} more groups" if len(groups) > 5 else "") + "; they stay as written and are counted apart")


def _listed(counter: Counter, show: int = 5, width: int = 60) -> str:
    """Up to `show` values of a Counter, most frequent first, each cut at `width` characters and written with ascii() so an
    invisible character shows, then how many more (round 5: one warning had reached 50 KB)."""
    values = [value for value, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]
    shown = ", ".join(ascii(value[:width]) + ("…" if len(value) > width else "") for value in values[:show])
    return shown + (f" and {len(values) - show} more" if len(values) > show else "")


def _capped(counter: Counter, limit: int = 20) -> dict[str, int]:
    """The `limit` most frequent values of a Counter as a dict (round 5: a result once held 2 MB of entry types)."""
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit])


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def tool_plot_design(registry: ArtifactRegistry, a: PlotDesignArgs) -> ToolResult:
    """Each plot's design from the plot list's position objects, in columns of their own: replicate (rep), block and plot
    codes, entryType in lower case (check / test; CHECK and Check are written check) and the grid row and column; empty where
    the position does not state them. Nothing is read from names or numbers. With only the plot list (roster_artifact_id,
    one study) it saves the plot list with these columns; count_records then gives the replicates (distinct_by rep) and the
    plots per replicate (group_by rep), and filter_rows on entryType eq check the check entries. With a measurements table of the
    same study (artifact_id), each of its rows gets its plot's columns, so group_stats by rep can follow; rows whose plot is not
    in the plot list keep empty fields, and the count is stated. Codes are stored as written (trimmed): numbers as their JSON
    text (1 and 1.0 are different codes, and codes equal as numbers but written differently are named); null, NaN and Infinity
    state nothing. A value of the wrong JSON type (true, a list, an object where text is read), a value longer than 1,000
    characters, or one the table cannot save makes the position unreadable: counted and warned, its fields empty. An entryType
    that is not check, test or filler is kept and named; level names (of the relations and of the plot's own level) and
    coordinate types the tool does not read are counted once per plot; plot-list rows without a plot ID are counted. Makes no
    claims of its own."""
    roster = _load(registry, a.roster_artifact_id)
    if roster.manifest.kind != "observation_units":
        raise AnalystError("invalid_argument", f"roster {roster.manifest.artifact_id} is {roster.manifest.kind}, not observation_units "
                                               "(the plot list from get_observation_units)")
    if len(set(roster.manifest.study_ids)) != 1:
        raise AnalystError("invalid_argument", f"the plot list {roster.manifest.artifact_id} is of {sorted(roster.manifest.study_ids)}; "
                                               "pass the plot list of one study (counts would mix studies)")
    tables = [roster]
    if a.artifact_id is not None:
        table = _load(registry, a.artifact_id)
        if table.manifest.kind != "observations":
            raise AnalystError("invalid_argument", f"{table.manifest.artifact_id} is {table.manifest.kind}, not observations (a measurements table)")
        if set(table.manifest.study_ids) != set(roster.manifest.study_ids):
            raise AnalystError("invalid_argument", f"the plot list is of {sorted(roster.manifest.study_ids)} and the measurements of "
                                                   f"{sorted(table.manifest.study_ids)}; join one study's plot list to the same study's measurements")
        tables.append(table)
    for t in tables:
        already = [c for c in DESIGN_COLUMNS if c in t.manifest.columns]
        if already:
            raise AnalystError("invalid_argument", f"{t.manifest.artifact_id} already has the design column(s) {already}")
        _require_column(t, UNIT_COL, "plot ID column")
        stray = sorted({row.get("studyDbId", "") for row in t.rows} - {""} - set(roster.manifest.study_ids))
        if stray:                                   # review of 2026-10-22: only the manifests were compared, never the rows
            raise AnalystError("invalid_argument", f"{t.manifest.artifact_id} has rows of {stray}, not of the plot list's study "
                                                   f"{sorted(roster.manifest.study_ids)}")
    _require_column(roster, POSITION_COL, "position column")
    units = sorted({row.get("levelName", "").strip() for row in roster.rows} - {""}, key=str.lower)
    if any(level.lower() != "plot" for level in units):   # review: plants counted as plots gave "plots per rep = 3"
        raise AnalystError("unsupported", f"the plot list {roster.manifest.artifact_id} holds units of level {units}; plot_design reads "
                                          "plots only, and counting other units as plots would be wrong")
    caveats = [c for c in (_completeness_gate(t, a.allow_incomplete) for t in tables) if c]

    per_row: list[dict[str, str]] = []
    by_unit: dict[str, dict[str, str]] = {}
    problems: Counter = Counter()
    odd_types: Counter = Counter()                  # stored entryType values that are not check, test or filler (round 4 (A), round 5)
    levels_not_read: Counter = Counter()            # plots per level name the tool does not read, as shown by _shown_key (round 5, 5b)
    types_not_read: Counter = Counter()             # plots per coordinate type other than GRID_ROW / GRID_COL, as shown (round 5, 5b)
    level_names_seen: set[str] = set()              # their full values, for the distinct counts (round 5b)
    type_names_seen: set[str] = set()
    no_unit = 0
    for row in roster.rows:
        unit = row.get(UNIT_COL, "")
        key = unit if _has_text(unit) else ""          # round 5: an ID of spaces or invisible characters is no ID and never joins
        if key and key in by_unit:
            raise AnalystError("unsupported", f"plot {unit} is listed twice in the plot list {roster.manifest.artifact_id}")
        fields, problem, odd, levels, kinds = _plot_fields(unit, row.get(POSITION_COL, ""))
        problems[problem] += 1
        if odd is not None:
            odd_types[odd] += 1
        levels_not_read.update({_shown_key(name) for name in levels})
        types_not_read.update({_shown_key(kind) for kind in kinds})
        level_names_seen |= levels
        type_names_seen |= kinds
        per_row.append(fields)
        if key:
            by_unit[key] = fields
        else:
            no_unit += 1
    same_number = _codes_equal_as_numbers(per_row)
    blank = dict.fromkeys(DESIGN_COLUMNS, "")
    if len(tables) == 1:
        rows = [{**row, **fields} for row, fields in zip(roster.rows, per_row)]
        unmatched = 0
    else:
        rows = [{**row, **by_unit.get(row.get(UNIT_COL, ""), blank)} for row in tables[1].rows]
        unmatched = sum(1 for row in tables[1].rows if row.get(UNIT_COL, "") not in by_unit)
    out = tables[-1]
    warnings = [*caveats]
    if problems["unreadable"]:
        warnings.append(f"{_plural(problems['unreadable'], 'plot position', 'plot positions')} could not be read; their design fields are empty")
    if problems["empty"]:
        warnings.append(f"{_plural(problems['empty'], 'plot has', 'plots have')} no position object; their design fields are empty")
    if unmatched:
        warnings.append(f"{_plural(unmatched, 'measurement row has', 'measurement rows have')} no plot in the plot list; "
                        "their design fields are empty")
    if odd_types:
        warnings.append(f"{_plural(sum(odd_types.values()), 'plot has', 'plots have')} an entryType that is not check, test or filler: "
                        f"{_listed(odd_types)}; the value is kept (trimmed, case-folded) and filter_rows sees it as shown")
    if levels_not_read:
        warnings.append(f"level names the tool does not read (it reads rep, block and plot): {_listed(levels_not_read)}")
    if types_not_read:
        warnings.append(f"coordinate types the tool does not read (it reads GRID_ROW and GRID_COL): {_listed(types_not_read)}")
    for column, groups in same_number.items():
        warnings.append(_same_number_text(column, groups))
    if no_unit:
        warnings.append(f"{_plural(no_unit, 'plot-list row has', 'plot-list rows have')} no plot ID; its design fields are read, "
                        "but no measurement can be joined to it")
    lineage = [f"lineage: plot_design of {roster.manifest.artifact_id} sha256={roster.manifest.sha256}"
               + (f" onto {out.manifest.artifact_id} sha256={out.manifest.sha256}" if out is not roster else ""),
               "design fields read from observationUnitPosition only: rep, block and plot (observationLevelRelationships), "
               "entryType, and the GRID_ROW / GRID_COL coordinates; nothing inferred", *warnings]
    complete = all(t.manifest.complete for t in tables)
    manifest = registry.save_table(rows, kind=out.manifest.kind, columns={**out.manifest.columns, **DESIGN_COLUMNS}, complete=complete,
                                   source_request_ids=list(dict.fromkeys(r for t in tables for r in t.manifest.source_request_ids)),
                                   study_ids=list(out.manifest.study_ids), variable_ids=list(out.manifest.variable_ids),
                                   measurement=out.manifest.measurement, quality_notes=lineage,
                                   entries_without_value=out.manifest.entries_without_value)   # every row kept (coach 09:51 UTC, (iv))
    preview = registry.preview(manifest.artifact_id, 5)
    data = {"roster_artifact_id": roster.manifest.artifact_id, "source_artifact_id": out.manifest.artifact_id,
            "artifact_id": manifest.artifact_id, "rows": len(rows),
            "stated": {column: sum(1 for row in rows if row[column]) for column in DESIGN_COLUMNS},
            "unreadable_positions": problems["unreadable"], "empty_positions": problems["empty"],
            "rows_without_a_plot_in_the_roster": unmatched, "rows_without_a_plot_id": no_unit,
            "entry_types_not_standard": _capped(odd_types), "entry_types_not_standard_distinct": len(odd_types),
            "level_names_not_read": _capped(levels_not_read), "level_names_not_read_distinct": len(level_names_seen),
            "coordinate_types_not_read": _capped(types_not_read), "coordinate_types_not_read_distinct": len(type_names_seen),
            "codes_equal_as_numbers": {column: _same_number_data(groups) for column, groups in same_number.items()},
            "codes_equal_as_numbers_groups": {column: len(groups) for column, groups in same_number.items()}, "complete": complete,
            "preview": {"displayed_rows": preview.displayed_rows, "rows": preview.rows},
            "note": "design fields as the positions state them; count_records, filter_rows and group_stats state the numbers"}
    ids = [t.manifest.artifact_id for t in tables] + [manifest.artifact_id]
    return ToolResult(ok=True, data=data, artifact_ids=ids, complete=complete, warnings=warnings)


# The analysis of variance of one trial (T2, coach 2026-10-22 20:48 UTC; research build order tool 2; notes Q1, Q2 and Q11 of
# Claude outputs/research_stats_tools_20261022/NOTES_plain_python_implementation.md): closed forms on complete data, and on an RCBD
# with missing plots the least-squares (fitting-constants) analysis with reps first, from a small reps x reps system.
ANOVA_NOTE = "descriptive analysis of variance of recorded plot values within one trial, not a recommendation"
ANOVA_RESULT_COLUMNS = {
    "source": "rep, entry, error or total (an RCBD with missing plots: sequential least squares, reps first (the OLS fit))",
    "df": "degrees of freedom",
    "ss": "sum of squares",
    "ms": "mean square = SS / df ('' for total)",
    "f": "F = MS / MS of error ('' for error and total)",
    "p": "upper-tail probability of F ('' for error and total)",
}
ZERO_ERROR = "the error mean square is 0: the plots of each entry are identical, or the reps explain them exactly"
P_UNDERFLOW = " | p below 1e-300: smaller than a double holds"     # an F tail of 0.0 (T3 review 2); the answer shows "< 1e-300"
BLOCK_COL = "block"                                       # plot_design's column (DESIGN_COLUMNS)
H2_TEXT = ("entry-mean heritability H2 = sigma2 g / (sigma2 g + sigma2 e / r) = 1 - MS error / MS entry, broad sense and within this "
           "trial: the genotype variance is confounded with genotype x environment here, which is why some tools call it repeatability")


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    """The solution of matrix @ x = vector by Gaussian elimination with partial pivoting; None when the system is singular."""
    n = len(vector)
    m = [list(row) + [vector[i]] for i, row in enumerate(matrix)]
    scale = max((abs(x) for row in matrix for x in row), default=0.0) or 1.0
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) <= 1e-12 * scale:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(n):
            if r != col and m[r][col]:
                factor = m[r][col] / m[col][col]
                for k in range(col, n + 1):
                    m[r][k] -= factor * m[col][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def _rep_code_clashes(codes: Iterable[str]) -> list[list[str]]:
    """Rep codes that are the same number written differently ("1" and "1.0"), compared as the plot list compares them:
    numerals only, exact Decimal equality (no rounding, no overflow)."""
    written: dict[Decimal, set[str]] = {}
    for code in codes:
        if not _NUMERAL.fullmatch(code):
            continue                                      # as the plot list: only numerals are compared as numbers
        try:
            number = Decimal(code)
        except InvalidOperation:
            continue
        if number.is_finite():
            written.setdefault(number, set()).add(code)
    return sorted(sorted(found) for found in written.values() if len(found) > 1)


def _groups_linked(cells: list[tuple[str, str]]) -> int:
    """How many separate groups the entries and reps fall into when each plot links its entry and its rep (1 = connected)."""
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def root(node: tuple[str, str]) -> tuple[str, str]:
        while parent.setdefault(node, node) != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for entry, rep in cells:
        parent[root(("entry", entry))] = root(("rep", rep))
    return len({root(node) for node in list(parent)})


@dataclass
class _AnovaResult:
    design: str                           # "RCBD" or "one-way"
    rows: list[dict[str, Any]]            # source, df, ss, ms, f, p (None where not defined)
    n: int
    entries: int
    reps: int | None
    balanced: bool
    adjusted: bool
    empty_cells: int
    plots_per_entry: dict[str, int]
    grand_mean: float
    ms_entry: float
    ms_error: float
    df_error: int


def _row(source: str, df: int, ss: float, ms_error: float) -> dict[str, Any]:
    """One ANOVA row, with its F against the error mean square (None when that is 0); p is added once the error df is known."""
    ms = ss / df
    return {"source": source, "df": df, "ss": ss, "ms": ms, "f": ms / ms_error if ms_error > 0 else None, "p": None}


def _anova(plots: list[tuple[str, str | None, float]]) -> _AnovaResult:
    """(entry, rep or None, value) per plot -> the analysis of variance. No rep anywhere: one-way (exact with unequal plot
    counts). Reps: an RCBD; complete -> the closed form; missing plots -> fitting constants, reps first: with entries absorbed,
    C beta = Q (C = diag(plots per rep) - N' diag(1 / plots per entry) N, Q the adjusted rep totals) gives SS reps adjusted for
    entries = beta'Q, and SS entries adjusted for reps = SS entries ignoring reps + beta'Q - SS reps ignoring entries. Values are
    centred on the grand mean first (the sums of squares do not change; less cancellation)."""
    n = len(plots)
    grand = math.fsum(v for _, _, v in plots) / n
    centred = [(e, r, v - grand) for e, r, v in plots]
    entries = sorted({e for e, _, _ in plots})
    per_entry = Counter(e for e, _, _ in plots)
    total_e = {e: 0.0 for e in entries}
    for e, _, v in centred:
        total_e[e] += v
    ss_total = math.fsum(v * v for _, _, v in centred)
    ss_entry_raw = math.fsum(total_e[e] ** 2 / per_entry[e] for e in entries)
    t = len(entries)
    if plots[0][1] is None:                                   # one-way
        df_e = n - t
        ss_error = math.fsum((v - total_e[e] / per_entry[e]) ** 2 for e, _, v in centred)   # from the residuals (review 2)
        if ss_error <= 1e-14 * ss_total:
            ss_error = 0.0                                    # an exact fit: floating residue, not error variance (review)
        ms_error = ss_error / df_e
        rows = [_row("entry", t - 1, ss_entry_raw, ms_error), {"source": "error", "df": df_e, "ss": ss_error, "ms": ms_error, "f": None, "p": None},
                {"source": "total", "df": n - 1, "ss": ss_total, "ms": None, "f": None, "p": None}]
        return _AnovaResult("one-way", rows, n, t, None, len(set(per_entry.values())) == 1, False, 0, dict(per_entry), grand,
                            rows[0]["ms"], ms_error, df_e)
    reps = sorted({r for _, r, _ in plots})  # type: ignore[type-var]
    per_rep = Counter(r for _, r, _ in plots)
    total_r = {r: 0.0 for r in reps}
    for _, r, v in centred:
        total_r[r] += v                                        # type: ignore[index]
    ss_rep_raw = math.fsum(total_r[r] ** 2 / per_rep[r] for r in reps)
    cell = {(e, r) for e, r, _ in plots}
    q = [total_r[r] - math.fsum(total_e[e] / per_entry[e] for e in entries if (e, r) in cell) for r in reps]
    c = [[(per_rep[r] if r == s else 0.0) - math.fsum(1.0 / per_entry[e] for e in entries if (e, r) in cell and (e, s) in cell)
          for s in reps] for r in reps]
    k = len(reps) - 1
    beta = _solve([row[:k] for row in c[:k]], q[:k]) if k else []
    if beta is None:
        raise AnalystError("unsupported", "the entries and reps cannot be separated (singular design); no analysis of variance")
    rep_effect = dict(zip(reps, [*beta, 0.0]))                # the last rep's effect fixed at 0
    entry_effect = {e: (total_e[e] - math.fsum(rep_effect[r] for r in reps if (e, r) in cell)) / per_entry[e] for e in entries}
    ss_error = math.fsum((v - entry_effect[e] - rep_effect[r]) ** 2 for e, r, v in centred)   # from the residuals (review 2)
    if ss_error <= 1e-14 * ss_total:
        ss_error = 0.0                                        # an exact fit: floating residue, not error variance (review)
    ss_entry = ss_total - ss_error - ss_rep_raw               # sequential, reps first: SS model - SS reps ignoring entries
    df_e = n - t - len(reps) + 1
    ms_error = ss_error / df_e
    rows = [_row("rep", len(reps) - 1, ss_rep_raw, ms_error), _row("entry", t - 1, ss_entry, ms_error),
            {"source": "error", "df": df_e, "ss": ss_error, "ms": ms_error, "f": None, "p": None},
            {"source": "total", "df": n - 1, "ss": ss_total, "ms": None, "f": None, "p": None}]
    empty = t * len(reps) - len(cell)
    return _AnovaResult("RCBD", rows, n, t, len(reps), empty == 0, empty > 0, empty, dict(per_entry), grand, rows[1]["ms"], ms_error, df_e)


def tool_anova_trial(registry: ArtifactRegistry, a: AnovaTrialArgs) -> ToolResult:
    """The analysis of variance of ONE trial's plot values for one trait (research build order, tool 2): an RCBD (entries +
    replicates) when every plot states its rep (the table plot_design saves with the measurements), one-way (entries only)
    when the table has no rep column or no plot states one. Returns the ANOVA table (source, df, SS, MS, F, p), the grand mean,
    CV % = 100 x sqrt(MS error) / grand mean, the SE of an entry mean sqrt(MS error / r), the SED sqrt(2 MS error / r), the LSD
    (5%) = t(0.975, df error) x SED, the variance components sigma2 e = MS error and sigma2 g = (MS entry - MS error) / r (set to
    0 and flagged when negative) and the entry-mean heritability sigma2 g / (sigma2 g + sigma2 e / r), within this trial. An RCBD
    with missing plots gets the least-squares analysis with reps first, labelled "sequential least squares, reps first (the OLS fit)"; with
    unequal plot counts r is the harmonic mean of the entries' plot counts, flagged. Values are averaged per plot first only
    with duplicate_policy='mean_per_unit'. Refused, never guessed: more than one study or variable, two plots of one entry in
    one rep (unless the person asked for cells='mean', the cell's plots averaged, or cells='single_only', only the entries with
    exactly one plot in every rep; either is labelled), a rep on only some plots, a design that is not connected, fewer than 2
    entries, an RCBD with fewer than 2 reps, no error degrees of freedom. Entries the person named can be left out (exclude, for
    example the checks): one exclusion per name, counted in plots, and the names on every claim. Descriptive numbers of recorded
    values, never a choice of entries."""
    table = _load(registry, a.artifact_id)
    variable = _single_variable(table)
    studies = set(table.manifest.study_ids)
    if "studyDbId" in table.manifest.columns:
        studies |= {r.get("studyDbId", "") for r in table.rows if r.get("studyDbId", "")}
    if len(studies) != 1:
        raise AnalystError("unsupported", f"{table.manifest.artifact_id} holds the plots of {len(studies)} studies {sorted(studies)}; an "
                                          "analysis of variance of one trial needs the plots of one study")
    _require_column(table, a.value, "value column")
    _require_column(table, a.entry, "entry column")
    gate = _completeness_gate(table, a.allow_incomplete)
    classified = _classify(table, a.value)
    used, transformation = _apply_duplicate_policy(table, classified, a.duplicate_policy, value=a.value)
    exclusions = _exclusions(classified)
    no_entry = sum(1 for row, _ in used if not row.get(a.entry, "").strip())
    if no_entry:
        exclusions.append(Exclusion(reason="rows without an entry", count=no_entry))
    used = [(row, value) for row, value in used if row.get(a.entry, "").strip()]
    left_out: dict[str, tuple[list[str], int]] = {}           # the person's names (decision (6)): label -> (entries, plots left out)
    no_value: list[str] = []                                  # the person's names for entries with no valid value in the trial (iv)
    dropped: set[str] = set()
    excluded_plots = 0
    if a.exclude:                                             # first: the dates, rep and cell checks see only the kept entries (review F3)
        label_col = LABEL_FOR.get(a.entry)
        label_col = label_col if label_col in table.manifest.columns else None
        names_of: dict[str, list[str]] = {}                   # an entry named twice (its ID and its name) is one exclusion (review F1)
        for name in dict.fromkeys(a.exclude):
            names_of.setdefault(_named_in_trial(table, a.entry, label_col, name), []).append(name)

        def plots_of(entry_ids: set[str]) -> int:             # plots, not values: a plot measured on two dates is one plot
            return len({row.get(UNIT_COL, "").strip() or f"(row {i})" for i, (row, _) in enumerate(used) if row[a.entry].strip() in entry_ids})

        for entry_id, names in names_of.items():
            label = " / ".join(names)
            left_out[label] = ([entry_id], plots_of({entry_id}))
            exclusions.append(Exclusion(reason=f"plots left out: excluded entry {label}", count=left_out[label][1]))
            if not left_out[label][1]:                        # named, no value in the trial: still counted with 0 plots, and named
                no_value.append(label)                        # apart in the words (coach 2026-10-23 09:51 UTC, (iv); T4's and T5's)
        dropped = set(names_of)
        excluded_plots = plots_of(dropped)
        used = [(row, value) for row, value in used if row[a.entry].strip() not in dropped]
    named_trait = (getattr(table.manifest.measurement, "trait", None) or variable).replace(" | ", " / ")
    named_apart = f"; named, no {named_trait} value in the trial: {', '.join(no_value)}" if no_value else ""
    with_plots = [label for label, (_, n) in left_out.items() if n]
    excluded_note = lineage_note = None
    if left_out:
        excluded_note = (f"left out by name: {', '.join(with_plots)} ({excluded_plots} plots)" if with_plots
                         else "left out by name: no entry with a value") + named_apart
        lineage_note = (f"left out by name: {', '.join(with_plots)} ({excluded_plots} plots; entries {sorted(dropped)})" if with_plots
                        else f"left out by name: no entry with a value (entries {sorted(dropped)})") + named_apart
    by_plot: dict[str, list[tuple[dict[str, str], float]]] = {}
    for i, (row, value) in enumerate(used):
        by_plot.setdefault(row.get(UNIT_COL, "").strip() or f"(row {i} without a plot ID)", []).append((row, value))
    repeated = {plot: rows for plot, rows in by_plot.items() if len(rows) > 1}
    if repeated:
        plot, rows = sorted(repeated.items())[0]
        if a.duplicate_policy != "mean_per_unit":
            raise AnalystError("unsupported", f"plot {plot} has {len(rows)} values (several dates or timepoints), and {len(repeated) - 1} other "
                                              "plot(s) more than one; an analysis of variance needs one value per plot: pass "
                                              "duplicate_policy='mean_per_unit' to average each plot's values first")
        for plot, rows in sorted(repeated.items()):
            if len({(r.get(a.entry, "").strip(), r.get(a.rep, "").strip()) for r, _ in rows}) > 1:
                raise AnalystError("unsupported", f"plot {plot} is recorded with more than one entry or rep; its values cannot be averaged")
        used = [(rows[0][0], math.fsum(v for _, v in rows) / len(rows)) for rows in by_plot.values()]
        averaged = (f"each plot's values averaged over its dates ({sum(len(rows) for rows in repeated.values())} values in "
                    f"{len(repeated)} plots)")
        transformation = averaged if transformation.startswith("valid values used as recorded") else f"{transformation}; {averaged}"
    has_reps = a.rep in table.manifest.columns
    stated = sum(1 for row, _ in used if row.get(a.rep, "").strip()) if has_reps else 0
    if 0 < stated < len(used):
        raise AnalystError("unsupported", f"{len(used) - stated} plot(s) state no rep in column {a.rep!r} while {stated} do; an RCBD needs "
                                          "every plot's rep (and a one-way analysis none)")
    rcbd = stated > 0
    if rcbd:
        clashes = _rep_code_clashes(row[a.rep].strip() for row, _ in used)
        if clashes:
            raise AnalystError("unsupported", f"rep codes {' and '.join(repr(c) for c in clashes[0])} are the same number written "
                                              "differently, so they would count as different reps; write each rep one way")
    stated_blocks = BLOCK_COL in table.manifest.columns and any(row.get(BLOCK_COL, "").strip() for row, _ in used)
    blocks_in_reps = False
    if stated_blocks and rcbd:
        per_rep: dict[str, set[str]] = {}
        for row, _ in used:
            per_rep.setdefault(row[a.rep].strip(), set()).add(row.get(BLOCK_COL, "").strip())
        blocks_in_reps = any(len(codes - {""}) > 1 for codes in per_rep.values())
    plots = [(row[a.entry].strip(), row[a.rep].strip() if rcbd else None, value) for row, value in used]
    entries = sorted({e for e, _, _ in plots})
    if len(entries) < 2:
        raise AnalystError("unsupported", f"an analysis of variance needs at least 2 entries; the table has {len(entries)} with a valid value")
    cells_note = None
    averaged_cells = False
    if rcbd:
        cells = Counter((e, r) for e, r, _ in plots)
        doubled = [(e, r, k) for (e, r), k in sorted(cells.items()) if k > 1]
        if doubled and a.cells == "refuse":
            e, r, k = doubled[0]
            raise AnalystError("unsupported", f"entry {e} has {k} plots in rep {r}: not an RCBD (one plot per entry and rep); "
                                              f"{len(doubled)} entry x rep cell(s) hold more than one plot. Only if the person asks: "
                                              "cells='mean' averages a cell's plots, cells='single_only' keeps only the entries with exactly "
                                              "one plot in every rep")
        if a.cells == "mean" and doubled:                         # C28 cell rule (coach 2026-10-23 00:49 UTC), the person's choice
            units_before = len({row[UNIT_COL] for row, _ in used}) if classified.units_known else None
            values_of: dict[tuple[str, Any], list[float]] = {}
            for e, r, v in plots:
                values_of.setdefault((e, r), []).append(v)
            plots = [(e, r, math.fsum(vs) / len(vs)) for (e, r), vs in values_of.items()]
            averaged_cells = True
            cells_note = (f"cells='mean': the plots of {len(doubled)} entry x rep cell(s) averaged ({sum(k for *_, k in doubled)} plots into "
                          "one value each); the error mean square mixes single plots and cell means: approximate")
        elif a.cells == "single_only":
            all_reps = {r for _, r, _ in plots}
            per_entry: dict[str, dict[Any, int]] = {}
            for (e, r), k in cells.items():
                per_entry.setdefault(e, {})[r] = k
            several = {e for e, counts in per_entry.items() if max(counts.values()) > 1}
            gaps = {e for e, counts in per_entry.items() if e not in several and len(counts) < len(all_reps)}
            keep = set(per_entry) - several - gaps
            if len(keep) < 2:
                raise AnalystError("unsupported", f"cells='single_only' keeps {len(keep)} entries with exactly one plot in every rep; an "
                                                  "analysis of variance needs at least 2")
            if keep != set(per_entry):                            # a rule that leaves nothing out changes nothing (review)
                for reason, left in (("entries with several plots in a rep", several), ("entries with a rep missing", gaps)):
                    if left:
                        exclusions.append(Exclusion(reason=f"plots left out by cells='single_only': {reason}",
                                                    count=sum(1 for p in plots if p[0] in left)))
                plots = [p for p in plots if p[0] in keep]
                used = [(row, value) for row, value in used if row[a.entry].strip() in keep]
                cells_note = (f"cells='single_only': the entries with exactly one plot in every rep, {len(keep)} of {len(per_entry)} "
                              f"({len(several)} with several plots in a rep and {len(gaps)} with a rep missing left out)")
        if cells_note:
            entries = sorted({e for e, _, _ in plots})
            cells = Counter((e, r) for e, r, _ in plots)
        reps = sorted({r for _, r, _ in plots})  # type: ignore[type-var]
        if len(reps) < 2:
            raise AnalystError("unsupported", f"an RCBD needs at least 2 reps; every plot states rep {reps[0]!r}")
        groups = _groups_linked(list(cells))
        if groups > 1:
            raise AnalystError("unsupported", f"the design is not connected: its entries and reps fall into {groups} separate groups, "
                                              "so entry and rep effects cannot be told apart")
        df_error = len(plots) - len(entries) - len(reps) + 1
    else:
        df_error = len(plots) - len(entries)
    if df_error < 1:
        raise AnalystError("unsupported", f"no error degrees of freedom ({len(plots)} plots, {len(entries)} entries"
                                          + (f", {len({r for _, r, _ in plots})} reps" if rcbd else "") + "): at least 2 plots of some entry are needed")
    res = _anova(plots)
    counts = list(res.plots_per_entry.values())
    equal = len(set(counts)) == 1
    r_eff = float(counts[0]) if equal else len(counts) / math.fsum(1.0 / k for k in counts)
    approximate = res.adjusted or not equal
    if not approximate:
        basis = f"every entry has {counts[0]} {'cell' if averaged_cells else 'plot'}(s)"
    elif equal:
        basis = (f"r = {r_eff:.6g} {'cells' if averaged_cells else 'plots'} per entry, but {res.empty_cells} entry x rep cell(s) have no plot, so the least-squares entry "
                 "means are correlated: SE, SED and LSD are approximate (the SED of least-squares means differs per pair: larger on average, and "
                 "for some pairs much larger)")
    else:
        basis = (f"r = {r_eff:.6g}, the harmonic mean of the entries' {'cell' if averaged_cells else 'plot'} counts ({min(counts)} to {max(counts)}): SE, SED, LSD and H2 "
                 "are approximate")
    ms_e, ms_g = res.ms_error, res.ms_entry
    zero_error = ms_e == 0
    se, sed = math.sqrt(ms_e / r_eff), math.sqrt(2 * ms_e / r_eff)
    lsd = None if zero_error else pvalues.t_quantile(0.975, res.df_error) * sed
    cv = 100 * math.sqrt(ms_e) / res.grand_mean if res.grand_mean > 0 else None
    s2g_raw = (ms_g - ms_e) / r_eff
    set_to_zero = s2g_raw < 0
    s2g = 0.0 if set_to_zero else s2g_raw
    h2 = (s2g / (s2g + ms_e / r_eff)) if (s2g + ms_e / r_eff) > 0 else None
    for row in res.rows:                                      # p of each F, from pvalues (T1)
        if row["f"] is not None:
            row["p"] = pvalues.f_upper_p(row["f"], row["df"], res.df_error)

    aid = table.manifest.artifact_id
    measurement = table.manifest.measurement
    unit = getattr(measurement, "unit", None)
    trait = getattr(measurement, "trait", None) or variable
    study = next(iter(studies))
    variant = hashlib.sha256(json.dumps([a.value, a.entry, a.rep, a.duplicate_policy, a.allow_incomplete]
                                        + ([a.cells] if cells_note else []) + ([sorted(left_out)] if left_out else [])).encode()).hexdigest()[:8]
    namespace = f"anova-{variant}"
    units_word = "entry x rep cells" if averaged_cells else "plots"
    shape = f"{res.n} {units_word}: {res.entries} entries x {res.reps} reps" if rcbd else f"{res.n} plots: {res.entries} entries, no reps"
    context = [f"{trait} in study {study}", shape]
    if res.adjusted:
        context.append(f"sequential least squares, reps first (the OLS fit): {res.empty_cells} entry x rep cell(s) without a plot")     # Luna E16.1: "adjusted" read as not OLS
    if approximate:
        context.append(basis)
    if blocks_in_reps:
        context.append("incomplete blocks not modelled: the plots state several blocks within a rep (an incomplete-block design such as "
                       "an alpha lattice); this analyses the enclosing RCBD, so any block variance stays in the error: if the blocks differ, "
                       "MS error, CV, SED and LSD are larger and H2 lower than an incomplete-block analysis would give")
    elif stated_blocks and not rcbd:
        context.append("blocks not modelled: the plots state blocks but no reps; this one-way analysis ignores the blocks (if the "
                       "blocks are the reps, pass rep='block')")
    if cells_note:
        context.append(cells_note)
    if excluded_note:
        context.append(excluded_note)
    context += [transformation, ANOVA_NOTE] + ([gate] if gate else [])
    tail = " | " + " | ".join(context)
    n_units = len({row[UNIT_COL] for row, _ in used}) if classified.units_known else None
    if averaged_cells:
        n_units = units_before                                # the distinct plots behind the cells; None when unknown (review)
    common = {"denominator": Denominator(name="entry x rep cells in the analysis (the plots of some averaged)" if averaged_cells
                                         else "plots in the analysis", n=res.n), "n_independent_units": n_units,
              "source_artifact_hashes": [table.manifest.sha256], "grouping": [a.entry] + ([a.rep] if rcbd else []), "exclusions": exclusions}
    claims: list[Claim] = []
    for row in res.rows:
        for quantity in ("df", "ss", "ms", "f", "p"):
            value = row[quantity]
            if value is None and not (quantity in ("f", "p") and row["source"] in ("rep", "entry")):
                continue                                      # not part of the table: MS of total, F and p of error and total
            claims.append(Claim(claim_id=_claim_id(aid, "anova", row["source"], quantity, namespace=namespace), kind="anova",
                                value=value, missing_reason=ZERO_ERROR if value is None else None,
                                transformation=f"{res.design} ANOVA: {quantity} of {row['source']}{tail}"
                                               + (P_UNDERFLOW if quantity == "p" and value == 0 else ""), **common))
    claims.append(Claim(claim_id=_claim_id(aid, "mean", "grand", namespace=namespace), kind="mean", value=res.grand_mean, unit=unit,
                        transformation=f"grand mean{tail}", **common))
    for name, value, value_unit, reason in (("CV", cv, "%", "the grand mean is not above 0"),
                                            ("SE of an entry mean", se, unit, None), ("SED", sed, unit, None), ("LSD (5%)", lsd, unit, ZERO_ERROR)):
        claims.append(Claim(claim_id=_claim_id(aid, "precision", name, namespace=namespace), kind="precision", value=value, unit=value_unit,
                            missing_reason=reason if value is None else None, transformation=f"{name}{tail}", **common))
    flag = " | sigma2 g estimate negative, set to 0" if set_to_zero else ""
    for name, value in (("genotype variance sigma2 g", s2g), ("error variance sigma2 e", ms_e)):
        claims.append(Claim(claim_id=_claim_id(aid, "variance", name, namespace=namespace), kind="variance", value=value,
                            transformation=f"{name}{tail}{flag if name.startswith('genotype') else ''}", **common))
    claims.append(Claim(claim_id=_claim_id(aid, "heritability", "entry-mean", namespace=namespace), kind="heritability", value=h2,
                        missing_reason=None if h2 is not None else "no variance between or within entries",
                        transformation=f"entry-mean heritability H2{tail} | {H2_TEXT}{flag}", **common))
    # the headline numbers first: the coordinator's view keeps 20 claims (coordination_runtime VIEW_CLAIMS), and on Luna
    # (2026-10-23, A40.5) H2 was claim 23 of 23 and never reached it
    headline = {name: i for i, name in enumerate((
        "entry-mean heritability H2", f"{res.design} ANOVA: ms of entry", f"{res.design} ANOVA: ms of error", "genotype variance sigma2 g",
        "error variance sigma2 e", f"{res.design} ANOVA: f of entry", f"{res.design} ANOVA: p of entry", "CV", "SED", "LSD (5%)",
        "SE of an entry mean", "grand mean"))}                        # the mean squares right after H2: 4 calls share 20 claims (review)
    claims.sort(key=lambda claim: headline.get(claim.transformation.split(" | ")[0], len(headline)))

    result_rows = [{"source": row["source"], "df": str(row["df"]),
                    **{q: "" if row[q] is None else repr(row[q]) for q in ("ss", "ms", "f", "p")}} for row in res.rows]
    lineage = [f"lineage: anova_trial of {aid} sha256={table.manifest.sha256}", f"design: {res.design}" + (", sequential least squares, reps first (the OLS fit)" if res.adjusted else "")
               + (f"; {cells_note}" if cells_note else ""),
               *([lineage_note] if lineage_note else []),
               f"analysis target: {ANALYSIS_TARGET}"]
    manifest = registry.save_table(result_rows, kind="analysis_result", columns=ANOVA_RESULT_COLUMNS, complete=table.manifest.complete,
                                   source_request_ids=list(table.manifest.source_request_ids), study_ids=[study], variable_ids=[variable],
                                   quality_notes=lineage)
    warnings = [w for w in (gate, classified.duplicate_note if a.duplicate_policy != "reject" else None) if w]
    if not has_reps:
        warnings.append(f"the table has no {a.rep!r} column: one-way analysis (entries only); for an RCBD, plot_design adds each plot's rep")
    elif not rcbd:
        warnings.append(f"no plot states a rep in column {a.rep!r}: one-way analysis (entries only)")
    if cells_note:
        warnings.append(cells_note)
    if excluded_note:
        warnings.append(excluded_note)
    if res.adjusted:
        warnings.append(f"{res.empty_cells} entry x rep cell(s) have no plot: sequential least squares, reps first (the OLS fit)")
    if approximate:
        warnings.append(f"approximate precision: {basis}")
    if zero_error:
        warnings.append(f"{ZERO_ERROR}; F, p and the LSD are not available")
    if blocks_in_reps:
        warnings.append("incomplete blocks not modelled: the plots state several blocks within a rep; analysed as the enclosing RCBD "
                        "(any block variance stays in the error)")
    elif stated_blocks and not rcbd:
        warnings.append("blocks not modelled: the plots state blocks but no reps; this one-way analysis ignores them (if the blocks "
                        "are the reps, pass rep='block')")
    if set_to_zero:
        warnings.append("MS entry is below MS error: the genotype variance estimate is negative and is set to 0, so H2 = 0")
    if not classified.units_known:
        warnings.append("independence unknown: some valid rows name no observation unit")
    warnings.extend(_timing_warnings(table, a.value))
    data = {"artifact_id": aid, "result_artifact_id": manifest.artifact_id, "design": res.design, "balanced": res.balanced,
            "sequential_least_squares": res.adjusted, "anova": res.rows, "plots_used": res.n, "cells_rule": a.cells, "cells_note": cells_note,
            "excluded": [{"name": name, "entries": ids, "plots": n} for name, (ids, n) in left_out.items()], "named_no_value": no_value,
            "excluded_plots": excluded_plots,
            "grand_mean": res.grand_mean, "cv_percent": cv, "se": se,
            "sed": sed, "lsd_5pct": lsd, "t_0975": pvalues.t_quantile(0.975, res.df_error), "sigma2_g": s2g, "sigma2_e": ms_e,
            "sigma2_g_estimate": s2g_raw, "h2": h2, "h2_set_to_zero": set_to_zero, "replication": r_eff, "replication_basis": basis,
            "balance": {"entries": res.entries, "reps": res.reps, "plots": res.n, "empty_cells": res.empty_cells,
                        "plots_per_entry_min": min(counts), "plots_per_entry_max": max(counts)},
            "display": {k: _display(v) for k, v in (("grand_mean", res.grand_mean), ("cv_percent", cv), ("lsd_5pct", lsd), ("h2", h2))},
            "claims": _claims_json(claims), "note": ANOVA_NOTE, "h2_text": H2_TEXT, "analysis_target": ANALYSIS_TARGET, "value_policy": VALUE_POLICY}
    return ToolResult(ok=True, data=data, artifact_ids=[aid, manifest.artifact_id], complete=table.manifest.complete, warnings=warnings)

# T3 (coach 2026-10-22 22:50 / 23:50 UTC; research build order, tool 5): the genotype x environment partition of ONE trait's
# entry means across studies. Measured first (t3_gxe_partition_20261023/UNLOCKS.md): D16.3, E12.1, E12.2, E12.5, E17.1, E17.2,
# E17.3 and E17.5 need it.
GXE_NOTE = "descriptive partition of recorded entry means across trials, not a recommendation"
GXE_SOURCES = ("environment", "entry", "entry x environment")
GXE_PLOT_SOURCES = (*GXE_SOURCES, "reps within environment", "error (pooled within trials)")   # the plot-level shares (D16.3)
GXE_ZERO = "the entry x environment mean square is 0: the entry means are exactly additive, so F is undefined"
GXE_H2_TEXT = ("across-environment entry-mean heritability H2 = 1 - MS entry x environment / MS entry of the means table (on balanced "
               "data, and when no variance component is set to 0, equal to sigma2 g / (sigma2 g + sigma2 ge / e + sigma2 e / (r e))); "
               "environments random, so G x E is its denominator")
GXE_BLOCKS = ("incomplete blocks not modelled: the plots state several blocks within a rep; each study is analysed as the enclosing RCBD, "
              "so any block variance stays in the error")
GXE_RESULT_COLUMNS = {
    "entry": "exact entry ID", "label": "name carried from the label column, if any", "study": "study ID (one environment)",
    "mean": "the entry's mean in the study: the mean of its plots' values, full precision", "n_plots": "plots behind the mean",
    "kept": "true when the entry has a mean in every study and is in the partition",
    "left_out": "why the entry is not in the partition ('' when kept): excluded, outside the name prefix, not in every study",
}


def _entries_in_any(tables: list[Table], by: str, label_col: str | None, wanted: list[str], what: str) -> set[str]:
    """_entries over several tables: each wanted entry is resolved in the first table that knows it; one no table knows is
    refused with the first table's message."""
    found: set[str] = set()
    for name in wanted:
        errors: list[AnalystError] = []
        for table in tables:
            try:
                found |= set(_entries(table, by, label_col, [name], what))
                break
            except AnalystError as error:
                errors.append(error)
        else:
            raise errors[0]
    return found


def tool_gxe_partition(registry: ArtifactRegistry, a: GxePartitionArgs) -> ToolResult:
    """The genotype x environment partition of ONE trait's entry means across studies (each study one environment; research
    build order, tool 5). An entry's mean in a study is the mean of its plots' values; only the entries with a mean in every
    study are kept (the others are counted and saved). On the entries x studies means table: the environment, entry and entry
    x environment sums of squares with df and MS, their shares of the table's sum of squares, the F for entries tested against
    entry x environment (environments random) with its p, the grand mean and the across-environment entry-mean heritability
    H2 = 1 - MS entry x environment / MS entry (set to 0 and flagged when negative). When every plot states its rep
    (plot_design's tables) and every study is a complete RCBD of the kept entries with the same r: the pooled within-trial
    error mean square and the variance components on the means scale, sigma2 e (pooled), sigma2 ge = MS entry x environment -
    sigma2 e / r, sigma2 g = (MS entry - MS entry x environment) / studies (each set to 0 and flagged when negative), and
    r_B = sigma2 g / (sigma2 g + sigma2 ge), and the shares of the plot values' total sum of squares: environment, entry and entry
    x environment (r x the means-table sums), reps within environment and the pooled error; otherwise they are not available,
    with the reason (a mixed model would be needed).
    Input: one table per study, or one table holding several studies (concat_tables). Refused, never guessed: fewer than 2
    studies, fewer than 2 entries with a mean in every study, more than one variable, the same table twice, a plot in two
    tables, several values of one plot unless duplicate_policy='mean_per_unit'. Descriptive numbers of recorded values, never
    a choice of entries or a stability call."""
    tables = [_load(registry, handle) for handle in a.artifact_ids]
    seen: dict[str, str] = {}
    for table in tables:
        for key in (table.manifest.artifact_id, table.manifest.sha256):
            if key in seen:
                raise AnalystError("invalid_argument", f"{seen[key]} and {table.manifest.artifact_id} hold the same rows; pass each table once")
            seen[key] = table.manifest.artifact_id
    if a.by.endswith("Name"):
        raise AnalystError("invalid_argument", f"use the entry ID column, not {a.by!r}; names are carried as labels")
    variables: set[str] = set()
    for table in tables:
        _require_column(table, a.by, "entry column")
        _require_column(table, a.value, "value column")
        _require_column(table, "studyDbId", "study column")
        variables.add(_single_variable(table))
    if len(variables) > 1:
        raise AnalystError("unsupported", f"the tables hold {len(variables)} variables {sorted(variables)}; a partition needs one variable (one trait)")
    variable = next(iter(variables))
    label_col = LABEL_FOR.get(a.by)
    label_col = label_col if label_col is not None and all(label_col in table.manifest.columns for table in tables) else None
    if a.name_prefix is not None and label_col is None:
        raise AnalystError("invalid_argument", f"name_prefix needs a name column beside {a.by!r} in every table")
    excluded = _entries_in_any(tables, a.by, label_col, a.exclude, "exclude")

    gates: list[str] = []
    notes: list[str] = []
    exclusions: list[Exclusion] = []
    labels: dict[str, set[str]] = {}
    plots: list[tuple[str, str, str, float, str, str]] = []      # (study, entry, plot, value, rep, block)
    plot_table: dict[str, str] = {}
    entry_peak: dict[str, float] = {}                         # entry -> the largest absolute raw value behind its means
    repeated_plots = repeated_values = no_study = no_entry = 0
    units_known = True
    for table in tables:
        gate = _completeness_gate(table, a.allow_incomplete)
        if gate:
            gates.append(f"{table.manifest.artifact_id}: {gate}")
        classified = _classify(table, a.value)
        units_known = units_known and classified.units_known
        for row, number in classified.valid:                  # the raw values, before any averaging (T4's residue rule)
            if row.get(a.by, "").strip():
                entry_peak[row[a.by].strip()] = max(entry_peak.get(row[a.by].strip(), 0.0), abs(number))
        used, transformation = _apply_duplicate_policy(table, classified, a.duplicate_policy, value=a.value)
        notes.append(transformation)
        exclusions += _exclusions(classified)
        if label_col:
            for row in table.rows:
                if row.get(a.by, "").strip() and row.get(label_col, ""):
                    labels.setdefault(row[a.by].strip(), set()).add(row[label_col])
        by_plot: dict[str, list[tuple[dict[str, str], float]]] = {}
        for i, (row, value) in enumerate(used):
            if not row.get("studyDbId", "").strip():
                no_study += 1
            elif not row.get(a.by, "").strip():
                no_entry += 1
            else:
                plot = row.get(UNIT_COL, "").strip() or f"(row {i} of {table.manifest.artifact_id} without a plot ID)"
                by_plot.setdefault(plot, []).append((row, value))
        for plot, rows in sorted(by_plot.items()):
            if plot in plot_table:
                raise AnalystError("unsupported", f"plot {plot} is in both {plot_table[plot]} and {table.manifest.artifact_id}; pass tables that "
                                                  "do not overlap")
            plot_table[plot] = table.manifest.artifact_id
            if len(rows) > 1:
                if a.duplicate_policy != "mean_per_unit":
                    raise AnalystError("unsupported", f"plot {plot} has {len(rows)} values (several dates or timepoints); a partition of entry "
                                                      "means needs one value per plot: pass duplicate_policy='mean_per_unit' to average each "
                                                      "plot's values first")
                if len({(r.get(a.by, "").strip(), r.get("studyDbId", "").strip(), r.get(a.rep, "").strip()) for r, _ in rows}) > 1:
                    raise AnalystError("unsupported", f"plot {plot} is recorded with more than one entry, study or rep; its values cannot be averaged")
                repeated_plots += 1
                repeated_values += len(rows)
            first = rows[0][0]
            plots.append((first["studyDbId"].strip(), first[a.by].strip(), plot, math.fsum(v for _, v in rows) / len(rows),
                          first.get(a.rep, "").strip(), first.get(BLOCK_COL, "").strip() if BLOCK_COL in table.manifest.columns else ""))
    kept_notes = list(dict.fromkeys(notes))
    if repeated_plots:                                        # every table's note stays; only "as recorded" is no longer true (review 1)
        kept_notes = [note for note in kept_notes if not note.startswith("valid values used as recorded")]
        kept_notes.append(f"each plot's values averaged over its dates ({repeated_values} values in {repeated_plots} plots)")
    transformation = "; ".join(kept_notes)
    for reason, count in (("rows with an empty studyDbId", no_study), (f"rows with an empty {a.by}", no_entry)):
        if count:
            exclusions.append(Exclusion(reason=reason, count=count))

    studies = sorted({study for study, *_ in plots})
    if len(studies) < 2:
        raise AnalystError("unsupported", f"a partition across environments needs at least 2 studies; the tables hold {len(studies)} with a "
                                          f"valid value ({', '.join(studies) or 'none'})")
    cells: dict[str, dict[str, list[float]]] = {}
    for study, entry, _, value, *_ in plots:
        cells.setdefault(entry, {}).setdefault(study, []).append(value)
    outside = {e for e in cells if a.name_prefix is not None and e not in excluded and not any(n.startswith(a.name_prefix) for n in labels.get(e, ()))}
    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    for entry in sorted(cells):
        reason = ("excluded" if entry in excluded else "outside the name prefix" if entry in outside
                  else "" if len(cells[entry]) == len(studies) else f"not in every study ({len(cells[entry])} of {len(studies)})")
        if reason:
            dropped.append((entry, reason))
        else:
            kept.append(entry)
    if len(kept) < 2:
        raise AnalystError("unsupported", f"a partition needs at least 2 entries with a mean in every study; {len(kept)} of {len(cells)} have "
                                          f"one ({len(dropped)} left out)")

    # the means table, centred on its grand mean (the sums of squares do not change; less cancellation)
    g, e = len(kept), len(studies)
    means = [[math.fsum(cells[entry][study]) / len(cells[entry][study]) for study in studies] for entry in kept]
    grand = math.fsum(v for row in means for v in row) / (g * e)
    centred = [[v - grand for v in row] for row in means]
    row_mean = [math.fsum(row) / e for row in centred]
    col_mean = [math.fsum(centred[i][j] for i in range(g)) / g for j in range(e)]
    ss_total = math.fsum(v * v for row in centred for v in row)
    ss = {"environment": g * math.fsum(m * m for m in col_mean), "entry": e * math.fsum(m * m for m in row_mean),
          "entry x environment": math.fsum((centred[i][j] - row_mean[i] - col_mean[j]) ** 2 for i in range(g) for j in range(e))}
    # Each source is judged against the largest raw plot value behind the means (T4's residue rule): a source whose effects are float
    # residue is 0, as on paper. The old rule judged G x E against the total SS, which stated H2 1.0 and F 1.8e10 from residue and set a
    # real G x E to 0 beside a large environment effect (the one review of NEXT 2, F1-F3).
    peak = max(entry_peak.get(entry, 0.0) for entry in kept)
    zeroed = [source for source in GXE_SOURCES if math.sqrt(ss[source] / (g * e)) <= PAIR_ZERO * peak]
    for source in zeroed:
        ss[source] = 0.0
    if zeroed:
        ss_total = math.fsum(ss.values())                     # the parts of a two-way table without replication add up to its total
    df = {"environment": e - 1, "entry": g - 1, "entry x environment": (g - 1) * (e - 1)}
    ms = {source: ss[source] / df[source] for source in GXE_SOURCES}
    f_entry = ms["entry"] / ms["entry x environment"] if ms["entry x environment"] > 0 else None
    p_entry = pvalues.f_upper_p(f_entry, df["entry"], df["entry x environment"]) if f_entry is not None else None
    rows = [{"source": source, "df": df[source], "ss": ss[source], "ms": ms[source], "f": f_entry if source == "entry" else None,
             "p": p_entry if source == "entry" else None} for source in GXE_SOURCES]
    rows.append({"source": "total", "df": g * e - 1, "ss": ss_total, "ms": None, "f": None, "p": None})
    explained = math.fsum(ss.values())
    shares = {source: (100 * ss[source] / explained if explained > 0 else None) for source in GXE_SOURCES}
    h2_raw = 1 - ms["entry x environment"] / ms["entry"] if ms["entry"] > 0 else None
    h2_set_to_zero = h2_raw is not None and h2_raw < 0
    h2 = 0.0 if h2_set_to_zero else h2_raw

    vc: dict[str, Any] = {"available": False, "reason": None, "r": None, "df_error": None, "sigma2_e": None, "sigma2_ge": None, "sigma2_g": None,
                          "r_b": None, "h2_long_form": None, "sigma2_g_set_to_zero": False, "sigma2_ge_set_to_zero": False, "plot_ss": None}
    kept_set = set(kept)
    in_partition = [p for p in plots if p[1] in kept_set]
    if not all(a.rep in table.manifest.columns for table in tables) or any(not plot[4] for plot in in_partition):
        vc["reason"] = (f"the plots state no rep in column {a.rep!r} (plot_design adds it): the means-table partition stands; the pooled error "
                        "and the variance components need each study's reps")
    else:
        per_study: dict[str, list[tuple[str, str, float]]] = {}
        blocks_in: dict[tuple[str, str], set[str]] = {}
        for study, entry, _, value, rep, block in in_partition:
            per_study.setdefault(study, []).append((entry, rep, value))
            if block:
                blocks_in.setdefault((study, rep), set()).add(block)
        reps_of = {study: sorted({rep for _, rep, _ in items}) for study, items in per_study.items()}
        cells_of = {study: Counter((entry, rep) for entry, rep, _ in items) for study, items in per_study.items()}
        # one reason per condition (review 1); rep codes belong to their own study, so they are compared within each study
        clash = next(((study, found[0]) for study in studies for found in [_rep_code_clashes(reps_of[study])] if found), None)
        doubled = [(study, entry, rep, n) for study in studies for (entry, rep), n in sorted(cells_of[study].items()) if n > 1]
        one_rep = [study for study in studies if len(reps_of[study]) < 2]
        missing = [(study, g * len(reps_of[study]) - len(cells_of[study])) for study in studies
                   if len(cells_of[study]) != g * len(reps_of[study])]     # every study counted, not the first (review 2)
        stands = "the means-table partition stands"
        if clash:
            vc["reason"] = (f"rep codes {' and '.join(repr(code) for code in clash[1])} of study {clash[0]} are the same number written "
                            f"differently; the variance components are not given ({stands})")
        elif doubled:
            first_cell = doubled[0]
            more = (f" ({len(doubled)} such cells in {len({cell[0] for cell in doubled})} studies)" if len(doubled) > 1 else "")
            vc["reason"] = (f"entry {first_cell[1]} has {first_cell[3]} plots in rep {first_cell[2]} of study {first_cell[0]}{more}: not an RCBD, "
                            f"so the variance components are not given ({stands})")
        elif one_rep:
            vc["reason"] = (f"{len(one_rep)} of {len(studies)} studies have one rep ({', '.join(one_rep[:10])}{', ...' if len(one_rep) > 10 else ''}): "
                            f"with one plot per entry, G x E and the plot error cannot be told apart, so the variance components are not given "
                            f"({stands})")
        elif len({len(reps_of[study]) for study in studies}) != 1:
            vc["reason"] = (f"the studies have different numbers of reps ({', '.join(f'{study}: {len(reps_of[study])}' for study in studies)}): "
                            f"the variance components need a mixed model (REML); {stands}")
        elif missing:
            vc["reason"] = (f"{sum(n for _, n in missing)} entry x rep cell(s) have no plot ({'; '.join(f'study {s}: {n}' for s, n in missing[:10])}"
                            f"{'; ...' if len(missing) > 10 else ''}): the variance components need a mixed model (REML); {stands}")
        else:
            r = len(reps_of[studies[0]])
            ss_e = df_e = ss_rep = 0.0
            for study in studies:
                result = _anova([(entry, rep, value) for entry, rep, value in per_study[study]])
                ss_e += next(row["ss"] for row in result.rows if row["source"] == "error")
                ss_rep += next(row["ss"] for row in result.rows if row["source"] == "rep")
                df_e += result.df_error
            # the plot values' total sum of squares, balanced: r x each means-table sum, the reps within each study and the error
            vc["plot_ss"] = {**{source: r * ss[source] for source in GXE_SOURCES}, "reps within environment": ss_rep,
                             "error (pooled within trials)": ss_e}
            s2e = ss_e / df_e
            s2ge_raw = ms["entry x environment"] - s2e / r
            s2g_raw = (ms["entry"] - ms["entry x environment"]) / e
            s2ge, s2g = max(s2ge_raw, 0.0), max(s2g_raw, 0.0)
            vc.update(available=True, r=r, df_error=int(df_e), sigma2_e=s2e, sigma2_ge=s2ge, sigma2_g=s2g,
                      sigma2_g_set_to_zero=s2g_raw < 0, sigma2_ge_set_to_zero=s2ge_raw < 0,
                      r_b=s2g / (s2g + s2ge) if s2g + s2ge > 0 else None,
                      h2_long_form=s2g / (s2g + s2ge / e + s2e / (r * e)) if s2g + s2ge / e + s2e / (r * e) > 0 else None,
                      note=GXE_BLOCKS if any(len(codes) > 1 for codes in blocks_in.values()) else None)

    first_table = tables[0]
    aid = first_table.manifest.artifact_id
    unit = first_table.manifest.measurement.unit
    trait = first_table.manifest.measurement.trait or variable
    variant = hashlib.sha256(json.dumps([[t.manifest.sha256 for t in tables], a.by, a.value, a.rep, sorted(excluded), a.name_prefix,
                                         a.duplicate_policy, a.allow_incomplete]).encode("utf-8")).hexdigest()[:8]
    namespace = f"gxe-{variant}"
    context = [f"{trait} in {e} studies ({', '.join(studies)})",
               f"{g} entries with a mean in every study (of {len(cells)}; {len(dropped)} left out)", transformation, GXE_NOTE, *gates]
    tail = " | " + " | ".join(context)
    hashes = [t.manifest.sha256 for t in tables]
    for reason, count in Counter(reason.split(" (")[0] for _, reason in dropped).items():
        exclusions.append(Exclusion(reason=f"entries left out: {reason}", count=count))
    common = {"denominator": Denominator(name="entry means in the partition (entries x studies)", n=g * e), "n_independent_units": None,
              "source_artifact_hashes": hashes, "grouping": [a.by, "studyDbId"], "exclusions": exclusions}
    claims: list[Claim] = []
    for row in rows:
        for quantity in ("df", "ss", "ms", "f", "p"):
            value = row[quantity]
            if value is None and not (quantity in ("f", "p") and row["source"] == "entry"):
                continue                                      # not part of the table
            claims.append(Claim(claim_id=_claim_id(aid, "anova", row["source"], quantity, namespace=namespace), kind="anova", value=value,
                                missing_reason=GXE_ZERO if value is None else None,
                                transformation=f"means-table ANOVA: {quantity} of {row['source']}{tail}"
                                               + (P_UNDERFLOW if quantity == "p" and value == 0 else ""), **common))
    for source in GXE_SOURCES:
        claims.append(Claim(claim_id=_claim_id(aid, "ratio", "share", source, namespace=namespace), kind="ratio", value=shares[source], unit="%",
                            missing_reason=None if shares[source] is not None else "all entry means are equal: no sum of squares to share",
                            transformation=f"share of the means-table SS: {source}{tail}", **common))
    # the five parts must add up to the plot values' own total sum of squares: a means-table sum the residue rule set to 0, or squares
    # at the edge of the float range, break that, and then the shares are not given; a source that is exactly 0 keeps them (review 2)
    parts_total = math.fsum(vc["plot_ss"].values()) if vc["plot_ss"] else 0.0
    if vc["plot_ss"]:
        plot_centre = math.fsum(plot[3] for plot in in_partition) / len(in_partition)
        own_total = math.fsum((plot[3] - plot_centre) ** 2 for plot in in_partition)
        adds_up = abs(parts_total - own_total) <= 1e-5 * own_total
    plot_reason = (vc["reason"] if not vc["plot_ss"] else None if adds_up else
                   "the five parts do not add up to the plot values' total sum of squares (float residue at this scale): the plot-level "
                   "shares are not given")
    plot_total = parts_total if vc["plot_ss"] and adds_up else 0.0
    plots_in = {**common, "denominator": Denominator(name="plots in the partition (entries x studies x reps)", n=len(in_partition))}
    for source in GXE_PLOT_SOURCES:
        value = 100 * vc["plot_ss"][source] / plot_total if plot_total > 0 else None
        claims.append(Claim(claim_id=_claim_id(aid, "ratio", "plot-share", source, namespace=namespace), kind="ratio", value=value, unit="%",
                            missing_reason=None if value is not None else (plot_reason or "all plot values are equal: no sum of squares to share"),
                            transformation=f"share of the plot-level SS: {source}{tail}" + (f" | r = {vc['r']}" if vc["available"] else "")
                                           + (f" | {vc['note']}" if vc.get("note") else ""),
                            **plots_in))
    claims.append(Claim(claim_id=_claim_id(aid, "mean", "grand", namespace=namespace), kind="mean", value=grand, unit=unit,
                        transformation=f"grand mean of the entry means{tail}", **common))
    flag = " | MS entry is below MS entry x environment: set to 0" if h2_set_to_zero else ""
    claims.append(Claim(claim_id=_claim_id(aid, "heritability", "across-environments", namespace=namespace), kind="heritability", value=h2,
                        missing_reason=None if h2 is not None else "no variation among the entries' means",
                        transformation=f"across-environment entry-mean heritability H2{tail} | {GXE_H2_TEXT}{flag}", **common))
    for name, count in (("entries with a mean in every study", g), ("entries dropped: not in every study",
                                                                    sum(1 for _, reason in dropped if reason.startswith("not in every study"))),
                        ("studies (environments)", e)):
        claims.append(Claim(claim_id=_claim_id(aid, "count", name.split(":")[0].replace(" ", "-").replace("(", "").replace(")", ""), namespace=namespace),
                            kind="count", value=count, transformation=f"{name}{tail}", **common))
    for name, key, flagged in (("genotype variance sigma2 g", "sigma2_g", "sigma2_g_set_to_zero"),
                               ("genotype x environment variance sigma2 ge", "sigma2_ge", "sigma2_ge_set_to_zero"),
                               ("error variance sigma2 e (pooled within trials)", "sigma2_e", None)):
        note = (f" | r = {vc['r']}" if vc["available"] else "") + (f" | {vc['note']}" if vc.get("note") else "")
        note += " | estimate negative, set to 0" if flagged and vc[flagged] else ""
        claims.append(Claim(claim_id=_claim_id(aid, "variance", key, namespace=namespace), kind="variance", value=vc[key],
                            missing_reason=None if vc[key] is not None else vc["reason"], transformation=f"{name}{tail}{note}", **common))
    claims.append(Claim(claim_id=_claim_id(aid, "ratio", "r-b", namespace=namespace), kind="ratio", value=vc["r_b"],
                        missing_reason=None if vc["r_b"] is not None else (vc["reason"] or "sigma2 g and sigma2 ge are both 0"),
                        transformation=f"type-B genetic correlation r_B = sigma2 g / (sigma2 g + sigma2 ge){tail}", **common))

    # the headline numbers first: the coordinator's view keeps 20 claims (coordination_runtime VIEW_CLAIMS), and on Luna (2026-10-23)
    # anova_trial's heritability, claim 25 of 25, never reached it
    headline = {name: i for i, name in enumerate((
        "across-environment entry-mean heritability H2", "genotype variance sigma2 g", "genotype x environment variance sigma2 ge",
        "error variance sigma2 e (pooled within trials)", "type-B genetic correlation r_B = sigma2 g / (sigma2 g + sigma2 ge)",
        *(f"share of the means-table SS: {source}" for source in GXE_SOURCES), "means-table ANOVA: f of entry", "means-table ANOVA: p of entry",
        "means-table ANOVA: ms of entry", "means-table ANOVA: ms of entry x environment", "grand mean of the entry means",
        *(f"share of the plot-level SS: {source}" for source in GXE_PLOT_SOURCES)))}    # after the older headline numbers (review)
    claims.sort(key=lambda claim: len(headline) if claim.value is None and claim.transformation.startswith("share of the plot-level SS")
                else headline.get(claim.transformation.split(" | ")[0], len(headline)))       # not available: last

    left = dict(dropped)
    result_rows = [{"entry": entry, "label": "; ".join(sorted(labels.get(entry, ()))), "study": study,
                    "mean": repr(math.fsum(cells[entry][study]) / len(cells[entry][study])), "n_plots": str(len(cells[entry][study])),
                    "kept": "true" if entry in kept_set else "false", "left_out": left.get(entry, "")}
                   for entry in sorted(cells) for study in studies if study in cells[entry]]
    lineage = [f"lineage: gxe_partition of {', '.join(f'{t.manifest.artifact_id} sha256={t.manifest.sha256}' for t in tables)}",
               f"studies: {', '.join(studies)}", f"left out: excluded {sorted(excluded)}, name_prefix {a.name_prefix!r}",
               f"analysis target: {ANALYSIS_TARGET}"]
    complete = all(t.manifest.complete for t in tables)
    manifest = registry.save_table(result_rows, kind="analysis_result", columns=GXE_RESULT_COLUMNS, complete=complete,
                                   source_request_ids=list(dict.fromkeys(i for t in tables for i in t.manifest.source_request_ids)),
                                   study_ids=studies, variable_ids=[variable], measurement=first_table.manifest.measurement, quality_notes=lineage)
    warnings = list(gates)
    if dropped:
        warnings.append(f"{len(dropped)} entries left out of the partition (not in every study, excluded or outside the name prefix); "
                        f"{g} kept")
    if not vc["available"]:
        warnings.append(f"variance components not given: {vc['reason']}")
    elif vc.get("note"):
        warnings.append(vc["note"])
    if vc["plot_ss"] and plot_reason:
        warnings.append(f"plot-level shares not given: {plot_reason}")
    if h2_set_to_zero:
        warnings.append("MS entry is below MS entry x environment: the across-environment heritability is set to 0")
    if not units_known:
        warnings.append("independence unknown: some valid rows name no observation unit")
    data = {"artifact_ids": [t.manifest.artifact_id for t in tables], "result_artifact_id": manifest.artifact_id, "variable_id": variable,
            "unit": unit, "studies": studies, "entries_kept": g, "entries_dropped": len(dropped),
            "dropped": [{"entry": entry, "label": "; ".join(sorted(labels.get(entry, ()))), "reason": reason} for entry, reason in dropped[:50]],
            "anova": rows, "shares": shares, "grand_mean": grand, "h2": h2, "h2_set_to_zero": h2_set_to_zero, "variance_components": vc,
            "f_denominator": "entry x environment (environments random)", "transformation": transformation,
            "claims": _claims_json(claims), "note": GXE_NOTE, "h2_text": GXE_H2_TEXT, "analysis_target": ANALYSIS_TARGET, "value_policy": VALUE_POLICY}
    return ToolResult(ok=True, data=data, artifact_ids=[*(t.manifest.artifact_id for t in tables), manifest.artifact_id], complete=complete,
                      warnings=warnings)


# T4 (coach 2026-10-23 01:51 UTC, NEXT 4; research build order T4): ONE trait in two studies, on the entries with a mean in both.
# Measured first (t4_compare_trials_paired_20261023/UNLOCKS.md): 12 benchmark rows in one call (the paired t of C33.2, D14.6 and
# D16.1; the means and the percent of C27.1-2, C30.1-4, C33.1, C34.1 and D14.1) and 3 in one call per pair. compare_means (frozen,
# item (12)) is not used.
PAIR_NOTE = "descriptive comparison of recorded entry means of two trials, not a recommendation"
PAIR_ZERO = 1e-12           # a mean or an SD at or below this share of the largest plot value behind the means is 0: float residue
PAIR_PAIRED_NOTE = "the entries as the sampling units: the plot error within each trial is not used (not a mixed-model contrast)"
PAIR_WELCH_NOTE = "the same entries as two independent samples (ignores the pairing)"
PAIR_RESULT_COLUMNS = {
    "entry": "exact entry ID", "label": "name carried from the label column, if any",
    "paired": "true when the entry has a mean in both trials and is compared",
    "left_out": "why an entry is not compared ('' when paired): only in base, only in other, excluded, outside the name prefix",
    "base_mean": "the entry's mean in the base trial, full precision ('' = none)", "base_n_values": "valid values behind base_mean",
    "other_mean": "the entry's mean in the other trial, full precision ('' = none)", "other_n_values": "valid values behind other_mean",
    "difference": "other_mean - base_mean (base_mean - other_mean under percent='reduction'), full precision ('' when not paired)",
    "percent": "100 x difference / base_mean ('' when not paired, or when base_mean is 0 or negative)",
    "percent_note": "why percent is empty for a paired entry ('' otherwise)",
}


def _scaled_sd(values: list[float]) -> float:
    """The sample SD (n - 1), computed on the values divided by their largest absolute value: squares of 1e180 would overflow and
    squares of 1e-160 underflow (T4)."""
    scale = max(abs(v) for v in values)
    if scale == 0:
        return 0.0
    scaled = [v / scale for v in values]
    mean = math.fsum(scaled) / len(scaled)
    return scale * math.sqrt(math.fsum((v - mean) ** 2 for v in scaled) / (len(scaled) - 1))


@dataclass
class _Pairing:
    """Two one-study tables of one trait and the entries with a mean in both (compare_trials_paired, T4; stress_indices, T5)."""
    base_table: Table
    other_table: Table
    base: _EntryMeans
    other: _EntryMeans
    label_col: str | None
    names: dict[str, set[str]]
    excluded: set[str]
    outside: set[str]
    candidates: list[str]
    paired: list[str]
    only_base: int
    only_other: int
    removed: int
    excluded_labels: list[str]                      # the person's names per left-out entry with a mean, in their order ("E5 / clone E5")
    named_no_mean: list[str]                        # the person's names for entries with no mean in either trial (review finding 1)
    plot_lists: list[str]                           # the studies' plot lists given (roster_artifact_ids; coach 07:48 UTC, (5))


def _pair_two_trials(registry: ArtifactRegistry, base_id: str, other_id: str, a: Any, sides: tuple[str, str] = ("base", "other"),
                     needs: str = "a paired comparison needs") -> _Pairing:
    """Two measurement tables of ONE study each and one variable, their entry means and the entries with a mean in both; named
    entries left out (exclude) and one family kept (name_prefix); a name neither table holds is looked up in the studies' plot lists
    (roster_artifact_ids). Refused, never guessed: the same table, rows or study twice, a table of other than one study, two
    variables, name_prefix without a name column, fewer than 3 shared entries."""
    base_table, other_table = _load(registry, base_id), _load(registry, other_id)
    if base_table.manifest.artifact_id == other_table.manifest.artifact_id or base_table.manifest.sha256 == other_table.manifest.sha256:
        raise AnalystError("invalid_argument", f"{base_table.manifest.artifact_id} and {other_table.manifest.artifact_id} hold the same rows; "
                                               "compare two different studies")
    if a.by.endswith("Name"):
        raise AnalystError("invalid_argument", f"pair by the ID column, not {a.by!r}; names are carried as labels")
    label_col = LABEL_FOR.get(a.by)
    label_col = label_col if label_col in base_table.manifest.columns and label_col in other_table.manifest.columns else None
    if a.name_prefix is not None and label_col is None:
        raise AnalystError("invalid_argument", f"name_prefix needs a name column beside {a.by!r} in both tables")
    base, other = _entry_means(base_table, a, label_col), _entry_means(other_table, a, label_col)
    for side, e in ((sides[0], base), (sides[1], other)):
        if len(e.studies) != 1:
            raise AnalystError("unsupported", f"the {side} table {e.table.manifest.artifact_id} holds {len(e.studies)} studies "
                                              f"({', '.join(e.studies) or 'none'}); each side is ONE study: split it with filter_rows on studyDbId")
    if base.studies == other.studies:
        raise AnalystError("unsupported", f"both tables hold the same study ({base.studies[0]}); compare two different studies")
    if base.variable != other.variable:
        raise AnalystError("unsupported", f"the {sides[0]} table measures {base.variable} and the {sides[1]} {other.variable}; {needs} "
                                          "ONE variable (one trait) in both studies")
    units = [((t.manifest.measurement.unit if t.manifest.measurement else None) or "").strip() for t in (base_table, other_table)]
    if all(units) and "".join(units[0].lower().split()) != "".join(units[1].lower().split()):   # review of T5: t/ha beside kg/ha
        raise AnalystError("unsupported", f"the {sides[0]} table is in {units[0]} and the {sides[1]} in {units[1]}; {needs} ONE unit "
                                          "in both studies")
    plot_lists = _plot_lists(registry, a.roster_artifact_ids, a.by, [base.studies[0], other.studies[0]])
    names = {entry: base.labels.get(entry, set()) | other.labels.get(entry, set()) for entry in set(base.labels) | set(other.labels)}
    named: dict[str, list[str]] = {}                          # entry -> the names the person gave for it (left_out_names, 06:49 UTC)
    for name in dict.fromkeys(a.exclude):
        for entry in sorted(_named_entries([base_table, other_table], plot_lists, a.by, label_col, name)):
            named.setdefault(entry, []).append(name)
    excluded = set(named)
    candidates = sorted(set(base.means) | set(other.means))
    outside = {e for e in candidates if a.name_prefix is not None and e not in excluded and not any(n.startswith(a.name_prefix) for n in names.get(e, ()))}
    kept = [e for e in candidates if e not in excluded and e not in outside]
    paired = [e for e in kept if e in base.means and e in other.means]
    only_base = sum(1 for e in kept if e in base.means and e not in other.means)
    only_other = sum(1 for e in kept if e in other.means and e not in base.means)
    removed = sum(1 for e in excluded if e in base.means or e in other.means)              # named entries that had a mean to lose
    if len(paired) < 3:
        raise AnalystError("unsupported", f"{len(paired)} entries have a mean in both trials; {needs} at least 3 (only in "
                                          f"{sides[0]}: {only_base}, only in {sides[1]}: {only_other}, excluded: {removed}, outside the name "
                                          f"prefix: {len(outside)})")
    return _Pairing(base_table, other_table, base, other, label_col, names, excluded, outside, candidates, paired, only_base, only_other,
                    removed, [_names_label(given) for entry, given in named.items() if entry in base.means or entry in other.means],
                    [_names_label(given) for entry, given in named.items() if not (entry in base.means or entry in other.means)],
                    [t.manifest.artifact_id for t in plot_lists])


def _plot_lists(registry: ArtifactRegistry, ids: list[str], by: str, studies: list[str]) -> list[Table]:
    """The studies' plot lists (get_observation_units): the trial's entries, with or without a value of the trait (coach
    2026-10-23 07:48 UTC, (5)). One per study, of a study compared, with the entry column; anything else is refused."""
    lists: list[Table] = []
    for artifact_id in dict.fromkeys(ids):
        table = _load(registry, artifact_id)
        where = sorted(set(table.manifest.study_ids))
        if table.manifest.kind != "observation_units":
            raise AnalystError("invalid_argument", f"plot list {artifact_id} is {table.manifest.kind}, not observation_units (the plot "
                                                   "list from get_observation_units)")
        if len(where) != 1:
            raise AnalystError("invalid_argument", f"the plot list {artifact_id} is of {where}; pass one plot list per study")
        if "studyDbId" not in table.manifest.columns:
            raise AnalystError("invalid_argument", f"the plot list {artifact_id} has no column 'studyDbId'")
        strays = sorted({row.get("studyDbId", "").strip() for row in table.rows} - set(where))
        if strays:                                            # the rows, not only the manifest (review of this set, finding 5)
            raise AnalystError("invalid_argument", f"the plot list {artifact_id} is of study {where[0]} but holds rows of {strays}; pass "
                                                   "the plot list as get_observation_units saved it")
        if where[0] not in studies:
            raise AnalystError("invalid_argument", f"the plot list {artifact_id} is of study {where[0]}, not of a study compared "
                                                   f"({', '.join(studies)})")
        if any(set(t.manifest.study_ids) == set(where) for t in lists):
            raise AnalystError("invalid_argument", f"two plot lists of study {where[0]}; pass one per study")
        if by not in table.manifest.columns:
            raise AnalystError("invalid_argument", f"the plot list {artifact_id} has no column {by!r}")
        lists.append(table)
    return lists


def _knows(table: Table, by: str, label_col: str | None, name: str) -> bool:
    """Whether the table holds the name as an entry ID or as an entry's name, as _entries reads them."""
    return bool(name) and any(row.get(by) == name or (label_col is not None and row.get(label_col) == name and bool(row.get(by)))
                              for row in table.rows)


def _named_entries(tables: list[Table], plot_lists: list[Table], by: str, label_col: str | None, name: str) -> set[str]:
    """The entries one exclude name gives: from the measurement tables, as before. A name no table holds at all is looked up among
    the entries the plot records already read hold without a value (the tables' manifests; coach 09:51 UTC), then in the studies'
    plot lists (coach 07:48 UTC, (5): in C27.3 Beauregard and Tanzania have no dry-matter plot, so no table holds
    them). A name a table holds never goes on (named twice there, or the ID of another: refused as before); a name in no plot
    list stays unknown and is refused: "unknown" = absent from the trial's entries. The plot lists name only entries no table holds:
    one a table holds under another name is refused (review of this set, finding 1), and an incomplete plot list is searched as the
    part fetched, never as the whole trial (finding 2)."""
    try:
        return _entries_in_any(tables, by, label_col, [name], "exclude")
    except AnalystError as error:
        if any(_knows(t, by, label_col, name) for t in tables):
            raise
        recorded = _recorded_without_value(tables, by, name)  # the plot records already read: no new request (coach 09:51 UTC, (5))
        if recorded:
            return recorded
        if not plot_lists:
            raise
        label = LABEL_FOR.get(by)
        if not any(_knows(t, by, label, name) for t in plot_lists):
            known = sorted({row[col] for t in plot_lists for row in t.rows for col in (by, label) if col and row.get(col)})
            close = difflib.get_close_matches(name, known, n=3, cutoff=0.6)
            raise AnalystError(error.code, f"exclude {name!r} is not an entry ID or name in {', '.join(t.manifest.artifact_id for t in tables)}"
                                           f" nor in the plot lists {', '.join(t.manifest.artifact_id for t in plot_lists)}"
                                           + ("" if all(t.manifest.complete for t in plot_lists) else
                                              " (incomplete: only the plots fetched were searched)")
                                           + (f"; the closest are {close}" if close else "")) from None
        found = _entries_in_any(plot_lists, by, label, [name], "exclude")
        held = sorted(e for e in found if any(_knows(t, by, None, e) for t in tables))
        if held:
            called = sorted({row[label_col] for t in tables for row in t.rows if label_col and row.get(by) == held[0] and row.get(label_col)})
            raise AnalystError("invalid_argument", f"exclude {name!r} is entry {held[0]} in the plot lists, which the tables hold as "
                                                   + (", ".join(repr(c) for c in called) if called else "an ID without a name")
                                                   + "; refused, never guessed: use the tables' name if the person wrote it, or ask which "
                                                   "entry is meant") from None
        return found


def _recorded_without_value(tables: list[Table], by: str, name: str) -> set[str]:
    """The entry a name gives among the entries the studies' plot records hold without a value of the trait (the tables' manifests,
    entries_without_value; plot-record hosts; coach 2026-10-23 09:51 UTC, (5)): by its exact ID or its exact name. Empty when no
    table records them or the name is not among them; a name of two entries there, or an ID that is another's name, is refused.
    Every table must record them (the review of this set, finding 3): a table derived by a tool that can drop rows records none, and
    one trial's list never speaks for another trial's filtered rows."""
    if by != "germplasmDbId" or any(table.manifest.entries_without_value is None for table in tables):
        return set()
    recorded: dict[str, str] = {}
    for table in tables:
        recorded.update(table.manifest.entries_without_value)
    named = sorted(entry for entry, entry_name in recorded.items() if entry_name == name and entry != name)
    if name in recorded:
        if named:
            raise AnalystError("invalid_argument", f"exclude {name!r} is the ID of one entry and the name of another ({named}) among the "
                                                   "entries the plot records hold without a value; give an ID that is not also a name")
        return {name}
    if len(named) > 1:
        raise AnalystError("invalid_argument", f"exclude {name!r} is the name of {len(named)} entries ({named}) the plot records hold "
                                               "without a value; give the entry ID")
    return set(named)


def _named_in_trial(table: Table, by: str, label_col: str | None, name: str) -> str:
    """One exclude name of anova_trial: resolved in the table as before; a name the table does not hold at all is looked up among
    the entries the study's plot records hold without a value (coach 2026-10-23 09:51 UTC, (iv)); a name in neither is refused, as
    before."""
    try:
        return _entries(table, by, label_col, [name], "exclude")[0]
    except AnalystError:
        if _knows(table, by, label_col, name):
            raise
        recorded = _recorded_without_value([table], by, name)
        if not recorded:
            raise
        return next(iter(recorded))


def _names_label(given: list[str]) -> str:
    """One left-out entry as the person named it ("E5 / clone E5"); " | " would split the claim's parts (review finding 3)."""
    return " / ".join(name.replace(" | ", " / ") for name in given)


def _unpaired(p: _Pairing) -> tuple[list[str], list[str], list[str]]:
    """The entries behind p.only_base, p.only_other and p.removed, to name them in the exclusions (GAP 2)."""
    kept = [e for e in p.candidates if e not in p.excluded and e not in p.outside]
    return ([e for e in kept if e in p.base.means and e not in p.other.means], [e for e in kept if e in p.other.means and e not in p.base.means],
            sorted(e for e in p.excluded if e in p.base.means or e in p.other.means))


def _left_out_by_name(p: _Pairing) -> list[str]:
    """anova_trial's words for the entries the person left out, with the count (Luna C27.1, 2026-10-23: the coordinator saw
    "excluded 3" and could not verify the restriction the question asked for, so it stopped a right answer)."""
    if not p.excluded_labels and not p.named_no_mean:
        return []
    count = len(p.excluded_labels)                        # the entries that had a mean: p.removed, listed one by one
    left = (f"{', '.join(p.excluded_labels)} ({count} {'entry' if count == 1 else 'entries'})" if count else "no entry with a mean")
    # an entry the person named that has no mean in either trial is named apart, never dropped (review of this set, finding 1), in
    # the coach's words with the trait (07:48 UTC, (5)): its plots hold no valid value, or only its plot list holds it
    trait = (p.base_table.manifest.measurement.trait or p.base.variable).replace(" | ", " / ")
    fetched = "" if p.base_table.manifest.complete and p.other_table.manifest.complete else " (in the rows fetched)"   # finding 3
    return [f"left out by name: {left}" + (f"; named, no {trait} value in either trial{fetched}: {', '.join(p.named_no_mean)}"
                                           if p.named_no_mean else "")]


def _left_out_by_prefix(p: _Pairing, prefix: str | None) -> list[str]:
    """The family the person kept, as the tool applied it (coach 07:48 UTC, (4); the form of 09:51 UTC): "left out by prefix: M entries
    outside PH17.* (N kept)", M = the entries with a mean outside the prefix, N = those it keeps, both by the tool's own startswith
    test; an entry the person named is counted by name, never again here."""
    if prefix is None:
        return []
    n = len(p.outside)
    kept = sum(1 for e in p.candidates if e not in p.excluded and e not in p.outside)
    return [f"left out by prefix: {n} {'entry' if n == 1 else 'entries'} outside {prefix.replace(' | ', ' / ')}* ({kept} kept)"]


def tool_compare_trials_paired(registry: ArtifactRegistry, a: CompareTrialsPairedArgs) -> ToolResult:
    """The change of ONE trait between two studies (trials), on the entries with a mean in both (research build order T4): the
    mean of the entry means in each trial, the mean difference and the percent of the base mean (percent="change": other - base;
    "reduction": base - other), the paired t on the entries' differences (n - 1 df) with its two-sided p, Welch's t with the
    Welch-Satterthwaite df and its p (the same entries as two independent samples, ignoring the pairing), the SD and SE of the
    differences and Pearson r of the entry means between the trials. An entry's mean in a trial is the mean of its valid values
    there. Named entries can be left out (exclude) and one family kept (name_prefix). Undefined numbers carry their reason: the
    percent when the base mean is 0 or any entry mean is negative (not a ratio scale), t when every entry changed by the same
    amount, r when one trial's entry means are all equal; 0 and equal are judged against the largest plot value behind the means. Refused, never guessed: fewer than 3 shared entries; a table with other than one study; the same study, table
    or rows twice; two variables; a value of 1e200 or more; several values of one plot unless duplicate_policy='mean_per_unit'.
    Every entry and why it was or was not compared is saved as an analysis_result artifact. Descriptive numbers of recorded
    values, never a choice of entries or a judgement such as an acceptable harvest date."""
    p = _pair_two_trials(registry, a.base_artifact_id, a.other_artifact_id, a)
    base_table, other_table, base, other, label_col = p.base_table, p.other_table, p.base, p.other, p.label_col
    names, excluded, outside, candidates, paired = p.names, p.excluded, p.outside, p.candidates, p.paired
    only_base, only_other, removed = p.only_base, p.only_other, p.removed
    unpaired_base, unpaired_other, removed_entries = _unpaired(p)
    left_out_named = [*_left_out_by_name(p), *_left_out_by_prefix(p, a.name_prefix)]   # before p is reused for the paired t's p-value
    plot_lists = p.plot_lists

    g = len(paired)
    sign = -1.0 if a.percent == "reduction" else 1.0
    direction = "base - other" if a.percent == "reduction" else "other - base"
    first_side, second_side = direction.split(" - ")
    xs, ys = [base.means[e][0] for e in paired], [other.means[e][0] for e in paired]
    diffs = [sign * (y - x) for x, y in zip(xs, ys)]
    mean_base, mean_other, dbar = math.fsum(xs) / g, math.fsum(ys) / g, math.fsum(diffs) / g
    # 0 and equal are judged against the largest plot value behind the paired means, never against the means themselves: when an
    # entry's plot values cancel (0.1 + 0.2 - 0.3) its mean is residue, and round 1 stated a percent, t and r from it (round 2)
    base_peak, other_peak = max(base.peaks[e] for e in paired), max(other.peaks[e] for e in paired)
    scale = max(base_peak, other_peak)
    negative = (sum(1 for v in xs if v < -PAIR_ZERO * base_peak), sum(1 for v in ys if v < -PAIR_ZERO * other_peak))
    not_ratio = any(negative)                                 # on either side the trait then has no ratio scale (review 2, F2)
    # the zero rule (compare_means round 4, review Q1): an exact 0 that float arithmetic leaves as 2e-9 must give no percent
    if not_ratio:
        percent, percent_reason = None, (f"the entry means include negative values ({negative[0]} in the base trial, {negative[1]} in the "
                                         "other): the values are not on a ratio scale, so a percent is not stated")
    elif mean_base > PAIR_ZERO * base_peak:
        percent, percent_reason = 100 * dbar / mean_base, None
    else:
        percent, percent_reason = None, "the base trial's mean is 0 at the data's precision: a percent of it is undefined"
    sd = _scaled_sd(diffs)
    t = p = t_reason = None
    if sd <= PAIR_ZERO * scale:
        sd = 0.0
        t_reason = ("every paired entry changed by the same amount (the SD of the differences is 0 at the data's precision): the paired t "
                    "is undefined")
    else:
        t = dbar / (sd / math.sqrt(g))
        p = pvalues.t_two_sided_p(t, g - 1)
    se = sd / math.sqrt(g)
    sx, sy = _scaled_sd(xs), _scaled_sd(ys)
    welch_t = welch_df = welch_p = welch_reason = None
    if sx <= PAIR_ZERO * base_peak and sy <= PAIR_ZERO * other_peak:
        welch_reason = "the entry means are all equal within each trial (both SDs are 0 at the data's precision): Welch's t is undefined"
    else:
        vx, vy = (sx / scale) ** 2 / g, (sy / scale) ** 2 / g          # divided by the scale squared: neither overflows nor underflows
        welch_t = (dbar / scale) / math.sqrt(vx + vy)
        welch_df = (vx + vy) ** 2 / ((vx ** 2 + vy ** 2) / (g - 1))
        welch_p = pvalues.t_two_sided_p(welch_t, welch_df)
    r = r_reason = None
    for side, values, side_peak in (("base", xs, base_peak), ("other", ys, other_peak)):
        if len({_tie_key(v / side_peak) if side_peak else 0.0 for v in values}) == 1:
            r_reason = f"the {side} trial's entry means are all equal at 12 significant digits over the {g} paired entries: r is undefined"
            break
    if r_reason is None:
        top_x, top_y = max(abs(v) for v in xs), max(abs(v) for v in ys)
        r = statistics.correlation([v / top_x for v in xs], [v / top_y for v in ys])

    exclusions: list[Exclusion] = []
    for side, e in (("base", base), ("other", other)):
        exclusions += [Exclusion(reason=f"{side}: {item.reason}", count=item.count) for item in _exclusions(e.classified)]
        if e.no_study_rows:
            exclusions.append(Exclusion(reason=f"{side}: rows with an empty studyDbId", count=e.no_study_rows))
        if e.no_entry_values:                                 # review F2: a valid value without an entry ID is counted
            exclusions.append(Exclusion(reason=f"{side}: rows with an empty {a.by}", count=e.no_entry_values))
    entry_label = {entry: "; ".join(sorted(found)) for entry, found in names.items()}
    for reason, entries in (("entries only in the base trial", unpaired_base), ("entries only in the other trial", unpaired_other),
                            ("entries left out: excluded", removed_entries), ("entries left out: outside the name prefix", sorted(outside))):
        if entries:                                  # GAP 2: the entries named, not only counted
            exclusions.append(Exclusion(reason=_named_reason(reason, entries, entry_label), count=len(entries)))
    unit = base_table.manifest.measurement.unit
    trait = base_table.manifest.measurement.trait or base.variable
    tail = " | " + " | ".join([f"base: {trait}{f' ({unit})' if unit else ''} in study {base.studies[0]}; other: study {other.studies[0]}",
                               f"{g} entries with a mean in both trials",
                               f"entry mean = the mean of the entry's valid values in the trial; base: {base.transformation}; other: "
                               f"{other.transformation}", *left_out_named, PAIR_NOTE])
    variant = hashlib.sha256(json.dumps([base_table.manifest.sha256, other_table.manifest.sha256, a.by, a.value, a.percent, sorted(excluded),
                                         a.name_prefix, a.duplicate_policy, a.allow_incomplete]).encode("utf-8")).hexdigest()[:8]
    namespace = f"pair-{variant}"
    aid = base_table.manifest.artifact_id
    paired_text = f"paired t = mean difference {direction} / (SD of the differences / sqrt(n)), n - 1 df; {PAIR_PAIRED_NOTE}"
    welch_text = (f"Welch t = (mean {first_side} - mean {second_side}) / sqrt(s2 base / n + s2 other / n), Welch-Satterthwaite df; "
                  f"{PAIR_WELCH_NOTE}")
    common = {"denominator": Denominator(name="entries with a mean in both trials (pairs)", n=g), "n_independent_units": None,
              "source_artifact_hashes": [base_table.manifest.sha256, other_table.manifest.sha256], "grouping": [a.by], "exclusions": exclusions}
    # the headline numbers first: the coordinator's view keeps 20 claims (coordination_runtime VIEW_CLAIMS)
    claims = [
        Claim(claim_id=_claim_id(aid, "mean", "base", namespace=namespace), kind="mean", value=mean_base, unit=unit,
              transformation=f"mean of the base trial's entry means{tail}", **common),
        Claim(claim_id=_claim_id(aid, "mean", "other", namespace=namespace), kind="mean", value=mean_other, unit=unit,
              transformation=f"mean of the other trial's entry means{tail}", **common),
        Claim(claim_id=_claim_id(aid, "ratio", "percent", namespace=namespace), kind="ratio", value=percent, unit="%", missing_reason=percent_reason,
              transformation=f"percent {a.percent} of the mean{tail} | percent {a.percent} = 100 x (mean {first_side} - mean {second_side}) / mean base",
              **common),
        Claim(claim_id=_claim_id(aid, "mean", "difference", namespace=namespace), kind="mean", value=dbar, unit=unit,
              transformation=f"mean difference {direction}{tail} | the mean of the entries' differences {direction} (= mean {first_side} - mean "
                             f"{second_side})", **common),
        Claim(claim_id=_claim_id(aid, "test", "paired", "t", namespace=namespace), kind="test", value=t, missing_reason=t_reason,
              transformation=f"paired t{tail} | {paired_text}", **common),
        Claim(claim_id=_claim_id(aid, "test", "paired", "p", namespace=namespace), kind="test", value=p, missing_reason=t_reason,
              transformation=f"paired t: p{tail} | two-sided" + (P_UNDERFLOW if p == 0 else ""), **common),
        Claim(claim_id=_claim_id(aid, "test", "paired", "df", namespace=namespace), kind="test", value=g - 1,
              transformation=f"paired t: df{tail}", **common),
        Claim(claim_id=_claim_id(aid, "test", "welch", "t", namespace=namespace), kind="test", value=welch_t, missing_reason=welch_reason,
              transformation=f"Welch t{tail} | {welch_text}", **common),
        Claim(claim_id=_claim_id(aid, "test", "welch", "p", namespace=namespace), kind="test", value=welch_p, missing_reason=welch_reason,
              transformation=f"Welch t: p{tail} | two-sided" + (P_UNDERFLOW if welch_p == 0 else ""), **common),
        Claim(claim_id=_claim_id(aid, "test", "welch", "df", namespace=namespace), kind="test", value=welch_df, missing_reason=welch_reason,
              transformation=f"Welch t: df{tail} | Welch-Satterthwaite", **common),
        Claim(claim_id=_claim_id(aid, "correlation", "pearson", namespace=namespace), kind="correlation", value=r, missing_reason=r_reason,
              transformation=f"Pearson r of the entry means between the trials{tail}", **common),
        Claim(claim_id=_claim_id(aid, "precision", "se", namespace=namespace), kind="precision", value=se, unit=unit,
              transformation=f"SE of the mean difference{tail} | SD of the differences / sqrt(n)", **common),
        Claim(claim_id=_claim_id(aid, "sd", "differences", namespace=namespace), kind="sd", value=sd, unit=unit,
              transformation=f"SD of the differences{tail} | sample SD (n - 1) of the entries' differences {direction}", **common),
        Claim(claim_id=_claim_id(aid, "count", "pairs", namespace=namespace), kind="count", value=g,
              transformation=f"entries in both trials{tail}", **common),
    ]

    paired_set = set(paired)
    label = {entry: "; ".join(sorted(found)) for entry, found in names.items()}
    result_rows = []
    for e in sorted(set(candidates) | excluded):
        bm, om = base.means.get(e), other.means.get(e)
        why = ("excluded" if e in excluded else "outside the name prefix" if e in outside else "" if e in paired_set
               else "only in base" if bm is not None else "only in other")
        difference = percent_cell = note = ""
        if e in paired_set:
            d = sign * (om[0] - bm[0])
            difference = repr(d)
            if not_ratio:                                     # the rows follow the headline (review 2, F2)
                note = "the entry means include negative values: no percent"
            elif bm[0] > PAIR_ZERO * base.peaks[e]:
                percent_cell = repr(100 * d / bm[0])
            else:
                note = "base mean is 0: no percent"
        result_rows.append({"entry": e, "label": label.get(e, ""), "paired": "true" if e in paired_set else "false", "left_out": why,
                            "base_mean": "" if bm is None else repr(bm[0]), "base_n_values": "" if bm is None else str(bm[1]),
                            "other_mean": "" if om is None else repr(om[0]), "other_n_values": "" if om is None else str(om[1]),
                            "difference": difference, "percent": percent_cell, "percent_note": note})
    lineage = [f"lineage: compare_trials_paired of {base_table.manifest.artifact_id} sha256={base_table.manifest.sha256} (base) and "
               f"{other_table.manifest.artifact_id} sha256={other_table.manifest.sha256} (other)", f"percent rule: {a.percent} ({direction})",
               f"left out: excluded {sorted(excluded)}, name_prefix {a.name_prefix!r}",
               *([f"plot lists: {', '.join(plot_lists)}"] if plot_lists else []), f"analysis target: {ANALYSIS_TARGET}"]
    complete = bool(base_table.manifest.complete and other_table.manifest.complete)
    manifest = registry.save_table(result_rows, kind="analysis_result", columns=PAIR_RESULT_COLUMNS, complete=complete,
                                   source_request_ids=list(dict.fromkeys([*base_table.manifest.source_request_ids, *other_table.manifest.source_request_ids])),
                                   study_ids=[base.studies[0], other.studies[0]], variable_ids=[base.variable],
                                   measurement=base_table.manifest.measurement, quality_notes=lineage)
    data = {"base_artifact_id": base_table.manifest.artifact_id, "other_artifact_id": other_table.manifest.artifact_id,
            "result_artifact_id": manifest.artifact_id, "base_study": base.studies[0], "other_study": other.studies[0], "variable_id": base.variable,
            "unit": unit, "percent_rule": a.percent, "direction": direction, "n_pairs": g, "mean_base": mean_base, "mean_other": mean_other,
            "mean_difference": dbar, "percent": percent, "percent_missing_reason": percent_reason, "sd_difference": sd, "se_difference": se,
            "paired": {"t": t, "df": g - 1, "p": p, "missing_reason": t_reason},
            "welch": {"t": welch_t, "df": welch_df, "p": welch_p, "missing_reason": welch_reason},
            "pearson_r": r, "pearson_missing_reason": r_reason, "unpaired": {"only in base": only_base, "only in other": only_other},
            "left_out": {"excluded": removed, "outside the name prefix": len(outside)}, "excluded_named": sorted(excluded),
            "claims": _claims_json(claims), "note": PAIR_NOTE, "paired_text": paired_text, "welch_text": welch_text,
            "analysis_target": ANALYSIS_TARGET, "value_policy": VALUE_POLICY}
    warnings = [w for e in (base, other) for w in (e.caveat, e.classified.duplicate_note if a.duplicate_policy != "reject" else None) if w]
    if only_base or only_other or removed or outside:
        warnings.append(f"{only_base + only_other + removed + len(outside)} entries not compared (only in one trial, excluded or outside the "
                        f"name prefix); {g} paired")
    for side, e in (("base", base), ("other", other)):
        if not e.classified.units_known:
            warnings.append(f"{side}: independence unknown: some valid rows name no observation unit")
        if e.manifest_only:
            warnings.append(f"{side}: the table's manifest lists {'a study' if len(e.manifest_only) == 1 else 'studies'} with no row: "
                            f"{', '.join(e.manifest_only)}")
        warnings.extend(f"{side}: {w}" for w in _timing_warnings(e.table, a.value))
    return ToolResult(ok=True, data=data, artifact_ids=[base_table.manifest.artifact_id, other_table.manifest.artifact_id, manifest.artifact_id],
                      complete=complete, warnings=warnings)


# T5 (coach 2026-10-23 03:48 UTC, NEXT 3; research build order T5): stress indices of ONE trait between a non-stress and a stress
# study, on the entries in both. Measured first (t5_stress_indices_20261023/UNLOCKS.md): C33.4 and C34.4 ask for the highest STI.
STRESS_INDICES = ("STI", "GMP", "MP", "HM", "YI", "YSI", "TOL", "SSI", "RDI")
STRESS_FORMULAS = {
    "STI": "STI = Yp x Ys / (mean Yp)^2 (Fernandez 1992)", "GMP": "GMP = sqrt(Yp x Ys) (Fernandez 1992)",
    "MP": "MP = (Yp + Ys) / 2 (Rosielle and Hamblin 1981)", "HM": "HM = 2 x Yp x Ys / (Yp + Ys) (Bidinger et al. 1987)",
    "YI": "YI = Ys / mean Ys (Gavuzzi et al. 1997)", "YSI": "YSI = Ys / Yp (Bouslama and Schapaugh 1984)",
    "TOL": "TOL = Yp - Ys (Rosielle and Hamblin 1981)",
    "SSI": "SSI = (1 - Ys / Yp) / SI, SI = 1 - mean Ys / mean Yp (Fischer and Maurer 1978)",
    "RDI": "RDI = (Ys / Yp) / (mean Ys / mean Yp) (Fischer and Wood 1979)",
}
STRESS_IN_UNITS = ("GMP", "MP", "HM", "TOL")                # in the trait's unit; the other indices are ratios
STRESS_NOTE = "descriptive stress indices of recorded entry means, not a recommendation"
STRESS_RESULT_COLUMNS = {
    "entry": "exact entry ID", "label": "name carried from the label column, if any",
    "paired": "true when the entry has a mean in both trials",
    "left_out": "why an entry has no indices ('' when paired): only in non-stress, only in stress, excluded, outside the name prefix",
    "yp": "the entry's mean in the non-stress trial (Yp), full precision ('' = none)",
    "ys": "the entry's mean in the stress trial (Ys), full precision ('' = none)",
    **{index: f"{STRESS_FORMULAS[index]}; '' = undefined, the reason in note" for index in STRESS_INDICES},
    "rank": "rank by the requested index ('' when not ranked)", "note": "why an index is undefined for the entry ('' otherwise)",
}


def tool_stress_indices(registry: ArtifactRegistry, a: StressIndicesArgs) -> ToolResult:
    """Stress indices of ONE trait between a non-stress study (Yp) and a stress study (Ys), on the entries with a mean in both
    (research build order T5): per entry STI = Yp x Ys / (mean Yp)^2, GMP = sqrt(Yp x Ys), MP = (Yp + Ys) / 2, HM = 2 x Yp x Ys /
    (Yp + Ys), YI = Ys / mean Ys, YSI = Ys / Yp, TOL = Yp - Ys, SSI = (1 - Ys / Yp) / SI with SI = 1 - mean Ys / mean Yp, and RDI =
    (Ys / Yp) / (mean Ys / mean Yp); the population means and SI; and the entries ranked by one index (rank_by, highest or
    lowest first; top_k stated with their Yp and Ys; ties share a rank). An entry's mean in a trial is the mean of its valid
    values there; the pairing, exclude and name_prefix are compare_trials_paired's. An index that is undefined for an entry (Yp
    0, Yp + Ys 0, SI 0, a population mean 0) is left empty with the reason; 0 is judged against the largest plot value.
    Refused, never guessed: negative entry means (no ratio scale), no entry with a defined rank_by, fewer than 3 shared
    entries, a table of other than one study, the same study, table or rows twice, two variables. Every entry's indices are
    saved as an analysis_result artifact. Descriptive numbers of recorded values, never a choice of entries or a label."""
    p = _pair_two_trials(registry, a.nonstress_artifact_id, a.stress_artifact_id, a, sides=("non-stress", "stress"), needs="stress indices need")
    paired, g = p.paired, len(p.paired)
    peak_p, peak_s = max(p.base.peaks[e] for e in paired), max(p.other.peaks[e] for e in paired)
    yp, ys = {e: p.base.means[e][0] for e in paired}, {e: p.other.means[e][0] for e in paired}
    negative = (sum(1 for v in yp.values() if v < -PAIR_ZERO * peak_p), sum(1 for v in ys.values() if v < -PAIR_ZERO * peak_s))
    if any(negative):
        raise AnalystError("unsupported", f"stress indices need values on a ratio scale; the entry means include negative values "
                                          f"({negative[0]} under non-stress, {negative[1]} under stress)")
    # a mean of float residue is 0 (T4's residue rule): plot values that cancel (0.1 + 0.2 - 0.3) leave 1e-17, not a tiny Yp
    yp = {e: 0.0 if v <= PAIR_ZERO * p.base.peaks[e] else v for e, v in yp.items()}
    ys = {e: 0.0 if v <= PAIR_ZERO * p.other.peaks[e] else v for e, v in ys.items()}
    mean_p, mean_s = math.fsum(yp.values()) / g, math.fsum(ys.values()) / g
    zero_p, zero_s = mean_p <= PAIR_ZERO * peak_p, mean_s <= PAIR_ZERO * peak_s
    si = None if zero_p else 1 - mean_s / mean_p
    si_reason = "the non-stress mean is 0: SI is undefined" if zero_p else None
    si_zero = si is not None and abs(si) <= PAIR_ZERO        # the two means equal at the data's precision
    if si_zero:
        si = 0.0
    values: dict[str, dict[str, float | None]] = {}
    why: dict[str, dict[str, str]] = {}
    for e in paired:
        yp_e, ys_e = yp[e], ys[e]
        v: dict[str, float | None] = {"TOL": yp_e - ys_e, "MP": (yp_e + ys_e) / 2, "GMP": math.sqrt(yp_e) * math.sqrt(ys_e)}
        reasons: dict[str, str] = {}
        # without the product Yp x Ys, which overflows for values of 1e180 and underflows for 1e-160
        v["HM"] = None if yp_e == 0 and ys_e == 0 else 0.0 if yp_e == 0 or ys_e == 0 else 2 / (1 / yp_e + 1 / ys_e)
        v["STI"] = None if zero_p else (yp_e / mean_p) * (ys_e / mean_p)
        v["YI"] = None if zero_s else ys_e / mean_s
        v["YSI"] = None if yp_e == 0 else ys_e / yp_e
        v["SSI"] = None if yp_e == 0 or si is None or si_zero else (1 - ys_e / yp_e) / si
        v["RDI"] = None if yp_e == 0 or zero_p or zero_s else (ys_e / yp_e) / (mean_s / mean_p)
        for index, reason in (("HM", "Yp + Ys is 0"), ("STI", "mean Yp is 0"), ("YI", "mean Ys is 0"), ("YSI", "Yp is 0"),
                              ("SSI", "Yp is 0" if yp_e == 0 else "SI is 0" if si_zero else "mean Yp is 0"),
                              ("RDI", "Yp is 0" if yp_e == 0 else "mean Yp is 0" if zero_p else "mean Ys is 0")):
            if v[index] is None:
                reasons[index] = reason
        for index, value in v.items():            # review of T5: Ys / Yp of 1e199 / 1e-199 is no number (it wrote 'inf')
            if value is not None and not math.isfinite(value):
                v[index], reasons[index] = None, "beyond the float range"
        values[e], why[e] = v, reasons
    ranked = [e for e in paired if values[e][a.rank_by] is not None]
    not_ranked = [e for e in paired if values[e][a.rank_by] is None]
    if not ranked:
        raise AnalystError("unsupported", f"no entry has a defined {a.rank_by}: {why[paired[0]][a.rank_by]}")
    tie = {e: float(format(values[e][a.rank_by], ".12g")) for e in ranked}          # the project's tie rule (rank_entries)
    ranked.sort(key=lambda e: ((-tie[e] if a.order == "highest" else tie[e]), e))
    rank = {e: 1 + sum(1 for o in ranked if (tie[o] > tie[e] if a.order == "highest" else tie[o] < tie[e])) for e in ranked}
    cut = rank[ranked[min(a.top_k, len(ranked)) - 1]]
    stated = [e for e in ranked if rank[e] <= cut]

    label = {entry: "; ".join(sorted(found)) for entry, found in p.names.items()}
    exclusions: list[Exclusion] = []
    for side, e in (("non-stress", p.base), ("stress", p.other)):
        exclusions += [Exclusion(reason=f"{side}: {item.reason}", count=item.count) for item in _exclusions(e.classified)]
        if e.no_study_rows:
            exclusions.append(Exclusion(reason=f"{side}: rows with an empty studyDbId", count=e.no_study_rows))
        if e.no_entry_values:
            exclusions.append(Exclusion(reason=f"{side}: rows with an empty {a.by}", count=e.no_entry_values))
    unpaired_base, unpaired_other, removed_entries = _unpaired(p)
    for reason, entries in (("entries only in the non-stress trial", unpaired_base), ("entries only in the stress trial", unpaired_other),
                            ("entries left out: excluded", removed_entries), ("entries left out: outside the name prefix", sorted(p.outside))):
        if entries:                                  # GAP 2: the entries named, not only counted
            exclusions.append(Exclusion(reason=_named_reason(reason, entries, label), count=len(entries)))
    if not_ranked:
        exclusions.append(Exclusion(reason=_named_reason(f"entries not ranked: {a.rank_by} undefined", not_ranked, label), count=len(not_ranked)))
    unit = p.base_table.manifest.measurement.unit
    trait = p.base_table.manifest.measurement.trait or p.base.variable
    first = "highest first" if a.order == "highest" else "lowest first"
    tail = " | " + " | ".join([f"non-stress (Yp): {trait}{f' ({unit})' if unit else ''} in study {p.base.studies[0]}; stress (Ys): study "
                               f"{p.other.studies[0]}", f"{g} entries with a mean in both trials",
                               f"entry mean = the mean of the entry's valid values in the trial; non-stress: {p.base.transformation}; "
                               f"stress: {p.other.transformation}", *_left_out_by_name(p), *_left_out_by_prefix(p, a.name_prefix), STRESS_NOTE])
    variant = hashlib.sha256(json.dumps([p.base_table.manifest.sha256, p.other_table.manifest.sha256, a.by, a.value, a.rank_by, a.order,
                                         sorted(p.excluded), a.name_prefix, a.duplicate_policy, a.allow_incomplete]).encode("utf-8")).hexdigest()[:8]
    namespace = f"stress-{variant}"
    aid = p.base_table.manifest.artifact_id
    pairs = Denominator(name="entries with a mean in both trials (pairs)", n=g)
    common = {"n_independent_units": None, "source_artifact_hashes": [p.base_table.manifest.sha256, p.other_table.manifest.sha256],
              "grouping": [a.by], "exclusions": exclusions}
    index_kind = "mean" if a.rank_by in STRESS_IN_UNITS else "ratio"
    claims: list[Claim] = []
    for e in stated:                                          # the top entries first: the coordinator's view keeps 20 claims
        f = {a.by: e}
        claims += [
            Claim(claim_id=_claim_id(aid, index_kind, a.rank_by.lower(), e, namespace=namespace), kind=index_kind, value=values[e][a.rank_by],
                  unit=unit if a.rank_by in STRESS_IN_UNITS else None, denominator=pairs, filters=f,
                  transformation=f"{a.rank_by} of the entry{tail} | {STRESS_FORMULAS[a.rank_by]}", **common),
            Claim(claim_id=_claim_id(aid, "rank", a.rank_by.lower(), e, namespace=namespace), kind="rank", value=rank[e], filters=f,
                  denominator=Denominator(name=f"entries ranked by {a.rank_by}, {first}", n=len(ranked)),
                  transformation=f"rank by {a.rank_by}, {first}{tail} | ties share a rank (12 significant digits)", **common),
            Claim(claim_id=_claim_id(aid, "mean", "yp", e, namespace=namespace), kind="mean", value=yp[e], unit=unit, filters=f,
                  denominator=Denominator(name=f"valid values of entry {e} in the non-stress trial", n=p.base.means[e][1]),
                  transformation=f"the entry's non-stress mean Yp{tail}", **common),
            Claim(claim_id=_claim_id(aid, "mean", "ys", e, namespace=namespace), kind="mean", value=ys[e], unit=unit, filters=f,
                  denominator=Denominator(name=f"valid values of entry {e} in the stress trial", n=p.other.means[e][1]),
                  transformation=f"the entry's stress mean Ys{tail}", **common),
        ]
    claims += [
        Claim(claim_id=_claim_id(aid, "mean", "nonstress", namespace=namespace), kind="mean", value=mean_p, unit=unit, denominator=pairs,
              transformation=f"mean of the non-stress entry means{tail}", **common),
        Claim(claim_id=_claim_id(aid, "mean", "stress", namespace=namespace), kind="mean", value=mean_s, unit=unit, denominator=pairs,
              transformation=f"mean of the stress entry means{tail}", **common),
        Claim(claim_id=_claim_id(aid, "ratio", "si", namespace=namespace), kind="ratio", value=si, missing_reason=si_reason, denominator=pairs,
              transformation=f"stress intensity SI{tail} | SI = 1 - mean Ys / mean Yp", **common),
        Claim(claim_id=_claim_id(aid, "count", "pairs", namespace=namespace), kind="count", value=g, denominator=pairs,
              transformation=f"entries in both trials{tail}", **common),
    ]

    result_rows = []
    for e in sorted(set(p.candidates) | p.excluded):
        bm, om = p.base.means.get(e), p.other.means.get(e)
        left = ("excluded" if e in p.excluded else "outside the name prefix" if e in p.outside else "" if e in values
                else "only in non-stress" if bm is not None else "only in stress")
        result_rows.append({"entry": e, "label": label.get(e, ""), "paired": "true" if e in values else "false", "left_out": left,
                            "yp": "" if bm is None else repr(yp.get(e, bm[0])), "ys": "" if om is None else repr(ys.get(e, om[0])),
                            **{index: "" if e not in values or values[e][index] is None else repr(values[e][index]) for index in STRESS_INDICES},
                            "rank": str(rank[e]) if e in rank else "",
                            "note": "; ".join(f"{index}: {reason}" for index, reason in why.get(e, {}).items())})
    lineage = [f"lineage: stress_indices of {p.base_table.manifest.artifact_id} sha256={p.base_table.manifest.sha256} (non-stress) and "
               f"{p.other_table.manifest.artifact_id} sha256={p.other_table.manifest.sha256} (stress)", f"ranked by {a.rank_by}, {first}",
               f"left out: excluded {sorted(p.excluded)}, name_prefix {a.name_prefix!r}",
               *([f"plot lists: {', '.join(p.plot_lists)}"] if p.plot_lists else []), f"analysis target: {ANALYSIS_TARGET}"]
    complete = bool(p.base_table.manifest.complete and p.other_table.manifest.complete)
    manifest = registry.save_table(result_rows, kind="analysis_result", columns=STRESS_RESULT_COLUMNS, complete=complete,
                                   source_request_ids=list(dict.fromkeys([*p.base_table.manifest.source_request_ids,
                                                                          *p.other_table.manifest.source_request_ids])),
                                   study_ids=[p.base.studies[0], p.other.studies[0]], variable_ids=[p.base.variable],
                                   measurement=p.base_table.manifest.measurement, quality_notes=lineage)
    data = {"nonstress_artifact_id": p.base_table.manifest.artifact_id, "stress_artifact_id": p.other_table.manifest.artifact_id,
            "result_artifact_id": manifest.artifact_id, "nonstress_study": p.base.studies[0], "stress_study": p.other.studies[0],
            "variable_id": p.base.variable, "unit": unit, "n_pairs": g, "mean_yp": mean_p, "mean_ys": mean_s, "si": si,
            "si_missing_reason": si_reason, "rank_by": a.rank_by, "order": first, "formula": STRESS_FORMULAS[a.rank_by],
            "stated": [{"entry": e, "label": label.get(e, ""), "rank": rank[e], "value": values[e][a.rank_by], "yp": yp[e], "ys": ys[e]}
                       for e in stated],
            "not_ranked": [{"entry": e, "label": label.get(e, ""), "reason": f"{a.rank_by} is undefined: {why[e][a.rank_by]}"} for e in not_ranked],
            "unpaired": {"only in non-stress": p.only_base, "only in stress": p.only_other},
            "left_out": {"excluded": p.removed, "outside the name prefix": len(p.outside)}, "excluded_named": sorted(p.excluded),
            "claims": _claims_json(claims), "note": STRESS_NOTE, "formulas": STRESS_FORMULAS, "analysis_target": ANALYSIS_TARGET,
            "value_policy": VALUE_POLICY}
    warnings = [w for e in (p.base, p.other) for w in (e.caveat, e.classified.duplicate_note if a.duplicate_policy != "reject" else None) if w]
    if p.only_base or p.only_other or p.removed or p.outside:
        warnings.append(f"{p.only_base + p.only_other + p.removed + len(p.outside)} entries have no indices (only in one trial, excluded or "
                        f"outside the name prefix); {g} paired")
    if not_ranked:
        warnings.append(f"{len(not_ranked)} entries not ranked: their {a.rank_by} is undefined")
    for side, e in (("non-stress", p.base), ("stress", p.other)):
        if not e.classified.units_known:
            warnings.append(f"{side}: independence unknown: some valid rows name no observation unit")
        if e.manifest_only:
            warnings.append(f"{side}: the table's manifest lists {'a study' if len(e.manifest_only) == 1 else 'studies'} with no row: "
                            f"{', '.join(e.manifest_only)}")
        warnings.extend(f"{side}: {w}" for w in _timing_warnings(e.table, a.value))
    return ToolResult(ok=True, data=data, artifact_ids=[p.base_table.manifest.artifact_id, p.other_table.manifest.artifact_id, manifest.artifact_id],
                      complete=complete, warnings=warnings)


# T6 (research build order tool 7): population-level heterosis from group means. The groups are the person's (BrAPI carries no
# grouping); the analyst's guard checks that every prefix and name is written in their words (agents/analyst.py).
HET_GROUPS = (("offspring", "offspring"), ("parent A", "parent_a"), ("parent B", "parent_b"), ("founder", "founder"))
HET_NOTE = "descriptive population means of the groups the person named, not a choice of parents"
HET_FORMULAS = {
    "mid-parent heterosis %": "% = 100 x (mean offspring - MP) / MP",
    "mid-parent heterosis": "mean offspring - MP",
    "high-parent heterosis %": "% = 100 x (mean offspring - HP) / HP, HP = the higher parent-group mean",
    "response of the parents over the founders %": "% = 100 x (MP - mean founder) / mean founder",
}
HET_RESULT_COLUMNS = {
    "study": "the site (studyDbId)", "group": "offspring, parent A, parent B or founder ('' = in no group)", "entry": "exact entry ID",
    "label": "name carried from the label column", "mean": "the entry's mean in the study, full precision", "values": "valid values behind it",
}


def _het_numbers(means: dict[str, float], peak: float) -> dict[str, tuple[float | None, str | None]]:
    """The mid-parent, the heterosis, the high-parent heterosis and the response over the founders from group means; a percent on a
    base that is 0 at the data's precision (judged against the largest plot value, T4's rule) is undefined with the reason."""
    o, pa, pb, f = means["offspring"], means["parent A"], means.get("parent B"), means.get("founder")
    mp = pa if pb is None else (pa + pb) / 2
    zero = PAIR_ZERO * peak
    out: dict[str, tuple[float | None, str | None]] = {"mid-parent mean": (mp, None), "mid-parent heterosis": (o - mp, None),
                                                        "mid-parent heterosis %": (None, "the mid-parent mean is 0") if mp <= zero else
                                                        (100 * (o - mp) / mp, None)}
    if pb is not None:
        hp = max(pa, pb)
        out["high-parent heterosis %"] = (None, "the high-parent mean is 0") if hp <= zero else (100 * (o - hp) / hp, None)
    if f is not None:
        out["response of the parents over the founders %"] = (None, "the founder mean is 0") if f <= zero else (100 * (mp - f) / f, None)
    return out


def tool_heterosis_population(registry: ArtifactRegistry, a: HeterosisPopulationArgs) -> ToolResult:
    """Population-level mid-parent heterosis of ONE trait (research build order tool 7): per site (study) the group means - each the
    unweighted mean of its entries' means, an entry's mean the mean of its valid values there - of the offspring and one or two
    parent groups, the mid-parent MP = (mean parent A + mean parent B) / 2 (parent A's mean for one group), the heterosis
    mean offspring - MP and 100 x (mean offspring - MP) / MP, the high-parent heterosis 100 x (mean offspring - HP) / HP (HP the
    higher parent mean), and with founders the response of the parents over the founders 100 x (MP - mean founder) / mean
    founder; with several sites the same on the group means averaged over the sites (each site weighted equally), which is not
    the mean of the per-site percents. The groups are name prefixes and names the person wrote. Refused, never guessed: an
    entry in two groups, a prefix or name that matches no entry, a group without a value in a site, negative group means (no
    ratio scale), two variables or units, the same rows twice. Every grouped entry's mean is saved as an analysis_result
    artifact. Descriptive numbers of recorded values, never a choice of parents or entries."""
    tables: list[Table] = []
    seen: dict[str, str] = {}
    for aid in a.artifact_ids:
        table = _load(registry, aid)
        for key in (table.manifest.artifact_id, table.manifest.sha256):
            if key in seen:
                raise AnalystError("invalid_argument", f"{seen[key]} and {table.manifest.artifact_id} hold the same rows; pass each table once")
            seen[key] = table.manifest.artifact_id
        tables.append(table)
    if a.by.endswith("Name"):
        raise AnalystError("invalid_argument", f"use the entry ID column, not {a.by!r}; names are carried as labels")
    label_col = LABEL_FOR.get(a.by)
    variables: set[str] = set()
    for table in tables:
        _require_column(table, a.by, "entry column")
        _require_column(table, a.value, "value column")
        _require_column(table, "studyDbId", "study column")
        if label_col is None or label_col not in table.manifest.columns:
            raise AnalystError("invalid_argument", f"{table.manifest.artifact_id} has no name column beside {a.by!r}; the groups are given by name")
        variables.add(_single_variable(table))
    if len(variables) > 1:
        raise AnalystError("unsupported", f"the tables hold {len(variables)} variables {sorted(variables)}; population heterosis needs one "
                                          "variable (one trait)")
    stated = [((t.manifest.measurement.unit if t.manifest.measurement else None) or "").strip() for t in tables]
    units = list(dict.fromkeys(u for u in stated if u))
    if len({"".join(u.lower().split()) for u in units}) > 1:
        raise AnalystError("unsupported", f"the tables are in {' and '.join(units)}; population heterosis needs ONE unit in every site")
    if units and not all(stated):                             # review of T6, finding 5: t/ha beside no unit
        raise AnalystError("unsupported", f"some tables state the unit {units[0]} and some state none; population heterosis needs ONE "
                                          "unit in every site")
    specs: dict[str, tuple[list[str], list[str]]] = {}
    starred: set[str] = set()                                 # prefixes passed with their * (written as patterns)
    for label, key in HET_GROUPS:
        starred |= {p[:-1] for p in getattr(a, f"{key}_prefixes") if p.endswith("*")}
        prefixes = [p[:-1] if p.endswith("*") else p for p in getattr(a, f"{key}_prefixes")]
        names = list(getattr(a, f"{key}_names"))
        if any(not p.strip() for p in prefixes) or any(not n.strip() for n in names):
            raise AnalystError("invalid_argument", f"the {label} group holds an empty prefix or name")
        if prefixes or names:
            specs[label] = (prefixes, names)
    if "offspring" not in specs or "parent A" not in specs:
        raise AnalystError("invalid_argument", "population heterosis needs an offspring group and at least one parent group (parent_a), "
                                               "each given by the name prefixes or names the person wrote")

    gates: list[str] = []
    notes: list[str] = []
    exclusions: list[Exclusion] = []
    values: dict[tuple[str, str], list[float]] = {}           # (study, entry) -> the valid values used
    peaks: dict[tuple[str, str], float] = {}                  # (study, entry) -> the largest absolute raw value (T4's residue rule)
    names_of: dict[str, set[str]] = {}
    plot_table: dict[str, str] = {}
    studies_seen: set[str] = set()                           # every study of the tables, with or without a valid value (review finding 1)
    no_study = no_entry = 0
    for table in tables:
        studies_seen |= set(table.manifest.study_ids) | {row.get("studyDbId", "").strip() for row in table.rows} - {""}
        gate = _completeness_gate(table, a.allow_incomplete)
        if gate:
            gates.append(f"{table.manifest.artifact_id}: {gate}")
        classified = _classify(table, a.value)
        big = max((abs(number) for _, number in classified.valid), default=0.0)
        if big >= 1e200:
            raise AnalystError("unsupported", f"{table.manifest.artifact_id} holds a value of 1e200 or more in absolute value ({big!r}); "
                                              "group means of such values are not computed")
        used, transformation = _apply_duplicate_policy(table, classified, a.duplicate_policy, value=a.value)
        notes.append(transformation)
        exclusions += _exclusions(classified)
        for row in table.rows:
            entry = row.get(a.by, "").strip()
            if entry:
                names_of.setdefault(entry, set())
                if row.get(label_col, ""):
                    names_of[entry].add(row[label_col])
            plot = row.get(UNIT_COL, "").strip()
            if plot:
                if plot_table.setdefault(plot, table.manifest.artifact_id) != table.manifest.artifact_id:
                    raise AnalystError("unsupported", f"plot {plot} is in both {plot_table[plot]} and {table.manifest.artifact_id}; pass tables "
                                                      "that do not overlap")
        for row, number in used:
            study, entry = row.get("studyDbId", "").strip(), row.get(a.by, "").strip()
            if not study:
                no_study += 1
            elif not entry:
                no_entry += 1
            else:
                values.setdefault((study, entry), []).append(number)
        for row, number in classified.valid:
            study, entry = row.get("studyDbId", "").strip(), row.get(a.by, "").strip()
            if study and entry:
                peaks[(study, entry)] = max(peaks.get((study, entry), 0.0), abs(number))

    def matches(entry: str, label: str) -> bool:
        prefixes, names = specs[label]
        own = names_of.get(entry, set())
        return any(n.startswith(p) for p in prefixes for n in own) or any(x == entry or x in own for x in names)

    member: dict[str, str] = {}
    for entry in sorted(names_of):
        found = [label for label in specs if matches(entry, label)]
        if len(found) > 1:
            raise AnalystError("invalid_argument", f"entry {entry} ({'; '.join(sorted(names_of[entry])) or 'no name'}) is in both the "
                                                   f"{found[0]} group and the {found[1]} group; give groups that do not overlap")
        if found:
            member[entry] = found[0]
    for label, (prefixes, names) in specs.items():
        for prefix in prefixes:
            if not any(n.startswith(prefix) for own in names_of.values() for n in own):
                raise AnalystError("invalid_argument", f"prefix {prefix!r} of the {label} group matches no entry name in the tables")
        for name in names:
            if not any(name == entry or name in own for entry, own in names_of.items()):
                raise AnalystError("invalid_argument", f"{name!r} of the {label} group is not an entry ID or name in the tables")
    all_names = {n for own in names_of.values() for n in own}
    for label, (prefixes, _) in specs.items():                # review finding 2: "Isabel" as a prefix took Isabela too
        for prefix in prefixes:
            longer = sorted(n for n in all_names if n.startswith(prefix) and n != prefix)
            if prefix not in starred and prefix in all_names and longer:
                raise AnalystError("invalid_argument", f"prefix {prefix!r} of the {label} group is an entry's whole name and also starts "
                                                       f"{', '.join(longer[:3])}{' and others' if len(longer) > 3 else ''}; pass it as a "
                                                       "name, or the prefix the person wrote")

    sites = sorted(studies_seen | {study for study, _ in values})
    group_mean: dict[tuple[str, str], float] = {}
    group_n: dict[tuple[str, str], int] = {}
    group_peak: dict[tuple[str, str], float] = {}
    entry_mean: dict[tuple[str, str], float] = {}
    for site in sites:
        for label in specs:
            entries = sorted(e for e, g in member.items() if g == label and (site, e) in values)
            if not entries:
                raise AnalystError("unsupported", f"the {label} group has no entry with a valid value in study {site}; each group needs a mean "
                                                  "in every site")
            for e in entries:
                entry_mean[(site, e)] = math.fsum(values[(site, e)]) / len(values[(site, e)])
            group_mean[(site, label)] = math.fsum(entry_mean[(site, e)] for e in entries) / len(entries)
            group_n[(site, label)] = len(entries)
            group_peak[(site, label)] = max(peaks.get((site, e), 0.0) for e in entries)
    for key, mean in list(group_mean.items()):
        if mean < -PAIR_ZERO * group_peak[key]:
            raise AnalystError("unsupported", f"population heterosis needs values on a ratio scale; the {key[1]} group's mean in study {key[0]} "
                                              f"is negative ({mean!r})")
        if abs(mean) <= PAIR_ZERO * group_peak[key]:             # a mean of float residue is 0 (0.1 + 0.2 - 0.3), T4's rule
            group_mean[key] = 0.0
    per_site = {site: _het_numbers({label: group_mean[(site, label)] for label in specs},
                                   max(group_peak[(site, label)] for label in specs)) for site in sites}
    averages = {label: math.fsum(group_mean[(site, label)] for site in sites) / len(sites) for label in specs} if len(sites) > 1 else None
    across = _het_numbers(averages, max(group_peak.values())) if averages is not None else None

    unit = units[0] if units else None
    trait = tables[0].manifest.measurement.trait if tables[0].manifest.measurement and tables[0].manifest.measurement.trait else next(iter(variables))
    groups_text = "; ".join(f"{label}: {', '.join([*(p + '*' for p in prefixes), *names])}" for label, (prefixes, names) in specs.items())
    mp_formula = "MP = (mean parent A + mean parent B) / 2" if "parent B" in specs else "MP = mean parent A (one parent group)"
    tail = " | ".join([f"{trait}{f' ({unit})' if unit else ''} in studies {', '.join(sites)}", f"groups: {groups_text}",
                       "group mean = the unweighted mean of its entries' means; entry mean = the mean of its valid values in the study; "
                       + "; ".join(dict.fromkeys(notes)), HET_NOTE])
    in_no_group = len({e for (_, e) in values if e not in member})
    for reason, n in (("entries in no group the person named", in_no_group), ("rows with an empty studyDbId", no_study),
                      (f"rows with an empty {a.by}", no_entry)):
        if n:
            exclusions.append(Exclusion(reason=reason, count=n))
    variant = hashlib.sha256(json.dumps([sorted(t.manifest.sha256 for t in tables), a.by, a.value, sorted((k, v) for k, v in specs.items()),
                                         a.duplicate_policy, a.allow_incomplete]).encode("utf-8")).hexdigest()[:8]
    namespace = f"het-{variant}"
    aid = tables[0].manifest.artifact_id
    common = {"n_independent_units": None, "source_artifact_hashes": sorted(t.manifest.sha256 for t in tables), "grouping": ["studyDbId"],
              "exclusions": exclusions}
    slug = {"mid-parent heterosis %": "mph-pct", "mid-parent heterosis": "mph", "mid-parent mean": "mp", "high-parent heterosis %": "hph-pct",
            "response of the parents over the founders %": "response-pct"}

    def formula(head: str) -> str:
        return f"{HET_FORMULAS[head]}; {mp_formula}" if head in HET_FORMULAS else mp_formula

    def number_claims(numbers, site: str | None) -> list[Claim]:
        out = []
        on = " on the site averages" if site is None else ""
        heads = ["mid-parent heterosis %", "mid-parent mean", "mid-parent heterosis", "response of the parents over the founders %",
                 "high-parent heterosis %"] if site is not None else ["mid-parent heterosis %", "mid-parent heterosis",
                 "response of the parents over the founders %", "high-parent heterosis %"]
        for head in heads:
            if head not in numbers:
                continue
            value, reason = numbers[head]
            percent = head.endswith("%")
            if site is None:
                denominator = Denominator(name="sites averaged, each site weighted equally", n=len(sites))
            elif head.startswith("response"):
                denominator = Denominator(name=f"entries of the founder group in study {site}", n=group_n[(site, "founder")])
            elif head == "mid-parent mean":
                denominator = Denominator(name=f"parent groups in study {site}", n=2 if "parent B" in specs else 1)
            else:
                denominator = Denominator(name=f"entries of the offspring group in study {site}", n=group_n[(site, "offspring")])
            out.append(Claim(claim_id=_claim_id(aid, "ratio" if percent else "mean", slug[head], site or "averages", namespace=namespace),
                             kind="ratio" if percent else "mean", value=value, unit="%" if percent else unit, missing_reason=reason,
                             denominator=denominator, filters={} if site is None else {"studyDbId": site},
                             transformation=f"{head}{on} | {tail} | {formula(head)}", **common))
        return out

    def mean_claims(site: str | None) -> list[Claim]:
        out = []
        over = " over the sites" if site is None else ""
        for label in specs:
            value = averages[label] if site is None else group_mean[(site, label)]
            denominator = (Denominator(name="sites averaged, each site weighted equally", n=len(sites)) if site is None else
                           Denominator(name=f"entries of the {label} group in study {site}", n=group_n[(site, label)]))
            out.append(Claim(claim_id=_claim_id(aid, "mean", f"group-{label.replace(' ', '-')}", site or "averages", namespace=namespace),
                             kind="mean", value=value, unit=unit, denominator=denominator, filters={} if site is None else {"studyDbId": site},
                             transformation=f"{label} mean{over} | {tail} | group mean = the unweighted mean of its entries' means", **common))
        return out

    claims: list[Claim] = []
    if across is not None:                                    # the averages over the sites first: the coordinator's view keeps 20 claims
        claims += number_claims(across, None)
        claims += mean_claims(None)
        mp_value, _ = across["mid-parent mean"]
        claims.append(Claim(claim_id=_claim_id(aid, "mean", "mp", "averages", namespace=namespace), kind="mean", value=mp_value, unit=unit,
                            denominator=Denominator(name="sites averaged, each site weighted equally", n=len(sites)),
                            transformation=f"mid-parent mean over the sites | {tail} | {mp_formula}", **common))
    for site in sites:
        claims += number_claims(per_site[site], site)
    for site in sites:
        claims += mean_claims(site)

    result_rows = [{"study": site, "group": member.get(e, ""), "entry": e, "label": "; ".join(sorted(names_of.get(e, set()))),
                    "mean": repr(math.fsum(values[(site, e)]) / len(values[(site, e)])), "values": str(len(values[(site, e)]))}
                   for site, e in sorted(values)]
    lineage = [f"lineage: heterosis_population of {', '.join(f'{t.manifest.artifact_id} sha256={t.manifest.sha256}' for t in tables)}",
               f"groups: {groups_text}", f"analysis target: {ANALYSIS_TARGET}"]
    complete = all(t.manifest.complete for t in tables)
    manifest = registry.save_table(result_rows, kind="analysis_result", columns=HET_RESULT_COLUMNS, complete=complete,
                                   source_request_ids=list(dict.fromkeys(r for t in tables for r in t.manifest.source_request_ids)),
                                   study_ids=sites, variable_ids=sorted(variables), measurement=tables[0].manifest.measurement,
                                   quality_notes=lineage)

    def as_data(numbers):
        return {head: {"value": value, "reason": reason} for head, (value, reason) in numbers.items()}

    data = {"artifact_ids": [t.manifest.artifact_id for t in tables], "result_artifact_id": manifest.artifact_id, "sites": sites,
            "variable_id": next(iter(variables)), "unit": unit, "groups_named": {label: {"prefixes": p, "names": n} for label, (p, n) in specs.items()},
            "groups": {site: {label: {"mean": group_mean[(site, label)], "entries": group_n[(site, label)]} for label in specs} for site in sites},
            "per_site": {site: as_data(per_site[site]) for site in sites}, "averages": averages,
            "on_the_averages": as_data(across) if across is not None else None, "entries_in_no_group": in_no_group,
            "claims": _claims_json(claims), "note": HET_NOTE, "formulas": HET_FORMULAS, "analysis_target": ANALYSIS_TARGET,
            "value_policy": VALUE_POLICY}
    warnings = [*gates]
    if in_no_group:
        warnings.append(f"{in_no_group} entries are in no group the person named and are left out")
    for table in tables:
        warnings.extend(_timing_warnings(table, a.value))
    return ToolResult(ok=True, data=data, artifact_ids=[*(t.manifest.artifact_id for t in tables), manifest.artifact_id], complete=complete,
                      warnings=list(dict.fromkeys(warnings)))                 # each warning once (review finding 6)


# T7a (research build order tool 8, notes Q8): the plot-basis heritability of an unreplicated offspring population. A convention
# (V_P = V_G + V_E, Iowa State Quantitative Genetics ch. 6); Westcott (1981) could not be verified; no published fixture.
PLOT_H2_NOTE = "plot-basis heritability, a convention (V_P = V_G + V_E); no published fixture; not a choice of entries"
PLOT_H2_FORMULA = ("h2 = 1 - s2e / s2P; s2e = sum (n_i - 1) s2_i / sum (n_i - 1) over the error entries; s2P = the sample variance "
                   "(n - 1) of the offspring plots; s2g = s2P - s2e")
PLOT_H2_RESULT_COLUMNS = {
    "study": "the site (studyDbId)", "role": "offspring, error or '' (neither)", "entry": "exact entry ID",
    "label": "name carried from the label column", "entry_type": "the entry type column's value ('' when none)",
    "plots": "plots with a valid value in the study", "mean": "the entry's mean in the study", "variance": "the entry's sample variance ('' under 2 plots)",
}


def tool_plot_basis_h2(registry: ArtifactRegistry, a: PlotBasisH2Args) -> ToolResult:
    """The plot-basis heritability of an unreplicated offspring population (research build order tool 8): per site (study)
    h2 = 1 - s2e / s2P, with s2e the pooled within-entry variance sum (n_i - 1) s2_i / sum (n_i - 1) of the error entries (every
    entry with at least 2 plots outside the offspring, or with error_from="checks" only the entries whose entryType is check) and
    s2P the sample variance of the offspring's plots; s2g = s2P - s2e; h2 set to 0 (and said) when s2e is above s2P; with several
    sites also the mean of the site h2 (each site weighted equally). A convention (V_P = V_G + V_E) with no published fixture.
    The offspring are the name prefixes and names the person wrote. Refused, never guessed: fewer than 2 offspring plots or no
    error entry with 2 plots in a site, error_from="checks" without an entryType column, a prefix or name matching nothing, two
    variables or units, overlapping plots. Descriptive numbers, never a choice of entries."""
    tables: list[Table] = []
    seen: dict[str, str] = {}
    for aid in a.artifact_ids:
        table = _load(registry, aid)
        for key in (table.manifest.artifact_id, table.manifest.sha256):
            if key in seen:
                raise AnalystError("invalid_argument", f"{seen[key]} and {table.manifest.artifact_id} hold the same rows; pass each table once")
            seen[key] = table.manifest.artifact_id
        tables.append(table)
    if a.by.endswith("Name"):
        raise AnalystError("invalid_argument", f"use the entry ID column, not {a.by!r}; names are carried as labels")
    label_col = LABEL_FOR.get(a.by)
    variables: set[str] = set()
    for table in tables:
        _require_column(table, a.by, "entry column")
        _require_column(table, a.value, "value column")
        _require_column(table, "studyDbId", "study column")
        if label_col is None or label_col not in table.manifest.columns:
            raise AnalystError("invalid_argument", f"{table.manifest.artifact_id} has no name column beside {a.by!r}; the offspring are given by name")
        if a.error_from == "checks" and a.entry_type not in table.manifest.columns:
            raise AnalystError("invalid_argument", f"error_from='checks' needs the {a.entry_type} column in {table.manifest.artifact_id}: run "
                                                   "plot_design on the plot list with these measurements, then pass its table")
        variables.add(_single_variable(table))
    if len(variables) > 1:
        raise AnalystError("unsupported", f"the tables hold {len(variables)} variables {sorted(variables)}; a plot-basis heritability needs one "
                                          "variable (one trait)")
    stated = [((t.manifest.measurement.unit if t.manifest.measurement else None) or "").strip() for t in tables]
    units = list(dict.fromkeys(u for u in stated if u))
    if len({"".join(u.lower().split()) for u in units}) > 1 or (units and not all(stated)):
        raise AnalystError("unsupported", f"the tables are in {' and '.join(units) or 'no unit'} with some stating none; one unit in every site is needed"
                           if units and not all(stated) else f"the tables are in {' and '.join(units)}; one unit in every site is needed")
    prefixes = [p[:-1] if p.endswith("*") else p for p in a.offspring_prefixes]
    starred = {p[:-1] for p in a.offspring_prefixes if p.endswith("*")}
    names = list(a.offspring_names)
    if any(not p.strip() for p in prefixes) or any(not n.strip() for n in names):
        raise AnalystError("invalid_argument", "the offspring group holds an empty prefix or name")
    if not prefixes and not names:
        raise AnalystError("invalid_argument", "a plot-basis heritability needs the offspring group: the name prefixes or names the person wrote")

    gates: list[str] = []
    notes: list[str] = []
    exclusions: list[Exclusion] = []
    values: dict[tuple[str, str], list[float]] = {}
    names_of: dict[str, set[str]] = {}
    types_of: dict[tuple[str, str], set[str]] = {}             # (study, entry) -> its entry types there (review finding 1)
    plot_table: dict[str, str] = {}
    studies_seen: set[str] = set()
    no_study = no_entry = 0
    for table in tables:
        studies_seen |= set(table.manifest.study_ids) | {row.get("studyDbId", "").strip() for row in table.rows} - {""}
        gate = _completeness_gate(table, a.allow_incomplete)
        if gate:
            gates.append(f"{table.manifest.artifact_id}: {gate}")
        classified = _classify(table, a.value)
        big = max((abs(number) for _, number in classified.valid), default=0.0)
        if big >= 1e200:
            raise AnalystError("unsupported", f"{table.manifest.artifact_id} holds a value of 1e200 or more in absolute value ({big!r}); its "
                                              "variances are not computed")
        used, transformation = _apply_duplicate_policy(table, classified, a.duplicate_policy, value=a.value)
        notes.append(transformation)
        exclusions += _exclusions(classified)
        for row in table.rows:
            entry = row.get(a.by, "").strip()
            if entry:
                names_of.setdefault(entry, set())
                if row.get(label_col, ""):
                    names_of[entry].add(row[label_col])
                if row.get(a.entry_type, "").strip():
                    types_of.setdefault((row.get("studyDbId", "").strip(), entry), set()).add(row[a.entry_type].strip().casefold())
            plot = row.get(UNIT_COL, "").strip()
            if plot and plot_table.setdefault(plot, table.manifest.artifact_id) != table.manifest.artifact_id:
                raise AnalystError("unsupported", f"plot {plot} is in both {plot_table[plot]} and {table.manifest.artifact_id}; pass tables that "
                                                  "do not overlap")
        for row, number in used:
            study, entry = row.get("studyDbId", "").strip(), row.get(a.by, "").strip()
            if not study:
                no_study += 1
            elif not entry:
                no_entry += 1
            else:
                values.setdefault((study, entry), []).append(number)
    for reason, n in (("rows with an empty studyDbId", no_study), (f"rows with an empty {a.by}", no_entry)):   # review finding 2
        if n:
            exclusions.append(Exclusion(reason=reason, count=n))

    offspring = {e for e, own in names_of.items() if any(n.startswith(p) for p in prefixes for n in own) or any(x == e or x in own for x in names)}
    for prefix in prefixes:
        if not any(n.startswith(prefix) for own in names_of.values() for n in own):
            raise AnalystError("invalid_argument", f"prefix {prefix!r} of the offspring group matches no entry name in the tables")
        longer = sorted(n for own in names_of.values() for n in own if n.startswith(prefix) and n != prefix)
        if prefix not in starred and any(prefix in own for own in names_of.values()) and longer:
            raise AnalystError("invalid_argument", f"prefix {prefix!r} of the offspring group is an entry's whole name and also starts "
                                                   f"{', '.join(longer[:3])}; pass it as a name, or the prefix the person wrote")
    for name in names:
        if not any(name == e or name in own for e, own in names_of.items()):
            raise AnalystError("invalid_argument", f"{name!r} of the offspring group is not an entry ID or name in the tables")

    sites = sorted(studies_seen | {study for study, _ in values})
    per_site: dict[str, dict[str, Any]] = {}
    in_error: set[str] = set()
    for site in sites:
        if a.error_from == "checks":                          # one entry type per entry and site (review finding 1)
            mixed = sorted(e for (s, e), kinds in types_of.items() if s == site and len(kinds) > 1)
            if mixed:
                raise AnalystError("unsupported", f"entry {mixed[0]} is {' and '.join(sorted(types_of[(site, mixed[0])]))} in study {site}; a "
                                                  "check-only error needs one entry type per entry")
        off_values = [v for e in sorted(offspring) for v in values.get((site, e), [])]
        if len(off_values) < 2:
            raise AnalystError("unsupported", f"the offspring group has {len(off_values)} plot(s) with a valid value in study {site}; s2P needs at "
                                              "least 2 plots")
        error_entries = sorted(e for (s, e), vals in values.items() if s == site and e not in offspring and len(vals) >= 2
                               and (a.error_from == "replicated" or types_of.get((site, e), set()) == {"check"}))
        in_error.update(error_entries)
        if not error_entries:
            raise AnalystError("unsupported", f"study {site} has no entry with 2 plots or more for the error ({a.error_from}); s2e needs replicated "
                                              "entries")
        df = sum(len(values[(site, e)]) - 1 for e in error_entries)
        s2e = math.fsum((len(values[(site, e)]) - 1) * statistics.variance(values[(site, e)]) for e in error_entries) / df
        s2p = statistics.variance(off_values)
        zero = math.sqrt(s2p) <= PAIR_ZERO * max(abs(v) for v in off_values)
        h2 = None if zero else 1 - s2e / s2p
        flag = h2 is not None and h2 < 0
        s2g = s2p - s2e
        per_site[site] = {"h2": 0.0 if flag else h2, "h2_raw": h2, "set_to_0": flag, "s2e": s2e, "df": df, "s2P": s2p, "plots": len(off_values),
                          "s2g": max(0.0, s2g), "s2g_negative": s2g < 0, "error_entries": len(error_entries),
                          "reason": "s2P is 0: the offspring plots are all equal" if zero else None}
    undefined = [s for s in sites if per_site[s]["h2"] is None]
    mean_h2 = None if undefined or len(sites) < 2 else math.fsum(per_site[s]["h2"] for s in sites) / len(sites)

    unit = units[0].replace(" | ", " / ") if units else None                  # " | " splits a claim's parts (review finding 6)
    trait = tables[0].manifest.measurement.trait if tables[0].manifest.measurement and tables[0].manifest.measurement.trait else next(iter(variables))
    trait = trait.replace(" | ", " / ")
    groups_text = f"offspring: {', '.join([*(p + '*' for p in prefixes), *names])}; error: " + (
        "every entry with at least 2 plots outside the offspring" if a.error_from == "replicated" else "the entries whose entryType is check")
    tail = " | ".join([f"{trait}{f' ({unit})' if unit else ''} in studies {', '.join(sites)}", groups_text, PLOT_H2_FORMULA,
                       "; ".join(dict.fromkeys(notes)), PLOT_H2_NOTE])
    variant = hashlib.sha256(json.dumps([sorted(t.manifest.sha256 for t in tables), a.by, a.value, prefixes, names, a.error_from, a.entry_type,
                                         a.duplicate_policy, a.allow_incomplete]).encode("utf-8")).hexdigest()[:8]
    namespace = f"h2plot-{variant}"
    aid = tables[0].manifest.artifact_id
    in_no_role = len({e for (_, e) in values if e not in offspring and e not in in_error})     # review finding 4
    common = {"n_independent_units": None, "source_artifact_hashes": sorted(t.manifest.sha256 for t in tables), "grouping": ["studyDbId"],
              "exclusions": exclusions}
    claims: list[Claim] = []
    if len(sites) > 1:
        claims.append(Claim(claim_id=_claim_id(aid, "heritability", "h2", "mean", namespace=namespace), kind="heritability", value=mean_h2,
                            missing_reason=None if mean_h2 is not None else f"h2 is undefined in study {', '.join(undefined)}",
                            denominator=Denominator(name="sites, each weighted equally", n=len(sites)),
                            transformation=f"plot-basis heritability h2, the mean of the sites | {tail}"
                            + (" | includes a site h2 set to 0: s2e is above s2P" if any(per_site[s]["set_to_0"] for s in sites) else ""),
                            **common))
    for site in sites:
        x = per_site[site]
        f = {"studyDbId": site}
        said = " | set to 0: s2e is above s2P" if x["set_to_0"] else ""
        claims += [
            Claim(claim_id=_claim_id(aid, "heritability", "h2", site, namespace=namespace), kind="heritability", value=x["h2"],
                  missing_reason=x["reason"], denominator=Denominator(name=f"offspring plots in study {site}", n=x["plots"]), filters=f,
                  transformation=f"plot-basis heritability h2 | {tail}{said} | error pool in study {site}: {x['error_entries']} "
                                 + ("entries with at least 2 plots outside the offspring" if a.error_from == "replicated" else
                                    "check entries with at least 2 plots") + f" (df {x['df']}); s2P from {x['plots']} offspring plots",
                  **common),
            Claim(claim_id=_claim_id(aid, "variance", "s2e", site, namespace=namespace), kind="variance", value=x["s2e"],
                  denominator=Denominator(name=f"degrees of freedom of the pooled error, sum of (n_i - 1) over {x['error_entries']} entries",
                                          n=x["df"]), filters=f, transformation=f"error variance s2e | {tail}", **common),
            Claim(claim_id=_claim_id(aid, "variance", "s2p", site, namespace=namespace), kind="variance", value=x["s2P"],
                  denominator=Denominator(name=f"offspring plots in study {site}", n=x["plots"]), filters=f,
                  transformation=f"phenotypic variance s2P | {tail}", **common),
            Claim(claim_id=_claim_id(aid, "variance", "s2g", site, namespace=namespace), kind="variance", value=x["s2g"],
                  denominator=Denominator(name=f"offspring plots in study {site}", n=x["plots"]), filters=f,
                  transformation=f"genetic variance s2g | {tail}" + (" | set to 0: s2e is above s2P" if x["s2g_negative"] else ""), **common),
        ]
    result_rows = []
    for (site, e), vals in sorted(values.items()):
        role = "offspring" if e in offspring else ("error" if len(vals) >= 2 and (a.error_from == "replicated" or types_of.get((site, e), set()) == {"check"}) else "")
        result_rows.append({"study": site, "role": role, "entry": e, "label": "; ".join(sorted(names_of.get(e, set()))),
                            "entry_type": "; ".join(sorted(types_of.get((site, e), set()))), "plots": str(len(vals)), "mean": repr(math.fsum(vals) / len(vals)),
                            "variance": repr(statistics.variance(vals)) if len(vals) >= 2 else ""})
    lineage = [f"lineage: plot_basis_h2 of {', '.join(f'{t.manifest.artifact_id} sha256={t.manifest.sha256}' for t in tables)}", groups_text,
               f"analysis target: {ANALYSIS_TARGET}"]
    complete = all(t.manifest.complete for t in tables)
    manifest = registry.save_table(result_rows, kind="analysis_result", columns=PLOT_H2_RESULT_COLUMNS, complete=complete,
                                   source_request_ids=list(dict.fromkeys(r for t in tables for r in t.manifest.source_request_ids)),
                                   study_ids=sites, variable_ids=sorted(variables), measurement=tables[0].manifest.measurement, quality_notes=lineage)
    data = {"artifact_ids": [t.manifest.artifact_id for t in tables], "result_artifact_id": manifest.artifact_id, "sites": sites,
            "variable_id": next(iter(variables)), "unit": unit, "error_from": a.error_from, "per_site": per_site, "mean_h2": mean_h2,
            "entries_in_no_role": in_no_role, "claims": _claims_json(claims), "note": PLOT_H2_NOTE, "formula": PLOT_H2_FORMULA,
            "analysis_target": ANALYSIS_TARGET, "value_policy": VALUE_POLICY}
    warnings = [*gates]
    if no_study or no_entry:
        warnings.append(f"{no_study + no_entry} rows with an empty studyDbId or {a.by} are left out")
    for table in tables:
        warnings.extend(_timing_warnings(table, a.value))
    return ToolResult(ok=True, data=data, artifact_ids=[*(t.manifest.artifact_id for t in tables), manifest.artifact_id], complete=complete,
                      warnings=list(dict.fromkeys(warnings)))


COMPARE_NOTE ="descriptive comparison of recorded values, not a test and not a recommendation"
COMPARE_RESULT_COLUMNS = {"entry": "entry ID", "label": "entry name(s)", "base_mean": "entry mean in the base table ('' = none)",
                          "other_mean": "entry mean in the other table ('' = none)", "used": "true when the entry is in the comparison (in both means; with shared_entries_only=false, in its own table's mean)",
                          "reason": "why the entry is not used ('' = used)"}
_UNKNOWN_ENTRY = "is not an entry ID or name in"


def _left_out_in(sides: dict[str, dict[str, Any]], by: str, wanted: list[str]) -> set[str]:
    """Entries given by exact ID or name, found in either table (a check may be in one only); unknown in both is refused."""
    found: set[str] = set()
    for name in wanted:
        hits, first_error = set(), None
        for side in sides.values():
            try:
                hits |= set(_entries(side["table"], by, side["label_col"], [name], "exclude"))
            except AnalystError as exc:
                if _UNKNOWN_ENTRY not in exc.message:                # an ambiguous name is refused, never guessed
                    raise
                first_error = first_error or exc
        if not hits and first_error is not None:
            raise first_error
        found |= hits
    return found


def tool_compare_means(registry: ArtifactRegistry, a: CompareMeansArgs) -> ToolResult:
    """Two tables' means of ONE trait and the percent between them, for example a drought arm against its irrigated control.
    Each table's mean is the mean of its entry means (mean_of=entry_means) or of its entries' valid values (mean_of=values),
    over the entries with a mean in both tables (unless shared_entries_only is false), after leaving out the excluded entries.
    The percent is a change, (other - base) / base x 100, or a reduction, (base - other) / base x 100, the formula named in
    the claim; a base mean that is negative, or that counts as zero (under 1e-9 in absolute value, or under 1e-9 of the largest
    absolute plot value behind it, also when it averages entry means), gives no percent.
    base_study_db_id and percent are the plan's own, and the base table must be the base study's table. Both tables must
    measure the same variable in the same unit. Claims: the two means and the percent; the entry table is saved as an
    analysis_result artifact. A descriptive comparison of recorded values, not a test and not a recommendation."""
    if a.base_artifact_id == a.other_artifact_id:
        raise AnalystError("invalid_argument", "base and other are the same table; compare two different tables")
    if a.by.endswith("Name"):
        raise AnalystError("invalid_argument", f"compare by the ID column, not {a.by!r}; names are carried as labels")
    sides: dict[str, dict[str, Any]] = {}
    for side, artifact in (("base", a.base_artifact_id), ("other", a.other_artifact_id)):
        table = _load(registry, artifact)
        _require_column(table, a.by, "entry column")
        _require_column(table, a.value, "value column")
        caveat = _completeness_gate(table, a.allow_incomplete)
        classified = _classify(table, a.value)
        used, transformation = _apply_duplicate_policy(table, classified, a.duplicate_policy, value=a.value)
        label_col = LABEL_FOR.get(a.by)
        label_col = label_col if label_col in table.manifest.columns else None
        results, no_group_rows = _group_results(table, a.by, a.value, classified, used, a.min_independent_n, label_col)
        sides[side] = {"table": table, "caveat": caveat, "classified": classified, "used": used, "transformation": transformation,
                       "label_col": label_col, "results": {g["group_id"]: g for g in results}, "no_group_rows": no_group_rows,
                       "variable": _single_variable(table), "unit": table.manifest.measurement.unit,
                       "study": ", ".join(sorted(table.manifest.study_ids)) or table.manifest.artifact_id}
    base, other = sides["base"], sides["other"]
    if base["table"].manifest.sha256 == other["table"].manifest.sha256:
        raise AnalystError("invalid_argument", "base and other hold the same data; compare two different tables")
    if set(base["table"].manifest.study_ids) != {a.base_study_db_id}:     # code checks the analyst's choice of base table
        raise AnalystError("invalid_argument", f"base_artifact_id is a table of {base['study']}, not of the plan's base study "
                                               f"{a.base_study_db_id!r}; pass that one study's table as base_artifact_id")
    if a.base_study_db_id in other["table"].manifest.study_ids:
        raise AnalystError("invalid_argument", f"other_artifact_id is also a table of the base study {a.base_study_db_id!r}")
    if base["variable"] != other["variable"]:
        raise AnalystError("invalid_argument", f"the tables measure different variables ({base['variable']!r} and {other['variable']!r}); "
                                               "a percent compares one trait")
    if base["unit"] != other["unit"]:
        raise AnalystError("invalid_argument", f"the tables state different units ({base['unit']!r} and {other['unit']!r})")
    left_out = _left_out_in(sides, a.by, a.exclude)
    eligible = {side: {e for e, g in s["results"].items() if g["included"] and e not in left_out} for side, s in sides.items()}
    under = {side: {e for e, g in s["results"].items() if not g["included"] and e not in left_out} for side, s in sides.items()}
    shared = eligible["base"] & eligible["other"]
    entries = {side: (shared if a.shared_entries_only else eligible[side]) for side in sides}
    if not entries["base"] or not entries["other"]:
        raise AnalystError("incomplete_data", ("no entry has a mean in both tables" if a.shared_entries_only else "a table has no entry mean")
                           + f" after {len(left_out)} left out; nothing to compare")
    used_entries = entries["base"] & entries["other"] if a.shared_entries_only else entries["base"] | entries["other"]

    def reason(entry: str) -> str:
        if entry in left_out:
            return "left out before comparing"
        for side, s in sides.items():
            g = s["results"].get(entry)
            if g is not None and not g["included"]:
                return f"{side}: {g['exclusion_reason']}"
        return "only in the base table" if entry in eligible["base"] else "only in the other table"

    # one namespace per comparison: the same base against another table, or with other options, is another set of claims
    variant = hashlib.sha256(json.dumps([a.base_artifact_id, a.other_artifact_id, a.percent, a.mean_of, a.shared_entries_only,
                                         sorted(left_out), a.min_independent_n, a.duplicate_policy, a.allow_incomplete, a.value,
                                         a.by]).encode("utf-8")).hexdigest()[:10]
    caveats = [s["caveat"] for s in sides.values() if s["caveat"]]
    claims: list[Claim] = []
    means: dict[str, float] = {}
    # Each entry a claim did not use is counted once, with the first reason that applies: left out, then under
    # min_independent_n, then no usable mean in the other table. Rows are counted for the entries that mean used; every line
    # names its study (coach 2026-10-22 03:48 UTC, round 3 (B)).
    for side, s in sides.items():
        mine, study = entries[side], s["study"]
        across = sides["other" if side == "base" else "base"]["study"]
        dropped = len(set(s["results"]) & left_out)
        which = ("used: those with a mean in both tables" if a.shared_entries_only else "of this table") + \
                (f", after {dropped} left out" if dropped else "")
        valid_mine = sum(1 for row, _ in s["classified"].valid if row.get(a.by, "") in mine)
        values = [number for row, number in s["used"] if row.get(a.by, "") in mine]
        averaged = len(values) != valid_mine                # mean_per_unit merged duplicates among these entries' rows
        rule = (f"duplicate_policy=mean_per_unit: {valid_mine} valid rows averaged within {len(values)} observation units before "
                "pooling" if averaged else f"valid values used as recorded ({_row_unit(s['table'], a.value, [r for r, _ in s['classified'].valid if r.get(a.by, '') in mine])})")
        if a.mean_of == "entry_means":
            means[side] = statistics.fmean(s["results"][e]["mean"] for e in sorted(mine))
            n, method = len(mine), f"mean of entry means over the {len(mine)} entries {which}; entry mean of valid values"
            denominator = "entries used" if a.shared_entries_only else "entries of this table"
        else:
            means[side] = statistics.fmean(values)
            n, method = len(values), f"mean of the valid values of the {len(mine)} entries {which}"
            denominator = (f"observation units of the entries used ({PER_UNIT})" if averaged else "valid numeric values of the entries used")
        # the largest absolute RAW plot value behind this mean, also when it averages entry means: values that cancel inside one
        # entry leave an entry mean of float residue as large as the entry means themselves (round 4 review, Finding A; T4's rule).
        # The zero rule below is relative to it.
        s["scale"] = max((abs(number) for row, number in s["classified"].valid if row.get(a.by, "") in mine), default=0.0)
        # with shared entries, which entries are used depends on both tables, so either table's incompleteness marks both means
        marks = caveats if a.shared_entries_only else ([s["caveat"]] if s["caveat"] else [])
        transformation = f"{method}; {rule}" + "".join(f"; {c}" for c in marks)
        exclusions = [Exclusion(reason=why, count=count) for why, count in (
            (f"{study}: rows with an empty {a.by}", s["no_group_rows"]),
            (f"{study}: entries left out before comparing", dropped),
            (f"{study}: entries under min_independent_n={a.min_independent_n}", len(under[side])),
            (f"{study}: entries without a usable mean in {across}", len(eligible[side] - mine)),
            (f"{study}: raw-missing value token (entries used)", sum(s["results"][e]["n_raw_missing"] for e in mine)),
            (f"{study}: invalid or non-finite value token (entries used)", sum(s["results"][e]["n_invalid"] for e in mine))) if count]
        claims.append(Claim(claim_id=_claim_id(s["table"].manifest.artifact_id, "mean", side, namespace=f"cmp-{variant}"), kind="mean",
                            value=means[side], unit=s["unit"], denominator=Denominator(name=denominator, n=n),
                            source_artifact_hashes=[s["table"].manifest.sha256], filters={"studyDbId": s["study"]},
                            transformation=transformation, exclusions=exclusions))
    formula = "(other - base) / base x 100" if a.percent == "change" else "(base - other) / base x 100"
    change = means["other"] - means["base"]
    scale = base["scale"]
    # a base mean under 1e-9, or under 1e-9 of the largest absolute plot value behind it, counts as zero (round 3 (C); round 4 made
    # it relative: values near 1e8 that cancel to an exact 0 leave a float residue near 1e-8, review of round 3; round 5 takes the
    # plot values in both modes)
    zero = abs(means["base"]) < 1e-9 or abs(means["base"]) < 1e-9 * scale
    percent = None if zero or means["base"] < 0 else (change if a.percent == "change" else -change) / means["base"] * 100
    missing = (None if percent is not None else
               f"the base mean counts as zero ({means['base']!r}: under 1e-9 in absolute value, or under 1e-9 of the largest absolute plot "
               f"value behind it, {scale!r}); a percent of it is not defined" if zero
               else f"the base mean is negative ({means['base']!r}); a percent of it is not defined")
    pair = f"{base['study']} vs {other['study']}"
    not_used = (set(base["results"]) | set(other["results"])) - used_entries
    under_any = (under["base"] | under["other"]) & not_used
    percent_exclusions = [Exclusion(reason=why, count=count) for why, count in (
        (f"{pair}: entries left out before comparing", len(left_out)),
        (f"{pair}: entries under min_independent_n={a.min_independent_n}", len(under_any)),
        (f"{pair}: entries with a mean in only one of {base['study']} and {other['study']}", len(not_used - left_out - under_any)))
        if count]
    claims.append(Claim(claim_id=_claim_id(base["table"].manifest.artifact_id, "ratio", "percent", other["table"].manifest.artifact_id[4:],
                                           namespace=f"cmp-{variant}"),
                        kind="ratio", value=percent, unit="%", missing_reason=missing,
                        denominator=Denominator(name="entries used" if a.shared_entries_only else "entries of the base table",
                                                n=len(entries["base"])),
                        source_artifact_hashes=[base["table"].manifest.sha256, other["table"].manifest.sha256],
                        filters={"studyDbId": pair},
                        transformation=(f"percent {a.percent} = {formula}; base {base['study']}, other {other['study']}; each mean as "
                                        f"stated in its own claim ({a.mean_of}); left out: {', '.join(sorted(left_out)) or 'none'}; "
                                        f"{COMPARE_NOTE}" + "".join(f"; {c}" for c in caveats)),
                        exclusions=percent_exclusions))
    union = sorted(set(base["results"]) | set(other["results"]))
    labels = {e: "; ".join(sorted({s["results"][e]["label"] for s in sides.values() if e in s["results"] and s["results"][e]["label"]}))
              for e in union}
    text_rows = [{"entry": e, "label": labels[e],
                  "base_mean": "" if e not in base["results"] or base["results"][e]["mean"] is None else repr(base["results"][e]["mean"]),
                  "other_mean": "" if e not in other["results"] or other["results"][e]["mean"] is None else repr(other["results"][e]["mean"]),
                  "used": "true" if e in used_entries else "false", "reason": "" if e in used_entries else reason(e)} for e in union]
    lineage = [f"lineage: compare_means by {a.by}: base {base['table'].manifest.artifact_id} sha256={base['table'].manifest.sha256}, "
               f"other {other['table'].manifest.artifact_id} sha256={other['table'].manifest.sha256}",
               f"percent {a.percent} = {formula} ({a.mean_of}; shared_entries_only={a.shared_entries_only})",
               f"left out: {sorted(left_out) or 'none'}", f"min_independent_n={a.min_independent_n} (exclusion threshold)",
               f"analysis target: {ANALYSIS_TARGET}"]
    complete = base["table"].manifest.complete and other["table"].manifest.complete
    manifest = registry.save_table(text_rows, kind="analysis_result", columns=COMPARE_RESULT_COLUMNS, complete=complete,
                                   source_request_ids=[*base["table"].manifest.source_request_ids, *other["table"].manifest.source_request_ids],
                                   study_ids=sorted(set(base["table"].manifest.study_ids) | set(other["table"].manifest.study_ids)),
                                   variable_ids=[base["variable"]], measurement=base["table"].manifest.measurement, quality_notes=lineage)
    data = {
        "base_artifact_id": a.base_artifact_id, "other_artifact_id": a.other_artifact_id, "result_artifact_id": manifest.artifact_id,
        "variable_id": base["variable"], "unit": base["unit"], "by": a.by, "percent": a.percent, "formula": formula, "mean_of": a.mean_of,
        "base_study": base["study"], "other_study": other["study"], "base_mean": means["base"], "other_mean": means["other"],
        "percent_value": percent, "display": {"base_mean": _display(means["base"]), "other_mean": _display(means["other"]),
                                              "percent": _one_decimal(percent)},
        "entries_used": sorted(used_entries), "only_in_base": sorted(eligible["base"] - eligible["other"]),
        "only_in_other": sorted(eligible["other"] - eligible["base"]),
        "not_used": [{"entry": r["entry"], "label": r["label"], "reason": r["reason"]} for r in text_rows if r["used"] == "false"],
        "left_out": sorted(left_out), "claims": _claims_json(claims), "analysis_target": ANALYSIS_TARGET, "note": COMPARE_NOTE,
        "value_policy": VALUE_POLICY,
    }
    warnings = [*caveats]
    for side, s in sides.items():
        if a.duplicate_policy != "reject" and s["classified"].duplicate_note:
            warnings.append(f"{side}: {s['classified'].duplicate_note}")
        if not s["classified"].units_known:
            warnings.append(f"{side}: independence unknown: min_independent_n was applied to valid values, not to units")
        warnings.extend(f"{side}: {w}" for w in _timing_warnings(s["table"], a.value))
    return ToolResult(ok=True, data=data, artifact_ids=[a.base_artifact_id, a.other_artifact_id, manifest.artifact_id],
                      complete=complete, warnings=warnings)


# --------------------------------------------------------------------------
# ONE registry of the nineteen tools, schemas and dispatch
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalystToolSpec:
    name: str
    args_model: type[ToolArgs]
    fn: Callable[[ArtifactRegistry, Any], ToolResult]

    @property
    def description(self) -> str:
        return " ".join((self.fn.__doc__ or "").split())

    def input_schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()


ANALYST_TOOLS: dict[str, AnalystToolSpec] = {spec.name: spec for spec in [
    AnalystToolSpec("table_info", TableInfoArgs, tool_table_info),
    AnalystToolSpec("numeric_summary", NumericSummaryArgs, tool_numeric_summary),
    AnalystToolSpec("group_stats", GroupStatsArgs, tool_group_stats),
    AnalystToolSpec("filter_rows", FilterRowsArgs, tool_filter_rows),
    AnalystToolSpec("concat_tables", ConcatTablesArgs, tool_concat_tables),
    AnalystToolSpec("missing_report", MissingReportArgs, tool_missing_report),
    AnalystToolSpec("plot_bar", PlotBarArgs, tool_plot_bar),
    AnalystToolSpec("count_records", CountRecordsArgs, tool_count_records),
    AnalystToolSpec("rank_entries", RankEntriesArgs, tool_rank_entries),
    AnalystToolSpec("threshold_count", ThresholdCountArgs, tool_threshold_count),
    AnalystToolSpec("correlate_entry_means", CorrelateEntryMeansArgs, tool_correlate_entry_means),
    AnalystToolSpec("plot_design", PlotDesignArgs, tool_plot_design),
    AnalystToolSpec("anova_trial", AnovaTrialArgs, tool_anova_trial),
    AnalystToolSpec("gxe_partition", GxePartitionArgs, tool_gxe_partition),
    AnalystToolSpec("compare_trials_paired", CompareTrialsPairedArgs, tool_compare_trials_paired),
    AnalystToolSpec("stress_indices", StressIndicesArgs, tool_stress_indices),
    AnalystToolSpec("heterosis_population", HeterosisPopulationArgs, tool_heterosis_population),
    AnalystToolSpec("plot_basis_h2", PlotBasisH2Args, tool_plot_basis_h2),
    AnalystToolSpec("compare_means", CompareMeansArgs, tool_compare_means),
]}
ANALYST_TOOL_NAMES = list(ANALYST_TOOLS)
assert len(ANALYST_TOOL_NAMES) == 19


def analyst_tool_schemas() -> list[dict[str, Any]]:
    """[{name, description, inputSchema}] — the same models dispatch_analyst validates with."""
    return [{"name": s.name, "description": s.description, "inputSchema": s.input_schema()} for s in ANALYST_TOOLS.values()]


def _map_exception(exc: Exception) -> ToolError:
    if isinstance(exc, AnalystError):
        return ToolError(code=exc.code, message=exc.message[:500])   # type: ignore[arg-type]
    if isinstance(exc, ArtifactError):
        mapping = {"invalid_handle": "invalid_argument", "invalid_argument": "invalid_argument"}
        return ToolError(code=mapping.get(exc.code, "internal_error"), message=exc.message[:500])
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        return ToolError(code="internal_error", message=f"contract violation at {list(first.get('loc', []))}: {first.get('msg')}")
    return ToolError(code="internal_error", message=f"{type(exc).__name__}: {str(exc)[:300]}")


def dispatch_analyst(registry: ArtifactRegistry, name: str, arguments: Mapping[str, Any] | None) -> ToolResult:
    """Validate arguments against the tool's model, run the fixed function, ALWAYS return a ToolResult."""
    spec = ANALYST_TOOLS.get(name)
    if spec is None:
        return ToolResult(ok=False, error=ToolError(code="invalid_argument", message=f"unknown tool {name!r}; known: {ANALYST_TOOL_NAMES}"))
    arguments = {} if arguments is None else arguments
    if not isinstance(arguments, Mapping):
        return ToolResult(ok=False, error=ToolError(code="invalid_argument", message="arguments must be a JSON object"))
    try:
        args = spec.args_model.model_validate(dict(arguments))
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ())) or "(root)"
        return ToolResult(ok=False, error=ToolError(code="invalid_argument", message=f"{loc}: {first.get('msg')}"))
    try:
        return spec.fn(registry, args)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a coded ToolResult
        return ToolResult(ok=False, error=_map_exception(exc))


class AnalystTools:
    """The fixed tools behind the agent loop's ToolInterface (async schemas()/call()), in-process, no network."""

    def __init__(self, registry: ArtifactRegistry) -> None:
        self.registry = registry
        self.calls_made = 0

    async def schemas(self) -> list[dict[str, Any]]:
        return analyst_tool_schemas()

    async def call(self, name: str, args: Mapping[str, Any] | None) -> ToolResult:
        import anyio

        self.calls_made += 1
        return await anyio.to_thread.run_sync(dispatch_analyst, self.registry, name, dict(args or {}))


def main(argv: list[str] | None = None) -> int:   # pragma: no cover - tiny helper for humans
    """Print the tool schemas; there is no other command-line behaviour."""
    import json

    names = argv if argv else ANALYST_TOOL_NAMES
    for spec in analyst_tool_schemas():
        if spec["name"] in names:
            print(json.dumps(spec, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
