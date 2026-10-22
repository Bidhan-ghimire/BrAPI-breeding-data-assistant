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
from typing import Any, Callable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from artifacts import ArtifactError, ArtifactRegistry
from contracts import ArtifactId, ArtifactManifest, Claim, Denominator, Exclusion, ToolError, ToolResult

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


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "sd": None}
    return {"mean": statistics.fmean(values), "median": statistics.median(values), "min": min(values), "max": max(values),
            "sd": statistics.stdev(values) if len(values) >= 2 else None}


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


def _apply_duplicate_policy(table: Table, classified: Classified, policy: str) -> tuple[list[tuple[dict[str, str], float]], str]:
    """Return (rows-with-values to analyse, transformation text). Refuses duplicates unless a rule is named."""
    if not (classified.repeated_observation_ids or classified.repeated_measurements):
        return classified.valid, "valid values used as recorded (one row per observation unit)"
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


# --------------------------------------------------------------------------
# The twelve tools
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
    """Counts (total, valid, raw-missing, invalid/non-finite) and mean, median, min, max, sample SD of the valid values
    of ONE variable, as typed claims with a named denominator, n_independent_units (or unknown) and the exclusions.
    Refuses mixed variables, unresolved duplicates and incomplete artifacts (unless allow_incomplete). SD with fewer
    than 2 valid values is null. Values keep full precision; `display` is rounded."""
    table = _load(registry, a.artifact_id)
    _require_column(table, a.value, "value column")
    caveat = _completeness_gate(table, a.allow_incomplete)
    variable = _single_variable(table)
    classified = _classify(table, a.value)
    used, transformation = _apply_duplicate_policy(table, classified, a.duplicate_policy)
    if caveat:
        transformation = f"{caveat}; {transformation}"
    unit = table.manifest.measurement.unit
    averaged = len(used) != len(classified.valid)                       # duplicate_policy=mean_per_unit merged rows
    denominator_name = (f"observation units ({PER_UNIT})" if averaged else
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
    data = {
        "artifact_id": table.manifest.artifact_id, "sha256": table.manifest.sha256, "variable_id": variable, "unit": unit,
        "value_column": a.value, "counts": {"total_rows": classified.n_rows, "valid": classified.n_valid, "raw_missing": classified.n_missing,
                                            "invalid_or_nonfinite": classified.n_invalid, "values_used": len(used)},
        "n_independent_units": claims[0].n_independent_units, "stats": stats, "display": {k: _display(v) for k, v in stats.items()},
        "transformation": transformation, "exclusions": [e.model_dump() for e in _exclusions(classified)],
        "claims": _claims_json(claims), "value_policy": VALUE_POLICY, "analysis_target": ANALYSIS_TARGET,
    }
    return ToolResult(ok=True, data=data, artifact_ids=[table.manifest.artifact_id], complete=table.manifest.complete, warnings=warnings)


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
    evidence of superiority."""
    table = _load(registry, a.artifact_id)
    _require_column(table, a.by, "group column")
    _require_column(table, a.value, "value column")
    if a.by.endswith("Name"):
        raise AnalystError("invalid_argument", f"group by the ID column, not {a.by!r}; names are carried as labels")
    caveat = _completeness_gate(table, a.allow_incomplete)
    variable = _single_variable(table)
    classified = _classify(table, a.value)
    used, transformation = _apply_duplicate_policy(table, classified, a.duplicate_policy)
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
    unknown_cells = sum(not row.get(a.column, "").strip() or row.get(a.column, "").strip() == "[]"
                        for row in table.rows) if catalog_comparison else 0
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
        if catalog_comparison and (not cell.strip() or cell.strip() == "[]"):
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
               f"rows in: {len(table.rows)}, rows kept: {len(kept)}, rows without a valid number to compare: {dropped_non_numeric}"]
    membership_incomplete = bool(unknown_cells and a.column in {"programDbId", "programName"})
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
        roster = _load(registry, a.roster_artifact_id)
        if roster.manifest.kind != "observation_units":
            raise AnalystError("invalid_argument", f"roster {roster.manifest.artifact_id} is {roster.manifest.kind}, not observation_units")
        if not roster.manifest.complete:
            raise AnalystError("incomplete_data", f"roster {roster.manifest.artifact_id} is not a complete collection; expected units are unknown")
        if set(roster.manifest.study_ids) != set(table.manifest.study_ids):
            raise AnalystError("invalid_argument", f"roster studies {sorted(roster.manifest.study_ids)} differ from the table's {sorted(table.manifest.study_ids)}")
        _require_column(roster, UNIT_COL, "roster unit column")
        _require_column(table, UNIT_COL, "unit column")
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
                       "left_out": "true when the entry was left out before ranking (exclude)"}


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
    used, transformation = _apply_duplicate_policy(table, classified, a.duplicate_policy)
    if caveat:
        transformation = f"{caveat}; {transformation}"
    unit = table.manifest.measurement.unit
    results, no_group_rows = _group_results(table, a.by, a.value, classified, used, a.min_independent_n, label_col)

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
               f"transformation: {transformation}", f"order: {first} entry mean first; ties share a rank",
               f"left out before ranking: {sorted(left_out) or 'none'}", f"min_independent_n={a.min_independent_n} (exclusion threshold)",
               f"rows with an empty {a.by}: {no_group_rows} (excluded, not dropped silently)", f"analysis target: {ANALYSIS_TARGET}"]
    text_rows = [{**g, "included": "true" if g["included"] else "false", "left_out": "true" if g["group_id"] in left_out else "false",
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
    if not_ranked:
        order_exclusions.append(Exclusion(reason=f"entries not ranked: fewer than {a.min_independent_n} independent units",
                                          count=len(not_ranked)))
    # one namespace per ranking: a second call with other entries left out ranks differently, so its claims must not share IDs
    variant = hashlib.sha256(json.dumps([a.order, sorted(left_out), a.min_independent_n, a.duplicate_policy, a.allow_incomplete,
                                         a.value]).encode("utf-8")).hexdigest()[:10]
    averaged = len(used) != len(classified.valid)
    claims: list[Claim] = []
    for g in stated:
        denominator = Denominator(name=(f"observation units in entry {g['group_id']} ({PER_UNIT})" if averaged else
                                        f"valid numeric values in entry {g['group_id']}" + ("" if classified.units_known else " (rows; independence unknown)")),
                                  n=g["n_valid_values"])
        claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, "mean", a.by, g["group_id"], namespace=f"rank-{variant}"),
                            kind="mean", value=g["mean"], unit=unit, denominator=denominator, n_independent_units=g["n_independent_units"],
                            source_artifact_hashes=[table.manifest.sha256], filters={a.by: g["group_id"]}, grouping=[a.by],
                            transformation=f"entry mean of valid values; {transformation}", exclusions=exclusions))
        claims.append(Claim(claim_id=_claim_id(table.manifest.artifact_id, "rank", a.by, g["group_id"], namespace=f"rank-{variant}"),
                            kind="rank", value=g["rank"], denominator=Denominator(name=f"entries ranked by entry mean, {first} first", n=len(ranked)),
                            source_artifact_hashes=[table.manifest.sha256], filters={a.by: g["group_id"]}, grouping=[a.by],
                            transformation=(f"rank among {len(ranked)} entries by entry mean, {first} first; ties share a rank; "
                                            f"{ORDERING_NOTE}; {transformation}"), exclusions=order_exclusions))
    data = {
        "artifact_id": table.manifest.artifact_id, "result_artifact_id": manifest.artifact_id, "variable_id": variable, "unit": unit,
        "by": a.by, "label_column": label_col, "order": f"{first} first", "entries_ranked": len(ranked),
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
    used, transformation = _apply_duplicate_policy(table, classified, a.duplicate_policy)
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
               f"analysis target: {ANALYSIS_TARGET}"]
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
    used, transformation = _apply_duplicate_policy(table, classified, a.duplicate_policy)
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
    for row, number in used:
        if row.get(a.by, "") and row.get("studyDbId", ""):
            per_study.setdefault(row[a.by], {}).setdefault(row["studyDbId"], []).append(number)
    means: dict[str, tuple[float, int, int]] = {}
    not_in_every_study = 0
    dropped: set[str] = set()
    for entry in sorted(per_study):
        study_means = {study: statistics.fmean(values) for study, values in per_study[entry].items()}
        if a.entries == "in_every_study" and len(study_means) < len(studies):
            not_in_every_study += 1
            dropped.add(entry)
            continue
        means[entry] = (statistics.fmean(study_means.values()), sum(len(v) for v in per_study[entry].values()), len(study_means))
    manifest_only = sorted(set(table.manifest.study_ids) - studies)
    return _EntryMeans(table, variable, means, labels, sorted(studies), not_in_every_study, no_study_rows, transformation, caveat, classified,
                       dropped, manifest_only)


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
    scaled = {side: [v / scale if scale else 0.0 for v in values]
              for side, values, scale in (("x", xs, max(abs(v) for v in xs)), ("y", ys, max(abs(v) for v in ys)))}
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
        if e.not_in_every_study:
            exclusions.append(Exclusion(reason=f"entries not in every study of {side}", count=e.not_in_every_study))
    removed = sum(1 for e in excluded if e in x.means or e in y.means)              # named entries that had a mean to lose (review)
    for reason, n in (("entries only in x", only_x), ("entries only in y", only_y), ("entries left out: excluded", removed),
                      ("entries left out: outside the name prefix", len(outside))):
        if n:
            exclusions.append(Exclusion(reason=reason, count=n))
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
                                   measurement=out.manifest.measurement, quality_notes=lineage)
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


# --------------------------------------------------------------------------
# ONE registry of the twelve tools, schemas and dispatch
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
]}
ANALYST_TOOL_NAMES = list(ANALYST_TOOLS)
assert len(ANALYST_TOOL_NAMES) == 12


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
