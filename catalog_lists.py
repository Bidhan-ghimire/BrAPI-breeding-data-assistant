"""catalog_lists.py — "list all traits" (and locations, seasons, programs, study types): a fixed list written by code from the cached
catalog.

Run 20261015T181947: "list all traits" went to the planner. For a large catalog the planner sees only the traits whose names
share words with the question, and it has no plan for listing everything, so it asked a question back. Now code answers five
commands itself, from the same checked catalog every run loads first (the cached study and trait exports):

* "list (all) traits | locations | seasons | programs | study types", also "show me ...", "can you list ...", "give me a list of ...". An
  optional filter word keeps only the names that contain it, ignoring case: "list traits with yield". Anything longer ("list
  traits for study 4501", "list traits with yield in Clinton") is not one of these commands and goes to the planner as before.
* No planner and no other model call, no measurements, and no web request when the catalog is cached. A run is still
  recorded (a fixed plan; the catalog table it read is its evidence); it asks no accept or reject.
* The chat shows the first 30 names and the total; the run's report (Full report, answer.md) keeps the whole list. An
  unfiltered list longer than 30 says how to narrow it.
* Every list says where it came from and when that catalog was fetched.
* While a run waits for a reply, the same command is answered from that run's own catalog and the run keeps waiting
  (app._sort_waiting_message).
* At the front door the label call can start the same list run for a question in other words (triage.py, catalog_list):
  "Which seasons are in the catalog?", "How many studies are there of each study type?", "How many trait names contain the
  word yield?" (eval baseline 20261016T150241Z: they got the catalog summary, or a run that ended "Question blocked"). The
  list and the word are checked by code (triage.catalog_list_request); the list run itself is the same.

* "list studies in season 2026" (also "with season 2026", "with a 2026 season", "from 2026"): the studies whose seasons
  name that season (coach check 3, 2026-10-19: the most common demo question, answered with no run of the planner).
  The season is required; "list studies with yield" is not this command. The person's word (read in lower case, as
  every list command) is first matched to ONE recorded season name, ignoring case; then a study is listed only when
  one of its recorded seasons is exactly that text, as the count of the same question checks it (catalog_answers:
  exact JSON membership). "12026" or "2026 rainy" is not "2026"; two recorded names that differ only in capitals
  ("Fall", "FALL") are not merged: the list names both and lists neither (review, 2026-10-19).

Traits are the trait catalog's variables, one entry per variable. Locations, seasons, programs and study types come from the
study catalog: each is listed with the number of studies that name it, and only those named by at least one study are known.
Study types are listed most studies first, each name as the catalog writes it ("Specialty trial" and "Specialty Trial" stay two).
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlsplit

from contracts import AnalysisReport, Claim, Denominator, Plan, PlanStep, ResolvedScope, RetrievalReport
from retrieval_review import _cell as _escaped       # the same escaping the data review uses for database text

__all__ = ["ListEntity", "ENTITIES", "SHOWN", "WORD_MAX_CHARS", "TRAIT_HINT", "ListRequest", "CatalogTable", "ListItem",
           "CatalogList", "parse_request", "words", "is_one_word", "config_problems", "catalog_tables", "build", "chat_message", "report_lines",
           "fixed_plan", "evidence", "run_note", "setup_plan", "setup_evidence", "is_refresh_request", "setup_message"]

ListEntity = Literal["traits", "locations", "seasons", "programs", "study_types", "studies"]
ENTITIES: tuple[str, ...] = ("traits", "locations", "seasons", "programs", "study_types", "studies")
SEASON_HINTS = 5              # season names containing the word, offered when no season is exactly that word
SHOWN = 30                    # names shown in the chat (as run.ANSWER_LIST_SHOWN); the report keeps all of them
WORD_MAX_CHARS = 60           # a longer filter word is not one of these commands
LABEL_MAX_CHARS = 200         # the most of one database name shown in the chat
TRAIT_HINT = "To see the trait names, ask **list all traits**, or add a word: **list traits with yield**."

_NOUNS = {"traits": ("trait", "traits"), "locations": ("location", "locations"), "seasons": ("season", "seasons"),
          "programs": ("program", "programs"), "study_types": ("study type", "study types"), "studies": ("study", "studies")}
_EXPORT = {"traits": "variables", "locations": "studies", "seasons": "studies", "programs": "studies",
           "study_types": "studies", "studies": "studies"}                                 # the table each is read from
_CATALOG_WORD = {"variables": "trait catalog", "studies": "study catalog"}
_ORDER = {"traits": "A to Z", "locations": "A to Z", "seasons": "newest first", "programs": "A to Z",
          "study_types": "most studies first", "studies": "A to Z"}
_ALIASES = {"trait": "traits", "traits": "traits", "variable": "traits", "variables": "traits", "location": "locations",
            "locations": "locations", "season": "seasons", "seasons": "seasons", "program": "programs", "programs": "programs",
            "programme": "programs", "programmes": "programs", "study type": "study_types", "study types": "study_types"}

# A word: letters and digits, with - _ . ' / : allowed inside ("dry-matter", "CO_331:0000217"). Quotes and a closing "?" or "."
# are not part of it. The whole message must be one command, nothing more.
_TOKEN = re.compile(r"[^\W_]+(?:[-_.'/:][^\W_]+)*")
_LEAD = (r"(?:(?:please|can you|could you|would you)(?: please)? )?"
         r"(?:(?:list|show|give)(?: me)?(?: (?:a|the) list of)?|(?:a|the) list of)(?: all| every)?(?: of)?(?: the)?(?: available)? ")
_REQUEST = re.compile(
    _LEAD +
    r"(?P<entity>traits?|variables?|locations?|seasons?|programs?|programmes?|study types?)(?: names?)?"
    r"(?: (?:in|from) (?:the )?catalogs?)?"
    r"(?: (?:with|containing|that contains?|matching) (?P<word>\S+))?"
    r"(?: please)?")
# "list studies in season 2026": the season is named as a season ("season 2026", "a 2026 season"), or is a year ("from
# 2026"). "list studies with yield" is not this command: a word that is neither goes to the planner as before.
_NOT_A_SEASON = (r"(?!(?:a|an|the|please|seasons?|studies|all|any|every|no|this|that|these|those|each|some|last|one|"
                 r"their|its|my|your|which|same|current|next|other|names?|data)(?: |$))")   # "list studies with a season"
_STUDIES = re.compile(
    _LEAD + r"studies(?: (?:in|from) (?:the )?catalogs?)? "
    r"(?:(?:with|in|from|for|of|that have|having|that had)(?: (?:a|an|the))? season " + _NOT_A_SEASON + r"(?P<named>\S+)"
    r"|(?:with|in|from|for|of|that have|having|that had)(?: (?:a|an|the))? " + _NOT_A_SEASON + r"(?P<before>\S+) season"
    r"|(?:in|from|for|of) (?P<year>1[89]\d\d|2\d\d\d))"
    r"(?: please)?")
_YEAR = re.compile(r"(?<!\d)(1[89]\d\d|2\d\d\d)(?!\d)")
# The front door's "studies" list: besides the season, the person's message may have only these words (review,
# 2026-10-19). "not", "Clinton", "or 2025", "yield" are not among them: such a question is not this list, it goes on as
# a data question ("Which studies do not have a 2026 season?" must never get the 2026 list). No word that points back
# to earlier messages (it, that, these, those, there): "Which of these studies have a 2026 season?" is a follow-up, not
# the whole list (skeptic, 2026-10-19).
STUDY_LIST_WORDS = frozenset(
    "a all an any are can catalog could did do does every for from give had has have having held i in is list me "
    "of please ran run saved season seasons see show studies study tell the to trials was were "
    "what which with would you".split())
# "refresh catalogs" (Item 5): download the saved study and trait catalogs again, with the catalog setup's approval.
_REFRESH = re.compile(r"(?:please )?(?:refresh|update|reload|re download|redownload)(?: the)?(?: saved)?(?: study and trait)? "
                      r"catalogs?(?: again)?(?: now)?(?: please)?")


@dataclass(frozen=True)
class ListRequest:
    entity: str                     # one of ENTITIES
    word: str | None = None         # keep only names containing it (ignoring case)


@dataclass(frozen=True)
class CatalogTable:
    """One cached catalog export behind a run's metadata: which table, and when the server sent it."""

    export: Literal["studies", "variables"]
    artifact_id: str
    sha256: str
    rows: int
    base_url: str
    fetched_at: datetime | None      # the earliest server reply in this export (a cached reply keeps its first fetch time)
    fetched_in_this_run: bool        # a live request in this run (the catalog setup), not a saved copy
    operator_approved: bool = False  # that catalog setup was approved by the operator's standing approval, not the visitor


@dataclass(frozen=True)
class ListItem:
    label: str                       # as shown: the name, with the ID added when two entries share a name
    key: str                         # the ID, or the name when the catalog gives no ID
    studies: int | None = None       # studies that name it (locations, seasons, programs); None for traits
    name: str = ""                   # the name alone, which a filter word is matched against ("" when there is none)


@dataclass(frozen=True)
class CatalogList:
    request: ListRequest
    items: tuple[ListItem, ...]      # the names that match, in the order shown
    total: int                       # names in the whole list, before the filter
    source: CatalogTable
    unrecorded: int = 0              # studies that name no location / season / program
    hints: tuple[str, ...] = ()      # studies list with no match: season names that contain the word (at most SEASON_HINTS)
    season: str = ""                 # studies list: the recorded season name the word was matched to ("" when none or several)


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------

def words(text: str) -> list[str]:
    """A message's words as the list commands read them: lower case; letters and digits, with - _ . ' / : inside."""
    return _TOKEN.findall((text or "").lower().replace("’", "'"))


def _normal(text: str) -> str:
    return " ".join(words(text))


def parse_request(text: str) -> ListRequest | None:
    """The whole message is one of the five list commands (with an optional filter word), or None."""
    normal = _normal(text or "")
    studies = _STUDIES.fullmatch(normal)
    if studies is not None:
        season = studies.group("named") or studies.group("before") or studies.group("year")
        return ListRequest("studies", season) if len(season) <= WORD_MAX_CHARS else None
    match = _REQUEST.fullmatch(normal)
    if match is None:
        return None
    word = match.group("word")
    if word is not None and len(word) > WORD_MAX_CHARS:
        return None
    return ListRequest(_ALIASES[match.group("entity")], word)


def is_one_word(word: str) -> bool:
    """One filter word as the typed command reads it: letters and digits, with - _ . ' / : allowed inside, at most
    WORD_MAX_CHARS long ("yield", "dry-matter"; not "yield?" or "root yield"). The front door's catalog_list label is
    checked with it (triage.catalog_list_request)."""
    return len(word) <= WORD_MAX_CHARS and _TOKEN.fullmatch(word) is not None


def config_problems(entity: str | None, word: str | None) -> list[str]:
    """RunConfig.validate: a list run names one of the five lists, and at most one filter word."""
    if entity is None:
        return ["catalog_list_word needs catalog_list"] if word is not None else []
    problems = [] if entity in ENTITIES else [f"catalog_list must be one of {', '.join(ENTITIES)}"]
    if word is not None and (not word or len(word) > WORD_MAX_CHARS or any(c.isspace() for c in word)):
        problems.append(f"catalog_list_word must be one word of at most {WORD_MAX_CHARS} characters")
    if entity == "studies" and word is None:
        problems.append("catalog_list studies needs catalog_list_word: the season")
    return problems


# --------------------------------------------------------------------------
# The list, built from the run's validated catalog (agents.coordinator.KnownMetadata)
# --------------------------------------------------------------------------

def catalog_tables(ctx: Any, *, studies: str, variables: str, operator_approved: bool = False) -> dict[str, CatalogTable]:
    """The two export tables a run's metadata was built from, with the fetch time of their own replies (Controller), and
    whether this run's catalog setup was approved by the operator's standing approval."""
    records = list(ctx.client.provenance())
    tables: dict[str, CatalogTable] = {}
    for export, artifact_id in (("studies", studies), ("variables", variables)):
        manifest = ctx.registry.manifest(artifact_id)
        own = set(manifest.source_request_ids)
        replies = [r for r in records if r.request_id in own]
        fetched = sorted(r.fetched_at_utc for r in replies if r.fetched_at_utc is not None)
        tables[export] = CatalogTable(export=export, artifact_id=artifact_id, sha256=manifest.sha256, rows=manifest.row_count,
                                      base_url=ctx.client.settings.base_url, fetched_at=fetched[0] if fetched else None,
                                      fetched_in_this_run=any(r.origin == "live" for r in replies),
                                      operator_approved=operator_approved)
    return tables


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _distinct_labels(items: list[ListItem]) -> list[ListItem]:
    """Two entries with the same name stay two entries; their IDs tell them apart."""
    seen: dict[str, int] = {}
    for item in items:
        seen[item.label.casefold()] = seen.get(item.label.casefold(), 0) + 1
    return [ListItem(f"{item.label} · {item.key}", item.key, item.studies, item.name)
            if seen[item.label.casefold()] > 1 and item.key != item.label else item for item in items]


def _traits(metadata: Any) -> tuple[list[ListItem], int]:
    items = []
    for v in metadata.variables:
        name = _clean(v.name) or _clean(v.trait)
        items.append(ListItem(name or f"(no name) · {v.variable_id}", v.variable_id, None, name))
    return _distinct_labels(items), 0


def _grouped(metadata: Any, field: str) -> tuple[list[ListItem], int]:
    """Locations or programs, from the studies that name them: one entry per ID (or per name when there is no ID)."""
    groups: dict[str, dict[str, Any]] = {}
    unrecorded = 0
    for study in metadata.studies:
        identifier, name = getattr(study, f"{field}_id"), _clean(getattr(study, f"{field}_name"))
        key = identifier or (f"name:{name}" if name else None)
        if key is None:
            unrecorded += 1
            continue
        group = groups.setdefault(key, {"id": identifier or name, "names": set(), "studies": set()})
        group["studies"].add(study.study_id)
        if name:
            group["names"].add(name)
    items = []
    for g in groups.values():
        name = " / ".join(sorted(g["names"]))
        items.append(ListItem(name or f"(no name) · {g['id']}", g["id"], len(g["studies"]), name))
    return _distinct_labels(items), unrecorded


def _seasons(metadata: Any) -> tuple[list[ListItem], int]:
    studies: dict[str, set[str]] = {}
    unrecorded = 0
    for study in metadata.studies:
        names = [_clean(s) for s in study.seasons if _clean(s)]
        if not names:
            unrecorded += 1
        for name in names:
            studies.setdefault(name, set()).add(study.study_id)
    return [ListItem(name, name, len(ids), name) for name, ids in studies.items()], unrecorded


def _study_types(metadata: Any) -> tuple[list[ListItem], int]:
    """Study types, from the studies that name one: one entry per name as the catalog writes it, with its number of studies."""
    studies: dict[str, set[str]] = {}
    unrecorded = 0
    for study in metadata.studies:
        name = _clean(study.study_type)
        if not name:
            unrecorded += 1
            continue
        studies.setdefault(name, set()).add(study.study_id)
    return [ListItem(name, name, len(ids), name) for name, ids in studies.items()], unrecorded


def season_names(metadata: Any, season: str) -> list[str]:
    """The recorded season names that are the person's word, ignoring case only ("fall" -> "Fall"): exactly one, or
    several written with different capitals, or none. Recorded text is compared as it is, as the count path does."""
    wanted = (season or "").casefold()
    return sorted({s for study in metadata.studies for s in study.seasons if wanted and s.casefold() == wanted})


def _studies(metadata: Any, season: str) -> tuple[list[ListItem], int]:
    """The studies one of whose recorded seasons is exactly this recorded season name; how many studies name no season."""
    items, unrecorded = [], 0
    for study in metadata.studies:
        if not any(_clean(s) for s in study.seasons):
            unrecorded += 1
        if season in study.seasons:
            name = _clean(study.name)
            items.append(ListItem(name or f"(no name) · {study.study_id}", study.study_id, None, name))
    return _distinct_labels(items), unrecorded


def _season_hints(metadata: Any, season: str) -> tuple[str, ...]:
    """Season names that contain the word, newest first: "2026" finds "2026A" when no season is exactly "2026"."""
    word = _clean(season).casefold()
    named = [item for item in _order("seasons", _seasons(metadata)[0]) if word and word in item.name.casefold()]
    return tuple(item.name for item in named[:SEASON_HINTS])


def _order(entity: str, items: list[ListItem]) -> list[ListItem]:
    if entity == "study_types":                                  # most studies first; the same number, A to Z
        return sorted(items, key=lambda item: (-(item.studies or 0), item.label.casefold(), item.label))
    if entity == "seasons":                                      # newest first; a season without a year goes last, A to Z
        def key(item: ListItem):
            years = [int(y) for y in _YEAR.findall(item.label)]
            return (0, -max(years), item.label.casefold()) if years else (1, 0, item.label.casefold())
        return sorted(items, key=key)
    return sorted(items, key=lambda item: (item.label.casefold(), item.key))


def build(request: ListRequest, metadata: Any, tables: dict[str, CatalogTable]) -> CatalogList:
    """The whole list from the catalog, then only the names containing the word (ignoring case; the name alone, not an ID)."""
    if request.entity == "studies":               # the word is the season: one recorded name, matched exactly
        names = season_names(metadata, request.word or "")
        matching, unrecorded = _studies(metadata, names[0]) if len(names) == 1 else ([], _studies(metadata, "")[1])
        return CatalogList(request=request, items=tuple(_order("studies", matching)), total=len(metadata.studies),
                           source=tables["studies"], unrecorded=unrecorded, season=names[0] if len(names) == 1 else "",
                           hints=tuple(names) if len(names) > 1 else
                           _season_hints(metadata, request.word or "") if not matching else ())
    make = {"traits": _traits, "seasons": _seasons, "locations": lambda m: _grouped(m, "location"),
            "programs": lambda m: _grouped(m, "program"), "study_types": _study_types}[request.entity]
    every, unrecorded = make(metadata)
    every = _order(request.entity, every)
    word = (request.word or "").casefold()
    matching = [item for item in every if word in item.name.casefold()] if word else every
    return CatalogList(request=request, items=tuple(matching), total=len(every), source=tables[_EXPORT[request.entity]],
                       unrecorded=unrecorded)


# --------------------------------------------------------------------------
# What the person sees
# --------------------------------------------------------------------------

def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _host(url: str) -> str:
    return urlsplit(url).hostname or url


def _when(table: CatalogTable) -> str:
    if table.fetched_at is None:
        return "at a time that was not recorded"
    return table.fetched_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _heading(listing: CatalogList, *, escape) -> str:
    one, many = _NOUNS[listing.request.entity]
    word, matches, total = listing.request.word, len(listing.items), listing.total
    if listing.request.entity == "studies":
        head = (f"**Studies with season “{escape(listing.season or word or '')}”: {'none' if not matches else matches} of the "
                f"{_count(total, 'study', 'studies')} in the catalog.**")
        if not matches:
            return head
    elif word:                                   # "names": a filter word is not a data question ("locations with yield")
        if not matches:
            return (f"**{one.capitalize()} names containing “{escape(word)}”: none of {total}.** Only the names are searched; "
                    f"try another word, or ask **list all {many}**.")
        head = f"**{one.capitalize()} names containing “{escape(word)}”: {matches} of {total}.**"
    elif not total:
        return f"**{many.capitalize()} in the catalog: none.**"
    else:
        head = f"**{many.capitalize()} in the catalog: {total}.**"
    order = _ORDER[listing.request.entity]
    return head + (f" The first {SHOWN}, {order}:" if matches > SHOWN else f" {order[0].upper()}{order[1:]}:")


def _item_line(item: ListItem, label: str) -> str:
    return f"- {label}" + (f" — {_count(item.studies, 'study', 'studies')}" if item.studies is not None else "")


def _example(listing: CatalogList, every: tuple[ListItem, ...]) -> str | None:
    """A filter word that keeps some names but not all: 'yield' for traits when it does, else a word from the names of the
    biggest entries. None when no word narrows the list."""
    names = [item.name.casefold() for item in every]

    def narrows(word: str) -> bool:
        return 0 < sum(word in name for name in names) < len(names)

    if listing.request.entity == "traits" and narrows("yield"):
        return "yield"
    for item in sorted(every, key=lambda item: -(item.studies or 0)):
        for token in _TOKEN.findall(item.name.casefold()):
            if len(token) >= 3 and narrows(token):
                return token
    return None


def _study_notes(listing: CatalogList) -> list[str]:
    """Locations, seasons, programs and study types come from the study records: what that means, in plain sentences."""
    one, many = _NOUNS[listing.request.entity]
    if listing.request.entity == "studies":
        season = _escaped(listing.season or listing.request.word or "", LABEL_MAX_CHARS)
        notes = [f"A study is listed when one of its recorded seasons is exactly “{season}”."]
        if listing.hints and not listing.season and len(listing.hints) > 1 and all(
                h.casefold() == (listing.request.word or "").casefold() for h in listing.hints):
            notes.append("More than one recorded season is written this way: "
                         + ", ".join(_escaped(h, LABEL_MAX_CHARS) for h in listing.hints)
                         + ". This list cannot tell them apart; ask about one of them as a data question.")
        elif listing.hints:
            notes.append("Season names that contain “" + season + "”: "
                         + ", ".join(_escaped(h, LABEL_MAX_CHARS) for h in listing.hints) + ".")
        elif not listing.items:
            notes.append("No season name contains it; ask **list all seasons** to see them.")
        if listing.unrecorded:
            notes.append(f"{_count(listing.unrecorded, 'study names', 'studies name')} no season.")
        return notes
    notes = [f"Only {many} named by at least one study are listed."] if listing.total else []
    if listing.unrecorded:
        notes.append(f"{_count(listing.unrecorded, 'study names', 'studies name')} no {one}.")
    if listing.request.entity == "programs" and not listing.total:      # SweetPotatoBase: the standard fields are empty
        notes.append("Some servers keep the program only in each study's additional information (additionalInfo); "
                     "this app does not use that field for programs.")
    return notes


def _source_line(listing: CatalogList) -> str:
    table = listing.source
    how = ("a saved copy; nothing new was requested" if not table.fetched_in_this_run else
           "fetched after the app's operator approved the catalog setup (standing approval)" if table.operator_approved else
           "fetched after you approved the catalog setup")
    return (f"Source: {_host(table.base_url)}, {_CATALOG_WORD[table.export]} fetched {_when(table)} ({how}). "
            "No measurements were used.")


def chat_message(listing: CatalogList, *, in_report: bool) -> str:
    """The chat's answer: the first 30 names, the total, how to narrow a long list, and the source with its fetch time.
    in_report: the run's Full report has the whole list (a list run), not a list shown while another run waits."""
    many = _NOUNS[listing.request.entity][1]
    escape = lambda text: _escaped(text, LABEL_MAX_CHARS)          # database names and the person's word, as text
    lines = [_heading(listing, escape=escape)]
    if listing.items:
        lines.append("")
        lines += [_item_line(item, escape(item.label)) for item in listing.items[:SHOWN]]
        if len(listing.items) > SHOWN:
            lines.append(f"- … and {len(listing.items) - SHOWN} more" + (" in the Full report below." if in_report else "."))
    if listing.request.entity != "traits":
        lines += ["", " ".join(_study_notes(listing))]
    if not listing.request.word and listing.total > SHOWN:          # a long list says how to narrow it
        example = _example(listing, listing.items)
        if example:
            narrow = (f"To see fewer, add a word: **list {many} with {escape(example)}** shows only the names that contain "
                      f"“{escape(example)}”.")
        else:
            narrow = f"To see fewer, add a word: **list {many} with** and a word from the names."
        lines += ["", narrow]
    lines += ["", _source_line(listing)]
    return "\n".join(lines)


def report_lines(listing: CatalogList) -> list[str]:
    """answer.md (the Full report): every name with its ID, and the exact table it was read from. Plain text."""
    one, many = _NOUNS[listing.request.entity]
    table, word = listing.source, listing.request.word
    head = (f'Studies with season "{listing.season or word}": {len(listing.items)} of {listing.total}' if listing.request.entity == "studies"
            else f'{one.capitalize()} names containing "{word}": {len(listing.items)} of {listing.total}' if word
            else f"{many.capitalize()} in the catalog: {listing.total}")
    lines = ["## Catalog list (written by code from the cached catalog)", "",
             f"{head}, {_ORDER[listing.request.entity]}.",
             f"Source: {table.base_url}, {_CATALOG_WORD[table.export]} (the {table.export} table {table.artifact_id}, {table.rows} rows, "
             f"sha256 {table.sha256}), fetched {_when(table)}; "
             + (("fetched after the catalog setup was approved in this run"
                 + (" by the operator's standing approval." if table.operator_approved else "."))
                if table.fetched_in_this_run else "a saved copy, nothing new was requested.")
             + " No planner, no model call, no measurements."]
    if listing.request.entity == "studies":
        lines.append(" ".join(_study_notes(listing)).replace("**", ""))
    elif listing.request.entity != "traits":
        lines.append(" ".join([f"{many.capitalize()} come from the study records.", *_study_notes(listing)]))
    lines.append("")
    for item in listing.items:
        label = item.label if item.key in (item.label, "") or item.label.endswith(f" · {item.key}") else f"{item.label} · {item.key}"
        lines.append(_item_line(item, label))
    if not listing.items:
        lines.append("(no matching studies)" if listing.request.entity == "studies" else "(no matching names)")
    return lines


# --------------------------------------------------------------------------
# The run's record: a fixed plan, and the catalog table as its evidence
# --------------------------------------------------------------------------

def fixed_plan(request: ListRequest, question: str, base_url: str, now: datetime) -> Plan:
    """The plan for a list run, written by code (never by the planner): read the cached export, list its names."""
    many = _NOUNS[request.entity][1]
    export = _EXPORT[request.entity]
    studies = request.entity == "studies"
    containing = (f" whose seasons include '{request.word}' exactly (ignoring case)" if studies else
                  f" that contain '{request.word}'" if request.word else "")
    steps = [PlanStep(step_id="step_catalog", agent="retriever", action="export_metadata", inputs={"entity": export}, max_tool_calls=1),
             PlanStep(step_id="step_list", agent="analyst", action="list_catalog_names",
                      inputs={"entity": request.entity, **({"season": request.word} if studies else
                                                           {"contains": request.word} if request.word else {})},
                      depends_on=["step_catalog"], max_tool_calls=1)]
    return Plan(plan_id=f"plan_catalog_list_{uuid.uuid4().hex[:6]}", question=question,
                interpretation=(f"List the {many} in the cached {_CATALOG_WORD[export]}{containing}: a fixed list written by code "
                                "(no planner, no model call, no measurements)."),
                status="ready", scope=ResolvedScope(base_url=base_url), steps=steps, created_at_utc=now)


def evidence(listing: CatalogList, registry: Any) -> tuple[RetrievalReport, AnalysisReport]:
    """The catalog table the list was read from (hash-checked by the registry) and one count, made by code."""
    many = _NOUNS[listing.request.entity][1]
    manifest = registry.manifest(listing.source.artifact_id)
    if manifest.sha256 != listing.source.sha256 or not manifest.complete:
        raise ValueError("the catalog table behind this list changed or is incomplete")
    word = listing.request.word
    retrieval = RetrievalReport(status="completed", complete=True, artifacts=[manifest], request_ids=list(manifest.source_request_ids),
                                warnings=["read from the cached catalog by controller code: no Retriever agent, no model call, "
                                          "no measurements"])
    studies = listing.request.entity == "studies"
    word = listing.season or word if studies else word
    claim = Claim(claim_id=f"clm_catalog_list_{listing.request.entity}", kind="count", value=len(listing.items),
                  denominator=Denominator(name=f"{many} in the {_CATALOG_WORD[manifest.kind]}", n=listing.total),
                  source_artifact_hashes=[manifest.sha256],
                  filters={"season": word} if studies else {"name contains": word} if word else {},
                  transformation=(f"number of studies one of whose recorded seasons is exactly '{word}'"
                                  if studies else f"number of {many}" + (f" whose name contains '{word}' (ignoring case)" if word else "")))
    caveats = [f"The list shows the catalog as fetched {_when(listing.source)}; records added later are not in it."]
    if studies:
        caveats.append(f"A study is listed when one of its recorded seasons is exactly '{word}'; a season name that only "
                       "contains it is another season.")
    elif listing.request.entity != "traits":
        caveats.append(f"{many.capitalize()} come from the study records: only those named by at least one study are listed.")
    analysis = AnalysisReport(status="completed", claims=[claim], caveats=caveats,
                              methods=[f"catalog list: {many} read by code from the hash-checked {manifest.kind} table "
                                       f"{manifest.artifact_id}; names compared ignoring case; no model inference, no measurements"])
    return retrieval, analysis


def run_note(listing: CatalogList) -> str:
    many = _NOUNS[listing.request.entity][1]
    table, word = listing.source, listing.request.word
    which = (f"{len(listing.items)} of {listing.total} studies with season '{listing.season or word}'" if listing.request.entity == "studies"
             else f"{len(listing.items)} of {listing.total} {many}" + (f" containing '{word}'" if word else ""))
    return (f"catalog list: {which}, read by code from the {_CATALOG_WORD[table.export]} of {table.base_url} "
            f"({table.artifact_id}, {table.rows} rows), fetched {_when(table)}"
            + (" in this run" if table.fetched_in_this_run else " (saved copy)") + "; no planner, no model call, no measurements")


# --------------------------------------------------------------------------
# A catalog setup only (app startup with the operator's standing approval, run.Controller.catalog_setup_phase)
# --------------------------------------------------------------------------

def setup_plan(question: str, base_url: str, now: datetime) -> Plan:
    """The plan of a run that only prepares the two catalogs, written by code (never by the planner)."""
    steps = [PlanStep(step_id="step_studies", agent="retriever", action="export_metadata", inputs={"entity": "studies"}, max_tool_calls=1),
             PlanStep(step_id="step_traits", agent="retriever", action="export_metadata", inputs={"entity": "variables"}, max_tool_calls=1),
             PlanStep(step_id="step_rows", agent="analyst", action="count_catalog_rows", inputs={},
                      depends_on=["step_studies", "step_traits"], max_tool_calls=1)]
    return Plan(plan_id=f"plan_catalog_setup_{uuid.uuid4().hex[:6]}", question=question,
                interpretation=("Prepare the public study and trait catalogs and count their rows: a fixed plan written by code "
                                "(no planner, no model call, no measurements)."),
                status="ready", scope=ResolvedScope(base_url=base_url), steps=steps, created_at_utc=now)


def setup_evidence(tables: dict[str, CatalogTable], registry: Any) -> tuple[RetrievalReport, AnalysisReport]:
    """The two catalog tables (hash-checked by the registry) and their row counts, made by code."""
    manifests = []
    for export in ("studies", "variables"):
        manifest = registry.manifest(tables[export].artifact_id)
        if manifest.sha256 != tables[export].sha256 or not manifest.complete:
            raise ValueError("a catalog table changed or is incomplete")
        manifests.append(manifest)
    retrieval = RetrievalReport(status="completed", complete=True, artifacts=manifests,
                                request_ids=[r for m in manifests for r in m.source_request_ids],
                                warnings=["catalog setup only, by controller code: no Retriever agent, no model call, no measurements"])
    claims = [Claim(claim_id=f"clm_catalog_setup_{m.kind}", kind="count", value=m.row_count,
                    denominator=Denominator(name=f"rows of the {_CATALOG_WORD[m.kind]}", n=m.row_count),
                    source_artifact_hashes=[m.sha256], transformation=f"number of rows in the {_CATALOG_WORD[m.kind]}")
              for m in manifests]
    analysis = AnalysisReport(status="completed", claims=claims,
                              caveats=[f"The catalogs as fetched {_when(tables['studies'])}; records added later are not in them."],
                              methods=["catalog setup: rows of the two hash-checked catalog tables counted by code; "
                                       "no model inference, no measurements"])
    return retrieval, analysis


def is_refresh_request(text: str) -> bool:
    """The whole message asks to download the saved catalogs again: "refresh catalogs", "please update the catalog"."""
    return _REFRESH.fullmatch(" ".join(_TOKEN.findall((text or "").lower().replace("-", " ")))) is not None


def setup_message(tables: dict[str, CatalogTable], *, fetched_now: bool, again: bool = True) -> str:
    """The chat's answer to a catalog setup run (a refresh): what the catalogs hold now and when they were fetched."""
    studies, traits = tables["studies"], tables["variables"]
    head = (("**The study and trait catalogs were downloaded again.**" if again else
             "**The study and trait catalogs were downloaded.**") if fetched_now
            else "**The study and trait catalogs were already saved; nothing new was requested.**")
    return (f"{head} The study catalog has {_count(studies.rows, 'study', 'studies')} and the trait catalog "
            f"{_count(traits.rows, 'trait', 'traits')}, from {_host(studies.base_url)}, fetched {_when(studies)}. "
            "No measurements were fetched. Lists and catalog answers use these catalogs from now on.")
