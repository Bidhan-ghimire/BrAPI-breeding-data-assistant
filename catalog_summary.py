"""catalog_summary.py — what the cached catalog holds: a short text the front-door model reads, and the answer code writes
for a question about the catalog (Step 2b).

"tell me about Florida" started a run before: the planner asked which location was meant, because the catalog can only be
read inside a run. Now code writes a short summary of the CACHED study and trait catalogs (the same two exports every run
loads first). The model reads it to choose the label; a question about what the catalog holds is then answered by CODE:

* The summary (about 1,400 characters for SweetPotatoBase, since it goes with every label-deciding call): the totals, the
  locations with the most studies, every season named by a study, the study types with the most studies, how many studies
  name a program, a few study names, and when the catalogs were fetched; every count is in brackets. For each message code
  adds one line: the catalog's location names the message mentions, with their number of studies, or that it mentions none.
* The answer to a catalog question (answer): written by code only — the totals; the locations, study types and seasons the
  message names, each with its number of studies; when it names two kinds of them, the number of studies that match them
  together (at Gurue in 2025: 0); when it names none, the locations and study types with the most studies; what a missing
  name does and does not mean; and where the facts came from. The model writes nothing in it, so no count can be wrong, in
  any language, or attached to the wrong place. Database names are escaped like database text.
* A location is "mentioned" when the message has a distinctive word of its name, ignoring case and accents: words of 3 or
  more letters that are not common words such as "station" or "state", numbers of 2 or more digits, and 2-letter words only
  in capitals in both and not everyday words ("NC" yes; "me", "OK", "ID" no; none in a message written all in capitals),
  or the whole name ("Wa", as written). A study type is named by its whole name ("variety release trials") or, when no
  whole name is there, by a distinctive word. A year names the seasons whose names have it as a word. The catalog stores
  location names only (often a station's name), without towns, states or countries, and every answer says so.
* Read from the cache only, by an offline client: nothing is requested, no run folder and no file is written. A catalog
  that is not cached completely gives no summary (then the model is not offered the catalog_question label). Kept in memory
  for CACHE_SECONDS per cache folder and server, so it is not rebuilt for every message.
"""
from __future__ import annotations

import re
import threading
import time
import unicodedata
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from agents.coordinator import KnownMetadata
from brapi_client import BrapiClient, load_settings
from brapi_mcp_server import EXPORT_SHAPES
from cache_store import CacheStore
from retrieval_review import _cell as _escaped       # the same escaping the data review uses for database text

__all__ = ["CatalogSummary", "summarize", "from_cache", "forget", "CACHE_SECONDS", "EXPORT_MAX_PAGES"]

CACHE_SECONDS = 600.0          # how long a summary is kept in memory before the cache is read again
EXPORT_MAX_PAGES = 20          # pages per catalog, as a run's cache read (brapi_mcp_server.ToolContext.export_max_pages; tested)
NAME_CHARS = 60                # the most of one database name in the summary and in the answer
LOCATIONS_LISTED = 10          # the locations with the most studies; the message's own mentions are added per message
SEASONS_LISTED = 40
TYPES_LISTED = 5
STUDIES_NAMED = 3
MENTIONS_LISTED = 10           # locations or seasons a message names, listed in the answer (whole items; the rest are counted)
TOP_IN_ANSWER = 5              # the locations with the most studies, in an answer whose message names none
_WORD = re.compile(r"[^\W_]+")
_COMMON = frozenset({
    "the", "and", "for", "with", "station", "stations", "farm", "farms", "field", "fields", "research", "center", "centre",
    "site", "sites", "location", "locations", "trial", "trials", "experimental", "experiment", "agricultural", "agriculture",
    "university", "institute", "national", "crops", "crop", "hall", "plot", "plots", "nursery", "south", "north", "east",
    "west", "central", "new", "old", "upper", "lower", "state", "states", "county", "city", "district", "region", "area",
    "village", "town", "valley", "river", "lake", "mount", "hill", "hills", "park", "garden", "gardens", "zone", "campus",
    "college", "school", "office", "unit", "main", "sub", "test", "breeding", "program", "programme", "study", "studies",
    "season", "seasons", "catalog", "catalogue", "about", "what", "which", "where", "there", "any", "anything", "tell",
    "how", "many", "much", "are", "were", "was", "has", "have", "this", "that", "these", "those", "from", "into", "type",
    "types", "kind", "kinds", "number", "count", "list", "show", "give", "all", "each", "every", "most", "trait", "traits",
    "name", "names", "contain", "contains", "word", "words", "data", "record", "records", "value", "values", "mean",
    "average", "total", "year", "years"})
# 2-letter words in capitals that are everyday words, not place codes ("OK", "ID" as in "study ID").
_EVERYDAY_CAPITALS = frozenset({"ID", "OK", "OR", "IN", "IT", "IS", "AT", "ON", "TO", "OF", "AN", "AS", "BY", "DO", "GO", "HI",
                                "IF", "ME", "MY", "NO", "SO", "UP", "WE", "AM", "PM", "US", "UK", "EU", "TV", "VS"})
_ALL_CAPITALS = re.compile(r"\b[A-Z]{3,}\b")     # a word of 3+ capitals: with no lower-case letter, a shouted message
_YEAR_RE = re.compile(r"(?<![A-Za-z0-9])(?:19|20)\d\d(?![A-Za-z0-9])")      # a year as a word, not inside an ID
# A range of seasons ("since 2019", "2019 to 2025"): each season is counted alone, and the answer says so.
_RANGE_RE = re.compile(r"\b(?:since|after|before|from|until|till|through|between|later than|earlier than)\s+(?:the\s+)?"
                       r"(?:(?:season|year)\s+)?(?:19|20)\d\d\b|\b(?:19|20)\d\d\s*(?:-|–|—|to\b|through\b|until\b|onwards?\b|"
                       r"and (?:later|after)\b|or (?:later|after)\b)", re.IGNORECASE)

_memo: dict[tuple[str, str], tuple[float, "CatalogSummary"]] = {}
_lock = threading.Lock()
_clock = time.monotonic                       # tests replace it


@dataclass(frozen=True)
class CatalogSummary:
    """The summary text for the model, and the catalog facts code writes for the person."""

    text: str                                          # what the model reads (facts, not instructions); for_message adds a line
    host: str
    fetched_at: datetime | None
    locations: tuple[tuple[str, int], ...] = ()        # EVERY location named by studies (its name, its number of studies)
    seasons: tuple[tuple[str, int], ...] = ()          # every season named by studies, newest first
    studies: int = 0
    traits: int = 0
    unnamed_locations: int = 0                         # location IDs that studies give without a name (never "mentioned")
    types: tuple[tuple[str, int], ...] = ()            # every study type (its name, its number of studies), most first
    rows: tuple[tuple[str, tuple[str, ...], str], ...] = ()   # per study: its location's name, its seasons, its type (combined counts)

    @property
    def location_count(self) -> int:
        return len(self.locations) + self.unnamed_locations

    def mentioned(self, message: str) -> list[tuple[str, int]]:
        """The catalog's location names the message mentions (see the module's notes): "clinton" mentions "Clinton, NC";
        "NC" mentions it too; "me", "ok" or "ID" mention nothing; "Florida" mentions no "..., FL"."""
        message = self._without_type_names(message)               # "Variety Release Trials" names a type, not a place
        long_words, capitals = _words(message)
        if _ALL_CAPITALS.search(message or "") and not any(c.islower() for c in message or ""):
            capitals = set()                                     # written all in capitals: a 2-letter word says nothing
        tokens = set(_WORD.findall(message or ""))
        return [(name, count) for name, count in self.locations
                if _name_words_meet(name, long_words, capitals) or (len(name) <= 2 and name in tokens)]

    def _whole_types(self, message: str) -> tuple[list[tuple[str, int]], str]:
        """The study types whose whole name the message has ("variety release trials"), longest name first, and the message
        with each of them blanked out (so "Yield Trial" is not also read inside "Advanced Yield Trial", and a type's words
        name no location)."""
        found, message = [], _plain_letters(message or "")       # accents dropped, capitals kept ("NC" still reads as NC)
        for name, count in sorted(self.types, key=lambda t: -len(_WORD.findall(t[0]))):
            words = _WORD.findall(_plain_letters(name))
            pattern = re.compile(r"(?<!\w)" + r"[\W_]+".join(map(re.escape, words)) + r"s?(?!\w)", re.IGNORECASE) if words else None
            if pattern is not None and pattern.search(message):
                found.append((name, count))
                message = pattern.sub(" ", message)
        order = {name: i for i, (name, _) in enumerate(self.types)}
        return sorted(found, key=lambda t: order[t[0]]), message or ""

    def _without_type_names(self, message: str) -> str:
        return self._whole_types(message)[1] if self.types else (message or "")

    def types_mentioned(self, message: str, places: list[tuple[str, int]] = ()) -> tuple[list[tuple[str, int]], bool]:
        """The study types the message names: by their whole name, or, when it has no whole name, by a distinctive word
        ("yield") that is not a word of a location it names ("Gwebi Variety Testing Center" names no type). The second value
        says whether whole names were found."""
        whole, _ = self._whole_types(message)
        if whole:
            return whole, True
        long_words, _ = _words(message)
        long_words -= {_fold(w) for name, _ in places for w in _WORD.findall(name)}
        return [(name, count) for name, count in self.types if _name_words_meet(name, long_words, set())], False

    def seasons_mentioned(self, message: str) -> tuple[list[tuple[str, int]], list[str]]:
        """Seasons the message names: a season's whole name ("Fall season"), or a year that is in a season's name ("2025"
        names "2025" and "Fall 2025"); and the years the message names that no season's name has."""
        folded = f" {' '.join(_WORD.findall(_fold(message)))} "
        years = list(dict.fromkeys(_YEAR_RE.findall(message)))
        named = [(name, count) for name, count in self.seasons
                 if f" {' '.join(_WORD.findall(_fold(name)))} " in folded or set(_YEAR_RE.findall(name)) & set(years)]
        missing = [y for y in years if not any(y in _YEAR_RE.findall(name) for name, _ in self.seasons)]
        return named, missing

    def mentions_line(self, message: str) -> str:
        """For the model: the locations the message mentions, checked by code (with their number of studies)."""
        found = self.mentioned(message)
        if not found:
            return (f"Locations the message mentions: none (no distinctive word of the message is in any of the "
                    f"{self.location_count} location names; common words such as 'station' are not compared, and the "
                    "catalog stores location names only, without towns, states or countries).")
        shown = "; ".join(f"{name} ({count})" for name, count in found[:MENTIONS_LISTED])
        more = f"; and {len(found) - MENTIONS_LISTED} more" if len(found) > MENTIONS_LISTED else ""
        return f"Locations the message mentions (checked by code; number of studies in brackets): {shown}{more}."

    def for_message(self, message: str) -> str:
        """The summary for one message: the facts kept for every message, then that message's own location line."""
        return f"{self.text}\n{self.mentions_line(message)}"

    def answer(self, message: str) -> str:
        """The chat's answer to a catalog question, written by code only: what the catalog holds; the locations, study
        types and seasons the message names (whole items), each with its number of studies, and, when it names two kinds
        of them, the studies that match them together; what a missing name means; and the source. Names are escaped."""
        name = lambda text: _escaped(text, NAME_CHARS)
        listed = lambda items: ("; ".join(f"{name(n)} ({_count(c, 'study', 'studies')})" for n, c in items[:MENTIONS_LISTED])
                                + (f"; and {len(items) - MENTIONS_LISTED} more" if len(items) > MENTIONS_LISTED else ""))
        newest = [_year(season) for season, _ in self.seasons if _year(season)]
        span = f", with seasons from {min(newest)} to {max(newest)}" if newest else ""
        lines = ["**From the saved catalog** (no records were looked up):", "",
                 f"- It has {_count(self.studies, 'study', 'studies')}, {_count(self.traits, 'trait', 'traits')} and "
                 f"{_count(self.location_count, 'location', 'locations')} named by studies{span}."]
        found = self.mentioned(message)
        types, whole_types = self.types_mentioned(message, found)
        seasons, missing = self.seasons_mentioned(message)
        places_note = ("The catalog stores only location names (often a station's name), without towns, states or countries, "
                       "so a place may also be there under another name; ask **list all locations** to see every name.")
        if found:
            lines.append(f"- Locations whose names contain a word of your message: {listed(found)}. {places_note}")
        elif types or seasons:                    # so no count below reads as the count of a place the catalog does not name
            lines.append("- No location name contains a distinctive word of your message, so every count here is for all "
                         f"locations. {places_note}")
        if types:
            lines.append(f"- Study types {'you named' if whole_types else 'whose names contain a word of your message'}: "
                         f"{listed(types)}.")
        if seasons:
            lines.append(f"- Seasons you named: {listed(seasons)}.")
        if missing:
            lines.append("- No study names a season with " + ", ".join(missing[:MENTIONS_LISTED]) + ".")
        named_types = types if whole_types else []            # a type guessed from one word is never combined
        if sum(map(bool, (found, named_types, seasons))) >= 2:
            lines.append(f"- Each count above is for that one name; together, studies "
                         f"{self._together_words(found, named_types, seasons)}: "
                         f"{_count(self._together(found, named_types, seasons), 'study', 'studies')}.")
        if seasons and _RANGE_RE.search(message or ""):
            lines.append("- Each season is counted alone: a range of seasons is not added up here.")
        if not (found or types or seasons):
            top_places = "; ".join(f"{name(n)} ({_count(c, 'study', 'studies')})" for n, c in self.locations[:TOP_IN_ANSWER])
            top_types = listed(self.types[:TOP_IN_ANSWER]) + (f"; and {len(self.types) - TOP_IN_ANSWER} more"
                                                              if len(self.types) > TOP_IN_ANSWER else "")
            lines.append("- Your message names no location, study type or season of the catalog."
                         + (f" The locations with the most studies: {top_places}." if top_places else "")
                         + (f" The study types with the most studies: {top_types}." if self.types else "")
                         + f" {places_note}")
        lines += ["", self.source_line()]
        return "\n".join(lines)

    def _together(self, found: list[tuple[str, int]], types: list[tuple[str, int]], seasons: list[tuple[str, int]]) -> int:
        places, kinds, named = {n for n, _ in found}, {n for n, _ in types}, {n for n, _ in seasons}
        return sum(1 for place, its_seasons, kind in self.rows
                   if (not places or place in places) and (not kinds or kind in kinds)
                   and (not named or any(s in named for s in its_seasons)))

    @staticmethod
    def _together_words(found: list[tuple[str, int]], types: list[tuple[str, int]], seasons: list[tuple[str, int]]) -> str:
        def either(items: list[tuple[str, int]], what: str) -> str:
            names = [_escaped(n, NAME_CHARS) for n, _ in items[:3]]
            return " or ".join(names) + (f" or {len(items) - 3} other {what}" if len(items) > 3 else "")
        parts = ([f"at {either(found, 'locations')}"] if found else []) + ([f"of type {either(types, 'types')}"] if types else []) \
            + ([f"in {either(seasons, 'seasons')}"] if seasons else [])
        return " ".join(parts)

    def source_line(self) -> str:
        """Written by code under an answer from the summary."""
        return (f"*From the saved catalog of {self.host}, fetched {_when(self.fetched_at)}. Nothing new was requested; "
                "ask a data question to look up records.*")


def _plain_letters(text: str) -> str:
    """Without accents, capitals kept: "Bogotá" -> "Bogota"."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _fold(text: str) -> str:
    """Lower case without accents: "Bogotá" -> "bogota"."""
    return _plain_letters(text).casefold()


def _words(text: str) -> tuple[set[str], set[str]]:
    """The message's words of 3 or more characters and its numbers of 2 or more digits (folded), and its 2-letter words
    written in capitals that are not everyday words ("NC")."""
    tokens = _WORD.findall(text or "")
    return ({_fold(w) for w in tokens if len(w) >= 3 or (len(w) == 2 and w.isdigit())},
            {w for w in tokens if len(w) == 2 and w.isupper() and w.isalpha() and w not in _EVERYDAY_CAPITALS})


def _name_words_meet(name: str, long_words: set[str], capitals: set[str]) -> bool:
    """A distinctive word of the name the message also has: 3 or more characters (not a common word), a number of 2 or more
    digits ("Station 12"), or a 2-letter word in capitals in both ("NC")."""
    tokens = _WORD.findall(name)
    return any(((len(w) >= 3 or w.isdigit()) and _fold(w) not in _COMMON and _fold(w) in long_words)
               or (len(w) == 2 and w.isupper() and w.isalpha() and w in capitals) for w in tokens)


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


# --------------------------------------------------------------------------
# The summary, written by code from the validated catalog (agents.coordinator.KnownMetadata)
# --------------------------------------------------------------------------

def _clean(text: Any) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= NAME_CHARS else text[:NAME_CHARS].rstrip() + "…"


def _when(fetched_at: datetime | None) -> str:
    return "at a time that was not recorded" if fetched_at is None else fetched_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _year(text: str) -> int | None:
    years = [int(y) for y in re.findall(r"(?<!\d)(1[89]\d\d|2\d\d\d)(?!\d)", text)]
    return max(years) if years else None


def _shown(items: list[str], limit: int) -> str:
    text = "; ".join(items[:limit])
    return text + (f"; and {len(items) - limit} more" if len(items) > limit else "")


def summarize(metadata: KnownMetadata, *, host: str, fetched_at: datetime | None) -> CatalogSummary:
    """The catalog in a few lines: only counts and names, all of them read from the catalog by code."""
    studies = list(metadata.studies)
    locations: dict[str, dict[str, Any]] = {}
    no_location = 0
    for study in studies:
        if not study.location_id and not study.location_name:
            no_location += 1
            continue
        entry = locations.setdefault(study.location_id or f"name:{study.location_name}",
                                     {"names": set(), "studies": set(), "id": study.location_id})
        entry["studies"].add(study.study_id)
        if study.location_name:
            entry["names"].add(_clean(study.location_name))
    ranked = sorted(((" / ".join(sorted(e["names"])), len(e["studies"])) for e in locations.values() if e["names"]),
                    key=lambda item: (-item[1], item[0].casefold()))
    place_of = {key: " / ".join(sorted(e["names"])) for key, e in locations.items()}
    unnamed = [len(e["studies"]) for e in locations.values() if not e["names"]]      # an ID without a name: counted, never matched

    seasons: dict[str, set[str]] = {}
    no_season = 0
    for study in studies:
        names = [_clean(s) for s in study.seasons if _clean(s)]
        no_season += not names
        for name in names:
            seasons.setdefault(name, set()).add(study.study_id)
    season_order = sorted(seasons, key=lambda s: (_year(s) is None, -(_year(s) or 0), s.casefold()))

    types: dict[str, int] = {}
    for study in studies:
        if study.study_type:
            types[_clean(study.study_type)] = types.get(_clean(study.study_type), 0) + 1
    type_items = [f"{name} ({count})" for name, count in sorted(types.items(), key=lambda t: (-t[1], t[0].casefold()))]

    newest = sorted(studies, key=lambda s: (-max([_year(x) or 0 for x in s.seasons] or [0]), s.study_id))
    named = [f"“{_clean(s.name) or '(no name)'}” (ID {_clean(s.study_id)}; {_clean(s.location_name) or 'no location'}; "
             f"{', '.join(_clean(x) for x in s.seasons) or 'no season'})" for s in newest[:STUDIES_NAMED]]
    programs = sum(bool(s.program_id or s.program_name) for s in studies)

    facts = [
        f"Totals: {len(studies)} studies, {len(metadata.variables)} traits, {len(ranked) + len(unnamed)} locations and "
        f"{len(seasons)} seasons named by studies, {len(types)} study types. Studies that name no location: {no_location}; "
        f"no season: {no_season}.",
        "Locations with the most studies (studies in brackets): "
        + (_shown([f"{name} ({count})" for name, count in ranked], LOCATIONS_LISTED) or "none")
        + (f"; and {len(unnamed)} location IDs without a name" if unnamed else ""),
        "Seasons named by studies, newest first (studies in brackets): "
        + (_shown([f"{name} ({len(seasons[name])})" for name in season_order], SEASONS_LISTED) or "none"),
        "Study types with the most studies (studies in brackets): " + (_shown(type_items, TYPES_LISTED) or "none recorded"),
        f"Programs: {programs} of {len(studies)} studies name a program in the standard program fields.",
        "A few study names, newest season first: " + ("; ".join(named) or "none"),
    ]
    head = f"Saved catalog of {host}, fetched {_when(fetched_at)}. These facts were read by code; no records were looked up."
    rows = tuple((place_of.get(study.location_id or f"name:{study.location_name}", "") if (study.location_id or study.location_name)
                  else "", tuple(_clean(s) for s in study.seasons if _clean(s)), _clean(study.study_type)) for study in studies)
    return CatalogSummary(text="\n".join([head, *facts]), host=host, fetched_at=fetched_at, locations=tuple(ranked),
                          seasons=tuple((name, len(seasons[name])) for name in season_order), studies=len(studies),
                          traits=len(metadata.variables), unnamed_locations=len(unnamed),
                          types=tuple(sorted(types.items(), key=lambda t: (-t[1], t[0].casefold()))), rows=rows)


# --------------------------------------------------------------------------
# Reading the cache: an offline client, no request, no file
# --------------------------------------------------------------------------

def from_cache(config: Any) -> CatalogSummary | None:
    """The summary of the catalogs a run with this configuration would load, or None when they are not completely cached
    (or anything else goes wrong: the front door then works as before, without it)."""
    try:
        settings = load_settings()
        cache_dir = Path(config.cache_dir) if getattr(config, "cache_dir", None) else settings.cache_dir / "live"
        key = (str(cache_dir.resolve()), settings.base_url)
        with _lock:
            hit = _memo.get(key)
            if hit is not None and _clock() - hit[0] < CACHE_SECONDS:
                return hit[1]
            metadata, fetched_at = _read(settings, cache_dir)
            if metadata is None:
                return None
            summary = summarize(metadata, host=urlsplit(settings.base_url).hostname or settings.base_url, fetched_at=fetched_at)
            _memo[key] = (_clock(), summary)
            return summary
    except Exception:  # noqa: BLE001 - a summary is a convenience; it must never stop the chat
        return None


def _read(settings: Any, cache_dir: Path) -> tuple[KnownMetadata | None, datetime | None]:
    """The two catalog exports from the cache, exactly as a run's cache read asks for them (offline: never a request)."""
    offline = replace(settings, mode="offline", cache_dir=cache_dir)
    client = BrapiClient(offline, cache=CacheStore(cache_dir), run_id="catalog_summary")
    studies = client.studies(None, approval=None, max_pages=EXPORT_MAX_PAGES)
    variables = client.variables(approval=None, max_pages=EXPORT_MAX_PAGES)
    if any(result.status == "failed" or not result.complete for result in (studies, variables)):
        return None, None
    shape_study, shape_variable = EXPORT_SHAPES["studies"][0], EXPORT_SHAPES["variables"][0]
    metadata = KnownMetadata.from_rows([shape_study(r) for r in studies.records], [shape_variable(r) for r in variables.records],
                                       base_url=settings.base_url, complete=True)
    fetched = sorted(r.fetched_at_utc for r in client.provenance() if r.fetched_at_utc is not None)
    return metadata, (fetched[0] if fetched else None)


def forget() -> None:
    """Drop the kept summaries (tests; a catalog refresh)."""
    with _lock:
        _memo.clear()
