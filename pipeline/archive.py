"""Long-term record of every story that reached the front page.

What is kept, and for how long (this is published on the "How it works" page):
- Headline, link, outlet name and publication time: kept indefinitely, for every outlet that
  covered the story and every headline seen (not only the three shown on the front page).
- Standfirsts and the AI description: deleted `ai_text_days` after the story was last
  on the front page.
- Stories whose headlines or standfirsts mention court proceedings (see COURT_WORDS):
  the AI description and standfirsts are never stored at all.
- Anything listed in config "archive" -> "removed" (a story id or any article URL) is
  deleted from the archive and kept off the front page.

The file lives at data/archive.json while the workflow runs and is stored between runs
on the `archive` branch, as a single commit that is replaced every run, so deleted text
really disappears from the branch.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

log = logging.getLogger("samestory")

FORMAT_VERSION = 1
MERGE_WINDOW_HOURS = 72   # a story whose links overlap an entry seen this recently is the same story

# Whole words only, so "courtesy" or "trialled" don't count. Deliberately cautious: a false
# match only means less AI text is archived, a missed one could be a legal problem.
DEFAULT_COURT_WORDS = [
    "accused", "acquitted", "alleged", "allegedly", "arrest", "arrested", "bail", "charged",
    "convicted", "conviction", "court", "courts", "defendant", "guilty", "inquest", "judge",
    "jury", "magistrates", "plead", "pleaded", "pleads", "prosecuted", "prosecution",
    "prosecutor", "remanded", "sentenced", "trial", "verdict",
]


class ArchiveError(Exception):
    """The archive file exists but can't be read. The run stops rather than overwrite it."""


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def parse_time(value) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def court_pattern(words: list) -> re.Pattern:
    words = sorted({w.strip().lower() for w in words if w and w.strip()}, key=len, reverse=True)
    if not words:
        return re.compile(r"(?!x)x")   # matches nothing
    return re.compile(r"\b(?:%s)\b" % "|".join(re.escape(w) for w in words), re.IGNORECASE)


def removed_keys(cfg_archive: dict) -> set:
    return {str(x).strip() for x in (cfg_archive or {}).get("removed", []) if str(x).strip()}


def story_urls(story) -> set:
    return {a.url for a in story.picks.values()} | {a.url for a in story.others}


def is_removed(story, removed: set) -> bool:
    return bool(removed) and bool(({story.id} | story_urls(story)) & removed)


class Archive:
    def __init__(self, path: Optional[str], cfg_archive: Optional[dict] = None):
        self.path = path
        self.cfg = cfg_archive or {}
        self.ai_days = float(self.cfg.get("ai_text_days", 14))
        self.court_re = court_pattern(self.cfg.get("court_words", DEFAULT_COURT_WORDS))
        self.removed = removed_keys(self.cfg)
        self.entries = {}
        self.blindspot_watch = {}   # state for pipeline/blindspots.py: when each one-sided group was first checked
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                entries = data["stories"]
                if not isinstance(entries, dict):
                    raise ValueError("'stories' is not an object")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ArchiveError(f"The archive file {path} could not be read ({exc}). "
                                   "Stopping so it isn't overwritten.") from exc
            self.entries = entries
            watch = data.get("blindspot_watch", {})
            self.blindspot_watch = watch if isinstance(watch, dict) else {}

    # ---- adding stories -------------------------------------------------
    def mentions_court(self, story) -> bool:
        arts = list(story.picks.values()) + list(story.others)
        return any(self.court_re.search(f"{a.title} {a.summary}") for a in arts)

    def _find(self, story, urls: set, now: datetime) -> Optional[dict]:
        if story.id in self.entries:
            return self.entries[story.id]
        cutoff = now - timedelta(hours=MERGE_WINDOW_HOURS)
        for entry in self.entries.values():
            last = parse_time(entry.get("last_seen"))
            if last and last >= cutoff and urls & set(entry.get("urls", [])):
                return entry
        return None

    def record(self, stories: list, now: datetime) -> int:
        """Add or update the given front-page stories. Returns how many were new."""
        new = 0
        for s in stories:
            if is_removed(s, self.removed):
                continue
            urls = story_urls(s)
            entry = self._find(s, urls, now)
            if entry is None:
                entry = {"id": s.id, "first_seen": now.isoformat(), "urls": []}
                self.entries[s.id] = entry
                new += 1
            court = bool(entry.get("court")) or self.mentions_court(s)
            # An AI description belongs to the exact headlines it was written about. If the headlines
            # shown for this story have changed, the old description is stale: drop it, and a fresh
            # one is stored below if this run has one.
            old_picks = {lean: a.get("url") for lean, a in (entry.get("articles") or {}).items()}
            if old_picks and old_picks != {lean: a.url for lean, a in s.picks.items()}:
                entry.pop("comparison", None)
            entry["headlines"] = self._merge_headlines(entry.get("headlines"), s)
            entry.update({
                "last_seen": now.isoformat(),
                "outlet_count": max(int(entry.get("outlet_count", 0)), s.outlet_count),
                "court": court,
                "urls": sorted(set(entry.get("urls", [])) | urls),
                "articles": {
                    lean: {"outlet_id": a.outlet_id, "outlet": a.outlet, "title": a.title,
                           "url": a.url, "published": _iso(a.published),
                           **({} if court or not a.summary else {"summary": a.summary})}
                    for lean, a in s.picks.items()
                },
                "others": [{"outlet": a.outlet, "title": a.title, "url": a.url} for a in s.others],
            })
            cmp_ = s.comparison if s.comparison and s.comparison.get("same_story", True) else None
            if cmp_ and not court:
                entry["comparison"] = cmp_
        self.prune(now)
        return new

    @staticmethod
    def _merge_headlines(existing, story) -> list:
        """Every headline ever seen for this story, one per article link, oldest first.
        The front page only shows one per column and replaces them as the story moves, but a
        permanent record of how each outlet headlined it over time needs all of them.
        Standfirsts are not stored here (they follow the 14-day rule); headlines are kept."""
        by_url = {h["url"]: h for h in (existing or []) if isinstance(h, dict) and h.get("url")}
        members = story.members or (list(story.picks.values()) + list(story.others))
        for a in members:
            by_url[a.url] = {"outlet_id": a.outlet_id, "outlet": a.outlet, "lean": a.lean,
                             "title": a.title, "url": a.url, "published": _iso(a.published)}
        return sorted(by_url.values(), key=lambda h: (h.get("published") or "", h["url"]))

    # ---- removing text ----------------------------------------------------
    @staticmethod
    def _strip_text(entry: dict) -> None:
        entry.pop("comparison", None)
        for a in (entry.get("articles") or {}).values():
            a.pop("summary", None)

    def prune(self, now: datetime) -> None:
        cutoff = now - timedelta(days=self.ai_days)
        for key, entry in list(self.entries.items()):
            if self.removed and ({key, entry.get("id")} | set(entry.get("urls", []))) & self.removed:
                del self.entries[key]
                continue
            last = parse_time(entry.get("last_seen"))
            if entry.get("court") or last is None or last < cutoff:
                self._strip_text(entry)

    def save(self, now: datetime) -> None:
        self.prune(now)
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"version": FORMAT_VERSION, "updated": now.isoformat(), "stories": self.entries,
                       "blindspot_watch": self.blindspot_watch},
                      fh, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, self.path)   # never leaves a half-written file behind

    def sorted_entries(self) -> list:
        """Newest first."""
        return sorted(self.entries.values(), key=lambda e: e.get("first_seen", ""), reverse=True)
