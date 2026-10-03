"""One-sided coverage ("blindspots"), computed by code only. No AI is involved.

A group of headlines is shown as covered on one side only when ALL of these hold
(the same rules are printed on the How it works page):

- at least `min_outlets` (2) different outlets on one side (left-leaning or right-leaning)
  cover it, and no outlet on the other side is in the group;
- every enabled feed on the other side loaded without error in this run, and returned at
  least one item, so a failed or dead feed can never look like missing coverage;
- no headline from that other side, anywhere in the whole time window, is even loosely
  similar to any headline in the group (similarity below `tfidf_max_similarity`, 0.15);
- it has passed those checks in at least `min_runs` (2) runs spread over at least
  `min_hours` (6) hours. Any run in which the other side is found resets it.

Wording on the site is always "not found in our sources", with the checks listed, never a
claim that an outlet ignored or suppressed a story.

Watch state (when a group was first checked and how many times) is kept in the archive file,
which is the only data that survives reliably between runs.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

SIDES = ("left", "right")
OTHER = {"left": "right", "right": "left"}
DEFAULTS = {
    "enabled": True,
    "min_outlets": 2,
    "tfidf_max_similarity": 0.15,
    "embedding_max_similarity": 0.6,
    "min_runs": 2,
    "min_hours": 6,
    "max_shown": 10,
}
FORGET_AFTER_HOURS = 48   # watch entries not checked for this long are dropped


@dataclass
class Blindspot:
    id: str
    side: str                     # the side that covered it
    articles: list                # Articles in the group (that side, plus any centre ones)
    first_checked: datetime
    last_checked: datetime
    runs: int
    checked_sources: list = field(default_factory=list)   # names of the other side's outlets
    max_similarity: float = 0.0

    @property
    def missing(self) -> str:
        return OTHER[self.side]

    @property
    def outlet_count(self) -> int:
        return len({a.outlet_id for a in self.articles})

    @property
    def latest(self) -> Optional[datetime]:
        dates = [a.published for a in self.articles if a.published]
        return max(dates) if dates else None


def settings(cfg: dict) -> dict:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (cfg.get("blindspots") or {}).items() if not k.startswith("_")})
    return out


def side_status(cfg: dict, status: dict, side: str) -> tuple:
    """(ready, names). Ready only if the side has at least one enabled outlet and every one of
    them loaded and returned items. A feed that loads but is empty or stale is not evidence."""
    sources = [s for s in cfg["sources"] if s.get("enabled", True) and s["lean"] == side]
    ready = bool(sources) and all((status.get(s["id"]) or {}).get("ok") for s in sources)
    return ready, [s["name"] for s in sources]


def _parse(value) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def find_candidates(articles: list, groups: list, vectors, min_outlets: int, exclude_urls: set) -> list:
    """[(side, member_indices, best_other_side_similarity)] for groups covered on one side only."""
    out = []
    by_side = {side: [i for i, a in enumerate(articles) if a.lean == side] for side in SIDES}
    for g in groups:
        arts = [articles[i] for i in g]
        if exclude_urls & {a.url for a in arts}:
            continue
        leans = {a.lean for a in arts}
        for side in SIDES:
            if OTHER[side] in leans:
                continue
            if len({a.outlet_id for a in arts if a.lean == side}) < min_outlets:
                continue
            others = by_side[OTHER[side]]
            best = float(np.max(vectors[g] @ vectors[others].T)) if others else 0.0
            out.append((side, list(g), best))
    return out


def update(watch: dict, articles: list, groups: list, vectors, status: dict, cfg: dict,
           now: datetime, method: str = "tfidf", exclude_urls: Optional[set] = None,
           removed: Optional[set] = None) -> list:
    """Update the watch state in place and return the Blindspots that may be shown now."""
    st = settings(cfg)
    if not st.get("enabled", True) or vectors is None or not articles:
        return []
    max_sim = float(st["embedding_max_similarity"] if method == "embeddings" else st["tfidf_max_similarity"])
    ready = {side: side_status(cfg, status, side) for side in SIDES}
    removed = removed or set()
    seen, shown = set(), []

    def find(urls):
        for wid, e in watch.items():
            if urls & set(e.get("urls", [])):
                return wid, e
        return None, None

    for side, g, best in find_candidates(articles, groups, vectors, int(st["min_outlets"]), exclude_urls or set()):
        arts = [articles[i] for i in g]
        urls = {a.url for a in arts}
        if urls & removed:
            continue
        wid, entry = find(urls)
        if wid in seen:
            continue
        missing_ready, names = ready[OTHER[side]]
        if not missing_ready:
            if wid:
                seen.add(wid)   # can't judge this run: leave the watch state exactly as it was
            continue
        if best >= max_sim:
            if wid:
                del watch[wid]  # something similar was found on the other side: start again
            continue
        if entry is None or entry.get("side") != side:
            if wid:
                del watch[wid]
            wid = hashlib.sha1("|".join(sorted(urls)).encode()).hexdigest()[:10]
            entry = {"side": side, "first_checked": now.isoformat(), "runs": 0, "urls": []}
            watch[wid] = entry
        entry["runs"] = int(entry.get("runs", 0)) + 1
        entry["last_checked"] = now.isoformat()
        entry["urls"] = sorted(set(entry.get("urls", [])) | urls)
        seen.add(wid)
        first = _parse(entry["first_checked"]) or now
        if entry["runs"] >= int(st["min_runs"]) and now - first >= timedelta(hours=float(st["min_hours"])):
            shown.append(Blindspot(id=wid, side=side, articles=arts, first_checked=first, last_checked=now,
                                   runs=entry["runs"], checked_sources=names, max_similarity=round(best, 3)))

    forget = now - timedelta(hours=FORGET_AFTER_HOURS)
    for wid, e in list(watch.items()):
        if wid in seen:
            continue
        last = _parse(e.get("last_checked") or e.get("first_checked"))
        side = e.get("side")
        if side not in SIDES or last is None or last < forget or set(e.get("urls", [])) & removed:
            del watch[wid]
        elif ready[OTHER[side]][0]:
            del watch[wid]  # the other side could be checked and the group no longer qualifies

    epoch = datetime(1970, 1, 1, tzinfo=now.tzinfo)
    shown.sort(key=lambda b: (-b.outlet_count, -(b.latest or epoch).timestamp()))
    return shown[: int(st["max_shown"])]
