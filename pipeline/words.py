"""Word tracker: how often each side's headlines use one word or the other, for the word pairs
listed (and published) in config.json. Pure counting, no AI.

Rules (printed on the Words page and on How it works):
- Only headlines are counted. Standfirsts are deleted from the archive after a fortnight, so
  counting them would make old and new periods incomparable.
- Only headlines from archived stories that both left-leaning and right-leaning outlets
  covered, so both sides are compared on the same set of stories.
- Each article link is counted once, however many times it was seen.
- Whole words only, with simple word forms: "migrant" also matches "migrants" and "migrant's".
  A hyphen or a space both match, so "asylum-seeker" and "asylum seeker" are counted together.
- Rates, not raw counts: the share of that column's headlines using the term, because the
  columns publish different numbers of headlines.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

from .models import LEAN_ORDER

DEFAULTS = {"days": 90, "min_headlines": 15}
MAX_RATE = 999.0   # ceiling for the bar chart, so one very high rate cannot squash the rest


def settings(cfg: dict) -> dict:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (cfg.get("words") or {}).items() if not k.startswith("_")})
    out.setdefault("pairs", [])
    return out


def _terms(value) -> list:
    if isinstance(value, str):
        value = [value]
    return [t.strip() for t in (value or []) if isinstance(t, str) and t.strip()]


def term_pattern(terms: list) -> re.Pattern:
    """Whole-word pattern with plural/possessive forms; hyphen or space both match."""
    alts = []
    for term in sorted(set(t.lower() for t in terms), key=len, reverse=True):
        parts = [re.escape(p) for p in re.split(r"[\s\-]+", term) if p]
        alts.append(r"[\s\-]+".join(parts) + r"(?:s|es)?(?:['\u2019]s?)?")
    if not alts:
        return re.compile(r"(?!x)x")
    return re.compile(r"(?<![\w])(?:%s)(?![\w])" % "|".join(alts), re.IGNORECASE)


def _parse(value) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def collect_headlines(entries: list, since: Optional[datetime]) -> list:
    """Unique (by URL) headline dicts from stories both sides covered, published since `since`."""
    by_url = {}
    for e in entries:
        hs = e.get("headlines")
        if not hs:   # entries archived before headline history existed
            hs = [dict(a, lean=lean) for lean, a in (e.get("articles") or {}).items()]
        if not {"left", "right"} <= {h.get("lean") for h in hs}:
            continue
        for h in hs:
            url, title = h.get("url"), h.get("title")
            if not url or not title or h.get("lean") not in LEAN_ORDER:
                continue
            when = _parse(h.get("published")) or _parse(e.get("first_seen"))
            if since and (when is None or when < since):
                continue
            by_url[url] = {"lean": h["lean"], "title": title}
    return list(by_url.values())


def compute(entries: list, cfg: dict, now: datetime) -> dict:
    st = settings(cfg)
    days = float(st["days"])
    heads = collect_headlines(entries, now - timedelta(days=days) if days > 0 else None)
    totals = {lean: sum(1 for h in heads if h["lean"] == lean) for lean in LEAN_ORDER}
    pairs = []
    for p in st["pairs"]:
        a, b = _terms(p.get("a")), _terms(p.get("b"))
        if not a or not b:
            continue
        ra, rb = term_pattern(a), term_pattern(b)
        by_lean = {}
        for lean in LEAN_ORDER:
            titles = [h["title"] for h in heads if h["lean"] == lean]
            by_lean[lean] = {"a": sum(1 for t in titles if ra.search(t)),
                             "b": sum(1 for t in titles if rb.search(t)),
                             "total": len(titles)}
        pairs.append({"a": a[0], "b": b[0], "a_forms": a, "b_forms": b,
                      "note": p.get("note", ""), "by_lean": by_lean})
    return {"days": days, "min_headlines": int(st["min_headlines"]), "totals": totals,
            "headline_count": len(heads), "pairs": pairs}


def rate(count: int, total: int) -> Optional[float]:
    return 100.0 * count / total if total else None


def bar_width(value: Optional[float], ceiling: float = 20.0) -> float:
    """Width in chart units (0-100) for a rate, capped so one huge rate cannot squash the rest."""
    if value is None:
        return 0.0
    return max(0.5, min(100.0, 100.0 * min(value, ceiling) / ceiling))


def describe(pair: dict, by_lean: dict, min_headlines: int) -> list:
    """One short factual sentence per column. No interpretation, no comparison of which side
    is 'harsher': a codepath that judged that would break design rule 2."""
    out = []
    for lean in ("left", "right"):
        v = by_lean.get(lean) or {}
        total = int(v.get("total") or 0)
        if total < min_headlines:
            out.append(f"Not enough headlines in this column yet ({total} so far).")
            continue
        a_c, b_c = int(v.get("a") or 0), int(v.get("b") or 0)
        if a_c + b_c == 0:
            out.append(f"Neither term appeared in this column's headlines ({total} headlines).")
            continue
        a_r, b_r = 100.0 * a_c / total, 100.0 * b_c / total
        out.append(f"{a_r:.1f}% of this column's headlines used \u201c{pair['a']}\u201d; "
                   f"{b_r:.1f}% used \u201c{pair['b']}\u201d ({total} headlines).")
    return out
