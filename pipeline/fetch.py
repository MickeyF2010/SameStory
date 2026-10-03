"""Fetch and parse RSS/Atom feeds using only the standard library plus requests."""
from __future__ import annotations

import email.utils
import html
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional

import requests

from .models import Article

log = logging.getLogger(__name__)

USER_AGENT = "SameStoryBot/1.0 (headline comparison; links back to every source)"
MAX_BYTES = 3_000_000
ATOM = "{http://www.w3.org/2005/Atom}"
DC = "{http://purl.org/dc/elements/1.1/}"
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def clean_text(value: Optional[str], limit: Optional[int] = None) -> str:
    if not value:
        return ""
    text = TAG_RE.sub(" ", value)
    text = html.unescape(text)
    text = WS_RE.sub(" ", text).strip()
    if limit and len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(",;:.") + "…"
    return text


def parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        dt = None
    if dt is None:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _atom_link(entry: ET.Element) -> str:
    best = ""
    for link in entry.findall(f"{ATOM}link"):
        href = link.get("href", "")
        if not href:
            continue
        if link.get("rel", "alternate") == "alternate":
            return href
        best = best or href
    return best


BARE_AMP = re.compile(rb"&(?!(?:[A-Za-z][A-Za-z0-9]{1,31}|#[0-9]{1,7}|#[xX][0-9A-Fa-f]{1,6});)")
BAD_CHARS = re.compile(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _load_xml(data: bytes) -> ET.Element:
    """Parse XML, retrying once after fixing the two most common feed defects:
    unescaped ampersands and illegal control characters."""
    head = data[:300].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise ValueError("returned a web page instead of a feed (the site may be blocking automated requests)")
    try:
        return ET.fromstring(data)
    except ET.ParseError:
        repaired = BAD_CHARS.sub(b"", BARE_AMP.sub(b"&amp;", data))
        return ET.fromstring(repaired)


def parse_feed(data: bytes, source: dict, summary_max: int = 240) -> list:
    """Parse RSS 2.0 or Atom bytes into Article objects."""
    if b"<!ENTITY" in data:  # refuse entity-expansion tricks
        raise ValueError("feed contains XML entity declarations")
    root = _load_xml(data)
    articles = []

    def add(title, link, summary, date):
        title = clean_text(title)
        link = (link or "").strip()
        if not title or not link.startswith("http"):
            return
        articles.append(
            Article(
                outlet_id=source["id"],
                outlet=source["name"],
                lean=source["lean"],
                title=title,
                summary=clean_text(summary, summary_max),
                url=link,
                published=parse_date(date),
            )
        )

    for item in root.iter("item"):
        add(
            item.findtext("title"),
            item.findtext("link"),
            item.findtext("description"),
            item.findtext("pubDate") or item.findtext(f"{DC}date"),
        )
    for entry in root.iter(f"{ATOM}entry"):
        add(
            entry.findtext(f"{ATOM}title"),
            _atom_link(entry),
            entry.findtext(f"{ATOM}summary") or entry.findtext(f"{ATOM}content"),
            entry.findtext(f"{ATOM}published") or entry.findtext(f"{ATOM}updated"),
        )
    return articles


def fetch_url(url: str, timeout: int = 20) -> bytes:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/atom+xml, text/xml, */*"}, timeout=timeout)
    resp.raise_for_status()
    if len(resp.content) > MAX_BYTES:
        raise ValueError("feed too large")
    return resp.content


def fetch_source(source: dict, settings: dict) -> tuple:
    """Return (articles, errors) for one outlet. Never raises for a single bad feed."""
    articles, errors = [], []
    limit = settings.get("per_feed_limit", 60)
    for url in source["feeds"]:
        try:
            parsed = parse_feed(fetch_url(url), source, settings.get("summary_max_chars", 240))
            articles.extend(parsed[:limit])
        except Exception as exc:  # network, HTTP, XML: log and carry on
            errors.append(f"{source['name']} ({url}): {exc}")
    seen, unique = set(), []
    for a in articles:
        if a.url not in seen:
            seen.add(a.url)
            unique.append(a)
    return unique, errors


def enabled_sources(cfg: dict) -> list:
    return [s for s in cfg["sources"] if s.get("enabled", True)]


def fetch_all_with_status(cfg: dict) -> tuple:
    """Return (articles, status). status maps each enabled source id to
    {name, lean, ok, count, errors, newest}. `ok` means every feed for that source loaded AND
    returned at least one recent item: a feed that parses but is empty or stale is not evidence
    that an outlet covered nothing, so blindspots must never treat it as a loaded feed."""
    all_articles, status = [], {}
    for source in enabled_sources(cfg):
        articles, errors = fetch_source(source, cfg["settings"])
        for err in errors:
            log.warning("Feed problem: %s", err)
        recent = filter_recent(articles, cfg["settings"].get("max_age_hours", 48))
        newest = max((a.published for a in articles if a.published), default=None)
        ok = not errors and bool(recent)
        if not errors and not recent:
            log.warning("%s: feed loaded but nothing recent in it (newest %s)", source["name"],
                        newest.strftime("%Y-%m-%d") if newest else "no dates")
        log.info("%s: %d articles (%d recent)", source["name"], len(articles), len(recent))
        status[source["id"]] = {"name": source["name"], "lean": source["lean"], "ok": ok,
                                "count": len(articles), "recent": len(recent),
                                "newest": newest.isoformat() if newest else None, "errors": errors}
        all_articles.extend(articles)
    return all_articles, status


def fetch_all(cfg: dict) -> list:
    return fetch_all_with_status(cfg)[0]


def filter_recent(articles: list, max_age_hours: float, now: Optional[datetime] = None) -> list:
    now = now or datetime.now(timezone.utc)
    kept = []
    for a in articles:
        if a.published is None:
            continue  # without a date we can't tell if it is current
        age = (now - a.published).total_seconds() / 3600
        if -1 <= age <= max_age_hours:
            kept.append(a)
    return kept


def check_feeds(cfg: dict) -> int:
    """Print a pass/fail line per feed. Returns the number of failing feeds."""
    failures = 0
    for source in cfg["sources"]:
        flag = "" if source.get("enabled", True) else " (disabled in config)"
        for url in source["feeds"]:
            try:
                items = parse_feed(fetch_url(url), source)
                newest = max((a.published for a in items if a.published), default=None)
                stamp = newest.strftime("%Y-%m-%d %H:%M UTC") if newest else "no dates"
                print(f"OK    {source['name']}{flag}: {len(items)} items, newest {stamp}")
            except Exception as exc:
                failures += 1
                print(f"FAIL  {source['name']}{flag}: {url}\n      {exc}")
    return failures
