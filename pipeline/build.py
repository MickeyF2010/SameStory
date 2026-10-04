"""Render the static site: index.html, how-it-works.html, the archive pages, style.css and data.json."""
from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

from . import words as words_mod
from .archive import parse_time
from .models import LEAN_LABELS, LEAN_ORDER, Story

try:
    from zoneinfo import ZoneInfo
    LONDON = ZoneInfo("Europe/London")
except Exception:  # tzdata missing: fall back to UTC
    LONDON = timezone.utc

HERE = os.path.dirname(os.path.abspath(__file__))


def esc(value) -> str:
    return html.escape(str(value or ""), quote=True)


def highlight(text: str, phrases: list) -> str:
    """Escape text and wrap each verified phrase in <mark>. Works on the raw text so
    phrases can never break the markup."""
    low, spans = text.lower(), []
    for phrase in sorted({p.strip() for p in phrases if p and p.strip()}, key=len, reverse=True):
        needle, start = phrase.lower(), 0
        while True:
            i = low.find(needle, start)
            if i < 0:
                break
            j = i + len(needle)
            if not any(i < e and j > s for s, e in spans):
                spans.append((i, j))
            start = j
    spans.sort()
    out, pos = [], 0
    for s, e in spans:
        out.append(esc(text[pos:s]))
        out.append(f"<mark>{esc(text[s:e])}</mark>")
        pos = e
    out.append(esc(text[pos:]))
    return "".join(out)


def fmt_dt(dt: Optional[datetime], with_date: bool = True) -> str:
    if not dt:
        return ""
    local = dt.astimezone(LONDON)
    stamp = f"{local:%H:%M}"
    return f"{local.day} {local:%b}, {stamp}" if with_date else stamp


def page(cfg: dict, title: str, body: str, active: str) -> str:
    site = cfg["site"]
    current = ' aria-current="page"'
    nav = "".join(
        '<a href="%s"%s>%s</a>' % (href, current if key == active else "", label)
        for key, href, label in (("stories", "index.html", "Stories"), ("words", "words.html", "Words"),
                                 ("archive", "archive.html", "Archive"),
                                 ("how", "how-it-works.html", "How it works"))
    )
    contact = ""
    if site.get("contact_url"):
        contact = '<p><a href="%s">Report an error or a correction</a></p>' % esc(site["contact_url"])
    return f"""<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'">
<meta name="referrer" content="strict-origin-when-cross-origin">
<title>{esc(title)}</title>
<meta name="description" content="{esc(site['tagline'])}">
<link rel="stylesheet" href="style.css">
</head>
<body>
<header class="masthead">
  <div class="wrap masthead-inner">
    <div>
      <p class="wordmark"><a href="index.html">{esc(site['title'])}</a></p>
      <p class="tagline">{esc(site['tagline'])}</p>
    </div>
    <nav aria-label="Main">{nav}</nav>
  </div>
</header>
<main class="wrap">
{body}
</main>
<footer class="wrap site-footer">
  <p>Headlines and short standfirsts belong to the outlets named and link to the original articles. Descriptions marked as AI are generated automatically and can be wrong.</p>
  {contact}
</footer>
</body>
</html>
"""


def _spectrum(lean: str) -> str:
    segs = "".join(f'<span class="{"on" if k == lean else ""}"></span>' for k in LEAN_ORDER)
    return f'<span class="spectrum" aria-hidden="true">{segs}</span>'


def _column(story: Story, lean: str, comparison: Optional[dict]) -> str:
    a = story.picks[lean]
    info = (comparison or {}).get("outlets", {}).get(a.outlet_id, {})
    words = info.get("wording", [])
    when = fmt_dt(a.published)
    parts = [
        '<article class="col">',
        f'<div class="lean">{_spectrum(lean)}<span class="lean-label">{esc(LEAN_LABELS[lean])}</span></div>',
        f'<p class="outlet">{esc(a.outlet)}</p>',
        f'<h3 class="headline"><a href="{esc(a.url)}" rel="noopener noreferrer">{highlight(a.title, words)}</a></h3>',
    ]
    if a.summary:
        parts.append(f'<p class="standfirst">{highlight(a.summary, words)}</p>')
    if info.get("emphasis"):
        parts.append(f'<p class="note"><strong>Puts first:</strong> {esc(info["emphasis"])}</p>')
    if when:
        parts.append(f'<p class="when"><time datetime="{a.published.isoformat()}">Published {when}</time></p>')
    parts.append("</article>")
    return "\n".join(parts)


def story_html(story: Story) -> str:
    cmp_ = story.comparison if story.comparison and story.comparison.get("same_story", True) else None
    topic = cmp_["topic"] if cmp_ else "Story covered by several outlets"
    leans = [l for l in LEAN_ORDER if l in story.picks]
    cols = "\n".join(_column(story, l, cmp_) for l in leans)

    if cmp_:
        facts = "".join(f"<li>{esc(f)}</li>" for f in cmp_["agreed_facts"])
        facts_block = f'<div><h4>Where the texts agree</h4><ul>{facts}</ul></div>' if facts else ""
        heading = "Main difference in framing" if cmp_.get("same_development", True) else "Different angles on the same topic"
        analysis = (
            '<section class="analysis" aria-label="AI description of the framing">'
            '<p class="ai-flag">Described by AI from the headlines and standfirsts above only. Check it against the originals.</p>'
            '<div class="analysis-grid">'
            f"{facts_block}"
            f'<div><h4>{heading}</h4><p>{esc(cmp_["main_difference"])}</p></div>'
            "</div></section>"
        )
    else:
        analysis = '<p class="pending">No framing description yet for this story. The headlines above are the originals.</p>'

    others = ""
    if story.others:
        links = ", ".join(f'<a href="{esc(a.url)}" rel="noopener noreferrer">{esc(a.outlet)}</a>' for a in story.others)
        others = f'<p class="others">Also covered by {links}.</p>'

    updated = fmt_dt(story.latest)
    latest = "<span>Latest coverage %s</span>" % esc(updated) if updated else ""
    return f"""<section class="story" id="s-{esc(story.id)}">
  <div class="story-head">
    <h2>{esc(topic)}</h2>
    <p class="meta"><span>{story.outlet_count} outlets covering it</span>{latest}</p>
  </div>
  <div class="cols cols-{len(leans)}">
{cols}
  </div>
  {analysis}
  {others}
</section>"""




def _spot_window(spot) -> str:
    first, last = fmt_dt(spot.first_checked), fmt_dt(spot.last_checked)
    return f"checked in {spot.runs} runs between {first} and {last}"


def blindspot_html(spot, demo: bool = False) -> str:
    """A story one side of the feeds covered and the other did not. Everything here is computed by
    code from the feeds: no AI is involved, and the wording never claims an outlet ignored anything."""
    side_label = LEAN_LABELS[spot.side].lower()
    missing_label = LEAN_LABELS[spot.missing].lower()
    checked = ", ".join(esc(n) for n in spot.checked_sources) or "the outlets in that column"
    rows = []
    for a in spot.articles:
        when = fmt_dt(a.published)
        rows.append(
            '<li class="spot-item">'
            f'<span class="arc-lean">{esc(a.outlet)} · {esc(LEAN_LABELS[a.lean])}</span>'
            f'<a class="arc-headline" href="{esc(a.url)}" rel="noopener noreferrer">{esc(a.title)}</a>'
            + (f'<span class="arc-when">Published {esc(when)}</span>' if when else "")
            + "</li>"
        )
    return f"""<li class="spot" id="b-{esc(spot.id)}">
  <p class="spot-flag">Covered by {spot.outlet_count} {esc(side_label)} outlets. Not found in our sources from the {esc(missing_label)} outlets ({checked}).</p>
  <ul class="spot-list">{"".join(rows)}</ul>
  <p class="spot-check">{esc(_spot_window(spot)).capitalize()}. The closest headline from the {esc(missing_label)} column scored {spot.max_similarity:.2f} similarity, below the limit that would have made it a match, and every feed in that column loaded with items. Matching is imperfect: a feed that publishes late, or a headline worded very differently, can produce a false result.</p>
</li>"""


def build_blindspots(spots: list, cfg: dict, demo: bool = False) -> str:
    if not spots:
        return ""
    st = cfg.get("blindspots") or {}
    intro = (f'<p class="spot-intro">Stories below were found in at least two outlets of one column and in none of '
             f'the other, in every feed checked. This is worked out by the program from the feeds alone, never by the AI. '
             f'"Not found in our sources" means exactly that. It is not a claim about what any outlet '
             f'chose to do. A story has to stay missing for {esc(st.get("min_runs", 2))} checks over at least '
             f'{esc(st.get("min_hours", 6))} hours before it appears here, and it is dropped the moment any headline on the '
             f'other side looks related. <a href="how-it-works.html">How this is worked out</a>.</p>')
    items = "".join(blindspot_html(b, demo) for b in spots)
    return f"""<section class="spots" aria-label="Covered on one side only">
  <h2>Covered on one side only</h2>
  {intro}
  <ol class="spot-list-all">{items}</ol>
</section>"""


def build_index(stories: list, cfg: dict, meta: dict, spots=None) -> str:
    updated = meta["generated"].astimezone(LONDON)
    demo = '<p class="demo-banner" role="note">Demo data. These outlets and stories are invented to show the layout.</p>' if meta.get("demo") else ""
    legend = """<aside class="legend" aria-label="How to read each story">
  <p><span class="lg-serif">Serif type</span> is what the outlet published.</p>
  <p><span class="lg-sans">Sans-serif type</span> is the site describing it.</p>
  <p><mark>Highlighted words</mark> carry an evaluative or emotional tone and are checked against the outlet's own text.</p>
</aside>"""
    if stories:
        body_stories = "\n".join(story_html(s) for s in stories)
    else:
        body_stories = """<section class="empty"><h2>No stories to compare right now</h2>
<p>A story appears here when at least three outlets, including a left-leaning and a right-leaning one, cover it. Check back after the next update.</p></section>"""
    body = f"""{demo}
<div class="intro">
  <p class="updated">Updated <time datetime="{meta['generated'].isoformat()}">{updated.day} {updated:%b %Y, %H:%M}</time>. Stories are listed by how many outlets cover them, then by recency, never by engagement.</p>
  {legend}
</div>
{body_stories}
{build_blindspots(spots or [], cfg, bool(meta.get("demo")))}"""
    return page(cfg, f"{cfg['site']['title']}: UK politics headlines compared", body, "stories")


def build_how(cfg: dict, meta: dict, system_prompt: str) -> str:
    s, m, llm = cfg["settings"], cfg["matching"], cfg["llm"]
    arc_days = int(float(cfg.get("archive", {}).get("ai_text_days", 14)))
    rows = []
    for src in cfg["sources"]:
        if not src.get("enabled", True):
            continue
        hosts = ", ".join(sorted({urlparse(u).netloc for u in src["feeds"]}))
        note = src.get("_feed_note", "")
        note_html = f'<p class="feed-note">{esc(note)}</p>' if note else ""
        rows.append(f"<tr><td>{esc(src['name'])}{note_html}</td><td>{esc(LEAN_LABELS.get(src['lean'], src['lean']))}</td><td>{esc(src.get('rating', ''))}</td><td>{esc(hosts)}</td></tr>")
    bs = cfg.get("blindspots") or {}
    bs_min = bs.get("min_outlets", 2)
    bs_runs = bs.get("min_runs", 2)
    bs_hours = bs.get("min_hours", 6)
    method = "text embeddings" if meta.get("match_method") == "embeddings" else "word-overlap (TF-IDF) similarity"
    threshold = m["embedding_threshold"] if meta.get("match_method") == "embeddings" else m["tfidf_threshold"]
    needs = [LEAN_LABELS.get(l, l).lower() for l in s.get("required_leans", [])]
    if needs:
        rule = "A group becomes a story only if at least %d different outlets cover it and it includes a %s outlet." % (s["min_outlets"], " and a ".join(needs))
    else:
        rule = "A group becomes a story only if at least %d different outlets cover it." % s["min_outlets"]
    body = f"""<article class="prose">
<h1>How this site works</h1>
<p>No news product is free of judgement, and this one doesn't claim to be. Instead it shows its working so you can check it.</p>

<h2>Where headlines come from</h2>
<p>Every few hours a program reads the public RSS feeds of the outlets below. It keeps only each outlet's headline, a short standfirst and the link. Each headline links back to the original article.</p>
<div class="table-scroll"><table>
<thead><tr><th scope="col">Outlet</th><th scope="col">Column</th><th scope="col">Published rating</th><th scope="col">Feed host</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p>{esc(cfg['site']['lean_rating_note'])}</p>

<h2>How stories are matched</h2>
<p>Headlines from the last {s['max_age_hours']} hours are grouped by {method}, with a similarity threshold of {threshold}. {rule} These rules are applied the same way to every story. The AI is also asked to say if a group is really about different events, and those groups are dropped.</p>

<h2>Which headline is shown for each outlet</h2>
<p>Several outlets can sit in the same column, and some cover a story more than once. For each column the site shows one headline, chosen so that the headlines across the columns are as similar in wording to each other as possible. That keeps the comparison like for like. It is still a choice, and it is made by wording similarity alone, never by how critical or favourable a headline is. The other outlets are listed under "Also covered by", and every headline links to the original.</p>

<h2>How stories are ordered</h2>
<p>By the number of outlets covering the story, then by how recent it is. Clicks and engagement play no part. At most {s['max_stories']} stories are shown.</p>

<h2>What the AI does</h2>
<p>For each story the model ({esc(llm['model'])}) is given only the headlines and standfirsts and asked to describe how they frame the story. To limit its own bias:</p>
<ul>
<li>Outlets are anonymised as A, B and C and shuffled, so the model sees no outlet names, no political labels and no fixed left-to-right order.</li>
<li>It may only describe, never judge. If its answer uses evaluative words such as "biased" or "misleading", the answer is thrown away.</li>
<li>Any word it flags as emotionally loaded must appear exactly in that outlet's own text, or it is removed.</li>
<li>It is not asked what an outlet "leaves out", because a headline and a few lines cannot show that.</li>
<li>If the texts turn out to cover different angles or stages of a story, the page says so instead of calling it a difference in framing.</li>
<li>Results are cached, so an unchanged story is not re-analysed and does not change from hour to hour.</li>
</ul>
<p>The model sees only a headline and a few lines, never the full article, so it can describe what a headline emphasises but not what an article says. It can still make mistakes.</p>

<h2>The exact instructions given to the AI (version {esc(meta.get('prompt_version', ''))})</h2>
<pre>{esc(system_prompt)}</pre>

<h2>The archive</h2>
<p>Every story that reaches the front page is added to the <a href="archive.html">archive</a>, listed by the month it first appeared. What is kept, and for how long:</p>
<ul>
<li>Each outlet's headline, its name, the link and the publication time are kept, for every outlet that covered the story, including headlines that were later replaced on the front page.</li>
<li>Standfirsts and the AI description are deleted {esc(arc_days)} days after a story was last on the front page. After that only the headlines remain.</li>
<li>If any headline or standfirst in a story mentions court proceedings (words such as "charged", "trial" or "jury"), the AI description and standfirsts are never stored at all, so nothing generated automatically stays online about a case that may still be live.</li>
<li>Removed stories are deleted from the archive completely. To ask for a removal or correction, use the link at the bottom of every page.</li>
</ul>

<h2>Covered on one side only ("blindspots")</h2>
<p>Some stories are picked up by one column of outlets and not the other. Those are listed in a separate section, and this is
worked out by the program from the feeds alone: <strong>the AI is not used at all</strong> for it, so no model ever forms a view
about what an outlet did or did not cover. A story is listed only when all of these are true:</p>
<ul>
<li>at least {bs_min} different outlets in one column covered it, and no outlet at all in the other column did;</li>
<li>every feed in the other column loaded and returned items in that same run, so a feed that was down, empty or stale can never look like missing coverage;</li>
<li>no headline in the other column, anywhere in the {s['max_age_hours']}h window, was even loosely similar to any headline in the story (below the match threshold);</li>
<li>it has stayed that way for {bs_runs} checks spread over at least {bs_hours} hours. Any run in which the other column turns out to cover it removes it again.</li>
</ul>
<p>The wording is always "not found in our sources", never "ignored" or "did not report". It means we looked, in every feed
listed in the table above, and did not find it. It is not a statement about any outlet's intent, and matching is imperfect: a
story published late, or headlined very differently, can be missed, which can produce a false result. The section shows which
outlets were checked and the window that was checked.</p>

<h2>Words</h2>
<p>The <a href="words.html">Words</a> page counts how often each column's headlines use one term or another, for word pairs
fixed in advance and published in the site's configuration. It is counting only: no AI is involved and the page makes no
judgement about which wording is better, fairer or more emotive. Figures are rates per column, not raw counts, because the
columns publish different numbers of headlines; only stories that <strong>both</strong> a left-leaning and a right-leaning outlet
covered are counted, so both columns are measured on the same stories; each article link is counted once; matching is on whole
words including simple forms such as plurals. Only headlines are counted, never the standfirsts, because standfirsts are deleted
from the archive after {arc_days} days and comparing a new period with an old one would then be misleading.</p>

<h2>Limits</h2>
<ul>
<li>Choosing which outlets to include is an editorial decision, and so are the left, centre and right labels.</li>
<li>The archive starts on the day it was switched on. Earlier stories are not in it, and a run that fails adds nothing.</li>
<li>Only outlets with usable public feeds can be included. Some outlets don't offer one.</li>
<li>Matching can occasionally group two different events or miss a pair about the same event.</li>
<li>Outlets publish at different times, so a development can appear in one headline simply because it happened later. Publication times are shown next to each headline.</li>
<li>The "covered on one side only" list depends on the outlets listed above. An outlet that was left out, or whose feed was down, is not being accused of anything: it is simply not part of what was checked.</li>
<li>The Words counts come from the archive, which begins on the day it was switched on, so early figures build up over time and a rare word may need several weeks before a pair is shown for both columns.</li>
</ul>
</article>"""
    return page(cfg, f"How it works: {cfg['site']['title']}", body, "how")


# ---- archive pages ----------------------------------------------------------

def month_file(key: str) -> str:
    return f"archive-{key}.html"


def _month_label(key: str) -> str:
    return datetime.strptime(key + "-01", "%Y-%m-%d").strftime("%B %Y")


def _month_key(entry: dict) -> Optional[str]:
    first = parse_time(entry.get("first_seen"))
    return first.astimezone(LONDON).strftime("%Y-%m") if first else None


def archive_entry_html(entry: dict) -> str:
    first = parse_time(entry.get("first_seen"))
    cmp_ = entry.get("comparison") if isinstance(entry.get("comparison"), dict) else None
    count = int(entry.get("outlet_count") or 0)
    meta = []
    if first:
        meta.append(f'<time datetime="{first.isoformat()}">First on the front page {fmt_dt(first)}</time>')
    if count:
        meta.append(f"<span>{count} outlets covering it</span>")
    parts = [f'<li class="arc-item" id="a-{esc(entry.get("id"))}">',
             f'<p class="arc-meta">{"".join(meta)}</p>']
    if cmp_ and cmp_.get("topic"):
        parts.append(f'<h3 class="arc-topic">{esc(cmp_["topic"])}</h3>')
    arts = entry.get("articles") or {}
    rows = []
    for lean in LEAN_ORDER:
        a = arts.get(lean)
        if not a:
            continue
        when = fmt_dt(parse_time(a.get("published")))
        rows.append(
            '<li>'
            f'<span class="arc-lean">{esc(LEAN_LABELS[lean])} · {esc(a.get("outlet"))}</span>'
            f'<a class="arc-headline" href="{esc(a.get("url"))}" rel="noopener noreferrer">{esc(a.get("title"))}</a>'
            + (f'<span class="arc-standfirst">{esc(a["summary"])}</span>' if a.get("summary") else "")
            + (f'<span class="arc-when">Published {esc(when)}</span>' if when else "")
            + '</li>')
    parts.append(f'<ul class="arc-heads">{"".join(rows)}</ul>')
    if cmp_ and cmp_.get("main_difference"):
        parts.append('<p class="arc-ai"><strong>AI description, deleted after a fortnight:</strong> '
                     f'{esc(cmp_["main_difference"])}</p>')
    others = entry.get("others") or []
    if others:
        links = ", ".join(f'<a href="{esc(o.get("url"))}" rel="noopener noreferrer">{esc(o.get("outlet"))}</a>' for o in others)
        parts.append(f'<p class="others">Also covered by {links}.</p>')
    parts.append("</li>")
    return "".join(parts)


def build_archive_pages(entries: list, cfg: dict) -> dict:
    """Return {filename: html} for archive.html and one page per month. No scripts:
    every page is plain HTML, so the strict CSP stays unchanged."""
    title = cfg["site"]["title"]
    days = int(float(cfg.get("archive", {}).get("ai_text_days", 14)))
    months = {}
    for e in entries:   # entries arrive newest first
        key = _month_key(e)
        if key:
            months.setdefault(key, []).append(e)
    keys = sorted(months, reverse=True)

    intro = (f'<p class="arc-intro">Every story that has reached the front page, by the month it first appeared. '
             f'Headlines and links are kept. Standfirsts and AI descriptions are deleted {days} days after a story '
             f'leaves the front page, and are never kept for stories that mention court proceedings. '
             f'<a href="how-it-works.html">More about the archive</a>.</p>')
    pages = {}
    if keys:
        items = "".join(f'<li><a href="{month_file(k)}">{esc(_month_label(k))}</a> '
                        f'<span class="arc-count">{len(months[k])} {"story" if len(months[k]) == 1 else "stories"}</span></li>'
                        for k in keys)
        listing = f'<ul class="arc-months">{items}</ul>'
    else:
        listing = '<p class="empty">Nothing has been archived yet. Stories are added after each update.</p>'
    pages["archive.html"] = page(cfg, f"Archive: {title}",
                                 f'<div class="arc"><h1>Archive</h1>{intro}{listing}</div>', "archive")

    for i, k in enumerate(keys):
        nav = []
        if i + 1 < len(keys):
            nav.append(f'<a href="{month_file(keys[i + 1])}">Earlier: {esc(_month_label(keys[i + 1]))}</a>')
        nav.append('<a href="archive.html">All months</a>')
        if i > 0:
            nav.append(f'<a href="{month_file(keys[i - 1])}">Later: {esc(_month_label(keys[i - 1]))}</a>')
        body = (f'<div class="arc"><h1>{esc(_month_label(k))}</h1>{intro}'
                f'<ol class="arc-list">{"".join(archive_entry_html(e) for e in months[k])}</ol>'
                f'<nav class="arc-nav" aria-label="Archive months">{"".join(nav)}</nav></div>')
        pages[month_file(k)] = page(cfg, f"{_month_label(k)}: {title} archive", body, "archive")
    return pages




# ---- words page (counting only, no AI) --------------------------------------

CHART_W, CHART_H = 720, 96
BAR_H, BAR_GAP = 16, 6
LABEL_W = 180
RATE_W = 96


def words_chart_html(pair: dict, min_headlines: int) -> str:
    """A two-bar comparison per column, drawn with CSS-width bars (no inline style attribute is
    used: the width is set with a small set of fixed classes)."""
    lines = []
    for lean in ("left", "right"):
        v = pair["by_lean"].get(lean) or {}
        total = int(v.get("total") or 0)
        if total < min_headlines:
            lines.append(f'<p class="words-thin">Not enough headlines in the {esc(LEAN_LABELS[lean].lower())} '
                         f'column yet ({total} so far, {min_headlines} needed).</p>')
            continue
        rows = []
        for key, term in (("a", pair["a"]), ("b", pair["b"])):
            count = int(v.get(key) or 0)
            pct = 100.0 * count / total
            bucket = int(min(round(pct / 5.0), 20))   # 0-20, maps to a fixed CSS class
            rows.append(
                f'<div class="ch-row"><span class="ch-label">{esc(term)}</span>'
                f'<span class="ch-track"><span class="ch-fill w-{bucket}"></span></span>'
                f'<span class="ch-value">{pct:.1f}% ({count} of {total})</span></div>'
            )
        lines.append(f'<div class="ch-col"><h4 class="ch-col-head">{esc(LEAN_LABELS[lean])}</h4>{"".join(rows)}</div>')
    return "".join(lines)


def _archive_line(meta: dict) -> str:
    a = meta.get("archive") or {}
    if not a:
        return ""
    return (f"The archive that these counts come from holds {a.get('stories', 0)} stories and "
            f"{a.get('headlines', 0)} headlines in total, and gains more with every update.")


def build_words(cfg: dict, stats: dict, meta: Optional[dict] = None) -> str:
    st = words_mod.settings(cfg)
    meta = meta or {}
    archive_line = _archive_line(meta)
    totals, pairs = stats["totals"], stats["pairs"]
    left_n, right_n = totals.get("left", 0), totals.get("right", 0)
    if not stats["headline_count"]:
        body = ('<div class="words"><h1>Words</h1>'
                '<p class="words-intro">No archived headlines yet. Counting starts once stories have been '
                'archived, and a pair is only shown when both columns have enough headlines.</p></div>')
        return page(cfg, f"Words: {cfg['site']['title']}", body, "words")

    pairs_html = []
    for pair in pairs:
        pairs_html.append(f"""<section class="words-pair">
  <h3>{esc(pair['a'])} vs {esc(pair['b'])}</h3>
  <div class="ch-cols">{words_chart_html(pair, stats['min_headlines'])}</div>
</section>""")

    body = f"""<div class="words">
<h1>Words</h1>
<p class="words-intro">How often each column's headlines use one term or the other. This is counting, not commentary:
no AI is involved, and the page does not say which wording is better, more accurate or more emotive. The pairs below are
chosen and published in advance.</p>
<div class="words-rules">
<p>Counted over the last {esc(st['days'])} days, and only over archived stories that <strong>both</strong> a left-leaning and a
right-leaning outlet covered, so both columns are being measured on the same stories. Each article link counts once.
Whole words only, including simple word forms ("migrant" also matches "migrants"), with a hyphen or a space
("asylum-seeker" and "asylum seeker" are counted together). Figures are the share of that column's own headlines that used the
term, so a column that publishes more is not flattered by the count.</p>
<p>Headlines counted: {stats['headline_count']} in total, {left_n} from left-leaning outlets and {right_n} from right-leaning
ones. {archive_line} A pair is only shown for a column once it has at least {esc(stats['min_headlines'])} headlines. Words chosen here, and the
method, are published on <a href="how-it-works.html">How it works</a>.</p>
</div>
{''.join(pairs_html)}
</div>"""
    return page(cfg, f"Words: {cfg['site']['title']}", body, "words")


def write_site(stories: list, cfg: dict, out_dir: str, meta: dict, system_prompt: str,
               archive_entries: Optional[list] = None, blindspots: Optional[list] = None,
               word_stats: Optional[dict] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def write(name, text):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write(text)

    write("index.html", build_index(stories, cfg, meta, blindspots))
    write("how-it-works.html", build_how(cfg, meta, system_prompt))
    if word_stats:
        write("words.html", build_words(cfg, word_stats, meta))
    for name, text in build_archive_pages(archive_entries or [], cfg).items():
        write(name, text)
    with open(os.path.join(HERE, "style.css"), encoding="utf-8") as src:
        write("style.css", src.read())

    payload = {
        "generated": meta["generated"].isoformat(),
        "dropped_as_unrelated": meta.get("dropped", []),
        "blindspots": meta.get("blindspots", []),
        "archive": meta.get("archive", {}),
        "stories": [
            {
                "id": s.id,
                "outlet_count": s.outlet_count,
                "comparison": s.comparison,
                "articles": {
                    lean: {"outlet": a.outlet, "title": a.title, "url": a.url,
                           "published": a.published.isoformat() if a.published else None}
                    for lean, a in s.picks.items()
                },
            }
            for s in stories
        ],
    }
    write("data.json", json.dumps(payload, ensure_ascii=False, indent=2))
