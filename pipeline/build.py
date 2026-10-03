"""Render the static site: index.html, how-it-works.html, style.css and data.json."""
from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

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
        for key, href, label in (("stories", "index.html", "Stories"), ("how", "how-it-works.html", "How it works"))
    )
    contact = ""
    if site.get("contact_url"):
        contact = '<p><a href="%s">Report an error or a correction</a></p>' % esc(site["contact_url"])
    return f"""<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
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
  <div class="cols" style="--cols:{len(leans)}">
{cols}
  </div>
  {analysis}
  {others}
</section>"""


def build_index(stories: list, cfg: dict, meta: dict) -> str:
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
{body_stories}"""
    return page(cfg, f"{cfg['site']['title']}: UK politics headlines compared", body, "stories")


def build_how(cfg: dict, meta: dict, system_prompt: str) -> str:
    s, m, llm = cfg["settings"], cfg["matching"], cfg["llm"]
    rows = []
    for src in cfg["sources"]:
        if not src.get("enabled", True):
            continue
        hosts = ", ".join(sorted({urlparse(u).netloc for u in src["feeds"]}))
        rows.append(f"<tr><td>{esc(src['name'])}</td><td>{esc(LEAN_LABELS.get(src['lean'], src['lean']))}</td><td>{esc(src.get('rating', ''))}</td><td>{esc(hosts)}</td></tr>")
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

<h2>Limits</h2>
<ul>
<li>Choosing which outlets to include is an editorial decision, and so are the left, centre and right labels.</li>
<li>Only outlets with usable public feeds can be included. Some outlets don't offer one.</li>
<li>Matching can occasionally group two different events or miss a pair about the same event.</li>
<li>Outlets publish at different times, so a development can appear in one headline simply because it happened later. Publication times are shown next to each headline.</li>
</ul>
</article>"""
    return page(cfg, f"How it works: {cfg['site']['title']}", body, "how")


def write_site(stories: list, cfg: dict, out_dir: str, meta: dict, system_prompt: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def write(name, text):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write(text)

    write("index.html", build_index(stories, cfg, meta))
    write("how-it-works.html", build_how(cfg, meta, system_prompt))
    with open(os.path.join(HERE, "style.css"), encoding="utf-8") as src:
        write("style.css", src.read())

    payload = {
        "generated": meta["generated"].isoformat(),
        "dropped_as_unrelated": meta.get("dropped", []),
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
