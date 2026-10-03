"""Command line entry point.

    python -m pipeline.run                 # full run (needs GEMINI_API_KEY or LLM_API_KEY)
    python -m pipeline.run --no-ai         # headlines only, no AI calls
    python -m pipeline.run --demo          # invented demo data, no network at all
    python -m pipeline.run --check-feeds   # test every feed URL
    python -m pipeline.run --debug-clusters  # print the story groups, to tune thresholds
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

from . import blindspots, build, compare, fetch, match, sample, stories as stories_mod, words as words_mod
from .archive import Archive, ArchiveError, is_removed, removed_keys
from .cache import Cache
from .llm import LLM, LLMError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log = logging.getLogger("samestory")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def make_llm(cfg: dict):
    key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    llm_cfg = cfg["llm"]
    return LLM(
        provider=os.environ.get("LLM_PROVIDER", llm_cfg["provider"]),
        model=os.environ.get("LLM_MODEL", llm_cfg["model"]),
        api_key=key,
        base_url=os.environ.get("LLM_BASE_URL", llm_cfg.get("base_url")),
    )


def vectorise(articles: list, cfg: dict, llm):
    """Return (vectors, threshold, method_used)."""
    m = cfg["matching"]
    if m.get("method") == "embeddings" and llm is not None and llm.can_embed:
        try:
            texts = [f"{a.title}. {a.summary}" for a in articles]
            return match.normalise(llm.embed(texts)), m["embedding_threshold"], "embeddings"
        except LLMError as exc:
            log.warning("Embeddings failed (%s); falling back to TF-IDF matching", exc)
    return match.tfidf_vectors(articles), m["tfidf_threshold"], "tfidf"


def print_groups(articles: list, groups: list) -> None:
    multi = [g for g in groups if len({articles[i].outlet_id for i in g}) > 1]
    print(f"{len(groups)} groups, {len(multi)} with more than one outlet\n")
    for g in sorted(multi, key=lambda g: -len({articles[i].outlet_id for i in g})):
        print(f"--- {len({articles[i].outlet_id for i in g})} outlets ---")
        for i in g:
            a = articles[i]
            print(f"  [{a.lean:6}] {a.outlet}: {a.title}")
        print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    parser.add_argument("--out", default=os.path.join(ROOT, "site_out"))
    parser.add_argument("--cache", default=os.path.join(ROOT, "data", "cache.json"))
    parser.add_argument("--archive", default=os.path.join(ROOT, "data", "archive.json"),
                        help="archive file to read and update (stored on the 'archive' branch by the workflow)")
    parser.add_argument("--demo", action="store_true", help="build the site from invented demo data")
    parser.add_argument("--no-ai", action="store_true", help="skip every AI call")
    parser.add_argument("--check-feeds", action="store_true", help="test each feed URL and exit")
    parser.add_argument("--debug-clusters", action="store_true", help="print story groups and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    with open(os.path.join(ROOT, "prompts", "system.txt"), encoding="utf-8") as fh:
        system_prompt = fh.read()
    now = datetime.now(timezone.utc)
    meta = {"generated": now, "prompt_version": compare.PROMPT_VERSION, "match_method": "tfidf"}

    if args.check_feeds:
        failures = fetch.check_feeds(cfg)
        print(f"\n{failures} feed(s) failed")
        return 1 if failures else 0

    if args.demo:
        cfg = sample.demo_config(cfg)
        meta["demo"] = True
        demo, demo_spots = sample.demo_stories(now)
        preview = Archive(None, cfg.get("archive"))   # in memory only: the demo never touches the real archive
        preview.record(demo, now)
        words_stats = words_mod.compute(preview.sorted_entries(), cfg, now)
        build.write_site(demo, cfg, args.out, meta, system_prompt, preview.sorted_entries(),
                         blindspots=demo_spots, word_stats=words_stats)
        print(f"Demo site written to {args.out}/index.html")
        return 0

    articles, source_status = fetch.fetch_all_with_status(cfg)
    if not articles:
        log.error("No articles could be fetched from any feed. Not publishing an empty site.")
        return 1
    articles = fetch.filter_recent(articles, cfg["settings"]["max_age_hours"], now)
    log.info("%d recent articles", len(articles))

    llm = None if args.no_ai else make_llm(cfg)
    if llm is None and not args.no_ai:
        log.warning("No GEMINI_API_KEY or LLM_API_KEY set: building without AI descriptions.")

    vectors, threshold, method = vectorise(articles, cfg, llm) if articles else (None, 0, "tfidf")
    meta["match_method"] = method
    groups = match.cluster(vectors, threshold) if articles else []
    if args.debug_clusters:
        print_groups(articles, groups)
        return 0

    try:
        archive = Archive(args.archive, cfg.get("archive"))
    except ArchiveError as exc:
        log.error("%s", exc)
        return 1

    story_list = stories_mod.select_stories(articles, groups, vectors, cfg["settings"]) if articles else []
    removed = removed_keys(cfg.get("archive"))
    if removed:
        before = len(story_list)
        story_list = [s for s in story_list if not is_removed(s, removed)]
        if len(story_list) < before:
            log.info("%d story(ies) withheld because they are on the removed list", before - len(story_list))
    log.info("%d stories meet the coverage rules", len(story_list))

    if llm is not None and story_list:
        cache = Cache(args.cache)
        calls = compare.compare_stories(story_list, llm, system_prompt, cache, cfg["llm"])
        cache.save()
        log.info("%d AI call(s) made", calls)
        # Drop groups the AI says are about unrelated subjects, but never silently:
        # they are logged here and listed in data.json so the choice can be audited.
        kept, dropped = [], []
        for s in story_list:
            (dropped if s.comparison and s.comparison.get("same_story") is False else kept).append(s)
        for s in dropped:
            log.warning("Dropped: the AI judged these headlines to be unrelated: %s",
                        " | ".join(f"{a.outlet}: {a.title}" for a in s.picks.values()))
        meta["dropped"] = [
            {"id": s.id, "headlines": [{"outlet": a.outlet, "title": a.title, "url": a.url} for a in s.picks.values()]}
            for s in dropped
        ]
        story_list = kept

    # One-sided coverage, computed by code only (never by the AI). Its watch state is kept in
    # the archive file, so a story has to go uncovered for several runs before it is shown.
    spots = []
    if cfg.get("blindspots", {}).get("enabled", True):
        spots = blindspots.update(archive.blindspot_watch, articles, groups, vectors, source_status,
                                  cfg, now, method, exclude_urls=removed, removed=removed)
        shown_urls = {a.url for s in story_list for a in s.picks.values()}
        shown_urls |= {a.url for s in story_list for a in s.others}
        spots = [b for b in spots if not ({a.url for a in b.articles} & shown_urls)]
        for b in spots:
            log.info("Blindspot: %s-leaning only (%d outlets): %s", b.side, b.outlet_count,
                     " | ".join(a.title for a in b.articles[:2]))
    meta["blindspots"] = [
        {"id": b.id, "side": b.side, "missing": b.missing, "outlet_count": b.outlet_count,
         "runs": b.runs, "first_checked": b.first_checked.isoformat(),
         "last_checked": b.last_checked.isoformat(), "checked_sources": b.checked_sources,
         "max_similarity": b.max_similarity,
         "articles": [{"outlet": a.outlet, "lean": a.lean, "title": a.title, "url": a.url,
                       "published": a.published.isoformat() if a.published else None} for a in b.articles]}
        for b in spots
    ]

    new = archive.record(story_list, now)
    archive.save(now)   # also saves the blindspot watch state kept in the same file
    log.info("Archive: %d new, %d in total", new, len(archive.entries))

    word_stats = words_mod.compute(archive.sorted_entries(), cfg, now)
    build.write_site(story_list, cfg, args.out, meta, system_prompt, archive.sorted_entries(),
                     blindspots=spots, word_stats=word_stats)
    log.info("Site written to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
