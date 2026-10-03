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

from . import build, compare, fetch, match, sample, stories as stories_mod
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
        build.write_site(sample.demo_stories(now), cfg, args.out, meta, system_prompt)
        print(f"Demo site written to {args.out}/index.html")
        return 0

    articles = fetch.fetch_all(cfg)
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

    story_list = stories_mod.select_stories(articles, groups, vectors, cfg["settings"]) if articles else []
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

    build.write_site(story_list, cfg, args.out, meta, system_prompt)
    log.info("Site written to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
