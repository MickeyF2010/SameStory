"""Turn groups of articles into displayable stories, using simple published rules."""
from __future__ import annotations

import hashlib
import itertools
from datetime import datetime, timezone

import numpy as np

from .models import LEAN_ORDER, Story


def best_combination(candidates: dict, vectors: np.ndarray, centroid: np.ndarray) -> dict:
    """Choose one article per column so the chosen headlines are as similar to each other as
    possible (we want to compare like with like). Ties go to the headline nearest the story's core."""
    leans = [l for l in LEAN_ORDER if candidates.get(l)]
    pools = []
    for lean in leans:
        ranked = sorted(candidates[lean], key=lambda i: -float(vectors[i] @ centroid))
        pools.append(ranked[:8])  # keeps the search tiny
    best, best_score = None, float("-inf")
    for combo in itertools.product(*pools):
        pair = sum(float(vectors[a] @ vectors[b]) for a, b in itertools.combinations(combo, 2))
        core = sum(float(vectors[i] @ centroid) for i in combo)
        score = pair + 0.1 * core
        if score > best_score:
            best, best_score = combo, score
    return dict(zip(leans, best))


def select_stories(articles: list, groups: list, vectors: np.ndarray, settings: dict) -> list:
    min_outlets = settings.get("min_outlets", 3)
    required = set(settings.get("required_leans", ["left", "right"]))
    stories = []
    for members in groups:
        arts = [articles[i] for i in members]
        outlets = {a.outlet_id for a in arts}
        if len(outlets) < min_outlets:
            continue
        if not required <= {a.lean for a in arts}:
            continue

        centroid = vectors[members].mean(axis=0)
        norm = np.linalg.norm(centroid) or 1.0
        centroid = centroid / norm
        ranked = sorted(members, key=lambda i: -float(vectors[i] @ centroid))

        candidates = {}
        for i in ranked:
            if articles[i].lean in LEAN_ORDER:
                candidates.setdefault(articles[i].lean, []).append(i)
        chosen = best_combination(candidates, vectors, centroid)
        picks = {lean: articles[i] for lean, i in chosen.items()}
        picked_outlets = {a.outlet_id for a in picks.values()}
        others, seen = [], set(picked_outlets)
        for i in ranked:
            a = articles[i]
            if a.outlet_id not in seen:
                others.append(a)
                seen.add(a.outlet_id)

        dates = [a.published for a in arts if a.published]
        key = "|".join(sorted(a.url for a in picks.values()))
        stories.append(
            Story(
                id=hashlib.sha1(key.encode()).hexdigest()[:10],
                picks=picks,
                others=others,
                outlet_count=len(outlets),
                latest=max(dates) if dates else None,
            )
        )

    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    # Ranking rule (published on the methodology page): breadth of coverage, then recency.
    stories.sort(key=lambda s: (-s.outlet_count, -(s.latest or epoch).timestamp()))
    return stories[: settings.get("max_stories", 20)]
