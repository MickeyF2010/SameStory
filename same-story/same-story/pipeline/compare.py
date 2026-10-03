"""Ask the AI to describe framing differences, then check what it says before we show it."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from typing import Optional

from .cache import Cache
from .llm import LLM, LLMError, LLMFatal
from .models import Story

log = logging.getLogger(__name__)

PROMPT_VERSION = "5"
LABELS = "ABCDEFGH"

# Evaluative words the prompt forbids. If the model uses them anyway, we discard the comparison.
BANNED = re.compile(
    r"\b(biased|bias|misleading|propaganda|dishonest|slanted|sensationalist|sensational|spin|spun|manipulative)\b",
    re.IGNORECASE,
)


def story_key(story: Story, model: str) -> str:
    parts = [PROMPT_VERSION, model]
    for lean in sorted(story.picks):
        a = story.picks[lean]
        parts.append(f"{a.outlet_id}|{a.title}|{a.summary}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def build_user_message(story: Story) -> tuple:
    """Anonymise outlets (A, B, C) and shuffle their order, so the model sees neither
    outlet names nor a fixed left-to-right order. Returns (message, {label: Article})."""
    articles = list(story.picks.values())
    random.Random(story.id).shuffle(articles)
    mapping, lines = {}, ["<texts>"]
    for label, a in zip(LABELS, articles):
        mapping[label] = a
        when = a.published.strftime("%Y-%m-%d %H:%M UTC") if a.published else "unknown"
        lines.append(f"[Outlet {label}]\nPublished: {when}\nHeadline: {a.title}\nStandfirst: {a.summary or '(none)'}\n")
    lines.append("</texts>")
    return "\n".join(lines), mapping


def _short(value, limit: int = 240) -> str:
    if not isinstance(value, str):
        return ""
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit].rsplit(" ", 1)[0] + "…" if len(value) > limit else value


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
        except ValueError:
            return None
    return data if isinstance(data, dict) else None


def validate(raw: str, mapping: dict) -> Optional[dict]:
    """Return a clean comparison dict, or None if the model's answer can't be trusted."""
    data = _extract_json(raw)
    if data is None:
        return None
    if data.get("same_story") is False:
        return {"same_story": False}

    names = {label: a.outlet for label, a in mapping.items()}

    def named(text: str) -> str:
        """Swap the anonymous labels for outlet names, now that the AI has finished."""
        return re.sub(r"\b[Oo]utlet ([A-H])\b", lambda m: names.get(m.group(1), m.group(0)), text)

    same_dev = data.get("same_development") is not False
    result = {
        "same_story": True,
        "same_development": same_dev,
        "topic": named(_short(data.get("topic"), 120)),
        "agreed_facts": [] if not same_dev else
            [named(_short(f)) for f in (data.get("agreed_facts") or []) if isinstance(f, str) and f.strip()][:4],
        "main_difference": named(_short(data.get("main_difference"), 400)),
        "outlets": {},
    }
    for item in data.get("outlets") or []:
        if not isinstance(item, dict):
            continue
        match = re.search(r"\b([A-H])\b", str(item.get("outlet_id", "")).upper())
        article = mapping.get(match.group(1)) if match else None
        if article is None:
            continue
        source_text = f"{article.title} {article.summary}".lower()
        wording, seen = [], set()
        for phrase in item.get("wording") or []:
            if not isinstance(phrase, str):
                continue
            phrase = phrase.strip()
            # Keep only words that really appear in the outlet's own text.
            if phrase and phrase.lower() in source_text and phrase.lower() not in seen:
                wording.append(phrase)
                seen.add(phrase.lower())
        result["outlets"][article.outlet_id] = {
            "emphasis": named(_short(item.get("emphasis"))),
            "wording": wording[:3],
        }

    prose = " ".join(
        [result["topic"], result["main_difference"], *result["agreed_facts"]]
        + [o["emphasis"] for o in result["outlets"].values()]
    )
    if BANNED.search(prose):
        log.info("Discarding comparison that used evaluative language")
        return None
    if not result["topic"] or not result["main_difference"] or not result["outlets"]:
        return None
    return result


def compare_stories(stories: list, llm: LLM, system_prompt: str, cache: Cache, llm_cfg: dict,
                    sleep=time.sleep) -> int:
    """Fill story.comparison from cache or the AI. Returns the number of live API calls made."""
    max_calls = llm_cfg.get("max_calls_per_run", 25)
    delay = llm_cfg.get("request_delay_seconds", 6.5)
    retry_after = llm_cfg.get("retry_failed_after_hours", 6)
    calls = failures_in_row = 0

    for story in stories:
        key = story_key(story, llm.model)
        cached = cache.get(key)
        if cached is not None:
            age = cache.age_hours(key)
            if cached.get("comparison") is not None or (age is not None and age < retry_after):
                story.comparison = cached.get("comparison")
                continue
        if calls >= max_calls or failures_in_row >= 3:
            continue

        message, mapping = build_user_message(story)
        if calls:
            sleep(delay)
        calls += 1
        try:
            raw = llm.generate_json(system_prompt, message)
        except LLMFatal as exc:
            log.error("AI setup problem, skipping all AI calls this run: %s", str(exc)[:300])
            log.error("Check the API key and the model name (llm.model in config.json, or LLM_MODEL).")
            break
        except LLMError as exc:
            failures_in_row += 1
            log.warning("AI call failed for story %s: %s", story.id, exc)
            continue
        failures_in_row = 0
        comparison = validate(raw, mapping)
        cache.put(key, comparison)  # a rejected answer is cached too, and retried after a delay
        story.comparison = comparison
    return calls
