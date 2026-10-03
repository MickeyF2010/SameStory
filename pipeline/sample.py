"""Invented demo data so you can preview the site with no internet and no API key.
Every outlet, person and policy here is fictional."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .models import Article, Story


def demo_config(cfg: dict) -> dict:
    demo = dict(cfg)
    demo["sources"] = [
        {"id": "lantern", "name": "The Daily Lantern (fictional)", "lean": "left", "feeds": ["https://example.com/lantern.xml"]},
        {"id": "wire", "name": "Channel Wire (fictional)", "lean": "centre", "feeds": ["https://example.com/wire.xml"]},
        {"id": "standard", "name": "The Albion Standard (fictional)", "lean": "right", "feeds": ["https://example.com/standard.xml"]},
    ]
    return demo


def demo_stories(now: datetime = None) -> list:
    now = now or datetime.now(timezone.utc)

    def art(oid, name, lean, title, summary, mins):
        return Article(oid, name, lean, title, summary, f"https://example.com/{oid}/{abs(hash(title)) % 9999}", now - timedelta(minutes=mins))

    s1 = Story(
        id="demo000001",
        picks={
            "left": art("lantern", "The Daily Lantern (fictional)", "left",
                        "Ministers push ahead with rail-freight levy to cut lorry pollution",
                        "The new charge on large retailers will fund cleaner rail links, the transport secretary said.", 95),
            "centre": art("wire", "Channel Wire (fictional)", "centre",
                          "Government confirms £2bn levy on large retailers' road deliveries",
                          "Transport secretary says the money will be spent on rail freight; retail groups say prices could rise.", 80),
            "right": art("standard", "The Albion Standard (fictional)", "right",
                         "Shoppers face higher bills as ministers hit supermarkets with £2bn delivery tax",
                         "Retailers warn the new tax will be passed on to customers, as the transport secretary defends the plan.", 70),
        },
        others=[art("morning", "Morning Ledger (fictional)", "centre", "x", "", 60)],
        outlet_count=4, latest=now - timedelta(minutes=60),
        comparison={
            "same_story": True,
            "topic": "Government announces a £2bn levy on large retailers' road deliveries",
            "same_development": True,
            "agreed_facts": [
                "The levy is £2bn and applies to large retailers.",
                "The transport secretary announced it and says the money will go to rail freight.",
            ],
            "main_difference": "One text opens with the environmental purpose, one with the cost to retailers and shoppers, and the third reports both the purpose and the price warning.",
            "outlets": {
                "lantern": {"emphasis": "The aim of cutting lorry pollution.", "wording": ["push ahead", "cleaner"]},
                "wire": {"emphasis": "The size of the levy and who pays it.", "wording": []},
                "standard": {"emphasis": "The likely effect on shoppers' bills.", "wording": ["hit supermarkets", "delivery tax", "higher bills"]},
            },
        },
    )
    s2 = Story(
        id="demo000002",
        picks={
            "left": art("lantern", "The Daily Lantern (fictional)", "left",
                        "Councils warn of cuts as funding review lands", "Local leaders say the review leaves libraries and bus routes at risk.", 200),
            "centre": art("wire", "Channel Wire (fictional)", "centre",
                          "Local government funding review published", "The review proposes a new formula for allocating grants to councils.", 190),
            "right": art("standard", "The Albion Standard (fictional)", "right",
                         "Funding review gives councils a chance to cut waste, say ministers", "The government says the new formula rewards efficient authorities.", 185),
        },
        outlet_count=3, latest=now - timedelta(minutes=185),
        comparison=None,
    )
    return [s1, s2]
