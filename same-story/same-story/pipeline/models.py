from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

LEAN_ORDER = ["left", "centre", "right"]
LEAN_LABELS = {"left": "Left-leaning", "centre": "Centre", "right": "Right-leaning"}


@dataclass
class Article:
    outlet_id: str
    outlet: str
    lean: str
    title: str
    summary: str
    url: str
    published: Optional[datetime] = None


@dataclass
class Story:
    id: str
    picks: dict                      # lean -> Article shown in the main columns
    others: list = field(default_factory=list)   # other outlets covering it
    outlet_count: int = 0
    latest: Optional[datetime] = None
    comparison: Optional[dict] = None
