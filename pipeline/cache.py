"""Tiny JSON cache so unchanged stories never trigger another AI call."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone


def _now():
    return datetime.now(timezone.utc)


class Cache:
    def __init__(self, path: str):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    self.data = json.load(fh)
            except (OSError, ValueError):
                self.data = {}

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, comparison):
        self.data[key] = {"comparison": comparison, "created": _now().isoformat()}

    def age_hours(self, key: str):
        entry = self.data.get(key)
        if not entry:
            return None
        try:
            created = datetime.fromisoformat(entry["created"])
        except (KeyError, ValueError):
            return None
        return (_now() - created).total_seconds() / 3600

    def save(self, keep_days: int = 14):
        cutoff = _now() - timedelta(days=keep_days)
        fresh = {}
        for k, v in self.data.items():
            try:
                if datetime.fromisoformat(v["created"]) >= cutoff:
                    fresh[k] = v
            except (KeyError, ValueError):
                continue
        self.data = fresh
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, ensure_ascii=False)
