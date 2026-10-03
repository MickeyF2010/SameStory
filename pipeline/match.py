"""Group articles about the same event.

Two methods:
  * tfidf       - free, offline, works well when outlets share names and key nouns.
  * embeddings  - uses the Gemini embedding endpoint; better at different wording.
Both produce L2-normalised row vectors so the same clustering code works for each.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

STOPWORDS = set(
    """a about above after again against all also am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have having he
    her here hers him his how i if in into is it its just me more most my no nor not now of off on once only or
    other our out over own same she should so some such than that the their them then there these they this those
    through to too under until up very was we were what when where which while who whom why will with would you your
    says say said new latest live news uk""".split()
)
WORD_RE = re.compile(r"[A-Za-z0-9£$€%][A-Za-z0-9£$€%'’\-]*")


def _stem(token: str) -> str:
    token = token.lower().replace("’", "'")
    if token.endswith("'s"):
        token = token[:-2]
    token = token.strip("'-")
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def tokenize(text: str) -> tuple:
    """Return (tokens, entity_tokens). Capitalised words that are not sentence-initial count as entities."""
    tokens, entities = [], set()
    words = WORD_RE.findall(text)
    for i, word in enumerate(words):
        tok = _stem(word)
        if len(tok) < 2 or tok in STOPWORDS:
            continue
        tokens.append(tok)
        if i > 0 and word[0].isupper():
            entities.add(tok)
    return tokens, entities


def tfidf_vectors(articles: list) -> np.ndarray:
    docs = []
    for a in articles:
        t_tokens, t_ents = tokenize(a.title)
        s_tokens, s_ents = tokenize(a.summary)
        counts = Counter()
        for tok in t_tokens:
            counts[tok] += 2  # headline words count double
        for tok in s_tokens:
            counts[tok] += 1
        docs.append((counts, t_ents | s_ents))

    n = len(docs)
    df = Counter()
    for counts, _ in docs:
        df.update(counts.keys())
    vocab = {tok: i for i, tok in enumerate(sorted(df))}
    matrix = np.zeros((n, max(len(vocab), 1)), dtype=np.float32)
    for row, (counts, ents) in enumerate(docs):
        for tok, tf in counts.items():
            idf = math.log((1 + n) / (1 + df[tok])) + 1.0
            boost = 1.6 if tok in ents else 1.0
            matrix[row, vocab[tok]] = (1 + math.log(tf)) * idf * boost
    return normalise(matrix)


def normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cluster(vectors: np.ndarray, threshold: float) -> list:
    """Single pass: each item joins the closest existing group if similar enough, else starts one.

    Compares to the group's running average (centroid), which stops long chains of
    loosely related headlines merging into one giant group.
    """
    n = len(vectors)
    if n == 0:
        return []
    dim = vectors.shape[1]
    sums = np.zeros((n, dim), dtype=np.float32)
    unit = np.zeros((n, dim), dtype=np.float32)
    groups = []
    for i in range(n):
        v = vectors[i]
        if groups:
            sims = unit[: len(groups)] @ v
            best = int(np.argmax(sims))
            if sims[best] >= threshold:
                groups[best].append(i)
                sums[best] += v
                norm = np.linalg.norm(sums[best])
                unit[best] = sums[best] / (norm or 1.0)
                continue
        g = len(groups)
        groups.append([i])
        sums[g] = v
        unit[g] = v
    return groups
