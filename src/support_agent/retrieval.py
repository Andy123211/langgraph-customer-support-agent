"""Dependency-free helpers for lexical retrieval and reciprocal-rank fusion."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, Mapping


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Tokenize English words and Chinese character bigrams for the demo KB."""
    tokens: list[str] = []
    for chunk in _TOKEN_PATTERN.findall(text.lower()):
        if chunk.isascii():
            tokens.append(chunk)
        elif len(chunk) == 1:
            tokens.append(chunk)
        else:
            tokens.extend(chunk[index:index + 2] for index in range(len(chunk) - 1))
    return tokens


def bm25_scores(
    query: str,
    documents: Mapping[str, str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    """Return BM25 scores keyed by document id; documents with no match score 0."""
    query_terms = tokenize(query)
    tokenized = {doc_id: tokenize(text) for doc_id, text in documents.items()}
    if not query_terms or not tokenized:
        return {doc_id: 0.0 for doc_id in documents}

    doc_lengths = {doc_id: len(tokens) for doc_id, tokens in tokenized.items()}
    average_length = sum(doc_lengths.values()) / max(len(doc_lengths), 1)
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized.values():
        document_frequency.update(set(tokens))

    scores: dict[str, float] = {}
    for doc_id, tokens in tokenized.items():
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if frequency == 0:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log1p(
                (len(tokenized) - df + 0.5) / (df + 0.5)
            )
            length_norm = k1 * (1 - b + b * doc_lengths[doc_id] / max(average_length, 1))
            score += inverse_frequency * frequency * (k1 + 1) / (frequency + length_norm)
        scores[doc_id] = score
    return scores


def reciprocal_rank_fusion(
    *ranked_lists: Iterable[str],
    rrf_k: int = 60,
) -> list[str]:
    """Fuse ranked document-id lists with reciprocal-rank fusion."""
    scores: defaultdict[str, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, document_id in enumerate(ranked_list, start=1):
            scores[document_id] += 1.0 / (rrf_k + rank)
    return sorted(scores, key=lambda document_id: (-scores[document_id], document_id))


def bm25_relevance(score: float) -> float:
    """Compress an uncalibrated BM25 score to a bounded relevance heuristic."""
    return score / (score + 1.0) if score > 0 else 0.0
