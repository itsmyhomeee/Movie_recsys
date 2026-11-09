from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def precision_at_k(recommended: Sequence[int], relevant: set, k: int) -> float:
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: Sequence[int], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: Sequence[int], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    gains = np.array([1.0 if item in relevant else 0.0 for item in recommended[:k]])
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal_hits = min(len(relevant), k)
    idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_hits + 2))))
    return dcg / idcg


def ranking_metrics(recommended: Sequence[int], relevant: set, ks: Iterable[int]) -> dict:
    out = {}
    for k in ks:
        out[f"precision@{k}"] = precision_at_k(recommended, relevant, k)
        out[f"recall@{k}"] = recall_at_k(recommended, relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(recommended, relevant, k)
    return out
