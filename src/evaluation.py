from __future__ import annotations

from itertools import product
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .data import N_ITEMS
from .metrics import ranking_metrics, rmse
from .models import user_item_matrix

RELEVANCE_THRESHOLD = 4


def recommend_top_k(model, train: pd.DataFrame, users, k: int, fallback=None,
                    batch_size: int = 256) -> np.ndarray:
    users = np.asarray(users)
    seen = user_item_matrix(train, values=None)
    n_seen = np.diff(seen.indptr)
    out = np.empty((len(users), k), dtype=int)
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        scores = np.asarray(model.score(batch), dtype=float)
        if fallback is not None:
            cold = n_seen[batch] == 0
            if cold.any():
                scores[cold] = fallback.score(batch[cold])
        scores[seen[batch].toarray() > 0] = -np.inf
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1, kind="stable")
        out[start:start + len(batch)] = np.take_along_axis(top, order, axis=1)
    return out


def relevant_items(test: pd.DataFrame, threshold: float = RELEVANCE_THRESHOLD) -> dict:
    rel = test[test["rating"] >= threshold]
    return rel.groupby("user_idx")["item_idx"].apply(set).to_dict()


def evaluate_ranking(model, train: pd.DataFrame, test: pd.DataFrame,
                     ks: Iterable[int] = (10,), fallback=None, users=None,
                     threshold: float = RELEVANCE_THRESHOLD, per_user: bool = False):
    ks = sorted(ks)
    rel = relevant_items(test, threshold)
    eval_users = np.array(sorted(u for u in rel if users is None or u in set(users)))
    recs = recommend_top_k(model, train, eval_users, max(ks), fallback=fallback)
    rows = []
    for u, rec in zip(eval_users, recs):
        row = ranking_metrics(rec.tolist(), rel[u], ks)
        row["user_idx"] = u
        rows.append(row)
    per = pd.DataFrame(rows).set_index("user_idx")
    summary = per.mean().to_dict()
    summary[f"coverage@{max(ks)}"] = len(np.unique(recs)) / N_ITEMS
    summary["n_users"] = len(eval_users)
    return (summary, per) if per_user else summary


def evaluate_rmse(model, test: pd.DataFrame) -> float:
    preds = model.predict(test["user_idx"].to_numpy(), test["item_idx"].to_numpy())
    return rmse(test["rating"].to_numpy(), preds)


def grid_search(make_model: Callable[..., object], grid: dict, train: pd.DataFrame,
                valid: pd.DataFrame, objective: str = "ndcg@10", k: int = 10,
                verbose: bool = True) -> pd.DataFrame:
    keys = list(grid)
    rows = []
    for values in product(*(grid[key] for key in keys)):
        params = dict(zip(keys, values))
        model = make_model(**params).fit(train)
        if objective == "rmse":
            row = {**params, "rmse": evaluate_rmse(model, valid)}
        else:
            metrics = evaluate_ranking(model, train, valid, ks=(k,))
            row = {**params, **{m: v for m, v in metrics.items() if m != "n_users"}}
        if verbose:
            print(row)
        rows.append(row)
    res = pd.DataFrame(rows)
    return res.sort_values(objective, ascending=(objective == "rmse")).reset_index(drop=True)
