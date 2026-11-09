"""The three recommenders: popularity, item-based CF and ALS.

Every model follows the same small interface:

* ``fit(train)`` - ``train`` is a ratings DataFrame with ``user_idx``, ``item_idx``, ``rating``;
* ``score(users)`` - dense ``(len(users), n_items)`` array, higher = recommend first;
* ``predict(users, items)`` - predicted star ratings (models that cannot predict
  ratings raise ``NotImplementedError``).

Items a user already rated are filtered out by the evaluator, not by the models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from .data import N_ITEMS, N_USERS


def user_item_matrix(train: pd.DataFrame, values: str | None = "rating",
                     n_users: int = N_USERS, n_items: int = N_ITEMS) -> sp.csr_matrix:
    """Sparse user x item matrix; ``values=None`` gives a binary matrix."""
    data = np.ones(len(train)) if values is None else train[values].to_numpy(float)
    return sp.csr_matrix(
        (data, (train["user_idx"].to_numpy(), train["item_idx"].to_numpy())),
        shape=(n_users, n_items),
    )


# --------------------------------------------------------------------------- #
# 1. Popularity
# --------------------------------------------------------------------------- #
class PopularityRecommender:
    """Non-personalised baseline.

    ``method``:
      * ``"count"``    - number of ratings (pure popularity);
      * ``"mean"``     - raw average rating (dominated by items with 1-2 ratings);
      * ``"bayesian"`` - damped mean ``(sum + m * mu) / (n + m)``, i.e. the average
        shrunk towards the global mean ``mu`` with strength ``m``. It combines the
        average rating with the number of ratings.

    Rating prediction always uses the damped mean.
    """

    def __init__(self, method: str = "bayesian", m: float = 10.0):
        self.method, self.m = method, m

    def fit(self, train: pd.DataFrame):
        self.global_mean_ = train["rating"].mean()
        stats = train.groupby("item_idx")["rating"].agg(["count", "sum"])
        count = np.zeros(N_ITEMS)
        total = np.zeros(N_ITEMS)
        count[stats.index] = stats["count"]
        total[stats.index] = stats["sum"]
        self.count_ = count
        self.mean_ = np.divide(total, count, out=np.zeros(N_ITEMS), where=count > 0)
        denom = count + self.m
        self.damped_ = np.where(denom > 0, (total + self.m * self.global_mean_) / np.maximum(denom, 1e-12),
                                self.global_mean_)  # unrated item -> global mean
        self.item_scores_ = {
            "count": count,
            "mean": self.mean_,
            "bayesian": self.damped_,
        }[self.method]
        return self

    def score(self, users) -> np.ndarray:
        return np.tile(self.item_scores_, (len(users), 1))

    def predict(self, users, items) -> np.ndarray:
        return self.damped_[np.asarray(items)]


# --------------------------------------------------------------------------- #
# 2. Item-based collaborative filtering
# --------------------------------------------------------------------------- #
class ItemKNNRecommender:
    """Item-based CF with cosine similarity between item rating vectors.

    * Similarity: cosine between columns of the user x item matrix. With
      ``centered=True`` each user's mean is subtracted first ("adjusted cosine").
    * Only the ``k`` most similar neighbours of each item are kept.
    * Ranking score: ``score(u, i) = sum_{j in I_u} sim(i, j) * w_uj`` where
      ``w_uj`` is 1 (``weighting="binary"``) or the rating (``"rating"``).
    * Rating prediction:
      ``r(u, i) = mean_u + sum_j sim(i, j) (r_uj - mean_u) / sum_j |sim(i, j)|``
      over the ``k`` items most similar to ``i`` that ``u`` has rated.
    """

    def __init__(self, k: int = 50, centered: bool = False, weighting: str = "binary"):
        self.k, self.centered, self.weighting = k, centered, weighting

    def fit(self, train: pd.DataFrame):
        R = user_item_matrix(train)
        self.R_ = R
        self.global_mean_ = train["rating"].mean()
        counts = np.diff(R.indptr)
        sums = np.asarray(R.sum(axis=1)).ravel()
        self.user_mean_ = np.where(counts > 0, sums / np.maximum(counts, 1), self.global_mean_)

        X = R.copy().astype(float)
        if self.centered:
            X.data -= np.repeat(self.user_mean_, counts)
        sim = cosine_similarity(X.T, dense_output=True)  # (n_items, n_items)
        np.fill_diagonal(sim, 0.0)
        self.sim_full_ = sim
        self.sim_ = self._top_k(sim, self.k)
        return self

    @staticmethod
    def _top_k(sim: np.ndarray, k: int) -> np.ndarray:
        """Keep the k largest similarities in every row (neighbours of item i)."""
        if k >= sim.shape[1]:
            return sim.copy()
        out = np.zeros_like(sim)
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        rows = np.arange(sim.shape[0])[:, None]
        out[rows, idx] = sim[rows, idx]
        return out

    def similar_items(self, item: int, n: int = 10):
        order = np.argsort(-self.sim_full_[item])[:n]
        return order, self.sim_full_[item, order]

    def score(self, users) -> np.ndarray:
        R = self.R_[np.asarray(users)]
        if self.weighting == "binary":
            R = R.copy()
            R.data[:] = 1.0
        # sim_[i, j]: j is a neighbour of i -> score_i = sum_j sim_[i, j] * R[u, j]
        return np.asarray(R @ self.sim_.T)

    def predict(self, users, items) -> np.ndarray:
        users, items = np.asarray(users), np.asarray(items)
        preds = np.empty(len(users))
        for n, (u, i) in enumerate(zip(users, items)):
            row = self.R_.getrow(u)
            rated, ratings = row.indices, row.data
            mu = self.user_mean_[u]
            if len(rated) == 0:
                preds[n] = self.global_mean_
                continue
            s = self.sim_full_[i, rated]
            top = np.argsort(-s)[: self.k]
            s, r = s[top], ratings[top]
            mask = s > 0
            denom = np.abs(s[mask]).sum()
            preds[n] = mu if denom == 0 else mu + np.dot(s[mask], r[mask] - mu) / denom
        return np.clip(preds, 1.0, 5.0)


# --------------------------------------------------------------------------- #
# 3a. ALS on implicit feedback (``implicit`` library) - used for ranking
# --------------------------------------------------------------------------- #
class ImplicitALSRecommender:
    """Hu, Koren & Volinsky (2008) weighted ALS via the ``implicit`` library.

    Every observed rating is treated as a positive interaction with confidence
    ``c_ui = 1 + alpha * r_ui`` (``use_ratings=True``) or ``1 + alpha``.
    Unobserved cells are negatives with confidence 1. The model minimises
    ``sum_ui c_ui (p_ui - x_u . y_i)^2 + reg (|x_u|^2 + |y_i|^2)`` where
    ``p_ui`` is 1 for observed pairs. Scores are ``x_u . y_i`` - they are *not*
    star ratings, so this model has no ``predict``.
    """

    def __init__(self, factors: int = 64, regularization: float = 0.05, alpha: float = 1.0,
                 iterations: int = 15, use_ratings: bool = True, random_state: int = 42):
        self.factors, self.regularization, self.alpha = factors, regularization, alpha
        self.iterations, self.use_ratings, self.random_state = iterations, use_ratings, random_state

    def fit(self, train: pd.DataFrame):
        from implicit.als import AlternatingLeastSquares
        from threadpoolctl import threadpool_limits

        R = user_item_matrix(train, values="rating" if self.use_ratings else None)
        with threadpool_limits(1, "blas"):  # recommended by implicit for CPU ALS
            self.model_ = AlternatingLeastSquares(
                factors=self.factors, regularization=self.regularization, alpha=self.alpha,
                iterations=self.iterations, random_state=self.random_state,
                use_gpu=False, calculate_training_loss=False,
            )
            self.model_.fit(R.astype(np.float32), show_progress=False)
        self.user_factors_ = np.asarray(self.model_.user_factors)
        self.item_factors_ = np.asarray(self.model_.item_factors)
        return self

    def score(self, users) -> np.ndarray:
        return self.user_factors_[np.asarray(users)] @ self.item_factors_.T

    def predict(self, users, items):
        raise NotImplementedError("Implicit ALS outputs preference scores, not ratings.")


# --------------------------------------------------------------------------- #
# 3b. ALS on explicit ratings (NumPy) - used for RMSE
# --------------------------------------------------------------------------- #
class ExplicitALSRecommender:
    """Biased matrix factorisation fitted with ALS on observed ratings only.

    ``r_ui ~ mu + b_u + b_i + p_u . q_i``. Biases are regularised averages of
    the residuals; the factors are fitted by alternating ridge regressions with
    ALS-WR regularisation ``reg * n_u`` (Zhou et al., 2008). Only observed cells
    enter the loss, which is what makes it a rating predictor.
    """

    def __init__(self, factors: int = 20, regularization: float = 0.1, iterations: int = 15,
                 bias_reg: float = 10.0, random_state: int = 42):
        self.factors, self.regularization, self.iterations = factors, regularization, iterations
        self.bias_reg, self.random_state = bias_reg, random_state

    def fit(self, train: pd.DataFrame):
        u = train["user_idx"].to_numpy()
        i = train["item_idx"].to_numpy()
        r = train["rating"].to_numpy(float)
        self.mu_ = r.mean()

        # Regularised biases (item first, then user), as in Koren's baseline predictor.
        self.bi_ = np.zeros(N_ITEMS)
        self.bu_ = np.zeros(N_USERS)
        for _ in range(3):
            res = r - self.mu_ - self.bu_[u]
            self.bi_ = np.bincount(i, res, N_ITEMS) / (np.bincount(i, minlength=N_ITEMS) + self.bias_reg)
            res = r - self.mu_ - self.bi_[i]
            self.bu_ = np.bincount(u, res, N_USERS) / (np.bincount(u, minlength=N_USERS) + self.bias_reg)
        resid = r - self.mu_ - self.bu_[u] - self.bi_[i]

        Ru = sp.csr_matrix((resid, (u, i)), shape=(N_USERS, N_ITEMS))
        Ri = Ru.T.tocsr()
        rng = np.random.default_rng(self.random_state)
        f = self.factors
        P = rng.normal(0, 0.1, (N_USERS, f))
        Q = rng.normal(0, 0.1, (N_ITEMS, f))
        eye = np.eye(f)
        for _ in range(self.iterations):
            P = self._solve(Ru, Q, eye)
            Q = self._solve(Ri, P, eye)
        self.P_, self.Q_ = P, Q
        return self

    def _solve(self, M: sp.csr_matrix, Y: np.ndarray, eye: np.ndarray) -> np.ndarray:
        X = np.zeros((M.shape[0], Y.shape[1]))
        for row in range(M.shape[0]):
            start, end = M.indptr[row], M.indptr[row + 1]
            if start == end:
                continue
            idx, vals = M.indices[start:end], M.data[start:end]
            Yi = Y[idx]
            A = Yi.T @ Yi + self.regularization * len(idx) * eye
            X[row] = np.linalg.solve(A, Yi.T @ vals)
        return X

    def score(self, users) -> np.ndarray:
        users = np.asarray(users)
        return (self.mu_ + self.bu_[users, None] + self.bi_[None, :]
                + self.P_[users] @ self.Q_.T)

    def predict(self, users, items) -> np.ndarray:
        users, items = np.asarray(users), np.asarray(items)
        pred = (self.mu_ + self.bu_[users] + self.bi_[items]
                + np.sum(self.P_[users] * self.Q_[items], axis=1))
        return np.clip(pred, 1.0, 5.0)
