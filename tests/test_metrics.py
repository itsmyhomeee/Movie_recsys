import math

from src.metrics import ndcg_at_k, precision_at_k, recall_at_k, rmse


def test_rmse():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0
    assert math.isclose(rmse([0, 0], [3, 4]), math.sqrt(12.5))


def test_precision_recall():
    rec, rel = [1, 2, 3, 4], {2, 4, 9}
    assert precision_at_k(rec, rel, 4) == 0.5
    assert math.isclose(recall_at_k(rec, rel, 4), 2 / 3)
    assert precision_at_k(rec, rel, 1) == 0


def test_ndcg():
    assert ndcg_at_k([1, 2, 3], {1}, 3) == 1.0
    assert math.isclose(ndcg_at_k([5, 1], {1}, 2), 1 / math.log2(3))
    # ideal DCG is capped at min(|rel|, k)
    assert ndcg_at_k([1, 2], {1, 2, 3}, 2) == 1.0
    assert ndcg_at_k([7, 8], set(), 2) == 0.0
