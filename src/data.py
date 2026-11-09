"""Download, load and split the MovieLens 100K dataset."""
from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ML_DIR = DATA_DIR / "ml-100k"

GROUPLENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
# Mirror of the unmodified raw files, used only if GroupLens is unreachable.
MIRROR_URL = "https://huggingface.co/datasets/harisarang/ml-100k/resolve/main/raw/{name}"
RAW_FILES = ["u.data", "u.item", "u.user", "u.genre", "u.occupation", "u.info", "README"]
U_DATA_MD5 = "6e47046882bad158b0efbb84cd5cb987"

N_USERS = 943
N_ITEMS = 1682

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def download_movielens(target: Path = ML_DIR) -> Path:
    """Download ML-100K into ``target`` (no-op if it is already there)."""
    if (target / "u.data").exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(GROUPLENS_URL, timeout=30) as resp:
            archive = zipfile.ZipFile(io.BytesIO(resp.read()))
        for name in RAW_FILES:
            (target / name).write_bytes(archive.read(f"ml-100k/{name}"))
    except Exception as err:  # network policy, outage, ...
        print(f"GroupLens download failed ({err}); using mirror.")
        for name in RAW_FILES:
            with urllib.request.urlopen(MIRROR_URL.format(name=name), timeout=30) as resp:
                (target / name).write_bytes(resp.read())
    if _md5(target / "u.data") != U_DATA_MD5:
        raise RuntimeError("u.data checksum mismatch: the download is not the original ML-100K")
    return target


def load_ratings(path: Path = ML_DIR) -> pd.DataFrame:
    """Ratings with 0-based ``user_idx`` / ``item_idx`` columns added."""
    download_movielens(path)
    df = pd.read_csv(
        path / "u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
    )
    df["user_idx"] = df["user_id"] - 1
    df["item_idx"] = df["item_id"] - 1
    return df


def load_movies(path: Path = ML_DIR) -> pd.DataFrame:
    download_movielens(path)
    cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES
    df = pd.read_csv(path / "u.item", sep="|", names=cols, encoding="latin-1")
    df["item_idx"] = df["item_id"] - 1
    df["genres"] = df[GENRES].apply(lambda r: "|".join(g for g in GENRES if r[g] == 1), axis=1)
    return df[["item_id", "item_idx", "title", "release_date", "genres"] + GENRES]


def load_users(path: Path = ML_DIR) -> pd.DataFrame:
    download_movielens(path)
    df = pd.read_csv(
        path / "u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip"]
    )
    df["user_idx"] = df["user_id"] - 1
    return df


def temporal_user_split(ratings: pd.DataFrame, test_frac: float = 0.2):
    """Per-user temporal hold-out.

    For every user the most recent ``test_frac`` of their ratings (by timestamp,
    ties broken by item id for determinism) go to the test set, the rest to train.
    Every user keeps at least one rating in train and one in test.
    """
    df = ratings.sort_values(["user_idx", "timestamp", "item_idx"], kind="mergesort")
    pos = df.groupby("user_idx").cumcount()
    n = df.groupby("user_idx")["item_idx"].transform("size")
    n_test = np.clip(np.round(n * test_frac).astype(int), 1, n - 1)
    is_test = pos >= (n - n_test)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def global_temporal_split(ratings: pd.DataFrame, test_frac: float = 0.2):
    """Single global time cut-off: everything after the quantile goes to test.

    Stricter than the per-user split (no rating from the "future" is ever used
    for training) but users who first appear after the cut-off are cold-start.
    """
    cutoff = ratings["timestamp"].quantile(1 - test_frac)
    train = ratings[ratings["timestamp"] <= cutoff].reset_index(drop=True)
    test = ratings[ratings["timestamp"] > cutoff].reset_index(drop=True)
    return train, test


if __name__ == "__main__":
    print("Data in", download_movielens())
