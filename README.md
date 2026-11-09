# Movie Recommendation System

Personalised movie recommendations from user ratings on **MovieLens 100K**, comparing three classic
approaches — a popularity baseline, item-based collaborative filtering and ALS matrix factorisation —
under one leakage-free evaluation protocol, with both rating-prediction (RMSE) and ranking
(Precision@K, Recall@K, NDCG@K) metrics.

All numbers below come from the executed notebooks in `notebooks/` (CSV copies in `results/`).

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) (GroupLens, 1998):

| | |
|---|---|
| ratings | 100,000 (integer 1–5 stars) |
| users / movies | 943 / 1,682 |
| density | 6.30% of the user × movie matrix is filled |
| period | 1997-09-20 → 1998-04-22 |
| ratings per user | min 20, median 65, max 737 |
| ratings per movie | min 1, median 27, max 583; 333 movies have < 5 ratings |
| long tail | the top 10% of movies receive 42.7% of all ratings |

The data is not committed (licence). `src/data.py` downloads it to `data/ml-100k/` on first use; if
`files.grouplens.org` is unreachable it uses a Hugging Face mirror of the raw files and verifies the MD5
of `u.data`.

## Algorithms and mathematical intuition

**1. Popularity** (`PopularityRecommender`). The same list for everyone, minus movies the user has
already rated. Three scoring rules:
- *count* `n_i` — how many people rated the movie;
- *mean* `r̄_i` — average rating, dominated by movies rated 5 by one or two people;
- *damped (Bayesian) mean* `(Σ_u r_ui + m·μ) / (n_i + m)` — the average shrunk towards the global mean
  `μ` as if each movie had `m` extra ratings equal to `μ`. It combines the average rating with the number
  of ratings. It is also the rating predictor used for RMSE.

**2. Item-based collaborative filtering** (`ItemKNNRecommender`). Build the user × movie matrix `R`,
compute cosine similarity between movie columns
`sim(i,j) = r_·i · r_·j / (‖r_·i‖ ‖r_·j‖)` (scikit-learn `cosine_similarity`; optionally after
subtracting each user's mean, "adjusted cosine") and keep the `k` nearest neighbours of each movie.
- Ranking: `s(u,i) = Σ_{j ∈ I_u} sim(i,j) · w_uj` — a movie scores high if it is similar to many
  movies in the user's history (`w_uj` = 1 or the rating).
- Rating prediction: `r̂_ui = r̄_u + Σ_j sim(i,j)(r_uj − r̄_u) / Σ_j |sim(i,j)|` over the `k` most
  similar movies the user rated.

**3. ALS matrix factorisation.** Approximate `R ≈ P Qᵀ` with `f`-dimensional user and movie vectors.
With one side fixed the loss is a ridge regression in the other, so ALS alternates closed-form solves.
Two variants are needed because the two tasks need different objectives:

| | implicit ALS (`implicit` library) | explicit ALS (NumPy, `src/models.py`) |
|---|---|---|
| training cells | **all** cells: rated = positive with confidence `c_ui = 1 + α·r_ui`, unrated = 0 with confidence 1 | **observed** ratings only |
| loss | `Σ_ui c_ui (p_ui − x_uᵀy_i)² + λ(‖X‖² + ‖Y‖²)` (Hu, Koren & Volinsky 2008) | `Σ_(u,i)∈Ω (r_ui − μ − b_u − b_i − p_uᵀq_i)² + λ(Σ_u n_u‖p_u‖² + Σ_i n_i‖q_i‖²)` (ALS-WR) |
| output | preference score, not a rating | star rating |
| used for | ranking metrics | RMSE |

`implicit` fits the ranking task well but cannot predict star ratings, so a small explicit ALS with
user/item biases was added for RMSE.

## Rating prediction vs. ranking

- **RMSE** measures how close the predicted stars are to the true stars **on movies the user actually
  rated**. It says nothing about the thousands of movies the user never watched.
- **Precision@K / Recall@K / NDCG@K** check whether the movies the user goes on to watch *and like*
  appear at the top of a list built from the **whole catalogue**. This is what a recommender is used
  for. It requires modelling *which* movies a user will choose, not just how many stars they would give.

The experiments show the gap clearly. Explicit ALS has the best RMSE (0.977) but the worst ranking
(NDCG@10 0.058, below the popularity baseline). Trained only on observed ratings, it predicts high
ratings for obscure movies that the user has no reason to pick. Implicit ALS is trained on what users
choose to rate. It has no RMSE at all, but it ranks best.

## Evaluation protocol (leakage control)

- **Per-user temporal split.** For every user, the most recent 20% of their ratings form the test set
  (80,000 train / 20,000 test). Every user's training ratings are no later than their test ratings
  (checked in `01_eda.ipynb`), and no (user, movie) pair is in both sets.
- **Separate validation split.** All hyper-parameters are chosen on a second split of the same kind
  *inside* the training set (64,005 / 15,995). The models are then refit on the full training set and
  evaluated on the test set once (`05_evaluation.ipynb`).
- Popularity statistics, similarities and factors are computed from training data only.
- **Ranking:** candidates are all movies the user has not rated in train. Test movies rated ≥ 4 are
  *relevant*. Metrics are averaged over the 906 users with at least one relevant test movie. NDCG uses
  binary relevance.
- **RMSE:** over all 20,000 test ratings (88 are on movies unseen in train; models fall back to biases
  or the global mean).
- **Robustness check:** a strict global time split, so that no future ratings from any user are used
  (section 5 of notebook 05).

## Architecture

```
.
├── data/                  # downloaded ML-100K (git-ignored) + README
├── notebooks/
│   ├── 01_eda.ipynb           # statistics, long tail, split + leakage checks
│   ├── 02_popularity.ipynb    # popularity variants, tuning, test metrics
│   ├── 03_itemknn.ipynb       # similar movies, tuning k / centring / weighting
│   ├── 04_als.ipynb           # implicit ALS + explicit ALS, tuning, factor-space neighbours
│   └── 05_evaluation.ipynb    # comparison, K, activity, cold start, global split, examples
├── src/
│   ├── data.py            # download, loading, per-user and global temporal splits
│   ├── models.py          # 4 model classes with fit / score / predict
│   ├── metrics.py         # RMSE, Precision@K, Recall@K, NDCG@K
│   └── evaluation.py      # top-K recommendation, evaluation loops, grid search
├── tests/test_metrics.py
├── results/               # CSV/JSON written by the notebooks
└── requirements.txt
```

Every model implements `fit(train_df)`, `score(users) → (n_users, n_items)` and
`predict(users, items) → ratings`. The evaluator removes already-rated movies and takes the top-K.
Notebooks 02–04 write the selected hyper-parameters to `results/best_params.json`, and notebook 05
reads them.

## Hyper-parameters (chosen on validation)

| model | grid | selected |
|---|---|---|
| Popularity (ranking) | rule ∈ {count, mean, bayesian}, m ∈ {5, 20, 50, 100} | count |
| Popularity (RMSE) | m ∈ {0, 1, 5, 10, 20, 50, 100} | damped mean, m = 1 |
| ItemKNN (ranking) | k ∈ {10, 20, 50, 100, 200, all}, centred ∈ {no, yes}, weight ∈ {binary, rating} | k = all, raw cosine, rating weights |
| ItemKNN (RMSE) | k ∈ {10, 20, 50, 100, 200}, centred ∈ {no, yes} | k = 20, adjusted cosine |
| implicit ALS | factors ∈ {8, …, 128}, λ ∈ {0.1, 1, 5, 20}, α ∈ {0.1, 0.25, 0.5, 2}, 15 iterations | 64 factors, λ = 5, α = 0.1 |
| explicit ALS | factors ∈ {10, 20, 50}, λ ∈ {0.05, …, 0.3}, 15 iterations | 50 factors, λ = 0.15 |

For explicit ALS, a preliminary run with 100 and 200 factors improved validation RMSE by less than
0.001, so the grid was capped at 50 for speed.

## Results

### Main comparison (test set, K = 10)

| model | RMSE ↓ | Precision@10 | Recall@10 | NDCG@10 | Coverage@50 |
|---|---|---|---|---|---|
| Popularity | 1.0688 | 0.0726 | 0.0722 | 0.0909 | 0.124 |
| ItemKNN | 1.0215 | 0.1221 | 0.1402 | 0.1651 | 0.257 |
| ALS (implicit) | — | **0.1295** | **0.1532** | **0.1744** | **0.384** |
| ALS (explicit) | **0.9773** | 0.0498 | 0.0448 | 0.0576 | 0.349 |

Coverage@50 is the share of the 1,682 movies that appear in at least one user's top-50 list.
Popularity's RMSE uses the damped mean (m = 1), its ranking uses counts.

- Both personalised ranking models clearly beat popularity: NDCG@10 is about 1.8–1.9× higher.
- Implicit ALS is best on every ranking metric and also covers the catalogue most widely. ItemKNN comes
  close.
- Explicit ALS has the lowest RMSE but the worst ranking (see "Rating prediction vs. ranking").
- Absolute RMSE values are higher than the ~0.90–0.93 often reported for ML-100K with *random* splits:
  predicting each user's *future* ratings is harder.

### Effect of K

| | K = 5 | K = 10 | K = 20 | K = 50 |
|---|---|---|---|---|
| Precision — Popularity / ItemKNN / iALS | 0.075 / 0.143 / 0.154 | 0.073 / 0.122 / 0.130 | 0.062 / 0.097 / 0.106 | 0.051 / 0.068 / 0.074 |
| Recall — Popularity / ItemKNN / iALS | 0.039 / 0.084 / 0.090 | 0.072 / 0.140 / 0.153 | 0.119 / 0.218 / 0.246 | 0.241 / 0.370 / 0.415 |
| NDCG — Popularity / ItemKNN / iALS | 0.085 / 0.163 / 0.172 | 0.091 / 0.165 / 0.174 | 0.103 / 0.182 / 0.197 | 0.148 / 0.234 / 0.255 |

As K grows, precision falls and recall rises (a median user has 7 relevant test movies). The model
ranking stays the same at every K, and the lead of implicit ALS over ItemKNN widens in recall at large K.

### Effect of the number of interactions (NDCG@10 by quartile of training ratings per user)

| training ratings | users | Popularity | ItemKNN | ALS (implicit) | ALS (explicit) |
|---|---|---|---|---|---|
| 16–26 | 219 | 0.064 | 0.135 | 0.146 | 0.028 |
| 27–52 | 225 | 0.060 | 0.151 | 0.156 | 0.042 |
| 53–118 | 230 | 0.114 | 0.171 | 0.170 | 0.073 |
| 119–590 | 232 | 0.123 | 0.202 | 0.223 | 0.086 |

RMSE by the same quartiles: explicit ALS 1.058 → 0.989 → 0.989 → 0.961, ItemKNN 1.101 → 1.037 → 1.034 →
1.005, popularity 1.129 → 1.064 → 1.057 → 1.068. Every personalised model improves with more history.
Popularity improves too for very active users, who also watch the popular titles.

### Cold-start users

Simulation: 200 random users keep only their first *n* training ratings, all models are retrained, and
the 200 users are evaluated on their unchanged test ratings. At n = 0 every model falls back to the
popularity list.

| n ratings kept | Popularity | ItemKNN | ALS (implicit) | ALS (explicit) |
|---|---|---|---|---|
| NDCG@10, n = 0 | 0.048 | 0.048 | 0.048 | 0.048 |
| 1 | 0.051 | 0.042 | 0.046 | 0.040 |
| 2 | 0.052 | 0.051 | 0.053 | 0.041 |
| 5 | 0.055 | 0.052 | 0.053 | 0.041 |
| 10 | 0.061 | 0.065 | 0.065 | 0.042 |
| 15 | 0.065 | 0.079 | 0.080 | 0.039 |
| RMSE, n = 1 | 1.040 | 1.573 | — | 1.069 |
| RMSE, n = 15 | 1.040 | 1.256 | — | 1.066 |

With 1–5 ratings, popularity is as good as or better than the personalised models. They overtake it
only at about 10 ratings. ItemKNN rating prediction is very poor for cold users because it centres on
the user's own mean, which is unreliable with 1–2 ratings. Practical conclusion: show popularity to new
users and switch to CF/ALS after roughly 10 interactions.

### Robustness check: strict global time split

Train on ratings before the 80th percentile of timestamps, test on the rest. Only 301 users rate after
the cut-off, and 192 of them have no training history, so they get the popularity list.

| model | RMSE | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| Popularity | 1.0306 | 0.2297 | 0.0853 | 0.2541 |
| ItemKNN | 1.1128 | 0.2352 | 0.0786 | 0.2570 |
| ALS (implicit) | — | 0.2345 | 0.0885 | 0.2582 |
| ALS (explicit) | 1.0228 | 0.2217 | 0.0876 | 0.2450 |

Under this split all models are almost tied, because most test users are cold-start. Precision is
higher than in the main table only because these users have many more test ratings. The per-user split
is therefore the more informative comparison, but its numbers should be read as optimistic about
deployment.

### Example recommendations

Notebook 05 (section 6) prints the top-10 lists of the three ranking models for a light user
(16 training ratings), a median user (52) and the heaviest user (590), with ✓ on hits. For example, for
user 54 (history: *Trainspotting*, *The Rock*, *The Long Kiss Goodnight*, …) the popularity list is
generic (*The English Patient*, *Raiders of the Lost Ark*, …), while ItemKNN and implicit ALS
recommend 1996–97 films close to the history (*Star Trek: First Contact*, *Liar Liar*, *Phenomenon*,
*Face/Off* ✓). Notebooks 03 and 04 show similar-movie lists, for example *Star Wars* → *Return of the
Jedi*, *The Empire Strikes Back*.

## How to run

```bash
pip install -r requirements.txt
```

```bash
python -m src.data
```

```bash
cd notebooks && for n in 01_eda 02_popularity 03_itemknn 04_als 05_evaluation; do jupyter nbconvert --to notebook --execute --inplace $n.ipynb; done
```

```bash
python -m pytest -q tests
```

Run the notebooks in order (05 needs `results/best_params.json` from 02–04). A full run takes about
3 minutes on a laptop CPU. Tested with Python 3.13, NumPy 2.5, pandas 3.0, scikit-learn 1.9,
implicit 0.7.3.

## Limitations

- **Small, old dataset.** 943 users, 1997–98. The conclusions need not transfer to large or modern
  catalogues.
- **Split optimism.** The per-user split still lets training contain other users' ratings made after a
  given user's test period. The strict global split removes this but leaves too few warm users for a
  meaningful comparison.
- **Ranking metrics only reward observed items.** A good recommendation the user never rated counts as
  a miss, which biases the evaluation towards popular movies.
- **Relevance threshold.** "Relevant = rating ≥ 4" is a convention, and the results depend on it.
- **Single run, no confidence intervals.** One split and one seed. Differences of about 0.01 NDCG (for
  example, ItemKNN vs. implicit ALS) were not tested for significance.
- **Limited tuning.** Small grids, a fixed 15 ALS iterations. Explicit ALS is a simple pure-NumPy
  implementation, not an optimised library.
- **Simple cold-start handling.** Popularity fallback only. No content features (genres, demographics)
  are used, although the dataset provides them.
