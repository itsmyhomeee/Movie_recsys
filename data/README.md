# Data

The MovieLens 100K files are not committed (the GroupLens licence does not allow
redistribution). They are downloaded automatically into `data/ml-100k/` the first
time `src.data.load_ratings()` is called, or explicitly with:

```bash
python -m src.data
```

Source: https://grouplens.org/datasets/movielens/100k/ (F. Maxwell Harper and
Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM TiiS 5, 4).
If `files.grouplens.org` is unreachable, the script falls back to a Hugging Face
mirror of the same raw files and checks the MD5 of `u.data`.
