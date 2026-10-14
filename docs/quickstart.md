# Quickstart

## Rankings

```python
import numpy as np
from uws.ranking import RankingLabelModel, RankingUtils, generate_true_rankings, sample_mallows_LFs

rng = np.random.default_rng(0)
r_utils = RankingUtils(d=5)
Y = generate_true_rankings(5, 1000, rng, r_utils)
L = sample_mallows_LFs(Y, 6, [0.2, 0.3, 2.0, 2.5, 0.4, 3.0], rng=rng)

lm = RankingLabelModel(r_utils, train_method="median_triplet_opt", inference_rule="weighted_kemeny").fit(L)
Y_hat = lm.predict(L)
print(r_utils.mean_kt_distance(Y_hat, Y))
```

`L` is a list (one entry per example) of lists (one per LF) of {class}`uws.ranking.Ranking` objects, or an integer
array of permutations with shape `(n, m, d)`. Partial rankings are built with `Ranking.mask_items`.

## Regression

```python
from uws.regression import ContinuousLabelModel, generate_lfs, generate_synthetic_xy

X, Y, var_Y = generate_synthetic_xy(5000, 10)
L, _ = generate_lfs(8, 5000, Y, var_Y)
lm = ContinuousLabelModel().fit(L, var_Y)
y_hat, posterior_var = lm.predict(L, return_var=True)
```

## Any finite metric space

```python
from uws.metric import MetricLabelModel, sample_metric_lfs

D = ...  # (k, k) distance matrix, e.g. shortest paths of a graph
lm = MetricLabelModel(D).fit(L)   # L: (n, m) ints in 0..k-1, -1 = abstain
y_hat = lm.predict(L)
```

## Command line

```bash
uws-experiment configs/boardgames_ranking_experiment.yaml --project-root . --set weak_sup_conf.p=0.4
```
